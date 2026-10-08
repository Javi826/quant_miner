# quant_miner/darwinex/BOT_sweep/sweep_research_fx.py 
import os
import sys
import json
import time
import logging
import itertools
from contextlib import contextmanager
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",".."))       # quant_miner
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "core"))
from setup import config_research as cr
sys.path.append(os.path.join(_ROOT, cr.BROKER))

from setup.config_core import settings
from research import artifacts as ra
from research.stages import screen, grids, combos
from research.stages.screen import ScreenStageConfig
from research.stages.combos import CombosStageConfig

logger = logging.getLogger("BOT_sweep.research")


LOG_LEVELS = {
    "screen": logging.INFO,
    "grids":  logging.INFO,
    "combos": logging.INFO,
}
# =============================================================================
# CONFIG ── every list is swept: one json per point of the cartesian product
# =============================================================================
SWEEP_SCREEN = {                                # ScreenStageConfig fields
    "luck_max": [0.1,0.2,0.3,0.4,0.5,0.6],                          # less luck_max: higher GROUP_N
    "top_i":    [4,5,6,7,8],
}
COMBOS = CombosStageConfig(                     # fixed for the whole sweep, as research_fx
    plus_minus         = 0.2,
    combo_sizes        = [1, 2],
    n_samples_per_size = {1: None, 2: 190},     # None = exhaustive
)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "sweep", "configs")    # read by sweep_backtest_fx
# a json already in CONFIGS_DIR is not run again: empty it after changing COMBOS or setup/config_research

STAGE_BY_NAME = {"screen": screen, "grids": grids, "combos": combos}
SEP   = "═" * 115
LBL_W = 12

# =============================================================================
# NO PERSIST ── the stages chain in memory; their checkpoints are not written
# =============================================================================
@contextmanager
def no_persist():
    orig = ra._write_json
    ra._write_json = lambda path, doc: doc      # save_grids_input / save_combos_input return the doc, unwritten
    try:
        yield
    finally:
        ra._write_json = orig


@contextmanager
def only_timeframes(timeframes: list):
    orig = cr.TIMEFRAMES
    cr.TIMEFRAMES = timeframes                  # combos reads cr.TIMEFRAMES: it runs only on these
    try:
        yield
    finally:
        cr.TIMEFRAMES = orig

# =============================================================================
# SWEEP POINTS AND OUTPUT
# =============================================================================
def _points(sweep: dict, cfg_cls) -> list:
    empty = [k for k, v in sweep.items() if not v]
    if empty:
        raise ValueError(f"{cfg_cls.__name__} sweep has empty lists: {empty}")
    keys   = list(sweep)
    points = [dict(zip(keys, values)) for values in itertools.product(*sweep.values())]
    return [(p, cfg_cls(**p)) for p in points]      # built upfront: an unknown field or a bad value fails before any run


def _config_path(params: dict) -> str:
    name = "_".join(f"{k}-{v}" for k, v in params.items())
    return os.path.join(CONFIGS_DIR, f"{name}.json")


def _save_config(params: dict, screen_cfg, result: dict) -> str:
    doc = {
        "created":       datetime.now().isoformat(timespec="seconds"),
        "params":        params,
        "dataset":       cr.DATASET,
        "backtest_mode": str(settings.BACKTEST_MODE),
        "stages":        {"screen": str(screen_cfg), "combos": str(COMBOS)},           # grids has no config
        "SYMBOL_COMBOS_BY_TIMEFRAME":       {tf: r["combos"]     for tf, r in result.items()},
        "PARAM_GRID_BY_TIMEFRAME":          {tf: r["param_grid"] for tf, r in result.items()},
        "SELECTED_INDICATORS_BY_TIMEFRAME": {tf: r["indicators"] for tf, r in result.items()},
    }
    path = _config_path(params)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    tmp  = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)                       # atomic: a half-written json never counts as done
    return path

# =============================================================================
# STAGES
# =============================================================================
def _run_stage(name: str, cfg=None, inputs=None):
    module = STAGE_BY_NAME[name]
    module.LOG_LEVEL = LOG_LEVELS[name]
    t0  = time.time()
    if name == "screen":
        out = module.run(cfg)
    elif name == "grids":
        out = module.run(inputs)                # no config
    else:
        out = module.run(cfg, inputs)
    logger.info(f"🏁 {name.upper()} done in {_elapsed(time.time() - t0)}")
    return out


def _run_combos(grids_out: dict) -> dict:
    winners     = (grids_out or {}).get("by_timeframe", {})
    with_winner = [tf for tf in cr.TIMEFRAMES if tf in winners]
    no_winner   = [tf for tf in cr.TIMEFRAMES if tf not in winners]
    if no_winner:
        logger.info(f"\n⚠  no grids winner for {no_winner}: saved with no combos")
    result = {}
    if with_winner:
        with only_timeframes(with_winner):
            result = _run_stage("combos", COMBOS, grids_out)
    empty = {"combos": [], "param_grid": {}, "indicators": []}      # dropped by sweep_backtest_fx
    return {tf: result.get(tf, empty) for tf in cr.TIMEFRAMES}

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    points  = _points(SWEEP_SCREEN, ScreenStageConfig)
    n_total = len(points)
    n_done  = sum(os.path.exists(_config_path(p)) for p, _ in points)
    log_sweep_config(n_total, n_done)

    with no_persist():
        for params, screen_cfg in points:
            if os.path.exists(_config_path(params)):
                continue
            screen_out = _run_stage("screen", screen_cfg)
            grids_out  = _run_stage("grids", inputs=screen_out)
            result     = _run_combos(grids_out)                                    # only the timeframes with a grids winner
            path       = _save_config(params, screen_cfg, result)
            n_done    += 1
            logger.info(f"\n💾 CONFIG {n_done}/{n_total} ── {os.path.basename(path)}")


def _elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600} h {(s % 3600) // 60} min {s % 60} s"


def log_sweep_config(n_total: int, n_done: int) -> None:
    logger.info(f"\n{SEP}")
    logger.info("  SWEEP RESEARCH START")
    logger.info(f"{SEP}")
    logger.info(f"  {'DATASET':<{LBL_W}}: {cr.DATASET} ── {cr.TIMEFRAMES} ── {len(cr.SYMBOLS)} symbols")
    logger.info(f"  {'SCREEN':<{LBL_W}}: {SWEEP_SCREEN}")
    logger.info(f"  {'GRIDS':<{LBL_W}}: no config (ranked by rules)")
    logger.info(f"  {'COMBOS':<{LBL_W}}: {COMBOS}")
    logger.info(f"  {'CONFIGS':<{LBL_W}}: {n_total} ── done: {n_done} ── pending: {n_total - n_done}")
    logger.info(f"  {'OUTPUT':<{LBL_W}}: {CONFIGS_DIR}")
    logger.info(f"{SEP}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    logger.setLevel(logging.INFO)
    start = time.time()
    try:
        main()
        logger.info(f"\n🏁 TOTAL ── {_elapsed(time.time() - start)}")
    except KeyboardInterrupt:
        logger.info(f"\n⛔  INTERRUPTED BY USER ── {_elapsed(time.time() - start)}")
        sys.exit(0)