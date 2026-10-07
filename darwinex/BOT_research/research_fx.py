# quant_miner/darwinex/BOT_research/research_fx.py (forex)
import os
import sys
import time
import logging

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))       # quant_miner
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "core"))
sys.path.append(os.path.join(_ROOT, "darwinex"))

from setup import config_research as cr
from setup.config_core import settings
from research import artifacts as ra
from research.stages import screen, grids, combos
from research.stages.screen import ScreenStageConfig
from research.stages.combos import CombosStageConfig

logger = logging.getLogger("BOT_research.research")

LOG_LEVELS = {
    "screen": logging.INFO,         # DEBUG: screen ── redundancy tables, TOP_I and the TOP symbols
    "grids":  logging.INFO,         # grids  ── every symbol and terna, as the backtest
    "combos": logging.INFO,         # combos ── every combo, as the backtest
}
# =============================================================================
# CONFIG
# =============================================================================
STAGES = ["screen", "grids", "combos"]     # in order, no gaps: the stage before the first one is read from its checkpoint
SCREEN = ScreenStageConfig(
    luck_max = 0.20,
    top_i    = 8,
)
COMBOS = CombosStageConfig(
    plus_minus         = 0.2,
    combo_sizes        = [1, 2],
    n_samples_per_size = {1: None, 2: 190},     # None = exhaustive
)

# =============================================================================
# STAGES (do not edit)
# =============================================================================
STAGE_BY_NAME = {
    "screen": (screen, SCREEN),
    "grids":  (grids,  None),       # no config
    "combos": (combos, COMBOS),
}
NAMES = list(STAGE_BY_NAME)
if not STAGES or any(s not in NAMES for s in STAGES):
    raise ValueError(f"STAGES must be a non-empty list of {NAMES}: {STAGES}")
_first = NAMES.index(STAGES[0])
if STAGES != NAMES[_first:_first + len(STAGES)]:
    raise ValueError(f"STAGES must be in order and with no gaps, as {NAMES}: {STAGES}")

SEP   = "═" * 115
LBL_W = 12

# =============================================================================
# CHECKPOINTS
# =============================================================================
def _differences(stored: dict, expected: dict) -> list:
    return [f"{k}: checkpoint={stored.get(k)} config={v}" for k, v in expected.items() if stored.get(k) != v]

def check_checkpoint() -> None:
    previous = NAMES[_first - 1] if _first > 0 else None
    if previous == "screen":
        for tf in cr.TIMEFRAMES:
            sel  = ra.load_grids_input(tf, settings.BACKTEST_MODE)["selection"]
            diff = _differences(sel, {"luck_max": SCREEN.luck_max, "top_i": SCREEN.top_i})
            if diff:
                logger.warning(f"⚠  screen checkpoint {tf} was built with another config ── {' | '.join(diff)}")

# =============================================================================
# MAIN
# =============================================================================
def main() -> dict:
    log_run_config()
    check_checkpoint()
    out = None
    for name in STAGES:
        module, cfg = STAGE_BY_NAME[name]
        module.LOG_LEVEL = LOG_LEVELS[name]
        t0  = time.time()
        if name == "screen":
            out = module.run(cfg)
        elif name == "grids":
            out = module.run(out)
        else:
            out = module.run(cfg, out)
        logger.info(f"🏁 {name.upper()} done in {_elapsed(time.time() - t0)}")
    return out

def _elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600} h {(s % 3600) // 60} min {s % 60} s"

def log_run_config() -> None:
    logger.info(f"\n{SEP}")
    logger.info("  RESEARCH START")
    logger.info(f"{SEP}")
    logger.info(f"  {'DATASET':<{LBL_W}}: {cr.DATASET} ── {cr.TIMEFRAMES} ── {len(cr.SYMBOLS)} symbols")
    logger.info(f"  {'CACHES':<{LBL_W}}: MODE={cr.MODE} NULL_PCT={cr.NULL_PCT} SELL_AFTER={cr.SELL_AFTER} "
                f"TP_PCT={cr.TP_PCT} SL_PCT={cr.SL_PCT}")
    logger.info(f"  {'ARTIFACTS':<{LBL_W}}: {cr.ARTIFACTS_DIR}")
    logger.info(f"  {'STAGES':<{LBL_W}}: {' → '.join(STAGES)}"
                + (f" (input: {NAMES[_first - 1]} checkpoint)" if _first > 0 else ""))
    for name in NAMES:
        mark = "🟢" if name in STAGES else "⚪"
        cfg  = STAGE_BY_NAME[name][1]
        logger.info(f"  {name.upper():<{LBL_W}}: {mark} {'no config' if cfg is None else cfg}")
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