# quant_miner/darwinex/BOT_sweep/sweep_backtest_fx.py (forex)
import os
import sys
import json
import time
import logging
from contextlib import contextmanager
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",".."))       # quant_miner
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "core"))
sys.path.append(os.path.join(_ROOT, "darwinex"))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import ORDER_AMOUNT
from setup.config_core import settings
from utils.ohlcv_utils import prepare_ohlcv_arrays
from rule_mining import rule_runner
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH, validate_indicators_by_timeframe

logger = logging.getLogger("BOT_sweep.backtest")

# =============================================================================
# CONFIG
# =============================================================================
PIPELINE_LOG_LEVEL = logging.INFO       # BOT_batch.* loggers ── WARNING: only this script and the bars

SPLIT_MODE = True                       # as backtesting_fx
DATASET_IS, DATASET_OOS = ("IS", "OOS") if SPLIT_MODE else ("MERGED", "MERGED")

SWEEP_DIR    = os.path.join(os.path.dirname(__file__), "sweep")
CONFIGS_DIR  = os.path.join(SWEEP_DIR, "configs")        # written by sweep_research_fx
RESULTS_PATH = os.path.join(SWEEP_DIR, "results.csv")    # one row per config: a config already here is not run again

FUNNEL_STAGES = ["mbias", "wfo"]        # captured by pipeline_patches: the pipeline stops right after POST-WFO

SEP   = "═" * 115
LBL_W = 12

# =============================================================================
# PIPELINE PATCHES ── funnel counts of rule_runner, stop right after POST-WFO
# =============================================================================
class _StopAfterWFO(Exception):
    """Raised after the POST-WFO table: correlation, multiverse and portfolio are not run."""


@contextmanager
def pipeline_patches():
    cap         = {}
    orig_min_is = rule_runner.print_rule_mining_min_by_group_is
    orig_min    = rule_runner.print_rule_mining_min_by_group

    def min_by_group_is(rows, stage_label, candidate_rows):
        cap["mbias"] = (len(rows), len(candidate_rows))
        return orig_min_is(rows, stage_label, candidate_rows)

    def min_by_group(all_raw_results, highlight_ids, stage_label, candidate_ids):
        if stage_label != "POST-WFO":
            return orig_min(all_raw_results, highlight_ids, stage_label, candidate_ids)
        cap["wfo"] = (len(highlight_ids), len(candidate_ids))
        orig_min(all_raw_results, highlight_ids, stage_label, candidate_ids)      # the POST-WFO table, then stop
        raise _StopAfterWFO

    rule_runner.print_rule_mining_min_by_group_is = min_by_group_is
    rule_runner.print_rule_mining_min_by_group    = min_by_group
    try:
        yield cap
    finally:
        rule_runner.print_rule_mining_min_by_group_is = orig_min_is
        rule_runner.print_rule_mining_min_by_group    = orig_min

# =============================================================================
# COMBOS AND DATA ── as backtesting_fx, with the config passed in
# =============================================================================
def build_combos(timeframes: list, symbol_combos_by_timeframe: dict) -> list:
    return [
        {
            "combo_key": f"{timeframe}_c{combo_idx:02d}",
            "timeframe": timeframe,
            "symbols":   list(symbols),
        }
        for timeframe in timeframes
        for combo_idx, symbols in enumerate(symbol_combos_by_timeframe[timeframe], start=1)
    ]


def load_ohlcv_by_combo(combos: list, data_folder: str, dataset: str) -> tuple:
    ohlcv_data_by_combo, ohlcv_arr_by_combo = {}, {}
    for combo in combos:
        combo_key, timeframe = combo["combo_key"], combo["timeframe"]
        ohlcv_data = build_universe(
            data_folder, {timeframe: combo["symbols"]}, dataset=dataset,
        )[timeframe]
        ohlcv_data_by_combo[combo_key] = ohlcv_data
        ohlcv_arr_by_combo[combo_key]  = prepare_ohlcv_arrays(ohlcv_data)
    return ohlcv_data_by_combo, ohlcv_arr_by_combo

# =============================================================================
# CONFIGS AND RESULTS
# =============================================================================
def _load_configs() -> list:
    if not os.path.isdir(CONFIGS_DIR):
        raise FileNotFoundError(f"No folder {CONFIGS_DIR} (run sweep_research_fx first)")
    configs = []
    for fname in sorted(os.listdir(CONFIGS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(CONFIGS_DIR, fname), encoding="utf-8") as fh:
            configs.append((fname[:-len(".json")], json.load(fh)))
    bad = [name for name, doc in configs if doc["backtest_mode"] != str(settings.BACKTEST_MODE)]
    if bad:
        raise ValueError(f"Configs researched with another BACKTEST_MODE (now {settings.BACKTEST_MODE}): {bad}")
    return configs


def _done_configs() -> set:
    return set(pd.read_csv(RESULTS_PATH)["config"]) if os.path.exists(RESULTS_PATH) else set()


def _append_row(row: dict) -> None:
    new = pd.DataFrame([row])
    df  = pd.concat([pd.read_csv(RESULTS_PATH), new], ignore_index=True) if os.path.exists(RESULTS_PATH) else new
    os.makedirs(SWEEP_DIR, exist_ok=True)
    tmp = f"{RESULTS_PATH}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, RESULTS_PATH)               # atomic: an interrupted write never leaves a broken csv


def _funnel(cap: dict) -> dict:
    row = {}
    for key in FUNNEL_STAGES:
        passed, candidates = cap.get(key, (0, 0))
        row[f"{key}_candidates"], row[f"{key}_passed"] = candidates, passed
    row["wfo_pct"] = round(100 * row["wfo_passed"] / row["wfo_candidates"], 2) if row["wfo_candidates"] else 0.0
    return row

# =============================================================================
# SINGLE CONFIG RUN
# =============================================================================
def run_one(doc: dict) -> dict:
    combos_by_tf = {tf: c for tf, c in doc["SYMBOL_COMBOS_BY_TIMEFRAME"].items() if c}   # a timeframe with no combos is dropped
    timeframes   = list(combos_by_tf)
    if not timeframes:
        return {"timeframes": "", **_funnel({})}
    param_grid = {tf: doc["PARAM_GRID_BY_TIMEFRAME"][tf] for tf in timeframes}
    indicators = {tf: doc["SELECTED_INDICATORS_BY_TIMEFRAME"][tf] for tf in timeframes}
    validate_indicators_by_timeframe(indicators, timeframes)

    combos = build_combos(timeframes, combos_by_tf)
    ohlcv_data_is, ohlcv_arr_is = load_ohlcv_by_combo(combos, DATA_FOLDER_BY_DATASET[DATASET_IS], DATASET_IS)
    if SPLIT_MODE:
        ohlcv_data_oos, ohlcv_arr_oos = load_ohlcv_by_combo(combos, DATA_FOLDER_BY_DATASET[DATASET_OOS], DATASET_OOS)
    else:
        ohlcv_data_oos, ohlcv_arr_oos = ohlcv_data_is, ohlcv_arr_is

    with pipeline_patches() as cap:
        try:
            rule_runner.run_rule_mining_pipeline(
                ohlcv_data_is_by_combo  = ohlcv_data_is,
                ohlcv_arr_is_by_combo   = ohlcv_arr_is,
                ohlcv_data_oos_by_combo = ohlcv_data_oos,
                ohlcv_arr_oos_by_combo  = ohlcv_arr_oos,
                combos                  = combos,
                param_grid              = param_grid,
                indicators_by_timeframe = indicators,
                order_amount            = ORDER_AMOUNT,
                data_folder             = DATA_FOLDER_BY_DATASET[DATASET_OOS],
                max_depth               = RULE_MAX_DEPTH,
                log_level               = PIPELINE_LOG_LEVEL,
                show_plots              = False,
                run_deploy              = False,
            )
        except _StopAfterWFO:
            pass
    missing = [key for key in FUNNEL_STAGES if key not in cap]
    if missing:
        raise RuntimeError(f"Funnel stages not captured: {missing} (update pipeline_patches)")
    return {"timeframes": ",".join(timeframes), **_funnel(cap)}

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    configs = _load_configs()
    done    = _done_configs()
    pending = [(name, doc) for name, doc in configs if name not in done]
    log_sweep_config(len(configs), len(pending))

    failed = []
    for i, (name, doc) in enumerate(pending, start=1):
        logger.info(f"\n{SEP}\n  CONFIG {i}/{len(pending)} ── {name}\n{SEP}")
        t0 = time.time()
        try:
            funnel = run_one(doc)
        except Exception:
            logger.exception(f"❌ {name} failed: not written, it is run again next time")
            failed.append(name)
            continue
        elapsed = time.time() - t0
        _append_row({"config": name, **doc["params"], **funnel, "elapsed_s": int(elapsed)})
        logger.info(f"\n💾 {name} ── WFO {funnel['wfo_passed']}/{funnel['wfo_candidates']} ({funnel['wfo_pct']}%) ── "
                    f"{_elapsed(elapsed)}")
    if failed:
        logger.warning(f"\n⚠  {len(failed)} configs failed: {failed}")


def _elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600} h {(s % 3600) // 60} min {s % 60} s"


def log_sweep_config(n_total: int, n_pending: int) -> None:
    logger.info(f"\n{SEP}")
    logger.info("  SWEEP BACKTEST START")
    logger.info(f"{SEP}")
    dataset = f"SPLIT ── IS: {DATASET_IS} | OOS: {DATASET_OOS}" if SPLIT_MODE else DATASET_IS
    logger.info(f"  {'DATASET':<{LBL_W}}: {dataset}")
    logger.info(f"  {'BACKTEST':<{LBL_W}}: {settings.BACKTEST_MODE}")
    logger.info(f"  {'CONFIGS':<{LBL_W}}: {n_total} ── done: {n_total - n_pending} ── pending: {n_pending}")
    logger.info(f"  {'INPUT':<{LBL_W}}: {CONFIGS_DIR}")
    logger.info(f"  {'OUTPUT':<{LBL_W}}: {RESULTS_PATH}")
    logger.info(f"{SEP}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    logger.setLevel(logging.INFO)
    logging.getLogger("BOT_batch").setLevel(PIPELINE_LOG_LEVEL)
    for noisy_logger in ("joblib", "matplotlib", "numba"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    start = time.time()
    try:
        main()
        logger.info(f"\n🏁 TOTAL ── {_elapsed(time.time() - start)}")
    except KeyboardInterrupt:
        logger.info(f"\n⛔  INTERRUPTED BY USER ── {_elapsed(time.time() - start)}")
        sys.exit(0)