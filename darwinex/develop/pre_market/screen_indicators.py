# develop/pre_market/screen_indicators.py
import os
import sys
import time
import math
import logging
import numpy as np
import pandas as pd

_HERE     = os.path.abspath(os.path.dirname(__file__))
_DARWINEX = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CORE     = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "core"))

for _p in (_HERE, _DARWINEX, _CORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.research.screen_rules")
logger.setLevel(logging.DEBUG)

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import build_flat_specs
from screen_engine import DATASET, SYMBOL_POOL, TIMEFRAME_GRID,grow_rules, build_validated_shortlist
from screen_engine import CTX_WARMUP_DAYS, WINDOW_MONTHS, N_HOLDOUT_WINDOWS, HOLDOUT_MIN_PASS_FRAC
from screen_engine import MAX_DEPTH, TOP_K_GROWTH, PHI_THRESHOLD,hr, format_rules
from screen_engine import compute_window_ids, forward_return, build_panel, build_mask_cache
from screen_engine import holdout_window_ids, split_selection_holdout_masks

# =============================================================================
# CONFIG — specific to this script
# =============================================================================
EVAL_HORIZON = 20

# =============================================================================
# TOP NON-REDUNDANT RULES — path stats (TP/SL exploration via MFE/MAE)
# =============================================================================
def report_top_rules(survivors_by_depth: dict, specs: list, panels: dict,
                      mask_cache: dict, fwd: dict, ok_fwd: dict, window_ids: dict,
                      holdout_ids: list) -> dict:
    horizon = EVAL_HORIZON
    top_rules_by_depth = {}
    holdout_required = math.ceil(HOLDOUT_MIN_PASS_FRAC * len(holdout_ids))

    for depth in range(2, MAX_DEPTH + 1):
        if depth not in survivors_by_depth:
            continue

        top, floor = build_validated_shortlist(survivors_by_depth[depth], specs, panels, mask_cache,
                                                 fwd, ok_fwd, window_ids, holdout_ids, horizon,
                                                 depth=depth, log_fn=logger.info)

        top_rules_by_depth[depth] = top
        indicators_involved = sorted({specs[i]["indicator"]
                                       for members in top["members"] for i in members})

        hr("=")
        logger.info(f"  DEPTH {depth} — TOP {len(top)} NON-REDUNDANT RULES "
                    f"(|edge| > junk floor {floor:.4f}, |phi| < {PHI_THRESHOLD}, "
                    f"holdout {holdout_required}/{len(holdout_ids)})")
        hr("=")
        disp = format_rules(top)
        disp["edge_nonoverlap"] = top["edge_nonoverlap"].round(4)
        disp["holdout"] = (top["holdout_n_pass"].astype(str) + "/"
                            + top["holdout_n_total"].astype(str))
        disp = disp[[
            "label", "depth",
            "edge", "edge_nonoverlap", "coverage", "pairs", "time", "holdout",
            "junk", "pass",
        ]]
        logger.info(disp.to_string(index=False))
        logger.info("")
        logger.info(f"  Indicators involved: {indicators_involved}")
        logger.info("")

    return top_rules_by_depth

# =============================================================================
# SINGLE-TIMEFRAME PIPELINE
# =============================================================================
def screen_one_timeframe(timeframe: str) -> dict:
    start = time.time()
    specs = build_flat_specs()

    hr("─")
    logger.info("  RULE PRE-SCREEN — THRESHOLD CROSSINGS AND AND-COMBINATIONS")
    hr("─")
    logger.info(f"  TIMEFRAME      : {timeframe}")
    logger.info(f"  SYMBOLS        : {len(SYMBOL_POOL)}")
    logger.info(f"  EVAL_HORIZON   : {EVAL_HORIZON} candles")
    logger.info(f"  CONDITIONS     : {len(specs)} (indicator x params x threshold x op)")
    logger.info(f"  METRIC         : Cohen's d of forward return, signal=True vs signal=False")
    logger.info(f"  GROWTH         : depth 1 selects top {TOP_K_GROWTH} indicators (only filter); "
                f"depth 2/3 each exhaustive & independent over that grid")
    logger.info(f"  SIGNIFICANCE   : no p-value — noise floor set by the JUNK controls' "
                f"own success rate at each depth")
    hr("─")

    ohlcv_by_tf = build_universe(
        DATA_FOLDER_BY_DATASET[DATASET], {timeframe: SYMBOL_POOL}, dataset=DATASET
    )
    ohlcv_data = ohlcv_by_tf[timeframe]
    ohlcv_arr  = prepare_ohlcv_arrays(ohlcv_data)

    first_ts   = min(pd.DatetimeIndex(df.index)[0] for df in ohlcv_data.values())
    warmup_end = first_ts + pd.Timedelta(days=CTX_WARMUP_DAYS)
    del ohlcv_data

    panels, fwd, ok_fwd, window_ids = {}, {}, {}, {}
    h = EVAL_HORIZON

    for i, sym in enumerate(SYMBOL_POOL, start=1):
        if sym not in ohlcv_arr:
            logger.warning(f"  [{i:>2}/{len(SYMBOL_POOL)}] {sym} missing, skipped")
            continue
        arr = ohlcv_arr[sym]
        n   = len(arr["close"])

        ok = pd.DatetimeIndex(arr["ts"]) > warmup_end
        fr = forward_return(arr["close"], h)
        fr[~ok] = np.nan

        panels[sym]     = build_panel(arr, sym, n)
        fwd[sym]        = fr
        ok_fwd[sym]     = ok & np.isfinite(fr)
        window_ids[sym] = compute_window_ids(arr["ts"], WINDOW_MONTHS)

    if not panels:
        logger.error("No pair was processed.")
        return None

    mask_cache  = build_mask_cache(panels, specs)
    holdout_ids = holdout_window_ids(window_ids, N_HOLDOUT_WINDOWS)
    selection_ok, _ = split_selection_holdout_masks(ok_fwd, window_ids, holdout_ids)

    logger.info(f"  HOLDOUT SPLIT  : last {N_HOLDOUT_WINDOWS} window(s) of {WINDOW_MONTHS}m "
                f"(window_ids {holdout_ids}) held out; selection uses everything before")

    survivors_by_depth, indicator_rates = grow_rules(specs, panels, fwd, selection_ok,
                                                       window_ids, mask_cache)
    top_rules_by_depth = report_top_rules(survivors_by_depth, specs, panels,
                                           mask_cache, fwd, ok_fwd, window_ids, holdout_ids)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 {timeframe} DONE — {elapsed // 3600}h {(elapsed % 3600) // 60}min {elapsed % 60}s")
    return indicator_rates

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    start = time.time()
    for tf in TIMEFRAME_GRID:
        screen_one_timeframe(tf)
        logger.info("")

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 60} min {elapsed % 60} s")


if __name__ == "__main__":
    main()