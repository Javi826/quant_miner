# develop/pre_market/trade_exits.py
import os
import sys
import time
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
logger = logging.getLogger("BOT_batch.research.trade_exits")

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import build_flat_specs
from screen_engine import DATASET, SYMBOL_POOL, TIMEFRAME_GRID, CTX_WARMUP_DAYS, WINDOW_MONTHS
from screen_engine import MAX_DEPTH, N_HOLDOUT_WINDOWS,combo_label,grow_rules, build_validated_shortlist
from screen_engine import build_panel, build_mask_cache, condition_mask, forward_return, compute_window_ids
from screen_engine import holdout_window_ids, split_selection_holdout_masks

# =============================================================================
# CONFIG — specific to this script
# =============================================================================
EVAL_HORIZON   = 100   # single horizon for the whole calibration — generous margin so

ATR_PERIOD     = 14
TP_MULTIPLIERS = [2,4,6,8,10,12,14]
SL_MULTIPLIERS = [2,4,6,8,10,12,14]

BARS_TO_CLOSE_PERCENTILE = 85   # percentile of bars-to-close used instead of the,median — e.g. 80 means 80% of trades have already,resolved (touched TP or SL) by that many candles
SELL_AFTER_SAFETY_MULT   = 1.2  # safety margin applied over the observed
TIMEOUT_WARNING_PCT      = 15.0 # if the winning combo's mean_timeout_pct exceeds this, EVAL_HORIZON is too short and is cutting trades of before they can resolve — results become unreliable

# =============================================================================
# SCREEN AT THE SINGLE EVAL HORIZON — same depth1/2/3 growth as screen_rules.py,
# =============================================================================
def screen_fixed_horizon(specs: list, panels: dict, ohlcv_arr: dict, plain_ok: dict,
                          window_ids: dict, mask_cache: dict, holdout_ids: list) -> dict:
    fwd, ok_fwd = {}, {}
    for sym, arr in ohlcv_arr.items():
        if sym not in panels:
            continue
        fr = forward_return(arr["close"], EVAL_HORIZON)
        fr[~plain_ok[sym]] = np.nan
        fwd[sym]    = fr
        ok_fwd[sym] = plain_ok[sym] & np.isfinite(fr)

    # exclude holdout windows from depth 1/2/3 growth, same as screen_indicators.py
    selection_ok, _ = split_selection_holdout_masks(ok_fwd, window_ids, holdout_ids)
    survivors_by_depth, _ = grow_rules(specs, panels, fwd, selection_ok, window_ids, mask_cache)

    shortlists = {}
    for depth in range(2, MAX_DEPTH + 1):
        shortlist, _floor = build_validated_shortlist(survivors_by_depth[depth], specs, panels, mask_cache,
                                                        fwd, ok_fwd, window_ids, holdout_ids, EVAL_HORIZON,
                                                        depth=depth, log_fn=logger.debug)
        shortlists[depth] = shortlist

    return shortlists, ok_fwd

# =============================================================================
# TP/SL CALIBRATION — each market's ATR%, decided BEFORE looking at any
# =============================================================================
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])


def compute_atr_pct_by_symbol(ohlcv_arr: dict, eval_ok: dict, period: int) -> dict:
    atr_pct = {}
    for sym, arr in ohlcv_arr.items():
        if sym not in eval_ok:
            continue
        tr  = _true_range(arr["high"], arr["low"], arr["close"])
        atr = pd.Series(tr).rolling(period).mean().to_numpy()
        ok  = eval_ok[sym] & np.isfinite(atr)
        if ok.sum() == 0:
            continue
        atr_pct[sym] = float(np.mean(atr[ok] / arr["close"][ok]))
    return atr_pct

def _rule_fixed_exit_stats(members: tuple, direction: int, specs: list, panels: dict,
                            ohlcv_arr: dict, eval_ok: dict, horizon: int,
                            tp_pct_by_sym: dict, sl_pct_by_sym: dict,
                            bars_percentile: float = BARS_TO_CLOSE_PERCENTILE) -> tuple:

    n_timeout, n_tp, n_sl = 0, 0, 0
    bars_to_close = []
    for sym, panel in panels.items():
        if sym not in tp_pct_by_sym:
            continue
        mask = None
        for i in members:
            m = condition_mask(panel, specs[i])
            mask = m if mask is None else (mask & m)

        arr = ohlcv_arr[sym]
        n = len(arr["close"])
        idx = np.where(eval_ok[sym] & mask)[0]
        idx = idx[idx + horizon < n]

        tp_pct, sl_pct = tp_pct_by_sym[sym], sl_pct_by_sym[sym]
        for t in idx:
            entry = arr["close"][t]
            if entry == 0 or not np.isfinite(entry):
                continue
            w_high = arr["high"][t + 1:t + horizon + 1]
            w_low  = arr["low"][t + 1:t + horizon + 1]

            if direction >= 0:
                tp_level, sl_level = entry * (1 + tp_pct), entry * (1 - sl_pct)
                tp_hits = np.where(w_high >= tp_level)[0]
                sl_hits = np.where(w_low  <= sl_level)[0]
            else:
                tp_level, sl_level = entry * (1 - tp_pct), entry * (1 + sl_pct)
                tp_hits = np.where(w_low  <= tp_level)[0]
                sl_hits = np.where(w_high >= sl_level)[0]

            first_tp = tp_hits[0] if len(tp_hits) else None
            first_sl = sl_hits[0] if len(sl_hits) else None

            if first_tp is None and first_sl is None:
                n_timeout += 1
            elif first_sl is None or (first_tp is not None and first_tp <= first_sl):
                n_tp += 1
                bars_to_close.append(first_tp + 1)
            else:
                n_sl += 1
                bars_to_close.append(first_sl + 1)

    total = n_tp + n_sl + n_timeout
    if total == 0:
        return np.nan, np.nan, np.nan, 0, (0, 0, 0), np.nan
    bars_pctl = float(np.percentile(bars_to_close, bars_percentile)) if bars_to_close else np.nan
    return n_tp / total, n_sl / total, n_timeout / total, total, (n_tp, n_sl, n_timeout), bars_pctl
def calibrate_tp_sl(shortlist_by_depth: dict, specs: list, panels: dict, ohlcv_arr: dict,
                     eval_ok: dict, horizon: int, atr_pct_by_sym: dict) -> pd.DataFrame:
    rows = []
    for depth, shortlist in shortlist_by_depth.items():
        for _, rule in shortlist.iterrows():
            direction = 1 if rule["edge_mean"] >= 0 else -1
            label = combo_label(rule["members"], specs)

            for k_tp in TP_MULTIPLIERS:
                for k_sl in SL_MULTIPLIERS:
                    tp_pct_by_sym = {s: k_tp * a for s, a in atr_pct_by_sym.items()}
                    sl_pct_by_sym = {s: k_sl * a for s, a in atr_pct_by_sym.items()}

                    hit_tp, hit_sl, timeout, n, raw_counts, bars_pctl = _rule_fixed_exit_stats(
                        rule["members"], direction, specs, panels, ohlcv_arr, eval_ok,
                        horizon, tp_pct_by_sym, sl_pct_by_sym
                    )
                    if n == 0:
                        continue

                    mean_atr_pct = float(np.mean(list(atr_pct_by_sym.values())))
                    tp_pct_avg   = k_tp * mean_atr_pct
                    sl_pct_avg   = k_sl * mean_atr_pct
                    expectancy   = hit_tp * tp_pct_avg - hit_sl * sl_pct_avg
                    n_tp, n_sl, n_timeout = raw_counts

                    rows.append({
                        "label": label, "depth": depth, "k_tp": k_tp, "k_sl": k_sl,
                        "tp_pct": round(tp_pct_avg * 100, 3), "sl_pct": round(sl_pct_avg * 100, 3),
                        "hit_tp_pct": round(hit_tp * 100, 1), "hit_sl_pct": round(hit_sl * 100, 1),
                        "timeout_pct": round(timeout * 100, 3), "n_timeout": n_timeout,
                        "n_triggers": n, "expectancy_pct": round(expectancy * 100, 4),
                        "bars_to_close": round(bars_pctl, 1) if np.isfinite(bars_pctl) else np.nan,
                    })

    return (pd.DataFrame(rows)
            .sort_values(["label", "expectancy_pct"], ascending=[True, False])
            .reset_index(drop=True))

def aggregate_grid_across_rules(tp_sl_table: pd.DataFrame) -> pd.DataFrame:

    if tp_sl_table.empty:
        return tp_sl_table

    def _wavg(g, col):
        sub = g.dropna(subset=[col])
        if sub.empty:
            return np.nan
        return np.average(sub[col], weights=sub["n_triggers"])

    rows = []
    for (k_tp, k_sl), g in tp_sl_table.groupby(["k_tp", "k_sl"]):
        rows.append({
            "k_tp": k_tp, "k_sl": k_sl,
            "mean_expectancy_pct":   round(_wavg(g, "expectancy_pct"), 4),
            "median_expectancy_pct": round(g["expectancy_pct"].median(), 4),
            "mean_hit_tp_pct":       round(_wavg(g, "hit_tp_pct"), 1),
            "mean_hit_sl_pct":       round(_wavg(g, "hit_sl_pct"), 1),
            "mean_timeout_pct":      round(_wavg(g, "timeout_pct"), 1),
            "mean_bars_to_close":    round(_wavg(g, "bars_to_close"), 1),
            "n_rules":               g["label"].nunique(),
            "total_triggers":        int(g["n_triggers"].sum()),
        })

    return (pd.DataFrame(rows)
            .sort_values("mean_expectancy_pct", ascending=False)
            .reset_index(drop=True))

def compute_tp_sl_recommendation(agg_grid: pd.DataFrame, atr_pct_by_sym: dict) -> dict:

    if agg_grid.empty:
        return None

    logger.debug(agg_grid.to_string(index=False))

    best = agg_grid.iloc[0]
    k_tp, k_sl = best["k_tp"], best["k_sl"]

    per_market = pd.DataFrame({
        "atr_pct":  {s: round(a * 100, 3) for s, a in atr_pct_by_sym.items()},
        "tp_pct":   {s: round(k_tp * a * 100, 3) for s, a in atr_pct_by_sym.items()},
        "sl_pct":   {s: round(k_sl * a * 100, 3) for s, a in atr_pct_by_sym.items()},
    }).sort_index()

    tp_sorted = per_market.sort_values("tp_pct")
    lo, mid, hi = tp_sorted.iloc[0], tp_sorted.iloc[len(tp_sorted) // 2], tp_sorted.iloc[-1]
    tp_array = [float(lo["tp_pct"]), float(mid["tp_pct"]), float(hi["tp_pct"])]
    sl_array = [float(lo["sl_pct"]), float(mid["sl_pct"]), float(hi["sl_pct"])]

    return {"best": best, "per_market": per_market, "tp_array": tp_array, "sl_array": sl_array}


# =============================================================================
# SELL_AFTER RECOMMENDATION — derived from the observed median bars-to-close
# =============================================================================
def compute_sell_after(best_combo: pd.Series, safety_mult: float = SELL_AFTER_SAFETY_MULT) -> int:
    bars = best_combo["mean_bars_to_close"] if best_combo is not None else np.nan
    if not np.isfinite(bars):
        return EVAL_HORIZON
    return int(np.ceil(bars * safety_mult))


# =============================================================================
# FINAL SUMMARY — single place where every recommendation is printed once.
# =============================================================================
def print_final_summary(tp_sl: dict, sell_after: int, safety_mult: float = SELL_AFTER_SAFETY_MULT) -> None:
    logger.info("\n" + "=" * 78)
    logger.info("  FINAL RECOMMENDATION")
    logger.info("=" * 78)

    if tp_sl is None:
        logger.info(f"  No surviving rule to calibrate TP/SL against.")
        logger.info(f"  sell_after fallback: {sell_after} candles (EVAL_HORIZON)")
        return

    best = tp_sl["best"]
    logger.info(f"  k_tp={best['k_tp']}, k_sl={best['k_sl']} "
                f"(mean_expectancy_pct={best['mean_expectancy_pct']}, "
                f"n_rules={int(best['n_rules'])}, total_triggers={int(best['total_triggers'])}, "
                f"mean_timeout_pct={best['mean_timeout_pct']})")

    if best["mean_timeout_pct"] > TIMEOUT_WARNING_PCT:
        logger.warning(f"⚠️ [WARNING] mean_timeout_pct={best['mean_timeout_pct']}% exceeds "
                        f"{TIMEOUT_WARNING_PCT}% — EVAL_HORIZON={EVAL_HORIZON} may be cutting "
                        f"trades off before they resolve. Consider raising EVAL_HORIZON.")

    logger.info(f"\n  Ready-to-copy TP/SL % per market:")
    logger.info(tp_sl["per_market"].to_string())

    logger.info(f"\n  BACKTEST ARRAY (min / median / max, same row per symbol so TP/SL "
                f"stay proportional):")
    logger.info(f"    TP array: {tp_sl['tp_array']}")
    logger.info(f"    SL array: {tp_sl['sl_array']}")

    logger.info(f"\n  sell_after = {sell_after} candles "
                f"(p{BARS_TO_CLOSE_PERCENTILE} bars-to-close {best['mean_bars_to_close']} "
                f"x safety margin {safety_mult})")
    logger.info("")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    start = time.time()
    timeframe = TIMEFRAME_GRID[0]
    specs = build_flat_specs()

    logger.info("─" * 110)
    logger.info("  TRADE EXITS — TP/SL AND SELL_AFTER CALIBRATION")
    logger.info("─" * 110)
    logger.info(f"  TIMEFRAME      : {timeframe}")
    logger.info(f"  SYMBOLS        : {len(SYMBOL_POOL)}")
    logger.info(f"  EVAL_HORIZON   : {EVAL_HORIZON} candles")
    logger.info("─" * 110)

    ohlcv_by_tf = build_universe(
        DATA_FOLDER_BY_DATASET[DATASET], {timeframe: SYMBOL_POOL}, dataset=DATASET
    )

    ohlcv_data = ohlcv_by_tf[timeframe]
    ohlcv_arr  = prepare_ohlcv_arrays(ohlcv_data)

    first_ts   = min(pd.DatetimeIndex(df.index)[0] for df in ohlcv_data.values())
    warmup_end = first_ts + pd.Timedelta(days=CTX_WARMUP_DAYS)
    del ohlcv_data

    panels, plain_ok, window_ids = {}, {}, {}
    for sym in SYMBOL_POOL:
        if sym not in ohlcv_arr:
            continue
        arr = ohlcv_arr[sym]
        n   = len(arr["close"])
        plain_ok[sym]   = pd.DatetimeIndex(arr["ts"]) > warmup_end
        panels[sym]     = build_panel(arr, sym, n)
        window_ids[sym] = compute_window_ids(arr["ts"], WINDOW_MONTHS)
    mask_cache  = build_mask_cache(panels, specs)
    holdout_ids = holdout_window_ids(window_ids, N_HOLDOUT_WINDOWS)
    logger.info(f"  HOLDOUT SPLIT  : last {N_HOLDOUT_WINDOWS} window(s) of {WINDOW_MONTHS}m "
                f"(window_ids {holdout_ids}) held out; selection uses everything before")

    # --- ATR% per market (decided before looking at any trigger's future) ---
    atr_pct_by_sym = compute_atr_pct_by_symbol(ohlcv_arr, plain_ok, ATR_PERIOD)
    logger.info("\n" + "=" * 78)
    logger.info(f"  ATR({ATR_PERIOD}) % BY MARKET")
    logger.info("=" * 78)
    atr_table = pd.Series(atr_pct_by_sym, name="atr_pct").mul(100).round(3).sort_values(ascending=False)
    logger.info(atr_table.to_string())
    logger.info("")

    # --- single screen at EVAL_HORIZON ---
    logger.info(f"\n{'=' * 78}\n  SCREENING at horizon={EVAL_HORIZON} candles\n{'=' * 78}")
    shortlists, ok_fwd = screen_fixed_horizon(specs, panels, ohlcv_arr, plain_ok, window_ids,
                                               mask_cache, holdout_ids)

    # --- TP/SL calibration ---
    logger.debug("\n" + "=" * 110)
    logger.debug(f"  TP/SL GRID — ATR-based, evaluated at horizon={EVAL_HORIZON}")
    logger.debug("=" * 110)
    tp_sl_table = calibrate_tp_sl(shortlists, specs, panels, ohlcv_arr, ok_fwd,
                                   EVAL_HORIZON, atr_pct_by_sym)
    if tp_sl_table.empty:
        tp_sl = None
    else:
        pd.set_option("display.width", 160)
        logger.debug(f"  {len(tp_sl_table)} rows")
        logger.debug(tp_sl_table.to_string(index=False))
        agg_grid = aggregate_grid_across_rules(tp_sl_table)
        tp_sl    = compute_tp_sl_recommendation(agg_grid, atr_pct_by_sym)

    # --- sell_after, derived from the winning combo's observed bars-to-close ---
    sell_after = compute_sell_after(tp_sl["best"] if tp_sl else None)

    # --- everything printed once, here ---
    print_final_summary(tp_sl, sell_after)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 {timeframe} DONE — {elapsed // 3600}h {(elapsed % 3600) // 60}min {elapsed % 60}s")


if __name__ == "__main__":
    main()