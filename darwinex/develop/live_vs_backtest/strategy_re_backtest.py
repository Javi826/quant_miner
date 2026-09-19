#quant_d/develop/live_vs_backtest/strategy_re_backtest.py
import os
import sys
import time
import logging

_BASE = os.path.dirname(__file__)
_BASE = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(_BASE, "..", "..", "darwinex", "shared")))
sys.path.append(os.path.abspath(os.path.join(_BASE, "..", "..", "darwinex", "BOT_forex", "darwinex")))

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", stream=sys.stdout, force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logger = logging.getLogger("BOT_forex.strategy_re_backtest")

from shared_batchs.symbols.universe import build_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_BY_DATASET
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import INITIAL_BALANCE, ORDER_AMOUNT
from shared_batchs.pipeline.wfo import build_ohlcv_with_signal
from shared_batchs.backtesters.ZX_compute_BT import prepare_backtest_data, run_backtest_from_prepared
from shared_batchs.utils.batch_metrics import compute_metrics
from live.strategies import STRATEGIES

# =============================================================================
# RUN CONFIGURATION
# =============================================================================
DATASET  = "OOS"
N_JOBS   = -1

SAVE_TRADES    = True
OUTPUT_FOLDER  = os.path.join(_BASE, "brief_trades")


# =============================================================================
# BUILD BACKTEST RULES FROM PRODUCTION STRATEGIES
# =============================================================================
def _build_rule(strategy: dict) -> dict:
    from signals.signal_builder import build_signal_fn

    return {
        "rule_id"     : strategy["id"],
        "timeframe"   : strategy["timeframe"],
        "side"        : strategy["direction"],
        "symbols"     : strategy["symbols"],
        "signal_fn"   : build_signal_fn(strategy["specs"], strategy["direction"]),
        "sell_after"  : strategy["sell_after_ncandles"],
        "tp_pct"      : strategy["tp_pct"],
        "sl_pct"      : strategy["sl_pct"],
        "order_amount": ORDER_AMOUNT,
    }


def _symbols_by_timeframe(strategies: list) -> dict:
    symbols_by_timeframe = {}
    for strategy in strategies:
        symbols = symbols_by_timeframe.setdefault(strategy["timeframe"], set())
        symbols.update(strategy["symbols"])
    return {tf: sorted(symbols) for tf, symbols in symbols_by_timeframe.items()}


# =============================================================================
# FULL BACKTEST — one rule at a time, fixed production params, full data range
# =============================================================================
def _run_full_backtest_for_rule(
    idx: int,
    total: int,
    rule: dict,
    ohlcv_arr: dict,
    log_level: int,
    save_trades: bool,
    output_folder: str,
) -> dict:

    logging.basicConfig(level=log_level, format="%(message)s", force=True)
    logging.getLogger("joblib").setLevel(logging.WARNING)

    rule_ohlcv_arr = {sym: ohlcv_arr[sym] for sym in rule["symbols"] if sym in ohlcv_arr}
    ohlcv_arrays   = build_ohlcv_with_signal(rule_ohlcv_arr, rule["signal_fn"], [], {})
    prepared_data = prepare_backtest_data(ohlcv_arrays)
    results       = run_backtest_from_prepared(
        prepared_data,
        sell_after   = rule["sell_after"],
        tp_pct       = rule["tp_pct"],
        sl_pct       = rule["sl_pct"],
        order_amount = rule["order_amount"],
    )

    trades             = results["__PORTFOLIO__"]["trade_log"].copy()
    trades.columns     = trades.columns.str.lower().str.strip()
    trades["buy_time"] = pd.to_datetime(trades["buy_time"])
    trades["strategy"] = rule["rule_id"]

    if save_trades and not trades.empty:
        os.makedirs(output_folder, exist_ok=True)
        trades.to_csv(
            os.path.join(output_folder, f"trades_full_{rule['rule_id']}.csv"),
            index=False,
        )

    metrics = compute_metrics(trades, capital=INITIAL_BALANCE, name="", include_weekly=False) if not trades.empty else None

    logger.debug(f"[{idx + 1}/{total}] {rule['side']:<5} {rule['rule_id']} -> n_trades={len(trades)}")

    return {
        **rule,
        "n_trades":     len(trades),
        "net_gain_pct": metrics["Net_Gain_pct"] if metrics else 0.0,
        "max_dd_pct":   metrics["Max_DD_pct"]   if metrics else 0.0,
        "win_rate":     metrics["Win_Rate"]     if metrics else 0.0,
        "first_trade":  trades["buy_time"].min() if not trades.empty else None,
        "last_trade":   trades["buy_time"].max() if not trades.empty else None,
    }


def _run_full_backtest(rules: list, ohlcv_arr: dict, timeframe: str) -> list:
    total = len(rules)

    with tqdm_joblib(tqdm(desc=f"FULL BACKTEST {timeframe}", total=total, dynamic_ncols=True)):
        results = Parallel(n_jobs=N_JOBS)(
            delayed(_run_full_backtest_for_rule)(
                i, total, rule, ohlcv_arr, LOG_LEVEL, SAVE_TRADES, OUTPUT_FOLDER,
            )
            for i, rule in enumerate(rules)
        )

    return results

# =============================================================================
# REPORT
# =============================================================================
def _print_summary(results: list, label: str) -> None:
    results_sorted = sorted(results, key=lambda r: r["net_gain_pct"], reverse=True)

    logger.info(f"\n{'=' * 110}")
    logger.info(f"  {label}")
    logger.info(f"{'=' * 110}")
    logger.info(
        f"  {'ID':<10} {'N_TRADES':>9} {'NET_GAIN%':>10} {'MAX_DD%':>9} {'WIN_RATE%':>10} "
        f"{'FIRST_TRADE':>13} {'LAST_TRADE':>13}"
    )
    logger.info(f"  {'-' * 106}")

    for r in results_sorted:
        short_id    = r["rule_id"].split("_")[0]
        first_trade = r["first_trade"].strftime("%Y-%m-%d") if r["first_trade"] is not None else "—"
        last_trade  = r["last_trade"].strftime("%Y-%m-%d")  if r["last_trade"]  is not None else "—"
        logger.info(
            f"  {short_id:<10} {r['n_trades']:>9} {r['net_gain_pct']:>10.1f} "
            f"{r['max_dd_pct']:>9.1f} {r['win_rate']:>10.1f} "
            f"{first_trade:>13} {last_trade:>13}"
        )

    logger.info(f"  {'=' * 110}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    strategies = [s for s in STRATEGIES if s.get("active", True)]
    timeframes = sorted({s["timeframe"] for s in strategies})
    symbols_by_timeframe = _symbols_by_timeframe(strategies)

    logger.info(f"\n{'=' * 115}")
    logger.info(f"  FULL BACKTEST START — production strategies vs current {DATASET} data (no WFO windowing)")
    logger.info(f"{'=' * 115}")
    logger.info(f"  STRATEGIES  : {len(strategies)}")
    logger.info(f"  TIMEFRAMES  : {timeframes}")
    logger.info(f"  SYMBOLS     : {symbols_by_timeframe}")
    logger.info(f"{'=' * 115}\n")

    all_results = []

    for timeframe in timeframes:
        tf_start = time.time()

        tf_strategies = [s for s in strategies if s["timeframe"] == timeframe]
        rules         = [_build_rule(s) for s in tf_strategies]

        ohlcv_data = build_universe(
            DATA_FOLDER_BY_DATASET[DATASET],
            {timeframe: symbols_by_timeframe[timeframe]},
            dataset=DATASET,
        )[timeframe]
        ohlcv_arr = prepare_ohlcv_arrays(ohlcv_data)

        tf_results = _run_full_backtest(rules, ohlcv_arr, timeframe)
        all_results.extend(tf_results)

        tf_elapsed = int(time.time() - tf_start)
        logger.info(f"\n🏁 {timeframe} DONE — {tf_elapsed // 3600} h {(tf_elapsed % 3600) // 60} min {tf_elapsed % 60} s")

    _print_summary(all_results, "PRODUCTION FULL BACKTEST")

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")