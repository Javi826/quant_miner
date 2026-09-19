#quant_b/bitget/BOT_batch_E1/research/strategy_re_backtest.py (crypto)
import os
import sys

_BITGET       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget"))
_BOT_BATCH_E1 = os.path.join(_BITGET, "BOT_batch_E1")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(_BITGET)
sys.path.append(os.path.join(_BITGET, "shared"))
sys.path.append(os.path.join(_BITGET, "shared", "shared_batchs"))
sys.path.append(os.path.join(_BITGET, "signals"))

import time
import logging
import importlib.util
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
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger("BOT_batch.main_rule_validation")

from shared_batchs.symbols.universe import build_universe
DATA_FOLDER_IS = os.path.join(
    _BITGET,
    "data_crypto", "data", "04_split", "expanding", "IS", "crypto_2022-01_2026-09_IS",
)

from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import INITIAL_BALANCE, ORDER_AMOUNT
from shared_batchs.pipeline.wfo import build_ohlcv_with_signal
from shared_batchs.backtesters.ZX_compute_BT import prepare_backtest_data, run_backtest_from_prepared
from shared_batchs.utils.batch_metrics import compute_metrics
from signals.signal_builder import build_signal_fn

# =============================================================================
# RUN CONFIGURATION
# =============================================================================
RULES_N_JOBS = -1

SAVE_TRADES          = True
STRATEGIES_E1_FOLDER = os.path.join(_BOT_BATCH_E1, "strategies_E1")
BRIEF_TRADES_FOLDER  = os.path.join(os.path.dirname(__file__), "brief_trades")
RULES_BATCH_PATH     = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch_topV.py")

# =============================================================================
# LOAD PRODUCTION STRATEGIES
# =============================================================================

def _load_strategies(rules_batch_path: str) -> list:
    spec   = importlib.util.spec_from_file_location("rules_batch", rules_batch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STRATEGIES

def _build_validation_rule(timeframe: str, entry: dict) -> dict:
    """Builds a rule dict carrying its own fixed production params (no grid search)."""
    side = entry["direction"]
    return {
        "rule_id":      entry["id"],
        "timeframe":    timeframe,
        "side":         side,
        "specs":        entry["specs"],
        "label":        entry["id"],
        "signal_fn":    build_signal_fn(entry["specs"], side),
        "symbols":      entry["symbols"],
        "sell_after":   entry["sell_after_ncandles"],
        "tp_pct":       entry["tp_pct"],
        "sl_pct":       entry["sl_pct"],
        "order_amount": ORDER_AMOUNT,
    }
# =============================================================================
# FULL BACKTEST — one rule at a time, fixed params, full data range (no windowing)
# =============================================================================

def _run_full_backtest_for_rule(
    idx: int,
    total: int,
    rule: dict,
    ohlcv_arr: dict,
    log_level: int,
    save_trades: bool,
    brief_trades_folder: str,
) -> dict:
    """Runs a single full backtest for one rule over the entire available data range.
    No walk-forward windowing, no edge-candle skipping: any trade still open at the
    end of the data closes at the last available candle's close price."""
    logging.basicConfig(level=log_level, format="%(message)s", force=True)
    logging.getLogger("joblib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

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

    if save_trades and not trades.empty:
        os.makedirs(brief_trades_folder, exist_ok=True)
        trades.to_csv(
            os.path.join(brief_trades_folder, f"trades_full_{rule['rule_id']}.csv"),
            index=False,
        )

    metrics = compute_metrics(trades, capital=INITIAL_BALANCE, name="", include_weekly=False) if not trades.empty else None

    logger.debug(f"[{idx + 1}/{total}] {rule['side']:<5} {rule['label']} -> n_trades={len(trades)}")

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
        results = Parallel(n_jobs=RULES_N_JOBS)(
            delayed(_run_full_backtest_for_rule)(
                i, total, rule, ohlcv_arr, LOG_LEVEL, SAVE_TRADES, BRIEF_TRADES_FOLDER,
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
        f"  {'ID':<8} {'N_TRADES':>9} {'NET_GAIN%':>10} {'MAX_DD%':>9} {'WIN_RATE%':>10} "
        f"{'FIRST_TRADE':>13} {'LAST_TRADE':>13}"
    )
    logger.info(f"  {'-' * 106}")

    for r in results_sorted:
        short_id    = r["label"].split("_")[0]
        first_trade = r["first_trade"].strftime("%Y-%m-%d") if r["first_trade"] is not None else "—"
        last_trade  = r["last_trade"].strftime("%Y-%m-%d")  if r["last_trade"]  is not None else "—"
        logger.info(
            f"  {short_id:<8} {r['n_trades']:>9} {r['net_gain_pct']:>10.1f} "
            f"{r['max_dd_pct']:>9.1f} {r['win_rate']:>10.1f} "
            f"{first_trade:>13} {last_trade:>13}"
        )

    logger.info(f"  {'=' * 110}\n")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    strategies = _load_strategies(RULES_BATCH_PATH)
    timeframes = sorted({entry["timeframe"] for entry in strategies})

    logger.info(f"\n{'=' * 115}")
    logger.info(f"  FULL BACKTEST START — production rules vs current IS data (no WFO windowing)")
    logger.info(f"{'=' * 115}")
    logger.info(f"  RULES FILE  : {RULES_BATCH_PATH}")
    logger.info(f"  STRATEGIES  : {len(strategies)}")
    logger.info(f"  TIMEFRAMES  : {timeframes}")
    logger.info(f"{'=' * 115}\n")

    all_results = []

    for timeframe in timeframes:
        tf_start = time.time()

        tf_strategies = [s for s in strategies if s["timeframe"] == timeframe]
        tf_symbols    = sorted({sym for entry in tf_strategies for sym in entry["symbols"]})

        ohlcv_is = build_universe(
            data_folder_is       = DATA_FOLDER_IS,
            symbols_by_timeframe = {timeframe: tf_symbols},
            dataset              = "IS",
        )[timeframe]
        ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)

        rules = [
            _build_validation_rule(timeframe, entry)
            for entry in tf_strategies
        ]

        tf_results = _run_full_backtest(rules, ohlcv_arr, timeframe)
        all_results.extend(tf_results)

        tf_elapsed = int(time.time() - tf_start)
        logger.info(f"\n🏁 {timeframe} DONE — {tf_elapsed // 3600} h {(tf_elapsed % 3600) // 60} min {tf_elapsed % 60} s")

    _print_summary(all_results, "PRODUCTION FULL BACKTEST")

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")