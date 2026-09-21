# core/pipeline/wfo.py
import os
import logging
import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from setup.config_backtest import INITIAL_BALANCE
import importlib
from setup.config_core import settings

_bt = importlib.import_module(f"backtesters.ZX_compute_BT_{settings.BACKTEST_MODE}")
prepare_backtest_data            = _bt.prepare_backtest_data
run_backtest_from_prepared       = _bt.run_backtest_from_prepared
run_backtest_from_prepared_light = _bt.run_backtest_from_prepared_light
from engines.wfo_WF import walk_forward_optimization
from utils.ohlcv_utils import get_bars_per_year
from utils.batch_metrics import compute_metrics
DTYPE  = np.float32
logger = logging.getLogger("BOT_batch.pipeline.wfo")

# =============================================================================
# WFO APPROVAL THRESHOLDS
# =============================================================================
# =============================================================================
# WFO_NET_GAIN_TH = 30
# WFO_DD_TH       = 15
# WFO_R2_TH       = 0.8
# WFO_WFR_TH      = 0.6
# 
# # =============================================================================
# # WFO EXECUTION CONFIG
# # =============================================================================
# WFO_WINDOW_CONFIG = {
#     "15m":    {"train_months": 9, "test_months": 2},
#     "30m":    {"train_months": 9, "test_months": 2},
#     "1H":     {"train_months": 9, "test_months": 2},
#     "4H":     {"train_months": 9, "test_months": 2},
#     "6Hutc":  {"train_months": 9, "test_months": 2},
#     "12Hutc": {"train_months": 9, "test_months": 2},
#     "1Dutc":  {"train_months": 9, "test_months": 2},
# }
# 
# =============================================================================
# =============================================================================
# WFO APPROVAL THRESHOLDS
# =============================================================================
WFO_NET_GAIN_TH = 5
WFO_DD_TH       = 5
WFO_R2_TH       = 0.6
WFO_WFR_TH      = 0.5

# =============================================================================
# WFO EXECUTION CONFIG
# =============================================================================
WFO_WINDOW_CONFIG = {
    "15m":    {"train_months": 12, "test_months": 3},
    "30m":    {"train_months": 12, "test_months": 3},
    "1H":     {"train_months": 12, "test_months": 3},
    "4H":     {"train_months": 12, "test_months": 3},
    "6Hutc":  {"train_months": 12, "test_months": 3},
    "12Hutc": {"train_months": 12, "test_months": 3},
    "1Dutc":  {"train_months": 12, "test_months": 3},
}

ANCHORED    = False
METRIC_MODE = "NET_GAIN_PCT"   # "NET_GAIN_PCT" or "CALMAR"
EMA_ALPHA   = 0.3

# =============================================================================
# WFO PARALLELIZATION
# =============================================================================
RULES_N_JOBS = -1  # parallelizes across rules
INNER_N_JOBS = 1   # parallelizes the param grid search within each rule's window

# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _window_cache_key(base_arrays: dict) -> tuple:

    return tuple(
        (sym, int(arr["ts"][0]), int(arr["ts"][-1]), len(arr["ts"]))
        for sym, arr in sorted(base_arrays.items())
    )

def build_ohlcv_with_signal(
    base_arrays: dict,
    signal_fn: callable,
    signal_params_keys: list,
    param_dict: dict,
    _signal_cache: dict = None,
) -> dict:

    signal_independent_of_params = not signal_params_keys
    cache_key = None
    if signal_independent_of_params and _signal_cache is not None:
        cache_key = _window_cache_key(base_arrays)
        cached    = _signal_cache.get(cache_key)
        if cached is not None:
            return cached

    ohlcv_arrays = {}
    for sym, arr in base_arrays.items():
        sig_kwargs = {k: param_dict[k.upper()] for k in signal_params_keys if k.upper() in param_dict}
        signals    = signal_fn(arr, **sig_kwargs, live_trading=False)
        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=DTYPE)}

    if cache_key is not None:
        _signal_cache.clear()  # only one window's signals need to live at a time
        _signal_cache[cache_key] = ohlcv_arrays

    return ohlcv_arrays

def _get_prepared_data(
    base_arrays: dict,
    ohlcv_arrays: dict,
    _prepared_cache: dict = None,
):

    if _prepared_cache is None:
        return prepare_backtest_data(ohlcv_arrays)

    cache_key = _window_cache_key(base_arrays)
    cached    = _prepared_cache.get(cache_key)
    if cached is not None:
        return cached

    prepared = prepare_backtest_data(ohlcv_arrays)
    _prepared_cache.clear()  # only one window's prepared data needs to live at a time
    _prepared_cache[cache_key] = prepared
    return prepared

def compute_metric(results: dict) -> float:

    trade_log = results.get("__PORTFOLIO__", {}).get("trade_log")
    if trade_log is None or trade_log.empty:
        return 0.0

    m            = compute_metrics(trade_log, capital=INITIAL_BALANCE, name="", include_weekly=False, include_skew_kurtosis=False, include_r2=False)
    net_gain_pct = m["Net_Gain_pct"]

    if METRIC_MODE == "NET_GAIN_PCT":
        return net_gain_pct

    if METRIC_MODE == "CALMAR":
        max_dd_pct = abs(m["Max_DD_pct"])
        return net_gain_pct / max_dd_pct if max_dd_pct > 0 else net_gain_pct

    raise ValueError(f"Unknown METRIC_MODE: {METRIC_MODE}")

def _evaluate_fn(
    params: dict,
    base_arrays: dict,
    train_start_ts,
    train_edge_ts,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    _signal_cache: dict = None,
    _prepared_cache: dict = None,
) -> tuple:
    """Single param combination evaluation for one WFO train window."""
    ohlcv_arrays = build_ohlcv_with_signal(
        base_arrays, signal_fn, signal_params_keys, params, _signal_cache=_signal_cache
    )
    prepared_data = _get_prepared_data(base_arrays, ohlcv_arrays, _prepared_cache=_prepared_cache)
    results = run_backtest_from_prepared_light(
        prepared_data,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )

    trade_log = results["__PORTFOLIO__"]["trade_log"]
    if not trade_log.empty:
        truncated_mask = (
            trade_log["exit_reason"].isin(["SELL_AFTER", "END_OF_DATA"]) &
            (trade_log["buy_time"] >= pd.Timestamp(train_start_ts)) &
            (trade_log["buy_time"] > pd.Timestamp(train_edge_ts))
        )
        trade_log = trade_log[~truncated_mask]
        results   = {"__PORTFOLIO__": {"trade_log": trade_log}}

    return compute_metric(results), params


def _collect_trades_fn(
    params: dict,
    base_arrays: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    _prepared_cache: dict = None,
) -> pd.DataFrame:
    """Run backtest with best_params on a window and return the trade log."""
    ohlcv_arrays  = build_ohlcv_with_signal(base_arrays, signal_fn, signal_params_keys, params)
    prepared_data = _get_prepared_data(base_arrays, ohlcv_arrays, _prepared_cache=_prepared_cache)
    results       = run_backtest_from_prepared(
        prepared_data,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    trades             = results["__PORTFOLIO__"]["trade_log"].copy()
    trades.columns     = trades.columns.str.lower().str.strip()
    trades["buy_time"] = pd.to_datetime(trades["buy_time"])
    return trades

# =============================================================================
# APPROVAL CRITERION
# =============================================================================

def _evaluate_wfo_approval(
    train_net_gain_is_avg: float,
    test_net_gain_oos_avg: float,
    wfo_test_trades: pd.DataFrame,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    train_months: float,
    test_months: float,
) -> tuple:

    if wfo_test_trades.empty:
        return False, 0.0, 0.0, 0.0, None

    m            = compute_metrics(wfo_test_trades, capital=INITIAL_BALANCE, name="", include_weekly=False, include_skew_kurtosis=False)
    net_gain_pct = m["Net_Gain_pct"]
    max_dd_pct   = m["Max_DD_pct"]
    r_squared    = m["R_Squared"]

    monthly_test = test_net_gain_oos_avg / test_months
    monthly_is   = train_net_gain_is_avg / train_months if train_months else 0.0
    wfr          = monthly_test / monthly_is if monthly_is > 0 else 0.0

    approved     = (
        net_gain_pct >= net_gain_th
        and abs(max_dd_pct) <= dd_th
        and r_squared >= r2_th
        and wfr >= wfr_th
    )

    return approved, net_gain_pct, max_dd_pct, wfr, m

# =============================================================================
# RUN WFO IS
# =============================================================================
def run_wfo_is(
    ohlcv_arr: dict,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    timeframe: str,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    n_jobs: int = -1,
    show_progress: bool = False,
    collect_test_fn_override: callable = None,
) -> tuple:

    param_ranges = dict(zip(param_names, lists_for_grid))

    _wfo_cfg = WFO_WINDOW_CONFIG.get(timeframe)
    if _wfo_cfg is None:
        raise ValueError(f"No WFO window config for timeframe: {timeframe}")
    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(_wfo_cfg["train_months"] * bars_per_month)
    pct_train_set    = _wfo_cfg["train_months"] / (_wfo_cfg["train_months"] + _wfo_cfg["test_months"])

    _signal_cache   = {}
    _prepared_cache = {}

    evaluate_fn = partial(
        _evaluate_fn,
        signal_fn          = signal_fn,
        signal_params_keys = signal_params_keys,
        order_amount       = order_amount,
        _signal_cache      = _signal_cache,
        _prepared_cache    = _prepared_cache,
    )

    collect_train_fn = partial(
        _collect_trades_fn,
        signal_fn          = signal_fn,
        signal_params_keys = signal_params_keys,
        order_amount       = order_amount,
        _prepared_cache    = _prepared_cache,
    )

    collect_test_fn = collect_test_fn_override if collect_test_fn_override is not None else collect_train_fn

    best_params, df_results, wfo_train_trades, wfo_test_trades, n_windows, train_net_gain_is_avg, test_net_gain_oos_avg = walk_forward_optimization(
        ohlcv_arr               = ohlcv_arr,
        param_ranges            = param_ranges,
        length_train_set        = length_train_set,
        pct_train_set           = pct_train_set,
        anchored                = ANCHORED,
        evaluate_fn             = evaluate_fn,
        ema_alpha               = EMA_ALPHA,
        n_jobs                  = n_jobs,
        show_progress           = show_progress,
        collect_train_trades_fn = collect_train_fn,
        collect_test_trades_fn  = collect_test_fn,
    )

    logger.debug(
        f"STAGE 1 ── WFO completed  ── {n_windows} windows | "
        f"train={_wfo_cfg['train_months']}m  test={_wfo_cfg['test_months']}m"
    )

    has_nan_window = df_results["best_crite"].iloc[:-1].isna().any()

    if has_nan_window:
        approved_wfo, wfo_net_gain, wfo_max_dd, wfo_wfr, wfo_metrics = False, 0.0, 0.0, 0.0, None
        logger.debug("STAGE 1 ── WFO rejected — at least one window had no trades (NaN)")
    else:
        approved_wfo, wfo_net_gain, wfo_max_dd, wfo_wfr, wfo_metrics = _evaluate_wfo_approval(
            train_net_gain_is_avg = train_net_gain_is_avg,
            test_net_gain_oos_avg = test_net_gain_oos_avg,
            wfo_test_trades       = wfo_test_trades,
            net_gain_th           = net_gain_th,
            dd_th                 = dd_th,
            r2_th                 = r2_th,
            wfr_th                = wfr_th,
            train_months          = _wfo_cfg["train_months"],
            test_months           = _wfo_cfg["test_months"],
        )

    return (
        best_params, approved_wfo, wfo_net_gain, wfo_max_dd, wfo_test_trades, df_results, wfo_wfr, wfo_metrics,
    )
# =============================================================================
# PIPE WFO — one timeframe at a time, parallelized by rule
# =============================================================================
def _empty_wfo_fields() -> dict:
    """Placeholder WFO fields for rules that were never evaluated (pipe disabled)."""
    return {
        "approved":        False,
        "net_gain":        0.0,
        "max_dd":          0.0,
        "n_trades":        0,
        "n_windows":       0,
        "win_rate":        0.0,
        "profit_factor":   0.0,
        "calmar":          0.0,
        "r_squared":       0.0,
        "wfr":             0.0,
        "duration_d":      0.0,
        "best_params":     None,
        "wfo_test_trades": None,
    }

def _run_wfo_for_rule(
    idx: int,
    total: int,
    rule: dict,
    ohlcv_arr: dict,
    param_names: list,
    lists_for_grid: list,
    order_amount: int,
    timeframe: str,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    inner_n_jobs: int,
    show_progress: bool,
    log_level: int,
    save_trades: bool,
    brief_trades_folder: str,
) -> dict:
    """Runs WFO for a single rule; returns the rule dict merged with WFO result fields."""
    logging.basicConfig(level=log_level, format="%(message)s", force=True)
    logging.getLogger("joblib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    rule_id = rule["rule_id"]

    (
        best_params, approved_wfo, wfo_net_gain, wfo_max_dd, wfo_test_trades, df_results, wfo_wfr, metrics,
    ) = run_wfo_is(
        ohlcv_arr           = ohlcv_arr,
        param_names         = param_names,
        lists_for_grid      = lists_for_grid,
        signal_fn           = rule["signal_fn"],
        signal_params_keys  = [],
        order_amount        = order_amount,
        timeframe           = timeframe,
        net_gain_th         = net_gain_th,
        dd_th               = dd_th,
        r2_th               = r2_th,
        wfr_th              = wfr_th,
        n_jobs              = inner_n_jobs,
        show_progress       = show_progress,
    )
    n_windows = len(df_results) - 1 if df_results is not None else 0
    n_trades  = 0 if wfo_test_trades is None else len(wfo_test_trades)

    if save_trades and wfo_test_trades is not None and not wfo_test_trades.empty:
        os.makedirs(brief_trades_folder, exist_ok=True)
        wfo_test_trades.to_csv(
            os.path.join(brief_trades_folder, f"trades_wfo_test_{rule_id}.csv"),
            index=False,
        )

    metrics = None
    if wfo_test_trades is not None and not wfo_test_trades.empty:
        metrics = compute_metrics(wfo_test_trades, capital=INITIAL_BALANCE, name="", include_weekly=False, include_skew_kurtosis=False)

    logger.debug(f"[{idx + 1}/{total}] {rule['side']:<5} {rule['label']} -> "
                 f"{'PASS' if approved_wfo else 'FAIL'} NetGain={wfo_net_gain:.1f}% DD={wfo_max_dd:.1f}%")

    return {
        **rule,
        "approved":        approved_wfo,
        "net_gain":        wfo_net_gain,
        "max_dd":          wfo_max_dd,
        "n_trades":        n_trades,
        "n_windows":       n_windows,
        "win_rate":        metrics["Win_Rate"]      if metrics else 0.0,
        "profit_factor":   metrics["Profit_Factor"] if metrics else 0.0,
        "calmar":          metrics["Calmar"]        if metrics else 0.0,
        "r_squared":       metrics["R_Squared"]     if metrics else 0.0,
        "wfr":             wfo_wfr,
        "duration_d":      metrics["Duration_d"]    if metrics else 0.0,
        "best_params":     best_params,
        "wfo_test_trades": wfo_test_trades,
    }

def pipe_wfo(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    timeframe: str,
    combo_key: str,
    net_gain_th: float = WFO_NET_GAIN_TH,
    dd_th: float = WFO_DD_TH,
    r2_th: float = WFO_R2_TH,
    wfr_th: float = WFO_WFR_TH,
    enabled: bool = True,
    rules_n_jobs: int = RULES_N_JOBS,
    inner_n_jobs: int = INNER_N_JOBS,
    show_progress: bool = False,
    log_level: int = logging.INFO,
    save_trades: bool = False,
    brief_trades_folder: str = None,
) -> list:
    if not enabled:
        logger.info(f"WFO ── {combo_key} ── disabled — passing all {len(rules)} rules through untouched")
        return [{**r, **_empty_wfo_fields()} for r in rules]

    param_names    = list(param_grid.keys())
    lists_for_grid = [param_grid[k] for k in param_names]
    total          = len(rules)

    results = list(tqdm(
        Parallel(n_jobs=rules_n_jobs, return_as="generator")(
            delayed(_run_wfo_for_rule)(
                i, total, rule, ohlcv_arr, param_names, lists_for_grid, order_amount,
                timeframe, net_gain_th, dd_th, r2_th, wfr_th, inner_n_jobs, show_progress,
                log_level, save_trades, brief_trades_folder,
            )
            for i, rule in enumerate(rules)
        ),
        desc=f"WFO LOOP       {combo_key}".ljust(12),
        total=total,
        dynamic_ncols=True,
    ))

    return results