# core/pipeline/multiverse.py OLD
import os
import sys
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from pipeline.wfo import run_wfo_is,block_sell_after_grid
from utils.plotting import plot_multiverse_synthetic_vs_historical
from utils.reporting import report_multiverse_debug
logger = logging.getLogger("BOT_batch.pipeline.multiverse")

# =============================================================================
# MCPT EXECUTION CONFIG
# =============================================================================
MULTIVERSE_PVALUE_TH    = 0.1

# =============================================================================
# BLOCK_SIZE_BY_TIMEFRAME = {
#     "1H":     150,
#     "4H":     120,
#     "6Hutc":  70,
#     "12Hutc": 30,
# }
# =============================================================================

BLOCK_SIZE_BY_TIMEFRAME = {
    "1H":     400,
    "4H":     100,
}

N_PERMUTATIONS = 1000
MCPT_N_JOBS    = -1
MCPT_BASE_SEED = 42

# Paths regenerated on demand for the debug plot only — independent of n_paths
MCPT_PLOT_N_PATHS = 100


# =============================================================================
# SAMPLING METHOD — disjoint block permutation (Masters, 2020)
# =============================================================================
_WFO_NEUTRAL_TH = {
    "net_gain_th": float("-inf"),
    "dd_th":       float("inf"),
    "r2_th":       float("-inf"),
    "wfr_th":      float("-inf"),
}

_WORKER_SILENCED_LOGGERS = (
    "BOT_batch.pipeline.wfo",
    "BOT_batch.engines.wfo_WF",
    "joblib",
    "matplotlib",
    "numba",
)

# Feature column layout of the permuted data array.
_COL_LOG_RET_CLOSE  = 0
_COL_LOG_OPEN_LOW   = 1
_COL_LOG_OPEN_HIGH  = 2
_COL_LOG_OPEN_CLOSE = 3
_COL_VAR_LOW_TIME   = 4
_COL_VAR_HIGH_TIME  = 5
_N_BASE_FEATURES    = 6

# =============================================================================
# CONFIG RESOLVERS — explicit failure, no silent fallbacks
# =============================================================================
def _resolve_block_size(timeframe: str) -> int:
    block_size = BLOCK_SIZE_BY_TIMEFRAME.get(timeframe)
    if block_size is None:
        raise ValueError(
            f"MULTIVERSE ── timeframe={timeframe!r} has no entry in "
            f"BLOCK_SIZE_BY_TIMEFRAME — add it explicitly, no fallback is defined."
        )
    return block_size
# =============================================================================
# LOG FEATURE EXTRACTION — done once per symbol, reused by every path
# =============================================================================
def _compute_log_features(df_hist: pd.DataFrame, raw_columns: list) -> tuple:

    close = df_hist["close"].to_numpy(np.float64)
    open_ = df_hist["open"].to_numpy(np.float64)
    high  = df_hist["high"].to_numpy(np.float64)
    low   = df_hist["low"].to_numpy(np.float64)

    prev_close     = np.empty_like(close)
    prev_close[0]  = open_[0]
    prev_close[1:] = close[:-1]

    ts_index = df_hist.index.to_numpy(dtype="datetime64[ns]")
    ts_sec   = ts_index.astype("datetime64[s]").astype(np.int64)
    low_sec  = pd.to_datetime(df_hist["low_time"]).to_numpy(dtype="datetime64[s]").astype(np.int64)
    high_sec = pd.to_datetime(df_hist["high_time"]).to_numpy(dtype="datetime64[s]").astype(np.int64)

    columns = [
        np.log(close / prev_close),
        np.log(low   / open_),
        np.log(high  / open_),
        np.log(close / open_),
        (low_sec  - ts_sec).astype(np.float64),
        (high_sec - ts_sec).astype(np.float64),
    ]
    for raw_col in raw_columns:
        columns.append(df_hist[raw_col].to_numpy(np.float64))

    data_array = np.column_stack(columns)
    return data_array, float(open_[0]), ts_index

def _build_path_bundle(ohlcv_data: dict, raw_columns: list, timeframe: str) -> dict:

    ref_ts = np.unique(np.concatenate([
        df_hist.index.to_numpy(dtype="datetime64[ns]") for df_hist in ohlcv_data.values()
    ]))
    n_ref_rows = len(ref_ts)
    ref_symbol = max(ohlcv_data.keys(), key=lambda k: len(ohlcv_data[k].index))

    symbols = {}
    for symbol, df_hist in ohlcv_data.items():
        data_array, start_price, ts_index = _compute_log_features(df_hist, raw_columns)

        global_pos = np.searchsorted(ref_ts, ts_index, side="left")

        if global_pos.size > 1 and np.any(np.diff(global_pos) <= 0):
            raise ValueError(
                f"MULTIVERSE ── {timeframe} ── symbol={symbol!r} has duplicated or "
                f"non-monotonic timestamps: the index must be strictly increasing."
            )

        symbols[symbol] = {
            "data_array":  data_array,
            "start_price": start_price,
            "ts_index":    ts_index,
            "global_pos":  global_pos,
        }

    logger.debug(
        f"MULTIVERSE ── {timeframe} ── union grid {n_ref_rows} bars ── "
        f"per-symbol bars {min(len(s['global_pos']) for s in symbols.values())}"
        f"..{max(len(s['global_pos']) for s in symbols.values())}"
    )

    return {
        "symbols":    symbols,
        "ref_symbol": ref_symbol,
        "n_ref_rows": n_ref_rows,
        "n_raw":      len(raw_columns),
    }

# =============================================================================
# BLOCK LAYOUT — drawn once per path, shared by every symbol
# =============================================================================
def _draw_block_layout(rng: np.random.Generator, n_ref_rows: int, block_size: int) -> tuple:

    if block_size <= 1:
        edges = np.arange(n_ref_rows + 1, dtype=np.int64)
        return edges, rng.permutation(n_ref_rows), 0

    phase = int(rng.integers(0, block_size))

    interior = np.arange(phase, n_ref_rows, block_size, dtype=np.int64)
    if phase > 0:
        interior = np.concatenate((np.zeros(1, dtype=np.int64), interior))
    edges = np.concatenate((interior, np.array([n_ref_rows], dtype=np.int64)))

    order = rng.permutation(len(edges) - 1)
    return edges, order, phase

def _gather_index_permutation(edges: np.ndarray, order: np.ndarray, global_pos: np.ndarray) -> np.ndarray:

    local_starts = np.searchsorted(global_pos, edges[:-1], side="left")
    local_ends   = np.searchsorted(global_pos, edges[1:],  side="left")

    starts  = local_starts[order]
    lengths = local_ends[order] - starts

    keep = lengths > 0
    starts, lengths = starts[keep], lengths[keep]

    total   = int(lengths.sum())
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)[:-1]))
    return np.repeat(starts - offsets, lengths) + np.arange(total, dtype=np.int64)

# =============================================================================
# PATH RECONSTRUCTION
# =============================================================================
def _reconstruct_symbol_path(symbol_bundle: dict, gather_index: np.ndarray, n_raw: int) -> dict:

    sampled  = symbol_bundle["data_array"][gather_index]
    ts_index = symbol_bundle["ts_index"]

    close_prices = symbol_bundle["start_price"] * np.exp(np.cumsum(sampled[:, _COL_LOG_RET_CLOSE]))
    open_prices  = close_prices * np.exp(-sampled[:, _COL_LOG_OPEN_CLOSE])
    low_prices   = open_prices  * np.exp(sampled[:, _COL_LOG_OPEN_LOW])
    high_prices  = open_prices  * np.exp(sampled[:, _COL_LOG_OPEN_HIGH])

    ts_seconds = ts_index.astype("datetime64[s]").astype(np.int64)
    low_time   = (ts_seconds + sampled[:, _COL_VAR_LOW_TIME].astype(np.int64)).astype("datetime64[s]")
    high_time  = (ts_seconds + sampled[:, _COL_VAR_HIGH_TIME].astype(np.int64)).astype("datetime64[s]")

    volume = (
        sampled[:, _N_BASE_FEATURES]
        if n_raw > 0
        else np.zeros(len(gather_index), dtype=np.float64)
    )

    return {
        "ts":        ts_index,
        "open":      open_prices,
        "high":      high_prices,
        "low":       low_prices,
        "close":     close_prices,
        "volume":    volume,
        "low_time":  low_time.astype("datetime64[ns]"),
        "high_time": high_time.astype("datetime64[ns]"),
    }   

def _build_synthetic_ohlcv_arr(
    bundle: dict,
    path_idx: int,
    block_size: int,
    base_seed: int,
) -> tuple:
    """Generate one full synthetic universe, reproducibly, from its path seed."""
    rng        = np.random.default_rng(base_seed + path_idx)
    n_ref_rows = bundle["n_ref_rows"]
    n_raw      = bundle["n_raw"]

    edges, order, phase = _draw_block_layout(rng, n_ref_rows, block_size)
    layout_info = {"phase": phase, "n_blocks": len(edges) - 1, "order_head": order[:5].tolist()}

    ohlcv_arr = {
        symbol: _reconstruct_symbol_path(
            symbol_bundle,
            _gather_index_permutation(edges, order, symbol_bundle["global_pos"]),
            n_raw,
        )
        for symbol, symbol_bundle in bundle["symbols"].items()
    }
    return ohlcv_arr, layout_info

# =============================================================================
# PATH EVALUATION — the full real WFO procedure, run on synthetic prices
# =============================================================================
def _evaluate_rules_on_path(
    ohlcv_arr: dict,
    rules: list,
    param_names: list,
    lists_by_id: dict,
    order_amount: int,
    timeframe: str,
    return_schedules: bool = False,
) -> dict:
    results = {}
    for rule in rules:
        try:
            (
                _best_params, _approved, _net_gain, _max_dd, wfo_test_trades, df_results, _wfr, _metrics,
            ) = run_wfo_is(
                ohlcv_arr          = ohlcv_arr,
                param_names        = param_names,
                lists_for_grid     = lists_by_id[rule["rule_id"]],
                signal_fn          = rule["signal_fn"],
                signal_params_keys = [],
                order_amount       = order_amount,
                timeframe          = timeframe,
                n_jobs             = 1,
                show_progress      = False,
                **_WFO_NEUTRAL_TH,
            )
        except Exception as exc:
            logger.debug(f"MULTIVERSE ── {timeframe} ── rule={rule['rule_id']} WFO failed on path: {exc}")
            results[rule["rule_id"]] = (0.0, None)
            continue
        
        profit = (
            0.0
            if wfo_test_trades is None or wfo_test_trades.empty
            else float(wfo_test_trades["profit"].sum())
        )
        schedule = None
        if return_schedules and df_results is not None and len(df_results) > 1:
            schedule = df_results.iloc[:-1][param_names].to_dict("records")
        results[rule["rule_id"]] = (profit, schedule)

    return results

def _evaluate_path_chunk(
    path_indices: list,
    bundle: dict,
    rules: list,
    param_names: list,
    lists_by_id: dict,
    order_amount: int,
    timeframe: str,
    block_size: int,
    base_seed: int,
) -> list:

    for logger_name in _WORKER_SILENCED_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    chunk_results = []
    for path_idx in path_indices:
        ohlcv_arr, _layout = _build_synthetic_ohlcv_arr(
            bundle, path_idx, block_size, base_seed,
        )
        path_results = _evaluate_rules_on_path(
            ohlcv_arr, rules, param_names, lists_by_id, order_amount, timeframe,
        )
        chunk_results.append({rid: profit for rid, (profit, _sched) in path_results.items()})
    return chunk_results

# =============================================================================
# APPROVAL CRITERION — Monte Carlo Permutation Test p-value
# =============================================================================
def _compute_p_value(real_profit: float, permuted_profits: np.ndarray) -> float:

    n_matching_or_beating = int(np.sum(permuted_profits >= real_profit))
    return (1 + n_matching_or_beating) / (1 + len(permuted_profits))

# =============================================================================
# PLOTTING — paths regenerated on demand, never materialized for all n_paths
# =============================================================================
_PLOT_CLOSE_COL_IDX = 3


def _build_paths_for_plotting(
    bundle: dict,
    n_plot_paths: int,
    block_size: int,
    base_seed: int,
) -> dict:
    paths_by_symbol = {
        symbol: np.zeros((n_plot_paths, len(symbol_bundle["ts_index"]), _PLOT_CLOSE_COL_IDX + 1), dtype=np.float32)
        for symbol, symbol_bundle in bundle["symbols"].items()
    }

    for path_idx in range(n_plot_paths):
        ohlcv_arr, _layout = _build_synthetic_ohlcv_arr(bundle, path_idx, block_size, base_seed)
        for symbol, arr in ohlcv_arr.items():
            paths_by_symbol[symbol][path_idx, :, _PLOT_CLOSE_COL_IDX] = arr["close"]

    return paths_by_symbol

# =============================================================================
# CORE MULTIVERSE EVALUATION (one timeframe, every rule of that timeframe)
# =============================================================================
def _evaluate_multiverse_batch(
    ohlcv_data: dict,
    rules: list,
    param_grid: dict,
    order_amount: int,
    p_value_th: float,
    block_size: int,
    n_paths: int,
    n_jobs: int,
    base_seed: int,
    timeframe: str,
    combo_key: str,
    show_plots: bool = False,
) -> tuple:
    if not ohlcv_data or not rules:
        return (
            {r["rule_id"]: 1.0   for r in rules},
            {r["rule_id"]: False for r in rules},
        )

    param_names = list(param_grid.keys())
    lists_by_id = {}
    for r in rules:
        grid = block_sell_after_grid(param_grid, r.get("best_combo_id"))
        lists_by_id[r["rule_id"]] = [grid[k] for k in param_names]

    bundle = _build_path_bundle(ohlcv_data, raw_columns=["volume"], timeframe=timeframe)

    logger.info(
        f"MULTIVERSE      {combo_key}: {len(rules)} rules ── {n_paths} paths ── "
        f"block_size={block_size}"
    )

    if show_plots:
        plot_paths = _build_paths_for_plotting(
            bundle, min(MCPT_PLOT_N_PATHS, n_paths), block_size, base_seed,
        )
        plot_multiverse_synthetic_vs_historical(ohlcv_data, plot_paths)

    n_workers   = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    n_chunks    = min(n_paths, n_workers * 2)
    path_chunks = [chunk for chunk in np.array_split(np.arange(n_paths), n_chunks) if len(chunk) > 0]

    chunked_results = list(tqdm(
        Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_evaluate_path_chunk)(
                chunk.tolist(), bundle, rules, param_names, lists_by_id,
                order_amount, timeframe, block_size, base_seed,
            )
            for chunk in path_chunks
        ),
        desc=f"MULTIVERSE      {combo_key}".strip().ljust(18),
        total=len(path_chunks),
        dynamic_ncols=True,
        file=sys.stdout,
    ))
    sys.stdout.flush()
    per_path_results = [res for chunk_res in chunked_results for res in chunk_res]

    profits_by_id  = {
        r["rule_id"]: np.array([res[r["rule_id"]] for res in per_path_results], dtype=np.float64)
        for r in rules
    }

    p_value_by_id, approved_by_id = {}, {}
    for rule in rules:
        rid   = rule["rule_id"]
        nulls = profits_by_id[rid]

        if nulls.size == 0:
            p_value_by_id[rid], approved_by_id[rid] = 1.0, False
            continue

        real_profit          = float(rule["wfo_test_trades"]["profit"].sum())
        p_value              = _compute_p_value(real_profit, nulls)
        p_value_by_id[rid]   = p_value
        approved_by_id[rid]  = p_value <= p_value_th

    if logger.isEnabledFor(logging.DEBUG):
        probe_arr, probe_layout = _build_synthetic_ohlcv_arr(bundle, 0, block_size, base_seed)
        probe_profit, probe_schedule = _evaluate_rules_on_path(
            probe_arr, rules[:1], param_names, lists_by_id, order_amount, timeframe,
            return_schedules=True,
        )[rules[0]["rule_id"]]

        report_multiverse_debug(
            ohlcv_data     = ohlcv_data,
            synthetic_arr  = probe_arr,
            layout_info    = probe_layout,
            ref_symbol     = bundle["ref_symbol"],
            n_ref_rows     = bundle["n_ref_rows"],
            probe_rule     = rules[0],
            probe_schedule = probe_schedule,
            probe_profit   = probe_profit,
            rules          = rules,
            param_names    = param_names,
            profits_by_id  = profits_by_id,
            p_value_by_id  = p_value_by_id,
            approved_by_id = approved_by_id,
            n_paths        = n_paths,
            block_size     = block_size,
            timeframe      = timeframe,
        )

    return p_value_by_id, approved_by_id

# =============================================================================
# PIPE MULTIVERSE
# =============================================================================
def pipe_multiverse(
    rules: list,
    ohlcv_data_by_combo: dict,
    param_grid: dict,   # dict keyed by timeframe: {"1H": {...}, "4H": {...}}
    order_amount: int,
    p_value_th: float = None,
    n_paths: int = N_PERMUTATIONS,
    block_size: int | None = None,
    n_jobs: int = MCPT_N_JOBS,
    base_seed: int = MCPT_BASE_SEED,
    show_plots: bool = False,
) -> list:
    p_value_th = p_value_th if p_value_th is not None else MULTIVERSE_PVALUE_TH

    evaluable_rules = [
        r for r in rules
        if r.get("wfo_test_trades") is not None and not r["wfo_test_trades"].empty
    ]
    unevaluable_ids = {r["rule_id"] for r in rules} - {r["rule_id"] for r in evaluable_rules}
    if unevaluable_ids:
        logger.warning(
            f"MULTIVERSE ── {len(unevaluable_ids)} rules have no WFO test trades — rejected"
        )

    rules_by_combo: dict = {}
    for rule in evaluable_rules:
        rules_by_combo.setdefault(rule["combo_key"], []).append(rule)

    p_value_by_id:  dict = {rid: 1.0   for rid in unevaluable_ids}
    approved_by_id: dict = {rid: False for rid in unevaluable_ids}

    for combo_key, combo_rules in rules_by_combo.items():
        timeframe           = combo_rules[0]["timeframe"]
        resolved_block_size = block_size if block_size is not None else _resolve_block_size(timeframe)

        combo_p_values, combo_approved = _evaluate_multiverse_batch(
            ohlcv_data    = ohlcv_data_by_combo[combo_key],
            rules         = combo_rules,
            param_grid    = param_grid[timeframe],
            order_amount  = order_amount,
            p_value_th    = p_value_th,
            block_size    = resolved_block_size,
            n_paths       = n_paths,
            n_jobs        = n_jobs,
            base_seed     = base_seed,
            timeframe     = timeframe,
            combo_key     = combo_key,
            show_plots    = show_plots,
        )
        p_value_by_id.update(combo_p_values)
        approved_by_id.update(combo_approved)

    results = [
        {
            **r,
            "passed_multiverse":  approved_by_id[r["rule_id"]],
            "multiverse_p_value": p_value_by_id[r["rule_id"]],
        }
        for r in rules
    ]

    n_passed = sum(1 for r in results if r["passed_multiverse"])
    logger.info(f"MULTIVERSE ── {n_passed}/{len(results)} rules pass (p <= {p_value_th})")

    return results
