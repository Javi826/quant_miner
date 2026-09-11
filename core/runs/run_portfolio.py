# core/runs/run_portfolio.py
import logging
import time
import numpy as np
import pandas as pd
from itertools import combinations
from utils.reporting import print_best_wfo_portfolio
from utils.plotting import plot_best_wfo_portfolio
logger = logging.getLogger("BOT_batch.runs.run_best_wfo_portfolio")

# =============================================================================
# CONFIGURATION
# =============================================================================
#NET_GAIN_PCT | CALMAR | R_SQUARED | MAX_DD_PCT — see _FAST_METRIC_MAP below
WFO_METRIC       = "R_SQUARED"
WFO_SPLIT_MONTHS = 2

def _generate_subperiod_weights(n_splits: int) -> list:

    base        = 1.0 / n_splits
    last_weight = 2.0 * base
    extra       = last_weight - base

    n_rest      = n_splits - 1
    rest_weight = base - extra / n_rest

    weights = [rest_weight] * n_rest + [last_weight]
    return [round(w, 6) for w in weights]

MIN_STRATEGIES        = 5
MAX_STRATEGIES        = 8
MAX_TOTAL_STRATEGIES  = 25
TOP_N                 = 3

REQUIRE_SUBPERIODS_POSITIVE = True
REQUIRE_LONG_SHORT          = True
REQUIRE_ALL_TIMEFRAMES      = True
# =============================================================================
# PRIVATE HELPERS — Validation
# =============================================================================

def _validate_config(metric: str) -> None:
    if metric not in _FAST_METRIC_MAP:
        raise ValueError(
            f"Unknown WFO_METRIC='{metric}'. "
            f"Valid options: {list(_FAST_METRIC_MAP.keys())}"
        )
# =============================================================================
# PRIVATE HELPERS — Identity
# =============================================================================

def _is_long(strategy_id: str) -> bool:
    return "_long_" in strategy_id

def _is_short(strategy_id: str) -> bool:
    return "_short_" in strategy_id

def _get_timeframe(strategy_id: str) -> str:
    return strategy_id.split("_")[1]

def _compute_net_gain_pct(trades_df: pd.DataFrame, initial_balance: float) -> float:
    if trades_df is None or trades_df.empty:
        return -np.inf
    return trades_df["profit"].sum() / initial_balance * 100

def _cap_strategies_round_robin(
    validated_wfo_trades: list,
    initial_balance: float,
    max_total_strategies: int,
) -> list:
    """Cap the strategy pool to max_total_strategies using round-robin
    selection across (timeframe, side) groups, best net_gain first."""
    trades_by_id = {sid: df for sid, df in validated_wfo_trades}

    if len(trades_by_id) <= max_total_strategies:
        return validated_wfo_trades

    groups = {}
    for sid in trades_by_id:
        key = (_get_timeframe(sid), "long" if _is_long(sid) else "short")
        groups.setdefault(key, []).append(sid)

    for key in groups:
        groups[key].sort(
            key=lambda sid: _compute_net_gain_pct(trades_by_id[sid], initial_balance),
            reverse=True,
        )

    selected   = []
    group_keys = list(groups.keys())
    cursor     = {key: 0 for key in group_keys}

    while len(selected) < max_total_strategies:
        added_this_round = False
        for key in group_keys:
            if cursor[key] < len(groups[key]):
                selected.append(groups[key][cursor[key]])
                cursor[key] += 1
                added_this_round = True
                if len(selected) >= max_total_strategies:
                    break
        if not added_this_round:
            break

    logger.info(
        f"  Strategy pool capped: {len(trades_by_id)} -> {len(selected)} "
        f"(max_total_strategies={max_total_strategies}, round-robin by net_gain)"
    )

    return [(sid, trades_by_id[sid]) for sid in selected]

# =============================================================================
# PRIVATE HELPERS — Splitting
# =============================================================================
AVG_DAYS_PER_MONTH = 30.4368  # 365.2425 / 12 — used only to decide where the

def _split_trades_by_time(
    trades_list: list,
    split_months: int,
) -> list:
    if not trades_list:
        return []
    all_times = pd.concat([df["sell_time"] for _, df in trades_list], ignore_index=True)
    t_min     = pd.Timestamp(all_times.min())
    t_max     = pd.Timestamp(all_times.max())

    min_split_len = pd.Timedelta(days=split_months * AVG_DAYS_PER_MONTH)

    # Walk backward from t_max in fixed calendar-month steps. Any leftover
    boundaries = [t_max]
    cursor     = t_max
    while True:
        next_cursor = cursor - pd.DateOffset(months=split_months)
        if (next_cursor - t_min) < min_split_len:
            break
        boundaries.append(next_cursor)
        cursor = next_cursor
    boundaries.append(t_min)
    boundaries = boundaries[::-1]  # oldest -> newest

    n_splits = len(boundaries) - 1
    result   = []
    for i in range(n_splits):
        is_last = (i == n_splits - 1)
        t_start = boundaries[i]
        t_end   = boundaries[i + 1]
        label   = f"S{i + 1}"

        subset = []
        for sid, df in trades_list:
            if is_last:
                df_split = df[(df["sell_time"] >= t_start) & (df["sell_time"] <= t_end)]
            else:
                df_split = df[(df["sell_time"] >= t_start) & (df["sell_time"] < t_end)]
            subset.append((sid, df_split))
        subset = [(sid, df) for sid, df in subset if len(df) > 0]

        if subset:
            result.append((label, t_start, t_end, subset))

    return result
# =============================================================================
# PRIVATE HELPERS — Vectorized combo scoring (matrix ops, no per-combo loop)
# =============================================================================

_FAST_METRIC_MAP = {
    "NET_GAIN_PCT": (0, True),
    "MAX_DD_PCT":   (1, False),
    "R_SQUARED":    (2, True),
    "CALMAR":       (3, True),
}

def _precompute_subperiod_matrices(subperiods: list, all_ids: list) -> list:

    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    subperiod_matrices = []

    for label, _t_start, _t_end, split_trades in subperiods:
        series_by_id = {}
        for sid, df in split_trades:
            tl          = df.copy()
            tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
            series_by_id[sid] = tl.groupby("_date")["profit"].sum()

        if not series_by_id:
            subperiod_matrices.append((label, None))
            continue

        all_dates  = pd.concat(series_by_id.values()).index
        date_range = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq="1D")
        n_days     = len(date_range)

        daily_matrix = np.zeros((len(all_ids), n_days), dtype=np.float64)
        for sid, series in series_by_id.items():
            row_idx = id_to_idx[sid]
            daily_matrix[row_idx, :] = series.reindex(date_range, fill_value=0.0).to_numpy()

        subperiod_matrices.append((label, daily_matrix))

    return subperiod_matrices

def _r_squared_windowed(equity: np.ndarray, profit_matrix: np.ndarray) -> np.ndarray:

    n_combos, n_days = equity.shape
    x = np.arange(n_days, dtype=np.float64)

    nonzero_mask = profit_matrix != 0
    has_any      = nonzero_mask.any(axis=1)

    first_idx = np.argmax(nonzero_mask, axis=1)
    last_idx  = n_days - 1 - np.argmax(nonzero_mask[:, ::-1], axis=1)

    # Combos with no nonzero profit day at all in this subperiod: fall back
    first_idx = np.where(has_any, first_idx, 0)
    last_idx  = np.where(has_any, last_idx, n_days - 1)

    zeros_col = np.zeros((n_combos, 1), dtype=np.float64)
    prefix_y  = np.concatenate([zeros_col, np.cumsum(equity, axis=1)], axis=1)
    prefix_y2 = np.concatenate([zeros_col, np.cumsum(equity ** 2, axis=1)], axis=1)
    prefix_xy = np.concatenate([zeros_col, np.cumsum(equity * x[None, :], axis=1)], axis=1)

    rows      = np.arange(n_combos)
    sum_y     = prefix_y[rows, last_idx + 1]  - prefix_y[rows, first_idx]
    sum_y2    = prefix_y2[rows, last_idx + 1] - prefix_y2[rows, first_idx]
    sum_xy    = prefix_xy[rows, last_idx + 1] - prefix_xy[rows, first_idx]

    a = first_idx.astype(np.float64)
    b = last_idx.astype(np.float64)
    n = b - a + 1.0

    # Closed-form sum of i and i² over the integer window [a, b] (0-indexed),
    sum_x  = (a + b) * n / 2.0
    sum_x2 = (b * (b + 1.0) * (2.0 * b + 1.0) - (a - 1.0) * a * (2.0 * a - 1.0)) / 6.0

    cov_xy  = n * sum_xy - sum_x * sum_y
    var_x_n = n * sum_x2 - sum_x ** 2
    var_y_n = n * sum_y2 - sum_y ** 2

    denom     = var_x_n * var_y_n
    r2        = np.full(n_combos, np.nan)
    valid     = denom > 0
    r2[valid] = (cov_xy[valid] ** 2) / denom[valid]
    
    return r2

def _score_all_combos_vectorized(
    combos: list,
    subperiods: list,
    all_ids: list,
    initial_balance: float,
    metric: str,
) -> list:

    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    n_combos  = len(combos)
    n_ids     = len(all_ids)

    combo_matrix = np.zeros((n_combos, n_ids), dtype=np.float64)
    for i, combo in enumerate(combos):
        idxs = [id_to_idx[sid] for sid in combo if sid in id_to_idx]
        combo_matrix[i, idxs] = 1.0

    subperiod_matrices = _precompute_subperiod_matrices(subperiods, all_ids)
    col_idx, higher_is_better = _FAST_METRIC_MAP[metric]

    scores_by_label = {}
    for label, daily_matrix in subperiod_matrices:
        if daily_matrix is None:
            scores_by_label[label] = np.full(n_combos, np.nan)
            continue

        presence_mask         = (daily_matrix.any(axis=1)).astype(np.float64)  # (n_ids,)
        effective_membership  = combo_matrix * presence_mask                   # (n_combos, n_ids)
        counts_per_combo      = effective_membership.sum(axis=1)               # (n_combos,)

        profit_matrix = effective_membership @ daily_matrix                    # (n_combos, n_days)

        valid_combo = counts_per_combo > 0
        capital     = np.where(valid_combo, initial_balance * counts_per_combo, np.nan)

        equity      = capital[:, None] + np.cumsum(profit_matrix, axis=1)
        running_max = np.maximum.accumulate(equity, axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            max_dd   = ((equity - running_max) / running_max * 100).min(axis=1)
            net_gain = (equity[:, -1] - capital) / capital * 100
            calmar   = np.where(max_dd < 0, net_gain / np.abs(max_dd), np.nan)

        r2 = _r_squared_windowed(equity, profit_matrix)

        stacked = np.column_stack([net_gain, max_dd, r2, calmar])
        val     = stacked[:, col_idx]
        val     = val if higher_is_better else -np.abs(val)
        val     = np.where((net_gain <= 0) | ~valid_combo, np.nan, val)

        scores_by_label[label] = np.round(val, 3)

    raw_scores = []
    for i, combo in enumerate(combos):
        entry = {"combo": combo}
        for label, arr in scores_by_label.items():
            v            = arr[i]
            entry[label] = float(v) if np.isfinite(v) else np.nan
        raw_scores.append(entry)

    return raw_scores

# =============================================================================
# PRIVATE HELPERS — Rank-based scoring across subperiods
# =============================================================================
def _rank_combos_by_subperiod(
    raw_scores: list,
    subperiods: list,
) -> list:

    split_labels = [label for label, _, _, _ in subperiods]

    for label in split_labels:
        values      = [(i, r[label]) for i, r in enumerate(raw_scores)]
        n_combos    = len(values)
        worst_rank  = n_combos

        valid = [(i, v) for i, v in values if not np.isnan(v)]
        valid.sort(key=lambda x: x[1], reverse=True)  # higher metric value = better

        for rank, (i, _) in enumerate(valid, start=1):
            raw_scores[i][f"{label}_rank"] = rank

        ranked_indices = {i for i, _ in valid}
        for i, v in values:
            if i not in ranked_indices:
                raw_scores[i][f"{label}_rank"] = worst_rank

    return raw_scores

def _weighted_rank_score(
    entry: dict,
    subperiods: list,
    weights: list,
) -> float:
    """Weighted average of per-subperiod ranks (lower = better)."""
    split_labels = [label for label, _, _, _ in subperiods]
    return sum(entry[f"{label}_rank"] * w for label, w in zip(split_labels, weights))

def _valid_subsets_for_timeframe(longs: list, shorts: list, size: int):

    pool = longs + shorts
    if len(pool) < size:
        return

    if longs and shorts:
        min_longs = max(1, size - len(shorts))
        max_longs = min(size - 1, len(longs))
        for n_longs in range(min_longs, max_longs + 1):
            n_shorts = size - n_longs
            for long_combo in combinations(longs, n_longs):
                for short_combo in combinations(shorts, n_shorts):
                    yield long_combo + short_combo
    else:
        for combo in combinations(pool, size):
            yield combo


def _generate_combos(
    all_ids: list,
    min_strategies: int,
    max_strategies: int,
    require_long_short: bool,
    require_all_timeframes: bool = False,
) -> list:

    if not require_long_short:
        combos = [
            combo
            for size in range(min_strategies, max_strategies + 1)
            for combo in combinations(all_ids, size)
        ]
        if require_all_timeframes:
            required_tfs = {_get_timeframe(s) for s in all_ids}
            combos = [
                combo for combo in combos
                if required_tfs.issubset({_get_timeframe(s) for s in combo})
            ]
        return combos

    timeframe_groups = {}
    for sid in all_ids:
        tf   = _get_timeframe(sid)
        side = "long" if _is_long(sid) else "short"
        timeframe_groups.setdefault(tf, {"long": [], "short": []})[side].append(sid)

    tf_list = list(timeframe_groups.keys())
    n_tfs   = len(tf_list)
    min_size_per_tf = {
        tf: (2 if groups["long"] and groups["short"] else 1)
        for tf, groups in timeframe_groups.items()
    }

    combos = []

    def backtrack(idx: int, current_combo: tuple, current_size: int) -> None:
        if idx == n_tfs:
            if min_strategies <= current_size <= max_strategies:
                combos.append(current_combo)
            return

        tf     = tf_list[idx]
        groups = timeframe_groups[tf]
        remaining_min = sum(min_size_per_tf[t] for t in tf_list[idx + 1:])

        min_size_here = min_size_per_tf[tf]
        max_size_here = max_strategies - current_size - remaining_min

        for size in range(min_size_here, max_size_here + 1):
            for subset in _valid_subsets_for_timeframe(groups["long"], groups["short"], size):
                backtrack(idx + 1, current_combo + subset, current_size + size)

    backtrack(0, tuple(), 0)
    return combos

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def find_best_portfolio_combination_wfo(
    validated_wfo_trades: list,
    initial_balance: float,
    metric: str                    = WFO_METRIC,
    split_months: int              = WFO_SPLIT_MONTHS,
    min_strategies: int            = MIN_STRATEGIES,
    max_strategies: int            = MAX_STRATEGIES,
    max_total_strategies: int      = MAX_TOTAL_STRATEGIES,
    top_n: int                     = TOP_N,
    require_long_short: bool       = REQUIRE_LONG_SHORT,
    require_all_timeframes: bool   = REQUIRE_ALL_TIMEFRAMES,
    show_plots: bool               = False,
) -> list:

    _validate_config(metric)

    if not validated_wfo_trades:
        logger.warning("No validated WFO trades — skipping best WFO portfolio search.")
        return []

    validated_wfo_trades = _cap_strategies_round_robin(
        validated_wfo_trades = validated_wfo_trades,
        initial_balance      = initial_balance,
        max_total_strategies = max_total_strategies,
    )

    all_ids = list({sid for sid, _ in validated_wfo_trades})
    logger.info(f"\n{'='*115}")
    logger.info(f"  BEST WFO PORTFOLIO — {len(all_ids)} validated strategies | metric: {metric}")
    logger.info(f"{'='*115}")

    subperiods = _split_trades_by_time(validated_wfo_trades, split_months)
    if not subperiods:
        logger.warning("No subperiods could be built — check WFO trades data.")
        return []

    n_splits          = len(subperiods)
    subperiod_weights = _generate_subperiod_weights(n_splits)

    logger.info(f"\n  Subperiods:")
    for i, (lbl, t_start, t_end, subset) in enumerate(subperiods):
        n_strats = len({sid for sid, _ in subset})
        logger.info(
            f"    {lbl:<4} {t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')}  "
            f"weight={subperiod_weights[i]:.2f}  strategies={n_strats}"
        )

    combos = _generate_combos(all_ids, min_strategies, max_strategies, require_long_short, require_all_timeframes)
    if not combos:
        logger.warning("No valid combinations found — check require_long_short or strategy count.")
        return []

    n_combos_str = f"{len(combos):,}".replace(",", ".")
    logger.info(f"\n  Evaluating {n_combos_str} combo(s)...\n")

    if metric not in _FAST_METRIC_MAP:
        raise ValueError(
            f"Vectorized combo scoring doesn't support metric='{metric}' "
            f"(needs trade-level data). Supported: {list(_FAST_METRIC_MAP.keys())}"
        )

    _combo_eval_start = time.perf_counter()
    raw_scores = _score_all_combos_vectorized(combos, subperiods, all_ids, initial_balance, metric)
    _combo_eval_elapsed = time.perf_counter() - _combo_eval_start
    logger.info(f"  Combo evaluation elapsed: {_combo_eval_elapsed:.2f}s (n_combos={len(combos):,}".replace(",", ".") + ")")

    split_labels    = [label for label, _, _, _ in subperiods]
    n_before_filter = len(raw_scores)

    if REQUIRE_SUBPERIODS_POSITIVE:
        raw_scores = [
            r for r in raw_scores
            if all(not np.isnan(r[label]) for label in split_labels)
        ]
        n_disqualified = n_before_filter - len(raw_scores)
        if n_disqualified > 0:
            n_ok              = n_before_filter - n_disqualified
            pct_disqualified  = n_disqualified / n_before_filter * 100
            pct_ok            = n_ok / n_before_filter * 100
            n_disqualified_str  = f"{n_disqualified:,}".replace(",", ".")
            n_before_filter_str = f"{n_before_filter:,}".replace(",", ".")
            n_ok_str            = f"{n_ok:,}".replace(",", ".")
            logger.info(
                f"  Disqualified {n_disqualified_str}/{n_before_filter_str} combo(s) with a losing/empty subperiod "
                f"({pct_disqualified:.2f}% discarded, {n_ok_str} OK = {pct_ok:.2f}% OK)."
            )
    else:
        logger.info(f"  REQUIRE_ALL_SUBPERIODS_POSITIVE=False — skipping positive-subperiod filter ({n_before_filter} combos kept).")

    if not raw_scores:
        logger.warning("All combos were disqualified (losing or empty subperiod in every combo). No portfolio found.")
        return []

    raw_scores = _rank_combos_by_subperiod(raw_scores, subperiods)

    for entry in raw_scores:
        entry["weighted_rank_score"] = _weighted_rank_score(entry, subperiods, subperiod_weights)

    raw_scores.sort(key=lambda x: x["weighted_rank_score"])  # lower rank = better

    top = raw_scores[:top_n]
    print_best_wfo_portfolio(top, subperiods, validated_wfo_trades, initial_balance, metric, subperiod_weights, len(raw_scores))

    df_scored = pd.DataFrame([
        {"combo": r["combo"], "weighted_rank_score": r["weighted_rank_score"]}
        for r in raw_scores
    ])

    if top and show_plots:
        for rank, top_entry in enumerate(top, start=1):
            top_subp_scores   = {lbl: top_entry.get(lbl, np.nan) for lbl, _, _, _ in subperiods}
            plot_best_wfo_portfolio(
                combo             = top_entry["combo"],
                trades_list       = validated_wfo_trades,
                subperiods        = subperiods,
                subperiod_scores  = top_subp_scores,
                df_scored         = df_scored,
                initial_balance   = initial_balance,
                metric            = metric,
                weights           = subperiod_weights,
                title             = f"Best WFO Portfolio #{rank} — {metric}",
                validated_trades  = validated_wfo_trades,
            )

    return top
