# core/pipeline/correlation.py
import logging
import numpy as np
import pandas as pd
from utils.batch_metrics import compute_metrics
logger = logging.getLogger("BOT_batch.pipeline.correlation")

CORRELATION_OOS_TH = 0.70   # post-WFO (OOS) greedy threshold
# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _num(sid: str) -> int:
    for part in sid.split("_"):
        if part.isdigit():
            return int(part)
    return 0

def _short_id(sid: str) -> str:
    parts = sid.split("_")
    return "_".join(parts[:3])

def _profit_series(df: pd.DataFrame, capital: float) -> pd.Series:
    tl          = df.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    daily       = tl.groupby("_date")["profit"].sum().groupby(level=0).sum()
    date_range  = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
    return daily.reindex(date_range).fillna(0.0)

# =============================================================================
# GREEDY CORE
# =============================================================================

class _DenseCorr:
    # OOS: precomputed (rounded) correlation matrix
    def __init__(self, corr_vals: np.ndarray):
        self.corr = corr_vals
        self.sel  = np.empty(corr_vals.shape[0], dtype=np.int64)
        self.n    = 0

    def row(self, pos: int) -> np.ndarray:
        return self.corr[pos, self.sel[:self.n]]

    def add(self, pos: int) -> None:
        self.sel[self.n] = pos
        self.n += 1

def _greedy_select(
    ranked: list,
    pos_by_id: dict,
    corr,
    threshold: float,
    score_by_id: dict,
    score_label: str,
) -> list:
    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    lines, id_width, sep_width = None, 0, 0
    if debug_enabled:
        id_width  = max(len(sid) for sid in ranked) + 2
        sep_width = 6 + id_width + 10 + 20 + 40
        lines     = [f"\n  {'Rank':<6} {'Strategy':<{id_width}} {score_label:>10} {'Action':<20} {'Reason'}"]
        lines.append(f"  {'─' * sep_width}")

    selected = []
    for rank_idx, sid in enumerate(ranked, start=1):
        pos   = pos_by_id[sid]
        hit_j = -1
        if corr.n:
            row  = corr.row(pos)
            hits = row > threshold               # NaN compares False, as pd.notna(val) and val > threshold
            if hits.any():
                hit_j = int(np.argmax(hits))     # first kept rule above the threshold

        if debug_enabled:
            sc = score_by_id.get(sid, 0)
            if hit_j >= 0:
                reason = f"corr={row[hit_j]:.2f} with {_short_id(selected[hit_j])}"
                lines.append(f"  {rank_idx:<6} {sid:<{id_width}} {sc:>9.2f}%  {'❌ DISCARDED':<20} {reason}")
            else:
                lines.append(f"  {rank_idx:<6} {sid:<{id_width}} {sc:>9.2f}%  {'✅ SELECTED':<20}")

        if hit_j < 0:
            corr.add(pos)
            selected.append(sid)

    if debug_enabled:
        lines.append(f"  {'─' * sep_width}")
        logger.debug("\n".join(lines))

    return selected

# =============================================================================
# OOS DECORRELATION (post-WFO): daily profit of the WFO test trades
# =============================================================================
def _column_keys(sids: list) -> dict:

    keys = {sid: _short_id(sid) for sid in sids}
    if len(set(keys.values())) != len(keys):
        keys = {sid: sid for sid in sids}
    return keys


def _decorrelate(
    strategy_trades_wfo_test: list,
    initial_balance: float,
    threshold: float,
    precomputed_metrics: dict,
    series_fn,
    label: str,
) -> list:
    metrics    = precomputed_metrics or {
        sid: compute_metrics(df, capital=initial_balance, name=sid)
        for sid, df in strategy_trades_wfo_test
    }
    trades_map = {sid: df for sid, df in strategy_trades_wfo_test}
    all_sids   = [sid for sid, _ in strategy_trades_wfo_test]

    series_combined = {}
    for sid in all_sids:
        s = series_fn(trades_map[sid], initial_balance)
        series_combined[sid] = s.groupby(level=0).mean()

    if len(series_combined) < 2:
        logger.info("  Not enough strategies for correlation analysis.")
        return strategy_trades_wfo_test

    key_map = _column_keys(list(series_combined))
    df_     = pd.DataFrame({key_map[sid]: s for sid, s in series_combined.items()}).fillna(0)
    corr_df = df_.corr().round(2)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\n{corr_df.to_string()}")

    col_pos   = {col: i for i, col in enumerate(corr_df.columns)}
    pos_by_id = {sid: col_pos[key_map[sid]] for sid in series_combined}
    score     = {sid: metrics.get(sid, {}).get("Net_Gain_pct", 0) for sid in all_sids}
    ranked    = sorted(all_sids, key=lambda s: metrics.get(s, {}).get("Net_Gain_pct", 0), reverse=True)

    selected = _greedy_select(ranked, pos_by_id, _DenseCorr(corr_df.to_numpy()), threshold, score, "NetGain%")

    return [(sid, trades_map[sid]) for sid in sorted(selected, key=_num) if sid in trades_map]


def decorrelate_by_profit(
    strategy_trades_wfo_test: list,
    initial_balance: float,
    threshold: float = 0.7,
    precomputed_metrics: dict = None,
) -> list:
    """Greedy profit-correlation filter. Keeps best NetGain from each correlated pair."""
    return _decorrelate(
        strategy_trades_wfo_test, initial_balance,
        threshold, precomputed_metrics,
        series_fn=_profit_series, label="Profit",
    )

# =============================================================================
# PIPE CORRELATION — greedy profit-correlation filter across all rules
# =============================================================================
def pipe_correlation_oos(
    rules: list,
    initial_balance: float,
    threshold: float = None,
) -> list:

    threshold = threshold if threshold is not None else CORRELATION_OOS_TH

    by_id = {r["rule_id"]: r for r in rules}
    strategy_trades_wfo_test = [(r["rule_id"], r["wfo_test_trades"]) for r in rules]
    precomputed_metrics      = {r["rule_id"]: {"Net_Gain_pct": r["net_gain"]} for r in rules}

    survivors = decorrelate_by_profit(
        strategy_trades_wfo_test = strategy_trades_wfo_test,
        initial_balance          = initial_balance,
        threshold                = threshold,
        precomputed_metrics      = precomputed_metrics,
    )

    return [by_id[rule_id] for rule_id, _ in survivors]