# core/pipeline/correlation.py
import logging
import pandas as pd
from utils.batch_metrics import compute_metrics
logger = logging.getLogger("BOT_batch.pipeline.correlation")


CORRELATION_DD_TH = 0.70
# =============================================================================
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


def _equity_series(strategy_trades, capital: float) -> pd.Series:
    """Convert trades to daily equity series."""
    all_tl = (
        pd.concat([df for _, df in strategy_trades], ignore_index=True)
        if isinstance(strategy_trades, list)
        else strategy_trades
    )
    all_tl          = all_tl.sort_values("sell_time").reset_index(drop=True)
    all_tl["_date"] = pd.to_datetime(all_tl["sell_time"]).dt.normalize()
    daily           = all_tl.groupby("_date")["profit"].sum()
    date_range      = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
    daily           = daily.reindex(date_range, fill_value=0.0)
    equity          = capital + daily.cumsum()
    return pd.Series(equity.values, index=date_range)


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

    num_map = {sid: f"{_num(sid):02d}" for sid in series_combined}
    df_     = pd.DataFrame({num_map[sid]: s for sid, s in series_combined.items()}).fillna(0)
    corr_mx = df_.corr().round(2)
    logger.debug(f"\n{corr_mx.to_string()}")

    ranked    = sorted(all_sids, key=lambda s: metrics.get(s, {}).get("Net_Gain_pct", 0), reverse=True)
    selected  = []
    discarded = []

    id_width = max(len(sid) for sid in ranked) + 2
    sep_width = 6 + id_width + 10 + 20 + 40

    lines = [f"\n  {'Rank':<6} {'Strategy':<{id_width}} {'NetGain%':>10} {'Action':<20} {'Reason'}"]
    lines.append(f"  {'─' * sep_width}")

    for sid in ranked:
        ng         = metrics.get(sid, {}).get("Net_Gain_pct", 0)
        num        = num_map.get(sid, sid)
        correlated = False
        reason     = ""
        for kept in selected:
            kept_num = num_map.get(kept, kept)
            val      = corr_mx.loc[num, kept_num] if num in corr_mx.index and kept_num in corr_mx.columns else 0.0
            if pd.notna(val) and val > threshold:
                correlated = True
                reason     = f"corr={val:.2f} with {_short_id(kept)}"
                discarded.append(sid)
                break
        if correlated:
            lines.append(f"  {ranked.index(sid)+1:<6} {sid:<{id_width}} {ng:>9.2f}%  {'❌ DISCARDED':<20} {reason}")
        else:
            selected.append(sid)
            lines.append(f"  {ranked.index(sid)+1:<6} {sid:<{id_width}} {ng:>9.2f}%  {'✅ SELECTED':<20}")

    lines.append(f"  {'─' * sep_width}")
    logger.debug("\n".join(lines))

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
def pipe_correlation(
    rules: list,
    initial_balance: float,
    threshold: float = None,
) -> list:

    threshold = threshold if threshold is not None else CORRELATION_DD_TH

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
