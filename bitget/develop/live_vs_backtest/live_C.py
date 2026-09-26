#develop/live_lab/live_lab_C.py (crypto)
import os
import glob
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("live_lab.compare")

# =============================================================================
# CONFIGURATION
# =============================================================================
PRODUCTION_XLSX = os.path.expanduser(
    "~/projects/quant/quant_miner/bitget/BOT_crypto/persistence/bot_files_00/bot_trades_00.xlsx"
)
BATCH_TRADES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brief_trades")



DATE_FROM = "2026-08-21"
DATE_TO   = "2026-09-25"

SELECTED_STRATEGIES = [
    "001280_4H_long_RSI14gt60_AND_HISTVOL20gtSMA_HISTVOL40",
    "022301_12H_long_RSI7gt60_AND_HISTVOL20gtSMA_HISTVOL20_AND_HISTVOL30ltSMA_HISTVOL20",
    "024772_12H_long_RSI7lt60_AND_HISTVOL10ltSMA_HISTVOL20_AND_HISTVOL10gtSMA_HISTVOL50",
    "048243_1H_long_RSI14gt70_AND_HISTVOL10gtSMA_HISTVOL30_AND_HISTVOL30ltSMA_HISTVOL50",
    "102727_1H_short_RSI7lt40_AND_HISTVOL30gtSMA_HISTVOL20_AND_HISTVOL30ltSMA_HISTVOL30",
    "107277_6H_short_RSI7lt50_AND_ATR14gtSMA_ATR20_AND_ATR21ltSMA_ATR30",
    "142800_12H_short_RSI21gt40_AND_ATR14ltSMA_ATR50_AND_HISTVOL10gtSMA_HISTVOL30",
    "167892_4H_short_ATR14gtSMA_ATR30_AND_ATR21ltSMA_ATR20_AND_HISTVOL30gtSMA_HISTVOL20",
]

# Symbols to exclude from production trades before comparison
EXCLUDE_SYMBOLS = []

# A production trade matches a batch trade if it opens within this many minutes
# AFTER the batch trade (never before): batch_time <= prod_time <= batch_time + window
MATCH_FORWARD_MINUTES = 3

# Max gap (seconds) between consecutive buy_times to treat them as the same
# simultaneous-open round (signals fired together when the system was flat)
ROUND_GAP_SECONDS = 5

# Strategy for the detailed per-trade / per-round inspection (None to skip)
DETAIL_STRATEGY = "001280_4H_long_RSI14gt60_AND_HISTVOL20gtSMA_HISTVOL40"
DETAIL_STRATEGY = "107277_6H_short_RSI7lt50_AND_ATR14gtSMA_ATR20_AND_ATR21ltSMA_ATR30"

# Batch exit reasons that are artificial (the data range ended before TP/SL/
# SELL_AFTER was reached) and therefore not comparable against a real
# production close
NON_COMPARABLE_EXITS = ["END_OF_DATA"]

# Daily win-rate charts: one across all strategies, one for DETAIL_STRATEGY
PLOT_WIN_RATE = True


# =============================================================================
# LOADERS
# =============================================================================
PROD_COLUMNS  = ["buy_time", "sell_time", "strategy", "symbol", "profit", "status"]
BATCH_COLUMNS = ["buy_time", "sell_time", "strategy", "symbol", "profit", "exit_reason"]


def load_production(path: str) -> pd.DataFrame:
    """Loads production trades from the bot's Excel export.

    Every row exported here is already closed (no open-position tracking in
    this Excel), so status is always CLOSED.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip().upper() for c in df.columns]

    df = df.rename(columns={
        "OPEN_AT":  "buy_time",
        "CLOSE_AT": "sell_time",
        "STRATEGY": "strategy",
        "SYMBOL":   "symbol",
        "PROFIT":   "profit",
    })

    df["buy_time"]  = pd.to_datetime(df["buy_time"],  errors="coerce", utc=True)
    df["sell_time"] = pd.to_datetime(df["sell_time"], errors="coerce", utc=True)

    # European-style decimal comma ("7,25" -> "7.25") before numeric parsing
    df["profit"] = df["profit"].astype(str).str.replace(",", ".", regex=False)
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

    df["status"] = "CLOSED"

    return df[PROD_COLUMNS].dropna(subset=["buy_time"])


def load_batch(trades_dir: str, strategy_ids: list[str]) -> pd.DataFrame:
    """Loads full-backtest trades, one CSV per strategy."""
    frames  = []
    pattern = os.path.join(trades_dir, "trades_full_*.csv")

    for path in glob.glob(pattern):
        fname    = os.path.basename(path)
        strat_id = fname.replace("trades_full_", "").replace(".csv", "")

        if strategy_ids and strat_id not in strategy_ids:
            continue

        try:
            df = pd.read_csv(path)
            df["strategy"] = strat_id
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Could not read {fname}: {e}")

    if not frames:
        return pd.DataFrame(columns=BATCH_COLUMNS)

    df = pd.concat(frames, ignore_index=True)

    df["buy_time"]  = pd.to_datetime(df["buy_time"],  errors="coerce", utc=True)
    df["sell_time"] = pd.to_datetime(df["sell_time"], errors="coerce", utc=True)
    df["profit"]    = pd.to_numeric(df["profit"], errors="coerce")

    if "exit_reason" not in df.columns:
        df["exit_reason"] = "UNKNOWN"

    return df[BATCH_COLUMNS].dropna(subset=["buy_time"])


# =============================================================================
# FILTERS
# =============================================================================
def apply_filters(
    df:              pd.DataFrame,
    date_from:       str | None,
    date_to:         str | None,
    strategy_ids:    list[str],
    exclude_symbols: list[str] | None = None,
) -> pd.DataFrame:
    if date_from:
        df = df[df["buy_time"] >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        df = df[df["buy_time"] <= pd.Timestamp(date_to, tz="UTC")]
    if strategy_ids:
        df = df[df["strategy"].isin(strategy_ids)]
    if exclude_symbols:
        df = df[~df["symbol"].isin(exclude_symbols)]
    return df.copy()


# =============================================================================
# HELPERS
# =============================================================================
def _short_id(strategy_id: str) -> str:
    """Extracts the leading numeric id from a strategy label (e.g. '001280')."""
    return strategy_id.split("_")[0]


def _timeframe_to_offset(strategy_id: str) -> pd.Timedelta:
    """Extracts candle size from the strategy_id timeframe token (e.g. '4H')."""
    token = strategy_id.split("_")[1]
    unit  = token[-1]
    value = int(token[:-1])

    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "H":
        return pd.Timedelta(hours=value)
    if unit == "D":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unknown timeframe in strategy_id: {strategy_id}")


def _is_match(prow: pd.Series, brow: pd.Series, window: pd.Timedelta) -> bool:
    """True if symbols match and prod opens within [0, window] after batch."""
    if prow["symbol"] != brow["symbol"]:
        return False
    delta = prow["buy_time"] - brow["buy_time"]
    return pd.Timedelta(0) <= delta <= window


def _pair_trades(p: pd.DataFrame, b: pd.DataFrame, window: pd.Timedelta) -> tuple:
    """Greedy one-to-one pairing over pre-sorted frames. Returns (pairs, prod_only_idx, batch_only_idx)."""
    pairs     = []
    prod_only = []
    used_b    = set()

    for pi in range(len(p)):
        match_idx = None
        for bj in range(len(b)):
            if bj in used_b:
                continue
            if _is_match(p.iloc[pi], b.iloc[bj], window):
                match_idx = bj
                used_b.add(bj)
                break

        if match_idx is None:
            prod_only.append(pi)
        else:
            pairs.append((pi, match_idx))

    batch_only = [bj for bj in range(len(b)) if bj not in used_b]
    return pairs, prod_only, batch_only


# =============================================================================
# ENTRY MATCHING — per strategy, anchored on the first coincident trade
# =============================================================================
def _match_entries(p: pd.DataFrame, b: pd.DataFrame, window: pd.Timedelta) -> dict:
    p = p.sort_values(["buy_time", "symbol"], kind="stable").reset_index(drop=True)
    b = b.sort_values(["buy_time", "symbol"], kind="stable").reset_index(drop=True)

    anchor_p = anchor_b = None
    for bi in range(len(b)):
        for pi in range(len(p)):
            if _is_match(p.iloc[pi], b.iloc[bi], window):
                anchor_p, anchor_b = pi, bi
                break
        if anchor_p is not None:
            break

    if anchor_p is None:
        return {
            "synced":     False,
            "anchor_ts":  None,
            "matched":    0,
            "prod_only":  len(p),
            "batch_only": len(b),
            "chain_len":  0,
        }

    p_sync = p.iloc[anchor_p:].reset_index(drop=True)
    b_sync = b.iloc[anchor_b:].reset_index(drop=True)

    pairs, prod_only, batch_only = _pair_trades(p_sync, b_sync, window)

    return {
        "synced":     True,
        "anchor_ts":  p_sync.iloc[0]["buy_time"],
        "matched":    len(pairs),
        "prod_only":  len(prod_only),
        "batch_only": len(batch_only),
        "chain_len":  max(len(p_sync), len(b_sync)),
    }


def match_entries(df_prod: pd.DataFrame, df_batch: pd.DataFrame, forward_minutes: int) -> dict:
    window     = pd.Timedelta(minutes=forward_minutes)
    strategies = sorted(set(df_prod["strategy"].unique()) | set(df_batch["strategy"].unique()))

    per_strategy = []
    totals       = {"matched": 0, "prod_only": 0, "batch_only": 0}

    for sid in strategies:
        p = df_prod[df_prod["strategy"]   == sid]
        b = df_batch[df_batch["strategy"] == sid]

        result = _match_entries(p, b, window)
        result["strategy_id"] = sid
        result["match_pct"]   = (
            round(result["matched"] / result["chain_len"] * 100, 1)
            if result["chain_len"] else None
        )
        per_strategy.append(result)

        for key in totals:
            totals[key] += result[key]

    return {"per_strategy": per_strategy, "totals": totals}


def _outcome(row: pd.Series) -> str:
    """WIN / LOSS / OPEN for a single trade, without exposing the amount."""
    if pd.isna(row["profit"]):
        return "OPEN"
    return "WIN" if row["profit"] > 0 else "LOSS"


def _close_agrees(prow: pd.Series, brow: pd.Series, timeframe_min: float) -> str:
    """'ok' if both closes fall within one candle of each other, 'x' if not,
    '—' if the pair isn't comparable (prod still open, or an artificial
    END_OF_DATA batch exit)."""
    if pd.isna(prow["sell_time"]) or brow["exit_reason"] in NON_COMPARABLE_EXITS:
        return "—"
    lag = abs((prow["sell_time"] - brow["sell_time"]).total_seconds()) / 60.0
    return "ok" if lag <= timeframe_min else "x"


def print_trade_pairs(
    df_prod:     pd.DataFrame,
    df_batch:    pd.DataFrame,
    strategy_id: str,
    anchor_ts:   pd.Timestamp,
    window:      pd.Timedelta,
) -> None:
    """Trade-by-trade view from the anchor onwards, outcomes only."""
    p = (
        df_prod[(df_prod["strategy"] == strategy_id) & (df_prod["buy_time"] >= anchor_ts)]
        .sort_values(["buy_time", "symbol"], kind="stable").reset_index(drop=True)
    )
    b = (
        df_batch[(df_batch["strategy"] == strategy_id) & (df_batch["buy_time"] >= anchor_ts - window)]
        .sort_values(["buy_time", "symbol"], kind="stable").reset_index(drop=True)
    )

    pairs, prod_only, batch_only = _pair_trades(p, b, window)
    paired_b      = {bj: pi for pi, bj in pairs}
    timeframe_min = _timeframe_to_offset(strategy_id).total_seconds() / 60.0

    logger.info(f"\n{'=' * 132}")
    logger.info(f"  TRADE BY TRADE — {strategy_id}")
    logger.info(f"  anchor {anchor_ts} | window +{int(window.total_seconds() / 60)} min | candle {timeframe_min:.0f} min")
    logger.info(f"{'=' * 132}")
    logger.info(
        f"  {'SYMBOL':<12} {'OPEN_PROD':<21} {'OPEN_BATCH':<21} "
        f"{'PROD':>6} {'BATCH':>6} {'AGREE':>6} {'CLOSE_PROD':<21} {'CLOSE_BATCH':<21} {'CLOSE':>6}"
    )
    logger.info(f"  {'-' * 128}")

    agree_count = 0
    comparable  = 0
    close_ok    = 0
    close_cmp   = 0

    for pi in range(len(p)):
        prow = p.iloc[pi]
        bj   = next((j for j, i in paired_b.items() if i == pi), None)

        if bj is None:
            close_prod = "OPEN" if pd.isna(prow["sell_time"]) else str(prow["sell_time"])[:19]
            logger.info(
                f"  {prow['symbol']:<12} {str(prow['buy_time'])[:19]:<21} {'— none —':<21} "
                f"{_outcome(prow):>6} {'—':>6} {'x':>6} {close_prod:<21} {'—':<21} {'—':>6}"
            )
            continue

        brow       = b.iloc[bj]
        p_out      = _outcome(prow)
        b_out      = _outcome(brow)
        artificial = brow["exit_reason"] in NON_COMPARABLE_EXITS

        if p_out == "OPEN" or artificial:
            agree = "—"
        else:
            comparable += 1
            same        = p_out == b_out
            agree_count += same
            agree       = "ok" if same else "x"

        close = _close_agrees(prow, brow, timeframe_min)
        if close != "—":
            close_cmp += 1
            close_ok  += close == "ok"

        close_prod  = "OPEN" if pd.isna(prow["sell_time"]) else str(prow["sell_time"])[:19]
        close_batch = "END_OF_DATA" if artificial else str(brow["sell_time"])[:19]

        logger.info(
            f"  {prow['symbol']:<12} {str(prow['buy_time'])[:19]:<21} {str(brow['buy_time'])[:19]:<21} "
            f"{p_out:>6} {'EOD' if artificial else b_out:>6} {agree:>6} "
            f"{close_prod:<21} {close_batch:<21} {close:>6}"
        )

    for bj in batch_only:
        brow        = b.iloc[bj]
        artificial  = brow["exit_reason"] in NON_COMPARABLE_EXITS
        close_batch = "END_OF_DATA" if artificial else str(brow["sell_time"])[:19]
        logger.info(
            f"  {brow['symbol']:<12} {'— none —':<21} {str(brow['buy_time'])[:19]:<21} "
            f"{'—':>6} {_outcome(brow):>6} {'x':>6} {'—':<21} {close_batch:<21} {'—':>6}"
        )

    logger.info(f"  {'-' * 128}")
    logger.info(
        f"  Paired: {len(pairs)} | prod only: {len(prod_only)} | batch only: {len(batch_only)}"
    )
    logger.info(f"  Outcome agreement on comparable pairs: {agree_count}/{comparable}")
    logger.info(f"  Close within one candle on comparable pairs: {close_ok}/{close_cmp}")
    logger.info(f"  {'=' * 132}\n")

# =============================================================================
# ENTRY ROUNDS — simultaneous-open groups, with per-round close lag
# =============================================================================
def _group_into_rounds(df: pd.DataFrame, gap_seconds: int) -> list[dict]:
    df = df.sort_values("buy_time").reset_index(drop=True)

    if df.empty:
        return []

    rounds    = []
    start     = df.loc[0, "buy_time"]
    symbols   = [df.loc[0, "symbol"]]
    close_max = df.loc[0, "sell_time"]
    any_open  = pd.isna(df.loc[0, "sell_time"])

    for i in range(1, len(df)):
        gap = (df.loc[i, "buy_time"] - df.loc[i - 1, "buy_time"]).total_seconds()

        if gap > gap_seconds:
            rounds.append({
                "round_start": start, "round_end": close_max,
                "symbols": symbols, "any_open": any_open,
            })
            start     = df.loc[i, "buy_time"]
            symbols   = []
            close_max = df.loc[i, "sell_time"]
            any_open  = pd.isna(df.loc[i, "sell_time"])

        symbols.append(df.loc[i, "symbol"])
        if pd.isna(df.loc[i, "sell_time"]):
            any_open = True
        else:
            close_max = df.loc[i, "sell_time"] if pd.isna(close_max) else max(close_max, df.loc[i, "sell_time"])

    rounds.append({
        "round_start": start, "round_end": close_max,
        "symbols": symbols, "any_open": any_open,
    })
    return rounds


def _nearest_round_delta(ts: pd.Timestamp, other_rounds: list[dict]) -> float | None:
    """Signed delta in minutes (ts - nearest.round_start) to the closest round."""
    if not other_rounds:
        return None
    deltas = [(ts - r["round_start"]).total_seconds() / 60.0 for r in other_rounds]
    return min(deltas, key=abs)


def _is_other_side_busy(other_df: pd.DataFrame, ts: pd.Timestamp) -> bool:
    """True if other_df has any trade open at ts (buy_time <= ts < sell_time)."""
    if other_df.empty:
        return False
    mask = (other_df["buy_time"] <= ts) & (other_df["sell_time"] > ts)
    return bool(mask.any())


def print_rounds_report(
    df_prod:      pd.DataFrame,
    df_batch:     pd.DataFrame,
    strategy_id:  str,
    anchor_ts:    pd.Timestamp,
    window:       pd.Timedelta,
    gap_seconds:  int,
) -> None:
    p = df_prod[(df_prod["strategy"] == strategy_id) & (df_prod["buy_time"] >= anchor_ts)]
    b = df_batch[(df_batch["strategy"] == strategy_id) & (df_batch["buy_time"] >= anchor_ts - window)]

    p_rounds      = _group_into_rounds(p, gap_seconds)
    b_rounds      = _group_into_rounds(b, gap_seconds)
    timeframe_min = _timeframe_to_offset(strategy_id).total_seconds() / 60.0

    artificial_ends = set(
        b[b["exit_reason"].isin(NON_COMPARABLE_EXITS)]["sell_time"].dropna()
    )

    logger.info(f"\n{'=' * 118}")
    logger.info(f"  ENTRY ROUNDS — {strategy_id}")
    logger.info(f"  anchor {anchor_ts} | window +{int(window.total_seconds() / 60)} min | candle {timeframe_min:.0f} min")
    logger.info(f"{'=' * 118}")
    logger.info(
        f"  {'PROD_ROUND':<21} {'BATCH_ROUND':<21} {'MATCH':>6} {'NEAREST':>9} {'STATUS':>8} "
        f"{'CLOSE_PROD':<21} {'CLOSE_BATCH':<21} {'LAG':>9}"
    )
    logger.info(f"  {'-' * 114}")

    used_b     = set()
    unmatched  = 0
    busy       = 0
    open_pend  = 0
    artificial = 0
    lags       = []

    for pr in p_rounds:
        match_idx = None
        for bi, br in enumerate(b_rounds):
            if bi in used_b:
                continue
            delta = pr["round_start"] - br["round_start"]
            if pd.Timedelta(0) <= delta <= window:
                match_idx = bi
                used_b.add(bi)
                break

        if match_idx is None:
            nearest   = _nearest_round_delta(pr["round_start"], b_rounds)
            delta_str = f"{nearest:+.1f}" if nearest is not None else "—"
            status    = "BUSY" if _is_other_side_busy(b, pr["round_start"]) else "FREE"
            busy     += status == "BUSY"
            unmatched += 1
            logger.info(
                f"  {str(pr['round_start'])[:19]:<21} {'— none —':<21} {'x':>6} {delta_str:>9} {status:>8} "
                f"{'—':<21} {'—':<21} {'—':>9}"
            )
            continue

        br = b_rounds[match_idx]

        if pr["any_open"]:
            close_prod, close_batch, lag_str = "OPEN", str(br["round_end"])[:19], "pending"
            open_pend += 1
        elif br["round_end"] in artificial_ends:
            close_prod, close_batch, lag_str = str(pr["round_end"])[:19], "END_OF_DATA", "n/a"
            artificial += 1
        else:
            lag = (pr["round_end"] - br["round_end"]).total_seconds() / 60.0
            lags.append(lag)
            close_prod  = str(pr["round_end"])[:19]
            close_batch = str(br["round_end"])[:19]
            lag_str     = f"{lag:+.1f}"

        logger.info(
            f"  {str(pr['round_start'])[:19]:<21} {str(br['round_start'])[:19]:<21} {'ok':>6} {'—':>9} {'—':>8} "
            f"{close_prod:<21} {close_batch:<21} {lag_str:>9}"
        )

    for bi in (bi for bi in range(len(b_rounds)) if bi not in used_b):
        br        = b_rounds[bi]
        nearest   = _nearest_round_delta(br["round_start"], p_rounds)
        delta_str = f"{nearest:+.1f}" if nearest is not None else "—"
        status    = "BUSY" if _is_other_side_busy(p, br["round_start"]) else "FREE"
        busy     += status == "BUSY"
        unmatched += 1
        logger.info(
            f"  {'— none —':<21} {str(br['round_start'])[:19]:<21} {'x':>6} {delta_str:>9} {status:>8} "
            f"{'—':<21} {'—':<21} {'—':>9}"
        )

    within = sum(1 for lag in lags if abs(lag) <= timeframe_min)

    logger.info(f"  {'-' * 114}")
    logger.info(f"  Rounds: prod={len(p_rounds)} batch={len(b_rounds)} matched={len(used_b)} unmatched={unmatched}")
    logger.info(f"  Unmatched rounds where the other side was busy: {busy}/{unmatched}")
    logger.info(f"  Close lag comparable on {len(lags)} rounds | within one candle: {within}/{len(lags)}")
    logger.info(f"  Excluded from lag: {open_pend} still open in prod, {artificial} artificial batch exits")
    logger.info(f"  {'=' * 118}\n")


# =============================================================================
# DAILY WIN RATE — a day counts only once every prod trade on it has closed
# =============================================================================
def _daily_win_rate(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, name="wr")

    df = df.copy()
    df["date"] = df["buy_time"].dt.tz_localize(None).dt.normalize()

    return (
        df.groupby("date")
          .apply(lambda x: round((x["profit"] > 0).sum() / len(x) * 100, 0), include_groups=False)
          .rename("wr")
    )
def _settled_dates(df_prod: pd.DataFrame) -> set:
    """Dates where every production trade has a close. Others are not scored."""
    if df_prod.empty:
        return set()

    df = df_prod.copy()
    df["date"] = df["buy_time"].dt.tz_localize(None).dt.normalize()
    settled    = df.groupby("date")["status"].apply(lambda s: (s == "CLOSED").all())

    return set(settled[settled].index)


# =============================================================================
# DAILY WIN RATE CHART
# =============================================================================
def plot_daily_wr(df_prod: pd.DataFrame, df_batch: pd.DataFrame, title: str) -> None:
    """Daily win rate, prod vs batch. Unsettled days are shaded, not scored."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if df_prod.empty and df_batch.empty:
        logger.warning(f"  Nothing to plot: {title}")
        return

    settled     = _settled_dates(df_prod)
    prod_closed = df_prod[df_prod["status"] == "CLOSED"]

    prod_wr  = _daily_win_rate(prod_closed)
    batch_wr = _daily_win_rate(df_batch)
    pending  = sorted(set(_daily_win_rate(df_prod).index) - settled)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Daily win rate — {title}", fontsize=12, fontweight="bold")

    for date in pending:
        ax.axvspan(
            date - pd.Timedelta(hours=12), date + pd.Timedelta(hours=12),
            color="gray", alpha=0.15, zorder=0,
        )

    if not prod_wr.empty:
        ax.plot(prod_wr.index, prod_wr.values, label="Production",
                color="#2196F3", linewidth=2, marker="o", markersize=4)
    if not batch_wr.empty:
        ax.plot(batch_wr.index, batch_wr.values, label="Backtest",
                color="#FF9800", linewidth=2, marker="s", markersize=4, linestyle="--")

    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Win rate % (daily)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    if pending:
        ax.text(
            0.99, 0.02, "shaded = open prod trade, day not scored",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray",
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# TRADE COUNTS
# =============================================================================
def print_counts_report(df_prod: pd.DataFrame, df_batch: pd.DataFrame, period: str) -> None:
    strategies = sorted(set(df_prod["strategy"].unique()) | set(df_batch["strategy"].unique()))

    logger.info(f"\n{'=' * 88}")
    logger.info(f"  TRADE COUNTS — production vs full backtest")
    logger.info(f"  Period   : {period}")
    logger.info(f"  Excluded : {EXCLUDE_SYMBOLS or '—'}")
    logger.info(f"{'=' * 88}")
    logger.info(
        f"  {'STRATEGY':<10} {'PROD':>7} {'BATCH':>7} {'DELTA':>7}"
    )
    logger.info(f"  {'-' * 84}")

    for sid in strategies:
        p = df_prod[df_prod["strategy"]   == sid]
        b = df_batch[df_batch["strategy"] == sid]

        logger.info(
            f"  {_short_id(sid):<10} {len(p):>7} {len(b):>7} {len(b) - len(p):>+7}"
        )

    logger.info(f"  {'=' * 88}\n")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    strategy_ids = SELECTED_STRATEGIES or []
    window       = pd.Timedelta(minutes=MATCH_FORWARD_MINUTES)

    logger.info("  Loading production trades...")
    df_prod = load_production(PRODUCTION_XLSX)
    df_prod = apply_filters(df_prod, None, DATE_TO, strategy_ids, EXCLUDE_SYMBOLS)

    if df_prod.empty:
        logger.warning("  No production trades found.")
        return

    effective_from = df_prod["buy_time"].min().strftime("%Y-%m-%d")
    if DATE_FROM and effective_from < DATE_FROM:
        effective_from = DATE_FROM

    df_prod = apply_filters(df_prod, effective_from, DATE_TO, strategy_ids, EXCLUDE_SYMBOLS)
    logger.info(f"  Effective start : {effective_from}")

    logger.info("  Loading backtest trades...")
    df_batch = load_batch(BATCH_TRADES_DIR, strategy_ids)

    if df_batch.empty:
        logger.warning(f"  No backtest trade files found in {BATCH_TRADES_DIR}")
        return

    df_batch = apply_filters(df_batch, effective_from, DATE_TO, strategy_ids)

    if df_prod.empty and df_batch.empty:
        logger.warning("  No trades found for the given filters.")
        return

    print_counts_report(df_prod, df_batch, f"{effective_from} -> {DATE_TO or '—'}")

    entry_result = match_entries(df_prod, df_batch, MATCH_FORWARD_MINUTES)

    if DETAIL_STRATEGY:
        anchor = next(
            (r["anchor_ts"] for r in entry_result["per_strategy"] if r["strategy_id"] == DETAIL_STRATEGY),
            None,
        )
        if anchor is None:
            logger.warning(f"  No anchor found for strategy: {DETAIL_STRATEGY}")
        else:
            print_trade_pairs(df_prod, df_batch, DETAIL_STRATEGY, anchor, window)
            print_rounds_report(
                df_prod, df_batch, DETAIL_STRATEGY, anchor, window, ROUND_GAP_SECONDS,
            )

    if PLOT_WIN_RATE:
        plot_daily_wr(df_prod, df_batch, "all strategies")

        if DETAIL_STRATEGY:
            plot_daily_wr(
                df_prod[df_prod["strategy"]   == DETAIL_STRATEGY],
                df_batch[df_batch["strategy"] == DETAIL_STRATEGY],
                _short_id(DETAIL_STRATEGY),
            )


if __name__ == "__main__":
    main()