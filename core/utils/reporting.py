#core/utils/reporting.py
import logging
import numpy as np
import pandas as pd
from utils.batch_metrics import compute_metrics
from utils.ohlcv_utils import get_bars_per_day
from setup.config_core import settings
logger = logging.getLogger("BOT_batch.utils.reporting")

# =============================================================================
# PRINT HELPERS
# =============================================================================
def print_rule_mining_min_by_group_is(rows: list, stage_label: str, candidate_rows: list) -> None:
    if not rows:
        return

    groups = {}
    for r in rows:
        key = (r.get("timeframe", ""), r.get("side", ""))
        groups.setdefault(key, []).append(r)

    candidate_groups = {}
    for r in candidate_rows:
        key = (r.get("timeframe", ""), r.get("side", ""))
        candidate_groups.setdefault(key, []).append(r)

    n_total_passed    = len(rows)
    n_total_candidate = len(candidate_rows)
    total_pass_pct    = n_total_passed / n_total_candidate if n_total_candidate else 0.0

    logger.info(f"\n{'─' * 142}")
    logger.info(f"  RULE MINING RESULTS (IS) — {stage_label} ── {n_total_passed} / {n_total_candidate} passed ({total_pass_pct:.1%}) ✅")
    logger.info(f"{'─' * 142}")
    logger.info(
        f"{'TIMEFRAME':<12}{'SIDE':<8}{'N':<6}{'PASS%':<9}"
        f"{'NET_GAIN_IS% min/max':<22}{'MAX_DD_IS% min/max':<20}"
        f"{'DAYS/CANDLES p90':<20}"
    )
    logger.info(f"{'─' * 142}")
    for (tf, side), group_rows in sorted(groups.items()):
        n_group_candidates = len(candidate_groups.get((tf, side), []))
        group_pass_pct     = len(group_rows) / n_group_candidates if n_group_candidates else 0.0

        net_gain_values = [r.get("net_gain_is", 0.0) for r in group_rows]
        max_dd_values    = [r.get("max_dd_is", 0.0) for r in group_rows]
        duration_values  = [r.get("duration_is", 0.0) for r in group_rows]
        bars_per_day = get_bars_per_day(tf) * settings.DAYS_PER_YEAR / 365.0

        p90_duration = np.percentile(duration_values, 90)
        p90_candles  = p90_duration * bars_per_day

        logger.info(
            f"{tf:<12}{side:<8}{len(group_rows):<6}{f'{group_pass_pct:.1%}':<9}"
            f"{f'{min(net_gain_values):.1f} / {max(net_gain_values):.1f}':<22}"
            f"{f'{min(max_dd_values):.1f} / {max(max_dd_values):.1f}':<20}"
            f"{f'{p90_duration:.2f}d / {p90_candles:.1f}c':<20}"
        )
    logger.info(f"{'─' * 142}\n")
def print_metrics_table(metrics_list: list, title: str) -> None:
    df          = pd.DataFrame(metrics_list).drop(columns=["Calmar"])
    df["Curve"] = df["Curve"].astype(str)
    max_len     = df["Curve"].str.len().max()
    df["Curve"] = df["Curve"].apply(lambda x: x.ljust(max_len))
    logger.debug(f"\n{title}\n{df.to_string(index=False)}")

def print_portfolio_metrics_table(
    strategy_trades: list,
    label: str,
    initial_balance: float,
) -> None:
    """Print individual + combined metrics table for a list of (strategy_id, trade_log)."""
    named        = {sid: df for sid, df in strategy_trades}
    metrics_list = [compute_metrics(df, capital=initial_balance, name=sid) for sid, df in named.items()]

    if len(named) > 1:
        combined_tl      = pd.concat(list(named.values()), ignore_index=True).sort_values("buy_time").reset_index(drop=True)
        combined_capital = initial_balance * len(named)
        metrics_list.append(compute_metrics(combined_tl, capital=combined_capital, name="Combined"))

    print_metrics_table(metrics_list, f"📊 METRICS TABLE — {label}")


def print_all_curves_table(
    strategy_trades: list,
    label: str,
    initial_balance: float,
) -> None:
    """Print metrics table for all curves plus long/short aggregates and a combined row."""
    named = {sid: df for sid, df in strategy_trades}
    rows  = [compute_metrics(df, capital=initial_balance, name=sid) for sid, df in named.items()]

    long_trades  = [(sid, df) for sid, df in named.items() if "_long_"  in sid]
    short_trades = [(sid, df) for sid, df in named.items() if "_short_" in sid]

    if long_trades:
        long_tl  = pd.concat([df for _, df in long_trades], ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
        rows.append(compute_metrics(long_tl, capital=initial_balance * len(long_trades), name="── Longs"))

    if short_trades:
        short_tl = pd.concat([df for _, df in short_trades], ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
        rows.append(compute_metrics(short_tl, capital=initial_balance * len(short_trades), name="── Shorts"))

    all_tl  = pd.concat(list(named.values()), ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
    rows.append(compute_metrics(all_tl, capital=initial_balance * len(named), name="── Combined"))

    cols   = ["Curve", "Net_Gain_pct", "Max_DD_pct", "Win_Rate", "R_Squared", "Profit_Factor", "Profit_abs", "Profit_pctT", "Weekly_pct"]
    df_out = pd.DataFrame(rows)

    strategy_rows         = df_out[~df_out["Curve"].str.strip().str.startswith("──")]
    total_profit          = strategy_rows["Profit_abs"].sum()
    df_out["Profit_pctT"] = df_out["Profit_abs"].apply(
        lambda x: round(x / total_profit * 100, 1) if total_profit != 0 else np.nan
    )

    df_out = df_out[cols].copy()
    df_out["Net_Gain_pct"]  = df_out["Net_Gain_pct"].round(1)
    df_out["Max_DD_pct"]    = df_out["Max_DD_pct"].round(1)
    df_out["Win_Rate"]      = df_out["Win_Rate"].round(1)
    df_out["R_Squared"]     = df_out["R_Squared"].round(2)
    df_out["Profit_Factor"] = df_out["Profit_Factor"].round(2)
    df_out["Profit_pctT"]   = df_out["Profit_pctT"].round(0)
    df_out["Weekly_pct"]    = df_out["Weekly_pct"].round(0)

    max_len         = df_out["Curve"].str.len().max()
    df_out["Curve"] = df_out["Curve"].apply(lambda x: x.ljust(max_len))
    df_out["Profit_abs"] = df_out["Profit_abs"].apply(
        lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    longs_idx = df_out[df_out["Curve"].str.strip() == "── Longs"].index
    if len(longs_idx) > 0:
        sep_agg = pd.DataFrame({col: ["-----"] for col in cols})
        sep_agg["Curve"] = "-" * max_len
        df_out = pd.concat([df_out.iloc[:longs_idx[0]], sep_agg, df_out.iloc[longs_idx[0]:]], ignore_index=True)

    combined_idx     = df_out[df_out["Curve"].str.strip() == "── Combined"].index[0]
    sep_row          = pd.DataFrame({col: ["─" * max(len(str(df_out[col].iloc[0])), 9)] for col in cols})
    sep_row["Curve"] = "─" * max_len
    df_out           = pd.concat([df_out.iloc[:combined_idx], sep_row, df_out.iloc[combined_idx:]], ignore_index=True)

    n     = len(named)
    lines = [
        f"\n{'─'*115}\n📊 ALL CURVES COMBINED ({n}) — {label}\n{'─'*115}\n",
        df_out.to_string(index=False),
    ]
    logger.info("\n".join(lines))


# =============================================================================
# WFO SUMMARY
# =============================================================================

def print_wfo_summary(wfo_results: list, validation_results: list = None) -> None:
    """Print fused WFO approval + strategy metrics table."""
    if not wfo_results:
        return
    n_pass        = sum(1 for w in wfo_results if "PASS" in w["verdict"])
    mean_net_gain = round(np.mean([w["net_gain"] for w in wfo_results]), 1)
    mean_max_dd   = round(np.mean([w["max_dd"] for w in wfo_results]), 1)
    val_map     = {v["strategy_id"]: v for v in validation_results} if validation_results else {}
    has_metrics = bool(val_map)
    header = (
        f"  {'Strategy':<27} {'Verdict':<10} {'NetGain%':>9} {'DD%':>7}"
        + (f"  {'NetGain%':>9} {'DD%':>7} {'WinRate%':>9} {'R2':>7} {'Trades':>7}" if has_metrics else "")
    )
    sep = (
        f"  {'-'*27} {'-'*10} {'-'*9} {'-'*7}"
        + (f"  {'-'*9} {'-'*7} {'-'*9} {'-'*7} {'-'*7}" if has_metrics else "")
    )
    lines = [
        f"\n{'─'*115}",
        f"  WFO SUMMARY — Pass: {n_pass}/{len(wfo_results)} | MeanNetGain: {mean_net_gain}% | MeanDD: {mean_max_dd}%",
        f"{'─'*115}",
        header,
        sep,
    ]
    for w in wfo_results:
        sid  = w["strategy_id"]
        line = f"  {sid:<27} {w['verdict']:<10} {w['net_gain']:>8.1f}% {w['max_dd']:>6.1f}%"
        if has_metrics and sid in val_map:
            v        = val_map[sid]
            n_trades = v.get("tn_trades", 0)
            line += (
                f"  {v['net_gain_pct']:>8.2f}%"
                f" {v['dd_pct']:>6.2f}%"
                f" {v['win_ratio']:>8.1f}%"
                f" {v['r2']:>7.3f}"
                f" {n_trades:>7}"
            )
        lines.append(line)
    lines.append(f" {'─'*115}")
    logger.info("\n".join(lines))

def print_best_wfo_portfolio(
    top: list,
    subperiods: list,
    trades_list: list,
    initial_balance: float,
    metric: str,
    weights: list,
    n_qualified: int,
) -> None:
    W          = 115
    split_keys = [label for label, _, _, _ in subperiods]
    logger.info(f"\n{'='*W}")
    logger.info(f"  BEST WFO PORTFOLIO — metric: {metric} | splits: {len(subperiods)}")
    logger.info(f"{'='*W}")
    total_weeks = (subperiods[-1][2] - subperiods[0][1]).days / 7.0

    for rank, entry in enumerate(top, start=1):
        combo                  = entry["combo"]
        score                  = entry["weighted_rank_score"]
        n_strategies           = len(combo)
        total_trades           = sum(len(df) for sid, df in trades_list if sid in combo)
        avg_trades_weekly_sys  = total_trades / total_weeks if total_weeks > 0 else float("nan")
        avg_trades_weekly_strat = avg_trades_weekly_sys / n_strategies if n_strategies > 0 else float("nan")
        percentile             = score / n_qualified * 100
        logger.info(f"\nBEST #{rank} — Strategies: {n_strategies}  |  AvgTrades/week(system)={avg_trades_weekly_sys:.2f}  |  AvgTrades/week(strat)={avg_trades_weekly_strat:.2f}  |  Top {percentile:.1f}%")
        logger.info(f"{'─'*W}")
        for s in sorted(combo, key=lambda s: int(s.split("_")[0])):
            icon = "🟢" if "_long_" in s else "🔴"
            logger.info(f"    {icon} {s}")
        logger.debug(f"\n  {'Subperiod':<10} {'Weight':>8} {'Value':>10} {'Rank':>6}  {'Period'}")
        logger.debug(f"  {'─'*65}")
        for i, (lbl, t_start, t_end, _) in enumerate(subperiods):
            val      = entry.get(lbl, np.nan)
            val_str  = f"{val:.3f}" if not np.isnan(val) else "N/A"
            rank_val = entry.get(f"{lbl}_rank", "-")
            logger.debug(f"  {lbl:<10} {weights[i]:>8.2f} {val_str:>10} {rank_val:>6}  ({t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')})")
        logger.debug(f"  {'─'*65}")
        logger.debug(f"  {'WEIGHTED RANK':<10} {'':>8} {'':>10} {score:>6.2f}")

        combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
        if combo_trades:
            tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
            total_capital = initial_balance * len(combo_trades)
            m             = compute_metrics(tl, capital=total_capital, name="")

            last_label, last_t_start, last_t_end, _ = subperiods[-1]
            tl_last = tl[(tl["sell_time"] >= last_t_start) & (tl["sell_time"] < last_t_end)]
            m_last  = compute_metrics(tl_last, capital=total_capital, name="") if len(tl_last) > 0 else None

            _cols = ["NetGain", "DD", "WinRate", "R2", "PF", "Sharpe", "Weekly%", "MaxWeeksToRecovery"]
            logger.info(f"\n  {'Period':<16} {' '.join(f'{c:>10}' for c in _cols)}")
            logger.info(f"  {'─'*16} {'─'*(10*len(_cols) + len(_cols) - 1)}")

            def _row(label: str, mm: dict) -> str:
                vals = [
                    f"{mm['Net_Gain_pct']:.1f}%",
                    f"{mm['Max_DD_pct']:.1f}%",
                    f"{mm['Win_Rate']:.1f}%",
                    f"{mm['R_Squared']:.3f}",
                    f"{mm['Profit_Factor']:.2f}",
                    f"{mm['Sharpe']:.2f}",
                    f"{mm['Weekly_pct']:.1f}%",
                    f"{mm['Max_Weeks_to_Recovery']}",
                ]
                return f"  {label:<16} " + " ".join(f"{v:>10}" for v in vals)
            
            logger.info(_row("Full period", m))
            if m_last is not None:
                logger.info(_row(f"Last split ({last_label})", m_last))
            else:
                logger.info(f"  {f'Last split ({last_label})':<16} {'N/A':>10}")

            n_months        = max((pd.to_datetime(tl["sell_time"]).max() - pd.to_datetime(tl["sell_time"]).min()).days / 30.44, 1)
            avg_monthly_pct = round(m["Net_Gain_pct"] / n_months, 2)
            logger.info(f"\n  Monthly NetGain  ── {avg_monthly_pct:+.2f}% / month  ({n_months:.1f} months)")
    logger.info(f"\n{'─'*W}")

def _short_id(rule_id: str) -> str:
    parts = rule_id.split("_")
    return "_".join(parts[:3])

def print_rule_mining_ranking(all_raw_results: list, candidate_ids: list, stage_label: str, survivor_ids: list = None) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    candidate_set = set(candidate_ids)
    rows = [r for r in all_raw_results if r["rule_id"] in candidate_set]
    rows.sort(key=lambda r: int(r["rule_id"].split("_")[0]))

    show_status  = survivor_ids is not None
    survivor_set = set(survivor_ids) if show_status else None
    log_fn       = logger.debug

    id_width    = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width = max((len(r["label"]) for r in rows), default=8) + 2

    count_str = f"{len(survivor_ids)} / {len(candidate_ids)} passed" if show_status else f"{len(rows)} / {len(candidate_ids)} tested"

    log_fn(f"\n{'─' * 180}")
    log_fn(f"  RULE MINING RESULTS (OOS) — {stage_label} ── {count_str}")
    log_fn(f"{'─' * 180}")

    status_header = f"  {'STATUS':<8}" if show_status else ""
    log_fn(
        f"{'ID':<{id_width}}{'NET_GAIN%':<12}{'MAX_DD%':<10}{'PF':<8}{'SHARPE':<8}{'R2':<8}"
        f"{'STEPM_P':<8}{'WFR':<8}{'MC_RUIN':<9}{'MV_PVAL':<9}{'TRADES':<8}"
        f"{'WIN_RATE%':<11}{'DUR_D':<8}{'RULE':<{label_width}}{status_header}"
    )
    log_fn(f"{'─' * 180}")
    for r in rows:
        status_cell = f"  {('✅' if r['rule_id'] in survivor_set else '❌'):<8}" if show_status else ""
        log_fn(
            f"{_short_id(r['rule_id']):<{id_width}}{r['net_gain']:<12.1f}{r['max_dd']:<10.1f}"
            f"{r['profit_factor']:<8.2f}{(r.get('sharpe') or 0.0):<8.3f}{r['r_squared']:<8.3f}"
            f"{r.get('stepm_p', 0.0):<8.3f}{r['wfr']:<8.2f}{r.get('montecarlo_prob_ruin', 0.0):<9.1f}"
            f"{r.get('multiverse_p_value', 0.0):<9.3f}"
            f"{r['n_trades']:<8}{r.get('win_rate', 0.0):<11.1f}{r.get('duration_d', 0.0):<8.2f}"
            f"{r['label']:<{label_width}}{status_cell}"
        )

    log_fn(f"{'─' * 180}\n")


def print_rule_mining_min_by_group(all_raw_results: list, highlight_ids: list, stage_label: str, candidate_ids: list) -> None:
    highlight_set = set(highlight_ids)
    rows = [r for r in all_raw_results if r["rule_id"] in highlight_set]
    if not rows:
        return
    threshold_metrics = ["net_gain", "max_dd", "r_squared", "stepm_p", "wfr"]
    groups = {}
    for r in rows:
        key = (r["timeframe"], r["side"])
        groups.setdefault(key, []).append(r)
    group_stats = {}
    for key, group_rows in groups.items():
        group_stats[key] = {
            m: (min(r[m] for r in group_rows), max(r[m] for r in group_rows))
            for m in threshold_metrics
        }

    candidate_set  = set(candidate_ids)
    candidate_rows = [r for r in all_raw_results if r["rule_id"] in candidate_set]
    candidate_groups = {}
    for r in candidate_rows:
        key = (r["timeframe"], r["side"])
        candidate_groups.setdefault(key, []).append(r)

    n_total_passed    = len(highlight_ids)
    n_total_candidate = len(candidate_ids)
    total_pass_pct    = n_total_passed / n_total_candidate if n_total_candidate else 0.0

    logger.info(f"\n{'─' * 142}")
    logger.info(f"  RULE MINING RESULTS (OOS) — {stage_label} ── {n_total_passed} / {n_total_candidate} passed ({total_pass_pct:.1%}) ✅")
    logger.info(f"{'─' * 142}")
    logger.info(
        f"{'TIMEFRAME':<12}{'SIDE':<8}{'N':<6}{'PASS%':<9}"
        f"{'NET_GAIN% min/max':<22}{'MAX_DD% min/max':<20}{'R2 min/max':<16}"
        f"{'STEPM_P min/max':<16}{'WFR min/max':<16}"
        f"{'DAYS/CANDLES p90':<20}"
    )
    logger.info(f"{'─' * 142}")
    for (tf, side), group_rows in sorted(groups.items()):
        s = group_stats[(tf, side)]
        n_group_candidates = len(candidate_groups.get((tf, side), []))
        group_pass_pct = len(group_rows) / n_group_candidates if n_group_candidates else 0.0

        duration_d_values = [r.get("duration_d", 0.0) for r in group_rows]
        bars_per_day       = get_bars_per_day(tf) * settings.DAYS_PER_YEAR / 365.0

        p90_duration_d = np.percentile(duration_d_values, 90)
        p90_candles    = p90_duration_d * bars_per_day

        logger.info(
            f"{tf:<12}{side:<8}{len(group_rows):<6}{f'{group_pass_pct:.1%}':<9}"
            f"{f'{s['net_gain'][0]:.1f} / {s['net_gain'][1]:.1f}':<22}"
            f"{f'{s['max_dd'][0]:.1f} / {s['max_dd'][1]:.1f}':<20}"
            f"{f'{s['r_squared'][0]:.3f} / {s['r_squared'][1]:.3f}':<16}"
            f"{f'{s['stepm_p'][0]:.3f} / {s['stepm_p'][1]:.3f}':<16}"
            f"{f'{s['wfr'][0]:.2f} / {s['wfr'][1]:.2f}':<16}"
            f"{f'{p90_duration_d:.2f}d / {p90_candles:.1f}c':<20}"
        )
    logger.info(f"{'─' * 142}")
    # ALL-SAFE: worst-case across the whole table -> every rule shown passes all conditions at once.
    all_safe_net_gain = min(s["net_gain"][0]   for s in group_stats.values())
    all_safe_max_dd   = max(abs(s["max_dd"][0]) for s in group_stats.values())
    all_safe_r2       = min(s["r_squared"][0]  for s in group_stats.values())
    all_safe_stepm_p  = min(s["stepm_p"][0]    for s in group_stats.values())
    all_safe_wfr      = min(s["wfr"][0]        for s in group_stats.values())

    anchors = {key: max(group_rows, key=lambda r: r["net_gain"]) for key, group_rows in groups.items()}
    safe_net_gain = min(a["net_gain"]  for a in anchors.values())
    safe_max_dd   = max(abs(a["max_dd"]) for a in anchors.values())
    safe_r2       = min(a["r_squared"] for a in anchors.values())
    safe_stepm_p  = min(a["stepm_p"] for a in anchors.values())
    safe_wfr      = min(a["wfr"] for a in anchors.values())

    logger.debug("\n  Anchor row per group (highest NET_GAIN, used to derive joint-safe thresholds):")
    for (tf, side), a in sorted(anchors.items()):
        logger.debug(f"    {tf:<10}{side:<8}{a['rule_id']}")

    label_all_safe   = "ALL - SAFE THRESHOLDS (guaranteed ALL rows in the table)"
    label_joint_safe = "ONE - SAFE THRESHOLDS (guaranteed ≥1 survivor per group)"
    label_width      = max(len(label_all_safe), len(label_joint_safe))

    logger.info(
        f"\n  {label_all_safe.ljust(label_width)} ── "
        f"NET_GAIN>={all_safe_net_gain:.1f}  MAX_DD<={all_safe_max_dd:.1f}  R2>={all_safe_r2:.3f}  "
        f"STEPM_P<={all_safe_stepm_p:.3f}  WFR>={all_safe_wfr:.2f}"
    )
    logger.info(
        f"  {label_joint_safe.ljust(label_width)} ── "
        f"NET_GAIN>={safe_net_gain:.1f}  MAX_DD<={safe_max_dd:.1f}  R2>={safe_r2:.3f}  "
        f"STEPM_P<={safe_stepm_p:.3f}  WFR>={safe_wfr:.2f}"
    )
    logger.info(f"{'─' * 142}\n")

# =============================================================================
# DSR — debug-only reporting (moved from pipeline/dsr.py)
# =============================================================================

def _dsr_is_period_str(r: dict) -> str:
    combo_daily_profit = r.get("combo_daily_profit") or {}
    best_combo_id       = r.get("best_combo_id")
    if best_combo_id is None or best_combo_id not in combo_daily_profit:
        return "n/a"
    daily_profit = combo_daily_profit[best_combo_id]
    if daily_profit is None:
        return "n/a"
    day_offsets, _values, start_day = daily_profit
    if day_offsets.size == 0:
        return "n/a"
    start = start_day + day_offsets.min().astype("timedelta64[D]")
    end   = start_day + day_offsets.max().astype("timedelta64[D]")
    start_dt = start.astype("datetime64[D]").astype(object)
    end_dt   = end.astype("datetime64[D]").astype(object)
    return f"{start_dt:%Y-%m-%d}..{end_dt:%Y-%m-%d}"

def print_dsr_is_metrics(raw_by_id: dict, dsr_by_id: dict, sr_by_id: dict, candidate_ids: set, passed_ids: set, sr0: float) -> None:

    rows = [raw_by_id[rid] for rid in candidate_ids if rid in raw_by_id]
    rows.sort(key=lambda r: dsr_by_id.get(r["rule_id"], 0.0), reverse=True)

    if not rows:
        return

    id_width     = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width  = max((len(r.get("label", "")) for r in rows), default=8) + 2
    combo_width  = max((len(r.get("best_combo_id", "") or "") for r in rows), default=8) + 2
    period_width = max((len(_dsr_is_period_str(r)) for r in rows), default=8) + 2

    logger.debug(f"\n{'─' * 200}")
    logger.debug(f"  DSR IS METRICS (full-period grid search) ── SR0={sr0:.4f} ── {len(rows)} candidates")
    logger.debug(f"{'─' * 200}")
    logger.debug(
        f"{'ID':<{id_width}}{'SIDE':<6}{'NET_GAIN_TR':<13}{'MAX_DD_TR':<11}{'SR_ANN':<10}{'SR_UNANN':<11}"
        f"{'SKEW_TR':<10}{'KURT_TR':<10}{'N_DAYS_TR':<11}{'DSR':<9}{'BEST_COMBO':<{combo_width}}"
        f"{'IS_PERIOD':<{period_width}}{'RULE':<{label_width}}{'STATUS':<8}"
    )
    logger.debug(f"{'─' * 200}")

    for r in rows:
        rule_id = r["rule_id"]
        status  = "✅" if rule_id in passed_ids else "❌"
        logger.debug(
            f"{_short_id(rule_id):<{id_width}}{r.get('side', ''):<6}"
            f"{r.get('net_gain_is', float('nan')):<13.1f}{r.get('max_dd_is', float('nan')):<11.1f}"
            f"{r.get('sharpe_is', float('nan')):<10.4f}{sr_by_id.get(rule_id, float('nan')):<11.4f}"
            f"{r.get('skew_is', float('nan')):<10.4f}{r.get('kurtosis_is', float('nan')):<10.4f}"
            f"{r.get('n_days_is', 0):<11}{dsr_by_id.get(rule_id, 0.0):<9.4f}"
            f"{(r.get('best_combo_id', '') or 'n/a'):<{combo_width}}"
            f"{_dsr_is_period_str(r):<{period_width}}"
            f"{r.get('label', ''):<{label_width}}{status:<8}"
        )
    logger.debug(f"{'─' * 200}\n")
 
# =============================================================================
# STEPM — debug-only reporting (moved from pipeline/stepm.py)
# =============================================================================

def print_stepm_matrix_debug(col_names: list, matrix_arr: np.ndarray, n_rows: int, all_dates: np.ndarray) -> None:
    logger.debug(
        f"MATRIX ── built {len(col_names)} columns (rule__combo) over "
        f"{n_rows} distinct days ── range [{all_dates.min()} .. {all_dates.max()}]"
    )
    zero_frac = (matrix_arr == 0).mean(axis=0)
    pct = np.percentile(zero_frac, [0, 50, 90, 99, 100])
    logger.debug(
        f"DESCRIBE[zero_fill] ── fraction of zero-filled days per column, "
        f"percentiles [min,p50,p90,p99,max] = "
        f"[{pct[0]:.3f}, {pct[1]:.3f}, {pct[2]:.3f}, {pct[3]:.3f}, {pct[4]:.3f}]"
    )


def print_stepm_real_variance_filter_debug(progress_label: str, n_cols_built: int, n_cols_after: int) -> None:
    n_dropped_real_variance = n_cols_built - n_cols_after
    logger.debug(
        f"MATRIX FILTER (real variance) {progress_label} ── "
        f"{n_dropped_real_variance}/{n_cols_built} columns dropped "
        f"(zero-variance original series) ── {n_cols_after} remain"
    )


def print_stepm_block_starts_debug(
    progress_label: str, n_blocks_needed: int, block_size: int, len_last: int, n_obs: int, n_cols: int,
) -> None:
    logger.debug(
        f"BLOCK STARTS {progress_label} ── n_blocks={n_blocks_needed} "
        f"block_size={block_size} last_block_len={len_last} "
        f"(reduced gather: {n_blocks_needed}x{n_cols} vs original {n_obs}x{n_cols} per replica)"
    )


def print_stepm_bootstrap_replicas_debug(progress_label: str, deviations: np.ndarray, n_cols: int, n_bootstrap: int) -> None:
    inf_mask = ~np.isfinite(deviations)
    n_inf_per_col = inf_mask.sum(axis=0)
    cols_with_inf_replica = int((n_inf_per_col > 0).sum())
    logger.debug(
        f"BOOTSTRAP REPLICAS {progress_label} ── "
        f"{cols_with_inf_replica}/{n_cols} columns hit a non-finite Sharpe "
        f"in at least one bootstrap replica (zero-variance block)"
    )
    affected = n_inf_per_col[n_inf_per_col > 0]
    if affected.size:
        pct = np.percentile(affected, [0, 50, 90, 100])
        logger.debug(
            f"DESCRIBE[inf_replicas] {progress_label} ── among affected columns, "
            f"non-finite replica count per column percentiles "
            f"[min,p50,p90,max] out of {n_bootstrap} = "
            f"[{pct[0]:.0f}, {pct[1]:.0f}, {pct[2]:.0f}, {pct[3]:.0f}]"
        )


def print_stepm_se_filter_debug(progress_label: str, n_cols_before: int, n_cols_after: int, sigma_hat: np.ndarray) -> None:
    n_dropped_bootstrap_se = n_cols_before - n_cols_after
    logger.debug(
        f"MATRIX FILTER (bootstrap SE) {progress_label} ── "
        f"{n_dropped_bootstrap_se}/{n_cols_before} columns dropped "
        f"(sigma_hat == 0 or non-finite after bootstrap) ── "
        f"{n_cols_after} remain"
    )
    pct_sigma = np.percentile(sigma_hat, [0, 50, 90, 99, 100])
    ratio_max_min = float(pct_sigma[-1] / max(pct_sigma[0], 1e-12))
    logger.debug(
        f"DESCRIBE[sigma_hat] {progress_label} ── bootstrap SE percentiles "
        f"[min,p50,p90,p99,max] = "
        f"[{pct_sigma[0]:.4f}, {pct_sigma[1]:.4f}, {pct_sigma[2]:.4f}, "
        f"{pct_sigma[3]:.4f}, {pct_sigma[4]:.4f}] ── ratio max/min = {ratio_max_min:.2f} "
        f"(White 2000 Sec.9 flagged a ratio of 22.2 as enough to break the basic method)"
    )


def print_stepm_studentization_debug(
    progress_label: str,
    studentized_deviations: np.ndarray,
    z_stat: np.ndarray,
    n_cols_built: int,
    n_cols_after_real_variance: int,
    n_cols_final: int,
) -> None:
    post_std = studentized_deviations.std(axis=0, ddof=1)
    studentization_ok = bool(np.allclose(post_std, 1.0, atol=1e-3))
    logger.debug(
        f"VERIFY[studentization] {progress_label} ── post-division std per column: "
        f"min={post_std.min():.6f} max={post_std.max():.6f} (expected ≡ 1.0 exactly "
        f"under Hansen-style constant sigma_hat*, NOT under the paper's per-replica "
        f"sigma_hat*,m) ── {'✅' if studentization_ok else '❌'}"
    )
    pct_z = np.percentile(z_stat, [0, 50, 90, 99, 100])
    logger.debug(
        f"DESCRIBE[z_stat] {progress_label} ── studentized statistic percentiles "
        f"[min,p50,p90,p99,max] = "
        f"[{pct_z[0]:.4f}, {pct_z[1]:.4f}, {pct_z[2]:.4f}, {pct_z[3]:.4f}, {pct_z[4]:.4f}]"
    )
    logger.debug(
        f"FUNNEL {progress_label} ── built={n_cols_built} → "
        f"after_real_variance_filter={n_cols_after_real_variance} → "
        f"after_bootstrap_se_filter={n_cols_final} "
        f"(survival rate={n_cols_final / n_cols_built:.2%})"
    )


def print_stepm_pvalue_quantile_equivalence_debug(
    k: int,
    kth_dev_active: np.ndarray,
    alpha: float,
    active_stat: np.ndarray,
    reject_local: np.ndarray,
    n_active: int,
) -> None:
    pct_dev = np.percentile(kth_dev_active, [0, 50, 90, 99, 100])
    logger.debug(
        f"DESCRIBE[kth_dev_active] iter0 (k={k}) ── percentiles "
        f"[min,p50,p90,p99,max] = "
        f"[{pct_dev[0]:.4f}, {pct_dev[1]:.4f}, {pct_dev[2]:.4f}, "
        f"{pct_dev[3]:.4f}, {pct_dev[4]:.4f}]"
    )
    quantile_val      = np.quantile(kth_dev_active, 1.0 - alpha)
    predicted_reject  = active_stat > quantile_val
    mismatches        = int(np.sum(predicted_reject != reject_local))
    mismatch_rate     = mismatches / max(n_active, 1)
    logger.debug(
        f"VERIFY[pvalue_quantile_equivalence] iter0 (k={k}) ── mismatches between "
        f"p-value rule and quantile-inversion rule = {mismatches}/{n_active} "
        f"({mismatch_rate:.4%}) ── {'✅' if mismatch_rate < 0.01 else '❌'}"
    )


def print_stepm_monotonicity_debug(k: int, adjusted_pval_sorted: np.ndarray) -> None:
    diffs = np.diff(adjusted_pval_sorted)
    monotonic_ok = bool(np.all(diffs >= -1e-9))
    min_diff = float(diffs.min()) if diffs.size else float("nan")
    logger.debug(
        f"VERIFY[monotonicity] (k={k}) ── adjusted p-values non-decreasing along "
        f"descending-statistic order ── {'✅' if monotonic_ok else '❌'} "
        f"(min diff={min_diff:.2e})"
    )


def print_stepm_brc_equivalence_debug(
    timeframe: str, k_fwe: int, global_p: float, stepm_p_by_col: dict, best_col_name: str,
) -> None:
    if k_fwe == 1:
        p_from_stepm = float(stepm_p_by_col.get(best_col_name, float("nan")))
        brc_match = bool(np.isclose(p_from_stepm, global_p, atol=1e-9))
        logger.debug(
            f"VERIFY[BRC_equivalence] {timeframe} (k={k_fwe}) ── global White p-value = "
            f"{global_p:.6f} vs StepM p-value of the same best column = "
            f"{p_from_stepm:.6f} ── {'✅' if brc_match else '❌'}"
        )
    else:
        logger.debug(
            f"VERIFY[BRC_equivalence] {timeframe} ── skipped: not applicable under "
            f"k-FWE (k={k_fwe} > 1) by construction"
        )
        
# =============================================================================
# MULTIVERSE — debug-only reporting (moved from pipeline/multiverse.py)
# =============================================================================

def _mean_offdiag_correlation(series_by_symbol: dict) -> float:
    """Mean off-diagonal correlation of a set of aligned series (tail-aligned)."""
    if len(series_by_symbol) < 2:
        return float("nan")

    common  = min(len(arr) for arr in series_by_symbol.values())
    matrix  = np.column_stack([arr[-common:] for arr in series_by_symbol.values()])
    corr    = np.corrcoef(matrix, rowvar=False)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return float(np.nanmean(offdiag))


def print_multiverse_path_validation(
    ohlcv_data: dict,
    synthetic_arr: dict,
    layout_info: dict,
    ref_symbol: str,
    n_ref_rows: int,
    timeframe: str,
    block_size: int,
) -> None:

    logger.debug(f"\n{'─' * 100}")
    logger.debug(f"  MULTIVERSE PATH VALIDATION ── {timeframe} ── path 0")
    logger.debug(f"{'─' * 100}")
    logger.debug(
        f"  layout      : block_size={block_size} n_blocks={layout_info['n_blocks']} "
        f"phase={layout_info['phase']} order_head={layout_info['order_head']}"
    )
    logger.debug(f"  grid        : ref={ref_symbol} n_ref_rows={n_ref_rows} symbols={len(synthetic_arr)}")
    logger.debug(f"  {'symbol':<12} {'rows':>7} {'real_end':>12} {'synth_end':>12} {'drift_err':>11}")

    real_returns, synth_returns = {}, {}
    for symbol, arr in synthetic_arr.items():
        real_close  = ohlcv_data[symbol]["close"].to_numpy(dtype=np.float64)
        synth_close = arr["close"]

        real_returns[symbol]  = np.diff(np.log(real_close))
        synth_returns[symbol] = np.diff(np.log(synth_close))

        drift_err = abs(float(synth_close[-1]) / float(real_close[-1]) - 1.0)
        logger.debug(
            f"  {symbol:<12} {len(synth_close):>7} {real_close[-1]:>12.4f} "
            f"{synth_close[-1]:>12.4f} {drift_err:>10.2%}"
        )

    logger.debug(
        f"  cross-sect  : mean off-diagonal corr of log returns ── "
        f"real={_mean_offdiag_correlation(real_returns):.4f} "
        f"synthetic={_mean_offdiag_correlation(synth_returns):.4f}"
    )
    logger.debug(f"{'─' * 100}\n")


def print_multiverse_wfo_validation(
    probe_rule: dict,
    probe_schedule: list,
    probe_profit: float,
    param_names: list,
    timeframe: str,
) -> None:

    logger.debug(f"\n{'─' * 100}")
    logger.debug(f"  MULTIVERSE WFO VALIDATION ── {timeframe} ── path 0 ── rule={probe_rule['rule_id']}")
    logger.debug(f"{'─' * 100}")
    logger.debug(
        f"  real side   : n_windows={probe_rule.get('n_windows')} "
        f"best_params={probe_rule.get('best_params')} "
        f"profit={float(probe_rule['wfo_test_trades']['profit'].sum()):.2f}"
    )

    if not probe_schedule:
        logger.debug("  synth side  : WFO produced no window schedule on this path")
    else:
        logger.debug(f"  synth side  : n_windows={len(probe_schedule)} profit={probe_profit:.2f}")
        for name in param_names:
            logger.debug(f"  {name:<12}: " + " → ".join(str(w[name]) for w in probe_schedule))

    logger.debug(f"{'─' * 100}\n")


def print_multiverse_null_distribution(
    rules: list,
    profits_by_id: dict,
    p_value_by_id: dict,
    approved_by_id: dict,
    n_paths: int,
    block_size: int,
) -> None:
    """Per-rule real statistic against the shape of its null distribution."""
    logger.debug(f"\n{'─' * 130}")
    logger.debug(f"  MULTIVERSE NULL DISTRIBUTION ── n_paths={n_paths} block_size={block_size}")
    logger.debug(f"{'─' * 130}")
    logger.debug(
        f"  {'RULE_ID':<44} {'REAL':>11} {'NULL_MEAN':>11} {'NULL_P90':>11} "
        f"{'NULL_MAX':>11} {'ZERO_PATHS':>11} {'P':>8} {'STATUS':>8}"
    )
    logger.debug(f"{'─' * 130}")

    for rule in rules:
        rule_id = rule["rule_id"]
        nulls   = np.asarray(profits_by_id[rule_id], dtype=np.float64)
        if nulls.size == 0:
            continue

        real       = float(rule["wfo_test_trades"]["profit"].sum())
        zero_paths = int(np.sum(nulls == 0.0))
        logger.debug(
            f"  {_short_id(rule_id):<44} {real:>11.2f} {nulls.mean():>11.2f} "
            f"{np.percentile(nulls, 90):>11.2f} {nulls.max():>11.2f} "
            f"{f'{zero_paths}/{nulls.size}':>11} {p_value_by_id[rule_id]:>8.4f} "
            f"{('✅' if approved_by_id[rule_id] else '❌'):>8}"
        )

    logger.debug(f"{'─' * 130}\n")

def report_multiverse_debug(
    ohlcv_data: dict,
    synthetic_arr: dict,
    layout_info: dict,
    ref_symbol: str,
    n_ref_rows: int,
    probe_rule: dict,
    probe_schedule: list,
    probe_profit: float,
    rules: list,
    param_names: list,
    profits_by_id: dict,
    p_value_by_id: dict,
    approved_by_id: dict,
    n_paths: int,
    block_size: int,
    timeframe: str,
) -> None:

    print_multiverse_path_validation(
        ohlcv_data, synthetic_arr, layout_info, ref_symbol, n_ref_rows, timeframe, block_size,
    )
    print_multiverse_wfo_validation(
        probe_rule, probe_schedule, probe_profit, param_names, timeframe,
    )
    print_multiverse_null_distribution(
        rules, profits_by_id, p_value_by_id, approved_by_id, n_paths, block_size,
    )
