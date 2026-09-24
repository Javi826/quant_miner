#core/utils/plotting.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
PLOT_SCALING     = 10
from utils.batch_metrics import compute_metrics
logger = logging.getLogger("BOT_batch.utils.plotting")


def _render_rule_mining_comparison_plot(
    ts_base, eq_base, m_base,
    ts_r01, eq_r01, m_r01,
    ref_ts, ref_pct,
    title: str,
    regime_enabled: bool = False,
) -> None:
    """Core rendering function for equity curve comparison plots."""
    eq_base = eq_base * PLOT_SCALING
    if eq_r01 is not None:
        eq_r01 = eq_r01 * PLOT_SCALING

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    if ts_r01 is not None and ref_ts is not None:
        btc_aligned = np.interp(
            pd.to_datetime(ts_r01).astype(np.int64) / 1e9,
            pd.to_datetime(ref_ts).astype(np.int64) / 1e9,
            ref_pct,
        )
        above = eq_r01 >= btc_aligned
        below = eq_r01 < btc_aligned
        ax.fill_between(ts_r01, eq_r01, 0, where=above, alpha=0.35, color="#00897B", interpolate=True)
        ax.fill_between(ts_r01, eq_r01, 0, where=below, alpha=0.35, color="#C62828", interpolate=True)

    base_name = "Regime" if regime_enabled else "Baseline"
    lbl_base  = (f"{base_name:<11} NetGain={m_base['Net_Gain_pct']:>6.1f}%  "
                 f"DD={m_base['Max_DD_pct']:>6.1f}%  R²={m_base['R_Squared']:.3f}")
    ax.plot(ts_base, eq_base, color="#2E86C1", linewidth=0.8, label=lbl_base)

    if ts_r01 is not None:
        lbl_r01 = (f"Regime 0+1  NetGain={m_r01['Net_Gain_pct']:>6.1f}%  "
                   f"DD={m_r01['Max_DD_pct']:>6.1f}%  R²={m_r01['R_Squared']:.3f}")
        ax.plot(ts_r01, eq_r01, color="#00897B", linewidth=1.4, label=lbl_r01)

    if ref_ts is not None:
        ax.plot(ref_ts, ref_pct, color="#FF8C00", linewidth=0.9,
                linestyle="--", alpha=0.6, label="_REF")

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Net Gain (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.8, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()

    legend = ax.legend(
        loc="upper left", fontsize=10, framealpha=0.95,
        facecolor="white", edgecolor="#AAAAAA", fancybox=False,
        borderpad=0.8, labelspacing=0.6, handlelength=2.5,
    )
    for text in legend.get_texts():
        text.set_fontfamily("monospace")

    plt.tight_layout()
    plt.show()


def plot_rule_mining_filter_comparison(
    strategy_id: str,
    trades_df_baseline: pd.DataFrame,
    trades_df_r01,
    data_folder: str,
    initial_balance: float,
    regime_enabled: bool = False,
) -> None:

    def _equity_pct(tl, t_start):
        tl  = tl.sort_values("sell_time").reset_index(drop=True)
        eq  = initial_balance + tl["profit"].cumsum().values
        pct = (eq - initial_balance) / initial_balance * 100
        m   = compute_metrics(tl, capital=initial_balance, name="")
        ts  = pd.to_datetime(tl["sell_time"]).values
        ts  = np.concatenate([[np.datetime64(t_start)], ts])
        pct = np.concatenate([[0.0], pct])
        return ts, pct, m

    t_start = pd.Timestamp(pd.to_datetime(trades_df_baseline["sell_time"]).min())
    t_end   = pd.Timestamp(pd.to_datetime(trades_df_baseline["sell_time"]).max())

    ts_base, eq_base, m_base = _equity_pct(trades_df_baseline, t_start)
    ts_r01, eq_r01, m_r01 = (
        _equity_pct(trades_df_r01, t_start)
        if trades_df_r01 is not None and len(trades_df_r01) > 0
        else (None, None, None)
    )
    ref_ts, ref_pct = None, None

    _render_rule_mining_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, ref_ts, ref_pct, strategy_id, regime_enabled=regime_enabled)

def plot_rule_mining_portfolio_comparison(
    strategy_trades_baseline: list,
    strategy_trades_regime01: list,
    data_folder: str,
    initial_balance: float,
    title: str = "Portfolio",
) -> None:

    if not strategy_trades_baseline:
        return

    def _combined_equity_pct(strategy_trades, capital_per_strategy):
        all_tl = pd.concat(
            [df for _, df in strategy_trades], ignore_index=True
        ).sort_values("buy_time").reset_index(drop=True)
        total_capital = capital_per_strategy * len(strategy_trades)
        eq    = total_capital + all_tl["profit"].cumsum().values
        pct   = (eq - total_capital) / total_capital * 100
        ts    = pd.to_datetime(all_tl["buy_time"]).values
        m     = compute_metrics(all_tl, capital=total_capital, name="")
        t_start = pd.Timestamp(all_tl["buy_time"].min())
        ts  = np.concatenate([[np.datetime64(t_start)], ts])
        pct = np.concatenate([[0.0], pct])
        return ts, pct, m, t_start

    ts_base, eq_base, m_base, t_start_base = _combined_equity_pct(strategy_trades_baseline, initial_balance)
    ts_r01, eq_r01, m_r01, t_start_r01 = (
        _combined_equity_pct(strategy_trades_regime01, initial_balance)
        if strategy_trades_regime01 else (None, None, None, None)
    )

    ref_ts, ref_pct = None, None

    _render_rule_mining_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, ref_ts, ref_pct, title)

def plot_best_wfo_portfolio(
    combo: tuple,
    trades_list: list,
    subperiods: list,
    subperiod_scores: dict,
    df_scored: pd.DataFrame,
    initial_balance: float,
    metric: str,
    weights: list,
    title: str,
    validated_trades: list = None,
) -> None:
    combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
    if not combo_trades:
        return

    tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
    total_capital = initial_balance * len(combo_trades)
    m             = compute_metrics(tl, capital=total_capital, name="")

    eq     = total_capital + tl["profit"].cumsum().values
    eq_pct = (eq - total_capital) / total_capital * 100
    ts     = pd.to_datetime(tl["sell_time"]).values

    ts_val = eq_val_pct = m_val = None
    if validated_trades:
        tl_val            = pd.concat([df for _, df in validated_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
        total_capital_val = initial_balance * len(validated_trades)
        m_val             = compute_metrics(tl_val, capital=total_capital_val, name="")
        eq_val            = total_capital_val + tl_val["profit"].cumsum().values
        eq_val_pct        = (eq_val - total_capital_val) / total_capital_val * 100
        ts_val            = pd.to_datetime(tl_val["sell_time"]).values

    _BG          = "#F8F9FA"
    _COLORS_BAND = ["#EBF5FB", "#EAFAF1", "#FEF9E7", "#FDEDEC"]
    _COLOR_EQ    = "#2E86C1"
    _COLOR_VAL   = "#00897B"

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(_BG)
    combo_str = " | ".join(sorted(combo, key=lambda s: int(s.split("_")[0])))
    fig.suptitle(title or f"Best WFO Portfolio — {combo_str}", fontsize=10, fontweight="bold")

    # ── Panel 1: Equity curve ─────────────────────────────────────────────────
    ax1.set_facecolor(_BG)

    for i, (_, t_start, t_end, _) in enumerate(subperiods):
        ax1.axvspan(t_start, t_end, alpha=0.25, color=_COLORS_BAND[i % len(_COLORS_BAND)])
        ax1.axvline(t_start, color="#AAAAAA", linewidth=0.5, linestyle="--", alpha=0.4)

    legend_label = (
        f"Best combo  NetGain={m['Net_Gain_pct']:.1f}%  "
        f"DD={m['Max_DD_pct']:.1f}%  "
        f"R²={m['R_Squared']:.3f}"
    )
    ax1.plot(ts, eq_pct, color=_COLOR_EQ, linewidth=1.0, label=legend_label)

    if ts_val is not None:
        legend_label_val = (
            f"Validated   NetGain={m_val['Net_Gain_pct']:.1f}%  "
            f"DD={m_val['Max_DD_pct']:.1f}%  "
            f"R²={m_val['R_Squared']:.3f}"
        )
        ax1.plot(ts_val, eq_val_pct, color=_COLOR_VAL, linewidth=0.8, alpha=0.8, label=legend_label_val)

    ax1.axhline(0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.5)

    _legend = ax1.legend(loc="upper left", fontsize=8, framealpha=0.9,
                         facecolor="white", edgecolor="#AAAAAA")
    for _text in _legend.get_texts():
        _text.set_fontfamily("monospace")
    ax1.set_title("Equity Curve (WFO Test)", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Net Gain (%)", fontsize=9)
    _locator = mdates.MonthLocator(interval=2)
    ax1.xaxis.set_major_locator(_locator)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="both", labelsize=7)
    ax1.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# =============================================================================
# MONTECARLO — debug-only plot (moved from pipeline/montecarlo.py)
# =============================================================================

def _mc_overlapping_blocks(profits: np.ndarray, block_size: int) -> np.ndarray:
    n_trades = len(profits)
    n_blocks = n_trades - block_size + 1
    return np.lib.stride_tricks.sliding_window_view(profits, block_size)[:n_blocks]

def plot_montecarlo_equity_curves(
    profits: np.ndarray,
    initial_balance: float,
    block_size: int,
    ruin_threshold_pct: float,
    n_curves: int = 100,
    seed: int = 42,
) -> None:

    n_trades = len(profits)
    blocks   = _mc_overlapping_blocks(profits, block_size)
    n_blocks = len(blocks)
    rng      = np.random.default_rng(seed + 1)
    n_blocks_needed = int(np.ceil(n_trades / block_size))

    fig, ax = plt.subplots(figsize=(12, 5))

    for _ in range(n_curves):
        chosen  = rng.integers(0, n_blocks, size=n_blocks_needed)
        sampled = np.concatenate(blocks[chosen])[:n_trades]
        equity  = initial_balance + np.cumsum(sampled)
        ax.plot(equity, color="gray", alpha=0.15, linewidth=0.7)

    original_equity      = initial_balance + np.cumsum(profits)
    original_running_max = np.maximum.accumulate(original_equity)
    ruin_band            = original_running_max * (1.0 - ruin_threshold_pct / 100.0)

    ax.plot(original_equity, color="red", linewidth=1.8, label="Original equity")
    ax.plot(ruin_band, color="red", linewidth=1.2, linestyle="--", label=f"Ruin threshold ({ruin_threshold_pct}%% DD)")

    ax.set_title(f"MONTECARLO — bootstrap equity curves (n_curves={n_curves})")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    plt.show()

# =============================================================================
# MULTIVERSE — debug-only plot (moved from pipeline/multiverse.py)
# =============================================================================

def plot_multiverse_synthetic_vs_historical(ohlcv_data: dict, paths: dict) -> None:
    for sym, df_hist in ohlcv_data.items():
        
        arr_paths = paths.get(sym)
        if arr_paths is None or arr_paths.shape[0] == 0:
            continue

        hist_close  = df_hist["close"].to_numpy(dtype=np.float64)
        synth_close = arr_paths[:, :, 3].astype(np.float64)

        fig, ax = plt.subplots(figsize=(12, 5))
        for path_idx in range(synth_close.shape[0]):
            ax.plot(synth_close[path_idx], color="gray", alpha=0.15, linewidth=0.7)
        ax.plot(hist_close, color="red", linewidth=1.8, label="Historical close")

        ax.set_title(f"{sym} — historical vs MCPT permuted paths (n_paths={synth_close.shape[0]})")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Close price")
        ax.legend()
        fig.tight_layout()
        plt.show()
