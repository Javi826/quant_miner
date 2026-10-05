# quant_miner/darwinex/sweep/sweep_analyze.py (forex)
import os
import sys
import math
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("BOT_sweep.analyze")

# =============================================================================
# CONFIG
# =============================================================================
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "sweep", "results.csv")     # written by sweep_backtest_fx

X = {                                               # research params ── None = numeric, dict = categorical coded 0/1
    "group_n":        None,
    "top_i":          None,
    "rank_by":        {"SYMBOLS": 0, "RULES": 1},   # r > 0: the value coded 1 gives more
    "symbols_deploy": {"all": 0, "terna": 1},
}
Y = {                                               # POST-WFO ── column: label
    "wfo_pct":    "%",
    "wfo_passed": "RULES",
}
COLORS = {"wfo_pct": "tab:blue", "wfo_passed": "tab:green"}     # one line per Y in the plot
CMAPS  = {"wfo_pct": "Blues",    "wfo_passed": "Greens"}        # one heatmap per Y
HEATMAP_X, HEATMAP_Y = "group_n", "top_i"
MIN_WFO_CANDIDATES = 1                             # a config enters the analysis only with at least these rules in WFO

TOP_N        = 5
TOP_RANKINGS = [                                    # one TOP table per ranking
    ["wfo_passed", "wfo_pct"],                      # by rules ── ties: highest % first
    ["wfo_pct", "wfo_passed"],                      # by %     ── ties: most rules first
]

SEP   = "═" * 80
X_W   = 18
COL_W = 10

# =============================================================================
# DATA
# =============================================================================
def _load() -> tuple:
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"No file {RESULTS_PATH} (run sweep_backtest_fx first)")
    df = pd.read_csv(RESULTS_PATH)
    missing = [c for c in [*X, *Y, "wfo_candidates"] if c not in df.columns]
    if missing:
        raise ValueError(f"results.csv has no columns: {missing}")
    for x, codes in X.items():
        if codes is not None:
            unknown = sorted(set(df[x]) - set(codes))
            if unknown:
                raise ValueError(f"{x} has values with no code in X: {unknown}")
    kept = df[df["wfo_candidates"] >= MIN_WFO_CANDIDATES].reset_index(drop=True)
    if kept.empty:
        raise ValueError(f"No config with wfo_candidates >= {MIN_WFO_CANDIDATES}: lower MIN_WFO_CANDIDATES")
    return kept, len(df)


def _encoded(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({x: df[x] if codes is None else df[x].map(codes) for x, codes in X.items()})

# =============================================================================
# CORRELATIONS
# =============================================================================
def _pearson(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")                         # a constant column has no correlation
    return a.corr(b)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return _pearson(a.rank(), b.rank())             # Pearson on average ranks


METHODS = {"SPEARMAN": _spearman, "PEARSON": _pearson}


def _critical_r(n: int) -> float:
    return 1.96 / math.sqrt(n - 1) if n > 1 else float("nan")      # two-sided p = 0.05, large-sample approx


def _fmt_r(r: float, crit: float) -> str:
    if math.isnan(r):
        return "- "
    return f"{r:+.2f}{'*' if abs(r) >= crit else ' '}"

# =============================================================================
# REPORT
# =============================================================================
def _log_top(df: pd.DataFrame) -> None:
    cols = [*X, "wfo_passed", "wfo_candidates", "wfo_pct"]
    for sort_by in TOP_RANKINGS:
        table = df.sort_values(sort_by, ascending=False, kind="stable").head(TOP_N)[cols]
        logger.info(f"\n{SEP}\n  TOP {TOP_N} ── sorted by {' then '.join(sort_by)}\n{SEP}")
        logger.info(table.to_string(index=False))


def _log_correlations(df: pd.DataFrame, n_total: int) -> None:
    enc  = _encoded(df)
    crit = _critical_r(len(df))
    logger.info(f"\n{SEP}")
    logger.info(f"  CORRELATIONS ── {len(df)}/{n_total} configs with wfo_candidates >= {MIN_WFO_CANDIDATES} "
                f"── * = |r| >= {crit:.2f} (p < 0.05)")
    logger.info(f"{SEP}")
    logger.info(f"  {'':<{X_W}}" + "".join(f"{m:^{COL_W * len(Y)}}" for m in METHODS))
    logger.info(f"  {'X':<{X_W}}" + "".join(f"{label:>{COL_W - 1}} " for _ in METHODS for label in Y.values()))
    logger.info(f"  {'─' * (X_W + COL_W * len(Y) * len(METHODS))}")
    for x in X:
        cells = "".join(f"{_fmt_r(fn(enc[x], df[y]), crit):>{COL_W}}" for fn in METHODS.values() for y in Y)
        logger.info(f"  {x:<{X_W}}{cells}")


def _paired(df: pd.DataFrame, x: str, codes: dict, y: str) -> tuple:
    a, b   = sorted(codes, key=codes.get)                               # value coded 0, value coded 1
    others = [k for k in X if k != x]
    wide   = df.pivot_table(index=others, columns=x, values=y, aggfunc="first")
    if a not in wide.columns or b not in wide.columns:
        return 0, 0, 0
    wide = wide.dropna(subset=[a, b])                                   # only pairs with both configs present
    return int((wide[a] > wide[b]).sum()), int((wide[b] > wide[a]).sum()), int((wide[a] == wide[b]).sum())


def _log_paired(df: pd.DataFrame) -> None:
    logger.info(f"\n{SEP}")
    logger.info("  PAIRED ── same config, only this variable changes")
    logger.info(f"{SEP}")
    logger.info(f"  {'VARIABLE':<{X_W}}{'BY':<9}{'WINNER':<11}{'WINS':>4}{'LOSSES':>9}{'TIES':>7}")
    logger.info(f"  {'─' * (X_W + 40)}")
    for x, codes in X.items():
        if codes is None:
            continue
        a, b = sorted(codes, key=codes.get)
        for y, label in Y.items():
            wins_a, wins_b, ties = _paired(df, x, codes, y)
            winner       = a if wins_a > wins_b else b if wins_b > wins_a else "-"
            wins, losses = max(wins_a, wins_b), min(wins_a, wins_b)
            logger.info(f"  {x:<{X_W}}{label:<9}{winner:<11}{wins:>4}{losses:>9}{ties:>7}")


def _plot_means(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(X), figsize=(4 * len(X), 3.5), squeeze=False)
    for col, (x, codes) in enumerate(X.items()):
        ax    = axes[0][col]
        means = df.groupby(x, sort=True)[list(Y)].mean()
        if codes is not None:
            means = means.reindex(sorted(codes, key=codes.get))           # categorical: value coded 0 first
        xs = means.index if codes is None else [str(v) for v in means.index]
        for y, label in Y.items():
            ax.plot(xs, means[y].values, marker="o", color=COLORS[y], label=label)
        if codes is None:
            ax.set_xticks(means.index)
        ax.set_title(x)
        if col == 0:
            ax.set_ylabel("MEAN")
            ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"MEAN BY VALUE ── {len(df)} configs")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def _plot_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(Y), figsize=(6 * len(Y), 4.5), squeeze=False)
    for col, (y, label) in enumerate(Y.items()):
        ax    = axes[0][col]
        table = df.pivot_table(index=HEATMAP_Y, columns=HEATMAP_X, values=y, aggfunc="mean")
        ax.imshow(table.values, cmap=CMAPS[y], origin="lower", aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns)
        ax.set_yticks(range(len(table.index)), table.index)
        light = table.max().max() / 2                                   # above it the cell is dark: white text
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.values[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                            color="white" if v > light else "black")
        ax.set_xlabel(HEATMAP_X)
        ax.set_ylabel(HEATMAP_Y)
        ax.set_title(f"MEAN {label}")
    fig.suptitle(f"{HEATMAP_X} × {HEATMAP_Y} ── {len(df)} configs")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    df, n_total = _load()
    _log_top(df)
    _log_correlations(df, n_total)
    _log_paired(df)
    _plot_means(df)
    _plot_heatmap(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    logger.setLevel(logging.INFO)
    main()