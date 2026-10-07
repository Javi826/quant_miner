# quant_miner/darwinex/sweep/sweep_analyze.py (forex)
import os
import sys
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("BOT_sweep.analyze")

# =============================================================================
# CONFIG
# =============================================================================
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "sweep", "results.csv")     # written by sweep_backtest_fx

X = {                                               # research params ── None = numeric, dict = categorical coded 0/1
    "luck_max": None,
    "top_i":    None,
}
Y = {                                               # column: label
    "wfo_pct":     "wfo_pct",                       # POST-WFO: rules passed / rules in WFO
    "corr_passed": "RULES_c",                       # POST-CORRELATION: rules left, repeated variants of one edge out
}
RATES  = {"wfo_pct": ("wfo_passed", "wfo_candidates")}     # a rate, pooled as sum / sum; any other Y is a count (mean)
COLORS = {"wfo_pct": "tab:blue", "corr_passed": "tab:orange"}   # one line per Y
CMAPS  = {"wfo_pct": "Blues",    "corr_passed": "Oranges"}      # one heatmap per Y
HEATMAP_X, HEATMAP_Y = "luck_max", "top_i"

# Every config enters, with no minimum of rules in WFO: how many rules a config sends to WFO is a result of its params,
# dropping the ones with few would bias the analysis. Instead, every wfo_pct is weighted by the rules in WFO:
#   wfo_pct of a group of configs: rules passed / rules in WFO over all of them (POOLED), not the mean of their wfo_pct
#   correlations with wfo_pct: weighted by the rules in WFO of every config
#   wfo_pct of a single config in the TOP: wfo_pct_low, the lower bound of its 95% interval (Wilson), low with few
#   rules in WFO
# RULES_c of a group of configs: its mean per config.
Z = 1.96                                            # 95%

TOP_N        = 5
TOP_RANKINGS = [                                    # one TOP table per ranking
    ["corr_passed", "wfo_pct_low"],                 # by RULES_c ── ties: highest wfo_pct (lower bound) first
    ["wfo_pct_low", "corr_passed"],                 # by wfo_pct (lower bound) ── ties: most RULES_c first
]

SEP   = "═" * 100
X_W   = 18
COL_W = 10

# =============================================================================
# DATA
# =============================================================================
def _wilson_low(passed: float, n: float) -> float:
    """Lower bound of the 95% Wilson interval of passed / n, in %: close to the % with many rules in WFO, much
    lower with few."""
    if n <= 0:
        return float("nan")
    p = passed / n
    den = 1.0 + Z * Z / n
    centre = p + Z * Z / (2.0 * n)
    radius = Z * math.sqrt(p * (1.0 - p) / n + Z * Z / (4.0 * n * n))
    return 100.0 * (centre - radius) / den


def _load() -> pd.DataFrame:
    """Every config, with its wfo_pct and its 95% lower bound (wfo_pct_low)."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"No file {RESULTS_PATH} (run sweep_backtest_fx first)")
    df = pd.read_csv(RESULTS_PATH)
    missing = [c for c in [*X, *Y, "wfo_passed", "wfo_candidates"] if c not in df.columns]
    if missing:
        raise ValueError(f"results.csv has no columns: {missing}")
    for x, codes in X.items():
        if codes is not None:
            unknown = sorted(set(df[x]) - set(codes))
            if unknown:
                raise ValueError(f"{x} has values with no code in X: {unknown}")
    if df.empty:
        raise ValueError(f"{RESULTS_PATH} has no configs")
    no_wfo = df["wfo_candidates"] <= 0
    df["wfo_pct"] = df["wfo_pct"].where(~no_wfo)                        # no rule in WFO: no wfo_pct
    df["wfo_pct_low"] = [round(_wilson_low(p, n), 2) for p, n in zip(df["wfo_passed"], df["wfo_candidates"])]
    return df


def _encoded(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({x: df[x] if codes is None else df[x].map(codes) for x, codes in X.items()})


def _values(df: pd.DataFrame, x: str) -> list:
    codes = X[x]
    return sorted(df[x].unique()) if codes is None else [v for v in sorted(codes, key=codes.get) if v in set(df[x])]


def _aggregate(g: pd.DataFrame, y: str) -> float:
    """A group of configs: a rate POOLED (sum / sum), a count its mean per config."""
    if y in RATES:
        num, den = RATES[y]
        n = g[den].sum()
        return 100.0 * g[num].sum() / n if n > 0 else float("nan")
    return float(g[y].mean())


def _title(y: str, label: str) -> str:
    return f"{label} POOLED" if y in RATES else f"MEAN {label}"


def _by_value(df: pd.DataFrame, x: str) -> pd.DataFrame:
    """Per value of x: configs and every Y (a rate POOLED, a count its mean)."""
    rows = []
    for v in _values(df, x):
        g = df[df[x] == v]
        rows.append({"value": v, "configs": len(g), **{y: _aggregate(g, y) for y in Y}})
    return pd.DataFrame(rows)

# =============================================================================
# CORRELATIONS
# =============================================================================
def _wpearson(a: pd.Series, b: pd.Series, w: pd.Series) -> float:
    keep = (w > 0) & a.notna() & b.notna()
    a, b, w = a[keep].astype(float), b[keep].astype(float), w[keep].astype(float)
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")                         # a constant column has no correlation
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return float(cov / math.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w)))


def _wspearman(a: pd.Series, b: pd.Series, w: pd.Series) -> float:
    return _wpearson(a.rank(), b.rank(), w)         # weighted Pearson on average ranks


METHODS = {"SPEARMAN": _wspearman, "PEARSON": _wpearson}


def _weights(df: pd.DataFrame, y: str) -> pd.Series:
    """A count: every config the same. A rate: the rules in WFO of every config (0: it has no wfo_pct)."""
    return df[RATES[y][1]].astype(float) if y in RATES else pd.Series(1.0, index=df.index)


def _n_eff(w: pd.Series) -> float:
    w = w[w > 0]
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def _critical_r(n: float) -> float:
    return Z / math.sqrt(n - 1) if n > 1 else float("nan")          # two-sided p = 0.05, large-sample approx


def _fmt_r(r: float, crit: float) -> str:
    if math.isnan(r):
        return "- "
    return f"{r:+.2f}{'*' if abs(r) >= crit else ' '}"

# =============================================================================
# REPORT
# =============================================================================
def _log_top(df: pd.DataFrame) -> None:
    names = {y: label for y, label in Y.items() if y != label}                # columns shown with their label
    cols  = [*X, "wfo_pct", "wfo_pct_low", "corr_passed"]
    for sort_by in TOP_RANKINGS:
        table = df.sort_values(sort_by, ascending=False, kind="stable", na_position="last").head(TOP_N)[cols]
        logger.info(f"\n{SEP}\n  TOP {TOP_N} ── sorted by {' then '.join(names.get(c, c) for c in sort_by)}\n{SEP}")
        logger.info(table.rename(columns=names).to_string(index=False))
    logger.info(f"  wfo_pct_low: lower bound of the 95% interval of wfo_pct ── few rules in WFO, low bound")


def _log_by_value(df: pd.DataFrame) -> None:
    heads = [_title(y, label) for y, label in Y.items()]
    w = max(11, *(len(h) + 2 for h in heads))
    logger.info(f"\n{SEP}")
    logger.info(f"  BY VALUE ── {len(df)} configs ── POOLED: sum of the configs / sum of their rules in WFO ── "
                f"MEAN: per config ── ✓ best")
    logger.info(f"{SEP}")
    logger.info(f"  {'VARIABLE':<{X_W}}{'VALUE':<10}{'CONFIGS':>8}" + "".join(f"{h:>{w + 2}}" for h in heads))
    logger.info(f"  {'─' * (X_W + 18 + (w + 2) * len(heads))}")
    for x in X:
        t = _by_value(df, x)
        best = {y: t[y].max() for y in Y}
        for r in t.to_dict("records"):                                   # records keep every column's type
            cells = ""
            for y in Y:
                v = f"{r[y]:.1f}" if not math.isnan(r[y]) else "-"
                cells += f"{v:>{w}} {'✓' if r[y] == best[y] else ' '}"
            logger.info(f"  {x:<{X_W}}{str(r['value']):<10}{r['configs']:>8}{cells}")
        logger.info("")


def _log_correlations(df: pd.DataFrame) -> None:
    enc   = _encoded(df)
    crits = {y: _critical_r(_n_eff(_weights(df, y))) for y in Y}
    logger.info(f"\n{SEP}")
    logger.info(f"  CORRELATIONS ── {len(df)} configs ── a rate weighted by its rules in WFO ── * = p < 0.05 ("
                + ", ".join(f"|r| >= {crits[y]:.2f} for {label}" for y, label in Y.items()) + ")")
    logger.info(f"{SEP}")
    logger.info(f"  {'':<{X_W}}" + "".join(f"{m:^{COL_W * len(Y)}}" for m in METHODS))
    logger.info(f"  {'X':<{X_W}}" + "".join(f"{label:>{COL_W - 1}} " for _ in METHODS for label in Y.values()))
    logger.info(f"  {'─' * (X_W + COL_W * len(Y) * len(METHODS))}")
    for x in X:
        cells = "".join(f"{_fmt_r(fn(enc[x], df[y], _weights(df, y)), crits[y]):>{COL_W}}"
                        for fn in METHODS.values() for y in Y)
        logger.info(f"  {x:<{X_W}}{cells}")
    logger.info(f"  a correlation sees only a trend: a best value in the middle (BY VALUE, heatmap) gives r near 0")


def _paired(df: pd.DataFrame, x: str, codes: dict, y: str) -> tuple:
    a, b   = sorted(codes, key=codes.get)                               # value coded 0, value coded 1
    others = [k for k in X if k != x]
    wide   = df.pivot_table(index=others, columns=x, values=y, aggfunc="first")
    if a not in wide.columns or b not in wide.columns:
        return 0, 0, 0
    wide = wide.dropna(subset=[a, b])                                   # only pairs with both configs present
    return int((wide[a] > wide[b]).sum()), int((wide[b] > wide[a]).sum()), int((wide[a] == wide[b]).sum())


def _log_paired(df: pd.DataFrame) -> None:
    if all(codes is None for codes in X.values()):
        return                                                          # no categorical X: nothing to pair
    logger.info(f"\n{SEP}")
    logger.info("  PAIRED ── same config, only this variable changes")
    logger.info(f"{SEP}")
    logger.info(f"  {'VARIABLE':<{X_W}}{'BY':<10}{'WINNER':<11}{'WINS':>4}{'LOSSES':>9}{'TIES':>7}")
    logger.info(f"  {'─' * (X_W + 41)}")
    for x, codes in X.items():
        if codes is None:
            continue
        a, b = sorted(codes, key=codes.get)
        for y, label in Y.items():
            wins_a, wins_b, ties = _paired(df, x, codes, y)
            winner       = a if wins_a > wins_b else b if wins_b > wins_a else "-"
            wins, losses = max(wins_a, wins_b), min(wins_a, wins_b)
            logger.info(f"  {x:<{X_W}}{label:<10}{winner:<11}{wins:>4}{losses:>9}{ties:>7}")


def _plot_means(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(X), figsize=(4 * len(X), 3.5), squeeze=False)
    for col, (x, codes) in enumerate(X.items()):
        ax = axes[0][col]
        t  = _by_value(df, x)
        xs = t["value"] if codes is None else [str(v) for v in t["value"]]
        for y, label in Y.items():
            ax.plot(xs, t[y].values, marker="o", color=COLORS[y], label=_title(y, label))
        if codes is None:
            ax.set_xticks(t["value"])
        ax.set_title(x)
        if col == 0:
            ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"BY VALUE ── {len(df)} configs ── " + ", ".join(_title(y, label) for y, label in Y.items()))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def _heatmap_table(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """Every cell (HEATMAP_X, HEATMAP_Y): a rate POOLED, a count its mean."""
    if y in RATES:
        num, den = RATES[y]
        sums = df.pivot_table(index=HEATMAP_Y, columns=HEATMAP_X, values=[num, den], aggfunc="sum")
        n = sums[den]
        return (100.0 * sums[num] / n.where(n > 0)).astype(float)
    return df.pivot_table(index=HEATMAP_Y, columns=HEATMAP_X, values=y, aggfunc="mean")


def _plot_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(Y), figsize=(6 * len(Y), 4.5), squeeze=False)
    for col, (y, label) in enumerate(Y.items()):
        ax    = axes[0][col]
        table = _heatmap_table(df, y)
        ax.imshow(table.values, cmap=CMAPS[y], origin="lower", aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns)
        ax.set_yticks(range(len(table.index)), table.index)
        light = np.nanmax(table.values) / 2                            # above it the cell is dark: white text
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.values[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                            color="white" if v > light else "black")
        ax.set_xlabel(HEATMAP_X)
        ax.set_ylabel(HEATMAP_Y)
        ax.set_title(_title(y, label))
    fig.suptitle(f"{HEATMAP_X} × {HEATMAP_Y} ── {len(df)} configs")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    df = _load()
    _log_top(df)
    _log_by_value(df)
    _log_correlations(df)
    _log_paired(df)
    _plot_means(df)
    _plot_heatmap(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    logger.setLevel(logging.INFO)
    main()