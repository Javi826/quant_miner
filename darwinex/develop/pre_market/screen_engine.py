# develop/pre_market/screen_engine.py
import os
import sys
import math
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

_HERE     = os.path.abspath(os.path.dirname(__file__))
_DARWINEX = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CORE     = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "core"))

for _p in (_HERE, _DARWINEX, _CORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from indicators.indicators_pool import CANDIDATE_REGISTRY, JUNK_NAMES, build_flat_instances, instance_key
from indicators.indicators_pool import describe_spec, generate_valid_combos
logger = logging.getLogger("BOT_batch.research.screen_engine")

# =============================================================================
# SHARED CONFIG
# =============================================================================
DATASET        = "IS"
TIMEFRAME_GRID = ["4H"]

SYMBOL_POOL = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
 #   "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]

TOP_K_GROWTH            = 20     # real indicators selected in depth 1 and grown in depth 2/3
MIN_COVERAGE            = 0.08   # min fraction of evaluable bars the signal must fire on
MAX_COVERAGE            = 0.50   # max fraction — protects the "no-signal" group from being too small
PAIR_CONS_MIN_THRESHOLD = 0.55
TIME_CONS_MIN_THRESHOLD = 0.55
N_HOLDOUT_WINDOWS       = 3      # last N windows of WINDOW_MONTHS each -> holdout
HOLDOUT_MIN_PASS_FRAC   = 2/3    # fraction of holdout windows that must pass
MAGNITUDE_FRAC          = 0.2    # min |edge_holdout_window| / |edge_selection| to not count as collapsed
PHI_THRESHOLD           = 0.80
JUNK_FLOOR_PERCENTILE   = 0.95
MAX_DEPTH               = 3

WINDOW_MONTHS           = 6
RULE_POOL_SIZE          = 1000
OPS                     = (">", "<")
CTX_WARMUP_DAYS         = 150

# =============================================================================
# SHARED HELPERS — panel / windows / fwd-return
# =============================================================================
def compute_window_ids(ts: np.ndarray, months: int) -> np.ndarray:
    idx = pd.DatetimeIndex(ts)
    return ((idx.year * 12 + (idx.month - 1)) // months).to_numpy()

def forward_return(close: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if h < len(close):
        out[:-h] = close[h:] / close[:-h] - 1.0
    return out

def build_panel(arr: dict, sym: str, n: int) -> dict:

    ctx   = {"sym": sym}
    panel = {}
    for inst in build_flat_instances():
        name, params = inst["indicator"], inst["params"]
        meta = CANDIDATE_REGISTRY[name]
        key  = instance_key(name, params)
        try:
            raw = np.asarray(meta["fn"](arr, ctx, params), dtype=np.float64)
        except Exception as exc:
            logger.warning(f"  [{sym}] FAILED {key}: {exc}")
            panel[key] = np.full(n, np.nan)
            continue
        if len(raw) != n:
            logger.warning(f"  [{sym}] {key}: length {len(raw)} != {n}, dropped")
            panel[key] = np.full(n, np.nan)
            continue
        shifted = np.full(n, np.nan)
        shifted[1:] = raw[:-1]
        panel[key] = shifted
    return panel

# =============================================================================
# SELECTION / HOLDOUT SPLIT — last N_HOLDOUT_WINDOWS calendar windows are
# =============================================================================
def holdout_window_ids(window_ids: dict, n_holdout: int) -> list:

    all_ids = np.unique(np.concatenate(list(window_ids.values())))
    all_ids.sort()
    return all_ids[-n_holdout:].tolist()


def split_selection_holdout_masks(ok_fwd: dict, window_ids: dict, holdout_ids: list) -> tuple:

    holdout_set = set(holdout_ids)
    selection_ok, holdout_ok = {}, {}
    for sym, ok in ok_fwd.items():
        is_holdout = np.isin(window_ids[sym], list(holdout_set))
        selection_ok[sym] = ok & ~is_holdout
        holdout_ok[sym]   = ok & is_holdout
    return selection_ok, holdout_ok

# =============================================================================
# CONDITION SPECS
# =============================================================================
def _spec_label(spec: dict) -> str:
    return describe_spec(spec)


def combo_label(members: tuple, specs: list) -> str:
    return " AND ".join(_spec_label(specs[i]) for i in members)


def _uses_junk(members: tuple, specs: list) -> bool:
    return any(specs[i]["indicator"] in JUNK_NAMES for i in members)


def condition_mask(panel: dict, spec: dict) -> np.ndarray:
    x = panel[spec["key"]]
    if spec["op"] == ">":
        return x > spec["threshold"]
    return x < spec["threshold"]


def build_mask_cache(panels: dict, specs: list) -> dict:

    cache = {}
    for sym, panel in panels.items():
        n_bars = len(next(iter(panel.values())))
        masks = np.empty((len(specs), n_bars), dtype=bool)
        for i, spec in enumerate(specs):
            masks[i] = condition_mask(panel, spec)
        cache[sym] = masks
    return cache


def _combo_mask(members: tuple, masks: np.ndarray) -> np.ndarray:
    """ANDs the condition masks for one symbol's mask_cache entry."""
    mask = masks[members[0]]
    for i in members[1:]:
        mask = mask & masks[i]
    return mask


def _build_window_slices(window_ids: dict) -> dict:

    slices = {}
    for sym, wids in window_ids.items():
        change  = np.flatnonzero(np.diff(wids)) + 1
        bounds  = np.concatenate(([0], change, [len(wids)]))
        slices[sym] = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    return slices

def report_depth1(df1: pd.DataFrame, selected: pd.DataFrame) -> None:
    hr("=")
    logger.info(f"  DEPTH 1 — {len(df1)} conditions evaluated, "
                f"{int(df1['passed'].sum())} passed the gate, "
                f"top {len(selected)} indicators selected for growth")
    hr("=")
    logger.info(format_rules(selected.head(TOP_K_GROWTH)).to_string(index=False))
    logger.info("")


def report_depth_n(depth: int, df: pd.DataFrame, rates: pd.DataFrame) -> None:
    floor = rates["noise_floor"].iloc[0] if not rates.empty else np.nan

    hr("=", debug=True)
    logger.debug(f"  DEPTH {depth} — {len(df)} combinations evaluated, "
                 f"{int(df['passed'].sum())} passed the gate")
    logger.debug(f"  Noise floor (best JUNK success rate): "
                 f"{floor:.1%}" if np.isfinite(floor) else "  Noise floor: n/a")
    hr("=", debug=True)
    logger.debug(format_rates(rates).to_string(index=False))
    logger.debug("")
# =============================================================================
# COHEN'S D — signal=True vs signal=False, on the forward return
# =============================================================================
def _edge_stats(mask: np.ndarray, fwd: np.ndarray, ok: np.ndarray) -> tuple:
    n_ok = int(ok.sum())
    if n_ok == 0:
        return np.nan, np.nan

    true_mask  = ok & mask
    false_mask = ok & ~mask
    n_true = int(true_mask.sum())

    coverage = n_true / n_ok
    if coverage < MIN_COVERAGE or coverage > MAX_COVERAGE:
        return np.nan, coverage

    group_true, group_false = fwd[true_mask], fwd[false_mask]
    n_true, n_false = len(group_true), len(group_false)
    if n_true < 2 or n_false < 2:
        return np.nan, coverage

    var_true, var_false = group_true.var(ddof=1), group_false.var(ddof=1)
    sd = np.sqrt(((n_true - 1) * var_true + (n_false - 1) * var_false) / (n_true + n_false - 2))
    if sd == 0.0:
        return np.nan, coverage

    edge = float((group_true.mean() - group_false.mean()) / sd)
    return edge, coverage


def evaluate_mask_for_symbol(mask: np.ndarray, fwd: np.ndarray, ok: np.ndarray,
                              window_slices: list) -> dict:
    edge_global, coverage = _edge_stats(mask, fwd, ok)

    edge_windows = []
    for start, end in window_slices:
        e, _ = _edge_stats(mask[start:end], fwd[start:end], ok[start:end])
        if np.isfinite(e):
            edge_windows.append(e)

    return {"edge": edge_global, "coverage": coverage, "windows": edge_windows}

# =============================================================================
# AGGREGATION ACROSS PAIRS
# =============================================================================
def aggregate_rule(per_symbol: dict) -> dict:
    edges = np.array([r["edge"] for r in per_symbol.values()], dtype=np.float64)
    edges = edges[np.isfinite(edges)]

    coverages = np.array([r["coverage"] for r in per_symbol.values()], dtype=np.float64)
    coverages = coverages[np.isfinite(coverages)]

    win_edges = np.array([e for r in per_symbol.values() for e in r["windows"]], dtype=np.float64)

    out = {"edge_mean": np.nan, "coverage_mean": np.nan, "pair_cons": np.nan, "time_cons": np.nan}

    if len(coverages):
        out["coverage_mean"] = float(coverages.mean())

    if len(edges):
        edge_mean = float(edges.mean())
        sign = 1.0 if edge_mean >= 0 else -1.0
        out["edge_mean"] = edge_mean
        out["pair_cons"] = float((np.sign(edges) == sign).mean())
        if len(win_edges):
            out["time_cons"] = float((np.sign(win_edges) == sign).mean())

    return out


def passes_gate(agg: dict) -> bool:
    if not np.isfinite(agg["edge_mean"]):
        return False
    return (MIN_COVERAGE <= agg["coverage_mean"] <= MAX_COVERAGE
            and agg["pair_cons"] >= PAIR_CONS_MIN_THRESHOLD
            and agg["time_cons"] >= TIME_CONS_MIN_THRESHOLD)

# =============================================================================
# ONE DEPTH LEVEL
# =============================================================================
def _evaluate_one_combo(members: tuple, specs: list, mask_cache: dict, fwd: dict,
                         eval_ok: dict, window_slices: dict) -> dict:
    per_symbol = {}
    for sym, masks in mask_cache.items():
        mask = _combo_mask(members, masks)
        per_symbol[sym] = evaluate_mask_for_symbol(
            mask, fwd[sym], eval_ok[sym], window_slices[sym]
        )
    agg = aggregate_rule(per_symbol)
    agg["members"] = members
    agg["label"]   = combo_label(members, specs)
    agg["depth"]   = len(members)
    agg["junk"]    = _uses_junk(members, specs)
    return agg


def evaluate_level(candidate_members: list, specs: list, mask_cache: dict, fwd: dict,
                    eval_ok: dict, window_slices: dict, progress_label: str = "") -> pd.DataFrame:
    desc = f"SCREEN {progress_label}".strip()

    rows = list(tqdm(
        Parallel(n_jobs=-1, batch_size=64, pre_dispatch="all", return_as="generator")(
            delayed(_evaluate_one_combo)(members, specs, mask_cache, fwd, eval_ok, window_slices)
            for members in candidate_members
        ),
        desc=desc,
        total=len(candidate_members),
        dynamic_ncols=True,
    ))

    sys.stderr.flush()
    sys.stdout.flush()

    return pd.DataFrame(rows)

# =============================================================================
# DEPTH 1 SELECTION — the only filtering step. Ranks indicators by their
# =============================================================================
def select_indicators(df: pd.DataFrame, specs: list, top_k: int) -> pd.DataFrame:
    passed = df[df["passed"]].copy()
    if passed.empty:
        return passed
    passed["score"]     = passed["edge_mean"].abs()
    passed["indicator"] = passed["members"].apply(lambda m: specs[m[0]]["indicator"])
    best = (passed.sort_values("score", ascending=False)
                  .groupby("indicator", as_index=False)
                  .first())
    return (best.sort_values("score", ascending=False)
                .head(top_k)
                .reset_index(drop=True))


def grid_indices_for(indicator_names: list, specs: list) -> list:
    names = set(indicator_names)
    return [i for i, spec in enumerate(specs) if spec["indicator"] in names]


def generate_combos(grid_idx: list, specs: list, depth: int) -> list:
    """All combinations of `depth` conditions, one per distinct indicator,
    exhaustive over the grid — excludes directionally-contradictory combos."""
    return generate_valid_combos(specs, depth, indices=grid_idx)

# =============================================================================
# INDICATOR SUCCESS RATE — per depth, independently. This replaces p-value
# =============================================================================
def indicator_success_rates(df: pd.DataFrame, specs: list, indicator_names: list) -> pd.DataFrame:
    member_names = df["members"].apply(lambda m: {specs[i]["indicator"] for i in m})

    rows = []
    for name in indicator_names:
        contains = member_names.apply(lambda s, n=name: n in s)
        subset   = df[contains]
        n_total  = len(subset)
        n_passed = int(subset["passed"].sum())
        rate     = n_passed / n_total if n_total else np.nan

        passed_edges = subset.loc[subset["passed"], "edge_mean"].abs()
        best_edge    = float(passed_edges.max()) if n_passed else np.nan

        rows.append({"indicator": name, "n_rules": n_total, "n_passed": n_passed,
                     "success_rate": rate, "best_edge": best_edge,
                     "junk": name in JUNK_NAMES})

    out   = pd.DataFrame(rows)
    floor = out.loc[out["junk"], "success_rate"].quantile(JUNK_FLOOR_PERCENTILE)
    out["noise_floor"] = floor
    out["above_floor"] = out["success_rate"] > floor
    return out.sort_values("success_rate", ascending=False).reset_index(drop=True)

# =============================================================================
# HIERARCHICAL GROWTH — only depth 1 filters. Depth 2 and 3 are each an
# =============================================================================
def grow_rules(specs: list, panels: dict, fwd: dict, eval_ok: dict, window_ids: dict,
               mask_cache: dict) -> tuple:
    window_slices = _build_window_slices(window_ids)

    level1_members = [(i,) for i in range(len(specs))]
    logger.info(f"  Depth 1: evaluating {len(level1_members)} single conditions")
    df1 = evaluate_level(level1_members, specs, mask_cache, fwd, eval_ok, window_slices, progress_label="depth1")
    df1["passed"] = df1.apply(passes_gate, axis=1)

    selected   = select_indicators(df1, specs, TOP_K_GROWTH)
    kept_names = list(selected["indicator"])
    grid_names = list(dict.fromkeys(kept_names + JUNK_NAMES))  # force the noise controls in
    grid_idx   = grid_indices_for(grid_names, specs)

    logger.info(f"  Depth 1: {int(df1['passed'].sum())} conditions passed the gate; "
                f"quick filter selected {len(kept_names)} indicators -> "
                f"grid of {len(grid_names)} (incl. {len(JUNK_NAMES)} noise controls)")

    survivors_by_depth = {1: df1}
    indicator_rates     = {1: selected}
    report_depth1(df1, selected)

    for depth in range(2, MAX_DEPTH + 1):
        candidates = generate_combos(grid_idx, specs, depth)
        logger.info(f"  Depth {depth}: evaluating {len(candidates):,} combinations "
                    f"(exhaustive over the {len(grid_names)}-indicator grid)")
        df = evaluate_level(candidates, specs, mask_cache, fwd, eval_ok, window_slices, progress_label=f"depth{depth}")
        df["passed"] = df.apply(passes_gate, axis=1)
        survivors_by_depth[depth] = df
        indicator_rates[depth]    = indicator_success_rates(df, specs, grid_names)
        report_depth_n(depth, df, indicator_rates[depth])

    return survivors_by_depth, indicator_rates

# =============================================================================
# TOP NON-REDUNDANT RULES — post-hoc only. Reads the already-evaluated rules
# =============================================================================
def _rule_masks(members: tuple, specs: list, panels: dict) -> dict:
    masks = {}
    for sym, panel in panels.items():
        mask = None
        for i in members:
            m = condition_mask(panel, specs[i])
            mask = m if mask is None else (mask & m)
        masks[sym] = mask
    return masks


def _phi(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = mask_a.astype(np.float64), mask_b.astype(np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _mean_phi(masks_a: dict, masks_b: dict) -> float:
    vals = []
    for sym in masks_a:
        if sym not in masks_b:
            continue
        p = _phi(masks_a[sym], masks_b[sym])
        if np.isfinite(p):
            vals.append(p)
    return float(np.mean(vals)) if vals else np.nan


def _junk_edge_floor(df: pd.DataFrame) -> float:
    passed_junk = df[df["passed"] & df["junk"]]
    if passed_junk.empty:
        return 0.0
    return float(passed_junk["edge_mean"].abs().quantile(JUNK_FLOOR_PERCENTILE))


def top_nonredundant_rules(df: pd.DataFrame, specs: list, panels: dict,
                            pool_size: int, top_k: int, threshold: float) -> tuple:
    floor = _junk_edge_floor(df)

    candidates = df[df["passed"] & ~df["junk"]].copy()
    candidates["score"] = candidates["edge_mean"].abs()
    candidates = candidates[candidates["score"] > floor]
    candidates = candidates.sort_values("score", ascending=False).head(pool_size)

    selected_masks, selected_rows = [], []
    for _, row in candidates.iterrows():
        if len(selected_rows) >= top_k:
            break
        masks = _rule_masks(row["members"], specs, panels)
        redundant = any(
            np.isfinite(phi := _mean_phi(masks, sel_masks)) and abs(phi) >= threshold
            for sel_masks in selected_masks
        )
        if not redundant:
            selected_masks.append(masks)
            selected_rows.append(row)

    shortlist = pd.DataFrame(selected_rows) if selected_rows else candidates.iloc[0:0]
    return shortlist, floor

# =============================================================================
# HOLDOUT GATE — post-hoc, hard filter. Applied only to rules that already
# =============================================================================
def _rule_edge_in_window(members: tuple, mask_cache: dict, fwd: dict, ok_fwd: dict,
                          window_ids: dict, wid: int) -> tuple:
    edges, coverages = [], []
    for sym, masks in mask_cache.items():
        mask   = _combo_mask(members, masks)
        ok_win = ok_fwd[sym] & (window_ids[sym] == wid)
        edge, coverage = _edge_stats(mask, fwd[sym], ok_win)
        if np.isfinite(edge):
            edges.append(edge)
        if np.isfinite(coverage):
            coverages.append(coverage)
    edge_mean     = float(np.mean(edges)) if edges else np.nan
    coverage_mean = float(np.mean(coverages)) if coverages else np.nan
    return edge_mean, coverage_mean


def _holdout_gate_for_rule(members: tuple, edge_selection: float, mask_cache: dict, fwd: dict,
                            ok_fwd: dict, window_ids: dict, holdout_ids: list) -> dict:
    sign_selection = 1.0 if edge_selection >= 0 else -1.0

    windows = []
    for wid in holdout_ids:
        edge, coverage = _rule_edge_in_window(members, mask_cache, fwd, ok_fwd, window_ids, wid)
        passed = (np.isfinite(edge)
                  and np.sign(edge) == sign_selection
                  and abs(edge) >= MAGNITUDE_FRAC * abs(edge_selection))
        windows.append({"window_id": int(wid), "edge": edge, "coverage": coverage, "passed": passed})

    n_pass   = sum(w["passed"] for w in windows)
    required = math.ceil(HOLDOUT_MIN_PASS_FRAC * len(holdout_ids))
    return {"holdout_windows": windows, "holdout_n_pass": n_pass,
            "holdout_n_total": len(holdout_ids), "holdout_pass": n_pass >= required}


def apply_holdout_gate(top: pd.DataFrame, mask_cache: dict, fwd: dict, ok_fwd: dict,
                        window_ids: dict, holdout_ids: list) -> pd.DataFrame:
    if top.empty:
        top = top.copy()
        top["holdout_n_pass"]  = pd.Series(dtype=int)
        top["holdout_n_total"] = pd.Series(dtype=int)
        top["holdout_pass"]    = pd.Series(dtype=bool)
        return top

    results = top.apply(
        lambda row: _holdout_gate_for_rule(
            row["members"], row["edge_mean"], mask_cache, fwd, ok_fwd, window_ids, holdout_ids
        ), axis=1
    )
    top = top.copy()
    top["holdout_n_pass"]  = [r["holdout_n_pass"] for r in results]
    top["holdout_n_total"] = [r["holdout_n_total"] for r in results]
    top["holdout_pass"]    = [r["holdout_pass"] for r in results]
    return top[top["holdout_pass"]].reset_index(drop=True)

# =============================================================================
# NON-OVERLAP EDGE — re-measures a rule's edge on independent, non-overlapping
# =============================================================================
NONOVERLAP_N_OFFSETS = 4  # starting offsets averaged for the non-overlap edge


def _nonoverlap_grid(n: int, h: int, offset: int) -> np.ndarray:
    """Boolean mask, True only every h bars starting at `offset`."""
    grid = np.zeros(n, dtype=bool)
    grid[offset::h] = True
    return grid


def _edge_stats_nonoverlap(mask: np.ndarray, fwd: np.ndarray, ok: np.ndarray,
                            h: int, n_offsets: int = NONOVERLAP_N_OFFSETS) -> tuple:

    n = len(mask)
    edges, coverages = [], []
    step = max(1, h // n_offsets)
    for offset in range(0, h, step):
        grid = _nonoverlap_grid(n, h, offset)
        edge, coverage = _edge_stats(mask, fwd, ok & grid)
        if np.isfinite(edge):
            edges.append(edge)
        if np.isfinite(coverage):
            coverages.append(coverage)
    edge_mean     = float(np.mean(edges)) if edges else np.nan
    coverage_mean = float(np.mean(coverages)) if coverages else np.nan
    return edge_mean, coverage_mean


def _rule_edge_nonoverlap(members: tuple, mask_cache: dict, fwd: dict, ok_fwd: dict,
                           h: int) -> float:
    """Cross-symbol mean of the non-overlap edge for one rule."""
    edges = []
    for sym, masks in mask_cache.items():
        mask = _combo_mask(members, masks)
        edge, _ = _edge_stats_nonoverlap(mask, fwd[sym], ok_fwd[sym], h)
        if np.isfinite(edge):
            edges.append(edge)
    return float(np.mean(edges)) if edges else np.nan


def _nonoverlap_gate_for_rule(members: tuple, edge_selection: float, mask_cache: dict,
                               fwd: dict, ok_fwd: dict, h: int,
                               magnitude_frac: float = MAGNITUDE_FRAC) -> dict:

    sign_selection  = 1.0 if edge_selection >= 0 else -1.0
    edge_nonoverlap = _rule_edge_nonoverlap(members, mask_cache, fwd, ok_fwd, h)
    passed = (np.isfinite(edge_nonoverlap)
              and np.sign(edge_nonoverlap) == sign_selection
              and abs(edge_nonoverlap) >= magnitude_frac * abs(edge_selection))
    return {"edge_nonoverlap": edge_nonoverlap, "nonoverlap_pass": passed}


def apply_nonoverlap_gate(top: pd.DataFrame, mask_cache: dict, fwd: dict, ok_fwd: dict,
                           h: int) -> pd.DataFrame:

    if top.empty:
        top = top.copy()
        top["edge_nonoverlap"] = pd.Series(dtype=float)
        top["nonoverlap_pass"] = pd.Series(dtype=bool)
        return top

    results = top.apply(
        lambda row: _nonoverlap_gate_for_rule(
            row["members"], row["edge_mean"], mask_cache, fwd, ok_fwd, h
        ), axis=1
    )
    top = top.copy()
    top["edge_nonoverlap"] = [r["edge_nonoverlap"] for r in results]
    top["nonoverlap_pass"] = [r["nonoverlap_pass"] for r in results]
    return top[top["nonoverlap_pass"]].reset_index(drop=True)

# =============================================================================
# SHORTLIST PIPELINE — top_nonredundant_rules + both hard gates, in the fixed
# =============================================================================
def build_validated_shortlist(df: pd.DataFrame, specs: list, panels: dict, mask_cache: dict,
                               fwd: dict, ok_fwd: dict, window_ids: dict, holdout_ids: list,
                               horizon: int, depth: int = None, log_fn=None) -> pd.DataFrame:
    top, floor = top_nonredundant_rules(df, specs, panels, RULE_POOL_SIZE, TOP_K_GROWTH, PHI_THRESHOLD)

    n_before_holdout = len(top)
    top = apply_holdout_gate(top, mask_cache, fwd, ok_fwd, window_ids, holdout_ids)
    if log_fn:
        holdout_required = math.ceil(HOLDOUT_MIN_PASS_FRAC * len(holdout_ids))
        log_fn(f"  DEPTH {depth}: holdout gate ({len(holdout_ids)} windows, "
               f">= {holdout_required}/{len(holdout_ids)} must pass): "
               f"{n_before_holdout} -> {len(top)} rules")

    n_before_nonoverlap = len(top)
    top = apply_nonoverlap_gate(top, mask_cache, fwd, ok_fwd, horizon)
    if log_fn:
        log_fn(f"  DEPTH {depth}: non-overlap gate (h={horizon}, "
               f"{NONOVERLAP_N_OFFSETS} offsets averaged): "
               f"{n_before_nonoverlap} -> {len(top)} rules")

    return top, floor

def annotate_nonoverlap_edge(shortlist: pd.DataFrame, mask_cache: dict, fwd: dict,
                              ok_fwd: dict, h: int) -> pd.DataFrame:

    shortlist = shortlist.copy()
    if shortlist.empty:
        shortlist["edge_nonoverlap"] = pd.Series(dtype=float)
        return shortlist

    shortlist["edge_nonoverlap"] = shortlist["members"].apply(
        lambda members: _rule_edge_nonoverlap(members, mask_cache, fwd, ok_fwd, h)
    )
    return shortlist

# =============================================================================
# REPORTING
# =============================================================================
def hr(char="─", debug=False):
    (logger.debug if debug else logger.info)(char * 110)

def format_rules(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["edge"]     = t["edge_mean"].round(4)
    t["coverage"] = (t["coverage_mean"] * 100).round(1).astype(str) + "%"
    t["pairs"]    = (t["pair_cons"] * 100).round(0).astype("Int64").astype(str) + "%"
    t["time"]     = (t["time_cons"] * 100).round(0).astype("Int64").astype(str) + "%"
    t["pass"]     = t["passed"].map({True: "✅", False: "❌"})
    t["junk"]     = t["junk"].map({True: "🚨", False: "–"})
    return t[["label", "depth", "edge", "coverage", "pairs", "time", "pass", "junk"]]

def format_rates(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["success_rate"] = (t["success_rate"] * 100).round(1).astype(str) + "%"
    t["best_edge"]     = t["best_edge"].round(4)
    t["above_floor"]   = t["above_floor"].map({True: "✅", False: "❌"})
    t["junk"]          = t["junk"].map({True: "🚨", False: "–"})
    return t[["indicator", "n_rules", "n_passed", "success_rate", "best_edge", "above_floor", "junk"]]