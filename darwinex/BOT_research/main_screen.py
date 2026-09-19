#BOT_batch_BZ/main_screen.py (forex)
"""
Indicator pre-screening before backtesting.

Decides which indicators from indicators_pool go to the backtest, aggregated (all symbols
together) and per symbol. Console output only.

  1. Target: trade opened at the close of every bar, TP/SL in % of the entry price, result
     in % (+TP, -SL, or the return at SELL_AFTER). Long and short per config.
     Configs = Cartesian product TP_PCT x SL_PCT x SELL_AFTER. TP or SL = 0: barrier disabled.
  2. Edge per segment (x <= c, x > c), with c at the per-symbol deciles q10..q90.
     T_symbol = max over params, cuts, segments, directions and configs.
     T_aggregate = max of Stouffer (sum of edges / sqrt(valid symbols)).
  3. Per-indicator null: circular shift of the targets (same k for every symbol and target).
     Floor = NULL_PCT percentile of the null. Beats the floor if T > floor.
  4. Stability: winning combination fixed, edge > 0 in >= MIN_BLOCKS_OK of N_BLOCKS
     contiguous time blocks.
  5. Check: N_FAKE fake indicators (real ones, shifted in time) through the same pre-screen.

Fully deterministic: null shifts come from SEED and fake indicators use no randomness.
"""
import os
import sys
import time
import logging
import warnings
import numpy as np
from numba import njit, prange

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import (
    CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key, implied_side,
)
LOG_LEVEL      = logging.INFO   # logging.DEBUG: legend, target stats, common bars, per-symbol matrix and detail, fake check table
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.screening")
logging.getLogger("numba").setLevel(logging.WARNING)

# =============================================================================
# CONFIG
# =============================================================================
DATASET   = "IS"
TIMEFRAME = "4H"
SYMBOLS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "EURJPY", "GBPJPY", "EURGBP", "EURCHF", "AUDJPY", "CADJPY", "CHFJPY",
           "EURAUD", "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF"]
# Target grids. Configs = Cartesian product TP_PCT x SL_PCT x SELL_AFTER, each one long and short.
#   TP_PCT, SL_PCT : TP and SL in % of the entry price.
#   SELL_AFTER     : max trade horizon, in bars. If no barrier is hit, exit at the close of t+SELL_AFTER.
# A value of 0 in TP_PCT or SL_PCT disables that barrier:
#   TP=0, SL>0 : only SL; if it is not hit, exit at t+SELL_AFTER.
#   TP>0, SL=0 : only TP; if it is not hit, exit at t+SELL_AFTER.
#   TP=0, SL=0 : no barriers, the target is the pure return at t+SELL_AFTER (edge measured by horizon only).
# With one value per grid it is a single config (e.g. [100], [0.5], [0.5] = TP 0.5% / SL 0.5% at 100 bars).
SELL_AFTER     = [10, 100]
TP_PCT         = [0.5, 1.0]
SL_PCT         = [0.5, 1.0]

N_NULL         = 1000   # null shifts per indicator
NULL_PCT       = 80     # null percentile used as floor

N_BLOCKS       = 4      # contiguous time blocks for the stability check
MIN_BLOCKS_OK  = 3      # blocks with edge > 0 required to pass

N_FAKE         = 20     # fake indicators for the check
FAKE_MIN_SHIFT = 1000   # fake indicators are shifted by more than this many bars

GROUP_N        = 5      # symbol pool: indicators passing in >= GROUP_N symbols, with the symbols where they pass

# Grid checks
if not (SELL_AFTER and TP_PCT and SL_PCT):
    raise ValueError("SELL_AFTER, TP_PCT and SL_PCT must have at least one value")
if any(int(v) != v or v < 1 for v in SELL_AFTER):
    raise ValueError(f"SELL_AFTER must be integers >= 1: {SELL_AFTER}")
if any(not np.isfinite(v) or v < 0 for v in list(TP_PCT) + list(SL_PCT)):
    raise ValueError(f"TP_PCT and SL_PCT must be finite and >= 0: {TP_PCT}, {SL_PCT}")
for _nm, _grid in (("SELL_AFTER", SELL_AFTER), ("TP_PCT", TP_PCT), ("SL_PCT", SL_PCT)):
    if len(set(_grid)) != len(_grid):
        raise ValueError(f"{_nm} has duplicated values: {_grid}")

# Derived (do not edit)
L_SHIFT = max(SELL_AFTER) + 100      # null k in [L_SHIFT, N - L_SHIFT]
DECILES = np.arange(1, 10) / 10.0    # q10..q90
NCUT    = 9                          # number of cuts
NBIN    = NCUT + 1                   # elementary segments between cuts
NAT_I64 = np.iinfo(np.int64).min     # NaT viewed as int64
SEED    = 42
CONFIGS = [(float(tp), float(sl), int(sa)) for tp in TP_PCT for sl in SL_PCT for sa in SELL_AFTER]
TARGETS = [(d, tp, sl, sa) for tp, sl, sa in CONFIGS for d in ("long", "short")]  # index g

# Pass marks: 2 = pass, 1 = beats floor but unstable, 0 = fail
_MARKS  = {2: "✅", 1: "⚠️", 0: "❌"}
_WIDTHS = {2: 1, 1: 2, 0: 1}                    # terminal columns (the emoji takes 2)

# =============================================================================
# 1. TARGET
# =============================================================================
@njit(cache=True)
def compute_targets(close, high, low, high_time, low_time, tp_pct, sl_pct, sell_after):

    n = close.shape[0]
    out_long = np.full(n, np.nan)
    out_short = np.full(n, np.nan)
    use_tp = tp_pct > 0.0
    use_sl = sl_pct > 0.0
    n_scan = sell_after if (use_tp or use_sl) else 0   # no barriers: skip the bar scan

    for t in range(n - sell_after):
        entry = close[t]
        if not (entry > 0.0):
            continue
        last = t + sell_after

        # --- long: TP above, SL below
        tp_px = entry * (1.0 + tp_pct / 100.0)
        sl_px = entry * (1.0 - sl_pct / 100.0)
        res = np.nan
        for j in range(t + 1, t + n_scan + 1):
            hit_tp = use_tp and high[j] >= tp_px
            hit_sl = use_sl and low[j] <= sl_px
            if hit_tp and hit_sl:
                ht, lt = high_time[j], low_time[j]
                if ht != NAT_I64 and lt != NAT_I64 and ht < lt:
                    res = tp_pct
                else:
                    res = -sl_pct
                break
            if hit_tp:
                res = tp_pct
                break
            if hit_sl:
                res = -sl_pct
                break
        if res != res:
            res = (close[last] / entry - 1.0) * 100.0
        out_long[t] = res

        # --- short: TP below, SL above
        tp_px = entry * (1.0 - tp_pct / 100.0)
        sl_px = entry * (1.0 + sl_pct / 100.0)
        res = np.nan
        for j in range(t + 1, t + n_scan + 1):
            hit_tp = use_tp and low[j] <= tp_px
            hit_sl = use_sl and high[j] >= sl_px
            if hit_tp and hit_sl:
                ht, lt = high_time[j], low_time[j]
                if ht != NAT_I64 and lt != NAT_I64 and lt < ht:
                    res = tp_pct
                else:
                    res = -sl_pct
                break
            if hit_tp:
                res = tp_pct
                break
            if hit_sl:
                res = -sl_pct
                break
        if res != res:
            res = (1.0 - close[last] / entry) * 100.0
        out_short[t] = res

    return out_long, out_short


# =============================================================================
# 2-3. EDGE, T AND NULL
# =============================================================================
@njit(cache=True)
def _fill_edges(bins, Y, k, out):
    """Edges with targets circularly shifted by k bars: y'[t] = y[(t + k) % N]."""
    n_inst, n_sym, n = bins.shape
    n_tg = Y.shape[2]
    cnt = np.empty((NBIN, n_tg))
    sm = np.empty((NBIN, n_tg))
    sq = np.empty((NBIN, n_tg))

    for i in range(n_inst):
        for s in range(n_sym):
            cnt[:, :] = 0.0
            sm[:, :] = 0.0
            sq[:, :] = 0.0
            for t in range(n):
                b = bins[i, s, t]
                if b < 0:
                    continue
                j = t + k
                if j >= n:
                    j -= n
                for g in range(n_tg):
                    y = Y[s, j, g]
                    if y == y:
                        cnt[b, g] += 1.0
                        sm[b, g] += y
                        sq[b, g] += y * y

            for g in range(n_tg):
                n_all = 0.0
                s_all = 0.0
                q_all = 0.0
                for b in range(NBIN):
                    n_all += cnt[b, g]
                    s_all += sm[b, g]
                    q_all += sq[b, g]
                mean = 0.0
                sd = 0.0
                if n_all > 1.0:
                    mean = s_all / n_all
                    var = q_all / n_all - mean * mean
                    if var > 0.0:
                        sd = np.sqrt(var)
                if sd <= 0.0:
                    for c in range(NCUT):
                        out[i, s, g, c, 0] = np.nan
                        out[i, s, g, c, 1] = np.nan
                    continue

                n_lo = 0.0
                s_lo = 0.0
                for c in range(NCUT):
                    n_lo += cnt[c, g]
                    s_lo += sm[c, g]
                    n_hi = n_all - n_lo
                    s_hi = s_all - s_lo
                    if n_lo > 0.0:
                        out[i, s, g, c, 0] = (s_lo / n_lo - mean) / (sd / np.sqrt(n_lo))
                    else:
                        out[i, s, g, c, 0] = np.nan
                    if n_hi > 0.0:
                        out[i, s, g, c, 1] = (s_hi / n_hi - mean) / (sd / np.sqrt(n_hi))
                    else:
                        out[i, s, g, c, 1] = np.nan


@njit(cache=True)
def _reduce_edges(edges):

    n_inst, n_sym, n_tg, n_cut, _ = edges.shape
    t_sym = np.full(n_sym, -np.inf)
    arg_sym = np.full(n_sym, -1, dtype=np.int64)
    t_agg = -np.inf
    arg_agg = -1

    for i in range(n_inst):
        for g in range(n_tg):
            for c in range(n_cut):
                for side in range(2):
                    flat = ((i * n_tg + g) * n_cut + c) * 2 + side
                    tot = 0.0
                    n_ok = 0
                    for s in range(n_sym):
                        v = edges[i, s, g, c, side]
                        if v == v:
                            tot += v
                            n_ok += 1
                            if v > t_sym[s]:
                                t_sym[s] = v
                                arg_sym[s] = flat
                    if n_ok > 0:
                        z = tot / np.sqrt(n_ok)
                        if z > t_agg:
                            t_agg = z
                            arg_agg = flat
    return t_agg, t_sym, arg_agg, arg_sym


@njit(parallel=True, cache=True)
def _null_distribution(bins, Y, shifts):
    """Aggregate T and per-symbol T for every null shift."""
    n_inst, n_sym, _ = bins.shape
    n_tg = Y.shape[2]
    n_k = shifts.shape[0]
    t_agg = np.empty(n_k)
    t_sym = np.empty((n_k, n_sym))
    for q in prange(n_k):
        edges = np.empty((n_inst, n_sym, n_tg, NCUT, 2))
        _fill_edges(bins, Y, shifts[q], edges)
        ta, ts, _a, _b = _reduce_edges(edges)
        t_agg[q] = ta
        for s in range(n_sym):
            t_sym[q, s] = ts[s]
    return t_agg, t_sym


# =============================================================================
# 4. STABILITY
# =============================================================================
@njit(cache=True)
def _edge_in_range(brow, y, start, end, cut, side):
    """Edge of a fixed segment using only bars [start, end) (block's own mean/std)."""
    cnt = np.zeros(NBIN)
    sm = np.zeros(NBIN)
    sq = np.zeros(NBIN)
    for t in range(start, end):
        b = brow[t]
        if b < 0:
            continue
        v = y[t]
        if v != v:
            continue
        cnt[b] += 1.0
        sm[b] += v
        sq[b] += v * v

    n_all = cnt.sum()
    if n_all <= 1.0:
        return np.nan
    s_all = sm.sum()
    mean = s_all / n_all
    var = sq.sum() / n_all - mean * mean
    if var <= 0.0:
        return np.nan

    n_lo = cnt[:cut + 1].sum()
    s_lo = sm[:cut + 1].sum()
    if side == 0:
        n_tr, s_tr = n_lo, s_lo
    else:
        n_tr, s_tr = n_all - n_lo, s_all - s_lo
    if n_tr <= 0.0:
        return np.nan
    return (s_tr / n_tr - mean) / (np.sqrt(var) / np.sqrt(n_tr))


def _blocks_ok_symbol(bins, Y, s, win, blocks):
    i, g, c, side = win
    y = np.ascontiguousarray(Y[s, :, g])
    return sum(1 for a, z in blocks if _edge_in_range(bins[i, s], y, a, z, c, side) > 0.0)


def _blocks_ok_aggregate(bins, Y, win, blocks):
    i, g, c, side = win
    ys = [np.ascontiguousarray(Y[s, :, g]) for s in range(Y.shape[0])]
    ok = 0
    for a, z in blocks:
        e = np.array([_edge_in_range(bins[i, s], ys[s], a, z, c, side) for s in range(Y.shape[0])])
        e = e[np.isfinite(e)]
        if len(e) > 0 and e.sum() / np.sqrt(len(e)) > 0.0:
            ok += 1
    return ok


# =============================================================================
# DATA
# =============================================================================
def _i64(a):
    return np.asarray(a).astype("datetime64[ns]").view("int64")


def load_aligned():
    """Loads the symbols and aligns them on the bars shared by all (exact timestamp intersection)."""
    ohlcv = build_universe(DATA_FOLDER_BY_DATASET[DATASET], {TIMEFRAME: SYMBOLS}, dataset=DATASET)[TIMEFRAME]
    ohlcv_arr = prepare_ohlcv_arrays(ohlcv)

    missing = [s for s in SYMBOLS if s not in ohlcv_arr]
    if missing:
        raise ValueError(f"Symbols missing from data: {missing}")

    ts = {}
    for s in SYMBOLS:
        t = _i64(ohlcv_arr[s]["ts"])
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            raise ValueError(f"{s}: timestamps are not strictly increasing")
        ts[s] = t

    grid = ts[SYMBOLS[0]]
    for s in SYMBOLS[1:]:
        grid = np.intersect1d(grid, ts[s], assume_unique=True)
    pos = {s: np.searchsorted(ts[s], grid) for s in SYMBOLS}
    native = {s: len(ts[s]) for s in SYMBOLS}
    return ohlcv_arr, grid, pos, native


def build_targets(ohlcv_arr, pos, n):
    """Targets on each symbol's native bars, mapped to the common grid. Y[s, t, g]."""
    Y = np.full((len(SYMBOLS), n, len(TARGETS)), np.nan)
    for si, s in enumerate(SYMBOLS):
        arr = ohlcv_arr[s]
        close = np.ascontiguousarray(arr["close"], dtype=np.float64)
        high = np.ascontiguousarray(arr["high"], dtype=np.float64)
        low = np.ascontiguousarray(arr["low"], dtype=np.float64)
        ht = np.ascontiguousarray(_i64(arr["high_time"]))
        lt = np.ascontiguousarray(_i64(arr["low_time"]))
        for ci, (tp, sl, sa) in enumerate(CONFIGS):
            y_long, y_short = compute_targets(close, high, low, ht, lt, tp, sl, sa)
            Y[si, :, 2 * ci] = y_long[pos[s]]
            Y[si, :, 2 * ci + 1] = y_short[pos[s]]
    return Y


def build_bins(ohlcv_arr, pos, n, instances):
    """Indicator on native bars, mapped to the common grid and turned into decile segments (per symbol)."""
    bins = np.full((len(instances), len(SYMBOLS), n), -1, dtype=np.int8)
    cuts = np.full((len(instances), len(SYMBOLS), NCUT), np.nan)
    empty = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for ii, inst in enumerate(instances):
            fn = CANDIDATE_REGISTRY[inst["indicator"]]["fn"]
            for si, s in enumerate(SYMBOLS):
                x = np.asarray(fn(ohlcv_arr[s], None, inst["params"]), dtype=np.float64)[pos[s]]
                ok = np.isfinite(x)
                if not ok.any():
                    empty.append((instance_key(inst["indicator"], inst["params"]), s))
                    continue
                c = np.quantile(x[ok], DECILES)
                cuts[ii, si] = c
                bins[ii, si, ok] = np.searchsorted(c, x[ok], side="left")
    return bins, cuts, empty


# =============================================================================
# SCREENING OF ONE INDICATOR
# =============================================================================
def screen_indicator(bins, Y, shifts, blocks):
    """bins: (indicator instances, symbols, N). Returns T, floor, blocks and winner, aggregate and per symbol."""
    n_inst, n_sym, _ = bins.shape
    shape = (n_inst, len(TARGETS), NCUT, 2)

    edges = np.empty((n_inst, n_sym, len(TARGETS), NCUT, 2))
    _fill_edges(bins, Y, 0, edges)
    t_agg, t_sym, arg_agg, arg_sym = _reduce_edges(edges)

    null_agg, null_sym = _null_distribution(bins, Y, shifts)
    with np.errstate(invalid="ignore"):
        floor_agg = float(np.percentile(null_agg, NULL_PCT))
        floor_sym = np.percentile(null_sym, NULL_PCT, axis=0)

    win_agg, blk_agg = None, 0
    if arg_agg >= 0:
        win_agg = tuple(int(v) for v in np.unravel_index(arg_agg, shape))
        blk_agg = _blocks_ok_aggregate(bins, Y, win_agg, blocks)

    win_sym = [None] * n_sym
    blk_sym = np.zeros(n_sym, dtype=int)
    for s in range(n_sym):
        if arg_sym[s] >= 0:
            win_sym[s] = tuple(int(v) for v in np.unravel_index(arg_sym[s], shape))
            blk_sym[s] = _blocks_ok_symbol(bins, Y, s, win_sym[s], blocks)

    over_agg = bool(t_agg > floor_agg)
    over_sym = t_sym > floor_sym
    return {
        "T_agg": float(t_agg), "floor_agg": floor_agg, "blocks_agg": blk_agg, "win_agg": win_agg,
        "over_agg": over_agg, "pass_agg": bool(over_agg and blk_agg >= MIN_BLOCKS_OK),
        "T_sym": t_sym, "floor_sym": floor_sym, "blocks_sym": blk_sym, "win_sym": win_sym,
        "over_sym": over_sym, "pass_sym": over_sym & (blk_sym >= MIN_BLOCKS_OK),
    }


def describe_win(name, inst_idx, instances, cuts, win, sym=None):
    """Readable winning combination. In aggregate the cut shown is the median across symbols."""
    i, g, c, side = win
    ii = inst_idx[i]
    inst = instances[ii]
    key = instance_key(inst["indicator"], inst["params"])
    params = key[len(name) + 2:] if key.startswith(name + "__") else "-"
    direction, tp, sl, sa = TARGETS[g]
    cval = float(np.nanmedian(cuts[ii, :, c])) if sym is None else float(cuts[ii, sym, c])
    rule = f"x {'<=' if side == 0 else '>'} q{10 * (c + 1)}"
    flag = False
    if CANDIDATE_REGISTRY[name]["directional"]:
        imp = implied_side({"indicator": name, "op": "<" if side == 0 else ">", "threshold": cval})
        flag = imp is not None and imp != direction
    return {"params": params, "rule": rule, "cut": cval, "dir": direction,
            "tpsl": f"{tp:g}/{sl:g}", "sa": sa, "flag": flag}


# =============================================================================
# OUTPUT
# =============================================================================
SEP = "=" * 124


def _mark(state, width=1):
    """Pass mark (2 pass, 1 beats floor but unstable, 0 fail), padded to width."""
    return _MARKS[state] + " " * max(width - _WIDTHS[state], 0)


def _state(over, passed):
    return 2 if passed else (1 if over else 0)


def _fmt(v, w=6):
    return f"{v:>{w}.2f}" if np.isfinite(v) else f"{'-':>{w}}"


def _group(name):
    return GROUP_NAMES.get(CANDIDATE_REGISTRY[name]["group"], CANDIDATE_REGISTRY[name]["group"])


def log_legend():
    logger.debug(SEP)
    logger.debug("LEGEND")
    logger.debug(SEP)
    logger.debug("  T      best edge of the indicator: max over params, cuts (q10..q90), segments (x <= c, x > c),")
    logger.debug("         directions and TP/SL/SA configs. edge = (segment mean - mean) / (std / sqrt(n_segment)).")
    logger.debug("         Aggregate T: Stouffer = sum of per-symbol edges / sqrt(n symbols), same rule in every symbol.")
    logger.debug(f"  floor  p{NULL_PCT} of T with the targets circularly shifted (no real relation). T > floor: beats chance.")
    logger.debug(f"  blk    time blocks (of {N_BLOCKS}) where the winning rule keeps edge > 0.")
    logger.debug(f"  pass   {_mark(2)} T > floor and blk >= {MIN_BLOCKS_OK}   {_mark(1)} T > floor but blk < {MIN_BLOCKS_OK}"
                 f"   {_mark(0)} T <= floor")
    logger.debug("  rule   winning cut as a decile (aggregate: cut value = median across symbols). TP/SL/SA is informative only.")
    logger.debug("  SA     sell_after: max trade horizon in bars (exit at the close of t+SA). TP or SL = 0: barrier disabled.")
    logger.debug("  flag   OPPOSITE: directional indicator winning on the side opposite to implied_side.")


def _table_header():
    return (f"{'indicator':<26}{'group':<20}{'T':>7}{'floor':>7}{'blk':>6}  {'pass':<5}"
            f"{'params':<30}{'rule':<10}{'cut':>10}  {'side':<6}{'TP/SL':<8}{'SA':<6}flag")


def _table_row(name, T, floor, blk, over, passed, d):
    head = f"{name:<26}{_group(name):<20}{_fmt(T, 7)}{_fmt(floor, 7)}"
    if d is None:
        return head + f"{'-':>6}  {_mark(0, 5)}"
    blk_s = f"{blk}/{N_BLOCKS}"
    return (head + f"{blk_s:>6}  {_mark(_state(over, passed), 5)}{d['params']:<30}{d['rule']:<10}"
            f"{d['cut']:>10.4g}  {d['dir']:<6}{d['tpsl']:<8}{d['sa']:<6}{'OPPOSITE' if d['flag'] else ''}")


def report(names, by_ind, instances, cuts, results, fakes):
    n_sym = len(SYMBOLS)

    # --- aggregate pool
    pool = [nm for nm in names if results[nm]["pass_agg"]]
    logger.info(f"\n{SEP}\nAGGREGATE POOL ({len(pool)}/{len(names)})\n{SEP}")
    logger.info(", ".join(pool) if pool else "(empty)")

    # --- aggregate table
    logger.info(f"\n{SEP}\nAGGREGATE TABLE\n{SEP}")
    logger.info(_table_header())
    order = sorted(names, key=lambda nm: (not results[nm]["pass_agg"],
                                          -(results[nm]["T_agg"] - results[nm]["floor_agg"])))
    for nm in order:
        r = results[nm]
        d = describe_win(nm, by_ind[nm], instances, cuts, r["win_agg"]) if r["win_agg"] else None
        logger.info(_table_row(nm, r["T_agg"], r["floor_agg"], r["blocks_agg"], r["over_agg"], r["pass_agg"], d))
    flagged = [nm for nm in names if results[nm]["win_agg"]
               and describe_win(nm, by_ind[nm], instances, cuts, results[nm]["win_agg"])["flag"]]
    logger.info(f"\nAggregate flags (directional winning on the opposite side): "
                f"{', '.join(flagged) if flagged else 'none'}")

    # --- symbol pool
    logger.info(f"\n{SEP}\nSYMBOL POOL  ((!) = directional winning on the side opposite to implied_side)\n{SEP}")
    for s, sym in enumerate(SYMBOLS):
        items = []
        for nm in names:
            r = results[nm]
            if r["pass_sym"][s]:
                d = describe_win(nm, by_ind[nm], instances, cuts, r["win_sym"][s], sym=s)
                items.append(nm + (" (!)" if d["flag"] else ""))
        logger.info(f"{sym} ({len(items)}): " + (", ".join(items) if items else "(empty)"))
    in_all = [nm for nm in names if results[nm]["pass_sym"].all()]
    logger.info(f"In all symbols ({len(in_all)}): " + (", ".join(in_all) if in_all else "(empty)"))

    # --- indicators passing in at least GROUP_N symbols, with the symbols where they pass
    if 0 < GROUP_N <= n_sym:
        rows = [(nm, [SYMBOLS[s] for s in range(n_sym) if results[nm]["pass_sym"][s]]) for nm in names]
        rows = [(nm, syms) for nm, syms in rows if len(syms) >= GROUP_N]
        rows.sort(key=lambda r: -len(r[1]))   # stable: ties keep indicator order
        logger.info(f"\nIndicators passing in >= {GROUP_N} symbols ({len(rows)}):")
        for nm, syms in rows:
            logger.info(f"  {nm} ({len(syms)}): {', '.join(syms)}")
        if not rows:
            logger.info("  (empty)")

    # --- per-symbol matrix (debug)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\n{SEP}\nPER-SYMBOL MATRIX  ({_mark(2)} pass | {_mark(1)} beats floor but unstable | "
                     f"{_mark(0)} fail)\n{SEP}")
        logger.debug(f"{'indicator':<26}{'agg':<5}" + "".join(f"{sym[:6]:<7}" for sym in SYMBOLS) + "n_pass")
        for nm in names:
            r = results[nm]
            agg = _mark(_state(r["over_agg"], r["pass_agg"]), 5)
            cells = "".join(_mark(_state(r["over_sym"][s], r["pass_sym"][s]), 7) for s in range(n_sym))
            logger.debug(f"{nm:<26}{agg}{cells}{int(r['pass_sym'].sum())}")

    # --- per-symbol detail (debug)
    if logger.isEnabledFor(logging.DEBUG):
        for s, sym in enumerate(SYMBOLS):
            logger.debug(f"\n{SEP}\nDETAIL {sym}\n{SEP}")
            logger.debug(_table_header())
            for nm in names:
                r = results[nm]
                w = r["win_sym"][s]
                d = describe_win(nm, by_ind[nm], instances, cuts, w, sym=s) if w else None
                logger.debug(_table_row(nm, r["T_sym"][s], r["floor_sym"][s], r["blocks_sym"][s],
                                        r["over_sym"][s], r["pass_sym"][s], d))

    # --- check with fake indicators (table in debug, summary in info)
    expected = len(fakes) * (1.0 - NULL_PCT / 100.0)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\n{SEP}\nCHECK: {len(fakes)} fake indicators (real ones shifted > {FAKE_MIN_SHIFT} bars, "
                     f"not sent to backtest)\n{SEP}")
        logger.debug(f"{'fake':<26}{'shift':>8}{'T':>7}{'floor':>7}{'blk':>6}  {'>floor':<8}{'pass':<6}"
                     f"{'sym >floor':>11}{'sym pass':>10}")
        for nm, kf, r in fakes:
            blk_s = f"{r['blocks_agg']}/{N_BLOCKS}"
            logger.debug(f"{nm:<26}{kf:>8}{_fmt(r['T_agg'], 7)}{_fmt(r['floor_agg'], 7)}{blk_s:>6}  "
                         f"{_mark(2 if r['over_agg'] else 0, 8)}{_mark(2 if r['pass_agg'] else 0, 6)}"
                         f"{int(r['over_sym'].sum()):>11}{int(r['pass_sym'].sum()):>10}")

    over_agg = sum(r["over_agg"] for _, _, r in fakes)
    pass_agg = sum(r["pass_agg"] for _, _, r in fakes)
    over_sym = np.array([[r["over_sym"][s] for _, _, r in fakes] for s in range(n_sym)]).sum(axis=1)
    pass_sym = np.array([[r["pass_sym"][s] for _, _, r in fakes] for s in range(n_sym)]).sum(axis=1)
    logger.info(f"\nAggregate:  beat the floor {over_agg}/{len(fakes)} (expected ~{expected:.0f}), "
                f"pass with stability {pass_agg}/{len(fakes)}")
    logger.info(f"Per symbol: beat the floor mean {over_sym.mean():.1f}/{len(fakes)} "
                f"(min {over_sym.min()}, max {over_sym.max()}), pass mean {pass_sym.mean():.1f}/{len(fakes)}")

    # Warning if clearly above expectation (~p99 of a binomial(len(fakes), 1 - NULL_PCT/100))
    p = 1.0 - NULL_PCT / 100.0
    limit_one = expected + 2.33 * np.sqrt(len(fakes) * p * (1.0 - p))
    if over_agg > limit_one or over_sym.mean() > expected + 0.5 * (limit_one - expected):
        logger.info(f"{_mark(0)} WARNING: far more fake indicators beat the floor than expected. Check for a bug.")
    else:
        logger.info(f"{_mark(2)} OK: the rate of fake indicators beating the floor matches expectation.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    logger.info(f"Screening {DATASET} {TIMEFRAME} | {len(SYMBOLS)} symbols | TP_PCT={TP_PCT} SL_PCT={SL_PCT} "
                f"SELL_AFTER={SELL_AFTER} ({len(CONFIGS)} configs, {len(TARGETS)} targets) | "
                f"N_NULL={N_NULL} NULL_PCT={NULL_PCT} SEED={SEED}")
    log_legend()

    # --- data and common grid
    ohlcv_arr, grid, pos, native = load_aligned()
    n = len(grid)
    min_n = 2 * max(L_SHIFT, FAKE_MIN_SHIFT + 1) + N_FAKE + N_BLOCKS
    if n < min_n:
        raise ValueError(f"Only {n} common bars; at least {min_n} are needed")
    first, last = np.datetime64(int(grid[0]), "ns"), np.datetime64(int(grid[-1]), "ns")
    logger.debug(f"Common bars: {n} ({str(first)[:16]} .. {str(last)[:16]})")
    logger.debug("Native bars: " + ", ".join(f"{s} {native[s]}" for s in SYMBOLS))

    # --- targets
    Y = build_targets(ohlcv_arr, pos, n)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("\nTargets (common grid, all symbols):")
        for g, (d, tp, sl, sa) in enumerate(TARGETS):
            y = Y[:, :, g].ravel()
            y = y[np.isfinite(y)]
            head = f"  {d:<5} TP{tp:g}/SL{sl:g}/SA{sa}: n={len(y)} mean={y.mean():+.4f}%  "
            if tp == 0.0 and sl == 0.0:
                logger.debug(head + f"std={y.std():.4f}%  (no barriers, return at SA)")
                continue
            p_tp = 100.0 * np.mean(y == tp) if tp > 0.0 else 0.0
            p_sl = 100.0 * np.mean(y == -sl) if sl > 0.0 else 0.0
            s_tp = f"TP {p_tp:.1f}%" if tp > 0.0 else "TP off"
            s_sl = f"SL {p_sl:.1f}%" if sl > 0.0 else "SL off"
            logger.debug(head + f"{s_tp}  {s_sl}  timeout {100.0 - p_tp - p_sl:.1f}%")

    # --- indicators -> decile segments
    instances = build_flat_instances()
    by_ind = {}
    for ii, inst in enumerate(instances):
        by_ind.setdefault(inst["indicator"], []).append(ii)
    names = list(by_ind.keys())
    logger.info(f"Computing {len(names)} indicators, {len(instances)} instances...")
    t1 = time.time()
    bins, cuts, empty = build_bins(ohlcv_arr, pos, n, instances)
    logger.info(f"  done in {time.time() - t1:.0f}s")
    for key, s in empty:
        logger.info(f"  WARNING: {key} has no values in {s}")

    # --- null and blocks (deterministic)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(L_SHIFT, n - L_SHIFT, size=N_NULL, endpoint=True).astype(np.int64)
    bounds = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    blocks = [(int(bounds[b]), int(bounds[b + 1])) for b in range(N_BLOCKS)]

    # --- screening
    logger.info(f"Screening ({N_NULL} shifts per indicator)...")
    results = {}
    for k, nm in enumerate(names):
        t1 = time.time()
        r = screen_indicator(np.ascontiguousarray(bins[by_ind[nm]]), Y, shifts, blocks)
        results[nm] = r
        logger.info(f"  [{k + 1:>2}/{len(names)}] {nm:<26} T={_fmt(r['T_agg'])} floor={_fmt(r['floor_agg'])} "
                    f"blk={r['blocks_agg']}/{N_BLOCKS} agg={_mark(_state(r['over_agg'], r['pass_agg']))} "
                    f"symbols={int(r['pass_sym'].sum())}/{len(SYMBOLS)}  {time.time() - t1:.1f}s")

    # --- check: deterministic fakes (indicators spread over the registry, fixed shifts)
    n_fake = min(N_FAKE, len(names))
    fake_names = [names[(i * len(names)) // n_fake] for i in range(n_fake)]
    span = n - 2 * (FAKE_MIN_SHIFT + 1)
    fake_shifts = [FAKE_MIN_SHIFT + 1 + (i * span) // max(n_fake - 1, 1) for i in range(n_fake)]
    logger.info(f"Check ({n_fake} fake indicators)...")
    fakes = []
    for nm, kf in zip(fake_names, fake_shifts):
        b = np.ascontiguousarray(np.roll(bins[by_ind[nm]], kf, axis=2))
        fakes.append((nm, kf, screen_indicator(b, Y, shifts, blocks)))

    report(names, by_ind, instances, cuts, results, fakes)
    logger.info(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()