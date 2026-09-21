#quant_miner/darwinex/BOT_research/main_screen.py (forex)

import os
import sys
import time
import pickle
import hashlib
import inspect
import logging
import warnings
from datetime import datetime

import numba
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import CANDIDATE_REGISTRY, build_flat_instances, instance_key

from indicators.screening_engine import (
    i64, make_synthetic,
    MIN_SEG, MIN_PILOT_N, NCUT, compute_targets, fill_edges, reduce_edges, path_T_all,
    path_edges_all, add_edge_moments, moments_inplace,
    pair_T, pair_edge_moments, pair_valid_mask, swap_pair,
    NULL_BATCH, null_stop_count, pair_null_early,
    report_roles, progress,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.screening")
logger.setLevel(logging.INFO)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
N_JOBS         = -1
# =============================================================================
# CONFIG
# =============================================================================
#Selection: can change freely, applied on top of the same cache
GROUP_N        = 5     # symbols where an indicator (alone) or a pair must pass to be selected
NULL_PCT       = 80  # higher: reuses the cache; lower than the cache's: recomputes (phase 2 early stop)
USE_CACHE      = True   # True: reuse the raw results if nothing that affects them changed (else compute and save)

#Computation: changing any of these computes a new cache
DATASET   = "IS"
TIMEFRAME = "1H"
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]

SELL_AFTER     = [10, 100]
TP_PCT         = [0.5,1.0,1.5]
SL_PCT         = [0.5,1.0,1.5]

N_NULL_PATHS   = 1000    # null: synthetic paths of phase 1 = shifts per null and pair of phase 2 (build the floor)
N_PILOTS       = 200    # pilot: paths of phase 1 = shifts per pilot and pair of phase 2 (null mean and std of every combination, for z)
# The indicator pool (indicators_pool.py), the data and the code are also part of the cache fingerprint

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
if not (0 <= NULL_PCT <= 100):             # also sets the early stop of the phase 2 nulls
    raise ValueError(f"NULL_PCT must be in [0, 100]: {NULL_PCT}")
if not (0 < GROUP_N <= len(SYMBOLS)):      # the whole selection depends on it
    raise ValueError(f"GROUP_N must be in [1, {len(SYMBOLS)}]: {GROUP_N}")
if N_PILOTS < MIN_PILOT_N:                 # else no combination could compete
    raise ValueError(f"N_PILOTS must be >= {MIN_PILOT_N} (MIN_PILOT_N): {N_PILOTS}")

# Derived (do not edit)
L_SHIFT    = max(SELL_AFTER) + 100   # phase 2 shifts k in [L_SHIFT, N - L_SHIFT]
SEED_NULL  = 10_000                  # nulls: first synthetic path of phase 1, and draw of the phase 2 shifts
SEED_PILOT = 20_000                  # pilots: first synthetic path of phase 1, and draw of the phase 2 shifts
CONFIGS = [(float(tp), float(sl), int(sa)) for tp in TP_PCT for sl in SL_PCT for sa in SELL_AFTER]
TARGETS = [(d, tp, sl, sa) for tp, sl, sa in CONFIGS for d in ("long", "short")]  # index g
if SEED_PILOT < SEED_NULL + N_NULL_PATHS and SEED_NULL < SEED_PILOT + N_PILOTS:   # pilot and floor must not share paths
    raise ValueError(f"Pilot paths [{SEED_PILOT}, {SEED_PILOT + N_PILOTS}) overlap the floor paths "
                     f"[{SEED_NULL}, {SEED_NULL + N_NULL_PATHS})")

# Cache (do not edit)
CACHE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_cache")
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# =============================================================================
# DATA
# =============================================================================
def load_aligned():
    """Loads the symbols and aligns them on the candles shared by all (exact timestamp intersection)."""
    ohlcv = build_universe(DATA_FOLDER_BY_DATASET[DATASET], {TIMEFRAME: SYMBOLS}, dataset=DATASET)[TIMEFRAME]
    ohlcv_arr = prepare_ohlcv_arrays(ohlcv)

    missing = [s for s in SYMBOLS if s not in ohlcv_arr]
    if missing:
        raise ValueError(f"Symbols missing from data: {missing}")

    ts = {}
    for s in SYMBOLS:
        t = i64(ohlcv_arr[s]["ts"])
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            raise ValueError(f"{s}: timestamps are not strictly increasing")
        ts[s] = t

    grid = ts[SYMBOLS[0]]
    for s in SYMBOLS[1:]:
        grid = np.intersect1d(grid, ts[s], assume_unique=True)
    pos = {s: np.searchsorted(ts[s], grid) for s in SYMBOLS}
    return ohlcv_arr, grid, pos


def build_targets(ohlcv_arr, pos, n):
    """Targets on each symbol's native candles, mapped to the common grid. Y[s, t, g]."""
    Y = np.full((len(SYMBOLS), n, len(TARGETS)), np.nan)
    for si, s in enumerate(SYMBOLS):
        arr = ohlcv_arr[s]
        close = np.ascontiguousarray(arr["close"], dtype=np.float64)
        high = np.ascontiguousarray(arr["high"], dtype=np.float64)
        low = np.ascontiguousarray(arr["low"], dtype=np.float64)
        ht = np.ascontiguousarray(i64(arr["high_time"]))
        lt = np.ascontiguousarray(i64(arr["low_time"]))
        for ci, (tp, sl, sa) in enumerate(CONFIGS):
            y_long, y_short = compute_targets(close, high, low, ht, lt, tp, sl, sa)
            Y[si, :, 2 * ci] = y_long[pos[s]]
            Y[si, :, 2 * ci + 1] = y_short[pos[s]]
    return Y


def build_bins(ohlcv_arr, pos, n, instances):
    """Segment of every candle for every instance and symbol, at the registry thresholds (-1: no value).
    Also the number of cuts of every instance and the (instance, symbol) without any value."""
    bins = np.full((len(instances), len(SYMBOLS), n), -1, dtype=np.int8)
    ncv = np.zeros(len(instances), dtype=np.int64)
    empty = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for ii, inst in enumerate(instances):
            meta = CANDIDATE_REGISTRY[inst["indicator"]]
            c = np.array(sorted(set(float(v) for v in meta["thresholds"])), dtype=np.float64)
            ncv[ii] = len(c)
            for si, s in enumerate(SYMBOLS):
                x = np.asarray(meta["fn"](ohlcv_arr[s], None, inst["params"]), dtype=np.float64)[pos[s]]
                ok = np.isfinite(x)
                if not ok.any():
                    empty.append((instance_key(inst["indicator"], inst["params"]), s))
                    continue
                xo = x[ok]
                bins[ii, si, ok] = (np.searchsorted(c, xo, side="left") + np.searchsorted(c, xo, side="right"))
    return bins, ncv, empty


def _ind_slices(names, by_ind):
    """Slice [start, end) of the instances of every indicator (they must be contiguous)."""
    for nm in names:
        idx = by_ind[nm]
        if idx != list(range(idx[0], idx[-1] + 1)):
            raise ValueError(f"{nm}: its instances are not contiguous in build_flat_instances()")
    starts = np.array([by_ind[nm][0] for nm in names], dtype=np.int64)
    ends = np.array([by_ind[nm][-1] + 1 for nm in names], dtype=np.int64)
    return starts, ends


_WORKER_CFG = ("SYMBOLS", "TP_PCT", "SL_PCT", "SELL_AFTER", "CONFIGS", "TARGETS")


def _path_chunk(kind, seeds, cfg, ohlcv_arr, pos, n, instances, starts, ends, z_mu, z_sd, in_worker):
    """A chunk of synthetic paths (runs in a worker process). kind "pilot": the edges of every path (the parent
    adds them in path order); kind "null": the per-symbol T1 of every indicator on every path."""
    globals().update(cfg)                  # a worker may import this file fresh: use the parent's config
    if in_worker:
        numba.set_num_threads(1)           # the paths are the parallel unit: one core per worker
        if z_mu is not None:
            z_mu, z_sd = np.array(z_mu), np.array(z_sd)   # joblib hands big arrays over as read-only memmaps
    out = []
    for seed in seeds:
        syn = make_synthetic(ohlcv_arr, seed, SYMBOLS)
        Y_p = build_targets(syn, pos, n)
        bins_p, ncv_p, _e = build_bins(syn, pos, n, instances)
        if kind == "pilot":
            out.append(path_edges_all(bins_p, ncv_p, Y_p, starts, ends))
        else:
            out.append(path_T_all(bins_p, ncv_p, Y_p, starts, ends, z_mu, z_sd))
    return out


def _run_paths(kind, seeds, label, ohlcv_arr, pos, n, instances, starts, ends, z_mu=None, z_sd=None):
    """Result of every path, yielded in path order. Chunks of consecutive paths run in N_JOBS processes;
    every path keeps its seed, so the results do not depend on N_JOBS."""
    t0 = time.time()
    n_jobs = effective_n_jobs(N_JOBS)
    cfg = {k: globals()[k] for k in _WORKER_CFG}
    done = 0
    if n_jobs == 1:
        for seed in seeds:
            done += 1
            yield _path_chunk(kind, [seed], cfg, ohlcv_arr, pos, n, instances, starts, ends, z_mu, z_sd, False)[0]
            progress(label, done, len(seeds), t0)
        return
    chunks = [[int(v) for v in c] for c in np.array_split(np.asarray(seeds), min(len(seeds), 4 * n_jobs))]
    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_path_chunk)(kind, ch, cfg, ohlcv_arr, pos, n, instances, starts, ends, z_mu, z_sd, True)
        for ch in chunks)
    for res in results:
        for r in res:
            done += 1
            yield r
        progress(label, done, len(seeds), t0)


def build_pilot_moments(ohlcv_arr, pos, n, instances, names, by_ind):
    """Phase 1 pilot: null mean and std of the edge of every combination (instance, symbol, target, cut, side)
    over N_PILOTS synthetic paths of their own (seeds from SEED_PILOT, never the floor's).
    Returns z_mu, z_sd (shape of the edges of all the instances), NaN where a combination is valid in fewer
    than MIN_PILOT_N paths."""
    starts, ends = _ind_slices(names, by_ind)
    shape = (len(instances), len(SYMBOLS), len(TARGETS), NCUT, 2)
    m_n = np.zeros(shape)
    m_s = np.zeros(shape)
    m_q = np.zeros(shape)
    seeds = [SEED_PILOT + r for r in range(N_PILOTS)]
    label = f"Phase 1 pilot, {N_PILOTS} synthetic paths"
    for edges in _run_paths("pilot", seeds, label, ohlcv_arr, pos, n, instances, starts, ends):
        add_edge_moments(edges, m_n, m_s, m_q)             # in path order: same sums as a serial run
    moments_inplace(m_n.reshape(-1), m_s.reshape(-1), m_q.reshape(-1), float(MIN_PILOT_N))
    return m_s, m_q                            # m_s is now the mean and m_q the std


def build_null_paths(ohlcv_arr, pos, n, instances, names, by_ind, z_mu, z_sd):
    """Phase 1 null: per-symbol T1 of every indicator on every synthetic path. Raw values (N_NULL_PATHS, ind, sym)."""
    starts, ends = _ind_slices(names, by_ind)

    null_sym = np.empty((N_NULL_PATHS, len(names), len(SYMBOLS)))
    seeds = [SEED_NULL + r for r in range(N_NULL_PATHS)]
    label = f"Phase 1 null, {N_NULL_PATHS} synthetic paths"
    for r, t_sym in enumerate(_run_paths("null", seeds, label, ohlcv_arr, pos, n, instances, starts, ends,
                                         z_mu, z_sd)):
        null_sym[r] = t_sym
    return null_sym


def pilot_shifts(n, exclude):
    """N_PILOTS distinct shifts in [L_SHIFT, n - L_SHIFT], none of them in exclude (the null's)."""
    cand = np.setdiff1d(np.arange(L_SHIFT, n - L_SHIFT + 1, dtype=np.int64), np.asarray(exclude, dtype=np.int64))
    if len(cand) < N_PILOTS:
        raise ValueError(f"Only {len(cand)} free shifts for the phase 2 pilot, {N_PILOTS} are needed")
    return np.random.default_rng(SEED_PILOT).choice(cand, size=N_PILOTS, replace=False).astype(np.int64)


def phase2_shifts(n):
    """Phase 2 null shifts and pilot shifts (disjoint from them). The same ones for both nulls of every pair."""
    rng = np.random.default_rng(SEED_NULL)
    shifts = rng.integers(L_SHIFT, n - L_SHIFT, size=N_NULL_PATHS, endpoint=True).astype(np.int64)
    return shifts, pilot_shifts(n, shifts)


def _null_stats(null_sym):
    """Floor (p NULL_PCT), median and p84 of a raw null, along its first axis."""
    with np.errstate(invalid="ignore"):
        floor_sym = np.percentile(null_sym, NULL_PCT, axis=0)
        med_sym = np.percentile(null_sym, 50, axis=0)
        p84_sym = np.percentile(null_sym, 84, axis=0)
    return floor_sym, med_sym, p84_sym


# =============================================================================
# RANKING SCORE
# =============================================================================

def _scores(t_sym, med_sym, p84_sym):
    """Per-symbol score. NaN where the null has no spread (p84 == median)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = (t_sym - med_sym) / (p84_sym - med_sym)
    return np.where(np.isfinite(sc), sc, np.nan)


def _score(score_sym, pass_sym):
    """One number per indicator or pair: median of the score over the symbols where it passes."""
    v = score_sym[pass_sym]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def _median_ok(v, mask):
    """Median of v over mask, ignoring NaN. NaN if empty."""
    w = np.asarray(v)[mask]
    w = w[np.isfinite(w)]
    return float(np.median(w)) if w.size else np.nan


# =============================================================================
# SCREENING
# =============================================================================
def _seg_mask(brow, cut, side):
    """Candles of one instance row inside segment (cut, side)."""
    return ((brow >= 0) & (brow <= 2 * cut)) if side == 0 else (brow >= 2 * cut + 2)


def _seg_mean(seg, ok, y):
    """Mean result (%) of a segment, or NaN if it is not a valid one (MIN_SEG <= n <= valid candles - MIN_SEG)."""
    m = seg & ok
    n = int(m.sum())
    return float(y[m].mean()) if MIN_SEG <= n <= int(ok.sum()) - MIN_SEG else np.nan


def screen_indicator(bins, ncv, Y, z_mu, z_sd):
    """Phase 1, raw part (cached): per-symbol T1 (max z) and mean result of its winning rule.
    z_mu, z_sd: pilot null mean and std of every combination of this indicator's instances."""
    n_inst, n_sym, _ = bins.shape
    shape = (n_inst, len(TARGETS), NCUT, 2)

    edges = np.empty((n_inst, n_sym, len(TARGETS), NCUT, 2))
    osum = np.empty((n_inst, n_sym, len(TARGETS), NCUT, 2))
    fill_edges(bins, ncv, Y, edges, osum)
    t_sym, arg_sym = reduce_edges(edges, osum, z_mu, z_sd)

    mean_sym = np.full(n_sym, np.nan)
    for s in range(n_sym):
        if arg_sym[s] >= 0:
            i, g, c, side = np.unravel_index(arg_sym[s], shape)
            y = Y[s, :, g]
            mean_sym[s] = _seg_mean(_seg_mask(bins[i, s], c, side), (bins[i, s] >= 0) & np.isfinite(y), y)
    return {"T1": t_sym, "mean1": mean_sym}


def screen_pair(bA, bB, ncvA, ncvB, Y, shifts, pilot, m_stop):
    """Phase 2, raw part (cached). bA, bB: (instances, symbols, N) of two indicators. Rule A AND B.
    Every combination gets two z, one per pilot (A shifted, B shifted), and scores the smaller of the two.
    T2 (per symbol) is the best score, and each null is T2 recomputed with one of the two shifted:
      null2_A  A shifted (B and the targets stay aligned)
      null2_B  B shifted (A and the targets stay aligned)
    Only real combinations compete (pair_valid_mask): the same set in T2, both pilots and both nulls.
    Also the mean result of the winning rule of T2.
    Early stop (m_stop, see null_stop_count): a symbol stops in a null as soon as it fails for sure, and null2_A
    only computes the symbols still alive after null2_B. NaN where not computed."""
    nA, n_sym, _ = bA.shape
    nB = bB.shape[0]
    n_tg = len(TARGETS)
    shape = (nA, nB, n_tg, NCUT, 2, NCUT, 2)

    # --- pilots: layout (A, B) shifts B, layout (B, A) shifts A. Same valid combinations in both.
    muB, sdB = pair_edge_moments(bA, bB, ncvA, ncvB, Y, pilot, float(MIN_PILOT_N))
    muA_ba, sdA_ba = pair_edge_moments(bB, bA, ncvB, ncvA, Y, pilot, float(MIN_PILOT_N))
    ok = pair_valid_mask(bA, bB, ncvA, ncvB, Y)
    ok_ba = swap_pair(ok, nA, nB, n_tg)
    muB[~ok] = np.nan
    sdB[~ok] = np.nan
    muA_ba[~ok_ba] = np.nan
    sdA_ba[~ok_ba] = np.nan
    del ok, ok_ba

    # --- T2 and the null shifting B, layout (A, B)
    muA, sdA = swap_pair(muA_ba, nB, nA, n_tg), swap_pair(sdA_ba, nB, nA, n_tg)
    t_sym, arg_sym = pair_T(bA, bB, ncvA, ncvB, Y, 0, muA, sdA, muB, sdB, np.arange(n_sym, dtype=np.int64))
    null_B, alive = pair_null_early(bA, bB, ncvA, ncvB, Y, shifts, muA, sdA, muB, sdB,
                                    t_sym, np.isfinite(t_sym), m_stop)
    muB_ba, sdB_ba = swap_pair(muB, nA, nB, n_tg), swap_pair(sdB, nA, nB, n_tg)
    del muA, sdA, muB, sdB

    # --- null shifting A, layout (B, A): only the symbols that have not failed in null2_B
    null_A, _alive = pair_null_early(bB, bA, ncvB, ncvA, Y, shifts, muA_ba, sdA_ba, muB_ba, sdB_ba,
                                     t_sym, alive, m_stop)
    del muA_ba, sdA_ba, muB_ba, sdB_ba

    mean_sym = np.full(n_sym, np.nan)
    for s in range(n_sym):
        if arg_sym[s] >= 0:
            iA, iB, g, cA, sA, cB, sB = np.unravel_index(arg_sym[s], shape)
            y = Y[s, :, g]
            seg = _seg_mask(bA[iA, s], cA, sA) & _seg_mask(bB[iB, s], cB, sB)
            valid = (bA[iA, s] >= 0) & (bB[iB, s] >= 0) & np.isfinite(y)
            mean_sym[s] = _seg_mean(seg, valid, y)
    return {"T2": t_sym, "mean2": mean_sym, "null2_A": null_A, "null2_B": null_B}


def _evaluate1(r):
    """Phase 1 selection (never cached): T1 > floor1 per symbol."""
    floor1, med1, p841 = _null_stats(r["null1"])
    pass1 = r["T1"] > floor1
    score1 = _scores(r["T1"], med1, p841)
    return {"n_pass": int(pass1.sum()), "score": _score(score1, pass1), "mean": _median_ok(r["mean1"], pass1)}


def _evaluate2(r):
    """Phase 2 selection (never cached): per symbol the pair passes only if T2 beats BOTH floors,
    T2 > floor2_A and T2 > floor2_B. Its score is the smaller of the two scores."""
    floor2_A, med2_A, p842_A = _null_stats(r["null2_A"])
    floor2_B, med2_B, p842_B = _null_stats(r["null2_B"])
    pass2 = (r["T2"] > floor2_A) & (r["T2"] > floor2_B)
    score2 = np.minimum(_scores(r["T2"], med2_A, p842_A), _scores(r["T2"], med2_B, p842_B))
    return {"n_pass": int(pass2.sum()), "score": _score(score2, pass2), "mean": _median_ok(r["mean2"], pass2)}


def compute_raw(ohlcv_arr, pos, n, instances, names, by_ind):
    """Everything expensive: targets, bins, pilot, synthetic paths, phase 1 and phase 2.
    Returns the raw results, independent of GROUP_N and NULL_PCT (what the cache stores)."""
    # --- targets and indicators -> segments at the registry thresholds
    t1 = time.time()
    Y = build_targets(ohlcv_arr, pos, n)
    bins, ncv, empty = build_bins(ohlcv_arr, pos, n, instances)
    logger.info(f"Targets and {len(names)} indicators ({len(instances)} instances): {time.time() - t1:.0f}s")
    for key, s in empty:
        logger.info(f"  WARNING: {key} has no values in {s}")

    # --- phase 1: pilot (null mean and std of every combination, for z), null, T1
    z_mu, z_sd = build_pilot_moments(ohlcv_arr, pos, n, instances, names, by_ind)
    null1 = build_null_paths(ohlcv_arr, pos, n, instances, names, by_ind, z_mu, z_sd)
    p1 = {}
    t1 = time.time()
    for k, nm in enumerate(names):
        p1[nm] = screen_indicator(np.ascontiguousarray(bins[by_ind[nm]]), ncv[by_ind[nm]], Y,
                                  np.ascontiguousarray(z_mu[by_ind[nm]]), np.ascontiguousarray(z_sd[by_ind[nm]]))
        p1[nm]["null1"] = np.ascontiguousarray(null1[:, k, :])
        progress(f"Phase 1, {len(names)} indicators", k + 1, len(names), t1)
    del z_mu, z_sd, null1

    # --- phase 2: every indicator with every other one (unordered pairs)
    shifts, pilot = phase2_shifts(n)
    m_stop = null_stop_count(N_NULL_PATHS, NULL_PCT)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    p2 = {}
    t1 = time.time()
    for k, (a, b) in enumerate(pairs):
        p2[(a, b)] = screen_pair(np.ascontiguousarray(bins[by_ind[a]]), np.ascontiguousarray(bins[by_ind[b]]),
                                 ncv[by_ind[a]], ncv[by_ind[b]], Y, shifts, pilot, m_stop)
        progress(f"Phase 2, {len(pairs)} pairs x 2 nulls", k + 1, len(pairs), t1)
    done = sum(int((~np.isnan(r[nk])).sum()) for r in p2.values() for nk in ("null2_A", "null2_B"))
    logger.info(f"Phase 2 early stop (NULL_PCT={NULL_PCT}: fails at {m_stop} null values >= T2): "
                f"{100 * done / max(2 * N_NULL_PATHS * len(SYMBOLS) * len(pairs), 1):.1f}% of the null shifts computed")

    return {"names": names, "empty": empty, "p1": p1, "pairs": pairs, "p2": p2, "null_pct": NULL_PCT,
            "created": datetime.now().isoformat(timespec="seconds")}


# =============================================================================
# CACHE
# =============================================================================
def _h_update(h, x):
    """Feeds one value into the hash: arrays by dtype, shape and bytes, anything else by repr."""
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            h.update(repr(x.tolist()).encode())
        else:
            h.update(f"{x.dtype.str}{x.shape}".encode())
            h.update(np.ascontiguousarray(x).tobytes())
    else:
        h.update(repr(x).encode())
    h.update(b"|")


def _source_files():
    """Source files of the project modules currently loaded (indicators_pool, screening_engine, utils...).
    This script is left out: its compute functions are hashed one by one, so editing GROUP_N does not count."""
    me = os.path.abspath(__file__)
    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.abspath(f)
        if (f != me and f.startswith(PROJECT_ROOT + os.sep) and os.path.isfile(f)
                and f"{os.sep}site-packages{os.sep}" not in f):
            files.add(f)
    return sorted(files)


def _cache_key(ohlcv_arr, grid, instances):
    """SHA-256 of everything that affects the raw results. Anything that changes it forces a recompute."""
    h = hashlib.sha256()

    # library versions (a numpy or numba upgrade could change the last bits)
    _h_update(h, (np.__version__, numba.__version__, tuple(sys.version_info[:2])))

    # config that affects the computation (GROUP_N, NULL_PCT and N_JOBS are left out on purpose;
    # NULL_PCT is checked against the one stored in the cache, see _load_cache)
    _h_update(h, (DATASET, TIMEFRAME, list(SYMBOLS), CONFIGS, N_NULL_PATHS, N_PILOTS,
                  MIN_PILOT_N, SEED_NULL, SEED_PILOT, L_SHIFT, MIN_SEG, NCUT, NULL_BATCH))

    # data: common grid and every column of every symbol
    _h_update(h, np.asarray(grid))
    for s in SYMBOLS:
        arr = ohlcv_arr[s]
        cols = sorted(arr.keys()) if hasattr(arr, "keys") else sorted(arr.dtype.names)
        for col in cols:
            _h_update(h, (s, col))
            _h_update(h, np.asarray(arr[col]))

    # indicator pool: instances, thresholds and roles
    _h_update(h, [instance_key(inst["indicator"], inst["params"]) for inst in instances])
    for nm in dict.fromkeys(inst["indicator"] for inst in instances):
        meta = CANDIDATE_REGISTRY[nm]
        _h_update(h, (nm, list(meta["thresholds"]), meta["role"]))

    # code: project modules and the compute functions of this script
    for f in _source_files():
        _h_update(h, os.path.relpath(f, PROJECT_ROOT))
        with open(f, "rb") as fh:
            h.update(fh.read())
    for fn in (load_aligned, build_targets, build_bins, _ind_slices,
               _path_chunk, _run_paths, build_pilot_moments, build_null_paths, pilot_shifts, phase2_shifts,
               _seg_mask, _seg_mean, screen_indicator, screen_pair, compute_raw):
        _h_update(h, inspect.getsource(fn))

    return h.hexdigest()


def _cache_path(key):
    return os.path.join(CACHE_DIR, f"screen_{DATASET}_{TIMEFRAME}_{key[:16]}.pkl")


def _load_cache(path, key, names):
    """Raw results of a previous run with the same fingerprint, or None. Also None if it was computed with a
    NULL_PCT above the current one (its phase 2 nulls stopped early for that NULL_PCT)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as e:
        logger.info(f"  WARNING: unreadable cache {os.path.basename(path)} ({e}), recomputing")
        return None
    if not isinstance(raw, dict) or raw.get("key") != key or raw.get("names") != names:
        return None
    if raw.get("null_pct") is None or raw["null_pct"] > NULL_PCT:
        logger.info(f"  Cache computed with NULL_PCT={raw.get('null_pct')} > {NULL_PCT}: recomputing")
        return None
    return raw


def _save_cache(path, raw):
    """Atomic write: a run cut halfway never leaves a broken cache file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(raw, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# =============================================================================
# MAIN
# =============================================================================
def main():
    log_run_config()

    # --- data and common grid
    ohlcv_arr, grid, pos = load_aligned()
    n = len(grid)
    min_n = 2 * L_SHIFT + 1
    if n < min_n:
        raise ValueError(f"Only {n} common candles; at least {min_n} are needed")

    # --- indicator instances
    instances = build_flat_instances()
    by_ind = {}
    for ii, inst in enumerate(instances):
        by_ind.setdefault(inst["indicator"], []).append(ii)
    names = list(by_ind.keys())

    # --- raw results: from the cache or computed
    raw = None
    if USE_CACHE:
        t1 = time.time()
        key = _cache_key(ohlcv_arr, grid, instances)
        path = _cache_path(key)
        raw = _load_cache(path, key, names)
        if raw is not None:
            logger.info(f"Cache loaded: {os.path.basename(path)} (created {raw['created']}, "
                        f"{time.time() - t1:.0f}s)")
            for k, s in raw["empty"]:
                logger.info(f"  WARNING: {k} has no values in {s}")
        else:
            logger.info(f"No cache for this configuration, computing: {os.path.basename(path)}\n")
    if raw is None:
        raw = compute_raw(ohlcv_arr, pos, n, instances, names, by_ind)
        if USE_CACHE:
            raw["key"] = key
            _save_cache(path, raw)
            logger.info(f"\nCache saved: {path}")

    # --- selection (GROUP_N, NULL_PCT), split by role and ranked
    res1 = {nm: _evaluate1(raw["p1"][nm]) for nm in names}
    res2 = {p: _evaluate2(raw["p2"][p]) for p in raw["pairs"]}
    report_roles(names, res1, res2, GROUP_N)


def log_run_config() -> None:
    logger.info(f"\n{'─' * 115}")
    logger.info(f"  SCREENING START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET     : {DATASET} ── {TIMEFRAME}")
    logger.info(f"  PARAM GRID  : TP_PCT={TP_PCT} SL_PCT={SL_PCT} SELL_AFTER={SELL_AFTER} "
                f"({len(CONFIGS)} configs, {len(TARGETS)} targets)")
    logger.info(f"  NULL FLOOR  : N_NULL_PATHS={N_NULL_PATHS} NULL_PCT={NULL_PCT}")
    logger.info(f"  PILOT (z)   : N_PILOTS={N_PILOTS} (MIN_PILOT_N={MIN_PILOT_N})")
    logger.info(f"  SELECTION   : GROUP_N={GROUP_N}")
    logger.info(f"  CACHE       : USE_CACHE={USE_CACHE}   N_JOBS={N_JOBS}")
    logger.info(f"{'─' * 115}\n")
if __name__ == "__main__":
    start = time.time()
    try:
        main()
        elapsed = int(time.time() - start)
        logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
    except KeyboardInterrupt:
        elapsed = int(time.time() - start)
        logger.info(f"\n⛔  INTERRUPTED BY USER — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
        sys.exit(0)