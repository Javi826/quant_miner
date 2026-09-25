# core/indicators/screen_engine.py
import os
import sys
import time
import pickle
import hashlib
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime

import numba
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs

from . import screen_kernels as kernels
from .screen_kernels import MIN_SEG, MIN_PILOT_N
from .screen_kernels import compute_targets, fill_edges, reduce_edges, path_T_all, path_edges_all
from .screen_kernels import add_edge_moments, moments_inplace
from .screen_kernels import pair_T, pair_null_distribution, pair_edge_moments, pair_valid_mask
from .screen_kernels import compute_exits, npy_walk

logger = logging.getLogger(__name__)

NULL_BATCH = 32   # phase 2 shifts per batch (in order). Speed only: the selection does not depend on it


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass(frozen=True)
class ScreenConfig:

    tp_pct: tuple
    sl_pct: tuple
    sell_after: tuple
    commission: float           # % of the notional, charged on entry and on exit
    n_null_paths: int           # null: synthetic paths of phase 1 = shifts per null and pair of phase 2 (the floor)
    n_pilots: int               # pilot: paths of phase 1 = shifts per pilot and pair of phase 2 (z of every combination)
    null_pct: float             # percentile of the floor. Also sets the phase 2 early stop
    seed_null: int = 10_000     # nulls: first synthetic path of phase 1, and draw of the phase 2 shifts
    seed_pilot: int = 20_000    # pilots: first synthetic path of phase 1, and draw of the phase 2 shifts
    lookback: int = 100         # candles an indicator value may use: phase 2 shifts >= max(sell_after) + lookback
    n_jobs: int = -1

    def __post_init__(self):
        for f in ("tp_pct", "sl_pct", "sell_after"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        if not (self.sell_after and self.tp_pct and self.sl_pct):
            raise ValueError("sell_after, tp_pct and sl_pct must have at least one value")
        if any(int(v) != v or v < 1 for v in self.sell_after):
            raise ValueError(f"sell_after must be integers >= 1: {self.sell_after}")
        if any(not np.isfinite(v) or v < 0 for v in self.tp_pct + self.sl_pct):
            raise ValueError(f"tp_pct and sl_pct must be finite and >= 0: {self.tp_pct}, {self.sl_pct}")
        for nm in ("sell_after", "tp_pct", "sl_pct"):
            grid = getattr(self, nm)
            if len(set(grid)) != len(grid):
                raise ValueError(f"{nm} has duplicated values: {grid}")
        if not np.isfinite(self.commission) or self.commission < 0:
            raise ValueError(f"commission must be finite and >= 0: {self.commission}")
        if not (0 <= self.null_pct <= 100):
            raise ValueError(f"null_pct must be in [0, 100]: {self.null_pct}")
        if self.n_null_paths < 1:
            raise ValueError(f"n_null_paths must be >= 1: {self.n_null_paths}")
        if self.n_pilots < MIN_PILOT_N:                     # else no combination could compete
            raise ValueError(f"n_pilots must be >= {MIN_PILOT_N} (MIN_PILOT_N): {self.n_pilots}")
        if self.lookback < 0:
            raise ValueError(f"lookback must be >= 0: {self.lookback}")
        if (self.seed_pilot < self.seed_null + self.n_null_paths
                and self.seed_null < self.seed_pilot + self.n_pilots):   # pilot and floor must not share paths
            raise ValueError(f"Pilot paths [{self.seed_pilot}, {self.seed_pilot + self.n_pilots}) overlap the floor "
                             f"paths [{self.seed_null}, {self.seed_null + self.n_null_paths})")

    @property
    def configs(self):
        return [(float(tp), float(sl), int(sa)) for tp in self.tp_pct for sl in self.sl_pct for sa in self.sell_after]

    @property
    def targets(self):
        """Index g of Y: (direction, tp, sl, sa)."""
        return [(d, tp, sl, sa) for tp, sl, sa in self.configs for d in ("long", "short")]

    @property
    def target_side(self):
        """Side of every target g: 0 long, 1 short."""
        return np.array([0 if d == "long" else 1 for d, _tp, _sl, _sa in self.targets], dtype=np.int8)

    @property
    def l_shift(self):
        """Phase 2 shifts k in [l_shift, n - l_shift]."""
        return max(self.sell_after) + self.lookback

    def identity(self):
        """What the raw results depend on (null_pct is checked apart: see load_cache)."""
        return (self.configs, float(self.commission), self.n_null_paths, self.n_pilots,
                self.seed_null, self.seed_pilot, self.l_shift)


# =============================================================================
# 2. INDICATOR POOL
# =============================================================================
class IndicatorPool:

    def __init__(self, registry, instances, instance_key, group_names=None):
        self.registry = registry
        self.instances = list(instances)
        self.instance_key = instance_key
        self.group_names = dict(group_names or {})
        self.by_ind = {}
        for ii, inst in enumerate(self.instances):
            self.by_ind.setdefault(inst["indicator"], []).append(ii)
        if not self.by_ind:
            raise ValueError("The pool has no instances")
        for nm, idx in self.by_ind.items():
            if nm not in registry:
                raise ValueError(f"{nm}: instance of an indicator missing from the registry")
            if idx != list(range(idx[0], idx[-1] + 1)):
                raise ValueError(f"{nm}: its instances are not contiguous")
            if len(self.cuts(nm)) == 0:
                raise ValueError(f"{nm}: no thresholds")
        self.names = list(self.by_ind)
        self.ncut = max(len(self.cuts(nm)) for nm in self.names)       # max cuts per indicator
        if 2 * self.ncut + 1 > np.iinfo(np.int8).max:                  # bins are int8
            raise ValueError(f"Too many thresholds in one indicator ({self.ncut})")

    def cuts(self, nm):
        return np.array(sorted(set(float(v) for v in self.registry[nm]["thresholds"])), dtype=np.float64)

    def slices(self):
        """Slice [start, end) of the instances of every indicator, in names order."""
        starts = np.array([self.by_ind[nm][0] for nm in self.names], dtype=np.int64)
        ends = np.array([self.by_ind[nm][-1] + 1 for nm in self.names], dtype=np.int64)
        return starts, ends

    def code_files(self):
        """Source files of the indicator functions and of instance_key."""
        fns = [m["fn"] for nm, m in self.registry.items() if nm in self.by_ind] + [self.instance_key]
        files = set()
        for fn in fns:
            f = getattr(sys.modules.get(getattr(fn, "__module__", None)), "__file__", None)
            if f and os.path.isfile(f):
                files.add(os.path.abspath(f))
        return sorted(files)


# =============================================================================
# 3. DATA
# =============================================================================
@dataclass
class Aligned:
    """The symbols' arrays and the candles shared by all of them."""
    ohlcv_arr: dict
    symbols: list
    grid: np.ndarray            # common timestamps (int64 ns)
    pos: dict                   # symbol -> index of every common candle in its own arrays

    @property
    def n(self):
        return len(self.grid)


def i64(a):
    return np.asarray(a).astype("datetime64[ns]").view("int64")


def align_symbols(ohlcv_arr, symbols):
    """Aligns the symbols on the candles shared by all (exact timestamp intersection)."""
    symbols = list(symbols)
    if not symbols:
        raise ValueError("No symbols")
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"Duplicated symbols: {symbols}")
    missing = [s for s in symbols if s not in ohlcv_arr]
    if missing:
        raise ValueError(f"Symbols missing from data: {missing}")

    ts = {}
    for s in symbols:
        t = i64(ohlcv_arr[s]["ts"])
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            raise ValueError(f"{s}: timestamps are not strictly increasing")
        ts[s] = t

    grid = ts[symbols[0]]
    for s in symbols[1:]:
        grid = np.intersect1d(grid, ts[s], assume_unique=True)
    pos = {s: np.searchsorted(ts[s], grid) for s in symbols}
    return Aligned(ohlcv_arr, symbols, grid, pos)


# =============================================================================
# 4. SYNTHETIC PATHS
# =============================================================================
def _price_decimals(close):

    x = np.asarray(close, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if len(x) == 0:
        raise ValueError("price_decimals: no valid prices")
    for d in range(0, 9):
        scaled = x * (10.0 ** d)
        tol    = 1e-4 + 5e-7 * np.abs(scaled)
        if np.all(np.abs(scaled - np.round(scaled)) < tol):
            return d
    raise ValueError("price_decimals: could not determine quote precision")


def make_synthetic(ohlcv_arr, seed, symbols):

    all_ts = np.unique(np.concatenate([i64(ohlcv_arr[s]["ts"]) for s in symbols]))
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(all_ts))
    syn = {}
    for s in symbols:
        arr = ohlcv_arr[s]
        o, h, l, c = (np.asarray(arr[k], dtype=np.float64) for k in ("open", "high", "low", "close"))
        eps = signs[np.searchsorted(all_ts, i64(arr["ts"]))]
        pc = np.concatenate(([o[0]], c[:-1]))                     # previous close (first candle: its open)
        ro, rh, rl, rc = np.log(o / pc), np.log(h / pc), np.log(l / pc), np.log(c / pc)
        flip = eps < 0
        ro, rc = np.where(flip, -ro, ro), np.where(flip, -rc, rc)
        rh, rl = np.where(flip, -rl, rh), np.where(flip, -rh, rl)
        new_pc = np.exp(np.concatenate(([np.log(o[0])], np.log(o[0]) + np.cumsum(rc)[:-1])))
        d = _price_decimals(c)
        new = dict(arr)
        new["open"] = np.round(new_pc * np.exp(ro), d)
        new["high"] = np.round(new_pc * np.exp(rh), d)
        new["low"] = np.round(new_pc * np.exp(rl), d)
        new["close"] = np.round(new_pc * np.exp(rc), d)
        new["high"] = np.maximum(new["high"], np.maximum(new["open"], new["close"]))
        new["low"] = np.minimum(new["low"], np.minimum(new["open"], new["close"]))
        ht, lt = np.asarray(arr["high_time"]), np.asarray(arr["low_time"])
        new["high_time"] = np.where(flip, lt, ht)
        new["low_time"] = np.where(flip, ht, lt)
        syn[s] = new
    return syn


# =============================================================================
# 5. TARGETS AND BINS
# =============================================================================
def build_targets(arrs, data, cfg):

    Y = np.full((len(data.symbols), data.n, len(cfg.targets)), np.nan)
    for si, s in enumerate(data.symbols):
        arr = arrs[s]
        close = np.ascontiguousarray(arr["close"], dtype=np.float64)
        high = np.ascontiguousarray(arr["high"], dtype=np.float64)
        low = np.ascontiguousarray(arr["low"], dtype=np.float64)
        ht = np.ascontiguousarray(i64(arr["high_time"]))
        lt = np.ascontiguousarray(i64(arr["low_time"]))
        for ci, (tp, sl, sa) in enumerate(cfg.configs):
            y_long, y_short = compute_targets(close, high, low, ht, lt, tp, sl, sa, float(cfg.commission))
            Y[si, :, 2 * ci] = y_long[data.pos[s]]
            Y[si, :, 2 * ci + 1] = y_short[data.pos[s]]
    return Y

def build_exits(arrs, data, cfg):

    E = np.full((len(data.symbols), data.n, len(cfg.targets)), -1, dtype=np.int64)
    for si, s in enumerate(data.symbols):
        arr = arrs[s]
        close = np.ascontiguousarray(arr["close"], dtype=np.float64)
        high = np.ascontiguousarray(arr["high"], dtype=np.float64)
        low = np.ascontiguousarray(arr["low"], dtype=np.float64)
        ht = np.ascontiguousarray(i64(arr["high_time"]))
        lt = np.ascontiguousarray(i64(arr["low_time"]))
        pos = data.pos[s]
        for ci, (tp, sl, sa) in enumerate(cfg.configs):
            e_long, e_short = compute_exits(close, high, low, ht, lt, tp, sl, sa)
            E[si, :, 2 * ci] = np.searchsorted(pos, e_long[pos], side="right")
            E[si, :, 2 * ci + 1] = np.searchsorted(pos, e_short[pos], side="right")
    return E


def _npy_stats(m, y, free, mean_ypy):

    if mean_ypy != mean_ypy:
        return np.nan, 0
    s, k = npy_walk(m, y, free)
    return (s / k if k else np.nan), k


def build_bins(arrs, data, pool):

    bins = np.full((len(pool.instances), len(data.symbols), data.n), -1, dtype=np.int8)
    ncv = np.zeros(len(pool.instances), dtype=np.int64)
    empty = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for ii, inst in enumerate(pool.instances):
            nm = inst["indicator"]
            c = pool.cuts(nm)
            ncv[ii] = len(c)
            for si, s in enumerate(data.symbols):
                x = np.asarray(pool.registry[nm]["fn"](arrs[s], None, inst["params"]), dtype=np.float64)[data.pos[s]]
                ok = np.isfinite(x)
                if not ok.any():
                    empty.append((pool.instance_key(nm, inst["params"]), s))
                    continue
                xo = x[ok]
                bins[ii, si, ok] = (np.searchsorted(c, xo, side="left") + np.searchsorted(c, xo, side="right"))
    return bins, ncv, empty


# =============================================================================
# 6. PHASE 1: EVERY INDICATOR ALONE
# =============================================================================
def progress(label, k, total, t0):

    if not logger.isEnabledFor(logging.INFO):
        return
    sys.stdout.write(f"\r{label}: {k}/{total} ({100 * k // max(total, 1)}%)")
    if k >= total:
        sys.stdout.write(f"  {(time.time() - t0) / 60:.1f} min\n")
    sys.stdout.flush()


def _path_chunk(kind, seeds, data, pool, cfg, starts, ends, z_mu, z_sd, in_worker):

    if in_worker:
        numba.set_num_threads(1)
        if z_mu is not None:
            z_mu, z_sd = np.array(z_mu), np.array(z_sd)
    out = []
    for seed in seeds:
        syn = make_synthetic(data.ohlcv_arr, seed, data.symbols)
        Y_p = build_targets(syn, data, cfg)
        bins_p, ncv_p, _e = build_bins(syn, data, pool)
        if kind == "pilot":
            out.append(path_edges_all(bins_p, ncv_p, Y_p, starts, ends, pool.ncut))
        else:
            out.append(path_T_all(bins_p, ncv_p, Y_p, starts, ends, z_mu, z_sd))
    return out


def _run_paths(kind, seeds, label, data, pool, cfg, z_mu=None, z_sd=None):

    t0 = time.time()
    starts, ends = pool.slices()
    n_jobs = effective_n_jobs(cfg.n_jobs)
    done = 0
    if n_jobs == 1:
        for seed in seeds:
            done += 1
            yield _path_chunk(kind, [seed], data, pool, cfg, starts, ends, z_mu, z_sd, False)[0]
            progress(label, done, len(seeds), t0)
        return
    chunks = [[int(v) for v in c] for c in np.array_split(np.asarray(seeds), min(len(seeds), 4 * n_jobs))]
    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_path_chunk)(kind, ch, data, pool, cfg, starts, ends, z_mu, z_sd, True) for ch in chunks)
    for res in results:
        for r in res:
            done += 1
            yield r
        progress(label, done, len(seeds), t0)


def build_pilot_moments(data, pool, cfg):

    shape = (len(pool.instances), len(data.symbols), len(cfg.targets), pool.ncut, 2)
    m_n = np.zeros(shape)
    m_s = np.zeros(shape)
    m_q = np.zeros(shape)
    seeds = [cfg.seed_pilot + r for r in range(cfg.n_pilots)]
    label = f"Phase 1 pilot, {cfg.n_pilots} synthetic paths"
    for edges in _run_paths("pilot", seeds, label, data, pool, cfg):
        add_edge_moments(edges, m_n, m_s, m_q)             # in path order: same sums as a serial run
    moments_inplace(m_n.reshape(-1), m_s.reshape(-1), m_q.reshape(-1), float(MIN_PILOT_N))
    return m_s, m_q                            # m_s is now the mean and m_q the std


def build_null_paths(data, pool, cfg, z_mu, z_sd):

    null_sym = np.empty((cfg.n_null_paths, len(pool.names), len(data.symbols)))
    seeds = [cfg.seed_null + r for r in range(cfg.n_null_paths)]
    label = f"Phase 1 null, {cfg.n_null_paths} synthetic paths"
    for r, t_sym in enumerate(_run_paths("null", seeds, label, data, pool, cfg, z_mu, z_sd)):
        null_sym[r] = t_sym
    return null_sym


def _seg_mask(brow, cut, side):

    return ((brow >= 0) & (brow <= 2 * cut)) if side == 0 else (brow >= 2 * cut + 2)


def _seg_mean(seg, ok, y):

    m = seg & ok
    n = int(m.sum())
    return float(y[m].mean()) if MIN_SEG <= n <= int(ok.sum()) - MIN_SEG else np.nan


def _pack_rules(rules, n_sym, n):

    side = np.full(n_sym, -1, dtype=np.int8)
    rows = []
    for s in sorted(rules):
        d, m = rules[s]
        side[s] = d
        rows.append(np.packbits(m))
    mask = np.array(rows, dtype=np.uint8) if rows else np.zeros((0, (n + 7) // 8), dtype=np.uint8)
    return {"side": side, "mask": mask}


def screen_indicator(bins, ncv, Y, E, z_mu, z_sd, target_side):

    n_inst, n_sym, n = bins.shape
    ncut = z_mu.shape[3]
    shape = (n_inst, Y.shape[2], ncut, 2)

    edges = np.empty((n_inst, n_sym, Y.shape[2], ncut, 2))
    osum = np.empty((n_inst, n_sym, Y.shape[2], ncut, 2))
    fill_edges(bins, ncv, Y, edges, osum)
    t_sym, arg_sym = reduce_edges(edges, osum, z_mu, z_sd)

    mean_sym = np.full(n_sym, np.nan)
    mean_npy = np.full(n_sym, np.nan)
    n_npy = np.zeros(n_sym, dtype=np.int64)
    rules = {}
    for s in range(n_sym):
        if arg_sym[s] >= 0:
            i, g, c, side = np.unravel_index(arg_sym[s], shape)
            y = Y[s, :, g]
            seg = _seg_mask(bins[i, s], c, side)
            ok = (bins[i, s] >= 0) & np.isfinite(y)
            mean_sym[s] = _seg_mean(seg, ok, y)
            mean_npy[s], n_npy[s] = _npy_stats(seg & ok, y, E[s, :, g], mean_sym[s])
            rules[s] = (target_side[g], seg)
    return {"T1": t_sym, "mean1": mean_sym, "mean_npy1": mean_npy, "n_npy1": n_npy,
            **_pack_rules(rules, n_sym, n)}

# =============================================================================
# 7. PHASE 2: EVERY PAIR A + B
# =============================================================================
def swap_pair(x, nA, nB, n_tg, ncut):

    n_sym = x.shape[1]
    y = x.reshape(nA, nB, n_tg, ncut, 2, ncut, 2, n_sym).transpose(1, 0, 2, 5, 6, 3, 4, 7)
    return np.ascontiguousarray(y).reshape(nA * nB * n_tg * ncut * 2 * ncut * 2, n_sym)


def null_stop_count(n_null, null_pct):

    lo = int(np.floor((n_null - 1) * (null_pct / 100) - 1e-9))
    lo = min(max(lo, 0), n_null - 1)
    return n_null - lo


def pair_null_early(bA, bB, ncvA, ncvB, Y, shifts, z1_mu, z1_sd, z2_mu, z2_sd, t_real, alive, m_stop, ncut):

    n_k = shifts.shape[0]
    n_sym = t_real.shape[0]
    null = np.full((n_k, n_sym), np.nan)
    hits = np.zeros(n_sym, dtype=np.int64)
    alive = np.array(alive, dtype=np.bool_)
    for q0 in range(0, n_k, NULL_BATCH):
        sym_idx = np.flatnonzero(alive).astype(np.int64)
        if sym_idx.size == 0:
            break
        sh = np.ascontiguousarray(shifts[q0:q0 + NULL_BATCH])
        ts = pair_null_distribution(bA, bB, ncvA, ncvB, Y, sh, z1_mu, z1_sd, z2_mu, z2_sd, sym_idx, ncut)[:, sym_idx]
        null[q0:q0 + sh.shape[0], sym_idx] = ts
        hits[sym_idx] += (ts >= t_real[sym_idx]).sum(axis=0)
        alive[sym_idx] = hits[sym_idx] < m_stop
    return null, alive


def pilot_shifts(n, exclude, cfg):

    cand = np.setdiff1d(np.arange(cfg.l_shift, n - cfg.l_shift + 1, dtype=np.int64),
                        np.asarray(exclude, dtype=np.int64))
    if len(cand) < cfg.n_pilots:
        raise ValueError(f"Only {len(cand)} free shifts for the phase 2 pilot, {cfg.n_pilots} are needed")
    return np.random.default_rng(cfg.seed_pilot).choice(cand, size=cfg.n_pilots, replace=False).astype(np.int64)


def phase2_shifts(n, cfg):

    rng = np.random.default_rng(cfg.seed_null)
    shifts = rng.integers(cfg.l_shift, n - cfg.l_shift, size=cfg.n_null_paths, endpoint=True).astype(np.int64)
    return shifts, pilot_shifts(n, shifts, cfg)


def screen_pair(bA, bB, ncvA, ncvB, Y, E, shifts, pilot, m_stop, target_side, ncut):

    nA, n_sym, n = bA.shape
    nB = bB.shape[0]
    n_tg = Y.shape[2]
    shape = (nA, nB, n_tg, ncut, 2, ncut, 2)

    # --- pilots: layout (A, B) shifts B, layout (B, A) shifts A. Same valid combinations in both.
    muB, sdB = pair_edge_moments(bA, bB, ncvA, ncvB, Y, pilot, float(MIN_PILOT_N), ncut)
    muA_ba, sdA_ba = pair_edge_moments(bB, bA, ncvB, ncvA, Y, pilot, float(MIN_PILOT_N), ncut)
    ok = pair_valid_mask(bA, bB, ncvA, ncvB, Y, ncut)
    ok_ba = swap_pair(ok, nA, nB, n_tg, ncut)
    muB[~ok] = np.nan
    sdB[~ok] = np.nan
    muA_ba[~ok_ba] = np.nan
    sdA_ba[~ok_ba] = np.nan
    del ok, ok_ba

    muA, sdA = swap_pair(muA_ba, nB, nA, n_tg, ncut), swap_pair(sdA_ba, nB, nA, n_tg, ncut)
    t_sym, arg_sym = pair_T(bA, bB, ncvA, ncvB, Y, 0, muA, sdA, muB, sdB, np.arange(n_sym, dtype=np.int64), ncut)
    null_B, alive = pair_null_early(bA, bB, ncvA, ncvB, Y, shifts, muA, sdA, muB, sdB,
                                    t_sym, np.isfinite(t_sym), m_stop, ncut)
    muB_ba, sdB_ba = swap_pair(muB, nA, nB, n_tg, ncut), swap_pair(sdB, nA, nB, n_tg, ncut)
    del muA, sdA, muB, sdB

    null_A, alive = pair_null_early(bB, bA, ncvB, ncvA, Y, shifts, muA_ba, sdA_ba, muB_ba, sdB_ba,
                                    t_sym, alive, m_stop, ncut)
    del muA_ba, sdA_ba, muB_ba, sdB_ba

    mean_sym = np.full(n_sym, np.nan)
    mean_npy = np.full(n_sym, np.nan)
    n_npy = np.zeros(n_sym, dtype=np.int64)
    rules = {}
    for s in range(n_sym):
        if arg_sym[s] >= 0:
            iA, iB, g, cA, sA, cB, sB = np.unravel_index(arg_sym[s], shape)
            y = Y[s, :, g]
            seg = _seg_mask(bA[iA, s], cA, sA) & _seg_mask(bB[iB, s], cB, sB)
            valid = (bA[iA, s] >= 0) & (bB[iB, s] >= 0) & np.isfinite(y)
            mean_sym[s] = _seg_mean(seg, valid, y)
            mean_npy[s], n_npy[s] = _npy_stats(seg & valid, y, E[s, :, g], mean_sym[s])
            if alive[s]:
                rules[s] = (target_side[g], seg)
    return {"T2": t_sym, "mean2": mean_sym, "mean_npy2": mean_npy, "n_npy2": n_npy,
            "null2_A": null_A, "null2_B": null_B, **_pack_rules(rules, n_sym, n)}


# =============================================================================
# 8. RUN
# =============================================================================
def compute_raw(data, pool, cfg):
    """Raw results of both phases. The selection is done later (screen_report), never cached."""
    n, names, by_ind = data.n, pool.names, pool.by_ind
    target_side = cfg.target_side

    t1 = time.time()
    Y = build_targets(data.ohlcv_arr, data, cfg)
    E = build_exits(data.ohlcv_arr, data, cfg)
    bins, ncv, empty = build_bins(data.ohlcv_arr, data, pool)
    logger.info(f"Targets and {len(names)} indicators ({len(pool.instances)} instances): {time.time() - t1:.0f}s")
    for key, s in empty:
        logger.info(f"  WARNING: {key} has no values in {s}")

    z_mu, z_sd = build_pilot_moments(data, pool, cfg)
    null1 = build_null_paths(data, pool, cfg, z_mu, z_sd)
    p1 = {}
    t1 = time.time()
    for k, nm in enumerate(names):
        p1[nm] = screen_indicator(np.ascontiguousarray(bins[by_ind[nm]]), ncv[by_ind[nm]], Y, E,
                                  np.ascontiguousarray(z_mu[by_ind[nm]]), np.ascontiguousarray(z_sd[by_ind[nm]]),
                                  target_side)
        p1[nm]["null1"] = np.ascontiguousarray(null1[:, k, :])
        progress(f"Phase 1, {len(names)} indicators", k + 1, len(names), t1)
    del z_mu, z_sd, null1

    shifts, pilot = phase2_shifts(n, cfg)
    m_stop = null_stop_count(cfg.n_null_paths, cfg.null_pct)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    p2 = {}
    t1 = time.time()
    for k, (a, b) in enumerate(pairs):
        p2[(a, b)] = screen_pair(np.ascontiguousarray(bins[by_ind[a]]), np.ascontiguousarray(bins[by_ind[b]]),
                                 ncv[by_ind[a]], ncv[by_ind[b]], Y, E, shifts, pilot, m_stop, target_side, pool.ncut)
        progress(f"Phase 2, {len(pairs)} pairs x 2 nulls", k + 1, len(pairs), t1)
    done = sum(int((~np.isnan(r[nk])).sum()) for r in p2.values() for nk in ("null2_A", "null2_B"))
    logger.info(f"Phase 2 early stop (NULL_PCT={cfg.null_pct}: fails at {m_stop} null values >= T2): "
                f"{100 * done / max(2 * cfg.n_null_paths * len(data.symbols) * len(pairs), 1):.1f}% "
                f"of the null shifts computed")

    return {"names": names, "symbols": list(data.symbols), "n": n, "empty": empty, "p1": p1, "pairs": pairs,
            "p2": p2, "null_pct": cfg.null_pct, "created": datetime.now().isoformat(timespec="seconds")}


def run_screen(ohlcv_arr, symbols, pool, cfg, cache_dir=None, cache_name="screen", cache_tag=(), source_files=()):

    data = align_symbols(ohlcv_arr, symbols)
    min_n = 2 * cfg.l_shift + 1
    if data.n < min_n:
        raise ValueError(f"Only {data.n} common candles; at least {min_n} are needed")
    if cache_dir is None:
        return compute_raw(data, pool, cfg)

    t1 = time.time()
    key = cache_key(data, pool, cfg, cache_tag, source_files)
    path = cache_path(cache_dir, cache_name, key)
    raw = load_cache(path, key, pool.names, cfg.null_pct)
    if raw is not None:
        logger.info(f"Cache loaded: {os.path.basename(path)} (created {raw['created']}, {time.time() - t1:.0f}s)")
        for k, s in raw["empty"]:
            logger.info(f"  WARNING: {k} has no values in {s}")
        return raw
    logger.info(f"No cache for this configuration, computing: {os.path.basename(path)}\n")
    raw = compute_raw(data, pool, cfg)
    raw["key"] = key
    save_cache(path, raw)
    logger.info(f"\nCache saved: {path}")
    return raw


# =============================================================================
# 9. CACHE
# =============================================================================
def _h_update(h, x):

    if isinstance(x, np.ndarray):
        if x.dtype == object:
            h.update(repr(x.tolist()).encode())
        else:
            h.update(f"{x.dtype.str}{x.shape}".encode())
            h.update(np.ascontiguousarray(x).tobytes())
    else:
        h.update(repr(x).encode())
    h.update(b"|")


def cache_key(data, pool, cfg, tag=(), source_files=()):

    h = hashlib.sha256()
    _h_update(h, (np.__version__, numba.__version__, tuple(sys.version_info[:2])))
    _h_update(h, tuple(tag))
    _h_update(h, (list(data.symbols), cfg.identity(), MIN_PILOT_N, MIN_SEG, pool.ncut, NULL_BATCH))

    # data: common grid and every column of every symbol
    _h_update(h, np.asarray(data.grid))
    for s in data.symbols:
        arr = data.ohlcv_arr[s]
        cols = sorted(arr.keys()) if hasattr(arr, "keys") else sorted(arr.dtype.names)
        for col in cols:
            _h_update(h, (s, col))
            _h_update(h, np.asarray(arr[col]))

    _h_update(h, [pool.instance_key(inst["indicator"], inst["params"]) for inst in pool.instances])
    for nm in pool.names:
        _h_update(h, (nm, list(pool.registry[nm]["thresholds"])))

    files = {os.path.abspath(f) for f in source_files}
    files |= {os.path.abspath(__file__), os.path.abspath(kernels.__file__)}
    files |= set(pool.code_files())
    for f in sorted(files):
        _h_update(h, os.path.basename(f))
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def cache_path(cache_dir, name, key):
    return os.path.join(cache_dir, f"{name}_{key[:16]}.pkl")


def load_cache(path, key, names, null_pct):

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
    if raw.get("null_pct") is None or raw["null_pct"] > null_pct:
        logger.info(f"  Cache computed with NULL_PCT={raw.get('null_pct')} > {null_pct}: recomputing")
        return None
    return raw


def save_cache(path, raw):
    """Atomic write: a run cut halfway never leaves a broken cache file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(raw, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)