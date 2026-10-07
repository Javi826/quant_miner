# core/research/screening/screen_engine.py
import os
import ast
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

from .screen_kernels_cpu import MIN_SEG, MIN_PILOT_N
from .screen_kernels_cpu import compute_targets, fill_edges, path_T_all, path_edges_all
from .screen_kernels_cpu import add_edge_moments, moments_inplace
from .screen_kernels_cpu import pair_z_edges, pair_null_distribution, pair_edge_moments, pair_valid_mask
from .screen_kernels_cpu import compute_exits, npy_walk

logger = logging.getLogger(__name__)

NULL_BATCH   = 32    # phase 2 shifts per batch on the CPU (YPY). Memory only: the results do not depend on it
N_NULL_PATHS = 500   # null: synthetic paths of phase 1 = shifts per null and pair of phase 2 (build the floor)
N_PILOTS     = 50    # pilot: paths of phase 1 = shifts per pilot and pair of phase 2 (mean and std of every combination, for z)
MODES        = ("YPY", "NPY")   # YPY: a trade on every signal. NPY: a trade only while flat on its symbol (GPU)


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass(frozen=True)
class ScreenConfig:

    tp_pct: tuple
    sl_pct: tuple
    sell_after: tuple
    commission: float           # % of the notional, charged on entry and on exit
    null_pct: float             # percentile of the floor: pass / no pass and combination picked
    mode: str = "YPY"           # how the edge of a segment is measured: one of MODES
    n_null_paths: int = N_NULL_PATHS
    n_pilots: int = N_PILOTS
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
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}: {self.mode}")
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
        """What the raw results depend on, besides null_pct (part of cache_key: it sets the combination picked)."""
        return (self.configs, float(self.commission), self.mode, self.n_null_paths, self.n_pilots,
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
        if kind == "npy":
            out.append((bins_p, ncv_p, Y_p, build_exits(syn, data, cfg).astype(np.int32)))
        elif kind == "pilot":
            out.append(path_edges_all(bins_p, ncv_p, Y_p, starts, ends, pool.ncut))
        else:
            out.append(path_T_all(bins_p, ncv_p, Y_p, starts, ends, z_mu, z_sd))
    return out


def _run_paths(kind, seeds, label, data, pool, cfg, z_mu=None, z_sd=None):
    """Results of every seed, in seed order. npy: one seed per task, so the paths' data stream to the GPU."""
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
    n_chunks = len(seeds) if kind == "npy" else min(len(seeds), 4 * n_jobs)
    chunks = [[int(v) for v in c] for c in np.array_split(np.asarray(seeds), n_chunks)]
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


def build_pilot_moments_npy(data, pool, cfg):
    """As build_pilot_moments, with the NPY statistic of every segment computed on the GPU."""
    import cupy as cp
    from .screen_kernels_gpu import npy_prepare, npy_edges

    shape = (len(pool.instances), len(data.symbols), len(cfg.targets), pool.ncut, 2)
    m_n = cp.zeros(shape)
    m_s = cp.zeros(shape)
    m_q = cp.zeros(shape)
    seeds = [cfg.seed_pilot + r for r in range(cfg.n_pilots)]
    label = f"Phase 1 pilot [NPY], {cfg.n_pilots} synthetic paths"
    for bins_p, ncv_p, Y_p, E_p in _run_paths("npy", seeds, label, data, pool, cfg):
        v, _osum = npy_edges(bins_p, ncv_p, npy_prepare(Y_p, E_p), pool.ncut)
        ok = ~cp.isnan(v)
        m_n += ok                                              # in path order: same sums as a serial run
        m_s += cp.where(ok, v, 0.0)
        m_q += cp.where(ok, v * v, 0.0)
    m_n, m_s, m_q = cp.asnumpy(m_n), cp.asnumpy(m_s), cp.asnumpy(m_q)
    moments_inplace(m_n.reshape(-1), m_s.reshape(-1), m_q.reshape(-1), float(MIN_PILOT_N))
    return m_s, m_q                            # m_s is now the mean and m_q the std


def _npy_z(v, osum, z_mu, z_sd, xp):
    """z of every combination (-inf where it does not compete: invalid, osum <= 0 or no pilot std), as reduce_edges."""
    ok = xp.isfinite(v) & (osum > 0.0) & (z_sd > 0.0)
    return xp.where(ok, (v - z_mu) / xp.where(ok, z_sd, 1.0), -xp.inf)


def build_null_paths_npy(data, pool, cfg, z_mu, z_sd):
    """As build_null_paths: T of every indicator and symbol in every null path, NPY statistic on the GPU."""
    import cupy as cp
    from .screen_kernels_gpu import npy_prepare, npy_edges

    starts, _ends = pool.slices()
    mu_d, sd_d = cp.asarray(z_mu), cp.asarray(z_sd)
    null_sym = np.empty((cfg.n_null_paths, len(pool.names), len(data.symbols)))
    seeds = [cfg.seed_null + r for r in range(cfg.n_null_paths)]
    label = f"Phase 1 null [NPY], {cfg.n_null_paths} synthetic paths"
    for r, (bins_p, ncv_p, Y_p, E_p) in enumerate(_run_paths("npy", seeds, label, data, pool, cfg)):
        v, osum = npy_edges(bins_p, ncv_p, npy_prepare(Y_p, E_p), pool.ncut)
        z = _npy_z(v, osum, mu_d, sd_d, cp)
        z_inst = cp.asnumpy(z.reshape(z.shape[0], z.shape[1], -1).max(axis=2))   # (n_inst, n_sym)
        null_sym[r] = np.maximum.reduceat(z_inst, starts, axis=0)                 # (n_ind, n_sym)
    return null_sym


def null_floor(null_sym, null_pct):
    """Floor per symbol: the NULL_PCT percentile of the null (as screen_report). The null is the best z of all the
    combinations on noise, so a combination whose own z is above the floor is significant on its own."""
    with np.errstate(invalid="ignore"):
        return np.percentile(null_sym, null_pct, axis=0)


def _pick(zs, es, floor):
    """Per symbol (rows; combinations in flat order): T, the highest z; the combination picked, the largest edge
    among those whose z is above the floor (first on ties; -1 if none, that is, if T is not above the floor); its z
    (-inf if none)."""
    zs = np.ascontiguousarray(zs)
    with np.errstate(invalid="ignore"):
        above = zs > np.asarray(floor)[:, None]
    t_sym = zs.max(axis=1)
    arg = np.where(above, es, -np.inf).argmax(axis=1)
    has = above.any(axis=1)
    z_sym = np.where(has, zs[np.arange(zs.shape[0]), arg], -np.inf)
    return t_sym, np.where(has, arg, -1).astype(np.int64), z_sym


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


def _indicator_stats(bins, Y, E, t_sym, arg_sym, z_sym, target_side, ncut):
    """Phase 1 results of one pick per symbol: T, its z, its means, its baseline and its rule (any mode). base1: mean
    of the target over the valid candles, the unconditional mean the excess is measured against."""
    n_inst, n_sym, n = bins.shape
    shape = (n_inst, Y.shape[2], ncut, 2)
    mean_sym = np.full(n_sym, np.nan)
    mean_npy = np.full(n_sym, np.nan)
    base_sym = np.full(n_sym, np.nan)
    n_npy = np.zeros(n_sym, dtype=np.int64)
    rules = {}
    for s in range(n_sym):
        if arg_sym[s] >= 0:
            i, g, c, side = np.unravel_index(arg_sym[s], shape)
            y = Y[s, :, g]
            seg = _seg_mask(bins[i, s], c, side)
            ok = (bins[i, s] >= 0) & np.isfinite(y)
            mean_sym[s] = _seg_mean(seg, ok, y)
            base_sym[s] = float(y[ok].mean()) if ok.any() else np.nan
            mean_npy[s], n_npy[s] = _npy_stats(seg & ok, y, E[s, :, g], mean_sym[s])
            rules[s] = (target_side[g], seg)
    return {"T1": t_sym, "z1": z_sym, "mean1": mean_sym, "mean_npy1": mean_npy, "n_npy1": n_npy,
            "base1": base_sym, **_pack_rules(rules, n_sym, n)}


def screen_indicator(bins, Y, E, v, osum, z_mu, z_sd, floor, target_side):
    """Phase 1 raw results of an indicator from the statistic (v, osum) of its instances, any mode (host arrays)."""
    n_sym = bins.shape[1]
    ncut = z_mu.shape[3]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = _npy_z(v, osum, z_mu, z_sd, np)
    zs = np.moveaxis(z, 1, 0).reshape(n_sym, -1)           # per symbol, combinations in reduce_edges order
    es = np.moveaxis(osum, 1, 0).reshape(n_sym, -1)
    return _indicator_stats(bins, Y, E, *_pick(zs, es, floor), target_side, ncut)

# =============================================================================
# 7. PHASE 2: EVERY PAIR A + B
# =============================================================================
def swap_pair(x, nA, nB, n_tg, ncut, xp=np):
    """x in layout (A, B) as layout (B, A). xp: np or cp (a permutation, exact on both)."""
    n_sym = x.shape[1]
    y = x.reshape(nA, nB, n_tg, ncut, 2, ncut, 2, n_sym).transpose(1, 0, 2, 5, 6, 3, 4, 7)
    return xp.ascontiguousarray(y).reshape(nA * nB * n_tg * ncut * 2 * ncut * 2, n_sym)


def pair_null(bA, bB, ncvA, ncvB, Y, shifts, z1_mu, z1_sd, z2_mu, z2_sd, t_real, ncut):
    """The whole null of a pair: T of every shift of B (rows) and symbol (columns), YPY statistic (CPU), in batches
    of NULL_BATCH. Only where T competes (finite t_real): NaN in the other symbols."""
    null = np.full((shifts.shape[0], t_real.shape[0]), np.nan)
    sym_idx = np.flatnonzero(np.isfinite(t_real)).astype(np.int64)
    if sym_idx.size == 0:
        return null
    for q0 in range(0, shifts.shape[0], NULL_BATCH):
        sh = np.ascontiguousarray(shifts[q0:q0 + NULL_BATCH])
        null[q0:q0 + sh.shape[0], sym_idx] = pair_null_distribution(bA, bB, ncvA, ncvB, Y, sh, z1_mu, z1_sd, z2_mu,
                                                                    z2_sd, sym_idx, ncut)[:, sym_idx]
    return null


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


def _pair_stats(bA, bB, Y, E, t_sym, arg_sym, z_sym, target_side, ncut):
    """Phase 2 results of one pick per symbol: T, its z, its means, its baseline and its rule (any mode). base2: mean
    of the target over the candles where both have a value, the unconditional mean the excess is measured against."""
    nA, n_sym, n = bA.shape
    shape = (nA, bB.shape[0], Y.shape[2], ncut, 2, ncut, 2)
    mean_sym = np.full(n_sym, np.nan)
    mean_npy = np.full(n_sym, np.nan)
    base_sym = np.full(n_sym, np.nan)
    n_npy = np.zeros(n_sym, dtype=np.int64)
    rules = {}
    for s in range(n_sym):
        if arg_sym[s] >= 0:
            iA, iB, g, cA, sA, cB, sB = np.unravel_index(arg_sym[s], shape)
            y = Y[s, :, g]
            seg = _seg_mask(bA[iA, s], cA, sA) & _seg_mask(bB[iB, s], cB, sB)
            valid = (bA[iA, s] >= 0) & (bB[iB, s] >= 0) & np.isfinite(y)
            mean_sym[s] = _seg_mean(seg, valid, y)
            base_sym[s] = float(y[valid].mean()) if valid.any() else np.nan
            mean_npy[s], n_npy[s] = _npy_stats(seg & valid, y, E[s, :, g], mean_sym[s])
            rules[s] = (target_side[g], seg)
    return {"T2": t_sym, "z2": z_sym, "mean2": mean_sym, "mean_npy2": mean_npy, "n_npy2": n_npy,
            "base2": base_sym, **_pack_rules(rules, n_sym, n)}


def _pair_result(bA, bB, Y, E, picked, null_A, null_B, target_side, ncut):
    """Phase 2 raw results of a pair from _pick: the pick of all the targets and its nulls (any mode)."""
    return {**_pair_stats(bA, bB, Y, E, *picked, target_side, ncut), "null2_A": null_A, "null2_B": null_B}


def screen_pair(bA, bB, ncvA, ncvB, Y, E, shifts, pilot, null_pct, target_side, ncut):

    nA, nB, n_tg = bA.shape[0], bB.shape[0], Y.shape[2]

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

    # --- real data: z and edge of every combination, T per symbol (as pair_T)
    muA, sdA = swap_pair(muA_ba, nB, nA, n_tg, ncut), swap_pair(sdA_ba, nB, nA, n_tg, ncut)
    z, e = pair_z_edges(bA, bB, ncvA, ncvB, Y, muA, sdA, muB, sdB, ncut)
    t_sym = z.max(axis=0)

    null_B = pair_null(bA, bB, ncvA, ncvB, Y, shifts, muA, sdA, muB, sdB, t_sym, ncut)
    muB_ba, sdB_ba = swap_pair(muB, nA, nB, n_tg, ncut), swap_pair(sdB, nA, nB, n_tg, ncut)
    del muA, sdA, muB, sdB

    null_A = pair_null(bB, bA, ncvB, ncvA, Y, shifts, muA_ba, sdA_ba, muB_ba, sdB_ba, t_sym, ncut)
    del muA_ba, sdA_ba, muB_ba, sdB_ba

    # --- pick: the largest edge among the combinations above both floors
    floor = np.maximum(null_floor(null_A, null_pct), null_floor(null_B, null_pct))
    return _pair_result(bA, bB, Y, E, _pick(z.T, e.T, floor), null_A, null_B, target_side, ncut)


def _pair_z_npy(v, osum, mu1, sd1, mu2, sd2, xp):
    """z of every combination as pair_T: the smaller of its two z (-inf where it does not compete)."""
    ok = xp.isfinite(v) & (osum > 0.0) & (sd1 > 0.0) & (sd2 > 0.0)
    z1 = (v - mu1) / xp.where(ok, sd1, 1.0)
    z2 = (v - mu2) / xp.where(ok, sd2, 1.0)
    return xp.where(ok, xp.minimum(z1, z2), -xp.inf)


def pair_null_npy(ctx, shifts, mu1, sd1, mu2, sd2, t_real):
    """As pair_null (B of ctx shifted), NPY statistic on the GPU: every shift at once (the kernel batches them by
    memory), z and its maximum in the kernel. ctx: npy_pair_setup; shifts: npy_shifts."""
    from .screen_kernels_gpu import npy_pair_null

    return npy_pair_null(ctx, shifts, mu1, sd1, mu2, sd2, np.flatnonzero(np.isfinite(t_real)).astype(np.int64))


def screen_pair_npy(bA, bB, ncvA, ncvB, Y, E, prep, shifts, pilot, null_pct, target_side, ncut):
    """As screen_pair, NPY statistic on the GPU. prep: npy_prepare(Y, E) of the real data; shifts and pilot:
    npy_shifts of the phase 2 shifts."""
    import cupy as cp
    from .screen_kernels_gpu import npy_edges, npy_pair_setup, npy_pair_moments

    nA, nB, n_tg = bA.shape[0], bB.shape[0], Y.shape[2]
    ctx_ab = npy_pair_setup(bA, ncvA, bB, ncvB, prep, ncut)                          # layout (A, B): B shifted
    ctx_ba = npy_pair_setup(ctx_ab["bins_b"], ncvB, ctx_ab["bins_a"], ncvA, prep, ncut)  # layout (B, A): A shifted

    # --- pilots: layout (A, B) shifts B, layout (B, A) shifts A. Same valid combinations in both.
    muB, sdB = npy_pair_moments(ctx_ab, pilot, float(MIN_PILOT_N))                  # queued on the GPU
    muA_ba, sdA_ba = npy_pair_moments(ctx_ba, pilot, float(MIN_PILOT_N))
    ok = cp.asarray(pair_valid_mask(bA, bB, ncvA, ncvB, Y, ncut))                    # CPU, while the GPU works
    ok_ba = swap_pair(ok, nA, nB, n_tg, ncut, cp)
    muB, sdB = cp.where(ok, muB, np.nan), cp.where(ok, sdB, np.nan)
    muA_ba, sdA_ba = cp.where(ok_ba, muA_ba, np.nan), cp.where(ok_ba, sdA_ba, np.nan)
    del ok, ok_ba
    ab = (swap_pair(muA_ba, nB, nA, n_tg, ncut, cp), swap_pair(sdA_ba, nB, nA, n_tg, ncut, cp), muB, sdB)
    ba = (muA_ba, sdA_ba, swap_pair(muB, nA, nB, n_tg, ncut, cp), swap_pair(sdB, nA, nB, n_tg, ncut, cp))
    del muB, sdB, muA_ba, sdA_ba

    # --- real data: z and edge of every combination, T per symbol (as pair_T)
    v, osum = npy_edges(ctx_ab["bins_a"], ncvA, prep, ncut, ctx_ab["bins_b"], ncvB)
    z = cp.asnumpy(_pair_z_npy(v, osum, *ab, cp))
    e = cp.asnumpy(osum)
    del v, osum
    t_sym = z.max(axis=0)

    null_B = pair_null_npy(ctx_ab, shifts, *ab, t_sym)
    del ab
    null_A = pair_null_npy(ctx_ba, shifts, *ba, t_sym)
    del ba

    # --- pick: the largest edge among the combinations above both floors
    floor = np.maximum(null_floor(null_A, null_pct), null_floor(null_B, null_pct))
    return _pair_result(bA, bB, Y, E, _pick(z.T, e.T, floor), null_A, null_B, target_side, ncut)


# =============================================================================
# 8. RUN
# =============================================================================
def _phase1_ypy(data, pool, cfg, bins, ncv, Y, E):
    """Phase 1 raw results of every indicator, YPY statistic (CPU)."""
    names, by_ind, target_side = pool.names, pool.by_ind, cfg.target_side
    z_mu, z_sd = build_pilot_moments(data, pool, cfg)
    null1 = build_null_paths(data, pool, cfg, z_mu, z_sd)
    shape = (len(data.symbols), Y.shape[2], pool.ncut, 2)
    p1 = {}
    t1 = time.time()
    for k, nm in enumerate(names):
        idx = by_ind[nm]
        b = np.ascontiguousarray(bins[idx])
        v = np.empty((len(idx),) + shape)
        osum = np.empty((len(idx),) + shape)
        fill_edges(b, ncv[idx], Y, v, osum)
        p1[nm] = screen_indicator(b, Y, E, v, osum, z_mu[idx], z_sd[idx],
                                  null_floor(null1[:, k, :], cfg.null_pct), target_side)
        p1[nm]["null1"] = np.ascontiguousarray(null1[:, k, :])
        progress(f"Phase 1, {len(names)} indicators", k + 1, len(names), t1)
    return p1


def _phase1_npy(data, pool, cfg, bins, ncv, Y, E):
    """Phase 1 raw results of every indicator, NPY statistic (GPU)."""
    import cupy as cp
    from .screen_kernels_gpu import npy_prepare, npy_edges

    names, by_ind, target_side = pool.names, pool.by_ind, cfg.target_side
    z_mu, z_sd = build_pilot_moments_npy(data, pool, cfg)
    null1 = build_null_paths_npy(data, pool, cfg, z_mu, z_sd)
    v, osum = (cp.asnumpy(x) for x in npy_edges(bins, ncv, npy_prepare(Y, E), pool.ncut))
    p1 = {}
    t1 = time.time()
    for k, nm in enumerate(names):
        idx = by_ind[nm]
        p1[nm] = screen_indicator(np.ascontiguousarray(bins[idx]), Y, E, v[idx], osum[idx], z_mu[idx], z_sd[idx],
                                  null_floor(null1[:, k, :], cfg.null_pct), target_side)
        p1[nm]["null1"] = np.ascontiguousarray(null1[:, k, :])
        progress(f"Phase 1 [NPY], {len(names)} indicators", k + 1, len(names), t1)
    return p1


def _phase2_ypy(data, pool, cfg, bins, ncv, Y, E):
    """(pairs, phase 2 raw results of every pair), YPY statistic (CPU)."""
    n, names, by_ind, target_side = data.n, pool.names, pool.by_ind, cfg.target_side
    shifts, pilot = phase2_shifts(n, cfg)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    p2 = {}
    t1 = time.time()
    for k, (a, b) in enumerate(pairs):
        p2[(a, b)] = screen_pair(np.ascontiguousarray(bins[by_ind[a]]), np.ascontiguousarray(bins[by_ind[b]]),
                                 ncv[by_ind[a]], ncv[by_ind[b]], Y, E, shifts, pilot, cfg.null_pct,
                                 target_side, pool.ncut)
        progress(f"Phase 2, {len(pairs)} pairs x 2 nulls", k + 1, len(pairs), t1)
    return pairs, p2


def _phase2_npy(data, pool, cfg, bins, ncv, Y, E):
    """(pairs, phase 2 raw results of every pair), NPY statistic (GPU)."""
    from .screen_kernels_gpu import npy_prepare, npy_shifts

    n, names, by_ind, target_side = data.n, pool.names, pool.by_ind, cfg.target_side
    shifts, pilot = phase2_shifts(n, cfg)
    prep = npy_prepare(Y, E)
    shifts_d, pilot_d = npy_shifts(shifts, n), npy_shifts(pilot, n)                 # on the GPU once
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    p2 = {}
    t1 = time.time()
    for k, (a, b) in enumerate(pairs):
        p2[(a, b)] = screen_pair_npy(np.ascontiguousarray(bins[by_ind[a]]), np.ascontiguousarray(bins[by_ind[b]]),
                                     ncv[by_ind[a]], ncv[by_ind[b]], Y, E, prep, shifts_d, pilot_d,
                                     cfg.null_pct, target_side, pool.ncut)
        progress(f"Phase 2 [NPY], {len(pairs)} pairs x 2 nulls", k + 1, len(pairs), t1)
    return pairs, p2


def compute_raw(data, pool, cfg):
    """Raw results of both phases, in cfg.mode. The selection is done later (screen_report), never cached."""
    t1 = time.time()
    Y = build_targets(data.ohlcv_arr, data, cfg)
    E = build_exits(data.ohlcv_arr, data, cfg)
    bins, ncv, empty = build_bins(data.ohlcv_arr, data, pool)
    logger.info(f"Targets and {len(pool.names)} indicators ({len(pool.instances)} instances): "
                f"{time.time() - t1:.0f}s")
    for key, s in empty:
        logger.info(f"  WARNING: {key} has no values in {s}")

    if cfg.mode == "NPY":
        p1 = _phase1_npy(data, pool, cfg, bins, ncv, Y, E)
        pairs, p2 = _phase2_npy(data, pool, cfg, bins, ncv, Y, E)
    else:
        p1 = _phase1_ypy(data, pool, cfg, bins, ncv, Y, E)
        pairs, p2 = _phase2_ypy(data, pool, cfg, bins, ncv, Y, E)

    return {"names": pool.names, "symbols": list(data.symbols), "n": data.n, "empty": empty, "p1": p1,
            "pairs": pairs, "p2": p2, "null_pct": cfg.null_pct, "mode": cfg.mode,
            "created": datetime.now().isoformat(timespec="seconds")}


def run_screen(ohlcv_arr, symbols, pool, cfg, cache_dir=None, cache_name="screen", cache_tag=None):
    """Raw results of both phases. cache_dir: reused from the cache if nothing that affects them changed
    (cache_key), else computed and saved. cache_tag: {name: value} of what else they depend on (dataset,
    timeframe), in the cache key and in the configuration shown."""
    data = align_symbols(ohlcv_arr, symbols)
    min_n = 2 * cfg.l_shift + 1
    if data.n < min_n:
        raise ValueError(f"Only {data.n} common candles; at least {min_n} are needed")
    config = cache_config(data.symbols, cfg, cache_tag)
    if cache_dir is None:
        return {**compute_raw(data, pool, cfg), "config": config}

    t1 = time.time()
    key = cache_key(data.symbols, cfg, cache_tag)
    path = cache_path(cache_dir, cache_name, cfg, len(data.symbols), key)
    raw = load_cache(path, key, pool.names)
    if raw is not None:
        logger.info(f"🟢 Cache loaded: {os.path.basename(path)} (created {raw['created']}, {time.time() - t1:.0f}s)")
        log_config(raw["config"])
        for k, s in raw["empty"]:
            logger.info(f"  WARNING: {k} has no values in {s}")
        return raw
    logger.info(f"🟡 No cache for this configuration, computing: {os.path.basename(path)}")
    log_config(config)
    logger.info("")
    raw = {**compute_raw(data, pool, cfg), "config": config, "key": key}
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


CODE_DIR  = os.path.dirname(os.path.abspath(__file__))                           # core/research/screening
CORE_DIR  = os.path.dirname(os.path.dirname(CODE_DIR))                           # core
CODE_DIRS = (                                                                    # their code is in the cache key
    (CODE_DIR, os.path.dirname(CODE_DIR)),                                       # (folder, root of its paths): screening/...
    (os.path.join(CORE_DIR, "indicators"), CORE_DIR),                            # indicators/...
)
CODE_EXCLUDE = (os.path.join(CODE_DIR, "screen_report.py"),)                     # does not change the raw results
def cache_key(symbols, cfg, tag=None):
    """Only the tag, the symbols, cfg.identity() (TP/SL/SELL_AFTER grid, commission, mode, N_NULL_PATHS, N_PILOTS,
    seeds and lookback), NULL_PCT and the code of CODE_DIRS."""
    h = hashlib.sha256()
    _h_update(h, tuple(dict(tag or {}).items()))
    _h_update(h, (list(symbols), cfg.identity(), float(cfg.null_pct)))
    for rel, f in _code_files():
        _h_update(h, rel)
        h.update(_code_digest(f))
    return h.hexdigest()


def _code_files():
    skip = {os.path.normcase(os.path.abspath(p)) for p in CODE_EXCLUDE}
    files = []
    for top, root in CODE_DIRS:
        if not os.path.isdir(top):
            raise FileNotFoundError(f"Code folder of the cache key not found: {top}")
        for d, subdirs, names in os.walk(top):
            subdirs[:] = sorted(x for x in subdirs if x != "__pycache__" and not x.startswith("."))
            for nm in names:
                f = os.path.join(d, nm)
                if nm.endswith(".py") and os.path.normcase(os.path.abspath(f)) not in skip:
                    files.append((os.path.relpath(f, root).replace(os.sep, "/"), f))
    return sorted(files)


def _code_digest(path):
    """Code of a file as its AST without docstrings: comments, blank lines and formatting do not count."""
    with open(path, "rb") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree).encode()


def cache_path(cache_dir, name, cfg, n_sym, key):
    """<name>_<MODE>_null<NULL_PCT>_sa<SELL_AFTER>_tp<TP_PCT>_sl<SL_PCT>_<n>sym_<hash>.pkl. The hash (start of the
    key) tells apart what is not in the name: which symbols, N_NULL_PATHS, N_PILOTS, commission, code."""
    sa = "-".join(str(int(v)) for v in cfg.sell_after)
    tp = "-".join(str(float(v)) for v in cfg.tp_pct)
    sl = "-".join(str(float(v)) for v in cfg.sl_pct)
    return os.path.join(cache_dir, f"{name}_{cfg.mode}_null{float(cfg.null_pct):g}_sa{sa}_tp{tp}_sl{sl}"
                                   f"_{n_sym}sym_{key[:8]}.pkl")


def cache_config(symbols, cfg, tag=None):
    """What the raw results were computed with (saved in the cache and shown): the tag, the mode, the number of
    symbols, the TP/SL/SELL_AFTER grid, NULL_PCT, N_NULL_PATHS, N_PILOTS and the commission."""
    return {**dict(tag or {}), "MODE": cfg.mode, "SYMBOLS": len(symbols), "SELL_AFTER": list(cfg.sell_after),
            "TP_PCT": list(cfg.tp_pct), "SL_PCT": list(cfg.sl_pct), "NULL_PCT": cfg.null_pct,
            "N_NULL_PATHS": cfg.n_null_paths, "N_PILOTS": cfg.n_pilots, "COMMISSION": cfg.commission}


_CONFIG_LINES = (("MODE", "SYMBOLS"), ("SELL_AFTER", "TP_PCT", "SL_PCT"),
                 ("NULL_PCT", "N_NULL_PATHS", "N_PILOTS", "COMMISSION"))


def log_config(config):
    """The configuration of cache_config in three lines, the tag first."""
    tag = [k for k in config if not any(k in keys for keys in _CONFIG_LINES)]
    for keys in ((*tag, *_CONFIG_LINES[0]),) + _CONFIG_LINES[1:]:
        logger.info("   " + " ".join(f"{k}={config[k]}" for k in keys if k in config))


def load_cache(path, key, names):

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
    return raw


def save_cache(path, raw):
    """Atomic write: a run cut halfway never leaves a broken cache file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(raw, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)