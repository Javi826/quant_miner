#quant_minr/core/indicators/screening_engine.py (forex)
import os
import sys
import time
import logging
import numpy as np
from numba import njit, prange

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, instance_key

logger = logging.getLogger("BOT_research.screening")


# =============================================================================
# 1. SYNTHETIC PATHS
# =============================================================================
from indicators.indicators_pool import _price_decimals


def i64(a):
    return np.asarray(a).astype("datetime64[ns]").view("int64")


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
# 2. KERNELS
# =============================================================================
# Constants frozen into the compiled code. Derived, except MIN_SEG and MIN_PILOT_N.
MIN_SEG     = 30  # min candles in a segment AND min candles it leaves out (its std, and not a test on a few)
MIN_PILOT_N = 30  # min pilot paths/shifts where a combination is valid (its null std needs enough of them)

NCUT    = max(len(set(m["thresholds"])) for m in CANDIDATE_REGISTRY.values())  # max cuts per indicator
NBIN    = 2 * NCUT + 1               # segments: x < c0, x == c0, c0 < x < c1, x == c1, ..., x > c_last
NAT_I64 = np.iinfo(np.int64).min     # NaT viewed as int64


# --- 2.1 TARGET
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


# --- 2.2 EDGE AND T
# Everything is per symbol: the backtest searches a rule for each symbol, so there is no
# cross-symbol pooling here. Breadth is required later, by GROUP_N, not inside the statistic.

@njit(cache=True)
def _seg_edge(n, s, q, mean, sd, n_all):

    if n < MIN_SEG or n > n_all - MIN_SEG:        # both sides of the cut need MIN_SEG candles
        return np.nan
    m = s / n
    var = q / n - m * m
    sd_seg = np.sqrt(var) if var > 0.0 else 0.0
    return (m - mean) / (max(sd_seg, sd) / np.sqrt(n))


@njit(cache=True)
def fill_edges(bins, ncv, Y, out, osum):

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
                for g in range(n_tg):
                    y = Y[s, t, g]
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
                        osum[i, s, g, c, 0] = 0.0
                        osum[i, s, g, c, 1] = 0.0
                    continue

                n_lo = 0.0
                s_lo = 0.0
                q_lo = 0.0
                for c in range(NCUT):
                    if c >= ncv[i]:
                        out[i, s, g, c, 0] = np.nan
                        out[i, s, g, c, 1] = np.nan
                        osum[i, s, g, c, 0] = 0.0
                        osum[i, s, g, c, 1] = 0.0
                        continue
                    # x < c_c: bins 0..2c. Bins 0..2c-2 are already accumulated.
                    if c > 0:
                        n_lo += cnt[2 * c - 1, g]
                        s_lo += sm[2 * c - 1, g]
                        q_lo += sq[2 * c - 1, g]
                    n_lo += cnt[2 * c, g]
                    s_lo += sm[2 * c, g]
                    q_lo += sq[2 * c, g]
                    # x > c_c: everything except x < c_c and x == c_c (bin 2c+1)
                    n_hi = n_all - n_lo - cnt[2 * c + 1, g]
                    s_hi = s_all - s_lo - sm[2 * c + 1, g]
                    q_hi = q_all - q_lo - sq[2 * c + 1, g]
                    out[i, s, g, c, 0] = _seg_edge(n_lo, s_lo, q_lo, mean, sd, n_all)
                    out[i, s, g, c, 1] = _seg_edge(n_hi, s_hi, q_hi, mean, sd, n_all)
                    osum[i, s, g, c, 0] = s_lo if out[i, s, g, c, 0] == out[i, s, g, c, 0] else 0.0
                    osum[i, s, g, c, 1] = s_hi if out[i, s, g, c, 1] == out[i, s, g, c, 1] else 0.0


@njit(cache=True)
def reduce_edges(edges, osum, z_mu, z_sd):
    """Max z per symbol, only over combinations whose trades make money on average (segment mean > 0).
    z = (edge - z_mu) / z_sd: every combination measured against the null of its own edge (pilot), so the
    noisiest ones (long SA, slow params) do not win the max just for being noisy.
    A combination without a usable null (z_sd NaN) does not compete. Each symbol picks its own winner."""
    n_inst, n_sym, n_tg, n_cut, _ = edges.shape
    t_sym = np.full(n_sym, -np.inf)
    arg_sym = np.full(n_sym, -1, dtype=np.int64)

    for i in range(n_inst):
        for g in range(n_tg):
            for c in range(n_cut):
                for side in range(2):
                    flat = ((i * n_tg + g) * n_cut + c) * 2 + side
                    for s in range(n_sym):
                        v = edges[i, s, g, c, side]
                        d = z_sd[i, s, g, c, side]
                        if v == v and osum[i, s, g, c, side] > 0.0 and d > 0.0:
                            z = (v - z_mu[i, s, g, c, side]) / d
                            if z > t_sym[s]:
                                t_sym[s] = z
                                arg_sym[s] = flat
    return t_sym, arg_sym


@njit(parallel=True, cache=True)
def path_T_all(bins, ncv, Y, starts, ends, z_mu, z_sd):
    """Per-symbol T of every indicator on one synthetic path.
    starts/ends: slice of instances of each indicator (they are contiguous in build_flat_instances).
    z_mu, z_sd: null mean and std of every combination of all the instances (pilot)."""
    n_ind = starts.shape[0]
    n_sym = bins.shape[1]
    n_tg = Y.shape[2]
    t_sym = np.empty((n_ind, n_sym))
    for i in prange(n_ind):
        b = bins[starts[i]:ends[i]]
        c = ncv[starts[i]:ends[i]]
        n_inst = b.shape[0]
        edges = np.empty((n_inst, n_sym, n_tg, NCUT, 2))
        osum = np.empty((n_inst, n_sym, n_tg, NCUT, 2))
        fill_edges(b, c, Y, edges, osum)
        ts, _a = reduce_edges(edges, osum, z_mu[starts[i]:ends[i]], z_sd[starts[i]:ends[i]])
        for s in range(n_sym):
            t_sym[i, s] = ts[s]
    return t_sym


# --- 2.3 NULL OF EACH COMBINATION (pilot, for z)
# The pilot uses its own synthetic paths (phase 1) or shifts (phase 2), never the ones of the floor.
# No mean > 0 gate here: it measures the whole null distribution of each edge. The gate is applied later,
# when taking the max, exactly as before (real data and floor).

@njit(cache=True)
def _moments(n, s, q, min_n):
    """Mean and std (ddof 1) from count, sum and sum of squares. NaN if n < min_n or there is no spread."""
    if n < min_n or n < 2.0:
        return np.nan, np.nan
    m = s / n
    var = (q - n * m * m) / (n - 1.0)
    if var > 0.0:
        return m, np.sqrt(var)
    return np.nan, np.nan


@njit(cache=True)
def moments_inplace(m_n, m_s, m_q, min_n):
    """1D arrays, in place: m_s becomes the mean and m_q the std of every combination (NaN if not usable)."""
    for j in range(m_n.shape[0]):
        m, d = _moments(m_n[j], m_s[j], m_q[j], min_n)
        m_s[j] = m
        m_q[j] = d


@njit(parallel=True, cache=True)
def path_edges_all(bins, ncv, Y, starts, ends):
    """Phase 1 pilot, one synthetic path: edge of every combination (instance, symbol, target, cut, side)
    of all the instances, NaN where not valid. Same values fill_edges gives indicator by indicator."""
    n_ind = starts.shape[0]
    n_all, n_sym, _n = bins.shape
    n_tg = Y.shape[2]
    edges = np.empty((n_all, n_sym, n_tg, NCUT, 2))
    for i in prange(n_ind):
        b = bins[starts[i]:ends[i]]
        c = ncv[starts[i]:ends[i]]
        n_inst = b.shape[0]
        e = np.empty((n_inst, n_sym, n_tg, NCUT, 2))
        osum = np.empty((n_inst, n_sym, n_tg, NCUT, 2))
        fill_edges(b, c, Y, e, osum)
        edges[starts[i]:ends[i]] = e
    return edges


@njit(cache=True)
def add_edge_moments(edges, m_n, m_s, m_q):
    """Phase 1 pilot: adds the valid edges of one path to the running count, sum and sum of squares of every
    combination. Called once per path IN PATH ORDER, so the sums are bit-identical whoever computed the edges."""
    e = edges.reshape(-1)
    a_n = m_n.reshape(-1)
    a_s = m_s.reshape(-1)
    a_q = m_q.reshape(-1)
    for j in range(e.shape[0]):
        v = e[j]
        if v == v:
            a_n[j] += 1.0
            a_s[j] += v
            a_q[j] += v * v


# --- 2.4 PHASE 2: PAIRS A + B (every indicator with every other one, unordered)
# Rule A AND B. The kernels shift their SECOND indicator (B) by k; A and the targets stay aligned.
# To shift A instead, call them with A and B swapped (and the per-combination arrays in the swapped layout).

@njit(cache=True)
def _pair_hist(bA, bB, iA, iB, s, Y, k, P_n, P_s, P_q, mean, sd, n_all):
    """Phase 2 pilot and valid mask: the histogram of pair_T for ONE symbol (same arithmetic). 2D histogram (A bin,
    B bin) per target, B shifted by k, then 2D prefix sums, plus the mean, std and count of the valid
    candles per target. P_*: (NBIN + 1, NBIN + 1, targets); mean, sd, n_all: (targets,). All overwritten.
    pair_T keeps its own inline copy: calling this from there is ~8% slower."""
    n = bA.shape[2]
    n_tg = Y.shape[2]
    nb = NBIN + 1
    P_n[:] = 0.0
    P_s[:] = 0.0
    P_q[:] = 0.0
    for t in range(n):
        b1 = bA[iA, s, t]
        if b1 < 0:
            continue
        j = t + k
        if j >= n:
            j -= n
        b2 = bB[iB, s, j]
        if b2 < 0:
            continue
        for g in range(n_tg):
            y = Y[s, t, g]
            if y == y:
                P_n[b1 + 1, b2 + 1, g] += 1.0
                P_s[b1 + 1, b2 + 1, g] += y
                P_q[b1 + 1, b2 + 1, g] += y * y
    for a in range(1, nb):
        for b in range(nb):
            for g in range(n_tg):
                P_n[a, b, g] += P_n[a - 1, b, g]
                P_s[a, b, g] += P_s[a - 1, b, g]
                P_q[a, b, g] += P_q[a - 1, b, g]
    for a in range(nb):
        for b in range(1, nb):
            for g in range(n_tg):
                P_n[a, b, g] += P_n[a, b - 1, g]
                P_s[a, b, g] += P_s[a, b - 1, g]
                P_q[a, b, g] += P_q[a, b - 1, g]
    for g in range(n_tg):
        na = P_n[NBIN, NBIN, g]
        n_all[g] = na
        mean[g] = 0.0
        sd[g] = 0.0
        if na > 1.0:
            m = P_s[NBIN, NBIN, g] / na
            var = P_q[NBIN, NBIN, g] / na - m * m
            mean[g] = m
            if var > 0.0:
                sd[g] = np.sqrt(var)


@njit(cache=True)
def pair_T(bA, bB, ncvA, ncvB, Y, k, z1_mu, z1_sd, z2_mu, z2_sd, sym_idx):
    """Per-symbol T2 and winners of a pair A + B, B shifted by k.
    z1_*, z2_*: (flat, symbols) null mean and std of the edge of every combination under the pair's two pilots
    (one shifts A, the other B), in this call's layout. Each combination scores the SMALLER of its two z: it has
    to stand out against both nulls. T2 is the best score. Only combinations whose trades make money on average
    and with both nulls usable compete (as in reduce_edges).
    sym_idx: symbols to compute (int64). The rest stay at -inf / -1. A computed symbol gets the same value
    whatever the other symbols in sym_idx."""
    nA, n_sym, n = bA.shape
    nB = bB.shape[0]
    n_tg = Y.shape[2]
    nb = NBIN + 1
    n_s = sym_idx.shape[0]
    P_n = np.empty((n_s, nb, nb, n_tg))
    P_s = np.empty((n_s, nb, nb, n_tg))
    P_q = np.empty((n_s, nb, nb, n_tg))
    mean = np.empty((n_s, n_tg))
    sd = np.empty((n_s, n_tg))
    n_all = np.empty((n_s, n_tg))

    t_sym = np.full(n_sym, -np.inf)
    arg_sym = np.full(n_sym, -1, dtype=np.int64)

    for iA in range(nA):
        for iB in range(nB):
            # --- 2D histogram (A bin, B bin) per symbol and target, then 2D prefix sums
            P_n[:] = 0.0
            P_s[:] = 0.0
            P_q[:] = 0.0
            for si in range(n_s):
                s = sym_idx[si]
                for t in range(n):
                    b1 = bA[iA, s, t]
                    if b1 < 0:
                        continue
                    j = t + k
                    if j >= n:
                        j -= n
                    b2 = bB[iB, s, j]
                    if b2 < 0:
                        continue
                    for g in range(n_tg):
                        y = Y[s, t, g]
                        if y == y:
                            P_n[si, b1 + 1, b2 + 1, g] += 1.0
                            P_s[si, b1 + 1, b2 + 1, g] += y
                            P_q[si, b1 + 1, b2 + 1, g] += y * y
                for a in range(1, nb):
                    for b in range(nb):
                        for g in range(n_tg):
                            P_n[si, a, b, g] += P_n[si, a - 1, b, g]
                            P_s[si, a, b, g] += P_s[si, a - 1, b, g]
                            P_q[si, a, b, g] += P_q[si, a - 1, b, g]
                for a in range(nb):
                    for b in range(1, nb):
                        for g in range(n_tg):
                            P_n[si, a, b, g] += P_n[si, a, b - 1, g]
                            P_s[si, a, b, g] += P_s[si, a, b - 1, g]
                            P_q[si, a, b, g] += P_q[si, a, b - 1, g]
                for g in range(n_tg):
                    na = P_n[si, NBIN, NBIN, g]
                    n_all[si, g] = na
                    mean[si, g] = 0.0
                    sd[si, g] = 0.0
                    if na > 1.0:
                        m = P_s[si, NBIN, NBIN, g] / na
                        var = P_q[si, NBIN, NBIN, g] / na - m * m
                        mean[si, g] = m
                        if var > 0.0:
                            sd[si, g] = np.sqrt(var)

            # --- every combination of A cut/side and B cut/side
            for g in range(n_tg):
                for cA in range(ncvA[iA]):
                    for sA in range(2):
                        if sA == 0:
                            a0, a1 = 0, 2 * cA + 1                 # x < c
                        else:
                            a0, a1 = 2 * cA + 2, NBIN              # x > c
                        for cB in range(ncvB[iB]):
                            for sB in range(2):
                                if sB == 0:
                                    b0, b1 = 0, 2 * cB + 1
                                else:
                                    b0, b1 = 2 * cB + 2, NBIN
                                flat = (((((iA * nB + iB) * n_tg + g) * NCUT + cA) * 2 + sA) * NCUT + cB) * 2 + sB
                                for si in range(n_s):
                                    if sd[si, g] <= 0.0:
                                        continue
                                    s = sym_idx[si]
                                    nn = P_n[si, a1, b1, g] - P_n[si, a0, b1, g] - P_n[si, a1, b0, g] + P_n[si, a0, b0, g]
                                    ss = P_s[si, a1, b1, g] - P_s[si, a0, b1, g] - P_s[si, a1, b0, g] + P_s[si, a0, b0, g]
                                    qq = P_q[si, a1, b1, g] - P_q[si, a0, b1, g] - P_q[si, a1, b0, g] + P_q[si, a0, b0, g]
                                    v = _seg_edge(nn, ss, qq, mean[si, g], sd[si, g], n_all[si, g])
                                    d1 = z1_sd[flat, s]
                                    d2 = z2_sd[flat, s]
                                    if v == v and ss > 0.0 and d1 > 0.0 and d2 > 0.0:
                                        z = min((v - z1_mu[flat, s]) / d1, (v - z2_mu[flat, s]) / d2)
                                        if z > t_sym[s]:
                                            t_sym[s] = z
                                            arg_sym[s] = flat
    return t_sym, arg_sym


@njit(parallel=True, cache=True)
def pair_null_distribution(bA, bB, ncvA, ncvB, Y, shifts, z1_mu, z1_sd, z2_mu, z2_sd, sym_idx):
    """Per-symbol T2 of a pair for every null shift of B. Only the symbols in sym_idx (the rest -inf)."""
    n_sym = bA.shape[1]
    n_k = shifts.shape[0]
    t_sym = np.empty((n_k, n_sym))
    for q in prange(n_k):
        ts, _a = pair_T(bA, bB, ncvA, ncvB, Y, shifts[q], z1_mu, z1_sd, z2_mu, z2_sd, sym_idx)
        for s in range(n_sym):
            t_sym[q, s] = ts[s]
    return t_sym


@njit(parallel=True, cache=True)
def pair_edge_moments(bA, bB, ncvA, ncvB, Y, shifts, min_n):
    """Phase 2 pilot: null mean and std of the edge of every combination of a pair, over its own shifts of B.
    Returns z_mu, z_sd: (flat, symbols), flat as in pair_T; NaN where the combination is valid in fewer
    than min_n shifts. Parallel over (symbol, A instance, B instance): each one owns a disjoint slice."""
    nA, n_sym, n = bA.shape
    nB = bB.shape[0]
    n_tg = Y.shape[2]
    nb = NBIN + 1
    n_blk = n_tg * NCUT * 2 * NCUT * 2                     # combinations of one (A, B) instance pair
    z_mu = np.full((nA * nB * n_blk, n_sym), np.nan)
    z_sd = np.full((nA * nB * n_blk, n_sym), np.nan)
    for u in prange(n_sym * nA * nB):
        s = u // (nA * nB)
        iA = (u // nB) % nA
        iB = u % nB
        off = (iA * nB + iB) * n_blk
        P_n = np.empty((nb, nb, n_tg))
        P_s = np.empty((nb, nb, n_tg))
        P_q = np.empty((nb, nb, n_tg))
        mean = np.empty(n_tg)
        sd = np.empty(n_tg)
        n_all = np.empty(n_tg)
        c_n = np.zeros(n_blk)
        c_s = np.zeros(n_blk)
        c_q = np.zeros(n_blk)
        for q in range(shifts.shape[0]):
            _pair_hist(bA, bB, iA, iB, s, Y, shifts[q], P_n, P_s, P_q, mean, sd, n_all)
            for g in range(n_tg):
                if sd[g] <= 0.0:
                    continue
                for cA in range(ncvA[iA]):
                    for sA in range(2):
                        if sA == 0:
                            a0, a1 = 0, 2 * cA + 1                 # x < c
                        else:
                            a0, a1 = 2 * cA + 2, NBIN              # x > c
                        for cB in range(ncvB[iB]):
                            for sB in range(2):
                                if sB == 0:
                                    b0, b1 = 0, 2 * cB + 1
                                else:
                                    b0, b1 = 2 * cB + 2, NBIN
                                nn = P_n[a1, b1, g] - P_n[a0, b1, g] - P_n[a1, b0, g] + P_n[a0, b0, g]
                                ss = P_s[a1, b1, g] - P_s[a0, b1, g] - P_s[a1, b0, g] + P_s[a0, b0, g]
                                qq = P_q[a1, b1, g] - P_q[a0, b1, g] - P_q[a1, b0, g] + P_q[a0, b0, g]
                                v = _seg_edge(nn, ss, qq, mean[g], sd[g], n_all[g])
                                if v == v:
                                    j = (((g * NCUT + cA) * 2 + sA) * NCUT + cB) * 2 + sB
                                    c_n[j] += 1.0
                                    c_s[j] += v
                                    c_q[j] += v * v
        for j in range(n_blk):
            m, d = _moments(c_n[j], c_s[j], c_q[j], min_n)
            z_mu[off + j, s] = m
            z_sd[off + j, s] = d
    return z_mu, z_sd


@njit(parallel=True, cache=True)
def pair_valid_mask(bA, bB, ncvA, ncvB, Y):
    """Phase 2: combinations of a pair that are a real combination, decided on the real data (nothing shifted).
    A AND B must leave out >= MIN_SEG candles of A and >= MIN_SEG candles of B. If not, the rule is one of the
    two indicators alone (phase 1 already tests it) and its null has no spread: shifting the one that covers
    (almost) every candle changes nothing, the pilot std tends to 0 and z blows up.
    Symmetric in A and B: swapping them gives the same mask in the swapped layout.
    Returns ok (flat, symbols), flat as in pair_T."""
    nA, n_sym, _n = bA.shape
    nB = bB.shape[0]
    n_tg = Y.shape[2]
    nb = NBIN + 1
    n_blk = n_tg * NCUT * 2 * NCUT * 2                     # combinations of one (A, B) instance pair
    ok = np.zeros((nA * nB * n_blk, n_sym), dtype=np.bool_)
    for u in prange(n_sym * nA * nB):
        s = u // (nA * nB)
        iA = (u // nB) % nA
        iB = u % nB
        off = (iA * nB + iB) * n_blk
        P_n = np.empty((nb, nb, n_tg))
        P_s = np.empty((nb, nb, n_tg))
        P_q = np.empty((nb, nb, n_tg))
        mean = np.empty(n_tg)
        sd = np.empty(n_tg)
        n_all = np.empty(n_tg)
        _pair_hist(bA, bB, iA, iB, s, Y, 0, P_n, P_s, P_q, mean, sd, n_all)
        for g in range(n_tg):
            if sd[g] <= 0.0:
                continue                                   # no combination competes there anyway
            for cA in range(ncvA[iA]):
                for sA in range(2):
                    if sA == 0:
                        a0, a1 = 0, 2 * cA + 1                     # x < c
                    else:
                        a0, a1 = 2 * cA + 2, NBIN                  # x > c
                    n_seg_a = P_n[a1, NBIN, g] - P_n[a0, NBIN, g]  # A segment, any B bin
                    for cB in range(ncvB[iB]):
                        for sB in range(2):
                            if sB == 0:
                                b0, b1 = 0, 2 * cB + 1
                            else:
                                b0, b1 = 2 * cB + 2, NBIN
                            n_seg_b = P_n[NBIN, b1, g] - P_n[NBIN, b0, g]
                            nn = P_n[a1, b1, g] - P_n[a0, b1, g] - P_n[a1, b0, g] + P_n[a0, b0, g]
                            good = (n_seg_a - nn >= MIN_SEG) and (n_seg_b - nn >= MIN_SEG)
                            j = (((g * NCUT + cA) * 2 + sA) * NCUT + cB) * 2 + sB
                            ok[off + j, s] = good
    return ok


def swap_pair(x, nA, nB, n_tg):
    """Per-combination array of a pair (flat, symbols) in layout (A, B) -> the same values in layout (B, A)."""
    n_sym = x.shape[1]
    y = x.reshape(nA, nB, n_tg, NCUT, 2, NCUT, 2, n_sym).transpose(1, 0, 2, 5, 6, 3, 4, 7)
    return np.ascontiguousarray(y).reshape(nA * nB * n_tg * NCUT * 2 * NCUT * 2, n_sym)


# --- 2.5 EARLY STOP OF THE PHASE 2 NULLS
# A symbol passes only if T2 > floor (percentile NULL_PCT of its null). Once enough null values are >= T2 the
# floor is >= T2 whatever the remaining shifts give: that symbol fails for sure and its null stops there.
# The symbols that do not fail get their whole null, with the same values as without the early stop, so the
# selection is identical. The cache then holds for any NULL_PCT >= the one it was computed with.
NULL_BATCH = 32   # shifts per batch (in order). Speed only: the selection does not depend on it


def null_stop_count(n_null, null_pct):
    """Null values >= T2 that make a symbol fail for sure at NULL_PCT = null_pct with n_null values.
    np.percentile (linear) puts the floor between the sorted values lo and lo + 1, lo = floor((n - 1) * pct / 100),
    and never below the value lo. With n - lo values >= T2, the value lo is >= T2 and so is the floor.
    The 1e-9 keeps lo on the safe side (never above numpy's) when (n - 1) * pct / 100 falls on an integer."""
    lo = int(np.floor((n_null - 1) * (null_pct / 100) - 1e-9))
    lo = min(max(lo, 0), n_null - 1)
    return n_null - lo


def pair_null_early(bA, bB, ncvA, ncvB, Y, shifts, z1_mu, z1_sd, z2_mu, z2_sd, t_real, alive, m_stop):
    """Null of a pair (B shifted, as pair_null_distribution) with early stop. Shifts in batches of NULL_BATCH, in
    order; only the symbols in alive are computed, and one stops as soon as m_stop of its null values are >= its
    real T2 (t_real). Returns the null (shifts, symbols), NaN where not computed, and alive updated: the symbols
    that got to the last shift without failing. Their null is complete."""
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
        ts = pair_null_distribution(bA, bB, ncvA, ncvB, Y, sh, z1_mu, z1_sd, z2_mu, z2_sd, sym_idx)[:, sym_idx]
        null[q0:q0 + sh.shape[0], sym_idx] = ts
        hits[sym_idx] += (ts >= t_real[sym_idx]).sum(axis=0)
        alive[sym_idx] = hits[sym_idx] < m_stop
    return null, alive


# =============================================================================
# 3. OUTPUT
# =============================================================================
SEP = "=" * 124


def progress(label, k, total, t0):
    """Progress line of one stage, rewritten in place. When the stage ends, its time (t0: its start)."""
    if not logger.isEnabledFor(logging.INFO):
        return
    sys.stdout.write(f"\r{label}: {k}/{total} ({100 * k // max(total, 1)}%)")
    if k >= total:
        sys.stdout.write(f"  {(time.time() - t0) / 60:.1f} min\n")
    sys.stdout.flush()


def pair_name(a, b):
    return f"{a} + {b}"


def _fmt(v, w=6):
    return f"{v:>{w}.2f}" if np.isfinite(v) else f"{'-':>{w}}"


def _fmt_mean(v, w=8):
    return f"{v:>+{w}.3f}" if np.isfinite(v) else f"{'-':>{w}}"


def _group(name):
    return GROUP_NAMES.get(CANDIDATE_REGISTRY[name]["group"], CANDIDATE_REGISTRY[name]["group"])


def _py_list(var_name, items):
    """Render items as a copy-pasteable Python list literal."""
    lines = [f"{var_name} = ["]
    for it in items:
        lines.append(f'    "{it}",')
    lines.append("]")
    return "\n".join(lines)


def _nan_low(v):
    """Sort key that pushes NaN to the bottom of a descending order."""
    return v if np.isfinite(v) else -np.inf


# --- SELECTION BY ROLE (the only output)
# One rule for both roles and both phases: an indicator is selected when it passes in >= GROUP_N symbols,
# alone (phase 1) or inside a pair (phase 2, both nulls). A passing pair selects both of its indicators.
def _role_header():
    return (f"{'indicator':<26}{'group':<20}{'n_pass':>7}{'score':>8}{'mean%':>9}"
            f"  {'via':<48}")


def _role_row(nm, n_pass, score, mean, via):
    return (f"{nm:<26}{_group(nm):<20}{n_pass:>7}{_fmt(score, 8)}{_fmt_mean(mean, 9)}"
            f"  {via:<48}")


def report_roles(names, res1, res2, group_n):
    """The selection handed to the backtest, split by registry role and ranked by score.
    res1: indicator -> phase 1, res2: pair -> phase 2; each one {"n_pass", "score", "mean"}.
    Every indicator is shown in its best version (highest score) among those passing: alone or a pair."""
    best = {}
    for nm in names:
        cand = []
        r = res1[nm]
        if r["n_pass"] >= group_n:
            cand.append((r["score"], r["n_pass"], r["mean"], "alone"))
        for p, q in res2.items():
            if nm in p and q["n_pass"] >= group_n:
                cand.append((q["score"], q["n_pass"], q["mean"], pair_name(*p)))
        if cand:
            best[nm] = max(cand, key=lambda c: _nan_low(c[0]))

    for title, role, var in (("SIGNALS", "signal", "SYMBOL_SIGNAL"),
                             ("FILTERS", "filter", "SYMBOL_FILTER")):
        items = [nm for nm in names if CANDIDATE_REGISTRY[nm]["role"] == role and nm in best]
        items.sort(key=lambda nm: -_nan_low(best[nm][0]))
        logger.info(f"\n{SEP}\n{title} ({len(items)}) | passing in >= {group_n} symbols, "
                    f"alone or inside a pair; ranked by score\n{SEP}")
        logger.info(_role_header())
        for nm in items:
            score, n_pass, mean, via = best[nm]
            logger.info(_role_row(nm, n_pass, score, mean, via))
        if not items:
            logger.info("(empty)")
        logger.info("\n" + _py_list(var, items))