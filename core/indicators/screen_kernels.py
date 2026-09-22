# core/indicators/screen_kernels.py
"""Numba kernels of the indicator screening. Market agnostic: they only see bins, targets and segment stats.

Layouts
    bins[i, s, t]              int8 bin of instance i, symbol s, candle t (-1: no value). With cuts c_0 < ... < c_k:
                               x < c_0 -> 0, x == c_0 -> 1, c_0 < x < c_1 -> 2, ..., x > c_k -> 2k + 2.
    Y[s, t, g]                 net result (%) of target g entering at candle t (NaN: not computable).
    phase 1  (i, s, g, c, side)                        side 0: x < c, side 1: x > c
    phase 2  flat = off(iA, iB) + combo(g, cA, sA, cB, sB), per symbol
ncut (max cuts per indicator) is an argument, never a global: numba freezes globals when it compiles.
"""
import numpy as np
from numba import njit, prange

MIN_SEG     = 30  # min candles in a segment AND min candles it leaves out (its std, and not a test on a few)
MIN_PILOT_N = 30  # min pilot paths/shifts where a combination is valid (its null std needs enough of them)
NAT_I64     = np.iinfo(np.int64).min     # NaT viewed as int64


# =============================================================================
# 1. TARGETS
# =============================================================================
@njit(cache=True)
def compute_targets(close, high, low, high_time, low_time, tp_pct, sl_pct, sell_after, comm_pct):

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
        out_long[t] = res - comm_pct * (2.0 + res / 100.0)    # net: commission on entry and exit notional

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
        out_short[t] = res - comm_pct * (2.0 - res / 100.0)   # net: exit notional shrinks when a short wins

    return out_long, out_short


# =============================================================================
# 2. SEGMENT STATISTICS
# =============================================================================
@njit(cache=True)
def _seg_edge(n, s, q, mean, sd, n_all):

    if n < MIN_SEG or n > n_all - MIN_SEG:        # both sides of the cut need MIN_SEG candles
        return np.nan
    m = s / n
    var = q / n - m * m
    sd_seg = np.sqrt(var) if var > 0.0 else 0.0
    return (m - mean) / (max(sd_seg, sd) / np.sqrt(n))


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


@njit(cache=True)
def add_edge_moments(edges, m_n, m_s, m_q):

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


# =============================================================================
# 3. PHASE 1: ONE INDICATOR
# =============================================================================
@njit(cache=True)
def fill_edges(bins, ncv, Y, out, osum):
    """Edge of every (instance, symbol, target, cut, side) into out, segment sum into osum (0 where no edge).
    out and osum: (n_inst, n_sym, n_tg, ncut, 2)."""
    n_inst, n_sym, n = bins.shape
    n_tg = Y.shape[2]
    ncut = out.shape[3]
    nbin = 2 * ncut + 1
    cnt = np.empty((nbin, n_tg))
    sm = np.empty((nbin, n_tg))
    sq = np.empty((nbin, n_tg))

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
                for b in range(nbin):
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
                    for c in range(ncut):
                        out[i, s, g, c, 0] = np.nan
                        out[i, s, g, c, 1] = np.nan
                        osum[i, s, g, c, 0] = 0.0
                        osum[i, s, g, c, 1] = 0.0
                    continue

                n_lo = 0.0
                s_lo = 0.0
                q_lo = 0.0
                for c in range(ncut):
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
def path_edges_all(bins, ncv, Y, starts, ends, ncut):
    """Edges of every instance on one path; indicator i owns instances [starts[i], ends[i])."""
    n_ind = starts.shape[0]
    n_all, n_sym, _n = bins.shape
    n_tg = Y.shape[2]
    edges = np.empty((n_all, n_sym, n_tg, ncut, 2))
    for i in prange(n_ind):
        b = bins[starts[i]:ends[i]]
        c = ncv[starts[i]:ends[i]]
        n_inst = b.shape[0]
        e = np.empty((n_inst, n_sym, n_tg, ncut, 2))
        osum = np.empty((n_inst, n_sym, n_tg, ncut, 2))
        fill_edges(b, c, Y, e, osum)
        edges[starts[i]:ends[i]] = e
    return edges


@njit(parallel=True, cache=True)
def path_T_all(bins, ncv, Y, starts, ends, z_mu, z_sd):
    """T1 of every indicator on one path, per symbol."""
    n_ind = starts.shape[0]
    n_sym = bins.shape[1]
    n_tg = Y.shape[2]
    ncut = z_mu.shape[3]
    t_sym = np.empty((n_ind, n_sym))
    for i in prange(n_ind):
        b = bins[starts[i]:ends[i]]
        c = ncv[starts[i]:ends[i]]
        n_inst = b.shape[0]
        edges = np.empty((n_inst, n_sym, n_tg, ncut, 2))
        osum = np.empty((n_inst, n_sym, n_tg, ncut, 2))
        fill_edges(b, c, Y, edges, osum)
        ts, _a = reduce_edges(edges, osum, z_mu[starts[i]:ends[i]], z_sd[starts[i]:ends[i]])
        for s in range(n_sym):
            t_sym[i, s] = ts[s]
    return t_sym


# =============================================================================
# 4. PHASE 2: PAIRS A + B
# =============================================================================
@njit(cache=True)
def _seg_bounds(c, side, nbin):
    """Segment (cut c, side) as a half-open range of the prefix-sum axis: side 0 x < c, side 1 x > c."""
    if side == 0:
        return 0, 2 * c + 1
    return 2 * c + 2, nbin


@njit(cache=True)
def _rect(P, a0, a1, b0, b1, g):
    """Sum over A bins [a0, a1) and B bins [b0, b1) of target g, from 2D prefix sums."""
    return P[a1, b1, g] - P[a0, b1, g] - P[a1, b0, g] + P[a0, b0, g]


@njit(cache=True)
def _combo(g, cA, sA, cB, sB, ncut):
    """Index of one combination inside the block of an (A, B) instance pair."""
    return (((g * ncut + cA) * 2 + sA) * ncut + cB) * 2 + sB


@njit(cache=True)
def _pair_hist(bA, bB, iA, iB, s, Y, k, P_n, P_s, P_q, mean, sd, n_all):
    """2D prefix sums (A bin, B bin, target) of one symbol, B shifted by k (circular), plus the mean, std
    and count of every target. P_*: (nbin + 1, nbin + 1, n_tg), index 0 is the empty prefix."""
    n = bA.shape[2]
    n_tg = Y.shape[2]
    nb = P_n.shape[0]
    nbin = nb - 1
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
        na = P_n[nbin, nbin, g]
        n_all[g] = na
        mean[g] = 0.0
        sd[g] = 0.0
        if na > 1.0:
            m = P_s[nbin, nbin, g] / na
            var = P_q[nbin, nbin, g] / na - m * m
            mean[g] = m
            if var > 0.0:
                sd[g] = np.sqrt(var)


@njit(cache=True)
def pair_T(bA, bB, ncvA, ncvB, Y, k, z1_mu, z1_sd, z2_mu, z2_sd, sym_idx, ncut):
    """T2 per symbol (only the symbols in sym_idx; -inf elsewhere) with B shifted by k, and its winner (flat).
    z = min of the z against both pilots (A shifted, B shifted): the pair must beat both.
    Every symbol visits the combinations in the same order as a symbol-inner loop would, so the winner is the
    same (first max)."""
    nA, n_sym, _n = bA.shape
    nB = bB.shape[0]
    n_tg = Y.shape[2]
    nbin = 2 * ncut + 1
    n_blk = n_tg * ncut * 2 * ncut * 2
    P_n = np.empty((nbin + 1, nbin + 1, n_tg))
    P_s = np.empty((nbin + 1, nbin + 1, n_tg))
    P_q = np.empty((nbin + 1, nbin + 1, n_tg))
    mean = np.empty(n_tg)
    sd = np.empty(n_tg)
    n_all = np.empty(n_tg)

    t_sym = np.full(n_sym, -np.inf)
    arg_sym = np.full(n_sym, -1, dtype=np.int64)

    for iA in range(nA):
        for iB in range(nB):
            off = (iA * nB + iB) * n_blk
            for si in range(sym_idx.shape[0]):
                s = sym_idx[si]
                _pair_hist(bA, bB, iA, iB, s, Y, k, P_n, P_s, P_q, mean, sd, n_all)
                for g in range(n_tg):
                    if sd[g] <= 0.0:
                        continue
                    for cA in range(ncvA[iA]):
                        for sA in range(2):
                            a0, a1 = _seg_bounds(cA, sA, nbin)
                            for cB in range(ncvB[iB]):
                                for sB in range(2):
                                    b0, b1 = _seg_bounds(cB, sB, nbin)
                                    flat = off + _combo(g, cA, sA, cB, sB, ncut)
                                    ss = _rect(P_s, a0, a1, b0, b1, g)
                                    d1 = z1_sd[flat, s]
                                    d2 = z2_sd[flat, s]
                                    if not (ss > 0.0 and d1 > 0.0 and d2 > 0.0):
                                        continue
                                    nn = _rect(P_n, a0, a1, b0, b1, g)
                                    qq = _rect(P_q, a0, a1, b0, b1, g)
                                    v = _seg_edge(nn, ss, qq, mean[g], sd[g], n_all[g])
                                    if v == v:
                                        z = min((v - z1_mu[flat, s]) / d1, (v - z2_mu[flat, s]) / d2)
                                        if z > t_sym[s]:
                                            t_sym[s] = z
                                            arg_sym[s] = flat
    return t_sym, arg_sym


@njit(parallel=True, cache=True)
def pair_null_distribution(bA, bB, ncvA, ncvB, Y, shifts, z1_mu, z1_sd, z2_mu, z2_sd, sym_idx, ncut):
    """T2 per shift and symbol (only sym_idx computed)."""
    n_sym = bA.shape[1]
    n_k = shifts.shape[0]
    t_sym = np.empty((n_k, n_sym))
    for q in prange(n_k):
        ts, _a = pair_T(bA, bB, ncvA, ncvB, Y, shifts[q], z1_mu, z1_sd, z2_mu, z2_sd, sym_idx, ncut)
        for s in range(n_sym):
            t_sym[q, s] = ts[s]
    return t_sym


@njit(parallel=True, cache=True)
def pair_edge_moments(bA, bB, ncvA, ncvB, Y, shifts, min_n, ncut):
    """Pilot: mean and std of the edge of every combination over the B shifts. (flat, symbol), NaN if not usable."""
    nA, n_sym, _n = bA.shape
    nB = bB.shape[0]
    n_tg = Y.shape[2]
    nbin = 2 * ncut + 1
    n_blk = n_tg * ncut * 2 * ncut * 2                     # combinations of one (A, B) instance pair
    z_mu = np.full((nA * nB * n_blk, n_sym), np.nan)
    z_sd = np.full((nA * nB * n_blk, n_sym), np.nan)
    for u in prange(n_sym * nA * nB):
        s = u // (nA * nB)
        iA = (u // nB) % nA
        iB = u % nB
        off = (iA * nB + iB) * n_blk
        P_n = np.empty((nbin + 1, nbin + 1, n_tg))
        P_s = np.empty((nbin + 1, nbin + 1, n_tg))
        P_q = np.empty((nbin + 1, nbin + 1, n_tg))
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
                        a0, a1 = _seg_bounds(cA, sA, nbin)
                        for cB in range(ncvB[iB]):
                            for sB in range(2):
                                b0, b1 = _seg_bounds(cB, sB, nbin)
                                nn = _rect(P_n, a0, a1, b0, b1, g)
                                ss = _rect(P_s, a0, a1, b0, b1, g)
                                qq = _rect(P_q, a0, a1, b0, b1, g)
                                v = _seg_edge(nn, ss, qq, mean[g], sd[g], n_all[g])
                                if v == v:
                                    j = _combo(g, cA, sA, cB, sB, ncut)
                                    c_n[j] += 1.0
                                    c_s[j] += v
                                    c_q[j] += v * v
        for j in range(n_blk):
            m, d = _moments(c_n[j], c_s[j], c_q[j], min_n)
            z_mu[off + j, s] = m
            z_sd[off + j, s] = d
    return z_mu, z_sd


@njit(parallel=True, cache=True)
def pair_valid_mask(bA, bB, ncvA, ncvB, Y, ncut):
    """Combinations whose intersection leaves MIN_SEG candles out of both the A and the B segment (flat, symbol)."""
    nA, n_sym, _n = bA.shape
    nB = bB.shape[0]
    n_tg = Y.shape[2]
    nbin = 2 * ncut + 1
    n_blk = n_tg * ncut * 2 * ncut * 2                     # combinations of one (A, B) instance pair
    ok = np.zeros((nA * nB * n_blk, n_sym), dtype=np.bool_)
    for u in prange(n_sym * nA * nB):
        s = u // (nA * nB)
        iA = (u // nB) % nA
        iB = u % nB
        off = (iA * nB + iB) * n_blk
        P_n = np.empty((nbin + 1, nbin + 1, n_tg))
        P_s = np.empty((nbin + 1, nbin + 1, n_tg))
        P_q = np.empty((nbin + 1, nbin + 1, n_tg))
        mean = np.empty(n_tg)
        sd = np.empty(n_tg)
        n_all = np.empty(n_tg)
        _pair_hist(bA, bB, iA, iB, s, Y, 0, P_n, P_s, P_q, mean, sd, n_all)
        for g in range(n_tg):
            if sd[g] <= 0.0:
                continue                                   # no combination competes there anyway
            for cA in range(ncvA[iA]):
                for sA in range(2):
                    a0, a1 = _seg_bounds(cA, sA, nbin)
                    n_seg_a = P_n[a1, nbin, g] - P_n[a0, nbin, g]  # A segment, any B bin
                    for cB in range(ncvB[iB]):
                        for sB in range(2):
                            b0, b1 = _seg_bounds(cB, sB, nbin)
                            n_seg_b = P_n[nbin, b1, g] - P_n[nbin, b0, g]
                            nn = _rect(P_n, a0, a1, b0, b1, g)
                            ok[off + _combo(g, cA, sA, cB, sB, ncut), s] = \
                                (n_seg_a - nn >= MIN_SEG) and (n_seg_b - nn >= MIN_SEG)
    return ok