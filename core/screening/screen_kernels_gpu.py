# core/screening/screen_kernels_gpu.py
import numpy as np
import cupy as cp

from .screen_kernels_cpu import MIN_SEG

THREADS          = 256    # threads per block (alone)
PAIR_TILE        = 4096   # candles per shared-memory tile of the pair kernels (2 bytes each)
W_SLOTS          = 8      # segments of the mask side per thread of the pair kernels (as in the CUDA source)
PAIR_MAX_THREADS = 256    # threads per block of the pair kernels, at most (as in the CUDA source)
PAIR_BUF_BYTES   = 1 << 30  # GPU memory per batch of shifts (pass 1 of the batch, and the pilot's v)
FX_Y_BITS        = 40     # pair walk sums in 64-bit fixed point: y * 2^40 (resolution ~1e-12 of a %)
FX_Y2_BITS       = 32     # and y^2 * 2^32

# =============================================================================
# 1. NPY WALK OF EVERY SEGMENT (ALONE) OR SEGMENT PAIR (A AND B)
# =============================================================================

_NPY_SOURCE = r"""
// ---------------------------------------------------------------------------------------------------------------
// Alone. Group = (instance, symbol, target), gid = (i * n_sym + s) * n_tg + g. npy_group_stats: count, sum and sum
// of squares of the group's valid candles and their bin histogram. npy_seg_walk: one thread per segment of a group,
// tid = gid * 2 ncut + (c * 2 + side), which is also its index in the output layout (n_inst, n_sym, n_tg, ncut, 2).
// ---------------------------------------------------------------------------------------------------------------
extern "C" __global__
void npy_group_stats(
    const signed char*   __restrict__ bins,
    const double*        __restrict__ YT,
    const unsigned char* __restrict__ VT,
    long long*           __restrict__ grp_n,
    double*              __restrict__ grp_s,
    double*              __restrict__ grp_q,
    int*                 __restrict__ grp_hist,
    const int n_inst, const int n_sym, const int n, const int n_tg, const int nbin)
{
    const long long total = (long long)n_inst * n_sym * n_tg;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    const int g = (int)(gid % n_tg);
    const int s = (int)((gid / n_tg) % n_sym);
    const int i = (int)(gid / ((long long)n_tg * n_sym));

    const signed char*   rb = bins + ((size_t)i * n_sym + s) * n;
    const double*        ry = YT + ((size_t)s * n_tg + g) * n;
    const unsigned char* rv = VT + ((size_t)s * n_tg + g) * n;
    int* h = grp_hist + (size_t)gid * nbin;
    for (int x = 0; x < nbin; ++x) h[x] = 0;

    long long cnt = 0;
    double sm = 0.0, sq = 0.0;
    for (int t = 0; t < n; ++t) {
        const int b = rb[t];
        if (b < 0 || !rv[t]) continue;
        const double y = ry[t];
        cnt += 1;
        sm += y;
        sq += y * y;
        h[b] += 1;
    }
    grp_n[gid] = cnt;
    grp_s[gid] = sm;
    grp_q[gid] = sq;
}


extern "C" __global__
void npy_seg_walk(
    const signed char*   __restrict__ bins,
    const long long*     __restrict__ ncv,
    const double*        __restrict__ YT,
    const int*           __restrict__ ET,
    const unsigned char* __restrict__ VT,
    const long long*     __restrict__ grp_n,
    const double*        __restrict__ grp_s,
    const double*        __restrict__ grp_q,
    const int*           __restrict__ grp_hist,
    double*              __restrict__ out_v,
    double*              __restrict__ out_osum,
    const int n_inst, const int n_sym, const int n, const int n_tg, const int ncut, const int min_seg)
{
    const long long total = (long long)n_inst * n_sym * n_tg * 2 * ncut;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const long long gid = tid / (2 * ncut);           // its group, as in npy_group_stats
    const int seg  = (int)(tid % (2 * ncut));
    const int c    = seg >> 1, side = seg & 1;
    const int g = (int)(gid % n_tg);
    const int s = (int)((gid / n_tg) % n_sym);
    const int i = (int)(gid / ((long long)n_tg * n_sym));

    double v = __longlong_as_double(0x7ff8000000000000ULL), osum = 0.0;
    if (c < ncv[i]) {
        // Segment bins (inclusive): side 0: [0, 2c]; side 1: [2c + 2, nbin - 1]
        const int nbin = 2 * ncut + 1;
        const int b0 = side ? 2 * c + 2 : 0, b1 = side ? nbin - 1 : 2 * c;
        const int* h = grp_hist + (size_t)gid * nbin;
        long long n_seg = 0;
        for (int b = b0; b <= b1; ++b)
            n_seg += h[b];

        const long long n_all = grp_n[gid];
        if (n_all > 1) {
            const double na      = (double)n_all;
            const double mean    = grp_s[gid] / na;
            const double var_all = grp_q[gid] / na - mean * mean;
            const double sd_all  = var_all > 0.0 ? sqrt(var_all) : 0.0;
            if (sd_all > 0.0 && n_seg >= min_seg && n_seg <= n_all - min_seg) {
                const signed char*   rb = bins + ((size_t)i * n_sym + s) * n;
                const double*        ry = YT + ((size_t)s * n_tg + g) * n;
                const int*           re = ET + ((size_t)s * n_tg + g) * n;
                const unsigned char* rv = VT + ((size_t)s * n_tg + g) * n;
                long long k = 0;
                double ts = 0.0, tq = 0.0;
                int t = 0;
                while (t < n) {
                    const int b = rb[t];
                    if (b >= 0 && rv[t] && (side ? b >= 2 * c + 2 : b <= 2 * c)) {
                        const double y = ry[t];
                        k += 1;
                        ts += y;
                        tq += y * y;
                        const int e = re[t];              // first candle after the exit
                        t = e > t ? e : t + 1;
                    } else {
                        ++t;
                    }
                }
                if (k >= min_seg) {
                    const double kk   = (double)k;
                    const double m    = ts / kk;
                    const double var  = tq / kk - m * m;
                    const double sd_k = var > 0.0 ? sqrt(var) : 0.0;
                    v = (m - mean) / (fmax(sd_k, sd_all) / sqrt(kk));
                    osum = ts - kk * mean;                // excess over the unconditional mean
                }
            }
        }
    }
    out_v[tid] = v;
    out_osum[tid] = osum;
}



// ---------------------------------------------------------------------------------------------------------------
// Pairs (A AND B, B shifted by `shift`). Group = (instance A, instance B, symbol, target) at one shift.
//
// 1. npy_pair_stats: the reference's pass 1, as it was. One block per (shift, group) with bdim_v = 4 nc_a nc_b
//    threads: count, sum and sum of squares of the group's valid candles (strided, then its tree reduction) and
//    their (bin A, bin B) histogram; then thread = segment pair (the reference's layout, tid = (ca * 2 + sa) * 2 nc_b
//    + cb * 2 + sb) says whether that segment is walked. Out: mean and sd of the group, and one byte per segment.
//
// 2. Bitset walk. Within a group every segment pair sees the same candles, the same y[t] and the same exit E[t];
//    the segments that take candle t are the free ones that contain it, and all of them are busy until the same
//    E[t]. So a thread owns one segment of the row side R and a chunk of up to W_SLOTS segments of the mask side M,
//    as bits: per candle one shared-memory read, a range test and an AND, and the accumulation only when a segment
//    of its chunk trades. Sums in 64-bit fixed point (y * 2^FX_Y_BITS and y^2 * 2^FX_Y2_BITS, precomputed and
//    rounded once per candle): integer, exact and independent of the order, and no double precision in the loop
//    (1/64 of the rate on GeForce GPUs). Only the statistic of each segment, at the end, is in double.
//    Block = (shift, instance A, instance B, symbol, chunk of G targets); thread tid = (row * n_w + w) * G + gl:
//    row = its segment of R (cr = row >> 1, sr = row & 1), w = its chunk of M (slot j is segment w * W_SLOTS + j of
//    M, cm = that >> 1, sm = that & 1), g = gc * G + gl its target. mask_a = 0: R = A, M = B; mask_a = 1: R = B,
//    M = A (the side with more segments goes into the bits). The shifts of a batch run in parallel.
// ---------------------------------------------------------------------------------------------------------------

#define W_SLOTS 8
#define PAIR_MAX_THREADS 256
#define NO_REL 0x7fffffff
#define FX_NAN ((long long)0x8000000000000000ULL)    // fixed-point y of a NaN target: not a valid candle

// Pass 1 of every (shift, group): block = bdim_v threads, blockIdx.x = q * groups + gid, gid = ((ia * n_b + ib) *
// n_si + si) * n_tg + g (the reference's group order). Shared memory: red_s, red_q (bdim_v doubles), red_c (bdim_v
// ints), hist ((2 nc_a + 1) x (2 nc_b + 1) ints).
extern "C" __global__
void npy_pair_stats(
    const signed char*   __restrict__ bins_a,
    const signed char*   __restrict__ bins_b,
    const long long*     __restrict__ ncv_a,
    const long long*     __restrict__ ncv_b,
    const double*        __restrict__ YT,
    const unsigned char* __restrict__ VT,
    const long long*     __restrict__ sym_idx,
    const long long*     __restrict__ shifts,
    double*              __restrict__ st_mean,
    double*              __restrict__ st_sd,
    unsigned char*       __restrict__ st_walk,
    const int n_a, const int n_b, const int n_sym, const int n, const int n_tg,
    const int nc_a, const int nc_b, const int n_si, const int min_seg)
{
    extern __shared__ double smem[];                  // doubles first: 8-byte aligned
    const int bdim = blockDim.x, tid = threadIdx.x;
    const int ha = 2 * nc_a + 1, hb = 2 * nc_b + 1;   // histogram sides: bins 0..2 nc
    double* red_s = smem;
    double* red_q = red_s + bdim;
    int*    red_c = (int*)(red_q + bdim);
    int*    hist  = red_c + bdim;

    const long long groups = (long long)n_a * n_b * n_si * n_tg;
    const int q = (int)(blockIdx.x / groups);
    const long long gid = blockIdx.x % groups;
    long long rest = gid;
    const int g  = (int)(rest % n_tg);  rest /= n_tg;
    const int si = (int)(rest % n_si);  rest /= n_si;
    const int ib = (int)(rest % n_b);
    const int ia = (int)(rest / n_b);
    const int s  = (int)sym_idx[si];
    const int shift = (int)shifts[q];
    // thread = segment pair: tid = (ca * 2 + sa) * 2 nc_b + (cb * 2 + sb)
    const int sa_i = tid / (2 * nc_b), sb_i = tid % (2 * nc_b);
    const int ca = sa_i >> 1, sa = sa_i & 1;
    const int cb = sb_i >> 1, sb = sb_i & 1;

    const signed char*   ra = bins_a + ((size_t)ia * n_sym + s) * n;
    const signed char*   rb = bins_b + ((size_t)ib * n_sym + s) * n;
    const double*        ry = YT + ((size_t)s * n_tg + g) * n;
    const unsigned char* rv = VT + ((size_t)s * n_tg + g) * n;

    // --- pass 1: statistics of the group's valid candles
    for (int x = tid; x < ha * hb; x += bdim) hist[x] = 0;
    __syncthreads();
    int cnt = 0;
    double sm = 0.0, sq = 0.0;
    for (int t = tid; t < n; t += bdim) {
        int j = t + shift;                            // (t + shift) mod n: B shifted like _pair_hist
        if (j >= n) j -= n;
        const int a = ra[t], b = rb[j];
        if (a < 0 || b < 0 || !rv[t]) continue;
        const double y = ry[t];
        cnt += 1;
        sm += y;
        sq += y * y;
        atomicAdd(&hist[a * hb + b], 1);
    }
    red_c[tid] = cnt;
    red_s[tid] = sm;
    red_q[tid] = sq;
    __syncthreads();
    for (int half = 1; half < bdim; half <<= 1) {
        if ((tid % (2 * half)) == 0 && tid + half < bdim) {
            red_c[tid] += red_c[tid + half];
            red_s[tid] += red_s[tid + half];
            red_q[tid] += red_q[tid + half];
        }
        __syncthreads();
    }
    const long long n_all = red_c[0];

    // --- this thread's segment: bins [a0, a1] x [b0, b1], its candles from the histogram
    const bool mine = ca < ncv_a[ia] && cb < ncv_b[ib];
    const int a0 = sa ? 2 * ca + 2 : 0, a1 = sa ? 2 * nc_a : 2 * ca;
    const int b0 = sb ? 2 * cb + 2 : 0, b1 = sb ? 2 * nc_b : 2 * cb;
    long long n_seg = 0;
    if (mine)
        for (int a = a0; a <= a1; ++a)
            for (int b = b0; b <= b1; ++b)
                n_seg += hist[a * hb + b];
    double mean = 0.0, sd_all = 0.0;
    if (n_all > 1) {
        const double na = (double)n_all;
        mean = red_s[0] / na;
        const double var_all = red_q[0] / na - mean * mean;
        sd_all = var_all > 0.0 ? sqrt(var_all) : 0.0;
    }
    const bool walk = mine && n_all > 1 && sd_all > 0.0 && n_seg >= min_seg && n_seg <= n_all - min_seg;

    const size_t b_out = (size_t)q * groups + gid;
    if (tid == 0) {
        st_mean[b_out] = mean;
        st_sd[b_out] = sd_all;
    }
    st_walk[b_out * bdim + tid] = walk ? 1 : 0;
}


// Shared memory of the bitset kernels: red_u (blockDim.x uint64), bm ((2 nc_M + 1) x n_w uint32: bits of the
// chunk's segments that contain a bin of M), code (tile shorts).
#define PAIR_SMEM                                                                                                  \
    extern __shared__ double smem[];                                                                               \
    const int nc_R = mask_a ? nc_b : nc_a, nc_M = mask_a ? nc_a : nc_b;                                            \
    unsigned long long* red_u = (unsigned long long*)smem;                                                         \
    unsigned int*       bm    = (unsigned int*)(red_u + blockDim.x);                                               \
    short*              code  = (short*)(bm + (2 * nc_M + 1) * n_w);

// Block gb = ((ia * n_b + ib) * n_si + si) * n_gc + gc of shift q, and this thread's segment of R, chunk of M and
// target. Its slots [0, n_slots) are the segments of M that exist (cut < ncv of M); W_SLOTS is even, so slot j is
// segment 2 cm0 + j of M and its output index (layout (n_a * n_b * n_tg * ncut * 2 * ncut * 2, n_sym)) is
// o0 + j * o_step. From pass 1 (index st of its group): mean, sd and the bits of its slots that are walked.
#define PAIR_DECODE(gb)                                                                                            \
    long long rest = (gb);                                                                                         \
    const int gc = (int)(rest % n_gc);  rest /= n_gc;                                                              \
    const int si = (int)(rest % n_si);  rest /= n_si;                                                              \
    const int ib = (int)(rest % n_b);                                                                              \
    const int ia = (int)(rest / n_b);                                                                              \
    const int s  = (int)sym_idx[si];                                                                               \
    const int gl = threadIdx.x % G, rw = threadIdx.x / G;                                                          \
    const int w  = rw % n_w, row = rw / n_w;                                                                       \
    const int g  = gc * G + gl;                                                                                    \
    const int cr = row >> 1, sr = row & 1;                                                                         \
    const long long ncv_R = mask_a ? ncv_b[ib] : ncv_a[ia];                                                        \
    const long long ncv_M = mask_a ? ncv_a[ia] : ncv_b[ib];                                                        \
    const bool row_mine = g < n_tg && cr < ncv_R;                                                                  \
    const int n_slots = row_mine ? max(0, min(W_SLOTS, 2 * (int)ncv_M - w * W_SLOTS)) : 0;                         \
    const int cm0 = (w * W_SLOTS) >> 1;                                                                            \
    const long long o0 = (long long)((((((((size_t)ia * n_b + ib) * n_tg + g) * ncut + (mask_a ? cm0 : cr)) * 2     \
                         + (mask_a ? 0 : sr)) * ncut + (mask_a ? cr : cm0)) * 2 + (mask_a ? sr : 0)) * n_sym + s);    \
    const long long o_step = mask_a ? 2LL * ncut * n_sym : (long long)n_sym;                                        \
    const signed char* ra = bins_a + ((size_t)ia * n_sym + s) * n;                                                 \
    const signed char* rb = bins_b + ((size_t)ib * n_sym + s) * n;                                                 \
    const long long*   qrow  = QY + (size_t)s * n * n_tg + (g < n_tg ? g : 0);                                     \
    const long long*   q2row = QY2 + (size_t)s * n * n_tg + (g < n_tg ? g : 0);                                    \
    const int*         erow = ER + (size_t)s * n * n_tg + (g < n_tg ? g : 0);                                      \
    double mean = 0.0, sd_all = 0.0;                                                                               \
    unsigned int walk_bits = 0u;                                                                                   \
    if (n_slots > 0) {                                                                                             \
        const size_t st = (size_t)q * ((size_t)n_a * n_b * n_si * n_tg)                                            \
                          + (((size_t)ia * n_b + ib) * n_si + si) * n_tg + g;                                      \
        mean = st_mean[st];                                                                                        \
        sd_all = st_sd[st];                                                                                        \
        const unsigned char* fl = st_walk + st * (size_t)bdim_v;                                                   \
        for (int j = 0; j < n_slots; ++j) {                                                                        \
            const int sj = w * W_SLOTS + j;                                                                        \
            if (fl[mask_a ? sj * 2 * nc_b + row : row * 2 * nc_b + sj]) walk_bits |= 1u << j;                     \
        }                                                                                                          \
    }


// bm[m * n_w + w] bit j: slot j of chunk w (segment w * W_SLOTS + j of M, if it exists) contains bin m of M.
__device__ __forceinline__ void pair_mask_table(unsigned int* bm, const int nc_M, const int n_w)
{
    for (int x = threadIdx.x; x < (2 * nc_M + 1) * n_w; x += blockDim.x) {
        const int m = x / n_w, w = x % n_w;
        unsigned int bits = 0u;
        for (int j = 0; j < W_SLOTS; ++j) {
            const int si = w * W_SLOTS + j;
            if (si >= 2 * nc_M) break;
            const int c = si >> 1;
            if ((si & 1) ? m >= 2 * c + 2 : m <= 2 * c) bits |= 1u << j;
        }
        bm[x] = bits;
    }
}


// The walk of this thread's slots (bits walk_bits) over the block's candles, B shifted by `shift`: k and the fixed-
// point sums of y and y^2 of every slot. Candles are packed in shared memory tiles as bin A * 256 + bin B (-1: a bin
// missing); a candle whose target is NaN (FX_NAN) is not taken (the reference's rv). Every thread of the block must
// call it (it synchronizes).
__device__ __forceinline__ void pair_walk_bits(
    const signed char* __restrict__ ra, const signed char* __restrict__ rb, const long long* __restrict__ qrow,
    const long long* __restrict__ q2row, const int* __restrict__ erow, const int n, const int n_tg, const int shift,
    const int tile, const int mask_a, const int r0, const int r1, const int w, const int n_w,
    const unsigned int walk_bits, const unsigned int* bm, short* code, int* kk, long long* ts, long long* tq)
{
    const int T = blockDim.x, tid = threadIdx.x;
    const int rs = mask_a ? 0 : 8, ms = mask_a ? 8 : 0;   // shifts of the row and mask bins in a code
    int busy[W_SLOTS];
    #pragma unroll
    for (int j = 0; j < W_SLOTS; ++j) {
        kk[j] = 0;
        ts[j] = 0;
        tq[j] = 0;
        busy[j] = 0;
    }
    unsigned int fr = walk_bits;                      // free slots
    int min_rel = NO_REL;                             // first candle at which a busy slot is free again
    for (int c0 = 0; c0 < n; c0 += tile) {
        const int len = (n - c0 < tile) ? n - c0 : tile;
        __syncthreads();                              // the previous tile (and the mask table) are ready / read
        for (int x = tid; x < len; x += T) {
            const int tt = c0 + x;
            int j = tt + shift;
            if (j >= n) j -= n;
            const int a = ra[tt], b = rb[j];
            code[x] = (a >= 0 && b >= 0) ? (short)((a << 8) | b) : (short)-1;
        }
        __syncthreads();
        if (!walk_bits) continue;
        for (int x = 0; x < len; ++x) {
            const int t = c0 + x;
            if (t >= min_rel) {                       // some slot is free again: rebuild the free bits
                unsigned int f = walk_bits;
                int mr = NO_REL;
                #pragma unroll
                for (int j = 0; j < W_SLOTS; ++j)
                    if (busy[j] > t) {
                        f &= ~(1u << j);
                        mr = busy[j] < mr ? busy[j] : mr;
                    }
                fr = f;
                min_rel = mr;
            }
            const int cd = code[x];
            if (cd < 0) continue;
            const int r = (cd >> rs) & 255;
            if (r < r0 || r > r1) continue;
            const unsigned int take = fr & bm[((cd >> ms) & 255) * n_w + w];
            if (!take) continue;
            const size_t it = (size_t)t * n_tg;
            const long long qy = qrow[it], qy2 = q2row[it];   // the three loads at once
            const int e = erow[it];                   // first candle after the exit
            if (qy == FX_NAN) continue;               // target NaN: not a valid candle
            const int nx = e > t ? e : t + 1;
            #pragma unroll
            for (int j = 0; j < W_SLOTS; ++j)
                if (take & (1u << j)) {
                    kk[j] += 1;
                    ts[j] += qy;
                    tq[j] += qy2;
                    busy[j] = nx;
                }
            fr &= ~take;
            min_rel = nx < min_rel ? nx : min_rel;
        }
    }
}


// NPY statistic of a walked segment (as the reference, from its fixed-point sums): v NaN and osum 0 unless k >=
// min_seg. inv_y = 2^-FX_Y_BITS and inv_y2 = 2^-FX_Y2_BITS: exact scalings.
__device__ __forceinline__ void seg_stat(const int k, const long long ts_q, const long long tq_q, const double inv_y,
                                         const double inv_y2, const double mean, const double sd_all,
                                         const int min_seg, double& v, double& osum)
{
    v = __longlong_as_double(0x7ff8000000000000ULL);
    osum = 0.0;
    if (k >= min_seg) {
        const double ts   = (double)ts_q * inv_y;
        const double tq   = (double)tq_q * inv_y2;
        const double kk   = (double)k;
        const double m    = ts / kk;
        const double var  = tq / kk - m * m;
        const double sd_k = var > 0.0 ? sqrt(var) : 0.0;
        v = (m - mean) / (fmax(sd_k, sd_all) / sqrt(kk));
        osum = ts - kk * mean;                        // excess over the unconditional mean
    }
}


#define PAIR_PARAMS                                                                                                \
    const signed char*   __restrict__ bins_a,                                                                      \
    const signed char*   __restrict__ bins_b,                                                                      \
    const long long*     __restrict__ ncv_a,                                                                       \
    const long long*     __restrict__ ncv_b,                                                                       \
    const long long*     __restrict__ QY,                                                                          \
    const long long*     __restrict__ QY2,                                                                         \
    const int*           __restrict__ ER,                                                                          \
    const long long*     __restrict__ sym_idx,                                                                     \
    const long long*     __restrict__ shifts,                                                                      \
    const double*        __restrict__ st_mean,                                                                     \
    const double*        __restrict__ st_sd,                                                                       \
    const unsigned char* __restrict__ st_walk,                                                                     \
    const int n_a, const int n_b, const int n_sym, const int n, const int n_tg, const int ncut,                    \
    const int nc_a, const int nc_b, const int n_si, const int mask_a, const int n_w, const int G,                  \
    const int n_gc, const int bdim_v, const int min_seg, const int tile,                                       \
    const double inv_y, const double inv_y2

// Block of a bitset kernel: shift q, group chunk gb; the mask table, then the walk of this thread's slots.
#define PAIR_WALK                                                                                                  \
    PAIR_SMEM                                                                                                      \
    const long long groups = (long long)n_a * n_b * n_si * n_gc;                                                   \
    const int q = (int)(blockIdx.x / groups);                                                                      \
    PAIR_DECODE(blockIdx.x % groups)                                                                               \
    pair_mask_table(bm, nc_M, n_w);                                                                                \
    int kk[W_SLOTS];                                                                                               \
    long long ts[W_SLOTS], tq[W_SLOTS];                                                                            \
    pair_walk_bits(ra, rb, qrow, q2row, erow, n, n_tg, (int)shifts[q], tile, mask_a, sr ? 2 * cr + 2 : 0,         \
                   sr ? 2 * nc_R : 2 * cr, w, n_w, walk_bits, bm, code, kk, ts, tq);


// (v, osum) of every segment pair at each shift of B of the launch: out_*[q * q_stride + o]. full = 0: only v (the
// pilot's buffer; out_osum is not touched).
extern "C" __global__ __launch_bounds__(PAIR_MAX_THREADS)
void npy_pair_walk_bits(
    PAIR_PARAMS, const long long q_stride, const int full,
    double* __restrict__ out_v, double* __restrict__ out_osum)
{
    PAIR_WALK
    const long long oq = (long long)q * q_stride + o0;
    #pragma unroll
    for (int j = 0; j < W_SLOTS; ++j) {
        if (j >= n_slots) break;
        const long long o = oq + j * o_step;
        double v = __longlong_as_double(0x7ff8000000000000ULL), osum = 0.0;
        const bool walked = (walk_bits >> j) & 1u;
        if (walked) seg_stat(kk[j], ts[j], tq[j], inv_y, inv_y2, mean, sd_all, min_seg, v, osum);
        out_v[o] = v;
        if (full) out_osum[o] = osum;
    }
}


// Pilot sums of a batch of n_q shifts, in shift order: for every element i and q = 0 .. n_q - 1, with ok = v not
// NaN: n += ok ? 1 : 0; s += ok ? v : 0; q += ok ? v * v : 0 (the reference's m_n, m_s and m_q).
extern "C" __global__
void npy_pilot_acc(const double* __restrict__ buf, const int n_q, const long long n_elem,
                   double* __restrict__ acc_n, double* __restrict__ acc_s, double* __restrict__ acc_q)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) return;
    double c_n = acc_n[i], c_s = acc_s[i], c_q = acc_q[i];
    for (int q = 0; q < n_q; ++q) {
        const double v = buf[(size_t)q * n_elem + i];
        const bool ok = !(v != v);
        c_n += ok ? 1.0 : 0.0;
        c_s += ok ? v : 0.0;
        c_q += ok ? v * v : 0.0;
    }
    acc_n[i] = c_n;
    acc_s[i] = c_s;
    acc_q[i] = c_q;
}


// _moments of the CPU, in place: acc_s becomes the mean and acc_q the std (NaN where n < min_n or var <= 0).
extern "C" __global__
void npy_pilot_moments(const double* __restrict__ acc_n, double* __restrict__ acc_s, double* __restrict__ acc_q,
                       const long long n_elem, const double min_n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) return;
    const double c_n = acc_n[i];
    double mu = __longlong_as_double(0x7ff8000000000000ULL), sd = mu;
    if (!(c_n < min_n || c_n < 2.0)) {
        const double m   = acc_s[i] / c_n;
        const double var = (acc_q[i] - c_n * m * m) / (c_n - 1.0);
        if (var > 0.0) {
            mu = m;
            sd = sqrt(var);
        }
    }
    acc_s[i] = mu;
    acc_q[i] = sd;
}


// Order-preserving bits of a double (NaN above +inf, as the maximum of NumPy/CuPy propagates it).
__device__ __forceinline__ unsigned long long ord_bits(const double x)
{
    if (x != x) return 0xFFF8000000000000ULL;
    const unsigned long long b = (unsigned long long)__double_as_longlong(x);
    return (b & 0x8000000000000000ULL) ? ~b : (b | 0x8000000000000000ULL);
}


// Null: for each shift of B of the launch, the largest z of the segment pairs of every symbol of sym_idx, z as
// _pair_z_npy (-inf where it does not compete). Block maximum, then one atomicMax on the order-preserving bits of
// out_t[q0 + q, s] (initialized to the bits of -inf). The maximum does not depend on the order.
extern "C" __global__ __launch_bounds__(PAIR_MAX_THREADS)
void npy_pair_null_bits(
    PAIR_PARAMS, const int q0,
    const double* __restrict__ mu1, const double* __restrict__ sd1, const double* __restrict__ mu2,
    const double* __restrict__ sd2, unsigned long long* __restrict__ out_t)
{
    PAIR_WALK
    unsigned long long best = ord_bits(__longlong_as_double(0xFFF0000000000000ULL));   // -inf
    #pragma unroll
    for (int j = 0; j < W_SLOTS; ++j) {
        if (!((walk_bits >> j) & 1u)) continue;       // not walked: v NaN, z -inf
        double v, osum;
        seg_stat(kk[j], ts[j], tq[j], inv_y, inv_y2, mean, sd_all, min_seg, v, osum);
        const long long o = o0 + j * o_step;
        const double d1 = sd1[o], d2 = sd2[o];
        if (!(isfinite(v) && osum > 0.0 && d1 > 0.0 && d2 > 0.0)) continue;
        const double z1 = (v - mu1[o]) / d1;
        const double z2 = (v - mu2[o]) / d2;
        const double z  = (z1 != z1) ? z1 : ((z2 != z2) ? z2 : (z2 < z1 ? z2 : z1));   // NumPy minimum
        const unsigned long long u = ord_bits(z);
        best = u > best ? u : best;
    }

    // --- block maximum on the order-preserving bits
    const int T = blockDim.x, tid = threadIdx.x;
    __syncthreads();
    red_u[tid] = best;
    __syncthreads();
    for (int half = 1; half < T; half <<= 1) {
        if ((tid % (2 * half)) == 0 && tid + half < T && red_u[tid + half] > red_u[tid])
            red_u[tid] = red_u[tid + half];
        __syncthreads();
    }
    if (tid == 0) atomicMax(&out_t[(size_t)(q0 + q) * n_sym + s], red_u[0]);
}
"""


# --fmad=false: no fused multiply-add, so the sums round like the CPU reference
_NPY_MODULE = cp.RawModule(code=_NPY_SOURCE, options=("--fmad=false",))

_ORD_NEG_INF = 0x000FFFFFFFFFFFFF        # ord_bits(-inf)
_SIGN        = np.uint64(1 << 63)
_SMEM_MAX    = 48 * 1024                 # dynamic shared memory of a block without opting in to more


class NpyPrep(tuple):
    """(YT, ET, VT): targets, exits and 'target is not NaN' as (n_sym, n_tg, n). rows: (QY, QY2, E) as (n_sym, n,
    n_tg), the pair kernels' layout for the trades (one candle of every target is contiguous), built on first use:
    y and y^2 in 64-bit fixed point, rint(y * 2^FX_Y_BITS) and rint(y * y * 2^FX_Y2_BITS) (NaN target: INT64_MIN in
    QY). The sums of any segment of n candles fit in int64 (checked)."""

    def __new__(cls, YT, ET, VT):
        obj = super().__new__(cls, (YT, ET, VT))
        obj._rows = None
        return obj

    @property
    def rows(self):
        if self._rows is None:
            YT, ET, _VT = self
            Y = cp.ascontiguousarray(YT.transpose(0, 2, 1))
            n = Y.shape[1]
            nan = cp.isnan(Y)
            y_max = float(cp.max(cp.where(nan, 0.0, cp.abs(Y)))) if Y.size else 0.0
            lim = float(2 ** 62)                       # margin under 2^63: no overflow for any segment
            if n * y_max * 2.0 ** FX_Y_BITS >= lim or n * y_max * y_max * 2.0 ** FX_Y2_BITS >= lim:
                raise ValueError(f"Targets too large for the fixed-point sums: max |y| = {y_max}, {n} candles")
            QY = cp.where(nan, np.iinfo(np.int64).min, cp.rint(Y * 2.0 ** FX_Y_BITS)).astype(cp.int64)
            QY2 = cp.where(nan, 0, cp.rint((Y * Y) * 2.0 ** FX_Y2_BITS)).astype(cp.int64)
            self._rows = (QY, QY2, cp.ascontiguousarray(ET.transpose(0, 2, 1)))
        return self._rows


def npy_prepare(Y, E):
    """(YT, ET, VT): targets, exits and 'target is not NaN' as (n_sym, n_tg, n) on the GPU, the kernels' layout.

    Once per data set (real or synthetic path): npy_edges reuses it for every call.
    """
    Y = cp.asarray(Y, dtype=cp.float64)
    E = cp.asarray(E)
    if Y.ndim != 3 or E.shape != Y.shape:
        raise ValueError(f"Y and E must be (n_sym, n, n_tg) with the same shape: {Y.shape}, {E.shape}")
    if Y.shape[1] >= np.iinfo(np.int32).max:
        raise ValueError(f"Too many candles for int32 indices: {Y.shape[1]}")
    YT = cp.ascontiguousarray(Y.transpose(0, 2, 1))
    ET = cp.ascontiguousarray(E.transpose(0, 2, 1)).astype(cp.int32)
    VT = (~cp.isnan(YT)).astype(cp.uint8)
    return NpyPrep(YT, ET, VT)


def _dev(x, dtype):
    return cp.ascontiguousarray(cp.asarray(x, dtype=dtype))


def _host_idx(x):
    """A host int64 copy of an index vector (NumPy or CuPy)."""
    return np.asarray(cp.asnumpy(x), dtype=np.int64).reshape(-1)


def npy_shifts(shifts, n):
    """The shifts as a device int64 array, checked to be in [0, n): the pair kernels take them as they are."""
    sh = _host_idx(shifts)
    if sh.size and (int(sh.min()) < 0 or int(sh.max()) >= n):
        raise ValueError(f"shifts must be in [0, {n})")
    return _dev(sh, cp.int64)


def npy_pair_setup(bins_a, ncv_a, bins_b, ncv_b, prep, ncut):
    """Inputs and block layout of the pair kernels for A and B (B is the one shifted), output layout (A, B).

    It does the host-device copies and the checks, so the launches that take it (npy_pair_moments) do not
    synchronize and host work can overlap them. Device bins are not copied again.
    """
    if not isinstance(prep, NpyPrep):
        prep = NpyPrep(*prep)
    YT, _ET, _VT = prep
    n_sym, n_tg, n = YT.shape
    bins_a = _dev(bins_a, cp.int8)
    bins_b = _dev(bins_b, cp.int8)
    ncv_a_h, ncv_b_h = _host_idx(ncv_a), _host_idx(ncv_b)
    n_a, n_b = bins_a.shape[0], bins_b.shape[0]
    if bins_a.shape != (n_a, n_sym, n):
        raise ValueError(f"bins_a must be (n_a, {n_sym}, {n}): {bins_a.shape}")
    if ncv_a_h.shape != (n_a,) or int(ncv_a_h.max()) > ncut:
        raise ValueError(f"ncv_a must be ({n_a},) with values <= ncut={ncut}")
    if bins_b.shape != (n_b, n_sym, n):
        raise ValueError(f"bins_b must be (n_b, {n_sym}, {n}): {bins_b.shape}")
    if ncv_b_h.shape != (n_b,) or int(ncv_b_h.max()) > ncut:
        raise ValueError(f"ncv_b must be ({n_b},) with values <= ncut={ncut}")
    nc_a, nc_b = int(ncv_a_h.max()), int(ncv_b_h.max())
    bdim_v = 4 * nc_a * nc_b                           # threads of pass 1 (the reference's block)
    if bdim_v > 1024:
        raise ValueError(f"Too many thresholds for the pair kernel: {nc_a} x {nc_b} (at most 1024 threads)")
    mask_a = int(nc_a > nc_b)                          # the side with more segments goes into the bits
    nc_r, nc_m = (nc_b, nc_a) if mask_a else (nc_a, nc_b)
    n_w = -(-2 * nc_m // W_SLOTS)                      # chunks of W_SLOTS segments of the mask side
    per_g = 2 * nc_r * n_w                             # threads per target
    g_blk = max(1, min(n_tg, PAIR_MAX_THREADS // per_g))
    threads = per_g * g_blk
    smem = 8 * threads + 4 * (2 * nc_m + 1) * n_w + 2 * PAIR_TILE
    smem_st = 8 * 2 * bdim_v + 4 * bdim_v + 4 * (2 * nc_a + 1) * (2 * nc_b + 1)
    if threads > PAIR_MAX_THREADS or max(smem, smem_st) > _SMEM_MAX:
        raise ValueError(f"Pair block too large: {threads} threads, {max(smem, smem_st)} bytes of shared memory")
    return {"bins_a": bins_a, "bins_b": bins_b, "ncv_a": _dev(ncv_a_h, cp.int64), "ncv_b": _dev(ncv_b_h, cp.int64),
            "prep": prep, "n_a": n_a, "n_b": n_b, "n_sym": n_sym, "n": n, "n_tg": n_tg, "ncut": int(ncut),
            "nc_a": nc_a, "nc_b": nc_b, "bdim_v": bdim_v, "mask_a": mask_a, "n_w": n_w, "g_blk": g_blk,
            "n_gc": -(-n_tg // g_blk), "threads": threads, "smem": smem, "smem_st": smem_st,
            "shape": (n_a * n_b * n_tg * ncut * 2 * ncut * 2, n_sym)}


def _launch(name, n_blocks, threads, smem, args):
    if n_blocks > np.iinfo(np.int32).max:
        raise ValueError(f"Too many blocks for one launch: {n_blocks}")
    _NPY_MODULE.get_function(name)((n_blocks,), (threads,), args, shared_mem=smem)


def _pair_stats(ctx, sym_d, n_si, shifts, min_seg):
    """Pass 1 of every (shift, group), the reference's code and order: (mean, sd, walked byte per segment)."""
    n_q = int(shifts.shape[0])
    groups = ctx["n_a"] * ctx["n_b"] * n_si * ctx["n_tg"]
    st_mean = cp.empty(n_q * groups, dtype=cp.float64)
    st_sd = cp.empty(n_q * groups, dtype=cp.float64)
    st_walk = cp.empty(n_q * groups * ctx["bdim_v"], dtype=cp.uint8)
    YT, _ET, VT = ctx["prep"]
    _launch("npy_pair_stats", n_q * groups, ctx["bdim_v"], ctx["smem_st"],
            (ctx["bins_a"], ctx["bins_b"], ctx["ncv_a"], ctx["ncv_b"], YT, VT, sym_d, shifts, st_mean, st_sd, st_walk,
             *(np.int32(ctx[k]) for k in ("n_a", "n_b", "n_sym", "n", "n_tg", "nc_a", "nc_b")), np.int32(n_si),
             np.int32(min_seg)))
    return st_mean, st_sd, st_walk


def _pair_walk(name, ctx, sym_d, n_si, shifts, min_seg, extra):
    """Pass 1 and a bitset kernel over the shifts (a batch: all of them in one launch each)."""
    n_k = int(shifts.shape[0])
    stats = _pair_stats(ctx, sym_d, n_si, shifts, min_seg)
    QY, QY2, ER = ctx["prep"].rows
    _launch(name, n_k * ctx["n_a"] * ctx["n_b"] * n_si * ctx["n_gc"], ctx["threads"], ctx["smem"],
            (ctx["bins_a"], ctx["bins_b"], ctx["ncv_a"], ctx["ncv_b"], QY, QY2, ER, sym_d, shifts, *stats,
             *(np.int32(ctx[k]) for k in ("n_a", "n_b", "n_sym", "n", "n_tg", "ncut", "nc_a", "nc_b")),
             np.int32(n_si), *(np.int32(ctx[k]) for k in ("mask_a", "n_w", "g_blk", "n_gc", "bdim_v")),
             np.int32(min_seg), np.int32(PAIR_TILE), np.float64(2.0 ** -FX_Y_BITS),
             np.float64(2.0 ** -FX_Y2_BITS), *extra))


def _batch(n_k, per_shift_bytes):
    """Shifts per batch so that a batch uses at most PAIR_BUF_BYTES (at least 1)."""
    return max(1, min(n_k, PAIR_BUF_BYTES // max(per_shift_bytes, 1)))


def npy_pair_moments(ctx, shifts, min_n, min_seg=MIN_SEG):
    """(mu, sd) of the NPY statistic of every segment pair of ctx over the shifts of B (from npy_shifts), with the
    sums of a serial run in shift order and then _moments: NaN where it is valid in fewer than min_n shifts.

    The shifts are walked in parallel, a batch at a time (v of the batch in a GPU buffer), and summed in their
    order. Layout as npy_edges, all symbols. CuPy outputs; it does not synchronize.
    """
    shape = ctx["shape"]
    n_elem = shape[0] * shape[1]
    n_k = int(shifts.shape[0])
    if not (n_k and n_elem):
        return cp.full(shape, np.nan, dtype=cp.float64), cp.full(shape, np.nan, dtype=cp.float64)
    groups = ctx["n_a"] * ctx["n_b"] * ctx["n_sym"] * ctx["n_tg"]
    n_batch = _batch(n_k, 8 * n_elem + groups * (16 + ctx["bdim_v"]))
    acc_n = cp.zeros(shape, dtype=cp.float64)
    mu = cp.zeros(shape, dtype=cp.float64)             # the sum of v, then the mean
    sd = cp.zeros(shape, dtype=cp.float64)             # the sum of v^2, then the std
    buf = cp.empty((n_batch, n_elem), dtype=cp.float64)
    sym_d = cp.arange(ctx["n_sym"], dtype=cp.int64)
    unused_o = cp.empty(1, dtype=cp.float64)
    n_thr = (n_elem + THREADS - 1) // THREADS
    for q0 in range(0, n_k, n_batch):
        nb = min(n_batch, n_k - q0)
        buf[:nb].fill(np.nan)                          # segments that do not exist stay NaN
        _pair_walk("npy_pair_walk_bits", ctx, sym_d, ctx["n_sym"], shifts[q0:q0 + nb], min_seg,
                   (np.int64(n_elem), np.int32(0), buf, unused_o))
        _launch("npy_pilot_acc", n_thr, THREADS, 0, (buf, np.int32(nb), np.int64(n_elem), acc_n, mu, sd))
    _launch("npy_pilot_moments", n_thr, THREADS, 0, (acc_n, mu, sd, np.int64(n_elem), np.float64(min_n)))
    return mu, sd


def _ord_to_float(u):
    """Inverse of ord_bits (host)."""
    u = np.asarray(u, dtype=np.uint64)
    return np.where((u & _SIGN) != 0, u ^ _SIGN, ~u).view(np.float64)


def npy_pair_null(ctx, shifts, mu1, sd1, mu2, sd2, sym_idx, min_seg=MIN_SEG):
    """T of every shift of B (rows; from npy_shifts) and symbol of sym_idx (columns; the others NaN): the largest
    z of all the segment pairs of ctx, z as _pair_z_npy (-inf if none competes). mu1, sd1, mu2, sd2: layout as
    npy_edges. The shifts run in parallel (in batches if pass 1 would not fit), host output.
    """
    n_sym, shape = ctx["n_sym"], ctx["shape"]
    mus = []
    for x in (mu1, sd1, mu2, sd2):
        if tuple(x.shape) != shape:
            raise ValueError(f"mu and sd must have the layout {shape}: {x.shape}")
        mus.append(_dev(x, cp.float64))
    sym_h = _host_idx(sym_idx)
    if sym_h.size and (int(sym_h.min()) < 0 or int(sym_h.max()) >= n_sym):
        raise ValueError(f"sym_idx must be in [0, {n_sym})")
    n_k, n_si = int(shifts.shape[0]), int(sym_h.size)
    res = np.full((n_k, n_sym), np.nan)
    if not (n_k and n_si):
        return res
    sym_d = _dev(sym_h, cp.int64)
    out = cp.full((n_k, n_sym), _ORD_NEG_INF, dtype=cp.uint64)
    n_batch = _batch(n_k, ctx["n_a"] * ctx["n_b"] * n_si * ctx["n_tg"] * (16 + ctx["bdim_v"]))
    for q0 in range(0, n_k, n_batch):
        nb = min(n_batch, n_k - q0)
        _pair_walk("npy_pair_null_bits", ctx, sym_d, n_si, shifts[q0:q0 + nb], min_seg,
                   (np.int32(q0), *mus, out))
    res[:, sym_h] = _ord_to_float(cp.asnumpy(out))[:, sym_h]
    return res


def npy_edges(bins_a, ncv_a, prep, ncut, bins_b=None, ncv_b=None, min_seg=MIN_SEG):
    """(v, osum) of the NPY walk of every segment of A (alone) or of A AND B (pairs), every symbol.

    prep: npy_prepare(Y, E). Alone: shape (n_a, n_sym, n_tg, ncut, 2), as fill_edges. Pair: shape
    (n_a * n_b * n_tg * ncut * 2 * ncut * 2, n_sym), as pair_edge_moments. Inputs NumPy or CuPy, outputs CuPy.
    """
    YT, ET, VT = prep
    n_sym, n_tg, n = YT.shape

    if bins_b is not None:                             # pairs: bitset walk, tiles in shared memory
        ctx = npy_pair_setup(bins_a, ncv_a, bins_b, ncv_b, prep, ncut)
        v = cp.full(ctx["shape"], np.nan, dtype=cp.float64)
        osum = cp.zeros(ctx["shape"], dtype=cp.float64)
        _pair_walk("npy_pair_walk_bits", ctx, cp.arange(n_sym, dtype=cp.int64), n_sym,
                   cp.zeros(1, dtype=cp.int64), min_seg,
                   (np.int64(ctx["shape"][0] * ctx["shape"][1]), np.int32(1), v, osum))
        return v, osum

    bins_a = _dev(bins_a, cp.int8)
    ncv_a = _dev(ncv_a, cp.int64)
    n_a = bins_a.shape[0]
    if bins_a.shape != (n_a, n_sym, n):
        raise ValueError(f"bins_a must be (n_a, {n_sym}, {n}): {bins_a.shape}")
    if ncv_a.shape != (n_a,) or int(ncv_a.max()) > ncut:
        raise ValueError(f"ncv_a must be ({n_a},) with values <= ncut={ncut}")
    shape = (n_a, n_sym, n_tg, ncut, 2)
    v = cp.full(shape, np.nan, dtype=cp.float64)
    osum = cp.zeros(shape, dtype=cp.float64)
    groups = n_a * n_sym * n_tg
    if not groups:
        return v, osum

    nbin = 2 * ncut + 1
    grp_n = cp.empty(groups, dtype=cp.int64)
    grp_s = cp.empty(groups, dtype=cp.float64)
    grp_q = cp.empty(groups, dtype=cp.float64)
    grp_hist = cp.empty(groups * nbin, dtype=cp.int32)
    ints = (np.int32(n_a), np.int32(n_sym), np.int32(n), np.int32(n_tg))
    _launch("npy_group_stats", (groups + THREADS - 1) // THREADS, THREADS, 0,
            (bins_a, YT, VT, grp_n, grp_s, grp_q, grp_hist, *ints, np.int32(nbin)))
    walks = groups * 2 * ncut
    _launch("npy_seg_walk", (walks + THREADS - 1) // THREADS, THREADS, 0,
            (bins_a, ncv_a, YT, ET, VT, grp_n, grp_s, grp_q, grp_hist, v, osum, *ints, np.int32(ncut),
             np.int32(min_seg)))
    return v, osum