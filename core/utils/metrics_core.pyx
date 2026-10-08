#core/utils/metrics_core.pyx
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

# Numerical kernels behind utils.batch_metrics, the only module that calls them.
# Each kernel performs the same float64 operations, in the same order, as the numpy expression it replaces, so the
# results are bit-identical:
#   np.add.reduce / np.mean   -> 0.0 + pairwise sum (8 accumulators in blocks of up to 128, halves split on multiples of 8)
#   elementwise expressions   -> one rounding per operation, intermediate values stored (no fused multiply-add)
#   np.cumsum                 -> sequential running sum
#   np.maximum.accumulate     -> running maximum, NaN propagates
#   ndarray.min               -> minimum, NaN propagates (a zero minimum is returned as +0.0, see equity_drawdown)

import numpy as np
from libc.math cimport sqrt, NAN
from libc.stdlib cimport malloc, free

cdef enum:
    _PW_BLOCKSIZE = 128


# ============================================================
# Pairwise sum (np.add.reduce on a contiguous float64 array)
# ============================================================
cdef double _pairwise_sum(const double* a, Py_ssize_t n) noexcept nogil:
    cdef double r[8]
    cdef double res
    cdef Py_ssize_t i, j, n2
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    if n <= _PW_BLOCKSIZE:
        for j in range(8):
            r[j] = a[j]
        i = 8
        while i < n - (n % 8):
            for j in range(8):
                r[j] += a[i + j]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            res += a[i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, n2) + _pairwise_sum(a + n2, n - n2)


cdef inline double _mean(const double* x, Py_ssize_t n) noexcept nogil:
    # np.add.reduce starts from the identity 0.0
    return (0.0 + _pairwise_sum(x, n)) / n


# ============================================================
# Mean and population std
#   mean = np.add.reduce(x) / n
#   std  = np.sqrt(np.add.reduce((x - mean) ** 2) / n)
# ============================================================
cdef void _mean_std(const double* x, Py_ssize_t n, double* work, double* mean, double* std) noexcept nogil:
    cdef Py_ssize_t i
    cdef double m, dev
    m = _mean(x, n)
    for i in range(n):
        dev     = x[i] - m
        work[i] = dev * dev
    mean[0] = m
    std[0]  = sqrt(_mean(work, n))


cdef double* _workspace(Py_ssize_t n) except NULL:
    cdef double* work = <double*>malloc(max(n, 1) * sizeof(double))
    if work == NULL:
        raise MemoryError()
    return work


def mean_std(const double[::1] x):
    # (mean, std) of x; NaN for an empty array, as numpy
    cdef Py_ssize_t n = x.shape[0]
    cdef double mean = NAN
    cdef double std  = NAN
    cdef double* work
    if n == 0:
        return mean, std
    work = _workspace(n)
    try:
        with nogil:
            _mean_std(&x[0], n, work, &mean, &std)
    finally:
        free(work)
    return mean, std


def mean_std_rows(const double[:, ::1] values, const long[::1] start, const long[::1] length):
    # (means, stds) of values[r, start[r]:start[r] + length[r]] per row; NaN where length <= 0
    cdef Py_ssize_t n_rows = values.shape[0]
    cdef Py_ssize_t r
    cdef double* work

    if start.shape[0] != n_rows or length.shape[0] != n_rows:
        raise ValueError("start and length need one entry per row of values")
    for r in range(n_rows):
        if length[r] > 0 and (start[r] < 0 or start[r] + length[r] > values.shape[1]):
            raise ValueError(f"row {r}: span [{start[r]}, {start[r] + length[r]}) outside values")

    means = np.full(n_rows, np.nan, dtype=np.float64)
    stds  = np.full(n_rows, np.nan, dtype=np.float64)
    cdef double[::1] means_mv = means
    cdef double[::1] stds_mv  = stds
    work = _workspace(values.shape[1])
    try:
        with nogil:
            for r in range(n_rows):
                if length[r] > 0:
                    _mean_std(&values[r, start[r]], length[r], work, &means_mv[r], &stds_mv[r])
    finally:
        free(work)
    return means, stds


# ============================================================
# Central moments
#   d  = x - x.mean()
#   d2 = d * d
#   m2 = np.mean(d2), m3 = np.mean(d2 * d), m4 = np.mean(d2 * d2)
# ============================================================
def central_moments(const double[::1] x):
    # (mean, m2, m3, m4) of x; NaN for an empty array, as numpy
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double mean = NAN
    cdef double m2   = NAN
    cdef double m3   = NAN
    cdef double m4   = NAN
    cdef double* dev
    cdef double* dev_sq
    cdef double* prod
    if n == 0:
        return mean, m2, m3, m4
    dev = _workspace(3 * n)
    dev_sq = dev + n
    prod   = dev + 2 * n
    try:
        with nogil:
            mean = _mean(&x[0], n)
            for i in range(n):
                dev[i]    = x[i] - mean
                dev_sq[i] = dev[i] * dev[i]
            m2 = _mean(dev_sq, n)
            for i in range(n):
                prod[i] = dev_sq[i] * dev[i]
            m3 = _mean(prod, n)
            for i in range(n):
                prod[i] = dev_sq[i] * dev_sq[i]
            m4 = _mean(prod, n)
    finally:
        free(dev)
    return mean, m2, m3, m4


# ============================================================
# Equity curve and maximum drawdown
#   eq     = capital + np.cumsum(x)
#   cm     = np.maximum.accumulate(eq)
#   max_dd = ((eq - cm) / cm * 100).min()
# ============================================================
def equity_drawdown(const double[::1] x, double capital):
    # (eq, max_dd); x must not be empty
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double cum, peak, dd, max_dd
    cdef bint nan_seen
    if n == 0:
        raise ValueError("equity_drawdown needs at least one daily value")

    eq = np.empty(n, dtype=np.float64)
    cdef double[::1] eq_mv = eq
    with nogil:
        cum      = x[0]
        eq_mv[0] = capital + cum
        for i in range(1, n):
            cum      = cum + x[i]
            eq_mv[i] = capital + cum

        peak     = eq_mv[0]
        nan_seen = False
        max_dd   = 0.0
        for i in range(n):
            if peak != peak or eq_mv[i] != eq_mv[i]:
                peak = NAN
            elif eq_mv[i] > peak:
                peak = eq_mv[i]
            dd = (eq_mv[i] - peak) / peak
            dd = dd * 100.0
            if dd != dd:
                nan_seen = True
            elif i == 0 or dd < max_dd:
                max_dd = dd
        if nan_seen:
            max_dd = NAN
        elif max_dd == 0.0:
            # -0.0 only appears when the running peak is negative (equity below zero); numpy's SIMD minimum returns
            # either zero sign there depending on the array layout, this kernel always returns +0.0
            max_dd = 0.0
    return eq, max_dd