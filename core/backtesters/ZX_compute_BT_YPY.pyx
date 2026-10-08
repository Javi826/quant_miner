#core/backtesters/ZX_compute_BT_YPY.pyx (PYRAMID)
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import logging
import warnings
import numpy as np
import pandas as pd
cimport numpy as np
from libc.math cimport HUGE_VAL
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t

cdef extern from *:
    int __builtin_ctzll(unsigned long long x) nogil

logging.basicConfig(level=logging.INFO)
from setup.config_backtest import INITIAL_BALANCE, COMISION, LEVERAGE
warnings.filterwarnings("ignore")

cdef long _NS_PER_DAY = 86400000000000

EXIT_REASON_NAMES   = np.array(['SELL_AFTER', 'TP', 'SL', 'END_OF_DATA'], dtype=object)
POSITION_TYPE_NAMES = np.array(['LONG', 'SHORT'], dtype=object)

# ============================================================
# C-level binary search helpers  (replaces np.searchsorted)
# ============================================================

cdef int _searchsorted_right(long* arr, long val, int n) noexcept nogil:
    # first index i where arr[i] > val (right side)
    cdef int lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo
# ============================================================
# Manual min-heap over (time, counter) -> slot  (replaces heapq of dicts)
# ============================================================
cdef inline void _heap_push(
    long* heap_time, long* heap_counter, int* heap_slot,
    int* heap_size, long time, long counter, int slot
) noexcept nogil:
    cdef int i = heap_size[0]
    cdef int parent
    cdef long tmp_l
    cdef int tmp_i

    heap_time[i]    = time
    heap_counter[i] = counter
    heap_slot[i]    = slot

    while i > 0:
        parent = (i - 1) >> 1
        if (heap_time[parent] > heap_time[i]) or \
           (heap_time[parent] == heap_time[i] and heap_counter[parent] > heap_counter[i]):
            tmp_l = heap_time[parent];    heap_time[parent] = heap_time[i];       heap_time[i] = tmp_l
            tmp_l = heap_counter[parent]; heap_counter[parent] = heap_counter[i]; heap_counter[i] = tmp_l
            tmp_i = heap_slot[parent];    heap_slot[parent] = heap_slot[i];       heap_slot[i] = tmp_i
            i = parent
        else:
            break

    heap_size[0] = heap_size[0] + 1


cdef inline int _heap_pop(
    long* heap_time, long* heap_counter, int* heap_slot, int* heap_size
) noexcept nogil:
    cdef int slot, i, left, right, smallest, n
    cdef long tmp_l
    cdef int tmp_i

    slot = heap_slot[0]
    n = heap_size[0] - 1

    heap_time[0]    = heap_time[n]
    heap_counter[0] = heap_counter[n]
    heap_slot[0]    = heap_slot[n]
    heap_size[0]    = n

    i = 0
    while True:
        left     = 2 * i + 1
        right    = 2 * i + 2
        smallest = i

        if left < n and (
            heap_time[left] < heap_time[smallest] or
            (heap_time[left] == heap_time[smallest] and heap_counter[left] < heap_counter[smallest])
        ):
            smallest = left

        if right < n and (
            heap_time[right] < heap_time[smallest] or
            (heap_time[right] == heap_time[smallest] and heap_counter[right] < heap_counter[smallest])
        ):
            smallest = right

        if smallest == i:
            break

        tmp_l = heap_time[i];    heap_time[i] = heap_time[smallest];       heap_time[smallest] = tmp_l
        tmp_l = heap_counter[i]; heap_counter[i] = heap_counter[smallest]; heap_counter[smallest] = tmp_l
        tmp_i = heap_slot[i];    heap_slot[i] = heap_slot[smallest];       heap_slot[smallest] = tmp_i
        i = smallest

    return slot


cdef inline void _detect_intrabar_exit_cy(
    float* high_row,
    float* low_row,
    long*   high_time_row,
    long*   low_time_row,
    int buy_idx, int sell_idx,
    double tp_price, double sl_price,
    bint is_short, bint scan_to_last,
    bint* out_intra, int* out_idx, int* out_reason, double* out_price
) noexcept nogil:
    cdef int bi, tp_first, sl_first, scan_last
    cdef long tp_t, sl_t

    tp_first = -1
    sl_first = -1

    if scan_to_last:
        scan_last = sell_idx
    else:
        scan_last = sell_idx - 1

    if is_short:
        for bi in range(buy_idx, scan_last + 1):
            if tp_first < 0 and low_row[bi] <= tp_price:
                tp_first = bi
            if sl_first < 0 and high_row[bi] >= sl_price:
                sl_first = bi
            if tp_first >= 0 or sl_first >= 0:
                break
    else:
        for bi in range(buy_idx, scan_last + 1):
            if tp_first < 0 and high_row[bi] >= tp_price:
                tp_first = bi
            if sl_first < 0 and low_row[bi] <= sl_price:
                sl_first = bi
            if tp_first >= 0 or sl_first >= 0:
                break

    if tp_first < 0 and sl_first < 0:
        out_intra[0]  = False
        out_idx[0]    = -1
        out_reason[0] = 0
        out_price[0]  = 0.0
        return

    if tp_first >= 0 and sl_first >= 0:
        if tp_first == sl_first:
            if is_short:
                tp_t = low_time_row[tp_first]
                sl_t = high_time_row[sl_first]
            else:
                tp_t = high_time_row[tp_first]
                sl_t = low_time_row[sl_first]
            if tp_t <= sl_t:
                out_intra[0] = True; out_idx[0] = tp_first; out_reason[0] = 1; out_price[0] = tp_price
            else:
                out_intra[0] = True; out_idx[0] = sl_first; out_reason[0] = 2; out_price[0] = sl_price
        elif sl_first < tp_first:
            out_intra[0] = True; out_idx[0] = sl_first; out_reason[0] = 2; out_price[0] = sl_price
        else:
            out_intra[0] = True; out_idx[0] = tp_first; out_reason[0] = 1; out_price[0] = tp_price
    elif sl_first >= 0:
        out_intra[0] = True; out_idx[0] = sl_first; out_reason[0] = 2; out_price[0] = sl_price
    else:
        out_intra[0] = True; out_idx[0] = tp_first; out_reason[0] = 1; out_price[0] = tp_price


# ============================================================
# Static market arrays  (built once per data window)
# ============================================================
def prepare_static_arrays(ohlcv_arrays):
    symbols = list(ohlcv_arrays.keys())
    n_syms  = len(symbols)
    sym_ids = {s: i for i, s in enumerate(sorted(symbols))}

    per_sym_ts       = {}
    all_ts_int_lists = []
    for sym in symbols:
        ts = ohlcv_arrays[sym]['ts']
        if ts.dtype.kind != 'M':
            ts = ts.astype('datetime64[ns]')
        ts_int = ts.view('int64').copy()  # own memory: bundle outlives the shm-backed input
        per_sym_ts[sym] = (ts_int, len(ts))
        all_ts_int_lists.append(ts_int)

    max_len      = max((per_sym_ts[s][1] for s in symbols), default=0)
    open_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float32)
    close_2d     = np.full((n_syms, max_len), np.nan, dtype=np.float32)
    high_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float32)
    low_2d       = np.full((n_syms, max_len), np.nan, dtype=np.float32)
    high_time_2d = np.full((n_syms, max_len), 0,      dtype=np.int64)
    low_time_2d  = np.full((n_syms, max_len), 0,      dtype=np.int64)
    ts_int_2d    = np.full((n_syms, max_len), 0,      dtype=np.int64)
    sym_len      = np.zeros(n_syms, dtype=np.int64)

    for sym in symbols:
        sid  = sym_ids[sym]
        data = ohlcv_arrays[sym]
        ts_int, n = per_sym_ts[sym]
        sym_len[sid]          = n
        open_2d[sid, :n]      = data['open'].astype(np.float32)
        close_2d[sid, :n]     = data['close'].astype(np.float32)
        high_2d[sid, :n]      = data['high'].astype(np.float32)
        low_2d[sid, :n]       = data['low'].astype(np.float32)
        high_time_2d[sid, :n] = data['high_time'].astype(np.int64)
        low_time_2d[sid, :n]  = data['low_time'].astype(np.int64)
        ts_int_2d[sid, :n]    = ts_int

    all_timestamps_int = (np.unique(np.concatenate(all_ts_int_lists)) if all_ts_int_lists
                          else np.empty(0, dtype=np.int64))

    # position of every bar timestamp in the global timeline (padding bars map to 0)
    tick_pos_2d = np.zeros((n_syms, max_len), dtype=np.int64)
    for sid in range(n_syms):
        n = int(sym_len[sid])
        tick_pos_2d[sid, :n] = np.searchsorted(all_timestamps_int, ts_int_2d[sid, :n])

    return {
        "symbols":            symbols,
        "sym_ids":            sym_ids,
        "symbols_by_sid":     tuple(sorted(symbols)),
        "all_timestamps_int": all_timestamps_int,
        "open_2d":            open_2d,
        "close_2d":           close_2d,
        "high_2d":            high_2d,
        "low_2d":             low_2d,
        "high_time_2d":       high_time_2d,
        "low_time_2d":        low_time_2d,
        "ts_int_2d":          ts_int_2d,
        "sym_len":            sym_len,
        "tick_pos_2d":        tick_pos_2d,
        "idx_workspace":      np.empty(int(sym_len.sum()), dtype=np.int32),
    }


def market_arrays(static_bundle):
    return (
        static_bundle["open_2d"], static_bundle["close_2d"],
        static_bundle["high_2d"], static_bundle["low_2d"],
        static_bundle["high_time_2d"], static_bundle["low_time_2d"],
        static_bundle["ts_int_2d"], static_bundle["sym_len"],
    )


# ============================================================
# Signal events  (single pass over the signals of one rule)
# ============================================================
cdef uint64_t _ABS_MASK_X2 = 0x7FFFFFFF7FFFFFFFULL   # clears both sign bits: +0.0 and -0.0 read as zero

cdef enum:
    _OUTER_BLOCK = 64   # floats tested at once before looking inside
    _INNER_BLOCK = 8    # floats per sub-block inside a non-zero outer block


cdef inline bint _any_nonzero(const float* sig, int n_floats) noexcept nogil:
    cdef uint64_t acc = 0, word
    cdef int w
    for w in range(n_floats // 2):
        memcpy(&word, sig + 2 * w, sizeof(uint64_t))
        acc |= word
    return (acc & _ABS_MASK_X2) != 0


cdef inline long _collect_range(const float* sig, long start, long stop, int* out, long k) noexcept nogil:
    cdef long j
    for j in range(start, stop):
        out[k] = <int>j
        k += sig[j] != 0.0
    return k


cdef inline long _collect_nonzero(const float* sig, long n, int* out, long k) noexcept nogil:
    # Appends the indices of non-zero values (NaN included, +/-0.0 excluded) to out[k:], returns the new k.
    cdef long i = 0, j, block_end
    while i + _OUTER_BLOCK <= n:
        if _any_nonzero(sig + i, _OUTER_BLOCK):
            j         = i
            block_end = i + _OUTER_BLOCK
            while j < block_end:
                if _any_nonzero(sig + j, _INNER_BLOCK):
                    k = _collect_range(sig, j, j + _INNER_BLOCK, out, k)
                j += _INNER_BLOCK
        i += _OUTER_BLOCK
    return _collect_range(sig, i, n, out, k)


def build_rule_events(
    tuple signals,
    long[:, ::1] ts_int_2d,
    long[::1] sym_len,
    long[:, ::1] tick_pos_2d,
    long[::1] all_timestamps_int,
    int[::1] idx_workspace,
):
    # Single pass over the signals: events sorted by (timestamp, sid) plus the compressed timeline.
    # idx_workspace must hold at least sum(sym_len) entries; it is reused across calls.
    cdef Py_ssize_t n_syms  = sym_len.shape[0]
    cdef Py_ssize_t n_ticks = all_timestamps_int.shape[0]
    if len(signals) != n_syms:
        raise ValueError(f"expected {n_syms} signal arrays, got {len(signals)}")

    cdef const float** sig_ptr = <const float**>malloc((n_syms + 1) * sizeof(float*))
    cdef long* heads           = <long*>malloc((n_syms + 1) * sizeof(long))
    cdef long* ends            = <long*>malloc((n_syms + 1) * sizeof(long))
    if sig_ptr == NULL or heads == NULL or ends == NULL:
        free(sig_ptr); free(heads); free(ends)
        raise MemoryError()

    cdef const float[::1] sig_mv
    cdef Py_ssize_t sid, i, k, n, best_sid, n_events, n_timeline
    cdef long best_t, t, bar_idx, tick, last_tick
    cdef long[:, ::1] ev_mv
    cdef signed char[::1] short_mv
    cdef long[::1] timeline_mv

    try:
        n = 0
        for sid in range(n_syms):
            sig_mv = signals[sid]
            if sig_mv.shape[0] < sym_len[sid]:
                raise ValueError(f"signal array {sid} shorter than its symbol length")
            sig_ptr[sid] = &sig_mv[0] if sym_len[sid] > 0 else NULL
            n += sym_len[sid]
        if idx_workspace.shape[0] < n:
            raise ValueError(f"idx_workspace too small: {idx_workspace.shape[0]} < {n}")

        with nogil:
            k = 0
            for sid in range(n_syms):
                heads[sid] = k
                if sym_len[sid] > 0:
                    k = _collect_nonzero(sig_ptr[sid], sym_len[sid], &idx_workspace[k], 0) + k
                ends[sid] = k
            n_events = k

        signal_events = np.empty((n_events, 3), dtype=np.int64)
        ev_short      = np.empty(n_events, dtype=np.int8)
        if n_events == 0:
            return signal_events, ev_short, np.asarray(all_timestamps_int)

        timeline_arr = np.empty(2 * n_events + 1, dtype=np.int64)
        ev_mv        = signal_events
        short_mv     = ev_short
        timeline_mv  = timeline_arr

        with nogil:
            n_timeline = 0
            last_tick  = -1
            for k in range(n_events):
                best_sid = -1
                best_t   = 0
                for sid in range(n_syms):
                    if heads[sid] < ends[sid]:
                        t = ts_int_2d[sid, idx_workspace[heads[sid]]]
                        if best_sid < 0 or t < best_t:
                            best_sid = sid
                            best_t   = t
                bar_idx = idx_workspace[heads[best_sid]]
                heads[best_sid] += 1

                ev_mv[k, 0] = best_t
                ev_mv[k, 1] = best_sid
                ev_mv[k, 2] = bar_idx
                short_mv[k] = 1 if (<int>(<long>sig_ptr[best_sid][bar_idx])) < 0 else 0

                tick = tick_pos_2d[best_sid, bar_idx]
                if tick > 0 and tick - 1 > last_tick:
                    timeline_mv[n_timeline] = all_timestamps_int[tick - 1]
                    n_timeline += 1
                    last_tick = tick - 1
                if tick > last_tick:
                    timeline_mv[n_timeline] = all_timestamps_int[tick]
                    n_timeline += 1
                    last_tick = tick

            if n_ticks - 1 > last_tick:
                timeline_mv[n_timeline] = all_timestamps_int[n_ticks - 1]
                n_timeline += 1

        return signal_events, ev_short, timeline_arr[:n_timeline]
    finally:
        free(sig_ptr)
        free(heads)
        free(ends)


# ============================================================
# Rule events from the spec mask table
#
#   spec_words[s, w]: bit b of word w is bar 64 * (w - word_offsets[sid]) + b of symbol sid for spec s,
#   already shifted one bar like the signal of a one-spec rule in backtest mode.
#   A rule's signal is the AND of the rows of its specs, so its events are the set bits of that AND.
# The output (events, short flags, timeline) matches build_rule_events for the equivalent signal arrays.
# ============================================================
cdef inline long _collect_and_bits(
    const uint64_t** rows, int n_rows, Py_ssize_t w0, Py_ssize_t w1, uint64_t tail_mask, int* out, long k
) noexcept nogil:
    # Appends to out[k:] the bar index of every bit set in the AND of the rows over words [w0, w1).
    cdef Py_ssize_t w
    cdef int r
    cdef uint64_t acc
    for w in range(w0, w1):
        acc = rows[0][w]
        for r in range(1, n_rows):
            acc &= rows[r][w]
        if w == w1 - 1:
            acc &= tail_mask
        while acc:
            out[k] = <int>((w - w0) * 64 + __builtin_ctzll(acc))
            k   += 1
            acc &= acc - 1
    return k


cdef inline uint64_t _tail_mask(long n_bars) noexcept nogil:
    # valid bits of the last word of a symbol with n_bars bars
    cdef long rem = n_bars & 63
    if rem == 0:
        return <uint64_t>0xFFFFFFFFFFFFFFFFULL
    return ((<uint64_t>1) << rem) - 1


def build_rule_events_from_words(
    const uint64_t[:, ::1] spec_words,
    const long[::1] word_offsets,
    const int[::1] spec_idx,
    bint is_short,
    const long[:, ::1] ts_int_2d,
    const long[::1] sym_len,
    const long[:, ::1] tick_pos_2d,
    const long[::1] all_timestamps_int,
    int[::1] idx_workspace,
):
    # Events of one rule sorted by (timestamp, sid) plus the compressed timeline.
    # spec_idx: rows of the rule's specs in spec_words (repeating a row is harmless: AND is idempotent).
    # idx_workspace must hold at least sum(sym_len) entries; it is reused across calls.
    cdef Py_ssize_t n_syms  = sym_len.shape[0]
    cdef Py_ssize_t n_ticks = all_timestamps_int.shape[0]
    cdef Py_ssize_t n_specs = spec_words.shape[0]
    cdef int n_rows = <int>spec_idx.shape[0]
    cdef Py_ssize_t sid, i, k, n, best_sid, n_events, n_timeline
    cdef long best_t, t, bar_idx, tick, last_tick
    cdef long[:, ::1] ev_mv
    cdef long[::1] timeline_mv

    if n_rows == 0:
        raise ValueError("a rule needs at least one spec")
    if word_offsets.shape[0] != n_syms + 1:
        raise ValueError(f"word_offsets must have {n_syms + 1} entries, got {word_offsets.shape[0]}")
    if word_offsets[n_syms] != spec_words.shape[1]:
        raise ValueError(f"word_offsets end at {word_offsets[n_syms]}, spec_words has {spec_words.shape[1]} words")
    n = 0
    for sid in range(n_syms):
        if word_offsets[sid + 1] - word_offsets[sid] != (sym_len[sid] + 63) // 64:
            raise ValueError(f"symbol {sid}: {word_offsets[sid + 1] - word_offsets[sid]} words for {sym_len[sid]} bars")
        n += sym_len[sid]
    for i in range(n_rows):
        if spec_idx[i] < 0 or spec_idx[i] >= n_specs:
            raise ValueError(f"spec index {spec_idx[i]} out of range [0, {n_specs})")
    if idx_workspace.shape[0] < n:
        raise ValueError(f"idx_workspace too small: {idx_workspace.shape[0]} < {n}")

    cdef const uint64_t** rows = <const uint64_t**>malloc(n_rows * sizeof(uint64_t*))
    cdef long* heads           = <long*>malloc((n_syms + 1) * sizeof(long))
    cdef long* ends            = <long*>malloc((n_syms + 1) * sizeof(long))
    cdef int* ws
    if rows == NULL or heads == NULL or ends == NULL:
        free(rows); free(heads); free(ends)
        raise MemoryError()

    try:
        n_events = 0
        if n > 0:
            for i in range(n_rows):
                rows[i] = &spec_words[spec_idx[i], 0]
            ws = &idx_workspace[0]
            with nogil:
                k = 0
                for sid in range(n_syms):
                    heads[sid] = k
                    if sym_len[sid] > 0:
                        k = _collect_and_bits(rows, n_rows, word_offsets[sid], word_offsets[sid + 1],
                                              _tail_mask(sym_len[sid]), ws, k)
                    ends[sid] = k
                n_events = k

        signal_events = np.empty((n_events, 3), dtype=np.int64)
        ev_short      = np.full(n_events, 1 if is_short else 0, dtype=np.int8)
        if n_events == 0:
            return signal_events, ev_short, np.asarray(all_timestamps_int)

        timeline_arr = np.empty(2 * n_events + 1, dtype=np.int64)
        ev_mv        = signal_events
        timeline_mv  = timeline_arr

        with nogil:
            n_timeline = 0
            last_tick  = -1
            for k in range(n_events):
                best_sid = -1
                best_t   = 0
                for sid in range(n_syms):
                    if heads[sid] < ends[sid]:
                        t = ts_int_2d[sid, ws[heads[sid]]]
                        if best_sid < 0 or t < best_t:
                            best_sid = sid
                            best_t   = t
                bar_idx = ws[heads[best_sid]]
                heads[best_sid] += 1

                ev_mv[k, 0] = best_t
                ev_mv[k, 1] = best_sid
                ev_mv[k, 2] = bar_idx

                tick = tick_pos_2d[best_sid, bar_idx]
                if tick > 0 and tick - 1 > last_tick:
                    timeline_mv[n_timeline] = all_timestamps_int[tick - 1]
                    n_timeline += 1
                    last_tick = tick - 1
                if tick > last_tick:
                    timeline_mv[n_timeline] = all_timestamps_int[tick]
                    n_timeline += 1
                    last_tick = tick

            if n_ticks - 1 > last_tick:
                timeline_mv[n_timeline] = all_timestamps_int[n_ticks - 1]
                n_timeline += 1

        return signal_events, ev_short, timeline_arr[:n_timeline]
    finally:
        free(rows)
        free(heads)
        free(ends)


cdef inline long _floor_div(long a, long b) noexcept nogil:
    cdef long q = a / b
    if (a % b != 0) and ((a < 0) != (b < 0)):
        q -= 1
    return q


# ============================================================
# Simulation engine  (PYRAMID: every signal opens a position while cash and free slots allow)
# ============================================================
cdef class _GridEngine:
    cdef float[:, ::1] open_mv, close_mv, high_mv, low_mv
    cdef long[:, ::1]  high_time_mv, low_time_mv, ts_int_mv, ev_mv
    cdef long[::1]     sym_len_mv, ts_all_mv
    cdef signed char[::1] ev_short_mv

    cdef double initial_balance, comi_factor, order_amount, margin_req
    cdef int    n_ticks, n_events, n_syms
    cdef bint   full_log
    cdef public int max_trades

    # trade log (tl_day is a scratch buffer for the daily aggregation)
    cdef long[::1]   tl_buy_time, tl_sell_time, tl_day
    cdef double[::1] tl_profit
    cdef int[::1]    tl_exit_reason
    # full trade log, only filled when full_log is set
    cdef long[::1]   tl_sym_id
    cdef double[::1] tl_buy_price, tl_sell_price, tl_qty, tl_comm_buy, tl_comm_sell
    cdef int[::1]    tl_is_short
    cdef dict        tl_arrays

    cdef long[::1]   pos_sym_id, pos_buy_time_int, pos_sell_time_int, pos_exec_time_int
    cdef double[::1] pos_qty, pos_buy_price, pos_commission_buy, pos_blocked_amount, pos_exec_price
    cdef int[::1]    pos_is_short, pos_has_exec, pos_exit_reason_code

    cdef long[::1] heap_time, heap_counter
    cdef int[::1]  heap_slot

    # free-slot pool (stack): released on close, consumed on open
    cdef int[::1]  free_slot

    def __cinit__(self, tuple market_arrays, signal_events, ev_short, timeline,
                  double initial_balance, double comi_factor, double order_amount, double leverage,
                  bint full_log=False):
        if leverage <= 0.0:
            raise ValueError(f"leverage must be > 0, got {leverage}")

        (self.open_mv, self.close_mv, self.high_mv, self.low_mv,
         self.high_time_mv, self.low_time_mv, self.ts_int_mv, self.sym_len_mv) = market_arrays
        self.ev_mv       = signal_events
        self.ev_short_mv = ev_short
        self.ts_all_mv   = timeline

        self.initial_balance = initial_balance
        self.comi_factor     = comi_factor
        self.order_amount    = order_amount
        self.margin_req      = order_amount / leverage

        self.n_ticks    = self.ts_all_mv.shape[0]
        self.n_events   = self.ev_mv.shape[0]
        self.n_syms     = self.sym_len_mv.shape[0]
        self.max_trades = self.n_events + 1
        self.full_log   = full_log

        self.tl_arrays = {
            "buy_time":        np.empty(self.max_trades, dtype=np.int64),
            "sell_time":       np.empty(self.max_trades, dtype=np.int64),
            "profit":          np.empty(self.max_trades, dtype=np.float64),
            "exit_reason":     np.empty(self.max_trades, dtype=np.int32),
        }
        if full_log:
            self.tl_arrays.update({
                "sym_id":          np.empty(self.max_trades, dtype=np.int64),
                "buy_price":       np.empty(self.max_trades, dtype=np.float64),
                "sell_price":      np.empty(self.max_trades, dtype=np.float64),
                "qty":             np.empty(self.max_trades, dtype=np.float64),
                "commission_buy":  np.empty(self.max_trades, dtype=np.float64),
                "commission_sell": np.empty(self.max_trades, dtype=np.float64),
                "is_short":        np.empty(self.max_trades, dtype=np.int32),
            })
            self.tl_sym_id      = self.tl_arrays["sym_id"]
            self.tl_buy_price   = self.tl_arrays["buy_price"]
            self.tl_sell_price  = self.tl_arrays["sell_price"]
            self.tl_qty         = self.tl_arrays["qty"]
            self.tl_comm_buy    = self.tl_arrays["commission_buy"]
            self.tl_comm_sell   = self.tl_arrays["commission_sell"]
            self.tl_is_short    = self.tl_arrays["is_short"]

        self.tl_buy_time  = self.tl_arrays["buy_time"]
        self.tl_sell_time = self.tl_arrays["sell_time"]
        self.tl_profit    = self.tl_arrays["profit"]
        self.tl_exit_reason = self.tl_arrays["exit_reason"]
        self.tl_day       = np.empty(self.max_trades, dtype=np.int64)

        self.pos_sym_id           = np.empty(self.max_trades, dtype=np.int64)
        self.pos_buy_time_int     = np.empty(self.max_trades, dtype=np.int64)
        self.pos_sell_time_int    = np.empty(self.max_trades, dtype=np.int64)
        self.pos_exec_time_int    = np.empty(self.max_trades, dtype=np.int64)
        self.pos_qty              = np.empty(self.max_trades, dtype=np.float64)
        self.pos_buy_price        = np.empty(self.max_trades, dtype=np.float64)
        self.pos_commission_buy   = np.empty(self.max_trades, dtype=np.float64)
        self.pos_blocked_amount   = np.empty(self.max_trades, dtype=np.float64)
        self.pos_exec_price       = np.empty(self.max_trades, dtype=np.float64)
        self.pos_is_short         = np.empty(self.max_trades, dtype=np.int32)
        self.pos_has_exec         = np.empty(self.max_trades, dtype=np.int32)
        self.pos_exit_reason_code = np.empty(self.max_trades, dtype=np.int32)

        self.heap_time    = np.empty(self.max_trades, dtype=np.int64)
        self.heap_counter = np.empty(self.max_trades, dtype=np.int64)
        self.heap_slot    = np.empty(self.max_trades, dtype=np.int32)
        self.free_slot    = np.empty(self.max_trades, dtype=np.int32)

    cdef int simulate(self, int sell_after, double tp_pct, double sl_pct) noexcept nogil:
        cdef double cash_bank    = self.initial_balance
        cdef double blocked_cash = 0.0
        cdef double comi_factor  = self.comi_factor
        cdef double order_amount = self.order_amount
        cdef double margin_req   = self.margin_req
        cdef int    n_trades     = 0
        cdef int    heap_size    = 0
        cdef long   counter      = 0

        cdef int    tick_i, ev_scan, ev_cursor
        cdef long   t_int
        cdef int    sid, buy_idx, n_bars, exit_idx, slot, free_slot_count
        cdef long   sell_time_int, exec_time_int
        cdef double price_t, qty, comm_buy
        cdef double tp_price, sl_price
        cdef double free_cash, blocked_amount
        cdef bint   is_short, intra
        cdef int    chosen_idx, reason_code
        cdef double exec_price_intra
        cdef double qty_c, buy_price_c, comm_buy_c, comm_sell_c, profit_c, blocked_amount_c
        cdef bint   is_short_c
        cdef long   n_sym

        for slot in range(self.max_trades):
            self.free_slot[slot] = slot
        free_slot_count = self.max_trades

        ev_cursor = 0
        for tick_i in range(self.n_ticks):
            t_int = self.ts_all_mv[tick_i]

            # ── 1. Close expired / intrabar-exit positions and release their slots ──
            while heap_size > 0 and self.heap_time[0] <= t_int:
                slot = _heap_pop(&self.heap_time[0], &self.heap_counter[0], &self.heap_slot[0], &heap_size)

                if self.pos_has_exec[slot] and self.pos_exec_time_int[slot] <= t_int:
                    exec_price_intra = self.pos_exec_price[slot]
                    exec_time_int    = self.pos_exec_time_int[slot]
                    reason_code      = self.pos_exit_reason_code[slot]
                else:
                    sid   = <int>self.pos_sym_id[slot]
                    n_sym = self.sym_len_mv[sid]
                    exit_idx = _searchsorted_right(&self.ts_int_mv[sid, 0], self.pos_sell_time_int[slot], <int>n_sym) - 1
                    if exit_idx < 0:
                        exit_idx = 0

                    if sell_after == 0 or exit_idx == <int>n_sym - 1:
                        exec_price_intra = self.close_mv[sid, exit_idx]
                        reason_code      = 3
                    else:
                        exec_price_intra = self.open_mv[sid, exit_idx]
                        reason_code      = 0
                    exec_time_int = self.pos_sell_time_int[slot]

                qty_c            = self.pos_qty[slot]
                buy_price_c      = self.pos_buy_price[slot]
                is_short_c       = self.pos_is_short[slot] != 0
                comm_buy_c       = self.pos_commission_buy[slot]
                blocked_amount_c = self.pos_blocked_amount[slot]
                comm_sell_c      = qty_c * exec_price_intra * comi_factor

                if is_short_c:
                    profit_c = (buy_price_c - exec_price_intra) * qty_c - comm_buy_c - comm_sell_c
                else:
                    profit_c = (exec_price_intra - buy_price_c) * qty_c - comm_buy_c - comm_sell_c

                cash_bank    += profit_c + comm_buy_c
                blocked_cash -= blocked_amount_c

                if blocked_cash < 0.0 and blocked_cash > -1e-9:
                    blocked_cash = 0.0

                self.tl_buy_time[n_trades]  = self.pos_buy_time_int[slot]
                self.tl_sell_time[n_trades] = exec_time_int
                self.tl_profit[n_trades]    = profit_c
                self.tl_exit_reason[n_trades] = reason_code
                if self.full_log:
                    self.tl_sym_id[n_trades]      = self.pos_sym_id[slot]
                    self.tl_buy_price[n_trades]   = buy_price_c
                    self.tl_sell_price[n_trades]  = exec_price_intra
                    self.tl_qty[n_trades]         = qty_c
                    self.tl_comm_buy[n_trades]    = comm_buy_c
                    self.tl_comm_sell[n_trades]   = comm_sell_c
                    self.tl_is_short[n_trades]    = 1 if is_short_c else 0
                n_trades += 1

                self.free_slot[free_slot_count] = slot
                free_slot_count += 1

            # ── 2. Open a position for every signal at this tick while slots and cash allow ──
            while ev_cursor < self.n_events and self.ev_mv[ev_cursor, 0] < t_int:
                ev_cursor += 1
            ev_scan = ev_cursor

            while ev_scan < self.n_events and self.ev_mv[ev_scan, 0] == t_int:
                sid      = <int>self.ev_mv[ev_scan, 1]
                buy_idx  = <int>self.ev_mv[ev_scan, 2]
                is_short = self.ev_short_mv[ev_scan] != 0
                ev_scan += 1

                if free_slot_count == 0:
                    break

                n_bars    = <int>self.sym_len_mv[sid]
                free_cash = cash_bank - blocked_cash

                if free_cash < margin_req + order_amount * comi_factor:
                    break

                price_t  = self.open_mv[sid, buy_idx]
                qty      = order_amount / price_t
                comm_buy = order_amount * comi_factor

                if sell_after == 0:
                    exit_idx = n_bars - 1
                else:
                    exit_idx = buy_idx + sell_after
                    if exit_idx >= n_bars:
                        exit_idx = n_bars - 1

                sell_time_int = self.ts_int_mv[sid, exit_idx]

                if is_short:
                    tp_price = price_t * (1.0 - tp_pct / 100.0) if tp_pct != 0.0 else -HUGE_VAL
                    sl_price = price_t * (1.0 + sl_pct / 100.0) if sl_pct != 0.0 else  HUGE_VAL
                else:
                    tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else  HUGE_VAL
                    sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -HUGE_VAL

                blocked_amount = margin_req
                cash_bank     -= comm_buy
                blocked_cash  += blocked_amount

                free_slot_count -= 1
                slot = self.free_slot[free_slot_count]

                self.pos_sym_id[slot]         = sid
                self.pos_qty[slot]            = qty
                self.pos_buy_price[slot]      = price_t
                self.pos_buy_time_int[slot]   = self.ts_int_mv[sid, buy_idx]
                self.pos_sell_time_int[slot]  = sell_time_int
                self.pos_commission_buy[slot] = comm_buy
                self.pos_is_short[slot]       = 1 if is_short else 0
                self.pos_blocked_amount[slot] = blocked_amount

                _detect_intrabar_exit_cy(
                    &self.high_mv[sid, 0], &self.low_mv[sid, 0],
                    &self.high_time_mv[sid, 0], &self.low_time_mv[sid, 0],
                    buy_idx, exit_idx, tp_price, sl_price, is_short,
                    sell_after == 0 or exit_idx == n_bars - 1,
                    &intra, &chosen_idx, &reason_code, &exec_price_intra
                )

                if intra:
                    exec_time_int = self.ts_int_mv[sid, chosen_idx]
                    self.pos_has_exec[slot]         = 1
                    self.pos_exec_price[slot]       = exec_price_intra
                    self.pos_exec_time_int[slot]    = exec_time_int
                    self.pos_exit_reason_code[slot] = reason_code
                    _heap_push(&self.heap_time[0], &self.heap_counter[0], &self.heap_slot[0], &heap_size,
                               exec_time_int, counter, slot)
                else:
                    self.pos_has_exec[slot] = 0
                    _heap_push(&self.heap_time[0], &self.heap_counter[0], &self.heap_slot[0], &heap_size,
                               sell_time_int, counter, slot)

                counter += 1

        return n_trades

    cdef void aggregate_daily(
        self, int n_trades, long[::1] trading_index, long table_first, long base_index,
        double[::1] daily_row, long* out_start, long* out_len, long* out_nonzero
    ) noexcept nogil:
        cdef int  i
        cdef long day_idx, lo, hi, j, nonzero

        lo = trading_index[_floor_div(self.tl_sell_time[0], _NS_PER_DAY) - table_first] - base_index
        hi = lo
        for i in range(n_trades):
            day_idx = trading_index[_floor_div(self.tl_sell_time[i], _NS_PER_DAY) - table_first] - base_index
            self.tl_day[i] = day_idx
            if day_idx < lo:
                lo = day_idx
            if day_idx > hi:
                hi = day_idx

        for j in range(lo, hi + 1):
            daily_row[j] = 0.0
        for i in range(n_trades):
            daily_row[self.tl_day[i]] += self.tl_profit[i]

        nonzero = 0
        for j in range(lo, hi + 1):
            if daily_row[j] != 0.0:
                nonzero += 1

        out_start[0]   = lo
        out_len[0]     = hi - lo + 1
        out_nonzero[0] = nonzero

    def trade_log(self, int n_trades):
        return {name: arr[:n_trades] for name, arr in self.tl_arrays.items()}


# ============================================================
# Grid backtest  (every parameter combo of one rule in a single call)
# ============================================================
def backtest_grid(
    tuple market_arrays,
    signal_events,
    ev_short,
    timeline,
    long[::1] sell_after_arr,
    double[::1] tp_arr,
    double[::1] sl_arr,
    double initial_balance,
    double comi_factor,
    double order_amount,
    long min_trades,
    long[::1] trading_index,
    long table_first,
    long base_index,
    long n_days_range,
    double leverage = LEVERAGE
):
    cdef _GridEngine engine = _GridEngine(
        market_arrays, signal_events, ev_short, timeline,
        initial_balance, comi_factor, order_amount, leverage,
    )
    cdef Py_ssize_t n_combos = sell_after_arr.shape[0]
    cdef Py_ssize_t c, i
    cdef int        n

    n_trades_arr  = np.zeros(n_combos, dtype=np.int64)
    day_start_arr = np.full(n_combos, -1, dtype=np.int64)
    n_days_arr    = np.zeros(n_combos, dtype=np.int64)
    nonzero_arr   = np.zeros(n_combos, dtype=np.int64)
    daily_arr     = np.empty((n_combos, n_days_range), dtype=np.float64)
    duration_arr  = np.empty((n_combos, engine.max_trades), dtype=np.int64)

    cdef long[::1]     n_trades_mv  = n_trades_arr
    cdef long[::1]     day_start_mv = day_start_arr
    cdef long[::1]     n_days_mv    = n_days_arr
    cdef long[::1]     nonzero_mv   = nonzero_arr
    cdef double[:, ::1] daily_mv    = daily_arr
    cdef long[:, ::1]  duration_mv  = duration_arr

    with nogil:
        for c in range(n_combos):
            n = engine.simulate(<int>sell_after_arr[c], tp_arr[c], sl_arr[c])
            n_trades_mv[c] = n
            for i in range(n):
                duration_mv[c, i] = engine.tl_sell_time[i] - engine.tl_buy_time[i]
            if n == 0 or n < min_trades:
                continue
            engine.aggregate_daily(
                n, trading_index, table_first, base_index, daily_mv[c],
                &day_start_mv[c], &n_days_mv[c], &nonzero_mv[c],
            )

    return n_trades_arr, day_start_arr, n_days_arr, nonzero_arr, daily_arr, duration_arr


# ============================================================
# Single backtest API  (prepare once per window, simulate per parameter set)
# ============================================================
def _normalized_signal(signal, long n):
    # float32 signals are used as-is; other dtypes are mapped to {-1, 0, 1} with the same
    # non-zero test and int64 -> int32 sign rule the engine applies to float32 values
    sig = np.asarray(signal)[:n]
    if sig.dtype == np.float32:
        return np.ascontiguousarray(sig)
    nonzero  = sig != 0
    is_short = sig.astype(np.int64).astype(np.int32) < 0
    return np.where(nonzero, np.where(is_short, -1.0, 1.0), 0.0).astype(np.float32)


def prepare_backtest_data(ohlcv_arrays):
    static_bundle = prepare_static_arrays(ohlcv_arrays)
    sym_ids       = static_bundle["sym_ids"]
    sym_len       = static_bundle["sym_len"]
    signals = tuple(
        _normalized_signal(ohlcv_arrays[sym]["signal"], sym_len[sym_ids[sym]])
        for sym in static_bundle["symbols_by_sid"]
    )
    signal_events, ev_short, _ = build_rule_events(
        signals, static_bundle["ts_int_2d"], sym_len, static_bundle["tick_pos_2d"],
        static_bundle["all_timestamps_int"], static_bundle["idx_workspace"],
    )
    return {"static": static_bundle, "signal_events": signal_events, "ev_short": ev_short}


def _simulate_prepared(prepared_data, sell_after, tp_pct, sl_pct, order_amount, bint full_log):
    static_bundle = prepared_data["static"]
    cdef _GridEngine engine = _GridEngine(
        market_arrays(static_bundle), prepared_data["signal_events"], prepared_data["ev_short"],
        static_bundle["all_timestamps_int"],
        float(INITIAL_BALANCE), float(COMISION) / 100.0, float(order_amount), LEVERAGE, full_log,
    )
    cdef int    sell_after_c = int(sell_after)
    cdef double tp_c         = float(tp_pct)
    cdef double sl_c         = float(sl_pct)
    cdef int    n_trades
    with nogil:
        n_trades = engine.simulate(sell_after_c, tp_c, sl_c)
    return engine.trade_log(n_trades)


def run_backtest_from_prepared(prepared_data, sell_after, tp_pct, sl_pct, order_amount):
    static_bundle = prepared_data["static"]
    log           = _simulate_prepared(prepared_data, sell_after, tp_pct, sl_pct, order_amount, True)

    sym_ids           = static_bundle["sym_ids"]
    sym_names         = np.empty(len(sym_ids), dtype=object)
    symbol_order_rank = np.empty(len(sym_ids), dtype=np.int64)
    for i, sym in enumerate(static_bundle["symbols"]):
        sid = sym_ids[sym]
        sym_names[sid]         = sym
        symbol_order_rank[sid] = i

    trade_log = pd.DataFrame({
        'symbol':          sym_names[log["sym_id"]],
        'buy_time':        log["buy_time"].astype('datetime64[ns]'),
        'buy_price':       log["buy_price"],
        'sell_time':       log["sell_time"].astype('datetime64[ns]'),
        'sell_price':      log["sell_price"],
        'qty':             log["qty"],
        'profit':          log["profit"],
        'exit_reason':     EXIT_REASON_NAMES[log["exit_reason"]],
        'commission_buy':  log["commission_buy"],
        'commission_sell': log["commission_sell"],
        'position_type':   POSITION_TYPE_NAMES[log["is_short"]],
    })

    order       = np.argsort(symbol_order_rank[log["sym_id"]], kind='stable')
    trades_list = log["profit"][order].tolist()

    return {
        "__PORTFOLIO__": {
            'trades':    trades_list,
            'trade_log': trade_log,
        }
    }


def trade_log_from_prepared(prepared_data, sell_after, tp_pct, sl_pct, order_amount):
    # Raw trade log without a DataFrame: buy_time and sell_time (int64 ns), profit, exit_reason (EXIT_REASON_NAMES index)
    return _simulate_prepared(prepared_data, sell_after, tp_pct, sl_pct, order_amount, False)