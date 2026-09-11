#core/backtesters/ZX_compute_BT_NPY.pyx (HEAP_SIZE==0,nopyramid)
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
logging.basicConfig(level=logging.INFO)
from setup.config_backtest import INITIAL_BALANCE, COMISION
warnings.filterwarnings("ignore")

# ============================================================
# C-level binary search helpers  (replaces np.searchsorted)
# ============================================================

cdef int _searchsorted_right(long* arr, long val, int n) noexcept nogil:
    """Return first index i where arr[i] > val (right side)."""
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

# ============================================================
# prepare_data  (pure Python, no Cython types needed)
# ============================================================
def prepare_static_arrays(ohlcv_arrays):
    if not ohlcv_arrays:
        return {
            "symbols": [], "sym_ids": {}, "ts_int_arrays": {}, "close_arrays": {},
            "all_timestamps_int": np.array([], dtype=np.int64),
            "all_timestamps_dt":  np.array([], dtype='datetime64[ns]'),
            "open_2d": np.empty((0, 0)), "close_2d": np.empty((0, 0)),
            "high_2d": np.empty((0, 0)), "low_2d": np.empty((0, 0)),
            "high_time_2d": np.empty((0, 0), dtype=np.int64),
            "low_time_2d":  np.empty((0, 0), dtype=np.int64),
            "ts_int_2d":    np.empty((0, 0), dtype=np.int64),
            "sym_len":      np.array([], dtype=np.int64),
        }

    symbols       = list(ohlcv_arrays.keys())
    n_syms        = len(symbols)
    sym_ids       = {s: i for i, s in enumerate(sorted(symbols))}
    ts_int_arrays = {}
    close_arrays  = {}
    per_sym_ts    = {}
    all_ts_int_lists = []

    for sym in symbols:
        data = ohlcv_arrays[sym]
        ts   = data['ts']
        if ts.dtype.kind != 'M':
            ts = ts.astype('datetime64[ns]')
        ts_int = ts.view('int64').copy()  # own memory: bundle outlives the shm-backed input
        per_sym_ts[sym] = (ts_int, len(ts))
        ts_int_arrays[sym] = ts_int
        close_arrays[sym]  = np.array(data['close'], dtype=np.float32, copy=True)
        all_ts_int_lists.append(ts_int)

    max_len      = max(per_sym_ts[s][1] for s in symbols)
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
        ts_int_2d[sid, :n]    = ts_int.astype(np.int64)

    all_timestamps_int = np.unique(np.concatenate(all_ts_int_lists))
    all_timestamps_dt  = all_timestamps_int.view('datetime64[ns]')

    return {
        "symbols":            symbols,
        "sym_ids":            sym_ids,
        "ts_int_arrays":      ts_int_arrays,
        "close_arrays":       close_arrays,
        "all_timestamps_int": all_timestamps_int,
        "all_timestamps_dt":  all_timestamps_dt,
        "open_2d":            open_2d,
        "close_2d":           close_2d,
        "high_2d":            high_2d,
        "low_2d":             low_2d,
        "high_time_2d":       high_time_2d,
        "low_time_2d":        low_time_2d,
        "ts_int_2d":          ts_int_2d,
        "sym_len":            sym_len,
    }

def prepare_signal_arrays(static_bundle, ohlcv_arrays):
    symbols = static_bundle["symbols"]
    sym_ids = static_bundle["sym_ids"]
    sym_len = static_bundle["sym_len"]
    max_len = static_bundle["open_2d"].shape[1]
    n_syms  = len(symbols)

    sym_data = {}
    for sym in symbols:
        data   = ohlcv_arrays[sym]
        sid    = sym_ids[sym]
        n      = int(sym_len[sid])
        ts_int = static_bundle["ts_int_arrays"][sym]
        sym_data[sym] = {
            'ts':        ts_int.view('datetime64[ns]'),
            'ts_int':    ts_int,
            'open':      data['open'],
            'close':     data['close'],
            'high':      data['high'],
            'low':       data['low'],
            'signal':    data['signal'][:n],
            'len':       n,
            'high_time': data['high_time'],
            'low_time':  data['low_time'],
        }

    signal_2d    = np.full((n_syms, max_len), 0, dtype=np.int64)
    event_chunks = []
    for sym in symbols:
        sid        = sym_ids[sym]
        d          = sym_data[sym]
        signal_arr = d['signal']
        n          = d['len']
        signal_2d[sid, :n] = signal_arr.astype(np.int64)

        sig_idxs = np.flatnonzero(signal_arr)
        sig_idxs = sig_idxs[sig_idxs < n]
        if sig_idxs.size:
            ts_ints = d['ts_int'][sig_idxs]
            chunk   = np.empty((sig_idxs.size, 3), dtype=np.int64)
            chunk[:, 0] = ts_ints
            chunk[:, 1] = sid
            chunk[:, 2] = sig_idxs
            event_chunks.append(chunk)

    if event_chunks:
        signal_events = np.concatenate(event_chunks, axis=0)
        order         = np.lexsort((signal_events[:, 1], signal_events[:, 0]))
        signal_events = signal_events[order]
    else:
        signal_events = np.empty((0, 3), dtype=np.int64)

    if signal_events.shape[0] > 0:
        ev_col0 = np.ascontiguousarray(signal_events[:, 0], dtype=np.int64)
    else:
        ev_col0 = np.empty((0,), dtype=np.int64)

    arrays = (
        static_bundle["open_2d"], static_bundle["close_2d"],
        static_bundle["high_2d"], static_bundle["low_2d"],
        static_bundle["high_time_2d"], static_bundle["low_time_2d"],
        static_bundle["ts_int_2d"], signal_2d, sym_len,
        signal_events, static_bundle["all_timestamps_int"], ev_col0
    )

    return (sym_data, {}, static_bundle["all_timestamps_int"], static_bundle["all_timestamps_dt"],
            sym_ids, static_bundle["ts_int_arrays"], static_bundle["close_arrays"], arrays)

def prepare_data(ohlcv_arrays):
    if not ohlcv_arrays:
        return ({}, {}, np.array([], dtype=np.int64),
                np.array([], dtype='datetime64[ns]'), {}, {}, {})
    static_bundle = prepare_static_arrays(ohlcv_arrays)
    return prepare_signal_arrays(static_bundle, ohlcv_arrays)

# ============================================================
# prepare_backtest_data  (public alias — cache-friendly prepare step)
# ============================================================
def prepare_backtest_data(ohlcv_arrays):

    return prepare_data(ohlcv_arrays)

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
# backtest_core  (typed Cython hot loop)
# ============================================================
def backtest_core(
    np.ndarray[float, ndim=2] open_2d,
    np.ndarray[float, ndim=2] close_2d,
    np.ndarray[float, ndim=2] high_2d,
    np.ndarray[float, ndim=2] low_2d,
    np.ndarray[long,   ndim=2] high_time_2d,
    np.ndarray[long,   ndim=2] low_time_2d,
    np.ndarray[long,   ndim=2] ts_int_2d,
    np.ndarray[long,   ndim=2] signal_2d,
    np.ndarray[long,   ndim=1] sym_len,
    np.ndarray[long,   ndim=2] signal_events,
    np.ndarray[long,   ndim=1] all_timestamps_int,
    np.ndarray[long,   ndim=1] ev_col0,
    double initial_balance,
    double comi_factor,
    double order_amount,
    int sell_after,
    double tp_pct,
    double sl_pct
):
    cdef int n_ticks    = len(all_timestamps_int)
    cdef int n_events   = len(signal_events)
    cdef int max_trades = n_events + 1
    cdef int n_syms     = sym_len.shape[0]

    # ── Pre-allocated trade log (typed memoryviews) ──
    cdef np.ndarray[long,   ndim=1] tl_sym_id_arr      = np.empty(max_trades, dtype=np.int64)
    cdef np.ndarray[long,   ndim=1] tl_buy_time_arr    = np.empty(max_trades, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] tl_buy_price_arr   = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[long,   ndim=1] tl_sell_time_arr   = np.empty(max_trades, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] tl_sell_price_arr  = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] tl_qty_arr         = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] tl_profit_arr      = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[int,    ndim=1] tl_exit_reason_arr = np.empty(max_trades, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] tl_comm_buy_arr    = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] tl_comm_sell_arr   = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[int,    ndim=1] tl_is_short_arr    = np.empty(max_trades, dtype=np.int32)

    cdef long[::1]   tl_sym_id      = tl_sym_id_arr
    cdef long[::1]   tl_buy_time    = tl_buy_time_arr
    cdef double[::1] tl_buy_price   = tl_buy_price_arr
    cdef long[::1]   tl_sell_time   = tl_sell_time_arr
    cdef double[::1] tl_sell_price  = tl_sell_price_arr
    cdef double[::1] tl_qty         = tl_qty_arr
    cdef double[::1] tl_profit      = tl_profit_arr
    cdef int[::1]    tl_exit_reason = tl_exit_reason_arr
    cdef double[::1] tl_comm_buy    = tl_comm_buy_arr
    cdef double[::1] tl_comm_sell   = tl_comm_sell_arr
    cdef int[::1]    tl_is_short    = tl_is_short_arr

    # ── Open-position state (flat arrays, indexed by slot 0..n_syms-1) ──
    cdef np.ndarray[long,   ndim=1] pos_sym_id_arr          = np.empty(n_syms, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] pos_qty_arr              = np.empty(n_syms, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] pos_buy_price_arr        = np.empty(n_syms, dtype=np.float64)
    cdef np.ndarray[long,   ndim=1] pos_buy_time_int_arr     = np.empty(n_syms, dtype=np.int64)
    cdef np.ndarray[long,   ndim=1] pos_sell_time_int_arr    = np.empty(n_syms, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] pos_commission_buy_arr   = np.empty(n_syms, dtype=np.float64)
    cdef np.ndarray[int,    ndim=1] pos_is_short_arr         = np.empty(n_syms, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] pos_blocked_amount_arr   = np.empty(n_syms, dtype=np.float64)
    cdef np.ndarray[int,    ndim=1] pos_has_exec_arr         = np.empty(n_syms, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] pos_exec_price_arr       = np.empty(n_syms, dtype=np.float64)
    cdef np.ndarray[long,   ndim=1] pos_exec_time_int_arr    = np.empty(n_syms, dtype=np.int64)
    cdef np.ndarray[int,    ndim=1] pos_exit_reason_code_arr = np.empty(n_syms, dtype=np.int32)

    cdef long[::1]   pos_sym_id          = pos_sym_id_arr
    cdef double[::1] pos_qty              = pos_qty_arr
    cdef double[::1] pos_buy_price        = pos_buy_price_arr
    cdef long[::1]   pos_buy_time_int     = pos_buy_time_int_arr
    cdef long[::1]   pos_sell_time_int    = pos_sell_time_int_arr
    cdef double[::1] pos_commission_buy   = pos_commission_buy_arr
    cdef int[::1]    pos_is_short         = pos_is_short_arr
    cdef double[::1] pos_blocked_amount   = pos_blocked_amount_arr
    cdef int[::1]    pos_has_exec         = pos_has_exec_arr
    cdef double[::1] pos_exec_price       = pos_exec_price_arr
    cdef long[::1]   pos_exec_time_int    = pos_exec_time_int_arr
    cdef int[::1]    pos_exit_reason_code = pos_exit_reason_code_arr

    # ── Manual min-heap over (exec_time, counter) -> slot ──
    cdef np.ndarray[long, ndim=1] heap_time_arr    = np.empty(n_syms, dtype=np.int64)
    cdef np.ndarray[long, ndim=1] heap_counter_arr = np.empty(n_syms, dtype=np.int64)
    cdef np.ndarray[int,  ndim=1] heap_slot_arr    = np.empty(n_syms, dtype=np.int32)

    cdef long[::1] heap_time    = heap_time_arr
    cdef long[::1] heap_counter = heap_counter_arr
    cdef int[::1]  heap_slot    = heap_slot_arr
    cdef int       heap_size    = 0

    # ── Account state ──
    cdef double cash_bank    = initial_balance
    cdef double blocked_cash = 0.0
    cdef int    n_trades     = 0

    # ── Loop vars ──
    cdef int    tick_i, ev_scan, ev_cursor
    cdef long   t_int, exp_time
    cdef int    sid, buy_idx, n_bars, exit_idx, slot, batch_slot
    cdef long   sell_time_int, exec_time_int
    cdef double price_t, qty, comm_buy
    cdef double tp_price, sl_price
    cdef double free_cash, proceeds, margin_req, blocked_amount
    cdef int    sig_val
    cdef bint   is_short, intra, was_empty_before, search_signals, closed_any_tp_sl
    cdef int    chosen_idx, reason_code
    cdef double exec_price_intra
    cdef double qty_c, buy_price_c, comm_buy_c, comm_sell_c, profit_c, blocked_amount_c
    cdef bint   is_short_c
    cdef long   n_sym
    cdef long   counter = 0

    # ── Typed memoryviews for 2D arrays ──
    cdef float[:, ::1] open_mv      = open_2d
    cdef float[:, ::1] close_mv     = close_2d
    cdef float[:, ::1] high_mv      = high_2d
    cdef float[:, ::1] low_mv       = low_2d
    cdef long[:, ::1]   high_time_mv = high_time_2d
    cdef long[:, ::1]   low_time_mv  = low_time_2d
    cdef long[:, ::1]   ts_int_mv    = ts_int_2d
    cdef long[:, ::1]   signal_mv    = signal_2d
    cdef long[::1]      sym_len_mv   = sym_len
    cdef long[:, ::1]   ev_mv        = signal_events
    cdef long[::1]      ts_all_mv    = all_timestamps_int
    cdef long[::1]      ev_col0_mv   = ev_col0

    with nogil:
        ev_cursor = 0
        for tick_i in range(n_ticks):
            t_int = ts_all_mv[tick_i]

            # ── 1. Close expired / intrabar-exit positions ──
            was_empty_before = (heap_size == 0)
            closed_any_tp_sl = False

            while heap_size > 0 and heap_time[0] <= t_int:
                slot = _heap_pop(&heap_time[0], &heap_counter[0], &heap_slot[0], &heap_size)

                if pos_has_exec[slot] and pos_exec_time_int[slot] <= t_int:
                    exec_price_intra = pos_exec_price[slot]
                    exec_time_int    = pos_exec_time_int[slot]
                    reason_code      = pos_exit_reason_code[slot]
                else:
                    sid   = <int>pos_sym_id[slot]
                    n_sym = sym_len_mv[sid]
                    exit_idx = _searchsorted_right(&ts_int_mv[sid, 0], pos_sell_time_int[slot], <int>n_sym) - 1
                    if exit_idx < 0:
                        exit_idx = 0

                    if sell_after == 0 or exit_idx == <int>n_sym - 1:
                        exec_price_intra = close_mv[sid, exit_idx]
                        reason_code      = 3
                    else:
                        exec_price_intra = open_mv[sid, exit_idx]
                        reason_code      = 0
                    exec_time_int    = pos_sell_time_int[slot]

                qty_c            = pos_qty[slot]
                buy_price_c       = pos_buy_price[slot]
                is_short_c        = pos_is_short[slot] != 0
                comm_buy_c        = pos_commission_buy[slot]
                blocked_amount_c  = pos_blocked_amount[slot]
                comm_sell_c       = qty_c * exec_price_intra * comi_factor

                if is_short_c:
                    cash_bank    -= qty_c * exec_price_intra + comm_sell_c
                    blocked_cash -= blocked_amount_c
                    profit_c      = (buy_price_c - exec_price_intra) * qty_c - comm_buy_c - comm_sell_c
                else:
                    cash_bank += qty_c * exec_price_intra - comm_sell_c
                    profit_c   = (exec_price_intra - buy_price_c) * qty_c - comm_buy_c - comm_sell_c

                if blocked_cash < 0.0 and blocked_cash > -1e-9:
                    blocked_cash = 0.0

                tl_sym_id[n_trades]      = pos_sym_id[slot]
                tl_buy_time[n_trades]    = pos_buy_time_int[slot]
                tl_buy_price[n_trades]   = buy_price_c
                tl_sell_time[n_trades]   = exec_time_int
                tl_sell_price[n_trades]  = exec_price_intra
                tl_qty[n_trades]         = qty_c
                tl_profit[n_trades]      = profit_c
                tl_exit_reason[n_trades] = reason_code
                tl_comm_buy[n_trades]    = comm_buy_c
                tl_comm_sell[n_trades]   = comm_sell_c
                tl_is_short[n_trades]    = 1 if is_short_c else 0
                n_trades += 1

                if reason_code == 1 or reason_code == 2:
                    closed_any_tp_sl = True

            # ── 2. Open new positions if heap empty ──
            if heap_size == 0:
                if was_empty_before:
                    search_signals = True
                else:
                    search_signals = not closed_any_tp_sl

                if search_signals:
                    while ev_cursor < n_events and ev_mv[ev_cursor, 0] < t_int:
                        ev_cursor += 1
                    ev_scan    = ev_cursor
                    batch_slot = 0

                    while ev_scan < n_events and ev_mv[ev_scan, 0] == t_int:
                        sid     = <int>ev_mv[ev_scan, 1]
                        buy_idx = <int>ev_mv[ev_scan, 2]
                        ev_scan += 1

                        n_bars    = <int>sym_len_mv[sid]
                        free_cash = cash_bank - blocked_cash

                        if free_cash < order_amount:
                            break

                        sig_val  = <int>signal_mv[sid, buy_idx]
                        is_short = sig_val < 0

                        if is_short and sl_pct == 0.0:
                            continue

                        if is_short:
                            if free_cash < order_amount * (sl_pct / 100.0) + order_amount * comi_factor:
                                continue

                        price_t  = open_mv[sid, buy_idx]
                        qty      = order_amount / price_t
                        comm_buy = order_amount * comi_factor

                        if sell_after == 0:
                            exit_idx = n_bars - 1
                        else:
                            exit_idx = buy_idx + sell_after
                            if exit_idx >= n_bars:
                                exit_idx = n_bars - 1

                        sell_time_int = ts_int_mv[sid, exit_idx]

                        if is_short:
                            tp_price = price_t * (1.0 - tp_pct / 100.0) if tp_pct != 0.0 else -HUGE_VAL
                            sl_price = price_t * (1.0 + sl_pct / 100.0) if sl_pct != 0.0 else  HUGE_VAL
                        else:
                            tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else  HUGE_VAL
                            sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -HUGE_VAL

                        if is_short:
                            proceeds       = order_amount - comm_buy
                            margin_req     = order_amount * (sl_pct / 100.0) if sl_pct != 0.0 else HUGE_VAL
                            blocked_amount = proceeds + margin_req
                            cash_bank     += proceeds
                            blocked_cash  += blocked_amount
                        else:
                            blocked_amount = 0.0
                            cash_bank     -= (order_amount + comm_buy)

                        slot = batch_slot
                        batch_slot += 1

                        pos_sym_id[slot]        = sid
                        pos_qty[slot]           = qty
                        pos_buy_price[slot]     = price_t
                        pos_buy_time_int[slot]  = ts_int_mv[sid, buy_idx]
                        pos_sell_time_int[slot] = sell_time_int
                        pos_commission_buy[slot] = comm_buy
                        pos_is_short[slot]      = 1 if is_short else 0
                        pos_blocked_amount[slot] = blocked_amount

                        _detect_intrabar_exit_cy(
                            &high_mv[sid, 0], &low_mv[sid, 0],
                            &high_time_mv[sid, 0], &low_time_mv[sid, 0],
                            buy_idx, exit_idx, tp_price, sl_price, is_short,
                            sell_after == 0 or exit_idx == n_bars - 1,
                            &intra, &chosen_idx, &reason_code, &exec_price_intra
                        )

                        if intra:
                            exec_time_int = ts_int_mv[sid, chosen_idx]
                            pos_has_exec[slot]         = 1
                            pos_exec_price[slot]       = exec_price_intra
                            pos_exec_time_int[slot]    = exec_time_int
                            pos_exit_reason_code[slot] = reason_code
                            _heap_push(&heap_time[0], &heap_counter[0], &heap_slot[0], &heap_size,
                                       exec_time_int, counter, slot)
                        else:
                            pos_has_exec[slot] = 0
                            _heap_push(&heap_time[0], &heap_counter[0], &heap_slot[0], &heap_size,
                                       sell_time_int, counter, slot)

                        counter += 1

    return (
        n_trades,
        tl_sym_id_arr[:n_trades],    tl_buy_time_arr[:n_trades],   tl_buy_price_arr[:n_trades],
        tl_sell_time_arr[:n_trades], tl_sell_price_arr[:n_trades], tl_qty_arr[:n_trades],
        tl_profit_arr[:n_trades],    tl_exit_reason_arr[:n_trades],
        tl_comm_buy_arr[:n_trades],  tl_comm_sell_arr[:n_trades],  tl_is_short_arr[:n_trades],
        cash_bank, blocked_cash
    )
# ============================================================
# _run_core_from_arrays  (shared simulation step)
# ============================================================
def _run_core_from_arrays(arrays, sym_ids, ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount):
    cdef_factor     = float(COMISION) / 100.0
    initial_balance = float(INITIAL_BALANCE)

    (open_2d, close_2d, high_2d, low_2d,
     high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
     signal_events, all_timestamps_int, ev_col0) = arrays

    symbols = list(ohlcv_arrays.keys())
    n_syms  = len(sym_ids)

    sym_names         = np.empty(n_syms, dtype=object)
    symbol_order_rank = np.empty(n_syms, dtype=np.int64)
    for i, sym in enumerate(symbols):
        sid = sym_ids[sym]
        sym_names[sid]         = sym
        symbol_order_rank[sid] = i

    (
        n_trades,
        tl_sym_id, tl_buy_time, tl_buy_price,
        tl_sell_time, tl_sell_price, tl_qty,
        tl_profit, tl_exit_reason,
        tl_comm_buy, tl_comm_sell, tl_is_short,
        final_cash_bank, _
    ) = backtest_core(
        open_2d, close_2d, high_2d, low_2d,
        high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
        signal_events, all_timestamps_int, ev_col0,
        initial_balance, cdef_factor, float(order_amount),
        int(sell_after), float(tp_pct), float(sl_pct)
    )

    exit_reason_names   = np.array(['SELL_AFTER', 'TP', 'SL', 'END_OF_DATA'], dtype=object)
    position_type_names = np.array(['LONG', 'SHORT'], dtype=object)

    trade_log = pd.DataFrame({
        'symbol':          sym_names[tl_sym_id],
        'buy_time':        tl_buy_time.astype('datetime64[ns]'),
        'buy_price':       tl_buy_price,
        'sell_time':       tl_sell_time.astype('datetime64[ns]'),
        'sell_price':      tl_sell_price,
        'qty':             tl_qty,
        'profit':          tl_profit,
        'exit_reason':     exit_reason_names[tl_exit_reason],
        'commission_buy':  tl_comm_buy,
        'commission_sell': tl_comm_sell,
        'position_type':   position_type_names[tl_is_short],
    })

    rank_per_trade = symbol_order_rank[tl_sym_id]
    order          = np.argsort(rank_per_trade, kind='stable')
    trades_list    = tl_profit[order].tolist()

    return {
        "__PORTFOLIO__": {
            'trades':    trades_list,
            'trade_log': trade_log,
        }
    }
# ============================================================
# run_backtest_from_prepared  (simulate step — reuse prepared data)
# ============================================================
def run_backtest_from_prepared(prepared_data, sell_after, tp_pct, sl_pct, order_amount):

    (sym_data, _, all_timestamps_int, all_timestamps_dt,
     sym_ids, ts_int_arrays, close_arrays, arrays) = prepared_data

    return _run_core_from_arrays(
        arrays, sym_ids, sym_data,
        sell_after, tp_pct, sl_pct, order_amount
    )

def _run_core_from_arrays_light(arrays, sell_after, tp_pct, sl_pct, order_amount):
    cdef_factor     = float(COMISION) / 100.0
    initial_balance = float(INITIAL_BALANCE)

    (open_2d, close_2d, high_2d, low_2d,
     high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
     signal_events, all_timestamps_int, ev_col0) = arrays

    (
        n_trades,
        tl_sym_id, tl_buy_time, tl_buy_price,
        tl_sell_time, tl_sell_price, tl_qty,
        tl_profit, tl_exit_reason,
        tl_comm_buy, tl_comm_sell, tl_is_short,
        final_cash_bank, _
    ) = backtest_core(
        open_2d, close_2d, high_2d, low_2d,
        high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
        signal_events, all_timestamps_int, ev_col0,
        initial_balance, cdef_factor, float(order_amount),
        int(sell_after), float(tp_pct), float(sl_pct)
    )

    trade_log = pd.DataFrame({
        'buy_time':  tl_buy_time.astype('datetime64[ns]'),
        'sell_time': tl_sell_time.astype('datetime64[ns]'),
        'profit':    tl_profit,
    })

    return {
        "__PORTFOLIO__": {
            'trade_log': trade_log,
        }
    }
# ============================================================
# run_backtest_from_prepared_light  (simulate step — metrics-only output)
# ============================================================
def run_backtest_from_prepared_light(prepared_data, sell_after, tp_pct, sl_pct, order_amount):

    (sym_data, _, all_timestamps_int, all_timestamps_dt,
     sym_ids, ts_int_arrays, close_arrays, arrays) = prepared_data

    return _run_core_from_arrays_light(
        arrays, sell_after, tp_pct, sl_pct, order_amount
    )
# ============================================================
# run_grid_backtest  --  public API (same signature & output)
# ============================================================
def run_grid_backtest(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount):

    prepared_data = prepare_backtest_data(ohlcv_arrays)

    return run_backtest_from_prepared(
        prepared_data,
        sell_after   = sell_after,
        tp_pct       = tp_pct,
        sl_pct       = sl_pct,
        order_amount = order_amount
    )