# core/pipeline/signal_cleaning.py NEW
import logging
import numpy as np
import cupy as cp
from joblib import Parallel, delayed
from tqdm import tqdm
from signals.indicators_bank import ConditionBank
from signals.signal_builder import build_signal_fn

logger = logging.getLogger("BOT_batch.pipeline.signal_cleaning")
#==============================================================================
# CONFIG
#==============================================================================
JACCARD_SIMILARITY_TH  = 0.82
#------------------------------------------------------------------------------


SIGNAL_MASK_N_JOBS     = -1
JACCARD_TILE           = 32     # output tile side; must stay in sync with the CUDA kernel
JACCARD_TILE_WORDS     = 8      # uint64 words staged in shared memory per tile pass
JACCARD_PRUNE_MARGIN   = 1e-4   # safety margin on the |A| window for pruning (only widens it)
JACCARD_BATCH_SIZE     = 1000
GPU_INITIAL_CAPACITY   = 20_000 # initial survivors buffer capacity, doubled on overflow
SPEC_GATHER_BLOCK      = 256    # threads per block in the rule gather-AND kernel
JACCARD_EXIT_CHECK     = 16     # tile passes between early-exit checks inside a candidate x survivor block


def _spec_identity(spec: dict) -> tuple:

    return (spec["key"], spec["op"], spec["threshold"])


def _collect_unique_specs(all_rules: list) -> tuple:

    unique_specs = []
    index_by_identity = {}
    for rule in all_rules:
        for spec in rule["specs"]:
            identity = _spec_identity(spec)
            if identity not in index_by_identity:
                index_by_identity[identity] = len(unique_specs)
                unique_specs.append(spec)
    return unique_specs, index_by_identity


def _compute_spec_signals_symbol(unique_specs: list, arr: dict) -> np.ndarray:

    bank = ConditionBank(arr)
    rows = np.empty((len(unique_specs), bank.n), dtype=bool)
    for i, spec in enumerate(unique_specs):
        signal = build_signal_fn([spec], "long")(arr, live_trading=False, bank=bank)
        rows[i] = signal.astype(bool)
    return rows


def _build_spec_word_table(unique_specs: list, ohlcv_arr: dict, n_jobs: int, timeframe: str = "") -> tuple:

    symbols = list(ohlcv_arr.keys())

    rows_by_symbol = list(tqdm(
        Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
            delayed(_compute_spec_signals_symbol)(unique_specs, ohlcv_arr[sym])
            for sym in symbols
        ),
        desc=f"SIGNAL MASK     {timeframe}",
        total=len(symbols),
        dynamic_ncols=True,
    ))

    packed = np.packbits(np.concatenate(rows_by_symbol, axis=1), axis=1)
    n_bytes = packed.shape[1]

    word_padding = (-n_bytes) % np.dtype(np.uint64).itemsize
    if word_padding:
        packed = np.pad(packed, ((0, 0), (0, word_padding)))

    words = np.ascontiguousarray(packed).view(np.uint64)
    return words, n_bytes


def _rule_spec_index(rules: list, index_by_identity: dict) -> tuple:

    rule_rows = [[index_by_identity[_spec_identity(spec)] for spec in rule["specs"]] for rule in rules]
    max_specs = max(len(rows) for rows in rule_rows)
    # AND is idempotent: padding with the rule's first spec leaves its signal unchanged.
    spec_idx = np.array([rows + [rows[0]] * (max_specs - len(rows)) for rows in rule_rows], dtype=np.int32)
    sides    = np.array([rule["side"] for rule in rules])
    return spec_idx, sides

# =============================================================================
# JACCARD SIMILARITY FILTER (GPU, bit-packed) — standalone, near-duplicate
# =============================================================================

_GATHER_AND_POPCOUNT_SOURCE = r"""
extern "C" __global__
void gather_and_popcount(
    const unsigned long long* __restrict__ spec_words,
    const int* __restrict__ spec_idx,
    unsigned long long* __restrict__ rule_words,
    unsigned long long* __restrict__ rule_cards,
    const int n_specs_per_rule,
    const int n_words)
{
    const int row  = blockIdx.x;
    const int word = blockIdx.y * blockDim.x + threadIdx.x;
    const int* idx = spec_idx + (size_t)row * n_specs_per_rule;

    unsigned int count = 0;
    if (word < n_words) {
        unsigned long long acc = spec_words[(size_t)idx[0] * n_words + word];
        for (int j = 1; j < n_specs_per_rule; ++j) {
            acc &= spec_words[(size_t)idx[j] * n_words + word];
        }
        rule_words[(size_t)row * n_words + word] = acc;
        count = __popcll(acc);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffffu, count, offset);
    }
    if ((threadIdx.x & 31) == 0 && count > 0) {
        atomicAdd(&rule_cards[row], (unsigned long long)count);
    }
}
"""

_GATHER_AND_POPCOUNT_KERNEL = cp.RawKernel(_GATHER_AND_POPCOUNT_SOURCE, "gather_and_popcount")


def _gather_rule_words_gpu(spec_words_gpu: cp.ndarray, batch_spec_idx: np.ndarray) -> tuple:

    n_rows, n_specs_per_rule = batch_spec_idx.shape
    n_words = spec_words_gpu.shape[1]

    idx_gpu = cp.asarray(np.ascontiguousarray(batch_spec_idx, dtype=np.int32))
    words   = cp.empty((n_rows, n_words), dtype=cp.uint64)
    cards   = cp.zeros(n_rows, dtype=cp.uint64)

    grid = (n_rows, (n_words + SPEC_GATHER_BLOCK - 1) // SPEC_GATHER_BLOCK)
    _GATHER_AND_POPCOUNT_KERNEL(
        grid,
        (SPEC_GATHER_BLOCK,),
        (spec_words_gpu, idx_gpu, words, cards, np.int32(n_specs_per_rule), np.int32(n_words)),
    )
    return words, cards


_MANAGED_POOL = cp.cuda.MemoryPool(cp.cuda.malloc_managed)

def _alloc_managed_packed_survivors(n_rows: int, n_words: int) -> cp.ndarray:

    mem = _MANAGED_POOL.malloc(n_rows * n_words * cp.dtype(cp.uint64).itemsize)
    return cp.ndarray((n_rows, n_words), dtype=cp.uint64, memptr=mem)


def _ensure_packed_survivor_capacity(survivors_gpu: cp.ndarray, n_survivors: int, n_new: int, n_words: int) -> cp.ndarray:

    capacity = survivors_gpu.shape[0]
    if n_survivors + n_new <= capacity:
        return survivors_gpu

    new_capacity = capacity
    while n_survivors + n_new > new_capacity:
        new_capacity *= 2

    grown = _alloc_managed_packed_survivors(new_capacity, n_words)
    grown[:n_survivors] = survivors_gpu[:n_survivors]
    return grown

_JACCARD_INTERSECTION_SOURCE = r"""
#define TILE   %d
#define TILE_W %d

extern "C" __global__
void jaccard_intersection(
    const unsigned long long* __restrict__ words_a,
    const unsigned long long* __restrict__ words_b,
    float* __restrict__ intersection,
    const int n_rows_a,
    const int n_rows_b,
    const int n_words)
{
    // +1 padding staggers shared-memory banks across the threadIdx.x reads.
    __shared__ unsigned long long tile_a[TILE][TILE_W + 1];
    __shared__ unsigned long long tile_b[TILE][TILE_W + 1];

    const int row_a = blockIdx.y * TILE + threadIdx.y;
    const int row_b = blockIdx.x * TILE + threadIdx.x;
    const int tid   = threadIdx.y * TILE + threadIdx.x;

    unsigned int accumulated = 0;

    for (int word_base = 0; word_base < n_words; word_base += TILE_W) {
        if (tid < TILE * TILE_W) {
            const int local_row   = tid / TILE_W;
            const int local_word  = tid %% TILE_W;
            const int global_row  = blockIdx.y * TILE + local_row;
            const int global_word = word_base + local_word;
            tile_a[local_row][local_word] =
                (global_row < n_rows_a && global_word < n_words)
                    ? words_a[(size_t)global_row * n_words + global_word] : 0ULL;
        } else if (tid < 2 * TILE * TILE_W) {
            const int offset      = tid - TILE * TILE_W;
            const int local_row   = offset / TILE_W;
            const int local_word  = offset %% TILE_W;
            const int global_row  = blockIdx.x * TILE + local_row;
            const int global_word = word_base + local_word;
            tile_b[local_row][local_word] =
                (global_row < n_rows_b && global_word < n_words)
                    ? words_b[(size_t)global_row * n_words + global_word] : 0ULL;
        }
        __syncthreads();

        for (int local_word = 0; local_word < TILE_W; ++local_word) {
            accumulated += __popcll(tile_a[threadIdx.y][local_word] & tile_b[threadIdx.x][local_word]);
        }
        __syncthreads();
    }

    if (row_a < n_rows_a && row_b < n_rows_b) {
        intersection[(size_t)row_a * n_rows_b + row_b] = (float)accumulated;
    }
}
""" % (JACCARD_TILE, JACCARD_TILE_WORDS)

_JACCARD_INTERSECTION_KERNEL = cp.RawKernel(_JACCARD_INTERSECTION_SOURCE, "jaccard_intersection")


def _pairwise_intersection_gpu(words_a: cp.ndarray, words_b: cp.ndarray) -> cp.ndarray:

    words_a = cp.ascontiguousarray(words_a)
    words_b = cp.ascontiguousarray(words_b)

    n_rows_a, n_words = words_a.shape
    n_rows_b          = words_b.shape[0]

    intersection = cp.empty((n_rows_a, n_rows_b), dtype=cp.float32)
    grid = (
        (n_rows_b + JACCARD_TILE - 1) // JACCARD_TILE,
        (n_rows_a + JACCARD_TILE - 1) // JACCARD_TILE,
    )
    _JACCARD_INTERSECTION_KERNEL(
        grid,
        (JACCARD_TILE, JACCARD_TILE),
        (words_a, words_b, intersection,
         np.int32(n_rows_a), np.int32(n_rows_b), np.int32(n_words)),
    )
    return intersection


_JACCARD_WINDOWED_SOURCE = r"""
#define TILE        %d
#define TILE_W      %d
#define EXIT_CHECK  %d

// Fresh read of a running max that other blocks update with atomicMax.
__device__ __forceinline__ int load_fresh(const int* p)
{
    return *((const volatile int*)p);
}

extern "C" __global__
void jaccard_max_windowed(
    const unsigned long long* __restrict__ cand_words,
    const float* __restrict__ cand_sums,
    const int* __restrict__ cand_order,
    const unsigned long long* __restrict__ surv_words,
    const float* __restrict__ surv_sums,
    const int* __restrict__ surv_order,
    const int* __restrict__ work_group,
    const int* __restrict__ work_start,
    const int* __restrict__ group_end,
    int* max_bits,
    const int n_cand,
    const int n_words,
    const int skip_bits)
{
    // One block = TILE candidates (sorted by |A|) x TILE survivors from that group's |A| window.
    __shared__ unsigned long long tile_a[TILE][TILE_W + 1];
    __shared__ unsigned long long tile_b[TILE][TILE_W + 1];
    __shared__ int rows_a[TILE];
    __shared__ int rows_b[TILE];
    __shared__ int row_done[TILE];

    const int group     = work_group[blockIdx.x];
    const int cand_base = group * TILE;
    const int surv_base = work_start[blockIdx.x];
    const int surv_end  = group_end[group];
    const int tid       = threadIdx.y * TILE + threadIdx.x;

    // A candidate is done once its running max reaches skip_bits, a bound strictly above the
    // threshold: it is already rejected, so its remaining pairs cannot change the decision.
    if (tid < TILE) {
        const int pos = cand_base + tid;
        const int row = (pos < n_cand) ? cand_order[pos] : -1;
        rows_a[tid]   = row;
        row_done[tid] = (row < 0) || (load_fresh(&max_bits[row]) >= skip_bits);
    } else if (tid < 2 * TILE) {
        const int pos = surv_base + (tid - TILE);
        rows_b[tid - TILE] = (pos < surv_end) ? surv_order[pos] : -1;
    }
    if (__syncthreads_and(tid >= TILE || row_done[tid])) return;

    unsigned int accumulated = 0;
    int tile_pass = 0;

    for (int word_base = 0; word_base < n_words; word_base += TILE_W, ++tile_pass) {
        if (tile_pass > 0 && tile_pass %% EXIT_CHECK == 0) {
            if (tid < TILE && !row_done[tid]) {
                row_done[tid] = (load_fresh(&max_bits[rows_a[tid]]) >= skip_bits);
            }
            if (__syncthreads_and(tid >= TILE || row_done[tid])) return;
        }

        if (tid < TILE * TILE_W) {
            const int local_row   = tid / TILE_W;
            const int local_word  = tid %% TILE_W;
            const int global_row  = rows_a[local_row];
            const int global_word = word_base + local_word;
            tile_a[local_row][local_word] =
                (!row_done[local_row] && global_word < n_words)
                    ? cand_words[(size_t)global_row * n_words + global_word] : 0ULL;
        } else if (tid < 2 * TILE * TILE_W) {
            const int offset      = tid - TILE * TILE_W;
            const int local_row   = offset / TILE_W;
            const int local_word  = offset %% TILE_W;
            const int global_row  = rows_b[local_row];
            const int global_word = word_base + local_word;
            tile_b[local_row][local_word] =
                (global_row >= 0 && global_word < n_words)
                    ? surv_words[(size_t)global_row * n_words + global_word] : 0ULL;
        }
        __syncthreads();

        // TILE == 32 == warp size: each warp is one candidate row, so this branch is warp-uniform.
        if (!row_done[threadIdx.y]) {
            for (int local_word = 0; local_word < TILE_W; ++local_word) {
                accumulated += __popcll(tile_a[threadIdx.y][local_word] & tile_b[threadIdx.x][local_word]);
            }
        }
        __syncthreads();
    }

    // Same float32 ops as the former CuPy epilogue: U = (|A| + |B|) - I ; J = (U == 0) ? 1 : I / U
    // A row that went done mid-block holds a partial count, so it publishes nothing.
    const int  row_a = rows_a[threadIdx.y];
    const int  row_b = rows_b[threadIdx.x];
    const bool live  = !row_done[threadIdx.y];
    int best = 0;  // bits of 0.0f; non-negative floats order like their int bit patterns
    if (live && row_b >= 0) {
        const float inter = __uint2float_rn(accumulated);
        const float uni   = __fsub_rn(__fadd_rn(cand_sums[row_a], surv_sums[row_b]), inter);
        const float jac   = (uni == 0.0f) ? 1.0f : __fdiv_rn(inter, uni);
        if (jac > 0.0f) best = __float_as_int(jac);
    }

    // Each warp is one candidate row -> warp max, one atomic per row.
    for (int offset = 16; offset > 0; offset >>= 1) {
        best = max(best, __shfl_xor_sync(0xffffffffu, best, offset));
    }
    if (threadIdx.x == 0 && live && best > 0) {
        atomicMax(&max_bits[row_a], best);
    }
}
""" % (JACCARD_TILE, JACCARD_TILE_WORDS, JACCARD_EXIT_CHECK)

_JACCARD_WINDOWED_KERNEL = cp.RawKernel(_JACCARD_WINDOWED_SOURCE, "jaccard_max_windowed")


def _early_exit_bits(threshold: float) -> int:

    # Smallest float32 above float32(threshold): a running max at or above it is rejected by
    # `max_jaccard <= threshold` whether that comparison runs in float32 or in float64.
    if not (0.0 < threshold < 1.0):
        return int(np.iinfo(np.int32).max)  # J <= 1 never reaches it: early exit disabled
    return int(np.nextafter(np.float32(threshold), np.float32(np.inf)).view(np.int32))


def _survivor_work_items(
    cand_cards_sorted: np.ndarray,
    surv_cards_sorted: np.ndarray,
    threshold: float,
    margin: float,
    prune: bool,
) -> tuple:

    n_cand   = cand_cards_sorted.shape[0]
    n_groups = -(-n_cand // JACCARD_TILE)
    group_first = np.arange(n_groups) * JACCARD_TILE
    group_min   = cand_cards_sorted[group_first]
    group_max   = cand_cards_sorted[np.minimum(group_first + JACCARD_TILE, n_cand) - 1]

    if prune:
        # J(A,B) <= min(|A|,|B|) / max(|A|,|B|): survivors outside this |A| window can never exceed the threshold.
        scale = float(threshold) * (1.0 - margin)
        win_start = np.searchsorted(surv_cards_sorted, np.floor(group_min * scale), side="left")
        win_end   = np.searchsorted(surv_cards_sorted, np.ceil(group_max / scale), side="right")
    else:
        win_start = np.zeros(n_groups, dtype=np.int64)
        win_end   = np.full(n_groups, surv_cards_sorted.shape[0], dtype=np.int64)

    n_tiles    = -(-(win_end - win_start) // JACCARD_TILE)
    n_work     = int(n_tiles.sum())
    work_group = np.repeat(np.arange(n_groups, dtype=np.int32), n_tiles)
    tile_rank  = np.arange(n_work) - np.repeat(np.cumsum(n_tiles) - n_tiles, n_tiles)

    # Visit each group's tiles center-out (closest |A| first, alternating sides): near-duplicates are found
    # early, so later tiles skip candidates already rejected. Only the visiting order changes, never the tile set.
    center_pos  = np.searchsorted(surv_cards_sorted, (group_min + group_max) / 2.0, side="left")
    center_tile = np.clip((center_pos - win_start) // JACCARD_TILE, 0, np.maximum(n_tiles - 1, 0))
    n_left      = center_tile
    n_right     = np.maximum(n_tiles - 1 - center_tile, 0)
    n_both      = np.repeat(np.minimum(n_left, n_right), n_tiles)
    alternating = tile_rank <= 2 * n_both
    step        = np.where(alternating, (tile_rank + 1) // 2, tile_rank - n_both)
    go_left     = np.where(alternating, tile_rank % 2 == 1, np.repeat(n_left > n_right, n_tiles))
    tile_index  = np.repeat(center_tile, n_tiles) + np.where(go_left, -step, step)
    work_start  = (np.repeat(win_start, n_tiles) + tile_index * JACCARD_TILE).astype(np.int32)

    return work_group, work_start, win_end.astype(np.int32)


def _max_jaccard_vs_survivors_gpu(
    batch_gpu: cp.ndarray,
    batch_sums: cp.ndarray,
    batch_cards: np.ndarray,
    survivors_gpu: cp.ndarray,
    survivor_sums_gpu: cp.ndarray,
    surv_cards_sorted: np.ndarray,
    surv_rows_sorted: np.ndarray,
    threshold: float,
    margin: float,
    prune: bool,
) -> tuple:

    n_batch_cols, n_words = batch_gpu.shape
    cand_order = np.argsort(batch_cards, kind="stable").astype(np.int32)
    work_group, work_start, group_end = _survivor_work_items(
        batch_cards[cand_order], surv_cards_sorted, threshold, margin, prune
    )

    max_bits = cp.zeros(n_batch_cols, dtype=cp.int32)  # float32 bit patterns, all >= 0.0f
    n_work   = work_group.shape[0]
    if n_work > 0:
        _JACCARD_WINDOWED_KERNEL(
            (n_work,),
            (JACCARD_TILE, JACCARD_TILE),
            (batch_gpu, batch_sums, cp.asarray(cand_order),
             survivors_gpu, survivor_sums_gpu, cp.asarray(surv_rows_sorted),
             cp.asarray(work_group), cp.asarray(work_start), cp.asarray(group_end),
             max_bits, np.int32(n_batch_cols), np.int32(n_words), np.int32(_early_exit_bits(threshold))),
        )
    return max_bits.view(cp.float32), n_work * JACCARD_TILE * JACCARD_TILE


def _greedy_keep_within(intra_jaccard: np.ndarray, threshold: float) -> np.ndarray:

    # Same decision as `intra_jaccard[i, kept].max() > threshold`, compared in the dtype NumPy uses
    # for that scalar comparison (float64 on NumPy 1.x, the array dtype on NumPy 2.x).
    cmp_dtype = np.result_type(intra_jaccard.dtype.type(0), threshold)
    conflict  = np.ascontiguousarray((intra_jaccard.astype(cmp_dtype) > cmp_dtype.type(threshold)).T)

    n_rows      = intra_jaccard.shape[0]
    keep_within = np.zeros(n_rows, dtype=bool)
    blocked     = np.zeros(n_rows, dtype=bool)  # blocked[i]: some kept j < i has J(i, j) > threshold
    for i in range(n_rows):
        if blocked[i]:
            continue
        keep_within[i] = True
        blocked |= conflict[i]
    return keep_within


def _jaccard_filter_side_gpu(
    spec_words_gpu: cp.ndarray,
    side_spec_idx: np.ndarray,
    threshold: float,
    batch_size: int,
    timeframe: str = "",
) -> np.ndarray:

    n_cols  = side_spec_idx.shape[0]
    n_words = spec_words_gpu.shape[1]
    if n_cols == 0:
        return np.array([], dtype=np.int64)

    # |A| is an exact integer count and its float32 cast is exact up to 2^24, so the former max_rel_err is always 0.
    prune  = bool(0.0 < threshold < 1.0)
    margin = JACCARD_PRUNE_MARGIN

    survivors_gpu     = _alloc_managed_packed_survivors(GPU_INITIAL_CAPACITY, n_words)
    survivor_sums_gpu = cp.zeros(GPU_INITIAL_CAPACITY, dtype=cp.float32)
    n_survivors = 0
    survivor_chunks = []
    surv_cards_sorted = np.empty(0, dtype=np.int64)  # exact |A| of survivors, ascending
    surv_rows_sorted  = np.empty(0, dtype=np.int32)  # matching row in survivors_gpu
    pairs_computed = 0
    pairs_total    = 0

    n_batches = int(np.ceil(n_cols / batch_size))
    desc = f"JACCARD GPU     {timeframe}"

    for batch_start in tqdm(range(0, n_cols, batch_size), desc=desc, total=n_batches, dynamic_ncols=True):
        batch_end    = min(batch_start + batch_size, n_cols)
        batch_gpu, batch_cards_gpu = _gather_rule_words_gpu(spec_words_gpu, side_spec_idx[batch_start:batch_end])
        batch_sums   = batch_cards_gpu.astype(cp.float32)
        batch_cards  = cp.asnumpy(batch_cards_gpu).astype(np.int64)
        n_batch_cols = batch_gpu.shape[0]

        keep_mask = cp.ones(n_batch_cols, dtype=cp.bool_)

        if n_survivors > 0:
            max_jaccard, n_pairs = _max_jaccard_vs_survivors_gpu(
                batch_gpu, batch_sums, batch_cards, survivors_gpu, survivor_sums_gpu,
                surv_cards_sorted, surv_rows_sorted, threshold, margin, prune,
            )
            pairs_computed += n_pairs
            pairs_total    += n_batch_cols * n_survivors
            keep_mask = max_jaccard <= threshold

        accepted_local = cp.where(keep_mask)[0]
        n_accepted = int(accepted_local.shape[0])

        if n_accepted > 0:
            cand_gpu  = batch_gpu[accepted_local]
            cand_sums = batch_sums[accepted_local]
            intra_intersection = _pairwise_intersection_gpu(cand_gpu, cand_gpu)

            intra_union = cand_sums[:, None] + cand_sums[None, :] - intra_intersection
            intra_empty_pair_mask = intra_union == 0  # both sets empty -> identical, not disjoint
            intra_safe_union = cp.where(intra_empty_pair_mask, 1.0, intra_union)
            intra_jaccard = cp.asnumpy(cp.where(intra_empty_pair_mask, 1.0, intra_intersection / intra_safe_union))

            keep_within = _greedy_keep_within(intra_jaccard, threshold)
            final_local = cp.asnumpy(accepted_local)[keep_within]
        else:
            final_local = np.array([], dtype=np.int64)

        n_final = final_local.shape[0]
        survivors_gpu = _ensure_packed_survivor_capacity(survivors_gpu, n_survivors, n_final, n_words)
        if survivor_sums_gpu.shape[0] < n_survivors + n_final:
            grown_sums = cp.zeros(survivors_gpu.shape[0], dtype=cp.float32)
            grown_sums[:n_survivors] = survivor_sums_gpu[:n_survivors]
            survivor_sums_gpu = grown_sums

        if n_final > 0:
            survivors_gpu[n_survivors:n_survivors + n_final] = batch_gpu[cp.asarray(final_local)]
            survivor_sums_gpu[n_survivors:n_survivors + n_final] = batch_sums[cp.asarray(final_local)]

            new_cards = batch_cards[final_local]
            new_order = np.argsort(new_cards, kind="stable")
            insert_at = np.searchsorted(surv_cards_sorted, new_cards[new_order], side="right")
            surv_cards_sorted = np.insert(surv_cards_sorted, insert_at, new_cards[new_order])
            surv_rows_sorted  = np.insert(surv_rows_sorted, insert_at, (n_survivors + new_order).astype(np.int32))
        n_survivors += n_final

        survivor_chunks.append(batch_start + final_local)

    del survivors_gpu, survivor_sums_gpu
    cp.get_default_memory_pool().free_all_blocks()

    if pairs_total:
        logger.debug(f"JACCARD PRUNE   {timeframe}: {pairs_computed / pairs_total:.1%} of candidate-survivor pairs scheduled (prune={prune})")

    return np.concatenate(survivor_chunks) if survivor_chunks else np.array([], dtype=np.int64)


def pipe_signal_cleaning_jaccard(
    rules: list,
    ohlcv_arr: dict,
    timeframe: str = "",
    threshold: float = JACCARD_SIMILARITY_TH,
    batch_size: int = JACCARD_BATCH_SIZE,
    n_jobs: int = SIGNAL_MASK_N_JOBS,
) -> list:

    _all_ts_dbg = np.concatenate([arr["ts"] for arr in ohlcv_arr.values()])
    logger.debug(f"JACCARD INPUT   {timeframe}: date range [{_all_ts_dbg.min()} .. {_all_ts_dbg.max()}] over {len(ohlcv_arr)} symbol(s)")

    unique_specs, index_by_identity = _collect_unique_specs(rules)
    spec_words, _ = _build_spec_word_table(unique_specs, ohlcv_arr, n_jobs, timeframe)
    spec_idx, sides = _rule_spec_index(rules, index_by_identity)
    spec_words_gpu = cp.asarray(spec_words)
    del spec_words

    kept_positions = []
    for side in np.unique(sides):
        side_positions = np.where(sides == side)[0]
        kept_local = _jaccard_filter_side_gpu(spec_words_gpu, spec_idx[side_positions], threshold, batch_size, timeframe)
        kept_positions.append(side_positions[kept_local])

    del spec_words_gpu
    cp.get_default_memory_pool().free_all_blocks()

    kept_positions = np.sort(np.concatenate(kept_positions)) if kept_positions else np.array([], dtype=np.int64)

    n_rules_total = len(rules)
    n_kept        = kept_positions.shape[0]

    logger.info(f"\n{'JACCARD FILTER':<16}{timeframe}: {n_kept / n_rules_total:.0%} │ {format(n_kept, ',').replace(',', '.')} / {format(n_rules_total, ',').replace(',', '.')} (th={threshold})")

    return [rules[i] for i in kept_positions]