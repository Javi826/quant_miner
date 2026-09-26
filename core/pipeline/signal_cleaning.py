# core/pipeline/signal_cleaning.py
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
JACCARD_SIMILARITY_TH  = 0.80
#------------------------------------------------------------------------------


SIGNAL_MASK_N_JOBS     = -1
SIGNAL_MASK_CHUNK_SIZE = None   # None -> auto-sized from n_jobs and rule count
JACCARD_TILE           = 32     # output tile side; must stay in sync with the CUDA kernel
JACCARD_TILE_WORDS     = 8      # uint64 words staged in shared memory per tile pass
JACCARD_POPCOUNT_ROWS  = 4096   # host-side popcount chunk, caps peak memory
JACCARD_PRUNE_MARGIN   = 1e-4   # safety margin on the |A| window for pruning (only widens it)
JACCARD_PRUNE_MAX_ERR  = 1e-3   # max float32 rel. error of |A| allowed to enable pruning
DECORRELATE_THRESHOLD  = 0.5
DECORRELATE_BATCH_SIZE = 1000
GPU_INITIAL_CAPACITY   = 20_000 
GPU_SURVIVOR_CHUNK     = 25_000 # initial survivors buffer capacity, doubled on overflow
RANDOM_SEED            = 42
METRIC_LABEL_WIDTH     = 28     # fixed label width so all metric-block prints align


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

def build_signal_mask_keys(
    all_rules: list,
    ohlcv_arr: dict,
    n_jobs: int = SIGNAL_MASK_N_JOBS,
    chunk_size: int = SIGNAL_MASK_CHUNK_SIZE,  # kept for API compatibility
    timeframe: str = "",
) -> list:

    unique_specs, index_by_identity = _collect_unique_specs(all_rules)
    words, n_bytes = _build_spec_word_table(unique_specs, ohlcv_arr, n_jobs, timeframe)

    all_keys = []
    for rule in all_rules:
        spec_rows = [index_by_identity[_spec_identity(spec)] for spec in rule["specs"]]

        combined = words[spec_rows[0]]
        for row_idx in spec_rows[1:]:
            combined = combined & words[row_idx]

        all_keys.append((rule["side"], combined.view(np.uint8)[:n_bytes].tobytes()))

    return all_keys

# =============================================================================
# JACCARD SIMILARITY FILTER (GPU, bit-packed) — standalone, near-duplicate
# =============================================================================

_POPCOUNT_TABLE_NP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

def _packed_signal_matrix(rules: list, ohlcv_arr: dict, n_jobs: int, timeframe: str = "") -> tuple:

    signal_keys = build_signal_mask_keys(rules, ohlcv_arr, n_jobs=n_jobs, timeframe=timeframe)
    sides       = np.array([side for side, _ in signal_keys])

    n_rules   = len(signal_keys)
    n_bytes   = len(signal_keys[0][1])
    word_size = np.dtype(np.uint64).itemsize
    n_padded  = -(-n_bytes // word_size) * word_size  # round up to a whole word

    packed = np.zeros((n_rules, n_padded), dtype=np.uint8)
    packed[:, :n_bytes] = np.frombuffer(
        b"".join(packed_bytes for _, packed_bytes in signal_keys), dtype=np.uint8
    ).reshape(n_rules, n_bytes)

    return packed.view(np.uint64), sides

def _popcount_packed(words: np.ndarray) -> np.ndarray:

    bytes_view = words.view(np.uint8)
    n_rows     = bytes_view.shape[0]
    cardinality = np.empty(n_rows, dtype=np.float32)

    for start in range(0, n_rows, JACCARD_POPCOUNT_ROWS):
        end = min(start + JACCARD_POPCOUNT_ROWS, n_rows)
        cardinality[start:end] = _POPCOUNT_TABLE_NP[bytes_view[start:end]].sum(axis=1, dtype=np.float32)

    return cardinality


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


def _popcount_packed_with_exact(words: np.ndarray) -> tuple:

    bytes_view  = words.view(np.uint8)
    n_rows      = bytes_view.shape[0]
    cardinality = np.empty(n_rows, dtype=np.float32)
    exact       = np.empty(n_rows, dtype=np.int64)

    for start in range(0, n_rows, JACCARD_POPCOUNT_ROWS):
        end = min(start + JACCARD_POPCOUNT_ROWS, n_rows)
        counts = _POPCOUNT_TABLE_NP[bytes_view[start:end]]
        cardinality[start:end] = counts.sum(axis=1, dtype=np.float32)  # same float32 values as _popcount_packed
        exact[start:end]       = counts.sum(axis=1, dtype=np.int64)

    return cardinality, exact


_JACCARD_WINDOWED_SOURCE = r"""
#define TILE   %d
#define TILE_W %d

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
    int* __restrict__ max_bits,
    const int n_cand,
    const int n_words)
{
    // One block = TILE candidates (sorted by |A|) x TILE survivors from that group's |A| window.
    __shared__ unsigned long long tile_a[TILE][TILE_W + 1];
    __shared__ unsigned long long tile_b[TILE][TILE_W + 1];
    __shared__ int rows_a[TILE];
    __shared__ int rows_b[TILE];

    const int group     = work_group[blockIdx.x];
    const int cand_base = group * TILE;
    const int surv_base = work_start[blockIdx.x];
    const int surv_end  = group_end[group];
    const int tid       = threadIdx.y * TILE + threadIdx.x;

    if (tid < TILE) {
        const int pos = cand_base + tid;
        rows_a[tid] = (pos < n_cand) ? cand_order[pos] : -1;
    } else if (tid < 2 * TILE) {
        const int pos = surv_base + (tid - TILE);
        rows_b[tid - TILE] = (pos < surv_end) ? surv_order[pos] : -1;
    }
    __syncthreads();

    unsigned int accumulated = 0;

    for (int word_base = 0; word_base < n_words; word_base += TILE_W) {
        if (tid < TILE * TILE_W) {
            const int local_row   = tid / TILE_W;
            const int local_word  = tid %% TILE_W;
            const int global_row  = rows_a[local_row];
            const int global_word = word_base + local_word;
            tile_a[local_row][local_word] =
                (global_row >= 0 && global_word < n_words)
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

        for (int local_word = 0; local_word < TILE_W; ++local_word) {
            accumulated += __popcll(tile_a[threadIdx.y][local_word] & tile_b[threadIdx.x][local_word]);
        }
        __syncthreads();
    }

    // Same float32 ops as the former CuPy epilogue: U = (|A| + |B|) - I ; J = (U == 0) ? 1 : I / U
    const int row_a = rows_a[threadIdx.y];
    const int row_b = rows_b[threadIdx.x];
    int best = 0;  // bits of 0.0f; non-negative floats order like their int bit patterns
    if (row_a >= 0 && row_b >= 0) {
        const float inter = __uint2float_rn(accumulated);
        const float uni   = __fsub_rn(__fadd_rn(cand_sums[row_a], surv_sums[row_b]), inter);
        const float jac   = (uni == 0.0f) ? 1.0f : __fdiv_rn(inter, uni);
        if (jac > 0.0f) best = __float_as_int(jac);
    }

    // TILE == 32 == warp size: each warp is one candidate row -> warp max, one atomic per row.
    for (int offset = 16; offset > 0; offset >>= 1) {
        best = max(best, __shfl_xor_sync(0xffffffffu, best, offset));
    }
    if (threadIdx.x == 0 && best > 0) {
        atomicMax(&max_bits[row_a], best);
    }
}
""" % (JACCARD_TILE, JACCARD_TILE_WORDS)

_JACCARD_WINDOWED_KERNEL = cp.RawKernel(_JACCARD_WINDOWED_SOURCE, "jaccard_max_windowed")


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
    work_start = (np.repeat(win_start, n_tiles) + tile_rank * JACCARD_TILE).astype(np.int32)
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
             max_bits, np.int32(n_batch_cols), np.int32(n_words)),
        )
    return max_bits.view(cp.float32), n_work * JACCARD_TILE * JACCARD_TILE


def _jaccard_filter_side_gpu(
    packed_matrix: np.ndarray,
    threshold: float,
    batch_size: int,
    survivor_chunk_size: int,  # kept for API compatibility: no intersection matrix is materialized now
    timeframe: str = "",
) -> np.ndarray:

    n_cols, n_words = packed_matrix.shape
    if n_cols == 0:
        return np.array([], dtype=np.int64)

    col_sums, exact_cards = _popcount_packed_with_exact(packed_matrix)  # |A| per rule: float32 (as before) + exact int64

    # Pruning is exact only while the float32 |A| used by the Jaccard formula stays close to the exact count.
    max_rel_err = float(np.max(np.abs(col_sums.astype(np.float64) - exact_cards) / np.maximum(exact_cards, 1)))
    prune  = bool(0.0 < threshold < 1.0) and max_rel_err <= JACCARD_PRUNE_MAX_ERR
    margin = JACCARD_PRUNE_MARGIN + 4.0 * max_rel_err

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
        batch_gpu    = cp.asarray(packed_matrix[batch_start:batch_end])
        batch_sums   = cp.asarray(col_sums[batch_start:batch_end])
        batch_cards  = exact_cards[batch_start:batch_end]
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

            keep_within = np.zeros(n_accepted, dtype=bool)
            accepted_within = []
            for i in range(n_accepted):
                if accepted_within and intra_jaccard[i, accepted_within].max() > threshold:
                    continue
                keep_within[i] = True
                accepted_within.append(i)

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

        del batch_gpu
        cp.get_default_memory_pool().free_all_blocks()

    del survivors_gpu, survivor_sums_gpu
    cp.get_default_memory_pool().free_all_blocks()

    if pairs_total:
        logger.debug(f"JACCARD PRUNE   {timeframe}: {pairs_computed / pairs_total:.1%} of candidate-survivor pairs computed (prune={prune})")

    return np.concatenate(survivor_chunks) if survivor_chunks else np.array([], dtype=np.int64)


def pipe_signal_cleaning_jaccard(
    rules: list,
    ohlcv_arr: dict,
    timeframe: str = "",
    threshold: float = JACCARD_SIMILARITY_TH,
    batch_size: int = DECORRELATE_BATCH_SIZE,
    survivor_chunk_size: int = GPU_SURVIVOR_CHUNK,
    n_jobs: int = SIGNAL_MASK_N_JOBS,
) -> list:

    _all_ts_dbg = np.concatenate([arr["ts"] for arr in ohlcv_arr.values()])
    logger.debug(f"JACCARD INPUT   {timeframe}: date range [{_all_ts_dbg.min()} .. {_all_ts_dbg.max()}] over {len(ohlcv_arr)} symbol(s)")

    packed_matrix, sides = _packed_signal_matrix(rules, ohlcv_arr, n_jobs, timeframe)
    kept_positions = []
    for side in np.unique(sides):
        side_positions = np.where(sides == side)[0]
        side_matrix    = packed_matrix[side_positions]
        kept_local = _jaccard_filter_side_gpu(side_matrix, threshold, batch_size, survivor_chunk_size, timeframe)
        kept_positions.append(side_positions[kept_local])

    kept_positions = np.sort(np.concatenate(kept_positions)) if kept_positions else np.array([], dtype=np.int64)

    n_rules_total = len(rules)
    n_kept        = kept_positions.shape[0]

    logger.info(f"\n{'JACCARD FILTER':<16}{timeframe}: {n_kept / n_rules_total:.0%} │ {format(n_kept, ',').replace(',', '.')} / {format(n_rules_total, ',').replace(',', '.')} (th={threshold})")

    return [rules[i] for i in kept_positions]

# =============================================================================
# COLUMN DECORRELATION (GPU) — post-backtest, pre-StepM redundancy filter
# =============================================================================
def _normalize_columns_for_correlation(matrix_arr: np.ndarray) -> np.ndarray:

    matrix_norm = matrix_arr.astype(np.float32, copy=True)
    matrix_norm -= matrix_norm.mean(axis=0, keepdims=True)
    col_norms = np.linalg.norm(matrix_norm, axis=0)
    col_norms[col_norms == 0] = 1.0  # guard degenerate constant columns
    matrix_norm /= col_norms[None, :]
    return matrix_norm


def _sequential_decorrelate_within_batch(candidate_norm: np.ndarray, threshold: float) -> np.ndarray:

    n_candidates = candidate_norm.shape[1]
    keep_mask = np.zeros(n_candidates, dtype=bool)
    if n_candidates == 0:
        return keep_mask

    intra_corr = candidate_norm.T @ candidate_norm  # (n_candidates, n_candidates)

    accepted_idx = []
    for i in range(n_candidates):
        if accepted_idx and intra_corr[i, accepted_idx].max() > threshold:
            continue
        keep_mask[i] = True
        accepted_idx.append(i)

    return keep_mask

_MANAGED_POOL = cp.cuda.MemoryPool(cp.cuda.malloc_managed)

def _alloc_managed_survivors(n_days: int, n_cols: int) -> cp.ndarray:

    n_bytes = n_days * n_cols * cp.dtype(cp.float32).itemsize
    mem = _MANAGED_POOL.malloc(n_bytes)
    return cp.ndarray((n_days, n_cols), dtype=cp.float32, memptr=mem)


def _ensure_survivor_capacity(survivors_gpu: cp.ndarray, n_survivors: int, n_new: int, n_days: int) -> cp.ndarray:

    capacity = survivors_gpu.shape[1]
    if n_survivors + n_new <= capacity:
        return survivors_gpu

    new_capacity = capacity
    while n_survivors + n_new > new_capacity:
        new_capacity *= 2

    grown = _alloc_managed_survivors(n_days, new_capacity)
    grown[:, :n_survivors] = survivors_gpu[:, :n_survivors]
    return grown


def _max_corr_against_survivors_gpu(
    batch_gpu: cp.ndarray,
    survivors_gpu: cp.ndarray,
    n_survivors: int,
    chunk_size: int,
) -> cp.ndarray:

    n_batch_cols = batch_gpu.shape[1]
    max_corr = cp.zeros(n_batch_cols, dtype=cp.float32)

    for start in range(0, n_survivors, chunk_size):
        end = min(start + chunk_size, n_survivors)
        corr_block = batch_gpu.T @ survivors_gpu[:, start:end]
        cp.maximum(max_corr, corr_block.max(axis=1), out=max_corr)

    return max_corr