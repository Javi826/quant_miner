# core/pipeline/signal_cleaning.py
import logging
import numpy as np
import cupy as cp
from joblib import Parallel, delayed
from tqdm import tqdm
from signals.indicators_bank import ConditionBank
from signals.signal_builder import build_signal_fn

logger = logging.getLogger("BOT_batch.pipeline.signal_cleaning")

# =============================================================================
# CONFIG
# =============================================================================
JACCARD_SIMILARITY_TH  = 0.80


SIGNAL_MASK_N_JOBS     = -1
SIGNAL_MASK_CHUNK_SIZE = None  # None -> auto-sized from n_jobs and rule count
JACCARD_TILE           = 32  # output tile side; must stay in sync with the CUDA kernel
JACCARD_TILE_WORDS     = 8   # uint64 words staged in shared memory per tile pass
JACCARD_POPCOUNT_ROWS  = 4096  # host-side popcount chunk, caps peak memory
DECORRELATE_THRESHOLD  = 0.5
DECORRELATE_BATCH_SIZE = 1000
GPU_INITIAL_CAPACITY   = 20_000 
GPU_SURVIVOR_CHUNK     = 25_000 # initial survivors buffer capacity, doubled on overflow
RANDOM_SEED            = 42
METRIC_LABEL_WIDTH     = 28  # fixed label width so all metric-block prints align


def _spec_identity(spec: dict) -> tuple:
    """Hashable identity of a condition spec, used to deduplicate specs across rules."""
    return (spec["key"], spec["op"], spec["threshold"])


def _collect_unique_specs(all_rules: list) -> tuple:
    """Return (unique_specs, index_by_identity) preserving first-seen order."""
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
    """Evaluate every unique spec on a single symbol through the production
    signal path, so the 1-bar shift stays owned by signal_builder."""
    bank = ConditionBank(arr)
    rows = np.empty((len(unique_specs), bank.n), dtype=bool)
    for i, spec in enumerate(unique_specs):
        signal = build_signal_fn([spec], "long")(arr, live_trading=False, bank=bank)
        rows[i] = signal.astype(bool)
    return rows


def _build_spec_word_table(unique_specs: list, ohlcv_arr: dict, n_jobs: int, timeframe: str = "") -> tuple:
    """Bit-pack the per-spec signal rows into uint64 words for fast AND-reduction.
    Returns (words, n_bytes) where n_bytes is the unpadded packed row length."""
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


def _jaccard_filter_side_gpu(
    packed_matrix: np.ndarray,
    threshold: float,
    batch_size: int,
    survivor_chunk_size: int,
    timeframe: str = "",
) -> np.ndarray:

    n_cols, n_words = packed_matrix.shape
    if n_cols == 0:
        return np.array([], dtype=np.int64)

    col_sums = _popcount_packed(packed_matrix)  # |A| per rule, CPU side

    survivors_gpu     = _alloc_managed_packed_survivors(GPU_INITIAL_CAPACITY, n_words)
    survivor_sums_gpu = cp.zeros(GPU_INITIAL_CAPACITY, dtype=cp.float32)
    n_survivors = 0
    survivor_chunks = []

    n_batches = int(np.ceil(n_cols / batch_size))
    desc = f"JACCARD GPU     {timeframe}"

    for batch_start in tqdm(range(0, n_cols, batch_size), desc=desc, total=n_batches, dynamic_ncols=True):
        batch_end    = min(batch_start + batch_size, n_cols)
        batch_gpu    = cp.asarray(packed_matrix[batch_start:batch_end])
        batch_sums   = cp.asarray(col_sums[batch_start:batch_end])
        n_batch_cols = batch_gpu.shape[0]

        keep_mask = cp.ones(n_batch_cols, dtype=cp.bool_)

        if n_survivors > 0:
            max_jaccard = cp.zeros(n_batch_cols, dtype=cp.float32)
            for start in range(0, n_survivors, survivor_chunk_size):
                end = min(start + survivor_chunk_size, n_survivors)
                intersection = _pairwise_intersection_gpu(batch_gpu, survivors_gpu[start:end])

                union = batch_sums[:, None] + survivor_sums_gpu[start:end][None, :] - intersection
                empty_pair_mask = union == 0  # both sets empty -> identical, not disjoint
                safe_union = cp.where(empty_pair_mask, 1.0, union)
                jaccard_block = cp.where(empty_pair_mask, 1.0, intersection / safe_union)
                cp.maximum(max_jaccard, jaccard_block.max(axis=1), out=max_jaccard)
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
        n_survivors += n_final

        survivor_chunks.append(batch_start + final_local)

        del batch_gpu
        cp.get_default_memory_pool().free_all_blocks()

    del survivors_gpu, survivor_sums_gpu
    cp.get_default_memory_pool().free_all_blocks()

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
    n_dropped     = n_rules_total - n_kept

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

