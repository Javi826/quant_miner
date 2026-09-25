# core/pipeline/stepM.py  (GPU v3 — streaming top-M)
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cupy as cp
import cupyx
from tqdm import tqdm
from setup.config_core import settings
from utils.paralelization import compact_columns_inplace
from utils.reporting import print_stepm_matrix_debug, print_stepm_real_variance_filter_debug, print_stepm_block_starts_debug
from utils.reporting import print_stepm_bootstrap_replicas_debug, print_stepm_se_filter_debug, print_stepm_studentization_debug
from utils.reporting import print_stepm_pvalue_quantile_equivalence_debug, print_stepm_monotonicity_debug, print_stepm_brc_equivalence_debug
logger = logging.getLogger("BOT_batch.pipeline.stepM_is")

# =============================================================================
# STATISTICAL TEST CONFIG + STEPDOWN / K-FWE CONFIG -ROmano Wolf
# =============================================================================
STEPM_ALPHA         = 0.10      # significance level used inside the Romano-Wolf stepdown search
FDP_GAMMA           = 0.10
# =============================================================================
# STATISTICAL TEST
# =============================================================================
FDP_K_MAX            = 8192           # upper bound of the geometric k sweep: 1, 2, 4, ... FDP_K_MAX
WHITE_PVALUE_TH      = STEPM_ALPHA
WHITE_N_BOOTSTRAP    = 1000
WHITE_BLOCK_SIZE     = 10            # fixed block length — mirrors montecarlo.py BLOCK_SIZE
STEPM_USE_SPA        = True
STEPM_MAX_ITERATIONS = 500           # safety cap on stepdown iterations
CROSS_SECTIONAL_PERCENTILES = np.array([50, 90, 95, 96, 97, 98, 99, 99.9, 99.99, 100])
# =============================================================================
# MEMORY-CHUNKING CONFIG — bounds peak RAM without changing any result
# =============================================================================
COLUMN_CHUNK_SIZE    = 5000     # columns processed per chunk for chunked reductions/compaction over dense matrices
PARTITION_ROW_CHUNK  = 50       # bootstrap replicas processed per np.partition call in the stepdown
BOOTSTRAP_CHUNK_SIZE = 8192     # columns processed per GEMM call in the bootstrap moment computation
RANDOM_SEED          = 42
FDP_TOPM_WORKERS     = max(1, min(os.cpu_count() or 1, 16))  # threads for the suffix k-th queries on the top-M index
# =============================================================================
# GPU CONFIG (CuPy, streaming top-M) — VRAM is bounded by n_bootstrap x (M + staging), not by N
# =============================================================================
GPU_TOPM_STAGING_COLS    = 65536    # candidate keys staged per row between two top-M merges (>= BOOTSTRAP_CHUNK_SIZE)
GPU_TOPM_BUDGET_MB       = 2048     # VRAM budget for the row-blocked sort that merges the top-M
GPU_SORT_BYTES_PER_ELEM  = 32       # sort workspace estimate per uint64 key
GPU_CHUNK_BYTES_PER_ELEM = 96       # peak temporaries per (replica, column) element of one bootstrap chunk
GPU_VRAM_MARGIN_MB       = 512      # VRAM kept free on top of the estimated streaming footprint

if os.environ.get("CUPY_TF32", "0") not in ("", "0"):
    raise RuntimeError("CUPY_TF32 is set: GEMMs would run in TF32 (lower precision). Unset it.")
if GPU_TOPM_STAGING_COLS < BOOTSTRAP_CHUNK_SIZE:
    raise ValueError("GPU_TOPM_STAGING_COLS must be >= BOOTSTRAP_CHUNK_SIZE.")

_SIGN32  = np.uint32(0x80000000)
_MAG32   = np.uint32(0x7FFFFFFF)
_SHIFT32 = np.uint64(32)
_LOW32   = np.uint64(0xFFFFFFFF)
_ZERO64  = np.uint64(0)

# =============================================================================
# CHUNKED REDUCTIONS — column-wise mean/std without materializing a full-size
# =============================================================================
def _mean_std_by_column_chunks(arr: np.ndarray, ddof: int = 0, chunk_size: int = COLUMN_CHUNK_SIZE):
    n_cols = arr.shape[1]
    means = np.empty(n_cols, dtype=np.float64)
    stds  = np.empty(n_cols, dtype=np.float64)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = arr[:, start:end]
        means[start:end] = chunk.mean(axis=0, dtype=np.float64)
        stds[start:end]  = chunk.std(axis=0, ddof=ddof, dtype=np.float64)
    return means, stds


def _std_by_column_chunks(arr: np.ndarray, ddof: int = 0, chunk_size: int = COLUMN_CHUNK_SIZE) -> np.ndarray:
    n_cols = arr.shape[1]
    stds = np.empty(n_cols, dtype=np.float64)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        stds[start:end] = arr[:, start:end].std(axis=0, ddof=ddof, dtype=np.float64)
    return stds

# =============================================================================
# STATISTIC — annualized Sharpe per trial column, vectorized over bootstrap replicas
# =============================================================================
def _sharpe_per_column(matrix_arr: np.ndarray) -> np.ndarray:

    means, stds = _mean_std_by_column_chunks(matrix_arr, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (means / stds) * np.sqrt(settings.DAYS_PER_YEAR)

    return np.where(stds > 0, sharpe, -np.inf)

# =============================================================================
# MOVING BLOCK BOOTSTRAP — WEIGHT-MATRIX (GEMM) FORMULATION
# =============================================================================
def _generate_block_starts(n_obs: int, block_size: int, n_replicas: int, rng: np.random.Generator):

    n_blocks_needed = int(np.ceil(n_obs / block_size))
    n_block_starts  = n_obs - block_size + 1

    starts = rng.integers(0, n_block_starts, size=(n_replicas, n_blocks_needed), dtype=np.int32)

    len_last = n_obs - (n_blocks_needed - 1) * block_size
    starts_full = starts[:, :-1] if n_blocks_needed > 1 else starts[:, :0]
    starts_last = starts[:, -1]

    return starts_full, starts_last, len_last, n_blocks_needed

def _build_bootstrap_weight_matrix(
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    n_obs: int,
    n_replicas: int,
) -> np.ndarray:

    diff = np.zeros((n_replicas, n_obs + 1), dtype=np.int32)
    row_idx = np.arange(n_replicas)

    n_blocks_full = starts_full.shape[1]
    if n_blocks_full > 0:
        rows_full   = np.repeat(row_idx, n_blocks_full)
        starts_flat = starts_full.ravel()
        ends_flat   = starts_flat + block_size
        np.add.at(diff, (rows_full, starts_flat), 1)
        np.add.at(diff, (rows_full, ends_flat), -1)

    ends_last = starts_last + len_last
    np.add.at(diff, (row_idx, starts_last), 1)
    np.add.at(diff, (row_idx, ends_last), -1)

    weights = np.cumsum(diff[:, :n_obs], axis=1)
    return weights.astype(np.float32)

def _bootstrap_moments_chunk(
    weight_matrix: np.ndarray,
    batch_values: np.ndarray,
    real_sharpe_batch: np.ndarray,
    n_obs: int,
) -> tuple:

    total_sum   = weight_matrix @ batch_values
    total_sumsq = weight_matrix @ (batch_values * batch_values)

    means = total_sum / n_obs
    var   = (total_sumsq - n_obs * means * means) / (n_obs - 1)
    np.maximum(var, 0.0, out=var)  # guard tiny negative fp error before sqrt
    stds = np.sqrt(var)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_sharpe = (means / stds) * np.sqrt(settings.DAYS_PER_YEAR)
    boot_sharpe = np.where(stds > 0, boot_sharpe, -np.inf)

    deviations_chunk = boot_sharpe - real_sharpe_batch[None, :]
    sigma_chunk       = deviations_chunk.std(axis=0, ddof=1)

    return deviations_chunk, sigma_chunk

# Hansen (2005) SPA_c: recenter clearly-losing columns (z_stat below a slowly-growing
# threshold) to 0 instead of their own negative mean, before computing the bootstrap null
def apply_spa_recentering(studentized_deviations: np.ndarray, z_stat: np.ndarray, n_obs: int) -> tuple:
    threshold = -np.sqrt(2.0 * np.log(np.log(max(n_obs, 3))))
    bad_mask  = z_stat < threshold
    if bad_mask.any():
        studentized_deviations[:, bad_mask] += z_stat[bad_mask][None, :]
    return studentized_deviations, bad_mask, threshold

# =============================================================================
# PARALLEL REAL SHARPE (CPU, bit-exact): identical numpy calls on the same
# =============================================================================
STEPM_WORKERS        = max(1, min(os.cpu_count() or 1, 32))  # threads for the column-parallel numpy work


def _sharpe_per_column_parallel(matrix_arr: np.ndarray, ex) -> np.ndarray:
    # identical calls to _mean_std_by_column_chunks (same COLUMN_CHUNK_SIZE slices), run concurrently
    n_cols = matrix_arr.shape[1]
    means = np.empty(n_cols, dtype=np.float64)
    stds  = np.empty(n_cols, dtype=np.float64)

    def job(start):
        end = min(start + COLUMN_CHUNK_SIZE, n_cols)
        chunk = matrix_arr[:, start:end]
        means[start:end] = chunk.mean(axis=0, dtype=np.float64)
        stds[start:end]  = chunk.std(axis=0, ddof=1, dtype=np.float64)

    list(ex.map(job, range(0, n_cols, COLUMN_CHUNK_SIZE)))

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (means / stds) * np.sqrt(settings.DAYS_PER_YEAR)
    return np.where(stds > 0, sharpe, -np.inf)

# =============================================================================
# GPU HELPERS
# =============================================================================
def _free_gpu_pools() -> None:
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _rows_for_budget(n_cols: int, bytes_per_elem: int, budget_mb: int, n_rows: int) -> int:
    free_b = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    budget = min(budget_mb * 2**20, int(0.6 * free_b))
    return int(max(1, min(n_rows, budget // max(1, n_cols * bytes_per_elem))))


def _sharpe_dtype_and_scale():
    # (float32 / float32) * np.sqrt(DAYS) -> float64 under numpy 2 (NEP 50), float32 under numpy 1
    sqrt_days = np.sqrt(settings.DAYS_PER_YEAR)
    dt = ((np.ones(1, np.float32) / np.ones(1, np.float32)) * sqrt_days).dtype
    return dt, dt.type(sqrt_days)


def _copy_into_pinned(ex, dst: np.ndarray, src: np.ndarray, n_parts: int) -> None:
    rows = dst.shape[0]
    step = max(1, -(-rows // n_parts))
    list(ex.map(lambda r: np.copyto(dst[r:r + step], src[r:r + step]), range(0, rows, step)))


def _gpu_bootstrap_deviations(w_gpu, x, real_blk, n_obs, sharpe_dt, sharpe_scale) -> tuple:
    # same expressions as _bootstrap_moments_chunk, float32 GEMMs
    total_sum   = w_gpu @ x
    total_sumsq = w_gpu @ (x * x)
    means = total_sum / n_obs
    var   = (total_sumsq - n_obs * means * means) / (n_obs - 1)
    var   = cp.maximum(var, 0.0)
    stds  = cp.sqrt(var)
    boot_sharpe = (means / stds).astype(sharpe_dt, copy=False) * sharpe_scale
    boot_sharpe = cp.where(stds > 0, boot_sharpe, -cp.inf)
    deviations  = boot_sharpe - real_blk[None, :]
    sigma       = deviations.std(axis=0, ddof=1)
    return deviations.astype(cp.float32), sigma


def _gpu_studentize(dev32, sigma):
    # columns with sigma <= 0 become garbage here; they are excluded from the top-M by the valid mask
    return (dev32.astype(cp.float64) / sigma[None, :]).astype(cp.float32)


def _gpu_spa_recenter(dev32, real_blk, sigma, threshold: float):
    # same op as apply_spa_recentering, on already studentized float32 deviations
    z   = real_blk / sigma
    bad = z < threshold
    return cp.where(bad[None, :], (dev32.astype(cp.float64) + z[None, :]).astype(cp.float32), dev32)


def _pack_keys(values, valid, col_offset: int):
    # order-preserving float32 -> uint32 key in the high word, finite-space column index in the low word;
    # invalid columns get key 0, strictly below any real key
    u   = cp.ascontiguousarray(values).view(cp.uint32)
    key = cp.where(u >= _SIGN32, ~u, u | _SIGN32).astype(cp.uint64)
    key <<= _SHIFT32
    key |= cp.arange(col_offset, col_offset + values.shape[1], dtype=cp.uint64)[None, :]
    return cp.where(valid[None, :], key, _ZERO64)


def _unpack_keys(keys) -> tuple:
    hi   = (keys >> _SHIFT32).astype(cp.uint32)
    vals = cp.ascontiguousarray(cp.where(hi >= _SIGN32, hi & _MAG32, ~hi)).view(cp.float32)
    cols = (keys & _LOW32).astype(cp.int64)
    return vals, cols


def _check_streaming_vram(n_bootstrap: int, n_obs: int, n_cols: int, m: int, g: int, label: str) -> None:
    chunk_w = min(BOOTSTRAP_CHUNK_SIZE, max(n_cols, 1))
    need_b = (
        n_bootstrap * n_obs * 4                                  # bootstrap weight matrix
        + n_obs * BOOTSTRAP_CHUNK_SIZE * 4                       # device buffer of one input chunk
        + n_bootstrap * chunk_w * GPU_CHUNK_BYTES_PER_ELEM       # bootstrap chunk temporaries
        + n_bootstrap * (g + m) * 8                              # staging + running top-M keys
        + n_cols * 8 * 2                                         # real Sharpe + sigma
        + GPU_VRAM_MARGIN_MB * 2**20
    )
    _free_gpu_pools()
    free_b = cp.cuda.runtime.memGetInfo()[0]
    if need_b > free_b:
        raise MemoryError(
            f"STEPM GPU {label}: needs ~{need_b / 2**30:.1f} GB free VRAM "
            f"(top-M {n_bootstrap}x{m} + staging {g} + chunk workspace), only {free_b / 2**30:.1f} GB free"
        )

# =============================================================================
# STREAMING TOP-M — exact per-row top-M of packed keys, fed chunk by chunk
# =============================================================================
class _TopMAccumulator:
    # Row layout: [staging (g) | running top-m (m)]. An ascending in-place sort leaves the running
    # top-m in the tail; only keys above the current m-th largest (thr) are staged. Keys are unique
    # and > 0, so stale staging entries (all <= thr) can never re-enter the top-m.

    def __init__(self, n_rows: int, m: int, g: int):
        self.n_rows, self.m, self.g = n_rows, m, g
        self.buf      = cp.zeros((n_rows, g + m), dtype=cp.uint64)
        self.thr      = cp.zeros(n_rows, dtype=cp.uint64)
        self.fill     = cp.zeros(n_rows, dtype=cp.int64)
        self._pending = False

    def push(self, keys) -> None:
        cand = keys > self.thr[:, None]
        cnt  = cand.sum(axis=1, dtype=cp.int64)
        if int((self.fill + cnt).max()) > self.g:
            self._flush()
            cand = keys > self.thr[:, None]
            cnt  = cand.sum(axis=1, dtype=cp.int64)
        rows, cols = cp.nonzero(cand)
        if rows.size == 0:
            return
        rank = cp.cumsum(cand, axis=1, dtype=cp.int32)
        self.buf[rows, self.fill[rows] + rank[rows, cols] - 1] = keys[rows, cols]
        self.fill += cnt
        self._pending = True

    def finalize(self, m_eff: int) -> tuple:
        if self._pending:
            self._flush()
        return _unpack_keys(self.buf[:, self.g + self.m - m_eff:][:, ::-1])

    def _flush(self) -> None:
        rows = _rows_for_budget(self.g + self.m, GPU_SORT_BYTES_PER_ELEM, GPU_TOPM_BUDGET_MB, self.n_rows)
        for r0 in range(0, self.n_rows, rows):
            self.buf[r0:r0 + rows].sort(axis=1)
        self.thr = self.buf[:, self.g].copy()
        self.fill.fill(0)
        self._pending = False

# =============================================================================
# STREAMING BOOTSTRAP — one GPU pass over the column chunks; nothing of size
# n_bootstrap x n_cols is ever materialized (except the host copies in DEBUG)
# =============================================================================
class _StreamingBootstrap:

    def __init__(self, matrix_arr: np.ndarray, real_sharpe: np.ndarray, weight_matrix: np.ndarray,
                 progress_label: str, workers: int):
        self.matrix_arr     = matrix_arr
        self.real_sharpe    = real_sharpe
        self.weight_matrix  = weight_matrix
        self.progress_label = progress_label
        self.workers        = workers
        self.n_bootstrap    = weight_matrix.shape[0]
        self.n_obs, self.n_cols = matrix_arr.shape
        self.spa_threshold  = -np.sqrt(2.0 * np.log(np.log(max(self.n_obs, 3)))) if STEPM_USE_SPA else None
        self.sharpe_dt, self.sharpe_scale = _sharpe_dtype_and_scale()

    def run(self, m: int, capture: bool = False, desc: str = "") -> dict:
        try:
            return self._run(m, capture, desc)
        finally:
            _free_gpu_pools()
            sys.stderr.flush()
            sys.stdout.flush()

    def _stage(self, ex, pinned: list, bounds: list, i: int) -> np.ndarray:
        start, end = bounds[i]
        host = pinned[i % 2][: self.n_obs * (end - start)].reshape(self.n_obs, end - start)
        _copy_into_pinned(ex, host, self.matrix_arr[:, start:end], self.workers)
        return host

    def _run(self, m: int, capture: bool, desc: str) -> dict:
        n_boot, n_obs, n_cols = self.n_bootstrap, self.n_obs, self.n_cols
        chunk_w = BOOTSTRAP_CHUNK_SIZE
        m_alloc = min(int(m), n_cols)
        g       = max(1, min(GPU_TOPM_STAGING_COLS, n_cols))
        _check_streaming_vram(n_boot, n_obs, n_cols, m_alloc, g, self.progress_label)

        bounds    = [(s, min(s + chunk_w, n_cols)) for s in range(0, n_cols, chunk_w)]
        raw_host  = np.empty((n_boot, n_cols), dtype=np.float32) if capture else None
        stud_host = np.empty((n_boot, n_cols), dtype=np.float32) if capture else None
        pinned    = [cupyx.empty_pinned(n_obs * chunk_w, dtype=np.float32) for _ in range(2)]
        staged    = [None, None]
        stream    = cp.cuda.Stream(non_blocking=True)

        with ThreadPoolExecutor(self.workers) as ex, stream:
            w_gpu     = cp.asarray(self.weight_matrix)
            real_gpu  = cp.asarray(self.real_sharpe)
            sigma_gpu = cp.empty(n_cols, dtype=cp.float64)
            x_flat    = cp.empty(n_obs * chunk_w, dtype=cp.float32)
            nan_found = cp.zeros((), dtype=cp.bool_)
            topm      = _TopMAccumulator(n_boot, m_alloc, g)

            if bounds:
                staged[0] = self._stage(ex, pinned, bounds, 0)
            for i, (start, end) in enumerate(tqdm(bounds, desc=desc, dynamic_ncols=True)):
                x = x_flat[: n_obs * (end - start)].reshape(n_obs, end - start)
                x.set(staged[i % 2], stream=stream)                       # async H2D from pinned memory
                real_blk = real_gpu[start:end]
                dev32, sigma = _gpu_bootstrap_deviations(w_gpu, x, real_blk, n_obs, self.sharpe_dt, self.sharpe_scale)
                if capture:
                    raw_host[:, start:end] = cp.asnumpy(dev32)
                dev32 = _gpu_studentize(dev32, sigma)
                if capture:
                    stud_host[:, start:end] = cp.asnumpy(dev32)
                if self.spa_threshold is not None:
                    dev32 = _gpu_spa_recenter(dev32, real_blk, sigma, self.spa_threshold)
                valid = sigma > 0
                nan_found |= (cp.isnan(dev32) & valid[None, :]).any()
                sigma_gpu[start:end] = sigma
                keys = _pack_keys(dev32, valid, start)
                del dev32
                if i + 1 < len(bounds):
                    staged[(i + 1) % 2] = self._stage(ex, pinned, bounds, i + 1)   # overlaps the queued GPU work
                topm.push(keys)                                            # synchronizes the stream
                del keys

            sigma_all = cp.asnumpy(sigma_gpu)
            valid_se  = sigma_all > 0
            vals_g, cols_g = topm.finalize(min(m_alloc, int(valid_se.sum())))
            remap_g   = cp.asarray(np.cumsum(valid_se, dtype=np.int64) - 1)   # finite-space -> kept-space index
            topm_vals = cp.asnumpy(vals_g)
            topm_cols = cp.asnumpy(remap_g[cols_g]).astype(np.int32)
            has_nan   = bool(nan_found)
            stream.synchronize()

        if has_nan:
            raise ValueError(f"STEPM {self.progress_label}: NaN in the studentized deviations of a kept column")

        return {"sigma": sigma_all, "vals": topm_vals, "cols": topm_cols, "raw": raw_host, "studentized": stud_host}

# =============================================================================
# SUFFIX K-TH LARGEST INDEX — exact k-th largest of any suffix dev_sorted[:, s:]
# from the per-row top-M, as long as k + s <= M
# =============================================================================
class _TopMTooSmall(Exception):

    def __init__(self, needed: int):
        super().__init__(f"top-M index too small: need {needed}")
        self.needed = int(needed)


class _SuffixKthIndex:

    def __init__(self, vals: np.ndarray, pos: np.ndarray, n_cols: int, workers: int = FDP_TOPM_WORKERS):
        if vals.dtype != np.float32:
            raise ValueError("_SuffixKthIndex expects float32 deviations")
        self.vals, self.pos = vals, pos
        self.m, self.n_cols = int(vals.shape[1]), int(n_cols)
        self.workers = max(1, int(workers))
        self._rows = np.arange(vals.shape[0])
        self._build_sparse()

    @property
    def shape(self) -> tuple:
        return self.vals.shape[0], self.n_cols

    def _build_sparse(self) -> None:
        m, n_cols, n_rows = self.m, self.n_cols, self.vals.shape[0]
        # Sparse view of the entries that can ever be excluded (pos < m), per row,
        # in ascending rank order: rank j and z-position. Padding never matches.
        excl = self.pos < m
        counts = excl.sum(axis=1)
        c = int(counts.max()) if n_rows else 0
        self.xj   = np.full((n_rows, c), m, dtype=np.int32)
        self.xpos = np.full((n_rows, c), n_cols, dtype=np.int32)
        for r in range(n_rows):
            j = np.flatnonzero(excl[r])
            self.xj[r, :j.size]   = j
            self.xpos[r, :j.size] = self.pos[r, j]
        self.c = c

    def kth_largest_suffix(self, s: int, k_eff: int) -> np.ndarray:
        L = k_eff + s
        if L > self.m:
            raise _TopMTooSmall(L)
        if s == 0:
            return self.vals[:, k_eff - 1].copy()
        n_rows = self.vals.shape[0]
        width  = min(self.c, L)
        if self.workers > 1 and n_rows * width > 2_000_000:
            step   = -(-n_rows // self.workers)
            bounds = [(r, min(r + step, n_rows)) for r in range(0, n_rows, step)]
            with ThreadPoolExecutor(len(bounds)) as ex:
                parts = list(ex.map(lambda b: self._rank_rows(b[0], b[1], s, k_eff, L), bounds))
            j = np.concatenate(parts)
        else:
            j = self._rank_rows(0, n_rows, s, k_eff, L)
        return self.vals[self._rows, j]

    def _rank_rows(self, r0: int, r1: int, s: int, k_eff: int, L: int) -> np.ndarray:
        if self.c < L:
            # rank of the k-th kept entry = (k_eff-1) + #excluded entries before it;
            # excluded e_i precedes it iff e_i - i <= k_eff - 1 (monotone in i)
            mask = self.xpos[r0:r1] < s
            rank = np.cumsum(mask, axis=1, dtype=np.int32)
            rank -= 1
            np.subtract(self.xj[r0:r1], rank, out=rank)
            return (k_eff - 1) + np.count_nonzero(mask & (rank <= k_eff - 1), axis=1)
        cnt = np.cumsum(self.pos[r0:r1, :L] >= s, axis=1, dtype=np.int32)
        return np.argmax(cnt >= k_eff, axis=1)

# =============================================================================
# BOOTSTRAP NULL — studentized deviations reduced to what StepM consumes: the
# per-row top-M (values desc + kept-column index); grows on demand by re-streaming
# =============================================================================
def _fdp_topm_size(k_max: int, gamma: float) -> int:
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"FDP_GAMMA must lie in (0, 1), got {gamma}.")
    return int(np.ceil(k_max / gamma - 1.0)) + 1          # abort_at(k_max), same formula as _fdp_try_k


class BootstrapNull:

    def __init__(self, engine: _StreamingBootstrap, run: dict, real_sharpe: np.ndarray, kept_columns: np.ndarray):
        valid_se = run["sigma"] > 0
        self._engine      = engine
        self._sigma_all   = run["sigma"]
        self._presorted   = None
        self.kept_columns = kept_columns[valid_se]
        self.real_sharpe  = real_sharpe[valid_se]
        self.sigma_hat    = run["sigma"][valid_se]
        self.z_stat       = self.real_sharpe / self.sigma_hat
        self.n_bootstrap  = engine.n_bootstrap
        self.n_kept       = int(valid_se.sum())
        self.topm_vals    = run["vals"]
        self.topm_cols    = run["cols"]

    @property
    def m(self) -> int:
        return self.topm_vals.shape[1]

    @property
    def row_max(self) -> np.ndarray:
        return self.topm_vals[:, 0]

    def ensure_topm(self, m: int) -> bool:
        m = min(int(m), self.n_kept)
        if m <= self.m:
            return False
        label = self._engine.progress_label
        run = self._engine.run(m, desc=f"STEPM TOP-M {label} m={m:,}".replace(",", ".").strip())
        if not np.array_equal(run["sigma"], self._sigma_all, equal_nan=True):
            raise RuntimeError(f"STEPM {label}: bootstrap re-stream is not bit-reproducible (sigma mismatch)")
        self.topm_vals, self.topm_cols = run["vals"], run["cols"]
        self._presorted = None
        return True

    def presorted(self) -> tuple:
        if self._presorted is None:
            order     = np.argsort(-self.z_stat)
            inv_order = np.empty(self.n_kept, dtype=np.int32)
            inv_order[order] = np.arange(self.n_kept, dtype=np.int32)
            index = _SuffixKthIndex(self.topm_vals, inv_order[self.topm_cols], self.n_kept)
            self._presorted = (order, index, self.z_stat[order])
        return self._presorted


def compute_bootstrap_null(
    matrix_arr: np.ndarray,
    col_names: list,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    topm_size: int = None,
    progress_label: str = "",
    workers: int = None,
    desc: str = None,
) -> BootstrapNull:

    workers   = STEPM_WORKERS if workers is None else max(1, int(workers))
    topm_size = _fdp_topm_size(FDP_K_MAX, FDP_GAMMA) if topm_size is None else int(topm_size)
    debug     = logger.isEnabledFor(logging.DEBUG)

    n_cols_built  = matrix_arr.shape[1]
    col_names_arr = np.asarray(col_names)

    with ThreadPoolExecutor(workers) as ex:
        real_sharpe = _sharpe_per_column_parallel(matrix_arr, ex)

    if debug:
        day_offsets = np.arange(matrix_arr.shape[0])
        print_stepm_matrix_debug(col_names, matrix_arr, matrix_arr.shape[0], day_offsets)

    finite_mask = np.isfinite(real_sharpe)
    if finite_mask.all():
        kept_columns = col_names_arr
    else:
        n_keep       = compact_columns_inplace(finite_mask, matrix_arr, real_sharpe, col_names_arr, chunk_size=COLUMN_CHUNK_SIZE)
        matrix_arr   = matrix_arr[:, :n_keep]
        real_sharpe  = real_sharpe[:n_keep]
        kept_columns = col_names_arr[:n_keep]

    if debug:
        print_stepm_real_variance_filter_debug(progress_label, n_cols_built, matrix_arr.shape[1])

    n_obs, n_cols = matrix_arr.shape

    rng = np.random.default_rng(seed)
    starts_full, starts_last, len_last, n_blocks_needed = _generate_block_starts(
        n_obs, block_size, n_bootstrap, rng,
    )

    if debug:
        print_stepm_block_starts_debug(progress_label, n_blocks_needed, block_size, len_last, n_obs, n_cols)

    weight_matrix = _build_bootstrap_weight_matrix(
        starts_full, starts_last, block_size, len_last, n_obs, n_bootstrap,
    )

    engine = _StreamingBootstrap(matrix_arr, real_sharpe, weight_matrix, progress_label, workers)
    run    = engine.run(topm_size, capture=debug, desc=desc or f"STEPM BOOTSTRAP {progress_label}".strip())
    null   = BootstrapNull(engine, run, real_sharpe, kept_columns)

    if debug:
        _log_bootstrap_debug(progress_label, run, null, engine.spa_threshold, n_cols_built, n_cols)

    return null


def _log_bootstrap_debug(progress_label: str, run: dict, null: BootstrapNull, spa_threshold,
                         n_cols_built: int, n_cols: int) -> None:
    valid_se = run["sigma"] > 0
    print_stepm_bootstrap_replicas_debug(progress_label, run["raw"], n_cols, null.n_bootstrap)
    print_stepm_se_filter_debug(progress_label, n_cols, null.n_kept, null.sigma_hat)
    print_stepm_studentization_debug(
        progress_label, run["studentized"][:, valid_se], null.z_stat, n_cols_built, n_cols, null.n_kept,
    )
    if spa_threshold is not None:
        spa_mask = null.z_stat < spa_threshold
        logger.debug(
            f"SPA RECENTERING {progress_label} ── threshold={spa_threshold:.4f} ── "
            f"{int(spa_mask.sum())}/{spa_mask.shape[0]} columns recentered to 0"
        )

# =============================================================================
# GLOBAL P-VALUE — single number per timeframe, the original White (2000) test.
# =============================================================================
def compute_global_pvalue(max_deviation: np.ndarray, statistic: np.ndarray) -> dict:
    # max_deviation: per-replica max of the studentized deviations, shape (n_bootstrap,);
    # a full (n_bootstrap, n_cols) matrix is also accepted and reduced here
    if max_deviation.ndim == 2:
        max_deviation = np.max(max_deviation, axis=1)
    best_col_idx   = int(np.argmax(statistic))
    best_statistic = float(statistic[best_col_idx])

    global_p = float(np.mean(max_deviation >= best_statistic))

    return {
        "global_p":       global_p,
        "best_col_idx":    best_col_idx,
        "best_statistic":  best_statistic,
    }

# =============================================================================
# ROW-CHUNKED K-TH LARGEST — same np.partition(...)[:, part_idx] result, but
# =============================================================================
def _kth_largest_by_row_chunks(values: np.ndarray, k_eff: int, chunk_size: int = PARTITION_ROW_CHUNK) -> np.ndarray:
    n_rows, n_cols = values.shape
    part_idx = n_cols - k_eff
    result = np.empty(n_rows, dtype=values.dtype)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        result[start:end] = np.partition(values[start:end], part_idx, axis=1)[:, part_idx]
    return result

# =============================================================================
# CUT DIAGNOSTIC — where the stepdown cut k lands on the cross-sectional grid
# =============================================================================
# Same operands, expression and compilation path as the linear-interpolation kernel
# of cupy.percentile, applied to the two order statistics read from the top-M
_PERCENTILE_LERP = cp.ElementwiseKernel(
    "T a_bottom, T a_top, float64 weight_above",
    "float64 ret",
    """
    double diff = a_top - a_bottom;
    if (weight_above < 0.5) {
        ret = a_bottom + diff * weight_above;
    } else {
        ret = a_top - diff * (1 - weight_above);
    }
    """,
    "stepm_percentile_lerp",
)


def _percentile_from_k(k: int, n_cols: int, grid: np.ndarray = CROSS_SECTIONAL_PERCENTILES) -> float:

    exact = 100.0 * (n_cols - k) / max(n_cols - 1, 1)
    return float(grid[np.argmin(np.abs(grid - exact))])


def compute_cut_diagnostic(null: BootstrapNull, k: int) -> dict:

    n_cols = null.n_kept
    pct    = _percentile_from_k(k, n_cols)
    real   = float(np.percentile(null.z_stat, pct))

    # same float64 index arithmetic as cupy.percentile (linear)
    idx          = (pct / 100.0) * (n_cols - 1.0)
    idx_below    = int(np.floor(idx))
    weight_above = idx - idx_below
    rank_bottom  = n_cols - 1 - idx_below          # ascending order statistic -> descending top-M rank
    rank_top     = max(rank_bottom - 1, 0)         # pct=100: weight_above is 0, the upper operand is unused
    null.ensure_topm(rank_bottom + 1)

    a_bottom = cp.asarray(null.topm_vals[:, rank_bottom])
    a_top    = cp.asarray(null.topm_vals[:, rank_top])
    values   = cp.asnumpy(_PERCENTILE_LERP(a_bottom, a_top, weight_above))
    del a_bottom, a_top
    below    = int((values < real).sum())

    return {
        "percentile":  pct,
        "real":        real,
        "pct_below":   100.0 * below / null.n_bootstrap,
    }
# =============================================================================
# STEPM (ROMANO & WOLF, 2005) — stepdown per-rule p-values controlling FWER
# =============================================================================
def stepwise_reality_check_pvalues(
    deviations,
    statistic: np.ndarray,
    alpha: float = STEPM_ALPHA,
    max_iterations: int = STEPM_MAX_ITERATIONS,
    k: int = 1,
    _presorted: tuple = None,   # (order, dev_sorted | _SuffixKthIndex, stat_sorted), reused across the FDP ladder
    _abort_at: int = None,      # stop early once active_start reaches this count
) -> np.ndarray:

    if k < 1:
        raise ValueError(f"k (k-FWE level) must be >= 1, got {k}.")

    n_bootstrap, n_cols = deviations.shape

    if _presorted is None:
        if isinstance(deviations, _SuffixKthIndex):
            raise ValueError("a _SuffixKthIndex must be passed with its (order, index, stat_sorted) as _presorted")
        order       = np.argsort(-statistic)
        dev_sorted  = deviations[:, order]
        stat_sorted = statistic[order]
    else:
        order, dev_sorted, stat_sorted = _presorted

    kth_index = dev_sorted if isinstance(dev_sorted, _SuffixKthIndex) else None
    if kth_index is not None and _abort_at is not None and min(int(_abort_at), n_cols) > kth_index.m:
        raise _TopMTooSmall(min(int(_abort_at), n_cols))

    raw_pval_sorted = np.full(n_cols, np.nan, dtype=np.float64)
    active_start = 0
    aborted = False

    for _iteration in range(max_iterations):
        n_active = n_cols - active_start
        if n_active <= 0:
            break

        if _abort_at is not None and active_start >= _abort_at:
            aborted = True
            break

        buffer_size    = min(active_start, k - 1)
        extended_start = active_start - buffer_size

        active_stat   = stat_sorted[active_start:]
        n_extended    = n_cols - extended_start
        k_eff         = min(k, n_extended)

        if k_eff == n_extended and n_extended != k:
            logger.debug(
                f"STEPDOWN iter={_iteration} ── requested k={k} exceeds extended set "
                f"size={n_extended} (active={n_active} + buffer={buffer_size}) ── "
                f"clamping to k_eff={k_eff} "
                f"(k-FWE degenerates toward the global minimum in this iteration)"
            )

        if kth_index is not None:
            kth_dev_extended = kth_index.kth_largest_suffix(extended_start, k_eff)
        elif k_eff == 1:
            kth_dev_extended = dev_sorted[:, extended_start:].max(axis=1)
        else:
            kth_dev_extended = _kth_largest_by_row_chunks(dev_sorted[:, extended_start:], k_eff)

        sorted_dev  = np.sort(kth_dev_extended)
        insert_pos  = np.searchsorted(sorted_dev, active_stat, side="left")
        candidate_p = (n_bootstrap - insert_pos) / n_bootstrap

        reject_local = candidate_p <= alpha
        n_reject     = int(reject_local.sum())

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"STEPD iter={_iteration} ── k={k} ── active={n_active} ── "
                f"rejected_this_iter={n_reject} ── "
                f"candidate_p range=[{candidate_p.min():.4f}, {candidate_p.max():.4f}]"
            )
            if _iteration == 0:
                print_stepm_pvalue_quantile_equivalence_debug(k, kth_dev_extended, alpha, active_stat, reject_local, n_active)

        if n_reject == 0:
            raw_pval_sorted[active_start:] = candidate_p
            break

        raw_pval_sorted[active_start:active_start + n_reject] = candidate_p[:n_reject]
        active_start += n_reject
    else:
        unresolved_sorted = np.flatnonzero(np.isnan(raw_pval_sorted))
        if unresolved_sorted.size > 0:
            unresolved = order[unresolved_sorted]
            raise RuntimeError(
                f"StepM stepdown did not converge within {max_iterations} "
                f"iterations; unresolved column indices: {unresolved.tolist()}"
            )

    if aborted:
        return None

    non_finite_sorted = ~np.isfinite(raw_pval_sorted)
    if non_finite_sorted.any():
        bad = order[non_finite_sorted]
        raise RuntimeError(
            f"StepM stepdown produced an undefined raw p-value for column "
            f"index(es) {bad.tolist()}; refusing to silently treat it as zero."
        )

    adjusted_pval_sorted = np.maximum.accumulate(raw_pval_sorted)

    adjusted_pval = np.empty(n_cols, dtype=np.float64)
    adjusted_pval[order] = adjusted_pval_sorted

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_monotonicity_debug(k, adjusted_pval_sorted)

    return adjusted_pval


def _full_stepdown_pvalues(null: BootstrapNull, k: int, alpha: float = STEPM_ALPHA) -> np.ndarray:
    # full (non-aborting) stepdown on the top-M index; the index is re-streamed larger until it covers the run
    while True:
        order, index, stat_sorted = null.presorted()
        try:
            return stepwise_reality_check_pvalues(
                index, null.z_stat, alpha=alpha, k=k, _presorted=(order, index, stat_sorted),
            )
        except _TopMTooSmall as exc:
            if not null.ensure_topm(max(2 * index.m, exc.needed)):
                raise RuntimeError(f"StepM top-M cannot grow beyond {index.m} columns") from exc

# =============================================================================
# FDP CONTROL (ROMANO & WOLF, 2007, ALGORITHM 4.1)
# =============================================================================
def _fdp_try_k(
    deviations,
    statistic: np.ndarray,
    k: int,
    gamma: float,
    alpha: float,
    presorted: tuple,
    timeframe: str,
    tag: str = "",
) -> tuple:

    abort_at = int(np.ceil(k / gamma - 1.0)) + 1

    pvals = stepwise_reality_check_pvalues(
        deviations, statistic, alpha=alpha, k=k,
        _presorted=presorted, _abort_at=abort_at,
    )
    aborted    = pvals is None
    n_rejected = abort_at if aborted else int((pvals <= alpha).sum())

    k_fmt = f"{k:,}".replace(",", ".")
    if aborted:
        # Exact count was never computed — only known to be >= abort_at.
        n_fmt     = f">{abort_at:,}".replace(",", ".")
        gamma_fmt = f"<{gamma:.4f}"
    else:
        n_fmt         = f"{n_rejected:,}".replace(",", ".")
        gamma_implied = k / n_rejected if n_rejected else float("inf")
        gamma_fmt     = f"{gamma_implied:.4f}"
    logger.debug(
        f"FDP SEARCH      {timeframe}: k={k_fmt:>9} ── {n_fmt:>9} columns rejected "
        f"── gamma_implied: {gamma_fmt:>7}{tag}"
    )

    return (k, pvals, n_rejected), n_rejected < k / gamma - 1.0


def resolve_k_by_fdp(
    null: BootstrapNull,
    gamma: float = FDP_GAMMA,
    alpha: float = STEPM_ALPHA,
    k_max: int = FDP_K_MAX,
    timeframe: str = "",
) -> tuple:

    null.ensure_topm(_fdp_topm_size(k_max, gamma))
    presorted = null.presorted()
    index     = presorted[1]
    statistic = null.z_stat

    k        = 1
    k_fail   = 0      # last k that failed the criterion (0 = none yet)
    last_run = None

    while k <= k_max:
        last_run, ok = _fdp_try_k(index, statistic, k, gamma, alpha, presorted, timeframe)
        if ok:
            break
        k_fail = k
        k *= 2
    else:
        logger.warning(
            f"FDP SEARCH     {timeframe}: stop criterion never met up to FDP_K_MAX={k_max} "
            f"── k clamped, realized FDP may exceed gamma={gamma}"
        )
        return last_run

    lo, hi, best = k_fail, k, last_run
    while hi - lo > 1:
        mid = (lo + hi) // 2
        run_mid, ok_mid = _fdp_try_k(
            index, statistic, mid, gamma, alpha, presorted, timeframe, tag="  (bisect)",
        )
        if ok_mid:
            hi, best = mid, run_mid
        else:
            lo = mid

    return best
# =============================================================================
# PIPE STEPM — orchestration layer, mirroring dsr.py's pipe_dsr exactly.
# =============================================================================
def _best_col_idx_by_rule(kept_columns: np.ndarray, stepm_pvals: np.ndarray, z_stat: np.ndarray) -> dict:

    n = len(kept_columns)
    names = [str(c) for c in kept_columns]
    if len(set(names)) != n:
        # duplicated column names: keep the original dict-by-name semantics verbatim
        p_by = dict(zip(kept_columns, stepm_pvals)); z_by = dict(zip(kept_columns, z_stat))
        idx_by = {c: i for i, c in enumerate(kept_columns)}
        best: dict = {}
        for c in kept_columns:
            rid = str(c).rsplit("__", 1)[0]
            cur = best.get(rid)
            if cur is None or (p_by[c], -z_by[c]) < (p_by[cur], -z_by[cur]):
                best[rid] = c
        return {rid: idx_by[c] for rid, c in best.items()}
    ids: dict = {}
    inv = np.fromiter((ids.setdefault(nm.rsplit("__", 1)[0], len(ids)) for nm in names), dtype=np.int64, count=n)
    order = np.lexsort((np.arange(n), -np.asarray(z_stat, dtype=np.float64), np.asarray(stepm_pvals, dtype=np.float64), inv))
    inv_sorted = inv[order]
    first = np.ones(n, dtype=bool)
    first[1:] = inv_sorted[1:] != inv_sorted[:-1]
    return dict(zip(ids.keys(), order[first].tolist()))


def empty_stepm_fields() -> dict:
    # placeholder StepM fields for rules that were never evaluated (pipe skipped)
    return {
        "passed_stepm": True,
        "passed_mbias": True,
        "stepm_p":      None,
        "sharpe":       None,
    }

def pipe_stepm(
    raw_results: list,
    matrix_arr: np.ndarray,
    col_names: list,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    timeframe: str = "",
) -> list:

    if matrix_arr is None:
        logger.warning(f"STEPM ── {timeframe} ── insufficient data — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]

    if matrix_arr.shape[1] < 2:
        logger.warning(f"STEPM ── {timeframe} ── insufficient columns — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]
    null = compute_bootstrap_null(
        matrix_arr, col_names, n_bootstrap=n_bootstrap, block_size=block_size,
        seed=seed, progress_label=timeframe, desc=f"{'STEPM BST IS':<16}{timeframe}",
    )
    kept_columns = null.kept_columns
    real_sharpe  = null.real_sharpe
    sigma_hat    = null.sigma_hat
    z_stat       = null.z_stat

    logger.debug(
        f"STEPM ── {timeframe} ── {matrix_arr.shape[1] - len(kept_columns)} degenerate "
        f"columns dropped ── {len(kept_columns)} columns remain"
    )

    if null.n_kept == 0:
        logger.warning(f"STEPM ── {timeframe} ── no non-degenerate columns — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]

    global_result = compute_global_pvalue(null.row_max, z_stat)
    best_col_idx  = global_result["best_col_idx"]
    best_col_name = str(kept_columns[best_col_idx])

    best_raw_idx  = int(np.argmax(real_sharpe))
    best_raw_name = str(kept_columns[best_raw_idx])

    k_fwe, stepm_pvals, _ = resolve_k_by_fdp(null, timeframe=timeframe)
    logger.debug(f"\n{'─' * 70}")
    logger.debug(f"  MAX RAW SHARPE (no bootstrap adjustment) ── {timeframe}")
    logger.debug(f"{'─' * 70}")
    logger.debug(f"  best column       : {best_raw_name}")
    logger.debug(f"  best real Sharpe  : {real_sharpe[best_raw_idx]:.4f}")
    logger.debug(f"{'─' * 70}\n")

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  GLOBAL WHITE p-value (studentized) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  best column(z) : {best_col_name}")
    logger.info(f"  best Sharpe(z) : {real_sharpe[best_col_idx]:.4f}")
    logger.debug(f" best z-statistic: {global_result['best_statistic']:.4f}  (sigma_hat={sigma_hat[best_col_idx]:.4f})")
    logger.info(f"  global p-value : {global_result['global_p']:.4f}")

    cut = compute_cut_diagnostic(null, k_fwe)
    logger.info(
        f"  P{cut['percentile']:<14.6g}: {cut['pct_below']:.2f}% <Real  "
        f"(k={f'{k_fwe:,}'.replace(',', '.')})"
    )
    logger.info(f"{'─' * 70}\n")

    logger.debug(f"STEPM ── {timeframe} ── k-FWE level k={k_fwe}" + (" (strict FWE)" if k_fwe == 1 else " (relaxed control — reasoned extension, see module docstring)"))

    if stepm_pvals is None:
        stepm_pvals = _full_stepdown_pvalues(null, k_fwe, STEPM_ALPHA)
    del null                                                      # release the top-M index and its host buffers

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_brc_equivalence_debug(timeframe, k_fwe, global_result["global_p"], dict(zip(kept_columns, stepm_pvals)), best_col_name)

    best_idx_by_rule = _best_col_idx_by_rule(kept_columns, stepm_pvals, z_stat)

    n_passed = 0
    results  = []
    for r in raw_results:
        idx           = best_idx_by_rule.get(r["rule_id"])
        col_name      = kept_columns[idx] if idx is not None else None
        best_combo_id = str(col_name).rsplit("__", 1)[1] if col_name else None
        stepm_p       = stepm_pvals[idx] if idx is not None else float("nan")
        sharpe_val    = real_sharpe[idx] if idx is not None else float("nan")
        z_val         = z_stat[idx] if idx is not None else float("nan")
        passed        = bool(np.isfinite(stepm_p) and stepm_p <= STEPM_ALPHA)
        n_passed     += int(passed)

        results.append({
            **r,
            "best_combo_id": best_combo_id,
            "passed_stepm":  passed,
            "passed_mbias":  passed,
            "stepm_p":       float(stepm_p) if np.isfinite(stepm_p) else None,
            "sharpe":        float(sharpe_val) if np.isfinite(sharpe_val) else None,
            "z_stat":        float(z_val) if np.isfinite(z_val) else None,
        })

    n_cols_rejected = int((stepm_pvals <= STEPM_ALPHA).sum())
    gamma_implied   = k_fwe / n_cols_rejected if n_cols_rejected else float("inf")

    k_fwe_fmt           = f"{k_fwe:,}".replace(",", ".")
    n_passed_fmt        = f"{n_passed:,}".replace(",", ".")
    n_total_fmt         = f"{len(raw_results):,}".replace(",", ".")
    n_cols_rejected_fmt = f"{n_cols_rejected:,}".replace(",", ".")
    n_cols_kept_fmt     = f"{len(kept_columns):,}".replace(",", ".")

    logger.info(
        f"STEPM COLUMNS  {timeframe}: k={k_fwe_fmt} ── {n_cols_rejected_fmt}/{n_cols_kept_fmt} columns rejected "
        f"── gamma_implied={gamma_implied:.4f}"
    )
    logger.info(f"STEPM RULES    {timeframe}: {n_passed_fmt}/{n_total_fmt} rules pass")

    return results