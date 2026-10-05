# quant_miner/darwinex/BOT_research/precompute/caches.py (forex)
import os
import sys
import time
import logging
from dataclasses import replace

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))       # quant_miner
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "core"))
sys.path.append(os.path.join(_ROOT, "darwinex"))

from setup import config_research as cr    # what defines the caches (dataset, symbols, grid, NULL_PCT, MODE)
from research import artifacts as ra

from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key
from research.screening.screen_engine import (NULL_BATCH, IndicatorPool, align_symbols, build_bins, build_targets,
                                     build_exits, phase2_shifts, swap_pair, run_screen, save_cache, progress,
                                     _pair_z_npy)
from research.screening.screen_kernels_cpu import (MIN_PILOT_N, pair_edge_moments, pair_valid_mask, pair_z_edges,
                                          pair_null_distribution)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.caches")
logger.setLevel(logging.INFO)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
N_JOBS    = -1     
SAVE_MIN  = 10   


# =============================================================================
# CONFIG
# =============================================================================
TIMEFRAME = "1H"


# =============================================================================
# PHASE 2 NULLS WITHOUT EARLY STOP
# =============================================================================

def _session(data, cfg, pool):

    Y = build_targets(data.ohlcv_arr, data, cfg)
    shifts, pilot = phase2_shifts(data.n, cfg)
    ses = {"Y": Y, "shifts": shifts, "pilot": pilot, "n_sym": len(data.symbols), "ncut": pool.ncut}
    if cfg.mode == "NPY":
        from research.screening.screen_kernels_gpu import npy_prepare, npy_shifts
        ses["prep"] = npy_prepare(Y, build_exits(data.ohlcv_arr, data, cfg))
        ses["shifts_d"] = npy_shifts(shifts, data.n)
        ses["pilot_d"] = npy_shifts(pilot, data.n)
    return ses


def pair_nulls_npy(bA, bB, ncvA, ncvB, ses):

    import cupy as cp
    from research.screening.screen_kernels_gpu import npy_edges, npy_pair_setup, npy_pair_moments, npy_pair_null

    Y, prep, ncut, n_sym = ses["Y"], ses["prep"], ses["ncut"], ses["n_sym"]
    nA, nB, n_tg         = bA.shape[0], bB.shape[0], Y.shape[2]
    ctx_ab               = npy_pair_setup(bA, ncvA, bB, ncvB, prep, ncut)                               
    ctx_ba               = npy_pair_setup(ctx_ab["bins_b"], ncvB, ctx_ab["bins_a"], ncvA, prep, ncut)     


    muB, sdB       = npy_pair_moments(ctx_ab, ses["pilot_d"], float(MIN_PILOT_N))
    muA_ba, sdA_ba = npy_pair_moments(ctx_ba, ses["pilot_d"], float(MIN_PILOT_N))
    ok             = cp.asarray(pair_valid_mask(bA, bB, ncvA, ncvB, Y, ncut))
    ok_ba          = swap_pair(ok, nA, nB, n_tg, ncut, cp)
    muB, sdB       = cp.where(ok, muB, np.nan), cp.where(ok, sdB, np.nan)
    muA_ba, sdA_ba = cp.where(ok_ba, muA_ba, np.nan), cp.where(ok_ba, sdA_ba, np.nan)
    del ok, ok_ba
    ab             = (swap_pair(muA_ba, nB, nA, n_tg, ncut, cp), swap_pair(sdA_ba, nB, nA, n_tg, ncut, cp), muB, sdB)
    ba             = (muA_ba, sdA_ba, swap_pair(muB, nA, nB, n_tg, ncut, cp), swap_pair(sdB, nA, nB, n_tg, ncut, cp))
    del muB, sdB, muA_ba, sdA_ba

    v, osum = npy_edges(ctx_ab["bins_a"], ncvA, prep, ncut, ctx_ab["bins_b"], ncvB)
    t_sym = cp.asnumpy(_pair_z_npy(v, osum, *ab, cp).max(axis=0))
    del v, osum

    sym = np.arange(n_sym, dtype=np.int64)                                                    # no early stop
    out = {"B": npy_pair_null(ctx_ab, ses["shifts_d"], *ab, sym),
           "A": npy_pair_null(ctx_ba, ses["shifts_d"], *ba, sym)}
    for x in out.values():
        x[:, ~np.isfinite(t_sym)] = np.nan                       # as the screening: no null where T2 does not compete
    return t_sym, out


def _null_ypy(bA, bB, ncvA, ncvB, Y, shifts, mus, sym, ncut):
    """T of every shift of B (rows) and symbol, YPY statistic (CPU), in batches of NULL_BATCH as the screening."""
    out = np.full((len(shifts), len(sym)), np.nan)
    for q0 in range(0, len(shifts), NULL_BATCH):
        sh = np.ascontiguousarray(shifts[q0:q0 + NULL_BATCH])
        out[q0:q0 + len(sh)] = pair_null_distribution(bA, bB, ncvA, ncvB, Y, sh, *mus, sym, ncut)[:, sym]
    return out


def pair_nulls_ypy(bA, bB, ncvA, ncvB, ses):
    """As pair_nulls_npy, YPY statistic on the CPU. Pilots and z as screen_pair."""
    Y, ncut, n_sym, shifts = ses["Y"], ses["ncut"], ses["n_sym"], ses["shifts"]
    nA, nB, n_tg = bA.shape[0], bB.shape[0], Y.shape[2]

    muB, sdB       = pair_edge_moments(bA, bB, ncvA, ncvB, Y, ses["pilot"], float(MIN_PILOT_N), ncut)
    muA_ba, sdA_ba = pair_edge_moments(bB, bA, ncvB, ncvA, Y, ses["pilot"], float(MIN_PILOT_N), ncut)
    ok             = pair_valid_mask(bA, bB, ncvA, ncvB, Y, ncut)
    ok_ba          = swap_pair(ok, nA, nB, n_tg, ncut)
    muB[~ok]       = np.nan
    sdB[~ok]       = np.nan
    muA_ba[~ok_ba] = np.nan
    sdA_ba[~ok_ba] = np.nan
    del ok, ok_ba
    muA, sdA       = swap_pair(muA_ba, nB, nA, n_tg, ncut), swap_pair(sdA_ba, nB, nA, n_tg, ncut)
    muB_ba, sdB_ba = swap_pair(muB, nA, nB, n_tg, ncut), swap_pair(sdB, nA, nB, n_tg, ncut)
    ab, ba         = (muA, sdA, muB, sdB), (muA_ba, sdA_ba, muB_ba, sdB_ba)


    z, _e = pair_z_edges(bA, bB, ncvA, ncvB, Y, muA, sdA, muB, sdB, ncut)
    t_sym = z.max(axis=0)

    sym = np.arange(n_sym, dtype=np.int64)                                                    # no early stop
    out = {"B": _null_ypy(bA, bB, ncvA, ncvB, Y, shifts, ab, sym, ncut),
           "A": _null_ypy(bB, bA, ncvB, ncvA, Y, shifts, ba, sym, ncut)}
    for x in out.values():
        x[:, ~np.isfinite(t_sym)] = np.nan                       # as the screening: no null where T2 does not compete
    return t_sym, out


def fill_nulls(data, pool, cfg, raw, bins, ncv, path):

    full = raw.setdefault(ra.FULL_KEY, set())
    todo = [p for p in raw["pairs"] if p not in full]
    logger.info(f"🟡 Phase 2 nulls without early stop, {len(todo)} of {len(raw['pairs'])} pairs to complete: "
                f"{os.path.basename(path)}")
    ses = _session(data, cfg, pool)
    compute = pair_nulls_npy if cfg.mode == "NPY" else pair_nulls_ypy
    by_ind = pool.by_ind
    off_t2 = off_null = 0                                    # pairs whose T2 / nulls differ from the screening's
    t0 = t_save = time.time()
    try:
        for j, (a, b) in enumerate(todo, start=1):
            r = raw["p2"][(a, b)]
            t_sym, out = compute(np.ascontiguousarray(bins[by_ind[a]]), np.ascontiguousarray(bins[by_ind[b]]),
                                 ncv[by_ind[a]], ncv[by_ind[b]], ses)
            off_t2 += not np.allclose(t_sym, r["T2"], rtol=1e-6, atol=1e-9, equal_nan=True)
            bad = False
            for k, x in (("null2_A", out["A"]), ("null2_B", out["B"])):
                old = np.asarray(r[k], dtype=np.float64)
                both = ~np.isnan(old) & ~np.isnan(x)
                bad |= not np.allclose(old[both], x[both], rtol=1e-6, atol=1e-9)
                r[k] = np.where(np.isnan(old), x, old)
            off_null += bad
            full.add((a, b))
            if time.time() - t_save > 60 * SAVE_MIN:
                save_cache(path, raw)
                t_save = time.time()
            progress(f"  Phase 2 nulls, {len(todo)} pairs", j, len(todo), t0)
    finally:
        save_cache(path, raw)
    if off_t2:
        logger.info(f"  WARNING: {off_t2} pairs with a T2 different from the screening's")
    if off_null:
        logger.info(f"  WARNING: {off_null} pairs with null values different from the screening's")


# =============================================================================
# MAIN
# =============================================================================
def main():
    log_run_config()
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    ohlcv = ra.load_data(TIMEFRAME)
    data = align_symbols(ohlcv, cr.SYMBOLS)
    bins = ncv = None
    for k, sa in enumerate(cr.SELL_AFTER, start=1):
        logger.info(f"\n{'─' * 115}\n  SELL_AFTER={sa} ({k}/{len(cr.SELL_AFTER)})\n{'─' * 115}")
        cfg = replace(ra.CFGS[sa], n_jobs=N_JOBS)
        path, _key = ra.cache_file(sa, TIMEFRAME)
        raw = run_screen(ohlcv, cr.SYMBOLS, pool, cfg, cache_dir=ra.CACHE_DIR, cache_name=ra.cache_name(TIMEFRAME),
                         cache_tag=ra.cache_tag(TIMEFRAME))
        if not ra.missing_nulls(raw):
            logger.info(f"🟢 Phase 2 nulls complete: {os.path.basename(path)}")
            continue
        if bins is None:
            bins, ncv, _empty = build_bins(data.ohlcv_arr, data, pool)
        fill_nulls(data, pool, cfg, raw, bins, ncv, path)
    logger.info(f"\n{'─' * 115}")


def log_run_config() -> None:
    cfg = ra.CFGS[cr.SELL_AFTER[0]]
    logger.info(f"\n{'─' * 115}")
    logger.info("  CACHES START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET     : {cr.DATASET} ── {TIMEFRAME}")
    logger.info(f"  MODE        : {cr.MODE}")
    logger.info(f"  PARAM GRID  : TP_PCT={cr.TP_PCT} SL_PCT={cr.SL_PCT} "
                f"({len(cfg.configs)} configs, {len(cfg.targets)} targets per SELL_AFTER)")
    logger.info(f"  SELL_AFTER  : {cr.SELL_AFTER} (one cache each)")
    logger.info(f"  NULL FLOOR  : N_NULL_PATHS={cfg.n_null_paths} NULL_PCT={cr.NULL_PCT}")
    logger.info(f"  PILOT (z)   : N_PILOTS={cfg.n_pilots}")
    logger.info(f"  COMPUTE     : N_JOBS={N_JOBS} SAVE_MIN={SAVE_MIN}")
    logger.info(f"{'─' * 115}\n")


if __name__ == "__main__":
    start = time.time()
    try:
        main()
        elapsed = int(time.time() - start)
        logger.info(f"\n🏁 TOTAL ── {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
    except KeyboardInterrupt:
        elapsed = int(time.time() - start)
        logger.info(f"\n⛔  INTERRUPTED BY USER ── {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
        sys.exit(0)