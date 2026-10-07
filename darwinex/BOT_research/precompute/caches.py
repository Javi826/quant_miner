# quant_miner/darwinex/BOT_research/precompute/caches.py (forex)
import os
import sys
import time
import logging
from dataclasses import replace

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))       # quant_miner
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "core"))
sys.path.append(os.path.join(_ROOT, "darwinex"))

from setup import config_research as cr    # what defines the caches (dataset, symbols, grid, NULL_PCT, MODE)
from research import artifacts as ra

from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key
from research.screening.screen_engine import IndicatorPool, run_screen

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.caches")
logger.setLevel(logging.INFO)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
N_JOBS    = -1


# =============================================================================
# CONFIG
# =============================================================================
TIMEFRAME = "1H"


# =============================================================================
# MAIN
# =============================================================================
def main():
    """One cache per SELL_AFTER: loaded if it exists, else computed (both phases, with the whole nulls) and saved."""
    log_run_config()
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    ohlcv = ra.load_data(TIMEFRAME)
    for k, sa in enumerate(cr.SELL_AFTER, start=1):
        logger.info(f"\n{'─' * 115}\n  SELL_AFTER={sa} ({k}/{len(cr.SELL_AFTER)})\n{'─' * 115}")
        cfg = replace(ra.CFGS[sa], n_jobs=N_JOBS)
        run_screen(ohlcv, cr.SYMBOLS, pool, cfg, cache_dir=ra.CACHE_DIR, cache_name=ra.cache_name(TIMEFRAME),
                   cache_tag=ra.cache_tag(TIMEFRAME))
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
    logger.info(f"  COMPUTE     : N_JOBS={N_JOBS}")
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