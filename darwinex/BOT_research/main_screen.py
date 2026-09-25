#quant_miner/darwinex/BOT_research/main_screen.py (forex)
import os
import sys
import time
import logging

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import COMISION
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key

from indicators import screen_report
from indicators.screen_engine import ScreenConfig, IndicatorPool, run_screen, MIN_PILOT_N
from indicators.screen_report import report_selection, validate_selection

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.screening")
logger.setLevel(logging.INFO)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
N_JOBS = -1
# =============================================================================
# CONFIG
# =============================================================================
NULL_PCT       = 80

TIMEFRAME      = "1H"
MODE           = "NPY" # edge_bp: "YPY" every signal is a trade, "NPY" only signals while flat on its symbol (same cache)
GROUP_N        = 3     # symbols where an indicator (alone) or a pair must pass to be selected
MIN_SCORE      = 2.0   # selection: an alone or pair counts only with score >= this (~sigmas over the null); None: off
PHI_TH         = 0.80  # redundancy: absorbed if its phi with a better candidate (same kind and side) is higher
MIN_COVER      = 0.05  # redundancy: dropped on a side if its rule fires on less of its symbols' candles
MAX_COVER      = 0.60  # redundancy: dropped on a side if its rule fires on more of its symbols' candles
USE_CACHE      = True  # True: reuse the raw results if nothing that affects them changed (else compute and save)

#Computation: changing any of these computes a new cache
DATASET   = "IS"
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]

SELL_AFTER     = [10, 100]
TP_PCT         = [0.5,1.0,1.5]
SL_PCT         = [0.5,1.0,1.5]

N_NULL_PATHS   = 1000    # null: synthetic paths of phase 1 = shifts per null and pair of phase 2 (build the floor)
N_PILOTS       = 200    # pilot: paths of phase 1 = shifts per pilot and pair of phase 2 (null mean and std of every combination, for z)

# Validated here, at import (do not edit)
CFG = ScreenConfig(tp_pct=TP_PCT, sl_pct=SL_PCT, sell_after=SELL_AFTER, commission=float(COMISION),
                   n_null_paths=N_NULL_PATHS, n_pilots=N_PILOTS, null_pct=NULL_PCT, n_jobs=N_JOBS)
validate_selection(GROUP_N, len(SYMBOLS), PHI_TH, MIN_COVER, MAX_COVER, MIN_SCORE, MODE)

# Cache (do not edit)
CACHE_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_cache")
INDICATORS_DIR = os.path.dirname(os.path.abspath(screen_report.__file__))

# =============================================================================
# DATA
# =============================================================================
def load_data():
    """OHLC arrays of every symbol, from the dataset's data folder."""
    ohlcv = build_universe(DATA_FOLDER_BY_DATASET[DATASET], {TIMEFRAME: SYMBOLS}, dataset=DATASET)[TIMEFRAME]
    return prepare_ohlcv_arrays(ohlcv)


def _source_files():
    """Indicators modules the raw results may depend on: every one loaded from that folder, except the report."""
    skip = os.path.abspath(screen_report.__file__)
    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.abspath(f)
        if f != skip and f.startswith(INDICATORS_DIR + os.sep) and os.path.isfile(f):
            files.add(f)
    return sorted(files)


# =============================================================================
# MAIN
# =============================================================================
def main():
    log_run_config()
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    raw = run_screen(load_data(), SYMBOLS, pool, CFG,
                     cache_dir=CACHE_DIR if USE_CACHE else None,
                     cache_name=f"screen_{DATASET}_{TIMEFRAME}",
                     cache_tag=(DATASET, TIMEFRAME),
                     source_files=_source_files())
    report_selection(raw, pool, NULL_PCT, GROUP_N, PHI_TH, MIN_COVER, MAX_COVER, MIN_SCORE, MODE)

def log_run_config() -> None:
    logger.info(f"\n{'─' * 115}")
    logger.info(f"  SCREENING START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET     : {DATASET} ── {TIMEFRAME}")
    logger.info(f"  PARAM GRID  : TP_PCT={TP_PCT} SL_PCT={SL_PCT} SELL_AFTER={SELL_AFTER} "
                f"({len(CFG.configs)} configs, {len(CFG.targets)} targets)")
    logger.info(f"  NULL FLOOR  : N_NULL_PATHS={N_NULL_PATHS} NULL_PCT={NULL_PCT}")
    logger.info(f"  PILOT (z)   : N_PILOTS={N_PILOTS} (MIN_PILOT_N={MIN_PILOT_N})")
    logger.info(f"  SELECTION   : GROUP_N={GROUP_N} MIN_SCORE={'off' if MIN_SCORE is None else MIN_SCORE} "
                f"PHI_TH={PHI_TH} MIN_COVER={MIN_COVER:.0%} "
            f"MAX_COVER={MAX_COVER:.0%}")
    logger.info(f"  EDGE MODE   : {MODE}")
    logger.info(f"  CACHE       : USE_CACHE={USE_CACHE}")
    logger.info(f"{'─' * 115}\n")
    
if __name__ == "__main__":
    start = time.time()
    try:
        main()
        elapsed = int(time.time() - start)
        logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
    except KeyboardInterrupt:
        elapsed = int(time.time() - start)
        logger.info(f"\n⛔  INTERRUPTED BY USER — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
        sys.exit(0)