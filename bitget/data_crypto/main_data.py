#data_pipline/main_data.py

import os
import sys
import time
import shutil
import logging

_BITGET_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _p in [_BITGET_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
        
from steps import integrity
from steps import step0_symbol_selection
from steps import step1_extraction
from steps import step3_cleaning
from steps import step5_highlow
from steps import step7_split

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("pipeline")

# =============================================================================
# FOLDERS
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data_cr")
RAW_DIR     = os.path.join(DATA_DIR, "01_raw")
CLEAN_DIR   = os.path.join(DATA_DIR, "02_clean")
HIGHLOW_DIR = os.path.join(DATA_DIR, "03_highlow")
SPLIT_DIR   = os.path.join(DATA_DIR, "04_split")

# =============================================================================
# PIPELINE CONFIG
# =============================================================================
EXPORT_CSV = False   

# =============================================================================
# SYMBOL SELECTION
# =============================================================================
 #manual
SELECTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "UNIUSDT",
    "XLMUSDT",
    "XRPUSDT",
]

SYMBOL_MODE        = "manual"   #manual o auto
N_SYMBOLS_DOWNLOAD = 25
RWA_MODE           = "crypto_only"   # "crypto_only" | "rwa_only"                                                            
REFERENCE_SYMBOL   = "BTCUSDT"
                                       
# =============================================================================
# EXTRACTION
# =============================================================================
TIMEFRAMES = ["1D","12H","6H","4H","1H","5m"]
START_DATE = "2019-01-01"
END_DATE   = None 

# =============================================================================
# HIGH/LOW TIMESTAMPS
# =============================================================================
TIMEFRAMES_HIGHLOW = [["1D","1H"],["12H","5m"],["6H","5m"],["4H","5m"],["1H","5m"]]

# =============================================================================
# SPLIT DATA
# =============================================================================
WINDOW_OOS_MONTHS    = 0
SPLIT_REFERENCE_DATE = None

# =============================================================================
# HELPERS
# =============================================================================

def _make_dirs() -> None:
    for d in [RAW_DIR, CLEAN_DIR, HIGHLOW_DIR, SPLIT_DIR]:
        os.makedirs(d, exist_ok=True)

def _run_step(name: str, fn, config: dict) -> bool:
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    try:
        result  = fn(config)
        elapsed = time.time() - t0
        m, s    = divmod(elapsed, 60)
        logger.info(f"  ✅ {name} completed in {int(m)}m {int(s)}s")
        return result if isinstance(result, bool) else True
    except Exception as e:
        elapsed = time.time() - t0
        logger.info(f"  ❌ {name} FAILED after {elapsed:.1f}s: {e}")
        return False


def _build_config(timeframe: str | None = None, selected_symbols: list | None = None) -> dict:
    return {
        "start_date":                    START_DATE,
        "end_date":                      END_DATE,
        "timeframe":                     timeframe,
        "selected_symbols":              selected_symbols or SELECTED_SYMBOLS,
        "symbol_mode":                   SYMBOL_MODE,
        "n_symbols_download":            N_SYMBOLS_DOWNLOAD,
        "timeframes_highlow":            TIMEFRAMES_HIGHLOW,
        "window_oos_months":             WINDOW_OOS_MONTHS,
        "split_reference_date":          SPLIT_REFERENCE_DATE,
        "export_csv":                    EXPORT_CSV,
        "raw_dir":                       RAW_DIR,
        "clean_dir":                     CLEAN_DIR,
        "highlow_dir":                   HIGHLOW_DIR,
        "split_dir":                     SPLIT_DIR,
        "rwa_mode": RWA_MODE,
        "reference_symbol": REFERENCE_SYMBOL,
    }

# =============================================================================
# PIPELINE
# =============================================================================

def _run_pipeline() -> None:
    collector = integrity.IssueCollector()

    # Split preview + confirmation before any work is done
    config_preview = _build_config()
    if not step7_split.print_split_preview(config_preview):
        logger.info("❌ Pipeline cancelled by user.")
        return

    # Step 0 — Symbol selection (resolves symbol list once for all timeframes)
    config_s0 = _build_config()
    ok = _run_step("STEP 0 — Symbol Selection", step0_symbol_selection.run, config_s0)
    if not ok:
        logger.info("❌ Pipeline aborted at STEP 0.")
        return
    selected_symbols = config_s0["selected_symbols"]

    # Steps 1-4 — Extraction + integrity + cleaning per timeframe
    logger.info(f"\n📋 Timeframes to extract: {TIMEFRAMES}\n")
    for tf in TIMEFRAMES:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  TIMEFRAME: {tf}")
        logger.info(f"{'#'*60}")
        config = _build_config(timeframe=tf, selected_symbols=selected_symbols)
        ok = _run_step(f"STEP 1 — Extraction [{tf}]", step1_extraction.run, config)
        if not ok:
            logger.info(f"❌ Extraction failed for {tf}. Skipping to next timeframe.")
            continue
        _run_step(f"STEP 2 — Raw Integrity [{tf}]", lambda c: integrity.run_raw(c, collector), config)
        ok = _run_step(f"STEP 3 — Cleaning [{tf}]", step3_cleaning.run, config)
        if not ok:
            logger.info(f"❌ Cleaning failed for {tf}. Skipping to next timeframe.")
            continue
        ok = _run_step(f"STEP 4 — Clean Integrity [{tf}]", lambda c: integrity.run_clean(c, collector), config)
        if not ok:
            logger.info(f"❌ Clean integrity failed for {tf}. Skipping to next timeframe.")

    # Coverage check — runs once after all timeframes are downloaded
    _run_step("STEP 2b — Coverage Integrity", lambda c: integrity.run_coverage(c, collector), _build_config(selected_symbols=selected_symbols))

    # Steps 5-7 — High/Low + integrity + IS/OOS split
    config = _build_config(selected_symbols=selected_symbols)
    ok = _run_step("STEP 5 — High/Low Timestamps", step5_highlow.run, config)
    if not ok:
        logger.info("❌ Pipeline aborted at STEP 5.")
        return
    _run_step("STEP 6 — High/Low Integrity", lambda c: integrity.run_highlow(c, collector), config)
    ok = _run_step("STEP 7 — IS/OOS Split", step7_split.run, config)
    if not ok:
        logger.info("❌ Pipeline aborted at STEP 7.")
        integrity.print_summary(collector)
        removed = config.get("removed_symbols", [])
        if removed:
            logger.info(f"\n{'='*60}")
            logger.info(f"  SYMBOLS REMOVED (< {step5_highlow.MIN_CANDLES_1D} daily candles)")
            logger.info(f"{'='*60}")
            for sym in sorted(removed):
                logger.info(f"  - {sym}")
            logger.info(f"{'='*60}")
        return

    for d in [CLEAN_DIR, HIGHLOW_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            logger.info(f"🗑 Cleaned up: {os.path.basename(d)}/")

    integrity.print_summary(collector)
    removed = config.get("removed_symbols", [])
    if removed:
        logger.info(f"\n{'='*60}")
        logger.info(f"  SYMBOLS REMOVED (< {step5_highlow.MIN_CANDLES_1D} daily candles)")
        logger.info(f"{'='*60}")
        for sym in sorted(removed):
            logger.info(f"  - {sym}")
        logger.info(f"{'='*60}")
# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("\n🚀 Starting data pipeline")
    _make_dirs()
    t_total = time.time()
    _run_pipeline()
    elapsed = time.time() - t_total
    m, s = divmod(elapsed, 60)
    logger.info(f"\n🏁 Pipeline completed in {int(m)}m {int(s)}s")


if __name__ == "__main__":
    main()