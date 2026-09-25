#BOT_batch_E1/main_pipeline.py (crypto)
import os
import sys
import time
import logging
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.DEBUG, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_rule_mining")
logger.setLevel(LOG_LEVEL)
MODULE_LOG_LEVELS = {
    "BOT_batch.pipeline.universe":          logging.INFO,
    "BOT_batch.rule_mining.generator":      logging.INFO,
    "BOT_batch.pipeline.signal_cleaning":   logging.INFO,
    "BOT_batch.pipeline.backtest_runner":   logging.INFO,
    "BOT_batch.pipeline.stepM_is":          logging.INFO,
    "BOT_batch.pipeline.stepM_oos":         logging.INFO,
    "BOT_batch.pipeline.wfo":               logging.INFO,
    "BOT_batch.engines.wfo_WF":             logging.INFO,
    "BOT_batch.pipeline.correlation":       logging.INFO,
    "BOT_batch.pipeline.multiverse":        logging.INFO,
    "BOT_batch.runs.run_best_wfo_portfolio":logging.INFO,
    "BOT_batch.rule_mining.writter":        logging.INFO,
    "BOT_batch.runs.run_deploy":            logging.INFO,
    "BOT_batch.utils.reporting":            logging.INFO,
}
for module_name, level in MODULE_LOG_LEVELS.items():
    logging.getLogger(module_name).setLevel(level)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
 #-----------------------------------------------------------------------------
   
from symbols.universe import build_universe, MIN_START_DATE_BY_DATASET
from setup.config_paths import DATA_FOLDER_BY_DATASET
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from pipeline.wfo import WFO_TRAIN_MONTHS, WFO_TEST_MONTHS, EMA_ALPHA, WFO_NET_GAIN_TH, WFO_DD_TH, WFO_R2_TH, WFO_WFR_TH
from pipeline.correlation import CORRELATION_DD_TH
from pipeline.multiverse import MULTIVERSE_PVALUE_TH
from pipeline.signal_cleaning import JACCARD_SIMILARITY_TH
from utils.ohlcv_utils import prepare_ohlcv_arrays
from setup.config_backtest import ORDER_AMOUNT
from rule_mining.rule_runner import run_rule_mining_pipeline

# =============================================================================
# RUNS + OUTPUTS — portfolio construction and output stages
# =============================================================================
SHOW_PLOTS    = True
SAVE_TRADES   = False
RUN_DEPLOY    = True
SPLIT_MODE    = False

DATASET_IS, DATASET_OOS = ("IS", "OOS") if SPLIT_MODE else ("MERGED", "MERGED")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

TIMEFRAMES = ["1H","4H","6Hutc","12Hutc"]
#TIMEFRAMES = ["12Hutc"]

SYMBOL_COMBOS_BY_TIMEFRAME = {
    "1H":     [["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]],
    "4H":     [["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]],
    "6Hutc":  [["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]],
    "12Hutc": [["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]],
}

PARAM_GRID_BY_TIMEFRAME = {
    "1H": {
        "SELL_AFTER": [0],
        "TP_PCT":     [6,8,10],
        "SL_PCT":     [6,8],
    },
    "4H": {
        "SELL_AFTER": [0],
        "TP_PCT":     [6,8,10],
        "SL_PCT":     [6,8],
    },
}

# =============================================================================
# PATHS
# =============================================================================
STRATEGIES_E1_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_E1")
BRIEF_TRADES_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "brief_trades")
DEPLOY_OUTPUT_PATH   = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch.py")

# =============================================================================
# RUN CONFIG — single source of truth: printed at startup AND persisted
# =============================================================================
run_config = {"SPLIT_MODE": SPLIT_MODE, "DATASET_IS": DATASET_IS, "DATASET_OOS": DATASET_OOS, 
              "TIMEFRAMES": TIMEFRAMES, "SYMBOL_COMBOS_BY_TIMEFRAME": SYMBOL_COMBOS_BY_TIMEFRAME, 
              "PARAM_GRID_BY_TIMEFRAME": PARAM_GRID_BY_TIMEFRAME}
# =============================================================================
# COMBOS — each timeframe can be mined with several independent symbol baskets
# =============================================================================
def build_combos() -> list:
    """Flatten SYMBOL_COMBOS_BY_TIMEFRAME into an ordered list of combo descriptors."""
    return [
        {
            "combo_key": f"{timeframe}_c{combo_idx:02d}",
            "timeframe": timeframe,
            "symbols":   list(symbols),
        }
        for timeframe in TIMEFRAMES
        for combo_idx, symbols in enumerate(SYMBOL_COMBOS_BY_TIMEFRAME[timeframe], start=1)
    ]

def load_ohlcv_by_combo(combos: list, data_folder: str, dataset: str) -> tuple:
    """Load and validate one independent universe per combo; returns (data, arrays) keyed by combo_key."""
    ohlcv_data_by_combo, ohlcv_arr_by_combo = {}, {}
    for combo in combos:
        combo_key, timeframe = combo["combo_key"], combo["timeframe"]
        ohlcv_data = build_universe(
            data_folder, {timeframe: combo["symbols"]}, dataset=dataset,
        )[timeframe]
        ohlcv_data_by_combo[combo_key] = ohlcv_data
        ohlcv_arr_by_combo[combo_key]  = prepare_ohlcv_arrays(ohlcv_data)
    return ohlcv_data_by_combo, ohlcv_arr_by_combo

# =============================================================================
# LOGGING HELPERS — render the startup banner from run_config
# =============================================================================
def _pipeline_icon(enabled: bool) -> str:
    return "🟢" if enabled else "⚪"

def log_run_config() -> None:
    logger.info(f"\n{'─' * 115}")
    logger.info(f"  RULE MINING START")
    logger.info(f"{'─' * 115}")
    if SPLIT_MODE:
        logger.info(
            f"  DATASET     : SPLIT ── IS: {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET_IS])} "
            f"({MIN_START_DATE_BY_DATASET[DATASET_IS]}) | "
            f"OOS: {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET_OOS])} "
            f"({MIN_START_DATE_BY_DATASET[DATASET_OOS]})"
        )
    else:
        logger.info(
            f"  DATASET     : {DATASET_IS} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET_IS])} "
            f"({MIN_START_DATE_BY_DATASET[DATASET_IS]})"
        )
    logger.info(f"  SYMBOLS     :")
    for combo in build_combos():
        logger.info(f"    {combo['combo_key']:<12}({len(combo['symbols'])}) {combo['symbols']}")
    logger.info(f"  TIMEFRAMES  : {TIMEFRAMES}")
    logger.debug(f"  MAX DEPTH  : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM GRID  : {PARAM_GRID_BY_TIMEFRAME}")
    logger.info(f"  WFO WINDOWS : train={WFO_TRAIN_MONTHS}m test={WFO_TEST_MONTHS}m | EMA_ALPHA: {EMA_ALPHA}")
    logger.info(
        f"  PIPES       : JACCARD_TH={JACCARD_SIMILARITY_TH} | "
        f"NET_GAIN_TH={WFO_NET_GAIN_TH} DD_TH={WFO_DD_TH} R2_TH={WFO_R2_TH} WFR_TH={WFO_WFR_TH} | "
        f"CORR_TH={CORRELATION_DD_TH} | "
        f"MV_PVALUE_TH={MULTIVERSE_PVALUE_TH}"
    )
    logger.info(
        f"  RUNS        : DEPLOY: {_pipeline_icon(RUN_DEPLOY)}"
    )
    logger.info(f"{'─' * 115}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    try:
        missing_tf = [tf for tf in TIMEFRAMES if not SYMBOL_COMBOS_BY_TIMEFRAME.get(tf)]
        if missing_tf:
            raise ValueError(f"SYMBOL_COMBOS_BY_TIMEFRAME has no combos for timeframes: {missing_tf}")

        log_run_config()

        # -------------------------------------------------------------------
        # DATA LOADING — one call validates and loads every symbol, every
        # timeframe, up front (fails fast if any symbol is bad).
        # -------------------------------------------------------------------
        combos = build_combos()

        ohlcv_data_is_by_combo, ohlcv_arr_is_by_combo = load_ohlcv_by_combo(
            combos, DATA_FOLDER_BY_DATASET[DATASET_IS], DATASET_IS,
        )

        if SPLIT_MODE:
            ohlcv_data_oos_by_combo, ohlcv_arr_oos_by_combo = load_ohlcv_by_combo(
                combos, DATA_FOLDER_BY_DATASET[DATASET_OOS], DATASET_OOS,
            )
        else:
            # Single-source mode: both roles share the same already-loaded dataset.
            ohlcv_data_oos_by_combo = ohlcv_data_is_by_combo
            ohlcv_arr_oos_by_combo  = ohlcv_arr_is_by_combo
        # -------------------------------------------------------------------
        # RULE MINING — Phase A: DSR for every timeframe, then a combined
        # -------------------------------------------------------------------
        validated_wfo_test, all_mbias_results = run_rule_mining_pipeline(
            ohlcv_data_is_by_combo             = ohlcv_data_is_by_combo,
            ohlcv_arr_is_by_combo              = ohlcv_arr_is_by_combo,
            ohlcv_data_oos_by_combo            = ohlcv_data_oos_by_combo,
            ohlcv_arr_oos_by_combo             = ohlcv_arr_oos_by_combo,
            combos                             = combos,
            param_grid                         = PARAM_GRID_BY_TIMEFRAME,
            order_amount                       = ORDER_AMOUNT,
            data_folder                        = DATA_FOLDER_BY_DATASET[DATASET_OOS],
            max_depth                          = RULE_MAX_DEPTH,
            log_level                          = MODULE_LOG_LEVELS["BOT_batch.pipeline.wfo"],
            save_trades                        = SAVE_TRADES,
            brief_trades_folder                = BRIEF_TRADES_FOLDER,
            show_plots                         = SHOW_PLOTS,
            deploy_output_path                 = DEPLOY_OUTPUT_PATH,
            run_config                         = run_config,
            run_deploy                         = RUN_DEPLOY,
        )

        elapsed = int(time.time() - start)
        logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    except KeyboardInterrupt:
        elapsed = int(time.time() - start)
        logger.info(f"\n⛔  INTERRUPTED BY USER — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
        sys.exit(0)