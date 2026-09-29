#quant_miner/darwinex/BOT_research/main_screen.py (forex)
import os
import sys
import json
import time
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import COMISION
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key

from screening.screen_report import report_selection, validate_selection, exclude_indicators
from screening.screen_engine import ScreenConfig, IndicatorPool, run_screen, align_symbols, build_bins
LOG_LEVEL = logging.INFO    # DEBUG: also the redundancy tables, TOP_I and the symbols of the TOP of every SELL_AFTER
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.screening")
logger.setLevel(logging.INFO)
logging.getLogger("screening.screen_report").setLevel(LOG_LEVEL)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
N_JOBS = -1
# =============================================================================
# CONFIG
# =============================================================================
TOP_I       = 8
GROUP_N     = 6     # symbols where an indicator (alone) or a pair must pass to be selected
J_TH        = 0.80  # redundancy of alones: two signals are twins if their Jaccard (candles) is >= this
X_TH        = 0.80  # redundancy of alones: absorbed if >= this share of its useful signals, and its rule, have a twin
MIN_COVER   = 0.05  # selection and pairs: rule firing on less of its symbols' candles out. Alones: useful signals
MAX_COVER   = 0.60  # selection and pairs: rule firing on more of its symbols' candles out. Alones: useful signals
USE_CACHE   = True  # True: reuse the raw results if nothing that affects them changed (else compute and save)
EXCLUDE     = []    # indicators left out of the selection (they stay in the cache: no recompute)]

#Computation: changing any of these computes a new cache
NULL_PCT  = 85     # floor: a symbol passes if its best z is above it; picked: the largest edge among those above it
TIMEFRAME = "4H"
MODE      = "NPY"
DATASET   = "IS"
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]

SELL_AFTER     = [20, 40, 100]   # one screening per value, with every TP_PCT x SL_PCT: its own cache, TOP and SYMBOL_POOL
TP_PCT         = [0.5,1.0,1.5]
SL_PCT         = [0.5,1.0,1.5]

# Validated here, at import (do not edit). Null paths and pilots: N_NULL_PATHS and N_PILOTS of screen_engine
if not SELL_AFTER or len(set(SELL_AFTER)) != len(SELL_AFTER):
    raise ValueError(f"SELL_AFTER must have at least one value and no duplicates: {SELL_AFTER}")
CFGS = {sa: ScreenConfig(tp_pct=TP_PCT, sl_pct=SL_PCT, sell_after=[sa], commission=float(COMISION),
                         null_pct=NULL_PCT, mode=MODE, n_jobs=N_JOBS) for sa in SELL_AFTER}
validate_selection(GROUP_N, len(SYMBOLS), J_TH, X_TH, MIN_COVER, MAX_COVER, MODE, TOP_I)
if set(EXCLUDE) - set(CANDIDATE_REGISTRY):
    raise ValueError(f"EXCLUDE has unknown indicators: {sorted(set(EXCLUDE) - set(CANDIDATE_REGISTRY))}")

# Cache and output (do not edit)
CACHE_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_caches")
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_grids")
OUTPUT_PATH    = os.path.join(OUTPUT_DIR, f"screen_{DATASET}_{TIMEFRAME}.json")

# =============================================================================
# DATA
# =============================================================================
def load_data():
    """OHLC arrays of every symbol, from the dataset's data folder."""
    ohlcv = build_universe(DATA_FOLDER_BY_DATASET[DATASET], {TIMEFRAME: SYMBOLS}, dataset=DATASET)[TIMEFRAME]
    return prepare_ohlcv_arrays(ohlcv)

# =============================================================================
# MAIN
# =============================================================================
def main():
    log_run_config()
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    ohlcv = load_data()
    data = align_symbols(ohlcv, SYMBOLS)
    bins = build_bins(data.ohlcv_arr, data, pool)[0]                # redundancy: the indicators' signals
    out, rules = {}, {}
    for k, sa in enumerate(SELL_AFTER, start=1):
        logger.info(f"\n{'─' * 115}\n  SELL_AFTER={sa} ({k}/{len(SELL_AFTER)})\n{'─' * 115}")
        raw = run_screen(ohlcv, SYMBOLS, pool, CFGS[sa],
                         cache_dir=CACHE_DIR if USE_CACHE else None,
                         cache_name=f"screen_{DATASET}_{TIMEFRAME}",
                         cache_tag={"DATASET": DATASET, "TIMEFRAME": TIMEFRAME})
        rep = report_selection(exclude_indicators(raw, EXCLUDE), pool, bins, NULL_PCT, GROUP_N, J_TH, X_TH,
                               MIN_COVER, MAX_COVER, MODE, TOP_I)
        out[sa] = {"top": rep["top"], "symbols": rep["symbols"]}
        rules[sa] = rep["rules"]
    save_output(out)
    log_output(out, rules)


def save_output(out) -> None:
    """TOP and SYMBOL_POOL of every SELL_AFTER, with what they were selected with, as JSON in OUTPUT_DIR. Atomic
    write: a run cut halfway never leaves a broken file."""
    doc = {"dataset": DATASET, "timeframe": TIMEFRAME, "mode": MODE, "null_pct": NULL_PCT,
           "tp_pct": TP_PCT, "sl_pct": SL_PCT,
           "selection": {"group_n": GROUP_N, "j_th": J_TH, "x_th": X_TH, "min_cover": MIN_COVER,
                         "max_cover": MAX_COVER, "top_i": TOP_I, "exclude": EXCLUDE},
           "created": datetime.now().isoformat(timespec="seconds"),
           "by_sell_after": {str(sa): o for sa, o in out.items()}}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = f"{OUTPUT_PATH}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUTPUT_PATH)


def log_output(out, rules) -> None:
    """TOP, SYMBOL_POOL and rules per symbol of the TOP (rule_generator, both sides) of every SELL_AFTER."""
    logger.info(f"\n{'─' * 115}")
    logger.info(f"  SCREENING OUTPUT ── {OUTPUT_PATH}")
    logger.info(f"{'─' * 115}")
    for sa, o in out.items():
        top_lbl, pool_lbl = f"TOP ({len(o['top'])})", f"SYMBOL_POOL ({len(o['symbols'])})"
        w = max(len(top_lbl), len(pool_lbl), len("RULES"))
        logger.info(f"  SELL_AFTER={sa:<5} {top_lbl:<{w}} : {', '.join(o['top']) or '(none)'}")
        logger.info(f"  {'':<16} {pool_lbl:<{w}} : {', '.join(o['symbols']) or '(none)'}")
        logger.info(f"  {'':<16} {'RULES':<{w}} : "
                    + " | ".join(f"MAX_DEPTH={d}: {format(int(v), ',').replace(',', '.')}" for d, v in rules[sa].items()))
    logger.info(f"{'─' * 115}")


def log_run_config() -> None:
    cfg = CFGS[SELL_AFTER[0]]
    logger.info(f"\n{'─' * 115}")
    logger.info(f"  SCREENING START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET     : {DATASET} ── {TIMEFRAME}")
    logger.info(f"  MODE        : {MODE}")
    logger.info(f"  PARAM GRID  : TP_PCT={TP_PCT} SL_PCT={SL_PCT} "
                f"({len(cfg.configs)} configs, {len(cfg.targets)} targets per SELL_AFTER)")
    logger.info(f"  SELL_AFTER  : {SELL_AFTER} (one screening each: its own cache, TOP and SYMBOL_POOL)")
    logger.info(f"  NULL FLOOR  : N_NULL_PATHS={cfg.n_null_paths} NULL_PCT={NULL_PCT}")
    logger.info(f"  PILOT (z)   : N_PILOTS={cfg.n_pilots}")
    logger.info(f"  SELECTION   : GROUP_N={GROUP_N} MIN_COVER={MIN_COVER:.0%} MAX_COVER={MAX_COVER:.0%} TOP_I={TOP_I}")
    logger.info(f"  REDUNDANCY  : J_TH={J_TH:.2f} X_TH={X_TH:.2f}")
    logger.info(f"  EXCLUDE     : {', '.join(EXCLUDE) if EXCLUDE else 'none'}")
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