# quant_miner/darwinex/BOT_research/config_cache_fx.py (forex)

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import COMISION
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import CANDIDATE_REGISTRY
from screening.screen_engine import ScreenConfig, cache_key, cache_path, load_cache

# =============================================================================
# CONFIG
# =============================================================================
# The caches: changing any of these is another cache (A0 computes it)
DATASET   = "IS"   
MODE      = "NPY"
NULL_PCT  = 85
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]
SELL_AFTER = [20, 40, 100]   
TP_PCT     = [0.5, 1.0, 1.5]
SL_PCT     = [0.5, 1.0, 1.5]
EXCLUDE    = [] #indicators

if not SELL_AFTER or len(set(SELL_AFTER)) != len(SELL_AFTER):
    raise ValueError(f"SELL_AFTER must have at least one value and no duplicates: {SELL_AFTER}")
if set(EXCLUDE) - set(CANDIDATE_REGISTRY):
    raise ValueError(f"EXCLUDE has unknown indicators: {sorted(set(EXCLUDE) - set(CANDIDATE_REGISTRY))}")
CFGS = {sa: ScreenConfig(tp_pct=TP_PCT, sl_pct=SL_PCT, sell_after=[sa], commission=float(COMISION),
                         null_pct=NULL_PCT, mode=MODE) for sa in SELL_AFTER}

# Caches (do not edit)
CACHE_DIR = os.path.join(_HERE, "screen_caches")
FULL_KEY  = "null2_full"       # in a cache: the pairs whose phase 2 nulls are complete (A0 completes them)


def cache_name(tf):
    """Start of the name of the caches of a timeframe."""
    return f"screen_{DATASET}_{tf}"


def cache_tag(tf):
    """What else the caches of a timeframe depend on, in their key."""
    return {"DATASET": DATASET, "TIMEFRAME": tf}


def load_data(tf):
    """OHLC arrays of every symbol in a timeframe, from the dataset's data folder."""
    ohlcv = build_universe(DATA_FOLDER_BY_DATASET[DATASET], {tf: SYMBOLS}, dataset=DATASET)[tf]
    return prepare_ohlcv_arrays(ohlcv)


def cache_file(sa, tf):
    """(path, key) of the cache of one SELL_AFTER in a timeframe."""
    cfg = CFGS[sa]
    key = cache_key(SYMBOLS, cfg, cache_tag(tf))
    return cache_path(CACHE_DIR, cache_name(tf), cfg, len(SYMBOLS), key), key


def load_raw(sa, names, tf):
    """The raw results of one SELL_AFTER in a timeframe (every indicator, EXCLUDE not applied), or None if there is
    no cache."""
    path, key = cache_file(sa, tf)
    return load_cache(path, key, names)


def missing_nulls(raw):
    """Pairs of raw whose phase 2 nulls are not complete yet."""
    full = raw.get(FULL_KEY, set())
    return [p for p in raw["pairs"] if p not in full]