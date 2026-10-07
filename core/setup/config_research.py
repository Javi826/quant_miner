# core/setup/config_research.py
import os

# =============================================================================
# RESEARCH ── what defines the caches and the research stages of a market
# =============================================================================

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "darwinex", "BOT_research", "artifacts"))
CACHE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "darwinex", "BOT_research", "precompute", "caches"))
BROKER        = "darwinex"             # darwinex | bitget: the signals package of the scripts


DATASET    = "IS"
MODE       = "NPY"
NULL_PCT   = 85
TIMEFRAMES = ["1H", "4H"]   # used by groupn and every stage (run smallest first)
SYMBOLS    = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]
SELL_AFTER = [20, 40, 100]
TP_PCT     = [0.5, 1.0, 1.5]
SL_PCT     = [0.5, 1.0, 1.5]
EXCLUDE    = []             # indicators


