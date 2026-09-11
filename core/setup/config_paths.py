import os
# =============================================================================
# DATASET FOLDERS — IS/OOS for split mode, MERGED for single-source mode
# =============================================================================
# =============================================================================
# BITGET_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget"))
# SPLIT_BASE          = os.path.join(BITGET_ROOT, "data_crypto", "data", "04_split", "expanding")
# 
# DATA_FOLDER_IS      = os.path.join(SPLIT_BASE, "IS",  "crypto_2022-01_2024-01_IS")
# DATA_FOLDER_OOS     = os.path.join(SPLIT_BASE, "OOS", "crypto_2024-01_2026-08_OOS")
# DATA_FOLDER_MERGED  = os.path.join(SPLIT_BASE, "IS",  "crypto_2022-01_2026-08_IS")
# 
# DATA_FOLDER_BY_DATASET = {
#     "IS":     DATA_FOLDER_IS,
#     "OOS":    DATA_FOLDER_OOS,
#     "MERGED": DATA_FOLDER_MERGED,
# }
# =============================================================================

BITGET_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "darwinex"))
SPLIT_BASE          = os.path.join(BITGET_ROOT, "data_forex", "data_fx", "04_split", "expanding")

DATA_FOLDER_IS      = os.path.join(SPLIT_BASE, "IS",  "fx_2017-01_2024-01_IS")
DATA_FOLDER_OOS     = os.path.join(SPLIT_BASE, "OOS", "fx_2024-01_2026-09_OOS")
DATA_FOLDER_MERGED  = os.path.join(SPLIT_BASE, "IS",  "fx_2018-01_2026-09_IS")

DATA_FOLDER_BY_DATASET = {
    "IS":     DATA_FOLDER_IS,
    "OOS":    DATA_FOLDER_OOS,
    "MERGED": DATA_FOLDER_MERGED,
}
