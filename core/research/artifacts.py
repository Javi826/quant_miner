# core/research/artifacts.py
import os
import json
from datetime import datetime

from setup import config_research as cr
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import COMISION
from symbols.universe import build_universe
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import CANDIDATE_REGISTRY
from research.screening.screen_engine import ScreenConfig, cache_key, cache_path, load_cache

# =============================================================================
# VALIDATION OF setup/config_research
# =============================================================================
if not cr.TIMEFRAMES:
    raise ValueError("TIMEFRAMES must have at least one timeframe")
if not cr.SELL_AFTER or len(set(cr.SELL_AFTER)) != len(cr.SELL_AFTER):
    raise ValueError(f"SELL_AFTER must have at least one value and no duplicates: {cr.SELL_AFTER}")
if set(cr.EXCLUDE) - set(CANDIDATE_REGISTRY):
    raise ValueError(f"EXCLUDE has unknown indicators: {sorted(set(cr.EXCLUDE) - set(CANDIDATE_REGISTRY))}")

CFGS = {sa: ScreenConfig(tp_pct=cr.TP_PCT, sl_pct=cr.SL_PCT, sell_after=[sa], commission=float(COMISION),
                         null_pct=cr.NULL_PCT, mode=cr.MODE) for sa in cr.SELL_AFTER}

# =============================================================================
# PATHS
# =============================================================================
CACHE_DIR  = cr.CACHE_DIR                                       # written by precompute/caches, read by groupn and screen
GRIDS_DIR  = os.path.join(cr.ARTIFACTS_DIR, "screen_grids")     # written by the screen stage, read by grids
COMBOS_DIR = os.path.join(cr.ARTIFACTS_DIR, "screen_combos")    # written by the grids stage, read by combos
FULL_KEY   = "null2_full"       # in a cache: the pairs whose phase 2 nulls are complete (caches completes them)


def _write_json(path: str, doc: dict) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)
    return doc


def _read_json(path: str, producer: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file {path} (run the {producer} stage first)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# =============================================================================
# DATA AND CACHES
# =============================================================================
def load_data(tf: str):
    ohlcv = build_universe(DATA_FOLDER_BY_DATASET[cr.DATASET], {tf: cr.SYMBOLS}, dataset=cr.DATASET)[tf]
    return prepare_ohlcv_arrays(ohlcv)


def cache_name(tf: str) -> str:
    return f"screen_{cr.DATASET}_{tf}"


def cache_tag(tf: str) -> dict:
    return {"DATASET": cr.DATASET, "TIMEFRAME": tf}


def cache_file(sa: int, tf: str) -> tuple:
    cfg = CFGS[sa]
    key = cache_key(cr.SYMBOLS, cfg, cache_tag(tf))
    return cache_path(CACHE_DIR, cache_name(tf), cfg, len(cr.SYMBOLS), key), key


def load_raw(sa: int, names, tf: str):
    path, key = cache_file(sa, tf)
    return load_cache(path, key, names)


def missing_nulls(raw: dict) -> list:
    full = raw.get(FULL_KEY, set())
    return [p for p in raw["pairs"] if p not in full]


# =============================================================================
# SCREEN OUTPUT ── input of grids
# =============================================================================
def grids_path(tf: str) -> str:
    return os.path.join(GRIDS_DIR, f"screen_{cr.DATASET}_{tf}.json")


def save_grids_input(tf: str, by_sell_after: dict, meta: dict) -> dict:
    doc = {"dataset": cr.DATASET, "timeframe": tf, **meta, "created": _now(),
           "by_sell_after": {str(sa): o for sa, o in by_sell_after.items()}}
    return _write_json(grids_path(tf), doc)


def check_grids_input(doc: dict, tf: str, mode: str) -> dict:
    if doc["dataset"] != cr.DATASET or doc["timeframe"] != tf:
        raise ValueError(f"Screen output of {tf} is {doc['dataset']} {doc['timeframe']}, expected {cr.DATASET} {tf}")
    if doc["mode"] != mode:
        raise ValueError(f"Screen output of {tf} was screened in MODE={doc['mode']} but the backtest runs in "
                         f"MODE={mode} (BACKTEST_MODE in setup/config_core): they must be the same")
    return doc


def load_grids_input(tf: str, mode: str) -> dict:
    return check_grids_input(_read_json(grids_path(tf), "screen"), tf, mode)


# =============================================================================
# GRIDS OUTPUT ── input of combos
# =============================================================================
def combos_path() -> str:
    return os.path.join(COMBOS_DIR, f"screen_{cr.DATASET}.json")


def save_combos_input(by_timeframe: dict, meta: dict) -> dict:
    doc = {"dataset": cr.DATASET, **meta, "created": _now(), "by_timeframe": by_timeframe}
    return _write_json(combos_path(), doc)


def check_combos_input(doc: dict, mode: str, timeframes: list) -> dict:
    if doc["dataset"] != cr.DATASET:
        raise ValueError(f"Grids output is dataset {doc['dataset']}, expected {cr.DATASET}")
    if doc["mode"] != mode:
        raise ValueError(f"Grids output was built in MODE={doc['mode']}, expected {mode}")
    missing = [tf for tf in timeframes if tf not in doc["by_timeframe"]]
    if missing:
        raise ValueError(f"Grids output has no winner for timeframes: {missing} (see the grids ranking)")
    return doc


def load_combos_input(mode: str, timeframes: list) -> dict:
    return check_combos_input(_read_json(combos_path(), "grids"), mode, timeframes)
