# quant_miner/darwinex/BOT_research/02_screen_fx.py (forex)
import os
import sys
import json
import time
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config_cache_fx as cc    # what defines the caches (dataset, symbols, grid, NULL_PCT, MODE, EXCLUDE); sets sys.path

from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key
from screening.screen_report import report_selection, validate_selection, exclude_indicators
from screening.screen_engine import IndicatorPool, align_symbols, build_bins, log_config

LOG_LEVEL = logging.INFO    # DEBUG: also the redundancy tables, TOP_I and the symbols of the TOP of every SELL_AFTER
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.screening")
logger.setLevel(logging.INFO)
logging.getLogger("screening.screen_report").setLevel(LOG_LEVEL)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# =============================================================================
# CONFIG
# =============================================================================
TIMEFRAME = "1H"
GROUP_N   = 1    
TOP_I     = 8
J_TH      = 0.80  
X_TH      = 0.80 
MIN_COVER = 0.05  
MAX_COVER = 0.60       
RULES_MAX = 1_000_000  

# Validated here, at import (do not edit)
validate_selection(GROUP_N, len(cc.SYMBOLS), J_TH, X_TH, MIN_COVER, MAX_COVER, cc.MODE, TOP_I)
if not (isinstance(RULES_MAX, int) and RULES_MAX > 0):
    raise ValueError(f"RULES_MAX must be an integer > 0: {RULES_MAX}")

# Output (do not edit)
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_grids")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"screen_{cc.DATASET}_{TIMEFRAME}.json")


# =============================================================================
# MAIN
# =============================================================================
def main():
    log_run_config()
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    raws = {}
    for sa in cc.SELL_AFTER:
        raw = cc.load_raw(sa, pool.names, TIMEFRAME)
        if raw is None:
            logger.info(f"🔴 SELL_AFTER={sa}: no cache {os.path.basename(cc.cache_file(sa, TIMEFRAME)[0])} "
                        f"(run A0_main_CACHES_fx with TIMEFRAME={TIMEFRAME} first)")
        else:
            raws[sa] = raw
    if not raws:
        return
    data = align_symbols(cc.load_data(TIMEFRAME), cc.SYMBOLS)
    bins = build_bins(data.ohlcv_arr, data, pool)[0]                # redundancy: the indicators' signals
    out, rules = {}, {}
    for k, (sa, raw) in enumerate(raws.items(), start=1):
        logger.info(f"\n{'─' * 115}\n  SELL_AFTER={sa} ({k}/{len(raws)})\n{'─' * 115}")
        logger.info(f"🟢 Cache loaded: {os.path.basename(cc.cache_file(sa, TIMEFRAME)[0])} (created {raw['created']})")
        log_config(raw["config"])
        for key, s in raw["empty"]:
            logger.info(f"  WARNING: {key} has no values in {s}")
        rep = report_selection(exclude_indicators(raw, cc.EXCLUDE), pool, bins, cc.NULL_PCT, GROUP_N, J_TH, X_TH,
                               MIN_COVER, MAX_COVER, cc.MODE, TOP_I)
        out[sa] = {"top": rep["top"], "symbols": rep["symbols"]}
        rules[sa] = rep["rules"]
    save_output(out)
    log_output(out, rules)


def save_output(out) -> None:
    """TOP and SYMBOL_POOL of every SELL_AFTER, with what they were selected with, as JSON in OUTPUT_DIR. Atomic
    write: a run cut halfway never leaves a broken file."""
    doc = {"dataset": cc.DATASET, "timeframe": TIMEFRAME, "mode": cc.MODE, "null_pct": cc.NULL_PCT,
           "tp_pct": cc.TP_PCT, "sl_pct": cc.SL_PCT,
           "selection": {"group_n": GROUP_N, "j_th": J_TH, "x_th": X_TH, "min_cover": MIN_COVER,
                         "max_cover": MAX_COVER, "top_i": TOP_I, "exclude": cc.EXCLUDE},
           "created": datetime.now().isoformat(timespec="seconds"),
           "by_sell_after": {str(sa): o for sa, o in out.items()}}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = f"{OUTPUT_PATH}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUTPUT_PATH)


def _fmt_int(v):
    return format(int(v), ",").replace(",", ".")


def log_output(out, rules) -> None:
    """TOP, SYMBOL_POOL and rules of the TOP (rule_generator, both sides) of every SELL_AFTER, with a warning where
    the rules are more than RULES_MAX."""
    logger.info(f"\n{'─' * 115}")
    for sa, o in out.items():
        top_lbl, pool_lbl = f"TOP ({len(o['top'])})", f"SYMBOL_POOL ({len(o['symbols'])})"
        w = max(len(top_lbl), len(pool_lbl), len("RULES"))
        logger.info(f"  SELL_AFTER={sa:<5} {top_lbl:<{w}} : {', '.join(o['top']) or '(none)'}")
        logger.info(f"  {'':<16} {pool_lbl:<{w}} : {', '.join(o['symbols']) or '(none)'}")
        logger.info(f"  {'':<16} {'RULES':<{w}} : "
                    + " | ".join(f"MAX_DEPTH={d}: {_fmt_int(v)}" for d, v in rules[sa].items()))
        big = [d for d, v in rules[sa].items() if v > RULES_MAX]
        if big:
            logger.info(f"  {'':<16} {'':<{w}}   ⚠  more than {_fmt_int(RULES_MAX)} rules at MAX_DEPTH="
                        + ", ".join(str(d) for d in big))
    logger.info(f"{'─' * 115}")
    logger.info(f"✅  SCREENING OUTPUT ── {OUTPUT_PATH}")
    logger.info(f"{'─' * 115}")


def log_run_config() -> None:
    cfg = cc.CFGS[cc.SELL_AFTER[0]]
    logger.info(f"\n{'─' * 115}")
    logger.info("  SCREENING START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET     : {cc.DATASET} ── {TIMEFRAME}")
    logger.info(f"  MODE        : {cc.MODE}")
    logger.info(f"  PARAM GRID  : TP_PCT={cc.TP_PCT} SL_PCT={cc.SL_PCT} "
                f"({len(cfg.configs)} configs, {len(cfg.targets)} targets per SELL_AFTER)")
    logger.info(f"  SELL_AFTER  : {cc.SELL_AFTER} (one screening each: its own cache, TOP and SYMBOL_POOL)")
    logger.info(f"  NULL FLOOR  : N_NULL_PATHS={cfg.n_null_paths} NULL_PCT={cc.NULL_PCT}")
    logger.info(f"  PILOT (z)   : N_PILOTS={cfg.n_pilots}")
    logger.info(f"  SELECTION   : GROUP_N={GROUP_N} MIN_COVER={MIN_COVER:.0%} MAX_COVER={MAX_COVER:.0%} TOP_I={TOP_I} "
                f"RULES_MAX={_fmt_int(RULES_MAX)}")
    logger.info(f"  REDUNDANCY  : J_TH={J_TH:.2f} X_TH={X_TH:.2f}")
    logger.info(f"  EXCLUDE     : {', '.join(cc.EXCLUDE) if cc.EXCLUDE else 'none'}")
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