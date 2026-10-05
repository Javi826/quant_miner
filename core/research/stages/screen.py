# core/research/stages/screen.py
import os
import logging
from dataclasses import dataclass

from setup import config_research as cr
from research import artifacts as ra
from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key
from research.screening.screen_report import report_selection, validate_selection, exclude_indicators
from research.screening.screen_engine import IndicatorPool, align_symbols, build_bins, log_config
from utils.ohlcv_utils import get_bars_per_day

LOG_LEVEL = logging.INFO    # DEBUG: also the redundancy tables, TOP_I and the symbols of the TOP of every SELL_AFTER
logger    = logging.getLogger("BOT_research.screening")

# =============================================================================
# MODULE CONFIG
# =============================================================================
J_TH      = 0.80  
X_TH      = 0.80 
MIN_COVER = 0.05  
MAX_COVER = 0.60       
RULES_MAX = 1_000_000  

if not (isinstance(RULES_MAX, int) and RULES_MAX > 0):
    raise ValueError(f"RULES_MAX must be an integer > 0: {RULES_MAX}")


# =============================================================================
# STAGE CONFIG (set by the orchestrator)
# =============================================================================
@dataclass(frozen=True)
class ScreenStageConfig:
    group_n: int = 1
    top_i:   int = 5

    def __post_init__(self):
        validate_selection(self.group_n, len(cr.SYMBOLS), J_TH, X_TH, MIN_COVER, MAX_COVER, cr.MODE, self.top_i)


def configure_loggers() -> None:
    logger.setLevel(logging.INFO)
    logging.getLogger("research.screening.screen_report").setLevel(LOG_LEVEL)
    for noisy_logger in ("joblib", "matplotlib", "numba"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _ordered_timeframes(timeframes: list) -> list:
    return sorted(timeframes, key=get_bars_per_day, reverse=True)    # smallest timeframe first


# =============================================================================
# RUN
# =============================================================================
def run(cfg: ScreenStageConfig) -> dict:
    configure_loggers()
    log_run_config(cfg)
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    tfs  = _ordered_timeframes(cr.TIMEFRAMES)
    docs = {}
    for tf in tfs:
        raws = {}
        for sa in cr.SELL_AFTER:
            raw = ra.load_raw(sa, pool.names, tf)
            if raw is None:
                logger.info(f"🔴 SELL_AFTER={sa}: no cache {os.path.basename(ra.cache_file(sa, tf)[0])} "
                            f"(run precompute/caches.py with TIMEFRAME={tf} first)")
            else:
                raws[sa] = raw
        if not raws:
            continue
        data = align_symbols(ra.load_data(tf), cr.SYMBOLS)
        bins = build_bins(data.ohlcv_arr, data, pool)[0]                # redundancy: the indicators' signals
        out, rules = {}, {}
        for k, (sa, raw) in enumerate(raws.items(), start=1):
            logger.info(f"\n{'─' * 115}\n  {tf} ── SELL_AFTER={sa} ({k}/{len(raws)})\n{'─' * 115}")
            logger.info(f"🟢 Cache loaded: {os.path.basename(ra.cache_file(sa, tf)[0])} (created {raw['created']})")
            log_config(raw["config"])
            for key, s in raw["empty"]:
                logger.info(f"  WARNING: {key} has no values in {s}")
            rep = report_selection(exclude_indicators(raw, cr.EXCLUDE), pool, bins, cr.NULL_PCT, cfg.group_n, J_TH,
                                   X_TH, MIN_COVER, MAX_COVER, cr.MODE, cfg.top_i)
            out[sa] = {"top": rep["top"], "symbols": rep["symbols"]}
            rules[sa] = rep["rules"]
        docs[tf] = save_output(cfg, tf, out)
        log_output(tf, out, rules)
    return docs


def save_output(cfg: ScreenStageConfig, timeframe: str, out) -> dict:
    meta = {"mode": cr.MODE, "null_pct": cr.NULL_PCT, "tp_pct": cr.TP_PCT, "sl_pct": cr.SL_PCT,
            "selection": {"group_n": cfg.group_n, "j_th": J_TH, "x_th": X_TH, "min_cover": MIN_COVER,
                          "max_cover": MAX_COVER, "top_i": cfg.top_i, "exclude": cr.EXCLUDE}}
    return ra.save_grids_input(timeframe, out, meta)


def _fmt_int(v):
    return format(int(v), ",").replace(",", ".")


def log_output(timeframe: str, out, rules) -> None:

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  TIMEFRAME={timeframe}")
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
    logger.info(f"✅  SCREENING OUTPUT ── {ra.grids_path(timeframe)}")
    logger.info(f"{'─' * 115}")


def log_run_config(stage: ScreenStageConfig) -> None:
    cfg = ra.CFGS[cr.SELL_AFTER[0]]
    logger.info(f"\n{'─' * 115}")
    logger.info("  SCREEN START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET     : {cr.DATASET} ── {_ordered_timeframes(cr.TIMEFRAMES)}")
    logger.info(f"  MODE        : {cr.MODE}")
    logger.info(f"  PARAM GRID  : TP_PCT={cr.TP_PCT} SL_PCT={cr.SL_PCT} "
                f"({len(cfg.configs)} configs, {len(cfg.targets)} targets per SELL_AFTER)")
    logger.info(f"  SELL_AFTER  : {cr.SELL_AFTER} (one screening each: its own cache, TOP and SYMBOL_POOL)")
    logger.info(f"  NULL FLOOR  : N_NULL_PATHS={cfg.n_null_paths} NULL_PCT={cr.NULL_PCT}")
    logger.info(f"  PILOT (z)   : N_PILOTS={cfg.n_pilots}")
    logger.info(f"  SELECTION   : GROUP_N={stage.group_n} MIN_COVER={MIN_COVER:.0%} MAX_COVER={MAX_COVER:.0%} "
                f"TOP_I={stage.top_i} "
                f"RULES_MAX={_fmt_int(RULES_MAX)}")
    logger.info(f"  REDUNDANCY  : J_TH={J_TH:.2f} X_TH={X_TH:.2f}")
    logger.info(f"  EXCLUDE     : {', '.join(cr.EXCLUDE) if cr.EXCLUDE else 'none'}")
    logger.info(f"{'─' * 115}\n")
