# quant_miner/darwinex/BOT_research/01_groupn_fx.py (forex)
import os
import sys
import time
import logging

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config_cache_fx as cc    # what defines the caches (dataset, symbols, grid, NULL_PCT, MODE, EXCLUDE); sets sys.path

from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key
from screening.screen_report import exclude_indicators, evaluate2
from screening.screen_engine import IndicatorPool, null_floor

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.groupn")
logger.setLevel(logging.INFO)

# =============================================================================
# CONFIG
# =============================================================================
TIMEFRAME = "1H"          # the caches it reads (A0_main_CACHES_fx must have computed them)
GN_RANGE  = range(3, 11)  # GROUP_N compared (only those <= the number of symbols)
LUCK_MAX  = 0.30          # ✅ if luck <= LUCK_MAX: share of the pairs that pass which would pass with no edge

# The caches (dataset, symbols, grid, NULL_PCT, MODE) and EXCLUDE: config_cache_fx

# Validated here, at import (do not edit)
if len(GN_RANGE) == 0 or min(GN_RANGE) < 1:
    raise ValueError(f"GN_RANGE must have values >= 1: {GN_RANGE}")
if not (0.0 < LUCK_MAX <= 1.0):
    raise ValueError(f"LUCK_MAX must be in (0, 1]: {LUCK_MAX}")


# =============================================================================
# PASS vs BY LUCK
# =============================================================================
def _luck_of(null, gns, pct):

    floor = null_floor(null, pct)
    with np.errstate(invalid="ignore"):
        n_null = (null > floor).sum(axis=1)                                   # passes of every null path
    return (n_null[:, None] >= gns).mean(axis=0)


def luck_pairs(raw, gns):

    gns = np.asarray(gns)
    ev = {p: evaluate2(raw["p2"][p], cc.NULL_PCT) for p in raw["pairs"]}
    n_hit = sum(int(e["pass"].sum()) for e in ev.values())
    n_comp = sum(int(np.isfinite(raw["p2"][p]["T2"]).sum()) for p in raw["pairs"])
    rate = min(n_hit / n_comp if n_comp else 0.0, 1.0 - cc.NULL_PCT / 100.0)
    pct = 100.0 * (1.0 - rate)
    n_pass = np.zeros(len(gns), dtype=np.int64)
    by_luck = np.zeros(len(gns))
    for p in raw["pairs"]:
        r = raw["p2"][p]
        null_a, null_b = (np.asarray(r[k], dtype=np.float64) for k in ("null2_A", "null2_B"))
        by_luck += np.maximum(_luck_of(null_a, gns, pct), _luck_of(null_b, gns, pct))
        n_pass += ev[p]["n_pass"] >= gns
    return n_pass, by_luck, rate


def log_table(title, gns, n_pass, by_luck):

    logger.info(f"\n  {title}")
    logger.info(f"  {'GROUP_N':>7}{'pass':>7}{'by_luck':>9}{'luck':>7}")
    status = []
    for j, k in enumerate(gns):
        if n_pass[j] > 0:
            luck = by_luck[j] / n_pass[j]
            ok = bool(luck <= LUCK_MAX)
            luck_txt, mark = f"{luck:.0%}", ("✅" if ok else "❌")
        else:
            ok, luck_txt, mark = None, "-", ""                                # nothing passes
        status.append(ok)
        logger.info(f"  {k:>7}{n_pass[j]:>7}{by_luck[j]:>9.1f}{luck_txt:>7}  {mark}")
    return status


# =============================================================================
# MAIN
# =============================================================================
def main():
    log_run_config()
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    gns = [k for k in GN_RANGE if k <= len(cc.SYMBOLS)]
    if not gns:
        raise ValueError(f"GN_RANGE has no value <= {len(cc.SYMBOLS)} symbols: {GN_RANGE}")
    firsts = {}                                                     # first ✅ of every SELL_AFTER (None: none)
    for k, sa in enumerate(cc.SELL_AFTER, start=1):
        logger.info(f"\n{'─' * 115}\n  SELL_AFTER={sa} ({k}/{len(cc.SELL_AFTER)})\n{'─' * 115}")
        name = os.path.basename(cc.cache_file(sa, TIMEFRAME)[0])
        raw = cc.load_raw(sa, pool.names, TIMEFRAME)
        if raw is None:
            logger.info(f"🔴 No cache {name} (run A0_main_CACHES_fx with TIMEFRAME={TIMEFRAME} first)")
            continue
        raw = exclude_indicators(raw, cc.EXCLUDE)
        missing = cc.missing_nulls(raw)
        if missing:
            logger.info(f"🔴 Phase 2 nulls incomplete in {name}: {len(missing)} of {len(raw['pairs'])} pairs "
                        f"(run A0_main_CACHES_fx with TIMEFRAME={TIMEFRAME} first)")
            continue
        logger.info(f"🟢 Cache loaded: {name}")
        n_pass, by_luck, rate = luck_pairs(raw, gns)
        status = log_table(f"Pairs: {len(raw['pairs'])} pairs, measured pass rate per symbol {rate:.1%} "
                           f"(nominal {1.0 - cc.NULL_PCT / 100.0:.0%})", gns, n_pass, by_luck)
        firsts[sa] = next((g for g, s in zip(gns, status) if s), None)
    log_result(gns, firsts)


def log_result(gns, firsts) -> None:
    """The legend and GROUP_N: the laxest of the first ✅ of every SELL_AFTER."""
    if not firsts:
        return
    logger.info(f"\n{'─' * 115}")
    logger.info(f"  pass: pairs passing in >= GROUP_N symbols. by_luck: how many would with no edge (their two nulls "
                f"at the measured rate). luck = by_luck / pass, ✅ if <= LUCK_MAX={LUCK_MAX:.0%}. -: nothing passes")
    detail = ", ".join(f"{sa}: {g if g is not None else '-'}" for sa, g in firsts.items())
    ok = [g for g in firsts.values() if g is not None]
    if ok:
        logger.info(f"\n→ GROUP_N: {min(ok)} (the laxest of the first ✅ of every SELL_AFTER; {detail})")
    else:
        logger.info(f"\n→ No SELL_AFTER with ✅ in [{gns[0]}, {gns[-1]}] ({detail})")
    logger.info(f"{'─' * 115}")


def log_run_config() -> None:
    logger.info(f"\n{'─' * 115}")
    logger.info("  GROUP_N CALIBRATION START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET     : {cc.DATASET} ── {TIMEFRAME}")
    logger.info(f"  MODE        : {cc.MODE} NULL_PCT={cc.NULL_PCT}")
    logger.info(f"  SELL_AFTER  : {cc.SELL_AFTER}")
    logger.info(f"  GROUP_N     : {list(GN_RANGE)}")
    logger.info(f"  DECISION    : LUCK_MAX={LUCK_MAX:.0%}")
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