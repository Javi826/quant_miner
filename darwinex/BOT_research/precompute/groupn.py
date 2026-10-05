# quant_miner/darwinex/BOT_research/precompute/groupn.py (forex)
import os
import sys
import time
import logging

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))       # quant_miner
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "core"))
sys.path.append(os.path.join(_ROOT, "darwinex"))

from setup import config_research as cr    # what defines the caches (dataset, symbols, grid, NULL_PCT, MODE, EXCLUDE)
from research import artifacts as ra

from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key
from research.screening.screen_report import exclude_indicators, evaluate2
from research.screening.screen_engine import IndicatorPool, null_floor
from utils.ohlcv_utils import get_bars_per_day

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.groupn")
logger.setLevel(logging.INFO)

# =============================================================================
# CONFIG
# =============================================================================
GN_RANGE   = range(3, 11)   # GROUP_N compared (only those <= the number of symbols)
LUCK_MAX   = 0.20           # ✅ if luck <= LUCK_MAX: share of the pairs that pass which would pass with no edge

# The caches (dataset, symbols, grid, NULL_PCT, MODE) and EXCLUDE: core/setup/config_research

if len(GN_RANGE) == 0 or min(GN_RANGE) < 1:
    raise ValueError(f"GN_RANGE must have values >= 1: {GN_RANGE}")
if not (0.0 < LUCK_MAX <= 1.0):
    raise ValueError(f"LUCK_MAX must be in (0, 1]: {LUCK_MAX}")

SEP   = "─" * 115
SEP2  = "═" * 115
LBL_W = 12
BLK_W = 24      # a SELL_AFTER block of the timeframe table: pass(6) by_luck(9) luck(6) " " mark(2)


def _ordered_timeframes(timeframes: list) -> list:
    return sorted(timeframes, key=get_bars_per_day, reverse=True)    # smallest timeframe first


# =============================================================================
# PASS vs BY LUCK (unchanged)
# =============================================================================
def _luck_of(null, gns, pct):

    floor = null_floor(null, pct)
    with np.errstate(invalid="ignore"):
        n_null = (null > floor).sum(axis=1)                                   # passes of every null path
    return (n_null[:, None] >= gns).mean(axis=0)


def luck_pairs(raw, gns):

    gns = np.asarray(gns)
    ev = {p: evaluate2(raw["p2"][p], cr.NULL_PCT) for p in raw["pairs"]}
    n_hit = sum(int(e["pass"].sum()) for e in ev.values())
    n_comp = sum(int(np.isfinite(raw["p2"][p]["T2"]).sum()) for p in raw["pairs"])
    rate = min(n_hit / n_comp if n_comp else 0.0, 1.0 - cr.NULL_PCT / 100.0)
    pct = 100.0 * (1.0 - rate)
    n_pass = np.zeros(len(gns), dtype=np.int64)
    by_luck = np.zeros(len(gns))
    for p in raw["pairs"]:
        r = raw["p2"][p]
        null_a, null_b = (np.asarray(r[k], dtype=np.float64) for k in ("null2_A", "null2_B"))
        by_luck += np.maximum(_luck_of(null_a, gns, pct), _luck_of(null_b, gns, pct))
        n_pass += ev[p]["n_pass"] >= gns
    return n_pass, by_luck, rate


def status_of(n_pass, by_luck):
    """Per GROUP_N: (luck, ok). ok True ✅, False ❌, None when nothing passes (luck NaN)."""
    out = []
    for p, b in zip(n_pass, by_luck):
        if p > 0:
            luck = b / p
            out.append((luck, bool(luck <= LUCK_MAX)))
        else:
            out.append((np.nan, None))
    return out


def first_ok(gns, status):
    """The first GROUP_N with ✅ (None: none)."""
    return next((g for g, (_, ok) in zip(gns, status) if ok), None)


# =============================================================================
# MAIN
# =============================================================================
def main():
    tfs = _ordered_timeframes(cr.TIMEFRAMES)
    log_run_config(tfs)
    pool = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    gns = [k for k in GN_RANGE if k <= len(cr.SYMBOLS)]
    if not gns:
        raise ValueError(f"GN_RANGE has no value <= {len(cr.SYMBOLS)} symbols: {GN_RANGE}")
    firsts_by_tf = {}                                               # {tf: {sa: first ✅ or None}}, only SA with cache
    for i_tf, tf in enumerate(tfs, start=1):
        logger.info(f"\n{SEP2}\n  TIMEFRAME={tf} ({i_tf}/{len(tfs)})\n{SEP2}")
        res = screen_timeframe(tf, pool, gns)
        firsts_by_tf[tf] = {sa: first_ok(gns, r["status"]) for sa, r in res.items()}
        if res:
            log_tf_table(gns, res)
    log_summary(gns, firsts_by_tf)


def screen_timeframe(tf: str, pool, gns) -> dict:
    """{sa: {"n_pass", "by_luck", "status"}} of every SELL_AFTER of the timeframe with a complete cache."""
    res = {}
    for sa in cr.SELL_AFTER:
        name = os.path.basename(ra.cache_file(sa, tf)[0])
        raw = ra.load_raw(sa, pool.names, tf)
        if raw is None:
            logger.info(f"🔴 SELL_AFTER={sa:<5} no cache {name} (run precompute/caches.py with TIMEFRAME={tf} first)")
            continue
        raw = exclude_indicators(raw, cr.EXCLUDE)
        missing = ra.missing_nulls(raw)
        if missing:
            logger.info(f"🔴 SELL_AFTER={sa:<5} phase 2 nulls incomplete in {name}: {len(missing)} of "
                        f"{len(raw['pairs'])} pairs (run precompute/caches.py with TIMEFRAME={tf} first)")
            continue
        n_pass, by_luck, rate = luck_pairs(raw, gns)
        logger.info(f"🟢 SELL_AFTER={sa:<5} {name} ── {len(raw['pairs'])} pairs ── pass rate per symbol {rate:.1%} "
                    f"(nominal {1.0 - cr.NULL_PCT / 100.0:.0%})")
        res[sa] = {"n_pass": n_pass, "by_luck": by_luck, "status": status_of(n_pass, by_luck)}
    return res


# =============================================================================
# PRINT
# =============================================================================
def _mark(ok) -> str:
    return "✅" if ok else ("❌" if ok is False else "  ")


def log_tf_table(gns, res: dict) -> None:
    """One row per GROUP_N, one block per SELL_AFTER: pass, by_luck, luck and its mark."""
    lead = f"  {'GROUP_N':>7}"
    logger.info("")
    logger.info(" " * len(lead) + "".join(f"  │{f'SELL_AFTER={sa}':^{BLK_W}}" for sa in res))
    logger.info(lead + "".join(f"  │{'pass':>6}{'by_luck':>9}{'luck':>6}   " for _ in res))
    for j, g in enumerate(gns):
        cells = []
        for r in res.values():
            luck, ok = r["status"][j]
            luck_txt = f"{luck:.0%}" if ok is not None else "-"
            cells.append(f"  │{r['n_pass'][j]:>6}{r['by_luck'][j]:>9.1f}{luck_txt:>6} {_mark(ok)}")
        logger.info(f"  {g:>7}" + "".join(cells))
    firsts = ", ".join(f"SA={sa}: {_gn_txt(first_ok(gns, r['status']))}" for sa, r in res.items())
    logger.info(f"\n  first ✅ ── {firsts}")


def _gn_txt(g) -> str:
    return str(g) if g is not None else "-"


def log_summary(gns, firsts_by_tf: dict) -> None:
    """First ✅ of every SELL_AFTER per timeframe, the GROUP_N of each timeframe (the laxest of them) and the global
    one (the laxest of every timeframe and SELL_AFTER)."""
    sas = cr.SELL_AFTER
    w = max(8, max(len(f"SA={sa}") for sa in sas) + 2)
    logger.info(f"\n{SEP}")
    logger.info(f"  SUMMARY ── first ✅ of every SELL_AFTER, GROUP_N = the laxest of them")
    logger.info(f"{SEP}")
    logger.info(f"  {'TF':<6}" + "".join(f"{f'SA={sa}':>{w}}" for sa in sas) + f"  │{'GROUP_N':>9}")
    for tf, firsts in firsts_by_tf.items():
        cells = "".join(f"{(_gn_txt(firsts[sa]) if sa in firsts else 'n/c'):>{w}}" for sa in sas)
        ok = [g for g in firsts.values() if g is not None]
        logger.info(f"  {tf:<6}{cells}  │{_gn_txt(min(ok) if ok else None):>9}")
    logger.info(f"{SEP}")
    logger.info(f"  pass: pairs passing in >= GROUP_N symbols. by_luck: how many would with no edge (their two nulls "
                f"at the measured rate). luck = by_luck / pass, ✅ if <= LUCK_MAX={LUCK_MAX:.0%}. -: nothing passes "
                f"or no ✅ in [{gns[0]}, {gns[-1]}]. n/c: no complete cache")
    every = [g for firsts in firsts_by_tf.values() for g in firsts.values() if g is not None]
    if every:
        logger.info(f"\n→ GROUP_N: {min(every)} (the laxest of every timeframe and SELL_AFTER; the screen stage uses one GROUP_N "
                    f"for all of them)")
    else:
        logger.info(f"\n→ No timeframe and SELL_AFTER with ✅ in [{gns[0]}, {gns[-1]}]")
    logger.info(f"{SEP}")


def log_run_config(tfs: list) -> None:
    logger.info(f"\n{SEP}")
    logger.info("  GROUP_N CALIBRATION START")
    logger.info(f"{SEP}")
    logger.info(f"  {'DATASET':<{LBL_W}}: {cr.DATASET} ── {tfs}")
    logger.info(f"  {'MODE':<{LBL_W}}: {cr.MODE} NULL_PCT={cr.NULL_PCT}")
    logger.info(f"  {'SELL_AFTER':<{LBL_W}}: {cr.SELL_AFTER}")
    logger.info(f"  {'GROUP_N':<{LBL_W}}: {list(GN_RANGE)}")
    logger.info(f"  {'DECISION':<{LBL_W}}: LUCK_MAX={LUCK_MAX:.0%}")
    logger.info(f"  {'EXCLUDE':<{LBL_W}}: {', '.join(cr.EXCLUDE) if cr.EXCLUDE else 'none'}")
    logger.info(f"{SEP}\n")


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