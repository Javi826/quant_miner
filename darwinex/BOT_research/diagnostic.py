#quant_miner/darwinex/BOT_research/diag_pairs.py (forex)
"""
Diagnostic of phase 2: are signals credited through a pair only because of the filter's own edge?

Uses the cache written by main_screen.py (same config) plus the real bins and targets.

Level A: for every (passing pair, symbol) it decomposes the winning rule S AND F into S alone and F alone,
         on the same candles and the same target:
           T              cached T of the pair (z)
           cov_S, cov_F   fraction of candles inside each segment
           e_pair         raw edge of S AND F
           e_S, e_F       raw edge of each segment alone
           F/pair         e_F / e_pair. Near or above 1 with high cov_S: S does nothing.
         Sanity: z of e_pair with the pair's pilot (same shifts as main_screen) must equal the cached T.
         Cost per pair: one pilot (N_PILOTS shifts).
Level B (definitive): symmetric null shifting the SIGNAL (filter and targets stay aligned), with its own pilot,
         so T_S (z against that null) is not the same number as T. A pair credits its signal in a symbol only if
         it passes in main_screen and T_S also beats that floor.
         Cost per pair: one pilot plus one null, as one pair of phase 2 in main_screen.
"""
import os
import sys
import glob
import time
import pickle
import logging

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import main_screen as ms                                   # also sets sys.path and logging

from indicators.indicators_pool import CANDIDATE_REGISTRY, build_flat_instances
from indicators.screening_engine import (MIN_SEG, NCUT, pair_T, pair_null_distribution, pair_edge_moments,
                                         pair_name)

logger = logging.getLogger("BOT_research.diag_pairs")
logger.setLevel(logging.INFO)

# =============================================================================
# CONFIG
# =============================================================================
CACHE_FILE      = None      # None: newest screen_{DATASET}_{TIMEFRAME}_*.pkl matching the current config
RUN_LEVEL_B     = True
CHECK_ALL_PAIRS = False     # False: level B only on pairs that credit a signal NOT selected alone
N_NULL_S        = ms.N_NULL_PATHS
SEED_S          = ms.SEED + 1
RATIO_DECOR     = 0.9       # level A flag: e_F >= RATIO_DECOR * e_pair


# =============================================================================
# HELPERS
# =============================================================================
def find_cache(names, n):
    """Newest cache file whose meta matches the current config (the hash key cannot be rebuilt from
    here: this script is itself a project module and would change it)."""
    if CACHE_FILE:
        files = [CACHE_FILE]
    else:
        pat = os.path.join(ms.CACHE_DIR, f"screen_{ms.DATASET}_{ms.TIMEFRAME}_*.pkl")
        files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    for f in files:
        with open(f, "rb") as fh:
            raw = pickle.load(fh)
        m = raw.get("meta", {})
        if (raw.get("names") == names and m.get("symbols") == list(ms.SYMBOLS) and m.get("n_bars") == n
                and m.get("tp_pct") == list(ms.TP_PCT) and m.get("sl_pct") == list(ms.SL_PCT)
                and m.get("sell_after") == list(ms.SELL_AFTER) and m.get("run_phase2")
                and m.get("statistic") == "z" and m.get("n_null_paths") == ms.N_NULL_PATHS
                and m.get("n_pilots") == ms.N_PILOTS and m.get("min_pilot_n") == ms.MIN_PILOT_N):
            return f, raw
    raise FileNotFoundError("No cache matches the current config: run main_screen.py first")


def edge(seg, ok, y):
    """Same formula as _seg_edge (raw edge), on the candles in ok."""
    n_all = int(ok.sum())
    if n_all <= 1:
        return np.nan
    ya = y[ok]
    mean, sd = ya.mean(), ya.std()
    m = seg & ok
    k = int(m.sum())
    if sd <= 0.0 or k < MIN_SEG or k >= n_all:
        return np.nan
    ys = y[m]
    return (ys.mean() - mean) / (max(ys.std(), sd) / np.sqrt(k))


def f2(v, w=7):
    return f"{v:>{w}.2f}" if np.isfinite(v) else f"{'-':>{w}}"


def pct(v, w=6):
    return f"{100 * v:>{w - 1}.0f}%" if np.isfinite(v) else f"{'-':>{w}}"


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()

    # --- real data, instances, cache
    ohlcv_arr, grid, pos, _native = ms.load_aligned()
    n = len(grid)
    instances = build_flat_instances()
    by_ind = {}
    for ii, inst in enumerate(instances):
        by_ind.setdefault(inst["indicator"], []).append(ii)
    names = list(by_ind.keys())

    path, raw = find_cache(names, n)
    logger.info(f"Cache: {os.path.basename(path)} (created {raw['meta']['created']})")

    # --- same selection as main_screen
    floor1, med1, p841 = ms._null_stats(raw["null1"])
    res1 = {nm: ms._evaluate(raw["p1"][nm], floor1[k], med1[k], p841[k]) for k, nm in enumerate(names)}
    single = {nm for nm in names if int(res1[nm]["pass_sym"].sum()) >= ms.GROUP_N}
    pairs = raw["pairs"]
    res2 = {p: ms._evaluate(raw["p2"][p], *ms._null_stats(raw["p2"][p]["null_sym"])) for p in pairs}
    pair_multi = [p for p in pairs if int(res2[p]["pass_sym"].sum()) >= ms.GROUP_N]

    role = {nm: CANDIDATE_REGISTRY[nm]["role"] for nm in names}
    sig_now = [nm for nm in names if role[nm] == "signal"
               and (nm in single or any(p[0] == nm for p in pair_multi))]
    via_only = [nm for nm in sig_now if nm not in single]
    logger.info(f"GROUP_N={ms.GROUP_N}  passing pairs: {len(pair_multi)}  "
                f"SYMBOL_SIGNAL now: {len(sig_now)}  of them only via a pair: {len(via_only)}")

    # --- real targets and bins
    logger.info("Computing real targets and bins...")
    t1 = time.time()
    Y = ms.build_targets(ohlcv_arr, pos, n)
    bins, _cuts, ncv, _empty = ms.build_bins(ohlcv_arr, pos, n, instances)
    logger.info(f"  done in {time.time() - t1:.0f}s")

    # =========================================================================
    # LEVEL A: decomposition S / F / S AND F
    # =========================================================================
    logger.info(f"\n{'=' * 124}")
    logger.info("LEVEL A: every passing (pair, symbol). F/pair >= "
                f"{RATIO_DECOR} with high cov_S -> the signal does nothing")
    logger.info("=" * 124)
    logger.info(f"{'pair (signal + filter)':<48}{'sym':<8}{'T':>7}{'e_pair':>8}{'e_S':>7}{'e_F':>7}"
                f"{'F/pair':>8}{'cov_S':>7}{'cov_F':>7}  {'S_alone':<8}{'F_alone':<8}{'decor':<5}")
    decor = {}                       # pair -> (flagged symbols, passing symbols)
    max_dev = 0.0
    _shifts_main, pilot_main = ms.phase2_shifts(n)
    for p in pair_multi:
        sn, fn = p
        r = res2[p]
        n_flag = 0
        n_rows = 0
        bS = np.ascontiguousarray(bins[by_ind[sn]])
        bF = np.ascontiguousarray(bins[by_ind[fn]])
        z_mu, z_sd = pair_edge_moments(bS, bF, ncv[by_ind[sn]], ncv[by_ind[fn]], Y, pilot_main,
                                        float(ms.MIN_PILOT_N))
        shape = (bS.shape[0], bF.shape[0], len(ms.TARGETS), NCUT, 2, NCUT, 2)
        for s in range(len(ms.SYMBOLS)):
            if not r["pass_sym"][s]:
                continue
            iS, iF, g, cs, sS, cf, sF = r["win_sym"][s]
            flat = int(np.ravel_multi_index(r["win_sym"][s], shape))
            rowS = bins[by_ind[sn][iS], s]
            rowF = bins[by_ind[fn][iF], s]
            y = Y[s, :, g]
            ok = (rowS >= 0) & (rowF >= 0) & np.isfinite(y)
            segS = ms._seg_mask(rowS, cs, sS)
            segF = ms._seg_mask(rowF, cf, sF)
            e_pair = edge(segS & segF, ok, y)
            e_S = edge(segS, ok, y)
            e_F = edge(segF, ok, y)
            n_ok = max(int(ok.sum()), 1)
            cov_S = (segS & ok).sum() / n_ok
            cov_F = (segF & ok).sum() / n_ok
            ratio = e_F / e_pair if np.isfinite(e_F) and e_pair > 0 else np.nan
            flag = bool(np.isfinite(ratio) and ratio >= RATIO_DECOR)
            n_flag += flag
            n_rows += 1
            z_pair = (e_pair - z_mu[flat, s]) / z_sd[flat, s]
            max_dev = max(max_dev, abs(z_pair - r["T_sym"][s])) if np.isfinite(z_pair) else np.inf
            logger.info(f"{pair_name(sn, fn):<48}{ms.SYMBOLS[s]:<8}{f2(r['T_sym'][s])}{f2(e_pair, 8)}"
                        f"{f2(e_S)}{f2(e_F)}{f2(ratio, 8)}{pct(cov_S, 7)}{pct(cov_F, 7)}  "
                        f"{'yes' if res1[sn]['pass_sym'][s] else 'no':<8}"
                        f"{'yes' if res1[fn]['pass_sym'][s] else 'no':<8}{'<<' if flag else '':<5}")
        decor[p] = (n_flag, n_rows)
        del z_mu, z_sd
    logger.info(f"\nSanity: max |z(e_pair) - cached T| = {max_dev:.2e} (must be ~0)")

    logger.info("\nSignals selected only via a pair: symbols where S looks decorative / passing symbols")
    for nm in via_only:
        parts = [f"{p[1]} {decor[p][0]}/{decor[p][1]}" for p in pair_multi if p[0] == nm]
        logger.info(f"  {nm:<26} " + ", ".join(parts))
    if not via_only:
        logger.info("  (none)")

    # =========================================================================
    # LEVEL B: symmetric null, signal shifted
    # =========================================================================
    if not RUN_LEVEL_B:
        logger.info(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
        return

    check = pair_multi if CHECK_ALL_PAIRS else [p for p in pair_multi if p[0] in via_only]
    rng = np.random.default_rng(SEED_S)
    shifts = rng.integers(ms.L_SHIFT, n - ms.L_SHIFT, size=N_NULL_S, endpoint=True).astype(np.int64)
    pilot_S = ms.pilot_shifts(n, shifts)             # own pilot, disjoint from this null's shifts

    logger.info(f"\n{'=' * 124}")
    logger.info(f"LEVEL B: null shifting the SIGNAL ({N_NULL_S} shifts + {len(pilot_S)} pilot) on {len(check)} pairs")
    logger.info("=" * 124)
    logger.info(f"{'pair (signal + filter)':<48}{'n_pass F-null':>14}{'n_pass both':>13}"
                f"{'med T-floorF':>14}{'med TS-floorS':>15}  {'credits S':<9}")
    credits_S = {}
    for k, p in enumerate(check):
        sn, fn = p
        t1 = time.time()
        bS = np.ascontiguousarray(bins[by_ind[sn]])
        bF = np.ascontiguousarray(bins[by_ind[fn]])
        # pair_T shifts its SECOND argument: passing (F, S) shifts the signal
        cF, cS = ncv[by_ind[fn]], ncv[by_ind[sn]]
        zS_mu, zS_sd = pair_edge_moments(bF, bS, cF, cS, Y, pilot_S, float(ms.MIN_PILOT_N))
        T_S, _a = pair_T(bF, bS, cF, cS, Y, 0, zS_mu, zS_sd)
        null_S = pair_null_distribution(bF, bS, cF, cS, Y, shifts, zS_mu, zS_sd)
        del zS_mu, zS_sd
        with np.errstate(invalid="ignore"):
            floor_S = np.percentile(null_S, ms.NULL_PCT, axis=0)
        r = res2[p]
        both = r["pass_sym"] & (T_S > floor_S)
        n_f, n_b = int(r["pass_sym"].sum()), int(both.sum())
        m = r["pass_sym"]
        d_f = float(np.median((r["T_sym"] - r["floor_sym"])[m])) if m.any() else np.nan
        d_s = float(np.median((T_S - floor_S)[m])) if m.any() else np.nan
        credits_S[p] = n_b >= ms.GROUP_N
        logger.info(f"{pair_name(sn, fn):<48}{n_f:>14}{n_b:>13}{f2(d_f, 14)}{f2(d_s, 15)}  "
                    f"{'yes' if credits_S[p] else 'NO':<9}   ({time.time() - t1:.0f}s, {k + 1}/{len(check)})")

    # --- corrected signal list
    sig_new = [nm for nm in sig_now
               if nm in single or any(credits_S.get(p, p[0] in single) for p in pair_multi if p[0] == nm)]
    dropped = [nm for nm in sig_now if nm not in sig_new]
    logger.info(f"\nSYMBOL_SIGNAL now: {len(sig_now)}   with symmetric null: {len(sig_new)}   "
                f"dropped: {len(dropped)}")
    for nm in dropped:
        logger.info(f"  - {nm}")
    logger.info(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()