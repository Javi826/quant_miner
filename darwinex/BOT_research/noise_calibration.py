# quant_miner/darwinex/BOT_research/check_calibration.py
"""Calibration of the screening nulls: on data with no signal, how much passes by chance, in YPY and in NPY.

The data is the real data of main_screen (DATASET, TIMEFRAME, SYMBOLS) with the direction of every candle flipped at
random, as a phase 1 null path: same candles, volatility and cross-symbol structure, but nothing can predict the
direction. With a well calibrated null, per symbol:
  alone: ~(100 - NULL_PCT)% of (indicator, symbol) above the floor, in both modes
  pairs: at most that (the pair null is conservative)
A clear excess in one mode (e.g. 25% where 15% is expected) means its null lets noise through.
Every run is saved as it finishes: changing only SETTINGS, or rerunning after a stop, reuses them.
"""
import os
import sys
import time
import logging
import pickle
import dataclasses
from math import comb

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import main_screen as ms                                   # its config and data loader (it also sets sys.path)

from indicators.indicators_pool import CANDIDATE_REGISTRY, GROUP_NAMES, build_flat_instances, instance_key
from indicators.screen_engine import IndicatorPool, align_symbols, make_synthetic, compute_raw, save_cache
from indicators.screen_report import evaluate1, evaluate2, report_selection

CALIB_PCT  = 85                                            # NULL_PCT of the runs: SETTINGS use this or higher
SETTINGS   = [(85, 3), (85, 6), (90, 5), (95, 3), (98, 2)] # (NULL_PCT, GROUP_N), all evaluated on the same runs
N_IND      = 12                                            # random indicators (fixed draw): N_IND*(N_IND-1)/2 pairs
INDICATORS = None                                          # or an explicit list of indicators (N_IND is ignored)
N_REPS     = 2                                             # sign-flipped copies of the data: one full run per mode each
RUN_MODES  = ("YPY", "NPY")
SEED       = 900_000                                       # first copy (must not be a null or pilot path)

logging.getLogger("indicators.screen_report").setLevel(logging.WARNING)   # report_selection: counts only, no tables


# =============================================================================
# RUNS
# =============================================================================
def cfg_of(mode):
    """main_screen's config, with the calibration NULL_PCT and this mode."""
    return dataclasses.replace(ms.CFG, null_pct=CALIB_PCT, mode=mode)


def sub_pool():
    """Pool of the calibration indicators: INDICATORS, or N_IND drawn at random (always the same draw)."""
    full = IndicatorPool(CANDIDATE_REGISTRY, build_flat_instances(), instance_key, GROUP_NAMES)
    if INDICATORS is not None:
        unknown = sorted(set(INDICATORS) - set(full.names))
        if unknown:
            raise ValueError(f"INDICATORS has unknown indicators: {unknown}")
        keep = set(INDICATORS)
    else:
        keep = {str(nm) for nm in np.random.default_rng(0).choice(full.names, min(N_IND, len(full.names)),
                                                                   replace=False)}
    return IndicatorPool(CANDIDATE_REGISTRY, [i for i in full.instances if i["indicator"] in keep],
                         instance_key, GROUP_NAMES)


def check_config():
    for pct, g in SETTINGS:
        if pct < CALIB_PCT:
            raise ValueError(f"SETTINGS has NULL_PCT={pct} < CALIB_PCT={CALIB_PCT}")
        if not 0 < g <= len(ms.SYMBOLS):
            raise ValueError(f"SETTINGS has GROUP_N={g} outside [1, {len(ms.SYMBOLS)}]")
    for mode in RUN_MODES:
        cfg = cfg_of(mode)
        for seed in range(SEED, SEED + N_REPS):
            if (cfg.seed_null <= seed < cfg.seed_null + cfg.n_null_paths
                    or cfg.seed_pilot <= seed < cfg.seed_pilot + cfg.n_pilots):
                raise ValueError(f"SEED {seed} is a null or pilot path of the engine: change SEED")


def runs(pool):
    """{(rep, mode): raw results}: loaded from the saved file when it matches, the missing ones computed and saved."""
    path = os.path.join(ms.CACHE_DIR, f"calib_{ms.DATASET}_{ms.TIMEFRAME}.pkl")
    ident = (ms.DATASET, ms.TIMEFRAME, tuple(ms.SYMBOLS), tuple(pool.names), SEED, CALIB_PCT,
             tuple(cfg_of(m).identity() for m in RUN_MODES))
    saved = {}
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if obj.get("ident") == ident:
            saved = obj["raws"]
            print(f"Saved runs loaded: {len(saved)} ({os.path.basename(path)})")
        else:
            print(f"Saved runs are for another configuration: recomputing ({os.path.basename(path)})")
    real = None
    for rep in range(N_REPS):
        for mode in RUN_MODES:
            if (rep, mode) in saved:
                continue
            if real is None:
                real = ms.load_data()
            t0 = time.time()
            print(f"\n--- copy {rep + 1}/{N_REPS}, {mode}")
            data = align_symbols(make_synthetic(real, SEED + rep, ms.SYMBOLS), ms.SYMBOLS)
            saved[(rep, mode)] = compute_raw(data, pool, cfg_of(mode))
            save_cache(path, {"ident": ident, "raws": saved})
            print(f"copy {rep + 1}/{N_REPS}, {mode}: {(time.time() - t0) / 60:.1f} min")
    return {k: saved[k] for k in saved if k[0] < N_REPS and k[1] in RUN_MODES}


# =============================================================================
# REPORT
# =============================================================================
def binom_tail(n, p, g):
    """P(at least g of n independent symbols pass), each with probability p."""
    return sum(comb(n, k) * p ** k * (1.0 - p) ** (n - k) for k in range(g, n + 1))


def ok(q, g):
    """As select: n_pass >= GROUP_N and score >= MIN_SCORE."""
    s = q["score"]
    return q["n_pass"] >= g and (ms.MIN_SCORE is None or (np.isfinite(s) and s >= ms.MIN_SCORE))


def pct_txt(k, n):
    return f"{100 * k / n:.1f}% ({k}/{n})" if n else "-"


def report(pool, raws):
    names = pool.names
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    reps = sorted({r for r, _m in raws})
    n_sym = len(ms.SYMBOLS)
    pcts = sorted({p for p, _g in SETTINGS})
    ev1 = {(r, m, p): {nm: evaluate1(raws[(r, m)]["p1"][nm], p) for nm in names} for (r, m) in raws for p in pcts}
    ev2 = {(r, m, p): {pr: evaluate2(raws[(r, m)]["p2"][pr], p) for pr in pairs} for (r, m) in raws for p in pcts}
    any_raw = next(iter(raws.values()))
    cfg = cfg_of(RUN_MODES[0])

    print(f"\n{'=' * 110}\nCALIBRATION {ms.DATASET} {ms.TIMEFRAME}: {n_sym} symbols, {any_raw['n']} common candles, "
          f"{len(names)} indicators ({len(pairs)} pairs), {len(reps)} sign-flipped copies, "
          f"N_NULL_PATHS={cfg.n_null_paths}, N_PILOTS={cfg.n_pilots}\n{'=' * 110}")
    print("Indicators: " + ", ".join(names))

    head = "".join(f"{m:>24}" for m in RUN_MODES)
    print(f"\nABOVE THE FLOOR, per symbol (expected: alone ~(100 - NULL_PCT)%, pairs at most that)")
    print(f"{'':<24}{'expected':>10}{head}")
    for kind, ev, items in (("alone", ev1, names), ("pairs", ev2, pairs)):
        for p in pcts:
            row = ""
            for m in RUN_MODES:
                k = sum(int(ev[(r, m, p)][x]["pass"].sum()) for r in reps if (r, m) in raws for x in items)
                n = sum(n_sym * len(items) for r in reps if (r, m) in raws)
                row += f"{pct_txt(k, n):>24}"
            exp = f"{'' if kind == 'alone' else '<='}{100 - p:.0f}%"
            print(f"{kind + ' NULL_PCT=' + str(p):<24}{exp:>10}{row}")

    score_txt = "off" if ms.MIN_SCORE is None else ms.MIN_SCORE
    print(f"\nPASSING BY CHANCE (n_pass >= GROUP_N and score >= MIN_SCORE={score_txt}), % of alones and of pairs")
    print(f"{'NULL_PCT/GROUP_N':<24}{'binomial*':>10}" + "".join(f"{m + ' alone':>24}" for m in RUN_MODES)
          + "".join(f"{m + ' pairs':>24}" for m in RUN_MODES))
    for p, g in SETTINGS:
        row = ""
        for kind, ev, items in (("alone", ev1, names), ("pairs", ev2, pairs)):
            for m in RUN_MODES:
                k = sum(int(ok(ev[(r, m, p)][x], g)) for r in reps if (r, m) in raws for x in items)
                n = sum(len(items) for r in reps if (r, m) in raws)
                row += f"{pct_txt(k, n):>24}"
        print(f"{f'{p}/{g}':<24}{100 * binom_tail(n_sym, 1 - p / 100, g):>9.1f}%{row}")
    print("* n_pass >= GROUP_N of one alone if the symbols were independent (no MIN_SCORE): the gap to the "
          "measured values\n  comes from the correlation between symbols")

    print(f"\nSELECTED BY CHANCE, out of {len(names)} noise indicators (mean per copy; after redundancy in brackets)")
    print(f"{'NULL_PCT/GROUP_N':<24}" + "".join(f"{m:>16}" for m in RUN_MODES))
    for p, g in SETTINGS:
        row = ""
        for m in RUN_MODES:
            sel, kept = [], []
            for r in reps:
                if (r, m) not in raws:
                    continue
                out = report_selection(raws[(r, m)], pool, p, g, ms.PHI_TH, ms.MIN_COVER, ms.MAX_COVER,
                                       ms.MIN_SCORE, m)
                sel.append(len(out["selected"]))
                kept.append(len(out["pruned"]))
            row += f"{np.mean(sel):>9.1f} ({np.mean(kept):4.1f})" if sel else f"{'-':>16}"
        print(f"{f'{p}/{g}':<24}{row}")

    n_tests = n_sym * len(reps)
    print(f"\nALONE ABOVE THE FLOOR AT NULL_PCT={CALIB_PCT}, per indicator "
          f"(expected ~{100 - CALIB_PCT}% of {n_tests} = {n_tests * (100 - CALIB_PCT) / 100:.1f})")
    print(f"{'indicator':<40}" + "".join(f"{m:>8}" for m in RUN_MODES))
    for nm in names:
        row = "".join(f"{sum(int(ev1[(r, m, CALIB_PCT)][nm]['pass'].sum()) for r in reps if (r, m) in raws):>8}"
                      for m in RUN_MODES)
        print(f"{nm:<40}{row}")


def main():
    check_config()
    pool = sub_pool()
    t0 = time.time()
    raws = runs(pool)
    report(pool, raws)
    print(f"\nTOTAL: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()