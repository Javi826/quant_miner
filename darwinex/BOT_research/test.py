#BOT_research/test_screen.py (forex)
# Checks for main_screen.py. Uses the same config (symbols, timeframe, TP/SL/SA, NULL_PCT...) and does not modify it.
#   Test 1  LOOKAHEAD  every indicator instance, computed with the data up to candle t, must give the same value
#                      at t as with the full series.
#   Test 2  SYNTHETIC  prices with no predictable direction (random sign flip of every candle, same sign in every
#                      symbol) through the screening: only ~(100 - NULL_PCT)% should beat the floor by chance.

import warnings
import numpy as np

import main_screen as ms
from synthetic import make_synthetic
from indicators.indicators_pool import CANDIDATE_REGISTRY, build_flat_instances, instance_key

# =============================================================================
# CONFIG
# =============================================================================
LOOKAHEAD_SYMBOLS = ["EURUSD", "USDJPY", "EURGBP"]
LOOKAHEAD_CUTS    = 20      # candles t checked per instance and symbol
LOOKAHEAD_WARMUP  = 500     # first candle that can be a cut

SYN_SEEDS         = [1, 2, 3]   # phase 1 runs once per seed
SYN_PAIRS         = 100         # phase 2: random signal x filter pairs (first seed only)
SYN_NULL_PHASE2   = 100         # phase 2: null shifts per pair

OK, KO = "✅", "❌"


# =============================================================================
# TEST 1: LOOKAHEAD
# =============================================================================
def _cut(arr, end):
    """Every per-candle field of a symbol, truncated to the first `end` candles."""
    n = len(arr["close"])
    out = {}
    for k, v in arr.items():
        if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n:
            out[k] = v[:end]
        else:
            out[k] = v
    return out


def _same(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))


def test_lookahead(ohlcv_arr):
    instances = build_flat_instances()
    print(f"\nTEST 1: LOOKAHEAD ({len(instances)} instances, {len(LOOKAHEAD_SYMBOLS)} symbols, "
          f"{LOOKAHEAD_CUTS} cuts each)")
    bad = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for inst in instances:
            fn = CANDIDATE_REGISTRY[inst["indicator"]]["fn"]
            key = instance_key(inst["indicator"], inst["params"])
            for sym in LOOKAHEAD_SYMBOLS:
                arr = ohlcv_arr[sym]
                n = len(arr["close"])
                full = np.asarray(fn(arr, None, inst["params"]), dtype=np.float64)
                cuts = np.linspace(min(LOOKAHEAD_WARMUP, n - 2), n - 2, LOOKAHEAD_CUTS).astype(int)
                n_diff, max_diff, first = 0, 0.0, None
                for t in cuts:
                    part = np.asarray(fn(_cut(arr, t + 1), None, inst["params"]), dtype=np.float64)
                    if not _same(full[t], part[t]):
                        n_diff += 1
                        max_diff = max(max_diff, abs(full[t] - part[t]) if np.isfinite(full[t] - part[t]) else np.inf)
                        first = t if first is None else first
                if n_diff:
                    bad.append((key, sym, n_diff, max_diff, first))

    if not bad:
        print(f"{OK} No indicator uses future data.")
        return True
    print(f"{KO} {len(bad)} instance/symbol combinations use future data:")
    for key, sym, n_diff, max_diff, first in bad:
        diff = "NaN on one side" if np.isinf(max_diff) else f"{max_diff:.4g}"
        print(f"   {key} ({sym}): {n_diff}/{LOOKAHEAD_CUTS} cuts differ, max diff {diff}, first at candle {first}")
    return False


# =============================================================================
# TEST 2: SYNTHETIC PRICES WITHOUT SIGNAL
# =============================================================================
def _check(label, rate, n):
    """Rate of beating the floor vs chance: fail if clearly above (~p99 of a binomial with n trials)."""
    p = 1.0 - ms.NULL_PCT / 100.0
    limit = p + 2.33 * np.sqrt(p * (1.0 - p) / n)
    ok = rate <= limit
    print(f"{OK if ok else KO} {label}: beat the floor {100 * rate:.1f}% "
          f"(expected ~{100 * p:.0f}%, limit {100 * limit:.1f}%)")
    return ok


def test_synthetic(ohlcv_arr, pos, n):
    print(f"\nTEST 2: SYNTHETIC PRICES WITHOUT SIGNAL (phase 1: {len(SYN_SEEDS)} seeds; "
          f"phase 2: {SYN_PAIRS} pairs, {SYN_NULL_PHASE2} null shifts)")
    instances = build_flat_instances()
    by_ind = {}
    for ii, inst in enumerate(instances):
        by_ind.setdefault(inst["indicator"], []).append(ii)
    names = list(by_ind.keys())
    rng = np.random.default_rng(ms.SEED)
    shifts = rng.integers(ms.L_SHIFT, n - ms.L_SHIFT, size=SYN_NULL_PHASE2, endpoint=True).astype(np.int64)
    bounds = np.linspace(0, n, ms.N_BLOCKS + 1).astype(int)
    blocks = [(int(bounds[b]), int(bounds[b + 1])) for b in range(ms.N_BLOCKS)]

    # Phase 1 floor: built once on the real data, as the screening does, and reused for every seed.
    print(f"   building the phase 1 floor ({ms.N_PATHS} paths)...")
    floor_sym, med_sym, p84_sym = ms.build_null_floors(ohlcv_arr, pos, n, instances, names, by_ind)

    ok = True
    over_sym, pass_sym = [], []
    p_over_sym, p_pass_sym = [], []
    for k, seed in enumerate(SYN_SEEDS):
        syn = make_synthetic(ohlcv_arr, seed, ms.SYMBOLS)
        Y = ms.build_targets(syn, pos, n)
        bins, cuts, ncv, _ = ms.build_bins(syn, pos, n, instances)

        for j, nm in enumerate(names):
            r = ms.screen_indicator(np.ascontiguousarray(bins[by_ind[nm]]), ncv[by_ind[nm]], Y,
                                    floor_sym[j], med_sym[j], p84_sym[j], blocks)
            over_sym.append(r["over_sym"].mean())
            pass_sym.append(r["pass_sym"].mean())
            ms._progress(f"Phase 1, seed {seed}", j + 1, len(names))

        if k == 0 and ms.RUN_PHASE2:
            signals = [nm for nm in names if CANDIDATE_REGISTRY[nm]["role"] == "signal"]
            filters = [nm for nm in names if CANDIDATE_REGISTRY[nm]["role"] == "filter"]
            all_pairs = [(sn, fn) for sn in signals for fn in filters]
            idx = np.random.default_rng(seed).choice(len(all_pairs), size=min(SYN_PAIRS, len(all_pairs)),
                                                     replace=False)
            for j, i in enumerate(idx):
                sn, fn = all_pairs[i]
                r = ms.screen_pair(np.ascontiguousarray(bins[by_ind[sn]]), np.ascontiguousarray(bins[by_ind[fn]]),
                                   ncv[by_ind[sn]], ncv[by_ind[fn]], Y, shifts, blocks)
                p_over_sym.append(r["over_sym"].mean())
                p_pass_sym.append(r["pass_sym"].mean())
                ms._progress(f"Phase 2, seed {seed}", j + 1, len(idx))

    print(f"\nPhase 1 ({len(names)} indicators x {len(SYN_SEEDS)} seeds):")
    ok &= _check("per symbol", float(np.mean(over_sym)), len(over_sym))
    print(f"   pass (with stability): {100 * np.mean(pass_sym):.1f}%")
    if ms.RUN_PHASE2:
        print(f"\nPhase 2 ({len(p_over_sym)} pairs):")
        ok &= _check("per symbol", float(np.mean(p_over_sym)), len(p_over_sym))
        print(f"   pass (with stability): {100 * np.mean(p_pass_sym):.1f}%")
    return ok


# =============================================================================
# MAIN
# =============================================================================
def main():
    ohlcv_arr, grid, pos, _ = ms.load_aligned()
    ok1 = test_lookahead(ohlcv_arr)
    ok2 = test_synthetic(ohlcv_arr, pos, len(grid))
    print(f"\n{OK if ok1 and ok2 else KO} "
          f"{'All tests passed.' if ok1 and ok2 else 'Some tests failed, see above.'}")


if __name__ == "__main__":
    main()