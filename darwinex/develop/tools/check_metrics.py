#BOT_batch/check_metrics.py
import os
import sys
import time
import struct
import argparse
import warnings

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import backtesting_fx as fx  # sets sys.path, logging and configuration; its main block does not run on import

import pipeline.backtest_runner as br
from pipeline.spec_table import build_spec_table, rule_spec_index
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from utils import batch_metrics as bm

DEFAULT_ARGS = "--n-rules 2000"   # used when no arguments arrive (Spyder Run)
LINE         = "=" * 110


# =============================================================================
# NUMPY REFERENCE (the batch_metrics formulas before metrics_core, kept here only to verify it)
# =============================================================================
def ref_sharpe(daily_values: np.ndarray) -> float:
    n          = daily_values.size
    daily_mean = np.add.reduce(daily_values, axis=None) / n
    dev        = daily_values - daily_mean
    np.multiply(dev, dev, out=dev)
    daily_std  = np.sqrt(np.add.reduce(dev, axis=None) / n)
    sharpe = (round(float(daily_mean / daily_std * np.sqrt(fx.settings.DAYS_PER_YEAR)), 3)
              if daily_std > 0 else np.nan)
    if sharpe is not None and np.isfinite(sharpe) and abs(sharpe) > bm.SHARPE_ABS_CAP:
        sharpe = np.nan
    return sharpe


def ref_skew_kurtosis(daily_values: np.ndarray) -> tuple:
    deviations = daily_values - daily_values.mean()
    dev_sq = deviations * deviations
    m2 = np.mean(dev_sq)
    m3 = np.mean(dev_sq * deviations)
    m4 = np.mean(dev_sq * dev_sq)
    return float(m3 / (m2 ** 1.5)), float(m4 / (m2 ** 2))


def ref_equity(daily_values: np.ndarray, capital: float) -> tuple:
    eq       = capital + np.cumsum(daily_values)
    cm       = np.maximum.accumulate(eq)
    max_dd   = ((eq - cm) / cm * 100).min()
    net_gain = (eq[-1] - capital) / capital * 100
    return eq, max_dd, net_gain


# =============================================================================
# COMPARISON
# =============================================================================
def _same(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)):
        return (np.isnan(a) and np.isnan(b)) or struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


def _check_series(x: np.ndarray, capital: float, diffs: list, label: str) -> None:
    if not _same(ref_sharpe(x.copy()), bm.sharpe_from_daily_values(x.copy())):
        diffs.append(f"{label}: sharpe")
    if x.size > 2 and not _same(ref_skew_kurtosis(x.copy()), bm.skew_kurtosis_from_daily_values(x.copy())):
        diffs.append(f"{label}: skew/kurtosis")
    ref, new = ref_equity(x.copy(), capital), bm.equity_from_daily_values(x.copy(), capital)
    if not _same(ref, new):
        # the only accepted difference: the sign of a zero drawdown when the equity is negative (see metrics_core)
        if not (_same(ref[0], new[0]) and _same(ref[2], new[2]) and ref[1] == new[1] == 0 and (ref[0] < 0).any()):
            diffs.append(f"{label}: equity / max_dd / net_gain")


# =============================================================================
# SYNTHETIC SERIES
# =============================================================================
def _synthetic(n_series: int, seed: int) -> tuple:
    rng   = np.random.default_rng(seed)
    sizes = [1, 2, 3, 7, 8, 9, 127, 128, 129, 2081, 8191, 8192, 8193, 20000]
    diffs = []
    for i in tqdm(range(n_series), desc="CHECK SYNTHETIC ", dynamic_ncols=True):
        n    = int(rng.choice(sizes)) if i % 3 == 0 else int(rng.integers(1, 4000))
        kind = i % 5
        if kind == 0:
            x = rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4)
        elif kind == 1:
            x = np.where(rng.random(n) < 0.85, 0.0, rng.standard_normal(n) * 40)
        elif kind == 2:
            x = np.full(n, float(rng.standard_normal()))
        elif kind == 3:
            x = rng.standard_normal(n) * 1e-3 + 5e-3
        else:
            x = np.round(rng.standard_normal(n) * 100, 2)
        _check_series(x, float(br.INITIAL_BALANCE), diffs, f"synthetic #{i} (n={n})")
    return n_series, diffs


# =============================================================================
# REAL BACKTEST SERIES (every valid combo of a sample of rules)
# =============================================================================
def _real(combo: dict, n_rules: int) -> tuple:
    timeframe = combo["timeframe"]
    _, arr_by_combo = fx.load_ohlcv_by_combo([combo], fx.DATA_FOLDER_BY_DATASET[fx.DATASET_IS], fx.DATASET_IS)
    ohlcv_arr  = arr_by_combo[combo["combo_key"]]
    templates  = build_rule_templates(ohlcv_arr, indicators=fx.SELECTED_INDICATORS_BY_TIMEFRAME[timeframe],
                                      max_depth=fx.RULE_MAX_DEPTH)
    rules      = build_rule_dicts(templates, combo["combo_key"], timeframe)
    sample     = np.unique(np.linspace(0, len(rules) - 1, min(n_rules, len(rules))).round().astype(np.int64))
    rules      = [rules[i] for i in sample]
    spec_table = build_spec_table(rules=rules, ohlcv_arr=ohlcv_arr, timeframe=timeframe, show_progress=False)

    static_bundle = br.prepare_static_arrays(ohlcv_arr)
    static_bundle["calendar"] = br._trading_calendar(ohlcv_arr, br._global_day_grid(ohlcv_arr)[0])
    _, n_days_range = br._global_day_grid(ohlcv_arr)
    spec_idx, sides = rule_spec_index(rules, spec_table.index_by_identity)
    engine_grid     = br._engine_grid(br._combo_grid(fx.PARAM_GRID_BY_TIMEFRAME[timeframe]))
    capital         = float(br.INITIAL_BALANCE)

    n_series, diffs = 0, []
    for r in tqdm(range(len(rules)), desc=f"CHECK REAL      {combo['combo_key']}", dynamic_ncols=True):
        events = br.build_rule_events_from_words(
            spec_table.words, spec_table.word_offsets, spec_idx[r], bool(sides[r] != "long"),
            static_bundle["ts_int_2d"], static_bundle["sym_len"],
            static_bundle["tick_pos_2d"], static_bundle["all_timestamps_int"], static_bundle["idx_workspace"],
        )
        if events[0].shape[0] < br.BACKTEST_MIN_TRADES:
            continue
        n_trades, day_start, n_days, _, daily, _ = br.backtest_grid(
            br.market_arrays(static_bundle), *events, *engine_grid,
            capital, float(br.COMISION) / 100.0, float(fx.ORDER_AMOUNT), br.BACKTEST_MIN_TRADES,
            *static_bundle["calendar"], n_days_range,
        )
        rows = bm.sharpes_from_daily_rows(daily, day_start, n_days)
        for c in range(n_trades.shape[0]):
            if n_days[c] <= 0:
                continue
            x = daily[c, day_start[c]:day_start[c] + n_days[c]]
            if not _same(ref_sharpe(x.copy()), rows[c]):
                diffs.append(f"{rules[r]['rule_id']} combo {c}: sharpe by rows")
            _check_series(x, capital, diffs, f"{rules[r]['rule_id']} combo {c}")
            n_series += 1
    return n_series, diffs


# =============================================================================
# MAIN
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="metrics_core vs the numpy formulas: synthetic series and real backtest series")
    p.add_argument("--combo", default=None, help="combo_key for the real series (default: every combo of build_combos())")
    p.add_argument("--n-rules", type=int, default=2000, help="rules per combo for the real series (0 = skip real)")
    p.add_argument("--n-synthetic", type=int, default=20000, help="synthetic series (0 = skip)")
    argv = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ARGS.split()
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    combos = fx.build_combos()
    if args.combo is not None:
        combos = [c for c in combos if c["combo_key"] == args.combo]
        if not combos:
            raise SystemExit(f"unknown combo_key: {args.combo}. Available: {[c['combo_key'] for c in fx.build_combos()]}")

    print(f"\n{LINE}")
    print(f" CHECK METRICS | numpy {np.__version__} | synthetic {args.n_synthetic:,} | real {args.n_rules:,} rules x "
          f"{len(combos) if args.n_rules else 0} combo(s)")
    print(LINE)

    t0, rows = time.perf_counter(), []
    if args.n_synthetic:
        rows.append(("synthetic", *_synthetic(args.n_synthetic, seed=7)))
    if args.n_rules:
        rows += [(c["combo_key"], *_real(c, args.n_rules)) for c in combos]

    print(f"\n {'source':<14}{'series':>12}   result")
    for name, n_series, diffs in rows:
        print(f" {name:<14}{n_series:>12,}   {'identical' if not diffs else f'DIFFERENT ── {len(diffs):,} ── ' + diffs[0]}")
    failed = [name for name, _, diffs in rows if diffs]
    print(f"\n {'ALL IDENTICAL' if not failed else f'DIFFERENT: {failed}'} ── {time.perf_counter() - t0:,.0f} s")
    print(LINE)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()