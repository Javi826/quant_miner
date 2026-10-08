#BOT_batch/check_ypy.py
import os
import sys
import time
import argparse

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import backtesting_fx as fx  # sets sys.path, logging and configuration; its main block does not run on import

import pipeline.backtest_runner as br
from pipeline.spec_table import build_spec_table, rule_spec_index
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from signals.indicators_bank import ConditionBank
from signals.signal_builder import build_signal_fn
from backtesters import ZX_compute_BT_YPY as ypy

DEFAULT_ARGS = "--n-rules 1000"   # used when no arguments arrive (Spyder Run)
LINE         = "=" * 110
DTYPE        = np.float32


# =============================================================================
# ARGS / DATA
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="YPY: rule events from the spec table vs from signal arrays, on real data")
    p.add_argument("--combo", default=None, help="combo_key (default: every combo of build_combos())")
    p.add_argument("--n-rules", type=int, default=1000, help="rules per combo, evenly spaced (0 = all)")
    p.add_argument("--no-engine", action="store_true", help="compare only events and timeline, skip backtest_grid")
    argv = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ARGS.split()
    return p.parse_args(argv)


def _combos(combo_key):
    combos = fx.build_combos()
    if combo_key is None:
        return combos
    selected = [c for c in combos if c["combo_key"] == combo_key]
    if not selected:
        raise SystemExit(f"unknown combo_key: {combo_key}. Available: {[c['combo_key'] for c in combos]}")
    return selected


def _load_ohlcv(combo):
    _, arr_by_combo = fx.load_ohlcv_by_combo([combo], fx.DATA_FOLDER_BY_DATASET[fx.DATASET_IS], fx.DATASET_IS)
    return arr_by_combo[combo["combo_key"]]


def _sample(n_total, n):
    if n <= 0 or n >= n_total:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, n).round().astype(np.int64))


# =============================================================================
# COMPARISON
# =============================================================================
def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def _events_diff(old, new):
    for name, a, b in zip(("signal_events", "ev_short", "timeline"), old, new):
        if not _same(a, b):
            return f"{name} differs (old {np.asarray(a).shape}, new {np.asarray(b).shape})"
    return None


def _engine_diff(old, new, min_trades):
    # Only the parts the runner reads: the daily and duration rows are np.empty outside the written span.
    n_trades_a, start_a, n_days_a, nonzero_a, daily_a, duration_a = old
    n_trades_b, start_b, n_days_b, nonzero_b, daily_b, duration_b = new
    for name, a, b in (("n_trades", n_trades_a, n_trades_b), ("day_start", start_a, start_b),
                       ("n_days", n_days_a, n_days_b), ("nonzero", nonzero_a, nonzero_b)):
        if not _same(a, b):
            return f"{name} differs"
    for c in range(n_trades_a.shape[0]):
        n = int(n_trades_a[c])
        if not _same(duration_a[c, :n], duration_b[c, :n]):
            return f"duration differs in combo {c}"
        if n == 0 or n < min_trades:
            continue
        lo, hi = int(start_a[c]), int(start_a[c]) + int(n_days_a[c])
        if not _same(daily_a[c, lo:hi], daily_b[c, lo:hi]):
            return f"daily values differ in combo {c}"
    return None


def _check_combo(combo, n_rules, run_engine) -> dict:
    timeframe  = combo["timeframe"]
    ohlcv_arr  = _load_ohlcv(combo)
    templates  = build_rule_templates(ohlcv_arr, indicators=fx.SELECTED_INDICATORS_BY_TIMEFRAME[timeframe],
                                      max_depth=fx.RULE_MAX_DEPTH)
    rules      = build_rule_dicts(templates, combo["combo_key"], timeframe)
    spec_table = build_spec_table(rules=rules, ohlcv_arr=ohlcv_arr, timeframe=timeframe)

    static_bundle = ypy.prepare_static_arrays(ohlcv_arr)
    br.check_spec_table_layout(spec_table.symbols, spec_table.n_bars, static_bundle)
    banks           = {sym: ConditionBank(arr) for sym, arr in ohlcv_arr.items()}
    spec_idx, sides = rule_spec_index(rules, spec_table.index_by_identity)
    is_short        = sides != "long"
    event_args      = (static_bundle["ts_int_2d"], static_bundle["sym_len"], static_bundle["tick_pos_2d"],
                       static_bundle["all_timestamps_int"], static_bundle["idx_workspace"])

    global_start, n_days_range = br._global_day_grid(ohlcv_arr)
    calendar    = br._trading_calendar(ohlcv_arr, global_start)
    engine_grid = br._engine_grid(br._combo_grid(fx.PARAM_GRID_BY_TIMEFRAME[timeframe]))
    market      = ypy.market_arrays(static_bundle)
    min_trades  = br.BACKTEST_MIN_TRADES
    engine_args = (float(br.INITIAL_BALANCE), float(br.COMISION) / 100.0, float(fx.ORDER_AMOUNT), min_trades,
                   *calendar, n_days_range)

    result = {"rules": 0, "events": 0, "engine": 0, "diff": None}
    for r in tqdm(_sample(len(rules), n_rules), desc=f"CHECK YPY       {combo['combo_key']}", dynamic_ncols=True):
        signal_fn = build_signal_fn(rules[r]["specs"], rules[r]["side"])
        signals   = tuple(
            np.ascontiguousarray(signal_fn(ohlcv_arr[sym], live_trading=False, bank=banks[sym]), dtype=DTYPE)
            for sym in static_bundle["symbols_by_sid"]
        )
        old = ypy.build_rule_events(signals, *event_args)
        new = ypy.build_rule_events_from_words(spec_table.words, spec_table.word_offsets, spec_idx[r],
                                               bool(is_short[r]), *event_args)
        result["rules"]  += 1
        result["events"] += int(old[0].shape[0])

        diff = _events_diff(old, new)
        if diff is None and run_engine and old[0].shape[0] >= min_trades:
            diff = _engine_diff(ypy.backtest_grid(market, *old, *engine_grid, *engine_args),
                                ypy.backtest_grid(market, *new, *engine_grid, *engine_args), min_trades)
            result["engine"] += 1
        if diff is not None:
            result["diff"] = f"rule {rules[r]['rule_id']}: {diff}"
            break
    return result


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args   = _parse_args()
    combos = _combos(args.combo)
    print(f"\n{LINE}")
    print(f" CHECK YPY | {len(combos)} combo(s) | {'all' if args.n_rules <= 0 else args.n_rules} rules/combo | "
          f"engine {'off' if args.no_engine else 'on'}")
    print(LINE)

    rows, t0 = [], time.perf_counter()
    for combo in combos:
        rows.append((combo["combo_key"], _check_combo(combo, args.n_rules, not args.no_engine)))

    print(f"\n {'combo':<12}{'rules':>8}{'events':>12}{'engine':>9}   result")
    for key, r in rows:
        status = "identical" if r["diff"] is None else f"DIFFERENT ── {r['diff']}"
        print(f" {key:<12}{r['rules']:>8,}{r['events']:>12,}{r['engine']:>9,}   {status}")
    failed = [key for key, r in rows if r["diff"] is not None]
    print(f"\n {'ALL IDENTICAL' if not failed else f'{len(failed)} combo(s) DIFFERENT: {failed}'} "
          f"── {time.perf_counter() - t0:,.0f} s")
    print(LINE)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()