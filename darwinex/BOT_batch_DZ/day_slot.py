# BOT_batch_BZ/explain_day_slot.py
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from utils.ohlcv_utils import prepare_ohlcv_arrays
from indicators.indicators_pool import (
    CANDIDATE_REGISTRY, SLOT_HOURS, build_flat_specs, instance_key, describe_spec, _day_slot, _bar_step_ns,
)
from signals.indicators_bank import ConditionBank, SELECTED_INDICATORS_BY_TIMEFRAME
from signals.signal_builder import build_signal_fn, describe_rule

# =============================================================================
# CONFIG
# =============================================================================
SYMBOL    = "CHFJPY"
TIMEFRAME = "1H"
DATASET   = "IS"
SLOT      = 2
SIDE      = "long"
FWD_BARS  = 10
SHOW_ROWS = 30
COMPANION = ("close_pos_in_bar", {}, ">", 0.8)

INDICATOR = "day_slot"
HOUR_NS   = 3_600_000_000_000

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)


# =============================================================================
# HELPERS
# =============================================================================
def print_section(title: str) -> None:
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")


def make_spec(indicator: str, params: dict, op: str, threshold: float) -> dict:
    return {
        "indicator": indicator,
        "params":    params,
        "key":       instance_key(indicator, params),
        "op":        op,
        "threshold": float(threshold),
    }


def load_arr(symbol: str, timeframe: str, dataset: str) -> dict:
    ohlcv_data = build_universe(
        DATA_FOLDER_BY_DATASET[dataset], {timeframe: [symbol]}, dataset=dataset,
    )[timeframe]
    sym = next(iter(ohlcv_data))
    return prepare_ohlcv_arrays({sym: ohlcv_data[sym]})[sym]


def forward_return(close: np.ndarray, bars: int, side: str) -> np.ndarray:
    out = np.full(len(close), np.nan)
    out[:-bars] = close[bars:] / close[:-bars] - 1.0
    return out if side == "long" else -out


def window_around_last_hit(mask: np.ndarray, rows: int) -> slice:
    hits   = np.flatnonzero(mask)
    center = hits[-1] if len(hits) else len(mask) - 1
    start  = max(0, center - rows // 2)
    return slice(start, min(len(mask), start + rows))


# =============================================================================
# SECTIONS
# =============================================================================
def show_data(ts: np.ndarray) -> None:
    print_section(f"1. DATA — {SYMBOL} {TIMEFRAME} {DATASET}")
    print(f"  candles : {len(ts):,}")
    print(f"  from    : {ts[0]}")
    print(f"  to      : {ts[-1]}")


def show_slot_construction(ts: np.ndarray, hour: np.ndarray, slot: np.ndarray) -> None:
    print_section("2. SLOT CONSTRUCTION — slot = open hour // slot hours")
    step_ns, step_count = _bar_step_ns(ts)
    print(f"  bar step (most common gap) : {step_ns / HOUR_NS:g} h ({step_count:,} gaps)")
    print(f"  SLOT_HOURS                 : {SLOT_HOURS}")
    print(f"  slots found                : {np.unique(slot).tolist()}\n")
    print("  open hour x slot (candle count):")
    print(pd.crosstab(pd.Series(hour, name="hour"), pd.Series(slot, name="slot")).to_string())


def show_indicator_values(arr: dict, ts: np.ndarray, hour: np.ndarray, slot: np.ndarray) -> None:
    print_section(f"3. INDICATOR VALUE — {INDICATOR}(slot={SLOT}) = 1.0 if slot == {SLOT} else 0.0")
    value = CANDIDATE_REGISTRY[INDICATOR]["fn"](arr, {}, {"slot": SLOT})
    rows  = slice(-SHOW_ROWS, None)
    df = pd.DataFrame({
        "ts":    ts[rows],
        "hour":  hour[rows],
        "slot":  slot[rows],
        "value": value[rows],
        ">0.5":  value[rows] > 0.5,
        "<0.5":  value[rows] < 0.5,
    })
    print(df.to_string(index=False, float_format=lambda x: f"{x:g}"))


def show_generated_specs(bank: ConditionBank) -> None:
    print_section(f"4. CONDITIONS GENERATED FOR THE MINER — {INDICATOR}")
    selected = INDICATOR in SELECTED_INDICATORS_BY_TIMEFRAME.get(TIMEFRAME, [])
    print(f"  selected for {TIMEFRAME}: {selected}\n")
    specs = [s for s in build_flat_specs() if s["indicator"] == INDICATOR]
    df = pd.DataFrame({
        "label":          [describe_spec(s) for s in specs],
        "pct_candles_on": [100.0 * bank.evaluate(s).mean() for s in specs],
    })
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


def show_rule_example(arr: dict, ts: np.ndarray, slot: np.ndarray, bank: ConditionBank) -> None:
    companion  = make_spec(*COMPANION)
    slot_spec  = make_spec(INDICATOR, {"slot": SLOT}, ">", 0.5)
    rule_specs = [companion, slot_spec]
    signal_fn  = build_signal_fn(rule_specs, SIDE)
    print_section(f"5. RULE EXAMPLE — {SIDE}: {describe_rule(bank, rule_specs)}")

    companion_value = CANDIDATE_REGISTRY[companion["indicator"]]["fn"](arr, {}, companion["params"])
    companion_mask  = bank.evaluate(companion)
    slot_mask       = bank.evaluate(slot_spec)
    signal_live     = signal_fn(arr, live_trading=True, bank=bank)
    signal_backtest = signal_fn(arr, live_trading=False, bank=bank)

    rows = window_around_last_hit(signal_live != 0, SHOW_ROWS)
    df = pd.DataFrame({
        "ts":                      ts[rows],
        "slot":                    slot[rows],
        companion["indicator"]:    companion_value[rows],
        describe_spec(companion):  companion_mask[rows],
        describe_spec(slot_spec):  slot_mask[rows],
        "AND":                     companion_mask[rows] & slot_mask[rows],
        "signal_live":             signal_live[rows],
        "signal_backtest(shift1)": signal_backtest[rows],
    })
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def show_rule_slicing(arr: dict, hour: np.ndarray, slot: np.ndarray, bank: ConditionBank) -> None:
    companion = make_spec(*COMPANION)
    base_mask = bank.evaluate(companion)
    fwd       = forward_return(arr["close"], FWD_BARS, SIDE) * 100.0
    n_base    = int(base_mask.sum())
    fwd_col   = f"fwd_ret_{FWD_BARS}_in (%)"
    excl_col  = f"fwd_ret_{FWD_BARS}_excl (%)"
    print_section(
        f"6. HOW {INDICATOR} SLICES A BASE RULE — base: {SIDE} {describe_spec(companion)} ({n_base:,} signals)"
    )

    rows = []
    for k in np.unique(slot):
        in_slot   = base_mask & (slot == k)
        excl_slot = base_mask & (slot != k)
        rows.append({
            "slot":        int(k),
            "hours":       ",".join(str(h) for h in np.unique(hour[slot == k])),
            "n_signals":   int(in_slot.sum()),
            "pct_of_base": 100.0 * in_slot.sum() / n_base if n_base else np.nan,
            fwd_col:       np.nanmean(fwd[in_slot]) if in_slot.any() else np.nan,
            excl_col:      np.nanmean(fwd[excl_slot]) if excl_slot.any() else np.nan,
        })

    df = pd.DataFrame(rows).round({"pct_of_base": 1, fwd_col: 4, excl_col: 4})
    print(df.to_string(index=False))
    print(f"\n  base fwd_ret_{FWD_BARS} (%) : {np.nanmean(fwd[base_mask]):.4f}")
    print(f"  {INDICATOR}_slotK>0.5 keeps the 'in' column | {INDICATOR}_slotK<0.5 keeps the 'excl' column")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    arr  = load_arr(SYMBOL, TIMEFRAME, DATASET)
    ts   = np.asarray(arr["ts"]).astype("datetime64[ns]")
    hour = pd.DatetimeIndex(ts).hour.to_numpy()
    slot = _day_slot(arr["ts"])
    bank = ConditionBank(arr, timeframe=TIMEFRAME)

    show_data(ts)
    show_slot_construction(ts, hour, slot)
    show_indicator_values(arr, ts, hour, slot)
    show_generated_specs(bank)
    show_rule_example(arr, ts, slot, bank)
    show_rule_slicing(arr, hour, slot, bank)