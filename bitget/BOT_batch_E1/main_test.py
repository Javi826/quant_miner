# signals/lookahead_audit.py
"""
Self-contained lookahead-bias audit for signal generation and backtest alignment.

Standalone run (synthetic data, no external files needed):
    python signals/lookahead_audit.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

from signals.condition_bank import ConditionBank
from signals.signal_builder import build_signal_fn, describe_rule

logger = logging.getLogger("signals.lookahead_audit")

# =============================================================================
# CONFIG
# =============================================================================

N_BARS          = 3000
SEED            = 7
BAR_FREQ        = "4h"
CUT_FRACTIONS   = (0.55, 0.75, 0.92)
PERTURB_SIGMA   = 0.35
PRICE_RTOL      = 1e-5

ENTRY_PRINT_N   = 5
ENTRY_PRINT_PAD = 3

BACKTEST_MODULE = "shared_batchs.backtesters.ZX_compute_BT"
BACKTEST_PARAMS = {"sell_after": 20, "tp_pct": 5.0, "sl_pct": 4.0, "order_amount": 1000.0}
AUDIT_SYMBOL    = "SYNTH"

# =============================================================================
# REPORT
# =============================================================================

@dataclass
class AuditIssue:
    check:  str
    target: str
    detail: str


CHECK_LABELS = {
    "prefix_invariance":   "Prefix invariance (truncated series gives identical past results)",
    "future_perturbation": "Future perturbation (randomizing future prices doesn't change the past)",
    "signal_alignment":    "Signal shift alignment (live vs backtest signal, 1-bar shift, prev-bar condition)",
    "trade_alignment":     "Trade alignment vs Cython backtester (fills, exit index, TP/SL prices)",
}

@dataclass
class AuditReport:
    issues:      list = field(default_factory=list)
    checks_run:  int  = 0
    per_check:   dict = field(default_factory=dict)  # check_name -> assertions run
    skipped:     list = field(default_factory=list)

    def record(self, check: str, n: int = 1) -> None:
        self.checks_run += n
        self.per_check[check] = self.per_check.get(check, 0) + n

    def add(self, check: str, target: str, detail: str) -> None:
        self.issues.append(AuditIssue(check, target, detail))

    def skip(self, check: str, reason: str) -> None:
        self.skipped.append((check, reason))

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self) -> None:
        by_check: dict = {}
        for issue in self.issues:
            by_check.setdefault(issue.check, []).append(issue)
        skipped_names = {check for check, _ in self.skipped}

        logger.info(f"{'─' * 100}")
        logger.info(f"  LOOKAHEAD AUDIT — {self.checks_run} assertions")
        logger.info(f"{'─' * 100}")

        for check, n_assertions in self.per_check.items():
            label = CHECK_LABELS.get(check, check)
            if check in by_check:
                logger.info(f"  🔴 FAILED   {check:<20} {n_assertions:>5} assertions   {label}")
                for issue in by_check[check]:
                    logger.info(f"       {issue.target:<40} {issue.detail}")
            else:
                logger.info(f"  🟢 PASSED   {check:<20} {n_assertions:>5} assertions   {label}")

        for check, reason in self.skipped:
            label = CHECK_LABELS.get(check, check)
            logger.info(f"  ⚪ SKIPPED  {check:<20} {'':>5}              {label} — {reason}")

        logger.info(f"{'─' * 100}")
        if self.ok:
            logger.info("  RESULT: no lookahead detected")
        else:
            logger.info(f"  RESULT: {len(self.issues)} issue(s) found — see FAILED checks above")
        logger.info(f"{'─' * 100}")

# =============================================================================
# SYNTHETIC DATA
# =============================================================================


def make_synthetic_ohlcv(n_bars: int = N_BARS, seed: int = SEED) -> dict:
    rng     = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.012, n_bars)
    close   = 100.0 * np.exp(np.cumsum(log_ret))

    open_     = np.empty(n_bars, dtype=np.float64)
    open_[0]  = close[0] * 0.999
    open_[1:] = close[:-1]

    body_high = np.maximum(open_, close)
    body_low  = np.minimum(open_, close)
    high      = body_high * (1.0 + np.abs(rng.normal(0.0, 0.004, n_bars)))
    low       = body_low  * (1.0 - np.abs(rng.normal(0.0, 0.004, n_bars)))

    ts     = pd.date_range("2020-01-01", periods=n_bars, freq=BAR_FREQ).to_numpy("datetime64[ns]")
    ts_int = ts.view("int64")
    bar_ns = int(ts_int[1] - ts_int[0])

    return {
        "ts":        ts,
        "open":      open_,
        "high":      high,
        "low":       low,
        "close":     close,
        "high_time": ts_int + rng.integers(1, bar_ns, n_bars),
        "low_time":  ts_int + rng.integers(1, bar_ns, n_bars),
    }


def _truncate(arr: dict, cut: int) -> dict:
    return {key: values[:cut] for key, values in arr.items()}


def _perturb_future(arr: dict, cut: int, rng: np.random.Generator) -> dict:
    out     = {key: np.array(values, copy=True) for key, values in arr.items()}
    n_tail  = len(out["close"]) - cut
    factors = np.clip(1.0 + rng.normal(0.0, PERTURB_SIGMA, n_tail), 0.1, None)
    for key in ("open", "high", "low", "close"):
        out[key][cut:] = out[key][cut:] * factors
    return out

# =============================================================================
# SPEC HELPERS
# =============================================================================


def pick_demo_rule(specs: list, max_conditions: int = 2) -> list:
    rule:  list = []
    types: set  = set()
    for spec in specs:
        if spec["type"] not in types:
            rule.append(spec)
            types.add(spec["type"])
        if len(rule) == max_conditions:
            break
    return rule

def _rule_mask(bank: ConditionBank, rule_specs: list) -> np.ndarray:
    mask = np.ones(bank.n, dtype=bool)
    for spec in rule_specs:
        mask &= bank.evaluate(spec)
    return mask

# =============================================================================
# CHECK 1 — PREFIX INVARIANCE
# =============================================================================
def check_prefix_invariance(arr: dict, specs: list, report: AuditReport) -> None:
    bank_full  = ConditionBank(arr)
    full_masks = [bank_full.evaluate(spec) for spec in specs]
    n_bars     = len(arr["close"])

    for fraction in CUT_FRACTIONS:
        cut      = int(n_bars * fraction)
        bank_cut = ConditionBank(_truncate(arr, cut))
        for spec, full_mask in zip(specs, full_masks):
            report.record("prefix_invariance")
            diff = np.flatnonzero(full_mask[:cut] != bank_cut.evaluate(spec)[:cut])
            if diff.size:
                report.add(
                    "prefix_invariance", bank_full.describe(spec),
                    f"cut={cut} mismatches={diff.size} first_idx={int(diff[0])}",
                )

# =============================================================================
# CHECK 2 — FUTURE PERTURBATION
# =============================================================================


def check_future_perturbation(arr: dict, specs: list, report: AuditReport) -> None:
    rng        = np.random.default_rng(SEED + 1)
    bank_full  = ConditionBank(arr)
    full_masks = [bank_full.evaluate(spec) for spec in specs]
    n_bars     = len(arr["close"])

    for fraction in CUT_FRACTIONS:
        cut            = int(n_bars * fraction)
        bank_perturbed = ConditionBank(_perturb_future(arr, cut, rng))
        for spec, full_mask in zip(specs, full_masks):
            report.record("future_perturbation")
            diff = np.flatnonzero(full_mask[:cut] != bank_perturbed.evaluate(spec)[:cut])
            if diff.size:
                report.add(
                    "future_perturbation", bank_full.describe(spec),
                    f"cut={cut} mismatches={diff.size} first_idx={int(diff[0])}",
                )

# =============================================================================
# CHECK 3 — SIGNAL SHIFT ALIGNMENT
# =============================================================================


def check_signal_alignment(arr: dict, rule_specs: list, side: str, report: AuditReport) -> np.ndarray:
    bank      = ConditionBank(arr)
    mask      = _rule_mask(bank, rule_specs)
    signal_fn = build_signal_fn(rule_specs, side)

    signal_live = signal_fn(arr, live_trading=True)
    signal_bt   = signal_fn(arr, live_trading=False)
    expected    = np.where(mask, 1 if side == "long" else -1, 0).astype(np.int32)

    report.record("signal_alignment")
    if not np.array_equal(signal_live, expected):
        report.add("signal_alignment", "live_mask_match", "live signal differs from evaluated rule mask")

    report.record("signal_alignment")
    if signal_bt[0] != 0:
        report.add("signal_alignment", "first_bar_zero", f"signal_bt[0]={int(signal_bt[0])}")

    report.record("signal_alignment")
    if not np.array_equal(signal_bt[1:], signal_live[:-1]):
        report.add("signal_alignment", "shift_by_one", "signal_bt is not signal_live shifted by 1 bar")

    report.record("signal_alignment")
    fired    = np.flatnonzero(signal_bt != 0)
    unbacked = fired[~mask[fired - 1]] if fired.size else np.array([], dtype=np.int64)
    if unbacked.size:
        report.add(
            "signal_alignment", "condition_on_previous_bar",
            f"fired_without_prev_bar_condition={unbacked.size} first_idx={int(unbacked[0])}",
        )

    return signal_bt


# =============================================================================
# CHECK 4 — TRADE ALIGNMENT (optional, requires the backtest module)
# =============================================================================


def _load_backtest_runner():
    try:
        module = __import__(BACKTEST_MODULE, fromlist=["run_grid_backtest"])
        return getattr(module, "run_grid_backtest")
    except Exception as exc:
        logger.debug(f"backtest runner import failed: {exc!r}")
        return None


def check_trade_alignment(
    arr: dict, rule_specs: list, signal_bt: np.ndarray, report: AuditReport
) -> pd.DataFrame | None:
    runner = _load_backtest_runner()
    if runner is None:
        report.skip("trade_alignment", f"{BACKTEST_MODULE} not importable")
        return None

    ohlcv_arrays = {AUDIT_SYMBOL: {**arr, "signal": signal_bt}}
    result       = runner(
        ohlcv_arrays,
        sell_after   = BACKTEST_PARAMS["sell_after"],
        tp_pct       = BACKTEST_PARAMS["tp_pct"],
        sl_pct       = BACKTEST_PARAMS["sl_pct"],
        order_amount = BACKTEST_PARAMS["order_amount"],
    )
    trade_log = result["__PORTFOLIO__"]["trade_log"]
    if trade_log.empty:
        report.skip("trade_alignment", "no trades generated")
        return trade_log

    bank      = ConditionBank(arr)
    mask      = _rule_mask(bank, rule_specs)
    ts_int    = arr["ts"].view("int64")
    buy_idxs  = np.searchsorted(ts_int, trade_log["buy_time"].to_numpy().view("int64"))
    sell_idxs = np.searchsorted(ts_int, trade_log["sell_time"].to_numpy().view("int64"))

    for row, buy_idx, sell_idx in zip(trade_log.itertuples(index=False), buy_idxs, sell_idxs):
        label = f"trade@{buy_idx}"

        report.record("trade_alignment")
        if not np.isclose(row.buy_price, arr["open"][buy_idx], rtol=PRICE_RTOL):
            report.add(
                "trade_alignment", label,
                f"buy_price={row.buy_price:.8f} != open[{buy_idx}]={arr['open'][buy_idx]:.8f}",
            )

        report.record("trade_alignment")
        if buy_idx == 0 or not mask[buy_idx - 1]:
            report.add("trade_alignment", label, f"rule not True at buy_idx-1={buy_idx - 1}")

        report.record("trade_alignment")
        if sell_idx < buy_idx:
            report.add("trade_alignment", label, f"sell_idx={sell_idx} before buy_idx={buy_idx}")

        report.record("trade_alignment")
        if row.exit_reason == "SELL_AFTER":
            expected_idx = min(buy_idx + BACKTEST_PARAMS["sell_after"], len(ts_int) - 1)
            if sell_idx != expected_idx:
                report.add("trade_alignment", label, f"sell_idx={sell_idx} != expected={expected_idx}")
            elif not np.isclose(row.sell_price, arr["open"][sell_idx], rtol=PRICE_RTOL):
                report.add("trade_alignment", label, "SELL_AFTER exit not filled at bar open")

        report.record("trade_alignment")
        if row.exit_reason in ("TP", "SL"):
            pct    = BACKTEST_PARAMS["tp_pct"] if row.exit_reason == "TP" else -BACKTEST_PARAMS["sl_pct"]
            sign   = -1.0 if row.position_type == "SHORT" else 1.0
            target = row.buy_price * (1.0 + sign * pct / 100.0)
            if not np.isclose(row.sell_price, target, rtol=PRICE_RTOL):
                report.add(
                    "trade_alignment", label,
                    f"{row.exit_reason} fill={row.sell_price:.8f} != target={target:.8f}",
                )

    logger.info(f"  ℹ️  trade_alignment verified {len(trade_log)} real trades from the Cython backtester")
    return trade_log
# =============================================================================
# MANUAL INSPECTION — bars around entries
# =============================================================================
def print_bars_around_entries(
    arr: dict, rule_specs: list, signal_bt: np.ndarray,
    n_entries: int = ENTRY_PRINT_N, pad: int = ENTRY_PRINT_PAD,
) -> None:
    bank        = ConditionBank(arr)
    mask        = _rule_mask(bank, rule_specs)
    cond_masks  = {bank.describe(spec): bank.evaluate(spec) for spec in rule_specs}
    entry_idxs  = np.flatnonzero(signal_bt != 0)[:n_entries]
    n_bars      = len(arr["close"])

    logger.info(f"\n  RULE: {describe_rule(bank, rule_specs)}")
    for entry_idx in entry_idxs:
        lo = max(entry_idx - pad, 0)
        hi = min(entry_idx + pad + 1, n_bars)
        logger.info(f"\n  ── ENTRY idx={entry_idx}  fill=open[{entry_idx}]={arr['open'][entry_idx]:.6f}")
        header = (
            f"    {'idx':>6} {'timestamp':<20} {'open':>11} {'high':>11} {'low':>11} {'close':>11} "
            f"{'rule':>6} {'signal':>7}   conditions"
        )
        logger.info(header)
        for i in range(lo, hi):
            flags = " ".join(
                f"{name}={'T' if values[i] else 'F'}" for name, values in cond_masks.items()
            )
            marker = " <== entry bar" if i == entry_idx else ""
            logger.info(
                f"    {i:>6} {str(arr['ts'][i])[:19]:<20} {arr['open'][i]:>11.6f} {arr['high'][i]:>11.6f} "
                f"{arr['low'][i]:>11.6f} {arr['close'][i]:>11.6f} "
                f"{('T' if mask[i] else 'F'):>6} {int(signal_bt[i]):>7}   {flags}{marker}"
            )
# =============================================================================
# ENTRY POINT
# =============================================================================

def run_audit(
    arr: dict | None = None,
    rule_specs: list | None = None,
    side: str = "long",
    with_backtest: bool = True,
    with_entry_dump: bool = True,
) -> AuditReport:
    report = AuditReport()
    arr    = arr if arr is not None else make_synthetic_ohlcv()
    bank   = ConditionBank(arr)
    specs  = bank.build_condition_specs()

    if not specs:
        report.skip("all_checks", "INDICATOR_REGISTRY produced no specs")
        report.render()
        return report

    rule_specs = rule_specs if rule_specs is not None else pick_demo_rule(specs)

    check_prefix_invariance(arr, specs, report)
    check_future_perturbation(arr, specs, report)
    signal_bt = check_signal_alignment(arr, rule_specs, side, report)

    if with_backtest:
        check_trade_alignment(arr, rule_specs, signal_bt, report)
    else:
        report.skip("trade_alignment", "disabled by caller")

    report.render()

    if with_entry_dump:
        print_bars_around_entries(arr, rule_specs, signal_bt)

    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    report = run_audit()
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())