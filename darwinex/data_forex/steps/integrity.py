import logging
import os
import re

import pandas as pd

logger = logging.getLogger("pipeline.integrity")

# =============================================================================
# ISSUE COLLECTOR
# =============================================================================

class IssueCollector:
    """Accumulates pipeline issues across all stages for final summary."""

    def __init__(self) -> None:
        self._issues: list[dict] = []

    def add(self, symbol: str, timeframe: str, stage: str, description: str) -> None:
        self._issues.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "stage": stage,
            "description": description,
        })

    def has_issues(self) -> bool:
        return bool(self._issues)

    def all_issues(self) -> list[dict]:
        return list(self._issues)


# =============================================================================
# CONSTANTS
# =============================================================================
OHLC_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume"]


# =============================================================================
# UTILITIES
# =============================================================================

def _parse_timeframe_to_ms(tf: str) -> int:
    s = str(tf).strip().lower()
    m = re.match(r"^(\d+)(min|m|h|d|w)$", s)
    if not m:
        return 86400 * 1000
    n, u = int(m.group(1)), m.group(2)
    mapping = {"min": 60, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return n * mapping.get(u, 86400) * 1000


def _is_weekend_gap(gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> bool:
    """A gap starting on Friday and resuming on Sat/Sun/Mon is market closure, not a data issue."""
    friday = 4
    return gap_start.weekday() == friday and gap_end.weekday() in (5, 6, 0)


def _list_parquet_files(folder: str, timeframe: str = "", selected_symbols: list | None = None) -> list[str]:
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".parquet")
        and (not timeframe or f.endswith(f"_{timeframe}.parquet"))
        and (not selected_symbols or any(f.startswith(s) for s in selected_symbols))
    ])


def _symbol_from_path(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[0]


def _timeframe_from_path(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[-1]


# =============================================================================
# OHLCV CHECKS — shared by run_raw and run_clean
# =============================================================================

def _check_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    critical: bool,
    timeframe: str = "",
    stage: str = "",
    collector: IssueCollector | None = None,
) -> int:
    """
    Validates OHLCV quality.
    critical=True  → uses ❌ prefix (run_clean)
    critical=False → uses ⚠ prefix (run_raw)
    Volume zeros are logged only — never added to collector (corrected by cleaning).
    """
    issues = 0
    prefix = "❌" if critical else "⚠"

    for col in OHLC_COLS + VOLUME_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                issues += n
                logger.debug(f"  {prefix} [{symbol}] NaN in '{col}': {n} rows")
                if collector:
                    collector.add(symbol, timeframe, stage, f"NaN in '{col}': {n} rows")

    for col in OHLC_COLS:
        if col in df.columns:
            n = (df[col] == 0).sum()
            if n > 0:
                issues += n
                logger.info(f"  {prefix} [{symbol}] Zero values in '{col}': {n} rows")
                if collector:
                    collector.add(symbol, timeframe, stage, f"Zero in '{col}': {n} rows")

    for col in VOLUME_COLS:
        if col in df.columns:
            n = (df[col] == 0).sum()
            if n > 0:
                issues += n
                logger.debug(f"  {prefix} [{symbol}] Zero volume in '{col}': {n} rows")

    if all(c in df.columns for c in OHLC_COLS):
        mask = (
            (df["low"] > df["open"]) | (df["low"] > df["close"]) |
            (df["high"] < df["open"]) | (df["high"] < df["close"])
        )
        n = mask.sum()
        if n > 0:
            issues += n
            logger.debug(f"  {prefix} [{symbol}] Incoherent OHLC: {n} rows")
            if collector:
                collector.add(symbol, timeframe, stage, f"Incoherent OHLC: {n} rows")

    if issues == 0:
        logger.debug(f"  ✅ [{symbol}] Integrity passed")

    return issues


# =============================================================================
# HIGH/LOW CHECK
# =============================================================================

def _check_highlow(
    df: pd.DataFrame,
    symbol: str,
    gran_ms: int,
    timeframe: str = "",
    collector: IssueCollector | None = None,
) -> int:
    violations = 0

    if "high_time" not in df.columns or "low_time" not in df.columns:
        logger.warning(f"  ⚠ [{symbol}] Missing high_time/low_time columns")
        if collector:
            collector.add(symbol, timeframe, "highlow", "Missing high_time/low_time columns")
        return 1

    if "timestamp" not in df.columns:
        df = df.reset_index()
        if "timestamp" not in df.columns:
            logger.warning(f"  ⚠ [{symbol}] No timestamp column found")
            if collector:
                collector.add(symbol, timeframe, "highlow", "No timestamp column found")
            return 1

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["high_time"] = pd.to_datetime(df["high_time"])
    df["low_time"] = pd.to_datetime(df["low_time"])
    bar_end = df["timestamp"] + pd.Timedelta(milliseconds=gran_ms)

    bad_high = df[
        df["high_time"].notna() &
        ((df["high_time"] < df["timestamp"]) | (df["high_time"] >= bar_end))
    ]
    if not bad_high.empty:
        violations += len(bad_high)
        logger.info(f"  ⚠ [{symbol}] high_time out of bar range: {len(bad_high)} rows")
        logger.info(f"\n{bad_high[['timestamp', 'high_time']].head(5)}")
        if collector:
            collector.add(symbol, timeframe, "highlow", f"high_time out of range: {len(bad_high)} rows")

    bad_low = df[
        df["low_time"].notna() &
        ((df["low_time"] < df["timestamp"]) | (df["low_time"] >= bar_end))
    ]
    if not bad_low.empty:
        violations += len(bad_low)
        logger.info(f"  ⚠ [{symbol}] low_time out of bar range: {len(bad_low)} rows")
        logger.info(f"\n{bad_low[['timestamp', 'low_time']].head(5)}")
        if collector:
            collector.add(symbol, timeframe, "highlow", f"low_time out of range: {len(bad_low)} rows")

    if violations == 0:
        logger.info(f"  ✅ [{symbol}] High/Low integrity passed")

    return violations


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def run_raw(config: dict, collector: IssueCollector | None = None) -> bool:
    """Validates raw extracted data — diagnostic only, never aborts pipeline."""
    input_dir: str = config["raw_dir"]
    timeframe: str = config.get("timeframe", "")
    selected_symbols = config.get("selected_symbols") or []
    files = _list_parquet_files(input_dir, timeframe, selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    logger.info(f"🔍 Raw integrity check — {len(files)} file(s)")
    total = 0

    for filepath in files:
        sym = _symbol_from_path(filepath)
        tf = _timeframe_from_path(filepath)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            continue

        total += _check_ohlcv(df, sym, critical=False, timeframe=tf, stage="raw", collector=collector)

        if "timestamp" in df.columns:
            gran_ms = _parse_timeframe_to_ms(tf)
            timestamps = pd.to_datetime(df["timestamp"])
            ts_ms = timestamps.astype("int64") // 10**6
            diffs = ts_ms.diff().dropna()
            gaps = diffs[diffs > gran_ms * 1.5]
            for idx, gap_ms in gaps.items():
                gap_start = timestamps.iloc[idx - 1]
                gap_end = timestamps.iloc[idx]
                if _is_weekend_gap(gap_start, gap_end):
                    continue
                gap_days = gap_ms / (1000 * 86400)
                total += 1
                logger.debug(f"  ⚠ [{sym}] GAP: {gap_start} → {gap_end} ({gap_days:.1f} days)")
                if collector:
                    collector.add(sym, tf, "raw", f"GAP: {gap_start.date()} → {gap_end.date()} ({gap_days:.1f}d)")

    if total == 0:
        logger.info("✅ Raw integrity check passed — no issues found")
    else:
        logger.info(f"⚠ Raw integrity check found {total} issue(s) — cleaning step will follow")

    return True


def run_clean(config: dict, collector: IssueCollector | None = None) -> bool:
    """Validates cleaned data — aborts pipeline if errors found."""
    input_dir: str = config["clean_dir"]
    timeframe: str = config.get("timeframe", "")
    selected_symbols = config.get("selected_symbols") or []
    files = _list_parquet_files(input_dir, timeframe, selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"🔍 Clean integrity check — {len(files)} file(s)")
    total = 0

    for filepath in files:
        sym = _symbol_from_path(filepath)
        tf = _timeframe_from_path(filepath)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            total += 1
            continue
        total += _check_ohlcv(df, sym, critical=True, timeframe=tf, stage="clean", collector=collector)

    if total == 0:
        logger.info("✅ Clean integrity check passed")
        return True

    logger.info(f"❌ Clean integrity check FAILED — {total} critical error(s). Pipeline aborted.")
    return False


def run_highlow(config: dict, collector: IssueCollector | None = None) -> bool:
    """Validates high_time/low_time are within bar interval — diagnostic only, never aborts."""
    input_dir: str = config["highlow_dir"]
    selected_symbols = config.get("selected_symbols") or []
    files = _list_parquet_files(input_dir, selected_symbols=selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    logger.info(f"🔍 High/Low integrity check — {len(files)} file(s)")
    total = 0

    for filepath in files:
        sym = _symbol_from_path(filepath)
        tf = _timeframe_from_path(filepath)
        gran_ms = _parse_timeframe_to_ms(tf)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            continue
        total += _check_highlow(df, sym, gran_ms, timeframe=tf, collector=collector)

    if total == 0:
        logger.info("✅ High/Low integrity check passed — no violations found")
    else:
        logger.info(f"⚠ High/Low integrity check found {total} violation(s)")

    return True


def run_coverage(config: dict, collector: IssueCollector | None = None, tolerance_days: int = 5) -> bool:
    """
    Validates that all timeframes for each symbol start/end at roughly the same date.
    Uses `reference_coverage_tf` (default "4h") as reference. Diagnostic only — never aborts.
    """
    input_dir: str = config["raw_dir"]
    selected_symbols = config.get("selected_symbols") or []
    reference_tf = config.get("reference_coverage_tf", "4h")
    files = _list_parquet_files(input_dir, selected_symbols=selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    symbol_files: dict[str, dict[str, str]] = {}
    for filepath in files:
        symbol = _symbol_from_path(filepath)
        filename = os.path.basename(filepath)
        tf = os.path.splitext(filename)[0].replace(f"{symbol}_", "", 1)
        symbol_files.setdefault(symbol, {})[tf] = filepath

    logger.info(f"🔍 Coverage integrity check — {len(symbol_files)} symbol(s)")

    issues = 0
    for symbol, tf_files in sorted(symbol_files.items()):
        ref_path = tf_files.get(reference_tf)
        if not ref_path:
            logger.info(f"  ⚠ [{symbol}] No {reference_tf} reference — skipping coverage check")
            continue

        try:
            df_ref = pd.read_parquet(ref_path)
            ref_min = pd.to_datetime(df_ref["timestamp"]).min()
            ref_max = pd.to_datetime(df_ref["timestamp"]).max()
        except Exception as e:
            logger.warning(f"  ⚠ [{symbol}] Could not read {reference_tf} reference: {e}")
            continue

        for tf, filepath in sorted(tf_files.items()):
            if tf == reference_tf:
                continue
            try:
                df = pd.read_parquet(filepath)
                tf_min = pd.to_datetime(df["timestamp"]).min()
                tf_max = pd.to_datetime(df["timestamp"]).max()

                diff_start = (tf_min - ref_min).days
                if diff_start > tolerance_days:
                    issues += 1
                    logger.info(
                        f"  ⚠ [{symbol}] {tf} starts {diff_start}d after {reference_tf} "
                        f"({tf_min.date()} vs {ref_min.date()})"
                    )
                    if collector:
                        collector.add(symbol, tf, "coverage", f"starts {diff_start}d after {reference_tf} ({tf_min.date()} vs {ref_min.date()})")
                else:
                    logger.debug(f"  ✅ [{symbol}] {tf} start coverage OK (diff: {diff_start}d)")

                diff_end = abs((tf_max - ref_max).days)
                if diff_end > tolerance_days:
                    issues += 1
                    logger.info(
                        f"  ⚠ [{symbol}] {tf} ends {diff_end}d away from {reference_tf} "
                        f"({tf_max.date()} vs {ref_max.date()})"
                    )
                    if collector:
                        collector.add(symbol, tf, "coverage", f"ends {diff_end}d away from {reference_tf} ({tf_max.date()} vs {ref_max.date()})")
                else:
                    logger.debug(f"  ✅ [{symbol}] {tf} end coverage OK (diff: {diff_end}d)")

            except Exception as e:
                logger.warning(f"  ⚠ [{symbol}] Could not read {tf}: {e}")

    if issues == 0:
        logger.info("✅ Coverage integrity check passed — all timeframes aligned")
    else:
        logger.info(f"⚠ Coverage integrity check found {issues} misaligned timeframe(s)")

    return True


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(collector: IssueCollector) -> None:
    """Prints a table of all collected issues at the end of the pipeline."""
    issues = collector.all_issues()
    sep = "=" * 74
    logger.info(f"\n{sep}")
    logger.info("  PIPELINE ISSUE SUMMARY")
    logger.info(sep)

    if not issues:
        logger.info("  ✅ No issues found")
        logger.info(sep)
        return

    col_w = {"symbol": 14, "timeframe": 10, "stage": 12, "description": 34}
    header = (
        f"  {'Symbol':<{col_w['symbol']}}"
        f"{'TF':<{col_w['timeframe']}}"
        f"{'Stage':<{col_w['stage']}}"
        f"{'Issue'}"
    )
    logger.info(header)
    logger.info("  " + "-" * 72)
    for issue in issues:
        row = (
            f"  {issue['symbol']:<{col_w['symbol']}}"
            f"{issue['timeframe']:<{col_w['timeframe']}}"
            f"{issue['stage']:<{col_w['stage']}}"
            f"{issue['description']}"
        )
        logger.debug(row)
    logger.info(sep)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _config = {
        "raw_dir": os.path.join(_base, "data_fx", "01_raw"),
        "clean_dir": os.path.join(_base, "data_fx", "02_clean"),
        "highlow_dir": os.path.join(_base, "data_fx", "03_highlow"),
        "timeframe": "4h",
        "reference_coverage_tf": "4h",
    }
    _collector = IssueCollector()
    run_raw(_config, _collector)
    run_clean(_config, _collector)
    run_highlow(_config, _collector)
    run_coverage(_config, _collector)
    print_summary(_collector)