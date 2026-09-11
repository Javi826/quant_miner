# # data_pipeline/steps/step5_highlow.py
# =============================================================================
# Step 5 — High/Low Timestamps — finds exact intrabar timestamp of high and low
# for each bar of the higher timeframe using a lower timeframe as reference.
# Supports multiple pairs: TIMEFRAMES_HIGHLOW = [["4H","1H"], ["1H","15m"]]
# =============================================================================
import logging
import os

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("pipeline.step5")

# =============================================================================
# UTILITIES
# =============================================================================

def _parse_filename(filename: str) -> tuple[str, str]:
    stem  = os.path.splitext(filename)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ""


def _read_parquet(filepath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(filepath)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
        return None


def _write_parquet(df: pd.DataFrame, filepath: str) -> None:
    df_out = df.reset_index()
    if "index" in df_out.columns:
        df_out = df_out.rename(columns={"index": "timestamp"})
    df_out.to_parquet(filepath, index=False)


def _write_csv(df: pd.DataFrame, filepath: str) -> None:
    df_out = df.reset_index()
    if "index" in df_out.columns:
        df_out = df_out.rename(columns={"index": "timestamp"})
    df_out.to_csv(filepath, index=False)

# =============================================================================
# CORE
# =============================================================================

def _find_timestamp_extremum___(df_high: pd.DataFrame, df_low: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df_high.copy()
    df = df.loc[df_low.index[0]:]
    df["low_time"]  = pd.NaT
    df["high_time"] = pd.NaT

    for i in tqdm(range(len(df) - 1), desc=f"{symbol}", leave=False):
        start    = df.index[i]
        end      = df.index[i + 1]
        intrabar = df_low[(df_low.index >= start) & (df_low.index < end)]
        if intrabar.empty:
            continue
        try:
            df.loc[start, "high_time"] = intrabar["high"].idxmax()
            df.loc[start, "low_time"]  = intrabar["low"].idxmin()
        except Exception as e:
            logger.debug(f"  ⚠ [{symbol}] Error at {start}: {e}")

    df    = df.iloc[:-1]
    valid = df[["low_time", "high_time"]].notna().all(axis=1).sum()
    total = len(df)
    pct   = valid / total * 100 if total > 0 else 0
    logger.info(f"  [{symbol}] Valid rows: {valid}/{total} ({pct:.1f}%)")

    return df

def _find_timestamp_extremum(df_high: pd.DataFrame, df_low: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df_high.copy()
    df = df.loc[df_low.index[0]:]
    
    if len(df) < 2:
        logger.info(f"  ⚠ [{symbol}] Not enough bars for high/low computation. Skipping.")
        return df

    bar_starts = df.index[:-1]
    bar_ends   = df.index[1:]
    bins       = bar_starts.append(bar_ends[-1:])

    df_low         = df_low.copy()
    df_low["bar"]  = pd.cut(df_low.index, bins=bins, labels=bar_starts, right=False)
    df_low         = df_low.dropna(subset=["bar"])

    high_times = df_low.groupby("bar", observed=True)["high"].idxmax()
    low_times  = df_low.groupby("bar", observed=True)["low"].idxmin()

    df             = df.iloc[:-1]
    df["high_time"] = high_times.reindex(df.index)
    df["low_time"]  = low_times.reindex(df.index)

    valid = df[["low_time", "high_time"]].notna().all(axis=1).sum()
    total = len(df)
    pct   = valid / total * 100 if total > 0 else 0
    logger.info(f"  [{symbol}] Valid rows: {valid}/{total} ({pct:.1f}%)")

    return df
# =============================================================================
# PAIR PROCESSOR
# =============================================================================

def _process_pair(
    tf_high: str,
    tf_low: str,
    file_index: dict,
    output_dir: str,
    export_csv: bool,
) -> int:
    """Processes a single [higher_tf, intrabar_tf] pair. Returns error count."""
    symbols = sorted({sym for sym, _ in file_index})
    logger.info(f"  Pair {tf_high} → {tf_low} | {len(symbols)} symbol(s)")
    errors = 0

    for sym in symbols:
        key_high = (sym, tf_high)
        key_low  = (sym, tf_low)

        if key_high not in file_index:
            logger.warning(f"  ⚠ [{sym}] Missing {tf_high} file. Skipping.")
            continue
        if key_low not in file_index:
            logger.warning(f"  ⚠ [{sym}] Missing {tf_low} file. Skipping.")
            continue

        df_high = _read_parquet(file_index[key_high])
        df_low  = _read_parquet(file_index[key_low])

        if df_high is None or df_low is None:
            errors += 1
            continue

        try:
            df_result = _find_timestamp_extremum(df_high, df_low, sym)
            out_name  = os.path.basename(file_index[key_high])
            out_path  = os.path.join(output_dir, out_name)
            _write_parquet(df_result, out_path)
            if export_csv:
                _write_csv(df_result, os.path.join(output_dir, os.path.splitext(out_name)[0] + ".csv"))
            logger.info(f"  💾 [{sym}] Saved → {out_name}")
        except Exception as e:
            logger.warning(f"  ❌ [{sym}] Failed: {e}")
            errors += 1

    return errors
MIN_CANDLES_1DUTC = 365
# =============================================================================
# RUN
# =============================================================================

def run(config: dict) -> bool:
    input_dir: str   = config["clean_dir"]
    output_dir: str  = config["highlow_dir"]
    tf_pairs: list   = config["timeframes_highlow"]
    export_csv: bool = config.get("export_csv", False)
    os.makedirs(output_dir, exist_ok=True)

    if tf_pairs and not isinstance(tf_pairs[0], list):
        tf_pairs = [tf_pairs]

    selected_symbols = config.get("selected_symbols") or []
    files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".parquet")
        and (not selected_symbols or any(f.startswith(s) for s in selected_symbols))
    ] if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    file_index: dict[tuple[str, str], str] = {}
    for f in files:
        sym, tf = _parse_filename(f)
        if sym and tf:
            file_index[(sym, tf)] = os.path.join(input_dir, f)

    logger.info(f"📊 High/Low timestamps — {len(tf_pairs)} pair(s)")

    total_errors = 0
    for pair in tf_pairs:
        if len(pair) != 2:
            logger.warning(f"⚠ Invalid pair {pair} — must be [higher_tf, intrabar_tf]. Skipping.")
            continue
        total_errors += _process_pair(pair[0], pair[1], file_index, output_dir, export_csv)

    logger.info("✅ High/Low timestamps complete")

    # Filter symbols with fewer than MIN_CANDLES_1DUTC daily candles
    removed_symbols = []
    for filename in os.listdir(output_dir):
        if not filename.endswith(".parquet"):
            continue
        sym, tf = _parse_filename(filename)
        if tf != "1Dutc":
            continue
        try:
            df = pd.read_parquet(os.path.join(output_dir, filename))
            if len(df) < MIN_CANDLES_1DUTC:
                removed_symbols.append(sym)
        except Exception:
            continue

    if removed_symbols:
        for filename in os.listdir(output_dir):
            if not filename.endswith(".parquet"):
                continue
            sym, _ = _parse_filename(filename)
            if sym in removed_symbols:
                os.remove(os.path.join(output_dir, filename))
        config["removed_symbols"] = removed_symbols
        logger.info(f"\n🗑 Symbols removed (< {MIN_CANDLES_1DUTC} daily candles): {len(removed_symbols)}")
        for sym in sorted(removed_symbols):
            logger.info(f"   - {sym}")
    else:
        logger.info(f"✅ All symbols passed minimum candle filter ({MIN_CANDLES_1DUTC} days)")

    if total_errors:
        logger.warning(f"⚠ Step 5 completed with {total_errors} error(s)")
        return False

    return True

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "clean_dir":          os.path.join(_base, "data", "02_clean"),
        "highlow_dir":        os.path.join(_base, "data", "03_highlow"),
        "timeframes_highlow": [["4H", "1H"], ["1H", "15m"]],
        "export_csv":         False,
    }
    run(_config)