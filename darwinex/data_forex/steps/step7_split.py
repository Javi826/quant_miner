import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datetime import datetime, timezone
from mt5_source import get_client
from broker_client.broker_config import SERVER_CLOCK_SYMBOL
import pandas as pd
from dateutil.relativedelta import relativedelta
import logging


logger = logging.getLogger("pipeline.step7")

REFERENCE_TIMEFRAME = "4h"  # Used to read available data range for preview

# =============================================================================
# WINDOW CALCULATION
# =============================================================================

def _compute_windows(config: dict) -> tuple[str, str, str, str]:
    window_oos = config.get("window_oos_months", 3)
    ref_date_str = config.get("split_reference_date", None)
    start_date = config.get("start_date", "2020-01-01")

    if ref_date_str:
        ref = pd.to_datetime(ref_date_str).to_pydatetime()
    else:
        # Dataset timestamps are in broker server time (see mt5_source.py),
        # so "now" must be read from the same clock, not from local/UTC time.
        server_now = get_client().symbol_info_tick(SERVER_CLOCK_SYMBOL).time
        ref = datetime.fromtimestamp(server_now, tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    is_start = pd.to_datetime(start_date).to_pydatetime()
    oos_end = ref
    oos_start = ref - relativedelta(months=window_oos)
    is_end = oos_start

    return (
        is_start.strftime("%Y-%m-%d"),
        is_end.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )

# =============================================================================
# FOLDER NAMING
# =============================================================================

def _make_folder_name(is_start: str, is_end: str, oos_start: str, oos_end: str, subset: str) -> str:
    if subset == "IS":
        start, end = is_start[:7], is_end[:7]
    else:
        start, end = oos_start[:7], oos_end[:7]
    return f"fx_{start}_{end}_{subset}"

# =============================================================================
# DATA RANGE READER — for preview
# =============================================================================

def _get_data_range(highlow_dir: str, reference_symbol: str) -> tuple[str, str] | None:
    filename = f"{reference_symbol}_{REFERENCE_TIMEFRAME}.parquet"
    path = os.path.join(highlow_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if "timestamp" not in df.columns:
            df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return (
            df["timestamp"].min().strftime("%Y-%m-%d"),
            df["timestamp"].max().strftime("%Y-%m-%d"),
        )
    except Exception:
        return None

# =============================================================================
# SPLIT PREVIEW
# =============================================================================

def print_split_preview(config: dict) -> bool:
    window_oos = config.get("window_oos_months", 3)
    start_date = config.get("start_date", "2020-01-01")
    ref_date_str = config.get("split_reference_date", None)
    highlow_dir = config.get("highlow_dir", "")

    is_start, is_end, oos_start, oos_end = _compute_windows(config)

    is_start_dt = pd.to_datetime(is_start).to_pydatetime()
    is_end_dt = pd.to_datetime(is_end).to_pydatetime()
    oos_start_dt = pd.to_datetime(oos_start).to_pydatetime()
    oos_end_dt = pd.to_datetime(oos_end).to_pydatetime()

    is_months = (is_end_dt.year - is_start_dt.year) * 12 + (is_end_dt.month - is_start_dt.month)
    oos_months = (oos_end_dt.year - oos_start_dt.year) * 12 + (oos_end_dt.month - oos_start_dt.month)

    is_folder = _make_folder_name(is_start, is_end, oos_start, oos_end, "IS")
    oos_folder = _make_folder_name(is_start, is_end, oos_start, oos_end, "OOS")

    data_range = _get_data_range(highlow_dir, config.get("reference_symbol", "EURUSD"))

    ref_label = ref_date_str if ref_date_str else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d") + " (today)"

    print(f"\n{'='*60}")
    print(f"  📊 Split preview — {ref_label}")
    print(f"{'='*60}")
    if data_range:
        print(f"  Data available    : {data_range[0]} → {data_range[1]}")
    print(f"  START_DATE        : {start_date}")
    print(f"  WINDOW_OOS_MONTHS : {window_oos}")
    print(f"")
    print(f"  IS  : {is_start} → {is_end}  ({is_months} months)")
    print(f"  OOS : {oos_start} → {oos_end}  ({oos_months} months available)")
    print(f"")
    print(f"  📁 Output folders:")
    print(f"  IS  → {os.path.join('IS', is_folder)}/")
    print(f"  OOS → {os.path.join('OOS', oos_folder)}/")
    print(f"{'='*60}")

    answer = input("\n  Continue? [y/n]: ").strip().lower()
    return answer == "y"

# =============================================================================
# GET LATEST SPLIT FOLDERS — utility for downstream scripts
# =============================================================================

def get_latest_split_folders(split_dir: str) -> dict | None:
    mode_dir = split_dir
    result = {}

    for subset in ["IS", "OOS"]:
        subset_dir = os.path.join(mode_dir, subset)
        if not os.path.exists(subset_dir):
            return None
        folders = sorted([
            f for f in os.listdir(subset_dir)
            if os.path.isdir(os.path.join(subset_dir, f)) and f.startswith("fx_")
        ])
        if not folders:
            return None
        result[subset] = os.path.join(subset_dir, folders[-1])

    return result

# =============================================================================
# SPLIT & SAVE
# =============================================================================

def _split(df: pd.DataFrame, is_start: str, is_end: str, oos_start: str, oos_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_is = df[(df["timestamp"] >= pd.to_datetime(is_start)) & (df["timestamp"] < pd.to_datetime(is_end))].copy()
    df_oos = df[(df["timestamp"] >= pd.to_datetime(oos_start)) & (df["timestamp"] < pd.to_datetime(oos_end))].copy()

    return df_is, df_oos


def _save(df: pd.DataFrame, path: str, export_csv: bool = False) -> None:
    df_save = df.copy()
    if "timestamp" in df_save.columns:
        df_save = df_save.set_index("timestamp")
    df_save.to_parquet(path, index=True)
    if export_csv:
        df_save.to_csv(os.path.splitext(path)[0] + ".csv", index=True)

# =============================================================================
# RUN
# =============================================================================

def run(config: dict) -> bool:
    input_dir: str = config["highlow_dir"]
    split_dir: str = config["split_dir"]
    export_csv: bool = config.get("export_csv", False)

    mode_dir = split_dir
    os.makedirs(mode_dir, exist_ok=True)

    is_start, is_end, oos_start, oos_end = _compute_windows(config)

    is_folder_name = _make_folder_name(is_start, is_end, oos_start, oos_end, "IS")
    oos_folder_name = _make_folder_name(is_start, is_end, oos_start, oos_end, "OOS")

    is_dir = os.path.join(mode_dir, "IS", is_folder_name)
    oos_dir = os.path.join(mode_dir, "OOS", oos_folder_name)
    os.makedirs(is_dir, exist_ok=True)
    os.makedirs(oos_dir, exist_ok=True)

    selected_symbols = config.get("selected_symbols") or []
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".parquet")
        and (not selected_symbols or any(f.startswith(s) for s in selected_symbols))
    ]) if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"✂️  IS/OOS split — {len(files)} file(s)")
    logger.info(f"   IS  : {is_start} → {is_end}  →  {is_folder_name}")
    logger.info(f"   OOS : {oos_start} → {oos_end}  →  {oos_folder_name}")

    skipped_is = skipped_oos = errors = processed = 0

    for filename in files:
        symbol = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {filename}: {e}")
            errors += 1
            continue

        try:
            df_is, df_oos = _split(df, is_start, is_end, oos_start, oos_end)
            has_is = not df_is.empty
            has_oos = not df_oos.empty

            if has_is:
                _save(df_is, os.path.join(is_dir, filename), export_csv)
            else:
                skipped_is += 1
                logger.debug(f"  ⚠ [{symbol}] No IS data")

            if has_oos:
                _save(df_oos, os.path.join(oos_dir, filename), export_csv)
            else:
                skipped_oos += 1
                logger.debug(f"  ⚠ [{symbol}] No OOS data")

            if has_is or has_oos:
                processed += 1
                logger.info(f"  ✅ [{symbol}] IS: {len(df_is)} rows | OOS: {len(df_oos)} rows")

        except Exception as e:
            logger.warning(f"  ❌ [{symbol}] Error: {e}")
            errors += 1

    logger.info(f"\n  Processed: {processed} | Skipped IS: {skipped_is} | Skipped OOS: {skipped_oos} | Errors: {errors}")

    if errors:
        logger.warning(f"⚠ Step 7 completed with {errors} error(s)")
        return False

    logger.info("✅ IS/OOS split complete")
    return True

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "highlow_dir": os.path.join(_base, "..", "data_fx", "03_highlow"),
        "split_dir": os.path.join(_base, "..", "data_fx", "04_split"),
        "window_oos_months": 3,
        "start_date": "2021-01-01",
        "split_reference_date": None,
        "export_csv": False,
        "reference_symbol": "EURUSD",
    }
    run(_config)