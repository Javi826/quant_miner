#quant_d/develop/live_vs_backtest/production_trades.py
import sys
import os

_BASE = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "darwinex")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "darwinex", "BOT_forex", "darwinex")))
from datetime import datetime

import pandas as pd
from shared.shared_trading_data_mt5.broker_api.mt5_client import get_client
from live.strategies import get_active_strategies

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATE_FROM = datetime(2026, 8, 21)
DATE_TO   = datetime(2026, 9, 10)

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "production_trades.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MAGIC -> STRATEGY_ID MAPPING
# ─────────────────────────────────────────────────────────────────────────────
def build_magic_map() -> dict:
    return {s["magic"]: s["id"] for s in get_active_strategies()}


# ─────────────────────────────────────────────────────────────────────────────
# DEALS -> TRADES
# ─────────────────────────────────────────────────────────────────────────────
def fetch_deals(client, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    deals = client.history_deals_get(date_from, date_to)

    if not deals:
        return pd.DataFrame()

    rows = [
        {
            "position_id": d.position_id,
            "time"       : d.time,
            "entry"      : d.entry,
            "symbol"     : d.symbol,
            "magic"      : d.magic,
            "profit"     : d.profit,
        }
        for d in deals
    ]
    return pd.DataFrame(rows)


def deals_to_trades(deals: pd.DataFrame, client) -> pd.DataFrame:
    empty_columns = ["buy_time", "sell_time", "strategy", "symbol", "profit", "status"]

    if deals.empty:
        return pd.DataFrame(columns=empty_columns)

    entry_in  = int(client.DEAL_ENTRY_IN)
    entry_out = int(client.DEAL_ENTRY_OUT)

    trades = []
    for position_id, group in deals.groupby("position_id"):
        ins  = group[group["entry"] == entry_in]
        outs = group[group["entry"] == entry_out]

        if ins.empty:
            continue

        is_open = outs.empty

        trades.append({
            "buy_time" : ins["time"].min(),
            "sell_time": None if is_open else outs["time"].max(),
            "magic"    : group["magic"].iloc[0],
            "symbol"   : group["symbol"].iloc[0],
            "profit"   : None if is_open else outs["profit"].sum(),
            "status"   : "OPEN" if is_open else "CLOSED",
        })

    df = pd.DataFrame(trades)

    if df.empty:
        return pd.DataFrame(columns=empty_columns)

    df["buy_time"]  = pd.to_datetime(df["buy_time"],  unit="s", utc=True)
    df["sell_time"] = pd.to_datetime(df["sell_time"], unit="s", utc=True)
    return df


def map_strategies(df: pd.DataFrame, magic_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["strategy"] = df["magic"].map(magic_map)

    unmapped = df[df["strategy"].isna()]
    if not unmapped.empty:
        unknown_magics = sorted(unmapped["magic"].unique())
        print(f"WARNING: dropping {len(unmapped)} trades with unknown magic(s): {unknown_magics}")

    df = df.dropna(subset=["strategy"])
    return df[["buy_time", "sell_time", "strategy", "symbol", "profit", "status"]].sort_values("buy_time")
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    client = get_client()

    magic_map = build_magic_map()
    print(f"Magic map: {magic_map}")

    deals  = fetch_deals(client, DATE_FROM, DATE_TO)
    trades = deals_to_trades(deals, client)
    trades = map_strategies(trades, magic_map)

    n_open = (trades["status"] == "OPEN").sum()
    trades.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(trades)} trades to {OUTPUT_CSV} ({n_open} still open)")


if __name__ == "__main__":
    main()
