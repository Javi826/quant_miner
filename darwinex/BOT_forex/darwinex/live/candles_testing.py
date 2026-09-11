import sys
import os
import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
from mt5linux import MetaTrader5

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from broker_client.broker_config import (
    MT5_HOST,
    MT5_PORT,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    TIMEFRAME_MINUTES,
    BAR_CLOSE_BUFFER_SECONDS,
)

N_BARS = 3

TIMEFRAME_CONSTANTS = {
    "1m" : "TIMEFRAME_M1",
    "5m" : "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1H" : "TIMEFRAME_H1",
    "4H" : "TIMEFRAME_H4",
    "1D" : "TIMEFRAME_D1",
}


DEFAULT_SYMBOL    = "EURGBP"
DEFAULT_TIMEFRAME = "5m"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare last-bar detection methods")
    parser.add_argument("symbol", type=str, nargs="?", default=DEFAULT_SYMBOL,
                         help=f"Symbol name (default: {DEFAULT_SYMBOL})")
    parser.add_argument("timeframe", type=str, nargs="?", default=DEFAULT_TIMEFRAME,
                         choices=list(TIMEFRAME_MINUTES),
                         help=f"Timeframe label (default: {DEFAULT_TIMEFRAME})")
    return parser.parse_args()


def connect():
    client = MetaTrader5(host=MT5_HOST, port=MT5_PORT)

    if not client.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {client.last_error()}")
    if not client.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise ConnectionError(f"MT5 login() failed: {client.last_error()}")

    return client


SERVER_OFFSET_STEP_SECONDS = 3600


def sync_offset(client, symbol):
    tick = client.symbol_info_tick(symbol)

    if tick is None:
        raise RuntimeError(f"No tick available for {symbol}")

    raw_offset = int(tick.time) - int(datetime.now().timestamp())
    return int(round(raw_offset / SERVER_OFFSET_STEP_SECONDS)) * SERVER_OFFSET_STEP_SECONDS


def rates_to_df(rates):
    df = pd.DataFrame(rates)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")


def method_current(client, symbol, timeframe, mt5_tf_const, offset_seconds, n):
    """Current stack method: copy_rates_from_pos + drop forming bar."""
    rates = client.copy_rates_from_pos(symbol, mt5_tf_const, 0, n + 1)
    df = rates_to_df(rates)

    if df.empty:
        return df

    bar_seconds  = TIMEFRAME_MINUTES[timeframe] * 60
    last_open_ts = int(df.index[-1].timestamp())
    last_end_ts  = last_open_ts + bar_seconds
    server_now   = datetime.now().timestamp() + offset_seconds

    if server_now < last_end_ts + BAR_CLOSE_BUFFER_SECONDS:
        df = df.iloc[:-1]

    return df.tail(n)


def method_quantreo(client, symbol, mt5_tf_const, n):
    """Quantreo-style method: copy_rates_from(now + 26h), no drop."""
    from_date = datetime.now() + timedelta(hours=26)
    rates = client.copy_rates_from(symbol, mt5_tf_const, from_date, n)
    return rates_to_df(rates)


def print_bars(label, df):
    print(f"\n[{label}]")

    if df.empty:
        print("  <no data>")
        return

    for ts, row in df.iterrows():
        epoch = int(ts.timestamp())
        print(f"  {ts} (epoch={epoch})  O={row['open']} H={row['high']} L={row['low']} C={row['close']}")


POLL_SECONDS = 5


def last_bar_epoch(df):
    if df.empty:
        return None
    return int(df.index[-1].timestamp())


def run_loop(client, symbol, timeframe, mt5_tf_const):
    offset_seconds = sync_offset(client, symbol)

    print("=" * 60)
    print(f"Symbol: {symbol} | Timeframe: {timeframe} | Poll: {POLL_SECONDS}s")
    print(f"Server offset: {offset_seconds}s")
    print("Waiting for new bars... (Ctrl+C to stop)")
    print("=" * 60)

    last_seen_current  = last_bar_epoch(method_current(client, symbol, timeframe, mt5_tf_const, offset_seconds, 1))
    last_seen_quantreo = last_bar_epoch(method_quantreo(client, symbol, mt5_tf_const, 1))

    print(f"Initial CURRENT bar:  {last_seen_current}")
    print(f"Initial QUANTREO bar: {last_seen_quantreo}\n")

    try:
        while True:
            detection_time = datetime.now()

            offset_seconds = sync_offset(client, symbol)

            current_epoch = last_bar_epoch(
                method_current(client, symbol, timeframe, mt5_tf_const, offset_seconds, 1)
            )
            quantreo_epoch = last_bar_epoch(
                method_quantreo(client, symbol, mt5_tf_const, 1)
            )

            if current_epoch is not None and current_epoch != last_seen_current:
                print(f"[CURRENT ] new bar detected at {detection_time} | bar_open_epoch={current_epoch}")
                last_seen_current = current_epoch

            if quantreo_epoch is not None and quantreo_epoch != last_seen_quantreo:
                print(f"[QUANTREO] new bar detected at {detection_time} | bar_open_epoch={quantreo_epoch}")
                last_seen_quantreo = quantreo_epoch

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopped by user.")


def main():
    args = parse_args()
    symbol = args.symbol
    timeframe = args.timeframe

    client = connect()
    mt5_tf_const = getattr(client, TIMEFRAME_CONSTANTS[timeframe])

    run_loop(client, symbol, timeframe, mt5_tf_const)


if __name__ == "__main__":
    main()