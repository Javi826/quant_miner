import sys
from datetime import datetime, timezone
from mt5linux import MetaTrader5

MT5_LOGIN    = 4000097952
MT5_PASSWORD = "52iYM6Q&K"
MT5_SERVER   = "Darwinex-Live"

SYMBOL = "USDJPY"
LOT    = 0.1

mt5 = MetaTrader5(host="localhost", port=8001)

if not mt5.initialize():
    print("initialize() FAILED:", mt5.last_error())
    sys.exit(1)

if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    print("login() FAILED:", mt5.last_error())
    sys.exit(1)

print("=" * 70)
print("ACCOUNT")
print("=" * 70)
info = mt5.account_info()
print(f"login          : {info.login}")
print(f"balance        : {info.balance} {info.currency}")
print(f"trade_allowed  : {info.trade_allowed}")
print(f"trade_expert   : {info.trade_expert}")
print(f"margin_free    : {info.margin_free}")

print()
print("=" * 70)
print(f"SYMBOL: {SYMBOL}")
print("=" * 70)
sym = mt5.symbol_info(SYMBOL)
if sym is None:
    print(f"symbol_info({SYMBOL}) returned None — symbol not found")
    sys.exit(1)

print(f"visible        : {sym.visible}")
print(f"trade_mode     : {sym.trade_mode}   (0=disabled 1=longonly 2=shortonly 3=closeonly 4=full)")
print(f"volume_min     : {sym.volume_min}")
print(f"volume_max     : {sym.volume_max}")
print(f"volume_step    : {sym.volume_step}")
print(f"filling_mode   : {sym.filling_mode}   (bitmask: 1=FOK 2=IOC)")
print(f"stops_level    : {sym.trade_stops_level}")
print(f"digits         : {sym.digits}")

if not sym.visible:
    print(f"\n{SYMBOL} not visible in Market Watch — selecting it...")
    if mt5.symbol_select(SYMBOL, True):
        print("symbol_select() OK")
        sym = mt5.symbol_info(SYMBOL)
    else:
        print("symbol_select() FAILED:", mt5.last_error())

print()
print("=" * 70)
print("TICK")
print("=" * 70)
tick = mt5.symbol_info_tick(SYMBOL)
print(f"time           : {tick.time}  ({datetime.fromtimestamp(tick.time, timezone.utc)})")
print(f"bid            : {tick.bid}")
print(f"ask            : {tick.ask}")
print(f"now utc        : {datetime.now(timezone.utc)}")
print(f"tick age (s)   : {datetime.now(timezone.utc).timestamp() - tick.time:.0f}")

print()
print("=" * 70)
print("ORDER_CHECK PER FILLING MODE (SELL, no TP/SL)")
print("=" * 70)

for filling in (0, 1, 2):
    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : SYMBOL,
        "volume"      : LOT,
        "type"        : mt5.ORDER_TYPE_SELL,
        "price"       : tick.bid,
        "deviation"   : 10,
        "magic"       : 999,
        "comment"     : "diagnostic",
        "type_filling": filling,
        "type_time"   : mt5.ORDER_TIME_GTC,
    }
    result = mt5.order_check(request)
    if result is None:
        print(f"filling={filling} -> order_check() returned None | last_error={mt5.last_error()}")
    else:
        print(f"filling={filling} -> retcode={result.retcode} comment='{result.comment}'")

print()
print("=" * 70)
print("ORDER_CHECK WITH TP/SL (as the bot sends it, filling=0)")
print("=" * 70)

pct_tp, pct_sl = 0.5, 0.35
tp_price = tick.bid * (1 - pct_tp / 100)
sl_price = tick.bid * (1 + pct_sl / 100)

request = {
    "action"      : mt5.TRADE_ACTION_DEAL,
    "symbol"      : SYMBOL,
    "volume"      : LOT,
    "type"        : mt5.ORDER_TYPE_SELL,
    "price"       : tick.bid,
    "deviation"   : 10,
    "tp"          : tp_price,
    "sl"          : sl_price,
    "magic"       : 999,
    "comment"     : "diagnostic",
    "type_filling": 0,
    "type_time"   : mt5.ORDER_TIME_GTC,
}
print(f"bid={tick.bid} tp={tp_price} sl={sl_price}")
result = mt5.order_check(request)
if result is None:
    print(f"order_check() returned None | last_error={mt5.last_error()}")
else:
    print(f"retcode={result.retcode} comment='{result.comment}'")

print()
print("=" * 70)
print("NOTE: this script only CHECKS orders, it never sends one.")
print("=" * 70)