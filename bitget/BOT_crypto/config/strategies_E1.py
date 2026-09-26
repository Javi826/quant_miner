"""

Train windows:
  6H    : 2025-10-23 → 2026-07-23  (9m train)  |  next train: 2026-09-23  (+2m)
  12H   : 2025-10-23 → 2026-07-23  (9m train)  |  next train: 2026-09-23  (+2m)
  1H    : 2025-10-23 → 2026-07-23  (9m train)  |  next train: 2026-09-23  (+2m)
  4H    : 2025-10-23 → 2026-07-23  (9m train)  |  next train: 2026-09-23  (+2m)
"""

STRATEGIES = [
    {
        "id": "08382_1H_long_RSI7gt70_AND_ATR7gtSMA_ATR50_AND_HISTVOL30ltSMA_HISTVOL20",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "XLMUSDT", "XRPUSDT"],
        "timeframe": "1H",
        "active": True,
        "direction": "long",
        "specs": [{"type": "rsi", "period": 7, "op": ">", "value": 70}, {"type": "atr_regime", "period": 7, "op": ">", "sma_period": 50}, {"type": "histvol_regime", "period": 30, "op": "<", "sma_period": 20}],
        "sell_after_ncandles": 50,
        "order_amount": 500,
        "tp_pct": 10,
        "sl_pct": 6,
    },

    {
        "id": "53019_4H_short_ADX7gt25_AND_CLOSEltMA100_AND_HISTVOL30gtSMA_HISTVOL50",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "XLMUSDT", "XRPUSDT"],
        "timeframe": "4H",
        "active": True,
        "direction": "short",
        "specs": [{"type": "adx", "period": 7, "op": ">", "value": 25}, {"type": "ma", "op": "<", "value": 100}, {"type": "histvol_regime", "period": 30, "op": ">", "sma_period": 50}],
        "sell_after_ncandles": 50,
        "order_amount": 500,
        "tp_pct": 6,
        "sl_pct": 6,
    },

]
