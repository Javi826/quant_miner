"""
Rule Mining Strategies Configuration — Deploy


Train windows:
  4H    : 2025-11-14 → 2026-08-14  (9m train)  |  next train: 2026-10-14  (+2m)
  12H   : 2025-11-14 → 2026-08-14  (9m train)  |  next train: 2026-10-14  (+2m)
  1H    : 2025-11-14 → 2026-08-14  (9m train)  |  next train: 2026-10-14  (+2m)
  6H    : 2025-11-14 → 2026-08-14  (9m train)  |  next train: 2026-10-14  (+2m)
"""

STRATEGIES = [
    {
        "id": "001280_4H_long_RSI14gt60_AND_HISTVOL20gtSMA_HISTVOL40",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "4H",
        "active": True,
        "direction": "long",
        "specs": [{"type": "rsi", "period": 14, "op": ">", "value": 60}, {"type": "histvol_regime", "period": 20, "op": ">", "sma_period": 40}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 10,
        "sl_pct": 8,
    },
    {
        "id": "022301_12H_long_RSI7gt60_AND_HISTVOL20gtSMA_HISTVOL20_AND_HISTVOL30ltSMA_HISTVOL20",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "12H",
        "active": True,
        "direction": "long",
        "specs": [{"type": "rsi", "period": 7, "op": ">", "value": 60}, {"type": "histvol_regime", "period": 20, "op": ">", "sma_period": 20}, {"type": "histvol_regime", "period": 30, "op": "<", "sma_period": 20}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 8,
        "sl_pct": 6,
    },
    {
        "id": "024772_12H_long_RSI7lt60_AND_HISTVOL10ltSMA_HISTVOL20_AND_HISTVOL10gtSMA_HISTVOL50",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "12H",
        "active": True,
        "direction": "long",
        "specs": [{"type": "rsi", "period": 7, "op": "<", "value": 60}, {"type": "histvol_regime", "period": 10, "op": "<", "sma_period": 20}, {"type": "histvol_regime", "period": 10, "op": ">", "sma_period": 50}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 10,
        "sl_pct": 8,
    },
    {
        "id": "048243_1H_long_RSI14gt70_AND_HISTVOL10gtSMA_HISTVOL30_AND_HISTVOL30ltSMA_HISTVOL50",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "1H",
        "active": True,
        "direction": "long",
        "specs": [{"type": "rsi", "period": 14, "op": ">", "value": 70}, {"type": "histvol_regime", "period": 10, "op": ">", "sma_period": 30}, {"type": "histvol_regime", "period": 30, "op": "<", "sma_period": 50}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 8,
        "sl_pct": 6,
    },
    {
        "id": "102727_1H_short_RSI7lt40_AND_HISTVOL30gtSMA_HISTVOL20_AND_HISTVOL30ltSMA_HISTVOL30",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "1H",
        "active": True,
        "direction": "short",
        "specs": [{"type": "rsi", "period": 7, "op": "<", "value": 40}, {"type": "histvol_regime", "period": 30, "op": ">", "sma_period": 20}, {"type": "histvol_regime", "period": 30, "op": "<", "sma_period": 30}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 10,
        "sl_pct": 6,
    },
    {
        "id": "107277_6H_short_RSI7lt50_AND_ATR14gtSMA_ATR20_AND_ATR21ltSMA_ATR30",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "6H",
        "active": True,
        "direction": "short",
        "specs": [{"type": "rsi", "period": 7, "op": "<", "value": 50}, {"type": "atr_regime", "period": 14, "op": ">", "sma_period": 20}, {"type": "atr_regime", "period": 21, "op": "<", "sma_period": 30}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 8,
        "sl_pct": 8,
    },
    {
        "id": "142800_12H_short_RSI21gt40_AND_ATR14ltSMA_ATR50_AND_HISTVOL10gtSMA_HISTVOL30",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "12H",
        "active": True,
        "direction": "short",
        "specs": [{"type": "rsi", "period": 21, "op": ">", "value": 40}, {"type": "atr_regime", "period": 14, "op": "<", "sma_period": 50}, {"type": "histvol_regime", "period": 10, "op": ">", "sma_period": 30}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 8,
        "sl_pct": 6,
    },
    {
        "id": "167892_4H_short_ATR14gtSMA_ATR30_AND_ATR21ltSMA_ATR20_AND_HISTVOL30gtSMA_HISTVOL20",
        "symbols": ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"],
        "timeframe": "4H",
        "active": True,
        "direction": "short",
        "specs": [{"type": "atr_regime", "period": 14, "op": ">", "sma_period": 30}, {"type": "atr_regime", "period": 21, "op": "<", "sma_period": 20}, {"type": "histvol_regime", "period": 30, "op": ">", "sma_period": 20}],
        "sell_after_ncandles": 50,
        "order_amount": 1000,
        "tp_pct": 10,
        "sl_pct": 8,
    },
]
