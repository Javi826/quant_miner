#quant_miner/darwinex/BOT_forex/darwinex/live/strategies_config.py (forex)
# Strategy definitions only. Replace the whole file on every deploy.

"""
Rule Mining Strategies Configuration — Deploy

Train windows:
  1H    : 2025-09-19 → 2026-09-09  (12m train)  |  next train: 2026-12-09  (+3m)
  4H    : 2025-09-19 → 2026-09-09  (12m train)  |  next train: 2026-12-09  (+3m)
"""

STRATEGIES = [
    {
        "id"                 : "010053_1H_c01_long_ichimoku_cloud_thickness_tenkan18_kijun26_senkou_b27gt0.4_AND_inertia_rvi40_21ltm0.1",
        "symbols"            : ["CHFJPY"],
        "timeframe"          : "1H",
        "direction"          : "long",
        "specs"              : [
            {"indicator": "ichimoku_cloud_thickness", "params": {"tenkan": 18, "kijun": 26, "senkou_b": 27}, "op": ">", "threshold": 0.4},
            {"indicator": "inertia", "params": {"rvi_period": 40, "period": 21}, "op": "<", "threshold": -0.1},
        ],
        "sell_after_ncandles": 400,
        "tp_pct"             : 1.18,
        "sl_pct"             : 1.18,
        "lot"                : 0.2,
        "magic"              : 2,
        "active"             : True,
    },
    {
        "id"                 : "010151_4H_c01_long_ichimoku_cloud_thickness_tenkan18_kijun26_senkou_b27gt0.6_AND_inertia_rvi10_21lt0.1",
        "symbols"            : ["CHFJPY"],
        "timeframe"          : "4H",
        "direction"          : "long",
        "specs"              : [
            {"indicator": "ichimoku_cloud_thickness", "params": {"tenkan": 18, "kijun": 26, "senkou_b": 27}, "op": ">", "threshold": 0.6},
            {"indicator": "inertia", "params": {"rvi_period": 10, "period": 21}, "op": "<", "threshold": 0.1},
        ],
        "sell_after_ncandles": 0,
        "tp_pct"             : 1.6,
        "sl_pct"             : 1.8,
        "lot"                : 0.2,
        "magic"              : 3,
        "active"             : True,
    },
    {
        "id"                 : "011916_1H_c07_long_ichimoku_cloud_thickness_tenkan9_kijun26_senkou_b27gt0.8_AND_donchian_width_20gt0.02",
        "symbols"            : ["AUDJPY", "EURJPY"],
        "timeframe"          : "1H",
        "direction"          : "long",
        "specs"              : [
            {"indicator": "ichimoku_cloud_thickness", "params": {"tenkan": 9, "kijun": 26, "senkou_b": 27}, "op": ">", "threshold": 0.8},
            {"indicator": "donchian_width", "params": {"period": 20}, "op": ">", "threshold": 0.02},
        ],
        "sell_after_ncandles": 400,
        "tp_pct"             : 1.58,
        "sl_pct"             : 2.38,
        "lot"                : 0.2,
        "magic"              : 4,
        "active"             : True,
    },
    {
        "id"                 : "015396_1H_c06_long_trend_intensity_index_15lt30_AND_donchian_width_20gt0.02",
        "symbols"            : ["AUDJPY", "CHFJPY"],
        "timeframe"          : "1H",
        "direction"          : "long",
        "specs"              : [
            {"indicator": "trend_intensity_index", "params": {"period": 15}, "op": "<", "threshold": 30.0},
            {"indicator": "donchian_width", "params": {"period": 20}, "op": ">", "threshold": 0.02},
        ],
        "sell_after_ncandles": 400,
        "tp_pct"             : 1.58,
        "sl_pct"             : 1.78,
        "lot"                : 0.2,
        "magic"              : 6,
        "active"             : True,
    },
]