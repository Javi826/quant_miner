# darwinex/broker_client/broker_config.py
# =============================================================================
# CONNECTION SETTINGS
# =============================================================================
MT5_HOST     = "localhost"
MT5_PORT     = 8001
MT5_LOGIN    = 4000097952
MT5_PASSWORD = "Tomatera86-"
MT5_SERVER   = "Darwinex-Live"

# =============================================================================
# TIMEFRAMES
# =============================================================================
# Canonical across the whole MT5 side of the project: minutes in lowercase,
# hours and days in uppercase. No aliases — an unknown label raises instead
# of being silently coerced.
TIMEFRAME_MINUTES = {
    "1m" : 1,
    "5m" : 5,
    "15m": 15,
    "30m": 30,
    "1H" : 60,
    "4H" : 240,
    "1D" : 1440,
}

# =============================================================================
# SERVER CLOCK
# =============================================================================
# Broker clocks always sit on a whole-hour offset from UTC, so the raw
# estimate is snapped to the nearest hour. This absorbs a stale tick feed.
SERVER_OFFSET_STEP_SECONDS       = 3600
SERVER_OFFSET_DRIFT_WARN_SECONDS = 300

# Grace period after a bar closes before it is treated as closed. Lets the
# broker settle the final ticks and materialise the next forming bar.
BAR_CLOSE_BUFFER_SECONDS = 30

# Any liquid symbol — used only to read the broker's clock from its last tick
# when no symbol list is available (e.g. one-off pipeline calls).
SERVER_CLOCK_SYMBOL = "EURGBP"