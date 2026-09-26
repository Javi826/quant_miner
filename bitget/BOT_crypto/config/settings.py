#BOT_crypto/config/settings.py
import sys, os
import socket
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from zoneinfo import ZoneInfo

# ==========================================================================
# ACCOUNT-SPECIFIC SETTINGS
# ==========================================================================

ACCOUNTS = {
    "E1": {
        "initial_capital": 40000,
        "dashboard_port": 5001,
        "description": "Elite Account",
        "type": "production",
        "risk_control_enabled": True,
        "postgresql_enabled": True,
        "reference_symbol": "BTCUSDT",
    },
    "00": {
        "initial_capital": 80000,
        "dashboard_port": 5000,
        "description": "Main Account",
        "type": "production",
        "risk_control_enabled": True,
        "postgresql_enabled": True,
        "reference_symbol": "BTCUSDT",
    },  
    "01": {
        "initial_capital": 12000,
        "dashboard_port": 5099,
        "description": "Testing Account",
        "type": "demo,production",
        "risk_control_enabled": False,
        "postgresql_enabled": False,
        "reference_symbol": "BTCUSDT",
    },
}

COMMISSION_PCT = 0.1

# ==========================================================================
# RISK CONTROL SETTINGS
# ==========================================================================
RISK_LIMITS = {
    'max_gross_exposure_pct': 10.0,
    'max_net_exposure_pct':   10.0,
}
LEVERAGE = 10

# =============================================================================
# QUALITY CONTROL PARAMETERS
# =============================================================================

# Execution quality
EXECUTION_WINDOW_SIZE  = 100
SLIPPAGE_WARNING_PCT   = 0.2
SLIPPAGE_CRITICAL_PCT  = 0.3
LATENCY_WARNING_SEC    = 0.5
LATENCY_CRITICAL_SEC   = 1.0

# ==========================================================================
# STRATEGY VALIDATION CONFIGURATION
# ==========================================================================

# Common parameters required for ALL strategies
COMMON_REQUIRED_PARAMS = [
    'id', 'timeframe', 'active', 'sell_after_ncandles',
    'order_amount', 'tp_pct', 'sl_pct', 'direction',
]

# Order amount limits (USDT)
MIN_ORDER_AMOUNT = 499
MAX_ORDER_AMOUNT = 1001

# TP/SL limits (%)
MIN_TP_PCT = 6
MAX_TP_PCT = 10
MIN_SL_PCT = 6
MAX_SL_PCT = 10

# Candles timeout limits
MIN_CANDLES = 49
MAX_CANDLES = 51

# Valid timeframes
VALID_TIMEFRAMES = ['1H', '4H', '6H', '12H', '1D']

# ==========================================================================
# POSTGRESQL CONFIGURATION
# ==========================================================================

# Environment detection
HOSTNAME      = socket.gethostname()
VPS_HOSTNAMES = ['srv1326826', 'hstgr.cloud']
IS_VPS        = any(h in HOSTNAME for h in VPS_HOSTNAMES)

POSTGRES_CONFIG = {
    'dbname':          'bot_trading',
    'user':            'javi',
    'password':        'Laplaciano86-',
    'host':            'localhost',
    'port':            5432,
    'connect_timeout': 3,
}

# VPS PostgreSQL connection (for split-brain check from LOCAL)
VPS_CHECK_CONFIG = {
    'host':     '100.123.10.95',
    'user':     'javi',
    'password': 'Laplaciano86-',
    'dbname':   'bot_trading',
    'timeout':  5,
}

# ==========================================================================
# EXCHANGE SETTINGS
# ==========================================================================
MARGIN_MODE     = "crossed"
MARGIN_COIN     = "USDT"
API_LIMIT_DATA  = 180

# ==========================================================================
# GENERAL BOT SETTINGS
# ==========================================================================
HOUR_ZONE             = ZoneInfo('UTC')
CHECK_INTERVAL        = 5
CANDLE_CLOSE_BUFFER   = 20
PERSISTENCE_DIR       = "persistence"

# ==========================================================================
# API - WEBSOCKET SETTINGS
# ==========================================================================
from broker_client.broker_config import BASE_URL, PRODUCT_TYPE, CANDLE_GRID_OFFSET_HOURS
# ==========================================================================
# LOGGER SETTINGS
# ==========================================================================
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL    = "INFO"
LOG_MAX_BYTES     = 10 * 1024 * 1024
LOG_BACKUP_COUNT  = 5
LOG_NAMESPACE     = "BOT_crypto"