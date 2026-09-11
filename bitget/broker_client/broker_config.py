# bitget/broker_client/broker_config.py

# =============================================================================
# API SETTINGS
# =============================================================================
BASE_URL        = "https://api.bitget.com"
PRODUCT_TYPE    = "USDT-FUTURES"
API_TIMEOUT     = 10
API_MAX_RETRIES = 3

# =============================================================================
# WEBSOCKET SETTINGS
# =============================================================================
WS_PUBLIC_URL  = "wss://ws.bitget.com/v2/ws/public"
WS_PRIVATE_URL = "wss://ws.bitget.com/v2/ws/private"