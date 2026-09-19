# core/config_core.py
import os
import logging

logger = logging.getLogger("BOT_batch.config_core")


class Settings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


CRYPTO = Settings(
    DAYS_PER_YEAR=365,
    WEEKMASK="1111111",
    CUSTOM_FIELDS={"order_amount": 500},
    BACKTEST_MODE="NPY",
)

FOREX = Settings(
    DAYS_PER_YEAR=252,
    WEEKMASK="1111100",
    CUSTOM_FIELDS={"lot": 0.2, "magic": "AUTO_INCREMENT"},
    BACKTEST_MODE="YPY",
    #BACKTEST_MODE="NPY",
)
_MARKET_SETTINGS = {
    "crypto": CRYPTO,
    "forex": FOREX,
}

_DEFAULT_MARKET = "forex"

_market = os.environ.get("QUANT_MARKET", _DEFAULT_MARKET)
if _market not in _MARKET_SETTINGS:
    raise RuntimeError(
        f"QUANT_MARKET not set or invalid: {_market!r}. "
        f"Must be one of {list(_MARKET_SETTINGS)}."
    )

settings = _MARKET_SETTINGS[_market]

