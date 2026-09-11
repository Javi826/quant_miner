"""
Brokers module - Exchange API clients.

This module provides unified clients for interacting with different
cryptocurrency exchanges (currently Bitget, future: Binance, Bybit, etc.).
"""

from .bitget_client import BitgetClient
from .base_client import BaseBrokerClient

__all__ = [
    'BitgetClient',
    'BaseBrokerClient',
]
