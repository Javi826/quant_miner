#BOT_crypto/execution/brokers/base_client.py
"""
Base Broker Client - Abstract base class for exchange clients.

This module provides an abstract base class that can be implemented
for different exchanges (Bitget, Binance, Bybit, etc.).

This is optional but follows best practices for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional


class BaseBrokerClient(ABC):

    
    @abstractmethod
    def send_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Any]:

        pass
    
    @abstractmethod
    def get_open_positions(
        self,
        product_type: str = "USDT-FUTURES"
    ) -> List[Dict[str, Any]]:

        pass
    
    @abstractmethod
    def get_usdt_balance(self, exchange=None) -> float:

        pass
