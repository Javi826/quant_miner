#BOT_crypto/execution/brokers/bitget_client.py
"""
Bitget API Client - Professional HTTP client for Bitget Futures API.

This module provides a unified client for interacting with Bitget's Futures API,
replacing the duplicated code from ZX_connect_live.py (7x accounts).

Features:
- HMAC signature authentication
- GET/POST requests
- Position management
- Balance queries
- Error handling with logging
"""

import time
import requests
import json
import hashlib
import base64
import hmac
import logging
from urllib.parse import urlencode
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger('BOT_crypto.execution.bitget_client')


class BitgetClient:

    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        base_url: str = "https://api.bitget.com",
        timeout: int = 15
    ):
        """
        Initialize Bitget API client.
        
        Args:
            api_key: Bitget API key
            api_secret: Bitget API secret
            api_passphrase: Bitget API passphrase
            base_url: Base URL for API (default: production)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.base_url = base_url
        self.timeout = timeout
        
        logger.debug(f"BitgetClient initialized for key: {api_key[:8]}...")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds."""
        return str(int(time.time() * 1000))
    
    def _sign_request(
        self,
        timestamp: str,
        method: str,
        path: str,
        query_string: str,
        body_str: str
    ) -> str:

        # Build string to sign
        to_sign = timestamp + method.upper() + path
        if query_string:
            to_sign += "?" + query_string
        to_sign += body_str
        
        # Generate HMAC signature
        digest = hmac.new(
            self.api_secret.encode("utf-8"),
            to_sign.encode("utf-8"),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(digest).decode()
    
    def _build_headers(self, timestamp: str, signature: str) -> Dict[str, str]:

        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.api_passphrase,
            "Content-Type": "application/json"
        }
    
    def send_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Any]:

        timestamp = self._get_timestamp()
        
        # Build query string and body
        query_string = urlencode(params) if params else ""
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        
        # Generate signature
        signature = self._sign_request(timestamp, method, path, query_string, body_str)
        
        # Build headers
        headers = self._build_headers(timestamp, signature)
        
        # Build URL
        url = self.base_url + path
        if query_string:
            url += f"?{query_string}"
        
        # Send request
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=self.timeout)
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    data=body_str.encode('utf-8'),
                    timeout=self.timeout
                )
            
            # Parse response
            content_type = response.headers.get("Content-Type", "")
            if content_type.startswith("application/json"):
                return response.status_code, response.json()
            else:
                return response.status_code, response.text
                
        except Exception as e:
            logger.error(f"Request failed: {method} {path} - {e}")
            return 0, {"error": str(e)}
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        code, resp = self.send_request("GET", endpoint, params=params)
        
        if code == 200 and isinstance(resp, dict):
            return resp
        
        raise Exception(f"GET {endpoint} failed: {resp}")
    
    def post(
        self,
        endpoint: str,
        body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        code, resp = self.send_request("POST", endpoint, body=body)
        
        if code == 200 and isinstance(resp, dict):
            return resp
        
        raise Exception(f"POST {endpoint} failed: {resp}")
    
    # ========================================================================
    # HIGH-LEVEL METHODS (convenience wrappers)
    # ========================================================================
    
    def get_open_positions(
        self,
        product_type: str = "USDT-FUTURES"
    ) -> List[Dict[str, Any]]:

        try:
            response = self.get(
                "/api/v2/mix/position/all-position",
                {"productType": product_type}
            )
            return response.get("data", [])
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_usdt_balance(self, exchange=None) -> float:

        if exchange is None:
            logger.error("Exchange object required for get_usdt_balance")
            return 0.0
        
        try:
            balance = exchange.fetch_balance()
            return balance['free']['USDT']
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def __repr__(self) -> str:
        """String representation of client."""
        return f"BitgetClient(api_key={self.api_key[:8]}...)"
