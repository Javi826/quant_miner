"""
alerts/telegram_notifier.py system for trading bot alerts.
"""

import requests
import logging
from typing import Optional

logger = logging.getLogger('BOT_crypto.alerts.telegram')

# Telegram API credentials
TELEGRAM_TOKEN = "8962549784:AAFvrdbv0fQKW_DAGRl17k8H3muIZsJGXd4"
TELEGRAM_CHAT_ID = "6327321903"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"


def send_alert(message: str, parse_mode: str = "HTML") -> bool:
    """
    Send alert message via Telegram.
    
    Args:
        message: Text message to send
        parse_mode: Telegram parse mode (HTML or Markdown)
    
    Returns:
        True if sent successfully, False otherwise
    """
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": parse_mode
        }
        
        response = requests.post(
            TELEGRAM_API_URL,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.debug(f"Telegram alert sent successfully")
            return True
        else:
            logger.warning(f"Telegram API returned {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Telegram alert timeout (>10s)")
        return False
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")
        return False


def send_sync_alert(account: str, symbol: str, issue_type: str, 
                   local_size: float, broker_size: float, 
                   strategies: list) -> bool:
    """
    Send sync discrepancy alert.
    
    Args:
        account: Account number
        symbol: Trading symbol
        issue_type: Type of issue (size_mismatch, not_in_broker, etc.)
        local_size: Local position size
        broker_size: Broker position size
        strategies: List of involved strategies
    
    Returns:
        True if sent successfully
    """
    strategies_str = ", ".join(strategies)
    
    message = f"""⚠️ <b>SYNC ISSUE - Account {account}</b>

Symbol: <code>{symbol}</code>
Issue: <b>{issue_type}</b>
Local: {local_size:.6f}
Broker: {broker_size:.6f}
Strategies: {strategies_str}

<i>Action required: Manual review</i>"""
    
    return send_alert(message)