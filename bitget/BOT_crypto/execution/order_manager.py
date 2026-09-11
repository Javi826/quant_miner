#BOT_crypto/execution/order_manager.py
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Tuple, Dict, Any

# WebSocket manager (global instance)
from market_data import get_ws_manager
from execution.trade_logger import log_closed_position, configure_log_path
import logging
logger = logging.getLogger('BOT_crypto.execution.order_manager')
from config.settings import HOUR_ZONE

# ==========================================================================
# CONSTANTS
# ==========================================================================
from config.settings import  PRODUCT_TYPE, MARGIN_MODE

# ==========================================================================
# CONFIGURATION
# ==========================================================================
# DESPUÉS
def configure_paths(trades_log_path: str) -> None:
    # Configure trade logger
    configure_log_path(trades_log_path)

    # Create directory if doesn't exist
    log_dir = os.path.dirname(trades_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


# ==========================================================================
# WEBSOCKET DATA FETCHING
# ==========================================================================
def fetch_ticker_ws(symbol: str) -> Tuple[Optional[Decimal], None]:

    if not get_ws_manager():
        raise RuntimeError("Error-WebSocket not initialized")
    
    # Subscribe if not already subscribed
    if symbol not in get_ws_manager().subscribed_public:
        get_ws_manager().subscribe_ticker(symbol)
        time.sleep(0.05)
    
    # Get from cache
    price_data = get_ws_manager().prices.get(symbol)
    
    if price_data:
        age = time.time() - price_data['timestamp']
        if age < 5.0:
            return price_data['price'], None
    
    # Wait for fresh data
    initial_ts = price_data['timestamp'] if price_data else 0
    timeout_start = time.time()
    
    while (time.time() - timeout_start) < 2.0:
        price_data = get_ws_manager().prices.get(symbol)
        if price_data and price_data['timestamp'] > initial_ts:
            return price_data['price'], None
        time.sleep(0.02)
    
    raise TimeoutError(f"WAR-NoWebSocketdatafor{symbol}")


def fetch_contracts_ws(symbol: str) -> Dict[str, Any]:

    if not get_ws_manager():
        raise RuntimeError("Error-WebSocket not initialized")
    
    contract = get_ws_manager().get_contract(symbol)
    if contract:
        return contract
    
    raise ValueError(f"Error-Contract for {symbol} not in cache")


def get_usdt_balance_ws(exchange=None) -> float:

    if not get_ws_manager():
        logger.error("Error-WS manager not init for balance.")
        return 0.0
    
    balance = get_ws_manager().get_usdt_balance()
    
    # If no equity data yet, wait briefly
    if balance == 0.0 and not get_ws_manager().equity:
        logger.warning("WAR-Waiting for equity data from WebSocket...")
        time.sleep(0.05)
        balance = get_ws_manager().get_usdt_balance()
    
    return balance


def get_current_price(symbol: str, max_cache_age: float = 0.5) -> Decimal:

    if not get_ws_manager():
        raise RuntimeError("Error-WS manager not init.")
    
    # Subscribe if needed
    if symbol not in get_ws_manager().subscribed_public:
        get_ws_manager().subscribe_ticker(symbol)
    
    price_data = get_ws_manager().prices.get(symbol)
    
    # Use cache if fresh enough
    if price_data:
        age = time.time() - price_data['timestamp']
        if age < max_cache_age:
            return price_data['price']
    
    # Wait for fresh data
    initial_timestamp = price_data.get('timestamp', 0) if price_data else 0
    timeout = 1.0
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        price_data = get_ws_manager().prices.get(symbol)
        if price_data and price_data['timestamp'] > initial_timestamp:
            return price_data['price']
        time.sleep(0.01)
    
    raise TimeoutError(f"No fresh-{symbol}")


# ==========================================================================
# ORDER SIZING & CONTRACT PARAMETERS
# ==========================================================================
def compute_size_base(usdt_amount: float, last_price: Decimal) -> Decimal:
    """Calculate base size from USDT amount."""
    return Decimal(str(usdt_amount)) / last_price


def extract_contract_params(c: Dict, last_price: Decimal) -> Tuple:
    """Extract contract parameters from contract info."""
    if c is None:
        return None, None, None, None, None
    
    try:
        price_tick      = Decimal(f"1e-{int(c['pricePlace'])}")
        size_scale      = int(c['volumePlace'])
        min_trade_num   = Decimal(c['minTradeNum'])
        size_multiplier = Decimal(c['sizeMultiplier'])
        min_trade_usdt  = Decimal(c['minTradeUSDT'])
        
        return price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt
        
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error-extracting contract params: {e}")
        return None, None, None, None, None


def fallback_params(price_tick: Optional[Decimal], 
                    size_scale: Optional[int],
                    last_price: Decimal, 
                    min_trade_num: Optional[Decimal] = None,
                    min_trade_usdt: Optional[Decimal] = None) -> Tuple:
    """Apply fallback values for missing contract parameters."""
    if price_tick is None:
        if last_price >= 1000:
            price_tick = Decimal("0.1")
        elif last_price >= 1:
            price_tick = Decimal("0.01")
        elif last_price >= 0.1:
            price_tick = Decimal("0.001")
        else:
            price_tick = Decimal("0.00001")
    
    if size_scale is None or size_scale < 0:
        size_scale = 6
    
    if min_trade_num is None:
        if last_price >= 100:
            min_trade_num = Decimal("0.01")
        elif last_price >= 10:
            min_trade_num = Decimal("0.1")
        else:
            min_trade_num = Decimal("1")
    
    return price_tick, size_scale, min_trade_num, min_trade_usdt


def quantize_size(size_base: Decimal, size_scale: int) -> Tuple[Optional[Decimal], Decimal]:
    """Quantize size to proper decimal places."""
    precision_size = Decimal(f"1e-{size_scale}")
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    
    if size_q == 0:
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    
    if size_q == 0:
        logger.warning("WAR-Size=0")
        return None, precision_size
    
    return size_q, precision_size


# ==========================================================================
# ORDER EXECUTION
# ==========================================================================
def build_order_body(symbol: str, product_type: str, margin_mode: str, 
                    margin_coin: str, size_q: Decimal, side: str, 
                    client_oid: Optional[str]) -> Dict:
    """Build order body for API request."""
    body = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": side,
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": client_oid if client_oid else f"script-{int(time.time() * 1000000)}"
    }
    return body

def place_market_order(send_request_func, body_order: Dict) -> Tuple[Optional[int], Optional[Dict]]:
    """Place market order via REST API with 1 retry."""
    
    # Primer intento
    code_order, resp_order = send_request_func(
        "POST", 
        "/api/v2/mix/order/place-order", 
        body=body_order
    )
    
    if code_order == 200 and resp_order.get("code") == "00000":
        return code_order, resp_order
    
    # Fall - retry
    logger.warning(f"WAR-Order failed, retrying in 0.5s... {resp_order}")
    time.sleep(0.5) 
    
    code_order, resp_order = send_request_func("POST", "/api/v2/mix/order/place-order", body=body_order)
    
    if code_order != 200 or resp_order.get("code") != "00000":
        logger.error(f"Error-Order failed after retry: {resp_order}")
        return None, None
    
    logger.info("INF-Order placed successfully on retry")
    return code_order, resp_order

def extract_filled_amount(resp_order: Dict, size_q: Decimal) -> Decimal:
    """Extract filled amount from order response."""
    filled_amount = Decimal("0")
    data = resp_order.get("data") or {}
    
    for k in ("baseVolume", "filledQty", "size", "filledSize", "sz", "filled_amount"):
        if k in data and data[k] is not None:
            try:
                filled_amount = Decimal(str(data[k]))
                if filled_amount > 0:
                    break
            except Exception:
                continue
    
    if filled_amount == 0:
        filled_amount = size_q
    
    return filled_amount


def get_exec_price(resp_order: Dict, last_price: Decimal) -> Decimal:
    """Get execution price from order response."""
    return Decimal(str(resp_order['data'].get('price', last_price)))


def place_order(symbol: str,
                direction: str,
                usdt_amount: float = 100,
                product_type: str = PRODUCT_TYPE,
                margin_coin: str = "USDT",
                margin_mode: str = MARGIN_MODE,
                send_request_func=None,
                client_oid: Optional[str] = None) -> Optional[Dict]:

    if send_request_func is None:
        raise ValueError("Error-Send request error.")
        
    # =========================================================================
    # BLOQUE 1: CÓDIGO ORIGINAL (NO SE TOCA)
    # =========================================================================
    # Get current price via WebSocket

    # Get current price via WebSocket
    last_price, _ = fetch_ticker_ws(symbol)
    if last_price is None:
        return None

    # Calculate size
    size_base = compute_size_base(usdt_amount, last_price)
    c         = fetch_contracts_ws(symbol)
    price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = \
        extract_contract_params(c, last_price)
    
    price_tick, size_scale, min_trade_num, min_trade_usdt = \
        fallback_params(price_tick, size_scale, last_price, min_trade_num, min_trade_usdt)

    size_q, _ = quantize_size(size_base, size_scale)
    if size_q is None:
        return None

    side = "buy" if direction.lower() == "long" else "sell"
    
    # Place order via REST API
    body_order = build_order_body(symbol, product_type, margin_mode, margin_coin, 
                                   size_q, side, client_oid)

    # =========================================================================
    # BLOQUE 3: ENVÍO DE ORDEN
    # =========================================================================

    code_order, resp_order = place_market_order(send_request_func, body_order)
        
    if code_order is None:
        logger.warning(f"WAR-:last_price={last_price}, price_tick={price_tick}, "
              f"min_num: {min_trade_num}, min_usdt: {min_trade_usdt}")
        return None

    filled_amount = extract_filled_amount(resp_order, size_q)
    exec_price    = get_exec_price(resp_order, last_price)
    
    # ===============================================================
    # NUEVO: Log de partial fills
    # ===============================================================
    if filled_amount < size_q * Decimal('0.95'):  # Tolerancia 5%
        logger.warning(f"WAR-Partial fill for {symbol}: requested={size_q}, filled={filled_amount}")
    
    logger.debug(f"{direction.upper():<6} {symbol:<10} | Amount: ${usdt_amount:.2f} | "
           f"Price: {exec_price}")
       

    return {
        'success': True,
        'resp_order': resp_order,           # Original (backward compatibility)
        'filled_amount': float(filled_amount) # Amount ejecutado
    }


# ==========================================================================
# ORDER FILLS
# ==========================================================================
def get_fills_for_order(order_id: str, 
                        symbol: str, 
                        product_type: str = PRODUCT_TYPE,
                        send_request_func=None, 
                        delay: float = 0.05) -> Tuple:

    time.sleep(delay)
    
    if not get_ws_manager():
        raise RuntimeError("Error-WebSocket manager not initialized")
    
    # Wait for fills via WebSocket
    start_time = time.time()
    timeout = 1.0
    
    while time.time() - start_time < timeout:
        fills = get_ws_manager().get_fills(order_id)
        if fills:
            # Process fills
            total_base   = Decimal('0')
            weighted     = Decimal('0')
            total_profit = Decimal('0')
            total_fee    = Decimal('0')
            
            for f in fills:
                bv         = f.get("baseVolume")
                price      = f.get("price")
                profit     = f.get("profit")
                fee_detail = f.get("feeDetail", [])
                
                if bv is None or price is None:
                    continue
                
                try:
                    bv_d = Decimal(str(bv))
                    p_d = Decimal(str(price))
                    total_base += bv_d
                    weighted += p_d * bv_d
                    
                    if profit is not None:
                        total_profit += Decimal(str(profit))
                    
                    for fee_item in fee_detail:
                        total_fee_val = fee_item.get("totalFee")
                        if total_fee_val is not None:
                            total_fee += abs(Decimal(str(total_fee_val)))
                except Exception:
                    pass
            
            entry_price = (weighted / total_base) if total_base > 0 and weighted > 0 else None
                        
            return total_base, entry_price, total_profit, total_fee
        
        time.sleep(0.05)
    
    # Timeout
    logger.warning(f"WAR-No fills received for order {order_id} via WebSocket (timeout)")
    return None, None, None, None


# ==========================================================================
# POSITION CLOSING
# ==========================================================================
def close_position(symbol: str, 
                   size: Decimal, 
                   direction: str, 
                   send_request_func,
                   reason: str = "NO_INFO", 
                   position_data: Optional[Dict] = None,
                   bot_state=None) -> bool:

    try:
        close_side = "sell" if direction.lower() == "short" else "buy"
                
        body = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginMode": MARGIN_MODE,
            "marginCoin": "USDT",
            "size": format(size, "f"),
            "side": close_side,
            "tradeSide": "close",
            "orderType": "market"
        }
        
        # Print closing message
        if reason == "TP":
            logger.info(f"TP for {symbol} ({position_data.get('strategy_id', 'N/A') if position_data else 'N/A'}) "
                  f"at {datetime.now(HOUR_ZONE).strftime('%H:%M')}")
        elif reason == "SL":
            logger.info(f"SL for {symbol} ({position_data.get('strategy_id', 'N/A') if position_data else 'N/A'}) "
                  f"at {datetime.now(HOUR_ZONE).strftime('%H:%M')}")            
        elif reason == "TIMEOUT":
            logger.info(f"TIMEOUT for {symbol} ({position_data.get('strategy_id', 'N/A') if position_data else 'N/A'}) "
                  f"at {datetime.now(HOUR_ZONE).strftime('%H:%M')}") 
       
        # =========================================================================
        # BLOQUE 2: NUEVO - CAPTURA PRECIO Y TIMESTAMP PRE-ORDEN
        # =========================================================================
        order_price_close = None
        order_ts_close    = None
        
        try:
            # Get current market price via WebSocket
            current_price = get_current_price(symbol, max_cache_age=0.5)
            order_price_close = float(current_price)
        except Exception as e:
            logger.warning(f"WAR-Could not capture order_price_close for {symbol}: {e}")
        
        order_ts_close = time.time()  # Timestamp just before sending order       
        code, resp = send_request_func("POST", "/api/v2/mix/order/place-order", body=body)
        exec_ts_close = time.time()
        
        time.sleep(0.05)
        
        if code == 200 and resp.get("code") == "00000":
            # Log closed position if position_data provided
            if position_data:
                data = resp.get('data', {})
                order_id = data.get('orderId')
                
                if order_id:
                    _, close_price_from_fills, profit_from_api, fee_from_api = \
                        get_fills_for_order(order_id=order_id, symbol=symbol, 
                                          send_request_func=send_request_func)
                    
                    if close_price_from_fills is None:
                        close_price_from_fills = Decimal(str(data.get('price', 0)))
                        if close_price_from_fills == 0:
                            close_price_from_fills = get_current_price(symbol)
                            
                    logger.debug(f"[DEBUG POSITION_DATA] {position_data}")
                            
                    tp_target = position_data.get('tp')
                    sl_target = position_data.get('sl')
                    logger.debug(f"[DEBUG] Closing position - tp_target={tp_target}, sl_target={sl_target}")
                    if close_price_from_fills:
                        log_closed_position(
                            opened_at=position_data.get('opened_at'),
                            strategy_id=position_data.get('strategy_id'),
                            symbol=symbol,
                            direction=direction,
                            usdt_amount=position_data.get('usdt_amount', 0),
                            entry_price=position_data.get('entry_price'),
                            close_price=close_price_from_fills,
                            reason=reason,
                            size=size,
                            profit_from_api=profit_from_api,
                            fee_from_api=fee_from_api,
                            bot_state=bot_state,
                            order_price_close=order_price_close,
                            order_ts_close=order_ts_close,
                            exec_ts_close=exec_ts_close,
                            tp_target=tp_target,
                            sl_target=sl_target  
                        )
            
            return True
        
        else:
            logger.warning(f"WAR-No closing position available {symbol}: {resp}")
            if resp.get("code") == "22002":
                logger.warning(f"WAR-Removing from local record (non-existent position)")
                if position_data:
                    logger.debug(f"[DEBUG POSITION_DATA OUT_OF_MARGIN] {position_data}")
                    tp_target = position_data.get('tp')
                    sl_target = position_data.get('sl')
                    logger.debug(f"[DEBUG OUT_OF_MARGIN] tp_target={tp_target}, sl_target={sl_target}")

                    current_price = get_current_price(symbol)
                    if current_price:
                        log_closed_position(
                            opened_at=position_data.get('opened_at'),
                            strategy_id=position_data.get('strategy_id'),
                            symbol=symbol,
                            direction=direction,
                            usdt_amount=position_data.get('usdt_amount', 0),
                            entry_price=position_data.get('entry_price'),
                            close_price=current_price,
                            reason="GHOST_POSITION",
                            size=size,
                            profit_from_api=None,
                            fee_from_api=None,
                            order_price_close=order_price_close,
                            order_ts_close=order_ts_close,
                            exec_ts_close=exec_ts_close,
                            tp_target=tp_target,  
                            sl_target=sl_target 
                        )
                return True  # Remove from open_positions
            return False
            
    except Exception as e:
        logger.error(f"Error-closing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False
