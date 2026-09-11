#BOT_crypto/market_data/websocket_manager.py
import logging

logger = logging.getLogger('BOT_crypto.market_data.websocket_manager')
import time
import json
import hmac
import base64
import hashlib
import websocket
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal

# ==========================================================================
# WEBSOCKET MANAGER - EXTENDED
# ==========================================================================
class BitgetWSManager:
    """Gestor de WebSocket unificado para canales públicos y privados"""
    
    def __init__(self, api_key=None, api_secret=None, api_passphrase=None,
                 public_url="wss://ws.bitget.com/v2/ws/public",
                 private_url="wss://ws.bitget.com/v2/ws/private"):
        self.public_url = public_url
        self.private_url = private_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        
        # WebSockets separados
        self.public_ws = None
        self.private_ws = None
        
        # Data storage
        self.prices = {}
        self.positions = {}
        self.orders = {}
        self.fills = {}
        self.account = {}
        self.equity = {}  # ⭐ Balance/equity data
        self.contracts = {}  # ⭐ Caché de contratos
        
        # Control
        self.subscribed_public = set()
        self.subscribed_private = set()
        self.running = False
        self.authenticated = False  # ⭐ Flag de autenticación
        
        # Tracking de desconexiones
        self.last_close_code = None
        self.last_close_msg = None
        
        # Threads
        self.public_thread = None
        self.private_thread = None
        self.ping_thread = None
        
        # Callbacks para eventos
        self.on_fill_callback = None
        self.on_order_callback = None
        self.on_position_callback = None
        
    def start(self):
        """Inicia ambos WebSockets"""
        if self.running:
            return
        self.running = True
        
        # Start public WS
        self.public_thread = threading.Thread(target=self._run_public, daemon=True)
        self.public_thread.start()
        
        # Start private WS (si hay credenciales)
        if self.api_key and self.api_secret:
            self.private_thread = threading.Thread(target=self._run_private, daemon=True)
            self.private_thread.start()
        
        # Start ping thread
        self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self.ping_thread.start()
        
    def _ping_loop(self):
        """Envía un ping de aplicación (cadena 'ping') cada 30s y keepalive de bajo impacto.
        También se encarga de reconexiones suaves si el socket está caído."""
        ping_interval = 25.0
        while self.running:
            try:
                # Enviar PING como cadena simple (Bitget espera "ping" como string, no JSON)
                if self.public_ws and getattr(self.public_ws, 'sock', None) and getattr(self.public_ws.sock, 'connected', False):
                    try:
                        self.public_ws.send("ping")
                    except Exception as e:
                        logger.error(f"Error-sending public ping: {e}")
    
                if self.private_ws and getattr(self.private_ws, 'sock', None) and getattr(self.private_ws.sock, 'connected', False):
                    try:
                        self.private_ws.send("ping")
                    except Exception as e:
                        logger.error(f"Error-sending private ping: {e}")
    
                # Esperar intervalo (loop más fino para reaccionar a stop)
                slept = 0.0
                while self.running and slept < ping_interval:
                    time.sleep(0.5)
                    slept += 0.5
    
            except Exception as e:
                logger.error(f"Error-Ping loop failed: {e}")
                time.sleep(1)

    
    # ==========================================================================
    # PUBLIC WEBSOCKET
    # ==========================================================================
    def _run_public(self):
        """Loop principal del WebSocket público"""
        while self.running:
            try:
                self.public_ws = websocket.WebSocketApp(
                    self.public_url,
                    on_message=self._on_public_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_public_open,
                    on_pong=self._on_pong
                )
                self.public_ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                logger.error(f"Error-Public WS error: {e}")
                time.sleep(0.5)
    
    def _on_public_open(self, ws):
        """Callback al conectar WS público"""
        now = datetime.now(ZoneInfo('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')
        logger.debug(f"PUBLIC- WS connected [{now}]")
        if self.subscribed_public:
            self._resubscribe_public()
    
    def _on_public_message(self, ws, message):
        try:
            # Ignorar pongs y mensajes no-JSON
            if not message or message == "pong":
                return
            if message[0] not in ("{", "["):
                return
    
            data = json.loads(message)
    
            if data.get('event') in ('pong', 'subscribe'):
                return
    
            if data.get('action') in ('snapshot', 'update'):
                arg = data.get('arg', {})
                if arg.get('channel') == 'ticker':
                    symbol = arg.get('instId')
                    ticker_data = data.get('data', [])
                    if ticker_data and symbol:
                        last_pr = ticker_data[0].get('lastPr')
                        if last_pr:
                            self.prices[symbol] = {
                                'price': Decimal(last_pr),
                                'timestamp': time.time()
                            }
    
        except Exception as e:
            logger.error(f"Error-processing public message: {e}")
  
    def subscribe_ticker(self, symbol):
        """Suscribe a ticker de un símbolo"""
        if symbol in self.subscribed_public:
            return
        
        msg = {
            "op": "subscribe",
            "args": [{
                "instType": "USDT-FUTURES",
                "channel": "ticker",
                "instId": symbol
            }]
        }
        
        if self.public_ws and self.public_ws.sock and self.public_ws.sock.connected:
            self.public_ws.send(json.dumps(msg))
            self.subscribed_public.add(symbol)
            logger.debug(f"Subscribed to ticker {symbol} via WebSocket")  # Silenciar para reducir spam
        else:
            self.subscribed_public.add(symbol)
            logger.warning(f"WAR-Cant subs to {symbol} - Public WS not connected.")
    
    def _resubscribe_public(self):
        """Resuscribe a canales públicos"""
        for symbol in self.subscribed_public:
            msg = {
                "op": "subscribe",
                "args": [{
                    "instType": "USDT-FUTURES",
                    "channel": "ticker",
                    "instId": symbol
                }]
            }
            if self.public_ws:
                self.public_ws.send(json.dumps(msg))
    
    # ==========================================================================
    # PRIVATE WEBSOCKET
    # ==========================================================================
    def _run_private(self):
        """Loop principal del WebSocket privado"""
        while self.running:
            try:
                self.private_ws = websocket.WebSocketApp(
                    self.private_url,
                    on_message=self._on_private_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_private_open,
                    on_pong=self._on_pong
                )
                self.private_ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                logger.error(f"Error-Private WS error: {e}")
                time.sleep(0.5)
    
    def _on_private_open(self, ws):
        """Callback al conectar WS privado - autenticar"""
        # Detectar si es reconexión (authenticated ya es True)
        is_reconnect = self.authenticated
        now = datetime.now(ZoneInfo('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        if is_reconnect:
            # Mostrar razón de la desconexión anterior si existe
            if self.last_close_code or self.last_close_msg:
                logger.debug(f"PRIVATE WS connected [{now}]")
                logger.debug(f"Previous: code={self.last_close_code}, msg={self.last_close_msg}")  # ⭐ Más legible
                self.last_close_code = None
                self.last_close_msg = None
            else:
                # Sin código de cierre guardado - puede ser timeout sin aviso
                logger.debug(f"PRIVATE WS connected [{now}]")
                logger.debug(f"Previous: code=unknown (likely timeout)")
        else:
            logger.debug(f"PRIVATE WS connected [{now}]")
        
        self._authenticate()
        time.sleep(0.5)
        self._subscribe_private_channels(is_reconnect=is_reconnect)
    
    def _authenticate(self):
        """Autentica el WebSocket privado"""
        timestamp = str(int(time.time()))
        sign_str = timestamp + 'GET' + '/user/verify'
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                sign_str.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        auth_msg = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.api_passphrase,
                "timestamp": timestamp,
                "sign": signature
            }]
        }
        
        if self.private_ws and self.private_ws.sock and self.private_ws.sock.connected:
            self.private_ws.send(json.dumps(auth_msg))
    
    def _subscribe_private_channels(self, is_reconnect=False):
        """Suscribe a canales privados esenciales"""
        channels = ['orders', 'fill', 'positions', 'account', 'equity']
        
        if not is_reconnect:
            logger.debug(f"Subscribing to {len(channels)} private channels...")
        
        for channel in channels:
            msg = {
                "op": "subscribe",
                "args": [{
                    "instType": "USDT-FUTURES",
                    "channel": channel,
                    "instId": "default" if channel in ['orders', 'fill', 'positions'] else None,
                    "coin": "default" if channel == 'account' else None
                }]
            }
            # Remove None values
            msg["args"][0] = {k: v for k, v in msg["args"][0].items() if v is not None}
            
            if self.private_ws and self.private_ws.sock and self.private_ws.sock.connected:
                self.private_ws.send(json.dumps(msg))
                self.subscribed_private.add(channel)
                # Solo mostrar en primera conexión
                if not is_reconnect:
                    logger.debug(f"OK-{channel}")
    
    def _on_private_message(self, ws, message):
        try:
            #  ignorar pong y basura
            if not message or message == "pong":
                return
            if message[0] not in ("{", "["):
                return
    
            data = json.loads(message)
    
            if data.get("event") in ("pong", "subscribe"):
                return
    
            # Login
            if data.get("event") == "login":
                code = data.get("code")
                if code == "0" or code == 0:
                    logger.info("WS authentication successful")
                    self.authenticated = True
                else:
                    logger.error(f"Error-WebSocket auth failed: {data}")
                    self.authenticated = False
                return
    
            arg = data.get("arg", {})
            channel = arg.get("channel")
            action = data.get("action")
    
            if not channel or action not in ("snapshot", "update"):
                return
    
            data_list = data.get("data", [])
    
            if channel == "orders":
                for order in data_list:
                    oid = order.get("orderId")
                    if oid:
                        self.orders[oid] = order
                        if self.on_order_callback:
                            self.on_order_callback(order)
    
            elif channel == "fill":
                for fill in data_list:
                    oid = fill.get("orderId")
                    if oid:
                        self.fills.setdefault(oid, []).append(fill)
                        if self.on_fill_callback:
                            self.on_fill_callback(fill)
    
            elif channel == "positions":
               if action == "snapshot":
                   self.positions.clear()
           
               for pos in data_list:
                   symbol = pos.get("instId")
                   hold_side = pos.get("holdSide", "").lower()
                   total = float(pos.get("total", 0))
                   
                   if symbol and hold_side:
                       # Use composite key for hedge mode support
                       key = f"{symbol}_{hold_side}"
                       
                       if total > 0:
                           self.positions[key] = pos
                       else:
                           self.positions.pop(key, None)
    
            elif channel == "account":
                for acc in data_list:
                    coin = acc.get("marginCoin")
                    if coin:
                        self.account[coin] = acc
    
            elif channel == "equity":
                for eq in data_list:
                    self.equity = eq
    
        except Exception as e:
            logger.error(f"Error-processing private message: {e}")

    
    # ==========================================================================
    # COMMON CALLBACKS
    # ==========================================================================
    def _on_pong(self, ws, message):
        """Callback al recibir pong"""
        #logger.debug("✅ Pong received")
        pass
    
    def _on_error(self, ws, error):
        """Callback en caso de error"""
        now = datetime.now(ZoneInfo('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')
        error_str = str(error)
        
        # Identificar tipo de WebSocket
        ws_type = "PUBLIC" if ws == self.public_ws else "PRIVATE" if ws == self.private_ws else "UNKNOWN"
        
        # Errores de red conocidos (no mostrar traceback)
        network_errors = [
            "Temporary failure in name resolution",  
            "Name or service not known",              
            "Connection refused",                     
            "Connection reset by peer",               
            "timeout",                                
            "timed out",                              
            "Connection to remote host was lost",     
            "No route to host"                        
        ]
        
        # Verificar si es error de red conocido
        is_network_error = any(err_type.lower() in error_str.lower() for err_type in network_errors)
        
        if is_network_error:
            # Solo warning para errores de red (sin traceback)
            logger.warning(f"WAR-{ws_type}  WS network issue [{now}]")
            logger.warning(f"{error_str}")
            logger.warning(f"Retrying connection...")
        else:
            # Traceback completo para errores inesperados
            logger.error(f"{ws_type}  WS unexpected error [{now}]: {error}")
            import traceback
            traceback.print_exc()
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback al cerrar conexión"""
        # Identificar cuál WebSocket se cerró
        now = datetime.now(ZoneInfo('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')
        ws_type = "PUBLIC " if ws == self.public_ws else "PRIVATE" if ws == self.private_ws else "UNKNOWN"
        
        # Guardar razón de cierre para mostrar en reconexión
        self.last_close_code = close_status_code
        self.last_close_msg = close_msg
        
        # Mostrar desconexión siempre con tipo de WS
        if close_status_code or close_msg:
            logger.warning(f"WAR-{ws_type} WS disconnected [{now}] | code={close_status_code}, msg={close_msg}") 
        else:
            logger.warning(f"WAR-{ws_type} WS disconnected [{now}] | code=None, msg=None (unclean close)")
 
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    def get_price(self, symbol):
        """Obtiene el último precio recibido"""
        data = self.prices.get(symbol)
        if data:
            return data['price']
        return None
    
    def get_fills(self, order_id):
        """Obtiene los fills de una orden"""
        return self.fills.get(order_id, [])
    
    def get_order(self, order_id):
        """Obtiene información de una orden"""
        return self.orders.get(order_id)
    
    def get_position(self, symbol):
        """
        DEPRECATED: Use get_positions_by_symbol() instead.
        
        This method is incompatible with hedge mode (same symbol can have 
        both LONG and SHORT positions). Returns None to prevent silent data loss.
        
        Args:
            symbol: Trading pair (e.g. 'BTCUSDT')
            
        Returns:
            None (always)
        """
        logger.warning(
            f"[WS] get_position() is DEPRECATED - use get_positions_by_symbol(). "
            f"Called with: {symbol}"
        )
        return None
    
    def get_positions_by_symbol(self, symbol):
        """
        Get all positions for a symbol (hedge mode support).
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
        
        Returns:
            dict: {'long': pos_data or None, 'short': pos_data or None}
        """
        result = {'long': None, 'short': None}
        
        for pos_data in self.positions.values():
            if pos_data.get('instId') == symbol:
                hold_side = pos_data.get('holdSide', '').lower()
                if hold_side == 'long':
                    result['long'] = pos_data
                elif hold_side == 'short':
                    result['short'] = pos_data
        
        return result
    
    def get_usdt_balance(self):
        """Obtiene el balance USDT disponible desde el canal equity"""
        if self.equity:
            # unionAvailable es el balance disponible en modo multi-asset margin
            available = self.equity.get('unionAvailable')
            if available:
                return float(available)
            
            # Si no hay unionAvailable, usar usdtEquity
            usdt_equity = self.equity.get('usdtEquity')
            if usdt_equity:
                return float(usdt_equity)
        
        return 0.0
    
    def refresh_positions(self):
        """
        Fuerza actualización REAL de posiciones re-suscribiéndose al canal.
        Esto obliga al servidor a enviar un snapshot fresco.
        """
        # Guardar posiciones actuales
        old_positions = {k: v.get('total') for k, v in self.positions.items()}
        
        # RE-SUSCRIBIRSE al canal (esto fuerza un snapshot fresco)
        if self.private_ws and self.private_ws.sock and self.private_ws.sock.connected:
            # Primero desuscribirse
            unsub_msg = {
                "op": "unsubscribe",
                "args": [{
                    "instType": "USDT-FUTURES",
                    "channel": "positions",
                    "instId": "default"
                }]
            }
            self.private_ws.send(json.dumps(unsub_msg))
            time.sleep(0.1)
            
            # Luego re-suscribirse (esto fuerza snapshot)
            sub_msg = {
                "op": "subscribe",
                "args": [{
                    "instType": "USDT-FUTURES",
                    "channel": "positions",
                    "instId": "default"
                }]
            }
            self.private_ws.send(json.dumps(sub_msg))
            
            # Esperar a recibir el snapshot fresco
            time.sleep(0.1)
            
            return True
        else:
            return False
    
    def get_contract(self, symbol):
        """Obtiene información del contrato (con caché)"""
        return self.contracts.get(symbol)
    
    def set_contract(self, symbol, contract_data):
        """Guarda información del contrato en caché"""
        self.contracts[symbol] = contract_data
        
    def stop(self):
        """Detiene los WebSockets"""
        self.running = False
        if self.public_ws:
            self.public_ws.close()
        if self.private_ws:
            self.private_ws.close()
    
    def preload_contracts(self, symbols, send_request_func=None, product_type="USDT-FUTURES"):
        """
        Pre-carga contratos via API REST (única excepción necesaria al inicio).
        send_request_func puede ser None - usa requests directo.
        """
        logger.debug(f"Pre-loading contract for {len(symbols)} symbols")
        
        loaded = 0
        
        for symbol in symbols:
            if symbol not in self.contracts:
                try:
                    # Usar requests directo (sin send_request_func)
                    import requests
                    url = f"https://api.bitget.com/api/v2/mix/market/contracts?productType={product_type}&symbol={symbol}"
                    resp = requests.get(url, timeout=10)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("code") == "00000":
                            data_list = data.get("data", [])
                            if data_list:
                                self.contracts[symbol] = data_list[0]
                                loaded += 1
                    
                    time.sleep(0.02)
                except Exception as e:
                    pass
        
        logger.info(f"Pre-loaded {loaded}/{len(symbols)} contracts")


# Instancia global
_ws_manager = None

def init_websocket(api_key=None, api_secret=None, api_passphrase=None):
    """Inicializa el gestor de WebSocket"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = BitgetWSManager(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase
        )
        _ws_manager.start()
        time.sleep(1)  # Dar tiempo a conectar y autenticar
        
        # Verificar conexión
        logger.debug(f"WebSocket status:")
        logger.debug(f"Public- WS    : {'Connected' if _ws_manager.public_ws and _ws_manager.public_ws.sock and _ws_manager.public_ws.sock.connected else 'Not connected'}")
        logger.debug(f"Private WS    : {'Connected' if _ws_manager.private_ws and _ws_manager.private_ws.sock and _ws_manager.private_ws.sock.connected else 'Not connected'}")
        logger.debug(f"Authenticated : {'Yes' if _ws_manager.authenticated else 'No'}")
        
        # Verificar si la autenticación fue exitosa
        if api_key and api_secret:
            # Dar más tiempo para que se complete la autenticación
            for _ in range(5):
                if _ws_manager.authenticated:
                    break
                time.sleep(0.5)
    return _ws_manager

def get_ws_manager():
    """
    Get the global WebSocket manager instance.
    
    Returns:
        BitgetWSManager instance or None if not initialized
    """
    return _ws_manager

