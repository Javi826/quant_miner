#BOT_crypto/core/orchestator.py
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
# Ensure BOT_crypto is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "broker_client", "broker_api")))
from api_client import get_futures_symbols_from_api
from market_data import load_final_symbols, init_websocket
from risk_control import RiskLimiter, ExposureCalculator
from validation import validate_strategy_configuration,validate_settings,validate_postgresql_connection
from validation.candle_validator import init_candle_validator, get_candle_validator, stop_candle_validator  # CANDLE
from api.backend import DashboardServer, create_dashboard_template
from execution import configure_paths, get_current_price, get_usdt_balance_ws, BitgetClient
from state import  check_candles_timeout_for_strategy
from bot_utils import calculate_next_candle_time, group_strategies_by_timeframe
from bot_utils import get_unique_timeframes
from strategies import StrategyProcessor, IMPLEMENTED_STRATEGIES,load_strategies
from config.utils.utils import get_account_config
from config.settings import PRODUCT_TYPE, CHECK_INTERVAL,HOUR_ZONE
from config.settings import LEVERAGE
from core.split_brain_checker import check_split_brain
from state.state_manager import BotState, configure_postgres as sm_configure_postgres, configure_demo as sm_configure_demo
from execution.trade_logger import configure_postgres as tl_configure_postgres

class BotOrchestrator:
    
    def __init__(
        self,
        account_number: str,
        bitget_client: BitgetClient,
        connect_bitget_func: callable,
        active_strategy_ids: Optional[List[str]] = None
    ):

        self.account_number = account_number
        self.logger = logging.getLogger(f'BOT_crypto.core.orchestrator.{account_number}')
        
        # Account configuration
        self.config          = get_account_config(account_number)
        self.initial_capital = self.config['initial_capital']
        self.dashboard_port  = self.config['dashboard_port']
        self.base_dir        = self.config['paths']['base_dir']
        self.state_file      = self.config['paths']['state_file']
        self.trades_log_path = self.config['paths']['trades_file']
        self.log_file_path   = self.config['paths']['log_file']
        # Account feature flags
        self.operative = None
        self.account_flags = self.config
        sm_configure_postgres(self.account_number)
        sm_configure_demo(self.account_number)
        tl_configure_postgres(self.account_number)
        
        # API clients
        self.bitget_client = bitget_client
        self.connect_bitget_func = connect_bitget_func
        
        # Bot state (encapsulated)
        self.open_positions: Dict[str, List[Dict]] = {}
        self.strategy_candles: Dict[str, int] = {}
        self.bot_state: Optional[BotState] = None
        
        # Strategies
        self.strategies: List[Dict] = []
        self.active_strategy_ids = active_strategy_ids
        
        # Market data & execution
        self.exchange = None
        self.ws_manager = None
        self.dashboard = None
        self.strategy_processor = None
        
        # Runtime state
        self.all_symbols: List[str]                  = []
        self.final_by_strat: Dict[str, List[str]]    = {}
        self.strategies_by_tf: Dict[str, List[Dict]] = {}
        self.unique_timeframes: List[str]            = []
        self.next_candle_times: Dict[str, datetime] = {}
        self.last_tpsl_check: float = 0
        
        # Control flags
        self._running = False
        self._initialized = False

        #Risk
        self.risk_limiter: Optional[RiskLimiter] = None
        self.exposure_calculator: Optional[ExposureCalculator] = None
        
    # ======================================================================
    # PUBLIC API
    # ======================================================================
    
    def run(self) -> None:

        if not self._initialized:
            self.initialize()
        
        self._running = True
        self._log_startup()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.shutdown()
    
    def initialize(self) -> None:

        if self._initialized:
            self.logger.warning("WAR-Bot already initialized, skipping...")
            return
        
        self._setup_directories()
        self._log_account_flags()
        self._load_bot_state()
        self._load_and_validate_strategies() 
        self._initialize_risk_management() 
        self._load_market_symbols()
        self._initialize_connections()
        self._start_dashboard()
        self._initialize_websocket()
        self._calculate_next_candles()
        init_candle_validator(self.config.get('reference_symbol', 'BTCUSDT'), self.unique_timeframes)  # CANDLE
        
        self._initialized = True
        self.logger.info("BOT Initialization completed\n")
        
        # Attach operative references after initialization
        self.operative.attach(
            open_positions=self.open_positions,
            strategy_candles=self.strategy_candles,
            strategies=self.strategies
        )
        # Update bot_state reference in operative
        self.operative.bot_state = self.bot_state
        
    def shutdown(self) -> None:

        self._running = False
        try:
           self.operative.save_state()
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR saving state during shutdown")
            self.logger.error(f"Account: {self.account_number}, Positions: {sum(len(p) for p in self.open_positions.values())}")
            self.logger.error(f"Error: {e}")
        stop_candle_validator()  # CANDLE
        self.logger.info("⛔ BOT Stopped")
    
    def get_status(self) -> Dict[str, Any]:

        total_positions   = sum(len(positions) for positions in self.open_positions.values())
        active_strategies = sum(1 for s in self.strategies if s.get('active', True))
        
        return {
            'account_number': self.account_number,
            'running': self._running,
            'initialized': self._initialized,
            'total_positions': total_positions,
            'active_strategies': active_strategies,
            'total_strategies': len(self.strategies),
            'websocket_connected': self.ws_manager.is_connected() if self.ws_manager else False,
            'dashboard_port': self.dashboard_port,
            'total_profit': self.bot_state.closed_total_profit if self.bot_state else 0
        }
        
    # ======================================================================
    # INITIALIZATION METHODS (Private)
    # ======================================================================
    
    def _log_account_flags(self) -> None:
        """Log account configuration flags at startup."""
        capital_str = f"${self.initial_capital:,.0f}"
        risk        = self.account_flags.get('risk_control_enabled', True)
        pg          = self.account_flags.get('postgresql_enabled', True)
        self.logger.info(f"[{self.account_number}] ════ Account Configuration ════")
        self.logger.info(f"[{self.account_number}] Initial capital:  {capital_str}")
        self.logger.info(f"[{self.account_number}] Risk control:     {'✅ enabled' if risk else '❌ disabled'}")
        self.logger.info(f"[{self.account_number}] PostgreSQL:       {'✅ enabled' if pg else '❌ disabled'}") 
        
    def _setup_directories(self) -> None:
        """Setup necessary directories and paths."""
        os.makedirs(self.base_dir, exist_ok=True)
        configure_paths(self.trades_log_path)
    
    def _load_bot_state(self) -> None:
        self.open_positions, self.strategy_candles = self.operative.load_state()
    
        self.bot_state = BotState()
        if os.path.exists(self.trades_log_path):
            import pandas as pd
            df = pd.read_excel(self.trades_log_path)
            if not df.empty:
                self.bot_state.closed_total_profit = df['PROFIT'].sum()
                
    
    def _load_and_validate_strategies(self) -> None:
        # Load strategies
        self.strategies = load_strategies(self.account_number)
        self.strategies = [
            s for s in self.strategies
            if s['id'] in IMPLEMENTED_STRATEGIES or 'specs' in s
        ]
        
        # Apply --set-active
        if self.active_strategy_ids:
            from strategies.strategy_loader import apply_set_active_argument
            apply_set_active_argument(self.strategies, self.active_strategy_ids)
        
        # Validate
        self.logger.info(f"Validating configuration...")
        self.logger.info(f"{'-' * 48}")
        
        if self.account_flags.get('postgresql_enabled', True):
            validate_postgresql_connection()
        
        all_errors   = []
        all_warnings = []
              
        # 1. Strategies
        strat_errors, strat_warnings = validate_strategy_configuration(self.strategies, IMPLEMENTED_STRATEGIES)
        all_errors.extend(strat_errors)
        all_warnings.extend(strat_warnings)
                
        # 3. Settings
        settings_errors, settings_warnings = validate_settings()
        all_errors.extend(settings_errors)
        all_warnings.extend(settings_warnings)
        
        
        # Check errors
        if all_errors:
            self.logger.error(f"{'=' * 48}")
            self.logger.error(f"CONFIGURATION ERRORS FOUND:\n")
            for err in all_errors:
                self.logger.error(f"  {err}")
            self.logger.error(f"\n⛔ BOT STOPPED - Fix configuration")
            self.logger.error(f"{'=' * 48}\n")
            
            # FLUSH antes de raise
            for handler in logging.getLogger('BOT_crypto').handlers:
                handler.flush()
            
            raise ValueError("Invalid configuration")
        
        if all_warnings:
            self.logger.warning(f"CONFIGURATION WARNINGS:")
            for warn in all_warnings:
                self.logger.warning(f"{warn}")
    
    def _load_market_symbols(self) -> None:
        """Load market symbols for each strategy."""
        self.all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
        
        self.logger.debug(f"Operative Strategies: {len(self.strategies)}")
        self.logger.debug(f"{'-' * 48}")
        
        for strat in self.strategies:
            if not strat.get('active', True):
                self.logger.debug(f"{strat['id']:<24}: DEPRECATED")
                continue
            
            self.final_by_strat[strat['id']] = load_final_symbols(
                self.all_symbols,
                strat['symbols'],
                strategy=strat['id'],
                timeframe=strat['timeframe'],
            )
            self.logger.debug(
                f"{strat['id']:<24} : {len(self.final_by_strat[strat['id']]):>2} symbols"
            )
        
        # Group strategies by timeframe
        self.strategies_by_tf  = group_strategies_by_timeframe(self.strategies)
        self.unique_timeframes = get_unique_timeframes(self.strategies)
        
        self.logger.info(f"Detected timeframes: {', '.join(self.unique_timeframes)}")
    
    def _initialize_connections(self) -> None:
        """Initialize exchange and strategy processor."""
        self.exchange = self.connect_bitget_func()
        
        self.strategy_processor = StrategyProcessor(
            send_request_func=self._send_request_wrapper,
            get_balance_func=get_usdt_balance_ws,
            hour_zone=HOUR_ZONE,
            account_number=self.account_number,
            state_file=self.state_file,
        )

        self.strategy_processor.operative = self.operative
    
    def _start_dashboard(self) -> None:
        """Start the web dashboard."""
        self.logger.info(f"Starting Web...")
        self.logger.info(f"{'-' * 48}")
        
        create_dashboard_template(self.base_dir)
        
        self.dashboard = DashboardServer(
            account_number=self.account_number,
            base_dir=self.base_dir,
            get_current_price_func=get_current_price,
            get_balance_func=get_usdt_balance_ws,
            strategies_config=self.strategies,
            initial_capital=self.initial_capital,
            implemented_strategies=IMPLEMENTED_STRATEGIES,
            symbols_by_strategy=self.final_by_strat,
            unique_timeframes=self.unique_timeframes
        )
        
        self.dashboard.start(port=self.dashboard_port)
        self.logger.debug(f"Bot monitoring at http://localhost:{self.dashboard_port}")
    
    def _initialize_websocket(self) -> None:
        """Initialize WebSocket connections."""
        self.logger.info(f"Init WebSocket...")
        self.logger.info(f"{'-' * 48}")
        
        self.ws_manager = init_websocket(
            api_key=self.bitget_client.api_key,
            api_secret=self.bitget_client.api_secret,
            api_passphrase=self.bitget_client.api_passphrase
        )
        
        # Pre-load contracts
        if self.ws_manager:
            all_strategy_symbols = set()
            for strat_id, symbols in self.final_by_strat.items():
                all_strategy_symbols.update(symbols)
            
            if all_strategy_symbols:
                self.ws_manager.preload_contracts(
                    list(all_strategy_symbols),
                    product_type=PRODUCT_TYPE
                )
    
    def _calculate_next_candles(self) -> None:
        """Calculate next candle close times."""
        self.logger.info(f"Candles incoming:")
        self.logger.info(f"{'-' * 48}")
        
        for tf in self.unique_timeframes:
            self.next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
            self.logger.info(
                f"Next for {tf:<5} : "
                f"{self.next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S'):<18} UTC"
            )
        
        self.last_tpsl_check = time.time()
        
    def _initialize_risk_management(self) -> None:
        """Initialize risk management components."""
        from risk_control import ExposureCalculator, RiskLimiter
        
        self.exposure_calculator = ExposureCalculator(logger=self.logger)
        self.risk_limiter = RiskLimiter(
            initial_capital=self.initial_capital,
            logger=self.logger
        )
        
    # ======================================================================
    # MAIN LOOP (Private)
    # ======================================================================
    
    def _main_loop(self) -> None:

        # Split-brain protection (only checks on LOCAL)
        while self._running:
            check_split_brain(self)
            current_time      = time.time()
            now_datetime      = datetime.now(HOUR_ZONE)
            closed_timeframes = self._get_closed_timeframes(now_datetime)
            
            if closed_timeframes:
                self._process_new_candles(closed_timeframes, now_datetime)
            else:
                self._periodic_tpsl_check(current_time)
            
            time.sleep(0.05)
    
    def _get_closed_timeframes(self, now_datetime: datetime) -> List[str]:
        """Check which timeframes have closed candles."""
        closed_timeframes = []
        for tf in self.unique_timeframes:
            if now_datetime >= self.next_candle_times[tf]:
                closed_timeframes.append(tf)
        return closed_timeframes
    
    def _process_new_candles(
        self,
        closed_timeframes: List[str],
        now_datetime: datetime
    ) -> None:

        self.logger.info(f"{'=' * 48}")
        self.logger.info(f"New candles {now_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self.logger.info(f"Timeframes: {', '.join(closed_timeframes)}")
        
        # CANDLE
        _validator = get_candle_validator()
        if _validator:
            for tf in closed_timeframes:
                _validator.register_clock_trigger(tf)
        
        # ========================================================================
        # BROKER SYNC: Load latest positions from exchange
        # ========================================================================
        self.operative.sync_broker()
        
        now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f"Searching Signals... - {now}")
        self.logger.info(f"{'-' * 48}")
        
        # Get strategies to process
        strategies_to_process = []
        for tf in closed_timeframes:
            strategies_to_process.extend(self.strategies_by_tf[tf])
                
        # ========================================================================
        # TIMEOUT CHECK: Increment candles and close expired positions
        # ========================================================================
        self._process_candle_timeouts(strategies_to_process)

        # ========================================================================
        # SIGNAL SEARCH
        # ========================================================================
        self._search_signals(strategies_to_process)

        self.logger.info("Signal cycle completed")
        self.logger.info(f"{'=' * 48}\n")
        
        # Recalculate next candle times
        self._update_next_candle_times(closed_timeframes)        
        self.last_tpsl_check = time.time()
   
    def _process_candle_timeouts(self, strategies_to_process: List[Dict]) -> None:

        for strat in strategies_to_process:
            strat_id = strat['id']
            has_positions = (
                strat_id in self.open_positions and
                len(self.open_positions[strat_id]) > 0
            )
            
            if has_positions:
                self.operative.increment_candles(strat_id)
            
                candles       = self.strategy_candles.get(strat_id, 0)
                num_positions = len(self.open_positions.get(strat_id, []))
            
                self.logger.info(
                    f"Skip {strat_id:<23} {candles:>2}/"
                    f"{strat['sell_after_ncandles']:<2} | {num_positions:>2} pos."
                )
            
                if self.account_flags.get('postgresql_enabled', True):
                    check_candles_timeout_for_strategy(
                        strat_id,
                        strat['sell_after_ncandles'],
                        self.open_positions,
                        self.strategy_candles,
                        self.account_number,
                        self.state_file,
                        self._send_request_wrapper,
                        bot_state=self.bot_state
                    )
                else:
                    self.operative.monitor_exits()
                    
    def _detect_and_execute(
        self,
        strat:          Dict,
        final_symbols:  List[str],
        adjusted_amount: float,
    ) -> None:     
        strat_id = strat['id']
        self.logger.debug(f"[D&E] {strat_id} | symbols={len(final_symbols)}")
  
        signals = self.strategy_processor.detect_signals(
            strat         = strat,
            final_symbols = final_symbols,
            exchange      = self.exchange,
        )
        approved_signals = signals
    
        self.strategy_processor.execute_signals(
            strat            = strat,
            signals          = approved_signals,
            open_positions   = self.open_positions,
            strategy_candles = self.strategy_candles,
            order_amount     = adjusted_amount,
        )
    
    def _search_signals(self, strategies_to_process: List[Dict]) -> None:

        for strat in strategies_to_process:
            strat_id = strat['id']
    
            # ====================================================================
            # STRATEGY PRE-CHECKS: Skip if deprecated or has positions
            # ====================================================================
            if not strat.get('active', True):
                continue
    
            num_positions = len(self.open_positions.get(strat_id, []))
            if num_positions > 0:
                continue
    
            # ====================================================================
            # ORDER AMOUNT
            # ====================================================================
            adjusted_amount = strat['order_amount']
    
            # ====================================================================
            # RISK CHECK
            # ====================================================================
            if self.account_flags.get('risk_control_enabled', True):
                current_exposure = self.exposure_calculator.calculate_current_exposure(
                    open_positions=self.open_positions,
                    closed_pnl=self.bot_state.closed_total_profit if self.bot_state else 0,
                    initial_capital=self.initial_capital,
                    leverage=LEVERAGE
                )
                blocked, reason, risk_metadata = self.risk_limiter.is_at_limit(
                    current_gross_pct=current_exposure['gross_exposure_pct']
                )
                log_msg = self.risk_limiter.format_log_message(strat_id, risk_metadata)
                self.logger.info(log_msg)
                if blocked:
                    continue
    
            # ====================================================================
            # SIGNAL PROCESSING: DETECT + SIZING + EXECUTE
            # ====================================================================
            try:
                self._detect_and_execute(
                    strat           = strat,
                    final_symbols   = self.final_by_strat.get(strat_id, []),
                    adjusted_amount = adjusted_amount,
                )
            except Exception as e:
                self.logger.warning(f"WAR-first try processing {strat_id}: {e}")
                opened_symbols = [pos['symbol'] for pos in self.open_positions.get(strat_id, [])]
    
                # ================================================================
                # RETRY: Some positions already opened — skip those symbols
                # ================================================================
                if opened_symbols:
                    self.logger.info(
                        f"Retrying {strat_id} after 3 seconds... "
                        f"({len(opened_symbols)} positions already opened, will skip those symbols)"
                    )
                    time.sleep(3)
                    remaining_symbols = [
                        s for s in self.final_by_strat.get(strat_id, [])
                        if s not in opened_symbols
                    ]
                    if remaining_symbols:
                        try:
                            self._detect_and_execute(
                                strat           = strat,
                                final_symbols   = remaining_symbols,
                                adjusted_amount = adjusted_amount,
                            )
                            self.logger.info(f"Retry successful for {strat_id} ({len(remaining_symbols)} remaining symbols processed)")
                        except Exception as e2:
                            self.logger.error(f"Error-Retry failed for {strat_id}: {e2}")
                    else:
                        self.logger.info(f"No remaining symbols to retry for {strat_id}")
    
                # ================================================================
                # RETRY: No positions opened yet — full retry
                # ================================================================
                else:
                    self.logger.info(f"Retrying {strat_id} after 3 seconds... (no positions opened yet)")
                    time.sleep(3)
                    try:
                        self._detect_and_execute(
                            strat           = strat,
                            final_symbols   = self.final_by_strat.get(strat_id, []),
                            adjusted_amount = adjusted_amount,
                        )
                        self.logger.info(f"Retry successful for {strat_id}")
                    except Exception as e2:
                        self.logger.error(f"Error-Retry failed for {strat_id}: {e2}")
  
    def _periodic_tpsl_check(self, current_time: float) -> None:

        if current_time - self.last_tpsl_check >= CHECK_INTERVAL:
            self.operative.monitor_exits()
            
            self.last_tpsl_check = current_time
    
    def _update_next_candle_times(self, closed_timeframes: List[str]) -> None:

        for tf in closed_timeframes:
            self.next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
            self.logger.info(
                f"Next for {tf}: "
                f"{self.next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )  
    # ======================================================================
    # HELPER METHODS (Private)
    # ======================================================================      
    def _send_request_wrapper(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None
    ) -> Any:

        return self.bitget_client.send_request(method, path, params, body)  
    def _log_startup(self) -> None:
        """Log bot startup information."""
        self.logger.info(f"{'=' * 48}")
        self.logger.info(f"STARTING BOT IN ACCOUNT: {self.account_number}")
        self.logger.info(f"{'=' * 48}")