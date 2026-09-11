#BOT_crypto/api/backend.py
import os
import json
import re
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, request
import logging
import time as time_module
import requests
from market_data.websocket_manager import get_ws_manager
logger = logging.getLogger('BOT_crypto.api.backend')
from config.settings import SLIPPAGE_WARNING_PCT, SLIPPAGE_CRITICAL_PCT
import psycopg2
from api.metrics import MetricsCalculator
from config.settings import POSTGRES_CONFIG, RISK_LIMITS, LEVERAGE
from config.settings import HOUR_ZONE
from config.utils.utils import get_account_config
from config.settings import ACCOUNTS

class DashboardServer:
    """Servidor web del dashboard para monitoreo en tiempo real del bot"""
    
    def __init__(self, account_number, base_dir, get_current_price_func, 
                 get_balance_func, strategies_config,
                 initial_capital=0, implemented_strategies=None, symbols_by_strategy=None,
                 unique_timeframes=None):

        self.account_number = account_number
        self.reference_symbol = ACCOUNTS.get(account_number, {}).get('reference_symbol')
        self.base_dir = base_dir
        self.get_current_price = get_current_price_func
        self.get_balance = get_balance_func
        self.strategies = strategies_config
        self.initial_capital = initial_capital
        self.implemented_strategies = implemented_strategies or set()
        self.symbols_by_strategy = symbols_by_strategy or {}
        self.unique_timeframes = unique_timeframes or []
        
        # ========================================================================
        # DEMO MODE DETECTION
        # ========================================================================

        self.is_demo = get_account_config(account_number).get('type') == 'demo'
        
        if self.is_demo:
            self.demo_state_path = os.path.join(base_dir, f'demo_state_{account_number}.json')
            self.demo_trades_path = os.path.join(base_dir, f'bot_trades_{account_number}.xlsx')
            logger.info(f"[DASHBOARD] Demo mode detected for account {account_number}")
            logger.info(f"[DASHBOARD] State: {self.demo_state_path}")
            logger.info(f"[DASHBOARD] Trades: {self.demo_trades_path}")
        # ========================================================================
        
        self.state_file = os.path.join(base_dir, f'bot_state_{account_number}.json')
        self.trades_file = os.path.join(base_dir, f'bot_trades_{account_number}.xlsx')
        self.log_file = os.path.join(base_dir, f'BOT_orchestator_{account_number}.log')
        
        self.templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(self.templates_dir, exist_ok=True)
        
        self.app = Flask(__name__, template_folder=self.templates_dir)
        self.app.last_log_position = 0
        # PostgreSQL configuration
        self.postgres_config = POSTGRES_CONFIG
        
        self._register_routes()
        
        self.server_thread = None
        self.running = False
        
        self.snapshot_thread = None
        self.snapshot_running = False
        self.dashboard_port = None
        
    def get_precision_for_price(self, price):
        """
        Determina el número de decimales a mostrar según la magnitud del precio.
        """
        price = abs(float(price))
        
        if price >= 10000:
            return 1
        elif price >= 1000:
            return 2
        elif price >= 100:
            return 2
        elif price >= 10:
            return 3
        elif price >= 1:
            return 4
        elif price >= 0.01:
            return 5
        else:
            return 5
    
    def _load_trades_dataframe(self):
        """
        Load and validate trades DataFrame from PostgreSQL or Excel (demo mode).
        """
        # DEMO MODE: Read from Excel
        if self.is_demo:
            try:
                if not os.path.exists(self.demo_trades_path):
                    return None
                
                df = pd.read_excel(self.demo_trades_path)
                
                if df.empty:
                    return None
                                
                return df
            
            except Exception as e:
                logger.error(f"Error loading trades from Excel (demo): {e}")
                return None
        
        # LIVE MODE: Read from PostgreSQL (código actual sin cambios)
        try:
            conn = psycopg2.connect(**self.postgres_config)
            query = f"SELECT * FROM trades WHERE account = '{self.account_number}'"
            df = pd.read_sql(query, conn)
            conn.close()
            
            if df.empty:
                return None
            
            # Rename columns to match Excel format (for compatibility)
            df.rename(columns={
                'open_at': 'OPEN_AT',
                'close_at': 'CLOSE_AT',
                'duration_days': 'DURATION_DAYS',
                'strategy': 'STRATEGY',
                'symbol': 'SYMBOL',
                'direction': 'DIRECTION',
                'usdt_amount': 'USDT_AMOUNT',
                'size': 'SIZE',
                'price_entry': 'PRICE_ENTRY',
                'price_close': 'PRICE_CLOSE',
                'profit': 'PROFIT',
                'fee': 'FEE',
                'profit_pct': 'PROFIT_PCT',
                'reason_out': 'REASON_OUT',
                'tp_target': 'TP_TARGET',
                'sl_target': 'SL_TARGET'
            }, inplace=True)
            # Force UTC to avoid local timezone conversion
            for col in ['OPEN_AT', 'CLOSE_AT']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            return df
        except Exception as e:
            logger.error(f"Error loading trades from PostgreSQL: {e}")
            return None
    
    def _prepare_trades_dataframe(self, df):
        """
        Prepara el DataFrame de trades con columnas de fechas procesadas.
        """
        df = df.copy()
        df['OPEN_AT'] = pd.to_datetime(df['OPEN_AT'])
        df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
        df['CLOSE_DATE'] = pd.to_datetime(df['CLOSE_AT'])
        df['DURATION'] = (df['CLOSE_AT'] - df['OPEN_AT']).dt.total_seconds() / 86400
        return df
    
    def _filter_df_by_dates(self, df, date_from=None, date_to=None):

        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        if 'CLOSE_AT' not in df.columns:
            return df
        
        if date_from:
            try:
                date_from_dt = pd.to_datetime(date_from)
                df = df[df['CLOSE_AT'] >= date_from_dt]
            except:
                pass
        
        if date_to:
            try:
                date_to_dt = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                df = df[df['CLOSE_AT'] <= date_to_dt]
            except:
                pass
        
        return df
    
    
    def _load_state(self):

        # DEMO MODE: Read from JSON
        if self.is_demo:
            try:
                if not os.path.exists(self.demo_state_path):
                    return {'positions': {}, 'strategy_candles': {}}
                
                with open(self.demo_state_path, 'r') as f:
                    state = json.load(f)
                
                # JSON structure already matches expected format
                # Keys: 'open_positions', 'strategy_candles'
                # Rename to match PostgreSQL format
                return {
                    'positions': state.get('open_positions', {}),
                    'strategy_candles': state.get('strategy_candles', {})
                }
            
            except Exception as e:
                logger.error(f"Error loading state from JSON (demo): {e}")
                return {'positions': {}, 'strategy_candles': {}}
        
        # LIVE MODE: Read from PostgreSQL (código actual sin cambios)
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT state_data FROM bot_state WHERE account = %s",
                (self.account_number,)
            )
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return result[0]  # JSONB data
            else:
                return {'positions': {}, 'strategy_candles': {}}
                
        except Exception as e:
            logger.error(f"Error loading state from PostgreSQL: {e}")
            return {'positions': {}, 'strategy_candles': {}}
    
    @staticmethod
    def _extract_number_from_id(strategy_id):
        """
        Extrae el número prefijo del ID de estrategia.
        """
        if not strategy_id or not isinstance(strategy_id, str):
            return '??'
        
        match = re.match(r'^(\d{2})_', strategy_id)
        if match:
            return match.group(1)
        
        match = re.match(r'^(\d+)', strategy_id)
        if match:
            return match.group(1).zfill(2)
        
        return '??'
        
    def _calculate_capital_allocation(self, num_strategies=None):
        """
        Calcula el capital asignado por estrategia.
        """
        if num_strategies is None:
            num_strategies = len(self.implemented_strategies)
        
        if num_strategies == 0:
            return 0.0
        
        return self.initial_capital / num_strategies
    
    def _get_full_strategies_list_with_numbers(self):
        """
        Genera la lista completa de estrategias con numeración extraída de los IDs.
        """
        COMMON_PARAMS = {
            'id', 'name', 'timeframe', 'direction', 'active',
            'tp_pct', 'sl_pct', 'order_amount', 'sell_after_ncandles'
        }
        
        EXCLUDE_PARAMS = {'active'}
        
        declared_ids = {s['id'] for s in self.strategies}
        strategies_list = []
        
        for strat in self.strategies:
            is_active = strat.get('active', True)
            status = 'ACTIVE' if is_active else 'DEPRECATING'
            symbols_count = len(self.symbols_by_strategy.get(strat['id'], []))
            
            strategy_dict = {
                'id': strat['id'],
                'name': strat.get('name', strat['id']),
                'timeframe': strat.get('timeframe', 'N/A'),
                'direction': strat.get('direction', 'N/A'),
                'status': status,
                'symbols_count': symbols_count
            }
            
            for key, value in strat.items():
                if key not in strategy_dict and key not in EXCLUDE_PARAMS:
                    strategy_dict[key] = value if value is not None else 'N/A'
            
            for param in COMMON_PARAMS - EXCLUDE_PARAMS - {'id', 'name'}:
                if param not in strategy_dict:
                    strategy_dict[param] = 'N/A'
            
            strategies_list.append(strategy_dict)
        
        not_declared = self.implemented_strategies - declared_ids
        for id in sorted(not_declared):
            strategies_list.append({
                'id': id,
                'name': id,
                'timeframe': 'N/A',
                'direction': 'N/A',
                'status': 'NOT IMPLE.',
                'symbols_count': 0,
                'tp_pct': 'N/A',
                'sl_pct': 'N/A',
                'order_amount': 'N/A',
                'sell_after_ncandles': 'N/A'
            })
        
        strategies_list.sort(key=lambda x: x['id'])
        for strat in strategies_list:
            strat['number'] = self._extract_number_from_id(strat['id'])
        
        return strategies_list
    
    def _register_routes(self):
        """Registra todas las rutas de la API del dashboard"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html', account=self.account_number)
        
        @self.app.route('/favicon.jpg')
        def favicon():
            return send_from_directory(self.base_dir, 'favicon.jpg', mimetype='image/jpeg')
        
        @self.app.route('/api/health')
        def health_check():
            return jsonify({
                'status': 'ready',
                'account': self.account_number,
                'timestamp': datetime.now(HOUR_ZONE).isoformat()
            })
        
        @self.app.route('/api/quality/thresholds')
        def get_quality_thresholds():
            return jsonify({
                'success': True,
                'thresholds': {
                    'slippage_warning_pct': SLIPPAGE_WARNING_PCT,
                    'slippage_critical_pct': SLIPPAGE_CRITICAL_PCT
                }
            })

        @self.app.route('/api/logs/stream')
        def stream_logs():
            try:
                if not os.path.exists(self.log_file):
                    return jsonify({'logs': []})
                
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    all_lines = f.readlines()
                
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                
                # Tomar últimas 200 líneas
                recent_lines = all_lines[-500:] if len(all_lines) > 500 else all_lines
                
                clean_logs = []
                for line in recent_lines:
                    line = ansi_escape.sub('', line).strip()
                    if line and ' - ' in line:
                        # Extraer solo el mensaje (después del último " - ")
                        message = line.split(' - ')[-1]
                        clean_logs.append(message)
                
                return jsonify({'logs': clean_logs})
            except Exception as e:
                return jsonify({'error': str(e), 'logs': []}), 500
            
        @self.app.route('/api/status')
        def get_status():
            try:
                state = self._load_state()
                
                total_positions = sum(len(positions) 
                                    for positions in state.get('positions', {}).values())
                
                try:
                    balance = self.get_balance(None)
                except Exception as e:
                    print(f"Error-getting balance: {e}")
                    balance = 0.0
                
                total_profit = 0
                num_trades = 0
                positive_trades = 0
                profit_pct = 0
                trades_pct = 0
                
                df = self._load_trades_dataframe()
                if df is not None:
                    total_profit = df['PROFIT'].sum()
                    num_trades = len(df)
                    positive_trades = len(df[df['PROFIT'] > 0])
                    
                    if num_trades > 0:
                        trades_pct = (positive_trades / num_trades) * 100
                    
                    if self.initial_capital > 0:
                        profit_pct = (total_profit / self.initial_capital) * 100
                
                open_pnl = 0
                for strat_id, positions in state.get('positions', {}).items():
                    for pos in positions:
                        try:
                            symbol = pos['symbol']
                            current_price = self.get_current_price(symbol)
                            entry_price = float(pos['entry_price'])
                            size = float(pos['size'])
                            direction = pos['direction'].lower()
                            
                            if direction == 'long':
                                pnl = (float(current_price) - entry_price) * size
                            else:
                                pnl = (entry_price - float(current_price)) * size
                            
                            open_pnl += pnl
                        except Exception as e:
                            print(f"No PnL - {pos.get('symbol')}: {e}")
                
                ref_price = 0
                try:
                    ref_price = float(self.get_current_price(self.reference_symbol))
                except:
                    pass
                
                return jsonify({
                    'status': 'running',
                    'account': self.account_number,
                    'total_positions': total_positions,
                    'strategies': state.get('positions', {}),
                    'candles': state.get('strategy_candles', {}),
                    'total_profit': float(total_profit),
                    'open_pnl': float(open_pnl),
                    'balance': float(balance),
                    'profit_pct': float(profit_pct),
                    'num_trades': num_trades,
                    'trades_pct': float(trades_pct),
                    'ref_price': float(ref_price),
                    'timestamp': datetime.now(HOUR_ZONE).isoformat(),
                    'ref_symbol': self.reference_symbol,
                    'unique_timeframes': self.unique_timeframes,
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/positions')
        def get_positions():
            try:
                state = self._load_state()
                
                positions_data = []
                for strategy_id, positions in state.get('positions', {}).items():
                    max_candles = 50
                    for strat in self.strategies:
                        if strat['id'] == strategy_id:
                            max_candles = strat.get('sell_after_ncandles', 50)
                            break
                    
                    for pos in positions:
                        try:
                            symbol        = pos['symbol']
                            current_price = self.get_current_price(symbol)
                            entry_price   = float(pos['entry_price'])
                            size          = float(pos['size'])
                            direction     = pos['direction'].lower()
                            tp_price      = float(pos['tp'])
                            sl_price      = float(pos['sl'])
                            
                            if direction == 'long':
                                pnl = (float(current_price) - entry_price) * size
                            else:
                                pnl = (entry_price - float(current_price)) * size
                            
                            candles = state.get('strategy_candles', {}).get(strategy_id, 0)
                            
                            precision = self.get_precision_for_price(current_price)
                            
                            current_price_rounded = round(float(current_price), precision)
                            tp_rounded = round(tp_price, precision)
                            sl_rounded = round(sl_price, precision)
                            entry_rounded = round(entry_price, precision)
                            
                            if direction == 'long':
                                distance_to_tp = ((tp_price - float(current_price)) / float(current_price)) * 100
                                distance_to_sl = ((float(current_price) - sl_price) / float(current_price)) * 100
                            else:
                                distance_to_tp = ((float(current_price) - tp_price) / float(current_price)) * 100
                                distance_to_sl = ((sl_price - float(current_price)) / float(current_price)) * 100
                            
                            positions_data.append({
                                'strategy': strategy_id,
                                'symbol': symbol,
                                'direction': pos['direction'],
                                'entry_price': entry_rounded,
                                'current_price': current_price_rounded,
                                'size': size,
                                'usdt_amount': pos.get('usdt_amount', 0),
                                'tp': tp_rounded,
                                'sl': sl_rounded,
                                'current_pnl': float(pnl),
                                'candles': candles,
                                'max_candles': max_candles,
                                'opened_at': pos['opened_at'],
                                'distance_to_tp_pct': round(distance_to_tp, 2),
                                'distance_to_sl_pct': round(distance_to_sl, 2),
                                'precision': precision
                            })
                        except Exception as e:
                            print(f"⚠️  Error processing position {pos.get('symbol')}: {e}")
                
                return jsonify(positions_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/bot/stop', methods=['POST'])
        def stop_bot():
            try:
                import subprocess
                import os
                import signal
                
                result = subprocess.run(
                    ['pgrep', '-f', f'main.py --account {self.account_number}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pid = int(result.stdout.strip())
                    
                    os.kill(pid, signal.SIGTERM)
                    
                    logger.info(f"Stop signal sent to PID {pid}")
                    
                    return jsonify({
                        'status': 'stopped',
                        'pid': pid,
                        'message': f'Bot process {pid} terminated'
                    })
                else:
                    return jsonify({
                        'status': 'not_found',
                        'message': 'Bot process not found'
                    }), 404
                    
            except subprocess.TimeoutExpired:
                logger.error("Error-pgrep command timed out")
                return jsonify({'error': 'Error-Command timeout'}), 500
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/bot/verify-stopped')
        def verify_stopped():
            try:
                import subprocess
                
                result = subprocess.run(
                    ['pgrep', '-f', f'main.py --account {self.account_number}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                running = result.returncode == 0 and result.stdout.strip()
                pid = int(result.stdout.strip()) if running else None
                
                return jsonify({'pid': pid, 'running': running})
                
            except Exception as e:
                logger.error(f"Error-verifying stop: {e}")
                return jsonify({'running': False, 'error': str(e)}), 200
                
        @self.app.route('/api/trades/recent')
        def get_recent_trades():
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df_sorted = df.sort_values('CLOSE_AT', ascending=False)
                
                # Take first 15 (most recent)
                recent = df_sorted.head(15).replace({np.nan: None}).to_dict('records')
                return jsonify(recent)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
        @self.app.route('/api/strategy-analysis')
        def get_strategy_analysis():
            try:
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    return jsonify([])
                
                results = []
                
                num_strategies = df['STRATEGY'].nunique()
                capital_per_strategy = self._calculate_capital_allocation(num_strategies)
                
                # Calculate profit per strategy first
                strategy_profits = {}
                for strategy in df['STRATEGY'].unique():
                    df_strategy = df[df['STRATEGY'] == strategy]
                    strategy_profits[strategy] = df_strategy['PROFIT'].sum()
                
                # Separate positive and negative STRATEGY profits
                total_profit_positive = sum(p for p in strategy_profits.values() if p > 0)
                total_profit_negative = sum(p for p in strategy_profits.values() if p < 0)
                
                for strategy in sorted(df['STRATEGY'].unique()):
                    df_strategy = df[df['STRATEGY'] == strategy]
                    
                    num_trades = len(df_strategy)
                    positive_trades = len(df_strategy[df_strategy['PROFIT'] > 0])
                    pct_positive = (positive_trades / num_trades * 100) if num_trades > 0 else 0
                    total_profit = strategy_profits[strategy]
                    profit_pct = (total_profit / capital_per_strategy * 100) if capital_per_strategy > 0 else 0
                    avg_duration = round(df_strategy['DURATION'].mean(), 2)
                    date_fo = df_strategy['OPEN_AT'].min()
                    
                    total_reasons = len(df_strategy)
                    tp_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('TP', na=False)])
                    sl_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('SL', na=False)])
                    oom_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('OUT_OF_MARGIN', na=False)])
                    timeout_count = len(df_strategy[df_strategy['REASON_OUT'] == 'TIMEOUT'])
                    
                    pct_tp = (tp_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_sl = (sl_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_oom = (oom_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_timeout = (timeout_count / total_reasons * 100) if total_reasons > 0 else 0
                    
                    # Calculate Total % (contribution to respective group)
                    if total_profit > 0:
                        # Winning strategy: % of total positive strategies profit
                        total_pct = (total_profit / total_profit_positive * 100) if total_profit_positive > 0 else 0
                    elif total_profit < 0:
                        # Losing strategy: % of total negative strategies profit
                        total_pct = (total_profit / abs(total_profit_negative) * 100) if total_profit_negative < 0 else 0
                    else:
                        # Break-even strategy
                        total_pct = 0.0
                    
                    results.append({
                        'Strategy': strategy,
                        'date_fo': date_fo.strftime('%Y-%m-%d'),
                        'Trades_num': num_trades,
                        'Trades_pct': round(pct_positive, 2),
                        'Total_profit': round(total_profit, 2),
                        'Profit_pct': round(profit_pct, 2),
                        'Total_pct': round(total_pct, 2),
                        'TP_pct': round(pct_tp, 2),
                        'SL_pct': round(pct_sl, 2),
                        'OOM_pct': round(pct_oom, 2),
                        'TIMEOUT_pct': round(pct_timeout, 2),
                        'Avg_days': avg_duration
                    })
                
                return jsonify(results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/bot-config')
        def get_bot_config():
            try:
                ws_status = {
                    'public_connected': False,
                    'private_connected': False,
                    'authenticated': False
                }
                
                try:
                    ws = get_ws_manager()
                    
                    if ws:
                        ws_status['public_connected'] = (
                            ws.public_ws and 
                            ws.public_ws.sock and 
                            ws.public_ws.sock.connected
                        )
                        ws_status['private_connected'] = (
                            ws.private_ws and 
                            ws.private_ws.sock and 
                            ws.private_ws.sock.connected
                        )
                        ws_status['authenticated'] = ws.authenticated
                except Exception as e:
                    logger.warning(f"WAR-Could not get WS status: {e}")
                
                timeframes_grouped = {}
                for strat in self.strategies:
                    tf = strat.get('timeframe', 'Unknown')
                    if tf not in timeframes_grouped:
                        timeframes_grouped[tf] = []
                    timeframes_grouped[tf].append(strat['id'])
                
                strategies_list = self._get_full_strategies_list_with_numbers()
                
                active_count = sum(1 for s in strategies_list if s['status'] == 'ACTIVE')
                deprecating_count = sum(1 for s in strategies_list if s['status'] == 'DEPRECATING')
                not_implemented_count = sum(1 for s in strategies_list if s['status'] == 'NOT IMPLEMENTED')
                
                return jsonify({
                    'account': self.account_number,
                    'initial_capital': self.initial_capital,
                    'websocket_status': ws_status,
                    'strategies': strategies_list,
                    'timeframes': timeframes_grouped,
                    'stats': {
                        'total': len(self.implemented_strategies),
                        'active': active_count,
                        'deprecating': deprecating_count,
                        'not_implemented': not_implemented_count
                    }
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

            
        @self.app.route('/api/weekly-analysis')
        def get_weekly_analysis():
            try:
                strategies_param = request.args.get('strategies', '')
                selected_strategies = [s.strip() for s in strategies_param.split(',') if s.strip()]

                if not selected_strategies:
                    return jsonify({'error': 'No strategies selected'}), 400

                date_from = request.args.get('date_from', '')
                date_to   = request.args.get('date_to', '')

                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])

                df = self._prepare_trades_dataframe(df)
                df = df[df['STRATEGY'].isin(selected_strategies)]

                if date_from:
                    df = df[df['CLOSE_AT'] >= pd.to_datetime(date_from)]
                if date_to:
                    df = df[df['CLOSE_AT'] <= pd.to_datetime(date_to) + pd.Timedelta(days=1)]

                if df.empty:
                    return jsonify([])

                strategies_with_trades = df['STRATEGY'].unique()
                num_strategies_with_trades = len(strategies_with_trades)
                capital_per_strat = self._calculate_capital_allocation(num_strategies_with_trades)
                capital_assigned  = capital_per_strat * num_strategies_with_trades

                # Group by ISO week (Monday–Sunday)
                df['week'] = df['CLOSE_AT'].dt.to_period('W')

                results = []

                for week in sorted(df['week'].unique()):
                    df_week = df[df['week'] == week]

                    num_trades     = len(df_week)
                    total_profit   = df_week['PROFIT'].sum()
                    profit_pct     = (total_profit / capital_assigned * 100) if capital_assigned > 0 else 0

                    positive_trades = len(df_week[df_week['PROFIT'] > 0])
                    win_rate        = (positive_trades / num_trades * 100) if num_trades > 0 else 0

                    # Actual date boundaries (handles partial first/last weeks)
                    actual_start = df_week['CLOSE_AT'].min()
                    actual_end   = df_week['CLOSE_AT'].max()

                    week_start = week.start_time  # Monday of that ISO week
                    week_end   = week.end_time    # Sunday of that ISO week
                    is_partial = (actual_start.date() > week_start.date()) or \
                                 (actual_end.date()   < week_end.date())

                    results.append({
                        'week':         str(week),
                        'week_label':   f"{actual_start.strftime('%d %b')} – {actual_end.strftime('%d %b %Y')}",
                        'num_trades':   num_trades,
                        'profit_usd':   round(float(total_profit), 2),
                        'profit_pct':   round(float(profit_pct), 2),
                        'win_rate':     round(float(win_rate), 1),
                        'is_partial':   is_partial
                    })

                return jsonify(results)

            except Exception as e:
                logger.error(f"Error in weekly analysis: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/monthly-analysis')
        def get_monthly_analysis():
            """
            Devuelve Profit % por mes para las estrategias seleccionadas.
            """
            try:
                strategies_param = request.args.get('strategies', '')
                selected_strategies = [s.strip() for s in strategies_param.split(',') if s.strip()]
                
                if not selected_strategies:
                    return jsonify({'error': 'No strategies selected'}), 400
                
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df = self._prepare_trades_dataframe(df)
                df = df[df['STRATEGY'].isin(selected_strategies)]
                
                if df.empty:
                    return jsonify([])
                
                # Calcular capital asignado - usar solo estrategias CON trades (igual que header)
                strategies_with_trades = df['STRATEGY'].unique()
                num_strategies_with_trades = len(strategies_with_trades)
                capital_per_strat = self._calculate_capital_allocation(num_strategies_with_trades)
                capital_assigned = capital_per_strat * num_strategies_with_trades
                
                # Agrupar por mes
                df['month'] = df['CLOSE_AT'].dt.to_period('M')
                
                results = []
                
                for month in sorted(df['month'].unique()):
                    df_month = df[df['month'] == month]
                    
                    num_trades = len(df_month)
                    total_profit = df_month['PROFIT'].sum()
                    profit_pct = (total_profit / capital_assigned * 100) if capital_assigned > 0 else 0
                    
                    positive_trades = len(df_month[df_month['PROFIT'] > 0])
                    win_rate = (positive_trades / num_trades * 100) if num_trades > 0 else 0
                    
                    results.append({
                        'month': str(month),
                        'month_name': month.strftime('%b %Y'),
                        'num_trades': num_trades,
                        'profit_usd': round(total_profit, 2),
                        'profit_pct': round(profit_pct, 2),
                        'win_rate': round(win_rate, 1)
                    })
                
                return jsonify(results)
                
            except Exception as e:
                print(f"Error in monthly analysis: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        

        
        @self.app.route('/api/equity-data')
        def get_equity_data():
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify({'error': 'Error-No trades file found'}), 404
                
                num_strategies_with_trades = len(df['STRATEGY'].unique())
                
                strategies_param = request.args.get('strategies', '')
                selected_strategies = [s.strip() for s in strategies_param.split(',') if s.strip()]
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                if not selected_strategies:
                    return jsonify({'error': 'Error-No strategies selected'}), 400
                
                df = df[df['STRATEGY'].isin(selected_strategies)]
                
                # Preparar y filtrar por fechas
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    num_selected = len(selected_strategies)
                    return jsonify({
                        'dates': [],
                        'equity_pct': [],
                        'drawdown_pct': [],
                        'capital_assigned': 0,
                        'num_selected': num_selected,
                        'total_strategies': num_strategies_with_trades,
                        'num_trades': 0,
                        'total_profit_usd': 0,
                        'profit_factor': 0,
                        'weekly_win_pct': 0,
                        'win_rate': 0,
                        'max_dd': 0,
                        'r_squared': 0,
                        'sharpe_ratio': 0,
                        'message': 'No trades found for selected strategies'
                    })
                
                df = df.sort_values('CLOSE_AT')
                df['date_str'] = df['CLOSE_AT'].dt.strftime('%Y-%m-%d')
                
                num_selected = len(selected_strategies)
                
                capital_per_strategy = self._calculate_capital_allocation(num_strategies_with_trades)
                capital_assigned = capital_per_strategy * num_selected
                
                # DESPUÉS:
                metrics_data = MetricsCalculator.calculate_all_metrics(
                    df=df,
                    capital_assigned=capital_assigned,
                    include_profit_pct=False
                )
                
                daily_profit = metrics_data['daily_profit']
                
                if not daily_profit.empty:
                    daily_profit['date_str'] = daily_profit['date'].astype(str)
                    
                    if capital_assigned > 0:
                        daily_profit['equity_pct'] = ((daily_profit['equity_usd'] / capital_assigned) - 1) * 100
                    else:
                        daily_profit['equity_pct'] = 0
                    
                    daily_profit['peak_usd'] = daily_profit['equity_usd'].cummax()
                    daily_profit['drawdown_pct'] = ((daily_profit['peak_usd'] - daily_profit['equity_usd']) / daily_profit['peak_usd']) * 100
                    
                    dates = daily_profit['date_str'].tolist()

                    equity_pct = [round(val, 2) for val in daily_profit['equity_pct'].tolist()]
                    drawdown_pct = [round(val, 2) for val in daily_profit['drawdown_pct'].tolist()]
                else:
                    dates = []
                    equity_pct = []
                    drawdown_pct = []
                
                return jsonify({
                    'dates': dates,
                    'equity_pct': equity_pct,
                    'drawdown_pct': drawdown_pct,
                    'capital_assigned': round(capital_assigned, 2),
                    'num_selected': num_selected,
                    'total_strategies': num_strategies_with_trades,
                    'num_trades': metrics_data['num_trades'],
                    'total_profit_usd': metrics_data['total_profit_usd'],
                    'profit_factor': metrics_data['profit_factor'],
                    'weekly_win_pct': metrics_data['weekly_win_pct'],
                    'win_rate': metrics_data['win_rate'],
                    'max_dd': metrics_data['max_dd'],
                    'r_squared': metrics_data['r_squared'],
                    'sharpe_ratio': metrics_data['sharpe_ratio']
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/symbols/unique')
        def get_unique_symbols():

            try:
                symbols_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'symbols_live', self.account_number)
                if not os.path.exists(symbols_dir):
                    return jsonify({'success': False, 'error': f'Directory not found: {symbols_dir}'}), 404

                symbol_set = set()
                for fname in os.listdir(symbols_dir):
                    if not fname.endswith('.csv'):
                        continue
                    fpath = os.path.join(symbols_dir, fname)
                    try:
                        df = pd.read_csv(fpath, header=None, names=['symbol'])
                        symbol_set.update(df['symbol'].dropna().str.strip().tolist())
                    except Exception as e:
                        logger.warning(f"[SYMBOLS] Error reading {fname}: {e}")

                return jsonify({
                    'success': True,
                    'symbols': sorted(symbol_set),
                    'count':   len(symbol_set)
                })

            except Exception as e:
                logger.error(f"[SYMBOLS] Error getting unique symbols: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # ==============================================================================
        # BTC DATA ENDPOINTS
        # ==============================================================================
        
        @self.app.route('/api/ref/history')
        def get_ref_history():

            try:
                date_from = request.args.get('date_from')
                date_to   = request.args.get('date_to')

                query  = """
                    SELECT date, price
                    FROM ref_history
                    WHERE symbol = %s
                """
                params = [self.reference_symbol]

                if date_from:
                    query += " AND date >= %s"
                    params.append(date_from)

                if date_to:
                    query += " AND date <= %s"
                    params.append(date_to)

                query += " ORDER BY date"

                conn   = psycopg2.connect(**self.postgres_config)
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows   = cursor.fetchall()
                cursor.close()
                conn.close()

                if not rows:
                    return jsonify({'success': True, 'dates': [], 'prices': []})

                dates  = [row[0].strftime('%Y-%m-%d') for row in rows]
                prices = [float(row[1]) if row[1] else None for row in rows]

                return jsonify({
                    'success': True,
                    'dates':   dates,
                    'prices':  prices,
                    'symbol':  self.reference_symbol
                })

            except Exception as e:
                logger.error(f"Error getting ref history: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
        @self.app.route('/api/risk/exposure')
        def get_risk_exposure():
            try:
                state = self._load_state()
        
                df = self._load_trades_dataframe()
                closed_pnl = df['PROFIT'].sum() if df is not None and not df.empty else 0
                available_capital = self.initial_capital + closed_pnl
        
                total_long_usdt = 0
                total_short_usdt = 0
        
                for strategy_id, positions in state.get('positions', {}).items():
                    for pos in positions:
                        real_exposure = float(pos.get('usdt_amount', 0)) / LEVERAGE
                        if pos['direction'].lower() == 'long':
                            total_long_usdt += real_exposure
                        else:
                            total_short_usdt += real_exposure
        
                cap = available_capital if available_capital > 0 else 1
        
                return jsonify({
                    'success': True,
                    'metrics': {
                        'gross_exposure_pct': round((total_long_usdt + total_short_usdt) / cap * 100, 2),
                        'net_exposure_pct':   round((total_long_usdt - total_short_usdt) / cap * 100, 2),
                        'long_exposure_pct':  round(total_long_usdt / cap * 100, 2),
                        'short_exposure_pct': round(total_short_usdt / cap * 100, 2),
                        'long_usdt':          round(total_long_usdt, 2),
                        'short_usdt':         round(total_short_usdt, 2),
                    },
                    'strategies': [],
                    'limits': {
                        'max_gross': RISK_LIMITS['max_gross_exposure_pct'],
                        'max_net':   RISK_LIMITS['max_net_exposure_pct']
                    }
                })
        
            except Exception as e:
                logger.error(f"Error getting risk exposure: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
                    
        @self.app.route('/api/quality/all')
        def get_quality_all():
            try:
                from quality_control.analyzer import analyze_execution_quality, analyze_target_deviation

                df = self._load_trades_dataframe()
                if df is None or df.empty:
                    return jsonify({
                        'success': False,
                        'error': 'No trades data available',
                        'data': {}
                    })

                strategies_list = self._get_full_strategies_list_with_numbers()
                active_strategies = [
                    s for s in strategies_list
                    if s['status'] in ('ACTIVE', 'DEPRECATING')
                ]

                return jsonify({
                    'success': True,
                    'data': {
                        'execution':  analyze_execution_quality(df, active_strategies),
                        'deviation':  analyze_target_deviation(df, active_strategies)
                    }
                })

            except Exception as e:
                logger.error(f"Error in quality analysis: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'data': {}
                }), 500
            
        @self.app.route('/api/quality/winrate-evolution')
        def get_winrate_evolution():

            try:
                # Get parameters
                strategies_param = request.args.get('strategies', '')
                selected_strategies = [s.strip() for s in strategies_param.split(',') if s.strip()]
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                if not selected_strategies:
                    return jsonify({
                        'success': False,
                        'error': 'No strategies selected'
                    }), 400
                
                # Load trades
                df = self._load_trades_dataframe()
                if df is None or df.empty:
                    return jsonify({
                        'success': True,
                        'dates': [],
                        'winrate': []
                    })
                
                # Filter by strategies
                df = df[df['STRATEGY'].isin(selected_strategies)].copy()
                
                if df.empty:
                    return jsonify({
                        'success': True,
                        'dates': [],
                        'winrate': []
                    })
                
                # Prepare and filter by dates
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    return jsonify({
                        'success': True,
                        'dates': [],
                        'winrate': []
                    })
                
                # Sort by close date
                df = df.sort_values('CLOSE_AT')
                
                # Group by date and calculate cumulative win rate
                df['date'] = df['CLOSE_AT'].dt.date
                df['is_winner'] = (df['PROFIT'] > 0).astype(int)
                
                # Calculate cumulative stats
                df['cumulative_wins'] = df['is_winner'].cumsum()
                df['cumulative_trades'] = range(1, len(df) + 1)
                df['cumulative_winrate'] = (df['cumulative_wins'] / df['cumulative_trades']) * 100
                
                # Get one value per day (last trade of the day)
                daily = df.groupby('date').last().reset_index()
                
                # Format output
                dates = [d.strftime('%Y-%m-%d') for d in daily['date']]
                winrate = [round(wr, 2) for wr in daily['cumulative_winrate']]
                
                return jsonify({
                    'success': True,
                    'dates': dates,
                    'winrate': winrate,
                    'total_trades': int(len(df))
                })
                
            except Exception as e:
                logger.error(f"Error in winrate evolution: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/ref/snapshot')
        def capture_ref_snapshot():

            try:
                from datetime import date

                today  = date.today()
                symbol = self.reference_symbol

                conn   = psycopg2.connect(**self.postgres_config)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT price FROM ref_history
                    WHERE date = %s AND symbol = %s
                """, [today, symbol])

                existing = cursor.fetchone()

                if existing:
                    cursor.close()
                    conn.close()
                    return jsonify({
                        'success': True,
                        'message': 'Snapshot already exists for today',
                        'price':   float(existing[0]),
                        'symbol':  symbol,
                        'date':    today.isoformat()
                    })

                try:
                    ref_price = float(self.get_current_price(symbol))
                except Exception as e:
                    cursor.close()
                    conn.close()
                    logger.error(f"[REF SNAPSHOT] Error getting {symbol} price: {e}")
                    return jsonify({
                        'success': False,
                        'error':   f'Failed to get {symbol} price: {str(e)}'
                    }), 500

                cursor.execute("""
                    INSERT INTO ref_history (date, symbol, price)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (date, symbol) DO NOTHING
                """, [today, symbol, ref_price])

                conn.commit()
                cursor.close()
                conn.close()

                logger.info(f"[REF SNAPSHOT] Captured: {today} {symbol} -> ${ref_price:.2f}")

                return jsonify({
                    'success': True,
                    'message': 'Ref snapshot captured successfully',
                    'price':   ref_price,
                    'symbol':  symbol,
                    'date':    today.isoformat()
                })

            except Exception as e:
                logger.error(f"[REF SNAPSHOT] Error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
    
    # ==========================================================================
    # DAILY SNAPSHOT SCHEDULER METHODS (CLASS LEVEL)
    # ==========================================================================
    
    def _capture_snapshot(self):
        """
        Capture daily exposure snapshot by calling internal endpoint.
        Triggered by scheduler at 23:55 UTC daily.
        """
        try:
            if not self.dashboard_port:
                logger.warning("[SNAPSHOT] Port not set, skipping")
                return
            
            url = f'http://localhost:{self.dashboard_port}/api/risk/exposure-history?days=1'
            response = requests.get(url, timeout=10)
            
            if response.ok:
                data = response.json()
                if data.get('success'):
                    logger.info(f"[SNAPSHOT] Daily exposure captured for {self.account_number}")
                else:
                    logger.warning(f"[SNAPSHOT] Endpoint returned error: {data.get('error')}")
            else:
                logger.warning(f"[SNAPSHOT] HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            logger.error("[SNAPSHOT] Request timeout (>10s)")
        except Exception as e:
            logger.error(f"[SNAPSHOT] Error capturing snapshot: {e}")
    
    def _capture_ref_snapshot(self):
        """
        Capture daily reference symbol price snapshot by calling internal endpoint.
        Triggered by scheduler at 00:05 UTC daily.
        """
        try:
            if not self.dashboard_port:
                logger.warning("[REF SNAPSHOT] Port not set, skipping")
                return

            url      = f'http://localhost:{self.dashboard_port}/api/ref/snapshot'
            response = requests.get(url, timeout=10)

            if response.ok:
                data = response.json()
                if data.get('success'):
                    logger.info(f"[REF SNAPSHOT] Daily {self.reference_symbol} price captured: ${data.get('price')}")
                else:
                    logger.warning(f"[REF SNAPSHOT] Endpoint returned error: {data.get('error')}")
            else:
                logger.warning(f"[REF SNAPSHOT] HTTP {response.status_code}: {response.text[:100]}")

        except requests.exceptions.Timeout:
            logger.error("[REF SNAPSHOT] Request timeout (>10s)")
        except Exception as e:
            logger.error(f"[REF SNAPSHOT] Error capturing snapshot: {e}")

    def _schedule_daily_snapshot(self):
        """
        Scheduler loop that runs in separate thread.
        Captures reference symbol price snapshot daily at 00:05 UTC.
        """
        if self.reference_symbol:
            logger.info(f"[SNAPSHOT] Scheduler started - captures {self.reference_symbol} price daily at 00:05 UTC")
        else:
            logger.info("[SNAPSHOT] Scheduler started - no reference symbol configured, nothing to capture")
    
        triggered_today = False
    
        while self.snapshot_running:
            now_utc = datetime.utcnow()
            if now_utc.hour == 0 and now_utc.minute == 5:
                if not triggered_today:
                    if self.reference_symbol:
                        self._capture_ref_snapshot()
                    triggered_today = True
            else:
                triggered_today = False
            time_module.sleep(30)
    
        logger.info("[SNAPSHOT] Scheduler stopped")
    
    def _start_snapshot_scheduler(self):
        """Start snapshot scheduler thread"""
        if self.snapshot_thread and self.snapshot_thread.is_alive():
            logger.warning("[SNAPSHOT] Scheduler already running")
            return
        
        self.snapshot_running = True
        self.snapshot_thread = threading.Thread(
            target=self._schedule_daily_snapshot,
            daemon=True,
            name=f'SnapshotScheduler-{self.account_number}'
        )
        self.snapshot_thread.start()
    
    def _stop_snapshot_scheduler(self):
        """Stop snapshot scheduler thread"""
        if self.snapshot_running:
            self.snapshot_running = False
            if self.snapshot_thread:
                self.snapshot_thread.join(timeout=2)
    
    def start(self, host='0.0.0.0', port=5000):
        """Inicia el servidor del dashboard"""
        if self.running:
            print("⚠️  Dashboard already running")
            return
        
        # Store port for snapshot scheduler
        self.dashboard_port = port
        
        def run_server():
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            
            try:
                self.app.run(
                    host=host, 
                    port=port, 
                    debug=False, 
                    use_reloader=False, 
                    threaded=True
                )
            except Exception as e:
                logger.error(f"Error-Dashboard server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        
        import time
        time.sleep(0.5)
        
        logger.info(f"Dashboard Web Started")
        logger.info(f"Local:   http://localhost:{port}")
        logger.info(f"Network: http://127.0.0.1:{port}")
        logger.info(f"LAN:     http://<your-ip>:{port}")
        
        # Start snapshot scheduler AFTER Flask is running
        self._start_snapshot_scheduler()
    
    def stop(self):
        """Detiene el servidor"""
        self._stop_snapshot_scheduler()  # Stop scheduler first
        self.running = False
        logger.info("Dashboard server stopped")

def create_dashboard_template(base_dir):
    """Crea el archivo HTML del dashboard si no existe en ruta común"""
    api_dir = os.path.join(os.path.dirname(__file__))
    templates_dir = os.path.join(api_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    template_path = os.path.join(templates_dir, 'dashboard.html')
    
    if os.path.exists(template_path):
        return
    
    logger.warning(f"WAR-Creating template at {template_path}")
    logger.warning(f"WAR-Please ensure the complete dashboard.html is in place")

if __name__ == '__main__':
    logger.info("This module should be imported, not run directly")
    logger.info("Usage:")
    logger.info("from ZX_BOT_backend import DashboardServer")
    logger.info("dashboard = DashboardServer(...)")
    logger.info("dashboard.start()")