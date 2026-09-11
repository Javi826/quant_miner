#BOT_crypto/quality_control/analyzer.py
import pandas as pd
from typing import Dict, List, Any
import logging
from config.settings import EXECUTION_WINDOW_SIZE
from config.settings import SLIPPAGE_WARNING_PCT, SLIPPAGE_CRITICAL_PCT, LATENCY_WARNING_SEC, LATENCY_CRITICAL_SEC

logger = logging.getLogger('BOT_crypto.quality_control.analyzer')

def analyze_execution_quality(df_trades: pd.DataFrame, strategies_config: List[Dict]) -> Dict[str, Any]:

    results = {}
    
    for strat_config in strategies_config:
        strategy_id = strat_config['id']
        
        # Get strategy trades
        df_strat = df_trades[df_trades['STRATEGY'] == strategy_id].copy()
        
        if len(df_strat) == 0:
            results[strategy_id] = {
                'total_trades': 0,
                'avg_close_slippage_pct': None,
                'close_slippage_status': 'NO_DATA',
                'avg_tp_slippage_pct': None,
                'tp_slippage_status': 'NO_DATA',
                'avg_sl_slippage_pct': None,
                'sl_slippage_status': 'NO_DATA',
                'avg_latency_sec': None,
                'latency_status': 'NO_DATA'
            }
            continue
        
        total_trades = len(df_strat)
        
        # Get last EXECUTION_WINDOW_SIZE trades
        df_last = df_strat.tail(EXECUTION_WINDOW_SIZE)
        
        # =====================================================================
        # CLOSE SLIPPAGE (execution slippage)
        # =====================================================================
        avg_close_slippage_pct = None
        close_slippage_status = 'NO_DATA'
        
        if 'order_price_close' in df_last.columns and 'PRICE_CLOSE' in df_last.columns:
            df_with_slippage = df_last[
                df_last['order_price_close'].notna() & 
                df_last['PRICE_CLOSE'].notna()
            ].copy()
            
            if len(df_with_slippage) > 0:
                df_with_slippage['close_slippage_pct'] = (
                    (df_with_slippage['PRICE_CLOSE'] - df_with_slippage['order_price_close']) 
                    / df_with_slippage['order_price_close'] 
                    * 100
                )
                avg_close_slippage_pct = df_with_slippage['close_slippage_pct'].mean()
                
                # Determine status
                abs_slippage = abs(avg_close_slippage_pct)
                if abs_slippage < SLIPPAGE_WARNING_PCT:
                    close_slippage_status = 'HEALTHY'
                elif abs_slippage < SLIPPAGE_CRITICAL_PCT:
                    close_slippage_status = 'WARNING'
                else:
                    close_slippage_status = 'DANGER'
        
        # =====================================================================
        # TP SLIPPAGE (target vs actual)
        # =====================================================================
        avg_tp_slippage_pct = None
        tp_slippage_status = 'NO_DATA'
        
        if 'TP_TARGET' in df_last.columns and 'PRICE_CLOSE' in df_last.columns and 'REASON_OUT' in df_last.columns:
            df_tp = df_last[
                (df_last['REASON_OUT'] == 'TP') &
                df_last['TP_TARGET'].notna() & 
                df_last['PRICE_CLOSE'].notna()
            ].copy()
            
            if len(df_tp) > 0:
                df_tp['tp_slippage_pct'] = (
                    (df_tp['PRICE_CLOSE'] - df_tp['TP_TARGET']) 
                    / df_tp['TP_TARGET'] 
                    * 100
                )
                avg_tp_slippage_pct = df_tp['tp_slippage_pct'].mean()
                
                # Determine status (same thresholds as close slippage)
                abs_slippage = abs(avg_tp_slippage_pct)
                if abs_slippage < SLIPPAGE_WARNING_PCT:
                    tp_slippage_status = 'HEALTHY'
                elif abs_slippage < SLIPPAGE_CRITICAL_PCT:
                    tp_slippage_status = 'WARNING'
                else:
                    tp_slippage_status = 'DANGER'
        
        # =====================================================================
        # SL SLIPPAGE (target vs actual)
        # =====================================================================
        avg_sl_slippage_pct = None
        sl_slippage_status = 'NO_DATA'
        
        if 'SL_TARGET' in df_last.columns and 'PRICE_CLOSE' in df_last.columns and 'REASON_OUT' in df_last.columns:
            df_sl = df_last[
                (df_last['REASON_OUT'] == 'SL') &
                df_last['SL_TARGET'].notna() & 
                df_last['PRICE_CLOSE'].notna()
            ].copy()
            
            if len(df_sl) > 0:
                df_sl['sl_slippage_pct'] = (
                    (df_sl['PRICE_CLOSE'] - df_sl['SL_TARGET']) 
                    / df_sl['SL_TARGET'] 
                    * 100
                )
                avg_sl_slippage_pct = df_sl['sl_slippage_pct'].mean()
                
                # Determine status (same thresholds as close slippage)
                abs_slippage = abs(avg_sl_slippage_pct)
                if abs_slippage < SLIPPAGE_WARNING_PCT:
                    sl_slippage_status = 'HEALTHY'
                elif abs_slippage < SLIPPAGE_CRITICAL_PCT:
                    sl_slippage_status = 'WARNING'
                else:
                    sl_slippage_status = 'DANGER'
        
        # =====================================================================
        # LATENCY
        # =====================================================================
        avg_latency_sec = None
        latency_status = 'NO_DATA'
        
        if 'exec_ts_close' in df_last.columns and 'order_ts_close' in df_last.columns:
            df_with_latency = df_last[
                df_last['exec_ts_close'].notna() & 
                df_last['order_ts_close'].notna()
            ].copy()
            
            if len(df_with_latency) > 0:
                # Latency in seconds (timestamps already in seconds with decimals)
                df_with_latency['latency_sec'] = (
                    df_with_latency['exec_ts_close'] - df_with_latency['order_ts_close']
                )
                avg_latency_sec = df_with_latency['latency_sec'].mean()
                
                # Determine status
                if avg_latency_sec < LATENCY_WARNING_SEC:
                    latency_status = 'HEALTHY'
                elif avg_latency_sec < LATENCY_CRITICAL_SEC:
                    latency_status = 'WARNING'
                else:
                    latency_status = 'DANGER'
        
        results[strategy_id] = {
            'total_trades': int(total_trades),
            'avg_close_slippage_pct': round(avg_close_slippage_pct, 4) if avg_close_slippage_pct is not None else None,
            'close_slippage_status': close_slippage_status,
            'avg_tp_slippage_pct': round(avg_tp_slippage_pct, 4) if avg_tp_slippage_pct is not None else None,
            'tp_slippage_status': tp_slippage_status,
            'avg_sl_slippage_pct': round(avg_sl_slippage_pct, 4) if avg_sl_slippage_pct is not None else None,
            'sl_slippage_status': sl_slippage_status,
            'avg_latency_sec': round(avg_latency_sec, 3) if avg_latency_sec is not None else None,
            'latency_status': latency_status
        }
    
    return results

def analyze_target_deviation(df_trades: pd.DataFrame, strategies_config: List[Dict]) -> Dict[str, Any]:

    results = {}
    
    for strat_config in strategies_config:
        strategy_id = strat_config['id']
        
        # Get configured TP/SL from strategy config
        tp_target_pct = strat_config.get('tp_pct')
        sl_target_pct_raw = strat_config.get('sl_pct')
        
        # Convert SL to negative (config stores positive values like 10.0)
        sl_target_pct = -abs(sl_target_pct_raw) if sl_target_pct_raw is not None else None
        
        # Get strategy trades
        df_strat = df_trades[df_trades['STRATEGY'] == strategy_id].copy()
        
        if len(df_strat) == 0:
            results[strategy_id] = {
                'tp_trades': 0,
                'tp_real_pct': None,
                'tp_target_pct': tp_target_pct,
                'tp_deviation': None,
                'sl_trades': 0,
                'sl_real_pct': None,
                'sl_target_pct': sl_target_pct,
                'timeout_trades': 0
            }
            continue
        
        # TP analysis
        df_tp = df_strat[df_strat['REASON_OUT'] == 'TP']
        tp_trades = len(df_tp)
        tp_real_pct = None
        tp_deviation = None
        
        if tp_trades > 0 and 'PROFIT_PCT' in df_tp.columns:
            tp_real_pct = df_tp['PROFIT_PCT'].mean()
            if tp_target_pct is not None:
                tp_deviation = tp_real_pct - tp_target_pct
        
        # SL analysis
        df_sl = df_strat[df_strat['REASON_OUT'] == 'SL']
        sl_trades = len(df_sl)
        sl_real_pct = None
        sl_deviation = None
        
        if sl_trades > 0 and 'PROFIT_PCT' in df_sl.columns:
            sl_real_pct = df_sl['PROFIT_PCT'].mean()
            if sl_target_pct is not None:
                sl_deviation = sl_real_pct - sl_target_pct
        
        timeout_trades = len(df_strat[df_strat['REASON_OUT'] == 'TIMEOUT'])
        
        results[strategy_id] = {
            'tp_trades': int(tp_trades),
            'tp_real_pct': round(tp_real_pct, 2) if tp_real_pct is not None else None,
            'tp_target_pct': round(tp_target_pct, 2) if tp_target_pct is not None else None,
            'tp_deviation': round(tp_deviation, 2) if tp_deviation is not None else None,
            'sl_trades': int(sl_trades),
            'sl_real_pct': round(sl_real_pct, 2) if sl_real_pct is not None else None,
            'sl_target_pct': round(sl_target_pct, 2) if sl_target_pct is not None else None,
            'sl_deviation': round(sl_deviation, 2) if sl_deviation is not None else None,
            'timeout_trades': int(timeout_trades)
        }
    
    return results
