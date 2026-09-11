"""
risk_control/exposure_calculator.py 

Exposure Calculator - Calculates current exposure metrics from open positions.
Used by both backend API and orchestrator risk limiter.
"""

import logging
from typing import Dict, List


class ExposureCalculator:
    """
    Calculator for exposure metrics with integrated logging.
    
    Calculates gross/net/long/short exposure percentages
    based on open positions, capital, and leverage.
    
    Standardized with PositionSizer pattern (instance-based with logger).
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize exposure calculator.
        
        Args:
            logger: Logger instance for exposure calculations
        """
        self.logger = logger
    
    def calculate_current_exposure(
        self,
        open_positions: Dict[str, List[Dict]],
        closed_pnl: float,
        initial_capital: float,
        leverage: int = 10
    ) -> Dict:
        """
        Calculate current exposure metrics with logging.
        
        Args:
            open_positions: Dictionary of positions by strategy
            closed_pnl: Total profit/loss from closed trades
            initial_capital: Initial account capital
            leverage: Leverage multiplier (default: 10)
        
        Returns:
            Dictionary with:
            - gross_exposure_pct: (long + short) / capital * 100
            - net_exposure_pct: (long - short) / capital * 100
            - long_exposure_pct: long / capital * 100
            - short_exposure_pct: short / capital * 100
            - available_capital: initial + closed_pnl
            - num_positions: total number of open positions
            - total_long_usdt: total long exposure in USDT
            - total_short_usdt: total short exposure in USDT
        
        Example:
            >>> calculator = ExposureCalculator(logger)
            >>> metrics = calculator.calculate_current_exposure(
            ...     open_positions={'strat_01': [...]},
            ...     closed_pnl=200,
            ...     initial_capital=1000,
            ...     leverage=10
            ... )
            >>> print(metrics['gross_exposure_pct'])
            15.5
        """
        total_long_usdt = 0.0
        total_short_usdt = 0.0
        num_positions = 0
        
        # Sum exposure from all positions (adjusted by leverage)
        for strategy_id, positions in open_positions.items():
            for pos in positions:
                usdt_amount = float(pos.get('usdt_amount', 0))
                real_exposure = usdt_amount / leverage
                
                if pos['direction'].lower() == 'long':
                    total_long_usdt += real_exposure
                else:
                    total_short_usdt += real_exposure
                
                num_positions += 1
        
        # Calculate available capital
        available_capital = initial_capital + closed_pnl
        
        # Calculate percentages
        if available_capital > 0:
            gross_pct = ((total_long_usdt + total_short_usdt) / available_capital) * 100
            net_pct = ((total_long_usdt - total_short_usdt) / available_capital) * 100
            long_pct = (total_long_usdt / available_capital) * 100
            short_pct = (total_short_usdt / available_capital) * 100
        else:
            gross_pct = net_pct = long_pct = short_pct = 0.0
        
        # Log calculated metrics
        self.logger.debug(
            f"[RISK] Exposure calculated: "
            f"gross={gross_pct:.1f}%, net={net_pct:.1f}%, "
            f"long={long_pct:.1f}%, short={short_pct:.1f}% | "
            f"Positions: {num_positions}"
        )
        
        return {
            'gross_exposure_pct': round(gross_pct, 2),
            'net_exposure_pct': round(net_pct, 2),
            'long_exposure_pct': round(long_pct, 2),
            'short_exposure_pct': round(short_pct, 2),
            'available_capital': round(available_capital, 2),
            'num_positions': num_positions,
            'total_long_usdt': round(total_long_usdt, 2),
            'total_short_usdt': round(total_short_usdt, 2)
        }