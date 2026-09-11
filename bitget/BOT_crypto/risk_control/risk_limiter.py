"""
risk_control/risk_limiter.py

Risk Limiter - Checks risk limits before opening new positions.
Prevents exceeding configured exposure thresholds.
"""

import logging
from typing import Tuple

from config.settings import RISK_LIMITS
from typing import Dict, Optional, Tuple



class RiskLimiter:
    """
    Risk limiter for position opening decisions with integrated logging.
    
    Performs simple binary check: is current exposure at/above limit?
    Used by orchestrator to skip strategy signal search when at capacity.
    
    Standardized with PositionSizer pattern (instance-based with logger).
    """
    
    def __init__(self, initial_capital: float, logger: logging.Logger):
        """
        Initialize risk limiter.
        
        Args:
            initial_capital: Initial account capital
            logger: Logger instance for risk decisions
        """
        self.initial_capital = initial_capital
        self.logger = logger
        self.max_gross_pct = RISK_LIMITS['max_gross_exposure_pct']
        self.max_net_pct = RISK_LIMITS['max_net_exposure_pct']
    
    def is_at_limit(self, current_gross_pct: float) -> Tuple[bool, str, Dict]:
        """
        Check if already at/above exposure limit.
        
        Args:
            current_gross_pct: Current gross exposure percentage
        
        Returns:
            Tuple of (blocked, reason, metadata)
            
            metadata contains:
                - current_gross_pct: Current exposure
                - max_gross_pct: Configured limit
                - blocked: Whether at/above limit
        
        Example:
            >>> limiter = RiskLimiter(initial_capital=1000, logger=logger)
            >>> blocked, reason, meta = limiter.is_at_limit(32.5)
            >>> # (True, "At limit", {'current_gross_pct': 32.5, ...})
        """
        blocked = current_gross_pct >= self.max_gross_pct
        
        metadata = {
            'current_gross_pct': current_gross_pct,
            'max_gross_pct': self.max_gross_pct,
            'blocked': blocked
        }
        
        if blocked:
            reason = "At limit"
        else:
            reason = "Below limit"
        
        return (blocked, reason, metadata)
    
    def format_log_message(self, strategy_id: str, metadata: Dict) -> str:
        """
        Format standardized log message for risk decision.
        
        Args:
            strategy_id: Strategy identifier
            metadata: Metadata dict from is_at_limit()
        
        Returns:
            Formatted log string
        
        Example:
            "[RISK] 06_reversal_long_1H: 28.5% < 60.0% ✓"
            "[RISK] 06_reversal_long_1H: 62.3% >= 60.0% BLOCKED"
        """
        status = "BLOCKED" if metadata['blocked'] else "✓"
        operator = ">=" if metadata['blocked'] else "<"
        
        return (
            f"[RISK] {strategy_id}: "
            f"{metadata['current_gross_pct']:.1f}% {operator} "
            f"{metadata['max_gross_pct']:.1f}% {status}"
        )