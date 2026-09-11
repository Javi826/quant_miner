"""
Quality Control module - Execution quality analysis.
"""
from .analyzer import analyze_execution_quality, analyze_target_deviation
__all__ = [
    'analyze_execution_quality',
    'analyze_target_deviation',
]