"""
API module - Dashboard and web interface.
"""

from .backend import DashboardServer, create_dashboard_template

__all__ = [
    'DashboardServer',
    'create_dashboard_template',
]
