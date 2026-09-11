#BOT_crypto/main.py
"""
main.py Trading Bot - Main Entry Point

Clean entry point that instantiates and runs the BotOrchestrator.
"""

import os
import sys
import argparse
import logging

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution import BitgetClient
from bot_utils.logger import setup_logger
from core import BotOrchestrator

from config.utils.utils import get_account_config
from config.utils.connect_pass import BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00
from config.utils.connect_pass import BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01
from config.utils.connect_pass import BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1
from config.utils.connect_pass import connect_bitget_00, connect_bitget_01, connect_bitget_E1
from core.demo_operative import DemoOperative
from core.production_operative import ProductionOperative



def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-strategy trading bot with WebSocket support')
    parser.add_argument(
        '--account',
        type=str,
        default='E1',
        choices=['00', 'E1', '01'],
        help='Account number (00, E1, 01)'
    )
    parser.add_argument(
        '--set-active',
        type=str,
        default=None,
        help='Comma-separated list of strategy IDs to set as active'
    )
    args = parser.parse_args()
    
    account_number = args.account
    
    # Map credentials
    BITGET_CLIENTS = {
        "00": BitgetClient(BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00),
        "01": BitgetClient(BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01),
        "E1": BitgetClient(BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1)
    }
    
    CCXT_CONNECTIONS = {
        "00": connect_bitget_00,
        "01": connect_bitget_01,
        "E1": connect_bitget_E1
    }
    
    # Get account config for logger setup
    account_config = get_account_config(account_number)
    base_dir       = account_config['paths']['base_dir']
    log_file       = account_config['paths']['log_file']
    
    # Create directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Setup logger
    setup_logger(base_dir, logfile_name=os.path.basename(log_file))
    logger = logging.getLogger('BOT_crypto')
        
    # Parse --set-active argument
    active_strategy_ids = None
    if args.set_active:
        active_strategy_ids = [s.strip() for s in args.set_active.split(',')]

    # Create bot
    bot = BotOrchestrator(
        account_number=account_number,
        bitget_client=BITGET_CLIENTS[account_number],
        connect_bitget_func=CCXT_CONNECTIONS[account_number],
        active_strategy_ids=active_strategy_ids
    )

    # Assign operative mode — single decision point
    if account_config.get('type') == 'demo':
        bot.operative = DemoOperative(
            account_number=account_number,
            ws_manager=None,
            excel_path=account_config['paths']['trades_file'],
            strategy_configs=[]
        )
    else:
        bot.operative = ProductionOperative(
            account_number=account_number,
            state_file=account_config['paths']['state_file'],
            send_request_func=bot._send_request_wrapper,
            bot_state=None
        )

    bot.run()

if __name__ == '__main__':
    main()