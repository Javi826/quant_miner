#BOT_crypto/strategies/strategy_loader.py (crypto)

import logging
from typing import List, Dict

logger = logging.getLogger('BOT_crypto.strategies.strategy_loader')


def load_strategies(account_number: str) -> List[Dict]:
    # Importar módulo según cuenta
    if account_number == '00':
        from config.strategies_00 import STRATEGIES
    elif account_number == 'E1':
        from config.strategies_E1 import STRATEGIES
    elif account_number == '01':
        from config.strategies_01 import STRATEGIES
    else:
        raise ValueError(f"Unknown account: {account_number}")
    
    logger.info(f"Loaded {len(STRATEGIES)} strategies for account {account_number}")
    return STRATEGIES


def apply_set_active_argument(
    strategies: List[Dict],
    active_ids: List[str]
) -> None:

    # Verify all requested IDs exist
    available_ids = {s['id'] for s in strategies}
    missing_ids = set(active_ids) - available_ids
    
    if missing_ids:
        raise ValueError(
            f"Strategy IDs not found: {', '.join(sorted(missing_ids))}. "
            f"Available IDs: {', '.join(sorted(available_ids))}"
        )
    
    # Set active flags
    for strat in strategies:
        if strat['id'] in active_ids:
            strat['active'] = True
        else:
            strat['active'] = False
    
    active_count = sum(1 for s in strategies if s.get('active', True))
    logger.info(
        f"Applied --set-active: {active_count}/{len(strategies)} strategies active"
    )

def get_all_strategy_ids(account_number: str) -> List[str]:

    strategies = load_strategies(account_number)
    return [s['id'] for s in strategies]


def get_strategy_config(account_number: str, strategy_id: str) -> Dict:

    strategies = load_strategies(account_number)
    
    for strat in strategies:
        if strat['id'] == strategy_id:
            return strat
    
    available_ids = [s['id'] for s in strategies]
    raise ValueError(
        f"Strategy ID '{strategy_id}' not found in account {account_number}. "
        f"Available IDs: {', '.join(available_ids)}"
    )

