# BOT_crypto/validation/validation_module.py
"""
Validation functions for bot configuration
"""
import re
import logging

logger = logging.getLogger('BOT_crypto.validation.validation_module')

from config.settings import MIN_ORDER_AMOUNT, MAX_ORDER_AMOUNT, MIN_TP_PCT, MAX_TP_PCT
from config.settings import MIN_SL_PCT, MAX_SL_PCT, MIN_CANDLES, MAX_CANDLES
from config.settings import VALID_TIMEFRAMES
from bot_utils.timeframes import timeframe_to_timedelta
from config.settings import COMMON_REQUIRED_PARAMS
from config.settings import ACCOUNTS, BASE_URL
from config.settings import POSTGRES_CONFIG
import psycopg2
from psycopg2 import OperationalError

# ==========================================================================
# POSTGRESQL VALIDATION
# ==========================================================================

def validate_postgresql_connection():

    try:
        logger.debug("Validating PostgreSQL connection...")
        
        # Try to connect
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.close()
        
        logger.info("PostgreSQL connection validated successfully ✓")
        return True
        
    except OperationalError as e:
        logger.error("Error: PostgreSQL connection failed")
        logger.error(f"Error: {e}")
        logger.error("Ensure PostgreSQL is running: sudo systemctl status postgresql")
        raise SystemExit(1)
        
    except Exception as e:
        logger.error(f"Error: Unexpected error validating PostgreSQL: {e}")
        raise SystemExit(1)
        
# ==========================================================================
# SETTINGS VALIDATION
# ==========================================================================

def validate_settings():

    
    errors = []
    warnings = []
    
    # ========================================================================
    # Val S1: Unique dashboard ports
    # ========================================================================
    validation_s1_errors = 0
    ports = [acc['dashboard_port'] for acc in ACCOUNTS.values()]
    if len(ports) != len(set(ports)):
        errors.append("Dashboard ports must be unique across accounts")
        validation_s1_errors += 1
    
    if validation_s1_errors == 0:
        logger.debug("Val S1: Dashboard ports are unique")
    
    # ========================================================================
    # Val S2: Valid timeframes
    # ========================================================================
    validation_s2_errors = 0
    if not VALID_TIMEFRAMES:
        errors.append("VALID_TIMEFRAMES cannot be empty")
        validation_s2_errors += 1
    for timeframe in VALID_TIMEFRAMES:
        try:
            timeframe_to_timedelta(timeframe)
        except ValueError as e:
            errors.append(f"VALID_TIMEFRAMES contains an unparseable timeframe: {e}")
            validation_s2_errors += 1
    
    if validation_s2_errors == 0:
        logger.debug("Val S2: VALID_TIMEFRAMES configured")
    
    # ========================================================================
    # Val S3: Order amount limits
    # ========================================================================
    validation_s3_errors = 0
    if MIN_ORDER_AMOUNT >= MAX_ORDER_AMOUNT:
        errors.append("MIN_ORDER_AMOUNT must be less than MAX_ORDER_AMOUNT")
        validation_s3_errors += 1
    
    if validation_s3_errors == 0:
        logger.debug("Val S3: Order amount limits valid")
    
    # ========================================================================
    # Val S4: TP percentage limits
    # ========================================================================
    validation_s4_errors = 0
    if MIN_TP_PCT >= MAX_TP_PCT:
        errors.append("MIN_TP_PCT must be less than MAX_TP_PCT")
        validation_s4_errors += 1
    
    if validation_s4_errors == 0:
        logger.debug("Val S4: TP percentage limits valid")
    
    # ========================================================================
    # Val S5: BASE_URL uses HTTPS
    # ========================================================================
    validation_s5_errors = 0
    if not BASE_URL.startswith("https://"):
        errors.append("BASE_URL must use HTTPS")
        validation_s5_errors += 1
    
    if validation_s5_errors == 0:
        logger.debug("Val S5: BASE_URL uses HTTPS")
    
    
    # ========================================================================
    # Val S7: SL percentage limits
    # ========================================================================
    validation_s7_errors = 0
    if MIN_SL_PCT >= MAX_SL_PCT:
        errors.append("MIN_SL_PCT must be less than MAX_SL_PCT")
        validation_s7_errors += 1
    
    if validation_s7_errors == 0:
        logger.debug("Val S7: SL percentage limits valid")
    
    # ========================================================================
    # Val S8: Candles limits
    # ========================================================================
    validation_s8_errors = 0
    if MIN_CANDLES >= MAX_CANDLES:
        errors.append("MIN_CANDLES must be less than MAX_CANDLES")
        validation_s8_errors += 1
    
    if validation_s8_errors == 0:
        logger.debug("Val S8: Candles limits valid")
    
    
    # ========================================================================
    # Val S17: Account numbers format
    # ========================================================================
    validation_s17_errors = 0
    
    for account_num in ACCOUNTS.keys():
        # Check format: 2 chars alphanumeric
        if not re.match(r'^[A-Z0-9]{2}$', account_num):
            errors.append(
                f"Account number '{account_num}' has invalid format "
                f"(must be 2 alphanumeric characters, e.g., '00', 'E1')"
            )
            validation_s17_errors += 1
    
    if validation_s17_errors == 0:
        logger.debug("Val S17: All account numbers have valid format")
    
    return errors, warnings


# ==========================================================================
# STRATEGY CONFIGURATION VALIDATION
# ==========================================================================

def validate_strategy_configuration(strategies, implemented_strategies):

    # Use IDs instead of names for validation
    declared_strategies    = {s['id'] for s in strategies}
    missing_implementation = declared_strategies - implemented_strategies
    unused_implementation  = implemented_strategies - declared_strategies
    
    errors   = []
    warnings = []
    
    # ========================================================================
    # Val Y1: Strategy IDs have implementations
    # ========================================================================
    if missing_implementation:
        warnings.append(f"Strategies WITHOUT implementation (will be skipped): {missing_implementation}")
    
    if unused_implementation:
        warnings.append(f"Implemented but NOT declared: {unused_implementation}")
    
    if not missing_implementation:
        logger.debug("Val Y1: All strategy IDs implemented")
    
    # ========================================================================
    # Val Y2: Direction is valid
    # ========================================================================
    validation_y2_errors = 0
    for strat in strategies:
        direction = strat.get('direction', '')
        strat_id = strat.get('id', 'UNKNOWN')

        if direction not in ['long', 'short']:
            errors.append(
                f"Strategy '{strat_id}' has invalid direction='{direction}' "
                f"(must be 'long' or 'short')"
            )
            validation_y2_errors += 1
    
    if validation_y2_errors == 0:
        logger.debug("Val Y2: All directions valid")
    
    # ========================================================================
    # Val Y3: Order amount within valid range
    # ========================================================================
    validation_y3_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        order_amount = strat.get('order_amount', None)
        
        if order_amount is None:
            errors.append(
                f"Strategy '{strat_id}' is missing 'order_amount' parameter"
            )
            validation_y3_errors += 1
        elif not isinstance(order_amount, (int, float)):
            errors.append(
                f"Strategy '{strat_id}' has invalid order_amount='{order_amount}' "
                f"(must be a number)"
            )
            validation_y3_errors += 1
        elif order_amount < MIN_ORDER_AMOUNT:
            errors.append(
                f"Strategy '{strat_id}' has order_amount={order_amount} "
                f"(minimum is {MIN_ORDER_AMOUNT})"
            )
            validation_y3_errors += 1
        elif order_amount > MAX_ORDER_AMOUNT:
            errors.append(
                f"Strategy '{strat_id}' has order_amount={order_amount} "
                f"(maximum is {MAX_ORDER_AMOUNT})"
            )
            validation_y3_errors += 1
    
    if validation_y3_errors == 0:
        logger.debug("Val Y3: All order amounts in range")
    
    # ========================================================================
    # Val Y4: Common required parameters
    # ========================================================================
    validation_y4_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        
        for param in COMMON_REQUIRED_PARAMS:
            if param not in strat:
                errors.append(
                    f"Strategy '{strat_id}' is missing required parameter: '{param}'"
                )
                validation_y4_errors += 1
    
    if validation_y4_errors == 0:
        logger.debug("Val Y4: All strategies have required parameters")
    
    # ========================================================================
    # Val Y5: TP/SL within valid ranges
    # ========================================================================
    validation_y5_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        tp_pct = strat.get('tp_pct', None)
        sl_pct = strat.get('sl_pct', None)
        
        if tp_pct and (tp_pct < MIN_TP_PCT or tp_pct > MAX_TP_PCT):
            errors.append(
                f"Strategy '{strat_id}' has tp_pct={tp_pct} "
                f"(valid range: {MIN_TP_PCT}-{MAX_TP_PCT}%)"
            )
            validation_y5_errors += 1
        
        if sl_pct and (sl_pct < MIN_SL_PCT or sl_pct > MAX_SL_PCT):
            errors.append(
                f"Strategy '{strat_id}' has sl_pct={sl_pct} "
                f"(valid range: {MIN_SL_PCT}-{MAX_SL_PCT}%)"
            )
            validation_y5_errors += 1
    
    if validation_y5_errors == 0:
        logger.debug("Val Y5: All TP/SL percentages within valid ranges")
    
    # ========================================================================
    # Val Y6: Unique strategy IDs
    # ========================================================================
    validation_y6_errors = 0
    ids = [s.get('id') for s in strategies]
    duplicates = [id for id in set(ids) if ids.count(id) > 1]
    
    if duplicates:
        errors.append(f"Duplicate strategy IDs found: {duplicates}")
        validation_y6_errors += 1
    
    if validation_y6_errors == 0:
        logger.debug("Val Y6: All strategy IDs are unique")
    
    # ========================================================================
    # Val Y7: Candles within reasonable range
    # ========================================================================
    validation_y7_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        candles = strat.get('sell_after_ncandles', None)
        
        if candles and (candles < MIN_CANDLES or candles > MAX_CANDLES):
            errors.append(
                f"Strategy '{strat_id}' has sell_after_ncandles={candles} "
                f"(typical range: {MIN_CANDLES}-{MAX_CANDLES})"
            )
            validation_y7_errors += 1
    
    if validation_y7_errors == 0:
        logger.debug("Val Y7: All sell_after_ncandles within range")
    
    
    # ========================================================================
    # Val Y9: Valid timeframes
    # ========================================================================
    validation_y9_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        timeframe = strat.get('timeframe', '')
        
        if timeframe not in VALID_TIMEFRAMES:
            errors.append(
                f"Strategy '{strat_id}' has invalid timeframe='{timeframe}' "
                f"(valid: {', '.join(VALID_TIMEFRAMES)})"
            )
            validation_y9_errors += 1
    
    if validation_y9_errors == 0:
        logger.debug("Val Y9: All timeframes are valid")
    
    # ========================================================================
    # Val Y10: ID format with numeric prefix
    # ========================================================================
    validation_y10_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', '')
        
        if not re.match(r'^\d{5,6}_\w+', strat_id):
            errors.append(
                f"Strategy '{strat_id}' has invalid ID format. "
                f"Expected: 'NNNNN_strategy_name' or 'NNNNNN_strategy_name' (e.g., '00053_rule_mining_id')"
            )
            validation_y10_errors += 1
            continue
        
        id_parts = strat_id.split('_', 1)
        id_number = id_parts[0]
        
        if len(id_number) not in (5, 6):
            errors.append(
                f"Strategy '{strat_id}' numeric prefix must be 5 or 6 digits "
                f"(e.g., '00053_name' or '100053_name')"
            )
            validation_y10_errors += 1
        
        try:
            num_value = int(id_number)
            if num_value < 1:
                errors.append(
                    f"Strategy '{strat_id}' numeric prefix must be a positive number "
                    f"(found: {id_number})"
                )
                validation_y10_errors += 1
        except ValueError:
            errors.append(
                f"Strategy '{strat_id}' numeric prefix is not a valid number"
            )
            validation_y10_errors += 1
    
    if validation_y10_errors == 0:
        logger.debug("Val Y10: All IDs have correct prefix format (NNNNN_name)")
    
    return errors, warnings