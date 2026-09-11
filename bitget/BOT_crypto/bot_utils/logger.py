"""
Professional logging system for trading bot.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import CONSOLE_LOG_LEVEL, FILE_LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT  # ⭐ Añadir


def setup_logger(
    log_dir: str,
    logfile_name: str = 'bot.log',
    console_level: str = None,  
    file_level: str = None,     
    max_bytes: int = None,
    backup_count: int = None
) -> logging.Logger:
    """
    Setup professional logging system with console and file handlers.
    
    If levels not provided, uses values from config.settings.
    """
    #  Usar valores de settings.py si no se proporcionan
    if console_level is None:
        console_level = CONSOLE_LOG_LEVEL
    if file_level is None:
        file_level = FILE_LOG_LEVEL
    if max_bytes is None:
        max_bytes = LOG_MAX_BYTES
    if backup_count is None:
        backup_count = LOG_BACKUP_COUNT
    
    # Convertir strings a niveles de logging
    console_level_int = getattr(logging, console_level.upper())
    file_level_int = getattr(logging, file_level.upper())
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, logfile_name)
    
    # Get root logger
    root_logger = logging.getLogger('BOT_crypto')
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.propagate = False
    
    # ========================================================================
    # CONSOLE HANDLER
    # ========================================================================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level_int) 
    
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # ========================================================================
    # FILE HANDLER
    # ========================================================================
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level_int)  # ⭐ Usar nivel convertido
    
    # Detailed format for file
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # ========================================================================
    # ADD HANDLERS TO LOGGER
    # ========================================================================
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log initial message
    root_logger.info("=" * 60)
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log file: {log_path}")
    root_logger.info(f"Console level: {console_level}")  
    root_logger.info(f"File level: {file_level}")        
    root_logger.info("=" * 60)
    
    return root_logger


def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f'BOT_crypto.{module_name}')


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
def setup_print_logger(logdir: str, logfile_name: str = None):
    """Backward compatibility wrapper."""
    if logfile_name is None:
        logfile_name = 'bot.log'
    
    # ⭐ Ahora usa settings.py automáticamente
    return setup_logger(
        log_dir=logdir,
        logfile_name=logfile_name
    )