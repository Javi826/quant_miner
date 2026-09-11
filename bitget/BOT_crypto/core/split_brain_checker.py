#BOT_crypto/core/split_brain_checker.py

import sys
import logging
import subprocess
from typing import Tuple

from config.settings import IS_VPS, VPS_CHECK_CONFIG

logger = logging.getLogger('BOT_crypto.core.split_brain_checker')


def check_vps_status() -> Tuple[bool, str]:

    if IS_VPS:
        # Running on VPS - no need to check
        return True, "Running on VPS - no split-brain check needed"
    
    try:
        # Query VPS PostgreSQL via Tailscale
        result = subprocess.run(
            ["psql",
             "-h", VPS_CHECK_CONFIG['host'],
             "-U", VPS_CHECK_CONFIG['user'],
             "-d", VPS_CHECK_CONFIG['dbname'],
             "-t", "-c", "SELECT pg_is_in_recovery();"],
            capture_output=True,
            text=True,
            timeout=VPS_CHECK_CONFIG['timeout'],
            env={'PGPASSWORD': VPS_CHECK_CONFIG['password']}
        )
        
        if result.returncode == 0:
            is_vps_standby = result.stdout.strip() == 't'
            
            if is_vps_standby:
                # VPS is standby - safe to run LOCAL
                return True, "VPS is standby - safe to proceed"
            else:
                # VPS is primary - LOCAL must stop
                return False, (
                    "🚨 VPS is PRIMARY - LOCAL must shutdown to avoid split-brain\n"
                    "VPS is currently operating in failover mode\n"
                    "Wait for failback procedure before restarting LOCAL"
                )
        else:
            # Connection failed but could be network issue
            logger.warning(f"Could not verify VPS status (exit code {result.returncode})")
            # Assume safe to proceed (VPS might be down)
            return True, "VPS check failed - assuming VPS is down/unreachable"
    
    except subprocess.TimeoutExpired:
        logger.warning("VPS check timeout - assuming VPS is down/unreachable")
        return True, "VPS check timeout - proceeding with LOCAL operation"
    
    except Exception as e:
        logger.warning(f"VPS check error: {e}")
        return True, f"VPS check error: {e} - proceeding with LOCAL operation"


def check_split_brain(orchestrator_instance=None) -> None:

    if IS_VPS:
        # Running on VPS - no check needed
        return
    
    is_safe, message = check_vps_status()
    
    if not is_safe:
        # VPS is primary - LOCAL must shutdown
        logger.critical("=" * 60)
        logger.critical("SPLIT-BRAIN PROTECTION TRIGGERED")
        logger.critical("=" * 60)
        logger.critical(message)
        logger.critical("Shutting down LOCAL bot immediately")
        logger.critical("=" * 60)
        
        # Print to console as well
        print("\n" + "=" * 60)
        print("🚨 CRITICAL: SPLIT-BRAIN PROTECTION")
        print("=" * 60)
        print(message)
        print("\nLOCAL bot is shutting down immediately")
        print("=" * 60 + "\n")
        
        # Graceful shutdown if orchestrator instance provided
        if orchestrator_instance:
            try:
                logger.info("Attempting graceful shutdown...")
                orchestrator_instance.shutdown()
            except Exception as e:
                logger.error(f"Error during graceful shutdown: {e}")
        
        # Exit
        sys.exit(1)