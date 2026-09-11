from mt5linux import MetaTrader5
from loguru import logger
from dotenv import load_dotenv
import os, time

load_dotenv()

MT5_LOGIN    = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "Darwinex-Live")

mt5 = MetaTrader5(host='localhost', port=8001)

def connect():
    if not mt5.initialize():
        logger.error(f"initialize() falló: {mt5.last_error()}")
        return False
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logger.error(f"login() falló: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    logger.success(f"Conectado | Balance: {info.balance} {info.currency}")
    return True

if __name__ == "__main__":
    for i in range(5):
        logger.info(f"Intento {i+1}/5...")
        if connect():
            break
        time.sleep(10)
