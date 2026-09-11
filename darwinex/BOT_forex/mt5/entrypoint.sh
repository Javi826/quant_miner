#!/bin/bash
set -e

Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99

if [ "${ENABLE_VNC}" = "true" ]; then
    x11vnc -display :99 -nopw -listen 0.0.0.0 -xkb -forever -bg
    echo "[MT5] VNC arrancado en puerto 5900"
fi

if [ -f /mt5/config/mt5.ini ]; then
    MT5_PATH="$HOME/.wine/drive_c/Program Files/MetaTrader 5"
    cp /mt5/config/mt5.ini "$MT5_PATH/MetaTrader5.ini" 2>/dev/null || true
fi

EA_DIR="$HOME/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts"
if [ -d /mt5/experts ] && [ "$(ls -A /mt5/experts)" ]; then
    cp -r /mt5/experts/* "$EA_DIR/" 2>/dev/null || true
    echo "[MT5] EAs copiados"
fi

echo "[MT5] Arrancando MetaTrader 5..."
wine "$HOME/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe" \
    /portable \
    2>/mt5/logs/wine.log &

MT5_PID=$!
echo "[MT5] PID: $MT5_PID"
wait $MT5_PID
