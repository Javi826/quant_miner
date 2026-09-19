#!/bin/bash
echo "[1/3] Arrancando MT5..."
docker start quant_mt5
sleep 8

echo "[2/3] Arrancando servidor RPyC..."
docker exec -d -u abc quant_mt5 bash -c "wine python -m mt5linux --host 0.0.0.0 --port 8001"
sleep 3

echo "[3/3] Arrancando bot en core 2..."
cd ~/projects/quant/quant_miner/darwinex/BOT_forex
source ~/projects/quant/env_quant/bin/activate
exec taskset -c 2 python darwinex/live/main.py
