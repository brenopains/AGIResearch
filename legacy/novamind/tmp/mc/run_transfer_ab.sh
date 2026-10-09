#!/bin/bash
cd /c/Users/usuario/Desktop/AGI/novamind/tmp/mc
powershell.exe -Command "Get-Process node -EA SilentlyContinue | Stop-Process -Force" 2>/dev/null
sleep 5
echo "=== ARM A: LOADED (cerebro treinado) ==="
node transfer.js loaded 150 > t_loaded.log 2>&1
sleep 10
echo "=== ARM B: WIPED (cerebro zerado) ==="
node transfer.js wiped 150 > t_wiped.log 2>&1
sleep 5
echo "=== RESULTADOS ==="
echo "--- LOADED ---"; grep -E "spawn|MORREU|RESULTADO|vivos no fim" t_loaded.log
echo "--- WIPED ---"; grep -E "spawn|MORREU|RESULTADO|vivos no fim" t_wiped.log
nohup node gamemaster.js > gm.log 2>&1 &
sleep 8
nohup node --max-old-space-size=3072 colony.js 6 > colony.log 2>&1 &
echo "=== colonia religada ==="
