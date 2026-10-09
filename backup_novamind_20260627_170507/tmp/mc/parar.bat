@echo off
echo Parando colonia (salva cerebros) e servidor (salva mundo)...
taskkill /IM node.exe /T >/dev/null 2>&1
taskkill /IM java.exe /T >/dev/null 2>&1
echo Parado. Mundo e cerebros salvos.
timeout /t 3 /nobreak >/dev/null
