@echo off
title NovaMind - Colonia AGI (10 bots)
cd /d "%~dp0..\tmp\mc"
:loop
echo [supervisor] iniciando colonia (heap 6GB, 10 bots) %TIME%
node --max-old-space-size=6144 colony.js 10
echo [supervisor] colonia saiu, reiniciando em 6s (o estado persiste em disco)...
timeout /t 6 /nobreak >nul
goto loop
