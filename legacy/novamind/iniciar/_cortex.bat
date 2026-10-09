@echo off
title NovaMind - Cortex :8793
cd /d "%~dp0.."
echo Cortex AO VIVO em http://127.0.0.1:8793/
python tmp\arc_agi\cortex_server.py 8793
echo.
echo [Cortex encerrou] Pressione uma tecla para fechar.
pause >nul
