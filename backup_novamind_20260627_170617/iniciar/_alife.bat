@echo off
title NovaMind - ALife web (maestro) :5005
cd /d "%~dp0.."
echo ALife web (maestro governando elite) em http://127.0.0.1:5005/
python tmp\game\server.py
echo.
echo [ALife encerrou] Pressione uma tecla para fechar.
pause >nul
