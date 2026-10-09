@echo off
title NovaMind - Laboratorio :8771
cd /d "%~dp0.."
echo Laboratorio cientifico em http://127.0.0.1:8771/
python sci\lab\lab_server.py 8771
echo.
echo [Lab encerrou] Pressione uma tecla para fechar.
pause >nul
