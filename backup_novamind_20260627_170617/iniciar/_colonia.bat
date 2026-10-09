@echo off
title NovaMind - Sala de Aula Multi-Mente :8000
cd /d "%~dp0.."

set "PATH=%PATH%;%USERPROFILE%\AppData\Roaming\Python\Python312\Scripts"

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║     NovaMind - Sala de Aula (Colonia)       ║
echo  ║  5 mentes conversando sobre o que aprendem  ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  Frontend: http://localhost:5173/
echo  Backend:  http://localhost:8000/ (modo colonia)
echo.

echo [1/2] Iniciando Backend (modo sala de aula)...
start "NovaMind - Sala de Aula" cmd /k "cd /d "%~dp0.." && set PATH=%PATH% && python server.py --colony"

echo [2/2] Iniciando Frontend...
timeout /t 3 /nobreak >nul
start "NovaMind Frontend" cmd /k "cd /d "%~dp0..\frontend" && npm run dev"

echo.
echo  Sala de aula rodando em http://localhost:5173/
echo.
pause >nul
