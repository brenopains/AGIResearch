@echo off
title NovaMind - AGI :8000 + Frontend :5173
cd /d "%~dp0.."

set "PATH=%PATH%;%USERPROFILE%\AppData\Roaming\Python\Python312\Scripts"

echo. 
echo  ╔══════════════════════════════════════════════╗
echo  ║           NovaMind AGI - Grounding           ║
echo  ╠══════════════════════════════════════════════╣
echo  ║  Ciencia, mundo modelo, memoria adaptativa,  ║
echo  ║  neuronios espelho, playground embody,        ║
echo  ║  codigo de sinais emergente.                 ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  Frontend: http://localhost:5173/
echo  Backend:  http://localhost:8000/
echo.
echo  Adicione --colony para modo sala de aula
echo  Exemplo: python server.py --colony
echo.

echo [1/2] Iniciando Backend...
start "NovaMind Backend" cmd /k "cd /d "%~dp0.." && set PATH=%PATH%;%USERPROFILE%\AppData\Roaming\Python\Python312\Scripts && python server.py"

echo [2/2] Iniciando Frontend...
timeout /t 3 /nobreak >nul
start "NovaMind Frontend" cmd /k "cd /d "%~dp0..\frontend" && npm run dev"

echo.
echo  Ambos servidores rodando. Abra http://localhost:5173/
echo  Para sala de aula: feche e rode: _colonia.bat
echo.
pause >nul
