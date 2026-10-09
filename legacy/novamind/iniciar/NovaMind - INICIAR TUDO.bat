@echo off
chcp 65001 >nul
title NovaMind - Iniciar Tudo
cls
echo ===================================================================
echo                 NovaMind  -  subindo todos os servidores
echo ===================================================================
echo.
echo   1) Minecraft real (Paper)  -^> localhost:25565  (jogo Java 1.20.4)
echo   2) Colonia AGI (10 bots)   -^> entra no servidor sozinha
echo   3) Laboratorio             -^> http://127.0.0.1:8771/
echo   4) Cortex                  -^> http://127.0.0.1:8793/
echo   5) ALife web               -^> http://127.0.0.1:5005/
echo.
echo -------------------------------------------------------------------

start "NovaMind Paper"   "%~dp0_paper.bat"
echo [1/5] Servidor Paper subindo... aguardando 25s para o mundo carregar
timeout /t 25 /nobreak >nul

start "NovaMind Colonia" "%~dp0_colonia.bat"
echo [2/5] Colonia AGI subindo (os 10 bots entram em ~1 min)

start "NovaMind Lab"     "%~dp0_lab.bat"
echo [3/5] Laboratorio  :8771
start "NovaMind Cortex"  "%~dp0_cortex.bat"
echo [4/5] Cortex       :8793
start "NovaMind ALife"   "%~dp0_alife.bat"
echo [5/5] ALife web    :5005

echo.
echo   Abrindo os fronts no navegador...
timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:8771/
start "" http://127.0.0.1:8793/
start "" http://127.0.0.1:5005/

echo.
echo ===================================================================
echo   PRONTO.
echo   Para o MINECRAFT: abra o jogo Java 1.20.4
echo      Multijogador  -  Conexao direta  -  localhost  -  Conectar
echo.
echo   Cada servidor abriu na sua propria janela.
echo   Para parar UM: feche a janela dele.
echo   Para parar TUDO: use  "NovaMind - PARAR TUDO.bat"
echo ===================================================================
echo.
pause
