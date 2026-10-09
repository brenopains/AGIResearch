@echo off
chcp 65001 >nul
title NovaMind - Parar Tudo
echo Encerrando os servidores NovaMind (janelas iniciadas por este launcher)...
echo.
taskkill /F /T /FI "WINDOWTITLE eq NovaMind*" >nul 2>&1
echo Janelas NovaMind encerradas (Paper, Colonia, Lab, Cortex, ALife).
echo.
echo Se algum servidor ainda aparecer no jogo/navegador, feche a janela dele
echo manualmente. Este script NAO mata outros java/node/python da sua maquina.
echo.
pause
