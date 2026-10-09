@echo off
title NovaMind - Servidor Minecraft (Paper 1.20.4) :25565
cd /d "%~dp0..\tmp\mc\server"
echo Subindo servidor Paper 1.20.4 em localhost:25565 ...
java -Xmx2G -Xms1G -jar paper.jar nogui
echo.
echo [Servidor Paper encerrou] Pressione uma tecla para fechar esta janela.
pause >nul
