@echo off
cd /d "%~dp0server"
start "MC-Server" java -Xmx2G -jar paper.jar nogui
cd /d "%~dp0"
timeout /t 18 /nobreak >/dev/null
start "Colonia" node colony.js 10
echo Servidor + colonia iniciados. Entre no Minecraft Java 1.20.4 -> Multiplayer -> localhost
pause
