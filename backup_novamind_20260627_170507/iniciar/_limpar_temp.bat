@echo off
title NovaMind - Limpeza de Arquivos Temporários
cd /d "%~dp0.."

echo Limpando arquivos temporários do YouTube...
echo.

set "YT_TEMP=%TEMP%\novamind_yt"

if exist "%YT_TEMP%" (
    echo Encontrado: %YT_TEMP%
    del /q "%YT_TEMP%\*" 2>nul
    rmdir /s /q "%YT_TEMP%\frames" 2>nul
    echo [OK] Arquivos temporários removidos.
) else (
    echo [INFO] Pasta temporária não existe: %YT_TEMP%
)

echo.
echo Limpando cache do Python...
for /d %%p in ("%~dp0..\__pycache__") do rmdir /s /q "%%p" 2>nul
for /d %%p in ("%~dp0..\novamind\__pycache__") do rmdir /s /q "%%p" 2>nul
for /d %%p in ("%~dp0..\novamind\sci\__pycache__") do rmdir /s /q "%%p" 2>nul
for /d %%p in ("%~dp0..\novamind\knowledge\__pycache__") do rmdir /s /q "%%p" 2>nul
echo [OK] Cache Python limpo.

echo.
echo Limpeza concluída!
pause >nul