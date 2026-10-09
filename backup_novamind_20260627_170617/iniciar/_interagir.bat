@echo off
title NovaMind - Interacao por Texto
cd /d "%~dp0..\"

echo ============================================
echo   NovaMind — Interacao por Texto
echo   (REPL com 13 camadas + memoria HDV)
echo ============================================
echo.

:: Verifica se o Python encontra o módulo
python -c "from core.interact import main; main()" 2>&1

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Falha ao iniciar. Verifique:
    echo   1. O ambiente virtual esta ativo
    echo   2. Os modulos estao instalados
    echo   3. Rode _ingestao.bat primeiro se quiser memoria HDV
    pause
    exit /b 1
)

pause
