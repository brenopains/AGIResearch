@echo off
title Ingestao CogLang - HDV Compiler
cd /d "%~dp0..\"

echo ============================================
echo   CogLang Compiler — Ingestao de Datasets
echo ============================================
echo.
echo Compila datasets do disco D: para .hdv (vetores HDC 10k dim)
echo RAM constante ~100 MB durante o processo
echo.

:: Verifica se Python encontra o módulo
python -c "from core.coglang_compiler import ingest_all; ingest_all()" 2>&1

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] A ingestao falhou. Verifique se:
    echo   1. O ambiente virtual esta ativo
    echo   2. D:/novamind_data existe
    echo   3. Ha datasets em D:/novamind_data/text_datasets/ e D:/novamind_data/MSVD/
    pause
    exit /b 1
)

echo.
echo Ingestao concluida com sucesso!
echo Arquivos .hdv salvos em: D:/novamind_data/hdv/
pause
