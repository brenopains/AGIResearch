# NovaMind Grounding System Launcher
# Frontend: http://localhost:5173/
# Backend:  http://localhost:8000/

$env:PATH += ";$env:USERPROFILE\AppData\Roaming\Python\Python312\Scripts"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  NovaMind Grounding System" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:5173/" -ForegroundColor Yellow
Write-Host "  Backend:  http://localhost:8000/" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendPath = Join-Path $projectRoot "frontend"

Write-Host "[1/2] Iniciando Backend (FastAPI :8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PATH += ';$env:USERPROFILE\AppData\Roaming\Python\Python312\Scripts'; cd '$projectRoot'; python server.py" -WindowStyle Normal

Write-Host "[2/2] Aguardando backend subir..." -ForegroundColor Gray
Start-Sleep -Seconds 3

Write-Host "[2/2] Iniciando Frontend (Vite :5173)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; npm run dev" -WindowStyle Normal

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Ambos servidores iniciados!" -ForegroundColor Green
Write-Host "  Abra http://localhost:5173/ no navegador" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pressione qualquer tecla para fechar esta janela..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")