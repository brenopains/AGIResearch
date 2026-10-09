param([switch]$Rebuild,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$python='C:\Program Files\Python312\python.exe'
$nodeRoot='C:\Program Files\nodejs'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PATH="$nodeRoot;$env:PATH"
if($Rebuild -or -not(Test-Path -LiteralPath "$taskRoot\frontend\dist\index.html")){
    Push-Location "$taskRoot\frontend"
    try { & "$nodeRoot\npm.cmd" run build; if($LASTEXITCODE -ne 0){throw 'A compilação falhou.'} }
    finally {Pop-Location}
}
$alive=$false
try {$health=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 2;$alive=$health.service -eq 'novamind-observatory'} catch {}
if(-not $alive){
    $logDir=Join-Path $taskRoot 'data'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $proc=Start-Process -FilePath $python -ArgumentList @('-B','-m','uvicorn','backend.server:app','--host','127.0.0.1','--port','8765') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput "$logDir\server.log" -RedirectStandardError "$logDir\server-error.log" -PassThru
    Set-Content -LiteralPath "$logDir\server.pid" -Value $proc.Id
    for($attempt=0;$attempt -lt 30;$attempt++){
        Start-Sleep -Milliseconds 400
        if($proc.HasExited){throw "O servidor encerrou. Consulte $logDir\server-error.log"}
        try {$health=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 1;if($health.ok){$alive=$true;break}} catch {}
    }
    if(-not $alive){throw 'O servidor ainda não respondeu; consulte os logs.'}
}
Write-Output 'NovaMind: http://127.0.0.1:8765'
if(-not $NoBrowser){Start-Process 'http://127.0.0.1:8765'}
