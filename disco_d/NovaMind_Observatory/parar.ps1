$ErrorActionPreference='Stop'
$pidFile=Join-Path $PSScriptRoot 'data\server.pid'
if(Test-Path -LiteralPath $pidFile){
    $serverPid=[int](Get-Content -LiteralPath $pidFile -Raw)
    $proc=Get-CimInstance Win32_Process -Filter "ProcessId=$serverPid" -ErrorAction SilentlyContinue
    if($proc -and $proc.CommandLine -match 'uvicorn\s+backend\.server:app' -and $proc.CommandLine -match '8765'){
        Stop-Process -Id $serverPid
        Write-Output 'Servidor NovaMind encerrado.'
    }else{Write-Output 'O processo salvo já encerrou ou não corresponde ao servidor.'}
}else{Write-Output 'Nenhum processo registrado por iniciar.ps1.'}
