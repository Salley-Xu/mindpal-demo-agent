Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$python = "D:\anaconda3\envs\xwjpy_312\python.exe"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location (Join-Path $projectRoot "frontend")
try {
    & $python -m streamlit run ".\frontend.py"
}
finally {
    Pop-Location
}
