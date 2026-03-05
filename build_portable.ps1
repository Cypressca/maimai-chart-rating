$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python not found at $python. Please create .venv first."
}

& $python -m pip install -U pyinstaller
& $python -m PyInstaller --clean --onefile --name maimai_const_predictor_portable maimai_const_predictor.py

$releaseDir = Join-Path $projectRoot "portable_release"
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

Copy-Item (Join-Path $projectRoot "dist\maimai_const_predictor_portable.exe") (Join-Path $releaseDir "maimai_const_predictor_portable.exe") -Force
Copy-Item (Join-Path $projectRoot "README_MODEL.md") (Join-Path $releaseDir "README_MODEL.md") -Force
Copy-Item (Join-Path $projectRoot "start_portable.bat") (Join-Path $releaseDir "start_portable.bat") -Force

Write-Host "Build complete. Portable folder: $releaseDir"
