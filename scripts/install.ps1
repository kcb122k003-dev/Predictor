# One-command install for Windows (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#   add -Dev to install test tools, -Neural for the optional neural embedding library
param([switch]$Dev, [switch]$Neural)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) { $python = "py"; $pyArgs = @("-3") } else { $python = "python"; $pyArgs = @() }
try { & $python @pyArgs -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" }
catch { Write-Host "Python 3.10 or newer is required: https://www.python.org/downloads/ (tick 'Add to PATH')."; exit 1 }
if ($LASTEXITCODE -ne 0) { Write-Host "Python 3.10 or newer is required."; exit 1 }

$extras = "ocr"
if ($Dev) { $extras += ",dev" }
if ($Neural) { $extras += ",neural" }

Write-Host "Creating a virtual environment in .venv ..."
& $python @pyArgs -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
Write-Host "Installing Exam Predictor (extras: $extras) ..."
& .\.venv\Scripts\python.exe -m pip install -e ".[$extras]"

$tess = Get-Command tesseract -ErrorAction SilentlyContinue
if (-not $tess -and -not (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe")) {
  Write-Host ""
  Write-Host "Tesseract OCR was not found. Scanned papers and photos need it."
  Write-Host "Install it from https://github.com/UB-Mannheim/tesseract/wiki (default folder is fine)."
}
Write-Host ""
& .\.venv\Scripts\predictor.exe doctor
Write-Host ""
Write-Host "Done. Start the app by double-clicking scripts\start.bat"
