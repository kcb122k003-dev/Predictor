@echo off
rem Start Exam Predictor and open it in your browser.
cd /d "%~dp0\.."
if not exist ".venv\Scripts\predictor.exe" (
  echo Not installed yet. Run: powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  pause
  exit /b 1
)
".venv\Scripts\predictor.exe" serve %*
pause
