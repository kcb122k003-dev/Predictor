@echo off
rem Double-click installer for Windows. Runs scripts\install.ps1.
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
