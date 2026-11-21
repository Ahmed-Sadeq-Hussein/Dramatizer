@echo off
rem Double-click: the menu of main_offline.py (a local Dramatizer for debugging, on a copy of the data).
rem DRAMATIZER_PYTHON picks the Python environment; otherwise E:\envs\dramatizer, else "python".
cd /d "%~dp0"
set "PY=%DRAMATIZER_PYTHON%"
if "%PY%"=="" if exist "E:\envs\dramatizer\python.exe" set "PY=E:\envs\dramatizer\python.exe"
if "%PY%"=="" set "PY=python"
"%PY%" main_offline.py %*
echo.
pause
