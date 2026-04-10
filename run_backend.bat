@echo off
echo ========================================================
echo   COVID-19 AI Detection System - Backend Server Launcher
echo ========================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo [*] Checking dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [*] Launching FastAPI Backend Server...
echo [*] The API documentation will be available at: http://127.0.0.1:8000/docs
echo.
python backend.py

pause
