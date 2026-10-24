@echo off
title SafePath AI Platform Launcher
color 0B

echo ================================================================
echo   SAFEPATH AI -- Intelligent Mobility ^& Pedestrian Safety
echo   Final-Year CSE Major Project Platform
echo ================================================================
echo.

echo [1/2] Verifying Python installation...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in system PATH!
    pause
    exit /b 1
)

echo.
echo [2/2] Launching SafePath AI Platform on http://localhost:8000 ...
echo [INFO] Opening default browser automatically...
python main.py

pause
