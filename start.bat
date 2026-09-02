@echo off
:menu
cls
echo ============================================================
echo               PEDESTRIAN DETECTION SYSTEM
echo ============================================================
echo.
echo Please select an option:
echo [1] Run the Matplotlib Popup Script (Pedestriansdetection.py)
echo [2] Start the Interactive Web Dashboard Server (app.py)
echo [3] Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running Pedestriansdetection.py...
    .\venv\Scripts\python.exe Pedestriansdetection.py
    echo.
    echo Script execution finished.
    pause
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Starting Web Dashboard on http://127.0.0.1:8000...
    echo (Opening browser...)
    start http://127.0.0.1:8000
    .\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
    pause
    goto menu
)

if "%choice%"=="3" (
    exit
)

goto menu
