@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Could not find the project Python environment:
    echo "%~dp0venv\Scripts\python.exe"
    echo.
    echo Make sure the project virtual environment exists in the venv folder.
    pause
    exit /b 1
)

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
choice /c 123 /n /m "Enter choice (1-3): "

if errorlevel 3 goto end
if errorlevel 2 goto web
if errorlevel 1 goto popup

:popup
if not exist "Pedestriansdetection.py" (
    echo Could not find Pedestriansdetection.py in the project folder.
    pause
    goto menu
)
echo.
echo Running Pedestriansdetection.py...
"venv\Scripts\python.exe" "Pedestriansdetection.py"
echo.
echo Script execution finished.
pause
goto menu

:web
if not exist "app.py" (
    echo Could not find app.py in the project folder.
    pause
    goto menu
)
echo.
echo Starting Web Dashboard on http://127.0.0.1:8000...
echo The browser may take a little while to load while the models start.
start "" "http://127.0.0.1:8000"
"venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
goto menu

:end
exit /b 0
