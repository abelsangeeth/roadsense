@echo off
rem RoadSense one-click launcher: creates a local Python environment on first run, then starts the dashboard.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found. Install Python 3.10+ from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist venv\Scripts\python.exe (
    echo First run: installing packages, this takes 5-10 minutes...
    python -m venv venv || (pause & exit /b 1)
    venv\Scripts\python -m pip install --upgrade pip
    venv\Scripts\python -m pip install torch torchvision || (pause & exit /b 1)
    venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
)
echo Starting RoadSense... open http://127.0.0.1:8765 in your browser once it says "Uvicorn running".
rem open the browser once the models have had time to load
start "" /min cmd /c "timeout /t 30 /nobreak >nul & start http://127.0.0.1:8765"
venv\Scripts\python -m uvicorn serve.app:app --host 127.0.0.1 --port 8765
pause
