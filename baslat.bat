@echo off
chcp 65001 > nul
echo ================================================================
echo           Ses Analizi Platformu Baslatiliyor...
echo ================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

echo [1/2] Arka Plan Analiz Isleyicisi (Stream Worker) baslatiliyor...
start "Ses Analizi - Stream Worker" cmd /k "%PYTHON_CMD% -m audio_analyzer.workers.stream_worker"

echo [2/2] Web Sunucusu (FastAPI + Uvicorn) baslatiliyor...
start "Ses Analizi - Web API" cmd /k "%PYTHON_CMD% -m uvicorn audio_analyzer.api.main:app --host 127.0.0.1 --port 8000 --reload"

echo.
echo ================================================================
echo   Her iki servis ayri pencerelerde baslatildi!
echo   Tarayicinizda acin: http://127.0.0.1:8000
echo ================================================================
