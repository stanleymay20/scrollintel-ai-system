@echo off
echo Starting ScrollIntel...

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker not available, starting simple mode...
    python run_simple.py
    goto :end
)

REM Try Docker Compose
echo Starting with Docker Compose...
docker-compose -f docker-compose.minimal.yml up -d postgres redis
timeout /t 10 /nobreak >nul

REM Start Python backend
echo Starting ScrollIntel backend...
python run_simple.py

:end
pause
