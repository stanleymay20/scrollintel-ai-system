@echo off
echo ================================
echo ScrollIntel Local Demo Launcher
echo ================================
echo.

REM Set local environment variables
set DATABASE_URL=sqlite:///./scrollintel_local.db
set ENVIRONMENT=development
set DEBUG=true

echo Starting ScrollIntel components...
echo.

REM Start frontend in a new window
echo Starting frontend server...
start "ScrollIntel Frontend" cmd /k "cd frontend && npm run dev"

REM Wait a moment for frontend to start
timeout /t 3 /nobreak > nul

REM Start backend
echo Starting backend API server...
python start_local_demo.py

pause