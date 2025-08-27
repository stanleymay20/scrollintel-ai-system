@echo off
echo ========================================
echo ScrollIntel Demo Launcher
echo ========================================
echo.

echo Starting ScrollIntel Backend Server...
start "ScrollIntel Backend" python simple_server.py

echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo Opening ScrollIntel Frontend...
start "ScrollIntel Frontend" simple_frontend.html

echo.
echo ========================================
echo ScrollIntel Demo is now running!
echo ========================================
echo.
echo Backend API: http://127.0.0.1:8000
echo Frontend:    simple_frontend.html (opened in browser)
echo.
echo Press any key to stop all services...
pause >nul

echo Stopping services...
taskkill /f /im python.exe /fi "WINDOWTITLE eq ScrollIntel Backend*" 2>nul
echo Services stopped.
pause