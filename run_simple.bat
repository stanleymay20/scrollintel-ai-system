@echo off
echo 🚀 Starting ScrollIntel Simple API...
echo 📍 This will try ports 8000, 8001, 8002, 8003, or 8080
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python from https://python.org
    pause
    exit /b 1
)

REM Install requirements if needed
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo 📦 Installing requirements...
pip install -q -r requirements_simple.txt

echo 🚀 Starting ScrollIntel API...
python run_simple.py

pause