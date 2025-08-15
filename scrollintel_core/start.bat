@echo off
REM ================================
REM ScrollIntel Core - Startup Script (Windows)
REM Starts the focused AI-CTO platform
REM ================================

echo 🚀 Starting ScrollIntel Core...

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found. Copying from .env.example...
    copy .env.example .env
    echo 📝 Please edit .env file with your configuration before running again.
    pause
    exit /b 1
)

REM Create necessary directories
if not exist uploads mkdir uploads
if not exist logs mkdir logs

echo 🐳 Starting Docker services...

REM Start with Docker Compose
if "%1"=="dev" (
    echo 🔧 Starting in development mode...
    docker-compose up --build
) else if "%1"=="prod" (
    echo 🏭 Starting in production mode...
    docker-compose -f docker-compose.yml up -d --build
) else (
    echo 📖 Usage: %0 [dev^|prod]
    echo    dev  - Start in development mode ^(with logs^)
    echo    prod - Start in production mode ^(detached^)
    pause
    exit /b 1
)