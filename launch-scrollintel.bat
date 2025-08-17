@echo off
REM ================================
REM ScrollIntel™ Launch Script (Windows)
REM One-click setup and launch for ScrollIntel AI Platform
REM ================================

setlocal enabledelayedexpansion

REM Colors (Windows doesn't support colors in batch easily, so we'll use text)
set "INFO=[INFO]"
set "WARN=[WARN]"
set "ERROR=[ERROR]"
set "SUCCESS=[SUCCESS]"

REM ASCII Art Banner
echo.
echo  ========================================
echo  SCROLLINTEL AI PLATFORM LAUNCHER
echo  ========================================
echo 🚀 ScrollIntel™ AI Platform Launcher
echo Replace your CTO with AI agents that analyze data, build models, and make technical decisions
echo.

REM Check if Docker is installed and running
echo %INFO% Checking Docker installation...

docker --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker is not installed. Please install Docker Desktop first:
    echo   - Visit: https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker Compose is not installed. Please install Docker Compose first:
    echo   - Visit: https://docs.docker.com/compose/install/
    pause
    exit /b 1
)

echo %SUCCESS% Docker and Docker Compose are ready!

REM Setup environment file
echo %INFO% Setting up environment configuration...

if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo %SUCCESS% Created .env from .env.example
    ) else (
        echo %ERROR% .env.example file not found!
        pause
        exit /b 1
    )
) else (
    echo %WARN% .env file already exists, skipping creation
)

REM Generate JWT secret if not set
findstr /C:"JWT_SECRET_KEY=" .env | findstr /R "[a-zA-Z0-9]" >nul
if errorlevel 1 (
    REM Generate a random JWT secret using PowerShell
    for /f %%i in ('powershell -command "[System.Web.Security.Membership]::GeneratePassword(64, 10)"') do set JWT_SECRET=%%i
    powershell -command "(Get-Content .env) -replace 'JWT_SECRET_KEY=.*', 'JWT_SECRET_KEY=!JWT_SECRET!' | Set-Content .env"
    echo %SUCCESS% Generated secure JWT secret
)

REM Set default database password if not set
findstr /C:"POSTGRES_PASSWORD=" .env | findstr /R "[a-zA-Z0-9]" >nul
if errorlevel 1 (
    REM Generate a random database password using PowerShell
    for /f %%i in ('powershell -command "[System.Web.Security.Membership]::GeneratePassword(16, 4)"') do set DB_PASSWORD=%%i
    powershell -command "(Get-Content .env) -replace 'POSTGRES_PASSWORD=.*', 'POSTGRES_PASSWORD=!DB_PASSWORD!' | Set-Content .env"
    echo %SUCCESS% Generated database password
)

REM Check for required API keys
echo %INFO% Checking API key configuration...

findstr /C:"OPENAI_API_KEY=sk-" .env >nul
if errorlevel 1 (
    echo %WARN% OpenAI API key not configured in .env file
    echo   To enable AI features, add your OpenAI API key to .env:
    echo   OPENAI_API_KEY=sk-your-key-here
    echo.
) else (
    echo %SUCCESS% OpenAI API key configured
)

REM Start services
echo %INFO% Starting ScrollIntel services...

REM Pull latest images
echo %INFO% Pulling Docker images...
docker-compose pull

REM Build and start services
echo %INFO% Building and starting containers...
docker-compose up -d --build

echo %SUCCESS% All services started!

REM Wait for services to be ready
echo %INFO% Waiting for services to be ready...

REM Wait for database (simplified check for Windows)
echo %INFO% Waiting for PostgreSQL...
timeout /t 10 /nobreak >nul

REM Wait for backend API (simplified check for Windows)
echo %INFO% Waiting for backend API...
timeout /t 15 /nobreak >nul

REM Wait for frontend (simplified check for Windows)
echo %INFO% Waiting for frontend...
timeout /t 10 /nobreak >nul

REM Display success information
echo.
echo 🎉 ScrollIntel™ is now running!
echo.
echo 📱 Access Points:
echo   🌐 Frontend:    http://localhost:3000
echo   🔧 API:         http://localhost:8000
echo   📚 API Docs:    http://localhost:8000/docs
echo   ❤️  Health:     http://localhost:8000/health
echo.
echo 🚀 Quick Start:
echo   1. Open http://localhost:3000 in your browser
echo   2. Upload your data files (CSV, Excel, JSON)
echo   3. Chat with AI agents for insights
echo   4. Build ML models with AutoML
echo   5. Create interactive dashboards
echo.
echo 🛠️  Management:
echo   📊 View logs:   docker-compose logs -f
echo   🔄 Restart:     docker-compose restart
echo   🛑 Stop:        docker-compose down
echo.
echo ScrollIntel™ - Where artificial intelligence meets unlimited potential! 🌟
echo.

REM Open browser automatically
echo %INFO% Opening ScrollIntel in your default browser...
start http://localhost:3000

echo Press any key to exit...
pause >nul