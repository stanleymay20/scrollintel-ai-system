@echo off
REM ScrollIntel Quick Start Script for Windows
REM This script sets up and runs the complete ScrollIntel system

echo 🚀 Starting ScrollIntel AI System...

REM Check if .env exists, if not copy from example
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your API keys before continuing!
    echo    Required: OPENAI_API_KEY, JWT_SECRET_KEY
    pause
)

REM Start all services
echo 🐳 Starting Docker containers...
docker-compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak > nul

REM Check service health
echo 🔍 Checking service health...

REM Check backend health
curl -f http://localhost:8000/health > nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Backend API is healthy
) else (
    echo ❌ Backend API is not responding
)

REM Check frontend health
curl -f http://localhost:3000 > nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Frontend is healthy
) else (
    echo ❌ Frontend is not responding
)

REM Display access information
echo.
echo 🎉 ScrollIntel is ready!
echo.
echo 📱 Frontend UI:      http://localhost:3000
echo 🔧 Backend API:      http://localhost:8000
echo 📚 API Docs:         http://localhost:8000/docs
echo 💾 Database:         localhost:5432
echo 🗄️  Redis Cache:      localhost:6379
echo.
echo 🔍 To view logs:     docker-compose logs -f
echo 🛑 To stop:          docker-compose down
echo.
echo Happy coding! 🚀
pause