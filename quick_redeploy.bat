@echo off
echo ScrollIntel Quick Redeployment
echo ==============================

echo Stopping existing containers...
docker-compose down

echo Building fresh containers...
docker-compose build --no-cache

echo Starting ScrollIntel...
docker-compose up -d

echo Waiting for services to start...
timeout /t 30 /nobreak

echo Checking status...
docker-compose ps

echo.
echo Redeployment complete!
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
echo Health: http://localhost:8000/health

pause