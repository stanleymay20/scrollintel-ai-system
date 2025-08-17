#!/usr/bin/env powershell

Write-Host "=== PostgreSQL Docker Setup ===" -ForegroundColor Cyan
Write-Host "This will start PostgreSQL in Docker as an alternative to the installed version" -ForegroundColor Yellow
Write-Host ""

# Check if Docker is available
try {
    docker --version | Out-Null
    Write-Host "✓ Docker is available" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not available. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "Download from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
    exit 1
}

Write-Host ""
Write-Host "Starting PostgreSQL container..." -ForegroundColor Yellow

# Stop existing PostgreSQL service to avoid port conflicts
Write-Host "Stopping existing PostgreSQL service..." -ForegroundColor Yellow
Stop-Service -Name "postgresql-x64-17" -Force -ErrorAction SilentlyContinue

# Remove existing container if it exists
docker rm -f scrollintel-postgres 2>$null

# Start PostgreSQL container
Write-Host "Running Docker command..." -ForegroundColor Cyan
docker run -d --name scrollintel-postgres -e POSTGRES_PASSWORD=scrollintel123 -e POSTGRES_USER=postgres -e POSTGRES_DB=scrollintel -p 5432:5432 postgres:17

Write-Host ""
Write-Host "Waiting for PostgreSQL to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test connection
Write-Host "Testing connection..." -ForegroundColor Yellow
$env:PGPASSWORD = "scrollintel123"

try {
    # Add PostgreSQL client to PATH if available
    $env:PATH += ";C:\Program Files\PostgreSQL\17\bin"
    
    $result = psql -h localhost -U postgres -d scrollintel -c "SELECT version();"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ PostgreSQL connection successful!" -ForegroundColor Green
        Write-Host $result -ForegroundColor White
    } else {
        Write-Host "Connection test failed, but container should be running" -ForegroundColor Yellow
        Write-Host "You can test manually with: docker exec -it scrollintel-postgres psql -U postgres" -ForegroundColor Cyan
    }
} catch {
    Write-Host "psql client not available, but PostgreSQL container is running" -ForegroundColor Yellow
    Write-Host "You can test with: docker exec -it scrollintel-postgres psql -U postgres" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=== Connection Details ===" -ForegroundColor Green
Write-Host "Host: localhost" -ForegroundColor White
Write-Host "Port: 5432" -ForegroundColor White
Write-Host "Username: postgres" -ForegroundColor White
Write-Host "Password: scrollintel123" -ForegroundColor White
Write-Host "Database: scrollintel" -ForegroundColor White
Write-Host ""
Write-Host "=== Useful Commands ===" -ForegroundColor Green
Write-Host "Stop container: docker stop scrollintel-postgres" -ForegroundColor White
Write-Host "Start container: docker start scrollintel-postgres" -ForegroundColor White
Write-Host "Connect to DB: docker exec -it scrollintel-postgres psql -U postgres" -ForegroundColor White
Write-Host "View logs: docker logs scrollintel-postgres" -ForegroundColor White