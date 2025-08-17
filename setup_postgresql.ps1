#!/usr/bin/env powershell

Write-Host "=== PostgreSQL Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Add PostgreSQL to PATH for this session
$env:PATH += ";C:\Program Files\PostgreSQL\17\bin"

Write-Host "Step 1: Stopping PostgreSQL service..." -ForegroundColor Yellow
Stop-Service -Name "postgresql-x64-17" -Force -ErrorAction SilentlyContinue

Write-Host "Step 2: Starting PostgreSQL in single-user mode to reset password..." -ForegroundColor Yellow

# Create a temporary batch file to reset password
$resetScript = @"
@echo off
echo Starting PostgreSQL password reset...
cd /d "C:\Program Files\PostgreSQL\17\bin"
echo ALTER USER postgres PASSWORD 'scrollintel123'; | postgres --single -D "C:\Program Files\PostgreSQL\17\data" postgres
echo Password reset complete.
pause
"@

$resetScript | Out-File -FilePath "reset_postgres_password.bat" -Encoding ASCII

Write-Host "Step 3: Created password reset script. Please run it as Administrator:" -ForegroundColor Green
Write-Host "  1. Right-click on 'reset_postgres_password.bat'" -ForegroundColor White
Write-Host "  2. Select 'Run as administrator'" -ForegroundColor White
Write-Host "  3. Wait for it to complete" -ForegroundColor White
Write-Host ""

Write-Host "Step 4: After running the reset script, restart PostgreSQL service:" -ForegroundColor Green
Write-Host "  Start-Service -Name 'postgresql-x64-17'" -ForegroundColor White
Write-Host ""

Write-Host "Step 5: Test connection with new password:" -ForegroundColor Green
Write-Host "  `$env:PGPASSWORD='scrollintel123'; psql -h localhost -U postgres -d postgres -c 'SELECT version();'" -ForegroundColor White
Write-Host ""

Write-Host "Alternative: Use pgAdmin or modify pg_hba.conf for trust authentication" -ForegroundColor Cyan