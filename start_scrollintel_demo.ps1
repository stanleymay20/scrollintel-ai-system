# ScrollIntel Demo Launcher (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ScrollIntel Demo Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Start backend server
Write-Host "🚀 Starting ScrollIntel Backend Server..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python simple_server.py
}

# Wait a moment for backend to start
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Test if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Backend server is running!" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Backend may still be starting..." -ForegroundColor Yellow
}

# Open frontend
Write-Host "🎨 Opening ScrollIntel Frontend..." -ForegroundColor Yellow
$frontendPath = Join-Path $PWD "simple_frontend.html"
Start-Process $frontendPath

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "ScrollIntel Demo is now running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Backend API: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "📱 Frontend:    simple_frontend.html (opened in browser)" -ForegroundColor Cyan
Write-Host "📚 API Docs:    http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available API endpoints:" -ForegroundColor White
Write-Host "  GET  /health" -ForegroundColor Gray
Write-Host "  GET  /api/agents" -ForegroundColor Gray
Write-Host "  GET  /api/monitoring/metrics" -ForegroundColor Gray
Write-Host "  POST /api/agents/chat" -ForegroundColor Gray
Write-Host "  GET  /api/dashboard" -ForegroundColor Gray
Write-Host ""

# Wait for user input to stop
Write-Host "Press any key to stop all services..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Stop services
Write-Host ""
Write-Host "🛑 Stopping services..." -ForegroundColor Red
Stop-Job $backendJob -Force
Remove-Job $backendJob -Force

# Kill any remaining Python processes (be careful with this)
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*ScrollIntel*" } | Stop-Process -Force

Write-Host "✓ Services stopped." -ForegroundColor Green
Write-Host "Thank you for using ScrollIntel!" -ForegroundColor Cyan
Read-Host "Press Enter to exit"