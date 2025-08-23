@echo off
REM Agent Steering System Integration Test Runner (Windows)
REM Comprehensive integration testing script for enterprise-grade validation

setlocal enabledelayedexpansion

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "TEST_DIR=%PROJECT_ROOT%\tests\integration"
set "RESULTS_DIR=%PROJECT_ROOT%\test_results\integration"
set "CONFIG_FILE=%TEST_DIR%\integration_test_config.json"

REM Create timestamp for log file
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"
set "LOG_FILE=%RESULTS_DIR%\integration_test_%TIMESTAMP%.log"

REM Colors (limited in batch)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function to print status messages
:print_status
echo %~1%~2%NC%
echo %~2 >> "%LOG_FILE%" 2>nul
goto :eof

:print_header
echo.
echo ================================
echo %~1
echo ================================
echo.
echo ================================ >> "%LOG_FILE%" 2>nul
echo %~1 >> "%LOG_FILE%" 2>nul
echo ================================ >> "%LOG_FILE%" 2>nul
echo. >> "%LOG_FILE%" 2>nul
goto :eof

REM Function to check prerequisites
:check_prerequisites
call :print_header "Checking Prerequisites"

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    call :print_status "%RED%" "❌ Python not found in PATH"
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :print_status "%GREEN%" "✅ Python version: %PYTHON_VERSION%"

REM Check required packages
set "REQUIRED_PACKAGES=pytest pytest-asyncio pytest-cov pytest-xdist pytest-timeout"
for %%p in (%REQUIRED_PACKAGES%) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        call :print_status "%YELLOW%" "⚠️ Installing missing package: %%p"
        pip install %%p
        if errorlevel 1 (
            call :print_status "%RED%" "❌ Failed to install %%p"
            exit /b 1
        )
    )
)
call :print_status "%GREEN%" "✅ Required packages available"

REM Check project structure
if not exist "%PROJECT_ROOT%\scrollintel\__init__.py" (
    call :print_status "%RED%" "❌ ScrollIntel package not found"
    exit /b 1
)
call :print_status "%GREEN%" "✅ Project structure OK"

goto :eof

REM Function to setup test environment
:setup_environment
call :print_header "Setting Up Test Environment"

REM Create directories
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%PROJECT_ROOT%\test_logs" mkdir "%PROJECT_ROOT%\test_logs"
if not exist "%PROJECT_ROOT%\test_data" mkdir "%PROJECT_ROOT%\test_data"

REM Set environment variables
set "SCROLLINTEL_ENV=test"
set "SCROLLINTEL_LOG_LEVEL=DEBUG"
set "SCROLLINTEL_TEST_MODE=integration"
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

call :print_status "%GREEN%" "✅ Test environment configured"

REM Generate test data if needed
if not exist "%PROJECT_ROOT%\test_data\customers.csv" (
    call :print_status "%YELLOW%" "📊 Generating test data..."
    python -c "import pandas as pd; import numpy as np; from pathlib import Path; customers = pd.DataFrame({'customer_id': range(1, 10001), 'name': [f'Customer_{i}' for i in range(1, 10001)], 'email': [f'customer{i}@company.com' for i in range(1, 10001)], 'signup_date': pd.date_range('2020-01-01', periods=10000), 'total_spent': np.random.uniform(100, 50000, 10000), 'segment': np.random.choice(['premium', 'standard', 'basic'], 10000)}); Path('%PROJECT_ROOT%/test_data').mkdir(exist_ok=True); customers.to_csv('%PROJECT_ROOT%/test_data/customers.csv', index=False); print('Test data generated successfully')"
    call :print_status "%GREEN%" "✅ Test data generated"
)

goto :eof

REM Function to run specific test suite
:run_test_suite
set "SUITE_NAME=%~1"
set "SUITE_FILE=%TEST_DIR%\test_%SUITE_NAME%.py"

if not exist "%SUITE_FILE%" (
    call :print_status "%RED%" "❌ Test suite not found: %SUITE_FILE%"
    exit /b 1
)

call :print_header "Running Test Suite: %SUITE_NAME%"

REM Get start time
for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set "START_TIME=%%i"

set "SUITE_RESULTS_DIR=%RESULTS_DIR%\%SUITE_NAME%"
if not exist "%SUITE_RESULTS_DIR%" mkdir "%SUITE_RESULTS_DIR%"

REM Build pytest command
set "PYTEST_CMD=python -m pytest "%SUITE_FILE%" -v --tb=short --asyncio-mode=auto --timeout=600"
set "PYTEST_CMD=%PYTEST_CMD% --json-report --json-report-file="%SUITE_RESULTS_DIR%\report.json""
set "PYTEST_CMD=%PYTEST_CMD% --html="%SUITE_RESULTS_DIR%\report.html" --self-contained-html"
set "PYTEST_CMD=%PYTEST_CMD% --cov=scrollintel --cov-report=html:"%SUITE_RESULTS_DIR%\coverage""
set "PYTEST_CMD=%PYTEST_CMD% --cov-report=json:"%SUITE_RESULTS_DIR%\coverage.json""
set "PYTEST_CMD=%PYTEST_CMD% --junit-xml="%SUITE_RESULTS_DIR%\junit.xml""

REM Add parallel execution for supported suites
echo %SUITE_NAME% | findstr /r "agent_interactions data_pipelines smoke_tests" >nul
if not errorlevel 1 (
    set "PYTEST_CMD=%PYTEST_CMD% -n auto"
)

REM Run the test
echo Running: %PYTEST_CMD%
%PYTEST_CMD% 2>&1 | tee -a "%LOG_FILE%"
set "TEST_EXIT_CODE=%ERRORLEVEL%"

REM Get end time and calculate duration
for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set "END_TIME=%%i"
set /a "DURATION=%END_TIME%-%START_TIME%"

if %TEST_EXIT_CODE% equ 0 (
    call :print_status "%GREEN%" "✅ %SUITE_NAME% completed successfully in %DURATION%s"
    exit /b 0
) else (
    call :print_status "%RED%" "❌ %SUITE_NAME% failed after %DURATION%s"
    exit /b 1
)

REM Function to run all test suites
:run_all_tests
call :print_header "Running All Integration Test Suites"

set "TEST_SUITES=enterprise_connectors end_to_end_workflows performance security_penetration agent_interactions data_pipelines ci_cd_pipeline smoke_tests"
set "TOTAL_SUITES=0"
set "PASSED_SUITES=0"
set "FAILED_SUITES=0"

REM Count total suites
for %%s in (%TEST_SUITES%) do set /a "TOTAL_SUITES+=1"

REM Get overall start time
for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set "OVERALL_START_TIME=%%i"

REM Run each suite
for %%s in (%TEST_SUITES%) do (
    call :run_test_suite %%s
    if !errorlevel! equ 0 (
        set /a "PASSED_SUITES+=1"
        set "SUITE_RESULTS=!SUITE_RESULTS! %%s:PASSED"
    ) else (
        set /a "FAILED_SUITES+=1"
        set "SUITE_RESULTS=!SUITE_RESULTS! %%s:FAILED"
    )
)

REM Get overall end time and calculate duration
for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set "OVERALL_END_TIME=%%i"
set /a "TOTAL_DURATION=%OVERALL_END_TIME%-%OVERALL_START_TIME%"

REM Generate summary report
call :generate_summary_report %TOTAL_SUITES% %PASSED_SUITES% %FAILED_SUITES% %TOTAL_DURATION%

REM Return success if all tests passed
if %FAILED_SUITES% equ 0 (
    call :print_status "%GREEN%" "🎉 All integration tests passed!"
    exit /b 0
) else (
    call :print_status "%RED%" "💥 %FAILED_SUITES% out of %TOTAL_SUITES% test suites failed"
    exit /b 1
)

REM Function to generate summary report
:generate_summary_report
set "TOTAL=%~1"
set "PASSED=%~2"
set "FAILED=%~3"
set "DURATION=%~4"

call :print_header "Integration Test Summary"

set /a "SUCCESS_RATE=%PASSED%*100/%TOTAL%"

call :print_status "%BLUE%" "📊 Test Execution Summary:"
call :print_status "" "   Total Suites: %TOTAL%"
call :print_status "%GREEN%" "   Passed: %PASSED%"
call :print_status "%RED%" "   Failed: %FAILED%"
call :print_status "" "   Success Rate: %SUCCESS_RATE%%%"
call :print_status "" "   Total Duration: %DURATION%s"

for /f "tokens=1-6 delims=/: " %%a in ("%date% %time%") do set "CURRENT_TIMESTAMP=%%c-%%a-%%b %%d:%%e:%%f"
call :print_status "" "   Timestamp: %CURRENT_TIMESTAMP%"

echo.
call :print_status "%BLUE%" "Suite Results:"
for %%r in (%SUITE_RESULTS%) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%r") do (
        if "%%b"=="PASSED" (
            call :print_status "%GREEN%" "   ✅ %%a"
        ) else (
            call :print_status "%RED%" "   ❌ %%a"
        )
    )
)

REM Generate JSON summary
set "SUMMARY_FILE=%RESULTS_DIR%\summary.json"
(
echo {
echo     "timestamp": "%CURRENT_TIMESTAMP%",
echo     "total_suites": %TOTAL%,
echo     "passed_suites": %PASSED%,
echo     "failed_suites": %FAILED%,
echo     "success_rate": %SUCCESS_RATE%,
echo     "duration_seconds": %DURATION%,
echo     "suite_results": {
for %%r in (%SUITE_RESULTS%) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%r") do (
        echo         "%%a": "%%b",
    )
)
echo     },
echo     "artifacts": {
echo         "log_file": "%LOG_FILE%",
echo         "results_directory": "%RESULTS_DIR%",
echo         "coverage_reports": "%RESULTS_DIR%/*/coverage",
echo         "html_reports": "%RESULTS_DIR%/*/report.html"
echo     }
echo }
) > "%SUMMARY_FILE%"

call :print_status "%BLUE%" "📄 Summary report saved to: %SUMMARY_FILE%"

goto :eof

REM Function to cleanup test environment
:cleanup_environment
call :print_header "Cleaning Up Test Environment"

REM Kill any remaining test processes
taskkill /f /im python.exe /fi "WINDOWTITLE eq *pytest*" >nul 2>&1

REM Clean up temporary files older than 7 days
forfiles /p "%PROJECT_ROOT%\test_logs" /s /m *.log /d -7 /c "cmd /c del @path" >nul 2>&1

REM Archive old test results
if exist "%RESULTS_DIR%" (
    for /f "tokens=1-3 delims=/" %%a in ("%date%") do set "ARCHIVE_DIR=%PROJECT_ROOT%\test_archives\%%c%%a"
    if not exist "!ARCHIVE_DIR!" mkdir "!ARCHIVE_DIR!"
    
    REM Move results older than 30 days to archive (simplified for batch)
    REM This would need PowerShell for proper date comparison
)

call :print_status "%GREEN%" "✅ Cleanup completed"

goto :eof

REM Function to show usage
:show_usage
echo Agent Steering System Integration Test Runner (Windows)
echo.
echo Usage: %~nx0 [OPTIONS] [SUITE_NAME]
echo.
echo Options:
echo   -h, --help              Show this help message
echo   -a, --all               Run all test suites (default)
echo   -s, --suite SUITE       Run specific test suite
echo   -l, --list              List available test suites
echo   --no-cleanup            Skip cleanup after tests
echo   --security-only         Run only security tests
echo   --performance-only      Run only performance tests
echo.
echo Available Test Suites:
echo   enterprise_connectors   - Enterprise system connector tests
echo   end_to_end_workflows    - Complete business workflow tests
echo   performance            - Performance and scalability tests
echo   security_penetration   - Security penetration tests
echo   agent_interactions     - Multi-agent coordination tests
echo   data_pipelines         - Data pipeline integration tests
echo   ci_cd_pipeline         - CI/CD pipeline validation tests
echo   smoke_tests            - Basic smoke tests
echo.
echo Examples:
echo   %~nx0                                    # Run all tests
echo   %~nx0 -s enterprise_connectors          # Run connector tests only
echo   %~nx0 --security-only                   # Run security tests only
goto :eof

REM Main execution
:main
set "RUN_ALL=true"
set "SUITE_NAME="
set "NO_CLEANUP=false"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :start_execution
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
if "%~1"=="-a" set "RUN_ALL=true" & shift & goto :parse_args
if "%~1"=="--all" set "RUN_ALL=true" & shift & goto :parse_args
if "%~1"=="-s" set "SUITE_NAME=%~2" & set "RUN_ALL=false" & shift & shift & goto :parse_args
if "%~1"=="--suite" set "SUITE_NAME=%~2" & set "RUN_ALL=false" & shift & shift & goto :parse_args
if "%~1"=="-l" goto :list_suites
if "%~1"=="--list" goto :list_suites
if "%~1"=="--no-cleanup" set "NO_CLEANUP=true" & shift & goto :parse_args
if "%~1"=="--security-only" set "SUITE_NAME=security_penetration" & set "RUN_ALL=false" & shift & goto :parse_args
if "%~1"=="--performance-only" set "SUITE_NAME=performance" & set "RUN_ALL=false" & shift & goto :parse_args

REM If no recognized option, treat as suite name
if "%RUN_ALL%"=="true" if "%SUITE_NAME%"=="" (
    set "SUITE_NAME=%~1"
    set "RUN_ALL=false"
)
shift
goto :parse_args

:show_help
call :show_usage
exit /b 0

:list_suites
echo Available test suites:
echo   - enterprise_connectors
echo   - end_to_end_workflows
echo   - performance
echo   - security_penetration
echo   - agent_interactions
echo   - data_pipelines
echo   - ci_cd_pipeline
echo   - smoke_tests
exit /b 0

:start_execution
REM Start execution
call :print_header "Agent Steering System Integration Tests"
call :print_status "%BLUE%" "🚀 Starting integration test execution..."

for /f "tokens=1-6 delims=/: " %%a in ("%date% %time%") do set "START_TIMESTAMP=%%c-%%a-%%b %%d:%%e:%%f"
call :print_status "" "   Timestamp: %START_TIMESTAMP%"
call :print_status "" "   Log file: %LOG_FILE%"
call :print_status "" "   Results directory: %RESULTS_DIR%"

REM Check prerequisites
call :check_prerequisites
if errorlevel 1 exit /b 1

REM Setup environment
call :setup_environment
if errorlevel 1 exit /b 1

REM Run tests
set "EXIT_CODE=0"

if "%RUN_ALL%"=="true" (
    call :run_all_tests
    if errorlevel 1 set "EXIT_CODE=1"
) else (
    call :run_test_suite "%SUITE_NAME%"
    if errorlevel 1 set "EXIT_CODE=1"
)

REM Cleanup
if "%NO_CLEANUP%"=="false" (
    call :cleanup_environment
)

REM Final status
if %EXIT_CODE% equ 0 (
    call :print_status "%GREEN%" "🎉 Integration tests completed successfully!"
) else (
    call :print_status "%RED%" "💥 Integration tests failed!"
)

exit /b %EXIT_CODE%

REM Call main function
call :main %*