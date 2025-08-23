#!/bin/bash

# Agent Steering System Integration Test Runner
# Comprehensive integration testing script for enterprise-grade validation

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_ROOT/tests/integration"
RESULTS_DIR="$PROJECT_ROOT/test_results/integration"
CONFIG_FILE="$TEST_DIR/integration_test_config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="$RESULTS_DIR/integration_test_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}================================${NC}\n" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python version
    if ! python3 --version | grep -q "Python 3\.[8-9]\|Python 3\.1[0-9]"; then
        print_status "$RED" "❌ Python 3.8+ required"
        exit 1
    fi
    print_status "$GREEN" "✅ Python version OK"
    
    # Check required packages
    local required_packages=("pytest" "pytest-asyncio" "pytest-cov" "pytest-xdist" "pytest-timeout")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            print_status "$YELLOW" "⚠️  Installing missing package: $package"
            pip install "$package"
        fi
    done
    print_status "$GREEN" "✅ Required packages available"
    
    # Check Docker (if needed)
    if command -v docker &> /dev/null; then
        print_status "$GREEN" "✅ Docker available"
    else
        print_status "$YELLOW" "⚠️  Docker not available (some tests may be skipped)"
    fi
    
    # Check project structure
    if [[ ! -f "$PROJECT_ROOT/scrollintel/__init__.py" ]]; then
        print_status "$RED" "❌ ScrollIntel package not found"
        exit 1
    fi
    print_status "$GREEN" "✅ Project structure OK"
}

# Function to setup test environment
setup_environment() {
    print_header "Setting Up Test Environment"
    
    # Create directories
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$PROJECT_ROOT/test_logs"
    mkdir -p "$PROJECT_ROOT/test_data"
    
    # Set environment variables
    export SCROLLINTEL_ENV="test"
    export SCROLLINTEL_LOG_LEVEL="DEBUG"
    export SCROLLINTEL_TEST_MODE="integration"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    print_status "$GREEN" "✅ Test environment configured"
    
    # Generate test data if needed
    if [[ ! -f "$PROJECT_ROOT/test_data/customers.csv" ]]; then
        print_status "$YELLOW" "📊 Generating test data..."
        python3 -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Generate customer data
customers = pd.DataFrame({
    'customer_id': range(1, 10001),
    'name': [f'Customer_{i}' for i in range(1, 10001)],
    'email': [f'customer{i}@company.com' for i in range(1, 10001)],
    'signup_date': pd.date_range('2020-01-01', periods=10000),
    'total_spent': np.random.uniform(100, 50000, 10000),
    'segment': np.random.choice(['premium', 'standard', 'basic'], 10000)
})

Path('$PROJECT_ROOT/test_data').mkdir(exist_ok=True)
customers.to_csv('$PROJECT_ROOT/test_data/customers.csv', index=False)
print('Test data generated successfully')
"
        print_status "$GREEN" "✅ Test data generated"
    fi
}

# Function to run specific test suite
run_test_suite() {
    local suite_name=$1
    local suite_file="$TEST_DIR/test_${suite_name}.py"
    
    if [[ ! -f "$suite_file" ]]; then
        print_status "$RED" "❌ Test suite not found: $suite_file"
        return 1
    fi
    
    print_header "Running Test Suite: $suite_name"
    
    local start_time=$(date +%s)
    local suite_results_dir="$RESULTS_DIR/$suite_name"
    mkdir -p "$suite_results_dir"
    
    # Build pytest command
    local pytest_cmd=(
        python3 -m pytest
        "$suite_file"
        -v
        --tb=short
        --asyncio-mode=auto
        --timeout=600
        --json-report
        "--json-report-file=$suite_results_dir/report.json"
        "--html=$suite_results_dir/report.html"
        "--self-contained-html"
        "--cov=scrollintel"
        "--cov-report=html:$suite_results_dir/coverage"
        "--cov-report=json:$suite_results_dir/coverage.json"
        "--junit-xml=$suite_results_dir/junit.xml"
    )
    
    # Add parallel execution for supported suites
    if [[ "$suite_name" =~ ^(agent_interactions|data_pipelines|smoke_tests)$ ]]; then
        pytest_cmd+=("-n" "auto")
    fi
    
    # Run the test
    if "${pytest_cmd[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_status "$GREEN" "✅ $suite_name completed successfully in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_status "$RED" "❌ $suite_name failed after ${duration}s"
        return 1
    fi
}

# Function to run all test suites
run_all_tests() {
    print_header "Running All Integration Test Suites"
    
    local test_suites=(
        "enterprise_connectors"
        "end_to_end_workflows"
        "performance"
        "security_penetration"
        "agent_interactions"
        "data_pipelines"
        "ci_cd_pipeline"
        "smoke_tests"
    )
    
    local total_suites=${#test_suites[@]}
    local passed_suites=0
    local failed_suites=0
    local suite_results=()
    
    local overall_start_time=$(date +%s)
    
    for suite in "${test_suites[@]}"; do
        if run_test_suite "$suite"; then
            ((passed_suites++))
            suite_results+=("$suite:PASSED")
        else
            ((failed_suites++))
            suite_results+=("$suite:FAILED")
        fi
    done
    
    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))
    
    # Generate summary report
    generate_summary_report "$total_suites" "$passed_suites" "$failed_suites" "$total_duration" "${suite_results[@]}"
    
    # Return success if all critical tests passed
    if [[ $failed_suites -eq 0 ]]; then
        print_status "$GREEN" "🎉 All integration tests passed!"
        return 0
    else
        print_status "$RED" "💥 $failed_suites out of $total_suites test suites failed"
        return 1
    fi
}

# Function to generate summary report
generate_summary_report() {
    local total=$1
    local passed=$2
    local failed=$3
    local duration=$4
    shift 4
    local results=("$@")
    
    print_header "Integration Test Summary"
    
    local success_rate=$((passed * 100 / total))
    
    print_status "$BLUE" "📊 Test Execution Summary:"
    print_status "$NC" "   Total Suites: $total"
    print_status "$GREEN" "   Passed: $passed"
    print_status "$RED" "   Failed: $failed"
    print_status "$NC" "   Success Rate: $success_rate%"
    print_status "$NC" "   Total Duration: ${duration}s"
    print_status "$NC" "   Timestamp: $(date)"
    
    echo -e "\n${BLUE}Suite Results:${NC}" | tee -a "$LOG_FILE"
    for result in "${results[@]}"; do
        local suite_name="${result%:*}"
        local suite_status="${result#*:}"
        if [[ "$suite_status" == "PASSED" ]]; then
            print_status "$GREEN" "   ✅ $suite_name"
        else
            print_status "$RED" "   ❌ $suite_name"
        fi
    done
    
    # Generate JSON summary
    local summary_file="$RESULTS_DIR/summary.json"
    cat > "$summary_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "total_suites": $total,
    "passed_suites": $passed,
    "failed_suites": $failed,
    "success_rate": $success_rate,
    "duration_seconds": $duration,
    "suite_results": {
$(for result in "${results[@]}"; do
    local suite_name="${result%:*}"
    local suite_status="${result#*:}"
    echo "        \"$suite_name\": \"$suite_status\","
done | sed '$ s/,$//')
    },
    "artifacts": {
        "log_file": "$LOG_FILE",
        "results_directory": "$RESULTS_DIR",
        "coverage_reports": "$RESULTS_DIR/*/coverage",
        "html_reports": "$RESULTS_DIR/*/report.html"
    }
}
EOF
    
    print_status "$BLUE" "📄 Summary report saved to: $summary_file"
}

# Function to cleanup test environment
cleanup_environment() {
    print_header "Cleaning Up Test Environment"
    
    # Kill any remaining test processes
    pkill -f "pytest.*integration" || true
    
    # Clean up temporary files older than 7 days
    find "$PROJECT_ROOT/test_logs" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Archive old test results
    if [[ -d "$RESULTS_DIR" ]]; then
        local archive_dir="$PROJECT_ROOT/test_archives/$(date +%Y%m)"
        mkdir -p "$archive_dir"
        
        # Move results older than 30 days to archive
        find "$RESULTS_DIR" -maxdepth 1 -type d -mtime +30 -exec mv {} "$archive_dir/" \; 2>/dev/null || true
    fi
    
    print_status "$GREEN" "✅ Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Agent Steering System Integration Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] [SUITE_NAME]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -a, --all               Run all test suites (default)"
    echo "  -s, --suite SUITE       Run specific test suite"
    echo "  -l, --list              List available test suites"
    echo "  -c, --config FILE       Use custom configuration file"
    echo "  -o, --output DIR        Set output directory"
    echo "  --no-cleanup            Skip cleanup after tests"
    echo "  --parallel              Enable parallel execution where supported"
    echo "  --coverage-only         Run only coverage analysis"
    echo "  --security-only         Run only security tests"
    echo "  --performance-only      Run only performance tests"
    echo ""
    echo "Available Test Suites:"
    echo "  enterprise_connectors   - Enterprise system connector tests"
    echo "  end_to_end_workflows    - Complete business workflow tests"
    echo "  performance            - Performance and scalability tests"
    echo "  security_penetration   - Security penetration tests"
    echo "  agent_interactions     - Multi-agent coordination tests"
    echo "  data_pipelines         - Data pipeline integration tests"
    echo "  ci_cd_pipeline         - CI/CD pipeline validation tests"
    echo "  smoke_tests            - Basic smoke tests"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests"
    echo "  $0 -s enterprise_connectors          # Run connector tests only"
    echo "  $0 --security-only                   # Run security tests only"
    echo "  $0 -o /tmp/test_results              # Use custom output directory"
}

# Main execution
main() {
    local run_all=true
    local suite_name=""
    local no_cleanup=false
    local parallel=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -a|--all)
                run_all=true
                shift
                ;;
            -s|--suite)
                suite_name="$2"
                run_all=false
                shift 2
                ;;
            -l|--list)
                echo "Available test suites:"
                echo "  - enterprise_connectors"
                echo "  - end_to_end_workflows"
                echo "  - performance"
                echo "  - security_penetration"
                echo "  - agent_interactions"
                echo "  - data_pipelines"
                echo "  - ci_cd_pipeline"
                echo "  - smoke_tests"
                exit 0
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -o|--output)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --no-cleanup)
                no_cleanup=true
                shift
                ;;
            --parallel)
                parallel=true
                shift
                ;;
            --coverage-only)
                suite_name="coverage_analysis"
                run_all=false
                shift
                ;;
            --security-only)
                suite_name="security_penetration"
                run_all=false
                shift
                ;;
            --performance-only)
                suite_name="performance"
                run_all=false
                shift
                ;;
            *)
                if [[ -z "$suite_name" && "$run_all" == false ]]; then
                    suite_name="$1"
                else
                    echo "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Start execution
    print_header "Agent Steering System Integration Tests"
    print_status "$BLUE" "🚀 Starting integration test execution..."
    print_status "$NC" "   Timestamp: $(date)"
    print_status "$NC" "   Log file: $LOG_FILE"
    print_status "$NC" "   Results directory: $RESULTS_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup environment
    setup_environment
    
    # Run tests
    local exit_code=0
    
    if [[ "$run_all" == true ]]; then
        if ! run_all_tests; then
            exit_code=1
        fi
    else
        if ! run_test_suite "$suite_name"; then
            exit_code=1
        fi
    fi
    
    # Cleanup
    if [[ "$no_cleanup" != true ]]; then
        cleanup_environment
    fi
    
    # Final status
    if [[ $exit_code -eq 0 ]]; then
        print_status "$GREEN" "🎉 Integration tests completed successfully!"
    else
        print_status "$RED" "💥 Integration tests failed!"
    fi
    
    exit $exit_code
}

# Trap to ensure cleanup on exit
trap 'cleanup_environment' EXIT

# Run main function
main "$@"