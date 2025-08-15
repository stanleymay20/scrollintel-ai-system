#!/bin/bash

# ScrollIntel Integration Test Execution Script
# Comprehensive test runner for all integration test suites

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_DIR="tests/integration"
REPORT_DIR="test_reports"
COVERAGE_DIR="htmlcov"
LOG_FILE="$REPORT_DIR/test_execution.log"

# Test suites configuration
declare -A TEST_SUITES=(
    ["agent_interactions"]="test_agent_interactions.py"
    ["end_to_end_workflows"]="test_end_to_end_workflows.py"
    ["performance"]="test_performance.py"
    ["data_pipelines"]="test_data_pipelines.py"
    ["security_penetration"]="test_security_penetration.py"
    ["smoke_tests"]="test_smoke_tests.py"
)

# Default configuration
RUN_ALL=true
PARALLEL=false
VERBOSE=true
COVERAGE=true
DOCKER_MODE=false
CLEANUP=true
TIMEOUT=300

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --suite SUITE     Run specific test suite only"
    echo "  -p, --parallel        Enable parallel test execution"
    echo "  -d, --docker          Run tests in Docker containers"
    echo "  -c, --no-coverage     Disable coverage reporting"
    echo "  -v, --no-verbose      Disable verbose output"
    echo "  -t, --timeout SECONDS Set test timeout (default: 300)"
    echo "  --no-cleanup          Don't cleanup test environment"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Available test suites:"
    for suite in "${!TEST_SUITES[@]}"; do
        echo "  - $suite"
    done
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests"
    echo "  $0 -s smoke_tests           # Run only smoke tests"
    echo "  $0 -p -d                    # Run all tests in parallel with Docker"
    echo "  $0 -s performance -t 600    # Run performance tests with 10min timeout"
}

# Function to setup test environment
setup_environment() {
    print_status $BLUE "Setting up test environment..."
    
    # Create directories
    mkdir -p "$REPORT_DIR"
    mkdir -p "$COVERAGE_DIR"
    
    # Initialize log file
    echo "ScrollIntel Integration Test Execution - $(date)" > "$LOG_FILE"
    
    if [ "$DOCKER_MODE" = true ]; then
        print_status $BLUE "Starting Docker services..."
        docker-compose -f docker-compose.test.yml up -d postgres redis >> "$LOG_FILE" 2>&1
        
        # Wait for services to be ready
        print_status $YELLOW "Waiting for services to be ready..."
        sleep 15
        
        # Verify services are running
        if ! docker-compose -f docker-compose.test.yml ps | grep -q "Up"; then
            print_status $RED "Failed to start Docker services"
            exit 1
        fi
    else
        # Check if services are available locally
        print_status $BLUE "Checking local services..."
        
        # Check PostgreSQL
        if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
            print_status $YELLOW "PostgreSQL not available locally, trying Docker..."
            docker-compose -f docker-compose.test.yml up -d postgres >> "$LOG_FILE" 2>&1
            sleep 10
        fi
        
        # Check Redis
        if ! redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
            print_status $YELLOW "Redis not available locally, trying Docker..."
            docker-compose -f docker-compose.test.yml up -d redis >> "$LOG_FILE" 2>&1
            sleep 5
        fi
    fi
    
    # Set environment variables
    export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/scrollintel_test"
    export REDIS_URL="redis://localhost:6379/0"
    export TESTING=true
    export PYTHONPATH="$(pwd)"
    
    print_status $GREEN "Test environment setup completed"
}

# Function to cleanup test environment
cleanup_environment() {
    if [ "$CLEANUP" = true ]; then
        print_status $BLUE "Cleaning up test environment..."
        
        if [ "$DOCKER_MODE" = true ]; then
            docker-compose -f docker-compose.test.yml down -v >> "$LOG_FILE" 2>&1
        fi
        
        print_status $GREEN "Cleanup completed"
    fi
}

# Function to run a single test suite
run_test_suite() {
    local suite_name=$1
    local test_file=$2
    local start_time=$(date +%s)
    
    print_status $BLUE "Running test suite: $suite_name"
    
    # Build pytest command
    local pytest_cmd="python -m pytest $TEST_DIR/$test_file"
    
    # Add common options
    if [ "$VERBOSE" = true ]; then
        pytest_cmd="$pytest_cmd -v"
    fi
    
    pytest_cmd="$pytest_cmd --tb=short"
    pytest_cmd="$pytest_cmd --timeout=$TIMEOUT"
    pytest_cmd="$pytest_cmd --json-report --json-report-file=$REPORT_DIR/${suite_name}_report.json"
    
    # Add coverage options
    if [ "$COVERAGE" = true ]; then
        pytest_cmd="$pytest_cmd --cov=scrollintel --cov-append"
        pytest_cmd="$pytest_cmd --cov-report=html:$COVERAGE_DIR/$suite_name"
    fi
    
    # Add parallel execution
    if [ "$PARALLEL" = true ] && [ "$suite_name" != "performance" ]; then
        pytest_cmd="$pytest_cmd -n auto"
    fi
    
    # Execute test suite
    print_status $YELLOW "Executing: $pytest_cmd"
    
    if eval "$pytest_cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_status $GREEN "✅ $suite_name completed successfully (${duration}s)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_status $RED "❌ $suite_name failed (${duration}s)"
        return 1
    fi
}

# Function to run all test suites
run_all_tests() {
    local total_suites=${#TEST_SUITES[@]}
    local passed_suites=0
    local failed_suites=0
    local start_time=$(date +%s)
    
    print_status $BLUE "Running $total_suites test suites..."
    
    # Run each test suite
    for suite_name in "${!TEST_SUITES[@]}"; do
        local test_file="${TEST_SUITES[$suite_name]}"
        
        if run_test_suite "$suite_name" "$test_file"; then
            ((passed_suites++))
        else
            ((failed_suites++))
            
            # For critical suites, consider stopping
            if [[ "$suite_name" == "smoke_tests" || "$suite_name" == "security_penetration" ]]; then
                print_status $RED "Critical test suite failed: $suite_name"
                print_status $YELLOW "Consider fixing critical issues before continuing"
            fi
        fi
    done
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    # Print summary
    print_status $BLUE "Test Execution Summary:"
    print_status $GREEN "  Passed: $passed_suites/$total_suites"
    print_status $RED "  Failed: $failed_suites/$total_suites"
    print_status $BLUE "  Total Duration: ${total_duration}s"
    
    # Generate final coverage report
    if [ "$COVERAGE" = true ]; then
        print_status $BLUE "Generating coverage report..."
        python -m coverage html -d "$COVERAGE_DIR/combined"
        python -m coverage xml -o "$REPORT_DIR/coverage.xml"
        python -m coverage report --show-missing
    fi
    
    # Generate comprehensive report
    print_status $BLUE "Generating comprehensive test report..."
    python tests/integration/test_runner.py --output-dir "$REPORT_DIR" >> "$LOG_FILE" 2>&1
    
    # Return success if all tests passed
    if [ $failed_suites -eq 0 ]; then
        print_status $GREEN "🎉 All integration tests passed!"
        return 0
    else
        print_status $RED "💥 Some integration tests failed"
        return 1
    fi
}

# Function to run specific test suite
run_specific_suite() {
    local suite_name=$1
    
    if [[ ! -v TEST_SUITES[$suite_name] ]]; then
        print_status $RED "Unknown test suite: $suite_name"
        print_status $YELLOW "Available suites: ${!TEST_SUITES[*]}"
        exit 1
    fi
    
    local test_file="${TEST_SUITES[$suite_name]}"
    
    print_status $BLUE "Running specific test suite: $suite_name"
    
    if run_test_suite "$suite_name" "$test_file"; then
        print_status $GREEN "🎉 Test suite $suite_name passed!"
        return 0
    else
        print_status $RED "💥 Test suite $suite_name failed"
        return 1
    fi
}

# Parse command line arguments
SPECIFIC_SUITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--suite)
            SPECIFIC_SUITE="$2"
            RUN_ALL=false
            shift 2
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -d|--docker)
            DOCKER_MODE=true
            shift
            ;;
        -c|--no-coverage)
            COVERAGE=false
            shift
            ;;
        -v|--no-verbose)
            VERBOSE=false
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_status $RED "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status $BLUE "ScrollIntel Integration Test Runner"
    print_status $BLUE "=================================="
    
    # Setup environment
    setup_environment
    
    # Trap cleanup on exit
    trap cleanup_environment EXIT
    
    # Run tests
    if [ "$RUN_ALL" = true ]; then
        run_all_tests
    else
        run_specific_suite "$SPECIFIC_SUITE"
    fi
}

# Execute main function
main "$@"