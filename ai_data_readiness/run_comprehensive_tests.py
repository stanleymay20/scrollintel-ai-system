#!/usr/bin/env python3
"""
Comprehensive test runner for AI Data Readiness Platform.

This script runs the complete testing framework including:
- Unit tests for all core components
- Integration tests for end-to-end workflows
- Performance and scalability benchmarks
- Data quality validation with synthetic data
- Bias detection validation with known bias patterns

Usage:
    python run_comprehensive_tests.py [--test-type TYPE] [--config CONFIG_FILE]
    
Test Types:
    - all: Run all tests (default)
    - unit: Run only unit tests
    - integration: Run only integration tests
    - performance: Run only performance tests
    - validation: Run only validation tests
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_data_readiness.tests.test_config import TestConfig
from ai_data_readiness.tests.validation import run_comprehensive_validation


def setup_logging(config: Dict) -> logging.Logger:
    """Set up logging configuration."""
    logging_config = config.get('logging_config', {})
    
    level = getattr(logging, logging_config.get('level', 'INFO'))
    format_str = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(level=level, format=format_str)
    logger = logging.getLogger('ComprehensiveTestRunner')
    
    # Add file handler if specified
    if logging_config.get('log_to_file', False):
        log_file = logging_config.get('log_file', 'test_execution.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)
    
    return logger


def run_unit_tests(config: Dict, logger: logging.Logger) -> Dict:
    """Run unit tests."""
    logger.info("Running unit tests...")
    
    try:
        import pytest
        
        # Run pytest on unit test directories
        unit_test_paths = [
            'ai_data_readiness/tests/unit/',
            'ai_data_readiness/tests/conftest.py'
        ]
        
        # Configure pytest arguments
        pytest_args = [
            '-v',  # Verbose output
            '--tb=short',  # Short traceback format
            '--durations=10',  # Show 10 slowest tests
            '--cov=ai_data_readiness',  # Coverage report
            '--cov-report=html:test_results/coverage_html',
            '--cov-report=json:test_results/coverage.json',
            '--junit-xml=test_results/unit_tests.xml'
        ]
        
        # Add test paths
        pytest_args.extend(unit_test_paths)
        
        # Run pytest
        exit_code = pytest.main(pytest_args)
        
        return {
            'status': 'passed' if exit_code == 0 else 'failed',
            'exit_code': exit_code,
            'test_type': 'unit_tests'
        }
        
    except ImportError:
        logger.error("pytest not installed. Please install with: pip install pytest pytest-cov")
        return {
            'status': 'failed',
            'error': 'pytest not available',
            'test_type': 'unit_tests'
        }
    except Exception as e:
        logger.error(f"Unit tests failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'test_type': 'unit_tests'
        }


def run_integration_tests(config: Dict, logger: logging.Logger) -> Dict:
    """Run integration tests."""
    logger.info("Running integration tests...")
    
    try:
        import pytest
        
        # Run pytest on integration test directories
        integration_test_paths = [
            'ai_data_readiness/tests/integration/',
        ]
        
        # Configure pytest arguments
        pytest_args = [
            '-v',
            '--tb=short',
            '--durations=10',
            '--junit-xml=test_results/integration_tests.xml'
        ]
        
        # Add test paths
        pytest_args.extend(integration_test_paths)
        
        # Run pytest
        exit_code = pytest.main(pytest_args)
        
        return {
            'status': 'passed' if exit_code == 0 else 'failed',
            'exit_code': exit_code,
            'test_type': 'integration_tests'
        }
        
    except Exception as e:
        logger.error(f"Integration tests failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'test_type': 'integration_tests'
        }


def run_performance_tests(config: Dict, logger: logging.Logger) -> Dict:
    """Run performance tests."""
    logger.info("Running performance tests...")
    
    try:
        import pytest
        
        # Run pytest on performance test directories
        performance_test_paths = [
            'ai_data_readiness/tests/performance/',
        ]
        
        # Configure pytest arguments for performance tests
        pytest_args = [
            '-v',
            '--tb=short',
            '--durations=0',  # Show all test durations
            '--benchmark-only',  # Only run benchmark tests if pytest-benchmark is available
            '--junit-xml=test_results/performance_tests.xml'
        ]
        
        # Add test paths
        pytest_args.extend(performance_test_paths)
        
        # Run pytest
        exit_code = pytest.main(pytest_args)
        
        return {
            'status': 'passed' if exit_code == 0 else 'failed',
            'exit_code': exit_code,
            'test_type': 'performance_tests'
        }
        
    except Exception as e:
        logger.error(f"Performance tests failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'test_type': 'performance_tests'
        }


def run_validation_tests(config: Dict, logger: logging.Logger) -> Dict:
    """Run validation tests using synthetic data."""
    logger.info("Running validation tests with synthetic data...")
    
    try:
        # Run comprehensive validation
        validation_report = run_comprehensive_validation(config)
        
        return {
            'status': 'passed' if validation_report.success_rate >= 0.8 else 'failed',
            'success_rate': validation_report.success_rate,
            'total_tests': validation_report.total_tests,
            'passed_tests': validation_report.passed_tests,
            'failed_tests': validation_report.failed_tests,
            'execution_time': validation_report.execution_time,
            'test_type': 'validation_tests'
        }
        
    except Exception as e:
        logger.error(f"Validation tests failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'test_type': 'validation_tests'
        }


def generate_summary_report(results: Dict, config: Dict, logger: logging.Logger) -> str:
    """Generate summary report of all test results."""
    logger.info("Generating summary report...")
    
    # Create output directory
    output_dir = Path(config['output_config']['base_directory'])
    output_dir.mkdir(exist_ok=True)
    
    # Calculate overall statistics
    total_test_suites = len(results)
    passed_suites = sum(1 for r in results.values() if r['status'] == 'passed')
    
    summary = {
        'test_execution_summary': {
            'total_test_suites': total_test_suites,
            'passed_test_suites': passed_suites,
            'failed_test_suites': total_test_suites - passed_suites,
            'overall_success_rate': passed_suites / total_test_suites if total_test_suites > 0 else 0
        },
        'test_suite_results': results,
        'configuration': {
            'dataset_sizes': config.get('dataset_sizes', []),
            'quality_thresholds': config.get('quality_thresholds', {}),
            'performance_thresholds': config.get('performance_thresholds', {})
        }
    }
    
    # Save JSON report
    json_report_path = output_dir / 'comprehensive_test_summary.json'
    with open(json_report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate HTML report
    html_report_path = output_dir / 'comprehensive_test_summary.html'
    html_content = generate_html_summary(summary)
    
    with open(html_report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Summary reports saved:")
    logger.info(f"  JSON: {json_report_path}")
    logger.info(f"  HTML: {html_report_path}")
    
    return str(html_report_path)


def generate_html_summary(summary: Dict) -> str:
    """Generate HTML summary report."""
    test_summary = summary['test_execution_summary']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Data Readiness Platform - Comprehensive Test Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .metric {{ text-align: center; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
            .test-suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .passed {{ border-left: 5px solid #4CAF50; }}
            .failed {{ border-left: 5px solid #f44336; }}
            .details {{ margin-top: 10px; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AI Data Readiness Platform - Comprehensive Test Summary</h1>
            <p>Complete testing framework execution results</p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <h3>{test_summary['total_test_suites']}</h3>
                <p>Test Suites</p>
            </div>
            <div class="metric">
                <h3>{test_summary['passed_test_suites']}</h3>
                <p>Passed</p>
            </div>
            <div class="metric">
                <h3>{test_summary['failed_test_suites']}</h3>
                <p>Failed</p>
            </div>
            <div class="metric">
                <h3>{test_summary['overall_success_rate']:.1%}</h3>
                <p>Success Rate</p>
            </div>
        </div>
        
        <h2>Test Suite Results</h2>
    """
    
    for test_type, result in summary['test_suite_results'].items():
        status_class = result['status']
        
        html += f"""
        <div class="test-suite {status_class}">
            <h3>{test_type.replace('_', ' ').title()}</h3>
            <p><strong>Status:</strong> {result['status'].upper()}</p>
        """
        
        if 'total_tests' in result:
            html += f"<p><strong>Tests:</strong> {result['passed_tests']}/{result['total_tests']} passed</p>"
        
        if 'execution_time' in result:
            html += f"<p><strong>Execution Time:</strong> {result['execution_time']:.2f} seconds</p>"
        
        if 'error' in result:
            html += f"<div class='details'><strong>Error:</strong> {result['error']}</div>"
        
        html += "</div>"
    
    html += """
        </body>
    </html>
    """
    
    return html


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for AI Data Readiness Platform')
    parser.add_argument('--test-type', choices=['all', 'unit', 'integration', 'performance', 'validation'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--config', help='Path to custom configuration file')
    parser.add_argument('--output-dir', help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = TestConfig.get_test_config('default')
    
    # Override output directory if specified
    if args.output_dir:
        config['output_config']['base_directory'] = args.output_dir
    
    # Set up logging
    logger = setup_logging(config)
    
    logger.info(f"Starting comprehensive test execution (type: {args.test_type})")
    logger.info(f"Output directory: {config['output_config']['base_directory']}")
    
    # Create output directory
    output_dir = Path(config['output_config']['base_directory'])
    output_dir.mkdir(exist_ok=True)
    
    # Run tests based on type
    results = {}
    
    if args.test_type in ['all', 'unit']:
        results['unit_tests'] = run_unit_tests(config, logger)
    
    if args.test_type in ['all', 'integration']:
        results['integration_tests'] = run_integration_tests(config, logger)
    
    if args.test_type in ['all', 'performance']:
        results['performance_tests'] = run_performance_tests(config, logger)
    
    if args.test_type in ['all', 'validation']:
        results['validation_tests'] = run_validation_tests(config, logger)
    
    # Generate summary report
    summary_report_path = generate_summary_report(results, config, logger)
    
    # Print summary
    total_suites = len(results)
    passed_suites = sum(1 for r in results.values() if r['status'] == 'passed')
    success_rate = passed_suites / total_suites if total_suites > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Test Suites Run: {total_suites}")
    print(f"Passed: {passed_suites}")
    print(f"Failed: {total_suites - passed_suites}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Summary Report: {summary_report_path}")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if success_rate >= 0.8 else 1)


if __name__ == '__main__':
    main()