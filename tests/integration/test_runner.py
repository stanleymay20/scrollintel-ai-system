"""
Integration Test Runner
Orchestrates execution of all integration tests with reporting
"""
import pytest
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Test suite result data structure"""
    name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    tests: List[TestResult]


class IntegrationTestRunner:
    """Runs and manages integration tests"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.results: List[TestSuiteResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default test configuration"""
        return {
            "test_suites": [
                {
                    "name": "agent_interactions",
                    "path": "tests/integration/test_agent_interactions.py",
                    "timeout": 300,
                    "required": True,
                    "parallel": False
                },
                {
                    "name": "end_to_end_workflows",
                    "path": "tests/integration/test_end_to_end_workflows.py",
                    "timeout": 600,
                    "required": True,
                    "parallel": False
                },
                {
                    "name": "performance",
                    "path": "tests/integration/test_performance.py",
                    "timeout": 900,
                    "required": False,
                    "parallel": False
                },
                {
                    "name": "data_pipelines",
                    "path": "tests/integration/test_data_pipelines.py",
                    "timeout": 450,
                    "required": True,
                    "parallel": True
                },
                {
                    "name": "security_penetration",
                    "path": "tests/integration/test_security_penetration.py",
                    "timeout": 300,
                    "required": True,
                    "parallel": False
                },
                {
                    "name": "smoke_tests",
                    "path": "tests/integration/test_smoke_tests.py",
                    "timeout": 120,
                    "required": True,
                    "parallel": True
                }
            ],
            "reporting": {
                "formats": ["json", "html", "junit"],
                "output_dir": "test_reports",
                "include_coverage": True
            },
            "thresholds": {
                "success_rate": 0.95,
                "performance_threshold": 2.0,
                "coverage_threshold": 0.80
            },
            "environment": {
                "setup_commands": [
                    "docker-compose -f docker-compose.test.yml up -d postgres redis",
                    "sleep 10"  # Wait for services to start
                ],
                "teardown_commands": [
                    "docker-compose -f docker-compose.test.yml down"
                ]
            }
        }
    
    async def setup_environment(self) -> bool:
        """Setup test environment"""
        print("Setting up test environment...")
        
        setup_commands = self.config.get("environment", {}).get("setup_commands", [])
        
        for command in setup_commands:
            print(f"Running: {command}")
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    print(f"Setup command failed: {command}")
                    print(f"Error: {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"Setup command timed out: {command}")
                return False
            except Exception as e:
                print(f"Setup command error: {command} - {e}")
                return False
        
        print("Environment setup completed successfully")
        return True
    
    async def teardown_environment(self) -> None:
        """Teardown test environment"""
        print("Tearing down test environment...")
        
        teardown_commands = self.config.get("environment", {}).get("teardown_commands", [])
        
        for command in teardown_commands:
            print(f"Running: {command}")
            try:
                subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            except Exception as e:
                print(f"Teardown command error: {command} - {e}")
        
        print("Environment teardown completed")
    
    def run_test_suite(self, suite_config: Dict[str, Any]) -> TestSuiteResult:
        """Run a single test suite"""
        suite_name = suite_config["name"]
        suite_path = suite_config["path"]
        timeout = suite_config.get("timeout", 300)
        
        print(f"Running test suite: {suite_name}")
        print(f"Path: {suite_path}")
        
        # Build pytest command
        pytest_args = [
            "pytest",
            suite_path,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=test_reports/{suite_name}_report.json"
        ]
        
        # Add coverage if enabled
        if self.config.get("reporting", {}).get("include_coverage", False):
            pytest_args.extend([
                "--cov=scrollintel",
                f"--cov-report=html:test_reports/{suite_name}_coverage"
            ])
        
        # Add parallel execution if enabled
        if suite_config.get("parallel", False):
            pytest_args.extend(["-n", "auto"])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                pytest_args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Parse pytest JSON report
            report_file = f"test_reports/{suite_name}_report.json"
            test_results = self._parse_pytest_report(report_file)
            
            # Create suite result
            suite_result = TestSuiteResult(
                name=suite_name,
                total_tests=len(test_results),
                passed=sum(1 for t in test_results if t.status == "passed"),
                failed=sum(1 for t in test_results if t.status == "failed"),
                skipped=sum(1 for t in test_results if t.status == "skipped"),
                errors=sum(1 for t in test_results if t.status == "error"),
                duration=duration,
                tests=test_results
            )
            
            print(f"Suite {suite_name} completed:")
            print(f"  Total: {suite_result.total_tests}")
            print(f"  Passed: {suite_result.passed}")
            print(f"  Failed: {suite_result.failed}")
            print(f"  Duration: {duration:.2f}s")
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            print(f"Test suite {suite_name} timed out after {timeout}s")
            return TestSuiteResult(
                name=suite_name,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=0,
                duration=timeout,
                tests=[TestResult(
                    name=f"{suite_name}_timeout",
                    status="error",
                    duration=timeout,
                    error_message="Test suite timed out"
                )]
            )
        
        except Exception as e:
            print(f"Error running test suite {suite_name}: {e}")
            return TestSuiteResult(
                name=suite_name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=time.time() - start_time,
                tests=[TestResult(
                    name=f"{suite_name}_error",
                    status="error",
                    duration=0,
                    error_message=str(e)
                )]
            )
    
    def _parse_pytest_report(self, report_file: str) -> List[TestResult]:
        """Parse pytest JSON report"""
        try:
            if not os.path.exists(report_file):
                return []
            
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            test_results = []
            
            for test in report_data.get("tests", []):
                result = TestResult(
                    name=test.get("nodeid", "unknown"),
                    status=test.get("outcome", "unknown"),
                    duration=test.get("duration", 0),
                    error_message=test.get("call", {}).get("longrepr", None) if test.get("outcome") == "failed" else None
                )
                test_results.append(result)
            
            return test_results
            
        except Exception as e:
            print(f"Error parsing pytest report {report_file}: {e}")
            return []
    
    async def run_all_tests(self) -> bool:
        """Run all test suites"""
        self.start_time = datetime.now()
        
        # Setup environment
        if not await self.setup_environment():
            print("Failed to setup test environment")
            return False
        
        try:
            # Create output directory
            output_dir = self.config.get("reporting", {}).get("output_dir", "test_reports")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run test suites
            for suite_config in self.config["test_suites"]:
                suite_result = self.run_test_suite(suite_config)
                self.results.append(suite_result)
                
                # Check if required suite failed
                if suite_config.get("required", False) and suite_result.failed > 0:
                    print(f"Required test suite {suite_result.name} failed, stopping execution")
                    break
            
            self.end_time = datetime.now()
            
            # Generate reports
            await self.generate_reports()
            
            # Check thresholds
            return self._check_thresholds()
            
        finally:
            # Always teardown environment
            await self.teardown_environment()
    
    async def generate_reports(self) -> None:
        """Generate test reports"""
        output_dir = self.config.get("reporting", {}).get("output_dir", "test_reports")
        formats = self.config.get("reporting", {}).get("formats", ["json"])
        
        # Calculate overall statistics
        total_tests = sum(suite.total_tests for suite in self.results)
        total_passed = sum(suite.passed for suite in self.results)
        total_failed = sum(suite.failed for suite in self.results)
        total_skipped = sum(suite.skipped for suite in self.results)
        total_errors = sum(suite.errors for suite in self.results)
        total_duration = sum(suite.duration for suite in self.results)
        
        overall_result = {
            "timestamp": self.start_time.isoformat() if self.start_time else None,
            "duration": total_duration,
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "errors": total_errors,
                "success_rate": total_passed / total_tests if total_tests > 0 else 0
            },
            "suites": [
                {
                    "name": suite.name,
                    "total_tests": suite.total_tests,
                    "passed": suite.passed,
                    "failed": suite.failed,
                    "skipped": suite.skipped,
                    "errors": suite.errors,
                    "duration": suite.duration,
                    "success_rate": suite.passed / suite.total_tests if suite.total_tests > 0 else 0
                }
                for suite in self.results
            ]
        }
        
        # Generate JSON report
        if "json" in formats:
            json_file = os.path.join(output_dir, "integration_test_report.json")
            with open(json_file, 'w') as f:
                json.dump(overall_result, f, indent=2)
            print(f"JSON report generated: {json_file}")
        
        # Generate HTML report
        if "html" in formats:
            html_file = os.path.join(output_dir, "integration_test_report.html")
            self._generate_html_report(overall_result, html_file)
            print(f"HTML report generated: {html_file}")
        
        # Generate JUnit XML report
        if "junit" in formats:
            junit_file = os.path.join(output_dir, "integration_test_report.xml")
            self._generate_junit_report(overall_result, junit_file)
            print(f"JUnit report generated: {junit_file}")
    
    def _generate_html_report(self, result_data: Dict[str, Any], output_file: str) -> None:
        """Generate HTML test report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ScrollIntel Integration Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
                .error {{ color: darkred; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>ScrollIntel Integration Test Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Timestamp:</strong> {result_data.get('timestamp', 'N/A')}</p>
                <p><strong>Duration:</strong> {result_data.get('duration', 0):.2f} seconds</p>
                <p><strong>Total Tests:</strong> {result_data['summary']['total_tests']}</p>
                <p><strong>Success Rate:</strong> {result_data['summary']['success_rate']:.2%}</p>
                
                <table>
                    <tr>
                        <th>Status</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    <tr class="passed">
                        <td>Passed</td>
                        <td>{result_data['summary']['passed']}</td>
                        <td>{result_data['summary']['passed'] / result_data['summary']['total_tests'] * 100:.1f}%</td>
                    </tr>
                    <tr class="failed">
                        <td>Failed</td>
                        <td>{result_data['summary']['failed']}</td>
                        <td>{result_data['summary']['failed'] / result_data['summary']['total_tests'] * 100:.1f}%</td>
                    </tr>
                    <tr class="skipped">
                        <td>Skipped</td>
                        <td>{result_data['summary']['skipped']}</td>
                        <td>{result_data['summary']['skipped'] / result_data['summary']['total_tests'] * 100:.1f}%</td>
                    </tr>
                    <tr class="error">
                        <td>Errors</td>
                        <td>{result_data['summary']['errors']}</td>
                        <td>{result_data['summary']['errors'] / result_data['summary']['total_tests'] * 100:.1f}%</td>
                    </tr>
                </table>
            </div>
            
            <h2>Test Suites</h2>
        """
        
        for suite in result_data['suites']:
            status_class = "passed" if suite['failed'] == 0 and suite['errors'] == 0 else "failed"
            html_content += f"""
            <div class="suite {status_class}">
                <h3>{suite['name']}</h3>
                <p><strong>Duration:</strong> {suite['duration']:.2f}s</p>
                <p><strong>Success Rate:</strong> {suite['success_rate']:.2%}</p>
                <p>
                    <span class="passed">Passed: {suite['passed']}</span> |
                    <span class="failed">Failed: {suite['failed']}</span> |
                    <span class="skipped">Skipped: {suite['skipped']}</span> |
                    <span class="error">Errors: {suite['errors']}</span>
                </p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_junit_report(self, result_data: Dict[str, Any], output_file: str) -> None:
        """Generate JUnit XML test report"""
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        
        testsuites = Element('testsuites')
        testsuites.set('name', 'ScrollIntel Integration Tests')
        testsuites.set('tests', str(result_data['summary']['total_tests']))
        testsuites.set('failures', str(result_data['summary']['failed']))
        testsuites.set('errors', str(result_data['summary']['errors']))
        testsuites.set('time', str(result_data.get('duration', 0)))
        
        for suite_data in result_data['suites']:
            testsuite = SubElement(testsuites, 'testsuite')
            testsuite.set('name', suite_data['name'])
            testsuite.set('tests', str(suite_data['total_tests']))
            testsuite.set('failures', str(suite_data['failed']))
            testsuite.set('errors', str(suite_data['errors']))
            testsuite.set('time', str(suite_data['duration']))
            
            # Add placeholder test cases (would need actual test data for full implementation)
            for i in range(suite_data['total_tests']):
                testcase = SubElement(testsuite, 'testcase')
                testcase.set('name', f"test_{i}")
                testcase.set('classname', suite_data['name'])
                testcase.set('time', str(suite_data['duration'] / suite_data['total_tests']))
        
        # Pretty print XML
        rough_string = tostring(testsuites, 'unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(output_file, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))
    
    def _check_thresholds(self) -> bool:
        """Check if test results meet configured thresholds"""
        thresholds = self.config.get("thresholds", {})
        
        # Calculate overall success rate
        total_tests = sum(suite.total_tests for suite in self.results)
        total_passed = sum(suite.passed for suite in self.results)
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # Check success rate threshold
        min_success_rate = thresholds.get("success_rate", 0.95)
        if success_rate < min_success_rate:
            print(f"Success rate {success_rate:.2%} below threshold {min_success_rate:.2%}")
            return False
        
        # Check performance threshold
        max_duration = thresholds.get("performance_threshold", 600)
        total_duration = sum(suite.duration for suite in self.results)
        if total_duration > max_duration:
            print(f"Total duration {total_duration:.2f}s exceeds threshold {max_duration}s")
            return False
        
        print(f"All thresholds met:")
        print(f"  Success rate: {success_rate:.2%} (>= {min_success_rate:.2%})")
        print(f"  Duration: {total_duration:.2f}s (<= {max_duration}s)")
        
        return True


async def main():
    """Main entry point for test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ScrollIntel Integration Test Runner")
    parser.add_argument("--config", help="Path to test configuration file")
    parser.add_argument("--suite", help="Run specific test suite only")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create test runner
    runner = IntegrationTestRunner(config)
    
    # Override configuration with command line arguments
    if args.output_dir:
        runner.config.setdefault("reporting", {})["output_dir"] = args.output_dir
    
    if args.suite:
        # Filter to specific suite
        runner.config["test_suites"] = [
            suite for suite in runner.config["test_suites"]
            if suite["name"] == args.suite
        ]
    
    if args.parallel:
        # Enable parallel execution for all suites
        for suite in runner.config["test_suites"]:
            suite["parallel"] = True
    
    # Run tests
    success = await runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())