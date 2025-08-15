"""
Bulletproof Testing Framework Runner

This module provides a comprehensive test runner for the bulletproof user experience
testing framework, including test orchestration, reporting, and validation.
"""

import asyncio
import pytest
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .test_chaos_engineering import ChaosEngineeringTestSuite
from .test_user_journey_failures import UserJourneyTestFramework
from .test_performance_degradation import PerformanceTestFramework
from .test_automated_recovery import AutomatedRecoveryTestFramework


@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    test_type: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Test suite result data structure."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    duration: float
    success_rate: float
    results: List[TestResult]
    timestamp: str


class BulletproofTestRunner:
    """Comprehensive test runner for bulletproof system validation."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_results = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for test runner."""
        logger = logging.getLogger("bulletproof_test_runner")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / "test_runner.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    async def run_chaos_engineering_tests(self) -> TestSuiteResult:
        """Run chaos engineering test suite."""
        self.logger.info("Starting chaos engineering tests...")
        start_time = time.time()
        
        chaos_suite = ChaosEngineeringTestSuite()
        test_results = []
        
        # Define chaos engineering tests
        chaos_tests = [
            ("network_failure_resilience", chaos_suite.inject_network_failure),
            ("database_failure_recovery", chaos_suite.inject_database_failure),
            ("memory_pressure_degradation", chaos_suite.inject_memory_pressure),
            ("cpu_stress_performance", chaos_suite.inject_cpu_stress),
        ]
        
        for test_name, test_func in chaos_tests:
            test_start = time.time()
            try:
                await test_func()
                success = True
                error_message = None
                self.logger.info(f"✓ Chaos test '{test_name}' passed")
            except Exception as e:
                success = False
                error_message = str(e)
                self.logger.error(f"✗ Chaos test '{test_name}' failed: {error_message}")
                
            test_duration = time.time() - test_start
            test_results.append(TestResult(
                test_name=test_name,
                test_type="chaos_engineering",
                success=success,
                duration=test_duration,
                error_message=error_message,
                timestamp=datetime.now().isoformat()
            ))
            
        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in test_results if r.success)
        
        suite_result = TestSuiteResult(
            suite_name="chaos_engineering",
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            duration=total_duration,
            success_rate=passed_tests / len(test_results) if test_results else 0,
            results=test_results,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Chaos engineering tests completed: {passed_tests}/{len(test_results)} passed")
        return suite_result
        
    async def run_user_journey_tests(self) -> TestSuiteResult:
        """Run user journey test suite."""
        self.logger.info("Starting user journey tests...")
        start_time = time.time()
        
        journey_framework = UserJourneyTestFramework()
        test_results = []
        
        # Define user journey tests
        journey_tests = [
            ("login_journey_auth_failure", lambda: journey_framework.simulate_user_login_journey(
                "test_user_1", ["validate_credentials"]
            )),
            ("data_analysis_ai_failure", lambda: journey_framework.simulate_data_analysis_journey(
                "test_analyst_1", ["generate_insights"]
            )),
            ("collaboration_sync_failure", lambda: journey_framework.simulate_collaboration_journey(
                ["user_1", "user_2", "user_3"], ["sync_initial_state"]
            )),
        ]
        
        for test_name, test_func in journey_tests:
            test_start = time.time()
            try:
                result = await test_func()
                # Validate journey results
                success = self._validate_journey_result(result)
                error_message = None if success else "Journey validation failed"
                self.logger.info(f"✓ Journey test '{test_name}' passed" if success else f"✗ Journey test '{test_name}' failed")
            except Exception as e:
                success = False
                error_message = str(e)
                self.logger.error(f"✗ Journey test '{test_name}' failed: {error_message}")
                
            test_duration = time.time() - test_start
            test_results.append(TestResult(
                test_name=test_name,
                test_type="user_journey",
                success=success,
                duration=test_duration,
                error_message=error_message,
                timestamp=datetime.now().isoformat()
            ))
            
        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in test_results if r.success)
        
        suite_result = TestSuiteResult(
            suite_name="user_journey",
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            duration=total_duration,
            success_rate=passed_tests / len(test_results) if test_results else 0,
            results=test_results,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"User journey tests completed: {passed_tests}/{len(test_results)} passed")
        return suite_result
        
    async def run_performance_tests(self) -> TestSuiteResult:
        """Run performance degradation test suite."""
        self.logger.info("Starting performance tests...")
        start_time = time.time()
        
        performance_framework = PerformanceTestFramework()
        test_results = []
        
        # Define performance tests
        performance_tests = [
            ("baseline_performance", self._test_baseline_performance),
            ("high_load_performance", lambda: performance_framework.simulate_high_load(30, 15)),
            ("memory_pressure_performance", lambda: performance_framework.simulate_memory_pressure("high")),
            ("degradation_levels", performance_framework.test_degradation_levels),
        ]
        
        for test_name, test_func in performance_tests:
            test_start = time.time()
            try:
                result = await test_func()
                success = self._validate_performance_result(test_name, result)
                error_message = None if success else "Performance validation failed"
                self.logger.info(f"✓ Performance test '{test_name}' passed" if success else f"✗ Performance test '{test_name}' failed")
            except Exception as e:
                success = False
                error_message = str(e)
                self.logger.error(f"✗ Performance test '{test_name}' failed: {error_message}")
                
            test_duration = time.time() - test_start
            test_results.append(TestResult(
                test_name=test_name,
                test_type="performance",
                success=success,
                duration=test_duration,
                error_message=error_message,
                timestamp=datetime.now().isoformat()
            ))
            
        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in test_results if r.success)
        
        suite_result = TestSuiteResult(
            suite_name="performance",
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            duration=total_duration,
            success_rate=passed_tests / len(test_results) if test_results else 0,
            results=test_results,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Performance tests completed: {passed_tests}/{len(test_results)} passed")
        return suite_result
        
    async def run_recovery_tests(self) -> TestSuiteResult:
        """Run automated recovery test suite."""
        self.logger.info("Starting recovery tests...")
        start_time = time.time()
        
        recovery_framework = AutomatedRecoveryTestFramework()
        test_results = []
        
        # Define recovery tests
        recovery_tests = [
            ("database_connection_recovery", lambda: self._test_service_recovery(
                recovery_framework, "database", "connection_refused"
            )),
            ("api_service_recovery", lambda: self._test_service_recovery(
                recovery_framework, "api_gateway", "timeout"
            )),
            ("data_corruption_recovery", lambda: recovery_framework.test_data_recovery_integrity(
                "checksum_mismatch"
            )),
            ("predictive_prevention", lambda: recovery_framework.test_predictive_recovery(
                "high_memory_usage"
            )),
        ]
        
        for test_name, test_func in recovery_tests:
            test_start = time.time()
            try:
                result = await test_func()
                success = self._validate_recovery_result(test_name, result)
                error_message = None if success else "Recovery validation failed"
                self.logger.info(f"✓ Recovery test '{test_name}' passed" if success else f"✗ Recovery test '{test_name}' failed")
            except Exception as e:
                success = False
                error_message = str(e)
                self.logger.error(f"✗ Recovery test '{test_name}' failed: {error_message}")
                
            test_duration = time.time() - test_start
            test_results.append(TestResult(
                test_name=test_name,
                test_type="recovery",
                success=success,
                duration=test_duration,
                error_message=error_message,
                timestamp=datetime.now().isoformat()
            ))
            
        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in test_results if r.success)
        
        suite_result = TestSuiteResult(
            suite_name="recovery",
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            duration=total_duration,
            success_rate=passed_tests / len(test_results) if test_results else 0,
            results=test_results,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Recovery tests completed: {passed_tests}/{len(test_results)} passed")
        return suite_result
        
    async def run_all_tests(self) -> Dict[str, TestSuiteResult]:
        """Run all bulletproof test suites."""
        self.logger.info("Starting comprehensive bulletproof testing...")
        overall_start = time.time()
        
        # Run all test suites
        test_suites = {
            "chaos_engineering": await self.run_chaos_engineering_tests(),
            "user_journey": await self.run_user_journey_tests(),
            "performance": await self.run_performance_tests(),
            "recovery": await self.run_recovery_tests(),
        }
        
        overall_duration = time.time() - overall_start
        
        # Generate comprehensive report
        await self._generate_comprehensive_report(test_suites, overall_duration)
        
        self.logger.info(f"All bulletproof tests completed in {overall_duration:.2f} seconds")
        return test_suites
        
    def _validate_journey_result(self, result: Any) -> bool:
        """Validate user journey test result."""
        if not result:
            return False
            
        # Check if most steps succeeded or were recovered
        if isinstance(result, list):
            successful_steps = sum(1 for step in result 
                                 if step.get('success', False) or step.get('recovered', False))
            return successful_steps >= len(result) * 0.8  # 80% success rate
            
        return True
        
    def _validate_performance_result(self, test_name: str, result: Any) -> bool:
        """Validate performance test result."""
        if test_name == "baseline_performance":
            return True  # Baseline is just for measurement
        elif test_name == "degradation_levels":
            if isinstance(result, dict):
                # Check that all degradation levels maintain reasonable success rates
                return all(level_data.get('success_rate', 0) >= 0.8 
                          for level_data in result.values())
        return True
        
    def _validate_recovery_result(self, test_name: str, result: Any) -> bool:
        """Validate recovery test result."""
        if isinstance(result, dict):
            if 'success' in result:
                return result['success']
            elif 'data_recovered' in result:
                return result['data_recovered'] and result.get('integrity_verified', False)
            elif 'prevention_successful' in result:
                return result['prevention_successful']
        return True
        
    async def _test_baseline_performance(self) -> Dict[str, Any]:
        """Test baseline performance."""
        framework = PerformanceTestFramework()
        metrics = []
        
        for i in range(10):
            metric = await framework.measure_operation_performance(
                framework.orchestrator.handle_user_action,
                {'action': 'baseline_test', 'user_id': f'baseline_user_{i}'}
            )
            metrics.append(metric)
            
        return {'metrics': metrics, 'baseline_established': True}
        
    async def _test_service_recovery(self, framework: AutomatedRecoveryTestFramework, 
                                   service: str, failure_type: str) -> Dict[str, Any]:
        """Test service recovery."""
        failure_info = await framework.simulate_service_failure(service, failure_type)
        recovery_result = await framework.trigger_recovery_sequence(failure_info)
        return recovery_result
        
    async def _generate_comprehensive_report(self, test_suites: Dict[str, TestSuiteResult], 
                                           overall_duration: float):
        """Generate comprehensive test report."""
        report = {
            "bulletproof_test_report": {
                "timestamp": datetime.now().isoformat(),
                "overall_duration": overall_duration,
                "test_suites": {name: asdict(suite) for name, suite in test_suites.items()},
                "summary": {
                    "total_suites": len(test_suites),
                    "total_tests": sum(suite.total_tests for suite in test_suites.values()),
                    "total_passed": sum(suite.passed_tests for suite in test_suites.values()),
                    "total_failed": sum(suite.failed_tests for suite in test_suites.values()),
                    "overall_success_rate": sum(suite.passed_tests for suite in test_suites.values()) / 
                                          sum(suite.total_tests for suite in test_suites.values()) 
                                          if sum(suite.total_tests for suite in test_suites.values()) > 0 else 0
                }
            }
        }
        
        # Save JSON report
        report_file = self.output_dir / f"bulletproof_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate HTML report
        await self._generate_html_report(report, report_file.with_suffix('.html'))
        
        self.logger.info(f"Test report saved to {report_file}")
        
    async def _generate_html_report(self, report_data: Dict[str, Any], html_file: Path):
        """Generate HTML test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bulletproof User Experience Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .test-suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
                .suite-header {{ background-color: #e0e0e0; padding: 10px; font-weight: bold; }}
                .test-result {{ padding: 10px; border-bottom: 1px solid #eee; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .metrics {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Bulletproof User Experience Test Report</h1>
                <p>Generated: {report_data['bulletproof_test_report']['timestamp']}</p>
                <p>Duration: {report_data['bulletproof_test_report']['overall_duration']:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metrics">
                    <p>Total Test Suites: {report_data['bulletproof_test_report']['summary']['total_suites']}</p>
                    <p>Total Tests: {report_data['bulletproof_test_report']['summary']['total_tests']}</p>
                    <p>Passed: <span class="passed">{report_data['bulletproof_test_report']['summary']['total_passed']}</span></p>
                    <p>Failed: <span class="failed">{report_data['bulletproof_test_report']['summary']['total_failed']}</span></p>
                    <p>Success Rate: {report_data['bulletproof_test_report']['summary']['overall_success_rate']:.2%}</p>
                </div>
            </div>
        """
        
        # Add test suite details
        for suite_name, suite_data in report_data['bulletproof_test_report']['test_suites'].items():
            html_content += f"""
            <div class="test-suite">
                <div class="suite-header">{suite_name.replace('_', ' ').title()} Tests</div>
                <div class="metrics">
                    <p>Tests: {suite_data['total_tests']} | 
                       Passed: <span class="passed">{suite_data['passed_tests']}</span> | 
                       Failed: <span class="failed">{suite_data['failed_tests']}</span> | 
                       Success Rate: {suite_data['success_rate']:.2%}</p>
                    <p>Duration: {suite_data['duration']:.2f} seconds</p>
                </div>
            """
            
            for test_result in suite_data['results']:
                status_class = "passed" if test_result['success'] else "failed"
                status_text = "✓ PASSED" if test_result['success'] else "✗ FAILED"
                error_info = f"<br>Error: {test_result['error_message']}" if test_result['error_message'] else ""
                
                html_content += f"""
                <div class="test-result">
                    <span class="{status_class}">{status_text}</span> 
                    {test_result['test_name']} ({test_result['duration']:.2f}s)
                    {error_info}
                </div>
                """
                
            html_content += "</div>"
            
        html_content += """
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)


async def main():
    """Main entry point for bulletproof test runner."""
    runner = BulletproofTestRunner()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Bulletproof User Experience Test Runner")
    parser.add_argument("--suite", choices=["chaos", "journey", "performance", "recovery", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--output", default="test_results", help="Output directory for test results")
    
    args = parser.parse_args()
    
    runner.output_dir = Path(args.output)
    runner.output_dir.mkdir(exist_ok=True)
    
    try:
        if args.suite == "all":
            results = await runner.run_all_tests()
        elif args.suite == "chaos":
            results = {"chaos_engineering": await runner.run_chaos_engineering_tests()}
        elif args.suite == "journey":
            results = {"user_journey": await runner.run_user_journey_tests()}
        elif args.suite == "performance":
            results = {"performance": await runner.run_performance_tests()}
        elif args.suite == "recovery":
            results = {"recovery": await runner.run_recovery_tests()}
            
        # Print summary
        total_tests = sum(suite.total_tests for suite in results.values())
        total_passed = sum(suite.passed_tests for suite in results.values())
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"BULLETPROOF TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"{'='*60}")
        
        # Exit with appropriate code
        sys.exit(0 if success_rate >= 0.8 else 1)
        
    except Exception as e:
        runner.logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())