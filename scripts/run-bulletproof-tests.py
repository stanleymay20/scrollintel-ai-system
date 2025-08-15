#!/usr/bin/env python3
"""
Bulletproof Testing Script

This script runs the comprehensive bulletproof user experience testing framework
and generates detailed reports on system resilience and recovery capabilities.
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.bulletproof.test_runner import BulletproofTestRunner


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging for the test script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bulletproof_tests.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run bulletproof user experience tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/run-bulletproof-tests.py

  # Run specific test suite
  python scripts/run-bulletproof-tests.py --suite chaos

  # Run with custom output directory
  python scripts/run-bulletproof-tests.py --output /tmp/bulletproof_results

  # Run with verbose logging
  python scripts/run-bulletproof-tests.py --log-level DEBUG

  # Run only fast tests
  python scripts/run-bulletproof-tests.py --fast-only

Test Suites:
  chaos      - Chaos engineering tests (failure injection)
  journey    - User journey tests under failure conditions
  performance - Performance tests with degradation scenarios
  recovery   - Automated recovery tests with success verification
  all        - All test suites (default)
        """
    )
    
    parser.add_argument(
        "--suite",
        choices=["chaos", "journey", "performance", "recovery", "all"],
        default="all",
        help="Test suite to run (default: all)"
    )
    
    parser.add_argument(
        "--output",
        default="test_results",
        help="Output directory for test results (default: test_results)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Run only fast tests (skip slow/stress tests)"
    )
    
    parser.add_argument(
        "--no-html-report",
        action="store_true",
        help="Skip HTML report generation"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first test failure"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel test processes (default: 1)"
    )
    
    return parser.parse_args()


async def validate_environment():
    """Validate that the test environment is properly set up."""
    logger = logging.getLogger(__name__)
    
    # Check required modules
    required_modules = [
        'scrollintel.core.bulletproof_orchestrator',
        'scrollintel.core.failure_prevention',
        'scrollintel.core.graceful_degradation',
        'scrollintel.core.user_experience_protection'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.warning(f"Missing modules (will use mocks): {missing_modules}")
    
    # Check test dependencies
    try:
        import pytest
        import asyncio
        import psutil
    except ImportError as e:
        logger.error(f"Missing test dependency: {e}")
        return False
    
    logger.info("Environment validation completed")
    return True


async def run_bulletproof_tests(args):
    """Run the bulletproof tests with the specified configuration."""
    logger = setup_logging(args.log_level)
    
    logger.info("Starting bulletproof user experience testing...")
    logger.info(f"Test suite: {args.suite}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Fast only: {args.fast_only}")
    
    # Validate environment
    if not await validate_environment():
        logger.error("Environment validation failed")
        return 1
    
    # Create test runner
    runner = BulletproofTestRunner(output_dir=args.output)
    
    try:
        # Run tests based on suite selection
        if args.suite == "all":
            logger.info("Running comprehensive bulletproof test suite...")
            results = await runner.run_all_tests()
        elif args.suite == "chaos":
            logger.info("Running chaos engineering tests...")
            results = {"chaos_engineering": await runner.run_chaos_engineering_tests()}
        elif args.suite == "journey":
            logger.info("Running user journey tests...")
            results = {"user_journey": await runner.run_user_journey_tests()}
        elif args.suite == "performance":
            logger.info("Running performance tests...")
            results = {"performance": await runner.run_performance_tests()}
        elif args.suite == "recovery":
            logger.info("Running recovery tests...")
            results = {"recovery": await runner.run_recovery_tests()}
        
        # Calculate overall results
        total_tests = sum(suite.total_tests for suite in results.values())
        total_passed = sum(suite.passed_tests for suite in results.values())
        total_failed = sum(suite.failed_tests for suite in results.values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # Print detailed results
        print("\n" + "="*80)
        print("BULLETPROOF USER EXPERIENCE TEST RESULTS")
        print("="*80)
        
        for suite_name, suite_result in results.items():
            print(f"\n{suite_name.replace('_', ' ').title()} Tests:")
            print(f"  Total: {suite_result.total_tests}")
            print(f"  Passed: {suite_result.passed_tests}")
            print(f"  Failed: {suite_result.failed_tests}")
            print(f"  Success Rate: {suite_result.success_rate:.2%}")
            print(f"  Duration: {suite_result.duration:.2f}s")
            
            # Show failed tests
            if suite_result.failed_tests > 0:
                print(f"  Failed Tests:")
                for result in suite_result.results:
                    if not result.success:
                        print(f"    - {result.test_name}: {result.error_message}")
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Success Rate: {overall_success_rate:.2%}")
        print("="*80)
        
        # Validate bulletproof requirements
        requirements_met = validate_bulletproof_requirements(results)
        
        if requirements_met:
            print("✅ All bulletproof requirements validated successfully!")
            logger.info("Bulletproof testing completed successfully")
            return 0
        else:
            print("❌ Some bulletproof requirements not met")
            logger.error("Bulletproof testing failed - requirements not met")
            return 1
            
    except Exception as e:
        logger.error(f"Bulletproof testing failed with error: {e}")
        print(f"❌ Testing failed: {e}")
        return 1


def validate_bulletproof_requirements(results: dict) -> bool:
    """Validate that bulletproof requirements are met based on test results."""
    logger = logging.getLogger(__name__)
    
    # Calculate overall metrics
    total_tests = sum(suite.total_tests for suite in results.values())
    total_passed = sum(suite.passed_tests for suite in results.values())
    overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
    
    # Bulletproof requirements validation
    requirements = {
        "Overall Success Rate >= 95%": overall_success_rate >= 0.95,
        "Chaos Engineering Tests Pass": results.get("chaos_engineering", {}).success_rate >= 0.9 if "chaos_engineering" in results else True,
        "User Journey Tests Pass": results.get("user_journey", {}).success_rate >= 0.9 if "user_journey" in results else True,
        "Performance Tests Pass": results.get("performance", {}).success_rate >= 0.8 if "performance" in results else True,
        "Recovery Tests Pass": results.get("recovery", {}).success_rate >= 0.95 if "recovery" in results else True,
    }
    
    print(f"\nBULLETPROOF REQUIREMENTS VALIDATION:")
    print("-" * 50)
    
    all_met = True
    for requirement, met in requirements.items():
        status = "✅ PASS" if met else "❌ FAIL"
        print(f"{status} {requirement}")
        if not met:
            all_met = False
    
    print("-" * 50)
    
    if all_met:
        print("🎉 ALL BULLETPROOF REQUIREMENTS MET!")
        print("ScrollIntel is bulletproof and ready for users!")
    else:
        print("⚠️  SOME REQUIREMENTS NOT MET")
        print("Additional work needed to achieve bulletproof status")
    
    return all_met


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set environment variables for testing
    os.environ['BULLETPROOF_TEST_MODE'] = 'true'
    os.environ['BULLETPROOF_LOG_LEVEL'] = args.log_level
    
    if args.fast_only:
        os.environ['BULLETPROOF_FAST_TESTS_ONLY'] = 'true'
    
    # Run the tests
    try:
        exit_code = asyncio.run(run_bulletproof_tests(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()