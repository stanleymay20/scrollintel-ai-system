#!/usr/bin/env python3
"""
Simple Bulletproof Testing Script

This script runs the simple bulletproof user experience validation tests
to verify that the core bulletproof functionality is working correctly.
"""

import asyncio
import sys
import os
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging for the test script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('simple_bulletproof_tests.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run simple bulletproof user experience validation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validation tests
  python scripts/run-simple-bulletproof-tests.py

  # Run with verbose logging
  python scripts/run-simple-bulletproof-tests.py --log-level DEBUG

  # Run specific test
  python scripts/run-simple-bulletproof-tests.py --test comprehensive

Test Types:
  comprehensive  - Run comprehensive bulletproof validation
  metrics       - Run bulletproof metrics validation
  all           - Run all tests (default)
        """
    )
    
    parser.add_argument(
        "--test",
        choices=["comprehensive", "metrics", "all"],
        default="all",
        help="Test type to run (default: all)"
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
    
    return parser.parse_args()


async def run_simple_bulletproof_validation():
    """Run simple bulletproof validation tests."""
    try:
        from tests.bulletproof.test_simple_validation import SimpleBulletproofValidator
        
        validator = SimpleBulletproofValidator()
        result = await validator.run_comprehensive_validation()
        
        return result
    except Exception as e:
        return {
            'error': str(e),
            'validation_results': {},
            'total_tests': 0,
            'passed_tests': 0,
            'success_rate': 0.0,
            'bulletproof_validated': False
        }


async def run_bulletproof_tests(args):
    """Run the bulletproof tests with the specified configuration."""
    logger = setup_logging(args.log_level)
    
    logger.info("Starting simple bulletproof user experience validation...")
    logger.info(f"Test type: {args.test}")
    logger.info(f"Output directory: {args.output}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Run validation
        logger.info("Running bulletproof validation...")
        validation_result = await run_simple_bulletproof_validation()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate report
        report = {
            "bulletproof_validation_report": {
                "timestamp": datetime.now().isoformat(),
                "duration": total_duration,
                "validation_result": validation_result,
                "summary": {
                    "total_tests": validation_result.get('total_tests', 0),
                    "passed_tests": validation_result.get('passed_tests', 0),
                    "success_rate": validation_result.get('success_rate', 0.0),
                    "bulletproof_validated": validation_result.get('bulletproof_validated', False)
                }
            }
        }
        
        # Save report
        import json
        report_file = output_dir / f"bulletproof_validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print results
        print("\n" + "="*80)
        print("BULLETPROOF USER EXPERIENCE VALIDATION RESULTS")
        print("="*80)
        
        if 'error' in validation_result:
            print(f"❌ Validation failed with error: {validation_result['error']}")
            return 1
        
        print(f"Total Tests: {validation_result['total_tests']}")
        print(f"Passed Tests: {validation_result['passed_tests']}")
        print(f"Success Rate: {validation_result['success_rate']:.2%}")
        print(f"Duration: {total_duration:.2f} seconds")
        
        print(f"\nDETAILED RESULTS:")
        for test_name, test_result in validation_result['validation_results'].items():
            if isinstance(test_result, dict) and 'error' not in test_result:
                print(f"  ✅ {test_name.replace('_', ' ').title()}: PASSED")
            else:
                print(f"  ❌ {test_name.replace('_', ' ').title()}: FAILED")
        
        # Validate bulletproof requirements
        requirements_met = validate_bulletproof_requirements(validation_result)
        
        print(f"\nBULLETPROOF REQUIREMENTS VALIDATION:")
        print("-" * 50)
        
        requirements = {
            "Never-Fail User Experience": validation_result['validation_results'].get('user_experience_protection', {}).get('user_action_protected', False),
            "Intelligent Error Recovery": validation_result['validation_results'].get('recovery_mechanism', {}).get('recovery_successful', False),
            "Proactive User Protection": validation_result['validation_results'].get('graceful_degradation', {}).get('functionality_maintained', False),
            "Predictive Failure Prevention": validation_result['validation_results'].get('failure_handling', {}).get('recovery_attempted', False),
            "Overall Success Rate >= 95%": validation_result['success_rate'] >= 0.95
        }
        
        all_met = True
        for requirement, met in requirements.items():
            status = "✅ PASS" if met else "❌ FAIL"
            print(f"{status} {requirement}")
            if not met:
                all_met = False
        
        print("-" * 50)
        
        if all_met and validation_result['bulletproof_validated']:
            print("🎉 ALL BULLETPROOF REQUIREMENTS MET!")
            print("ScrollIntel is bulletproof and ready for users!")
            logger.info("Bulletproof validation completed successfully")
            return 0
        else:
            print("⚠️  SOME REQUIREMENTS NOT MET")
            print("Additional work needed to achieve bulletproof status")
            logger.warning("Bulletproof validation failed - requirements not met")
            return 1
            
    except Exception as e:
        logger.error(f"Bulletproof validation failed with error: {e}")
        print(f"❌ Validation failed: {e}")
        return 1


def validate_bulletproof_requirements(validation_result: dict) -> bool:
    """Validate that bulletproof requirements are met."""
    if 'error' in validation_result:
        return False
    
    # Check core requirements
    success_rate = validation_result.get('success_rate', 0.0)
    bulletproof_validated = validation_result.get('bulletproof_validated', False)
    
    return success_rate >= 0.95 and bulletproof_validated


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set environment variables for testing
    os.environ['BULLETPROOF_TEST_MODE'] = 'true'
    os.environ['BULLETPROOF_LOG_LEVEL'] = args.log_level
    
    # Run the tests
    try:
        exit_code = asyncio.run(run_bulletproof_tests(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()