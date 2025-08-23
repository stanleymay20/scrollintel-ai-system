"""
Demo script for Security Testing and Validation Framework
"""

import asyncio
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_security_testing_framework():
    """Demonstrate the security testing framework"""
    try:
        # Import the security testing framework
        from security.testing import SecurityTestFramework
        
        logger.info("=== Security Testing and Validation Framework Demo ===")
        
        # Initialize the framework
        config = {
            'penetration': {
                'enabled': True,
                'scenarios': ['sql_injection', 'xss_attacks', 'authentication_bypass']
            },
            'vulnerability': {
                'enabled': True,
                'scan_modules': ['network_scan', 'web_application_scan', 'dependency_scan']
            },
            'chaos': {
                'enabled': True,
                'attack_types': ['ddos', 'credential_stuffing', 'privilege_escalation']
            },
            'performance': {
                'enabled': True,
                'load_profiles': ['light_load', 'moderate_load', 'heavy_load']
            },
            'regression': {
                'enabled': True,
                'test_types': ['authentication', 'authorization', 'input_validation']
            },
            'metrics': {
                'enabled': True,
                'metrics_db_path': 'demo_security_metrics.db'
            }
        }
        
        framework = SecurityTestFramework(config)
        
        # Define target configuration
        target_config = {
            'base_url': 'http://localhost:8000',
            'host': 'localhost',
            'version': '1.0.0',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'username': 'testuser',
                'password': 'testpass'
            }
        }
        
        logger.info("Starting comprehensive security testing...")
        
        # Run comprehensive security tests
        start_time = datetime.now()
        results = await framework.run_comprehensive_security_tests(target_config)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Security testing completed in {execution_time:.2f} seconds")
        logger.info(f"Total tests executed: {len(results)}")
        
        # Analyze results
        passed_tests = [r for r in results if r.status == "passed"]
        failed_tests = [r for r in results if r.status == "failed"]
        error_tests = [r for r in results if r.status == "error"]
        
        logger.info(f"Passed tests: {len(passed_tests)}")
        logger.info(f"Failed tests: {len(failed_tests)}")
        logger.info(f"Error tests: {len(error_tests)}")
        
        # Display results by test type
        test_types = {}
        for result in results:
            test_type = result.test_type.value
            if test_type not in test_types:
                test_types[test_type] = {'passed': 0, 'failed': 0, 'error': 0}
            test_types[test_type][result.status] += 1
        
        logger.info("\n=== Results by Test Type ===")
        for test_type, counts in test_types.items():
            logger.info(f"{test_type.upper()}:")
            logger.info(f"  Passed: {counts['passed']}")
            logger.info(f"  Failed: {counts['failed']}")
            logger.info(f"  Errors: {counts['error']}")
        
        # Display critical findings
        critical_findings = []
        high_findings = []
        
        for result in results:
            for finding in result.findings:
                if finding.get('severity') == 'critical':
                    critical_findings.append({
                        'test': result.test_name,
                        'finding': finding
                    })
                elif finding.get('severity') == 'high':
                    high_findings.append({
                        'test': result.test_name,
                        'finding': finding
                    })
        
        if critical_findings:
            logger.info(f"\n=== CRITICAL FINDINGS ({len(critical_findings)}) ===")
            for item in critical_findings[:5]:  # Show top 5
                logger.info(f"Test: {item['test']}")
                logger.info(f"Finding: {item['finding'].get('title', 'Unknown')}")
                logger.info(f"Description: {item['finding'].get('description', 'No description')}")
                logger.info("---")
        
        if high_findings:
            logger.info(f"\n=== HIGH SEVERITY FINDINGS ({len(high_findings)}) ===")
            for item in high_findings[:3]:  # Show top 3
                logger.info(f"Test: {item['test']}")
                logger.info(f"Finding: {item['finding'].get('title', 'Unknown')}")
                logger.info("---")
        
        # Display recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Get unique recommendations
        unique_recommendations = list(set(all_recommendations))
        
        if unique_recommendations:
            logger.info(f"\n=== TOP RECOMMENDATIONS ({len(unique_recommendations)}) ===")
            for i, recommendation in enumerate(unique_recommendations[:10], 1):
                logger.info(f"{i}. {recommendation}")
        
        # Display performance metrics
        performance_results = [r for r in results if r.test_type.value == "performance"]
        if performance_results:
            logger.info(f"\n=== PERFORMANCE IMPACT ===")
            for result in performance_results:
                overhead = result.metadata.get('security_overhead', 0)
                throughput_impact = result.metadata.get('throughput_impact', 0)
                logger.info(f"Test: {result.test_name}")
                logger.info(f"  Security Overhead: {overhead:.1f}%")
                logger.info(f"  Throughput Impact: {throughput_impact:.1f}%")
        
        # Display chaos engineering results
        chaos_results = [r for r in results if r.test_type.value == "chaos"]
        if chaos_results:
            logger.info(f"\n=== CHAOS ENGINEERING RESULTS ===")
            for result in chaos_results:
                resilience_score = result.metadata.get('resilience_score', 0)
                recovery_time = result.metadata.get('recovery_time', 0)
                attack_type = result.metadata.get('attack_type', 'unknown')
                logger.info(f"Attack Type: {attack_type}")
                logger.info(f"  Resilience Score: {resilience_score:.2f}")
                logger.info(f"  Recovery Time: {recovery_time:.1f}s")
                logger.info(f"  Status: {result.status}")
        
        # Generate summary report
        summary_report = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "total_tests": len(results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "error_tests": len(error_tests),
            "test_coverage": (len(passed_tests) / len(results)) * 100 if results else 0,
            "critical_findings": len(critical_findings),
            "high_findings": len(high_findings),
            "test_types": test_types,
            "top_recommendations": unique_recommendations[:5]
        }
        
        # Save summary report
        with open('security_testing_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"\nSummary report saved to: security_testing_summary.json")
        
        # Display final summary
        logger.info(f"\n=== FINAL SUMMARY ===")
        logger.info(f"Overall Security Score: {summary_report['test_coverage']:.1f}%")
        logger.info(f"Critical Issues: {len(critical_findings)}")
        logger.info(f"High Priority Issues: {len(high_findings)}")
        
        if len(critical_findings) > 0:
            logger.info("🔴 CRITICAL: Immediate action required!")
        elif len(high_findings) > 0:
            logger.info("🟡 WARNING: High priority issues need attention")
        else:
            logger.info("🟢 GOOD: No critical security issues found")
        
        logger.info("=== Security Testing Framework Demo Completed ===")
        
        return summary_report
        
    except ImportError as e:
        logger.error(f"Failed to import security testing framework: {e}")
        logger.info("Make sure the security testing framework is properly installed")
        return None
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def demo_individual_test_types():
    """Demonstrate individual test types"""
    try:
        from security.testing import SecurityTestFramework
        
        logger.info("\n=== Individual Test Types Demo ===")
        
        framework = SecurityTestFramework({})
        target_config = {'base_url': 'http://localhost:8000'}
        
        # Demo penetration testing
        logger.info("Running penetration tests...")
        pentest_results = await framework._run_penetration_tests(target_config)
        logger.info(f"Penetration tests completed: {len(pentest_results)} tests")
        
        # Demo vulnerability scanning
        logger.info("Running vulnerability scans...")
        vuln_results = await framework._run_vulnerability_scans(target_config)
        logger.info(f"Vulnerability scans completed: {len(vuln_results)} scans")
        
        # Demo chaos engineering
        logger.info("Running chaos engineering tests...")
        chaos_results = await framework._run_chaos_tests(target_config)
        logger.info(f"Chaos tests completed: {len(chaos_results)} experiments")
        
        # Demo performance testing
        logger.info("Running security performance tests...")
        perf_results = await framework._run_performance_tests(target_config)
        logger.info(f"Performance tests completed: {len(perf_results)} scenarios")
        
        # Demo regression testing
        logger.info("Running regression tests...")
        regression_results = await framework._run_regression_tests(target_config)
        logger.info(f"Regression tests completed: {len(regression_results)} test cases")
        
        logger.info("Individual test types demo completed")
        
    except Exception as e:
        logger.error(f"Individual test types demo failed: {e}")

if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(demo_security_testing_framework())
    
    # Run individual test types demo
    asyncio.run(demo_individual_test_types())