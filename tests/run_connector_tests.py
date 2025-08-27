"""
Comprehensive test runner for all data connector implementations.
Runs all connector tests and generates a detailed report.
"""

import asyncio
import sys
import time
from pathlib import Path
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ConnectorTestRunner:
    """Test runner for all connector implementations"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all connector tests and return results"""
        self.start_time = datetime.utcnow()
        
        print("🚀 Starting comprehensive connector tests...")
        print("=" * 60)
        
        # Define test modules to run
        test_modules = [
            'tests.test_connector_integration',
            'tests.test_erp_connectors',
            'tests.test_crm_connectors',
            'tests.test_bi_connectors',
            'tests.test_cloud_connectors'
        ]
        
        # Run each test module
        for module in test_modules:
            print(f"\n📋 Running {module}...")
            result = self._run_test_module(module)
            self.test_results[module] = result
            
            if result['success']:
                print(f"✅ {module} - PASSED ({result['tests_run']} tests)")
            else:
                print(f"❌ {module} - FAILED ({result['failures']} failures)")
        
        self.end_time = datetime.utcnow()
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.test_results
    
    def _run_test_module(self, module_name: str) -> Dict[str, Any]:
        """Run a specific test module using pytest"""
        try:
            # Convert module name to file path
            file_path = module_name.replace('.', '/') + '.py'
            
            # Run pytest with verbose output and JSON report
            cmd = [
                sys.executable, '-m', 'pytest', 
                file_path, 
                '-v', 
                '--tb=short',
                '--json-report',
                '--json-report-file=temp_test_report.json'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout per module
            )
            
            # Parse JSON report if available
            test_details = self._parse_json_report()
            
            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'tests_run': test_details.get('summary', {}).get('total', 0),
                'failures': test_details.get('summary', {}).get('failed', 0),
                'errors': test_details.get('summary', {}).get('error', 0),
                'skipped': test_details.get('summary', {}).get('skipped', 0),
                'duration': test_details.get('duration', 0),
                'details': test_details
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test timed out after 5 minutes',
                'tests_run': 0,
                'failures': 1,
                'errors': 0,
                'skipped': 0,
                'duration': 300,
                'details': {}
            }
        except Exception as e:
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'tests_run': 0,
                'failures': 1,
                'errors': 0,
                'skipped': 0,
                'duration': 0,
                'details': {}
            }
    
    def _parse_json_report(self) -> Dict[str, Any]:
        """Parse pytest JSON report"""
        try:
            with open('temp_test_report.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        finally:
            # Clean up temp file
            try:
                Path('temp_test_report.json').unlink(missing_ok=True)
            except:
                pass
    
    def _generate_summary_report(self):
        """Generate and display summary report"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate totals
        total_tests = sum(r['tests_run'] for r in self.test_results.values())
        total_failures = sum(r['failures'] for r in self.test_results.values())
        total_errors = sum(r['errors'] for r in self.test_results.values())
        total_skipped = sum(r['skipped'] for r in self.test_results.values())
        
        successful_modules = sum(1 for r in self.test_results.values() if r['success'])
        total_modules = len(self.test_results)
        
        print("\n" + "=" * 60)
        print("📊 CONNECTOR TEST SUMMARY REPORT")
        print("=" * 60)
        
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        print(f"📦 Modules Tested: {total_modules}")
        print(f"✅ Successful Modules: {successful_modules}")
        print(f"❌ Failed Modules: {total_modules - successful_modules}")
        print()
        print(f"🧪 Total Tests: {total_tests}")
        print(f"✅ Passed: {total_tests - total_failures - total_errors}")
        print(f"❌ Failed: {total_failures}")
        print(f"💥 Errors: {total_errors}")
        print(f"⏭️  Skipped: {total_skipped}")
        
        # Module breakdown
        print("\n📋 MODULE BREAKDOWN:")
        print("-" * 60)
        
        for module, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            module_name = module.split('.')[-1]
            
            print(f"{status} {module_name:25} "
                  f"Tests: {result['tests_run']:3d} "
                  f"Failures: {result['failures']:2d} "
                  f"Duration: {result['duration']:6.2f}s")
        
        # Connector type summary
        print("\n🔌 CONNECTOR TYPE COVERAGE:")
        print("-" * 60)
        
        connector_types = {
            'ERP Connectors': ['test_erp_connectors'],
            'CRM Connectors': ['test_crm_connectors'],
            'BI Connectors': ['test_bi_connectors'],
            'Cloud Connectors': ['test_cloud_connectors'],
            'Integration Tests': ['test_connector_integration']
        }
        
        for connector_type, modules in connector_types.items():
            type_results = [self.test_results.get(f'tests.{m}', {}) for m in modules]
            type_success = all(r.get('success', False) for r in type_results)
            type_tests = sum(r.get('tests_run', 0) for r in type_results)
            
            status = "✅" if type_success else "❌"
            print(f"{status} {connector_type:20} Tests: {type_tests:3d}")
        
        # Error details for failed modules
        failed_modules = [m for m, r in self.test_results.items() if not r['success']]
        
        if failed_modules:
            print("\n💥 FAILURE DETAILS:")
            print("-" * 60)
            
            for module in failed_modules:
                result = self.test_results[module]
                print(f"\n❌ {module}:")
                if result['stderr']:
                    print(f"   Error: {result['stderr'][:200]}...")
                if result['stdout']:
                    # Extract key failure info from stdout
                    lines = result['stdout'].split('\n')
                    failure_lines = [l for l in lines if 'FAILED' in l or 'ERROR' in l]
                    for line in failure_lines[:3]:  # Show first 3 failures
                        print(f"   {line}")
        
        # Overall result
        print("\n" + "=" * 60)
        if total_failures == 0 and total_errors == 0:
            print("🎉 ALL CONNECTOR TESTS PASSED! 🎉")
        else:
            print(f"⚠️  TESTS COMPLETED WITH {total_failures + total_errors} ISSUES")
        print("=" * 60)
        
        # Save detailed report to file
        self._save_detailed_report()
    
    def _save_detailed_report(self):
        """Save detailed test report to file"""
        report_data = {
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'summary': {
                'total_modules': len(self.test_results),
                'successful_modules': sum(1 for r in self.test_results.values() if r['success']),
                'total_tests': sum(r['tests_run'] for r in self.test_results.values()),
                'total_failures': sum(r['failures'] for r in self.test_results.values()),
                'total_errors': sum(r['errors'] for r in self.test_results.values()),
                'total_skipped': sum(r['skipped'] for r in self.test_results.values())
            },
            'module_results': self.test_results
        }
        
        report_file = f"connector_test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save detailed report: {e}")


async def run_async_connector_tests():
    """Run connector tests that require async functionality"""
    print("\n🔄 Running async connector validation tests...")
    
    try:
        # Import and test the data integration setup
        from scrollintel.core.data_integration_setup import setup_data_integration, create_sample_configurations
        
        # Test integration manager setup
        manager = setup_data_integration()
        print("✅ Data integration manager setup successful")
        
        # Test sample configurations
        configs = create_sample_configurations()
        print(f"✅ Created {len(configs)} sample configurations")
        
        # Test connector registry
        registry = manager.registry
        expected_connectors = [
            'sap', 'oracle_erp', 'microsoft_dynamics',  # ERP
            'salesforce', 'hubspot', 'microsoft_crm',   # CRM
            'tableau', 'powerbi', 'looker', 'qlik',     # BI
            'aws', 'azure', 'gcp'                       # Cloud
        ]
        
        for connector_name in expected_connectors:
            if connector_name in registry.connector_classes:
                print(f"✅ {connector_name} connector registered")
            else:
                print(f"❌ {connector_name} connector missing")
        
        print("✅ Async connector validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Async connector validation failed: {e}")
        return False


def main():
    """Main test runner function"""
    print("🧪 Advanced Analytics Dashboard - Connector Test Suite")
    print("=" * 60)
    
    # Run async validation first
    async_success = asyncio.run(run_async_connector_tests())
    
    if not async_success:
        print("❌ Async validation failed, skipping main tests")
        return 1
    
    # Run main test suite
    runner = ConnectorTestRunner()
    results = runner.run_all_tests()
    
    # Determine exit code
    total_failures = sum(r['failures'] + r['errors'] for r in results.values())
    failed_modules = sum(1 for r in results.values() if not r['success'])
    
    if total_failures == 0 and failed_modules == 0:
        print("\n🎉 All connector tests completed successfully!")
        return 0
    else:
        print(f"\n⚠️  Tests completed with {total_failures} test failures in {failed_modules} modules")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)