#!/usr/bin/env python3
"""
Simple test for testing automation framework functionality
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_testing_automation_framework():
    """Test testing automation framework functionality"""
    try:
        print("Testing testing automation framework...")
        
        # Import required modules
        from scrollintel.models.prototype_models import (
            Concept, Prototype, PrototypeType, PrototypeStatus,
            ConceptCategory, QualityMetrics, create_concept_from_description
        )
        from scrollintel.engines.testing_automation_framework import (
            TestingAutomationFramework, TestGenerator, TestExecutor,
            TestType, TestPriority, TestCase
        )
        
        # Create a test prototype
        concept = create_concept_from_description(
            name="E-commerce API",
            description="A REST API service for e-commerce platform with user management",
            category=ConceptCategory.PRODUCT
        )
        
        prototype = Prototype(
            concept_id=concept.id,
            name="E-commerce API Prototype",
            description="A prototype API for e-commerce testing",
            prototype_type=PrototypeType.API_SERVICE,
            status=PrototypeStatus.FUNCTIONAL,
            generated_code={
                "main": '''
from fastapi import FastAPI
app = FastAPI(title="E-commerce API")

@app.get("/")
async def root():
    return {"message": "Welcome to E-commerce API"}

@app.get("/products")
async def get_products():
    return []

@app.post("/products")
async def create_product(product: dict):
    return product

def calculate_total(items):
    return sum(item.get("price", 0) for item in items)

async def process_order(order_data):
    return {"order_id": "12345", "status": "processed"}
''',
                "requirements": "fastapi==0.104.1\nuvicorn==0.24.0"
            },
            quality_metrics=QualityMetrics(
                code_coverage=0.6,
                performance_score=0.7,
                usability_score=0.8,
                reliability_score=0.7,
                security_score=0.6,
                maintainability_score=0.8,
                scalability_score=0.7
            )
        )
        
        print(f"✅ Created test prototype: {prototype.name}")
        print(f"   Prototype type: {prototype.prototype_type.value}")
        print(f"   Generated code files: {len(prototype.generated_code)}")
        
        # Create testing automation framework
        testing_framework = TestingAutomationFramework()
        print("✅ Testing automation framework created successfully")
        
        # Test test generator
        print("\n🔄 Testing test generator...")
        test_generator = TestGenerator()
        test_suite = await test_generator.generate_test_suite(prototype)
        
        print(f"✅ Test suite generated:")
        print(f"   - Suite name: {test_suite.name}")
        print(f"   - Total test cases: {len(test_suite.test_cases)}")
        print(f"   - Parallel execution: {test_suite.parallel_execution}")
        
        # Display test cases by type
        test_types = {}
        for test_case in test_suite.test_cases:
            test_type = test_case.test_type.value
            test_types[test_type] = test_types.get(test_type, 0) + 1
        
        print(f"   - Test breakdown:")
        for test_type, count in test_types.items():
            print(f"     * {test_type}: {count} tests")
        
        # Test individual test case
        if test_suite.test_cases:
            sample_test = test_suite.test_cases[0]
            print(f"\n📋 Sample test case:")
            print(f"   - Name: {sample_test.name}")
            print(f"   - Type: {sample_test.test_type.value}")
            print(f"   - Priority: {sample_test.priority.value}")
            print(f"   - Timeout: {sample_test.timeout_seconds}s")
            print(f"   - Has test code: {len(sample_test.test_code) > 0}")
        
        # Test test executor
        print("\n🔄 Testing test executor...")
        test_executor = TestExecutor()
        test_report = await test_executor.execute_test_suite(test_suite, prototype)
        
        print(f"✅ Test execution completed:")
        print(f"   - Total tests: {test_report.total_tests}")
        print(f"   - Passed tests: {test_report.passed_tests}")
        print(f"   - Failed tests: {test_report.failed_tests}")
        print(f"   - Success rate: {test_report.success_rate:.1%}")
        print(f"   - Total duration: {test_report.total_duration_seconds:.2f}s")
        print(f"   - Average test duration: {test_report.average_test_duration:.2f}s")
        
        # Test quality scores
        print(f"\n📊 Quality scores:")
        print(f"   - Overall coverage: {test_report.overall_coverage:.1%}")
        print(f"   - Performance score: {test_report.performance_score:.2f}")
        print(f"   - Security score: {test_report.security_score:.2f}")
        print(f"   - Code quality score: {test_report.code_quality_score:.2f}")
        
        # Test failure analysis
        if test_report.failure_analysis:
            print(f"\n🔍 Failure analysis:")
            failure_analysis = test_report.failure_analysis
            print(f"   - Total failures: {failure_analysis.get('total_failures', 0)}")
            print(f"   - Failure rate: {failure_analysis.get('failure_rate', 0):.1%}")
            if failure_analysis.get('most_common_error'):
                print(f"   - Most common error: {failure_analysis['most_common_error']}")
        
        # Test performance analysis
        if test_report.performance_analysis:
            print(f"\n⚡ Performance analysis:")
            perf_analysis = test_report.performance_analysis
            if 'average_execution_time' in perf_analysis:
                print(f"   - Average execution time: {perf_analysis['average_execution_time']:.3f}s")
                print(f"   - Average memory usage: {perf_analysis['average_memory_usage']:.1f}MB")
                print(f"   - Average CPU usage: {perf_analysis['average_cpu_usage']:.1f}%")
        
        # Test recommendations
        if test_report.recommendations:
            print(f"\n💡 Recommendations:")
            for i, recommendation in enumerate(test_report.recommendations[:3], 1):
                print(f"   {i}. {recommendation}")
        
        # Test full framework integration
        print("\n🔄 Testing full framework integration...")
        full_report = await testing_framework.create_and_execute_tests(prototype)
        
        print(f"✅ Full framework test completed:")
        print(f"   - Success rate: {full_report.success_rate:.1%}")
        print(f"   - Total duration: {full_report.total_duration_seconds:.2f}s")
        print(f"   - Test results added to prototype: {len(prototype.test_results)}")
        
        # Test comprehensive report generation
        print("\n📋 Generating comprehensive report...")
        comprehensive_report = await testing_framework.generate_comprehensive_report(prototype.id)
        
        if 'error' not in comprehensive_report:
            print(f"✅ Comprehensive report generated:")
            summary = comprehensive_report['test_results_summary']
            print(f"   - Total tests: {summary['total_tests']}")
            print(f"   - Success rate: {summary['success_rate']:.1%}")
            print(f"   - Total duration: {summary['total_duration']:.2f}s")
            
            quality = comprehensive_report['quality_metrics']
            print(f"   - Overall coverage: {quality['overall_coverage']:.1%}")
            print(f"   - Performance score: {quality['performance_score']:.2f}")
        
        # Test analytics
        print("\n📈 Testing analytics...")
        analytics = await testing_framework.get_testing_analytics()
        
        if 'error' not in analytics:
            print(f"✅ Analytics generated:")
            print(f"   - Total test suites: {analytics['total_test_suites']}")
            print(f"   - Total test reports: {analytics['total_test_reports']}")
            
            if 'aggregate_statistics' in analytics:
                stats = analytics['aggregate_statistics']
                print(f"   - Total tests executed: {stats['total_tests_executed']}")
                print(f"   - Overall success rate: {stats['overall_success_rate']:.1%}")
        
        print("\n🎉 Testing automation framework test completed successfully!")
        print("\n📋 Task 3.3 Implementation Summary:")
        print("   ✅ Automated testing and validation of prototypes")
        print("   ✅ Comprehensive testing protocol and execution")
        print("   ✅ Testing result analysis and interpretation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing automation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_testing_automation_framework())
    sys.exit(0 if success else 1)