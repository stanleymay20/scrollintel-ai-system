"""
Demo script for Error Detection and Correction system

This script demonstrates the autonomous error detection, correction, and prevention
capabilities for all innovation lab processes.
"""

import asyncio
import json
from datetime import datetime
from scrollintel.engines.error_detection_correction import ErrorDetectionCorrection

async def demo_error_detection_correction():
    """Demonstrate error detection and correction capabilities"""
    print("🔍 Error Detection and Correction Demo")
    print("=" * 50)
    
    # Initialize error detection system
    error_system = ErrorDetectionCorrection()
    print("✅ Error Detection and Correction system initialized")
    
    # Demo 1: Research Error Detection
    print("\n📊 Demo 1: Research Error Detection")
    print("-" * 40)
    
    # Good research data (should have few/no errors)
    good_research_data = {
        "methodology": "experimental",
        "sample_size": 100,
        "hypothesis": "This is a comprehensive testable hypothesis to measure the effectiveness of the intervention",
        "literature_sources": 15,
        "has_control_group": True,
        "randomized": True
    }
    
    good_errors = await error_system.detect_errors(
        process_id="research_good",
        process_type="research",
        process_data=good_research_data
    )
    
    print(f"Good research data - Errors detected: {len(good_errors)}")
    
    # Poor research data (should have multiple errors)
    poor_research_data = {
        "sample_size": 5,  # Too small
        "hypothesis": "Bad",  # Too vague
        "literature_sources": 2,  # Too few
        "has_control_group": False,  # Missing control
        "randomized": False  # Not randomized
    }
    
    poor_errors = await error_system.detect_errors(
        process_id="research_poor",
        process_type="research",
        process_data=poor_research_data
    )
    
    print(f"Poor research data - Errors detected: {len(poor_errors)}")
    
    if poor_errors:
        print("Detected errors:")
        for error in poor_errors[:3]:  # Show first 3
            print(f"  - {error.severity.value.upper()}: {error.error_message}")
    
    # Demo 2: Experiment Error Detection
    print("\n🧪 Demo 2: Experiment Error Detection")
    print("-" * 40)
    
    experiment_data = {
        "has_control_group": False,  # Missing control
        "randomized": False,  # Not randomized
        "missing_data_percentage": 25,  # High missing data
        "data_type": "categorical",
        "statistical_test": "t-test"  # Wrong test for data type
    }
    
    experiment_errors = await error_system.detect_errors(
        process_id="experiment_001",
        process_type="experiment",
        process_data=experiment_data
    )
    
    print(f"Experiment errors detected: {len(experiment_errors)}")
    
    if experiment_errors:
        print("Experiment errors:")
        for error in experiment_errors:
            print(f"  - {error.error_type.value}: {error.error_message}")
    
    # Demo 3: Prototype Error Detection
    print("\n🔧 Demo 3: Prototype Error Detection")
    print("-" * 40)
    
    prototype_data = {
        "core_functions": {
            "login": True,
            "data_processing": True,
            "reporting": False,  # Failed function
            "export": False  # Failed function
        },
        "response_time": 8000,  # Too slow
        "integrations": {
            "database": True,
            "api": False,  # Failed integration
            "payment": False  # Failed integration
        }
    }
    
    prototype_errors = await error_system.detect_errors(
        process_id="prototype_001",
        process_type="prototype",
        process_data=prototype_data
    )
    
    print(f"Prototype errors detected: {len(prototype_errors)}")
    
    if prototype_errors:
        print("Prototype errors:")
        for error in prototype_errors:
            print(f"  - {error.severity.value.upper()}: {error.error_message}")
    
    # Demo 4: Error Correction
    print("\n⚡ Demo 4: Error Correction")
    print("-" * 40)
    
    # Collect all errors for correction
    all_errors = poor_errors + experiment_errors + prototype_errors
    
    if all_errors:
        print(f"Attempting to correct {len(all_errors)} errors...")
        
        correction_results = await error_system.correct_errors(all_errors)
        
        print(f"Correction attempts: {len(correction_results)}")
        
        successful_corrections = [r for r in correction_results if r.success]
        failed_corrections = [r for r in correction_results if not r.success]
        
        print(f"Successful corrections: {len(successful_corrections)}")
        print(f"Failed corrections: {len(failed_corrections)}")
        
        if correction_results:
            print("Sample correction results:")
            for result in correction_results[:3]:  # Show first 3
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"  - {status}: {result.result_message}")
    
    # Demo 5: Error Prevention
    print("\n🛡️ Demo 5: Error Prevention")
    print("-" * 40)
    
    # Test prevention on new process data
    new_process_data = {
        "sample_size": 20,
        "hypothesis": "Testing prevention",
        "literature_sources": 5,
        "methodology": "survey"
    }
    
    prevention_result = await error_system.prevent_errors(
        process_id="prevention_test",
        process_type="research",
        process_data=new_process_data
    )
    
    print(f"Risk factors identified: {len(prevention_result.get('risk_factors', []))}")
    print(f"Prevention actions applied: {len(prevention_result.get('prevention_actions', []))}")
    print(f"Risk mitigation score: {prevention_result.get('risk_mitigation_score', 0):.3f}")
    
    if prevention_result.get('risk_factors'):
        print("Risk factors:")
        for factor in prevention_result['risk_factors']:
            print(f"  - {factor.get('description', 'Unknown')} (risk: {factor.get('risk_score', 0):.2f})")
    
    # Demo 6: Error Patterns
    print("\n📈 Demo 6: Error Patterns")
    print("-" * 40)
    
    print("Learned error patterns:")
    for pattern_id, pattern in error_system.error_patterns.items():
        print(f"  - {pattern_id}: {pattern.occurrence_count} occurrences")
        print(f"    Description: {pattern.pattern_description}")
    
    # Demo 7: Learning from Errors
    print("\n🧠 Demo 7: Learning from Errors")
    print("-" * 40)
    
    learning_results = await error_system.learn_from_errors()
    
    print("Learning results:")
    print(f"  New patterns: {len(learning_results.get('new_patterns', []))}")
    print(f"  Updated patterns: {len(learning_results.get('updated_patterns', []))}")
    print(f"  New prevention rules: {len(learning_results.get('new_prevention_rules', []))}")
    print(f"  Improved corrections: {len(learning_results.get('improved_corrections', []))}")
    
    # Demo 8: System Error Detection
    print("\n💻 Demo 8: System Error Detection")
    print("-" * 40)
    
    system_data = {
        "memory_usage": 0.95,  # Critical memory usage
        "connection_failures": 5,  # Multiple failures
        "required_configs": ["database_url", "api_key", "secret"],
        "missing_configs": ["api_key", "secret"]  # Missing configs
    }
    
    system_errors = await error_system.detect_errors(
        process_id="system_001",
        process_type="system",
        process_data=system_data
    )
    
    print(f"System errors detected: {len(system_errors)}")
    
    if system_errors:
        print("System errors:")
        for error in system_errors:
            print(f"  - {error.severity.value.upper()}: {error.error_message}")
    
    # Demo 9: Data Quality Error Detection
    print("\n📊 Demo 9: Data Quality Error Detection")
    print("-" * 40)
    
    data_quality_data = {
        "anomaly_score": 0.85,  # High anomaly score
        "format_consistency": 0.75,  # Low consistency
        "completeness_score": 0.65,  # Low completeness
        "missing_data_percentage": 35  # High missing data
    }
    
    data_errors = await error_system.detect_errors(
        process_id="data_001",
        process_type="data_processing",
        process_data=data_quality_data
    )
    
    print(f"Data quality errors detected: {len(data_errors)}")
    
    if data_errors:
        print("Data quality errors:")
        for error in data_errors:
            print(f"  - {error.error_type.value}: {error.error_message}")
    
    # Demo 10: Continuous Monitoring (brief demo)
    print("\n🔄 Demo 10: Continuous Monitoring")
    print("-" * 40)
    
    print("Starting continuous monitoring...")
    monitoring_task = asyncio.create_task(error_system.start_continuous_monitoring())
    
    # Let it run briefly
    await asyncio.sleep(2)
    print(f"Monitoring active: {error_system.monitoring_active}")
    
    # Stop monitoring
    error_system.stop_continuous_monitoring()
    print("Monitoring stopped")
    
    # Wait for task to complete
    try:
        await asyncio.wait_for(monitoring_task, timeout=1.0)
    except asyncio.TimeoutError:
        monitoring_task.cancel()
    
    # Demo 11: Error Statistics
    print("\n📊 Demo 11: Error Statistics")
    print("-" * 40)
    
    total_errors = len(error_system.detected_errors)
    total_corrections = sum(len(corrections) for corrections in error_system.correction_history.values())
    
    print(f"Total errors detected: {total_errors}")
    print(f"Total correction attempts: {total_corrections}")
    
    # Error type distribution
    error_types = {}
    severity_distribution = {}
    
    for error in error_system.detected_errors.values():
        error_type = error.error_type.value
        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        severity = error.severity.value
        severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
    
    print("\nError type distribution:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    print("\nSeverity distribution:")
    for severity, count in severity_distribution.items():
        print(f"  {severity}: {count}")
    
    # Demo 12: Correction Success Rate
    print("\n📈 Demo 12: Correction Success Rate")
    print("-" * 40)
    
    successful_corrections = 0
    total_correction_attempts = 0
    
    for corrections in error_system.correction_history.values():
        for correction in corrections:
            total_correction_attempts += 1
            if correction.success:
                successful_corrections += 1
    
    if total_correction_attempts > 0:
        success_rate = (successful_corrections / total_correction_attempts) * 100
        print(f"Correction success rate: {success_rate:.1f}%")
        print(f"Successful corrections: {successful_corrections}/{total_correction_attempts}")
    else:
        print("No correction attempts recorded yet")
    
    print("\n🎉 Error Detection and Correction Demo Complete!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(demo_error_detection_correction())