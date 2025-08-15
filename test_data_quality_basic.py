#!/usr/bin/env python3
"""
Basic test for data quality system components
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_normalizer():
    """Test basic data normalizer functionality"""
    try:
        from scrollintel.core.data_normalizer import (
            DataNormalizer, DataSchema, SchemaField, SchemaMapping, 
            DataType, TransformationType
        )
        
        print("✓ DataNormalizer imports successful")
        
        # Create normalizer
        normalizer = DataNormalizer()
        
        # Create simple schemas
        source_schema = DataSchema(
            name="source",
            version="1.0",
            fields=[
                SchemaField("id", DataType.STRING),
                SchemaField("name", DataType.STRING)
            ]
        )
        
        target_schema = DataSchema(
            name="target",
            version="1.0",
            fields=[
                SchemaField("customer_id", DataType.STRING),
                SchemaField("customer_name", DataType.STRING)
            ]
        )
        
        # Register schemas
        assert normalizer.register_schema(source_schema)
        assert normalizer.register_schema(target_schema)
        print("✓ Schema registration successful")
        
        # Create mappings
        mappings = [
            SchemaMapping("id", "customer_id", TransformationType.DIRECT_MAPPING),
            SchemaMapping("name", "customer_name", TransformationType.DIRECT_MAPPING)
        ]
        
        assert normalizer.register_mapping("source", "target", mappings)
        print("✓ Mapping registration successful")
        
        # Test data normalization
        test_data = pd.DataFrame([
            {"id": "1", "name": "John Doe"},
            {"id": "2", "name": "Jane Smith"}
        ])
        
        result = normalizer.normalize_data(test_data, "source", "target")
        assert result.success
        assert len(result.normalized_data) == 2
        assert "customer_id" in result.normalized_data.columns
        print("✓ Data normalization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ DataNormalizer test failed: {str(e)}")
        return False

def test_data_quality_monitor():
    """Test basic data quality monitor functionality"""
    try:
        # Try to import the module first
        import scrollintel.core.data_quality_monitor as dqm_module
        
        # Check if the classes are available
        if not hasattr(dqm_module, 'DataQualityMonitor'):
            print("✗ DataQualityMonitor class not found in module")
            return False
            
        from scrollintel.core.data_quality_monitor import (
            DataQualityMonitor, QualityRule, QualityRuleType, QualitySeverity
        )
        
        print("✓ DataQualityMonitor imports successful")
        
        # Create monitor
        monitor = DataQualityMonitor()
        
        # Create a simple quality rule
        rule = QualityRule(
            id="test_rule",
            name="Test Rule",
            description="Test completeness rule",
            rule_type=QualityRuleType.COMPLETENESS,
            severity=QualitySeverity.HIGH,
            target_fields=["name"],
            parameters={"min_completeness": 0.8}
        )
        
        assert monitor.register_quality_rule(rule)
        print("✓ Quality rule registration successful")
        
        # Test data quality assessment
        test_data = pd.DataFrame([
            {"id": "1", "name": "John Doe"},
            {"id": "2", "name": None},  # Missing name
            {"id": "3", "name": "Jane Smith"}
        ])
        
        report = monitor.assess_data_quality(test_data, "test_dataset")
        assert report.dataset_name == "test_dataset"
        assert report.total_records == 3
        print("✓ Data quality assessment successful")
        
        return True
        
    except Exception as e:
        print(f"✗ DataQualityMonitor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run basic tests"""
    print("Running basic data quality system tests...\n")
    
    tests = [
        ("Data Normalizer", test_data_normalizer),
        ("Data Quality Monitor", test_data_quality_monitor)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}:")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All basic tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)