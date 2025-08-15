#!/usr/bin/env python3
"""
Simple test for data quality components
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the file
exec(open('scrollintel/core/data_quality_monitor.py').read())

def test_quality_monitor():
    """Test the data quality monitor"""
    try:
        # Create monitor
        monitor = DataQualityMonitor()
        print("✓ DataQualityMonitor created successfully")
        
        # Create a simple quality rule
        rule = QualityRule(
            id="test_rule",
            name="Test Completeness Rule",
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
        print(f"  - Overall score: {report.overall_score:.1f}%")
        print(f"  - Issues found: {len(report.issues)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Data Quality Monitor...\n")
    success = test_quality_monitor()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)