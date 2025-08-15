#!/usr/bin/env python3
"""Simple compliance analyzer test"""

import pandas as pd
from ai_data_readiness.models.compliance_models import (
    ComplianceReport, RegulationType, ComplianceStatus
)
from datetime import datetime

class SimpleComplianceAnalyzer:
    """Simple compliance analyzer for testing"""
    
    def analyze_compliance(self, dataset_id: str, data: pd.DataFrame) -> ComplianceReport:
        """Simple compliance analysis"""
        return ComplianceReport(
            dataset_id=dataset_id,
            regulations_checked=[RegulationType.GDPR],
            compliance_status=ComplianceStatus.COMPLIANT,
            compliance_score=1.0,
            sensitive_data_detections=[],
            violations=[],
            recommendations=[],
            analysis_timestamp=datetime.utcnow(),
            total_records=len(data),
            sensitive_records_count=0
        )

# Test the simple analyzer
if __name__ == "__main__":
    analyzer = SimpleComplianceAnalyzer()
    test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    report = analyzer.analyze_compliance('test', test_data)
    print(f"✓ Simple analyzer works: {report.compliance_status}")