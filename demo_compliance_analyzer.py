#!/usr/bin/env python3
"""
Demo script for AI Data Readiness Platform - Compliance Analyzer

This script demonstrates the regulatory compliance analysis capabilities
including GDPR/CCPA compliance checking, sensitive data detection,
and privacy-preserving recommendations.
"""

print("Loading demo script...")

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import the models directly
from ai_data_readiness.models.compliance_models import (
    ComplianceReport, ComplianceViolation, PrivacyRecommendation,
    SensitiveDataDetection, RegulationType, ViolationSeverity,
    PrivacyTechnique, ComplianceStatus, SensitiveDataType
)


@dataclass
class SensitiveDataPattern:
    """Pattern definition for sensitive data detection"""
    name: str
    pattern: str
    data_type: SensitiveDataType
    confidence_threshold: float
    description: str


class ComplianceAnalyzer:
    """
    Comprehensive regulatory compliance analyzer for AI datasets
    
    Supports GDPR, CCPA, and other privacy regulations with automated
    sensitive data detection and privacy-preserving recommendations.
    """
    
    def __init__(self):
        self.sensitive_patterns = self._initialize_patterns()
        
    def analyze_compliance(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        regulations: List[RegulationType] = None
    ) -> ComplianceReport:
        """
        Perform comprehensive compliance analysis on dataset
        """
        if regulations is None:
            regulations = [RegulationType.GDPR, RegulationType.CCPA]
            
        print(f"🔍 Starting compliance analysis for dataset {dataset_id}")
        
        # Detect sensitive data
        sensitive_data = self._detect_sensitive_data(data)
        
        # Check regulatory compliance
        violations = []
        recommendations = []
        
        for regulation in regulations:
            reg_violations, reg_recommendations = self._check_regulation_compliance(
                data, sensitive_data, regulation
            )
            violations.extend(reg_violations)
            recommendations.extend(reg_recommendations)
        
        # Determine overall compliance status
        status = self._determine_compliance_status(violations)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(violations, len(data))
        
        report = ComplianceReport(
            dataset_id=dataset_id,
            regulations_checked=regulations,
            compliance_status=status,
            compliance_score=compliance_score,
            sensitive_data_detections=sensitive_data,
            violations=violations,
            recommendations=recommendations,
            analysis_timestamp=datetime.now(),
            total_records=len(data),
            sensitive_records_count=self._count_sensitive_records(sensitive_data)
        )
        
        print(f"✅ Compliance analysis completed for dataset {dataset_id}")
        return report
    
    def _detect_sensitive_data(self, data: pd.DataFrame) -> List[SensitiveDataDetection]:
        """Detect sensitive data using pattern matching"""
        detections = []
        
        for column in data.columns:
            column_data = data[column].astype(str)
            
            # Pattern-based detection
            for pattern in self.sensitive_patterns:
                matches = self._find_pattern_matches(column_data, pattern)
                if matches:
                    detection = SensitiveDataDetection(
                        column_name=column,
                        data_type=pattern.data_type,
                        confidence_score=matches['confidence'],
                        sample_values=matches['samples'],
                        detection_method="pattern_matching",
                        pattern_name=pattern.name,
                        affected_rows=matches['row_count']
                    )
                    detections.append(detection)
        
        return detections
    
    def _find_pattern_matches(self, column_data: pd.Series, pattern: SensitiveDataPattern) -> Optional[Dict]:
        """Find matches for a specific sensitive data pattern"""
        try:
            regex = re.compile(pattern.pattern, re.IGNORECASE)
            matches = column_data.str.contains(regex, na=False)
            match_count = matches.sum()
            
            if match_count == 0:
                return None
            
            # Calculate confidence based on match percentage
            confidence = min(match_count / len(column_data), 1.0)
            
            if confidence < pattern.confidence_threshold:
                return None
            
            # Get sample values (anonymized)
            sample_values = column_data[matches].head(3).tolist()
            anonymized_samples = [self._anonymize_sample(val) for val in sample_values]
            
            return {
                'confidence': confidence,
                'samples': anonymized_samples,
                'row_count': match_count
            }
            
        except Exception as e:
            print(f"⚠️ Error in pattern matching for {pattern.name}: {str(e)}")
            return None
    
    def _check_regulation_compliance(
        self, 
        data: pd.DataFrame, 
        sensitive_data: List[SensitiveDataDetection],
        regulation: RegulationType
    ) -> Tuple[List[ComplianceViolation], List[PrivacyRecommendation]]:
        """Check compliance against specific regulation"""
        violations = []
        recommendations = []
        
        if regulation == RegulationType.GDPR:
            gdpr_violations, gdpr_recommendations = self._check_gdpr_compliance(data, sensitive_data)
            violations.extend(gdpr_violations)
            recommendations.extend(gdpr_recommendations)
        
        elif regulation == RegulationType.CCPA:
            ccpa_violations, ccpa_recommendations = self._check_ccpa_compliance(data, sensitive_data)
            violations.extend(ccpa_violations)
            recommendations.extend(ccpa_recommendations)
        
        return violations, recommendations
    
    def _check_gdpr_compliance(
        self, 
        data: pd.DataFrame, 
        sensitive_data: List[SensitiveDataDetection]
    ) -> Tuple[List[ComplianceViolation], List[PrivacyRecommendation]]:
        """Check GDPR compliance requirements"""
        violations = []
        recommendations = []
        
        # Check for personal data without proper safeguards
        personal_data_columns = [
            detection.column_name for detection in sensitive_data 
            if detection.data_type in [SensitiveDataType.PII, SensitiveDataType.CONTACT]
        ]
        
        if personal_data_columns:
            violation = ComplianceViolation(
                regulation=RegulationType.GDPR,
                article="Article 6",
                description="Personal data detected without explicit consent mechanism",
                severity=ViolationSeverity.HIGH,
                affected_columns=personal_data_columns,
                recommendation="Implement consent management and data minimization"
            )
            violations.append(violation)
            
            recommendation = PrivacyRecommendation(
                technique=PrivacyTechnique.PSEUDONYMIZATION,
                description="Apply pseudonymization to personal identifiers",
                affected_columns=personal_data_columns,
                implementation_priority="HIGH",
                estimated_privacy_gain=0.8
            )
            recommendations.append(recommendation)
        
        return violations, recommendations
    
    def _check_ccpa_compliance(
        self, 
        data: pd.DataFrame, 
        sensitive_data: List[SensitiveDataDetection]
    ) -> Tuple[List[ComplianceViolation], List[PrivacyRecommendation]]:
        """Check CCPA compliance requirements"""
        violations = []
        recommendations = []
        
        # Check for personal information
        personal_info_columns = [
            detection.column_name for detection in sensitive_data 
            if detection.data_type in [
                SensitiveDataType.PII, SensitiveDataType.CONTACT, 
                SensitiveDataType.BEHAVIORAL, SensitiveDataType.LOCATION
            ]
        ]
        
        if personal_info_columns:
            violation = ComplianceViolation(
                regulation=RegulationType.CCPA,
                article="Section 1798.140",
                description="Personal information requires consumer rights implementation",
                severity=ViolationSeverity.MEDIUM,
                affected_columns=personal_info_columns,
                recommendation="Implement consumer rights (access, delete, opt-out)"
            )
            violations.append(violation)
            
            recommendation = PrivacyRecommendation(
                technique=PrivacyTechnique.DATA_MINIMIZATION,
                description="Minimize collection of personal information",
                affected_columns=personal_info_columns,
                implementation_priority="MEDIUM",
                estimated_privacy_gain=0.6
            )
            recommendations.append(recommendation)
        
        return violations, recommendations
    
    def _determine_compliance_status(self, violations: List[ComplianceViolation]) -> ComplianceStatus:
        """Determine overall compliance status based on violations"""
        if not violations:
            return ComplianceStatus.COMPLIANT
        
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        high_violations = [v for v in violations if v.severity == ViolationSeverity.HIGH]
        
        if critical_violations:
            return ComplianceStatus.NON_COMPLIANT
        elif high_violations:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NEEDS_REVIEW
    
    def _calculate_compliance_score(self, violations: List[ComplianceViolation], total_records: int) -> float:
        """Calculate numerical compliance score (0-1)"""
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.LOW: 0.1,
            ViolationSeverity.MEDIUM: 0.3,
            ViolationSeverity.HIGH: 0.6,
            ViolationSeverity.CRITICAL: 1.0
        }
        
        total_penalty = sum(severity_weights[v.severity] for v in violations)
        max_possible_penalty = len(violations) * 1.0
        
        # Normalize to 0-1 scale
        compliance_score = max(0.0, 1.0 - (total_penalty / max_possible_penalty))
        return round(compliance_score, 3)
    
    def _count_sensitive_records(self, sensitive_data: List[SensitiveDataDetection]) -> int:
        """Count total records containing sensitive data"""
        if not sensitive_data:
            return 0
        return max(detection.affected_rows for detection in sensitive_data)
    
    def _anonymize_sample(self, value: str) -> str:
        """Anonymize sample values for reporting"""
        if len(value) <= 3:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    
    def _initialize_patterns(self) -> List[SensitiveDataPattern]:
        """Initialize sensitive data detection patterns"""
        return [
            # Email addresses
            SensitiveDataPattern(
                name="email_address",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                data_type=SensitiveDataType.CONTACT,
                confidence_threshold=0.1,
                description="Email address pattern"
            ),
            
            # Phone numbers
            SensitiveDataPattern(
                name="phone_number",
                pattern=r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
                data_type=SensitiveDataType.CONTACT,
                confidence_threshold=0.1,
                description="Phone number pattern"
            ),
            
            # Social Security Numbers
            SensitiveDataPattern(
                name="ssn",
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                data_type=SensitiveDataType.PII,
                confidence_threshold=0.05,
                description="Social Security Number pattern"
            )
        ]


def create_sample_datasets():
    """Create sample datasets for demonstration"""
    
    # Clean dataset (no sensitive data)
    clean_data = pd.DataFrame({
        'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'category': ['Electronics', 'Books', 'Clothing', 'Electronics', 'Books'],
        'price': [299.99, 19.99, 49.99, 399.99, 24.99],
        'rating': [4.5, 4.2, 3.8, 4.7, 4.1]
    })
    
    # Customer dataset with PII
    customer_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'email': [
            'john.doe@email.com', 
            'jane.smith@company.org', 
            'bob.wilson@test.com',
            'alice.brown@example.net'
        ],
        'phone': ['555-123-4567', '555-987-6543', '555-555-5555', '555-111-2222'],
        'ssn': ['123-45-6789', '987-65-4321', '555-44-3333', '111-22-3333'],
        'age': [25, 34, 45, 28]
    })
    
    return {
        'clean': clean_data,
        'customer': customer_data
    }


def print_compliance_report(report):
    """Print formatted compliance report"""
    
    print(f"\n📊 COMPLIANCE ANALYSIS REPORT")
    print(f"Dataset ID: {report.dataset_id}")
    print(f"Analysis Date: {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Regulations Checked: {', '.join([reg.value.upper() for reg in report.regulations_checked])}")
    print()
    
    # Overall status
    status_emoji = {
        'compliant': '✅',
        'partially_compliant': '⚠️',
        'non_compliant': '❌',
        'needs_review': '🔍'
    }
    
    print(f"🎯 COMPLIANCE STATUS: {status_emoji.get(report.compliance_status.value, '❓')} {report.compliance_status.value.upper()}")
    print(f"📈 COMPLIANCE SCORE: {report.compliance_score:.1%}")
    print(f"📋 TOTAL RECORDS: {report.total_records:,}")
    print(f"🔒 SENSITIVE RECORDS: {report.sensitive_records_count:,}")
    print()
    
    # Sensitive data detections
    if report.sensitive_data_detections:
        print("🔍 SENSITIVE DATA DETECTED:")
        for detection in report.sensitive_data_detections:
            print(f"  • Column: {detection.column_name}")
            print(f"    Type: {detection.data_type.value}")
            print(f"    Confidence: {detection.confidence_score:.1%}")
            print(f"    Affected Rows: {detection.affected_rows:,}")
            print(f"    Sample: {', '.join(detection.sample_values[:2])}")
            print()
    else:
        print("✅ NO SENSITIVE DATA DETECTED")
        print()
    
    # Compliance violations
    if report.violations:
        print("⚠️ COMPLIANCE VIOLATIONS:")
        for i, violation in enumerate(report.violations, 1):
            severity_emoji = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }
            
            print(f"  {i}. {severity_emoji.get(violation.severity.value, '❓')} {violation.regulation.value.upper()} - {violation.article}")
            print(f"     Severity: {violation.severity.value.upper()}")
            print(f"     Description: {violation.description}")
            print(f"     Affected Columns: {', '.join(violation.affected_columns)}")
            print(f"     Recommendation: {violation.recommendation}")
            print()
    else:
        print("✅ NO COMPLIANCE VIOLATIONS FOUND")
        print()
    
    # Privacy recommendations
    if report.recommendations:
        print("💡 PRIVACY RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            priority_emoji = {
                'LOW': '🟢',
                'MEDIUM': '🟡',
                'HIGH': '🟠',
                'CRITICAL': '🔴'
            }
            
            print(f"  {i}. {priority_emoji.get(rec.implementation_priority, '❓')} {rec.technique.value.upper()}")
            print(f"     Priority: {rec.implementation_priority}")
            print(f"     Description: {rec.description}")
            print(f"     Affected Columns: {', '.join(rec.affected_columns)}")
            print(f"     Privacy Gain: {rec.estimated_privacy_gain:.1%}")
            print()
    else:
        print("ℹ️ NO PRIVACY RECOMMENDATIONS NEEDED")
        print()


def main():
    """Main demo function"""
    print("=" * 80)
    print("AI DATA READINESS PLATFORM - COMPLIANCE ANALYZER DEMO")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    analyzer = ComplianceAnalyzer()
    datasets = create_sample_datasets()
    
    # Analyze each dataset
    for dataset_name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Perform compliance analysis
        report = analyzer.analyze_compliance(
            dataset_id=f"{dataset_name}_dataset",
            data=data,
            regulations=[RegulationType.GDPR, RegulationType.CCPA]
        )
        
        # Display results
        print_compliance_report(report)
    
    print("\n" + "="*80)
    print("✅ COMPLIANCE ANALYZER DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("• Automated sensitive data detection")
    print("• GDPR and CCPA compliance analysis")
    print("• Privacy-preserving recommendations")
    print("• Comprehensive reporting")


if __name__ == "__main__":
    try:
        print("Starting demo...")
        main()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()