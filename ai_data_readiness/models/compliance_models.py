"""
Data models for compliance analysis and reporting

This module defines the data structures used for regulatory compliance
analysis, violation tracking, and privacy recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import json


class RegulationType(Enum):
    """Supported regulatory frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PIPEDA = "pipeda"
    LGPD = "lgpd"
    PDPA = "pdpa"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Overall compliance status"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    UNKNOWN = "unknown"


class PrivacyTechnique(Enum):
    """Privacy-preserving techniques"""
    ANONYMIZATION = "anonymization"
    PSEUDONYMIZATION = "pseudonymization"
    DATA_MINIMIZATION = "data_minimization"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"
    ENCRYPTION = "encryption"
    TOKENIZATION = "tokenization"
    MASKING = "masking"
    SYNTHETIC_DATA = "synthetic_data"


class SensitiveDataType(Enum):
    """Types of sensitive data"""
    PII = "personally_identifiable_information"
    PHI = "protected_health_information"
    FINANCIAL = "financial_information"
    BIOMETRIC = "biometric_data"
    LOCATION = "location_data"
    BEHAVIORAL = "behavioral_data"
    DEMOGRAPHIC = "demographic_data"
    CONTACT = "contact_information"
    GENETIC = "genetic_information"
    POLITICAL = "political_opinions"
    RELIGIOUS = "religious_beliefs"


@dataclass
class SensitiveDataDetection:
    """Detection of sensitive data in a dataset column"""
    column_name: str
    data_type: SensitiveDataType
    confidence_score: float
    sample_values: List[str]
    detection_method: str
    pattern_name: str
    affected_rows: int
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'column_name': self.column_name,
            'data_type': self.data_type.value,
            'confidence_score': self.confidence_score,
            'sample_values': self.sample_values,
            'detection_method': self.detection_method,
            'pattern_name': self.pattern_name,
            'affected_rows': self.affected_rows,
            'additional_metadata': self.additional_metadata
        }


@dataclass
class ComplianceViolation:
    """A specific compliance violation found in the dataset"""
    regulation: RegulationType
    article: str
    description: str
    severity: ViolationSeverity
    affected_columns: List[str]
    recommendation: str
    violation_id: Optional[str] = None
    legal_basis_missing: bool = False
    consent_required: bool = False
    data_subject_rights_impact: List[str] = field(default_factory=list)
    potential_fine_range: Optional[str] = None
    remediation_effort: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'violation_id': self.violation_id,
            'regulation': self.regulation.value,
            'article': self.article,
            'description': self.description,
            'severity': self.severity.value,
            'affected_columns': self.affected_columns,
            'recommendation': self.recommendation,
            'legal_basis_missing': self.legal_basis_missing,
            'consent_required': self.consent_required,
            'data_subject_rights_impact': self.data_subject_rights_impact,
            'potential_fine_range': self.potential_fine_range,
            'remediation_effort': self.remediation_effort
        }


@dataclass
class PrivacyRecommendation:
    """Recommendation for privacy-preserving techniques"""
    technique: PrivacyTechnique
    description: str
    affected_columns: List[str]
    implementation_priority: str
    estimated_privacy_gain: float
    implementation_complexity: Optional[str] = None
    performance_impact: Optional[str] = None
    utility_preservation: Optional[float] = None
    regulatory_compliance: List[RegulationType] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'technique': self.technique.value,
            'description': self.description,
            'affected_columns': self.affected_columns,
            'implementation_priority': self.implementation_priority,
            'estimated_privacy_gain': self.estimated_privacy_gain,
            'implementation_complexity': self.implementation_complexity,
            'performance_impact': self.performance_impact,
            'utility_preservation': self.utility_preservation,
            'regulatory_compliance': [reg.value for reg in self.regulatory_compliance],
            'implementation_steps': self.implementation_steps
        }


@dataclass
class ComplianceMetrics:
    """Metrics for compliance assessment"""
    total_columns: int
    sensitive_columns: int
    compliant_columns: int
    violation_count_by_severity: Dict[ViolationSeverity, int]
    privacy_techniques_recommended: int
    estimated_remediation_time: Optional[str] = None
    risk_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_columns': self.total_columns,
            'sensitive_columns': self.sensitive_columns,
            'compliant_columns': self.compliant_columns,
            'violation_count_by_severity': {k.value: v for k, v in self.violation_count_by_severity.items()},
            'privacy_techniques_recommended': self.privacy_techniques_recommended,
            'estimated_remediation_time': self.estimated_remediation_time,
            'risk_score': self.risk_score
        }


@dataclass
class ComplianceReport:
    """Comprehensive compliance analysis report"""
    dataset_id: str
    regulations_checked: List[RegulationType]
    compliance_status: ComplianceStatus
    compliance_score: float
    sensitive_data_detections: List[SensitiveDataDetection]
    violations: List[ComplianceViolation]
    recommendations: List[PrivacyRecommendation]
    analysis_timestamp: datetime
    total_records: int
    sensitive_records_count: int
    metrics: Optional[ComplianceMetrics] = None
    executive_summary: Optional[str] = None
    next_review_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'dataset_id': self.dataset_id,
            'regulations_checked': [reg.value for reg in self.regulations_checked],
            'compliance_status': self.compliance_status.value,
            'compliance_score': self.compliance_score,
            'sensitive_data_detections': [detection.to_dict() for detection in self.sensitive_data_detections],
            'violations': [violation.to_dict() for violation in self.violations],
            'recommendations': [rec.to_dict() for rec in self.recommendations],
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'total_records': self.total_records,
            'sensitive_records_count': self.sensitive_records_count,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'executive_summary': self.executive_summary,
            'next_review_date': self.next_review_date.isoformat() if self.next_review_date else None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def get_critical_violations(self) -> List[ComplianceViolation]:
        """Get only critical violations"""
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]
    
    def get_high_priority_recommendations(self) -> List[PrivacyRecommendation]:
        """Get high priority recommendations"""
        return [r for r in self.recommendations if r.implementation_priority in ['HIGH', 'CRITICAL']]
    
    def is_compliant(self) -> bool:
        """Check if dataset is fully compliant"""
        return self.compliance_status == ComplianceStatus.COMPLIANT
    
    def requires_immediate_action(self) -> bool:
        """Check if dataset requires immediate remediation"""
        return (
            self.compliance_status == ComplianceStatus.NON_COMPLIANT or
            len(self.get_critical_violations()) > 0
        )


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA) results"""
    dataset_id: str
    assessment_date: datetime
    risk_level: str
    privacy_risks: List[str]
    mitigation_measures: List[str]
    residual_risks: List[str]
    approval_status: str
    reviewer: Optional[str] = None
    review_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'dataset_id': self.dataset_id,
            'assessment_date': self.assessment_date.isoformat(),
            'risk_level': self.risk_level,
            'privacy_risks': self.privacy_risks,
            'mitigation_measures': self.mitigation_measures,
            'residual_risks': self.residual_risks,
            'approval_status': self.approval_status,
            'reviewer': self.reviewer,
            'review_date': self.review_date.isoformat() if self.review_date else None
        }


@dataclass
class ConsentRecord:
    """Record of data subject consent"""
    subject_id: str
    consent_type: str
    purpose: str
    granted_date: datetime
    expiry_date: Optional[datetime]
    withdrawn_date: Optional[datetime]
    consent_mechanism: str
    legal_basis: str
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid"""
        now = datetime.utcnow()
        if self.withdrawn_date and self.withdrawn_date <= now:
            return False
        if self.expiry_date and self.expiry_date <= now:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'subject_id': self.subject_id,
            'consent_type': self.consent_type,
            'purpose': self.purpose,
            'granted_date': self.granted_date.isoformat(),
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'withdrawn_date': self.withdrawn_date.isoformat() if self.withdrawn_date else None,
            'consent_mechanism': self.consent_mechanism,
            'legal_basis': self.legal_basis,
            'is_valid': self.is_valid()
        }


# Utility functions for working with compliance models

def create_compliance_metrics(
    total_columns: int,
    sensitive_detections: List[SensitiveDataDetection],
    violations: List[ComplianceViolation],
    recommendations: List[PrivacyRecommendation]
) -> ComplianceMetrics:
    """Create compliance metrics from analysis results"""
    
    sensitive_columns = len(set(detection.column_name for detection in sensitive_detections))
    compliant_columns = total_columns - sensitive_columns
    
    violation_counts = {severity: 0 for severity in ViolationSeverity}
    for violation in violations:
        violation_counts[violation.severity] += 1
    
    return ComplianceMetrics(
        total_columns=total_columns,
        sensitive_columns=sensitive_columns,
        compliant_columns=compliant_columns,
        violation_count_by_severity=violation_counts,
        privacy_techniques_recommended=len(recommendations)
    )


def generate_executive_summary(report: ComplianceReport) -> str:
    """Generate executive summary for compliance report"""
    
    status_descriptions = {
        ComplianceStatus.COMPLIANT: "fully compliant",
        ComplianceStatus.PARTIALLY_COMPLIANT: "partially compliant with some issues",
        ComplianceStatus.NON_COMPLIANT: "non-compliant with critical violations",
        ComplianceStatus.NEEDS_REVIEW: "requires review for minor issues"
    }
    
    summary_parts = [
        f"Dataset {report.dataset_id} is {status_descriptions.get(report.compliance_status, 'unknown status')}.",
        f"Compliance score: {report.compliance_score:.1%}",
        f"Found {len(report.sensitive_data_detections)} sensitive data types across {report.sensitive_records_count} records.",
        f"Identified {len(report.violations)} compliance violations.",
        f"Provided {len(report.recommendations)} privacy-preserving recommendations."
    ]
    
    if report.requires_immediate_action():
        summary_parts.append("IMMEDIATE ACTION REQUIRED due to critical violations.")
    
    return " ".join(summary_parts)