"""
Vendor Security Assessment and Due Diligence Automation
Implements automated vendor security assessment workflows
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import requests
from pathlib import Path

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AssessmentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class VendorProfile:
    vendor_id: str
    name: str
    contact_email: str
    business_type: str
    services_provided: List[str]
    data_access_level: str
    compliance_certifications: List[str]
    created_at: datetime
    last_updated: datetime

@dataclass
class SecurityAssessment:
    assessment_id: str
    vendor_id: str
    assessment_type: str
    status: AssessmentStatus
    risk_score: float
    risk_level: RiskLevel
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    compliance_gaps: List[str]
    next_review_date: datetime
    assessor: str
    created_at: datetime
    completed_at: Optional[datetime] = None

class VendorSecurityAssessor:
    def __init__(self, config_path: str = "security/config/vendor_assessment_config.yaml"):
        self.config = self._load_config(config_path)
        self.assessment_templates = self._load_assessment_templates()
        self.compliance_frameworks = self._load_compliance_frameworks()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load vendor assessment configuration"""
        default_config = {
            "assessment_intervals": {
                "critical_vendors": 90,  # days
                "high_risk_vendors": 180,
                "standard_vendors": 365
            },
            "risk_thresholds": {
                "critical": 8.0,
                "high": 6.0,
                "medium": 4.0,
                "low": 0.0
            },
            "required_certifications": [
                "SOC2_TYPE_II",
                "ISO27001",
                "PCI_DSS"
            ],
            "assessment_criteria": {
                "data_security": 0.3,
                "access_controls": 0.25,
                "incident_response": 0.2,
                "compliance": 0.15,
                "business_continuity": 0.1
            }
        }
        
        try:
            # In production, load from actual config file
            return default_config
        except Exception:
            return default_config
    
    def _load_assessment_templates(self) -> Dict[str, Any]:
        """Load assessment question templates"""
        return {
            "data_security": [
                {
                    "question": "Does the vendor encrypt data at rest using AES-256 or equivalent?",
                    "weight": 0.4,
                    "required": True
                },
                {
                    "question": "Is data encrypted in transit using TLS 1.3 or higher?",
                    "weight": 0.3,
                    "required": True
                },
                {
                    "question": "Are encryption keys managed using HSM or equivalent?",
                    "weight": 0.3,
                    "required": False
                }
            ],
            "access_controls": [
                {
                    "question": "Is multi-factor authentication enforced for all users?",
                    "weight": 0.4,
                    "required": True
                },
                {
                    "question": "Are privileged access controls implemented with just-in-time access?",
                    "weight": 0.3,
                    "required": True
                },
                {
                    "question": "Is role-based access control (RBAC) implemented?",
                    "weight": 0.3,
                    "required": True
                }
            ],
            "incident_response": [
                {
                    "question": "Does the vendor have a documented incident response plan?",
                    "weight": 0.4,
                    "required": True
                },
                {
                    "question": "Are security incidents reported within 24 hours?",
                    "weight": 0.3,
                    "required": True
                },
                {
                    "question": "Is there a dedicated security operations center (SOC)?",
                    "weight": 0.3,
                    "required": False
                }
            ]
        }
    
    def _load_compliance_frameworks(self) -> Dict[str, Any]:
        """Load compliance framework requirements"""
        return {
            "SOC2_TYPE_II": {
                "required_controls": [
                    "CC6.1", "CC6.2", "CC6.3", "CC6.7", "CC6.8"
                ],
                "validity_period": 365
            },
            "ISO27001": {
                "required_controls": [
                    "A.9.1.1", "A.9.2.1", "A.12.6.1", "A.13.1.1"
                ],
                "validity_period": 1095
            },
            "PCI_DSS": {
                "required_controls": [
                    "1.1", "2.1", "3.4", "8.1", "11.2"
                ],
                "validity_period": 365
            }
        }
    
    async def assess_vendor(self, vendor_profile: VendorProfile) -> SecurityAssessment:
        """Perform comprehensive vendor security assessment"""
        assessment_id = self._generate_assessment_id(vendor_profile.vendor_id)
        
        # Initialize assessment
        assessment = SecurityAssessment(
            assessment_id=assessment_id,
            vendor_id=vendor_profile.vendor_id,
            assessment_type="comprehensive",
            status=AssessmentStatus.IN_PROGRESS,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            findings=[],
            recommendations=[],
            compliance_gaps=[],
            next_review_date=datetime.now() + timedelta(days=365),
            assessor="automated_system",
            created_at=datetime.now()
        )
        
        try:
            # Perform assessment components
            security_findings = await self._assess_security_controls(vendor_profile)
            compliance_findings = await self._assess_compliance(vendor_profile)
            risk_findings = await self._assess_risk_factors(vendor_profile)
            
            # Aggregate findings
            assessment.findings.extend(security_findings)
            assessment.findings.extend(compliance_findings)
            assessment.findings.extend(risk_findings)
            
            # Calculate risk score
            assessment.risk_score = self._calculate_risk_score(assessment.findings)
            assessment.risk_level = self._determine_risk_level(assessment.risk_score)
            
            # Generate recommendations
            assessment.recommendations = self._generate_recommendations(assessment.findings)
            assessment.compliance_gaps = self._identify_compliance_gaps(compliance_findings)
            
            # Set next review date based on risk level
            assessment.next_review_date = self._calculate_next_review_date(assessment.risk_level)
            
            assessment.status = AssessmentStatus.COMPLETED
            assessment.completed_at = datetime.now()
            
        except Exception as e:
            assessment.status = AssessmentStatus.FAILED
            assessment.findings.append({
                "category": "system_error",
                "severity": "high",
                "description": f"Assessment failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return assessment
    
    async def _assess_security_controls(self, vendor_profile: VendorProfile) -> List[Dict[str, Any]]:
        """Assess vendor security controls"""
        findings = []
        
        for category, questions in self.assessment_templates.items():
            category_score = 0.0
            category_findings = []
            
            for question in questions:
                # Simulate assessment (in production, this would involve actual checks)
                response = await self._evaluate_security_question(
                    vendor_profile, question
                )
                
                category_findings.append({
                    "question": question["question"],
                    "response": response,
                    "weight": question["weight"],
                    "required": question["required"]
                })
                
                category_score += response * question["weight"]
            
            findings.append({
                "category": category,
                "score": category_score,
                "findings": category_findings,
                "timestamp": datetime.now().isoformat()
            })
        
        return findings
    
    async def _assess_compliance(self, vendor_profile: VendorProfile) -> List[Dict[str, Any]]:
        """Assess vendor compliance status"""
        findings = []
        
        for cert in vendor_profile.compliance_certifications:
            if cert in self.compliance_frameworks:
                framework = self.compliance_frameworks[cert]
                
                # Check certification validity
                is_valid = await self._verify_certification(vendor_profile.vendor_id, cert)
                
                findings.append({
                    "category": "compliance",
                    "certification": cert,
                    "valid": is_valid,
                    "required_controls": framework["required_controls"],
                    "validity_period": framework["validity_period"],
                    "timestamp": datetime.now().isoformat()
                })
        
        return findings
    
    async def _assess_risk_factors(self, vendor_profile: VendorProfile) -> List[Dict[str, Any]]:
        """Assess vendor risk factors"""
        findings = []
        
        # Data access level risk
        data_risk = self._assess_data_access_risk(vendor_profile.data_access_level)
        findings.append({
            "category": "data_risk",
            "level": vendor_profile.data_access_level,
            "risk_score": data_risk,
            "timestamp": datetime.now().isoformat()
        })
        
        # Business type risk
        business_risk = self._assess_business_type_risk(vendor_profile.business_type)
        findings.append({
            "category": "business_risk",
            "type": vendor_profile.business_type,
            "risk_score": business_risk,
            "timestamp": datetime.now().isoformat()
        })
        
        # Service criticality risk
        service_risk = self._assess_service_criticality(vendor_profile.services_provided)
        findings.append({
            "category": "service_risk",
            "services": vendor_profile.services_provided,
            "risk_score": service_risk,
            "timestamp": datetime.now().isoformat()
        })
        
        return findings
    
    async def _evaluate_security_question(self, vendor_profile: VendorProfile, question: Dict[str, Any]) -> float:
        """Evaluate a security assessment question"""
        # In production, this would involve actual security checks
        # For now, simulate based on vendor profile
        
        if "encryption" in question["question"].lower():
            return 0.9 if "security" in vendor_profile.services_provided else 0.6
        elif "authentication" in question["question"].lower():
            return 0.8 if len(vendor_profile.compliance_certifications) > 2 else 0.5
        elif "incident" in question["question"].lower():
            return 0.7 if "SOC2_TYPE_II" in vendor_profile.compliance_certifications else 0.4
        else:
            return 0.6  # Default moderate score
    
    async def _verify_certification(self, vendor_id: str, certification: str) -> bool:
        """Verify vendor certification status"""
        # In production, this would check with certification authorities
        # For now, simulate verification
        return True
    
    def _assess_data_access_risk(self, data_access_level: str) -> float:
        """Assess risk based on data access level"""
        risk_mapping = {
            "none": 0.1,
            "public": 0.2,
            "internal": 0.5,
            "confidential": 0.8,
            "restricted": 1.0
        }
        return risk_mapping.get(data_access_level.lower(), 0.5)
    
    def _assess_business_type_risk(self, business_type: str) -> float:
        """Assess risk based on business type"""
        high_risk_types = ["cloud_provider", "data_processor", "payment_processor"]
        medium_risk_types = ["software_vendor", "consulting", "support"]
        
        if business_type.lower() in high_risk_types:
            return 0.8
        elif business_type.lower() in medium_risk_types:
            return 0.5
        else:
            return 0.3
    
    def _assess_service_criticality(self, services: List[str]) -> float:
        """Assess risk based on service criticality"""
        critical_services = ["data_processing", "authentication", "payment", "backup"]
        
        critical_count = sum(1 for service in services if any(
            critical in service.lower() for critical in critical_services
        ))
        
        return min(critical_count * 0.2, 1.0)
    
    def _calculate_risk_score(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score from findings"""
        total_score = 0.0
        weight_sum = 0.0
        
        for finding in findings:
            if finding.get("category") == "data_risk":
                total_score += finding["risk_score"] * 0.4
                weight_sum += 0.4
            elif finding.get("category") == "business_risk":
                total_score += finding["risk_score"] * 0.3
                weight_sum += 0.3
            elif finding.get("category") == "service_risk":
                total_score += finding["risk_score"] * 0.3
                weight_sum += 0.3
            elif finding.get("score") is not None:
                # Security control scores (inverted - lower score = higher risk)
                risk_contribution = (1.0 - finding["score"]) * 0.5
                total_score += risk_contribution
                weight_sum += 0.5
        
        return (total_score / weight_sum * 10) if weight_sum > 0 else 5.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        thresholds = self.config["risk_thresholds"]
        
        if risk_score >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        for finding in findings:
            if finding.get("category") == "compliance" and not finding.get("valid", True):
                recommendations.append(
                    f"Renew or obtain {finding['certification']} certification"
                )
            
            if finding.get("score", 1.0) < 0.7:
                recommendations.append(
                    f"Improve {finding.get('category', 'security')} controls"
                )
            
            if finding.get("risk_score", 0.0) > 0.7:
                recommendations.append(
                    f"Implement additional controls for {finding.get('category', 'identified')} risks"
                )
        
        # Add standard recommendations
        recommendations.extend([
            "Implement continuous security monitoring",
            "Establish regular security review meetings",
            "Require security incident notification within 24 hours",
            "Implement data encryption for all sensitive data"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_compliance_gaps(self, compliance_findings: List[Dict[str, Any]]) -> List[str]:
        """Identify compliance gaps"""
        gaps = []
        
        for finding in compliance_findings:
            if finding.get("category") == "compliance" and not finding.get("valid", True):
                gaps.append(f"Missing or expired {finding['certification']} certification")
        
        return gaps
    
    def _calculate_next_review_date(self, risk_level: RiskLevel) -> datetime:
        """Calculate next review date based on risk level"""
        intervals = self.config["assessment_intervals"]
        
        if risk_level == RiskLevel.CRITICAL:
            days = intervals["critical_vendors"]
        elif risk_level == RiskLevel.HIGH:
            days = intervals["high_risk_vendors"]
        else:
            days = intervals["standard_vendors"]
        
        return datetime.now() + timedelta(days=days)
    
    def _generate_assessment_id(self, vendor_id: str) -> str:
        """Generate unique assessment ID"""
        timestamp = datetime.now().isoformat()
        content = f"{vendor_id}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def generate_assessment_report(self, assessment: SecurityAssessment) -> Dict[str, Any]:
        """Generate comprehensive assessment report"""
        return {
            "assessment_summary": {
                "assessment_id": assessment.assessment_id,
                "vendor_id": assessment.vendor_id,
                "risk_level": assessment.risk_level.value,
                "risk_score": assessment.risk_score,
                "status": assessment.status.value,
                "completed_at": assessment.completed_at.isoformat() if assessment.completed_at else None
            },
            "findings": assessment.findings,
            "recommendations": assessment.recommendations,
            "compliance_gaps": assessment.compliance_gaps,
            "next_review_date": assessment.next_review_date.isoformat(),
            "generated_at": datetime.now().isoformat()
        }
    
    async def schedule_reassessment(self, vendor_id: str, assessment_date: datetime) -> bool:
        """Schedule vendor reassessment"""
        try:
            # In production, this would integrate with scheduling system
            print(f"Scheduled reassessment for vendor {vendor_id} on {assessment_date}")
            return True
        except Exception as e:
            print(f"Failed to schedule reassessment: {e}")
            return False