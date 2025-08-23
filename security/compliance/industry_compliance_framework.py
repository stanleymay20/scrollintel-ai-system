"""
Industry-Tailored Compliance Framework
Provides specialized compliance modules for different industries
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REMEDIATION_REQUIRED = "remediation_required"

class IndustryType(Enum):
    BANKING = "banking"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    GOVERNMENT = "government"
    FINANCIAL_SERVICES = "financial_services"
    PHARMACEUTICAL = "pharmaceutical"

@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    description: str
    regulation: str
    severity: str
    automated_check: bool
    remediation_steps: List[str]
    evidence_requirements: List[str]

@dataclass
class ComplianceAssessment:
    assessment_id: str
    industry: IndustryType
    timestamp: datetime
    rules_evaluated: List[str]
    compliant_rules: List[str]
    non_compliant_rules: List[str]
    overall_status: ComplianceStatus
    risk_score: float
    recommendations: List[str]

class BaseComplianceModule(ABC):
    """Base class for industry-specific compliance modules"""
    
    def __init__(self, industry: IndustryType):
        self.industry = industry
        self.rules: Dict[str, ComplianceRule] = {}
        self.assessments: List[ComplianceAssessment] = []
        self._load_rules()
    
    @abstractmethod
    def _load_rules(self):
        """Load industry-specific compliance rules"""
        pass
    
    @abstractmethod
    def assess_compliance(self, system_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess compliance against industry regulations"""
        pass
    
    @abstractmethod
    def generate_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate compliance report"""
        pass
    
    def get_rule(self, rule_id: str) -> Optional[ComplianceRule]:
        """Get specific compliance rule"""
        return self.rules.get(rule_id)
    
    def get_all_rules(self) -> List[ComplianceRule]:
        """Get all compliance rules for this industry"""
        return list(self.rules.values())
    
    def calculate_risk_score(self, non_compliant_rules: List[str]) -> float:
        """Calculate overall risk score based on non-compliant rules"""
        if not non_compliant_rules:
            return 0.0
        
        total_weight = 0
        risk_score = 0
        
        for rule_id in non_compliant_rules:
            rule = self.rules.get(rule_id)
            if rule:
                weight = self._get_severity_weight(rule.severity)
                total_weight += weight
                risk_score += weight
        
        return min(risk_score / max(total_weight, 1) * 100, 100.0)
    
    def _get_severity_weight(self, severity: str) -> float:
        """Get weight for severity level"""
        weights = {
            "critical": 10.0,
            "high": 7.0,
            "medium": 4.0,
            "low": 1.0
        }
        return weights.get(severity.lower(), 1.0)

class IndustryComplianceOrchestrator:
    """Orchestrates compliance across multiple industries"""
    
    def __init__(self):
        self.modules: Dict[IndustryType, BaseComplianceModule] = {}
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all industry compliance modules"""
        from .banking_compliance import BankingComplianceModule
        from .healthcare_compliance import HealthcareComplianceModule
        from .manufacturing_compliance import ManufacturingComplianceModule
        from .government_compliance import GovernmentComplianceModule
        from .financial_services_compliance import FinancialServicesComplianceModule
        from .pharmaceutical_compliance import PharmaceuticalComplianceModule
        
        self.modules[IndustryType.BANKING] = BankingComplianceModule()
        self.modules[IndustryType.HEALTHCARE] = HealthcareComplianceModule()
        self.modules[IndustryType.MANUFACTURING] = ManufacturingComplianceModule()
        self.modules[IndustryType.GOVERNMENT] = GovernmentComplianceModule()
        self.modules[IndustryType.FINANCIAL_SERVICES] = FinancialServicesComplianceModule()
        self.modules[IndustryType.PHARMACEUTICAL] = PharmaceuticalComplianceModule()
    
    def assess_industry_compliance(self, industry: IndustryType, system_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess compliance for specific industry"""
        module = self.modules.get(industry)
        if not module:
            raise ValueError(f"No compliance module found for industry: {industry}")
        
        return module.assess_compliance(system_data)
    
    def get_industry_rules(self, industry: IndustryType) -> List[ComplianceRule]:
        """Get all rules for specific industry"""
        module = self.modules.get(industry)
        if not module:
            raise ValueError(f"No compliance module found for industry: {industry}")
        
        return module.get_all_rules()
    
    def generate_cross_industry_report(self, assessments: List[ComplianceAssessment]) -> Dict[str, Any]:
        """Generate comprehensive report across multiple industries"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "industries_assessed": len(assessments),
            "overall_compliance_rate": 0.0,
            "industry_breakdown": {},
            "critical_findings": [],
            "recommendations": []
        }
        
        total_rules = 0
        compliant_rules = 0
        
        for assessment in assessments:
            industry_name = assessment.industry.value
            total_industry_rules = len(assessment.rules_evaluated)
            compliant_industry_rules = len(assessment.compliant_rules)
            
            total_rules += total_industry_rules
            compliant_rules += compliant_industry_rules
            
            report["industry_breakdown"][industry_name] = {
                "status": assessment.overall_status.value,
                "compliance_rate": (compliant_industry_rules / max(total_industry_rules, 1)) * 100,
                "risk_score": assessment.risk_score,
                "non_compliant_rules": len(assessment.non_compliant_rules)
            }
            
            # Add critical findings
            if assessment.overall_status == ComplianceStatus.NON_COMPLIANT:
                report["critical_findings"].append({
                    "industry": industry_name,
                    "risk_score": assessment.risk_score,
                    "non_compliant_rules": assessment.non_compliant_rules
                })
        
        report["overall_compliance_rate"] = (compliant_rules / max(total_rules, 1)) * 100
        
        return report