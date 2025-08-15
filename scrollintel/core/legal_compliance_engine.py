"""
Legal Compliance Engine - Core compliance validation and enforcement system
Ensures ScrollIntel operates within all applicable laws and regulations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4

logger = logging.getLogger(__name__)


class ComplianceJurisdiction(Enum):
    """Supported legal jurisdictions"""
    US_FEDERAL = "us_federal"
    EU = "european_union"
    UK = "united_kingdom"
    CANADA = "canada"
    AUSTRALIA = "australia"
    CALIFORNIA = "california"
    NEW_YORK = "new_york"


class ComplianceArea(Enum):
    """Areas of legal compliance"""
    EMPLOYMENT_LAW = "employment_law"
    DATA_PRIVACY = "data_privacy"
    AI_REGULATION = "ai_regulation"
    FINANCIAL_REGULATION = "financial_regulation"
    PROFESSIONAL_LICENSING = "professional_licensing"
    ANTITRUST = "antitrust"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    CONSUMER_PROTECTION = "consumer_protection"
    INTERNATIONAL_TRADE = "international_trade"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    IN_PROGRESS = "in_progress"


@dataclass
class ComplianceContext:
    """Context for compliance validation"""
    jurisdiction: ComplianceJurisdiction
    user_location: str
    data_types: List[str]
    operation_type: str
    risk_level: str
    applicable_regulations: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
@data
class
class ComplianceViolation:
    """Represents a compliance violation"""
    violation_id: str
    regulation: str
    severity: ViolationSeverity
    description: str
    remediation_steps: List[str]
    deadline: datetime
    status: str = "open"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ComplianceResult:
    """Result of compliance validation"""
    is_compliant: bool
    violations: List[ComplianceViolation]
    warnings: List[str]
    required_actions: List[str]
    risk_assessment: Dict[str, Any]
    compliance_score: float
    status: ComplianceStatus


class LegalComplianceEngine:
    """Core legal compliance engine for ScrollIntel"""
    
    def __init__(self):
        self.jurisdiction_managers = {}
        self.compliance_rules = {}
        self.violation_handlers = {}
        self.active_violations = {}
        self.compliance_history = []
        
        # Initialize compliance systems
        self._initialize_jurisdiction_managers()
        self._initialize_compliance_rules()
        self._initialize_violation_handlers()
    
    def _initialize_jurisdiction_managers(self):
        """Initialize jurisdiction-specific compliance managers"""
        from .jurisdiction_manager import JurisdictionManager
        
        for jurisdiction in ComplianceJurisdiction:
            self.jurisdiction_managers[jurisdiction] = JurisdictionManager(jurisdiction)
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rule engines"""
        from .compliance_rule_engine import ComplianceRuleEngine
        
        for area in ComplianceArea:
            self.compliance_rules[area] = ComplianceRuleEngine(area)
    
    def _initialize_violation_handlers(self):
        """Initialize violation detection and handling"""
        from .violation_detector import ViolationDetector
        from .remediation_orchestrator import RemediationOrchestrator
        
        self.violation_detector = ViolationDetector()
        self.remediation_orchestrator = RemediationOrchestrator()
    
    async def validate_operation(self, operation: Dict[str, Any], context: ComplianceContext) -> ComplianceResult:
        """Validate an operation for legal compliance"""
        try:
            logger.info(f"Validating operation for compliance: {operation.get('type', 'unknown')}")
            
            # Get jurisdiction-specific requirements
            jurisdiction_manager = self.jurisdiction_managers[context.jurisdiction]
            requirements = await jurisdiction_manager.get_requirements(context)
            
            # Run compliance checks for each applicable area
            violations = []
            warnings = []
            required_actions = []
            
            for area in ComplianceArea:
                if area.value in context.applicable_regulations:
                    rule_engine = self.compliance_rules[area]
                    area_result = await rule_engine.validate(operation, context, requirements)
                    
                    violations.extend(area_result.get('violations', []))
                    warnings.extend(area_result.get('warnings', []))
                    required_actions.extend(area_result.get('actions', []))
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(violations, warnings)
            
            # Determine overall compliance status
            status = self._determine_compliance_status(compliance_score, violations)
            
            # Create risk assessment
            risk_assessment = await self._assess_legal_risk(violations, context)
            
            result = ComplianceResult(
                is_compliant=(status == ComplianceStatus.COMPLIANT),
                violations=violations,
                warnings=warnings,
                required_actions=required_actions,
                risk_assessment=risk_assessment,
                compliance_score=compliance_score,
                status=status
            )
            
            # Store compliance history
            self.compliance_history.append({
                'timestamp': datetime.utcnow(),
                'operation': operation,
                'context': context,
                'result': result
            })
            
            # Trigger automatic remediation if needed
            if violations:
                await self._trigger_remediation(violations, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Compliance validation failed: {str(e)}")
            raise    
async def check_jurisdiction_compliance(self, jurisdiction: ComplianceJurisdiction, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance for a specific jurisdiction"""
        try:
            jurisdiction_manager = self.jurisdiction_managers[jurisdiction]
            return await jurisdiction_manager.check_compliance(operation)
        except Exception as e:
            logger.error(f"Jurisdiction compliance check failed for {jurisdiction}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def assess_legal_risk(self, action: Dict[str, Any], context: ComplianceContext) -> Dict[str, Any]:
        """Assess legal risk for a proposed action"""
        try:
            # Analyze potential violations
            potential_violations = await self.violation_detector.detect_potential_violations(action, context)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(potential_violations, context)
            
            # Generate risk mitigation recommendations
            mitigation_recommendations = await self._generate_risk_mitigation(potential_violations, context)
            
            return {
                "risk_score": risk_score,
                "risk_level": self._categorize_risk_level(risk_score),
                "potential_violations": potential_violations,
                "mitigation_recommendations": mitigation_recommendations,
                "assessment_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Legal risk assessment failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def generate_compliance_report(self, scope: str, timeframe: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        try:
            start_time, end_time = timeframe
            
            # Filter compliance history by timeframe
            relevant_history = [
                entry for entry in self.compliance_history
                if start_time <= entry['timestamp'] <= end_time
            ]
            
            # Generate report sections
            executive_summary = self._generate_executive_summary(relevant_history)
            violation_analysis = self._analyze_violations(relevant_history)
            compliance_trends = self._analyze_compliance_trends(relevant_history)
            recommendations = self._generate_compliance_recommendations(relevant_history)
            
            return {
                "report_id": str(uuid4()),
                "scope": scope,
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "executive_summary": executive_summary,
                "violation_analysis": violation_analysis,
                "compliance_trends": compliance_trends,
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Compliance report generation failed: {str(e)}")
            raise
    
    def _calculate_compliance_score(self, violations: List[ComplianceViolation], warnings: List[str]) -> float:
        """Calculate overall compliance score"""
        if not violations and not warnings:
            return 100.0
        
        # Weight violations by severity
        violation_penalty = 0
        for violation in violations:
            if violation.severity == ViolationSeverity.CRITICAL:
                violation_penalty += 30
            elif violation.severity == ViolationSeverity.HIGH:
                violation_penalty += 20
            elif violation.severity == ViolationSeverity.MEDIUM:
                violation_penalty += 10
            elif violation.severity == ViolationSeverity.LOW:
                violation_penalty += 5
        
        # Warnings have minimal impact
        warning_penalty = len(warnings) * 2
        
        total_penalty = violation_penalty + warning_penalty
        return max(0.0, 100.0 - total_penalty)
    
    def _determine_compliance_status(self, score: float, violations: List[ComplianceViolation]) -> ComplianceStatus:
        """Determine overall compliance status"""
        # Critical violations always result in non-compliance
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            return ComplianceStatus.NON_COMPLIANT
        
        # Score-based determination
        if score >= 95:
            return ComplianceStatus.COMPLIANT
        elif score >= 80:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        elif score >= 60:
            return ComplianceStatus.NEEDS_REVIEW
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    async def _assess_legal_risk(self, violations: List[ComplianceViolation], context: ComplianceContext) -> Dict[str, Any]:
        """Assess legal risk based on violations and context"""
        if not violations:
            return {"risk_level": "low", "risk_score": 0.1}
        
        # Calculate risk based on violation severity and jurisdiction
        risk_score = 0
        for violation in violations:
            if violation.severity == ViolationSeverity.CRITICAL:
                risk_score += 0.4
            elif violation.severity == ViolationSeverity.HIGH:
                risk_score += 0.3
            elif violation.severity == ViolationSeverity.MEDIUM:
                risk_score += 0.2
            elif violation.severity == ViolationSeverity.LOW:
                risk_score += 0.1
        
        # Jurisdiction-specific risk multipliers
        jurisdiction_multipliers = {
            ComplianceJurisdiction.EU: 1.3,  # Stricter regulations
            ComplianceJurisdiction.CALIFORNIA: 1.2,  # Strong privacy laws
            ComplianceJurisdiction.US_FEDERAL: 1.0,
            ComplianceJurisdiction.UK: 1.1,
            ComplianceJurisdiction.CANADA: 1.0,
            ComplianceJurisdiction.AUSTRALIA: 1.0
        }
        
        risk_score *= jurisdiction_multipliers.get(context.jurisdiction, 1.0)
        risk_score = min(1.0, risk_score)  # Cap at 1.0
        
        # Categorize risk level
        if risk_score >= 0.8:
            risk_level = "critical"
        elif risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        elif risk_score >= 0.2:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "jurisdiction_factor": jurisdiction_multipliers.get(context.jurisdiction, 1.0)
        }
    
    async def _trigger_remediation(self, violations: List[ComplianceViolation], context: ComplianceContext):
        """Trigger automatic remediation for violations"""
        try:
            for violation in violations:
                if violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH]:
                    await self.remediation_orchestrator.remediate_violation(violation, context)
        except Exception as e:
            logger.error(f"Automatic remediation failed: {str(e)}")
    
    def _calculate_risk_score(self, potential_violations: List[Dict], context: ComplianceContext) -> float:
        """Calculate risk score for potential violations"""
        if not potential_violations:
            return 0.1
        
        risk_score = sum(v.get('risk_weight', 0.1) for v in potential_violations)
        return min(1.0, risk_score)
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on score"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    async def _generate_risk_mitigation(self, potential_violations: List[Dict], context: ComplianceContext) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        for violation in potential_violations:
            violation_type = violation.get('type', 'unknown')
            
            if violation_type == 'employment_law':
                recommendations.append("Implement gradual workforce transition with proper notice periods")
                recommendations.append("Provide comprehensive retraining and support programs")
            elif violation_type == 'data_privacy':
                recommendations.append("Ensure proper consent mechanisms are in place")
                recommendations.append("Implement data minimization and privacy-by-design principles")
            elif violation_type == 'ai_regulation':
                recommendations.append("Provide algorithmic transparency and explainability")
                recommendations.append("Implement bias detection and mitigation measures")
            elif violation_type == 'professional_licensing':
                recommendations.append("Require licensed professional oversight for regulated activities")
                recommendations.append("Include appropriate disclaimers for AI-generated content")
        
        return recommendations
    
    def _generate_executive_summary(self, history: List[Dict]) -> Dict[str, Any]:
        """Generate executive summary for compliance report"""
        total_operations = len(history)
        compliant_operations = len([h for h in history if h['result'].is_compliant])
        
        return {
            "total_operations": total_operations,
            "compliant_operations": compliant_operations,
            "compliance_rate": (compliant_operations / total_operations * 100) if total_operations > 0 else 100,
            "period_summary": f"Processed {total_operations} operations with {compliant_operations} fully compliant"
        }
    
    def _analyze_violations(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze violations from compliance history"""
        all_violations = []
        for entry in history:
            all_violations.extend(entry['result'].violations)
        
        violation_by_type = {}
        violation_by_severity = {}
        
        for violation in all_violations:
            # Group by regulation type
            reg_type = violation.regulation
            violation_by_type[reg_type] = violation_by_type.get(reg_type, 0) + 1
            
            # Group by severity
            severity = violation.severity.value
            violation_by_severity[severity] = violation_by_severity.get(severity, 0) + 1
        
        return {
            "total_violations": len(all_violations),
            "violations_by_type": violation_by_type,
            "violations_by_severity": violation_by_severity
        }
    
    def _analyze_compliance_trends(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze compliance trends over time"""
        if not history:
            return {"trend": "no_data"}
        
        # Calculate compliance scores over time
        scores = [entry['result'].compliance_score for entry in history]
        
        if len(scores) < 2:
            return {"trend": "insufficient_data", "current_score": scores[0] if scores else 0}
        
        # Simple trend analysis
        recent_avg = sum(scores[-5:]) / min(5, len(scores))
        overall_avg = sum(scores) / len(scores)
        
        if recent_avg > overall_avg + 5:
            trend = "improving"
        elif recent_avg < overall_avg - 5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_average": recent_avg,
            "overall_average": overall_avg,
            "total_assessments": len(scores)
        }
    
    def _generate_compliance_recommendations(self, history: List[Dict]) -> List[str]:
        """Generate compliance recommendations based on history"""
        recommendations = []
        
        # Analyze common violation patterns
        all_violations = []
        for entry in history:
            all_violations.extend(entry['result'].violations)
        
        # Count violations by type
        violation_counts = {}
        for violation in all_violations:
            violation_counts[violation.regulation] = violation_counts.get(violation.regulation, 0) + 1
        
        # Generate recommendations for most common violations
        for regulation, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            if regulation == "employment_law":
                recommendations.append("Strengthen employment law compliance with enhanced worker transition programs")
            elif regulation == "data_privacy":
                recommendations.append("Improve data privacy controls and consent management systems")
            elif regulation == "ai_regulation":
                recommendations.append("Enhance AI transparency and bias detection capabilities")
        
        if not recommendations:
            recommendations.append("Continue monitoring compliance and maintain current standards")
        
        return recommendations