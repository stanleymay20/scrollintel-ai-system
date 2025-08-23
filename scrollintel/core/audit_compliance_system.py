"""
Comprehensive Audit Logging and Compliance Reporting System
Implements enterprise-grade audit trails and compliance frameworks
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from ..models.security_compliance_models import (
    AuditLog, User, UserSession, Role, Permission,
    AuditEventType, ComplianceFramework, SecurityLevel
)
from ..security.enterprise_security_framework import EnterpriseSecurityFramework

logger = logging.getLogger(__name__)

class ComplianceStatus(str, Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"

class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    control_objective: str
    risk_level: RiskLevel
    automated_check: bool = True
    check_function: Optional[str] = None

@dataclass
class AuditFinding:
    """Audit finding structure"""
    id: str
    rule_id: str
    status: ComplianceStatus
    severity: RiskLevel
    description: str
    evidence: List[str]
    remediation: str
    timestamp: datetime
    affected_resources: List[str]

class ComprehensiveAuditSystem:
    """
    Enterprise audit logging and compliance system implementing:
    - Real-time audit event capture
    - Immutable audit trails
    - Compliance framework mapping
    - Automated compliance checking
    - Risk assessment and reporting
    - Evidence collection and preservation
    """
    
    def __init__(self, security_framework: EnterpriseSecurityFramework):
        self.security_framework = security_framework
        self.compliance_rules = self._initialize_compliance_rules()
        self.audit_processors = {}
        self._initialize_processors()
    
    def _initialize_compliance_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize compliance rules for different frameworks"""
        
        rules = {}
        
        # GDPR Rules
        rules["GDPR-001"] = ComplianceRule(
            id="GDPR-001",
            framework=ComplianceFramework.GDPR,
            title="Data Processing Consent",
            description="Ensure explicit consent for personal data processing",
            requirement="Article 6 - Lawfulness of processing",
            control_objective="Verify consent before processing personal data",
            risk_level=RiskLevel.HIGH,
            check_function="check_gdpr_consent"
        )
        
        rules["GDPR-002"] = ComplianceRule(
            id="GDPR-002",
            framework=ComplianceFramework.GDPR,
            title="Data Subject Rights",
            description="Implement data subject access, rectification, and erasure rights",
            requirement="Articles 15-17 - Rights of the data subject",
            control_objective="Enable data subject rights exercise",
            risk_level=RiskLevel.MEDIUM,
            check_function="check_data_subject_rights"
        )
        
        rules["GDPR-003"] = ComplianceRule(
            id="GDPR-003",
            framework=ComplianceFramework.GDPR,
            title="Data Breach Notification",
            description="Report data breaches within 72 hours",
            requirement="Article 33 - Notification of a personal data breach",
            control_objective="Timely breach notification",
            risk_level=RiskLevel.CRITICAL,
            check_function="check_breach_notification"
        )
        
        # SOX Rules
        rules["SOX-001"] = ComplianceRule(
            id="SOX-001",
            framework=ComplianceFramework.SOX,
            title="Financial Data Access Controls",
            description="Implement proper access controls for financial data",
            requirement="Section 404 - Management assessment of internal controls",
            control_objective="Restrict access to financial systems",
            risk_level=RiskLevel.HIGH,
            check_function="check_financial_access_controls"
        )
        
        rules["SOX-002"] = ComplianceRule(
            id="SOX-002",
            framework=ComplianceFramework.SOX,
            title="Change Management",
            description="Document and approve all changes to financial systems",
            requirement="Section 302 - Corporate responsibility for financial reports",
            control_objective="Controlled change management",
            risk_level=RiskLevel.MEDIUM,
            check_function="check_change_management"
        )
        
        # HIPAA Rules
        rules["HIPAA-001"] = ComplianceRule(
            id="HIPAA-001",
            framework=ComplianceFramework.HIPAA,
            title="PHI Access Controls",
            description="Implement access controls for protected health information",
            requirement="164.312(a)(1) - Access control",
            control_objective="Restrict PHI access to authorized users",
            risk_level=RiskLevel.HIGH,
            check_function="check_phi_access_controls"
        )
        
        rules["HIPAA-002"] = ComplianceRule(
            id="HIPAA-002",
            framework=ComplianceFramework.HIPAA,
            title="Audit Controls",
            description="Implement audit controls for PHI access",
            requirement="164.312(b) - Audit controls",
            control_objective="Log and monitor PHI access",
            risk_level=RiskLevel.MEDIUM,
            check_function="check_audit_controls"
        )
        
        # ISO 27001 Rules
        rules["ISO27001-001"] = ComplianceRule(
            id="ISO27001-001",
            framework=ComplianceFramework.ISO_27001,
            title="Information Security Policy",
            description="Establish and maintain information security policy",
            requirement="A.5.1.1 - Information security policy",
            control_objective="Define security governance",
            risk_level=RiskLevel.MEDIUM,
            check_function="check_security_policy"
        )
        
        rules["ISO27001-002"] = ComplianceRule(
            id="ISO27001-002",
            framework=ComplianceFramework.ISO_27001,
            title="Access Management",
            description="Implement user access management controls",
            requirement="A.9.1.1 - Access control policy",
            control_objective="Control user access to information",
            risk_level=RiskLevel.HIGH,
            check_function="check_access_management"
        )
        
        return rules
    
    def _initialize_processors(self):
        """Initialize audit event processors"""
        
        self.audit_processors = {
            AuditEventType.AUTHENTICATION: self._process_authentication_event,
            AuditEventType.AUTHORIZATION: self._process_authorization_event,
            AuditEventType.DATA_ACCESS: self._process_data_access_event,
            AuditEventType.CONFIGURATION_CHANGE: self._process_configuration_event,
            AuditEventType.SECURITY_INCIDENT: self._process_security_incident,
            AuditEventType.COMPLIANCE_VIOLATION: self._process_compliance_violation
        }
    
    # Real-time Audit Event Processing
    
    async def process_audit_event(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Process audit event and generate compliance findings"""
        
        findings = []
        
        try:
            # Get event processor
            event_type = AuditEventType(audit_log.event_type)
            processor = self.audit_processors.get(event_type)
            
            if processor:
                event_findings = await processor(db, audit_log)
                findings.extend(event_findings)
            
            # Run compliance checks
            compliance_findings = await self._run_compliance_checks(db, audit_log)
            findings.extend(compliance_findings)
            
            # Store findings
            for finding in findings:
                await self._store_audit_finding(db, finding)
            
            return findings
            
        except Exception as e:
            logger.error(f"Error processing audit event {audit_log.id}: {e}")
            return []
    
    async def _process_authentication_event(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Process authentication events for compliance"""
        
        findings = []
        
        # Check for failed authentication patterns
        if not audit_log.success:
            # Check for brute force attempts
            recent_failures = db.query(AuditLog).filter(
                AuditLog.event_type == AuditEventType.AUTHENTICATION.value,
                AuditLog.success == False,
                AuditLog.ip_address == audit_log.ip_address,
                AuditLog.timestamp >= datetime.utcnow() - timedelta(minutes=15)
            ).count()
            
            if recent_failures >= 5:
                findings.append(AuditFinding(
                    id=f"AUTH-{audit_log.id}-001",
                    rule_id="SECURITY-001",
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=RiskLevel.HIGH,
                    description="Potential brute force attack detected",
                    evidence=[f"IP {audit_log.ip_address} had {recent_failures} failed attempts"],
                    remediation="Block IP address and investigate",
                    timestamp=datetime.utcnow(),
                    affected_resources=[audit_log.ip_address]
                ))
        
        # Check MFA compliance
        if audit_log.success and audit_log.user_id:
            user = db.query(User).filter(User.id == audit_log.user_id).first()
            if user and not user.mfa_enabled:
                findings.append(AuditFinding(
                    id=f"AUTH-{audit_log.id}-002",
                    rule_id="SECURITY-002",
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=RiskLevel.MEDIUM,
                    description="User authenticated without MFA",
                    evidence=[f"User {user.username} does not have MFA enabled"],
                    remediation="Enforce MFA for all users",
                    timestamp=datetime.utcnow(),
                    affected_resources=[user.id]
                ))
        
        return findings
    
    async def _process_authorization_event(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Process authorization events for compliance"""
        
        findings = []
        
        # Check for privilege escalation attempts
        if not audit_log.success and audit_log.error_code == "INSUFFICIENT_PERMISSIONS":
            # Check if user is attempting to access higher security level resources
            if audit_log.user_id and audit_log.resource_type:
                user = db.query(User).filter(User.id == audit_log.user_id).first()
                if user:
                    user_level = SecurityLevel(user.security_level)
                    
                    # Check recent failed authorization attempts
                    recent_failures = db.query(AuditLog).filter(
                        AuditLog.event_type == AuditEventType.AUTHORIZATION.value,
                        AuditLog.user_id == audit_log.user_id,
                        AuditLog.success == False,
                        AuditLog.timestamp >= datetime.utcnow() - timedelta(hours=1)
                    ).count()
                    
                    if recent_failures >= 3:
                        findings.append(AuditFinding(
                            id=f"AUTHZ-{audit_log.id}-001",
                            rule_id="SECURITY-003",
                            status=ComplianceStatus.NON_COMPLIANT,
                            severity=RiskLevel.HIGH,
                            description="Potential privilege escalation attempt",
                            evidence=[f"User {user.username} had {recent_failures} failed authorization attempts"],
                            remediation="Review user permissions and investigate intent",
                            timestamp=datetime.utcnow(),
                            affected_resources=[user.id, audit_log.resource_id]
                        ))
        
        return findings
    
    async def _process_data_access_event(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Process data access events for compliance"""
        
        findings = []
        
        # Check for sensitive data access
        if audit_log.sensitive_data_accessed:
            # Verify proper authorization
            if not audit_log.success:
                findings.append(AuditFinding(
                    id=f"DATA-{audit_log.id}-001",
                    rule_id="GDPR-001",
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=RiskLevel.CRITICAL,
                    description="Unauthorized access to sensitive data attempted",
                    evidence=[f"Failed access to {audit_log.resource_type}:{audit_log.resource_id}"],
                    remediation="Investigate unauthorized access attempt",
                    timestamp=datetime.utcnow(),
                    affected_resources=[audit_log.resource_id]
                ))
            
            # Check for unusual access patterns
            if audit_log.success and audit_log.user_id:
                # Check access outside normal hours
                current_hour = datetime.utcnow().hour
                if current_hour < 6 or current_hour > 22:  # Outside 6 AM - 10 PM
                    findings.append(AuditFinding(
                        id=f"DATA-{audit_log.id}-002",
                        rule_id="SECURITY-004",
                        status=ComplianceStatus.UNDER_REVIEW,
                        severity=RiskLevel.MEDIUM,
                        description="Sensitive data accessed outside normal hours",
                        evidence=[f"Access at {audit_log.timestamp}"],
                        remediation="Review if access was legitimate",
                        timestamp=datetime.utcnow(),
                        affected_resources=[audit_log.user_id, audit_log.resource_id]
                    ))
        
        return findings
    
    async def _process_configuration_event(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Process configuration change events for compliance"""
        
        findings = []
        
        # Check for unauthorized configuration changes
        if audit_log.success and audit_log.user_id:
            user = db.query(User).filter(User.id == audit_log.user_id).first()
            if user:
                # Check if user has appropriate security level for configuration changes
                user_level = SecurityLevel(user.security_level)
                
                if user_level not in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                    findings.append(AuditFinding(
                        id=f"CONFIG-{audit_log.id}-001",
                        rule_id="SOX-002",
                        status=ComplianceStatus.NON_COMPLIANT,
                        severity=RiskLevel.HIGH,
                        description="Configuration change by user with insufficient clearance",
                        evidence=[f"User {user.username} with {user_level.value} clearance made changes"],
                        remediation="Review user permissions and change approval process",
                        timestamp=datetime.utcnow(),
                        affected_resources=[user.id, audit_log.resource_id]
                    ))
        
        return findings
    
    async def _process_security_incident(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Process security incident events"""
        
        findings = []
        
        # All security incidents are compliance violations
        findings.append(AuditFinding(
            id=f"INCIDENT-{audit_log.id}-001",
            rule_id="SECURITY-005",
            status=ComplianceStatus.NON_COMPLIANT,
            severity=RiskLevel.CRITICAL,
            description="Security incident detected",
            evidence=[audit_log.event_description],
            remediation="Investigate and remediate security incident",
            timestamp=datetime.utcnow(),
            affected_resources=[audit_log.resource_id] if audit_log.resource_id else []
        ))
        
        return findings
    
    async def _process_compliance_violation(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Process compliance violation events"""
        
        findings = []
        
        # Direct compliance violations
        frameworks = audit_log.compliance_frameworks or []
        
        for framework in frameworks:
            findings.append(AuditFinding(
                id=f"COMPLIANCE-{audit_log.id}-{framework}",
                rule_id=f"{framework.upper()}-VIOLATION",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=RiskLevel.HIGH,
                description=f"Compliance violation for {framework}",
                evidence=[audit_log.event_description],
                remediation=f"Address {framework} compliance requirements",
                timestamp=datetime.utcnow(),
                affected_resources=[audit_log.resource_id] if audit_log.resource_id else []
            ))
        
        return findings
    
    async def _run_compliance_checks(self, db: Session, audit_log: AuditLog) -> List[AuditFinding]:
        """Run automated compliance checks"""
        
        findings = []
        
        # Get applicable compliance frameworks
        applicable_frameworks = self._get_applicable_frameworks(audit_log)
        
        for framework in applicable_frameworks:
            framework_rules = [rule for rule in self.compliance_rules.values() 
                             if rule.framework == framework and rule.automated_check]
            
            for rule in framework_rules:
                if rule.check_function:
                    try:
                        check_result = await self._execute_compliance_check(db, rule, audit_log)
                        if check_result:
                            findings.append(check_result)
                    except Exception as e:
                        logger.error(f"Error executing compliance check {rule.id}: {e}")
        
        return findings
    
    def _get_applicable_frameworks(self, audit_log: AuditLog) -> List[ComplianceFramework]:
        """Determine applicable compliance frameworks for audit event"""
        
        frameworks = []
        
        # Always check general security frameworks
        frameworks.append(ComplianceFramework.ISO_27001)
        
        # Check for specific data types
        if audit_log.sensitive_data_accessed:
            frameworks.append(ComplianceFramework.GDPR)
        
        if audit_log.resource_type == "financial_data":
            frameworks.append(ComplianceFramework.SOX)
        
        if audit_log.resource_type == "health_data":
            frameworks.append(ComplianceFramework.HIPAA)
        
        return frameworks
    
    async def _execute_compliance_check(
        self, 
        db: Session, 
        rule: ComplianceRule, 
        audit_log: AuditLog
    ) -> Optional[AuditFinding]:
        """Execute specific compliance check"""
        
        check_function = getattr(self, rule.check_function, None)
        if not check_function:
            return None
        
        try:
            return await check_function(db, rule, audit_log)
        except Exception as e:
            logger.error(f"Compliance check {rule.id} failed: {e}")
            return None
    
    # Specific Compliance Checks
    
    async def check_gdpr_consent(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check GDPR consent requirements"""
        
        if audit_log.sensitive_data_accessed and audit_log.success:
            # Check if consent was recorded (simplified check)
            # In real implementation, check consent management system
            
            if not audit_log.event_metadata or not audit_log.event_metadata.get('consent_verified'):
                return AuditFinding(
                    id=f"GDPR-CONSENT-{audit_log.id}",
                    rule_id=rule.id,
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=rule.risk_level,
                    description="Personal data processed without verified consent",
                    evidence=[f"Data access without consent verification"],
                    remediation="Implement consent verification before data processing",
                    timestamp=datetime.utcnow(),
                    affected_resources=[audit_log.resource_id]
                )
        
        return None
    
    async def check_data_subject_rights(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check data subject rights implementation"""
        
        # Check if data subject rights requests are handled within required timeframe
        if audit_log.action in ['data_export', 'data_deletion', 'data_rectification']:
            # Check processing time (simplified)
            processing_time = audit_log.duration_ms or 0
            
            if processing_time > 30 * 24 * 60 * 60 * 1000:  # 30 days in milliseconds
                return AuditFinding(
                    id=f"GDPR-RIGHTS-{audit_log.id}",
                    rule_id=rule.id,
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=rule.risk_level,
                    description="Data subject rights request not processed within required timeframe",
                    evidence=[f"Processing time: {processing_time}ms"],
                    remediation="Improve data subject rights processing efficiency",
                    timestamp=datetime.utcnow(),
                    affected_resources=[audit_log.user_id]
                )
        
        return None
    
    async def check_breach_notification(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check breach notification requirements"""
        
        if audit_log.event_type == AuditEventType.SECURITY_INCIDENT.value:
            # Check if breach was reported within 72 hours
            incident_time = audit_log.timestamp
            current_time = datetime.utcnow()
            
            if (current_time - incident_time).total_seconds() > 72 * 3600:  # 72 hours
                return AuditFinding(
                    id=f"GDPR-BREACH-{audit_log.id}",
                    rule_id=rule.id,
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=RiskLevel.CRITICAL,
                    description="Data breach not reported within 72 hours",
                    evidence=[f"Incident occurred at {incident_time}, not reported"],
                    remediation="Implement automated breach notification system",
                    timestamp=datetime.utcnow(),
                    affected_resources=[audit_log.resource_id]
                )
        
        return None
    
    async def check_financial_access_controls(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check SOX financial access controls"""
        
        if audit_log.resource_type == "financial_data" and audit_log.success:
            # Check if user has appropriate role for financial data access
            if audit_log.user_id:
                user = db.query(User).filter(User.id == audit_log.user_id).first()
                if user:
                    # Check user roles (simplified)
                    user_roles = db.query(Role).join(UserRole).filter(
                        UserRole.user_id == user.id
                    ).all()
                    
                    financial_roles = ['financial_analyst', 'accountant', 'cfo', 'admin']
                    has_financial_role = any(role.name in financial_roles for role in user_roles)
                    
                    if not has_financial_role:
                        return AuditFinding(
                            id=f"SOX-ACCESS-{audit_log.id}",
                            rule_id=rule.id,
                            status=ComplianceStatus.NON_COMPLIANT,
                            severity=rule.risk_level,
                            description="Financial data accessed by user without appropriate role",
                            evidence=[f"User {user.username} accessed financial data"],
                            remediation="Restrict financial data access to authorized roles",
                            timestamp=datetime.utcnow(),
                            affected_resources=[user.id, audit_log.resource_id]
                        )
        
        return None
    
    async def check_change_management(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check change management controls"""
        
        if audit_log.event_type == AuditEventType.CONFIGURATION_CHANGE.value:
            # Check if change was approved (simplified)
            if not audit_log.event_metadata or not audit_log.event_metadata.get('change_approved'):
                return AuditFinding(
                    id=f"SOX-CHANGE-{audit_log.id}",
                    rule_id=rule.id,
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=rule.risk_level,
                    description="Configuration change made without proper approval",
                    evidence=[f"Change to {audit_log.resource_type} without approval"],
                    remediation="Implement change approval workflow",
                    timestamp=datetime.utcnow(),
                    affected_resources=[audit_log.resource_id]
                )
        
        return None
    
    async def check_phi_access_controls(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check HIPAA PHI access controls"""
        
        if audit_log.resource_type == "health_data" and audit_log.success:
            # Check minimum necessary principle
            if audit_log.user_id:
                user = db.query(User).filter(User.id == audit_log.user_id).first()
                if user:
                    # Check if access was for legitimate healthcare purpose
                    if not audit_log.event_metadata or not audit_log.event_metadata.get('healthcare_purpose'):
                        return AuditFinding(
                            id=f"HIPAA-PHI-{audit_log.id}",
                            rule_id=rule.id,
                            status=ComplianceStatus.NON_COMPLIANT,
                            severity=rule.risk_level,
                            description="PHI accessed without documented healthcare purpose",
                            evidence=[f"User {user.username} accessed PHI"],
                            remediation="Document healthcare purpose for all PHI access",
                            timestamp=datetime.utcnow(),
                            affected_resources=[user.id, audit_log.resource_id]
                        )
        
        return None
    
    async def check_audit_controls(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check audit controls implementation"""
        
        # This is a meta-check - ensure audit logging is working
        # Check for gaps in audit logging
        
        return None  # Simplified for this implementation
    
    async def check_security_policy(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check security policy compliance"""
        
        # Check if security policies are being followed
        if audit_log.event_type == AuditEventType.SECURITY_INCIDENT.value:
            return AuditFinding(
                id=f"ISO27001-POLICY-{audit_log.id}",
                rule_id=rule.id,
                status=ComplianceStatus.NON_COMPLIANT,
                severity=rule.risk_level,
                description="Security incident indicates policy violation",
                evidence=[audit_log.event_description],
                remediation="Review and strengthen security policies",
                timestamp=datetime.utcnow(),
                affected_resources=[audit_log.resource_id]
            )
        
        return None
    
    async def check_access_management(self, db: Session, rule: ComplianceRule, audit_log: AuditLog) -> Optional[AuditFinding]:
        """Check access management controls"""
        
        if audit_log.event_type == AuditEventType.AUTHORIZATION.value and not audit_log.success:
            # Multiple failed authorization attempts indicate access management issues
            recent_failures = db.query(AuditLog).filter(
                AuditLog.event_type == AuditEventType.AUTHORIZATION.value,
                AuditLog.user_id == audit_log.user_id,
                AuditLog.success == False,
                AuditLog.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ).count()
            
            if recent_failures >= 5:
                return AuditFinding(
                    id=f"ISO27001-ACCESS-{audit_log.id}",
                    rule_id=rule.id,
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity=rule.risk_level,
                    description="Excessive authorization failures indicate access management issues",
                    evidence=[f"{recent_failures} failed authorization attempts"],
                    remediation="Review user access rights and permissions",
                    timestamp=datetime.utcnow(),
                    affected_resources=[audit_log.user_id]
                )
        
        return None
    
    # Audit Finding Storage and Retrieval
    
    async def _store_audit_finding(self, db: Session, finding: AuditFinding):
        """Store audit finding in database"""
        
        try:
            # In a real implementation, you would have a dedicated findings table
            # For now, we'll log as a special audit event
            
            self.security_framework._log_audit_event(
                db=db,
                user_id=None,
                event_type=AuditEventType.COMPLIANCE_VIOLATION,
                event_category="audit_finding",
                event_description=finding.description,
                resource_type="compliance_finding",
                resource_id=finding.id,
                action="finding_created",
                success=True,
                metadata={
                    'finding_id': finding.id,
                    'rule_id': finding.rule_id,
                    'status': finding.status.value,
                    'severity': finding.severity.value,
                    'evidence': finding.evidence,
                    'remediation': finding.remediation,
                    'affected_resources': finding.affected_resources
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store audit finding {finding.id}: {e}")
    
    # Compliance Reporting
    
    async def generate_comprehensive_compliance_report(
        self,
        db: Session,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        try:
            # Get all audit events for the period
            audit_events = db.query(AuditLog).filter(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date
            ).all()
            
            # Get compliance findings
            compliance_findings = db.query(AuditLog).filter(
                AuditLog.event_type == AuditEventType.COMPLIANCE_VIOLATION.value,
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date,
                AuditLog.event_metadata.contains(f'"framework": "{framework.value}"')
            ).all()
            
            # Calculate compliance metrics
            total_events = len(audit_events)
            violation_events = len(compliance_findings)
            compliance_score = ((total_events - violation_events) / total_events * 100) if total_events > 0 else 100
            
            # Analyze findings by severity
            severity_breakdown = {
                RiskLevel.LOW.value: 0,
                RiskLevel.MEDIUM.value: 0,
                RiskLevel.HIGH.value: 0,
                RiskLevel.CRITICAL.value: 0
            }
            
            for finding in compliance_findings:
                if finding.event_metadata and 'severity' in finding.event_metadata:
                    severity = finding.event_metadata['severity']
                    if severity in severity_breakdown:
                        severity_breakdown[severity] += 1
            
            # Get framework-specific rules
            framework_rules = [rule for rule in self.compliance_rules.values() 
                             if rule.framework == framework]
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(
                framework, compliance_findings, severity_breakdown
            )
            
            # Create executive summary
            executive_summary = self._create_executive_summary(
                framework, compliance_score, severity_breakdown, total_events
            )
            
            return {
                'framework': framework.value,
                'reporting_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'executive_summary': executive_summary,
                'compliance_metrics': {
                    'overall_score': compliance_score,
                    'total_events': total_events,
                    'violation_events': violation_events,
                    'compliance_rate': compliance_score / 100
                },
                'risk_assessment': {
                    'severity_breakdown': severity_breakdown,
                    'risk_trend': self._calculate_risk_trend(db, framework, start_date, end_date),
                    'high_risk_areas': self._identify_high_risk_areas(compliance_findings)
                },
                'control_effectiveness': {
                    'implemented_controls': len(framework_rules),
                    'effective_controls': len(framework_rules) - len([f for f in compliance_findings if f.event_metadata.get('severity') in ['high', 'critical']]),
                    'control_gaps': self._identify_control_gaps(framework_rules, compliance_findings)
                },
                'findings_summary': {
                    'total_findings': len(compliance_findings),
                    'by_severity': severity_breakdown,
                    'by_category': self._categorize_findings(compliance_findings),
                    'remediation_status': self._get_remediation_status(compliance_findings)
                },
                'recommendations': recommendations,
                'action_plan': self._create_action_plan(compliance_findings, recommendations),
                'appendices': {
                    'detailed_findings': [self._format_finding_detail(f) for f in compliance_findings[:50]],  # Limit for report size
                    'compliance_rules': [self._format_rule_detail(r) for r in framework_rules],
                    'methodology': self._get_assessment_methodology()
                },
                'generated_at': datetime.utcnow().isoformat(),
                'report_version': '1.0',
                'next_assessment_date': (end_date + timedelta(days=90)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise
    
    def _generate_compliance_recommendations(
        self, 
        framework: ComplianceFramework, 
        findings: List[AuditLog], 
        severity_breakdown: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Generate compliance recommendations based on findings"""
        
        recommendations = []
        
        # High-level recommendations based on severity
        if severity_breakdown[RiskLevel.CRITICAL.value] > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Immediate Action Required',
                'recommendation': 'Address all critical compliance violations immediately',
                'timeline': '24-48 hours',
                'impact': 'Prevents regulatory penalties and data breaches'
            })
        
        if severity_breakdown[RiskLevel.HIGH.value] > 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Security Controls',
                'recommendation': 'Strengthen access controls and monitoring',
                'timeline': '1-2 weeks',
                'impact': 'Reduces risk of unauthorized access'
            })
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                {
                    'priority': 'HIGH',
                    'category': 'Data Protection',
                    'recommendation': 'Implement privacy by design principles',
                    'timeline': '1 month',
                    'impact': 'Ensures GDPR compliance for new systems'
                },
                {
                    'priority': 'MEDIUM',
                    'category': 'Consent Management',
                    'recommendation': 'Deploy automated consent management system',
                    'timeline': '2 months',
                    'impact': 'Streamlines consent collection and management'
                }
            ])
        
        elif framework == ComplianceFramework.SOX:
            recommendations.extend([
                {
                    'priority': 'HIGH',
                    'category': 'Financial Controls',
                    'recommendation': 'Implement segregation of duties for financial processes',
                    'timeline': '2 weeks',
                    'impact': 'Prevents financial fraud and errors'
                },
                {
                    'priority': 'MEDIUM',
                    'category': 'Change Management',
                    'recommendation': 'Establish formal change approval process',
                    'timeline': '1 month',
                    'impact': 'Ensures all changes are authorized and documented'
                }
            ])
        
        return recommendations
    
    def _create_executive_summary(
        self, 
        framework: ComplianceFramework, 
        compliance_score: float, 
        severity_breakdown: Dict[str, int], 
        total_events: int
    ) -> str:
        """Create executive summary for compliance report"""
        
        risk_level = "LOW"
        if severity_breakdown[RiskLevel.CRITICAL.value] > 0:
            risk_level = "CRITICAL"
        elif severity_breakdown[RiskLevel.HIGH.value] > 5:
            risk_level = "HIGH"
        elif severity_breakdown[RiskLevel.MEDIUM.value] > 10:
            risk_level = "MEDIUM"
        
        summary = f"""
        EXECUTIVE SUMMARY - {framework.value.upper()} COMPLIANCE ASSESSMENT
        
        Overall Compliance Score: {compliance_score:.1f}%
        Risk Level: {risk_level}
        Total Events Analyzed: {total_events:,}
        
        Key Findings:
        - {severity_breakdown[RiskLevel.CRITICAL.value]} Critical violations requiring immediate attention
        - {severity_breakdown[RiskLevel.HIGH.value]} High-risk issues needing prompt resolution
        - {severity_breakdown[RiskLevel.MEDIUM.value]} Medium-risk items for planned remediation
        
        The organization demonstrates {'strong' if compliance_score >= 90 else 'adequate' if compliance_score >= 75 else 'weak'} 
        compliance with {framework.value.upper()} requirements. 
        {'Immediate action is required to address critical violations.' if risk_level == 'CRITICAL' else 
         'Continued monitoring and improvement recommended.' if risk_level in ['HIGH', 'MEDIUM'] else 
         'Maintain current compliance posture with regular assessments.'}
        """
        
        return summary.strip()
    
    def _calculate_risk_trend(
        self, 
        db: Session, 
        framework: ComplianceFramework, 
        start_date: datetime, 
        end_date: datetime
    ) -> str:
        """Calculate risk trend over time"""
        
        # Compare with previous period
        period_length = end_date - start_date
        previous_start = start_date - period_length
        previous_end = start_date
        
        current_violations = db.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.COMPLIANCE_VIOLATION.value,
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        previous_violations = db.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.COMPLIANCE_VIOLATION.value,
            AuditLog.timestamp >= previous_start,
            AuditLog.timestamp <= previous_end
        ).count()
        
        if previous_violations == 0:
            return "STABLE" if current_violations == 0 else "INCREASING"
        
        change_rate = (current_violations - previous_violations) / previous_violations
        
        if change_rate > 0.2:
            return "INCREASING"
        elif change_rate < -0.2:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _identify_high_risk_areas(self, findings: List[AuditLog]) -> List[str]:
        """Identify high-risk areas from findings"""
        
        risk_areas = {}
        
        for finding in findings:
            if finding.event_metadata and finding.event_metadata.get('severity') in ['high', 'critical']:
                resource_type = finding.resource_type or 'unknown'
                risk_areas[resource_type] = risk_areas.get(resource_type, 0) + 1
        
        # Sort by frequency and return top areas
        sorted_areas = sorted(risk_areas.items(), key=lambda x: x[1], reverse=True)
        return [area[0] for area in sorted_areas[:5]]
    
    def _identify_control_gaps(self, rules: List[ComplianceRule], findings: List[AuditLog]) -> List[str]:
        """Identify control gaps based on findings"""
        
        gaps = []
        
        # Check which rules have violations
        violated_rules = set()
        for finding in findings:
            if finding.event_metadata and 'rule_id' in finding.event_metadata:
                violated_rules.add(finding.event_metadata['rule_id'])
        
        for rule in rules:
            if rule.id in violated_rules:
                gaps.append(f"{rule.title}: {rule.control_objective}")
        
        return gaps
    
    def _categorize_findings(self, findings: List[AuditLog]) -> Dict[str, int]:
        """Categorize findings by type"""
        
        categories = {}
        
        for finding in findings:
            category = finding.event_category or 'unknown'
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def _get_remediation_status(self, findings: List[AuditLog]) -> Dict[str, int]:
        """Get remediation status of findings"""
        
        # Simplified - in real implementation, track remediation progress
        return {
            'open': len(findings),
            'in_progress': 0,
            'resolved': 0,
            'closed': 0
        }
    
    def _create_action_plan(self, findings: List[AuditLog], recommendations: List[Dict]) -> List[Dict[str, Any]]:
        """Create action plan based on findings and recommendations"""
        
        action_items = []
        
        # Convert recommendations to action items
        for i, rec in enumerate(recommendations[:10]):  # Limit to top 10
            action_items.append({
                'id': f"ACTION-{i+1:03d}",
                'title': rec['recommendation'],
                'priority': rec['priority'],
                'category': rec['category'],
                'timeline': rec['timeline'],
                'owner': 'Security Team',  # Default owner
                'status': 'Not Started',
                'dependencies': [],
                'success_criteria': f"Implement {rec['recommendation'].lower()}"
            })
        
        return action_items
    
    def _format_finding_detail(self, finding: AuditLog) -> Dict[str, Any]:
        """Format finding detail for report"""
        
        return {
            'id': finding.id,
            'timestamp': finding.timestamp.isoformat(),
            'description': finding.event_description,
            'severity': finding.event_metadata.get('severity', 'unknown') if finding.event_metadata else 'unknown',
            'resource': f"{finding.resource_type}:{finding.resource_id}" if finding.resource_type else 'unknown',
            'user': finding.user_id,
            'remediation': finding.event_metadata.get('remediation', 'Not specified') if finding.event_metadata else 'Not specified'
        }
    
    def _format_rule_detail(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Format rule detail for report"""
        
        return {
            'id': rule.id,
            'title': rule.title,
            'description': rule.description,
            'requirement': rule.requirement,
            'control_objective': rule.control_objective,
            'risk_level': rule.risk_level.value,
            'automated': rule.automated_check
        }
    
    def _get_assessment_methodology(self) -> Dict[str, Any]:
        """Get assessment methodology description"""
        
        return {
            'approach': 'Automated continuous compliance monitoring',
            'data_sources': ['Audit logs', 'System configurations', 'User activities'],
            'analysis_methods': ['Rule-based checking', 'Pattern analysis', 'Risk assessment'],
            'coverage': 'All system activities and configurations',
            'frequency': 'Real-time monitoring with periodic reporting',
            'limitations': 'Automated checks may not capture all compliance nuances'
        }