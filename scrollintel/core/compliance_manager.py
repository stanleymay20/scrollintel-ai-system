"""
Compliance Manager for ScrollIntel Launch MVP

Handles compliance reporting, data retention policies, and regulatory requirements.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func, text
from sqlalchemy.orm import selectinload

from ..models.database import AuditLog, User
from ..models.database_utils import get_db
from ..core.logging_config import get_logger
from ..core.audit_system import audit_system, AuditAction

logger = get_logger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CUSTOM = "custom"


class DataClassification(str, Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class RetentionPeriod(str, Enum):
    """Standard retention periods"""
    DAYS_30 = "30_days"
    DAYS_90 = "90_days"
    MONTHS_6 = "6_months"
    YEAR_1 = "1_year"
    YEARS_3 = "3_years"
    YEARS_7 = "7_years"
    PERMANENT = "permanent"


@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    id: str
    framework: ComplianceFramework
    rule_name: str
    description: str
    severity: str  # low, medium, high, critical
    data_types: List[str]
    retention_period: RetentionPeriod
    access_controls: Dict[str, Any]
    monitoring_required: bool
    automated_check: bool


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    id: str
    rule_id: str
    violation_type: str
    severity: str
    detected_at: datetime
    resource_type: str
    resource_id: str
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    status: str  # open, investigating, resolved, false_positive
    resolution_notes: Optional[str]
    resolved_at: Optional[datetime]


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition"""
    id: str
    name: str
    description: str
    data_types: List[str]
    retention_period: RetentionPeriod
    deletion_method: str  # soft_delete, hard_delete, archive
    compliance_frameworks: List[ComplianceFramework]
    exceptions: List[str]
    auto_cleanup: bool
    notification_before_deletion: int  # days


class ComplianceManager:
    """Manages compliance rules, violations, and data retention"""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.retention_policies = self._load_retention_policies()
        self._violation_queue: asyncio.Queue = asyncio.Queue()
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    def _load_compliance_rules(self) -> Dict[str, ComplianceRule]:
        """Load compliance rules configuration"""
        return {
            "gdpr_data_access": ComplianceRule(
                id="gdpr_data_access",
                framework=ComplianceFramework.GDPR,
                rule_name="Data Access Logging",
                description="All personal data access must be logged",
                severity="high",
                data_types=["personal_data", "user_data"],
                retention_period=RetentionPeriod.YEARS_3,
                access_controls={"require_justification": True, "approval_required": False},
                monitoring_required=True,
                automated_check=True
            ),
            "gdpr_data_deletion": ComplianceRule(
                id="gdpr_data_deletion",
                framework=ComplianceFramework.GDPR,
                rule_name="Right to be Forgotten",
                description="User data deletion requests must be processed within 30 days",
                severity="critical",
                data_types=["personal_data", "user_data"],
                retention_period=RetentionPeriod.PERMANENT,
                access_controls={"require_justification": True, "approval_required": True},
                monitoring_required=True,
                automated_check=True
            ),
            "sox_financial_data": ComplianceRule(
                id="sox_financial_data",
                framework=ComplianceFramework.SOX,
                rule_name="Financial Data Controls",
                description="Financial data access requires approval and audit trail",
                severity="critical",
                data_types=["financial_data", "billing_data"],
                retention_period=RetentionPeriod.YEARS_7,
                access_controls={"require_justification": True, "approval_required": True},
                monitoring_required=True,
                automated_check=True
            ),
            "iso27001_access_control": ComplianceRule(
                id="iso27001_access_control",
                framework=ComplianceFramework.ISO_27001,
                rule_name="Access Control Management",
                description="User access must be regularly reviewed and validated",
                severity="medium",
                data_types=["all"],
                retention_period=RetentionPeriod.YEAR_1,
                access_controls={"require_justification": False, "approval_required": False},
                monitoring_required=True,
                automated_check=True
            )
        }
    
    def _load_retention_policies(self) -> Dict[str, DataRetentionPolicy]:
        """Load data retention policies"""
        return {
            "audit_logs_standard": DataRetentionPolicy(
                id="audit_logs_standard",
                name="Standard Audit Log Retention",
                description="Standard retention for audit logs",
                data_types=["audit_logs"],
                retention_period=RetentionPeriod.YEARS_3,
                deletion_method="hard_delete",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.ISO_27001],
                exceptions=["security_incidents", "legal_holds"],
                auto_cleanup=True,
                notification_before_deletion=30
            ),
            "user_data_gdpr": DataRetentionPolicy(
                id="user_data_gdpr",
                name="GDPR User Data Retention",
                description="GDPR compliant user data retention",
                data_types=["personal_data", "user_profiles"],
                retention_period=RetentionPeriod.YEARS_3,
                deletion_method="soft_delete",
                compliance_frameworks=[ComplianceFramework.GDPR],
                exceptions=["active_contracts", "legal_obligations"],
                auto_cleanup=False,
                notification_before_deletion=90
            ),
            "financial_data_sox": DataRetentionPolicy(
                id="financial_data_sox",
                name="SOX Financial Data Retention",
                description="SOX compliant financial data retention",
                data_types=["financial_data", "billing_records"],
                retention_period=RetentionPeriod.YEARS_7,
                deletion_method="archive",
                compliance_frameworks=[ComplianceFramework.SOX],
                exceptions=[],
                auto_cleanup=False,
                notification_before_deletion=180
            )
        }
    
    async def start(self) -> None:
        """Start compliance monitoring"""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._process_violations())
            logger.info("Compliance manager started")
    
    async def stop(self) -> None:
        """Stop compliance monitoring"""
        self._shutdown_event.set()
        if self._background_task:
            await self._background_task
        logger.info("Compliance manager stopped")
    
    async def check_compliance(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        data_classification: DataClassification = DataClassification.INTERNAL
    ) -> List[ComplianceViolation]:
        """Check action against compliance rules"""
        
        violations = []
        
        for rule_id, rule in self.compliance_rules.items():
            if await self._violates_rule(rule, action, resource_type, resource_id, user_id, data_classification):
                violation = ComplianceViolation(
                    id=str(uuid4()),
                    rule_id=rule_id,
                    violation_type="access_control",
                    severity=rule.severity,
                    detected_at=datetime.utcnow(),
                    resource_type=resource_type,
                    resource_id=resource_id,
                    user_id=user_id,
                    description=f"Violation of {rule.rule_name}",
                    details={
                        "action": action,
                        "data_classification": data_classification.value,
                        "rule_description": rule.description
                    },
                    status="open",
                    resolution_notes=None,
                    resolved_at=None
                )
                violations.append(violation)
                
                # Queue for processing
                await self._violation_queue.put(violation)
        
        return violations
    
    async def _violates_rule(
        self,
        rule: ComplianceRule,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        data_classification: DataClassification
    ) -> bool:
        """Check if action violates specific compliance rule"""
        
        # Check if rule applies to this data type
        if rule.data_types != ["all"] and resource_type not in rule.data_types:
            return False
        
        # Example compliance checks
        if rule.id == "gdpr_data_access":
            # Check if personal data access is properly logged
            if resource_type in ["personal_data", "user_data"]:
                # This would be a violation if not properly justified
                return data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]
        
        elif rule.id == "sox_financial_data":
            # Check financial data access controls
            if resource_type in ["financial_data", "billing_data"]:
                # This would require additional approval checks
                return action in ["delete", "modify"] and data_classification == DataClassification.RESTRICTED
        
        return False
    
    async def _process_violations(self) -> None:
        """Background task to process compliance violations"""
        while not self._shutdown_event.is_set():
            try:
                violation = await asyncio.wait_for(
                    self._violation_queue.get(),
                    timeout=1.0
                )
                
                await self._handle_violation(violation)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing compliance violation: {e}")
                continue
    
    async def _handle_violation(self, violation: ComplianceViolation) -> None:
        """Handle a compliance violation"""
        
        # Log the violation
        await audit_system.log_event(
            action=AuditAction.PERMISSION_DENIED,
            resource_type="compliance_violation",
            resource_id=violation.id,
            details={
                "violation_type": violation.violation_type,
                "severity": violation.severity,
                "rule_id": violation.rule_id,
                "original_resource_type": violation.resource_type,
                "original_resource_id": violation.resource_id,
                "description": violation.description
            },
            user_id=violation.user_id,
            success=False,
            error_message=f"Compliance violation: {violation.description}"
        )
        
        # Store violation (in practice, you'd save to database)
        logger.warning(
            f"Compliance violation detected: {violation.description}",
            violation_id=violation.id,
            rule_id=violation.rule_id,
            severity=violation.severity,
            user_id=violation.user_id
        )
        
        # Send notifications for high/critical violations
        if violation.severity in ["high", "critical"]:
            await self._send_violation_notification(violation)
    
    async def _send_violation_notification(self, violation: ComplianceViolation) -> None:
        """Send notification for compliance violation"""
        
        # In practice, this would send email/Slack notifications
        logger.critical(
            f"CRITICAL COMPLIANCE VIOLATION: {violation.description}",
            violation_id=violation.id,
            rule_id=violation.rule_id,
            user_id=violation.user_id,
            resource_type=violation.resource_type,
            resource_id=violation.resource_id
        )
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate framework-specific compliance report"""
        
        # Get relevant audit logs
        logs = await audit_system.search_audit_logs(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        # Filter rules for framework
        framework_rules = {
            rule_id: rule for rule_id, rule in self.compliance_rules.items()
            if rule.framework == framework
        }
        
        # Analyze compliance
        compliance_status = {}
        violations = []
        
        for rule_id, rule in framework_rules.items():
            rule_violations = []
            
            # Check each log against rule
            for log in logs:
                if await self._log_violates_rule(log, rule):
                    rule_violations.append({
                        "log_id": log["id"],
                        "timestamp": log["timestamp"],
                        "user_id": log["user_id"],
                        "action": log["action"],
                        "resource_type": log["resource_type"],
                        "details": log["details"]
                    })
            
            compliance_status[rule_id] = {
                "rule_name": rule.rule_name,
                "description": rule.description,
                "severity": rule.severity,
                "violations_count": len(rule_violations),
                "compliance_rate": max(0, 100 - (len(rule_violations) / len(logs) * 100)) if logs else 100,
                "violations": rule_violations[:10]  # Limit to first 10
            }
            
            violations.extend(rule_violations)
        
        return {
            "framework": framework.value,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events_analyzed": len(logs),
                "total_violations": len(violations),
                "compliance_rate": max(0, 100 - (len(violations) / len(logs) * 100)) if logs else 100,
                "rules_evaluated": len(framework_rules)
            },
            "rule_compliance": compliance_status,
            "recommendations": self._generate_compliance_recommendations(compliance_status),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _log_violates_rule(self, log: Dict[str, Any], rule: ComplianceRule) -> bool:
        """Check if audit log violates compliance rule"""
        
        # Simplified violation detection
        if rule.id == "gdpr_data_access":
            return (
                log["resource_type"] in ["personal_data", "user_data"] and
                not log["success"] and
                "permission_denied" in log.get("error_message", "").lower()
            )
        
        elif rule.id == "sox_financial_data":
            return (
                log["resource_type"] in ["financial_data", "billing_data"] and
                log["action"] in ["delete", "modify"] and
                not log["success"]
            )
        
        return False
    
    def _generate_compliance_recommendations(
        self,
        compliance_status: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        for rule_id, status in compliance_status.items():
            if status["violations_count"] > 0:
                if status["severity"] == "critical":
                    recommendations.append(
                        f"URGENT: Address {status['violations_count']} critical violations "
                        f"for {status['rule_name']}"
                    )
                elif status["violations_count"] > 10:
                    recommendations.append(
                        f"Review access controls for {status['rule_name']} - "
                        f"{status['violations_count']} violations detected"
                    )
            
            if status["compliance_rate"] < 95:
                recommendations.append(
                    f"Improve compliance rate for {status['rule_name']} "
                    f"(currently {status['compliance_rate']:.1f}%)"
                )
        
        if not recommendations:
            recommendations.append("All compliance rules are being followed effectively")
        
        return recommendations
    
    async def apply_retention_policies(self) -> Dict[str, int]:
        """Apply data retention policies and clean up expired data"""
        
        cleanup_results = {}
        
        for policy_id, policy in self.retention_policies.items():
            if not policy.auto_cleanup:
                continue
            
            try:
                # Calculate retention cutoff date
                retention_days = self._get_retention_days(policy.retention_period)
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Apply retention policy
                cleaned_count = await self._cleanup_data_by_policy(policy, cutoff_date)
                cleanup_results[policy_id] = cleaned_count
                
                if cleaned_count > 0:
                    logger.info(
                        f"Applied retention policy {policy.name}: cleaned {cleaned_count} records",
                        policy_id=policy_id,
                        retention_period=policy.retention_period.value,
                        cutoff_date=cutoff_date.isoformat()
                    )
                
            except Exception as e:
                logger.error(f"Error applying retention policy {policy_id}: {e}")
                cleanup_results[policy_id] = -1
        
        return cleanup_results
    
    def _get_retention_days(self, period: RetentionPeriod) -> int:
        """Convert retention period to days"""
        period_map = {
            RetentionPeriod.DAYS_30: 30,
            RetentionPeriod.DAYS_90: 90,
            RetentionPeriod.MONTHS_6: 180,
            RetentionPeriod.YEAR_1: 365,
            RetentionPeriod.YEARS_3: 1095,
            RetentionPeriod.YEARS_7: 2555,
            RetentionPeriod.PERMANENT: 36500
        }
        return period_map.get(period, 365)
    
    async def _cleanup_data_by_policy(
        self,
        policy: DataRetentionPolicy,
        cutoff_date: datetime
    ) -> int:
        """Clean up data according to retention policy"""
        
        cleaned_count = 0
        
        if "audit_logs" in policy.data_types:
            # Clean up audit logs
            async for session in get_db():
                if policy.deletion_method == "hard_delete":
                    delete_query = text("""
                        DELETE FROM audit_logs 
                        WHERE timestamp < :cutoff_date
                        AND action NOT IN (
                            SELECT unnest(:exceptions)
                        )
                    """)
                    
                    result = await session.execute(
                        delete_query,
                        {
                            "cutoff_date": cutoff_date,
                            "exceptions": policy.exceptions
                        }
                    )
                    cleaned_count += result.rowcount
                    await session.commit()
                
                elif policy.deletion_method == "soft_delete":
                    # Mark as deleted instead of hard delete
                    update_query = text("""
                        UPDATE audit_logs 
                        SET details = jsonb_set(details, '{deleted}', 'true')
                        WHERE timestamp < :cutoff_date
                        AND action NOT IN (
                            SELECT unnest(:exceptions)
                        )
                    """)
                    
                    result = await session.execute(
                        update_query,
                        {
                            "cutoff_date": cutoff_date,
                            "exceptions": policy.exceptions
                        }
                    )
                    cleaned_count += result.rowcount
                    await session.commit()
                
                break
        
        return cleaned_count
    
    async def export_compliance_data(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> str:
        """Export compliance data for auditing"""
        
        # Generate compliance report
        report = await self.generate_compliance_report(framework, start_date, end_date)
        
        # Create export directory
        export_dir = Path("exports/compliance")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"compliance_{framework.value}_{timestamp}.{format}"
        filepath = export_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Log the export
        await audit_system.log_event(
            action=AuditAction.COMPLIANCE_EXPORT,
            resource_type="compliance_report",
            details={
                "framework": framework.value,
                "export_format": format,
                "date_range_start": start_date.isoformat(),
                "date_range_end": end_date.isoformat(),
                "file_path": str(filepath)
            }
        )
        
        return str(filepath)


# Global compliance manager instance
compliance_manager = ComplianceManager()