"""
Compliance management service for prompt management system.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import re
import json

from scrollintel.models.audit_models import (
    ComplianceRule, ComplianceViolation, ComplianceRuleCreate,
    ComplianceRuleResponse, ComplianceViolationResponse, ComplianceReport
)


class ComplianceManager:
    """Service for managing compliance rules and violations."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default compliance rules."""
        
        default_rules = [
            {
                "name": "Sensitive Content Detection",
                "description": "Detect prompts containing sensitive information",
                "rule_type": "content",
                "conditions": {
                    "patterns": [
                        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
                    ],
                    "keywords": ["password", "secret", "api_key", "token"]
                },
                "actions": {
                    "block": True,
                    "alert": True,
                    "require_approval": True
                },
                "severity": "high"
            },
            {
                "name": "Production Prompt Approval",
                "description": "Require approval for production prompt changes",
                "rule_type": "approval",
                "conditions": {
                    "resource_tags": ["production"],
                    "actions": ["update", "delete"]
                },
                "actions": {
                    "require_approval": True,
                    "notify_admins": True
                },
                "severity": "medium"
            },
            {
                "name": "Bulk Operations Monitoring",
                "description": "Monitor bulk operations for suspicious activity",
                "rule_type": "access",
                "conditions": {
                    "bulk_threshold": 10,
                    "time_window": 300  # 5 minutes
                },
                "actions": {
                    "alert": True,
                    "rate_limit": True
                },
                "severity": "medium"
            }
        ]
        
        for rule_data in default_rules:
            existing = self.db.query(ComplianceRule).filter(
                ComplianceRule.name == rule_data["name"]
            ).first()
            
            if not existing:
                rule = ComplianceRule(
                    id=str(uuid.uuid4()),
                    name=rule_data["name"],
                    description=rule_data["description"],
                    rule_type=rule_data["rule_type"],
                    conditions=rule_data["conditions"],
                    actions=rule_data["actions"],
                    severity=rule_data["severity"],
                    created_by="system",
                    created_at=datetime.utcnow()
                )
                self.db.add(rule)
        
        self.db.commit()
    
    def create_rule(self, rule_data: ComplianceRuleCreate, created_by: str) -> ComplianceRuleResponse:
        """Create a new compliance rule."""
        
        rule = ComplianceRule(
            id=str(uuid.uuid4()),
            name=rule_data.name,
            description=rule_data.description,
            rule_type=rule_data.rule_type,
            conditions=rule_data.conditions,
            actions=rule_data.actions,
            severity=rule_data.severity,
            enabled=rule_data.enabled,
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        
        self.db.add(rule)
        self.db.commit()
        
        return ComplianceRuleResponse.model_validate(rule)
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Optional[ComplianceRuleResponse]:
        """Update a compliance rule."""
        
        rule = self.db.query(ComplianceRule).filter(ComplianceRule.id == rule_id).first()
        if not rule:
            return None
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.utcnow()
        self.db.commit()
        
        return ComplianceRuleResponse.model_validate(rule)
    
    def get_rules(self, rule_type: Optional[str] = None, enabled_only: bool = True) -> List[ComplianceRuleResponse]:
        """Get compliance rules."""
        
        query = self.db.query(ComplianceRule)
        
        if rule_type:
            query = query.filter(ComplianceRule.rule_type == rule_type)
        
        if enabled_only:
            query = query.filter(ComplianceRule.enabled == True)
        
        rules = query.order_by(ComplianceRule.created_at).all()
        
        return [ComplianceRuleResponse.model_validate(rule) for rule in rules]
    
    def check_compliance(
        self,
        resource_type: str,
        resource_data: Dict[str, Any],
        action: str,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check compliance for a resource operation."""
        
        violations = []
        actions_required = []
        
        # Get applicable rules
        rules = self.get_rules()
        
        for rule in rules:
            violation = self._evaluate_rule(rule, resource_type, resource_data, action, user_context)
            if violation:
                violations.append(violation)
                
                # Determine required actions
                rule_obj = self.db.query(ComplianceRule).filter(ComplianceRule.id == rule.id).first()
                if rule_obj and rule_obj.actions:
                    actions_required.extend(self._get_required_actions(rule_obj.actions))
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "actions_required": list(set(actions_required)),
            "risk_level": self._calculate_overall_risk(violations)
        }
    
    def _evaluate_rule(
        self,
        rule: ComplianceRuleResponse,
        resource_type: str,
        resource_data: Dict[str, Any],
        action: str,
        user_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single compliance rule."""
        
        rule_obj = self.db.query(ComplianceRule).filter(ComplianceRule.id == rule.id).first()
        if not rule_obj or not rule_obj.enabled:
            return None
        
        conditions = rule_obj.conditions
        
        # Content-based rules
        if rule.rule_type == "content":
            return self._check_content_compliance(rule_obj, resource_data)
        
        # Access-based rules
        elif rule.rule_type == "access":
            return self._check_access_compliance(rule_obj, action, user_context)
        
        # Approval-based rules
        elif rule.rule_type == "approval":
            return self._check_approval_compliance(rule_obj, resource_type, resource_data, action)
        
        # Retention-based rules
        elif rule.rule_type == "retention":
            return self._check_retention_compliance(rule_obj, resource_data)
        
        return None
    
    def _check_content_compliance(self, rule: ComplianceRule, resource_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check content-based compliance rules."""
        
        conditions = rule.conditions
        content_fields = ["content", "description", "variables"]
        
        # Check for sensitive patterns
        if "patterns" in conditions:
            for field in content_fields:
                if field in resource_data:
                    content = str(resource_data[field])
                    for pattern in conditions["patterns"]:
                        if re.search(pattern, content, re.IGNORECASE):
                            return {
                                "rule_id": rule.id,
                                "violation_type": "sensitive_pattern",
                                "description": f"Sensitive pattern detected in {field}",
                                "severity": rule.severity,
                                "field": field,
                                "pattern": pattern
                            }
        
        # Check for sensitive keywords
        if "keywords" in conditions:
            for field in content_fields:
                if field in resource_data:
                    content = str(resource_data[field]).lower()
                    for keyword in conditions["keywords"]:
                        if keyword.lower() in content:
                            return {
                                "rule_id": rule.id,
                                "violation_type": "sensitive_keyword",
                                "description": f"Sensitive keyword '{keyword}' found in {field}",
                                "severity": rule.severity,
                                "field": field,
                                "keyword": keyword
                            }
        
        return None
    
    def _check_access_compliance(self, rule: ComplianceRule, action: str, user_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check access-based compliance rules."""
        
        conditions = rule.conditions
        
        # Check bulk operations
        if "bulk_threshold" in conditions:
            user_id = user_context.get("user_id")
            time_window = conditions.get("time_window", 300)
            
            # Count recent actions by user
            recent_actions = self._count_recent_actions(user_id, time_window)
            
            if recent_actions >= conditions["bulk_threshold"]:
                return {
                    "rule_id": rule.id,
                    "violation_type": "bulk_operation",
                    "description": f"Bulk operation detected: {recent_actions} actions in {time_window} seconds",
                    "severity": rule.severity,
                    "action_count": recent_actions
                }
        
        return None
    
    def _check_approval_compliance(self, rule: ComplianceRule, resource_type: str, resource_data: Dict[str, Any], action: str) -> Optional[Dict[str, Any]]:
        """Check approval-based compliance rules."""
        
        conditions = rule.conditions
        
        # Check if resource requires approval
        if "resource_tags" in conditions:
            resource_tags = resource_data.get("tags", [])
            if any(tag in resource_tags for tag in conditions["resource_tags"]):
                if "actions" in conditions and action in conditions["actions"]:
                    return {
                        "rule_id": rule.id,
                        "violation_type": "approval_required",
                        "description": f"Action '{action}' requires approval for tagged resource",
                        "severity": rule.severity,
                        "required_action": "approval"
                    }
        
        return None
    
    def _check_retention_compliance(self, rule: ComplianceRule, resource_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check retention-based compliance rules."""
        
        conditions = rule.conditions
        
        # Check data retention
        if "retention_days" in conditions:
            created_at = resource_data.get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                
                retention_period = timedelta(days=conditions["retention_days"])
                if datetime.utcnow() - created_at > retention_period:
                    return {
                        "rule_id": rule.id,
                        "violation_type": "retention_exceeded",
                        "description": f"Resource exceeds retention period of {conditions['retention_days']} days",
                        "severity": rule.severity,
                        "age_days": (datetime.utcnow() - created_at).days
                    }
        
        return None
    
    def _get_required_actions(self, rule_actions: Dict[str, Any]) -> List[str]:
        """Get required actions from rule configuration."""
        
        actions = []
        
        if rule_actions.get("block"):
            actions.append("block")
        
        if rule_actions.get("require_approval"):
            actions.append("require_approval")
        
        if rule_actions.get("alert"):
            actions.append("alert")
        
        if rule_actions.get("rate_limit"):
            actions.append("rate_limit")
        
        return actions
    
    def _calculate_overall_risk(self, violations: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level from violations."""
        
        if not violations:
            return "low"
        
        severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        max_severity = max(severity_scores.get(v.get("severity", "low"), 1) for v in violations)
        
        if max_severity >= 4:
            return "critical"
        elif max_severity >= 3:
            return "high"
        elif max_severity >= 2:
            return "medium"
        else:
            return "low"
    
    def _count_recent_actions(self, user_id: str, time_window: int) -> int:
        """Count recent actions by user."""
        
        from scrollintel.models.audit_models import AuditLog
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        
        count = self.db.query(AuditLog).filter(
            and_(
                AuditLog.user_id == user_id,
                AuditLog.timestamp >= cutoff_time
            )
        ).count()
        
        return count
    
    def record_violation(self, violation_data: Dict[str, Any], resource_type: str, resource_id: str) -> str:
        """Record a compliance violation."""
        
        violation = ComplianceViolation(
            id=str(uuid.uuid4()),
            rule_id=violation_data["rule_id"],
            resource_type=resource_type,
            resource_id=resource_id,
            violation_type=violation_data["violation_type"],
            description=violation_data["description"],
            severity=violation_data["severity"],
            status="open",
            detected_at=datetime.utcnow(),
            violation_metadata=violation_data
        )
        
        self.db.add(violation)
        self.db.commit()
        
        return violation.id
    
    def get_violations(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ComplianceViolationResponse]:
        """Get compliance violations."""
        
        query = self.db.query(ComplianceViolation)
        
        if status:
            query = query.filter(ComplianceViolation.status == status)
        
        if severity:
            query = query.filter(ComplianceViolation.severity == severity)
        
        if resource_type:
            query = query.filter(ComplianceViolation.resource_type == resource_type)
        
        violations = query.order_by(desc(ComplianceViolation.detected_at)).limit(limit).all()
        
        return [ComplianceViolationResponse.model_validate(v) for v in violations]
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> ComplianceReport:
        """Generate compliance report."""
        
        from scrollintel.models.audit_models import AuditLog
        
        # Count total actions in period
        total_actions = self.db.query(AuditLog).filter(
            and_(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date
            )
        ).count()
        
        # Count violations in period
        violations = self.db.query(ComplianceViolation).filter(
            and_(
                ComplianceViolation.detected_at >= start_date,
                ComplianceViolation.detected_at <= end_date
            )
        ).all()
        
        violations_count = len(violations)
        
        # Count pending approvals
        from scrollintel.models.audit_models import ChangeApproval
        pending_approvals = self.db.query(ChangeApproval).filter(
            ChangeApproval.status == "pending"
        ).count()
        
        # Calculate compliance score
        compliance_score = max(0, 100 - (violations_count / max(total_actions, 1)) * 100)
        
        # Risk summary
        risk_summary = {}
        for violation in violations:
            severity = violation.severity
            risk_summary[severity] = risk_summary.get(severity, 0) + 1
        
        # Top violations
        top_violations = [ComplianceViolationResponse.model_validate(v) for v in violations[:10]]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, total_actions)
        
        return ComplianceReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            period_start=start_date,
            period_end=end_date,
            total_actions=total_actions,
            violations_count=violations_count,
            pending_approvals=pending_approvals,
            compliance_score=compliance_score,
            risk_summary=risk_summary,
            top_violations=top_violations,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, violations: List[ComplianceViolation], total_actions: int) -> List[str]:
        """Generate compliance recommendations."""
        
        recommendations = []
        
        # High violation rate
        if len(violations) / max(total_actions, 1) > 0.1:
            recommendations.append("Consider reviewing and updating compliance rules")
        
        # Sensitive content violations
        content_violations = [v for v in violations if v.violation_type in ["sensitive_pattern", "sensitive_keyword"]]
        if content_violations:
            recommendations.append("Implement content scanning before prompt creation")
        
        # Bulk operation violations
        bulk_violations = [v for v in violations if v.violation_type == "bulk_operation"]
        if bulk_violations:
            recommendations.append("Consider implementing stricter rate limiting")
        
        # High severity violations
        high_severity = [v for v in violations if v.severity in ["high", "critical"]]
        if high_severity:
            recommendations.append("Review high-severity violations and strengthen controls")
        
        return recommendations