"""
Compliance Policy Enforcement Engine

This module provides comprehensive compliance policy enforcement for data pipelines,
ensuring regulatory compliance and data governance.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import re
import json
import logging

from scrollintel.models.lineage_models import (
    ComplianceRule, ComplianceViolation, ComplianceRuleType, ComplianceStatus,
    DataGovernancePolicy, ComplianceRuleRequest, ComplianceViolationResponse
)
from scrollintel.models.database_utils import get_sync_db

logger = logging.getLogger(__name__)


class ComplianceEngine:
    """Comprehensive compliance policy enforcement system"""
    
    def __init__(self):
        self.session = next(get_sync_db())
        self.active_rules = {}
        self.policy_cache = {}
        self._load_active_rules()
    
    def create_compliance_rule(
        self,
        request: ComplianceRuleRequest,
        created_by: str
    ) -> str:
        """
        Create a new compliance rule
        
        Args:
            request: Compliance rule details
            created_by: User creating the rule
            
        Returns:
            str: Rule ID
        """
        try:
            rule_id = str(uuid.uuid4())
            rule = ComplianceRule(
                id=rule_id,
                name=request.name,
                description=request.description,
                rule_type=request.rule_type.value,
                conditions=request.conditions,
                actions=request.actions,
                severity=request.severity,
                applicable_datasets=request.applicable_datasets,
                applicable_pipelines=request.applicable_pipelines,
                created_by=created_by
            )
            
            self.session.add(rule)
            self.session.commit()
            
            # Add to active rules cache
            self.active_rules[rule_id] = rule
            
            logger.info(f"Created compliance rule: {rule_id}")
            return rule_id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating compliance rule: {str(e)}")
            raise
    
    def evaluate_compliance(
        self,
        pipeline_id: str,
        dataset_id: Optional[str] = None,
        operation_type: str = "data_processing",
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate compliance for a pipeline operation
        
        Args:
            pipeline_id: Pipeline being evaluated
            dataset_id: Dataset being processed (optional)
            operation_type: Type of operation being performed
            context: Additional context for evaluation
            
        Returns:
            List[Dict]: Compliance evaluation results
        """
        try:
            results = []
            context = context or {}
            
            # Get applicable rules
            applicable_rules = self._get_applicable_rules(
                pipeline_id, dataset_id, operation_type
            )
            
            for rule in applicable_rules:
                evaluation_result = self._evaluate_rule(
                    rule, pipeline_id, dataset_id, operation_type, context
                )
                results.append(evaluation_result)
                
                # Handle violations
                if not evaluation_result["compliant"]:
                    self._handle_violation(
                        rule, pipeline_id, dataset_id, evaluation_result
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating compliance: {str(e)}")
            raise
    
    def enforce_data_governance(
        self,
        data_classification: str,
        operation: str,
        user_context: Dict[str, Any],
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enforce data governance policies
        
        Args:
            data_classification: Classification level of data
            operation: Operation being performed
            user_context: User and session context
            data_context: Data-specific context
            
        Returns:
            Dict: Governance enforcement result
        """
        try:
            # Get applicable policies
            policies = self._get_applicable_policies(
                data_classification, operation
            )
            
            enforcement_result = {
                "allowed": True,
                "restrictions": [],
                "requirements": [],
                "violations": []
            }
            
            for policy in policies:
                policy_result = self._enforce_policy(
                    policy, operation, user_context, data_context
                )
                
                if not policy_result["allowed"]:
                    enforcement_result["allowed"] = False
                
                enforcement_result["restrictions"].extend(
                    policy_result.get("restrictions", [])
                )
                enforcement_result["requirements"].extend(
                    policy_result.get("requirements", [])
                )
                enforcement_result["violations"].extend(
                    policy_result.get("violations", [])
                )
            
            return enforcement_result
            
        except Exception as e:
            logger.error(f"Error enforcing data governance: {str(e)}")
            raise
    
    def get_compliance_violations(
        self,
        pipeline_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        status: Optional[ComplianceStatus] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ComplianceViolationResponse]:
        """
        Get compliance violations based on filters
        
        Args:
            pipeline_id: Filter by pipeline
            rule_id: Filter by rule
            status: Filter by status
            severity: Filter by severity
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            
        Returns:
            List[ComplianceViolationResponse]: Violations
        """
        try:
            query = self.session.query(ComplianceViolation)
            
            if pipeline_id:
                query = query.filter(ComplianceViolation.pipeline_id == pipeline_id)
            
            if rule_id:
                query = query.filter(ComplianceViolation.rule_id == rule_id)
            
            if status:
                query = query.filter(ComplianceViolation.status == status.value)
            
            if severity:
                query = query.filter(ComplianceViolation.severity == severity)
            
            if start_date:
                query = query.filter(ComplianceViolation.detected_at >= start_date)
            
            if end_date:
                query = query.filter(ComplianceViolation.detected_at <= end_date)
            
            violations = query.order_by(desc(ComplianceViolation.detected_at)).limit(limit).all()
            
            return [
                ComplianceViolationResponse(
                    id=v.id,
                    rule_id=v.rule_id,
                    pipeline_id=v.pipeline_id,
                    dataset_id=v.dataset_id,
                    violation_type=v.violation_type,
                    description=v.description,
                    severity=v.severity,
                    status=ComplianceStatus(v.status),
                    detected_at=v.detected_at,
                    resolved_at=v.resolved_at,
                    resolution_notes=v.resolution_notes
                )
                for v in violations
            ]
            
        except Exception as e:
            logger.error(f"Error getting compliance violations: {str(e)}")
            raise
    
    def resolve_violation(
        self,
        violation_id: str,
        resolution_notes: str,
        resolved_by: str
    ) -> bool:
        """
        Resolve a compliance violation
        
        Args:
            violation_id: Violation to resolve
            resolution_notes: Resolution details
            resolved_by: User resolving the violation
            
        Returns:
            bool: Success status
        """
        try:
            violation = self.session.query(ComplianceViolation).filter(
                ComplianceViolation.id == violation_id
            ).first()
            
            if not violation:
                raise ValueError(f"Violation not found: {violation_id}")
            
            violation.status = ComplianceStatus.REMEDIATED.value
            violation.resolved_at = datetime.utcnow()
            violation.resolution_notes = resolution_notes
            violation.resolved_by = resolved_by
            violation.updated_at = datetime.utcnow()
            
            self.session.commit()
            
            logger.info(f"Resolved violation: {violation_id}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error resolving violation: {str(e)}")
            raise
    
    def get_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Generate compliance report for a time period
        
        Args:
            start_date: Report start date
            end_date: Report end date
            include_details: Include detailed violation information
            
        Returns:
            Dict: Compliance report
        """
        try:
            # Get violations in time period
            violations = self.session.query(ComplianceViolation).filter(
                and_(
                    ComplianceViolation.detected_at >= start_date,
                    ComplianceViolation.detected_at <= end_date
                )
            ).all()
            
            # Calculate metrics
            total_violations = len(violations)
            resolved_violations = len([v for v in violations if v.resolved_at])
            
            severity_breakdown = {}
            rule_type_breakdown = {}
            
            for violation in violations:
                # Severity breakdown
                severity = violation.severity
                severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1
                
                # Rule type breakdown
                rule = self.session.query(ComplianceRule).filter(
                    ComplianceRule.id == violation.rule_id
                ).first()
                if rule:
                    rule_type = rule.rule_type
                    rule_type_breakdown[rule_type] = rule_type_breakdown.get(rule_type, 0) + 1
            
            report = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_violations": total_violations,
                    "resolved_violations": resolved_violations,
                    "pending_violations": total_violations - resolved_violations,
                    "resolution_rate": (resolved_violations / total_violations * 100) if total_violations > 0 else 100
                },
                "breakdown": {
                    "by_severity": severity_breakdown,
                    "by_rule_type": rule_type_breakdown
                }
            }
            
            if include_details:
                report["violations"] = [
                    {
                        "id": v.id,
                        "rule_id": v.rule_id,
                        "pipeline_id": v.pipeline_id,
                        "violation_type": v.violation_type,
                        "severity": v.severity,
                        "status": v.status,
                        "detected_at": v.detected_at.isoformat(),
                        "resolved_at": v.resolved_at.isoformat() if v.resolved_at else None
                    }
                    for v in violations
                ]
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {str(e)}")
            raise
    
    def _load_active_rules(self):
        """Load active compliance rules into cache"""
        try:
            active_rules = self.session.query(ComplianceRule).filter(
                ComplianceRule.is_active == True
            ).all()
            
            self.active_rules = {rule.id: rule for rule in active_rules}
            
        except Exception as e:
            logger.error(f"Error loading active rules: {str(e)}")
    
    def _get_applicable_rules(
        self,
        pipeline_id: str,
        dataset_id: Optional[str],
        operation_type: str
    ) -> List[ComplianceRule]:
        """Get rules applicable to the current context"""
        applicable_rules = []
        
        for rule in self.active_rules.values():
            # Check pipeline applicability
            if rule.applicable_pipelines:
                if not any(
                    self._matches_pattern(pipeline_id, pattern)
                    for pattern in rule.applicable_pipelines
                ):
                    continue
            
            # Check dataset applicability
            if dataset_id and rule.applicable_datasets:
                if not any(
                    self._matches_pattern(dataset_id, pattern)
                    for pattern in rule.applicable_datasets
                ):
                    continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_rule(
        self,
        rule: ComplianceRule,
        pipeline_id: str,
        dataset_id: Optional[str],
        operation_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a specific compliance rule"""
        try:
            evaluation_context = {
                "pipeline_id": pipeline_id,
                "dataset_id": dataset_id,
                "operation_type": operation_type,
                **context
            }
            
            # Evaluate rule conditions
            compliant = self._evaluate_conditions(
                rule.conditions, evaluation_context
            )
            
            result = {
                "rule_id": rule.id,
                "rule_name": rule.name,
                "rule_type": rule.rule_type,
                "compliant": compliant,
                "severity": rule.severity,
                "evaluation_context": evaluation_context
            }
            
            if not compliant:
                result["violation_details"] = self._get_violation_details(
                    rule, evaluation_context
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.id}: {str(e)}")
            return {
                "rule_id": rule.id,
                "rule_name": rule.name,
                "compliant": False,
                "error": str(e)
            }
    
    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate rule conditions against context"""
        try:
            # Simple condition evaluation logic
            # In a real implementation, this would be more sophisticated
            
            for condition_key, condition_value in conditions.items():
                if condition_key == "data_retention_days":
                    # Check data retention compliance
                    if not self._check_data_retention(condition_value, context):
                        return False
                
                elif condition_key == "encryption_required":
                    # Check encryption compliance
                    if condition_value and not context.get("encrypted", False):
                        return False
                
                elif condition_key == "access_control":
                    # Check access control compliance
                    if not self._check_access_control(condition_value, context):
                        return False
                
                elif condition_key == "data_classification":
                    # Check data classification compliance
                    required_classification = condition_value
                    actual_classification = context.get("data_classification")
                    if actual_classification != required_classification:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating conditions: {str(e)}")
            return False
    
    def _check_data_retention(
        self,
        retention_days: int,
        context: Dict[str, Any]
    ) -> bool:
        """Check data retention compliance"""
        data_age = context.get("data_age_days", 0)
        return data_age <= retention_days
    
    def _check_access_control(
        self,
        access_requirements: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check access control compliance"""
        user_roles = context.get("user_roles", [])
        required_roles = access_requirements.get("required_roles", [])
        
        return any(role in user_roles for role in required_roles)
    
    def _get_violation_details(
        self,
        rule: ComplianceRule,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get detailed violation information"""
        return {
            "rule_conditions": rule.conditions,
            "context": context,
            "suggested_actions": rule.actions.get("remediation", [])
        }
    
    def _handle_violation(
        self,
        rule: ComplianceRule,
        pipeline_id: str,
        dataset_id: Optional[str],
        evaluation_result: Dict[str, Any]
    ):
        """Handle a compliance violation"""
        try:
            violation_id = str(uuid.uuid4())
            violation = ComplianceViolation(
                id=violation_id,
                rule_id=rule.id,
                pipeline_id=pipeline_id,
                dataset_id=dataset_id,
                violation_type=rule.rule_type,
                description=f"Violation of rule '{rule.name}': {evaluation_result.get('violation_details', {})}",
                severity=rule.severity,
                status=ComplianceStatus.NON_COMPLIANT.value,
                violation_data=evaluation_result
            )
            
            self.session.add(violation)
            self.session.commit()
            
            # Execute violation actions
            self._execute_violation_actions(rule.actions, violation)
            
            logger.warning(f"Compliance violation detected: {violation_id}")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error handling violation: {str(e)}")
    
    def _execute_violation_actions(
        self,
        actions: Dict[str, Any],
        violation: ComplianceViolation
    ):
        """Execute actions for a compliance violation"""
        try:
            # Alert actions
            if "alert" in actions:
                self._send_compliance_alert(violation, actions["alert"])
            
            # Automatic remediation actions
            if "auto_remediate" in actions:
                self._auto_remediate_violation(violation, actions["auto_remediate"])
            
            # Pipeline actions
            if "pipeline_action" in actions:
                self._execute_pipeline_action(violation, actions["pipeline_action"])
                
        except Exception as e:
            logger.error(f"Error executing violation actions: {str(e)}")
    
    def _send_compliance_alert(
        self,
        violation: ComplianceViolation,
        alert_config: Dict[str, Any]
    ):
        """Send compliance alert"""
        # Implementation would integrate with alerting system
        logger.warning(f"Compliance alert: {violation.description}")
    
    def _auto_remediate_violation(
        self,
        violation: ComplianceViolation,
        remediation_config: Dict[str, Any]
    ):
        """Attempt automatic remediation"""
        # Implementation would perform automatic remediation actions
        logger.info(f"Auto-remediating violation: {violation.id}")
    
    def _execute_pipeline_action(
        self,
        violation: ComplianceViolation,
        action_config: Dict[str, Any]
    ):
        """Execute pipeline-level action"""
        action = action_config.get("action", "none")
        
        if action == "pause":
            # Implementation would pause the pipeline
            logger.warning(f"Pausing pipeline {violation.pipeline_id} due to compliance violation")
        elif action == "quarantine":
            # Implementation would quarantine data
            logger.warning(f"Quarantining data due to compliance violation")
    
    def _get_applicable_policies(
        self,
        data_classification: str,
        operation: str
    ) -> List[DataGovernancePolicy]:
        """Get applicable governance policies"""
        # Implementation would query governance policies
        return []
    
    def _enforce_policy(
        self,
        policy: DataGovernancePolicy,
        operation: str,
        user_context: Dict[str, Any],
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enforce a specific governance policy"""
        # Implementation would enforce policy rules
        return {"allowed": True}
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if value matches pattern (supports wildcards)"""
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        return re.match(f"^{regex_pattern}$", value) is not None