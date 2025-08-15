"""Policy enforcement engine for data governance."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import json
from sqlalchemy.orm import Session

from ..models.governance_models import (
    GovernancePolicy, PolicyType, PolicyStatus, AccessLevel,
    AccessControlEntry, AuditEvent, AuditEventType
)
from ..models.governance_database import (
    GovernancePolicyModel, AccessControlEntryModel, AuditEventModel,
    UserModel, DataCatalogEntryModel
)
from ..models.database import get_db_session
from ..core.exceptions import AIDataReadinessError


logger = logging.getLogger(__name__)


class PolicyEngineError(AIDataReadinessError):
    """Exception raised for policy engine errors."""
    pass


class PolicyEngine:
    """Policy enforcement engine for data governance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_policy(
        self,
        name: str,
        description: str,
        policy_type: PolicyType,
        rules: List[Dict[str, Any]],
        created_by: str,
        conditions: Optional[Dict[str, Any]] = None,
        enforcement_level: str = "strict",
        applicable_resources: Optional[List[str]] = None
    ) -> GovernancePolicy:
        """Create a new governance policy."""
        try:
            with get_db_session() as session:
                # Validate rules
                self._validate_policy_rules(rules, policy_type)
                
                policy = GovernancePolicyModel(
                    name=name,
                    description=description,
                    policy_type=policy_type.value,
                    rules=rules,
                    conditions=conditions or {},
                    enforcement_level=enforcement_level,
                    applicable_resources=applicable_resources or [],
                    created_by=created_by
                )
                
                session.add(policy)
                session.commit()
                session.refresh(policy)
                
                self.logger.info(f"Policy '{name}' created successfully")
                
                return self._model_to_dataclass(policy)
                
        except Exception as e:
            self.logger.error(f"Error creating policy: {str(e)}")
            raise PolicyEngineError(f"Failed to create policy: {str(e)}")
    
    def activate_policy(self, policy_id: str, approved_by: str) -> GovernancePolicy:
        """Activate a governance policy."""
        try:
            with get_db_session() as session:
                policy = session.query(GovernancePolicyModel).filter(
                    GovernancePolicyModel.id == policy_id
                ).first()
                
                if not policy:
                    raise PolicyEngineError(f"Policy {policy_id} not found")
                
                policy.status = PolicyStatus.ACTIVE.value
                policy.approved_by = approved_by
                policy.effective_date = datetime.utcnow()
                policy.updated_at = datetime.utcnow()
                
                session.commit()
                session.refresh(policy)
                
                self.logger.info(f"Policy {policy_id} activated")
                
                return self._model_to_dataclass(policy)
                
        except Exception as e:
            self.logger.error(f"Error activating policy: {str(e)}")
            raise PolicyEngineError(f"Failed to activate policy: {str(e)}")
    
    def enforce_policy(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Enforce policies for a user action."""
        try:
            with get_db_session() as session:
                # Get applicable policies
                policies = session.query(GovernancePolicyModel).filter(
                    GovernancePolicyModel.status == PolicyStatus.ACTIVE.value
                ).all()
                
                violations = []
                allowed = True
                
                for policy in policies:
                    # Check if policy applies to this resource
                    if not self._policy_applies(policy, resource_type, resource_id):
                        continue
                    
                    # Evaluate policy rules
                    policy_result = self._evaluate_policy(
                        policy, user_id, resource_id, resource_type, action, context
                    )
                    
                    if not policy_result['allowed']:
                        violations.extend(policy_result['violations'])
                        
                        if policy.enforcement_level == "strict":
                            allowed = False
                        elif policy.enforcement_level == "warning":
                            self.logger.warning(f"Policy violation (warning): {policy_result['violations']}")
                
                # Log enforcement result
                self._log_policy_enforcement(
                    user_id, resource_id, resource_type, action, allowed, violations
                )
                
                return allowed, violations
                
        except Exception as e:
            self.logger.error(f"Error enforcing policy: {str(e)}")
            raise PolicyEngineError(f"Failed to enforce policy: {str(e)}")
    
    def check_access_permission(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        required_access: AccessLevel
    ) -> bool:
        """Check if user has required access permission."""
        try:
            with get_db_session() as session:
                # Check direct access control entries
                access_entry = session.query(AccessControlEntryModel).filter(
                    AccessControlEntryModel.resource_id == resource_id,
                    AccessControlEntryModel.resource_type == resource_type,
                    AccessControlEntryModel.principal_id == user_id,
                    AccessControlEntryModel.is_active == True
                ).first()
                
                if access_entry:
                    user_access = AccessLevel(access_entry.access_level)
                    return self._access_level_sufficient(user_access, required_access)
                
                # Check role-based access (simplified - would need full RBAC implementation)
                # For now, return False if no direct access found
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking access permission: {str(e)}")
            return False
    
    def grant_access(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        access_level: AccessLevel,
        granted_by: str,
        expires_at: Optional[datetime] = None,
        conditions: Optional[Dict[str, Any]] = None
    ) -> AccessControlEntry:
        """Grant access to a resource."""
        try:
            with get_db_session() as session:
                # Check if access already exists
                existing_access = session.query(AccessControlEntryModel).filter(
                    AccessControlEntryModel.resource_id == resource_id,
                    AccessControlEntryModel.resource_type == resource_type,
                    AccessControlEntryModel.principal_id == user_id,
                    AccessControlEntryModel.is_active == True
                ).first()
                
                if existing_access:
                    # Update existing access
                    existing_access.access_level = access_level.value
                    existing_access.granted_by = granted_by
                    existing_access.granted_at = datetime.utcnow()
                    existing_access.expires_at = expires_at
                    existing_access.conditions = conditions or {}
                    
                    session.commit()
                    session.refresh(existing_access)
                    
                    access_entry = existing_access
                else:
                    # Create new access entry
                    access_entry = AccessControlEntryModel(
                        resource_id=resource_id,
                        resource_type=resource_type,
                        principal_id=user_id,
                        principal_type="user",
                        access_level=access_level.value,
                        granted_by=granted_by,
                        expires_at=expires_at,
                        conditions=conditions or {}
                    )
                    
                    session.add(access_entry)
                    session.commit()
                    session.refresh(access_entry)
                
                self.logger.info(f"Access granted to user {user_id} for resource {resource_id}")
                
                return self._access_model_to_dataclass(access_entry)
                
        except Exception as e:
            self.logger.error(f"Error granting access: {str(e)}")
            raise PolicyEngineError(f"Failed to grant access: {str(e)}")
    
    def revoke_access(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        revoked_by: str
    ) -> bool:
        """Revoke access to a resource."""
        try:
            with get_db_session() as session:
                access_entry = session.query(AccessControlEntryModel).filter(
                    AccessControlEntryModel.resource_id == resource_id,
                    AccessControlEntryModel.resource_type == resource_type,
                    AccessControlEntryModel.principal_id == user_id,
                    AccessControlEntryModel.is_active == True
                ).first()
                
                if not access_entry:
                    return False
                
                access_entry.is_active = False
                session.commit()
                
                # Log access revocation
                self._log_audit_event(
                    AuditEventType.USER_ACTION,
                    revoked_by,
                    resource_id,
                    resource_type,
                    f"revoke_access_for_{user_id}",
                    {"revoked_user": user_id}
                )
                
                self.logger.info(f"Access revoked for user {user_id} on resource {resource_id}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error revoking access: {str(e)}")
            raise PolicyEngineError(f"Failed to revoke access: {str(e)}")
    
    def get_user_permissions(
        self,
        user_id: str,
        resource_type: Optional[str] = None
    ) -> List[AccessControlEntry]:
        """Get all permissions for a user."""
        try:
            with get_db_session() as session:
                query = session.query(AccessControlEntryModel).filter(
                    AccessControlEntryModel.principal_id == user_id,
                    AccessControlEntryModel.is_active == True
                )
                
                if resource_type:
                    query = query.filter(
                        AccessControlEntryModel.resource_type == resource_type
                    )
                
                access_entries = query.all()
                
                return [self._access_model_to_dataclass(entry) for entry in access_entries]
                
        except Exception as e:
            self.logger.error(f"Error getting user permissions: {str(e)}")
            raise PolicyEngineError(f"Failed to get user permissions: {str(e)}")
    
    def _validate_policy_rules(self, rules: List[Dict[str, Any]], policy_type: PolicyType) -> None:
        """Validate policy rules based on policy type."""
        required_fields = {
            PolicyType.ACCESS_CONTROL: ['condition', 'action'],
            PolicyType.DATA_CLASSIFICATION: ['classification_rules'],
            PolicyType.DATA_RETENTION: ['retention_period'],
            PolicyType.DATA_QUALITY: ['quality_thresholds'],
            PolicyType.PRIVACY_PROTECTION: ['privacy_rules'],
            PolicyType.COMPLIANCE: ['compliance_requirements']
        }
        
        required = required_fields.get(policy_type, [])
        
        for rule in rules:
            for field in required:
                if field not in rule:
                    raise PolicyEngineError(f"Missing required field '{field}' in policy rule")
    
    def _policy_applies(
        self,
        policy: GovernancePolicyModel,
        resource_type: str,
        resource_id: str
    ) -> bool:
        """Check if policy applies to the given resource."""
        if not policy.applicable_resources:
            return True  # Policy applies to all resources
        
        return (
            resource_type in policy.applicable_resources or
            resource_id in policy.applicable_resources or
            "*" in policy.applicable_resources
        )
    
    def _evaluate_policy(
        self,
        policy: GovernancePolicyModel,
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a policy against user action."""
        violations = []
        allowed = True
        
        for rule in policy.rules:
            rule_result = self._evaluate_rule(
                rule, user_id, resource_id, resource_type, action, context
            )
            
            if not rule_result['allowed']:
                violations.append(rule_result['message'])
                allowed = False
        
        return {
            'allowed': allowed,
            'violations': violations
        }
    
    def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single policy rule."""
        # Simplified rule evaluation - would need more sophisticated logic
        condition = rule.get('condition', {})
        
        # Check action match
        if 'actions' in condition:
            if action not in condition['actions']:
                return {'allowed': True, 'message': ''}
        
        # Check resource type match
        if 'resource_types' in condition:
            if resource_type not in condition['resource_types']:
                return {'allowed': True, 'message': ''}
        
        # Check time-based conditions
        if 'time_restrictions' in condition:
            current_hour = datetime.utcnow().hour
            allowed_hours = condition['time_restrictions'].get('allowed_hours', [])
            if allowed_hours and current_hour not in allowed_hours:
                return {
                    'allowed': False,
                    'message': f'Action not allowed at current time: {current_hour}'
                }
        
        # Default to allowed if no conditions match
        return {'allowed': True, 'message': ''}
    
    def _access_level_sufficient(
        self,
        user_access: AccessLevel,
        required_access: AccessLevel
    ) -> bool:
        """Check if user access level is sufficient for required access."""
        access_hierarchy = {
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.OWNER: 4
        }
        
        return access_hierarchy[user_access] >= access_hierarchy[required_access]
    
    def _log_policy_enforcement(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str,
        allowed: bool,
        violations: List[str]
    ) -> None:
        """Log policy enforcement result."""
        self._log_audit_event(
            AuditEventType.SYSTEM_EVENT,
            user_id,
            resource_id,
            resource_type,
            f"policy_enforcement_{action}",
            {
                'allowed': allowed,
                'violations': violations
            },
            success=allowed
        )
    
    def _log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        resource_id: Optional[str],
        resource_type: Optional[str],
        action: str,
        details: Dict[str, Any],
        success: bool = True
    ) -> None:
        """Log an audit event."""
        try:
            with get_db_session() as session:
                audit_event = AuditEventModel(
                    event_type=event_type.value,
                    user_id=user_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    action=action,
                    details=details,
                    success=success
                )
                
                session.add(audit_event)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
    
    def _model_to_dataclass(self, model: GovernancePolicyModel) -> GovernancePolicy:
        """Convert database model to dataclass."""
        return GovernancePolicy(
            id=str(model.id),
            name=model.name,
            description=model.description or "",
            policy_type=PolicyType(model.policy_type),
            status=PolicyStatus(model.status),
            rules=model.rules or [],
            conditions=model.conditions or {},
            enforcement_level=model.enforcement_level,
            applicable_resources=model.applicable_resources or [],
            created_by=str(model.created_by) if model.created_by else "",
            approved_by=str(model.approved_by) if model.approved_by else None,
            created_at=model.created_at,
            updated_at=model.updated_at,
            effective_date=model.effective_date,
            expiry_date=model.expiry_date,
            version=model.version
        )
    
    def _access_model_to_dataclass(self, model: AccessControlEntryModel) -> AccessControlEntry:
        """Convert access control model to dataclass."""
        return AccessControlEntry(
            id=str(model.id),
            resource_id=model.resource_id,
            resource_type=model.resource_type,
            principal_id=str(model.principal_id),
            principal_type=model.principal_type,
            access_level=AccessLevel(model.access_level),
            granted_by=str(model.granted_by),
            granted_at=model.granted_at,
            expires_at=model.expires_at,
            conditions=model.conditions or {},
            is_active=model.is_active
        )