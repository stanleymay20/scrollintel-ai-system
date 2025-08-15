"""
Access Control and Permission Management System

This module provides comprehensive access control and permission management
for the advanced prompt management system, ensuring secure and compliant
access to prompt resources.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models.audit_models import (
    AccessControl, AuditAction, RiskLevel,
    AccessControlCreate, AccessControlResponse
)
from ..models.database_utils import get_sync_db
from ..core.audit_logger import audit_logger, audit_action
from ..core.config import get_settings


class Permission:
    """Permission constants"""
    # Prompt permissions
    PROMPT_CREATE = "prompt:create"
    PROMPT_READ = "prompt:read"
    PROMPT_UPDATE = "prompt:update"
    PROMPT_DELETE = "prompt:delete"
    PROMPT_EXPORT = "prompt:export"
    PROMPT_IMPORT = "prompt:import"
    
    # Version control permissions
    VERSION_CREATE = "version:create"
    VERSION_READ = "version:read"
    VERSION_ROLLBACK = "version:rollback"
    
    # Experiment permissions
    EXPERIMENT_CREATE = "experiment:create"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_UPDATE = "experiment:update"
    EXPERIMENT_DELETE = "experiment:delete"
    EXPERIMENT_RUN = "experiment:run"
    
    # Optimization permissions
    OPTIMIZATION_CREATE = "optimization:create"
    OPTIMIZATION_READ = "optimization:read"
    OPTIMIZATION_CANCEL = "optimization:cancel"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"
    
    # Admin permissions
    ADMIN_USER_MANAGE = "admin:user_manage"
    ADMIN_AUDIT_READ = "admin:audit_read"
    ADMIN_COMPLIANCE_MANAGE = "admin:compliance_manage"
    ADMIN_SYSTEM_CONFIG = "admin:system_config"


class Role:
    """Predefined roles with permission sets"""
    
    VIEWER = {
        "name": "viewer",
        "description": "Read-only access to prompts and analytics",
        "permissions": {
            Permission.PROMPT_READ,
            Permission.VERSION_READ,
            Permission.EXPERIMENT_READ,
            Permission.OPTIMIZATION_READ,
            Permission.ANALYTICS_READ
        }
    }
    
    PROMPT_ENGINEER = {
        "name": "prompt_engineer",
        "description": "Full prompt management capabilities",
        "permissions": {
            Permission.PROMPT_CREATE,
            Permission.PROMPT_READ,
            Permission.PROMPT_UPDATE,
            Permission.PROMPT_DELETE,
            Permission.PROMPT_EXPORT,
            Permission.PROMPT_IMPORT,
            Permission.VERSION_CREATE,
            Permission.VERSION_READ,
            Permission.VERSION_ROLLBACK,
            Permission.EXPERIMENT_CREATE,
            Permission.EXPERIMENT_READ,
            Permission.EXPERIMENT_UPDATE,
            Permission.EXPERIMENT_RUN,
            Permission.OPTIMIZATION_CREATE,
            Permission.OPTIMIZATION_READ,
            Permission.ANALYTICS_READ
        }
    }
    
    DATA_SCIENTIST = {
        "name": "data_scientist",
        "description": "Advanced analytics and optimization access",
        "permissions": {
            Permission.PROMPT_READ,
            Permission.VERSION_READ,
            Permission.EXPERIMENT_CREATE,
            Permission.EXPERIMENT_READ,
            Permission.EXPERIMENT_UPDATE,
            Permission.EXPERIMENT_RUN,
            Permission.OPTIMIZATION_CREATE,
            Permission.OPTIMIZATION_READ,
            Permission.OPTIMIZATION_CANCEL,
            Permission.ANALYTICS_READ,
            Permission.ANALYTICS_EXPORT
        }
    }
    
    TEAM_LEAD = {
        "name": "team_lead",
        "description": "Team management and oversight capabilities",
        "permissions": {
            Permission.PROMPT_CREATE,
            Permission.PROMPT_READ,
            Permission.PROMPT_UPDATE,
            Permission.PROMPT_EXPORT,
            Permission.PROMPT_IMPORT,
            Permission.VERSION_READ,
            Permission.EXPERIMENT_READ,
            Permission.OPTIMIZATION_READ,
            Permission.ANALYTICS_READ,
            Permission.ANALYTICS_EXPORT,
            Permission.ADMIN_AUDIT_READ
        }
    }
    
    COMPLIANCE_OFFICER = {
        "name": "compliance_officer",
        "description": "Compliance monitoring and audit access",
        "permissions": {
            Permission.PROMPT_READ,
            Permission.VERSION_READ,
            Permission.EXPERIMENT_READ,
            Permission.ANALYTICS_READ,
            Permission.ANALYTICS_EXPORT,
            Permission.ADMIN_AUDIT_READ,
            Permission.ADMIN_COMPLIANCE_MANAGE
        }
    }
    
    ADMIN = {
        "name": "admin",
        "description": "Full system administration access",
        "permissions": {
            # All permissions
            Permission.PROMPT_CREATE,
            Permission.PROMPT_READ,
            Permission.PROMPT_UPDATE,
            Permission.PROMPT_DELETE,
            Permission.PROMPT_EXPORT,
            Permission.PROMPT_IMPORT,
            Permission.VERSION_CREATE,
            Permission.VERSION_READ,
            Permission.VERSION_ROLLBACK,
            Permission.EXPERIMENT_CREATE,
            Permission.EXPERIMENT_READ,
            Permission.EXPERIMENT_UPDATE,
            Permission.EXPERIMENT_DELETE,
            Permission.EXPERIMENT_RUN,
            Permission.OPTIMIZATION_CREATE,
            Permission.OPTIMIZATION_READ,
            Permission.OPTIMIZATION_CANCEL,
            Permission.ANALYTICS_READ,
            Permission.ANALYTICS_EXPORT,
            Permission.ADMIN_USER_MANAGE,
            Permission.ADMIN_AUDIT_READ,
            Permission.ADMIN_COMPLIANCE_MANAGE,
            Permission.ADMIN_SYSTEM_CONFIG
        }
    }


class AccessControlManager:
    """Comprehensive access control and permission management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.roles = {
            Role.VIEWER["name"]: Role.VIEWER,
            Role.PROMPT_ENGINEER["name"]: Role.PROMPT_ENGINEER,
            Role.DATA_SCIENTIST["name"]: Role.DATA_SCIENTIST,
            Role.TEAM_LEAD["name"]: Role.TEAM_LEAD,
            Role.COMPLIANCE_OFFICER["name"]: Role.COMPLIANCE_OFFICER,
            Role.ADMIN["name"]: Role.ADMIN
        }
    
    @audit_action(AuditAction.CREATE, "access_control", RiskLevel.HIGH)
    def grant_access(
        self,
        user_id: str,
        user_email: str,
        role: str,
        granted_by: str,
        custom_permissions: Optional[Set[str]] = None,
        resource_restrictions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> str:
        """
        Grant access permissions to a user
        
        Args:
            user_id: User ID to grant access to
            user_email: User email
            role: Role name to assign
            granted_by: User ID who is granting access
            custom_permissions: Custom permissions (overrides role permissions)
            resource_restrictions: Resource-level access restrictions
            expires_at: Access expiration date
            
        Returns:
            str: Access control entry ID
        """
        with get_sync_db() as db:
            # Validate role
            if role not in self.roles:
                raise ValueError(f"Invalid role: {role}")
            
            # Get permissions from role or use custom permissions
            if custom_permissions:
                permissions = list(custom_permissions)
            else:
                permissions = list(self.roles[role]["permissions"])
            
            # Check if user already has access
            existing_access = db.query(AccessControl).filter(
                and_(
                    AccessControl.user_id == user_id,
                    AccessControl.is_active == True
                )
            ).first()
            
            if existing_access:
                # Update existing access
                existing_access.role = role
                existing_access.permissions = permissions
                existing_access.resource_restrictions = resource_restrictions or {}
                existing_access.granted_by = granted_by
                existing_access.granted_at = datetime.utcnow()
                existing_access.expires_at = expires_at
                
                db.commit()
                return existing_access.id
            
            # Create new access control entry
            access_control = AccessControl(
                id=str(uuid.uuid4()),
                user_id=user_id,
                user_email=user_email,
                role=role,
                permissions=permissions,
                resource_restrictions=resource_restrictions or {},
                granted_by=granted_by,
                granted_at=datetime.utcnow(),
                expires_at=expires_at,
                is_active=True,
                access_count=0
            )
            
            db.add(access_control)
            db.commit()
            
            return access_control.id
    
    @audit_action(AuditAction.UPDATE, "access_control", RiskLevel.HIGH)
    def revoke_access(
        self,
        user_id: str,
        revoked_by: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Revoke user access permissions
        
        Args:
            user_id: User ID to revoke access from
            revoked_by: User ID who is revoking access
            reason: Reason for revocation
            
        Returns:
            bool: True if access was revoked
        """
        with get_sync_db() as db:
            access_control = db.query(AccessControl).filter(
                and_(
                    AccessControl.user_id == user_id,
                    AccessControl.is_active == True
                )
            ).first()
            
            if not access_control:
                return False
            
            access_control.is_active = False
            db.commit()
            
            return True
    
    def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None
    ) -> bool:
        """
        Check if user has specific permission
        
        Args:
            user_id: User ID to check
            permission: Permission to check
            resource_id: Specific resource ID (for resource-level restrictions)
            resource_type: Resource type (for resource-level restrictions)
            
        Returns:
            bool: True if user has permission
        """
        with get_sync_db() as db:
            access_control = db.query(AccessControl).filter(
                and_(
                    AccessControl.user_id == user_id,
                    AccessControl.is_active == True,
                    or_(
                        AccessControl.expires_at.is_(None),
                        AccessControl.expires_at > datetime.utcnow()
                    )
                )
            ).first()
            
            if not access_control:
                return False
            
            # Check if user has the permission
            if permission not in access_control.permissions:
                return False
            
            # Check resource-level restrictions
            if resource_id and resource_type and access_control.resource_restrictions:
                restrictions = access_control.resource_restrictions.get(resource_type, {})
                
                # Check if resource is explicitly denied
                denied_resources = restrictions.get("denied", [])
                if resource_id in denied_resources:
                    return False
                
                # Check if only specific resources are allowed
                allowed_resources = restrictions.get("allowed", [])
                if allowed_resources and resource_id not in allowed_resources:
                    return False
            
            # Update access tracking
            access_control.last_access = datetime.utcnow()
            access_control.access_count += 1
            db.commit()
            
            return True
    
    def get_user_permissions(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user permissions and access info
        
        Args:
            user_id: User ID to get permissions for
            
        Returns:
            Dict[str, Any]: User permissions and access information
        """
        with get_sync_db() as db:
            access_control = db.query(AccessControl).filter(
                and_(
                    AccessControl.user_id == user_id,
                    AccessControl.is_active == True,
                    or_(
                        AccessControl.expires_at.is_(None),
                        AccessControl.expires_at > datetime.utcnow()
                    )
                )
            ).first()
            
            if not access_control:
                return {
                    "has_access": False,
                    "role": None,
                    "permissions": [],
                    "resource_restrictions": {},
                    "expires_at": None
                }
            
            return {
                "has_access": True,
                "role": access_control.role,
                "permissions": access_control.permissions,
                "resource_restrictions": access_control.resource_restrictions,
                "granted_at": access_control.granted_at.isoformat(),
                "expires_at": access_control.expires_at.isoformat() if access_control.expires_at else None,
                "last_access": access_control.last_access.isoformat() if access_control.last_access else None,
                "access_count": access_control.access_count
            }
    
    def list_user_access(
        self,
        include_inactive: bool = False,
        role_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AccessControlResponse]:
        """
        List all user access entries
        
        Args:
            include_inactive: Include inactive access entries
            role_filter: Filter by specific role
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List[AccessControlResponse]: List of access control entries
        """
        with get_sync_db() as db:
            query = db.query(AccessControl)
            
            if not include_inactive:
                query = query.filter(AccessControl.is_active == True)
            
            if role_filter:
                query = query.filter(AccessControl.role == role_filter)
            
            access_entries = query.order_by(desc(AccessControl.granted_at)).offset(offset).limit(limit).all()
            
            return [AccessControlResponse.from_orm(entry) for entry in access_entries]
    
    def get_role_permissions(self, role: str) -> Dict[str, Any]:
        """
        Get permissions for a specific role
        
        Args:
            role: Role name
            
        Returns:
            Dict[str, Any]: Role information and permissions
        """
        if role not in self.roles:
            raise ValueError(f"Invalid role: {role}")
        
        return self.roles[role]
    
    def list_available_roles(self) -> List[Dict[str, Any]]:
        """
        List all available roles
        
        Returns:
            List[Dict[str, Any]]: List of available roles
        """
        return list(self.roles.values())
    
    def validate_permissions(self, permissions: List[str]) -> Dict[str, Any]:
        """
        Validate a list of permissions
        
        Args:
            permissions: List of permissions to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Get all valid permissions
        all_permissions = set()
        for role_info in self.roles.values():
            all_permissions.update(role_info["permissions"])
        
        valid_permissions = []
        invalid_permissions = []
        
        for permission in permissions:
            if permission in all_permissions:
                valid_permissions.append(permission)
            else:
                invalid_permissions.append(permission)
        
        return {
            "valid": valid_permissions,
            "invalid": invalid_permissions,
            "is_valid": len(invalid_permissions) == 0
        }
    
    def cleanup_expired_access(self) -> int:
        """
        Clean up expired access entries
        
        Returns:
            int: Number of entries cleaned up
        """
        with get_sync_db() as db:
            expired_entries = db.query(AccessControl).filter(
                and_(
                    AccessControl.is_active == True,
                    AccessControl.expires_at < datetime.utcnow()
                )
            ).all()
            
            count = len(expired_entries)
            
            for entry in expired_entries:
                entry.is_active = False
            
            db.commit()
            
            return count


# Global access control manager instance
access_control_manager = AccessControlManager()


# Decorator for permission checking
def require_permission(permission: str, resource_type: Optional[str] = None):
    """
    Decorator to require specific permission for function access
    
    Args:
        permission: Required permission
        resource_type: Resource type for resource-level checks
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract user context
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None)
            resource_id = kwargs.get('resource_id') or kwargs.get('id')
            
            if not user_id:
                raise PermissionError("User context required for permission check")
            
            # Check permission
            has_permission = access_control_manager.check_permission(
                user_id=user_id,
                permission=permission,
                resource_id=resource_id,
                resource_type=resource_type
            )
            
            if not has_permission:
                raise PermissionError(f"User {user_id} lacks required permission: {permission}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
