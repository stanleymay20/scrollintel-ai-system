"""
Role-Based Access Control (RBAC) System
Implements principle of least privilege enforcement
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import secrets

logger = logging.getLogger(__name__)

class PermissionType(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"

@dataclass
class Permission:
    permission_id: str
    name: str
    description: str
    resource_type: str
    actions: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Role:
    role_id: str
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UserRoleAssignment:
    assignment_id: str
    user_id: str
    role_id: str
    assigned_by: str
    assigned_at: datetime
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

@dataclass
class AccessContext:
    user_id: str
    resource_id: str
    resource_type: str
    action: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_context: Dict[str, Any] = field(default_factory=dict)

class RBACSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.user_role_assignments: Dict[str, List[UserRoleAssignment]] = {}
        self.role_hierarchy_cache: Dict[str, Set[str]] = {}
        
        # Initialize system roles and permissions
        self._initialize_system_roles()
    
    def create_permission(self, name: str, description: str, resource_type: str,
                         actions: List[str], conditions: Optional[Dict[str, Any]] = None) -> str:
        """Create a new permission"""
        try:
            permission_id = secrets.token_urlsafe(16)
            
            permission = Permission(
                permission_id=permission_id,
                name=name,
                description=description,
                resource_type=resource_type,
                actions=actions,
                conditions=conditions or {}
            )
            
            self.permissions[permission_id] = permission
            
            logger.info(f"Permission created: {name} ({permission_id})")
            return permission_id
            
        except Exception as e:
            logger.error(f"Permission creation failed: {str(e)}")
            raise
    
    def create_role(self, name: str, description: str, 
                   permissions: Optional[List[str]] = None,
                   parent_roles: Optional[List[str]] = None) -> str:
        """Create a new role"""
        try:
            role_id = secrets.token_urlsafe(16)
            
            # Validate permissions exist
            permission_set = set()
            if permissions:
                for perm_id in permissions:
                    if perm_id not in self.permissions:
                        raise ValueError(f"Permission not found: {perm_id}")
                    permission_set.add(perm_id)
            
            # Validate parent roles exist
            parent_role_set = set()
            if parent_roles:
                for parent_id in parent_roles:
                    if parent_id not in self.roles:
                        raise ValueError(f"Parent role not found: {parent_id}")
                    parent_role_set.add(parent_id)
            
            role = Role(
                role_id=role_id,
                name=name,
                description=description,
                permissions=permission_set,
                parent_roles=parent_role_set
            )
            
            self.roles[role_id] = role
            
            # Clear hierarchy cache
            self.role_hierarchy_cache.clear()
            
            logger.info(f"Role created: {name} ({role_id})")
            return role_id
            
        except Exception as e:
            logger.error(f"Role creation failed: {str(e)}")
            raise
    
    def assign_role_to_user(self, user_id: str, role_id: str, assigned_by: str,
                           expires_at: Optional[datetime] = None,
                           conditions: Optional[Dict[str, Any]] = None) -> str:
        """Assign role to user"""
        try:
            if role_id not in self.roles:
                raise ValueError(f"Role not found: {role_id}")
            
            assignment_id = secrets.token_urlsafe(16)
            
            assignment = UserRoleAssignment(
                assignment_id=assignment_id,
                user_id=user_id,
                role_id=role_id,
                assigned_by=assigned_by,
                assigned_at=datetime.utcnow(),
                expires_at=expires_at,
                conditions=conditions or {}
            )
            
            if user_id not in self.user_role_assignments:
                self.user_role_assignments[user_id] = []
            
            self.user_role_assignments[user_id].append(assignment)
            
            logger.info(f"Role {role_id} assigned to user {user_id}")
            return assignment_id
            
        except Exception as e:
            logger.error(f"Role assignment failed: {str(e)}")
            raise
    
    def revoke_role_from_user(self, user_id: str, role_id: str, revoked_by: str) -> bool:
        """Revoke role from user"""
        try:
            if user_id not in self.user_role_assignments:
                return False
            
            assignments = self.user_role_assignments[user_id]
            revoked = False
            
            for assignment in assignments:
                if assignment.role_id == role_id and assignment.is_active:
                    assignment.is_active = False
                    assignment.conditions["revoked_by"] = revoked_by
                    assignment.conditions["revoked_at"] = datetime.utcnow().isoformat()
                    revoked = True
            
            if revoked:
                logger.info(f"Role {role_id} revoked from user {user_id}")
            
            return revoked
            
        except Exception as e:
            logger.error(f"Role revocation failed: {str(e)}")
            return False
    
    def check_permission(self, context: AccessContext) -> bool:
        """Check if user has permission for the requested action"""
        try:
            user_permissions = self.get_user_effective_permissions(context.user_id)
            
            # Find matching permissions
            for perm_id in user_permissions:
                permission = self.permissions[perm_id]
                
                # Check resource type match
                if (permission.resource_type == "*" or 
                    permission.resource_type == context.resource_type):
                    
                    # Check action match
                    if (context.action in permission.actions or 
                        "*" in permission.actions):
                        
                        # Check conditions
                        if self._check_permission_conditions(permission, context):
                            logger.debug(f"Permission granted for user {context.user_id}")
                            return True
            
            logger.warning(f"Permission denied for user {context.user_id}")
            return False
            
        except Exception as e:
            logger.error(f"Permission check failed: {str(e)}")
            return False
    
    def get_user_roles(self, user_id: str, include_inherited: bool = True) -> List[Role]:
        """Get roles assigned to user"""
        try:
            if user_id not in self.user_role_assignments:
                return []
            
            current_time = datetime.utcnow()
            active_role_ids = set()
            
            # Get directly assigned roles
            for assignment in self.user_role_assignments[user_id]:
                if (assignment.is_active and 
                    (assignment.expires_at is None or assignment.expires_at > current_time)):
                    active_role_ids.add(assignment.role_id)
            
            # Get inherited roles if requested
            if include_inherited:
                inherited_roles = set()
                for role_id in active_role_ids:
                    inherited_roles.update(self._get_inherited_roles(role_id))
                active_role_ids.update(inherited_roles)
            
            return [self.roles[role_id] for role_id in active_role_ids if role_id in self.roles]
            
        except Exception as e:
            logger.error(f"Get user roles failed: {str(e)}")
            return []
    
    def get_user_effective_permissions(self, user_id: str) -> Set[str]:
        """Get all effective permissions for user (including inherited)"""
        try:
            user_roles = self.get_user_roles(user_id, include_inherited=True)
            effective_permissions = set()
            
            for role in user_roles:
                effective_permissions.update(role.permissions)
            
            return effective_permissions
            
        except Exception as e:
            logger.error(f"Get effective permissions failed: {str(e)}")
            return set()
    
    def _get_inherited_roles(self, role_id: str) -> Set[str]:
        """Get all inherited roles for a role (recursive)"""
        if role_id in self.role_hierarchy_cache:
            return self.role_hierarchy_cache[role_id]
        
        inherited = set()
        
        if role_id in self.roles:
            role = self.roles[role_id]
            
            for parent_id in role.parent_roles:
                inherited.add(parent_id)
                inherited.update(self._get_inherited_roles(parent_id))
        
        self.role_hierarchy_cache[role_id] = inherited
        return inherited
    
    def _check_permission_conditions(self, permission: Permission, context: AccessContext) -> bool:
        """Check if permission conditions are met"""
        try:
            conditions = permission.conditions
            
            if not conditions:
                return True
            
            # Time-based conditions
            if "allowed_hours" in conditions:
                current_hour = context.timestamp.hour
                if current_hour not in conditions["allowed_hours"]:
                    return False
            
            # IP-based conditions
            if "allowed_ips" in conditions and context.ip_address:
                if context.ip_address not in conditions["allowed_ips"]:
                    return False
            
            # Resource-specific conditions
            if "resource_conditions" in conditions:
                resource_conditions = conditions["resource_conditions"]
                
                # Check resource ownership
                if "owner_only" in resource_conditions and resource_conditions["owner_only"]:
                    resource_owner = context.additional_context.get("resource_owner")
                    if resource_owner != context.user_id:
                        return False
                
                # Check resource attributes
                if "required_attributes" in resource_conditions:
                    required_attrs = resource_conditions["required_attributes"]
                    resource_attrs = context.additional_context.get("resource_attributes", {})
                    
                    for attr, value in required_attrs.items():
                        if resource_attrs.get(attr) != value:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Permission condition check failed: {str(e)}")
            return False
    
    def get_role_permissions(self, role_id: str, include_inherited: bool = True) -> Set[str]:
        """Get all permissions for a role"""
        if role_id not in self.roles:
            return set()
        
        role = self.roles[role_id]
        permissions = set(role.permissions)
        
        if include_inherited:
            inherited_roles = self._get_inherited_roles(role_id)
            for inherited_role_id in inherited_roles:
                if inherited_role_id in self.roles:
                    permissions.update(self.roles[inherited_role_id].permissions)
        
        return permissions
    
    def cleanup_expired_assignments(self):
        """Remove expired role assignments"""
        current_time = datetime.utcnow()
        expired_count = 0
        
        for user_id, assignments in self.user_role_assignments.items():
            for assignment in assignments:
                if (assignment.is_active and 
                    assignment.expires_at and 
                    assignment.expires_at <= current_time):
                    assignment.is_active = False
                    expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Expired {expired_count} role assignments")
    
    def audit_user_permissions(self, user_id: str) -> Dict[str, Any]:
        """Generate audit report for user permissions"""
        try:
            user_roles = self.get_user_roles(user_id, include_inherited=True)
            effective_permissions = self.get_user_effective_permissions(user_id)
            
            # Get assignment details
            assignments = []
            if user_id in self.user_role_assignments:
                for assignment in self.user_role_assignments[user_id]:
                    if assignment.is_active:
                        assignments.append({
                            "role_id": assignment.role_id,
                            "role_name": self.roles[assignment.role_id].name if assignment.role_id in self.roles else "Unknown",
                            "assigned_by": assignment.assigned_by,
                            "assigned_at": assignment.assigned_at.isoformat(),
                            "expires_at": assignment.expires_at.isoformat() if assignment.expires_at else None
                        })
            
            # Get permission details
            permission_details = []
            for perm_id in effective_permissions:
                if perm_id in self.permissions:
                    perm = self.permissions[perm_id]
                    permission_details.append({
                        "permission_id": perm_id,
                        "name": perm.name,
                        "resource_type": perm.resource_type,
                        "actions": perm.actions
                    })
            
            return {
                "user_id": user_id,
                "audit_timestamp": datetime.utcnow().isoformat(),
                "active_roles": len(user_roles),
                "effective_permissions": len(effective_permissions),
                "role_assignments": assignments,
                "permission_details": permission_details
            }
            
        except Exception as e:
            logger.error(f"User permission audit failed: {str(e)}")
            return {}
    
    def _initialize_system_roles(self):
        """Initialize default system roles and permissions"""
        try:
            # Create basic permissions
            read_perm = self.create_permission(
                "read_access", "Read access to resources", "*", ["read"]
            )
            
            write_perm = self.create_permission(
                "write_access", "Write access to resources", "*", ["write", "create", "update"]
            )
            
            delete_perm = self.create_permission(
                "delete_access", "Delete access to resources", "*", ["delete"]
            )
            
            admin_perm = self.create_permission(
                "admin_access", "Administrative access", "*", ["*"]
            )
            
            # Create system roles
            viewer_role = self.create_role(
                "viewer", "Read-only access", [read_perm]
            )
            
            editor_role = self.create_role(
                "editor", "Read and write access", [read_perm, write_perm], [viewer_role]
            )
            
            admin_role = self.create_role(
                "admin", "Full administrative access", [admin_perm]
            )
            
            # Mark as system roles
            for role_id in [viewer_role, editor_role, admin_role]:
                self.roles[role_id].is_system_role = True
            
            logger.info("System roles and permissions initialized")
            
        except Exception as e:
            logger.error(f"System role initialization failed: {str(e)}")
            raise