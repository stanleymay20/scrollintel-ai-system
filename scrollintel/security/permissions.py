"""
Role-based permission system for EXOUSIA security.
Handles user roles, permissions, and authorization checks.
"""

from typing import Dict, List, Set, Optional
from enum import Enum

from ..core.interfaces import UserRole, SecurityContext, SecurityError


class Permission(str, Enum):
    """System permissions."""
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_LIST = "user:list"
    
    # Agent management
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_LIST = "agent:list"
    AGENT_EXECUTE = "agent:execute"
    
    # Dataset management
    DATASET_CREATE = "dataset:create"
    DATASET_READ = "dataset:read"
    DATASET_UPDATE = "dataset:update"
    DATASET_DELETE = "dataset:delete"
    DATASET_LIST = "dataset:list"
    DATASET_UPLOAD = "dataset:upload"
    
    # Data management (file uploads)
    DATA_UPLOAD = "data:upload"
    DATA_READ = "data:read"
    DATA_CREATE = "data:create"
    DATA_DELETE = "data:delete"
    
    # ML Model management
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_LIST = "model:list"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    
    # Dashboard management
    DASHBOARD_CREATE = "dashboard:create"
    DASHBOARD_READ = "dashboard:read"
    DASHBOARD_UPDATE = "dashboard:update"
    DASHBOARD_DELETE = "dashboard:delete"
    DASHBOARD_LIST = "dashboard:list"
    DASHBOARD_SHARE = "dashboard:share"
    
    # System administration
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_HEALTH = "system:health"
    SYSTEM_METRICS = "system:metrics"
    
    # Audit and security
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"
    SECURITY_CONFIG = "security:config"


class ResourceType(str, Enum):
    """Resource types for permission checking."""
    USER = "user"
    AGENT = "agent"
    DATASET = "dataset"
    MODEL = "model"
    DASHBOARD = "dashboard"
    SYSTEM = "system"
    AUDIT = "audit"


class Action(str, Enum):
    """Actions that can be performed on resources."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    EXECUTE = "execute"
    UPLOAD = "upload"
    TRAIN = "train"
    DEPLOY = "deploy"
    SHARE = "share"
    CONFIG = "config"
    LOGS = "logs"
    HEALTH = "health"
    METRICS = "metrics"
    EXPORT = "export"


class RolePermissionManager:
    """Manages role-based permissions."""
    
    def __init__(self):
        self._role_permissions = self._initialize_role_permissions()
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize default permissions for each role."""
        return {
            UserRole.ADMIN: {
                # Full system access
                Permission.USER_CREATE,
                Permission.USER_READ,
                Permission.USER_UPDATE,
                Permission.USER_DELETE,
                Permission.USER_LIST,
                Permission.AGENT_CREATE,
                Permission.AGENT_READ,
                Permission.AGENT_UPDATE,
                Permission.AGENT_DELETE,
                Permission.AGENT_LIST,
                Permission.AGENT_EXECUTE,
                Permission.DATASET_CREATE,
                Permission.DATASET_READ,
                Permission.DATASET_UPDATE,
                Permission.DATASET_DELETE,
                Permission.DATASET_LIST,
                Permission.DATASET_UPLOAD,
                Permission.DATA_UPLOAD,
                Permission.DATA_READ,
                Permission.DATA_CREATE,
                Permission.DATA_DELETE,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_DELETE,
                Permission.MODEL_LIST,
                Permission.MODEL_TRAIN,
                Permission.MODEL_DEPLOY,
                Permission.DASHBOARD_CREATE,
                Permission.DASHBOARD_READ,
                Permission.DASHBOARD_UPDATE,
                Permission.DASHBOARD_DELETE,
                Permission.DASHBOARD_LIST,
                Permission.DASHBOARD_SHARE,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_LOGS,
                Permission.SYSTEM_HEALTH,
                Permission.SYSTEM_METRICS,
                Permission.AUDIT_READ,
                Permission.AUDIT_EXPORT,
                Permission.SECURITY_CONFIG,
            },
            
            UserRole.ANALYST: {
                # Data analysis and visualization
                Permission.USER_READ,  # Can read own profile
                Permission.AGENT_READ,
                Permission.AGENT_LIST,
                Permission.AGENT_EXECUTE,
                Permission.DATASET_CREATE,
                Permission.DATASET_READ,
                Permission.DATASET_UPDATE,
                Permission.DATASET_LIST,
                Permission.DATASET_UPLOAD,
                Permission.DATA_UPLOAD,
                Permission.DATA_READ,
                Permission.DATA_CREATE,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_LIST,
                Permission.MODEL_TRAIN,
                Permission.DASHBOARD_CREATE,
                Permission.DASHBOARD_READ,
                Permission.DASHBOARD_UPDATE,
                Permission.DASHBOARD_LIST,
                Permission.DASHBOARD_SHARE,
                Permission.SYSTEM_HEALTH,
            },
            
            UserRole.VIEWER: {
                # Read-only access
                Permission.USER_READ,  # Can read own profile
                Permission.AGENT_READ,
                Permission.AGENT_LIST,
                Permission.DATASET_READ,
                Permission.DATASET_LIST,
                Permission.MODEL_READ,
                Permission.MODEL_LIST,
                Permission.DASHBOARD_READ,
                Permission.DASHBOARD_LIST,
                Permission.SYSTEM_HEALTH,
            },
            
            UserRole.API: {
                # Programmatic access
                Permission.AGENT_READ,
                Permission.AGENT_LIST,
                Permission.AGENT_EXECUTE,
                Permission.DATASET_READ,
                Permission.DATASET_LIST,
                Permission.MODEL_READ,
                Permission.MODEL_LIST,
                Permission.DASHBOARD_READ,
                Permission.DASHBOARD_LIST,
                Permission.SYSTEM_HEALTH,
            }
        }
    
    def get_role_permissions(self, role: UserRole) -> Set[Permission]:
        """Get all permissions for a role."""
        return self._role_permissions.get(role, set())
    
    def has_permission(self, role: UserRole, permission: Permission) -> bool:
        """Check if a role has a specific permission."""
        return permission in self._role_permissions.get(role, set())
    
    def can_perform_action(self, role: UserRole, resource_type: ResourceType, action: Action) -> bool:
        """Check if a role can perform an action on a resource type."""
        permission_str = f"{resource_type.value}:{action.value}"
        try:
            permission = Permission(permission_str)
            return self.has_permission(role, permission)
        except ValueError:
            return False
    
    def add_permission_to_role(self, role: UserRole, permission: Permission) -> None:
        """Add a permission to a role."""
        if role not in self._role_permissions:
            self._role_permissions[role] = set()
        self._role_permissions[role].add(permission)
    
    def remove_permission_from_role(self, role: UserRole, permission: Permission) -> None:
        """Remove a permission from a role."""
        if role in self._role_permissions:
            self._role_permissions[role].discard(permission)
    
    def get_all_permissions(self) -> List[Permission]:
        """Get all available permissions."""
        return list(Permission)
    
    def get_resource_types(self) -> List[ResourceType]:
        """Get all resource types."""
        return list(ResourceType)
    
    def get_actions(self) -> List[Action]:
        """Get all actions."""
        return list(Action)


class PermissionChecker:
    """Handles permission checking logic."""
    
    def __init__(self):
        self.role_manager = RolePermissionManager()
    
    def check_permission(self, context: SecurityContext, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        # Check role-based permissions
        if self.role_manager.has_permission(context.role, permission):
            return True
        
        # Check user-specific permissions
        if permission.value in context.permissions:
            return True
        
        return False
    
    def check_resource_access(self, context: SecurityContext, resource_type: ResourceType, 
                            action: Action, resource_id: Optional[str] = None) -> bool:
        """Check if user can perform an action on a resource."""
        # Check basic permission
        if not self.role_manager.can_perform_action(context.role, resource_type, action):
            return False
        
        # Additional checks for specific resources
        if resource_id:
            return self._check_resource_ownership(context, resource_type, resource_id, action)
        
        return True
    
    def _check_resource_ownership(self, context: SecurityContext, resource_type: ResourceType,
                                resource_id: str, action: Action) -> bool:
        """Check resource ownership for additional security."""
        # Admin can access everything
        if context.role == UserRole.ADMIN:
            return True
        
        # For user resources, users can only access their own
        if resource_type == ResourceType.USER and action in [Action.READ, Action.UPDATE]:
            return resource_id == context.user_id
        
        # For other resources, implement ownership checks as needed
        # This would typically involve database queries to check ownership
        return True
    
    def require_permission(self, context: SecurityContext, permission: Permission) -> None:
        """Require a specific permission, raise SecurityError if not granted."""
        if not self.check_permission(context, permission):
            raise SecurityError(f"Permission denied: {permission.value}")
    
    def require_resource_access(self, context: SecurityContext, resource_type: ResourceType,
                              action: Action, resource_id: Optional[str] = None) -> None:
        """Require resource access, raise SecurityError if not granted."""
        if not self.check_resource_access(context, resource_type, action, resource_id):
            raise SecurityError(f"Access denied: {action.value} on {resource_type.value}")
    
    def get_user_permissions(self, context: SecurityContext) -> List[str]:
        """Get all permissions for a user."""
        role_permissions = self.role_manager.get_role_permissions(context.role)
        user_permissions = set(context.permissions)
        
        all_permissions = {p.value for p in role_permissions} | user_permissions
        return sorted(list(all_permissions))
    
    def can_access_admin_panel(self, context: SecurityContext) -> bool:
        """Check if user can access admin panel."""
        return context.role == UserRole.ADMIN
    
    def can_manage_users(self, context: SecurityContext) -> bool:
        """Check if user can manage other users."""
        return self.check_permission(context, Permission.USER_CREATE) or \
               self.check_permission(context, Permission.USER_UPDATE) or \
               self.check_permission(context, Permission.USER_DELETE)
    
    def can_execute_agents(self, context: SecurityContext) -> bool:
        """Check if user can execute AI agents."""
        return self.check_permission(context, Permission.AGENT_EXECUTE)
    
    def can_train_models(self, context: SecurityContext) -> bool:
        """Check if user can train ML models."""
        return self.check_permission(context, Permission.MODEL_TRAIN)
    
    def can_view_audit_logs(self, context: SecurityContext) -> bool:
        """Check if user can view audit logs."""
        return self.check_permission(context, Permission.AUDIT_READ)


# Global permission checker instance
permission_checker = PermissionChecker()


# Convenience functions for FastAPI dependencies
async def require_permission(user, permission_name: str) -> None:
    """Require a specific permission for FastAPI routes"""
    from ..models.database import User
    from ..core.interfaces import SecurityContext, UserRole
    
    if not isinstance(user, User):
        raise SecurityError("Invalid user object")
    
    # Create security context
    context = SecurityContext(
        user_id=str(user.id),
        role=user.role,
        permissions=user.permissions or [],
        session_id=None,
        ip_address=None
    )
    
    # Map permission name to Permission enum
    permission_map = {
        "audit.logs.view": Permission.AUDIT_READ,
        "audit.logs.export": Permission.AUDIT_EXPORT,
        "audit.logs.manage": Permission.AUDIT_EXPORT,  # Use export permission for management
        "compliance.reports.generate": Permission.AUDIT_EXPORT,
        "user.manage": Permission.USER_UPDATE,
        "system.admin": Permission.SYSTEM_CONFIG
    }
    
    permission = permission_map.get(permission_name)
    if not permission:
        # If permission not mapped, check if user is admin
        if user.role != UserRole.ADMIN:
            raise SecurityError(f"Permission denied: {permission_name}")
        return
    
    # Check permission
    permission_checker.require_permission(context, permission)