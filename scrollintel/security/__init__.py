"""
ScrollIntel Security Module (EXOUSIA)
Contains authentication, authorization, and audit logging components.
"""

from ..core.interfaces import (
    BaseSecurityProvider,
    UserRole,
    SecurityContext,
    AuditEvent,
    SecurityError,
)

from .auth import JWTAuthenticator, PasswordManager, authenticator
from .permissions import (
    Permission, ResourceType, Action, RolePermissionManager, 
    PermissionChecker, permission_checker
)
from .audit import AuditLogger, AuditAction, audit_logger
from .session import SessionManager, SessionData, session_manager
from .middleware import (
    SecurityMiddleware, RateLimitMiddleware, PermissionDependency,
    ResourceAccessDependency, require_permission, require_resource_access,
    require_admin, require_user_management, require_agent_execution,
    require_model_training, require_audit_access, security_scheme
)

__all__ = [
    # Core interfaces
    "BaseSecurityProvider",
    "UserRole",
    "SecurityContext", 
    "AuditEvent",
    "SecurityError",
    
    # Authentication
    "JWTAuthenticator",
    "PasswordManager", 
    "authenticator",
    
    # Permissions
    "Permission",
    "ResourceType",
    "Action",
    "RolePermissionManager",
    "PermissionChecker",
    "permission_checker",
    
    # Audit logging
    "AuditLogger",
    "AuditAction",
    "audit_logger",
    
    # Session management
    "SessionManager",
    "SessionData",
    "session_manager",
    
    # Middleware
    "SecurityMiddleware",
    "RateLimitMiddleware",
    "PermissionDependency",
    "ResourceAccessDependency",
    "require_permission",
    "require_resource_access",
    "require_admin",
    "require_user_management", 
    "require_agent_execution",
    "require_model_training",
    "require_audit_access",
    "security_scheme",
]