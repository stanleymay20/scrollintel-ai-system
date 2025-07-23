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

__all__ = [
    "BaseSecurityProvider",
    "UserRole",
    "SecurityContext", 
    "AuditEvent",
    "SecurityError",
]