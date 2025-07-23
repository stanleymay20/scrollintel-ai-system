"""
ScrollIntel™ - Sovereign AI Intelligence System
A comprehensive AI system that replaces traditional data/AI teams through autonomous, specialized AI agents.
"""

__version__ = "1.0.0"
__author__ = "ScrollIntel Team"

# Core imports
from .core.interfaces import (
    BaseAgent,
    BaseEngine,
    BaseSecurityProvider,
    AgentType,
    AgentStatus,
    ResponseStatus,
    UserRole,
    AgentRequest,
    AgentResponse,
    AgentCapability,
    SecurityContext,
    AuditEvent,
    ConfigurationError,
    AgentError,
    EngineError,
    SecurityError,
)

from .core.config import (
    ScrollIntelConfig,
    get_config,
)

from .core.registry import (
    AgentRegistry,
    TaskOrchestrator,
)

__all__ = [
    # Core classes
    "BaseAgent",
    "BaseEngine", 
    "BaseSecurityProvider",
    # Enums
    "AgentType",
    "AgentStatus",
    "ResponseStatus",
    "UserRole",
    # Models
    "AgentRequest",
    "AgentResponse",
    "AgentCapability",
    "SecurityContext",
    "AuditEvent",
    # Exceptions
    "ConfigurationError",
    "AgentError",
    "EngineError",
    "SecurityError",
    # Configuration
    "ScrollIntelConfig",
    "get_config",
    # Registry
    "AgentRegistry",
    "TaskOrchestrator",
]