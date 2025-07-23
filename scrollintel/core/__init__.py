"""
ScrollIntel Core Module
Contains base classes, interfaces, and core system components.
"""

from .interfaces import (
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

from .config import (
    ScrollIntelConfig,
    DatabaseConfig,
    AIServiceConfig,
    SecurityConfig,
    SystemConfig,
    get_config,
    load_config_from_file,
)

from .registry import (
    AgentRegistry,
    TaskOrchestrator,
)

__all__ = [
    # Interfaces
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
    "DatabaseConfig",
    "AIServiceConfig",
    "SecurityConfig",
    "SystemConfig",
    "get_config",
    "load_config_from_file",
    # Registry
    "AgentRegistry",
    "TaskOrchestrator",
]