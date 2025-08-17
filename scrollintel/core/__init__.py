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
    get_config,
    get_default_config,
    validate_config,
)

from .registry import (
    AgentRegistry,
)

from .orchestrator import (
    TaskOrchestrator,
    Workflow,
    WorkflowTask,
    TaskDependency,
    WorkflowTemplate,
    TaskStatus,
    WorkflowStatus,
)

from .message_bus import (
    MessageBus,
    Message,
    MessageType,
    MessagePriority,
    MessageHandler,
    get_message_bus,
    initialize_message_bus,
    shutdown_message_bus,
)

from .workflow_templates import (
    WorkflowTemplateLibrary,
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
    "get_config",
    "get_default_config",
    "validate_config",
    # Registry
    "AgentRegistry",
    # Orchestration
    "TaskOrchestrator",
    "Workflow",
    "WorkflowTask",
    "TaskDependency",
    "WorkflowTemplate",
    "TaskStatus",
    "WorkflowStatus",
    # Message Bus
    "MessageBus",
    "Message",
    "MessageType",
    "MessagePriority",
    "MessageHandler",
    "get_message_bus",
    "initialize_message_bus",
    "shutdown_message_bus",
    # Workflow Templates
    "WorkflowTemplateLibrary",
]