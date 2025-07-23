"""
Core interfaces and base classes for ScrollIntel system components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel


class AgentType(str, Enum):
    """Types of AI agents in the system."""
    CTO = "cto"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    AI_ENGINEER = "ai_engineer"
    ANALYST = "analyst"
    BI_DEVELOPER = "bi_developer"


class AgentStatus(str, Enum):
    """Status of an AI agent."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"


class ResponseStatus(str, Enum):
    """Status of an agent response."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class UserRole(str, Enum):
    """User roles for EXOUSIA security system."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API = "api"


class AgentRequest(BaseModel):
    """Request model for agent communication."""
    id: str
    user_id: str
    agent_id: str
    prompt: str
    context: Dict[str, Any] = {}
    priority: int = 1
    created_at: datetime


class AgentResponse(BaseModel):
    """Response model for agent communication."""
    id: str
    request_id: str
    content: str
    artifacts: List[str] = []
    execution_time: float
    status: ResponseStatus
    error_message: Optional[str] = None


class AgentCapability(BaseModel):
    """Capability definition for agents."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]


class BaseAgent(ABC):
    """Abstract base class for all ScrollIntel AI agents."""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.status = AgentStatus.INACTIVE
        self.capabilities: List[AgentCapability] = []
    
    @abstractmethod
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process an incoming request and return a response."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the capabilities of this agent."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the agent is healthy and ready to process requests."""
        pass
    
    def start(self) -> None:
        """Start the agent."""
        self.status = AgentStatus.ACTIVE
    
    def stop(self) -> None:
        """Stop the agent."""
        self.status = AgentStatus.INACTIVE


class BaseEngine(ABC):
    """Abstract base class for all ScrollIntel processing engines."""
    
    def __init__(self, engine_id: str, name: str):
        self.engine_id = engine_id
        self.name = name
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine with required resources."""
        pass
    
    @abstractmethod
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process input data and return results."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the engine."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the engine."""
        pass


class SecurityContext(BaseModel):
    """Security context for operations."""
    user_id: str
    role: UserRole
    permissions: List[str]
    session_id: str
    ip_address: str


class AuditEvent(BaseModel):
    """Audit event for logging."""
    id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str
    timestamp: datetime


class BaseSecurityProvider(ABC):
    """Abstract base class for security providers in EXOUSIA system."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[SecurityContext]:
        """Authenticate user and return security context."""
        pass
    
    @abstractmethod
    async def authorize(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Check if user is authorized to perform action on resource."""
        pass
    
    @abstractmethod
    async def audit_log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        pass
    
    @abstractmethod
    async def create_session(self, user_id: str) -> str:
        """Create a new session and return session ID."""
        pass
    
    @abstractmethod
    async def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate session and return security context."""
        pass


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class AgentError(Exception):
    """Exception raised for agent-related errors."""
    pass


class EngineError(Exception):
    """Exception raised for engine-related errors."""
    pass


class SecurityError(Exception):
    """Exception raised for security-related errors."""
    pass