"""
Base Agent class for all ScrollIntel Core agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AgentRequest(BaseModel):
    """Standard request format for all agents"""
    query: str
    context: Dict[str, Any] = {}
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parameters: Dict[str, Any] = {}
    request_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high
    timeout: Optional[int] = None


class AgentResponse(BaseModel):
    """Standard response format for all agents"""
    agent_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = {}
    processing_time: float = 0.0
    timestamp: datetime = datetime.utcnow()
    request_id: Optional[str] = None
    confidence_score: Optional[float] = None
    suggestions: List[str] = []


class Agent(ABC):
    """Base class for all ScrollIntel agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_healthy = True
        self.last_health_check = datetime.utcnow()
        self.capabilities = self.get_capabilities()
        logger.info(f"Initialized {self.name} agent")
    
    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a request and return a response"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        try:
            # Perform basic health check
            self.last_health_check = datetime.utcnow()
            self.is_healthy = True
            
            return {
                "agent": self.name,
                "healthy": self.is_healthy,
                "last_check": self.last_health_check.isoformat(),
                "capabilities": self.capabilities
            }
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            self.is_healthy = False
            return {
                "agent": self.name,
                "healthy": False,
                "error": str(e),
                "last_check": self.last_health_check.isoformat()
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "healthy": self.is_healthy
        }