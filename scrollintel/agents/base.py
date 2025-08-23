"""
Base agent class for ScrollIntel agents
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all ScrollIntel agents"""
    
    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = "inactive"
        self.created_at = datetime.utcnow()
        self.last_activity = None
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "last_activity": self.last_activity,
            "healthy": True
        }
    
    def activate(self):
        """Activate the agent"""
        self.status = "active"
        self.last_activity = datetime.utcnow()
        logger.info(f"Agent {self.name} activated")
    
    def deactivate(self):
        """Deactivate the agent"""
        self.status = "inactive"
        logger.info(f"Agent {self.name} deactivated")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.agent_id}, name={self.name}, status={self.status})>"
