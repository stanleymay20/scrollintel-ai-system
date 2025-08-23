"""
Concrete Agent Implementation
"""
from .base import BaseAgent
from typing import Dict, Any, Optional
import asyncio

class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent"""
    
    def __init__(self, agent_id: str, name: str, description: str = "", capabilities: list = None):
        super().__init__(agent_id, name, description)
        self.capabilities = capabilities or []
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        try:
            # Basic processing logic
            result = {
                'agent_id': self.agent_id,
                'agent_name': self.name,
                'input_received': input_data,
                'processing_time': 0.1,
                'status': 'success',
                'output': f"Processed by {self.name}"
            }
            
            # Simulate processing time
            await asyncio.sleep(0.01)
            
            return result
            
        except Exception as e:
            return {
                'agent_id': self.agent_id,
                'agent_name': self.name,
                'status': 'error',
                'error': str(e)
            }
    
    def get_capabilities(self) -> list:
        """Get agent capabilities"""
        return self.capabilities
    
    def add_capability(self, capability: str):
        """Add a capability to the agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)

class QuickTestAgent(ConcreteAgent):
    """Quick test agent for validation"""
    
    def __init__(self):
        super().__init__(
            agent_id="quick_test_agent",
            name="Quick Test Agent",
            description="Agent for testing and validation",
            capabilities=["testing", "validation", "health_check"]
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': 'healthy',
            'capabilities': self.capabilities,
            'timestamp': time.time()
        }
