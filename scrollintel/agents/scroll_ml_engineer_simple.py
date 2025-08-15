"""
Simple test version of ScrollMLEngineer to debug import issues.
"""

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentCapability

class ScrollMLEngineerSimple(BaseAgent):
    """Simple test version."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-ml-engineer-simple",
            name="ScrollMLEngineer Simple",
            agent_type=AgentType.ML_ENGINEER
        )
        self.capabilities = []
    
    async def process_request(self, request):
        return None
    
    def get_capabilities(self):
        return self.capabilities
    
    async def health_check(self):
        return True