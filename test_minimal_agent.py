"""Test minimal agent to debug import issues."""

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentCapability

class TestAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="test-agent",
            name="Test Agent",
            agent_type=AgentType.ML_ENGINEER
        )
        self.capabilities = []
    
    async def process_request(self, request):
        return None
    
    def get_capabilities(self):
        return self.capabilities
    
    async def health_check(self):
        return True

if __name__ == "__main__":
    agent = TestAgent()
    print(f"Agent created: {agent.name}")