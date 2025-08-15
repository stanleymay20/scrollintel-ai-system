"""
Minimal ScrollMLEngineer Agent for debugging
"""

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus
from datetime import datetime
from uuid import uuid4

class ScrollMLEngineer(BaseAgent):
    """Minimal ScrollMLEngineer agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-ml-engineer",
            name="ScrollMLEngineer Agent",
            agent_type=AgentType.ML_ENGINEER
        )
        
        self.capabilities = [
            AgentCapability(
                name="ml_pipeline_setup",
                description="Set up automated ML pipelines with data preprocessing",
                input_types=["dataset", "requirements"],
                output_types=["pipeline", "configuration"]
            )
        ]
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process ML engineering requests."""
        return AgentResponse(
            id=str(uuid4()),
            request_id=request.id,
            content="ML Engineering response",
            artifacts=[],
            execution_time=0.1,
            status=ResponseStatus.SUCCESS
        )
    
    def get_capabilities(self):
        return self.capabilities
    
    async def health_check(self):
        return True