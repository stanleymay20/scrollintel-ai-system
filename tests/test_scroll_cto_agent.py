"""
Unit tests for ScrollCTOAgent.
Tests the technical decision-making capabilities and various request handling methods.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.agents.scroll_cto_agent import (
    ScrollCTOAgent, 
    TechnologyStack, 
    DatabaseType, 
    CloudProvider, 
    ArchitectureTemplate,
    TechnologyComparison,
    ScalingStrategy
)
from scrollintel.core.interfaces import (
    AgentRequest,
    AgentResponse,
    AgentCapability,
    AgentType,
    AgentStatus,
    ResponseStatus,
)


class TestScrollCTOAgent:
    """Test suite for ScrollCTOAgent."""
    
    @pytest.fixture
    def cto_agent(self):
        """Create a ScrollCTOAgent instance for testing."""
        return ScrollCTOAgent()
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample agent request for testing."""
        return AgentRequest(
            id="test-request-1",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Design a scalable web application architecture",
            context={"business_type": "startup", "expected_users": 10000, "budget": 500},
            created_at=datetime.utcnow()
        )
    
    def test_agent_initialization(self, cto_agent):
        """Test that the CTO agent initializes correctly."""
        assert cto_agent.agent_id == "scroll-cto-agent"
        assert cto_agent.name == "ScrollCTO Agent"
        assert cto_agent.agent_type == AgentType.CTO
        
        # Check that capabilities are properly initialized
        capabilities = cto_agent.get_capabilities()
        assert len(capabilities) == 4
        
        capability_names = [cap.name for cap in capabilities]
        expected_capabilities = [
            "architecture_design",
            "technology_comparison", 
            "scaling_strategy",
            "technical_decision"
        ]
        
        for expected_cap in expected_capabilities:
            assert expected_cap in capability_names
    
    def test_architecture_templates_initialization(self, cto_agent):
        """Test that architecture templates are properly initialized."""
        templates = cto_agent.get_architecture_templates()
        assert len(templates) == 3
        
        template_names = [template.name for template in templates]
        expected_templates = ["Startup MVP", "Enterprise Scale", "AI/ML Platform"]
        
        for expected_template in expected_templates:
            assert expected_template in template_names
        
        # Test template properties
        startup_template = next(t for t in templates if t.name == "Startup MVP")
        assert startup_template.tech_stack == TechnologyStack.PYTHON_FASTAPI
        assert startup_template.database == DatabaseType.POSTGRESQL
        assert startup_template.cloud_provider == CloudProvider.RENDER
        assert startup_template.estimated_cost_monthly == 50.0
    
    @pytest.mark.asyncio
    async def test_architecture_request_handling(self, cto_agent, sample_request):
        """Test architecture request handling."""
        sample_request.prompt = "Design a microservices architecture for a web application"
        
        response = await cto_agent.process_request(sample_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Architecture Recommendation" in response.content
        assert "Technology Stack" in response.content
        assert "Cost Analysis" in response.content
        assert "Implementation Roadmap" in response.content
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_technology_comparison_handling(self, cto_agent, sample_request):
        """Test technology comparison request handling."""
        sample_request.prompt = "Compare Python FastAPI vs Node.js Express for my project"
        
        response = await cto_agent.process_request(sample_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Technology Comparison Analysis" in response.content
        assert "Scoring Matrix" in response.content
        assert "Recommendation" in response.content
        assert "Cost Analysis" in response.content
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_scaling_strategy_handling(self, cto_agent, sample_request):
        """Test scaling strategy request handling."""
        sample_request.prompt = "How can I scale my application to handle more users?"
        sample_request.context = {
            "current_users": 1000,
            "projected_users": 50000,
            "current_cost": 200
        }
        
        response = await cto_agent.process_request(sample_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Scaling Strategy" in response.content
        assert "Infrastructure Changes Required" in response.content
        assert "Cost Impact" in response.content
        assert "Risk Assessment" in response.content
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_technical_decision_handling(self, cto_agent, sample_request):
        """Test technical decision request handling."""
        sample_request.prompt = "Should we migrate from monolith to microservices?"
        
        response = await cto_agent.process_request(sample_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Technical Decision Analysis" in response.content
        assert "Decision Framework Applied" in response.content
        assert "Recommendation" in response.content
        assert "Success Criteria" in response.content
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cto_agent):
        """Test error handling in request processing."""
        # Create a request that might cause an error
        bad_request = AgentRequest(
            id="bad-request",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="",  # Empty prompt
            created_at=datetime.utcnow()
        )
        
        # Mock an exception in the processing
        with patch.object(cto_agent, '_generate_architecture_recommendation', side_effect=Exception("Test error")):
            response = await cto_agent.process_request(bad_request)
            
            assert response.status == ResponseStatus.ERROR
            assert "Error processing CTO request" in response.content
            assert response.error_message == "Test error"
            assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, cto_agent):
        """Test the health check functionality."""
        # Test healthy agent
        is_healthy = await cto_agent.health_check()
        assert is_healthy is True
    
    def test_capability_details(self, cto_agent):
        """Test that capabilities have proper details."""
        capabilities = cto_agent.get_capabilities()
        
        for capability in capabilities:
            assert isinstance(capability, AgentCapability)
            assert capability.name
            assert capability.description
            assert len(capability.input_types) > 0
            assert len(capability.output_types) > 0
    
    def test_technology_stack_enum(self):
        """Test the TechnologyStack enum."""
        assert TechnologyStack.PYTHON_FASTAPI.value == "python_fastapi"
        assert TechnologyStack.NODE_EXPRESS.value == "node_express"
        assert TechnologyStack.JAVA_SPRING.value == "java_spring"
    
    def test_database_type_enum(self):
        """Test the DatabaseType enum."""
        assert DatabaseType.POSTGRESQL.value == "postgresql"
        assert DatabaseType.MYSQL.value == "mysql"
        assert DatabaseType.MONGODB.value == "mongodb"
    
    def test_cloud_provider_enum(self):
        """Test the CloudProvider enum."""
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.VERCEL.value == "vercel"
    
    def test_architecture_template_dataclass(self):
        """Test the ArchitectureTemplate dataclass."""
        template = ArchitectureTemplate(
            name="Test Architecture",
            description="Test description",
            tech_stack=TechnologyStack.PYTHON_FASTAPI,
            database=DatabaseType.POSTGRESQL,
            cloud_provider=CloudProvider.AWS,
            estimated_cost_monthly=100.0,
            scalability_rating=8,
            complexity_rating=5,
            use_cases=["Web apps"],
            pros=["Fast", "Scalable"],
            cons=["Complex"]
        )
        
        assert template.name == "Test Architecture"
        assert template.description == "Test description"
        assert template.tech_stack == TechnologyStack.PYTHON_FASTAPI
        assert template.estimated_cost_monthly == 100.0
        assert "Web apps" in template.use_cases
        assert "Fast" in template.pros
        assert "Complex" in template.cons
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, cto_agent):
        """Test handling multiple concurrent requests."""
        requests = []
        for i in range(3):
            request = AgentRequest(
                id=f"concurrent-request-{i}",
                user_id="test-user",
                agent_id="scroll-cto-agent",
                prompt=f"Design architecture for project {i}",
                context={"business_type": "startup", "expected_users": 1000 * (i + 1)},
                created_at=datetime.utcnow()
            )
            requests.append(request)
        
        # Process requests concurrently
        tasks = [cto_agent.process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses are successful
        for response in responses:
            assert response.status == ResponseStatus.SUCCESS
            assert "Architecture Recommendation" in response.content
            assert response.execution_time > 0
    
    def test_add_architecture_template(self, cto_agent):
        """Test adding new architecture templates."""
        initial_count = len(cto_agent.get_architecture_templates())
        
        new_template = ArchitectureTemplate(
            name="Custom Template",
            description="Custom architecture",
            tech_stack=TechnologyStack.GOLANG_GIN,
            database=DatabaseType.MONGODB,
            cloud_provider=CloudProvider.GCP,
            estimated_cost_monthly=300.0,
            scalability_rating=7,
            complexity_rating=6,
            use_cases=["Custom apps"],
            pros=["Fast", "Efficient"],
            cons=["New technology"]
        )
        
        cto_agent.add_architecture_template(new_template)
        
        templates = cto_agent.get_architecture_templates()
        assert len(templates) == initial_count + 1
        assert any(t.name == "Custom Template" for t in templates)


if __name__ == "__main__":
    pytest.main([__file__])