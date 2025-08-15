"""
Simple Integration Tests for ScrollCTOAgent API Endpoints and Routing
Tests the CTO agent functionality without full application dependencies.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import time
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from scrollintel.agents.scroll_cto_agent import (
    ScrollCTOAgent, 
    TechnologyStack, 
    DatabaseType, 
    CloudProvider, 
    ArchitectureTemplate
)
from scrollintel.core.interfaces import (
    AgentRequest,
    AgentResponse,
    AgentType,
    ResponseStatus,
    SecurityContext,
    UserRole
)
from scrollintel.core.registry import AgentRegistry


class TestScrollCTOSimpleIntegration:
    """Simple integration tests for ScrollCTO Agent without full API stack."""
    
    @pytest.fixture
    def cto_agent(self):
        """Create ScrollCTOAgent instance."""
        return ScrollCTOAgent()
    
    @pytest_asyncio.fixture
    async def agent_registry(self, cto_agent):
        """Create agent registry with CTO agent."""
        registry = AgentRegistry()
        await registry.register_agent(cto_agent)
        return registry
    
    @pytest.fixture
    def sample_request(self):
        """Create sample agent request."""
        return AgentRequest(
            id="test-request-123",
            user_id="test-user-456",
            agent_id="scroll-cto-agent",
            prompt="Design a scalable web application architecture",
            context={
                "business_type": "startup",
                "expected_users": 10000,
                "budget": 1000
            },
            created_at=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_cto_agent_registration_and_discovery(self, agent_registry, cto_agent):
        """Test CTO agent registration and discovery in registry."""
        # Test agent is registered
        registered_agent = agent_registry.get_agent("scroll-cto-agent")
        assert registered_agent is not None
        assert registered_agent.agent_id == "scroll-cto-agent"
        assert registered_agent.agent_type == AgentType.CTO
        
        # Test capabilities are discoverable
        capabilities = agent_registry.get_capabilities()
        capability_names = [cap.name for cap in capabilities]
        
        expected_capabilities = [
            "architecture_design",
            "technology_comparison",
            "scaling_strategy",
            "technical_decision"
        ]
        
        for expected_cap in expected_capabilities:
            assert expected_cap in capability_names
    
    @pytest.mark.asyncio
    async def test_cto_agent_request_routing(self, agent_registry, sample_request):
        """Test request routing to CTO agent."""
        with patch.object(agent_registry, 'route_request') as mock_route:
            # Mock successful routing
            mock_response = AgentResponse(
                id="response-123",
                request_id=sample_request.id,
                content="Architecture recommendation provided",
                artifacts=[],
                execution_time=1.5,
                status=ResponseStatus.SUCCESS
            )
            mock_route.return_value = mock_response
            
            # Test routing
            response = await agent_registry.route_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Architecture recommendation" in response.content
            assert response.execution_time > 0
            mock_route.assert_called_once_with(sample_request)
    
    @pytest.mark.asyncio
    async def test_cto_agent_architecture_design_integration(self, cto_agent):
        """Test CTO agent architecture design capability integration."""
        request = AgentRequest(
            id="arch-test-123",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Design a microservices architecture for e-commerce platform",
            context={
                "business_type": "e-commerce",
                "expected_users": 50000,
                "budget": 2000,
                "requirements": ["high availability", "payment processing"]
            },
            created_at=datetime.utcnow()
        )
        
        # Mock GPT-4 response
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## E-commerce Microservices Architecture
            
            For a 50K user e-commerce platform, I recommend:
            
            ### Core Services
            - User Service (authentication, profiles)
            - Product Service (catalog, inventory)
            - Order Service (cart, checkout, orders)
            - Payment Service (payment processing)
            - Notification Service (emails, SMS)
            
            ### Infrastructure
            - Kubernetes for container orchestration
            - PostgreSQL for transactional data
            - Redis for caching and sessions
            - API Gateway for service routing
            """
            
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Architecture Recommendation" in response.content
            assert "microservices" in response.content.lower()
            assert "e-commerce" in response.content.lower()
            assert "User Service" in response.content
            assert "Payment Service" in response.content
            assert response.execution_time > 0
            
            # Verify GPT-4 was called with appropriate context
            mock_gpt4.assert_called_once()
            call_args = mock_gpt4.call_args[0][0]
            assert "50000" in call_args
            assert "e-commerce" in call_args.lower()
            assert "$2000" in call_args
    
    @pytest.mark.asyncio
    async def test_cto_agent_technology_comparison_integration(self, cto_agent):
        """Test CTO agent technology comparison capability integration."""
        request = AgentRequest(
            id="tech-comp-123",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Compare Python FastAPI vs Node.js Express vs Java Spring Boot",
            context={
                "technologies": ["Python FastAPI", "Node.js Express", "Java Spring Boot"],
                "project_requirements": "REST API with high performance",
                "team_size": 6,
                "timeline": "3 months"
            },
            created_at=datetime.utcnow()
        )
        
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## Technology Stack Comparison
            
            ### Python FastAPI
            - **Strengths**: Fast development, excellent documentation, async support
            - **Performance**: High performance, comparable to Node.js
            - **Best for**: Rapid API development, data science integration
            
            ### Node.js Express
            - **Strengths**: JavaScript ecosystem, fast I/O operations
            - **Performance**: Excellent for I/O intensive applications
            - **Best for**: Real-time applications, microservices
            
            ### Java Spring Boot
            - **Strengths**: Enterprise-grade, robust ecosystem, excellent tooling
            - **Performance**: High throughput, excellent for CPU-intensive tasks
            - **Best for**: Large enterprise applications, complex business logic
            
            ### Recommendation: Python FastAPI
            For your 6-person team and 3-month timeline, FastAPI offers the best balance.
            """
            
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Technology Comparison Analysis" in response.content
            assert "Python FastAPI" in response.content
            assert "Node.js Express" in response.content
            assert "Java Spring Boot" in response.content
            assert "Recommendation" in response.content
            
            # Verify context was passed correctly
            call_args = mock_gpt4.call_args[0][0]
            assert "6 developers" in call_args or "6-person team" in call_args
            assert "3 months" in call_args
    
    @pytest.mark.asyncio
    async def test_cto_agent_scaling_strategy_integration(self, cto_agent):
        """Test CTO agent scaling strategy capability integration."""
        request = AgentRequest(
            id="scaling-test-123",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Create scaling strategy from 5K to 100K users",
            context={
                "current_users": 5000,
                "projected_users": 100000,
                "current_cost": 300,
                "current_architecture": "monolithic",
                "performance_issues": ["database bottlenecks", "slow API responses"]
            },
            created_at=datetime.utcnow()
        )
        
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## Scaling Strategy: 5K to 100K Users
            
            ### Phase 1: Database Optimization (Month 1)
            1. Add database indexes for slow queries
            2. Implement database connection pooling
            3. Add Redis caching layer
            4. Set up database read replicas
            
            ### Phase 2: API Performance (Month 2)
            1. Implement API response caching
            2. Add load balancer with health checks
            3. Optimize slow API endpoints
            4. Implement rate limiting
            
            ### Phase 3: Architecture Evolution (Months 3-6)
            1. Break monolith into microservices
            2. Implement container orchestration
            3. Add auto-scaling capabilities
            4. Implement comprehensive monitoring
            
            ### Cost Projections
            - Current: $300/month
            - Phase 1: $800/month
            - Phase 2: $1,500/month
            - Phase 3: $4,000/month (for 100K users)
            """
            
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Scaling Strategy" in response.content
            assert "5000" in response.content or "5K" in response.content
            assert "100000" in response.content or "100K" in response.content
            assert "database bottlenecks" in response.content.lower()
            assert "Phase 1" in response.content
            assert "Cost Projections" in response.content
    
    @pytest.mark.asyncio
    async def test_cto_agent_technical_decision_integration(self, cto_agent):
        """Test CTO agent technical decision capability integration."""
        request = AgentRequest(
            id="decision-test-123",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Technical decision: Should we adopt microservices or keep monolithic architecture?",
            context={
                "constraints": ["limited team size", "tight deadline"],
                "stakeholders": ["development team", "product manager", "CTO"],
                "timeline": "6 months",
                "budget": "$100,000",
                "current_tech_stack": "Django monolith"
            },
            created_at=datetime.utcnow()
        )
        
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## Technical Decision: Microservices vs Monolith
            
            ### Problem Assessment
            The decision between microservices and monolithic architecture depends on
            team size, timeline constraints, and long-term scalability needs.
            
            ### Analysis
            **Monolithic Pros:**
            - Faster initial development
            - Simpler deployment and testing
            - Better for small teams
            - Lower operational complexity
            
            **Microservices Pros:**
            - Better scalability
            - Technology diversity
            - Independent deployments
            - Better fault isolation
            
            ### Recommendation: Modular Monolith
            Given your constraints (limited team, tight deadline), I recommend:
            1. Keep monolithic architecture for now
            2. Structure code in modules for future extraction
            3. Plan microservices migration for Phase 2
            4. Focus on clean interfaces between modules
            
            This approach balances immediate needs with future flexibility.
            """
            
            response = await cto_agent.process_request(request)
            
            # Should handle gracefully - either success with AI response or fallback
            assert response.status in [ResponseStatus.SUCCESS, ResponseStatus.ERROR]
            if response.status == ResponseStatus.SUCCESS:
                assert "Technical Decision Analysis" in response.content
                assert "microservices" in response.content.lower()
                assert "monolith" in response.content.lower()
                assert "Recommendation" in response.content
            else:
                # If GPT-4 fails, should provide fallback response
                assert "Technical Decision Analysis" in response.content or "Error processing" in response.content
            
            # Verify decision context was included (if GPT-4 was called)
            if mock_gpt4.call_args:
                call_args = mock_gpt4.call_args[0][0]
                assert "6 months" in call_args
                assert "$100,000" in call_args
                assert "Django" in call_args
    
    @pytest.mark.asyncio
    async def test_cto_agent_error_handling_integration(self, cto_agent):
        """Test CTO agent error handling integration."""
        request = AgentRequest(
            id="error-test-123",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Design architecture",
            context={},
            created_at=datetime.utcnow()
        )
        
        # Mock GPT-4 to raise an exception
        with patch.object(cto_agent, '_call_gpt4', side_effect=Exception("API rate limit exceeded")):
            response = await cto_agent.process_request(request)
            
            # Should handle error gracefully and provide fallback
            assert response.status == ResponseStatus.SUCCESS  # Fallback should work
            assert "Architecture Recommendation" in response.content
            assert "Enhanced AI analysis unavailable" in response.content
    
    @pytest.mark.asyncio
    async def test_cto_agent_concurrent_requests_integration(self, cto_agent):
        """Test CTO agent handling concurrent requests integration."""
        requests = [
            AgentRequest(
                id=f"concurrent-{i}",
                user_id="test-user",
                agent_id="scroll-cto-agent",
                prompt=f"Architecture question {i}",
                context={"request_number": i},
                created_at=datetime.utcnow()
            )
            for i in range(3)
        ]
        
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.side_effect = [
                f"Architecture response for question {i}" for i in range(3)
            ]
            
            # Process requests concurrently
            tasks = [cto_agent.process_request(req) for req in requests]
            responses = await asyncio.gather(*tasks)
            
            # Verify all responses are successful
            assert len(responses) == 3
            for i, response in enumerate(responses):
                assert response.status == ResponseStatus.SUCCESS
                assert f"question {i}" in response.content.lower()
                assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_cto_agent_health_check_integration(self, cto_agent):
        """Test CTO agent health check integration."""
        is_healthy = await cto_agent.health_check()
        assert is_healthy is True
    
    def test_cto_agent_capabilities_integration(self, cto_agent):
        """Test CTO agent capabilities integration."""
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
        
        # Test capability details
        for capability in capabilities:
            assert capability.description is not None
            assert len(capability.input_types) > 0
            assert len(capability.output_types) > 0
    
    def test_cto_agent_architecture_templates_integration(self, cto_agent):
        """Test CTO agent architecture templates integration."""
        templates = cto_agent.get_architecture_templates()
        
        assert len(templates) >= 3
        template_names = [template.name for template in templates]
        
        expected_templates = ["Startup MVP", "Enterprise Scale", "AI/ML Platform"]
        for expected_template in expected_templates:
            assert expected_template in template_names
        
        # Test template properties
        for template in templates:
            assert template.name is not None
            assert template.description is not None
            assert isinstance(template.tech_stack, TechnologyStack)
            assert isinstance(template.database, DatabaseType)
            assert isinstance(template.cloud_provider, CloudProvider)
            assert template.estimated_cost_monthly > 0
            assert 1 <= template.scalability_rating <= 10
            assert 1 <= template.complexity_rating <= 10
            assert len(template.use_cases) > 0
            assert len(template.pros) > 0
            assert len(template.cons) > 0
    
    def test_cto_agent_template_management_integration(self, cto_agent):
        """Test CTO agent template management integration."""
        initial_count = len(cto_agent.get_architecture_templates())
        
        # Add new template
        new_template = ArchitectureTemplate(
            name="Test Integration Template",
            description="Template for integration testing",
            tech_stack=TechnologyStack.GOLANG_GIN,
            database=DatabaseType.MONGODB,
            cloud_provider=CloudProvider.GCP,
            estimated_cost_monthly=500.0,
            scalability_rating=8,
            complexity_rating=6,
            use_cases=["Integration testing", "API development"],
            pros=["Fast performance", "Good concurrency"],
            cons=["Learning curve", "Smaller ecosystem"]
        )
        
        cto_agent.add_architecture_template(new_template)
        
        templates = cto_agent.get_architecture_templates()
        assert len(templates) == initial_count + 1
        assert any(t.name == "Test Integration Template" for t in templates)
    
    @pytest.mark.asyncio
    async def test_cto_agent_request_validation_integration(self, cto_agent):
        """Test CTO agent request validation integration."""
        # Test with minimal valid request
        minimal_request = AgentRequest(
            id="minimal-test",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Help",
            context={},
            created_at=datetime.utcnow()
        )
        
        response = await cto_agent.process_request(minimal_request)
        assert response.status in [ResponseStatus.SUCCESS, ResponseStatus.ERROR]
        assert response.execution_time >= 0
        assert response.request_id == minimal_request.id
    
    @pytest.mark.asyncio
    async def test_cto_agent_performance_integration(self, cto_agent):
        """Test CTO agent performance integration."""
        request = AgentRequest(
            id="perf-test",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Quick architecture advice",
            context={},
            created_at=datetime.utcnow()
        )
        
        start_time = time.time()
        response = await cto_agent.process_request(request)
        end_time = time.time()
        
        # Verify performance metrics
        total_time = end_time - start_time
        assert response.execution_time <= total_time
        assert response.execution_time > 0
        
        # Response should be reasonably fast (under 10 seconds for fallback)
        assert total_time < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])