"""
Integration tests for ScrollCTOAgent with API endpoints and routing.
Tests the complete flow from API request to agent execution and response.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime

from scrollintel.api.gateway import create_app
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


class TestScrollCTOIntegration:
    """Integration tests for ScrollCTO Agent with API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test application."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def cto_agent(self):
        """Create ScrollCTOAgent instance."""
        return ScrollCTOAgent()
    
    @pytest.fixture
    def mock_auth_context(self):
        """Create mock authentication context."""
        return SecurityContext(
            user_id="test-user-123",
            session_id="test-session-456",
            role=UserRole.ANALYST,
            permissions=["agent:execute", "agent:read"],
            ip_address="127.0.0.1"
        )
    
    @pytest.fixture
    def mock_agent_registry(self, cto_agent):
        """Create mock agent registry with CTO agent."""
        registry = Mock(spec=AgentRegistry)
        registry.get_agent.return_value = cto_agent
        registry.get_all_agents.return_value = [cto_agent]
        registry.get_active_agents.return_value = [cto_agent]
        registry.get_capabilities.return_value = cto_agent.get_capabilities()
        registry.get_registry_status.return_value = {
            "total_agents": 1,
            "agent_status": {"active": 1, "inactive": 0, "busy": 0, "error": 0},
            "agent_types": {"CTO": 1},
            "capabilities": [cap.name for cap in cto_agent.get_capabilities()]
        }
        return registry
    
    @pytest.mark.asyncio
    async def test_cto_agent_architecture_request_processing(self, cto_agent):
        """Test CTO agent processes architecture requests correctly."""
        request = AgentRequest(
            id="test-arch-request",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Design a scalable e-commerce platform architecture",
            context={
                "business_type": "e-commerce",
                "expected_users": 50000,
                "budget": 1000,
                "requirements": "high availability, payment processing, inventory management"
            },
            created_at=datetime.utcnow()
        )
        
        # Mock GPT-4 response
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## Detailed Architecture Analysis
            
            For an e-commerce platform with 50,000 users, I recommend a microservices architecture with:
            
            ### Core Services
            - User Service (authentication, profiles)
            - Product Service (catalog, inventory)
            - Order Service (cart, checkout, orders)
            - Payment Service (payment processing)
            - Notification Service (emails, SMS)
            
            ### Infrastructure Recommendations
            - Use containerization with Docker and Kubernetes
            - Implement API Gateway for service routing
            - Use Redis for session management and caching
            - PostgreSQL for transactional data
            - Elasticsearch for product search
            
            ### Scaling Considerations
            - Horizontal scaling for stateless services
            - Database sharding for high-volume tables
            - CDN for static assets
            - Load balancing with health checks
            """
            
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Architecture Recommendation" in response.content
            assert "AI-Enhanced Analysis" in response.content
            assert "microservices architecture" in response.content.lower()
            assert "e-commerce" in response.content.lower()
            assert response.execution_time > 0
            
            # Verify GPT-4 was called with appropriate prompt
            mock_gpt4.assert_called_once()
            call_args = mock_gpt4.call_args[0][0]
            assert "e-commerce platform" in call_args.lower()
            assert "50000" in call_args
            assert "$1000" in call_args
    
    @pytest.mark.asyncio
    async def test_cto_agent_technology_comparison_processing(self, cto_agent):
        """Test CTO agent processes technology comparison requests."""
        request = AgentRequest(
            id="test-tech-comparison",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Compare React vs Vue.js vs Angular for our frontend",
            context={
                "technologies": ["React", "Vue.js", "Angular"],
                "project_requirements": "complex dashboard with real-time updates",
                "team_size": 8,
                "timeline": "4 months"
            },
            created_at=datetime.utcnow()
        )
        
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## Frontend Framework Comparison
            
            ### React
            - **Strengths**: Large ecosystem, flexible, excellent for complex UIs
            - **Weaknesses**: Steeper learning curve, more boilerplate
            - **Best for**: Complex dashboards with custom components
            
            ### Vue.js
            - **Strengths**: Gentle learning curve, great documentation
            - **Weaknesses**: Smaller ecosystem, less enterprise adoption
            - **Best for**: Rapid prototyping, smaller teams
            
            ### Angular
            - **Strengths**: Full framework, TypeScript by default, enterprise-ready
            - **Weaknesses**: Heavy, complex, slower development initially
            - **Best for**: Large enterprise applications
            
            ### Recommendation
            For your complex dashboard with real-time updates and 8-person team, I recommend **React** because:
            1. Excellent real-time capabilities with libraries like Socket.io
            2. Rich ecosystem for dashboard components
            3. Team size can handle the complexity
            4. Strong community support for troubleshooting
            """
            
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Technology Comparison Analysis" in response.content
            assert "React" in response.content
            assert "Vue.js" in response.content
            assert "Angular" in response.content
            assert "dashboard" in response.content.lower()
            
            # Verify GPT-4 was called with comparison context
            call_args = mock_gpt4.call_args[0][0]
            assert "React" in call_args and "Vue.js" in call_args and "Angular" in call_args
            assert "8 developers" in call_args
            assert "4 months" in call_args
    
    @pytest.mark.asyncio
    async def test_cto_agent_scaling_strategy_processing(self, cto_agent):
        """Test CTO agent processes scaling strategy requests."""
        request = AgentRequest(
            id="test-scaling-strategy",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="How do I scale my application from 10K to 1M users?",
            context={
                "current_users": 10000,
                "projected_users": 1000000,
                "current_cost": 500,
                "current_architecture": "monolithic",
                "performance_issues": ["slow database queries", "memory leaks"]
            },
            created_at=datetime.utcnow()
        )
        
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## Comprehensive Scaling Strategy
            
            ### Phase 1: Immediate Optimizations (1-2 months)
            1. Database query optimization and indexing
            2. Memory leak fixes and performance profiling
            3. Implement Redis caching layer
            4. Add database read replicas
            
            ### Phase 2: Architecture Transition (3-6 months)
            1. Break monolith into microservices
            2. Implement API Gateway
            3. Containerize services with Docker
            4. Set up Kubernetes orchestration
            
            ### Phase 3: Advanced Scaling (6-12 months)
            1. Implement auto-scaling groups
            2. Database sharding strategy
            3. CDN implementation
            4. Advanced monitoring and alerting
            
            ### Cost Projections
            - Current: $500/month
            - Phase 1: $1,500/month
            - Phase 2: $5,000/month
            - Phase 3: $15,000/month (for 1M users)
            
            ### Risk Mitigation
            - Gradual migration with feature flags
            - Comprehensive testing at each phase
            - Rollback procedures for each change
            """
            
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Scaling Strategy" in response.content
            assert "1000000" in response.content or "1M" in response.content
            assert "monolithic" in response.content.lower()
            assert "database queries" in response.content.lower()
            
            # Verify scaling context was passed to GPT-4
            call_args = mock_gpt4.call_args[0][0]
            assert "10000" in call_args and "1000000" in call_args
            assert "monolithic" in call_args.lower()
            assert "slow database queries" in call_args.lower()
    
    @pytest.mark.asyncio
    async def test_cto_agent_technical_decision_processing(self, cto_agent):
        """Test CTO agent processes technical decision requests."""
        request = AgentRequest(
            id="test-tech-decision",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Should we migrate from REST API to GraphQL?",
            context={
                "constraints": ["limited development time", "existing mobile apps"],
                "stakeholders": ["mobile team", "backend team", "product manager"],
                "timeline": "3 months",
                "budget": "$50,000",
                "current_tech_stack": "Node.js REST API with Express"
            },
            created_at=datetime.utcnow()
        )
        
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.return_value = """
            ## Technical Decision Analysis: REST to GraphQL Migration
            
            ### Problem Assessment
            The decision to migrate from REST to GraphQL involves significant architectural changes
            that must be evaluated against current constraints and stakeholder needs.
            
            ### Technical Feasibility
            - **Pros**: Better data fetching, single endpoint, strong typing
            - **Cons**: Learning curve, existing mobile app compatibility, caching complexity
            
            ### Business Impact
            - Improved mobile app performance through reduced over-fetching
            - Faster frontend development with better tooling
            - Potential breaking changes for existing integrations
            
            ### Recommendation: Hybrid Approach
            1. Implement GraphQL alongside existing REST API
            2. Migrate high-traffic mobile endpoints first
            3. Maintain REST for legacy integrations
            4. Gradual migration over 6-month period
            
            ### Implementation Plan
            - Month 1: GraphQL setup and core schema design
            - Month 2: Migrate user and product endpoints
            - Month 3: Mobile app integration and testing
            
            ### Success Metrics
            - 30% reduction in mobile data usage
            - 50% faster mobile app load times
            - Zero downtime during migration
            """
            
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Technical Decision Analysis" in response.content
            assert "GraphQL" in response.content
            assert "REST" in response.content
            assert "mobile" in response.content.lower()
            
            # Verify decision context was included
            call_args = mock_gpt4.call_args[0][0]
            assert "GraphQL" in call_args
            assert "3 months" in call_args
            assert "$50,000" in call_args
            assert "mobile team" in call_args
    
    @pytest.mark.asyncio
    async def test_cto_agent_gpt4_fallback_handling(self, cto_agent):
        """Test CTO agent handles GPT-4 failures gracefully."""
        request = AgentRequest(
            id="test-fallback",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="Design a microservices architecture",
            context={"business_type": "startup", "expected_users": 5000, "budget": 300},
            created_at=datetime.utcnow()
        )
        
        # Mock GPT-4 to raise an exception
        with patch.object(cto_agent, '_call_gpt4', side_effect=Exception("API rate limit exceeded")):
            response = await cto_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Architecture Recommendation" in response.content
            assert "Enhanced AI analysis unavailable" in response.content
            # Should still provide template-based recommendation
            assert "Startup MVP" in response.content
    
    def test_agent_registration_with_registry(self, mock_agent_registry, cto_agent):
        """Test CTO agent can be registered and retrieved from registry."""
        # Test agent retrieval
        retrieved_agent = mock_agent_registry.get_agent("scroll-cto-agent")
        assert retrieved_agent == cto_agent
        assert retrieved_agent.agent_id == "scroll-cto-agent"
        assert retrieved_agent.agent_type == AgentType.CTO
        
        # Test capabilities retrieval
        capabilities = mock_agent_registry.get_capabilities()
        capability_names = [cap.name for cap in capabilities]
        assert "architecture_design" in capability_names
        assert "technology_comparison" in capability_names
        assert "scaling_strategy" in capability_names
        assert "technical_decision" in capability_names
    
    @pytest.mark.asyncio
    async def test_concurrent_cto_requests(self, cto_agent):
        """Test CTO agent handles concurrent requests correctly."""
        requests = [
            AgentRequest(
                id=f"concurrent-{i}",
                user_id="test-user",
                agent_id="scroll-cto-agent",
                prompt=f"Architecture advice for project {i}",
                context={"business_type": "startup", "expected_users": 1000 * i},
                created_at=datetime.utcnow()
            )
            for i in range(1, 4)
        ]
        
        # Mock GPT-4 to return different responses
        with patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            mock_gpt4.side_effect = [
                f"Architecture analysis for project {i}" for i in range(1, 4)
            ]
            
            # Process requests concurrently
            tasks = [cto_agent.process_request(req) for req in requests]
            responses = await asyncio.gather(*tasks)
            
            # Verify all responses are successful
            assert len(responses) == 3
            for i, response in enumerate(responses):
                assert response.status == ResponseStatus.SUCCESS
                assert f"project {i+1}" in response.content.lower()
                assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_cto_agent_health_check(self, cto_agent):
        """Test CTO agent health check functionality."""
        is_healthy = await cto_agent.health_check()
        assert is_healthy is True
    
    def test_architecture_template_management(self, cto_agent):
        """Test architecture template management functionality."""
        initial_count = len(cto_agent.get_architecture_templates())
        
        # Add new template
        new_template = ArchitectureTemplate(
            name="Serverless Architecture",
            description="Event-driven serverless architecture",
            tech_stack=TechnologyStack.NODE_EXPRESS,
            database=DatabaseType.MONGODB,
            cloud_provider=CloudProvider.AWS,
            estimated_cost_monthly=150.0,
            scalability_rating=9,
            complexity_rating=4,
            use_cases=["Event processing", "API backends"],
            pros=["Auto-scaling", "Pay per use"],
            cons=["Cold starts", "Vendor lock-in"]
        )
        
        cto_agent.add_architecture_template(new_template)
        
        templates = cto_agent.get_architecture_templates()
        assert len(templates) == initial_count + 1
        assert any(t.name == "Serverless Architecture" for t in templates)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_request_processing(self, cto_agent):
        """Test error handling during request processing."""
        # Create request that might cause processing errors
        bad_request = AgentRequest(
            id="error-test",
            user_id="test-user",
            agent_id="scroll-cto-agent",
            prompt="",  # Empty prompt
            context=None,  # No context
            created_at=datetime.utcnow()
        )
        
        response = await cto_agent.process_request(bad_request)
        
        # Should handle gracefully and return error response
        assert response.status == ResponseStatus.ERROR
        assert "Error processing CTO request" in response.content
        assert response.execution_time > 0
        assert response.error_message is not None


class TestScrollCTOAPIIntegration:
    """Integration tests for ScrollCTO Agent through API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create mock authentication headers."""
        # In a real test, this would be a valid JWT token
        return {"Authorization": "Bearer mock-jwt-token"}
    
    def test_agent_list_includes_cto_agent(self, client):
        """Test that CTO agent appears in agent list endpoint."""
        # This test would require proper authentication setup
        # For now, we test that the endpoint exists and requires auth
        response = client.get("/agents/")
        assert response.status_code == 401  # Requires authentication
    
    def test_agent_capabilities_endpoint(self, client):
        """Test agent capabilities endpoint."""
        response = client.get("/agents/capabilities")
        assert response.status_code == 401  # Requires authentication
    
    def test_agent_execution_endpoint_structure(self, client):
        """Test agent execution endpoint structure."""
        response = client.post("/agents/execute", json={
            "prompt": "Design a web application architecture",
            "agent_id": "scroll-cto-agent",
            "context": {"business_type": "startup"},
            "priority": 1
        })
        assert response.status_code == 401  # Requires authentication
    
    def test_workflow_execution_endpoint(self, client):
        """Test workflow execution endpoint."""
        response = client.post("/agents/workflow", json={
            "workflow_name": "architecture_design",
            "steps": [
                {
                    "agent_id": "scroll-cto-agent",
                    "prompt": "Analyze requirements",
                    "context": {}
                }
            ],
            "context": {},
            "continue_on_error": False
        })
        assert response.status_code == 401  # Requires authentication
    
    def test_agent_health_check_endpoint(self, client):
        """Test agent health check endpoint."""
        response = client.post("/agents/scroll-cto-agent/health")
        assert response.status_code == 401  # Requires authentication
    
    def test_registry_status_endpoint(self, client):
        """Test registry status endpoint."""
        response = client.get("/agents/registry/status")
        assert response.status_code == 401  # Requires authentication


if __name__ == "__main__":
    pytest.main([__file__, "-v"])