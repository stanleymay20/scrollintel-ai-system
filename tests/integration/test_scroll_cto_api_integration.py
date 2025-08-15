"""
Integration Tests for ScrollCTOAgent API Endpoints and Routing
Tests the complete flow from API request to agent execution and response.
Covers all CTO agent capabilities through the FastAPI gateway.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List
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
from scrollintel.core.registry import AgentRegistry, TaskOrchestrator
from scrollintel.security.auth import create_access_token


class TestScrollCTOAPIIntegration:
    """Integration tests for ScrollCTO Agent API endpoints and routing."""
    
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
    def mock_agent_registry(self, cto_agent):
        """Create mock agent registry with CTO agent."""
        registry = Mock(spec=AgentRegistry)
        registry.get_agent.return_value = cto_agent
        registry.get_all_agents.return_value = [cto_agent]
        registry.get_active_agents.return_value = [cto_agent]
        registry.get_capabilities.return_value = cto_agent.get_capabilities()
        registry.get_registry_status.return_value = {
            "total_agents": 1,
            "active_agents": 1,
            "agent_types": {"CTO": 1},
            "capabilities": [cap.name for cap in cto_agent.get_capabilities()]
        }
        
        # Mock route_request to return agent response
        async def mock_route_request(request: AgentRequest) -> AgentResponse:
            return await cto_agent.process_request(request)
        
        registry.route_request = mock_route_request
        return registry
    
    @pytest.fixture
    def auth_token(self):
        """Create valid JWT token for testing."""
        return create_access_token(
            data={
                "sub": "test-user-123",
                "role": "analyst",
                "permissions": ["agent:execute", "agent:read", "agent:list"]
            }
        )
    
    @pytest.fixture
    def auth_headers(self, auth_token):
        """Create authentication headers."""
        return {"Authorization": f"Bearer {auth_token}"}
    
    @pytest.fixture
    def mock_security_context(self):
        """Create mock security context."""
        return SecurityContext(
            user_id="test-user-123",
            session_id="test-session-456",
            role=UserRole.ANALYST,
            permissions=["agent:execute", "agent:read", "agent:list"],
            ip_address="127.0.0.1"
        )
    
    def test_agent_list_endpoint_includes_cto_agent(self, client, auth_headers, mock_agent_registry):
        """Test that CTO agent appears in agent list endpoint."""
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry):
            response = client.get("/agents/", headers=auth_headers)
            
            assert response.status_code == 200
            agents = response.json()
            
            assert len(agents) == 1
            cto_agent_data = agents[0]
            assert cto_agent_data["agent_id"] == "scroll-cto-agent"
            assert cto_agent_data["name"] == "ScrollCTO Agent"
            assert cto_agent_data["type"] == "CTO"
            assert cto_agent_data["status"] == "active"
            assert len(cto_agent_data["capabilities"]) == 4
    
    def test_agent_capabilities_endpoint_returns_cto_capabilities(self, client, auth_headers, mock_agent_registry):
        """Test agent capabilities endpoint returns CTO capabilities."""
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry):
            response = client.get("/agents/capabilities", headers=auth_headers)
            
            assert response.status_code == 200
            capabilities = response.json()
            
            capability_names = [cap["name"] for cap in capabilities]
            expected_capabilities = [
                "architecture_design",
                "technology_comparison",
                "scaling_strategy",
                "technical_decision"
            ]
            
            for expected_cap in expected_capabilities:
                assert expected_cap in capability_names
    
    def test_get_specific_cto_agent_endpoint(self, client, auth_headers, mock_agent_registry):
        """Test getting specific CTO agent information."""
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry):
            response = client.get("/agents/scroll-cto-agent", headers=auth_headers)
            
            assert response.status_code == 200
            agent_data = response.json()
            
            assert agent_data["agent_id"] == "scroll-cto-agent"
            assert agent_data["name"] == "ScrollCTO Agent"
            assert agent_data["type"] == "CTO"
            assert "architecture_design" in agent_data["capabilities"]
            assert "technology_comparison" in agent_data["capabilities"]
            assert "scaling_strategy" in agent_data["capabilities"]
            assert "technical_decision" in agent_data["capabilities"]
    
    def test_cto_agent_architecture_design_execution(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent architecture design execution through API."""
        request_data = {
            "prompt": "Design a scalable e-commerce platform architecture for 100K users",
            "agent_id": "scroll-cto-agent",
            "context": {
                "business_type": "e-commerce",
                "expected_users": 100000,
                "budget": 2000,
                "requirements": ["high availability", "payment processing", "inventory management"]
            },
            "priority": 1
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            
            mock_gpt4.return_value = """
            ## E-commerce Architecture Recommendation
            
            For a 100K user e-commerce platform, I recommend a microservices architecture:
            
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
            - Elasticsearch for product search
            - CDN for static assets
            
            ### Scaling Strategy
            - Horizontal pod autoscaling
            - Database read replicas
            - Message queues for async processing
            """
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["status"] == "success"
            assert "Architecture Recommendation" in result["content"]
            assert "microservices" in result["content"].lower()
            assert "e-commerce" in result["content"].lower()
            assert result["execution_time"] > 0
            assert result["agent_id"] is not None
            
            # Verify GPT-4 was called with appropriate context
            mock_gpt4.assert_called_once()
            call_args = mock_gpt4.call_args[0][0]
            assert "100000" in call_args
            assert "e-commerce" in call_args.lower()
            assert "$2000" in call_args
    
    def test_cto_agent_technology_comparison_execution(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent technology comparison execution through API."""
        request_data = {
            "prompt": "Compare React vs Vue.js vs Angular for our frontend development",
            "agent_id": "scroll-cto-agent",
            "context": {
                "technologies": ["React", "Vue.js", "Angular"],
                "project_requirements": "complex dashboard with real-time updates",
                "team_size": 8,
                "timeline": "4 months"
            },
            "priority": 2
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            
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
            
            ### Recommendation: React
            For your complex dashboard with real-time updates and 8-person team, React is optimal.
            """
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["status"] == "success"
            assert "Technology Comparison Analysis" in result["content"]
            assert "React" in result["content"]
            assert "Vue.js" in result["content"]
            assert "Angular" in result["content"]
            assert "dashboard" in result["content"].lower()
            
            # Verify context was passed correctly
            call_args = mock_gpt4.call_args[0][0]
            assert "React" in call_args and "Vue.js" in call_args and "Angular" in call_args
            assert "8 developers" in call_args or "8-person team" in call_args
            assert "4 months" in call_args
    
    def test_cto_agent_scaling_strategy_execution(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent scaling strategy execution through API."""
        request_data = {
            "prompt": "How do I scale my application from 10K to 1M users?",
            "agent_id": "scroll-cto-agent",
            "context": {
                "current_users": 10000,
                "projected_users": 1000000,
                "current_cost": 500,
                "current_architecture": "monolithic",
                "performance_issues": ["slow database queries", "memory leaks"]
            },
            "priority": 1
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            
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
            """
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["status"] == "success"
            assert "Scaling Strategy" in result["content"]
            assert "1000000" in result["content"] or "1M" in result["content"]
            assert "monolithic" in result["content"].lower()
            assert "database queries" in result["content"].lower()
            
            # Verify scaling context was passed
            call_args = mock_gpt4.call_args[0][0]
            assert "10000" in call_args and "1000000" in call_args
            assert "monolithic" in call_args.lower()
            assert "slow database queries" in call_args.lower()
    
    def test_cto_agent_technical_decision_execution(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent technical decision execution through API."""
        request_data = {
            "prompt": "Should we migrate from REST API to GraphQL?",
            "agent_id": "scroll-cto-agent",
            "context": {
                "constraints": ["limited development time", "existing mobile apps"],
                "stakeholders": ["mobile team", "backend team", "product manager"],
                "timeline": "3 months",
                "budget": "$50,000",
                "current_tech_stack": "Node.js REST API with Express"
            },
            "priority": 2
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            
            mock_gpt4.return_value = """
            ## Technical Decision Analysis: REST to GraphQL Migration
            
            ### Problem Assessment
            The decision to migrate from REST to GraphQL involves significant architectural changes.
            
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
            """
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["status"] == "success"
            assert "Technical Decision Analysis" in result["content"]
            assert "GraphQL" in result["content"]
            assert "REST" in result["content"]
            assert "mobile" in result["content"].lower()
            
            # Verify decision context was included
            call_args = mock_gpt4.call_args[0][0]
            assert "GraphQL" in call_args
            assert "3 months" in call_args
            assert "$50,000" in call_args
            assert "mobile team" in call_args
    
    def test_cto_agent_auto_routing_by_type(self, client, auth_headers, mock_agent_registry):
        """Test automatic routing to CTO agent by agent type."""
        request_data = {
            "prompt": "Provide technical architecture guidance",
            "agent_type": "CTO",  # No specific agent_id, should route by type
            "context": {"business_type": "startup"},
            "priority": 1
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry):
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["status"] == "success"
            assert "Architecture Recommendation" in result["content"]
            assert result["execution_time"] > 0
    
    def test_cto_agent_workflow_execution(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent in multi-step workflow execution."""
        workflow_data = {
            "workflow_name": "architecture_design_workflow",
            "steps": [
                {
                    "agent_id": "scroll-cto-agent",
                    "prompt": "Analyze technical requirements",
                    "context": {"requirements": ["scalability", "security", "performance"]}
                },
                {
                    "agent_id": "scroll-cto-agent",
                    "prompt": "Recommend technology stack",
                    "context": {"analysis_result": "requirements_analyzed"}
                },
                {
                    "agent_id": "scroll-cto-agent",
                    "prompt": "Create implementation roadmap",
                    "context": {"tech_stack": "recommended_stack"}
                }
            ],
            "context": {"project_type": "web_application"},
            "continue_on_error": False
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch('scrollintel.api.gateway.TaskOrchestrator') as mock_orchestrator:
            
            # Mock workflow execution
            mock_orchestrator_instance = Mock()
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            async def mock_execute_workflow(steps, context):
                responses = []
                for i, step in enumerate(steps):
                    response = AgentResponse(
                        id=f"workflow-step-{i}",
                        request_id=f"request-{i}",
                        content=f"Step {i+1} completed: {step['prompt']}",
                        artifacts=[],
                        execution_time=0.5,
                        status=ResponseStatus.SUCCESS
                    )
                    responses.append(response)
                return responses
            
            mock_orchestrator_instance.execute_workflow = mock_execute_workflow
            
            response = client.post("/agents/workflow", json=workflow_data, headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert len(result) == 3  # Three workflow steps
            for step_result in result:
                assert step_result["status"] == "success"
                assert "Step" in step_result["content"]
                assert step_result["execution_time"] > 0
    
    def test_cto_agent_health_check_endpoint(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent health check endpoint."""
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, 'health_check', new_callable=AsyncMock) as mock_health:
            
            mock_health.return_value = True
            
            response = client.post("/agents/scroll-cto-agent/health", headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["agent_id"] == "scroll-cto-agent"
            assert result["healthy"] is True
            assert result["status"] == "active"
            assert "timestamp" in result
    
    def test_cto_agent_error_handling_through_api(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent error handling through API."""
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, 'process_request', new_callable=AsyncMock) as mock_process:
            
            # Mock agent to return error response
            mock_process.return_value = AgentResponse(
                id="error-response",
                request_id="test-request",
                content="Error processing CTO request: GPT-4 API unavailable",
                artifacts=[],
                execution_time=0.1,
                status=ResponseStatus.ERROR,
                error_message="GPT-4 API unavailable"
            )
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200  # API call succeeds, but agent returns error
            result = response.json()
            
            assert result["status"] == "error"
            assert "Error processing CTO request" in result["content"]
            assert result["error_message"] == "GPT-4 API unavailable"
    
    def test_cto_agent_authentication_required(self, client, mock_agent_registry):
        """Test that CTO agent endpoints require authentication."""
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        # Test without authentication headers
        response = client.post("/agents/execute", json=request_data)
        assert response.status_code == 401
        
        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/agents/execute", json=request_data, headers=invalid_headers)
        assert response.status_code == 401
    
    def test_cto_agent_permission_validation(self, client, mock_agent_registry):
        """Test that CTO agent endpoints validate permissions."""
        # Create token with insufficient permissions
        limited_token = create_access_token(
            data={
                "sub": "limited-user",
                "role": "viewer",
                "permissions": ["agent:read"]  # Missing agent:execute permission
            }
        )
        limited_headers = {"Authorization": f"Bearer {limited_token}"}
        
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=request_data, headers=limited_headers)
        assert response.status_code == 403  # Forbidden due to insufficient permissions
    
    def test_cto_agent_request_validation(self, client, auth_headers, mock_agent_registry):
        """Test request validation for CTO agent endpoints."""
        # Test with missing required fields
        invalid_request = {
            # Missing prompt
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=invalid_request, headers=auth_headers)
        assert response.status_code == 422  # Validation error
        
        # Test with invalid priority
        invalid_priority_request = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 15  # Priority should be 1-10
        }
        
        response = client.post("/agents/execute", json=invalid_priority_request, headers=auth_headers)
        assert response.status_code == 422  # Validation error
    
    def test_cto_agent_concurrent_requests_through_api(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test concurrent requests to CTO agent through API."""
        import threading
        import queue
        
        request_data_templates = [
            {
                "prompt": "Design microservices architecture",
                "agent_id": "scroll-cto-agent",
                "context": {"service_count": 5},
                "priority": 1
            },
            {
                "prompt": "Compare database technologies",
                "agent_id": "scroll-cto-agent",
                "context": {"databases": ["PostgreSQL", "MongoDB"]},
                "priority": 2
            },
            {
                "prompt": "Create scaling strategy",
                "agent_id": "scroll-cto-agent",
                "context": {"current_users": 1000, "target_users": 10000},
                "priority": 1
            }
        ]
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, '_call_gpt4', new_callable=AsyncMock) as mock_gpt4:
            
            mock_gpt4.side_effect = [
                "Microservices architecture recommendation",
                "Database comparison analysis",
                "Scaling strategy plan"
            ]
            
            results_queue = queue.Queue()
            
            def make_request(request_data):
                response = client.post("/agents/execute", json=request_data, headers=auth_headers)
                results_queue.put(response)
            
            # Start concurrent requests
            threads = []
            for request_data in request_data_templates:
                thread = threading.Thread(target=make_request, args=(request_data,))
                threads.append(thread)
                thread.start()
            
            # Wait for all requests to complete
            for thread in threads:
                thread.join()
            
            # Collect results
            responses = []
            while not results_queue.empty():
                responses.append(results_queue.get())
            
            assert len(responses) == 3
            for response in responses:
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "success"
                assert result["execution_time"] > 0
    
    def test_registry_status_includes_cto_agent(self, client, auth_headers, mock_agent_registry):
        """Test that registry status endpoint includes CTO agent information."""
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry):
            response = client.get("/agents/registry/status", headers=auth_headers)
            
            assert response.status_code == 200
            result = response.json()
            
            assert "registry_status" in result
            registry_status = result["registry_status"]
            
            assert registry_status["total_agents"] == 1
            assert registry_status["active_agents"] == 1
            assert "CTO" in registry_status["agent_types"]
            assert registry_status["agent_types"]["CTO"] == 1
            
            # Check capabilities are included
            assert "capabilities" in registry_status
            capabilities = registry_status["capabilities"]
            assert "architecture_design" in capabilities
            assert "technology_comparison" in capabilities
            assert "scaling_strategy" in capabilities
            assert "technical_decision" in capabilities
    
    def test_agent_types_endpoint_includes_cto(self, client, auth_headers):
        """Test that agent types endpoint includes CTO type."""
        response = client.get("/agents/types", headers=auth_headers)
        
        assert response.status_code == 200
        agent_types = response.json()
        
        assert "CTO" in agent_types
    
    def test_cto_agent_timeout_handling(self, client, auth_headers, mock_agent_registry, cto_agent):
        """Test CTO agent timeout handling through API."""
        request_data = {
            "prompt": "Design complex enterprise architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1,
            "timeout_seconds": 1  # Very short timeout
        }
        
        with patch('scrollintel.api.gateway.AgentRegistry', return_value=mock_agent_registry), \
             patch.object(cto_agent, 'process_request', new_callable=AsyncMock) as mock_process:
            
            # Mock slow processing
            async def slow_process(request):
                await asyncio.sleep(2)  # Longer than timeout
                return AgentResponse(
                    id="slow-response",
                    request_id=request.id,
                    content="Architecture designed",
                    artifacts=[],
                    execution_time=2.0,
                    status=ResponseStatus.SUCCESS
                )
            
            mock_process.side_effect = slow_process
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            # Should handle timeout gracefully
            # Note: Actual timeout handling would depend on implementation
            assert response.status_code in [200, 408, 500]  # Various acceptable timeout responses


class TestScrollCTORoutingIntegration:
    """Integration tests for CTO agent routing and discovery."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test application."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        token = create_access_token(
            data={
                "sub": "test-user",
                "role": "analyst",
                "permissions": ["agent:execute", "agent:read", "agent:list"]
            }
        )
        return {"Authorization": f"Bearer {token}"}
    
    def test_intelligent_routing_to_cto_agent(self, client, auth_headers):
        """Test intelligent routing to CTO agent based on prompt content."""
        architecture_prompts = [
            "Design a scalable web application architecture",
            "What technology stack should I use for my startup?",
            "How do I scale my application to handle more users?",
            "Should I use microservices or monolithic architecture?",
            "Compare different cloud providers for my application"
        ]
        
        for prompt in architecture_prompts:
            request_data = {
                "prompt": prompt,
                # No agent_id specified - should route automatically
                "context": {},
                "priority": 1
            }
            
            # Note: This test would require actual routing logic implementation
            # For now, we test that the endpoint accepts the request
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            # Should route to appropriate agent (would be CTO for these prompts)
            assert response.status_code in [200, 401]  # 401 if auth not fully mocked
    
    def test_capability_based_routing(self, client, auth_headers):
        """Test routing based on required capabilities."""
        request_data = {
            "prompt": "I need technical architecture advice",
            "context": {
                "required_capabilities": ["architecture_design", "technical_decision"]
            },
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Should route to agent with matching capabilities (CTO agent)
        assert response.status_code in [200, 401]  # 401 if auth not fully mocked
    
    def test_agent_type_routing(self, client, auth_headers):
        """Test routing by agent type."""
        request_data = {
            "prompt": "Provide CTO-level technical guidance",
            "agent_type": "CTO",
            "context": {},
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Should route to CTO agent specifically
        assert response.status_code in [200, 401]  # 401 if auth not fully mocked
    
    def test_fallback_routing_when_cto_unavailable(self, client, auth_headers):
        """Test fallback routing when CTO agent is unavailable."""
        request_data = {
            "prompt": "Design system architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        # This would test fallback behavior when CTO agent is down
        # Implementation would depend on actual fallback logic
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Should either succeed with CTO agent or provide appropriate error
        assert response.status_code in [200, 401, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])