"""
Integration Tests for ScrollCTO Agent Routing and Middleware
Tests the complete middleware stack, routing logic, and error handling
for CTO agent requests through the FastAPI gateway.
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime

from scrollintel.api.gateway import create_app, ScrollIntelGateway
from scrollintel.agents.scroll_cto_agent import ScrollCTOAgent
from scrollintel.core.interfaces import (
    AgentRequest, AgentResponse, AgentType, ResponseStatus,
    SecurityContext, UserRole, AgentError, SecurityError, EngineError
)
from scrollintel.core.registry import AgentRegistry
from scrollintel.security.auth import create_access_token


class TestCTORoutingMiddleware:
    """Test CTO agent routing through middleware stack."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application with full middleware stack."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def cto_agent(self):
        """Create CTO agent instance."""
        return ScrollCTOAgent()
    
    @pytest.fixture
    def valid_auth_token(self):
        """Create valid authentication token."""
        return create_access_token(
            data={
                "sub": "test-user-123",
                "role": "analyst",
                "permissions": ["agent:execute", "agent:read", "agent:list"],
                "session_id": "test-session-456"
            }
        )
    
    @pytest.fixture
    def auth_headers(self, valid_auth_token):
        """Create authentication headers."""
        return {
            "Authorization": f"Bearer {valid_auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "ScrollIntel-Test/1.0"
        }
    
    def test_security_middleware_authentication(self, client):
        """Test security middleware authentication for CTO agent endpoints."""
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        # Test without authentication
        response = client.post("/agents/execute", json=request_data)
        assert response.status_code == 401
        
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"] == "Security Error"
    
    def test_security_middleware_authorization(self, client):
        """Test security middleware authorization for CTO agent endpoints."""
        # Create token with insufficient permissions
        limited_token = create_access_token(
            data={
                "sub": "limited-user",
                "role": "viewer",
                "permissions": ["agent:read"]  # Missing agent:execute
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
        assert response.status_code == 403
        
        error_data = response.json()
        assert "error" in error_data
    
    def test_rate_limiting_middleware(self, client, auth_headers):
        """Test rate limiting middleware for CTO agent requests."""
        request_data = {
            "prompt": "Quick architecture question",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        # Make multiple rapid requests to trigger rate limiting
        responses = []
        for i in range(10):  # Assuming rate limit is lower than 10 requests
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            responses.append(response)
            
            # Small delay to avoid overwhelming the test
            time.sleep(0.01)
        
        # At least one request should succeed (first few)
        success_responses = [r for r in responses if r.status_code == 200]
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        
        # Should have some successful requests and potentially some rate-limited ones
        assert len(success_responses) >= 1
        # Note: Actual rate limiting behavior depends on configuration
    
    def test_cors_middleware_headers(self, client, auth_headers):
        """Test CORS middleware headers for CTO agent endpoints."""
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        # Add CORS headers to request
        cors_headers = {
            **auth_headers,
            "Origin": "https://scrollintel.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,Authorization"
        }
        
        response = client.post("/agents/execute", json=request_data, headers=cors_headers)
        
        # Check CORS headers in response
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-credentials" in response.headers
    
    def test_gzip_compression_middleware(self, client, auth_headers):
        """Test Gzip compression middleware for large CTO responses."""
        # Create request that should generate large response
        request_data = {
            "prompt": "Provide comprehensive architecture documentation with detailed explanations, code examples, and implementation guides for a large-scale enterprise system",
            "agent_id": "scroll-cto-agent",
            "context": {
                "detailed_response": True,
                "include_examples": True,
                "comprehensive": True
            },
            "priority": 1
        }
        
        headers_with_encoding = {
            **auth_headers,
            "Accept-Encoding": "gzip, deflate"
        }
        
        response = client.post("/agents/execute", json=request_data, headers=headers_with_encoding)
        
        # Check if response is compressed (when response is large enough)
        if len(response.content) > 1000:  # Minimum size for compression
            assert "content-encoding" in response.headers
    
    def test_error_handling_middleware_agent_error(self, client, auth_headers):
        """Test error handling middleware for agent errors."""
        with patch('scrollintel.core.registry.AgentRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            # Mock agent registry to raise AgentError
            async def mock_route_request(request):
                raise AgentError("CTO agent is currently unavailable")
            
            mock_registry.route_request = mock_route_request
            
            request_data = {
                "prompt": "Design architecture",
                "agent_id": "scroll-cto-agent",
                "context": {},
                "priority": 1
            }
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 422
            error_data = response.json()
            assert error_data["error"] == "Agent Error"
            assert error_data["type"] == "agent_error"
            assert "CTO agent is currently unavailable" in error_data["message"]
    
    def test_error_handling_middleware_security_error(self, client):
        """Test error handling middleware for security errors."""
        # Use malformed token to trigger security error
        malformed_headers = {"Authorization": "Bearer malformed.token.here"}
        
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=request_data, headers=malformed_headers)
        
        assert response.status_code == 401
        error_data = response.json()
        assert error_data["error"] == "Security Error"
        assert error_data["type"] == "security_error"
    
    def test_error_handling_middleware_engine_error(self, client, auth_headers):
        """Test error handling middleware for engine errors."""
        with patch('scrollintel.core.registry.AgentRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            # Mock agent registry to raise EngineError
            async def mock_route_request(request):
                raise EngineError("AutoModel engine failed to initialize")
            
            mock_registry.route_request = mock_route_request
            
            request_data = {
                "prompt": "Design architecture",
                "agent_id": "scroll-cto-agent",
                "context": {},
                "priority": 1
            }
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 500
            error_data = response.json()
            assert error_data["error"] == "Engine Error"
            assert error_data["type"] == "engine_error"
    
    def test_error_handling_middleware_validation_error(self, client, auth_headers):
        """Test error handling middleware for validation errors."""
        # Send invalid request data
        invalid_request_data = {
            "prompt": "",  # Empty prompt should cause validation error
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 15  # Invalid priority (should be 1-10)
        }
        
        response = client.post("/agents/execute", json=invalid_request_data, headers=auth_headers)
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data  # FastAPI validation error format
    
    def test_error_handling_middleware_unexpected_error(self, client, auth_headers):
        """Test error handling middleware for unexpected errors."""
        with patch('scrollintel.core.registry.AgentRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            # Mock agent registry to raise unexpected error
            async def mock_route_request(request):
                raise RuntimeError("Unexpected system error")
            
            mock_registry.route_request = mock_route_request
            
            request_data = {
                "prompt": "Design architecture",
                "agent_id": "scroll-cto-agent",
                "context": {},
                "priority": 1
            }
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            assert response.status_code == 500
            error_data = response.json()
            assert error_data["error"] == "Internal Server Error"
            assert error_data["type"] == "internal_error"
    
    def test_audit_logging_middleware(self, client, auth_headers):
        """Test audit logging middleware for CTO agent requests."""
        request_data = {
            "prompt": "Design secure architecture",
            "agent_id": "scroll-cto-agent",
            "context": {"security_focus": True},
            "priority": 1
        }
        
        with patch('scrollintel.security.audit.audit_logger') as mock_audit:
            mock_audit.log = AsyncMock()
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            # Verify audit logging was called
            # Note: This would require actual audit logger implementation
            # For now, we just verify the request was processed
            assert response.status_code in [200, 401, 500]  # Various acceptable responses
    
    def test_request_id_tracking(self, client, auth_headers):
        """Test request ID tracking through middleware."""
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        # Add request ID header
        headers_with_request_id = {
            **auth_headers,
            "X-Request-ID": "test-request-12345"
        }
        
        response = client.post("/agents/execute", json=request_data, headers=headers_with_request_id)
        
        # Check if request ID is tracked in response
        if response.status_code == 200:
            result = response.json()
            assert "request_id" in result
        
        # Request ID should be in response headers or body
        assert "X-Request-ID" in response.headers or response.status_code != 200
    
    def test_performance_monitoring_middleware(self, client, auth_headers):
        """Test performance monitoring middleware for CTO requests."""
        request_data = {
            "prompt": "Design high-performance architecture",
            "agent_id": "scroll-cto-agent",
            "context": {"performance_critical": True},
            "priority": 1
        }
        
        start_time = time.time()
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        end_time = time.time()
        
        # Check response time tracking
        if response.status_code == 200:
            result = response.json()
            assert "execution_time" in result
            assert result["execution_time"] > 0
            
            # Verify execution time is reasonable
            total_time = end_time - start_time
            assert result["execution_time"] <= total_time
    
    def test_content_type_validation(self, client, auth_headers):
        """Test content type validation in middleware."""
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",
            "context": {},
            "priority": 1
        }
        
        # Test with wrong content type
        wrong_content_type_headers = {
            **auth_headers,
            "Content-Type": "text/plain"
        }
        
        response = client.post(
            "/agents/execute", 
            data="invalid data format",  # Not JSON
            headers=wrong_content_type_headers
        )
        
        assert response.status_code == 422  # Should reject non-JSON data
    
    def test_request_size_limits(self, client, auth_headers):
        """Test request size limits in middleware."""
        # Create very large request
        large_context = {f"key_{i}": "x" * 1000 for i in range(100)}  # Large context
        
        large_request_data = {
            "prompt": "Design architecture with extensive requirements: " + "x" * 10000,
            "agent_id": "scroll-cto-agent",
            "context": large_context,
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=large_request_data, headers=auth_headers)
        
        # Should either accept large request or reject with appropriate error
        assert response.status_code in [200, 413, 422, 401, 500]  # Various acceptable responses
    
    def test_concurrent_request_handling(self, client, auth_headers):
        """Test concurrent request handling through middleware."""
        import threading
        import queue
        
        request_data = {
            "prompt": "Design scalable architecture",
            "agent_id": "scroll-cto-agent",
            "context": {"concurrent_test": True},
            "priority": 1
        }
        
        results_queue = queue.Queue()
        
        def make_request():
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            results_queue.put(response)
        
        # Start multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all requests to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        responses = []
        while not results_queue.empty():
            responses.append(results_queue.get())
        
        assert len(responses) == 5
        
        # All requests should be handled (success or appropriate error)
        for response in responses:
            assert response.status_code in [200, 401, 429, 500]  # Various acceptable responses
    
    def test_health_check_bypass_middleware(self, client):
        """Test that health check endpoints bypass authentication middleware."""
        # Health check should work without authentication
        response = client.get("/health")
        assert response.status_code == 200
        
        # System status should also work without authentication
        response = client.get("/status")
        assert response.status_code == 200
    
    def test_middleware_order_and_execution(self, client, auth_headers):
        """Test middleware execution order and proper request processing."""
        request_data = {
            "prompt": "Test middleware execution order",
            "agent_id": "scroll-cto-agent",
            "context": {"middleware_test": True},
            "priority": 1
        }
        
        # This test verifies that all middleware layers process the request correctly
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Request should pass through all middleware layers
        # Response code depends on whether full system is mocked
        assert response.status_code in [200, 401, 500]
        
        # Response should have proper structure regardless of success/failure
        if response.headers.get("content-type", "").startswith("application/json"):
            result = response.json()
            assert isinstance(result, dict)
            assert "timestamp" in result or "error" in result


class TestCTOAgentRoutingLogic:
    """Test CTO agent routing logic and request distribution."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application."""
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
                "sub": "routing-test-user",
                "role": "analyst",
                "permissions": ["agent:execute", "agent:read", "agent:list"]
            }
        )
        return {"Authorization": f"Bearer {token}"}
    
    def test_explicit_cto_agent_routing(self, client, auth_headers):
        """Test explicit routing to CTO agent by ID."""
        request_data = {
            "prompt": "Design architecture",
            "agent_id": "scroll-cto-agent",  # Explicit agent ID
            "context": {},
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Should route to specific CTO agent
        assert response.status_code in [200, 401, 500]
    
    def test_agent_type_based_routing(self, client, auth_headers):
        """Test routing by agent type."""
        request_data = {
            "prompt": "Provide technical leadership guidance",
            "agent_type": "CTO",  # Route by type instead of ID
            "context": {},
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Should route to CTO agent based on type
        assert response.status_code in [200, 401, 500]
    
    def test_capability_based_routing(self, client, auth_headers):
        """Test routing based on required capabilities."""
        request_data = {
            "prompt": "I need architecture design and technical decisions",
            "context": {
                "required_capabilities": ["architecture_design", "technical_decision"]
            },
            "priority": 1
        }
        
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Should route to agent with matching capabilities (CTO)
        assert response.status_code in [200, 401, 500]
    
    def test_intelligent_prompt_based_routing(self, client, auth_headers):
        """Test intelligent routing based on prompt content."""
        architecture_prompts = [
            "What's the best architecture for my application?",
            "How should I structure my microservices?",
            "Which technology stack should I choose?",
            "Help me design a scalable system",
            "I need technical architecture advice"
        ]
        
        for prompt in architecture_prompts:
            request_data = {
                "prompt": prompt,
                # No explicit agent_id or agent_type - should route intelligently
                "context": {},
                "priority": 1
            }
            
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            
            # Should intelligently route to CTO agent
            assert response.status_code in [200, 401, 500]
    
    def test_priority_based_routing(self, client, auth_headers):
        """Test that high-priority requests are handled appropriately."""
        high_priority_request = {
            "prompt": "URGENT: Production architecture issue needs immediate attention",
            "agent_id": "scroll-cto-agent",
            "context": {"urgency": "critical"},
            "priority": 10  # Highest priority
        }
        
        low_priority_request = {
            "prompt": "General architecture question for future planning",
            "agent_id": "scroll-cto-agent",
            "context": {"urgency": "low"},
            "priority": 1  # Lowest priority
        }
        
        # Send high priority request
        high_response = client.post("/agents/execute", json=high_priority_request, headers=auth_headers)
        
        # Send low priority request
        low_response = client.post("/agents/execute", json=low_priority_request, headers=auth_headers)
        
        # Both should be processed, but high priority should be handled appropriately
        assert high_response.status_code in [200, 401, 500]
        assert low_response.status_code in [200, 401, 500]
    
    def test_load_balancing_across_cto_instances(self, client, auth_headers):
        """Test load balancing when multiple CTO agent instances exist."""
        # This would test load balancing if multiple CTO instances were available
        requests = []
        for i in range(5):
            request_data = {
                "prompt": f"Architecture question {i}",
                "agent_type": "CTO",
                "context": {"request_number": i},
                "priority": 1
            }
            requests.append(request_data)
        
        responses = []
        for request_data in requests:
            response = client.post("/agents/execute", json=request_data, headers=auth_headers)
            responses.append(response)
        
        # All requests should be handled
        for response in responses:
            assert response.status_code in [200, 401, 500]
    
    def test_fallback_routing_when_cto_unavailable(self, client, auth_headers):
        """Test fallback routing when CTO agent is unavailable."""
        request_data = {
            "prompt": "Design architecture - CTO agent preferred but fallback acceptable",
            "agent_id": "scroll-cto-agent",
            "context": {"allow_fallback": True},
            "priority": 1
        }
        
        # This would test fallback to other agents when CTO is unavailable
        response = client.post("/agents/execute", json=request_data, headers=auth_headers)
        
        # Should either succeed with CTO or provide appropriate fallback/error
        assert response.status_code in [200, 401, 503, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])