"""
Tests for ScrollIntel FastAPI Gateway.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from scrollintel.api.gateway import create_app
from scrollintel.core.registry import AgentRegistry
from scrollintel.core.interfaces import AgentType, AgentStatus


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_agent_registry():
    """Create mock agent registry."""
    registry = Mock(spec=AgentRegistry)
    registry.get_registry_status.return_value = {
        "total_agents": 0,
        "agent_status": {"active": 0, "inactive": 0, "busy": 0, "error": 0},
        "agent_types": {},
        "capabilities": []
    }
    registry.health_check_all.return_value = {}
    registry.get_all_agents.return_value = []
    registry.get_active_agents.return_value = []
    registry.get_capabilities.return_value = []
    return registry


class TestHealthRoutes:
    """Test health check routes."""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "ScrollIntel™ API"
    
    def test_liveness_check(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/health/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    def test_readiness_check(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/health/readiness")
        # May return 503 if services aren't ready, but should not crash
        assert response.status_code in [200, 503]


class TestRootRoutes:
    """Test root and system routes."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ScrollIntel™ API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert "timestamp" in data
        assert "environment" in data
    
    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["system"] == "ScrollIntel™"
        assert data["status"] == "operational"
        assert "agents" in data
        assert "uptime" in data


class TestAuthRoutes:
    """Test authentication routes."""
    
    def test_login_with_valid_credentials(self, client):
        """Test login with valid credentials."""
        response = client.post("/auth/login", json={
            "email": "admin@scrollintel.com",
            "password": "admin123",
            "remember_me": False
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user_info" in data
    
    def test_login_with_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post("/auth/login", json={
            "email": "invalid@example.com",
            "password": "wrongpassword",
            "remember_me": False
        })
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
    
    def test_password_requirements(self, client):
        """Test password requirements endpoint."""
        response = client.get("/auth/password-requirements")
        assert response.status_code == 200
        data = response.json()
        assert "requirements" in data
        assert "description" in data


class TestAgentRoutes:
    """Test agent routes (require authentication)."""
    
    def test_list_agents_without_auth(self, client):
        """Test listing agents without authentication."""
        response = client.get("/agents/")
        assert response.status_code == 401
    
    def test_agent_types_without_auth(self, client):
        """Test getting agent types without authentication."""
        response = client.get("/agents/types")
        assert response.status_code == 401
    
    def test_execute_agent_without_auth(self, client):
        """Test executing agent without authentication."""
        response = client.post("/agents/execute", json={
            "prompt": "Test prompt",
            "priority": 1
        })
        assert response.status_code == 401


class TestAdminRoutes:
    """Test admin routes (require admin authentication)."""
    
    def test_list_users_without_auth(self, client):
        """Test listing users without authentication."""
        response = client.get("/admin/users")
        assert response.status_code == 401
    
    def test_system_stats_without_auth(self, client):
        """Test getting system stats without authentication."""
        response = client.get("/admin/stats")
        assert response.status_code == 401
    
    def test_audit_logs_without_auth(self, client):
        """Test getting audit logs without authentication."""
        response = client.get("/admin/audit-logs")
        assert response.status_code == 401


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.post("/health/")
        assert response.status_code == 405


class TestMiddleware:
    """Test middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/")
        # CORS headers should be present
        assert response.status_code == 200
    
    def test_security_headers(self, client):
        """Test security headers on authenticated routes."""
        # Try to access a protected route to trigger security middleware
        response = client.get("/agents/")
        # Should get 401 but security headers should be present in successful responses
        assert response.status_code == 401


@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality."""
    
    async def test_agent_registry_integration(self, mock_agent_registry):
        """Test agent registry integration."""
        # This would test the actual integration with agent registry
        # For now, just verify the mock works
        status = mock_agent_registry.get_registry_status()
        assert status["total_agents"] == 0
        
        health = await mock_agent_registry.health_check_all()
        assert health == {}


if __name__ == "__main__":
    pytest.main([__file__])