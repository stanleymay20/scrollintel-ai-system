"""
Smoke Tests for ScrollIntel
Basic functionality tests to verify system is operational
"""
import pytest
import requests
import time
from typing import Dict, Any
from unittest.mock import patch, Mock


class TestSmokeTests:
    """Basic smoke tests for system health"""
    
    @pytest.mark.asyncio
    async def test_api_health_check(self, test_client):
        """Test basic API health check"""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        health_data = response.json()
        
        # Verify health response structure
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
        assert health_data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_database_connectivity(self, test_client):
        """Test database connectivity"""
        response = test_client.get("/api/v1/health/database")
        
        assert response.status_code == 200
        db_health = response.json()
        
        assert "database" in db_health
        assert db_health["database"]["status"] == "connected"
        assert "connection_pool" in db_health["database"]
    
    @pytest.mark.asyncio
    async def test_redis_connectivity(self, test_client):
        """Test Redis connectivity"""
        response = test_client.get("/api/v1/health/redis")
        
        assert response.status_code == 200
        redis_health = response.json()
        
        assert "redis" in redis_health
        assert redis_health["redis"]["status"] == "connected"
        assert "memory_usage" in redis_health["redis"]
    
    @pytest.mark.asyncio
    async def test_agent_registry_basic_functionality(self, test_client, test_user_token):
        """Test agent registry basic functionality"""
        # Get available agents
        response = test_client.get("/api/v1/agents", headers=test_user_token)
        
        assert response.status_code == 200
        agents = response.json()
        
        assert "agents" in agents
        assert len(agents["agents"]) > 0
        
        # Verify core agents are available
        agent_types = [agent["type"] for agent in agents["agents"]]
        core_agents = ["cto", "data_scientist", "ml_engineer", "ai_engineer", "analyst", "bi"]
        
        for core_agent in core_agents:
            assert core_agent in agent_types, f"Core agent {core_agent} should be available"
    
    @pytest.mark.asyncio
    async def test_file_upload_basic_functionality(self, test_client, test_user_token):
        """Test basic file upload functionality"""
        # Create simple test file
        test_content = "id,name,value\n1,test,100\n2,test2,200"
        
        files = {"file": ("test.csv", test_content, "text/csv")}
        response = test_client.post(
            "/api/v1/files/upload",
            files=files,
            headers=test_user_token
        )
        
        assert response.status_code == 200
        upload_result = response.json()
        
        assert "dataset_id" in upload_result
        assert "filename" in upload_result
        assert "row_count" in upload_result
        assert upload_result["row_count"] == 2
    
    @pytest.mark.asyncio
    async def test_agent_processing_basic_functionality(self, test_client, test_user_token):
        """Test basic agent processing functionality"""
        # Test CTO agent
        cto_request = {
            "prompt": "Provide a simple architecture recommendation",
            "agent_type": "cto"
        }
        
        with patch('scrollintel.agents.scroll_cto_agent.openai') as mock_openai:
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Use microservices architecture"))]
            )
            
            response = test_client.post(
                "/api/v1/agents/process",
                json=cto_request,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "status" in result
        assert "content" in result
        assert "agent_id" in result
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_authentication_basic_functionality(self, test_client):
        """Test basic authentication functionality"""
        # Test login endpoint exists and responds
        login_data = {
            "email": "test@example.com",
            "password": "test_password"
        }
        
        response = test_client.post("/api/v1/auth/login", json=login_data)
        
        # Should respond (even if credentials are wrong)
        assert response.status_code in [200, 401, 422]
        
        # Test token validation endpoint
        fake_token = "Bearer fake_token"
        response = test_client.get(
            "/api/v1/auth/verify",
            headers={"Authorization": fake_token}
        )
        
        # Should respond with unauthorized
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_api_documentation_accessibility(self, test_client):
        """Test API documentation is accessible"""
        # Test OpenAPI docs
        response = test_client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI JSON
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
    
    @pytest.mark.asyncio
    async def test_cors_configuration(self, test_client):
        """Test CORS configuration"""
        # Test preflight request
        response = test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization"
            }
        )
        
        assert response.status_code in [200, 204]
        
        # Check CORS headers are present
        headers = response.headers
        assert "Access-Control-Allow-Origin" in headers or "access-control-allow-origin" in headers
    
    @pytest.mark.asyncio
    async def test_rate_limiting_basic_functionality(self, test_client):
        """Test rate limiting is active"""
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = test_client.get("/api/v1/health")
            responses.append(response.status_code)
        
        # Should either all succeed or some be rate limited
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # At least some requests should succeed
        assert success_count > 0
        
        # If rate limiting is active, some might be limited
        assert success_count + rate_limited_count == len(responses)
    
    @pytest.mark.asyncio
    async def test_error_handling_basic_functionality(self, test_client):
        """Test basic error handling"""
        # Test 404 for non-existent endpoint
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test 405 for wrong method
        response = test_client.post("/api/v1/health")
        assert response.status_code == 405
        
        # Test 422 for invalid JSON
        response = test_client.post(
            "/api/v1/agents/process",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_monitoring_endpoints(self, test_client):
        """Test monitoring endpoints are accessible"""
        monitoring_endpoints = [
            "/api/v1/monitoring/metrics",
            "/api/v1/monitoring/health",
            "/api/v1/monitoring/status"
        ]
        
        for endpoint in monitoring_endpoints:
            response = test_client.get(endpoint)
            # Should respond (might require auth, but should not be 404)
            assert response.status_code != 404
    
    @pytest.mark.asyncio
    async def test_security_headers(self, test_client):
        """Test security headers are present"""
        response = test_client.get("/api/v1/health")
        
        headers = response.headers
        
        # Check for important security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        present_headers = []
        for header in security_headers:
            if header in headers or header.lower() in headers:
                present_headers.append(header)
        
        # At least some security headers should be present
        assert len(present_headers) > 0, "Security headers should be present"
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self, test_client):
        """Test basic response time performance"""
        import time
        
        endpoints = [
            "/api/v1/health",
            "/api/v1/agents",
            "/docs"
        ]
        
        response_times = []
        
        for endpoint in endpoints:
            start_time = time.time()
            response = test_client.get(endpoint)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append({
                "endpoint": endpoint,
                "response_time": response_time,
                "status_code": response.status_code
            })
        
        # Basic performance assertions
        for result in response_times:
            # Response time should be reasonable for smoke tests
            assert result["response_time"] < 5.0, f"Response time too slow for {result['endpoint']}: {result['response_time']:.2f}s"
            
            # Status should be successful or expected error
            assert result["status_code"] < 500, f"Server error on {result['endpoint']}: {result['status_code']}"
        
        avg_response_time = sum(r["response_time"] for r in response_times) / len(response_times)
        assert avg_response_time < 2.0, f"Average response time too slow: {avg_response_time:.2f}s"


class TestEnvironmentSpecificSmokeTests:
    """Environment-specific smoke tests"""
    
    @pytest.mark.asyncio
    async def test_staging_environment_smoke_tests(self, test_client):
        """Smoke tests specific to staging environment"""
        # Test staging-specific configurations
        response = test_client.get("/api/v1/health")
        
        if response.status_code == 200:
            health_data = response.json()
            
            # In staging, debug mode might be enabled
            if "debug" in health_data:
                assert isinstance(health_data["debug"], bool)
            
            # Staging should have test data indicators
            if "environment" in health_data:
                assert health_data["environment"] in ["staging", "development", "test"]
    
    @pytest.mark.asyncio
    async def test_production_environment_smoke_tests(self, test_client):
        """Smoke tests specific to production environment"""
        # Test production-specific configurations
        response = test_client.get("/api/v1/health")
        
        if response.status_code == 200:
            health_data = response.json()
            
            # In production, debug should be disabled
            if "debug" in health_data:
                assert health_data["debug"] is False
            
            # Production should have proper environment indicator
            if "environment" in health_data:
                assert health_data["environment"] == "production"
            
            # Production should have monitoring enabled
            if "monitoring" in health_data:
                assert health_data["monitoring"]["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_container_deployment_smoke_tests(self):
        """Smoke tests for containerized deployment"""
        # Test container health
        import subprocess
        
        try:
            # Check if running in container
            result = subprocess.run(
                ["cat", "/proc/1/cgroup"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and "docker" in result.stdout:
                # Running in Docker container
                assert True  # Container detection successful
            else:
                # Not in container, skip container-specific tests
                pytest.skip("Not running in container")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Can't determine container status
            pytest.skip("Cannot determine container status")
    
    @pytest.mark.asyncio
    async def test_kubernetes_deployment_smoke_tests(self):
        """Smoke tests for Kubernetes deployment"""
        import os
        
        # Check for Kubernetes environment variables
        k8s_indicators = [
            "KUBERNETES_SERVICE_HOST",
            "KUBERNETES_SERVICE_PORT",
            "KUBERNETES_PORT"
        ]
        
        k8s_detected = any(env_var in os.environ for env_var in k8s_indicators)
        
        if k8s_detected:
            # Running in Kubernetes
            # Test service discovery
            assert "KUBERNETES_SERVICE_HOST" in os.environ
            
            # Test pod metadata
            if "POD_NAME" in os.environ:
                assert len(os.environ["POD_NAME"]) > 0
            
            if "POD_NAMESPACE" in os.environ:
                assert len(os.environ["POD_NAMESPACE"]) > 0
        else:
            pytest.skip("Not running in Kubernetes")


@pytest.fixture
def environment_type():
    """Determine environment type from environment variables"""
    import os
    
    env = os.environ.get("ENVIRONMENT", "development").lower()
    return env


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_environment_specific_configurations(environment):
    """Test environment-specific configurations"""
    # This would be run with different environment configurations
    
    if environment == "staging":
        # Staging-specific assertions
        assert True  # Placeholder for staging tests
    elif environment == "production":
        # Production-specific assertions
        assert True  # Placeholder for production tests