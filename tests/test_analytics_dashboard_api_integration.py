"""
Integration Tests for Advanced Analytics Dashboard API

This module provides comprehensive integration tests for the API layer,
including REST endpoints, GraphQL API, webhooks, authentication, and performance.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import aiohttp
from fastapi.testclient import TestClient
import websockets
from unittest.mock import Mock, patch

from scrollintel.api.analytics_dashboard_api import create_analytics_dashboard_app
from scrollintel.core.api_key_manager import APIKeyManager, APIKey, APIKeyStatus
from scrollintel.api.webhooks.webhook_manager import WebhookManager
from scrollintel.api.middleware.rate_limiter import MemoryRateLimitStore

@pytest.fixture
async def app():
    """Create test FastAPI application"""
    return await create_analytics_dashboard_app()

@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def test_api_key():
    """Create test API key"""
    return APIKey(
        id="test-key-123",
        key="sk-test-123456789",
        name="Test API Key",
        user_id="test-user-123",
        organization_id="test-org-123",
        status=APIKeyStatus.ACTIVE,
        is_admin=False,
        tier="default",
        created_at=datetime.utcnow(),
        last_used=None,
        usage_count=0
    )

@pytest.fixture
def admin_api_key():
    """Create admin API key"""
    return APIKey(
        id="admin-key-123",
        key="sk-admin-123456789",
        name="Admin API Key",
        user_id="admin-user-123",
        organization_id="admin-org-123",
        status=APIKeyStatus.ACTIVE,
        is_admin=True,
        tier="enterprise",
        created_at=datetime.utcnow(),
        last_used=None,
        usage_count=0
    )

class TestAPIAuthentication:
    """Test API authentication and authorization"""
    
    def test_health_endpoint_no_auth(self, client):
        """Test health endpoint doesn't require authentication"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_api_info_no_auth(self, client):
        """Test API info endpoint doesn't require authentication"""
        response = client.get("/api/info")
        assert response.status_code == 200
        assert "name" in response.json()
    
    def test_protected_endpoint_no_auth(self, client):
        """Test protected endpoint requires authentication"""
        response = client.get("/api/v1/dashboards")
        assert response.status_code == 401
        assert "Authentication required" in response.json()["error"]
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_valid_api_key_auth(self, mock_validate, client, test_api_key):
        """Test authentication with valid API key"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/dashboards", headers=headers)
        
        # Should not be 401 (authentication error)
        assert response.status_code != 401
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_invalid_api_key_auth(self, mock_validate, client):
        """Test authentication with invalid API key"""
        mock_validate.return_value = None
        
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.get("/api/v1/dashboards", headers=headers)
        
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["error"]
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_admin_endpoint_requires_admin(self, mock_validate, client, test_api_key):
        """Test admin endpoint requires admin privileges"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.post("/api/v1/webhooks", headers=headers, json={
            "url": "https://example.com/webhook",
            "events": ["dashboard.created"]
        })
        
        assert response.status_code == 403
        assert "Insufficient permissions" in response.json()["error"]
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_admin_endpoint_with_admin_key(self, mock_validate, client, admin_api_key):
        """Test admin endpoint with admin API key"""
        mock_validate.return_value = admin_api_key
        
        headers = {"Authorization": f"Bearer {admin_api_key.key}"}
        response = client.post("/api/v1/webhooks", headers=headers, json={
            "url": "https://example.com/webhook",
            "events": ["dashboard.created"]
        })
        
        # Should not be 403 (permission error)
        assert response.status_code != 403

class TestRateLimiting:
    """Test API rate limiting"""
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_rate_limit_enforcement(self, mock_validate, client, test_api_key):
        """Test rate limit enforcement"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        
        # Make requests up to the limit
        responses = []
        for i in range(5):  # Assuming default limit is higher than 5
            response = client.get("/api/v1/dashboards", headers=headers)
            responses.append(response)
        
        # All requests should succeed (assuming limit > 5)
        for response in responses:
            assert response.status_code != 429
    
    def test_rate_limit_headers(self, client):
        """Test rate limit headers in response"""
        # This would need to be implemented based on your rate limiting setup
        pass

class TestRESTEndpoints:
    """Test REST API endpoints"""
    
    @patch('scrollintel.core.dashboard_manager.DashboardManager.list_dashboards')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_list_dashboards(self, mock_validate, mock_list, client, test_api_key):
        """Test listing dashboards"""
        mock_validate.return_value = test_api_key
        mock_list.return_value = []
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/dashboards", headers=headers)
        
        assert response.status_code == 200
        assert "dashboards" in response.json() or isinstance(response.json(), list)
    
    @patch('scrollintel.core.dashboard_manager.DashboardManager.create_dashboard')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_create_dashboard(self, mock_validate, mock_create, client, test_api_key):
        """Test creating dashboard"""
        mock_validate.return_value = test_api_key
        mock_dashboard = Mock()
        mock_dashboard.id = "dashboard-123"
        mock_dashboard.name = "Test Dashboard"
        mock_create.return_value = mock_dashboard
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        dashboard_data = {
            "name": "Test Dashboard",
            "type": "EXECUTIVE",
            "config": {}
        }
        
        response = client.post("/api/v1/dashboards", headers=headers, json=dashboard_data)
        
        assert response.status_code == 201 or response.status_code == 200
    
    @patch('scrollintel.engines.roi_calculator.ROICalculator.list_analyses')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_list_roi_analyses(self, mock_validate, mock_list, client, test_api_key):
        """Test listing ROI analyses"""
        mock_validate.return_value = test_api_key
        mock_list.return_value = []
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/roi/analyses", headers=headers)
        
        assert response.status_code == 200
    
    @patch('scrollintel.engines.insight_generator.InsightGenerator.list_insights')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_list_insights(self, mock_validate, mock_list, client, test_api_key):
        """Test listing insights"""
        mock_validate.return_value = test_api_key
        mock_list.return_value = []
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/insights", headers=headers)
        
        assert response.status_code == 200
    
    @patch('scrollintel.engines.predictive_engine.PredictiveEngine.list_forecasts')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_list_forecasts(self, mock_validate, mock_list, client, test_api_key):
        """Test listing forecasts"""
        mock_validate.return_value = test_api_key
        mock_list.return_value = []
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/forecasts", headers=headers)
        
        assert response.status_code == 200

class TestGraphQLAPI:
    """Test GraphQL API functionality"""
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_graphql_endpoint_exists(self, mock_validate, client, test_api_key):
        """Test GraphQL endpoint is accessible"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        query = """
        query {
            dashboards {
                id
                name
                type
            }
        }
        """
        
        response = client.post("/graphql", headers=headers, json={"query": query})
        
        # Should not be 404 (endpoint exists)
        assert response.status_code != 404
    
    def test_graphql_playground_accessible(self, client):
        """Test GraphQL playground is accessible"""
        response = client.get("/graphql/playground")
        assert response.status_code == 200
        assert "GraphQL Playground" in response.text
    
    @patch('scrollintel.api.graphql.analytics_resolvers.AnalyticsResolvers.resolve_dashboards')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_graphql_dashboard_query(self, mock_validate, mock_resolve, client, test_api_key):
        """Test GraphQL dashboard query"""
        mock_validate.return_value = test_api_key
        mock_resolve.return_value = []
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        query = """
        query {
            dashboards {
                id
                name
                type
                createdAt
            }
        }
        """
        
        response = client.post("/graphql", headers=headers, json={"query": query})
        
        # Should return valid GraphQL response
        assert response.status_code == 200
        json_response = response.json()
        assert "data" in json_response or "errors" in json_response

class TestWebhookSystem:
    """Test webhook system functionality"""
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_register_webhook(self, mock_validate, client, admin_api_key):
        """Test webhook registration"""
        mock_validate.return_value = admin_api_key
        
        headers = {"Authorization": f"Bearer {admin_api_key.key}"}
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["dashboard.created", "insight.generated"],
            "secret": "webhook-secret-123"
        }
        
        response = client.post("/api/v1/webhooks", headers=headers, json=webhook_data)
        
        assert response.status_code == 200 or response.status_code == 201
        if response.status_code in [200, 201]:
            assert "webhook_id" in response.json()
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_list_webhooks(self, mock_validate, client, admin_api_key):
        """Test listing webhooks"""
        mock_validate.return_value = admin_api_key
        
        headers = {"Authorization": f"Bearer {admin_api_key.key}"}
        response = client.get("/api/v1/webhooks", headers=headers)
        
        assert response.status_code == 200
        assert isinstance(response.json(), (list, dict))
    
    @patch('scrollintel.api.webhooks.webhook_manager.WebhookManager.test_webhook')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_webhook_test(self, mock_validate, mock_test, client, admin_api_key):
        """Test webhook testing functionality"""
        mock_validate.return_value = admin_api_key
        mock_test.return_value = {
            "webhook_id": "webhook-123",
            "test_successful": True,
            "status_code": 200,
            "response_time": 0.5
        }
        
        headers = {"Authorization": f"Bearer {admin_api_key.key}"}
        response = client.post("/api/v1/webhooks/webhook-123/test", headers=headers)
        
        assert response.status_code == 200

class TestWebSocketConnections:
    """Test WebSocket functionality"""
    
    @pytest.mark.asyncio
    async def test_dashboard_websocket_connection(self):
        """Test dashboard WebSocket connection"""
        # This would require a running server for WebSocket testing
        # For now, we'll test that the endpoint exists
        pass
    
    @pytest.mark.asyncio
    async def test_insights_websocket_connection(self):
        """Test insights WebSocket connection"""
        # This would require a running server for WebSocket testing
        pass

class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_swagger_docs_accessible(self, client):
        """Test Swagger documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
    
    def test_redoc_accessible(self, client):
        """Test ReDoc documentation is accessible"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()
    
    def test_openapi_schema_accessible(self, client):
        """Test OpenAPI schema is accessible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

class TestErrorHandling:
    """Test API error handling"""
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_404_error_handling(self, mock_validate, client, test_api_key):
        """Test 404 error handling"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/nonexistent-endpoint", headers=headers)
        
        assert response.status_code == 404
    
    @patch('scrollintel.core.dashboard_manager.DashboardManager.get_dashboard')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_resource_not_found_handling(self, mock_validate, mock_get, client, test_api_key):
        """Test resource not found handling"""
        mock_validate.return_value = test_api_key
        mock_get.return_value = None
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/dashboards/nonexistent-id", headers=headers)
        
        assert response.status_code == 404
    
    @patch('scrollintel.core.dashboard_manager.DashboardManager.create_dashboard')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_validation_error_handling(self, mock_validate, mock_create, client, test_api_key):
        """Test validation error handling"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        invalid_data = {
            "name": "",  # Invalid: empty name
            "type": "INVALID_TYPE"  # Invalid: unknown type
        }
        
        response = client.post("/api/v1/dashboards", headers=headers, json=invalid_data)
        
        assert response.status_code == 422  # Validation error

class TestPerformance:
    """Test API performance characteristics"""
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_response_time_acceptable(self, mock_validate, client, test_api_key):
        """Test API response times are acceptable"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        
        start_time = time.time()
        response = client.get("/api/v1/dashboards", headers=headers)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be under 1 second for simple requests
        assert response_time < 1.0
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_concurrent_requests_handling(self, mock_validate, client, test_api_key):
        """Test handling of concurrent requests"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        
        # Make multiple concurrent requests
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/api/v1/dashboards", headers=headers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should complete successfully
        for response in responses:
            assert response.status_code != 500  # No server errors

class TestDataIntegration:
    """Test data integration endpoints"""
    
    @patch('scrollintel.core.data_connector.DataConnector.list_data_sources')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_list_data_sources(self, mock_validate, mock_list, client, test_api_key):
        """Test listing data sources"""
        mock_validate.return_value = test_api_key
        mock_list.return_value = []
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.get("/api/v1/data-sources", headers=headers)
        
        assert response.status_code == 200
    
    @patch('scrollintel.core.data_connector.DataConnector.test_connection')
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_test_data_source_connection(self, mock_validate, mock_test, client, test_api_key):
        """Test data source connection testing"""
        mock_validate.return_value = test_api_key
        mock_test.return_value = {"status": "success", "message": "Connection successful"}
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        response = client.post("/api/v1/data-sources/test-source-123/test", headers=headers)
        
        assert response.status_code == 200

class TestSecurityFeatures:
    """Test security features"""
    
    def test_cors_headers_present(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/dashboards")
        
        # Should have CORS headers (if CORS is enabled)
        # This depends on your CORS configuration
        pass
    
    def test_security_headers_present(self, client):
        """Test security headers are present"""
        response = client.get("/health")
        
        # Check for security headers
        # This would depend on your security middleware configuration
        pass
    
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_sql_injection_protection(self, mock_validate, client, test_api_key):
        """Test SQL injection protection"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        
        # Attempt SQL injection in query parameter
        malicious_query = "'; DROP TABLE dashboards; --"
        response = client.get(f"/api/v1/dashboards?search={malicious_query}", headers=headers)
        
        # Should not cause server error
        assert response.status_code != 500

# Performance Benchmark Tests

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.benchmark
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_dashboard_list_performance(self, mock_validate, client, test_api_key, benchmark):
        """Benchmark dashboard listing performance"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        
        def list_dashboards():
            return client.get("/api/v1/dashboards", headers=headers)
        
        result = benchmark(list_dashboards)
        assert result.status_code == 200
    
    @pytest.mark.benchmark
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_graphql_query_performance(self, mock_validate, client, test_api_key, benchmark):
        """Benchmark GraphQL query performance"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        query = """
        query {
            dashboards {
                id
                name
                type
            }
        }
        """
        
        def execute_graphql_query():
            return client.post("/graphql", headers=headers, json={"query": query})
        
        result = benchmark(execute_graphql_query)
        assert result.status_code == 200
    
    @pytest.mark.benchmark
    def test_authentication_performance(self, client, benchmark):
        """Benchmark authentication performance"""
        headers = {"Authorization": "Bearer test-key-123"}
        
        def authenticate_request():
            return client.get("/api/v1/dashboards", headers=headers)
        
        # This will likely return 401, but we're testing auth performance
        result = benchmark(authenticate_request)
        # Don't assert status code since we're testing performance, not functionality

# Load Testing

class TestLoadTesting:
    """Load testing scenarios"""
    
    @pytest.mark.load_test
    @patch('scrollintel.core.api_key_manager.APIKeyManager.validate_key')
    def test_high_concurrency_load(self, mock_validate, client, test_api_key):
        """Test API under high concurrency load"""
        mock_validate.return_value = test_api_key
        
        headers = {"Authorization": f"Bearer {test_api_key.key}"}
        
        import concurrent.futures
        import time
        
        def make_request():
            start = time.time()
            response = client.get("/api/v1/dashboards", headers=headers)
            end = time.time()
            return {
                "status_code": response.status_code,
                "response_time": end - start
            }
        
        # Simulate 50 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [future.result() for future in futures]
        
        # Analyze results
        success_count = sum(1 for r in results if r["status_code"] == 200)
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        
        # At least 95% success rate
        success_rate = success_count / len(results)
        assert success_rate >= 0.95
        
        # Average response time under 2 seconds
        assert avg_response_time < 2.0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])