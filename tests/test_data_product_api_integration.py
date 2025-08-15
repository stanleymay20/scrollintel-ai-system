"""
Integration Tests for Data Product API

Tests for REST API endpoints, GraphQL queries, WebSocket functionality,
and middleware integration.
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import jwt
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.api.data_product_app import app
from scrollintel.models.database import Base, get_db
from scrollintel.models.data_product_models import DataProduct, AccessLevel, VerificationStatus
from scrollintel.api.middleware.data_product_middleware import generate_access_token

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_data_products.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create test database
Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Test client
client = TestClient(app)


@pytest.fixture
def test_token():
    """Generate test JWT token"""
    return generate_access_token(
        user_id="test_user",
        permissions=[
            "data_product:create",
            "data_product:read",
            "data_product:update",
            "data_product:delete",
            "data_product:search",
            "data_product:manage_provenance",
            "data_product:manage_quality",
            "data_product:manage_bias",
            "data_product:verify"
        ],
        expires_in=3600
    )


@pytest.fixture
def auth_headers(test_token):
    """Generate authorization headers"""
    return {"Authorization": f"Bearer {test_token}"}


@pytest.fixture
def sample_data_product():
    """Sample data product for testing"""
    return {
        "name": "test_product",
        "description": "Test data product",
        "schema_definition": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"}
            }
        },
        "metadata": {"source": "test", "category": "sample"},
        "access_level": "INTERNAL",
        "compliance_tags": ["GDPR", "SOX"]
    }


class TestDataProductRestAPI:
    """Test REST API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_create_data_product_success(self, auth_headers, sample_data_product):
        """Test successful data product creation"""
        response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_data_product["name"]
        assert data["owner"] == "test_user"
        assert data["access_level"] == "INTERNAL"
    
    def test_create_data_product_unauthorized(self, sample_data_product):
        """Test data product creation without authentication"""
        response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product
        )
        
        assert response.status_code == 401
    
    def test_create_data_product_insufficient_permissions(self, sample_data_product):
        """Test data product creation with insufficient permissions"""
        # Token with limited permissions
        limited_token = generate_access_token(
            user_id="limited_user",
            permissions=["data_product:read"],
            expires_in=3600
        )
        
        headers = {"Authorization": f"Bearer {limited_token}"}
        
        response = client.post(
            "/api/v1/data-products/?owner=limited_user",
            json=sample_data_product,
            headers=headers
        )
        
        assert response.status_code == 403
    
    def test_get_data_product_success(self, auth_headers, sample_data_product):
        """Test successful data product retrieval"""
        # First create a product
        create_response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        product_id = create_response.json()["id"]
        
        # Then retrieve it
        response = client.get(
            f"/api/v1/data-products/{product_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == product_id
        assert data["name"] == sample_data_product["name"]
    
    def test_get_data_product_not_found(self, auth_headers):
        """Test data product retrieval with non-existent ID"""
        response = client.get(
            "/api/v1/data-products/non-existent-id",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_search_data_products(self, auth_headers, sample_data_product):
        """Test data product search"""
        # Create a test product
        client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        
        # Search for products
        response = client.get(
            "/api/v1/data-products/?query=test&limit=10",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_count" in data
    
    def test_update_data_product(self, auth_headers, sample_data_product):
        """Test data product update"""
        # Create a product
        create_response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        product_id = create_response.json()["id"]
        
        # Update the product
        update_data = {
            "description": "Updated description",
            "metadata": {"updated": True}
        }
        
        response = client.put(
            f"/api/v1/data-products/{product_id}?updated_by=test_user",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
    
    def test_delete_data_product(self, auth_headers, sample_data_product):
        """Test data product deletion"""
        # Create a product
        create_response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        product_id = create_response.json()["id"]
        
        # Delete the product
        response = client.delete(
            f"/api/v1/data-products/{product_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 204
        
        # Verify deletion
        get_response = client.get(
            f"/api/v1/data-products/{product_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404
    
    def test_rate_limiting(self, auth_headers, sample_data_product):
        """Test rate limiting functionality"""
        # This test would need to be adjusted based on actual rate limits
        # For now, we'll test that the endpoint responds correctly
        response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        
        # Check that rate limit headers are present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestDataProductGraphQL:
    """Test GraphQL functionality"""
    
    def test_graphql_query_data_product(self, auth_headers, sample_data_product):
        """Test GraphQL query for data product"""
        # Create a product first
        create_response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        product_id = create_response.json()["id"]
        
        # GraphQL query
        query = """
        query GetDataProduct($id: String!) {
            dataProduct(id: $id) {
                id
                name
                description
                owner
                accessLevel
                verificationStatus
                qualityScore
                biasScore
            }
        }
        """
        
        response = client.post(
            "/graphql",
            json={
                "query": query,
                "variables": {"id": product_id}
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["dataProduct"]["id"] == product_id
    
    def test_graphql_search_data_products(self, auth_headers, sample_data_product):
        """Test GraphQL search functionality"""
        # Create a product first
        client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        
        # GraphQL search query
        query = """
        query SearchDataProducts($input: SearchInput!) {
            searchDataProducts(input: $input) {
                products {
                    edges {
                        id
                        name
                        owner
                    }
                    totalCount
                }
                totalCount
                queryTimeMs
            }
        }
        """
        
        response = client.post(
            "/graphql",
            json={
                "query": query,
                "variables": {
                    "input": {
                        "query": "test",
                        "limit": 10
                    }
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "searchDataProducts" in data["data"]
    
    def test_graphql_mutation_create_product(self, auth_headers, sample_data_product):
        """Test GraphQL mutation for creating data product"""
        mutation = """
        mutation CreateDataProduct($input: DataProductCreateInput!) {
            createDataProduct(input: $input) {
                id
                name
                description
                owner
                accessLevel
            }
        }
        """
        
        response = client.post(
            "/graphql",
            json={
                "query": mutation,
                "variables": {
                    "input": {
                        "name": sample_data_product["name"],
                        "description": sample_data_product["description"],
                        "schemaDefinition": sample_data_product["schema_definition"],
                        "metadata": sample_data_product["metadata"],
                        "accessLevel": sample_data_product["access_level"],
                        "complianceTags": sample_data_product["compliance_tags"],
                        "owner": "test_user"
                    }
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["createDataProduct"]["name"] == sample_data_product["name"]


class TestDataProductWebSocket:
    """Test WebSocket functionality"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws/data-products") as websocket:
            # Should receive welcome message
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert "connection_id" in data["data"]
    
    @pytest.mark.asyncio
    async def test_websocket_subscription(self):
        """Test WebSocket subscription functionality"""
        with client.websocket_connect("/ws/data-products") as websocket:
            # Receive welcome message
            welcome = websocket.receive_json()
            assert welcome["type"] == "connected"
            
            # Subscribe to quality alerts
            websocket.send_json({
                "type": "subscribe_quality_alerts",
                "data": {}
            })
            
            # Should receive subscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscription_confirmed"
            assert response["data"]["type"] == "quality_alerts"
    
    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong functionality"""
        with client.websocket_connect("/ws/data-products") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Send ping
            websocket.send_json({
                "type": "ping",
                "data": {}
            })
            
            # Should receive pong
            response = websocket.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response["data"]


class TestMiddleware:
    """Test middleware functionality"""
    
    def test_authentication_middleware_missing_token(self, sample_data_product):
        """Test authentication middleware with missing token"""
        response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product
        )
        
        assert response.status_code == 401
        assert "authorization header" in response.json()["error"].lower()
    
    def test_authentication_middleware_invalid_token(self, sample_data_product):
        """Test authentication middleware with invalid token"""
        headers = {"Authorization": "Bearer invalid-token"}
        
        response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=headers
        )
        
        assert response.status_code == 401
        assert "invalid token" in response.json()["error"].lower()
    
    def test_authentication_middleware_expired_token(self, sample_data_product):
        """Test authentication middleware with expired token"""
        # Create expired token
        expired_token = generate_access_token(
            user_id="test_user",
            permissions=["data_product:create"],
            expires_in=-3600  # Expired 1 hour ago
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        
        response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=headers
        )
        
        assert response.status_code == 401
        assert "expired" in response.json()["error"].lower()
    
    def test_security_headers(self, auth_headers):
        """Test security headers middleware"""
        response = client.get("/health", headers=auth_headers)
        
        # Check security headers
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_error_handler(self, auth_headers):
        """Test 404 error handler"""
        response = client.get("/non-existent-endpoint", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Not Found"
        assert "non-existent-endpoint" in data["message"]
    
    def test_validation_error_handling(self, auth_headers):
        """Test validation error handling"""
        invalid_data = {
            "name": "",  # Empty name should fail validation
            "schema_definition": "invalid"  # Should be object, not string
        }
        
        response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=invalid_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error


class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint"""
        response = client.get("/api/v1/schema")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_list_endpoints(self):
        """Test endpoint listing"""
        response = client.get("/api/v1/endpoints")
        
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "total_count" in data
        assert len(data["endpoints"]) > 0
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "api_metrics" in data
        assert "websocket_connections" in data["api_metrics"]


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_complete_data_product_lifecycle(self, auth_headers, sample_data_product):
        """Test complete data product lifecycle"""
        # 1. Create data product
        create_response = client.post(
            "/api/v1/data-products/?owner=test_user",
            json=sample_data_product,
            headers=auth_headers
        )
        assert create_response.status_code == 201
        product_id = create_response.json()["id"]
        
        # 2. Retrieve data product
        get_response = client.get(
            f"/api/v1/data-products/{product_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        
        # 3. Update quality metrics
        quality_update = {
            "completeness_score": 0.95,
            "accuracy_score": 0.90,
            "consistency_score": 0.85,
            "timeliness_score": 0.92,
            "assessed_by": "test_user"
        }
        
        quality_response = client.put(
            f"/api/v1/data-products/{product_id}/quality-metrics",
            json=quality_update,
            headers=auth_headers
        )
        assert quality_response.status_code == 200
        
        # 4. Update verification status
        verification_update = {
            "verification_status": "VERIFIED",
            "verified_by": "test_user"
        }
        
        verification_response = client.put(
            f"/api/v1/data-products/{product_id}/verification",
            json=verification_update,
            headers=auth_headers
        )
        assert verification_response.status_code == 200
        
        # 5. Search for the product
        search_response = client.get(
            f"/api/v1/data-products/?query={sample_data_product['name']}",
            headers=auth_headers
        )
        assert search_response.status_code == 200
        assert search_response.json()["total_count"] >= 1
        
        # 6. Delete data product
        delete_response = client.delete(
            f"/api/v1/data-products/{product_id}",
            headers=auth_headers
        )
        assert delete_response.status_code == 204
        
        # 7. Verify deletion
        final_get_response = client.get(
            f"/api/v1/data-products/{product_id}",
            headers=auth_headers
        )
        assert final_get_response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])