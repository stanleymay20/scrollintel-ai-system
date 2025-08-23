"""
Comprehensive tests for API Integration Framework
Tests REST, GraphQL, and SOAP connectors with various authentication methods
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from scrollintel.connectors.api_connectors import (
    APIConnectorFactory, RESTAPIConnector, GraphQLAPIConnector, SOAPAPIConnector,
    APIType, AuthType, AuthConfig, RateLimitConfig, APIEndpoint,
    RateLimiter, WebhookManager, APISchemaDiscovery
)


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_limit=10
        )
        limiter = RateLimiter(config)
        
        assert limiter.config.requests_per_minute == 60
        assert limiter.config.requests_per_hour == 1000
        assert limiter.tokens == 10
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_tokens(self):
        """Test token acquisition"""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=5
        )
        limiter = RateLimiter(config)
        
        # Should be able to acquire up to burst limit
        for _ in range(5):
            assert await limiter.acquire() == True
        
        # Should fail after burst limit
        assert await limiter.acquire() == False
    
    def test_rate_limiter_wait_time(self):
        """Test wait time calculation"""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=1)
        limiter = RateLimiter(config)
        limiter.tokens = 0.5
        
        wait_time = limiter.get_wait_time()
        assert wait_time > 0


class TestAPIConnectorFactory:
    """Test API connector factory"""
    
    def test_create_rest_connector(self):
        """Test REST connector creation"""
        auth_config = AuthConfig(auth_type=AuthType.NONE)
        connector = APIConnectorFactory.create_connector(
            APIType.REST, "https://api.example.com", auth_config
        )
        
        assert isinstance(connector, RESTAPIConnector)
        assert connector.base_url == "https://api.example.com"
    
    def test_create_graphql_connector(self):
        """Test GraphQL connector creation"""
        auth_config = AuthConfig(auth_type=AuthType.BEARER, credentials={"token": "test"})
        connector = APIConnectorFactory.create_connector(
            APIType.GRAPHQL, "https://api.example.com/graphql", auth_config
        )
        
        assert isinstance(connector, GraphQLAPIConnector)
        assert connector.base_url == "https://api.example.com/graphql"
    
    def test_create_soap_connector(self):
        """Test SOAP connector creation"""
        auth_config = AuthConfig(auth_type=AuthType.BASIC, credentials={"username": "user", "password": "pass"})
        connector = APIConnectorFactory.create_connector(
            APIType.SOAP, "https://api.example.com/soap", auth_config
        )
        
        assert isinstance(connector, SOAPAPIConnector)
        assert connector.base_url == "https://api.example.com/soap"
    
    def test_invalid_api_type(self):
        """Test invalid API type handling"""
        auth_config = AuthConfig(auth_type=AuthType.NONE)
        
        with pytest.raises(ValueError):
            APIConnectorFactory.create_connector(
                "invalid_type", "https://api.example.com", auth_config
            )


class TestRESTAPIConnector:
    """Test REST API connector functionality"""
    
    @pytest.fixture
    def rest_connector(self):
        """Create REST connector for testing"""
        auth_config = AuthConfig(auth_type=AuthType.API_KEY, credentials={"api_key": "test_key"})
        rate_config = RateLimitConfig(requests_per_minute=100, burst_limit=10)
        return RESTAPIConnector("https://api.example.com", auth_config, rate_config)
    
    @pytest.mark.asyncio
    async def test_rest_authentication_headers(self, rest_connector):
        """Test REST authentication header generation"""
        headers = rest_connector._get_auth_headers()
        assert "X-API-Key" in headers
        assert headers["X-API-Key"] == "test_key"
    
    @pytest.mark.asyncio
    async def test_oauth2_authentication(self):
        """Test OAuth2 authentication flow"""
        auth_config = AuthConfig(
            auth_type=AuthType.OAUTH2,
            credentials={"client_id": "test_id", "client_secret": "test_secret"},
            token_url="https://auth.example.com/token"
        )
        connector = RESTAPIConnector("https://api.example.com", auth_config)
        
        # Mock the session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "access_token": "test_token",
            "expires_in": 3600
        })
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            connector.session = mock_session
            
            await connector._oauth2_authenticate()
            
            assert connector._auth_token == "test_token"
            assert connector._token_expires_at is not None
    
    @pytest.mark.asyncio
    async def test_rest_make_request_success(self, rest_connector):
        """Test successful REST API request"""
        endpoint = APIEndpoint(url="/users", method="GET", params={"limit": 10})
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"users": [{"id": 1, "name": "Test"}]})
        mock_response.raise_for_status = Mock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            rest_connector.session = mock_session
            
            result = await rest_connector.make_request(endpoint)
            
            assert result == {"users": [{"id": 1, "name": "Test"}]}
    
    @pytest.mark.asyncio
    async def test_rest_make_request_with_retry(self, rest_connector):
        """Test REST API request with retry on rate limit"""
        endpoint = APIEndpoint(url="/users", method="GET")
        
        # Mock rate limited response then success
        mock_rate_limited = AsyncMock()
        mock_rate_limited.status = 429
        
        mock_success = AsyncMock()
        mock_success.status = 200
        mock_success.headers = {"Content-Type": "application/json"}
        mock_success.json = AsyncMock(return_value={"success": True})
        mock_success.raise_for_status = Mock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            # First call returns 429, second call succeeds
            mock_session.request.return_value.__aenter__.side_effect = [mock_rate_limited, mock_success]
            mock_session_class.return_value = mock_session
            rest_connector.session = mock_session
            
            with patch('asyncio.sleep'):  # Mock sleep to speed up test
                result = await rest_connector.make_request(endpoint)
            
            assert result == {"success": True}
            assert mock_session.request.call_count == 2


class TestGraphQLAPIConnector:
    """Test GraphQL API connector functionality"""
    
    @pytest.fixture
    def graphql_connector(self):
        """Create GraphQL connector for testing"""
        auth_config = AuthConfig(auth_type=AuthType.BEARER, credentials={"token": "test_token"})
        return GraphQLAPIConnector("https://api.example.com/graphql", auth_config)
    
    @pytest.mark.asyncio
    async def test_graphql_query_execution(self, graphql_connector):
        """Test GraphQL query execution"""
        query = "query { users { id name } }"
        
        # Mock successful GraphQL response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {"users": [{"id": "1", "name": "Test User"}]}
        })
        mock_response.raise_for_status = Mock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            graphql_connector.session = mock_session
            
            result = await graphql_connector.execute_query(query)
            
            assert result == {"users": [{"id": "1", "name": "Test User"}]}
    
    @pytest.mark.asyncio
    async def test_graphql_mutation_execution(self, graphql_connector):
        """Test GraphQL mutation execution"""
        mutation = "mutation { createUser(input: {name: \"Test\"}) { id name } }"
        variables = {"name": "Test User"}
        
        # Mock successful mutation response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {"createUser": {"id": "2", "name": "Test User"}}
        })
        mock_response.raise_for_status = Mock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            graphql_connector.session = mock_session
            
            result = await graphql_connector.execute_mutation(mutation, variables)
            
            assert result == {"createUser": {"id": "2", "name": "Test User"}}
    
    @pytest.mark.asyncio
    async def test_graphql_error_handling(self, graphql_connector):
        """Test GraphQL error handling"""
        query = "query { invalidField }"
        
        # Mock GraphQL error response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "errors": [{"message": "Field 'invalidField' not found"}]
        })
        mock_response.raise_for_status = Mock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            graphql_connector.session = mock_session
            
            with pytest.raises(Exception) as exc_info:
                await graphql_connector.execute_query(query)
            
            assert "GraphQL errors" in str(exc_info.value)


class TestSOAPAPIConnector:
    """Test SOAP API connector functionality"""
    
    @pytest.fixture
    def soap_connector(self):
        """Create SOAP connector for testing"""
        auth_config = AuthConfig(
            auth_type=AuthType.BASIC,
            credentials={"username": "testuser", "password": "testpass"}
        )
        return SOAPAPIConnector("https://api.example.com/soap", auth_config)
    
    def test_soap_envelope_building(self, soap_connector):
        """Test SOAP envelope construction"""
        method_name = "GetUser"
        parameters = {"userId": "123", "includeDetails": "true"}
        
        envelope = soap_connector._build_soap_envelope(method_name, parameters)
        
        assert "GetUser" in envelope
        assert "<userId>123</userId>" in envelope
        assert "<includeDetails>true</includeDetails>" in envelope
        assert "soap:Envelope" in envelope
    
    @pytest.mark.asyncio
    async def test_soap_method_call(self, soap_connector):
        """Test SOAP method call"""
        # Mock successful SOAP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""
            <?xml version="1.0"?>
            <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
                <soap:Body>
                    <GetUserResponse>
                        <User>
                            <Id>123</Id>
                            <Name>Test User</Name>
                        </User>
                    </GetUserResponse>
                </soap:Body>
            </soap:Envelope>
        """)
        mock_response.raise_for_status = Mock()
        mock_response.headers = {"Content-Type": "text/xml"}
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            soap_connector.session = mock_session
            
            result = await soap_connector.call_soap_method(
                "GetUser", 
                {"userId": "123"}, 
                soap_action="GetUser"
            )
            
            assert result["status_code"] == 200
            assert "GetUserResponse" in result["soap_response"]


class TestWebhookManager:
    """Test webhook management functionality"""
    
    @pytest.fixture
    def webhook_manager(self):
        """Create webhook manager for testing"""
        return WebhookManager()
    
    def test_webhook_registration(self, webhook_manager):
        """Test webhook registration"""
        webhook_manager.register_webhook(
            "test_webhook",
            "https://myapp.com/webhooks/test",
            secret="secret123"
        )
        
        assert "test_webhook" in webhook_manager.webhooks
        webhook_config = webhook_manager.webhooks["test_webhook"]
        assert webhook_config["url"] == "https://myapp.com/webhooks/test"
        assert webhook_config["secret"] == "secret123"
    
    def test_handler_registration(self, webhook_manager):
        """Test webhook handler registration"""
        async def test_handler(payload, headers):
            return {"processed": True}
        
        webhook_manager.register_handler("test_webhook", test_handler)
        
        assert "test_webhook" in webhook_manager.handlers
        assert webhook_manager.handlers["test_webhook"] == test_handler
    
    @pytest.mark.asyncio
    async def test_webhook_processing(self, webhook_manager):
        """Test webhook payload processing"""
        # Register webhook and handler
        webhook_manager.register_webhook("test_webhook", "https://myapp.com/webhooks/test")
        
        processed_data = []
        
        async def test_handler(payload, headers):
            processed_data.append(payload)
            return True
        
        webhook_manager.register_handler("test_webhook", test_handler)
        
        # Process webhook
        payload = {"event": "user.created", "data": {"id": "123"}}
        headers = {"Content-Type": "application/json"}
        
        result = await webhook_manager.process_webhook("test_webhook", payload, headers)
        
        assert result == True
        assert len(processed_data) == 1
        assert processed_data[0] == payload


class TestAPISchemaDiscovery:
    """Test API schema discovery functionality"""
    
    @pytest.fixture
    def rest_connector(self):
        """Create REST connector for schema discovery"""
        auth_config = AuthConfig(auth_type=AuthType.NONE)
        return RESTAPIConnector("https://api.example.com", auth_config)
    
    @pytest.fixture
    def graphql_connector(self):
        """Create GraphQL connector for schema discovery"""
        auth_config = AuthConfig(auth_type=AuthType.NONE)
        return GraphQLAPIConnector("https://api.example.com/graphql", auth_config)
    
    @pytest.mark.asyncio
    async def test_openapi_schema_discovery(self, rest_connector):
        """Test OpenAPI schema discovery"""
        discovery = APISchemaDiscovery(rest_connector)
        
        # Mock OpenAPI schema response
        openapi_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "summary": "Get users",
                        "parameters": [{"name": "limit", "in": "query"}]
                    },
                    "post": {
                        "summary": "Create user",
                        "parameters": []
                    }
                }
            }
        }
        
        with patch.object(rest_connector, 'make_request', return_value=openapi_schema):
            result = await discovery.discover_rest_schema("https://api.example.com/openapi.json")
        
        assert result["api_type"] == "REST"
        assert result["title"] == "Test API"
        assert len(result["endpoints"]) == 2
        
        # Check endpoints
        endpoints = {ep["path"]: ep for ep in result["endpoints"]}
        assert "/users" in endpoints
        assert endpoints["/users"]["method"] in ["GET", "POST"]
    
    @pytest.mark.asyncio
    async def test_graphql_schema_discovery(self, graphql_connector):
        """Test GraphQL schema introspection"""
        discovery = APISchemaDiscovery(graphql_connector)
        
        # Mock GraphQL introspection response
        introspection_result = {
            "__schema": {
                "types": [
                    {
                        "name": "User",
                        "kind": "OBJECT",
                        "description": "User type",
                        "fields": [
                            {"name": "id", "type": {"name": "ID", "kind": "SCALAR"}},
                            {"name": "name", "type": {"name": "String", "kind": "SCALAR"}}
                        ]
                    },
                    {
                        "name": "__Schema",  # Should be filtered out
                        "kind": "OBJECT",
                        "fields": []
                    }
                ]
            }
        }
        
        with patch.object(graphql_connector, 'execute_query', return_value=introspection_result):
            result = await discovery.discover_graphql_schema()
        
        assert result["api_type"] == "GraphQL"
        assert len(result["types"]) == 1  # __Schema should be filtered out
        
        user_type = result["types"][0]
        assert user_type["name"] == "User"
        assert user_type["kind"] == "OBJECT"
        assert "id" in user_type["fields"]
        assert "name" in user_type["fields"]


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_rest_api_full_workflow(self):
        """Test complete REST API integration workflow"""
        # Setup
        auth_config = AuthConfig(
            auth_type=AuthType.OAUTH2,
            credentials={"client_id": "test", "client_secret": "secret"},
            token_url="https://auth.example.com/token"
        )
        rate_config = RateLimitConfig(requests_per_minute=60, burst_limit=5)
        
        connector = RESTAPIConnector("https://api.example.com", auth_config, rate_config)
        
        # Mock authentication
        auth_response = AsyncMock()
        auth_response.status = 200
        auth_response.json = AsyncMock(return_value={"access_token": "token123", "expires_in": 3600})
        
        # Mock API request
        api_response = AsyncMock()
        api_response.status = 200
        api_response.headers = {"Content-Type": "application/json"}
        api_response.json = AsyncMock(return_value={"data": "success"})
        api_response.raise_for_status = Mock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = auth_response
            mock_session.request.return_value.__aenter__.return_value = api_response
            mock_session_class.return_value = mock_session
            
            async with connector:
                # Test authentication
                assert connector._auth_token == "token123"
                
                # Test API call
                endpoint = APIEndpoint(url="/data", method="GET")
                result = await connector.make_request(endpoint)
                
                assert result == {"data": "success"}
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting in real scenario"""
        auth_config = AuthConfig(auth_type=AuthType.API_KEY, credentials={"api_key": "test"})
        rate_config = RateLimitConfig(requests_per_minute=2, burst_limit=1)  # Very restrictive
        
        connector = RESTAPIConnector("https://api.example.com", auth_config, rate_config)
        
        # Mock successful responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = Mock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with connector:
                endpoint = APIEndpoint(url="/test", method="GET")
                
                # First request should succeed immediately
                start_time = datetime.now()
                result1 = await connector.make_request(endpoint)
                first_duration = (datetime.now() - start_time).total_seconds()
                
                assert result1 == {"success": True}
                assert first_duration < 0.1  # Should be fast
                
                # Second request should be rate limited and take longer
                with patch('asyncio.sleep') as mock_sleep:
                    start_time = datetime.now()
                    result2 = await connector.make_request(endpoint)
                    
                    assert result2 == {"success": True}
                    assert mock_sleep.called  # Should have waited


if __name__ == "__main__":
    pytest.main([__file__, "-v"])