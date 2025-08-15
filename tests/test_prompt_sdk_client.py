"""
Unit tests for ScrollIntel Prompt Management SDK Client.
"""
import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch
import requests_mock

from scrollintel.sdk.prompt_client import PromptClient
from scrollintel.sdk.models import (
    PromptTemplate, PromptVersion, PromptVariable, SearchQuery,
    APIResponse, PaginatedResponse, PromptUsageMetrics, BatchOperation
)
from scrollintel.sdk.exceptions import (
    ScrollIntelSDKError, AuthenticationError, AuthorizationError,
    RateLimitError, ValidationError, NotFoundError, ServerError,
    NetworkError, TimeoutError
)


class TestPromptClient:
    """Test cases for PromptClient."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return PromptClient(
            base_url="https://api.test.com",
            api_key="test-api-key",
            timeout=10
        )
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = PromptClient(
            base_url="https://api.test.com",
            api_key="test-key",
            api_version="v2",
            timeout=30,
            max_retries=5
        )
        
        assert client.base_url == "https://api.test.com"
        assert client.api_key == "test-key"
        assert client.api_version == "v2"
        assert client.timeout == 30
        assert "Bearer test-key" in client.session.headers["Authorization"]
    
    def test_get_url(self, client):
        """Test URL construction."""
        url = client._get_url("/test")
        assert url == "https://api.test.com/api/v1/prompts/test"
    
    def test_handle_response_success(self, client):
        """Test successful response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": {"id": "123"}}
        
        result = client._handle_response(mock_response)
        assert result["success"] is True
        assert result["data"]["id"] == "123"
    
    def test_handle_response_authentication_error(self, client):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Invalid API key"}
        
        with pytest.raises(AuthenticationError) as exc_info:
            client._handle_response(mock_response)
        
        assert "Invalid API key" in str(exc_info.value)
    
    def test_handle_response_rate_limit_error(self, client):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"message": "Rate limit exceeded"}
        mock_response.headers = {"Retry-After": "60"}
        
        with pytest.raises(RateLimitError) as exc_info:
            client._handle_response(mock_response)
        
        assert exc_info.value.retry_after == 60
    
    def test_handle_response_validation_error(self, client):
        """Test validation error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "message": "Validation failed",
            "errors": [{"field": "name", "message": "Required"}]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            client._handle_response(mock_response)
        
        assert len(exc_info.value.errors) == 1
        assert exc_info.value.errors[0]["field"] == "name"
    
    @requests_mock.Mocker()
    def test_create_prompt(self, m, client):
        """Test creating a prompt."""
        m.post(
            "https://api.test.com/api/v1/prompts/",
            json={
                "success": True,
                "data": {"id": "prompt-123"},
                "message": "Created successfully"
            }
        )
        
        variables = [
            PromptVariable(name="name", type="string", required=True)
        ]
        
        prompt_id = client.create_prompt(
            name="Test Prompt",
            content="Hello {{name}}",
            category="test",
            tags=["test"],
            variables=variables,
            description="Test description"
        )
        
        assert prompt_id == "prompt-123"
        
        # Verify request payload
        request = m.last_request
        payload = json.loads(request.text)
        assert payload["name"] == "Test Prompt"
        assert payload["content"] == "Hello {{name}}"
        assert payload["category"] == "test"
        assert payload["tags"] == ["test"]
        assert len(payload["variables"]) == 1
        assert payload["variables"][0]["name"] == "name"
    
    @requests_mock.Mocker()
    def test_get_prompt(self, m, client):
        """Test getting a prompt."""
        m.get(
            "https://api.test.com/api/v1/prompts/prompt-123",
            json={
                "success": True,
                "data": {
                    "id": "prompt-123",
                    "name": "Test Prompt",
                    "content": "Hello {{name}}",
                    "category": "test",
                    "tags": ["test"],
                    "variables": [
                        {
                            "name": "name",
                            "type": "string",
                            "required": True,
                            "description": "User name"
                        }
                    ],
                    "description": "Test description",
                    "is_active": True,
                    "created_by": "user-123",
                    "created_at": "2024-01-01T12:00:00Z",
                    "updated_at": "2024-01-01T12:00:00Z"
                }
            }
        )
        
        prompt = client.get_prompt("prompt-123")
        
        assert isinstance(prompt, PromptTemplate)
        assert prompt.id == "prompt-123"
        assert prompt.name == "Test Prompt"
        assert prompt.content == "Hello {{name}}"
        assert prompt.category == "test"
        assert len(prompt.variables) == 1
        assert prompt.variables[0].name == "name"
    
    @requests_mock.Mocker()
    def test_update_prompt(self, m, client):
        """Test updating a prompt."""
        m.put(
            "https://api.test.com/api/v1/prompts/prompt-123",
            json={
                "success": True,
                "data": {
                    "id": "version-456",
                    "prompt_id": "prompt-123",
                    "version": "1.0.1",
                    "content": "Hello {{name}}, welcome!",
                    "changes": "Added welcome message",
                    "variables": [],
                    "tags": ["test"],
                    "created_by": "user-123",
                    "created_at": "2024-01-01T12:30:00Z"
                }
            }
        )
        
        new_version = client.update_prompt(
            prompt_id="prompt-123",
            content="Hello {{name}}, welcome!",
            changes_description="Added welcome message"
        )
        
        assert isinstance(new_version, PromptVersion)
        assert new_version.id == "version-456"
        assert new_version.prompt_id == "prompt-123"
        assert new_version.version == "1.0.1"
        assert new_version.changes == "Added welcome message"
    
    @requests_mock.Mocker()
    def test_delete_prompt(self, m, client):
        """Test deleting a prompt."""
        m.delete(
            "https://api.test.com/api/v1/prompts/prompt-123",
            json={
                "success": True,
                "message": "Deleted successfully"
            }
        )
        
        result = client.delete_prompt("prompt-123")
        assert result is True
    
    @requests_mock.Mocker()
    def test_search_prompts(self, m, client):
        """Test searching prompts."""
        m.post(
            "https://api.test.com/api/v1/prompts/search",
            json={
                "success": True,
                "data": {
                    "items": [
                        {
                            "id": "prompt-1",
                            "name": "Prompt 1",
                            "content": "Content 1",
                            "category": "test",
                            "tags": ["test"],
                            "variables": [],
                            "description": None,
                            "is_active": True,
                            "created_by": "user-123",
                            "created_at": "2024-01-01T12:00:00Z",
                            "updated_at": "2024-01-01T12:00:00Z"
                        }
                    ],
                    "total": 1,
                    "page": 1,
                    "page_size": 50,
                    "has_next": False,
                    "has_previous": False
                }
            }
        )
        
        query = SearchQuery(
            text="test",
            category="test",
            tags=["test"],
            limit=50
        )
        
        result = client.search_prompts(query)
        
        assert isinstance(result, PaginatedResponse)
        assert len(result.items) == 1
        assert isinstance(result.items[0], PromptTemplate)
        assert result.items[0].name == "Prompt 1"
        assert result.total == 1
    
    @requests_mock.Mocker()
    def test_get_prompt_history(self, m, client):
        """Test getting prompt history."""
        m.get(
            "https://api.test.com/api/v1/prompts/prompt-123/history",
            json={
                "success": True,
                "data": [
                    {
                        "id": "version-2",
                        "prompt_id": "prompt-123",
                        "version": "1.0.1",
                        "content": "Updated content",
                        "changes": "Updated",
                        "variables": [],
                        "tags": ["test"],
                        "created_by": "user-123",
                        "created_at": "2024-01-01T12:30:00Z"
                    },
                    {
                        "id": "version-1",
                        "prompt_id": "prompt-123",
                        "version": "1.0.0",
                        "content": "Original content",
                        "changes": "Initial version",
                        "variables": [],
                        "tags": ["test"],
                        "created_by": "user-123",
                        "created_at": "2024-01-01T12:00:00Z"
                    }
                ]
            }
        )
        
        history = client.get_prompt_history("prompt-123")
        
        assert len(history) == 2
        assert all(isinstance(v, PromptVersion) for v in history)
        assert history[0].version == "1.0.1"
        assert history[1].version == "1.0.0"
    
    @requests_mock.Mocker()
    def test_get_prompt_metrics(self, m, client):
        """Test getting prompt metrics."""
        m.get(
            "https://api.test.com/api/v1/prompts/prompt-123/metrics",
            json={
                "success": True,
                "data": {
                    "prompt_id": "prompt-123",
                    "total_uses": 100,
                    "unique_users": 25,
                    "avg_response_time": 0.5,
                    "success_rate": 0.95,
                    "last_used": "2024-01-01T11:30:00Z"
                }
            }
        )
        
        metrics = client.get_prompt_metrics("prompt-123")
        
        assert isinstance(metrics, PromptUsageMetrics)
        assert metrics.prompt_id == "prompt-123"
        assert metrics.total_uses == 100
        assert metrics.unique_users == 25
        assert metrics.success_rate == 0.95
    
    @requests_mock.Mocker()
    def test_batch_operations(self, m, client):
        """Test batch operations."""
        m.post(
            "https://api.test.com/api/v1/prompts/batch",
            json={
                "success": True,
                "data": {
                    "results": [
                        {
                            "operation": 0,
                            "type": "update",
                            "prompt_id": "prompt-1",
                            "version": "1.0.1"
                        }
                    ],
                    "errors": []
                }
            }
        )
        
        operations = [
            BatchOperation(
                type="update",
                prompt_id="prompt-1",
                data={"name": "Updated Name"}
            )
        ]
        
        result = client.batch_operations(operations)
        
        assert "results" in result
        assert "errors" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["type"] == "update"
    
    @requests_mock.Mocker()
    def test_substitute_variables(self, m, client):
        """Test variable substitution."""
        m.post(
            "https://api.test.com/api/prompts/substitute",
            json={"result": "Hello John, how are you?"}
        )
        
        result = client.substitute_variables(
            content="Hello {{name}}, how are you?",
            variables={"name": "John"}
        )
        
        assert result == "Hello John, how are you?"
    
    @requests_mock.Mocker()
    def test_validate_prompt(self, m, client):
        """Test prompt validation."""
        m.post(
            "https://api.test.com/api/prompts/prompt-123/validate",
            json={
                "valid": False,
                "errors": ["Variable 'name' is required but not defined"]
            }
        )
        
        result = client.validate_prompt("prompt-123")
        
        assert result["valid"] is False
        assert len(result["errors"]) == 1
    
    @requests_mock.Mocker()
    def test_export_prompts(self, m, client):
        """Test exporting prompts."""
        m.post(
            "https://api.test.com/api/prompts/export",
            content=b'{"prompts": []}',
            headers={"Content-Type": "application/json"}
        )
        
        result = client.export_prompts(
            prompt_ids=["prompt-1", "prompt-2"],
            format="json"
        )
        
        assert isinstance(result, bytes)
        assert b'{"prompts": []}' == result
    
    @requests_mock.Mocker()
    def test_import_prompts(self, m, client):
        """Test importing prompts."""
        m.post(
            "https://api.test.com/api/prompts/import",
            json={
                "imported": 2,
                "updated": 0,
                "skipped": 0,
                "errors": []
            }
        )
        
        result = client.import_prompts(
            file_data='{"prompts": []}',
            format="json",
            overwrite=False
        )
        
        assert result["imported"] == 2
        assert result["errors"] == []
    
    @requests_mock.Mocker()
    def test_register_webhook(self, m, client):
        """Test registering a webhook."""
        m.post(
            "https://api.test.com/api/v1/prompts/webhooks",
            json={
                "success": True,
                "data": {"webhook_id": "webhook-123"}
            }
        )
        
        webhook_id = client.register_webhook(
            url="https://example.com/webhook",
            events=["prompt.created"],
            secret="test-secret"
        )
        
        assert webhook_id == "webhook-123"
    
    @requests_mock.Mocker()
    def test_list_webhooks(self, m, client):
        """Test listing webhooks."""
        m.get(
            "https://api.test.com/api/v1/prompts/webhooks",
            json={
                "success": True,
                "data": [
                    {
                        "id": "webhook-123",
                        "url": "https://example.com/webhook",
                        "events": ["prompt.created"],
                        "active": True
                    }
                ]
            }
        )
        
        webhooks = client.list_webhooks()
        
        assert len(webhooks) == 1
        assert webhooks[0]["id"] == "webhook-123"
    
    @requests_mock.Mocker()
    def test_update_webhook(self, m, client):
        """Test updating a webhook."""
        m.put(
            "https://api.test.com/api/v1/prompts/webhooks/webhook-123",
            json={"success": True}
        )
        
        result = client.update_webhook(
            webhook_id="webhook-123",
            url="https://example.com/new-webhook",
            active=False
        )
        
        assert result is True
    
    @requests_mock.Mocker()
    def test_delete_webhook(self, m, client):
        """Test deleting a webhook."""
        m.delete(
            "https://api.test.com/api/v1/prompts/webhooks/webhook-123",
            json={"success": True}
        )
        
        result = client.delete_webhook("webhook-123")
        assert result is True
    
    @requests_mock.Mocker()
    def test_test_webhook(self, m, client):
        """Test testing a webhook."""
        m.post(
            "https://api.test.com/api/v1/prompts/webhooks/webhook-123/test",
            json={
                "success": True,
                "data": {
                    "success": True,
                    "status": "delivered",
                    "response_status": 200
                }
            }
        )
        
        result = client.test_webhook("webhook-123")
        
        assert result["success"] is True
        assert result["status"] == "delivered"
    
    @requests_mock.Mocker()
    def test_get_usage_summary(self, m, client):
        """Test getting usage summary."""
        m.get(
            "https://api.test.com/api/v1/prompts/usage/summary",
            json={
                "success": True,
                "data": {
                    "total_requests": 1000,
                    "total_tokens": 50000,
                    "total_errors": 10,
                    "avg_response_time": 0.25,
                    "error_rate": 0.01
                }
            }
        )
        
        summary = client.get_usage_summary()
        
        assert summary["total_requests"] == 1000
        assert summary["total_tokens"] == 50000
        assert summary["error_rate"] == 0.01
    
    @requests_mock.Mocker()
    def test_health_check(self, m, client):
        """Test health check."""
        m.get(
            "https://api.test.com/health",
            json={
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        )
        
        result = client.health_check()
        
        assert result["status"] == "healthy"
    
    def test_context_manager(self, client):
        """Test context manager support."""
        with client as c:
            assert c is client
        
        # In real implementation, this would verify session is closed
    
    @requests_mock.Mocker()
    def test_network_error_handling(self, m, client):
        """Test network error handling."""
        m.get(
            "https://api.test.com/api/v1/prompts/test",
            exc=requests_mock.exceptions.ConnectTimeout
        )
        
        with pytest.raises(NetworkError):
            client._make_request("GET", "/test")
    
    @requests_mock.Mocker()
    def test_timeout_error_handling(self, m, client):
        """Test timeout error handling."""
        m.get(
            "https://api.test.com/api/v1/prompts/test",
            exc=requests_mock.exceptions.ReadTimeout
        )
        
        with pytest.raises(TimeoutError):
            client._make_request("GET", "/test")
    
    def test_list_prompts_with_pagination(self, client):
        """Test list prompts with pagination parameters."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://api.test.com/api/v1/prompts/",
                json={
                    "success": True,
                    "data": {
                        "items": [],
                        "total": 0,
                        "page": 2,
                        "page_size": 25,
                        "has_next": False,
                        "has_previous": True
                    }
                }
            )
            
            result = client.list_prompts(
                page=2,
                page_size=25,
                category="test",
                tags=["tag1", "tag2"]
            )
            
            assert isinstance(result, PaginatedResponse)
            assert result.page == 2
            assert result.page_size == 25
            assert result.has_previous is True
            
            # Verify query parameters
            request = m.last_request
            assert "page=2" in request.url
            assert "page_size=25" in request.url
            assert "category=test" in request.url
            assert "tags=tag1%2Ctag2" in request.url


class TestPromptClientErrorScenarios:
    """Test error scenarios for PromptClient."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return PromptClient(
            base_url="https://api.test.com",
            api_key="test-api-key"
        )
    
    @requests_mock.Mocker()
    def test_server_error_handling(self, m, client):
        """Test server error handling."""
        m.post(
            "https://api.test.com/api/v1/prompts/",
            status_code=500,
            json={"message": "Internal server error"}
        )
        
        with pytest.raises(ServerError) as exc_info:
            client.create_prompt(
                name="Test",
                content="Test",
                category="test"
            )
        
        assert exc_info.value.status_code == 500
    
    @requests_mock.Mocker()
    def test_not_found_error_handling(self, m, client):
        """Test not found error handling."""
        m.get(
            "https://api.test.com/api/v1/prompts/nonexistent",
            status_code=404,
            json={"message": "Prompt not found"}
        )
        
        with pytest.raises(NotFoundError):
            client.get_prompt("nonexistent")
    
    @requests_mock.Mocker()
    def test_authorization_error_handling(self, m, client):
        """Test authorization error handling."""
        m.post(
            "https://api.test.com/api/v1/prompts/",
            status_code=403,
            json={"message": "Insufficient permissions"}
        )
        
        with pytest.raises(AuthorizationError):
            client.create_prompt(
                name="Test",
                content="Test",
                category="test"
            )
    
    @requests_mock.Mocker()
    def test_invalid_json_response(self, m, client):
        """Test handling of invalid JSON responses."""
        m.get(
            "https://api.test.com/api/v1/prompts/test",
            text="Invalid JSON response"
        )
        
        with pytest.raises(ScrollIntelSDKError):
            client.get_prompt("test")
    
    @requests_mock.Mocker()
    def test_unexpected_status_code(self, m, client):
        """Test handling of unexpected status codes."""
        m.get(
            "https://api.test.com/api/v1/prompts/test",
            status_code=418,  # I'm a teapot
            json={"message": "Unexpected error"}
        )
        
        with pytest.raises(ScrollIntelSDKError) as exc_info:
            client.get_prompt("test")
        
        assert exc_info.value.status_code == 418


if __name__ == "__main__":
    pytest.main([__file__])