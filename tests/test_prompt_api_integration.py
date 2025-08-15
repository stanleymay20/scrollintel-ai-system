"""
Integration tests for Advanced Prompt Management API.
"""
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from scrollintel.api.routes.prompt_api_v1 import router
from scrollintel.models.database_utils import get_db
from scrollintel.core.prompt_manager import PromptManager
from scrollintel.core.webhook_system import webhook_manager
from scrollintel.sdk.prompt_client import PromptClient
from scrollintel.sdk.models import PromptVariable, SearchQuery, WebhookConfig
from scrollintel.sdk.exceptions import (
    ScrollIntelSDKError, RateLimitError, ValidationError, NotFoundError
)


class TestPromptAPIIntegration:
    """Integration tests for Prompt Management API."""
    
    @pytest.fixture
    def client(self, test_db: Session):
        """Create test client."""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        # Override database dependency
        def override_get_db():
            try:
                yield test_db
            finally:
                pass
        
        app.dependency_overrides[get_db] = override_get_db
        
        return TestClient(app)
    
    @pytest.fixture
    def sdk_client(self):
        """Create SDK client for testing."""
        return PromptClient(
            base_url="http://testserver",
            api_key="test-api-key",
            timeout=10
        )
    
    def test_create_prompt_api(self, client: TestClient):
        """Test creating a prompt via API."""
        prompt_data = {
            "name": "Test Prompt",
            "content": "Hello {{name}}, how are you?",
            "category": "greeting",
            "tags": ["test", "greeting"],
            "variables": [
                {
                    "name": "name",
                    "type": "string",
                    "required": True,
                    "description": "User's name"
                }
            ],
            "description": "A test greeting prompt"
        }
        
        response = client.post("/api/v1/prompts/", json=prompt_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "id" in data["data"]
        assert data["message"] == "Prompt created successfully"
        assert "timestamp" in data
        assert "request_id" in data
    
    def test_get_prompt_api(self, client: TestClient, test_db: Session):
        """Test getting a prompt via API."""
        # Create a prompt first
        prompt_manager = PromptManager(test_db)
        prompt_id = prompt_manager.create_prompt(
            name="Test Prompt",
            content="Hello {{name}}",
            category="test",
            created_by="test_user"
        )
        
        response = client.get(f"/api/v1/prompts/{prompt_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == prompt_id
        assert data["data"]["name"] == "Test Prompt"
    
    def test_update_prompt_api(self, client: TestClient, test_db: Session):
        """Test updating a prompt via API."""
        # Create a prompt first
        prompt_manager = PromptManager(test_db)
        prompt_id = prompt_manager.create_prompt(
            name="Test Prompt",
            content="Hello {{name}}",
            category="test",
            created_by="test_user"
        )
        
        update_data = {
            "name": "Updated Prompt",
            "content": "Hello {{name}}, welcome!",
            "changes_description": "Added welcome message"
        }
        
        response = client.put(f"/api/v1/prompts/{prompt_id}", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"]["version"] == "1.0.1"
        assert data["data"]["changes"] == "Added welcome message"
    
    def test_search_prompts_api(self, client: TestClient, test_db: Session):
        """Test searching prompts via API."""
        # Create test prompts
        prompt_manager = PromptManager(test_db)
        for i in range(5):
            prompt_manager.create_prompt(
                name=f"Test Prompt {i}",
                content=f"Content {i}",
                category="test",
                created_by="test_user",
                tags=["test", f"tag{i}"]
            )
        
        search_data = {
            "text": "Test",
            "category": "test",
            "tags": ["test"],
            "limit": 3,
            "offset": 0
        }
        
        response = client.post("/api/v1/prompts/search", json=search_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["items"]) <= 3
        assert data["data"]["total"] >= 3
    
    def test_prompt_history_api(self, client: TestClient, test_db: Session):
        """Test getting prompt history via API."""
        # Create and update a prompt
        prompt_manager = PromptManager(test_db)
        prompt_id = prompt_manager.create_prompt(
            name="Test Prompt",
            content="Hello {{name}}",
            category="test",
            created_by="test_user"
        )
        
        # Update it a few times
        from scrollintel.core.prompt_manager import PromptChanges
        for i in range(3):
            changes = PromptChanges(
                content=f"Updated content {i}",
                changes_description=f"Update {i}"
            )
            prompt_manager.update_prompt(prompt_id, changes, "test_user")
        
        response = client.get(f"/api/v1/prompts/{prompt_id}/history")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 4  # Initial + 3 updates
    
    def test_batch_operations_api(self, client: TestClient, test_db: Session):
        """Test batch operations via API."""
        # Create test prompts
        prompt_manager = PromptManager(test_db)
        prompt_ids = []
        for i in range(3):
            prompt_id = prompt_manager.create_prompt(
                name=f"Test Prompt {i}",
                content=f"Content {i}",
                category="test",
                created_by="test_user"
            )
            prompt_ids.append(prompt_id)
        
        batch_operations = [
            {
                "type": "update",
                "prompt_id": prompt_ids[0],
                "changes": {
                    "name": "Updated Prompt 0",
                    "changes_description": "Batch update"
                }
            },
            {
                "type": "delete",
                "prompt_id": prompt_ids[1]
            }
        ]
        
        response = client.post("/api/v1/prompts/batch", json=batch_operations)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["results"]) == 2
        assert len(data["data"]["errors"]) == 0
    
    def test_webhook_management_api(self, client: TestClient):
        """Test webhook management via API."""
        # Register webhook
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["prompt.created", "prompt.updated"],
            "secret": "test-secret",
            "active": True
        }
        
        response = client.post("/api/v1/prompts/webhooks", json=webhook_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        webhook_id = data["data"]["webhook_id"]
        
        # List webhooks
        response = client.get("/api/v1/prompts/webhooks")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1
        
        # Update webhook
        update_data = {
            "url": "https://example.com/new-webhook",
            "events": ["prompt.created"],
            "secret": "new-secret",
            "active": False
        }
        
        response = client.put(f"/api/v1/prompts/webhooks/{webhook_id}", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
        # Test webhook
        response = client.post(f"/api/v1/prompts/webhooks/{webhook_id}/test")
        assert response.status_code == 200
        
        # Delete webhook
        response = client.delete(f"/api/v1/prompts/webhooks/{webhook_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_usage_summary_api(self, client: TestClient):
        """Test usage summary API."""
        response = client.get("/api/v1/prompts/usage/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "total_requests" in data["data"]
        assert "total_tokens" in data["data"]
        assert "total_errors" in data["data"]


class TestPromptSDKIntegration:
    """Integration tests for Prompt Management SDK."""
    
    @pytest.fixture
    def mock_client(self, monkeypatch):
        """Create mock SDK client."""
        import requests_mock
        
        with requests_mock.Mocker() as m:
            # Mock successful responses
            m.post(
                "http://testserver/api/v1/prompts/",
                json={
                    "success": True,
                    "data": {"id": "test-prompt-id"},
                    "message": "Prompt created successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            m.get(
                "http://testserver/api/v1/prompts/test-prompt-id",
                json={
                    "success": True,
                    "data": {
                        "id": "test-prompt-id",
                        "name": "Test Prompt",
                        "content": "Hello {{name}}",
                        "category": "test",
                        "tags": ["test"],
                        "variables": [],
                        "is_active": True,
                        "created_by": "test_user",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    },
                    "message": "Prompt retrieved successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            yield PromptClient(
                base_url="http://testserver",
                api_key="test-api-key"
            )
    
    def test_sdk_create_prompt(self, mock_client: PromptClient):
        """Test creating a prompt via SDK."""
        variables = [
            PromptVariable(
                name="name",
                type="string",
                required=True,
                description="User's name"
            )
        ]
        
        prompt_id = mock_client.create_prompt(
            name="Test Prompt",
            content="Hello {{name}}",
            category="test",
            tags=["test"],
            variables=variables,
            description="Test prompt"
        )
        
        assert prompt_id == "test-prompt-id"
    
    def test_sdk_get_prompt(self, mock_client: PromptClient):
        """Test getting a prompt via SDK."""
        prompt = mock_client.get_prompt("test-prompt-id")
        
        assert prompt.id == "test-prompt-id"
        assert prompt.name == "Test Prompt"
        assert prompt.content == "Hello {{name}}"
        assert prompt.category == "test"
    
    def test_sdk_search_prompts(self, mock_client: PromptClient, monkeypatch):
        """Test searching prompts via SDK."""
        import requests_mock
        
        with requests_mock.Mocker() as m:
            m.post(
                "http://testserver/api/v1/prompts/search",
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
                                "is_active": True,
                                "created_by": "test_user",
                                "created_at": datetime.utcnow().isoformat(),
                                "updated_at": datetime.utcnow().isoformat()
                            }
                        ],
                        "total": 1,
                        "page": 1,
                        "page_size": 50,
                        "has_next": False,
                        "has_previous": False
                    },
                    "message": "Search completed successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            query = SearchQuery(
                text="test",
                category="test",
                limit=50
            )
            
            result = mock_client.search_prompts(query)
            
            assert len(result.items) == 1
            assert result.total == 1
            assert result.items[0].name == "Prompt 1"
    
    def test_sdk_webhook_management(self, mock_client: PromptClient, monkeypatch):
        """Test webhook management via SDK."""
        import requests_mock
        
        with requests_mock.Mocker() as m:
            # Mock webhook registration
            m.post(
                "http://testserver/api/v1/prompts/webhooks",
                json={
                    "success": True,
                    "data": {"webhook_id": "webhook-123"},
                    "message": "Webhook registered successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Mock webhook listing
            m.get(
                "http://testserver/api/v1/prompts/webhooks",
                json={
                    "success": True,
                    "data": [
                        {
                            "id": "webhook-123",
                            "url": "https://example.com/webhook",
                            "events": ["prompt.created"],
                            "active": True
                        }
                    ],
                    "message": "Webhooks retrieved successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Register webhook
            webhook_id = mock_client.register_webhook(
                url="https://example.com/webhook",
                events=["prompt.created"],
                secret="test-secret"
            )
            
            assert webhook_id == "webhook-123"
            
            # List webhooks
            webhooks = mock_client.list_webhooks()
            assert len(webhooks) == 1
            assert webhooks[0]["id"] == "webhook-123"
    
    def test_sdk_error_handling(self, monkeypatch):
        """Test SDK error handling."""
        import requests_mock
        
        with requests_mock.Mocker() as m:
            # Mock rate limit error
            m.post(
                "http://testserver/api/v1/prompts/",
                status_code=429,
                json={
                    "message": "Rate limit exceeded",
                    "detail": "Too many requests"
                },
                headers={"Retry-After": "60"}
            )
            
            client = PromptClient(
                base_url="http://testserver",
                api_key="test-api-key"
            )
            
            with pytest.raises(RateLimitError) as exc_info:
                client.create_prompt(
                    name="Test",
                    content="Test",
                    category="test"
                )
            
            assert exc_info.value.retry_after == 60
    
    def test_sdk_context_manager(self, mock_client: PromptClient):
        """Test SDK context manager support."""
        with mock_client as client:
            # Should work normally
            assert client is not None
        
        # Client should be closed after context
        # (In real implementation, this would close the session)


class TestAPIVersioning:
    """Test API versioning and backward compatibility."""
    
    def test_api_version_headers(self, client: TestClient):
        """Test API version headers."""
        response = client.get("/api/v1/prompts/", headers={"Accept": "application/json"})
        
        # Should include version information in response
        if response.status_code == 200:
            data = response.json()
            assert "version" in data
    
    def test_backward_compatibility(self, client: TestClient):
        """Test backward compatibility with older API versions."""
        # Test that old API endpoints still work
        # This would be more comprehensive in a real implementation
        pass


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_headers(self, client: TestClient):
        """Test rate limit headers in responses."""
        response = client.get("/api/v1/prompts/")
        
        # Should include rate limit information
        if response.status_code == 200:
            data = response.json()
            if "rate_limit" in data:
                assert isinstance(data["rate_limit"], dict)
    
    def test_rate_limit_exceeded(self, client: TestClient, monkeypatch):
        """Test rate limit exceeded scenario."""
        # This would require mocking the rate limiter to simulate exceeded limits
        pass


class TestWebhookDelivery:
    """Test webhook delivery functionality."""
    
    @pytest.mark.asyncio
    async def test_webhook_event_triggering(self, test_db: Session):
        """Test webhook event triggering."""
        # Start webhook manager
        webhook_manager.db = test_db
        await webhook_manager.start()
        
        try:
            # Register a test webhook
            webhook_id = await webhook_manager.register_endpoint(
                user_id="test_user",
                name="Test Webhook",
                url="https://httpbin.org/post",
                events=["prompt.created"]
            )
            
            # Trigger an event
            from scrollintel.core.webhook_system import trigger_webhook_event, WebhookEventType
            await trigger_webhook_event(
                event_type=WebhookEventType.PROMPT_CREATED,
                resource_type="prompt",
                resource_id="test-prompt-id",
                action="create",
                user_id="test_user",
                data={"name": "Test Prompt"}
            )
            
            # Wait a bit for delivery
            await asyncio.sleep(1)
            
            # Check delivery status
            deliveries = await webhook_manager.get_delivery_status(
                endpoint_id=webhook_id,
                user_id="test_user"
            )
            
            assert len(deliveries) >= 1
            
        finally:
            await webhook_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__])