"""
Integration tests for ScrollIntel Vault API routes.
Tests the complete vault workflow including authentication and database operations.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from scrollintel.api.gateway import app
from scrollintel.models.database import User, VaultInsight, VaultAccessLog
from scrollintel.models.database import get_db
from scrollintel.security.auth import create_access_token
from scrollintel.core.interfaces import UserRole
from scrollintel.engines.vault_engine import ScrollVaultEngine
from scrollintel.core.registry import EngineRegistry


class TestVaultIntegration:
    """Integration test suite for vault functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def test_user(self, db_session):
        """Create test user."""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            full_name="Test User",
            role=UserRole.ANALYST,
            is_active=True,
            is_verified=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user
    
    @pytest.fixture
    def admin_user(self, db_session):
        """Create admin user."""
        user = User(
            email="admin@example.com",
            hashed_password="hashed_password",
            full_name="Admin User",
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user
    
    @pytest.fixture
    def auth_headers(self, test_user):
        """Create authentication headers."""
        token = create_access_token(
            data={"sub": test_user.email, "user_id": str(test_user.id)}
        )
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def admin_headers(self, admin_user):
        """Create admin authentication headers."""
        token = create_access_token(
            data={"sub": admin_user.email, "user_id": str(admin_user.id)}
        )
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    async def vault_engine(self):
        """Create and initialize vault engine."""
        engine = ScrollVaultEngine()
        await engine.initialize()
        
        # Register engine
        registry = EngineRegistry()
        registry.register_engine(engine)
        
        return engine
    
    @pytest.fixture
    def sample_insight_data(self):
        """Sample insight data for testing."""
        return {
            "title": "Test ML Model Analysis",
            "content": {
                "model_performance": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.89,
                    "f1_score": 0.90
                },
                "feature_importance": {
                    "age": 0.25,
                    "income": 0.30,
                    "credit_score": 0.45
                },
                "recommendations": [
                    "Consider adding more demographic features",
                    "Investigate potential bias in credit score feature",
                    "Validate model performance on recent data"
                ]
            },
            "insight_type": "analysis_result",
            "access_level": "internal",
            "retention_policy": "medium_term",
            "organization_id": "test_org",
            "tags": ["ml", "analysis", "credit_model"],
            "metadata": {
                "model_id": "credit_model_v2",
                "dataset": "customer_credit_data",
                "training_date": "2024-01-15",
                "data_scientist": "john.doe@company.com"
            }
        }
    
    def test_store_insight_success(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test successful insight storage via API."""
        response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "insight_id" in data
        assert data["version"] == 1
        assert data["encrypted"] is True
        assert "created_at" in data
    
    def test_store_insight_unauthorized(self, client, sample_insight_data):
        """Test insight storage without authentication."""
        response = client.post(
            "/vault/insights",
            json=sample_insight_data
        )
        
        assert response.status_code == 401
    
    def test_store_insight_invalid_data(self, client, auth_headers):
        """Test insight storage with invalid data."""
        invalid_data = {
            "title": "",  # Empty title
            "content": {},
            "insight_type": "invalid_type",  # Invalid type
            "access_level": "internal",
            "retention_policy": "medium_term"
        }
        
        response = client.post(
            "/vault/insights",
            json=invalid_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_retrieve_insight_success(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test successful insight retrieval via API."""
        # Store insight first
        store_response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        insight_id = store_response.json()["insight_id"]
        
        # Retrieve insight
        response = client.get(
            f"/vault/insights/{insight_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "insight" in data
        
        insight = data["insight"]
        assert insight["id"] == insight_id
        assert insight["title"] == sample_insight_data["title"]
        assert insight["type"] == sample_insight_data["insight_type"]
        assert insight["content"] == sample_insight_data["content"]
        assert insight["access_count"] == 1
    
    def test_retrieve_nonexistent_insight(self, client, auth_headers):
        """Test retrieval of non-existent insight."""
        fake_id = str(uuid4())
        response = client.get(
            f"/vault/insights/{fake_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_update_insight_success(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test successful insight update via API."""
        # Store insight first
        store_response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        insight_id = store_response.json()["insight_id"]
        
        # Update insight
        update_data = {
            "title": "Updated ML Model Analysis",
            "content": {
                "model_performance": {
                    "accuracy": 0.97,  # Improved accuracy
                    "precision": 0.94,
                    "recall": 0.91,
                    "f1_score": 0.92
                },
                "update_notes": "Model retrained with additional data"
            },
            "tags": ["ml", "analysis", "credit_model", "updated"]
        }
        
        response = client.put(
            f"/vault/insights/{insight_id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["version"] == 2
        assert data["parent_id"] == insight_id
        assert "updated_at" in data
    
    def test_delete_insight_success(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test successful insight deletion via API."""
        # Store insight first
        store_response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        insight_id = store_response.json()["insight_id"]
        
        # Delete insight
        response = client.delete(
            f"/vault/insights/{insight_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["insight_id"] == insight_id
        assert "deleted_at" in data
    
    def test_search_insights_success(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test successful insight search via API."""
        # Store multiple insights
        insights = []
        for i in range(3):
            data = sample_insight_data.copy()
            data["title"] = f"ML Analysis {i}"
            data["tags"] = ["ml", f"test{i}"]
            
            response = client.post(
                "/vault/insights",
                json=data,
                headers=auth_headers
            )
            insights.append(response.json()["insight_id"])
        
        # Search insights
        search_data = {
            "query": "ML Analysis",
            "filters": {},
            "limit": 10,
            "offset": 0
        }
        
        response = client.post(
            "/vault/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 3
        assert data["total_count"] == 3
        assert data["query"] == "ML Analysis"
    
    def test_search_insights_with_filters(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test insight search with filters via API."""
        # Store insights with different types
        data1 = sample_insight_data.copy()
        data1["insight_type"] = "analysis_result"
        data1["title"] = "Analysis Result"
        
        data2 = sample_insight_data.copy()
        data2["insight_type"] = "report"
        data2["title"] = "Report Document"
        
        for data in [data1, data2]:
            client.post(
                "/vault/insights",
                json=data,
                headers=auth_headers
            )
        
        # Search with type filter
        search_data = {
            "query": "",
            "filters": {},
            "insight_types": ["analysis_result"],
            "limit": 10,
            "offset": 0
        }
        
        response = client.post(
            "/vault/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["type"] == "analysis_result"
    
    def test_get_insight_history_success(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test insight history retrieval via API."""
        # Store insight
        store_response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        insight_id = store_response.json()["insight_id"]
        
        # Update insight to create version history
        update_data = {"title": "Updated Analysis"}
        client.put(
            f"/vault/insights/{insight_id}",
            json=update_data,
            headers=auth_headers
        )
        
        # Get history
        response = client.get(
            f"/vault/insights/{insight_id}/history",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["insight_id"] == insight_id
        assert len(data["versions"]) == 2
        assert data["total_versions"] == 2
    
    def test_get_access_audit_success(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test access audit retrieval via API."""
        # Store and retrieve insight to generate audit logs
        store_response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        insight_id = store_response.json()["insight_id"]
        
        client.get(
            f"/vault/insights/{insight_id}",
            headers=auth_headers
        )
        
        # Get audit logs
        response = client.get(
            "/vault/audit",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["audit_logs"]) >= 2  # store + retrieve
        assert data["total_count"] >= 2
    
    def test_get_access_audit_with_filters(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test access audit retrieval with filters via API."""
        # Store insight
        store_response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        insight_id = store_response.json()["insight_id"]
        
        # Get audit logs for specific insight
        response = client.get(
            f"/vault/audit?insight_id={insight_id}&action=write",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["audit_logs"]) == 1
        assert data["audit_logs"][0]["action"] == "write"
        assert data["audit_logs"][0]["insight_id"] == insight_id
    
    def test_cleanup_expired_insights_admin(self, client, admin_headers, sample_insight_data, vault_engine):
        """Test cleanup of expired insights (admin only)."""
        response = client.post(
            "/vault/cleanup",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "cleaned_up_count" in data
        assert "cleanup_timestamp" in data
    
    def test_cleanup_expired_insights_non_admin(self, client, auth_headers):
        """Test cleanup attempt by non-admin user."""
        response = client.post(
            "/vault/cleanup",
            headers=auth_headers
        )
        
        assert response.status_code == 403  # Forbidden
    
    def test_get_vault_stats_success(self, client, auth_headers, vault_engine):
        """Test vault statistics retrieval via API."""
        response = client.get(
            "/vault/stats",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "vault_status" in data
        assert "recent_activity" in data
        assert "timestamp" in data
        
        vault_status = data["vault_status"]
        assert vault_status["engine_id"] == "scroll-vault-engine"
        assert "healthy" in vault_status
    
    def test_pagination_in_search(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test pagination in search results."""
        # Store multiple insights
        for i in range(5):
            data = sample_insight_data.copy()
            data["title"] = f"Paginated Insight {i}"
            
            client.post(
                "/vault/insights",
                json=data,
                headers=auth_headers
            )
        
        # Search with pagination
        search_data = {
            "query": "Paginated",
            "limit": 2,
            "offset": 0
        }
        
        response = client.post(
            "/vault/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["total_count"] == 5
        assert data["offset"] == 0
        assert data["limit"] == 2
        
        # Get next page
        search_data["offset"] = 2
        response = client.post(
            "/vault/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 2
        assert data["offset"] == 2
    
    def test_concurrent_access(self, client, auth_headers, sample_insight_data, vault_engine):
        """Test concurrent access to insights."""
        # Store insight
        store_response = client.post(
            "/vault/insights",
            json=sample_insight_data,
            headers=auth_headers
        )
        insight_id = store_response.json()["insight_id"]
        
        # Simulate concurrent reads
        responses = []
        for _ in range(5):
            response = client.get(
                f"/vault/insights/{insight_id}",
                headers=auth_headers
            )
            responses.append(response)
        
        # Verify all requests succeeded
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_error_handling(self, client, auth_headers, vault_engine):
        """Test error handling in vault operations."""
        # Test invalid UUID format
        response = client.get(
            "/vault/insights/invalid-uuid",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
        
        # Test non-existent insight
        fake_id = str(uuid4())
        response = client.get(
            f"/vault/insights/{fake_id}",
            headers=auth_headers
        )
        assert response.status_code == 404
        
        # Test invalid search data
        invalid_search = {
            "limit": -1,  # Invalid limit
            "offset": -1   # Invalid offset
        }
        response = client.post(
            "/vault/search",
            json=invalid_search,
            headers=auth_headers
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__])