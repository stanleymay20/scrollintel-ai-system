"""
Unit tests for ScrollVaultEngine - Secure Insight Storage.
Tests encryption, access control, search, and audit functionality.
"""

import pytest
import pytest_asyncio
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from scrollintel.engines.vault_engine import (
    ScrollVaultEngine, VaultInsight, AccessAuditLog, SearchQuery,
    InsightType, AccessLevel, RetentionPolicy
)
from scrollintel.engines.base_engine import EngineStatus


class TestScrollVaultEngine:
    """Test suite for ScrollVaultEngine."""
    
    @pytest_asyncio.fixture
    async def vault_engine(self):
        """Create a vault engine instance for testing."""
        engine = ScrollVaultEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_insight_data(self):
        """Sample insight data for testing."""
        return {
            "title": "Test Analysis Result",
            "content": {
                "analysis": "Sample analysis data",
                "metrics": {"accuracy": 0.95, "precision": 0.92},
                "recommendations": ["Improve data quality", "Add more features"]
            },
            "type": "analysis_result",
            "access_level": "internal",
            "retention_policy": "medium_term",
            "tags": ["ml", "analysis", "test"],
            "metadata": {
                "model_id": "test-model-123",
                "dataset": "customer_data",
                "created_by": "data_scientist"
            }
        }
    
    @pytest.fixture
    def sample_user_permissions(self):
        """Sample user permissions for testing."""
        return {
            "admin": AccessLevel.TOP_SECRET,
            "analyst": AccessLevel.CONFIDENTIAL,
            "viewer": AccessLevel.INTERNAL,
            "guest": AccessLevel.PUBLIC
        }
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, vault_engine):
        """Test vault engine initialization."""
        assert vault_engine.engine_id == "scroll-vault-engine"
        assert vault_engine.name == "ScrollVault Engine"
        assert vault_engine.status == EngineStatus.READY
        assert vault_engine.insights == {}
        assert vault_engine.audit_logs == []
    
    @pytest.mark.asyncio
    async def test_store_insight_success(self, vault_engine, sample_insight_data):
        """Test successful insight storage."""
        # Set up user permissions
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store insight
        result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Verify result
        assert result["success"] is True
        assert "insight_id" in result
        assert result["version"] == 1
        assert result["encrypted"] is True
        
        # Verify insight was stored
        insight_id = result["insight_id"]
        assert insight_id in vault_engine.insights
        
        stored_insight = vault_engine.insights[insight_id]
        assert stored_insight.title == sample_insight_data["title"]
        assert stored_insight.insight_type == InsightType.ANALYSIS_RESULT
        assert stored_insight.access_level == AccessLevel.INTERNAL
        assert stored_insight.creator_id == "test_user"
        assert stored_insight.tags == sample_insight_data["tags"]
        
        # Verify audit log
        assert len(vault_engine.audit_logs) == 1
        audit_log = vault_engine.audit_logs[0]
        assert audit_log.insight_id == insight_id
        assert audit_log.user_id == "test_user"
        assert audit_log.action == "write"
        assert audit_log.success is True
    
    @pytest.mark.asyncio
    async def test_store_insight_permission_denied(self, vault_engine, sample_insight_data):
        """Test insight storage with insufficient permissions."""
        # Set up user with low permissions
        vault_engine.user_permissions["low_user"] = AccessLevel.PUBLIC
        
        # Try to store confidential insight
        sample_insight_data["access_level"] = "confidential"
        
        with pytest.raises(PermissionError):
            await vault_engine.process(
                input_data=sample_insight_data,
                parameters={
                    "operation": "store_insight",
                    "user_id": "low_user",
                    "organization_id": "test_org",
                    "ip_address": "127.0.0.1"
                }
            )
        
        # Verify audit log for failed attempt
        assert len(vault_engine.audit_logs) == 1
        audit_log = vault_engine.audit_logs[0]
        assert audit_log.user_id == "low_user"
        assert audit_log.action == "write"
        assert audit_log.success is False
    
    @pytest.mark.asyncio
    async def test_retrieve_insight_success(self, vault_engine, sample_insight_data):
        """Test successful insight retrieval."""
        # Set up user permissions
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store insight first
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        insight_id = store_result["insight_id"]
        
        # Retrieve insight
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "retrieve_insight",
                "insight_id": insight_id,
                "user_id": "test_user",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Verify result
        assert result["success"] is True
        assert "insight" in result
        
        insight = result["insight"]
        assert insight["id"] == insight_id
        assert insight["title"] == sample_insight_data["title"]
        assert insight["type"] == sample_insight_data["type"]
        assert insight["content"] == sample_insight_data["content"]
        assert insight["access_count"] == 1
        
        # Verify access tracking
        stored_insight = vault_engine.insights[insight_id]
        assert stored_insight.access_count == 1
        assert stored_insight.last_accessed is not None
    
    @pytest.mark.asyncio
    async def test_retrieve_insight_permission_denied(self, vault_engine, sample_insight_data):
        """Test insight retrieval with insufficient permissions."""
        # Set up users with different permissions
        vault_engine.user_permissions["creator"] = AccessLevel.CONFIDENTIAL
        vault_engine.user_permissions["viewer"] = AccessLevel.PUBLIC
        
        # Store confidential insight
        sample_insight_data["access_level"] = "confidential"
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "creator",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        insight_id = store_result["insight_id"]
        
        # Try to retrieve with low permissions
        with pytest.raises(PermissionError):
            await vault_engine.process(
                input_data=None,
                parameters={
                    "operation": "retrieve_insight",
                    "insight_id": insight_id,
                    "user_id": "viewer",
                    "ip_address": "127.0.0.1"
                }
            )
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_insight(self, vault_engine):
        """Test retrieval of non-existent insight."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        with pytest.raises(ValueError, match="not found"):
            await vault_engine.process(
                input_data=None,
                parameters={
                    "operation": "retrieve_insight",
                    "insight_id": "nonexistent-id",
                    "user_id": "test_user",
                    "ip_address": "127.0.0.1"
                }
            )
    
    @pytest.mark.asyncio
    async def test_search_insights_basic(self, vault_engine, sample_insight_data):
        """Test basic insight search functionality."""
        # Set up user permissions
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store multiple insights
        insights = []
        for i in range(3):
            data = sample_insight_data.copy()
            data["title"] = f"Test Insight {i}"
            data["tags"] = ["ml", f"test{i}"]
            
            result = await vault_engine.process(
                input_data=data,
                parameters={
                    "operation": "store_insight",
                    "user_id": "test_user",
                    "organization_id": "test_org",
                    "ip_address": "127.0.0.1"
                }
            )
            insights.append(result["insight_id"])
        
        # Search insights
        search_data = {
            "query": "Test",
            "filters": {},
            "limit": 10,
            "offset": 0
        }
        
        result = await vault_engine.process(
            input_data=search_data,
            parameters={
                "operation": "search_insights",
                "user_id": "test_user",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Verify results
        assert result["success"] is True
        assert len(result["results"]) == 3
        assert result["total_count"] == 3
        assert result["query"] == "Test"
        
        # Verify result format
        for insight_result in result["results"]:
            assert "id" in insight_result
            assert "title" in insight_result
            assert "type" in insight_result
            assert "access_level" in insight_result
            assert "tags" in insight_result
    
    @pytest.mark.asyncio
    async def test_search_insights_with_filters(self, vault_engine, sample_insight_data):
        """Test insight search with filters."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store insights with different types
        data1 = sample_insight_data.copy()
        data1["type"] = "analysis_result"
        data1["tags"] = ["ml", "analysis"]
        
        data2 = sample_insight_data.copy()
        data2["type"] = "report"
        data2["tags"] = ["report", "summary"]
        
        for data in [data1, data2]:
            await vault_engine.process(
                input_data=data,
                parameters={
                    "operation": "store_insight",
                    "user_id": "test_user",
                    "organization_id": "test_org",
                    "ip_address": "127.0.0.1"
                }
            )
        
        # Search with type filter
        search_data = {
            "query": "",
            "filters": {},
            "insight_types": ["analysis_result"],
            "limit": 10,
            "offset": 0
        }
        
        result = await vault_engine.process(
            input_data=search_data,
            parameters={
                "operation": "search_insights",
                "user_id": "test_user",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Verify filtered results
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["type"] == "analysis_result"
    
    @pytest.mark.asyncio
    async def test_update_insight_success(self, vault_engine, sample_insight_data):
        """Test successful insight update with version control."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store original insight
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        original_id = store_result["insight_id"]
        
        # Update insight
        update_data = {
            "title": "Updated Test Analysis",
            "content": {"updated": True, "version": 2},
            "tags": ["ml", "analysis", "updated"]
        }
        
        result = await vault_engine.process(
            input_data=update_data,
            parameters={
                "operation": "update_insight",
                "insight_id": original_id,
                "user_id": "test_user",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Verify update result
        assert result["success"] is True
        assert result["version"] == 2
        assert result["parent_id"] == original_id
        
        # Verify new version was created
        new_id = result["insight_id"]
        assert new_id != original_id
        assert new_id in vault_engine.insights
        
        new_insight = vault_engine.insights[new_id]
        assert new_insight.title == "Updated Test Analysis"
        assert new_insight.version == 2
        assert new_insight.parent_id == original_id
    
    @pytest.mark.asyncio
    async def test_delete_insight_success(self, vault_engine, sample_insight_data):
        """Test successful insight deletion."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store insight
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        insight_id = store_result["insight_id"]
        
        # Delete insight
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "delete_insight",
                "insight_id": insight_id,
                "user_id": "test_user",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Verify deletion
        assert result["success"] is True
        assert result["insight_id"] == insight_id
        assert insight_id not in vault_engine.insights
        assert insight_id not in vault_engine.embeddings_cache
    
    @pytest.mark.asyncio
    async def test_delete_insight_permission_denied(self, vault_engine, sample_insight_data):
        """Test insight deletion with insufficient permissions."""
        vault_engine.user_permissions["creator"] = AccessLevel.INTERNAL
        vault_engine.user_permissions["other_user"] = AccessLevel.INTERNAL
        
        # Store insight
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "creator",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        insight_id = store_result["insight_id"]
        
        # Try to delete as different user
        with pytest.raises(PermissionError):
            await vault_engine.process(
                input_data=None,
                parameters={
                    "operation": "delete_insight",
                    "insight_id": insight_id,
                    "user_id": "other_user",
                    "ip_address": "127.0.0.1"
                }
            )
    
    @pytest.mark.asyncio
    async def test_get_insight_history(self, vault_engine, sample_insight_data):
        """Test insight version history retrieval."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store original insight
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        original_id = store_result["insight_id"]
        
        # Update insight to create version 2
        update_data = {"title": "Updated Version"}
        await vault_engine.process(
            input_data=update_data,
            parameters={
                "operation": "update_insight",
                "insight_id": original_id,
                "user_id": "test_user",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Get history
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "get_insight_history",
                "insight_id": original_id,
                "user_id": "test_user"
            }
        )
        
        # Verify history
        assert result["success"] is True
        assert result["insight_id"] == original_id
        assert len(result["versions"]) == 2
        assert result["total_versions"] == 2
        
        # Verify version ordering
        versions = result["versions"]
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2
        assert versions[0]["is_current"] is True
        assert versions[1]["is_current"] is False
    
    @pytest.mark.asyncio
    async def test_get_access_audit(self, vault_engine, sample_insight_data):
        """Test access audit log retrieval."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store and retrieve insight to generate audit logs
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        insight_id = store_result["insight_id"]
        
        await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "retrieve_insight",
                "insight_id": insight_id,
                "user_id": "test_user",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Get audit logs
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "audit_access",
                "user_id": "test_user",
                "limit": 10,
                "offset": 0
            }
        )
        
        # Verify audit logs
        assert result["success"] is True
        assert len(result["audit_logs"]) == 2  # store + retrieve
        assert result["total_count"] == 2
        
        # Verify log details
        logs = result["audit_logs"]
        assert logs[0]["action"] == "write"
        assert logs[1]["action"] == "read"
        assert all(log["user_id"] == "test_user" for log in logs)
        assert all(log["success"] is True for log in logs)
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_insights(self, vault_engine, sample_insight_data):
        """Test cleanup of expired insights."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store insight with short expiry
        sample_insight_data["retention_policy"] = "temporary"
        store_result = await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        insight_id = store_result["insight_id"]
        
        # Manually expire the insight
        insight = vault_engine.insights[insight_id]
        insight.expires_at = datetime.utcnow() - timedelta(days=1)
        
        # Run cleanup
        result = await vault_engine.process(
            input_data=None,
            parameters={
                "operation": "cleanup_expired",
                "user_id": "system"
            }
        )
        
        # Verify cleanup
        assert result["success"] is True
        assert result["cleaned_up_count"] == 1
        assert insight_id in result["expired_insights"]
        assert insight_id not in vault_engine.insights
    
    @pytest.mark.asyncio
    async def test_encryption_decryption(self, vault_engine):
        """Test content encryption and decryption."""
        test_content = "This is sensitive test content"
        
        # Encrypt content
        encrypted_content, key_id = await vault_engine._encrypt_content(test_content)
        
        # Verify encryption
        assert encrypted_content != test_content
        assert key_id is not None
        assert key_id in vault_engine.encryption_keys
        
        # Decrypt content
        decrypted_content = await vault_engine._decrypt_content(encrypted_content, key_id)
        
        # Verify decryption
        assert decrypted_content == test_content
    
    @pytest.mark.asyncio
    async def test_permission_checks(self, vault_engine):
        """Test access permission checking."""
        # Set up user permissions
        vault_engine.user_permissions = {
            "admin": AccessLevel.TOP_SECRET,
            "analyst": AccessLevel.CONFIDENTIAL,
            "viewer": AccessLevel.INTERNAL,
            "guest": AccessLevel.PUBLIC
        }
        
        # Test read permissions
        assert await vault_engine._check_read_permission("admin", AccessLevel.TOP_SECRET, "other")
        assert await vault_engine._check_read_permission("analyst", AccessLevel.CONFIDENTIAL, "other")
        assert not await vault_engine._check_read_permission("viewer", AccessLevel.CONFIDENTIAL, "other")
        assert not await vault_engine._check_read_permission("guest", AccessLevel.INTERNAL, "other")
        
        # Test creator access
        assert await vault_engine._check_read_permission("guest", AccessLevel.TOP_SECRET, "guest")
        
        # Test admin permissions
        assert await vault_engine._check_admin_permission("admin")
        assert not await vault_engine._check_admin_permission("viewer")
    
    @pytest.mark.asyncio
    async def test_engine_status(self, vault_engine):
        """Test vault engine status reporting."""
        status = vault_engine.get_status()
        
        assert status["engine_id"] == "scroll-vault-engine"
        assert status["status"] == "ready"
        assert "crypto_available" in status
        assert "embeddings_available" in status
        assert "stored_insights" in status
        assert "audit_logs" in status
        assert status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_engine_cleanup(self, vault_engine, sample_insight_data):
        """Test vault engine cleanup."""
        vault_engine.user_permissions["test_user"] = AccessLevel.INTERNAL
        
        # Store some data
        await vault_engine.process(
            input_data=sample_insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "test_user",
                "organization_id": "test_org",
                "ip_address": "127.0.0.1"
            }
        )
        
        # Verify data exists
        assert len(vault_engine.insights) > 0
        assert len(vault_engine.audit_logs) > 0
        
        # Cleanup
        await vault_engine.cleanup()
        
        # Verify cleanup
        assert len(vault_engine.insights) == 0
        assert len(vault_engine.audit_logs) == 0
        assert len(vault_engine.search_index) == 0
        assert len(vault_engine.embeddings_cache) == 0
        assert len(vault_engine.encryption_keys) == 0


if __name__ == "__main__":
    pytest.main([__file__])