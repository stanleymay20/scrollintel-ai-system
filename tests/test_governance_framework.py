"""Tests for the data governance framework."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

from ai_data_readiness.engines.data_catalog import DataCatalog, DataCatalogError
from ai_data_readiness.engines.policy_engine import PolicyEngine, PolicyEngineError
from ai_data_readiness.engines.audit_logger import AuditLogger, AuditLoggerError
from ai_data_readiness.models.governance_models import (
    DataClassification, PolicyType, AccessLevel, AuditEventType
)


class TestDataCatalog:
    """Test cases for DataCatalog."""
    
    @pytest.fixture
    def catalog(self):
        return DataCatalog()
    
    @pytest.fixture
    def mock_session(self):
        with patch('ai_data_readiness.engines.data_catalog.get_db_session') as mock:
            session = Mock()
            mock.return_value.__enter__.return_value = session
            yield session
    
    def test_register_dataset_success(self, catalog, mock_session):
        """Test successful dataset registration."""
        # Mock dataset exists
        mock_dataset = Mock()
        mock_dataset.id = str(uuid.uuid4())
        mock_session.query.return_value.filter.return_value.first.return_value = mock_dataset
        
        # Mock no existing catalog entry
        mock_session.query.return_value.filter.return_value.first.side_effect = [mock_dataset, None]
        
        # Mock catalog entry creation
        mock_catalog_entry = Mock()
        mock_catalog_entry.id = str(uuid.uuid4())
        mock_catalog_entry.dataset_id = mock_dataset.id
        mock_catalog_entry.name = "Test Dataset"
        mock_catalog_entry.description = "Test Description"
        mock_catalog_entry.classification = DataClassification.INTERNAL.value
        mock_catalog_entry.owner = "test_user"
        mock_catalog_entry.steward = None
        mock_catalog_entry.business_glossary_terms = []
        mock_catalog_entry.tags = []
        mock_catalog_entry.schema_info = {}
        mock_catalog_entry.lineage_info = {}
        mock_catalog_entry.quality_metrics = {}
        mock_catalog_entry.usage_statistics = {}
        mock_catalog_entry.retention_policy = None
        mock_catalog_entry.compliance_requirements = []
        mock_catalog_entry.created_at = datetime.utcnow()
        mock_catalog_entry.updated_at = datetime.utcnow()
        mock_catalog_entry.last_accessed = None
        
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        # Test registration
        result = catalog.register_dataset(
            dataset_id=mock_dataset.id,
            name="Test Dataset",
            description="Test Description",
            owner="test_user"
        )
        
        assert result.name == "Test Dataset"
        assert result.description == "Test Description"
        assert result.owner == "test_user"
        assert result.classification == DataClassification.INTERNAL
    
    def test_register_dataset_not_found(self, catalog, mock_session):
        """Test registration with non-existent dataset."""
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(DataCatalogError, match="Dataset .* not found"):
            catalog.register_dataset(
                dataset_id=str(uuid.uuid4()),
                name="Test Dataset",
                description="Test Description",
                owner="test_user"
            )
    
    def test_search_catalog(self, catalog, mock_session):
        """Test catalog search functionality."""
        mock_entries = [Mock(), Mock()]
        for i, entry in enumerate(mock_entries):
            entry.id = str(uuid.uuid4())
            entry.dataset_id = str(uuid.uuid4())
            entry.name = f"Dataset {i}"
            entry.description = f"Description {i}"
            entry.classification = DataClassification.INTERNAL.value
            entry.owner = "test_user"
            entry.steward = None
            entry.business_glossary_terms = []
            entry.tags = []
            entry.schema_info = {}
            entry.lineage_info = {}
            entry.quality_metrics = {}
            entry.usage_statistics = {}
            entry.retention_policy = None
            entry.compliance_requirements = []
            entry.created_at = datetime.utcnow()
            entry.updated_at = datetime.utcnow()
            entry.last_accessed = None
        
        mock_session.query.return_value.filter.return_value.limit.return_value.all.return_value = mock_entries
        
        results = catalog.search_catalog(query="test", limit=10)
        
        assert len(results) == 2
        assert all(result.name.startswith("Dataset") for result in results)


class TestPolicyEngine:
    """Test cases for PolicyEngine."""
    
    @pytest.fixture
    def policy_engine(self):
        return PolicyEngine()
    
    @pytest.fixture
    def mock_session(self):
        with patch('ai_data_readiness.engines.policy_engine.get_db_session') as mock:
            session = Mock()
            mock.return_value.__enter__.return_value = session
            yield session
    
    def test_create_policy_success(self, policy_engine, mock_session):
        """Test successful policy creation."""
        mock_policy = Mock()
        mock_policy.id = str(uuid.uuid4())
        mock_policy.name = "Test Policy"
        mock_policy.description = "Test Description"
        mock_policy.policy_type = PolicyType.ACCESS_CONTROL.value
        mock_policy.status = "draft"
        mock_policy.rules = [{"condition": {"actions": ["read"]}, "action": "allow"}]
        mock_policy.conditions = {}
        mock_policy.enforcement_level = "strict"
        mock_policy.applicable_resources = []
        mock_policy.created_by = "test_user"
        mock_policy.approved_by = None
        mock_policy.created_at = datetime.utcnow()
        mock_policy.updated_at = datetime.utcnow()
        mock_policy.effective_date = None
        mock_policy.expiry_date = None
        mock_policy.version = "1.0"
        
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        rules = [{"condition": {"actions": ["read"]}, "action": "allow"}]
        
        result = policy_engine.create_policy(
            name="Test Policy",
            description="Test Description",
            policy_type=PolicyType.ACCESS_CONTROL,
            rules=rules,
            created_by="test_user"
        )
        
        assert result.name == "Test Policy"
        assert result.policy_type == PolicyType.ACCESS_CONTROL
        assert len(result.rules) == 1
    
    def test_enforce_policy_allowed(self, policy_engine, mock_session):
        """Test policy enforcement allowing access."""
        mock_policies = []
        mock_session.query.return_value.filter.return_value.all.return_value = mock_policies
        
        allowed, violations = policy_engine.enforce_policy(
            user_id="test_user",
            resource_id="test_resource",
            resource_type="dataset",
            action="read"
        )
        
        assert allowed is True
        assert len(violations) == 0
    
    def test_grant_access_success(self, policy_engine, mock_session):
        """Test successful access granting."""
        # Mock no existing access
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        mock_access_entry = Mock()
        mock_access_entry.id = str(uuid.uuid4())
        mock_access_entry.resource_id = "test_resource"
        mock_access_entry.resource_type = "dataset"
        mock_access_entry.principal_id = "test_user"
        mock_access_entry.principal_type = "user"
        mock_access_entry.access_level = AccessLevel.READ.value
        mock_access_entry.granted_by = "admin_user"
        mock_access_entry.granted_at = datetime.utcnow()
        mock_access_entry.expires_at = None
        mock_access_entry.conditions = {}
        mock_access_entry.is_active = True
        
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        result = policy_engine.grant_access(
            user_id="test_user",
            resource_id="test_resource",
            resource_type="dataset",
            access_level=AccessLevel.READ,
            granted_by="admin_user"
        )
        
        assert result.resource_id == "test_resource"
        assert result.access_level == AccessLevel.READ
        assert result.is_active is True


class TestAuditLogger:
    """Test cases for AuditLogger."""
    
    @pytest.fixture
    def audit_logger(self):
        return AuditLogger()
    
    @pytest.fixture
    def mock_session(self):
        with patch('ai_data_readiness.engines.audit_logger.get_db_session') as mock:
            session = Mock()
            mock.return_value.__enter__.return_value = session
            yield session
    
    def test_log_data_access(self, audit_logger, mock_session):
        """Test logging data access event."""
        mock_audit_event = Mock()
        mock_audit_event.id = str(uuid.uuid4())
        mock_audit_event.event_type = AuditEventType.DATA_ACCESS.value
        mock_audit_event.user_id = "test_user"
        mock_audit_event.resource_id = "test_resource"
        mock_audit_event.resource_type = "dataset"
        mock_audit_event.action = "read"
        mock_audit_event.details = {}
        mock_audit_event.ip_address = "127.0.0.1"
        mock_audit_event.user_agent = "test-agent"
        mock_audit_event.session_id = "test-session"
        mock_audit_event.timestamp = datetime.utcnow()
        mock_audit_event.success = True
        mock_audit_event.error_message = None
        
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        result = audit_logger.log_data_access(
            user_id="test_user",
            resource_id="test_resource",
            resource_type="dataset",
            action="read",
            ip_address="127.0.0.1",
            user_agent="test-agent",
            session_id="test-session"
        )
        
        assert result.event_type == AuditEventType.DATA_ACCESS
        assert result.user_id == "test_user"
        assert result.resource_id == "test_resource"
        assert result.action == "read"
    
    def test_get_audit_trail(self, audit_logger, mock_session):
        """Test retrieving audit trail."""
        mock_events = [Mock(), Mock()]
        for i, event in enumerate(mock_events):
            event.id = str(uuid.uuid4())
            event.event_type = AuditEventType.DATA_ACCESS.value
            event.user_id = f"user_{i}"
            event.resource_id = "test_resource"
            event.resource_type = "dataset"
            event.action = "read"
            event.details = {}
            event.ip_address = "127.0.0.1"
            event.user_agent = "test-agent"
            event.session_id = "test-session"
            event.timestamp = datetime.utcnow()
            event.success = True
            event.error_message = None
        
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = mock_events
        
        results = audit_logger.get_audit_trail(
            resource_id="test_resource",
            resource_type="dataset",
            limit=10
        )
        
        assert len(results) == 2
        assert all(result.resource_id == "test_resource" for result in results)
    
    def test_get_user_activity_summary(self, audit_logger, mock_session):
        """Test getting user activity summary."""
        mock_events = []
        for i in range(5):
            event = Mock()
            event.event_type = AuditEventType.DATA_ACCESS.value
            event.action = "read"
            event.success = True
            event.resource_id = f"resource_{i}"
            event.resource_type = "dataset"
            event.timestamp = datetime.utcnow()
            mock_events.append(event)
        
        mock_session.query.return_value.filter.return_value.all.return_value = mock_events
        
        summary = audit_logger.get_user_activity_summary(
            user_id="test_user",
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        assert summary['user_id'] == "test_user"
        assert summary['total_events'] == 5
        assert summary['successful_events'] == 5
        assert summary['failed_events'] == 0
        assert summary['success_rate'] == 1.0
        assert 'event_types' in summary
        assert 'actions' in summary


class TestGovernanceIntegration:
    """Integration tests for governance components."""
    
    @pytest.fixture
    def catalog(self):
        return DataCatalog()
    
    @pytest.fixture
    def policy_engine(self):
        return PolicyEngine()
    
    @pytest.fixture
    def audit_logger(self):
        return AuditLogger()
    
    @patch('ai_data_readiness.engines.data_catalog.get_db_session')
    @patch('ai_data_readiness.engines.policy_engine.get_db_session')
    @patch('ai_data_readiness.engines.audit_logger.get_db_session')
    def test_governance_workflow(self, mock_audit_session, mock_policy_session, mock_catalog_session, 
                                catalog, policy_engine, audit_logger):
        """Test integrated governance workflow."""
        # Setup mocks for all sessions
        for mock_session_func in [mock_catalog_session, mock_policy_session, mock_audit_session]:
            session = Mock()
            mock_session_func.return_value.__enter__.return_value = session
            session.query.return_value.filter.return_value.first.return_value = None
            session.add.return_value = None
            session.commit.return_value = None
            session.refresh.return_value = None
        
        # This would test a complete workflow:
        # 1. Register dataset in catalog
        # 2. Create access policy
        # 3. Grant access to user
        # 4. Log access event
        # 5. Verify audit trail
        
        # For now, just verify components can be instantiated together
        assert catalog is not None
        assert policy_engine is not None
        assert audit_logger is not None


if __name__ == "__main__":
    pytest.main([__file__])