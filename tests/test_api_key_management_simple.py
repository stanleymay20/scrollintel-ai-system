"""
Simple test suite for API Key Management system.
Tests core functionality without full application dependencies.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from scrollintel.models.api_key_models import APIKey, APIUsage, APIQuota, RateLimitRecord
from scrollintel.core.api_key_manager import APIKeyManager
from scrollintel.core.api_billing_service import APIBillingService


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def api_key_manager(mock_db):
    """API key manager fixture."""
    return APIKeyManager(mock_db)


@pytest.fixture
def billing_service(mock_db):
    """Billing service fixture."""
    return APIBillingService(mock_db)


class TestAPIKeyGeneration:
    """Test API key generation and validation."""
    
    def test_generate_key_format(self):
        """Test API key generation format."""
        raw_key, key_hash = APIKey.generate_key()
        
        # Test key format
        assert raw_key.startswith("sk-")
        assert len(raw_key) > 40
        
        # Test hash generation
        assert len(key_hash) == 64  # SHA-256 hex digest
        assert key_hash != raw_key
    
    def test_hash_key_consistency(self):
        """Test key hashing consistency."""
        test_key = "sk-test-key-12345"
        
        hash1 = APIKey.hash_key(test_key)
        hash2 = APIKey.hash_key(test_key)
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_api_key_display(self):
        """Test API key display functionality."""
        # Create mock API key
        api_key = APIKey(
            user_id="test-user",
            name="Test Key",
            key_hash="abcd1234" * 8,  # 64 chars
            key_prefix="sk-test1"
        )
        
        display_key = api_key.get_display_key()
        assert display_key == "sk-test1...1234"
    
    def test_api_key_expiration(self):
        """Test API key expiration logic."""
        # Non-expiring key
        api_key1 = APIKey(
            user_id="test-user",
            name="Test Key 1",
            key_hash="hash1",
            key_prefix="sk-test1",
            expires_at=None
        )
        assert not api_key1.is_expired()
        
        # Expired key
        api_key2 = APIKey(
            user_id="test-user",
            name="Test Key 2",
            key_hash="hash2",
            key_prefix="sk-test2",
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
        assert api_key2.is_expired()
        
        # Future expiring key
        api_key3 = APIKey(
            user_id="test-user",
            name="Test Key 3",
            key_hash="hash3",
            key_prefix="sk-test3",
            expires_at=datetime.utcnow() + timedelta(days=1)
        )
        assert not api_key3.is_expired()


class TestAPIKeyManager:
    """Test API key manager functionality."""
    
    def test_create_api_key_basic(self, api_key_manager, mock_db):
        """Test basic API key creation."""
        # Mock database operations
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Create API key
        api_key, raw_key = api_key_manager.create_api_key(
            user_id="test-user-id",
            name="Test API Key",
            description="Test description"
        )
        
        # Verify API key properties
        assert api_key.name == "Test API Key"
        assert api_key.description == "Test description"
        assert api_key.user_id == "test-user-id"
        assert api_key.is_active is True
        
        # Verify raw key format
        assert raw_key.startswith("sk-")
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
    
    def test_create_api_key_with_permissions(self, api_key_manager, mock_db):
        """Test API key creation with permissions."""
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        permissions = ["agents:read", "data:write"]
        
        api_key, raw_key = api_key_manager.create_api_key(
            user_id="test-user-id",
            name="Test API Key",
            permissions=permissions,
            rate_limit_per_minute=100,
            rate_limit_per_hour=1000,
            rate_limit_per_day=10000
        )
        
        assert api_key.permissions == permissions
        assert api_key.rate_limit_per_minute == 100
        assert api_key.rate_limit_per_hour == 1000
        assert api_key.rate_limit_per_day == 10000
    
    def test_validate_api_key_success(self, api_key_manager, mock_db):
        """Test successful API key validation."""
        # Create test API key
        test_key = Mock()
        test_key.id = "test-key-id"
        test_key.is_active = True
        test_key.expires_at = None
        test_key.is_expired.return_value = False
        test_key.last_used = None
        
        # Mock database query
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = test_key
        mock_db.query.return_value = query_mock
        mock_db.commit = Mock()
        
        # Validate key
        raw_key = "sk-test-key-12345"
        result = api_key_manager.validate_api_key(raw_key)
        
        # Verify result
        assert result == test_key
        mock_db.commit.assert_called_once()
    
    def test_validate_api_key_invalid_format(self, api_key_manager, mock_db):
        """Test API key validation with invalid format."""
        # Test invalid formats
        invalid_keys = [
            "",
            "invalid-key",
            "sk-",
            None
        ]
        
        for invalid_key in invalid_keys:
            result = api_key_manager.validate_api_key(invalid_key)
            assert result is None
    
    def test_validate_api_key_not_found(self, api_key_manager, mock_db):
        """Test API key validation when key not found."""
        # Mock database query returning None
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = None
        mock_db.query.return_value = query_mock
        
        result = api_key_manager.validate_api_key("sk-nonexistent-key")
        assert result is None


class TestAPIUsageModels:
    """Test API usage model functionality."""
    
    def test_api_usage_creation(self):
        """Test API usage record creation."""
        usage = APIUsage(
            api_key_id="test-key-id",
            user_id="test-user-id",
            endpoint="/api/v1/test",
            method="POST",
            status_code=200,
            response_time_ms=150.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0",
            request_metadata={"test": True}
        )
        
        assert usage.api_key_id == "test-key-id"
        assert usage.user_id == "test-user-id"
        assert usage.endpoint == "/api/v1/test"
        assert usage.method == "POST"
        assert usage.status_code == 200
        assert usage.response_time_ms == 150.5
        assert usage.request_metadata == {"test": True}
    
    def test_api_quota_usage_percentage(self):
        """Test API quota usage percentage calculation."""
        quota = APIQuota(
            api_key_id="test-key-id",
            user_id="test-user-id",
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=30),
            requests_count=500,
            requests_limit=1000,
            data_transfer_bytes=1073741824,  # 1GB
            data_transfer_limit_bytes=5368709120,  # 5GB
            compute_time_seconds=300.0,
            compute_time_limit_seconds=600.0,
            cost_usd=5.0,
            cost_limit_usd=10.0
        )
        
        # Test percentage calculations
        assert quota.get_usage_percentage('requests') == 50.0
        assert quota.get_usage_percentage('data_transfer') == 20.0
        assert quota.get_usage_percentage('compute_time') == 50.0
        assert quota.get_usage_percentage('cost') == 50.0
    
    def test_api_quota_exceeded_checks(self):
        """Test API quota exceeded checks."""
        quota = APIQuota(
            api_key_id="test-key-id",
            user_id="test-user-id",
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=30),
            requests_count=1000,
            requests_limit=1000,
            data_transfer_bytes=1073741824,
            data_transfer_limit_bytes=1073741824,
            compute_time_seconds=600.0,
            compute_time_limit_seconds=600.0,
            cost_usd=10.0,
            cost_limit_usd=10.0
        )
        
        # Test exceeded checks
        assert quota.is_requests_exceeded() is True
        assert quota.is_data_transfer_exceeded() is True
        assert quota.is_compute_time_exceeded() is True
        assert quota.is_cost_exceeded() is True
    
    def test_rate_limit_record_window_active(self):
        """Test rate limit record window activity."""
        now = datetime.utcnow()
        
        # Active window (just started)
        active_record = RateLimitRecord(
            api_key_id="test-key-id",
            window_start=now,
            window_duration_seconds=60,
            request_count=10,
            first_request_at=now,
            last_request_at=now
        )
        
        assert active_record.is_window_active() is True
        
        # Expired window
        expired_record = RateLimitRecord(
            api_key_id="test-key-id",
            window_start=now - timedelta(minutes=2),
            window_duration_seconds=60,
            request_count=10,
            first_request_at=now - timedelta(minutes=2),
            last_request_at=now - timedelta(minutes=1)
        )
        
        assert expired_record.is_window_active() is False


class TestBillingService:
    """Test billing service functionality."""
    
    def test_pricing_tiers_structure(self, billing_service):
        """Test pricing tiers structure."""
        pricing_info = billing_service.get_all_pricing_tiers()
        
        assert 'pricing_tiers' in pricing_info
        assert 'currency' in pricing_info
        assert 'billing_period' in pricing_info
        
        # Check required tiers exist
        tiers = pricing_info['pricing_tiers']
        assert 'free' in tiers
        assert 'starter' in tiers
        assert 'professional' in tiers
        assert 'enterprise' in tiers
        
        # Check tier structure
        for tier_name, tier_config in tiers.items():
            assert 'cost_per_request' in tier_config
            assert 'data_transfer_gb_included' in tier_config
            assert 'cost_per_gb' in tier_config
            assert 'compute_seconds_included' in tier_config
            assert 'cost_per_compute_second' in tier_config
    
    def test_pricing_estimate(self, billing_service):
        """Test pricing estimation."""
        estimate = billing_service.get_pricing_estimate(
            requests_per_month=10000,
            data_transfer_gb_per_month=5.0,
            compute_seconds_per_month=300.0,
            pricing_tier="starter"
        )
        
        assert 'pricing_tier' in estimate
        assert 'projected_usage' in estimate
        assert 'estimated_costs' in estimate
        assert 'monthly_estimate_usd' in estimate
        
        # Verify projected usage
        usage = estimate['projected_usage']
        assert usage['requests_per_month'] == 10000
        assert usage['data_transfer_gb_per_month'] == 5.0
        assert usage['compute_seconds_per_month'] == 300.0
        
        # Verify costs structure
        costs = estimate['estimated_costs']
        assert 'requests' in costs
        assert 'data_transfer' in costs
        assert 'compute' in costs
        assert 'total' in costs
    
    def test_calculate_tier_costs(self, billing_service):
        """Test tier cost calculation."""
        # Test starter tier
        starter_config = billing_service.PRICING_TIERS['starter']
        
        costs = billing_service._calculate_tier_costs(
            requests=15000,  # 5000 over limit
            data_transfer_gb=15.0,  # 5GB over limit
            compute_seconds=900.0,  # 300s over limit
            tier_config=starter_config
        )
        
        # Verify cost structure
        assert 'requests' in costs
        assert 'data_transfer' in costs
        assert 'compute' in costs
        assert 'total' in costs
        assert 'breakdown' in costs
        
        # Verify breakdown
        breakdown = costs['breakdown']
        assert breakdown['billable_requests'] == 5000  # Over the 10k limit
        assert breakdown['billable_data_gb'] == 5.0  # Over the 10GB limit
        assert breakdown['billable_compute_seconds'] == 300.0  # Over the 600s limit
        
        # Verify total is sum of components
        expected_total = costs['requests'] + costs['data_transfer'] + costs['compute']
        assert abs(costs['total'] - expected_total) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])