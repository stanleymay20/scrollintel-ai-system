"""
Comprehensive test suite for API Key Management system.
Tests all components of the API key management and usage tracking.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from fastapi import FastAPI

from scrollintel.models.api_key_models import APIKey, APIUsage, APIQuota, RateLimitRecord
from scrollintel.models.database import User
from scrollintel.core.api_key_manager import APIKeyManager
from scrollintel.core.api_billing_service import APIBillingService
from scrollintel.engines.usage_analytics_engine import UsageAnalyticsEngine
from scrollintel.api.routes.api_key_routes import router


@pytest.fixture
def db_session():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def test_user():
    """Test user fixture."""
    user = Mock(spec=User)
    user.id = "test-user-id"
    user.email = "test@example.com"
    user.is_active = True
    return user


@pytest.fixture
def api_key_manager(db_session):
    """API key manager fixture."""
    return APIKeyManager(db_session)


@pytest.fixture
def billing_service(db_session):
    """Billing service fixture."""
    return APIBillingService(db_session)


@pytest.fixture
def analytics_engine(db_session):
    """Analytics engine fixture."""
    return UsageAnalyticsEngine(db_session)


@pytest.fixture
def test_app():
    """Test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def test_client(test_app):
    """Test client fixture."""
    return TestClient(test_app)


class TestAPIKeyManager:
    """Test cases for APIKeyManager."""
    
    def test_create_api_key(self, api_key_manager, db_session):
        """Test API key creation."""
        # Mock database operations
        db_session.add = Mock()
        db_session.commit = Mock()
        db_session.refresh = Mock()
        
        # Create API key
        api_key, raw_key = api_key_manager.create_api_key(
            user_id="test-user-id",
            name="Test API Key",
            description="Test description",
            permissions=["agents:read", "data:read"],
            rate_limit_per_minute=100,
            rate_limit_per_hour=1000,
            rate_limit_per_day=10000
        )
        
        # Verify API key properties
        assert isinstance(api_key, APIKey)
        assert api_key.name == "Test API Key"
        assert api_key.description == "Test description"
        assert api_key.permissions == ["agents:read", "data:read"]
        assert api_key.rate_limit_per_minute == 100
        assert api_key.rate_limit_per_hour == 1000
        assert api_key.rate_limit_per_day == 10000
        
        # Verify raw key format
        assert raw_key.startswith("sk-")
        assert len(raw_key) > 40
        
        # Verify database operations
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()
        db_session.refresh.assert_called_once()
    
    def test_validate_api_key_valid(self, api_key_manager, db_session):
        """Test API key validation with valid key."""
        # Create test API key
        test_key = Mock(spec=APIKey)
        test_key.id = "test-key-id"
        test_key.is_active = True
        test_key.expires_at = None
        test_key.last_used = None
        
        # Mock database query
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = test_key
        db_session.query.return_value = query_mock
        
        # Validate key
        raw_key = "sk-test-key-12345"
        result = api_key_manager.validate_api_key(raw_key)
        
        # Verify result
        assert result == test_key
        assert test_key.last_used is not None
        db_session.commit.assert_called_once()
    
    def test_validate_api_key_invalid(self, api_key_manager, db_session):
        """Test API key validation with invalid key."""
        # Mock database query returning None
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = None
        db_session.query.return_value = query_mock
        
        # Validate invalid key
        result = api_key_manager.validate_api_key("invalid-key")
        
        # Verify result
        assert result is None
    
    def test_validate_api_key_expired(self, api_key_manager, db_session):
        """Test API key validation with expired key."""
        # Create expired API key
        test_key = Mock(spec=APIKey)
        test_key.id = "test-key-id"
        test_key.is_active = True
        test_key.expires_at = datetime.utcnow() - timedelta(days=1)
        test_key.is_expired.return_value = True
        
        # Mock database query
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = test_key
        db_session.query.return_value = query_mock
        
        # Validate expired key
        result = api_key_manager.validate_api_key("sk-expired-key")
        
        # Verify result
        assert result is None
    
    def test_check_rate_limit(self, api_key_manager, db_session):
        """Test rate limit checking."""
        # Create test API key
        test_key = Mock(spec=APIKey)
        test_key.id = "test-key-id"
        test_key.rate_limit_per_minute = 60
        test_key.rate_limit_per_hour = 1000
        test_key.rate_limit_per_day = 10000
        
        # Mock rate limit records
        minute_record = Mock(spec=RateLimitRecord)
        minute_record.request_count = 30
        
        hour_record = Mock(spec=RateLimitRecord)
        hour_record.request_count = 500
        
        day_record = Mock(spec=RateLimitRecord)
        day_record.request_count = 2000
        
        # Mock _get_or_create_rate_limit_record
        api_key_manager._get_or_create_rate_limit_record = Mock()
        api_key_manager._get_or_create_rate_limit_record.side_effect = [
            minute_record, hour_record, day_record
        ]
        
        # Check rate limit
        result = api_key_manager.check_rate_limit(test_key)
        
        # Verify result
        assert result['allowed'] is True
        assert result['minute']['used'] == 30
        assert result['minute']['remaining'] == 30
        assert result['hour']['used'] == 500
        assert result['hour']['remaining'] == 500
        assert result['day']['used'] == 2000
        assert result['day']['remaining'] == 8000
    
    def test_record_api_usage(self, api_key_manager, db_session):
        """Test API usage recording."""
        # Create test API key
        test_key = Mock(spec=APIKey)
        test_key.id = "test-key-id"
        test_key.user_id = "test-user-id"
        
        # Mock database operations
        db_session.add = Mock()
        db_session.commit = Mock()
        
        # Mock helper methods
        api_key_manager._increment_rate_limit_counters = Mock()
        api_key_manager._update_quota_usage = Mock()
        
        # Record usage
        usage = api_key_manager.record_api_usage(
            api_key=test_key,
            endpoint="/api/v1/agents",
            method="POST",
            status_code=200,
            response_time_ms=150.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0"
        )
        
        # Verify usage record
        assert isinstance(usage, APIUsage)
        assert usage.api_key_id == test_key.id
        assert usage.user_id == test_key.user_id
        assert usage.endpoint == "/api/v1/agents"
        assert usage.method == "POST"
        assert usage.status_code == 200
        assert usage.response_time_ms == 150.5
        
        # Verify database operations
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()
        api_key_manager._increment_rate_limit_counters.assert_called_once()
        api_key_manager._update_quota_usage.assert_called_once()


class TestAPIBillingService:
    """Test cases for APIBillingService."""
    
    def test_calculate_usage_cost(self, billing_service, db_session):
        """Test usage cost calculation."""
        # Mock usage query
        usage_query = Mock()
        usage_query.count.return_value = 5000  # 5000 requests
        usage_query.filter.return_value = usage_query
        
        # Mock database queries
        db_session.query.return_value = usage_query
        
        # Mock scalar queries for data transfer and compute time
        db_session.query.return_value.filter.return_value.scalar.side_effect = [
            1073741824,  # 1GB in bytes
            300000       # 300 seconds in milliseconds
        ]
        
        # Calculate cost
        result = billing_service.calculate_usage_cost(
            api_key_id="test-key-id",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            pricing_tier="starter"
        )
        
        # Verify result structure
        assert 'api_key_id' in result
        assert 'usage' in result
        assert 'costs' in result
        assert 'total_cost_usd' in result
        
        # Verify usage data
        assert result['usage']['requests'] == 5000
        assert result['usage']['data_transfer_gb'] == 1.0
        assert result['usage']['compute_seconds'] == 300.0
    
    def test_calculate_monthly_bill(self, billing_service, db_session):
        """Test monthly bill calculation."""
        # Mock user and API keys
        test_user = Mock(spec=User)
        test_user.email = "test@example.com"
        
        test_key = Mock(spec=APIKey)
        test_key.id = "test-key-id"
        test_key.name = "Test Key"
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.all.return_value = [test_key]
        db_session.query.return_value.filter.return_value.first.return_value = test_user
        
        # Mock calculate_usage_cost
        billing_service.calculate_usage_cost = Mock(return_value={
            'api_key_id': 'test-key-id',
            'usage': {'requests': 1000, 'data_transfer_gb': 0.5, 'compute_seconds': 60.0},
            'costs': {'requests': 1.0, 'data_transfer': 0.0, 'compute': 0.0, 'total': 1.0},
            'total_cost_usd': 1.0
        })
        
        # Calculate monthly bill
        result = billing_service.calculate_monthly_bill(
            user_id="test-user-id",
            year=2024,
            month=1,
            pricing_tier="starter"
        )
        
        # Verify result structure
        assert 'bill_metadata' in result
        assert 'api_key_costs' in result
        assert 'summary' in result
        
        # Verify bill metadata
        assert result['bill_metadata']['user_id'] == "test-user-id"
        assert result['bill_metadata']['billing_period'] == "2024-01"
        
        # Verify summary
        assert result['summary']['total_api_keys'] == 1
        assert result['summary']['total_cost_usd'] == 1.0
    
    def test_get_pricing_estimate(self, billing_service):
        """Test pricing estimation."""
        result = billing_service.get_pricing_estimate(
            requests_per_month=10000,
            data_transfer_gb_per_month=5.0,
            compute_seconds_per_month=300.0,
            pricing_tier="starter"
        )
        
        # Verify result structure
        assert 'pricing_tier' in result
        assert 'projected_usage' in result
        assert 'estimated_costs' in result
        assert 'monthly_estimate_usd' in result
        
        # Verify projected usage
        assert result['projected_usage']['requests_per_month'] == 10000
        assert result['projected_usage']['data_transfer_gb_per_month'] == 5.0
        assert result['projected_usage']['compute_seconds_per_month'] == 300.0


class TestUsageAnalyticsEngine:
    """Test cases for UsageAnalyticsEngine."""
    
    def test_get_user_overview(self, analytics_engine, db_session):
        """Test user overview analytics."""
        # Mock API keys
        test_keys = [
            Mock(spec=APIKey, id="key1", is_active=True),
            Mock(spec=APIKey, id="key2", is_active=False)
        ]
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.all.return_value = test_keys
        
        # Mock usage statistics
        usage_query = Mock()
        usage_query.count.return_value = 1000
        usage_query.filter.return_value.count.return_value = 950
        db_session.query.return_value.filter.return_value = usage_query
        
        # Mock scalar queries
        db_session.query.return_value.filter.return_value.scalar.side_effect = [
            150.5,  # avg response time
            1073741824,  # data transfer bytes
        ]
        
        # Mock aggregated queries
        db_session.query.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [
            ("/api/v1/agents", 500, 120.0),
            ("/api/v1/data", 300, 180.0)
        ]
        
        db_session.query.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = [
            (datetime(2024, 1, 1).date(), 100, 95, 125.0),
            (datetime(2024, 1, 2).date(), 150, 140, 130.0)
        ]
        
        # Get user overview
        result = analytics_engine.get_user_overview("test-user-id", days=30)
        
        # Verify result structure
        assert 'total_api_keys' in result
        assert 'active_api_keys' in result
        assert 'total_requests' in result
        assert 'successful_requests' in result
        assert 'error_rate' in result
        assert 'average_response_time' in result
        assert 'data_transfer_gb' in result
        assert 'top_endpoints' in result
        assert 'daily_usage' in result
        assert 'error_breakdown' in result
        
        # Verify calculated values
        assert result['total_api_keys'] == 2
        assert result['active_api_keys'] == 1
        assert result['total_requests'] == 1000
        assert result['successful_requests'] == 950
        assert result['error_rate'] == 5.0
    
    def test_get_quota_status(self, analytics_engine, db_session):
        """Test quota status retrieval."""
        # Mock API keys
        test_key = Mock(spec=APIKey)
        test_key.id = "test-key-id"
        test_key.name = "Test Key"
        
        # Mock quota
        test_quota = Mock(spec=APIQuota)
        test_quota.api_key_id = "test-key-id"
        test_quota.period_start = datetime(2024, 1, 1)
        test_quota.period_end = datetime(2024, 2, 1)
        test_quota.requests_count = 5000
        test_quota.requests_limit = 10000
        test_quota.data_transfer_bytes = 1073741824
        test_quota.data_transfer_limit_bytes = None
        test_quota.compute_time_seconds = 300.0
        test_quota.compute_time_limit_seconds = None
        test_quota.cost_usd = 5.0
        test_quota.cost_limit_usd = None
        test_quota.is_exceeded = False
        test_quota.exceeded_at = None
        test_quota.get_usage_percentage.side_effect = lambda metric: {
            'requests': 50.0,
            'data_transfer': 0.0,
            'compute_time': 0.0,
            'cost': 0.0
        }[metric]
        
        # Mock database queries
        db_session.query.return_value.filter.return_value.all.return_value = [test_key]
        db_session.query.return_value.filter.return_value.first.return_value = test_quota
        
        # Get quota status
        result = analytics_engine.get_quota_status("test-user-id")
        
        # Verify result structure
        assert 'user_id' in result
        assert 'current_period' in result
        assert 'api_keys' in result
        
        # Verify API key quota data
        assert len(result['api_keys']) == 1
        key_quota = result['api_keys'][0]
        assert key_quota['api_key_id'] == "test-key-id"
        assert key_quota['api_key_name'] == "Test Key"
        assert key_quota['requests']['used'] == 5000
        assert key_quota['requests']['limit'] == 10000


class TestAPIKeyRoutes:
    """Test cases for API key routes."""
    
    @patch('scrollintel.api.routes.api_key_routes.get_current_active_user')
    @patch('scrollintel.api.routes.api_key_routes.get_db')
    def test_create_api_key_endpoint(self, mock_get_db, mock_get_user, test_client):
        """Test API key creation endpoint."""
        # Mock dependencies
        mock_user = Mock(spec=User)
        mock_user.id = "test-user-id"
        mock_get_user.return_value = mock_user
        
        mock_db = Mock(spec=Session)
        mock_get_db.return_value = mock_db
        
        # Mock APIKeyManager
        with patch('scrollintel.api.routes.api_key_routes.APIKeyManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock create_api_key response
            mock_api_key = Mock(spec=APIKey)
            mock_api_key.id = "test-key-id"
            mock_api_key.name = "Test Key"
            mock_api_key.description = "Test description"
            mock_api_key.permissions = ["agents:read"]
            mock_api_key.rate_limit_per_minute = 60
            mock_api_key.rate_limit_per_hour = 1000
            mock_api_key.rate_limit_per_day = 10000
            mock_api_key.quota_requests_per_month = None
            mock_api_key.is_active = True
            mock_api_key.last_used = None
            mock_api_key.expires_at = None
            mock_api_key.created_at = datetime.utcnow()
            mock_api_key.updated_at = datetime.utcnow()
            mock_api_key.get_display_key.return_value = "sk-test...1234"
            
            mock_manager.create_api_key.return_value = (mock_api_key, "sk-test-key-12345")
            
            # Make request
            response = test_client.post(
                "/api/v1/keys/",
                json={
                    "name": "Test Key",
                    "description": "Test description",
                    "permissions": ["agents:read"],
                    "rate_limit_per_minute": 60,
                    "rate_limit_per_hour": 1000,
                    "rate_limit_per_day": 10000
                },
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Verify response
            assert response.status_code == 201
            data = response.json()
            assert 'api_key' in data
            assert 'key' in data
            assert data['key'] == "sk-test-key-12345"
            assert data['api_key']['name'] == "Test Key"
    
    @patch('scrollintel.api.routes.api_key_routes.get_current_active_user')
    @patch('scrollintel.api.routes.api_key_routes.get_db')
    def test_list_api_keys_endpoint(self, mock_get_db, mock_get_user, test_client):
        """Test API key listing endpoint."""
        # Mock dependencies
        mock_user = Mock(spec=User)
        mock_user.id = "test-user-id"
        mock_get_user.return_value = mock_user
        
        mock_db = Mock(spec=Session)
        mock_get_db.return_value = mock_db
        
        # Mock APIKeyManager
        with patch('scrollintel.api.routes.api_key_routes.APIKeyManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock API keys
            mock_api_key = Mock(spec=APIKey)
            mock_api_key.id = "test-key-id"
            mock_api_key.name = "Test Key"
            mock_api_key.description = "Test description"
            mock_api_key.permissions = ["agents:read"]
            mock_api_key.rate_limit_per_minute = 60
            mock_api_key.rate_limit_per_hour = 1000
            mock_api_key.rate_limit_per_day = 10000
            mock_api_key.quota_requests_per_month = None
            mock_api_key.is_active = True
            mock_api_key.last_used = None
            mock_api_key.expires_at = None
            mock_api_key.created_at = datetime.utcnow()
            mock_api_key.updated_at = datetime.utcnow()
            mock_api_key.get_display_key.return_value = "sk-test...1234"
            
            mock_manager.get_user_api_keys.return_value = [mock_api_key]
            
            # Make request
            response = test_client.get(
                "/api/v1/keys/",
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]['name'] == "Test Key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])