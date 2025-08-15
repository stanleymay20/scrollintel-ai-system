"""
Integration tests for the billing and subscription management API.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from decimal import Decimal
import json

from scrollintel.api.main import app
from scrollintel.models.billing_models import (
    Subscription, ScrollCoinWallet, Payment, Invoice,
    SubscriptionTier, SubscriptionStatus, PaymentStatus
)


class TestBillingAPIIntegration:
    """Integration tests for billing API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        return {"Authorization": "Bearer test-token"}
    
    @pytest.fixture
    def mock_user(self):
        """Mock user for testing."""
        return Mock(
            id="test-user-123",
            email="test@example.com",
            full_name="Test User"
        )
    
    def test_get_current_subscription_new_user(self, client, auth_headers):
        """Test getting subscription for new user (should create free tier)."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id="new-user-123")
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                mock_session.query.return_value.filter.return_value.first.return_value = None
                
                response = client.get("/api/billing/subscription", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert "subscription" in data
                assert data["subscription"]["tier"] == "free"
                assert "tier_limits" in data
                assert "usage_stats" in data
    
    def test_get_current_subscription_existing_user(self, client, auth_headers, mock_user):
        """Test getting subscription for existing user."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock existing subscription
                mock_subscription = Mock()
                mock_subscription.id = "sub-123"
                mock_subscription.tier = SubscriptionTier.STARTER
                mock_subscription.status = SubscriptionStatus.ACTIVE
                mock_subscription.billing_cycle.value = "monthly"
                mock_subscription.base_price = Decimal("29.00")
                mock_subscription.currency = "USD"
                mock_subscription.current_period_start = datetime.utcnow()
                mock_subscription.current_period_end = datetime.utcnow() + timedelta(days=30)
                mock_subscription.next_billing_date = datetime.utcnow() + timedelta(days=30)
                mock_subscription.trial_end = None
                mock_subscription.is_trial = False
                mock_subscription.days_until_renewal = 30
                mock_subscription.get_tier_limits.return_value = {
                    "api_calls": 10000,
                    "training_jobs": 10,
                    "storage_gb": 10,
                    "scrollcoins": 1000
                }
                
                mock_session.query.return_value.filter.return_value.first.return_value = mock_subscription
                
                response = client.get("/api/billing/subscription", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert data["subscription"]["tier"] == "starter"
                assert data["subscription"]["base_price"] == 29.0
    
    def test_create_subscription_free_tier(self, client, auth_headers, mock_user):
        """Test creating free tier subscription."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                mock_session.query.return_value.filter.return_value.first.return_value = None
                
                request_data = {
                    "tier": "free",
                    "billing_cycle": "monthly"
                }
                
                response = client.post(
                    "/api/billing/subscription",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["tier"] == "free"
                assert "subscription_id" in data
    
    def test_create_subscription_paid_tier(self, client, auth_headers, mock_user):
        """Test creating paid tier subscription."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                mock_session.query.return_value.filter.return_value.first.return_value = None
                
                with patch('stripe.Customer.create') as mock_customer:
                    mock_customer.return_value = Mock(id="cus_test123")
                    
                    with patch('stripe.Subscription.create') as mock_subscription:
                        mock_subscription.return_value = Mock(
                            id="sub_test123",
                            latest_invoice=Mock(
                                payment_intent=Mock(client_secret="pi_test_secret")
                            )
                        )
                        
                        request_data = {
                            "tier": "starter",
                            "billing_cycle": "monthly",
                            "payment_method_id": "pm_test123"
                        }
                        
                        response = client.post(
                            "/api/billing/subscription",
                            json=request_data,
                            headers=auth_headers
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data["success"] is True
                        assert data["tier"] == "starter"
    
    def test_update_subscription(self, client, auth_headers, mock_user):
        """Test updating existing subscription."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock existing subscription
                mock_subscription = Mock()
                mock_subscription.tier = SubscriptionTier.STARTER
                mock_subscription.billing_cycle.value = "monthly"
                mock_subscription.stripe_subscription_id = "sub_test123"
                mock_subscription.stripe_customer_id = "cus_test123"
                
                mock_session.query.return_value.filter.return_value.first.return_value = mock_subscription
                
                with patch('stripe.Customer.modify') as mock_stripe_update:
                    request_data = {
                        "tier": "professional",
                        "payment_method_id": "pm_new123"
                    }
                    
                    response = client.put(
                        "/api/billing/subscription",
                        json=request_data,
                        headers=auth_headers
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
    
    def test_cancel_subscription(self, client, auth_headers, mock_user):
        """Test cancelling subscription."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock existing subscription
                mock_subscription = Mock()
                mock_subscription.stripe_subscription_id = "sub_test123"
                
                mock_session.query.return_value.filter.return_value.first.return_value = mock_subscription
                
                with patch('stripe.Subscription.modify') as mock_stripe_cancel:
                    response = client.delete("/api/billing/subscription", headers=auth_headers)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert "cancellation_date" in data
    
    def test_get_scrollcoin_wallet(self, client, auth_headers, mock_user):
        """Test getting ScrollCoin wallet."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock wallet
                mock_wallet = Mock()
                mock_wallet.id = "wallet-123"
                mock_wallet.balance = Decimal("1000.00")
                mock_wallet.reserved_balance = Decimal("100.00")
                mock_wallet.available_balance = Decimal("900.00")
                mock_wallet.last_transaction_at = datetime.utcnow()
                mock_wallet.created_at = datetime.utcnow()
                
                mock_session.query.return_value.filter.return_value.first.return_value = mock_wallet
                mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
                
                response = client.get("/api/billing/wallet", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert "wallet" in data
                assert data["wallet"]["balance"] == 1000.0
                assert data["wallet"]["available_balance"] == 900.0
                assert "recent_transactions" in data
    
    def test_recharge_scrollcoin_wallet(self, client, auth_headers, mock_user):
        """Test recharging ScrollCoin wallet."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.engines.billing_engine.ScrollBillingEngine.process') as mock_process:
                mock_process.return_value = {
                    "success": True,
                    "transaction_id": "tx-123",
                    "amount_charged": 50.0,
                    "scrollcoins_added": 5000.0,
                    "new_balance": 6000.0
                }
                
                request_data = {
                    "amount": 50.0,
                    "payment_method_id": "pm_test123"
                }
                
                response = client.post(
                    "/api/billing/wallet/recharge",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["amount_charged"] == 50.0
                assert data["scrollcoins_added"] == 5000.0
    
    def test_track_usage(self, client, auth_headers, mock_user):
        """Test tracking usage."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock subscription
                mock_subscription = Mock()
                mock_subscription.id = "sub-123"
                mock_session.query.return_value.filter.return_value.first.return_value = mock_subscription
                
                request_data = {
                    "metric_type": "api_calls",
                    "quantity": 100,
                    "resource_id": "resource-123",
                    "metadata": {"test": True}
                }
                
                response = client.post(
                    "/api/billing/usage/track",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "usage_id" in data
    
    def test_get_payment_methods(self, client, auth_headers, mock_user):
        """Test getting payment methods."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock payment methods
                mock_pm = Mock()
                mock_pm.id = "pm-123"
                mock_pm.type = "card"
                mock_pm.last_four = "4242"
                mock_pm.brand = "visa"
                mock_pm.exp_month = 12
                mock_pm.exp_year = 2025
                mock_pm.is_default = True
                mock_pm.nickname = "Main Card"
                mock_pm.created_at = datetime.utcnow()
                
                mock_session.query.return_value.filter.return_value.all.return_value = [mock_pm]
                
                response = client.get("/api/billing/payment-methods", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert "payment_methods" in data
                assert len(data["payment_methods"]) == 1
                assert data["payment_methods"][0]["last_four"] == "4242"
    
    def test_add_payment_method(self, client, auth_headers, mock_user):
        """Test adding payment method."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                with patch('stripe.PaymentMethod.retrieve') as mock_stripe_pm:
                    mock_stripe_pm.return_value = Mock(
                        type="card",
                        card=Mock(
                            last4="4242",
                            brand="visa",
                            exp_month=12,
                            exp_year=2025
                        )
                    )
                    
                    request_data = {
                        "stripe_payment_method_id": "pm_test123",
                        "is_default": True,
                        "nickname": "Test Card"
                    }
                    
                    response = client.post(
                        "/api/billing/payment-methods",
                        json=request_data,
                        headers=auth_headers
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert "payment_method_id" in data
    
    def test_get_invoices(self, client, auth_headers, mock_user):
        """Test getting invoices."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock invoice
                mock_invoice = Mock()
                mock_invoice.id = "inv-123"
                mock_invoice.invoice_number = "INV-001"
                mock_invoice.status = "paid"
                mock_invoice.total_amount = Decimal("99.00")
                mock_invoice.currency = "USD"
                mock_invoice.period_start = datetime.utcnow() - timedelta(days=30)
                mock_invoice.period_end = datetime.utcnow()
                mock_invoice.issued_at = datetime.utcnow() - timedelta(days=5)
                mock_invoice.due_date = datetime.utcnow() + timedelta(days=25)
                mock_invoice.paid_at = datetime.utcnow() - timedelta(days=3)
                mock_invoice.is_overdue = False
                
                mock_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_invoice]
                
                response = client.get("/api/billing/invoices", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert "invoices" in data
                assert len(data["invoices"]) == 1
                assert data["invoices"][0]["invoice_number"] == "INV-001"
    
    def test_get_billing_alerts(self, client, auth_headers, mock_user):
        """Test getting billing alerts."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Mock alert
                mock_alert = Mock()
                mock_alert.id = "alert-123"
                mock_alert.alert_type = "usage_limit"
                mock_alert.severity = "warning"
                mock_alert.title = "Usage Limit Warning"
                mock_alert.message = "You have used 90% of your API calls"
                mock_alert.is_read = False
                mock_alert.action_required = True
                mock_alert.action_url = "/billing/upgrade"
                mock_alert.action_text = "Upgrade Plan"
                mock_alert.created_at = datetime.utcnow()
                
                mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_alert]
                
                response = client.get("/api/billing/alerts", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert "alerts" in data
                assert len(data["alerts"]) == 1
                assert data["alerts"][0]["title"] == "Usage Limit Warning"
    
    def test_stripe_webhook_valid_signature(self, client):
        """Test Stripe webhook with valid signature."""
        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_event = {
                "type": "invoice.payment_succeeded",
                "data": {
                    "object": {
                        "id": "in_test123",
                        "customer": "cus_test123",
                        "amount_paid": 2900
                    }
                }
            }
            mock_construct.return_value = mock_event
            
            response = client.post(
                "/api/billing/webhook/stripe",
                json=mock_event,
                headers={"stripe-signature": "test-signature"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_stripe_webhook_invalid_signature(self, client):
        """Test Stripe webhook with invalid signature."""
        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_construct.side_effect = stripe.error.SignatureVerificationError(
                "Invalid signature", "test-signature"
            )
            
            response = client.post(
                "/api/billing/webhook/stripe",
                json={"type": "test"},
                headers={"stripe-signature": "invalid-signature"}
            )
            
            assert response.status_code == 400
    
    def test_subscription_upgrade_flow(self, client, auth_headers, mock_user):
        """Test complete subscription upgrade flow."""
        with patch('scrollintel.security.auth.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            
            with patch('scrollintel.models.database.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value = mock_session
                
                # Step 1: Get current subscription (free)
                mock_subscription = Mock()
                mock_subscription.tier = SubscriptionTier.FREE
                mock_session.query.return_value.filter.return_value.first.return_value = mock_subscription
                
                response = client.get("/api/billing/subscription", headers=auth_headers)
                assert response.status_code == 200
                assert response.json()["subscription"]["tier"] == "free"
                
                # Step 2: Upgrade to starter
                with patch('stripe.Customer.create') as mock_customer:
                    mock_customer.return_value = Mock(id="cus_test123")
                    
                    with patch('stripe.Subscription.create') as mock_stripe_sub:
                        mock_stripe_sub.return_value = Mock(
                            id="sub_test123",
                            latest_invoice=Mock(
                                payment_intent=Mock(client_secret="pi_test_secret")
                            )
                        )
                        
                        request_data = {
                            "tier": "starter",
                            "billing_cycle": "monthly",
                            "payment_method_id": "pm_test123"
                        }
                        
                        response = client.post(
                            "/api/billing/subscription",
                            json=request_data,
                            headers=auth_headers
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data["success"] is True
                        assert data["tier"] == "starter"


if __name__ == "__main__":
    pytest.main([__file__])