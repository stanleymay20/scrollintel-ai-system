"""
Comprehensive tests for the billing and subscription management system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from scrollintel.engines.billing_engine import ScrollBillingEngine, SubscriptionTier, BillingType
from scrollintel.models.billing_models import (
    Subscription, ScrollCoinWallet, Payment, Invoice, UsageRecord,
    BillingAlert, PaymentMethodModel, SubscriptionStatus, PaymentStatus
)


class TestScrollBillingEngine:
    """Test the ScrollBilling engine functionality."""
    
    @pytest.fixture
    def billing_engine(self):
        """Create a billing engine instance for testing."""
        engine = ScrollBillingEngine()
        return engine
    
    @pytest.fixture
    def mock_user_id(self):
        """Mock user ID for testing."""
        return str(uuid4())
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, billing_engine):
        """Test billing engine initialization."""
        await billing_engine.initialize()
        assert billing_engine.status.value in ["ready", "error"]
        assert hasattr(billing_engine, 'scrollcoin_rates')
        assert hasattr(billing_engine, 'subscription_plans')
    
    @pytest.mark.asyncio
    async def test_create_subscription_free_tier(self, billing_engine, mock_user_id):
        """Test creating a free tier subscription."""
        await billing_engine.initialize()
        
        result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "create_subscription",
                "user_id": mock_user_id,
                "tier": "free",
                "billing_type": "fiat"
            }
        )
        
        assert result["success"] is True
        assert result["tier"] == "free"
        assert result["billing_type"] == "fiat"
        assert "subscription_id" in result
    
    @pytest.mark.asyncio
    async def test_create_subscription_paid_tier(self, billing_engine, mock_user_id):
        """Test creating a paid subscription."""
        await billing_engine.initialize()
        
        with patch('stripe.Subscription.create') as mock_stripe:
            mock_stripe.return_value = Mock(id="sub_test123")
            
            result = await billing_engine.process(
                input_data={},
                parameters={
                    "operation": "create_subscription",
                    "user_id": mock_user_id,
                    "tier": "starter",
                    "billing_type": "fiat",
                    "billing_cycle": "monthly"
                }
            )
            
            assert result["success"] is True
            assert result["tier"] == "starter"
            assert "subscription_id" in result
    
    @pytest.mark.asyncio
    async def test_charge_scrollcoins_sufficient_balance(self, billing_engine, mock_user_id):
        """Test charging ScrollCoins with sufficient balance."""
        await billing_engine.initialize()
        
        # Mock wallet with sufficient balance
        billing_engine.wallet_cache[mock_user_id] = {
            "id": f"wallet-{mock_user_id}",
            "user_id": mock_user_id,
            "balance": Decimal("1000.0"),
            "last_updated": datetime.utcnow()
        }
        
        result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "charge_scrollcoins",
                "user_id": mock_user_id,
                "action": "model_inference",
                "quantity": 5
            }
        )
        
        assert result["success"] is True
        assert result["cost"] == 0.5  # 5 * 0.1
        assert result["action"] == "model_inference"
        assert "transaction_id" in result
    
    @pytest.mark.asyncio
    async def test_charge_scrollcoins_insufficient_balance(self, billing_engine, mock_user_id):
        """Test charging ScrollCoins with insufficient balance."""
        await billing_engine.initialize()
        
        # Mock wallet with insufficient balance
        billing_engine.wallet_cache[mock_user_id] = {
            "id": f"wallet-{mock_user_id}",
            "user_id": mock_user_id,
            "balance": Decimal("0.05"),  # Less than required
            "last_updated": datetime.utcnow()
        }
        
        result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "charge_scrollcoins",
                "user_id": mock_user_id,
                "action": "model_inference",
                "quantity": 1
            }
        )
        
        assert result["success"] is False
        assert result["error"] == "insufficient_balance"
        assert "required" in result
        assert "available" in result
    
    @pytest.mark.asyncio
    async def test_recharge_wallet(self, billing_engine, mock_user_id):
        """Test recharging ScrollCoin wallet."""
        await billing_engine.initialize()
        
        # Mock wallet
        billing_engine.wallet_cache[mock_user_id] = {
            "id": f"wallet-{mock_user_id}",
            "user_id": mock_user_id,
            "balance": Decimal("100.0"),
            "last_updated": datetime.utcnow()
        }
        
        with patch.object(billing_engine, '_process_stripe_payment') as mock_payment:
            mock_payment.return_value = {"success": True, "payment_intent_id": "pi_test123"}
            
            result = await billing_engine.process(
                input_data={},
                parameters={
                    "operation": "recharge_wallet",
                    "user_id": mock_user_id,
                    "amount": 50.0,
                    "payment_method": "stripe"
                }
            )
            
            assert result["success"] is True
            assert result["amount_charged"] == 50.0
            assert result["scrollcoins_added"] == 5000.0  # 50 * 100
            assert result["new_balance"] == 5100.0  # 100 + 5000
    
    @pytest.mark.asyncio
    async def test_get_wallet_balance(self, billing_engine, mock_user_id):
        """Test getting wallet balance."""
        await billing_engine.initialize()
        
        # Mock wallet
        billing_engine.wallet_cache[mock_user_id] = {
            "id": f"wallet-{mock_user_id}",
            "user_id": mock_user_id,
            "balance": Decimal("1500.0"),
            "last_updated": datetime.utcnow()
        }
        
        result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "get_wallet_balance",
                "user_id": mock_user_id
            }
        )
        
        assert result["balance"] == 1500.0
        assert result["currency"] == "ScrollCoins"
        assert "wallet_id" in result
    
    @pytest.mark.asyncio
    async def test_generate_invoice(self, billing_engine, mock_user_id):
        """Test invoice generation."""
        await billing_engine.initialize()
        
        period_start = datetime.utcnow() - timedelta(days=30)
        period_end = datetime.utcnow()
        
        result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "generate_invoice",
                "user_id": mock_user_id,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat()
            }
        )
        
        assert "id" in result
        assert result["user_id"] == mock_user_id
        assert "total_cost" in result
        assert "line_items" in result
        assert result["currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_usage_analytics(self, billing_engine, mock_user_id):
        """Test usage analytics generation."""
        await billing_engine.initialize()
        
        # Mock usage tracking data
        billing_engine.usage_tracking[mock_user_id] = {
            "model_inference": {"count": 100, "cost": Decimal("10.0")},
            "data_analysis": {"count": 50, "cost": Decimal("50.0")}
        }
        
        result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "usage_analytics",
                "user_id": mock_user_id,
                "period_days": 30
            }
        )
        
        assert result["period_days"] == 30
        assert result["total_actions"] == 150
        assert result["total_cost"] == 60.0
        assert "action_breakdown" in result
        assert "cost_breakdown" in result
        assert "top_actions" in result
    
    def test_subscription_tier_limits(self, billing_engine):
        """Test subscription tier limits configuration."""
        plans = billing_engine.subscription_plans
        
        # Test free tier
        free_plan = plans[SubscriptionTier.FREE]
        assert free_plan["price_monthly"] == 0
        assert free_plan["api_calls_limit"] == 1000
        assert free_plan["scrollcoins_included"] == 100
        
        # Test starter tier
        starter_plan = plans[SubscriptionTier.STARTER]
        assert starter_plan["price_monthly"] == 29
        assert starter_plan["api_calls_limit"] == 10000
        assert starter_plan["scrollcoins_included"] == 1000
        
        # Test enterprise tier
        enterprise_plan = plans[SubscriptionTier.ENTERPRISE]
        assert enterprise_plan["price_monthly"] == 299
        assert enterprise_plan["api_calls_limit"] == 1000000
        assert enterprise_plan["scrollcoins_included"] == 20000
    
    def test_scrollcoin_rates(self, billing_engine):
        """Test ScrollCoin consumption rates."""
        rates = billing_engine.scrollcoin_rates
        
        assert rates["model_inference"] == Decimal("0.1")
        assert rates["training_job"] == Decimal("10.0")
        assert rates["data_analysis"] == Decimal("1.0")
        assert rates["api_call"] == Decimal("0.05")
    
    @pytest.mark.asyncio
    async def test_stripe_webhook_processing(self, billing_engine):
        """Test Stripe webhook processing."""
        await billing_engine.initialize()
        
        # Mock webhook event
        webhook_event = {
            "type": "invoice.payment_succeeded",
            "data": {
                "object": {
                    "id": "in_test123",
                    "customer": "cus_test123",
                    "amount_paid": 2900,
                    "currency": "usd"
                }
            }
        }
        
        result = await billing_engine.process(
            input_data=webhook_event,
            parameters={
                "operation": "process_stripe_webhook"
            }
        )
        
        # Should process without error
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_engine_cleanup(self, billing_engine):
        """Test engine cleanup."""
        await billing_engine.initialize()
        
        # Add some test data
        billing_engine.active_subscriptions["test"] = {"id": "test"}
        billing_engine.wallet_cache["test"] = {"balance": Decimal("100")}
        
        await billing_engine.cleanup()
        
        assert len(billing_engine.active_subscriptions) == 0
        assert len(billing_engine.wallet_cache) == 0
        assert len(billing_engine.usage_tracking) == 0
    
    def test_engine_status(self, billing_engine):
        """Test engine status reporting."""
        status = billing_engine.get_status()
        
        assert "engine_id" in status
        assert status["engine_id"] == "scroll-billing-engine"
        assert "status" in status
        assert "stripe_available" in status
        assert "supported_tiers" in status
        assert "supported_actions" in status


class TestBillingModels:
    """Test billing database models."""
    
    def test_subscription_model_properties(self):
        """Test subscription model properties."""
        subscription = Subscription(
            user_id=uuid4(),
            tier=SubscriptionTier.STARTER,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            base_price=Decimal("29.00"),
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=30),
            next_billing_date=datetime.utcnow() + timedelta(days=30)
        )
        
        assert subscription.is_active is True
        assert subscription.is_trial is False
        assert subscription.days_until_renewal is not None
        
        # Test tier limits
        limits = subscription.get_tier_limits()
        assert limits["api_calls"] == 10000
        assert limits["scrollcoins"] == 1000
    
    def test_scrollcoin_wallet_operations(self):
        """Test ScrollCoin wallet operations."""
        wallet = ScrollCoinWallet(
            user_id=uuid4(),
            balance=Decimal("1000.00"),
            reserved_balance=Decimal("100.00")
        )
        
        assert wallet.available_balance == Decimal("900.00")
        assert wallet.can_spend(Decimal("500.00")) is True
        assert wallet.can_spend(Decimal("1000.00")) is False
        
        # Test reservation
        assert wallet.reserve_amount(Decimal("200.00")) is True
        assert wallet.reserved_balance == Decimal("300.00")
        assert wallet.can_spend(Decimal("800.00")) is False
        
        # Test debit
        assert wallet.debit(Decimal("100.00")) is True
        assert wallet.balance == Decimal("900.00")
        
        # Test credit
        wallet.credit(Decimal("500.00"))
        assert wallet.balance == Decimal("1400.00")
    
    def test_payment_model_properties(self):
        """Test payment model properties."""
        payment = Payment(
            user_id=uuid4(),
            amount=Decimal("29.00"),
            currency="USD",
            status=PaymentStatus.SUCCEEDED,
            payment_method=PaymentMethod.STRIPE
        )
        
        assert payment.is_successful is True
        assert payment.refunded_amount == Decimal("0.00")
        assert payment.net_amount == Decimal("29.00")
    
    def test_invoice_model_properties(self):
        """Test invoice model properties."""
        invoice = Invoice(
            user_id=uuid4(),
            invoice_number="INV-001",
            status="draft",
            total_amount=Decimal("99.00"),
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            issued_at=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=30)
        )
        
        assert invoice.is_overdue is False
        assert invoice.days_overdue == 0
        
        # Test overdue invoice
        overdue_invoice = Invoice(
            user_id=uuid4(),
            invoice_number="INV-002",
            status="sent",
            total_amount=Decimal("99.00"),
            period_start=datetime.utcnow() - timedelta(days=60),
            period_end=datetime.utcnow() - timedelta(days=30),
            issued_at=datetime.utcnow() - timedelta(days=30),
            due_date=datetime.utcnow() - timedelta(days=5)
        )
        
        assert overdue_invoice.is_overdue is True
        assert overdue_invoice.days_overdue > 0


class TestBillingIntegration:
    """Test billing system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_subscription_upgrade_flow(self):
        """Test complete subscription upgrade flow."""
        billing_engine = ScrollBillingEngine()
        await billing_engine.initialize()
        
        user_id = str(uuid4())
        
        # Create free subscription
        free_result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "create_subscription",
                "user_id": user_id,
                "tier": "free"
            }
        )
        assert free_result["success"] is True
        
        # Upgrade to starter
        with patch('stripe.Subscription.create') as mock_stripe:
            mock_stripe.return_value = Mock(id="sub_test123")
            
            upgrade_result = await billing_engine.process(
                input_data={},
                parameters={
                    "operation": "create_subscription",
                    "user_id": user_id,
                    "tier": "starter",
                    "billing_type": "fiat"
                }
            )
            assert upgrade_result["success"] is True
            assert upgrade_result["tier"] == "starter"
    
    @pytest.mark.asyncio
    async def test_usage_billing_cycle(self):
        """Test complete usage billing cycle."""
        billing_engine = ScrollBillingEngine()
        await billing_engine.initialize()
        
        user_id = str(uuid4())
        
        # Initialize wallet
        billing_engine.wallet_cache[user_id] = {
            "id": f"wallet-{user_id}",
            "user_id": user_id,
            "balance": Decimal("1000.0"),
            "last_updated": datetime.utcnow()
        }
        
        # Perform multiple actions
        actions = [
            ("model_inference", 10),
            ("data_analysis", 5),
            ("visualization", 3)
        ]
        
        total_cost = Decimal("0")
        for action, quantity in actions:
            result = await billing_engine.process(
                input_data={},
                parameters={
                    "operation": "charge_scrollcoins",
                    "user_id": user_id,
                    "action": action,
                    "quantity": quantity
                }
            )
            assert result["success"] is True
            total_cost += Decimal(str(result["cost"]))
        
        # Check final balance
        balance_result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "get_wallet_balance",
                "user_id": user_id
            }
        )
        
        expected_balance = 1000.0 - float(total_cost)
        assert balance_result["balance"] == expected_balance
        
        # Get usage analytics
        analytics_result = await billing_engine.process(
            input_data={},
            parameters={
                "operation": "usage_analytics",
                "user_id": user_id
            }
        )
        
        assert analytics_result["total_actions"] == 18  # 10 + 5 + 3
        assert analytics_result["total_cost"] == float(total_cost)


if __name__ == "__main__":
    pytest.main([__file__])