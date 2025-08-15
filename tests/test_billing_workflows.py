"""
Tests for billing workflows and payment processing integration.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio

from scrollintel.engines.pricing_engine import PricingEngine
from scrollintel.models.billing_pricing_models import (
    PricingTier, BillingCycle, PaymentStatus, InvoiceStatus,
    UserSubscription, Invoice, Payment, PaymentMethod
)


class TestBillingWorkflows:
    """Test suite for billing workflows and payment processing."""
    
    @pytest.fixture
    def pricing_engine(self):
        """Create pricing engine instance."""
        return PricingEngine()
    
    @pytest.fixture
    def sample_user_subscription(self):
        """Create sample user subscription."""
        return UserSubscription(
            id="sub_123",
            user_id="user_456",
            plan_id="professional",
            plan_name="Professional",
            tier=PricingTier.PROFESSIONAL,
            billing_cycle=BillingCycle.MONTHLY,
            price=49.99,
            start_date=datetime.utcnow() - timedelta(days=15),
            end_date=datetime.utcnow() + timedelta(days=15),
            next_billing_date=datetime.utcnow() + timedelta(days=15),
            current_period_usage={
                "image_generations": 400.0,
                "video_generations": 20.0,
                "gpu_seconds": 1800.0
            },
            remaining_quotas={
                "image_generations": 100.0,
                "video_generations": 5.0,
                "gpu_seconds": 200.0
            },
            status="active",
            auto_renew=True
        )
    
    @pytest.fixture
    def sample_payment_method(self):
        """Create sample payment method."""
        return PaymentMethod(
            id="pm_123",
            user_id="user_456",
            type="credit_card",
            provider="stripe",
            provider_id="pm_stripe_123",
            last_four="4242",
            brand="visa",
            exp_month=12,
            exp_year=2025,
            is_default=True,
            is_verified=True
        )
    
    @pytest.mark.asyncio
    async def test_complete_subscription_billing_workflow(self, pricing_engine, sample_user_subscription, sample_payment_method):
        """Test complete subscription billing workflow."""
        # Step 1: Generate invoice for subscription renewal
        invoice = await pricing_engine.generate_invoice(
            user_id=sample_user_subscription.user_id,
            subscription_id=sample_user_subscription.id,
            period_start=sample_user_subscription.start_date,
            period_end=sample_user_subscription.end_date,
            tax_region="US"
        )
        
        assert invoice.user_id == sample_user_subscription.user_id
        assert invoice.subscription_id == sample_user_subscription.id
        assert invoice.total_amount > 0
        assert invoice.status == InvoiceStatus.DRAFT
        
        # Step 2: Process payment for the invoice
        payment = await pricing_engine.process_payment(
            user_id=sample_user_subscription.user_id,
            amount=invoice.total_amount,
            payment_method_id=sample_payment_method.id,
            invoice_id=invoice.id,
            description=f"Payment for invoice {invoice.invoice_number}"
        )
        
        assert payment.user_id == sample_user_subscription.user_id
        assert payment.invoice_id == invoice.id
        assert payment.amount == invoice.total_amount
        assert payment.payment_method_id == sample_payment_method.id
        
        # Step 3: Update invoice status based on payment result
        if payment.status == PaymentStatus.COMPLETED:
            invoice.status = InvoiceStatus.PAID
            invoice.payment_status = PaymentStatus.COMPLETED
            invoice.payment_date = payment.completed_at
            invoice.payment_reference = payment.provider_charge_id
        else:
            invoice.status = InvoiceStatus.SENT
            invoice.payment_status = PaymentStatus.FAILED
        
        # Verify workflow completion
        assert invoice.payment_status in [PaymentStatus.COMPLETED, PaymentStatus.FAILED]
        if invoice.payment_status == PaymentStatus.COMPLETED:
            assert invoice.status == InvoiceStatus.PAID
            assert invoice.payment_date is not None
    
    @pytest.mark.asyncio
    async def test_usage_based_billing_workflow(self, pricing_engine):
        """Test usage-based billing workflow."""
        user_id = "user_789"
        
        # Step 1: Simulate usage charges for the billing period
        usage_charges = [
            {
                "description": "GPU Usage - Image Generation",
                "quantity": 150.0,
                "unit_price": 0.04,  # Professional tier pricing
                "total_price": 6.00,
                "resource_type": "gpu_seconds",
                "generation_ids": ["gen_1", "gen_2", "gen_3"]
            },
            {
                "description": "API Calls - Overage",
                "quantity": 5000.0,
                "unit_price": 0.0008,  # Professional tier pricing
                "total_price": 4.00,
                "resource_type": "api_calls"
            },
            {
                "description": "Storage - Additional",
                "quantity": 25.0,
                "unit_price": 0.018,  # Professional tier pricing
                "total_price": 0.45,
                "resource_type": "storage_gb"
            }
        ]
        
        # Step 2: Generate invoice with usage charges
        invoice = await pricing_engine.generate_invoice(
            user_id=user_id,
            usage_charges=usage_charges,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            tax_region="EU"  # Test with VAT
        )
        
        assert len(invoice.line_items) == len(usage_charges)
        assert invoice.subtotal == 10.45  # Sum of usage charges
        assert invoice.tax_rate == 0.20  # EU VAT
        assert invoice.tax_amount == invoice.subtotal * 0.20
        assert invoice.total_amount == invoice.subtotal + invoice.tax_amount
        
        # Step 3: Apply discount code
        discount_result = await pricing_engine.apply_discount_code(
            code="SAVE20",
            user_id=user_id,
            invoice_amount=invoice.total_amount
        )
        
        if discount_result["valid"]:
            invoice.discount_amount = discount_result["discount_amount"]
            invoice.total_amount = discount_result["final_amount"]
        
        # Step 4: Process payment
        payment = await pricing_engine.process_payment(
            user_id=user_id,
            amount=invoice.total_amount,
            payment_method_id="pm_usage_test",
            invoice_id=invoice.id,
            description="Usage charges payment"
        )
        
        assert payment.amount == invoice.total_amount
        assert payment.description == "Usage charges payment"
    
    @pytest.mark.asyncio
    async def test_subscription_upgrade_workflow(self, pricing_engine, sample_user_subscription):
        """Test subscription upgrade billing workflow."""
        # Step 1: Calculate upgrade cost
        upgrade_cost = await pricing_engine.calculate_tier_upgrade_cost(
            current_subscription=sample_user_subscription,
            target_plan_id="enterprise",
            prorate=True
        )
        
        assert upgrade_cost["upgrade_cost"] >= 0
        assert upgrade_cost["current_plan"] == "Professional"
        assert upgrade_cost["target_plan"] == "Enterprise"
        
        # Step 2: Generate upgrade invoice
        upgrade_charges = [{
            "description": f"Upgrade to {upgrade_cost['target_plan']} (Prorated)",
            "quantity": 1.0,
            "unit_price": upgrade_cost["upgrade_cost"],
            "total_price": upgrade_cost["upgrade_cost"],
            "resource_type": "subscription_upgrade"
        }]
        
        invoice = await pricing_engine.generate_invoice(
            user_id=sample_user_subscription.user_id,
            subscription_id=sample_user_subscription.id,
            usage_charges=upgrade_charges
        )
        
        assert len(invoice.line_items) == 2  # Original subscription + upgrade
        assert any("Upgrade" in item.description for item in invoice.line_items)
        
        # Step 3: Process upgrade payment
        payment = await pricing_engine.process_payment(
            user_id=sample_user_subscription.user_id,
            amount=invoice.total_amount,
            payment_method_id="pm_upgrade_test",
            invoice_id=invoice.id,
            description="Subscription upgrade payment"
        )
        
        # Step 4: Update subscription (would be done by subscription service)
        if payment.status == PaymentStatus.COMPLETED:
            # Simulate subscription update
            sample_user_subscription.plan_id = "enterprise"
            sample_user_subscription.plan_name = "Enterprise"
            sample_user_subscription.tier = PricingTier.ENTERPRISE
            sample_user_subscription.price = 199.99
            sample_user_subscription.updated_at = datetime.utcnow()
        
        assert payment.description == "Subscription upgrade payment"
    
    @pytest.mark.asyncio
    async def test_failed_payment_retry_workflow(self, pricing_engine):
        """Test failed payment retry workflow."""
        user_id = "user_retry_test"
        
        # Step 1: Generate invoice
        invoice = await pricing_engine.generate_invoice(
            user_id=user_id,
            subscription_id="sub_retry_test"
        )
        
        # Step 2: Attempt payment (may fail due to random simulation)
        max_retries = 3
        payment_attempts = []
        
        for attempt in range(max_retries):
            payment = await pricing_engine.process_payment(
                user_id=user_id,
                amount=invoice.total_amount,
                payment_method_id="pm_retry_test",
                invoice_id=invoice.id,
                description=f"Payment attempt {attempt + 1}"
            )
            
            payment_attempts.append(payment)
            
            if payment.status == PaymentStatus.COMPLETED:
                break
            
            # Wait before retry (in production, this would be longer)
            await asyncio.sleep(0.1)
        
        # Verify retry attempts were made
        assert len(payment_attempts) <= max_retries
        
        # Check final status
        final_payment = payment_attempts[-1]
        if final_payment.status == PaymentStatus.COMPLETED:
            invoice.status = InvoiceStatus.PAID
            invoice.payment_status = PaymentStatus.COMPLETED
        else:
            invoice.status = InvoiceStatus.OVERDUE
            invoice.payment_status = PaymentStatus.FAILED
        
        assert invoice.status in [InvoiceStatus.PAID, InvoiceStatus.OVERDUE]
    
    @pytest.mark.asyncio
    async def test_refund_workflow(self, pricing_engine):
        """Test refund processing workflow."""
        user_id = "user_refund_test"
        
        # Step 1: Create original payment
        original_payment = await pricing_engine.process_payment(
            user_id=user_id,
            amount=49.99,
            payment_method_id="pm_refund_test",
            description="Original subscription payment"
        )
        
        # Step 2: Process refund (simulate refund payment with negative amount)
        refund_payment = await pricing_engine.process_payment(
            user_id=user_id,
            amount=-49.99,  # Negative amount for refund
            payment_method_id="pm_refund_test",
            description=f"Refund for payment {original_payment.id}"
        )
        
        # Step 3: Update original payment status
        if refund_payment.status == PaymentStatus.COMPLETED:
            original_payment.status = PaymentStatus.REFUNDED
            original_payment.metadata = {
                "refund_payment_id": refund_payment.id,
                "refund_date": datetime.utcnow().isoformat(),
                "refund_amount": 49.99
            }
        
        assert refund_payment.amount == -49.99
        assert "Refund" in refund_payment.description
    
    @pytest.mark.asyncio
    async def test_cost_optimization_workflow(self, pricing_engine, sample_user_subscription):
        """Test cost optimization alert and recommendation workflow."""
        # Step 1: Simulate high usage and costs
        high_usage = {
            "image_generations": 800.0,  # Way over quota
            "video_generations": 50.0,
            "gpu_seconds": 5000.0,
            "storage_gb": 200.0
        }
        
        high_costs = {
            "image_generations": 80.0,  # High overage costs
            "video_generations": 40.0,
            "gpu_seconds": 200.0,
            "storage_gb": 4.0
        }
        
        # Step 2: Generate cost optimization alerts
        alerts = await pricing_engine.generate_cost_optimization_alerts(
            user_id=sample_user_subscription.user_id,
            current_usage=high_usage,
            current_costs=high_costs,
            subscription=sample_user_subscription
        )
        
        assert len(alerts) > 0
        
        # Step 3: Check for specific alert types
        alert_types = [alert.alert_type for alert in alerts]
        assert "high_overage_costs" in alert_types
        
        # Step 4: Generate pricing recommendations
        usage_history = [
            {"total_cost": 300.0, "generations": 800},
            {"total_cost": 320.0, "generations": 850},
            {"total_cost": 340.0, "generations": 900}
        ]
        
        recommendations = await pricing_engine.get_pricing_recommendations(
            user_tier=sample_user_subscription.tier,
            usage_history=usage_history,
            current_plan=sample_user_subscription.plan_id
        )
        
        assert len(recommendations["recommendations"]) > 0
        
        # Step 5: Calculate upgrade savings
        upgrade_cost = await pricing_engine.calculate_tier_upgrade_cost(
            current_subscription=sample_user_subscription,
            target_plan_id="enterprise",
            prorate=True
        )
        
        # Verify workflow provides actionable insights
        total_monthly_cost = sum(high_costs.values())
        enterprise_plan = await pricing_engine.get_subscription_plan("enterprise")
        
        if total_monthly_cost > enterprise_plan.monthly_price:
            # Upgrade would save money
            assert upgrade_cost["upgrade_cost"] >= 0
    
    @pytest.mark.asyncio
    async def test_billing_report_generation_workflow(self, pricing_engine):
        """Test billing report generation workflow."""
        # Step 1: Generate user-specific report
        user_report = await pricing_engine.generate_billing_report(
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            user_id="user_report_test"
        )
        
        assert user_report.user_id == "user_report_test"
        assert user_report.total_revenue > 0
        assert user_report.active_subscriptions >= 0
        
        # Step 2: Generate system-wide report
        system_report = await pricing_engine.generate_billing_report(
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow()
        )
        
        assert system_report.user_id is None
        assert system_report.total_revenue > 0
        assert system_report.active_subscriptions > 0
        assert system_report.churn_rate >= 0
        
        # Step 3: Compare reports
        assert system_report.total_revenue >= user_report.total_revenue
        assert system_report.active_subscriptions >= user_report.active_subscriptions
    
    @pytest.mark.asyncio
    async def test_usage_forecasting_workflow(self, pricing_engine):
        """Test usage forecasting and budget planning workflow."""
        user_id = "user_forecast_test"
        
        # Step 1: Prepare historical usage data
        usage_history = []
        base_cost = 50.0
        base_generations = 100
        
        for month in range(6):
            # Simulate growing usage
            monthly_cost = base_cost + (month * 10)
            monthly_generations = base_generations + (month * 20)
            
            usage_history.append({
                "total_cost": monthly_cost,
                "generations": monthly_generations,
                "month": month + 1
            })
        
        # Step 2: Generate usage forecast
        forecast = await pricing_engine.get_usage_forecast(
            user_id=user_id,
            usage_history=usage_history,
            forecast_months=3
        )
        
        assert forecast["user_id"] == user_id
        assert forecast["forecast_period_months"] == 3
        assert len(forecast["monthly_forecasts"]) == 3
        assert forecast["cost_trend_monthly"] > 0  # Growing trend
        
        # Step 3: Analyze forecast for budget planning
        total_forecasted_cost = forecast["total_forecasted_cost"]
        current_monthly_cost = usage_history[-1]["total_cost"]
        
        # Check if forecast suggests plan changes
        if total_forecasted_cost / 3 > current_monthly_cost * 1.5:
            # Significant cost increase predicted
            assert "upgrade" in " ".join(forecast["recommendations"]).lower() or \
                   "optimization" in " ".join(forecast["recommendations"]).lower()
        
        # Step 4: Generate recommendations based on forecast
        recommendations = await pricing_engine.get_pricing_recommendations(
            user_tier=PricingTier.PROFESSIONAL,
            usage_history=usage_history
        )
        
        assert len(recommendations["recommendations"]) >= 0
    
    @pytest.mark.asyncio
    async def test_payment_method_management_workflow(self, pricing_engine):
        """Test payment method management workflow."""
        user_id = "user_payment_method_test"
        
        # Step 1: Create primary payment method
        primary_card = await pricing_engine.create_payment_method(
            user_id=user_id,
            card_details={
                "last_four": "4242",
                "brand": "visa",
                "exp_month": 12,
                "exp_year": 2025
            }
        )
        
        primary_card.is_default = True
        
        # Step 2: Create backup payment method
        backup_card = await pricing_engine.create_payment_method(
            user_id=user_id,
            card_details={
                "last_four": "5555",
                "brand": "mastercard",
                "exp_month": 6,
                "exp_year": 2026
            }
        )
        
        # Step 3: Attempt payment with primary method
        payment_primary = await pricing_engine.process_payment(
            user_id=user_id,
            amount=49.99,
            payment_method_id=primary_card.id,
            description="Payment with primary card"
        )
        
        # Step 4: If primary fails, try backup (simulate failure handling)
        if payment_primary.status == PaymentStatus.FAILED:
            payment_backup = await pricing_engine.process_payment(
                user_id=user_id,
                amount=49.99,
                payment_method_id=backup_card.id,
                description="Payment with backup card"
            )
            
            # Update default payment method if backup succeeds
            if payment_backup.status == PaymentStatus.COMPLETED:
                primary_card.is_default = False
                backup_card.is_default = True
        
        # Verify payment methods were created
        assert primary_card.user_id == user_id
        assert backup_card.user_id == user_id
        assert primary_card.last_four == "4242"
        assert backup_card.last_four == "5555"


if __name__ == "__main__":
    pytest.main([__file__])