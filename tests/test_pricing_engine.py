"""
Tests for the pricing engine with tiered pricing models and billing workflows.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio

from scrollintel.engines.pricing_engine import PricingEngine
from scrollintel.models.billing_pricing_models import (
    PricingTier, BillingCycle, PaymentStatus, InvoiceStatus,
    UserSubscription, CostOptimizationAlert
)
from scrollintel.models.usage_tracking_models import (
    GenerationType, ResourceType, GenerationUsage, ResourceUsage
)


class TestPricingEngine:
    """Test suite for PricingEngine."""
    
    @pytest.fixture
    def pricing_engine(self):
        """Create pricing engine instance."""
        return PricingEngine()
    
    @pytest.fixture
    def sample_subscription(self):
        """Create sample user subscription."""
        return UserSubscription(
            user_id="test_user_123",
            plan_id="professional",
            plan_name="Professional",
            tier=PricingTier.PROFESSIONAL,
            billing_cycle=BillingCycle.MONTHLY,
            price=49.99,
            current_period_usage={
                "image_generations": 300.0,
                "video_generations": 15.0,
                "gpu_seconds": 1500.0,
                "storage_gb": 30.0
            },
            remaining_quotas={
                "image_generations": 200.0,
                "video_generations": 10.0,
                "gpu_seconds": 500.0,
                "storage_gb": 20.0
            }
        )
    
    @pytest.fixture
    def sample_generation_usage(self):
        """Create sample generation usage."""
        return GenerationUsage(
            id="gen_123",
            user_id="test_user_123",
            generation_type=GenerationType.IMAGE,
            model_used="stable-diffusion-xl",
            resources=[
                ResourceUsage(
                    resource_type=ResourceType.GPU_SECONDS,
                    amount=10.0,
                    timestamp=datetime.utcnow()
                ),
                ResourceUsage(
                    resource_type=ResourceType.API_CALLS,
                    amount=5.0,
                    timestamp=datetime.utcnow()
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_calculate_usage_cost_basic(self, pricing_engine):
        """Test basic usage cost calculation."""
        result = await pricing_engine.calculate_usage_cost(
            user_tier=PricingTier.BASIC,
            resource_type="gpu_seconds",
            quantity=10.0
        )
        
        assert result["resource_type"] == "gpu_seconds"
        assert result["quantity"] == 10.0
        assert result["base_unit_price"] == 0.05
        assert result["tier_multiplier"] == 1.0  # Basic tier
        expected_cost = 10.0 * 0.05 * 1.0 * result["time_multiplier"]
        assert abs(result["total_cost"] - expected_cost) < 0.001
        assert result["currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_calculate_usage_cost_with_tier_discount(self, pricing_engine):
        """Test usage cost calculation with tier discount."""
        result = await pricing_engine.calculate_usage_cost(
            user_tier=PricingTier.ENTERPRISE,
            resource_type="gpu_seconds",
            quantity=10.0
        )
        
        assert result["tier_multiplier"] == 0.6  # Enterprise tier discount
        expected_cost = 10.0 * 0.05 * 0.6 * result["time_multiplier"]
        assert abs(result["total_cost"] - expected_cost) < 0.001
    
    @pytest.mark.asyncio
    async def test_calculate_usage_cost_with_volume_discount(self, pricing_engine):
        """Test usage cost calculation with volume discount."""
        # High volume usage to trigger volume discount
        result = await pricing_engine.calculate_usage_cost(
            user_tier=PricingTier.BASIC,
            resource_type="gpu_seconds",
            quantity=10.0,
            user_monthly_usage={"gpu_seconds": 600.0}  # Triggers volume discount
        )
        
        assert result["volume_discount"] > 0.0
        assert result["total_cost"] < 10.0 * 0.05 * result["time_multiplier"]
    
    @pytest.mark.asyncio
    async def test_calculate_usage_cost_peak_hours(self, pricing_engine):
        """Test usage cost calculation during peak hours."""
        peak_time = datetime.utcnow().replace(hour=12)  # Noon - peak hour
        
        result = await pricing_engine.calculate_usage_cost(
            user_tier=PricingTier.BASIC,
            resource_type="gpu_seconds",
            quantity=10.0,
            timestamp=peak_time
        )
        
        assert result["time_multiplier"] == 1.2  # Peak hours multiplier
    
    @pytest.mark.asyncio
    async def test_calculate_usage_cost_off_peak_hours(self, pricing_engine):
        """Test usage cost calculation during off-peak hours."""
        off_peak_time = datetime.utcnow().replace(hour=22)  # 10 PM - off-peak
        
        result = await pricing_engine.calculate_usage_cost(
            user_tier=PricingTier.BASIC,
            resource_type="gpu_seconds",
            quantity=10.0,
            timestamp=off_peak_time
        )
        
        assert result["time_multiplier"] == 0.8  # Off-peak hours multiplier
    
    @pytest.mark.asyncio
    async def test_calculate_generation_cost(self, pricing_engine, sample_generation_usage):
        """Test generation cost calculation."""
        result = await pricing_engine.calculate_generation_cost(
            user_tier=PricingTier.PROFESSIONAL,
            generation_usage=sample_generation_usage
        )
        
        assert result["generation_id"] == "gen_123"
        assert result["generation_type"] == "image"
        assert result["model_used"] == "stable-diffusion-xl"
        assert result["total_cost"] > 0
        assert len(result["cost_breakdown"]) == 2  # GPU and API costs
        assert result["currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_get_subscription_plan(self, pricing_engine):
        """Test subscription plan retrieval."""
        plan = await pricing_engine.get_subscription_plan("professional")
        
        assert plan is not None
        assert plan.name == "Professional"
        assert plan.tier == PricingTier.PROFESSIONAL
        assert plan.monthly_price == 49.99
        assert plan.yearly_price == 499.99
        assert "API access" in plan.features
    
    @pytest.mark.asyncio
    async def test_calculate_subscription_cost_monthly(self, pricing_engine):
        """Test monthly subscription cost calculation."""
        result = await pricing_engine.calculate_subscription_cost(
            plan_id="professional",
            billing_cycle=BillingCycle.MONTHLY
        )
        
        assert result["plan_id"] == "professional"
        assert result["plan_name"] == "Professional"
        assert result["billing_cycle"] == "monthly"
        assert result["base_cost"] == 49.99
        assert result["total_cost"] == 49.99  # No tax
        assert result["currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_calculate_subscription_cost_with_tax(self, pricing_engine):
        """Test subscription cost calculation with tax."""
        result = await pricing_engine.calculate_subscription_cost(
            plan_id="professional",
            billing_cycle=BillingCycle.MONTHLY,
            tax_region="US"
        )
        
        assert result["tax_rate"] == 0.08
        assert result["tax_amount"] == 49.99 * 0.08
        assert result["total_cost"] == 49.99 + (49.99 * 0.08)
    
    @pytest.mark.asyncio
    async def test_check_subscription_quotas_within_limit(self, pricing_engine, sample_subscription):
        """Test quota check when within limits."""
        result = await pricing_engine.check_subscription_quotas(
            subscription=sample_subscription,
            resource_type="image_generations",
            requested_quantity=50.0
        )
        
        assert result["allowed"] is True
        assert "quota_exceeded" not in result
        assert result["remaining_quota"] == 200.0
        assert result["requested_quantity"] == 50.0
    
    @pytest.mark.asyncio
    async def test_check_subscription_quotas_with_overage(self, pricing_engine, sample_subscription):
        """Test quota check with overage charges."""
        result = await pricing_engine.check_subscription_quotas(
            subscription=sample_subscription,
            resource_type="image_generations",
            requested_quantity=300.0  # Exceeds remaining quota
        )
        
        assert result["allowed"] is True
        assert result["quota_exceeded"] is True
        assert result["overage_quantity"] == 100.0  # 300 - 200 remaining
        assert result["overage_cost"] > 0
    
    @pytest.mark.asyncio
    async def test_generate_invoice_subscription_only(self, pricing_engine):
        """Test invoice generation for subscription only."""
        invoice = await pricing_engine.generate_invoice(
            user_id="test_user_123",
            subscription_id="sub_123"
        )
        
        assert invoice.user_id == "test_user_123"
        assert invoice.subscription_id == "sub_123"
        assert len(invoice.line_items) == 1
        assert invoice.subtotal == 49.99
        assert invoice.total_amount == 49.99
        assert invoice.status == InvoiceStatus.DRAFT
    
    @pytest.mark.asyncio
    async def test_generate_invoice_with_usage_charges(self, pricing_engine):
        """Test invoice generation with usage charges."""
        usage_charges = [
            {
                "description": "GPU Usage - Image Generation",
                "quantity": 100.0,
                "unit_price": 0.05,
                "total_price": 5.00,
                "resource_type": "gpu_seconds"
            },
            {
                "description": "API Calls",
                "quantity": 1000.0,
                "unit_price": 0.001,
                "total_price": 1.00,
                "resource_type": "api_calls"
            }
        ]
        
        invoice = await pricing_engine.generate_invoice(
            user_id="test_user_123",
            subscription_id="sub_123",
            usage_charges=usage_charges
        )
        
        assert len(invoice.line_items) == 3  # 1 subscription + 2 usage
        assert invoice.subtotal == 49.99 + 5.00 + 1.00
        assert invoice.total_amount == invoice.subtotal
    
    @pytest.mark.asyncio
    async def test_generate_cost_optimization_alerts(self, pricing_engine, sample_subscription):
        """Test cost optimization alert generation."""
        current_usage = {
            "image_generations": 600.0,  # High usage
            "gpu_seconds": 3000.0,
            "storage_gb": 100.0
        }
        
        current_costs = {
            "image_generations": 50.0,
            "gpu_seconds": 150.0,  # High GPU costs
            "storage_gb": 2.0
        }
        
        alerts = await pricing_engine.generate_cost_optimization_alerts(
            user_id="test_user_123",
            current_usage=current_usage,
            current_costs=current_costs,
            subscription=sample_subscription
        )
        
        assert len(alerts) > 0
        
        # Check for high overage alert
        overage_alert = next((a for a in alerts if a.alert_type == "high_overage_costs"), None)
        assert overage_alert is not None
        assert overage_alert.severity == "high"
        assert len(overage_alert.recommendations) > 0
        
        # Check for GPU optimization alert
        gpu_alert = next((a for a in alerts if a.alert_type == "gpu_cost_optimization"), None)
        assert gpu_alert is not None
        assert "batch processing" in " ".join(gpu_alert.recommendations).lower()
    
    @pytest.mark.asyncio
    async def test_generate_billing_report_user_specific(self, pricing_engine):
        """Test user-specific billing report generation."""
        period_start = datetime.utcnow() - timedelta(days=30)
        period_end = datetime.utcnow()
        
        report = await pricing_engine.generate_billing_report(
            period_start=period_start,
            period_end=period_end,
            user_id="test_user_123"
        )
        
        assert report.user_id == "test_user_123"
        assert report.period_start == period_start
        assert report.period_end == period_end
        assert report.total_revenue > 0
        assert report.subscription_revenue > 0
        assert report.usage_revenue > 0
        assert report.active_subscriptions == 1
    
    @pytest.mark.asyncio
    async def test_generate_billing_report_system_wide(self, pricing_engine):
        """Test system-wide billing report generation."""
        period_start = datetime.utcnow() - timedelta(days=30)
        period_end = datetime.utcnow()
        
        report = await pricing_engine.generate_billing_report(
            period_start=period_start,
            period_end=period_end
        )
        
        assert report.user_id is None
        assert report.total_revenue > 0
        assert report.active_subscriptions > 0
        assert report.new_subscriptions > 0
        assert report.churn_rate >= 0
        assert report.average_revenue_per_user > 0
    
    @pytest.mark.asyncio
    async def test_process_payment_success(self, pricing_engine):
        """Test successful payment processing."""
        payment = await pricing_engine.process_payment(
            user_id="test_user_123",
            amount=49.99,
            payment_method_id="pm_test_123",
            description="Monthly subscription"
        )
        
        assert payment.user_id == "test_user_123"
        assert payment.amount == 49.99
        assert payment.currency == "USD"
        assert payment.provider == "stripe"
        assert payment.provider_payment_id.startswith("pi_")
        # Note: Status might be COMPLETED or FAILED due to random simulation
    
    @pytest.mark.asyncio
    async def test_create_payment_method(self, pricing_engine):
        """Test payment method creation."""
        card_details = {
            "last_four": "4242",
            "brand": "visa",
            "exp_month": 12,
            "exp_year": 2025
        }
        
        payment_method = await pricing_engine.create_payment_method(
            user_id="test_user_123",
            card_details=card_details
        )
        
        assert payment_method.user_id == "test_user_123"
        assert payment_method.type == "credit_card"
        assert payment_method.provider == "stripe"
        assert payment_method.last_four == "4242"
        assert payment_method.brand == "visa"
        assert payment_method.is_verified is True
    
    @pytest.mark.asyncio
    async def test_calculate_tier_upgrade_cost(self, pricing_engine, sample_subscription):
        """Test tier upgrade cost calculation."""
        result = await pricing_engine.calculate_tier_upgrade_cost(
            current_subscription=sample_subscription,
            target_plan_id="enterprise",
            prorate=True
        )
        
        assert result["current_plan"] == "Professional"
        assert result["target_plan"] == "Enterprise"
        assert result["upgrade_cost"] >= 0
        assert result["prorated"] is True
        assert result["billing_cycle"] == "monthly"
        assert result["currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_apply_discount_code_valid(self, pricing_engine):
        """Test applying valid discount code."""
        result = await pricing_engine.apply_discount_code(
            code="WELCOME10",
            user_id="test_user_123",
            invoice_amount=100.0
        )
        
        assert result["valid"] is True
        assert result["code"] == "WELCOME10"
        assert result["discount_type"] == "percentage"
        assert result["discount_amount"] == 10.0  # 10% of 100
        assert result["final_amount"] == 90.0
    
    @pytest.mark.asyncio
    async def test_apply_discount_code_invalid(self, pricing_engine):
        """Test applying invalid discount code."""
        result = await pricing_engine.apply_discount_code(
            code="INVALID_CODE",
            user_id="test_user_123",
            invoice_amount=100.0
        )
        
        assert result["valid"] is False
        assert result["error"] == "Invalid discount code"
        assert result["discount_amount"] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_usage_forecast(self, pricing_engine):
        """Test usage forecasting."""
        usage_history = [
            {"total_cost": 50.0, "generations": 100},
            {"total_cost": 55.0, "generations": 110},
            {"total_cost": 60.0, "generations": 120},
            {"total_cost": 65.0, "generations": 130},
            {"total_cost": 70.0, "generations": 140},
            {"total_cost": 75.0, "generations": 150}
        ]
        
        forecast = await pricing_engine.get_usage_forecast(
            user_id="test_user_123",
            usage_history=usage_history,
            forecast_months=3
        )
        
        assert forecast["user_id"] == "test_user_123"
        assert forecast["forecast_period_months"] == 3
        assert len(forecast["monthly_forecasts"]) == 3
        assert forecast["cost_trend_monthly"] > 0  # Increasing trend
        assert forecast["total_forecasted_cost"] > 0
        assert len(forecast["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_pricing_recommendations(self, pricing_engine):
        """Test pricing recommendations."""
        usage_history = [
            {"total_cost": 80.0, "generations": 200},
            {"total_cost": 85.0, "generations": 210},
            {"total_cost": 90.0, "generations": 220}
        ]
        
        recommendations = await pricing_engine.get_pricing_recommendations(
            user_tier=PricingTier.BASIC,
            usage_history=usage_history,
            current_plan="basic"
        )
        
        assert recommendations["current_avg_monthly_cost"] == 85.0
        assert recommendations["current_avg_generations"] == 210.0
        assert len(recommendations["recommendations"]) > 0
        assert recommendations["analysis_period_months"] == 3
    
    @pytest.mark.asyncio
    async def test_validate_pricing_rules(self, pricing_engine):
        """Test pricing rules validation."""
        validation = await pricing_engine.validate_pricing_rules()
        
        assert validation["valid"] is True
        assert validation["rules_checked"] > 0
        assert len(validation["errors"]) == 0
        # May have warnings, which is acceptable
    
    @pytest.mark.asyncio
    async def test_invalid_resource_type(self, pricing_engine):
        """Test error handling for invalid resource type."""
        with pytest.raises(ValueError, match="No pricing rule found"):
            await pricing_engine.calculate_usage_cost(
                user_tier=PricingTier.BASIC,
                resource_type="invalid_resource",
                quantity=10.0
            )
    
    @pytest.mark.asyncio
    async def test_invalid_subscription_plan(self, pricing_engine):
        """Test error handling for invalid subscription plan."""
        plan = await pricing_engine.get_subscription_plan("nonexistent_plan")
        assert plan is None
        
        with pytest.raises(ValueError, match="Subscription plan not found"):
            await pricing_engine.calculate_subscription_cost(
                plan_id="nonexistent_plan",
                billing_cycle=BillingCycle.MONTHLY
            )


if __name__ == "__main__":
    pytest.main([__file__])