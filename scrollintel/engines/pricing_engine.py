"""
Pricing engine for visual content generation billing system.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
import uuid
import asyncio

from ..models.billing_pricing_models import (
    PricingTier, BillingCycle, PaymentStatus, InvoiceStatus,
    PricingRule, SubscriptionPlan, UserSubscription, Invoice,
    InvoiceLineItem, Payment, PaymentMethod, CostOptimizationAlert,
    BillingReport, TaxConfiguration
)
from ..models.usage_tracking_models import GenerationType, ResourceType, GenerationUsage


class PricingEngine:
    """
    Comprehensive pricing engine with tiered pricing models, volume discounts,
    and cost optimization recommendations.
    """
    
    def __init__(self, storage_backend=None):
        self.logger = logging.getLogger(__name__)
        self.storage = storage_backend or {}
        self.pricing_rules = self._initialize_pricing_rules()
        self.subscription_plans = self._initialize_subscription_plans()
        self.tax_configurations = self._initialize_tax_configurations()
        
    def _initialize_pricing_rules(self) -> Dict[str, PricingRule]:
        """Initialize default pricing rules."""
        rules = {}
        
        # GPU pricing with tier multipliers
        gpu_rule = PricingRule(
            name="GPU Compute Pricing",
            description="Pricing for GPU computation time",
            resource_type="gpu_seconds",
            base_price=0.0,
            unit_price=0.05,  # $0.05 per GPU second
            tier_multipliers={
                PricingTier.FREE: 1.5,      # 50% markup for free tier
                PricingTier.BASIC: 1.0,     # Standard pricing
                PricingTier.PROFESSIONAL: 0.8,  # 20% discount
                PricingTier.ENTERPRISE: 0.6,    # 40% discount
                PricingTier.CUSTOM: 0.5     # 50% discount
            },
            volume_tiers=[
                {"min_quantity": 0, "max_quantity": 100, "discount": 0.0},
                {"min_quantity": 100, "max_quantity": 500, "discount": 0.1},
                {"min_quantity": 500, "max_quantity": 2000, "discount": 0.2},
                {"min_quantity": 2000, "max_quantity": float('inf'), "discount": 0.3}
            ],
            peak_hours_multiplier=1.2,
            off_peak_hours_multiplier=0.8
        )
        rules["gpu_seconds"] = gpu_rule
        
        # API calls pricing
        api_rule = PricingRule(
            name="API Calls Pricing",
            description="Pricing for API calls",
            resource_type="api_calls",
            unit_price=0.001,  # $0.001 per API call
            tier_multipliers={
                PricingTier.FREE: 2.0,
                PricingTier.BASIC: 1.0,
                PricingTier.PROFESSIONAL: 0.8,
                PricingTier.ENTERPRISE: 0.5,
                PricingTier.CUSTOM: 0.3
            },
            volume_tiers=[
                {"min_quantity": 0, "max_quantity": 1000, "discount": 0.0},
                {"min_quantity": 1000, "max_quantity": 10000, "discount": 0.15},
                {"min_quantity": 10000, "max_quantity": float('inf'), "discount": 0.25}
            ]
        )
        rules["api_calls"] = api_rule
        
        # Storage pricing
        storage_rule = PricingRule(
            name="Storage Pricing",
            description="Pricing for data storage",
            resource_type="storage_gb",
            unit_price=0.02,  # $0.02 per GB
            tier_multipliers={
                PricingTier.FREE: 1.0,
                PricingTier.BASIC: 1.0,
                PricingTier.PROFESSIONAL: 0.9,
                PricingTier.ENTERPRISE: 0.7,
                PricingTier.CUSTOM: 0.5
            }
        )
        rules["storage_gb"] = storage_rule
        
        # Bandwidth pricing
        bandwidth_rule = PricingRule(
            name="Bandwidth Pricing",
            description="Pricing for data transfer",
            resource_type="bandwidth_gb",
            unit_price=0.01,  # $0.01 per GB
            tier_multipliers={
                PricingTier.FREE: 1.0,
                PricingTier.BASIC: 1.0,
                PricingTier.PROFESSIONAL: 0.8,
                PricingTier.ENTERPRISE: 0.6,
                PricingTier.CUSTOM: 0.4
            }
        )
        rules["bandwidth_gb"] = bandwidth_rule
        
        return rules
    
    def _initialize_subscription_plans(self) -> Dict[str, SubscriptionPlan]:
        """Initialize default subscription plans."""
        plans = {}
        
        # Free plan
        free_plan = SubscriptionPlan(
            name="Free",
            description="Basic access with limited usage",
            tier=PricingTier.FREE,
            monthly_price=0.0,
            yearly_price=0.0,
            included_image_generations=10,
            included_video_generations=0,
            included_enhancement_operations=5,
            included_gpu_seconds=50.0,
            included_storage_gb=1.0,
            included_bandwidth_gb=5.0,
            overage_pricing={
                "image_generations": 0.10,
                "video_generations": 1.00,
                "enhancement_operations": 0.05,
                "gpu_seconds": 0.075,
                "storage_gb": 0.02,
                "bandwidth_gb": 0.01
            },
            features=["Basic image generation", "Standard models", "Community support"],
            max_concurrent_generations=1,
            max_resolution=(1024, 1024),
            max_video_duration=10.0,
            max_batch_size=1
        )
        plans["free"] = free_plan
        
        # Basic plan
        basic_plan = SubscriptionPlan(
            name="Basic",
            description="Perfect for individuals and small projects",
            tier=PricingTier.BASIC,
            monthly_price=19.99,
            yearly_price=199.99,  # 2 months free
            included_image_generations=100,
            included_video_generations=5,
            included_enhancement_operations=50,
            included_gpu_seconds=500.0,
            included_storage_gb=10.0,
            included_bandwidth_gb=50.0,
            overage_pricing={
                "image_generations": 0.08,
                "video_generations": 0.80,
                "enhancement_operations": 0.04,
                "gpu_seconds": 0.05,
                "storage_gb": 0.02,
                "bandwidth_gb": 0.01
            },
            features=[
                "All image generation models",
                "Basic video generation",
                "Image enhancement tools",
                "Email support",
                "Higher resolution support"
            ],
            max_concurrent_generations=2,
            max_resolution=(2048, 2048),
            max_video_duration=30.0,
            max_batch_size=5
        )
        plans["basic"] = basic_plan
        
        # Professional plan
        professional_plan = SubscriptionPlan(
            name="Professional",
            description="For professionals and growing businesses",
            tier=PricingTier.PROFESSIONAL,
            monthly_price=49.99,
            yearly_price=499.99,
            included_image_generations=500,
            included_video_generations=25,
            included_enhancement_operations=200,
            included_gpu_seconds=2000.0,
            included_storage_gb=50.0,
            included_bandwidth_gb=200.0,
            overage_pricing={
                "image_generations": 0.06,
                "video_generations": 0.60,
                "enhancement_operations": 0.03,
                "gpu_seconds": 0.04,
                "storage_gb": 0.018,
                "bandwidth_gb": 0.008
            },
            features=[
                "All generation models",
                "Advanced video generation",
                "Batch processing",
                "Priority processing",
                "API access",
                "Priority support",
                "Advanced analytics"
            ],
            max_concurrent_generations=5,
            priority_processing=True,
            advanced_models_access=True,
            api_access=True,
            max_resolution=(4096, 4096),
            max_video_duration=120.0,
            max_batch_size=20
        )
        plans["professional"] = professional_plan
        
        # Enterprise plan
        enterprise_plan = SubscriptionPlan(
            name="Enterprise",
            description="For large organizations with high-volume needs",
            tier=PricingTier.ENTERPRISE,
            monthly_price=199.99,
            yearly_price=1999.99,
            included_image_generations=2500,
            included_video_generations=100,
            included_enhancement_operations=1000,
            included_gpu_seconds=10000.0,
            included_storage_gb=500.0,
            included_bandwidth_gb=1000.0,
            overage_pricing={
                "image_generations": 0.04,
                "video_generations": 0.40,
                "enhancement_operations": 0.02,
                "gpu_seconds": 0.03,
                "storage_gb": 0.014,
                "bandwidth_gb": 0.006
            },
            features=[
                "All premium models",
                "Unlimited concurrent generations",
                "Custom model training",
                "Dedicated support",
                "SLA guarantees",
                "Advanced security",
                "Custom integrations",
                "White-label options"
            ],
            max_concurrent_generations=50,
            priority_processing=True,
            advanced_models_access=True,
            api_access=True,
            max_resolution=(8192, 8192),
            max_video_duration=600.0,
            max_batch_size=100
        )
        plans["enterprise"] = enterprise_plan
        
        return plans
    
    def _initialize_tax_configurations(self) -> Dict[str, TaxConfiguration]:
        """Initialize tax configurations for different regions."""
        configs = {}
        
        # US tax configurations
        configs["US"] = TaxConfiguration(
            country_code="US",
            tax_name="Sales Tax",
            tax_rate=0.08,  # 8% average
            applies_to_subscriptions=True,
            applies_to_usage=True
        )
        
        # EU VAT
        configs["EU"] = TaxConfiguration(
            country_code="EU",
            tax_name="VAT",
            tax_rate=0.20,  # 20% VAT
            applies_to_subscriptions=True,
            applies_to_usage=True
        )
        
        return configs
    
    async def calculate_usage_cost(
        self,
        user_tier: PricingTier,
        resource_type: str,
        quantity: float,
        timestamp: Optional[datetime] = None,
        user_monthly_usage: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate cost for resource usage with tier and volume discounts."""
        if resource_type not in self.pricing_rules:
            raise ValueError(f"No pricing rule found for resource type: {resource_type}")
        
        rule = self.pricing_rules[resource_type]
        timestamp = timestamp or datetime.utcnow()
        user_monthly_usage = user_monthly_usage or {}
        
        # Base unit price
        base_unit_price = rule.unit_price
        
        # Apply tier multiplier
        tier_multiplier = rule.tier_multipliers.get(user_tier, 1.0)
        tier_adjusted_price = base_unit_price * tier_multiplier
        
        # Apply time-based pricing
        hour = timestamp.hour
        if hour in rule.peak_hours:
            time_multiplier = rule.peak_hours_multiplier
        else:
            time_multiplier = rule.off_peak_hours_multiplier
        
        time_adjusted_price = tier_adjusted_price * time_multiplier
        
        # Apply volume discount based on monthly usage
        current_monthly_usage = user_monthly_usage.get(resource_type, 0.0)
        total_monthly_usage = current_monthly_usage + quantity
        
        volume_discount = 0.0
        for tier in rule.volume_tiers:
            if tier["min_quantity"] <= total_monthly_usage <= tier["max_quantity"]:
                volume_discount = tier["discount"]
                break
        
        volume_adjusted_price = time_adjusted_price * (1 - volume_discount)
        
        # Calculate final cost
        total_cost = quantity * volume_adjusted_price
        
        return {
            "resource_type": resource_type,
            "quantity": quantity,
            "base_unit_price": base_unit_price,
            "tier_multiplier": tier_multiplier,
            "time_multiplier": time_multiplier,
            "volume_discount": volume_discount,
            "final_unit_price": volume_adjusted_price,
            "total_cost": total_cost,
            "currency": rule.currency,
            "calculation_timestamp": timestamp.isoformat()
        }
    
    async def calculate_generation_cost(
        self,
        user_tier: PricingTier,
        generation_usage: GenerationUsage,
        user_monthly_usage: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate total cost for a generation including all resources."""
        total_cost = 0.0
        cost_breakdown = []
        
        for resource in generation_usage.resources:
            resource_cost = await self.calculate_usage_cost(
                user_tier=user_tier,
                resource_type=resource.resource_type.value,
                quantity=resource.amount,
                timestamp=resource.timestamp,
                user_monthly_usage=user_monthly_usage
            )
            
            cost_breakdown.append(resource_cost)
            total_cost += resource_cost["total_cost"]
        
        return {
            "generation_id": generation_usage.id,
            "generation_type": generation_usage.generation_type.value,
            "model_used": generation_usage.model_used,
            "total_cost": total_cost,
            "cost_breakdown": cost_breakdown,
            "currency": "USD"
        }
    
    async def get_subscription_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Get subscription plan by ID."""
        return self.subscription_plans.get(plan_id)
    
    async def calculate_subscription_cost(
        self,
        plan_id: str,
        billing_cycle: BillingCycle,
        tax_region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate subscription cost including taxes."""
        plan = await self.get_subscription_plan(plan_id)
        if not plan:
            raise ValueError(f"Subscription plan not found: {plan_id}")
        
        # Base cost based on billing cycle
        if billing_cycle == BillingCycle.MONTHLY:
            base_cost = plan.monthly_price
        elif billing_cycle == BillingCycle.YEARLY:
            base_cost = plan.yearly_price
        else:
            raise ValueError(f"Unsupported billing cycle: {billing_cycle}")
        
        # Calculate tax
        tax_amount = 0.0
        tax_rate = 0.0
        if tax_region and tax_region in self.tax_configurations:
            tax_config = self.tax_configurations[tax_region]
            if tax_config.applies_to_subscriptions:
                tax_rate = tax_config.tax_rate
                tax_amount = base_cost * tax_rate
        
        total_cost = base_cost + tax_amount
        
        return {
            "plan_id": plan_id,
            "plan_name": plan.name,
            "billing_cycle": billing_cycle.value,
            "base_cost": base_cost,
            "tax_rate": tax_rate,
            "tax_amount": tax_amount,
            "total_cost": total_cost,
            "currency": plan.currency
        }
    
    async def check_subscription_quotas(
        self,
        subscription: UserSubscription,
        resource_type: str,
        requested_quantity: float
    ) -> Dict[str, Any]:
        """Check if user has sufficient quota for requested resource usage."""
        plan = await self.get_subscription_plan(subscription.plan_id)
        if not plan:
            return {"allowed": False, "reason": "Invalid subscription plan"}
        
        # Get quota limits
        quota_mapping = {
            "image_generations": plan.included_image_generations,
            "video_generations": plan.included_video_generations,
            "enhancement_operations": plan.included_enhancement_operations,
            "gpu_seconds": plan.included_gpu_seconds,
            "storage_gb": plan.included_storage_gb,
            "bandwidth_gb": plan.included_bandwidth_gb
        }
        
        if resource_type not in quota_mapping:
            return {"allowed": True, "reason": "No quota limit for this resource"}
        
        quota_limit = quota_mapping[resource_type]
        current_usage = subscription.current_period_usage.get(resource_type, 0.0)
        remaining_quota = quota_limit - current_usage
        
        if requested_quantity <= remaining_quota:
            return {
                "allowed": True,
                "quota_limit": quota_limit,
                "current_usage": current_usage,
                "remaining_quota": remaining_quota,
                "requested_quantity": requested_quantity
            }
        else:
            overage_quantity = requested_quantity - remaining_quota
            overage_cost = overage_quantity * plan.overage_pricing.get(resource_type, 0.0)
            
            return {
                "allowed": True,  # Allow with overage charges
                "quota_exceeded": True,
                "quota_limit": quota_limit,
                "current_usage": current_usage,
                "remaining_quota": remaining_quota,
                "requested_quantity": requested_quantity,
                "overage_quantity": overage_quantity,
                "overage_cost": overage_cost,
                "currency": plan.currency
            }
    
    async def generate_invoice(
        self,
        user_id: str,
        subscription_id: Optional[str] = None,
        usage_charges: Optional[List[Dict[str, Any]]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        tax_region: Optional[str] = None
    ) -> Invoice:
        """Generate invoice for subscription and usage charges."""
        period_start = period_start or datetime.utcnow().replace(day=1)
        period_end = period_end or datetime.utcnow()
        
        invoice = Invoice(
            invoice_number=f"INV-{datetime.utcnow().strftime('%Y%m%d')}-{user_id[:8]}",
            user_id=user_id,
            subscription_id=subscription_id,
            period_start=period_start,
            period_end=period_end
        )
        
        # Add subscription charges
        if subscription_id:
            # This would typically fetch subscription details from storage
            subscription_charge = InvoiceLineItem(
                description="Monthly Subscription - Professional Plan",
                quantity=1.0,
                unit_price=49.99,
                total_price=49.99,
                period_start=period_start,
                period_end=period_end
            )
            invoice.line_items.append(subscription_charge)
            invoice.subtotal += subscription_charge.total_price
        
        # Add usage charges
        if usage_charges:
            for charge in usage_charges:
                line_item = InvoiceLineItem(
                    description=charge.get("description", "Usage charges"),
                    quantity=charge.get("quantity", 1.0),
                    unit_price=charge.get("unit_price", 0.0),
                    total_price=charge.get("total_price", 0.0),
                    resource_type=charge.get("resource_type"),
                    generation_ids=charge.get("generation_ids", [])
                )
                invoice.line_items.append(line_item)
                invoice.subtotal += line_item.total_price
        
        # Calculate tax
        if tax_region and tax_region in self.tax_configurations:
            tax_config = self.tax_configurations[tax_region]
            invoice.tax_rate = tax_config.tax_rate
            invoice.tax_amount = invoice.subtotal * tax_config.tax_rate
        
        # Calculate total
        invoice.total_amount = invoice.subtotal + invoice.tax_amount - invoice.discount_amount
        
        return invoice
    
    async def generate_cost_optimization_alerts(
        self,
        user_id: str,
        current_usage: Dict[str, float],
        current_costs: Dict[str, float],
        subscription: Optional[UserSubscription] = None
    ) -> List[CostOptimizationAlert]:
        """Generate cost optimization alerts based on usage patterns."""
        alerts = []
        
        # Check for high overage costs
        if subscription:
            plan = await self.get_subscription_plan(subscription.plan_id)
            if plan:
                total_overage_cost = 0.0
                overage_resources = []
                
                quota_mapping = {
                    "image_generations": plan.included_image_generations,
                    "video_generations": plan.included_video_generations,
                    "gpu_seconds": plan.included_gpu_seconds,
                    "storage_gb": plan.included_storage_gb
                }
                
                for resource, usage in current_usage.items():
                    if resource in quota_mapping:
                        quota = quota_mapping[resource]
                        if usage > quota:
                            overage = usage - quota
                            overage_cost = overage * plan.overage_pricing.get(resource, 0.0)
                            total_overage_cost += overage_cost
                            overage_resources.append(resource)
                
                if total_overage_cost > subscription.price * 0.5:  # Overage > 50% of subscription
                    alerts.append(CostOptimizationAlert(
                        user_id=user_id,
                        alert_type="high_overage_costs",
                        title="High Overage Costs Detected",
                        message=f"Your overage costs (${total_overage_cost:.2f}) are {(total_overage_cost/subscription.price)*100:.0f}% of your subscription fee. Consider upgrading your plan.",
                        severity="high",
                        current_cost=total_overage_cost,
                        potential_savings=total_overage_cost * 0.6,  # Assume 60% savings with upgrade
                        recommendations=[
                            "Upgrade to a higher-tier subscription plan",
                            f"Focus on optimizing usage for: {', '.join(overage_resources)}",
                            "Consider batch processing to reduce costs"
                        ]
                    ))
        
        # Check for inefficient resource usage
        total_cost = sum(current_costs.values())
        gpu_cost = current_costs.get("gpu_seconds", 0.0)
        
        if gpu_cost > total_cost * 0.8:  # GPU costs > 80% of total
            alerts.append(CostOptimizationAlert(
                user_id=user_id,
                alert_type="gpu_cost_optimization",
                title="High GPU Usage Costs",
                message="GPU costs represent a large portion of your bill. Consider optimization strategies.",
                severity="medium",
                current_cost=gpu_cost,
                potential_savings=gpu_cost * 0.25,
                recommendations=[
                    "Use batch processing for multiple generations",
                    "Schedule generations during off-peak hours (20% discount)",
                    "Consider lower-resolution generations when appropriate",
                    "Use more efficient models when quality requirements allow"
                ]
            ))
        
        # Check for plan upgrade opportunities
        if subscription and subscription.tier in [PricingTier.FREE, PricingTier.BASIC]:
            monthly_usage_cost = sum(current_costs.values())
            next_tier_plans = {
                PricingTier.FREE: "basic",
                PricingTier.BASIC: "professional"
            }
            
            next_plan_id = next_tier_plans.get(subscription.tier)
            if next_plan_id:
                next_plan = self.subscription_plans.get(next_plan_id)
                if next_plan and monthly_usage_cost > next_plan.monthly_price * 0.8:
                    alerts.append(CostOptimizationAlert(
                        user_id=user_id,
                        alert_type="plan_upgrade_opportunity",
                        title="Plan Upgrade Recommended",
                        message=f"Your usage costs (${monthly_usage_cost:.2f}) are close to the {next_plan.name} plan price (${next_plan.monthly_price:.2f}). Upgrading would provide better value.",
                        severity="medium",
                        current_cost=monthly_usage_cost,
                        projected_cost=next_plan.monthly_price,
                        potential_savings=monthly_usage_cost - next_plan.monthly_price,
                        recommendations=[
                            f"Upgrade to {next_plan.name} plan",
                            "Get included quotas and lower overage rates",
                            "Access to premium features and priority support"
                        ]
                    ))
        
        return alerts
    
    async def generate_billing_report(
        self,
        period_start: datetime,
        period_end: datetime,
        user_id: Optional[str] = None
    ) -> BillingReport:
        """Generate comprehensive billing report."""
        report = BillingReport(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end
        )
        
        # This would typically aggregate data from storage
        # For demo purposes, we'll use sample data
        
        if user_id:
            # User-specific report
            report.total_revenue = 149.97  # Sample data
            report.subscription_revenue = 49.99
            report.usage_revenue = 99.98
            report.total_generations = 150
            report.total_gpu_seconds = 500.0
            report.total_storage_gb = 25.0
            report.active_subscriptions = 1
        else:
            # System-wide report
            report.total_revenue = 15000.00
            report.subscription_revenue = 8000.00
            report.usage_revenue = 7000.00
            report.total_generations = 5000
            report.total_gpu_seconds = 25000.0
            report.total_storage_gb = 1000.0
            report.active_subscriptions = 200
            report.new_subscriptions = 25
            report.cancelled_subscriptions = 5
            report.churn_rate = 0.025  # 2.5%
            report.average_revenue_per_user = 75.00
            report.cost_per_generation = 3.00
        
        return report
    
    async def get_pricing_recommendations(
        self,
        user_tier: PricingTier,
        usage_history: List[Dict[str, Any]],
        current_plan: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get personalized pricing recommendations based on usage history."""
        if not usage_history:
            return {"recommendations": [], "message": "Insufficient usage history"}
        
        # Analyze usage patterns
        avg_monthly_cost = sum(usage["total_cost"] for usage in usage_history) / len(usage_history)
        avg_generations = sum(usage.get("generations", 0) for usage in usage_history) / len(usage_history)
        
        recommendations = []
        
        # Compare with subscription plans
        for plan_id, plan in self.subscription_plans.items():
            if plan.tier == user_tier:
                continue
                
            monthly_savings = avg_monthly_cost - plan.monthly_price
            yearly_savings = monthly_savings * 12
            
            if monthly_savings > 0:
                recommendations.append({
                    "plan_id": plan_id,
                    "plan_name": plan.name,
                    "monthly_price": plan.monthly_price,
                    "monthly_savings": monthly_savings,
                    "yearly_savings": yearly_savings,
                    "recommendation_reason": f"Save ${monthly_savings:.2f}/month based on your usage patterns",
                    "features": plan.features[:3]  # Top 3 features
                })
        
        # Sort by savings
        recommendations.sort(key=lambda x: x["monthly_savings"], reverse=True)
        
        return {
            "current_avg_monthly_cost": avg_monthly_cost,
            "current_avg_generations": avg_generations,
            "recommendations": recommendations[:3],  # Top 3 recommendations
            "analysis_period_months": len(usage_history)
        }
    
    async def process_payment(
        self,
        user_id: str,
        amount: float,
        currency: str = "USD",
        payment_method_id: str = "",
        invoice_id: Optional[str] = None,
        description: str = ""
    ) -> Payment:
        """Process payment for invoice or subscription."""
        payment = Payment(
            user_id=user_id,
            invoice_id=invoice_id,
            payment_method_id=payment_method_id,
            amount=amount,
            currency=currency,
            description=description,
            provider="stripe",  # Default provider
            provider_payment_id=f"pi_{uuid.uuid4().hex[:24]}"  # Mock payment intent ID
        )
        
        try:
            # In production, this would integrate with actual payment processor
            # For now, we'll simulate payment processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Simulate payment success (90% success rate)
            import random
            if random.random() < 0.9:
                payment.status = PaymentStatus.COMPLETED
                payment.processed_at = datetime.utcnow()
                payment.completed_at = datetime.utcnow()
                payment.provider_charge_id = f"ch_{uuid.uuid4().hex[:24]}"
            else:
                payment.status = PaymentStatus.FAILED
                payment.failure_reason = "Card declined"
                payment.processed_at = datetime.utcnow()
            
            self.logger.info(f"Payment processed: {payment.id} - Status: {payment.status.value}")
            
        except Exception as e:
            payment.status = PaymentStatus.FAILED
            payment.failure_reason = str(e)
            payment.processed_at = datetime.utcnow()
            self.logger.error(f"Payment processing failed: {e}")
        
        return payment
    
    async def create_payment_method(
        self,
        user_id: str,
        payment_type: str = "credit_card",
        provider: str = "stripe",
        card_details: Optional[Dict[str, Any]] = None
    ) -> PaymentMethod:
        """Create and store payment method."""
        payment_method = PaymentMethod(
            user_id=user_id,
            type=payment_type,
            provider=provider,
            provider_id=f"pm_{uuid.uuid4().hex[:24]}",  # Mock provider ID
            is_verified=True  # Mock verification
        )
        
        if card_details:
            payment_method.last_four = card_details.get("last_four")
            payment_method.brand = card_details.get("brand")
            payment_method.exp_month = card_details.get("exp_month")
            payment_method.exp_year = card_details.get("exp_year")
        
        self.logger.info(f"Payment method created: {payment_method.id}")
        return payment_method
    
    async def calculate_tier_upgrade_cost(
        self,
        current_subscription: UserSubscription,
        target_plan_id: str,
        prorate: bool = True
    ) -> Dict[str, Any]:
        """Calculate cost for upgrading subscription tier."""
        current_plan = await self.get_subscription_plan(current_subscription.plan_id)
        target_plan = await self.get_subscription_plan(target_plan_id)
        
        if not current_plan or not target_plan:
            raise ValueError("Invalid subscription plan")
        
        if target_plan.tier.value <= current_plan.tier.value:
            raise ValueError("Target plan must be a higher tier")
        
        # Calculate prorated costs
        if prorate:
            days_remaining = (current_subscription.end_date - datetime.utcnow()).days
            days_in_cycle = 30 if current_subscription.billing_cycle == BillingCycle.MONTHLY else 365
            
            # Refund for unused portion of current plan
            unused_current_cost = (current_subscription.price * days_remaining) / days_in_cycle
            
            # Cost for remaining period on new plan
            if current_subscription.billing_cycle == BillingCycle.MONTHLY:
                new_period_cost = (target_plan.monthly_price * days_remaining) / days_in_cycle
            else:
                new_period_cost = (target_plan.yearly_price * days_remaining) / days_in_cycle
            
            upgrade_cost = new_period_cost - unused_current_cost
        else:
            # Full cost of new plan
            upgrade_cost = target_plan.monthly_price if current_subscription.billing_cycle == BillingCycle.MONTHLY else target_plan.yearly_price
        
        return {
            "current_plan": current_plan.name,
            "target_plan": target_plan.name,
            "upgrade_cost": max(0, upgrade_cost),  # Never negative
            "prorated": prorate,
            "billing_cycle": current_subscription.billing_cycle.value,
            "effective_date": datetime.utcnow().isoformat(),
            "currency": target_plan.currency
        }
    
    async def apply_discount_code(
        self,
        code: str,
        user_id: str,
        invoice_amount: float
    ) -> Dict[str, Any]:
        """Apply discount code to invoice."""
        # Mock discount codes
        discount_codes = {
            "WELCOME10": {"type": "percentage", "value": 0.10, "description": "10% off first invoice"},
            "SAVE20": {"type": "percentage", "value": 0.20, "description": "20% off"},
            "FLAT50": {"type": "fixed", "value": 50.0, "description": "$50 off"},
            "NEWUSER": {"type": "percentage", "value": 0.15, "description": "15% off for new users"}
        }
        
        if code not in discount_codes:
            return {
                "valid": False,
                "error": "Invalid discount code",
                "discount_amount": 0.0
            }
        
        discount = discount_codes[code]
        
        if discount["type"] == "percentage":
            discount_amount = invoice_amount * discount["value"]
        else:  # fixed
            discount_amount = min(discount["value"], invoice_amount)
        
        return {
            "valid": True,
            "code": code,
            "description": discount["description"],
            "discount_type": discount["type"],
            "discount_value": discount["value"],
            "discount_amount": discount_amount,
            "final_amount": invoice_amount - discount_amount
        }
    
    async def get_usage_forecast(
        self,
        user_id: str,
        usage_history: List[Dict[str, Any]],
        forecast_months: int = 3
    ) -> Dict[str, Any]:
        """Generate usage and cost forecast based on historical data."""
        if len(usage_history) < 2:
            return {"error": "Insufficient data for forecasting"}
        
        # Simple linear trend analysis
        monthly_costs = [usage["total_cost"] for usage in usage_history[-6:]]  # Last 6 months
        monthly_generations = [usage.get("generations", 0) for usage in usage_history[-6:]]
        
        # Calculate trends
        if len(monthly_costs) >= 2:
            cost_trend = (monthly_costs[-1] - monthly_costs[0]) / len(monthly_costs)
            generation_trend = (monthly_generations[-1] - monthly_generations[0]) / len(monthly_generations)
        else:
            cost_trend = 0
            generation_trend = 0
        
        # Generate forecasts
        current_cost = monthly_costs[-1] if monthly_costs else 0
        current_generations = monthly_generations[-1] if monthly_generations else 0
        
        forecasts = []
        for month in range(1, forecast_months + 1):
            forecasted_cost = max(0, current_cost + (cost_trend * month))
            forecasted_generations = max(0, current_generations + (generation_trend * month))
            
            forecasts.append({
                "month": month,
                "forecasted_cost": round(forecasted_cost, 2),
                "forecasted_generations": int(forecasted_generations),
                "confidence": max(0.5, 1.0 - (month * 0.1))  # Decreasing confidence
            })
        
        total_forecasted_cost = sum(f["forecasted_cost"] for f in forecasts)
        
        return {
            "user_id": user_id,
            "forecast_period_months": forecast_months,
            "historical_data_points": len(usage_history),
            "cost_trend_monthly": round(cost_trend, 2),
            "generation_trend_monthly": int(generation_trend),
            "total_forecasted_cost": round(total_forecasted_cost, 2),
            "monthly_forecasts": forecasts,
            "recommendations": self._generate_forecast_recommendations(forecasts, current_cost)
        }
    
    def _generate_forecast_recommendations(
        self,
        forecasts: List[Dict[str, Any]],
        current_cost: float
    ) -> List[str]:
        """Generate recommendations based on usage forecast."""
        recommendations = []
        
        avg_forecasted_cost = sum(f["forecasted_cost"] for f in forecasts) / len(forecasts)
        
        if avg_forecasted_cost > current_cost * 1.5:
            recommendations.append("Consider upgrading to a higher-tier plan to reduce overage costs")
            recommendations.append("Review usage patterns to identify optimization opportunities")
        
        if avg_forecasted_cost < current_cost * 0.7:
            recommendations.append("Consider downgrading to a lower-tier plan to save money")
            recommendations.append("Your usage appears to be decreasing")
        
        # Check for seasonal patterns
        cost_variance = max(forecasts, key=lambda x: x["forecasted_cost"])["forecasted_cost"] - \
                       min(forecasts, key=lambda x: x["forecasted_cost"])["forecasted_cost"]
        
        if cost_variance > avg_forecasted_cost * 0.3:
            recommendations.append("Consider flexible billing options due to variable usage patterns")
        
        return recommendations
    
    async def validate_pricing_rules(self) -> Dict[str, Any]:
        """Validate all pricing rules for consistency and correctness."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "rules_checked": len(self.pricing_rules)
        }
        
        for rule_id, rule in self.pricing_rules.items():
            # Check for negative prices
            if rule.unit_price < 0:
                validation_results["errors"].append(f"Rule {rule_id}: Negative unit price")
                validation_results["valid"] = False
            
            # Check tier multipliers
            for tier, multiplier in rule.tier_multipliers.items():
                if multiplier <= 0:
                    validation_results["errors"].append(f"Rule {rule_id}: Invalid tier multiplier for {tier}")
                    validation_results["valid"] = False
            
            # Check volume tiers
            for i, tier in enumerate(rule.volume_tiers):
                if tier["discount"] < 0 or tier["discount"] > 1:
                    validation_results["errors"].append(f"Rule {rule_id}: Invalid discount in volume tier {i}")
                    validation_results["valid"] = False
                
                if tier["min_quantity"] > tier["max_quantity"]:
                    validation_results["errors"].append(f"Rule {rule_id}: Invalid quantity range in volume tier {i}")
                    validation_results["valid"] = False
            
            # Check time multipliers
            if rule.peak_hours_multiplier <= 0 or rule.off_peak_hours_multiplier <= 0:
                validation_results["errors"].append(f"Rule {rule_id}: Invalid time multipliers")
                validation_results["valid"] = False
            
            # Warnings for unusual configurations
            if rule.peak_hours_multiplier < rule.off_peak_hours_multiplier:
                validation_results["warnings"].append(f"Rule {rule_id}: Peak hours cheaper than off-peak")
        
        return validation_results