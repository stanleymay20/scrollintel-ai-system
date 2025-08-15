"""
Demo script for the pricing and billing engine.
Showcases tiered pricing models, cost optimization, and billing workflows.
"""

import asyncio
from datetime import datetime, timedelta
from scrollintel.engines.pricing_engine import PricingEngine
from scrollintel.models.billing_pricing_models import (
    PricingTier, BillingCycle, UserSubscription, PaymentMethod
)
from scrollintel.models.usage_tracking_models import (
    GenerationType, ResourceType, GenerationUsage, ResourceUsage
)


async def demo_pricing_calculations():
    """Demonstrate pricing calculations with different tiers and scenarios."""
    print("🔢 PRICING CALCULATIONS DEMO")
    print("=" * 50)
    
    pricing_engine = PricingEngine()
    
    # Test different pricing tiers
    tiers = [PricingTier.FREE, PricingTier.BASIC, PricingTier.PROFESSIONAL, PricingTier.ENTERPRISE]
    
    print("\n📊 GPU Pricing Comparison Across Tiers:")
    for tier in tiers:
        result = await pricing_engine.calculate_usage_cost(
            user_tier=tier,
            resource_type="gpu_seconds",
            quantity=100.0
        )
        print(f"  {tier.value.upper():12} | ${result['total_cost']:.2f} | "
              f"Tier Multiplier: {result['tier_multiplier']:.1f}x | "
              f"Time Multiplier: {result['time_multiplier']:.1f}x")
    
    # Test volume discounts
    print("\n📈 Volume Discount Demonstration:")
    quantities = [50, 200, 600, 2500]
    for quantity in quantities:
        result = await pricing_engine.calculate_usage_cost(
            user_tier=PricingTier.PROFESSIONAL,
            resource_type="gpu_seconds",
            quantity=quantity,
            user_monthly_usage={"gpu_seconds": quantity}
        )
        print(f"  {quantity:4} GPU seconds | ${result['total_cost']:6.2f} | "
              f"Volume Discount: {result['volume_discount']*100:4.1f}% | "
              f"Unit Price: ${result['final_unit_price']:.4f}")
    
    # Test peak vs off-peak pricing
    print("\n⏰ Peak vs Off-Peak Pricing:")
    peak_time = datetime.now().replace(hour=12)  # Noon
    off_peak_time = datetime.now().replace(hour=22)  # 10 PM
    
    peak_result = await pricing_engine.calculate_usage_cost(
        user_tier=PricingTier.PROFESSIONAL,
        resource_type="gpu_seconds",
        quantity=100.0,
        timestamp=peak_time
    )
    
    off_peak_result = await pricing_engine.calculate_usage_cost(
        user_tier=PricingTier.PROFESSIONAL,
        resource_type="gpu_seconds",
        quantity=100.0,
        timestamp=off_peak_time
    )
    
    print(f"  Peak Hours (12 PM):     ${peak_result['total_cost']:.2f}")
    print(f"  Off-Peak Hours (10 PM): ${off_peak_result['total_cost']:.2f}")
    print(f"  Savings with off-peak:  ${peak_result['total_cost'] - off_peak_result['total_cost']:.2f} ({((peak_result['total_cost'] - off_peak_result['total_cost']) / peak_result['total_cost'] * 100):.1f}%)")


async def demo_subscription_plans():
    """Demonstrate subscription plan features and pricing."""
    print("\n\n💳 SUBSCRIPTION PLANS DEMO")
    print("=" * 50)
    
    pricing_engine = PricingEngine()
    
    plans = ["free", "basic", "professional", "enterprise"]
    
    print("\n📋 Subscription Plan Comparison:")
    print(f"{'Plan':<12} | {'Monthly':<8} | {'Yearly':<8} | {'Images':<7} | {'Videos':<6} | {'GPU Hrs':<7} | {'Features'}")
    print("-" * 90)
    
    for plan_id in plans:
        plan = await pricing_engine.get_subscription_plan(plan_id)
        if plan:
            gpu_hours = plan.included_gpu_seconds / 3600  # Convert to hours
            features_str = ", ".join(plan.features[:2]) + "..." if len(plan.features) > 2 else ", ".join(plan.features)
            print(f"{plan.name:<12} | ${plan.monthly_price:<7.2f} | ${plan.yearly_price:<7.2f} | "
                  f"{plan.included_image_generations:<7} | {plan.included_video_generations:<6} | "
                  f"{gpu_hours:<7.1f} | {features_str[:30]}")
    
    # Calculate subscription costs with tax
    print("\n💰 Subscription Cost Calculation (with tax):")
    professional_monthly = await pricing_engine.calculate_subscription_cost(
        plan_id="professional",
        billing_cycle=BillingCycle.MONTHLY,
        tax_region="US"
    )
    
    professional_yearly = await pricing_engine.calculate_subscription_cost(
        plan_id="professional",
        billing_cycle=BillingCycle.YEARLY,
        tax_region="US"
    )
    
    print(f"  Professional Monthly: ${professional_monthly['base_cost']:.2f} + "
          f"${professional_monthly['tax_amount']:.2f} tax = ${professional_monthly['total_cost']:.2f}")
    print(f"  Professional Yearly:  ${professional_yearly['base_cost']:.2f} + "
          f"${professional_yearly['tax_amount']:.2f} tax = ${professional_yearly['total_cost']:.2f}")
    print(f"  Yearly Savings: ${(professional_monthly['total_cost'] * 12) - professional_yearly['total_cost']:.2f}")


async def demo_quota_management():
    """Demonstrate quota checking and overage calculations."""
    print("\n\n📊 QUOTA MANAGEMENT DEMO")
    print("=" * 50)
    
    pricing_engine = PricingEngine()
    
    # Create sample subscription
    subscription = UserSubscription(
        user_id="demo_user",
        plan_id="professional",
        plan_name="Professional",
        tier=PricingTier.PROFESSIONAL,
        billing_cycle=BillingCycle.MONTHLY,
        price=49.99,
        current_period_usage={
            "image_generations": 400.0,
            "video_generations": 20.0,
            "gpu_seconds": 1800.0
        },
        remaining_quotas={
            "image_generations": 100.0,
            "video_generations": 5.0,
            "gpu_seconds": 200.0
        }
    )
    
    print(f"\n👤 User: {subscription.user_id}")
    print(f"📋 Plan: {subscription.plan_name} (${subscription.price}/month)")
    print(f"📅 Current Usage:")
    for resource, usage in subscription.current_period_usage.items():
        remaining = subscription.remaining_quotas.get(resource, 0)
        total_quota = usage + remaining
        print(f"  {resource.replace('_', ' ').title():<20}: {usage:6.0f} / {total_quota:6.0f} ({(usage/total_quota*100):5.1f}%)")
    
    # Test quota checks
    print(f"\n🔍 Quota Check Scenarios:")
    
    scenarios = [
        ("image_generations", 50.0, "Within quota"),
        ("image_generations", 150.0, "Exceeds quota"),
        ("video_generations", 10.0, "Exceeds quota"),
        ("gpu_seconds", 100.0, "Within quota")
    ]
    
    for resource, requested, description in scenarios:
        result = await pricing_engine.check_subscription_quotas(
            subscription=subscription,
            resource_type=resource,
            requested_quantity=requested
        )
        
        print(f"  {description:<15} | {resource.replace('_', ' ').title():<20} | "
              f"Requested: {requested:6.0f} | ", end="")
        
        if result.get("quota_exceeded"):
            print(f"Overage: {result['overage_quantity']:6.0f} | "
                  f"Cost: ${result['overage_cost']:.2f}")
        else:
            print(f"Remaining: {result['remaining_quota']:6.0f}")


async def demo_cost_optimization():
    """Demonstrate cost optimization alerts and recommendations."""
    print("\n\n🎯 COST OPTIMIZATION DEMO")
    print("=" * 50)
    
    pricing_engine = PricingEngine()
    
    # Create high-usage scenario
    subscription = UserSubscription(
        user_id="high_usage_user",
        plan_id="basic",
        plan_name="Basic",
        tier=PricingTier.BASIC,
        price=19.99
    )
    
    high_usage = {
        "image_generations": 500.0,  # Way over Basic plan quota
        "video_generations": 20.0,
        "gpu_seconds": 2000.0,
        "storage_gb": 50.0
    }
    
    high_costs = {
        "image_generations": 40.0,  # High overage costs
        "video_generations": 16.0,
        "gpu_seconds": 80.0,
        "storage_gb": 1.0
    }
    
    print(f"\n📊 High Usage Scenario:")
    print(f"  Plan: {subscription.plan_name} (${subscription.price}/month)")
    print(f"  Total Usage Costs: ${sum(high_costs.values()):.2f}")
    print(f"  Cost vs Plan Price: {(sum(high_costs.values()) / subscription.price * 100):.0f}%")
    
    # Generate optimization alerts
    alerts = await pricing_engine.generate_cost_optimization_alerts(
        user_id=subscription.user_id,
        current_usage=high_usage,
        current_costs=high_costs,
        subscription=subscription
    )
    
    print(f"\n🚨 Cost Optimization Alerts ({len(alerts)} found):")
    for i, alert in enumerate(alerts, 1):
        print(f"\n  Alert {i}: {alert.title}")
        print(f"    Severity: {alert.severity.upper()}")
        print(f"    Message: {alert.message}")
        print(f"    Potential Savings: ${alert.potential_savings:.2f}")
        print(f"    Recommendations:")
        for rec in alert.recommendations[:2]:  # Show first 2 recommendations
            print(f"      • {rec}")
    
    # Generate pricing recommendations
    usage_history = [
        {"total_cost": 120.0, "generations": 450},
        {"total_cost": 130.0, "generations": 480},
        {"total_cost": 137.0, "generations": 500}
    ]
    
    recommendations = await pricing_engine.get_pricing_recommendations(
        user_tier=subscription.tier,
        usage_history=usage_history,
        current_plan=subscription.plan_id
    )
    
    print(f"\n💡 Pricing Recommendations:")
    print(f"  Current Avg Monthly Cost: ${recommendations['current_avg_monthly_cost']:.2f}")
    print(f"  Analysis Period: {recommendations['analysis_period_months']} months")
    
    for i, rec in enumerate(recommendations['recommendations'][:2], 1):
        print(f"\n  Recommendation {i}: {rec['plan_name']}")
        print(f"    Monthly Price: ${rec['monthly_price']:.2f}")
        print(f"    Monthly Savings: ${rec['monthly_savings']:.2f}")
        print(f"    Yearly Savings: ${rec['yearly_savings']:.2f}")
        print(f"    Reason: {rec['recommendation_reason']}")


async def demo_billing_workflow():
    """Demonstrate complete billing workflow."""
    print("\n\n💳 BILLING WORKFLOW DEMO")
    print("=" * 50)
    
    pricing_engine = PricingEngine()
    
    # Step 1: Generate invoice
    print("\n📄 Step 1: Generate Invoice")
    
    usage_charges = [
        {
            "description": "GPU Usage - Image Generation",
            "quantity": 200.0,
            "unit_price": 0.04,
            "total_price": 8.00,
            "resource_type": "gpu_seconds"
        },
        {
            "description": "API Calls - Overage",
            "quantity": 2000.0,
            "unit_price": 0.0008,
            "total_price": 1.60,
            "resource_type": "api_calls"
        }
    ]
    
    invoice = await pricing_engine.generate_invoice(
        user_id="demo_billing_user",
        subscription_id="sub_demo_123",
        usage_charges=usage_charges,
        tax_region="US"
    )
    
    print(f"  Invoice Number: {invoice.invoice_number}")
    print(f"  Subtotal: ${invoice.subtotal:.2f}")
    print(f"  Tax ({invoice.tax_rate*100:.0f}%): ${invoice.tax_amount:.2f}")
    print(f"  Total: ${invoice.total_amount:.2f}")
    print(f"  Line Items: {len(invoice.line_items)}")
    
    # Step 2: Apply discount
    print("\n🎫 Step 2: Apply Discount Code")
    
    discount_result = await pricing_engine.apply_discount_code(
        code="SAVE20",
        user_id="demo_billing_user",
        invoice_amount=invoice.total_amount
    )
    
    if discount_result["valid"]:
        print(f"  Discount Code: {discount_result['code']}")
        print(f"  Discount: ${discount_result['discount_amount']:.2f}")
        print(f"  Final Amount: ${discount_result['final_amount']:.2f}")
        invoice.total_amount = discount_result['final_amount']
    
    # Step 3: Create payment method
    print("\n💳 Step 3: Create Payment Method")
    
    payment_method = await pricing_engine.create_payment_method(
        user_id="demo_billing_user",
        card_details={
            "last_four": "4242",
            "brand": "visa",
            "exp_month": 12,
            "exp_year": 2025
        }
    )
    
    print(f"  Payment Method ID: {payment_method.id}")
    print(f"  Card: **** **** **** {payment_method.last_four}")
    print(f"  Brand: {payment_method.brand.upper()}")
    print(f"  Verified: {payment_method.is_verified}")
    
    # Step 4: Process payment
    print("\n💰 Step 4: Process Payment")
    
    payment = await pricing_engine.process_payment(
        user_id="demo_billing_user",
        amount=invoice.total_amount,
        payment_method_id=payment_method.id,
        invoice_id=invoice.id,
        description=f"Payment for {invoice.invoice_number}"
    )
    
    print(f"  Payment ID: {payment.id}")
    print(f"  Amount: ${payment.amount:.2f}")
    print(f"  Status: {payment.status.value.upper()}")
    print(f"  Provider: {payment.provider}")
    
    if payment.status.value == "completed":
        print(f"  ✅ Payment successful!")
        print(f"  Charge ID: {payment.provider_charge_id}")
    else:
        print(f"  ❌ Payment failed: {payment.failure_reason}")


async def demo_usage_forecasting():
    """Demonstrate usage forecasting and budget planning."""
    print("\n\n📈 USAGE FORECASTING DEMO")
    print("=" * 50)
    
    pricing_engine = PricingEngine()
    
    # Create historical usage data with growth trend
    print("\n📊 Historical Usage Data:")
    usage_history = []
    base_cost = 45.0
    base_generations = 120
    
    print(f"{'Month':<8} | {'Cost':<8} | {'Generations':<12} | {'Trend'}")
    print("-" * 45)
    
    for month in range(6):
        monthly_cost = base_cost + (month * 8) + (month * month * 1.5)  # Accelerating growth
        monthly_generations = base_generations + (month * 25) + (month * month * 3)
        
        usage_history.append({
            "total_cost": monthly_cost,
            "generations": monthly_generations,
            "month": month + 1
        })
        
        if month == 0:
            trend = "Baseline"
        else:
            prev_cost = usage_history[month-1]["total_cost"]
            growth = ((monthly_cost - prev_cost) / prev_cost) * 100
            trend = f"+{growth:.1f}%"
        
        print(f"Month {month+1:<2} | ${monthly_cost:<7.2f} | {monthly_generations:<12.0f} | {trend}")
    
    # Generate forecast
    forecast = await pricing_engine.get_usage_forecast(
        user_id="forecast_demo_user",
        usage_history=usage_history,
        forecast_months=3
    )
    
    print(f"\n🔮 3-Month Forecast:")
    print(f"  Cost Trend: ${forecast['cost_trend_monthly']:.2f}/month")
    print(f"  Generation Trend: {forecast['generation_trend_monthly']:.0f}/month")
    print(f"  Total Forecasted Cost: ${forecast['total_forecasted_cost']:.2f}")
    
    print(f"\n📅 Monthly Forecasts:")
    print(f"{'Month':<8} | {'Cost':<8} | {'Generations':<12} | {'Confidence'}")
    print("-" * 50)
    
    for forecast_month in forecast['monthly_forecasts']:
        print(f"Month {forecast_month['month']:<2} | ${forecast_month['forecasted_cost']:<7.2f} | "
              f"{forecast_month['forecasted_generations']:<12.0f} | {forecast_month['confidence']*100:.0f}%")
    
    print(f"\n💡 Forecast Recommendations:")
    for rec in forecast['recommendations']:
        print(f"  • {rec}")


async def demo_billing_reports():
    """Demonstrate billing report generation."""
    print("\n\n📊 BILLING REPORTS DEMO")
    print("=" * 50)
    
    pricing_engine = PricingEngine()
    
    # Generate user report
    print("\n👤 User-Specific Report:")
    user_report = await pricing_engine.generate_billing_report(
        period_start=datetime.now() - timedelta(days=30),
        period_end=datetime.now(),
        user_id="report_demo_user"
    )
    
    print(f"  User ID: {user_report.user_id}")
    print(f"  Total Revenue: ${user_report.total_revenue:.2f}")
    print(f"  Subscription Revenue: ${user_report.subscription_revenue:.2f}")
    print(f"  Usage Revenue: ${user_report.usage_revenue:.2f}")
    print(f"  Total Generations: {user_report.total_generations}")
    print(f"  GPU Hours Used: {user_report.total_gpu_seconds / 3600:.1f}")
    print(f"  Storage Used: {user_report.total_storage_gb:.1f} GB")
    
    # Generate system-wide report
    print("\n🌐 System-Wide Report:")
    system_report = await pricing_engine.generate_billing_report(
        period_start=datetime.now() - timedelta(days=30),
        period_end=datetime.now()
    )
    
    print(f"  Total Revenue: ${system_report.total_revenue:,.2f}")
    print(f"  Active Subscriptions: {system_report.active_subscriptions:,}")
    print(f"  New Subscriptions: {system_report.new_subscriptions}")
    print(f"  Cancelled Subscriptions: {system_report.cancelled_subscriptions}")
    print(f"  Churn Rate: {system_report.churn_rate*100:.1f}%")
    print(f"  Average Revenue Per User: ${system_report.average_revenue_per_user:.2f}")
    print(f"  Cost Per Generation: ${system_report.cost_per_generation:.2f}")
    print(f"  Total Generations: {system_report.total_generations:,}")


async def main():
    """Run all pricing and billing demos."""
    print("🚀 SCROLLINTEL PRICING & BILLING ENGINE DEMO")
    print("=" * 60)
    print("Demonstrating tiered pricing models, cost optimization,")
    print("invoice generation, and payment processing workflows.")
    print("=" * 60)
    
    try:
        await demo_pricing_calculations()
        await demo_subscription_plans()
        await demo_quota_management()
        await demo_cost_optimization()
        await demo_billing_workflow()
        await demo_usage_forecasting()
        await demo_billing_reports()
        
        print("\n\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The pricing and billing engine demonstrates:")
        print("• Tiered pricing with volume discounts")
        print("• Peak/off-peak pricing optimization")
        print("• Subscription plan management")
        print("• Quota tracking and overage calculations")
        print("• Cost optimization alerts and recommendations")
        print("• Complete billing workflows with payment processing")
        print("• Usage forecasting and budget planning")
        print("• Comprehensive billing reports and analytics")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())