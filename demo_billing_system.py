#!/usr/bin/env python3
"""
ScrollIntel Billing System Demo

This demo showcases the comprehensive billing and subscription management system
including Stripe integration, ScrollCoin wallet, usage tracking, and invoice generation.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scrollintel.engines.billing_engine import ScrollBillingEngine, SubscriptionTier
from scrollintel.models.billing_models import (
    BillingCycle, PaymentStatus, TransactionType
)


class BillingSystemDemo:
    """Demo class for the billing system."""
    
    def __init__(self):
        self.billing_engine = ScrollBillingEngine()
        self.demo_user_id = "demo-user-12345"
        self.demo_results = []
    
    async def run_demo(self):
        """Run the complete billing system demo."""
        print("🚀 ScrollIntel Billing System Demo")
        print("=" * 50)
        
        try:
            # Initialize the billing engine
            await self.demo_engine_initialization()
            
            # Demo subscription management
            await self.demo_subscription_management()
            
            # Demo ScrollCoin wallet operations
            await self.demo_scrollcoin_wallet()
            
            # Demo usage tracking and billing
            await self.demo_usage_tracking()
            
            # Demo invoice generation
            await self.demo_invoice_generation()
            
            # Demo analytics and reporting
            await self.demo_analytics_reporting()
            
            # Demo Stripe integration
            await self.demo_stripe_integration()
            
            # Show summary
            self.show_demo_summary()
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise
        finally:
            await self.billing_engine.cleanup()
    
    async def demo_engine_initialization(self):
        """Demo billing engine initialization."""
        print("\n📋 1. Billing Engine Initialization")
        print("-" * 30)
        
        await self.billing_engine.initialize()
        status = self.billing_engine.get_status()
        
        print(f"✅ Engine Status: {status['status']}")
        print(f"✅ Stripe Available: {status['stripe_available']}")
        print(f"✅ Supported Tiers: {', '.join(status['supported_tiers'])}")
        print(f"✅ Supported Actions: {len(status['supported_actions'])} actions")
        
        self.demo_results.append({
            "section": "Engine Initialization",
            "status": "success",
            "details": status
        })
    
    async def demo_subscription_management(self):
        """Demo subscription management features."""
        print("\n💳 2. Subscription Management")
        print("-" * 30)
        
        # Create free subscription
        print("Creating free subscription...")
        free_sub = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "create_subscription",
                "user_id": self.demo_user_id,
                "tier": "free",
                "billing_type": "fiat"
            }
        )
        
        print(f"✅ Free subscription created: {free_sub['subscription_id']}")
        print(f"   Tier: {free_sub['tier']}")
        print(f"   Features: {', '.join(free_sub['features'])}")
        print(f"   ScrollCoins included: {free_sub['scrollcoins_included']}")
        
        # Upgrade to starter subscription
        print("\nUpgrading to starter subscription...")
        starter_sub = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "create_subscription",
                "user_id": self.demo_user_id,
                "tier": "starter",
                "billing_type": "fiat",
                "billing_cycle": "monthly"
            }
        )
        
        print(f"✅ Starter subscription created: {starter_sub['subscription_id']}")
        print(f"   Tier: {starter_sub['tier']}")
        print(f"   ScrollCoins included: {starter_sub['scrollcoins_included']}")
        
        # Upgrade to professional subscription
        print("\nUpgrading to professional subscription...")
        pro_sub = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "create_subscription",
                "user_id": self.demo_user_id,
                "tier": "professional",
                "billing_type": "hybrid",
                "billing_cycle": "yearly"
            }
        )
        
        print(f"✅ Professional subscription created: {pro_sub['subscription_id']}")
        print(f"   Tier: {pro_sub['tier']}")
        print(f"   Billing type: {pro_sub['billing_type']}")
        print(f"   ScrollCoins included: {pro_sub['scrollcoins_included']}")
        
        self.demo_results.append({
            "section": "Subscription Management",
            "status": "success",
            "subscriptions_created": 3,
            "final_tier": pro_sub['tier']
        })
    
    async def demo_scrollcoin_wallet(self):
        """Demo ScrollCoin wallet operations."""
        print("\n💰 3. ScrollCoin Wallet Operations")
        print("-" * 30)
        
        # Get initial wallet balance
        wallet_balance = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "get_wallet_balance",
                "user_id": self.demo_user_id
            }
        )
        
        print(f"✅ Initial wallet balance: {wallet_balance['balance']:,.0f} ScrollCoins")
        print(f"   Wallet ID: {wallet_balance['wallet_id']}")
        
        # Recharge wallet
        print("\nRecharging wallet with $100...")
        recharge_result = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "recharge_wallet",
                "user_id": self.demo_user_id,
                "amount": 100.0,
                "payment_method": "stripe"
            }
        )
        
        print(f"✅ Wallet recharged successfully")
        print(f"   Amount charged: ${recharge_result['amount_charged']}")
        print(f"   ScrollCoins added: {recharge_result['scrollcoins_added']:,.0f}")
        print(f"   New balance: {recharge_result['new_balance']:,.0f} ScrollCoins")
        
        # Get updated balance
        updated_balance = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "get_wallet_balance",
                "user_id": self.demo_user_id
            }
        )
        
        print(f"✅ Updated wallet balance: {updated_balance['balance']:,.0f} ScrollCoins")
        
        self.demo_results.append({
            "section": "ScrollCoin Wallet",
            "status": "success",
            "initial_balance": wallet_balance['balance'],
            "final_balance": updated_balance['balance'],
            "recharge_amount": 100.0
        })
    
    async def demo_usage_tracking(self):
        """Demo usage tracking and ScrollCoin charging."""
        print("\n📊 4. Usage Tracking & ScrollCoin Charging")
        print("-" * 30)
        
        # Define demo actions
        demo_actions = [
            ("model_inference", 50, "Running 50 model inferences"),
            ("data_analysis", 10, "Performing 10 data analyses"),
            ("training_job", 2, "Training 2 ML models"),
            ("visualization", 20, "Creating 20 visualizations"),
            ("report_generation", 5, "Generating 5 reports"),
            ("api_call", 1000, "Making 1000 API calls")
        ]
        
        total_cost = 0
        successful_charges = 0
        
        for action, quantity, description in demo_actions:
            print(f"\n{description}...")
            
            charge_result = await self.billing_engine.process(
                input_data={},
                parameters={
                    "operation": "charge_scrollcoins",
                    "user_id": self.demo_user_id,
                    "action": action,
                    "quantity": quantity,
                    "metadata": {"demo": True, "description": description}
                }
            )
            
            if charge_result["success"]:
                print(f"✅ Charged {charge_result['cost']:.2f} ScrollCoins")
                print(f"   Remaining balance: {charge_result['remaining_balance']:,.0f}")
                total_cost += charge_result['cost']
                successful_charges += 1
            else:
                print(f"❌ Charge failed: {charge_result['error']}")
                print(f"   Required: {charge_result['required']:.2f}")
                print(f"   Available: {charge_result['available']:.2f}")
        
        print(f"\n✅ Usage tracking completed")
        print(f"   Successful charges: {successful_charges}/{len(demo_actions)}")
        print(f"   Total cost: {total_cost:.2f} ScrollCoins")
        
        self.demo_results.append({
            "section": "Usage Tracking",
            "status": "success",
            "total_actions": len(demo_actions),
            "successful_charges": successful_charges,
            "total_cost": total_cost
        })
    
    async def demo_invoice_generation(self):
        """Demo invoice generation."""
        print("\n🧾 5. Invoice Generation")
        print("-" * 30)
        
        # Generate invoice for the past month
        period_start = datetime.utcnow() - timedelta(days=30)
        period_end = datetime.utcnow()
        
        print(f"Generating invoice for period: {period_start.date()} to {period_end.date()}")
        
        invoice = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "generate_invoice",
                "user_id": self.demo_user_id,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat()
            }
        )
        
        print(f"✅ Invoice generated: {invoice['id']}")
        print(f"   Subscription cost: ${invoice['subscription_cost']:.2f}")
        print(f"   Usage cost: ${invoice['usage_cost']:.2f}")
        print(f"   Total cost: ${invoice['total_cost']:.2f}")
        print(f"   Currency: {invoice['currency']}")
        print(f"   Status: {invoice['status']}")
        print(f"   Line items: {len(invoice['line_items'])}")
        
        # Show line items
        print("\n   Line items:")
        for item in invoice['line_items'][:3]:  # Show first 3 items
            print(f"   - {item['description']}: ${item['amount']:.2f}")
        
        if len(invoice['line_items']) > 3:
            print(f"   ... and {len(invoice['line_items']) - 3} more items")
        
        self.demo_results.append({
            "section": "Invoice Generation",
            "status": "success",
            "invoice_id": invoice['id'],
            "total_cost": invoice['total_cost'],
            "line_items": len(invoice['line_items'])
        })
    
    async def demo_analytics_reporting(self):
        """Demo analytics and reporting."""
        print("\n📈 6. Analytics & Reporting")
        print("-" * 30)
        
        # Get usage analytics
        analytics = await self.billing_engine.process(
            input_data={},
            parameters={
                "operation": "usage_analytics",
                "user_id": self.demo_user_id,
                "period_days": 30
            }
        )
        
        print(f"✅ Usage analytics for last {analytics['period_days']} days:")
        print(f"   Total actions: {analytics['total_actions']:,}")
        print(f"   Total cost: {analytics['total_cost']:.2f} ScrollCoins")
        
        print("\n   Top actions:")
        for action, count in analytics['top_actions'][:5]:
            cost = analytics['cost_breakdown'].get(action, 0)
            print(f"   - {action.replace('_', ' ').title()}: {count:,} actions (${cost:.2f})")
        
        print("\n   Action breakdown:")
        for action, count in analytics['action_breakdown'].items():
            if count > 0:
                cost = analytics['cost_breakdown'].get(action, 0)
                print(f"   - {action}: {count} ({cost:.2f} ScrollCoins)")
        
        self.demo_results.append({
            "section": "Analytics & Reporting",
            "status": "success",
            "total_actions": analytics['total_actions'],
            "total_cost": analytics['total_cost'],
            "unique_actions": len([a for a, c in analytics['action_breakdown'].items() if c > 0])
        })
    
    async def demo_stripe_integration(self):
        """Demo Stripe integration features."""
        print("\n💳 7. Stripe Integration")
        print("-" * 30)
        
        # Mock Stripe webhook event
        webhook_event = {
            "type": "invoice.payment_succeeded",
            "data": {
                "object": {
                    "id": "in_demo123",
                    "customer": "cus_demo123",
                    "amount_paid": 9900,  # $99.00 in cents
                    "currency": "usd",
                    "subscription": "sub_demo123"
                }
            }
        }
        
        print("Processing Stripe webhook: invoice.payment_succeeded")
        
        webhook_result = await self.billing_engine.process(
            input_data=webhook_event,
            parameters={
                "operation": "process_stripe_webhook"
            }
        )
        
        print("✅ Stripe webhook processed successfully")
        print(f"   Event type: {webhook_event['type']}")
        print(f"   Amount: ${webhook_event['data']['object']['amount_paid'] / 100:.2f}")
        print(f"   Currency: {webhook_event['data']['object']['currency'].upper()}")
        
        # Mock other webhook events
        webhook_events = [
            "customer.subscription.updated",
            "invoice.payment_failed",
            "customer.subscription.deleted"
        ]
        
        for event_type in webhook_events:
            mock_event = {
                "type": event_type,
                "data": {"object": {"id": f"demo_{event_type}", "customer": "cus_demo123"}}
            }
            
            result = await self.billing_engine.process(
                input_data=mock_event,
                parameters={"operation": "process_stripe_webhook"}
            )
            
            print(f"✅ Processed webhook: {event_type}")
        
        self.demo_results.append({
            "section": "Stripe Integration",
            "status": "success",
            "webhooks_processed": len(webhook_events) + 1,
            "payment_amount": 99.00
        })
    
    def show_demo_summary(self):
        """Show demo summary and results."""
        print("\n🎉 Demo Summary")
        print("=" * 50)
        
        total_sections = len(self.demo_results)
        successful_sections = len([r for r in self.demo_results if r["status"] == "success"])
        
        print(f"✅ Demo completed successfully!")
        print(f"   Sections completed: {successful_sections}/{total_sections}")
        
        print("\n📊 Key Metrics:")
        for result in self.demo_results:
            section = result["section"]
            print(f"\n   {section}:")
            
            if section == "Engine Initialization":
                print(f"   - Engine status: {result['details']['status']}")
                print(f"   - Stripe available: {result['details']['stripe_available']}")
            
            elif section == "Subscription Management":
                print(f"   - Subscriptions created: {result['subscriptions_created']}")
                print(f"   - Final tier: {result['final_tier']}")
            
            elif section == "ScrollCoin Wallet":
                print(f"   - Initial balance: {result['initial_balance']:,.0f} ScrollCoins")
                print(f"   - Final balance: {result['final_balance']:,.0f} ScrollCoins")
                print(f"   - Recharge amount: ${result['recharge_amount']}")
            
            elif section == "Usage Tracking":
                print(f"   - Total actions: {result['total_actions']}")
                print(f"   - Successful charges: {result['successful_charges']}")
                print(f"   - Total cost: {result['total_cost']:.2f} ScrollCoins")
            
            elif section == "Invoice Generation":
                print(f"   - Invoice ID: {result['invoice_id']}")
                print(f"   - Total cost: ${result['total_cost']:.2f}")
                print(f"   - Line items: {result['line_items']}")
            
            elif section == "Analytics & Reporting":
                print(f"   - Total actions: {result['total_actions']:,}")
                print(f"   - Total cost: {result['total_cost']:.2f} ScrollCoins")
                print(f"   - Unique actions: {result['unique_actions']}")
            
            elif section == "Stripe Integration":
                print(f"   - Webhooks processed: {result['webhooks_processed']}")
                print(f"   - Payment amount: ${result['payment_amount']}")
        
        print("\n🚀 ScrollIntel Billing System Features Demonstrated:")
        print("   ✅ Subscription tier management (Free → Starter → Professional)")
        print("   ✅ ScrollCoin wallet operations (balance, recharge, transactions)")
        print("   ✅ Usage tracking and billing (6 different action types)")
        print("   ✅ Invoice generation with line items")
        print("   ✅ Analytics and reporting (usage patterns, costs)")
        print("   ✅ Stripe webhook processing (payment events)")
        print("   ✅ Multi-currency support (USD, ScrollCoins)")
        print("   ✅ Comprehensive error handling")
        
        print("\n💡 Next Steps:")
        print("   1. Set up Stripe account and configure API keys")
        print("   2. Run database migrations for billing tables")
        print("   3. Configure webhook endpoints for Stripe events")
        print("   4. Test with real payment methods in Stripe test mode")
        print("   5. Implement frontend billing dashboard")
        print("   6. Set up monitoring and alerting for billing events")


async def main():
    """Main function to run the billing system demo."""
    demo = BillingSystemDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("🎯 Starting ScrollIntel Billing System Demo...")
    asyncio.run(main())