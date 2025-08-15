"""
Demo script for API Key Management system.
Demonstrates all features of the API key management and usage tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import requests
from sqlalchemy.orm import Session

from scrollintel.models.database import get_db, User
from scrollintel.core.api_key_manager import APIKeyManager
from scrollintel.core.api_billing_service import APIBillingService
from scrollintel.engines.usage_analytics_engine import UsageAnalyticsEngine


class APIKeyManagementDemo:
    """Demo class for API Key Management system."""
    
    def __init__(self):
        self.db = next(get_db())
        self.api_key_manager = APIKeyManager(self.db)
        self.billing_service = APIBillingService(self.db)
        self.analytics_engine = UsageAnalyticsEngine(self.db)
        self.demo_user_id = None
        self.demo_api_keys = []
    
    def run_demo(self):
        """Run the complete API key management demo."""
        print("🚀 ScrollIntel API Key Management Demo")
        print("=" * 50)
        
        try:
            # Setup demo user
            self.setup_demo_user()
            
            # Demo 1: API Key Creation
            self.demo_api_key_creation()
            
            # Demo 2: API Key Validation and Rate Limiting
            self.demo_api_key_validation()
            
            # Demo 3: Usage Tracking
            self.demo_usage_tracking()
            
            # Demo 4: Analytics and Reporting
            self.demo_analytics_reporting()
            
            # Demo 5: Billing and Quotas
            self.demo_billing_quotas()
            
            # Demo 6: API Key Management Operations
            self.demo_key_management()
            
            print("\n🎉 Demo completed successfully!")
            print("\nKey Features Demonstrated:")
            print("✅ API Key Creation and Management")
            print("✅ Rate Limiting and Throttling")
            print("✅ Usage Tracking and Analytics")
            print("✅ Billing and Quota Management")
            print("✅ Security and Validation")
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise
        finally:
            self.cleanup_demo_data()
    
    def setup_demo_user(self):
        """Setup a demo user for testing."""
        print("\n1. Setting up demo user...")
        
        # Find or create demo user
        demo_user = self.db.query(User).filter(
            User.email == "demo@scrollintel.com"
        ).first()
        
        if not demo_user:
            from scrollintel.models.database import UserRole
            demo_user = User(
                email="demo@scrollintel.com",
                hashed_password="demo_password_hash",
                full_name="Demo User",
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True
            )
            self.db.add(demo_user)
            self.db.commit()
            self.db.refresh(demo_user)
        
        self.demo_user_id = str(demo_user.id)
        print(f"✅ Demo user created: {demo_user.email} (ID: {self.demo_user_id})")
    
    def demo_api_key_creation(self):
        """Demonstrate API key creation with different configurations."""
        print("\n2. API Key Creation Demo")
        print("-" * 30)
        
        # Create development API key
        print("Creating development API key...")
        dev_key, dev_raw_key = self.api_key_manager.create_api_key(
            user_id=self.demo_user_id,
            name="Development Key",
            description="API key for development and testing",
            permissions=["agents:read", "data:read"],
            rate_limit_per_minute=10,
            rate_limit_per_hour=100,
            rate_limit_per_day=1000,
            expires_in_days=30
        )
        self.demo_api_keys.append((dev_key, dev_raw_key))
        print(f"✅ Development key created: {dev_key.get_display_key()}")
        
        # Create production API key
        print("Creating production API key...")
        prod_key, prod_raw_key = self.api_key_manager.create_api_key(
            user_id=self.demo_user_id,
            name="Production Key",
            description="API key for production applications",
            permissions=["agents:read", "agents:execute", "data:read", "data:write"],
            rate_limit_per_minute=100,
            rate_limit_per_hour=2000,
            rate_limit_per_day=20000,
            quota_requests_per_month=100000
        )
        self.demo_api_keys.append((prod_key, prod_raw_key))
        print(f"✅ Production key created: {prod_key.get_display_key()}")
        
        # Create high-volume API key
        print("Creating high-volume API key...")
        hv_key, hv_raw_key = self.api_key_manager.create_api_key(
            user_id=self.demo_user_id,
            name="High Volume Key",
            description="API key for high-traffic applications",
            permissions=["agents:read", "agents:execute", "data:read", "models:execute"],
            rate_limit_per_minute=500,
            rate_limit_per_hour=10000,
            rate_limit_per_day=100000
        )
        self.demo_api_keys.append((hv_key, hv_raw_key))
        print(f"✅ High-volume key created: {hv_key.get_display_key()}")
        
        print(f"\n📊 Created {len(self.demo_api_keys)} API keys")
    
    def demo_api_key_validation(self):
        """Demonstrate API key validation and rate limiting."""
        print("\n3. API Key Validation & Rate Limiting Demo")
        print("-" * 45)
        
        # Test valid key validation
        dev_key, dev_raw_key = self.demo_api_keys[0]
        print(f"Validating key: {dev_key.get_display_key()}")
        
        validated_key = self.api_key_manager.validate_api_key(dev_raw_key)
        if validated_key:
            print("✅ Key validation successful")
        else:
            print("❌ Key validation failed")
        
        # Test invalid key validation
        print("Testing invalid key validation...")
        invalid_result = self.api_key_manager.validate_api_key("sk-invalid-key")
        if invalid_result is None:
            print("✅ Invalid key correctly rejected")
        
        # Test rate limiting
        print("Testing rate limiting...")
        rate_limit_status = self.api_key_manager.check_rate_limit(dev_key)
        print(f"Rate limit status: {json.dumps(rate_limit_status, indent=2, default=str)}")
        
        # Simulate multiple requests to test rate limiting
        print("Simulating multiple requests...")
        for i in range(5):
            self.api_key_manager.record_api_usage(
                api_key=dev_key,
                endpoint=f"/api/v1/test/{i}",
                method="GET",
                status_code=200,
                response_time_ms=100 + i * 10
            )
        
        # Check updated rate limit status
        updated_status = self.api_key_manager.check_rate_limit(dev_key)
        print(f"Updated rate limit (after 5 requests):")
        print(f"  Minute: {updated_status['minute']['used']}/{updated_status['minute']['limit']}")
        print(f"  Hour: {updated_status['hour']['used']}/{updated_status['hour']['limit']}")
        print(f"  Day: {updated_status['day']['used']}/{updated_status['day']['limit']}")
    
    def demo_usage_tracking(self):
        """Demonstrate usage tracking functionality."""
        print("\n4. Usage Tracking Demo")
        print("-" * 25)
        
        # Simulate various API calls
        endpoints = [
            ("/api/v1/agents/cto", "POST", 200, 150.5),
            ("/api/v1/agents/data-scientist", "POST", 200, 220.3),
            ("/api/v1/data/upload", "POST", 201, 500.0),
            ("/api/v1/models/predict", "POST", 200, 300.2),
            ("/api/v1/dashboards", "GET", 200, 80.1),
            ("/api/v1/agents/cto", "POST", 400, 50.0),  # Error case
            ("/api/v1/data/analyze", "POST", 200, 1200.5),
        ]
        
        print("Simulating API usage across different keys...")
        
        for i, (api_key, raw_key) in enumerate(self.demo_api_keys):
            print(f"Recording usage for {api_key.name}...")
            
            # Record different usage patterns for each key
            for j, (endpoint, method, status, response_time) in enumerate(endpoints):
                if j % (i + 1) == 0:  # Different patterns for different keys
                    usage = self.api_key_manager.record_api_usage(
                        api_key=api_key,
                        endpoint=endpoint,
                        method=method,
                        status_code=status,
                        response_time_ms=response_time,
                        request_size_bytes=1024 * (j + 1),
                        response_size_bytes=2048 * (j + 1),
                        ip_address=f"192.168.1.{10 + i}",
                        user_agent=f"DemoClient/{i + 1}.0",
                        request_metadata={"demo": True, "key_index": i}
                    )
                    print(f"  ✅ Recorded: {method} {endpoint} -> {status} ({response_time}ms)")
        
        print(f"📊 Usage tracking completed for {len(self.demo_api_keys)} keys")
    
    def demo_analytics_reporting(self):
        """Demonstrate analytics and reporting features."""
        print("\n5. Analytics & Reporting Demo")
        print("-" * 35)
        
        # Get user overview
        print("Generating user overview analytics...")
        overview = self.analytics_engine.get_user_overview(self.demo_user_id, days=30)
        
        print(f"📊 User Overview:")
        print(f"  Total API Keys: {overview['total_api_keys']}")
        print(f"  Active API Keys: {overview['active_api_keys']}")
        print(f"  Total Requests: {overview['total_requests']}")
        print(f"  Success Rate: {100 - overview['error_rate']:.1f}%")
        print(f"  Avg Response Time: {overview['average_response_time']:.1f}ms")
        print(f"  Data Transfer: {overview['data_transfer_gb']:.4f} GB")
        
        print(f"\n🔝 Top Endpoints:")
        for endpoint in overview['top_endpoints'][:3]:
            print(f"  {endpoint['endpoint']}: {endpoint['count']} requests")
        
        if overview['error_breakdown']:
            print(f"\n❌ Error Breakdown:")
            for error in overview['error_breakdown']:
                print(f"  HTTP {error['status_code']}: {error['count']} ({error['percentage']:.1f}%)")
        
        # Get detailed analytics for first API key
        if self.demo_api_keys:
            first_key, _ = self.demo_api_keys[0]
            print(f"\n📈 Detailed Analytics for '{first_key.name}':")
            
            analytics = self.analytics_engine.get_api_key_analytics(
                str(first_key.id),
                self.demo_user_id,
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow()
            )
            
            if analytics:
                summary = analytics['summary']
                print(f"  Total Requests: {summary['total_requests']}")
                print(f"  Error Rate: {summary['error_rate']:.1f}%")
                print(f"  Avg Response Time: {summary['avg_response_time']:.1f}ms")
                print(f"  P95 Response Time: {summary['p95_response_time']:.1f}ms")
        
        # Get quota status
        print("\n💰 Quota Status:")
        quota_status = self.analytics_engine.get_quota_status(self.demo_user_id)
        
        for key_quota in quota_status['api_keys']:
            print(f"  {key_quota['api_key_name']}:")
            print(f"    Requests: {key_quota['requests']['used']}/{key_quota['requests']['limit'] or 'unlimited'}")
            if key_quota['is_exceeded']:
                print(f"    ⚠️  QUOTA EXCEEDED at {key_quota['exceeded_at']}")
    
    def demo_billing_quotas(self):
        """Demonstrate billing and quota management."""
        print("\n6. Billing & Quotas Demo")
        print("-" * 28)
        
        # Show pricing tiers
        print("📋 Available Pricing Tiers:")
        pricing_info = self.billing_service.get_all_pricing_tiers()
        
        for tier_name, tier_config in pricing_info['pricing_tiers'].items():
            print(f"  {tier_name.title()}:")
            print(f"    Requests/month: {tier_config['requests_per_month'] or 'Unlimited'}")
            print(f"    Cost per request: ${tier_config['cost_per_request']}")
            print(f"    Data transfer included: {tier_config['data_transfer_gb_included']} GB")
        
        # Calculate usage costs for demo keys
        print("\n💵 Usage Cost Calculation:")
        
        for api_key, _ in self.demo_api_keys[:2]:  # Calculate for first 2 keys
            cost_data = self.billing_service.calculate_usage_cost(
                api_key_id=str(api_key.id),
                period_start=datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                period_end=datetime.utcnow(),
                pricing_tier="starter"
            )
            
            print(f"  {api_key.name}:")
            print(f"    Requests: {cost_data['usage']['requests']}")
            print(f"    Data Transfer: {cost_data['usage']['data_transfer_gb']:.4f} GB")
            print(f"    Compute Time: {cost_data['usage']['compute_seconds']:.2f}s")
            print(f"    Total Cost: ${cost_data['total_cost_usd']:.4f}")
        
        # Generate monthly bill
        print("\n📄 Monthly Bill Generation:")
        now = datetime.utcnow()
        monthly_bill = self.billing_service.calculate_monthly_bill(
            user_id=self.demo_user_id,
            year=now.year,
            month=now.month,
            pricing_tier="starter"
        )
        
        print(f"  Billing Period: {monthly_bill['bill_metadata']['billing_period']}")
        print(f"  Total API Keys: {monthly_bill['summary']['total_api_keys']}")
        print(f"  Total Requests: {monthly_bill['summary']['total_requests']}")
        print(f"  Total Cost: ${monthly_bill['summary']['total_cost_usd']:.4f}")
        
        # Pricing estimation
        print("\n🔮 Pricing Estimation:")
        estimate = self.billing_service.get_pricing_estimate(
            requests_per_month=50000,
            data_transfer_gb_per_month=10.0,
            compute_seconds_per_month=1800.0,
            pricing_tier="professional"
        )
        
        print(f"  Projected Usage: 50K requests, 10GB transfer, 30min compute")
        print(f"  Estimated Monthly Cost: ${estimate['monthly_estimate_usd']:.2f}")
    
    def demo_key_management(self):
        """Demonstrate API key management operations."""
        print("\n7. API Key Management Operations")
        print("-" * 40)
        
        # List all keys
        print("📋 Listing all API keys:")
        user_keys = self.api_key_manager.get_user_api_keys(self.demo_user_id)
        
        for key in user_keys:
            print(f"  {key.name}: {key.get_display_key()} (Active: {key.is_active})")
        
        # Update a key
        if self.demo_api_keys:
            first_key, _ = self.demo_api_keys[0]
            print(f"\n✏️  Updating key: {first_key.name}")
            
            updated_key = self.api_key_manager.update_api_key(
                api_key_id=str(first_key.id),
                user_id=self.demo_user_id,
                description="Updated description for demo",
                rate_limit_per_minute=20,  # Increased limit
                permissions=["agents:read", "data:read", "models:read"]  # Added permission
            )
            
            if updated_key:
                print("✅ Key updated successfully")
                print(f"  New rate limit: {updated_key.rate_limit_per_minute}/min")
                print(f"  New permissions: {updated_key.permissions}")
        
        # Get usage analytics for a specific key
        if self.demo_api_keys:
            key_for_analytics, _ = self.demo_api_keys[0]
            print(f"\n📊 Usage analytics for: {key_for_analytics.name}")
            
            analytics = self.api_key_manager.get_usage_analytics(
                api_key_id=str(key_for_analytics.id),
                user_id=self.demo_user_id,
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow()
            )
            
            print(f"  Total requests: {analytics['total_requests']}")
            print(f"  Success rate: {100 - analytics['error_rate']:.1f}%")
            print(f"  Avg response time: {analytics['average_response_time']:.1f}ms")
            
            if analytics['top_endpoints']:
                print("  Top endpoints:")
                for endpoint in analytics['top_endpoints'][:3]:
                    print(f"    {endpoint['endpoint']}: {endpoint['count']} requests")
    
    def cleanup_demo_data(self):
        """Clean up demo data."""
        print("\n🧹 Cleaning up demo data...")
        
        try:
            # Delete demo API keys (this will cascade to usage records)
            for api_key, _ in self.demo_api_keys:
                success = self.api_key_manager.delete_api_key(
                    str(api_key.id),
                    self.demo_user_id
                )
                if success:
                    print(f"✅ Deleted key: {api_key.name}")
            
            # Note: In a real scenario, you might want to keep the demo user
            # For this demo, we'll leave it for future runs
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")
        finally:
            self.db.close()


def main():
    """Run the API key management demo."""
    demo = APIKeyManagementDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()