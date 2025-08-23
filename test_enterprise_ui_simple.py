#!/usr/bin/env python3
"""
Simple test for Enterprise UI functionality
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_enterprise_ui_components():
    """Test Enterprise UI components"""
    print("🧪 Testing Enterprise UI Components")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        
        # Test API routes import
        from scrollintel.api.routes.enterprise_ui_routes import router
        print("✅ Enterprise UI routes imported successfully")
        
        # Test models import
        from scrollintel.models.enterprise_ui_models import UserProfile, DashboardConfig
        print("✅ Enterprise UI models imported successfully")
        
        # Test component creation
        print("\n🏗️ Testing component creation...")
        
        # Create a test user profile
        from scrollintel.models.enterprise_ui_models import create_default_user_profile
        user_profile = create_default_user_profile(
            user_id="test_123",
            name="Test User",
            email="test@example.com",
            role="executive"
        )
        print(f"✅ Created user profile: {user_profile.name} ({user_profile.role})")
        
        # Create dashboard config
        from scrollintel.models.enterprise_ui_models import create_default_dashboard_config
        dashboard_config = create_default_dashboard_config(
            user_profile_id=user_profile.id,
            role="executive"
        )
        print(f"✅ Created dashboard config with {len(dashboard_config.widgets)} widgets")
        
        # Test route functions
        print("\n🔧 Testing route functions...")
        
        from scrollintel.api.routes.enterprise_ui_routes import _generate_role_metrics
        metrics = await _generate_role_metrics("executive", "24h", None)
        print(f"✅ Generated {len(metrics)} executive metrics")
        
        from scrollintel.api.routes.enterprise_ui_routes import _process_nl_query
        query_result = await _process_nl_query(
            "What is our revenue?", 
            {"role": "executive"}, 
            None, 
            None
        )
        print(f"✅ Processed NL query with {query_result['confidence']:.1%} confidence")
        
        print("\n🎉 All Enterprise UI components tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Enterprise UI components: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_enterprise_ui_components()
    
    if success:
        print("\n✅ Enterprise UI implementation is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Enterprise UI implementation has issues!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())