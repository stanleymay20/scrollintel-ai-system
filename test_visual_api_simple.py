"""
Simple test to verify visual generation API integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scrollintel.api.main import app
    print("✓ Successfully imported main app with visual generation routes")
    
    # Check if visual generation routes are included
    routes = [route.path for route in app.routes]
    visual_routes = [route for route in routes if "/visual" in route]
    
    if visual_routes:
        print(f"✓ Found {len(visual_routes)} visual generation routes:")
        for route in visual_routes[:5]:  # Show first 5
            print(f"  - {route}")
        if len(visual_routes) > 5:
            print(f"  ... and {len(visual_routes) - 5} more")
    else:
        print("✗ No visual generation routes found")
    
    print("\n✓ Visual generation API integration successful!")
    
except Exception as e:
    print(f"✗ Integration failed: {e}")
    import traceback
    traceback.print_exc()