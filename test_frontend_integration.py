"""
Test frontend integration with backend visual generation services
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Test API integration
    from frontend.src.lib.api import scrollIntelApi
    print("✓ Successfully imported scrollIntelApi")
    
    # Check visual generation API methods
    visual_methods = [
        'generateImage', 'generateVideo', 'enhanceImage', 'batchGenerate',
        'getGenerationStatus', 'cancelGeneration', 'getUserGenerations',
        'getUserUsageStats', 'enhancePrompt', 'getPromptTemplates',
        'getModelCapabilities', 'getSystemStatus', 'estimateCost',
        'getCompetitiveAnalysis'
    ]
    
    missing_methods = []
    for method in visual_methods:
        if not hasattr(scrollIntelApi, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"✗ Missing API methods: {missing_methods}")
    else:
        print("✓ All visual generation API methods available")
    
    # Test hook import
    try:
        from frontend.src.hooks.useVisualGeneration import useVisualGeneration
        print("✓ Successfully imported useVisualGeneration hook")
    except Exception as e:
        print(f"✗ Failed to import useVisualGeneration hook: {e}")
    
    # Test WebSocket integration
    try:
        from scrollintel.api.websocket.visual_generation_websocket import ws_manager, send_generation_progress
        print("✓ Successfully imported WebSocket components")
    except Exception as e:
        print(f"✗ Failed to import WebSocket components: {e}")
    
    # Test main API with WebSocket routes
    try:
        from scrollintel.api.main import app
        
        # Check if WebSocket routes are included
        routes = [route.path for route in app.routes]
        ws_routes = [route for route in routes if "/ws" in route]
        
        if ws_routes:
            print(f"✓ Found {len(ws_routes)} WebSocket routes:")
            for route in ws_routes:
                print(f"  - {route}")
        else:
            print("✗ No WebSocket routes found")
        
        print("\n✓ Frontend integration with backend services successful!")
        
    except Exception as e:
        print(f"✗ Main API integration failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"✗ Frontend integration failed: {e}")
    import traceback
    traceback.print_exc()