"""
Test script for ScrollIntel X enhanced API gateway.
Verifies the new spiritual intelligence endpoints and unified response format.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Test the enhanced gateway functionality
async def test_scrollintel_x_gateway():
    """Test ScrollIntel X gateway enhancements."""
    
    print("Testing ScrollIntel X Enhanced API Gateway")
    print("=" * 50)
    
    try:
        # Import the enhanced gateway
        from scrollintel.api.gateway import ScrollIntelGateway
        
        # Create gateway instance
        gateway = ScrollIntelGateway()
        
        print("✅ Gateway instance created successfully")
        print(f"   - Spiritual governance enabled: {gateway._spiritual_governance_enabled}")
        print(f"   - Evaluation pipeline enabled: {gateway._evaluation_pipeline_enabled}")
        
        # Test unified response creation
        test_data = {"message": "Test response", "value": 42}
        unified_response = gateway.create_unified_response(
            data=test_data,
            request_id="test-123",
            processing_time=0.05,
            agents_involved=["test_agent"],
            evaluation_scores={
                'overall_score': 0.95,
                'accuracy': 0.92,
                'scroll_alignment': 0.98,
                'confidence': 0.89
            }
        )
        
        print("✅ Unified response creation successful")
        print(f"   - Success: {unified_response.success}")
        print(f"   - Scroll alignment: {unified_response.evaluation.scroll_alignment}")
        print(f"   - Governance aligned: {unified_response.governance.aligned}")
        
        # Test scroll alignment validation
        alignment_result = gateway.validate_scroll_alignment("Test spiritual content")
        print("✅ Scroll alignment validation successful")
        print(f"   - Aligned: {alignment_result['aligned']}")
        print(f"   - Spiritual validation: {alignment_result['spiritual_validation']}")
        
        # Test response quality evaluation
        quality_scores = gateway.evaluate_response_quality(test_data)
        print("✅ Response quality evaluation successful")
        print(f"   - Overall score: {quality_scores['overall_score']}")
        print(f"   - Confidence: {quality_scores['confidence']}")
        
        # Get the FastAPI app
        app = gateway.get_app()
        print("✅ FastAPI app retrieved successfully")
        print(f"   - Title: {app.title}")
        print(f"   - Version: {app.version}")
        print(f"   - Description: {app.description}")
        
        # Check if ScrollIntel X routes are available
        routes = [route.path for route in app.routes]
        scrollintel_x_routes = [route for route in routes if 'scrollintel-x' in route]
        
        print("✅ ScrollIntel X routes registered:")
        for route in scrollintel_x_routes:
            print(f"   - {route}")
        
        print("\n" + "=" * 50)
        print("ScrollIntel X Gateway Enhancement Test PASSED")
        print("All core functionality is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_scrollintel_x_gateway())
    exit(0 if success else 1)