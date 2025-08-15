"""
Simple test for Media Management API Routes
"""

import asyncio
from datetime import datetime, timedelta

from scrollintel.engines.media_management_engine import MediaManagementEngine
from scrollintel.models.media_management_models import (
    MediaInquiry, MediaOutlet, PRStrategy, MediaInquiryType, 
    MediaOutletType
)


async def test_media_management_api():
    """Test media management API functionality"""
    print("Testing Media Management API...")
    
    # Test engine initialization
    engine = MediaManagementEngine()
    print("✅ Engine initialized successfully")
    
    # Test media inquiry handling
    outlet = MediaOutlet(
        name="Test News",
        outlet_type=MediaOutletType.ONLINE_NEWS,
        reach=100000,
        influence_score=70.0
    )
    
    inquiry = MediaInquiry(
        crisis_id="test_crisis",
        outlet=outlet,
        reporter_name="Test Reporter",
        inquiry_type=MediaInquiryType.STATEMENT_REQUEST,
        subject="Test Subject",
        questions=["Test question?"],
        deadline=datetime.now() + timedelta(hours=2)
    )
    
    response = await engine.handle_media_inquiry(inquiry)
    assert response is not None
    assert response.inquiry_id == inquiry.id
    print("✅ Media inquiry handling works")
    
    # Test PR strategy coordination
    strategy = PRStrategy(
        crisis_id="test_crisis",
        strategy_name="Test Strategy",
        objectives=["Test objective"],
        target_audiences=["Test audience"],
        key_messages=["Test message"],
        communication_channels=["Test channel"]
    )
    
    coordination_result = await engine.coordinate_pr_strategy("test_crisis", strategy)
    assert coordination_result is not None
    assert coordination_result['strategy_id'] == strategy.id
    print("✅ PR strategy coordination works")
    
    # Test sentiment monitoring
    analysis = await engine.monitor_media_sentiment("test_crisis", timedelta(hours=24))
    assert analysis is not None
    assert analysis.crisis_id == "test_crisis"
    print("✅ Sentiment monitoring works")
    
    # Test metrics calculation
    metrics = await engine.get_media_management_metrics("test_crisis")
    assert metrics is not None
    assert metrics.crisis_id == "test_crisis"
    print("✅ Metrics calculation works")
    
    print("\n🎉 All media management API tests passed!")


if __name__ == "__main__":
    asyncio.run(test_media_management_api())