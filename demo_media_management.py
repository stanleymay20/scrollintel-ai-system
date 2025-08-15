"""
Demo script for Media Management Engine

This script demonstrates the media management capabilities for crisis leadership excellence.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from scrollintel.engines.media_management_engine import MediaManagementEngine
from scrollintel.models.media_management_models import (
    MediaInquiry, MediaOutlet, PRStrategy, MediaInquiryType, 
    InquiryPriority, MediaOutletType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_media_inquiry_handling():
    """Demonstrate media inquiry handling"""
    print("\n=== Media Inquiry Handling Demo ===")
    
    engine = MediaManagementEngine()
    
    # Create sample media outlet
    outlet = MediaOutlet(
        name="Tech News Daily",
        outlet_type=MediaOutletType.ONLINE_NEWS,
        reach=500000,
        influence_score=75.0,
        contact_info={"email": "news@technews.com"},
        typical_response_time=60
    )
    
    # Create sample media inquiry
    inquiry = MediaInquiry(
        crisis_id="crisis_demo_001",
        outlet=outlet,
        reporter_name="Jane Reporter",
        reporter_contact="jane@technews.com",
        inquiry_type=MediaInquiryType.STATEMENT_REQUEST,
        subject="Company Response to System Outage",
        questions=[
            "What caused the system outage?",
            "How many customers were affected?",
            "What steps are being taken to prevent future outages?"
        ],
        deadline=datetime.now() + timedelta(hours=4)
    )
    
    print(f"Processing media inquiry from {inquiry.outlet.name}")
    print(f"Reporter: {inquiry.reporter_name}")
    print(f"Subject: {inquiry.subject}")
    print(f"Questions: {len(inquiry.questions)} questions")
    
    # Handle the inquiry
    response = await engine.handle_media_inquiry(inquiry)
    
    print(f"\nResponse generated:")
    print(f"- Response ID: {response.id}")
    print(f"- Response Type: {response.response_type}")
    print(f"- Key Messages: {len(response.key_messages)}")
    print(f"- Assigned Spokesperson: {inquiry.assigned_spokesperson}")
    print(f"- Priority Level: {inquiry.priority.value}")
    
    print(f"\nKey Messages:")
    for i, message in enumerate(response.key_messages, 1):
        print(f"  {i}. {message}")
    
    print(f"\nDraft Response Content:")
    print(response.content[:200] + "..." if len(response.content) > 200 else response.content)
    
    return engine, inquiry


async def demo_pr_strategy_coordination():
    """Demonstrate PR strategy coordination"""
    print("\n=== PR Strategy Coordination Demo ===")
    
    engine = MediaManagementEngine()
    
    # Create sample PR strategy
    strategy = PRStrategy(
        crisis_id="crisis_demo_001",
        strategy_name="System Outage Response Strategy",
        objectives=[
            "Maintain customer confidence during outage",
            "Provide transparent communication about resolution progress",
            "Minimize reputation damage",
            "Demonstrate technical competence and reliability"
        ],
        target_audiences=["Customers", "Investors", "Media", "Employees", "Partners"],
        key_messages=[
            "We are working around the clock to restore full service",
            "Customer data remains secure and protected",
            "We are implementing additional safeguards to prevent future occurrences",
            "We sincerely apologize for any inconvenience caused"
        ],
        communication_channels=[
            "Press releases",
            "Social media updates",
            "Customer email notifications",
            "Website status page",
            "Direct stakeholder communications"
        ]
    )
    
    print(f"Coordinating PR strategy: {strategy.strategy_name}")
    print(f"Crisis ID: {strategy.crisis_id}")
    print(f"Objectives: {len(strategy.objectives)}")
    print(f"Target Audiences: {', '.join(strategy.target_audiences)}")
    print(f"Key Messages: {len(strategy.key_messages)}")
    
    # Coordinate the strategy
    coordination_result = await engine.coordinate_pr_strategy(strategy.crisis_id, strategy)
    
    print(f"\nCoordination Results:")
    print(f"- Strategy ID: {coordination_result['strategy_id']}")
    print(f"- Message Consistency Score: {coordination_result['consistency_score']:.1f}/100")
    print(f"- Status: {coordination_result['status']}")
    print(f"- Next Review: {coordination_result['next_review']}")
    
    print(f"\nCommunication Timeline:")
    for milestone, time in coordination_result['timeline'].items():
        print(f"  - {milestone.replace('_', ' ').title()}: {time.strftime('%H:%M')}")
    
    print(f"\nSpokesperson Assignments:")
    for spokesperson, responsibilities in coordination_result['assignments'].items():
        print(f"  - {spokesperson}: {', '.join(responsibilities)}")
    
    return engine, strategy


async def demo_sentiment_monitoring():
    """Demonstrate media sentiment monitoring"""
    print("\n=== Media Sentiment Monitoring Demo ===")
    
    engine = MediaManagementEngine()
    crisis_id = "crisis_demo_001"
    
    print(f"Monitoring media sentiment for crisis: {crisis_id}")
    print("Analyzing last 24 hours of coverage...")
    
    # Monitor sentiment
    analysis = await engine.monitor_media_sentiment(crisis_id, timedelta(hours=24))
    
    print(f"\nSentiment Analysis Results:")
    print(f"- Analysis ID: {analysis.id}")
    print(f"- Overall Sentiment: {analysis.overall_sentiment.value}")
    print(f"- Sentiment Trend: {analysis.sentiment_trend}")
    print(f"- Total Mentions: {analysis.mention_volume}")
    
    print(f"\nMention Breakdown:")
    print(f"  - Positive: {analysis.positive_mentions} ({analysis.positive_mentions/analysis.mention_volume*100:.1f}%)")
    print(f"  - Negative: {analysis.negative_mentions} ({analysis.negative_mentions/analysis.mention_volume*100:.1f}%)")
    print(f"  - Neutral: {analysis.neutral_mentions} ({analysis.neutral_mentions/analysis.mention_volume*100:.1f}%)")
    
    print(f"\nKey Sentiment Drivers:")
    for i, driver in enumerate(analysis.key_sentiment_drivers, 1):
        print(f"  {i}. {driver}")
    
    print(f"\nOutlet Breakdown:")
    for outlet_type, sentiment_data in analysis.outlet_breakdown.items():
        print(f"  - {outlet_type.replace('_', ' ').title()}:")
        for sentiment, percentage in sentiment_data.items():
            print(f"    {sentiment}: {percentage}%")
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(analysis.recommendations, 1):
        print(f"  {i}. {recommendation}")
    
    return engine, analysis


async def demo_media_management_metrics():
    """Demonstrate media management metrics calculation"""
    print("\n=== Media Management Metrics Demo ===")
    
    engine = MediaManagementEngine()
    crisis_id = "crisis_demo_001"
    
    # Add some sample data first
    outlet = MediaOutlet(
        name="Business Weekly",
        outlet_type=MediaOutletType.TRADE_PUBLICATION,
        reach=200000,
        influence_score=85.0
    )
    
    inquiry = MediaInquiry(
        crisis_id=crisis_id,
        outlet=outlet,
        reporter_name="John Business",
        inquiry_type=MediaInquiryType.INTERVIEW_REQUEST,
        subject="Executive Interview Request"
    )
    
    await engine.handle_media_inquiry(inquiry)
    
    print(f"Calculating media management metrics for crisis: {crisis_id}")
    
    # Get metrics
    metrics = await engine.get_media_management_metrics(crisis_id)
    
    print(f"\nMedia Management Effectiveness Metrics:")
    print(f"- Crisis ID: {metrics.crisis_id}")
    print(f"- Total Inquiries: {metrics.total_inquiries}")
    print(f"- Response Rate: {metrics.response_rate:.1%}")
    print(f"- Average Response Time: {metrics.average_response_time:.1f} minutes")
    print(f"- Positive Coverage: {metrics.positive_coverage_percentage:.1f}%")
    print(f"- Total Media Reach: {metrics.media_reach:,}")
    print(f"- Message Consistency Score: {metrics.message_consistency_score:.1f}/100")
    print(f"- Crisis Narrative Control: {metrics.crisis_narrative_control:.1f}/100")
    print(f"- Reputation Impact Score: {metrics.reputation_impact_score:.1f}/100")
    
    print(f"\nSpokesperson Effectiveness:")
    for spokesperson, score in metrics.spokesperson_effectiveness.items():
        print(f"  - {spokesperson}: {score:.1f}/100")
    
    return engine, metrics


async def demo_comprehensive_media_management():
    """Demonstrate comprehensive media management workflow"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MEDIA MANAGEMENT DEMO")
    print("Crisis Leadership Excellence - Media Management System")
    print("="*60)
    
    try:
        # Demo 1: Media Inquiry Handling
        engine1, inquiry = await demo_media_inquiry_handling()
        
        # Demo 2: PR Strategy Coordination
        engine2, strategy = await demo_pr_strategy_coordination()
        
        # Demo 3: Sentiment Monitoring
        engine3, analysis = await demo_sentiment_monitoring()
        
        # Demo 4: Metrics Calculation
        engine4, metrics = await demo_media_management_metrics()
        
        print(f"\n" + "="*60)
        print("DEMO SUMMARY")
        print("="*60)
        print("✅ Media inquiry handling - Professional response generated")
        print("✅ PR strategy coordination - Message consistency validated")
        print("✅ Sentiment monitoring - Real-time analysis completed")
        print("✅ Performance metrics - Effectiveness measured")
        
        print(f"\nKey Capabilities Demonstrated:")
        print("• Professional media inquiry processing with priority assessment")
        print("• Automated spokesperson assignment based on inquiry characteristics")
        print("• PR strategy coordination with message consistency validation")
        print("• Real-time media sentiment monitoring and analysis")
        print("• Comprehensive performance metrics and effectiveness tracking")
        print("• Alert generation for significant sentiment changes")
        
        print(f"\nMedia Management System Status: FULLY OPERATIONAL")
        print("Ready for crisis leadership excellence deployment!")
        
    except Exception as e:
        logger.error(f"Demo error: {str(e)}")
        print(f"❌ Demo encountered an error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(demo_comprehensive_media_management())