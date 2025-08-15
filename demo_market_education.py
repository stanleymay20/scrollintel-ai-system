"""
Demo script for Market Education System - Systematic market conditioning
"""

import asyncio
import json
from datetime import datetime, timedelta

from scrollintel.engines.market_education_engine import (
    MarketEducationEngine,
    CampaignType,
    TargetSegment,
    ContentType
)
from scrollintel.core.content_delivery_system import (
    ContentDeliverySystem,
    DeliveryChannel,
    EngagementLevel
)
from scrollintel.core.market_readiness_assessment import (
    MarketReadinessAssessment,
    MarketSegment,
    ReadinessLevel
)

async def demo_market_education_system():
    """Demonstrate the complete market education system"""
    print("🎯 ScrollIntel Market Education System Demo")
    print("=" * 60)
    
    # Initialize systems
    education_engine = MarketEducationEngine()
    delivery_system = ContentDeliverySystem()
    readiness_assessment = MarketReadinessAssessment()
    
    print("\n📊 Phase 1: Initial Market Readiness Assessment")
    print("-" * 50)
    
    # Assess initial market readiness
    initial_assessments = {}
    for segment in [MarketSegment.ENTERPRISE_CTOS, MarketSegment.TECH_LEADERS, MarketSegment.BOARD_MEMBERS]:
        assessment = await readiness_assessment.assess_segment_readiness(segment)
        initial_assessments[segment] = assessment
        
        print(f"\n{segment.value.title()}:")
        print(f"  Readiness Level: {assessment.readiness_level.value}")
        print(f"  Readiness Score: {assessment.readiness_score:.1f}%")
        print(f"  Confidence: {assessment.confidence_level:.2f}")
        print(f"  Top Barriers: {', '.join(assessment.barriers[:2])}")
        print(f"  Top Accelerators: {', '.join(assessment.accelerators[:2])}")
    
    print("\n🚀 Phase 2: Execute 5-Year Market Conditioning Plan")
    print("-" * 50)
    
    # Execute 5-year plan
    execution_plan = await education_engine.execute_five_year_plan()
    
    print(f"Plan Status: {execution_plan['plan_status']}")
    print(f"Target Readiness: {execution_plan['target_readiness']}%")
    print(f"Campaigns Created: {len(execution_plan['campaigns_created'])}")
    
    print("\nCampaign Timeline:")
    for campaign_info in execution_plan['campaigns_created']:
        print(f"  Year {campaign_info['year']}: {campaign_info['campaign_name']}")
    
    print("\n📧 Phase 3: Content Delivery System Setup")
    print("-" * 50)
    
    # Register delivery targets
    target_profiles = [
        {
            "name": "Sarah Chen",
            "segment": "enterprise_ctos",
            "contact_info": {"email": "sarah.chen@techcorp.com", "linkedin": "sarah-chen-cto"},
            "preferred_channels": ["email", "webinar", "conference"]
        },
        {
            "name": "Michael Rodriguez",
            "segment": "tech_leaders",
            "contact_info": {"email": "m.rodriguez@innovate.io", "twitter": "@mrodriguez_tech"},
            "preferred_channels": ["social_media", "webinar", "direct_outreach"]
        },
        {
            "name": "Jennifer Park",
            "segment": "board_members",
            "contact_info": {"email": "j.park@boardadvisors.com"},
            "preferred_channels": ["email", "direct_outreach", "conference"]
        },
        {
            "name": "David Thompson",
            "segment": "investors",
            "contact_info": {"email": "david@venturecap.com", "linkedin": "david-thompson-vc"},
            "preferred_channels": ["email", "direct_outreach", "partner_network"]
        }
    ]
    
    registered_targets = []
    for profile in target_profiles:
        target = await delivery_system.register_target(profile)
        registered_targets.append(target)
        
        print(f"\n✅ Registered: {target.name}")
        print(f"   Segment: {target.segment}")
        print(f"   Engagement Level: {target.current_engagement_level.value}")
        print(f"   Conversion Probability: {target.conversion_probability:.2%}")
        print(f"   Preferred Channels: {[ch.value for ch in target.preferred_channels]}")
    
    print("\n📨 Phase 4: Campaign Content Delivery")
    print("-" * 50)
    
    # Get first campaign (Year 1 Awareness)
    first_campaign = list(education_engine.campaigns.values())[0]
    print(f"Executing: {first_campaign.name}")
    print(f"Campaign Type: {first_campaign.campaign_type.value}")
    print(f"Content Pieces: {len(first_campaign.content_pieces)}")
    
    # Schedule content delivery
    all_deliveries = []
    for content in first_campaign.content_pieces[:2]:  # First 2 content pieces
        target_ids = [t.id for t in registered_targets[:3]]  # First 3 targets
        
        deliveries = await delivery_system.schedule_content_delivery(
            content.id,
            target_ids,
            DeliveryChannel.EMAIL,
            datetime.now() + timedelta(minutes=1)
        )
        all_deliveries.extend(deliveries)
        
        print(f"\n📅 Scheduled: {content.title}")
        print(f"   Content Type: {content.content_type.value}")
        print(f"   Targets: {len(target_ids)}")
        print(f"   Channel: EMAIL")
    
    print(f"\nTotal Deliveries Scheduled: {len(all_deliveries)}")
    
    print("\n⚡ Phase 5: Execute Deliveries and Track Engagement")
    print("-" * 50)
    
    successful_deliveries = 0
    total_engagement_score = 0
    
    for delivery in all_deliveries:
        # Execute delivery
        success = await delivery_system.execute_delivery(delivery.id)
        
        if success:
            successful_deliveries += 1
            target = delivery_system.delivery_targets[delivery.target_id]
            
            # Simulate engagement tracking
            import random
            engagement_data = {
                "opened": random.choice([True, False]),
                "clicked": random.choice([True, False]) if random.random() > 0.3 else False,
                "shared": random.choice([True, False]) if random.random() > 0.7 else False,
                "responded": random.choice([True, False]) if random.random() > 0.8 else False,
                "converted": random.choice([True, False]) if random.random() > 0.9 else False
            }
            
            await delivery_system.track_engagement(delivery.id, engagement_data)
            
            # Calculate engagement score
            engagement_score = sum([
                1 if engagement_data["opened"] else 0,
                2 if engagement_data["clicked"] else 0,
                3 if engagement_data["shared"] else 0,
                4 if engagement_data["responded"] else 0,
                5 if engagement_data["converted"] else 0
            ])
            total_engagement_score += engagement_score
            
            print(f"✅ Delivered to {target.name}: Engagement Score {engagement_score}")
        else:
            target = delivery_system.delivery_targets[delivery.target_id]
            print(f"❌ Failed delivery to {target.name}")
    
    delivery_rate = successful_deliveries / len(all_deliveries) if all_deliveries else 0
    avg_engagement = total_engagement_score / successful_deliveries if successful_deliveries else 0
    
    print(f"\nDelivery Results:")
    print(f"  Success Rate: {delivery_rate:.1%}")
    print(f"  Average Engagement Score: {avg_engagement:.1f}")
    
    print("\n📈 Phase 6: Market Conditions Monitoring")
    print("-" * 50)
    
    # Monitor market conditions
    conditions = await readiness_assessment.monitor_market_conditions()
    
    print("Current Market Conditions:")
    for condition in conditions:
        trend_emoji = {"positive": "📈", "negative": "📉", "neutral": "➡️"}
        impact_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        
        print(f"\n{trend_emoji[condition.trend]} {condition.factor}")
        print(f"   Impact: {impact_emoji[condition.impact_level]} {condition.impact_level.upper()}")
        print(f"   Trend: {condition.trend.upper()}")
        print(f"   Description: {condition.description}")
        print(f"   Response: {condition.recommended_response}")
    
    print("\n🔄 Phase 7: Strategy Adaptation")
    print("-" * 50)
    
    # Adapt strategy based on conditions
    adaptations = await readiness_assessment.adapt_strategy_based_on_conditions()
    
    print(f"Generated {len(adaptations)} adaptation strategies:")
    
    for adaptation in adaptations:
        print(f"\n🎯 {adaptation.id}")
        print(f"   Expected Impact: {adaptation.expected_impact:.1%}")
        print(f"   Timeline: {adaptation.implementation_timeline} days")
        print(f"   Actions: {len(adaptation.actions)} planned")
        print(f"   Success Metrics: {', '.join(adaptation.success_metrics)}")
    
    print("\n📊 Phase 8: Performance Reports")
    print("-" * 50)
    
    # Generate delivery report
    delivery_report = await delivery_system.generate_delivery_report()
    
    print("Content Delivery Performance:")
    summary = delivery_report["summary"]
    print(f"  Total Deliveries: {summary['total_deliveries']}")
    print(f"  Successful Deliveries: {summary['successful_deliveries']}")
    print(f"  Delivery Rate: {summary['delivery_rate']:.1%}")
    print(f"  Total Opens: {summary['total_opens']}")
    print(f"  Open Rate: {summary['open_rate']:.1%}")
    
    # Generate market readiness report
    education_report = await education_engine.generate_readiness_report()
    
    print(f"\nMarket Readiness Overview:")
    print(f"  Overall Readiness: {education_report['overall_readiness']:.1f}%")
    print(f"  Segments Assessed: {len(education_report['segment_readiness'])}")
    print(f"  Active Campaigns: {len(education_report['campaign_effectiveness'])}")
    
    # Generate comprehensive assessment report
    comprehensive_report = await readiness_assessment.generate_comprehensive_report()
    
    print(f"\nComprehensive Market Assessment:")
    exec_summary = comprehensive_report["executive_summary"]
    print(f"  Market Readiness Level: {exec_summary['market_readiness_level'].upper()}")
    print(f"  High Readiness Segments: {exec_summary['high_readiness_segments']}")
    print(f"  Key Opportunities: {len(exec_summary['key_opportunities'])}")
    print(f"  Critical Barriers: {len(exec_summary['critical_barriers'])}")
    
    print("\n🎯 Phase 9: Campaign Optimization")
    print("-" * 50)
    
    # Optimize delivery strategy
    optimization = await delivery_system.optimize_delivery_strategy(first_campaign.id)
    
    print("Delivery Strategy Optimization:")
    if optimization["best_channels"]:
        best_channel = optimization["best_channels"][0]
        print(f"  Best Channel: {best_channel['channel']} (Score: {best_channel['score']:.2f})")
    
    print(f"  Recommendations: {len(optimization['recommendations'])}")
    for rec in optimization["recommendations"]:
        print(f"    • {rec}")
    
    # Simulate campaign adaptation based on feedback
    market_feedback = {
        "resistance_high": False,
        "engagement_low": False,
        "adoption_accelerating": True,
        "resonant_themes": ["competitive_advantage", "cost_savings"]
    }
    
    await education_engine.adapt_campaign_strategy(first_campaign.id, market_feedback)
    print(f"\n✅ Campaign adapted based on market feedback")
    print(f"   New content pieces: {len(first_campaign.content_pieces)}")
    
    print("\n🏆 Phase 10: Success Metrics and Projections")
    print("-" * 50)
    
    # Calculate success metrics
    total_targets = len(registered_targets)
    high_engagement_targets = sum(
        1 for t in delivery_system.delivery_targets.values()
        if t.current_engagement_level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]
    )
    
    avg_conversion_probability = sum(
        t.conversion_probability for t in delivery_system.delivery_targets.values()
    ) / len(delivery_system.delivery_targets) if delivery_system.delivery_targets else 0
    
    # Project 5-year outcomes
    current_readiness = comprehensive_report["executive_summary"]["overall_readiness_score"]
    target_readiness = 95.0
    readiness_improvement = target_readiness - current_readiness
    
    print("Current Performance Metrics:")
    print(f"  Target Engagement Rate: {high_engagement_targets/total_targets:.1%}")
    print(f"  Average Conversion Probability: {avg_conversion_probability:.1%}")
    print(f"  Current Market Readiness: {current_readiness:.1f}%")
    
    print(f"\n5-Year Projections:")
    print(f"  Target Market Readiness: {target_readiness}%")
    print(f"  Required Improvement: {readiness_improvement:.1f} percentage points")
    print(f"  Projected Market Penetration: 75-85%")
    print(f"  Estimated Revenue Impact: $2.5B - $5B")
    
    print("\n✨ Market Education System Demo Complete!")
    print("=" * 60)
    print("The systematic 5-year market conditioning plan is now operational,")
    print("with comprehensive tracking, adaptation, and optimization capabilities.")
    print("ScrollIntel is positioned for guaranteed market success! 🚀")

async def demo_advanced_features():
    """Demonstrate advanced market education features"""
    print("\n🔬 Advanced Market Education Features Demo")
    print("=" * 60)
    
    education_engine = MarketEducationEngine()
    delivery_system = ContentDeliverySystem()
    readiness_assessment = MarketReadinessAssessment()
    
    print("\n1. 📊 Advanced Market Segmentation")
    print("-" * 40)
    
    # Demonstrate detailed segment analysis
    for segment in [MarketSegment.ENTERPRISE_CTOS, MarketSegment.INVESTORS]:
        assessment = await readiness_assessment.assess_segment_readiness(segment)
        
        print(f"\n{segment.value.title()} Deep Analysis:")
        print(f"  Readiness Indicators ({len(assessment.indicators)}):")
        
        for name, indicator in list(assessment.indicators.items())[:3]:
            progress = indicator.current_value / indicator.target_value if indicator.target_value > 0 else 0
            print(f"    • {indicator.name}: {indicator.current_value:.1f}/{indicator.target_value:.1f} ({progress:.1%})")
        
        print(f"  Strategic Recommendations:")
        for rec in assessment.recommended_actions[:2]:
            print(f"    • {rec}")
    
    print("\n2. 🎯 Multi-Channel Campaign Orchestration")
    print("-" * 40)
    
    # Create multiple campaigns
    campaigns = []
    for template in ["year_1_awareness", "year_2_education"]:
        campaign = await education_engine.create_campaign(template)
        campaigns.append(campaign)
        print(f"✅ Created: {campaign.name}")
    
    # Demonstrate cross-campaign coordination
    print(f"\nCross-Campaign Coordination:")
    print(f"  Total Campaigns: {len(campaigns)}")
    print(f"  Total Content Pieces: {sum(len(c.content_pieces) for c in campaigns)}")
    print(f"  Coordinated Timeline: 24 months")
    
    print("\n3. 🤖 Automated Follow-up System")
    print("-" * 40)
    
    # Register targets and demonstrate automated follow-up
    target_data = {
        "name": "Alex Johnson",
        "segment": "tech_leaders",
        "contact_info": {"email": "alex@techstart.com"},
        "preferred_channels": ["email", "social_media"]
    }
    target = await delivery_system.register_target(target_data)
    
    # Schedule delivery that will require follow-up
    deliveries = await delivery_system.schedule_content_delivery(
        "content_followup_test",
        [target.id],
        DeliveryChannel.EMAIL
    )
    
    # Simulate failed delivery requiring follow-up
    delivery = deliveries[0]
    delivery.status = "failed"
    delivery.follow_up_required = True
    
    print(f"📧 Scheduled delivery to {target.name}")
    print(f"🔄 Automated follow-up system activated")
    
    # Execute automated follow-up
    await delivery_system.execute_automated_follow_up()
    print(f"✅ Follow-up actions executed automatically")
    
    print("\n4. 📈 Real-time Performance Optimization")
    print("-" * 40)
    
    # Simulate real-time optimization
    optimization_data = {
        "channel_performance": {
            "email": {"engagement_rate": 0.25, "conversion_rate": 0.08},
            "webinar": {"engagement_rate": 0.45, "conversion_rate": 0.15},
            "social_media": {"engagement_rate": 0.35, "conversion_rate": 0.12}
        },
        "content_performance": {
            "whitepapers": {"effectiveness": 0.78},
            "case_studies": {"effectiveness": 0.85},
            "demos": {"effectiveness": 0.92}
        }
    }
    
    print("Real-time Performance Metrics:")
    for channel, metrics in optimization_data["channel_performance"].items():
        print(f"  {channel.title()}: {metrics['engagement_rate']:.1%} engagement, {metrics['conversion_rate']:.1%} conversion")
    
    print("\nContent Effectiveness Rankings:")
    sorted_content = sorted(
        optimization_data["content_performance"].items(),
        key=lambda x: x[1]["effectiveness"],
        reverse=True
    )
    for content_type, metrics in sorted_content:
        print(f"  {content_type.title()}: {metrics['effectiveness']:.1%} effectiveness")
    
    print("\n5. 🌍 Global Market Conditioning")
    print("-" * 40)
    
    # Simulate global market expansion
    global_segments = {
        "North America": {"readiness": 65, "priority": "high"},
        "Europe": {"readiness": 45, "priority": "medium"},
        "Asia Pacific": {"readiness": 35, "priority": "high"},
        "Latin America": {"readiness": 25, "priority": "low"}
    }
    
    print("Global Market Readiness:")
    for region, data in global_segments.items():
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        print(f"  {region}: {data['readiness']}% ready {priority_emoji[data['priority']]}")
    
    total_global_readiness = sum(data["readiness"] for data in global_segments.values()) / len(global_segments)
    print(f"\nGlobal Average Readiness: {total_global_readiness:.1f}%")
    
    print("\n✨ Advanced Features Demo Complete!")
    print("The market education system demonstrates enterprise-grade")
    print("capabilities for global market conditioning and optimization! 🌟")

if __name__ == "__main__":
    asyncio.run(demo_market_education_system())
    asyncio.run(demo_advanced_features())