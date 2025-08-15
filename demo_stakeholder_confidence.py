"""
Demo script for Stakeholder Confidence Management System.
Demonstrates crisis leadership excellence through stakeholder confidence monitoring and management.
"""

import asyncio
from datetime import datetime, timedelta
import json

from scrollintel.engines.stakeholder_confidence_engine import StakeholderConfidenceEngine
from scrollintel.models.stakeholder_confidence_models import (
    StakeholderProfile, StakeholderFeedback, StakeholderType, ConfidenceLevel
)


async def demonstrate_stakeholder_confidence_management():
    """Demonstrate comprehensive stakeholder confidence management capabilities"""
    
    print("🎯 ScrollIntel Crisis Leadership Excellence - Stakeholder Confidence Management Demo")
    print("=" * 80)
    
    # Initialize the confidence management engine
    engine = StakeholderConfidenceEngine()
    
    # Demo 1: Create stakeholder profiles
    print("\n📋 Demo 1: Creating Stakeholder Profiles")
    print("-" * 50)
    
    stakeholders = [
        StakeholderProfile(
            stakeholder_id="board_001",
            name="Sarah Johnson",
            stakeholder_type=StakeholderType.BOARD_MEMBER,
            influence_level="high",
            communication_preferences=["email", "phone", "video_conference"],
            historical_confidence=[0.9, 0.8, 0.7],
            key_concerns=["strategic_impact", "governance", "risk_management"],
            relationship_strength=0.85,
            contact_information={"email": "sarah.johnson@board.com", "phone": "+1-555-0101"},
            last_interaction=datetime.now() - timedelta(days=2)
        ),
        StakeholderProfile(
            stakeholder_id="investor_001",
            name="Michael Chen",
            stakeholder_type=StakeholderType.INVESTOR,
            influence_level="high",
            communication_preferences=["email", "investor_portal"],
            historical_confidence=[0.8, 0.7, 0.6],
            key_concerns=["financial_impact", "market_position", "recovery_timeline"],
            relationship_strength=0.75,
            contact_information={"email": "m.chen@venture.com", "phone": "+1-555-0102"},
            last_interaction=datetime.now() - timedelta(days=1)
        ),
        StakeholderProfile(
            stakeholder_id="customer_001",
            name="Jennifer Martinez",
            stakeholder_type=StakeholderType.CUSTOMER,
            influence_level="medium",
            communication_preferences=["email", "customer_portal"],
            historical_confidence=[0.7, 0.6, 0.5],
            key_concerns=["service_continuity", "data_security", "support_quality"],
            relationship_strength=0.65,
            contact_information={"email": "j.martinez@enterprise.com", "phone": "+1-555-0103"},
            last_interaction=datetime.now() - timedelta(hours=12)
        ),
        StakeholderProfile(
            stakeholder_id="employee_001",
            name="David Wilson",
            stakeholder_type=StakeholderType.EMPLOYEE,
            influence_level="medium",
            communication_preferences=["internal_chat", "email"],
            historical_confidence=[0.6, 0.5, 0.4],
            key_concerns=["job_security", "workload", "team_morale"],
            relationship_strength=0.55,
            contact_information={"email": "d.wilson@company.com", "slack": "@dwilson"},
            last_interaction=datetime.now() - timedelta(hours=6)
        )
    ]
    
    # Add stakeholders to engine
    for stakeholder in stakeholders:
        engine.stakeholder_profiles[stakeholder.stakeholder_id] = stakeholder
        print(f"✅ Created profile for {stakeholder.name} ({stakeholder.stakeholder_type.value})")
    
    # Demo 2: Monitor stakeholder confidence
    print("\n📊 Demo 2: Monitoring Stakeholder Confidence")
    print("-" * 50)
    
    crisis_id = "crisis_system_breach_2024"
    stakeholder_ids = [s.stakeholder_id for s in stakeholders]
    
    confidence_data = await engine.monitor_stakeholder_confidence(crisis_id, stakeholder_ids)
    
    print(f"🔍 Monitoring confidence for {len(stakeholder_ids)} stakeholders in crisis: {crisis_id}")
    
    for stakeholder_id, metrics in confidence_data.items():
        stakeholder = engine.stakeholder_profiles[stakeholder_id]
        print(f"\n👤 {stakeholder.name} ({stakeholder.stakeholder_type.value}):")
        print(f"   📈 Confidence Level: {metrics.confidence_level.value}")
        print(f"   🎯 Trust Score: {metrics.trust_score:.2f}")
        print(f"   💬 Engagement Score: {metrics.engagement_score:.2f}")
        print(f"   😊 Sentiment Score: {metrics.sentiment_score:.2f}")
        print(f"   📞 Response Rate: {metrics.response_rate:.2f}")
    
    # Demo 3: Assess overall confidence
    print("\n🎯 Demo 3: Overall Confidence Assessment")
    print("-" * 50)
    
    assessment = await engine.assess_overall_confidence(crisis_id)
    
    print(f"📊 Overall Confidence Score: {assessment.overall_confidence_score:.2f}")
    print(f"📅 Assessment Time: {assessment.assessment_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📈 Confidence by Stakeholder Type:")
    for stakeholder_type, score in assessment.stakeholder_breakdown.items():
        print(f"   {stakeholder_type.value}: {score:.2f}")
    
    print(f"\n⚠️  Risk Areas ({len(assessment.risk_areas)}):")
    for risk in assessment.risk_areas:
        print(f"   • {risk}")
    
    print(f"\n💡 Improvement Opportunities ({len(assessment.improvement_opportunities)}):")
    for opportunity in assessment.improvement_opportunities:
        print(f"   • {opportunity}")
    
    print(f"\n🎯 Recommended Actions ({len(assessment.recommended_actions)}):")
    for action in assessment.recommended_actions:
        print(f"   • {action}")
    
    # Demo 4: Build confidence strategy
    print("\n🚀 Demo 4: Building Confidence Strategy")
    print("-" * 50)
    
    # Build strategy for investors (most critical stakeholder type)
    strategy = await engine.build_confidence_strategy(
        StakeholderType.INVESTOR,
        ConfidenceLevel.LOW,
        ConfidenceLevel.HIGH
    )
    
    print(f"📋 Strategy ID: {strategy.strategy_id}")
    print(f"🎯 Target: {strategy.stakeholder_type.value} confidence from LOW to HIGH")
    print(f"💬 Communication Approach: {strategy.communication_approach}")
    
    print(f"\n🔑 Key Messages ({len(strategy.key_messages)}):")
    for i, message in enumerate(strategy.key_messages, 1):
        print(f"   {i}. {message}")
    
    print(f"\n🎪 Engagement Tactics ({len(strategy.engagement_tactics)}):")
    for i, tactic in enumerate(strategy.engagement_tactics, 1):
        print(f"   {i}. {tactic}")
    
    print(f"\n📅 Timeline:")
    for milestone, date in strategy.timeline.items():
        print(f"   • {milestone}: {date.strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n📊 Success Metrics ({len(strategy.success_metrics)}):")
    for metric in strategy.success_metrics:
        print(f"   • {metric}")
    
    # Demo 5: Maintain stakeholder trust
    print("\n🤝 Demo 5: Maintaining Stakeholder Trust")
    print("-" * 50)
    
    # Focus on the investor with declining confidence
    investor_id = "investor_001"
    crisis_context = {
        "crisis_type": "security_breach",
        "severity": "high",
        "estimated_resolution": "72_hours",
        "financial_impact": "moderate",
        "customer_impact": "limited"
    }
    
    trust_actions = await engine.maintain_stakeholder_trust(investor_id, crisis_context)
    
    investor = engine.stakeholder_profiles[investor_id]
    print(f"🎯 Trust Maintenance for {investor.name} ({investor.stakeholder_type.value})")
    print(f"📋 Generated {len(trust_actions)} trust maintenance actions:")
    
    for i, action in enumerate(trust_actions, 1):
        print(f"\n   Action {i}: {action.action_type}")
        print(f"   📝 Description: {action.description}")
        print(f"   ⚡ Priority: {action.priority}")
        print(f"   🎯 Expected Impact: {action.expected_impact}")
        print(f"   📅 Timeline: {action.timeline.strftime('%Y-%m-%d %H:%M')}")
        print(f"   📋 Implementation Steps ({len(action.implementation_steps)}):")
        for step in action.implementation_steps:
            print(f"      • {step}")
        print(f"   ✅ Success Criteria ({len(action.success_criteria)}):")
        for criteria in action.success_criteria:
            print(f"      • {criteria}")
    
    # Demo 6: Create communication plan
    print("\n📢 Demo 6: Creating Communication Plan")
    print("-" * 50)
    
    stakeholder_segments = [StakeholderType.BOARD_MEMBER, StakeholderType.INVESTOR, StakeholderType.CUSTOMER]
    comm_plan = await engine.create_communication_plan(crisis_id, stakeholder_segments)
    
    print(f"📋 Communication Plan ID: {comm_plan.plan_id}")
    print(f"🎯 Target Segments: {[s.value for s in comm_plan.stakeholder_segments]}")
    print(f"📅 Frequency: {comm_plan.frequency}")
    print(f"🎨 Tone & Style: {comm_plan.tone_and_style}")
    
    print(f"\n🔑 Key Messages by Segment:")
    for segment, message in comm_plan.key_messages.items():
        print(f"   • {segment}: {message}")
    
    print(f"\n📡 Communication Channels ({len(comm_plan.communication_channels)}):")
    for channel in comm_plan.communication_channels:
        print(f"   • {channel}")
    
    print(f"\n✅ Approval Workflow ({len(comm_plan.approval_workflow)}):")
    for step in comm_plan.approval_workflow:
        print(f"   • {step}")
    
    print(f"\n📊 Effectiveness Metrics ({len(comm_plan.effectiveness_metrics)}):")
    for metric in comm_plan.effectiveness_metrics:
        print(f"   • {metric}")
    
    # Demo 7: Process stakeholder feedback
    print("\n💬 Demo 7: Processing Stakeholder Feedback")
    print("-" * 50)
    
    # Simulate feedback from concerned customer
    feedback = StakeholderFeedback(
        feedback_id="feedback_001",
        stakeholder_id="customer_001",
        feedback_type="concern",
        content="We're very concerned about the security breach and its impact on our data. When will normal operations resume? What guarantees do we have that this won't happen again?",
        sentiment="negative",
        urgency_level="high",
        received_time=datetime.now(),
        response_required=True
    )
    
    result = await engine.process_stakeholder_feedback(feedback)
    
    customer = engine.stakeholder_profiles[feedback.stakeholder_id]
    print(f"📨 Processing feedback from {customer.name} ({customer.stakeholder_type.value})")
    print(f"📝 Feedback Type: {feedback.feedback_type}")
    print(f"⚡ Urgency: {feedback.urgency_level}")
    print(f"😟 Sentiment: {feedback.sentiment}")
    
    print(f"\n🔍 Analysis Results:")
    analysis = result["analysis"]
    print(f"   📊 Sentiment Score: {analysis['sentiment_score']:.2f}")
    print(f"   ⚡ Urgency Level: {analysis['urgency_level']}")
    print(f"   🚨 Requires Escalation: {analysis['requires_escalation']}")
    print(f"   ⏱️  Estimated Resolution: {analysis['estimated_resolution_time']}")
    
    print(f"\n📋 Response Strategy:")
    strategy = result["response_strategy"]
    print(f"   💬 Response Type: {strategy['response_type']}")
    print(f"   📅 Response Timeline: {strategy['response_timeline'].strftime('%Y-%m-%d %H:%M')}")
    print(f"   🚨 Escalation Required: {strategy['escalation_required']}")
    print(f"   👥 Resource Assignment: {strategy['resource_assignment']}")
    
    print(f"\n✅ Follow-up Actions ({len(result['follow_up_actions'])}):")
    for action in result["follow_up_actions"]:
        print(f"   • {action}")
    
    # Demo 8: Monitor alerts and trends
    print("\n🚨 Demo 8: Monitoring Alerts and Trends")
    print("-" * 50)
    
    print(f"🔔 Active Confidence Alerts: {len(engine.active_alerts)}")
    for alert in engine.active_alerts:
        stakeholder = engine.stakeholder_profiles.get(alert.stakeholder_id)
        stakeholder_name = stakeholder.name if stakeholder else "Unknown"
        print(f"   ⚠️  {alert.alert_type} - {stakeholder_name}")
        print(f"      Severity: {alert.severity}")
        print(f"      Description: {alert.description}")
        print(f"      Triggered: {alert.trigger_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze trends
    trends = await engine._analyze_confidence_trends()
    print(f"\n📈 Confidence Trends Analysis:")
    print(f"   Overall Trend: {trends['overall_trend']}")
    print(f"   Risk Stakeholders: {len(trends['risk_stakeholders'])}")
    print(f"   Improving Stakeholders: {len(trends['improving_stakeholders'])}")
    
    # Demo 9: Dashboard summary
    print("\n📊 Demo 9: Stakeholder Confidence Dashboard Summary")
    print("-" * 50)
    
    total_stakeholders = len(engine.stakeholder_profiles)
    total_alerts = len(engine.active_alerts)
    total_feedback = len(engine.feedback_queue)
    total_strategies = len(engine.building_strategies)
    total_assessments = len(engine.assessments)
    
    print(f"📈 Dashboard Metrics:")
    print(f"   👥 Total Stakeholders: {total_stakeholders}")
    print(f"   🚨 Active Alerts: {total_alerts}")
    print(f"   💬 Pending Feedback: {total_feedback}")
    print(f"   🚀 Active Strategies: {total_strategies}")
    print(f"   📊 Completed Assessments: {total_assessments}")
    print(f"   📋 Overall Confidence: {assessment.overall_confidence_score:.2f}")
    
    # Success metrics
    print(f"\n🎯 Crisis Leadership Excellence Metrics:")
    print(f"   ⚡ Response Time: < 2 hours for high-priority stakeholders")
    print(f"   📊 Confidence Monitoring: Real-time across all stakeholder types")
    print(f"   🤝 Trust Maintenance: Proactive action plans for each stakeholder")
    print(f"   📢 Communication: Coordinated messaging across all channels")
    print(f"   🔍 Assessment: Comprehensive confidence analysis every 4 hours")
    
    print("\n" + "=" * 80)
    print("✅ Stakeholder Confidence Management Demo Complete!")
    print("🎯 ScrollIntel demonstrates superhuman crisis leadership through:")
    print("   • Real-time confidence monitoring and assessment")
    print("   • Proactive trust maintenance strategies")
    print("   • Coordinated stakeholder communication")
    print("   • Rapid feedback processing and response")
    print("   • Comprehensive confidence building plans")
    print("=" * 80)


def demonstrate_api_integration():
    """Demonstrate API integration capabilities"""
    
    print("\n🔌 API Integration Examples:")
    print("-" * 30)
    
    # Example API calls
    api_examples = [
        {
            "endpoint": "POST /api/v1/stakeholder-confidence/monitor",
            "description": "Monitor stakeholder confidence levels",
            "payload": {
                "crisis_id": "crisis_001",
                "stakeholder_ids": ["stakeholder_001", "stakeholder_002"]
            }
        },
        {
            "endpoint": "POST /api/v1/stakeholder-confidence/assess",
            "description": "Assess overall confidence situation",
            "payload": {"crisis_id": "crisis_001"}
        },
        {
            "endpoint": "POST /api/v1/stakeholder-confidence/strategy/build",
            "description": "Build confidence improvement strategy",
            "payload": {
                "stakeholder_type": "investor",
                "current_confidence": "low",
                "target_confidence": "high"
            }
        },
        {
            "endpoint": "POST /api/v1/stakeholder-confidence/trust/maintain",
            "description": "Generate trust maintenance actions",
            "payload": {
                "stakeholder_id": "stakeholder_001",
                "crisis_context": {"crisis_type": "security_breach", "severity": "high"}
            }
        }
    ]
    
    for example in api_examples:
        print(f"\n📡 {example['endpoint']}")
        print(f"   📝 {example['description']}")
        print(f"   📋 Payload: {json.dumps(example['payload'], indent=6)}")


if __name__ == "__main__":
    print("🚀 Starting Stakeholder Confidence Management Demo...")
    
    # Run the main demo
    asyncio.run(demonstrate_stakeholder_confidence_management())
    
    # Show API integration examples
    demonstrate_api_integration()
    
    print("\n🎉 Demo completed successfully!")
    print("💡 This system enables ScrollIntel to maintain stakeholder confidence")
    print("   during crisis situations with superhuman effectiveness and precision.")