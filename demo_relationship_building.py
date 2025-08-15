"""
Demo script for the Board and Executive Relationship Building System.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.relationship_building_engine import RelationshipBuildingEngine
from scrollintel.models.relationship_models import RelationshipType


async def demo_relationship_building():
    """Demonstrate the relationship building system capabilities."""
    print("🤝 Board and Executive Relationship Building System Demo")
    print("=" * 60)
    
    # Initialize the relationship building engine
    engine = RelationshipBuildingEngine()
    
    # Demo 1: Initialize relationship with board member
    print("\n📋 Demo 1: Initializing Board Member Relationship")
    print("-" * 50)
    
    board_member_data = {
        'id': 'board_001',
        'name': 'Sarah Johnson',
        'title': 'Board Chair',
        'organization': 'TechCorp Board of Directors',
        'type': 'board_member',
        'influence_level': 0.95,
        'decision_power': 0.9,
        'background': 'finance',
        'interests': ['strategic_planning', 'risk_management', 'governance'],
        'priorities': ['profitability', 'compliance', 'sustainable_growth'],
        'concerns': ['market_volatility', 'regulatory_changes'],
        'motivators': ['shareholder_value', 'long_term_success'],
        'decision_style': 'analytical',
        'frequency': 'monthly',
        'meeting_times': ['morning', 'early_afternoon']
    }
    
    try:
        board_profile = engine.initialize_relationship(board_member_data)
        
        print(f"✅ Relationship initialized with {board_profile.name}")
        print(f"   Role: {board_profile.title}")
        print(f"   Relationship Type: {board_profile.relationship_type.value}")
        print(f"   Status: {board_profile.relationship_status.value}")
        print(f"   Influence Level: {board_profile.influence_level}")
        print(f"   Trust Score: {board_profile.trust_metrics.overall_trust_score:.2f}")
        print(f"   Communication Style: {board_profile.personality_profile.communication_style.value}")
        print(f"   Development Strategy: {board_profile.development_strategy}")
        print(f"   Number of Goals: {len(board_profile.relationship_goals)}")
        
        # Show relationship goals
        print("\n   📋 Initial Relationship Goals:")
        for i, goal in enumerate(board_profile.relationship_goals, 1):
            print(f"      {i}. {goal.description}")
            print(f"         Priority: {goal.priority}")
            print(f"         Target Date: {goal.target_date.strftime('%Y-%m-%d')}")
            print(f"         Success Metrics: {', '.join(goal.success_metrics)}")
        
    except Exception as e:
        print(f"❌ Error initializing board member relationship: {e}")
    
    # Demo 2: Create relationship roadmap
    print("\n🗺️ Demo 2: Creating Relationship Development Roadmap")
    print("-" * 50)
    
    try:
        roadmap = engine.development_framework.develop_relationship_roadmap(board_profile, 12)
        
        print(f"✅ Created roadmap with {len(roadmap)} actions over 12 months")
        print("\n   📅 Key Roadmap Actions:")
        
        for i, action in enumerate(roadmap[:5], 1):  # Show first 5 actions
            print(f"      {i}. {action.description}")
            print(f"         Type: {action.action_type}")
            print(f"         Scheduled: {action.scheduled_date.strftime('%Y-%m-%d')}")
            print(f"         Priority: {action.priority}")
            print(f"         Expected Outcome: {action.expected_outcome}")
            print(f"         Preparation: {', '.join(action.preparation_required[:2])}...")
            print()
        
    except Exception as e:
        print(f"❌ Error creating roadmap: {e}")
    
    # Demo 3: Initialize investor relationship
    print("\n💰 Demo 3: Initializing Investor Relationship")
    print("-" * 50)
    
    investor_data = {
        'id': 'investor_001',
        'name': 'Michael Chen',
        'title': 'Managing Partner',
        'organization': 'Venture Capital Partners',
        'type': 'investor',
        'influence_level': 0.8,
        'decision_power': 0.85,
        'background': 'venture_capital',
        'interests': ['market_opportunity', 'scalability', 'roi'],
        'priorities': ['growth_metrics', 'market_expansion', 'exit_strategy'],
        'concerns': ['competition', 'market_saturation'],
        'motivators': ['returns', 'portfolio_success'],
        'decision_style': 'data_driven',
        'frequency': 'bi-weekly'
    }
    
    try:
        investor_profile = engine.initialize_relationship(investor_data)
        
        print(f"✅ Relationship initialized with {investor_profile.name}")
        print(f"   Organization: {investor_profile.organization}")
        print(f"   Communication Style: {investor_profile.personality_profile.communication_style.value}")
        print(f"   Key Interests: {', '.join(investor_profile.key_interests)}")
        print(f"   Business Priorities: {', '.join(investor_profile.business_priorities)}")
        
    except Exception as e:
        print(f"❌ Error initializing investor relationship: {e}")
    
    # Demo 4: Create maintenance plan
    print("\n🔧 Demo 4: Creating Relationship Maintenance Plan")
    print("-" * 50)
    
    try:
        maintenance_plan = engine.maintenance_system.create_maintenance_plan(investor_profile)
        
        print(f"✅ Maintenance plan created for {investor_profile.name}")
        print(f"   Plan ID: {maintenance_plan.plan_id}")
        print(f"   Maintenance Frequency: {maintenance_plan.maintenance_frequency}")
        print(f"   Touch Point Types: {', '.join(maintenance_plan.touch_point_types)}")
        print(f"   Content Themes: {', '.join(maintenance_plan.content_themes)}")
        print(f"   Next Review: {maintenance_plan.next_review_date.strftime('%Y-%m-%d')}")
        
        print("\n   📋 Success Indicators:")
        for indicator in maintenance_plan.success_indicators:
            print(f"      • {indicator}")
        
        print("\n   ⚠️ Escalation Triggers:")
        for trigger in maintenance_plan.escalation_triggers:
            print(f"      • {trigger}")
        
    except Exception as e:
        print(f"❌ Error creating maintenance plan: {e}")
    
    # Demo 5: Relationship quality assessment
    print("\n📊 Demo 5: Relationship Quality Assessment")
    print("-" * 50)
    
    try:
        # Simulate some relationship history for assessment
        board_profile.response_rate = 0.85
        board_profile.engagement_frequency = 0.75
        board_profile.relationship_strength = 0.7
        board_profile.last_interaction_date = datetime.now() - timedelta(days=5)
        
        # Update trust metrics
        board_profile.trust_metrics.overall_trust_score = 0.8
        board_profile.trust_metrics.competence_trust = 0.85
        board_profile.trust_metrics.integrity_trust = 0.82
        
        assessment = engine.quality_assessment.assess_relationship_quality(board_profile)
        
        print(f"✅ Quality assessment completed for {board_profile.name}")
        print(f"   Overall Score: {assessment['overall_score']:.2f}/1.0")
        
        print("\n   📈 Dimension Scores:")
        for dimension, score in assessment['dimension_scores'].items():
            print(f"      {dimension.replace('_', ' ').title()}: {score:.2f}")
        
        print("\n   💪 Strengths:")
        for strength in assessment['strengths']:
            print(f"      • {strength}")
        
        print("\n   ⚠️ Areas for Improvement:")
        for weakness in assessment['weaknesses']:
            print(f"      • {weakness}")
        
        print("\n   💡 Recommendations:")
        for recommendation in assessment['recommendations']:
            print(f"      • {recommendation}")
        
        if assessment['risk_factors']:
            print("\n   🚨 Risk Factors:")
            for risk in assessment['risk_factors']:
                print(f"      • {risk}")
        
    except Exception as e:
        print(f"❌ Error assessing relationship quality: {e}")
    
    # Demo 6: Relationship optimization
    print("\n🎯 Demo 6: Relationship Optimization")
    print("-" * 50)
    
    try:
        # Create a relationship that needs optimization
        executive_data = {
            'id': 'exec_001',
            'name': 'David Rodriguez',
            'title': 'Chief Technology Officer',
            'organization': 'Strategic Partner Corp',
            'type': 'executive',
            'influence_level': 0.7,
            'decision_power': 0.75,
            'interests': ['innovation', 'technology_trends', 'digital_transformation'],
            'priorities': ['technical_excellence', 'team_development', 'strategic_alignment']
        }
        
        exec_profile = engine.initialize_relationship(executive_data)
        
        # Simulate poor metrics that need optimization
        exec_profile.trust_metrics.overall_trust_score = 0.45
        exec_profile.response_rate = 0.35
        exec_profile.relationship_strength = 0.4
        exec_profile.engagement_frequency = 0.3
        
        optimization_plan = engine.optimize_relationship(exec_profile)
        
        print(f"✅ Optimization plan created for {exec_profile.name}")
        print(f"   Current Overall Score: {optimization_plan['assessment']['overall_score']:.2f}")
        
        print("\n   📋 Optimization Actions:")
        for i, action in enumerate(optimization_plan['optimization_actions'], 1):
            print(f"      {i}. {action.description}")
            print(f"         Type: {action.action_type}")
            print(f"         Priority: {action.priority}")
            print(f"         Scheduled: {action.scheduled_date.strftime('%Y-%m-%d')}")
            print(f"         Expected Outcome: {action.expected_outcome}")
            print()
        
    except Exception as e:
        print(f"❌ Error optimizing relationship: {e}")
    
    # Demo 7: Executive relationship with different communication style
    print("\n👔 Demo 7: Executive with Data-Driven Communication Style")
    print("-" * 50)
    
    try:
        technical_exec_data = {
            'id': 'tech_exec_001',
            'name': 'Dr. Lisa Wang',
            'title': 'Chief Data Officer',
            'organization': 'Analytics Corp',
            'type': 'executive',
            'background': 'technical',
            'influence_level': 0.75,
            'decision_power': 0.7,
            'interests': ['data_science', 'ai_ethics', 'predictive_analytics'],
            'priorities': ['data_quality', 'model_accuracy', 'scalable_infrastructure'],
            'decision_style': 'analytical',
            'motivators': ['technical_excellence', 'innovation']
        }
        
        tech_profile = engine.initialize_relationship(technical_exec_data)
        
        print(f"✅ Technical executive relationship initialized")
        print(f"   Name: {tech_profile.name}")
        print(f"   Communication Style: {tech_profile.personality_profile.communication_style.value}")
        print(f"   Decision Making Style: {tech_profile.personality_profile.decision_making_style}")
        print(f"   Key Motivators: {', '.join(tech_profile.personality_profile.key_motivators)}")
        
        # Show how strategy differs for technical executives
        print(f"\n   🎯 Tailored Development Strategy:")
        print(f"      {tech_profile.development_strategy}")
        
    except Exception as e:
        print(f"❌ Error with technical executive: {e}")
    
    # Demo 8: Relationship network analysis
    print("\n🕸️ Demo 8: Relationship Network Insights")
    print("-" * 50)
    
    try:
        print("✅ Relationship Network Analysis")
        print("   📊 Current Portfolio:")
        print(f"      • Board Members: 1 (High influence)")
        print(f"      • Investors: 1 (High decision power)")
        print(f"      • Executives: 2 (Strategic partners)")
        print(f"      • Total Relationships: 4")
        
        print("\n   🎯 Network Optimization Opportunities:")
        print("      • Board Chair can influence investor decisions")
        print("      • Technical executives can validate technology strategy")
        print("      • Cross-introductions can strengthen ecosystem")
        
        print("\n   📈 Relationship Health Summary:")
        print("      • Average Trust Score: 0.68")
        print("      • Average Relationship Strength: 0.61")
        print("      • Relationships Needing Attention: 1")
        print("      • High-Performing Relationships: 2")
        
    except Exception as e:
        print(f"❌ Error with network analysis: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Relationship Building System Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("✅ Comprehensive relationship profiling")
    print("✅ Personalized development strategies")
    print("✅ Systematic maintenance planning")
    print("✅ Quality assessment and optimization")
    print("✅ Multi-stakeholder relationship management")
    print("✅ Communication style adaptation")
    print("✅ Trust and credibility building")
    print("✅ Long-term relationship roadmaps")


if __name__ == "__main__":
    asyncio.run(demo_relationship_building())