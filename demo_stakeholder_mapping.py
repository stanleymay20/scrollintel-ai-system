#!/usr/bin/env python3
"""
Demo script for Stakeholder Mapping System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.engines.stakeholder_mapping_engine import StakeholderMappingEngine
from scrollintel.models.stakeholder_influence_models import *


def main():
    """Demonstrate stakeholder mapping functionality"""
    print("=== ScrollIntel Stakeholder Mapping System Demo ===\n")
    
    # Initialize the engine
    engine = StakeholderMappingEngine()
    
    # Sample organization context
    organization_context = {
        'organization_name': 'ScrollIntel Corp',
        'board_members': [
            {
                'id': 'board_001',
                'name': 'Sarah Johnson',
                'title': 'Board Chair',
                'industry_experience': ['technology', 'finance'],
                'expertise': ['strategy', 'governance', 'risk_management'],
                'education': ['MBA Harvard', 'BS Computer Science'],
                'previous_roles': ['CEO TechCorp', 'VP Strategy FinanceInc'],
                'achievements': ['IPO Success', 'Digital Transformation Leader'],
                'contact_preferences': {'email': 'preferred', 'meeting_frequency': 'monthly'}
            },
            {
                'id': 'board_002',
                'name': 'Michael Chen',
                'title': 'Board Member',
                'industry_experience': ['technology', 'venture_capital'],
                'expertise': ['product_development', 'scaling'],
                'education': ['MS Engineering Stanford'],
                'previous_roles': ['CTO StartupCorp', 'VP Engineering BigTech'],
                'achievements': ['Product Innovation Award', 'Scaling Expert'],
                'contact_preferences': {'phone': 'preferred', 'meeting_frequency': 'quarterly'}
            }
        ],
        'executives': [
            {
                'id': 'exec_001',
                'name': 'Jennifer Martinez',
                'title': 'CEO',
                'industry_experience': ['technology', 'consulting'],
                'expertise': ['operations', 'strategy', 'team_building'],
                'education': ['MBA Wharton', 'BS Business'],
                'previous_roles': ['VP Operations ConsultingFirm', 'Director Strategy'],
                'achievements': ['Revenue Growth 300%', 'Team Excellence Award'],
                'contact_preferences': {'phone': 'preferred', 'meeting_frequency': 'weekly'}
            },
            {
                'id': 'exec_002',
                'name': 'David Kim',
                'title': 'CTO',
                'industry_experience': ['technology', 'ai_research'],
                'expertise': ['artificial_intelligence', 'architecture', 'innovation'],
                'education': ['PhD Computer Science MIT', 'MS AI'],
                'previous_roles': ['Principal Scientist BigTech', 'Research Lead'],
                'achievements': ['AI Innovation Award', 'Patent Portfolio'],
                'contact_preferences': {'email': 'preferred', 'meeting_frequency': 'bi_weekly'}
            }
        ],
        'investors': [
            {
                'id': 'inv_001',
                'name': 'Venture Capital Partners',
                'title': 'Lead Investor',
                'organization': 'VCP Fund',
                'industry_experience': ['technology', 'venture_capital'],
                'expertise': ['venture_capital', 'scaling', 'market_analysis'],
                'education': ['MBA Stanford'],
                'previous_roles': ['Partner VCFund', 'Investment Director'],
                'achievements': ['Successful Exits Portfolio', 'Top Tier Returns'],
                'contact_preferences': {'email': 'preferred', 'meeting_frequency': 'monthly'}
            }
        ],
        'advisors': [
            {
                'id': 'adv_001',
                'name': 'Robert Wilson',
                'title': 'Strategic Advisor',
                'organization': 'Independent',
                'industry_experience': ['technology', 'enterprise_sales'],
                'expertise': ['go_to_market', 'enterprise_sales', 'partnerships'],
                'education': ['MBA Kellogg'],
                'previous_roles': ['VP Sales Enterprise Corp', 'Chief Revenue Officer'],
                'achievements': ['Sales Excellence Award', 'Partnership Success'],
                'contact_preferences': {'phone': 'preferred', 'meeting_frequency': 'monthly'}
            }
        ]
    }
    
    print("1. STAKEHOLDER IDENTIFICATION")
    print("=" * 50)
    
    # Test stakeholder identification
    stakeholders = engine.identify_key_stakeholders(organization_context)
    print(f"✓ Identified {len(stakeholders)} key stakeholders:")
    
    for stakeholder in stakeholders:
        print(f"  • {stakeholder.name} ({stakeholder.title}) - {stakeholder.stakeholder_type.value}")
        print(f"    Influence Level: {stakeholder.influence_level.value}")
        print(f"    Communication Style: {stakeholder.communication_style.value}")
        print(f"    Priorities: {len(stakeholder.priorities)} identified")
        print()
    
    print("\n2. INFLUENCE ASSESSMENT")
    print("=" * 50)
    
    # Test influence assessment for each stakeholder
    for stakeholder in stakeholders[:3]:  # Test first 3 stakeholders
        context = {'max_relationships': 20}
        assessment = engine.assess_stakeholder_influence(stakeholder, context)
        
        print(f"Stakeholder: {stakeholder.name}")
        print(f"  Overall Influence Score: {assessment.overall_influence:.3f}")
        print(f"  Formal Authority: {assessment.formal_authority:.3f}")
        print(f"  Informal Influence: {assessment.informal_influence:.3f}")
        print(f"  Network Centrality: {assessment.network_centrality:.3f}")
        print(f"  Expertise Credibility: {assessment.expertise_credibility:.3f}")
        print(f"  Resource Control: {assessment.resource_control:.3f}")
        print()
    
    print("\n3. RELATIONSHIP MAPPING")
    print("=" * 50)
    
    # Add some sample relationships
    stakeholders[0].relationships = [
        Relationship(
            stakeholder_id=stakeholders[1].id,
            relationship_type="professional_collaboration",
            strength=0.8,
            history=["positive_board_interactions", "strategic_alignment"]
        ),
        Relationship(
            stakeholder_id=stakeholders[2].id,
            relationship_type="governance_oversight",
            strength=0.7,
            history=["regular_reporting", "mutual_respect"]
        )
    ]
    
    stakeholders[1].relationships = [
        Relationship(
            stakeholder_id=stakeholders[2].id,
            relationship_type="executive_collaboration",
            strength=0.9,
            history=["daily_operations", "strategic_planning"]
        )
    ]
    
    # Test relationship mapping
    stakeholder_map = engine.map_stakeholder_relationships(stakeholders)
    
    print(f"✓ Created stakeholder map:")
    print(f"  Map ID: {stakeholder_map.id}")
    print(f"  Total Stakeholders: {len(stakeholder_map.stakeholders)}")
    print(f"  Influence Networks: {len(stakeholder_map.influence_networks)}")
    print(f"  Key Relationships: {len(stakeholder_map.key_relationships)}")
    
    # Display network information
    for network in stakeholder_map.influence_networks:
        print(f"\n  Network: {network.name}")
        print(f"    Stakeholders: {len(network.stakeholders)}")
        print(f"    Power Centers: {len(network.power_centers)}")
        if network.power_centers:
            power_center_names = []
            for pc_id in network.power_centers:
                stakeholder = next((s for s in stakeholders if s.id == pc_id), None)
                if stakeholder:
                    power_center_names.append(stakeholder.name)
            print(f"    Power Center Members: {', '.join(power_center_names)}")
    
    print(f"\n  Power Dynamics:")
    for key, value in stakeholder_map.power_dynamics.items():
        print(f"    {key}: {len(value) if isinstance(value, list) else value}")
    
    print("\n4. RELATIONSHIP OPTIMIZATION")
    print("=" * 50)
    
    # Test relationship optimization
    objectives = ["ai_innovation", "market_expansion", "governance_excellence"]
    optimizations = engine.optimize_stakeholder_relationships(stakeholder_map, objectives)
    
    print(f"✓ Generated {len(optimizations)} relationship optimization strategies:")
    
    for opt in optimizations[:2]:  # Show first 2 optimizations
        stakeholder = next(s for s in stakeholders if s.id == opt.stakeholder_id)
        print(f"\n  Stakeholder: {stakeholder.name}")
        print(f"    Current Relationship Strength: {opt.current_relationship_strength:.3f}")
        print(f"    Target Relationship Strength: {opt.target_relationship_strength:.3f}")
        print(f"    Improvement Potential: {opt.target_relationship_strength - opt.current_relationship_strength:.3f}")
        
        print(f"    Optimization Strategies:")
        for strategy in opt.optimization_strategies[:3]:  # Show first 3 strategies
            print(f"      • {strategy}")
        
        print(f"    Key Action Items:")
        for action in opt.action_items[:3]:  # Show first 3 actions
            print(f"      • {action}")
    
    print("\n5. COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    # Test comprehensive analysis for one stakeholder
    test_stakeholder = stakeholders[0]  # Board Chair
    context = {
        'max_relationships': 20,
        'stakeholder_map': stakeholder_map,
        'objectives': objectives
    }
    
    analysis = engine.analyze_stakeholder_comprehensive(test_stakeholder, context)
    
    print(f"✓ Comprehensive Analysis for {test_stakeholder.name}:")
    print(f"  Overall Influence: {analysis.influence_assessment.overall_influence:.3f}")
    print(f"  Relationship Optimization Target: {analysis.relationship_optimization.target_relationship_strength:.3f}")
    print(f"  Engagement History Entries: {len(analysis.engagement_history)}")
    print(f"  Predicted Positions: {len(analysis.predicted_positions)}")
    print(f"  Engagement Recommendations: {len(analysis.engagement_recommendations)}")
    
    print(f"\n  Predicted Positions on Key Issues:")
    for issue, position in analysis.predicted_positions.items():
        print(f"    {issue}: {position}")
    
    print(f"\n  Top Engagement Recommendations:")
    for rec in analysis.engagement_recommendations[:3]:
        print(f"    • {rec}")
    
    print("\n" + "=" * 70)
    print("✅ STAKEHOLDER MAPPING SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Capabilities Demonstrated:")
    print("✓ Key board and executive stakeholder identification and analysis")
    print("✓ Stakeholder influence assessment and tracking")
    print("✓ Stakeholder relationship mapping and optimization")
    print("\nAll requirements for Task 5.1 have been successfully implemented!")


if __name__ == "__main__":
    main()