"""
Demo script for Influence Strategy Development System
"""

import asyncio
import json
from datetime import datetime, timedelta

from scrollintel.engines.influence_strategy_engine import InfluenceStrategyEngine
from scrollintel.models.influence_strategy_models import (
    InfluenceObjective, InfluenceContext, InfluenceExecution
)
from scrollintel.models.stakeholder_influence_models import Stakeholder


def create_sample_stakeholders():
    """Create sample board and executive stakeholders"""
    from scrollintel.models.stakeholder_influence_models import (
        StakeholderType, InfluenceLevel, CommunicationStyle, 
        Background, Priority, DecisionPattern
    )
    
    return [
        Stakeholder(
            id="board_chair_001",
            name="Margaret Thompson",
            title="Board Chair",
            organization="TechCorp Inc",
            stakeholder_type=StakeholderType.BOARD_MEMBER,
            background=Background(
                industry_experience=["technology", "finance", "governance"],
                functional_expertise=["strategic_planning", "risk_management", "compliance"],
                education=["MBA Harvard", "JD Stanford"],
                previous_roles=["CEO TechStart", "Board Member FinCorp"],
                achievements=["IPO Leadership", "Governance Excellence Award"]
            ),
            influence_level=InfluenceLevel.CRITICAL,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_pattern=DecisionPattern(
                decision_style="analytical_consensus",
                key_factors=["data_quality", "risk_assessment", "stakeholder_impact"],
                typical_concerns=["regulatory_compliance", "market_volatility", "succession_planning"],
                influence_tactics=["rational_persuasion", "consultation", "coalition_building"]
            ),
            priorities=[
                Priority(name="shareholder_value", description="Maximize shareholder returns", importance=0.9, category="financial"),
                Priority(name="risk_management", description="Minimize organizational risks", importance=0.85, category="governance"),
                Priority(name="governance_excellence", description="Maintain best practices", importance=0.8, category="governance")
            ],
            relationships=[],
            contact_preferences={"email": "mthompson@boardchair.com", "phone": "+1-555-0101", "preferred_time": "morning"}
        ),
        Stakeholder(
            id="ceo_001",
            name="David Chen",
            title="Chief Executive Officer",
            organization="TechCorp Inc",
            stakeholder_type=StakeholderType.EXECUTIVE,
            background=Background(
                industry_experience=["technology", "startups", "enterprise_software"],
                functional_expertise=["product_development", "strategic_planning", "team_leadership"],
                education=["MS Computer Science MIT", "MBA Wharton"],
                previous_roles=["CTO StartupCorp", "VP Engineering BigTech"],
                achievements=["Successful IPO", "Innovation Leader Award", "Team Builder Recognition"]
            ),
            influence_level=InfluenceLevel.CRITICAL,
            communication_style=CommunicationStyle.VISIONARY,
            decision_pattern=DecisionPattern(
                decision_style="collaborative_strategic",
                key_factors=["market_opportunity", "team_capability", "innovation_potential"],
                typical_concerns=["competitive_pressure", "talent_retention", "digital_transformation"],
                influence_tactics=["inspirational_appeals", "consultation", "personal_appeals"]
            ),
            priorities=[
                Priority(name="market_leadership", description="Achieve market dominance", importance=0.95, category="strategic"),
                Priority(name="innovation", description="Drive technological innovation", importance=0.9, category="product"),
                Priority(name="team_development", description="Build exceptional teams", importance=0.85, category="organizational")
            ],
            relationships=[],
            contact_preferences={"email": "dchen@company.com", "phone": "+1-555-0102", "preferred_time": "afternoon"}
        ),
        Stakeholder(
            id="cfo_001",
            name="Sarah Rodriguez",
            title="Chief Financial Officer",
            organization="TechCorp Inc",
            stakeholder_type=StakeholderType.EXECUTIVE,
            background=Background(
                industry_experience=["finance", "technology", "public_companies"],
                functional_expertise=["financial_planning", "risk_management", "investor_relations"],
                education=["CPA", "MBA Finance Northwestern"],
                previous_roles=["Finance Director TechFirm", "Senior Analyst InvestCorp"],
                achievements=["CFO of the Year", "Cost Optimization Excellence", "IPO Success"]
            ),
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_pattern=DecisionPattern(
                decision_style="data_driven_cautious",
                key_factors=["financial_impact", "risk_assessment", "roi_analysis"],
                typical_concerns=["budget_constraints", "financial_reporting", "market_conditions"],
                influence_tactics=["rational_persuasion", "legitimating_tactics", "exchange_tactics"]
            ),
            priorities=[
                Priority(name="financial_performance", description="Optimize financial results", importance=0.95, category="financial"),
                Priority(name="cost_optimization", description="Minimize operational costs", importance=0.85, category="operational"),
                Priority(name="risk_mitigation", description="Reduce financial risks", importance=0.9, category="risk")
            ],
            relationships=[],
            contact_preferences={"email": "srodriguez@company.com", "phone": "+1-555-0103", "preferred_time": "morning"}
        ),
        Stakeholder(
            id="board_member_001",
            name="Robert Kim",
            title="Independent Board Member",
            organization="TechCorp Inc",
            stakeholder_type=StakeholderType.BOARD_MEMBER,
            background=Background(
                industry_experience=["technology", "consulting", "venture_capital"],
                functional_expertise=["strategic_oversight", "governance", "risk_assessment"],
                education=["PhD Economics", "MBA Strategy"],
                previous_roles=["Partner ConsultingFirm", "Board Member Multiple Companies"],
                achievements=["Board Excellence Award", "Strategic Advisor Recognition"]
            ),
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.DETAIL_ORIENTED,
            decision_pattern=DecisionPattern(
                decision_style="independent_questioning",
                key_factors=["thorough_analysis", "long_term_impact", "stakeholder_interests"],
                typical_concerns=["management_accountability", "strategic_direction", "competitive_positioning"],
                influence_tactics=["rational_persuasion", "consultation", "legitimating_tactics"]
            ),
            priorities=[
                Priority(name="independent_oversight", description="Provide independent governance", importance=0.9, category="governance"),
                Priority(name="strategic_guidance", description="Guide strategic decisions", importance=0.85, category="strategic"),
                Priority(name="stakeholder_protection", description="Protect all stakeholders", importance=0.8, category="governance")
            ],
            relationships=[],
            contact_preferences={"email": "rkim@independent.com", "phone": "+1-555-0104", "preferred_time": "evening"}
        )
    ]


async def demo_influence_strategy_development():
    """Demonstrate influence strategy development"""
    print("🎯 Board Executive Mastery - Influence Strategy Development Demo")
    print("=" * 70)
    
    # Initialize engine
    engine = InfluenceStrategyEngine()
    stakeholders = create_sample_stakeholders()
    
    print(f"\n📊 Stakeholder Analysis:")
    print(f"Total stakeholders: {len(stakeholders)}")
    for stakeholder in stakeholders:
        print(f"  • {stakeholder.name} ({stakeholder.title})")
        print(f"    Influence Level: {stakeholder.influence_level.value}")
        print(f"    Communication Style: {stakeholder.communication_style.value}")
        print(f"    Decision Style: {stakeholder.decision_pattern.decision_style}")
    
    # Scenario 1: Build Support for Strategic Initiative
    print(f"\n🎯 Scenario 1: Building Support for AI Integration Initiative")
    print("-" * 50)
    
    strategy1 = engine.develop_influence_strategy(
        objective=InfluenceObjective.BUILD_SUPPORT,
        target_stakeholders=stakeholders,
        context=InfluenceContext.BOARD_MEETING,
        constraints={"timeline": "2_weeks", "formality": "high"}
    )
    
    print(f"Strategy ID: {strategy1.id}")
    print(f"Objective: {strategy1.objective.value}")
    print(f"Context: {strategy1.context.value}")
    print(f"Expected Effectiveness: {strategy1.expected_effectiveness:.2f}")
    print(f"Target Stakeholders: {len(strategy1.target_stakeholders)}")
    
    print(f"\n📋 Primary Tactics:")
    for i, tactic in enumerate(strategy1.primary_tactics, 1):
        print(f"  {i}. {tactic.name}")
        print(f"     Type: {tactic.influence_type.value}")
        print(f"     Effectiveness: {tactic.effectiveness_score:.2f}")
        print(f"     Description: {tactic.description}")
    
    print(f"\n📋 Secondary Tactics:")
    for i, tactic in enumerate(strategy1.secondary_tactics, 1):
        print(f"  {i}. {tactic.name}")
        print(f"     Type: {tactic.influence_type.value}")
        print(f"     Effectiveness: {tactic.effectiveness_score:.2f}")
    
    print(f"\n📅 Timeline:")
    for phase, date in strategy1.timeline.items():
        print(f"  • {phase.replace('_', ' ').title()}: {date.strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n⚠️ Risk Mitigation:")
    for risk, mitigation in strategy1.risk_mitigation.items():
        print(f"  • {risk.replace('_', ' ').title()}: {mitigation}")
    
    print(f"\n📊 Success Metrics:")
    for metric in strategy1.success_metrics:
        print(f"  • {metric.replace('_', ' ').title()}")
    
    # Scenario 2: Gain Consensus for Budget Approval
    print(f"\n🎯 Scenario 2: Gaining Consensus for Technology Budget Increase")
    print("-" * 60)
    
    # Focus on CFO and Board Chair for budget consensus
    budget_stakeholders = [stakeholders[0], stakeholders[2]]  # Board Chair and CFO
    
    strategy2 = engine.develop_influence_strategy(
        objective=InfluenceObjective.GAIN_CONSENSUS,
        target_stakeholders=budget_stakeholders,
        context=InfluenceContext.COMMITTEE_MEETING,
        constraints={"focus": "financial_justification"}
    )
    
    print(f"Strategy ID: {strategy2.id}")
    print(f"Objective: {strategy2.objective.value}")
    print(f"Context: {strategy2.context.value}")
    print(f"Expected Effectiveness: {strategy2.expected_effectiveness:.2f}")
    
    print(f"\n📋 Tailored Tactics for Budget Consensus:")
    for i, tactic in enumerate(strategy2.primary_tactics, 1):
        print(f"  {i}. {tactic.name}")
        print(f"     Focus: Financial justification and risk mitigation")
        print(f"     Key Preparation: {', '.join(tactic.required_preparation)}")
    
    # Demonstrate tactic selection optimization
    print(f"\n🔧 Tactic Selection Optimization")
    print("-" * 40)
    
    targets = engine._convert_to_influence_targets(stakeholders[:2])
    primary_tactics, secondary_tactics = engine.select_optimal_tactics(
        objective=InfluenceObjective.CHANGE_OPINION,
        targets=targets,
        context=InfluenceContext.ONE_ON_ONE,
        constraints={"relationship_preservation": True}
    )
    
    print(f"Optimized for one-on-one opinion change:")
    print(f"Primary tactics selected: {len(primary_tactics)}")
    print(f"Secondary tactics available: {len(secondary_tactics)}")
    
    for tactic in primary_tactics:
        print(f"  • {tactic.name} (Score: {tactic.effectiveness_score:.2f})")
        print(f"    Success Indicators: {', '.join(tactic.success_indicators)}")
    
    # Simulate strategy execution and measurement
    print(f"\n📈 Strategy Execution Simulation")
    print("-" * 40)
    
    # Create simulated execution
    execution = InfluenceExecution(
        id="exec_demo_001",
        strategy_id=strategy1.id,
        execution_date=datetime.now(),
        context_details={
            "meeting_type": "board_meeting",
            "duration_minutes": 90,
            "attendees": 8,
            "presentation_used": True
        },
        tactics_used=[tactic.id for tactic in strategy1.primary_tactics],
        target_responses={
            "board_chair_001": "engaged_with_questions",
            "ceo_001": "supportive_with_suggestions",
            "cfo_001": "cautious_but_interested",
            "board_member_001": "skeptical_but_listening"
        },
        immediate_outcomes=[
            "increased_engagement",
            "detailed_questions_asked",
            "follow_up_requested",
            "concerns_raised_about_timeline"
        ],
        effectiveness_rating=0.75,
        lessons_learned=[
            "Need more detailed financial projections",
            "Timeline concerns need addressing",
            "Board Chair very engaged with data"
        ],
        follow_up_actions=[
            "Prepare detailed ROI analysis",
            "Schedule one-on-one with CFO",
            "Develop phased implementation plan"
        ]
    )
    
    # Measure effectiveness
    metrics = engine.measure_influence_effectiveness(execution, strategy1)
    
    print(f"Execution Results:")
    print(f"  • Objective Achievement: {metrics.objective_achievement:.2f}")
    print(f"  • Consensus Level: {metrics.consensus_level:.2f}")
    print(f"  • Support Gained: {metrics.support_gained:.2f}")
    print(f"  • Opposition Reduced: {metrics.opposition_reduced:.2f}")
    print(f"  • Relationship Health: {metrics.long_term_relationship_health:.2f}")
    
    print(f"\n👥 Stakeholder-Specific Results:")
    for stakeholder_id, satisfaction in metrics.stakeholder_satisfaction.items():
        stakeholder_name = next(s.name for s in stakeholders if s.id == stakeholder_id)
        relationship_impact = metrics.relationship_impact.get(stakeholder_id, 0)
        print(f"  • {stakeholder_name}: Satisfaction {satisfaction:.2f}, "
              f"Relationship Impact {relationship_impact:+.2f}")
    
    # Demonstrate strategy optimization
    print(f"\n🔄 Strategy Optimization")
    print("-" * 30)
    
    # Create additional simulated metrics for optimization
    additional_metrics = [
        metrics,  # First execution
        # Simulate second execution with improvements
        type(metrics)(
            strategy_id=strategy1.id,
            execution_id="exec_demo_002",
            objective_achievement=0.82,
            stakeholder_satisfaction={
                "board_chair_001": 0.85,
                "ceo_001": 0.90,
                "cfo_001": 0.70,
                "board_member_001": 0.65
            },
            relationship_impact={
                "board_chair_001": 0.05,
                "ceo_001": 0.03,
                "cfo_001": 0.08,
                "board_member_001": 0.02
            },
            consensus_level=0.78,
            support_gained=0.80,
            opposition_reduced=0.65,
            long_term_relationship_health=0.82
        )
    ]
    
    optimization = engine.optimize_influence_strategy(strategy1, additional_metrics)
    
    print(f"Optimization Analysis:")
    print(f"  • Current Effectiveness: {optimization.current_effectiveness:.2f}")
    print(f"  • Expected Improvement: {optimization.expected_improvement:.2f}")
    print(f"  • Confidence Level: {optimization.confidence_level:.2f}")
    
    print(f"\n💡 Optimization Opportunities:")
    for opportunity in optimization.optimization_opportunities:
        print(f"  • {opportunity}")
    
    print(f"\n🔧 Recommended Changes:")
    for change in optimization.recommended_tactic_changes:
        print(f"  • {change.get('change_type', 'Unknown')}: {change.get('reason', 'No reason provided')}")
    
    print(f"\n⏰ Timing Adjustments:")
    for adjustment in optimization.timing_adjustments:
        print(f"  • {adjustment}")
    
    print(f"\n🎯 Target-Specific Refinements:")
    for target_id, refinements in optimization.target_approach_refinements.items():
        target_name = next(s.name for s in stakeholders if s.id == target_id)
        print(f"  • {target_name}:")
        for refinement in refinements:
            print(f"    - {refinement}")
    
    # Demonstrate tactic library
    print(f"\n📚 Available Influence Tactics Library")
    print("-" * 40)
    
    print(f"Total tactics available: {len(engine.tactic_library)}")
    for tactic in engine.tactic_library:
        print(f"\n🎯 {tactic.name}")
        print(f"   Type: {tactic.influence_type.value}")
        print(f"   Effectiveness: {tactic.effectiveness_score:.2f}")
        print(f"   Best Contexts: {', '.join([ctx.value for ctx in tactic.context_suitability])}")
        print(f"   Target Types: {', '.join(tactic.target_personality_types)}")
        print(f"   Description: {tactic.description}")
    
    print(f"\n✅ Influence Strategy Development Demo Complete!")
    print(f"The system successfully demonstrated:")
    print(f"  • Strategic influence planning for board engagement")
    print(f"  • Tactic selection and optimization")
    print(f"  • Effectiveness measurement and tracking")
    print(f"  • Continuous strategy optimization")
    print(f"  • Stakeholder-specific approach refinement")


if __name__ == "__main__":
    asyncio.run(demo_influence_strategy_development())