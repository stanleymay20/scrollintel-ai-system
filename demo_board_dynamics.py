"""
Demo script for Board Dynamics Analysis Engine

This script demonstrates the capabilities of the board dynamics analysis engine,
including composition analysis, power structure mapping, meeting dynamics assessment,
and governance framework understanding.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.engines.board_dynamics_engine import (
    BoardDynamicsAnalysisEngine, BoardMember, Background, Priority, Relationship,
    InfluenceLevel, CommunicationStyle, DecisionPattern
)


def create_sample_board_members() -> List[BoardMember]:
    """Create sample board members for demonstration"""
    
    # Board Chair - Technology Executive
    chair = BoardMember(
        id="chair_001",
        name="Sarah Chen",
        background=Background(
            industry_experience=["technology", "software", "ai", "cloud_computing"],
            functional_expertise=["engineering", "product_strategy", "innovation", "digital_transformation"],
            education=["Stanford Computer Science", "MIT Sloan MBA"],
            previous_roles=["CEO", "CTO", "VP Engineering", "Chief Innovation Officer"],
            years_experience=20
        ),
        expertise_areas=["technology", "cybersecurity", "data_privacy", "ai_governance", "digital_strategy"],
        influence_level=InfluenceLevel.VERY_HIGH,
        communication_style=CommunicationStyle.VISIONARY,
        decision_making_pattern=DecisionPattern.QUICK_DECIDER,
        relationships=[
            Relationship("cfo_002", "strategic_alliance", 0.9, "bidirectional"),
            Relationship("audit_003", "professional_respect", 0.7, "bidirectional"),
            Relationship("marketing_004", "mentorship", 0.8, "outgoing")
        ],
        priorities=[
            Priority("digital_transformation", 0.9, "Lead company's AI and digital initiatives", "ongoing"),
            Priority("cybersecurity", 0.8, "Ensure robust security posture", "immediate"),
            Priority("innovation", 0.9, "Drive breakthrough product development", "long_term")
        ],
        tenure=5,
        committee_memberships=["executive", "technology", "strategy"]
    )
    
    # CFO - Financial Expert
    cfo = BoardMember(
        id="cfo_002",
        name="Michael Rodriguez",
        background=Background(
            industry_experience=["finance", "banking", "investment", "public_companies"],
            functional_expertise=["finance", "accounting", "risk_management", "investor_relations"],
            education=["Wharton Finance", "CPA Certification"],
            previous_roles=["CFO", "Finance Director", "Investment Banker", "Controller"],
            years_experience=18
        ),
        expertise_areas=["finance", "risk_management", "compliance", "investor_relations", "m&a"],
        influence_level=InfluenceLevel.HIGH,
        communication_style=CommunicationStyle.ANALYTICAL,
        decision_making_pattern=DecisionPattern.DATA_DRIVEN,
        relationships=[
            Relationship("chair_001", "strategic_alliance", 0.9, "bidirectional"),
            Relationship("audit_003", "professional_collaboration", 0.8, "bidirectional"),
            Relationship("independent_005", "advisory", 0.6, "bidirectional")
        ],
        priorities=[
            Priority("financial_performance", 0.9, "Optimize financial metrics and profitability", "ongoing"),
            Priority("risk_management", 0.8, "Implement comprehensive risk framework", "immediate"),
            Priority("investor_confidence", 0.7, "Maintain strong investor relationships", "ongoing")
        ],
        tenure=4,
        committee_memberships=["audit", "finance", "risk"]
    )
    
    # Audit Committee Chair - Governance Expert
    audit_chair = BoardMember(
        id="audit_003",
        name="Jennifer Washington",
        background=Background(
            industry_experience=["accounting", "consulting", "public_companies", "regulatory"],
            functional_expertise=["audit", "compliance", "governance", "risk_assessment"],
            education=["Harvard Business School", "CPA", "Certified Internal Auditor"],
            previous_roles=["Chief Audit Executive", "Partner at Big 4", "Compliance Officer"],
            years_experience=22
        ),
        expertise_areas=["audit", "compliance", "governance", "risk_assessment", "regulatory_affairs"],
        influence_level=InfluenceLevel.HIGH,
        communication_style=CommunicationStyle.DETAIL_ORIENTED,
        decision_making_pattern=DecisionPattern.DATA_DRIVEN,
        relationships=[
            Relationship("chair_001", "professional_respect", 0.7, "bidirectional"),
            Relationship("cfo_002", "professional_collaboration", 0.8, "bidirectional"),
            Relationship("independent_006", "coalition", 0.7, "bidirectional")
        ],
        priorities=[
            Priority("governance_excellence", 0.9, "Ensure best-in-class governance practices", "ongoing"),
            Priority("compliance", 0.9, "Maintain regulatory compliance", "immediate"),
            Priority("risk_oversight", 0.8, "Provide effective risk oversight", "ongoing")
        ],
        tenure=6,
        committee_memberships=["audit", "governance", "risk"]
    )
    
    # Marketing Executive - Customer Focus
    marketing_exec = BoardMember(
        id="marketing_004",
        name="David Kim",
        background=Background(
            industry_experience=["marketing", "consumer_goods", "technology", "digital_media"],
            functional_expertise=["marketing", "brand_management", "customer_experience", "digital_marketing"],
            education=["Northwestern Kellogg MBA", "Marketing Analytics Certification"],
            previous_roles=["CMO", "VP Marketing", "Brand Director", "Digital Marketing Head"],
            years_experience=16
        ),
        expertise_areas=["marketing", "brand_management", "customer_experience", "digital_transformation"],
        influence_level=InfluenceLevel.MEDIUM,
        communication_style=CommunicationStyle.RELATIONSHIP_FOCUSED,
        decision_making_pattern=DecisionPattern.COLLABORATIVE,
        relationships=[
            Relationship("chair_001", "mentee", 0.8, "incoming"),
            Relationship("independent_005", "collaboration", 0.6, "bidirectional"),
            Relationship("operations_007", "working_relationship", 0.5, "bidirectional")
        ],
        priorities=[
            Priority("customer_satisfaction", 0.9, "Enhance customer experience and loyalty", "ongoing"),
            Priority("brand_strength", 0.8, "Build and protect brand reputation", "long_term"),
            Priority("market_expansion", 0.7, "Drive growth in new markets", "medium_term")
        ],
        tenure=3,
        committee_memberships=["marketing", "strategy"]
    )
    
    # Independent Director - Industry Veteran
    independent_1 = BoardMember(
        id="independent_005",
        name="Patricia Thompson",
        background=Background(
            industry_experience=["technology", "telecommunications", "media", "international"],
            functional_expertise=["general_management", "strategy", "international_business", "m&a"],
            education=["Yale MBA", "International Business Certificate"],
            previous_roles=["CEO", "President", "General Manager", "Strategy Director"],
            years_experience=25
        ),
        expertise_areas=["general_management", "strategy", "international_business", "m&a", "board_governance"],
        influence_level=InfluenceLevel.HIGH,
        communication_style=CommunicationStyle.VISIONARY,
        decision_making_pattern=DecisionPattern.CONSENSUS_BUILDER,
        relationships=[
            Relationship("cfo_002", "advisory", 0.6, "bidirectional"),
            Relationship("marketing_004", "collaboration", 0.6, "bidirectional"),
            Relationship("independent_006", "alliance", 0.8, "bidirectional")
        ],
        priorities=[
            Priority("strategic_direction", 0.9, "Guide long-term strategic planning", "ongoing"),
            Priority("international_expansion", 0.8, "Support global growth initiatives", "medium_term"),
            Priority("board_effectiveness", 0.7, "Enhance board governance and effectiveness", "ongoing")
        ],
        tenure=4,
        committee_memberships=["strategy", "nominating", "governance"]
    )
    
    # Independent Director - Legal/Regulatory Expert
    independent_2 = BoardMember(
        id="independent_006",
        name="Robert Chen",
        background=Background(
            industry_experience=["legal", "technology", "regulatory", "public_policy"],
            functional_expertise=["legal", "regulatory_affairs", "public_policy", "intellectual_property"],
            education=["Harvard Law School", "Georgetown Public Policy"],
            previous_roles=["General Counsel", "Chief Legal Officer", "Regulatory Affairs Director"],
            years_experience=19
        ),
        expertise_areas=["legal", "regulatory_affairs", "intellectual_property", "data_privacy", "public_policy"],
        influence_level=InfluenceLevel.MEDIUM,
        communication_style=CommunicationStyle.ANALYTICAL,
        decision_making_pattern=DecisionPattern.DATA_DRIVEN,
        relationships=[
            Relationship("audit_003", "coalition", 0.7, "bidirectional"),
            Relationship("independent_005", "alliance", 0.8, "bidirectional"),
            Relationship("operations_007", "professional_respect", 0.5, "bidirectional")
        ],
        priorities=[
            Priority("legal_compliance", 0.9, "Ensure comprehensive legal compliance", "immediate"),
            Priority("regulatory_strategy", 0.8, "Navigate regulatory landscape effectively", "ongoing"),
            Priority("ip_protection", 0.7, "Protect intellectual property assets", "ongoing")
        ],
        tenure=3,
        committee_memberships=["audit", "governance", "legal"]
    )
    
    # Operations Executive - Operational Excellence
    operations_exec = BoardMember(
        id="operations_007",
        name="Lisa Anderson",
        background=Background(
            industry_experience=["operations", "manufacturing", "supply_chain", "technology"],
            functional_expertise=["operations", "supply_chain", "quality_management", "process_improvement"],
            education=["MIT Operations Research", "Six Sigma Black Belt"],
            previous_roles=["COO", "VP Operations", "Operations Director", "Plant Manager"],
            years_experience=17
        ),
        expertise_areas=["operations", "supply_chain", "quality_management", "process_improvement", "sustainability"],
        influence_level=InfluenceLevel.MEDIUM,
        communication_style=CommunicationStyle.RESULTS_ORIENTED,
        decision_making_pattern=DecisionPattern.DATA_DRIVEN,
        relationships=[
            Relationship("marketing_004", "working_relationship", 0.5, "bidirectional"),
            Relationship("independent_006", "professional_respect", 0.5, "bidirectional"),
            Relationship("cfo_002", "coordination", 0.6, "bidirectional")
        ],
        priorities=[
            Priority("operational_efficiency", 0.9, "Optimize operational performance", "ongoing"),
            Priority("quality_excellence", 0.8, "Maintain highest quality standards", "ongoing"),
            Priority("sustainability", 0.7, "Implement sustainable business practices", "long_term")
        ],
        tenure=2,
        committee_memberships=["operations", "sustainability"]
    )
    
    return [chair, cfo, audit_chair, marketing_exec, independent_1, independent_2, operations_exec]


def create_sample_meeting_data() -> Dict[str, Any]:
    """Create sample meeting data for demonstration"""
    return {
        "participation": {
            "chair_001": {
                "speaking_time": 420,  # 7 minutes
                "questions_asked": 5,
                "contributions": 4,
                "topic_leadership": ["strategy", "technology"]
            },
            "cfo_002": {
                "speaking_time": 360,  # 6 minutes
                "questions_asked": 3,
                "contributions": 3,
                "topic_leadership": ["finance", "risk"]
            },
            "audit_003": {
                "speaking_time": 300,  # 5 minutes
                "questions_asked": 4,
                "contributions": 2,
                "topic_leadership": ["governance", "compliance"]
            },
            "marketing_004": {
                "speaking_time": 240,  # 4 minutes
                "questions_asked": 2,
                "contributions": 2,
                "topic_leadership": ["customer", "brand"]
            },
            "independent_005": {
                "speaking_time": 300,  # 5 minutes
                "questions_asked": 3,
                "contributions": 3,
                "topic_leadership": ["strategy", "international"]
            },
            "independent_006": {
                "speaking_time": 180,  # 3 minutes
                "questions_asked": 2,
                "contributions": 1,
                "topic_leadership": ["legal", "regulatory"]
            },
            "operations_007": {
                "speaking_time": 200,  # 3.3 minutes
                "questions_asked": 1,
                "contributions": 2,
                "topic_leadership": ["operations"]
            }
        },
        "agenda_completion_rate": 0.85,
        "decision_quality_score": 0.8,
        "time_efficiency": 0.75,
        "average_engagement": 0.8,
        "conflict_resolution_score": 0.9,
        "interruption_rate": 0.15,
        "speaking_times": {
            "chair_001": 420,
            "cfo_002": 360,
            "audit_003": 300,
            "marketing_004": 240,
            "independent_005": 300,
            "independent_006": 180,
            "operations_007": 200
        },
        "questions": {
            "chair_001": 5,
            "cfo_002": 3,
            "audit_003": 4,
            "marketing_004": 2,
            "independent_005": 3,
            "independent_006": 2,
            "operations_007": 1
        },
        "interruptions": {
            "chair_001": {"cfo_002": 1, "marketing_004": 1},
            "cfo_002": {"audit_003": 1},
            "audit_003": {"independent_006": 1},
            "marketing_004": {},
            "independent_005": {"operations_007": 1},
            "independent_006": {},
            "operations_007": {}
        },
        "consensus_actions": {
            "chair_001": 3,
            "cfo_002": 2,
            "audit_003": 1,
            "marketing_004": 4,
            "independent_005": 5,
            "independent_006": 2,
            "operations_007": 1
        },
        "topic_avoidance_score": 0.1,
        "decision_delay_count": 1,
        "constructive_question_ratio": 0.85,
        "participation_variance": 0.25,
        "unresolved_conflicts": []
    }


def create_sample_board_info() -> Dict[str, Any]:
    """Create sample board information for governance analysis"""
    return {
        "board_size": 7,
        "independent_directors": 4,  # 4 out of 7 are independent
        "leadership_structure": "separate",  # Separate Chair and CEO
        "term_limits": True,
        "diversity_metrics": {
            "gender_diversity": 0.43,  # 3 out of 7
            "ethnic_diversity": 0.29,  # 2 out of 7
            "age_diversity": 0.8,
            "experience_diversity": 0.9
        },
        "committees": {
            "audit": {
                "members": 3,
                "chair": "audit_003",
                "independence_ratio": 1.0,
                "financial_expert": True
            },
            "compensation": {
                "members": 3,
                "chair": "independent_005",
                "independence_ratio": 1.0
            },
            "nominating": {
                "members": 3,
                "chair": "independent_006",
                "independence_ratio": 1.0
            },
            "technology": {
                "members": 3,
                "chair": "chair_001",
                "independence_ratio": 0.67
            },
            "strategy": {
                "members": 4,
                "chair": "independent_005",
                "independence_ratio": 0.75
            }
        },
        "voting_procedures": {
            "majority_rule": True,
            "written_consent": True,
            "proxy_voting": False
        },
        "quorum_requirements": {
            "minimum_members": 4,
            "minimum_independent": 2
        },
        "approval_authorities": {
            "major_transactions": "board",
            "executive_compensation": "compensation_committee",
            "audit_matters": "audit_committee",
            "routine_matters": "management"
        },
        "escalation_procedures": {
            "conflict_resolution": "independent_directors",
            "whistleblower": "audit_committee",
            "ethics_violations": "governance_committee"
        },
        "reporting_requirements": [
            "quarterly_financial_reports",
            "annual_governance_report",
            "risk_management_reports",
            "compliance_reports",
            "executive_compensation_reports",
            "cybersecurity_reports"
        ],
        "compliance_frameworks": [
            "sarbanes_oxley",
            "sec_regulations",
            "nasdaq_listing_standards",
            "gdpr",
            "ccpa",
            "industry_specific_regulations"
        ]
    }


async def demonstrate_board_dynamics_analysis():
    """Main demonstration function"""
    print("🏛️  Board Dynamics Analysis Engine Demo")
    print("=" * 50)
    
    # Initialize the engine
    engine = BoardDynamicsAnalysisEngine()
    
    # Create sample data
    print("\n📊 Creating sample board data...")
    board_members = create_sample_board_members()
    meeting_data = create_sample_meeting_data()
    board_info = create_sample_board_info()
    
    print(f"✅ Created {len(board_members)} board members")
    print(f"✅ Created meeting data with {len(meeting_data['participation'])} participants")
    print(f"✅ Created board governance information")
    
    # 1. Board Composition Analysis
    print("\n" + "=" * 50)
    print("1️⃣  BOARD COMPOSITION ANALYSIS")
    print("=" * 50)
    
    composition_analysis = engine.analyze_board_composition(board_members)
    
    print(f"\n📈 Board Composition Results:")
    print(f"   • Total Members: {len(composition_analysis.member_profiles)}")
    print(f"   • Expertise Areas Covered: {len(composition_analysis.expertise_coverage)}")
    print(f"   • Skill Gaps Identified: {len(composition_analysis.skill_gaps)}")
    print(f"   • Key Strengths: {len(composition_analysis.strengths)}")
    
    print(f"\n🎯 Expertise Coverage:")
    for expertise, members in list(composition_analysis.expertise_coverage.items())[:5]:
        print(f"   • {expertise}: {len(members)} member(s)")
    
    print(f"\n⚠️  Skill Gaps:")
    for gap in composition_analysis.skill_gaps[:5]:
        print(f"   • {gap}")
    
    print(f"\n📊 Experience Distribution:")
    for bucket, count in composition_analysis.experience_distribution.items():
        print(f"   • {bucket}: {count} member(s)")
    
    # 2. Power Structure Analysis
    print("\n" + "=" * 50)
    print("2️⃣  POWER STRUCTURE ANALYSIS")
    print("=" * 50)
    
    power_structure = engine.map_power_structures(board_members)
    
    print(f"\n🔗 Power Structure Results:")
    print(f"   • Influence Networks: {len(power_structure.influence_networks)}")
    print(f"   • Key Decision Makers: {len(power_structure.decision_makers)}")
    print(f"   • Coalition Groups: {len(power_structure.coalition_groups)}")
    print(f"   • Strong Relationships: {len(power_structure.key_relationships)}")
    
    print(f"\n👑 Key Decision Makers:")
    for decision_maker in power_structure.decision_makers:
        member = next(m for m in board_members if m.id == decision_maker)
        print(f"   • {member.name} ({decision_maker})")
    
    print(f"\n🤝 Coalition Groups:")
    for i, coalition in enumerate(power_structure.coalition_groups, 1):
        member_names = [next(m.name for m in board_members if m.id == mid) for mid in coalition]
        print(f"   • Coalition {i}: {', '.join(member_names)}")
    
    print(f"\n💪 Influence Networks:")
    for member_id, network in list(power_structure.influence_networks.items())[:3]:
        member_name = next(m.name for m in board_members if m.id == member_id)
        network_names = [next(m.name for m in board_members if m.id == nid) for nid in network]
        print(f"   • {member_name}: {', '.join(network_names)}")
    
    # 3. Meeting Dynamics Analysis
    print("\n" + "=" * 50)
    print("3️⃣  MEETING DYNAMICS ANALYSIS")
    print("=" * 50)
    
    dynamics_assessment = engine.assess_meeting_dynamics(meeting_data, board_members)
    
    print(f"\n📋 Meeting Dynamics Results:")
    print(f"   • Meeting Effectiveness: {dynamics_assessment.meeting_effectiveness:.2f}")
    print(f"   • Decision Efficiency: {dynamics_assessment.decision_efficiency:.2f}")
    print(f"   • Collaboration Quality: {dynamics_assessment.collaboration_quality:.2f}")
    print(f"   • Conflict Indicators: {len(dynamics_assessment.conflict_indicators)}")
    
    print(f"\n🗣️  Member Engagement Levels:")
    sorted_engagement = sorted(dynamics_assessment.engagement_levels.items(), 
                             key=lambda x: x[1], reverse=True)
    for member_id, engagement in sorted_engagement:
        member_name = next(m.name for m in board_members if m.id == member_id)
        print(f"   • {member_name}: {engagement:.2f}")
    
    print(f"\n📢 Communication Patterns:")
    comm_patterns = dynamics_assessment.communication_patterns
    if "dominant_speakers" in comm_patterns:
        dominant_names = [next(m.name for m in board_members if m.id == mid) 
                         for mid in comm_patterns["dominant_speakers"]]
        print(f"   • Dominant Speakers: {', '.join(dominant_names)}")
    
    if "consensus_builders" in comm_patterns:
        consensus_names = [next(m.name for m in board_members if m.id == mid) 
                          for mid in comm_patterns["consensus_builders"]]
        print(f"   • Consensus Builders: {', '.join(consensus_names)}")
    
    if dynamics_assessment.conflict_indicators:
        print(f"\n⚠️  Conflict Indicators:")
        for indicator in dynamics_assessment.conflict_indicators:
            print(f"   • {indicator.replace('_', ' ').title()}")
    
    # 4. Governance Framework Analysis
    print("\n" + "=" * 50)
    print("4️⃣  GOVERNANCE FRAMEWORK ANALYSIS")
    print("=" * 50)
    
    governance_analysis = engine.analyze_governance_framework(board_info)
    
    print(f"\n🏛️  Governance Analysis Results:")
    print(f"   • Overall Governance Score: {governance_analysis['overall_score']:.2f}")
    print(f"   • Governance Gaps: {len(governance_analysis['governance_gaps'])}")
    print(f"   • Recommendations: {len(governance_analysis['recommendations'])}")
    
    print(f"\n📊 Effectiveness Scores:")
    for area, score in governance_analysis['effectiveness_scores'].items():
        print(f"   • {area.replace('_', ' ').title()}: {score:.2f}")
    
    if governance_analysis['governance_gaps']:
        print(f"\n⚠️  Governance Gaps:")
        for gap in governance_analysis['governance_gaps']:
            print(f"   • {gap.replace('_', ' ').title()}")
    
    print(f"\n💡 Recommendations:")
    for rec in governance_analysis['recommendations'][:3]:
        print(f"   • {rec['area']}: {rec['recommendation']}")
        print(f"     Priority: {rec['priority']}, Timeline: {rec['timeline']}")
    
    # 5. Comprehensive Analysis
    print("\n" + "=" * 50)
    print("5️⃣  COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    comprehensive_analysis = engine.generate_comprehensive_analysis(
        board_members, meeting_data, board_info
    )
    
    print(f"\n🔍 Comprehensive Analysis Results:")
    print(f"   • Analysis Timestamp: {comprehensive_analysis['analysis_timestamp']}")
    print(f"   • Key Insights: {len(comprehensive_analysis['insights'])}")
    print(f"   • Action Recommendations: {len(comprehensive_analysis['recommendations'])}")
    
    print(f"\n💡 Key Insights:")
    for insight in comprehensive_analysis['insights']:
        print(f"   • {insight}")
    
    print(f"\n🎯 Action Recommendations:")
    for rec in comprehensive_analysis['recommendations']:
        print(f"   • {rec['area']}: {rec['recommendation']}")
        print(f"     Priority: {rec['priority']}, Timeline: {rec['timeline']}")
    
    # 6. Board Member Spotlight
    print("\n" + "=" * 50)
    print("6️⃣  BOARD MEMBER SPOTLIGHT")
    print("=" * 50)
    
    # Highlight the most influential member
    chair_member = next(m for m in board_members if m.id == "chair_001")
    
    print(f"\n⭐ Featured Member: {chair_member.name}")
    print(f"   • Role: Board Chair")
    print(f"   • Influence Level: {chair_member.influence_level.value}")
    print(f"   • Communication Style: {chair_member.communication_style.value}")
    print(f"   • Decision Pattern: {chair_member.decision_making_pattern.value}")
    print(f"   • Tenure: {chair_member.tenure} years")
    print(f"   • Expertise Areas: {', '.join(chair_member.expertise_areas[:3])}")
    print(f"   • Committee Memberships: {', '.join(chair_member.committee_memberships)}")
    
    print(f"\n🎯 Top Priorities:")
    for priority in chair_member.priorities:
        print(f"   • {priority.area}: {priority.description}")
        print(f"     Importance: {priority.importance:.1f}, Timeline: {priority.timeline}")
    
    print(f"\n🤝 Key Relationships:")
    for relationship in chair_member.relationships:
        related_member = next(m for m in board_members if m.id == relationship.member_id)
        print(f"   • {related_member.name}: {relationship.relationship_type} (strength: {relationship.strength:.1f})")
    
    # 7. Meeting Performance Summary
    print("\n" + "=" * 50)
    print("7️⃣  MEETING PERFORMANCE SUMMARY")
    print("=" * 50)
    
    total_speaking_time = sum(meeting_data['speaking_times'].values())
    total_questions = sum(meeting_data['questions'].values())
    
    print(f"\n📊 Meeting Statistics:")
    print(f"   • Total Speaking Time: {total_speaking_time // 60} minutes {total_speaking_time % 60} seconds")
    print(f"   • Total Questions Asked: {total_questions}")
    print(f"   • Agenda Completion: {meeting_data['agenda_completion_rate']:.1%}")
    print(f"   • Time Efficiency: {meeting_data['time_efficiency']:.1%}")
    print(f"   • Average Engagement: {meeting_data['average_engagement']:.1%}")
    
    print(f"\n🏆 Top Performers:")
    # Most active speaker
    most_active = max(meeting_data['speaking_times'].items(), key=lambda x: x[1])
    most_active_name = next(m.name for m in board_members if m.id == most_active[0])
    print(f"   • Most Active Speaker: {most_active_name} ({most_active[1]//60}:{most_active[1]%60:02d})")
    
    # Most inquisitive
    most_questions = max(meeting_data['questions'].items(), key=lambda x: x[1])
    most_questions_name = next(m.name for m in board_members if m.id == most_questions[0])
    print(f"   • Most Inquisitive: {most_questions_name} ({most_questions[1]} questions)")
    
    # Best consensus builder
    most_consensus = max(meeting_data['consensus_actions'].items(), key=lambda x: x[1])
    most_consensus_name = next(m.name for m in board_members if m.id == most_consensus[0])
    print(f"   • Best Consensus Builder: {most_consensus_name} ({most_consensus[1]} actions)")
    
    print("\n" + "=" * 50)
    print("✅ Board Dynamics Analysis Demo Complete!")
    print("=" * 50)
    
    return {
        "composition_analysis": composition_analysis,
        "power_structure": power_structure,
        "dynamics_assessment": dynamics_assessment,
        "governance_analysis": governance_analysis,
        "comprehensive_analysis": comprehensive_analysis
    }


def save_demo_results(results: Dict[str, Any]):
    """Save demo results to JSON file"""
    # Convert dataclass objects to dictionaries for JSON serialization
    serializable_results = {}
    
    for key, value in results.items():
        if hasattr(value, '__dict__'):
            serializable_results[key] = value.__dict__
        else:
            serializable_results[key] = value
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"board_dynamics_demo_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n💾 Demo results saved to: {filename}")


if __name__ == "__main__":
    print("Starting Board Dynamics Analysis Engine Demo...")
    
    # Run the demonstration
    results = asyncio.run(demonstrate_board_dynamics_analysis())
    
    # Save results
    save_demo_results(results)
    
    print("\n🎉 Demo completed successfully!")
    print("\nThe Board Dynamics Analysis Engine provides comprehensive insights into:")
    print("• Board composition and expertise coverage")
    print("• Power structures and influence networks")
    print("• Meeting dynamics and engagement patterns")
    print("• Governance framework effectiveness")
    print("• Actionable recommendations for improvement")
    print("\nThis enables ScrollIntel to excel in board relationships and executive leadership!")