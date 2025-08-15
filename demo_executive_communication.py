"""
Demo script for Executive Communication System
"""

import asyncio
from datetime import datetime
from scrollintel.engines.executive_communication_engine import ExecutiveCommunicationSystem
from scrollintel.engines.strategic_narrative_engine import (
    StrategicNarrativeSystem, StrategicContext, NarrativeType
)
from scrollintel.engines.information_synthesis_engine import (
    InformationSynthesisSystem, ComplexInformation, InformationType
)
from scrollintel.models.executive_communication_models import (
    ExecutiveAudience, Message, ExecutiveLevel, CommunicationStyle, MessageType
)


async def demo_executive_communication():
    """Demonstrate executive communication adaptation"""
    print("🎯 Executive Communication System Demo")
    print("=" * 50)
    
    # Initialize the systems
    communication_system = ExecutiveCommunicationSystem()
    narrative_system = StrategicNarrativeSystem()
    synthesis_system = InformationSynthesisSystem()
    
    # Create a technical message
    technical_message = Message(
        id="tech_msg_001",
        content="""
        Our current API microservices architecture is experiencing scalability bottlenecks 
        due to inefficient containerization and lack of proper CI/CD pipelines. The system 
        latency has increased by 40% over the past quarter, and we're seeing throughput 
        degradation during peak loads. We need to implement horizontal scaling with 
        Kubernetes orchestration, optimize our caching layers, and establish proper 
        load balancing mechanisms. This technical debt is impacting our ability to 
        onboard new enterprise clients and could result in SLA violations.
        
        Recommended actions:
        1. Migrate to containerized microservices with Kubernetes
        2. Implement Redis caching for improved performance
        3. Establish automated CI/CD pipelines
        4. Upgrade load balancing infrastructure
        5. Conduct performance testing and monitoring
        """,
        message_type=MessageType.STRATEGIC_UPDATE,
        technical_complexity=0.9,
        urgency_level="high",
        key_points=[
            "System performance degraded by 40%",
            "Scalability bottlenecks affecting client onboarding",
            "Technical infrastructure modernization required",
            "Risk of SLA violations"
        ],
        supporting_data={
            "performance_degradation": "40%",
            "estimated_cost": 500000,
            "timeline": "3 months",
            "risk_level": "high"
        },
        created_at=datetime.now()
    )
    
    # Define different executive audiences
    audiences = [
        ExecutiveAudience(
            id="ceo_001",
            name="Sarah Johnson",
            title="Chief Executive Officer",
            executive_level=ExecutiveLevel.CEO,
            communication_style=CommunicationStyle.DIRECT,
            expertise_areas=["business", "strategy", "growth"],
            decision_making_pattern="quick_decisive",
            influence_level=1.0,
            preferred_communication_format="executive_summary",
            attention_span=5,  # 5 minutes
            detail_preference="low",
            risk_tolerance="medium",
            created_at=datetime.now()
        ),
        ExecutiveAudience(
            id="cto_001",
            name="Michael Chen",
            title="Chief Technology Officer",
            executive_level=ExecutiveLevel.CTO,
            communication_style=CommunicationStyle.ANALYTICAL,
            expertise_areas=["technical", "architecture", "engineering"],
            decision_making_pattern="analytical_thorough",
            influence_level=0.9,
            preferred_communication_format="detailed_report",
            attention_span=20,  # 20 minutes
            detail_preference="high",
            risk_tolerance="high",
            created_at=datetime.now()
        ),
        ExecutiveAudience(
            id="board_chair_001",
            name="Robert Williams",
            title="Chairman of the Board",
            executive_level=ExecutiveLevel.BOARD_CHAIR,
            communication_style=CommunicationStyle.STRATEGIC,
            expertise_areas=["governance", "strategy", "risk"],
            decision_making_pattern="consensus_building",
            influence_level=1.0,
            preferred_communication_format="board_presentation",
            attention_span=10,  # 10 minutes
            detail_preference="medium",
            risk_tolerance="low",
            created_at=datetime.now()
        )
    ]
    
    # Demonstrate adaptation for each audience
    for audience in audiences:
        print(f"\n📋 Adapting message for: {audience.name} ({audience.title})")
        print(f"   Executive Level: {audience.executive_level.value}")
        print(f"   Communication Style: {audience.communication_style.value}")
        print(f"   Attention Span: {audience.attention_span} minutes")
        print(f"   Detail Preference: {audience.detail_preference}")
        print("-" * 60)
        
        try:
            # Adapt the message
            adapted_message = communication_system.process_executive_communication(
                technical_message, audience
            )
            
            print(f"✅ Adaptation successful!")
            print(f"   Tone: {adapted_message.tone}")
            print(f"   Language Complexity: {adapted_message.language_complexity}")
            print(f"   Reading Time: {adapted_message.estimated_reading_time} minutes")
            print(f"   Effectiveness Score: {adapted_message.effectiveness_score:.2f}")
            
            print(f"\n📝 Executive Summary:")
            print(f"   {adapted_message.executive_summary}")
            
            print(f"\n🎯 Key Recommendations:")
            for i, rec in enumerate(adapted_message.key_recommendations, 1):
                print(f"   {i}. {rec}")
            
            print(f"\n📄 Adapted Content (first 200 chars):")
            print(f"   {adapted_message.adapted_content[:200]}...")
            
            print(f"\n🔍 Adaptation Rationale:")
            print(f"   {adapted_message.adaptation_rationale}")
            
            # Simulate effectiveness tracking
            if audience.executive_level == ExecutiveLevel.CEO:
                engagement_data = {
                    "engagement_score": 0.9,  # High engagement for CEO
                    "comprehension_score": 0.8,
                    "action_taken": True,
                    "decision_influenced": True,
                    "response_time": 10,  # Quick response
                    "follow_up_questions": 1
                }
            elif audience.executive_level == ExecutiveLevel.CTO:
                engagement_data = {
                    "engagement_score": 0.95,  # Very high for technical audience
                    "comprehension_score": 0.95,
                    "action_taken": True,
                    "decision_influenced": True,
                    "response_time": 30,  # More detailed review
                    "follow_up_questions": 3
                }
            else:  # Board Chair
                engagement_data = {
                    "engagement_score": 0.8,
                    "comprehension_score": 0.85,
                    "action_taken": True,
                    "decision_influenced": True,
                    "response_time": 45,  # Deliberative approach
                    "follow_up_questions": 2
                }
            
            effectiveness = communication_system.track_communication_effectiveness(
                adapted_message.id, audience.id, engagement_data
            )
            
            print(f"\n📊 Communication Effectiveness:")
            print(f"   Engagement Score: {effectiveness.engagement_score:.2f}")
            print(f"   Comprehension Score: {effectiveness.comprehension_score:.2f}")
            print(f"   Action Taken: {effectiveness.action_taken}")
            print(f"   Decision Influenced: {effectiveness.decision_influenced}")
            print(f"   Response Time: {effectiveness.response_time} minutes")
            
            # Get optimization recommendations
            recommendations = communication_system.get_optimization_recommendations(effectiveness)
            if recommendations:
                print(f"\n💡 Optimization Recommendations:")
                for category, recommendation in recommendations.items():
                    print(f"   {category.title()}: {recommendation}")
            else:
                print(f"\n✨ Communication was highly effective - no optimizations needed!")
                
        except Exception as e:
            print(f"❌ Error adapting message: {str(e)}")
    
    # Demonstrate different message types
    print(f"\n" + "=" * 60)
    print("🚨 Crisis Communication Example")
    print("=" * 60)
    
    crisis_message = Message(
        id="crisis_msg_001",
        content="""
        URGENT: We have detected a critical security vulnerability in our production 
        systems that could potentially expose customer data. Our security team has 
        identified unauthorized access attempts, and we need immediate action to 
        contain the breach and protect our customers. This requires board approval 
        for emergency security measures and potential customer notification.
        """,
        message_type=MessageType.CRISIS_COMMUNICATION,
        technical_complexity=0.6,
        urgency_level="critical",
        key_points=[
            "Critical security vulnerability detected",
            "Potential customer data exposure",
            "Immediate containment required",
            "Board approval needed for emergency measures"
        ],
        supporting_data={
            "severity": "critical",
            "affected_customers": "potentially_all",
            "estimated_cost": 1000000
        },
        created_at=datetime.now()
    )
    
    # Adapt crisis message for Board Chair
    board_chair = audiences[2]  # Board Chair audience
    crisis_adapted = communication_system.process_executive_communication(
        crisis_message, board_chair
    )
    
    print(f"📋 Crisis Communication for Board Chair:")
    print(f"   Tone: {crisis_adapted.tone}")
    print(f"   Effectiveness Score: {crisis_adapted.effectiveness_score:.2f}")
    print(f"\n📝 Executive Summary:")
    print(f"   {crisis_adapted.executive_summary}")
    print(f"\n🎯 Key Recommendations:")
    for i, rec in enumerate(crisis_adapted.key_recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n" + "=" * 60)
    print("📖 Strategic Narrative Development Demo")
    print("=" * 60)
    
    # Demonstrate strategic narrative development
    strategy_data = {
        "id": "digital_transformation",
        "name": "Digital Transformation Initiative",
        "focus": "transformation technology",
        "core_approach": "AI-driven operational excellence",
        "primary_benefit": "40% efficiency improvement",
        "secondary_benefit": "competitive market advantage",
        "value_proposition": "Transform operations through AI automation",
        "competitive_advantage": "First-mover advantage in AI-driven customer service",
        "risk_mitigation": "Phased rollout with continuous monitoring",
        "budget_request": "$3M over 12 months",
        "projected_impact": "Market leadership in customer satisfaction"
    }
    
    strategic_context = StrategicContext(
        company_position="Strong market position with growth potential",
        market_conditions="Rapidly evolving AI landscape",
        competitive_landscape="Traditional competitors slow to adopt AI",
        key_challenges=[
            "Legacy system integration",
            "Staff training requirements",
            "Customer adoption curve"
        ],
        opportunities=[
            "AI automation potential",
            "Enhanced customer experience",
            "Operational cost reduction"
        ],
        stakeholder_concerns=[
            "Implementation complexity",
            "ROI timeline",
            "Customer service disruption"
        ],
        success_metrics=[
            "Customer satisfaction scores",
            "Operational efficiency metrics",
            "Cost reduction percentage"
        ],
        timeline="12 months"
    )
    
    # Create strategic narrative for Board Chair
    print(f"📋 Creating Strategic Narrative for Board Chair...")
    board_narrative = narrative_system.create_strategic_narrative(
        strategy_data, board_chair, strategic_context
    )
    
    print(f"✅ Strategic Narrative Created!")
    print(f"   Title: {board_narrative.title}")
    print(f"   Type: {board_narrative.narrative_type.value}")
    print(f"   Structure: {board_narrative.structure.value}")
    print(f"   Impact Score: {board_narrative.impact_score:.2f}")
    
    print(f"\n🎯 Key Messages:")
    for i, message in enumerate(board_narrative.key_messages, 1):
        print(f"   {i}. {message}")
    
    print(f"\n🎭 Emotional Arc:")
    print(f"   {board_narrative.emotional_arc}")
    
    print(f"\n📢 Call to Action:")
    print(f"   {board_narrative.call_to_action}")
    
    print(f"\n🎨 Supporting Visuals:")
    for visual in board_narrative.supporting_visuals:
        print(f"   • {visual}")
    
    print(f"\n📝 Narrative Elements:")
    for i, element in enumerate(board_narrative.elements, 1):
        print(f"   {i}. {element.element_type.title()}: {element.content[:100]}...")
        print(f"      Emotional Tone: {element.emotional_tone}")
        print(f"      Relevance: {element.audience_relevance:.2f}")
    
    # Demonstrate impact assessment
    print(f"\n📊 Assessing Narrative Impact...")
    engagement_data = {
        "engagement_level": 0.9,
        "emotional_resonance": 0.85,
        "message_retention": 0.88,
        "action_likelihood": 0.92,
        "credibility_score": 0.87,
        "feedback": "Compelling narrative with clear strategic vision"
    }
    
    impact, recommendations = narrative_system.assess_and_optimize(
        board_narrative, board_chair, engagement_data
    )
    
    print(f"✅ Impact Assessment Complete!")
    print(f"   Overall Impact: {impact.overall_impact:.2f}")
    print(f"   Engagement Level: {impact.engagement_level:.2f}")
    print(f"   Emotional Resonance: {impact.emotional_resonance:.2f}")
    print(f"   Message Retention: {impact.message_retention:.2f}")
    print(f"   Action Likelihood: {impact.action_likelihood:.2f}")
    print(f"   Credibility Score: {impact.credibility_score:.2f}")
    
    if recommendations:
        print(f"\n💡 Optimization Recommendations:")
        for category, recommendation in recommendations.items():
            print(f"   {category.title()}: {recommendation}")
    else:
        print(f"\n✨ Narrative is highly effective - no optimizations needed!")
    
    print(f"\n" + "=" * 60)
    print("📊 Information Synthesis Demo")
    print("=" * 60)
    
    # Demonstrate information synthesis
    complex_information = ComplexInformation(
        id="financial_report_q3",
        title="Q3 Financial Performance and Market Analysis",
        information_type=InformationType.FINANCIAL_DATA,
        raw_content="""
        The third quarter financial results demonstrate strong performance across key metrics. 
        Revenue reached $15.2M, representing a 22% increase over Q2 and 35% year-over-year growth. 
        Gross margin improved to 68%, up from 64% in the previous quarter, driven by operational 
        efficiency improvements and premium product mix. Customer acquisition cost decreased by 18% 
        to $850 per customer, while customer lifetime value increased by 25% to $12,500. 
        
        Market analysis indicates continued expansion opportunities in the enterprise segment, 
        where we've captured 14% market share, up from 9% last quarter. Competitive positioning 
        remains strong with our AI-driven features providing significant differentiation. 
        
        Risk factors include potential economic headwinds and increased competition from well-funded 
        startups. Recommended actions include accelerating product development, expanding sales team, 
        and establishing strategic partnerships to maintain competitive advantage.
        """,
        data_points=[
            {"metric": "revenue", "value": 15200000, "change": 0.22, "period": "Q3"},
            {"metric": "gross_margin", "value": 0.68, "change": 0.04, "period": "Q3"},
            {"metric": "customer_acquisition_cost", "value": 850, "change": -0.18, "period": "Q3"},
            {"metric": "customer_lifetime_value", "value": 12500, "change": 0.25, "period": "Q3"},
            {"metric": "market_share", "value": 0.14, "change": 0.05, "period": "Q3"},
            {"metric": "yoy_growth", "value": 0.35, "change": 0.35, "period": "Q3"}
        ],
        source="Financial Systems & Market Research",
        complexity_score=0.8,
        urgency_level="high",
        stakeholders=["CEO", "CFO", "Board", "Investors"],
        created_at=datetime.now()
    )
    
    print(f"📋 Synthesizing Complex Financial Information for CEO...")
    
    # Synthesize for CEO
    ceo_synthesis = synthesis_system.process_complex_information(complex_information, audiences[0])  # CEO
    
    print(f"✅ Information Synthesis Complete!")
    print(f"   Title: {ceo_synthesis.title}")
    print(f"   Readability Score: {ceo_synthesis.readability_score:.2f}")
    print(f"   Confidence Level: {ceo_synthesis.confidence_level:.2f}")
    
    print(f"\n📝 Executive Summary:")
    print(f"   {ceo_synthesis.executive_summary}")
    
    print(f"\n💡 Key Insights:")
    for i, insight in enumerate(ceo_synthesis.key_insights, 1):
        print(f"   {i}. {insight}")
    
    print(f"\n📊 Critical Data Points:")
    for i, data_point in enumerate(ceo_synthesis.critical_data_points, 1):
        print(f"   {i}. {data_point.get('metric', 'Unknown')}: {data_point.get('value', 'N/A')}")
        if 'executive_relevance' in data_point:
            print(f"      Relevance: {data_point['executive_relevance']}")
    
    print(f"\n🎯 Recommendations:")
    for i, rec in enumerate(ceo_synthesis.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n⚠️ Risk Factors:")
    for i, risk in enumerate(ceo_synthesis.risk_factors, 1):
        print(f"   {i}. {risk}")
    
    print(f"\n📋 Next Steps:")
    for i, step in enumerate(ceo_synthesis.next_steps, 1):
        print(f"   {i}. {step}")
    
    # Demonstrate information prioritization
    print(f"\n📊 Information Prioritization Demo...")
    
    # Create multiple information items
    info_batch = [
        complex_information,
        ComplexInformation(
            id="routine_update",
            title="Weekly Operations Update",
            information_type=InformationType.OPERATIONAL_METRICS,
            raw_content="Routine weekly operations metrics showing normal performance levels.",
            data_points=[{"uptime": 0.995, "response_time": 120}],
            source="Operations Team",
            complexity_score=0.3,
            urgency_level="low",
            stakeholders=["Operations Manager"],
            created_at=datetime.now()
        ),
        ComplexInformation(
            id="security_alert",
            title="Critical Security Vulnerability",
            information_type=InformationType.RISK_ASSESSMENT,
            raw_content="Critical security vulnerability detected in production systems requiring immediate attention.",
            data_points=[{"severity": "critical", "affected_systems": 12, "risk_score": 9.5}],
            source="Security Team",
            complexity_score=0.9,
            urgency_level="critical",
            stakeholders=["CEO", "CTO", "CISO", "Board"],
            created_at=datetime.now()
        )
    ]
    
    priorities = synthesis_system.prioritize_information_batch(info_batch, audiences[0])  # CEO
    
    print(f"✅ Information Prioritized!")
    for i, priority in enumerate(priorities, 1):
        print(f"   {i}. Priority: {priority.priority_level.upper()}")
        print(f"      Information: {info_batch[i-1].title}")
        print(f"      Relevance: {priority.relevance_score:.2f}")
        print(f"      Urgency: {priority.urgency_score:.2f}")
        print(f"      Impact: {priority.impact_score:.2f}")
        print(f"      Reasoning: {priority.reasoning}")
        print()
    
    # Generate optimized executive summary
    print(f"📄 Generating Optimized Executive Summary...")
    optimized_summary = synthesis_system.generate_optimized_summary(ceo_synthesis, audiences[0])
    
    print(f"✅ Optimized Summary Generated!")
    print(f"\n{optimized_summary}")
    
    print(f"\n" + "=" * 60)
    print("✅ Executive Communication System Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_executive_communication())