"""
Demo of Board Executive Mastery Integration Systems

This demo shows the integration between board executive mastery capabilities
and other ScrollIntel systems including strategic planning and communication.
"""

import asyncio
from datetime import datetime, date
from scrollintel.core.board_strategic_integration import (
    BoardStrategicIntegration,
    BoardFeedbackIntegration
)
from scrollintel.core.board_communication_integration import (
    BoardCommunicationIntegration,
    CommunicationChannel,
    CommunicationIntegrationConfig
)
from scrollintel.models.board_dynamics_models import Board
from scrollintel.engines.board_dynamics_engine import (
    BoardMember, Background, Priority, InfluenceLevel, 
    CommunicationStyle, DecisionPattern
)
from scrollintel.models.strategic_planning_models import TechnologyVision
from scrollintel.models.executive_communication_models import Message, MessageType


async def demo_board_strategic_integration():
    """Demo board strategic planning integration"""
    print("=== Board Strategic Planning Integration Demo ===\n")
    
    # Create mock board
    board_member = BoardMember(
        id="member_1",
        name="Sarah Chen",
        background=Background(
            industry_experience=["technology", "venture_capital"],
            functional_expertise=["strategy", "innovation"],
            education=["MBA", "Computer Science"],
            previous_roles=["CTO", "VP Engineering"],
            years_experience=18
        ),
        expertise_areas=["AI", "strategic planning", "innovation"],
        influence_level=InfluenceLevel.HIGH,
        communication_style=CommunicationStyle.VISIONARY,
        decision_making_pattern=DecisionPattern.INTUITIVE,
        priorities=[
            Priority(
                area="AI Innovation",
                importance=0.9,
                description="Drive AI technology leadership",
                timeline="2-3 years"
            ),
            Priority(
                area="Market Expansion",
                importance=0.7,
                description="Expand into new markets",
                timeline="1-2 years"
            )
        ]
    )
    
    board = Board(
        id="board_1",
        name="Technology Board",
        members=[board_member],
        committees=["Audit", "Technology"],
        governance_structure={"type": "technology_focused"},
        meeting_schedule=[],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Create technology vision
    vision = TechnologyVision(
        id="vision_1",
        title="AI-First Technology Strategy",
        description="Transform company through AI technology leadership",
        time_horizon=5,
        key_principles=["AI-first approach", "Innovation leadership"],
        strategic_objectives=["Market leadership in AI solutions", "Technology excellence"],
        success_criteria=["Market leadership in AI solutions"],
        market_assumptions=["AI market growth at 30% annually", "Competitive landscape intensifying"]
    )
    
    # Initialize strategic integration system
    strategic_integration = BoardStrategicIntegration()
    
    print("1. Creating board-aligned strategic plan...")
    try:
        # Create board-aligned strategic plan
        roadmap = await strategic_integration.create_board_aligned_strategic_plan(
            board, vision, 5
        )
        
        print(f"   ✓ Strategic roadmap created: {roadmap.name}")
        print(f"   ✓ Time horizon: {roadmap.time_horizon} years")
        print(f"   ✓ Technology bets: {len(roadmap.technology_bets)}")
        print(f"   ✓ Milestones: {len(roadmap.milestones)}")
        print(f"   ✓ Stakeholders: {', '.join(roadmap.stakeholders)}")
        
    except Exception as e:
        print(f"   ✗ Error creating strategic plan: {str(e)}")
    
    print("\n2. Integrating board feedback...")
    try:
        # Create mock feedback
        feedback = BoardFeedbackIntegration(
            feedback_id="feedback_1",
            board_member_id="member_1",
            strategic_element="AI Investment",
            feedback_type="timeline_adjustment",
            feedback_content="Accelerate AI development timeline by 6 months",
            impact_assessment=0.8,
            integration_status="pending",
            created_at=datetime.now()
        )
        
        # Mock roadmap for feedback integration
        mock_roadmap = roadmap if 'roadmap' in locals() else None
        if mock_roadmap:
            updated_roadmap = await strategic_integration.integrate_board_feedback(
                [feedback], mock_roadmap
            )
            print(f"   ✓ Feedback integrated successfully")
            print(f"   ✓ Updated roadmap: {updated_roadmap.name}")
        else:
            print("   ⚠ Skipping feedback integration (no roadmap available)")
            
    except Exception as e:
        print(f"   ✗ Error integrating feedback: {str(e)}")
    
    print("\n3. Tracking board approval...")
    try:
        # Track board approval
        voting_record = {"member_1": "approve"}
        approval_tracking = await strategic_integration.track_board_approval(
            "initiative_1", board, voting_record
        )
        
        print(f"   ✓ Approval tracking created")
        print(f"   ✓ Status: {approval_tracking.approval_status}")
        print(f"   ✓ Next review: {approval_tracking.next_review_date}")
        
    except Exception as e:
        print(f"   ✗ Error tracking approval: {str(e)}")


async def demo_board_communication_integration():
    """Demo board communication integration"""
    print("\n=== Board Communication Integration Demo ===\n")
    
    # Create mock board (reuse from strategic demo)
    board_member = BoardMember(
        id="member_1",
        name="Michael Rodriguez",
        background=Background(
            industry_experience=["finance", "technology"],
            functional_expertise=["finance", "risk management"],
            education=["MBA", "Finance"],
            previous_roles=["CFO", "Finance Director"],
            years_experience=22
        ),
        expertise_areas=["financial planning", "risk assessment"],
        influence_level=InfluenceLevel.HIGH,
        communication_style=CommunicationStyle.ANALYTICAL,
        decision_making_pattern=DecisionPattern.DATA_DRIVEN,
        priorities=[
            Priority(
                area="Financial Performance",
                importance=0.8,
                description="Optimize financial performance",
                timeline="ongoing"
            )
        ]
    )
    
    board = Board(
        id="board_1",
        name="Executive Board",
        members=[board_member],
        committees=["Audit", "Finance"],
        governance_structure={"type": "traditional"},
        meeting_schedule=[],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Create mock message
    message = Message(
        id="msg_1",
        content="Q3 financial results show 15% revenue growth with strong AI product performance. Board review requested for strategic investment decisions.",
        message_type=MessageType.PERFORMANCE_REPORT,
        technical_complexity=0.3,
        urgency_level="medium",
        key_points=[
            "15% revenue growth in Q3",
            "AI products performing well",
            "Strategic investment decisions needed"
        ],
        supporting_data={"revenue_growth": 0.15, "ai_performance": "strong"},
        created_at=datetime.now()
    )
    
    # Initialize communication integration system
    communication_integration = BoardCommunicationIntegration()
    
    print("1. Creating board contextual communication...")
    try:
        # Create integration config
        config = CommunicationIntegrationConfig(
            board_id="board_1",
            enabled_channels=[CommunicationChannel.EMAIL, CommunicationChannel.BOARD_MEETING],
            auto_adaptation=True,
            context_awareness_level="high",
            response_generation_mode="automatic",
            escalation_rules={}
        )
        
        # Create board contextual communication
        contextual_message = await communication_integration.create_board_contextual_communication(
            message, board, CommunicationChannel.EMAIL, config
        )
        
        print(f"   ✓ Contextual message created: {contextual_message.message_id}")
        print(f"   ✓ Channel: {contextual_message.channel.value}")
        print(f"   ✓ Board relevance score: {contextual_message.board_relevance_score:.2f}")
        print(f"   ✓ Urgency level: {contextual_message.urgency_level}")
        print(f"   ✓ Adapted versions: {len(contextual_message.adapted_versions)}")
        
    except Exception as e:
        print(f"   ✗ Error creating contextual communication: {str(e)}")
    
    print("\n2. Generating board-appropriate response...")
    try:
        # Generate board-appropriate response
        response_context = {"meeting_type": "board_meeting", "urgency": "medium"}
        response_generation = await communication_integration.generate_board_appropriate_response(
            message, board_member, response_context
        )
        
        print(f"   ✓ Response generated: {response_generation.response_id}")
        print(f"   ✓ Response tone: {response_generation.response_tone}")
        print(f"   ✓ Appropriateness score: {response_generation.board_appropriateness_score:.2f}")
        print(f"   ✓ Key messages: {len(response_generation.key_messages)}")
        print(f"   ✓ Follow-up actions: {len(response_generation.follow_up_actions)}")
        print(f"   ✓ Generated response preview: {response_generation.generated_response[:100]}...")
        
    except Exception as e:
        print(f"   ✗ Error generating response: {str(e)}")
    
    print("\n3. Building board context awareness...")
    try:
        # Create mock communication history
        communication_history = [message]  # Use the message we created
        
        # Build context awareness
        context_awareness = await communication_integration.build_board_context_awareness(
            communication_history, board, 30
        )
        
        print(f"   ✓ Context awareness built for {context_awareness['analysis_period']}")
        print(f"   ✓ Communication patterns analyzed")
        print(f"   ✓ Recurring themes identified: {len(context_awareness['recurring_themes'])}")
        print(f"   ✓ Member engagement tracked")
        print(f"   ✓ Context insights: {len(context_awareness['context_insights'])}")
        
    except Exception as e:
        print(f"   ✗ Error building context awareness: {str(e)}")


async def main():
    """Run the complete board integration demo"""
    print("ScrollIntel Board Executive Mastery Integration Demo")
    print("=" * 60)
    
    # Run strategic integration demo
    await demo_board_strategic_integration()
    
    # Run communication integration demo
    await demo_board_communication_integration()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("\nKey Integration Capabilities Demonstrated:")
    print("• Board-aligned strategic planning")
    print("• Board feedback integration")
    print("• Board approval tracking")
    print("• Board contextual communication")
    print("• Board-appropriate response generation")
    print("• Board context awareness building")


if __name__ == "__main__":
    asyncio.run(main())