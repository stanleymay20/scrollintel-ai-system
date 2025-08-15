"""
Simple Tests for Board Executive Mastery System
Tests core functionality without complex dependencies
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock

from scrollintel.core.board_executive_mastery_system import (
    BoardExecutiveMasterySystem,
    BoardExecutiveMasteryConfig
)
from scrollintel.models.board_executive_mastery_models import (
    BoardExecutiveMasteryRequest,
    BoardInfo,
    BoardMemberProfile,
    ExecutiveProfile,
    CommunicationContext,
    PresentationRequirements,
    StrategicContext,
    MeetingContext,
    CredibilityContext,
    BoardMemberRole,
    CommunicationStyle,
    EngagementType
)

class TestBoardExecutiveMasterySystemSimple:
    """Simple tests for board executive mastery system"""
    
    @pytest.fixture
    def mastery_config(self):
        """Create test configuration"""
        return BoardExecutiveMasteryConfig(
            enable_real_time_adaptation=True,
            enable_predictive_analytics=True,
            enable_continuous_learning=True,
            board_confidence_threshold=0.85,
            executive_trust_threshold=0.80,
            strategic_alignment_threshold=0.90
        )
    
    @pytest.fixture
    def mastery_system(self, mastery_config):
        """Create test mastery system"""
        return BoardExecutiveMasterySystem(mastery_config)
    
    @pytest.fixture
    def sample_mastery_request(self):
        """Create sample mastery request"""
        board_member = BoardMemberProfile(
            id="member_1",
            name="John Smith",
            role=BoardMemberRole.CHAIR,
            background="CEO background",
            expertise_areas=["Strategy", "Operations"],
            communication_style=CommunicationStyle.RESULTS_ORIENTED,
            influence_level=0.9,
            decision_making_pattern="Data-driven",
            key_concerns=["Growth", "Risk"],
            relationship_dynamics={"member_2": 0.8},
            preferred_information_format="Executive summaries",
            trust_level=0.85
        )
        
        board_info = BoardInfo(
            id="board_1",
            company_name="Test Corp",
            board_size=5,
            members=[board_member],
            governance_structure={"committees": ["Audit", "Compensation"]},
            meeting_frequency="Monthly",
            decision_making_process="Consensus",
            current_priorities=["Growth", "Innovation"],
            recent_challenges=["Competition"],
            performance_metrics={"revenue_growth": 0.15}
        )
        
        executive = ExecutiveProfile(
            id="exec_1",
            name="Jane Doe",
            title="CTO",
            department="Technology",
            influence_level=0.8,
            communication_style=CommunicationStyle.ANALYTICAL,
            key_relationships=["member_1"],
            strategic_priorities=["Innovation"],
            trust_level=0.75
        )
        
        return BoardExecutiveMasteryRequest(
            id="req_1",
            board_info=board_info,
            executives=[executive],
            communication_context=CommunicationContext(
                engagement_type=EngagementType.BOARD_MEETING,
                audience_profiles=[board_member],
                key_messages=["Q3 Update"],
                sensitive_topics=["Budget"],
                desired_outcomes=["Approval"],
                time_constraints={"duration": 30},
                cultural_considerations=["Direct communication"]
            ),
            presentation_requirements=PresentationRequirements(
                presentation_type="Board Update",
                duration_minutes=30,
                audience_size=5,
                key_topics=["Performance"],
                data_requirements=["Metrics"],
                visual_preferences={"charts": True},
                interaction_level="High",
                follow_up_requirements=["Action items"]
            ),
            strategic_context=StrategicContext(
                current_strategy={"focus": "growth"},
                market_conditions={"growth_rate": 0.1},
                competitive_landscape={"leaders": 3},
                financial_position={"revenue": 1000000},
                risk_factors=["Competition"],
                growth_opportunities=["New markets"],
                stakeholder_expectations={"investors": "Growth"}
            ),
            meeting_context=MeetingContext(
                meeting_type="Board Meeting",
                agenda_items=["CEO Report"],
                expected_attendees=["Board"],
                decision_points=["Budget"],
                preparation_time=60,
                follow_up_requirements=["Minutes"],
                success_criteria=["Decisions"]
            ),
            credibility_context=CredibilityContext(
                current_credibility_level=0.75,
                credibility_challenges=["New role"],
                trust_building_opportunities=["Results"],
                reputation_factors={"expertise": 0.9},
                stakeholder_perceptions={"board": "Positive"},
                improvement_areas=["Communication"]
            ),
            success_criteria={"board_confidence": 0.85},
            timeline={"preparation": datetime.now()},
            created_at=datetime.now()
        )
    
    def test_system_initialization(self, mastery_config):
        """Test system initialization"""
        system = BoardExecutiveMasterySystem(mastery_config)
        
        assert system.config == mastery_config
        assert system.board_dynamics is not None
        assert system.executive_communication is not None
        assert system.presentation_design is not None
        assert system.strategic_recommendations is not None
        assert system.stakeholder_mapping is not None
        assert system.meeting_preparation is not None
        assert system.decision_analysis is not None
        assert system.credibility_building is not None
        assert len(system.active_engagements) == 0
        assert len(system.performance_metrics) == 0
        assert len(system.learning_history) == 0
    
    @pytest.mark.asyncio
    async def test_create_engagement_plan(self, mastery_system, sample_mastery_request):
        """Test creating engagement plan"""
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
            sample_mastery_request
        )
        
        assert engagement_plan is not None
        assert engagement_plan.board_id == sample_mastery_request.board_info.id
        assert engagement_plan.id in mastery_system.active_engagements
        assert engagement_plan.board_analysis is not None
        assert engagement_plan.stakeholder_map is not None
        assert engagement_plan.communication_strategy is not None
        assert engagement_plan.presentation_plan is not None
        assert engagement_plan.strategic_plan is not None
        assert engagement_plan.meeting_strategy is not None
        assert engagement_plan.credibility_plan is not None
    
    @pytest.mark.asyncio
    async def test_execute_board_interaction(self, mastery_system, sample_mastery_request):
        """Test executing board interaction"""
        # Create engagement plan first
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
            sample_mastery_request
        )
        
        # Execute interaction
        interaction_context = {"topic": "Q3 Results", "mood": "positive"}
        interaction_strategy = await mastery_system.execute_board_interaction(
            engagement_plan.id,
            interaction_context
        )
        
        assert interaction_strategy is not None
        assert interaction_strategy.engagement_id == engagement_plan.id
        assert interaction_strategy.interaction_context == interaction_context
        assert interaction_strategy.confidence_level > 0.0
        assert interaction_strategy.adapted_communication is not None
        assert interaction_strategy.strategic_responses is not None
        assert interaction_strategy.decision_support is not None
    
    @pytest.mark.asyncio
    async def test_validate_mastery_effectiveness(self, mastery_system, sample_mastery_request):
        """Test validating mastery effectiveness"""
        # Create engagement plan first
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
            sample_mastery_request
        )
        
        # Validate mastery
        validation_context = {
            "positive_board_feedback": True,
            "executive_endorsement": True,
            "strategic_approval": True
        }
        
        mastery_metrics = await mastery_system.validate_board_mastery_effectiveness(
            engagement_plan.id,
            validation_context
        )
        
        assert mastery_metrics is not None
        assert mastery_metrics.engagement_id == engagement_plan.id
        assert mastery_metrics.board_confidence_score > 0.0
        assert mastery_metrics.executive_trust_score > 0.0
        assert mastery_metrics.strategic_alignment_score > 0.0
        assert mastery_metrics.communication_effectiveness_score > 0.0
        assert mastery_metrics.stakeholder_influence_score > 0.0
        assert mastery_metrics.overall_mastery_score > 0.0
        assert mastery_metrics.meets_success_criteria is not None
        
        # Verify metrics are stored
        assert engagement_plan.id in mastery_system.performance_metrics
    
    @pytest.mark.asyncio
    async def test_optimize_mastery(self, mastery_system, sample_mastery_request):
        """Test optimizing mastery"""
        # Create engagement plan and metrics first
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
            sample_mastery_request
        )
        
        validation_context = {"positive_board_feedback": False}
        await mastery_system.validate_board_mastery_effectiveness(
            engagement_plan.id,
            validation_context
        )
        
        # Optimize mastery
        optimization_context = {"focus_areas": ["communication", "trust"]}
        optimized_plan = await mastery_system.optimize_board_executive_mastery(
            engagement_plan.id,
            optimization_context
        )
        
        assert optimized_plan is not None
        assert optimized_plan.id == engagement_plan.id
        assert mastery_system.active_engagements[engagement_plan.id] == optimized_plan
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, mastery_system):
        """Test getting system status"""
        system_status = await mastery_system.get_mastery_system_status()
        
        assert system_status is not None
        assert "system_status" in system_status
        assert "active_engagements" in system_status
        assert "total_validations" in system_status
        assert "performance_averages" in system_status
        assert "system_health" in system_status
        assert "configuration" in system_status
        assert "timestamp" in system_status
        
        assert system_status["system_status"] == "operational"
        assert system_status["active_engagements"] == 0
        assert system_status["total_validations"] == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mastery_system):
        """Test error handling for invalid operations"""
        # Test with invalid engagement ID
        with pytest.raises(ValueError, match="Engagement plan not found"):
            await mastery_system.execute_board_interaction("invalid_id", {})
        
        with pytest.raises(ValueError, match="Engagement plan not found"):
            await mastery_system.validate_board_mastery_effectiveness("invalid_id", {})
        
        with pytest.raises(ValueError, match="Engagement data not found"):
            await mastery_system.optimize_board_executive_mastery("invalid_id", {})
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test with custom thresholds
        config = BoardExecutiveMasteryConfig(
            enable_real_time_adaptation=False,
            enable_predictive_analytics=False,
            enable_continuous_learning=False,
            board_confidence_threshold=0.75,
            executive_trust_threshold=0.70,
            strategic_alignment_threshold=0.80
        )
        
        system = BoardExecutiveMasterySystem(config)
        assert system.config.board_confidence_threshold == 0.75
        assert system.config.executive_trust_threshold == 0.70
        assert system.config.strategic_alignment_threshold == 0.80
        assert system.config.enable_real_time_adaptation == False
        assert system.config.enable_predictive_analytics == False
        assert system.config.enable_continuous_learning == False
    
    def test_success_criteria_evaluation(self, mastery_system):
        """Test success criteria evaluation"""
        # Test meeting criteria
        meets_criteria = mastery_system._meets_success_criteria(0.9, 0.85, 0.95)
        assert meets_criteria == True
        
        # Test not meeting criteria
        meets_criteria = mastery_system._meets_success_criteria(0.7, 0.75, 0.8)
        assert meets_criteria == False
    
    def test_overall_mastery_score_calculation(self, mastery_system):
        """Test overall mastery score calculation"""
        score = mastery_system._calculate_overall_mastery_score(
            board_confidence=0.9,
            executive_trust=0.85,
            strategic_alignment=0.95,
            communication_effectiveness=0.8,
            stakeholder_influence=0.75
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high with good individual scores
    
    def test_confidence_level_calculation(self, mastery_system):
        """Test confidence level calculation"""
        # Mock objects with scores
        dynamics = Mock(confidence_score=0.8)
        communication = Mock(effectiveness_score=0.9)
        responses = Mock(quality_score=0.85)
        
        confidence = mastery_system._calculate_confidence_level(
            dynamics, communication, responses
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high with good scores
    
    @pytest.mark.asyncio
    async def test_continuous_learning(self, mastery_system, sample_mastery_request):
        """Test continuous learning functionality"""
        # Enable continuous learning
        mastery_system.config.enable_continuous_learning = True
        
        # Create engagement and execute interaction
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
            sample_mastery_request
        )
        
        interaction_context = {"topic": "test"}
        interaction_strategy = await mastery_system.execute_board_interaction(
            engagement_plan.id,
            interaction_context
        )
        
        # Verify learning data was captured
        assert len(mastery_system.learning_history) > 0
        
        # Verify learning data structure
        latest_learning = mastery_system.learning_history[-1]
        assert "engagement_id" in latest_learning
        assert "interaction_context" in latest_learning
        assert "confidence_level" in latest_learning
        assert "timestamp" in latest_learning
    
    @pytest.mark.asyncio
    async def test_multiple_engagements(self, mastery_system):
        """Test handling multiple engagements"""
        # Create multiple engagement plans
        requests = []
        for i in range(3):
            board_member = BoardMemberProfile(
                id=f"member_{i}",
                name=f"Member {i}",
                role=BoardMemberRole.INDEPENDENT_DIRECTOR,
                background="Background",
                expertise_areas=["Strategy"],
                communication_style=CommunicationStyle.ANALYTICAL,
                influence_level=0.8,
                decision_making_pattern="Data-driven",
                key_concerns=["Growth"],
                relationship_dynamics={},
                preferred_information_format="Reports",
                trust_level=0.8
            )
            
            board_info = BoardInfo(
                id=f"board_{i}",
                company_name=f"Company {i}",
                board_size=5,
                members=[board_member],
                governance_structure={},
                meeting_frequency="Monthly",
                decision_making_process="Consensus",
                current_priorities=["Growth"],
                recent_challenges=[],
                performance_metrics={}
            )
            
            request = BoardExecutiveMasteryRequest(
                id=f"req_{i}",
                board_info=board_info,
                executives=[],
                communication_context=CommunicationContext(
                    engagement_type=EngagementType.BOARD_MEETING,
                    audience_profiles=[board_member],
                    key_messages=["Update"],
                    sensitive_topics=[],
                    desired_outcomes=["Approval"],
                    time_constraints={},
                    cultural_considerations=[]
                ),
                presentation_requirements=PresentationRequirements(
                    presentation_type="Update",
                    duration_minutes=30,
                    audience_size=5,
                    key_topics=["Performance"],
                    data_requirements=[],
                    visual_preferences={},
                    interaction_level="Medium",
                    follow_up_requirements=[]
                ),
                strategic_context=StrategicContext(
                    current_strategy={},
                    market_conditions={},
                    competitive_landscape={},
                    financial_position={},
                    risk_factors=[],
                    growth_opportunities=[],
                    stakeholder_expectations={}
                ),
                meeting_context=MeetingContext(
                    meeting_type="Board Meeting",
                    agenda_items=[],
                    expected_attendees=[],
                    decision_points=[],
                    preparation_time=60,
                    follow_up_requirements=[],
                    success_criteria=[]
                ),
                credibility_context=CredibilityContext(
                    current_credibility_level=0.75,
                    credibility_challenges=[],
                    trust_building_opportunities=[],
                    reputation_factors={},
                    stakeholder_perceptions={},
                    improvement_areas=[]
                ),
                success_criteria={},
                timeline={},
                created_at=datetime.now()
            )
            requests.append(request)
        
        # Create engagement plans
        engagement_plans = []
        for request in requests:
            plan = await mastery_system.create_comprehensive_engagement_plan(request)
            engagement_plans.append(plan)
        
        # Verify all plans were created
        assert len(engagement_plans) == 3
        assert len(mastery_system.active_engagements) == 3
        
        # Verify each plan has unique ID
        plan_ids = [plan.id for plan in engagement_plans]
        assert len(set(plan_ids)) == 3  # All unique
        
        # Verify system status reflects multiple engagements
        system_status = await mastery_system.get_mastery_system_status()
        assert system_status["active_engagements"] == 3

if __name__ == "__main__":
    pytest.main([__file__])