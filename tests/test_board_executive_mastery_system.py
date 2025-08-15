"""
Tests for Board Executive Mastery System
Comprehensive testing for board and executive engagement mastery
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

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

class TestBoardExecutiveMasterySystem:
    """Test cases for Board Executive Mastery System"""
    
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
    def sample_board_member(self):
        """Create sample board member"""
        return BoardMemberProfile(
            id="board_member_1",
            name="John Smith",
            role=BoardMemberRole.CHAIR,
            background="Former CEO with 20 years experience",
            expertise_areas=["Strategy", "Operations", "Technology"],
            communication_style=CommunicationStyle.RESULTS_ORIENTED,
            influence_level=0.9,
            decision_making_pattern="Data-driven with quick decisions",
            key_concerns=["Growth", "Risk Management", "Innovation"],
            relationship_dynamics={"board_member_2": 0.8, "board_member_3": 0.7},
            preferred_information_format="Executive summaries with key metrics",
            trust_level=0.85
        )
    
    @pytest.fixture
    def sample_board_info(self, sample_board_member):
        """Create sample board info"""
        return BoardInfo(
            id="board_1",
            company_name="TechCorp Inc",
            board_size=7,
            members=[sample_board_member],
            governance_structure={"committees": ["Audit", "Compensation", "Nominating"]},
            meeting_frequency="Monthly",
            decision_making_process="Consensus with majority vote",
            current_priorities=["Digital Transformation", "Market Expansion", "Talent Acquisition"],
            recent_challenges=["Supply Chain Disruption", "Competitive Pressure"],
            performance_metrics={"revenue_growth": 0.15, "market_share": 0.25}
        )
    
    @pytest.fixture
    def sample_executive(self):
        """Create sample executive"""
        return ExecutiveProfile(
            id="exec_1",
            name="Jane Doe",
            title="Chief Technology Officer",
            department="Technology",
            influence_level=0.8,
            communication_style=CommunicationStyle.ANALYTICAL,
            key_relationships=["board_member_1", "board_member_2"],
            strategic_priorities=["Innovation", "Digital Transformation", "Team Building"],
            trust_level=0.75
        )
    
    @pytest.fixture
    def sample_mastery_request(self, sample_board_info, sample_executive):
        """Create sample mastery request"""
        return BoardExecutiveMasteryRequest(
            id="mastery_req_1",
            board_info=sample_board_info,
            executives=[sample_executive],
            communication_context=CommunicationContext(
                engagement_type=EngagementType.BOARD_MEETING,
                audience_profiles=[sample_board_info.members[0]],
                key_messages=["Q3 Performance Update", "Strategic Initiative Progress"],
                sensitive_topics=["Budget Constraints", "Personnel Changes"],
                desired_outcomes=["Board Approval", "Strategic Alignment"],
                time_constraints={"presentation_duration": 30, "qa_duration": 15},
                cultural_considerations=["Direct Communication", "Data-Driven Decisions"]
            ),
            presentation_requirements=PresentationRequirements(
                presentation_type="Board Update",
                duration_minutes=30,
                audience_size=7,
                key_topics=["Performance", "Strategy", "Operations"],
                data_requirements=["Financial Metrics", "KPIs", "Market Data"],
                visual_preferences={"charts": True, "tables": True, "infographics": False},
                interaction_level="High",
                follow_up_requirements=["Action Items", "Next Steps"]
            ),
            strategic_context=StrategicContext(
                current_strategy={"focus": "Growth", "timeline": "3 years"},
                market_conditions={"growth_rate": 0.08, "competition": "High"},
                competitive_landscape={"market_leaders": 3, "market_share": 0.25},
                financial_position={"revenue": 100000000, "profit_margin": 0.15},
                risk_factors=["Market Volatility", "Technology Disruption"],
                growth_opportunities=["New Markets", "Product Innovation"],
                stakeholder_expectations={"investors": "Growth", "employees": "Stability"}
            ),
            meeting_context=MeetingContext(
                meeting_type="Regular Board Meeting",
                agenda_items=["CEO Report", "Financial Update", "Strategic Review"],
                expected_attendees=["All Board Members", "C-Suite"],
                decision_points=["Budget Approval", "Strategic Initiative Authorization"],
                preparation_time=120,
                follow_up_requirements=["Meeting Minutes", "Action Items"],
                success_criteria=["Clear Decisions", "Stakeholder Alignment"]
            ),
            credibility_context=CredibilityContext(
                current_credibility_level=0.75,
                credibility_challenges=["New to Role", "Complex Technical Topics"],
                trust_building_opportunities=["Demonstrate Results", "Clear Communication"],
                reputation_factors={"technical_expertise": 0.9, "leadership_experience": 0.6},
                stakeholder_perceptions={"board": "Promising", "executives": "Competent"},
                improvement_areas=["Board Dynamics", "Strategic Communication"]
            ),
            success_criteria={"board_confidence": 0.85, "strategic_alignment": 0.90},
            timeline={"preparation": datetime.now(), "execution": datetime.now()},
            created_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_create_comprehensive_engagement_plan(self, mastery_system, sample_mastery_request):
        """Test creating comprehensive engagement plan"""
        # Mock engine methods
        with patch.object(mastery_system.board_dynamics, 'analyze_board_composition', new_callable=AsyncMock) as mock_analyze, \
             patch.object(mastery_system.stakeholder_mapping, 'map_key_stakeholders', new_callable=AsyncMock) as mock_map, \
             patch.object(mastery_system.executive_communication, 'develop_communication_strategy', new_callable=AsyncMock) as mock_comm, \
             patch.object(mastery_system.presentation_design, 'create_board_presentation_plan', new_callable=AsyncMock) as mock_pres, \
             patch.object(mastery_system.strategic_recommendations, 'develop_strategic_recommendations', new_callable=AsyncMock) as mock_strat, \
             patch.object(mastery_system.meeting_preparation, 'create_comprehensive_meeting_plan', new_callable=AsyncMock) as mock_meet, \
             patch.object(mastery_system.credibility_building, 'create_credibility_building_plan', new_callable=AsyncMock) as mock_cred:
            
            # Configure mocks
            mock_analyze.return_value = Mock(priorities=["Growth", "Innovation"])
            mock_map.return_value = Mock()
            mock_comm.return_value = Mock()
            mock_pres.return_value = Mock()
            mock_strat.return_value = Mock()
            mock_meet.return_value = Mock()
            mock_cred.return_value = Mock()
            
            # Create engagement plan
            engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
                sample_mastery_request
            )
            
            # Verify plan creation
            assert engagement_plan is not None
            assert engagement_plan.board_id == sample_mastery_request.board_info.id
            assert engagement_plan.id in mastery_system.active_engagements
            
            # Verify all engines were called
            mock_analyze.assert_called_once()
            mock_map.assert_called_once()
            mock_comm.assert_called_once()
            mock_pres.assert_called_once()
            mock_strat.assert_called_once()
            mock_meet.assert_called_once()
            mock_cred.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_board_interaction(self, mastery_system, sample_mastery_request):
        """Test executing board interaction"""
        # Create engagement plan first
        with patch.object(mastery_system.board_dynamics, 'analyze_board_composition', new_callable=AsyncMock) as mock_analyze, \
             patch.object(mastery_system.stakeholder_mapping, 'map_key_stakeholders', new_callable=AsyncMock), \
             patch.object(mastery_system.executive_communication, 'develop_communication_strategy', new_callable=AsyncMock), \
             patch.object(mastery_system.presentation_design, 'create_board_presentation_plan', new_callable=AsyncMock), \
             patch.object(mastery_system.strategic_recommendations, 'develop_strategic_recommendations', new_callable=AsyncMock), \
             patch.object(mastery_system.meeting_preparation, 'create_comprehensive_meeting_plan', new_callable=AsyncMock), \
             patch.object(mastery_system.credibility_building, 'create_credibility_building_plan', new_callable=AsyncMock):
            
            mock_analyze.return_value = Mock(priorities=["Growth", "Innovation"])
            
            engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
                sample_mastery_request
            )
        
        # Mock interaction methods
        with patch.object(mastery_system.board_dynamics, 'assess_real_time_dynamics', new_callable=AsyncMock) as mock_dynamics, \
             patch.object(mastery_system.executive_communication, 'adapt_real_time_communication', new_callable=AsyncMock) as mock_adapt, \
             patch.object(mastery_system.strategic_recommendations, 'generate_contextual_responses', new_callable=AsyncMock) as mock_responses, \
             patch.object(mastery_system.decision_analysis, 'provide_real_time_decision_support', new_callable=AsyncMock) as mock_decision:
            
            # Configure mocks
            mock_dynamics.return_value = Mock(confidence_score=0.8)
            mock_adapt.return_value = Mock(effectiveness_score=0.85)
            mock_responses.return_value = Mock(quality_score=0.9)
            mock_decision.return_value = Mock()
            
            # Execute interaction
            interaction_context = {"current_topic": "Q3 Results", "board_mood": "positive"}
            interaction_strategy = await mastery_system.execute_board_interaction(
                engagement_plan.id,
                interaction_context
            )
            
            # Verify interaction execution
            assert interaction_strategy is not None
            assert interaction_strategy.engagement_id == engagement_plan.id
            assert interaction_strategy.confidence_level > 0.0
            
            # Verify all methods were called
            mock_dynamics.assert_called_once()
            mock_adapt.assert_called_once()
            mock_responses.assert_called_once()
            mock_decision.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_board_mastery_effectiveness(self, mastery_system, sample_mastery_request):
        """Test validating board mastery effectiveness"""
        # Create engagement plan first
        with patch.object(mastery_system.board_dynamics, 'analyze_board_composition', new_callable=AsyncMock) as mock_analyze, \
             patch.object(mastery_system.stakeholder_mapping, 'map_key_stakeholders', new_callable=AsyncMock), \
             patch.object(mastery_system.executive_communication, 'develop_communication_strategy', new_callable=AsyncMock), \
             patch.object(mastery_system.presentation_design, 'create_board_presentation_plan', new_callable=AsyncMock), \
             patch.object(mastery_system.strategic_recommendations, 'develop_strategic_recommendations', new_callable=AsyncMock), \
             patch.object(mastery_system.meeting_preparation, 'create_comprehensive_meeting_plan', new_callable=AsyncMock), \
             patch.object(mastery_system.credibility_building, 'create_credibility_building_plan', new_callable=AsyncMock):
            
            mock_analyze.return_value = Mock(priorities=["Growth", "Innovation"])
            
            engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
                sample_mastery_request
            )
        
        # Validate mastery effectiveness
        validation_context = {
            "positive_board_feedback": True,
            "executive_endorsement": True,
            "strategic_approval": True,
            "clear_communication_feedback": True,
            "stakeholder_support": True
        }
        
        mastery_metrics = await mastery_system.validate_board_mastery_effectiveness(
            engagement_plan.id,
            validation_context
        )
        
        # Verify validation results
        assert mastery_metrics is not None
        assert mastery_metrics.engagement_id == engagement_plan.id
        assert mastery_metrics.board_confidence_score > 0.0
        assert mastery_metrics.executive_trust_score > 0.0
        assert mastery_metrics.strategic_alignment_score > 0.0
        assert mastery_metrics.overall_mastery_score > 0.0
        assert mastery_metrics.meets_success_criteria is not None
        
        # Verify metrics are stored
        assert engagement_plan.id in mastery_system.performance_metrics
    
    @pytest.mark.asyncio
    async def test_optimize_board_executive_mastery(self, mastery_system, sample_mastery_request):
        """Test optimizing board executive mastery"""
        # Create engagement plan and metrics first
        with patch.object(mastery_system.board_dynamics, 'analyze_board_composition', new_callable=AsyncMock) as mock_analyze, \
             patch.object(mastery_system.stakeholder_mapping, 'map_key_stakeholders', new_callable=AsyncMock), \
             patch.object(mastery_system.executive_communication, 'develop_communication_strategy', new_callable=AsyncMock), \
             patch.object(mastery_system.presentation_design, 'create_board_presentation_plan', new_callable=AsyncMock), \
             patch.object(mastery_system.strategic_recommendations, 'develop_strategic_recommendations', new_callable=AsyncMock), \
             patch.object(mastery_system.meeting_preparation, 'create_comprehensive_meeting_plan', new_callable=AsyncMock), \
             patch.object(mastery_system.credibility_building, 'create_credibility_building_plan', new_callable=AsyncMock):
            
            mock_analyze.return_value = Mock(priorities=["Growth", "Innovation"])
            
            engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
                sample_mastery_request
            )
        
        # Create metrics with low scores to trigger optimization
        validation_context = {"positive_board_feedback": False}
        mastery_metrics = await mastery_system.validate_board_mastery_effectiveness(
            engagement_plan.id,
            validation_context
        )
        
        # Mock optimization methods
        with patch.object(mastery_system.board_dynamics, 'optimize_board_engagement', new_callable=AsyncMock) as mock_opt_board, \
             patch.object(mastery_system.credibility_building, 'optimize_trust_building', new_callable=AsyncMock) as mock_opt_trust, \
             patch.object(mastery_system.strategic_recommendations, 'optimize_strategic_alignment', new_callable=AsyncMock) as mock_opt_strat:
            
            # Configure mocks
            mock_opt_board.return_value = Mock()
            mock_opt_trust.return_value = Mock()
            mock_opt_strat.return_value = Mock()
            
            # Optimize mastery
            optimization_context = {"focus_areas": ["board_confidence", "trust_building"]}
            optimized_plan = await mastery_system.optimize_board_executive_mastery(
                engagement_plan.id,
                optimization_context
            )
            
            # Verify optimization
            assert optimized_plan is not None
            assert optimized_plan.id == engagement_plan.id
            
            # Verify updated plan is stored
            assert mastery_system.active_engagements[engagement_plan.id] == optimized_plan
    
    @pytest.mark.asyncio
    async def test_get_mastery_system_status(self, mastery_system):
        """Test getting mastery system status"""
        # Mock engine status methods
        with patch.object(mastery_system.board_dynamics, 'get_engine_status', new_callable=AsyncMock) as mock_status1, \
             patch.object(mastery_system.executive_communication, 'get_engine_status', new_callable=AsyncMock) as mock_status2, \
             patch.object(mastery_system.presentation_design, 'get_engine_status', new_callable=AsyncMock) as mock_status3, \
             patch.object(mastery_system.strategic_recommendations, 'get_engine_status', new_callable=AsyncMock) as mock_status4, \
             patch.object(mastery_system.stakeholder_mapping, 'get_engine_status', new_callable=AsyncMock) as mock_status5, \
             patch.object(mastery_system.meeting_preparation, 'get_engine_status', new_callable=AsyncMock) as mock_status6, \
             patch.object(mastery_system.decision_analysis, 'get_engine_status', new_callable=AsyncMock) as mock_status7, \
             patch.object(mastery_system.credibility_building, 'get_engine_status', new_callable=AsyncMock) as mock_status8:
            
            # Configure mocks
            for mock_status in [mock_status1, mock_status2, mock_status3, mock_status4, 
                              mock_status5, mock_status6, mock_status7, mock_status8]:
                mock_status.return_value = {"status": "operational", "health": "good"}
            
            # Get system status
            system_status = await mastery_system.get_mastery_system_status()
            
            # Verify status structure
            assert system_status is not None
            assert "system_status" in system_status
            assert "active_engagements" in system_status
            assert "total_validations" in system_status
            assert "performance_averages" in system_status
            assert "system_health" in system_status
            assert "configuration" in system_status
            assert "timestamp" in system_status
            
            # Verify configuration
            config = system_status["configuration"]
            assert config["real_time_adaptation"] == True
            assert config["predictive_analytics"] == True
            assert config["continuous_learning"] == True
    
    def test_mastery_config_initialization(self):
        """Test mastery configuration initialization"""
        config = BoardExecutiveMasteryConfig(
            enable_real_time_adaptation=False,
            enable_predictive_analytics=False,
            enable_continuous_learning=False,
            board_confidence_threshold=0.75,
            executive_trust_threshold=0.70,
            strategic_alignment_threshold=0.80
        )
        
        assert config.enable_real_time_adaptation == False
        assert config.enable_predictive_analytics == False
        assert config.enable_continuous_learning == False
        assert config.board_confidence_threshold == 0.75
        assert config.executive_trust_threshold == 0.70
        assert config.strategic_alignment_threshold == 0.80
    
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
    async def test_error_handling_invalid_engagement_id(self, mastery_system):
        """Test error handling for invalid engagement ID"""
        with pytest.raises(ValueError, match="Engagement plan not found"):
            await mastery_system.execute_board_interaction(
                "invalid_id",
                {"context": "test"}
            )
        
        with pytest.raises(ValueError, match="Engagement plan not found"):
            await mastery_system.validate_board_mastery_effectiveness(
                "invalid_id",
                {"context": "test"}
            )
        
        with pytest.raises(ValueError, match="Engagement data not found"):
            await mastery_system.optimize_board_executive_mastery(
                "invalid_id",
                {"context": "test"}
            )
    
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
    
    @pytest.mark.asyncio
    async def test_continuous_learning(self, mastery_system, sample_mastery_request):
        """Test continuous learning functionality"""
        # Enable continuous learning
        mastery_system.config.enable_continuous_learning = True
        
        # Create mock interaction strategy
        interaction_strategy = Mock()
        interaction_strategy.engagement_id = "test_engagement"
        interaction_strategy.interaction_context = {"test": "context"}
        interaction_strategy.confidence_level = 0.85
        interaction_strategy.timestamp = datetime.now()
        
        # Test learning from interaction
        initial_history_length = len(mastery_system.learning_history)
        await mastery_system._learn_from_interaction(interaction_strategy)
        
        # Verify learning data was stored
        assert len(mastery_system.learning_history) == initial_history_length + 1
        
        # Verify learning data structure
        latest_learning = mastery_system.learning_history[-1]
        assert "engagement_id" in latest_learning
        assert "interaction_context" in latest_learning
        assert "confidence_level" in latest_learning
        assert "timestamp" in latest_learning

if __name__ == "__main__":
    pytest.main([__file__])