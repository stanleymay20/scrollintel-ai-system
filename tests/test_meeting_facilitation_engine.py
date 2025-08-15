"""
Tests for Meeting Facilitation Engine
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.meeting_facilitation_engine import (
    MeetingFacilitationEngine, FacilitationPhase, InteractionType,
    FacilitationGuidance, MeetingFlow, EngagementMetrics,
    FacilitationIntervention, MeetingOutcome
)
from scrollintel.models.meeting_preparation_models import (
    MeetingPreparation, BoardMember, MeetingObjective, AgendaItem,
    MeetingType, PreparationStatus, ContentType
)


class TestMeetingFacilitationEngine:
    """Test cases for Meeting Facilitation Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create meeting facilitation engine instance"""
        return MeetingFacilitationEngine()
    
    @pytest.fixture
    def sample_preparation(self):
        """Create sample meeting preparation"""
        board_members = [
            BoardMember(
                id="member_1",
                name="John Smith",
                role="Board Chair",
                expertise_areas=["Finance", "Strategy"],
                communication_preferences={"style": "direct"},
                influence_level=0.9,
                typical_concerns=["Financial performance"],
                decision_patterns={"speed": "deliberate"}
            ),
            BoardMember(
                id="member_2",
                name="Sarah Johnson",
                role="Independent Director",
                expertise_areas=["Technology"],
                communication_preferences={"style": "analytical"},
                influence_level=0.7,
                typical_concerns=["Technology strategy"],
                decision_patterns={"speed": "quick"}
            )
        ]
        
        objectives = [
            MeetingObjective(
                id="obj_1",
                title="Financial Review",
                description="Review Q4 financial performance",
                priority=1,
                success_criteria=["Clear financial understanding"],
                required_decisions=["Approve budget"],
                stakeholders=["Board", "CFO"]
            )
        ]
        
        agenda_items = [
            AgendaItem(
                id="item_1",
                title="Financial Review",
                description="Q4 financial performance review",
                presenter="CFO",
                duration_minutes=30,
                content_type=ContentType.FINANCIAL_REPORT,
                objectives=["obj_1"],
                materials_required=["Financial reports"],
                key_messages=["Strong Q4 performance"],
                anticipated_questions=["What about Q1 outlook?"],
                decision_required=True,
                priority=1
            ),
            AgendaItem(
                id="item_2",
                title="Strategic Update",
                description="Strategic initiatives update",
                presenter="CEO",
                duration_minutes=20,
                content_type=ContentType.STRATEGIC_UPDATE,
                objectives=[],
                materials_required=["Strategy docs"],
                key_messages=["Progress on key initiatives"],
                anticipated_questions=["Timeline concerns?"],
                decision_required=False,
                priority=2
            )
        ]
        
        return MeetingPreparation(
            id="prep_test_001",
            meeting_id="meeting_test_001",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=datetime.now() + timedelta(days=1),
            board_members=board_members,
            objectives=objectives,
            agenda_items=agenda_items,
            content_materials=[],
            preparation_tasks=[],
            success_metrics=[],
            status=PreparationStatus.IN_PROGRESS,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def test_generate_facilitation_guidance(self, engine, sample_preparation):
        """Test generation of comprehensive facilitation guidance"""
        guidance_list = engine.generate_facilitation_guidance(
            sample_preparation, FacilitationPhase.PRE_MEETING
        )
        
        # Verify guidance structure
        assert len(guidance_list) > 0
        
        # Check for different types of guidance
        guidance_types = [g.guidance_type for g in guidance_list]
        assert "preparation" in guidance_types
        assert "discussion" in guidance_types
        
        # Verify guidance properties
        for guidance in guidance_list:
            assert guidance.id is not None
            assert guidance.phase in FacilitationPhase
            assert guidance.title is not None
            assert guidance.description is not None
            assert len(guidance.key_actions) > 0
            assert len(guidance.engagement_strategies) > 0
            assert len(guidance.potential_challenges) > 0
            assert len(guidance.mitigation_strategies) > 0
            assert len(guidance.success_indicators) > 0
    
    def test_monitor_meeting_flow(self, engine, sample_preparation):
        """Test real-time meeting flow monitoring"""
        current_time = datetime.now()
        current_agenda_item = "item_1"
        
        meeting_flow = engine.monitor_meeting_flow(
            sample_preparation, current_time, current_agenda_item
        )
        
        # Verify flow structure
        assert meeting_flow.id.startswith("flow_")
        assert meeting_flow.meeting_preparation_id == sample_preparation.id
        assert meeting_flow.current_phase in FacilitationPhase
        assert meeting_flow.current_agenda_item == current_agenda_item
        assert meeting_flow.elapsed_time >= 0
        assert meeting_flow.remaining_time >= 0
        assert meeting_flow.flow_status in ["on_schedule", "ahead_of_schedule", "behind_schedule"]
        assert 0 <= meeting_flow.engagement_level <= 1
        assert isinstance(meeting_flow.decision_progress, dict)
        assert len(meeting_flow.next_actions) >= 0
        assert len(meeting_flow.flow_adjustments) >= 0
    
    def test_track_engagement_metrics(self, engine, sample_preparation):
        """Test engagement metrics tracking"""
        current_time = datetime.now()
        
        metrics = engine.track_engagement_metrics(
            sample_preparation.meeting_id,
            sample_preparation.board_members,
            current_time
        )
        
        # Verify metrics structure
        assert metrics.id.startswith("metrics_")
        assert metrics.meeting_id == sample_preparation.meeting_id
        assert metrics.timestamp == current_time
        assert 0 <= metrics.overall_engagement <= 1
        assert len(metrics.individual_engagement) == len(sample_preparation.board_members)
        assert 0 <= metrics.participation_balance <= 1
        assert 0 <= metrics.discussion_quality <= 1
        assert 0 <= metrics.decision_momentum <= 1
        assert 0 <= metrics.energy_level <= 1
        
        # Verify individual engagement
        for member_id, engagement in metrics.individual_engagement.items():
            assert member_id in [m.id for m in sample_preparation.board_members]
            assert 0 <= engagement <= 1
    
    def test_suggest_facilitation_interventions(self, engine, sample_preparation):
        """Test facilitation intervention suggestions"""
        # Create mock meeting flow and engagement metrics
        meeting_flow = MeetingFlow(
            id="flow_test",
            meeting_preparation_id=sample_preparation.id,
            current_phase=FacilitationPhase.DISCUSSION,
            current_agenda_item="item_1",
            elapsed_time=45,
            remaining_time=15,
            flow_status="behind_schedule",
            engagement_level=0.5,  # Low engagement to trigger intervention
            decision_progress={"item_1": 0.3},
            next_actions=["Focus on decisions"],
            flow_adjustments=["Accelerate discussion"]
        )
        
        engagement_metrics = EngagementMetrics(
            id="metrics_test",
            meeting_id=sample_preparation.meeting_id,
            timestamp=datetime.now(),
            overall_engagement=0.5,  # Low engagement
            individual_engagement={"member_1": 0.6, "member_2": 0.4},
            participation_balance=0.5,  # Imbalanced
            discussion_quality=0.6,
            decision_momentum=0.4,  # Low momentum
            energy_level=0.5
        )
        
        interventions = engine.suggest_facilitation_interventions(
            meeting_flow, engagement_metrics
        )
        
        # Verify interventions
        assert len(interventions) > 0
        
        intervention_types = [i.intervention_type for i in interventions]
        assert "engagement_boost" in intervention_types
        assert "participation_balance" in intervention_types
        assert "time_management" in intervention_types
        assert "decision_facilitation" in intervention_types
        
        # Verify intervention structure
        for intervention in interventions:
            assert intervention.id.startswith("intervention_")
            assert intervention.meeting_id == sample_preparation.meeting_id
            assert intervention.intervention_type is not None
            assert intervention.trigger is not None
            assert intervention.description is not None
            assert len(intervention.actions_taken) > 0
            assert intervention.expected_outcome is not None
    
    def test_track_meeting_outcomes(self, engine, sample_preparation):
        """Test meeting outcome tracking"""
        completed_agenda_items = ["item_1", "item_2"]
        
        outcomes = engine.track_meeting_outcomes(
            sample_preparation, completed_agenda_items
        )
        
        # Verify outcomes
        assert len(outcomes) == len(completed_agenda_items)
        
        for outcome in outcomes:
            assert outcome.id.startswith("outcome_")
            assert outcome.meeting_id == sample_preparation.meeting_id
            assert outcome.agenda_item_id in completed_agenda_items
            assert outcome.outcome_type in ["decision", "information"]
            assert outcome.description is not None
            assert isinstance(outcome.decisions_made, list)
            assert isinstance(outcome.action_items, list)
            assert isinstance(outcome.follow_up_required, bool)
            assert isinstance(outcome.stakeholder_satisfaction, dict)
            assert 0 <= outcome.success_score <= 1
            
            # Verify stakeholder satisfaction
            for member_id, satisfaction in outcome.stakeholder_satisfaction.items():
                assert member_id in [m.id for m in sample_preparation.board_members]
                assert 0 <= satisfaction <= 1
    
    def test_generate_post_meeting_insights(self, engine, sample_preparation):
        """Test post-meeting insights generation"""
        # Create mock data
        meeting_flow = MeetingFlow(
            id="flow_test",
            meeting_preparation_id=sample_preparation.id,
            current_phase=FacilitationPhase.CLOSING,
            current_agenda_item=None,
            elapsed_time=60,
            remaining_time=0,
            flow_status="on_schedule",
            engagement_level=0.8,
            decision_progress={"item_1": 1.0},
            next_actions=[],
            flow_adjustments=[]
        )
        
        engagement_metrics = [
            EngagementMetrics(
                id="metrics_1",
                meeting_id=sample_preparation.meeting_id,
                timestamp=datetime.now() - timedelta(minutes=30),
                overall_engagement=0.7,
                individual_engagement={"member_1": 0.8, "member_2": 0.6},
                participation_balance=0.8,
                discussion_quality=0.7,
                decision_momentum=0.8,
                energy_level=0.7
            ),
            EngagementMetrics(
                id="metrics_2",
                meeting_id=sample_preparation.meeting_id,
                timestamp=datetime.now(),
                overall_engagement=0.8,
                individual_engagement={"member_1": 0.9, "member_2": 0.7},
                participation_balance=0.9,
                discussion_quality=0.8,
                decision_momentum=0.9,
                energy_level=0.8
            )
        ]
        
        outcomes = [
            MeetingOutcome(
                id="outcome_1",
                meeting_id=sample_preparation.meeting_id,
                agenda_item_id="item_1",
                outcome_type="decision",
                description="Financial review completed",
                decisions_made=["Budget approved"],
                action_items=[{"description": "Follow up", "assignee": "CFO"}],
                follow_up_required=True,
                stakeholder_satisfaction={"member_1": 0.9, "member_2": 0.8},
                success_score=0.85
            )
        ]
        
        insights = engine.generate_post_meeting_insights(
            sample_preparation, meeting_flow, engagement_metrics, outcomes
        )
        
        # Verify insights structure
        assert insights["meeting_id"] == sample_preparation.meeting_id
        assert 0 <= insights["overall_effectiveness"] <= 1
        assert "objective_achievement" in insights
        assert "engagement_analysis" in insights
        assert "time_management" in insights
        assert "decision_quality" in insights
        assert "success_factors" in insights
        assert "improvement_areas" in insights
        assert "recommendations" in insights
        assert "generated_at" in insights
        
        # Verify engagement analysis
        engagement_analysis = insights["engagement_analysis"]
        assert "average_engagement" in engagement_analysis
        assert "trend" in engagement_analysis
        assert engagement_analysis["trend"] in ["improving", "declining", "stable"]
        
        # Verify time management analysis
        time_analysis = insights["time_management"]
        assert "planned_duration" in time_analysis
        assert "actual_duration" in time_analysis
        assert "variance" in time_analysis
        assert "efficiency" in time_analysis
        
        # Verify decision quality analysis
        decision_analysis = insights["decision_quality"]
        assert "decisions_made" in decision_analysis
        assert "average_quality" in decision_analysis
    
    def test_facilitation_phases(self, engine, sample_preparation):
        """Test different facilitation phases"""
        phases = [
            FacilitationPhase.PRE_MEETING,
            FacilitationPhase.OPENING,
            FacilitationPhase.DISCUSSION,
            FacilitationPhase.DECISION,
            FacilitationPhase.CLOSING
        ]
        
        for phase in phases:
            guidance_list = engine.generate_facilitation_guidance(
                sample_preparation, phase
            )
            assert len(guidance_list) > 0
            
            # Verify phase-appropriate guidance
            if phase == FacilitationPhase.PRE_MEETING:
                assert any(g.guidance_type == "preparation" for g in guidance_list)
    
    def test_engagement_thresholds(self, engine):
        """Test engagement threshold loading and usage"""
        thresholds = engine._load_engagement_thresholds()
        
        assert "low" in thresholds
        assert "medium" in thresholds
        assert "high" in thresholds
        
        # Verify threshold values are reasonable
        assert 0 < thresholds["low"] < thresholds["medium"] < thresholds["high"] <= 1
    
    def test_facilitation_templates(self, engine):
        """Test facilitation template loading"""
        templates = engine._load_facilitation_templates()
        
        assert "opening" in templates
        assert "discussion" in templates
        assert "decision" in templates
        
        # Verify template structure
        for template_name, template in templates.items():
            assert "key_actions" in template
            assert "timing" in template
            assert isinstance(template["key_actions"], list)
            assert len(template["key_actions"]) > 0
    
    def test_flow_status_assessment(self, engine, sample_preparation):
        """Test meeting flow status assessment"""
        # Test on schedule
        status = engine._assess_flow_status(sample_preparation, 30, 30)
        assert status == "on_schedule"
        
        # Test behind schedule
        status = engine._assess_flow_status(sample_preparation, 40, 30)
        assert status == "behind_schedule"
        
        # Test ahead of schedule
        status = engine._assess_flow_status(sample_preparation, 20, 30)
        assert status == "ahead_of_schedule"
    
    def test_current_phase_determination(self, engine, sample_preparation):
        """Test current phase determination"""
        # Test opening phase
        phase = engine._determine_current_phase(sample_preparation, None, 5)
        assert phase == FacilitationPhase.OPENING
        
        # Test discussion phase
        phase = engine._determine_current_phase(sample_preparation, "item_1", 25)
        assert phase == FacilitationPhase.DISCUSSION
        
        # Test closing phase
        total_duration = sum(item.duration_minutes for item in sample_preparation.agenda_items)
        phase = engine._determine_current_phase(sample_preparation, None, total_duration - 5)
        assert phase == FacilitationPhase.CLOSING
    
    def test_engagement_calculations(self, engine, sample_preparation):
        """Test engagement calculation methods"""
        # Test overall engagement
        engagement = engine._calculate_overall_engagement(sample_preparation.board_members)
        assert 0 <= engagement <= 1
        
        # Test individual engagement
        for member in sample_preparation.board_members:
            individual_engagement = engine._calculate_individual_engagement(member)
            assert 0 <= individual_engagement <= 1
        
        # Test participation balance
        individual_engagement = {"member_1": 0.8, "member_2": 0.6}
        balance = engine._calculate_participation_balance(individual_engagement)
        assert 0 <= balance <= 1
    
    def test_error_handling(self, engine):
        """Test error handling in facilitation engine"""
        # Test with invalid preparation
        with pytest.raises(Exception):
            engine.generate_facilitation_guidance(None)
        
        # Test with invalid meeting flow data
        with pytest.raises(Exception):
            engine.monitor_meeting_flow(None, datetime.now())


class TestMeetingFacilitationIntegration:
    """Integration tests for meeting facilitation system"""
    
    @pytest.fixture
    def engine(self):
        return MeetingFacilitationEngine()
    
    def test_full_facilitation_workflow(self, engine):
        """Test complete meeting facilitation workflow"""
        # Setup
        board_members = [
            BoardMember(
                id="member_1",
                name="Test Member",
                role="Board Chair",
                expertise_areas=["Strategy"],
                communication_preferences={},
                influence_level=0.8,
                typical_concerns=[],
                decision_patterns={}
            )
        ]
        
        objectives = [
            MeetingObjective(
                id="obj_1",
                title="Test Objective",
                description="Test objective description",
                priority=1,
                success_criteria=["Success criteria"],
                required_decisions=["Test decision"],
                stakeholders=["Board"]
            )
        ]
        
        agenda_items = [
            AgendaItem(
                id="item_1",
                title="Test Item",
                description="Test agenda item",
                presenter="CEO",
                duration_minutes=30,
                content_type=ContentType.PRESENTATION,
                objectives=["obj_1"],
                materials_required=[],
                key_messages=["Key message"],
                anticipated_questions=[],
                decision_required=True,
                priority=1
            )
        ]
        
        preparation = MeetingPreparation(
            id="prep_integration_test",
            meeting_id="meeting_integration_test",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=datetime.now() + timedelta(days=1),
            board_members=board_members,
            objectives=objectives,
            agenda_items=agenda_items,
            content_materials=[],
            preparation_tasks=[],
            success_metrics=[],
            status=PreparationStatus.IN_PROGRESS,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Step 1: Generate facilitation guidance
        guidance = engine.generate_facilitation_guidance(preparation)
        assert len(guidance) > 0
        
        # Step 2: Monitor meeting flow
        current_time = datetime.now()
        meeting_flow = engine.monitor_meeting_flow(preparation, current_time, "item_1")
        assert meeting_flow is not None
        
        # Step 3: Track engagement metrics
        engagement_metrics = engine.track_engagement_metrics(
            preparation.meeting_id, board_members, current_time
        )
        assert engagement_metrics is not None
        
        # Step 4: Suggest interventions
        interventions = engine.suggest_facilitation_interventions(
            meeting_flow, engagement_metrics
        )
        assert isinstance(interventions, list)
        
        # Step 5: Track outcomes
        outcomes = engine.track_meeting_outcomes(preparation, ["item_1"])
        assert len(outcomes) > 0
        
        # Step 6: Generate insights
        insights = engine.generate_post_meeting_insights(
            preparation, meeting_flow, [engagement_metrics], outcomes
        )
        assert insights is not None
        assert "overall_effectiveness" in insights
        
        # Verify workflow completion
        assert meeting_flow.meeting_preparation_id == preparation.id
        assert engagement_metrics.meeting_id == preparation.meeting_id
        assert all(outcome.meeting_id == preparation.meeting_id for outcome in outcomes)
        assert insights["meeting_id"] == preparation.meeting_id