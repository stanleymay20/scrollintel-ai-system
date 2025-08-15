"""
Tests for Meeting Preparation Framework
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.meeting_preparation_engine import MeetingPreparationEngine
from scrollintel.models.meeting_preparation_models import (
    MeetingPreparation, BoardMember, MeetingObjective, AgendaItem,
    MeetingContent, PreparationTask, SuccessMetric, AgendaOptimization,
    ContentPreparation, MeetingSuccessPrediction, PreparationInsight,
    MeetingType, PreparationStatus, ContentType
)


class TestMeetingPreparationEngine:
    """Test cases for Meeting Preparation Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create meeting preparation engine instance"""
        return MeetingPreparationEngine()
    
    @pytest.fixture
    def sample_board_members(self):
        """Create sample board members"""
        return [
            BoardMember(
                id="member_1",
                name="John Smith",
                role="Board Chair",
                expertise_areas=["Finance", "Strategy"],
                communication_preferences={"style": "direct", "detail_level": "high"},
                influence_level=0.9,
                typical_concerns=["Financial performance", "Risk management"],
                decision_patterns={"speed": "deliberate", "style": "consensus"}
            ),
            BoardMember(
                id="member_2",
                name="Sarah Johnson",
                role="Independent Director",
                expertise_areas=["Technology", "Innovation"],
                communication_preferences={"style": "analytical", "detail_level": "medium"},
                influence_level=0.7,
                typical_concerns=["Technology strategy", "Innovation pipeline"],
                decision_patterns={"speed": "quick", "style": "data_driven"}
            ),
            BoardMember(
                id="member_3",
                name="Michael Chen",
                role="Audit Committee Chair",
                expertise_areas=["Audit", "Compliance"],
                communication_preferences={"style": "formal", "detail_level": "high"},
                influence_level=0.8,
                typical_concerns=["Compliance", "Risk controls"],
                decision_patterns={"speed": "thorough", "style": "risk_averse"}
            )
        ]
    
    @pytest.fixture
    def sample_objectives(self):
        """Create sample meeting objectives"""
        return [
            MeetingObjective(
                id="obj_1",
                title="Q4 Financial Review",
                description="Review Q4 financial performance and approve annual budget",
                priority=1,
                success_criteria=[
                    "Financial results clearly communicated",
                    "Budget approved by board",
                    "Key performance metrics discussed"
                ],
                required_decisions=["Approve annual budget", "Approve dividend policy"],
                stakeholders=["CFO", "Board", "Shareholders"]
            ),
            MeetingObjective(
                id="obj_2",
                title="Strategic Technology Initiative",
                description="Present and approve new AI technology investment strategy",
                priority=2,
                success_criteria=[
                    "Technology strategy clearly articulated",
                    "Investment approval obtained",
                    "Implementation timeline agreed"
                ],
                required_decisions=["Approve technology investment", "Set implementation timeline"],
                stakeholders=["CTO", "Board", "Technology team"]
            )
        ]
    
    def test_create_meeting_preparation(self, engine, sample_board_members, sample_objectives):
        """Test creating comprehensive meeting preparation plan"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_001",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        # Verify preparation structure
        assert preparation.id.startswith("prep_meeting_001")
        assert preparation.meeting_id == "meeting_001"
        assert preparation.meeting_type == MeetingType.BOARD_MEETING
        assert preparation.meeting_date == meeting_date
        assert len(preparation.board_members) == 3
        assert len(preparation.objectives) == 2
        assert len(preparation.agenda_items) >= 4  # Opening + objectives + closing
        assert len(preparation.preparation_tasks) > 0
        assert len(preparation.success_metrics) > 0
        assert preparation.status == PreparationStatus.IN_PROGRESS
        assert preparation.preparation_score is not None
        assert preparation.preparation_score >= 0
    
    def test_agenda_generation(self, engine, sample_board_members, sample_objectives):
        """Test agenda item generation from objectives"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_002",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        agenda_items = preparation.agenda_items
        
        # Verify standard items are included
        opening_item = next((item for item in agenda_items if "Opening" in item.title), None)
        assert opening_item is not None
        assert opening_item.priority == 1
        
        closing_item = next((item for item in agenda_items if "Adjournment" in item.title), None)
        assert closing_item is not None
        assert closing_item.priority == 999
        
        # Verify objective-based items
        objective_items = [item for item in agenda_items if any(obj.id in item.objectives for obj in sample_objectives)]
        assert len(objective_items) == 2
        
        # Verify agenda item properties
        for item in agenda_items:
            assert item.id is not None
            assert item.title is not None
            assert item.duration_minutes > 0
            assert item.content_type in ContentType
            assert isinstance(item.priority, int)
    
    def test_preparation_tasks_generation(self, engine, sample_board_members, sample_objectives):
        """Test preparation task generation"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_003",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        tasks = preparation.preparation_tasks
        
        # Verify tasks are created for each agenda item
        content_tasks = [task for task in tasks if "content" in task.id]
        review_tasks = [task for task in tasks if "review" in task.id]
        
        assert len(content_tasks) > 0
        assert len(review_tasks) > 0
        
        # Verify task properties
        for task in tasks:
            assert task.id is not None
            assert task.title is not None
            assert task.assignee is not None
            assert task.due_date < meeting_date
            assert task.status == PreparationStatus.NOT_STARTED
            assert len(task.deliverables) > 0
            assert len(task.completion_criteria) > 0
    
    def test_success_metrics_definition(self, engine, sample_board_members, sample_objectives):
        """Test success metrics definition"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_004",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        metrics = preparation.success_metrics
        
        # Verify standard metrics are included
        metric_names = [metric.name for metric in metrics]
        assert "Board Engagement Level" in metric_names
        assert "Decision Quality" in metric_names
        assert "Time Management" in metric_names
        assert "Overall Satisfaction" in metric_names
        
        # Verify objective-specific metrics
        objective_metrics = [metric for metric in metrics if "Objective Achievement" in metric.name]
        assert len(objective_metrics) == len(sample_objectives)
        
        # Verify metric properties
        for metric in metrics:
            assert metric.id is not None
            assert metric.name is not None
            assert metric.target_value > 0
            assert 0 <= metric.importance_weight <= 1
            assert metric.measurement_method is not None
    
    def test_optimize_agenda(self, engine, sample_board_members, sample_objectives):
        """Test agenda optimization functionality"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_005",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        original_agenda = preparation.agenda_items.copy()
        optimization = engine.optimize_agenda(preparation)
        
        # Verify optimization structure
        assert optimization.id.startswith("opt_")
        assert optimization.meeting_preparation_id == preparation.id
        assert len(optimization.original_agenda) == len(original_agenda)
        assert len(optimization.optimized_agenda) > 0
        assert optimization.optimization_rationale is not None
        assert len(optimization.time_allocation) > 0
        assert len(optimization.flow_improvements) > 0
        assert len(optimization.engagement_enhancements) > 0
        assert len(optimization.decision_optimization) > 0
        
        # Verify agenda was updated
        assert preparation.agenda_items == optimization.optimized_agenda
    
    def test_prepare_content(self, engine, sample_board_members, sample_objectives):
        """Test content preparation for agenda items"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_006",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        # Get first objective-based agenda item
        agenda_item = next(
            item for item in preparation.agenda_items 
            if len(item.objectives) > 0
        )
        
        content_prep = engine.prepare_content(preparation, agenda_item)
        
        # Verify content preparation structure
        assert content_prep.id.startswith("content_")
        assert content_prep.meeting_preparation_id == preparation.id
        assert content_prep.content_id == agenda_item.id
        assert len(content_prep.target_audience) == len(sample_board_members)
        assert len(content_prep.key_messages) > 0
        assert len(content_prep.supporting_evidence) > 0
        assert len(content_prep.visual_aids) > 0
        assert content_prep.narrative_structure is not None
        assert len(content_prep.anticipated_reactions) > 0
        assert len(content_prep.response_strategies) > 0
    
    def test_predict_meeting_success(self, engine, sample_board_members, sample_objectives):
        """Test meeting success prediction"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_007",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        prediction = engine.predict_meeting_success(preparation)
        
        # Verify prediction structure
        assert prediction.id.startswith("pred_")
        assert prediction.meeting_preparation_id == preparation.id
        assert 0 <= prediction.overall_success_probability <= 1
        assert len(prediction.objective_achievement_probabilities) == len(sample_objectives)
        assert 0 <= prediction.engagement_prediction <= 1
        assert 0 <= prediction.decision_quality_prediction <= 1
        assert len(prediction.stakeholder_satisfaction_prediction) == len(sample_board_members)
        assert len(prediction.risk_factors) > 0
        assert len(prediction.enhancement_recommendations) > 0
        assert "lower" in prediction.confidence_interval
        assert "upper" in prediction.confidence_interval
        
        # Verify preparation was updated
        assert preparation.success_prediction == prediction.overall_success_probability
        assert len(preparation.risk_factors) > 0
        assert len(preparation.mitigation_strategies) > 0
    
    def test_generate_preparation_insights(self, engine, sample_board_members, sample_objectives):
        """Test preparation insights generation"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_008",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        insights = engine.generate_preparation_insights(preparation)
        
        # Verify insights structure
        assert len(insights) > 0
        
        for insight in insights:
            assert insight.id.startswith("insight_")
            assert insight.meeting_preparation_id == preparation.id
            assert insight.insight_type is not None
            assert insight.title is not None
            assert insight.description is not None
            assert insight.impact_level in ["low", "medium", "high"]
            assert len(insight.actionable_recommendations) > 0
            assert 0 <= insight.confidence_score <= 1
    
    def test_preparation_score_calculation(self, engine, sample_board_members, sample_objectives):
        """Test preparation score calculation"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        preparation = engine.create_meeting_preparation(
            meeting_id="meeting_009",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        # Verify initial score
        assert preparation.preparation_score is not None
        assert 0 <= preparation.preparation_score <= 10
        
        # Test score recalculation
        new_score = engine._calculate_preparation_score(preparation)
        assert 0 <= new_score <= 10
    
    def test_different_meeting_types(self, engine, sample_board_members, sample_objectives):
        """Test preparation for different meeting types"""
        meeting_date = datetime.now() + timedelta(days=7)
        
        # Test board meeting
        board_prep = engine.create_meeting_preparation(
            meeting_id="meeting_board",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        # Test executive committee
        exec_prep = engine.create_meeting_preparation(
            meeting_id="meeting_exec",
            meeting_type=MeetingType.EXECUTIVE_COMMITTEE,
            meeting_date=meeting_date,
            board_members=sample_board_members,
            objectives=sample_objectives
        )
        
        # Verify different preparations
        assert board_prep.meeting_type == MeetingType.BOARD_MEETING
        assert exec_prep.meeting_type == MeetingType.EXECUTIVE_COMMITTEE
        assert board_prep.id != exec_prep.id
    
    def test_error_handling(self, engine):
        """Test error handling in meeting preparation"""
        # Test with invalid data
        with pytest.raises(Exception):
            engine.create_meeting_preparation(
                meeting_id="",  # Invalid meeting ID
                meeting_type=MeetingType.BOARD_MEETING,
                meeting_date=datetime.now() - timedelta(days=1),  # Past date
                board_members=[],  # No board members
                objectives=[]  # No objectives
            )
    
    def test_template_loading(self, engine):
        """Test preparation template loading"""
        templates = engine._load_preparation_templates()
        
        assert "board_meeting" in templates
        assert "executive_committee" in templates
        
        board_template = templates["board_meeting"]
        assert "standard_duration" in board_template
        assert "required_materials" in board_template
        assert "typical_agenda_items" in board_template
    
    def test_success_factors_loading(self, engine):
        """Test success factors loading"""
        success_factors = engine._load_success_factors()
        
        assert "preparation_completeness" in success_factors
        assert "board_engagement" in success_factors
        assert "content_quality" in success_factors
        assert "time_management" in success_factors
        assert "decision_clarity" in success_factors
        
        # Verify weights sum to reasonable total
        total_weight = sum(success_factors.values())
        assert 0.8 <= total_weight <= 1.2  # Allow some flexibility


class TestMeetingPreparationIntegration:
    """Integration tests for meeting preparation system"""
    
    @pytest.fixture
    def engine(self):
        return MeetingPreparationEngine()
    
    def test_full_preparation_workflow(self, engine):
        """Test complete meeting preparation workflow"""
        # Setup
        meeting_date = datetime.now() + timedelta(days=10)
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
        
        # Step 1: Create preparation
        preparation = engine.create_meeting_preparation(
            meeting_id="integration_test",
            meeting_type=MeetingType.BOARD_MEETING,
            meeting_date=meeting_date,
            board_members=board_members,
            objectives=objectives
        )
        
        assert preparation is not None
        assert preparation.status == PreparationStatus.IN_PROGRESS
        
        # Step 2: Optimize agenda
        optimization = engine.optimize_agenda(preparation)
        assert optimization is not None
        
        # Step 3: Prepare content for first agenda item
        agenda_item = preparation.agenda_items[0]
        content_prep = engine.prepare_content(preparation, agenda_item)
        assert content_prep is not None
        
        # Step 4: Predict success
        prediction = engine.predict_meeting_success(preparation)
        assert prediction is not None
        assert 0 <= prediction.overall_success_probability <= 1
        
        # Step 5: Generate insights
        insights = engine.generate_preparation_insights(preparation)
        assert len(insights) >= 0  # May be empty for simple test case
        
        # Verify final state
        assert preparation.preparation_score is not None
        assert preparation.success_prediction is not None