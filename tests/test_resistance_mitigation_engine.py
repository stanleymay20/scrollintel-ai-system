"""
Tests for Cultural Change Resistance Mitigation Engine
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

from scrollintel.engines.resistance_mitigation_engine import ResistanceMitigationEngine
from scrollintel.models.resistance_mitigation_models import (
    MitigationPlan, MitigationExecution, ResistanceResolution,
    MitigationValidation, MitigationStrategy, InterventionType, MitigationStatus
)
from scrollintel.models.resistance_detection_models import (
    ResistanceDetection, ResistanceType, ResistanceSeverity
)
from scrollintel.models.cultural_assessment_models import Organization
from scrollintel.models.transformation_roadmap_models import Transformation


class TestResistanceMitigationEngine:
    """Test suite for ResistanceMitigationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create ResistanceMitigationEngine instance"""
        return ResistanceMitigationEngine()
    
    @pytest.fixture
    def sample_organization(self):
        """Create sample organization"""
        return Organization(
            id="org_001",
            name="Test Organization",
            cultural_dimensions={"collaboration": 0.7, "innovation": 0.6},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_score=0.75,
            assessment_date=datetime.now()
        )
    
    @pytest.fixture
    def sample_transformation(self):
        """Create sample transformation"""
        return Transformation(
            id="trans_001",
            organization_id="org_001",
            current_culture=None,
            target_culture=None,
            vision=None,
            roadmap=None,
            interventions=[],
            progress=0.4,
            start_date=datetime.now() - timedelta(days=30),
            target_completion=datetime.now() + timedelta(days=90)
        )
    
    @pytest.fixture
    def sample_detection(self):
        """Create sample resistance detection"""
        return ResistanceDetection(
            id="det_001",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            source=None,
            severity=ResistanceSeverity.MODERATE,
            confidence_score=0.8,
            detected_at=datetime.now(),
            indicators_triggered=["low_participation", "delayed_compliance"],
            affected_areas=["team_alpha", "department_beta"],
            potential_impact={"timeline_delay": 0.15},
            detection_method="behavioral_analysis",
            raw_data={}
        )
    
    def test_create_mitigation_plan_success(self, engine, sample_detection, sample_organization, sample_transformation):
        """Test successful mitigation plan creation"""
        constraints = {
            "budget_limit": 10000,
            "timeline_limit": 30,
            "resource_availability": 0.8
        }
        
        plan = engine.create_mitigation_plan(
            detection=sample_detection,
            organization=sample_organization,
            transformation=sample_transformation,
            constraints=constraints
        )
        
        assert isinstance(plan, MitigationPlan)
        assert plan.detection_id == sample_detection.id
        assert plan.organization_id == sample_organization.id
        assert plan.transformation_id == sample_detection.transformation_id
        assert plan.resistance_type == sample_detection.resistance_type
        assert plan.severity == sample_detection.severity
        assert isinstance(plan.strategies, list)
        assert len(plan.strategies) > 0
        assert isinstance(plan.interventions, list)
        assert isinstance(plan.target_stakeholders, list)
        assert isinstance(plan.success_criteria, dict)
        assert isinstance(plan.timeline, dict)
        assert isinstance(plan.resource_requirements, dict)
        assert isinstance(plan.risk_factors, list)
        assert isinstance(plan.contingency_plans, list)
    
    def test_create_mitigation_plan_different_resistance_types(self, engine, sample_organization, sample_transformation):
        """Test mitigation plan creation for different resistance types"""
        resistance_types = [
            ResistanceType.ACTIVE_OPPOSITION,
            ResistanceType.PASSIVE_RESISTANCE,
            ResistanceType.SKEPTICISM,
            ResistanceType.FEAR_BASED,
            ResistanceType.RESOURCE_BASED
        ]
        
        for resistance_type in resistance_types:
            detection = ResistanceDetection(
                id=f"det_{resistance_type.value}",
                organization_id="org_001",
                transformation_id="trans_001",
                resistance_type=resistance_type,
                source=None,
                severity=ResistanceSeverity.MODERATE,
                confidence_score=0.8,
                detected_at=datetime.now(),
                indicators_triggered=[],
                affected_areas=["test_area"],
                potential_impact={},
                detection_method="test",
                raw_data={}
            )
            
            plan = engine.create_mitigation_plan(
                detection=detection,
                organization=sample_organization,
                transformation=sample_transformation
            )
            
            assert isinstance(plan, MitigationPlan)
            assert plan.resistance_type == resistance_type
            assert len(plan.strategies) > 0
    
    def test_execute_mitigation_plan_success(self, engine, sample_organization):
        """Test successful mitigation plan execution"""
        from scrollintel.models.resistance_mitigation_models import MitigationIntervention
        
        # Create a sample plan with interventions
        plan = MitigationPlan(
            id="plan_001",
            detection_id="det_001",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            severity=ResistanceSeverity.MODERATE,
            strategies=[MitigationStrategy.TRAINING_SUPPORT],
            interventions=[
                MitigationIntervention(
                    id="int_001",
                    plan_id="plan_001",
                    intervention_type=InterventionType.TRAINING_SESSION,
                    strategy=MitigationStrategy.TRAINING_SUPPORT,
                    title="Test Training",
                    description="Test training session",
                    target_audience=["team_alpha"],
                    facilitators=["trainer_1"],
                    duration_hours=4.0,
                    scheduled_date=datetime.now(),
                    completion_date=None,
                    status=MitigationStatus.PLANNED,
                    success_metrics={},
                    actual_results={},
                    participant_feedback=[],
                    effectiveness_score=None,
                    lessons_learned=[],
                    follow_up_actions=[]
                )
            ],
            target_stakeholders=["team_alpha"],
            success_criteria={},
            timeline={},
            resource_requirements={},
            risk_factors=[],
            contingency_plans=[],
            created_at=datetime.now(),
            created_by="test"
        )
        
        execution = engine.execute_mitigation_plan(
            plan=plan,
            organization=sample_organization
        )
        
        assert isinstance(execution, MitigationExecution)
        assert execution.plan_id == plan.id
        assert execution.status == MitigationStatus.COMPLETED
        assert execution.progress_percentage == 100.0
        assert isinstance(execution.completed_interventions, list)
        assert isinstance(execution.interim_results, dict)
    
    def test_track_resistance_resolution_success(self, engine, sample_detection):
        """Test successful resistance resolution tracking"""
        from scrollintel.models.resistance_mitigation_models import MitigationIntervention
        
        plan = MitigationPlan(
            id="plan_001",
            detection_id=sample_detection.id,
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            severity=ResistanceSeverity.MODERATE,
            strategies=[MitigationStrategy.TRAINING_SUPPORT],
            interventions=[],
            target_stakeholders=[],
            success_criteria={},
            timeline={},
            resource_requirements={},
            risk_factors=[],
            contingency_plans=[],
            created_at=datetime.now(),
            created_by="test"
        )
        
        execution = MitigationExecution(
            id="exec_001",
            plan_id="plan_001",
            execution_phase="completed",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            status=MitigationStatus.COMPLETED,
            progress_percentage=100.0,
            completed_interventions=["int_001"],
            active_interventions=[],
            pending_interventions=[],
            resource_utilization={"facilitator_hours": 20},
            stakeholder_engagement={"team_alpha": 0.8},
            interim_results={"satisfaction": 0.85},
            challenges_encountered=[],
            adjustments_made=[],
            next_steps=[]
        )
        
        resolution = engine.track_resistance_resolution(
            detection=sample_detection,
            plan=plan,
            execution=execution
        )
        
        assert isinstance(resolution, ResistanceResolution)
        assert resolution.detection_id == sample_detection.id
        assert resolution.plan_id == plan.id
        assert isinstance(resolution.effectiveness_rating, float)
        assert 0 <= resolution.effectiveness_rating <= 1
        assert isinstance(resolution.stakeholder_satisfaction, dict)
        assert isinstance(resolution.behavioral_changes, list)
        assert isinstance(resolution.cultural_impact, dict)
        assert isinstance(resolution.lessons_learned, list)
        assert isinstance(resolution.recommendations, list)
        assert isinstance(resolution.follow_up_required, bool)
    
    def test_validate_mitigation_effectiveness_success(self, engine):
        """Test successful mitigation effectiveness validation"""
        from scrollintel.models.resistance_mitigation_models import MitigationIntervention
        
        plan = MitigationPlan(
            id="plan_001",
            detection_id="det_001",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            severity=ResistanceSeverity.MODERATE,
            strategies=[MitigationStrategy.TRAINING_SUPPORT],
            interventions=[],
            target_stakeholders=[],
            success_criteria={
                "resistance_reduction": 0.7,
                "engagement_improvement": 0.2,
                "sentiment_improvement": 0.3
            },
            timeline={},
            resource_requirements={},
            risk_factors=[],
            contingency_plans=[],
            created_at=datetime.now(),
            created_by="test"
        )
        
        execution = MitigationExecution(
            id="exec_001",
            plan_id="plan_001",
            execution_phase="completed",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            status=MitigationStatus.COMPLETED,
            progress_percentage=100.0,
            completed_interventions=[],
            active_interventions=[],
            pending_interventions=[],
            resource_utilization={},
            stakeholder_engagement={},
            interim_results={},
            challenges_encountered=[],
            adjustments_made=[],
            next_steps=[]
        )
        
        post_intervention_data = {
            "engagement_scores": {"before": 0.6, "after": 0.82},
            "sentiment_scores": {"before": -0.2, "after": 0.1},
            "behavioral_compliance": {"before": 0.65, "after": 0.88},
            "stakeholder_feedback": [
                {"stakeholder": "team_alpha", "satisfaction": 0.85},
                {"stakeholder": "team_beta", "satisfaction": 0.78}
            ]
        }
        
        validation = engine.validate_mitigation_effectiveness(
            plan=plan,
            execution=execution,
            post_intervention_data=post_intervention_data
        )
        
        assert isinstance(validation, MitigationValidation)
        assert validation.plan_id == plan.id
        assert isinstance(validation.success_criteria_met, dict)
        assert isinstance(validation.quantitative_results, dict)
        assert isinstance(validation.qualitative_feedback, list)
        assert isinstance(validation.stakeholder_assessments, dict)
        assert isinstance(validation.behavioral_indicators, dict)
        assert isinstance(validation.cultural_metrics, dict)
        assert isinstance(validation.sustainability_assessment, float)
        assert 0 <= validation.sustainability_assessment <= 1
        assert isinstance(validation.improvement_recommendations, list)
        assert isinstance(validation.validation_confidence, float)
        assert 0 <= validation.validation_confidence <= 1
    
    def test_select_mitigation_strategies(self, engine, sample_organization, sample_transformation):
        """Test mitigation strategy selection logic"""
        # Test different resistance types get different strategies
        detection_active = ResistanceDetection(
            id="det_active",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.ACTIVE_OPPOSITION,
            source=None,
            severity=ResistanceSeverity.HIGH,
            confidence_score=0.9,
            detected_at=datetime.now(),
            indicators_triggered=[],
            affected_areas=[],
            potential_impact={},
            detection_method="test",
            raw_data={}
        )
        
        strategies_active = engine._select_mitigation_strategies(
            detection_active, sample_organization, sample_transformation, None
        )
        
        assert isinstance(strategies_active, list)
        assert len(strategies_active) > 0
        assert MitigationStrategy.STAKEHOLDER_ENGAGEMENT in strategies_active
        assert MitigationStrategy.LEADERSHIP_INTERVENTION in strategies_active
        
        detection_passive = ResistanceDetection(
            id="det_passive",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            source=None,
            severity=ResistanceSeverity.MODERATE,
            confidence_score=0.8,
            detected_at=datetime.now(),
            indicators_triggered=[],
            affected_areas=[],
            potential_impact={},
            detection_method="test",
            raw_data={}
        )
        
        strategies_passive = engine._select_mitigation_strategies(
            detection_passive, sample_organization, sample_transformation, None
        )
        
        assert isinstance(strategies_passive, list)
        assert len(strategies_passive) > 0
        assert MitigationStrategy.INCENTIVE_ALIGNMENT in strategies_passive
        assert MitigationStrategy.TRAINING_SUPPORT in strategies_passive
    
    def test_design_interventions(self, engine, sample_detection, sample_organization):
        """Test intervention design functionality"""
        strategies = [MitigationStrategy.TRAINING_SUPPORT, MitigationStrategy.COMMUNICATION_ENHANCEMENT]
        
        interventions = engine._design_interventions(
            detection=sample_detection,
            strategies=strategies,
            organization=sample_organization,
            constraints=None
        )
        
        assert isinstance(interventions, list)
        assert len(interventions) > 0
        
        for intervention in interventions:
            assert hasattr(intervention, 'id')
            assert hasattr(intervention, 'intervention_type')
            assert hasattr(intervention, 'strategy')
            assert hasattr(intervention, 'title')
            assert hasattr(intervention, 'description')
            assert hasattr(intervention, 'target_audience')
            assert hasattr(intervention, 'duration_hours')
            assert intervention.strategy in strategies
    
    def test_addressing_strategies_initialization(self, engine):
        """Test addressing strategies are properly initialized"""
        assert hasattr(engine, 'addressing_strategies')
        assert isinstance(engine.addressing_strategies, list)
        assert len(engine.addressing_strategies) > 0
        
        strategy = engine.addressing_strategies[0]
        assert hasattr(strategy, 'id')
        assert hasattr(strategy, 'resistance_type')
        assert hasattr(strategy, 'strategy_name')
        assert hasattr(strategy, 'description')
        assert hasattr(strategy, 'approach_steps')
        assert hasattr(strategy, 'success_rate')
    
    def test_mitigation_templates_initialization(self, engine):
        """Test mitigation templates are properly initialized"""
        assert hasattr(engine, 'mitigation_templates')
        assert isinstance(engine.mitigation_templates, list)
        assert len(engine.mitigation_templates) > 0
        
        template = engine.mitigation_templates[0]
        assert hasattr(template, 'id')
        assert hasattr(template, 'template_name')
        assert hasattr(template, 'resistance_types')
        assert hasattr(template, 'severity_levels')
        assert hasattr(template, 'template_strategies')
        assert hasattr(template, 'template_interventions')
    
    def test_error_handling(self, engine):
        """Test error handling in mitigation engine"""
        with pytest.raises(Exception):
            engine.create_mitigation_plan(
                detection=None,  # Invalid input
                organization=None,
                transformation=None
            )
    
    @patch('scrollintel.engines.resistance_mitigation_engine.logging')
    def test_logging(self, mock_logging, engine, sample_detection, sample_organization, sample_transformation):
        """Test logging functionality"""
        engine.create_mitigation_plan(
            detection=sample_detection,
            organization=sample_organization,
            transformation=sample_transformation
        )
        
        # Verify logging was called
        assert mock_logging.getLogger.called
    
    def test_timeline_creation(self, engine):
        """Test mitigation timeline creation"""
        interventions = []  # Empty for simplicity
        constraints = {"timeline_limit": 21}
        
        timeline = engine._create_mitigation_timeline(interventions, constraints)
        
        assert isinstance(timeline, dict)
        assert "start_date" in timeline
        assert "end_date" in timeline
        assert isinstance(timeline["start_date"], datetime)
        assert isinstance(timeline["end_date"], datetime)
        assert timeline["end_date"] > timeline["start_date"]
    
    def test_resource_estimation(self, engine, sample_organization):
        """Test resource requirement estimation"""
        from scrollintel.models.resistance_mitigation_models import MitigationIntervention
        
        interventions = [
            MitigationIntervention(
                id="int_001",
                plan_id="plan_001",
                intervention_type=InterventionType.TRAINING_SESSION,
                strategy=MitigationStrategy.TRAINING_SUPPORT,
                title="Test Training",
                description="Test",
                target_audience=[],
                facilitators=[],
                duration_hours=4.0,
                scheduled_date=datetime.now(),
                completion_date=None,
                status=MitigationStatus.PLANNED,
                success_metrics={},
                actual_results={},
                participant_feedback=[],
                effectiveness_score=None,
                lessons_learned=[],
                follow_up_actions=[]
            )
        ]
        
        resources = engine._estimate_resource_requirements(interventions, sample_organization)
        
        assert isinstance(resources, dict)
        assert "facilitator_hours" in resources
        assert "participant_hours" in resources
        assert "budget_estimate" in resources
        assert isinstance(resources["facilitator_hours"], (int, float))
        assert isinstance(resources["budget_estimate"], (int, float))