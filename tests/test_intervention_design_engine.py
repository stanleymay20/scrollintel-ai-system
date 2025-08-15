"""
Tests for Intervention Design Engine

Tests the strategic intervention design, effectiveness prediction,
and sequence optimization systems.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.intervention_design_engine import InterventionDesignEngine
from scrollintel.models.intervention_design_models import (
    InterventionDesignRequest, InterventionDesign, InterventionTarget,
    InterventionType, InterventionScope, EffectivenessLevel
)


class TestInterventionDesignEngine:
    """Test cases for Intervention Design Engine"""
    
    @pytest.fixture
    def intervention_engine(self):
        """Create intervention design engine instance"""
        return InterventionDesignEngine()
    
    @pytest.fixture
    def sample_cultural_gaps(self):
        """Create sample cultural gaps"""
        return [
            {
                "type": "behavioral",
                "category": "collaboration",
                "current_state": "Siloed working",
                "desired_state": "Cross-functional collaboration",
                "size": 0.7,
                "priority": 8,
                "affected_population": "organization",
                "indicators": ["Team interaction frequency", "Cross-functional project success"]
            },
            {
                "type": "communication",
                "category": "transparency",
                "current_state": "Limited information sharing",
                "desired_state": "Open and transparent communication",
                "size": 0.6,
                "priority": 7,
                "affected_population": "department",
                "indicators": ["Communication satisfaction", "Information accessibility"]
            },
            {
                "type": "process",
                "category": "decision_making",
                "current_state": "Hierarchical decision making",
                "desired_state": "Empowered decision making",
                "size": 0.5,
                "priority": 6,
                "affected_population": "team",
                "indicators": ["Decision speed", "Employee empowerment score"]
            }
        ]
    
    @pytest.fixture
    def sample_design_request(self, sample_cultural_gaps):
        """Create sample intervention design request"""
        return InterventionDesignRequest(
            organization_id="org123",
            roadmap_id="roadmap456",
            cultural_gaps=sample_cultural_gaps,
            target_behaviors=["Collaborative working", "Open communication", "Empowered decisions"],
            available_resources={
                "budget": 100000,
                "trainers": 3,
                "time_weeks": 12,
                "facilities": ["Training rooms", "Online platforms"]
            },
            constraints=["Limited budget", "Remote workforce", "Tight timeline"],
            timeline=timedelta(days=90),
            stakeholder_preferences={
                "leadership": ["Quick wins", "Measurable results"],
                "employees": ["Practical training", "Flexible scheduling"]
            },
            risk_tolerance=0.6,
            effectiveness_threshold=0.7
        )
    
    def test_design_interventions_success(self, intervention_engine, sample_design_request):
        """Test successful intervention design"""
        # Act
        result = intervention_engine.design_interventions(sample_design_request)
        
        # Assert
        assert result is not None
        assert len(result.interventions) > 0
        assert result.sequence is not None
        assert len(result.effectiveness_predictions) == len(result.interventions)
        assert result.resource_requirements is not None
        assert result.implementation_plan is not None
        assert len(result.risk_assessment) > 0
        assert 0 <= result.success_probability <= 1
        
        # Verify interventions are properly designed
        for intervention in result.interventions:
            assert intervention.id is not None
            assert intervention.name != ""
            assert intervention.description != ""
            assert isinstance(intervention.intervention_type, InterventionType)
            assert isinstance(intervention.scope, InterventionScope)
            assert len(intervention.targets) > 0
            assert len(intervention.activities) > 0
            assert intervention.duration > timedelta(0)
    
    def test_identify_intervention_opportunities(self, intervention_engine, sample_design_request):
        """Test intervention opportunity identification"""
        # Act
        opportunities = intervention_engine._identify_intervention_opportunities(sample_design_request)
        
        # Assert
        assert len(opportunities) > 0
        
        for opportunity in opportunities:
            assert "gap" in opportunity
            assert "intervention_type" in opportunity
            assert "priority" in opportunity
            assert "estimated_impact" in opportunity
            assert "scope" in opportunity
            assert isinstance(opportunity["intervention_type"], InterventionType)
            assert isinstance(opportunity["scope"], InterventionScope)
    
    def test_design_individual_interventions(self, intervention_engine, sample_design_request):
        """Test individual intervention design"""
        # Arrange
        opportunities = intervention_engine._identify_intervention_opportunities(sample_design_request)
        
        # Act
        interventions = intervention_engine._design_individual_interventions(
            opportunities, sample_design_request
        )
        
        # Assert
        assert len(interventions) > 0
        
        for intervention in interventions:
            assert isinstance(intervention, InterventionDesign)
            assert intervention.id is not None
            assert len(intervention.targets) > 0
            assert len(intervention.activities) > 0
            assert intervention.duration > timedelta(0)
            assert len(intervention.success_criteria) > 0
            assert len(intervention.measurement_methods) > 0
    
    def test_predict_intervention_effectiveness(self, intervention_engine):
        """Test intervention effectiveness prediction"""
        # Arrange
        intervention = InterventionDesign(
            id="test_intervention",
            name="Test Training Intervention",
            description="Test training for collaboration",
            intervention_type=InterventionType.TRAINING,
            scope=InterventionScope.ORGANIZATION,
            targets=[
                InterventionTarget(
                    target_type="behavioral",
                    current_state="Siloed working",
                    desired_state="Collaborative working",
                    gap_size=0.7,
                    priority=8,
                    measurable_indicators=["Team interaction", "Project success"]
                )
            ],
            objectives=["Improve collaboration"],
            activities=["Team building", "Collaboration training"],
            resources_required={"trainers": 2, "budget": 10000},
            duration=timedelta(days=30),
            participants=["All employees"],
            facilitators=["HR team"],
            materials=["Training materials"],
            success_criteria=["Increased collaboration score"],
            measurement_methods=["Survey", "Observation"]
        )
        
        context = {
            "organization_id": "org123",
            "available_resources": {"budget": 50000, "trainers": 3},
            "constraints": ["Remote workforce"],
            "risk_tolerance": 0.6
        }
        
        # Act
        prediction = intervention_engine.predict_intervention_effectiveness(intervention, context)
        
        # Assert
        assert prediction is not None
        assert prediction.intervention_id == "test_intervention"
        assert isinstance(prediction.predicted_effectiveness, EffectivenessLevel)
        assert 0 <= prediction.confidence_score <= 1
        assert 0 <= prediction.success_probability <= 1
        assert len(prediction.contributing_factors) > 0
        assert len(prediction.risk_factors) > 0
        assert len(prediction.expected_outcomes) > 0
        assert len(prediction.measurement_timeline) > 0
    
    def test_optimize_intervention_sequence(self, intervention_engine):
        """Test intervention sequence optimization"""
        # Arrange
        interventions = [
            InterventionDesign(
                id="comm_intervention",
                name="Communication Intervention",
                description="Improve communication",
                intervention_type=InterventionType.COMMUNICATION,
                scope=InterventionScope.ORGANIZATION,
                targets=[],
                objectives=["Improve communication"],
                activities=["Communication training"],
                resources_required={},
                duration=timedelta(days=14),
                participants=[],
                facilitators=[],
                materials=[],
                success_criteria=[],
                measurement_methods=[]
            ),
            InterventionDesign(
                id="training_intervention",
                name="Training Intervention",
                description="Skills training",
                intervention_type=InterventionType.TRAINING,
                scope=InterventionScope.ORGANIZATION,
                targets=[],
                objectives=["Improve skills"],
                activities=["Skills training"],
                resources_required={},
                duration=timedelta(days=30),
                participants=[],
                facilitators=[],
                materials=[],
                success_criteria=[],
                measurement_methods=[]
            )
        ]
        
        constraints = {
            "timeline": timedelta(days=90),
            "resources": {"budget": 50000},
            "risk_tolerance": 0.6
        }
        
        # Act
        sequence = intervention_engine.optimize_intervention_sequence(interventions, constraints)
        
        # Assert
        assert sequence is not None
        assert sequence.id is not None
        assert len(sequence.interventions) == 2
        assert sequence.total_duration > timedelta(0)
        assert len(sequence.coordination_requirements) > 0
        assert sequence.sequencing_rationale != ""
    
    def test_effectiveness_level_determination(self, intervention_engine):
        """Test effectiveness level determination"""
        # Test different effectiveness scores
        assert intervention_engine._determine_effectiveness_level(0.9) == EffectivenessLevel.VERY_HIGH
        assert intervention_engine._determine_effectiveness_level(0.75) == EffectivenessLevel.HIGH
        assert intervention_engine._determine_effectiveness_level(0.6) == EffectivenessLevel.MEDIUM
        assert intervention_engine._determine_effectiveness_level(0.3) == EffectivenessLevel.LOW
    
    def test_intervention_type_suitability(self, intervention_engine):
        """Test determination of suitable intervention types"""
        # Test behavioral gap
        behavioral_gap = {
            "type": "behavioral",
            "category": "collaboration",
            "size": 0.7
        }
        
        suitable_types = intervention_engine._determine_suitable_intervention_types(behavioral_gap)
        assert InterventionType.TRAINING in suitable_types
        assert InterventionType.BEHAVIORAL_NUDGE in suitable_types
        
        # Test communication gap
        communication_gap = {
            "type": "communication",
            "category": "transparency",
            "size": 0.6
        }
        
        suitable_types = intervention_engine._determine_suitable_intervention_types(communication_gap)
        assert InterventionType.COMMUNICATION in suitable_types
        assert InterventionType.TRAINING in suitable_types
    
    def test_opportunity_priority_calculation(self, intervention_engine):
        """Test opportunity priority calculation"""
        gap = {
            "priority": 8,
            "size": 0.7,
            "type": "behavioral"
        }
        
        # Test different intervention types
        training_priority = intervention_engine._calculate_opportunity_priority(
            gap, InterventionType.TRAINING
        )
        communication_priority = intervention_engine._calculate_opportunity_priority(
            gap, InterventionType.COMMUNICATION
        )
        
        assert training_priority > 0
        assert communication_priority > 0
        # Training should generally have higher priority for behavioral gaps
        assert training_priority >= communication_priority
    
    def test_intervention_scope_determination(self, intervention_engine):
        """Test intervention scope determination"""
        assert intervention_engine._determine_intervention_scope("individual") == InterventionScope.INDIVIDUAL
        assert intervention_engine._determine_intervention_scope("team") == InterventionScope.TEAM
        assert intervention_engine._determine_intervention_scope("department") == InterventionScope.DEPARTMENT
        assert intervention_engine._determine_intervention_scope("organization") == InterventionScope.ORGANIZATION
        assert intervention_engine._determine_intervention_scope("unknown") == InterventionScope.ORGANIZATION
    
    def test_resource_requirements_assessment(self, intervention_engine, sample_design_request):
        """Test resource requirements assessment"""
        # Arrange
        result = intervention_engine.design_interventions(sample_design_request)
        
        # Act
        resource_requirements = intervention_engine._assess_intervention_resources(
            result.interventions, result.sequence
        )
        
        # Assert
        assert resource_requirements is not None
        assert "human_resources" in resource_requirements
        assert "financial_resources" in resource_requirements
        assert "time_resources" in resource_requirements
        assert "coordination_overhead" in resource_requirements
    
    def test_success_probability_calculation(self, intervention_engine, sample_design_request):
        """Test overall success probability calculation"""
        # Act
        result = intervention_engine.design_interventions(sample_design_request)
        
        # Assert
        assert 0 <= result.success_probability <= 1
        
        # Test with different risk tolerances
        sample_design_request.risk_tolerance = 0.2  # Risk-averse
        low_risk_result = intervention_engine.design_interventions(sample_design_request)
        
        sample_design_request.risk_tolerance = 0.8  # Risk-tolerant
        high_risk_result = intervention_engine.design_interventions(sample_design_request)
        
        # Success probability should vary with risk tolerance
        assert low_risk_result.success_probability != high_risk_result.success_probability
    
    def test_intervention_optimization(self, intervention_engine):
        """Test intervention optimization"""
        # Arrange
        intervention = InterventionDesign(
            id="test_intervention",
            name="Test Intervention",
            description="Test intervention",
            intervention_type=InterventionType.TRAINING,
            scope=InterventionScope.ORGANIZATION,
            targets=[],
            objectives=["Test objective"],
            activities=["Test activity"],
            resources_required={"budget": 10000},
            duration=timedelta(days=30),
            participants=[],
            facilitators=[],
            materials=[],
            success_criteria=[],
            measurement_methods=[]
        )
        
        prediction = Mock()
        prediction.predicted_effectiveness = EffectivenessLevel.LOW
        prediction.risk_factors = ["resource constraints", "timeline pressure"]
        
        request = Mock()
        request.risk_tolerance = 0.6
        
        # Act
        optimized = intervention_engine._optimize_single_intervention(
            intervention, prediction, request
        )
        
        # Assert
        assert optimized is not None
        assert optimized.id == intervention.id
        # Should have additional resources due to resource risk factor
        assert "additional_support" in optimized.resources_required
        # Should have extended duration due to timeline risk factor
        assert optimized.duration > intervention.duration
    
    def test_template_loading(self, intervention_engine):
        """Test template loading"""
        # Test intervention templates
        templates = intervention_engine._load_intervention_templates()
        assert isinstance(templates, dict)
        assert "training" in templates
        assert "communication" in templates
        
        # Verify template structure
        training_template = templates["training"]
        assert "name" in training_template
        assert "template_activities" in training_template
        assert "typical_duration" in training_template
        assert "resource_requirements" in training_template
        
        # Test effectiveness models
        models = intervention_engine._load_effectiveness_models()
        assert isinstance(models, dict)
        assert "base_effectiveness" in models
        
        # Test sequencing rules
        rules = intervention_engine._load_sequencing_rules()
        assert isinstance(rules, dict)
        assert "prerequisites" in rules
    
    def test_error_handling(self, intervention_engine):
        """Test error handling in intervention design"""
        # Test with invalid request
        invalid_request = InterventionDesignRequest(
            organization_id="",  # Invalid empty org ID
            roadmap_id="",
            cultural_gaps=[],  # No gaps
            target_behaviors=[],
            available_resources={},
            constraints=[],
            timeline=timedelta(days=1),  # Very short timeline
            stakeholder_preferences={}
        )
        
        # Should handle gracefully and still return a result
        result = intervention_engine.design_interventions(invalid_request)
        assert result is not None
        # May have no interventions due to no gaps
        assert isinstance(result.interventions, list)
    
    def test_intervention_coordination(self, intervention_engine):
        """Test intervention coordination requirements"""
        # Arrange
        interventions = [
            InterventionDesign(
                id="intervention1",
                name="First Intervention",
                description="First intervention",
                intervention_type=InterventionType.COMMUNICATION,
                scope=InterventionScope.ORGANIZATION,
                targets=[], objectives=[], activities=[], resources_required={},
                duration=timedelta(days=14), participants=[], facilitators=[],
                materials=[], success_criteria=[], measurement_methods=[]
            ),
            InterventionDesign(
                id="intervention2",
                name="Second Intervention",
                description="Second intervention",
                intervention_type=InterventionType.TRAINING,
                scope=InterventionScope.ORGANIZATION,
                targets=[], objectives=[], activities=[], resources_required={},
                duration=timedelta(days=30), participants=[], facilitators=[],
                materials=[], success_criteria=[], measurement_methods=[]
            )
        ]
        
        # Act
        dependencies = intervention_engine._analyze_intervention_dependencies(interventions)
        parallel_groups = intervention_engine._identify_parallel_groups(interventions, dependencies)
        
        # Assert
        assert isinstance(dependencies, dict)
        assert isinstance(parallel_groups, list)
        
        # Should identify that training can depend on communication
        # (based on sequencing rules in the engine)
        if dependencies:
            assert all(isinstance(deps, list) for deps in dependencies.values())


@pytest.mark.integration
class TestInterventionDesignIntegration:
    """Integration tests for intervention design system"""
    
    def test_end_to_end_intervention_design(self):
        """Test complete intervention design workflow"""
        # Arrange
        engine = InterventionDesignEngine()
        
        request = InterventionDesignRequest(
            organization_id="test_org",
            roadmap_id="test_roadmap",
            cultural_gaps=[
                {
                    "type": "behavioral",
                    "category": "innovation",
                    "current_state": "Risk-averse culture",
                    "desired_state": "Innovation-driven culture",
                    "size": 0.8,
                    "priority": 9,
                    "affected_population": "organization",
                    "indicators": ["Innovation index", "Risk-taking behavior"]
                },
                {
                    "type": "communication",
                    "category": "feedback",
                    "current_state": "Limited feedback culture",
                    "desired_state": "Continuous feedback culture",
                    "size": 0.6,
                    "priority": 7,
                    "affected_population": "organization",
                    "indicators": ["Feedback frequency", "Feedback quality"]
                }
            ],
            target_behaviors=["Risk-taking", "Innovation", "Feedback giving"],
            available_resources={
                "budget": 150000,
                "trainers": 4,
                "time_weeks": 16,
                "facilities": ["Training centers", "Online platforms"]
            },
            constraints=["Global workforce", "Multiple time zones", "Budget approval required"],
            timeline=timedelta(days=120),
            stakeholder_preferences={
                "leadership": ["Measurable ROI", "Quick wins", "Scalable solutions"],
                "employees": ["Practical skills", "Career development", "Flexible delivery"],
                "managers": ["Management tools", "Team development", "Performance improvement"]
            },
            risk_tolerance=0.7,
            effectiveness_threshold=0.75
        )
        
        # Act
        result = engine.design_interventions(request)
        
        # Assert - Complete workflow validation
        assert result.organization_id == "test_org"
        assert len(result.interventions) >= 2  # Should have interventions for both gaps
        assert result.sequence is not None
        assert len(result.effectiveness_predictions) == len(result.interventions)
        assert result.success_probability > 0.5  # Should be reasonably optimistic
        
        # Validate intervention quality
        for intervention in result.interventions:
            assert intervention.name != ""
            assert intervention.description != ""
            assert len(intervention.targets) > 0
            assert len(intervention.activities) > 0
            assert intervention.duration > timedelta(0)
            assert len(intervention.success_criteria) > 0
        
        # Validate effectiveness predictions
        for prediction in result.effectiveness_predictions:
            assert prediction.intervention_id in [i.id for i in result.interventions]
            assert isinstance(prediction.predicted_effectiveness, EffectivenessLevel)
            assert 0 <= prediction.confidence_score <= 1
            assert 0 <= prediction.success_probability <= 1
        
        # Validate sequence
        assert len(result.sequence.interventions) == len(result.interventions)
        assert result.sequence.total_duration > timedelta(0)
        assert len(result.sequence.coordination_requirements) > 0
        
        # Validate resource requirements
        assert "human_resources" in result.resource_requirements
        assert "financial_resources" in result.resource_requirements
        
        # Validate implementation plan
        assert "phases" in result.implementation_plan
        assert "timeline" in result.implementation_plan
        assert "coordination_plan" in result.implementation_plan
    
    def test_intervention_effectiveness_optimization_cycle(self):
        """Test iterative intervention optimization"""
        # Arrange
        engine = InterventionDesignEngine()
        
        # Initial intervention with low effectiveness
        intervention = InterventionDesign(
            id="optimization_test",
            name="Low Effectiveness Intervention",
            description="Intervention needing optimization",
            intervention_type=InterventionType.TRAINING,
            scope=InterventionScope.ORGANIZATION,
            targets=[
                InterventionTarget(
                    target_type="behavioral",
                    current_state="Low engagement",
                    desired_state="High engagement",
                    gap_size=0.8,
                    priority=8,
                    measurable_indicators=["Engagement score"]
                )
            ],
            objectives=["Improve engagement"],
            activities=["Basic training session"],
            resources_required={"budget": 5000, "trainers": 1},
            duration=timedelta(days=7),
            participants=["All employees"],
            facilitators=["Internal trainer"],
            materials=["Basic materials"],
            success_criteria=["Improved engagement"],
            measurement_methods=["Survey"]
        )
        
        context = {
            "organization_id": "test_org",
            "available_resources": {"budget": 50000, "trainers": 5},
            "constraints": ["Remote workforce", "Limited time"],
            "risk_tolerance": 0.5
        }
        
        # Act - Initial prediction
        initial_prediction = engine.predict_intervention_effectiveness(intervention, context)
        
        # Optimize if effectiveness is low
        if initial_prediction.predicted_effectiveness in [EffectivenessLevel.LOW, EffectivenessLevel.MEDIUM]:
            request = Mock()
            request.risk_tolerance = 0.5
            
            optimized_intervention = engine._optimize_single_intervention(
                intervention, initial_prediction, request
            )
            
            # Re-predict effectiveness
            optimized_prediction = engine.predict_intervention_effectiveness(
                optimized_intervention, context
            )
            
            # Assert - Optimization should improve or maintain effectiveness
            assert optimized_intervention.id == intervention.id
            # Should have some optimization applied
            assert (optimized_intervention.duration >= intervention.duration or
                   len(optimized_intervention.activities) >= len(intervention.activities) or
                   len(optimized_intervention.resources_required) >= len(intervention.resources_required))
    
    def test_complex_intervention_sequencing(self):
        """Test sequencing of complex intervention portfolio"""
        # Arrange
        engine = InterventionDesignEngine()
        
        # Create diverse set of interventions
        interventions = [
            InterventionDesign(
                id="comm_foundation",
                name="Communication Foundation",
                description="Establish communication baseline",
                intervention_type=InterventionType.COMMUNICATION,
                scope=InterventionScope.ORGANIZATION,
                targets=[], objectives=[], activities=[], resources_required={},
                duration=timedelta(days=14), participants=[], facilitators=[],
                materials=[], success_criteria=[], measurement_methods=[]
            ),
            InterventionDesign(
                id="leadership_modeling",
                name="Leadership Modeling",
                description="Leadership behavior demonstration",
                intervention_type=InterventionType.LEADERSHIP_MODELING,
                scope=InterventionScope.ORGANIZATION,
                targets=[], objectives=[], activities=[], resources_required={},
                duration=timedelta(days=60), participants=[], facilitators=[],
                materials=[], success_criteria=[], measurement_methods=[]
            ),
            InterventionDesign(
                id="skills_training",
                name="Skills Training",
                description="Core skills development",
                intervention_type=InterventionType.TRAINING,
                scope=InterventionScope.ORGANIZATION,
                targets=[], objectives=[], activities=[], resources_required={},
                duration=timedelta(days=30), participants=[], facilitators=[],
                materials=[], success_criteria=[], measurement_methods=[]
            ),
            InterventionDesign(
                id="process_change",
                name="Process Optimization",
                description="Process improvements",
                intervention_type=InterventionType.PROCESS_CHANGE,
                scope=InterventionScope.DEPARTMENT,
                targets=[], objectives=[], activities=[], resources_required={},
                duration=timedelta(days=45), participants=[], facilitators=[],
                materials=[], success_criteria=[], measurement_methods=[]
            ),
            InterventionDesign(
                id="behavioral_nudges",
                name="Behavioral Nudges",
                description="Environmental behavior cues",
                intervention_type=InterventionType.BEHAVIORAL_NUDGE,
                scope=InterventionScope.ORGANIZATION,
                targets=[], objectives=[], activities=[], resources_required={},
                duration=timedelta(days=21), participants=[], facilitators=[],
                materials=[], success_criteria=[], measurement_methods=[]
            )
        ]
        
        constraints = {
            "timeline": timedelta(days=150),
            "resources": {"budget": 200000, "trainers": 6},
            "risk_tolerance": 0.6
        }
        
        # Act
        sequence = engine.optimize_intervention_sequence(interventions, constraints)
        
        # Assert - Complex sequencing validation
        assert sequence is not None
        assert len(sequence.interventions) == 5
        assert sequence.total_duration <= timedelta(days=150)  # Should fit within constraints
        
        # Validate logical sequencing (communication should come early)
        comm_position = sequence.interventions.index("comm_foundation")
        training_position = sequence.interventions.index("skills_training")
        # Communication should generally precede training
        assert comm_position <= training_position or len(sequence.parallel_groups) > 0
        
        # Should have coordination requirements for complex sequence
        assert len(sequence.coordination_requirements) > 0
        assert sequence.sequencing_rationale != ""
        
        # Should identify some parallel execution opportunities
        total_sequential_duration = sum(
            next(i.duration for i in interventions if i.id == intervention_id)
            for intervention_id in sequence.interventions
        )
        # If parallel groups exist, total duration should be less than sequential
        if len(sequence.parallel_groups) > 1:
            assert sequence.total_duration < total_sequential_duration