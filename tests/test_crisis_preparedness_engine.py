"""
Tests for Crisis Preparedness Engine

Test crisis preparedness assessment and enhancement functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.crisis_preparedness_engine import CrisisPreparednessEngine
from scrollintel.models.crisis_preparedness_models import (
    PreparednessAssessment, PreparednessLevel, CrisisSimulation, SimulationType,
    TrainingProgram, TrainingType, CapabilityDevelopment, CapabilityArea,
    PreparednessReport
)


class TestCrisisPreparednessEngine:
    """Test cases for CrisisPreparednessEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        return CrisisPreparednessEngine()
    
    @pytest.fixture
    def sample_organization_data(self):
        """Create sample organization data for testing"""
        return {
            "industry": "technology",
            "organizational_maturity": 0.8,
            "crisis_detection_score": 75.0,
            "decision_making_score": 80.0,
            "communication_score": 85.0,
            "resource_mobilization_score": 70.0,
            "team_coordination_score": 78.0,
            "stakeholder_management_score": 82.0,
            "recovery_planning_score": 73.0,
            "employee_count": 500,
            "annual_revenue": 50000000,
            "previous_crises": 2,
            "training_budget": 100000
        }
    
    def test_assess_crisis_preparedness(self, engine, sample_organization_data):
        """Test comprehensive crisis preparedness assessment"""
        assessor_id = "assessor_001"
        
        assessment = engine.assess_crisis_preparedness(
            organization_data=sample_organization_data,
            assessor_id=assessor_id
        )
        
        assert isinstance(assessment, PreparednessAssessment)
        assert assessment.assessor_id == assessor_id
        assert assessment.overall_preparedness_level in PreparednessLevel
        assert 0 <= assessment.overall_score <= 100
        assert len(assessment.capability_scores) == len(CapabilityArea)
        assert len(assessment.capability_levels) == len(CapabilityArea)
        assert assessment.confidence_level > 0
        assert len(assessment.high_risk_scenarios) > 0
    
    def test_create_crisis_simulation(self, engine):
        """Test crisis simulation creation"""
        simulation_type = SimulationType.SYSTEM_OUTAGE
        participants = ["user1", "user2", "user3"]
        facilitator_id = "facilitator_001"
        
        simulation = engine.create_crisis_simulation(
            simulation_type=simulation_type,
            participants=participants,
            facilitator_id=facilitator_id
        )
        
        assert isinstance(simulation, CrisisSimulation)
        assert simulation.simulation_type == simulation_type
        assert simulation.participants == participants
        assert simulation.facilitator_id == facilitator_id
        assert simulation.simulation_status == "scheduled"
        assert len(simulation.title) > 0
        assert len(simulation.description) > 0
        assert simulation.duration_minutes > 0
    
    def test_execute_simulation(self, engine):
        """Test simulation execution"""
        # Create a simulation first
        simulation = CrisisSimulation(
            id="sim_001",
            simulation_type=SimulationType.SECURITY_BREACH,
            title="Security Breach Simulation",
            description="Test security incident response",
            scenario_details="Simulated data breach scenario",
            complexity_level="High",
            duration_minutes=180,
            participants=["user1", "user2"],
            learning_objectives=["Incident response", "Communication"],
            success_criteria=["Response time < 30 min", "All stakeholders notified"],
            facilitator_id="facilitator_001",
            scheduled_date=datetime.now(),
            actual_start_time=None,
            actual_end_time=None,
            participant_performance={},
            objectives_achieved=[],
            lessons_learned=[],
            improvement_areas=[],
            simulation_status="scheduled",
            feedback_collected=False,
            report_generated=False
        )
        
        results = engine.execute_simulation(simulation)
        
        assert isinstance(results, dict)
        assert "participant_performance" in results
        assert "objectives_achieved" in results
        assert "lessons_learned" in results
        assert "improvement_areas" in results
        assert simulation.simulation_status == "completed"
        assert simulation.actual_start_time is not None
        assert simulation.actual_end_time is not None
        assert len(simulation.participant_performance) == len(simulation.participants)
    
    def test_develop_training_program(self, engine):
        """Test training program development"""
        capability_area = CapabilityArea.CRISIS_DETECTION
        target_audience = ["managers", "team_leads"]
        training_type = TrainingType.TABLETOP_EXERCISE
        
        program = engine.develop_training_program(
            capability_area=capability_area,
            target_audience=target_audience,
            training_type=training_type
        )
        
        assert isinstance(program, TrainingProgram)
        assert program.training_type == training_type
        assert program.target_audience == target_audience
        assert len(program.program_name) > 0
        assert len(program.description) > 0
        assert program.duration_hours > 0
        assert program.approval_status == "draft"
        assert program.version == "1.0"
    
    def test_create_capability_development_plan(self, engine, sample_organization_data):
        """Test capability development plan creation"""
        # First create an assessment
        assessment = engine.assess_crisis_preparedness(
            organization_data=sample_organization_data,
            assessor_id="assessor_001"
        )
        
        capability_area = CapabilityArea.COMMUNICATION
        target_level = PreparednessLevel.EXCELLENT
        
        plan = engine.create_capability_development_plan(
            capability_area=capability_area,
            current_assessment=assessment,
            target_level=target_level
        )
        
        assert isinstance(plan, CapabilityDevelopment)
        assert plan.capability_area == capability_area
        assert plan.target_level == target_level
        assert plan.current_level == assessment.capability_levels[capability_area]
        assert len(plan.development_objectives) > 0
        assert len(plan.improvement_actions) > 0
        assert plan.current_progress == 0.0
        assert plan.status == "planned"
        assert plan.budget_allocated > 0
    
    def test_generate_preparedness_report(self, engine, sample_organization_data):
        """Test preparedness report generation"""
        # Create assessment
        assessment = engine.assess_crisis_preparedness(
            organization_data=sample_organization_data,
            assessor_id="assessor_001"
        )
        
        # Create simulation
        simulation = engine.create_crisis_simulation(
            simulation_type=SimulationType.SYSTEM_OUTAGE,
            participants=["user1", "user2"],
            facilitator_id="facilitator_001"
        )
        
        # Create training program
        training = engine.develop_training_program(
            capability_area=CapabilityArea.CRISIS_DETECTION,
            target_audience=["managers"],
            training_type=TrainingType.SIMULATION_DRILL
        )
        
        report = engine.generate_preparedness_report(
            assessment=assessment,
            simulations=[simulation],
            training_programs=[training]
        )
        
        assert isinstance(report, PreparednessReport)
        assert len(report.report_title) > 0
        assert len(report.executive_summary) > 0
        assert len(report.current_state_assessment) > 0
        assert len(report.gap_analysis) > 0
        assert len(report.improvement_recommendations) > 0
        assert len(report.implementation_roadmap) > 0
        assert report.report_type == "comprehensive"
        assert report.review_status == "draft"
    
    def test_assess_capabilities(self, engine, sample_organization_data):
        """Test capability assessment"""
        scores = engine._assess_capabilities(sample_organization_data)
        
        assert isinstance(scores, dict)
        assert len(scores) == len(CapabilityArea)
        
        for capability, score in scores.items():
            assert isinstance(capability, CapabilityArea)
            assert isinstance(score, float)
            assert 0 <= score <= 100
    
    def test_determine_capability_levels(self, engine):
        """Test capability level determination"""
        scores = {
            CapabilityArea.CRISIS_DETECTION: 95.0,
            CapabilityArea.DECISION_MAKING: 85.0,
            CapabilityArea.COMMUNICATION: 75.0,
            CapabilityArea.RESOURCE_MOBILIZATION: 65.0,
            CapabilityArea.TEAM_COORDINATION: 55.0
        }
        
        levels = engine._determine_capability_levels(scores)
        
        assert levels[CapabilityArea.CRISIS_DETECTION] == PreparednessLevel.EXCELLENT
        assert levels[CapabilityArea.DECISION_MAKING] == PreparednessLevel.GOOD
        assert levels[CapabilityArea.COMMUNICATION] == PreparednessLevel.ADEQUATE
        assert levels[CapabilityArea.RESOURCE_MOBILIZATION] == PreparednessLevel.POOR
        assert levels[CapabilityArea.TEAM_COORDINATION] == PreparednessLevel.CRITICAL
    
    def test_determine_overall_preparedness_level(self, engine):
        """Test overall preparedness level determination"""
        assert engine._determine_overall_preparedness_level(95.0) == PreparednessLevel.EXCELLENT
        assert engine._determine_overall_preparedness_level(85.0) == PreparednessLevel.GOOD
        assert engine._determine_overall_preparedness_level(75.0) == PreparednessLevel.ADEQUATE
        assert engine._determine_overall_preparedness_level(65.0) == PreparednessLevel.POOR
        assert engine._determine_overall_preparedness_level(55.0) == PreparednessLevel.CRITICAL
    
    def test_identify_preparedness_strengths(self, engine, sample_organization_data):
        """Test preparedness strengths identification"""
        scores = {
            CapabilityArea.CRISIS_DETECTION: 90.0,
            CapabilityArea.COMMUNICATION: 88.0,
            CapabilityArea.DECISION_MAKING: 75.0
        }
        
        strengths = engine._identify_preparedness_strengths(scores, sample_organization_data)
        
        assert isinstance(strengths, list)
        assert any("crisis detection" in strength.lower() for strength in strengths)
        assert any("communication" in strength.lower() for strength in strengths)
    
    def test_identify_preparedness_weaknesses(self, engine, sample_organization_data):
        """Test preparedness weaknesses identification"""
        scores = {
            CapabilityArea.RESOURCE_MOBILIZATION: 65.0,
            CapabilityArea.TEAM_COORDINATION: 60.0,
            CapabilityArea.COMMUNICATION: 85.0
        }
        
        weaknesses = engine._identify_preparedness_weaknesses(scores, sample_organization_data)
        
        assert isinstance(weaknesses, list)
        assert any("resource mobilization" in weakness.lower() for weakness in weaknesses)
        assert any("team coordination" in weakness.lower() for weakness in weaknesses)
    
    def test_identify_high_risk_scenarios(self, engine, sample_organization_data):
        """Test high-risk scenarios identification"""
        scenarios = engine._identify_high_risk_scenarios(sample_organization_data)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        # Technology industry should have system outage as high risk
        assert any("system outage" in scenario.lower() for scenario in scenarios)
    
    def test_identify_vulnerability_areas(self, engine):
        """Test vulnerability areas identification"""
        scores = {
            CapabilityArea.CRISIS_DETECTION: 90.0,
            CapabilityArea.COMMUNICATION: 85.0,
            CapabilityArea.DECISION_MAKING: 75.0,
            CapabilityArea.RESOURCE_MOBILIZATION: 65.0,
            CapabilityArea.TEAM_COORDINATION: 60.0
        }
        
        vulnerabilities = engine._identify_vulnerability_areas(scores)
        
        assert isinstance(vulnerabilities, list)
        assert len(vulnerabilities) <= 3  # Top 3 vulnerabilities
        # Should include lowest scoring areas
        assert "Team Coordination" in vulnerabilities
        assert "Resource Mobilization" in vulnerabilities
    
    def test_simulate_execution(self, engine):
        """Test simulation execution logic"""
        simulation = CrisisSimulation(
            id="sim_001",
            simulation_type=SimulationType.SYSTEM_OUTAGE,
            title="Test Simulation",
            description="Test description",
            scenario_details="Test scenario",
            complexity_level="Medium",
            duration_minutes=120,
            participants=["user1", "user2", "user3"],
            learning_objectives=["Objective 1", "Objective 2"],
            success_criteria=["Criteria 1", "Criteria 2"],
            facilitator_id="facilitator_001",
            scheduled_date=datetime.now(),
            actual_start_time=None,
            actual_end_time=None,
            participant_performance={},
            objectives_achieved=[],
            lessons_learned=[],
            improvement_areas=[],
            simulation_status="scheduled",
            feedback_collected=False,
            report_generated=False
        )
        
        results = engine._simulate_execution(simulation)
        
        assert isinstance(results, dict)
        assert "participant_performance" in results
        assert "objectives_achieved" in results
        assert "lessons_learned" in results
        assert "improvement_areas" in results
        
        # Check participant performance
        performance = results["participant_performance"]
        assert len(performance) == len(simulation.participants)
        for participant, score in performance.items():
            assert participant in simulation.participants
            assert 0 <= score <= 100
    
    def test_calculate_assessment_confidence(self, engine, sample_organization_data):
        """Test assessment confidence calculation"""
        confidence = engine._calculate_assessment_confidence(sample_organization_data)
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100
    
    def test_prioritize_improvements(self, engine):
        """Test improvement prioritization"""
        scores = {
            CapabilityArea.CRISIS_DETECTION: 90.0,
            CapabilityArea.COMMUNICATION: 85.0,
            CapabilityArea.DECISION_MAKING: 70.0,
            CapabilityArea.RESOURCE_MOBILIZATION: 65.0,
            CapabilityArea.TEAM_COORDINATION: 60.0
        }
        gaps = ["Gap 1", "Gap 2"]
        
        priorities = engine._prioritize_improvements(scores, gaps)
        
        assert isinstance(priorities, list)
        assert len(priorities) <= 5  # Top 5 priorities
        # Should prioritize lowest scoring capabilities
        assert any("team coordination" in priority.lower() for priority in priorities)
        assert any("resource mobilization" in priority.lower() for priority in priorities)
    
    def test_initialization_methods(self, engine):
        """Test engine initialization methods"""
        # Test assessment frameworks initialization
        assert hasattr(engine, 'assessment_frameworks')
        assert isinstance(engine.assessment_frameworks, dict)
        assert len(engine.assessment_frameworks) > 0
        
        # Test simulation templates initialization
        assert hasattr(engine, 'simulation_templates')
        assert isinstance(engine.simulation_templates, dict)
        assert len(engine.simulation_templates) > 0
        
        # Test training catalog initialization
        assert hasattr(engine, 'training_catalog')
        assert isinstance(engine.training_catalog, dict)
        assert len(engine.training_catalog) > 0
    
    def test_custom_simulation_parameters(self, engine):
        """Test simulation creation with custom parameters"""
        custom_params = {
            "title": "Custom Simulation Title",
            "duration_minutes": 240,
            "complexity_level": "High"
        }
        
        simulation = engine.create_crisis_simulation(
            simulation_type=SimulationType.DATA_LOSS,
            participants=["user1", "user2"],
            facilitator_id="facilitator_001",
            custom_parameters=custom_params
        )
        
        assert simulation.title == custom_params["title"]
        assert simulation.duration_minutes == custom_params["duration_minutes"]
        assert simulation.complexity_level == custom_params["complexity_level"]
    
    def test_empty_organization_data(self, engine):
        """Test assessment with minimal organization data"""
        minimal_data = {"industry": "technology"}
        
        assessment = engine.assess_crisis_preparedness(
            organization_data=minimal_data,
            assessor_id="assessor_001"
        )
        
        assert isinstance(assessment, PreparednessAssessment)
        assert assessment.overall_score >= 0
        assert assessment.confidence_level >= 0
    
    @patch('scrollintel.engines.crisis_preparedness_engine.logging')
    def test_error_handling(self, mock_logging, engine):
        """Test error handling in preparedness engine"""
        # Test with invalid data
        with pytest.raises(Exception):
            engine.assess_crisis_preparedness(
                organization_data=None,
                assessor_id="assessor_001"
            )
    
    def test_capability_development_milestones(self, engine):
        """Test milestone creation in capability development"""
        actions = ["Action 1", "Action 2", "Action 3"]
        milestones = engine._create_development_milestones(actions)
        
        assert isinstance(milestones, list)
        assert len(milestones) == len(actions)
        
        for milestone in milestones:
            assert "milestone" in milestone
            assert "target_date" in milestone
            assert "status" in milestone
            assert milestone["status"] == "pending"
    
    def test_budget_estimation(self, engine):
        """Test budget estimation for capability development"""
        actions = ["Action 1", "Action 2"]
        resources = ["Resource 1", "Resource 2", "Resource 3"]
        
        budget = engine._estimate_budget(actions, resources)
        
        assert isinstance(budget, float)
        assert budget > 0
        # Should be based on number of actions and resources
        expected_budget = len(actions) * 5000 + len(resources) * 2000
        assert budget == expected_budget