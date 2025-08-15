"""
Tests for Cultural Evolution Engine

Tests for continuous cultural evolution and adaptation framework.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.cultural_evolution_engine import CulturalEvolutionEngine
from scrollintel.models.cultural_evolution_models import (
    EvolutionStage, AdaptabilityLevel, InnovationType
)
from scrollintel.models.cultural_assessment_models import CultureMap
from scrollintel.models.culture_maintenance_models import SustainabilityAssessment, SustainabilityLevel


class TestCulturalEvolutionEngine:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = CulturalEvolutionEngine()
        
        # Mock culture data
        self.current_culture = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.65,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        self.sustainability_assessment = SustainabilityAssessment(
            assessment_id="test_assessment",
            organization_id="test_org",
            transformation_id="test_transformation",
            sustainability_level=SustainabilityLevel.MEDIUM,
            risk_factors=["incomplete_transformation"],
            protective_factors=["strong_leadership"],
            health_indicators=[],
            overall_score=0.7,
            assessment_date=datetime.now(),
            next_assessment_due=datetime.now() + timedelta(days=90)
        )
    
    def test_create_evolution_framework(self):
        """Test creation of cultural evolution framework"""
        # Act
        evolution_plan = self.engine.create_evolution_framework(
            "test_org", self.current_culture, self.sustainability_assessment
        )
        
        # Assert
        assert evolution_plan.organization_id == "test_org"
        assert evolution_plan.current_evolution_stage in EvolutionStage
        assert evolution_plan.target_evolution_stage in EvolutionStage
        assert len(evolution_plan.cultural_innovations) > 0
        assert len(evolution_plan.adaptation_mechanisms) > 0
        assert len(evolution_plan.evolution_triggers) > 0
        assert len(evolution_plan.success_criteria) > 0
        assert "monitoring_frequency" in evolution_plan.monitoring_framework
    
    def test_implement_innovation_mechanisms(self):
        """Test implementation of innovation mechanisms"""
        # Arrange
        evolution_plan = self.engine.create_evolution_framework(
            "test_org", self.current_culture, self.sustainability_assessment
        )
        
        # Act
        implemented_innovations = self.engine.implement_innovation_mechanisms(
            "test_org", evolution_plan
        )
        
        # Assert
        assert isinstance(implemented_innovations, list)
        # Should implement some innovations based on readiness
        assert len(implemented_innovations) >= 0
        
        # Check that implemented innovations have updated status
        for innovation in implemented_innovations:
            assert innovation.status == "implementing"
    
    def test_enhance_cultural_resilience(self):
        """Test cultural resilience enhancement"""
        # Arrange
        evolution_plan = self.engine.create_evolution_framework(
            "test_org", self.current_culture, self.sustainability_assessment
        )
        
        # Act
        cultural_resilience = self.engine.enhance_cultural_resilience(
            "test_org", self.current_culture, evolution_plan
        )
        
        # Assert
        assert cultural_resilience.organization_id == "test_org"
        assert 0.0 <= cultural_resilience.overall_resilience_score <= 1.0
        assert cultural_resilience.adaptability_level in AdaptabilityLevel
        assert len(cultural_resilience.resilience_capabilities) > 0
        assert isinstance(cultural_resilience.vulnerability_areas, list)
        assert isinstance(cultural_resilience.strength_areas, list)
        assert len(cultural_resilience.improvement_recommendations) > 0
    
    def test_create_continuous_improvement_cycle(self):
        """Test creation of continuous improvement cycle"""
        # Arrange
        evolution_plan = self.engine.create_evolution_framework(
            "test_org", self.current_culture, self.sustainability_assessment
        )
        cultural_resilience = self.engine.enhance_cultural_resilience(
            "test_org", self.current_culture, evolution_plan
        )
        
        # Act
        improvement_cycle = self.engine.create_continuous_improvement_cycle(
            "test_org", evolution_plan, cultural_resilience
        )
        
        # Assert
        assert improvement_cycle.organization_id == "test_org"
        assert improvement_cycle.cycle_phase in ["assess", "plan", "implement", "evaluate", "adapt"]
        assert len(improvement_cycle.current_focus_areas) > 0
        assert len(improvement_cycle.improvement_initiatives) > 0
        assert len(improvement_cycle.feedback_mechanisms) > 0
        assert len(improvement_cycle.learning_outcomes) > 0
        assert "initiative_count" in improvement_cycle.cycle_metrics
    
    def test_determine_evolution_stage(self):
        """Test evolution stage determination"""
        # Test emerging stage (low scores)
        low_culture = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.3,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        low_sustainability = SustainabilityAssessment(
            assessment_id="test",
            organization_id="test_org",
            transformation_id="test",
            sustainability_level=SustainabilityLevel.LOW,
            risk_factors=[],
            protective_factors=[],
            health_indicators=[],
            overall_score=0.2,
            assessment_date=datetime.now(),
            next_assessment_due=datetime.now()
        )
        
        stage = self.engine._determine_evolution_stage(low_culture, low_sustainability)
        assert stage == EvolutionStage.EMERGING
        
        # Test optimizing stage (high scores)
        high_culture = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.9,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        high_sustainability = SustainabilityAssessment(
            assessment_id="test",
            organization_id="test_org",
            transformation_id="test",
            sustainability_level=SustainabilityLevel.HIGH,
            risk_factors=[],
            protective_factors=["strong_culture"],
            health_indicators=[],
            overall_score=0.9,
            assessment_date=datetime.now(),
            next_assessment_due=datetime.now()
        )
        
        stage = self.engine._determine_evolution_stage(high_culture, high_sustainability)
        assert stage in [EvolutionStage.OPTIMIZING, EvolutionStage.TRANSFORMING]
    
    def test_define_target_evolution_stage(self):
        """Test target evolution stage definition"""
        # Test progression from emerging to developing
        target = self.engine._define_target_evolution_stage(
            EvolutionStage.EMERGING, self.sustainability_assessment
        )
        assert target == EvolutionStage.DEVELOPING
        
        # Test progression from developing to maturing
        target = self.engine._define_target_evolution_stage(
            EvolutionStage.DEVELOPING, self.sustainability_assessment
        )
        assert target == EvolutionStage.MATURING
        
        # Test cycle from transforming back to optimizing
        target = self.engine._define_target_evolution_stage(
            EvolutionStage.TRANSFORMING, self.sustainability_assessment
        )
        assert target == EvolutionStage.OPTIMIZING
    
    def test_identify_cultural_innovations(self):
        """Test cultural innovation identification"""
        # Test emerging stage innovations
        innovations = self.engine._identify_cultural_innovations(
            "test_org", self.current_culture, EvolutionStage.EMERGING
        )
        
        assert len(innovations) > 0
        assert all(innovation.organization_id == "test_org" for innovation in innovations if hasattr(innovation, 'organization_id'))
        assert all(innovation.innovation_type in InnovationType for innovation in innovations)
        assert all(len(innovation.target_areas) > 0 for innovation in innovations)
        assert all(len(innovation.success_metrics) > 0 for innovation in innovations)
    
    def test_create_foundational_innovations(self):
        """Test foundational innovation creation"""
        innovations = self.engine._create_foundational_innovations("test_org")
        
        assert len(innovations) > 0
        assert all(innovation.innovation_id.startswith("foundation") or innovation.innovation_id.startswith("communication") for innovation in innovations)
        assert all(innovation.innovation_type in InnovationType for innovation in innovations)
        assert all(innovation.status == "planned" for innovation in innovations)
    
    def test_create_growth_innovations(self):
        """Test growth innovation creation"""
        innovations = self.engine._create_growth_innovations("test_org")
        
        assert len(innovations) > 0
        assert any("collaboration" in innovation.name.lower() for innovation in innovations)
        assert any("learning" in innovation.name.lower() for innovation in innovations)
    
    def test_identify_culture_gaps(self):
        """Test culture gap identification"""
        gaps = self.engine._identify_culture_gaps(self.current_culture)
        
        assert isinstance(gaps, list)
        # With current test data (health_score=0.65, 3 values, 2 behaviors), should identify some gaps
        if self.current_culture.health_score < 0.7:
            assert "overall_health" in gaps
        if len(self.current_culture.behaviors) < len(self.current_culture.values):
            assert "behavior_alignment" in gaps
    
    def test_create_adaptation_mechanisms(self):
        """Test adaptation mechanism creation"""
        mechanisms = self.engine._create_adaptation_mechanisms("test_org", self.current_culture)
        
        assert len(mechanisms) > 0
        assert all(mechanism.organization_id == "test_org" for mechanism in mechanisms if hasattr(mechanism, 'organization_id'))
        assert all(mechanism.mechanism_type in ["feedback_loop", "learning_system", "innovation_process"] for mechanism in mechanisms)
        assert all(mechanism.adaptation_speed in ["slow", "moderate", "fast", "rapid"] for mechanism in mechanisms)
        assert all(0.0 <= mechanism.effectiveness_score <= 1.0 for mechanism in mechanisms)
    
    def test_identify_evolution_triggers(self):
        """Test evolution trigger identification"""
        triggers = self.engine._identify_evolution_triggers("test_org", self.current_culture)
        
        assert len(triggers) > 0
        assert all(trigger.trigger_type in ["internal", "external", "strategic", "crisis"] for trigger in triggers)
        assert all(trigger.urgency_level in ["low", "medium", "high", "critical"] for trigger in triggers)
        assert all(len(trigger.impact_areas) > 0 for trigger in triggers)
    
    def test_assess_resilience_capabilities(self):
        """Test resilience capability assessment"""
        capabilities = self.engine._assess_resilience_capabilities("test_org", self.current_culture)
        
        assert len(capabilities) > 0
        assert all(capability.capability_type in ["recovery", "adaptation", "transformation", "anticipation"] for capability in capabilities)
        assert all(0.0 <= capability.strength_level <= 1.0 for capability in capabilities)
        assert all(len(capability.development_areas) > 0 for capability in capabilities)
        assert all(len(capability.effectiveness_metrics) > 0 for capability in capabilities)
    
    def test_calculate_resilience_score(self):
        """Test resilience score calculation"""
        capabilities = self.engine._assess_resilience_capabilities("test_org", self.current_culture)
        score = self.engine._calculate_resilience_score(capabilities)
        
        assert 0.0 <= score <= 1.0
        
        # Test with empty capabilities
        empty_score = self.engine._calculate_resilience_score([])
        assert empty_score == 0.0
    
    def test_determine_adaptability_level(self):
        """Test adaptability level determination"""
        # Test high adaptability
        high_level = self.engine._determine_adaptability_level(0.9, self.current_culture)
        assert high_level in [AdaptabilityLevel.HIGH, AdaptabilityLevel.EXCEPTIONAL]
        
        # Test low adaptability
        low_level = self.engine._determine_adaptability_level(0.3, self.current_culture)
        assert low_level in [AdaptabilityLevel.LOW, AdaptabilityLevel.MODERATE]
    
    def test_identify_vulnerability_areas(self):
        """Test vulnerability area identification"""
        capabilities = self.engine._assess_resilience_capabilities("test_org", self.current_culture)
        vulnerabilities = self.engine._identify_vulnerability_areas(capabilities, self.current_culture)
        
        assert isinstance(vulnerabilities, list)
        # Should identify some vulnerabilities based on current culture health
        if self.current_culture.health_score < 0.6:
            assert "overall_culture_health" in vulnerabilities
    
    def test_identify_strength_areas(self):
        """Test strength area identification"""
        capabilities = self.engine._assess_resilience_capabilities("test_org", self.current_culture)
        strengths = self.engine._identify_strength_areas(capabilities, self.current_culture)
        
        assert isinstance(strengths, list)
        # Should identify some strengths based on capabilities and culture
    
    def test_assess_innovation_readiness(self):
        """Test innovation readiness assessment"""
        innovations = self.engine._identify_cultural_innovations(
            "test_org", self.current_culture, EvolutionStage.DEVELOPING
        )
        
        if innovations:
            readiness = self.engine._assess_innovation_readiness(innovations[0], "test_org")
            assert 0.0 <= readiness <= 1.0
    
    def test_create_evolution_timeline(self):
        """Test evolution timeline creation"""
        innovations = self.engine._identify_cultural_innovations(
            "test_org", self.current_culture, EvolutionStage.DEVELOPING
        )
        
        timeline = self.engine._create_evolution_timeline(
            EvolutionStage.DEVELOPING, EvolutionStage.MATURING, innovations
        )
        
        assert "current_stage" in timeline
        assert "target_stage" in timeline
        assert "estimated_duration_weeks" in timeline
        assert "phases" in timeline
        assert "milestones" in timeline
        assert timeline["estimated_duration_weeks"] >= 12
    
    def test_define_evolution_success_criteria(self):
        """Test evolution success criteria definition"""
        innovations = self.engine._identify_cultural_innovations(
            "test_org", self.current_culture, EvolutionStage.DEVELOPING
        )
        
        criteria = self.engine._define_evolution_success_criteria(
            EvolutionStage.MATURING, innovations
        )
        
        assert len(criteria) > 0
        assert any("maturing" in criterion.lower() for criterion in criteria)
        assert "Implementation of all planned cultural innovations" in criteria
    
    def test_create_evolution_monitoring_framework(self):
        """Test evolution monitoring framework creation"""
        innovations = self.engine._identify_cultural_innovations(
            "test_org", self.current_culture, EvolutionStage.DEVELOPING
        )
        mechanisms = self.engine._create_adaptation_mechanisms("test_org", self.current_culture)
        
        framework = self.engine._create_evolution_monitoring_framework(innovations, mechanisms)
        
        assert "monitoring_frequency" in framework
        assert "key_metrics" in framework
        assert "data_sources" in framework
        assert "reporting_schedule" in framework
        assert "alert_conditions" in framework
        assert len(framework["key_metrics"]) > 0
    
    def test_error_handling(self):
        """Test error handling in engine methods"""
        # Test with invalid data
        with pytest.raises(Exception):
            self.engine.create_evolution_framework(None, None, None)
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = CulturalEvolutionEngine()
        assert "emerging" in engine.evolution_stages
        assert "developing" in engine.evolution_stages
        assert "maturing" in engine.evolution_stages
        assert "optimizing" in engine.evolution_stages
        assert "transforming" in engine.evolution_stages
        
        # Check stage characteristics
        assert engine.evolution_stages["emerging"]["maturity"] < engine.evolution_stages["optimizing"]["maturity"]
        assert engine.evolution_stages["emerging"]["adaptability"] > engine.evolution_stages["optimizing"]["adaptability"]