"""
Tests for Cultural Strategic Integration Engine

Tests the integration between cultural transformation and strategic planning systems.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.cultural_strategic_integration import CulturalStrategicIntegrationEngine
from scrollintel.models.cultural_strategic_integration_models import (
    StrategicObjective, StrategicInitiative, StrategicObjectiveType, ImpactLevel
)
from scrollintel.models.cultural_assessment_models import Culture


class TestCulturalStrategicIntegrationEngine:
    """Test cases for Cultural Strategic Integration Engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = CulturalStrategicIntegrationEngine()
        
        # Sample strategic objective
        self.sample_objective = StrategicObjective(
            id="obj_001",
            name="Digital Transformation Initiative",
            description="Transform organization to digital-first approach",
            objective_type=StrategicObjectiveType.DIGITAL_TRANSFORMATION,
            target_date=datetime.now() + timedelta(days=365),
            success_metrics=["Digital adoption rate > 80%", "Process efficiency +30%"],
            priority_level=1,
            owner="CTO",
            cultural_requirements=[
                "Innovation mindset",
                "Agility and adaptability",
                "Continuous learning culture"
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Sample strategic initiative
        self.sample_initiative = StrategicInitiative(
            id="init_001",
            name="Cloud Migration Project",
            description="Migrate all systems to cloud infrastructure",
            objective_ids=["obj_001"],
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=180),
            budget=500000.0,
            team_size=25,
            cultural_impact_level=ImpactLevel.HIGH,
            cultural_requirements=[
                "Technical expertise",
                "Change readiness",
                "Collaboration skills"
            ],
            success_criteria=["100% system migration", "Zero downtime"],
            status="planned"
        )
        
        # Sample culture
        self.sample_culture = Culture(
            organization_id="org_001",
            cultural_dimensions={
                "innovation": 0.7,
                "agility": 0.6,
                "learning": 0.8,
                "collaboration": 0.9
            },
            values=["Innovation", "Excellence", "Teamwork"],
            behaviors=["Knowledge sharing", "Risk taking", "Continuous improvement"],
            norms=["Open communication", "Data-driven decisions"],
            subcultures=[],
            health_score=0.75,
            assessment_date=datetime.now()
        )
    
    def test_assess_cultural_alignment(self):
        """Test cultural alignment assessment"""
        alignment = self.engine.assess_cultural_alignment(
            self.sample_objective,
            self.sample_culture
        )
        
        assert alignment is not None
        assert alignment.objective_id == "obj_001"
        assert 0.0 <= alignment.alignment_score <= 1.0
        assert alignment.alignment_level is not None
        assert isinstance(alignment.gap_analysis, dict)
        assert isinstance(alignment.recommendations, list)
        assert alignment.assessed_by == "cultural_strategic_integration_engine"
    
    def test_assess_cultural_impact(self):
        """Test cultural impact assessment"""
        assessment = self.engine.assess_cultural_impact(
            self.sample_initiative,
            self.sample_culture
        )
        
        assert assessment is not None
        assert assessment.initiative_id == "init_001"
        assert 0.0 <= assessment.impact_score <= 1.0
        assert assessment.impact_level is not None
        assert isinstance(assessment.cultural_enablers, list)
        assert isinstance(assessment.cultural_barriers, list)
        assert isinstance(assessment.mitigation_strategies, list)
        assert 0.0 <= assessment.success_probability <= 1.0
        assert assessment.assessor == "cultural_strategic_integration_engine"
    
    def test_make_culture_aware_decision(self):
        """Test culture-aware decision making"""
        decision_context = "Choose technology stack for new platform"
        strategic_options = [
            {"name": "Option A", "technology": "React/Node.js", "cost": 100000},
            {"name": "Option B", "technology": "Angular/.NET", "cost": 150000},
            {"name": "Option C", "technology": "Vue/Python", "cost": 120000}
        ]
        
        decision = self.engine.make_culture_aware_decision(
            decision_context,
            strategic_options,
            self.sample_culture
        )
        
        assert decision is not None
        assert decision.decision_context == decision_context
        assert decision.strategic_options == strategic_options
        assert decision.recommended_option is not None
        assert decision.cultural_rationale is not None
        assert isinstance(decision.cultural_considerations, dict)
        assert isinstance(decision.risk_assessment, dict)
        assert isinstance(decision.implementation_plan, list)
        assert decision.decision_maker == "cultural_strategic_integration_engine"
    
    def test_generate_integration_report(self):
        """Test integration report generation"""
        objectives = [self.sample_objective]
        initiatives = [self.sample_initiative]
        
        report = self.engine.generate_integration_report(
            objectives,
            initiatives,
            self.sample_culture
        )
        
        assert report is not None
        assert report.report_type == "cultural_strategic_integration"
        assert isinstance(report.alignment_summary, dict)
        assert isinstance(report.impact_assessments, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.success_metrics, dict)
        assert report.generated_by == "cultural_strategic_integration_engine"
    
    def test_alignment_caching(self):
        """Test that alignment assessments are cached"""
        # First assessment
        alignment1 = self.engine.assess_cultural_alignment(
            self.sample_objective,
            self.sample_culture
        )
        
        # Check cache
        cached_alignment = self.engine.alignment_cache.get("obj_001")
        assert cached_alignment is not None
        assert cached_alignment.id == alignment1.id
    
    def test_impact_caching(self):
        """Test that impact assessments are cached"""
        # First assessment
        assessment1 = self.engine.assess_cultural_impact(
            self.sample_initiative,
            self.sample_culture
        )
        
        # Check cache
        cached_assessment = self.engine.impact_assessments.get("init_001")
        assert cached_assessment is not None
        assert cached_assessment.id == assessment1.id
    
    def test_alignment_level_determination(self):
        """Test alignment level determination logic"""
        # Test different scores
        assert self.engine._determine_alignment_level(0.95).value == "fully_aligned"
        assert self.engine._determine_alignment_level(0.75).value == "mostly_aligned"
        assert self.engine._determine_alignment_level(0.55).value == "partially_aligned"
        assert self.engine._determine_alignment_level(0.35).value == "misaligned"
        assert self.engine._determine_alignment_level(0.15).value == "conflicting"
    
    def test_impact_level_determination(self):
        """Test impact level determination logic"""
        # Test different scores
        assert self.engine._determine_impact_level(0.85).value == "critical"
        assert self.engine._determine_impact_level(0.65).value == "high"
        assert self.engine._determine_impact_level(0.45).value == "medium"
        assert self.engine._determine_impact_level(0.25).value == "low"
        assert self.engine._determine_impact_level(0.15).value == "minimal"
    
    def test_requirement_alignment_calculation(self):
        """Test cultural requirement alignment calculation"""
        requirement = "Innovation mindset"
        score = self.engine._calculate_requirement_alignment(
            requirement,
            self.sample_culture
        )
        
        assert 0.0 <= score <= 1.0
        # Should be higher due to innovation dimension in culture
        assert score > 0.5
    
    def test_impact_factors_analysis(self):
        """Test impact factors analysis"""
        factors = self.engine._analyze_impact_factors(
            self.sample_initiative,
            self.sample_culture
        )
        
        assert isinstance(factors, dict)
        assert "scope" in factors
        assert "duration" in factors
        assert "cultural_requirements" in factors
        assert "budget_impact" in factors
        
        # All factors should be between 0 and 1
        for factor_value in factors.values():
            assert 0.0 <= factor_value <= 1.0
    
    def test_success_probability_calculation(self):
        """Test success probability calculation"""
        impact_score = 0.7
        enablers = ["Supportive culture", "Strong leadership"]
        barriers = ["Resistance to change"]
        
        probability = self.engine._calculate_success_probability(
            impact_score,
            enablers,
            barriers
        )
        
        assert 0.0 <= probability <= 1.0
        # Should be reasonable given the inputs
        assert 0.3 <= probability <= 0.8
    
    def test_integration_metrics_calculation(self):
        """Test integration metrics calculation"""
        # Create sample alignments and assessments
        alignments = [
            Mock(alignment_score=0.8),
            Mock(alignment_score=0.6),
            Mock(alignment_score=0.9)
        ]
        
        assessments = [
            Mock(success_probability=0.7),
            Mock(success_probability=0.8)
        ]
        
        metrics = self.engine._calculate_integration_metrics(alignments, assessments)
        
        assert isinstance(metrics, dict)
        assert "average_cultural_alignment" in metrics
        assert "average_success_probability" in metrics
        assert "integration_readiness_score" in metrics
        
        # Check calculated values
        assert metrics["average_cultural_alignment"] == 0.7667  # (0.8 + 0.6 + 0.9) / 3
        assert metrics["average_success_probability"] == 0.75   # (0.7 + 0.8) / 2
    
    def test_error_handling(self):
        """Test error handling in engine methods"""
        # Test with invalid data
        invalid_objective = Mock()
        invalid_objective.cultural_requirements = None  # This should cause an error
        
        with pytest.raises(Exception):
            self.engine.assess_cultural_alignment(invalid_objective, self.sample_culture)
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        # Test with empty objectives and initiatives
        report = self.engine.generate_integration_report([], [], self.sample_culture)
        
        assert report is not None
        assert report.alignment_summary == {}
        assert report.impact_assessments == []


if __name__ == "__main__":
    pytest.main([__file__])