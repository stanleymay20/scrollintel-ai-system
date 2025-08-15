"""
Tests for Cultural Assessment Engine
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import numpy as np

from scrollintel.engines.cultural_assessment_engine import (
    CulturalAssessmentEngine, CultureMapper, DimensionAnalyzer,
    SubcultureIdentifier, HealthMetrics
)
from scrollintel.models.cultural_assessment_models import (
    CulturalAssessmentRequest, CulturalDimension, CultureData,
    SubcultureType, CulturalValue, CulturalBehavior, CulturalNorm
)


class TestCultureMapper:
    """Test cases for CultureMapper"""
    
    def setup_method(self):
        self.culture_mapper = CultureMapper()
        self.sample_data = [
            CultureData(
                organization_id="test_org",
                data_type="survey",
                data_points=[
                    {
                        "employee_id": "emp_1",
                        "department": "Engineering",
                        CulturalDimension.INNOVATION_ORIENTATION.value: 0.8,
                        CulturalDimension.COLLABORATION_STYLE.value: 0.7,
                        "engagement": 0.75,
                        "values": [
                            {
                                "name": "Innovation",
                                "importance": 0.9,
                                "alignment": 0.7,
                                "evidence": ["hackathons", "innovation time"]
                            }
                        ],
                        "behaviors": [
                            {
                                "description": "Regular team collaboration",
                                "frequency": 0.8,
                                "impact": 0.6,
                                "context": "daily work",
                                "examples": ["stand-ups", "pair programming"]
                            }
                        ]
                    }
                ],
                collection_date=datetime.now(),
                source="Employee Survey",
                reliability_score=0.85,
                sample_size=50
            )
        ]
    
    def test_map_organizational_culture(self):
        """Test comprehensive culture mapping"""
        culture_map = self.culture_mapper.map_organizational_culture("test_org", self.sample_data)
        
        assert culture_map.organization_id == "test_org"
        assert culture_map.overall_health_score > 0
        assert culture_map.assessment_confidence > 0
        assert len(culture_map.cultural_dimensions) > 0
        assert CulturalDimension.INNOVATION_ORIENTATION in culture_map.cultural_dimensions
        assert len(culture_map.values) > 0
        assert len(culture_map.behaviors) > 0
    
    def test_analyze_dimensions(self):
        """Test cultural dimensions analysis"""
        dimensions = self.culture_mapper._analyze_dimensions(self.sample_data)
        
        assert isinstance(dimensions, dict)
        assert CulturalDimension.INNOVATION_ORIENTATION in dimensions
        assert CulturalDimension.COLLABORATION_STYLE in dimensions
        assert 0 <= dimensions[CulturalDimension.INNOVATION_ORIENTATION] <= 1
        assert 0 <= dimensions[CulturalDimension.COLLABORATION_STYLE] <= 1
    
    def test_extract_cultural_values(self):
        """Test cultural values extraction"""
        values = self.culture_mapper._extract_cultural_values(self.sample_data)
        
        assert len(values) > 0
        assert isinstance(values[0], CulturalValue)
        assert values[0].name == "Innovation"
        assert 0 <= values[0].importance_score <= 1
        assert 0 <= values[0].alignment_score <= 1
        assert len(values[0].evidence) > 0
    
    def test_extract_cultural_behaviors(self):
        """Test cultural behaviors extraction"""
        behaviors = self.culture_mapper._extract_cultural_behaviors(self.sample_data)
        
        assert len(behaviors) > 0
        assert isinstance(behaviors[0], CulturalBehavior)
        assert behaviors[0].description == "Regular team collaboration"
        assert 0 <= behaviors[0].frequency <= 1
        assert len(behaviors[0].examples) > 0
    
    def test_calculate_overall_health(self):
        """Test overall health calculation"""
        from scrollintel.models.cultural_assessment_models import CulturalHealthMetric
        
        health_metrics = [
            CulturalHealthMetric(
                metric_id="test_metric",
                name="Test Metric",
                value=0.8,
                target_value=0.75,
                trend="improving",
                measurement_date=datetime.now(),
                data_sources=["test"],
                confidence_level=0.9
            )
        ]
        dimensions = {CulturalDimension.INNOVATION_ORIENTATION: 0.7}
        
        health_score = self.culture_mapper._calculate_overall_health(health_metrics, dimensions)
        
        assert 0 <= health_score <= 1
        assert health_score > 0.5  # Should be reasonable given input values


class TestDimensionAnalyzer:
    """Test cases for DimensionAnalyzer"""
    
    def setup_method(self):
        self.dimension_analyzer = DimensionAnalyzer()
        self.sample_culture_data = CultureData(
            organization_id="test_org",
            data_type="survey",
            data_points=[
                {
                    CulturalDimension.INNOVATION_ORIENTATION.value: 0.6,
                    CulturalDimension.COLLABORATION_STYLE.value: 0.8,
                    f"{CulturalDimension.INNOVATION_ORIENTATION.value}_factors": ["leadership support", "resources"]
                }
            ],
            collection_date=datetime.now(),
            source="Test Survey",
            reliability_score=0.8,
            sample_size=30
        )
    
    def test_analyze_cultural_dimensions(self):
        """Test cultural dimensions analysis"""
        analyses = self.dimension_analyzer.analyze_cultural_dimensions(self.sample_culture_data)
        
        assert len(analyses) == len(CulturalDimension)
        assert all(hasattr(analysis, 'dimension') for analysis in analyses)
        assert all(hasattr(analysis, 'current_score') for analysis in analyses)
        assert all(0 <= analysis.current_score <= 1 for analysis in analyses)
    
    def test_analyze_single_dimension(self):
        """Test single dimension analysis"""
        analysis = self.dimension_analyzer._analyze_single_dimension(
            CulturalDimension.INNOVATION_ORIENTATION, 
            self.sample_culture_data
        )
        
        assert analysis.dimension == CulturalDimension.INNOVATION_ORIENTATION
        assert 0 <= analysis.current_score <= 1
        assert analysis.measurement_confidence > 0
        assert isinstance(analysis.contributing_factors, list)
        assert isinstance(analysis.improvement_recommendations, list)
    
    def test_calculate_dimension_score(self):
        """Test dimension score calculation"""
        score = self.dimension_analyzer._calculate_dimension_score(
            CulturalDimension.INNOVATION_ORIENTATION,
            self.sample_culture_data
        )
        
        assert 0 <= score <= 1
        assert score == 0.6  # Should match the test data
    
    def test_generate_improvement_recommendations(self):
        """Test improvement recommendations generation"""
        recommendations = self.dimension_analyzer._generate_improvement_recommendations(
            CulturalDimension.INNOVATION_ORIENTATION,
            0.4,  # Current score
            0.8   # Ideal score
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("innovation" in rec.lower() for rec in recommendations)


class TestSubcultureIdentifier:
    """Test cases for SubcultureIdentifier"""
    
    def setup_method(self):
        self.subculture_identifier = SubcultureIdentifier()
        self.sample_data = [
            CultureData(
                organization_id="test_org",
                data_type="survey",
                data_points=[
                    {
                        "employee_id": "emp_1",
                        "department": "Engineering",
                        "hierarchy_level": "Senior",
                        "generation": "Millennial",
                        CulturalDimension.INNOVATION_ORIENTATION.value: 0.8
                    },
                    {
                        "employee_id": "emp_2",
                        "department": "Engineering",
                        "hierarchy_level": "Junior",
                        "generation": "Gen Z",
                        CulturalDimension.INNOVATION_ORIENTATION.value: 0.9
                    },
                    {
                        "employee_id": "emp_3",
                        "department": "Sales",
                        "hierarchy_level": "Senior",
                        "generation": "Gen X",
                        CulturalDimension.INNOVATION_ORIENTATION.value: 0.5
                    }
                ],
                collection_date=datetime.now(),
                source="Employee Survey",
                reliability_score=0.85,
                sample_size=3
            )
        ]
    
    def test_identify_subcultures(self):
        """Test subculture identification"""
        subcultures = self.subculture_identifier.identify_subcultures("test_org", self.sample_data)
        
        assert len(subcultures) > 0
        assert any(s.type == SubcultureType.DEPARTMENTAL for s in subcultures)
        assert any(s.type == SubcultureType.HIERARCHICAL for s in subcultures)
        assert any(s.type == SubcultureType.GENERATIONAL for s in subcultures)
        
        # Check that all subcultures have required attributes
        for subculture in subcultures:
            assert subculture.subculture_id
            assert subculture.name
            assert len(subculture.members) > 0
            assert 0 <= subculture.strength <= 1
            assert 0 <= subculture.influence <= 1
    
    def test_identify_departmental_subcultures(self):
        """Test departmental subculture identification"""
        subcultures = self.subculture_identifier._identify_departmental_subcultures(self.sample_data)
        
        assert len(subcultures) >= 2  # Engineering and Sales
        dept_names = [s.name for s in subcultures]
        assert any("Engineering" in name for name in dept_names)
        assert any("Sales" in name for name in dept_names)
        
        # Check Engineering department has 2 members
        eng_subculture = next(s for s in subcultures if "Engineering" in s.name)
        assert len(eng_subculture.members) == 2
    
    def test_identify_hierarchical_subcultures(self):
        """Test hierarchical subculture identification"""
        subcultures = self.subculture_identifier._identify_hierarchical_subcultures(self.sample_data)
        
        assert len(subcultures) >= 2  # Senior and Junior
        level_names = [s.name for s in subcultures]
        assert any("Senior" in name for name in level_names)
        assert any("Junior" in name for name in level_names)
    
    def test_identify_generational_subcultures(self):
        """Test generational subculture identification"""
        subcultures = self.subculture_identifier._identify_generational_subcultures(self.sample_data)
        
        assert len(subcultures) >= 3  # Millennial, Gen Z, Gen X
        gen_names = [s.name for s in subcultures]
        assert any("Millennial" in name for name in gen_names)
        assert any("Gen Z" in name for name in gen_names)
        assert any("Gen X" in name for name in gen_names)
    
    def test_assess_subculture_strength(self):
        """Test subculture strength assessment"""
        from scrollintel.models.cultural_assessment_models import Subculture
        
        subculture = Subculture(
            subculture_id="test_sub",
            name="Test Subculture",
            type=SubcultureType.DEPARTMENTAL,
            members=["emp_1", "emp_2"],
            characteristics={
                "innovation_score": [0.8, 0.85, 0.82],  # Low variance = high consistency
                "collaboration": [0.7, 0.75, 0.72]
            },
            values=[],
            behaviors=[],
            strength=0.0,
            influence=0.0
        )
        
        strength = self.subculture_identifier._assess_subculture_strength(subculture, self.sample_data)
        
        assert 0 <= strength <= 1
        assert strength > 0.5  # Should be high due to low variance in characteristics


class TestHealthMetrics:
    """Test cases for HealthMetrics"""
    
    def setup_method(self):
        self.health_metrics = HealthMetrics()
        
        # Create sample culture map
        from scrollintel.models.cultural_assessment_models import CultureMap
        
        self.sample_culture_map = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={
                CulturalDimension.INNOVATION_ORIENTATION: 0.8,
                CulturalDimension.COLLABORATION_STYLE: 0.7,
                CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.4
            },
            values=[
                CulturalValue(
                    name="Innovation",
                    description="Innovation value",
                    importance_score=0.9,
                    alignment_score=0.7,
                    evidence=["hackathons"]
                )
            ],
            behaviors=[
                CulturalBehavior(
                    behavior_id="b1",
                    description="High engagement behavior",
                    frequency=0.8,
                    impact_score=0.7,
                    context="work",
                    examples=["collaboration"]
                )
            ],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.75,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
    
    def test_calculate_cultural_health_metrics(self):
        """Test cultural health metrics calculation"""
        metrics = self.health_metrics.calculate_cultural_health_metrics(self.sample_culture_map)
        
        assert len(metrics) > 0
        assert all(hasattr(metric, 'metric_id') for metric in metrics)
        assert all(hasattr(metric, 'name') for metric in metrics)
        assert all(hasattr(metric, 'value') for metric in metrics)
        assert all(0 <= metric.value <= 1 for metric in metrics)
        assert all(0 <= metric.confidence_level <= 1 for metric in metrics)
        
        # Check for expected metrics
        metric_ids = [m.metric_id for m in metrics]
        assert "cultural_alignment" in metric_ids
        assert "innovation_index" in metric_ids
        assert "collaboration_effectiveness" in metric_ids
    
    def test_calculate_alignment_metric(self):
        """Test cultural alignment metric calculation"""
        metric = self.health_metrics._calculate_alignment_metric(self.sample_culture_map)
        
        assert metric.metric_id == "cultural_alignment"
        assert metric.name == "Cultural Alignment"
        assert 0 <= metric.value <= 1
        assert metric.value == 0.7  # Should match the alignment score in sample data
        assert metric.target_value == 0.75
    
    def test_calculate_innovation_metric(self):
        """Test innovation index calculation"""
        metric = self.health_metrics._calculate_innovation_metric(self.sample_culture_map)
        
        assert metric.metric_id == "innovation_index"
        assert metric.name == "Innovation Index"
        assert metric.value == 0.8  # Should match the innovation orientation score
        assert metric.target_value == 0.8
    
    def test_calculate_change_readiness_metric(self):
        """Test change readiness score calculation"""
        metric = self.health_metrics._calculate_change_readiness_metric(self.sample_culture_map)
        
        assert metric.metric_id == "change_readiness"
        assert metric.name == "Change Readiness Score"
        assert 0 <= metric.value <= 1
        
        # Should be based on uncertainty avoidance and innovation orientation
        expected_value = (0.6 + 0.8) / 2.0  # (1-0.4) + 0.8 / 2
        assert abs(metric.value - expected_value) < 0.01


class TestCulturalAssessmentEngine:
    """Test cases for main CulturalAssessmentEngine"""
    
    def setup_method(self):
        self.engine = CulturalAssessmentEngine()
        self.sample_request = CulturalAssessmentRequest(
            organization_id="test_org",
            assessment_type="comprehensive",
            focus_areas=[CulturalDimension.INNOVATION_ORIENTATION, CulturalDimension.COLLABORATION_STYLE],
            data_sources=["survey", "observation"],
            timeline="2 weeks",
            stakeholders=["HR", "Leadership"]
        )
    
    def test_conduct_cultural_assessment(self):
        """Test comprehensive cultural assessment"""
        result = self.engine.conduct_cultural_assessment(self.sample_request)
        
        assert result.organization_id == "test_org"
        assert result.culture_map is not None
        assert result.culture_map.organization_id == "test_org"
        assert len(result.dimension_analyses) > 0
        assert len(result.key_findings) > 0
        assert len(result.recommendations) > 0
        assert result.assessment_summary
        assert 0 <= result.confidence_score <= 1
        assert result.completion_date is not None
    
    def test_collect_culture_data(self):
        """Test culture data collection simulation"""
        data_sources = self.engine._collect_culture_data(self.sample_request)
        
        assert len(data_sources) > 0
        assert all(isinstance(ds, CultureData) for ds in data_sources)
        assert all(ds.organization_id == "test_org" for ds in data_sources)
        assert all(len(ds.data_points) > 0 for ds in data_sources)
        assert all(0 <= ds.reliability_score <= 1 for ds in data_sources)
    
    def test_generate_key_findings(self):
        """Test key findings generation"""
        # Create a mock culture map
        from scrollintel.models.cultural_assessment_models import CultureMap
        
        culture_map = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={CulturalDimension.INNOVATION_ORIENTATION: 0.8},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.85,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        findings = self.engine._generate_key_findings(culture_map, [])
        
        assert isinstance(findings, list)
        assert len(findings) > 0
        assert any("strong cultural health" in finding.lower() for finding in findings)
    
    def test_generate_recommendations(self):
        """Test recommendations generation"""
        from scrollintel.models.cultural_assessment_models import CultureMap, DimensionAnalysis
        
        culture_map = CultureMap(
            organization_id="test_org",
            assessment_date=datetime.now(),
            cultural_dimensions={CulturalDimension.INNOVATION_ORIENTATION: 0.4},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=0.5,
            assessment_confidence=0.8,
            data_sources=["survey"]
        )
        
        dimension_analyses = [
            DimensionAnalysis(
                dimension=CulturalDimension.INNOVATION_ORIENTATION,
                current_score=0.4,
                ideal_score=0.8,
                gap_analysis="Large gap",
                contributing_factors=["lack of resources"],
                improvement_recommendations=["Implement innovation time", "Create innovation challenges"],
                measurement_confidence=0.8
            )
        ]
        
        recommendations = self.engine._generate_recommendations(culture_map, dimension_analyses)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("cultural transformation" in rec.lower() for rec in recommendations)
    
    @patch('numpy.random.normal')
    def test_assessment_with_mocked_data(self, mock_normal):
        """Test assessment with controlled random data"""
        # Mock random data generation for consistent testing
        mock_normal.return_value = 0.7
        
        result = self.engine.conduct_cultural_assessment(self.sample_request)
        
        assert result is not None
        assert result.confidence_score > 0
        assert len(result.key_findings) > 0
        assert len(result.recommendations) > 0


class TestIntegration:
    """Integration tests for Cultural Assessment Engine"""
    
    def test_full_assessment_workflow(self):
        """Test complete assessment workflow"""
        engine = CulturalAssessmentEngine()
        
        request = CulturalAssessmentRequest(
            organization_id="integration_test_org",
            assessment_type="comprehensive",
            focus_areas=list(CulturalDimension),
            data_sources=["survey", "observation", "interview"],
            timeline="3 weeks",
            stakeholders=["HR", "Leadership", "Employees"]
        )
        
        # Conduct assessment
        result = engine.conduct_cultural_assessment(request)
        
        # Verify complete result structure
        assert result.organization_id == "integration_test_org"
        assert result.culture_map.overall_health_score > 0
        assert len(result.culture_map.cultural_dimensions) == len(CulturalDimension)
        assert len(result.dimension_analyses) > 0
        assert len(result.key_findings) > 0
        assert len(result.recommendations) > 0
        assert result.confidence_score > 0
        
        # Verify culture map completeness
        culture_map = result.culture_map
        assert len(culture_map.values) > 0
        assert len(culture_map.behaviors) > 0
        assert len(culture_map.subcultures) > 0
        assert len(culture_map.health_metrics) > 0
        
        # Verify subcultures have been identified
        subculture_types = {s.type for s in culture_map.subcultures}
        assert len(subculture_types) >= 1  # At least one type should be identified
        
        # Verify health metrics are reasonable
        for metric in culture_map.health_metrics:
            assert 0 <= metric.value <= 1
            assert 0 <= metric.confidence_level <= 1
            assert metric.trend in ["improving", "declining", "stable"]
    
    def test_component_integration(self):
        """Test integration between different engine components"""
        engine = CulturalAssessmentEngine()
        
        # Test that all components work together
        assert engine.culture_mapper is not None
        assert engine.dimension_analyzer is not None
        assert engine.subculture_identifier is not None
        assert engine.health_metrics is not None
        
        # Test data flow between components
        request = CulturalAssessmentRequest(
            organization_id="component_test_org",
            assessment_type="focused",
            focus_areas=[CulturalDimension.INNOVATION_ORIENTATION],
            data_sources=["survey"],
            timeline="1 week",
            stakeholders=["HR"]
        )
        
        result = engine.conduct_cultural_assessment(request)
        
        # Verify that data flows correctly between components
        assert result.culture_map.cultural_dimensions
        assert result.dimension_analyses
        assert result.culture_map.health_metrics
        
        # Verify consistency between components
        innovation_score = result.culture_map.cultural_dimensions.get(CulturalDimension.INNOVATION_ORIENTATION)
        innovation_analysis = next(
            (a for a in result.dimension_analyses if a.dimension == CulturalDimension.INNOVATION_ORIENTATION),
            None
        )
        
        if innovation_analysis:
            assert abs(innovation_score - innovation_analysis.current_score) < 0.1  # Should be similar


if __name__ == "__main__":
    pytest.main([__file__])