"""
Tests for Cultural Change Resistance Detection Engine
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

from scrollintel.engines.resistance_detection_engine import ResistanceDetectionEngine
from scrollintel.models.resistance_detection_models import (
    ResistanceDetection, ResistanceSource, ResistanceImpactAssessment,
    ResistancePrediction, ResistanceType, ResistanceSeverity
)
from scrollintel.models.cultural_assessment_models import Organization
from scrollintel.models.transformation_roadmap_models import Transformation


class TestResistanceDetectionEngine:
    """Test suite for ResistanceDetectionEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create ResistanceDetectionEngine instance"""
        return ResistanceDetectionEngine()
    
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
    def sample_monitoring_data(self):
        """Create sample monitoring data"""
        return {
            "behavioral_data": {
                "attendance": {"meeting_attendance": 0.65, "training_attendance": 0.70},
                "participation": {"active_participation": 0.60, "voluntary_feedback": 0.55},
                "compliance": {"policy_compliance": 0.80, "process_adherence": 0.75}
            },
            "communication_data": {
                "sentiment": {"overall_sentiment": -0.15, "change_sentiment": -0.25},
                "feedback": {"negative_feedback_rate": 0.30, "concern_reports": 15},
                "concerns": {"resistance_indicators": 8, "rumor_reports": 3}
            },
            "engagement_data": {
                "scores": {"engagement_score": 0.65, "satisfaction_score": 0.60},
                "voluntary_participation": {"initiative_participation": 0.45}
            },
            "performance_data": {
                "productivity": {"team_productivity": 0.85, "individual_productivity": 0.80},
                "quality": {"work_quality": 0.90, "error_rate": 0.05}
            }
        }
    
    def test_detect_resistance_patterns_success(self, engine, sample_organization, sample_transformation, sample_monitoring_data):
        """Test successful resistance pattern detection"""
        detections = engine.detect_resistance_patterns(
            organization=sample_organization,
            transformation=sample_transformation,
            monitoring_data=sample_monitoring_data
        )
        
        assert isinstance(detections, list)
        # Detections would be empty in this mock implementation
        # In a real implementation, this would detect actual patterns
    
    def test_detect_resistance_patterns_empty_data(self, engine, sample_organization, sample_transformation):
        """Test resistance detection with empty monitoring data"""
        empty_data = {}
        
        detections = engine.detect_resistance_patterns(
            organization=sample_organization,
            transformation=sample_transformation,
            monitoring_data=empty_data
        )
        
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_analyze_resistance_sources_success(self, engine, sample_organization):
        """Test successful resistance source analysis"""
        detection = ResistanceDetection(
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
        
        sources = engine.analyze_resistance_sources(
            detection=detection,
            organization=sample_organization
        )
        
        assert isinstance(sources, list)
        # Sources would be empty in this mock implementation
    
    def test_assess_resistance_impact_success(self, engine, sample_transformation):
        """Test successful resistance impact assessment"""
        detection = ResistanceDetection(
            id="det_001",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.ACTIVE_OPPOSITION,
            source=None,
            severity=ResistanceSeverity.HIGH,
            confidence_score=0.85,
            detected_at=datetime.now(),
            indicators_triggered=["negative_feedback", "public_criticism"],
            affected_areas=["leadership_team"],
            potential_impact={"success_reduction": 0.25},
            detection_method="sentiment_analysis",
            raw_data={}
        )
        
        impact_assessment = engine.assess_resistance_impact(
            detection=detection,
            transformation=sample_transformation
        )
        
        assert isinstance(impact_assessment, ResistanceImpactAssessment)
        assert impact_assessment.detection_id == detection.id
        assert isinstance(impact_assessment.transformation_impact, dict)
        assert isinstance(impact_assessment.timeline_impact, dict)
        assert isinstance(impact_assessment.resource_impact, dict)
        assert isinstance(impact_assessment.stakeholder_impact, dict)
        assert isinstance(impact_assessment.success_probability_reduction, float)
        assert isinstance(impact_assessment.cascading_effects, list)
        assert isinstance(impact_assessment.critical_path_disruption, bool)
        assert 0 <= impact_assessment.assessment_confidence <= 1
    
    def test_predict_future_resistance_success(self, engine, sample_organization, sample_transformation):
        """Test successful future resistance prediction"""
        historical_data = {
            "past_detections": [
                {"type": "passive_resistance", "phase": "implementation", "probability": 0.7}
            ],
            "stakeholder_patterns": {
                "middle_management": {"resistance_tendency": 0.6},
                "front_line_employees": {"resistance_tendency": 0.4}
            },
            "intervention_history": {
                "training_programs": {"resistance_rate": 0.3},
                "policy_changes": {"resistance_rate": 0.5}
            }
        }
        
        predictions = engine.predict_future_resistance(
            organization=sample_organization,
            transformation=sample_transformation,
            historical_data=historical_data
        )
        
        assert isinstance(predictions, list)
        # Predictions would be empty in this mock implementation
    
    def test_resistance_patterns_initialization(self, engine):
        """Test resistance patterns are properly initialized"""
        assert hasattr(engine, 'resistance_patterns')
        assert isinstance(engine.resistance_patterns, list)
        assert len(engine.resistance_patterns) > 0
        
        # Check first pattern structure
        pattern = engine.resistance_patterns[0]
        assert hasattr(pattern, 'id')
        assert hasattr(pattern, 'pattern_type')
        assert hasattr(pattern, 'description')
        assert hasattr(pattern, 'indicators')
        assert hasattr(pattern, 'typical_sources')
    
    def test_detection_indicators_initialization(self, engine):
        """Test detection indicators are properly initialized"""
        assert hasattr(engine, 'detection_indicators')
        assert isinstance(engine.detection_indicators, list)
        assert len(engine.detection_indicators) > 0
        
        # Check first indicator structure
        indicator = engine.detection_indicators[0]
        assert hasattr(indicator, 'id')
        assert hasattr(indicator, 'indicator_type')
        assert hasattr(indicator, 'description')
        assert hasattr(indicator, 'measurement_method')
        assert hasattr(indicator, 'threshold_values')
        assert hasattr(indicator, 'weight')
        assert hasattr(indicator, 'reliability_score')
    
    def test_validate_detections(self, engine):
        """Test detection validation logic"""
        detections = [
            ResistanceDetection(
                id="det_001",
                organization_id="org_001",
                transformation_id="trans_001",
                resistance_type=ResistanceType.PASSIVE_RESISTANCE,
                source=None,
                severity=ResistanceSeverity.MODERATE,
                confidence_score=0.8,  # Above threshold
                detected_at=datetime.now(),
                indicators_triggered=[],
                affected_areas=[],
                potential_impact={},
                detection_method="test",
                raw_data={}
            ),
            ResistanceDetection(
                id="det_002",
                organization_id="org_001",
                transformation_id="trans_001",
                resistance_type=ResistanceType.SKEPTICISM,
                source=None,
                severity=ResistanceSeverity.LOW,
                confidence_score=0.5,  # Below threshold
                detected_at=datetime.now(),
                indicators_triggered=[],
                affected_areas=[],
                potential_impact={},
                detection_method="test",
                raw_data={}
            )
        ]
        
        validated = engine._validate_detections(detections)
        
        assert len(validated) == 1
        assert validated[0].confidence_score >= 0.6
    
    def test_calculate_success_probability_reduction(self, engine):
        """Test success probability reduction calculation"""
        detection_low = ResistanceDetection(
            id="det_001",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.SKEPTICISM,
            source=None,
            severity=ResistanceSeverity.LOW,
            confidence_score=0.7,
            detected_at=datetime.now(),
            indicators_triggered=[],
            affected_areas=[],
            potential_impact={},
            detection_method="test",
            raw_data={}
        )
        
        detection_critical = ResistanceDetection(
            id="det_002",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.ACTIVE_OPPOSITION,
            source=None,
            severity=ResistanceSeverity.CRITICAL,
            confidence_score=0.9,
            detected_at=datetime.now(),
            indicators_triggered=[],
            affected_areas=[],
            potential_impact={},
            detection_method="test",
            raw_data={}
        )
        
        transformation = Mock()
        
        reduction_low = engine._calculate_success_probability_reduction(detection_low, transformation)
        reduction_critical = engine._calculate_success_probability_reduction(detection_critical, transformation)
        
        assert reduction_critical > reduction_low
        assert 0 <= reduction_low <= 1
        assert 0 <= reduction_critical <= 1
    
    def test_error_handling(self, engine):
        """Test error handling in resistance detection"""
        with pytest.raises(Exception):
            engine.detect_resistance_patterns(
                organization=None,  # Invalid input
                transformation=None,
                monitoring_data={}
            )
    
    @patch('scrollintel.engines.resistance_detection_engine.logging')
    def test_logging(self, mock_logging, engine, sample_organization, sample_transformation, sample_monitoring_data):
        """Test logging functionality"""
        engine.detect_resistance_patterns(
            organization=sample_organization,
            transformation=sample_transformation,
            monitoring_data=sample_monitoring_data
        )
        
        # Verify logging was called
        assert mock_logging.getLogger.called