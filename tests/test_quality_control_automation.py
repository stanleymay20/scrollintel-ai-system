"""
Tests for Quality Control Automation system
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.quality_control_automation import (
    QualityControlAutomation, QualityStandard, QualityLevel,
    QualityMetric, QualityAssessment
)

@pytest.fixture
def quality_control():
    """Create quality control automation instance"""
    return QualityControlAutomation()

@pytest.fixture
def sample_process_data():
    """Sample process data for testing"""
    return {
        "clarity_score": 0.85,
        "methodology_score": 0.9,
        "literature_score": 0.8,
        "power_analysis": 0.85,
        "reproducibility": 0.8,
        "testable": True,
        "peer_reviewed": True,
        "comprehensive_literature": True
    }

@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing"""
    return {
        "design_score": 0.9,
        "control_score": 0.85,
        "sample_adequacy": 0.8,
        "precision_score": 0.95,
        "bias_score": 0.9,
        "has_control": True,
        "adequate_sample": True,
        "calibrated": True
    }

class TestQualityControlAutomation:
    """Test quality control automation functionality"""
    
    @pytest.mark.asyncio
    async def test_assess_quality_research(self, quality_control, sample_process_data):
        """Test quality assessment for research process"""
        assessment = await quality_control.assess_quality(
            process_id="research_001",
            process_type="research",
            process_data=sample_process_data
        )
        
        assert assessment.process_id == "research_001"
        assert assessment.process_type == "research"
        assert isinstance(assessment.overall_score, float)
        assert 0.0 <= assessment.overall_score <= 1.0
        assert isinstance(assessment.quality_level, QualityLevel)
        assert len(assessment.metrics) > 0
        assert isinstance(assessment.issues, list)
        assert isinstance(assessment.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_assess_quality_experiment(self, quality_control, sample_experiment_data):
        """Test quality assessment for experiment process"""
        assessment = await quality_control.assess_quality(
            process_id="experiment_001",
            process_type="experiment",
            process_data=sample_experiment_data
        )
        
        assert assessment.process_id == "experiment_001"
        assert assessment.process_type == "experiment"
        assert isinstance(assessment.overall_score, float)
        assert len(assessment.metrics) > 0
        
        # Check that experiment-specific metrics are included
        metric_names = [metric.name for metric in assessment.metrics]
        assert "design_validity" in metric_names
        assert "control_adequacy" in metric_names
    
    @pytest.mark.asyncio
    async def test_enforce_quality_standards_good_quality(self, quality_control):
        """Test quality standards enforcement for good quality"""
        # Create high-quality assessment
        metrics = [
            QualityMetric("test_metric", 0.9, 0.8, 1.0, "Test metric")
        ]
        assessment = QualityAssessment(
            process_id="test_001",
            process_type="test",
            overall_score=0.9,
            quality_level=QualityLevel.GOOD,
            metrics=metrics,
            issues=[],
            recommendations=[]
        )
        
        allowed = await quality_control.enforce_quality_standards("test_001", assessment)
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_enforce_quality_standards_poor_quality(self, quality_control):
        """Test quality standards enforcement for poor quality"""
        # Create poor-quality assessment
        metrics = [
            QualityMetric("test_metric", 0.4, 0.8, 1.0, "Test metric")
        ]
        assessment = QualityAssessment(
            process_id="test_002",
            process_type="test",
            overall_score=0.4,
            quality_level=QualityLevel.POOR,
            metrics=metrics,
            issues=["Critical issue found"],
            recommendations=["Fix critical issue"]
        )
        
        allowed = await quality_control.enforce_quality_standards("test_002", assessment)
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_enforce_quality_standards_acceptable_quality(self, quality_control):
        """Test quality standards enforcement for acceptable quality"""
        # Create acceptable-quality assessment
        metrics = [
            QualityMetric("test_metric", 0.75, 0.7, 1.0, "Test metric")
        ]
        assessment = QualityAssessment(
            process_id="test_003",
            process_type="test",
            overall_score=0.75,
            quality_level=QualityLevel.ACCEPTABLE,
            metrics=metrics,
            issues=["Minor issue"],
            recommendations=["Consider improvement"]
        )
        
        allowed = await quality_control.enforce_quality_standards("test_003", assessment)
        assert allowed is True
    
    def test_get_applicable_standards(self, quality_control):
        """Test getting applicable standards for different process types"""
        research_standards = quality_control._get_applicable_standards("research")
        assert QualityStandard.RESEARCH_RIGOR in research_standards
        assert QualityStandard.KNOWLEDGE_ACCURACY in research_standards
        
        experiment_standards = quality_control._get_applicable_standards("experiment")
        assert QualityStandard.EXPERIMENTAL_VALIDITY in experiment_standards
        assert QualityStandard.VALIDATION_COMPLETENESS in experiment_standards
        
        prototype_standards = quality_control._get_applicable_standards("prototype")
        assert QualityStandard.PROTOTYPE_FUNCTIONALITY in prototype_standards
    
    @pytest.mark.asyncio
    async def test_calculate_metric_value(self, quality_control):
        """Test metric value calculation"""
        process_data = {"clarity_score": 0.85}
        
        value = await quality_control._calculate_metric_value("hypothesis_clarity", process_data)
        assert value == 0.85
        
        # Test default value for unknown metric
        value = await quality_control._calculate_metric_value("unknown_metric", process_data)
        assert value == 0.5
    
    @pytest.mark.asyncio
    async def test_validate_rule(self, quality_control):
        """Test rule validation"""
        process_data = {"testable": True, "peer_reviewed": False}
        
        # Test passing rule
        result = await quality_control._validate_rule("hypothesis_must_be_testable", process_data)
        assert result is True
        
        # Test failing rule
        result = await quality_control._validate_rule("methodology_must_be_peer_reviewed", process_data)
        assert result is False
        
        # Test unknown rule (should default to True)
        result = await quality_control._validate_rule("unknown_rule", process_data)
        assert result is True
    
    def test_calculate_overall_score(self, quality_control):
        """Test overall score calculation"""
        metrics = [
            QualityMetric("metric1", 0.8, 0.7, 0.5, "Metric 1"),
            QualityMetric("metric2", 0.9, 0.8, 0.3, "Metric 2"),
            QualityMetric("metric3", 0.7, 0.6, 0.2, "Metric 3")
        ]
        
        score = quality_control._calculate_overall_score(metrics)
        expected_score = (0.8 * 0.5 + 0.9 * 0.3 + 0.7 * 0.2) / (0.5 + 0.3 + 0.2)
        assert abs(score - expected_score) < 0.001
        
        # Test empty metrics
        score = quality_control._calculate_overall_score([])
        assert score == 0.0
    
    def test_determine_quality_level(self, quality_control):
        """Test quality level determination"""
        assert quality_control._determine_quality_level(0.95) == QualityLevel.EXCELLENT
        assert quality_control._determine_quality_level(0.85) == QualityLevel.GOOD
        assert quality_control._determine_quality_level(0.75) == QualityLevel.ACCEPTABLE
        assert quality_control._determine_quality_level(0.55) == QualityLevel.POOR
        assert quality_control._determine_quality_level(0.35) == QualityLevel.UNACCEPTABLE
    
    def test_store_assessment(self, quality_control):
        """Test assessment storage"""
        assessment = QualityAssessment(
            process_id="test_store",
            process_type="test",
            overall_score=0.8,
            quality_level=QualityLevel.GOOD,
            metrics=[],
            issues=[],
            recommendations=[]
        )
        
        quality_control._store_assessment(assessment)
        
        assert "test_store" in quality_control.quality_history
        assert len(quality_control.quality_history["test_store"]) == 1
        assert quality_control.quality_history["test_store"][0] == assessment
    
    @pytest.mark.asyncio
    async def test_optimize_quality_processes(self, quality_control):
        """Test quality process optimization"""
        # Add some sample history
        assessment = QualityAssessment(
            process_id="test_optimize",
            process_type="test",
            overall_score=0.8,
            quality_level=QualityLevel.GOOD,
            metrics=[],
            issues=[],
            recommendations=[]
        )
        quality_control._store_assessment(assessment)
        
        optimization_results = await quality_control.optimize_quality_processes()
        
        assert isinstance(optimization_results, dict)
        assert "threshold_adjustments" in optimization_results
        assert "weight_optimizations" in optimization_results
        assert "new_metrics" in optimization_results
        assert "process_improvements" in optimization_results
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring_start_stop(self, quality_control):
        """Test continuous monitoring start and stop"""
        assert quality_control.monitoring_active is False
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(quality_control.start_continuous_monitoring())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        assert quality_control.monitoring_active is True
        
        # Stop monitoring
        quality_control.stop_continuous_monitoring()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()
        
        assert quality_control.monitoring_active is False

class TestQualityStandardDefinition:
    """Test quality standard definition functionality"""
    
    def test_quality_standard_initialization(self, quality_control):
        """Test quality standard initialization"""
        standards = quality_control.quality_standards
        
        assert len(standards) == 5
        assert QualityStandard.RESEARCH_RIGOR in standards
        assert QualityStandard.EXPERIMENTAL_VALIDITY in standards
        assert QualityStandard.PROTOTYPE_FUNCTIONALITY in standards
        assert QualityStandard.VALIDATION_COMPLETENESS in standards
        assert QualityStandard.KNOWLEDGE_ACCURACY in standards
        
        # Check research rigor standard
        research_standard = standards[QualityStandard.RESEARCH_RIGOR]
        assert "hypothesis_clarity" in research_standard.metrics
        assert "methodology_soundness" in research_standard.metrics
        assert research_standard.thresholds["hypothesis_clarity"] == 0.8
        assert research_standard.weights["hypothesis_clarity"] == 0.2

class TestQualityMetric:
    """Test quality metric functionality"""
    
    def test_quality_metric_creation(self):
        """Test quality metric creation"""
        metric = QualityMetric(
            name="test_metric",
            value=0.85,
            threshold=0.8,
            weight=0.5,
            description="Test metric"
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 0.85
        assert metric.threshold == 0.8
        assert metric.weight == 0.5
        assert metric.description == "Test metric"
        assert isinstance(metric.measurement_time, datetime)

class TestQualityAssessment:
    """Test quality assessment functionality"""
    
    def test_quality_assessment_creation(self):
        """Test quality assessment creation"""
        metrics = [
            QualityMetric("metric1", 0.8, 0.7, 0.5, "Metric 1"),
            QualityMetric("metric2", 0.9, 0.8, 0.5, "Metric 2")
        ]
        
        assessment = QualityAssessment(
            process_id="test_assessment",
            process_type="test",
            overall_score=0.85,
            quality_level=QualityLevel.GOOD,
            metrics=metrics,
            issues=["Minor issue"],
            recommendations=["Improve metric1"]
        )
        
        assert assessment.process_id == "test_assessment"
        assert assessment.process_type == "test"
        assert assessment.overall_score == 0.85
        assert assessment.quality_level == QualityLevel.GOOD
        assert len(assessment.metrics) == 2
        assert len(assessment.issues) == 1
        assert len(assessment.recommendations) == 1
        assert isinstance(assessment.assessment_time, datetime)

@pytest.mark.asyncio
async def test_integration_quality_assessment_flow():
    """Test complete quality assessment flow"""
    quality_control = QualityControlAutomation()
    
    # Sample process data
    process_data = {
        "clarity_score": 0.9,
        "methodology_score": 0.85,
        "literature_score": 0.8,
        "power_analysis": 0.9,
        "reproducibility": 0.85,
        "testable": True,
        "peer_reviewed": True,
        "comprehensive_literature": True
    }
    
    # Perform assessment
    assessment = await quality_control.assess_quality(
        process_id="integration_test",
        process_type="research",
        process_data=process_data
    )
    
    # Enforce standards
    allowed = await quality_control.enforce_quality_standards("integration_test", assessment)
    
    # Verify results
    assert assessment.process_id == "integration_test"
    assert assessment.overall_score > 0.0
    assert isinstance(allowed, bool)
    assert "integration_test" in quality_control.quality_history
    
    # Test optimization
    optimization_results = await quality_control.optimize_quality_processes()
    assert isinstance(optimization_results, dict)