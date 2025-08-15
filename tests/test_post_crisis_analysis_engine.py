"""
Tests for Post-Crisis Analysis Engine

Test comprehensive crisis response analysis and evaluation functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.post_crisis_analysis_engine import PostCrisisAnalysisEngine
from scrollintel.models.post_crisis_analysis_models import (
    PostCrisisAnalysis, AnalysisType, LessonLearned, LessonCategory,
    ImprovementRecommendation, RecommendationPriority, CrisisMetric
)
from scrollintel.models.crisis_detection_models import Crisis, CrisisType, SeverityLevel, CrisisStatus


class TestPostCrisisAnalysisEngine:
    """Test cases for PostCrisisAnalysisEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        return PostCrisisAnalysisEngine()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis for testing"""
        return Crisis(
            id="crisis_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now() - timedelta(hours=2),
            affected_areas=["production", "api"],
            stakeholders_impacted=["customers", "partners"],
            current_status=CrisisStatus.RESOLVED,
            response_actions=["emergency_response", "communication_sent"],
            resolution_time=datetime.now()
        )
    
    @pytest.fixture
    def sample_response_data(self):
        """Create sample response data for testing"""
        return {
            "response_time": 25.0,
            "communication_score": 85.0,
            "resource_efficiency": 78.0,
            "stakeholder_satisfaction": 82,
            "team_coordination_score": 88,
            "communication_delays": 0,
            "resource_shortages": 1,
            "customer_impact": "Medium",
            "employee_impact": "Low",
            "investor_impact": "Medium",
            "revenue_impact": 50000,
            "operational_disruption": "Medium",
            "recovery_time": 120,
            "response_cost": 25000,
            "media_sentiment": "Neutral",
            "social_impact": "Low",
            "brand_impact": -5
        }
    
    def test_conduct_comprehensive_analysis(self, engine, sample_crisis, sample_response_data):
        """Test comprehensive crisis analysis"""
        analyst_id = "analyst_001"
        
        analysis = engine.conduct_comprehensive_analysis(
            crisis=sample_crisis,
            response_data=sample_response_data,
            analyst_id=analyst_id
        )
        
        assert isinstance(analysis, PostCrisisAnalysis)
        assert analysis.crisis_id == sample_crisis.id
        assert analysis.analyst_id == analyst_id
        assert analysis.analysis_type == AnalysisType.RESPONSE_EFFECTIVENESS
        assert len(analysis.response_metrics) > 0
        assert analysis.overall_performance_score > 0
        assert len(analysis.strengths_identified) >= 0
        assert len(analysis.weaknesses_identified) >= 0
        assert analysis.confidence_level > 0
    
    def test_identify_lessons_learned(self, engine, sample_crisis, sample_response_data):
        """Test lessons learned identification"""
        lessons = engine.identify_lessons_learned(
            crisis=sample_crisis,
            response_data=sample_response_data
        )
        
        assert isinstance(lessons, list)
        for lesson in lessons:
            assert isinstance(lesson, LessonLearned)
            assert lesson.crisis_id == sample_crisis.id
            assert lesson.category in LessonCategory
            assert len(lesson.title) > 0
            assert len(lesson.description) > 0
    
    def test_generate_improvement_recommendations(self, engine):
        """Test improvement recommendations generation"""
        # Create sample lessons
        lessons = [
            LessonLearned(
                id="lesson_001",
                crisis_id="crisis_001",
                category=LessonCategory.COMMUNICATION,
                title="Communication delay identified",
                description="Stakeholder notification was delayed by 15 minutes",
                root_cause="Manual notification process",
                impact_assessment="Medium impact on stakeholder confidence",
                evidence=["notification_logs", "stakeholder_feedback"],
                identified_by="analyst_001",
                identification_date=datetime.now(),
                validation_status="pending"
            )
        ]
        
        recommendations = engine.generate_improvement_recommendations(lessons)
        
        assert isinstance(recommendations, list)
        for recommendation in recommendations:
            assert isinstance(recommendation, ImprovementRecommendation)
            assert recommendation.lesson_id in [lesson.id for lesson in lessons]
            assert recommendation.priority in RecommendationPriority
            assert len(recommendation.title) > 0
            assert len(recommendation.description) > 0
    
    def test_generate_analysis_report(self, engine, sample_crisis, sample_response_data):
        """Test analysis report generation"""
        # First conduct analysis
        analysis = engine.conduct_comprehensive_analysis(
            crisis=sample_crisis,
            response_data=sample_response_data,
            analyst_id="analyst_001"
        )
        
        # Generate report
        report = engine.generate_analysis_report(analysis)
        
        assert report.analysis_id == analysis.id
        assert len(report.report_title) > 0
        assert len(report.executive_summary) > 0
        assert len(report.detailed_findings) > 0
        assert len(report.recommendations_summary) > 0
        assert len(report.appendices) > 0
        assert report.report_format == "comprehensive"
    
    def test_calculate_response_metrics(self, engine, sample_crisis, sample_response_data):
        """Test response metrics calculation"""
        metrics = engine._calculate_response_metrics(sample_crisis, sample_response_data)
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        for metric in metrics:
            assert isinstance(metric, CrisisMetric)
            assert len(metric.metric_name) > 0
            assert metric.target_value > 0
            assert metric.actual_value >= 0
            assert len(metric.measurement_unit) > 0
    
    def test_calculate_overall_performance_score(self, engine):
        """Test overall performance score calculation"""
        metrics = [
            CrisisMetric("Response Time", 30.0, 25.0, -5.0, 90.0, "minutes"),
            CrisisMetric("Communication", 90.0, 85.0, -5.0, 85.0, "percentage"),
            CrisisMetric("Resource Efficiency", 85.0, 78.0, -7.0, 78.0, "percentage")
        ]
        
        score = engine._calculate_overall_performance_score(metrics)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score == (90.0 + 85.0 + 78.0) / 3
    
    def test_identify_strengths(self, engine, sample_crisis, sample_response_data):
        """Test strengths identification"""
        metrics = [
            CrisisMetric("Response Time", 30.0, 25.0, -5.0, 90.0, "minutes"),
            CrisisMetric("Communication", 90.0, 85.0, -5.0, 85.0, "percentage")
        ]
        
        strengths = engine._identify_strengths(sample_crisis, sample_response_data, metrics)
        
        assert isinstance(strengths, list)
        # Should identify high-performing metrics as strengths
        assert any("response time" in strength.lower() for strength in strengths)
        assert any("communication" in strength.lower() for strength in strengths)
    
    def test_identify_weaknesses(self, engine, sample_crisis, sample_response_data):
        """Test weaknesses identification"""
        # Create metrics with poor performance
        metrics = [
            CrisisMetric("Response Time", 30.0, 60.0, 30.0, 40.0, "minutes"),
            CrisisMetric("Communication", 90.0, 50.0, -40.0, 50.0, "percentage")
        ]
        
        weaknesses = engine._identify_weaknesses(sample_crisis, sample_response_data, metrics)
        
        assert isinstance(weaknesses, list)
        # Should identify low-performing metrics as weaknesses
        assert any("response time" in weakness.lower() for weakness in weaknesses)
        assert any("communication" in weakness.lower() for weakness in weaknesses)
    
    def test_assess_stakeholder_impact(self, engine, sample_crisis, sample_response_data):
        """Test stakeholder impact assessment"""
        impact = engine._assess_stakeholder_impact(sample_crisis, sample_response_data)
        
        assert isinstance(impact, dict)
        assert "customers" in impact
        assert "employees" in impact
        assert "investors" in impact
        assert "partners" in impact
        assert "overall_satisfaction" in impact
        assert isinstance(impact["overall_satisfaction"], (int, float))
    
    def test_assess_business_impact(self, engine, sample_crisis, sample_response_data):
        """Test business impact assessment"""
        impact = engine._assess_business_impact(sample_crisis, sample_response_data)
        
        assert isinstance(impact, dict)
        assert "revenue_impact" in impact
        assert "operational_disruption" in impact
        assert "recovery_time" in impact
        assert "cost_of_response" in impact
    
    def test_assess_reputation_impact(self, engine, sample_crisis, sample_response_data):
        """Test reputation impact assessment"""
        impact = engine._assess_reputation_impact(sample_crisis, sample_response_data)
        
        assert isinstance(impact, dict)
        assert "media_sentiment" in impact
        assert "social_media_impact" in impact
        assert "brand_perception_change" in impact
        assert "recovery_timeline" in impact
    
    def test_generate_crisis_summary(self, engine, sample_crisis):
        """Test crisis summary generation"""
        summary = engine._generate_crisis_summary(sample_crisis)
        
        assert isinstance(summary, str)
        assert sample_crisis.id in summary
        assert sample_crisis.crisis_type.value in summary
        assert sample_crisis.severity_level.value in summary
    
    def test_calculate_crisis_duration(self, engine, sample_crisis):
        """Test crisis duration calculation"""
        duration = engine._calculate_crisis_duration(sample_crisis)
        
        assert isinstance(duration, float)
        assert duration > 0  # Should be positive for resolved crisis
    
    def test_calculate_confidence_level(self, engine, sample_response_data):
        """Test confidence level calculation"""
        confidence = engine._calculate_confidence_level(sample_response_data)
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100
    
    def test_empty_metrics_handling(self, engine):
        """Test handling of empty metrics list"""
        score = engine._calculate_overall_performance_score([])
        assert score == 0.0
    
    def test_missing_response_data(self, engine, sample_crisis):
        """Test handling of missing response data"""
        empty_data = {}
        
        analysis = engine.conduct_comprehensive_analysis(
            crisis=sample_crisis,
            response_data=empty_data,
            analyst_id="analyst_001"
        )
        
        assert isinstance(analysis, PostCrisisAnalysis)
        assert analysis.overall_performance_score >= 0
        assert analysis.confidence_level >= 0
    
    @patch('scrollintel.engines.post_crisis_analysis_engine.logging')
    def test_error_handling(self, mock_logging, engine):
        """Test error handling in analysis engine"""
        # Test with invalid crisis data
        with pytest.raises(Exception):
            engine.conduct_comprehensive_analysis(
                crisis=None,
                response_data={},
                analyst_id="analyst_001"
            )
    
    def test_analysis_templates_initialization(self, engine):
        """Test analysis templates initialization"""
        assert hasattr(engine, 'analysis_templates')
        assert isinstance(engine.analysis_templates, dict)
        assert len(engine.analysis_templates) > 0
    
    def test_metric_calculators_initialization(self, engine):
        """Test metric calculators initialization"""
        assert hasattr(engine, 'metric_calculators')
        assert isinstance(engine.metric_calculators, dict)
        assert len(engine.metric_calculators) > 0