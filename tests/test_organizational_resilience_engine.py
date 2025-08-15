"""
Tests for Organizational Resilience Engine

Comprehensive tests for organizational resilience assessment, enhancement, and monitoring.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.organizational_resilience_engine import OrganizationalResilienceEngine
from scrollintel.models.organizational_resilience_models import (
    ResilienceAssessment, ResilienceStrategy, ResilienceMonitoringData,
    ResilienceImprovement, ResilienceReport, ResilienceLevel,
    ResilienceCategory, ResilienceMetricType
)


@pytest.fixture
def resilience_engine():
    """Create resilience engine instance for testing"""
    return OrganizationalResilienceEngine()


@pytest.fixture
def sample_assessment():
    """Create sample resilience assessment for testing"""
    return ResilienceAssessment(
        id="test_assessment_001",
        organization_id="test_org_001",
        assessment_date=datetime.now(),
        overall_resilience_level=ResilienceLevel.ROBUST,
        category_scores={
            ResilienceCategory.OPERATIONAL: 0.75,
            ResilienceCategory.FINANCIAL: 0.80,
            ResilienceCategory.TECHNOLOGICAL: 0.70,
            ResilienceCategory.HUMAN_CAPITAL: 0.85
        },
        strengths=["Strong financial position", "Excellent team"],
        vulnerabilities=["Technology gaps", "Process inefficiencies"],
        improvement_areas=["Technology modernization", "Process optimization"],
        assessment_methodology="comprehensive_analysis",
        confidence_score=0.85
    )


@pytest.fixture
def sample_monitoring_data():
    """Create sample monitoring data for testing"""
    return [
        ResilienceMonitoringData(
            id="monitoring_001",
            monitoring_date=datetime.now(),
            category=ResilienceCategory.OPERATIONAL,
            metric_values={
                ResilienceMetricType.RECOVERY_TIME: 0.75,
                ResilienceMetricType.ADAPTATION_SPEED: 0.70,
                ResilienceMetricType.STRESS_TOLERANCE: 0.80
            },
            alert_triggers=["Low adaptation speed"],
            trend_analysis={"operational": {"trend": "stable"}},
            anomaly_detection=[],
            recommendations=["Focus on adaptation capabilities"]
        )
    ]


class TestOrganizationalResilienceEngine:
    """Test cases for organizational resilience engine"""
    
    @pytest.mark.asyncio
    async def test_assess_organizational_resilience(self, resilience_engine):
        """Test organizational resilience assessment"""
        # Test assessment
        assessment = await resilience_engine.assess_organizational_resilience(
            organization_id="test_org_001"
        )
        
        # Verify assessment structure
        assert assessment.id is not None
        assert assessment.organization_id == "test_org_001"
        assert isinstance(assessment.overall_resilience_level, ResilienceLevel)
        assert len(assessment.category_scores) > 0
        assert isinstance(assessment.strengths, list)
        assert isinstance(assessment.vulnerabilities, list)
        assert isinstance(assessment.improvement_areas, list)
        assert 0 <= assessment.confidence_score <= 1
    
    @pytest.mark.asyncio
    async def test_assess_resilience_with_scope(self, resilience_engine):
        """Test resilience assessment with specific scope"""
        scope = [ResilienceCategory.OPERATIONAL, ResilienceCategory.FINANCIAL]
        
        assessment = await resilience_engine.assess_organizational_resilience(
            organization_id="test_org_001",
            assessment_scope=scope
        )
        
        # Verify scope is respected
        assert len(assessment.category_scores) == len(scope)
        for category in scope:
            assert category in assessment.category_scores
    
    @pytest.mark.asyncio
    async def test_develop_resilience_strategy(self, resilience_engine, sample_assessment):
        """Test resilience strategy development"""
        strategy = await resilience_engine.develop_resilience_strategy(
            assessment=sample_assessment
        )
        
        # Verify strategy structure
        assert strategy.id is not None
        assert strategy.strategy_name is not None
        assert len(strategy.target_categories) > 0
        assert len(strategy.objectives) > 0
        assert len(strategy.initiatives) > 0
        assert isinstance(strategy.timeline, dict)
        assert isinstance(strategy.resource_requirements, dict)
        assert len(strategy.success_metrics) > 0
        assert len(strategy.risk_factors) > 0
        assert isinstance(strategy.expected_impact, dict)
    
    @pytest.mark.asyncio
    async def test_develop_strategy_with_priorities(self, resilience_engine, sample_assessment):
        """Test strategy development with strategic priorities"""
        priorities = ["operational_excellence", "technology_modernization"]
        
        strategy = await resilience_engine.develop_resilience_strategy(
            assessment=sample_assessment,
            strategic_priorities=priorities
        )
        
        # Verify priorities influence strategy
        assert len(strategy.target_categories) > 0
        assert len(strategy.objectives) > 0
    
    @pytest.mark.asyncio
    async def test_monitor_resilience_continuously(self, resilience_engine):
        """Test continuous resilience monitoring"""
        monitoring_data = await resilience_engine.monitor_resilience_continuously(
            organization_id="test_org_001"
        )
        
        # Verify monitoring data structure
        assert monitoring_data.id is not None
        assert monitoring_data.monitoring_date is not None
        assert len(monitoring_data.metric_values) > 0
        assert isinstance(monitoring_data.alert_triggers, list)
        assert isinstance(monitoring_data.trend_analysis, dict)
        assert isinstance(monitoring_data.anomaly_detection, list)
        assert isinstance(monitoring_data.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_monitor_with_frequency(self, resilience_engine):
        """Test monitoring with different frequencies"""
        frequencies = ["daily", "weekly", "monthly"]
        
        for frequency in frequencies:
            monitoring_data = await resilience_engine.monitor_resilience_continuously(
                organization_id="test_org_001",
                monitoring_frequency=frequency
            )
            
            assert monitoring_data is not None
            assert len(monitoring_data.metric_values) > 0
    
    @pytest.mark.asyncio
    async def test_implement_continuous_improvement(self, resilience_engine, sample_monitoring_data):
        """Test continuous improvement implementation"""
        improvements = await resilience_engine.implement_continuous_improvement(
            organization_id="test_org_001",
            monitoring_data=sample_monitoring_data
        )
        
        # Verify improvements structure
        assert isinstance(improvements, list)
        for improvement in improvements:
            assert improvement.id is not None
            assert improvement.improvement_type is not None
            assert isinstance(improvement.category, ResilienceCategory)
            assert improvement.priority is not None
            assert improvement.description is not None
            assert isinstance(improvement.implementation_steps, list)
            assert improvement.estimated_timeline is not None
            assert isinstance(improvement.resource_requirements, dict)
            assert isinstance(improvement.expected_benefits, list)
            assert isinstance(improvement.success_metrics, list)
    
    @pytest.mark.asyncio
    async def test_improvement_with_cycle(self, resilience_engine, sample_monitoring_data):
        """Test improvement with different cycles"""
        cycles = ["monthly", "quarterly", "annually"]
        
        for cycle in cycles:
            improvements = await resilience_engine.implement_continuous_improvement(
                organization_id="test_org_001",
                monitoring_data=sample_monitoring_data,
                improvement_cycle=cycle
            )
            
            assert isinstance(improvements, list)
    
    @pytest.mark.asyncio
    async def test_generate_resilience_report(
        self, 
        resilience_engine, 
        sample_assessment, 
        sample_monitoring_data
    ):
        """Test resilience report generation"""
        # Create sample improvements
        improvements = [
            ResilienceImprovement(
                id="improvement_001",
                assessment_id=sample_assessment.id,
                improvement_type="Technology Upgrade",
                category=ResilienceCategory.TECHNOLOGICAL,
                priority="high",
                description="Modernize technology infrastructure",
                implementation_steps=["Assess", "Plan", "Execute"],
                estimated_timeline="6 months",
                resource_requirements={"budget": 100000},
                expected_benefits=["Better performance"],
                success_metrics=["System uptime"],
                dependencies=[]
            )
        ]
        
        report = await resilience_engine.generate_resilience_report(
            organization_id="test_org_001",
            assessment=sample_assessment,
            monitoring_data=sample_monitoring_data,
            improvements=improvements
        )
        
        # Verify report structure
        assert report.id is not None
        assert report.report_date is not None
        assert report.organization_id == "test_org_001"
        assert report.executive_summary is not None
        assert 0 <= report.overall_resilience_score <= 1
        assert isinstance(report.category_breakdown, dict)
        assert isinstance(report.trend_analysis, dict)
        assert isinstance(report.benchmark_comparison, dict)
        assert isinstance(report.key_findings, list)
        assert len(report.recommendations) > 0
        assert isinstance(report.action_plan, list)
        assert report.next_assessment_date > report.report_date
    
    @pytest.mark.asyncio
    async def test_resilience_level_determination(self, resilience_engine):
        """Test resilience level determination logic"""
        # Test different score ranges
        test_cases = [
            (0.90, ResilienceLevel.ANTIFRAGILE),
            (0.75, ResilienceLevel.ROBUST),
            (0.60, ResilienceLevel.BASIC),
            (0.40, ResilienceLevel.FRAGILE)
        ]
        
        for score, expected_level in test_cases:
            level = resilience_engine._determine_resilience_level(score)
            assert level == expected_level
    
    @pytest.mark.asyncio
    async def test_category_prioritization(self, resilience_engine, sample_assessment):
        """Test resilience category prioritization"""
        categories = await resilience_engine._prioritize_resilience_categories(
            sample_assessment
        )
        
        # Verify prioritization logic
        assert isinstance(categories, list)
        assert len(categories) <= len(ResilienceCategory)
        
        # Should prioritize lowest scoring categories first
        if len(categories) > 1:
            scores = [sample_assessment.category_scores[cat] for cat in categories]
            assert scores == sorted(scores)  # Should be in ascending order
    
    @pytest.mark.asyncio
    async def test_metric_measurement(self, resilience_engine):
        """Test resilience metric measurement"""
        for metric_type in ResilienceMetricType:
            value = await resilience_engine._measure_resilience_metric(
                "test_org_001", metric_type
            )
            
            assert isinstance(value, float)
            assert 0 <= value <= 1
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, resilience_engine):
        """Test resilience anomaly detection"""
        # Test with extreme values
        extreme_metrics = {
            ResilienceMetricType.RECOVERY_TIME: 0.1,  # Very low
            ResilienceMetricType.ADAPTATION_SPEED: 0.99,  # Very high
            ResilienceMetricType.STRESS_TOLERANCE: 0.5   # Normal
        }
        
        anomalies = await resilience_engine._detect_resilience_anomalies(
            "test_org_001", extreme_metrics, {}
        )
        
        assert isinstance(anomalies, list)
        # Should detect both low and high anomalies
        assert len(anomalies) >= 2
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, resilience_engine):
        """Test resilience alert generation"""
        # Test with low metrics
        low_metrics = {
            ResilienceMetricType.RECOVERY_TIME: 0.3,
            ResilienceMetricType.ADAPTATION_SPEED: 0.2,
            ResilienceMetricType.STRESS_TOLERANCE: 0.6
        }
        
        alerts = await resilience_engine._generate_resilience_alerts(
            low_metrics, []
        )
        
        assert isinstance(alerts, list)
        assert len(alerts) > 0  # Should generate alerts for low values
    
    @pytest.mark.asyncio
    async def test_improvement_feasibility_validation(self, resilience_engine):
        """Test improvement feasibility validation"""
        improvements = [
            ResilienceImprovement(
                id="imp_1",
                assessment_id="test",
                improvement_type="High Priority",
                category=ResilienceCategory.OPERATIONAL,
                priority="high",
                description="High priority improvement",
                implementation_steps=[],
                estimated_timeline="",
                resource_requirements={},
                expected_benefits=[],
                success_metrics=[],
                dependencies=[]
            ),
            ResilienceImprovement(
                id="imp_2",
                assessment_id="test",
                improvement_type="Low Priority",
                category=ResilienceCategory.OPERATIONAL,
                priority="low",
                description="Low priority improvement",
                implementation_steps=[],
                estimated_timeline="",
                resource_requirements={},
                expected_benefits=[],
                success_metrics=[],
                dependencies=[]
            )
        ]
        
        validated = await resilience_engine._validate_improvement_feasibility(
            improvements, "test_org_001"
        )
        
        # Should filter out low priority improvements
        assert len(validated) < len(improvements)
        assert all(imp.priority in ["high", "medium"] for imp in validated)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, resilience_engine):
        """Test error handling in resilience operations"""
        # Test with invalid organization ID
        with patch.object(resilience_engine, '_collect_assessment_data', 
                         side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                await resilience_engine.assess_organizational_resilience("invalid_org")
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, resilience_engine):
        """Test complete resilience management workflow"""
        organization_id = "test_org_workflow"
        
        # Step 1: Assess resilience
        assessment = await resilience_engine.assess_organizational_resilience(
            organization_id
        )
        assert assessment is not None
        
        # Step 2: Develop strategy
        strategy = await resilience_engine.develop_resilience_strategy(
            assessment
        )
        assert strategy is not None
        
        # Step 3: Monitor resilience
        monitoring_data = await resilience_engine.monitor_resilience_continuously(
            organization_id
        )
        assert monitoring_data is not None
        
        # Step 4: Implement improvements
        improvements = await resilience_engine.implement_continuous_improvement(
            organization_id, [monitoring_data]
        )
        assert improvements is not None
        
        # Step 5: Generate report
        report = await resilience_engine.generate_resilience_report(
            organization_id, assessment, [monitoring_data], improvements
        )
        assert report is not None
        
        # Verify workflow coherence
        assert report.organization_id == organization_id
        assert len(report.recommendations) > 0
        assert len(report.action_plan) > 0


if __name__ == "__main__":
    pytest.main([__file__])