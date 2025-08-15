"""
Tests for Performance Monitoring Engine

This module tests the real-time team performance tracking, issue identification,
and optimization capabilities during crisis situations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.performance_monitoring_engine import PerformanceMonitoringEngine
from scrollintel.models.performance_monitoring_models import (
    PerformanceStatus, InterventionType, SupportType,
    TeamMemberPerformance, PerformanceIssue, PerformanceIntervention,
    SupportProvision, TeamPerformanceOverview, PerformanceOptimization,
    PerformanceAlert, PerformanceReport
)


@pytest.fixture
def performance_engine():
    """Create performance monitoring engine instance"""
    return PerformanceMonitoringEngine()


@pytest.fixture
def sample_team_members():
    """Sample team members for testing"""
    return ["member_001", "member_002", "member_003"]


@pytest.fixture
def sample_crisis_id():
    """Sample crisis ID for testing"""
    return "crisis_001"


class TestPerformanceMonitoringEngine:
    """Test cases for Performance Monitoring Engine"""
    
    @pytest.mark.asyncio
    async def test_track_team_performance(self, performance_engine, sample_crisis_id, sample_team_members):
        """Test real-time team performance tracking"""
        # Track team performance
        team_overview = await performance_engine.track_team_performance(sample_crisis_id, sample_team_members)
        
        # Verify team overview structure
        assert isinstance(team_overview, TeamPerformanceOverview)
        assert team_overview.crisis_id == sample_crisis_id
        assert team_overview.team_id == f"crisis_team_{sample_crisis_id}"
        assert len(team_overview.member_performances) == len(sample_team_members)
        
        # Verify performance metrics
        assert 0 <= team_overview.overall_performance_score <= 100
        assert 0 <= team_overview.team_efficiency <= 100
        assert 0 <= team_overview.collaboration_index <= 100
        assert 0 <= team_overview.stress_level_avg <= 10
        assert 0 <= team_overview.task_completion_rate <= 100
        assert team_overview.response_time_avg >= 0
        
        # Verify member performances
        for member_performance in team_overview.member_performances:
            assert isinstance(member_performance, TeamMemberPerformance)
            assert member_performance.member_id in sample_team_members
            assert isinstance(member_performance.performance_status, PerformanceStatus)
            assert len(member_performance.metrics) > 0
        
        # Verify data storage
        assert sample_crisis_id in performance_engine.performance_data
        assert performance_engine.performance_data[sample_crisis_id] == team_overview
    
    @pytest.mark.asyncio
    async def test_track_team_performance_empty_members(self, performance_engine, sample_crisis_id):
        """Test tracking performance with empty team members list"""
        with pytest.raises(Exception):
            await performance_engine.track_team_performance(sample_crisis_id, [])
    
    @pytest.mark.asyncio
    async def test_assess_member_performance(self, performance_engine, sample_crisis_id):
        """Test individual member performance assessment"""
        member_id = "test_member"
        
        # Assess member performance
        member_performance = await performance_engine._assess_member_performance(sample_crisis_id, member_id)
        
        # Verify member performance structure
        assert isinstance(member_performance, TeamMemberPerformance)
        assert member_performance.member_id == member_id
        assert member_performance.crisis_id == sample_crisis_id
        assert isinstance(member_performance.performance_status, PerformanceStatus)
        
        # Verify metrics
        assert len(member_performance.metrics) > 0
        assert 0 <= member_performance.overall_score <= 100
        assert 0 <= member_performance.task_completion_rate <= 100
        assert member_performance.response_time_avg >= 0
        assert 0 <= member_performance.quality_score <= 100
        assert 0 <= member_performance.stress_level <= 10
        assert 0 <= member_performance.collaboration_score <= 100
        
        # Verify issues and interventions are lists
        assert isinstance(member_performance.issues_identified, list)
        assert isinstance(member_performance.interventions_needed, list)
    
    @pytest.mark.asyncio
    async def test_identify_performance_issues(self, performance_engine, sample_crisis_id, sample_team_members):
        """Test performance issue identification"""
        # First track team performance
        await performance_engine.track_team_performance(sample_crisis_id, sample_team_members)
        
        # Identify performance issues
        issues = await performance_engine.identify_performance_issues(sample_crisis_id)
        
        # Verify issues structure
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, PerformanceIssue)
            assert issue.crisis_id == sample_crisis_id
            assert issue.member_id in sample_team_members
            assert issue.issue_type in ["RESPONSE_TIME", "STRESS_MANAGEMENT", "QUALITY_CONTROL", "GENERAL_PERFORMANCE"]
            assert issue.severity in ["HIGH", "MEDIUM", "LOW"]
            assert isinstance(issue.identified_at, datetime)
    
    @pytest.mark.asyncio
    async def test_identify_performance_issues_no_data(self, performance_engine):
        """Test issue identification with no performance data"""
        issues = await performance_engine.identify_performance_issues("nonexistent_crisis")
        assert issues == []
    
    @pytest.mark.asyncio
    async def test_implement_intervention(self, performance_engine, sample_crisis_id):
        """Test performance intervention implementation"""
        member_id = "test_member"
        intervention_type = InterventionType.COACHING
        
        # Implement intervention
        intervention = await performance_engine.implement_intervention(sample_crisis_id, member_id, intervention_type)
        
        # Verify intervention structure
        assert isinstance(intervention, PerformanceIntervention)
        assert intervention.member_id == member_id
        assert intervention.crisis_id == sample_crisis_id
        assert intervention.intervention_type == intervention_type
        assert isinstance(intervention.implemented_at, datetime)
        assert intervention.description
        assert intervention.expected_outcome
        assert intervention.completion_status == "in_progress"
        
        # Verify intervention storage
        assert sample_crisis_id in performance_engine.intervention_history
        assert intervention in performance_engine.intervention_history[sample_crisis_id]
    
    @pytest.mark.asyncio
    async def test_provide_support(self, performance_engine, sample_crisis_id):
        """Test support provision to team member"""
        member_id = "test_member"
        support_type = SupportType.TECHNICAL_SUPPORT
        provider = "support_specialist"
        
        # Provide support
        support = await performance_engine.provide_support(sample_crisis_id, member_id, support_type, provider)
        
        # Verify support structure
        assert isinstance(support, SupportProvision)
        assert support.member_id == member_id
        assert support.crisis_id == sample_crisis_id
        assert support.support_type == support_type
        assert support.provider == provider
        assert isinstance(support.provided_at, datetime)
        assert support.description
        
        # Verify support storage
        assert support.support_id in performance_engine.support_provisions
        assert performance_engine.support_provisions[support.support_id] == support
    
    @pytest.mark.asyncio
    async def test_optimize_team_performance(self, performance_engine, sample_crisis_id, sample_team_members):
        """Test team performance optimization"""
        # First track team performance
        await performance_engine.track_team_performance(sample_crisis_id, sample_team_members)
        
        # Generate optimizations
        optimizations = await performance_engine.optimize_team_performance(sample_crisis_id)
        
        # Verify optimizations structure
        assert isinstance(optimizations, list)
        for optimization in optimizations:
            assert isinstance(optimization, PerformanceOptimization)
            assert optimization.crisis_id == sample_crisis_id
            assert optimization.target_area
            assert optimization.current_performance >= 0
            assert optimization.target_performance >= 0
            assert optimization.optimization_strategy
            assert isinstance(optimization.implementation_steps, list)
            assert optimization.priority_level in ["HIGH", "MEDIUM", "LOW"]
            assert optimization.estimated_completion_time > 0
            assert isinstance(optimization.resources_required, list)
    
    @pytest.mark.asyncio
    async def test_optimize_team_performance_no_data(self, performance_engine):
        """Test optimization with no performance data"""
        optimizations = await performance_engine.optimize_team_performance("nonexistent_crisis")
        assert optimizations == []
    
    @pytest.mark.asyncio
    async def test_generate_performance_alerts(self, performance_engine, sample_crisis_id, sample_team_members):
        """Test performance alert generation"""
        # First track team performance
        await performance_engine.track_team_performance(sample_crisis_id, sample_team_members)
        
        # Generate alerts
        alerts = await performance_engine.generate_performance_alerts(sample_crisis_id)
        
        # Verify alerts structure
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, PerformanceAlert)
            assert alert.crisis_id == sample_crisis_id
            assert alert.alert_type in ["CRITICAL_PERFORMANCE", "HIGH_STRESS"]
            assert alert.severity in ["HIGH", "MEDIUM", "LOW"]
            assert isinstance(alert.triggered_at, datetime)
            assert alert.message
        
        # Verify alert storage
        if alerts:
            assert sample_crisis_id in performance_engine.active_alerts
            for alert in alerts:
                assert alert in performance_engine.active_alerts[sample_crisis_id]
    
    @pytest.mark.asyncio
    async def test_generate_performance_alerts_no_data(self, performance_engine):
        """Test alert generation with no performance data"""
        alerts = await performance_engine.generate_performance_alerts("nonexistent_crisis")
        assert alerts == []
    
    @pytest.mark.asyncio
    async def test_generate_performance_report(self, performance_engine, sample_crisis_id, sample_team_members):
        """Test comprehensive performance report generation"""
        # First track team performance
        await performance_engine.track_team_performance(sample_crisis_id, sample_team_members)
        
        # Generate report
        report = await performance_engine.generate_performance_report(sample_crisis_id, 24)
        
        # Verify report structure
        assert isinstance(report, PerformanceReport)
        assert report.crisis_id == sample_crisis_id
        assert report.report_type == "COMPREHENSIVE_PERFORMANCE"
        assert isinstance(report.generated_at, datetime)
        assert isinstance(report.time_period_start, datetime)
        assert isinstance(report.time_period_end, datetime)
        assert isinstance(report.team_overview, TeamPerformanceOverview)
        assert isinstance(report.key_insights, list)
        assert isinstance(report.performance_trends, dict)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.success_metrics, dict)
        
        # Verify time period
        time_diff = report.time_period_end - report.time_period_start
        assert abs(time_diff.total_seconds() - 24 * 3600) < 60  # Within 1 minute tolerance
    
    @pytest.mark.asyncio
    async def test_generate_performance_report_no_data(self, performance_engine):
        """Test report generation with no performance data"""
        with pytest.raises(ValueError, match="No performance data found"):
            await performance_engine.generate_performance_report("nonexistent_crisis", 24)
    
    @pytest.mark.asyncio
    async def test_calculate_team_efficiency(self, performance_engine):
        """Test team efficiency calculation"""
        # Create sample member performances
        member_performances = [
            Mock(task_completion_rate=85.0, response_time_avg=2.0),
            Mock(task_completion_rate=90.0, response_time_avg=1.5),
            Mock(task_completion_rate=80.0, response_time_avg=3.0)
        ]
        
        efficiency = await performance_engine._calculate_team_efficiency(member_performances)
        
        assert 0 <= efficiency <= 100
        assert isinstance(efficiency, float)
    
    @pytest.mark.asyncio
    async def test_calculate_team_efficiency_empty(self, performance_engine):
        """Test team efficiency calculation with empty list"""
        efficiency = await performance_engine._calculate_team_efficiency([])
        assert efficiency == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_collaboration_index(self, performance_engine):
        """Test collaboration index calculation"""
        # Create sample member performances
        member_performances = [
            Mock(collaboration_score=85.0),
            Mock(collaboration_score=90.0),
            Mock(collaboration_score=88.0)
        ]
        
        collaboration_index = await performance_engine._calculate_collaboration_index(member_performances)
        
        assert 0 <= collaboration_index <= 100
        assert isinstance(collaboration_index, float)
        assert collaboration_index == (85.0 + 90.0 + 88.0) / 3
    
    @pytest.mark.asyncio
    async def test_calculate_collaboration_index_empty(self, performance_engine):
        """Test collaboration index calculation with empty list"""
        collaboration_index = await performance_engine._calculate_collaboration_index([])
        assert collaboration_index == 0.0
    
    def test_categorize_issue(self, performance_engine):
        """Test issue categorization"""
        assert performance_engine._categorize_issue("Slow response time") == "RESPONSE_TIME"
        assert performance_engine._categorize_issue("High stress level") == "STRESS_MANAGEMENT"
        assert performance_engine._categorize_issue("Quality concerns") == "QUALITY_CONTROL"
        assert performance_engine._categorize_issue("General issue") == "GENERAL_PERFORMANCE"
    
    def test_assess_issue_severity(self, performance_engine):
        """Test issue severity assessment"""
        assert performance_engine._assess_issue_severity(PerformanceStatus.CRITICAL) == "HIGH"
        assert performance_engine._assess_issue_severity(PerformanceStatus.BELOW_AVERAGE) == "MEDIUM"
        assert performance_engine._assess_issue_severity(PerformanceStatus.GOOD) == "LOW"
    
    @pytest.mark.asyncio
    async def test_assess_issue_impact(self, performance_engine):
        """Test issue impact assessment"""
        member_performance = Mock(
            performance_status=PerformanceStatus.CRITICAL,
            member_name="Test Member"
        )
        
        impact = await performance_engine._assess_issue_impact("test issue", member_performance)
        
        assert isinstance(impact, str)
        assert len(impact) > 0
        assert "Test Member" in impact
    
    @pytest.mark.asyncio
    async def test_generate_intervention_description(self, performance_engine):
        """Test intervention description generation"""
        description = await performance_engine._generate_intervention_description(InterventionType.COACHING)
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "coaching" in description.lower()
    
    @pytest.mark.asyncio
    async def test_define_expected_outcome(self, performance_engine):
        """Test expected outcome definition"""
        outcome = await performance_engine._define_expected_outcome(InterventionType.TRAINING)
        
        assert isinstance(outcome, str)
        assert len(outcome) > 0
    
    @pytest.mark.asyncio
    async def test_generate_support_description(self, performance_engine):
        """Test support description generation"""
        description = await performance_engine._generate_support_description(SupportType.TECHNICAL_SUPPORT)
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "technical" in description.lower()
    
    @pytest.mark.asyncio
    async def test_generate_performance_insights(self, performance_engine):
        """Test performance insights generation"""
        team_overview = Mock(
            overall_performance_score=85.0,
            stress_level_avg=6.0,
            collaboration_index=92.0,
            critical_issues_count=1
        )
        
        insights = await performance_engine._generate_performance_insights(team_overview)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_performance_trends(self, performance_engine, sample_crisis_id):
        """Test performance trends analysis"""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        trends = await performance_engine._analyze_performance_trends(sample_crisis_id, start_time, end_time)
        
        assert isinstance(trends, dict)
        assert "performance_trend" in trends
        assert "efficiency_trend" in trends
        assert "stress_trend" in trends
        assert "collaboration_trend" in trends
        assert "response_time_trend" in trends
    
    @pytest.mark.asyncio
    async def test_multiple_interventions_same_member(self, performance_engine, sample_crisis_id):
        """Test multiple interventions for same member"""
        member_id = "test_member"
        
        # Implement multiple interventions
        intervention1 = await performance_engine.implement_intervention(
            sample_crisis_id, member_id, InterventionType.COACHING
        )
        intervention2 = await performance_engine.implement_intervention(
            sample_crisis_id, member_id, InterventionType.TRAINING
        )
        
        # Verify both interventions are stored
        assert sample_crisis_id in performance_engine.intervention_history
        interventions = performance_engine.intervention_history[sample_crisis_id]
        assert len(interventions) == 2
        assert intervention1 in interventions
        assert intervention2 in interventions
    
    @pytest.mark.asyncio
    async def test_multiple_support_provisions(self, performance_engine, sample_crisis_id):
        """Test multiple support provisions for same member"""
        member_id = "test_member"
        provider = "support_specialist"
        
        # Provide multiple types of support
        support1 = await performance_engine.provide_support(
            sample_crisis_id, member_id, SupportType.TECHNICAL_SUPPORT, provider
        )
        support2 = await performance_engine.provide_support(
            sample_crisis_id, member_id, SupportType.EMOTIONAL_SUPPORT, provider
        )
        
        # Verify both support provisions are stored
        assert support1.support_id in performance_engine.support_provisions
        assert support2.support_id in performance_engine.support_provisions
        assert support1.support_id != support2.support_id
    
    @pytest.mark.asyncio
    async def test_performance_tracking_consistency(self, performance_engine, sample_crisis_id, sample_team_members):
        """Test consistency of performance tracking over multiple calls"""
        # Track performance multiple times
        overview1 = await performance_engine.track_team_performance(sample_crisis_id, sample_team_members)
        overview2 = await performance_engine.track_team_performance(sample_crisis_id, sample_team_members)
        
        # Verify structure consistency
        assert len(overview1.member_performances) == len(overview2.member_performances)
        assert overview1.crisis_id == overview2.crisis_id
        assert overview1.team_id == overview2.team_id
        
        # Verify data is updated (timestamps should be different)
        assert overview1.last_updated != overview2.last_updated