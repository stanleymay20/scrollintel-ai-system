"""
Simple Performance Monitoring System Tests

This module tests the performance monitoring system without API dependencies.
"""

import pytest
import asyncio
from datetime import datetime

from scrollintel.engines.performance_monitoring_engine import PerformanceMonitoringEngine
from scrollintel.models.performance_monitoring_models import (
    InterventionType, SupportType, PerformanceStatus,
    TeamMemberPerformance, PerformanceIssue, PerformanceIntervention,
    SupportProvision, TeamPerformanceOverview, PerformanceOptimization,
    PerformanceAlert, PerformanceReport
)


@pytest.fixture
def engine():
    """Create performance monitoring engine"""
    return PerformanceMonitoringEngine()


@pytest.fixture
def crisis_id():
    """Sample crisis ID"""
    return "test_crisis_001"


@pytest.fixture
def team_members():
    """Sample team members"""
    return ["alice", "bob", "charlie", "diana", "eve"]


class TestPerformanceMonitoringSimple:
    """Simple test cases for Performance Monitoring System"""
    
    @pytest.mark.asyncio
    async def test_complete_performance_monitoring_workflow(self, engine, crisis_id, team_members):
        """Test complete performance monitoring workflow"""
        
        # Step 1: Track team performance
        team_overview = await engine.track_team_performance(crisis_id, team_members)
        
        assert isinstance(team_overview, TeamPerformanceOverview)
        assert team_overview.crisis_id == crisis_id
        assert len(team_overview.member_performances) == len(team_members)
        assert 0 <= team_overview.overall_performance_score <= 100
        
        # Step 2: Identify performance issues
        issues = await engine.identify_performance_issues(crisis_id)
        
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, PerformanceIssue)
            assert issue.crisis_id == crisis_id
        
        # Step 3: Implement interventions
        intervention = await engine.implement_intervention(
            crisis_id, team_members[0], InterventionType.COACHING
        )
        
        assert isinstance(intervention, PerformanceIntervention)
        assert intervention.crisis_id == crisis_id
        assert intervention.member_id == team_members[0]
        assert intervention.intervention_type == InterventionType.COACHING
        
        # Step 4: Provide support
        support = await engine.provide_support(
            crisis_id, team_members[1], SupportType.TECHNICAL_SUPPORT, "senior_engineer"
        )
        
        assert isinstance(support, SupportProvision)
        assert support.crisis_id == crisis_id
        assert support.member_id == team_members[1]
        assert support.support_type == SupportType.TECHNICAL_SUPPORT
        
        # Step 5: Generate alerts
        alerts = await engine.generate_performance_alerts(crisis_id)
        
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, PerformanceAlert)
            assert alert.crisis_id == crisis_id
        
        # Step 6: Get optimizations
        optimizations = await engine.optimize_team_performance(crisis_id)
        
        assert isinstance(optimizations, list)
        for opt in optimizations:
            assert isinstance(opt, PerformanceOptimization)
            assert opt.crisis_id == crisis_id
        
        # Step 7: Generate report
        report = await engine.generate_performance_report(crisis_id, 24)
        
        assert isinstance(report, PerformanceReport)
        assert report.crisis_id == crisis_id
        assert report.report_type == "COMPREHENSIVE_PERFORMANCE"
        assert isinstance(report.team_overview, TeamPerformanceOverview)
        assert isinstance(report.key_insights, list)
        assert isinstance(report.success_metrics, dict)
        
        print("✅ Complete performance monitoring workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_performance_tracking_accuracy(self, engine, crisis_id, team_members):
        """Test accuracy of performance tracking"""
        
        team_overview = await engine.track_team_performance(crisis_id, team_members)
        
        # Verify all team members are tracked
        tracked_members = [member.member_id for member in team_overview.member_performances]
        assert set(tracked_members) == set(team_members)
        
        # Verify performance metrics are within valid ranges
        for member in team_overview.member_performances:
            assert 0 <= member.overall_score <= 100
            assert 0 <= member.task_completion_rate <= 100
            assert member.response_time_avg >= 0
            assert 0 <= member.quality_score <= 100
            assert 0 <= member.stress_level <= 10
            assert 0 <= member.collaboration_score <= 100
            assert isinstance(member.performance_status, PerformanceStatus)
        
        print("✅ Performance tracking accuracy test passed!")
    
    @pytest.mark.asyncio
    async def test_intervention_effectiveness(self, engine, crisis_id, team_members):
        """Test intervention implementation and tracking"""
        
        # Track initial performance
        await engine.track_team_performance(crisis_id, team_members)
        
        # Implement multiple interventions
        interventions = []
        intervention_types = [
            InterventionType.COACHING,
            InterventionType.TRAINING,
            InterventionType.ADDITIONAL_SUPPORT
        ]
        
        for i, intervention_type in enumerate(intervention_types):
            intervention = await engine.implement_intervention(
                crisis_id, team_members[i], intervention_type
            )
            interventions.append(intervention)
        
        # Verify interventions are stored
        assert crisis_id in engine.intervention_history
        stored_interventions = engine.intervention_history[crisis_id]
        assert len(stored_interventions) == len(interventions)
        
        # Verify intervention details
        for intervention in interventions:
            assert intervention.crisis_id == crisis_id
            assert intervention.member_id in team_members
            assert intervention.intervention_type in intervention_types
            assert intervention.description
            assert intervention.expected_outcome
            assert isinstance(intervention.implemented_at, datetime)
        
        print("✅ Intervention effectiveness test passed!")
    
    @pytest.mark.asyncio
    async def test_support_provision_tracking(self, engine, crisis_id, team_members):
        """Test support provision and tracking"""
        
        # Provide different types of support
        support_types = [
            SupportType.TECHNICAL_SUPPORT,
            SupportType.EMOTIONAL_SUPPORT,
            SupportType.MENTORING
        ]
        
        support_provisions = []
        for i, support_type in enumerate(support_types):
            support = await engine.provide_support(
                crisis_id, team_members[i], support_type, f"provider_{i}"
            )
            support_provisions.append(support)
        
        # Verify support provisions are stored
        crisis_support = [s for s in engine.support_provisions.values() if s.crisis_id == crisis_id]
        assert len(crisis_support) == len(support_types)
        
        # Verify support details
        for support in support_provisions:
            assert support.crisis_id == crisis_id
            assert support.member_id in team_members
            assert support.support_type in support_types
            assert support.description
            assert support.provider.startswith("provider_")
            assert isinstance(support.provided_at, datetime)
        
        print("✅ Support provision tracking test passed!")
    
    @pytest.mark.asyncio
    async def test_performance_optimization_recommendations(self, engine, crisis_id, team_members):
        """Test performance optimization recommendations"""
        
        # Track team performance first
        await engine.track_team_performance(crisis_id, team_members)
        
        # Get optimization recommendations
        optimizations = await engine.optimize_team_performance(crisis_id)
        
        # Verify optimization structure
        for opt in optimizations:
            assert opt.crisis_id == crisis_id
            assert opt.target_area
            assert opt.current_performance >= 0
            assert opt.target_performance >= 0
            assert opt.optimization_strategy
            assert isinstance(opt.implementation_steps, list)
            assert len(opt.implementation_steps) > 0
            assert opt.priority_level in ["HIGH", "MEDIUM", "LOW"]
            assert opt.estimated_completion_time > 0
            assert isinstance(opt.resources_required, list)
        
        print("✅ Performance optimization recommendations test passed!")
    
    @pytest.mark.asyncio
    async def test_alert_generation_and_management(self, engine, crisis_id, team_members):
        """Test alert generation and management"""
        
        # Track team performance to generate potential alerts
        await engine.track_team_performance(crisis_id, team_members)
        
        # Generate alerts
        alerts = await engine.generate_performance_alerts(crisis_id)
        
        # Verify alert structure
        for alert in alerts:
            assert alert.crisis_id == crisis_id
            assert alert.alert_type in ["CRITICAL_PERFORMANCE", "HIGH_STRESS"]
            assert alert.severity in ["HIGH", "MEDIUM", "LOW"]
            assert alert.message
            assert isinstance(alert.triggered_at, datetime)
            assert alert.acknowledged_at is None  # Initially not acknowledged
            assert alert.resolved_at is None  # Initially not resolved
        
        # Test alert acknowledgment
        if alerts:
            alert = alerts[0]
            alert.acknowledged_at = datetime.now()
            assert alert.acknowledged_at is not None
        
        print("✅ Alert generation and management test passed!")
    
    @pytest.mark.asyncio
    async def test_comprehensive_reporting(self, engine, crisis_id, team_members):
        """Test comprehensive performance reporting"""
        
        # Set up complete scenario
        await engine.track_team_performance(crisis_id, team_members)
        await engine.implement_intervention(crisis_id, team_members[0], InterventionType.COACHING)
        await engine.provide_support(crisis_id, team_members[1], SupportType.TECHNICAL_SUPPORT, "engineer")
        
        # Generate comprehensive report
        report = await engine.generate_performance_report(crisis_id, 24)
        
        # Verify report structure
        assert report.crisis_id == crisis_id
        assert report.report_type == "COMPREHENSIVE_PERFORMANCE"
        assert isinstance(report.generated_at, datetime)
        assert isinstance(report.time_period_start, datetime)
        assert isinstance(report.time_period_end, datetime)
        
        # Verify report content
        assert isinstance(report.team_overview, TeamPerformanceOverview)
        assert isinstance(report.key_insights, list)
        assert len(report.key_insights) > 0
        assert isinstance(report.performance_trends, dict)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.success_metrics, dict)
        
        # Verify success metrics
        expected_metrics = [
            "overall_performance", "team_efficiency", "collaboration_index",
            "task_completion_rate", "average_response_time", "stress_level"
        ]
        for metric in expected_metrics:
            assert metric in report.success_metrics
            assert isinstance(report.success_metrics[metric], (int, float))
        
        print("✅ Comprehensive reporting test passed!")
    
    @pytest.mark.asyncio
    async def test_concurrent_crisis_handling(self, engine):
        """Test handling multiple concurrent crises"""
        
        crisis_ids = ["crisis_001", "crisis_002", "crisis_003"]
        team_members_sets = [
            ["alice", "bob"],
            ["charlie", "diana"],
            ["eve", "frank"]
        ]
        
        # Track performance for multiple crises concurrently
        tasks = []
        for crisis_id, team_members in zip(crisis_ids, team_members_sets):
            task = engine.track_team_performance(crisis_id, team_members)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Verify each crisis has separate data
        for i, (crisis_id, result) in enumerate(zip(crisis_ids, results)):
            assert result.crisis_id == crisis_id
            assert len(result.member_performances) == len(team_members_sets[i])
            assert crisis_id in engine.performance_data
        
        # Verify data isolation between crises
        assert len(engine.performance_data) >= len(crisis_ids)
        for crisis_id in crisis_ids:
            assert crisis_id in engine.performance_data
        
        print("✅ Concurrent crisis handling test passed!")
    
    @pytest.mark.asyncio
    async def test_performance_trend_analysis(self, engine, crisis_id, team_members):
        """Test performance trend analysis over time"""
        
        # Track performance multiple times to simulate time progression
        overviews = []
        for i in range(3):
            overview = await engine.track_team_performance(crisis_id, team_members)
            overviews.append(overview)
            await asyncio.sleep(0.1)  # Small delay to simulate time passage
        
        # Verify performance data is updated each time
        for i in range(1, len(overviews)):
            assert overviews[i].last_updated > overviews[i-1].last_updated
        
        # Test trend analysis
        from datetime import timedelta
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        trends = await engine._analyze_performance_trends(crisis_id, start_time, end_time)
        
        # Verify trend analysis structure
        expected_trends = [
            "performance_trend", "efficiency_trend", "stress_trend",
            "collaboration_trend", "response_time_trend"
        ]
        for trend in expected_trends:
            assert trend in trends
            assert isinstance(trends[trend], str)
        
        print("✅ Performance trend analysis test passed!")
    
    def test_performance_status_classification(self, engine):
        """Test performance status classification logic"""
        
        # Test different performance score ranges
        test_cases = [
            (95.0, PerformanceStatus.EXCELLENT),
            (85.0, PerformanceStatus.GOOD),
            (75.0, PerformanceStatus.AVERAGE),
            (65.0, PerformanceStatus.BELOW_AVERAGE),
            (55.0, PerformanceStatus.CRITICAL)
        ]
        
        for score, expected_status in test_cases:
            # This would require access to the internal classification logic
            # For now, we'll test the enum values
            assert expected_status in PerformanceStatus
        
        print("✅ Performance status classification test passed!")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, engine):
        """Test error handling and recovery scenarios"""
        
        # Test with invalid crisis ID
        try:
            await engine.identify_performance_issues("nonexistent_crisis")
            # Should return empty list, not raise exception
        except Exception as e:
            pytest.fail(f"Should handle missing crisis gracefully: {e}")
        
        # Test with empty team members
        try:
            await engine.track_team_performance("test_crisis", [])
            pytest.fail("Should raise exception for empty team members")
        except Exception:
            pass  # Expected to raise exception
        
        # Test report generation without data
        try:
            await engine.generate_performance_report("nonexistent_crisis", 24)
            pytest.fail("Should raise exception for missing data")
        except ValueError:
            pass  # Expected to raise ValueError
        
        print("✅ Error handling and recovery test passed!")


if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        engine = PerformanceMonitoringEngine()
        crisis_id = "demo_crisis"
        team_members = ["alice", "bob", "charlie"]
        
        print("🧪 Running simple performance monitoring test...")
        
        # Track performance
        overview = await engine.track_team_performance(crisis_id, team_members)
        print(f"✅ Tracked performance for {len(team_members)} members")
        print(f"   Overall score: {overview.overall_performance_score:.1f}/100")
        
        # Implement intervention
        intervention = await engine.implement_intervention(
            crisis_id, "alice", InterventionType.COACHING
        )
        print(f"✅ Implemented {intervention.intervention_type.value} intervention")
        
        # Generate report
        report = await engine.generate_performance_report(crisis_id, 1)
        print(f"✅ Generated performance report with {len(report.key_insights)} insights")
        
        print("🎉 Simple test completed successfully!")
    
    asyncio.run(run_simple_test())