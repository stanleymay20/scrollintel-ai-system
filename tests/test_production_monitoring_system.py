"""
Comprehensive Test Suite for Production Monitoring and Alerting System

This module tests all components of the production monitoring system including
real-time monitoring, user experience tracking, failure pattern detection,
and comprehensive reporting.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from scrollintel.core.production_monitoring import (
    ProductionMonitor, SystemMetric, Alert, AlertSeverity, MetricType
)
from scrollintel.core.ux_quality_monitor import (
    UXQualityMonitor, UXMetric, UXMetricType, UserSession
)
from scrollintel.core.failure_pattern_detector import (
    FailurePatternDetector, FailureEvent, FailureType, PatternSeverity
)
from scrollintel.core.comprehensive_reporting import (
    ComprehensiveReporter, Report, ReportType, ReportFrequency
)

class TestProductionMonitor:
    """Test suite for ProductionMonitor"""
    
    @pytest.fixture
    def monitor(self):
        return ProductionMonitor()
    
    def test_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.metrics_buffer is not None
        assert monitor.alerts == {}
        assert monitor.alert_rules is not None
        assert len(monitor.alert_rules) > 0
        assert not monitor.monitoring_active
    
    def test_record_user_action(self, monitor):
        """Test recording user actions"""
        monitor.record_user_action(
            user_id="user123",
            action="page_load",
            response_time=1500.0,
            success=True,
            satisfaction=0.9
        )
        
        # Check that metrics were recorded
        assert len(monitor.metrics_buffer) > 0
        assert "user123" in monitor.user_sessions
        
        # Check user session data
        session = monitor.user_sessions["user123"]
        assert len(session["actions"]) == 1
        assert session["satisfaction"] == 0.9
    
    def test_get_system_health(self, monitor):
        """Test system health retrieval"""
        # Add some test metrics
        monitor.record_user_action("user1", "test", 1000, True, 0.8)
        
        health = monitor.get_system_health()
        
        assert "timestamp" in health
        assert "health_score" in health
        assert "status" in health
        assert "metrics" in health
        assert "active_alerts" in health
        assert health["health_score"] >= 0
        assert health["health_score"] <= 100
    
    def test_alert_evaluation(self, monitor):
        """Test alert condition evaluation"""
        rule = {
            "operator": ">",
            "threshold": 100
        }
        
        assert monitor._evaluate_alert_condition(150, rule) == True
        assert monitor._evaluate_alert_condition(50, rule) == False
        assert monitor._evaluate_alert_condition(100, rule) == False
    
    def test_health_score_calculation(self, monitor):
        """Test health score calculation"""
        # Test with good metrics
        good_metrics = {
            "cpu_usage": {"current": 50},
            "memory_usage": {"current": 60},
            "error_rate": {"current": 0.001},
            "avg_response_time": {"current": 500},
            "user_satisfaction": {"current": 0.9}
        }
        
        score = monitor._calculate_health_score(good_metrics)
        assert score > 80
        
        # Test with poor metrics
        poor_metrics = {
            "cpu_usage": {"current": 95},
            "memory_usage": {"current": 95},
            "error_rate": {"current": 0.1},
            "avg_response_time": {"current": 5000},
            "user_satisfaction": {"current": 0.5}
        }
        
        score = monitor._calculate_health_score(poor_metrics)
        assert score < 50

class TestUXQualityMonitor:
    """Test suite for UXQualityMonitor"""
    
    @pytest.fixture
    def ux_monitor(self):
        return UXQualityMonitor()
    
    def test_initialization(self, ux_monitor):
        """Test UX monitor initialization"""
        assert ux_monitor.ux_metrics is not None
        assert ux_monitor.user_sessions == {}
        assert ux_monitor.optimization_rules is not None
        assert not ux_monitor.monitoring_active
    
    def test_start_user_session(self, ux_monitor):
        """Test starting user session"""
        session_id = ux_monitor.start_user_session(
            user_id="user123",
            context={"page": "dashboard"}
        )
        
        assert session_id is not None
        assert session_id in ux_monitor.user_sessions
        
        session = ux_monitor.user_sessions[session_id]
        assert session.user_id == "user123"
        assert session.context["page"] == "dashboard"
    
    def test_record_ux_metric(self, ux_monitor):
        """Test recording UX metrics"""
        session_id = ux_monitor.start_user_session("user123")
        
        ux_monitor.record_ux_metric(
            user_id="user123",
            session_id=session_id,
            metric_type=UXMetricType.PERFORMANCE,
            metric_name="page_load_time",
            value=2500.0,
            context={"page": "dashboard"}
        )
        
        assert len(ux_monitor.ux_metrics) == 1
        metric = ux_monitor.ux_metrics[0]
        assert metric.metric_name == "page_load_time"
        assert metric.value == 2500.0
    
    def test_end_user_session(self, ux_monitor):
        """Test ending user session"""
        session_id = ux_monitor.start_user_session("user123")
        
        # Simulate some activity
        session = ux_monitor.user_sessions[session_id]
        session.actions_completed = 5
        
        ux_monitor.end_user_session(session_id, final_satisfaction=0.85)
        
        assert session.satisfaction_score == 0.85
        assert session.engagement_score > 0
    
    def test_get_ux_dashboard(self, ux_monitor):
        """Test UX dashboard data retrieval"""
        # Add some test data
        session_id = ux_monitor.start_user_session("user123")
        ux_monitor.record_ux_metric(
            "user123", session_id, UXMetricType.SATISFACTION,
            "satisfaction_rating", 0.9
        )
        
        dashboard = ux_monitor.get_ux_dashboard()
        
        assert "timestamp" in dashboard
        assert "active_sessions" in dashboard
        assert "average_satisfaction" in dashboard
        assert "satisfaction_level" in dashboard
    
    def test_satisfaction_level_calculation(self, ux_monitor):
        """Test satisfaction level calculation"""
        assert ux_monitor._get_satisfaction_level(0.95) == "excellent"
        assert ux_monitor._get_satisfaction_level(0.85) == "good"
        assert ux_monitor._get_satisfaction_level(0.75) == "acceptable"
        assert ux_monitor._get_satisfaction_level(0.65) == "poor"

class TestFailurePatternDetector:
    """Test suite for FailurePatternDetector"""
    
    @pytest.fixture
    def detector(self):
        return FailurePatternDetector()
    
    def test_initialization(self, detector):
        """Test detector initialization"""
        assert detector.failure_events is not None
        assert detector.detected_patterns == {}
        assert detector.prevention_rules is not None
        assert not detector.detection_active
    
    def test_record_failure(self, detector):
        """Test recording failure events"""
        event_id = detector.record_failure(
            failure_type=FailureType.SYSTEM_ERROR,
            component="api_gateway",
            error_message="Connection timeout",
            stack_trace="Stack trace here",
            user_id="user123",
            severity=PatternSeverity.HIGH
        )
        
        assert event_id is not None
        assert len(detector.failure_events) == 1
        
        event = detector.failure_events[0]
        assert event.failure_type == FailureType.SYSTEM_ERROR
        assert event.component == "api_gateway"
        assert event.error_message == "Connection timeout"
    
    def test_normalize_error_message(self, detector):
        """Test error message normalization"""
        original = "User 12345 failed at 2023-01-01T10:30:00 with IP 192.168.1.1"
        normalized = detector._normalize_error_message(original)
        
        assert "NUM" in normalized
        assert "TIMESTAMP" in normalized
        assert "IP" in normalized
        assert "12345" not in normalized
        assert "192.168.1.1" not in normalized
    
    def test_calculate_pattern_severity(self, detector):
        """Test pattern severity calculation"""
        # Create test events
        events = []
        for i in range(25):  # High frequency
            event = Mock()
            event.timestamp = datetime.now() - timedelta(minutes=i)
            events.append(event)
        
        severity = detector._calculate_pattern_severity(events)
        assert severity == PatternSeverity.CRITICAL
        
        # Test with fewer events
        events = events[:5]
        severity = detector._calculate_pattern_severity(events)
        assert severity == PatternSeverity.MEDIUM
    
    def test_get_component_health(self, detector):
        """Test component health retrieval"""
        # Record some failures
        detector.record_failure(
            FailureType.SYSTEM_ERROR, "component1", "Error 1"
        )
        detector.record_failure(
            FailureType.PERFORMANCE_DEGRADATION, "component1", "Error 2"
        )
        
        health = detector.get_component_health("component1")
        
        assert "component1" in health
        assert "recent_failures" in health["component1"]
        assert "failure_rate" in health["component1"]
        assert "health_status" in health["component1"]
    
    def test_failure_rate_trend_analysis(self, detector):
        """Test failure rate trend analysis"""
        # Create events with increasing frequency
        events = []
        base_time = datetime.now() - timedelta(hours=2)
        
        # First half: 2 events
        for i in range(2):
            event = Mock()
            event.timestamp = base_time + timedelta(minutes=i * 30)
            events.append(event)
        
        # Second half: 6 events (increasing trend)
        for i in range(6):
            event = Mock()
            event.timestamp = base_time + timedelta(hours=1, minutes=i * 10)
            events.append(event)
        
        trend = detector._analyze_failure_rate_trend(events)
        assert trend == "increasing"

class TestComprehensiveReporter:
    """Test suite for ComprehensiveReporter"""
    
    @pytest.fixture
    def reporter(self):
        # Create mock dependencies
        mock_monitor = Mock()
        mock_ux_monitor = Mock()
        mock_detector = Mock()
        
        return ComprehensiveReporter(
            production_monitor=mock_monitor,
            ux_monitor=mock_ux_monitor,
            pattern_detector=mock_detector
        )
    
    def test_initialization(self, reporter):
        """Test reporter initialization"""
        assert reporter.generated_reports == {}
        assert reporter.report_schedules is not None
        assert reporter.insight_templates is not None
        assert not reporter.reporting_active
    
    @pytest.mark.asyncio
    async def test_generate_system_health_report(self, reporter):
        """Test system health report generation"""
        # Mock production monitor data
        reporter.production_monitor.get_system_health.return_value = {
            "health_score": 85.0,
            "status": "healthy",
            "metrics": {
                "cpu_usage": {"current": 60.0},
                "memory_usage": {"current": 70.0}
            },
            "active_alerts": 2
        }
        
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        report = await reporter._generate_system_health_report(start_time, end_time)
        
        assert report.report_type == ReportType.SYSTEM_HEALTH
        assert report.title == "System Health Report"
        assert "health_score" in report.metrics
        assert len(report.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_generate_ux_report(self, reporter):
        """Test UX report generation"""
        # Mock UX monitor data
        reporter.ux_monitor.get_ux_dashboard.return_value = {
            "active_sessions": 50,
            "average_satisfaction": 0.75,  # Below threshold
            "average_load_time": 3500.0    # Above threshold
        }
        reporter.ux_monitor.get_optimization_recommendations.return_value = [
            {"description": "Optimize page loading"}
        ]
        
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        report = await reporter._generate_ux_report(start_time, end_time)
        
        assert report.report_type == ReportType.USER_EXPERIENCE
        assert len(report.insights) >= 2  # Should detect satisfaction and load time issues
        assert any("satisfaction" in insight.title.lower() for insight in report.insights)
    
    @pytest.mark.asyncio
    async def test_generate_failure_analysis_report(self, reporter):
        """Test failure analysis report generation"""
        # Mock pattern detector data
        reporter.pattern_detector.get_detected_patterns.return_value = [
            {"severity": "critical", "pattern_type": "cascade_failure"},
            {"severity": "high", "pattern_type": "error_pattern"}
        ]
        reporter.pattern_detector.get_component_health.return_value = {
            "component1": {"health_status": "degraded"},
            "component2": {"health_status": "healthy"}
        }
        reporter.pattern_detector.get_prevention_status.return_value = {
            "active_rules": 5,
            "recently_triggered": 2
        }
        
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        report = await reporter._generate_failure_analysis_report(start_time, end_time)
        
        assert report.report_type == ReportType.FAILURE_ANALYSIS
        assert len(report.insights) >= 1  # Should detect critical patterns
        assert "total_patterns" in report.metrics
    
    def test_get_report_schedules(self, reporter):
        """Test report schedules retrieval"""
        schedules = reporter.get_report_schedules()
        
        assert len(schedules) > 0
        assert all("schedule_id" in schedule for schedule in schedules)
        assert all("report_type" in schedule for schedule in schedules)
        assert all("frequency" in schedule for schedule in schedules)
    
    def test_update_report_schedule(self, reporter):
        """Test report schedule updates"""
        # Get first schedule ID
        schedules = reporter.get_report_schedules()
        schedule_id = schedules[0]["schedule_id"]
        
        # Update the schedule
        updates = {"enabled": False}
        success = reporter.update_report_schedule(schedule_id, updates)
        
        assert success == True
        
        # Verify update
        updated_schedules = reporter.get_report_schedules()
        updated_schedule = next(s for s in updated_schedules if s["schedule_id"] == schedule_id)
        assert updated_schedule["enabled"] == False
    
    def test_get_insights_summary(self, reporter):
        """Test insights summary generation"""
        # Create a mock report with insights
        mock_report = Mock()
        mock_report.generated_at = datetime.now()
        mock_report.insights = [
            Mock(insight_type=Mock(value="trend"), impact_level="high", confidence=0.8, recommendations=["rec1"]),
            Mock(insight_type=Mock(value="alert"), impact_level="medium", confidence=0.9, recommendations=["rec2"])
        ]
        
        reporter.generated_reports["test_report"] = mock_report
        
        summary = reporter.get_insights_summary(days=7)
        
        assert "total_insights" in summary
        assert "insight_types" in summary
        assert "impact_levels" in summary
        assert "average_confidence" in summary

class TestIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create a complete monitoring system"""
        monitor = ProductionMonitor()
        ux_monitor = UXQualityMonitor()
        detector = FailurePatternDetector()
        reporter = ComprehensiveReporter(monitor, ux_monitor, detector)
        
        return {
            "monitor": monitor,
            "ux_monitor": ux_monitor,
            "detector": detector,
            "reporter": reporter
        }
    
    def test_end_to_end_monitoring_flow(self, monitoring_system):
        """Test complete monitoring flow"""
        monitor = monitoring_system["monitor"]
        ux_monitor = monitoring_system["ux_monitor"]
        detector = monitoring_system["detector"]
        
        # 1. Record user action
        monitor.record_user_action("user123", "page_load", 2500, True, 0.8)
        
        # 2. Start UX session and record metrics
        session_id = ux_monitor.start_user_session("user123")
        ux_monitor.record_ux_metric(
            "user123", session_id, UXMetricType.PERFORMANCE,
            "page_load_time", 2500.0
        )
        
        # 3. Record a failure
        detector.record_failure(
            FailureType.PERFORMANCE_DEGRADATION,
            "web_server",
            "Slow response time detected"
        )
        
        # 4. Check system health
        health = monitor.get_system_health()
        assert health["health_score"] >= 0
        
        # 5. Check UX dashboard
        ux_dashboard = ux_monitor.get_ux_dashboard()
        assert ux_dashboard["active_sessions"] > 0
        
        # 6. Check component health
        component_health = detector.get_component_health()
        assert "web_server" in component_health
    
    @pytest.mark.asyncio
    async def test_report_generation_integration(self, monitoring_system):
        """Test report generation with real data"""
        monitor = monitoring_system["monitor"]
        ux_monitor = monitoring_system["ux_monitor"]
        detector = monitoring_system["detector"]
        reporter = monitoring_system["reporter"]
        
        # Generate some test data
        monitor.record_user_action("user1", "login", 1000, True, 0.9)
        monitor.record_user_action("user2", "search", 3000, False, 0.6)  # Slow and failed
        
        session_id = ux_monitor.start_user_session("user1")
        ux_monitor.record_ux_metric(
            "user1", session_id, UXMetricType.SATISFACTION,
            "satisfaction_rating", 0.9
        )
        
        detector.record_failure(
            FailureType.SYSTEM_ERROR, "search_service", "Database timeout"
        )
        
        # Generate reports
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        health_report = await reporter.generate_report(
            ReportType.SYSTEM_HEALTH, (start_time, end_time)
        )
        
        ux_report = await reporter.generate_report(
            ReportType.USER_EXPERIENCE, (start_time, end_time)
        )
        
        failure_report = await reporter.generate_report(
            ReportType.FAILURE_ANALYSIS, (start_time, end_time)
        )
        
        # Verify reports were generated
        assert health_report.report_type == ReportType.SYSTEM_HEALTH
        assert ux_report.report_type == ReportType.USER_EXPERIENCE
        assert failure_report.report_type == ReportType.FAILURE_ANALYSIS
        
        # Verify reports contain data
        assert len(health_report.recommendations) > 0
        assert len(ux_report.recommendations) > 0
        assert len(failure_report.recommendations) > 0

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])