"""
Tests for automated reporting and alerting system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from scrollintel.core.automated_reporting import (
    AutomatedReportingSystem,
    ReportSchedule,
    AlertRule,
    Alert,
    ReportFrequency,
    AlertSeverity
)

class TestAutomatedReportingSystem:
    """Test automated reporting system functionality."""
    
    @pytest.fixture
    def reporting_system(self):
        """Create a fresh reporting system for each test."""
        return AutomatedReportingSystem()
    
    @pytest.mark.asyncio
    async def test_create_report_schedule_daily(self, reporting_system):
        """Test creating a daily report schedule."""
        schedule_id = await reporting_system.create_report_schedule(
            name="Daily Performance Report",
            report_type="performance",
            frequency=ReportFrequency.DAILY,
            recipients=["admin@example.com", "manager@example.com"],
            team_ids=["team_001", "team_002"]
        )
        
        assert schedule_id is not None
        assert schedule_id in reporting_system.report_schedules
        
        schedule = reporting_system.report_schedules[schedule_id]
        assert schedule.name == "Daily Performance Report"
        assert schedule.report_type == "performance"
        assert schedule.frequency == ReportFrequency.DAILY
        assert len(schedule.recipients) == 2
        assert len(schedule.team_ids) == 2
        assert schedule.active is True
        assert schedule.next_run > datetime.utcnow()
    
    @pytest.mark.asyncio
    async def test_create_report_schedule_weekly(self, reporting_system):
        """Test creating a weekly report schedule."""
        schedule_id = await reporting_system.create_report_schedule(
            name="Weekly Analytics Summary",
            report_type="analytics",
            frequency=ReportFrequency.WEEKLY,
            recipients=["executive@example.com"],
            team_ids=["team_001"],
            prompt_ids=["prompt_123", "prompt_456"]
        )
        
        schedule = reporting_system.report_schedules[schedule_id]
        assert schedule.frequency == ReportFrequency.WEEKLY
        assert len(schedule.prompt_ids) == 2
        
        # Next run should be scheduled for next Monday
        next_monday = schedule.next_run
        assert next_monday.weekday() == 0  # Monday
    
    @pytest.mark.asyncio
    async def test_create_alert_rule_threshold(self, reporting_system):
        """Test creating a threshold-based alert rule."""
        rule_id = await reporting_system.create_alert_rule(
            name="Low Success Rate Alert",
            rule_type="threshold",
            metric_name="success_rate",
            condition="less_than",
            threshold_value=95.0,
            severity="high",
            notification_channels=["email", "slack"],
            recipients=["ops@example.com"],
            team_ids=["team_001"]
        )
        
        assert rule_id is not None
        assert rule_id in reporting_system.alert_rules
        
        rule = reporting_system.alert_rules[rule_id]
        assert rule.name == "Low Success Rate Alert"
        assert rule.rule_type == "threshold"
        assert rule.metric_name == "success_rate"
        assert rule.condition == "less_than"
        assert rule.threshold_value == 95.0
        assert rule.severity == AlertSeverity.HIGH
        assert "email" in rule.notification_channels
        assert "slack" in rule.notification_channels
    
    @pytest.mark.asyncio
    async def test_generate_report_basic(self, reporting_system):
        """Test basic report generation."""
        # Mock team dashboard data
        mock_dashboard_data = {
            'team_id': 'team_001',
            'summary': {
                'total_requests': 150,
                'success_rate': 94.0,
                'avg_accuracy_score': 0.87
            },
            'insights': [
                {
                    'type': 'performance',
                    'title': 'Good Performance',
                    'description': 'Team is performing well'
                }
            ]
        }
        
        with patch('scrollintel.core.automated_reporting.team_analytics_dashboard') as mock_dashboard:
            mock_dashboard.get_team_dashboard_data = AsyncMock(return_value=mock_dashboard_data)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            report = await reporting_system.generate_report(
                report_type="performance",
                team_ids=["team_001"],
                date_range=(start_date, end_date)
            )
            
            assert 'error' not in report
            assert report['report_type'] == "performance"
            assert len(report['teams']) == 1
            assert report['teams'][0]['team_id'] == 'team_001'
            assert 'summary' in report
            assert 'insights' in report
            assert 'recommendations' in report
    
    @pytest.mark.asyncio
    async def test_generate_report_multiple_teams(self, reporting_system):
        """Test report generation for multiple teams."""
        # Mock data for multiple teams
        def mock_dashboard_side_effect(team_id, date_range):
            return {
                'team_id': team_id,
                'summary': {
                    'total_requests': 100 if team_id == 'team_001' else 200,
                    'success_rate': 95.0 if team_id == 'team_001' else 88.0,
                    'total_cost': 15.0 if team_id == 'team_001' else 30.0
                }
            }
        
        with patch('scrollintel.core.automated_reporting.team_analytics_dashboard') as mock_dashboard:
            mock_dashboard.get_team_dashboard_data = AsyncMock(side_effect=mock_dashboard_side_effect)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            report = await reporting_system.generate_report(
                report_type="team_comparison",
                team_ids=["team_001", "team_002"],
                date_range=(start_date, end_date)
            )
            
            assert len(report['teams']) == 2
            
            # Verify summary aggregation
            summary = report['summary']
            assert summary['total_teams'] == 2
            assert summary['total_requests'] == 300  # 100 + 200
            assert summary['total_cost'] == 45.0     # 15.0 + 30.0
    
    @pytest.mark.asyncio
    async def test_report_summary_generation(self, reporting_system):
        """Test report summary generation from team data."""
        teams_data = [
            {
                'team_id': 'team_001',
                'data': {
                    'summary': {
                        'total_requests': 150,
                        'successful_requests': 142,
                        'total_cost': 25.0,
                        'avg_accuracy_score': 0.88,
                        'avg_response_time_ms': 1500
                    }
                }
            },
            {
                'team_id': 'team_002',
                'data': {
                    'summary': {
                        'total_requests': 200,
                        'successful_requests': 190,
                        'total_cost': 35.0,
                        'avg_accuracy_score': 0.92,
                        'avg_response_time_ms': 1200
                    }
                }
            }
        ]
        
        summary = await reporting_system._generate_report_summary(teams_data)
        
        assert summary['total_teams'] == 2
        assert summary['total_requests'] == 350
        assert summary['total_successful_requests'] == 332
        assert summary['overall_success_rate'] == (332 / 350 * 100)
        assert summary['total_cost'] == 60.0
        assert summary['avg_accuracy_score'] == 0.90  # (0.88 + 0.92) / 2
        assert summary['avg_response_time_ms'] == 1350  # (1500 + 1200) / 2
    
    @pytest.mark.asyncio
    async def test_report_insights_generation(self, reporting_system):
        """Test report insights generation."""
        teams_data = [
            {
                'team_id': 'team_high_performance',
                'data': {
                    'summary': {
                        'success_rate': 98.0,
                        'total_cost': 10.0
                    }
                }
            },
            {
                'team_id': 'team_low_performance',
                'data': {
                    'summary': {
                        'success_rate': 85.0,
                        'total_cost': 50.0
                    }
                }
            }
        ]
        
        insights = await reporting_system._generate_report_insights(teams_data)
        
        assert isinstance(insights, list)
        
        # Should have insights about performance variation
        performance_insights = [i for i in insights if i['type'] == 'performance']
        assert len(performance_insights) > 0
        
        # Should have cost insights for high total cost
        cost_insights = [i for i in insights if i['type'] == 'cost']
        assert len(cost_insights) > 0
    
    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_threshold(self, reporting_system):
        """Test alert rule evaluation for threshold conditions."""
        # Create alert rule
        rule_id = await reporting_system.create_alert_rule(
            name="Test Alert",
            rule_type="threshold",
            metric_name="success_rate",
            condition="less_than",
            threshold_value=90.0,
            team_ids=["test_team"]
        )
        
        rule = reporting_system.alert_rules[rule_id]
        
        # Mock team analytics with low success rate
        mock_analytics = {
            'summary': {
                'success_rate': 85.0  # Below threshold
            }
        }
        
        with patch('scrollintel.core.automated_reporting.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            should_trigger = await reporting_system._evaluate_alert_rule(rule)
            assert should_trigger is True
        
        # Test with success rate above threshold
        mock_analytics['summary']['success_rate'] = 95.0
        
        with patch('scrollintel.core.automated_reporting.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            should_trigger = await reporting_system._evaluate_alert_rule(rule)
            assert should_trigger is False
    
    @pytest.mark.asyncio
    async def test_threshold_condition_checking(self, reporting_system):
        """Test threshold condition checking logic."""
        # Create a mock rule for testing
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description=None,
            rule_type="threshold",
            metric_name="test_metric",
            condition="greater_than",
            threshold_value=100.0,
            trend_direction=None,
            severity=AlertSeverity.MEDIUM,
            notification_channels=["email"],
            recipients=["test@example.com"],
            prompt_ids=None,
            team_ids=["test_team"],
            active=True,
            last_triggered=None,
            trigger_count=0,
            created_by="test",
            created_at=datetime.utcnow()
        )
        
        # Test greater_than condition
        rule.condition = "greater_than"
        rule.threshold_value = 100.0
        
        assert reporting_system._check_threshold_condition(150.0, rule) is True
        assert reporting_system._check_threshold_condition(50.0, rule) is False
        
        # Test less_than condition
        rule.condition = "less_than"
        rule.threshold_value = 100.0
        
        assert reporting_system._check_threshold_condition(50.0, rule) is True
        assert reporting_system._check_threshold_condition(150.0, rule) is False
        
        # Test equals condition
        rule.condition = "equals"
        rule.threshold_value = 100.0
        
        assert reporting_system._check_threshold_condition(100.0, rule) is True
        assert reporting_system._check_threshold_condition(100.0001, rule) is False
    
    @pytest.mark.asyncio
    async def test_alert_triggering(self, reporting_system):
        """Test alert triggering and notification."""
        # Create alert rule
        rule_id = await reporting_system.create_alert_rule(
            name="Critical Performance Alert",
            rule_type="threshold",
            metric_name="success_rate",
            condition="less_than",
            threshold_value=80.0,
            severity="critical",
            recipients=["critical@example.com"]
        )
        
        rule = reporting_system.alert_rules[rule_id]
        
        # Mock email sending
        with patch.object(reporting_system, '_send_email') as mock_send_email:
            mock_send_email.return_value = None
            
            await reporting_system._trigger_alert(rule)
            
            # Verify alert was created
            assert len(reporting_system.active_alerts) == 1
            
            alert = list(reporting_system.active_alerts.values())[0]
            assert alert.rule_id == rule_id
            assert alert.severity == AlertSeverity.CRITICAL
            assert alert.status == 'active'
            
            # Verify rule statistics were updated
            assert rule.last_triggered is not None
            assert rule.trigger_count == 1
            
            # Verify email was sent
            mock_send_email.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ad_hoc_report_generation(self, reporting_system):
        """Test ad-hoc report generation."""
        mock_dashboard_data = {
            'team_id': 'adhoc_team',
            'summary': {'total_requests': 75}
        }
        
        with patch('scrollintel.core.automated_reporting.team_analytics_dashboard') as mock_dashboard:
            mock_dashboard.get_team_dashboard_data = AsyncMock(return_value=mock_dashboard_data)
            
            with patch.object(reporting_system, '_send_report') as mock_send_report:
                mock_send_report.return_value = None
                
                start_date = datetime(2024, 1, 1)
                end_date = datetime(2024, 1, 7)
                
                await reporting_system.generate_ad_hoc_report(
                    team_id="adhoc_team",
                    report_type="performance",
                    date_range=(start_date, end_date),
                    recipients=["adhoc@example.com"]
                )
                
                # Verify report was sent
                mock_send_report.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_report_html_formatting(self, reporting_system):
        """Test HTML report formatting."""
        sample_report = {
            'report_type': 'performance',
            'generated_at': datetime(2024, 1, 15, 10, 30, 0),
            'date_range': {
                'start': '2024-01-01',
                'end': '2024-01-07'
            },
            'summary': {
                'total_requests': 500,
                'total_cost': 75.50,
                'overall_success_rate': 94.5
            },
            'insights': [
                {
                    'title': 'Good Performance',
                    'description': 'System is performing well overall'
                }
            ]
        }
        
        html_content = reporting_system._format_report_html(sample_report)
        
        assert '<html>' in html_content
        assert '<title>Prompt Analytics Report</title>' in html_content
        assert 'Performance' in html_content
        assert '2024-01-15T10:30:00' in html_content
        assert '500' in html_content
        assert 'Good Performance' in html_content
    
    @pytest.mark.asyncio
    async def test_system_start_stop(self, reporting_system):
        """Test system start and stop functionality."""
        assert reporting_system.running is False
        
        # Start system
        await reporting_system.start_system()
        assert reporting_system.running is True
        assert reporting_system.scheduler_task is not None
        
        # Stop system
        await reporting_system.stop_system()
        assert reporting_system.running is False
    
    @pytest.mark.asyncio
    async def test_scheduled_report_execution(self, reporting_system):
        """Test scheduled report execution."""
        # Create a schedule that should run immediately
        schedule = ReportSchedule(
            schedule_id="test_schedule",
            name="Test Schedule",
            report_type="performance",
            frequency=ReportFrequency.DAILY,
            recipients=["test@example.com"],
            team_ids=["test_team"],
            prompt_ids=None,
            next_run=datetime.utcnow() - timedelta(minutes=1),  # Past due
            last_run=None,
            active=True,
            created_by="test",
            created_at=datetime.utcnow()
        )
        
        reporting_system.report_schedules["test_schedule"] = schedule
        
        # Mock report generation and sending
        with patch.object(reporting_system, 'generate_report') as mock_generate:
            mock_generate.return_value = {'report_type': 'performance', 'teams': []}
            
            with patch.object(reporting_system, '_send_report') as mock_send:
                mock_send.return_value = None
                
                await reporting_system._execute_scheduled_report(schedule)
                
                # Verify report was generated and sent
                mock_generate.assert_called_once()
                mock_send.assert_called_once()
                
                # Verify schedule was updated
                assert schedule.last_run is not None
                assert schedule.next_run > datetime.utcnow()

class TestReportFrequencyCalculation:
    """Test report frequency and scheduling calculations."""
    
    @pytest.mark.asyncio
    async def test_next_run_calculation_hourly(self):
        """Test next run calculation for hourly reports."""
        system = AutomatedReportingSystem()
        
        schedule_id = await system.create_report_schedule(
            name="Hourly Test",
            report_type="performance",
            frequency=ReportFrequency.HOURLY,
            recipients=["test@example.com"],
            team_ids=["test_team"]
        )
        
        schedule = system.report_schedules[schedule_id]
        
        # Next run should be at the next hour boundary
        assert schedule.next_run.minute == 0
        assert schedule.next_run.second == 0
        assert schedule.next_run > datetime.utcnow()
    
    @pytest.mark.asyncio
    async def test_next_run_calculation_monthly(self):
        """Test next run calculation for monthly reports."""
        system = AutomatedReportingSystem()
        
        schedule_id = await system.create_report_schedule(
            name="Monthly Test",
            report_type="analytics",
            frequency=ReportFrequency.MONTHLY,
            recipients=["test@example.com"],
            team_ids=["test_team"]
        )
        
        schedule = system.report_schedules[schedule_id]
        
        # Next run should be first day of next month
        assert schedule.next_run.day == 1
        assert schedule.next_run.hour == 8
        assert schedule.next_run > datetime.utcnow()

class TestErrorHandling:
    """Test error handling in automated reporting."""
    
    @pytest.mark.asyncio
    async def test_report_generation_with_team_error(self):
        """Test report generation when team data has errors."""
        system = AutomatedReportingSystem()
        
        # Mock team dashboard returning error
        with patch('scrollintel.core.automated_reporting.team_analytics_dashboard') as mock_dashboard:
            mock_dashboard.get_team_dashboard_data = AsyncMock(return_value={'error': 'Team not found'})
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            report = await system.generate_report(
                report_type="performance",
                team_ids=["nonexistent_team"],
                date_range=(start_date, end_date)
            )
            
            # Report should still be generated but with empty teams
            assert 'error' not in report
            assert len(report['teams']) == 0
    
    @pytest.mark.asyncio
    async def test_alert_evaluation_with_missing_data(self):
        """Test alert evaluation when analytics data is missing."""
        system = AutomatedReportingSystem()
        
        rule_id = await system.create_alert_rule(
            name="Test Alert",
            rule_type="threshold",
            metric_name="success_rate",
            condition="less_than",
            threshold_value=90.0,
            team_ids=["missing_team"]
        )
        
        rule = system.alert_rules[rule_id]
        
        # Mock analytics returning error
        with patch('scrollintel.core.automated_reporting.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value={'error': 'No data'})
            
            should_trigger = await system._evaluate_alert_rule(rule)
            assert should_trigger is False  # Should not trigger on missing data

if __name__ == "__main__":
    pytest.main([__file__])