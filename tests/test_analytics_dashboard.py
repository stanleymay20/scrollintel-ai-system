"""
Tests for analytics dashboard and reporting system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from scrollintel.core.analytics_dashboard import (
    TeamAnalyticsDashboard, InsightsGenerator
)
from scrollintel.core.automated_reporting import (
    AutomatedReportingSystem, ReportFrequency, AlertSeverity
)
from scrollintel.core.trend_pattern_analysis import (
    AdvancedTrendAnalyzer, AdvancedPatternRecognizer, TrendType, PatternType
)

class TestTeamAnalyticsDashboard:
    """Test team analytics dashboard functionality."""
    
    @pytest.fixture
    def dashboard(self):
        return TeamAnalyticsDashboard()
    
    @pytest.fixture
    def mock_metrics_data(self):
        """Create mock metrics data for testing."""
        mock_metrics = []
        for i in range(50):
            metric = Mock()
            metric.team_id = "team123"
            metric.prompt_id = f"prompt{i % 5}"  # 5 different prompts
            metric.user_id = f"user{i % 10}"  # 10 different users
            metric.accuracy_score = 0.7 + (i % 5) * 0.05 + np.random.normal(0, 0.02)
            metric.relevance_score = 0.8 + np.random.normal(0, 0.03)
            metric.efficiency_score = 0.75 + np.random.normal(0, 0.02)
            metric.user_satisfaction = 3.5 + (i % 5) * 0.2 + np.random.normal(0, 0.1)
            metric.response_time_ms = 1500 + np.random.normal(0, 200)
            metric.token_usage = 150 + np.random.normal(0, 20)
            metric.cost_per_request = 0.003 + np.random.normal(0, 0.0005)
            metric.created_at = datetime.utcnow() - timedelta(hours=i)
            mock_metrics.append(metric)
        return mock_metrics
    
    @pytest.mark.asyncio
    async def test_get_team_dashboard_data(self, dashboard, mock_metrics_data):
        """Test getting comprehensive team dashboard data."""
        with patch('scrollintel.core.analytics_dashboard.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics_data
            
            # Mock usage analytics
            mock_analytics = []
            for i in range(7):
                analytics = Mock()
                analytics.prompt_id = f"prompt{i % 5}"
                analytics.total_requests = 100 + i * 10
                analytics.successful_requests = 95 + i * 9
                analytics.failed_requests = 5 + i
                analytics.hourly_patterns = {str(h): 10 + h for h in range(24)}
                analytics.daily_usage = {f"2024-01-{i+1:02d}": 100 + i * 10}
                analytics.analysis_period_start = datetime.utcnow() - timedelta(days=6-i)
                analytics.analysis_period_end = datetime.utcnow() - timedelta(days=5-i)
                mock_analytics.append(analytics)
            
            # Setup different query results for different calls
            def side_effect(*args, **kwargs):
                query_mock = Mock()
                if hasattr(args[0], '__name__') and 'UsageAnalytics' in str(args[0]):
                    query_mock.filter.return_value.all.return_value = mock_analytics
                else:
                    query_mock.filter.return_value.all.return_value = mock_metrics_data
                    query_mock.filter.return_value.distinct.return_value.all.return_value = [(f"prompt{i}",) for i in range(5)]
                return query_mock
            
            mock_session.query.side_effect = side_effect
            
            dashboard_data = await dashboard.get_team_dashboard_data("team123")
            
            assert dashboard_data["team_id"] == "team123"
            assert "overview" in dashboard_data
            assert "performance_metrics" in dashboard_data
            assert "usage_analytics" in dashboard_data
            assert "top_prompts" in dashboard_data
            assert "improvement_opportunities" in dashboard_data
            assert "trends" in dashboard_data
            assert "recommendations" in dashboard_data
            
            # Check overview data
            overview = dashboard_data["overview"]
            assert overview["total_requests"] > 0
            assert overview["unique_prompts"] > 0
            assert overview["active_users"] > 0
    
    @pytest.mark.asyncio
    async def test_get_team_overview(self, dashboard, mock_metrics_data):
        """Test team overview calculation."""
        with patch('scrollintel.core.analytics_dashboard.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics_data
            
            date_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())
            overview = await dashboard._get_team_overview("team123", date_range)
            
            assert overview["total_requests"] == len(mock_metrics_data)
            assert overview["unique_prompts"] == 5  # 5 different prompts in mock data
            assert overview["active_users"] == 10  # 10 different users in mock data
            assert "avg_accuracy" in overview
            assert "avg_satisfaction" in overview
            assert "total_cost" in overview
    
    @pytest.mark.asyncio
    async def test_get_top_performing_prompts(self, dashboard, mock_metrics_data):
        """Test identification of top performing prompts."""
        with patch('scrollintel.core.analytics_dashboard.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics_data
            
            date_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())
            top_prompts = await dashboard._get_top_performing_prompts("team123", date_range, limit=3)
            
            assert len(top_prompts) <= 3
            assert all("prompt_id" in prompt for prompt in top_prompts)
            assert all("performance_score" in prompt for prompt in top_prompts)
            assert all("usage_count" in prompt for prompt in top_prompts)
            
            # Check that prompts are sorted by performance score
            if len(top_prompts) > 1:
                for i in range(len(top_prompts) - 1):
                    assert top_prompts[i]["performance_score"] >= top_prompts[i+1]["performance_score"]
    
    @pytest.mark.asyncio
    async def test_get_improvement_opportunities(self, dashboard, mock_metrics_data):
        """Test identification of improvement opportunities."""
        # Modify some metrics to have clear issues
        for i in range(0, 10, 2):  # Every other metric has low accuracy
            mock_metrics_data[i].accuracy_score = 0.4
            mock_metrics_data[i].user_satisfaction = 2.0
            mock_metrics_data[i].response_time_ms = 5000
        
        with patch('scrollintel.core.analytics_dashboard.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics_data
            
            date_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())
            opportunities = await dashboard._get_improvement_opportunities("team123", date_range, limit=5)
            
            assert len(opportunities) > 0
            assert all("prompt_id" in opp for opp in opportunities)
            assert all("issues" in opp for opp in opportunities)
            assert all("improvement_potential" in opp for opp in opportunities)
            assert all("recommended_actions" in opp for opp in opportunities)
            
            # Check that opportunities are sorted by improvement potential
            if len(opportunities) > 1:
                for i in range(len(opportunities) - 1):
                    assert opportunities[i]["improvement_potential"] >= opportunities[i+1]["improvement_potential"]
    
    def test_identify_performance_issues(self, dashboard):
        """Test performance issue identification."""
        # Create metrics with known issues
        mock_metrics = []
        for i in range(5):
            metric = Mock()
            metric.accuracy_score = 0.5  # Low accuracy
            metric.response_time_ms = 3000  # Slow response
            metric.user_satisfaction = 2.0  # Low satisfaction
            mock_metrics.append(metric)
        
        issues = dashboard._identify_performance_issues(mock_metrics)
        
        assert len(issues) >= 3  # Should detect all three issues
        issue_types = [issue["type"] for issue in issues]
        assert "low_accuracy" in issue_types
        assert "slow_response" in issue_types
        assert "low_satisfaction" in issue_types
        
        # Check severity levels
        assert all(issue["severity"] in ["high", "medium", "low"] for issue in issues)
    
    def test_calculate_improvement_potential(self, dashboard):
        """Test improvement potential calculation."""
        # High severity issues
        high_issues = [
            {"severity": "high"},
            {"severity": "high"},
            {"severity": "medium"}
        ]
        potential = dashboard._calculate_improvement_potential(high_issues)
        assert potential > 0.5
        
        # Low severity issues
        low_issues = [
            {"severity": "low"},
            {"severity": "low"}
        ]
        potential = dashboard._calculate_improvement_potential(low_issues)
        assert potential < 0.8
        
        # No issues
        no_issues = []
        potential = dashboard._calculate_improvement_potential(no_issues)
        assert potential == 0.0

class TestInsightsGenerator:
    """Test insights generation functionality."""
    
    @pytest.fixture
    def generator(self):
        return InsightsGenerator()
    
    @pytest.fixture
    def sample_team_data(self):
        return {
            "team_id": "team123",
            "overview": {
                "avg_accuracy": 0.6,  # Below threshold
                "cost_per_request": 0.008,  # High cost
                "avg_satisfaction": 3.0  # Low satisfaction
            },
            "performance_metrics": {
                "accuracy_score": {
                    "trend": "declining",
                    "average": 0.6
                },
                "response_time_ms": {
                    "trend": "stable",
                    "average": 1200
                }
            },
            "usage_analytics": {
                "usage_trend": "increasing",
                "success_rate": 92  # Below 95%
            }
        }
    
    @pytest.mark.asyncio
    async def test_generate_insights(self, generator, sample_team_data):
        """Test comprehensive insights generation."""
        with patch.object(generator, '_generate_performance_insights', return_value=[
            {"type": "performance_decline", "priority": "high", "title": "Accuracy Declining"}
        ]):
            with patch.object(generator, '_generate_usage_insights', return_value=[
                {"type": "usage_spike", "priority": "medium", "title": "Usage Increasing"}
            ]):
                with patch.object(generator, '_generate_cost_insights', return_value=[
                    {"type": "cost_optimization", "priority": "medium", "title": "High Cost"}
                ]):
                    with patch.object(generator, '_generate_satisfaction_insights', return_value=[
                        {"type": "user_satisfaction", "priority": "high", "title": "Low Satisfaction"}
                    ]):
                        insights = await generator.generate_insights("team123", (datetime.utcnow() - timedelta(days=7), datetime.utcnow()))
                        
                        assert len(insights) > 0
                        assert all("type" in insight for insight in insights)
                        assert all("priority" in insight for insight in insights)
                        assert all("title" in insight for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_performance_insights(self, generator, sample_team_data):
        """Test performance insights generation."""
        insights = await generator._generate_performance_insights(sample_team_data)
        
        assert len(insights) > 0
        # Should detect declining accuracy
        assert any(insight["type"] == "performance_decline" for insight in insights)
        assert any("accuracy" in insight["title"].lower() for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_cost_insights(self, generator, sample_team_data):
        """Test cost insights generation."""
        insights = await generator._generate_cost_insights(sample_team_data)
        
        assert len(insights) > 0
        # Should detect high cost
        assert any(insight["type"] == "cost_optimization" for insight in insights)
        assert any("cost" in insight["title"].lower() for insight in insights)
    
    def test_get_priority_score(self, generator):
        """Test priority score calculation."""
        high_priority_insight = {
            "priority": "high",
            "impact": "high",
            "actionable": True
        }
        score = generator._get_priority_score(high_priority_insight)
        assert score == 18  # 3 * 3 * 2
        
        low_priority_insight = {
            "priority": "low",
            "impact": "low",
            "actionable": False
        }
        score = generator._get_priority_score(low_priority_insight)
        assert score == 1  # 1 * 1 * 1

class TestAutomatedReportingSystem:
    """Test automated reporting system."""
    
    @pytest.fixture
    def reporting_system(self):
        return AutomatedReportingSystem()
    
    @pytest.mark.asyncio
    async def test_create_report_schedule(self, reporting_system):
        """Test creating a report schedule."""
        schedule_id = await reporting_system.create_report_schedule(
            name="Weekly Team Report",
            report_type="performance",
            frequency=ReportFrequency.WEEKLY,
            recipients=["manager@company.com"],
            team_ids=["team123"]
        )
        
        assert schedule_id is not None
        assert schedule_id in reporting_system.report_schedules
        
        schedule = reporting_system.report_schedules[schedule_id]
        assert schedule.name == "Weekly Team Report"
        assert schedule.frequency == ReportFrequency.WEEKLY
        assert schedule.active == True
    
    @pytest.mark.asyncio
    async def test_create_alert_rule(self, reporting_system):
        """Test creating an alert rule."""
        with patch('scrollintel.core.automated_reporting.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            mock_alert = Mock()
            mock_alert.id = "alert123"
            mock_session.add.return_value = None
            mock_session.commit.return_value = None
            
            alert_id = await reporting_system.create_alert_rule(
                name="Low Accuracy Alert",
                rule_type="threshold",
                metric_name="accuracy_score",
                condition="less_than",
                threshold_value=0.7,
                severity="high",
                recipients=["admin@company.com"],
                team_ids=["team123"]
            )
            
            assert alert_id is not None
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_comprehensive_report(self, reporting_system):
        """Test comprehensive report creation."""
        with patch.object(reporting_system.dashboard, 'get_team_dashboard_data', return_value={
            "team_id": "team123",
            "overview": {"total_requests": 1000},
            "performance_metrics": {},
            "trends": {}
        }):
            with patch.object(reporting_system.insights_generator, 'generate_insights', return_value=[
                {"type": "performance", "priority": "high", "title": "Test Insight"}
            ]):
                date_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())
                
                report_data = await reporting_system._create_comprehensive_report(
                    team_id="team123",
                    report_type="performance",
                    date_range=date_range
                )
                
                assert report_data["team_id"] == "team123"
                assert report_data["report_type"] == "performance"
                assert "executive_summary" in report_data
                assert "detailed_metrics" in report_data
                assert "key_insights" in report_data
                assert "recommendations" in report_data
    
    def test_check_threshold_condition(self, reporting_system):
        """Test threshold condition checking."""
        values = [0.8, 0.7, 0.6, 0.5, 0.4]  # Average = 0.6
        
        # Test greater_than condition
        result = reporting_system._check_threshold_condition(values, "greater_than", 0.5)
        assert result == True  # 0.6 > 0.5
        
        result = reporting_system._check_threshold_condition(values, "greater_than", 0.7)
        assert result == False  # 0.6 < 0.7
        
        # Test less_than condition
        result = reporting_system._check_threshold_condition(values, "less_than", 0.7)
        assert result == True  # 0.6 < 0.7
        
        result = reporting_system._check_threshold_condition(values, "less_than", 0.5)
        assert result == False  # 0.6 > 0.5
        
        # Test equals condition
        result = reporting_system._check_threshold_condition([0.6, 0.6, 0.6], "equals", 0.6)
        assert result == True
    
    def test_calculate_next_generation_time(self, reporting_system):
        """Test next generation time calculation."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Test daily frequency
        next_time = reporting_system._calculate_next_generation_time(ReportFrequency.DAILY, base_time)
        expected = base_time + timedelta(days=1)
        assert next_time == expected
        
        # Test weekly frequency
        next_time = reporting_system._calculate_next_generation_time(ReportFrequency.WEEKLY, base_time)
        expected = base_time + timedelta(weeks=1)
        assert next_time == expected
        
        # Test monthly frequency
        next_time = reporting_system._calculate_next_generation_time(ReportFrequency.MONTHLY, base_time)
        expected = base_time + timedelta(days=30)
        assert next_time == expected

class TestAdvancedTrendAnalyzer:
    """Test advanced trend analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return AdvancedTrendAnalyzer()
    
    def test_fit_linear_trend(self, analyzer):
        """Test linear trend fitting."""
        # Create sample data with clear linear trend
        timestamps = [datetime.utcnow() - timedelta(days=i) for i in range(10, 0, -1)]
        values = [i * 0.1 + 0.5 for i in range(10)]  # Linear increase
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        result = analyzer._fit_linear_trend(df, forecast_days=3)
        
        assert result['model_type'] == 'linear'
        assert result['trend_direction'] == 'increasing'
        assert result['fit_score'] > 0.9  # Should be high for perfect linear data
        assert 'forecast' in result
        assert len(result['forecast']['timestamps']) == 3
    
    def test_fit_exponential_trend(self, analyzer):
        """Test exponential trend fitting."""
        # Create sample data with exponential trend
        timestamps = [datetime.utcnow() - timedelta(days=i) for i in range(10, 0, -1)]
        values = [np.exp(i * 0.1) for i in range(10)]  # Exponential growth
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        result = analyzer._fit_exponential_trend(df, forecast_days=3)
        
        assert result['model_type'] == 'exponential'
        assert result['trend_direction'] == 'exponential_growth'
        assert result['fit_score'] > 0.8
        assert 'forecast' in result
    
    def test_determine_polynomial_trend(self, analyzer):
        """Test polynomial trend determination."""
        # Test quadratic with positive leading coefficient (accelerating upward)
        coeffs = np.array([1.0, 0.0, 0.0])  # x^2
        trend = analyzer._determine_polynomial_trend(coeffs)
        assert trend == "accelerating_upward"
        
        # Test quadratic with negative leading coefficient (accelerating downward)
        coeffs = np.array([-1.0, 0.0, 0.0])  # -x^2
        trend = analyzer._determine_polynomial_trend(coeffs)
        assert trend == "accelerating_downward"
        
        # Test linear (no quadratic term)
        coeffs = np.array([0.0, 1.0, 0.0])  # x
        trend = analyzer._determine_polynomial_trend(coeffs)
        assert trend == "linear"

class TestAdvancedPatternRecognizer:
    """Test advanced pattern recognition."""
    
    @pytest.fixture
    def recognizer(self):
        return AdvancedPatternRecognizer()
    
    def test_detect_spikes(self, recognizer):
        """Test spike detection."""
        # Create data with clear spike
        base_values = [1.0] * 10
        base_values[5] = 5.0  # Spike at index 5
        
        timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(10)]
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': base_values,
            'prompt_id': ['prompt123'] * 10,
            'metric_name': ['accuracy_score'] * 10
        })
        
        patterns = recognizer._detect_spikes(df, "prompt123", "accuracy_score")
        
        assert len(patterns) > 0
        spike_pattern = patterns[0]
        assert spike_pattern.pattern_type == PatternType.SPIKE
        assert spike_pattern.confidence_score > 0.5
        assert 'spike_magnitude' in spike_pattern.parameters
    
    def test_detect_drops(self, recognizer):
        """Test drop detection."""
        # Create data with clear drop
        base_values = [5.0] * 10
        base_values[5] = 1.0  # Drop at index 5
        
        timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(10)]
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': base_values,
            'prompt_id': ['prompt123'] * 10,
            'metric_name': ['accuracy_score'] * 10
        })
        
        patterns = recognizer._detect_drops(df, "prompt123", "accuracy_score")
        
        assert len(patterns) > 0
        drop_pattern = patterns[0]
        assert drop_pattern.pattern_type == PatternType.DROP
        assert drop_pattern.confidence_score > 0.5
        assert 'drop_magnitude' in drop_pattern.parameters
    
    def test_detect_plateaus(self, recognizer):
        """Test plateau detection."""
        # Create data with plateau
        values = [1.0, 1.1, 0.9] + [2.0] * 8 + [1.8, 2.1, 1.9]  # Plateau in middle
        
        timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(len(values))]
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'prompt_id': ['prompt123'] * len(values),
            'metric_name': ['accuracy_score'] * len(values)
        })
        
        patterns = recognizer._detect_plateaus(df, "prompt123", "accuracy_score")
        
        # Should detect the stable period in the middle
        if len(patterns) > 0:
            plateau_pattern = patterns[0]
            assert plateau_pattern.pattern_type == PatternType.PLATEAU
            assert plateau_pattern.confidence_score > 0.5
            assert 'plateau_value' in plateau_pattern.parameters

if __name__ == "__main__":
    pytest.main([__file__])