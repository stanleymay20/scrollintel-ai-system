"""
Tests for analytics dashboard system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from scrollintel.core.analytics_dashboard import (
    TeamAnalyticsDashboard,
    InsightsGenerator,
    DashboardWidget,
    TeamDashboard
)

class TestTeamAnalyticsDashboard:
    """Test team analytics dashboard functionality."""
    
    @pytest.fixture
    def dashboard(self):
        """Create a fresh dashboard instance for each test."""
        return TeamAnalyticsDashboard()
    
    @pytest.mark.asyncio
    async def test_get_team_dashboard_data_basic(self, dashboard):
        """Test basic team dashboard data retrieval."""
        # Mock the prompt performance tracker
        mock_analytics = {
            'team_id': 'test_team',
            'summary': {
                'total_requests': 150,
                'successful_requests': 142,
                'success_rate': 94.67,
                'unique_prompts': 8,
                'unique_users': 5,
                'avg_accuracy_score': 0.87,
                'avg_response_time_ms': 1800,
                'total_cost': 25.50
            },
            'top_prompts': [
                {'prompt_id': 'prompt_1', 'usage_count': 50},
                {'prompt_id': 'prompt_2', 'usage_count': 35}
            ],
            'top_users': [
                {'user_id': 'user_1', 'request_count': 45},
                {'user_id': 'user_2', 'request_count': 38}
            ],
            'daily_usage': {
                '2024-01-01': 25,
                '2024-01-02': 30,
                '2024-01-03': 28
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 3)
            
            dashboard_data = await dashboard.get_team_dashboard_data(
                team_id='test_team',
                date_range=(start_date, end_date)
            )
            
            assert 'error' not in dashboard_data
            assert dashboard_data['team_id'] == 'test_team'
            assert 'summary_metrics' in dashboard_data
            assert 'performance_trends' in dashboard_data
            assert 'usage_patterns' in dashboard_data
            assert 'cost_analysis' in dashboard_data
            assert 'quality_metrics' in dashboard_data
            assert 'user_activity' in dashboard_data
            assert 'alerts' in dashboard_data
            assert 'recommendations' in dashboard_data
    
    @pytest.mark.asyncio
    async def test_summary_metrics_calculation(self, dashboard):
        """Test summary metrics calculation with comparison."""
        # Mock current and previous period data
        current_analytics = {
            'summary': {
                'total_requests': 200,
                'success_rate': 95.0,
                'avg_accuracy_score': 0.88,
                'avg_response_time_ms': 1600,
                'total_cost': 30.0
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=current_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            metrics = await dashboard._get_summary_metrics(
                team_id='test_team',
                date_range=(start_date, end_date)
            )
            
            assert 'total_requests' in metrics
            assert 'success_rate' in metrics
            assert 'avg_accuracy' in metrics
            assert 'avg_response_time' in metrics
            assert 'total_cost' in metrics
            
            # Each metric should have value and change
            for metric_name, metric_data in metrics.items():
                assert 'value' in metric_data
                assert 'change' in metric_data
    
    @pytest.mark.asyncio
    async def test_performance_trends_generation(self, dashboard):
        """Test performance trends data generation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)
        
        trends = await dashboard._get_performance_trends(
            team_id='test_team',
            date_range=(start_date, end_date)
        )
        
        assert 'daily_trends' in trends
        assert 'trend_directions' in trends
        assert 'period_days' in trends
        
        # Should have data for each day
        assert len(trends['daily_trends']) == 7
        
        # Each daily trend should have required fields
        for daily_trend in trends['daily_trends']:
            assert 'date' in daily_trend
            assert 'requests' in daily_trend
            assert 'success_rate' in daily_trend
            assert 'avg_accuracy' in daily_trend
            assert 'avg_response_time' in daily_trend
    
    @pytest.mark.asyncio
    async def test_usage_patterns_analysis(self, dashboard):
        """Test usage patterns analysis."""
        mock_analytics = {
            'daily_usage': {
                '2024-01-01': 25,  # Monday
                '2024-01-02': 30,  # Tuesday
                '2024-01-03': 28,  # Wednesday
                '2024-01-04': 35,  # Thursday
                '2024-01-05': 40,  # Friday
                '2024-01-06': 15,  # Saturday
                '2024-01-07': 12   # Sunday
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            patterns = await dashboard._get_usage_patterns(
                team_id='test_team',
                date_range=(start_date, end_date)
            )
            
            assert 'daily_usage' in patterns
            assert 'peak_day' in patterns
            assert 'avg_daily_usage' in patterns
            assert 'weekday_patterns' in patterns
            
            # Peak day should be Friday (40 requests)
            assert patterns['peak_day']['date'] == '2024-01-05'
            assert patterns['peak_day']['requests'] == 40
    
    @pytest.mark.asyncio
    async def test_cost_analysis_generation(self, dashboard):
        """Test cost analysis generation."""
        mock_analytics = {
            'summary': {
                'total_cost': 125.50,
                'total_requests': 500
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 30)
            
            cost_analysis = await dashboard._get_cost_analysis(
                team_id='test_team',
                date_range=(start_date, end_date)
            )
            
            assert 'total_cost' in cost_analysis
            assert 'avg_cost_per_request' in cost_analysis
            assert 'daily_cost_breakdown' in cost_analysis
            assert 'cost_optimization_opportunities' in cost_analysis
            
            # Should calculate correct average cost per request
            expected_avg_cost = 125.50 / 500
            assert cost_analysis['avg_cost_per_request'] == expected_avg_cost
            
            # Should suggest optimizations for high cost
            assert len(cost_analysis['cost_optimization_opportunities']) > 0
    
    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, dashboard):
        """Test quality metrics calculation."""
        mock_analytics = {
            'summary': {
                'avg_accuracy_score': 0.85,
                'success_rate': 92.0
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            quality_metrics = await dashboard._get_quality_metrics(
                team_id='test_team',
                date_range=(start_date, end_date)
            )
            
            assert 'overall_quality_score' in quality_metrics
            assert 'accuracy_distribution' in quality_metrics
            assert 'quality_trends' in quality_metrics
            assert 'improvement_suggestions' in quality_metrics
            
            # Quality score should be calculated correctly
            # (0.85 * 0.6 + 0.92 * 0.4) * 100 = 87.8
            expected_quality_score = (0.85 * 0.6 + 0.92 * 0.4) * 100
            assert abs(quality_metrics['overall_quality_score'] - expected_quality_score) < 0.1
    
    @pytest.mark.asyncio
    async def test_user_activity_analysis(self, dashboard):
        """Test user activity analysis."""
        mock_analytics = {
            'top_users': [
                {'user_id': 'user_1', 'request_count': 50},
                {'user_id': 'user_2', 'request_count': 35},
                {'user_id': 'user_3', 'request_count': 25}
            ],
            'summary': {
                'unique_users': 3
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            user_activity = await dashboard._get_user_activity(
                team_id='test_team',
                date_range=(start_date, end_date)
            )
            
            assert 'total_active_users' in user_activity
            assert 'top_users' in user_activity
            assert 'user_engagement_score' in user_activity
            
            assert user_activity['total_active_users'] == 3
            assert len(user_activity['top_users']) == 3
            
            # Engagement score should be average requests per user
            expected_engagement = (50 + 35 + 25) / 3
            assert abs(user_activity['user_engagement_score'] - expected_engagement) < 0.1
    
    @pytest.mark.asyncio
    async def test_active_alerts_retrieval(self, dashboard):
        """Test active alerts retrieval."""
        # Mock low success rate to trigger alert
        mock_analytics = {
            'summary': {
                'success_rate': 85.0  # Below 95% threshold
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            alerts = await dashboard._get_active_alerts('test_team')
            
            assert isinstance(alerts, list)
            
            # Should have at least one alert for low success rate
            assert len(alerts) > 0
            
            # Check alert structure
            alert = alerts[0]
            assert 'alert_id' in alert
            assert 'type' in alert
            assert 'severity' in alert
            assert 'title' in alert
            assert 'message' in alert
            assert 'triggered_at' in alert
            assert 'status' in alert
    
    @pytest.mark.asyncio
    async def test_recommendations_generation(self, dashboard):
        """Test recommendations generation."""
        mock_analytics = {
            'summary': {
                'success_rate': 88.0,  # Below optimal
                'total_cost': 750.0,   # High cost
                'total_requests': 1000
            },
            'top_prompts': [
                {'prompt_id': 'dominant_prompt', 'usage_count': 600}  # 60% of requests
            ]
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 30)
            
            recommendations = await dashboard._generate_recommendations(
                team_id='test_team',
                date_range=(start_date, end_date)
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            
            # Should have recommendations for performance, cost, and usage
            recommendation_types = [rec['type'] for rec in recommendations]
            assert 'performance' in recommendation_types
            assert 'cost' in recommendation_types
            assert 'usage' in recommendation_types
            
            # Each recommendation should have required fields
            for rec in recommendations:
                assert 'type' in rec
                assert 'priority' in rec
                assert 'title' in rec
                assert 'description' in rec
                assert 'action_items' in rec
    
    @pytest.mark.asyncio
    async def test_error_handling(self, dashboard):
        """Test error handling in dashboard functions."""
        # Mock error from analytics tracker
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value={'error': 'Team not found'})
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            dashboard_data = await dashboard.get_team_dashboard_data(
                team_id='nonexistent_team',
                date_range=(start_date, end_date)
            )
            
            assert 'error' in dashboard_data

class TestInsightsGenerator:
    """Test insights generator functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create a fresh insights generator for each test."""
        return InsightsGenerator()
    
    @pytest.mark.asyncio
    async def test_generate_insights_positive_performance(self, generator):
        """Test insights generation for positive performance."""
        mock_analytics = {
            'summary': {
                'success_rate': 99.2,  # Excellent
                'total_requests': 500,
                'unique_prompts': 10,
                'unique_users': 8,
                'total_cost': 25.0
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            insights = await generator.generate_insights(
                team_id='high_performing_team',
                date_range=(start_date, end_date)
            )
            
            assert isinstance(insights, list)
            assert len(insights) > 0
            
            # Should have positive insights for excellent performance
            positive_insights = [i for i in insights if i['category'] == 'positive']
            assert len(positive_insights) > 0
            
            # Check insight structure
            insight = insights[0]
            assert 'type' in insight
            assert 'category' in insight
            assert 'title' in insight
            assert 'description' in insight
            assert 'impact' in insight
            assert 'confidence' in insight
    
    @pytest.mark.asyncio
    async def test_generate_insights_performance_concerns(self, generator):
        """Test insights generation for performance concerns."""
        mock_analytics = {
            'summary': {
                'success_rate': 82.0,  # Below optimal
                'total_requests': 200,
                'unique_prompts': 5,
                'unique_users': 2,  # Low collaboration
                'total_cost': 150.0  # High cost per request
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            insights = await generator.generate_insights(
                team_id='struggling_team',
                date_range=(start_date, end_date)
            )
            
            assert isinstance(insights, list)
            
            # Should have concern insights for poor performance
            concern_insights = [i for i in insights if i['category'] == 'concern']
            assert len(concern_insights) > 0
            
            # Should identify reliability issues
            reliability_insights = [i for i in insights if 'reliability' in i['title'].lower()]
            assert len(reliability_insights) > 0
    
    @pytest.mark.asyncio
    async def test_generate_insights_optimization_opportunities(self, generator):
        """Test insights generation for optimization opportunities."""
        mock_analytics = {
            'summary': {
                'success_rate': 94.0,
                'total_requests': 1000,
                'unique_prompts': 5,  # High utilization
                'unique_users': 12,   # Good collaboration
                'total_cost': 5.0     # Very cost efficient
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 30)
            
            insights = await generator.generate_insights(
                team_id='efficient_team',
                date_range=(start_date, end_date)
            )
            
            assert isinstance(insights, list)
            
            # Should have optimization and positive insights
            optimization_insights = [i for i in insights if i['category'] == 'optimization']
            positive_insights = [i for i in insights if i['category'] == 'positive']
            
            assert len(optimization_insights) > 0 or len(positive_insights) > 0
    
    @pytest.mark.asyncio
    async def test_generate_insights_no_data(self, generator):
        """Test insights generation when no data is available."""
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value={'error': 'No data found'})
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            insights = await generator.generate_insights(
                team_id='empty_team',
                date_range=(start_date, end_date)
            )
            
            assert isinstance(insights, list)
            assert len(insights) == 0

class TestDashboardDataStructures:
    """Test dashboard data structures."""
    
    def test_dashboard_widget_creation(self):
        """Test dashboard widget creation."""
        widget = DashboardWidget(
            widget_id="widget_001",
            widget_type="chart",
            title="Usage Trends",
            data_source="usage_analytics",
            configuration={
                "chart_type": "line",
                "metrics": ["requests", "success_rate"]
            },
            position={"x": 0, "y": 0, "width": 6, "height": 4}
        )
        
        assert widget.widget_id == "widget_001"
        assert widget.widget_type == "chart"
        assert widget.title == "Usage Trends"
        assert widget.refresh_interval == 300  # Default value
    
    def test_team_dashboard_creation(self):
        """Test team dashboard creation."""
        widgets = [
            DashboardWidget(
                widget_id="widget_001",
                widget_type="metric",
                title="Total Requests",
                data_source="summary_metrics",
                configuration={"metric": "total_requests"},
                position={"x": 0, "y": 0, "width": 3, "height": 2}
            )
        ]
        
        dashboard = TeamDashboard(
            dashboard_id="dashboard_001",
            team_id="team_123",
            name="Team Performance Dashboard",
            description="Main dashboard for team performance monitoring",
            widgets=widgets,
            layout={"columns": 12, "rows": 8},
            created_by="admin",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert dashboard.dashboard_id == "dashboard_001"
        assert dashboard.team_id == "team_123"
        assert len(dashboard.widgets) == 1
        assert dashboard.widgets[0].widget_id == "widget_001"

class TestIntegration:
    """Integration tests for analytics dashboard."""
    
    @pytest.mark.asyncio
    async def test_complete_dashboard_workflow(self):
        """Test complete dashboard data generation workflow."""
        dashboard = TeamAnalyticsDashboard()
        generator = InsightsGenerator()
        
        # Mock comprehensive analytics data
        mock_analytics = {
            'team_id': 'integration_team',
            'summary': {
                'total_requests': 300,
                'successful_requests': 285,
                'success_rate': 95.0,
                'unique_prompts': 6,
                'unique_users': 4,
                'avg_accuracy_score': 0.89,
                'avg_response_time_ms': 1400,
                'total_cost': 45.0
            },
            'top_prompts': [
                {'prompt_id': 'prompt_1', 'usage_count': 120},
                {'prompt_id': 'prompt_2', 'usage_count': 80}
            ],
            'top_users': [
                {'user_id': 'user_1', 'request_count': 100},
                {'user_id': 'user_2', 'request_count': 75}
            ],
            'daily_usage': {
                '2024-01-01': 40,
                '2024-01-02': 45,
                '2024-01-03': 50,
                '2024-01-04': 55,
                '2024-01-05': 60,
                '2024-01-06': 25,
                '2024-01-07': 25
            }
        }
        
        with patch('scrollintel.core.analytics_dashboard.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_team_analytics = AsyncMock(return_value=mock_analytics)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 7)
            
            # Get complete dashboard data
            dashboard_data = await dashboard.get_team_dashboard_data(
                team_id='integration_team',
                date_range=(start_date, end_date)
            )
            
            # Verify all components are present
            assert 'error' not in dashboard_data
            assert dashboard_data['team_id'] == 'integration_team'
            
            # Verify summary metrics
            summary_metrics = dashboard_data['summary_metrics']
            assert 'total_requests' in summary_metrics
            assert summary_metrics['total_requests']['value'] == 300
            
            # Verify performance trends
            performance_trends = dashboard_data['performance_trends']
            assert len(performance_trends['daily_trends']) == 7
            
            # Verify usage patterns
            usage_patterns = dashboard_data['usage_patterns']
            assert usage_patterns['peak_day']['date'] == '2024-01-05'
            assert usage_patterns['peak_day']['requests'] == 60
            
            # Verify recommendations are generated
            recommendations = dashboard_data['recommendations']
            assert isinstance(recommendations, list)
            
            # Generate insights separately
            insights = await generator.generate_insights(
                team_id='integration_team',
                date_range=(start_date, end_date)
            )
            
            assert isinstance(insights, list)
            assert len(insights) > 0

if __name__ == "__main__":
    pytest.main([__file__])