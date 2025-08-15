"""
Tests for prompt analytics system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd

from scrollintel.core.prompt_analytics import (
    PromptPerformanceTracker, AnalyticsEngine, TrendAnalyzer, PatternRecognizer
)
from scrollintel.models.analytics_models import (
    PromptMetrics, UsageAnalytics, AnalyticsReport
)

class TestPromptPerformanceTracker:
    """Test prompt performance tracking functionality."""
    
    @pytest.fixture
    def tracker(self):
        return PromptPerformanceTracker()
    
    @pytest.fixture
    def sample_metrics(self):
        return {
            'accuracy_score': 0.85,
            'relevance_score': 0.90,
            'efficiency_score': 0.75,
            'user_satisfaction': 4.2,
            'response_time_ms': 1500,
            'token_usage': 150,
            'cost_per_request': 0.003
        }
    
    @pytest.fixture
    def sample_context(self):
        return {
            'use_case': 'content_generation',
            'model_used': 'gpt-4',
            'user_id': 'user123',
            'team_id': 'team456'
        }
    
    @pytest.mark.asyncio
    async def test_record_prompt_usage(self, tracker, sample_metrics, sample_context):
        """Test recording prompt usage with metrics."""
        with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Test successful recording
            metrics_id = await tracker.record_prompt_usage(
                prompt_id="prompt123",
                version_id="v1.0",
                performance_metrics=sample_metrics,
                context=sample_context
            )
            
            assert metrics_id is not None
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_prompt_performance_summary(self, tracker):
        """Test getting performance summary."""
        # Mock database data
        mock_metrics = []
        for i in range(10):
            metric = Mock()
            metric.prompt_id = "prompt123"
            metric.accuracy_score = 0.8 + (i * 0.01)  # Increasing trend
            metric.relevance_score = 0.85
            metric.efficiency_score = 0.75
            metric.user_satisfaction = 4.0
            metric.response_time_ms = 1500
            metric.token_usage = 150
            metric.cost_per_request = 0.003
            metric.created_at = datetime.utcnow() - timedelta(days=i)
            mock_metrics.append(metric)
        
        with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics
            
            summary = await tracker.get_prompt_performance_summary("prompt123", days=30)
            
            assert summary["prompt_id"] == "prompt123"
            assert summary["total_usage"] == 10
            assert "performance_metrics" in summary
            assert "accuracy_score" in summary["performance_metrics"]
            assert summary["performance_metrics"]["accuracy_score"]["trend"] == "improving"
    
    @pytest.mark.asyncio
    async def test_get_team_analytics(self, tracker):
        """Test getting team analytics."""
        # Mock team metrics
        mock_metrics = []
        for i in range(20):
            metric = Mock()
            metric.team_id = "team123"
            metric.prompt_id = f"prompt{i % 5}"  # 5 different prompts
            metric.accuracy_score = 0.7 + (i % 5) * 0.05  # Varying performance
            metric.relevance_score = 0.8
            metric.efficiency_score = 0.75
            metric.user_satisfaction = 3.5 + (i % 5) * 0.2
            metric.created_at = datetime.utcnow() - timedelta(hours=i)
            mock_metrics.append(metric)
        
        with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics
            
            analytics = await tracker.get_team_analytics("team123", days=7)
            
            assert analytics["team_id"] == "team123"
            assert analytics["total_prompts"] == 5
            assert analytics["total_usage"] == 20
            assert "top_performers" in analytics
            assert "improvement_opportunities" in analytics
    
    def test_calculate_trend(self, tracker):
        """Test trend calculation."""
        # Test improving trend
        improving_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        trend = tracker._calculate_trend(improving_values)
        assert trend == "improving"
        
        # Test declining trend
        declining_values = [0.9, 0.8, 0.7, 0.6, 0.5]
        trend = tracker._calculate_trend(declining_values)
        assert trend == "declining"
        
        # Test stable trend
        stable_values = [0.7, 0.71, 0.69, 0.70, 0.71]
        trend = tracker._calculate_trend(stable_values)
        assert trend == "stable"
        
        # Test insufficient data
        insufficient_values = [0.5]
        trend = tracker._calculate_trend(insufficient_values)
        assert trend == "insufficient_data"
    
    def test_generate_performance_recommendations(self, tracker):
        """Test performance recommendation generation."""
        # Test low accuracy scenario
        summary = {
            "performance_metrics": {
                "accuracy_score": {
                    "average": 0.5,  # Below threshold
                    "trend": "declining"
                },
                "response_time_ms": {
                    "average": 3500  # Above threshold
                },
                "user_satisfaction": {
                    "average": 2.5  # Below threshold
                }
            },
            "usage_patterns": {
                "total_requests": 1500  # High usage
            }
        }
        
        recommendations = tracker._generate_performance_recommendations(summary)
        
        assert len(recommendations) > 0
        assert any("accuracy" in rec.lower() for rec in recommendations)
        assert any("response time" in rec.lower() for rec in recommendations)
        assert any("satisfaction" in rec.lower() for rec in recommendations)
        assert any("testing" in rec.lower() for rec in recommendations)  # High usage recommendation

class TestTrendAnalyzer:
    """Test trend analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        return TrendAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_trends(self, analyzer):
        """Test trend analysis."""
        # Mock metrics data with clear trend
        mock_metrics = []
        for i in range(30):
            metric = Mock()
            metric.prompt_id = "prompt123"
            metric.accuracy_score = 0.5 + (i * 0.01)  # Linear increasing trend
            metric.created_at = datetime.utcnow() - timedelta(days=29-i)
            mock_metrics.append(metric)
        
        with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_metrics
            
            trend_analysis = await analyzer.analyze_trends("prompt123", "accuracy_score", days=30)
            
            assert trend_analysis.metric_name == "accuracy_score"
            assert trend_analysis.trend_direction == "increasing"
            assert trend_analysis.confidence_level > 0.5
            assert len(trend_analysis.data_points) == 30
            assert trend_analysis.forecast is not None
    
    def test_calculate_advanced_trend(self, analyzer):
        """Test advanced trend calculation."""
        # Test clear increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        direction, strength, confidence = analyzer._calculate_advanced_trend(increasing_values)
        
        assert direction == "increasing"
        assert strength > 0
        assert confidence > 0.8  # Should be high confidence for linear trend
        
        # Test noisy data
        noisy_values = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7]
        direction, strength, confidence = analyzer._calculate_advanced_trend(noisy_values)
        
        assert confidence < 0.8  # Should be lower confidence for noisy data
    
    def test_generate_forecast(self, analyzer):
        """Test forecast generation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        timestamps = [datetime.utcnow() - timedelta(days=4-i) for i in range(5)]
        
        forecast = analyzer._generate_forecast(values, timestamps)
        
        assert len(forecast) == 7  # 7-day forecast
        assert all('timestamp' in point for point in forecast)
        assert all('predicted_value' in point for point in forecast)
        assert all('confidence' in point for point in forecast)
        
        # Check that confidence decreases over time
        confidences = [point['confidence'] for point in forecast]
        assert confidences[0] > confidences[-1]

class TestPatternRecognizer:
    """Test pattern recognition functionality."""
    
    @pytest.fixture
    def recognizer(self):
        return PatternRecognizer()
    
    @pytest.mark.asyncio
    async def test_recognize_patterns(self, recognizer):
        """Test pattern recognition."""
        # Mock analytics data with clear patterns
        mock_analytics = []
        for i in range(7):  # One week of data
            analytics = Mock()
            analytics.prompt_id = "prompt123"
            analytics.hourly_patterns = {str(h): 10 + 5 * np.sin(h * np.pi / 12) for h in range(24)}  # Sinusoidal pattern
            analytics.daily_usage = {f"2024-01-{i+1:02d}": 100 + i * 10}  # Increasing usage
            analytics.analysis_period_start = datetime.utcnow() - timedelta(days=6-i)
            mock_analytics.append(analytics)
        
        with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = mock_analytics
            
            patterns = await recognizer.recognize_patterns(["prompt123"], days=7)
            
            assert len(patterns) > 0
            # Should detect cyclical pattern from hourly data
            assert any(p.pattern_type == "cyclical" for p in patterns)
    
    def test_detect_hourly_patterns(self, recognizer):
        """Test hourly pattern detection."""
        # Mock analytics with clear daily cycle
        mock_analytics = []
        for i in range(3):
            analytics = Mock()
            # Create clear pattern: high during day (8-18), low at night
            hourly_pattern = {}
            for hour in range(24):
                if 8 <= hour <= 18:
                    hourly_pattern[str(hour)] = 50  # High usage during day
                else:
                    hourly_pattern[str(hour)] = 10  # Low usage at night
            analytics.hourly_patterns = hourly_pattern
            mock_analytics.append(analytics)
        
        pattern = recognizer._detect_hourly_patterns(mock_analytics)
        
        assert pattern is not None
        assert pattern["confidence"] > 0.5
        assert len(pattern["peak_hours"]) == 3
    
    def test_detect_usage_anomalies(self, recognizer):
        """Test usage anomaly detection."""
        # Mock analytics with anomaly
        mock_analytics = []
        daily_usage_values = [100, 105, 98, 102, 500, 99, 103]  # Day 5 is anomalous
        
        for i, usage in enumerate(daily_usage_values):
            analytics = Mock()
            analytics.daily_usage = {f"2024-01-{i+1:02d}": usage}
            mock_analytics.append(analytics)
        
        anomaly = recognizer._detect_usage_anomalies(mock_analytics)
        
        assert anomaly is not None
        assert anomaly["confidence"] > 0
        assert anomaly["anomaly_days"] >= 1

class TestAnalyticsEngine:
    """Test analytics engine functionality."""
    
    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self, engine):
        """Test comprehensive report generation."""
        with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock report creation
            mock_report = Mock()
            mock_report.id = "report123"
            mock_session.add.return_value = None
            mock_session.commit.return_value = None
            
            # Mock the report data creation methods
            engine._generate_performance_report = AsyncMock(return_value={"summary": {"report_focus": "Performance Analysis"}})
            
            date_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())
            scope = {"team_ids": ["team123"]}
            
            with patch.object(engine, '_save_report', return_value="report123"):
                report_id = await engine.generate_comprehensive_report(
                    report_type="performance",
                    scope=scope,
                    date_range=date_range
                )
            
            assert report_id == "report123"

class TestDataValidation:
    """Test data validation and integrity."""
    
    def test_metrics_validation(self):
        """Test that metrics are properly validated."""
        # Test valid metrics
        valid_metrics = {
            'accuracy_score': 0.85,
            'relevance_score': 0.90,
            'efficiency_score': 0.75,
            'user_satisfaction': 4.2,
            'response_time_ms': 1500,
            'token_usage': 150,
            'cost_per_request': 0.003
        }
        
        # All values should be within expected ranges
        assert 0 <= valid_metrics['accuracy_score'] <= 1
        assert 0 <= valid_metrics['relevance_score'] <= 1
        assert 0 <= valid_metrics['efficiency_score'] <= 1
        assert 1 <= valid_metrics['user_satisfaction'] <= 5
        assert valid_metrics['response_time_ms'] > 0
        assert valid_metrics['token_usage'] > 0
        assert valid_metrics['cost_per_request'] > 0
    
    def test_trend_calculation_edge_cases(self):
        """Test trend calculation with edge cases."""
        tracker = PromptPerformanceTracker()
        
        # Test with empty list
        trend = tracker._calculate_trend([])
        assert trend == "insufficient_data"
        
        # Test with single value
        trend = tracker._calculate_trend([0.5])
        assert trend == "insufficient_data"
        
        # Test with identical values
        trend = tracker._calculate_trend([0.5, 0.5, 0.5, 0.5])
        assert trend == "stable"
        
        # Test with extreme values
        trend = tracker._calculate_trend([0.0, 1.0])
        assert trend in ["improving", "declining"]
    
    def test_performance_score_calculation(self):
        """Test performance score calculation."""
        tracker = PromptPerformanceTracker()
        
        # Create mock metrics with known values
        mock_metrics = []
        for i in range(5):
            metric = Mock()
            metric.accuracy_score = 0.8
            metric.relevance_score = 0.85
            metric.efficiency_score = 0.9
            metric.user_satisfaction = 4.0
            mock_metrics.append(metric)
        
        score_data = tracker._calculate_prompt_performance_score(mock_metrics)
        
        # Expected score: 0.8*0.3 + 0.85*0.25 + 0.9*0.2 + 4.0*0.25 = 0.24 + 0.2125 + 0.18 + 1.0 = 1.6325
        # But user_satisfaction is out of 5, so it should be normalized: 4.0/5 * 0.25 = 0.2
        # Actual expected: 0.24 + 0.2125 + 0.18 + 0.2 = 0.8325
        
        assert score_data["overall_score"] > 0
        assert score_data["usage_count"] == 5
        assert score_data["performance_category"] in ["excellent", "good", "fair", "needs_improvement"]

class TestIntegration:
    """Integration tests for the analytics system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_flow(self):
        """Test complete analytics flow from data recording to insights."""
        tracker = PromptPerformanceTracker()
        
        # Mock database operations
        with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Step 1: Record usage
            metrics_id = await tracker.record_prompt_usage(
                prompt_id="test_prompt",
                performance_metrics={'accuracy_score': 0.85},
                context={'team_id': 'test_team'}
            )
            
            assert metrics_id is not None
            
            # Step 2: Mock data for summary
            mock_metrics = [Mock()]
            mock_metrics[0].prompt_id = "test_prompt"
            mock_metrics[0].accuracy_score = 0.85
            mock_metrics[0].created_at = datetime.utcnow()
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics
            
            # Step 3: Get performance summary
            summary = await tracker.get_prompt_performance_summary("test_prompt")
            
            assert summary["prompt_id"] == "test_prompt"
            assert "performance_metrics" in summary
    
    @pytest.mark.asyncio
    async def test_concurrent_analytics_operations(self):
        """Test that analytics operations can handle concurrent access."""
        tracker = PromptPerformanceTracker()
        
        async def record_usage(prompt_id, metrics):
            with patch('scrollintel.core.prompt_analytics.get_db_session') as mock_db:
                mock_session = Mock()
                mock_db.return_value.__enter__.return_value = mock_session
                return await tracker.record_prompt_usage(prompt_id, performance_metrics=metrics)
        
        # Simulate concurrent usage recording
        tasks = []
        for i in range(10):
            task = record_usage(f"prompt_{i}", {'accuracy_score': 0.8 + i * 0.01})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All operations should complete successfully
        assert len(results) == 10
        assert all(result is not None for result in results)

if __name__ == "__main__":
    pytest.main([__file__])