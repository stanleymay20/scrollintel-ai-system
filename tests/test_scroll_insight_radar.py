"""
Unit tests for ScrollInsightRadar engine.

Tests pattern detection algorithms, trend analysis, anomaly detection,
insight generation, and statistical significance testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from scrollintel.engines.scroll_insight_radar import ScrollInsightRadar
from scrollintel.models.schemas import PatternDetectionConfig


class TestScrollInsightRadar:
    """Test suite for ScrollInsightRadar engine."""
    
    @pytest.fixture
    def insight_radar(self):
        """Create ScrollInsightRadar instance for testing."""
        return ScrollInsightRadar()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = pd.DataFrame({
            'date': dates,
            'sales': np.random.normal(1000, 200, 100) + np.sin(np.arange(100) * 0.1) * 100,
            'marketing_spend': np.random.normal(500, 100, 100),
            'temperature': np.random.normal(20, 5, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'outlier_column': np.concatenate([np.random.normal(10, 2, 95), [50, 60, 70, 80, 90]])
        })
        
        # Add correlation between sales and marketing_spend
        data['marketing_spend'] = data['sales'] * 0.3 + np.random.normal(0, 50, 100)
        
        return data
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data with trend and seasonality."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        
        # Create trend + seasonality + noise
        trend = np.linspace(100, 200, 365)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 30)  # Monthly seasonality
        noise = np.random.normal(0, 10, 365)
        
        data = pd.DataFrame({
            'date': dates,
            'value': trend + seasonal + noise,
            'metric_2': np.random.normal(50, 10, 365)
        })
        
        return data
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PatternDetectionConfig(
            correlation_threshold=0.5,
            anomaly_contamination=0.1,
            significance_level=0.05
        )

    @pytest.mark.asyncio
    async def test_engine_initialization(self, insight_radar):
        """Test ScrollInsightRadar engine initialization."""
        assert insight_radar.name == "ScrollInsightRadar"
        assert insight_radar.version == "1.0.0"
        assert "pattern_detection" in insight_radar.capabilities
        assert "trend_analysis" in insight_radar.capabilities
        assert "anomaly_detection" in insight_radar.capabilities
        assert "insight_ranking" in insight_radar.capabilities
        assert "automated_notifications" in insight_radar.capabilities
        assert "statistical_significance_testing" in insight_radar.capabilities

    @pytest.mark.asyncio
    async def test_detect_patterns_comprehensive(self, insight_radar, sample_data, config):
        """Test comprehensive pattern detection."""
        results = await insight_radar.detect_patterns(sample_data, config)
        
        assert "timestamp" in results
        assert "dataset_info" in results
        assert "patterns" in results
        assert "trends" in results
        assert "anomalies" in results
        assert "insights" in results
        assert "statistical_tests" in results
        assert "business_impact_score" in results
        
        # Check dataset info
        dataset_info = results["dataset_info"]
        assert dataset_info["rows"] == 100
        assert dataset_info["columns"] == 6
        assert dataset_info["numeric_columns"] >= 3
        
        # Check business impact score is valid
        assert 0.0 <= results["business_impact_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_detect_correlation_patterns(self, insight_radar, sample_data):
        """Test correlation pattern detection."""
        results = await insight_radar._detect_correlation_patterns(sample_data)
        
        assert "correlation_matrix" in results
        assert "strong_correlations" in results
        assert "total_correlations_found" in results
        
        # Should find correlation between sales and marketing_spend
        strong_corrs = results["strong_correlations"]
        correlation_found = any(
            (corr["variable_1"] == "sales" and corr["variable_2"] == "marketing_spend") or
            (corr["variable_1"] == "marketing_spend" and corr["variable_2"] == "sales")
            for corr in strong_corrs
        )
        assert correlation_found

    @pytest.mark.asyncio
    async def test_detect_seasonal_patterns(self, insight_radar, time_series_data):
        """Test seasonal pattern detection."""
        results = await insight_radar._detect_seasonal_patterns(time_series_data)
        
        assert "seasonal_patterns" in results
        assert "patterns_found" in results
        
        if results["patterns_found"] > 0:
            pattern = results["seasonal_patterns"][0]
            assert "datetime_column" in pattern
            assert "numeric_column" in pattern
            assert "seasonal_strength" in pattern
            assert "trend_strength" in pattern
            assert "has_seasonality" in pattern
            assert "has_trend" in pattern

    @pytest.mark.asyncio
    async def test_detect_clustering_patterns(self, insight_radar, sample_data):
        """Test clustering pattern detection."""
        results = await insight_radar._detect_clustering_patterns(sample_data)
        
        if "error" not in results:
            assert "clusters_found" in results
            assert "cluster_details" in results
            assert "noise_points" in results
            assert "noise_percentage" in results
            
            if results["clusters_found"] > 0:
                cluster = results["cluster_details"][0]
                assert "cluster_id" in cluster
                assert "size" in cluster
                assert "percentage" in cluster
                assert "centroid" in cluster

    @pytest.mark.asyncio
    async def test_detect_distribution_patterns(self, insight_radar, sample_data):
        """Test distribution pattern detection."""
        results = await insight_radar._detect_distribution_patterns(sample_data)
        
        assert "distribution_patterns" in results
        assert "columns_analyzed" in results
        
        if results["columns_analyzed"] > 0:
            pattern = results["distribution_patterns"][0]
            assert "column" in pattern
            assert "mean" in pattern
            assert "median" in pattern
            assert "std" in pattern
            assert "skewness" in pattern
            assert "kurtosis" in pattern
            assert "is_normal" in pattern
            assert "outlier_count" in pattern
            assert "outlier_percentage" in pattern

    @pytest.mark.asyncio
    async def test_analyze_trends(self, insight_radar, time_series_data, config):
        """Test trend analysis with statistical significance."""
        results = await insight_radar._analyze_trends(time_series_data, config)
        
        if "error" not in results:
            assert "trend_analysis" in results
            assert "significant_trends" in results
            assert "total_trends_analyzed" in results
            
            if results["total_trends_analyzed"] > 0:
                trend = results["trend_analysis"][0]
                assert "datetime_column" in trend
                assert "numeric_column" in trend
                assert "slope" in trend
                assert "r_squared" in trend
                assert "p_value" in trend
                assert "trend_direction" in trend
                assert "is_significant" in trend
                assert "trend_strength" in trend
                assert "confidence_level" in trend
                
                assert trend["trend_direction"] in ["increasing", "decreasing"]
                assert trend["confidence_level"] in ["high", "medium", "low"]
                assert 0.0 <= trend["trend_strength"] <= 1.0

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, insight_radar, sample_data, config):
        """Test anomaly detection."""
        results = await insight_radar._detect_anomalies(sample_data, config)
        
        if "error" not in results:
            assert "isolation_forest_anomalies" in results
            assert "column_anomalies" in results
            assert "total_anomalies_found" in results
            assert "anomaly_percentage" in results
            
            # Should detect anomalies in outlier_column
            column_anomalies = results["column_anomalies"]
            if "outlier_column" in column_anomalies:
                outlier_stats = column_anomalies["outlier_column"]
                assert outlier_stats["total_anomalies"] > 0

    @pytest.mark.asyncio
    async def test_generate_insights(self, insight_radar, sample_data):
        """Test insight generation and ranking."""
        # First get analysis results
        analysis_results = await insight_radar.detect_patterns(sample_data)
        
        insights = analysis_results["insights"]
        
        # Check insights structure
        for insight in insights:
            assert "type" in insight
            assert "title" in insight
            assert "description" in insight
            assert "impact_score" in insight
            assert "actionable" in insight
            assert "category" in insight
            assert "rank" in insight
            assert "priority" in insight
            
            assert insight["type"] in ["correlation", "trend", "anomaly", "clustering"]
            assert insight["priority"] in ["high", "medium", "low"]
            assert 0.0 <= insight["impact_score"] <= 1.0
        
        # Check insights are ranked by impact score
        if len(insights) > 1:
            for i in range(len(insights) - 1):
                assert insights[i]["impact_score"] >= insights[i + 1]["impact_score"]

    @pytest.mark.asyncio
    async def test_calculate_business_impact(self, insight_radar):
        """Test business impact calculation."""
        mock_results = {
            "insights": [
                {"type": "correlation", "impact_score": 0.8},
                {"type": "trend", "impact_score": 0.9},
                {"type": "anomaly", "impact_score": 0.6},
                {"type": "clustering", "impact_score": 0.7}
            ]
        }
        
        impact_score = await insight_radar._calculate_business_impact(mock_results)
        
        assert 0.0 <= impact_score <= 1.0
        assert impact_score > 0  # Should have some impact with these insights

    @pytest.mark.asyncio
    async def test_perform_statistical_tests(self, insight_radar, sample_data):
        """Test statistical significance testing."""
        results = await insight_radar._perform_statistical_tests(sample_data)
        
        if "error" not in results:
            assert "correlation_tests" in results
            assert "normality_tests" in results
            
            # Check correlation tests
            if results["correlation_tests"]:
                corr_test = results["correlation_tests"][0]
                assert "variable_1" in corr_test
                assert "variable_2" in corr_test
                assert "correlation" in corr_test
                assert "p_value" in corr_test
                assert "is_significant" in corr_test
                
                assert -1.0 <= corr_test["correlation"] <= 1.0
                assert 0.0 <= corr_test["p_value"] <= 1.0
            
            # Check normality tests
            if results["normality_tests"]:
                norm_test = results["normality_tests"][0]
                assert "column" in norm_test
                assert "p_value" in norm_test
                assert "is_normal" in norm_test
                
                assert 0.0 <= norm_test["p_value"] <= 1.0

    @pytest.mark.asyncio
    async def test_send_insight_notification(self, insight_radar):
        """Test insight notification system."""
        insights = [
            {"priority": "high", "title": "High priority insight"},
            {"priority": "medium", "title": "Medium priority insight"},
            {"priority": "low", "title": "Low priority insight"}
        ]
        
        result = await insight_radar.send_insight_notification(insights, "test_user")
        assert result is True  # Should return True for high priority insights
        
        # Test with no high priority insights
        low_priority_insights = [{"priority": "low", "title": "Low priority insight"}]
        result = await insight_radar.send_insight_notification(low_priority_insights, "test_user")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_health_status(self, insight_radar):
        """Test health status endpoint."""
        health = await insight_radar.get_health_status()
        
        assert health["status"] == "healthy"
        assert health["engine"] == "ScrollInsightRadar"
        assert health["version"] == "1.0.0"
        assert health["capabilities"] == insight_radar.capabilities
        assert "last_check" in health

    @pytest.mark.asyncio
    async def test_get_dataset_info(self, insight_radar, sample_data):
        """Test dataset information extraction."""
        info = insight_radar._get_dataset_info(sample_data)
        
        assert info["rows"] == 100
        assert info["columns"] == 6
        assert info["numeric_columns"] >= 3
        assert info["categorical_columns"] >= 1
        assert info["missing_values"] >= 0
        assert "memory_usage" in info

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, insight_radar):
        """Test handling of empty or invalid data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):
            await insight_radar.detect_patterns(empty_data)

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, insight_radar):
        """Test handling of insufficient data."""
        small_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        results = await insight_radar.detect_patterns(small_data)
        
        # Should handle gracefully and return results with appropriate messages
        assert "dataset_info" in results
        assert results["dataset_info"]["rows"] == 3

    @pytest.mark.asyncio
    async def test_non_numeric_data_handling(self, insight_radar):
        """Test handling of non-numeric data."""
        text_data = pd.DataFrame({
            'text_col': ['a', 'b', 'c', 'd', 'e'],
            'category_col': ['cat1', 'cat2', 'cat1', 'cat2', 'cat1']
        })
        
        results = await insight_radar.detect_patterns(text_data)
        
        # Should handle gracefully
        assert "dataset_info" in results
        assert results["dataset_info"]["numeric_columns"] == 0

    @pytest.mark.asyncio
    async def test_missing_data_handling(self, insight_radar):
        """Test handling of data with missing values."""
        data_with_nulls = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [None, 2, 3, 4, None],
            'col3': [1, 2, 3, 4, 5]
        })
        
        results = await insight_radar.detect_patterns(data_with_nulls)
        
        assert "dataset_info" in results
        assert results["dataset_info"]["missing_values"] > 0

    @pytest.mark.asyncio
    async def test_configuration_validation(self, insight_radar, sample_data):
        """Test configuration parameter validation."""
        # Test with custom configuration
        custom_config = PatternDetectionConfig(
            correlation_threshold=0.8,
            anomaly_contamination=0.05,
            significance_level=0.01
        )
        
        results = await insight_radar.detect_patterns(sample_data, custom_config)
        
        assert "patterns" in results
        assert "trends" in results
        assert "anomalies" in results

    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, insight_radar):
        """Test handling of larger datasets."""
        # Create a larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 1000),
            'col2': np.random.normal(0, 1, 1000),
            'col3': np.random.normal(0, 1, 1000),
            'date': pd.date_range('2020-01-01', periods=1000, freq='D')
        })
        
        results = await insight_radar.detect_patterns(large_data)
        
        assert "dataset_info" in results
        assert results["dataset_info"]["rows"] == 1000

    @pytest.mark.asyncio
    async def test_error_handling_in_pattern_detection(self, insight_radar):
        """Test error handling in pattern detection methods."""
        # Test with problematic data that might cause errors
        problematic_data = pd.DataFrame({
            'inf_col': [1, 2, np.inf, 4, 5],
            'nan_col': [1, np.nan, np.nan, np.nan, 5]
        })
        
        # Should handle errors gracefully
        results = await insight_radar.detect_patterns(problematic_data)
        assert "dataset_info" in results

    def test_pattern_detection_config_defaults(self):
        """Test PatternDetectionConfig default values."""
        config = PatternDetectionConfig()
        
        assert config.correlation_threshold == 0.7
        assert config.anomaly_contamination == 0.1
        assert config.significance_level == 0.05
        assert config.clustering_eps == 0.5
        assert config.clustering_min_samples == 5

    @pytest.mark.asyncio
    async def test_concurrent_pattern_detection(self, insight_radar, sample_data):
        """Test concurrent pattern detection requests."""
        # Run multiple pattern detection tasks concurrently
        tasks = [
            insight_radar.detect_patterns(sample_data),
            insight_radar.detect_patterns(sample_data),
            insight_radar.detect_patterns(sample_data)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert "patterns" in result
            assert "insights" in result

    @pytest.mark.asyncio
    async def test_insight_ranking_consistency(self, insight_radar, sample_data):
        """Test that insight ranking is consistent."""
        results1 = await insight_radar.detect_patterns(sample_data)
        results2 = await insight_radar.detect_patterns(sample_data)
        
        insights1 = results1["insights"]
        insights2 = results2["insights"]
        
        # Rankings should be consistent for the same data
        if len(insights1) > 0 and len(insights2) > 0:
            assert insights1[0]["rank"] == insights2[0]["rank"]

    @pytest.mark.asyncio
    async def test_business_impact_score_calculation(self, insight_radar, sample_data):
        """Test business impact score calculation logic."""
        results = await insight_radar.detect_patterns(sample_data)
        
        impact_score = results["business_impact_score"]
        insights = results["insights"]
        
        # Impact score should correlate with insight quality
        if len(insights) > 0:
            high_impact_insights = [i for i in insights if i["impact_score"] > 0.7]
            if len(high_impact_insights) > 0:
                assert impact_score > 0.3  # Should have reasonable impact
        
        assert 0.0 <= impact_score <= 1.0