"""
Tests for trend analysis and pattern recognition system.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from scrollintel.core.trend_pattern_analysis import (
    AdvancedTrendAnalyzer,
    AdvancedPatternRecognizer,
    TrendType,
    PatternType,
    TrendAnalysis,
    Pattern
)

class TestAdvancedTrendAnalyzer:
    """Test advanced trend analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a fresh trend analyzer for each test."""
        return AdvancedTrendAnalyzer()
    
    @pytest.mark.asyncio
    async def test_comprehensive_trend_analysis_linear(self, analyzer):
        """Test comprehensive trend analysis with linear data."""
        # Mock performance summary with linear trend
        mock_summary = {
            'prompt_id': 'test_prompt',
            'daily_usage': {
                '2024-01-01': 10,
                '2024-01-02': 12,
                '2024-01-03': 14,
                '2024-01-04': 16,
                '2024-01-05': 18,
                '2024-01-06': 20,
                '2024-01-07': 22
            }
        }
        
        with patch('scrollintel.core.trend_pattern_analysis.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_prompt_performance_summary = AsyncMock(return_value=mock_summary)
            
            analysis = await analyzer.analyze_comprehensive_trends(
                prompt_id='test_prompt',
                metric_name='usage_count',
                days=7,
                forecast_days=3
            )
            
            assert 'error' not in analysis
            assert analysis['prompt_id'] == 'test_prompt'
            assert analysis['data_points'] == 7
            assert 'trend_analyses' in analysis
            assert 'best_fit' in analysis
            assert 'forecast' in analysis
            
            # Linear analysis should be present
            assert 'linear' in analysis['trend_analyses']
            linear_analysis = analysis['trend_analyses']['linear']
            assert linear_analysis['trend_direction'] == 'increasing'
            assert linear_analysis['r_squared'] > 0.9  # Should be very high for perfect linear data
    
    @pytest.mark.asyncio
    async def test_linear_trend_analysis(self, analyzer):
        """Test linear trend analysis specifically."""
        # Create perfect linear increasing data
        time_series = [
            {'date': f'2024-01-0{i+1}', 'day_index': i, 'value': 10 + i * 2}
            for i in range(7)
        ]
        
        analysis = await analyzer._analyze_linear_trend(time_series)
        
        assert 'error' not in analysis
        assert analysis['trend_type'] == TrendType.LINEAR.value
        assert analysis['trend_direction'] == 'increasing'
        assert analysis['slope'] > 0
        assert analysis['r_squared'] > 0.95  # Should be very high for perfect linear data
    
    @pytest.mark.asyncio
    async def test_linear_trend_analysis_decreasing(self, analyzer):
        """Test linear trend analysis with decreasing data."""
        # Create decreasing linear data
        time_series = [
            {'date': f'2024-01-0{i+1}', 'day_index': i, 'value': 20 - i * 2}
            for i in range(7)
        ]
        
        analysis = await analyzer._analyze_linear_trend(time_series)
        
        assert analysis['trend_direction'] == 'decreasing'
        assert analysis['slope'] < 0
    
    @pytest.mark.asyncio
    async def test_linear_trend_analysis_stable(self, analyzer):
        """Test linear trend analysis with stable data."""
        # Create stable data
        time_series = [
            {'date': f'2024-01-0{i+1}', 'day_index': i, 'value': 15 + (i % 2) * 0.1}
            for i in range(7)
        ]
        
        analysis = await analyzer._analyze_linear_trend(time_series)
        
        assert analysis['trend_direction'] == 'stable'
        assert abs(analysis['slope']) < 0.1
    
    @pytest.mark.asyncio
    async def test_polynomial_trend_analysis(self, analyzer):
        """Test polynomial trend analysis."""
        # Create quadratic data
        time_series = [
            {'date': f'2024-01-0{i+1}', 'day_index': i, 'value': i**2 + 5}
            for i in range(7)
        ]
        
        analysis = await analyzer._analyze_polynomial_trend(time_series)
        
        assert 'error' not in analysis
        assert analysis['trend_type'] == TrendType.POLYNOMIAL.value
        assert analysis['degree'] >= 2
        assert analysis['r_squared'] > 0.8  # Should fit well for quadratic data
    
    @pytest.mark.asyncio
    async def test_seasonal_pattern_analysis(self, analyzer):
        """Test seasonal pattern analysis."""
        # Create weekly seasonal data (14 days, 2 weeks)
        time_series = []
        base_values = [20, 25, 30, 35, 40, 15, 10]  # Weekly pattern
        
        for week in range(2):
            for day in range(7):
                time_series.append({
                    'date': f'2024-01-{week*7 + day + 1:02d}',
                    'day_index': week * 7 + day,
                    'value': base_values[day] + week * 2  # Slight growth
                })
        
        analysis = await analyzer._analyze_seasonal_pattern(time_series)
        
        assert 'error' not in analysis
        assert analysis['trend_type'] == TrendType.SEASONAL.value
        assert 'weekday_pattern' in analysis
        assert len(analysis['weekday_pattern']) == 7
        assert analysis['peak_day'] == 4  # Friday (index 4) has highest value (40)
        assert analysis['low_day'] == 6   # Sunday (index 6) has lowest value (10)
    
    @pytest.mark.asyncio
    async def test_forecast_generation_linear(self, analyzer):
        """Test forecast generation for linear trends."""
        time_series = [
            {'date': f'2024-01-0{i+1}', 'day_index': i, 'value': 10 + i * 3}
            for i in range(5)
        ]
        
        # Mock linear trend analysis
        trend_analysis = {
            'analysis_type': 'linear',
            'slope': 3.0,
            'intercept': 10.0,
            'confidence_level': 0.9
        }
        
        forecast = await analyzer._generate_forecast(
            time_series=time_series,
            trend_analysis=trend_analysis,
            forecast_days=3
        )
        
        assert len(forecast) == 3
        
        # Check forecast values (should continue linear trend)
        for i, forecast_point in enumerate(forecast):
            expected_value = 10.0 + (5 + i) * 3.0  # Continue the linear trend
            assert abs(forecast_point['predicted_value'] - expected_value) < 0.1
            assert forecast_point['confidence'] > 0
    
    @pytest.mark.asyncio
    async def test_forecast_generation_seasonal(self, analyzer):
        """Test forecast generation for seasonal patterns."""
        time_series = [
            {'date': f'2024-01-0{i+1}', 'day_index': i, 'value': 20 + i}
            for i in range(7)
        ]
        
        # Mock seasonal trend analysis
        trend_analysis = {
            'analysis_type': 'seasonal',
            'weekday_pattern': [20, 25, 30, 35, 40, 15, 10],
            'confidence_level': 0.8
        }
        
        forecast = await analyzer._generate_forecast(
            time_series=time_series,
            trend_analysis=trend_analysis,
            forecast_days=7
        )
        
        assert len(forecast) == 7
        
        # Forecast should follow the weekly pattern
        expected_pattern = [20, 25, 30, 35, 40, 15, 10]
        for i, forecast_point in enumerate(forecast):
            expected_value = expected_pattern[i]
            assert abs(forecast_point['predicted_value'] - expected_value) < 0.1
    
    @pytest.mark.asyncio
    async def test_trend_statistics_calculation(self, analyzer):
        """Test trend statistics calculation."""
        time_series = [
            {'date': f'2024-01-0{i+1}', 'day_index': i, 'value': value}
            for i, value in enumerate([10, 15, 12, 18, 20, 16, 22])
        ]
        
        stats = analyzer._calculate_trend_statistics(time_series)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std_dev' in stats
        assert 'min_value' in stats
        assert 'max_value' in stats
        assert 'range' in stats
        assert 'coefficient_of_variation' in stats
        
        values = [10, 15, 12, 18, 20, 16, 22]
        assert stats['mean'] == sum(values) / len(values)
        assert stats['min_value'] == min(values)
        assert stats['max_value'] == max(values)
        assert stats['range'] == max(values) - min(values)
    
    @pytest.mark.asyncio
    async def test_best_trend_analysis_selection(self, analyzer):
        """Test selection of best trend analysis."""
        analyses = {
            'linear': {
                'r_squared': 0.85,
                'confidence_level': 0.8,
                'trend_type': 'linear'
            },
            'polynomial': {
                'r_squared': 0.95,
                'confidence_level': 0.9,
                'trend_type': 'polynomial'
            },
            'seasonal': {
                'confidence_level': 0.7,
                'trend_type': 'seasonal'
            }
        }
        
        best_analysis = analyzer._select_best_trend_analysis(analyses)
        
        assert best_analysis is not None
        assert best_analysis['analysis_type'] == 'polynomial'  # Should have highest composite score
        assert 'composite_score' in best_analysis
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data scenarios."""
        # Mock performance summary with insufficient data
        mock_summary = {
            'prompt_id': 'test_prompt',
            'daily_usage': {
                '2024-01-01': 10,
                '2024-01-02': 12
            }
        }
        
        with patch('scrollintel.core.trend_pattern_analysis.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_prompt_performance_summary = AsyncMock(return_value=mock_summary)
            
            analysis = await analyzer.analyze_comprehensive_trends(
                prompt_id='test_prompt',
                metric_name='usage_count',
                days=7
            )
            
            assert 'error' in analysis
            assert 'Insufficient data points' in analysis['error']

class TestAdvancedPatternRecognizer:
    """Test advanced pattern recognition functionality."""
    
    @pytest.fixture
    def recognizer(self):
        """Create a fresh pattern recognizer for each test."""
        return AdvancedPatternRecognizer()
    
    @pytest.mark.asyncio
    async def test_comprehensive_pattern_detection(self, recognizer):
        """Test comprehensive pattern detection."""
        # Mock prompt data with various patterns
        mock_prompt_data = {
            'prompt_1': {
                'daily_usage': {
                    '2024-01-01': 20,
                    '2024-01-02': 22,
                    '2024-01-03': 100,  # Spike
                    '2024-01-04': 21,
                    '2024-01-05': 23,
                    '2024-01-06': 20,
                    '2024-01-07': 22
                }
            }
        }
        
        with patch('scrollintel.core.trend_pattern_analysis.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_prompt_performance_summary = AsyncMock(
                side_effect=lambda prompt_id, days: mock_prompt_data.get(prompt_id, {'error': 'Not found'})
            )
            
            patterns = await recognizer.detect_comprehensive_patterns(
                prompt_ids=['prompt_1'],
                metric_names=['usage_count'],
                days=7
            )
            
            assert isinstance(patterns, list)
            # Should detect the spike pattern
            spike_patterns = [p for p in patterns if p.pattern_type == PatternType.SPIKE]
            assert len(spike_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_anomaly_pattern_detection(self, recognizer):
        """Test anomaly pattern detection."""
        prompt_data = {
            'test_prompt': {
                'daily_usage': {
                    '2024-01-01': 20,
                    '2024-01-02': 22,
                    '2024-01-03': 21,
                    '2024-01-04': 19,
                    '2024-01-05': 23,
                    '2024-01-06': 150,  # Clear anomaly
                    '2024-01-07': 20,
                    '2024-01-08': 21
                }
            }
        }
        
        patterns = await recognizer._detect_anomaly_patterns(prompt_data, ['usage_count'])
        
        assert len(patterns) > 0
        anomaly_pattern = patterns[0]
        assert anomaly_pattern.pattern_type == PatternType.ANOMALY
        assert anomaly_pattern.parameters['anomaly_value'] == 150
        assert anomaly_pattern.confidence_score > 0.5
    
    @pytest.mark.asyncio
    async def test_spike_pattern_detection(self, recognizer):
        """Test spike pattern detection."""
        prompt_data = {
            'spike_prompt': {
                'daily_usage': {
                    '2024-01-01': 15,
                    '2024-01-02': 18,
                    '2024-01-03': 16,
                    '2024-01-04': 200,  # Spike
                    '2024-01-05': 17,
                    '2024-01-06': 19,
                    '2024-01-07': 16
                }
            }
        }
        
        patterns = await recognizer._detect_spike_patterns(prompt_data, ['usage_count'])
        
        assert len(patterns) > 0
        spike_pattern = patterns[0]
        assert spike_pattern.pattern_type == PatternType.SPIKE
        assert spike_pattern.parameters['spike_value'] == 200
        assert spike_pattern.parameters['magnitude'] > 10  # Should be much higher than baseline
    
    @pytest.mark.asyncio
    async def test_plateau_pattern_detection(self, recognizer):
        """Test plateau pattern detection."""
        prompt_data = {
            'plateau_prompt': {
                'daily_usage': {
                    '2024-01-01': 10,
                    '2024-01-02': 25,
                    '2024-01-03': 25,
                    '2024-01-04': 24,
                    '2024-01-05': 25,
                    '2024-01-06': 26,
                    '2024-01-07': 25,
                    '2024-01-08': 15
                }
            }
        }
        
        patterns = await recognizer._detect_plateau_patterns(prompt_data, ['usage_count'])
        
        assert len(patterns) > 0
        plateau_pattern = patterns[0]
        assert plateau_pattern.pattern_type == PatternType.PLATEAU
        assert plateau_pattern.parameters['plateau_length'] >= 3
        assert abs(plateau_pattern.parameters['plateau_value'] - 25) < 1
    
    @pytest.mark.asyncio
    async def test_oscillation_pattern_detection(self, recognizer):
        """Test oscillation pattern detection."""
        prompt_data = {
            'oscillating_prompt': {
                'daily_usage': {
                    '2024-01-01': 20,
                    '2024-01-02': 30,
                    '2024-01-03': 20,
                    '2024-01-04': 30,
                    '2024-01-05': 20,
                    '2024-01-06': 30,
                    '2024-01-07': 20,
                    '2024-01-08': 30,
                    '2024-01-09': 20,
                    '2024-01-10': 30
                }
            }
        }
        
        patterns = await recognizer._detect_oscillation_patterns(prompt_data, ['usage_count'])
        
        assert len(patterns) > 0
        oscillation_pattern = patterns[0]
        assert oscillation_pattern.pattern_type == PatternType.OSCILLATION
        assert oscillation_pattern.parameters['oscillation_frequency'] > 0.3
        assert oscillation_pattern.parameters['direction_changes'] > 5
    
    @pytest.mark.asyncio
    async def test_pattern_confidence_scoring(self, recognizer):
        """Test pattern confidence scoring."""
        # Test with very clear spike
        prompt_data = {
            'clear_spike': {
                'daily_usage': {
                    '2024-01-01': 10,
                    '2024-01-02': 10,
                    '2024-01-03': 1000,  # Very clear spike
                    '2024-01-04': 10,
                    '2024-01-05': 10
                }
            }
        }
        
        patterns = await recognizer._detect_spike_patterns(prompt_data, ['usage_count'])
        
        if patterns:
            clear_spike = patterns[0]
            assert clear_spike.confidence_score > 0.8  # Should have high confidence
        
        # Test with marginal spike
        prompt_data_marginal = {
            'marginal_spike': {
                'daily_usage': {
                    '2024-01-01': 10,
                    '2024-01-02': 10,
                    '2024-01-03': 25,  # Marginal spike
                    '2024-01-04': 10,
                    '2024-01-05': 10
                }
            }
        }
        
        patterns_marginal = await recognizer._detect_spike_patterns(prompt_data_marginal, ['usage_count'])
        
        if patterns_marginal:
            marginal_spike = patterns_marginal[0]
            assert marginal_spike.confidence_score < clear_spike.confidence_score
    
    @pytest.mark.asyncio
    async def test_pattern_sorting_by_confidence(self, recognizer):
        """Test that patterns are sorted by confidence score."""
        # Create data with multiple patterns of different strengths
        prompt_data = {
            'multi_pattern': {
                'daily_usage': {
                    '2024-01-01': 20,
                    '2024-01-02': 21,
                    '2024-01-03': 500,  # Strong spike
                    '2024-01-04': 22,
                    '2024-01-05': 50,   # Weaker spike
                    '2024-01-06': 21,
                    '2024-01-07': 20,
                    '2024-01-08': 22,
                    '2024-01-09': 21,
                    '2024-01-10': 20
                }
            }
        }
        
        with patch('scrollintel.core.trend_pattern_analysis.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_prompt_performance_summary = AsyncMock(
                return_value=prompt_data['multi_pattern']
            )
            
            patterns = await recognizer.detect_comprehensive_patterns(
                prompt_ids=['multi_pattern'],
                metric_names=['usage_count'],
                days=10
            )
            
            # Patterns should be sorted by confidence (highest first)
            if len(patterns) > 1:
                for i in range(len(patterns) - 1):
                    assert patterns[i].confidence_score >= patterns[i + 1].confidence_score
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, recognizer):
        """Test handling of insufficient data for pattern recognition."""
        prompt_data = {
            'insufficient_data': {
                'daily_usage': {
                    '2024-01-01': 20,
                    '2024-01-02': 22
                }
            }
        }
        
        patterns = await recognizer._detect_spike_patterns(prompt_data, ['usage_count'])
        
        # Should not detect patterns with insufficient data
        assert len(patterns) == 0

class TestDataStructures:
    """Test data structures used in trend analysis."""
    
    def test_trend_analysis_creation(self):
        """Test TrendAnalysis data structure creation."""
        analysis = TrendAnalysis(
            metric_name="usage_count",
            trend_type=TrendType.LINEAR,
            trend_direction="increasing",
            trend_strength=0.85,
            confidence_level=0.92,
            r_squared=0.88,
            slope=2.5,
            data_points=[],
            forecast=None,
            analysis_period=(datetime(2024, 1, 1), datetime(2024, 1, 7)),
            generated_at=datetime.utcnow()
        )
        
        assert analysis.metric_name == "usage_count"
        assert analysis.trend_type == TrendType.LINEAR
        assert analysis.trend_direction == "increasing"
        assert analysis.trend_strength == 0.85
        assert analysis.confidence_level == 0.92
    
    def test_pattern_creation(self):
        """Test Pattern data structure creation."""
        pattern = Pattern(
            pattern_type=PatternType.SPIKE,
            start_time=datetime(2024, 1, 3),
            end_time=datetime(2024, 1, 3),
            confidence_score=0.95,
            parameters={
                'spike_value': 150,
                'baseline': 25,
                'magnitude': 6.0
            },
            description="Usage spike detected",
            affected_metrics=['usage_count']
        )
        
        assert pattern.pattern_type == PatternType.SPIKE
        assert pattern.confidence_score == 0.95
        assert pattern.parameters['spike_value'] == 150
        assert 'usage_count' in pattern.affected_metrics

class TestIntegration:
    """Integration tests for trend analysis and pattern recognition."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_trend_and_pattern_analysis(self):
        """Test complete workflow from data to insights."""
        analyzer = AdvancedTrendAnalyzer()
        recognizer = AdvancedPatternRecognizer()
        
        # Mock comprehensive data with trends and patterns
        mock_summary = {
            'prompt_id': 'integration_test',
            'daily_usage': {
                '2024-01-01': 10,
                '2024-01-02': 12,
                '2024-01-03': 14,
                '2024-01-04': 100,  # Spike
                '2024-01-05': 18,
                '2024-01-06': 20,
                '2024-01-07': 22,
                '2024-01-08': 24,
                '2024-01-09': 26,
                '2024-01-10': 28
            }
        }
        
        with patch('scrollintel.core.trend_pattern_analysis.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_prompt_performance_summary = AsyncMock(return_value=mock_summary)
            
            # Perform trend analysis
            trend_analysis = await analyzer.analyze_comprehensive_trends(
                prompt_id='integration_test',
                metric_name='usage_count',
                days=10,
                forecast_days=3
            )
            
            # Perform pattern recognition
            patterns = await recognizer.detect_comprehensive_patterns(
                prompt_ids=['integration_test'],
                metric_names=['usage_count'],
                days=10
            )
            
            # Verify trend analysis results
            assert 'error' not in trend_analysis
            assert trend_analysis['data_points'] == 10
            assert 'best_fit' in trend_analysis
            assert len(trend_analysis['forecast']) == 3
            
            # Verify pattern recognition results
            assert len(patterns) > 0
            spike_patterns = [p for p in patterns if p.pattern_type == PatternType.SPIKE]
            assert len(spike_patterns) > 0
            
            # The spike should be detected on day 4
            spike_pattern = spike_patterns[0]
            assert spike_pattern.parameters['spike_value'] == 100
    
    @pytest.mark.asyncio
    async def test_multiple_prompts_analysis(self):
        """Test analysis across multiple prompts."""
        analyzer = AdvancedTrendAnalyzer()
        recognizer = AdvancedPatternRecognizer()
        
        # Mock data for multiple prompts
        def mock_summary_side_effect(prompt_id, days):
            if prompt_id == 'trending_prompt':
                return {
                    'daily_usage': {f'2024-01-{i+1:02d}': 10 + i * 2 for i in range(10)}
                }
            elif prompt_id == 'spiking_prompt':
                usage = {f'2024-01-{i+1:02d}': 20 for i in range(10)}
                usage['2024-01-05'] = 200  # Spike on day 5
                return {'daily_usage': usage}
            else:
                return {'error': 'Not found'}
        
        with patch('scrollintel.core.trend_pattern_analysis.prompt_performance_tracker') as mock_tracker:
            mock_tracker.get_prompt_performance_summary = AsyncMock(side_effect=mock_summary_side_effect)
            
            # Analyze trends for trending prompt
            trend_analysis = await analyzer.analyze_comprehensive_trends(
                prompt_id='trending_prompt',
                metric_name='usage_count',
                days=10
            )
            
            # Detect patterns across both prompts
            patterns = await recognizer.detect_comprehensive_patterns(
                prompt_ids=['trending_prompt', 'spiking_prompt'],
                metric_names=['usage_count'],
                days=10
            )
            
            # Should detect linear trend in trending_prompt
            assert trend_analysis['best_fit']['analysis_type'] == 'linear'
            assert trend_analysis['best_fit']['trend_direction'] == 'increasing'
            
            # Should detect spike in spiking_prompt
            spike_patterns = [p for p in patterns if p.pattern_type == PatternType.SPIKE]
            assert len(spike_patterns) > 0

if __name__ == "__main__":
    pytest.main([__file__])