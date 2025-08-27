"""
Advanced trend analysis and pattern recognition for prompt analytics.
Provides statistical analysis, forecasting, and anomaly detection.
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

from ..core.config import get_settings
from ..core.logging_config import get_logger
from .prompt_analytics import prompt_performance_tracker

settings = get_settings()
logger = get_logger(__name__)

class TrendType(Enum):
    """Types of trends that can be detected."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POLYNOMIAL = "polynomial"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    STABLE = "stable"

class PatternType(Enum):
    """Types of patterns that can be recognized."""
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    ANOMALY = "anomaly"
    SPIKE = "spike"
    DROP = "drop"
    PLATEAU = "plateau"
    OSCILLATION = "oscillation"

@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    metric_name: str
    trend_type: TrendType
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    r_squared: float
    slope: Optional[float]
    data_points: List[Dict[str, Any]]
    forecast: Optional[List[Dict[str, Any]]]
    analysis_period: Tuple[datetime, datetime]
    generated_at: datetime

@dataclass
class Pattern:
    """Detected pattern in data."""
    pattern_type: PatternType
    start_time: datetime
    end_time: datetime
    confidence_score: float
    parameters: Dict[str, Any]
    description: str
    affected_metrics: List[str]

class AdvancedTrendAnalyzer:
    """Advanced statistical trend analysis system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def analyze_comprehensive_trends(
        self,
        prompt_id: str,
        metric_name: str,
        days: int = 30,
        forecast_days: int = 7
    ) -> Dict[str, Any]:
        """Perform comprehensive trend analysis with forecasting."""
        try:
            # Get performance data
            performance_summary = await prompt_performance_tracker.get_prompt_performance_summary(
                prompt_id=prompt_id,
                days=days
            )
            
            if 'error' in performance_summary:
                return performance_summary
            
            # Extract time series data
            daily_usage = performance_summary.get('daily_usage', {})
            if not daily_usage:
                return {
                    'error': 'Insufficient data for trend analysis',
                    'prompt_id': prompt_id,
                    'metric_name': metric_name
                }
            
            # Convert to time series
            time_series = self._prepare_time_series(daily_usage, metric_name)
            
            if len(time_series) < 3:
                return {
                    'error': 'Insufficient data points for trend analysis',
                    'prompt_id': prompt_id,
                    'metric_name': metric_name,
                    'data_points': len(time_series)
                }
            
            # Perform multiple trend analyses
            analyses = {}
            
            # Linear trend analysis
            linear_analysis = await self._analyze_linear_trend(time_series)
            analyses['linear'] = linear_analysis
            
            # Polynomial trend analysis
            polynomial_analysis = await self._analyze_polynomial_trend(time_series)
            analyses['polynomial'] = polynomial_analysis
            
            # Seasonal analysis
            if len(time_series) >= 7:  # Need at least a week of data
                seasonal_analysis = await self._analyze_seasonal_pattern(time_series)
                analyses['seasonal'] = seasonal_analysis
            
            # Determine best fit
            best_analysis = self._select_best_trend_analysis(analyses)
            
            # Generate forecast
            forecast = None
            if best_analysis and forecast_days > 0:
                forecast = await self._generate_forecast(
                    time_series,
                    best_analysis,
                    forecast_days
                )
            
            # Calculate trend statistics
            trend_stats = self._calculate_trend_statistics(time_series)
            
            result = {
                'prompt_id': prompt_id,
                'metric_name': metric_name,
                'analysis_period': {
                    'start': min(point['date'] for point in time_series),
                    'end': max(point['date'] for point in time_series),
                    'days': days
                },
                'data_points': len(time_series),
                'trend_analyses': analyses,
                'best_fit': best_analysis,
                'forecast': forecast,
                'statistics': trend_stats,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive trend analysis: {e}")
            return {'error': str(e)}
    
    def _prepare_time_series(self, daily_data: Dict[str, int], metric_name: str) -> List[Dict[str, Any]]:
        """Prepare time series data for analysis."""
        try:
            time_series = []
            
            # Sort by date
            sorted_dates = sorted(daily_data.keys())
            
            for i, date_str in enumerate(sorted_dates):
                value = daily_data[date_str]
                time_series.append({
                    'date': date_str,
                    'day_index': i,
                    'value': value,
                    'timestamp': datetime.fromisoformat(date_str).timestamp()
                })
            
            return time_series
            
        except Exception as e:
            self.logger.error(f"Error preparing time series: {e}")
            return []
    
    async def _analyze_linear_trend(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze linear trend in time series data."""
        try:
            if len(time_series) < 2:
                return {'error': 'Insufficient data for linear analysis'}
            
            # Prepare data for regression
            X = np.array([point['day_index'] for point in time_series]).reshape(-1, 1)
            y = np.array([point['value'] for point in time_series])
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate predictions and R²
            y_pred = model.predict(X)
            r_squared = r2_score(y, y_pred)
            
            # Determine trend direction and strength
            slope = model.coef_[0]
            
            if abs(slope) < 0.1:
                trend_direction = 'stable'
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = 'increasing'
                trend_strength = min(abs(slope) / max(y), 1.0)
            else:
                trend_direction = 'decreasing'
                trend_strength = min(abs(slope) / max(y), 1.0)
            
            # Calculate confidence level based on R² and data points
            confidence_level = r_squared * min(len(time_series) / 10, 1.0)
            
            analysis = {
                'trend_type': TrendType.LINEAR.value,
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 3),
                'confidence_level': round(confidence_level, 3),
                'r_squared': round(r_squared, 3),
                'slope': round(slope, 3),
                'intercept': round(model.intercept_, 3),
                'predictions': [round(pred, 2) for pred in y_pred.tolist()]
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in linear trend analysis: {e}")
            return {'error': str(e)}
    
    async def _analyze_polynomial_trend(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze polynomial trend in time series data."""
        try:
            if len(time_series) < 4:
                return {'error': 'Insufficient data for polynomial analysis'}
            
            # Prepare data
            X = np.array([point['day_index'] for point in time_series]).reshape(-1, 1)
            y = np.array([point['value'] for point in time_series])
            
            # Try different polynomial degrees
            best_degree = 1
            best_r_squared = 0
            best_model = None
            best_poly_features = None
            
            for degree in range(2, min(4, len(time_series) - 1)):
                try:
                    poly_features = PolynomialFeatures(degree=degree)
                    X_poly = poly_features.fit_transform(X)
                    
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    
                    y_pred = model.predict(X_poly)
                    r_squared = r2_score(y, y_pred)
                    
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_degree = degree
                        best_model = model
                        best_poly_features = poly_features
                        
                except Exception:
                    continue
            
            if best_model is None:
                return {'error': 'Could not fit polynomial model'}
            
            # Generate predictions with best model
            X_poly = best_poly_features.transform(X)
            y_pred = best_model.predict(X_poly)
            
            # Determine trend characteristics
            if len(y_pred) >= 2:
                first_half_avg = np.mean(y_pred[:len(y_pred)//2])
                second_half_avg = np.mean(y_pred[len(y_pred)//2:])
                
                if second_half_avg > first_half_avg * 1.05:
                    trend_direction = 'increasing'
                elif second_half_avg < first_half_avg * 0.95:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'stable'
            
            # Calculate trend strength based on variation
            trend_strength = min(np.std(y_pred) / np.mean(y_pred), 1.0) if np.mean(y_pred) > 0 else 0
            confidence_level = best_r_squared * min(len(time_series) / 10, 1.0)
            
            analysis = {
                'trend_type': TrendType.POLYNOMIAL.value,
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 3),
                'confidence_level': round(confidence_level, 3),
                'r_squared': round(best_r_squared, 3),
                'degree': best_degree,
                'predictions': [round(pred, 2) for pred in y_pred.tolist()]
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in polynomial trend analysis: {e}")
            return {'error': str(e)}
    
    async def _analyze_seasonal_pattern(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonal patterns in time series data."""
        try:
            if len(time_series) < 7:
                return {'error': 'Insufficient data for seasonal analysis'}
            
            values = [point['value'] for point in time_series]
            
            # Analyze weekly seasonality (if enough data)
            seasonal_analysis = {}
            
            if len(time_series) >= 14:  # At least 2 weeks
                # Group by day of week
                weekday_values = [[] for _ in range(7)]
                
                for i, point in enumerate(time_series):
                    weekday = i % 7
                    weekday_values[weekday].append(point['value'])
                
                # Calculate average for each day of week
                weekday_averages = []
                for day_values in weekday_values:
                    if day_values:
                        weekday_averages.append(statistics.mean(day_values))
                    else:
                        weekday_averages.append(0)
                
                # Calculate seasonality strength
                overall_mean = statistics.mean(values)
                seasonal_variation = statistics.stdev(weekday_averages) if len(weekday_averages) > 1 else 0
                seasonality_strength = seasonal_variation / overall_mean if overall_mean > 0 else 0
                
                seasonal_analysis = {
                    'trend_type': TrendType.SEASONAL.value,
                    'seasonality_strength': round(seasonality_strength, 3),
                    'weekday_pattern': [round(avg, 2) for avg in weekday_averages],
                    'peak_day': weekday_averages.index(max(weekday_averages)),
                    'low_day': weekday_averages.index(min(weekday_averages)),
                    'confidence_level': min(seasonality_strength * 2, 1.0)
                }
            
            return seasonal_analysis
            
        except Exception as e:
            self.logger.error(f"Error in seasonal analysis: {e}")
            return {'error': str(e)}
    
    def _select_best_trend_analysis(self, analyses: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best trend analysis based on R² and confidence."""
        try:
            best_analysis = None
            best_score = 0
            
            for analysis_type, analysis in analyses.items():
                if 'error' in analysis:
                    continue
                
                # Calculate composite score
                r_squared = analysis.get('r_squared', 0)
                confidence = analysis.get('confidence_level', 0)
                
                # Weight R² more heavily for statistical models
                if analysis_type in ['linear', 'polynomial']:
                    score = r_squared * 0.7 + confidence * 0.3
                else:
                    score = confidence
                
                if score > best_score:
                    best_score = score
                    best_analysis = analysis.copy()
                    best_analysis['analysis_type'] = analysis_type
                    best_analysis['composite_score'] = round(score, 3)
            
            return best_analysis
            
        except Exception as e:
            self.logger.error(f"Error selecting best trend analysis: {e}")
            return None
    
    async def _generate_forecast(
        self,
        time_series: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
        forecast_days: int
    ) -> List[Dict[str, Any]]:
        """Generate forecast based on trend analysis."""
        try:
            forecast = []
            
            if trend_analysis.get('analysis_type') == 'linear':
                # Linear forecast
                slope = trend_analysis.get('slope', 0)
                intercept = trend_analysis.get('intercept', 0)
                
                last_day_index = len(time_series) - 1
                last_date = datetime.fromisoformat(time_series[-1]['date']).date()
                
                for i in range(1, forecast_days + 1):
                    forecast_date = last_date + timedelta(days=i)
                    forecast_day_index = last_day_index + i
                    forecast_value = slope * forecast_day_index + intercept
                    
                    forecast.append({
                        'date': forecast_date.isoformat(),
                        'day_index': forecast_day_index,
                        'predicted_value': max(0, round(forecast_value, 2)),
                        'confidence': max(0, trend_analysis.get('confidence_level', 0) - (i * 0.1))
                    })
            
            elif trend_analysis.get('analysis_type') == 'seasonal':
                # Seasonal forecast
                weekday_pattern = trend_analysis.get('weekday_pattern', [])
                if weekday_pattern:
                    last_date = datetime.fromisoformat(time_series[-1]['date']).date()
                    
                    for i in range(1, forecast_days + 1):
                        forecast_date = last_date + timedelta(days=i)
                        weekday = (forecast_date.weekday() + 1) % 7  # Adjust for Monday=0
                        
                        if weekday < len(weekday_pattern):
                            predicted_value = weekday_pattern[weekday]
                        else:
                            predicted_value = statistics.mean(weekday_pattern)
                        
                        forecast.append({
                            'date': forecast_date.isoformat(),
                            'day_index': len(time_series) + i - 1,
                            'predicted_value': max(0, round(predicted_value, 2)),
                            'confidence': max(0, trend_analysis.get('confidence_level', 0) - (i * 0.05))
                        })
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return []
    
    def _calculate_trend_statistics(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic trend statistics."""
        try:
            values = [point['value'] for point in time_series]
            
            if not values:
                return {}
            
            stats_dict = {
                'mean': round(statistics.mean(values), 2),
                'median': round(statistics.median(values), 2),
                'std_dev': round(statistics.stdev(values), 2) if len(values) > 1 else 0,
                'min_value': min(values),
                'max_value': max(values),
                'range': max(values) - min(values),
                'coefficient_of_variation': 0
            }
            
            if stats_dict['mean'] > 0:
                stats_dict['coefficient_of_variation'] = round(
                    stats_dict['std_dev'] / stats_dict['mean'], 3
                )
            
            # Calculate growth rate
            if len(values) >= 2:
                first_value = values[0]
                last_value = values[-1]
                
                if first_value > 0:
                    growth_rate = ((last_value - first_value) / first_value) * 100
                    stats_dict['growth_rate_percent'] = round(growth_rate, 2)
            
            return stats_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating trend statistics: {e}")
            return {}

class AdvancedPatternRecognizer:
    """Advanced pattern recognition system for prompt analytics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def detect_comprehensive_patterns(
        self,
        prompt_ids: List[str],
        metric_names: List[str],
        days: int = 30
    ) -> List[Pattern]:
        """Detect comprehensive patterns across multiple prompts and metrics."""
        try:
            patterns = []
            
            # Collect data for all prompts
            prompt_data = {}
            for prompt_id in prompt_ids:
                performance_summary = await prompt_performance_tracker.get_prompt_performance_summary(
                    prompt_id=prompt_id,
                    days=days
                )
                
                if 'error' not in performance_summary:
                    prompt_data[prompt_id] = performance_summary
            
            if not prompt_data:
                return patterns
            
            # Detect different types of patterns
            patterns.extend(await self._detect_anomaly_patterns(prompt_data, metric_names))
            patterns.extend(await self._detect_spike_patterns(prompt_data, metric_names))
            patterns.extend(await self._detect_plateau_patterns(prompt_data, metric_names))
            patterns.extend(await self._detect_oscillation_patterns(prompt_data, metric_names))
            
            # Sort patterns by confidence score
            patterns.sort(key=lambda p: p.confidence_score, reverse=True)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting comprehensive patterns: {e}")
            return []
    
    async def _detect_anomaly_patterns(
        self,
        prompt_data: Dict[str, Dict[str, Any]],
        metric_names: List[str]
    ) -> List[Pattern]:
        """Detect anomaly patterns in the data."""
        try:
            patterns = []
            
            for prompt_id, data in prompt_data.items():
                daily_usage = data.get('daily_usage', {})
                
                if len(daily_usage) < 7:  # Need at least a week of data
                    continue
                
                values = list(daily_usage.values())
                dates = list(daily_usage.keys())
                
                # Calculate statistical thresholds
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                # Detect outliers (values beyond 2 standard deviations)
                threshold = 2 * std_val
                
                for i, (date, value) in enumerate(zip(dates, values)):
                    if abs(value - mean_val) > threshold and std_val > 0:
                        # Found an anomaly
                        confidence = min((abs(value - mean_val) / threshold) / 2, 1.0)
                        
                        pattern = Pattern(
                            pattern_type=PatternType.ANOMALY,
                            start_time=datetime.fromisoformat(date),
                            end_time=datetime.fromisoformat(date),
                            confidence_score=confidence,
                            parameters={
                                'prompt_id': prompt_id,
                                'anomaly_value': value,
                                'expected_range': [mean_val - threshold, mean_val + threshold],
                                'deviation_magnitude': abs(value - mean_val) / std_val
                            },
                            description=f"Anomalous usage detected: {value} (expected ~{mean_val:.1f})",
                            affected_metrics=['usage_count']
                        )
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting anomaly patterns: {e}")
            return []
    
    async def _detect_spike_patterns(
        self,
        prompt_data: Dict[str, Dict[str, Any]],
        metric_names: List[str]
    ) -> List[Pattern]:
        """Detect spike patterns in the data."""
        try:
            patterns = []
            
            for prompt_id, data in prompt_data.items():
                daily_usage = data.get('daily_usage', {})
                
                if len(daily_usage) < 5:
                    continue
                
                values = list(daily_usage.values())
                dates = list(daily_usage.keys())
                
                # Look for sudden increases
                for i in range(2, len(values) - 2):
                    current_value = values[i]
                    
                    # Compare with surrounding values
                    before_avg = statistics.mean(values[max(0, i-2):i])
                    after_avg = statistics.mean(values[i+1:min(len(values), i+3)])
                    
                    # Check if current value is significantly higher
                    if (current_value > before_avg * 2 and 
                        current_value > after_avg * 2 and
                        current_value > 10):  # Minimum threshold
                        
                        spike_magnitude = current_value / max(before_avg, after_avg, 1)
                        confidence = min(spike_magnitude / 5, 1.0)  # Normalize confidence
                        
                        pattern = Pattern(
                            pattern_type=PatternType.SPIKE,
                            start_time=datetime.fromisoformat(dates[i]),
                            end_time=datetime.fromisoformat(dates[i]),
                            confidence_score=confidence,
                            parameters={
                                'prompt_id': prompt_id,
                                'spike_value': current_value,
                                'baseline_before': before_avg,
                                'baseline_after': after_avg,
                                'magnitude': spike_magnitude
                            },
                            description=f"Usage spike detected: {current_value} (baseline ~{before_avg:.1f})",
                            affected_metrics=['usage_count']
                        )
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting spike patterns: {e}")
            return []
    
    async def _detect_plateau_patterns(
        self,
        prompt_data: Dict[str, Dict[str, Any]],
        metric_names: List[str]
    ) -> List[Pattern]:
        """Detect plateau patterns in the data."""
        try:
            patterns = []
            
            for prompt_id, data in prompt_data.items():
                daily_usage = data.get('daily_usage', {})
                
                if len(daily_usage) < 7:
                    continue
                
                values = list(daily_usage.values())
                dates = list(daily_usage.keys())
                
                # Look for periods of stable values
                plateau_threshold = 0.1  # 10% variation allowed
                min_plateau_length = 3
                
                i = 0
                while i < len(values) - min_plateau_length:
                    plateau_start = i
                    plateau_values = [values[i]]
                    
                    # Extend plateau as long as values are similar
                    j = i + 1
                    while j < len(values):
                        current_avg = statistics.mean(plateau_values)
                        
                        if abs(values[j] - current_avg) / max(current_avg, 1) <= plateau_threshold:
                            plateau_values.append(values[j])
                            j += 1
                        else:
                            break
                    
                    # Check if plateau is significant
                    if len(plateau_values) >= min_plateau_length:
                        plateau_avg = statistics.mean(plateau_values)
                        plateau_std = statistics.stdev(plateau_values) if len(plateau_values) > 1 else 0
                        
                        # Calculate confidence based on stability and length
                        stability = 1 - (plateau_std / max(plateau_avg, 1))
                        length_factor = min(len(plateau_values) / 7, 1.0)
                        confidence = stability * length_factor
                        
                        if confidence > 0.5:  # Only report significant plateaus
                            pattern = Pattern(
                                pattern_type=PatternType.PLATEAU,
                                start_time=datetime.fromisoformat(dates[plateau_start]),
                                end_time=datetime.fromisoformat(dates[plateau_start + len(plateau_values) - 1]),
                                confidence_score=confidence,
                                parameters={
                                    'prompt_id': prompt_id,
                                    'plateau_value': plateau_avg,
                                    'plateau_length': len(plateau_values),
                                    'stability_score': stability
                                },
                                description=f"Stable usage plateau: {plateau_avg:.1f} for {len(plateau_values)} days",
                                affected_metrics=['usage_count']
                            )
                            
                            patterns.append(pattern)
                    
                    i = j if j > i + 1 else i + 1
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting plateau patterns: {e}")
            return []
    
    async def _detect_oscillation_patterns(
        self,
        prompt_data: Dict[str, Dict[str, Any]],
        metric_names: List[str]
    ) -> List[Pattern]:
        """Detect oscillation patterns in the data."""
        try:
            patterns = []
            
            for prompt_id, data in prompt_data.items():
                daily_usage = data.get('daily_usage', {})
                
                if len(daily_usage) < 10:  # Need enough data to detect oscillations
                    continue
                
                values = list(daily_usage.values())
                dates = list(daily_usage.keys())
                
                # Look for regular oscillations
                # Simple approach: count direction changes
                direction_changes = 0
                last_direction = None
                
                for i in range(1, len(values)):
                    if values[i] > values[i-1]:
                        current_direction = 'up'
                    elif values[i] < values[i-1]:
                        current_direction = 'down'
                    else:
                        continue  # No change
                    
                    if last_direction and current_direction != last_direction:
                        direction_changes += 1
                    
                    last_direction = current_direction
                
                # Calculate oscillation frequency
                if len(values) > 2:
                    oscillation_frequency = direction_changes / (len(values) - 1)
                    
                    # High frequency of direction changes indicates oscillation
                    if oscillation_frequency > 0.3:  # At least 30% of points are turning points
                        # Calculate amplitude of oscillations
                        mean_val = statistics.mean(values)
                        deviations = [abs(v - mean_val) for v in values]
                        avg_amplitude = statistics.mean(deviations)
                        
                        confidence = min(oscillation_frequency * 2, 1.0)
                        
                        pattern = Pattern(
                            pattern_type=PatternType.OSCILLATION,
                            start_time=datetime.fromisoformat(dates[0]),
                            end_time=datetime.fromisoformat(dates[-1]),
                            confidence_score=confidence,
                            parameters={
                                'prompt_id': prompt_id,
                                'oscillation_frequency': oscillation_frequency,
                                'avg_amplitude': avg_amplitude,
                                'direction_changes': direction_changes,
                                'period_estimate': len(values) / max(direction_changes / 2, 1)
                            },
                            description=f"Oscillating usage pattern detected (frequency: {oscillation_frequency:.2f})",
                            affected_metrics=['usage_count']
                        )
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting oscillation patterns: {e}")
            return []

# Global instances
advanced_trend_analyzer = AdvancedTrendAnalyzer()
advanced_pattern_recognizer = AdvancedPatternRecognizer()