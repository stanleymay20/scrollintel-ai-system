"""
Advanced trend analysis and pattern recognition for prompt analytics.
Provides sophisticated statistical analysis, forecasting, and pattern detection.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass
from enum import Enum

from ..models.analytics_models import PromptMetrics, UsageAnalytics, TrendAnalysis, PatternRecognition
from ..models.database import get_db_session

logger = logging.getLogger(__name__)

class TrendType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POLYNOMIAL = "polynomial"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"

class PatternType(Enum):
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    ANOMALY = "anomaly"
    SPIKE = "spike"
    DROP = "drop"
    PLATEAU = "plateau"
    OSCILLATION = "oscillation"

@dataclass
class TrendForecast:
    """Forecast data structure."""
    timestamps: List[datetime]
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    forecast_accuracy: float
    model_type: str

@dataclass
class DetectedPattern:
    """Detected pattern data structure."""
    pattern_type: PatternType
    start_time: datetime
    end_time: datetime
    confidence_score: float
    parameters: Dict[str, Any]
    description: str
    affected_metrics: List[str]

class AdvancedTrendAnalyzer:
    """Advanced trend analysis with multiple algorithms."""
    
    def __init__(self):
        self.trend_models = {
            TrendType.LINEAR: self._fit_linear_trend,
            TrendType.EXPONENTIAL: self._fit_exponential_trend,
            TrendType.LOGARITHMIC: self._fit_logarithmic_trend,
            TrendType.POLYNOMIAL: self._fit_polynomial_trend,
            TrendType.SEASONAL: self._fit_seasonal_trend
        }
    
    async def analyze_comprehensive_trends(
        self,
        prompt_id: str,
        metric_name: str,
        days: int = 30,
        forecast_days: int = 7
    ) -> Dict[str, Any]:
        """Perform comprehensive trend analysis with multiple models."""
        try:
            # Get historical data
            data = await self._get_time_series_data(prompt_id, metric_name, days)
            
            if len(data) < 5:
                return {"error": "Insufficient data for trend analysis"}
            
            # Prepare time series
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Test multiple trend models
            trend_results = {}
            best_model = None
            best_score = float('-inf')
            
            for trend_type, model_func in self.trend_models.items():
                try:
                    result = model_func(df, forecast_days)
                    trend_results[trend_type.value] = result
                    
                    if result['fit_score'] > best_score:
                        best_score = result['fit_score']
                        best_model = trend_type.value
                        
                except Exception as e:
                    logger.warning(f"Failed to fit {trend_type.value} model: {str(e)}")
                    continue
            
            # Generate comprehensive analysis
            analysis = {
                "prompt_id": prompt_id,
                "metric_name": metric_name,
                "analysis_period_days": days,
                "data_points": len(data),
                "best_model": best_model,
                "model_results": trend_results,
                "trend_summary": self._summarize_trends(trend_results),
                "forecast": trend_results.get(best_model, {}).get('forecast') if best_model else None,
                "confidence_metrics": self._calculate_confidence_metrics(trend_results),
                "recommendations": self._generate_trend_recommendations(trend_results, best_model)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive trend analysis: {str(e)}")
            return {"error": str(e)}
    
    async def _get_time_series_data(
        self,
        prompt_id: str,
        metric_name: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """Get time series data for analysis."""
        try:
            with get_db_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.prompt_id == prompt_id,
                        PromptMetrics.created_at >= cutoff_date
                    )
                ).order_by(PromptMetrics.created_at).all()
                
                data = []
                for metric in metrics:
                    value = getattr(metric, metric_name)
                    if value is not None:
                        data.append({
                            'timestamp': metric.created_at,
                            'value': float(value)
                        })
                
                return data
                
        except Exception as e:
            logger.error(f"Error getting time series data: {str(e)}")
            return []
    
    def _fit_linear_trend(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Fit linear trend model."""
        try:
            # Convert timestamps to numeric for regression
            df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['time_numeric'], df['value']
            )
            
            # Calculate predictions
            predictions = slope * df['time_numeric'] + intercept
            
            # Calculate fit metrics
            r_squared = r_value ** 2
            mse = np.mean((df['value'] - predictions) ** 2)
            
            # Generate forecast
            forecast = self._generate_linear_forecast(
                df, slope, intercept, forecast_days
            )
            
            return {
                'model_type': 'linear',
                'parameters': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'std_error': std_err
                },
                'fit_score': r_squared,
                'mse': mse,
                'predictions': predictions.tolist(),
                'forecast': forecast,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'trend_strength': abs(slope),
                'statistical_significance': p_value < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error fitting linear trend: {str(e)}")
            raise
    
    def _fit_exponential_trend(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Fit exponential trend model."""
        try:
            # Ensure positive values for log transformation
            min_val = df['value'].min()
            if min_val <= 0:
                df['log_value'] = np.log(df['value'] - min_val + 1)
            else:
                df['log_value'] = np.log(df['value'])
            
            df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            
            # Fit linear regression on log-transformed data
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['time_numeric'], df['log_value']
            )
            
            # Transform back to original scale
            predictions = np.exp(slope * df['time_numeric'] + intercept)
            if min_val <= 0:
                predictions = predictions + min_val - 1
            
            # Calculate fit metrics
            r_squared = r_value ** 2
            mse = np.mean((df['value'] - predictions) ** 2)
            
            # Generate forecast
            forecast = self._generate_exponential_forecast(
                df, slope, intercept, forecast_days, min_val
            )
            
            return {
                'model_type': 'exponential',
                'parameters': {
                    'growth_rate': slope,
                    'initial_value': np.exp(intercept),
                    'r_squared': r_squared,
                    'p_value': p_value
                },
                'fit_score': r_squared,
                'mse': mse,
                'predictions': predictions.tolist(),
                'forecast': forecast,
                'trend_direction': 'exponential_growth' if slope > 0 else 'exponential_decay',
                'trend_strength': abs(slope),
                'statistical_significance': p_value < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error fitting exponential trend: {str(e)}")
            raise
    
    def _fit_polynomial_trend(self, df: pd.DataFrame, forecast_days: int, degree: int = 2) -> Dict[str, Any]:
        """Fit polynomial trend model."""
        try:
            df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            
            # Fit polynomial
            coefficients = np.polyfit(df['time_numeric'], df['value'], degree)
            poly_func = np.poly1d(coefficients)
            
            # Calculate predictions
            predictions = poly_func(df['time_numeric'])
            
            # Calculate R-squared
            ss_res = np.sum((df['value'] - predictions) ** 2)
            ss_tot = np.sum((df['value'] - np.mean(df['value'])) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            mse = np.mean((df['value'] - predictions) ** 2)
            
            # Generate forecast
            forecast = self._generate_polynomial_forecast(
                df, coefficients, forecast_days
            )
            
            return {
                'model_type': f'polynomial_degree_{degree}',
                'parameters': {
                    'coefficients': coefficients.tolist(),
                    'degree': degree,
                    'r_squared': r_squared
                },
                'fit_score': r_squared,
                'mse': mse,
                'predictions': predictions.tolist(),
                'forecast': forecast,
                'trend_direction': self._determine_polynomial_trend(coefficients),
                'trend_strength': abs(coefficients[0]) if len(coefficients) > 0 else 0,
                'statistical_significance': r_squared > 0.5
            }
            
        except Exception as e:
            logger.error(f"Error fitting polynomial trend: {str(e)}")
            raise
    
    def _fit_seasonal_trend(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Fit seasonal trend model using Fourier analysis."""
        try:
            if len(df) < 14:  # Need at least 2 weeks of data
                raise ValueError("Insufficient data for seasonal analysis")
            
            # Extract time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            
            # Detect seasonality using FFT
            values = df['value'].values
            fft = np.fft.fft(values)
            frequencies = np.fft.fftfreq(len(values))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft) ** 2
            dominant_freq_idx = np.argsort(power_spectrum)[-3:]  # Top 3 frequencies
            
            # Fit seasonal components
            seasonal_components = []
            for idx in dominant_freq_idx:
                if frequencies[idx] != 0:  # Skip DC component
                    period = 1 / abs(frequencies[idx])
                    amplitude = np.abs(fft[idx]) / len(values)
                    phase = np.angle(fft[idx])
                    
                    seasonal_components.append({
                        'period': period,
                        'amplitude': amplitude,
                        'phase': phase,
                        'frequency': frequencies[idx]
                    })
            
            # Reconstruct signal
            reconstructed = np.zeros(len(values))
            for component in seasonal_components:
                reconstructed += component['amplitude'] * np.cos(
                    2 * np.pi * component['frequency'] * np.arange(len(values)) + component['phase']
                )
            
            # Add trend component
            trend_slope, trend_intercept = np.polyfit(range(len(values)), values, 1)
            trend_component = trend_slope * np.arange(len(values)) + trend_intercept
            
            predictions = reconstructed + trend_component
            
            # Calculate fit metrics
            r_squared = 1 - np.sum((values - predictions) ** 2) / np.sum((values - np.mean(values)) ** 2)
            mse = np.mean((values - predictions) ** 2)
            
            # Generate forecast
            forecast = self._generate_seasonal_forecast(
                df, seasonal_components, trend_slope, trend_intercept, forecast_days
            )
            
            return {
                'model_type': 'seasonal',
                'parameters': {
                    'seasonal_components': seasonal_components,
                    'trend_slope': trend_slope,
                    'trend_intercept': trend_intercept,
                    'r_squared': r_squared
                },
                'fit_score': r_squared,
                'mse': mse,
                'predictions': predictions.tolist(),
                'forecast': forecast,
                'trend_direction': 'seasonal_with_trend',
                'trend_strength': len(seasonal_components),
                'statistical_significance': r_squared > 0.3
            }
            
        except Exception as e:
            logger.error(f"Error fitting seasonal trend: {str(e)}")
            raise
    
    def _fit_logarithmic_trend(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Fit logarithmic trend model."""
        try:
            df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            
            # Ensure positive time values for log
            df['log_time'] = np.log(df['time_numeric'] + 1)
            
            # Fit linear regression on log-transformed time
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['log_time'], df['value']
            )
            
            # Calculate predictions
            predictions = slope * df['log_time'] + intercept
            
            # Calculate fit metrics
            r_squared = r_value ** 2
            mse = np.mean((df['value'] - predictions) ** 2)
            
            # Generate forecast
            forecast = self._generate_logarithmic_forecast(
                df, slope, intercept, forecast_days
            )
            
            return {
                'model_type': 'logarithmic',
                'parameters': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'p_value': p_value
                },
                'fit_score': r_squared,
                'mse': mse,
                'predictions': predictions.tolist(),
                'forecast': forecast,
                'trend_direction': 'logarithmic_growth' if slope > 0 else 'logarithmic_decay',
                'trend_strength': abs(slope),
                'statistical_significance': p_value < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error fitting logarithmic trend: {str(e)}")
            raise
    
    def _generate_linear_forecast(
        self,
        df: pd.DataFrame,
        slope: float,
        intercept: float,
        forecast_days: int
    ) -> Dict[str, Any]:
        """Generate linear forecast."""
        try:
            last_timestamp = df['timestamp'].max()
            last_time_numeric = df['time_numeric'].max()
            
            forecast_timestamps = []
            forecast_values = []
            confidence_intervals = []
            
            # Calculate standard error for confidence intervals
            residuals = df['value'] - (slope * df['time_numeric'] + intercept)
            std_error = np.std(residuals)
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                future_time_numeric = last_time_numeric + (i * 24 * 3600)  # seconds in a day
                
                predicted_value = slope * future_time_numeric + intercept
                
                # 95% confidence interval
                margin_of_error = 1.96 * std_error
                lower_bound = predicted_value - margin_of_error
                upper_bound = predicted_value + margin_of_error
                
                forecast_timestamps.append(future_timestamp.isoformat())
                forecast_values.append(predicted_value)
                confidence_intervals.append((lower_bound, upper_bound))
            
            return {
                'timestamps': forecast_timestamps,
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'forecast_accuracy': max(0, 1 - std_error / np.mean(df['value'])) if np.mean(df['value']) != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating linear forecast: {str(e)}")
            return {}
    
    def _generate_exponential_forecast(
        self,
        df: pd.DataFrame,
        slope: float,
        intercept: float,
        forecast_days: int,
        min_val: float
    ) -> Dict[str, Any]:
        """Generate exponential forecast."""
        try:
            last_timestamp = df['timestamp'].max()
            last_time_numeric = df['time_numeric'].max()
            
            forecast_timestamps = []
            forecast_values = []
            confidence_intervals = []
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                future_time_numeric = last_time_numeric + (i * 24 * 3600)
                
                log_predicted = slope * future_time_numeric + intercept
                predicted_value = np.exp(log_predicted)
                
                if min_val <= 0:
                    predicted_value = predicted_value + min_val - 1
                
                # Simple confidence interval (could be improved)
                margin_of_error = predicted_value * 0.1  # 10% margin
                lower_bound = max(0, predicted_value - margin_of_error)
                upper_bound = predicted_value + margin_of_error
                
                forecast_timestamps.append(future_timestamp.isoformat())
                forecast_values.append(predicted_value)
                confidence_intervals.append((lower_bound, upper_bound))
            
            return {
                'timestamps': forecast_timestamps,
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'forecast_accuracy': 0.8  # Placeholder - would calculate based on historical accuracy
            }
            
        except Exception as e:
            logger.error(f"Error generating exponential forecast: {str(e)}")
            return {}
    
    def _generate_polynomial_forecast(
        self,
        df: pd.DataFrame,
        coefficients: np.ndarray,
        forecast_days: int
    ) -> Dict[str, Any]:
        """Generate polynomial forecast."""
        try:
            last_timestamp = df['timestamp'].max()
            last_time_numeric = df['time_numeric'].max()
            poly_func = np.poly1d(coefficients)
            
            forecast_timestamps = []
            forecast_values = []
            confidence_intervals = []
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                future_time_numeric = last_time_numeric + (i * 24 * 3600)
                
                predicted_value = poly_func(future_time_numeric)
                
                # Simple confidence interval
                margin_of_error = abs(predicted_value) * 0.15  # 15% margin
                lower_bound = predicted_value - margin_of_error
                upper_bound = predicted_value + margin_of_error
                
                forecast_timestamps.append(future_timestamp.isoformat())
                forecast_values.append(predicted_value)
                confidence_intervals.append((lower_bound, upper_bound))
            
            return {
                'timestamps': forecast_timestamps,
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'forecast_accuracy': 0.7  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error generating polynomial forecast: {str(e)}")
            return {}
    
    def _generate_seasonal_forecast(
        self,
        df: pd.DataFrame,
        seasonal_components: List[Dict[str, Any]],
        trend_slope: float,
        trend_intercept: float,
        forecast_days: int
    ) -> Dict[str, Any]:
        """Generate seasonal forecast."""
        try:
            last_timestamp = df['timestamp'].max()
            data_length = len(df)
            
            forecast_timestamps = []
            forecast_values = []
            confidence_intervals = []
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                future_index = data_length + i - 1
                
                # Calculate trend component
                trend_value = trend_slope * future_index + trend_intercept
                
                # Calculate seasonal component
                seasonal_value = 0
                for component in seasonal_components:
                    seasonal_value += component['amplitude'] * np.cos(
                        2 * np.pi * component['frequency'] * future_index + component['phase']
                    )
                
                predicted_value = trend_value + seasonal_value
                
                # Confidence interval based on seasonal variation
                seasonal_std = np.std([comp['amplitude'] for comp in seasonal_components])
                margin_of_error = 1.96 * seasonal_std
                lower_bound = predicted_value - margin_of_error
                upper_bound = predicted_value + margin_of_error
                
                forecast_timestamps.append(future_timestamp.isoformat())
                forecast_values.append(predicted_value)
                confidence_intervals.append((lower_bound, upper_bound))
            
            return {
                'timestamps': forecast_timestamps,
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'forecast_accuracy': 0.75  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error generating seasonal forecast: {str(e)}")
            return {}
    
    def _generate_logarithmic_forecast(
        self,
        df: pd.DataFrame,
        slope: float,
        intercept: float,
        forecast_days: int
    ) -> Dict[str, Any]:
        """Generate logarithmic forecast."""
        try:
            last_timestamp = df['timestamp'].max()
            last_time_numeric = df['time_numeric'].max()
            
            forecast_timestamps = []
            forecast_values = []
            confidence_intervals = []
            
            # Calculate standard error
            log_time = np.log(df['time_numeric'] + 1)
            predictions = slope * log_time + intercept
            residuals = df['value'] - predictions
            std_error = np.std(residuals)
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                future_time_numeric = last_time_numeric + (i * 24 * 3600)
                future_log_time = np.log(future_time_numeric + 1)
                
                predicted_value = slope * future_log_time + intercept
                
                # Confidence interval
                margin_of_error = 1.96 * std_error
                lower_bound = predicted_value - margin_of_error
                upper_bound = predicted_value + margin_of_error
                
                forecast_timestamps.append(future_timestamp.isoformat())
                forecast_values.append(predicted_value)
                confidence_intervals.append((lower_bound, upper_bound))
            
            return {
                'timestamps': forecast_timestamps,
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'forecast_accuracy': max(0, 1 - std_error / np.mean(df['value'])) if np.mean(df['value']) != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating logarithmic forecast: {str(e)}")
            return {}
    
    def _determine_polynomial_trend(self, coefficients: np.ndarray) -> str:
        """Determine trend direction for polynomial."""
        if len(coefficients) < 2:
            return "unknown"
        
        # For quadratic, check the sign of the leading coefficient
        if len(coefficients) == 3:  # ax^2 + bx + c
            if coefficients[0] > 0:
                return "accelerating_upward"
            elif coefficients[0] < 0:
                return "accelerating_downward"
            else:
                return "linear"
        
        # For higher order, use the derivative at the end
        derivative = np.polyder(coefficients)
        end_slope = np.polyval(derivative, 100)  # Evaluate at a large x
        
        if end_slope > 0:
            return "increasing"
        elif end_slope < 0:
            return "decreasing"
        else:
            return "stable"
    
    def _summarize_trends(self, trend_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize trend analysis results."""
        if not trend_results:
            return {}
        
        # Find best performing model
        best_model = max(trend_results.keys(), key=lambda k: trend_results[k].get('fit_score', 0))
        best_result = trend_results[best_model]
        
        # Calculate consensus trend direction
        directions = [result.get('trend_direction', 'unknown') for result in trend_results.values()]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        consensus_direction = max(direction_counts.keys(), key=lambda k: direction_counts[k])
        
        return {
            'best_model': best_model,
            'best_fit_score': best_result.get('fit_score', 0),
            'consensus_direction': consensus_direction,
            'model_agreement': direction_counts.get(consensus_direction, 0) / len(directions),
            'average_fit_score': np.mean([r.get('fit_score', 0) for r in trend_results.values()]),
            'trend_strength': best_result.get('trend_strength', 0),
            'statistical_significance': best_result.get('statistical_significance', False)
        }
    
    def _calculate_confidence_metrics(self, trend_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for trend analysis."""
        if not trend_results:
            return {}
        
        fit_scores = [r.get('fit_score', 0) for r in trend_results.values()]
        mse_values = [r.get('mse', float('inf')) for r in trend_results.values()]
        
        return {
            'max_fit_score': max(fit_scores),
            'min_fit_score': min(fit_scores),
            'avg_fit_score': np.mean(fit_scores),
            'fit_score_std': np.std(fit_scores),
            'min_mse': min(mse_values),
            'avg_mse': np.mean([mse for mse in mse_values if mse != float('inf')]),
            'model_consistency': 1 - np.std(fit_scores) if np.std(fit_scores) < 1 else 0
        }
    
    def _generate_trend_recommendations(
        self,
        trend_results: Dict[str, Any],
        best_model: Optional[str]
    ) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if not trend_results or not best_model:
            return ["Insufficient data for trend analysis"]
        
        best_result = trend_results[best_model]
        fit_score = best_result.get('fit_score', 0)
        trend_direction = best_result.get('trend_direction', 'unknown')
        
        # Recommendations based on fit quality
        if fit_score < 0.3:
            recommendations.append("Trend analysis shows low predictability - consider investigating external factors")
        elif fit_score > 0.8:
            recommendations.append("Strong trend detected - suitable for forecasting and planning")
        
        # Recommendations based on trend direction
        if 'declining' in trend_direction or 'decay' in trend_direction:
            recommendations.append("Performance is declining - immediate intervention recommended")
        elif 'increasing' in trend_direction or 'growth' in trend_direction:
            recommendations.append("Positive trend detected - consider scaling or optimization")
        elif 'seasonal' in trend_direction:
            recommendations.append("Seasonal patterns detected - plan for cyclical variations")
        
        # Model-specific recommendations
        if best_model == 'exponential':
            if best_result.get('parameters', {}).get('growth_rate', 0) > 0.1:
                recommendations.append("Exponential growth detected - monitor for sustainability")
        elif best_model == 'polynomial':
            recommendations.append("Complex trend pattern - monitor closely for inflection points")
        
        return recommendations

class AdvancedPatternRecognizer:
    """Advanced pattern recognition using machine learning techniques."""
    
    def __init__(self):
        self.pattern_detectors = {
            PatternType.ANOMALY: self._detect_anomalies,
            PatternType.SEASONAL: self._detect_seasonality,
            PatternType.SPIKE: self._detect_spikes,
            PatternType.DROP: self._detect_drops,
            PatternType.PLATEAU: self._detect_plateaus,
            PatternType.OSCILLATION: self._detect_oscillations
        }
    
    async def detect_comprehensive_patterns(
        self,
        prompt_ids: List[str],
        metric_names: List[str],
        days: int = 30
    ) -> List[DetectedPattern]:
        """Detect comprehensive patterns across multiple prompts and metrics."""
        try:
            all_patterns = []
            
            for prompt_id in prompt_ids:
                for metric_name in metric_names:
                    # Get time series data
                    data = await self._get_pattern_data(prompt_id, metric_name, days)
                    
                    if len(data) < 10:  # Need minimum data
                        continue
                    
                    # Run all pattern detectors
                    for pattern_type, detector_func in self.pattern_detectors.items():
                        try:
                            patterns = detector_func(data, prompt_id, metric_name)
                            all_patterns.extend(patterns)
                        except Exception as e:
                            logger.warning(f"Pattern detector {pattern_type.value} failed: {str(e)}")
                            continue
            
            # Filter and rank patterns by confidence
            filtered_patterns = [p for p in all_patterns if p.confidence_score > 0.5]
            filtered_patterns.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return filtered_patterns[:20]  # Return top 20 patterns
            
        except Exception as e:
            logger.error(f"Error in comprehensive pattern detection: {str(e)}")
            return []
    
    async def _get_pattern_data(
        self,
        prompt_id: str,
        metric_name: str,
        days: int
    ) -> pd.DataFrame:
        """Get data for pattern analysis."""
        try:
            with get_db_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.prompt_id == prompt_id,
                        PromptMetrics.created_at >= cutoff_date
                    )
                ).order_by(PromptMetrics.created_at).all()
                
                data = []
                for metric in metrics:
                    value = getattr(metric, metric_name)
                    if value is not None:
                        data.append({
                            'timestamp': metric.created_at,
                            'value': float(value),
                            'prompt_id': prompt_id,
                            'metric_name': metric_name
                        })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting pattern data: {str(e)}")
            return pd.DataFrame()
    
    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        prompt_id: str,
        metric_name: str
    ) -> List[DetectedPattern]:
        """Detect anomalies using statistical methods."""
        patterns = []
        
        try:
            if len(df) < 10:
                return patterns
            
            values = df['value'].values
            
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(values))
            anomaly_threshold = 2.5
            anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
            
            # Group consecutive anomalies
            if len(anomaly_indices) > 0:
                anomaly_groups = []
                current_group = [anomaly_indices[0]]
                
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] - anomaly_indices[i-1] <= 2:  # Within 2 points
                        current_group.append(anomaly_indices[i])
                    else:
                        anomaly_groups.append(current_group)
                        current_group = [anomaly_indices[i]]
                
                anomaly_groups.append(current_group)
                
                # Create pattern for each group
                for group in anomaly_groups:
                    if len(group) >= 1:  # At least 1 anomalous point
                        start_idx = group[0]
                        end_idx = group[-1]
                        
                        confidence = min(1.0, np.mean(z_scores[group]) / anomaly_threshold)
                        
                        pattern = DetectedPattern(
                            pattern_type=PatternType.ANOMALY,
                            start_time=df.iloc[start_idx]['timestamp'],
                            end_time=df.iloc[end_idx]['timestamp'],
                            confidence_score=confidence,
                            parameters={
                                'anomaly_count': len(group),
                                'max_z_score': np.max(z_scores[group]),
                                'anomaly_values': values[group].tolist()
                            },
                            description=f"Anomalous values detected in {metric_name}",
                            affected_metrics=[metric_name]
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return patterns
    
    def _detect_seasonality(
        self,
        df: pd.DataFrame,
        prompt_id: str,
        metric_name: str
    ) -> List[DetectedPattern]:
        """Detect seasonal patterns using FFT."""
        patterns = []
        
        try:
            if len(df) < 24:  # Need at least 24 hours of data
                return patterns
            
            values = df['value'].values
            
            # Perform FFT
            fft = np.fft.fft(values)
            frequencies = np.fft.fftfreq(len(values))
            power_spectrum = np.abs(fft) ** 2
            
            # Find dominant frequencies (excluding DC component)
            non_dc_indices = np.where(frequencies != 0)[0]
            if len(non_dc_indices) == 0:
                return patterns
            
            dominant_freq_idx = non_dc_indices[np.argmax(power_spectrum[non_dc_indices])]
            dominant_frequency = frequencies[dominant_freq_idx]
            dominant_power = power_spectrum[dominant_freq_idx]
            
            # Calculate period in hours (assuming hourly data)
            if abs(dominant_frequency) > 0:
                period_hours = 1 / abs(dominant_frequency)
                
                # Check if it's a meaningful seasonal pattern
                if 12 <= period_hours <= 168:  # Between 12 hours and 1 week
                    # Calculate confidence based on power spectrum
                    total_power = np.sum(power_spectrum[non_dc_indices])
                    confidence = min(1.0, dominant_power / total_power * 2)
                    
                    if confidence > 0.3:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.SEASONAL,
                            start_time=df['timestamp'].min(),
                            end_time=df['timestamp'].max(),
                            confidence_score=confidence,
                            parameters={
                                'period_hours': period_hours,
                                'dominant_frequency': dominant_frequency,
                                'power_ratio': dominant_power / total_power,
                                'amplitude': np.abs(fft[dominant_freq_idx]) / len(values)
                            },
                            description=f"Seasonal pattern with {period_hours:.1f} hour period",
                            affected_metrics=[metric_name]
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {str(e)}")
            return patterns
    
    def _detect_spikes(
        self,
        df: pd.DataFrame,
        prompt_id: str,
        metric_name: str
    ) -> List[DetectedPattern]:
        """Detect spike patterns."""
        patterns = []
        
        try:
            if len(df) < 5:
                return patterns
            
            values = df['value'].values
            
            # Find peaks using scipy
            peaks, properties = find_peaks(
                values,
                height=np.mean(values) + 2 * np.std(values),  # 2 std above mean
                distance=3  # Minimum distance between peaks
            )
            
            for peak_idx in peaks:
                # Calculate spike characteristics
                peak_value = values[peak_idx]
                baseline = np.mean(values)
                spike_magnitude = peak_value - baseline
                
                # Find spike boundaries
                start_idx = max(0, peak_idx - 2)
                end_idx = min(len(values) - 1, peak_idx + 2)
                
                # Calculate confidence based on spike magnitude
                confidence = min(1.0, spike_magnitude / (2 * np.std(values)))
                
                if confidence > 0.5:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.SPIKE,
                        start_time=df.iloc[start_idx]['timestamp'],
                        end_time=df.iloc[end_idx]['timestamp'],
                        confidence_score=confidence,
                        parameters={
                            'peak_value': peak_value,
                            'baseline_value': baseline,
                            'spike_magnitude': spike_magnitude,
                            'peak_index': peak_idx
                        },
                        description=f"Spike detected with magnitude {spike_magnitude:.2f}",
                        affected_metrics=[metric_name]
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting spikes: {str(e)}")
            return patterns
    
    def _detect_drops(
        self,
        df: pd.DataFrame,
        prompt_id: str,
        metric_name: str
    ) -> List[DetectedPattern]:
        """Detect drop patterns."""
        patterns = []
        
        try:
            if len(df) < 5:
                return patterns
            
            values = df['value'].values
            
            # Find valleys (inverted peaks)
            inverted_values = -values
            peaks, properties = find_peaks(
                inverted_values,
                height=-np.mean(values) + 2 * np.std(values),  # 2 std below mean
                distance=3
            )
            
            for valley_idx in peaks:
                # Calculate drop characteristics
                valley_value = values[valley_idx]
                baseline = np.mean(values)
                drop_magnitude = baseline - valley_value
                
                # Find drop boundaries
                start_idx = max(0, valley_idx - 2)
                end_idx = min(len(values) - 1, valley_idx + 2)
                
                # Calculate confidence
                confidence = min(1.0, drop_magnitude / (2 * np.std(values)))
                
                if confidence > 0.5:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.DROP,
                        start_time=df.iloc[start_idx]['timestamp'],
                        end_time=df.iloc[end_idx]['timestamp'],
                        confidence_score=confidence,
                        parameters={
                            'valley_value': valley_value,
                            'baseline_value': baseline,
                            'drop_magnitude': drop_magnitude,
                            'valley_index': valley_idx
                        },
                        description=f"Drop detected with magnitude {drop_magnitude:.2f}",
                        affected_metrics=[metric_name]
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting drops: {str(e)}")
            return patterns
    
    def _detect_plateaus(
        self,
        df: pd.DataFrame,
        prompt_id: str,
        metric_name: str
    ) -> List[DetectedPattern]:
        """Detect plateau patterns."""
        patterns = []
        
        try:
            if len(df) < 10:
                return patterns
            
            values = df['value'].values
            
            # Calculate rolling standard deviation to find stable periods
            window_size = min(5, len(values) // 3)
            rolling_std = pd.Series(values).rolling(window=window_size).std()
            
            # Find periods with low variability
            stability_threshold = np.std(values) * 0.2  # 20% of overall std
            stable_periods = rolling_std < stability_threshold
            
            # Find consecutive stable periods
            stable_indices = np.where(stable_periods)[0]
            
            if len(stable_indices) > 0:
                # Group consecutive indices
                groups = []
                current_group = [stable_indices[0]]
                
                for i in range(1, len(stable_indices)):
                    if stable_indices[i] - stable_indices[i-1] <= 2:
                        current_group.append(stable_indices[i])
                    else:
                        if len(current_group) >= window_size:
                            groups.append(current_group)
                        current_group = [stable_indices[i]]
                
                if len(current_group) >= window_size:
                    groups.append(current_group)
                
                # Create patterns for significant plateaus
                for group in groups:
                    if len(group) >= window_size:
                        start_idx = group[0]
                        end_idx = group[-1]
                        
                        plateau_values = values[start_idx:end_idx+1]
                        plateau_std = np.std(plateau_values)
                        plateau_mean = np.mean(plateau_values)
                        
                        # Calculate confidence based on stability
                        confidence = min(1.0, 1 - (plateau_std / np.std(values)))
                        
                        if confidence > 0.6:
                            pattern = DetectedPattern(
                                pattern_type=PatternType.PLATEAU,
                                start_time=df.iloc[start_idx]['timestamp'],
                                end_time=df.iloc[end_idx]['timestamp'],
                                confidence_score=confidence,
                                parameters={
                                    'plateau_value': plateau_mean,
                                    'plateau_std': plateau_std,
                                    'duration_points': len(group),
                                    'stability_ratio': plateau_std / np.std(values)
                                },
                                description=f"Plateau detected at value {plateau_mean:.2f}",
                                affected_metrics=[metric_name]
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting plateaus: {str(e)}")
            return patterns
    
    def _detect_oscillations(
        self,
        df: pd.DataFrame,
        prompt_id: str,
        metric_name: str
    ) -> List[DetectedPattern]:
        """Detect oscillation patterns."""
        patterns = []
        
        try:
            if len(df) < 8:
                return patterns
            
            values = df['value'].values
            
            # Find peaks and valleys
            peaks, _ = find_peaks(values, distance=2)
            valleys, _ = find_peaks(-values, distance=2)
            
            # Combine and sort extrema
            extrema = np.concatenate([peaks, valleys])
            extrema_types = ['peak'] * len(peaks) + ['valley'] * len(valleys)
            
            if len(extrema) < 4:  # Need at least 2 peaks and 2 valleys
                return patterns
            
            # Sort by index
            sorted_indices = np.argsort(extrema)
            sorted_extrema = extrema[sorted_indices]
            sorted_types = [extrema_types[i] for i in sorted_indices]
            
            # Check for alternating pattern
            alternating_count = 0
            for i in range(1, len(sorted_types)):
                if sorted_types[i] != sorted_types[i-1]:
                    alternating_count += 1
            
            alternating_ratio = alternating_count / (len(sorted_types) - 1)
            
            if alternating_ratio > 0.6:  # At least 60% alternating
                # Calculate oscillation characteristics
                extrema_values = values[sorted_extrema]
                amplitude = (np.max(extrema_values) - np.min(extrema_values)) / 2
                
                # Estimate period
                if len(sorted_extrema) >= 4:
                    periods = []
                    for i in range(2, len(sorted_extrema)):
                        if sorted_types[i] == sorted_types[i-2]:  # Same type of extrema
                            period = sorted_extrema[i] - sorted_extrema[i-2]
                            periods.append(period)
                    
                    avg_period = np.mean(periods) if periods else 0
                    
                    confidence = min(1.0, alternating_ratio * (amplitude / np.std(values)))
                    
                    if confidence > 0.5:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.OSCILLATION,
                            start_time=df.iloc[sorted_extrema[0]]['timestamp'],
                            end_time=df.iloc[sorted_extrema[-1]]['timestamp'],
                            confidence_score=confidence,
                            parameters={
                                'amplitude': amplitude,
                                'average_period': avg_period,
                                'extrema_count': len(sorted_extrema),
                                'alternating_ratio': alternating_ratio
                            },
                            description=f"Oscillation with amplitude {amplitude:.2f}",
                            affected_metrics=[metric_name]
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting oscillations: {str(e)}")
            return patterns