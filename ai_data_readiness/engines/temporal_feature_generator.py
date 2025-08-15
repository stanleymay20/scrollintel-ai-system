"""
Advanced Temporal Feature Engineering for AI Data Readiness Platform.

This module provides comprehensive time-series feature engineering capabilities
including lag features, rolling statistics, seasonal decomposition, and trend analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
import warnings

from ..models.feature_models import TemporalFeatures, FeatureType
from ..core.exceptions import FeatureEngineeringError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class TemporalConfig:
    """Configuration for temporal feature generation."""
    enable_lag_features: bool = True
    enable_rolling_features: bool = True
    enable_seasonal_features: bool = True
    enable_trend_features: bool = True
    enable_fourier_features: bool = True
    max_lag_periods: int = 30
    rolling_windows: List[int] = None
    seasonal_periods: List[int] = None
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 30]
        if self.seasonal_periods is None:
            self.seasonal_periods = [7, 30, 365]  # Weekly, monthly, yearly


class AdvancedTemporalFeatureGenerator:
    """
    Advanced temporal feature generator with comprehensive time-series capabilities.
    
    Provides intelligent temporal feature engineering including lag features,
    rolling statistics, seasonal decomposition, and trend analysis.
    """
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        """Initialize the temporal feature generator."""
        self.config = config or TemporalConfig()
        self.fitted_parameters = {}
        
    def generate_comprehensive_temporal_features(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate comprehensive temporal features from time series data.
        
        Args:
            data: Input DataFrame with time series data
            time_column: Name of the time column
            value_columns: List of value columns to create features for
            target_column: Target variable column name
            
        Returns:
            Tuple of (enhanced_data, feature_metadata)
        """
        try:
            logger.info(f"Generating comprehensive temporal features for column: {time_column}")
            
            enhanced_data = data.copy()
            metadata = {
                "time_column": time_column,
                "features_created": [],
                "feature_categories": {},
                "temporal_patterns": {}
            }
            
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(enhanced_data[time_column]):
                enhanced_data[time_column] = pd.to_datetime(enhanced_data[time_column])
            
            # Sort by time column
            enhanced_data = enhanced_data.sort_values(time_column).reset_index(drop=True)
            
            # Determine value columns if not provided
            if value_columns is None:
                value_columns = enhanced_data.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in value_columns:
                    value_columns.remove(target_column)
                if time_column in value_columns:
                    value_columns.remove(time_column)
            
            # Generate basic temporal components
            enhanced_data, basic_metadata = self._create_basic_temporal_components(
                enhanced_data, time_column
            )
            metadata["features_created"].extend(basic_metadata["features_created"])
            metadata["feature_categories"]["basic_temporal"] = basic_metadata["features_created"]
            
            # Generate cyclical features
            enhanced_data, cyclical_metadata = self._create_cyclical_features(
                enhanced_data, time_column
            )
            metadata["features_created"].extend(cyclical_metadata["features_created"])
            metadata["feature_categories"]["cyclical"] = cyclical_metadata["features_created"]
            
            # Generate lag features for value columns
            if self.config.enable_lag_features and value_columns:
                enhanced_data, lag_metadata = self._create_lag_features(
                    enhanced_data, time_column, value_columns
                )
                metadata["features_created"].extend(lag_metadata["features_created"])
                metadata["feature_categories"]["lag_features"] = lag_metadata["features_created"]
            
            # Generate rolling statistics
            if self.config.enable_rolling_features and value_columns:
                enhanced_data, rolling_metadata = self._create_rolling_features(
                    enhanced_data, time_column, value_columns
                )
                metadata["features_created"].extend(rolling_metadata["features_created"])
                metadata["feature_categories"]["rolling_features"] = rolling_metadata["features_created"]
            
            # Generate seasonal features
            if self.config.enable_seasonal_features and value_columns:
                enhanced_data, seasonal_metadata = self._create_seasonal_features(
                    enhanced_data, time_column, value_columns
                )
                metadata["features_created"].extend(seasonal_metadata["features_created"])
                metadata["feature_categories"]["seasonal_features"] = seasonal_metadata["features_created"]
                metadata["temporal_patterns"]["seasonality"] = seasonal_metadata.get("patterns", {})
            
            # Generate trend features
            if self.config.enable_trend_features and value_columns:
                enhanced_data, trend_metadata = self._create_trend_features(
                    enhanced_data, time_column, value_columns
                )
                metadata["features_created"].extend(trend_metadata["features_created"])
                metadata["feature_categories"]["trend_features"] = trend_metadata["features_created"]
                metadata["temporal_patterns"]["trends"] = trend_metadata.get("patterns", {})
            
            # Generate Fourier features for periodicity
            if self.config.enable_fourier_features and value_columns:
                enhanced_data, fourier_metadata = self._create_fourier_features(
                    enhanced_data, time_column, value_columns
                )
                metadata["features_created"].extend(fourier_metadata["features_created"])
                metadata["feature_categories"]["fourier_features"] = fourier_metadata["features_created"]
            
            # Generate time-based aggregations
            enhanced_data, agg_metadata = self._create_time_based_aggregations(
                enhanced_data, time_column, value_columns
            )
            metadata["features_created"].extend(agg_metadata["features_created"])
            metadata["feature_categories"]["time_aggregations"] = agg_metadata["features_created"]
            
            logger.info(f"Created {len(metadata['features_created'])} temporal features")
            return enhanced_data, metadata
            
        except Exception as e:
            logger.error(f"Error generating temporal features: {str(e)}")
            raise FeatureEngineeringError(f"Temporal feature generation failed: {str(e)}")
    
    def _create_basic_temporal_components(
        self,
        data: pd.DataFrame,
        time_column: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create basic temporal components."""
        metadata = {"features_created": []}
        
        # Basic time components
        data[f"{time_column}_year"] = data[time_column].dt.year
        data[f"{time_column}_month"] = data[time_column].dt.month
        data[f"{time_column}_day"] = data[time_column].dt.day
        data[f"{time_column}_hour"] = data[time_column].dt.hour
        data[f"{time_column}_minute"] = data[time_column].dt.minute
        data[f"{time_column}_dayofweek"] = data[time_column].dt.dayofweek
        data[f"{time_column}_dayofyear"] = data[time_column].dt.dayofyear
        data[f"{time_column}_weekofyear"] = data[time_column].dt.isocalendar().week
        data[f"{time_column}_quarter"] = data[time_column].dt.quarter
        
        # Boolean time features
        data[f"{time_column}_is_weekend"] = (data[time_column].dt.dayofweek >= 5).astype(int)
        data[f"{time_column}_is_month_start"] = data[time_column].dt.is_month_start.astype(int)
        data[f"{time_column}_is_month_end"] = data[time_column].dt.is_month_end.astype(int)
        data[f"{time_column}_is_quarter_start"] = data[time_column].dt.is_quarter_start.astype(int)
        data[f"{time_column}_is_quarter_end"] = data[time_column].dt.is_quarter_end.astype(int)
        data[f"{time_column}_is_year_start"] = data[time_column].dt.is_year_start.astype(int)
        data[f"{time_column}_is_year_end"] = data[time_column].dt.is_year_end.astype(int)
        
        # Time since epoch
        data[f"{time_column}_timestamp"] = data[time_column].astype(np.int64) // 10**9
        
        # Days since minimum date
        min_date = data[time_column].min()
        data[f"{time_column}_days_since_start"] = (data[time_column] - min_date).dt.days
        
        # Add all created features to metadata
        basic_features = [col for col in data.columns if col.startswith(f"{time_column}_")]
        metadata["features_created"] = basic_features
        
        return data, metadata
    
    def _create_cyclical_features(
        self,
        data: pd.DataFrame,
        time_column: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create cyclical features using sine and cosine transformations."""
        metadata = {"features_created": []}
        
        # Monthly cyclical features
        data[f"{time_column}_month_sin"] = np.sin(2 * np.pi * data[time_column].dt.month / 12)
        data[f"{time_column}_month_cos"] = np.cos(2 * np.pi * data[time_column].dt.month / 12)
        
        # Daily cyclical features
        data[f"{time_column}_day_sin"] = np.sin(2 * np.pi * data[time_column].dt.day / 31)
        data[f"{time_column}_day_cos"] = np.cos(2 * np.pi * data[time_column].dt.day / 31)
        
        # Hourly cyclical features
        data[f"{time_column}_hour_sin"] = np.sin(2 * np.pi * data[time_column].dt.hour / 24)
        data[f"{time_column}_hour_cos"] = np.cos(2 * np.pi * data[time_column].dt.hour / 24)
        
        # Day of week cyclical features
        data[f"{time_column}_dayofweek_sin"] = np.sin(2 * np.pi * data[time_column].dt.dayofweek / 7)
        data[f"{time_column}_dayofweek_cos"] = np.cos(2 * np.pi * data[time_column].dt.dayofweek / 7)
        
        # Day of year cyclical features
        data[f"{time_column}_dayofyear_sin"] = np.sin(2 * np.pi * data[time_column].dt.dayofyear / 365)
        data[f"{time_column}_dayofyear_cos"] = np.cos(2 * np.pi * data[time_column].dt.dayofyear / 365)
        
        cyclical_features = [col for col in data.columns 
                           if any(suffix in col for suffix in ["_sin", "_cos"]) 
                           and col.startswith(f"{time_column}_")]
        metadata["features_created"] = cyclical_features
        
        return data, metadata
    
    def _create_lag_features(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create lag features for value columns."""
        metadata = {"features_created": []}
        
        # Define lag periods
        lag_periods = [1, 2, 3, 7, 14, 30]  # 1 day, 2 days, 3 days, 1 week, 2 weeks, 1 month
        lag_periods = [lag for lag in lag_periods if lag <= self.config.max_lag_periods]
        
        for column in value_columns:
            for lag in lag_periods:
                lag_col = f"{column}_lag_{lag}"
                data[lag_col] = data[column].shift(lag)
                metadata["features_created"].append(lag_col)
                
                # Lag differences
                if lag <= 7:  # Only for short lags to avoid too many features
                    lag_diff_col = f"{column}_lag_{lag}_diff"
                    data[lag_diff_col] = data[column] - data[column].shift(lag)
                    metadata["features_created"].append(lag_diff_col)
        
        return data, metadata
    
    def _create_rolling_features(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create rolling window statistics."""
        metadata = {"features_created": []}
        
        for column in value_columns:
            for window in self.config.rolling_windows:
                # Rolling mean
                rolling_mean_col = f"{column}_rolling_mean_{window}"
                data[rolling_mean_col] = data[column].rolling(window=window, min_periods=1).mean()
                metadata["features_created"].append(rolling_mean_col)
                
                # Rolling standard deviation
                rolling_std_col = f"{column}_rolling_std_{window}"
                data[rolling_std_col] = data[column].rolling(window=window, min_periods=1).std()
                metadata["features_created"].append(rolling_std_col)
                
                # Rolling min and max
                rolling_min_col = f"{column}_rolling_min_{window}"
                data[rolling_min_col] = data[column].rolling(window=window, min_periods=1).min()
                metadata["features_created"].append(rolling_min_col)
                
                rolling_max_col = f"{column}_rolling_max_{window}"
                data[rolling_max_col] = data[column].rolling(window=window, min_periods=1).max()
                metadata["features_created"].append(rolling_max_col)
                
                # Rolling median
                rolling_median_col = f"{column}_rolling_median_{window}"
                data[rolling_median_col] = data[column].rolling(window=window, min_periods=1).median()
                metadata["features_created"].append(rolling_median_col)
                
                # Rolling quantiles
                for q in [0.25, 0.75]:
                    rolling_q_col = f"{column}_rolling_q{int(q*100)}_{window}"
                    data[rolling_q_col] = data[column].rolling(window=window, min_periods=1).quantile(q)
                    metadata["features_created"].append(rolling_q_col)
                
                # Rolling skewness and kurtosis (for larger windows)
                if window >= 7:
                    rolling_skew_col = f"{column}_rolling_skew_{window}"
                    data[rolling_skew_col] = data[column].rolling(window=window, min_periods=3).skew()
                    metadata["features_created"].append(rolling_skew_col)
                    
                    rolling_kurt_col = f"{column}_rolling_kurt_{window}"
                    data[rolling_kurt_col] = data[column].rolling(window=window, min_periods=4).kurt()
                    metadata["features_created"].append(rolling_kurt_col)
        
        return data, metadata
    
    def _create_seasonal_features(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create seasonal decomposition features."""
        metadata = {"features_created": [], "patterns": {}}
        
        for column in value_columns:
            # Seasonal means for different periods
            for period in self.config.seasonal_periods:
                if len(data) >= period * 2:  # Need at least 2 full periods
                    # Create seasonal index
                    seasonal_index = data.index % period
                    seasonal_means = data.groupby(seasonal_index)[column].transform('mean')
                    
                    seasonal_col = f"{column}_seasonal_mean_{period}"
                    data[seasonal_col] = seasonal_means
                    metadata["features_created"].append(seasonal_col)
                    
                    # Seasonal deviations
                    seasonal_dev_col = f"{column}_seasonal_dev_{period}"
                    data[seasonal_dev_col] = data[column] - seasonal_means
                    metadata["features_created"].append(seasonal_dev_col)
                    
                    # Store seasonal pattern strength
                    seasonal_strength = seasonal_means.std() / data[column].std()
                    metadata["patterns"][f"{column}_seasonal_strength_{period}"] = seasonal_strength
        
        return data, metadata
    
    def _create_trend_features(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create trend-based features."""
        metadata = {"features_created": [], "patterns": {}}
        
        for column in value_columns:
            # Linear trend
            x = np.arange(len(data))
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, data[column].fillna(0))
                
                trend_col = f"{column}_linear_trend"
                data[trend_col] = slope * x + intercept
                metadata["features_created"].append(trend_col)
                
                # Detrended values
                detrend_col = f"{column}_detrended"
                data[detrend_col] = data[column] - data[trend_col]
                metadata["features_created"].append(detrend_col)
                
                # Store trend strength
                metadata["patterns"][f"{column}_trend_strength"] = abs(r_value)
                metadata["patterns"][f"{column}_trend_slope"] = slope
                
            except Exception as e:
                logger.warning(f"Could not calculate trend for {column}: {str(e)}")
            
            # Moving average trend
            for window in [7, 30]:
                if len(data) >= window:
                    ma_col = f"{column}_ma_trend_{window}"
                    data[ma_col] = data[column].rolling(window=window, min_periods=1).mean()
                    metadata["features_created"].append(ma_col)
                    
                    # Trend direction
                    trend_dir_col = f"{column}_trend_direction_{window}"
                    data[trend_dir_col] = (data[ma_col] > data[ma_col].shift(1)).astype(int)
                    metadata["features_created"].append(trend_dir_col)
        
        return data, metadata
    
    def _create_fourier_features(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create Fourier transform features for periodicity detection."""
        metadata = {"features_created": [], "dominant_frequencies": {}}
        
        for column in value_columns:
            # Skip if too few data points
            if len(data) < 50:
                continue
                
            try:
                # Apply FFT to detect dominant frequencies
                values = data[column].fillna(data[column].mean()).values
                fft = np.fft.fft(values)
                freqs = np.fft.fftfreq(len(values))
                
                # Find dominant frequencies (top 5)
                magnitude = np.abs(fft)
                top_freq_indices = np.argsort(magnitude)[-6:-1]  # Exclude DC component
                
                dominant_freqs = []
                for i, freq_idx in enumerate(top_freq_indices):
                    freq = freqs[freq_idx]
                    if freq != 0:  # Avoid division by zero
                        period = 1 / abs(freq)
                        dominant_freqs.append({"frequency": freq, "period": period, "magnitude": magnitude[freq_idx]})
                        
                        # Create sine and cosine features for dominant frequencies
                        sin_col = f"{column}_fourier_sin_{i}"
                        cos_col = f"{column}_fourier_cos_{i}"
                        
                        t = np.arange(len(data))
                        data[sin_col] = np.sin(2 * np.pi * freq * t)
                        data[cos_col] = np.cos(2 * np.pi * freq * t)
                        
                        metadata["features_created"].extend([sin_col, cos_col])
                        
                        # Create amplitude and phase features
                        amplitude_col = f"{column}_fourier_amplitude_{i}"
                        phase_col = f"{column}_fourier_phase_{i}"
                        
                        data[amplitude_col] = np.sqrt(data[sin_col]**2 + data[cos_col]**2)
                        data[phase_col] = np.arctan2(data[sin_col], data[cos_col])
                        
                        metadata["features_created"].extend([amplitude_col, phase_col])
                
                metadata["dominant_frequencies"][column] = dominant_freqs
                        
            except Exception as e:
                logger.warning(f"Could not create Fourier features for {column}: {str(e)}")
        
        return data, metadata
    
    def _create_time_based_aggregations(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create time-based aggregation features."""
        metadata = {"features_created": []}
        
        # Create time-based groupings
        data['hour_of_day'] = data[time_column].dt.hour
        data['day_of_week'] = data[time_column].dt.dayofweek
        data['month_of_year'] = data[time_column].dt.month
        
        for column in value_columns:
            # Hourly aggregations
            hourly_means = data.groupby('hour_of_day')[column].transform('mean')
            hourly_mean_col = f"{column}_hourly_mean"
            data[hourly_mean_col] = hourly_means
            metadata["features_created"].append(hourly_mean_col)
            
            # Daily aggregations
            daily_means = data.groupby('day_of_week')[column].transform('mean')
            daily_mean_col = f"{column}_daily_mean"
            data[daily_mean_col] = daily_means
            metadata["features_created"].append(daily_mean_col)
            
            # Monthly aggregations
            monthly_means = data.groupby('month_of_year')[column].transform('mean')
            monthly_mean_col = f"{column}_monthly_mean"
            data[monthly_mean_col] = monthly_means
            metadata["features_created"].append(monthly_mean_col)
            
            # Deviations from time-based means
            hourly_dev_col = f"{column}_hourly_deviation"
            data[hourly_dev_col] = data[column] - hourly_means
            metadata["features_created"].append(hourly_dev_col)
            
            daily_dev_col = f"{column}_daily_deviation"
            data[daily_dev_col] = data[column] - daily_means
            metadata["features_created"].append(daily_dev_col)
        
        # Clean up temporary columns
        data.drop(['hour_of_day', 'day_of_week', 'month_of_year'], axis=1, inplace=True)
        
        return data, metadata
    
    def detect_temporal_patterns(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Detect temporal patterns in the data."""
        try:
            patterns = {}
            
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Sort by time
            data_sorted = data.sort_values(time_column)
            values = data_sorted[value_column].dropna()
            
            if len(values) < 10:
                return {"error": "Insufficient data for pattern detection"}
            
            # Trend detection
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            patterns["trend"] = {
                "slope": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            }
            
            # Seasonality detection (simplified)
            if len(values) >= 24:  # Need at least 24 points
                # Check for daily pattern (if hourly data)
                if len(values) >= 24:
                    daily_pattern = values.groupby(data_sorted[time_column].dt.hour).mean()
                    daily_variation = daily_pattern.std() / daily_pattern.mean()
                    patterns["daily_seasonality"] = {
                        "strength": daily_variation,
                        "present": daily_variation > 0.1
                    }
                
                # Check for weekly pattern (if daily data)
                if len(values) >= 7:
                    weekly_pattern = values.groupby(data_sorted[time_column].dt.dayofweek).mean()
                    weekly_variation = weekly_pattern.std() / weekly_pattern.mean()
                    patterns["weekly_seasonality"] = {
                        "strength": weekly_variation,
                        "present": weekly_variation > 0.1
                    }
            
            # Volatility analysis
            if len(values) > 1:
                returns = values.pct_change().dropna()
                patterns["volatility"] = {
                    "std": returns.std(),
                    "mean_absolute_change": returns.abs().mean(),
                    "max_change": returns.abs().max()
                }
            
            # Outlier detection
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = 1.5 * IQR
            outliers = values[(values < Q1 - outlier_threshold) | (values > Q3 + outlier_threshold)]
            patterns["outliers"] = {
                "count": len(outliers),
                "percentage": len(outliers) / len(values) * 100
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting temporal patterns: {str(e)}")
            return {"error": str(e)}
    
    def create_advanced_temporal_transformations(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create advanced temporal transformations including change point detection and regime analysis."""
        try:
            enhanced_data = data.copy()
            metadata = {"features_created": [], "transformations": {}}
            
            for column in value_columns:
                # Change point detection features
                change_points = self._detect_change_points(enhanced_data[column])
                if change_points:
                    # Distance to nearest change point
                    change_point_dist_col = f"{column}_change_point_distance"
                    enhanced_data[change_point_dist_col] = self._calculate_change_point_distance(
                        enhanced_data.index, change_points
                    )
                    metadata["features_created"].append(change_point_dist_col)
                    
                    # Regime indicator (which regime are we in)
                    regime_col = f"{column}_regime"
                    enhanced_data[regime_col] = self._assign_regimes(enhanced_data.index, change_points)
                    metadata["features_created"].append(regime_col)
                
                # Momentum and acceleration features
                momentum_col = f"{column}_momentum"
                enhanced_data[momentum_col] = enhanced_data[column].diff()
                metadata["features_created"].append(momentum_col)
                
                acceleration_col = f"{column}_acceleration"
                enhanced_data[acceleration_col] = enhanced_data[momentum_col].diff()
                metadata["features_created"].append(acceleration_col)
                
                # Volatility clustering features
                volatility_col = f"{column}_volatility"
                enhanced_data[volatility_col] = enhanced_data[column].rolling(window=10).std()
                metadata["features_created"].append(volatility_col)
                
                # Relative strength index (RSI) style features
                rsi_col = f"{column}_rsi"
                enhanced_data[rsi_col] = self._calculate_rsi(enhanced_data[column])
                metadata["features_created"].append(rsi_col)
                
                # Bollinger band features
                bb_upper_col, bb_lower_col, bb_position_col = self._create_bollinger_features(
                    enhanced_data, column
                )
                metadata["features_created"].extend([bb_upper_col, bb_lower_col, bb_position_col])
                
                # Fractal dimension features
                fractal_col = f"{column}_fractal_dimension"
                enhanced_data[fractal_col] = self._calculate_fractal_dimension(enhanced_data[column])
                metadata["features_created"].append(fractal_col)
            
            return enhanced_data, metadata
            
        except Exception as e:
            logger.error(f"Error creating advanced temporal transformations: {str(e)}")
            raise FeatureEngineeringError(f"Advanced temporal transformation failed: {str(e)}")
    
    def _detect_change_points(self, series: pd.Series, min_size: int = 10) -> List[int]:
        """Detect change points in time series using simple variance-based method."""
        try:
            if len(series) < min_size * 2:
                return []
            
            change_points = []
            window_size = max(min_size, len(series) // 20)
            
            for i in range(window_size, len(series) - window_size):
                left_window = series.iloc[i-window_size:i]
                right_window = series.iloc[i:i+window_size]
                
                # Calculate variance difference
                left_var = left_window.var()
                right_var = right_window.var()
                
                # Simple change point detection based on variance change
                if abs(left_var - right_var) > (left_var + right_var) * 0.5:
                    change_points.append(i)
            
            return change_points
            
        except Exception:
            return []
    
    def _calculate_change_point_distance(self, index: pd.Index, change_points: List[int]) -> pd.Series:
        """Calculate distance to nearest change point."""
        distances = []
        for i in range(len(index)):
            if not change_points:
                distances.append(len(index))
            else:
                min_distance = min(abs(i - cp) for cp in change_points)
                distances.append(min_distance)
        return pd.Series(distances, index=index)
    
    def _assign_regimes(self, index: pd.Index, change_points: List[int]) -> pd.Series:
        """Assign regime numbers based on change points."""
        regimes = []
        current_regime = 0
        
        for i in range(len(index)):
            if i in change_points:
                current_regime += 1
            regimes.append(current_regime)
        
        return pd.Series(regimes, index=index)
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral value
            
        except Exception:
            return pd.Series([50] * len(series), index=series.index)
    
    def _create_bollinger_features(
        self, 
        data: pd.DataFrame, 
        column: str, 
        window: int = 20, 
        num_std: float = 2
    ) -> Tuple[str, str, str]:
        """Create Bollinger Band features."""
        try:
            rolling_mean = data[column].rolling(window=window).mean()
            rolling_std = data[column].rolling(window=window).std()
            
            upper_col = f"{column}_bb_upper"
            lower_col = f"{column}_bb_lower"
            position_col = f"{column}_bb_position"
            
            data[upper_col] = rolling_mean + (rolling_std * num_std)
            data[lower_col] = rolling_mean - (rolling_std * num_std)
            
            # Position within bands (0 = at lower band, 1 = at upper band)
            data[position_col] = (data[column] - data[lower_col]) / (data[upper_col] - data[lower_col])
            data[position_col] = data[position_col].clip(0, 1)
            
            return upper_col, lower_col, position_col
            
        except Exception:
            # Return dummy column names if calculation fails
            return f"{column}_bb_upper", f"{column}_bb_lower", f"{column}_bb_position"
    
    def _calculate_fractal_dimension(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Calculate fractal dimension using box-counting method (simplified)."""
        try:
            fractal_dims = []
            
            for i in range(len(series)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(series), i + window // 2)
                window_data = series.iloc[start_idx:end_idx]
                
                if len(window_data) < 10:
                    fractal_dims.append(1.5)  # Default value
                    continue
                
                # Simplified fractal dimension calculation
                # Based on the relationship between range and length
                data_range = window_data.max() - window_data.min()
                data_length = len(window_data)
                
                if data_range == 0:
                    fractal_dims.append(1.0)
                else:
                    # Simplified Hurst exponent estimation
                    log_range = np.log(data_range)
                    log_length = np.log(data_length)
                    hurst = log_range / log_length if log_length != 0 else 0.5
                    fractal_dim = 2 - hurst
                    fractal_dims.append(max(1.0, min(2.0, fractal_dim)))
            
            return pd.Series(fractal_dims, index=series.index)
            
        except Exception:
            return pd.Series([1.5] * len(series), index=series.index)