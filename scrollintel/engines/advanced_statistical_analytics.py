"""
Advanced Statistical Analysis and ML-Powered Insights Engine

This module provides sophisticated statistical analysis capabilities and machine learning
powered insights for the advanced analytics dashboard system.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Statistical libraries
try:
    from scipy import stats
    from scipy.signal import find_peaks
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of statistical analysis"""
    DESCRIPTIVE = "descriptive"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    TREND_ANALYSIS = "trend_analysis"
    SEASONALITY = "seasonality"


class InsightType(Enum):
    """Types of ML-powered insights"""
    PATTERN_DETECTION = "pattern_detection"
    TREND_IDENTIFICATION = "trend_identification"
    ANOMALY_ALERT = "anomaly_alert"
    CORRELATION_DISCOVERY = "correlation_discovery"
    FORECAST_PREDICTION = "forecast_prediction"
    CLUSTER_ANALYSIS = "cluster_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class StatisticalResult:
    """Result of statistical analysis"""
    analysis_type: AnalysisType
    metric_name: str
    result: Dict[str, Any]
    confidence_level: float
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    interpretation: str = ""
    recommendations: List[str] = None


@dataclass
class MLInsight:
    """Machine learning powered insight"""
    insight_type: InsightType
    title: str
    description: str
    confidence_score: float
    impact_score: float
    data_points: List[Dict] = None
    visualizations: List[Dict] = None
    action_items: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis"""
    analysis_types: List[AnalysisType]
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    min_data_points: int = 30
    include_visualizations: bool = True
    custom_parameters: Dict[str, Any] = None


class AdvancedStatisticalAnalytics:
    """
    Advanced statistical analysis and ML-powered insights engine
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler() if ADVANCED_STATS_AVAILABLE else None
        self.models = {}
        self.analysis_cache = {}
        
        if not ADVANCED_STATS_AVAILABLE:
            self.logger.warning("Advanced statistical libraries not available. Some features will be limited.")
    
    async def perform_comprehensive_analysis(
        self, 
        data: pd.DataFrame, 
        config: AnalysisConfig
    ) -> Dict[str, List[StatisticalResult]]:
        """
        Perform comprehensive statistical analysis on the provided data
        
        Args:
            data: DataFrame containing the data to analyze
            config: Analysis configuration
            
        Returns:
            Dictionary of analysis results grouped by type
        """
        try:
            self.logger.info("Starting comprehensive statistical analysis")
            
            results = {}
            
            # Validate data
            if data.empty or len(data) < config.min_data_points:
                raise ValueError(f"Insufficient data points. Minimum required: {config.min_data_points}")
            
            # Perform each requested analysis type
            for analysis_type in config.analysis_types:
                self.logger.info(f"Performing {analysis_type.value} analysis")
                
                if analysis_type == AnalysisType.DESCRIPTIVE:
                    results[analysis_type.value] = await self._descriptive_analysis(data, config)
                elif analysis_type == AnalysisType.CORRELATION:
                    results[analysis_type.value] = await self._correlation_analysis(data, config)
                elif analysis_type == AnalysisType.REGRESSION:
                    results[analysis_type.value] = await self._regression_analysis(data, config)
                elif analysis_type == AnalysisType.TIME_SERIES:
                    results[analysis_type.value] = await self._time_series_analysis(data, config)
                elif analysis_type == AnalysisType.ANOMALY_DETECTION:
                    results[analysis_type.value] = await self._anomaly_detection(data, config)
                elif analysis_type == AnalysisType.CLUSTERING:
                    results[analysis_type.value] = await self._clustering_analysis(data, config)
                elif analysis_type == AnalysisType.FORECASTING:
                    results[analysis_type.value] = await self._forecasting_analysis(data, config)
                elif analysis_type == AnalysisType.HYPOTHESIS_TESTING:
                    results[analysis_type.value] = await self._hypothesis_testing(data, config)
                elif analysis_type == AnalysisType.TREND_ANALYSIS:
                    results[analysis_type.value] = await self._trend_analysis(data, config)
                elif analysis_type == AnalysisType.SEASONALITY:
                    results[analysis_type.value] = await self._seasonality_analysis(data, config)
            
            self.logger.info("Comprehensive statistical analysis completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    async def generate_ml_insights(
        self, 
        data: pd.DataFrame, 
        analysis_results: Dict[str, List[StatisticalResult]]
    ) -> List[MLInsight]:
        """
        Generate ML-powered insights from statistical analysis results
        
        Args:
            data: Original data
            analysis_results: Results from statistical analysis
            
        Returns:
            List of ML-powered insights
        """
        try:
            self.logger.info("Generating ML-powered insights")
            
            insights = []
            
            # Pattern detection insights
            pattern_insights = await self._detect_patterns(data, analysis_results)
            insights.extend(pattern_insights)
            
            # Trend identification insights
            trend_insights = await self._identify_trends(data, analysis_results)
            insights.extend(trend_insights)
            
            # Anomaly alerts
            anomaly_insights = await self._generate_anomaly_alerts(data, analysis_results)
            insights.extend(anomaly_insights)
            
            # Correlation discoveries
            correlation_insights = await self._discover_correlations(data, analysis_results)
            insights.extend(correlation_insights)
            
            # Forecast predictions
            forecast_insights = await self._generate_forecast_insights(data, analysis_results)
            insights.extend(forecast_insights)
            
            # Performance optimization suggestions
            optimization_insights = await self._suggest_optimizations(data, analysis_results)
            insights.extend(optimization_insights)
            
            # Risk assessments
            risk_insights = await self._assess_risks(data, analysis_results)
            insights.extend(risk_insights)
            
            # Sort insights by impact and confidence
            insights.sort(key=lambda x: (x.impact_score * x.confidence_score), reverse=True)
            
            self.logger.info(f"Generated {len(insights)} ML-powered insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating ML insights: {str(e)}")
            raise
    
    async def _descriptive_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform descriptive statistical analysis"""
        results = []
        
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                series = data[column].dropna()
                
                if len(series) < config.min_data_points:
                    continue
                
                # Basic statistics
                desc_stats = {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis())
                }
                
                # Normality test
                if ADVANCED_STATS_AVAILABLE and len(series) >= 8:
                    shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))
                    desc_stats["normality_test"] = {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > config.significance_threshold
                    }
                
                # Interpretation
                interpretation = self._interpret_descriptive_stats(desc_stats)
                
                results.append(StatisticalResult(
                    analysis_type=AnalysisType.DESCRIPTIVE,
                    metric_name=column,
                    result=desc_stats,
                    confidence_level=config.confidence_level,
                    interpretation=interpretation
                ))
            
        except Exception as e:
            self.logger.error(f"Error in descriptive analysis: {str(e)}")
        
        return results
    
    async def _correlation_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform correlation analysis"""
        results = []
        
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return results
            
            # Pearson correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Find significant correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if abs(correlation) > 0.3:  # Threshold for meaningful correlation
                        # Calculate p-value if possible
                        p_value = None
                        if ADVANCED_STATS_AVAILABLE:
                            try:
                                _, p_value = stats.pearsonr(
                                    numeric_data[col1].dropna(), 
                                    numeric_data[col2].dropna()
                                )
                            except:
                                pass
                        
                        interpretation = self._interpret_correlation(correlation, p_value)
                        
                        results.append(StatisticalResult(
                            analysis_type=AnalysisType.CORRELATION,
                            metric_name=f"{col1} vs {col2}",
                            result={
                                "correlation": float(correlation),
                                "strength": self._correlation_strength(correlation),
                                "direction": "positive" if correlation > 0 else "negative"
                            },
                            confidence_level=config.confidence_level,
                            p_value=float(p_value) if p_value else None,
                            interpretation=interpretation
                        ))
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
        
        return results
    
    async def _time_series_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform time series analysis"""
        results = []
        
        try:
            # Look for datetime columns
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) == 0 or len(numeric_cols) == 0:
                return results
            
            # Use first datetime column as index
            time_col = datetime_cols[0]
            ts_data = data.set_index(time_col).sort_index()
            
            for metric in numeric_cols:
                series = ts_data[metric].dropna()
                
                if len(series) < config.min_data_points:
                    continue
                
                # Trend analysis
                if ADVANCED_STATS_AVAILABLE:
                    try:
                        # Decomposition
                        decomposition = seasonal_decompose(
                            series, 
                            model='additive', 
                            period=min(12, len(series) // 2)
                        )
                        
                        trend_slope = self._calculate_trend_slope(decomposition.trend.dropna())
                        
                        results.append(StatisticalResult(
                            analysis_type=AnalysisType.TIME_SERIES,
                            metric_name=metric,
                            result={
                                "trend_slope": float(trend_slope),
                                "trend_direction": "increasing" if trend_slope > 0 else "decreasing",
                                "seasonality_strength": float(np.std(decomposition.seasonal.dropna())),
                                "residual_variance": float(np.var(decomposition.resid.dropna()))
                            },
                            confidence_level=config.confidence_level,
                            interpretation=self._interpret_time_series(trend_slope)
                        ))
                    except Exception as e:
                        self.logger.warning(f"Time series decomposition failed for {metric}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in time series analysis: {str(e)}")
        
        return results
    
    async def _anomaly_detection(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform anomaly detection"""
        results = []
        
        try:
            if not ADVANCED_STATS_AVAILABLE:
                return results
            
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if numeric_data.empty:
                return results
            
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(numeric_data)
            
            anomaly_count = np.sum(anomaly_labels == -1)
            anomaly_percentage = (anomaly_count / len(anomaly_labels)) * 100
            
            # Get anomaly scores
            anomaly_scores = iso_forest.decision_function(numeric_data)
            
            results.append(StatisticalResult(
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                metric_name="Overall Dataset",
                result={
                    "anomaly_count": int(anomaly_count),
                    "anomaly_percentage": float(anomaly_percentage),
                    "mean_anomaly_score": float(np.mean(anomaly_scores)),
                    "anomaly_threshold": float(np.percentile(anomaly_scores, 10))
                },
                confidence_level=config.confidence_level,
                interpretation=self._interpret_anomalies(anomaly_percentage)
            ))
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
        
        return results
    
    async def _clustering_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform clustering analysis"""
        results = []
        
        try:
            if not ADVANCED_STATS_AVAILABLE:
                return results
            
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if numeric_data.empty or len(numeric_data) < 10:
                return results
            
            # Standardize data
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # K-means clustering with different k values
            best_k = 2
            best_score = -1
            
            for k in range(2, min(10, len(numeric_data) // 2)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_data)
                    score = silhouette_score(scaled_data, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            
            # Final clustering with best k
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            results.append(StatisticalResult(
                analysis_type=AnalysisType.CLUSTERING,
                metric_name="Dataset Clustering",
                result={
                    "optimal_clusters": int(best_k),
                    "silhouette_score": float(best_score),
                    "cluster_sizes": [int(np.sum(cluster_labels == i)) for i in range(best_k)],
                    "inertia": float(kmeans.inertia_)
                },
                confidence_level=config.confidence_level,
                interpretation=self._interpret_clustering(best_k, best_score)
            ))
            
        except Exception as e:
            self.logger.error(f"Error in clustering analysis: {str(e)}")
        
        return results
    
    async def _forecasting_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform forecasting analysis"""
        results = []
        
        try:
            if not ADVANCED_STATS_AVAILABLE:
                return results
            
            # Look for time series data
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) == 0 or len(numeric_cols) == 0:
                return results
            
            time_col = datetime_cols[0]
            ts_data = data.set_index(time_col).sort_index()
            
            for metric in numeric_cols[:3]:  # Limit to first 3 metrics
                series = ts_data[metric].dropna()
                
                if len(series) < config.min_data_points:
                    continue
                
                try:
                    # Simple ARIMA forecast
                    model = ARIMA(series, order=(1, 1, 1))
                    fitted_model = model.fit()
                    
                    # Forecast next 5 periods
                    forecast = fitted_model.forecast(steps=5)
                    forecast_conf_int = fitted_model.get_forecast(steps=5).conf_int()
                    
                    results.append(StatisticalResult(
                        analysis_type=AnalysisType.FORECASTING,
                        metric_name=metric,
                        result={
                            "forecast_values": [float(x) for x in forecast],
                            "confidence_intervals": {
                                "lower": [float(x) for x in forecast_conf_int.iloc[:, 0]],
                                "upper": [float(x) for x in forecast_conf_int.iloc[:, 1]]
                            },
                            "model_aic": float(fitted_model.aic),
                            "model_bic": float(fitted_model.bic)
                        },
                        confidence_level=config.confidence_level,
                        interpretation=self._interpret_forecast(forecast, series)
                    ))
                    
                except Exception as e:
                    self.logger.warning(f"Forecasting failed for {metric}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in forecasting analysis: {str(e)}")
        
        return results
    
    async def _hypothesis_testing(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform hypothesis testing"""
        results = []
        
        try:
            if not ADVANCED_STATS_AVAILABLE:
                return results
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # One-sample t-tests against zero
            for col in numeric_cols:
                series = data[col].dropna()
                
                if len(series) < config.min_data_points:
                    continue
                
                # One-sample t-test
                t_stat, p_value = stats.ttest_1samp(series, 0)
                
                results.append(StatisticalResult(
                    analysis_type=AnalysisType.HYPOTHESIS_TESTING,
                    metric_name=f"{col} vs Zero",
                    result={
                        "t_statistic": float(t_stat),
                        "degrees_freedom": len(series) - 1,
                        "test_type": "one_sample_t_test",
                        "null_hypothesis": f"Mean of {col} equals 0",
                        "alternative_hypothesis": f"Mean of {col} does not equal 0"
                    },
                    confidence_level=config.confidence_level,
                    p_value=float(p_value),
                    interpretation=self._interpret_hypothesis_test(p_value, config.significance_threshold)
                ))
            
        except Exception as e:
            self.logger.error(f"Error in hypothesis testing: {str(e)}")
        
        return results
    
    async def _trend_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform trend analysis"""
        results = []
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                series = data[col].dropna()
                
                if len(series) < config.min_data_points:
                    continue
                
                # Calculate trend using linear regression
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                
                # Mann-Kendall trend test if available
                mk_trend = None
                mk_p_value = None
                
                results.append(StatisticalResult(
                    analysis_type=AnalysisType.TREND_ANALYSIS,
                    metric_name=col,
                    result={
                        "linear_slope": float(slope),
                        "r_squared": float(r_value ** 2),
                        "trend_direction": "increasing" if slope > 0 else "decreasing",
                        "trend_strength": abs(float(r_value)),
                        "standard_error": float(std_err)
                    },
                    confidence_level=config.confidence_level,
                    p_value=float(p_value),
                    interpretation=self._interpret_trend(slope, r_value, p_value)
                ))
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
        
        return results
    
    async def _seasonality_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> List[StatisticalResult]:
        """Perform seasonality analysis"""
        results = []
        
        try:
            if not ADVANCED_STATS_AVAILABLE:
                return results
            
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) == 0 or len(numeric_cols) == 0:
                return results
            
            time_col = datetime_cols[0]
            ts_data = data.set_index(time_col).sort_index()
            
            for metric in numeric_cols:
                series = ts_data[metric].dropna()
                
                if len(series) < config.min_data_points:
                    continue
                
                try:
                    # Seasonal decomposition
                    decomposition = seasonal_decompose(
                        series, 
                        model='additive', 
                        period=min(12, len(series) // 2)
                    )
                    
                    seasonal_strength = np.std(decomposition.seasonal.dropna()) / np.std(series)
                    
                    results.append(StatisticalResult(
                        analysis_type=AnalysisType.SEASONALITY,
                        metric_name=metric,
                        result={
                            "seasonal_strength": float(seasonal_strength),
                            "has_seasonality": seasonal_strength > 0.1,
                            "seasonal_period": min(12, len(series) // 2),
                            "seasonal_variance": float(np.var(decomposition.seasonal.dropna()))
                        },
                        confidence_level=config.confidence_level,
                        interpretation=self._interpret_seasonality(seasonal_strength)
                    ))
                    
                except Exception as e:
                    self.logger.warning(f"Seasonality analysis failed for {metric}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in seasonality analysis: {str(e)}")
        
        return results
    
    # ML Insight Generation Methods
    
    async def _detect_patterns(self, data: pd.DataFrame, analysis_results: Dict) -> List[MLInsight]:
        """Detect patterns in the data"""
        insights = []
        
        try:
            # Look for correlation patterns
            if 'correlation' in analysis_results:
                strong_correlations = [
                    result for result in analysis_results['correlation']
                    if abs(result.result.get('correlation', 0)) > 0.7
                ]
                
                if strong_correlations:
                    insights.append(MLInsight(
                        insight_type=InsightType.PATTERN_DETECTION,
                        title="Strong Correlation Patterns Detected",
                        description=f"Found {len(strong_correlations)} strong correlations between metrics",
                        confidence_score=0.85,
                        impact_score=0.7,
                        action_items=[
                            "Investigate causal relationships between strongly correlated metrics",
                            "Consider using correlated metrics for predictive modeling",
                            "Monitor for changes in correlation patterns"
                        ]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
        
        return insights
    
    async def _identify_trends(self, data: pd.DataFrame, analysis_results: Dict) -> List[MLInsight]:
        """Identify significant trends"""
        insights = []
        
        try:
            if 'trend_analysis' in analysis_results:
                significant_trends = [
                    result for result in analysis_results['trend_analysis']
                    if result.p_value and result.p_value < 0.05 and 
                    result.result.get('trend_strength', 0) > 0.5
                ]
                
                for trend in significant_trends:
                    direction = trend.result.get('trend_direction', 'unknown')
                    strength = trend.result.get('trend_strength', 0)
                    
                    insights.append(MLInsight(
                        insight_type=InsightType.TREND_IDENTIFICATION,
                        title=f"Significant {direction.title()} Trend in {trend.metric_name}",
                        description=f"Strong {direction} trend detected with {strength:.2f} correlation strength",
                        confidence_score=min(0.95, strength + 0.2),
                        impact_score=strength,
                        action_items=[
                            f"Monitor {trend.metric_name} closely for continued {direction} movement",
                            "Investigate underlying factors driving this trend",
                            "Consider adjusting strategies based on trend direction"
                        ]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error identifying trends: {str(e)}")
        
        return insights
    
    async def _generate_anomaly_alerts(self, data: pd.DataFrame, analysis_results: Dict) -> List[MLInsight]:
        """Generate anomaly alerts"""
        insights = []
        
        try:
            if 'anomaly_detection' in analysis_results:
                for result in analysis_results['anomaly_detection']:
                    anomaly_pct = result.result.get('anomaly_percentage', 0)
                    
                    if anomaly_pct > 5:  # More than 5% anomalies
                        insights.append(MLInsight(
                            insight_type=InsightType.ANOMALY_ALERT,
                            title="High Anomaly Rate Detected",
                            description=f"Detected {anomaly_pct:.1f}% anomalous data points",
                            confidence_score=0.8,
                            impact_score=min(1.0, anomaly_pct / 10),
                            action_items=[
                                "Investigate root causes of anomalous data points",
                                "Review data collection and processing procedures",
                                "Consider implementing real-time anomaly monitoring"
                            ]
                        ))
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly alerts: {str(e)}")
        
        return insights
    
    async def _discover_correlations(self, data: pd.DataFrame, analysis_results: Dict) -> List[MLInsight]:
        """Discover interesting correlations"""
        insights = []
        
        try:
            if 'correlation' in analysis_results:
                unexpected_correlations = [
                    result for result in analysis_results['correlation']
                    if abs(result.result.get('correlation', 0)) > 0.6 and
                    result.p_value and result.p_value < 0.01
                ]
                
                if unexpected_correlations:
                    insights.append(MLInsight(
                        insight_type=InsightType.CORRELATION_DISCOVERY,
                        title="Unexpected Correlations Discovered",
                        description=f"Found {len(unexpected_correlations)} statistically significant correlations",
                        confidence_score=0.9,
                        impact_score=0.8,
                        action_items=[
                            "Explore business implications of discovered correlations",
                            "Test for causation vs correlation",
                            "Leverage correlations for predictive analytics"
                        ]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error discovering correlations: {str(e)}")
        
        return insights
    
    async def _generate_forecast_insights(self, data: pd.DataFrame, analysis_results: Dict) -> List[MLInsight]:
        """Generate forecast-based insights"""
        insights = []
        
        try:
            if 'forecasting' in analysis_results:
                for result in analysis_results['forecasting']:
                    forecast_values = result.result.get('forecast_values', [])
                    
                    if forecast_values:
                        current_value = data[result.metric_name].iloc[-1] if result.metric_name in data.columns else 0
                        forecast_change = (forecast_values[-1] - current_value) / current_value * 100
                        
                        insights.append(MLInsight(
                            insight_type=InsightType.FORECAST_PREDICTION,
                            title=f"Forecast for {result.metric_name}",
                            description=f"Predicted {forecast_change:+.1f}% change over next 5 periods",
                            confidence_score=0.75,
                            impact_score=min(1.0, abs(forecast_change) / 50),
                            action_items=[
                                "Plan for predicted changes in metrics",
                                "Monitor actual vs predicted values",
                                "Adjust strategies based on forecasts"
                            ]
                        ))
            
        except Exception as e:
            self.logger.error(f"Error generating forecast insights: {str(e)}")
        
        return insights
    
    async def _suggest_optimizations(self, data: pd.DataFrame, analysis_results: Dict) -> List[MLInsight]:
        """Suggest performance optimizations"""
        insights = []
        
        try:
            # Look for optimization opportunities based on analysis
            if 'descriptive' in analysis_results:
                high_variance_metrics = [
                    result for result in analysis_results['descriptive']
                    if result.result.get('std', 0) / result.result.get('mean', 1) > 0.5
                ]
                
                if high_variance_metrics:
                    insights.append(MLInsight(
                        insight_type=InsightType.PERFORMANCE_OPTIMIZATION,
                        title="High Variability Metrics Identified",
                        description=f"Found {len(high_variance_metrics)} metrics with high variability",
                        confidence_score=0.8,
                        impact_score=0.7,
                        action_items=[
                            "Investigate causes of high variability",
                            "Implement process controls to reduce variance",
                            "Monitor variability trends over time"
                        ]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error suggesting optimizations: {str(e)}")
        
        return insights
    
    async def _assess_risks(self, data: pd.DataFrame, analysis_results: Dict) -> List[MLInsight]:
        """Assess risks based on analysis"""
        insights = []
        
        try:
            # Risk assessment based on trends and anomalies
            risk_factors = []
            
            if 'trend_analysis' in analysis_results:
                declining_trends = [
                    result for result in analysis_results['trend_analysis']
                    if result.result.get('trend_direction') == 'decreasing' and
                    result.result.get('trend_strength', 0) > 0.5
                ]
                risk_factors.extend(declining_trends)
            
            if 'anomaly_detection' in analysis_results:
                high_anomaly_results = [
                    result for result in analysis_results['anomaly_detection']
                    if result.result.get('anomaly_percentage', 0) > 10
                ]
                risk_factors.extend(high_anomaly_results)
            
            if risk_factors:
                insights.append(MLInsight(
                    insight_type=InsightType.RISK_ASSESSMENT,
                    title="Potential Risk Factors Identified",
                    description=f"Identified {len(risk_factors)} potential risk indicators",
                    confidence_score=0.75,
                    impact_score=0.9,
                    action_items=[
                        "Develop mitigation strategies for identified risks",
                        "Implement early warning systems",
                        "Regular monitoring of risk indicators"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error assessing risks: {str(e)}")
        
        return insights
    
    # Helper methods for interpretation
    
    def _interpret_descriptive_stats(self, stats: Dict) -> str:
        """Interpret descriptive statistics"""
        skewness = stats.get('skewness', 0)
        kurtosis = stats.get('kurtosis', 0)
        
        interpretation = f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}. "
        
        if abs(skewness) > 1:
            interpretation += f"High {'positive' if skewness > 0 else 'negative'} skewness ({skewness:.2f}). "
        
        if abs(kurtosis) > 1:
            interpretation += f"{'Heavy' if kurtosis > 0 else 'Light'} tails (kurtosis: {kurtosis:.2f}). "
        
        return interpretation
    
    def _interpret_correlation(self, correlation: float, p_value: Optional[float]) -> str:
        """Interpret correlation results"""
        strength = self._correlation_strength(correlation)
        direction = "positive" if correlation > 0 else "negative"
        
        interpretation = f"{strength.title()} {direction} correlation ({correlation:.3f}). "
        
        if p_value:
            if p_value < 0.01:
                interpretation += "Highly statistically significant."
            elif p_value < 0.05:
                interpretation += "Statistically significant."
            else:
                interpretation += "Not statistically significant."
        
        return interpretation
    
    def _correlation_strength(self, correlation: float) -> str:
        """Determine correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        else:
            return "weak"
    
    def _interpret_time_series(self, trend_slope: float) -> str:
        """Interpret time series analysis"""
        if abs(trend_slope) < 0.01:
            return "No significant trend detected."
        elif trend_slope > 0:
            return f"Increasing trend with slope {trend_slope:.4f}."
        else:
            return f"Decreasing trend with slope {trend_slope:.4f}."
    
    def _interpret_anomalies(self, anomaly_percentage: float) -> str:
        """Interpret anomaly detection results"""
        if anomaly_percentage < 5:
            return "Normal anomaly rate detected."
        elif anomaly_percentage < 15:
            return "Elevated anomaly rate - investigation recommended."
        else:
            return "High anomaly rate - immediate attention required."
    
    def _interpret_clustering(self, n_clusters: int, silhouette_score: float) -> str:
        """Interpret clustering results"""
        if silhouette_score > 0.7:
            quality = "excellent"
        elif silhouette_score > 0.5:
            quality = "good"
        elif silhouette_score > 0.25:
            quality = "fair"
        else:
            quality = "poor"
        
        return f"Optimal {n_clusters} clusters identified with {quality} separation (silhouette: {silhouette_score:.3f})."
    
    def _interpret_forecast(self, forecast: np.ndarray, historical: pd.Series) -> str:
        """Interpret forecasting results"""
        forecast_mean = np.mean(forecast)
        historical_mean = historical.mean()
        change_pct = (forecast_mean - historical_mean) / historical_mean * 100
        
        return f"Forecast shows {change_pct:+.1f}% change from historical average."
    
    def _interpret_hypothesis_test(self, p_value: float, alpha: float) -> str:
        """Interpret hypothesis test results"""
        if p_value < alpha:
            return f"Reject null hypothesis (p={p_value:.4f} < α={alpha})."
        else:
            return f"Fail to reject null hypothesis (p={p_value:.4f} ≥ α={alpha})."
    
    def _interpret_trend(self, slope: float, r_value: float, p_value: float) -> str:
        """Interpret trend analysis"""
        direction = "increasing" if slope > 0 else "decreasing"
        strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
        significance = "significant" if p_value < 0.05 else "not significant"
        
        return f"{strength.title()} {direction} trend (R²={r_value**2:.3f}), {significance} (p={p_value:.4f})."
    
    def _interpret_seasonality(self, seasonal_strength: float) -> str:
        """Interpret seasonality analysis"""
        if seasonal_strength > 0.3:
            return f"Strong seasonality detected (strength: {seasonal_strength:.3f})."
        elif seasonal_strength > 0.1:
            return f"Moderate seasonality detected (strength: {seasonal_strength:.3f})."
        else:
            return "No significant seasonality detected."
    
    def _calculate_trend_slope(self, trend_series: pd.Series) -> float:
        """Calculate trend slope from decomposed trend"""
        if len(trend_series) < 2:
            return 0.0
        
        x = np.arange(len(trend_series))
        slope, _, _, _, _ = stats.linregress(x, trend_series)
        return slope