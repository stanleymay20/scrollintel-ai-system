"""
ScrollInsightRadar Engine - Automated Pattern Detection and Trend Analysis

This engine provides comprehensive pattern detection across all data sources,
trend analysis with statistical significance testing, anomaly detection,
insight ranking, and automated notifications.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

from ..core.interfaces import BaseEngine
from ..models.schemas import InsightRadarResult, PatternDetectionConfig, TrendAnalysis, AnomalyDetection

logger = logging.getLogger(__name__)

class ScrollInsightRadar(BaseEngine):
    """
    Advanced pattern detection and insight radar engine that automatically
    identifies trends, anomalies, and patterns across all data sources.
    """
    
    def __init__(self):
        super().__init__(engine_id="scroll_insight_radar", name="ScrollInsightRadar")
        self.version = "1.0.0"
        self.capabilities = [
            "pattern_detection",
            "trend_analysis", 
            "anomaly_detection",
            "insight_ranking",
            "automated_notifications",
            "statistical_significance_testing"
        ]
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    async def detect_patterns(self, data: pd.DataFrame, config: PatternDetectionConfig = None) -> Dict[str, Any]:
        """
        Detect patterns across all data sources with comprehensive analysis.
        
        Args:
            data: Input DataFrame to analyze
            config: Configuration for pattern detection
            
        Returns:
            Dictionary containing detected patterns and insights
        """
        try:
            if config is None:
                config = PatternDetectionConfig()
                
            logger.info(f"Starting pattern detection on dataset with {len(data)} rows")
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "dataset_info": self._get_dataset_info(data),
                "patterns": {},
                "trends": {},
                "anomalies": {},
                "insights": [],
                "statistical_tests": {},
                "business_impact_score": 0.0
            }
            
            # Detect different types of patterns
            results["patterns"]["correlation_patterns"] = await self._detect_correlation_patterns(data)
            results["patterns"]["seasonal_patterns"] = await self._detect_seasonal_patterns(data)
            results["patterns"]["clustering_patterns"] = await self._detect_clustering_patterns(data)
            results["patterns"]["distribution_patterns"] = await self._detect_distribution_patterns(data)
            
            # Perform trend analysis
            results["trends"] = await self._analyze_trends(data, config)
            
            # Detect anomalies
            results["anomalies"] = await self._detect_anomalies(data, config)
            
            # Generate insights and rank them
            results["insights"] = await self._generate_insights(results)
            results["business_impact_score"] = await self._calculate_business_impact(results)
            
            # Perform statistical significance testing
            results["statistical_tests"] = await self._perform_statistical_tests(data)
            
            logger.info(f"Pattern detection completed. Found {len(results['insights'])} insights")
            return results
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    async def _detect_correlation_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect correlation patterns between variables."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return {"message": "Insufficient numeric columns for correlation analysis"}
            
            correlation_matrix = data[numeric_cols].corr()
            
            # Find strong correlations (> 0.7 or < -0.7)
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            "variable_1": correlation_matrix.columns[i],
                            "variable_2": correlation_matrix.columns[j],
                            "correlation": float(corr_value),
                            "strength": "strong" if abs(corr_value) > 0.8 else "moderate",
                            "direction": "positive" if corr_value > 0 else "negative"
                        })
            
            return {
                "correlation_matrix": correlation_matrix.to_dict(),
                "strong_correlations": strong_correlations,
                "total_correlations_found": len(strong_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error in correlation pattern detection: {str(e)}")
            return {"error": str(e)}
    
    async def _detect_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal and cyclical patterns in time series data."""
        try:
            # Look for datetime columns
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) == 0 or len(numeric_cols) == 0:
                return {"message": "No datetime or numeric columns found for seasonal analysis"}
            
            seasonal_patterns = []
            
            for datetime_col in datetime_cols[:1]:  # Analyze first datetime column
                for numeric_col in numeric_cols[:3]:  # Analyze first 3 numeric columns
                    try:
                        # Prepare time series data
                        ts_data = data[[datetime_col, numeric_col]].dropna()
                        ts_data = ts_data.set_index(datetime_col).sort_index()
                        
                        if len(ts_data) < 24:  # Need minimum data points
                            continue
                            
                        # Perform seasonal decomposition
                        decomposition = seasonal_decompose(
                            ts_data[numeric_col], 
                            model='additive', 
                            period=min(12, len(ts_data)//2)
                        )
                        
                        # Calculate seasonality strength
                        seasonal_strength = np.var(decomposition.seasonal) / np.var(ts_data[numeric_col])
                        trend_strength = np.var(decomposition.trend.dropna()) / np.var(ts_data[numeric_col])
                        
                        seasonal_patterns.append({
                            "datetime_column": datetime_col,
                            "numeric_column": numeric_col,
                            "seasonal_strength": float(seasonal_strength),
                            "trend_strength": float(trend_strength),
                            "has_seasonality": seasonal_strength > 0.1,
                            "has_trend": trend_strength > 0.1
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not analyze seasonality for {numeric_col}: {str(e)}")
                        continue
            
            return {
                "seasonal_patterns": seasonal_patterns,
                "patterns_found": len(seasonal_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern detection: {str(e)}")
            return {"error": str(e)}
    
    async def _detect_clustering_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect clustering patterns in the data."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return {"message": "Insufficient numeric columns for clustering analysis"}
            
            # Prepare data for clustering
            cluster_data = data[numeric_cols].dropna()
            if len(cluster_data) < 10:
                return {"message": "Insufficient data points for clustering"}
            
            # Standardize the data
            scaled_data = self.scaler.fit_transform(cluster_data)
            
            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(scaled_data)
            
            # Analyze clusters
            unique_clusters = np.unique(clusters)
            cluster_info = []
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Noise points
                    continue
                    
                cluster_mask = clusters == cluster_id
                cluster_size = np.sum(cluster_mask)
                cluster_data_subset = cluster_data[cluster_mask]
                
                cluster_info.append({
                    "cluster_id": int(cluster_id),
                    "size": int(cluster_size),
                    "percentage": float(cluster_size / len(cluster_data) * 100),
                    "centroid": cluster_data_subset.mean().to_dict()
                })
            
            noise_points = np.sum(clusters == -1)
            
            return {
                "clusters_found": len(cluster_info),
                "cluster_details": cluster_info,
                "noise_points": int(noise_points),
                "noise_percentage": float(noise_points / len(cluster_data) * 100)
            }
            
        except Exception as e:
            logger.error(f"Error in clustering pattern detection: {str(e)}")
            return {"error": str(e)}
    
    async def _detect_distribution_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect distribution patterns and statistical properties."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {"message": "No numeric columns found for distribution analysis"}
            
            distribution_patterns = []
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) < 10:
                    continue
                
                # Basic statistics
                stats_dict = {
                    "column": col,
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "skewness": float(stats.skew(col_data)),
                    "kurtosis": float(stats.kurtosis(col_data)),
                    "min": float(col_data.min()),
                    "max": float(col_data.max())
                }
                
                # Test for normality
                _, p_value = stats.normaltest(col_data)
                stats_dict["is_normal"] = p_value > 0.05
                stats_dict["normality_p_value"] = float(p_value)
                
                # Detect outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                stats_dict["outlier_count"] = len(outliers)
                stats_dict["outlier_percentage"] = float(len(outliers) / len(col_data) * 100)
                
                distribution_patterns.append(stats_dict)
            
            return {
                "distribution_patterns": distribution_patterns,
                "columns_analyzed": len(distribution_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error in distribution pattern detection: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_trends(self, data: pd.DataFrame, config: PatternDetectionConfig) -> Dict[str, Any]:
        """Analyze trends in the data with statistical significance testing."""
        try:
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) == 0 or len(numeric_cols) == 0:
                return {"message": "No datetime or numeric columns found for trend analysis"}
            
            trend_results = []
            
            for datetime_col in datetime_cols[:1]:  # Analyze first datetime column
                for numeric_col in numeric_cols[:5]:  # Analyze first 5 numeric columns
                    try:
                        # Prepare time series data
                        ts_data = data[[datetime_col, numeric_col]].dropna()
                        ts_data = ts_data.set_index(datetime_col).sort_index()
                        
                        if len(ts_data) < 10:
                            continue
                        
                        # Calculate trend using linear regression
                        x = np.arange(len(ts_data))
                        y = ts_data[numeric_col].values
                        
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        # Determine trend direction and significance
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        is_significant = p_value < 0.05
                        
                        # Calculate trend strength
                        trend_strength = abs(r_value)
                        
                        trend_results.append({
                            "datetime_column": datetime_col,
                            "numeric_column": numeric_col,
                            "slope": float(slope),
                            "r_squared": float(r_value ** 2),
                            "p_value": float(p_value),
                            "trend_direction": trend_direction,
                            "is_significant": is_significant,
                            "trend_strength": float(trend_strength),
                            "confidence_level": "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not analyze trend for {numeric_col}: {str(e)}")
                        continue
            
            return {
                "trend_analysis": trend_results,
                "significant_trends": len([t for t in trend_results if t["is_significant"]]),
                "total_trends_analyzed": len(trend_results)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {"error": str(e)}
    
    async def _detect_anomalies(self, data: pd.DataFrame, config: PatternDetectionConfig) -> Dict[str, Any]:
        """Detect anomalies and unusual patterns in the data."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {"message": "No numeric columns found for anomaly detection"}
            
            anomaly_results = []
            
            # Prepare data for anomaly detection
            anomaly_data = data[numeric_cols].dropna()
            if len(anomaly_data) < 10:
                return {"message": "Insufficient data for anomaly detection"}
            
            # Apply Isolation Forest
            outlier_scores = self.isolation_forest.fit_predict(anomaly_data)
            anomaly_scores = self.isolation_forest.decision_function(anomaly_data)
            
            # Identify anomalies
            anomaly_indices = np.where(outlier_scores == -1)[0]
            
            for idx in anomaly_indices[:20]:  # Limit to top 20 anomalies
                anomaly_row = anomaly_data.iloc[idx]
                anomaly_results.append({
                    "index": int(idx),
                    "anomaly_score": float(anomaly_scores[idx]),
                    "values": anomaly_row.to_dict(),
                    "severity": "high" if anomaly_scores[idx] < -0.5 else "medium"
                })
            
            # Statistical anomaly detection for each column
            column_anomalies = {}
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) < 10:
                    continue
                
                # Z-score based anomaly detection
                z_scores = np.abs(stats.zscore(col_data))
                z_anomalies = np.where(z_scores > 3)[0]
                
                # IQR based anomaly detection
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                iqr_anomalies = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                
                column_anomalies[col] = {
                    "z_score_anomalies": len(z_anomalies),
                    "iqr_anomalies": len(iqr_anomalies),
                    "total_anomalies": len(z_anomalies) + len(iqr_anomalies)
                }
            
            return {
                "isolation_forest_anomalies": anomaly_results,
                "column_anomalies": column_anomalies,
                "total_anomalies_found": len(anomaly_results),
                "anomaly_percentage": float(len(anomaly_results) / len(anomaly_data) * 100)
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ranked insights from pattern detection results."""
        try:
            insights = []
            
            # Generate insights from correlation patterns
            if "correlation_patterns" in analysis_results.get("patterns", {}):
                corr_patterns = analysis_results["patterns"]["correlation_patterns"]
                if "strong_correlations" in corr_patterns:
                    for corr in corr_patterns["strong_correlations"]:
                        insights.append({
                            "type": "correlation",
                            "title": f"Strong {corr['direction']} correlation detected",
                            "description": f"{corr['variable_1']} and {corr['variable_2']} show a {corr['strength']} {corr['direction']} correlation ({corr['correlation']:.3f})",
                            "impact_score": abs(corr['correlation']) * 0.8,
                            "actionable": True,
                            "category": "relationship"
                        })
            
            # Generate insights from trend analysis
            if "trend_analysis" in analysis_results.get("trends", {}):
                for trend in analysis_results["trends"]["trend_analysis"]:
                    if trend["is_significant"]:
                        insights.append({
                            "type": "trend",
                            "title": f"Significant {trend['trend_direction']} trend detected",
                            "description": f"{trend['numeric_column']} shows a {trend['confidence_level']} confidence {trend['trend_direction']} trend (R² = {trend['r_squared']:.3f})",
                            "impact_score": trend["trend_strength"] * 0.9,
                            "actionable": True,
                            "category": "temporal"
                        })
            
            # Generate insights from anomaly detection
            if "total_anomalies_found" in analysis_results.get("anomalies", {}):
                anomaly_count = analysis_results["anomalies"]["total_anomalies_found"]
                if anomaly_count > 0:
                    insights.append({
                        "type": "anomaly",
                        "title": f"{anomaly_count} anomalies detected",
                        "description": f"Found {anomaly_count} unusual data points that deviate from normal patterns",
                        "impact_score": min(anomaly_count / 10, 1.0) * 0.7,
                        "actionable": True,
                        "category": "quality"
                    })
            
            # Generate insights from clustering patterns
            if "clusters_found" in analysis_results.get("patterns", {}).get("clustering_patterns", {}):
                cluster_count = analysis_results["patterns"]["clustering_patterns"]["clusters_found"]
                if cluster_count > 1:
                    insights.append({
                        "type": "clustering",
                        "title": f"{cluster_count} distinct data clusters identified",
                        "description": f"Data naturally groups into {cluster_count} distinct clusters, suggesting different behavioral patterns",
                        "impact_score": min(cluster_count / 5, 1.0) * 0.6,
                        "actionable": True,
                        "category": "segmentation"
                    })
            
            # Sort insights by impact score
            insights.sort(key=lambda x: x["impact_score"], reverse=True)
            
            # Add ranking
            for i, insight in enumerate(insights):
                insight["rank"] = i + 1
                insight["priority"] = "high" if insight["impact_score"] > 0.7 else "medium" if insight["impact_score"] > 0.4 else "low"
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    async def _calculate_business_impact(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall business impact score."""
        try:
            impact_score = 0.0
            
            # Weight different types of findings
            weights = {
                "correlation": 0.3,
                "trend": 0.4,
                "anomaly": 0.2,
                "clustering": 0.1
            }
            
            insights = analysis_results.get("insights", [])
            
            for insight in insights:
                insight_type = insight.get("type", "")
                if insight_type in weights:
                    impact_score += insight["impact_score"] * weights[insight_type]
            
            return min(impact_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating business impact: {str(e)}")
            return 0.0
    
    async def _perform_statistical_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        try:
            test_results = {}
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Perform correlation significance tests
                correlation_tests = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        data_subset = data[[col1, col2]].dropna()
                        
                        if len(data_subset) > 10:
                            corr_coef, p_value = stats.pearsonr(data_subset[col1], data_subset[col2])
                            correlation_tests.append({
                                "variable_1": col1,
                                "variable_2": col2,
                                "correlation": float(corr_coef),
                                "p_value": float(p_value),
                                "is_significant": p_value < 0.05
                            })
                
                test_results["correlation_tests"] = correlation_tests
            
            # Perform normality tests
            normality_tests = []
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 10:
                    _, p_value = stats.normaltest(col_data)
                    normality_tests.append({
                        "column": col,
                        "p_value": float(p_value),
                        "is_normal": p_value > 0.05
                    })
            
            test_results["normality_tests"] = normality_tests
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error in statistical tests: {str(e)}")
            return {"error": str(e)}
    
    def _get_dataset_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            "rows": len(data),
            "columns": len(data.columns),
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(data.select_dtypes(include=['object']).columns),
            "datetime_columns": len(data.select_dtypes(include=['datetime64']).columns),
            "missing_values": int(data.isnull().sum().sum()),
            "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
    
    async def send_insight_notification(self, insights: List[Dict[str, Any]], user_id: str = None) -> bool:
        """Send automated notifications for high-priority insights."""
        try:
            high_priority_insights = [i for i in insights if i.get("priority") == "high"]
            
            if not high_priority_insights:
                return False
            
            # In a real implementation, this would send notifications via email, Slack, etc.
            logger.info(f"Would send notification for {len(high_priority_insights)} high-priority insights to user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending insight notification: {str(e)}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the ScrollInsightRadar engine."""
        return {
            "status": "healthy",
            "engine": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "last_check": datetime.now().isoformat()
        }
    
    # Implementation of abstract methods from BaseEngine
    async def initialize(self) -> None:
        """Initialize the ScrollInsightRadar engine."""
        logger.info(f"Initializing {self.name} engine...")
        self.is_initialized = True
        logger.info(f"{self.name} engine initialized successfully")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process input data for pattern detection."""
        if parameters is None:
            parameters = {}
        
        # Convert input data to DataFrame if needed
        if isinstance(input_data, pd.DataFrame):
            data = input_data
        else:
            # Try to convert other formats to DataFrame
            try:
                data = pd.DataFrame(input_data)
            except Exception as e:
                raise ValueError(f"Cannot convert input data to DataFrame: {str(e)}")
        
        # Create config from parameters
        config = PatternDetectionConfig(**parameters) if parameters else PatternDetectionConfig()
        
        # Perform pattern detection
        return await self.detect_patterns(data, config)
    
    async def cleanup(self) -> None:
        """Clean up resources used by the engine."""
        logger.info(f"Cleaning up {self.name} engine...")
        self.is_initialized = False
        logger.info(f"{self.name} engine cleanup completed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the engine."""
        return {
            "name": self.name,
            "version": self.version,
            "is_initialized": self.is_initialized,
            "capabilities": self.capabilities,
            "status": "ready" if self.is_initialized else "not_initialized",
            "last_updated": datetime.now().isoformat()
        }