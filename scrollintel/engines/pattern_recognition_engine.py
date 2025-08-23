"""
Pattern Recognition Engine for Business Opportunity Identification

This engine identifies patterns, trends, anomalies, and business opportunities
in enterprise data using advanced machine learning and statistical techniques.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import json

from ..models.advanced_analytics_models import (
    PatternRecognitionRequest, RecognizedPattern, PatternRecognitionResult,
    PatternType, AnalyticsInsight
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PatternRecognitionEngine:
    """
    Advanced pattern recognition engine for identifying business opportunities.
    
    Capabilities:
    - Trend detection and analysis
    - Anomaly detection using multiple algorithms
    - Cyclical pattern identification
    - Clustering and segmentation analysis
    - Correlation pattern discovery
    - Real-time pattern monitoring
    - Predictive pattern modeling
    """
    
    def __init__(self):
        self.data_cache = {}
        self.pattern_cache = {}
        self.model_cache = {}
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    async def recognize_patterns(self, request: PatternRecognitionRequest) -> PatternRecognitionResult:
        """
        Recognize patterns in business data based on the request parameters.
        
        Args:
            request: Pattern recognition request with data source and parameters
            
        Returns:
            Comprehensive pattern recognition results
        """
        start_time = datetime.utcnow()
        
        try:
            # Load and prepare data
            data = await self._load_data_from_source(request.data_source)
            processed_data = await self._preprocess_data(data, request)
            
            recognized_patterns = []
            
            # Perform pattern recognition based on requested types
            for pattern_type in request.pattern_types:
                patterns = await self._detect_pattern_type(pattern_type, processed_data, request)
                recognized_patterns.extend(patterns)
            
            # Filter patterns by strength and confidence
            filtered_patterns = [
                p for p in recognized_patterns 
                if p.strength >= request.min_pattern_strength and p.confidence >= request.sensitivity
            ]
            
            # Generate insights and opportunities
            summary_insights = await self._generate_pattern_insights(filtered_patterns, processed_data)
            business_opportunities = await self._identify_business_opportunities(filtered_patterns)
            risk_indicators = await self._identify_risk_indicators(filtered_patterns)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = PatternRecognitionResult(
                request=request,
                patterns=filtered_patterns,
                summary_insights=summary_insights,
                business_opportunities=business_opportunities,
                risk_indicators=risk_indicators,
                execution_time_ms=execution_time
            )
            
            # Cache results
            self.pattern_cache[result.analysis_id] = result
            
            logger.info(f"Pattern recognition completed: {len(filtered_patterns)} patterns found in {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {str(e)}")
            raise
    
    async def detect_emerging_opportunities(self, data_sources: List[str], 
                                         lookback_days: int = 90) -> List[AnalyticsInsight]:
        """
        Detect emerging business opportunities through pattern analysis.
        
        Args:
            data_sources: List of data sources to analyze
            lookback_days: Number of days to look back for pattern analysis
            
        Returns:
            List of identified emerging opportunities
        """
        try:
            opportunities = []
            
            for source in data_sources:
                # Analyze recent patterns
                recent_patterns = await self._analyze_recent_patterns(source, lookback_days)
                
                # Identify growth patterns
                growth_opportunities = await self._identify_growth_patterns(recent_patterns)
                opportunities.extend(growth_opportunities)
                
                # Identify market shift patterns
                shift_opportunities = await self._identify_market_shifts(recent_patterns)
                opportunities.extend(shift_opportunities)
                
                # Identify efficiency opportunities
                efficiency_opportunities = await self._identify_efficiency_patterns(recent_patterns)
                opportunities.extend(efficiency_opportunities)
            
            # Rank and filter opportunities
            ranked_opportunities = self._rank_opportunities(opportunities)
            
            logger.info(f"Detected {len(ranked_opportunities)} emerging opportunities")
            
            return ranked_opportunities
            
        except Exception as e:
            logger.error(f"Error detecting emerging opportunities: {str(e)}")
            return []
    
    async def monitor_pattern_changes(self, baseline_patterns: List[RecognizedPattern], 
                                   current_data_source: str) -> Dict[str, Any]:
        """
        Monitor changes in patterns compared to a baseline.
        
        Args:
            baseline_patterns: Previously identified patterns as baseline
            current_data_source: Current data source to analyze
            
        Returns:
            Pattern change analysis results
        """
        try:
            # Analyze current patterns
            current_request = PatternRecognitionRequest(
                data_source=current_data_source,
                pattern_types=[PatternType.TREND, PatternType.ANOMALY, PatternType.CYCLE],
                sensitivity=0.7
            )
            
            current_result = await self.recognize_patterns(current_request)
            current_patterns = current_result.patterns
            
            # Compare patterns
            changes = {
                "new_patterns": [],
                "disappeared_patterns": [],
                "strengthened_patterns": [],
                "weakened_patterns": [],
                "change_summary": []
            }
            
            # Identify new patterns
            baseline_signatures = {self._get_pattern_signature(p): p for p in baseline_patterns}
            current_signatures = {self._get_pattern_signature(p): p for p in current_patterns}
            
            for signature, pattern in current_signatures.items():
                if signature not in baseline_signatures:
                    changes["new_patterns"].append(pattern)
                else:
                    # Compare strength changes
                    baseline_pattern = baseline_signatures[signature]
                    strength_change = pattern.strength - baseline_pattern.strength
                    
                    if strength_change > 0.1:
                        changes["strengthened_patterns"].append({
                            "pattern": pattern,
                            "strength_change": strength_change
                        })
                    elif strength_change < -0.1:
                        changes["weakened_patterns"].append({
                            "pattern": pattern,
                            "strength_change": strength_change
                        })
            
            # Identify disappeared patterns
            for signature, pattern in baseline_signatures.items():
                if signature not in current_signatures:
                    changes["disappeared_patterns"].append(pattern)
            
            # Generate change summary
            changes["change_summary"] = [
                f"Detected {len(changes['new_patterns'])} new patterns",
                f"Lost {len(changes['disappeared_patterns'])} previous patterns",
                f"{len(changes['strengthened_patterns'])} patterns strengthened",
                f"{len(changes['weakened_patterns'])} patterns weakened"
            ]
            
            logger.info(f"Pattern monitoring completed: {len(changes['new_patterns'])} new patterns detected")
            
            return changes
            
        except Exception as e:
            logger.error(f"Error monitoring pattern changes: {str(e)}")
            return {}
    
    async def _load_data_from_source(self, data_source: str) -> pd.DataFrame:
        """Load data from the specified source."""
        # Check cache first
        if data_source in self.data_cache:
            cache_time, data = self.data_cache[data_source]
            if (datetime.utcnow() - cache_time).seconds < 300:  # 5 minute cache
                return data
        
        # Generate sample data based on source type
        if data_source == "sales_data":
            data = self._generate_sales_data()
        elif data_source == "customer_behavior":
            data = self._generate_customer_behavior_data()
        elif data_source == "operational_metrics":
            data = self._generate_operational_data()
        elif data_source == "financial_metrics":
            data = self._generate_financial_data()
        elif data_source == "market_data":
            data = self._generate_market_data()
        else:
            data = self._generate_generic_time_series_data()
        
        # Cache the data
        self.data_cache[data_source] = (datetime.utcnow(), data)
        
        return data
    
    def _generate_sales_data(self) -> pd.DataFrame:
        """Generate sample sales data with various patterns."""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Base trend with seasonal patterns
        base_trend = np.linspace(1000, 1500, len(dates))
        seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Yearly cycle
        weekly = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly cycle
        
        # Add some anomalies
        anomalies = np.zeros(len(dates))
        anomaly_indices = np.random.choice(len(dates), size=20, replace=False)
        anomalies[anomaly_indices] = np.random.normal(0, 300, 20)
        
        # Random noise
        noise = np.random.normal(0, 50, len(dates))
        
        sales = base_trend + seasonal + weekly + anomalies + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative
        
        return pd.DataFrame({
            'date': dates,
            'sales_amount': sales,
            'customer_count': np.random.poisson(50, len(dates)),
            'avg_order_value': sales / np.maximum(np.random.poisson(50, len(dates)), 1),
            'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
            'product_category': np.random.choice(['A', 'B', 'C', 'D'], len(dates))
        })
    
    def _generate_customer_behavior_data(self) -> pd.DataFrame:
        """Generate sample customer behavior data."""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(123)
        
        # Website visits with growth trend
        base_visits = np.linspace(5000, 8000, len(dates))
        visits = base_visits + np.random.normal(0, 500, len(dates))
        
        # Conversion rate with some patterns
        base_conversion = 0.05 + 0.02 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        conversion_rate = np.maximum(0.01, base_conversion + np.random.normal(0, 0.005, len(dates)))
        
        return pd.DataFrame({
            'date': dates,
            'website_visits': np.maximum(0, visits),
            'page_views': visits * np.random.uniform(2, 5, len(dates)),
            'conversion_rate': conversion_rate,
            'bounce_rate': np.random.uniform(0.3, 0.7, len(dates)),
            'session_duration': np.random.exponential(180, len(dates)),  # seconds
            'channel': np.random.choice(['organic', 'paid', 'social', 'email'], len(dates))
        })
    
    def _generate_operational_data(self) -> pd.DataFrame:
        """Generate sample operational metrics data."""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='H')  # Hourly data
        np.random.seed(456)
        
        # System performance metrics
        cpu_usage = 30 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 5, len(dates))
        cpu_usage = np.clip(cpu_usage, 0, 100)
        
        memory_usage = 40 + 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 3, len(dates))
        memory_usage = np.clip(memory_usage, 0, 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'cpu_usage_percent': cpu_usage,
            'memory_usage_percent': memory_usage,
            'response_time_ms': np.random.exponential(100, len(dates)),
            'error_rate': np.random.exponential(0.01, len(dates)),
            'throughput_rps': np.random.poisson(1000, len(dates)),
            'system': np.random.choice(['web', 'api', 'database', 'cache'], len(dates))
        })
    
    def _generate_financial_data(self) -> pd.DataFrame:
        """Generate sample financial metrics data."""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')  # Monthly data
        np.random.seed(789)
        
        # Revenue with growth trend
        base_revenue = np.linspace(1000000, 1500000, len(dates))
        revenue = base_revenue * (1 + np.random.normal(0, 0.1, len(dates)))
        
        # Costs with some efficiency improvements
        base_costs = revenue * 0.7  # 70% cost ratio
        costs = base_costs * (1 + np.random.normal(0, 0.05, len(dates)))
        
        return pd.DataFrame({
            'date': dates,
            'revenue': np.maximum(0, revenue),
            'costs': np.maximum(0, costs),
            'profit': revenue - costs,
            'cash_flow': (revenue - costs) + np.random.normal(0, 50000, len(dates)),
            'accounts_receivable': revenue * np.random.uniform(0.1, 0.3, len(dates)),
            'department': np.random.choice(['sales', 'marketing', 'operations', 'r&d'], len(dates))
        })
    
    def _generate_market_data(self) -> pd.DataFrame:
        """Generate sample market data."""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(101)
        
        # Market indicators
        market_index = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
        
        return pd.DataFrame({
            'date': dates,
            'market_index': market_index,
            'competitor_price': 100 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
            'market_share': 0.15 + 0.05 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.01, len(dates)),
            'customer_sentiment': np.random.uniform(0.3, 0.8, len(dates)),
            'brand_mentions': np.random.poisson(100, len(dates)),
            'industry': np.random.choice(['tech', 'finance', 'retail', 'healthcare'], len(dates))
        })
    
    def _generate_generic_time_series_data(self) -> pd.DataFrame:
        """Generate generic time series data."""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(202)
        
        # Generic metric with trend and seasonality
        trend = np.linspace(100, 200, len(dates))
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.normal(0, 10, len(dates))
        
        metric = trend + seasonal + noise
        
        return pd.DataFrame({
            'date': dates,
            'metric_value': metric,
            'category': np.random.choice(['A', 'B', 'C'], len(dates)),
            'region': np.random.choice(['North', 'South'], len(dates))
        })
    
    async def _preprocess_data(self, data: pd.DataFrame, request: PatternRecognitionRequest) -> pd.DataFrame:
        """Preprocess data for pattern recognition."""
        processed_data = data.copy()
        
        # Apply time window filter if specified
        if request.time_window:
            start_date = request.time_window.get('start')
            end_date = request.time_window.get('end')
            
            date_column = None
            for col in ['date', 'timestamp', 'time']:
                if col in processed_data.columns:
                    date_column = col
                    break
            
            if date_column and start_date and end_date:
                processed_data = processed_data[
                    (processed_data[date_column] >= start_date) & 
                    (processed_data[date_column] <= end_date)
                ]
        
        # Apply context filters
        for filter_key, filter_value in request.context_filters.items():
            if filter_key in processed_data.columns:
                if isinstance(filter_value, list):
                    processed_data = processed_data[processed_data[filter_key].isin(filter_value)]
                else:
                    processed_data = processed_data[processed_data[filter_key] == filter_value]
        
        # Sort by date/time if available
        date_columns = ['date', 'timestamp', 'time']
        for col in date_columns:
            if col in processed_data.columns:
                processed_data = processed_data.sort_values(col)
                break
        
        return processed_data
    
    async def _detect_pattern_type(self, pattern_type: PatternType, data: pd.DataFrame, 
                                 request: PatternRecognitionRequest) -> List[RecognizedPattern]:
        """Detect patterns of a specific type."""
        patterns = []
        
        if pattern_type == PatternType.TREND:
            patterns = await self._detect_trends(data, request)
        elif pattern_type == PatternType.ANOMALY:
            patterns = await self._detect_anomalies(data, request)
        elif pattern_type == PatternType.CYCLE:
            patterns = await self._detect_cycles(data, request)
        elif pattern_type == PatternType.CLUSTER:
            patterns = await self._detect_clusters(data, request)
        elif pattern_type == PatternType.OUTLIER:
            patterns = await self._detect_outliers(data, request)
        elif pattern_type == PatternType.CORRELATION:
            patterns = await self._detect_correlations(data, request)
        
        return patterns
    
    async def _detect_trends(self, data: pd.DataFrame, request: PatternRecognitionRequest) -> List[RecognizedPattern]:
        """Detect trend patterns in the data."""
        patterns = []
        
        # Find numeric columns for trend analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(data[column].dropna()) < 10:  # Need minimum data points
                continue
            
            values = data[column].dropna().values
            x = np.arange(len(values))
            
            # Calculate linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine trend significance
            if abs(r_value) > 0.3 and p_value < 0.05:  # Significant trend
                trend_direction = "increasing" if slope > 0 else "decreasing"
                trend_strength = abs(r_value)
                
                # Calculate business impact
                if len(values) > 1:
                    total_change = (values[-1] - values[0]) / values[0] * 100
                    business_impact = f"{trend_direction.title()} trend with {total_change:.1f}% total change"
                else:
                    business_impact = f"{trend_direction.title()} trend detected"
                
                pattern = RecognizedPattern(
                    pattern_type=PatternType.TREND,
                    description=f"{trend_direction.title()} trend in {column}",
                    strength=trend_strength,
                    confidence=1 - p_value,
                    data_points=[{"column": column, "slope": slope, "r_squared": r_value**2}],
                    business_impact=business_impact,
                    recommended_actions=[
                        f"Monitor {column} trend continuation",
                        f"Investigate factors driving {trend_direction} trend",
                        f"Develop strategies to {'sustain' if slope > 0 else 'reverse'} the trend"
                    ],
                    metadata={
                        "slope": slope,
                        "r_value": r_value,
                        "p_value": p_value,
                        "trend_direction": trend_direction
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_anomalies(self, data: pd.DataFrame, request: PatternRecognitionRequest) -> List[RecognizedPattern]:
        """Detect anomaly patterns in the data."""
        patterns = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(data[column].dropna()) < 10:
                continue
            
            values = data[column].dropna().values
            
            # Statistical anomaly detection (Z-score method)
            z_scores = np.abs(stats.zscore(values))
            anomaly_threshold = 2.5
            anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
            
            if len(anomaly_indices) > 0:
                anomaly_values = values[anomaly_indices]
                anomaly_strength = np.mean(z_scores[anomaly_indices]) / 3.0  # Normalize to 0-1
                
                pattern = RecognizedPattern(
                    pattern_type=PatternType.ANOMALY,
                    description=f"Statistical anomalies detected in {column}",
                    strength=min(1.0, anomaly_strength),
                    confidence=0.8,
                    data_points=[{
                        "column": column,
                        "anomaly_count": len(anomaly_indices),
                        "anomaly_values": anomaly_values.tolist()[:10]  # Limit to first 10
                    }],
                    business_impact=f"Found {len(anomaly_indices)} anomalous values that may indicate unusual business events",
                    recommended_actions=[
                        f"Investigate causes of anomalies in {column}",
                        "Review data quality and collection processes",
                        "Determine if anomalies represent opportunities or risks"
                    ],
                    metadata={
                        "anomaly_count": len(anomaly_indices),
                        "threshold": anomaly_threshold,
                        "max_z_score": np.max(z_scores)
                    }
                )
                patterns.append(pattern)
        
        # Machine learning-based anomaly detection
        if len(numeric_columns) > 1:
            try:
                # Prepare data for ML anomaly detection
                ml_data = data[numeric_columns].dropna()
                if len(ml_data) > 20:  # Need sufficient data
                    scaled_data = self.scaler.fit_transform(ml_data)
                    
                    # Fit isolation forest
                    anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
                    anomaly_indices = np.where(anomaly_labels == -1)[0]
                    
                    if len(anomaly_indices) > 0:
                        anomaly_ratio = len(anomaly_indices) / len(ml_data)
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.ANOMALY,
                            description="Multivariate anomalies detected using machine learning",
                            strength=min(1.0, anomaly_ratio * 10),  # Scale anomaly ratio
                            confidence=0.85,
                            data_points=[{
                                "method": "isolation_forest",
                                "anomaly_count": len(anomaly_indices),
                                "anomaly_ratio": anomaly_ratio
                            }],
                            business_impact=f"ML analysis identified {len(anomaly_indices)} complex anomalous patterns across multiple metrics",
                            recommended_actions=[
                                "Investigate multivariate anomalies for root causes",
                                "Review business processes during anomalous periods",
                                "Consider implementing real-time anomaly monitoring"
                            ],
                            metadata={
                                "method": "isolation_forest",
                                "contamination": 0.1,
                                "features_used": list(numeric_columns)
                            }
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning(f"ML anomaly detection failed: {str(e)}")
        
        return patterns
    
    async def _detect_cycles(self, data: pd.DataFrame, request: PatternRecognitionRequest) -> List[RecognizedPattern]:
        """Detect cyclical patterns in the data."""
        patterns = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(data[column].dropna()) < 50:  # Need sufficient data for cycle detection
                continue
            
            values = data[column].dropna().values
            
            # Simple cycle detection using autocorrelation
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation (potential cycle periods)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr[1:], height=np.max(autocorr) * 0.3)
            
            if len(peaks) > 0:
                # Most prominent cycle
                main_peak = peaks[np.argmax(autocorr[peaks + 1])]
                cycle_period = main_peak + 1
                cycle_strength = autocorr[main_peak + 1] / autocorr[0]
                
                if cycle_strength > 0.3:  # Significant cycle
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.CYCLE,
                        description=f"Cyclical pattern detected in {column} with period {cycle_period}",
                        strength=cycle_strength,
                        confidence=0.75,
                        data_points=[{
                            "column": column,
                            "cycle_period": cycle_period,
                            "autocorrelation": cycle_strength
                        }],
                        business_impact=f"Regular {cycle_period}-period cycle suggests predictable business patterns",
                        recommended_actions=[
                            f"Plan business activities around {cycle_period}-period cycles",
                            "Develop forecasting models incorporating cyclical patterns",
                            "Optimize resource allocation based on cycle timing"
                        ],
                        metadata={
                            "cycle_period": cycle_period,
                            "autocorrelation_strength": cycle_strength,
                            "peaks_found": len(peaks)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_clusters(self, data: pd.DataFrame, request: PatternRecognitionRequest) -> List[RecognizedPattern]:
        """Detect cluster patterns in the data."""
        patterns = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) >= 2:  # Need at least 2 dimensions for clustering
            try:
                # Prepare data for clustering
                cluster_data = data[numeric_columns].dropna()
                
                if len(cluster_data) > 10:
                    scaled_data = self.scaler.fit_transform(cluster_data)
                    
                    # Try different clustering algorithms
                    # K-means clustering
                    for n_clusters in [2, 3, 4, 5]:
                        if len(cluster_data) > n_clusters * 3:  # Ensure sufficient data per cluster
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            cluster_labels = kmeans.fit_predict(scaled_data)
                            
                            # Calculate cluster quality (silhouette-like measure)
                            cluster_centers = kmeans.cluster_centers_
                            inertia = kmeans.inertia_
                            
                            # Simple cluster quality measure
                            cluster_quality = 1 / (1 + inertia / len(scaled_data))
                            
                            if cluster_quality > 0.3:  # Reasonable clustering
                                cluster_sizes = np.bincount(cluster_labels)
                                
                                pattern = RecognizedPattern(
                                    pattern_type=PatternType.CLUSTER,
                                    description=f"Data clusters into {n_clusters} distinct groups",
                                    strength=cluster_quality,
                                    confidence=0.7,
                                    data_points=[{
                                        "n_clusters": n_clusters,
                                        "cluster_sizes": cluster_sizes.tolist(),
                                        "inertia": inertia
                                    }],
                                    business_impact=f"Data naturally segments into {n_clusters} groups, suggesting distinct business categories",
                                    recommended_actions=[
                                        f"Develop targeted strategies for each of the {n_clusters} segments",
                                        "Analyze characteristics of each cluster",
                                        "Customize products/services for different segments"
                                    ],
                                    metadata={
                                        "algorithm": "kmeans",
                                        "n_clusters": n_clusters,
                                        "cluster_quality": cluster_quality,
                                        "features_used": list(numeric_columns)
                                    }
                                )
                                patterns.append(pattern)
                                break  # Use the first good clustering result
                    
                    # DBSCAN clustering for density-based clusters
                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    dbscan_labels = dbscan.fit_predict(scaled_data)
                    
                    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                    n_noise = list(dbscan_labels).count(-1)
                    
                    if n_clusters_dbscan > 1 and n_noise < len(scaled_data) * 0.5:  # Good clustering
                        cluster_quality = (len(scaled_data) - n_noise) / len(scaled_data)
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.CLUSTER,
                            description=f"Density-based clustering reveals {n_clusters_dbscan} natural groups",
                            strength=cluster_quality,
                            confidence=0.75,
                            data_points=[{
                                "n_clusters": n_clusters_dbscan,
                                "n_noise_points": n_noise,
                                "algorithm": "dbscan"
                            }],
                            business_impact=f"Natural density clusters suggest {n_clusters_dbscan} distinct business segments with clear boundaries",
                            recommended_actions=[
                                "Focus on core clusters while investigating outlier patterns",
                                "Develop cluster-specific business strategies",
                                "Monitor cluster evolution over time"
                            ],
                            metadata={
                                "algorithm": "dbscan",
                                "n_clusters": n_clusters_dbscan,
                                "noise_ratio": n_noise / len(scaled_data),
                                "features_used": list(numeric_columns)
                            }
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning(f"Clustering analysis failed: {str(e)}")
        
        return patterns
    
    async def _detect_outliers(self, data: pd.DataFrame, request: PatternRecognitionRequest) -> List[RecognizedPattern]:
        """Detect outlier patterns in the data."""
        patterns = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(data[column].dropna()) < 10:
                continue
            
            values = data[column].dropna().values
            
            # IQR method for outlier detection
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            if len(outliers) > 0:
                outlier_ratio = len(outliers) / len(values)
                
                if outlier_ratio > 0.01:  # At least 1% outliers
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.OUTLIER,
                        description=f"Outliers detected in {column} using IQR method",
                        strength=min(1.0, outlier_ratio * 10),  # Scale outlier ratio
                        confidence=0.8,
                        data_points=[{
                            "column": column,
                            "outlier_count": len(outliers),
                            "outlier_ratio": outlier_ratio,
                            "outlier_values": outliers.tolist()[:10]  # First 10 outliers
                        }],
                        business_impact=f"Found {len(outliers)} outlier values ({outlier_ratio:.1%}) that may represent exceptional cases",
                        recommended_actions=[
                            f"Investigate outlier cases in {column}",
                            "Determine if outliers represent errors or genuine exceptions",
                            "Consider separate handling for outlier cases"
                        ],
                        metadata={
                            "method": "iqr",
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "outlier_count": len(outliers)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_correlations(self, data: pd.DataFrame, request: PatternRecognitionRequest) -> List[RecognizedPattern]:
        """Detect correlation patterns between variables."""
        patterns = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) >= 2:
            # Calculate correlation matrix
            corr_matrix = data[numeric_columns].corr()
            
            # Find strong correlations (excluding diagonal)
            strong_correlations = []
            
            for i in range(len(numeric_columns)):
                for j in range(i + 1, len(numeric_columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.5:  # Strong correlation threshold
                        strong_correlations.append({
                            'var1': numeric_columns[i],
                            'var2': numeric_columns[j],
                            'correlation': corr_value
                        })
            
            if strong_correlations:
                # Group correlations by strength
                very_strong = [c for c in strong_correlations if abs(c['correlation']) > 0.8]
                strong = [c for c in strong_correlations if 0.5 < abs(c['correlation']) <= 0.8]
                
                if very_strong:
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.CORRELATION,
                        description=f"Very strong correlations found between {len(very_strong)} variable pairs",
                        strength=np.mean([abs(c['correlation']) for c in very_strong]),
                        confidence=0.9,
                        data_points=very_strong,
                        business_impact="Strong correlations indicate predictable relationships that can be leveraged for forecasting and optimization",
                        recommended_actions=[
                            "Leverage strong correlations for predictive modeling",
                            "Investigate causal relationships behind correlations",
                            "Use correlated variables for cross-validation and quality checks"
                        ],
                        metadata={
                            "correlation_type": "very_strong",
                            "threshold": 0.8,
                            "correlation_count": len(very_strong)
                        }
                    )
                    patterns.append(pattern)
                
                if strong:
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.CORRELATION,
                        description=f"Strong correlations found between {len(strong)} variable pairs",
                        strength=np.mean([abs(c['correlation']) for c in strong]),
                        confidence=0.8,
                        data_points=strong,
                        business_impact="Moderate correlations suggest relationships that could be strengthened or leveraged",
                        recommended_actions=[
                            "Explore ways to strengthen moderate correlations",
                            "Monitor correlation stability over time",
                            "Consider correlation patterns in business planning"
                        ],
                        metadata={
                            "correlation_type": "strong",
                            "threshold": 0.5,
                            "correlation_count": len(strong)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _generate_pattern_insights(self, patterns: List[RecognizedPattern], 
                                       data: pd.DataFrame) -> List[str]:
        """Generate summary insights from recognized patterns."""
        insights = []
        
        if not patterns:
            insights.append("No significant patterns detected in the current dataset")
            return insights
        
        # Pattern type distribution
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            pattern_counts[pattern.pattern_type] += 1
        
        insights.append(f"Detected {len(patterns)} patterns across {len(pattern_counts)} different types")
        
        # Strongest patterns
        strongest_patterns = sorted(patterns, key=lambda p: p.strength, reverse=True)[:3]
        if strongest_patterns:
            insights.append(f"Strongest pattern: {strongest_patterns[0].description} (strength: {strongest_patterns[0].strength:.2f})")
        
        # Pattern confidence analysis
        avg_confidence = np.mean([p.confidence for p in patterns])
        insights.append(f"Average pattern confidence: {avg_confidence:.2f}")
        
        # Business impact summary
        high_impact_patterns = [p for p in patterns if p.strength > 0.7]
        if high_impact_patterns:
            insights.append(f"{len(high_impact_patterns)} high-impact patterns identified requiring immediate attention")
        
        return insights
    
    async def _identify_business_opportunities(self, patterns: List[RecognizedPattern]) -> List[str]:
        """Identify business opportunities from recognized patterns."""
        opportunities = []
        
        for pattern in patterns:
            if pattern.pattern_type == PatternType.TREND and pattern.strength > 0.6:
                if "increasing" in pattern.description.lower():
                    opportunities.append(f"Capitalize on positive trend in {pattern.description}")
                else:
                    opportunities.append(f"Address declining trend in {pattern.description}")
            
            elif pattern.pattern_type == PatternType.CLUSTER and pattern.strength > 0.5:
                opportunities.append("Develop targeted strategies for identified customer/market segments")
            
            elif pattern.pattern_type == PatternType.CORRELATION and pattern.strength > 0.7:
                opportunities.append("Leverage strong correlations for predictive analytics and optimization")
            
            elif pattern.pattern_type == PatternType.CYCLE and pattern.strength > 0.6:
                opportunities.append("Optimize resource allocation and planning based on cyclical patterns")
        
        # Remove duplicates
        opportunities = list(set(opportunities))
        
        return opportunities[:10]  # Top 10 opportunities
    
    async def _identify_risk_indicators(self, patterns: List[RecognizedPattern]) -> List[str]:
        """Identify risk indicators from recognized patterns."""
        risks = []
        
        for pattern in patterns:
            if pattern.pattern_type == PatternType.ANOMALY and pattern.strength > 0.7:
                risks.append(f"High anomaly activity detected: {pattern.description}")
            
            elif pattern.pattern_type == PatternType.TREND and pattern.strength > 0.6:
                if "decreasing" in pattern.description.lower():
                    risks.append(f"Negative trend risk: {pattern.description}")
            
            elif pattern.pattern_type == PatternType.OUTLIER and pattern.strength > 0.8:
                risks.append(f"Significant outliers detected that may indicate data quality or process issues")
        
        # Remove duplicates
        risks = list(set(risks))
        
        return risks[:10]  # Top 10 risks
    
    def _get_pattern_signature(self, pattern: RecognizedPattern) -> str:
        """Generate a signature for pattern comparison."""
        # Create a signature based on pattern type and key characteristics
        signature_parts = [
            pattern.pattern_type.value,
            pattern.description[:50],  # First 50 chars of description
            str(round(pattern.strength, 1))
        ]
        
        return "|".join(signature_parts)
    
    async def _analyze_recent_patterns(self, data_source: str, lookback_days: int) -> List[RecognizedPattern]:
        """Analyze patterns in recent data."""
        # Load recent data
        data = await self._load_data_from_source(data_source)
        
        # Filter to recent data
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        date_column = None
        for col in ['date', 'timestamp', 'time']:
            if col in data.columns:
                date_column = col
                break
        
        if date_column:
            data = data[data[date_column] >= cutoff_date]
        
        # Analyze patterns
        request = PatternRecognitionRequest(
            data_source=data_source,
            pattern_types=[PatternType.TREND, PatternType.ANOMALY, PatternType.CORRELATION],
            sensitivity=0.6
        )
        
        result = await self.recognize_patterns(request)
        return result.patterns
    
    async def _identify_growth_patterns(self, patterns: List[RecognizedPattern]) -> List[AnalyticsInsight]:
        """Identify growth opportunities from patterns."""
        opportunities = []
        
        growth_trends = [p for p in patterns 
                        if p.pattern_type == PatternType.TREND and 
                        "increasing" in p.description.lower() and 
                        p.strength > 0.6]
        
        if growth_trends:
            opportunity = AnalyticsInsight(
                title="Growth Opportunity Detected",
                description=f"Identified {len(growth_trends)} positive growth trends in recent data",
                insight_type="growth_opportunity",
                confidence=0.8,
                business_impact="Positive trends indicate areas with growth potential that should be prioritized",
                supporting_data={"growth_trends": len(growth_trends)},
                recommended_actions=[
                    "Increase investment in areas showing positive trends",
                    "Analyze factors driving growth for replication",
                    "Develop strategies to accelerate positive trends"
                ],
                priority=8
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_market_shifts(self, patterns: List[RecognizedPattern]) -> List[AnalyticsInsight]:
        """Identify market shift opportunities from patterns."""
        opportunities = []
        
        # Look for anomalies that might indicate market shifts
        anomaly_patterns = [p for p in patterns 
                          if p.pattern_type == PatternType.ANOMALY and 
                          p.strength > 0.7]
        
        if len(anomaly_patterns) > 2:  # Multiple anomalies suggest shifts
            opportunity = AnalyticsInsight(
                title="Potential Market Shift Detected",
                description=f"Multiple anomaly patterns ({len(anomaly_patterns)}) suggest possible market or operational shifts",
                insight_type="market_shift",
                confidence=0.7,
                business_impact="Market shifts can represent both opportunities and threats requiring strategic response",
                supporting_data={"anomaly_count": len(anomaly_patterns)},
                recommended_actions=[
                    "Investigate root causes of anomalous patterns",
                    "Assess competitive landscape for market changes",
                    "Develop adaptive strategies for market shifts"
                ],
                priority=7
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_efficiency_patterns(self, patterns: List[RecognizedPattern]) -> List[AnalyticsInsight]:
        """Identify efficiency opportunities from patterns."""
        opportunities = []
        
        # Look for correlation patterns that suggest optimization opportunities
        correlation_patterns = [p for p in patterns 
                              if p.pattern_type == PatternType.CORRELATION and 
                              p.strength > 0.6]
        
        if correlation_patterns:
            opportunity = AnalyticsInsight(
                title="Process Optimization Opportunity",
                description=f"Strong correlations ({len(correlation_patterns)}) identified that could be leveraged for efficiency improvements",
                insight_type="efficiency_opportunity",
                confidence=0.75,
                business_impact="Correlation patterns can be used to optimize processes and improve operational efficiency",
                supporting_data={"correlation_patterns": len(correlation_patterns)},
                recommended_actions=[
                    "Develop optimization models based on identified correlations",
                    "Implement process improvements leveraging correlation insights",
                    "Monitor correlation stability for sustained optimization"
                ],
                priority=6
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    def _rank_opportunities(self, opportunities: List[AnalyticsInsight]) -> List[AnalyticsInsight]:
        """Rank opportunities by priority and confidence."""
        return sorted(opportunities, key=lambda x: (x.priority, x.confidence), reverse=True)[:10]