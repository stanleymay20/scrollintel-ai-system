"""
Advanced Statistical Analytics Engine
Provides comprehensive statistical analysis and ML insights for reports
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class StatisticalResult:
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    interpretation: str
    confidence_level: float
    effect_size: Optional[float] = None

@dataclass
class AnomalyDetectionResult:
    anomalies: List[int]
    anomaly_scores: List[float]
    threshold: float
    method: str
    total_anomalies: int
    anomaly_percentage: float

@dataclass
class ClusterAnalysisResult:
    cluster_labels: List[int]
    cluster_centers: List[List[float]]
    n_clusters: int
    silhouette_score: float
    inertia: Optional[float]
    method: str

@dataclass
class TrendAnalysisResult:
    trend_direction: str
    trend_strength: float
    seasonal_component: Optional[List[float]]
    residuals: List[float]
    forecast: List[float]
    confidence_intervals: List[Tuple[float, float]]

@dataclass
class CorrelationAnalysisResult:
    correlation_matrix: Dict[str, Dict[str, float]]
    significant_correlations: List[Dict[str, Any]]
    correlation_method: str
    p_values: Optional[Dict[str, Dict[str, float]]]

class AdvancedStatisticalAnalytics:
    """Advanced statistical analysis engine with ML insights"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.analysis_cache = {}
    
    def comprehensive_analysis(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on dataset"""
        results = {
            'descriptive_statistics': self.descriptive_statistics(data),
            'correlation_analysis': self.correlation_analysis(data),
            'distribution_analysis': self.distribution_analysis(data),
            'outlier_detection': self.detect_outliers(data),
            'normality_tests': self.test_normality(data),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add clustering if enough numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            results['cluster_analysis'] = self.cluster_analysis(data[numeric_cols])
        
        # Add trend analysis if target column specified
        if target_column and target_column in data.columns:
            results['trend_analysis'] = self.trend_analysis(data, target_column)
            results['feature_importance'] = self.feature_importance_analysis(data, target_column)
        
        return results
    
    def descriptive_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No numeric columns found'}
        
        stats_dict = {}
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            
            if len(col_data) == 0:
                continue
            
            stats_dict[column] = {
                'count': len(col_data),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'mode': float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                'std': float(col_data.std()),
                'variance': float(col_data.var()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'range': float(col_data.max() - col_data.min()),
                'q1': float(col_data.quantile(0.25)),
                'q3': float(col_data.quantile(0.75)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'coefficient_of_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else None
            }
        
        return {
            'column_statistics': stats_dict,
            'dataset_summary': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'numeric_columns': len(numeric_data.columns),
                'categorical_columns': len(data.select_dtypes(include=['object']).columns),
                'missing_values': data.isnull().sum().to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum()
            }
        }
    
    def correlation_analysis(self, data: pd.DataFrame, method: str = 'pearson') -> CorrelationAnalysisResult:
        """Perform correlation analysis with significance testing"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return CorrelationAnalysisResult(
                correlation_matrix={},
                significant_correlations=[],
                correlation_method=method,
                p_values=None
            )
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = numeric_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = numeric_data.corr(method='spearman')
        else:
            corr_matrix = numeric_data.corr(method='kendall')
        
        # Calculate p-values for correlations
        p_values = {}
        for col1 in numeric_data.columns:
            p_values[col1] = {}
            for col2 in numeric_data.columns:
                if col1 != col2:
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(numeric_data[col1].dropna(), numeric_data[col2].dropna())
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(numeric_data[col1].dropna(), numeric_data[col2].dropna())
                    else:
                        _, p_val = stats.kendalltau(numeric_data[col1].dropna(), numeric_data[col2].dropna())
                    p_values[col1][col2] = p_val
                else:
                    p_values[col1][col2] = 0.0
        
        # Find significant correlations
        significant_correlations = []
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 < col2:  # Avoid duplicates
                    corr_val = corr_matrix.loc[col1, col2]
                    p_val = p_values[col1][col2]
                    
                    if abs(corr_val) > 0.3 and p_val < 0.05:  # Significant correlation
                        significant_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_val),
                            'p_value': float(p_val),
                            'strength': self._interpret_correlation_strength(abs(corr_val)),
                            'direction': 'positive' if corr_val > 0 else 'negative'
                        })
        
        return CorrelationAnalysisResult(
            correlation_matrix=corr_matrix.to_dict(),
            significant_correlations=significant_correlations,
            correlation_method=method,
            p_values=p_values
        )
    
    def distribution_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No numeric columns found'}
        
        distribution_results = {}
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            
            if len(col_data) < 10:  # Need sufficient data
                continue
            
            # Test for various distributions
            distributions = {
                'normal': stats.normaltest(col_data),
                'uniform': stats.kstest(col_data, 'uniform'),
                'exponential': stats.kstest(col_data, 'expon')
            }
            
            # Find best fitting distribution
            best_fit = min(distributions.items(), key=lambda x: x[1][1])  # Lowest p-value
            
            distribution_results[column] = {
                'distribution_tests': {
                    name: {'statistic': float(stat), 'p_value': float(p_val)}
                    for name, (stat, p_val) in distributions.items()
                },
                'best_fit_distribution': best_fit[0],
                'best_fit_p_value': float(best_fit[1][1]),
                'histogram_bins': self._calculate_optimal_bins(col_data),
                'distribution_parameters': self._estimate_distribution_parameters(col_data)
            }
        
        return distribution_results
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'isolation_forest') -> AnomalyDetectionResult:
        """Detect outliers using various methods"""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if numeric_data.empty or len(numeric_data) < 10:
            return AnomalyDetectionResult(
                anomalies=[],
                anomaly_scores=[],
                threshold=0.0,
                method=method,
                total_anomalies=0,
                anomaly_percentage=0.0
            )
        
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = detector.fit_predict(numeric_data)
            anomaly_scores = detector.decision_function(numeric_data)
            anomalies = np.where(outlier_labels == -1)[0].tolist()
            threshold = np.percentile(anomaly_scores, 10)
            
        elif method == 'statistical':
            # Z-score method
            z_scores = np.abs(stats.zscore(numeric_data))
            threshold = 3.0
            anomalies = np.where((z_scores > threshold).any(axis=1))[0].tolist()
            anomaly_scores = np.max(z_scores, axis=1).tolist()
            
        elif method == 'iqr':
            # Interquartile range method
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).any(axis=1)
            anomalies = np.where(outlier_mask)[0].tolist()
            anomaly_scores = np.sum((numeric_data < lower_bound) | (numeric_data > upper_bound), axis=1).tolist()
            threshold = 1.0
        
        return AnomalyDetectionResult(
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            threshold=threshold,
            method=method,
            total_anomalies=len(anomalies),
            anomaly_percentage=len(anomalies) / len(numeric_data) * 100
        )
    
    def cluster_analysis(self, data: pd.DataFrame, method: str = 'kmeans', n_clusters: Optional[int] = None) -> ClusterAnalysisResult:
        """Perform cluster analysis"""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if numeric_data.empty or len(numeric_data) < 10:
            return ClusterAnalysisResult(
                cluster_labels=[],
                cluster_centers=[],
                n_clusters=0,
                silhouette_score=0.0,
                inertia=None,
                method=method
            )
        
        # Standardize data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        if method == 'kmeans':
            if n_clusters is None:
                # Find optimal number of clusters using elbow method
                n_clusters = self._find_optimal_clusters(scaled_data)
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(scaled_data)
            cluster_centers = clusterer.cluster_centers_.tolist()
            inertia = clusterer.inertia_
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(scaled_data)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            cluster_centers = []
            inertia = None
            
            # Calculate cluster centers for DBSCAN
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:  # Ignore noise points
                    cluster_points = scaled_data[cluster_labels == cluster_id]
                    center = np.mean(cluster_points, axis=0).tolist()
                    cluster_centers.append(center)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(scaled_data, cluster_labels)
        else:
            sil_score = 0.0
        
        return ClusterAnalysisResult(
            cluster_labels=cluster_labels.tolist(),
            cluster_centers=cluster_centers,
            n_clusters=n_clusters,
            silhouette_score=float(sil_score),
            inertia=inertia,
            method=method
        )
    
    def trend_analysis(self, data: pd.DataFrame, target_column: str, time_column: Optional[str] = None) -> TrendAnalysisResult:
        """Perform trend analysis on time series data"""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Prepare data
        if time_column and time_column in data.columns:
            time_series = data.set_index(time_column)[target_column].dropna()
        else:
            time_series = data[target_column].dropna()
        
        if len(time_series) < 10:
            return TrendAnalysisResult(
                trend_direction='insufficient_data',
                trend_strength=0.0,
                seasonal_component=None,
                residuals=[],
                forecast=[],
                confidence_intervals=[]
            )
        
        # Calculate trend using linear regression
        x = np.arange(len(time_series)).reshape(-1, 1)
        y = time_series.values
        
        reg = LinearRegression().fit(x, y)
        trend_slope = reg.coef_[0]
        trend_strength = abs(reg.score(x, y))  # R-squared
        
        # Determine trend direction
        if abs(trend_slope) < 0.01:
            trend_direction = 'stable'
        elif trend_slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate residuals
        predicted = reg.predict(x)
        residuals = (y - predicted).tolist()
        
        # Simple forecast (extend trend)
        forecast_steps = min(10, len(time_series) // 4)
        future_x = np.arange(len(time_series), len(time_series) + forecast_steps).reshape(-1, 1)
        forecast = reg.predict(future_x).tolist()
        
        # Calculate confidence intervals (simplified)
        residual_std = np.std(residuals)
        confidence_intervals = [
            (pred - 1.96 * residual_std, pred + 1.96 * residual_std)
            for pred in forecast
        ]
        
        return TrendAnalysisResult(
            trend_direction=trend_direction,
            trend_strength=float(trend_strength),
            seasonal_component=None,  # Would need more sophisticated analysis
            residuals=residuals,
            forecast=forecast,
            confidence_intervals=confidence_intervals
        )
    
    def feature_importance_analysis(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze feature importance using Random Forest"""
        if target_column not in data.columns:
            return {'error': f"Target column '{target_column}' not found"}
        
        # Prepare features and target
        features = data.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
        target = data[target_column]
        
        if features.empty or len(features.columns) < 1:
            return {'error': 'No numeric features found'}
        
        # Handle missing values
        features = features.fillna(features.mean())
        target = target.fillna(target.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_names = features.columns.tolist()
        
        # Create importance ranking
        importance_ranking = [
            {
                'feature': feature_names[i],
                'importance': float(importances[i]),
                'rank': i + 1
            }
            for i in np.argsort(importances)[::-1]
        ]
        
        # Model performance
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        return {
            'feature_importance_ranking': importance_ranking,
            'model_performance': {
                'train_r2': float(train_score),
                'test_r2': float(test_score),
                'overfitting_indicator': float(train_score - test_score)
            },
            'top_features': importance_ranking[:5]  # Top 5 features
        }
    
    def test_normality(self, data: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Test normality of numeric columns"""
        numeric_data = data.select_dtypes(include=[np.number])
        results = {}
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            
            if len(col_data) < 8:  # Minimum sample size for normality tests
                continue
            
            # Shapiro-Wilk test (for smaller samples)
            if len(col_data) <= 5000:
                stat, p_val = stats.shapiro(col_data)
                test_name = 'Shapiro-Wilk'
            else:
                # D'Agostino's normality test (for larger samples)
                stat, p_val = stats.normaltest(col_data)
                test_name = "D'Agostino"
            
            # Interpretation
            if p_val > 0.05:
                interpretation = "Data appears to be normally distributed (fail to reject H0)"
            else:
                interpretation = "Data does not appear to be normally distributed (reject H0)"
            
            results[column] = StatisticalResult(
                test_name=test_name,
                statistic=float(stat),
                p_value=float(p_val),
                critical_value=None,
                interpretation=interpretation,
                confidence_level=0.95
            )
        
        return results
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation < 0.3:
            return 'weak'
        elif correlation < 0.7:
            return 'moderate'
        else:
            return 'strong'
    
    def _calculate_optimal_bins(self, data: np.ndarray) -> int:
        """Calculate optimal number of bins for histogram"""
        # Sturges' rule
        return int(np.ceil(np.log2(len(data)) + 1))
    
    def _estimate_distribution_parameters(self, data: np.ndarray) -> Dict[str, float]:
        """Estimate parameters for common distributions"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }
    
    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        k_range = range(2, min(max_clusters + 1, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        if len(inertias) < 2:
            return 2
        
        # Calculate rate of change
        rates = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
        
        # Find the point where rate of change starts to level off
        optimal_k = 2
        for i in range(1, len(rates)):
            if rates[i] < rates[i-1] * 0.5:  # Significant decrease in rate
                optimal_k = i + 2
                break
        
        return min(optimal_k, max_clusters)
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate natural language insights from analysis results"""
        insights = []
        
        # Descriptive statistics insights
        if 'descriptive_statistics' in analysis_results:
            stats = analysis_results['descriptive_statistics']
            if 'column_statistics' in stats:
                for col, col_stats in stats['column_statistics'].items():
                    if col_stats.get('coefficient_of_variation', 0) > 1:
                        insights.append(f"{col} shows high variability (CV > 1)")
                    
                    if abs(col_stats.get('skewness', 0)) > 1:
                        skew_direction = "right" if col_stats['skewness'] > 0 else "left"
                        insights.append(f"{col} is significantly skewed to the {skew_direction}")
        
        # Correlation insights
        if 'correlation_analysis' in analysis_results:
            corr = analysis_results['correlation_analysis']
            if corr.get('significant_correlations'):
                strong_corrs = [c for c in corr['significant_correlations'] if c['strength'] == 'strong']
                if strong_corrs:
                    insights.append(f"Found {len(strong_corrs)} strong correlations between variables")
        
        # Outlier insights
        if 'outlier_detection' in analysis_results:
            outliers = analysis_results['outlier_detection']
            if outliers.get('anomaly_percentage', 0) > 10:
                insights.append(f"High percentage of outliers detected ({outliers['anomaly_percentage']:.1f}%)")
        
        # Cluster insights
        if 'cluster_analysis' in analysis_results:
            clusters = analysis_results['cluster_analysis']
            if clusters.get('silhouette_score', 0) > 0.5:
                insights.append(f"Data shows good clustering structure with {clusters['n_clusters']} distinct groups")
        
        # Trend insights
        if 'trend_analysis' in analysis_results:
            trend = analysis_results['trend_analysis']
            if trend.get('trend_strength', 0) > 0.7:
                direction = trend.get('trend_direction', 'unknown')
                insights.append(f"Strong {direction} trend detected in the data")
        
        return insights