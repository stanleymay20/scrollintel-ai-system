"""
Drift monitoring engine for detecting data distribution changes.

This module implements statistical drift detection algorithms to monitor
data distribution changes over time and assess their impact on AI model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

from ..models.drift_models import (
    DriftReport, DriftAlert, DriftMetrics, DriftThresholds,
    DriftRecommendation, StatisticalTest, DriftType, AlertSeverity
)


class DriftMonitor:
    """
    Advanced drift detection engine with statistical algorithms.
    
    Implements multiple drift detection methods including:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Jensen-Shannon divergence
    - Chi-square test for categorical variables
    - Wasserstein distance
    """
    
    def __init__(self, thresholds: Optional[DriftThresholds] = None):
        """Initialize drift monitor with configurable thresholds."""
        self.thresholds = thresholds or DriftThresholds()
        self.logger = logging.getLogger(__name__)
        self._reference_stats = {}
        self._feature_types = {}
        
    def monitor_drift(self, dataset_id: str, reference_dataset_id: str, 
                     current_data: pd.DataFrame, reference_data: pd.DataFrame) -> DriftReport:
        """
        Monitor drift between current and reference datasets.
        
        Args:
            dataset_id: Current dataset identifier
            reference_dataset_id: Reference dataset identifier
            current_data: Current dataset to analyze
            reference_data: Reference dataset for comparison
            
        Returns:
            DriftReport: Comprehensive drift analysis report
        """
        try:
            self.logger.info(f"Starting drift monitoring for dataset {dataset_id}")
            
            # Validate inputs
            self._validate_datasets(current_data, reference_data)
            
            # Calculate drift metrics
            drift_metrics = self._calculate_drift_metrics(current_data, reference_data)
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(current_data, reference_data)
            
            # Calculate feature-level drift scores
            feature_drift_scores = self._calculate_feature_drift_scores(
                current_data, reference_data
            )
            
            # Calculate overall drift score
            overall_drift_score = self._calculate_overall_drift_score(feature_drift_scores)
            
            # Generate alerts
            alerts = self._generate_alerts(
                dataset_id, feature_drift_scores, overall_drift_score
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_drift_score, feature_drift_scores, alerts
            )
            
            # Create drift report
            report = DriftReport(
                dataset_id=dataset_id,
                reference_dataset_id=reference_dataset_id,
                drift_score=overall_drift_score,
                feature_drift_scores=feature_drift_scores,
                statistical_tests=statistical_tests,
                alerts=alerts,
                recommendations=recommendations,
                metrics=drift_metrics
            )
            
            self.logger.info(f"Drift monitoring completed. Overall score: {overall_drift_score:.3f}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error in drift monitoring: {str(e)}")
            raise
    
    def calculate_drift_metrics(self, current_data: pd.DataFrame, 
                              reference_data: pd.DataFrame) -> DriftMetrics:
        """Calculate comprehensive drift metrics."""
        return self._calculate_drift_metrics(current_data, reference_data)
    
    def set_drift_thresholds(self, dataset_id: str, thresholds: DriftThresholds) -> None:
        """Set custom drift thresholds for a dataset."""
        self.thresholds = thresholds
        self.logger.info(f"Updated drift thresholds for dataset {dataset_id}")
    
    def get_drift_alerts(self, dataset_id: str, severity_filter: Optional[AlertSeverity] = None) -> List[DriftAlert]:
        """Get drift alerts for a dataset with optional severity filtering."""
        # This would typically query a database or cache
        # For now, return empty list as placeholder
        return []
    
    def _validate_datasets(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> None:
        """Validate input datasets."""
        if current_data.empty or reference_data.empty:
            raise ValueError("Datasets cannot be empty")
        
        if len(current_data.columns) != len(reference_data.columns):
            raise ValueError("Datasets must have the same number of columns")
        
        if not all(col in reference_data.columns for col in current_data.columns):
            raise ValueError("Column names must match between datasets")
        
        if len(current_data) < self.thresholds.minimum_samples:
            raise ValueError(f"Current data has insufficient samples: {len(current_data)} < {self.thresholds.minimum_samples}")
        
        if len(reference_data) < self.thresholds.minimum_samples:
            raise ValueError(f"Reference data has insufficient samples: {len(reference_data)} < {self.thresholds.minimum_samples}")
    
    def _calculate_drift_metrics(self, current_data: pd.DataFrame, 
                               reference_data: pd.DataFrame) -> DriftMetrics:
        """Calculate comprehensive drift metrics."""
        feature_drift_scores = self._calculate_feature_drift_scores(current_data, reference_data)
        overall_drift_score = self._calculate_overall_drift_score(feature_drift_scores)
        
        # Calculate distribution distances
        distribution_distances = {}
        for column in current_data.columns:
            if self._is_numeric_column(current_data[column]):
                # Jensen-Shannon divergence for numeric features
                distance = self._calculate_js_divergence(
                    current_data[column].values, reference_data[column].values
                )
            else:
                # Chi-square distance for categorical features
                distance = self._calculate_chi_square_distance(
                    current_data[column], reference_data[column]
                )
            distribution_distances[column] = distance
        
        # Calculate drift velocity and magnitude
        drift_velocity = self._calculate_drift_velocity(feature_drift_scores)
        drift_magnitude = max(feature_drift_scores.values()) if feature_drift_scores else 0.0
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(feature_drift_scores)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(current_data, reference_data)
        
        return DriftMetrics(
            overall_drift_score=overall_drift_score,
            feature_drift_scores=feature_drift_scores,
            distribution_distances=distribution_distances,
            statistical_tests=statistical_tests,
            drift_velocity=drift_velocity,
            drift_magnitude=drift_magnitude,
            confidence_interval=confidence_interval
        )
    
    def _calculate_feature_drift_scores(self, current_data: pd.DataFrame, 
                                      reference_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate drift scores for individual features."""
        drift_scores = {}
        
        for column in current_data.columns:
            try:
                if self._is_numeric_column(current_data[column]):
                    # Use PSI for numeric features
                    score = self._calculate_psi(
                        current_data[column].values, reference_data[column].values
                    )
                else:
                    # Use chi-square test for categorical features
                    score = self._calculate_categorical_drift(
                        current_data[column], reference_data[column]
                    )
                
                drift_scores[column] = score
                
            except Exception as e:
                self.logger.warning(f"Error calculating drift for column {column}: {str(e)}")
                drift_scores[column] = 0.0
        
        return drift_scores
    
    def _calculate_overall_drift_score(self, feature_drift_scores: Dict[str, float]) -> float:
        """Calculate overall drift score from feature-level scores."""
        if not feature_drift_scores:
            return 0.0
        
        # Use weighted average with higher weight for features with more drift
        scores = list(feature_drift_scores.values())
        weights = [score + 0.1 for score in scores]  # Add small constant to avoid zero weights
        
        overall_score = np.average(scores, weights=weights)
        return min(overall_score, 1.0)  # Cap at 1.0
    
    def _calculate_psi(self, current: np.ndarray, reference: np.ndarray, 
                      bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Remove NaN values
            current = current[~np.isnan(current)]
            reference = reference[~np.isnan(reference)]
            
            if len(current) == 0 or len(reference) == 0:
                return 0.0
            
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Calculate distributions
            current_dist, _ = np.histogram(current, bins=bin_edges)
            reference_dist, _ = np.histogram(reference, bins=bin_edges)
            
            # Convert to proportions
            current_prop = current_dist / len(current)
            reference_prop = reference_dist / len(reference)
            
            # Add small constant to avoid division by zero
            epsilon = 1e-10
            current_prop = current_prop + epsilon
            reference_prop = reference_prop + epsilon
            
            # Calculate PSI
            psi = np.sum((current_prop - reference_prop) * np.log(current_prop / reference_prop))
            
            return abs(psi)
            
        except Exception as e:
            self.logger.warning(f"Error calculating PSI: {str(e)}")
            return 0.0
    
    def _calculate_categorical_drift(self, current: pd.Series, reference: pd.Series) -> float:
        """Calculate drift score for categorical features using chi-square test."""
        try:
            # Get value counts
            current_counts = current.value_counts()
            reference_counts = reference.value_counts()
            
            # Align categories
            all_categories = set(current_counts.index) | set(reference_counts.index)
            
            current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
            reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
            
            # Perform chi-square test
            if sum(current_aligned) == 0 or sum(reference_aligned) == 0:
                return 0.0
            
            chi2, p_value = stats.chisquare(current_aligned, reference_aligned)
            
            # Convert chi-square statistic to drift score (0-1 range)
            drift_score = min(chi2 / (len(all_categories) * 100), 1.0)
            
            return drift_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating categorical drift: {str(e)}")
            return 0.0
    
    def _calculate_js_divergence(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        try:
            # Remove NaN values
            current = current[~np.isnan(current)]
            reference = reference[~np.isnan(reference)]
            
            if len(current) == 0 or len(reference) == 0:
                return 0.0
            
            # Create histograms
            min_val = min(current.min(), reference.min())
            max_val = max(current.max(), reference.max())
            bins = np.linspace(min_val, max_val, 50)
            
            current_hist, _ = np.histogram(current, bins=bins, density=True)
            reference_hist, _ = np.histogram(reference, bins=bins, density=True)
            
            # Normalize to probabilities
            current_prob = current_hist / current_hist.sum()
            reference_prob = reference_hist / reference_hist.sum()
            
            # Calculate Jensen-Shannon divergence
            js_div = jensenshannon(current_prob, reference_prob)
            
            return js_div
            
        except Exception as e:
            self.logger.warning(f"Error calculating JS divergence: {str(e)}")
            return 0.0
    
    def _calculate_chi_square_distance(self, current: pd.Series, reference: pd.Series) -> float:
        """Calculate chi-square distance for categorical variables."""
        try:
            current_counts = current.value_counts(normalize=True)
            reference_counts = reference.value_counts(normalize=True)
            
            # Align categories
            all_categories = set(current_counts.index) | set(reference_counts.index)
            
            distance = 0.0
            for category in all_categories:
                p1 = current_counts.get(category, 0)
                p2 = reference_counts.get(category, 0)
                
                if p1 + p2 > 0:
                    distance += ((p1 - p2) ** 2) / (p1 + p2)
            
            return distance / 2  # Normalize
            
        except Exception as e:
            self.logger.warning(f"Error calculating chi-square distance: {str(e)}")
            return 0.0
    
    def _perform_statistical_tests(self, current_data: pd.DataFrame, 
                                 reference_data: pd.DataFrame) -> Dict[str, StatisticalTest]:
        """Perform statistical tests for drift detection."""
        tests = {}
        
        for column in current_data.columns:
            try:
                if self._is_numeric_column(current_data[column]):
                    # Kolmogorov-Smirnov test for numeric features
                    statistic, p_value = stats.ks_2samp(
                        current_data[column].dropna(), 
                        reference_data[column].dropna()
                    )
                    
                    test = StatisticalTest(
                        test_name="Kolmogorov-Smirnov",
                        statistic=statistic,
                        p_value=p_value,
                        threshold=self.thresholds.statistical_significance,
                        is_significant=p_value < self.thresholds.statistical_significance,
                        interpretation=self._interpret_ks_test(statistic, p_value)
                    )
                else:
                    # Chi-square test for categorical features
                    current_counts = current_data[column].value_counts()
                    reference_counts = reference_data[column].value_counts()
                    
                    # Align categories
                    all_categories = set(current_counts.index) | set(reference_counts.index)
                    current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                    reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                    
                    chi2, p_value = stats.chisquare(current_aligned, reference_aligned)
                    
                    test = StatisticalTest(
                        test_name="Chi-square",
                        statistic=chi2,
                        p_value=p_value,
                        threshold=self.thresholds.statistical_significance,
                        is_significant=p_value < self.thresholds.statistical_significance,
                        interpretation=self._interpret_chi_square_test(chi2, p_value)
                    )
                
                tests[column] = test
                
            except Exception as e:
                self.logger.warning(f"Error performing statistical test for {column}: {str(e)}")
        
        return tests
    
    def _generate_alerts(self, dataset_id: str, feature_drift_scores: Dict[str, float], 
                        overall_drift_score: float) -> List[DriftAlert]:
        """Generate drift alerts based on scores and thresholds."""
        alerts = []
        
        # Generate overall drift alert
        severity = self._determine_severity(overall_drift_score)
        if severity != AlertSeverity.LOW:
            alert = DriftAlert(
                id=f"drift_{dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                dataset_id=dataset_id,
                drift_type=DriftType.COVARIATE_SHIFT,
                severity=severity,
                message=f"Overall drift detected with score {overall_drift_score:.3f}",
                affected_features=list(feature_drift_scores.keys()),
                drift_score=overall_drift_score,
                threshold=self._get_threshold_for_severity(severity)
            )
            alerts.append(alert)
        
        # Generate feature-level alerts
        for feature, score in feature_drift_scores.items():
            feature_severity = self._determine_severity(score)
            if feature_severity != AlertSeverity.LOW:
                alert = DriftAlert(
                    id=f"drift_{dataset_id}_{feature}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    dataset_id=dataset_id,
                    drift_type=DriftType.FEATURE_DRIFT,
                    severity=feature_severity,
                    message=f"Feature drift detected in '{feature}' with score {score:.3f}",
                    affected_features=[feature],
                    drift_score=score,
                    threshold=self._get_threshold_for_severity(feature_severity)
                )
                alerts.append(alert)
        
        return alerts
    
    def _generate_recommendations(self, overall_drift_score: float, 
                                feature_drift_scores: Dict[str, float],
                                alerts: List[DriftAlert]) -> List[DriftRecommendation]:
        """Generate recommendations for handling detected drift."""
        recommendations = []
        
        if overall_drift_score >= self.thresholds.critical_threshold:
            recommendations.append(DriftRecommendation(
                type="retrain",
                priority="high",
                description="Critical drift detected. Immediate model retraining recommended.",
                action_items=[
                    "Stop using current model for predictions",
                    "Collect fresh training data",
                    "Retrain model with recent data",
                    "Validate model performance before deployment"
                ],
                estimated_effort="2-3 days",
                expected_impact="High - prevents model degradation"
            ))
        
        elif overall_drift_score >= self.thresholds.high_threshold:
            recommendations.append(DriftRecommendation(
                type="investigate",
                priority="medium",
                description="Significant drift detected. Investigation and potential retraining needed.",
                action_items=[
                    "Analyze root cause of drift",
                    "Assess model performance impact",
                    "Consider incremental model updates",
                    "Monitor drift trends closely"
                ],
                estimated_effort="1-2 days",
                expected_impact="Medium - maintains model accuracy"
            ))
        
        elif overall_drift_score >= self.thresholds.medium_threshold:
            recommendations.append(DriftRecommendation(
                type="monitor",
                priority="low",
                description="Moderate drift detected. Enhanced monitoring recommended.",
                action_items=[
                    "Increase monitoring frequency",
                    "Set up automated alerts",
                    "Track drift trends over time",
                    "Prepare for potential retraining"
                ],
                estimated_effort="0.5 days",
                expected_impact="Low - early warning system"
            ))
        
        # Feature-specific recommendations
        high_drift_features = [
            feature for feature, score in feature_drift_scores.items()
            if score >= self.thresholds.high_threshold
        ]
        
        if high_drift_features:
            recommendations.append(DriftRecommendation(
                type="feature_analysis",
                priority="medium",
                description=f"High drift detected in features: {', '.join(high_drift_features)}",
                action_items=[
                    "Analyze feature importance changes",
                    "Check data collection processes",
                    "Validate feature engineering pipeline",
                    "Consider feature selection updates"
                ],
                estimated_effort="1 day",
                expected_impact="Medium - targeted improvement"
            ))
        
        return recommendations
    
    def _determine_severity(self, drift_score: float) -> AlertSeverity:
        """Determine alert severity based on drift score."""
        if drift_score >= self.thresholds.critical_threshold:
            return AlertSeverity.CRITICAL
        elif drift_score >= self.thresholds.high_threshold:
            return AlertSeverity.HIGH
        elif drift_score >= self.thresholds.medium_threshold:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _get_threshold_for_severity(self, severity: AlertSeverity) -> float:
        """Get threshold value for given severity level."""
        threshold_map = {
            AlertSeverity.LOW: self.thresholds.low_threshold,
            AlertSeverity.MEDIUM: self.thresholds.medium_threshold,
            AlertSeverity.HIGH: self.thresholds.high_threshold,
            AlertSeverity.CRITICAL: self.thresholds.critical_threshold
        }
        return threshold_map.get(severity, 0.0)
    
    def _calculate_drift_velocity(self, feature_drift_scores: Dict[str, float]) -> float:
        """Calculate drift velocity (rate of change)."""
        if not feature_drift_scores:
            return 0.0
        
        # This is a simplified calculation
        # In practice, you'd track scores over time
        scores = list(feature_drift_scores.values())
        return np.std(scores)  # Use standard deviation as proxy for velocity
    
    def _calculate_confidence_interval(self, feature_drift_scores: Dict[str, float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for drift scores."""
        if not feature_drift_scores:
            return (0.0, 0.0)
        
        scores = list(feature_drift_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * std_score / np.sqrt(len(scores))
        
        return (max(0.0, mean_score - margin), min(1.0, mean_score + margin))
    
    def _is_numeric_column(self, column: pd.Series) -> bool:
        """Check if column contains numeric data."""
        return pd.api.types.is_numeric_dtype(column)
    
    def _interpret_ks_test(self, statistic: float, p_value: float) -> str:
        """Interpret Kolmogorov-Smirnov test results."""
        if p_value < 0.001:
            return f"Strong evidence of distribution change (D={statistic:.3f}, p<0.001)"
        elif p_value < 0.01:
            return f"Moderate evidence of distribution change (D={statistic:.3f}, p={p_value:.3f})"
        elif p_value < 0.05:
            return f"Weak evidence of distribution change (D={statistic:.3f}, p={p_value:.3f})"
        else:
            return f"No significant distribution change detected (D={statistic:.3f}, p={p_value:.3f})"
    
    def _interpret_chi_square_test(self, chi2: float, p_value: float) -> str:
        """Interpret chi-square test results."""
        if p_value < 0.001:
            return f"Strong evidence of category distribution change (χ²={chi2:.3f}, p<0.001)"
        elif p_value < 0.01:
            return f"Moderate evidence of category distribution change (χ²={chi2:.3f}, p={p_value:.3f})"
        elif p_value < 0.05:
            return f"Weak evidence of category distribution change (χ²={chi2:.3f}, p={p_value:.3f})"
        else:
            return f"No significant category distribution change detected (χ²={chi2:.3f}, p={p_value:.3f})"
    
    def calculate_drift(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate drift between current and reference data for streaming processing.
        
        This method provides a simplified drift calculation suitable for
        real-time streaming data processing.
        
        Args:
            current_data: Current data to compare
            reference_data: Reference data for comparison
            
        Returns:
            Dictionary containing drift metrics and overall score
        """
        try:
            if current_data.empty or reference_data.empty:
                return {
                    'overall_drift_score': 0.0,
                    'feature_drift_scores': {},
                    'drift_detected': False,
                    'record_count_current': len(current_data),
                    'record_count_reference': len(reference_data)
                }
            
            # Align columns between datasets
            common_columns = list(set(current_data.columns) & set(reference_data.columns))
            if not common_columns:
                return {
                    'overall_drift_score': 1.0,  # Maximum drift if no common columns
                    'feature_drift_scores': {},
                    'drift_detected': True,
                    'error': 'No common columns between datasets'
                }
            
            current_aligned = current_data[common_columns]
            reference_aligned = reference_data[common_columns]
            
            # Calculate drift for each feature
            feature_drift_scores = {}
            
            for column in common_columns:
                try:
                    current_col = current_aligned[column].dropna()
                    reference_col = reference_aligned[column].dropna()
                    
                    if len(current_col) == 0 or len(reference_col) == 0:
                        feature_drift_scores[column] = 0.0
                        continue
                    
                    if self._is_numeric_column(current_col):
                        # Use KS test for numeric columns
                        ks_stat, p_value = stats.ks_2samp(current_col, reference_col)
                        drift_score = ks_stat  # KS statistic as drift score
                    else:
                        # Use chi-square test for categorical columns
                        drift_score = self._calculate_categorical_drift(current_col, reference_col)
                    
                    feature_drift_scores[column] = min(1.0, max(0.0, drift_score))
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating drift for column {column}: {str(e)}")
                    feature_drift_scores[column] = 0.0
            
            # Calculate overall drift score
            if feature_drift_scores:
                overall_drift_score = np.mean(list(feature_drift_scores.values()))
            else:
                overall_drift_score = 0.0
            
            # Determine if drift is detected based on threshold
            drift_detected = overall_drift_score > self.thresholds.medium_threshold
            
            return {
                'overall_drift_score': overall_drift_score,
                'feature_drift_scores': feature_drift_scores,
                'drift_detected': drift_detected,
                'record_count_current': len(current_data),
                'record_count_reference': len(reference_data),
                'common_columns': len(common_columns),
                'threshold_used': self.thresholds.medium_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error in streaming drift calculation: {str(e)}")
            return {
                'overall_drift_score': 0.0,
                'feature_drift_scores': {},
                'drift_detected': False,
                'record_count_current': len(current_data) if not current_data.empty else 0,
                'record_count_reference': len(reference_data) if not reference_data.empty else 0,
                'error': str(e)
            }
    
    def _calculate_categorical_drift(self, current_col: pd.Series, reference_col: pd.Series) -> float:
        """Calculate drift for categorical columns using distribution comparison."""
        try:
            # Get value counts for both columns
            current_counts = current_col.value_counts(normalize=True)
            reference_counts = reference_col.value_counts(normalize=True)
            
            # Get all unique values
            all_values = set(current_counts.index) | set(reference_counts.index)
            
            # Create aligned probability distributions
            current_probs = []
            reference_probs = []
            
            for value in all_values:
                current_probs.append(current_counts.get(value, 0.0))
                reference_probs.append(reference_counts.get(value, 0.0))
            
            # Calculate Jensen-Shannon divergence
            current_probs = np.array(current_probs)
            reference_probs = np.array(reference_probs)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            current_probs = current_probs + epsilon
            reference_probs = reference_probs + epsilon
            
            # Normalize
            current_probs = current_probs / current_probs.sum()
            reference_probs = reference_probs / reference_probs.sum()
            
            # Calculate JS divergence
            js_divergence = jensenshannon(current_probs, reference_probs)
            
            return js_divergence
            
        except Exception as e:
            self.logger.warning(f"Error calculating categorical drift: {str(e)}")
            return 0.0