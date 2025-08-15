"""
Tests for drift monitoring engine.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from ai_data_readiness.engines.drift_monitor import DriftMonitor
from ai_data_readiness.models.drift_models import (
    DriftThresholds, DriftType, AlertSeverity, DriftReport
)


class TestDriftMonitor:
    """Test cases for DriftMonitor class."""
    
    @pytest.fixture
    def drift_monitor(self):
        """Create DriftMonitor instance for testing."""
        return DriftMonitor()
    
    @pytest.fixture
    def custom_thresholds(self):
        """Create custom thresholds for testing."""
        return DriftThresholds(
            low_threshold=0.05,
            medium_threshold=0.2,
            high_threshold=0.4,
            critical_threshold=0.6,
            statistical_significance=0.01,
            minimum_samples=50
        )
    
    @pytest.fixture
    def reference_data(self):
        """Create reference dataset for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_feature': np.random.normal(0, 1, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
            'binary_feature': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })
    
    @pytest.fixture
    def current_data_no_drift(self, reference_data):
        """Create current dataset with no drift."""
        np.random.seed(43)
        return pd.DataFrame({
            'numeric_feature': np.random.normal(0, 1, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
            'binary_feature': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })
    
    @pytest.fixture
    def current_data_with_drift(self, reference_data):
        """Create current dataset with significant drift."""
        np.random.seed(44)
        return pd.DataFrame({
            'numeric_feature': np.random.normal(2, 1.5, 1000),  # Mean and std shift
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.2, 0.3, 0.5]),  # Distribution shift
            'binary_feature': np.random.choice([0, 1], 1000, p=[0.3, 0.7])  # Probability shift
        })
    
    def test_drift_monitor_initialization(self):
        """Test DriftMonitor initialization."""
        monitor = DriftMonitor()
        assert monitor.thresholds is not None
        assert isinstance(monitor.thresholds, DriftThresholds)
        
        # Test with custom thresholds
        custom_thresholds = DriftThresholds(low_threshold=0.05)
        monitor_custom = DriftMonitor(custom_thresholds)
        assert monitor_custom.thresholds.low_threshold == 0.05
    
    def test_monitor_drift_no_drift(self, drift_monitor, reference_data, current_data_no_drift):
        """Test drift monitoring with no significant drift."""
        report = drift_monitor.monitor_drift(
            dataset_id="test_dataset",
            reference_dataset_id="reference_dataset",
            current_data=current_data_no_drift,
            reference_data=reference_data
        )
        
        assert isinstance(report, DriftReport)
        assert report.dataset_id == "test_dataset"
        assert report.reference_dataset_id == "reference_dataset"
        assert report.drift_score < 0.3  # Should be low drift
        assert len(report.feature_drift_scores) == 3
        assert report.metrics is not None
        assert len(report.alerts) == 0  # No significant drift alerts
    
    def test_monitor_drift_with_drift(self, drift_monitor, reference_data, current_data_with_drift):
        """Test drift monitoring with significant drift."""
        report = drift_monitor.monitor_drift(
            dataset_id="test_dataset",
            reference_dataset_id="reference_dataset",
            current_data=current_data_with_drift,
            reference_data=reference_data
        )
        
        assert isinstance(report, DriftReport)
        assert report.drift_score > 0.3  # Should detect significant drift
        assert len(report.feature_drift_scores) == 3
        assert len(report.alerts) > 0  # Should generate alerts
        assert len(report.recommendations) > 0  # Should generate recommendations
        
        # Check that numeric feature has high drift
        assert report.feature_drift_scores['numeric_feature'] > 0.2
    
    def test_calculate_psi(self, drift_monitor):
        """Test Population Stability Index calculation."""
        # No drift case
        reference = np.random.normal(0, 1, 1000)
        current_no_drift = np.random.normal(0, 1, 1000)
        psi_no_drift = drift_monitor._calculate_psi(current_no_drift, reference)
        assert psi_no_drift < 0.2
        
        # High drift case
        current_drift = np.random.normal(2, 1.5, 1000)
        psi_drift = drift_monitor._calculate_psi(current_drift, reference)
        assert psi_drift > psi_no_drift
    
    def test_calculate_categorical_drift(self, drift_monitor):
        """Test categorical drift calculation."""
        # No drift case
        reference = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]))
        current_no_drift = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]))
        drift_no_drift = drift_monitor._calculate_categorical_drift(current_no_drift, reference)
        
        # High drift case
        current_drift = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.2, 0.3, 0.5]))
        drift_with_drift = drift_monitor._calculate_categorical_drift(current_drift, reference)
        
        assert drift_with_drift > drift_no_drift
    
    def test_statistical_tests(self, drift_monitor, reference_data, current_data_with_drift):
        """Test statistical tests for drift detection."""
        tests = drift_monitor._perform_statistical_tests(current_data_with_drift, reference_data)
        
        assert len(tests) == 3
        assert 'numeric_feature' in tests
        assert 'categorical_feature' in tests
        assert 'binary_feature' in tests
        
        # Numeric feature should show significant drift
        numeric_test = tests['numeric_feature']
        assert numeric_test.test_name == "Kolmogorov-Smirnov"
        assert numeric_test.is_significant  # Should detect drift
        
        # Categorical feature should show significant drift
        categorical_test = tests['categorical_feature']
        assert categorical_test.test_name == "Chi-square"
        assert categorical_test.is_significant  # Should detect drift
    
    def test_alert_generation(self, drift_monitor):
        """Test drift alert generation."""
        feature_scores = {
            'feature1': 0.1,  # Low drift
            'feature2': 0.4,  # High drift
            'feature3': 0.8   # Critical drift
        }
        overall_score = 0.5
        
        alerts = drift_monitor._generate_alerts("test_dataset", feature_scores, overall_score)
        
        assert len(alerts) > 0
        
        # Check overall drift alert
        overall_alerts = [a for a in alerts if a.drift_type == DriftType.COVARIATE_SHIFT]
        assert len(overall_alerts) == 1
        assert overall_alerts[0].severity == AlertSeverity.HIGH
        
        # Check feature-level alerts
        feature_alerts = [a for a in alerts if a.drift_type == DriftType.FEATURE_DRIFT]
        assert len(feature_alerts) == 2  # feature2 and feature3 should generate alerts
    
    def test_recommendation_generation(self, drift_monitor):
        """Test drift recommendation generation."""
        # Critical drift case
        recommendations = drift_monitor._generate_recommendations(
            overall_drift_score=0.8,
            feature_drift_scores={'feature1': 0.8},
            alerts=[]
        )
        
        assert len(recommendations) > 0
        retrain_recs = [r for r in recommendations if r.type == "retrain"]
        assert len(retrain_recs) == 1
        assert retrain_recs[0].priority == "high"
        
        # Medium drift case
        recommendations_medium = drift_monitor._generate_recommendations(
            overall_drift_score=0.3,
            feature_drift_scores={'feature1': 0.3},
            alerts=[]
        )
        
        monitor_recs = [r for r in recommendations_medium if r.type == "monitor"]
        assert len(monitor_recs) == 1
        assert monitor_recs[0].priority == "low"
    
    def test_severity_determination(self, drift_monitor):
        """Test severity level determination."""
        assert drift_monitor._determine_severity(0.05) == AlertSeverity.LOW
        assert drift_monitor._determine_severity(0.35) == AlertSeverity.MEDIUM
        assert drift_monitor._determine_severity(0.55) == AlertSeverity.HIGH
        assert drift_monitor._determine_severity(0.75) == AlertSeverity.CRITICAL
    
    def test_drift_metrics_calculation(self, drift_monitor, reference_data, current_data_with_drift):
        """Test comprehensive drift metrics calculation."""
        metrics = drift_monitor._calculate_drift_metrics(current_data_with_drift, reference_data)
        
        assert metrics.overall_drift_score > 0
        assert len(metrics.feature_drift_scores) == 3
        assert len(metrics.distribution_distances) == 3
        assert len(metrics.statistical_tests) == 3
        assert metrics.drift_velocity >= 0
        assert metrics.drift_magnitude >= 0
        assert len(metrics.confidence_interval) == 2
        assert metrics.confidence_interval[0] <= metrics.confidence_interval[1]
    
    def test_custom_thresholds(self, custom_thresholds):
        """Test drift monitor with custom thresholds."""
        monitor = DriftMonitor(custom_thresholds)
        
        assert monitor.thresholds.low_threshold == 0.05
        assert monitor.thresholds.medium_threshold == 0.2
        assert monitor.thresholds.high_threshold == 0.4
        assert monitor.thresholds.critical_threshold == 0.6
        
        # Test severity determination with custom thresholds
        assert monitor._determine_severity(0.1) == AlertSeverity.LOW
        assert monitor._determine_severity(0.25) == AlertSeverity.MEDIUM
        assert monitor._determine_severity(0.45) == AlertSeverity.HIGH
        assert monitor._determine_severity(0.65) == AlertSeverity.CRITICAL
    
    def test_set_drift_thresholds(self, drift_monitor, custom_thresholds):
        """Test setting custom drift thresholds."""
        drift_monitor.set_drift_thresholds("test_dataset", custom_thresholds)
        assert drift_monitor.thresholds.low_threshold == 0.05
    
    def test_get_drift_alerts(self, drift_monitor):
        """Test getting drift alerts."""
        alerts = drift_monitor.get_drift_alerts("test_dataset")
        assert isinstance(alerts, list)
        
        # Test with severity filter
        alerts_filtered = drift_monitor.get_drift_alerts("test_dataset", AlertSeverity.HIGH)
        assert isinstance(alerts_filtered, list)
    
    def test_validation_errors(self, drift_monitor):
        """Test dataset validation errors."""
        # Empty datasets
        with pytest.raises(ValueError, match="Datasets cannot be empty"):
            drift_monitor.monitor_drift(
                "test", "ref", pd.DataFrame(), pd.DataFrame({'col': [1, 2, 3]})
            )
        
        # Mismatched columns
        with pytest.raises(ValueError, match="Column names must match"):
            drift_monitor.monitor_drift(
                "test", "ref", 
                pd.DataFrame({'col1': [1, 2, 3]}),
                pd.DataFrame({'col2': [1, 2, 3]})
            )
        
        # Insufficient samples
        small_data = pd.DataFrame({'col': [1, 2]})
        with pytest.raises(ValueError, match="insufficient samples"):
            drift_monitor.monitor_drift("test", "ref", small_data, small_data)
    
    def test_edge_cases(self, drift_monitor):
        """Test edge cases and error handling."""
        # Data with NaN values
        reference_with_nan = pd.DataFrame({
            'feature': [1, 2, np.nan, 4, 5] * 200
        })
        current_with_nan = pd.DataFrame({
            'feature': [1.1, 2.1, np.nan, 4.1, 5.1] * 200
        })
        
        report = drift_monitor.monitor_drift(
            "test", "ref", current_with_nan, reference_with_nan
        )
        assert isinstance(report, DriftReport)
        assert report.drift_score >= 0
        
        # Single category data
        single_cat_ref = pd.DataFrame({
            'cat_feature': ['A'] * 1000
        })
        single_cat_cur = pd.DataFrame({
            'cat_feature': ['A'] * 1000
        })
        
        report_single = drift_monitor.monitor_drift(
            "test", "ref", single_cat_cur, single_cat_ref
        )
        assert isinstance(report_single, DriftReport)
    
    def test_js_divergence_calculation(self, drift_monitor):
        """Test Jensen-Shannon divergence calculation."""
        # Identical distributions
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 1, 1000)
        js_div_same = drift_monitor._calculate_js_divergence(data1, data2)
        
        # Different distributions
        data3 = np.random.normal(2, 1, 1000)
        js_div_diff = drift_monitor._calculate_js_divergence(data1, data3)
        
        assert js_div_diff > js_div_same
        assert 0 <= js_div_same <= 1
        assert 0 <= js_div_diff <= 1
    
    def test_chi_square_distance(self, drift_monitor):
        """Test chi-square distance calculation."""
        # Similar distributions
        cat1 = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]))
        cat2 = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]))
        dist_same = drift_monitor._calculate_chi_square_distance(cat1, cat2)
        
        # Different distributions
        cat3 = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.2, 0.3, 0.5]))
        dist_diff = drift_monitor._calculate_chi_square_distance(cat1, cat3)
        
        assert dist_diff > dist_same
        assert dist_same >= 0
        assert dist_diff >= 0
    
    def test_confidence_interval_calculation(self, drift_monitor):
        """Test confidence interval calculation."""
        feature_scores = {'f1': 0.1, 'f2': 0.2, 'f3': 0.3, 'f4': 0.4}
        ci = drift_monitor._calculate_confidence_interval(feature_scores)
        
        assert len(ci) == 2
        assert ci[0] <= ci[1]
        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1
        
        # Empty scores
        ci_empty = drift_monitor._calculate_confidence_interval({})
        assert ci_empty == (0.0, 0.0)
    
    def test_drift_velocity_calculation(self, drift_monitor):
        """Test drift velocity calculation."""
        # Uniform scores (low velocity)
        uniform_scores = {'f1': 0.2, 'f2': 0.2, 'f3': 0.2}
        velocity_low = drift_monitor._calculate_drift_velocity(uniform_scores)
        
        # Varied scores (high velocity)
        varied_scores = {'f1': 0.1, 'f2': 0.5, 'f3': 0.9}
        velocity_high = drift_monitor._calculate_drift_velocity(varied_scores)
        
        assert velocity_high > velocity_low
        assert velocity_low >= 0
        assert velocity_high >= 0
        
        # Empty scores
        velocity_empty = drift_monitor._calculate_drift_velocity({})
        assert velocity_empty == 0.0


if __name__ == "__main__":
    pytest.main([__file__])