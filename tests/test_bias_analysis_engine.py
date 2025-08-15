"""Tests for the Bias Analysis Engine."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from ai_data_readiness.engines.bias_analysis_engine import (
    BiasAnalysisEngine, FairnessMetrics, ProtectedAttribute
)
from ai_data_readiness.models.base_models import (
    BiasReport, FairnessViolation, MitigationStrategy, BiasType
)
from ai_data_readiness.core.exceptions import BiasDetectionError, InsufficientDataError


class TestBiasAnalysisEngine:
    """Test cases for BiasAnalysisEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a BiasAnalysisEngine instance for testing."""
        return BiasAnalysisEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create biased dataset
        data = {
            'gender': np.random.choice(['male', 'female'], n_samples, p=[0.7, 0.3]),
            'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'education': np.random.choice(['high_school', 'college', 'graduate'], n_samples, p=[0.4, 0.4, 0.2]),
            'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
        
        # Introduce bias - males more likely to have positive outcomes
        male_mask = data['gender'] == 'male'
        data['target'][male_mask] = np.random.choice([0, 1], sum(male_mask), p=[0.4, 0.6])
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def balanced_data(self):
        """Create balanced dataset for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'gender': np.random.choice(['male', 'female'], n_samples, p=[0.5, 0.5]),
            'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        }
        
        return pd.DataFrame(data)
    
    def test_engine_initialization(self):
        """Test engine initialization with default and custom thresholds."""
        # Test default initialization
        engine = BiasAnalysisEngine()
        assert engine.fairness_thresholds['demographic_parity'] == 0.1
        assert engine.fairness_thresholds['disparate_impact'] == 0.8
        
        # Test custom thresholds
        custom_thresholds = {'demographic_parity': 0.05, 'disparate_impact': 0.9}
        engine = BiasAnalysisEngine(custom_thresholds)
        assert engine.fairness_thresholds['demographic_parity'] == 0.05
        assert engine.fairness_thresholds['disparate_impact'] == 0.9
    
    def test_detect_bias_basic(self, engine, sample_data):
        """Test basic bias detection functionality."""
        protected_attrs = ['gender', 'race']
        
        report = engine.detect_bias(
            dataset_id="test_dataset",
            data=sample_data,
            protected_attributes=protected_attrs,
            target_column='target'
        )
        
        assert isinstance(report, BiasReport)
        assert report.dataset_id == "test_dataset"
        assert report.protected_attributes == protected_attrs
        assert len(report.bias_metrics) == 2
        assert 'gender' in report.bias_metrics
        assert 'race' in report.bias_metrics
        assert isinstance(report.fairness_violations, list)
        assert isinstance(report.mitigation_strategies, list)
    
    def test_detect_bias_empty_data(self, engine):
        """Test bias detection with empty dataset."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(BiasDetectionError):
            engine.detect_bias("test", empty_data, ['gender'])
    
    def test_detect_bias_missing_attributes(self, engine, sample_data):
        """Test bias detection with missing protected attributes."""
        with pytest.raises(BiasDetectionError):
            engine.detect_bias(
                "test", 
                sample_data, 
                ['nonexistent_column']
            )
    
    def test_identify_protected_attributes(self, engine, sample_data):
        """Test automatic identification of protected attributes."""
        # Remove explicit protected attributes and test identification
        test_data = sample_data.copy()
        test_data = test_data.rename(columns={'gender': 'sex', 'race': 'ethnicity'})
        
        identified = engine._identify_protected_attributes(test_data)
        
        # Should identify renamed columns
        assert 'sex' in identified or 'ethnicity' in identified
    
    def test_analyze_protected_attribute(self, engine, sample_data):
        """Test protected attribute analysis."""
        attr_info = engine._analyze_protected_attribute(sample_data, 'gender')
        
        assert isinstance(attr_info, ProtectedAttribute)
        assert attr_info.name == 'gender'
        assert attr_info.is_binary == True
        assert len(attr_info.values) == 2
        assert 'male' in attr_info.values
        assert 'female' in attr_info.values
    
    def test_calculate_fairness_metrics(self, engine, sample_data):
        """Test fairness metrics calculation."""
        metrics = engine._calculate_fairness_metrics(
            sample_data, 'gender', 'target'
        )
        
        assert isinstance(metrics, FairnessMetrics)
        assert 0 <= metrics.demographic_parity <= 1
        assert 0 <= metrics.disparate_impact <= 1
        assert metrics.equalized_odds >= 0
        assert metrics.statistical_parity >= 0
        assert metrics.individual_fairness >= 0
    
    def test_demographic_parity_calculation(self, engine, sample_data):
        """Test demographic parity calculation."""
        # Test with biased data
        dp_biased = engine._calculate_demographic_parity(sample_data, 'gender')
        assert dp_biased > 0  # Should detect imbalance
        
        # Test with balanced data
        balanced_data = pd.DataFrame({
            'gender': ['male'] * 500 + ['female'] * 500
        })
        dp_balanced = engine._calculate_demographic_parity(balanced_data, 'gender')
        assert dp_balanced == 0  # Should be perfectly balanced
    
    def test_disparate_impact_calculation(self, engine, sample_data):
        """Test disparate impact calculation."""
        # Test with target variable
        di_with_target = engine._calculate_disparate_impact(
            sample_data, 'gender', 'target'
        )
        assert 0 <= di_with_target <= 1
        
        # Test without target variable
        di_without_target = engine._calculate_disparate_impact(
            sample_data, 'gender', None
        )
        assert 0 <= di_without_target <= 1
    
    def test_check_fairness_violations(self, engine, sample_data):
        """Test fairness violation detection."""
        metrics = FairnessMetrics(
            demographic_parity=0.2,  # Above threshold
            equalized_odds=0.15,    # Above threshold
            statistical_parity=0.05, # Below threshold
            individual_fairness=0.05, # Below threshold
            disparate_impact=0.6     # Below threshold
        )
        
        violations = engine._check_fairness_violations('gender', metrics)
        
        assert len(violations) >= 2  # Should detect multiple violations
        violation_types = [v.bias_type for v in violations]
        assert BiasType.DEMOGRAPHIC_PARITY in violation_types
        assert BiasType.EQUALIZED_ODDS in violation_types
    
    def test_generate_mitigation_strategies(self, engine):
        """Test mitigation strategy generation."""
        violations = [
            FairnessViolation(
                bias_type=BiasType.DEMOGRAPHIC_PARITY,
                protected_attribute='gender',
                severity='high',
                description='Test violation',
                metric_value=0.3,
                threshold=0.1,
                affected_groups=['female']
            ),
            FairnessViolation(
                bias_type=BiasType.EQUALIZED_ODDS,
                protected_attribute='race',
                severity='medium',
                description='Test violation',
                metric_value=0.2,
                threshold=0.1,
                affected_groups=['minority']
            )
        ]
        
        strategies = engine._generate_mitigation_strategies(violations)
        
        assert len(strategies) >= 2  # Should generate multiple strategies
        strategy_types = [s.strategy_type for s in strategies]
        assert 'data_balancing' in strategy_types
        assert 'algorithmic_fairness' in strategy_types
    
    def test_calculate_fairness_metrics_multiple_attributes(self, engine, sample_data):
        """Test fairness metrics calculation for multiple attributes."""
        protected_attrs = ['gender', 'race']
        
        metrics_dict = engine.calculate_fairness_metrics(
            "test_dataset",
            sample_data,
            protected_attrs,
            'target'
        )
        
        assert len(metrics_dict) == 2
        assert 'gender' in metrics_dict
        assert 'race' in metrics_dict
        assert isinstance(metrics_dict['gender'], FairnessMetrics)
        assert isinstance(metrics_dict['race'], FairnessMetrics)
    
    def test_validate_fairness_pass(self, engine, balanced_data):
        """Test fairness validation with data that passes constraints."""
        constraints = {
            'demographic_parity': 0.1,
            'disparate_impact': 0.8
        }
        
        result = engine.validate_fairness(
            balanced_data,
            ['gender'],
            constraints
        )
        
        assert result == True
    
    def test_validate_fairness_fail(self, engine, sample_data):
        """Test fairness validation with data that fails constraints."""
        strict_constraints = {
            'demographic_parity': 0.01,  # Very strict
            'disparate_impact': 0.99     # Very strict
        }
        
        result = engine.validate_fairness(
            sample_data,
            ['gender'],
            strict_constraints
        )
        
        assert result == False
    
    def test_fairness_metrics_to_dict(self):
        """Test FairnessMetrics to_dict conversion."""
        metrics = FairnessMetrics(
            demographic_parity=0.1,
            equalized_odds=0.2,
            statistical_parity=0.15,
            individual_fairness=0.05,
            disparate_impact=0.8
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['demographic_parity'] == 0.1
        assert metrics_dict['equalized_odds'] == 0.2
        assert metrics_dict['statistical_parity'] == 0.15
        assert metrics_dict['individual_fairness'] == 0.05
        assert metrics_dict['disparate_impact'] == 0.8
    
    def test_bias_detection_with_categorical_target(self, engine):
        """Test bias detection with categorical target variable."""
        data = pd.DataFrame({
            'gender': ['male', 'female'] * 500,
            'target': ['approved', 'denied'] * 500
        })
        
        report = engine.detect_bias(
            "test_categorical",
            data,
            ['gender'],
            'target'
        )
        
        assert isinstance(report, BiasReport)
        assert len(report.bias_metrics) == 1
    
    def test_bias_detection_error_handling(self, engine):
        """Test error handling in bias detection."""
        # Test with invalid data types
        invalid_data = pd.DataFrame({
            'gender': [1, 2, 3, 4, 5],  # Numeric instead of categorical
            'target': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Should not raise exception but handle gracefully
        report = engine.detect_bias(
            "test_invalid",
            invalid_data,
            ['gender'],
            'target'
        )
        
        assert isinstance(report, BiasReport)
    
    @patch('ai_data_readiness.engines.bias_analysis_engine.logger')
    def test_logging(self, mock_logger, engine, sample_data):
        """Test that appropriate logging occurs."""
        engine.detect_bias(
            "test_logging",
            sample_data,
            ['gender'],
            'target'
        )
        
        # Verify that info logging occurred
        mock_logger.info.assert_called()
    
    def test_edge_case_single_group(self, engine):
        """Test handling of edge case with single group in protected attribute."""
        data = pd.DataFrame({
            'gender': ['male'] * 100,  # Only one group
            'target': [0, 1] * 50
        })
        
        report = engine.detect_bias(
            "test_single_group",
            data,
            ['gender'],
            'target'
        )
        
        assert isinstance(report, BiasReport)
        # Should handle gracefully without errors
    
    def test_missing_target_handling(self, engine, sample_data):
        """Test bias detection without target variable."""
        report = engine.detect_bias(
            "test_no_target",
            sample_data,
            ['gender', 'race'],
            target_column=None
        )
        
        assert isinstance(report, BiasReport)
        assert len(report.bias_metrics) == 2
        # Should still calculate representation-based metrics


if __name__ == "__main__":
    pytest.main([__file__])