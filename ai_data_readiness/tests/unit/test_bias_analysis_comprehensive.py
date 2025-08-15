"""
Comprehensive unit tests for Bias Analysis Engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.core.exceptions import AIDataReadinessError


class TestBiasAnalysisEngineComprehensive:
    """Comprehensive test suite for BiasAnalysisEngine."""
    
    @pytest.fixture
    def bias_engine(self, test_config):
        """Create BiasAnalysisEngine instance for testing."""
        return BiasAnalysisEngine(test_config)
    
    def test_init(self, test_config):
        """Test BiasAnalysisEngine initialization."""
        engine = BiasAnalysisEngine(test_config)
        assert engine.config == test_config
        assert hasattr(engine, 'fairness_metrics')
        assert hasattr(engine, 'bias_thresholds')
        assert hasattr(engine, 'protected_attributes')
    
    def test_detect_gender_bias(self, bias_engine, sample_biased_data):
        """Test gender bias detection."""
        dataset_id = "biased_dataset"
        protected_attributes = ['gender']
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            result = bias_engine.detect_bias(dataset_id, protected_attributes)
            
            assert result['dataset_id'] == dataset_id
            assert result['bias_detected'] is True
            assert 'gender' in result['protected_attributes']
            assert result['bias_score'] > 0.1  # Should detect significant bias
            
            # Check bias details
            assert 'bias_details' in result
            bias_details = result['bias_details']
            assert 'gender' in bias_details
            
            gender_bias = bias_details['gender']
            assert 'statistical_parity' in gender_bias
            assert 'equalized_odds' in gender_bias
            assert 'demographic_parity' in gender_bias
    
    def test_detect_age_bias(self, bias_engine):
        """Test age-based bias detection."""
        # Create age-biased dataset
        np.random.seed(42)
        n_samples = 1000
        
        age = np.random.randint(22, 65, n_samples)
        # Create age groups
        age_group = np.where(age < 35, 'young', 
                    np.where(age < 50, 'middle', 'senior'))
        
        # Introduce age bias in approval rates
        approval_prob = np.where(age_group == 'young', 0.8,
                        np.where(age_group == 'middle', 0.6, 0.3))
        
        approved = np.random.binomial(1, approval_prob)
        
        biased_data = pd.DataFrame({
            'age': age,
            'age_group': age_group,
            'approved': approved
        })
        
        dataset_id = "age_biased_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = biased_data
            
            result = bias_engine.detect_bias(dataset_id, ['age_group'])
            
            assert result['bias_detected'] is True
            assert 'age_group' in result['protected_attributes']
            assert result['bias_score'] > 0.2  # Significant age bias
    
    def test_intersectional_bias_detection(self, bias_engine):
        """Test intersectional bias detection (multiple protected attributes)."""
        np.random.seed(42)
        n_samples = 1000
        
        gender = np.random.choice(['M', 'F'], n_samples)
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)
        
        # Create intersectional bias
        # Black females have lowest approval rate
        approval_prob = np.where(
            (gender == 'F') & (race == 'Black'), 0.3,
            np.where(gender == 'F', 0.5,
                    np.where(race == 'Black', 0.4, 0.7))
        )
        
        approved = np.random.binomial(1, approval_prob)
        
        intersectional_data = pd.DataFrame({
            'gender': gender,
            'race': race,
            'approved': approved
        })
        
        dataset_id = "intersectional_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = intersectional_data
            
            result = bias_engine.detect_bias(dataset_id, ['gender', 'race'])
            
            assert result['bias_detected'] is True
            assert 'intersectional_analysis' in result
            
            intersectional = result['intersectional_analysis']
            assert 'gender_race' in intersectional
            assert intersectional['gender_race']['bias_detected'] is True
    
    def test_calculate_fairness_metrics(self, bias_engine, sample_biased_data):
        """Test fairness metrics calculation."""
        dataset_id = "test_dataset"
        target_column = 'approved'
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            metrics = bias_engine.calculate_fairness_metrics(dataset_id, target_column)
            
            assert 'demographic_parity' in metrics
            assert 'equalized_odds' in metrics
            assert 'equality_of_opportunity' in metrics
            assert 'calibration' in metrics
            
            # All metrics should be between 0 and 1
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric in metric_value.values():
                        if isinstance(sub_metric, (int, float)):
                            assert 0 <= sub_metric <= 1
                else:
                    assert 0 <= metric_value <= 1
    
    def test_demographic_parity_calculation(self, bias_engine):
        """Test demographic parity calculation."""
        # Create data with known demographic parity violation
        data = pd.DataFrame({
            'gender': ['M'] * 500 + ['F'] * 500,
            'approved': [1] * 400 + [0] * 100 + [1] * 200 + [0] * 300  # M: 80%, F: 40%
        })
        
        dp_score = bias_engine._calculate_demographic_parity(data, 'gender', 'approved')
        
        # Demographic parity difference should be 0.4 (80% - 40%)
        assert abs(dp_score - 0.4) < 0.01
    
    def test_equalized_odds_calculation(self, bias_engine):
        """Test equalized odds calculation."""
        # Create data with known equalized odds violation
        data = pd.DataFrame({
            'gender': ['M'] * 400 + ['F'] * 400,
            'approved': [1] * 200 + [0] * 200 + [1] * 100 + [0] * 300,
            'actual': [1] * 150 + [0] * 50 + [1] * 150 + [0] * 50 + 
                     [1] * 75 + [0] * 25 + [1] * 75 + [0] * 225
        })
        
        eo_score = bias_engine._calculate_equalized_odds(data, 'gender', 'approved', 'actual')
        
        # Should detect equalized odds violation
        assert eo_score > 0.1
    
    def test_statistical_parity_test(self, bias_engine, sample_biased_data):
        """Test statistical parity test."""
        result = bias_engine._statistical_parity_test(
            sample_biased_data, 'gender', 'approved'
        )
        
        assert 'parity_difference' in result
        assert 'p_value' in result
        assert 'significant' in result
        
        # Should detect significant parity violation
        assert result['significant'] is True
        assert result['parity_difference'] > 0.1
    
    def test_chi_square_independence_test(self, bias_engine, sample_biased_data):
        """Test chi-square independence test."""
        result = bias_engine._chi_square_independence_test(
            sample_biased_data, 'gender', 'approved'
        )
        
        assert 'chi2_statistic' in result
        assert 'p_value' in result
        assert 'degrees_of_freedom' in result
        assert 'dependent' in result
        
        # Should detect dependence between gender and approval
        assert result['dependent'] is True
        assert result['p_value'] < 0.05
    
    def test_recommend_mitigation_strategies(self, bias_engine):
        """Test bias mitigation recommendations."""
        bias_report = {
            'dataset_id': 'test_dataset',
            'bias_detected': True,
            'protected_attributes': ['gender', 'age'],
            'bias_score': 0.4,
            'bias_details': {
                'gender': {
                    'demographic_parity': 0.3,
                    'equalized_odds': 0.25,
                    'severity': 'high'
                },
                'age': {
                    'demographic_parity': 0.15,
                    'equalized_odds': 0.1,
                    'severity': 'medium'
                }
            }
        }
        
        mitigation = bias_engine.recommend_mitigation(bias_report)
        
        assert 'strategies' in mitigation
        assert 'priority_order' in mitigation
        assert 'estimated_effectiveness' in mitigation
        
        strategies = mitigation['strategies']
        assert len(strategies) > 0
        
        # Should recommend appropriate strategies
        strategy_types = [s['type'] for s in strategies]
        assert 'data_preprocessing' in strategy_types
        assert 'algorithmic_fairness' in strategy_types
        
        # High-severity bias should have high-priority recommendations
        high_priority_strategies = [s for s in strategies if s['priority'] == 'high']
        assert len(high_priority_strategies) > 0
    
    def test_validate_fairness_constraints(self, bias_engine, sample_biased_data):
        """Test fairness constraint validation."""
        dataset_id = "test_dataset"
        
        fairness_constraints = {
            'demographic_parity_threshold': 0.1,
            'equalized_odds_threshold': 0.1,
            'equality_of_opportunity_threshold': 0.1,
            'protected_attributes': ['gender'],
            'target_column': 'approved'
        }
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            result = bias_engine.validate_fairness(dataset_id, fairness_constraints)
            
            assert 'constraints_satisfied' in result
            assert 'violations' in result
            assert 'fairness_score' in result
            
            # Should detect constraint violations
            assert result['constraints_satisfied'] is False
            assert len(result['violations']) > 0
    
    def test_bias_detection_with_missing_values(self, bias_engine):
        """Test bias detection with missing values in protected attributes."""
        data_with_missing = pd.DataFrame({
            'gender': ['M', 'F', None, 'M', 'F', None, 'M', 'F'],
            'age': [25, 30, 35, None, 45, 50, 55, None],
            'approved': [1, 0, 1, 1, 0, 0, 1, 0]
        })
        
        dataset_id = "missing_data_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = data_with_missing
            
            result = bias_engine.detect_bias(dataset_id, ['gender'])
            
            assert 'missing_value_handling' in result
            assert result['missing_value_handling']['method'] in ['exclude', 'impute', 'separate_category']
    
    def test_continuous_protected_attribute_binning(self, bias_engine):
        """Test handling of continuous protected attributes through binning."""
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, 1000),
            'income': np.random.normal(50000, 15000, 1000),
            'approved': np.random.choice([0, 1], 1000)
        })
        
        dataset_id = "continuous_attr_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = data
            
            result = bias_engine.detect_bias(dataset_id, ['age'])
            
            assert 'binning_applied' in result
            assert result['binning_applied'] is True
            assert 'age_bins' in result
    
    def test_bias_trend_analysis(self, bias_engine):
        """Test bias trend analysis over time."""
        # Create temporal data with changing bias patterns
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        n_samples = len(dates)
        
        # Bias decreases over time
        bias_factor = np.linspace(0.4, 0.1, n_samples)
        
        data = []
        for i, (date, bias) in enumerate(zip(dates, bias_factor)):
            gender = np.random.choice(['M', 'F'], 100)
            approval_prob = np.where(gender == 'M', 0.7, 0.7 - bias)
            approved = np.random.binomial(1, approval_prob)
            
            day_data = pd.DataFrame({
                'date': [date] * 100,
                'gender': gender,
                'approved': approved
            })
            data.append(day_data)
        
        temporal_data = pd.concat(data, ignore_index=True)
        dataset_id = "temporal_bias_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = temporal_data
            
            result = bias_engine.analyze_bias_trends(dataset_id, ['gender'], 'date')
            
            assert 'trend_analysis' in result
            assert 'bias_over_time' in result['trend_analysis']
            assert 'trend_direction' in result['trend_analysis']
            
            # Should detect decreasing bias trend
            assert result['trend_analysis']['trend_direction'] == 'decreasing'
    
    def test_group_fairness_metrics(self, bias_engine, sample_biased_data):
        """Test group fairness metrics calculation."""
        dataset_id = "test_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            group_metrics = bias_engine.calculate_group_fairness_metrics(
                dataset_id, ['gender'], 'approved'
            )
            
            assert 'group_statistics' in group_metrics
            assert 'fairness_metrics' in group_metrics
            assert 'disparity_measures' in group_metrics
            
            group_stats = group_metrics['group_statistics']
            assert 'M' in group_stats
            assert 'F' in group_stats
            
            for group in ['M', 'F']:
                assert 'count' in group_stats[group]
                assert 'approval_rate' in group_stats[group]
                assert 'base_rate' in group_stats[group]
    
    def test_individual_fairness_assessment(self, bias_engine):
        """Test individual fairness assessment."""
        # Create data where similar individuals have different outcomes
        data = pd.DataFrame({
            'feature1': [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            'feature2': [0.5, 0.6, 1.0, 1.1, 1.5, 1.6],
            'protected_attr': ['A', 'B', 'A', 'B', 'A', 'B'],
            'outcome': [1, 0, 1, 0, 1, 0]  # Similar individuals, different outcomes
        })
        
        dataset_id = "individual_fairness_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = data
            
            result = bias_engine.assess_individual_fairness(
                dataset_id, ['feature1', 'feature2'], 'protected_attr', 'outcome'
            )
            
            assert 'individual_fairness_score' in result
            assert 'similar_pairs_analysis' in result
            assert 'fairness_violations' in result
            
            # Should detect individual fairness violations
            assert result['individual_fairness_score'] < 0.8
            assert len(result['fairness_violations']) > 0
    
    def test_bias_amplification_detection(self, bias_engine):
        """Test detection of bias amplification."""
        # Create data where model amplifies existing bias
        original_data = pd.DataFrame({
            'gender': ['M'] * 500 + ['F'] * 500,
            'qualification_score': np.concatenate([
                np.random.normal(70, 10, 500),  # Males
                np.random.normal(68, 10, 500)   # Females (slight difference)
            ]),
            'hired': np.concatenate([
                np.random.choice([0, 1], 500, p=[0.3, 0.7]),  # Males: 70% hired
                np.random.choice([0, 1], 500, p=[0.6, 0.4])   # Females: 40% hired (amplified)
            ])
        })
        
        dataset_id = "amplification_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = original_data
            
            result = bias_engine.detect_bias_amplification(
                dataset_id, ['gender'], 'qualification_score', 'hired'
            )
            
            assert 'amplification_detected' in result
            assert 'amplification_factor' in result
            assert 'original_bias' in result
            assert 'amplified_bias' in result
            
            # Should detect bias amplification
            assert result['amplification_detected'] is True
            assert result['amplification_factor'] > 1.0
    
    def test_error_handling_empty_dataset(self, bias_engine):
        """Test error handling for empty datasets."""
        empty_data = pd.DataFrame()
        dataset_id = "empty_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = empty_data
            
            with pytest.raises(AIDataReadinessError):
                bias_engine.detect_bias(dataset_id, ['gender'])
    
    def test_error_handling_missing_protected_attributes(self, bias_engine, sample_csv_data):
        """Test error handling for missing protected attributes."""
        dataset_id = "test_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            with pytest.raises(AIDataReadinessError):
                bias_engine.detect_bias(dataset_id, ['nonexistent_attribute'])
    
    def test_error_handling_missing_target_column(self, bias_engine, sample_biased_data):
        """Test error handling for missing target column."""
        dataset_id = "test_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            with pytest.raises(AIDataReadinessError):
                bias_engine.calculate_fairness_metrics(dataset_id, 'nonexistent_target')
    
    def test_performance_with_large_dataset(self, bias_engine, performance_timer):
        """Test performance with large dataset."""
        # Create large biased dataset
        np.random.seed(42)
        n_samples = 10000
        
        gender = np.random.choice(['M', 'F'], n_samples)
        approved = np.where(
            gender == 'M',
            np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        )
        
        large_biased_data = pd.DataFrame({
            'gender': gender,
            'approved': approved
        })
        
        dataset_id = "large_biased_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = large_biased_data
            
            with performance_timer() as timer:
                result = bias_engine.detect_bias(dataset_id, ['gender'])
            
            # Should complete within reasonable time
            assert timer.duration < 5.0  # 5 seconds max
            assert result['bias_detected'] is True