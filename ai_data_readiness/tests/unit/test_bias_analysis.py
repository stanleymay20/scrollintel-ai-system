"""
Unit tests for Bias Analysis Engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.core.exceptions import AIDataReadinessError


class TestBiasAnalysisEngine:
    """Test suite for BiasAnalysisEngine."""
    
    @pytest.fixture
    def bias_engine(self, test_config):
        """Create BiasAnalysisEngine instance for testing."""
        return BiasAnalysisEngine(test_config)
    
    def test_init(self, test_config):
        """Test BiasAnalysisEngine initialization."""
        engine = BiasAnalysisEngine(test_config)
        assert engine.config == test_config
        assert hasattr(engine, 'fairness_metrics')
        assert hasattr(engine, 'protected_attributes')
    
    def test_detect_bias_gender(self, bias_engine, sample_biased_data):
        """Test bias detection for gender attribute."""
        dataset_id = "biased_dataset_123"
        protected_attributes = ['gender']
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            result = bias_engine.detect_bias(dataset_id, protected_attributes)
            
            assert result is not None
            assert 'dataset_id' in result
            assert result['dataset_id'] == dataset_id
            
            assert 'bias_detected' in result
            assert 'protected_attributes' in result
            assert 'bias_score' in result
            assert 'bias_details' in result
            
            # Should detect bias in the biased dataset
            assert result['bias_detected'] is True
            assert result['protected_attributes'] == protected_attributes
            assert 0 <= result['bias_score'] <= 1
    
    def test_calculate_demographic_parity(self, bias_engine, sample_biased_data):
        """Test demographic parity calculation."""
        target_column = 'approved'
        protected_attribute = 'gender'
        
        dp_score = bias_engine._calculate_demographic_parity(
            sample_biased_data, protected_attribute, target_column
        )
        
        assert 'demographic_parity' in dp_score
        assert 'group_rates' in dp_score
        assert 'parity_difference' in dp_score
        
        # Check that we have rates for both groups
        group_rates = dp_score['group_rates']
        assert 'M' in group_rates
        assert 'F' in group_rates
        
        # Parity difference should indicate bias
        assert dp_score['parity_difference'] > 0.1  # Significant difference
    
    def test_calculate_equalized_odds(self, bias_engine, sample_biased_data):
        """Test equalized odds calculation."""
        target_column = 'approved'
        protected_attribute = 'gender'
        
        eo_score = bias_engine._calculate_equalized_odds(
            sample_biased_data, protected_attribute, target_column
        )
        
        assert 'equalized_odds' in eo_score
        assert 'true_positive_rates' in eo_score
        assert 'false_positive_rates' in eo_score
        assert 'odds_difference' in eo_score
        
        # Check that we have rates for both groups
        tpr = eo_score['true_positive_rates']
        fpr = eo_score['false_positive_rates']
        
        assert 'M' in tpr and 'F' in tpr
        assert 'M' in fpr and 'F' in fpr
    
    def test_calculate_equal_opportunity(self, bias_engine, sample_biased_data):
        """Test equal opportunity calculation."""
        target_column = 'approved'
        protected_attribute = 'gender'
        
        eo_score = bias_engine._calculate_equal_opportunity(
            sample_biased_data, protected_attribute, target_column
        )
        
        assert 'equal_opportunity' in eo_score
        assert 'true_positive_rates' in eo_score
        assert 'opportunity_difference' in eo_score
        
        # Should detect difference in true positive rates
        assert eo_score['opportunity_difference'] > 0
    
    def test_calculate_fairness_metrics(self, bias_engine, sample_biased_data):
        """Test comprehensive fairness metrics calculation."""
        dataset_id = "biased_dataset_123"
        target_column = 'approved'
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            metrics = bias_engine.calculate_fairness_metrics(dataset_id, target_column)
            
            assert 'fairness_metrics' in metrics
            assert 'overall_fairness_score' in metrics
            assert 'metric_details' in metrics
            
            fairness_metrics = metrics['fairness_metrics']
            
            # Should include all major fairness metrics
            expected_metrics = [
                'demographic_parity', 'equalized_odds', 'equal_opportunity',
                'calibration', 'individual_fairness'
            ]
            
            for metric in expected_metrics:
                assert metric in fairness_metrics
                assert 0 <= fairness_metrics[metric] <= 1
    
    def test_identify_protected_attributes(self, bias_engine):
        """Test automatic identification of protected attributes."""
        # Create dataset with potential protected attributes
        data_with_protected = pd.DataFrame({
            'age': np.random.randint(18, 80, 1000),
            'gender': np.random.choice(['M', 'F'], 1000),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], 1000),
            'religion': np.random.choice(['Christian', 'Muslim', 'Jewish', 'Other'], 1000),
            'income': np.random.normal(50000, 15000, 1000),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        protected_attrs = bias_engine._identify_protected_attributes(data_with_protected)
        
        assert 'identified_attributes' in protected_attrs
        assert 'confidence_scores' in protected_attrs
        
        identified = protected_attrs['identified_attributes']
        
        # Should identify common protected attributes
        expected_protected = ['gender', 'race', 'religion']
        for attr in expected_protected:
            assert attr in identified
    
    def test_statistical_parity_test(self, bias_engine, sample_biased_data):
        """Test statistical parity testing."""
        protected_attribute = 'gender'
        target_column = 'approved'
        
        parity_test = bias_engine._statistical_parity_test(
            sample_biased_data, protected_attribute, target_column
        )
        
        assert 'test_statistic' in parity_test
        assert 'p_value' in parity_test
        assert 'is_significant' in parity_test
        assert 'effect_size' in parity_test
        
        # Should detect significant bias
        assert parity_test['is_significant'] is True
        assert parity_test['p_value'] < 0.05
    
    def test_intersectional_bias_analysis(self, bias_engine):
        """Test intersectional bias analysis."""
        # Create data with intersectional bias
        np.random.seed(42)
        n_samples = 1000
        
        gender = np.random.choice(['M', 'F'], n_samples)
        race = np.random.choice(['White', 'Black'], n_samples)
        
        # Create intersectional bias: Black females most disadvantaged
        bias_factor = np.where(
            (gender == 'F') & (race == 'Black'), 0.2,  # Lowest approval rate
            np.where(gender == 'F', 0.4,               # Female disadvantage
                    np.where(race == 'Black', 0.5, 0.7))  # Racial disadvantage
        )
        
        approved = np.random.binomial(1, bias_factor)
        
        intersectional_data = pd.DataFrame({
            'gender': gender,
            'race': race,
            'approved': approved
        })
        
        dataset_id = "intersectional_dataset"
        protected_attributes = ['gender', 'race']
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = intersectional_data
            
            result = bias_engine.detect_bias(dataset_id, protected_attributes)
            
            assert 'intersectional_analysis' in result
            intersectional = result['intersectional_analysis']
            
            assert 'group_combinations' in intersectional
            assert 'bias_amplification' in intersectional
            
            # Should detect amplified bias for intersectional groups
            assert intersectional['bias_amplification'] is True
    
    def test_recommend_mitigation_strategies(self, bias_engine):
        """Test bias mitigation recommendations."""
        # Create bias report
        bias_report = {
            'bias_detected': True,
            'bias_score': 0.7,
            'protected_attributes': ['gender', 'race'],
            'fairness_violations': [
                {
                    'metric': 'demographic_parity',
                    'severity': 'high',
                    'affected_groups': ['F', 'Black']
                },
                {
                    'metric': 'equal_opportunity',
                    'severity': 'medium',
                    'affected_groups': ['F']
                }
            ]
        }
        
        mitigation = bias_engine.recommend_mitigation(bias_report)
        
        assert 'mitigation_strategies' in mitigation
        assert 'priority_actions' in mitigation
        assert 'implementation_guidance' in mitigation
        
        strategies = mitigation['mitigation_strategies']
        assert len(strategies) > 0
        
        # Should include different types of mitigation
        strategy_types = [s['type'] for s in strategies]
        expected_types = ['preprocessing', 'in_processing', 'post_processing']
        
        for strategy_type in expected_types:
            assert strategy_type in strategy_types
    
    def test_validate_fairness_constraints(self, bias_engine, sample_biased_data):
        """Test fairness constraint validation."""
        dataset_id = "test_dataset"
        
        fairness_constraints = {
            'demographic_parity': {'threshold': 0.8, 'protected_attributes': ['gender']},
            'equal_opportunity': {'threshold': 0.9, 'protected_attributes': ['gender']},
            'equalized_odds': {'threshold': 0.85, 'protected_attributes': ['gender']}
        }
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_biased_data
            
            validation = bias_engine.validate_fairness(dataset_id, fairness_constraints)
            
            assert 'validation_passed' in validation
            assert 'constraint_violations' in validation
            assert 'compliance_score' in validation
            
            # Should fail validation due to bias in data
            assert validation['validation_passed'] is False
            assert len(validation['constraint_violations']) > 0
    
    def test_bias_trend_analysis(self, bias_engine):
        """Test bias trend analysis over time."""
        # Create time-series data with changing bias
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        n_samples_per_day = 10
        
        all_data = []
        for i, date in enumerate(dates):
            # Gradually reduce bias over time
            bias_factor = 0.8 - (i / len(dates)) * 0.3  # From 0.8 to 0.5
            
            gender = np.random.choice(['M', 'F'], n_samples_per_day)
            approved = np.where(
                gender == 'M',
                np.random.binomial(1, bias_factor, n_samples_per_day),
                np.random.binomial(1, bias_factor * 0.7, n_samples_per_day)
            )
            
            day_data = pd.DataFrame({
                'date': [date] * n_samples_per_day,
                'gender': gender,
                'approved': approved
            })
            all_data.append(day_data)
        
        time_series_data = pd.concat(all_data, ignore_index=True)
        
        trend_analysis = bias_engine._analyze_bias_trends(
            time_series_data, 'gender', 'approved', 'date'
        )
        
        assert 'trend_direction' in trend_analysis
        assert 'trend_significance' in trend_analysis
        assert 'monthly_bias_scores' in trend_analysis
        
        # Should detect improving trend (decreasing bias)
        assert trend_analysis['trend_direction'] == 'improving'
    
    def test_performance_with_large_dataset(self, bias_engine, performance_timer):
        """Test performance with large dataset."""
        # Create large biased dataset
        np.random.seed(42)
        n_samples = 50000
        
        gender = np.random.choice(['M', 'F'], n_samples)
        approved = np.where(
            gender == 'M',
            np.random.binomial(1, 0.7, n_samples),
            np.random.binomial(1, 0.4, n_samples)
        )
        
        large_biased_data = pd.DataFrame({
            'gender': gender,
            'approved': approved
        })
        
        dataset_id = "large_biased_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = large_biased_data
            
            with performance_timer:
                result = bias_engine.detect_bias(dataset_id, ['gender'])
            
            # Should complete within reasonable time
            assert performance_timer.duration < 15.0  # 15 seconds max
            assert result is not None
            assert result['bias_detected'] is True
    
    def test_error_handling_missing_target(self, bias_engine, sample_csv_data):
        """Test error handling when target column is missing."""
        dataset_id = "test_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            with pytest.raises(AIDataReadinessError):
                bias_engine.calculate_fairness_metrics(dataset_id, 'nonexistent_target')
    
    def test_error_handling_missing_protected_attribute(self, bias_engine, sample_csv_data):
        """Test error handling when protected attribute is missing."""
        dataset_id = "test_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            with pytest.raises(AIDataReadinessError):
                bias_engine.detect_bias(dataset_id, ['nonexistent_attribute'])
    
    def test_error_handling_insufficient_data(self, bias_engine):
        """Test error handling with insufficient data."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'gender': ['M', 'F'],
            'approved': [1, 0]
        })
        
        dataset_id = "small_dataset"
        
        with patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = small_data
            
            with pytest.raises(AIDataReadinessError):
                bias_engine.detect_bias(dataset_id, ['gender'])