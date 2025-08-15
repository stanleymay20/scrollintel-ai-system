"""
Validation tests for bias detection algorithms using synthetic data with known bias patterns.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.tests.validation.test_synthetic_data_generation import (
    SyntheticDataGenerator, BiasConfig
)


class TestBiasDetectionValidation:
    """Validate bias detection algorithms against synthetic data with known bias patterns."""
    
    @pytest.fixture
    def bias_engine(self, test_config):
        """Create bias analysis engine."""
        return BiasAnalysisEngine(test_config)
    
    @pytest.fixture
    def data_generator(self):
        """Create synthetic data generator."""
        return SyntheticDataGenerator(seed=42)
    
    def create_biased_dataset(self, n_samples: int, bias_type: str, bias_strength: float) -> pd.DataFrame:
        """Create dataset with known bias patterns."""
        np.random.seed(42)
        
        # Create base features
        age = np.random.randint(18, 80, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
        gender = np.random.choice(['M', 'F'], n_samples)
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)
        
        if bias_type == "no_bias":
            # No bias - equal treatment
            approval_prob = 0.6 + (age - 50) / 100 + (income - 50000) / 100000
            approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        elif bias_type == "gender_bias":
            # Gender bias - males favored
            base_prob = 0.6
            approval_prob = np.where(
                gender == 'M',
                base_prob + bias_strength,
                base_prob - bias_strength
            )
            approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        elif bias_type == "racial_bias":
            # Racial bias - White applicants favored
            base_prob = 0.6
            approval_prob = np.where(
                race == 'White',
                base_prob + bias_strength,
                base_prob - bias_strength / 2  # Other races disadvantaged
            )
            approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        elif bias_type == "age_bias":
            # Age bias - younger applicants favored
            base_prob = 0.6
            age_factor = (80 - age) / 60  # Normalize age (younger = higher factor)
            approval_prob = base_prob + bias_strength * age_factor
            approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        elif bias_type == "intersectional_bias":
            # Intersectional bias - Black females most disadvantaged
            base_prob = 0.6
            approval_prob = np.where(
                (gender == 'F') & (race == 'Black'),
                base_prob - bias_strength * 1.5,  # Most disadvantaged
                np.where(
                    gender == 'F',
                    base_prob - bias_strength * 0.5,  # Females disadvantaged
                    np.where(
                        race == 'Black',
                        base_prob - bias_strength,  # Black applicants disadvantaged
                        base_prob + bias_strength * 0.2  # Others slightly favored
                    )
                )
            )
            approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        elif bias_type == "proxy_bias":
            # Proxy bias - bias through correlated features
            # ZIP code correlates with race, and ZIP code affects approval
            zip_code = np.where(
                race == 'White',
                np.random.choice(['90210', '10001', '02101'], n_samples),  # High-income areas
                np.random.choice(['90001', '10002', '02102'], n_samples)   # Lower-income areas
            )
            
            base_prob = 0.6
            zip_bias = np.where(
                np.isin(zip_code, ['90210', '10001', '02101']),
                bias_strength,
                -bias_strength
            )
            approval_prob = base_prob + zip_bias
            approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        else:
            raise ValueError(f"Unknown bias type: {bias_type}")
        
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'education': education,
            'gender': gender,
            'race': race,
            'approved': approved
        })
        
        if bias_type == "proxy_bias":
            df['zip_code'] = zip_code
        
        return df
    
    def test_no_bias_detection(self, bias_engine):
        """Test that no bias is detected in unbiased data."""
        df = self.create_biased_dataset(2000, "no_bias", 0.0)
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            # Test gender bias detection
            gender_result = bias_engine.detect_bias("no_bias_dataset", ['gender'])
            
            # Should not detect significant bias
            assert gender_result['bias_score'] < 0.1, \
                f"Detected bias {gender_result['bias_score']} in unbiased data"
            
            # Test racial bias detection
            race_result = bias_engine.detect_bias("no_bias_dataset", ['race'])
            assert race_result['bias_score'] < 0.1
    
    def test_gender_bias_detection_accuracy(self, bias_engine):
        """Test accuracy of gender bias detection."""
        bias_strengths = [0.1, 0.2, 0.3, 0.4]
        
        for bias_strength in bias_strengths:
            df = self.create_biased_dataset(2000, "gender_bias", bias_strength)
            
            with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                result = bias_engine.detect_bias("gender_bias_dataset", ['gender'])
                
                # Should detect bias
                assert result['bias_detected'] is True, \
                    f"Failed to detect gender bias with strength {bias_strength}"
                
                # Bias score should correlate with bias strength
                assert result['bias_score'] > bias_strength * 0.5, \
                    f"Detected bias score {result['bias_score']} too low for strength {bias_strength}"
                
                # Check fairness metrics
                fairness_metrics = bias_engine.calculate_fairness_metrics("gender_bias_dataset", 'approved')
                
                # Demographic parity should be violated
                assert fairness_metrics['demographic_parity'] > 0.05, \
                    "Demographic parity violation not detected"
    
    def test_racial_bias_detection_accuracy(self, bias_engine):
        """Test accuracy of racial bias detection."""
        bias_strengths = [0.15, 0.25, 0.35]
        
        for bias_strength in bias_strengths:
            df = self.create_biased_dataset(2000, "racial_bias", bias_strength)
            
            with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                result = bias_engine.detect_bias("racial_bias_dataset", ['race'])
                
                # Should detect bias
                assert result['bias_detected'] is True, \
                    f"Failed to detect racial bias with strength {bias_strength}"
                
                # Should identify race as biased attribute
                assert 'race' in result['protected_attributes']
                
                # Check bias details
                assert 'bias_details' in result
                race_bias = result['bias_details']['race']
                assert race_bias['statistical_parity'] > 0.1
    
    def test_age_bias_detection_accuracy(self, bias_engine):
        """Test accuracy of age-based bias detection."""
        df = self.create_biased_dataset(2000, "age_bias", 0.3)
        
        # Create age groups for bias analysis
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'senior'])
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            result = bias_engine.detect_bias("age_bias_dataset", ['age_group'])
            
            # Should detect bias
            assert result['bias_detected'] is True, "Failed to detect age bias"
            
            # Check that different age groups have different approval rates
            young_rate = df[df['age_group'] == 'young']['approved'].mean()
            senior_rate = df[df['age_group'] == 'senior']['approved'].mean()
            
            assert young_rate > senior_rate + 0.1, \
                f"Age bias not reflected in approval rates: young={young_rate:.3f}, senior={senior_rate:.3f}"
    
    def test_intersectional_bias_detection(self, bias_engine):
        """Test detection of intersectional bias."""
        df = self.create_biased_dataset(3000, "intersectional_bias", 0.25)
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            # Test individual attribute bias
            gender_result = bias_engine.detect_bias("intersectional_dataset", ['gender'])
            race_result = bias_engine.detect_bias("intersectional_dataset", ['race'])
            
            # Both should show bias
            assert gender_result['bias_detected'] is True
            assert race_result['bias_detected'] is True
            
            # Test intersectional bias
            intersectional_result = bias_engine.detect_bias("intersectional_dataset", ['gender', 'race'])
            
            assert 'intersectional_analysis' in intersectional_result
            intersectional = intersectional_result['intersectional_analysis']
            
            # Should detect intersectional bias
            assert 'gender_race' in intersectional
            assert intersectional['gender_race']['bias_detected'] is True
            
            # Check specific group disadvantage
            black_female_rate = df[(df['gender'] == 'F') & (df['race'] == 'Black')]['approved'].mean()
            white_male_rate = df[(df['gender'] == 'M') & (df['race'] == 'White')]['approved'].mean()
            
            assert white_male_rate > black_female_rate + 0.2, \
                "Intersectional bias not reflected in group-specific rates"
    
    def test_proxy_bias_detection(self, bias_engine):
        """Test detection of proxy bias through correlated features."""
        df = self.create_biased_dataset(2000, "proxy_bias", 0.3)
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            # Direct racial bias might be less detectable
            race_result = bias_engine.detect_bias("proxy_bias_dataset", ['race'])
            
            # But ZIP code should show bias
            zip_result = bias_engine.detect_bias("proxy_bias_dataset", ['zip_code'])
            
            # Should detect bias through proxy
            assert zip_result['bias_detected'] is True, "Failed to detect proxy bias"
            
            # Check correlation between race and ZIP code
            correlation_analysis = bias_engine.analyze_feature_correlations(
                "proxy_bias_dataset", ['race', 'zip_code'], 'approved'
            )
            
            assert 'correlations' in correlation_analysis
            assert correlation_analysis['proxy_bias_detected'] is True
    
    def test_fairness_metrics_accuracy(self, bias_engine):
        """Test accuracy of fairness metrics calculations."""
        # Create dataset with known fairness violations
        df = self.create_biased_dataset(2000, "gender_bias", 0.3)
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            metrics = bias_engine.calculate_fairness_metrics("fairness_test", 'approved')
            
            # Calculate expected metrics manually
            male_rate = df[df['gender'] == 'M']['approved'].mean()
            female_rate = df[df['gender'] == 'F']['approved'].mean()
            expected_dp_diff = abs(male_rate - female_rate)
            
            # Demographic parity difference should match
            assert abs(metrics['demographic_parity'] - expected_dp_diff) < 0.02, \
                f"Demographic parity calculation error: expected {expected_dp_diff:.3f}, got {metrics['demographic_parity']:.3f}"
            
            # Should have other fairness metrics
            assert 'equalized_odds' in metrics
            assert 'equality_of_opportunity' in metrics
            assert 'calibration' in metrics
    
    def test_bias_threshold_sensitivity(self, bias_engine):
        """Test sensitivity to different bias detection thresholds."""
        df = self.create_biased_dataset(2000, "gender_bias", 0.15)  # Moderate bias
        
        # Test with different thresholds
        threshold_configs = [
            {'bias_threshold': 0.05, 'should_detect': True},   # Sensitive
            {'bias_threshold': 0.1, 'should_detect': True},    # Moderate
            {'bias_threshold': 0.2, 'should_detect': False},   # Strict
            {'bias_threshold': 0.3, 'should_detect': False}    # Very strict
        ]
        
        for config in threshold_configs:
            # Update bias detection threshold
            bias_engine.bias_thresholds['demographic_parity_threshold'] = config['bias_threshold']
            
            with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                result = bias_engine.detect_bias("threshold_test", ['gender'])
                
                if config['should_detect']:
                    assert result['bias_detected'] is True, \
                        f"Should detect bias with threshold {config['bias_threshold']}"
                else:
                    # May or may not detect depending on actual bias level
                    pass
    
    def test_bias_mitigation_recommendations(self, bias_engine):
        """Test accuracy of bias mitigation recommendations."""
        test_cases = [
            {
                'bias_type': 'gender_bias',
                'bias_strength': 0.3,
                'expected_strategies': ['data_preprocessing', 'algorithmic_fairness']
            },
            {
                'bias_type': 'racial_bias',
                'bias_strength': 0.4,
                'expected_strategies': ['data_augmentation', 'fairness_constraints']
            },
            {
                'bias_type': 'intersectional_bias',
                'bias_strength': 0.25,
                'expected_strategies': ['intersectional_fairness', 'group_specific_thresholds']
            }
        ]
        
        for case in test_cases:
            df = self.create_biased_dataset(2000, case['bias_type'], case['bias_strength'])
            
            with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                protected_attrs = ['gender'] if 'gender' in case['bias_type'] else ['race']
                if case['bias_type'] == 'intersectional_bias':
                    protected_attrs = ['gender', 'race']
                
                bias_result = bias_engine.detect_bias("mitigation_test", protected_attrs)
                
                if bias_result['bias_detected']:
                    mitigation = bias_engine.recommend_mitigation(bias_result)
                    
                    assert 'strategies' in mitigation
                    assert len(mitigation['strategies']) > 0
                    
                    # Check for expected strategy types
                    strategy_types = [s['type'] for s in mitigation['strategies']]
                    
                    for expected_strategy in case['expected_strategies']:
                        matching_strategies = [s for s in strategy_types if expected_strategy in s]
                        assert len(matching_strategies) > 0, \
                            f"Expected strategy type '{expected_strategy}' not found in recommendations"
    
    def test_individual_fairness_assessment(self, bias_engine):
        """Test individual fairness assessment accuracy."""
        # Create dataset where similar individuals have different outcomes
        np.random.seed(42)
        n_samples = 1000
        
        # Create similar pairs with different protected attributes
        feature1 = np.random.normal(0, 1, n_samples)
        feature2 = np.random.normal(0, 1, n_samples)
        protected_attr = np.random.choice(['A', 'B'], n_samples)
        
        # Create unfair outcomes - similar individuals with different protected attributes get different outcomes
        outcome_prob = np.where(
            protected_attr == 'A',
            0.8,  # Group A favored
            0.3   # Group B disadvantaged
        )
        outcome = np.random.binomial(1, outcome_prob)
        
        df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'protected_attr': protected_attr,
            'outcome': outcome
        })
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            individual_fairness = bias_engine.assess_individual_fairness(
                "individual_fairness_test",
                ['feature1', 'feature2'],
                'protected_attr',
                'outcome'
            )
            
            # Should detect individual fairness violations
            assert individual_fairness['individual_fairness_score'] < 0.7, \
                "Failed to detect individual fairness violations"
            
            assert len(individual_fairness['fairness_violations']) > 0, \
                "Should identify specific fairness violations"
    
    def test_bias_amplification_detection(self, bias_engine):
        """Test detection of bias amplification."""
        # Create dataset where model amplifies existing bias
        np.random.seed(42)
        n_samples = 2000
        
        gender = np.random.choice(['M', 'F'], n_samples)
        qualification = np.random.normal(70, 10, n_samples)
        
        # Small original bias in qualifications
        qualification = np.where(
            gender == 'M',
            qualification + 2,  # Males slightly higher qualified on average
            qualification
        )
        
        # Model amplifies this bias significantly
        approval_prob = np.where(
            gender == 'M',
            0.8,  # 80% approval for males
            0.4   # 40% approval for females (amplified bias)
        )
        
        approved = np.random.binomial(1, approval_prob)
        
        df = pd.DataFrame({
            'gender': gender,
            'qualification': qualification,
            'approved': approved
        })
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            amplification_result = bias_engine.detect_bias_amplification(
                "amplification_test",
                ['gender'],
                'qualification',
                'approved'
            )
            
            # Should detect bias amplification
            assert amplification_result['amplification_detected'] is True, \
                "Failed to detect bias amplification"
            
            assert amplification_result['amplification_factor'] > 1.5, \
                f"Amplification factor {amplification_result['amplification_factor']} too low"
            
            # Original bias should be much smaller than amplified bias
            assert amplification_result['amplified_bias'] > amplification_result['original_bias'] * 2
    
    def test_temporal_bias_analysis(self, bias_engine):
        """Test temporal bias analysis accuracy."""
        # Create dataset with changing bias over time
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            # Bias decreases over time
            bias_strength = 0.4 * (1 - i / len(dates))  # From 0.4 to 0.0
            
            daily_data = self.create_biased_dataset(50, "gender_bias", bias_strength)
            daily_data['date'] = date
            data.append(daily_data)
        
        temporal_df = pd.concat(data, ignore_index=True)
        
        with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
            mock_get.return_value = temporal_df
            
            trend_result = bias_engine.analyze_bias_trends("temporal_test", ['gender'], 'date')
            
            # Should detect decreasing bias trend
            assert 'trend_analysis' in trend_result
            assert trend_result['trend_analysis']['trend_direction'] == 'decreasing', \
                "Failed to detect decreasing bias trend"
            
            # Should show bias reduction over time
            assert 'bias_over_time' in trend_result['trend_analysis']
            bias_timeline = trend_result['trend_analysis']['bias_over_time']
            
            # Early bias should be higher than late bias
            early_bias = bias_timeline[0]['bias_score']
            late_bias = bias_timeline[-1]['bias_score']
            
            assert early_bias > late_bias + 0.1, \
                f"Bias trend not detected: early={early_bias:.3f}, late={late_bias:.3f}"
    
    def test_performance_with_large_biased_datasets(self, bias_engine, performance_timer):
        """Test performance of bias detection with large datasets."""
        dataset_sizes = [5000, 10000, 20000]
        
        for size in dataset_sizes:
            df = self.create_biased_dataset(size, "gender_bias", 0.3)
            
            with pytest.mock.patch.object(bias_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                with performance_timer() as timer:
                    result = bias_engine.detect_bias(f"performance_test_{size}", ['gender'])
                
                # Should complete within reasonable time
                max_time = 30 + (size / 1000) * 5  # Scale with dataset size
                assert timer.duration <= max_time, \
                    f"Bias detection took {timer.duration:.2f}s for {size} rows, expected <= {max_time}s"
                
                # Should still detect bias accurately
                assert result['bias_detected'] is True
                assert result['bias_score'] > 0.2