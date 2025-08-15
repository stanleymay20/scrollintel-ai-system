"""
Unit tests for AI-specific quality metrics in the Quality Assessment Engine.

Tests the AI readiness scoring algorithm, feature correlation detection,
target leakage detection, and statistical anomaly detection capabilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.models.base_models import (
    Dataset, DatasetMetadata, Schema, AIReadinessScore, DimensionScore
)
from ai_data_readiness.core.config import Config


class TestAIQualityMetrics:
    """Test suite for AI-specific quality metrics."""
    
    @pytest.fixture
    def engine(self):
        """Create a quality assessment engine instance."""
        config = Mock(spec=Config)
        config.quality = Mock()
        config.quality.completeness_threshold = 0.8
        config.quality.accuracy_threshold = 0.8
        config.quality.consistency_threshold = 0.8
        config.quality.validity_threshold = 0.8
        return QualityAssessmentEngine(config)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        schema = Schema(
            columns={
                'feature1': 'float',
                'feature2': 'float', 
                'feature3': 'integer',
                'category': 'categorical',
                'target': 'integer'
            }
        )
        
        metadata = DatasetMetadata(
            name="test_dataset",
            description="Test dataset for AI quality metrics",
            row_count=1000,
            column_count=5
        )
        
        return Dataset(
            id="test_dataset_001",
            name="Test Dataset",
            schema=schema,
            metadata=metadata
        )
    
    @pytest.fixture
    def good_quality_data(self):
        """Create high-quality data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create uncorrelated features
        feature1 = np.random.normal(0, 1, n_samples)
        feature2 = np.random.normal(5, 2, n_samples)
        feature3 = np.random.randint(1, 100, n_samples)
        category = np.random.choice(['A', 'B', 'C'], n_samples)
        
        # Create balanced target
        target = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        
        return pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'category': category,
            'target': target
        })
    
    @pytest.fixture
    def poor_quality_data(self):
        """Create poor-quality data for testing."""
        np.random.seed(42)
        n_samples = 100  # Small dataset
        
        # Create highly correlated features
        base_feature = np.random.normal(0, 1, n_samples)
        feature1 = base_feature + np.random.normal(0, 0.1, n_samples)  # High correlation
        feature2 = base_feature * 2 + np.random.normal(0, 0.1, n_samples)  # High correlation
        
        # Zero variance feature
        feature3 = np.ones(n_samples)
        
        # Imbalanced target with potential leakage
        target = (base_feature > 0).astype(int)  # Perfect correlation with features
        
        # Add missing values
        feature1[::10] = np.nan
        
        return pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'target': target
        })
    
    def test_calculate_ai_readiness_score_good_data(self, engine, sample_dataset, good_quality_data):
        """Test AI readiness score calculation with good quality data."""
        score = engine.calculate_ai_readiness_score(
            sample_dataset, 
            good_quality_data, 
            target_column='target'
        )
        
        assert isinstance(score, AIReadinessScore)
        assert 0.0 <= score.overall_score <= 1.0
        assert score.overall_score > 0.7  # Should be high for good data
        
        # Check all dimension scores are present
        assert 'data_quality' in score.dimensions
        assert 'feature_quality' in score.dimensions
        assert 'bias_fairness' in score.dimensions
        assert 'anomaly_detection' in score.dimensions
        assert 'scalability' in score.dimensions
        
        # Check dimension scores are reasonable
        assert score.data_quality_score > 0.8
        assert score.feature_quality_score > 0.7
        assert score.bias_score > 0.8  # Balanced classes
    
    def test_calculate_ai_readiness_score_poor_data(self, engine, sample_dataset, poor_quality_data):
        """Test AI readiness score calculation with poor quality data."""
        score = engine.calculate_ai_readiness_score(
            sample_dataset, 
            poor_quality_data, 
            target_column='target'
        )
        
        assert isinstance(score, AIReadinessScore)
        assert 0.0 <= score.overall_score <= 1.0
        assert score.overall_score < 0.5  # Should be low for poor data
        
        # Should identify improvement areas
        assert len(score.improvement_areas) > 0
        
        # Feature quality should be low due to correlation and zero variance
        assert score.feature_quality_score < 0.5
    
    def test_feature_correlation_score_high_correlation(self, engine):
        """Test feature correlation score with highly correlated features."""
        # Create highly correlated data
        np.random.seed(42)
        base = np.random.normal(0, 1, 100)
        data = pd.DataFrame({
            'feature1': base,
            'feature2': base + np.random.normal(0, 0.01, 100),  # Very high correlation
            'feature3': base * 2 + np.random.normal(0, 0.01, 100),  # Very high correlation
            'feature4': np.random.normal(0, 1, 100)  # Independent
        })
        
        score = engine._calculate_feature_correlation_score(data)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low due to high correlations
    
    def test_feature_correlation_score_low_correlation(self, engine):
        """Test feature correlation score with uncorrelated features."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.normal(-2, 3, 100),
            'feature4': np.random.uniform(0, 10, 100)
        })
        
        score = engine._calculate_feature_correlation_score(data)
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high for uncorrelated features
    
    def test_target_leakage_score_no_leakage(self, engine):
        """Test target leakage detection with no leakage."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        score = engine._calculate_target_leakage_score(data, 'target')
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high with no leakage
    
    def test_target_leakage_score_with_leakage(self, engine):
        """Test target leakage detection with obvious leakage."""
        np.random.seed(42)
        feature = np.random.normal(0, 1, 100)
        data = pd.DataFrame({
            'feature1': feature,
            'feature2': np.random.normal(0, 1, 100),
            'target': (feature > 0).astype(int)  # Perfect correlation = leakage
        })
        
        score = engine._calculate_target_leakage_score(data, 'target')
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low due to leakage
    
    def test_anomaly_score_clean_data(self, engine):
        """Test anomaly detection with clean data."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.uniform(0, 10, 1000)
        })
        
        score = engine._calculate_anomaly_score(data)
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high for clean data
    
    def test_anomaly_score_with_outliers(self, engine):
        """Test anomaly detection with outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 900)
        outliers = np.random.normal(0, 1, 100) * 10  # Extreme outliers
        
        data = pd.DataFrame({
            'feature1': np.concatenate([normal_data, outliers]),
            'feature2': np.random.normal(5, 2, 1000)
        })
        
        score = engine._calculate_anomaly_score(data)
        assert 0.0 <= score <= 1.0
        assert score < 0.8  # Should be lower due to outliers
    
    def test_feature_importance_score_good_features(self, engine):
        """Test feature importance score with informative features."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.uniform(0, 10, 100),
            'feature3': np.random.exponential(2, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        score = engine._calculate_feature_importance_score(data, 'target')
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high for varied features
    
    def test_feature_importance_score_zero_variance(self, engine):
        """Test feature importance score with zero variance features."""
        data = pd.DataFrame({
            'feature1': np.ones(100),  # Zero variance
            'feature2': np.full(100, 5),  # Zero variance
            'feature3': np.random.normal(0, 1, 100),  # Good variance
            'target': np.random.choice([0, 1], 100)
        })
        
        score = engine._calculate_feature_importance_score(data, 'target')
        assert 0.0 <= score <= 1.0
        assert score < 0.7  # Should be lower due to zero variance features
    
    def test_class_balance_score_balanced(self, engine):
        """Test class balance score with balanced classes."""
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100, p=[0.5, 0.5])
        })
        
        score = engine._calculate_class_balance_score(data, 'target')
        assert 0.0 <= score <= 1.0
        assert score > 0.9  # Should be high for balanced classes
    
    def test_class_balance_score_imbalanced(self, engine):
        """Test class balance score with imbalanced classes."""
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100, p=[0.95, 0.05])  # Highly imbalanced
        })
        
        score = engine._calculate_class_balance_score(data, 'target')
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low for imbalanced classes
    
    def test_class_balance_score_regression_target(self, engine):
        """Test class balance score with regression target (should return 1.0)."""
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)  # Continuous target
        })
        
        score = engine._calculate_class_balance_score(data, 'target')
        assert score == 1.0  # Should return 1.0 for regression tasks
    
    def test_dimensionality_score_good_ratio(self, engine):
        """Test dimensionality score with good feature-to-sample ratio."""
        # 1000 samples, 5 features = 0.005 ratio (very good)
        data = pd.DataFrame(np.random.normal(0, 1, (1000, 5)))
        
        score = engine._calculate_dimensionality_score(data)
        assert 0.0 <= score <= 1.0
        assert score >= 0.9  # Should be high for good ratio
    
    def test_dimensionality_score_poor_ratio(self, engine):
        """Test dimensionality score with poor feature-to-sample ratio."""
        # 50 samples, 100 features = 2.0 ratio (very poor)
        data = pd.DataFrame(np.random.normal(0, 1, (50, 100)))
        
        score = engine._calculate_dimensionality_score(data)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low for poor ratio
    
    def test_scalability_score_large_dataset(self, engine):
        """Test scalability score with large dataset."""
        # Large dataset should score well
        data = pd.DataFrame(np.random.normal(0, 1, (50000, 10)))
        
        score = engine._calculate_scalability_score(data)
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high for large dataset
    
    def test_scalability_score_small_dataset(self, engine):
        """Test scalability score with small dataset."""
        # Small dataset should score poorly
        data = pd.DataFrame(np.random.normal(0, 1, (50, 10)))
        
        score = engine._calculate_scalability_score(data)
        assert 0.0 <= score <= 1.0
        assert score < 0.7  # Should be lower for small dataset
    
    def test_memory_efficiency_calculation(self, engine):
        """Test memory efficiency calculation."""
        # Create data with different memory footprints
        efficient_data = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.random(1000)
        })
        
        score = engine._calculate_memory_efficiency(efficient_data)
        assert 0.0 <= score <= 1.0
    
    def test_improvement_areas_identification(self, engine):
        """Test identification of improvement areas."""
        # Create dimensions with varying scores
        dimensions = {
            'data_quality': DimensionScore('data_quality', 0.4, 0.25),  # Low score
            'feature_quality': DimensionScore('feature_quality', 0.9, 0.25),  # High score
            'bias_fairness': DimensionScore('bias_fairness', 0.6, 0.20),  # Medium score
            'anomaly_detection': DimensionScore('anomaly_detection', 0.3, 0.15),  # Low score
            'scalability': DimensionScore('scalability', 0.8, 0.15)  # Good score
        }
        
        improvement_areas = engine._identify_improvement_areas(dimensions)
        
        # Should identify areas with scores < 0.8
        assert len(improvement_areas) == 3  # data_quality, bias_fairness, anomaly_detection
        
        # Check priorities are assigned correctly
        high_priority_areas = [area for area in improvement_areas if area.priority == 'high']
        assert len(high_priority_areas) == 2  # data_quality and anomaly_detection (< 0.5)
    
    def test_get_improvement_actions(self, engine):
        """Test generation of improvement actions."""
        # Test actions for different dimensions
        data_quality_actions = engine._get_improvement_actions('data_quality', 0.3)
        assert len(data_quality_actions) > 0
        assert any('data cleaning' in action.lower() for action in data_quality_actions)
        
        feature_quality_actions = engine._get_improvement_actions('feature_quality', 0.4)
        assert len(feature_quality_actions) > 0
        assert any('correlation' in action.lower() for action in feature_quality_actions)
        
        bias_fairness_actions = engine._get_improvement_actions('bias_fairness', 0.3)
        assert len(bias_fairness_actions) > 0
        assert any('class imbalance' in action.lower() for action in bias_fairness_actions)
    
    def test_ai_readiness_score_no_target_column(self, engine, sample_dataset, good_quality_data):
        """Test AI readiness score calculation without target column."""
        # Remove target column
        data_no_target = good_quality_data.drop(columns=['target'])
        
        score = engine.calculate_ai_readiness_score(sample_dataset, data_no_target)
        
        assert isinstance(score, AIReadinessScore)
        assert 0.0 <= score.overall_score <= 1.0
        
        # Should still calculate most metrics
        assert score.data_quality_score > 0
        assert score.feature_quality_score > 0
    
    def test_ai_readiness_score_empty_data(self, engine, sample_dataset):
        """Test AI readiness score calculation with empty data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):
            engine.calculate_ai_readiness_score(sample_dataset, empty_data)
    
    def test_ai_readiness_score_single_column(self, engine, sample_dataset):
        """Test AI readiness score calculation with single column."""
        single_col_data = pd.DataFrame({'feature1': np.random.normal(0, 1, 100)})
        
        score = engine.calculate_ai_readiness_score(sample_dataset, single_col_data)
        
        assert isinstance(score, AIReadinessScore)
        assert 0.0 <= score.overall_score <= 1.0
    
    @patch('ai_data_readiness.engines.quality_assessment_engine.IsolationForest')
    def test_anomaly_score_isolation_forest_error(self, mock_isolation_forest, engine):
        """Test anomaly score calculation when IsolationForest fails."""
        mock_isolation_forest.side_effect = Exception("IsolationForest failed")
        
        data = pd.DataFrame({'feature1': np.random.normal(0, 1, 100)})
        score = engine._calculate_anomaly_score(data)
        
        assert score == 0.5  # Should return neutral score on error
    
    def test_target_leakage_score_categorical_target(self, engine):
        """Test target leakage detection with categorical target."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.choice(['X', 'Y', 'Z'], 100),
            'target': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        score = engine._calculate_target_leakage_score(data, 'target')
        assert 0.0 <= score <= 1.0
    
    def test_feature_correlation_score_non_numeric_data(self, engine):
        """Test feature correlation score with non-numeric data."""
        data = pd.DataFrame({
            'category1': np.random.choice(['A', 'B', 'C'], 100),
            'category2': np.random.choice(['X', 'Y', 'Z'], 100),
            'text': ['text_' + str(i) for i in range(100)]
        })
        
        score = engine._calculate_feature_correlation_score(data)
        assert score == 1.0  # Should return 1.0 for non-numeric data
    
    def test_ai_readiness_score_weights_sum_to_one(self, engine, sample_dataset, good_quality_data):
        """Test that dimension weights sum to 1.0."""
        score = engine.calculate_ai_readiness_score(sample_dataset, good_quality_data, 'target')
        
        total_weight = sum(dim.weight for dim in score.dimensions.values())
        assert abs(total_weight - 1.0) < 1e-6  # Should sum to 1.0 (within floating point precision)