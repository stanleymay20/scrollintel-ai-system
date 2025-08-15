"""Tests for AI-specific quality metrics and scoring functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.ai_quality_metrics import AIQualityMetrics
from ai_data_readiness.models.base_models import (
    Dataset, DatasetMetadata, Schema, QualityDimension,
    AIReadinessScore, DimensionScore, ImprovementArea
)
from ai_data_readiness.core.config import Config


class TestAISpecificQualityMetrics:
    """Test suite for AI-specific quality metrics and scoring."""
    
    @pytest.fixture
    def engine(self):
        """Create a quality assessment engine for testing."""
        config = Config()
        return QualityAssessmentEngine(config)
    
    @pytest.fixture
    def ai_metrics(self):
        """Create an AI quality metrics engine for testing."""
        return AIQualityMetrics()
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        schema = Schema(
            columns={
                'feature1': 'float',
                'feature2': 'integer',
                'category': 'categorical',
                'target': 'integer'
            }
        )
        
        metadata = DatasetMetadata(
            name="test_dataset",
            description="Test dataset for AI metrics",
            row_count=1000,
            column_count=4
        )
        
        return Dataset(
            id="test_dataset_001",
            name="Test Dataset",
            schema=schema,
            metadata=metadata
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.randint(1, 100, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.7, 0.2, 0.1]),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Add some missing values
        data.loc[np.random.choice(data.index, 50, replace=False), 'feature1'] = np.nan
        
        return data
    
    def test_calculate_ai_readiness_score(self, engine, ai_metrics, sample_dataset, sample_data):
        """Test AI readiness score calculation."""
        # Get base quality report first
        quality_report = engine.assess_quality(sample_dataset, sample_data)
        ai_score = ai_metrics.calculate_ai_readiness_score(sample_dataset, sample_data, quality_report)
        
        assert isinstance(ai_score, AIReadinessScore)
        assert 0.0 <= ai_score.overall_score <= 1.0
        assert 0.0 <= ai_score.data_quality_score <= 1.0
        assert 0.0 <= ai_score.feature_quality_score <= 1.0
        assert 0.0 <= ai_score.bias_score <= 1.0
        assert 0.0 <= ai_score.compliance_score <= 1.0
        assert 0.0 <= ai_score.scalability_score <= 1.0
        
        # Check dimensions
        assert len(ai_score.dimensions) == 5
        assert 'data_quality' in ai_score.dimensions
        assert 'feature_quality' in ai_score.dimensions
        assert 'bias_fairness' in ai_score.dimensions
        assert 'compliance' in ai_score.dimensions
        assert 'scalability' in ai_score.dimensions
        
        # Check improvement areas
        assert isinstance(ai_score.improvement_areas, list)
    
    def test_feature_quality_score_calculation(self, engine, sample_data):
        """Test feature quality score calculation."""
        score = engine._calculate_feature_quality_score(sample_data)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_analyze_feature_correlations(self, engine):
        """Test feature correlation analysis."""
        # Create data with high correlation
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        data['feature3'] = data['feature1'] * 0.95 + np.random.normal(0, 0.1, 100)  # High correlation
        
        score = engine._analyze_feature_correlations(data)
        
        assert 0.0 <= score <= 1.0
        # Should detect high correlation and reduce score
        assert score < 1.0
    
    def test_detect_target_leakage(self, engine):
        """Test target leakage detection."""
        # Create data with potential target leakage
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'target_derived': np.random.normal(0, 1, 100),  # Suspicious name
            'prediction_score': np.random.normal(0, 1, 100)  # Suspicious name
        })
        
        score = engine._detect_target_leakage(data)
        
        assert 0.0 <= score <= 1.0
        # Should detect suspicious columns and reduce score
        assert score < 1.0
    
    def test_analyze_feature_distributions(self, engine):
        """Test feature distribution analysis."""
        # Create data with various distribution characteristics
        data = pd.DataFrame({
            'normal_feature': np.random.normal(0, 1, 100),
            'skewed_feature': np.random.exponential(2, 100),  # Highly skewed
            'constant_feature': np.ones(100),  # No variance
            'categorical_good': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_bad': np.random.choice([f'cat_{i}' for i in range(100)], 100)  # Too many categories
        })
        
        score = engine._analyze_feature_distributions(data)
        
        assert 0.0 <= score <= 1.0
    
    def test_analyze_feature_variance(self, engine):
        """Test feature variance analysis."""
        # Create data with low and high variance features
        data = pd.DataFrame({
            'high_variance': np.random.normal(0, 10, 100),
            'low_variance': np.random.normal(100, 0.01, 100),  # Very low variance
            'zero_variance': np.ones(100)  # Zero variance
        })
        
        score = engine._analyze_feature_variance(data)
        
        assert 0.0 <= score <= 1.0
        # Should detect low/zero variance features and reduce score
        assert score < 1.0
    
    def test_calculate_bias_score(self, engine):
        """Test bias score calculation."""
        # Create data with class imbalance
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'imbalanced_category': np.random.choice(['A', 'B'], 1000, p=[0.95, 0.05])  # Highly imbalanced
        })
        
        score = engine._calculate_bias_score(data)
        
        assert 0.0 <= score <= 1.0
        # Should detect imbalance and reduce score
        assert score < 1.0
    
    def test_calculate_compliance_score(self, engine, sample_dataset):
        """Test compliance score calculation."""
        # Create data with potential PII
        data = pd.DataFrame({
            'email_address': ['user1@example.com', 'user2@example.com'] * 50,
            'phone_number': ['555-123-4567', '555-987-6543'] * 50,
            'normal_feature': np.random.normal(0, 1, 100)
        })
        
        score = engine._calculate_compliance_score(data, sample_dataset.schema)
        
        assert 0.0 <= score <= 1.0
        # Should detect PII and reduce score
        assert score < 1.0
    
    def test_detect_pii_compliance(self, engine):
        """Test PII detection for compliance."""
        # Create data with various PII patterns
        data = pd.DataFrame({
            'email': ['user@example.com'] * 50,
            'phone': ['555-123-4567'] * 50,
            'user_name': ['John Doe'] * 50,
            'safe_feature': np.random.normal(0, 1, 50)
        })
        
        score = engine._detect_pii_compliance(data)
        
        assert 0.0 <= score <= 1.0
        # Should detect multiple PII columns and significantly reduce score
        assert score < 0.5
    
    def test_calculate_scalability_score(self, engine):
        """Test scalability score calculation."""
        # Create data with different scalability characteristics
        small_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, 10000) for i in range(50)
        })
        
        small_score = engine._calculate_scalability_score(small_data)
        large_score = engine._calculate_scalability_score(large_data)
        
        assert 0.0 <= small_score <= 1.0
        assert 0.0 <= large_score <= 1.0
        
        # Large data should have better scalability score (within reasonable limits)
        assert large_score > small_score
    
    def test_detect_anomalies(self, engine):
        """Test anomaly detection functionality."""
        # Create data with anomalies
        normal_data = np.random.normal(0, 1, 95)
        outliers = np.array([10, -10, 15, -15, 20])  # Clear outliers
        
        data = pd.DataFrame({
            'feature1': np.concatenate([normal_data, outliers]),
            'category': ['normal'] * 95 + ['rare'] * 5
        })
        
        result = engine.detect_anomalies(data)
        
        assert isinstance(result, dict)
        assert 'anomalies' in result
        assert 'anomaly_score' in result
        assert 'total_anomalies' in result
        assert 'anomalous_rows' in result
        
        assert isinstance(result['anomalies'], list)
        assert 0.0 <= result['anomaly_score'] <= 1.0
        assert result['total_anomalies'] >= 0
        
        # Should detect some anomalies
        assert len(result['anomalies']) > 0
    
    def test_get_feature_quality_details(self, engine, sample_data):
        """Test feature quality details extraction."""
        details = engine._get_feature_quality_details(sample_data)
        
        assert isinstance(details, dict)
        assert 'correlation_analysis' in details
        assert 'target_leakage_score' in details
        assert 'distribution_score' in details
        assert 'variance_score' in details
        assert 'numeric_features' in details
        assert 'categorical_features' in details
        
        # Check data types
        assert isinstance(details['correlation_analysis'], float)
        assert isinstance(details['target_leakage_score'], float)
        assert isinstance(details['distribution_score'], float)
        assert isinstance(details['variance_score'], float)
        assert isinstance(details['numeric_features'], int)
        assert isinstance(details['categorical_features'], int)
    
    def test_get_bias_details(self, engine):
        """Test bias details extraction."""
        # Create data with known imbalance
        data = pd.DataFrame({
            'balanced_cat': np.random.choice(['A', 'B'], 100),
            'imbalanced_cat': np.random.choice(['X', 'Y'], 100, p=[0.9, 0.1])
        })
        
        details = engine._get_bias_details(data)
        
        assert isinstance(details, dict)
        assert 'overall_bias_score' in details
        assert 'class_imbalance' in details
        assert 'categorical_columns_analyzed' in details
        
        # Check imbalance detection
        assert 'imbalanced_cat' in details['class_imbalance']
        imbalance_info = details['class_imbalance']['imbalanced_cat']
        assert imbalance_info['imbalance_ratio'] > 5  # Should detect high imbalance
    
    def test_identify_improvement_areas(self, engine):
        """Test improvement area identification."""
        # Create dimensions with low scores
        dimensions = {
            'data_quality': DimensionScore('data_quality', 0.4, 0.3),
            'feature_quality': DimensionScore('feature_quality', 0.6, 0.25),
            'bias_fairness': DimensionScore('bias_fairness', 0.9, 0.2),
            'compliance': DimensionScore('compliance', 0.3, 0.15),
            'scalability': DimensionScore('scalability', 0.8, 0.1)
        }
        
        improvement_areas = engine._identify_improvement_areas(dimensions)
        
        assert isinstance(improvement_areas, list)
        
        # Should identify areas with scores < 0.8
        low_score_areas = [area.area for area in improvement_areas]
        assert 'data_quality' in low_score_areas
        assert 'feature_quality' in low_score_areas
        assert 'compliance' in low_score_areas
        
        # High-scoring areas should not be included
        assert 'bias_fairness' not in low_score_areas
        
        # Check improvement area structure
        for area in improvement_areas:
            assert isinstance(area, ImprovementArea)
            assert area.current_score < 0.8
            assert area.target_score > area.current_score
            assert area.priority in ['low', 'medium', 'high']
            assert isinstance(area.actions, list)
            assert len(area.actions) > 0
    
    def test_get_improvement_actions(self, engine):
        """Test improvement action generation."""
        # Test different dimensions and scores
        test_cases = [
            ('data_quality', 0.3),
            ('feature_quality', 0.6),
            ('bias_fairness', 0.4),
            ('compliance', 0.5),
            ('scalability', 0.7)
        ]
        
        for dimension, score in test_cases:
            actions = engine._get_improvement_actions(dimension, score)
            
            assert isinstance(actions, list)
            assert len(actions) > 0
            
            # All actions should be strings
            assert all(isinstance(action, str) for action in actions)
            
            # Actions should be relevant to the dimension
            if dimension == 'data_quality':
                assert any('missing' in action.lower() or 'accuracy' in action.lower() 
                          for action in actions)
            elif dimension == 'feature_quality':
                assert any('correlation' in action.lower() or 'feature' in action.lower() 
                          for action in actions)
            elif dimension == 'bias_fairness':
                assert any('bias' in action.lower() or 'fairness' in action.lower() 
                          for action in actions)
            elif dimension == 'compliance':
                assert any('pii' in action.lower() or 'compliance' in action.lower() 
                          for action in actions)
            elif dimension == 'scalability':
                assert any('memory' in action.lower() or 'performance' in action.lower() 
                          for action in actions)
    
    def test_empty_data_handling(self, engine, sample_dataset):
        """Test handling of empty datasets."""
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        ai_score = engine.calculate_ai_readiness_score(sample_dataset, empty_data)
        
        assert isinstance(ai_score, AIReadinessScore)
        assert ai_score.overall_score == 0.0
        
        # Individual scores should also be 0 or appropriate defaults
        assert ai_score.data_quality_score == 0.0
        assert ai_score.feature_quality_score == 0.0
        
        # Anomaly detection should also handle empty data
        anomaly_result = engine.detect_anomalies(empty_data)
        assert anomaly_result['anomaly_score'] == 1.0
        assert len(anomaly_result['anomalies']) == 0
    
    def test_single_column_data(self, engine, sample_dataset):
        """Test handling of single-column datasets."""
        single_col_data = pd.DataFrame({
            'single_feature': np.random.normal(0, 1, 100)
        })
        
        ai_score = engine.calculate_ai_readiness_score(sample_dataset, single_col_data)
        
        assert isinstance(ai_score, AIReadinessScore)
        assert 0.0 <= ai_score.overall_score <= 1.0
        
        # Feature correlation should handle single column
        correlation_score = engine._analyze_feature_correlations(single_col_data)
        assert correlation_score == 1.0  # No correlation issues with single feature
    
    def test_all_missing_data(self, engine, sample_dataset):
        """Test handling of data with all missing values."""
        missing_data = pd.DataFrame({
            'feature1': [np.nan] * 100,
            'feature2': [np.nan] * 100
        })
        
        ai_score = engine.calculate_ai_readiness_score(sample_dataset, missing_data)
        
        assert isinstance(ai_score, AIReadinessScore)
        # Should have very low scores due to complete missing data
        assert ai_score.data_quality_score < 0.1
    
    @pytest.mark.parametrize("contamination", [0.05, 0.1, 0.2])
    def test_anomaly_detection_sensitivity(self, engine, contamination):
        """Test anomaly detection with different contamination levels."""
        # Create data with known outliers
        normal_data = np.random.normal(0, 1, 100)
        outlier_count = int(100 * contamination)
        outliers = np.random.normal(10, 1, outlier_count)  # Clear outliers
        
        data = pd.DataFrame({
            'feature': np.concatenate([normal_data, outliers])
        })
        
        result = engine.detect_anomalies(data)
        
        # Should detect some anomalies
        assert len(result['anomalies']) > 0
        assert result['anomaly_score'] < 1.0
    
    def test_memory_efficiency_assessment(self, engine):
        """Test memory efficiency assessment."""
        # Create data with different memory characteristics
        efficient_data = pd.DataFrame({
            'int_feature': np.random.randint(0, 100, 1000).astype('int16'),
            'category': pd.Categorical(np.random.choice(['A', 'B', 'C'], 1000))
        })
        
        inefficient_data = pd.DataFrame({
            'string_feature': [f'very_long_string_value_{i}' for i in range(1000)],
            'large_int': np.random.randint(0, 1000000, 1000).astype('int64')
        })
        
        efficient_score = engine._assess_memory_efficiency(efficient_data)
        inefficient_score = engine._assess_memory_efficiency(inefficient_data)
        
        assert 0.0 <= efficient_score <= 1.0
        assert 0.0 <= inefficient_score <= 1.0
        
        # Efficient data should score higher
        assert efficient_score >= inefficient_score


if __name__ == "__main__":
    pytest.main([__file__])