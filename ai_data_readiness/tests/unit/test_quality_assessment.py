"""
Unit tests for Quality Assessment Engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.core.exceptions import AIDataReadinessError


class TestQualityAssessmentEngine:
    """Test suite for QualityAssessmentEngine."""
    
    @pytest.fixture
    def quality_engine(self, test_config):
        """Create QualityAssessmentEngine instance for testing."""
        return QualityAssessmentEngine(test_config)
    
    def test_init(self, test_config):
        """Test QualityAssessmentEngine initialization."""
        engine = QualityAssessmentEngine(test_config)
        assert engine.config == test_config
        assert hasattr(engine, 'quality_metrics')
        assert hasattr(engine, 'thresholds')
    
    def test_assess_completeness(self, quality_engine, sample_missing_data):
        """Test completeness assessment."""
        result = quality_engine._assess_completeness(sample_missing_data)
        
        assert 'completeness_score' in result
        assert 'column_completeness' in result
        assert 0 <= result['completeness_score'] <= 1
        
        # Check individual column completeness
        col_completeness = result['column_completeness']
        assert col_completeness['complete_col'] == 1.0  # No missing values
        assert col_completeness['partial_missing'] == 0.8  # 20% missing
        assert col_completeness['mostly_missing'] == 0.1  # 90% missing
        assert col_completeness['all_missing'] == 0.0  # 100% missing
    
    def test_assess_accuracy(self, quality_engine, sample_csv_data):
        """Test accuracy assessment."""
        result = quality_engine._assess_accuracy(sample_csv_data)
        
        assert 'accuracy_score' in result
        assert 'accuracy_issues' in result
        assert 0 <= result['accuracy_score'] <= 1
        
        # Should detect no major accuracy issues in clean data
        assert result['accuracy_score'] > 0.8
    
    def test_assess_consistency(self, quality_engine):
        """Test consistency assessment."""
        # Create data with consistency issues
        inconsistent_data = pd.DataFrame({
            'name': ['John Doe', 'john doe', 'JOHN DOE', 'J. Doe'],
            'email': ['john@email.com', 'john@email.com', 'john@email.com', 'john@email.com'],
            'phone': ['123-456-7890', '(123) 456-7890', '123.456.7890', '1234567890']
        })
        
        result = quality_engine._assess_consistency(inconsistent_data)
        
        assert 'consistency_score' in result
        assert 'consistency_issues' in result
        assert 0 <= result['consistency_score'] <= 1
        
        # Should detect consistency issues
        assert len(result['consistency_issues']) > 0
    
    def test_assess_validity(self, quality_engine):
        """Test validity assessment."""
        # Create data with validity issues
        invalid_data = pd.DataFrame({
            'age': [25, 30, -5, 150, 35],  # Negative age and unrealistic age
            'email': ['valid@email.com', 'invalid-email', 'another@valid.com', '', 'test@test.com'],
            'score': [85, 92, 105, 78, -10]  # Scores outside valid range
        })
        
        validation_rules = {
            'age': {'min': 0, 'max': 120},
            'email': {'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            'score': {'min': 0, 'max': 100}
        }
        
        result = quality_engine._assess_validity(invalid_data, validation_rules)
        
        assert 'validity_score' in result
        assert 'validity_issues' in result
        assert 0 <= result['validity_score'] <= 1
        
        # Should detect validity issues
        assert len(result['validity_issues']) > 0
        validity_issues = result['validity_issues']
        
        # Check specific validity issues
        age_issues = [issue for issue in validity_issues if issue['column'] == 'age']
        assert len(age_issues) > 0
        
        email_issues = [issue for issue in validity_issues if issue['column'] == 'email']
        assert len(email_issues) > 0
        
        score_issues = [issue for issue in validity_issues if issue['column'] == 'score']
        assert len(score_issues) > 0
    
    def test_calculate_overall_quality_score(self, quality_engine):
        """Test overall quality score calculation."""
        dimension_scores = {
            'completeness_score': 0.9,
            'accuracy_score': 0.8,
            'consistency_score': 0.7,
            'validity_score': 0.85
        }
        
        overall_score = quality_engine._calculate_overall_score(dimension_scores)
        
        assert 0 <= overall_score <= 1
        # Should be weighted average of dimension scores
        expected_score = (0.9 + 0.8 + 0.7 + 0.85) / 4
        assert abs(overall_score - expected_score) < 0.01
    
    def test_assess_quality_complete_workflow(self, quality_engine, sample_csv_data):
        """Test complete quality assessment workflow."""
        dataset_id = "test_dataset_123"
        
        # Mock dataset retrieval
        with patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            result = quality_engine.assess_quality(dataset_id)
            
            assert result is not None
            assert 'dataset_id' in result
            assert result['dataset_id'] == dataset_id
            
            # Check all quality dimensions
            assert 'overall_score' in result
            assert 'completeness_score' in result
            assert 'accuracy_score' in result
            assert 'consistency_score' in result
            assert 'validity_score' in result
            
            # Check additional fields
            assert 'issues' in result
            assert 'recommendations' in result
            assert 'generated_at' in result
            
            # All scores should be between 0 and 1
            for score_key in ['overall_score', 'completeness_score', 'accuracy_score', 
                            'consistency_score', 'validity_score']:
                assert 0 <= result[score_key] <= 1
    
    def test_detect_outliers(self, quality_engine):
        """Test outlier detection."""
        # Create data with outliers
        data_with_outliers = pd.DataFrame({
            'normal_values': np.random.normal(50, 10, 100).tolist() + [200, -100],  # Add outliers
            'another_col': np.random.normal(0, 1, 102)
        })
        
        outliers = quality_engine._detect_outliers(data_with_outliers)
        
        assert 'outliers_detected' in outliers
        assert 'outlier_indices' in outliers
        assert 'outlier_columns' in outliers
        
        # Should detect the extreme outliers
        assert outliers['outliers_detected'] is True
        assert len(outliers['outlier_indices']) > 0
    
    def test_assess_data_distribution(self, quality_engine, sample_csv_data):
        """Test data distribution assessment."""
        distribution_analysis = quality_engine._assess_data_distribution(sample_csv_data)
        
        assert 'distribution_analysis' in distribution_analysis
        assert 'skewness' in distribution_analysis
        assert 'kurtosis' in distribution_analysis
        
        # Check that numerical columns are analyzed
        skewness = distribution_analysis['skewness']
        assert 'age' in skewness
        assert 'income' in skewness
        assert 'score' in skewness
    
    def test_calculate_ai_readiness_score(self, quality_engine, sample_csv_data):
        """Test AI readiness score calculation."""
        dataset_id = "test_dataset_123"
        
        with patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            ai_score = quality_engine.calculate_ai_readiness_score(dataset_id)
            
            assert 'overall_score' in ai_score
            assert 'data_quality_score' in ai_score
            assert 'feature_quality_score' in ai_score
            assert 'bias_score' in ai_score
            assert 'compliance_score' in ai_score
            assert 'scalability_score' in ai_score
            
            # Check dimension details
            assert 'dimensions' in ai_score
            dimensions = ai_score['dimensions']
            
            for dimension in ['data_quality', 'feature_quality', 'bias', 'compliance', 'scalability']:
                assert dimension in dimensions
                assert 'score' in dimensions[dimension]
                assert 'details' in dimensions[dimension]
    
    def test_detect_anomalies(self, quality_engine):
        """Test anomaly detection."""
        # Create data with anomalies
        normal_data = np.random.normal(50, 10, 95)
        anomalous_data = [200, -100, 300, -200, 400]  # Clear anomalies
        data_with_anomalies = pd.DataFrame({
            'values': np.concatenate([normal_data, anomalous_data])
        })
        
        dataset_id = "test_dataset_123"
        
        with patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = data_with_anomalies
            
            anomaly_report = quality_engine.detect_anomalies(dataset_id)
            
            assert 'dataset_id' in anomaly_report
            assert 'anomalies_detected' in anomaly_report
            assert 'anomaly_count' in anomaly_report
            assert 'anomaly_details' in anomaly_report
            
            # Should detect the anomalies
            assert anomaly_report['anomalies_detected'] is True
            assert anomaly_report['anomaly_count'] > 0
    
    def test_recommend_improvements(self, quality_engine):
        """Test improvement recommendations."""
        # Create a quality report with issues
        quality_report = {
            'overall_score': 0.6,
            'completeness_score': 0.4,  # Low completeness
            'accuracy_score': 0.8,
            'consistency_score': 0.5,   # Low consistency
            'validity_score': 0.7,
            'issues': [
                {'type': 'completeness', 'column': 'age', 'severity': 'high'},
                {'type': 'consistency', 'column': 'name', 'severity': 'medium'},
                {'type': 'validity', 'column': 'email', 'severity': 'low'}
            ]
        }
        
        recommendations = quality_engine.recommend_improvements(quality_report)
        
        assert 'recommendations' in recommendations
        assert 'priority_actions' in recommendations
        assert 'estimated_impact' in recommendations
        
        recs = recommendations['recommendations']
        assert len(recs) > 0
        
        # Should prioritize high-severity issues
        high_priority_recs = [r for r in recs if r['priority'] == 'high']
        assert len(high_priority_recs) > 0
        
        # Check recommendation structure
        for rec in recs:
            assert 'issue_type' in rec
            assert 'description' in rec
            assert 'action' in rec
            assert 'priority' in rec
            assert 'estimated_effort' in rec
    
    def test_performance_with_large_dataset(self, quality_engine, performance_test_data, performance_timer):
        """Test performance with large dataset."""
        dataset_id = "large_test_dataset"
        
        with patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = performance_test_data
            
            with performance_timer:
                result = quality_engine.assess_quality(dataset_id)
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert performance_timer.duration < 10.0  # 10 seconds max
            assert result is not None
            assert 'overall_score' in result
    
    def test_error_handling_empty_dataset(self, quality_engine):
        """Test error handling for empty datasets."""
        empty_data = pd.DataFrame()
        dataset_id = "empty_dataset"
        
        with patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = empty_data
            
            with pytest.raises(AIDataReadinessError):
                quality_engine.assess_quality(dataset_id)
    
    def test_error_handling_invalid_dataset_id(self, quality_engine):
        """Test error handling for invalid dataset IDs."""
        with patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.side_effect = FileNotFoundError("Dataset not found")
            
            with pytest.raises(AIDataReadinessError):
                quality_engine.assess_quality("nonexistent_dataset")
    
    def test_custom_quality_thresholds(self, test_config):
        """Test custom quality thresholds."""
        custom_thresholds = {
            'completeness_threshold': 0.95,
            'accuracy_threshold': 0.90,
            'consistency_threshold': 0.85,
            'validity_threshold': 0.90
        }
        
        test_config.QUALITY_THRESHOLDS = custom_thresholds
        engine = QualityAssessmentEngine(test_config)
        
        assert engine.thresholds == custom_thresholds
        
        # Test that thresholds are used in assessment
        dimension_scores = {
            'completeness_score': 0.92,  # Below custom threshold
            'accuracy_score': 0.95,
            'consistency_score': 0.80,   # Below custom threshold
            'validity_score': 0.95
        }
        
        issues = engine._identify_quality_issues(dimension_scores)
        
        # Should identify issues based on custom thresholds
        completeness_issues = [i for i in issues if i['dimension'] == 'completeness']
        consistency_issues = [i for i in issues if i['dimension'] == 'consistency']
        
        assert len(completeness_issues) > 0
        assert len(consistency_issues) > 0