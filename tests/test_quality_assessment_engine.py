"""Unit tests for the Quality Assessment Engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine, QualityMetrics
from ai_data_readiness.models.base_models import (
    Dataset, DatasetMetadata, Schema, QualityDimension, QualityReport
)
from ai_data_readiness.core.config import Config


class TestQualityAssessmentEngine:
    """Test suite for Quality Assessment Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a quality assessment engine instance."""
        config = Config()
        return QualityAssessmentEngine(config)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        schema = Schema(
            columns={
                'id': 'integer',
                'name': 'string',
                'email': 'string',
                'age': 'integer',
                'salary': 'float',
                'is_active': 'boolean',
                'created_at': 'datetime'
            },
            primary_key='id'
        )
        
        metadata = DatasetMetadata(
            name="test_dataset",
            description="Test dataset for quality assessment",
            row_count=100,
            column_count=7
        )
        
        return Dataset(
            id="test-dataset-1",
            name="Test Dataset",
            schema=schema,
            metadata=metadata
        )
    
    @pytest.fixture
    def perfect_data(self):
        """Create perfect quality data for testing."""
        np.random.seed(42)
        # Use recent dates for better timeliness score
        recent_start = datetime.now() - timedelta(days=30)
        return pd.DataFrame({
            'id': range(1, 101),
            'name': [f'USER_{i}' for i in range(1, 101)],  # Consistent case
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'age': np.random.randint(18, 65, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'is_active': np.random.choice([True, False], 100),
            'created_at': pd.date_range(recent_start, periods=100, freq='D')
        })
    
    @pytest.fixture
    def poor_quality_data(self):
        """Create poor quality data for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1, 2, None, 4, 5, 5, 7, 8, 9, 10],  # Missing value and duplicate
            'name': ['John', '', 'JANE', 'bob', None, 'Alice', 'charlie', 'DAVID', 'eve', 'Frank'],  # Mixed case, empty, missing
            'email': ['john@email', 'invalid-email', 'jane@example.com', 'bob@test.com', None, 'alice@example.com', 'charlie@test', 'david@example.com', 'eve@test.com', 'frank@example.com'],  # Invalid formats
            'age': [25, -5, 150, 30, None, 35, 40, 999, 45, 50],  # Negative, unrealistic values
            'salary': [50000, None, 1000000, 60000, 55000, None, 70000, -10000, 80000, 90000],  # Missing, extreme values
            'is_active': [True, 'yes', False, 1, None, True, 'no', False, True, 0],  # Mixed types
            'created_at': ['2023-01-01', None, '2025-12-31', '2023-03-01', '2023-04-01', 'invalid-date', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01']  # Future date, invalid format
        })
        return data
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        config = Config()
        engine = QualityAssessmentEngine(config)
        
        assert engine.config == config
        assert len(engine.dimension_weights) == 6
        assert sum(engine.dimension_weights.values()) == 1.0
    
    def test_assess_quality_perfect_data(self, engine, sample_dataset, perfect_data):
        """Test quality assessment with perfect data."""
        report = engine.assess_quality(sample_dataset, perfect_data)
        
        assert isinstance(report, QualityReport)
        assert report.dataset_id == sample_dataset.id
        assert report.overall_score > 0.8  # Should be high for perfect data (adjusted for timeliness)
        assert report.completeness_score == 1.0  # No missing values
        assert report.accuracy_score == 1.0  # All data types match
        assert report.validity_score == 1.0  # No outliers in this data
        assert report.uniqueness_score == 1.0  # No duplicates
        assert isinstance(report.generated_at, datetime)
    
    def test_assess_quality_poor_data(self, engine, sample_dataset, poor_quality_data):
        """Test quality assessment with poor quality data."""
        report = engine.assess_quality(sample_dataset, poor_quality_data)
        
        assert isinstance(report, QualityReport)
        assert report.dataset_id == sample_dataset.id
        assert report.overall_score < 0.9  # Should be lower for poor data
        assert report.completeness_score < 1.0  # Has missing values
        assert len(report.issues) > 0  # Should identify issues
        assert len(report.recommendations) > 0  # Should provide recommendations
    
    def test_calculate_completeness_perfect(self, engine, perfect_data):
        """Test completeness calculation with perfect data."""
        metrics = engine._calculate_quality_metrics(perfect_data, None)
        assert metrics.completeness == 1.0
    
    def test_calculate_completeness_missing_data(self, engine):
        """Test completeness calculation with missing data."""
        data = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [1, None, 3, None, 5]
        })
        
        metrics = engine._calculate_quality_metrics(data, None)
        # 3 missing values out of 10 total cells = 0.7 completeness
        assert metrics.completeness == 0.7
    
    def test_calculate_completeness_empty_data(self, engine):
        """Test completeness calculation with empty data."""
        data = pd.DataFrame()
        metrics = engine._calculate_quality_metrics(data, None)
        assert metrics.completeness == 0.0
    
    def test_calculate_accuracy_with_schema(self, engine, sample_dataset):
        """Test accuracy calculation with schema validation."""
        data = pd.DataFrame({
            'id': [1, 2, 'invalid', 4, 5],  # One invalid integer
            'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],  # All valid strings
            'age': [25, 30, 35, 'invalid', 45]  # One invalid integer
        })
        
        schema = Schema(columns={'id': 'integer', 'name': 'string', 'age': 'integer'})
        metrics = engine._calculate_quality_metrics(data, schema)
        
        # Should detect type mismatches
        assert metrics.accuracy < 1.0
    
    def test_value_matches_type(self, engine):
        """Test type matching validation."""
        assert engine._value_matches_type(42, 'integer') == True
        assert engine._value_matches_type('42', 'integer') == True
        assert engine._value_matches_type('abc', 'integer') == False
        
        assert engine._value_matches_type(3.14, 'float') == True
        assert engine._value_matches_type('3.14', 'float') == True
        assert engine._value_matches_type('abc', 'float') == False
        
        assert engine._value_matches_type('hello', 'string') == True
        assert engine._value_matches_type(123, 'string') == False
        
        assert engine._value_matches_type(True, 'boolean') == True
        assert engine._value_matches_type('true', 'boolean') == True
        assert engine._value_matches_type('1', 'boolean') == True
        assert engine._value_matches_type('maybe', 'boolean') == False
    
    def test_calculate_consistency_format(self, engine):
        """Test format consistency calculation."""
        # Consistent case formatting
        consistent_data = pd.DataFrame({
            'names': ['JOHN', 'JANE', 'BOB', 'ALICE']
        })
        
        # Inconsistent case formatting
        inconsistent_data = pd.DataFrame({
            'names': ['John', 'JANE', 'bob', 'Alice']
        })
        
        consistent_metrics = engine._calculate_quality_metrics(consistent_data, None)
        inconsistent_metrics = engine._calculate_quality_metrics(inconsistent_data, None)
        
        assert consistent_metrics.consistency > inconsistent_metrics.consistency
    
    def test_calculate_validity_outliers(self, engine):
        """Test validity calculation with outliers."""
        # Data with outliers
        data_with_outliers = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 1000]  # 1000 is an outlier
        })
        
        # Data without outliers
        data_without_outliers = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6]
        })
        
        outlier_metrics = engine._calculate_quality_metrics(data_with_outliers, None)
        normal_metrics = engine._calculate_quality_metrics(data_without_outliers, None)
        
        assert normal_metrics.validity > outlier_metrics.validity
    
    def test_calculate_uniqueness(self, engine):
        """Test uniqueness calculation."""
        # Data with duplicates
        duplicate_data = pd.DataFrame({
            'col1': [1, 2, 3, 1, 2],
            'col2': ['a', 'b', 'c', 'a', 'b']
        })
        
        # Unique data
        unique_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        duplicate_metrics = engine._calculate_quality_metrics(duplicate_data, None)
        unique_metrics = engine._calculate_quality_metrics(unique_data, None)
        
        assert unique_metrics.uniqueness > duplicate_metrics.uniqueness
        assert unique_metrics.uniqueness == 1.0
        assert duplicate_metrics.uniqueness == 0.6  # 3 unique out of 5 total
    
    def test_calculate_timeliness(self, engine):
        """Test timeliness calculation."""
        # Recent data
        recent_data = pd.DataFrame({
            'created_at': [datetime.now() - timedelta(hours=1)]
        })
        
        # Old data
        old_data = pd.DataFrame({
            'created_at': [datetime.now() - timedelta(days=100)]
        })
        
        recent_metrics = engine._calculate_quality_metrics(recent_data, None)
        old_metrics = engine._calculate_quality_metrics(old_data, None)
        
        assert recent_metrics.timeliness > old_metrics.timeliness
    
    def test_identify_completeness_issues(self, engine, poor_quality_data):
        """Test identification of completeness issues."""
        metrics = engine._calculate_quality_metrics(poor_quality_data, None)
        issues = engine._identify_quality_issues(poor_quality_data, metrics, None)
        
        completeness_issues = [issue for issue in issues if issue.dimension == QualityDimension.COMPLETENESS]
        assert len(completeness_issues) > 0
        
        issue = completeness_issues[0]
        assert "Missing values" in issue.description
        assert len(issue.affected_columns) > 0
    
    def test_identify_accuracy_issues(self, engine, sample_dataset):
        """Test identification of accuracy issues."""
        # Create data with more significant type mismatches (>10% threshold)
        data = pd.DataFrame({
            'id': [1, 2, 'invalid', 'bad', 'wrong', 'error', 7, 8, 9, 10],  # 40% invalid
            'age': [25, 30, 'invalid', 'bad', 45, 50, 55, 60, 65, 70]  # 20% invalid
        })
        
        metrics = engine._calculate_quality_metrics(data, sample_dataset.schema)
        issues = engine._identify_quality_issues(data, metrics, sample_dataset.schema)
        
        accuracy_issues = [issue for issue in issues if issue.dimension == QualityDimension.ACCURACY]
        assert len(accuracy_issues) > 0
    
    def test_generate_recommendations(self, engine, poor_quality_data):
        """Test recommendation generation."""
        metrics = engine._calculate_quality_metrics(poor_quality_data, None)
        issues = engine._identify_quality_issues(poor_quality_data, metrics, None)
        recommendations = engine._generate_recommendations(issues, metrics)
        
        assert len(recommendations) > 0
        
        # Check that recommendations have required fields
        for rec in recommendations:
            assert rec.type is not None
            assert rec.priority is not None
            assert rec.description is not None
            assert rec.implementation is not None
            assert 0 <= rec.estimated_impact <= 1
    
    def test_overall_score_calculation(self, engine):
        """Test overall score calculation with weighted dimensions."""
        metrics = QualityMetrics(
            completeness=0.8,
            accuracy=0.9,
            consistency=0.7,
            validity=0.85,
            uniqueness=0.95,
            timeliness=0.6
        )
        
        overall_score = engine._calculate_overall_score(metrics)
        
        # Should be weighted average
        expected_score = (
            0.8 * 0.25 +   # completeness
            0.9 * 0.20 +   # accuracy
            0.7 * 0.20 +   # consistency
            0.85 * 0.20 +  # validity
            0.95 * 0.10 +  # uniqueness
            0.6 * 0.05     # timeliness
        )
        
        assert abs(overall_score - expected_score) < 0.001
    
    def test_empty_dataframe_handling(self, engine, sample_dataset):
        """Test handling of empty dataframes."""
        empty_data = pd.DataFrame()
        
        report = engine.assess_quality(sample_dataset, empty_data)
        
        assert report.overall_score == 0.0
        assert report.completeness_score == 0.0
        assert len(report.issues) == 0
    
    def test_single_row_data(self, engine, sample_dataset):
        """Test handling of single row data."""
        single_row_data = pd.DataFrame({
            'id': [1],
            'name': ['John'],
            'email': ['john@example.com']
        })
        
        report = engine.assess_quality(sample_dataset, single_row_data)
        
        assert isinstance(report, QualityReport)
        assert report.completeness_score == 1.0  # No missing values
        assert report.uniqueness_score == 1.0    # Single row is unique
    
    def test_all_null_column(self, engine, sample_dataset):
        """Test handling of columns with all null values."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': [None, None, None],  # All null
            'email': ['a@b.com', 'c@d.com', 'e@f.com']
        })
        
        report = engine.assess_quality(sample_dataset, data)
        
        assert report.completeness_score < 1.0
        completeness_issues = [issue for issue in report.issues if issue.dimension == QualityDimension.COMPLETENESS]
        assert len(completeness_issues) > 0
    
    def test_numeric_column_statistics(self, engine):
        """Test statistical analysis of numeric columns."""
        data = pd.DataFrame({
            'normal_values': [1, 2, 3, 4, 5],
            'with_outliers': [1, 2, 3, 4, 100]  # 100 is an outlier
        })
        
        metrics = engine._calculate_quality_metrics(data, None)
        
        # Should detect validity issues due to outliers
        assert metrics.validity < 1.0
    
    def test_date_column_parsing(self, engine):
        """Test parsing and validation of date columns."""
        data = pd.DataFrame({
            'valid_dates': pd.date_range('2023-01-01', periods=5),
            'mixed_dates': ['2023-01-01', '2023-02-01', 'invalid', '2023-04-01', None]
        })
        
        metrics = engine._calculate_quality_metrics(data, None)
        
        # Should handle mixed date formats
        assert 0 <= metrics.timeliness <= 1
        assert 0 <= metrics.completeness <= 1
    
    @pytest.mark.parametrize("dimension,weight", [
        (QualityDimension.COMPLETENESS, 0.25),
        (QualityDimension.ACCURACY, 0.20),
        (QualityDimension.CONSISTENCY, 0.20),
        (QualityDimension.VALIDITY, 0.20),
        (QualityDimension.UNIQUENESS, 0.10),
        (QualityDimension.TIMELINESS, 0.05)
    ])
    def test_dimension_weights(self, engine, dimension, weight):
        """Test that dimension weights are correctly configured."""
        assert engine.dimension_weights[dimension] == weight
    
    def test_error_handling_invalid_data(self, engine, sample_dataset):
        """Test error handling with invalid data types."""
        # This should not crash the engine
        invalid_data = "not a dataframe"
        
        with pytest.raises(Exception):
            engine.assess_quality(sample_dataset, invalid_data)
    
    def test_configuration_thresholds(self, engine):
        """Test that configuration thresholds are used correctly."""
        config = engine.config
        
        # Verify thresholds are within valid range
        assert 0 <= config.quality.completeness_threshold <= 1
        assert 0 <= config.quality.accuracy_threshold <= 1
        assert 0 <= config.quality.consistency_threshold <= 1
        assert 0 <= config.quality.validity_threshold <= 1
    
    def test_quality_report_structure(self, engine, sample_dataset, perfect_data):
        """Test that quality report has correct structure."""
        report = engine.assess_quality(sample_dataset, perfect_data)
        
        # Verify all required fields are present
        assert hasattr(report, 'dataset_id')
        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'completeness_score')
        assert hasattr(report, 'accuracy_score')
        assert hasattr(report, 'consistency_score')
        assert hasattr(report, 'validity_score')
        assert hasattr(report, 'uniqueness_score')
        assert hasattr(report, 'timeliness_score')
        assert hasattr(report, 'issues')
        assert hasattr(report, 'recommendations')
        assert hasattr(report, 'generated_at')
        
        # Verify score ranges
        assert 0 <= report.overall_score <= 1
        assert 0 <= report.completeness_score <= 1
        assert 0 <= report.accuracy_score <= 1
        assert 0 <= report.consistency_score <= 1
        assert 0 <= report.validity_score <= 1
        assert 0 <= report.uniqueness_score <= 1
        assert 0 <= report.timeliness_score <= 1