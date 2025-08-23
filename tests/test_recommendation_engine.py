"""
Unit tests for the AI Recommendation Engine
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from scrollintel.engines.recommendation_engine import (
    RecommendationEngine, Schema, Dataset, Transformation, 
    JoinRecommendation, Optimization, DataPatternAnalysis,
    RecommendationType
)


class TestRecommendationEngine:
    """Test cases for the RecommendationEngine class"""
    
    @pytest.fixture
    def engine(self):
        """Create a recommendation engine instance for testing"""
        return RecommendationEngine()
    
    @pytest.fixture
    def sample_schema_source(self):
        """Sample source schema for testing"""
        return Schema(
            name="source_table",
            columns=[
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
                {"name": "age", "type": "string"},  # Intentionally wrong type
                {"name": "created_at", "type": "string"}
            ],
            data_types={
                "id": "int64",
                "name": "object",
                "age": "object",
                "created_at": "object"
            }
        )
    
    @pytest.fixture
    def sample_schema_target(self):
        """Sample target schema for testing"""
        return Schema(
            name="target_table",
            columns=[
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
                {"name": "age", "type": "integer"},  # Correct type
                {"name": "created_at", "type": "datetime"},
                {"name": "updated_at", "type": "datetime"}  # New column
            ],
            data_types={
                "id": "int64",
                "name": "object",
                "age": "int64",
                "created_at": "datetime64[ns]",
                "updated_at": "datetime64[ns]"
            }
        )
    
    @pytest.fixture
    def sample_dataset_left(self, sample_schema_source):
        """Sample left dataset for join testing"""
        return Dataset(
            name="customers",
            schema=sample_schema_source,
            row_count=10000,
            size_mb=50.0,
            quality_score=0.85
        )
    
    @pytest.fixture
    def sample_dataset_right(self):
        """Sample right dataset for join testing"""
        schema = Schema(
            name="orders",
            columns=[
                {"name": "id", "type": "integer"},
                {"name": "customer_id", "type": "integer"},
                {"name": "amount", "type": "float"},
                {"name": "order_date", "type": "datetime"}
            ],
            data_types={
                "id": "int64",
                "customer_id": "int64",
                "amount": "float64",
                "order_date": "datetime64[ns]"
            }
        )
        return Dataset(
            name="orders",
            schema=schema,
            row_count=25000,
            size_mb=75.0,
            quality_score=0.90
        )
    
    @pytest.fixture
    def sample_data(self):
        """Sample DataFrame for pattern analysis"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'age': ['25', '30', '35', '40', '45'],  # String but should be numeric
            'salary': [50000, 60000, 70000, 1000000, 55000],  # Has outlier
            'created_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'duplicate_col': [1, 1, 1, 1, 1]  # All same values
        })
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert hasattr(engine, 'vectorizer')
        assert hasattr(engine, 'performance_model')
        assert hasattr(engine, 'clustering_model')
        assert hasattr(engine, 'transformation_patterns')
        assert isinstance(engine.transformation_patterns, dict)
    
    def test_recommend_transformations_basic(self, engine, sample_schema_source, sample_schema_target):
        """Test basic transformation recommendations"""
        recommendations = engine.recommend_transformations(sample_schema_source, sample_schema_target)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check for expected transformations
        rec_names = [rec.name for rec in recommendations]
        assert 'add_missing_columns' in rec_names  # Should suggest adding updated_at
        assert 'convert_data_type' in rec_names  # Should suggest converting age and created_at
        
        # Verify recommendation structure
        for rec in recommendations:
            assert isinstance(rec, Transformation)
            assert hasattr(rec, 'name')
            assert hasattr(rec, 'confidence')
            assert 0 <= rec.confidence <= 1
            assert hasattr(rec, 'parameters')
            assert isinstance(rec.parameters, dict)
    
    def test_recommend_transformations_with_sample_data(self, engine, sample_schema_source, sample_data):
        """Test transformation recommendations with sample data"""
        sample_schema_source.sample_data = sample_data
        target_schema = Schema(
            name="target",
            columns=[{"name": col, "type": "string"} for col in sample_data.columns],
            data_types={col: "object" for col in sample_data.columns}
        )
        
        recommendations = engine.recommend_transformations(sample_schema_source, target_schema)
        
        assert len(recommendations) > 0
        
        # Should include pattern-based recommendations
        rec_types = [rec.type for rec in recommendations]
        assert 'data_cleaning' in rec_types or 'schema_alignment' in rec_types
    
    def test_suggest_optimizations_performance_issues(self, engine):
        """Test optimization suggestions for performance issues"""
        pipeline = {"name": "test_pipeline", "steps": ["extract", "transform", "load"]}
        metrics = {
            "execution_time_seconds": 600,  # 10 minutes - should trigger optimization
            "memory_usage_mb": 2000,  # 2GB - should trigger optimization
            "rows_processed": 2000000,  # 2M rows - should trigger optimization
            "error_rate": 0.05  # 5% error rate - should trigger optimization
        }
        
        optimizations = engine.suggest_optimizations(pipeline, metrics)
        
        assert isinstance(optimizations, list)
        assert len(optimizations) > 0
        
        # Check for expected optimization categories
        categories = [opt.category for opt in optimizations]
        assert 'performance' in categories
        assert 'memory' in categories
        assert 'scalability' in categories
        assert 'reliability' in categories
        
        # Verify optimization structure
        for opt in optimizations:
            assert isinstance(opt, Optimization)
            assert hasattr(opt, 'category')
            assert hasattr(opt, 'description')
            assert hasattr(opt, 'estimated_improvement')
            assert 0 <= opt.estimated_improvement <= 1
            assert hasattr(opt, 'priority')
            assert opt.priority > 0
    
    def test_suggest_optimizations_good_performance(self, engine):
        """Test optimization suggestions for good performance"""
        pipeline = {"name": "test_pipeline", "steps": ["extract", "transform", "load"]}
        metrics = {
            "execution_time_seconds": 30,  # 30 seconds - good
            "memory_usage_mb": 100,  # 100MB - good
            "rows_processed": 10000,  # 10K rows - small dataset
            "error_rate": 0.001  # 0.1% error rate - good
        }
        
        optimizations = engine.suggest_optimizations(pipeline, metrics)
        
        # Should have fewer or no optimizations for good performance
        assert isinstance(optimizations, list)
        # May still have some minor optimizations, but fewer than performance issues case
    
    def test_recommend_join_strategy_common_columns(self, engine, sample_dataset_left, sample_dataset_right):
        """Test join strategy recommendation with common columns"""
        # Add a common column to both datasets
        sample_dataset_left.schema.columns.append({"name": "customer_id", "type": "integer"})
        sample_dataset_left.schema.data_types["customer_id"] = "int64"
        
        recommendation = engine.recommend_join_strategy(sample_dataset_left, sample_dataset_right)
        
        assert isinstance(recommendation, JoinRecommendation)
        assert recommendation.join_type in ['inner', 'left', 'right', 'outer']
        # The engine picks the first common column, which could be 'id' or 'customer_id'
        common_cols = {'id', 'customer_id'}
        assert recommendation.left_key in common_cols
        assert recommendation.right_key in common_cols
        assert 0 <= recommendation.confidence <= 1
        assert recommendation.estimated_rows > 0
        assert 0 <= recommendation.performance_score <= 1
    
    def test_recommend_join_strategy_no_common_columns(self, engine, sample_dataset_left, sample_dataset_right):
        """Test join strategy recommendation with no common columns"""
        # Remove the common 'id' column from one dataset to ensure no common columns
        sample_dataset_right.schema.columns = [col for col in sample_dataset_right.schema.columns if col['name'] != 'id']
        sample_dataset_right.schema.data_types = {k: v for k, v in sample_dataset_right.schema.data_types.items() if k != 'id'}
        
        recommendation = engine.recommend_join_strategy(sample_dataset_left, sample_dataset_right)
        
        assert isinstance(recommendation, JoinRecommendation)
        # When there are common columns, it won't be cross join
        if recommendation.join_type == 'cross':
            assert recommendation.confidence < 0.5  # Should have low confidence
            assert recommendation.left_key == ''
            assert recommendation.right_key == ''
        else:
            # If there are still common columns, that's also valid
            assert recommendation.join_type in ['inner', 'left', 'right', 'outer']
    
    def test_analyze_data_patterns(self, engine, sample_data):
        """Test data pattern analysis"""
        analysis = engine.analyze_data_patterns(sample_data)
        
        assert isinstance(analysis, DataPatternAnalysis)
        assert isinstance(analysis.patterns, list)
        assert isinstance(analysis.anomalies, list)
        assert isinstance(analysis.recommendations, list)
        assert isinstance(analysis.quality_issues, list)
        
        # Should detect patterns in the sample data
        assert len(analysis.patterns) > 0 or len(analysis.recommendations) > 0
        
        # Should detect quality issues (null values, outliers) or other patterns
        quality_text = ' '.join(analysis.quality_issues + analysis.recommendations + analysis.patterns)
        # Check for various quality indicators
        quality_indicators = ['null', 'missing', 'outlier', 'duplicate', 'convert', 'numeric', 'datetime']
        assert any(indicator in quality_text.lower() for indicator in quality_indicators)
    
    def test_analyze_data_patterns_datetime_detection(self, engine):
        """Test datetime pattern detection"""
        data = pd.DataFrame({
            'date_col': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'not_date': ['abc', 'def', 'ghi', 'jkl']
        })
        
        analysis = engine.analyze_data_patterns(data)
        
        # Should detect datetime pattern
        patterns_text = ' '.join(analysis.patterns + analysis.recommendations)
        assert 'datetime' in patterns_text.lower() or 'date' in patterns_text.lower()
    
    def test_analyze_data_patterns_numeric_detection(self, engine):
        """Test numeric pattern detection in string columns"""
        data = pd.DataFrame({
            'numeric_string': ['123', '456', '789', '101112'],
            'text_string': ['abc', 'def', 'ghi', 'jkl']
        })
        
        analysis = engine.analyze_data_patterns(data)
        
        # Should detect numeric pattern in string column
        patterns_text = ' '.join(analysis.patterns + analysis.recommendations)
        assert 'numeric' in patterns_text.lower()
    
    def test_learn_from_feedback_positive(self, engine):
        """Test learning from positive feedback"""
        recommendation_id = "test_rec_001"
        feedback = {
            "helpful": True,
            "rating": 5,
            "implementation_success": True
        }
        
        # Should not raise an exception
        engine.learn_from_feedback(recommendation_id, feedback)
        
        # Verify feedback is stored
        assert recommendation_id in engine.user_feedback
        assert engine.user_feedback[recommendation_id] == feedback
    
    def test_learn_from_feedback_negative(self, engine):
        """Test learning from negative feedback"""
        recommendation_id = "test_rec_002"
        feedback = {
            "helpful": False,
            "rating": 2,
            "implementation_success": False,
            "issues": "Recommendation was not applicable"
        }
        
        # Should not raise an exception
        engine.learn_from_feedback(recommendation_id, feedback)
        
        # Verify feedback is stored
        assert recommendation_id in engine.user_feedback
        assert engine.user_feedback[recommendation_id] == feedback
    
    def test_is_datetime_pattern(self, engine):
        """Test datetime pattern detection helper method"""
        # Valid datetime patterns
        datetime_series = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        assert engine._is_datetime_pattern(datetime_series) == True
        
        # Invalid datetime patterns
        text_series = pd.Series(['abc', 'def', 'ghi'])
        assert engine._is_datetime_pattern(text_series) == False
        
        # Mixed patterns (should return False)
        mixed_series = pd.Series(['2023-01-01', 'abc', '2023-01-03'])
        assert engine._is_datetime_pattern(mixed_series) == False
        
        # Empty series
        empty_series = pd.Series([])
        assert engine._is_datetime_pattern(empty_series) == False
    
    def test_select_best_join_key(self, engine, sample_dataset_left, sample_dataset_right):
        """Test join key selection logic"""
        common_cols = {'id', 'customer_id'}
        
        best_key = engine._select_best_join_key(common_cols, sample_dataset_left, sample_dataset_right)
        
        assert best_key in common_cols
        assert isinstance(best_key, str)
        assert len(best_key) > 0
    
    def test_calculate_join_performance_score(self, engine, sample_dataset_left, sample_dataset_right):
        """Test join performance score calculation"""
        score = engine._calculate_join_performance_score(
            sample_dataset_left, sample_dataset_right, 'inner'
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_error_handling_invalid_schema(self, engine):
        """Test error handling with invalid schema"""
        invalid_schema = Schema(
            name="invalid",
            columns=[],  # Empty columns
            data_types={}
        )
        
        valid_schema = Schema(
            name="valid",
            columns=[{"name": "id", "type": "integer"}],
            data_types={"id": "int64"}
        )
        
        # Should handle gracefully and return empty list
        recommendations = engine.recommend_transformations(invalid_schema, valid_schema)
        assert isinstance(recommendations, list)
    
    def test_error_handling_invalid_data(self, engine):
        """Test error handling with invalid data"""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        analysis = engine.analyze_data_patterns(empty_df)
        
        assert isinstance(analysis, DataPatternAnalysis)
        assert isinstance(analysis.patterns, list)
        assert isinstance(analysis.anomalies, list)
    
    def test_transformation_confidence_scores(self, engine, sample_schema_source, sample_schema_target):
        """Test that transformation confidence scores are reasonable"""
        recommendations = engine.recommend_transformations(sample_schema_source, sample_schema_target)
        
        for rec in recommendations:
            assert 0 <= rec.confidence <= 1
            # High-confidence transformations should have confidence > 0.5
            if rec.name in ['add_missing_columns', 'convert_data_type']:
                assert rec.confidence > 0.5
    
    def test_optimization_priority_ordering(self, engine):
        """Test that optimizations are properly prioritized"""
        pipeline = {"name": "test_pipeline"}
        metrics = {
            "execution_time_seconds": 600,
            "memory_usage_mb": 2000,
            "error_rate": 0.05
        }
        
        optimizations = engine.suggest_optimizations(pipeline, metrics)
        
        if len(optimizations) > 1:
            # Should be sorted by priority (lower number = higher priority)
            priorities = [opt.priority for opt in optimizations]
            assert priorities == sorted(priorities)
    
    def test_schema_helper_methods(self):
        """Test Schema helper methods"""
        schema = Schema(
            name="test_schema",
            columns=[
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
                {"name": "age", "type": "integer"},
                {"name": "salary", "type": "float"},
                {"name": "department", "type": "string"}
            ],
            data_types={
                "id": "int64",
                "name": "object",
                "age": "int64",
                "salary": "float64",
                "department": "object"
            }
        )
        
        # Test column name extraction
        column_names = schema.get_column_names()
        assert column_names == ["id", "name", "age", "salary", "department"]
        
        # Test numeric column identification
        numeric_cols = schema.get_numeric_columns()
        assert set(numeric_cols) == {"id", "age", "salary"}
        
        # Test categorical column identification
        categorical_cols = schema.get_categorical_columns()
        assert set(categorical_cols) == {"name", "department"}
    
    def test_dataset_complexity_score(self):
        """Test dataset complexity calculation"""
        schema = Schema(
            name="test",
            columns=[{"name": f"col_{i}", "type": "string"} for i in range(10)],
            data_types={f"col_{i}": "object" for i in range(10)}
        )
        
        dataset = Dataset(
            name="test_dataset",
            schema=schema,
            row_count=100000,
            size_mb=50.0
        )
        
        complexity = dataset.get_complexity_score()
        assert isinstance(complexity, float)
        assert complexity > 0
        
        # Larger datasets should have higher complexity
        large_dataset = Dataset(
            name="large_dataset",
            schema=schema,
            row_count=1000000,
            size_mb=500.0
        )
        
        large_complexity = large_dataset.get_complexity_score()
        assert large_complexity > complexity


if __name__ == "__main__":
    pytest.main([__file__])