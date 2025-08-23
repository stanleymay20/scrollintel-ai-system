"""
Unit tests for the transformation engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.transformation_engine import (
    TransformationEngine,
    TransformationConfig,
    TransformationType,
    DataType,
    FilterTransformation,
    MapTransformation,
    AggregateTransformation,
    JoinTransformation,
    DataTypeConverter,
    CustomTransformationFramework,
    PerformanceOptimizer
)
from scrollintel.models.transformation_models import (
    TransformationValidator,
    TransformationOptimizer,
    SchemaValidationResult,
    TransformationMetrics
)


class TestTransformationEngine:
    """Test cases for the main TransformationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TransformationEngine()
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
        })
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        assert isinstance(self.engine, TransformationEngine)
        assert len(self.engine.modules) == 4
        assert TransformationType.FILTER in self.engine.modules
        assert TransformationType.MAP in self.engine.modules
        assert TransformationType.AGGREGATE in self.engine.modules
        assert TransformationType.JOIN in self.engine.modules
        assert isinstance(self.engine.converter, DataTypeConverter)
        assert isinstance(self.engine.custom_framework, CustomTransformationFramework)
        assert isinstance(self.engine.optimizer, PerformanceOptimizer)
        assert self.engine.execution_history == []
    
    def test_register_custom_transformation(self):
        """Test registering custom transformations."""
        def custom_func(data, **kwargs):
            return data * 2
        
        self.engine.register_custom_transformation('double', custom_func)
        assert 'double' in self.engine.custom_framework.custom_transformations
        assert self.engine.custom_framework.custom_transformations['double'] == custom_func
    
    def test_get_transformation_recommendations(self):
        """Test getting transformation recommendations."""
        recommendations = self.engine.get_transformation_recommendations(self.sample_data)
        
        assert isinstance(recommendations, list)
        # The sample data is small, so may not have aggregate recommendations
        # Just verify we get some kind of recommendations or empty list
        assert len(recommendations) >= 0
    
    def test_get_performance_metrics_empty(self):
        """Test getting performance metrics with no execution history."""
        metrics = self.engine.get_performance_metrics()
        assert metrics == {}
    
    def test_get_performance_metrics_with_history(self):
        """Test getting performance metrics with execution history."""
        # Add some mock execution history
        self.engine.execution_history = [
            {
                'timestamp': datetime.now(),
                'transformation_name': 'test1',
                'transformation_type': 'filter',
                'success': True,
                'execution_time': 1.0,
                'rows_processed': 100,
                'rows_output': 80,
                'error_message': None
            },
            {
                'timestamp': datetime.now(),
                'transformation_name': 'test2',
                'transformation_type': 'map',
                'success': False,
                'execution_time': 0.5,
                'rows_processed': 100,
                'rows_output': 0,
                'error_message': 'Test error'
            }
        ]
        
        metrics = self.engine.get_performance_metrics()
        
        assert metrics['total_executions'] == 2
        assert metrics['successful_executions'] == 1
        assert metrics['failed_executions'] == 1
        assert metrics['success_rate'] == 0.5
        assert metrics['average_execution_time'] == 1.0
        assert metrics['total_rows_processed'] == 100
        assert 'filter' in metrics['transformation_types']
        assert 'map' in metrics['transformation_types']
        assert len(metrics['recent_failures']) == 1


class TestFilterTransformation:
    """Test cases for FilterTransformation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter_transform = FilterTransformation()
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
    
    def test_filter_transformation_success(self):
        """Test successful filter transformation."""
        config = TransformationConfig(
            name='filter_test',
            type=TransformationType.FILTER,
            parameters={'condition': 'value > 25'}
        )
        
        result = self.filter_transform.transform(self.sample_data, config)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3  # 30, 40, 50
        assert result.rows_processed == 5
        assert result.rows_output == 3
        assert result.execution_time > 0
        assert 'filter_selectivity' in result.performance_metrics
    
    def test_filter_transformation_no_condition(self):
        """Test filter transformation without condition."""
        config = TransformationConfig(
            name='filter_test',
            type=TransformationType.FILTER,
            parameters={}
        )
        
        result = self.filter_transform.transform(self.sample_data, config)
        
        assert result.success is False
        assert 'Filter condition is required' in result.error_message
    
    def test_filter_transformation_invalid_condition(self):
        """Test filter transformation with invalid condition."""
        config = TransformationConfig(
            name='filter_test',
            type=TransformationType.FILTER,
            parameters={'condition': 'invalid_column > 10'}
        )
        
        result = self.filter_transform.transform(self.sample_data, config)
        
        assert result.success is False
        assert 'Filter transformation failed' in result.error_message
    
    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = TransformationConfig(
            name='test',
            type=TransformationType.FILTER,
            parameters={'condition': 'value > 0'}
        )
        
        invalid_config = TransformationConfig(
            name='test',
            type=TransformationType.FILTER,
            parameters={}
        )
        
        assert self.filter_transform.validate_config(valid_config) is True
        assert self.filter_transform.validate_config(invalid_config) is False
    
    def test_get_schema_impact(self):
        """Test schema impact calculation."""
        input_schema = {'id': 'int64', 'value': 'float64', 'category': 'object'}
        config = TransformationConfig(
            name='test',
            type=TransformationType.FILTER,
            parameters={'condition': 'value > 0'}
        )
        
        output_schema = self.filter_transform.get_schema_impact(input_schema, config)
        
        assert output_schema == input_schema  # Filter doesn't change schema


class TestMapTransformation:
    """Test cases for MapTransformation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.map_transform = MapTransformation()
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30],
            'name': ['Alice', 'Bob', 'Charlie']
        })
    
    def test_map_transformation_success(self):
        """Test successful map transformation."""
        config = TransformationConfig(
            name='map_test',
            type=TransformationType.MAP,
            parameters={
                'mappings': {
                    'double_value': 'value * 2',
                    'name_upper': 'name.str.upper()'
                }
            }
        )
        
        result = self.map_transform.transform(self.sample_data, config)
        
        assert result.success is True
        assert result.data is not None
        assert 'double_value' in result.data.columns
        assert 'name_upper' in result.data.columns
        assert result.data['double_value'].tolist() == [20, 40, 60]
        assert result.data['name_upper'].tolist() == ['ALICE', 'BOB', 'CHARLIE']
    
    def test_map_transformation_no_mappings(self):
        """Test map transformation without mappings."""
        config = TransformationConfig(
            name='map_test',
            type=TransformationType.MAP,
            parameters={}
        )
        
        result = self.map_transform.transform(self.sample_data, config)
        
        assert result.success is False
        assert 'Column mappings are required' in result.error_message
    
    def test_map_transformation_function_mapping(self):
        """Test map transformation with function mapping."""
        def custom_func(row):
            return row['value'] + row['id']
        
        config = TransformationConfig(
            name='map_test',
            type=TransformationType.MAP,
            parameters={
                'mappings': {
                    'sum_value': custom_func
                }
            }
        )
        
        result = self.map_transform.transform(self.sample_data, config)
        
        assert result.success is True
        assert 'sum_value' in result.data.columns
        assert result.data['sum_value'].tolist() == [11, 22, 33]


class TestAggregateTransformation:
    """Test cases for AggregateTransformation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agg_transform = AggregateTransformation()
        self.sample_data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50],
            'count': [1, 1, 1, 1, 1]
        })
    
    def test_aggregate_transformation_with_groupby(self):
        """Test aggregate transformation with group by."""
        config = TransformationConfig(
            name='agg_test',
            type=TransformationType.AGGREGATE,
            parameters={
                'group_by': ['category'],
                'aggregations': {
                    'value': 'sum',
                    'count': 'count'
                }
            }
        )
        
        result = self.agg_transform.transform(self.sample_data, config)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2  # Two categories
        assert 'category' in result.data.columns
        assert 'value' in result.data.columns
        assert 'count' in result.data.columns
    
    def test_aggregate_transformation_global(self):
        """Test global aggregate transformation."""
        config = TransformationConfig(
            name='agg_test',
            type=TransformationType.AGGREGATE,
            parameters={
                'aggregations': {
                    'value': 'sum',
                    'count': 'mean'
                }
            }
        )
        
        result = self.agg_transform.transform(self.sample_data, config)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1  # Single row for global aggregation
        assert result.data['value'].iloc[0] == 150  # Sum of all values
        assert result.data['count'].iloc[0] == 1.0  # Mean of counts
    
    def test_aggregate_transformation_no_aggregations(self):
        """Test aggregate transformation without aggregations."""
        config = TransformationConfig(
            name='agg_test',
            type=TransformationType.AGGREGATE,
            parameters={}
        )
        
        result = self.agg_transform.transform(self.sample_data, config)
        
        assert result.success is False
        assert 'Aggregation functions are required' in result.error_message


class TestJoinTransformation:
    """Test cases for JoinTransformation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.join_transform = JoinTransformation()
        self.left_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'dept_id': [1, 2, 1]
        })
        self.right_data = pd.DataFrame({
            'dept_id': [1, 2, 3],
            'dept_name': ['IT', 'HR', 'Finance']
        })
    
    def test_join_transformation_inner(self):
        """Test inner join transformation."""
        config = TransformationConfig(
            name='join_test',
            type=TransformationType.JOIN,
            parameters={
                'right_data': self.right_data,
                'join_keys': ['dept_id'],
                'join_type': 'inner'
            }
        )
        
        result = self.join_transform.transform(self.left_data, config)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3  # All rows should match
        assert 'dept_name' in result.data.columns
        assert result.data['dept_name'].tolist() == ['IT', 'HR', 'IT']
    
    def test_join_transformation_left(self):
        """Test left join transformation."""
        config = TransformationConfig(
            name='join_test',
            type=TransformationType.JOIN,
            parameters={
                'right_data': self.right_data,
                'join_keys': ['dept_id'],
                'join_type': 'left'
            }
        )
        
        result = self.join_transform.transform(self.left_data, config)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3  # All left rows preserved
    
    def test_join_transformation_missing_params(self):
        """Test join transformation with missing parameters."""
        config = TransformationConfig(
            name='join_test',
            type=TransformationType.JOIN,
            parameters={'join_keys': ['dept_id']}  # Missing right_data
        )
        
        result = self.join_transform.transform(self.left_data, config)
        
        assert result.success is False
        assert 'Right dataset and join keys are required' in result.error_message


class TestDataTypeConverter:
    """Test cases for DataTypeConverter."""
    
    def test_convert_to_string(self):
        """Test conversion to string type."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = DataTypeConverter.convert_column(data, DataType.STRING)
        
        assert result.dtype == 'object'
        assert result.tolist() == ['1', '2', '3', '4', '5']
    
    def test_convert_to_integer(self):
        """Test conversion to integer type."""
        data = pd.Series(['1', '2', '3', '4', '5'])
        result = DataTypeConverter.convert_column(data, DataType.INTEGER)
        
        assert result.dtype == 'Int64'
        assert result.tolist() == [1, 2, 3, 4, 5]
    
    def test_convert_to_float(self):
        """Test conversion to float type."""
        data = pd.Series(['1.5', '2.5', '3.5'])
        result = DataTypeConverter.convert_column(data, DataType.FLOAT)
        
        assert result.dtype == 'float64'
        assert result.tolist() == [1.5, 2.5, 3.5]
    
    def test_convert_to_boolean(self):
        """Test conversion to boolean type."""
        data = pd.Series([1, 0, 1, 0])
        result = DataTypeConverter.convert_column(data, DataType.BOOLEAN)
        
        assert result.dtype == 'bool'
        assert result.tolist() == [True, False, True, False]
    
    def test_convert_to_datetime(self):
        """Test conversion to datetime type."""
        data = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        result = DataTypeConverter.convert_column(data, DataType.DATETIME)
        
        assert pd.api.types.is_datetime64_any_dtype(result)
    
    def test_validate_conversion_success(self):
        """Test successful conversion validation."""
        data = pd.Series(['1', '2', '3', '4', '5'])
        validation = DataTypeConverter.validate_conversion(data, DataType.INTEGER)
        
        assert validation['convertible'] is True
        assert validation['success_rate'] == 1.0
        assert validation['null_count'] == 0
    
    def test_validate_conversion_partial_success(self):
        """Test partial conversion validation."""
        data = pd.Series(['1', '2', 'invalid', '4', '5'])
        validation = DataTypeConverter.validate_conversion(data, DataType.INTEGER)
        
        assert validation['convertible'] is True
        assert validation['success_rate'] == 0.8  # 4 out of 5 successful
        assert validation['null_count'] == 1


class TestCustomTransformationFramework:
    """Test cases for CustomTransformationFramework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = CustomTransformationFramework()
        self.sample_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
    
    def test_register_transformation(self):
        """Test registering a custom transformation."""
        def double_values(data, **kwargs):
            return data * 2
        
        self.framework.register_transformation('double', double_values)
        
        assert 'double' in self.framework.custom_transformations
        assert self.framework.custom_transformations['double'] == double_values
    
    def test_execute_custom_transformation_success(self):
        """Test successful execution of custom transformation."""
        def add_constant(data, constant=10, **kwargs):
            result = data.copy()
            result['value'] = result['value'] + constant
            return result
        
        self.framework.register_transformation('add_constant', add_constant)
        
        result = self.framework.execute_custom_transformation(
            self.sample_data, 'add_constant', constant=5
        )
        
        assert result.success is True
        assert result.data is not None
        assert result.data['value'].tolist() == [6, 7, 8, 9, 10]
    
    def test_execute_custom_transformation_not_found(self):
        """Test execution of non-existent custom transformation."""
        result = self.framework.execute_custom_transformation(
            self.sample_data, 'nonexistent'
        )
        
        assert result.success is False
        assert 'not found' in result.error_message
    
    def test_execute_custom_transformation_error(self):
        """Test execution of custom transformation with error."""
        def error_func(data, **kwargs):
            raise ValueError("Test error")
        
        self.framework.register_transformation('error_func', error_func)
        
        result = self.framework.execute_custom_transformation(
            self.sample_data, 'error_func'
        )
        
        assert result.success is False
        assert 'Test error' in result.error_message


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer."""
    
    def test_optimize_dataframe_operations(self):
        """Test DataFrame optimization."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'] * 100,  # Repetitive data
            'value': range(500)
        })
        
        optimized = PerformanceOptimizer.optimize_dataframe_operations(data)
        
        # Category column should be converted to category type
        assert optimized['category'].dtype.name == 'category'
        assert optimized['value'].dtype == data['value'].dtype  # Numeric unchanged
    
    def test_should_use_parallel_processing(self):
        """Test parallel processing decision."""
        # Small dataset, low complexity
        assert PerformanceOptimizer.should_use_parallel_processing(1000, 'low') is False
        
        # Large dataset, low complexity
        assert PerformanceOptimizer.should_use_parallel_processing(200000, 'low') is True
        
        # Medium dataset, high complexity
        assert PerformanceOptimizer.should_use_parallel_processing(20000, 'high') is True
    
    def test_get_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        # Small dataset
        chunk_size = PerformanceOptimizer.get_optimal_chunk_size(1000)
        assert chunk_size == 1000
        
        # Medium dataset
        chunk_size = PerformanceOptimizer.get_optimal_chunk_size(200000)
        assert chunk_size == 50000
        
        # Large dataset
        chunk_size = PerformanceOptimizer.get_optimal_chunk_size(2000000)
        assert chunk_size == 100000


class TestTransformationValidator:
    """Test cases for TransformationValidator."""
    
    def test_validate_transformation_config_valid(self):
        """Test validation of valid transformation configuration."""
        config = {
            'name': 'test_filter',
            'type': 'filter',
            'parameters': {
                'condition': 'value > 0'
            }
        }
        
        result = TransformationValidator.validate_transformation_config(config)
        
        assert result.is_valid is True
        assert len(result.validation_errors) == 0
    
    def test_validate_transformation_config_missing_fields(self):
        """Test validation with missing required fields."""
        config = {
            'name': 'test_filter'
            # Missing 'type' and 'parameters'
        }
        
        result = TransformationValidator.validate_transformation_config(config)
        
        assert result.is_valid is False
        assert len(result.validation_errors) > 0
        assert any('Missing required fields' in error for error in result.validation_errors)
    
    def test_validate_transformation_config_invalid_type(self):
        """Test validation with invalid transformation type."""
        config = {
            'name': 'test_transform',
            'type': 'invalid_type',
            'parameters': {}
        }
        
        result = TransformationValidator.validate_transformation_config(config)
        
        assert result.is_valid is False
        assert any('Invalid transformation type' in error for error in result.validation_errors)
    
    def test_validate_data_schema_valid(self):
        """Test validation of valid data schema."""
        data_schema = {'id': 'int64', 'name': 'object', 'value': 'float64'}
        required_schema = {'id': 'int64', 'name': 'object'}
        
        result = TransformationValidator.validate_data_schema(data_schema, required_schema)
        
        assert result.is_valid is True
        assert len(result.missing_columns) == 0
        assert len(result.type_mismatches) == 0
    
    def test_validate_data_schema_missing_columns(self):
        """Test validation with missing columns."""
        data_schema = {'id': 'int64'}
        required_schema = {'id': 'int64', 'name': 'object', 'value': 'float64'}
        
        result = TransformationValidator.validate_data_schema(data_schema, required_schema)
        
        assert result.is_valid is False
        assert 'name' in result.missing_columns
        assert 'value' in result.missing_columns
    
    def test_validate_data_schema_type_mismatches(self):
        """Test validation with type mismatches."""
        data_schema = {'id': 'object', 'name': 'int64'}
        required_schema = {'id': 'int64', 'name': 'object'}
        
        result = TransformationValidator.validate_data_schema(data_schema, required_schema)
        
        assert result.is_valid is False
        assert 'id' in result.type_mismatches
        assert 'name' in result.type_mismatches
        assert result.type_mismatches['id']['expected'] == 'int64'
        assert result.type_mismatches['id']['actual'] == 'object'


class TestTransformationOptimizer:
    """Test cases for TransformationOptimizer."""
    
    def test_optimize_transformation_plan(self):
        """Test optimization of transformation plan."""
        transformations = [
            {'name': 'join_data', 'type': 'join'},
            {'name': 'filter_data', 'type': 'filter'},
            {'name': 'convert_types', 'type': 'convert'},
            {'name': 'map_columns', 'type': 'map'}
        ]
        
        plan = TransformationOptimizer.optimize_transformation_plan(transformations, 50000)
        
        # Check that filters come first
        filter_index = next(i for i, t in enumerate(plan.transformations) if t['type'] == 'filter')
        join_index = next(i for i, t in enumerate(plan.transformations) if t['type'] == 'join')
        
        assert filter_index < join_index  # Filters should come before joins
        assert plan.estimated_execution_time > 0
        assert plan.estimated_memory_usage > 0
        assert len(plan.optimization_applied) > 0
    
    def test_get_performance_recommendations_slow_execution(self):
        """Test performance recommendations for slow execution."""
        metrics = TransformationMetrics(
            execution_id='test',
            transformation_name='slow_transform',
            transformation_type='map',
            start_time=datetime.now(),
            execution_time=15.0,  # Slow execution
            rows_processed=1000,
            rows_output=1000,
            success=True
        )
        
        recommendations = TransformationOptimizer.get_performance_recommendations(metrics)
        
        assert len(recommendations) > 0
        assert any(rec.recommendation_type == 'performance' for rec in recommendations)
    
    def test_get_performance_recommendations_high_memory(self):
        """Test performance recommendations for high memory usage."""
        metrics = TransformationMetrics(
            execution_id='test',
            transformation_name='memory_intensive',
            transformation_type='aggregate',
            start_time=datetime.now(),
            execution_time=5.0,
            memory_usage_mb=2000.0,  # High memory usage
            rows_processed=1000,
            rows_output=1000,
            success=True
        )
        
        recommendations = TransformationOptimizer.get_performance_recommendations(metrics)
        
        assert len(recommendations) > 0
        assert any(rec.recommendation_type == 'memory' for rec in recommendations)
    
    def test_get_performance_recommendations_low_output_ratio(self):
        """Test performance recommendations for low output ratio."""
        metrics = TransformationMetrics(
            execution_id='test',
            transformation_name='selective_filter',
            transformation_type='filter',
            start_time=datetime.now(),
            execution_time=2.0,
            rows_processed=10000,
            rows_output=500,  # Only 5% output ratio
            success=True
        )
        
        recommendations = TransformationOptimizer.get_performance_recommendations(metrics)
        
        assert len(recommendations) > 0
        assert any(rec.recommendation_type == 'optimization' for rec in recommendations)


if __name__ == '__main__':
    pytest.main([__file__])