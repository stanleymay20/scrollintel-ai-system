"""
Data Pipeline Transformation Engine

This module provides the core transformation engine for the data pipeline automation system.
It supports pluggable transformation modules, common transformations, data type conversion,
custom transformation framework, and performance optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Types of transformations supported by the engine."""
    FILTER = "filter"
    MAP = "map"
    AGGREGATE = "aggregate"
    JOIN = "join"
    CONVERT = "convert"
    VALIDATE = "validate"
    CUSTOM = "custom"


class DataType(Enum):
    """Supported data types for conversion."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"


@dataclass
class TransformationConfig:
    """Configuration for a transformation operation."""
    name: str
    type: TransformationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    performance_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationResult:
    """Result of a transformation operation."""
    success: bool
    data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    rows_processed: int = 0
    rows_output: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class TransformationModule(ABC):
    """Abstract base class for transformation modules."""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, config: TransformationConfig) -> TransformationResult:
        """Execute the transformation on the provided data."""
        pass
    
    @abstractmethod
    def validate_config(self, config: TransformationConfig) -> bool:
        """Validate the transformation configuration."""
        pass
    
    @abstractmethod
    def get_schema_impact(self, input_schema: Dict[str, str], config: TransformationConfig) -> Dict[str, str]:
        """Determine the output schema based on input schema and configuration."""
        pass


class FilterTransformation(TransformationModule):
    """Filter transformation module for data filtering operations."""
    
    def transform(self, data: pd.DataFrame, config: TransformationConfig) -> TransformationResult:
        """Apply filter transformation to data."""
        start_time = datetime.now()
        
        try:
            condition = config.parameters.get('condition')
            if not condition:
                return TransformationResult(
                    success=False,
                    error_message="Filter condition is required"
                )
            
            # Apply filter condition
            filtered_data = data.query(condition)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TransformationResult(
                success=True,
                data=filtered_data,
                execution_time=execution_time,
                rows_processed=len(data),
                rows_output=len(filtered_data),
                performance_metrics={
                    'filter_selectivity': len(filtered_data) / len(data) if len(data) > 0 else 0,
                    'condition': condition
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TransformationResult(
                success=False,
                error_message=f"Filter transformation failed: {str(e)}",
                execution_time=execution_time,
                rows_processed=len(data)
            )
    
    def validate_config(self, config: TransformationConfig) -> bool:
        """Validate filter configuration."""
        return 'condition' in config.parameters
    
    def get_schema_impact(self, input_schema: Dict[str, str], config: TransformationConfig) -> Dict[str, str]:
        """Filter doesn't change schema, only row count."""
        return input_schema


class MapTransformation(TransformationModule):
    """Map transformation module for column transformations."""
    
    def transform(self, data: pd.DataFrame, config: TransformationConfig) -> TransformationResult:
        """Apply map transformation to data."""
        start_time = datetime.now()
        
        try:
            mappings = config.parameters.get('mappings', {})
            if not mappings:
                return TransformationResult(
                    success=False,
                    error_message="Column mappings are required"
                )
            
            result_data = data.copy()
            
            for column, expression in mappings.items():
                if isinstance(expression, str):
                    # Evaluate string expression with pandas namespace
                    result_data[column] = result_data.eval(expression, local_dict={'pd': pd})
                elif callable(expression):
                    # Apply function
                    result_data[column] = result_data.apply(expression, axis=1)
                else:
                    # Direct assignment
                    result_data[column] = expression
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TransformationResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                rows_processed=len(data),
                rows_output=len(result_data),
                performance_metrics={
                    'columns_mapped': len(mappings),
                    'mappings': list(mappings.keys())
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TransformationResult(
                success=False,
                error_message=f"Map transformation failed: {str(e)}",
                execution_time=execution_time,
                rows_processed=len(data)
            )
    
    def validate_config(self, config: TransformationConfig) -> bool:
        """Validate map configuration."""
        return 'mappings' in config.parameters and isinstance(config.parameters['mappings'], dict)
    
    def get_schema_impact(self, input_schema: Dict[str, str], config: TransformationConfig) -> Dict[str, str]:
        """Map can add or modify columns."""
        output_schema = input_schema.copy()
        mappings = config.parameters.get('mappings', {})
        
        for column in mappings.keys():
            # For simplicity, assume string type for new columns
            output_schema[column] = 'string'
        
        return output_schema


class AggregateTransformation(TransformationModule):
    """Aggregate transformation module for data aggregation operations."""
    
    def transform(self, data: pd.DataFrame, config: TransformationConfig) -> TransformationResult:
        """Apply aggregate transformation to data."""
        start_time = datetime.now()
        
        try:
            group_by = config.parameters.get('group_by', [])
            aggregations = config.parameters.get('aggregations', {})
            
            if not aggregations:
                return TransformationResult(
                    success=False,
                    error_message="Aggregation functions are required"
                )
            
            if group_by:
                grouped = data.groupby(group_by)
                result_data = grouped.agg(aggregations).reset_index()
            else:
                # Global aggregation
                result_data = pd.DataFrame([data.agg(aggregations)])
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TransformationResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                rows_processed=len(data),
                rows_output=len(result_data),
                performance_metrics={
                    'group_by_columns': group_by,
                    'aggregation_functions': list(aggregations.keys()),
                    'reduction_ratio': len(result_data) / len(data) if len(data) > 0 else 0
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TransformationResult(
                success=False,
                error_message=f"Aggregate transformation failed: {str(e)}",
                execution_time=execution_time,
                rows_processed=len(data)
            )
    
    def validate_config(self, config: TransformationConfig) -> bool:
        """Validate aggregate configuration."""
        return 'aggregations' in config.parameters and isinstance(config.parameters['aggregations'], dict)
    
    def get_schema_impact(self, input_schema: Dict[str, str], config: TransformationConfig) -> Dict[str, str]:
        """Aggregation changes schema based on group by and aggregation functions."""
        output_schema = {}
        
        group_by = config.parameters.get('group_by', [])
        aggregations = config.parameters.get('aggregations', {})
        
        # Add group by columns
        for col in group_by:
            if col in input_schema:
                output_schema[col] = input_schema[col]
        
        # Add aggregated columns
        for col in aggregations.keys():
            output_schema[col] = 'float'  # Most aggregations result in numeric values
        
        return output_schema


class JoinTransformation(TransformationModule):
    """Join transformation module for joining datasets."""
    
    def transform(self, data: pd.DataFrame, config: TransformationConfig) -> TransformationResult:
        """Apply join transformation to data."""
        start_time = datetime.now()
        
        try:
            right_data = config.parameters.get('right_data')
            join_keys = config.parameters.get('join_keys', [])
            join_type = config.parameters.get('join_type', 'inner')
            
            if right_data is None or not join_keys:
                return TransformationResult(
                    success=False,
                    error_message="Right dataset and join keys are required"
                )
            
            # Perform join
            result_data = data.merge(
                right_data,
                on=join_keys,
                how=join_type,
                suffixes=('_left', '_right')
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TransformationResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                rows_processed=len(data),
                rows_output=len(result_data),
                performance_metrics={
                    'join_type': join_type,
                    'join_keys': join_keys,
                    'left_rows': len(data),
                    'right_rows': len(right_data),
                    'join_ratio': len(result_data) / len(data) if len(data) > 0 else 0
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TransformationResult(
                success=False,
                error_message=f"Join transformation failed: {str(e)}",
                execution_time=execution_time,
                rows_processed=len(data)
            )
    
    def validate_config(self, config: TransformationConfig) -> bool:
        """Validate join configuration."""
        return ('right_data' in config.parameters and 
                'join_keys' in config.parameters and
                isinstance(config.parameters['join_keys'], list))
    
    def get_schema_impact(self, input_schema: Dict[str, str], config: TransformationConfig) -> Dict[str, str]:
        """Join combines schemas from both datasets."""
        output_schema = input_schema.copy()
        
        right_schema = config.parameters.get('right_schema', {})
        join_keys = config.parameters.get('join_keys', [])
        
        for col, dtype in right_schema.items():
            if col not in join_keys:  # Don't duplicate join keys
                output_schema[f"{col}_right"] = dtype
        
        return output_schema


class DataTypeConverter:
    """Handles data type conversions and validation."""
    
    @staticmethod
    def convert_column(data: pd.Series, target_type: DataType) -> pd.Series:
        """Convert a pandas Series to the target data type."""
        try:
            if target_type == DataType.STRING:
                return data.astype(str)
            elif target_type == DataType.INTEGER:
                return pd.to_numeric(data, errors='coerce').astype('Int64')
            elif target_type == DataType.FLOAT:
                return pd.to_numeric(data, errors='coerce')
            elif target_type == DataType.BOOLEAN:
                return data.astype(bool)
            elif target_type == DataType.DATETIME:
                return pd.to_datetime(data, errors='coerce')
            elif target_type == DataType.JSON:
                return data.apply(lambda x: str(x) if pd.notna(x) else None)
            elif target_type == DataType.ARRAY:
                return data.apply(lambda x: [x] if pd.notna(x) else [])
            else:
                return data
        except Exception as e:
            logger.warning(f"Type conversion failed: {e}")
            return data
    
    @staticmethod
    def validate_conversion(data: pd.Series, target_type: DataType) -> Dict[str, Any]:
        """Validate if conversion is possible and return conversion statistics."""
        try:
            converted = DataTypeConverter.convert_column(data, target_type)
            null_count = converted.isnull().sum()
            
            return {
                'convertible': True,
                'null_count': int(null_count),
                'success_rate': (len(data) - null_count) / len(data) if len(data) > 0 else 0,
                'sample_converted': converted.head(5).tolist()
            }
        except Exception as e:
            return {
                'convertible': False,
                'error': str(e),
                'success_rate': 0
            }


class CustomTransformationFramework:
    """Framework for creating and managing custom transformations."""
    
    def __init__(self):
        self.custom_transformations: Dict[str, Callable] = {}
    
    def register_transformation(self, name: str, func: Callable) -> None:
        """Register a custom transformation function."""
        self.custom_transformations[name] = func
    
    def execute_custom_transformation(self, data: pd.DataFrame, name: str, **kwargs) -> TransformationResult:
        """Execute a registered custom transformation."""
        start_time = datetime.now()
        
        try:
            if name not in self.custom_transformations:
                return TransformationResult(
                    success=False,
                    error_message=f"Custom transformation '{name}' not found"
                )
            
            func = self.custom_transformations[name]
            result_data = func(data, **kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TransformationResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                rows_processed=len(data),
                rows_output=len(result_data) if isinstance(result_data, pd.DataFrame) else 0,
                performance_metrics={
                    'custom_transformation': name,
                    'parameters': kwargs
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TransformationResult(
                success=False,
                error_message=f"Custom transformation '{name}' failed: {str(e)}",
                execution_time=execution_time,
                rows_processed=len(data)
            )


class PerformanceOptimizer:
    """Handles performance optimization for transformations."""
    
    @staticmethod
    def optimize_dataframe_operations(data: pd.DataFrame) -> pd.DataFrame:
        """Apply performance optimizations to DataFrame operations."""
        # Convert object columns to category if beneficial
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() / len(data) < 0.5:  # Less than 50% unique values
                data[col] = data[col].astype('category')
        
        return data
    
    @staticmethod
    def should_use_parallel_processing(data_size: int, operation_complexity: str = 'medium') -> bool:
        """Determine if parallel processing should be used based on data size and complexity."""
        complexity_thresholds = {
            'low': 100000,
            'medium': 50000,
            'high': 10000
        }
        
        threshold = complexity_thresholds.get(operation_complexity, 50000)
        return data_size > threshold
    
    @staticmethod
    def get_optimal_chunk_size(data_size: int, available_memory_gb: float = 4.0) -> int:
        """Calculate optimal chunk size for processing large datasets."""
        # Estimate memory usage per row (rough approximation)
        estimated_row_size_mb = 0.001  # 1KB per row
        max_chunk_size = int((available_memory_gb * 1024) / estimated_row_size_mb)
        
        # Use smaller chunks for very large datasets
        if data_size > 1000000:
            return min(max_chunk_size, 100000)
        elif data_size > 100000:
            return min(max_chunk_size, 50000)
        else:
            return data_size


class TransformationEngine:
    """Main transformation engine that orchestrates all transformation operations."""
    
    def __init__(self):
        self.modules: Dict[TransformationType, TransformationModule] = {
            TransformationType.FILTER: FilterTransformation(),
            TransformationType.MAP: MapTransformation(),
            TransformationType.AGGREGATE: AggregateTransformation(),
            TransformationType.JOIN: JoinTransformation()
        }
        self.converter = DataTypeConverter()
        self.custom_framework = CustomTransformationFramework()
        self.optimizer = PerformanceOptimizer()
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_custom_transformation(self, name: str, func: Callable) -> None:
        """Register a custom transformation function."""
        self.custom_framework.register_transformation(name, func)
    
    def execute_transformation(self, data: pd.DataFrame, config: TransformationConfig) -> TransformationResult:
        """Execute a transformation based on the provided configuration."""
        logger.info(f"Executing transformation: {config.name} ({config.type.value})")
        
        # Validate configuration
        if config.type in self.modules:
            module = self.modules[config.type]
            if not module.validate_config(config):
                return TransformationResult(
                    success=False,
                    error_message=f"Invalid configuration for {config.type.value} transformation"
                )
        
        # Apply performance optimizations
        optimized_data = self.optimizer.optimize_dataframe_operations(data.copy())
        
        # Execute transformation
        if config.type == TransformationType.CUSTOM:
            custom_name = config.parameters.get('custom_name')
            custom_params = config.parameters.get('custom_params', {})
            result = self.custom_framework.execute_custom_transformation(
                optimized_data, custom_name, **custom_params
            )
        elif config.type == TransformationType.CONVERT:
            result = self._execute_type_conversion(optimized_data, config)
        else:
            module = self.modules[config.type]
            result = module.transform(optimized_data, config)
        
        # Record execution history
        self.execution_history.append({
            'timestamp': datetime.now(),
            'transformation_name': config.name,
            'transformation_type': config.type.value,
            'success': result.success,
            'execution_time': result.execution_time,
            'rows_processed': result.rows_processed,
            'rows_output': result.rows_output,
            'error_message': result.error_message
        })
        
        return result
    
    def _execute_type_conversion(self, data: pd.DataFrame, config: TransformationConfig) -> TransformationResult:
        """Execute data type conversion transformation."""
        start_time = datetime.now()
        
        try:
            conversions = config.parameters.get('conversions', {})
            if not conversions:
                return TransformationResult(
                    success=False,
                    error_message="Type conversions are required"
                )
            
            result_data = data.copy()
            conversion_stats = {}
            
            for column, target_type_str in conversions.items():
                if column not in result_data.columns:
                    continue
                
                target_type = DataType(target_type_str)
                validation = self.converter.validate_conversion(result_data[column], target_type)
                
                if validation['convertible']:
                    result_data[column] = self.converter.convert_column(result_data[column], target_type)
                    conversion_stats[column] = validation
                else:
                    logger.warning(f"Skipping conversion for column {column}: {validation.get('error')}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TransformationResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                rows_processed=len(data),
                rows_output=len(result_data),
                performance_metrics={
                    'conversions_attempted': len(conversions),
                    'conversions_successful': len(conversion_stats),
                    'conversion_stats': conversion_stats
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TransformationResult(
                success=False,
                error_message=f"Type conversion failed: {str(e)}",
                execution_time=execution_time,
                rows_processed=len(data)
            )
    
    def execute_transformation_pipeline(self, data: pd.DataFrame, 
                                      transformations: List[TransformationConfig]) -> List[TransformationResult]:
        """Execute a series of transformations in sequence."""
        results = []
        current_data = data
        
        for config in transformations:
            result = self.execute_transformation(current_data, config)
            results.append(result)
            
            if result.success and result.data is not None:
                current_data = result.data
            else:
                logger.error(f"Transformation pipeline stopped at {config.name}: {result.error_message}")
                break
        
        return results
    
    def get_transformation_recommendations(self, data: pd.DataFrame, 
                                        target_schema: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Get intelligent transformation recommendations based on data analysis."""
        recommendations = []
        
        # Analyze data characteristics
        data_profile = self._profile_data(data)
        
        # Recommend data type conversions
        for column, current_type in data_profile['column_types'].items():
            if target_schema and column in target_schema:
                target_type = target_schema[column]
                if current_type != target_type:
                    validation = self.converter.validate_conversion(data[column], DataType(target_type))
                    if validation['convertible'] and validation['success_rate'] > 0.8:
                        recommendations.append({
                            'type': 'conversion',
                            'column': column,
                            'from_type': current_type,
                            'to_type': target_type,
                            'confidence': validation['success_rate'],
                            'reason': f"High conversion success rate ({validation['success_rate']:.2%})"
                        })
        
        # Recommend filtering for data quality
        for column, stats in data_profile['column_stats'].items():
            if stats.get('null_percentage', 0) > 0.5:
                recommendations.append({
                    'type': 'filter',
                    'column': column,
                    'suggestion': f"Filter out rows where {column} is null",
                    'reason': f"High null percentage ({stats['null_percentage']:.2%})"
                })
        
        # Recommend aggregations for large datasets
        if len(data) > 100000:
            categorical_columns = [col for col, dtype in data_profile['column_types'].items() 
                                 if dtype in ['object', 'category']]
            if categorical_columns:
                recommendations.append({
                    'type': 'aggregate',
                    'suggestion': f"Consider aggregating by {categorical_columns[:2]}",
                    'reason': "Large dataset could benefit from aggregation"
                })
        
        return recommendations
    
    def _profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile the data to understand its characteristics."""
        profile = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'column_types': {},
            'column_stats': {},
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        for column in data.columns:
            dtype = str(data[column].dtype)
            profile['column_types'][column] = dtype
            
            stats = {
                'null_count': data[column].isnull().sum(),
                'null_percentage': data[column].isnull().sum() / len(data),
                'unique_count': data[column].nunique(),
                'unique_percentage': data[column].nunique() / len(data)
            }
            
            if data[column].dtype in ['int64', 'float64']:
                stats.update({
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'mean': data[column].mean(),
                    'std': data[column].std()
                })
            
            profile['column_stats'][column] = stats
        
        return profile
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for executed transformations."""
        if not self.execution_history:
            return {}
        
        successful_executions = [h for h in self.execution_history if h['success']]
        failed_executions = [h for h in self.execution_history if not h['success']]
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'average_execution_time': np.mean([h['execution_time'] for h in successful_executions]) if successful_executions else 0,
            'total_rows_processed': sum(h['rows_processed'] for h in successful_executions),
            'transformation_types': list(set(h['transformation_type'] for h in self.execution_history)),
            'recent_failures': [h['error_message'] for h in failed_executions[-5:]]
        }