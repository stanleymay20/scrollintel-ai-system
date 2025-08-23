"""
API routes for the transformation engine.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import json
from datetime import datetime
import uuid

from scrollintel.engines.transformation_engine import (
    TransformationEngine, 
    TransformationConfig, 
    TransformationType,
    DataType
)
from scrollintel.models.transformation_models import (
    TransformationTemplate,
    TransformationExecution,
    TransformationValidator,
    TransformationOptimizer,
    TransformationMetrics
)

router = APIRouter(prefix="/api/v1/transformations", tags=["transformations"])

# Global transformation engine instance
transformation_engine = TransformationEngine()


@router.post("/execute")
async def execute_transformation(
    transformation_config: Dict[str, Any],
    data_file: UploadFile = File(...),
    optimize: bool = True
):
    """Execute a single transformation on uploaded data."""
    try:
        # Validate transformation configuration
        validation_result = TransformationValidator.validate_transformation_config(transformation_config)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transformation configuration: {validation_result.validation_errors}"
            )
        
        # Read uploaded data
        content = await data_file.read()
        
        # Determine file type and read data
        if data_file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif data_file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(content.decode('utf-8')))
        elif data_file.filename.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create transformation config
        config = TransformationConfig(
            name=transformation_config['name'],
            type=TransformationType(transformation_config['type']),
            parameters=transformation_config.get('parameters', {}),
            validation_rules=transformation_config.get('validation_rules', []),
            performance_hints=transformation_config.get('performance_hints', {})
        )
        
        # Execute transformation
        result = transformation_engine.execute_transformation(data, config)
        
        # Prepare response
        response_data = {
            'success': result.success,
            'execution_time': result.execution_time,
            'rows_processed': result.rows_processed,
            'rows_output': result.rows_output,
            'performance_metrics': result.performance_metrics
        }
        
        if result.success and result.data is not None:
            # Convert result data to JSON-serializable format
            response_data['output_data'] = result.data.to_dict('records')
            response_data['output_schema'] = {col: str(dtype) for col, dtype in result.data.dtypes.items()}
        else:
            response_data['error_message'] = result.error_message
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation execution failed: {str(e)}")


@router.post("/execute-pipeline")
async def execute_transformation_pipeline(
    transformations: List[Dict[str, Any]],
    data_file: UploadFile = File(...),
    optimize_plan: bool = True
):
    """Execute a pipeline of transformations on uploaded data."""
    try:
        # Read uploaded data
        content = await data_file.read()
        
        if data_file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif data_file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(content.decode('utf-8')))
        elif data_file.filename.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate all transformation configurations
        for i, transform_config in enumerate(transformations):
            validation_result = TransformationValidator.validate_transformation_config(transform_config)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration for transformation {i}: {validation_result.validation_errors}"
                )
        
        # Optimize transformation plan if requested
        if optimize_plan:
            optimization_plan = TransformationOptimizer.optimize_transformation_plan(
                transformations, len(data)
            )
            transformations = optimization_plan.transformations
        
        # Create transformation configs
        configs = []
        for transform_config in transformations:
            config = TransformationConfig(
                name=transform_config['name'],
                type=TransformationType(transform_config['type']),
                parameters=transform_config.get('parameters', {}),
                validation_rules=transform_config.get('validation_rules', []),
                performance_hints=transform_config.get('performance_hints', {})
            )
            configs.append(config)
        
        # Execute transformation pipeline
        results = transformation_engine.execute_transformation_pipeline(data, configs)
        
        # Prepare response
        pipeline_results = []
        final_data = None
        
        for i, result in enumerate(results):
            result_data = {
                'step': i + 1,
                'transformation_name': configs[i].name,
                'success': result.success,
                'execution_time': result.execution_time,
                'rows_processed': result.rows_processed,
                'rows_output': result.rows_output,
                'performance_metrics': result.performance_metrics
            }
            
            if not result.success:
                result_data['error_message'] = result.error_message
            else:
                final_data = result.data
            
            pipeline_results.append(result_data)
        
        response_data = {
            'pipeline_results': pipeline_results,
            'total_execution_time': sum(r.execution_time for r in results),
            'overall_success': all(r.success for r in results)
        }
        
        if final_data is not None:
            response_data['final_output'] = final_data.to_dict('records')
            response_data['final_schema'] = {col: str(dtype) for col, dtype in final_data.dtypes.items()}
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.post("/recommendations")
async def get_transformation_recommendations(
    data_file: UploadFile = File(...),
    target_schema: Optional[Dict[str, str]] = None
):
    """Get intelligent transformation recommendations for uploaded data."""
    try:
        # Read uploaded data
        content = await data_file.read()
        
        if data_file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif data_file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(content.decode('utf-8')))
        elif data_file.filename.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Get recommendations
        recommendations = transformation_engine.get_transformation_recommendations(data, target_schema)
        
        # Get data profile
        data_profile = transformation_engine._profile_data(data)
        
        return JSONResponse(content={
            'data_profile': data_profile,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.post("/validate-conversion")
async def validate_data_type_conversion(
    column_data: List[Any],
    target_type: str
):
    """Validate if data type conversion is possible and get conversion statistics."""
    try:
        # Convert to pandas Series
        data_series = pd.Series(column_data)
        
        # Validate conversion
        target_data_type = DataType(target_type)
        validation_result = transformation_engine.converter.validate_conversion(data_series, target_data_type)
        
        return JSONResponse(content=validation_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion validation failed: {str(e)}")


@router.post("/custom-transformation")
async def register_custom_transformation(
    name: str = Form(...),
    description: str = Form(...),
    function_code: str = Form(...),
    input_schema: str = Form(...),
    output_schema: str = Form(...)
):
    """Register a custom transformation function."""
    try:
        # Parse schemas
        input_schema_dict = json.loads(input_schema)
        output_schema_dict = json.loads(output_schema)
        
        # Create and compile the function
        exec_globals = {'pd': pd, 'np': __import__('numpy')}
        exec(function_code, exec_globals)
        
        # The function should be named 'transform_function'
        if 'transform_function' not in exec_globals:
            raise HTTPException(
                status_code=400,
                detail="Custom function must define a function named 'transform_function'"
            )
        
        custom_func = exec_globals['transform_function']
        
        # Register the transformation
        transformation_engine.register_custom_transformation(name, custom_func)
        
        return JSONResponse(content={
            'success': True,
            'message': f"Custom transformation '{name}' registered successfully",
            'function_name': name
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register custom transformation: {str(e)}")


@router.get("/templates")
async def get_transformation_templates():
    """Get available transformation templates."""
    try:
        # Predefined templates
        templates = [
            {
                'id': 'filter_nulls',
                'name': 'Filter Null Values',
                'description': 'Remove rows with null values in specified columns',
                'type': 'filter',
                'category': 'data_cleaning',
                'template_config': {
                    'type': 'filter',
                    'parameters': {
                        'condition': 'column_name.notna()'
                    }
                }
            },
            {
                'id': 'normalize_text',
                'name': 'Normalize Text',
                'description': 'Convert text to lowercase and remove extra spaces',
                'type': 'map',
                'category': 'text_processing',
                'template_config': {
                    'type': 'map',
                    'parameters': {
                        'mappings': {
                            'text_column': 'text_column.str.lower().str.strip()'
                        }
                    }
                }
            },
            {
                'id': 'aggregate_by_group',
                'name': 'Group By Aggregation',
                'description': 'Aggregate numeric columns by categorical groups',
                'type': 'aggregate',
                'category': 'aggregation',
                'template_config': {
                    'type': 'aggregate',
                    'parameters': {
                        'group_by': ['category_column'],
                        'aggregations': {
                            'numeric_column': 'sum',
                            'count_column': 'count'
                        }
                    }
                }
            },
            {
                'id': 'convert_to_datetime',
                'name': 'Convert to DateTime',
                'description': 'Convert string columns to datetime format',
                'type': 'convert',
                'category': 'type_conversion',
                'template_config': {
                    'type': 'convert',
                    'parameters': {
                        'conversions': {
                            'date_column': 'datetime'
                        }
                    }
                }
            }
        ]
        
        return JSONResponse(content={
            'templates': templates,
            'total_templates': len(templates)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@router.get("/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics for executed transformations."""
    try:
        metrics = transformation_engine.get_performance_metrics()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.post("/optimize-plan")
async def optimize_transformation_plan(
    transformations: List[Dict[str, Any]],
    data_size: int
):
    """Optimize a transformation plan for better performance."""
    try:
        # Validate transformations
        for i, transform_config in enumerate(transformations):
            validation_result = TransformationValidator.validate_transformation_config(transform_config)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration for transformation {i}: {validation_result.validation_errors}"
                )
        
        # Optimize plan
        optimized_plan = TransformationOptimizer.optimize_transformation_plan(transformations, data_size)
        
        return JSONResponse(content={
            'plan_id': optimized_plan.plan_id,
            'original_transformations': transformations,
            'optimized_transformations': optimized_plan.transformations,
            'estimated_execution_time': optimized_plan.estimated_execution_time,
            'estimated_memory_usage': optimized_plan.estimated_memory_usage,
            'optimization_applied': optimized_plan.optimization_applied,
            'data_quality_checks': optimized_plan.data_quality_checks
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan optimization failed: {str(e)}")


@router.get("/data-types")
async def get_supported_data_types():
    """Get list of supported data types for conversion."""
    try:
        data_types = [
            {
                'type': dt.value,
                'description': f"Convert to {dt.value} type"
            }
            for dt in DataType
        ]
        
        return JSONResponse(content={
            'supported_types': data_types,
            'total_types': len(data_types)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data types: {str(e)}")


@router.get("/transformation-types")
async def get_transformation_types():
    """Get list of supported transformation types."""
    try:
        transformation_types = [
            {
                'type': tt.value,
                'description': f"{tt.value.title()} transformation"
            }
            for tt in TransformationType
        ]
        
        return JSONResponse(content={
            'transformation_types': transformation_types,
            'total_types': len(transformation_types)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transformation types: {str(e)}")


@router.delete("/clear-history")
async def clear_execution_history():
    """Clear transformation execution history."""
    try:
        transformation_engine.execution_history.clear()
        
        return JSONResponse(content={
            'success': True,
            'message': 'Execution history cleared successfully'
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for the transformation engine."""
    try:
        # Basic health check
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        test_config = TransformationConfig(
            name='health_check',
            type=TransformationType.FILTER,
            parameters={'condition': 'test > 0'}
        )
        
        result = transformation_engine.execute_transformation(test_data, test_config)
        
        return JSONResponse(content={
            'status': 'healthy' if result.success else 'unhealthy',
            'engine_ready': True,
            'test_execution_time': result.execution_time,
            'registered_custom_transformations': len(transformation_engine.custom_framework.custom_transformations),
            'execution_history_count': len(transformation_engine.execution_history)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e)
            }
        )