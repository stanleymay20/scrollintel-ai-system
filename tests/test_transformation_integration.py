"""
Integration tests for the transformation engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io

from scrollintel.engines.transformation_engine import (
    TransformationEngine,
    TransformationConfig,
    TransformationType,
    DataType
)
from scrollintel.api.routes.transformation_routes import router
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestTransformationEngineIntegration:
    """Integration tests for the transformation engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TransformationEngine()
        
        # Create comprehensive test dataset
        np.random.seed(42)
        self.large_dataset = pd.DataFrame({
            'id': range(1000),
            'name': [f'User_{i}' for i in range(1000)],
            'age': np.random.randint(18, 80, 1000),
            'salary': np.random.normal(50000, 15000, 1000),
            'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 1000),
            'join_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'is_active': np.random.choice([True, False], 1000, p=[0.8, 0.2]),
            'score': np.random.uniform(0, 100, 1000)
        })
        
        # Dataset with data quality issues
        self.dirty_dataset = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, None, 7, 8],
            'name': ['Alice', 'Bob', None, 'David', '', 'Frank', 'Grace', 'Henry'],
            'age': ['25', '30', 'invalid', '40', '45', '50', '55', '60'],
            'salary': [50000, None, 70000, 80000, 90000, 100000, 110000, 120000],
            'email': ['alice@test.com', 'bob@test', 'charlie@test.com', None, 
                     'eve@test.com', 'frank@test.com', 'grace@test.com', 'henry@test.com']
        })
    
    def test_end_to_end_data_pipeline(self):
        """Test complete end-to-end data pipeline with multiple transformations."""
        # Define transformation pipeline
        transformations = [
            TransformationConfig(
                name='filter_active_users',
                type=TransformationType.FILTER,
                parameters={'condition': 'is_active == True'}
            ),
            TransformationConfig(
                name='convert_salary_to_k',
                type=TransformationType.MAP,
                parameters={
                    'mappings': {
                        'salary_k': 'salary / 1000'
                    }
                }
            ),
            TransformationConfig(
                name='aggregate_by_department',
                type=TransformationType.AGGREGATE,
                parameters={
                    'group_by': ['department'],
                    'aggregations': {
                        'salary_k': 'mean',
                        'age': 'mean',
                        'score': ['mean', 'std'],
                        'id': 'count'
                    }
                }
            )
        ]
        
        # Execute pipeline
        results = self.engine.execute_transformation_pipeline(self.large_dataset, transformations)
        
        # Verify results
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Check final result
        final_result = results[-1]
        assert final_result.data is not None
        assert len(final_result.data) <= 4  # Max 4 departments
        assert 'department' in final_result.data.columns
        assert 'salary_k' in final_result.data.columns
        assert 'age' in final_result.data.columns
        
        # Verify performance metrics
        total_time = sum(result.execution_time for result in results)
        assert total_time > 0
        
        # Check execution history
        assert len(self.engine.execution_history) == 3
    
    def test_data_quality_pipeline(self):
        """Test pipeline for handling data quality issues."""
        # Define data cleaning pipeline
        transformations = [
            # Remove rows with null IDs
            TransformationConfig(
                name='filter_valid_ids',
                type=TransformationType.FILTER,
                parameters={'condition': 'id.notna()'}
            ),
            # Clean name column
            TransformationConfig(
                name='clean_names',
                type=TransformationType.MAP,
                parameters={
                    'mappings': {
                        'name': 'name.fillna("Unknown").replace("", "Unknown")'
                    }
                }
            ),
            # Convert age to numeric
            TransformationConfig(
                name='convert_age',
                type=TransformationType.CONVERT,
                parameters={
                    'conversions': {
                        'age': 'integer'
                    }
                }
            ),
            # Fill missing salaries with median
            TransformationConfig(
                name='fill_salary',
                type=TransformationType.MAP,
                parameters={
                    'mappings': {
                        'salary': 'salary.fillna(salary.median())'
                    }
                }
            )
        ]
        
        # Execute pipeline
        results = self.engine.execute_transformation_pipeline(self.dirty_dataset, transformations)
        
        # Verify data cleaning results
        assert len(results) == 4
        
        # Check that most transformations succeeded
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 3  # At least 3 should succeed
        
        # Verify final data quality
        if results[-1].success:
            final_data = results[-1].data
            assert final_data['id'].notna().all()  # No null IDs
            assert not final_data['name'].isin(['', None]).any()  # No empty names
    
    def test_performance_optimization_pipeline(self):
        """Test pipeline with performance optimizations."""
        # Create large dataset for performance testing
        large_data = pd.DataFrame({
            'id': range(10000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
            'value': np.random.normal(100, 25, 10000),
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='H')
        })
        
        # Define computationally intensive pipeline
        transformations = [
            TransformationConfig(
                name='complex_calculation',
                type=TransformationType.MAP,
                parameters={
                    'mappings': {
                        'complex_value': 'value * np.sin(value) + np.log(value + 1)'
                    }
                },
                performance_hints={'use_chunking': True}
            ),
            TransformationConfig(
                name='aggregate_by_category',
                type=TransformationType.AGGREGATE,
                parameters={
                    'group_by': ['category'],
                    'aggregations': {
                        'value': ['mean', 'std', 'min', 'max'],
                        'complex_value': ['mean', 'std'],
                        'id': 'count'
                    }
                }
            )
        ]
        
        # Execute with performance monitoring
        start_time = datetime.now()
        results = self.engine.execute_transformation_pipeline(large_data, transformations)
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Verify performance
        assert all(result.success for result in results)
        assert total_time < 30  # Should complete within 30 seconds
        
        # Check performance metrics
        for result in results:
            assert result.execution_time > 0
            assert result.rows_processed > 0
            assert 'performance_metrics' in result.__dict__
    
    def test_custom_transformation_integration(self):
        """Test integration with custom transformations."""
        # Register custom transformation
        def calculate_z_score(data, column='score', **kwargs):
            """Calculate z-score for a column."""
            result = data.copy()
            mean_val = result[column].mean()
            std_val = result[column].std()
            result[f'{column}_zscore'] = (result[column] - mean_val) / std_val
            return result
        
        self.engine.register_custom_transformation('z_score', calculate_z_score)
        
        # Create pipeline with custom transformation
        transformations = [
            TransformationConfig(
                name='calculate_score_zscore',
                type=TransformationType.CUSTOM,
                parameters={
                    'custom_name': 'z_score',
                    'custom_params': {'column': 'score'}
                }
            ),
            TransformationConfig(
                name='filter_outliers',
                type=TransformationType.FILTER,
                parameters={'condition': 'abs(score_zscore) < 2'}
            )
        ]
        
        # Execute pipeline
        results = self.engine.execute_transformation_pipeline(self.large_dataset, transformations)
        
        # Verify custom transformation worked
        assert len(results) == 2
        assert all(result.success for result in results)
        
        # Check that z-score column was created
        first_result = results[0]
        assert 'score_zscore' in first_result.data.columns
        
        # Check that outliers were filtered
        final_result = results[1]
        assert len(final_result.data) < len(self.large_dataset)  # Some rows filtered
    
    def test_join_transformation_integration(self):
        """Test join transformation with real datasets."""
        # Create employee dataset
        employees = pd.DataFrame({
            'emp_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'dept_id': [1, 2, 1, 3, 2]
        })
        
        # Create department dataset
        departments = pd.DataFrame({
            'dept_id': [1, 2, 3, 4],
            'dept_name': ['Engineering', 'Sales', 'Marketing', 'HR'],
            'budget': [1000000, 500000, 300000, 200000]
        })
        
        # Define join transformation
        join_config = TransformationConfig(
            name='join_employee_department',
            type=TransformationType.JOIN,
            parameters={
                'right_data': departments,
                'join_keys': ['dept_id'],
                'join_type': 'left'
            }
        )
        
        # Execute join
        result = self.engine.execute_transformation(employees, join_config)
        
        # Verify join results
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 5  # All employees preserved
        assert 'dept_name' in result.data.columns
        assert 'budget' in result.data.columns
        
        # Verify join correctness
        alice_row = result.data[result.data['name'] == 'Alice'].iloc[0]
        assert alice_row['dept_name'] == 'Engineering'
        assert alice_row['budget'] == 1000000
    
    def test_type_conversion_integration(self):
        """Test comprehensive type conversion scenarios."""
        # Create dataset with mixed types
        mixed_data = pd.DataFrame({
            'string_numbers': ['1', '2', '3', '4', '5'],
            'float_strings': ['1.5', '2.5', '3.5', '4.5', '5.5'],
            'date_strings': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'boolean_ints': [1, 0, 1, 0, 1],
            'mixed_quality': ['1', '2', 'invalid', '4', '5']
        })
        
        # Define type conversion pipeline
        conversion_config = TransformationConfig(
            name='convert_types',
            type=TransformationType.CONVERT,
            parameters={
                'conversions': {
                    'string_numbers': 'integer',
                    'float_strings': 'float',
                    'date_strings': 'datetime',
                    'boolean_ints': 'boolean',
                    'mixed_quality': 'integer'
                }
            }
        )
        
        # Execute conversion
        result = self.engine.execute_transformation(mixed_data, conversion_config)
        
        # Verify conversions
        assert result.success is True
        assert result.data is not None
        
        # Check successful conversions
        assert result.data['string_numbers'].dtype == 'Int64'
        assert result.data['float_strings'].dtype == 'float64'
        assert pd.api.types.is_datetime64_any_dtype(result.data['date_strings'])
        assert result.data['boolean_ints'].dtype == 'bool'
        
        # Check partial conversion (mixed_quality should have nulls)
        assert result.data['mixed_quality'].dtype == 'Int64'
        assert result.data['mixed_quality'].isna().sum() == 1  # 'invalid' becomes null
    
    def test_recommendation_engine_integration(self):
        """Test integration with recommendation engine."""
        # Get recommendations for dirty dataset
        recommendations = self.engine.get_transformation_recommendations(
            self.dirty_dataset,
            target_schema={
                'id': 'integer',
                'name': 'string',
                'age': 'integer',
                'salary': 'float',
                'email': 'string'
            }
        )
        
        # Verify recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend type conversions
        conversion_recs = [r for r in recommendations if r['type'] == 'conversion']
        assert len(conversion_recs) > 0
        
        # Should recommend filtering for high null columns
        filter_recs = [r for r in recommendations if r['type'] == 'filter']
        assert len(filter_recs) >= 0  # May or may not have filter recommendations
    
    def test_error_handling_and_recovery(self):
        """Test error handling and pipeline recovery."""
        # Create pipeline with intentional errors
        transformations = [
            TransformationConfig(
                name='valid_filter',
                type=TransformationType.FILTER,
                parameters={'condition': 'age > 20'}
            ),
            TransformationConfig(
                name='invalid_map',
                type=TransformationType.MAP,
                parameters={
                    'mappings': {
                        'invalid_column': 'nonexistent_column * 2'  # This will fail
                    }
                }
            ),
            TransformationConfig(
                name='another_valid_transform',
                type=TransformationType.AGGREGATE,
                parameters={
                    'aggregations': {'age': 'mean'}
                }
            )
        ]
        
        # Execute pipeline
        results = self.engine.execute_transformation_pipeline(self.large_dataset, transformations)
        
        # Verify error handling
        assert len(results) == 2  # Should stop after error
        assert results[0].success is True  # First transformation succeeds
        assert results[1].success is False  # Second transformation fails
        assert results[1].error_message is not None
        
        # Verify execution history includes both successful and failed attempts
        history_entries = [h for h in self.engine.execution_history if h['transformation_name'] in ['valid_filter', 'invalid_map']]
        assert len(history_entries) == 2
        assert any(h['success'] for h in history_entries)  # At least one success
        assert any(not h['success'] for h in history_entries)  # At least one failure


class TestTransformationAPIIntegration:
    """Integration tests for transformation API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data_csv = """id,name,age,salary,department
1,Alice,25,50000,IT
2,Bob,30,60000,HR
3,Charlie,35,70000,IT
4,David,40,80000,Finance
5,Eve,45,90000,IT"""
    
    def create_test_file(self, content, filename="test.csv"):
        """Create a temporary test file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'.{filename.split(".")[-1]}', delete=False)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    def test_execute_transformation_api(self):
        """Test the execute transformation API endpoint."""
        # Create test file
        test_file_path = self.create_test_file(self.test_data_csv)
        
        try:
            # Prepare transformation config
            transformation_config = {
                "name": "filter_it_department",
                "type": "filter",
                "parameters": {
                    "condition": "department == 'IT'"
                }
            }
            
            # Make API request
            with open(test_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/transformations/execute",
                    data={"transformation_config": str(transformation_config).replace("'", '"')},
                    files={"data_file": ("test.csv", f, "text/csv")}
                )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            
            assert result['success'] is True
            assert result['rows_processed'] == 5
            assert result['rows_output'] == 3  # 3 IT employees
            assert 'output_data' in result
            assert len(result['output_data']) == 3
            
        finally:
            # Clean up
            os.unlink(test_file_path)
    
    def test_execute_pipeline_api(self):
        """Test the execute pipeline API endpoint."""
        # Create test file
        test_file_path = self.create_test_file(self.test_data_csv)
        
        try:
            # Prepare transformation pipeline
            transformations = [
                {
                    "name": "filter_high_salary",
                    "type": "filter",
                    "parameters": {
                        "condition": "salary > 60000"
                    }
                },
                {
                    "name": "calculate_salary_k",
                    "type": "map",
                    "parameters": {
                        "mappings": {
                            "salary_k": "salary / 1000"
                        }
                    }
                }
            ]
            
            # Make API request
            with open(test_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/transformations/execute-pipeline",
                    json=transformations,
                    files={"data_file": ("test.csv", f, "text/csv")}
                )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            
            assert result['overall_success'] is True
            assert len(result['pipeline_results']) == 2
            assert 'final_output' in result
            assert 'salary_k' in result['final_schema']
            
        finally:
            # Clean up
            os.unlink(test_file_path)
    
    def test_get_recommendations_api(self):
        """Test the get recommendations API endpoint."""
        # Create test file
        test_file_path = self.create_test_file(self.test_data_csv)
        
        try:
            # Make API request
            with open(test_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/transformations/recommendations",
                    files={"data_file": ("test.csv", f, "text/csv")}
                )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            
            assert 'data_profile' in result
            assert 'recommendations' in result
            assert 'total_recommendations' in result
            
            # Check data profile
            profile = result['data_profile']
            assert profile['row_count'] == 5
            assert profile['column_count'] == 5
            assert 'column_types' in profile
            assert 'column_stats' in profile
            
        finally:
            # Clean up
            os.unlink(test_file_path)
    
    def test_validate_conversion_api(self):
        """Test the validate conversion API endpoint."""
        # Test data for conversion
        test_data = ['1', '2', '3', '4', '5']
        
        # Make API request
        response = client.post(
            "/api/v1/transformations/validate-conversion",
            json={
                "column_data": test_data,
                "target_type": "integer"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        
        assert result['convertible'] is True
        assert result['success_rate'] == 1.0
        assert result['null_count'] == 0
    
    def test_get_templates_api(self):
        """Test the get templates API endpoint."""
        # Make API request
        response = client.get("/api/v1/transformations/templates")
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        
        assert 'templates' in result
        assert 'total_templates' in result
        assert len(result['templates']) > 0
        
        # Check template structure
        template = result['templates'][0]
        assert 'id' in template
        assert 'name' in template
        assert 'description' in template
        assert 'type' in template
        assert 'template_config' in template
    
    def test_health_check_api(self):
        """Test the health check API endpoint."""
        # Make API request
        response = client.get("/api/v1/transformations/health")
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        
        assert result['status'] == 'healthy'
        assert result['engine_ready'] is True
        assert 'test_execution_time' in result
        assert 'registered_custom_transformations' in result
        assert 'execution_history_count' in result


if __name__ == '__main__':
    pytest.main([__file__])