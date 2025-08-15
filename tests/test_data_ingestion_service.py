"""
Unit tests for Data Ingestion Service

Tests cover batch and streaming data ingestion, schema detection,
validation, and error handling scenarios.
"""

import pytest
import pandas as pd
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ai_data_readiness.core.data_ingestion_service import (
    DataIngestionService,
    SourceConfig,
    StreamConfig,
    StreamHandler,
    DataFormat,
    ProcessingMode
)
from ai_data_readiness.core.exceptions import DataIngestionError, SchemaValidationError
from ai_data_readiness.models.base_models import Schema, ColumnSchema, ValidationResult


class TestDataIngestionService:
    """Test suite for DataIngestionService"""
    
    @pytest.fixture
    def ingestion_service(self):
        """Create DataIngestionService instance for testing"""
        return DataIngestionService()
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing"""
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
            'active': [True, True, False, True, True]
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_json_file(self):
        """Create a temporary JSON file for testing"""
        data = [
            {'id': 1, 'name': 'Alice', 'age': 25, 'salary': 50000.0, 'active': True},
            {'id': 2, 'name': 'Bob', 'age': 30, 'salary': 60000.0, 'active': True},
            {'id': 3, 'name': 'Charlie', 'age': 35, 'salary': 70000.0, 'active': False}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_ingest_batch_csv_data(self, ingestion_service, sample_csv_file):
        """Test batch ingestion of CSV data"""
        source_config = SourceConfig(
            source_type="file",
            connection_params={'file_path': sample_csv_file},
            data_format=DataFormat.CSV
        )
        
        with patch.object(ingestion_service, '_store_dataset', new_callable=AsyncMock):
            metadata = await ingestion_service.ingest_batch_data(source_config)
        
        assert metadata is not None
        assert metadata.row_count == 5
        assert metadata.column_count == 5
        assert metadata.processing_mode == ProcessingMode.BATCH.value
        assert len(metadata.schema.columns) == 5
        
        # Check schema detection
        column_names = [col.name for col in metadata.schema.columns]
        assert 'id' in column_names
        assert 'name' in column_names
        assert 'age' in column_names
        assert 'salary' in column_names
        assert 'active' in column_names
    
    @pytest.mark.asyncio
    async def test_ingest_batch_json_data(self, ingestion_service, sample_json_file):
        """Test batch ingestion of JSON data"""
        source_config = SourceConfig(
            source_type="file",
            connection_params={'file_path': sample_json_file},
            data_format=DataFormat.JSON
        )
        
        with patch.object(ingestion_service, '_store_dataset', new_callable=AsyncMock):
            metadata = await ingestion_service.ingest_batch_data(source_config)
        
        assert metadata is not None
        assert metadata.row_count == 3
        assert metadata.column_count == 5
        assert metadata.processing_mode == ProcessingMode.BATCH.value
    
    @pytest.mark.asyncio
    async def test_schema_detection_data_types(self, ingestion_service):
        """Test automatic schema detection for different data types"""
        # Create DataFrame with various data types
        data = {
            'integer_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'string_col': ['a', 'b', 'c'],
            'boolean_col': [True, False, True],
            'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        }
        df = pd.DataFrame(data)
        
        schema = await ingestion_service._detect_schema(df)
        
        assert len(schema.columns) == 5
        
        # Check data type detection
        col_types = {col.name: col.data_type for col in schema.columns}
        assert col_types['integer_col'] == 'integer'
        assert col_types['float_col'] == 'float'
        assert col_types['string_col'] == 'string'
        assert col_types['boolean_col'] == 'boolean'
        assert col_types['datetime_col'] == 'datetime'
    
    @pytest.mark.asyncio
    async def test_data_validation_rules(self, ingestion_service):
        """Test data validation against specified rules"""
        # Create DataFrame with validation issues
        data = {
            'id': [1, 2, None, 4, 5],  # Has null value
            'name': ['Alice', 'Bob', 'Alice', 'David', 'Eve'],  # Has duplicate
            'age': [25, 30, 150, 28, -5]  # Has values outside range
        }
        df = pd.DataFrame(data)
        
        validation_rules = [
            {'type': 'not_null', 'columns': ['id']},
            {'type': 'unique', 'columns': ['name']},
            {'type': 'range', 'column': 'age', 'min': 0, 'max': 120}
        ]
        
        result = await ingestion_service._validate_data(df, validation_rules)
        
        assert not result.is_valid
        assert len(result.errors) == 3  # null, duplicate, and range violations
        assert any('null values' in error for error in result.errors)
        assert any('duplicate values' in error for error in result.errors)
        assert any('below minimum' in error for error in result.errors)
        assert any('above maximum' in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_data_sampling(self, ingestion_service):
        """Test data sampling functionality"""
        # Create large DataFrame
        data = {'col1': range(1000), 'col2': range(1000, 2000)}
        df = pd.DataFrame(data)
        
        # Test random sampling with absolute count
        sampling_config = {'method': 'random', 'sample_size': 100}
        sampled_df = ingestion_service._apply_sampling(df, sampling_config)
        assert len(sampled_df) == 100
        
        # Test random sampling with fraction
        sampling_config = {'method': 'random', 'sample_size': 0.1}
        sampled_df = ingestion_service._apply_sampling(df, sampling_config)
        assert len(sampled_df) == 100  # 10% of 1000
        
        # Test head sampling
        sampling_config = {'method': 'head', 'sample_size': 50}
        sampled_df = ingestion_service._apply_sampling(df, sampling_config)
        assert len(sampled_df) == 50
        assert sampled_df.iloc[0]['col1'] == 0  # First row
        
        # Test tail sampling
        sampling_config = {'method': 'tail', 'sample_size': 50}
        sampled_df = ingestion_service._apply_sampling(df, sampling_config)
        assert len(sampled_df) == 50
        assert sampled_df.iloc[-1]['col1'] == 999  # Last row
    
    @pytest.mark.asyncio
    async def test_schema_validation_success(self, ingestion_service):
        """Test successful schema validation"""
        # Mock dataset metadata
        expected_schema = Schema(columns=[
            ColumnSchema(name='id', data_type='integer', nullable=False),
            ColumnSchema(name='name', data_type='string', nullable=False)
        ])
        
        actual_metadata = Mock()
        actual_metadata.schema = expected_schema
        
        with patch.object(ingestion_service, '_load_dataset_metadata', return_value=actual_metadata):
            result = await ingestion_service.validate_schema('test_dataset', expected_schema)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_schema_validation_failure(self, ingestion_service):
        """Test schema validation with mismatches"""
        expected_schema = Schema(columns=[
            ColumnSchema(name='id', data_type='integer', nullable=False),
            ColumnSchema(name='name', data_type='string', nullable=False)
        ])
        
        actual_schema = Schema(columns=[
            ColumnSchema(name='id', data_type='string', nullable=False),  # Type mismatch
            ColumnSchema(name='email', data_type='string', nullable=False)  # Different column
        ])
        
        actual_metadata = Mock()
        actual_metadata.schema = actual_schema
        
        with patch.object(ingestion_service, '_load_dataset_metadata', return_value=actual_metadata):
            result = await ingestion_service.validate_schema('test_dataset', expected_schema)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert len(result.warnings) > 0
    
    @pytest.mark.asyncio
    async def test_unsupported_format_error(self, ingestion_service):
        """Test error handling for unsupported data formats"""
        source_config = SourceConfig(
            source_type="file",
            connection_params={'file_path': 'test.xyz'},
            data_format="unsupported_format"  # This will cause an error
        )
        
        with pytest.raises(DataIngestionError):
            await ingestion_service.ingest_batch_data(source_config)
    
    @pytest.mark.asyncio
    async def test_missing_file_error(self, ingestion_service):
        """Test error handling for missing files"""
        source_config = SourceConfig(
            source_type="file",
            connection_params={'file_path': 'nonexistent_file.csv'},
            data_format=DataFormat.CSV
        )
        
        with pytest.raises(DataIngestionError):
            await ingestion_service.ingest_batch_data(source_config)
    
    @pytest.mark.asyncio
    async def test_streaming_data_setup(self, ingestion_service):
        """Test streaming data ingestion setup"""
        source_config = SourceConfig(
            source_type="stream",
            connection_params={'stream_endpoint': 'test://stream'},
            data_format=DataFormat.JSON
        )
        
        stream_config = StreamConfig(
            stream_name="test_stream",
            source_config=source_config,
            buffer_size=1000
        )
        
        handler = await ingestion_service.ingest_streaming_data(stream_config)
        
        assert isinstance(handler, StreamHandler)
        assert handler.config == stream_config
        assert handler.ingestion_service == ingestion_service


class TestStreamHandler:
    """Test suite for StreamHandler"""
    
    @pytest.fixture
    def stream_handler(self):
        """Create StreamHandler instance for testing"""
        source_config = SourceConfig(
            source_type="stream",
            connection_params={'stream_endpoint': 'test://stream'},
            data_format=DataFormat.JSON
        )
        
        stream_config = StreamConfig(
            stream_name="test_stream",
            source_config=source_config
        )
        
        ingestion_service = Mock()
        return StreamHandler(stream_config, ingestion_service)
    
    @pytest.mark.asyncio
    async def test_stream_handler_start_stop(self, stream_handler):
        """Test stream handler start and stop functionality"""
        assert not stream_handler.is_running
        
        await stream_handler.start()
        assert stream_handler.is_running
        
        await stream_handler.stop()
        assert not stream_handler.is_running
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, stream_handler):
        """Test successful batch processing in stream handler"""
        data_batch = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'}
        ]
        
        # Mock the validation method
        mock_validation_result = ValidationResult(is_valid=True, errors=[], warnings=[])
        stream_handler.ingestion_service._validate_data = AsyncMock(return_value=mock_validation_result)
        
        result = await stream_handler.process_batch(data_batch)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_process_batch_validation_failure(self, stream_handler):
        """Test batch processing with validation failure"""
        data_batch = [
            {'id': None, 'name': 'Alice'},  # Invalid data
        ]
        
        # Mock validation failure
        mock_validation_result = ValidationResult(
            is_valid=False, 
            errors=['Validation failed'], 
            warnings=[]
        )
        stream_handler.ingestion_service._validate_data = AsyncMock(return_value=mock_validation_result)
        
        # Test with continue error handling (default)
        result = await stream_handler.process_batch(data_batch)
        assert result is True  # Should continue despite validation failure
        
        # Test with stop error handling
        stream_handler.config.error_handling = "stop"
        result = await stream_handler.process_batch(data_batch)
        assert result is False  # Should stop on validation failure


@pytest.mark.integration
class TestDataIngestionIntegration:
    """Integration tests for data ingestion service"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_csv_ingestion(self):
        """Test complete CSV ingestion workflow"""
        # Create test CSV file
        data = {
            'user_id': [1, 2, 3],
            'username': ['alice', 'bob', 'charlie'],
            'score': [95.5, 87.2, 92.8],
            'is_active': [True, False, True]
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_file = f.name
        
        try:
            # Configure ingestion
            source_config = SourceConfig(
                source_type="file",
                connection_params={'file_path': csv_file},
                data_format=DataFormat.CSV,
                validation_rules=[
                    {'type': 'not_null', 'columns': ['user_id', 'username']},
                    {'type': 'unique', 'columns': ['user_id']},
                    {'type': 'range', 'column': 'score', 'min': 0, 'max': 100}
                ]
            )
            
            # Perform ingestion
            service = DataIngestionService()
            with patch.object(service, '_store_dataset', new_callable=AsyncMock):
                metadata = await service.ingest_batch_data(source_config)
            
            # Verify results
            assert metadata.row_count == 3
            assert metadata.column_count == 4
            assert metadata.validation_result.is_valid
            
            # Verify schema detection
            schema_types = {col.name: col.data_type for col in metadata.schema.columns}
            assert schema_types['user_id'] == 'integer'
            assert schema_types['username'] == 'string'
            assert schema_types['score'] == 'float'
            assert schema_types['is_active'] == 'boolean'
            
        finally:
            # Cleanup
            Path(csv_file).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__])