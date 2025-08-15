"""
Unit tests for Data Ingestion Service.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ai_data_readiness.core.data_ingestion_service import DataIngestionService
from ai_data_readiness.core.exceptions import DataIngestionError
from ai_data_readiness.core.config import Config


class TestDataIngestionService:
    """Test suite for DataIngestionService."""
    
    @pytest.fixture
    def ingestion_service(self, test_config):
        """Create DataIngestionService instance for testing."""
        return DataIngestionService(test_config)
    
    def test_init(self, test_config):
        """Test DataIngestionService initialization."""
        service = DataIngestionService(test_config)
        assert service.config == test_config
        assert hasattr(service, 'supported_formats')
        assert 'csv' in service.supported_formats
        assert 'json' in service.supported_formats
        assert 'parquet' in service.supported_formats
    
    def test_ingest_csv_data(self, ingestion_service, sample_csv_data, temp_directory):
        """Test CSV data ingestion."""
        # Save sample data to CSV
        csv_file = temp_directory / "test_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        # Test ingestion
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result is not None
        assert 'dataset_id' in result
        assert 'rows' in result
        assert 'columns' in result
        assert result['rows'] == len(sample_csv_data)
        assert result['columns'] == len(sample_csv_data.columns)
    
    def test_ingest_json_data(self, ingestion_service, sample_csv_data, temp_directory):
        """Test JSON data ingestion."""
        # Save sample data to JSON
        json_file = temp_directory / "test_data.json"
        sample_csv_data.to_json(json_file, orient='records')
        
        source_config = {
            'source_type': 'file',
            'file_path': str(json_file),
            'format': 'json'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result is not None
        assert result['rows'] == len(sample_csv_data)
        assert result['columns'] == len(sample_csv_data.columns)
    
    def test_ingest_parquet_data(self, ingestion_service, sample_csv_data, temp_directory):
        """Test Parquet data ingestion."""
        # Save sample data to Parquet
        parquet_file = temp_directory / "test_data.parquet"
        sample_csv_data.to_parquet(parquet_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(parquet_file),
            'format': 'parquet'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result is not None
        assert result['rows'] == len(sample_csv_data)
        assert result['columns'] == len(sample_csv_data.columns)
    
    def test_schema_inference(self, ingestion_service, sample_csv_data, temp_directory):
        """Test automatic schema inference."""
        csv_file = temp_directory / "test_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert 'schema' in result
        schema = result['schema']
        
        # Check that schema contains expected data types
        assert 'id' in schema
        assert 'name' in schema
        assert 'age' in schema
        assert 'income' in schema
        
        # Verify data type inference
        assert 'int' in str(schema['id']).lower()
        assert 'object' in str(schema['name']).lower() or 'string' in str(schema['name']).lower()
    
    def test_data_validation(self, ingestion_service, temp_directory):
        """Test data validation during ingestion."""
        # Create invalid CSV data
        invalid_data = pd.DataFrame({
            'id': [1, 2, 'invalid', 4],
            'value': [1.0, 2.0, 3.0, 'invalid']
        })
        
        csv_file = temp_directory / "invalid_data.csv"
        invalid_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'validation_rules': [
                {'column': 'id', 'type': 'integer'},
                {'column': 'value', 'type': 'float'}
            ]
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        # Should still ingest but report validation issues
        assert result is not None
        assert 'validation_issues' in result
        assert len(result['validation_issues']) > 0
    
    def test_sampling_configuration(self, ingestion_service, sample_csv_data, temp_directory):
        """Test data sampling during ingestion."""
        # Create larger dataset
        large_data = pd.concat([sample_csv_data] * 10, ignore_index=True)
        csv_file = temp_directory / "large_data.csv"
        large_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'sampling_config': {
                'method': 'random',
                'sample_size': 100
            }
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result is not None
        assert result['rows'] == 100  # Should be sampled
        assert 'is_sampled' in result
        assert result['is_sampled'] is True
    
    def test_streaming_data_ingestion(self, ingestion_service):
        """Test streaming data ingestion setup."""
        stream_config = {
            'source_type': 'kafka',
            'topic': 'test_topic',
            'bootstrap_servers': 'localhost:9092'
        }
        
        # Mock the streaming setup
        with patch.object(ingestion_service, '_setup_kafka_consumer') as mock_setup:
            mock_setup.return_value = Mock()
            
            result = ingestion_service.ingest_streaming_data(stream_config)
            
            assert result is not None
            assert hasattr(result, 'start')
            assert hasattr(result, 'stop')
            mock_setup.assert_called_once()
    
    def test_metadata_extraction(self, ingestion_service, sample_csv_data, temp_directory):
        """Test metadata extraction during ingestion."""
        csv_file = temp_directory / "test_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert 'metadata' in result
        metadata = result['metadata']
        
        # Check metadata structure
        assert 'file_size' in metadata
        assert 'created_at' in metadata
        assert 'column_stats' in metadata
        
        # Verify column statistics
        column_stats = metadata['column_stats']
        assert 'age' in column_stats
        assert 'min' in column_stats['age']
        assert 'max' in column_stats['age']
        assert 'mean' in column_stats['age']
    
    def test_error_handling_invalid_file(self, ingestion_service):
        """Test error handling for invalid file paths."""
        source_config = {
            'source_type': 'file',
            'file_path': '/nonexistent/file.csv',
            'format': 'csv'
        }
        
        with pytest.raises(DataIngestionError):
            ingestion_service.ingest_batch_data(source_config)
    
    def test_error_handling_unsupported_format(self, ingestion_service, temp_directory):
        """Test error handling for unsupported file formats."""
        # Create a file with unsupported extension
        unsupported_file = temp_directory / "test_data.xyz"
        unsupported_file.write_text("some data")
        
        source_config = {
            'source_type': 'file',
            'file_path': str(unsupported_file),
            'format': 'xyz'
        }
        
        with pytest.raises(DataIngestionError):
            ingestion_service.ingest_batch_data(source_config)
    
    def test_error_handling_corrupted_data(self, ingestion_service, temp_directory):
        """Test error handling for corrupted data files."""
        # Create corrupted CSV file
        corrupted_file = temp_directory / "corrupted.csv"
        corrupted_file.write_text("invalid,csv,data\n1,2\n3,4,5,6")
        
        source_config = {
            'source_type': 'file',
            'file_path': str(corrupted_file),
            'format': 'csv'
        }
        
        with pytest.raises(DataIngestionError):
            ingestion_service.ingest_batch_data(source_config)
    
    def test_batch_processing_large_files(self, ingestion_service, temp_directory):
        """Test batch processing for large files."""
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'value': np.random.random(10000)
        })
        
        csv_file = temp_directory / "large_data.csv"
        large_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'batch_size': 1000
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result is not None
        assert result['rows'] == 10000
        assert 'processing_batches' in result
        assert result['processing_batches'] == 10
    
    def test_concurrent_ingestion(self, ingestion_service, sample_csv_data, temp_directory):
        """Test concurrent data ingestion."""
        # Create multiple files
        files = []
        for i in range(3):
            csv_file = temp_directory / f"test_data_{i}.csv"
            sample_csv_data.to_csv(csv_file, index=False)
            files.append(csv_file)
        
        source_configs = [
            {
                'source_type': 'file',
                'file_path': str(file),
                'format': 'csv'
            }
            for file in files
        ]
        
        results = ingestion_service.ingest_batch_data_concurrent(source_configs)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'dataset_id' in result
            assert result['rows'] == len(sample_csv_data)