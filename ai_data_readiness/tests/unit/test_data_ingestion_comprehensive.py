"""
Comprehensive unit tests for Data Ingestion Service.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ai_data_readiness.core.data_ingestion_service import DataIngestionService
from ai_data_readiness.core.exceptions import AIDataReadinessError


class TestDataIngestionServiceComprehensive:
    """Comprehensive test suite for DataIngestionService."""
    
    @pytest.fixture
    def ingestion_service(self, test_config):
        """Create DataIngestionService instance for testing."""
        return DataIngestionService(test_config)
    
    def test_init_with_config(self, test_config):
        """Test DataIngestionService initialization with configuration."""
        service = DataIngestionService(test_config)
        assert service.config == test_config
        assert hasattr(service, 'supported_formats')
        assert hasattr(service, 'batch_size')
        assert hasattr(service, 'validation_rules')
    
    def test_ingest_csv_file(self, ingestion_service, sample_csv_data, temp_directory):
        """Test CSV file ingestion."""
        csv_file = temp_directory / "test_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'delimiter': ',',
            'encoding': 'utf-8'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result is not None
        assert 'dataset_id' in result
        assert 'rows' in result
        assert 'columns' in result
        assert 'schema' in result
        assert 'metadata' in result
        
        assert result['rows'] == len(sample_csv_data)
        assert result['columns'] == len(sample_csv_data.columns)
        
        # Verify schema detection
        schema = result['schema']
        assert 'id' in schema
        assert 'name' in schema
        assert 'age' in schema
        assert 'income' in schema
    
    def test_ingest_json_file(self, ingestion_service, sample_csv_data, temp_directory):
        """Test JSON file ingestion."""
        json_file = temp_directory / "test_data.json"
        sample_csv_data.to_json(json_file, orient='records')
        
        source_config = {
            'source_type': 'file',
            'file_path': str(json_file),
            'format': 'json',
            'orient': 'records'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result['rows'] == len(sample_csv_data)
        assert result['columns'] == len(sample_csv_data.columns)
    
    def test_ingest_parquet_file(self, ingestion_service, sample_csv_data, temp_directory):
        """Test Parquet file ingestion."""
        parquet_file = temp_directory / "test_data.parquet"
        sample_csv_data.to_parquet(parquet_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(parquet_file),
            'format': 'parquet'
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result['rows'] == len(sample_csv_data)
        assert result['columns'] == len(sample_csv_data.columns)
    
    def test_batch_processing_large_file(self, ingestion_service, temp_directory):
        """Test batch processing of large files."""
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(5000),
            'value': np.random.random(5000),
            'category': np.random.choice(['A', 'B', 'C'], 5000)
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
        
        assert result['rows'] == 5000
        assert 'processing_batches' in result
        assert result['processing_batches'] == 5  # 5000 / 1000
    
    def test_schema_validation(self, ingestion_service, temp_directory):
        """Test schema validation during ingestion."""
        # Create data with schema issues
        problematic_data = pd.DataFrame({
            'id': ['1', '2', 'invalid', '4'],  # Mixed types
            'amount': [100.5, 200.0, 'invalid', 400.0],  # Mixed types
            'date': ['2023-01-01', '2023-01-02', 'invalid-date', '2023-01-04']
        })
        
        csv_file = temp_directory / "problematic_data.csv"
        problematic_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'validate_schema': True
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert 'schema_issues' in result
        assert 'data_type_inconsistencies' in result['schema_issues']
    
    def test_metadata_extraction(self, ingestion_service, sample_csv_data, temp_directory):
        """Test metadata extraction during ingestion."""
        csv_file = temp_directory / "metadata_test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'extract_metadata': True
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        metadata = result['metadata']
        assert 'file_size' in metadata
        assert 'creation_time' in metadata
        assert 'column_statistics' in metadata
        
        # Check column statistics
        col_stats = metadata['column_statistics']
        assert 'age' in col_stats
        assert 'mean' in col_stats['age']
        assert 'std' in col_stats['age']
        assert 'min' in col_stats['age']
        assert 'max' in col_stats['age']
    
    def test_data_sampling(self, ingestion_service, temp_directory):
        """Test data sampling for large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'value': np.random.random(10000)
        })
        
        csv_file = temp_directory / "sampling_test.csv"
        large_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'sampling_config': {
                'method': 'random',
                'sample_size': 1000,
                'seed': 42
            }
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert result['rows'] == 1000  # Sampled size
        assert 'original_rows' in result
        assert result['original_rows'] == 10000
        assert 'sampling_applied' in result
        assert result['sampling_applied'] is True
    
    def test_streaming_data_ingestion(self, ingestion_service):
        """Test streaming data ingestion setup."""
        stream_config = {
            'source_type': 'kafka',
            'topic': 'test_topic',
            'bootstrap_servers': 'localhost:9092',
            'consumer_group': 'test_group'
        }
        
        # Mock Kafka consumer
        with patch('ai_data_readiness.core.data_ingestion_service.KafkaConsumer') as mock_consumer:
            mock_consumer.return_value = Mock()
            
            stream_handler = ingestion_service.ingest_streaming_data(stream_config)
            
            assert stream_handler is not None
            assert hasattr(stream_handler, 'start')
            assert hasattr(stream_handler, 'stop')
    
    def test_database_ingestion(self, ingestion_service):
        """Test database ingestion."""
        db_config = {
            'source_type': 'database',
            'connection_string': 'postgresql://user:pass@localhost/db',
            'query': 'SELECT * FROM test_table',
            'batch_size': 1000
        }
        
        # Mock database connection
        with patch('ai_data_readiness.core.data_ingestion_service.create_engine') as mock_engine:
            mock_connection = Mock()
            mock_engine.return_value.connect.return_value = mock_connection
            
            # Mock query result
            mock_result = pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['A', 'B', 'C']
            })
            
            with patch('pandas.read_sql') as mock_read_sql:
                mock_read_sql.return_value = mock_result
                
                result = ingestion_service.ingest_batch_data(db_config)
                
                assert result['rows'] == 3
                assert result['columns'] == 2
    
    def test_api_ingestion(self, ingestion_service):
        """Test API data ingestion."""
        api_config = {
            'source_type': 'api',
            'url': 'https://api.example.com/data',
            'method': 'GET',
            'headers': {'Authorization': 'Bearer token'},
            'pagination': {
                'type': 'offset',
                'limit': 100
            }
        }
        
        # Mock API response
        mock_response_data = [
            {'id': 1, 'name': 'Item 1'},
            {'id': 2, 'name': 'Item 2'}
        ]
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = ingestion_service.ingest_batch_data(api_config)
            
            assert result['rows'] == 2
            assert result['columns'] == 2
    
    def test_error_handling_file_not_found(self, ingestion_service):
        """Test error handling for missing files."""
        source_config = {
            'source_type': 'file',
            'file_path': '/nonexistent/file.csv',
            'format': 'csv'
        }
        
        with pytest.raises(AIDataReadinessError) as exc_info:
            ingestion_service.ingest_batch_data(source_config)
        
        assert "File not found" in str(exc_info.value)
    
    def test_error_handling_invalid_format(self, ingestion_service, temp_directory):
        """Test error handling for invalid file formats."""
        # Create invalid CSV file
        invalid_file = temp_directory / "invalid.csv"
        with open(invalid_file, 'w') as f:
            f.write("invalid,csv,content\nwith,mismatched,columns,extra")
        
        source_config = {
            'source_type': 'file',
            'file_path': str(invalid_file),
            'format': 'csv'
        }
        
        with pytest.raises(AIDataReadinessError) as exc_info:
            ingestion_service.ingest_batch_data(source_config)
        
        assert "parsing" in str(exc_info.value).lower()
    
    def test_error_handling_unsupported_format(self, ingestion_service, temp_directory):
        """Test error handling for unsupported formats."""
        source_config = {
            'source_type': 'file',
            'file_path': str(temp_directory / "test.xyz"),
            'format': 'unsupported_format'
        }
        
        with pytest.raises(AIDataReadinessError) as exc_info:
            ingestion_service.ingest_batch_data(source_config)
        
        assert "Unsupported format" in str(exc_info.value)
    
    def test_validate_schema_method(self, ingestion_service):
        """Test schema validation method."""
        dataset_id = "test_dataset"
        
        expected_schema = {
            'id': 'int64',
            'name': 'object',
            'age': 'int64',
            'income': 'float64'
        }
        
        # Mock dataset retrieval
        with patch.object(ingestion_service, '_get_dataset_schema') as mock_get_schema:
            mock_get_schema.return_value = expected_schema
            
            result = ingestion_service.validate_schema(dataset_id, expected_schema)
            
            assert result['valid'] is True
            assert result['schema_matches'] is True
    
    def test_validate_schema_mismatch(self, ingestion_service):
        """Test schema validation with mismatches."""
        dataset_id = "test_dataset"
        
        expected_schema = {
            'id': 'int64',
            'name': 'object',
            'age': 'int64'
        }
        
        actual_schema = {
            'id': 'object',  # Type mismatch
            'name': 'object',
            'age': 'int64',
            'extra_column': 'float64'  # Extra column
        }
        
        with patch.object(ingestion_service, '_get_dataset_schema') as mock_get_schema:
            mock_get_schema.return_value = actual_schema
            
            result = ingestion_service.validate_schema(dataset_id, expected_schema)
            
            assert result['valid'] is False
            assert 'schema_mismatches' in result
            assert len(result['schema_mismatches']) > 0
    
    def test_extract_metadata_method(self, ingestion_service, sample_csv_data):
        """Test metadata extraction method."""
        dataset_id = "test_dataset"
        
        with patch.object(ingestion_service, '_get_dataset') as mock_get_dataset:
            mock_get_dataset.return_value = sample_csv_data
            
            metadata = ingestion_service.extract_metadata(dataset_id)
            
            assert 'dataset_id' in metadata
            assert 'row_count' in metadata
            assert 'column_count' in metadata
            assert 'column_types' in metadata
            assert 'missing_values' in metadata
            assert 'data_statistics' in metadata
            
            assert metadata['row_count'] == len(sample_csv_data)
            assert metadata['column_count'] == len(sample_csv_data.columns)
    
    def test_performance_monitoring(self, ingestion_service, temp_directory, performance_timer):
        """Test performance monitoring during ingestion."""
        # Create medium-sized dataset
        data = pd.DataFrame({
            'id': range(5000),
            'value': np.random.random(5000),
            'category': np.random.choice(['A', 'B', 'C'], 5000)
        })
        
        csv_file = temp_directory / "performance_test.csv"
        data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'monitor_performance': True
        }
        
        with performance_timer() as timer:
            result = ingestion_service.ingest_batch_data(source_config)
        
        assert result is not None
        assert 'performance_metrics' in result
        
        perf_metrics = result['performance_metrics']
        assert 'ingestion_time' in perf_metrics
        assert 'rows_per_second' in perf_metrics
        assert 'memory_usage' in perf_metrics
        
        # Verify reasonable performance
        assert perf_metrics['rows_per_second'] > 100  # At least 100 rows/sec
    
    def test_concurrent_ingestion(self, ingestion_service, temp_directory):
        """Test concurrent data ingestion."""
        import threading
        import time
        
        # Create multiple test files
        files = []
        for i in range(3):
            data = pd.DataFrame({
                'id': range(i*100, (i+1)*100),
                'value': np.random.random(100)
            })
            
            csv_file = temp_directory / f"concurrent_test_{i}.csv"
            data.to_csv(csv_file, index=False)
            files.append(csv_file)
        
        results = {}
        threads = []
        
        def ingest_file(file_path, file_id):
            source_config = {
                'source_type': 'file',
                'file_path': str(file_path),
                'format': 'csv'
            }
            results[file_id] = ingestion_service.ingest_batch_data(source_config)
        
        # Start concurrent ingestion
        for i, file_path in enumerate(files):
            thread = threading.Thread(target=ingest_file, args=(file_path, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all ingestions completed successfully
        assert len(results) == 3
        for i in range(3):
            assert results[i]['rows'] == 100
            assert results[i]['columns'] == 2
    
    def test_data_lineage_tracking(self, ingestion_service, sample_csv_data, temp_directory):
        """Test data lineage tracking during ingestion."""
        csv_file = temp_directory / "lineage_test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'track_lineage': True,
            'source_metadata': {
                'system': 'test_system',
                'owner': 'test_user',
                'purpose': 'unit_testing'
            }
        }
        
        result = ingestion_service.ingest_batch_data(source_config)
        
        assert 'lineage' in result
        lineage = result['lineage']
        
        assert 'source_system' in lineage
        assert 'ingestion_timestamp' in lineage
        assert 'transformations_applied' in lineage
        assert 'data_owner' in lineage
        
        assert lineage['source_system'] == 'test_system'
        assert lineage['data_owner'] == 'test_user'