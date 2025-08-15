"""
Data Pipeline Integration Tests
Tests data processing pipelines with various file formats and sizes
"""
import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, Mock
import io

from scrollintel.engines.file_processor import FileProcessor
from scrollintel.engines.automodel_engine import AutoModelEngine
from scrollintel.engines.scroll_viz_engine import ScrollVizEngine
from scrollintel.engines.scroll_qa_engine import ScrollQAEngine


class TestDataPipelines:
    """Test data processing pipelines"""
    
    @pytest.fixture
    def file_processor(self):
        """Create file processor instance"""
        return FileProcessor()
    
    @pytest.fixture
    def large_datasets(self):
        """Create large datasets for testing"""
        datasets = {}
        
        # Large CSV dataset (10K rows)
        datasets['large_csv'] = pd.DataFrame({
            'id': range(10000),
            'timestamp': pd.date_range('2020-01-01', periods=10000, freq='H'),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
            'score': np.random.uniform(0, 100, 10000),
            'flag': np.random.choice([True, False], 10000)
        })
        
        # Wide dataset (many columns)
        wide_data = {}
        for i in range(100):
            wide_data[f'feature_{i}'] = np.random.randn(1000)
        wide_data['target'] = np.random.choice([0, 1], 1000)
        datasets['wide_csv'] = pd.DataFrame(wide_data)
        
        # Time series dataset
        dates = pd.date_range('2020-01-01', periods=5000, freq='D')
        datasets['time_series'] = pd.DataFrame({
            'date': dates,
            'sales': np.random.randn(5000).cumsum() + 1000,
            'temperature': 20 + 10 * np.sin(np.arange(5000) * 2 * np.pi / 365) + np.random.randn(5000),
            'humidity': 50 + 20 * np.cos(np.arange(5000) * 2 * np.pi / 365) + np.random.randn(5000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 5000)
        })
        
        return datasets
    
    @pytest.fixture
    def complex_json_data(self):
        """Create complex JSON data structures"""
        return {
            "metadata": {
                "version": "1.0",
                "created": "2024-01-01",
                "source": "test_system"
            },
            "users": [
                {
                    "id": i,
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "profile": {
                        "age": np.random.randint(18, 80),
                        "location": np.random.choice(["NYC", "LA", "Chicago", "Houston"]),
                        "preferences": {
                            "theme": np.random.choice(["dark", "light"]),
                            "notifications": np.random.choice([True, False])
                        }
                    },
                    "activity": [
                        {
                            "date": f"2024-01-{j:02d}",
                            "action": np.random.choice(["login", "view", "purchase", "logout"]),
                            "value": np.random.uniform(0, 100)
                        }
                        for j in range(1, np.random.randint(5, 15))
                    ]
                }
                for i in range(100)
            ],
            "transactions": [
                {
                    "id": f"txn_{i}",
                    "user_id": np.random.randint(0, 99),
                    "amount": np.random.uniform(10, 1000),
                    "currency": np.random.choice(["USD", "EUR", "GBP"]),
                    "timestamp": f"2024-01-{np.random.randint(1, 31):02d}T{np.random.randint(0, 24):02d}:00:00Z"
                }
                for i in range(500)
            ]
        }
    
    @pytest.mark.asyncio
    async def test_csv_processing_pipeline(self, file_processor, large_datasets, temp_files):
        """Test CSV file processing pipeline"""
        # Save large dataset to file
        temp_dir = Path(tempfile.mkdtemp())
        csv_file = temp_dir / "large_data.csv"
        large_datasets['large_csv'].to_csv(csv_file, index=False)
        
        try:
            # Test file detection and parsing
            file_info = await file_processor.detect_file_type(str(csv_file))
            assert file_info['type'] == 'csv'
            assert file_info['size'] > 0
            
            # Test data loading
            data = await file_processor.load_data(str(csv_file))
            assert len(data) == 10000
            assert len(data.columns) == 6
            
            # Test schema inference
            schema = await file_processor.infer_schema(data)
            assert 'id' in schema
            assert schema['id']['type'] == 'integer'
            assert schema['timestamp']['type'] == 'datetime'
            assert schema['value']['type'] == 'float'
            assert schema['category']['type'] == 'string'
            
            # Test data quality validation
            quality_report = await file_processor.validate_data_quality(data)
            assert 'missing_values' in quality_report
            assert 'duplicates' in quality_report
            assert 'outliers' in quality_report
            
            # Test data preview generation
            preview = await file_processor.generate_preview(data)
            assert 'head' in preview
            assert 'tail' in preview
            assert 'summary' in preview
            assert len(preview['head']) <= 10
            
        finally:
            # Cleanup
            os.unlink(csv_file)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_excel_processing_pipeline(self, file_processor, large_datasets):
        """Test Excel file processing pipeline"""
        temp_dir = Path(tempfile.mkdtemp())
        excel_file = temp_dir / "large_data.xlsx"
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            large_datasets['large_csv'].to_excel(writer, sheet_name='main_data', index=False)
            large_datasets['wide_csv'].to_excel(writer, sheet_name='features', index=False)
            large_datasets['time_series'].to_excel(writer, sheet_name='time_series', index=False)
        
        try:
            # Test file detection
            file_info = await file_processor.detect_file_type(str(excel_file))
            assert file_info['type'] == 'excel'
            
            # Test sheet detection
            sheets = await file_processor.get_excel_sheets(str(excel_file))
            assert len(sheets) == 3
            assert 'main_data' in sheets
            assert 'features' in sheets
            assert 'time_series' in sheets
            
            # Test loading specific sheet
            main_data = await file_processor.load_data(str(excel_file), sheet_name='main_data')
            assert len(main_data) == 10000
            
            features_data = await file_processor.load_data(str(excel_file), sheet_name='features')
            assert len(features_data.columns) == 101  # 100 features + target
            
            # Test schema inference for different sheets
            main_schema = await file_processor.infer_schema(main_data)
            features_schema = await file_processor.infer_schema(features_data)
            
            assert len(main_schema) == 6
            assert len(features_schema) == 101
            
        finally:
            # Cleanup
            os.unlink(excel_file)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_json_processing_pipeline(self, file_processor, complex_json_data):
        """Test JSON file processing pipeline"""
        temp_dir = Path(tempfile.mkdtemp())
        json_file = temp_dir / "complex_data.json"
        
        # Save complex JSON data
        with open(json_file, 'w') as f:
            json.dump(complex_json_data, f, indent=2, default=str)
        
        try:
            # Test file detection
            file_info = await file_processor.detect_file_type(str(json_file))
            assert file_info['type'] == 'json'
            
            # Test JSON structure analysis
            structure = await file_processor.analyze_json_structure(str(json_file))
            assert 'metadata' in structure
            assert 'users' in structure
            assert 'transactions' in structure
            
            # Test flattening nested JSON
            flattened_users = await file_processor.flatten_json(
                complex_json_data['users'], 
                prefix='user'
            )
            assert len(flattened_users) == 100
            assert 'user_profile_age' in flattened_users[0]
            
            # Test converting to DataFrame
            users_df = await file_processor.json_to_dataframe(
                complex_json_data['users']
            )
            assert len(users_df) == 100
            assert 'id' in users_df.columns
            assert 'name' in users_df.columns
            
            transactions_df = await file_processor.json_to_dataframe(
                complex_json_data['transactions']
            )
            assert len(transactions_df) == 500
            assert 'amount' in transactions_df.columns
            
        finally:
            # Cleanup
            os.unlink(json_file)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_sql_query_processing_pipeline(self, file_processor):
        """Test SQL query processing pipeline"""
        # Test SQL parsing and validation
        sql_queries = [
            "SELECT * FROM users WHERE age > 25",
            "SELECT COUNT(*) as total_users FROM users GROUP BY region",
            "SELECT u.name, t.amount FROM users u JOIN transactions t ON u.id = t.user_id",
            "CREATE TABLE test_table (id INT, name VARCHAR(50))",
            "INSERT INTO test_table VALUES (1, 'Test')"
        ]
        
        for query in sql_queries:
            # Test SQL parsing
            parsed = await file_processor.parse_sql(query)
            assert 'type' in parsed
            assert 'tables' in parsed
            
            # Test SQL validation
            is_valid = await file_processor.validate_sql(query)
            assert isinstance(is_valid, bool)
            
            # Test SQL to natural language conversion
            description = await file_processor.sql_to_description(query)
            assert isinstance(description, str)
            assert len(description) > 0
    
    @pytest.mark.asyncio
    async def test_large_file_streaming_pipeline(self, file_processor):
        """Test streaming processing of large files"""
        # Create very large dataset
        temp_dir = Path(tempfile.mkdtemp())
        large_file = temp_dir / "very_large.csv"
        
        # Generate large CSV file (50K rows)
        chunk_size = 10000
        total_rows = 50000
        
        # Write file in chunks to simulate large file
        with open(large_file, 'w') as f:
            # Write header
            f.write("id,value,category,timestamp\n")
            
            for chunk_start in range(0, total_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_rows)
                chunk_data = pd.DataFrame({
                    'id': range(chunk_start, chunk_end),
                    'value': np.random.randn(chunk_end - chunk_start),
                    'category': np.random.choice(['A', 'B', 'C'], chunk_end - chunk_start),
                    'timestamp': pd.date_range('2020-01-01', periods=chunk_end - chunk_start, freq='H')
                })
                chunk_data.to_csv(f, header=False, index=False)
        
        try:
            # Test streaming file processing
            chunk_results = []
            async for chunk in file_processor.stream_process_file(str(large_file), chunk_size=10000):
                assert len(chunk) <= 10000
                chunk_results.append(len(chunk))
            
            # Verify all data was processed
            total_processed = sum(chunk_results)
            assert total_processed == total_rows
            
            # Test memory-efficient schema inference
            schema = await file_processor.infer_schema_streaming(str(large_file))
            assert 'id' in schema
            assert 'value' in schema
            assert 'category' in schema
            assert 'timestamp' in schema
            
        finally:
            # Cleanup
            os.unlink(large_file)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_data_transformation_pipeline(self, file_processor, large_datasets):
        """Test data transformation pipeline"""
        data = large_datasets['large_csv']
        
        # Test data cleaning
        cleaned_data = await file_processor.clean_data(data)
        assert len(cleaned_data) <= len(data)  # May remove some rows
        
        # Test feature engineering
        engineered_data = await file_processor.engineer_features(cleaned_data)
        assert len(engineered_data.columns) >= len(cleaned_data.columns)
        
        # Test data normalization
        normalized_data = await file_processor.normalize_data(engineered_data)
        
        # Check that numeric columns are normalized
        numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'id':  # Skip ID column
                col_mean = normalized_data[col].mean()
                col_std = normalized_data[col].std()
                assert abs(col_mean) < 0.1  # Should be close to 0
                assert abs(col_std - 1.0) < 0.1  # Should be close to 1
        
        # Test data splitting
        train_data, test_data = await file_processor.split_data(normalized_data, test_size=0.2)
        assert len(train_data) + len(test_data) == len(normalized_data)
        assert len(test_data) / len(normalized_data) == pytest.approx(0.2, rel=0.05)
    
    @pytest.mark.asyncio
    async def test_multi_format_integration_pipeline(self, file_processor, large_datasets, complex_json_data):
        """Test integration pipeline with multiple file formats"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create files in different formats
        csv_file = temp_dir / "data.csv"
        excel_file = temp_dir / "data.xlsx"
        json_file = temp_dir / "data.json"
        
        large_datasets['large_csv'].to_csv(csv_file, index=False)
        large_datasets['large_csv'].to_excel(excel_file, index=False)
        
        with open(json_file, 'w') as f:
            json.dump(complex_json_data, f, default=str)
        
        try:
            # Process all files
            files = [csv_file, excel_file, json_file]
            processed_data = {}
            
            for file_path in files:
                file_info = await file_processor.detect_file_type(str(file_path))
                
                if file_info['type'] in ['csv', 'excel']:
                    data = await file_processor.load_data(str(file_path))
                    processed_data[file_path.suffix] = data
                elif file_info['type'] == 'json':
                    # Convert JSON to DataFrames
                    users_df = await file_processor.json_to_dataframe(complex_json_data['users'])
                    processed_data['.json'] = users_df
            
            # Verify all formats were processed
            assert '.csv' in processed_data
            assert '.xlsx' in processed_data
            assert '.json' in processed_data
            
            # Verify data consistency between CSV and Excel
            csv_data = processed_data['.csv']
            excel_data = processed_data['.xlsx']
            
            assert len(csv_data) == len(excel_data)
            assert list(csv_data.columns) == list(excel_data.columns)
            
            # Test data merging from different sources
            merged_data = await file_processor.merge_datasets([
                csv_data,
                processed_data['.json']
            ])
            
            assert len(merged_data.columns) > len(csv_data.columns)
            
        finally:
            # Cleanup
            for file_path in files:
                if file_path.exists():
                    os.unlink(file_path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipelines(self, file_processor):
        """Test error handling in data pipelines"""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test with corrupted CSV
            corrupted_csv = temp_dir / "corrupted.csv"
            with open(corrupted_csv, 'w') as f:
                f.write("id,name,value\n")
                f.write("1,John,100\n")
                f.write("2,Jane,invalid_number\n")  # Invalid data
                f.write("3,Bob\n")  # Missing column
            
            # Should handle gracefully
            try:
                data = await file_processor.load_data(str(corrupted_csv))
                quality_report = await file_processor.validate_data_quality(data)
                assert 'errors' in quality_report
                assert quality_report['errors'] > 0
            except Exception as e:
                # Should provide meaningful error message
                assert "data quality" in str(e).lower() or "parsing" in str(e).lower()
            
            # Test with empty file
            empty_file = temp_dir / "empty.csv"
            with open(empty_file, 'w') as f:
                f.write("")
            
            try:
                data = await file_processor.load_data(str(empty_file))
                assert len(data) == 0
            except Exception as e:
                assert "empty" in str(e).lower() or "no data" in str(e).lower()
            
            # Test with invalid JSON
            invalid_json = temp_dir / "invalid.json"
            with open(invalid_json, 'w') as f:
                f.write('{"invalid": json syntax}')
            
            try:
                await file_processor.analyze_json_structure(str(invalid_json))
            except Exception as e:
                assert "json" in str(e).lower() or "parsing" in str(e).lower()
            
        finally:
            # Cleanup
            for file_path in temp_dir.glob("*"):
                os.unlink(file_path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_performance_with_various_sizes(self, file_processor):
        """Test pipeline performance with various file sizes"""
        import time
        
        sizes = [100, 1000, 10000]  # Different dataset sizes
        performance_results = {}
        
        for size in sizes:
            # Create dataset of specific size
            data = pd.DataFrame({
                'id': range(size),
                'value': np.random.randn(size),
                'category': np.random.choice(['A', 'B', 'C'], size)
            })
            
            temp_dir = Path(tempfile.mkdtemp())
            csv_file = temp_dir / f"data_{size}.csv"
            data.to_csv(csv_file, index=False)
            
            try:
                # Measure processing time
                start_time = time.time()
                
                # Full pipeline
                file_info = await file_processor.detect_file_type(str(csv_file))
                loaded_data = await file_processor.load_data(str(csv_file))
                schema = await file_processor.infer_schema(loaded_data)
                quality_report = await file_processor.validate_data_quality(loaded_data)
                preview = await file_processor.generate_preview(loaded_data)
                
                processing_time = time.time() - start_time
                performance_results[size] = processing_time
                
                # Verify processing was successful
                assert len(loaded_data) == size
                assert len(schema) == 3
                assert 'missing_values' in quality_report
                assert 'head' in preview
                
            finally:
                # Cleanup
                os.unlink(csv_file)
                os.rmdir(temp_dir)
        
        # Verify performance scales reasonably
        assert performance_results[100] < performance_results[1000]
        assert performance_results[1000] < performance_results[10000]
        
        # Performance should be sub-linear (not 100x slower for 100x data)
        ratio_10x = performance_results[1000] / performance_results[100]
        ratio_100x = performance_results[10000] / performance_results[100]
        
        assert ratio_10x < 20  # Should be less than 20x slower
        assert ratio_100x < 200  # Should be less than 200x slower
        
        print(f"Performance Results:")
        for size, time_taken in performance_results.items():
            print(f"  {size} rows: {time_taken:.3f}s")
            print(f"  Rate: {size / time_taken:.0f} rows/second")