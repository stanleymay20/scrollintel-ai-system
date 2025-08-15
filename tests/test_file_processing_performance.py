"""
Integration tests for file processing performance optimizations.
Tests database connection pooling, file processing with progress tracking,
and background job processing.
"""

import pytest
import asyncio
import tempfile
import os
import pandas as pd
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi import UploadFile
from io import BytesIO, StringIO

from scrollintel.engines.file_processor import FileProcessorEngine
from scrollintel.core.database_pool import OptimizedDatabasePool, get_optimized_db_pool
from scrollintel.core.database_optimizer import DatabaseOptimizer, get_database_optimizer
from scrollintel.core.background_jobs import BackgroundJobProcessor, get_job_processor, JobPriority
from scrollintel.core.config import get_settings
from scrollintel.models.database import FileUpload, Dataset


class TestDatabasePerformanceOptimizations:
    """Test database performance optimizations."""
    
    @pytest.fixture
    async def db_pool(self):
        """Create optimized database pool for testing."""
        settings = get_settings()
        pool = OptimizedDatabasePool(database_url=settings.database_url)
        await pool.initialize()
        yield pool
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, db_pool):
        """Test database connection pool initialization."""
        
        # Test pool status
        pool_status = db_pool.get_pool_status()
        
        assert pool_status['pool_size'] > 0
        assert pool_status['checked_in'] >= 0
        assert pool_status['checked_out'] >= 0
        assert 'pool_class' in pool_status
        
        # Test performance stats
        performance_stats = db_pool.get_performance_stats()
        
        assert 'pool_status' in performance_stats
        assert 'performance_report' in performance_stats
        assert 'timestamp' in performance_stats
    
    @pytest.mark.asyncio
    async def test_connection_pool_sessions(self, db_pool):
        """Test database session management."""
        
        # Test sync session
        with db_pool.get_session() as session:
            result = session.execute("SELECT 1 as test_value")
            row = result.fetchone()
            assert row.test_value == 1
        
        # Test async session
        async with db_pool.get_async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1 as test_value"))
            row = result.fetchone()
            assert row.test_value == 1
    
    @pytest.mark.asyncio
    async def test_query_caching(self, db_pool):
        """Test query caching functionality."""
        
        if not db_pool.query_cache:
            pytest.skip("Query caching not enabled")
        
        from sqlalchemy import text
        
        # Execute cached query
        query = text("SELECT COUNT(*) as count FROM information_schema.tables")
        
        # First execution - cache miss
        start_time = time.time()
        result1 = await db_pool.execute_cached_query(query, cache_key="test_query_1")
        first_duration = time.time() - start_time
        
        # Second execution - cache hit
        start_time = time.time()
        result2 = await db_pool.execute_cached_query(query, cache_key="test_query_1")
        second_duration = time.time() - start_time
        
        # Results should be the same
        assert result1 == result2
        
        # Second query should be faster (cached)
        assert second_duration < first_duration
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, db_pool):
        """Test performance monitoring functionality."""
        
        # Execute some queries to generate metrics
        async with db_pool.get_async_session() as session:
            from sqlalchemy import text
            
            for i in range(5):
                await session.execute(text("SELECT pg_sleep(0.1)"))  # Slow query
                await session.execute(text("SELECT 1"))  # Fast query
        
        # Check performance metrics
        performance_stats = db_pool.get_performance_stats()
        
        assert performance_stats['performance_report']['query_stats']['total_queries'] >= 10
        
        # Check for slow queries
        slow_queries = db_pool.monitor.get_slow_queries(limit=5)
        assert len(slow_queries) > 0
        
        # Check for frequent queries
        frequent_queries = db_pool.monitor.get_frequent_queries(limit=5)
        assert len(frequent_queries) > 0


class TestDatabaseOptimizer:
    """Test database optimizer functionality."""
    
    @pytest.fixture
    async def optimizer(self):
        """Create database optimizer for testing."""
        optimizer = DatabaseOptimizer()
        await optimizer.initialize()
        return optimizer
    
    @pytest.mark.asyncio
    async def test_table_statistics_analysis(self, optimizer):
        """Test table statistics analysis."""
        
        table_stats = await optimizer._analyze_table_statistics()
        
        # Should have some tables
        assert len(table_stats) > 0
        
        # Check table stats structure
        for stats in table_stats:
            assert hasattr(stats, 'table_name')
            assert hasattr(stats, 'row_count')
            assert hasattr(stats, 'table_size_bytes')
            assert hasattr(stats, 'vacuum_needed')
            assert hasattr(stats, 'analyze_needed')
    
    @pytest.mark.asyncio
    async def test_index_recommendations(self, optimizer):
        """Test index recommendation generation."""
        
        # Get table statistics first
        table_stats = await optimizer._analyze_table_statistics()
        
        # Generate index recommendations
        recommendations = await optimizer._generate_index_recommendations(table_stats)
        
        # Should be a list (may be empty)
        assert isinstance(recommendations, list)
        
        # If there are recommendations, check structure
        for rec in recommendations:
            assert hasattr(rec, 'table_name')
            assert hasattr(rec, 'column_names')
            assert hasattr(rec, 'index_type')
            assert hasattr(rec, 'create_statement')
            assert hasattr(rec, 'estimated_benefit')
    
    @pytest.mark.asyncio
    async def test_database_health_report(self, optimizer):
        """Test database health report generation."""
        
        health_report = await optimizer.get_database_health_report()
        
        assert 'timestamp' in health_report
        assert 'overall_health' in health_report
        assert 'health_score' in health_report
        assert 'metrics' in health_report
        
        # Health score should be between 0 and 100
        if 'health_score' in health_report:
            assert 0 <= health_report['health_score'] <= 100
    
    @pytest.mark.asyncio
    async def test_comprehensive_optimization(self, optimizer):
        """Test comprehensive database optimization."""
        
        results = await optimizer.run_comprehensive_optimization()
        
        assert 'started_at' in results
        assert 'status' in results
        assert 'execution_time' in results
        
        if results['status'] == 'completed':
            assert 'table_stats' in results
            assert 'index_recommendations' in results
            assert 'maintenance' in results
            assert 'statistics_update' in results


class TestFileProcessingPerformance:
    """Test file processing performance optimizations."""
    
    @pytest.fixture
    def file_processor(self):
        """Create file processor engine for testing."""
        return FileProcessorEngine()
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        
        # Create sample data
        data = {
            'id': range(1, 1001),
            'name': [f'User {i}' for i in range(1, 1001)],
            'email': [f'user{i}@example.com' for i in range(1, 1001)],
            'age': [20 + (i % 50) for i in range(1, 1001)],
            'created_at': ['2024-01-01'] * 1000
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def large_csv_file(self):
        """Create a large CSV file for testing background processing."""
        
        # Create larger sample data (10k rows)
        data = {
            'id': range(1, 10001),
            'name': [f'User {i}' for i in range(1, 10001)],
            'email': [f'user{i}@example.com' for i in range(1, 10001)],
            'age': [20 + (i % 50) for i in range(1, 10001)],
            'score': [i * 0.1 for i in range(1, 10001)],
            'category': [f'Category {i % 10}' for i in range(1, 10001)],
            'created_at': ['2024-01-01'] * 10000
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def create_upload_file(self, file_path: str, filename: str = None) -> UploadFile:
        """Create UploadFile object from file path."""
        
        if not filename:
            filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        file_obj = BytesIO(content)
        file_obj.seek(0)
        
        return UploadFile(
            filename=filename,
            file=file_obj,
            size=len(content),
            headers={'content-type': 'text/csv'}
        )
    
    @pytest.mark.asyncio
    async def test_file_validation_optimized(self, file_processor, sample_csv_file):
        """Test optimized file validation."""
        
        upload_file = self.create_upload_file(sample_csv_file, "test.csv")
        
        # Should not raise exception for valid file
        await file_processor._validate_file_optimized(upload_file)
        
        # Test invalid file size
        upload_file.size = file_processor.max_file_size + 1
        
        with pytest.raises(Exception) as exc_info:
            await file_processor._validate_file_optimized(upload_file)
        
        assert "exceeds maximum allowed size" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_file_type_detection_optimized(self, file_processor, sample_csv_file):
        """Test optimized file type detection."""
        
        upload_file = self.create_upload_file(sample_csv_file, "test.csv")
        
        detected_type = await file_processor._detect_file_type_optimized(
            "text/csv", "test.csv", upload_file
        )
        
        assert detected_type == "csv"
    
    @pytest.mark.asyncio
    async def test_file_saving_optimized(self, file_processor, sample_csv_file):
        """Test optimized file saving with streaming."""
        
        upload_file = self.create_upload_file(sample_csv_file, "test.csv")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path, file_hash = await file_processor._save_file_optimized(
                upload_file, "test-upload-id", temp_dir
            )
            
            # Check file was saved
            assert os.path.exists(file_path)
            
            # Check file hash was generated
            assert file_hash is not None
            assert len(file_hash) == 64  # SHA256 hash length
            
            # Check file size matches
            saved_size = os.path.getsize(file_path)
            assert saved_size == upload_file.size
    
    @pytest.mark.asyncio
    async def test_chunked_csv_loading(self, file_processor, large_csv_file):
        """Test chunked loading of large CSV files."""
        
        # Mock progress tracker
        progress_tracker = Mock()
        progress_tracker.update = AsyncMock()
        
        # Load data with chunked processing
        df = await file_processor._load_csv_chunked(large_csv_file, progress_tracker)
        
        # Check data was loaded correctly
        assert len(df) == 10000
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'email' in df.columns
        
        # Check progress tracker was called
        assert progress_tracker.update.call_count > 0
    
    @pytest.mark.asyncio
    async def test_schema_inference_async(self, file_processor, sample_csv_file):
        """Test async schema inference with progress tracking."""
        
        # Load data first
        df = await file_processor._load_data(sample_csv_file, "csv")
        
        # Mock progress tracker
        progress_tracker = Mock()
        progress_tracker.update = AsyncMock()
        
        # Infer schema
        schema_info = await file_processor._infer_schema_async(df, progress_tracker)
        
        # Check schema structure
        assert 'columns' in schema_info
        assert 'total_rows' in schema_info
        assert 'total_columns' in schema_info
        
        # Check columns were analyzed
        assert len(schema_info['columns']) == len(df.columns)
        
        # Check progress tracker was called
        assert progress_tracker.update.call_count > 0
    
    @pytest.mark.asyncio
    async def test_quality_report_async(self, file_processor, sample_csv_file):
        """Test async quality report generation with progress tracking."""
        
        # Load data first
        df = await file_processor._load_data(sample_csv_file, "csv")
        
        # Mock progress tracker
        progress_tracker = Mock()
        progress_tracker.update = AsyncMock()
        
        # Generate quality report
        quality_report = await file_processor._generate_quality_report_async(df, progress_tracker)
        
        # Check quality report structure
        assert 'total_rows' in quality_report
        assert 'total_columns' in quality_report
        assert 'missing_values' in quality_report
        assert 'quality_score' in quality_report
        assert 'recommendations' in quality_report
        
        # Check progress tracker was called
        assert progress_tracker.update.call_count > 0
    
    @pytest.mark.asyncio
    async def test_processing_status_tracking(self, file_processor):
        """Test file processing status tracking."""
        
        upload_id = "test-upload-123"
        
        # Mock database session
        with patch('scrollintel.engines.file_processor.get_optimized_db_pool') as mock_pool:
            mock_session = AsyncMock()
            mock_pool.return_value.get_async_session.return_value.__aenter__.return_value = mock_session
            
            # Mock file upload record
            mock_file_upload = Mock()
            mock_file_upload.processing_status = "processing"
            mock_file_upload.processing_progress = 0.5
            mock_file_upload.processing_message = "Processing in progress..."
            mock_file_upload.original_filename = "test.csv"
            mock_file_upload.file_size = 1024
            mock_file_upload.created_at = None
            
            mock_session.get.return_value = mock_file_upload
            
            # Get processing status
            status = await file_processor.get_processing_status(upload_id)
            
            # Check status structure
            assert status['upload_id'] == upload_id
            assert status['status'] == "processing"
            assert status['progress'] == 0.5
            assert status['message'] == "Processing in progress..."
    
    @pytest.mark.asyncio
    async def test_processing_cancellation(self, file_processor):
        """Test file processing cancellation."""
        
        upload_id = "test-upload-123"
        user_id = "test-user-456"
        
        # Mock database session
        with patch('scrollintel.engines.file_processor.get_optimized_db_pool') as mock_pool:
            mock_session = AsyncMock()
            mock_pool.return_value.get_async_session.return_value.__aenter__.return_value = mock_session
            
            # Mock file upload record
            mock_file_upload = Mock()
            mock_file_upload.user_id = user_id
            mock_file_upload.processing_status = "processing"
            
            mock_session.get.return_value = mock_file_upload
            
            # Cancel processing
            result = await file_processor.cancel_processing(upload_id, user_id)
            
            # Check result
            assert result['upload_id'] == upload_id
            assert result['status'] == "cancelled"
            
            # Check file upload was updated
            assert mock_file_upload.processing_status == "cancelled"


class TestBackgroundJobProcessing:
    """Test background job processing for file operations."""
    
    @pytest.fixture
    async def job_processor(self):
        """Create background job processor for testing."""
        processor = BackgroundJobProcessor(max_workers=2)
        await processor.initialize()
        
        # Register test job function
        async def test_job_function(test_param: str, progress_tracker=None):
            if progress_tracker:
                await progress_tracker.update(50.0, "Processing...")
                await asyncio.sleep(0.1)  # Simulate work
                await progress_tracker.update(100.0, "Completed")
            return {"result": f"processed_{test_param}"}
        
        processor.register_job_function("test_job_function", test_job_function)
        
        yield processor
        
        await processor.cleanup()
    
    @pytest.mark.asyncio
    async def test_job_submission_and_execution(self, job_processor):
        """Test job submission and execution."""
        
        # Submit a job
        job_id = await job_processor.submit_job(
            job_type="test",
            function_name="test_job_function",
            args=["test_value"],
            priority=JobPriority.HIGH,
            user_id="test-user"
        )
        
        assert job_id is not None
        
        # Start workers to process the job
        await job_processor.start_workers()
        
        # Wait for job completion
        max_wait = 10  # seconds
        wait_time = 0
        result = None
        
        while wait_time < max_wait:
            result = await job_processor.get_job_result(job_id)
            if result and result.status.value in ['completed', 'failed']:
                break
            
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        # Check job completed successfully
        assert result is not None
        assert result.status.value == 'completed'
        assert result.result == {"result": "processed_test_value"}
        
        await job_processor.stop_workers()
    
    @pytest.mark.asyncio
    async def test_job_progress_tracking(self, job_processor):
        """Test job progress tracking."""
        
        # Submit a job
        job_id = await job_processor.submit_job(
            job_type="test",
            function_name="test_job_function",
            args=["progress_test"],
            priority=JobPriority.NORMAL
        )
        
        # Start workers
        await job_processor.start_workers()
        
        # Check progress updates
        max_wait = 10
        wait_time = 0
        progress_found = False
        
        while wait_time < max_wait:
            progress = await job_processor.get_job_progress(job_id)
            
            if progress.get('progress', 0) > 0:
                progress_found = True
                break
            
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        assert progress_found, "Job progress should be tracked"
        
        await job_processor.stop_workers()
    
    @pytest.mark.asyncio
    async def test_job_processor_stats(self, job_processor):
        """Test job processor statistics."""
        
        stats = await job_processor.get_stats()
        
        assert 'running' in stats
        assert 'workers' in stats
        assert 'max_workers' in stats
        assert 'registered_functions' in stats
        assert 'queue_stats' in stats
        
        # Check registered functions
        assert 'test_job_function' in stats['registered_functions']


class TestIntegrationWorkflows:
    """Test complete file processing workflows."""
    
    @pytest.fixture
    def file_processor(self):
        return FileProcessorEngine()
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file with various data types."""
        
        data = {
            'id': range(1, 101),
            'name': [f'User {i}' for i in range(1, 101)],
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'age': [20 + (i % 50) for i in range(1, 101)],
            'salary': [30000 + (i * 1000) for i in range(1, 101)],
            'is_active': [i % 2 == 0 for i in range(1, 101)],
            'created_at': ['2024-01-01'] * 100,
            'notes': [f'Notes for user {i}' if i % 5 == 0 else None for i in range(1, 101)]
        }
        
        df = pd.DataFrame(data)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_complete_file_processing_workflow(self, file_processor, sample_data_file):
        """Test complete file processing workflow with all optimizations."""
        
        # Create upload file
        with open(sample_data_file, 'rb') as f:
            content = f.read()
        
        file_obj = BytesIO(content)
        upload_file = UploadFile(
            filename="test_data.csv",
            file=file_obj,
            size=len(content),
            headers={'content-type': 'text/csv'}
        )
        
        # Mock database operations
        with patch('scrollintel.engines.file_processor.get_optimized_db_pool') as mock_pool, \
             patch('scrollintel.engines.file_processor.get_job_processor') as mock_job_processor:
            
            mock_session = AsyncMock()
            mock_pool.return_value.get_async_session.return_value.__aenter__.return_value = mock_session
            
            mock_job_processor.return_value.submit_job = AsyncMock(return_value="job-123")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process upload
                result = await file_processor.process_upload(
                    file=upload_file,
                    user_id="test-user",
                    storage_path=temp_dir,
                    db_session=mock_session,
                    auto_detect_schema=True,
                    generate_preview=True,
                    use_background_processing=False  # Process immediately for testing
                )
                
                # Check result structure
                assert result.upload_id is not None
                assert result.filename is not None
                assert result.detected_type == "csv"
                assert result.file_size == len(content)
                assert result.processing_time > 0
                
                # Check file was saved
                assert os.path.exists(result.file_path)
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, file_processor, sample_data_file):
        """Test file processing performance under concurrent load."""
        
        async def process_single_file():
            with open(sample_data_file, 'rb') as f:
                content = f.read()
            
            file_obj = BytesIO(content)
            upload_file = UploadFile(
                filename="load_test.csv",
                file=file_obj,
                size=len(content),
                headers={'content-type': 'text/csv'}
            )
            
            # Mock database operations
            with patch('scrollintel.engines.file_processor.get_optimized_db_pool') as mock_pool:
                mock_session = AsyncMock()
                mock_pool.return_value.get_async_session.return_value.__aenter__.return_value = mock_session
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    start_time = time.time()
                    
                    result = await file_processor.process_upload(
                        file=upload_file,
                        user_id="load-test-user",
                        storage_path=temp_dir,
                        db_session=mock_session,
                        use_background_processing=False
                    )
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'processing_time': processing_time,
                        'upload_id': result.upload_id
                    }
        
        # Process multiple files concurrently
        num_concurrent = 5
        tasks = [process_single_file() for _ in range(num_concurrent)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Check all tasks completed successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        assert len(successful_results) == num_concurrent
        
        # Check performance metrics
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        
        # Log performance metrics
        print(f"Concurrent processing: {num_concurrent} files")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average processing time: {avg_processing_time:.2f}s")
        print(f"Throughput: {num_concurrent / total_time:.2f} files/second")
        
        # Performance assertions
        assert total_time < 30  # Should complete within 30 seconds
        assert avg_processing_time < 10  # Each file should process within 10 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])