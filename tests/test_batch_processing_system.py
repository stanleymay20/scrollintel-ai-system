"""
Tests for the Batch Processing System

This module contains comprehensive tests for the batch processing system,
including efficiency, reliability, resource allocation, and error handling tests.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import uuid

from scrollintel.engines.visual_generation.batch_processor import (
    BatchProcessor, BatchJob, BatchJobRequest, BatchJobStatus, BatchJobPriority,
    ResourceAllocator, BatchJobScheduler, ProgressTracker, ResultAggregator,
    BatchProcessingMetrics
)
from scrollintel.engines.visual_generation.base import (
    GenerationRequest, GenerationResult, GenerationStatus, QualityMetrics
)
from scrollintel.engines.visual_generation.exceptions import BatchProcessingError


class TestResourceAllocator:
    """Test resource allocation functionality."""
    
    @pytest.fixture
    def allocator(self):
        return ResourceAllocator(max_concurrent_jobs=2, max_concurrent_requests=5)
    
    @pytest.fixture
    def sample_job(self):
        requests = [
            BatchJobRequest(id=f"req_{i}", request=Mock(spec=GenerationRequest))
            for i in range(3)
        ]
        return BatchJob(
            id="test_job",
            name="Test Job",
            user_id="user123",
            requests=requests
        )
    
    @pytest.mark.asyncio
    async def test_can_start_job_with_available_resources(self, allocator, sample_job):
        """Test that jobs can start when resources are available."""
        can_start = await allocator.can_start_job(sample_job)
        assert can_start is True
    
    @pytest.mark.asyncio
    async def test_cannot_start_job_when_max_jobs_reached(self, allocator, sample_job):
        """Test that jobs cannot start when max concurrent jobs reached."""
        # Fill up job slots
        for i in range(2):
            job = BatchJob(id=f"job_{i}", name=f"Job {i}", user_id="user", requests=[])
            await allocator.allocate_job(job)
        
        can_start = await allocator.can_start_job(sample_job)
        assert can_start is False
    
    @pytest.mark.asyncio
    async def test_cannot_start_job_when_max_requests_reached(self, allocator):
        """Test that jobs cannot start when max concurrent requests reached."""
        # Create job with many requests
        requests = [
            BatchJobRequest(id=f"req_{i}", request=Mock(spec=GenerationRequest))
            for i in range(6)  # More than max_concurrent_requests
        ]
        large_job = BatchJob(
            id="large_job",
            name="Large Job",
            user_id="user123",
            requests=requests
        )
        
        can_start = await allocator.can_start_job(large_job)
        assert can_start is False
    
    @pytest.mark.asyncio
    async def test_allocate_and_deallocate_job(self, allocator, sample_job):
        """Test job allocation and deallocation."""
        # Allocate job
        allocated = await allocator.allocate_job(sample_job)
        assert allocated is True
        assert sample_job.id in allocator.active_jobs
        
        # Deallocate job
        await allocator.deallocate_job(sample_job.id)
        assert sample_job.id not in allocator.active_jobs
    
    @pytest.mark.asyncio
    async def test_allocate_and_deallocate_request(self, allocator):
        """Test request allocation and deallocation."""
        request = BatchJobRequest(id="test_req", request=Mock(spec=GenerationRequest))
        
        # Allocate request
        allocated = await allocator.allocate_request(request)
        assert allocated is True
        assert request.id in allocator.active_requests
        
        # Deallocate request
        await allocator.deallocate_request(request.id)
        assert request.id not in allocator.active_requests


class TestBatchJobScheduler:
    """Test job scheduling functionality."""
    
    @pytest.fixture
    def scheduler(self):
        return BatchJobScheduler()
    
    @pytest.fixture
    def sample_jobs(self):
        jobs = []
        priorities = [BatchJobPriority.LOW, BatchJobPriority.NORMAL, BatchJobPriority.HIGH, BatchJobPriority.URGENT]
        for i, priority in enumerate(priorities):
            job = BatchJob(
                id=f"job_{i}",
                name=f"Job {i}",
                user_id="user123",
                requests=[],
                priority=priority
            )
            jobs.append(job)
        return jobs
    
    def test_add_job_to_queue(self, scheduler, sample_jobs):
        """Test adding jobs to the scheduling queue."""
        for job in sample_jobs:
            scheduler.add_job(job)
        
        assert len(scheduler.job_queue) == len(sample_jobs)
    
    def test_get_next_job_by_priority(self, scheduler, sample_jobs):
        """Test that jobs are returned in priority order."""
        # Add jobs in random order
        for job in reversed(sample_jobs):
            scheduler.add_job(job)
        
        # Should get URGENT priority job first
        next_job = scheduler.get_next_job()
        assert next_job.priority == BatchJobPriority.URGENT
        
        # Then HIGH priority
        next_job = scheduler.get_next_job()
        assert next_job.priority == BatchJobPriority.HIGH
    
    def test_get_next_job_by_creation_time(self, scheduler):
        """Test that jobs with same priority are ordered by creation time."""
        # Create jobs with same priority but different creation times
        job1 = BatchJob(id="job1", name="Job 1", user_id="user", requests=[], priority=BatchJobPriority.NORMAL)
        job2 = BatchJob(id="job2", name="Job 2", user_id="user", requests=[], priority=BatchJobPriority.NORMAL)
        job2.created_at = job1.created_at + timedelta(seconds=1)
        
        scheduler.add_job(job2)
        scheduler.add_job(job1)
        
        # Should get job1 first (earlier creation time)
        next_job = scheduler.get_next_job()
        assert next_job.id == "job1"
    
    def test_get_queue_status(self, scheduler, sample_jobs):
        """Test getting queue status by priority."""
        for job in sample_jobs:
            scheduler.add_job(job)
        
        status = scheduler.get_queue_status()
        assert status["LOW"] == 1
        assert status["NORMAL"] == 1
        assert status["HIGH"] == 1
        assert status["URGENT"] == 1


class TestProgressTracker:
    """Test progress tracking functionality."""
    
    @pytest.fixture
    def tracker(self):
        return ProgressTracker()
    
    @pytest.fixture
    def sample_job(self):
        requests = [
            BatchJobRequest(id=f"req_{i}", request=Mock(spec=GenerationRequest))
            for i in range(4)
        ]
        return BatchJob(
            id="test_job",
            name="Test Job",
            user_id="user123",
            requests=requests
        )
    
    def test_update_request_progress(self, tracker):
        """Test updating progress for individual requests."""
        tracker.update_request_progress("req_1", 0.5)
        assert tracker.request_progress["req_1"] == 0.5
        
        # Test bounds checking
        tracker.update_request_progress("req_1", 1.5)
        assert tracker.request_progress["req_1"] == 1.0
        
        tracker.update_request_progress("req_1", -0.5)
        assert tracker.request_progress["req_1"] == 0.0
    
    def test_update_job_progress(self, tracker, sample_job):
        """Test updating overall job progress."""
        # Initially all pending
        tracker.update_job_progress(sample_job)
        assert sample_job.progress == 0.0
        assert sample_job.completed_requests == 0
        
        # Complete one request
        sample_job.requests[0].status = GenerationStatus.COMPLETED
        tracker.update_job_progress(sample_job)
        assert sample_job.progress == 0.25
        assert sample_job.completed_requests == 1
        
        # One in progress
        sample_job.requests[1].status = GenerationStatus.IN_PROGRESS
        tracker.update_request_progress(sample_job.requests[1].id, 0.5)
        tracker.update_job_progress(sample_job)
        assert sample_job.progress == 0.375  # (1.0 + 0.5) / 4
        
        # One failed
        sample_job.requests[2].status = GenerationStatus.FAILED
        tracker.update_job_progress(sample_job)
        assert sample_job.failed_requests == 1
    
    def test_progress_callbacks(self, tracker, sample_job):
        """Test progress callback functionality."""
        callback_called = False
        callback_job = None
        
        def progress_callback(job):
            nonlocal callback_called, callback_job
            callback_called = True
            callback_job = job
        
        tracker.add_progress_callback(sample_job.id, progress_callback)
        tracker.update_job_progress(sample_job)
        
        assert callback_called is True
        assert callback_job == sample_job
        
        # Test callback removal
        tracker.remove_progress_callbacks(sample_job.id)
        assert sample_job.id not in tracker.progress_callbacks


class TestResultAggregator:
    """Test result aggregation functionality."""
    
    @pytest.fixture
    def aggregator(self):
        return ResultAggregator()
    
    @pytest.fixture
    def completed_job(self):
        requests = []
        for i in range(3):
            request = BatchJobRequest(id=f"req_{i}", request=Mock(spec=GenerationRequest))
            if i < 2:  # First two succeed
                request.status = GenerationStatus.COMPLETED
                request.result = GenerationResult(
                    id=f"result_{i}",
                    status=GenerationStatus.COMPLETED,
                    content_urls=[f"url_{i}"],
                    metadata={},
                    quality_metrics=QualityMetrics(overall_score=0.8 + i * 0.1),
                    generation_time=5.0 + i,
                    cost=1.0,
                    model_used="test_model",
                    created_at=datetime.utcnow()
                )
            else:  # Last one fails
                request.status = GenerationStatus.FAILED
                request.error = "Test error"
                request.retry_count = 3
            requests.append(request)
        
        job = BatchJob(
            id="test_job",
            name="Test Job",
            user_id="user123",
            requests=requests,
            started_at=datetime.utcnow() - timedelta(seconds=30),
            completed_at=datetime.utcnow()
        )
        return job
    
    def test_aggregate_job_results(self, aggregator, completed_job):
        """Test aggregating results for a completed job."""
        results = aggregator.aggregate_job_results(completed_job)
        
        assert results["job_id"] == "test_job"
        assert results["total_requests"] == 3
        assert results["successful_requests"] == 2
        assert results["failed_requests"] == 1
        assert results["success_rate"] == 2/3
        assert results["total_duration"] == 30.0
        assert results["average_generation_time"] == 5.5  # (5.0 + 6.0) / 2
        assert results["average_quality_score"] == 0.85  # (0.8 + 0.9) / 2
        assert len(results["successful_results"]) == 2
        assert len(results["failed_results"]) == 1
    
    def test_get_job_results(self, aggregator, completed_job):
        """Test retrieving aggregated results."""
        # Aggregate results first
        aggregator.aggregate_job_results(completed_job)
        
        # Retrieve results
        results = aggregator.get_job_results("test_job")
        assert results is not None
        assert results["job_id"] == "test_job"
        
        # Non-existent job
        results = aggregator.get_job_results("non_existent")
        assert results is None
    
    def test_export_results(self, aggregator, completed_job):
        """Test exporting results in different formats."""
        # Aggregate results first
        aggregator.aggregate_job_results(completed_job)
        
        # Export as JSON
        json_export = aggregator.export_results("test_job", "json")
        assert isinstance(json_export, str)
        assert "test_job" in json_export
        
        # Test unsupported format
        with pytest.raises(ValueError):
            aggregator.export_results("test_job", "xml")
        
        # Test non-existent job
        with pytest.raises(ValueError):
            aggregator.export_results("non_existent", "json")


class TestBatchProcessor:
    """Test the main batch processor functionality."""
    
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = AsyncMock()
        pipeline.generate.return_value = GenerationResult(
            id="test_result",
            status=GenerationStatus.COMPLETED,
            content_urls=["test_url"],
            metadata={},
            quality_metrics=QualityMetrics(overall_score=0.8),
            generation_time=5.0,
            cost=1.0,
            model_used="test_model",
            created_at=datetime.utcnow()
        )
        return pipeline
    
    @pytest.fixture
    async def processor(self, mock_pipeline):
        processor = BatchProcessor(generation_pipeline=mock_pipeline)
        await processor.start()
        yield processor
        await processor.stop()
    
    @pytest.fixture
    def sample_requests(self):
        return [
            Mock(spec=GenerationRequest, prompt=f"Test prompt {i}")
            for i in range(3)
        ]
    
    @pytest.mark.asyncio
    async def test_submit_batch_job(self, processor, sample_requests):
        """Test submitting a batch job."""
        job_id = await processor.submit_batch_job(
            name="Test Batch",
            user_id="user123",
            requests=sample_requests,
            priority=BatchJobPriority.HIGH
        )
        
        assert job_id is not None
        assert job_id in processor.jobs
        
        job = processor.jobs[job_id]
        assert job.name == "Test Batch"
        assert job.user_id == "user123"
        assert len(job.requests) == 3
        assert job.priority == BatchJobPriority.HIGH
        assert job.status == BatchJobStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_job_status(self, processor, sample_requests):
        """Test getting job status."""
        job_id = await processor.submit_batch_job(
            name="Test Batch",
            user_id="user123",
            requests=sample_requests
        )
        
        job = await processor.get_job_status(job_id)
        assert job is not None
        assert job.id == job_id
        
        # Non-existent job
        job = await processor.get_job_status("non_existent")
        assert job is None
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, processor, sample_requests):
        """Test cancelling a batch job."""
        job_id = await processor.submit_batch_job(
            name="Test Batch",
            user_id="user123",
            requests=sample_requests
        )
        
        # Cancel job
        cancelled = await processor.cancel_job(job_id)
        assert cancelled is True
        
        job = processor.jobs[job_id]
        assert job.status == BatchJobStatus.CANCELLED
        
        # Try to cancel already cancelled job
        cancelled = await processor.cancel_job(job_id)
        assert cancelled is False
    
    @pytest.mark.asyncio
    async def test_pause_and_resume_job(self, processor, sample_requests):
        """Test pausing and resuming a batch job."""
        job_id = await processor.submit_batch_job(
            name="Test Batch",
            user_id="user123",
            requests=sample_requests
        )
        
        # Set job to running state
        job = processor.jobs[job_id]
        job.status = BatchJobStatus.RUNNING
        
        # Pause job
        paused = await processor.pause_job(job_id)
        assert paused is True
        assert job.status == BatchJobStatus.PAUSED
        
        # Resume job
        resumed = await processor.resume_job(job_id)
        assert resumed is True
        assert job.status == BatchJobStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, processor, sample_requests):
        """Test getting batch processing metrics."""
        # Submit some jobs
        for i in range(3):
            await processor.submit_batch_job(
                name=f"Test Batch {i}",
                user_id="user123",
                requests=sample_requests
            )
        
        metrics = await processor.get_metrics()
        assert isinstance(metrics, BatchProcessingMetrics)
        assert metrics.total_jobs == 3
        assert metrics.queue_length == 3
    
    @pytest.mark.asyncio
    async def test_job_processing_flow(self, processor, sample_requests):
        """Test the complete job processing flow."""
        job_id = await processor.submit_batch_job(
            name="Test Batch",
            user_id="user123",
            requests=sample_requests
        )
        
        # Wait for job to be processed
        max_wait = 30  # seconds
        wait_time = 0
        while wait_time < max_wait:
            job = processor.jobs[job_id]
            if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
                break
            await asyncio.sleep(1)
            wait_time += 1
        
        job = processor.jobs[job_id]
        assert job.status == BatchJobStatus.COMPLETED
        assert job.progress == 1.0
        assert job.completed_requests == len(sample_requests)
        
        # Check that results were aggregated
        results = processor.result_aggregator.get_job_results(job_id)
        assert results is not None
        assert results["successful_requests"] == len(sample_requests)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, processor):
        """Test error handling during job processing."""
        # Create a mock pipeline that raises an error
        error_pipeline = AsyncMock()
        error_pipeline.generate.side_effect = Exception("Test error")
        processor.generation_pipeline = error_pipeline
        
        requests = [Mock(spec=GenerationRequest, prompt="Test prompt")]
        job_id = await processor.submit_batch_job(
            name="Error Test",
            user_id="user123",
            requests=requests
        )
        
        # Wait for job to be processed
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            job = processor.jobs[job_id]
            if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
                break
            await asyncio.sleep(1)
            wait_time += 1
        
        job = processor.jobs[job_id]
        # Job should complete even with failed requests
        assert job.status == BatchJobStatus.COMPLETED
        assert job.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_retry_logic(self, processor):
        """Test retry logic for failed requests."""
        # Create a pipeline that fails first few times then succeeds
        call_count = 0
        
        async def failing_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise Exception("Temporary error")
            return GenerationResult(
                id="test_result",
                status=GenerationStatus.COMPLETED,
                content_urls=["test_url"],
                metadata={},
                quality_metrics=QualityMetrics(overall_score=0.8),
                generation_time=5.0,
                cost=1.0,
                model_used="test_model",
                created_at=datetime.utcnow()
            )
        
        processor.generation_pipeline.generate = failing_generate
        
        requests = [Mock(spec=GenerationRequest, prompt="Test prompt")]
        job_id = await processor.submit_batch_job(
            name="Retry Test",
            user_id="user123",
            requests=requests
        )
        
        # Wait for job to be processed
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            job = processor.jobs[job_id]
            if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
                break
            await asyncio.sleep(1)
            wait_time += 1
        
        job = processor.jobs[job_id]
        assert job.status == BatchJobStatus.COMPLETED
        assert job.completed_requests == 1
        assert call_count == 3  # Should have retried twice


class TestBatchProcessingIntegration:
    """Integration tests for the complete batch processing system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self):
        """Test processing multiple jobs concurrently."""
        mock_pipeline = AsyncMock()
        mock_pipeline.generate.return_value = GenerationResult(
            id="test_result",
            status=GenerationStatus.COMPLETED,
            content_urls=["test_url"],
            metadata={},
            quality_metrics=QualityMetrics(overall_score=0.8),
            generation_time=1.0,
            cost=1.0,
            model_used="test_model",
            created_at=datetime.utcnow()
        )
        
        processor = BatchProcessor(generation_pipeline=mock_pipeline)
        await processor.start()
        
        try:
            # Submit multiple jobs
            job_ids = []
            for i in range(5):
                requests = [Mock(spec=GenerationRequest, prompt=f"Prompt {j}") for j in range(2)]
                job_id = await processor.submit_batch_job(
                    name=f"Concurrent Job {i}",
                    user_id="user123",
                    requests=requests
                )
                job_ids.append(job_id)
            
            # Wait for all jobs to complete
            max_wait = 60
            wait_time = 0
            while wait_time < max_wait:
                completed_jobs = sum(1 for job_id in job_ids 
                                   if processor.jobs[job_id].status == BatchJobStatus.COMPLETED)
                if completed_jobs == len(job_ids):
                    break
                await asyncio.sleep(1)
                wait_time += 1
            
            # Verify all jobs completed
            for job_id in job_ids:
                job = processor.jobs[job_id]
                assert job.status == BatchJobStatus.COMPLETED
                assert job.completed_requests == 2
        
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_resource_limits_enforcement(self):
        """Test that resource limits are properly enforced."""
        # Create processor with very limited resources
        processor = BatchProcessor()
        processor.resource_allocator = ResourceAllocator(
            max_concurrent_jobs=1,
            max_concurrent_requests=2
        )
        await processor.start()
        
        try:
            # Submit jobs that exceed limits
            job_ids = []
            for i in range(3):
                requests = [Mock(spec=GenerationRequest, prompt=f"Prompt {j}") for j in range(3)]
                job_id = await processor.submit_batch_job(
                    name=f"Limited Job {i}",
                    user_id="user123",
                    requests=requests
                )
                job_ids.append(job_id)
            
            # Wait a bit for processing to start
            await asyncio.sleep(2)
            
            # Check that only one job is running
            running_jobs = [job for job in processor.jobs.values() 
                          if job.status == BatchJobStatus.RUNNING]
            assert len(running_jobs) <= 1
            
            # Check that resource limits are respected
            assert len(processor.resource_allocator.active_jobs) <= 1
            assert len(processor.resource_allocator.active_requests) <= 2
        
        finally:
            await processor.stop()


if __name__ == "__main__":
    pytest.main([__file__])