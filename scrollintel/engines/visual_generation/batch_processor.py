"""
Batch Processing System for Visual Content Generation

This module implements a comprehensive batch processing system that handles
multiple generation requests efficiently with job scheduling, resource allocation,
progress tracking, and result aggregation.

Requirements: 5.2, 7.2
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from scrollintel.engines.visual_generation.base import (
    GenerationRequest, GenerationResult, GenerationStatus
)
from scrollintel.engines.visual_generation.exceptions import BatchProcessingError
from scrollintel.core.config import get_config

logger = logging.getLogger(__name__)


class BatchJobStatus(Enum):
    """Status of batch processing jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class BatchJobPriority(Enum):
    """Priority levels for batch jobs."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchJobRequest:
    """Individual request within a batch job."""
    id: str
    request: GenerationRequest
    priority: BatchJobPriority = BatchJobPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[GenerationResult] = None
    error: Optional[str] = None
    status: GenerationStatus = GenerationStatus.PENDING


@dataclass
class BatchJob:
    """Batch processing job containing multiple generation requests."""
    id: str
    name: str
    user_id: str
    requests: List[BatchJobRequest]
    priority: BatchJobPriority = BatchJobPriority.NORMAL
    status: BatchJobStatus = BatchJobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    estimated_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_requests = len(self.requests)


@dataclass
class BatchProcessingMetrics:
    """Metrics for batch processing performance."""
    total_jobs: int = 0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    average_job_duration: float = 0.0
    throughput_per_hour: float = 0.0
    resource_utilization: float = 0.0
    queue_length: int = 0


class ResourceAllocator:
    """Manages resource allocation for batch processing."""
    
    def __init__(self, max_concurrent_jobs: int = 10, max_concurrent_requests: int = 50):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_concurrent_requests = max_concurrent_requests
        self.active_jobs: Dict[str, BatchJob] = {}
        self.active_requests: Dict[str, BatchJobRequest] = {}
        self.resource_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
    async def can_start_job(self, job: BatchJob) -> bool:
        """Check if resources are available to start a job."""
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return False
        
        estimated_requests = len([r for r in job.requests if r.status == GenerationStatus.PENDING])
        if len(self.active_requests) + estimated_requests > self.max_concurrent_requests:
            return False
        
        return True
    
    async def allocate_job(self, job: BatchJob) -> bool:
        """Allocate resources for a batch job."""
        if await self.can_start_job(job):
            self.active_jobs[job.id] = job
            return True
        return False
    
    async def deallocate_job(self, job_id: str):
        """Deallocate resources for a completed job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            # Remove all requests for this job
            for request in job.requests:
                if request.id in self.active_requests:
                    del self.active_requests[request.id]
            del self.active_jobs[job_id]
    
    async def allocate_request(self, request: BatchJobRequest) -> bool:
        """Allocate resources for an individual request."""
        await self.resource_semaphore.acquire()
        self.active_requests[request.id] = request
        return True
    
    async def deallocate_request(self, request_id: str):
        """Deallocate resources for a completed request."""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
            self.resource_semaphore.release()


class BatchJobScheduler:
    """Schedules and prioritizes batch jobs."""
    
    def __init__(self):
        self.job_queue: List[BatchJob] = []
        self.priority_weights = {
            BatchJobPriority.LOW: 1,
            BatchJobPriority.NORMAL: 2,
            BatchJobPriority.HIGH: 4,
            BatchJobPriority.URGENT: 8
        }
    
    def add_job(self, job: BatchJob):
        """Add a job to the scheduling queue."""
        self.job_queue.append(job)
        self._sort_queue()
    
    def get_next_job(self) -> Optional[BatchJob]:
        """Get the next job to process based on priority and creation time."""
        if not self.job_queue:
            return None
        
        # Filter pending jobs
        pending_jobs = [job for job in self.job_queue if job.status == BatchJobStatus.PENDING]
        if not pending_jobs:
            return None
        
        # Sort by priority and creation time
        pending_jobs.sort(key=lambda j: (
            -self.priority_weights[j.priority],  # Higher priority first
            j.created_at  # Earlier creation time first
        ))
        
        next_job = pending_jobs[0]
        self.job_queue.remove(next_job)
        return next_job
    
    def _sort_queue(self):
        """Sort the job queue by priority and creation time."""
        self.job_queue.sort(key=lambda j: (
            -self.priority_weights[j.priority],
            j.created_at
        ))
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue status by priority."""
        status = {priority.name: 0 for priority in BatchJobPriority}
        for job in self.job_queue:
            if job.status == BatchJobStatus.PENDING:
                status[job.priority.name] += 1
        return status


class ProgressTracker:
    """Tracks progress of batch jobs and individual requests."""
    
    def __init__(self):
        self.job_progress: Dict[str, float] = {}
        self.request_progress: Dict[str, float] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
    
    def update_request_progress(self, request_id: str, progress: float):
        """Update progress for an individual request."""
        self.request_progress[request_id] = max(0.0, min(1.0, progress))
    
    def update_job_progress(self, job: BatchJob):
        """Update overall progress for a batch job."""
        if not job.requests:
            job.progress = 0.0
            return
        
        total_progress = 0.0
        for request in job.requests:
            if request.status == GenerationStatus.COMPLETED:
                total_progress += 1.0
            elif request.status == GenerationStatus.IN_PROGRESS:
                request_progress = self.request_progress.get(request.id, 0.0)
                total_progress += request_progress
            # Pending and failed requests contribute 0
        
        job.progress = total_progress / len(job.requests)
        job.completed_requests = len([r for r in job.requests if r.status == GenerationStatus.COMPLETED])
        job.failed_requests = len([r for r in job.requests if r.status == GenerationStatus.FAILED])
        
        # Trigger progress callbacks
        if job.id in self.progress_callbacks:
            for callback in self.progress_callbacks[job.id]:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Progress callback error for job {job.id}: {e}")
    
    def add_progress_callback(self, job_id: str, callback: Callable):
        """Add a progress callback for a job."""
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)
    
    def remove_progress_callbacks(self, job_id: str):
        """Remove all progress callbacks for a job."""
        if job_id in self.progress_callbacks:
            del self.progress_callbacks[job_id]


class ResultAggregator:
    """Aggregates results from batch processing jobs."""
    
    def __init__(self):
        self.job_results: Dict[str, Dict[str, Any]] = {}
    
    def aggregate_job_results(self, job: BatchJob) -> Dict[str, Any]:
        """Aggregate results for a completed batch job."""
        successful_results = []
        failed_results = []
        
        for request in job.requests:
            if request.result and request.status == GenerationStatus.COMPLETED:
                successful_results.append({
                    'request_id': request.id,
                    'result': request.result,
                    'generation_time': request.result.generation_time,
                    'quality_score': request.result.quality_metrics.overall_score if request.result.quality_metrics else None
                })
            elif request.status == GenerationStatus.FAILED:
                failed_results.append({
                    'request_id': request.id,
                    'error': request.error,
                    'retry_count': request.retry_count
                })
        
        aggregated_results = {
            'job_id': job.id,
            'job_name': job.name,
            'total_requests': job.total_requests,
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'success_rate': len(successful_results) / job.total_requests if job.total_requests > 0 else 0,
            'total_duration': (job.completed_at - job.started_at).total_seconds() if job.completed_at and job.started_at else 0,
            'average_generation_time': sum(r['generation_time'] for r in successful_results) / len(successful_results) if successful_results else 0,
            'average_quality_score': sum(r['quality_score'] for r in successful_results if r['quality_score']) / len([r for r in successful_results if r['quality_score']]) if successful_results else 0,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'metadata': job.metadata
        }
        
        self.job_results[job.id] = aggregated_results
        return aggregated_results
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get aggregated results for a specific job."""
        return self.job_results.get(job_id)
    
    def export_results(self, job_id: str, format: str = 'json') -> str:
        """Export job results in specified format."""
        results = self.get_job_results(job_id)
        if not results:
            raise ValueError(f"No results found for job {job_id}")
        
        if format.lower() == 'json':
            return json.dumps(results, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class BatchProcessor:
    """Main batch processing system for visual content generation."""
    
    def __init__(self, generation_pipeline=None):
        self.config = get_config()
        self.generation_pipeline = generation_pipeline
        self.resource_allocator = ResourceAllocator(
            max_concurrent_jobs=self.config.get('batch_max_concurrent_jobs', 10),
            max_concurrent_requests=self.config.get('batch_max_concurrent_requests', 50)
        )
        self.scheduler = BatchJobScheduler()
        self.progress_tracker = ProgressTracker()
        self.result_aggregator = ResultAggregator()
        self.jobs: Dict[str, BatchJob] = {}
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.metrics = BatchProcessingMetrics()
        
    async def start(self):
        """Start the batch processing system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Batch processor started")
    
    async def stop(self):
        """Stop the batch processing system."""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Batch processor stopped")
    
    async def submit_batch_job(self, 
                              name: str,
                              user_id: str,
                              requests: List[GenerationRequest],
                              priority: BatchJobPriority = BatchJobPriority.NORMAL,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a new batch job for processing."""
        job_id = str(uuid.uuid4())
        
        # Create batch job requests
        batch_requests = []
        for i, request in enumerate(requests):
            batch_request = BatchJobRequest(
                id=f"{job_id}_{i}",
                request=request,
                priority=priority
            )
            batch_requests.append(batch_request)
        
        # Create batch job
        job = BatchJob(
            id=job_id,
            name=name,
            user_id=user_id,
            requests=batch_requests,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store and schedule job
        self.jobs[job_id] = job
        self.scheduler.add_job(job)
        self.metrics.total_jobs += 1
        
        logger.info(f"Submitted batch job {job_id} with {len(requests)} requests")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get the current status of a batch job."""
        return self.jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED]:
            return False
        
        job.status = BatchJobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        # Cancel pending requests
        for request in job.requests:
            if request.status == GenerationStatus.PENDING:
                request.status = GenerationStatus.CANCELLED
        
        await self.resource_allocator.deallocate_job(job_id)
        logger.info(f"Cancelled batch job {job_id}")
        return True
    
    async def pause_job(self, job_id: str) -> bool:
        """Pause a batch job."""
        job = self.jobs.get(job_id)
        if not job or job.status != BatchJobStatus.RUNNING:
            return False
        
        job.status = BatchJobStatus.PAUSED
        logger.info(f"Paused batch job {job_id}")
        return True
    
    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused batch job."""
        job = self.jobs.get(job_id)
        if not job or job.status != BatchJobStatus.PAUSED:
            return False
        
        job.status = BatchJobStatus.PENDING
        self.scheduler.add_job(job)
        logger.info(f"Resumed batch job {job_id}")
        return True
    
    async def get_metrics(self) -> BatchProcessingMetrics:
        """Get current batch processing metrics."""
        self.metrics.active_jobs = len([j for j in self.jobs.values() if j.status == BatchJobStatus.RUNNING])
        self.metrics.completed_jobs = len([j for j in self.jobs.values() if j.status == BatchJobStatus.COMPLETED])
        self.metrics.failed_jobs = len([j for j in self.jobs.values() if j.status == BatchJobStatus.FAILED])
        self.metrics.queue_length = len([j for j in self.jobs.values() if j.status == BatchJobStatus.PENDING])
        
        # Calculate average job duration
        completed_jobs = [j for j in self.jobs.values() if j.status == BatchJobStatus.COMPLETED and j.started_at and j.completed_at]
        if completed_jobs:
            total_duration = sum((j.completed_at - j.started_at).total_seconds() for j in completed_jobs)
            self.metrics.average_job_duration = total_duration / len(completed_jobs)
        
        return self.metrics
    
    async def _processing_loop(self):
        """Main processing loop for batch jobs."""
        while self.is_running:
            try:
                # Get next job to process
                next_job = self.scheduler.get_next_job()
                if not next_job:
                    await asyncio.sleep(1)
                    continue
                
                # Check if we can allocate resources
                if not await self.resource_allocator.allocate_job(next_job):
                    # Put job back in queue
                    self.scheduler.add_job(next_job)
                    await asyncio.sleep(5)
                    continue
                
                # Start processing job
                asyncio.create_task(self._process_job(next_job))
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_job(self, job: BatchJob):
        """Process a single batch job."""
        try:
            job.status = BatchJobStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            logger.info(f"Starting batch job {job.id} with {len(job.requests)} requests")
            
            # Process requests concurrently with resource limits
            tasks = []
            for request in job.requests:
                if request.status == GenerationStatus.PENDING:
                    task = asyncio.create_task(self._process_request(request))
                    tasks.append(task)
            
            # Wait for all requests to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update job status
            if all(r.status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED] for r in job.requests):
                job.status = BatchJobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                
                # Aggregate results
                self.result_aggregator.aggregate_job_results(job)
                
                logger.info(f"Completed batch job {job.id}")
            else:
                job.status = BatchJobStatus.FAILED
                job.completed_at = datetime.utcnow()
                logger.error(f"Batch job {job.id} failed")
            
        except Exception as e:
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.utcnow()
            logger.error(f"Error processing batch job {job.id}: {e}")
        
        finally:
            # Clean up resources
            await self.resource_allocator.deallocate_job(job.id)
            self.progress_tracker.remove_progress_callbacks(job.id)
    
    async def _process_request(self, request: BatchJobRequest):
        """Process a single generation request."""
        try:
            # Allocate resources for request
            await self.resource_allocator.allocate_request(request)
            
            request.status = GenerationStatus.IN_PROGRESS
            request.started_at = datetime.utcnow()
            
            # Update progress
            self.progress_tracker.update_request_progress(request.id, 0.1)
            
            # Generate content using the pipeline
            if self.generation_pipeline:
                result = await self.generation_pipeline.generate(request.request)
                request.result = result
                request.status = GenerationStatus.COMPLETED
            else:
                # Mock result for testing
                await asyncio.sleep(2)  # Simulate processing time
                request.status = GenerationStatus.COMPLETED
            
            request.completed_at = datetime.utcnow()
            self.progress_tracker.update_request_progress(request.id, 1.0)
            
        except Exception as e:
            request.status = GenerationStatus.FAILED
            request.error = str(e)
            request.completed_at = datetime.utcnow()
            
            # Retry logic
            if request.retry_count < request.max_retries:
                request.retry_count += 1
                request.status = GenerationStatus.PENDING
                logger.warning(f"Retrying request {request.id}, attempt {request.retry_count}")
            else:
                logger.error(f"Request {request.id} failed after {request.retry_count} retries: {e}")
        
        finally:
            # Clean up resources
            await self.resource_allocator.deallocate_request(request.id)


# Global batch processor instance
_batch_processor = None

async def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
        await _batch_processor.start()
    return _batch_processor