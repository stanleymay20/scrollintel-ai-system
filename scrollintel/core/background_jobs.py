"""
Background Job Processing System for Heavy Volume Operations
Handles asynchronous processing of large datasets and long-running tasks.
"""

import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
import traceback

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class JobResult:
    """Job execution result."""
    job_id: str
    status: JobStatus
    result: Any = None
    error: str = None
    start_time: datetime = None
    end_time: datetime = None
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class Job:
    """Background job definition."""
    job_id: str
    job_type: str
    function_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    priority: JobPriority
    user_id: str
    created_at: datetime
    scheduled_at: datetime = None
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 3600  # 1 hour default
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        if self.scheduled_at:
            data['scheduled_at'] = self.scheduled_at.isoformat()
        return data


class BackgroundJobProcessor:
    """Processes background jobs for heavy volume operations."""
    
    def __init__(self, max_concurrent_jobs: int = 10):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.job_results: Dict[str, JobResult] = {}
        self.job_registry: Dict[str, Callable] = {}
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Performance monitoring
        self.stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'jobs_cancelled': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
    
    def register_job_function(self, name: str, function: Callable) -> None:
        """Register a function that can be executed as a background job."""
        self.job_registry[name] = function
        logger.info(f"Registered job function: {name}")
    
    async def submit_job(
        self,
        job_type: str,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: JobPriority = JobPriority.NORMAL,
        user_id: str = None,
        scheduled_at: datetime = None,
        max_retries: int = 3,
        timeout_seconds: int = 3600,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Submit a job for background processing."""
        
        job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            user_id=user_id,
            created_at=datetime.utcnow(),
            scheduled_at=scheduled_at,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {}
        )
        
        # Add to queue with priority (lower number = higher priority)
        priority_value = 5 - priority.value  # Invert for queue ordering
        await self.job_queue.put((priority_value, datetime.utcnow(), job))
        
        self.stats['jobs_submitted'] += 1
        
        logger.info(f"Submitted job {job_id}: {job_type}.{function_name} "
                   f"(priority: {priority.name})")
        
        return job_id
    
    async def start_processing(self) -> None:
        """Start the background job processing workers."""
        
        if self.is_running:
            logger.warning("Job processor is already running")
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent_jobs):
            worker_task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        logger.info(f"Started {self.max_concurrent_jobs} background job workers")
    
    async def stop_processing(self) -> None:
        """Stop the background job processing workers."""
        
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Cancel running jobs
        for job_id, task in self.running_jobs.items():
            task.cancel()
            logger.info(f"Cancelled running job: {job_id}")
        
        self.worker_tasks.clear()
        self.running_jobs.clear()
        
        logger.info("Background job processor stopped")
    
    async def _worker(self, worker_name: str) -> None:
        """Background worker that processes jobs from the queue."""
        
        logger.info(f"Started background worker: {worker_name}")
        
        while self.is_running:
            try:
                # Get job from queue with timeout
                try:
                    priority, queued_at, job = await asyncio.wait_for(
                        self.job_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if job should be executed now
                if job.scheduled_at and job.scheduled_at > datetime.utcnow():
                    # Re-queue for later
                    await self.job_queue.put((priority, queued_at, job))
                    await asyncio.sleep(1)
                    continue
                
                # Execute job
                await self._execute_job(job, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {str(e)}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _execute_job(self, job: Job, worker_name: str) -> None:
        """Execute a single job."""
        
        start_time = datetime.utcnow()
        
        # Create job result
        result = JobResult(
            job_id=job.job_id,
            status=JobStatus.RUNNING,
            start_time=start_time,
            metadata=job.metadata
        )
        
        self.job_results[job.job_id] = result
        
        logger.info(f"[{worker_name}] Executing job {job.job_id}: "
                   f"{job.job_type}.{job.function_name}")
        
        try:
            # Get the function to execute
            if job.function_name not in self.job_registry:
                raise ValueError(f"Unknown job function: {job.function_name}")
            
            function = self.job_registry[job.function_name]
            
            # Create execution task with timeout
            execution_task = asyncio.create_task(
                function(*job.args, **job.kwargs)
            )
            
            self.running_jobs[job.job_id] = execution_task
            
            # Execute with timeout
            job_result = await asyncio.wait_for(
                execution_task, timeout=job.timeout_seconds
            )
            
            # Job completed successfully
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            result.status = JobStatus.COMPLETED
            result.result = job_result
            result.end_time = end_time
            result.execution_time_seconds = execution_time
            
            self.stats['jobs_completed'] += 1
            self.stats['total_execution_time'] += execution_time
            self.stats['average_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['jobs_completed']
            )
            
            logger.info(f"[{worker_name}] Job {job.job_id} completed in "
                       f"{execution_time:.1f}s")
            
        except asyncio.TimeoutError:
            # Job timed out
            result.status = JobStatus.FAILED
            result.error = f"Job timed out after {job.timeout_seconds} seconds"
            result.end_time = datetime.utcnow()
            
            self.stats['jobs_failed'] += 1
            
            logger.error(f"[{worker_name}] Job {job.job_id} timed out")
            
        except asyncio.CancelledError:
            # Job was cancelled
            result.status = JobStatus.CANCELLED
            result.error = "Job was cancelled"
            result.end_time = datetime.utcnow()
            
            self.stats['jobs_cancelled'] += 1
            
            logger.info(f"[{worker_name}] Job {job.job_id} cancelled")
            
        except Exception as e:
            # Job failed with error
            result.status = JobStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.utcnow()
            
            self.stats['jobs_failed'] += 1
            
            logger.error(f"[{worker_name}] Job {job.job_id} failed: {str(e)}")
            logger.debug(f"Job error traceback: {traceback.format_exc()}")
            
            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                retry_delay = min(300, 2 ** job.retry_count)  # Exponential backoff, max 5 min
                job.scheduled_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                
                # Re-queue for retry
                priority_value = 5 - job.priority.value
                await self.job_queue.put((priority_value, datetime.utcnow(), job))
                
                logger.info(f"Job {job.job_id} scheduled for retry {job.retry_count}/"
                           f"{job.max_retries} in {retry_delay}s")
        
        finally:
            # Clean up
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
    
    async def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get the status and result of a job."""
        return self.job_results.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            logger.info(f"Cancelled job: {job_id}")
            return True
        
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get job processor statistics."""
        
        stats = self.stats.copy()
        stats.update({
            'running_jobs': len(self.running_jobs),
            'queued_jobs': self.job_queue.qsize(),
            'registered_functions': len(self.job_registry),
            'is_running': self.is_running,
            'max_concurrent_jobs': self.max_concurrent_jobs
        })
        
        return stats


# Global job processor instance
_job_processor: Optional[BackgroundJobProcessor] = None


async def get_job_processor() -> BackgroundJobProcessor:
    """Get or create the global job processor."""
    
    global _job_processor
    
    if _job_processor is None:
        _job_processor = BackgroundJobProcessor()
        await _job_processor.start_processing()
    
    return _job_processor


def background_job(
    job_type: str,
    priority: JobPriority = JobPriority.NORMAL,
    max_retries: int = 3,
    timeout_seconds: int = 3600
):
    """Decorator to register a function as a background job."""
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            processor = await get_job_processor()
            processor.register_job_function(func.__name__, func)
            return await func(*args, **kwargs)
        
        # Register the function when first called, not during import
        wrapper._job_type = job_type
        wrapper._priority = priority
        wrapper._max_retries = max_retries
        wrapper._timeout_seconds = timeout_seconds
        wrapper._registered = False
        
        return wrapper
    
    return decorator


async def _register_function(func: Callable) -> None:
    """Helper to register function with job processor."""
    try:
        processor = await get_job_processor()
        processor.register_job_function(func.__name__, func)
    except Exception as e:
        logger.error(f"Failed to register job function {func.__name__}: {str(e)}")