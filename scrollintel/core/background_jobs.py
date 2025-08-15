"""
Background Job Processing System for ScrollIntel.
Handles long-running tasks like file processing, model training, and data analysis.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import traceback

import redis.asyncio as redis
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..core.config import get_settings
from ..core.database_pool import get_optimized_db_pool
from ..models.database import Base
from ..core.interfaces import EngineError

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class JobResult:
    """Result of job execution."""
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobDefinition:
    """Definition of a background job."""
    job_id: str
    job_type: str
    function_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 1800  # 30 minutes
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Track job progress and provide updates."""
    
    def __init__(self, job_id: str, redis_client: Optional[redis.Redis] = None):
        self.job_id = job_id
        self.redis_client = redis_client
        self._progress = 0.0
        self._message = ""
        self._metadata = {}
    
    async def update(
        self,
        progress: float,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update job progress."""
        self._progress = max(0.0, min(100.0, progress))
        self._message = message
        if metadata:
            self._metadata.update(metadata)
        
        # Store progress in Redis if available
        if self.redis_client:
            try:
                progress_data = {
                    'progress': self._progress,
                    'message': self._message,
                    'metadata': self._metadata,
                    'updated_at': datetime.utcnow().isoformat()
                }
                
                await self.redis_client.setex(
                    f"job_progress:{self.job_id}",
                    3600,  # 1 hour TTL
                    json.dumps(progress_data)
                )
            except Exception as e:
                logger.warning(f"Failed to update progress in Redis: {e}")
    
    async def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"job_progress:{self.job_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Failed to get progress from Redis: {e}")
        
        return {
            'progress': self._progress,
            'message': self._message,
            'metadata': self._metadata,
            'updated_at': datetime.utcnow().isoformat()
        }


class JobQueue:
    """Priority-based job queue with Redis backend."""
    
    def __init__(self, redis_client: redis.Redis, queue_name: str = "scrollintel_jobs"):
        self.redis_client = redis_client
        self.queue_name = queue_name
        self.processing_queue = f"{queue_name}:processing"
        self.failed_queue = f"{queue_name}:failed"
        self.completed_queue = f"{queue_name}:completed"
    
    async def enqueue(self, job: JobDefinition) -> str:
        """Add job to queue."""
        try:
            job_data = {
                'job_id': job.job_id,
                'job_type': job.job_type,
                'function_name': job.function_name,
                'args': job.args,
                'kwargs': job.kwargs,
                'priority': job.priority.value,
                'max_retries': job.max_retries,
                'retry_delay': job.retry_delay,
                'timeout': job.timeout,
                'created_at': job.created_at.isoformat(),
                'scheduled_at': job.scheduled_at.isoformat() if job.scheduled_at else None,
                'user_id': job.user_id,
                'metadata': job.metadata,
                'retry_count': 0
            }
            
            # Add to priority queue (higher priority = higher score)
            await self.redis_client.zadd(
                self.queue_name,
                {json.dumps(job_data): job.priority.value}
            )
            
            # Store job details
            await self.redis_client.setex(
                f"job:{job.job_id}",
                86400,  # 24 hours TTL
                json.dumps(job_data)
            )
            
            logger.info(f"Job {job.job_id} enqueued with priority {job.priority}")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue job {job.job_id}: {e}")
            raise
    
    async def dequeue(self, timeout: int = 10) -> Optional[JobDefinition]:
        """Get next job from queue."""
        try:
            # Get highest priority job
            result = await self.redis_client.zpopmax(self.queue_name, 1)
            
            if not result:
                return None
            
            job_data_str, priority = result[0]
            job_data = json.loads(job_data_str)
            
            # Move to processing queue
            await self.redis_client.setex(
                f"processing:{job_data['job_id']}",
                3600,  # 1 hour TTL
                job_data_str
            )
            
            # Convert back to JobDefinition
            job = JobDefinition(
                job_id=job_data['job_id'],
                job_type=job_data['job_type'],
                function_name=job_data['function_name'],
                args=job_data['args'],
                kwargs=job_data['kwargs'],
                priority=JobPriority(job_data['priority']),
                max_retries=job_data['max_retries'],
                retry_delay=job_data['retry_delay'],
                timeout=job_data['timeout'],
                created_at=datetime.fromisoformat(job_data['created_at']),
                scheduled_at=datetime.fromisoformat(job_data['scheduled_at']) if job_data['scheduled_at'] else None,
                user_id=job_data['user_id'],
                metadata=job_data['metadata']
            )
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to dequeue job: {e}")
            return None
    
    async def complete_job(self, job_id: str, result: JobResult) -> None:
        """Mark job as completed."""
        try:
            # Remove from processing
            await self.redis_client.delete(f"processing:{job_id}")
            
            # Add to completed queue
            result_data = {
                'job_id': result.job_id,
                'status': result.status.value,
                'result': result.result,
                'error': result.error,
                'execution_time': result.execution_time,
                'started_at': result.started_at.isoformat() if result.started_at else None,
                'completed_at': result.completed_at.isoformat() if result.completed_at else None,
                'progress': result.progress,
                'metadata': result.metadata
            }
            
            await self.redis_client.setex(
                f"result:{job_id}",
                86400,  # 24 hours TTL
                json.dumps(result_data)
            )
            
            logger.info(f"Job {job_id} completed with status {result.status}")
            
        except Exception as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
    
    async def fail_job(self, job_id: str, error: str, retry: bool = False) -> None:
        """Mark job as failed."""
        try:
            if retry:
                # Get job data and increment retry count
                job_data_str = await self.redis_client.get(f"processing:{job_id}")
                if job_data_str:
                    job_data = json.loads(job_data_str)
                    job_data['retry_count'] = job_data.get('retry_count', 0) + 1
                    
                    if job_data['retry_count'] < job_data['max_retries']:
                        # Re-queue for retry
                        await self.redis_client.zadd(
                            self.queue_name,
                            {json.dumps(job_data): job_data['priority']}
                        )
                        logger.info(f"Job {job_id} queued for retry ({job_data['retry_count']}/{job_data['max_retries']})")
                        return
            
            # Remove from processing
            await self.redis_client.delete(f"processing:{job_id}")
            
            # Add to failed queue
            failure_data = {
                'job_id': job_id,
                'error': error,
                'failed_at': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                f"failed:{job_id}",
                86400,  # 24 hours TTL
                json.dumps(failure_data)
            )
            
            logger.error(f"Job {job_id} failed: {error}")
            
        except Exception as e:
            logger.error(f"Failed to fail job {job_id}: {e}")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            pending_count = await self.redis_client.zcard(self.queue_name)
            processing_keys = await self.redis_client.keys("processing:*")
            processing_count = len(processing_keys)
            
            failed_keys = await self.redis_client.keys("failed:*")
            failed_count = len(failed_keys)
            
            completed_keys = await self.redis_client.keys("result:*")
            completed_count = len(completed_keys)
            
            return {
                'pending': pending_count,
                'processing': processing_count,
                'failed': failed_count,
                'completed': completed_count,
                'total': pending_count + processing_count + failed_count + completed_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {'error': str(e)}


class BackgroundJobProcessor:
    """Process background jobs with multiple workers."""
    
    def __init__(self, max_workers: int = 4, max_process_workers: int = 2):
        self.settings = get_settings()
        self.max_workers = max_workers
        self.max_process_workers = max_process_workers
        self.redis_client: Optional[redis.Redis] = None
        self.job_queue: Optional[JobQueue] = None
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_process_workers)
        self.running = False
        self.workers: List[asyncio.Task] = []
        
        # Registry of job functions
        self.job_functions: Dict[str, Callable] = {}
    
    async def initialize(self) -> None:
        """Initialize the job processor."""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                password=self.settings.redis_password,
                db=self.settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            await self.redis_client.ping()
            
            # Initialize job queue
            self.job_queue = JobQueue(self.redis_client)
            
            logger.info("Background job processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize job processor: {e}")
            raise
    
    def register_job_function(self, name: str, func: Callable) -> None:
        """Register a job function."""
        self.job_functions[name] = func
        logger.info(f"Registered job function: {name}")
    
    async def submit_job(
        self,
        job_type: str,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: JobPriority = JobPriority.NORMAL,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """Submit a job for background processing."""
        if not self.job_queue:
            raise EngineError("Job processor not initialized")
        
        job_id = str(uuid.uuid4())
        
        job = JobDefinition(
            job_id=job_id,
            job_type=job_type,
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            user_id=user_id,
            metadata=metadata or {},
            scheduled_at=scheduled_at
        )
        
        await self.job_queue.enqueue(job)
        return job_id
    
    async def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get job result."""
        if not self.redis_client:
            return None
        
        try:
            result_data = await self.redis_client.get(f"result:{job_id}")
            if result_data:
                data = json.loads(result_data)
                return JobResult(
                    job_id=data['job_id'],
                    status=JobStatus(data['status']),
                    result=data['result'],
                    error=data['error'],
                    execution_time=data['execution_time'],
                    started_at=datetime.fromisoformat(data['started_at']) if data['started_at'] else None,
                    completed_at=datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None,
                    progress=data['progress'],
                    metadata=data['metadata']
                )
        except Exception as e:
            logger.error(f"Failed to get job result {job_id}: {e}")
        
        return None
    
    async def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get job progress."""
        if not self.redis_client:
            return {'progress': 0.0, 'message': 'Job processor not available'}
        
        tracker = ProgressTracker(job_id, self.redis_client)
        return await tracker.get_progress()
    
    async def start_workers(self) -> None:
        """Start background workers."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.workers.append(worker_task)
        
        logger.info(f"Started {self.max_workers} background workers")
    
    async def stop_workers(self) -> None:
        """Stop background workers."""
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("Background workers stopped")
    
    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop."""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get next job
                job = await self.job_queue.dequeue(timeout=10)
                
                if not job:
                    await asyncio.sleep(1)
                    continue
                
                # Check if job is scheduled for future
                if job.scheduled_at and job.scheduled_at > datetime.utcnow():
                    # Re-queue for later
                    await self.job_queue.enqueue(job)
                    await asyncio.sleep(1)
                    continue
                
                logger.info(f"Worker {worker_name} processing job {job.job_id}")
                
                # Execute job
                result = await self._execute_job(job)
                
                # Complete job
                await self.job_queue.complete_job(job.job_id, result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _execute_job(self, job: JobDefinition) -> JobResult:
        """Execute a single job."""
        start_time = time.time()
        result = JobResult(
            job_id=job.job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Get job function
            if job.function_name not in self.job_functions:
                raise EngineError(f"Job function '{job.function_name}' not registered")
            
            func = self.job_functions[job.function_name]
            
            # Create progress tracker
            progress_tracker = ProgressTracker(job.job_id, self.redis_client)
            
            # Add progress tracker to kwargs
            job.kwargs['progress_tracker'] = progress_tracker
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                # Async function
                job_result = await asyncio.wait_for(
                    func(*job.args, **job.kwargs),
                    timeout=job.timeout
                )
            else:
                # Sync function - run in thread pool
                job_result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_executor,
                    lambda: func(*job.args, **job.kwargs)
                )
            
            # Update final progress
            await progress_tracker.update(100.0, "Job completed successfully")
            
            result.status = JobStatus.COMPLETED
            result.result = job_result
            result.progress = 100.0
            
        except asyncio.TimeoutError:
            result.status = JobStatus.FAILED
            result.error = f"Job timed out after {job.timeout} seconds"
            logger.error(f"Job {job.job_id} timed out")
            
        except Exception as e:
            result.status = JobStatus.FAILED
            result.error = str(e)
            result.metadata['traceback'] = traceback.format_exc()
            logger.error(f"Job {job.job_id} failed: {e}")
        
        finally:
            result.execution_time = time.time() - start_time
            result.completed_at = datetime.utcnow()
        
        return result
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = {
            'running': self.running,
            'workers': len(self.workers),
            'max_workers': self.max_workers,
            'registered_functions': list(self.job_functions.keys()),
            'queue_stats': {}
        }
        
        if self.job_queue:
            stats['queue_stats'] = await self.job_queue.get_queue_stats()
        
        return stats
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.stop_workers()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


# Global job processor instance
_job_processor: Optional[BackgroundJobProcessor] = None


async def get_job_processor() -> BackgroundJobProcessor:
    """Get the global job processor."""
    global _job_processor
    
    if _job_processor is None:
        _job_processor = BackgroundJobProcessor()
        await _job_processor.initialize()
    
    return _job_processor


async def cleanup_job_processor() -> None:
    """Cleanup the global job processor."""
    global _job_processor
    
    if _job_processor:
        await _job_processor.cleanup()
        _job_processor = None


# Decorator for registering job functions
def background_job(job_type: str):
    """Decorator to register a function as a background job."""
    def decorator(func: Callable):
        async def register_func():
            try:
                processor = await get_job_processor()
                processor.register_job_function(func.__name__, func)
            except RuntimeError:
                # No event loop running, skip registration for now
                pass
        
        # Try to register the function when the module is imported
        try:
            asyncio.create_task(register_func())
        except RuntimeError:
            # No event loop running, skip registration for now
            pass
        
        return func
    return decorator