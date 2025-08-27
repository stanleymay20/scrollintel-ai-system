"""
Horizontal scaling coordinator for visual generation system.
Orchestrates load balancing, caching, session management, and database optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .load_balancer import VisualGenerationLoadBalancer, WorkerNode, LoadBalancingStrategy
from .distributed_cache import DistributedVisualCache, CacheStrategy
from .session_manager import DistributedSessionManager, GenerationRequest, RequestStatus
from .database_optimizer import DatabaseOptimizer, ConnectionPoolConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for horizontal scaling decisions."""
    total_workers: int
    healthy_workers: int
    average_load: float
    queue_length: int
    cache_hit_rate: float
    database_performance: float
    requests_per_second: float
    average_response_time: float
    error_rate: float
    timestamp: datetime


class HorizontalScalingCoordinator:
    """
    Coordinates all horizontal scaling components for visual generation.
    Manages load balancing, caching, sessions, and database optimization.
    """
    
    def __init__(self, 
                 database_url: str,
                 redis_nodes: List[Dict[str, Any]] = None,
                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED,
                 cache_strategy: CacheStrategy = CacheStrategy.HYBRID):
        
        # Initialize components
        self.load_balancer = VisualGenerationLoadBalancer(load_balancing_strategy)
        self.distributed_cache = DistributedVisualCache(redis_nodes, cache_strategy)
        self.session_manager = DistributedSessionManager(redis_nodes)
        self.database_optimizer = DatabaseOptimizer(database_url)
        
        # Scaling configuration
        self.auto_scaling_enabled = True
        self.min_workers = 2
        self.max_workers = 20
        self.scale_up_threshold = 0.8  # 80% load
        self.scale_down_threshold = 0.3  # 30% load
        self.scaling_cooldown = 300  # 5 minutes between scaling actions
        
        # Monitoring
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scaling_action: Optional[datetime] = None
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._request_processor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self):
        """Initialize all scaling components."""
        logger.info("Initializing horizontal scaling coordinator...")
        
        try:
            # Initialize all components
            await self.load_balancer.start()
            await self.distributed_cache.initialize()
            await self.session_manager.initialize()
            await self.database_optimizer.initialize()
            
            # Start background tasks
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._request_processor_task = asyncio.create_task(self._request_processing_loop())
            
            logger.info("Horizontal scaling coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize horizontal scaling coordinator: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown all scaling components."""
        logger.info("Shutting down horizontal scaling coordinator...")
        
        self._running = False
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._request_processor_task:
            self._request_processor_task.cancel()
        
        # Wait for tasks to complete
        tasks = [self._monitoring_task, self._request_processor_task]
        for task in tasks:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown components
        await self.load_balancer.stop()
        await self.distributed_cache.close()
        await self.session_manager.close()
        await self.database_optimizer.close()
        
        logger.info("Horizontal scaling coordinator shutdown complete")
    
    async def register_worker(self, worker_id: str, endpoint: str, 
                            capabilities: List[str], max_concurrent_jobs: int = 4):
        """Register a new worker node."""
        worker = WorkerNode(
            id=worker_id,
            endpoint=endpoint,
            capabilities=capabilities,
            max_concurrent_jobs=max_concurrent_jobs
        )
        
        self.load_balancer.register_worker(worker)
        logger.info(f"Registered worker {worker_id} with capabilities: {capabilities}")
    
    async def unregister_worker(self, worker_id: str):
        """Unregister a worker node."""
        self.load_balancer.unregister_worker(worker_id)
        logger.info(f"Unregistered worker {worker_id}")
    
    async def submit_generation_request(self, user_id: str, request_type: str, 
                                      prompt: str, parameters: Dict[str, Any],
                                      priority: int = 0) -> Optional[str]:
        """
        Submit a new generation request.
        
        Returns:
            Request ID if successful, None if rejected
        """
        try:
            # Get or create user session
            session = await self._get_or_create_session(user_id)
            if not session:
                logger.error(f"Failed to get session for user {user_id}")
                return None
            
            # Check cache first
            cached_result = await self.distributed_cache.get(prompt, parameters, request_type)
            if cached_result:
                logger.info(f"Cache hit for request - returning cached result")
                # Create a completed request record
                request = await self.session_manager.create_request(
                    session.id, request_type, prompt, parameters, priority
                )
                if request:
                    await self.session_manager.complete_request(
                        request.id, [f"cached_result_{cached_result.content_hash}"]
                    )
                    return request.id
            
            # Create new request
            request = await self.session_manager.create_request(
                session.id, request_type, prompt, parameters, priority
            )
            
            if request:
                logger.info(f"Created generation request {request.id} for user {user_id}")
                return request.id
            else:
                logger.warning(f"Request rejected for user {user_id} - limits exceeded")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting generation request: {e}")
            return None
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a generation request."""
        request = await self.session_manager.get_request(request_id)
        if not request:
            return None
        
        return {
            'id': request.id,
            'status': request.status.value,
            'progress': request.progress,
            'result_urls': request.result_urls,
            'error_message': request.error_message,
            'created_at': request.created_at.isoformat(),
            'processing_time': request.processing_time
        }
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a generation request."""
        try:
            await self.session_manager.cancel_request(request_id)
            logger.info(f"Cancelled request {request_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling request {request_id}: {e}")
            return False
    
    async def get_user_requests(self, user_id: str, status: Optional[str] = None,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get requests for a user."""
        try:
            request_status = RequestStatus(status) if status else None
            requests = await self.session_manager.get_user_requests(
                user_id, request_status, limit
            )
            
            return [
                {
                    'id': req.id,
                    'request_type': req.request_type,
                    'prompt': req.prompt,
                    'status': req.status.value,
                    'progress': req.progress,
                    'result_urls': req.result_urls,
                    'created_at': req.created_at.isoformat(),
                    'processing_time': req.processing_time
                }
                for req in requests
            ]
            
        except Exception as e:
            logger.error(f"Error getting user requests: {e}")
            return []
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            # Get metrics from all components
            cluster_status = self.load_balancer.get_cluster_status()
            cache_stats = await self.distributed_cache.get_cache_stats()
            queue_status = await self.session_manager.get_queue_status()
            db_metrics = await self.database_optimizer.get_performance_metrics()
            
            return {
                'cluster': cluster_status,
                'cache': cache_stats,
                'queue': queue_status,
                'database': db_metrics,
                'scaling': {
                    'auto_scaling_enabled': self.auto_scaling_enabled,
                    'last_scaling_action': self.last_scaling_action.isoformat() if self.last_scaling_action else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def _get_or_create_session(self, user_id: str):
        """Get existing session or create new one for user."""
        # Try to find existing active session
        # In a real implementation, you'd query by user_id
        # For now, create a new session
        return await self.session_manager.create_session(user_id)
    
    async def _monitoring_loop(self):
        """Background monitoring and auto-scaling loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = await self._collect_scaling_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history
                    if m.timestamp > cutoff_time
                ]
                
                # Make scaling decisions
                if self.auto_scaling_enabled:
                    await self._make_scaling_decision(metrics)
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _request_processing_loop(self):
        """Background loop to process queued requests."""
        while self._running:
            try:
                # Get queued requests from database
                queued_requests = await self.database_optimizer.get_generation_requests_batch(
                    'queued', limit=100
                )
                
                # Process each request
                for request_data in queued_requests:
                    await self._process_queued_request(request_data)
                
                # Sleep if no requests to process
                if not queued_requests:
                    await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in request processing loop: {e}")
                await asyncio.sleep(10)
    
    async def _process_queued_request(self, request_data: Dict[str, Any]):
        """Process a single queued request."""
        try:
            request_id = request_data['id']
            request_type = request_data['request_type']
            
            # Select appropriate worker
            worker = await self.load_balancer.select_worker(request_type)
            if not worker:
                logger.warning(f"No available worker for request {request_id}")
                return
            
            # Assign request to worker
            await self.session_manager.start_request_processing(request_id, worker.id)
            
            # Update worker metrics
            await self.load_balancer.update_worker_metrics(worker.id, {
                'current_jobs': worker.current_jobs + 1
            })
            
            logger.info(f"Assigned request {request_id} to worker {worker.id}")
            
        except Exception as e:
            logger.error(f"Error processing queued request: {e}")
    
    async def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect metrics for scaling decisions."""
        cluster_status = self.load_balancer.get_cluster_status()
        cache_stats = await self.distributed_cache.get_cache_stats()
        queue_status = await self.session_manager.get_queue_status()
        db_metrics = await self.database_optimizer.get_performance_metrics()
        
        return ScalingMetrics(
            total_workers=cluster_status['total_workers'],
            healthy_workers=cluster_status['healthy_workers'],
            average_load=cluster_status['load_percentage'] / 100.0,
            queue_length=queue_status['queued_requests'],
            cache_hit_rate=cache_stats['hit_rate'],
            database_performance=1.0 - db_metrics['error_rate'],  # Simplified metric
            requests_per_second=db_metrics['queries_per_second'],
            average_response_time=cluster_status['average_response_time'],
            error_rate=db_metrics['error_rate'],
            timestamp=datetime.now()
        )
    
    async def _make_scaling_decision(self, metrics: ScalingMetrics):
        """Make auto-scaling decisions based on metrics."""
        # Check if we're in cooldown period
        if (self.last_scaling_action and 
            datetime.now() - self.last_scaling_action < timedelta(seconds=self.scaling_cooldown)):
            return
        
        # Scale up conditions
        should_scale_up = (
            metrics.average_load > self.scale_up_threshold or
            metrics.queue_length > 50 or
            metrics.average_response_time > 30.0
        )
        
        # Scale down conditions
        should_scale_down = (
            metrics.average_load < self.scale_down_threshold and
            metrics.queue_length < 5 and
            metrics.healthy_workers > self.min_workers
        )
        
        if should_scale_up and metrics.total_workers < self.max_workers:
            await self._scale_up()
        elif should_scale_down and metrics.total_workers > self.min_workers:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up by adding more workers."""
        try:
            # In a real implementation, this would trigger cloud infrastructure
            # to spin up new worker instances
            logger.info("Scaling up - would add new worker instances")
            self.last_scaling_action = datetime.now()
            
            # Placeholder for actual scaling logic
            # await self._provision_new_worker()
            
        except Exception as e:
            logger.error(f"Error scaling up: {e}")
    
    async def _scale_down(self):
        """Scale down by removing workers."""
        try:
            # In a real implementation, this would gracefully remove workers
            logger.info("Scaling down - would remove worker instances")
            self.last_scaling_action = datetime.now()
            
            # Placeholder for actual scaling logic
            # await self._decommission_worker()
            
        except Exception as e:
            logger.error(f"Error scaling down: {e}")


# Global horizontal scaling coordinator
horizontal_scaler = HorizontalScalingCoordinator(
    database_url="postgresql://user:password@localhost/visual_generation"
)