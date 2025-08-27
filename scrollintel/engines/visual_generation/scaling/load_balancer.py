"""
Load balancer for visual generation workers.
Implements intelligent load balancing across multiple generation workers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from enum import Enum

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    HEALTHY = "healthy"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class WorkerNode:
    """Represents a visual generation worker node."""
    id: str
    endpoint: str
    capabilities: List[str]  # e.g., ['image_generation', 'video_generation']
    max_concurrent_jobs: int = 4
    current_jobs: int = 0
    status: WorkerStatus = WorkerStatus.HEALTHY
    last_heartbeat: datetime = field(default_factory=datetime.now)
    average_response_time: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    gpu_memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)."""
        if self.max_concurrent_jobs == 0:
            return 1.0
        return self.current_jobs / self.max_concurrent_jobs
    
    @property
    def health_score(self) -> float:
        """Calculate health score based on multiple factors."""
        if self.status == WorkerStatus.OFFLINE:
            return 0.0
        
        # Base score from status
        status_score = {
            WorkerStatus.HEALTHY: 1.0,
            WorkerStatus.BUSY: 0.7,
            WorkerStatus.UNHEALTHY: 0.3
        }.get(self.status, 0.0)
        
        # Adjust for load
        load_penalty = self.load_factor * 0.3
        
        # Adjust for response time (normalize to 0-1 range)
        response_penalty = min(self.average_response_time / 30.0, 0.2)
        
        # Adjust for failure rate
        failure_rate = self.failed_requests / max(self.total_requests, 1)
        failure_penalty = failure_rate * 0.2
        
        return max(0.0, status_score - load_penalty - response_penalty - failure_penalty)


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"
    CAPABILITY_AWARE = "capability_aware"


class VisualGenerationLoadBalancer:
    """
    Intelligent load balancer for visual generation workers.
    Supports multiple load balancing strategies and health monitoring.
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED):
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.round_robin_index = 0
        self.health_check_interval = 30  # seconds
        self.unhealthy_threshold = 3  # failed health checks
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the load balancer and health monitoring."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Visual generation load balancer started")
    
    async def stop(self):
        """Stop the load balancer."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Visual generation load balancer stopped")
    
    def register_worker(self, worker: WorkerNode):
        """Register a new worker node."""
        self.workers[worker.id] = worker
        logger.info(f"Registered worker {worker.id} with capabilities: {worker.capabilities}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker node."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Unregistered worker {worker_id}")
    
    async def select_worker(self, request_type: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[WorkerNode]:
        """
        Select the best worker for a request based on the load balancing strategy.
        
        Args:
            request_type: Type of request ('image_generation', 'video_generation', etc.)
            requirements: Additional requirements (GPU memory, etc.)
        
        Returns:
            Selected worker node or None if no suitable worker available
        """
        # Filter workers by capability and availability
        available_workers = [
            worker for worker in self.workers.values()
            if (request_type in worker.capabilities and 
                worker.status in [WorkerStatus.HEALTHY, WorkerStatus.BUSY] and
                worker.load_factor < 1.0 and
                self._meets_requirements(worker, requirements))
        ]
        
        if not available_workers:
            logger.warning(f"No available workers for request type: {request_type}")
            return None
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_workers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_workers)
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_select(available_workers)
        elif self.strategy == LoadBalancingStrategy.CAPABILITY_AWARE:
            return self._capability_aware_select(available_workers, request_type, requirements)
        else:
            return available_workers[0]  # Fallback
    
    def _meets_requirements(self, worker: WorkerNode, requirements: Optional[Dict[str, Any]]) -> bool:
        """Check if worker meets specific requirements."""
        if not requirements:
            return True
        
        # Check GPU memory requirement
        if 'min_gpu_memory' in requirements:
            # Assume worker has enough free GPU memory if usage is below threshold
            free_gpu_memory = (1.0 - worker.gpu_memory_usage) * 100  # Simplified calculation
            if free_gpu_memory < requirements['min_gpu_memory']:
                return False
        
        # Check CPU requirement
        if 'max_cpu_usage' in requirements:
            if worker.cpu_usage > requirements['max_cpu_usage']:
                return False
        
        return True
    
    def _round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin selection."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_connections_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least current connections."""
        return min(workers, key=lambda w: w.current_jobs)
    
    def _weighted_round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin based on worker capacity."""
        # Create weighted list based on max concurrent jobs
        weighted_workers = []
        for worker in workers:
            weight = max(1, worker.max_concurrent_jobs - worker.current_jobs)
            weighted_workers.extend([worker] * weight)
        
        if not weighted_workers:
            return workers[0]
        
        worker = weighted_workers[self.round_robin_index % len(weighted_workers)]
        self.round_robin_index += 1
        return worker
    
    def _health_based_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on health score."""
        return max(workers, key=lambda w: w.health_score)
    
    def _capability_aware_select(self, workers: List[WorkerNode], request_type: str, 
                                requirements: Optional[Dict[str, Any]]) -> WorkerNode:
        """Advanced selection considering capabilities and requirements."""
        # Score workers based on multiple factors
        scored_workers = []
        
        for worker in workers:
            score = worker.health_score
            
            # Bonus for specialized capabilities
            if request_type == 'video_generation' and 'video_generation' in worker.capabilities:
                score += 0.2
            elif request_type == 'image_generation' and 'image_generation' in worker.capabilities:
                score += 0.1
            
            # Penalty for high load
            score -= worker.load_factor * 0.3
            
            # Bonus for low response time
            if worker.average_response_time < 10.0:
                score += 0.1
            
            scored_workers.append((score, worker))
        
        # Return worker with highest score
        return max(scored_workers, key=lambda x: x[0])[1]
    
    async def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """Update worker metrics from health check or job completion."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        
        # Update metrics
        if 'current_jobs' in metrics:
            worker.current_jobs = metrics['current_jobs']
        if 'average_response_time' in metrics:
            worker.average_response_time = metrics['average_response_time']
        if 'gpu_memory_usage' in metrics:
            worker.gpu_memory_usage = metrics['gpu_memory_usage']
        if 'cpu_usage' in metrics:
            worker.cpu_usage = metrics['cpu_usage']
        if 'status' in metrics:
            worker.status = WorkerStatus(metrics['status'])
        
        worker.last_heartbeat = datetime.now()
    
    async def report_job_completion(self, worker_id: str, success: bool, response_time: float):
        """Report job completion to update worker statistics."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        worker.total_requests += 1
        
        if not success:
            worker.failed_requests += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        worker.average_response_time = (
            alpha * response_time + (1 - alpha) * worker.average_response_time
        )
        
        # Decrease current job count
        worker.current_jobs = max(0, worker.current_jobs - 1)
    
    async def _health_check_loop(self):
        """Continuous health checking of worker nodes."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered workers."""
        health_check_tasks = []
        
        for worker_id, worker in self.workers.items():
            task = asyncio.create_task(self._check_worker_health(worker))
            health_check_tasks.append(task)
        
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_worker_health(self, worker: WorkerNode):
        """Check health of a specific worker."""
        try:
            # Simulate health check (in real implementation, this would be an HTTP request)
            # For now, mark workers as unhealthy if they haven't sent heartbeat recently
            time_since_heartbeat = datetime.now() - worker.last_heartbeat
            
            if time_since_heartbeat > timedelta(seconds=self.health_check_interval * 2):
                if worker.status != WorkerStatus.OFFLINE:
                    worker.status = WorkerStatus.UNHEALTHY
                    logger.warning(f"Worker {worker.id} marked as unhealthy - no recent heartbeat")
            
            # Mark as offline if no heartbeat for extended period
            if time_since_heartbeat > timedelta(seconds=self.health_check_interval * 4):
                if worker.status != WorkerStatus.OFFLINE:
                    worker.status = WorkerStatus.OFFLINE
                    logger.error(f"Worker {worker.id} marked as offline")
        
        except Exception as e:
            logger.error(f"Health check failed for worker {worker.id}: {e}")
            worker.status = WorkerStatus.UNHEALTHY
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status and metrics."""
        total_workers = len(self.workers)
        healthy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.HEALTHY)
        busy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
        unhealthy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.UNHEALTHY)
        offline_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.OFFLINE)
        
        total_capacity = sum(w.max_concurrent_jobs for w in self.workers.values())
        current_load = sum(w.current_jobs for w in self.workers.values())
        
        return {
            'total_workers': total_workers,
            'healthy_workers': healthy_workers,
            'busy_workers': busy_workers,
            'unhealthy_workers': unhealthy_workers,
            'offline_workers': offline_workers,
            'total_capacity': total_capacity,
            'current_load': current_load,
            'load_percentage': (current_load / max(total_capacity, 1)) * 100,
            'average_response_time': sum(w.average_response_time for w in self.workers.values()) / max(total_workers, 1),
            'strategy': self.strategy.value
        }


# Global load balancer instance
load_balancer = VisualGenerationLoadBalancer()