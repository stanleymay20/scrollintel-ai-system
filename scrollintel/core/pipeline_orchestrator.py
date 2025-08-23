"""
Data Pipeline Automation - Pipeline Orchestrator
Handles pipeline scheduling, execution ordering, retry mechanisms, and resource management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, PriorityQueue
import heapq

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.pipeline_models import Pipeline, PipelineExecution, PipelineNode, PipelineConnection
from ..core.database import get_db


class ExecutionStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ResourceType(Enum):
    """Resource types for allocation"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ExecutionContext:
    """Context for pipeline execution"""
    execution_id: str
    pipeline_id: str
    start_time: datetime
    status: ExecutionStatus = ExecutionStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 5  # 1-10, higher is more priority
    dependencies: Set[str] = field(default_factory=set)
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority > other.priority  # Higher priority first


@dataclass
class ResourceAllocation:
    """Resource allocation tracking"""
    total_cpu: float = 100.0
    total_memory: float = 100.0  # GB
    total_storage: float = 1000.0  # GB
    total_network: float = 100.0  # Mbps
    
    allocated_cpu: float = 0.0
    allocated_memory: float = 0.0
    allocated_storage: float = 0.0
    allocated_network: float = 0.0
    
    def can_allocate(self, requirements: Dict[ResourceType, float]) -> bool:
        """Check if resources can be allocated"""
        cpu_req = requirements.get(ResourceType.CPU, 0)
        memory_req = requirements.get(ResourceType.MEMORY, 0)
        storage_req = requirements.get(ResourceType.STORAGE, 0)
        network_req = requirements.get(ResourceType.NETWORK, 0)
        
        return (
            self.allocated_cpu + cpu_req <= self.total_cpu and
            self.allocated_memory + memory_req <= self.total_memory and
            self.allocated_storage + storage_req <= self.total_storage and
            self.allocated_network + network_req <= self.total_network
        )
    
    def allocate(self, requirements: Dict[ResourceType, float]) -> bool:
        """Allocate resources if available"""
        if not self.can_allocate(requirements):
            return False
        
        self.allocated_cpu += requirements.get(ResourceType.CPU, 0)
        self.allocated_memory += requirements.get(ResourceType.MEMORY, 0)
        self.allocated_storage += requirements.get(ResourceType.STORAGE, 0)
        self.allocated_network += requirements.get(ResourceType.NETWORK, 0)
        return True
    
    def deallocate(self, requirements: Dict[ResourceType, float]):
        """Deallocate resources"""
        self.allocated_cpu -= requirements.get(ResourceType.CPU, 0)
        self.allocated_memory -= requirements.get(ResourceType.MEMORY, 0)
        self.allocated_storage -= requirements.get(ResourceType.STORAGE, 0)
        self.allocated_network -= requirements.get(ResourceType.NETWORK, 0)
        
        # Ensure non-negative values
        self.allocated_cpu = max(0, self.allocated_cpu)
        self.allocated_memory = max(0, self.allocated_memory)
        self.allocated_storage = max(0, self.allocated_storage)
        self.allocated_network = max(0, self.allocated_network)


class ScheduleType(Enum):
    """Types of pipeline schedules"""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    EVENT_DRIVEN = "event_driven"


@dataclass
class ScheduleConfig:
    """Pipeline schedule configuration"""
    schedule_type: ScheduleType
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    enabled: bool = True
    timezone: str = "UTC"


class PipelineOrchestrator:
    """
    Main orchestrator for pipeline scheduling and execution management.
    Handles dependency resolution, resource allocation, and retry mechanisms.
    """
    
    def __init__(self, max_concurrent_executions: int = 10):
        self.logger = logging.getLogger(__name__)
        self.max_concurrent_executions = max_concurrent_executions
        self.execution_queue = PriorityQueue()
        self.running_executions: Dict[str, ExecutionContext] = {}
        self.completed_executions: Dict[str, ExecutionContext] = {}
        self.resource_manager = ResourceAllocation()
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_executions)
        self.is_running = False
        self.scheduler_thread = None
        self._lock = threading.Lock()
        
        # Monitoring metrics
        self.metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'retry_executions': 0,
            'average_execution_time': 0.0,
            'resource_utilization': {}
        }
    
    def start(self):
        """Start the orchestrator"""
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Pipeline orchestrator started")
    
    def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        self.logger.info("Pipeline orchestrator stopped")
    
    def schedule_pipeline(self, pipeline_id: str, schedule_config: ScheduleConfig, 
                         priority: int = 5, dependencies: List[str] = None,
                         resource_requirements: Dict[ResourceType, float] = None) -> str:
        """
        Schedule a pipeline for execution
        
        Args:
            pipeline_id: ID of the pipeline to execute
            schedule_config: Schedule configuration
            priority: Execution priority (1-10, higher is more priority)
            dependencies: List of pipeline IDs that must complete first
            resource_requirements: Required resources for execution
            
        Returns:
            Execution ID
        """
        execution_id = str(uuid.uuid4())
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            start_time=datetime.utcnow(),
            priority=priority,
            dependencies=set(dependencies or []),
            resource_requirements=resource_requirements or {},
            metadata={'schedule_config': schedule_config.__dict__}
        )
        
        # Add to dependency graph
        if dependencies:
            self.dependency_graph[execution_id] = set(dependencies)
        
        # Queue for execution
        with self._lock:
            self.execution_queue.put(context)
        
        self.logger.info(f"Scheduled pipeline {pipeline_id} with execution ID {execution_id}")
        return execution_id
    
    def execute_pipeline_now(self, pipeline_id: str, priority: int = 10,
                           resource_requirements: Dict[ResourceType, float] = None) -> str:
        """
        Execute a pipeline immediately with high priority
        
        Args:
            pipeline_id: ID of the pipeline to execute
            priority: Execution priority (default: 10 for immediate)
            resource_requirements: Required resources for execution
            
        Returns:
            Execution ID
        """
        schedule_config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
        return self.schedule_pipeline(
            pipeline_id=pipeline_id,
            schedule_config=schedule_config,
            priority=priority,
            resource_requirements=resource_requirements
        )
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a pending or running execution"""
        with self._lock:
            # Check if in queue
            temp_queue = []
            found = False
            while not self.execution_queue.empty():
                context = self.execution_queue.get()
                if context.execution_id == execution_id:
                    context.status = ExecutionStatus.CANCELLED
                    found = True
                else:
                    temp_queue.append(context)
            
            # Restore queue
            for context in temp_queue:
                self.execution_queue.put(context)
            
            # Check if running
            if execution_id in self.running_executions:
                self.running_executions[execution_id].status = ExecutionStatus.CANCELLED
                found = True
        
        if found:
            self.logger.info(f"Cancelled execution {execution_id}")
        return found
    
    def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution"""
        with self._lock:
            if execution_id in self.running_executions:
                self.running_executions[execution_id].status = ExecutionStatus.PAUSED
                self.logger.info(f"Paused execution {execution_id}")
                return True
        return False
    
    def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution"""
        with self._lock:
            if execution_id in self.running_executions:
                context = self.running_executions[execution_id]
                if context.status == ExecutionStatus.PAUSED:
                    context.status = ExecutionStatus.RUNNING
                    self.logger.info(f"Resumed execution {execution_id}")
                    return True
        return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an execution"""
        with self._lock:
            # Check running executions
            if execution_id in self.running_executions:
                context = self.running_executions[execution_id]
                return {
                    'execution_id': execution_id,
                    'pipeline_id': context.pipeline_id,
                    'status': context.status.value,
                    'start_time': context.start_time.isoformat(),
                    'retry_count': context.retry_count,
                    'priority': context.priority,
                    'resource_requirements': {k.value: v for k, v in context.resource_requirements.items()}
                }
            
            # Check completed executions
            if execution_id in self.completed_executions:
                context = self.completed_executions[execution_id]
                return {
                    'execution_id': execution_id,
                    'pipeline_id': context.pipeline_id,
                    'status': context.status.value,
                    'start_time': context.start_time.isoformat(),
                    'retry_count': context.retry_count,
                    'priority': context.priority,
                    'resource_requirements': {k.value: v for k, v in context.resource_requirements.items()}
                }
        
        return None
    
    def get_pipeline_dependencies(self, pipeline_id: str) -> List[str]:
        """Get dependency chain for a pipeline"""
        db = next(get_db())
        try:
            pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
            if not pipeline:
                return []
            
            # Build dependency graph from pipeline connections
            dependencies = []
            nodes = {node.id: node for node in pipeline.nodes}
            
            # Find source nodes (no incoming connections)
            source_nodes = set(nodes.keys())
            for conn in pipeline.connections:
                source_nodes.discard(conn.target_node_id)
            
            # Topological sort to determine execution order
            visited = set()
            temp_visited = set()
            result = []
            
            def dfs(node_id):
                if node_id in temp_visited:
                    raise ValueError(f"Circular dependency detected involving node {node_id}")
                if node_id in visited:
                    return
                
                temp_visited.add(node_id)
                
                # Visit dependencies first
                for conn in pipeline.connections:
                    if conn.source_node_id == node_id:
                        dfs(conn.target_node_id)
                
                temp_visited.remove(node_id)
                visited.add(node_id)
                result.append(node_id)
            
            for source_node in source_nodes:
                if source_node not in visited:
                    dfs(source_node)
            
            return result
        finally:
            db.close()
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        return {
            'cpu_utilization': (self.resource_manager.allocated_cpu / self.resource_manager.total_cpu) * 100,
            'memory_utilization': (self.resource_manager.allocated_memory / self.resource_manager.total_memory) * 100,
            'storage_utilization': (self.resource_manager.allocated_storage / self.resource_manager.total_storage) * 100,
            'network_utilization': (self.resource_manager.allocated_network / self.resource_manager.total_network) * 100
        }
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        with self._lock:
            utilization = self.get_resource_utilization()
            return {
                **self.metrics,
                'resource_utilization': utilization,
                'queue_size': self.execution_queue.qsize(),
                'running_executions': len(self.running_executions),
                'completed_executions': len(self.completed_executions)
            }
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                self._process_execution_queue()
                self._check_running_executions()
                self._cleanup_completed_executions()
                time.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _process_execution_queue(self):
        """Process pending executions in the queue"""
        with self._lock:
            if (self.execution_queue.empty() or 
                len(self.running_executions) >= self.max_concurrent_executions):
                return
            
            # Get next execution
            context = self.execution_queue.get()
            
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(context):
                # Put back in queue
                self.execution_queue.put(context)
                return
            
            # Check resource availability
            if not self.resource_manager.can_allocate(context.resource_requirements):
                # Put back in queue
                self.execution_queue.put(context)
                return
            
            # Allocate resources and start execution
            self.resource_manager.allocate(context.resource_requirements)
            context.status = ExecutionStatus.RUNNING
            self.running_executions[context.execution_id] = context
            
            # Submit to executor
            future = self.executor.submit(self._execute_pipeline, context)
            context.metadata['future'] = future
            
            self.logger.info(f"Started execution {context.execution_id} for pipeline {context.pipeline_id}")
    
    def _dependencies_satisfied(self, context: ExecutionContext) -> bool:
        """Check if all dependencies are satisfied"""
        for dep_id in context.dependencies:
            if dep_id not in self.completed_executions:
                return False
            if self.completed_executions[dep_id].status != ExecutionStatus.COMPLETED:
                return False
        return True
    
    def _check_running_executions(self):
        """Check status of running executions"""
        with self._lock:
            completed_executions = []
            
            for execution_id, context in self.running_executions.items():
                future = context.metadata.get('future')
                if future and future.done():
                    completed_executions.append(execution_id)
            
            # Move completed executions
            for execution_id in completed_executions:
                context = self.running_executions.pop(execution_id)
                future = context.metadata.get('future')
                
                try:
                    result = future.result()
                    if result.get('success', False):
                        context.status = ExecutionStatus.COMPLETED
                        self.metrics['successful_executions'] += 1
                    else:
                        context.status = ExecutionStatus.FAILED
                        self.metrics['failed_executions'] += 1
                        
                        # Handle retry logic
                        if context.retry_count < context.max_retries:
                            self._schedule_retry(context)
                            continue
                
                except Exception as e:
                    context.status = ExecutionStatus.FAILED
                    self.metrics['failed_executions'] += 1
                    self.logger.error(f"Execution {execution_id} failed: {e}")
                    
                    # Handle retry logic
                    if context.retry_count < context.max_retries:
                        self._schedule_retry(context)
                        continue
                
                # Deallocate resources
                self.resource_manager.deallocate(context.resource_requirements)
                
                # Move to completed
                self.completed_executions[execution_id] = context
                self.metrics['total_executions'] += 1
    
    def _schedule_retry(self, context: ExecutionContext):
        """Schedule a retry for failed execution"""
        context.retry_count += 1
        context.status = ExecutionStatus.RETRYING
        
        # Exponential backoff
        delay = min(300, 2 ** context.retry_count)  # Max 5 minutes
        retry_time = datetime.utcnow() + timedelta(seconds=delay)
        context.start_time = retry_time
        
        # Put back in queue
        self.execution_queue.put(context)
        self.metrics['retry_executions'] += 1
        
        self.logger.info(f"Scheduled retry {context.retry_count} for execution {context.execution_id} in {delay} seconds")
    
    def _execute_pipeline(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute a pipeline (placeholder implementation)
        In a real implementation, this would integrate with the transformation engine
        """
        try:
            self.logger.info(f"Executing pipeline {context.pipeline_id}")
            
            # Simulate pipeline execution
            execution_time = 5  # Simulate 5 second execution
            time.sleep(execution_time)
            
            # Update database
            db = next(get_db())
            try:
                execution = PipelineExecution(
                    id=context.execution_id,
                    pipeline_id=context.pipeline_id,
                    status="completed",
                    start_time=context.start_time,
                    end_time=datetime.utcnow(),
                    records_processed=1000,  # Simulated
                    execution_metrics={
                        'execution_time_seconds': execution_time,
                        'retry_count': context.retry_count,
                        'resource_usage': context.resource_requirements
                    }
                )
                db.add(execution)
                db.commit()
            finally:
                db.close()
            
            return {
                'success': True,
                'execution_time': execution_time,
                'records_processed': 1000
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cleanup_completed_executions(self):
        """Clean up old completed executions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self._lock:
            to_remove = []
            for execution_id, context in self.completed_executions.items():
                if context.start_time < cutoff_time:
                    to_remove.append(execution_id)
            
            for execution_id in to_remove:
                del self.completed_executions[execution_id]


# Global orchestrator instance
_orchestrator_instance = None


def get_orchestrator() -> PipelineOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = PipelineOrchestrator()
        _orchestrator_instance.start()
    return _orchestrator_instance


def shutdown_orchestrator():
    """Shutdown the global orchestrator"""
    global _orchestrator_instance
    if _orchestrator_instance:
        _orchestrator_instance.stop()
        _orchestrator_instance = None