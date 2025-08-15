"""
Distributed Data Processing Engine for AI Data Readiness Platform

This module provides scalable data processing capabilities with auto-scaling,
distributed transformation, and resource optimization for large-scale AI data preparation.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from threading import Lock
import psutil
import queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a data processing task"""
    task_id: str
    data_chunk: pd.DataFrame
    transformation_func: Callable
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ResourceMetrics:
    """System resource metrics for auto-scaling decisions"""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    active_tasks: int
    queue_size: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessingConfig:
    """Configuration for distributed processing"""
    max_workers: int = cpu_count()
    min_workers: int = 2
    chunk_size: int = 10000
    memory_threshold: float = 0.8
    cpu_threshold: float = 0.9
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.3
    monitoring_interval: float = 5.0
    task_timeout: float = 300.0


class ResourceMonitor:
    """Monitors system resources for auto-scaling decisions"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.metrics_history: List[ResourceMetrics] = []
        self.lock = Lock()
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 100 metrics
                    if len(self.metrics_history) > 100:
                        self.metrics_history.pop(0)
                
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_io=disk_io.read_bytes + disk_io.write_bytes if disk_io else 0,
            network_io=network_io.bytes_sent + network_io.bytes_recv if network_io else 0,
            active_tasks=0,  # Will be updated by processor
            queue_size=0     # Will be updated by processor
        )
    
    def get_latest_metrics(self) -> Optional[ResourceMetrics]:
        """Get the latest resource metrics"""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def should_scale_up(self, active_tasks: int, queue_size: int) -> bool:
        """Determine if we should scale up workers"""
        metrics = self.get_latest_metrics()
        if not metrics:
            return False
        
        # Update metrics with current task info
        metrics.active_tasks = active_tasks
        metrics.queue_size = queue_size
        
        # Scale up if CPU/memory usage is high and we have queued tasks
        high_resource_usage = (
            metrics.cpu_usage > self.config.scale_up_threshold * 100 or
            metrics.memory_usage > self.config.scale_up_threshold * 100
        )
        
        return high_resource_usage and queue_size > 0
    
    def should_scale_down(self, active_tasks: int, queue_size: int) -> bool:
        """Determine if we should scale down workers"""
        metrics = self.get_latest_metrics()
        if not metrics:
            return False
        
        # Update metrics with current task info
        metrics.active_tasks = active_tasks
        metrics.queue_size = queue_size
        
        # Scale down if resource usage is low and no queued tasks
        low_resource_usage = (
            metrics.cpu_usage < self.config.scale_down_threshold * 100 and
            metrics.memory_usage < self.config.scale_down_threshold * 100
        )
        
        return low_resource_usage and queue_size == 0 and active_tasks < self.config.min_workers


class LoadBalancer:
    """Intelligent load balancer for distributing tasks across workers"""
    
    def __init__(self):
        self.worker_loads: Dict[str, float] = {}
        self.task_history: Dict[str, List[float]] = {}
        self.worker_capabilities: Dict[str, Dict[str, float]] = {}
        self.lock = Lock()
        self.balancing_strategy = 'weighted_round_robin'  # Options: round_robin, least_loaded, weighted_round_robin, performance_based
    
    def assign_task(self, task: ProcessingTask, available_workers: List[str]) -> str:
        """Assign task to the best available worker using intelligent algorithms"""
        if not available_workers:
            return None
        
        if len(available_workers) == 1:
            return available_workers[0]
        
        with self.lock:
            if self.balancing_strategy == 'round_robin':
                return self._round_robin_assignment(available_workers)
            elif self.balancing_strategy == 'least_loaded':
                return self._least_loaded_assignment(available_workers)
            elif self.balancing_strategy == 'weighted_round_robin':
                return self._weighted_round_robin_assignment(available_workers, task)
            elif self.balancing_strategy == 'performance_based':
                return self._performance_based_assignment(available_workers, task)
            else:
                return self._least_loaded_assignment(available_workers)
    
    def _round_robin_assignment(self, available_workers: List[str]) -> str:
        """Simple round-robin assignment"""
        # Use worker with lowest load as a simple round-robin approximation
        worker_loads = [(worker, self.worker_loads.get(worker, 0.0)) for worker in available_workers]
        worker_loads.sort(key=lambda x: x[1])
        return worker_loads[0][0]
    
    def _least_loaded_assignment(self, available_workers: List[str]) -> str:
        """Assign to worker with least current load"""
        worker_loads = [(worker, self.worker_loads.get(worker, 0.0)) for worker in available_workers]
        worker_loads.sort(key=lambda x: x[1])
        selected_worker = worker_loads[0][0]
        
        # Update worker load
        self.worker_loads[selected_worker] = self.worker_loads.get(selected_worker, 0.0) + 1.0
        return selected_worker
    
    def _weighted_round_robin_assignment(self, available_workers: List[str], task: ProcessingTask) -> str:
        """Weighted round-robin based on worker performance history"""
        worker_scores = []
        
        for worker in available_workers:
            current_load = self.worker_loads.get(worker, 0.0)
            
            # Calculate performance weight (inverse of average execution time)
            history = self.task_history.get(worker, [])
            if history:
                avg_time = sum(history) / len(history)
                performance_weight = 1.0 / max(0.1, avg_time)  # Avoid division by zero
            else:
                performance_weight = 1.0  # Default weight for new workers
            
            # Combine load and performance (lower is better)
            score = current_load / max(0.1, performance_weight)
            worker_scores.append((worker, score))
        
        # Select worker with best score
        worker_scores.sort(key=lambda x: x[1])
        selected_worker = worker_scores[0][0]
        
        # Update worker load
        self.worker_loads[selected_worker] = self.worker_loads.get(selected_worker, 0.0) + 1.0
        return selected_worker
    
    def _performance_based_assignment(self, available_workers: List[str], task: ProcessingTask) -> str:
        """Advanced performance-based assignment considering task characteristics"""
        worker_scores = []
        
        for worker in available_workers:
            current_load = self.worker_loads.get(worker, 0.0)
            capabilities = self.worker_capabilities.get(worker, {})
            
            # Base score from current load (lower is better)
            load_score = current_load
            
            # Performance score from execution history
            history = self.task_history.get(worker, [])
            if history:
                avg_time = sum(history) / len(history)
                performance_score = avg_time
            else:
                performance_score = 1.0  # Default for new workers
            
            # Task complexity factor (could be enhanced based on task metadata)
            complexity_factor = task.metadata.get('complexity', 1.0)
            
            # Combined score (lower is better)
            total_score = (load_score * 0.4) + (performance_score * 0.6) * complexity_factor
            worker_scores.append((worker, total_score))
        
        # Select worker with best score
        worker_scores.sort(key=lambda x: x[1])
        selected_worker = worker_scores[0][0]
        
        # Update worker load
        self.worker_loads[selected_worker] = self.worker_loads.get(selected_worker, 0.0) + 1.0
        return selected_worker
    
    def task_completed(self, worker_id: str, execution_time: float):
        """Update worker load and performance metrics after task completion"""
        with self.lock:
            # Update load
            if worker_id in self.worker_loads:
                self.worker_loads[worker_id] = max(0.0, self.worker_loads[worker_id] - 1.0)
            
            # Track execution time for performance optimization
            if worker_id not in self.task_history:
                self.task_history[worker_id] = []
            self.task_history[worker_id].append(execution_time)
            
            # Keep only last 100 execution times for better statistics
            if len(self.task_history[worker_id]) > 100:
                self.task_history[worker_id].pop(0)
            
            # Update worker capabilities
            self._update_worker_capabilities(worker_id, execution_time)
    
    def _update_worker_capabilities(self, worker_id: str, execution_time: float):
        """Update worker capability metrics"""
        if worker_id not in self.worker_capabilities:
            self.worker_capabilities[worker_id] = {
                'avg_execution_time': execution_time,
                'min_execution_time': execution_time,
                'max_execution_time': execution_time,
                'task_count': 1,
                'reliability_score': 1.0
            }
        else:
            caps = self.worker_capabilities[worker_id]
            caps['task_count'] += 1
            caps['avg_execution_time'] = (
                (caps['avg_execution_time'] * (caps['task_count'] - 1) + execution_time) / 
                caps['task_count']
            )
            caps['min_execution_time'] = min(caps['min_execution_time'], execution_time)
            caps['max_execution_time'] = max(caps['max_execution_time'], execution_time)
            
            # Update reliability score based on consistency
            if caps['task_count'] > 5:
                time_variance = caps['max_execution_time'] - caps['min_execution_time']
                caps['reliability_score'] = max(0.1, 1.0 - (time_variance / caps['avg_execution_time']))
    
    def set_balancing_strategy(self, strategy: str):
        """Set the load balancing strategy"""
        valid_strategies = ['round_robin', 'least_loaded', 'weighted_round_robin', 'performance_based']
        if strategy in valid_strategies:
            self.balancing_strategy = strategy
            logger.info(f"Load balancing strategy set to: {strategy}")
        else:
            logger.warning(f"Invalid strategy: {strategy}. Valid options: {valid_strategies}")
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workers"""
        with self.lock:
            stats = {}
            for worker_id in set(list(self.worker_loads.keys()) + list(self.task_history.keys())):
                stats[worker_id] = {
                    'current_load': self.worker_loads.get(worker_id, 0.0),
                    'task_history_count': len(self.task_history.get(worker_id, [])),
                    'capabilities': self.worker_capabilities.get(worker_id, {}),
                    'recent_avg_time': (
                        sum(self.task_history[worker_id][-10:]) / len(self.task_history[worker_id][-10:])
                        if worker_id in self.task_history and self.task_history[worker_id]
                        else 0.0
                    )
                }
            return stats


class DistributedDataProcessor:
    """
    Scalable data processor with auto-scaling capabilities and distributed transformation engine.
    
    Features:
    - Auto-scaling based on resource utilization
    - Intelligent load balancing
    - Resource optimization
    - Fault tolerance and error handling
    - Performance monitoring
    - Dynamic worker pool management
    - Advanced scheduling algorithms
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.resource_monitor = ResourceMonitor(self.config)
        self.load_balancer = LoadBalancer()
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        
        # Enhanced worker management
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        self.current_workers = self.config.min_workers
        self.worker_futures: Dict[str, Future] = {}
        self.worker_performance: Dict[str, List[float]] = {}
        
        # Synchronization
        self.lock = Lock()
        self._shutdown = False
        self._processing_thread = None
        
        # Enhanced performance metrics
        self.processing_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'throughput_per_second': 0.0,
            'peak_memory_usage': 0.0,
            'peak_cpu_usage': 0.0,
            'scaling_events': 0,
            'error_rate': 0.0
        }
        
        # Resource optimization
        self.optimization_enabled = True
        self.adaptive_chunk_size = self.config.chunk_size
        self.last_optimization = time.time()
        
        logger.info(f"DistributedDataProcessor initialized with {self.current_workers} workers")
    
    def start(self):
        """Start the distributed processor"""
        self.resource_monitor.start_monitoring()
        self._start_auto_scaler()
        self._start_task_processor()
        self._start_optimization_loop()
        logger.info("DistributedDataProcessor started")
    
    def stop(self):
        """Stop the distributed processor"""
        self._shutdown = True
        self.resource_monitor.stop_monitoring()
        
        # Wait for processing thread to finish
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        
        # Shutdown executors gracefully
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("DistributedDataProcessor stopped")
    
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a processing task"""
        with self.lock:
            self.task_queue.put((task.priority, task))
            self.processing_stats['total_tasks'] += 1
        
        logger.debug(f"Task {task.task_id} submitted with priority {task.priority}")
        return task.task_id
    
    def process_dataframe(self, 
                         df: pd.DataFrame, 
                         transformation_func: Callable,
                         chunk_size: Optional[int] = None,
                         priority: int = 1) -> List[str]:
        """
        Process a large DataFrame by splitting it into chunks and distributing across workers
        
        Args:
            df: DataFrame to process
            transformation_func: Function to apply to each chunk
            chunk_size: Size of each chunk (defaults to config chunk_size)
            priority: Task priority (lower numbers = higher priority)
        
        Returns:
            List of task IDs for tracking
        """
        chunk_size = chunk_size or self.config.chunk_size
        task_ids = []
        
        # Split DataFrame into chunks
        num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            
            task = ProcessingTask(
                task_id=f"chunk_{i}_{int(time.time() * 1000)}",
                data_chunk=chunk,
                transformation_func=transformation_func,
                priority=priority,
                metadata={
                    'chunk_index': i,
                    'total_chunks': num_chunks,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
            )
            
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        
        logger.info(f"DataFrame split into {num_chunks} chunks for processing")
        return task_ids
    
    def get_results(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get results for a list of task IDs
        
        Args:
            task_ids: List of task IDs to get results for
            timeout: Maximum time to wait for results
        
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        start_time = time.time()
        
        while len(results) < len(task_ids):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            with self.lock:
                for task_id in task_ids:
                    if task_id not in results:
                        if task_id in self.completed_tasks:
                            results[task_id] = self.completed_tasks[task_id]
                        elif task_id in self.failed_tasks:
                            results[task_id] = {'error': self.failed_tasks[task_id]}
            
            if len(results) < len(task_ids):
                time.sleep(0.1)  # Short sleep to avoid busy waiting
        
        return results
    
    def combine_chunk_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Combine results from chunk processing back into a single DataFrame
        
        Args:
            results: Dictionary of task results
        
        Returns:
            Combined DataFrame
        """
        # Sort results by chunk index
        sorted_results = []
        for task_id, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                logger.error(f"Task {task_id} failed: {result['error']}")
                continue
            
            if isinstance(result, pd.DataFrame):
                # Extract chunk index from task metadata if available
                chunk_index = 0  # Default fallback
                sorted_results.append((chunk_index, result))
        
        # Sort by chunk index and combine
        sorted_results.sort(key=lambda x: x[0])
        combined_df = pd.concat([result for _, result in sorted_results], ignore_index=True)
        
        logger.info(f"Combined {len(sorted_results)} chunks into DataFrame with {len(combined_df)} rows")
        return combined_df
    
    def _start_auto_scaler(self):
        """Start the auto-scaling thread"""
        def auto_scale_loop():
            while not self._shutdown:
                try:
                    self._auto_scale()
                    time.sleep(self.config.monitoring_interval)
                except Exception as e:
                    logger.error(f"Error in auto-scaling: {e}")
        
        auto_scale_thread = threading.Thread(target=auto_scale_loop, daemon=True)
        auto_scale_thread.start()
    
    def _auto_scale(self):
        """Auto-scale workers based on resource utilization and queue size"""
        with self.lock:
            queue_size = self.task_queue.qsize()
            active_tasks = len(self.active_tasks)
        
        if self.resource_monitor.should_scale_up(active_tasks, queue_size):
            if self.current_workers < self.config.max_workers:
                self.current_workers += 1
                logger.info(f"Scaled up to {self.current_workers} workers")
        
        elif self.resource_monitor.should_scale_down(active_tasks, queue_size):
            if self.current_workers > self.config.min_workers:
                self.current_workers -= 1
                logger.info(f"Scaled down to {self.current_workers} workers")
    
    def _start_task_processor(self):
        """Start the main task processing thread"""
        def process_tasks():
            while not self._shutdown:
                try:
                    self._process_pending_tasks()
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                except Exception as e:
                    logger.error(f"Error in task processing: {e}")
        
        self._processing_thread = threading.Thread(target=process_tasks, daemon=True)
        self._processing_thread.start()
    
    def _process_pending_tasks(self):
        """Process pending tasks from the queue"""
        try:
            # Get task from queue with timeout
            priority, task = self.task_queue.get(timeout=1.0)
            
            # Check if we have available workers
            if len(self.active_tasks) >= self.current_workers:
                # Put task back in queue if no workers available
                self.task_queue.put((priority, task))
                return
            
            # Submit task to executor
            future = self._submit_task_to_executor(task)
            
            with self.lock:
                self.active_tasks[task.task_id] = task
                self.worker_futures[task.task_id] = future
            
            # Set up completion callback
            future.add_done_callback(lambda f: self._task_completed(task.task_id, f))
            
        except queue.Empty:
            pass  # No tasks in queue
        except Exception as e:
            logger.error(f"Error processing task: {e}")
    
    def _submit_task_to_executor(self, task: ProcessingTask) -> Future:
        """Submit task to appropriate executor based on task characteristics"""
        # Determine if task should use thread or process executor
        # For now, use thread executor for most tasks
        return self.thread_executor.submit(self._execute_task, task)
    
    def _execute_task(self, task: ProcessingTask) -> Any:
        """Execute a single task"""
        start_time = time.time()
        
        try:
            # Apply transformation function to data chunk
            result = task.transformation_func(task.data_chunk)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            with self.lock:
                self.processing_stats['completed_tasks'] += 1
                self.processing_stats['total_processing_time'] += execution_time
                self.processing_stats['average_processing_time'] = (
                    self.processing_stats['total_processing_time'] / 
                    self.processing_stats['completed_tasks']
                )
            
            logger.debug(f"Task {task.task_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed after {execution_time:.2f}s: {e}")
            
            with self.lock:
                self.processing_stats['failed_tasks'] += 1
                self.processing_stats['error_rate'] = (
                    self.processing_stats['failed_tasks'] / 
                    max(1, self.processing_stats['total_tasks'])
                )
            
            raise e
    
    def _task_completed(self, task_id: str, future: Future):
        """Handle task completion"""
        try:
            result = future.result()
            
            with self.lock:
                self.completed_tasks[task_id] = result
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                if task_id in self.worker_futures:
                    del self.worker_futures[task_id]
            
        except Exception as e:
            with self.lock:
                self.failed_tasks[task_id] = e
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                if task_id in self.worker_futures:
                    del self.worker_futures[task_id]
    
    def _start_optimization_loop(self):
        """Start the resource optimization loop"""
        def optimize_resources():
            while not self._shutdown:
                try:
                    if self.optimization_enabled:
                        self._optimize_performance()
                    time.sleep(30.0)  # Optimize every 30 seconds
                except Exception as e:
                    logger.error(f"Error in resource optimization: {e}")
        
        optimization_thread = threading.Thread(target=optimize_resources, daemon=True)
        optimization_thread.start()
    
    def _optimize_performance(self):
        """Optimize performance based on current metrics"""
        current_time = time.time()
        
        # Only optimize if enough time has passed
        if current_time - self.last_optimization < 30.0:
            return
        
        with self.lock:
            stats = self.processing_stats.copy()
            queue_size = self.task_queue.qsize()
        
        # Adaptive chunk size optimization
        if stats['completed_tasks'] > 10:  # Need some data to optimize
            avg_time = stats['average_processing_time']
            
            if avg_time > 5.0 and self.adaptive_chunk_size > 1000:
                # Tasks taking too long, reduce chunk size
                self.adaptive_chunk_size = max(1000, int(self.adaptive_chunk_size * 0.8))
                logger.info(f"Reduced chunk size to {self.adaptive_chunk_size}")
                
            elif avg_time < 1.0 and self.adaptive_chunk_size < 50000:
                # Tasks completing quickly, increase chunk size
                self.adaptive_chunk_size = min(50000, int(self.adaptive_chunk_size * 1.2))
                logger.info(f"Increased chunk size to {self.adaptive_chunk_size}")
        
        self.last_optimization = current_time
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        with self.lock:
            stats = self.processing_stats.copy()
            
            # Calculate throughput
            if stats['total_processing_time'] > 0:
                stats['throughput_per_second'] = stats['completed_tasks'] / stats['total_processing_time']
            
            stats.update({
                'current_workers': self.current_workers,
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'adaptive_chunk_size': self.adaptive_chunk_size,
                'resource_metrics': self.resource_monitor.get_latest_metrics()
            })
        return stats
    
    def optimize_chunk_size(self, data_size: int, complexity_factor: float = 1.0) -> int:
        """
        Optimize chunk size based on data characteristics and system performance
        
        Args:
            data_size: Total size of data to process
            complexity_factor: Factor indicating processing complexity (1.0 = normal)
        
        Returns:
            Optimized chunk size
        """
        base_chunk_size = self.adaptive_chunk_size
        
        # Adjust based on data size
        if data_size < 10000:
            # Small datasets - use smaller chunks
            chunk_size = min(base_chunk_size, data_size // 4)
        elif data_size > 1000000:
            # Large datasets - use larger chunks for efficiency
            chunk_size = min(base_chunk_size * 2, data_size // (self.current_workers * 4))
        else:
            chunk_size = base_chunk_size
        
        # Adjust based on complexity
        chunk_size = int(chunk_size / complexity_factor)
        
        # Ensure minimum chunk size
        return max(100, chunk_size)
    
    def enable_optimization(self, enabled: bool = True):
        """Enable or disable automatic performance optimization"""
        self.optimization_enabled = enabled
        logger.info(f"Performance optimization {'enabled' if enabled else 'disabled'}")
    
    def get_worker_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for individual workers"""
        with self.lock:
            performance = {}
            for worker_id, times in self.worker_performance.items():
                if times:
                    performance[worker_id] = {
                        'average_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'task_count': len(times)
                    }
            return performance


# Example transformation functions for testing
def sample_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Sample transformation function for testing"""
    # Simulate some processing time
    time.sleep(0.1)
    
    # Apply some transformations
    result = df.copy()
    if 'numeric_column' in result.columns:
        result['numeric_column'] = result['numeric_column'] * 2
    
    return result


def heavy_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Heavy transformation function for testing scaling"""
    # Simulate heavy processing
    time.sleep(1.0)
    
    result = df.copy()
    # Simulate complex calculations
    for col in result.select_dtypes(include=[np.number]).columns:
        result[f'{col}_squared'] = result[col] ** 2
        result[f'{col}_log'] = np.log1p(np.abs(result[col]))
    
    return result