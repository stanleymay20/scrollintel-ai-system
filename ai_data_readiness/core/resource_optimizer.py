"""Resource utilization tracking and optimization for AI Data Readiness Platform."""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .platform_monitor import get_platform_monitor
from .config import get_settings


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""
    timestamp: datetime
    cpu_cores_used: float
    memory_mb_used: float
    disk_io_mb_per_sec: float
    network_io_mb_per_sec: float
    active_threads: int
    active_processes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_cores_used': self.cpu_cores_used,
            'memory_mb_used': self.memory_mb_used,
            'disk_io_mb_per_sec': self.disk_io_mb_per_sec,
            'network_io_mb_per_sec': self.network_io_mb_per_sec,
            'active_threads': self.active_threads,
            'active_processes': self.active_processes
        }


@dataclass
class OptimizationRecommendation:
    """Resource optimization recommendation."""
    category: str  # cpu, memory, disk, network
    priority: str  # high, medium, low
    title: str
    description: str
    action: str
    estimated_improvement: str
    implementation_effort: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'action': self.action,
            'estimated_improvement': self.estimated_improvement,
            'implementation_effort': self.implementation_effort,
            'created_at': self.created_at.isoformat()
        }


class ResourceOptimizer:
    """Resource utilization tracking and optimization system."""
    
    def __init__(self):
        self.config = get_settings()
        self.monitor = get_platform_monitor()
        self.logger = logging.getLogger(__name__)
        
        # Resource tracking
        self.resource_history: deque = deque(maxlen=1000)
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Resource pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Optimization settings
        self.optimization_enabled = True
        self.auto_scaling_enabled = True
        self.memory_cleanup_threshold = 0.85  # 85% memory usage
        self.cpu_scaling_threshold = 0.80     # 80% CPU usage
        
        # Tracking variables
        self.last_optimization_check = datetime.utcnow()
        self.last_memory_cleanup = datetime.utcnow()
        self.optimization_thread: Optional[threading.Thread] = None
        self.optimization_active = False
        
        # Performance baselines
        self.performance_baselines = {
            'cpu_efficiency': 0.0,
            'memory_efficiency': 0.0,
            'throughput_ops_per_sec': 0.0,
            'avg_response_time_ms': 0.0
        }
        
        self._initialize_resource_pools()
    
    def _initialize_resource_pools(self):
        """Initialize thread and process pools based on system resources."""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Calculate optimal pool sizes
        max_workers = min(self.config.processing.max_workers, cpu_count)
        
        # Adjust based on available memory
        if memory_gb < 8:
            max_workers = max(2, max_workers // 2)
        elif memory_gb > 32:
            max_workers = min(max_workers * 2, cpu_count * 2)
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ai_data_readiness"
        )
        
        # Process pool for CPU-intensive tasks
        process_workers = max(2, cpu_count // 2)
        self.process_pool = ProcessPoolExecutor(max_workers=process_workers)
        
        self.logger.info(f"Initialized resource pools: {max_workers} threads, {process_workers} processes")
    
    def start_optimization(self, interval_seconds: int = 300):
        """Start continuous resource optimization."""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.optimization_thread.start()
        self.logger.info("Resource optimization started")
    
    def stop_optimization(self):
        """Stop resource optimization."""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        
        # Cleanup resource pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self.logger.info("Resource optimization stopped")
    
    def _optimization_loop(self, interval_seconds: int):
        """Main optimization loop."""
        while self.optimization_active:
            try:
                self.track_resource_usage()
                self.optimize_resources()
                self.generate_recommendations()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(interval_seconds)
    
    def track_resource_usage(self) -> ResourceUsage:
        """Track current resource usage."""
        try:
            # Get current process info
            process = psutil.Process()
            
            # CPU usage (cores)
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_cores_used = (cpu_percent / 100) * psutil.cpu_count()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb_used = memory_info.rss / (1024 * 1024)
            
            # I/O metrics
            io_counters = psutil.disk_io_counters()
            net_counters = psutil.net_io_counters()
            
            # Calculate rates (simplified - would need previous values for accurate rates)
            disk_io_mb_per_sec = 0.0  # Placeholder
            network_io_mb_per_sec = 0.0  # Placeholder
            
            # Thread and process counts
            active_threads = threading.active_count()
            active_processes = len(psutil.pids())
            
            usage = ResourceUsage(
                timestamp=datetime.utcnow(),
                cpu_cores_used=cpu_cores_used,
                memory_mb_used=memory_mb_used,
                disk_io_mb_per_sec=disk_io_mb_per_sec,
                network_io_mb_per_sec=network_io_mb_per_sec,
                active_threads=active_threads,
                active_processes=active_processes
            )
            
            self.resource_history.append(usage)
            return usage
            
        except Exception as e:
            self.logger.error(f"Error tracking resource usage: {e}")
            raise
    
    def optimize_resources(self):
        """Perform resource optimization."""
        if not self.optimization_enabled:
            return
        
        current_usage = self.get_current_resource_usage()
        if not current_usage:
            return
        
        # Memory optimization
        self._optimize_memory(current_usage)
        
        # CPU optimization
        self._optimize_cpu(current_usage)
        
        # Thread pool optimization
        self._optimize_thread_pools(current_usage)
        
        self.last_optimization_check = datetime.utcnow()
    
    def _optimize_memory(self, usage: ResourceUsage):
        """Optimize memory usage."""
        memory_percent = psutil.virtual_memory().percent / 100
        
        if memory_percent > self.memory_cleanup_threshold:
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"Memory cleanup: collected {collected} objects")
            
            # Update last cleanup time
            self.last_memory_cleanup = datetime.utcnow()
            
            # Add recommendation for memory optimization
            if memory_percent > 0.90:
                self.optimization_recommendations.append(
                    OptimizationRecommendation(
                        category="memory",
                        priority="high",
                        title="Critical Memory Usage",
                        description=f"Memory usage at {memory_percent*100:.1f}%",
                        action="Consider increasing system memory or optimizing data processing batch sizes",
                        estimated_improvement="20-30% memory reduction",
                        implementation_effort="Medium"
                    )
                )
    
    def _optimize_cpu(self, usage: ResourceUsage):
        """Optimize CPU usage."""
        cpu_percent = psutil.cpu_percent() / 100
        
        if cpu_percent > self.cpu_scaling_threshold and self.auto_scaling_enabled:
            # Suggest scaling recommendations
            self.optimization_recommendations.append(
                OptimizationRecommendation(
                    category="cpu",
                    priority="medium",
                    title="High CPU Usage Detected",
                    description=f"CPU usage at {cpu_percent*100:.1f}%",
                    action="Consider distributing workload or increasing processing capacity",
                    estimated_improvement="15-25% performance improvement",
                    implementation_effort="Low"
                )
            )
    
    def _optimize_thread_pools(self, usage: ResourceUsage):
        """Optimize thread pool configuration."""
        if not self.thread_pool:
            return
        
        # Check if thread pool is under or over-utilized
        active_threads = usage.active_threads
        max_workers = self.thread_pool._max_workers
        
        utilization = active_threads / max_workers
        
        if utilization > 0.90:
            # High utilization - suggest scaling up
            self.optimization_recommendations.append(
                OptimizationRecommendation(
                    category="cpu",
                    priority="medium",
                    title="Thread Pool Saturation",
                    description=f"Thread utilization at {utilization*100:.1f}%",
                    action="Consider increasing thread pool size or implementing queue management",
                    estimated_improvement="10-20% throughput improvement",
                    implementation_effort="Low"
                )
            )
        elif utilization < 0.30:
            # Low utilization - suggest scaling down
            self.optimization_recommendations.append(
                OptimizationRecommendation(
                    category="cpu",
                    priority="low",
                    title="Thread Pool Under-utilization",
                    description=f"Thread utilization at {utilization*100:.1f}%",
                    action="Consider reducing thread pool size to free up resources",
                    estimated_improvement="5-10% resource savings",
                    implementation_effort="Low"
                )
            )
    
    def generate_recommendations(self):
        """Generate optimization recommendations based on resource patterns."""
        if len(self.resource_history) < 10:
            return
        
        # Analyze resource trends
        recent_usage = list(self.resource_history)[-10:]
        
        # CPU trend analysis
        cpu_trend = self._analyze_trend([u.cpu_cores_used for u in recent_usage])
        if cpu_trend > 0.1:  # Increasing CPU usage
            self.optimization_recommendations.append(
                OptimizationRecommendation(
                    category="cpu",
                    priority="medium",
                    title="Increasing CPU Usage Trend",
                    description="CPU usage has been steadily increasing",
                    action="Monitor workload patterns and consider capacity planning",
                    estimated_improvement="Prevent performance degradation",
                    implementation_effort="Low"
                )
            )
        
        # Memory trend analysis
        memory_trend = self._analyze_trend([u.memory_mb_used for u in recent_usage])
        if memory_trend > 50:  # Increasing memory usage by >50MB
            self.optimization_recommendations.append(
                OptimizationRecommendation(
                    category="memory",
                    priority="medium",
                    title="Memory Usage Growth",
                    description="Memory usage is growing over time",
                    action="Check for memory leaks and optimize data structures",
                    estimated_improvement="Prevent memory exhaustion",
                    implementation_effort="Medium"
                )
            )
        
        # Keep only recent recommendations (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.optimization_recommendations = [
            rec for rec in self.optimization_recommendations
            if rec.created_at > cutoff_time
        ]
    
    def _analyze_trend(self, values: List[float]) -> float:
        """Analyze trend in values using simple linear regression."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def get_current_resource_usage(self) -> Optional[ResourceUsage]:
        """Get the most recent resource usage."""
        return self.resource_history[-1] if self.resource_history else None
    
    def get_resource_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate resource efficiency metrics."""
        if len(self.resource_history) < 10:
            return {}
        
        recent_usage = list(self.resource_history)[-10:]
        
        # CPU efficiency (work done per CPU core)
        avg_cpu_usage = sum(u.cpu_cores_used for u in recent_usage) / len(recent_usage)
        cpu_efficiency = min(1.0, avg_cpu_usage / psutil.cpu_count())
        
        # Memory efficiency (data processed per MB of memory)
        avg_memory_usage = sum(u.memory_mb_used for u in recent_usage) / len(recent_usage)
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        memory_efficiency = avg_memory_usage / total_memory_mb
        
        return {
            'cpu_efficiency': cpu_efficiency,
            'memory_efficiency': memory_efficiency,
            'avg_cpu_cores_used': avg_cpu_usage,
            'avg_memory_mb_used': avg_memory_usage,
            'resource_utilization_score': (cpu_efficiency + (1 - memory_efficiency)) / 2
        }
    
    def get_optimization_recommendations(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization recommendations."""
        recommendations = self.optimization_recommendations
        
        if category:
            recommendations = [r for r in recommendations if r.category == category]
        
        return [r.to_dict() for r in recommendations]
    
    def get_resource_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get resource usage history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            usage.to_dict() for usage in self.resource_history
            if usage.timestamp > cutoff_time
        ]
    
    def scale_thread_pool(self, new_size: int):
        """Dynamically scale thread pool size."""
        if not self.thread_pool:
            return
        
        # Shutdown current pool
        self.thread_pool.shutdown(wait=False)
        
        # Create new pool with updated size
        self.thread_pool = ThreadPoolExecutor(
            max_workers=new_size,
            thread_name_prefix="ai_data_readiness"
        )
        
        self.logger.info(f"Thread pool scaled to {new_size} workers")
    
    def get_optimal_batch_size(self, data_size_mb: float, available_memory_mb: float) -> int:
        """Calculate optimal batch size based on available resources."""
        # Conservative approach: use 50% of available memory
        usable_memory_mb = available_memory_mb * 0.5
        
        # Estimate memory per row (rough approximation)
        estimated_memory_per_row_kb = 1.0  # 1KB per row average
        
        # Calculate rows that fit in memory
        rows_in_memory = int((usable_memory_mb * 1024) / estimated_memory_per_row_kb)
        
        # Set reasonable bounds
        min_batch_size = 100
        max_batch_size = 10000
        
        optimal_batch_size = max(min_batch_size, min(rows_in_memory, max_batch_size))
        
        return optimal_batch_size
    
    def cleanup_resources(self):
        """Cleanup unused resources."""
        # Force garbage collection
        collected = gc.collect()
        
        # Clear old recommendations
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.optimization_recommendations = [
            rec for rec in self.optimization_recommendations
            if rec.created_at > cutoff_time
        ]
        
        self.logger.info(f"Resource cleanup completed: {collected} objects collected")


# Global optimizer instance
_optimizer_instance = None


def get_resource_optimizer() -> ResourceOptimizer:
    """Get global resource optimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ResourceOptimizer()
    return _optimizer_instance