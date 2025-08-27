"""
Comprehensive metrics collection for visual generation system.
Integrates with existing ScrollIntel monitoring infrastructure.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import psutil
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Represents a single metric measurement."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""
    description: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for visual generation operations."""
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cancelled_requests: int = 0
    
    # Timing metrics
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    queue_wait_time: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    images_per_minute: float = 0.0
    videos_per_minute: float = 0.0
    
    # Quality metrics
    average_quality_score: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Resource metrics
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    retry_rate: float = 0.0


class MetricsCollector:
    """
    Comprehensive metrics collector for visual generation system.
    Integrates with ScrollIntel monitoring infrastructure.
    """
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_buffer: List[Metric] = []
        self.performance_history: deque = deque(maxlen=1000)
        
        # Metric aggregators
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Request tracking
        self.active_requests: Dict[str, datetime] = {}
        self.completed_requests: List[Dict[str, Any]] = []
        
        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Integration with ScrollIntel monitoring
        self.scrollintel_metrics_endpoint = None
        self.custom_metric_handlers: List[Callable] = []
    
    async def start(self):
        """Start the metrics collector."""
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Visual generation metrics collector started")
    
    async def stop(self):
        """Stop the metrics collector."""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Visual generation metrics collector stopped")
    
    def register_scrollintel_integration(self, metrics_endpoint: str):
        """Register integration with ScrollIntel monitoring."""
        self.scrollintel_metrics_endpoint = metrics_endpoint
        logger.info(f"Registered ScrollIntel metrics integration: {metrics_endpoint}")
    
    def register_custom_handler(self, handler: Callable[[List[Metric]], None]):
        """Register custom metric handler."""
        self.custom_metric_handlers.append(handler)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        key = self._make_metric_key(name, labels)
        self.counters[key] += value
        
        self._add_metric(Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {},
            description=f"Counter for {name}"
        ))
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        key = self._make_metric_key(name, labels)
        self.gauges[key] = value
        
        self._add_metric(Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {},
            description=f"Gauge for {name}"
        ))
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value."""
        key = self._make_metric_key(name, labels)
        self.histograms[key].append(value)
        
        # Keep only recent values (last 1000)
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        self._add_metric(Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {},
            description=f"Histogram for {name}"
        ))
    
    def start_timer(self, name: str, request_id: str = None, labels: Dict[str, str] = None):
        """Start a timer for an operation."""
        timer_key = f"{name}:{request_id or 'default'}"
        self.active_requests[timer_key] = datetime.now()
        
        return timer_key
    
    def end_timer(self, timer_key: str, labels: Dict[str, str] = None):
        """End a timer and record the duration."""
        if timer_key in self.active_requests:
            start_time = self.active_requests.pop(timer_key)
            duration = (datetime.now() - start_time).total_seconds()
            
            name = timer_key.split(':')[0]
            self.timers[name].append(duration)
            
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            
            self._add_metric(Metric(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                labels=labels or {},
                unit="seconds",
                description=f"Timer for {name}"
            ))
            
            return duration
        
        return None
    
    def record_request_start(self, request_id: str, request_type: str, user_id: str = None):
        """Record the start of a generation request."""
        self.increment_counter("visual_generation_requests_total", labels={
            "type": request_type,
            "status": "started"
        })
        
        self.start_timer("request_duration", request_id, labels={
            "type": request_type
        })
    
    def record_request_completion(self, request_id: str, request_type: str, 
                                success: bool, quality_score: float = None,
                                error_type: str = None):
        """Record the completion of a generation request."""
        # End timer
        timer_key = f"request_duration:{request_id}"
        duration = self.end_timer(timer_key, labels={"type": request_type})
        
        # Record completion metrics
        status = "success" if success else "failure"
        self.increment_counter("visual_generation_requests_total", labels={
            "type": request_type,
            "status": status
        })
        
        if success:
            self.increment_counter("visual_generation_successful_requests", labels={
                "type": request_type
            })
            
            if quality_score is not None:
                self.record_histogram("visual_generation_quality_score", quality_score, labels={
                    "type": request_type
                })
        else:
            self.increment_counter("visual_generation_failed_requests", labels={
                "type": request_type,
                "error_type": error_type or "unknown"
            })
        
        # Record request details
        self.completed_requests.append({
            "request_id": request_id,
            "request_type": request_type,
            "success": success,
            "duration": duration,
            "quality_score": quality_score,
            "error_type": error_type,
            "timestamp": datetime.now()
        })
        
        # Keep only recent requests
        if len(self.completed_requests) > 10000:
            self.completed_requests = self.completed_requests[-5000:]
    
    def record_cache_operation(self, operation: str, hit: bool, content_type: str = None):
        """Record cache operation metrics."""
        status = "hit" if hit else "miss"
        self.increment_counter("visual_generation_cache_operations", labels={
            "operation": operation,
            "status": status,
            "content_type": content_type or "unknown"
        })
    
    def record_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """Record worker-specific metrics."""
        labels = {"worker_id": worker_id}
        
        if "gpu_utilization" in metrics:
            self.set_gauge("visual_generation_worker_gpu_utilization", 
                          metrics["gpu_utilization"], labels)
        
        if "memory_usage" in metrics:
            self.set_gauge("visual_generation_worker_memory_usage", 
                          metrics["memory_usage"], labels)
        
        if "cpu_usage" in metrics:
            self.set_gauge("visual_generation_worker_cpu_usage", 
                          metrics["cpu_usage"], labels)
        
        if "active_jobs" in metrics:
            self.set_gauge("visual_generation_worker_active_jobs", 
                          metrics["active_jobs"], labels)
        
        if "queue_length" in metrics:
            self.set_gauge("visual_generation_worker_queue_length", 
                          metrics["queue_length"], labels)
    
    def record_model_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Record model-specific metrics."""
        labels = {"model": model_name}
        
        if "inference_time" in metrics:
            self.record_histogram("visual_generation_model_inference_time", 
                                metrics["inference_time"], labels)
        
        if "memory_usage" in metrics:
            self.set_gauge("visual_generation_model_memory_usage", 
                          metrics["memory_usage"], labels)
        
        if "throughput" in metrics:
            self.set_gauge("visual_generation_model_throughput", 
                          metrics["throughput"], labels)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Filter recent requests
        recent_requests = [
            req for req in self.completed_requests
            if req["timestamp"] > hour_ago
        ]
        
        if not recent_requests:
            return PerformanceMetrics()
        
        # Calculate metrics
        total_requests = len(recent_requests)
        successful_requests = len([r for r in recent_requests if r["success"]])
        failed_requests = total_requests - successful_requests
        
        # Response times
        durations = [r["duration"] for r in recent_requests if r["duration"]]
        avg_response_time = sum(durations) / len(durations) if durations else 0.0
        
        # Percentiles
        sorted_durations = sorted(durations)
        p95_index = int(0.95 * len(sorted_durations)) if sorted_durations else 0
        p99_index = int(0.99 * len(sorted_durations)) if sorted_durations else 0
        
        p95_response_time = sorted_durations[p95_index] if sorted_durations else 0.0
        p99_response_time = sorted_durations[p99_index] if sorted_durations else 0.0
        
        # Throughput
        requests_per_second = total_requests / 3600.0  # Per hour converted to per second
        
        # Quality scores
        quality_scores = [r["quality_score"] for r in recent_requests 
                         if r["quality_score"] is not None]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Error rate
        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
        
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        return PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            average_quality_score=avg_quality_score,
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        performance = self.get_performance_metrics()
        
        return {
            "performance": {
                "total_requests": performance.total_requests,
                "success_rate": (performance.successful_requests / max(performance.total_requests, 1)) * 100,
                "error_rate": performance.error_rate * 100,
                "average_response_time": performance.average_response_time,
                "p95_response_time": performance.p95_response_time,
                "requests_per_second": performance.requests_per_second,
                "average_quality_score": performance.average_quality_score
            },
            "system": {
                "cpu_usage": performance.cpu_usage,
                "memory_usage": performance.memory_usage
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "active_timers": len(self.active_requests),
            "collection_timestamp": datetime.now().isoformat()
        }
    
    def _make_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{name}[{label_str}]"
    
    def _add_metric(self, metric: Metric):
        """Add a metric to the buffer."""
        self.metrics_buffer.append(metric)
        
        # Keep buffer size manageable
        if len(self.metrics_buffer) > 10000:
            self.metrics_buffer = self.metrics_buffer[-5000:]
    
    async def _collection_loop(self):
        """Background loop for metrics collection and export."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._export_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        self.set_gauge("system_cpu_usage_percent", cpu_percent)
        self.set_gauge("system_memory_usage_percent", memory.percent)
        self.set_gauge("system_memory_available_bytes", memory.available)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.set_gauge("system_disk_usage_percent", (disk.used / disk.total) * 100)
        
        # Network I/O
        network = psutil.net_io_counters()
        self.set_gauge("system_network_bytes_sent", network.bytes_sent)
        self.set_gauge("system_network_bytes_recv", network.bytes_recv)
    
    async def _export_metrics(self):
        """Export metrics to configured endpoints."""
        if not self.metrics_buffer:
            return
        
        # Export to ScrollIntel monitoring
        if self.scrollintel_metrics_endpoint:
            await self._export_to_scrollintel()
        
        # Export to custom handlers
        for handler in self.custom_metric_handlers:
            try:
                handler(self.metrics_buffer.copy())
            except Exception as e:
                logger.error(f"Error in custom metric handler: {e}")
        
        # Clear buffer after export
        self.metrics_buffer.clear()
    
    async def _export_to_scrollintel(self):
        """Export metrics to ScrollIntel monitoring system."""
        try:
            # Convert metrics to ScrollIntel format
            scrollintel_metrics = []
            
            for metric in self.metrics_buffer:
                scrollintel_metric = {
                    "name": f"visual_generation.{metric.name}",
                    "value": metric.value,
                    "type": metric.metric_type.value,
                    "labels": metric.labels,
                    "timestamp": metric.timestamp.isoformat(),
                    "unit": metric.unit,
                    "description": metric.description
                }
                scrollintel_metrics.append(scrollintel_metric)
            
            # Send to ScrollIntel (placeholder - implement actual HTTP client)
            logger.debug(f"Exporting {len(scrollintel_metrics)} metrics to ScrollIntel")
            
            # In a real implementation, this would be an HTTP POST to the metrics endpoint
            # import aiohttp
            # async with aiohttp.ClientSession() as session:
            #     await session.post(self.scrollintel_metrics_endpoint, json=scrollintel_metrics)
            
        except Exception as e:
            logger.error(f"Error exporting metrics to ScrollIntel: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()