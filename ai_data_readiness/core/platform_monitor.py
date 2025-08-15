"""Platform monitoring and health tracking for AI Data Readiness Platform."""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
from pathlib import Path

from ..models.base_models import Dataset, DatasetStatus
from .config import get_settings


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_available_gb': self.memory_available_gb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'active_connections': self.active_connections
        }


@dataclass
class PlatformMetrics:
    """Platform-specific metrics."""
    timestamp: datetime
    active_datasets: int
    processing_datasets: int
    failed_datasets: int
    total_data_processed_gb: float
    avg_processing_time_seconds: float
    quality_assessments_completed: int
    bias_analyses_completed: int
    api_requests_per_minute: float
    error_rate_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_datasets': self.active_datasets,
            'processing_datasets': self.processing_datasets,
            'failed_datasets': self.failed_datasets,
            'total_data_processed_gb': self.total_data_processed_gb,
            'avg_processing_time_seconds': self.avg_processing_time_seconds,
            'quality_assessments_completed': self.quality_assessments_completed,
            'bias_analyses_completed': self.bias_analyses_completed,
            'api_requests_per_minute': self.api_requests_per_minute,
            'error_rate_percent': self.error_rate_percent
        }


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """Mark operation as complete."""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class PlatformMonitor:
    """Comprehensive platform monitoring system."""
    
    def __init__(self, metrics_retention_hours: int = 24):
        self.config = get_settings()
        self.logger = logging.getLogger(__name__)
        self.metrics_retention_hours = metrics_retention_hours
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=1000)
        self.platform_metrics: deque = deque(maxlen=1000)
        self.performance_metrics: deque = deque(maxlen=5000)
        
        # Counters and tracking
        self.operation_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.api_request_counter = 0
        self.last_api_reset = datetime.utcnow()
        
        # Active operations tracking
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate_percent': 5.0,
            'avg_processing_time_seconds': 300.0
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Platform monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Platform monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.collect_system_metrics()
                self.collect_platform_metrics()
                self._cleanup_old_metrics()
                self._check_alerts()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Connection count
            connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=connections
            )
            
            self.system_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            raise
    
    def collect_platform_metrics(self) -> PlatformMetrics:
        """Collect platform-specific metrics."""
        try:
            # Calculate API requests per minute
            now = datetime.utcnow()
            time_diff = (now - self.last_api_reset).total_seconds() / 60
            api_requests_per_minute = self.api_request_counter / max(time_diff, 1)
            
            # Reset API counter every hour
            if time_diff >= 60:
                self.api_request_counter = 0
                self.last_api_reset = now
            
            # Calculate error rate
            total_operations = sum(self.operation_counters.values())
            total_errors = sum(self.error_counters.values())
            error_rate_percent = (total_errors / max(total_operations, 1)) * 100
            
            # Calculate average processing time
            recent_operations = [
                m for m in self.performance_metrics
                if m.duration_seconds and m.end_time and 
                m.end_time > now - timedelta(hours=1)
            ]
            avg_processing_time = (
                sum(op.duration_seconds for op in recent_operations) / 
                max(len(recent_operations), 1)
            )
            
            metrics = PlatformMetrics(
                timestamp=now,
                active_datasets=self.operation_counters.get('active_datasets', 0),
                processing_datasets=self.operation_counters.get('processing_datasets', 0),
                failed_datasets=self.error_counters.get('dataset_processing_failed', 0),
                total_data_processed_gb=self.operation_counters.get('data_processed_gb', 0),
                avg_processing_time_seconds=avg_processing_time,
                quality_assessments_completed=self.operation_counters.get('quality_assessments', 0),
                bias_analyses_completed=self.operation_counters.get('bias_analyses', 0),
                api_requests_per_minute=api_requests_per_minute,
                error_rate_percent=error_rate_percent
            )
            
            self.platform_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting platform metrics: {e}")
            raise
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.active_operations[operation_id] = metrics
        self.operation_counters[operation_name] += 1
        
        return operation_id
    
    def complete_operation(self, operation_id: str, success: bool = True, 
                          error_message: Optional[str] = None):
        """Complete an operation and record metrics."""
        if operation_id not in self.active_operations:
            self.logger.warning(f"Operation {operation_id} not found in active operations")
            return
        
        metrics = self.active_operations.pop(operation_id)
        metrics.complete(success=success, error_message=error_message)
        
        self.performance_metrics.append(metrics)
        
        if not success:
            self.error_counters[metrics.operation_name] += 1
    
    def record_api_request(self):
        """Record an API request."""
        self.api_request_counter += 1
    
    def update_dataset_count(self, status: DatasetStatus, delta: int = 1):
        """Update dataset counters."""
        if status == DatasetStatus.READY:
            self.operation_counters['active_datasets'] += delta
        elif status == DatasetStatus.PROCESSING:
            self.operation_counters['processing_datasets'] += delta
        elif status == DatasetStatus.ERROR:
            self.operation_counters['failed_datasets'] += delta
    
    def record_data_processed(self, size_gb: float):
        """Record amount of data processed."""
        self.operation_counters['data_processed_gb'] += size_gb
    
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.system_metrics[-1] if self.system_metrics else None
    
    def get_current_platform_metrics(self) -> Optional[PlatformMetrics]:
        """Get the most recent platform metrics."""
        return self.platform_metrics[-1] if self.platform_metrics else None
    
    def get_metrics_history(self, hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        system_history = [
            m.to_dict() for m in self.system_metrics
            if m.timestamp > cutoff_time
        ]
        
        platform_history = [
            m.to_dict() for m in self.platform_metrics
            if m.timestamp > cutoff_time
        ]
        
        performance_history = [
            m.to_dict() for m in self.performance_metrics
            if m.start_time > cutoff_time
        ]
        
        return {
            'system_metrics': system_history,
            'platform_metrics': platform_history,
            'performance_metrics': performance_history
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall platform health status."""
        current_system = self.get_current_system_metrics()
        current_platform = self.get_current_platform_metrics()
        
        if not current_system or not current_platform:
            return {'status': 'unknown', 'message': 'Insufficient metrics data'}
        
        health_issues = []
        
        # Check system health
        if current_system.cpu_percent > self.alert_thresholds['cpu_percent']:
            health_issues.append(f"High CPU usage: {current_system.cpu_percent:.1f}%")
        
        if current_system.memory_percent > self.alert_thresholds['memory_percent']:
            health_issues.append(f"High memory usage: {current_system.memory_percent:.1f}%")
        
        if current_system.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            health_issues.append(f"High disk usage: {current_system.disk_usage_percent:.1f}%")
        
        # Check platform health
        if current_platform.error_rate_percent > self.alert_thresholds['error_rate_percent']:
            health_issues.append(f"High error rate: {current_platform.error_rate_percent:.1f}%")
        
        if current_platform.avg_processing_time_seconds > self.alert_thresholds['avg_processing_time_seconds']:
            health_issues.append(f"Slow processing: {current_platform.avg_processing_time_seconds:.1f}s avg")
        
        if health_issues:
            status = 'warning' if len(health_issues) <= 2 else 'critical'
            return {
                'status': status,
                'issues': health_issues,
                'system_metrics': current_system.to_dict(),
                'platform_metrics': current_platform.to_dict()
            }
        
        return {
            'status': 'healthy',
            'message': 'All systems operating normally',
            'system_metrics': current_system.to_dict(),
            'platform_metrics': current_platform.to_dict()
        }
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self):
        """Check for alert conditions and trigger callbacks."""
        health_status = self.get_health_status()
        
        if health_status['status'] in ['warning', 'critical']:
            alert_data = {
                'severity': health_status['status'],
                'timestamp': datetime.utcnow().isoformat(),
                'issues': health_status.get('issues', []),
                'metrics': {
                    'system': health_status.get('system_metrics', {}),
                    'platform': health_status.get('platform_metrics', {})
                }
            }
            
            for callback in self.alert_callbacks:
                try:
                    callback(health_status['status'], alert_data)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean up performance metrics
        self.performance_metrics = deque([
            m for m in self.performance_metrics
            if m.start_time > cutoff_time
        ], maxlen=5000)
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file."""
        metrics_data = self.get_metrics_history(hours=hours)
        metrics_data['export_timestamp'] = datetime.utcnow().isoformat()
        metrics_data['health_status'] = self.get_health_status()
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")


# Global monitor instance
_monitor_instance = None


def get_platform_monitor() -> PlatformMonitor:
    """Get global platform monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PlatformMonitor()
    return _monitor_instance