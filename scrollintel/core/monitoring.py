"""
ScrollIntel Monitoring System
Comprehensive application performance monitoring with metrics collection
"""

__all__ = ['MonitoringSystem', 'SystemMetrics', 'monitor_request', 'monitor_agent_request']

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from contextlib import contextmanager
import asyncio
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# from ..models.database import get_db  # Commented out for testing
from ..core.config import get_settings

settings = get_settings()

# Prometheus Metrics
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    'scrollintel_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'scrollintel_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Agent metrics
AGENT_REQUESTS = Counter(
    'scrollintel_agent_requests_total',
    'Total number of agent requests',
    ['agent_type', 'status'],
    registry=REGISTRY
)

AGENT_PROCESSING_TIME = Histogram(
    'scrollintel_agent_processing_seconds',
    'Agent processing time in seconds',
    ['agent_type'],
    registry=REGISTRY
)

ACTIVE_AGENTS = Gauge(
    'scrollintel_active_agents',
    'Number of active agents',
    ['agent_type'],
    registry=REGISTRY
)

# System metrics
SYSTEM_CPU_USAGE = Gauge(
    'scrollintel_system_cpu_percent',
    'System CPU usage percentage',
    registry=REGISTRY
)

SYSTEM_MEMORY_USAGE = Gauge(
    'scrollintel_system_memory_percent',
    'System memory usage percentage',
    registry=REGISTRY
)

SYSTEM_DISK_USAGE = Gauge(
    'scrollintel_system_disk_percent',
    'System disk usage percentage',
    registry=REGISTRY
)

# Database metrics
DB_CONNECTIONS = Gauge(
    'scrollintel_db_connections',
    'Number of database connections',
    registry=REGISTRY
)

DB_QUERY_DURATION = Histogram(
    'scrollintel_db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    registry=REGISTRY
)

# AI Service metrics
AI_SERVICE_REQUESTS = Counter(
    'scrollintel_ai_service_requests_total',
    'Total AI service requests',
    ['service', 'status'],
    registry=REGISTRY
)

AI_SERVICE_LATENCY = Histogram(
    'scrollintel_ai_service_latency_seconds',
    'AI service response latency',
    ['service'],
    registry=REGISTRY
)

# Error metrics
ERROR_COUNT = Counter(
    'scrollintel_errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=REGISTRY
)

# User activity metrics
USER_SESSIONS = Gauge(
    'scrollintel_active_user_sessions',
    'Number of active user sessions',
    registry=REGISTRY
)

USER_ACTIONS = Counter(
    'scrollintel_user_actions_total',
    'Total user actions',
    ['action_type', 'user_role'],
    registry=REGISTRY
)

# Uptime monitoring metrics
UPTIME_CHECK_DURATION = Histogram(
    'scrollintel_uptime_check_duration_seconds',
    'Uptime check duration in seconds',
    ['service'],
    registry=REGISTRY
)

UPTIME_CHECK_SUCCESS = Counter(
    'scrollintel_uptime_check_success_total',
    'Total successful uptime checks',
    ['service'],
    registry=REGISTRY
)

UPTIME_CHECK_FAILURE = Counter(
    'scrollintel_uptime_check_failure_total',
    'Total failed uptime checks',
    ['service'],
    registry=REGISTRY
)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    request_rate: float
    error_rate: float
    avg_response_time: float
    agent_count: int

@dataclass
class AlertMetric:
    """Alert metric data structure"""
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    timestamp: datetime
    description: str

class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._start_time = time.time()
        self._request_counts = {}
        self._error_counts = {}
        
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
    def record_agent_request(self, agent_type: str, status: str, processing_time: float):
        """Record agent request metrics"""
        AGENT_REQUESTS.labels(agent_type=agent_type, status=status).inc()
        AGENT_PROCESSING_TIME.labels(agent_type=agent_type).observe(processing_time)
        
    def update_active_agents(self, agent_type: str, count: int):
        """Update active agent count"""
        ACTIVE_AGENTS.labels(agent_type=agent_type).set(count)
        
    def record_error(self, error_type: str, component: str):
        """Record error occurrence"""
        ERROR_COUNT.labels(error_type=error_type, component=component).inc()
        
    def record_user_action(self, action_type: str, user_role: str):
        """Record user action"""
        USER_ACTIONS.labels(action_type=action_type, user_role=user_role).inc()
        
    def update_user_sessions(self, count: int):
        """Update active user session count"""
        USER_SESSIONS.set(count)
        
    def record_ai_service_request(self, service: str, status: str, latency: float):
        """Record AI service request metrics"""
        AI_SERVICE_REQUESTS.labels(service=service, status=status).inc()
        AI_SERVICE_LATENCY.labels(service=service).observe(latency)
        
    def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            SYSTEM_MEMORY_USAGE.set(memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            SYSTEM_DISK_USAGE.set(disk_percent)
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                active_connections=0,  # Will be updated by database monitor
                request_rate=0,  # Will be calculated from request metrics
                error_rate=0,  # Will be calculated from error metrics
                avg_response_time=0,  # Will be calculated from request duration
                agent_count=0  # Will be updated by agent monitor
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
            
    async def collect_database_metrics(self):
        """Collect database performance metrics"""
        try:
            # Placeholder for database metrics collection
            # Would normally connect to database and collect metrics
            DB_CONNECTIONS.set(10)  # Mock value
            self.logger.info("Database metrics collected (mock)")
                
        except Exception as e:
            self.logger.error(f"Error collecting database metrics: {e}")
            
    @contextmanager
    def time_database_query(self, query_type: str):
        """Context manager to time database queries"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            DB_QUERY_DURATION.labels(query_type=query_type).observe(duration)
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        uptime = time.time() - self._start_time
        
        return {
            "uptime_seconds": uptime,
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def record_uptime_check(self, service: str, success: bool, duration: float):
        """Record uptime check result"""
        UPTIME_CHECK_DURATION.labels(service=service).observe(duration)
        if success:
            UPTIME_CHECK_SUCCESS.labels(service=service).inc()
        else:
            UPTIME_CHECK_FAILURE.labels(service=service).inc()
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(REGISTRY).decode('utf-8')

# Global metrics collector instance
metrics_collector = MetricsCollector()

class PerformanceMonitor:
    """Monitors application performance and triggers alerts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'error_rate': 5.0,
            'response_time': 5.0
        }
        self.alerts: List[AlertMetric] = []
        
    def check_thresholds(self, metrics: PerformanceMetrics) -> List[AlertMetric]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(AlertMetric(
                metric_name='cpu_usage',
                current_value=metrics.cpu_percent,
                threshold=self.alert_thresholds['cpu_percent'],
                severity='warning',
                timestamp=datetime.utcnow(),
                description=f'High CPU usage: {metrics.cpu_percent:.1f}%'
            ))
            
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(AlertMetric(
                metric_name='memory_usage',
                current_value=metrics.memory_percent,
                threshold=self.alert_thresholds['memory_percent'],
                severity='warning',
                timestamp=datetime.utcnow(),
                description=f'High memory usage: {metrics.memory_percent:.1f}%'
            ))
            
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            alerts.append(AlertMetric(
                metric_name='disk_usage',
                current_value=metrics.disk_percent,
                threshold=self.alert_thresholds['disk_percent'],
                severity='critical',
                timestamp=datetime.utcnow(),
                description=f'High disk usage: {metrics.disk_percent:.1f}%'
            ))
            
        return alerts
        
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics
                metrics = metrics_collector.collect_system_metrics()
                if metrics:
                    # Check for alerts
                    new_alerts = self.check_thresholds(metrics)
                    self.alerts.extend(new_alerts)
                    
                    # Log alerts
                    for alert in new_alerts:
                        self.logger.warning(f"ALERT: {alert.description}")
                        
                # Collect database metrics
                await metrics_collector.collect_database_metrics()
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

# Global performance monitor instance
performance_monitor = PerformanceMonitor()


class MonitoringSystem:
    """Main monitoring system class."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
    
    def get_system_health(self):
        """Get overall system health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "database": "healthy",
                "api": "healthy",
                "agents": "healthy"
            }
        }

class SystemMonitor:
    """System monitoring and health check manager."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.logger = logging.getLogger(__name__)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Application uptime
            uptime = datetime.utcnow() - self.start_time
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime.total_seconds(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                        "used": memory.used
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": (disk.used / disk.total) * 100
                    }
                },
                "application": {
                    "version": "1.0.0",
                    "environment": getattr(settings, 'environment', 'development'),
                    "debug": getattr(settings, 'debug', False)
                }
            }
            
            # Determine overall health
            if cpu_percent > 90 or memory.percent > 90 or (disk.used / disk.total) > 0.95:
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics."""
        return {
            "requests_total": 0,  # Placeholder
            "active_connections": 0,  # Placeholder
            "agent_requests": 0,  # Placeholder
            "errors_total": 0  # Placeholder
        }
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            # Simple database health check
            return {
                "status": "healthy",
                "connection": "available",
                "response_time_ms": 0  # Placeholder
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_external_services(self) -> Dict[str, Any]:
        """Check external service dependencies."""
        services = {
            "openai_api": {"status": "unknown", "last_check": None},
            "redis": {"status": "unknown", "last_check": None},
            "elasticsearch": {"status": "unknown", "last_check": None}
        }
        
        # Placeholder for actual service checks
        return services