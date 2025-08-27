"""
Production Monitoring and Alerting for Visual Generation System
Comprehensive logging, metrics collection, and real-time alerting
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import torch
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

@dataclass
class Alert:
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    generation_requests_total: int
    generation_requests_per_second: float
    average_generation_time: float
    error_rate: float
    gpu_utilization: float
    memory_usage: float
    queue_length: int
    active_workers: int

class PrometheusMetrics:
    """Prometheus metrics collector for visual generation"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Generation metrics
        self.generation_requests_total = Counter(
            'visual_generation_requests_total',
            'Total number of generation requests',
            ['model', 'content_type', 'status'],
            registry=self.registry
        )
        
        self.generation_duration = Histogram(
            'visual_generation_duration_seconds',
            'Time spent generating content',
            ['model', 'content_type'],
            registry=self.registry
        )
        
        self.queue_length = Gauge(
            'visual_generation_queue_length',
            'Number of requests in queue',
            registry=self.registry
        )
        
        self.active_workers = Gauge(
            'visual_generation_active_workers',
            'Number of active worker processes',
            registry=self.registry
        )
        
        # GPU metrics
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        # Business metrics
        self.generation_costs = Counter(
            'visual_generation_costs_total',
            'Total generation costs',
            ['model', 'content_type'],
            registry=self.registry
        )
        
        self.user_requests = Counter(
            'user_requests_total',
            'Total user requests',
            ['user_id', 'plan_type'],
            registry=self.registry
        )
        
        self.quality_scores = Histogram(
            'content_quality_scores',
            'Content quality scores',
            ['model', 'content_type'],
            registry=self.registry
        )
    
    def record_generation_request(self, model: str, content_type: str, status: str):
        """Record a generation request"""
        self.generation_requests_total.labels(
            model=model, 
            content_type=content_type, 
            status=status
        ).inc()
    
    def record_generation_duration(self, model: str, content_type: str, duration: float):
        """Record generation duration"""
        self.generation_duration.labels(
            model=model, 
            content_type=content_type
        ).observe(duration)
    
    def update_queue_length(self, length: int):
        """Update queue length"""
        self.queue_length.set(length)
    
    def update_active_workers(self, count: int):
        """Update active worker count"""
        self.active_workers.set(count)
    
    def update_gpu_metrics(self, gpu_id: str, utilization: float, memory_used: int, temperature: float):
        """Update GPU metrics"""
        self.gpu_utilization.labels(gpu_id=gpu_id).set(utilization)
        self.gpu_memory_used.labels(gpu_id=gpu_id).set(memory_used)
        self.gpu_temperature.labels(gpu_id=gpu_id).set(temperature)
    
    def update_system_metrics(self, cpu_percent: float, memory_bytes: int, disk_usage: Dict[str, int]):
        """Update system metrics"""
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_bytes)
        
        for mount_point, usage in disk_usage.items():
            self.disk_usage.labels(mount_point=mount_point).set(usage)
    
    def record_generation_cost(self, model: str, content_type: str, cost: float):
        """Record generation cost"""
        self.generation_costs.labels(model=model, content_type=content_type).inc(cost)
    
    def record_user_request(self, user_id: str, plan_type: str):
        """Record user request"""
        self.user_requests.labels(user_id=user_id, plan_type=plan_type).inc()
    
    def record_quality_score(self, model: str, content_type: str, score: float):
        """Record content quality score"""
        self.quality_scores.labels(model=model, content_type=content_type).observe(score)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')

class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        
    async def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring_active = True
        logger.info("Starting system monitoring")
        
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await self._collect_gpu_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error("Error in system monitoring", error=str(e))
                await asyncio.sleep(5)  # Short delay before retry
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        logger.info("Stopping system monitoring")
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_bytes = memory.used
            
            # Disk usage
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = usage.used
                except PermissionError:
                    continue
            
            # Update metrics
            self.metrics.update_system_metrics(cpu_percent, memory_bytes, disk_usage)
            
            logger.debug("System metrics collected", 
                        cpu_percent=cpu_percent, 
                        memory_gb=memory_bytes / (1024**3))
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    async def _collect_gpu_metrics(self):
        """Collect GPU metrics"""
        try:
            if not torch.cuda.is_available():
                return
            
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                
                utilization = (memory_reserved / memory_total) * 100
                
                # Temperature (would need nvidia-ml-py for real implementation)
                temperature = 65.0  # Placeholder
                
                # Update metrics
                self.metrics.update_gpu_metrics(
                    gpu_id=str(i),
                    utilization=utilization,
                    memory_used=memory_allocated,
                    temperature=temperature
                )
                
                logger.debug("GPU metrics collected",
                           gpu_id=i,
                           utilization=utilization,
                           memory_gb=memory_allocated / (1024**3))
            
        except Exception as e:
            logger.error("Failed to collect GPU metrics", error=str(e))

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Dict[str, Any]] = []
        
        # Load configuration
        self._load_alert_rules()
        self._load_notification_channels()
    
    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        self.alert_rules = [
            {
                "name": "high_gpu_utilization",
                "condition": lambda metrics: any(
                    gpu_util > 95 for gpu_util in metrics.get("gpu_utilization", {}).values()
                ),
                "severity": AlertSeverity.WARNING,
                "message": "GPU utilization is critically high"
            },
            {
                "name": "high_error_rate",
                "condition": lambda metrics: metrics.get("error_rate", 0) > 0.1,
                "severity": AlertSeverity.ERROR,
                "message": "Error rate is above 10%"
            },
            {
                "name": "long_queue_length",
                "condition": lambda metrics: metrics.get("queue_length", 0) > 50,
                "severity": AlertSeverity.WARNING,
                "message": "Generation queue is backing up"
            },
            {
                "name": "slow_generation_time",
                "condition": lambda metrics: metrics.get("average_generation_time", 0) > 120,
                "severity": AlertSeverity.WARNING,
                "message": "Average generation time is too slow"
            },
            {
                "name": "no_active_workers",
                "condition": lambda metrics: metrics.get("active_workers", 0) == 0,
                "severity": AlertSeverity.CRITICAL,
                "message": "No active workers available"
            }
        ]
    
    def _load_notification_channels(self):
        """Load notification channels from configuration"""
        self.notification_channels = []
        
        # Slack webhook
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            self.notification_channels.append({
                "type": "slack",
                "webhook_url": slack_webhook,
                "channel": "#alerts"
            })
        
        # Email (would need SMTP configuration)
        email_config = os.getenv("ALERT_EMAIL")
        if email_config:
            self.notification_channels.append({
                "type": "email",
                "recipients": email_config.split(",")
            })
        
        # PagerDuty
        pagerduty_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
        if pagerduty_key:
            self.notification_channels.append({
                "type": "pagerduty",
                "integration_key": pagerduty_key
            })
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        try:
            for rule in self.alert_rules:
                try:
                    if rule["condition"](metrics):
                        await self._trigger_alert(rule, metrics)
                    else:
                        # Check if we should resolve an existing alert
                        await self._resolve_alert(rule["name"])
                        
                except Exception as e:
                    logger.error("Error checking alert rule", 
                               rule=rule["name"], 
                               error=str(e))
            
        except Exception as e:
            logger.error("Error in alert checking", error=str(e))
    
    async def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger an alert"""
        try:
            alert_id = f"{rule['name']}_{int(time.time())}"
            
            # Check if alert already exists and is not resolved
            existing_alerts = [
                alert for alert in self.alerts.values()
                if alert.source == rule["name"] and not alert.resolved
            ]
            
            if existing_alerts:
                # Alert already active, don't spam
                return
            
            alert = Alert(
                alert_id=alert_id,
                severity=rule["severity"],
                title=f"Alert: {rule['name']}",
                description=rule["message"],
                source=rule["name"],
                timestamp=datetime.utcnow(),
                metadata={"metrics": metrics}
            )
            
            self.alerts[alert_id] = alert
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.warning("Alert triggered",
                         alert_id=alert_id,
                         severity=alert.severity.value,
                         message=alert.description)
            
        except Exception as e:
            logger.error("Failed to trigger alert", error=str(e))
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve alerts for a rule"""
        try:
            for alert in self.alerts.values():
                if alert.source == rule_name and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    
                    logger.info("Alert resolved",
                              alert_id=alert.alert_id,
                              rule=rule_name)
            
        except Exception as e:
            logger.error("Failed to resolve alert", rule=rule_name, error=str(e))
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications to configured channels"""
        for channel in self.notification_channels:
            try:
                if channel["type"] == "slack":
                    await self._send_slack_notification(alert, channel)
                elif channel["type"] == "email":
                    await self._send_email_notification(alert, channel)
                elif channel["type"] == "pagerduty":
                    await self._send_pagerduty_notification(alert, channel)
                    
            except Exception as e:
                logger.error("Failed to send notification",
                           channel_type=channel["type"],
                           error=str(e))
    
    async def _send_slack_notification(self, alert: Alert, channel: Dict[str, Any]):
        """Send Slack notification"""
        try:
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "channel": channel.get("channel", "#alerts"),
                "username": "Visual Generation Monitor",
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        }
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(channel["webhook_url"], json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack notification sent", alert_id=alert.alert_id)
                    else:
                        logger.error("Slack notification failed", 
                                   status=response.status,
                                   alert_id=alert.alert_id)
            
        except Exception as e:
            logger.error("Slack notification error", error=str(e))
    
    async def _send_email_notification(self, alert: Alert, channel: Dict[str, Any]):
        """Send email notification (placeholder)"""
        # Would implement SMTP email sending here
        logger.info("Email notification would be sent", 
                   alert_id=alert.alert_id,
                   recipients=channel["recipients"])
    
    async def _send_pagerduty_notification(self, alert: Alert, channel: Dict[str, Any]):
        """Send PagerDuty notification (placeholder)"""
        # Would implement PagerDuty API integration here
        logger.info("PagerDuty notification would be sent",
                   alert_id=alert.alert_id,
                   integration_key=channel["integration_key"][:8] + "...")

class PerformanceTracker:
    """Tracks performance metrics and trends"""
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        
    async def collect_performance_snapshot(self) -> PerformanceMetrics:
        """Collect current performance snapshot"""
        try:
            # This would collect actual metrics from the system
            # For now, we'll create a placeholder
            snapshot = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                generation_requests_total=0,
                generation_requests_per_second=0.0,
                average_generation_time=0.0,
                error_rate=0.0,
                gpu_utilization=0.0,
                memory_usage=0.0,
                queue_length=0,
                active_workers=0
            )
            
            # Add to history
            self.performance_history.append(snapshot)
            
            # Maintain history size
            if len(self.performance_history) > self.max_history_size:
                self.performance_history = self.performance_history[-self.max_history_size:]
            
            return snapshot
            
        except Exception as e:
            logger.error("Failed to collect performance snapshot", error=str(e))
            raise
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over specified time period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.performance_history 
                if m.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            # Calculate trends
            trends = {
                "time_period_hours": hours,
                "data_points": len(recent_metrics),
                "average_generation_time": {
                    "current": recent_metrics[-1].average_generation_time,
                    "average": sum(m.average_generation_time for m in recent_metrics) / len(recent_metrics),
                    "min": min(m.average_generation_time for m in recent_metrics),
                    "max": max(m.average_generation_time for m in recent_metrics)
                },
                "error_rate": {
                    "current": recent_metrics[-1].error_rate,
                    "average": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
                    "max": max(m.error_rate for m in recent_metrics)
                },
                "gpu_utilization": {
                    "current": recent_metrics[-1].gpu_utilization,
                    "average": sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
                    "max": max(m.gpu_utilization for m in recent_metrics)
                },
                "requests_per_second": {
                    "current": recent_metrics[-1].generation_requests_per_second,
                    "average": sum(m.generation_requests_per_second for m in recent_metrics) / len(recent_metrics),
                    "max": max(m.generation_requests_per_second for m in recent_metrics)
                }
            }
            
            return trends
            
        except Exception as e:
            logger.error("Failed to calculate performance trends", error=str(e))
            return {}

class ProductionMonitoringManager:
    """Main monitoring manager that coordinates all monitoring components"""
    
    def __init__(self):
        self.metrics = PrometheusMetrics()
        self.system_monitor = SystemMonitor(self.metrics)
        self.alert_manager = AlertManager()
        self.performance_tracker = PerformanceTracker(self.metrics)
        
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
    
    async def start_monitoring(self):
        """Start all monitoring components"""
        try:
            self.monitoring_active = True
            logger.info("Starting production monitoring")
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self.system_monitor.start_monitoring()),
                asyncio.create_task(self._performance_monitoring_loop()),
                asyncio.create_task(self._alert_checking_loop())
            ]
            
            logger.info("Production monitoring started successfully")
            
        except Exception as e:
            logger.error("Failed to start monitoring", error=str(e))
            raise
    
    async def stop_monitoring(self):
        """Stop all monitoring components"""
        try:
            self.monitoring_active = False
            self.system_monitor.stop_monitoring()
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            logger.info("Production monitoring stopped")
            
        except Exception as e:
            logger.error("Error stopping monitoring", error=str(e))
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.monitoring_active:
            try:
                await self.performance_tracker.collect_performance_snapshot()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error("Error in performance monitoring", error=str(e))
                await asyncio.sleep(10)
    
    async def _alert_checking_loop(self):
        """Alert checking loop"""
        while self.monitoring_active:
            try:
                # Get current metrics for alert checking
                metrics = await self._get_current_metrics()
                await self.alert_manager.check_alerts(metrics)
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error("Error in alert checking", error=str(e))
                await asyncio.sleep(10)
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for alert checking"""
        try:
            # This would collect real-time metrics
            # For now, return placeholder data
            return {
                "gpu_utilization": {f"gpu_{i}": 50.0 for i in range(torch.cuda.device_count())},
                "error_rate": 0.02,
                "queue_length": 5,
                "average_generation_time": 45.0,
                "active_workers": 4
            }
        except Exception as e:
            logger.error("Failed to get current metrics", error=str(e))
            return {}
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            "monitoring_active": self.monitoring_active,
            "active_tasks": len([t for t in self.monitoring_tasks if not t.done()]),
            "total_alerts": len(self.alert_manager.alerts),
            "active_alerts": len([a for a in self.alert_manager.alerts.values() if not a.resolved]),
            "performance_history_size": len(self.performance_tracker.performance_history),
            "notification_channels": len(self.alert_manager.notification_channels)
        }
    
    def get_metrics_endpoint(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics.get_metrics()

# Global monitoring manager
monitoring_manager = ProductionMonitoringManager()

async def initialize_monitoring():
    """Initialize production monitoring"""
    try:
        await monitoring_manager.start_monitoring()
        logger.info("Production monitoring initialized successfully")
        return True
    except Exception as e:
        logger.error("Failed to initialize monitoring", error=str(e))
        return False

async def shutdown_monitoring():
    """Shutdown production monitoring"""
    try:
        await monitoring_manager.stop_monitoring()
        logger.info("Production monitoring shutdown complete")
        return True
    except Exception as e:
        logger.error("Failed to shutdown monitoring", error=str(e))
        return False

def get_monitoring_status():
    """Get monitoring status"""
    return monitoring_manager.get_monitoring_status()

def get_prometheus_metrics():
    """Get Prometheus metrics"""
    return monitoring_manager.get_metrics_endpoint()