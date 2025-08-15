"""
Monitoring and Logging Infrastructure with Prometheus and Grafana
"""
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import threading
import queue
import requests

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricDefinition:
    """Metric definition for Prometheus"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    query: str
    threshold: float
    severity: AlertSeverity
    duration: str = "5m"
    description: str = ""

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    message: str
    component: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        core_metrics = [
            MetricDefinition("scrollintel_requests_total", "Total HTTP requests", MetricType.COUNTER, ["method", "endpoint", "status"]),
            MetricDefinition("scrollintel_request_duration_seconds", "HTTP request duration", MetricType.HISTOGRAM, ["method", "endpoint"]),
            MetricDefinition("scrollintel_active_users", "Number of active users", MetricType.GAUGE),
            MetricDefinition("scrollintel_data_products_total", "Total data products", MetricType.GAUGE, ["status"]),
            MetricDefinition("scrollintel_agent_executions_total", "Total agent executions", MetricType.COUNTER, ["agent_type", "status"]),
            MetricDefinition("scrollintel_agent_execution_duration_seconds", "Agent execution duration", MetricType.HISTOGRAM, ["agent_type"]),
            MetricDefinition("scrollintel_cache_hits_total", "Cache hits", MetricType.COUNTER, ["cache_type"]),
            MetricDefinition("scrollintel_cache_misses_total", "Cache misses", MetricType.COUNTER, ["cache_type"]),
            MetricDefinition("scrollintel_errors_total", "Total errors", MetricType.COUNTER, ["component", "error_type"]),
            MetricDefinition("scrollintel_system_cpu_usage", "System CPU usage", MetricType.GAUGE),
            MetricDefinition("scrollintel_system_memory_usage", "System memory usage", MetricType.GAUGE),
            MetricDefinition("scrollintel_kubernetes_pods", "Number of Kubernetes pods", MetricType.GAUGE, ["status"]),
        ]
        
        for metric_def in core_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric"""
        if metric_def.metric_type == MetricType.COUNTER:
            metric = Counter(
                metric_def.name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.HISTOGRAM:
            metric = Histogram(
                metric_def.name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.GAUGE:
            metric = Gauge(
                metric_def.name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {metric_def.metric_type}")
        
        self.metrics[metric_def.name] = metric
        logger.info(f"Registered metric: {metric_def.name}")
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1):
        """Increment counter metric"""
        metric = self.metrics.get(name)
        if metric and hasattr(metric, 'inc'):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe histogram metric"""
        metric = self.metrics.get(name)
        if metric and hasattr(metric, 'observe'):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge metric value"""
        metric = self.metrics.get(name)
        if metric and hasattr(metric, 'set'):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

class StructuredLogger:
    """Structured logging with JSON format"""
    
    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(component)
        self.log_queue = queue.Queue()
        self._setup_handler()
    
    def _setup_handler(self):
        """Setup structured logging handler"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _create_log_entry(self, level: str, message: str, **kwargs) -> LogEntry:
        """Create structured log entry"""
        return LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            component=self.component,
            user_id=kwargs.get('user_id'),
            request_id=kwargs.get('request_id'),
            metadata=kwargs.get('metadata', {})
        )
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        entry = self._create_log_entry("INFO", message, **kwargs)
        self.logger.info(json.dumps({
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level,
            "message": entry.message,
            "component": entry.component,
            "user_id": entry.user_id,
            "request_id": entry.request_id,
            "metadata": entry.metadata
        }))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        entry = self._create_log_entry("WARNING", message, **kwargs)
        self.logger.warning(json.dumps({
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level,
            "message": entry.message,
            "component": entry.component,
            "user_id": entry.user_id,
            "request_id": entry.request_id,
            "metadata": entry.metadata
        }))
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        entry = self._create_log_entry("ERROR", message, **kwargs)
        self.logger.error(json.dumps({
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level,
            "message": entry.message,
            "component": entry.component,
            "user_id": entry.user_id,
            "request_id": entry.request_id,
            "metadata": entry.metadata
        }))
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        entry = self._create_log_entry("CRITICAL", message, **kwargs)
        self.logger.critical(json.dumps({
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level,
            "message": entry.message,
            "component": entry.component,
            "user_id": entry.user_id,
            "request_id": entry.request_id,
            "metadata": entry.metadata
        }))

class AlertManager:
    """Alert management system"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, datetime] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                query="rate(scrollintel_errors_total[5m]) > 0.1",
                threshold=0.1,
                severity=AlertSeverity.WARNING,
                description="High error rate detected"
            ),
            AlertRule(
                name="high_response_time",
                query="histogram_quantile(0.95, rate(scrollintel_request_duration_seconds_bucket[5m])) > 2",
                threshold=2.0,
                severity=AlertSeverity.WARNING,
                description="High response time detected"
            ),
            AlertRule(
                name="low_cache_hit_rate",
                query="rate(scrollintel_cache_hits_total[5m]) / (rate(scrollintel_cache_hits_total[5m]) + rate(scrollintel_cache_misses_total[5m])) < 0.8",
                threshold=0.8,
                severity=AlertSeverity.WARNING,
                description="Low cache hit rate"
            ),
            AlertRule(
                name="high_cpu_usage",
                query="scrollintel_system_cpu_usage > 80",
                threshold=80.0,
                severity=AlertSeverity.CRITICAL,
                description="High CPU usage"
            ),
            AlertRule(
                name="high_memory_usage",
                query="scrollintel_system_memory_usage > 85",
                threshold=85.0,
                severity=AlertSeverity.CRITICAL,
                description="High memory usage"
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def trigger_alert(self, rule_name: str, message: str, severity: AlertSeverity):
        """Trigger an alert"""
        alert_key = f"{rule_name}_{severity.value}"
        
        # Check if alert is already active (avoid spam)
        if alert_key in self.active_alerts:
            last_triggered = self.active_alerts[alert_key]
            if datetime.utcnow() - last_triggered < timedelta(minutes=5):
                return  # Skip if triggered recently
        
        self.active_alerts[alert_key] = datetime.utcnow()
        
        alert_data = {
            "rule_name": rule_name,
            "message": message,
            "severity": severity.value,
            "timestamp": datetime.utcnow().isoformat(),
            "component": "scrollintel-g6"
        }
        
        # Send to webhook if configured
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=alert_data, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send alert webhook: {e}")
        
        # Log alert
        logger.warning(f"ALERT [{severity.value.upper()}] {rule_name}: {message}")
    
    def resolve_alert(self, rule_name: str, severity: AlertSeverity):
        """Resolve an active alert"""
        alert_key = f"{rule_name}_{severity.value}"
        if alert_key in self.active_alerts:
            del self.active_alerts[alert_key]
            logger.info(f"Resolved alert: {rule_name}")

class MonitoringInfrastructure:
    """Main monitoring infrastructure coordinator"""
    
    def __init__(self):
        self.metrics = PrometheusMetrics()
        self.logger = StructuredLogger("monitoring")
        self.alert_manager = AlertManager()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
    
    def start_monitoring(self):
        """Start monitoring background tasks"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
            self.logger.info("Started monitoring infrastructure")
    
    def stop_monitoring(self):
        """Stop monitoring background tasks"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Stopped monitoring infrastructure")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                self._collect_system_metrics()
                self._check_alerts()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("scrollintel_system_cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics.set_gauge("scrollintel_system_memory_usage", memory_percent)
            
            # Check for alerts
            if cpu_percent > 80:
                self.alert_manager.trigger_alert(
                    "high_cpu_usage",
                    f"CPU usage is {cpu_percent:.1f}%",
                    AlertSeverity.CRITICAL
                )
            
            if memory_percent > 85:
                self.alert_manager.trigger_alert(
                    "high_memory_usage",
                    f"Memory usage is {memory_percent:.1f}%",
                    AlertSeverity.CRITICAL
                )
                
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _check_alerts(self):
        """Check alert conditions"""
        # This would typically query Prometheus for alert conditions
        # For now, we'll implement basic checks
        pass
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.metrics.increment_counter(
            "scrollintel_requests_total",
            {"method": method, "endpoint": endpoint, "status": str(status_code)}
        )
        
        self.metrics.observe_histogram(
            "scrollintel_request_duration_seconds",
            duration,
            {"method": method, "endpoint": endpoint}
        )
    
    def record_agent_execution(self, agent_type: str, status: str, duration: float):
        """Record agent execution metrics"""
        self.metrics.increment_counter(
            "scrollintel_agent_executions_total",
            {"agent_type": agent_type, "status": status}
        )
        
        self.metrics.observe_histogram(
            "scrollintel_agent_execution_duration_seconds",
            duration,
            {"agent_type": agent_type}
        )
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics"""
        if hit:
            self.metrics.increment_counter(
                "scrollintel_cache_hits_total",
                {"cache_type": cache_type}
            )
        else:
            self.metrics.increment_counter(
                "scrollintel_cache_misses_total",
                {"cache_type": cache_type}
            )
    
    def record_error(self, component: str, error_type: str):
        """Record error metrics"""
        self.metrics.increment_counter(
            "scrollintel_errors_total",
            {"component": component, "error_type": error_type}
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": len(self.alert_manager.active_alerts),
            "monitoring_active": not self._stop_monitoring.is_set()
        }

# Global monitoring infrastructure
_monitoring_infrastructure: Optional[MonitoringInfrastructure] = None

def get_monitoring_infrastructure() -> MonitoringInfrastructure:
    """Get global monitoring infrastructure instance"""
    global _monitoring_infrastructure
    
    if _monitoring_infrastructure is None:
        _monitoring_infrastructure = MonitoringInfrastructure()
        _monitoring_infrastructure.start_monitoring()
    
    return _monitoring_infrastructure

def monitor_request(method: str, endpoint: str):
    """Decorator to monitor HTTP requests"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            monitoring = get_monitoring_infrastructure()
            
            try:
                result = f(*args, **kwargs)
                status_code = getattr(result, 'status_code', 200)
                duration = time.time() - start_time
                
                monitoring.record_request(method, endpoint, status_code, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitoring.record_request(method, endpoint, 500, duration)
                monitoring.record_error("api", type(e).__name__)
                raise
        
        return decorated_function
    return decorator