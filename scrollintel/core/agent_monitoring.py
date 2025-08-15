"""
Agent Monitoring System for ScrollIntel

This module provides comprehensive monitoring, metrics collection, and health
checking capabilities for all agents in the ScrollIntel system.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
from collections import defaultdict, deque
import statistics
import weakref

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class HealthCheckStatus(Enum):
    """Health check status values"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"

@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    """Metric definition and current state"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthCheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert definition and state"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    agent_id: Optional[str] = None
    metric_name: Optional[str] = None
    active: bool = False
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentHealthStatus:
    """Comprehensive agent health status"""
    agent_id: str
    overall_status: HealthCheckStatus
    checks: List[HealthCheckResult] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    last_heartbeat: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    build_info: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """
    Collects and manages metrics for agents.
    """
    
    def __init__(self, max_metrics_per_agent: int = 100):
        self.max_metrics_per_agent = max_metrics_per_agent
        self._metrics: Dict[str, Dict[str, Metric]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._collectors: Dict[str, Callable] = {}
        
    def register_metric(self, agent_id: str, name: str, metric_type: MetricType,
                       description: str, unit: str = "", labels: Dict[str, str] = None):
        """Register a new metric for an agent"""
        with self._lock:
            if len(self._metrics[agent_id]) >= self.max_metrics_per_agent:
                logger.warning(f"Maximum metrics reached for agent {agent_id}")
                return False
            
            metric = Metric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                labels=labels or {}
            )
            
            self._metrics[agent_id][name] = metric
            logger.debug(f"Registered metric {name} for agent {agent_id}")
            return True
    
    def record_value(self, agent_id: str, metric_name: str, value: float,
                    labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric value"""
        with self._lock:
            if agent_id not in self._metrics or metric_name not in self._metrics[agent_id]:
                logger.warning(f"Metric {metric_name} not found for agent {agent_id}")
                return False
            
            metric = self._metrics[agent_id][metric_name]
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            metric.values.append(metric_value)
            metric.last_updated = datetime.now()
            return True
    
    def increment_counter(self, agent_id: str, metric_name: str, 
                         increment: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            if agent_id in self._metrics and metric_name in self._metrics[agent_id]:
                metric = self._metrics[agent_id][metric_name]
                if metric.metric_type == MetricType.COUNTER:
                    current_value = self.get_current_value(agent_id, metric_name) or 0
                    self.record_value(agent_id, metric_name, current_value + increment, labels)
                    return True
            return False
    
    def set_gauge(self, agent_id: str, metric_name: str, value: float,
                 labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        with self._lock:
            if agent_id in self._metrics and metric_name in self._metrics[agent_id]:
                metric = self._metrics[agent_id][metric_name]
                if metric.metric_type == MetricType.GAUGE:
                    self.record_value(agent_id, metric_name, value, labels)
                    return True
            return False
    
    def record_timer(self, agent_id: str, metric_name: str, duration: float,
                    labels: Dict[str, str] = None):
        """Record a timer metric"""
        with self._lock:
            if agent_id in self._metrics and metric_name in self._metrics[agent_id]:
                metric = self._metrics[agent_id][metric_name]
                if metric.metric_type == MetricType.TIMER:
                    self.record_value(agent_id, metric_name, duration, labels)
                    return True
            return False
    
    def get_current_value(self, agent_id: str, metric_name: str) -> Optional[float]:
        """Get the current value of a metric"""
        with self._lock:
            if agent_id in self._metrics and metric_name in self._metrics[agent_id]:
                metric = self._metrics[agent_id][metric_name]
                if metric.values:
                    return metric.values[-1].value
            return None
    
    def get_metric_statistics(self, agent_id: str, metric_name: str,
                            time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        with self._lock:
            if agent_id not in self._metrics or metric_name not in self._metrics[agent_id]:
                return {}
            
            metric = self._metrics[agent_id][metric_name]
            values = []
            
            cutoff_time = datetime.now() - time_window if time_window else None
            
            for metric_value in metric.values:
                if cutoff_time is None or metric_value.timestamp >= cutoff_time:
                    values.append(metric_value.value)
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "sum": sum(values),
                "latest": values[-1] if values else 0.0
            }
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Metric]:
        """Get all metrics for an agent"""
        with self._lock:
            return dict(self._metrics.get(agent_id, {}))
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Metric]]:
        """Get all metrics for all agents"""
        with self._lock:
            return {agent_id: dict(metrics) for agent_id, metrics in self._metrics.items()}
    
    def cleanup_old_metrics(self, max_age: timedelta = timedelta(hours=24)):
        """Clean up old metric values"""
        cutoff_time = datetime.now() - max_age
        
        with self._lock:
            for agent_metrics in self._metrics.values():
                for metric in agent_metrics.values():
                    # Remove old values
                    while metric.values and metric.values[0].timestamp < cutoff_time:
                        metric.values.popleft()

class HealthChecker:
    """
    Manages health checks for agents.
    """
    
    def __init__(self):
        self._health_checks: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self._health_results: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        self._lock = threading.RLock()
        self._check_intervals: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._last_check_times: Dict[str, Dict[str, datetime]] = defaultdict(dict)
    
    def register_health_check(self, agent_id: str, check_name: str, 
                            check_function: Callable, interval_seconds: int = 30):
        """Register a health check for an agent"""
        with self._lock:
            self._health_checks[agent_id][check_name] = check_function
            self._check_intervals[agent_id][check_name] = interval_seconds
            logger.debug(f"Registered health check {check_name} for agent {agent_id}")
    
    def unregister_health_check(self, agent_id: str, check_name: str):
        """Unregister a health check"""
        with self._lock:
            self._health_checks[agent_id].pop(check_name, None)
            self._check_intervals[agent_id].pop(check_name, None)
            self._last_check_times[agent_id].pop(check_name, None)
    
    async def run_health_check(self, agent_id: str, check_name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if agent_id not in self._health_checks or check_name not in self._health_checks[agent_id]:
            return HealthCheckResult(
                name=check_name,
                status=HealthCheckStatus.UNKNOWN,
                message=f"Health check {check_name} not found for agent {agent_id}"
            )
        
        check_function = self._health_checks[agent_id][check_name]
        start_time = time.time()
        
        try:
            # Run the health check
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
            
            duration = time.time() - start_time
            
            # Convert result to HealthCheckResult if needed
            if isinstance(result, HealthCheckResult):
                result.duration = duration
                health_result = result
            elif isinstance(result, dict):
                health_result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus(result.get("status", "unknown")),
                    message=result.get("message", ""),
                    details=result.get("details", {}),
                    duration=duration,
                    metadata=result.get("metadata", {})
                )
            else:
                # Assume boolean result
                health_result = HealthCheckResult(
                    name=check_name,
                    status=HealthCheckStatus.PASS if result else HealthCheckStatus.FAIL,
                    message="Health check passed" if result else "Health check failed",
                    duration=duration
                )
            
            # Store result
            with self._lock:
                self._health_results[agent_id].append(health_result)
                # Keep only last 100 results per check
                if len(self._health_results[agent_id]) > 100:
                    self._health_results[agent_id] = self._health_results[agent_id][-100:]
                
                self._last_check_times[agent_id][check_name] = datetime.now()
            
            return health_result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Health check {check_name} failed for agent {agent_id}: {e}")
            
            error_result = HealthCheckResult(
                name=check_name,
                status=HealthCheckStatus.FAIL,
                message=f"Health check failed: {str(e)}",
                duration=duration,
                metadata={"exception": str(e)}
            )
            
            with self._lock:
                self._health_results[agent_id].append(error_result)
                self._last_check_times[agent_id][check_name] = datetime.now()
            
            return error_result
    
    async def run_all_health_checks(self, agent_id: str) -> List[HealthCheckResult]:
        """Run all health checks for an agent"""
        if agent_id not in self._health_checks:
            return []
        
        results = []
        for check_name in self._health_checks[agent_id]:
            result = await self.run_health_check(agent_id, check_name)
            results.append(result)
        
        return results
    
    def get_health_status(self, agent_id: str) -> AgentHealthStatus:
        """Get comprehensive health status for an agent"""
        with self._lock:
            recent_results = []
            
            # Get most recent result for each check
            check_results = {}
            for result in reversed(self._health_results.get(agent_id, [])):
                if result.name not in check_results:
                    check_results[result.name] = result
            
            recent_results = list(check_results.values())
            
            # Determine overall status
            if not recent_results:
                overall_status = HealthCheckStatus.UNKNOWN
            elif any(r.status == HealthCheckStatus.FAIL for r in recent_results):
                overall_status = HealthCheckStatus.FAIL
            elif any(r.status == HealthCheckStatus.WARN for r in recent_results):
                overall_status = HealthCheckStatus.WARN
            else:
                overall_status = HealthCheckStatus.PASS
            
            return AgentHealthStatus(
                agent_id=agent_id,
                overall_status=overall_status,
                checks=recent_results,
                last_heartbeat=datetime.now()
            )
    
    def should_run_check(self, agent_id: str, check_name: str) -> bool:
        """Check if a health check should be run based on its interval"""
        with self._lock:
            if agent_id not in self._check_intervals or check_name not in self._check_intervals[agent_id]:
                return True
            
            interval = self._check_intervals[agent_id][check_name]
            last_check = self._last_check_times[agent_id].get(check_name)
            
            if last_check is None:
                return True
            
            return (datetime.now() - last_check).total_seconds() >= interval

class AlertManager:
    """
    Manages alerts based on metrics and health checks.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, health_checker: HealthChecker):
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self._alerts: Dict[str, Alert] = {}
        self._alert_handlers: List[Callable] = []
        self._lock = threading.RLock()
        self._evaluation_task = None
        self._shutdown_event = asyncio.Event()
    
    def register_alert(self, alert: Alert):
        """Register a new alert"""
        with self._lock:
            self._alerts[alert.id] = alert
            logger.info(f"Registered alert {alert.name} ({alert.id})")
    
    def unregister_alert(self, alert_id: str):
        """Unregister an alert"""
        with self._lock:
            if alert_id in self._alerts:
                del self._alerts[alert_id]
                logger.info(f"Unregistered alert {alert_id}")
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function"""
        self._alert_handlers.append(handler)
    
    async def start_monitoring(self, evaluation_interval: int = 30):
        """Start the alert evaluation loop"""
        self._evaluation_task = asyncio.create_task(
            self._evaluation_loop(evaluation_interval)
        )
    
    async def stop_monitoring(self):
        """Stop the alert evaluation loop"""
        self._shutdown_event.set()
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
    
    async def _evaluation_loop(self, interval: int):
        """Main alert evaluation loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(interval)
    
    async def _evaluate_alerts(self):
        """Evaluate all alerts"""
        with self._lock:
            alerts_to_evaluate = list(self._alerts.values())
        
        for alert in alerts_to_evaluate:
            try:
                should_trigger = await self._evaluate_alert_condition(alert)
                
                if should_trigger and not alert.active:
                    # Trigger alert
                    alert.active = True
                    alert.triggered_at = datetime.now()
                    alert.trigger_count += 1
                    await self._trigger_alert(alert)
                    
                elif not should_trigger and alert.active:
                    # Resolve alert
                    alert.active = False
                    alert.resolved_at = datetime.now()
                    await self._resolve_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert {alert.id}: {e}")
    
    async def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate if an alert condition is met"""
        if alert.metric_name and alert.agent_id:
            # Metric-based alert
            current_value = self.metrics_collector.get_current_value(
                alert.agent_id, alert.metric_name
            )
            
            if current_value is None:
                return False
            
            # Simple threshold comparison for now
            # In a full implementation, this would support complex conditions
            if ">" in alert.condition:
                return current_value > alert.threshold
            elif "<" in alert.condition:
                return current_value < alert.threshold
            elif "==" in alert.condition:
                return abs(current_value - alert.threshold) < 0.001
        
        return False
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        logger.warning(f"Alert triggered: {alert.name} ({alert.id})")
        
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, "triggered")
                else:
                    handler(alert, "triggered")
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        logger.info(f"Alert resolved: {alert.name} ({alert.id})")
        
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, "resolved")
                else:
                    handler(alert, "resolved")
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return [alert for alert in self._alerts.values() if alert.active]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts"""
        with self._lock:
            return list(self._alerts.values())

class AgentMonitor:
    """
    Main monitoring system that coordinates metrics, health checks, and alerts.
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager(self.metrics_collector, self.health_checker)
        self._monitored_agents: Set[str] = set()
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
        
        # Register default metrics and health checks
        self._setup_default_monitoring()
    
    def _setup_default_monitoring(self):
        """Setup default metrics and health checks"""
        # Default metrics that all agents should have
        default_metrics = [
            ("requests_total", MetricType.COUNTER, "Total number of requests processed"),
            ("requests_successful", MetricType.COUNTER, "Number of successful requests"),
            ("requests_failed", MetricType.COUNTER, "Number of failed requests"),
            ("response_time", MetricType.HISTOGRAM, "Request response time", "seconds"),
            ("current_load", MetricType.GAUGE, "Current load percentage", "percent"),
            ("memory_usage", MetricType.GAUGE, "Memory usage", "bytes"),
            ("cpu_usage", MetricType.GAUGE, "CPU usage", "percent")
        ]
        
        # These will be registered when agents are added
        self._default_metrics = default_metrics
    
    async def add_agent(self, agent_id: str, agent_name: str = ""):
        """Add an agent to monitoring"""
        with self._lock:
            if agent_id in self._monitored_agents:
                logger.warning(f"Agent {agent_id} already being monitored")
                return
            
            self._monitored_agents.add(agent_id)
        
        # Register default metrics
        for name, metric_type, description, *unit in self._default_metrics:
            unit_str = unit[0] if unit else ""
            self.metrics_collector.register_metric(
                agent_id, name, metric_type, description, unit_str
            )
        
        # Register default health checks
        self.health_checker.register_health_check(
            agent_id, "basic_health", self._basic_health_check
        )
        
        # Start monitoring task for this agent
        task = asyncio.create_task(self._monitor_agent(agent_id))
        self._monitoring_tasks[agent_id] = task
        
        logger.info(f"Started monitoring agent {agent_id}")
    
    async def remove_agent(self, agent_id: str):
        """Remove an agent from monitoring"""
        with self._lock:
            if agent_id not in self._monitored_agents:
                return
            
            self._monitored_agents.remove(agent_id)
        
        # Cancel monitoring task
        if agent_id in self._monitoring_tasks:
            self._monitoring_tasks[agent_id].cancel()
            del self._monitoring_tasks[agent_id]
        
        logger.info(f"Stopped monitoring agent {agent_id}")
    
    async def _monitor_agent(self, agent_id: str):
        """Monitor a specific agent"""
        while agent_id in self._monitored_agents:
            try:
                # Run health checks that are due
                for check_name in self.health_checker._health_checks.get(agent_id, {}):
                    if self.health_checker.should_run_check(agent_id, check_name):
                        await self.health_checker.run_health_check(agent_id, check_name)
                
                # Clean up old metrics periodically
                self.metrics_collector.cleanup_old_metrics()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring agent {agent_id}: {e}")
                await asyncio.sleep(30)
    
    async def _basic_health_check(self) -> HealthCheckResult:
        """Basic health check that always passes"""
        return HealthCheckResult(
            name="basic_health",
            status=HealthCheckStatus.PASS,
            message="Basic health check passed"
        )
    
    def record_request_metric(self, agent_id: str, success: bool, response_time: float):
        """Record request metrics for an agent"""
        self.metrics_collector.increment_counter(agent_id, "requests_total")
        
        if success:
            self.metrics_collector.increment_counter(agent_id, "requests_successful")
        else:
            self.metrics_collector.increment_counter(agent_id, "requests_failed")
        
        self.metrics_collector.record_timer(agent_id, "response_time", response_time)
    
    def update_load_metric(self, agent_id: str, load_percentage: float):
        """Update load metric for an agent"""
        self.metrics_collector.set_gauge(agent_id, "current_load", load_percentage)
    
    def update_resource_metrics(self, agent_id: str, memory_bytes: float, cpu_percent: float):
        """Update resource usage metrics"""
        self.metrics_collector.set_gauge(agent_id, "memory_usage", memory_bytes)
        self.metrics_collector.set_gauge(agent_id, "cpu_usage", cpu_percent)
    
    def get_agent_status(self, agent_id: str) -> AgentHealthStatus:
        """Get comprehensive status for an agent"""
        health_status = self.health_checker.get_health_status(agent_id)
        
        # Add metrics summary
        metrics = self.metrics_collector.get_agent_metrics(agent_id)
        metrics_summary = {}
        
        for name, metric in metrics.items():
            stats = self.metrics_collector.get_metric_statistics(agent_id, name)
            metrics_summary[name] = {
                "type": metric.metric_type.value,
                "current": stats.get("latest", 0),
                "stats": stats
            }
        
        health_status.metrics_summary = metrics_summary
        return health_status
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide monitoring overview"""
        with self._lock:
            monitored_agents = list(self._monitored_agents)
        
        overview = {
            "total_agents": len(monitored_agents),
            "healthy_agents": 0,
            "degraded_agents": 0,
            "unhealthy_agents": 0,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "agents": {}
        }
        
        for agent_id in monitored_agents:
            status = self.get_agent_status(agent_id)
            overview["agents"][agent_id] = {
                "status": status.overall_status.value,
                "checks": len(status.checks),
                "metrics": len(status.metrics_summary)
            }
            
            if status.overall_status == HealthCheckStatus.PASS:
                overview["healthy_agents"] += 1
            elif status.overall_status == HealthCheckStatus.WARN:
                overview["degraded_agents"] += 1
            else:
                overview["unhealthy_agents"] += 1
        
        return overview
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        await self.alert_manager.start_monitoring()
        logger.info("Agent monitoring system started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        
        await self.alert_manager.stop_monitoring()
        logger.info("Agent monitoring system stopped")

# Global monitoring instance
agent_monitor = AgentMonitor()

# Utility functions
async def start_agent_monitoring():
    """Start the global agent monitoring system"""
    await agent_monitor.start_monitoring()

async def stop_agent_monitoring():
    """Stop the global agent monitoring system"""
    await agent_monitor.stop_monitoring()

def register_agent_for_monitoring(agent_id: str, agent_name: str = ""):
    """Register an agent for monitoring"""
    asyncio.create_task(agent_monitor.add_agent(agent_id, agent_name))

def record_agent_request(agent_id: str, success: bool, response_time: float):
    """Record request metrics for an agent"""
    agent_monitor.record_request_metric(agent_id, success, response_time)

def update_agent_load(agent_id: str, load_percentage: float):
    """Update agent load metric"""
    agent_monitor.update_load_metric(agent_id, load_percentage)

def get_agent_health(agent_id: str) -> AgentHealthStatus:
    """Get agent health status"""
    return agent_monitor.get_agent_status(agent_id)

def get_monitoring_overview() -> Dict[str, Any]:
    """Get system monitoring overview"""
    return agent_monitor.get_system_overview()

# Monitoring decorators
def monitor_performance(agent_id: str):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                response_time = time.time() - start_time
                record_agent_request(agent_id, success, response_time)
        
        return wrapper
    return decorator

def track_metric(agent_id: str, metric_name: str, metric_type: MetricType = MetricType.COUNTER):
    """Decorator to track custom metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if metric_type == MetricType.COUNTER:
                agent_monitor.metrics_collector.increment_counter(agent_id, metric_name)
            elif metric_type == MetricType.TIMER:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    agent_monitor.metrics_collector.record_timer(agent_id, metric_name, duration)
            else:
                result = await func(*args, **kwargs)
                return result
        
        return wrapper
    return decorator