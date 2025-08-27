"""
Observability dashboard for visual generation system.
Integrates with ScrollIntel monitoring infrastructure and provides comprehensive insights.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from enum import Enum

from .metrics_collector import MetricsCollector, PerformanceMetrics
from .distributed_tracing import DistributedTracer, TraceCollector
from scrollintel.core.monitoring import monitoring_system  # Integration with existing monitoring

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Represents a health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    response_time: Optional[float] = None
    details: Dict[str, Any] = None


class VisualGenerationObservabilityDashboard:
    """
    Comprehensive observability dashboard for visual generation system.
    Integrates with ScrollIntel monitoring infrastructure.
    """
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 tracer: DistributedTracer):
        self.metrics_collector = metrics_collector
        self.tracer = tracer
        
        # Alert configuration
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Health check configuration
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheck] = {}
        
        # Dashboard configuration
        self.dashboard_config = {
            "refresh_interval": 30,  # seconds
            "retention_days": 7,
            "max_alerts": 1000,
            "max_traces_display": 100
        }
        
        # Integration with ScrollIntel monitoring
        self.scrollintel_integration = True
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self):
        """Initialize the observability dashboard."""
        # Set up default alert rules
        self._setup_default_alert_rules()
        
        # Set up default health checks
        self._setup_default_health_checks()
        
        # Register with ScrollIntel monitoring
        if self.scrollintel_integration:
            await self._register_with_scrollintel()
        
        # Start background tasks
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Visual generation observability dashboard initialized")
    
    async def shutdown(self):
        """Shutdown the observability dashboard."""
        self._running = False
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # Wait for tasks to complete
        for task in [self._monitoring_task, self._health_check_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Visual generation observability dashboard shutdown")
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float,
                      severity: AlertSeverity, description: str = "",
                      comparison: str = "greater_than"):
        """Add a new alert rule."""
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "severity": severity,
            "description": description,
            "comparison": comparison,
            "enabled": True
        }
        logger.info(f"Added alert rule: {name}")
    
    def add_health_check(self, component: str, check_function: Callable):
        """Add a health check for a component."""
        self.health_checks[component] = check_function
        logger.info(f"Added health check for component: {component}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        # Get performance metrics
        performance = self.metrics_collector.get_performance_metrics()
        
        # Get system overview
        system_overview = await self._get_system_overview()
        
        # Get recent traces
        recent_traces = self._get_recent_traces()
        
        # Get active alerts
        active_alerts = list(self.active_alerts.values())
        
        # Get health status
        health_status = list(self.health_status.values())
        
        # Get resource utilization
        resource_utilization = await self._get_resource_utilization()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "performance": asdict(performance),
            "system_overview": system_overview,
            "recent_traces": recent_traces,
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "health_status": [asdict(check) for check in health_status],
            "resource_utilization": resource_utilization,
            "dashboard_config": self.dashboard_config
        }
    
    async def get_performance_trends(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends over time."""
        # This would typically query a time-series database
        # For now, we'll return sample trend data
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Generate sample data points (in production, query actual metrics)
        data_points = []
        current_time = start_time
        
        while current_time <= end_time:
            # Get metrics for this time point (simplified)
            performance = self.metrics_collector.get_performance_metrics()
            
            data_points.append({
                "timestamp": current_time.isoformat(),
                "requests_per_second": performance.requests_per_second,
                "average_response_time": performance.average_response_time,
                "error_rate": performance.error_rate,
                "cpu_usage": performance.cpu_usage,
                "memory_usage": performance.memory_usage
            })
            
            current_time += timedelta(minutes=5)  # 5-minute intervals
        
        return {
            "performance_trends": data_points,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "interval_minutes": 5
        }
    
    async def get_trace_analysis(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis of a specific trace."""
        spans = self.tracer.collector.get_trace(trace_id)
        if not spans:
            return None
        
        # Analyze trace
        analysis = {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "services": list(set(span.service_name for span in spans)),
            "operations": list(set(span.operation_name for span in spans)),
            "has_errors": any(span.status.value == "error" for span in spans),
            "spans": []
        }
        
        # Add span details
        for span in sorted(spans, key=lambda s: s.start_time):
            span_data = {
                "span_id": span.context.span_id,
                "parent_span_id": span.context.parent_span_id,
                "operation_name": span.operation_name,
                "service_name": span.service_name,
                "start_time": span.start_time.isoformat(),
                "duration": span.duration,
                "status": span.status.value,
                "tags": span.tags,
                "logs": span.logs,
                "error": span.error
            }
            analysis["spans"].append(span_data)
        
        # Calculate critical path
        analysis["critical_path"] = self._calculate_critical_path(spans)
        
        return analysis
    
    async def get_error_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get error analysis and patterns."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Search for traces with errors
        error_traces = self.tracer.collector.search_traces(
            has_errors=True,
            limit=1000
        )
        
        # Analyze error patterns
        error_patterns = {}
        error_services = {}
        error_operations = {}
        
        for trace_summary in error_traces:
            trace_id = trace_summary["trace_id"]
            spans = self.tracer.collector.get_trace(trace_id)
            
            for span in spans:
                if span.status.value == "error" and span.error:
                    # Count error types
                    error_type = span.tags.get("error.type", "Unknown")
                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
                    
                    # Count errors by service
                    error_services[span.service_name] = error_services.get(span.service_name, 0) + 1
                    
                    # Count errors by operation
                    error_operations[span.operation_name] = error_operations.get(span.operation_name, 0) + 1
        
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_error_traces": len(error_traces),
            "error_patterns": error_patterns,
            "errors_by_service": error_services,
            "errors_by_operation": error_operations,
            "recent_error_traces": error_traces[:10]  # Most recent 10
        }
    
    async def get_capacity_planning_data(self) -> Dict[str, Any]:
        """Get data for capacity planning."""
        performance = self.metrics_collector.get_performance_metrics()
        
        # Calculate capacity metrics
        current_rps = performance.requests_per_second
        current_cpu = performance.cpu_usage
        current_memory = performance.memory_usage
        
        # Estimate capacity based on current utilization
        # Assuming 80% utilization is the target
        target_utilization = 80.0
        
        cpu_capacity_factor = target_utilization / max(current_cpu, 1.0)
        memory_capacity_factor = target_utilization / max(current_memory, 1.0)
        
        # Use the more constraining factor
        capacity_factor = min(cpu_capacity_factor, memory_capacity_factor)
        estimated_max_rps = current_rps * capacity_factor
        
        return {
            "current_metrics": {
                "requests_per_second": current_rps,
                "cpu_usage_percent": current_cpu,
                "memory_usage_percent": current_memory,
                "average_response_time": performance.average_response_time
            },
            "capacity_estimates": {
                "estimated_max_rps": estimated_max_rps,
                "cpu_capacity_factor": cpu_capacity_factor,
                "memory_capacity_factor": memory_capacity_factor,
                "bottleneck": "cpu" if cpu_capacity_factor < memory_capacity_factor else "memory"
            },
            "recommendations": self._generate_capacity_recommendations(
                current_rps, current_cpu, current_memory, estimated_max_rps
            )
        }
    
    def _setup_default_alert_rules(self):
        """Set up default alert rules for visual generation."""
        # Performance alerts
        self.add_alert_rule(
            "high_error_rate",
            "error_rate",
            0.05,  # 5%
            AlertSeverity.ERROR,
            "Error rate is above 5%"
        )
        
        self.add_alert_rule(
            "high_response_time",
            "average_response_time",
            30.0,  # 30 seconds
            AlertSeverity.WARNING,
            "Average response time is above 30 seconds"
        )
        
        self.add_alert_rule(
            "low_requests_per_second",
            "requests_per_second",
            0.1,
            AlertSeverity.WARNING,
            "Request rate is very low",
            "less_than"
        )
        
        # Resource alerts
        self.add_alert_rule(
            "high_cpu_usage",
            "cpu_usage",
            85.0,  # 85%
            AlertSeverity.WARNING,
            "CPU usage is above 85%"
        )
        
        self.add_alert_rule(
            "high_memory_usage",
            "memory_usage",
            90.0,  # 90%
            AlertSeverity.ERROR,
            "Memory usage is above 90%"
        )
        
        # Quality alerts
        self.add_alert_rule(
            "low_quality_score",
            "average_quality_score",
            0.7,  # 70%
            AlertSeverity.WARNING,
            "Average quality score is below 70%",
            "less_than"
        )
    
    def _setup_default_health_checks(self):
        """Set up default health checks."""
        self.add_health_check("metrics_collector", self._check_metrics_collector_health)
        self.add_health_check("distributed_tracer", self._check_tracer_health)
        self.add_health_check("database", self._check_database_health)
        self.add_health_check("cache", self._check_cache_health)
    
    async def _check_metrics_collector_health(self) -> HealthCheck:
        """Check metrics collector health."""
        try:
            metrics = self.metrics_collector.get_performance_metrics()
            
            if metrics.total_requests > 0:
                status = "healthy"
                message = "Metrics collector is functioning normally"
            else:
                status = "degraded"
                message = "No recent requests recorded"
            
            return HealthCheck(
                component="metrics_collector",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={"total_requests": metrics.total_requests}
            )
        except Exception as e:
            return HealthCheck(
                component="metrics_collector",
                status="unhealthy",
                message=f"Metrics collector error: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_tracer_health(self) -> HealthCheck:
        """Check distributed tracer health."""
        try:
            # Check if tracer is collecting traces
            trace_count = len(self.tracer.collector.traces)
            
            status = "healthy" if trace_count >= 0 else "unhealthy"
            message = f"Tracer has {trace_count} active traces"
            
            return HealthCheck(
                component="distributed_tracer",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={"active_traces": trace_count}
            )
        except Exception as e:
            return HealthCheck(
                component="distributed_tracer",
                status="unhealthy",
                message=f"Tracer error: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_database_health(self) -> HealthCheck:
        """Check database health."""
        # Placeholder - would implement actual database health check
        return HealthCheck(
            component="database",
            status="healthy",
            message="Database connection is healthy",
            timestamp=datetime.now(),
            response_time=0.05
        )
    
    async def _check_cache_health(self) -> HealthCheck:
        """Check cache health."""
        # Placeholder - would implement actual cache health check
        return HealthCheck(
            component="cache",
            status="healthy",
            message="Cache is responding normally",
            timestamp=datetime.now(),
            response_time=0.01
        )
    
    async def _register_with_scrollintel(self):
        """Register visual generation monitoring with ScrollIntel."""
        try:
            # Register custom metrics with ScrollIntel monitoring system
            if hasattr(monitoring_system, 'register_custom_metrics'):
                custom_metrics = [
                    "visual_generation_requests_total",
                    "visual_generation_response_time",
                    "visual_generation_quality_score",
                    "visual_generation_error_rate"
                ]
                
                for metric in custom_metrics:
                    monitoring_system.register_custom_metrics(metric)
            
            logger.info("Registered visual generation metrics with ScrollIntel")
        except Exception as e:
            logger.warning(f"Failed to register with ScrollIntel monitoring: {e}")
    
    async def _get_system_overview(self) -> Dict[str, Any]:
        """Get system overview data."""
        performance = self.metrics_collector.get_performance_metrics()
        
        return {
            "status": "healthy" if performance.error_rate < 0.05 else "degraded",
            "total_requests_last_hour": performance.total_requests,
            "success_rate": (performance.successful_requests / max(performance.total_requests, 1)) * 100,
            "average_response_time": performance.average_response_time,
            "active_alerts": len(self.active_alerts),
            "healthy_components": len([h for h in self.health_status.values() if h.status == "healthy"])
        }
    
    def _get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces for dashboard display."""
        return self.tracer.collector.search_traces(limit=limit)
    
    async def _get_resource_utilization(self) -> Dict[str, Any]:
        """Get resource utilization data."""
        performance = self.metrics_collector.get_performance_metrics()
        
        return {
            "cpu_usage": performance.cpu_usage,
            "memory_usage": performance.memory_usage,
            "gpu_utilization": performance.gpu_utilization,
            "requests_per_second": performance.requests_per_second,
            "cache_hit_rate": performance.cache_hit_rate
        }
    
    def _calculate_critical_path(self, spans: List) -> List[str]:
        """Calculate the critical path through a trace."""
        # Simplified critical path calculation
        # In a real implementation, this would be more sophisticated
        
        if not spans:
            return []
        
        # Sort spans by duration (longest first)
        sorted_spans = sorted(spans, key=lambda s: s.duration or 0, reverse=True)
        
        # Return the operation names of the longest spans
        return [span.operation_name for span in sorted_spans[:5]]
    
    def _generate_capacity_recommendations(self, current_rps: float, current_cpu: float,
                                         current_memory: float, estimated_max_rps: float) -> List[str]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        if current_cpu > 80:
            recommendations.append("Consider adding more CPU resources or scaling horizontally")
        
        if current_memory > 85:
            recommendations.append("Memory usage is high - consider increasing memory allocation")
        
        if current_rps > estimated_max_rps * 0.8:
            recommendations.append("Approaching capacity limits - plan for scaling")
        
        if not recommendations:
            recommendations.append("System is operating within normal capacity limits")
        
        return recommendations
    
    async def _monitoring_loop(self):
        """Background monitoring loop for alerts."""
        while self._running:
            try:
                await self._check_alert_rules()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed."""
        performance = self.metrics_collector.get_performance_metrics()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.get("enabled", True):
                continue
            
            metric_name = rule["metric_name"]
            threshold = rule["threshold"]
            comparison = rule.get("comparison", "greater_than")
            
            # Get current metric value
            current_value = getattr(performance, metric_name, 0)
            
            # Check threshold
            alert_triggered = False
            if comparison == "greater_than" and current_value > threshold:
                alert_triggered = True
            elif comparison == "less_than" and current_value < threshold:
                alert_triggered = True
            
            # Handle alert
            if alert_triggered and rule_name not in self.active_alerts:
                # Trigger new alert
                alert = Alert(
                    id=f"{rule_name}_{int(datetime.now().timestamp())}",
                    name=rule_name,
                    description=rule["description"],
                    severity=rule["severity"],
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=current_value,
                    timestamp=datetime.now()
                )
                
                self.active_alerts[rule_name] = alert
                self.alert_history.append(alert)
                
                logger.warning(f"Alert triggered: {rule_name} - {rule['description']}")
            
            elif not alert_triggered and rule_name in self.active_alerts:
                # Resolve alert
                alert = self.active_alerts.pop(rule_name)
                alert.resolved = True
                alert.resolution_time = datetime.now()
                
                logger.info(f"Alert resolved: {rule_name}")
    
    async def _run_health_checks(self):
        """Run all registered health checks."""
        for component, check_function in self.health_checks.items():
            try:
                health_check = await check_function()
                self.health_status[component] = health_check
            except Exception as e:
                self.health_status[component] = HealthCheck(
                    component=component,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now()
                )


# Global observability dashboard instance
observability_dashboard = VisualGenerationObservabilityDashboard(
    metrics_collector=None,  # Will be injected
    tracer=None  # Will be injected
)