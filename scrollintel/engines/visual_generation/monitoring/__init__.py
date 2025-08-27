"""
Comprehensive monitoring and observability module for visual generation system.
Integrates with ScrollIntel monitoring infrastructure.
"""

from .metrics_collector import (
    MetricsCollector,
    PerformanceMetrics,
    Metric,
    MetricType,
    metrics_collector
)

from .distributed_tracing import (
    DistributedTracer,
    TraceCollector,
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    distributed_tracer
)

from .observability_dashboard import (
    VisualGenerationObservabilityDashboard,
    Alert,
    AlertSeverity,
    HealthCheck,
    observability_dashboard
)

__all__ = [
    # Metrics Collection
    'MetricsCollector',
    'PerformanceMetrics',
    'Metric',
    'MetricType',
    'metrics_collector',
    
    # Distributed Tracing
    'DistributedTracer',
    'TraceCollector',
    'Span',
    'SpanContext',
    'SpanKind',
    'SpanStatus',
    'distributed_tracer',
    
    # Observability Dashboard
    'VisualGenerationObservabilityDashboard',
    'Alert',
    'AlertSeverity',
    'HealthCheck',
    'observability_dashboard'
]


async def initialize_monitoring_system():
    """Initialize the complete monitoring system."""
    # Start metrics collector
    await metrics_collector.start()
    
    # Start distributed tracer
    await distributed_tracer.start()
    
    # Initialize observability dashboard with components
    observability_dashboard.metrics_collector = metrics_collector
    observability_dashboard.tracer = distributed_tracer
    await observability_dashboard.initialize()
    
    return {
        'metrics_collector': metrics_collector,
        'distributed_tracer': distributed_tracer,
        'observability_dashboard': observability_dashboard
    }


async def shutdown_monitoring_system():
    """Shutdown the complete monitoring system."""
    await observability_dashboard.shutdown()
    await distributed_tracer.stop()
    await metrics_collector.stop()