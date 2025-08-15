"""Data models for platform monitoring and metrics."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics."""
    SYSTEM = "system"
    PLATFORM = "platform"
    PERFORMANCE = "performance"
    BUSINESS = "business"


class HealthStatus(Enum):
    """Platform health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    unit: str
    aggregation_method: str  # avg, sum, max, min, count
    collection_interval_seconds: int = 60
    retention_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'unit': self.unit,
            'aggregation_method': self.aggregation_method,
            'collection_interval_seconds': self.collection_interval_seconds,
            'retention_days': self.retention_days,
            'alert_thresholds': self.alert_thresholds
        }


@dataclass
class MetricValue:
    """A single metric value."""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class Alert:
    """System alert."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = ""
    description: str = ""
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.resolved_at is None
    
    @property
    def is_acknowledged(self) -> bool:
        """Check if alert has been acknowledged."""
        return self.acknowledged_at is not None
    
    def acknowledge(self, user: str):
        """Acknowledge the alert."""
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = user
    
    def resolve(self):
        """Resolve the alert."""
        self.resolved_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'metric_name': self.metric_name,
            'threshold_value': self.threshold_value,
            'actual_value': self.actual_value,
            'created_at': self.created_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'is_active': self.is_active,
            'is_acknowledged': self.is_acknowledged,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component,
            'status': self.status.value,
            'message': self.message,
            'response_time_ms': self.response_time_ms,
            'last_check': self.last_check.isoformat(),
            'details': self.details
        }


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    benchmark_name: str
    operation_type: str
    data_size_mb: float
    duration_seconds: float
    throughput_ops_per_sec: float
    memory_peak_mb: float
    cpu_avg_percent: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'benchmark_name': self.benchmark_name,
            'operation_type': self.operation_type,
            'data_size_mb': self.data_size_mb,
            'duration_seconds': self.duration_seconds,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'memory_peak_mb': self.memory_peak_mb,
            'cpu_avg_percent': self.cpu_avg_percent,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class MonitoringDashboard:
    """Monitoring dashboard configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metrics: List[str] = field(default_factory=list)
    refresh_interval_seconds: int = 60
    time_range_hours: int = 24
    layout: Dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'metrics': self.metrics,
            'refresh_interval_seconds': self.refresh_interval_seconds,
            'time_range_hours': self.time_range_hours,
            'layout': self.layout,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class MonitoringReport:
    """Comprehensive monitoring report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = "daily"  # daily, weekly, monthly
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Summary metrics
    total_datasets_processed: int = 0
    total_data_volume_gb: float = 0.0
    avg_processing_time_seconds: float = 0.0
    success_rate_percent: float = 0.0
    error_count: int = 0
    
    # Performance metrics
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    peak_memory_usage_gb: float = 0.0
    avg_response_time_ms: float = 0.0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    quality_issues_detected: int = 0
    bias_violations_detected: int = 0
    compliance_violations: int = 0
    
    # Alerts and incidents
    total_alerts: int = 0
    critical_alerts: int = 0
    avg_resolution_time_minutes: float = 0.0
    
    # Recommendations
    optimization_recommendations: List[str] = field(default_factory=list)
    capacity_recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'report_type': self.report_type,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'total_datasets_processed': self.total_datasets_processed,
            'total_data_volume_gb': self.total_data_volume_gb,
            'avg_processing_time_seconds': self.avg_processing_time_seconds,
            'success_rate_percent': self.success_rate_percent,
            'error_count': self.error_count,
            'avg_cpu_utilization': self.avg_cpu_utilization,
            'avg_memory_utilization': self.avg_memory_utilization,
            'peak_memory_usage_gb': self.peak_memory_usage_gb,
            'avg_response_time_ms': self.avg_response_time_ms,
            'avg_quality_score': self.avg_quality_score,
            'quality_issues_detected': self.quality_issues_detected,
            'bias_violations_detected': self.bias_violations_detected,
            'compliance_violations': self.compliance_violations,
            'total_alerts': self.total_alerts,
            'critical_alerts': self.critical_alerts,
            'avg_resolution_time_minutes': self.avg_resolution_time_minutes,
            'optimization_recommendations': self.optimization_recommendations,
            'capacity_recommendations': self.capacity_recommendations
        }


@dataclass
class CapacityPlan:
    """Capacity planning recommendation."""
    component: str  # cpu, memory, storage, network
    current_utilization: float
    projected_utilization: float
    time_horizon_days: int
    recommendation: str
    confidence_level: float
    estimated_cost_impact: Optional[str] = None
    implementation_timeline: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component,
            'current_utilization': self.current_utilization,
            'projected_utilization': self.projected_utilization,
            'time_horizon_days': self.time_horizon_days,
            'recommendation': self.recommendation,
            'confidence_level': self.confidence_level,
            'estimated_cost_impact': self.estimated_cost_impact,
            'implementation_timeline': self.implementation_timeline,
            'created_at': self.created_at.isoformat()
        }


# Predefined metric definitions
SYSTEM_METRICS = [
    MetricDefinition(
        name="cpu_utilization_percent",
        type=MetricType.SYSTEM,
        description="CPU utilization percentage",
        unit="percent",
        aggregation_method="avg",
        alert_thresholds={"warning": 80.0, "critical": 90.0}
    ),
    MetricDefinition(
        name="memory_utilization_percent",
        type=MetricType.SYSTEM,
        description="Memory utilization percentage",
        unit="percent",
        aggregation_method="avg",
        alert_thresholds={"warning": 85.0, "critical": 95.0}
    ),
    MetricDefinition(
        name="disk_utilization_percent",
        type=MetricType.SYSTEM,
        description="Disk utilization percentage",
        unit="percent",
        aggregation_method="avg",
        alert_thresholds={"warning": 85.0, "critical": 95.0}
    ),
    MetricDefinition(
        name="network_throughput_mbps",
        type=MetricType.SYSTEM,
        description="Network throughput in Mbps",
        unit="mbps",
        aggregation_method="avg"
    )
]

PLATFORM_METRICS = [
    MetricDefinition(
        name="datasets_processed_per_hour",
        type=MetricType.PLATFORM,
        description="Number of datasets processed per hour",
        unit="count",
        aggregation_method="sum"
    ),
    MetricDefinition(
        name="avg_processing_time_seconds",
        type=MetricType.PLATFORM,
        description="Average dataset processing time",
        unit="seconds",
        aggregation_method="avg",
        alert_thresholds={"warning": 300.0, "critical": 600.0}
    ),
    MetricDefinition(
        name="error_rate_percent",
        type=MetricType.PLATFORM,
        description="Processing error rate percentage",
        unit="percent",
        aggregation_method="avg",
        alert_thresholds={"warning": 5.0, "critical": 10.0}
    ),
    MetricDefinition(
        name="api_response_time_ms",
        type=MetricType.PLATFORM,
        description="API response time in milliseconds",
        unit="milliseconds",
        aggregation_method="avg",
        alert_thresholds={"warning": 1000.0, "critical": 2000.0}
    ),
    MetricDefinition(
        name="quality_score_avg",
        type=MetricType.PLATFORM,
        description="Average data quality score",
        unit="score",
        aggregation_method="avg",
        alert_thresholds={"warning": 0.7, "critical": 0.5}
    )
]

PERFORMANCE_METRICS = [
    MetricDefinition(
        name="throughput_ops_per_second",
        type=MetricType.PERFORMANCE,
        description="Operations processed per second",
        unit="ops/sec",
        aggregation_method="avg"
    ),
    MetricDefinition(
        name="concurrent_operations",
        type=MetricType.PERFORMANCE,
        description="Number of concurrent operations",
        unit="count",
        aggregation_method="avg"
    ),
    MetricDefinition(
        name="queue_depth",
        type=MetricType.PERFORMANCE,
        description="Processing queue depth",
        unit="count",
        aggregation_method="avg",
        alert_thresholds={"warning": 100.0, "critical": 500.0}
    )
]

# All predefined metrics
ALL_METRICS = SYSTEM_METRICS + PLATFORM_METRICS + PERFORMANCE_METRICS