"""
Hyperscale Monitoring Models for Big Tech CTO Capabilities

This module defines data models for monitoring billion-user systems,
global infrastructure performance, and executive-level metrics.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(Enum):
    """Severity levels for incidents and alerts"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SystemStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


class IncidentStatus(Enum):
    """Incident lifecycle status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"


@dataclass
class GlobalMetrics:
    """Global system metrics for hyperscale monitoring"""
    timestamp: datetime
    total_requests_per_second: int
    active_users: int
    global_latency_p99: float
    global_latency_p95: float
    global_latency_p50: float
    error_rate: float
    availability: float
    throughput: int
    cpu_utilization: float
    memory_utilization: float
    disk_utilization: float
    network_utilization: float


@dataclass
class RegionalMetrics:
    """Regional infrastructure metrics"""
    region: str
    timestamp: datetime
    requests_per_second: int
    active_users: int
    latency_p99: float
    latency_p95: float
    error_rate: float
    availability: float
    server_count: int
    load_balancer_health: float
    database_connections: int
    cache_hit_rate: float


@dataclass
class ServiceMetrics:
    """Individual service metrics"""
    service_name: str
    timestamp: datetime
    requests_per_second: int
    error_rate: float
    latency_p99: float
    cpu_usage: float
    memory_usage: float
    instance_count: int
    health_score: float


@dataclass
class PredictiveAlert:
    """Predictive analytics alert"""
    id: str
    timestamp: datetime
    alert_type: str
    severity: SeverityLevel
    predicted_failure_time: datetime
    confidence: float
    affected_systems: List[str]
    recommended_actions: List[str]
    description: str


@dataclass
class SystemIncident:
    """System incident tracking"""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    affected_services: List[str]
    affected_regions: List[str]
    impact_assessment: str
    root_cause: Optional[str]
    resolution_steps: List[str]
    estimated_users_affected: int


@dataclass
class AutomatedResponse:
    """Automated incident response action"""
    id: str
    incident_id: str
    action_type: str
    timestamp: datetime
    status: str
    description: str
    parameters: Dict[str, Any]
    success: bool
    error_message: Optional[str]


@dataclass
class ExecutiveDashboardMetrics:
    """Executive-level dashboard metrics"""
    timestamp: datetime
    global_system_health: SystemStatus
    total_active_users: int
    revenue_impact: float
    customer_satisfaction_score: float
    system_availability: float
    performance_score: float
    security_incidents: int
    cost_efficiency: float
    innovation_velocity: float
    competitive_advantage_score: float


@dataclass
class CapacityForecast:
    """Capacity planning forecast"""
    timestamp: datetime
    forecast_horizon_days: int
    predicted_user_growth: float
    predicted_traffic_growth: float
    required_server_capacity: int
    estimated_cost: float
    scaling_recommendations: List[str]
    risk_factors: List[str]


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    service_name: str
    metric_name: str
    baseline_value: float
    acceptable_deviation: float
    critical_threshold: float
    measurement_period: str
    last_updated: datetime


@dataclass
class GlobalInfrastructureHealth:
    """Overall global infrastructure health"""
    timestamp: datetime
    overall_health_score: float
    regional_health: Dict[str, float]
    service_health: Dict[str, float]
    critical_alerts: int
    active_incidents: int
    system_capacity_utilization: float
    predicted_issues: List[PredictiveAlert]


@dataclass
class BusinessImpactMetrics:
    """Business impact metrics for executives"""
    timestamp: datetime
    revenue_per_minute: float
    customer_acquisition_rate: float
    customer_retention_rate: float
    system_downtime_cost: float
    performance_impact_on_revenue: float
    competitive_performance_gap: float
    innovation_delivery_rate: float


@dataclass
class ScalingEvent:
    """Auto-scaling event tracking"""
    id: str
    timestamp: datetime
    trigger_type: str
    service_name: str
    region: str
    scale_direction: str  # up/down
    instances_before: int
    instances_after: int
    trigger_metric: str
    trigger_value: float
    threshold: float
    duration_seconds: int
    success: bool


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    description: str
    metric_name: str
    condition: str
    threshold: float
    severity: SeverityLevel
    enabled: bool
    notification_channels: List[str]
    auto_resolve: bool
    escalation_policy: str


@dataclass
class MonitoringDashboard:
    """Dashboard configuration"""
    id: str
    name: str
    description: str
    dashboard_type: str  # executive, operational, technical
    widgets: List[Dict[str, Any]]
    refresh_interval: int
    access_permissions: List[str]
    created_by: str
    created_at: datetime