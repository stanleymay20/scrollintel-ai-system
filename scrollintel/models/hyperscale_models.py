"""
Hyperscale Infrastructure Management Models

Data models for managing billion-user scale infrastructure across multiple cloud regions.
Supports real-time auto-scaling, performance optimization, and cost management.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid


class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    ORACLE = "oracle"


class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    CDN = "cdn"


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    OUT = "out"
    IN = "in"


@dataclass
class RegionalMetrics:
    """Metrics for a specific cloud region"""
    region: str
    provider: CloudProvider
    active_users: int
    requests_per_second: int
    cpu_utilization: float
    memory_utilization: float
    network_throughput: float
    storage_iops: int
    latency_p95: float
    error_rate: float
    cost_per_hour: float
    timestamp: datetime


@dataclass
class HyperscaleMetrics:
    """Global hyperscale infrastructure metrics"""
    id: str
    timestamp: datetime
    global_requests_per_second: int
    active_users: int
    total_servers: int
    total_data_centers: int
    infrastructure_utilization: Dict[str, float]
    performance_metrics: Dict[str, float]
    cost_metrics: Dict[str, float]
    regional_distribution: Dict[str, RegionalMetrics]
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class GlobalInfrastructure:
    """Global infrastructure configuration and state"""
    id: str
    name: str
    total_capacity: Dict[ResourceType, int]
    current_utilization: Dict[ResourceType, float]
    regions: List[str]
    providers: List[CloudProvider]
    auto_scaling_enabled: bool
    cost_optimization_enabled: bool
    performance_targets: Dict[str, float]
    cost_targets: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ScalingEvent:
    """Auto-scaling event record"""
    id: str
    timestamp: datetime
    region: str
    resource_type: ResourceType
    direction: ScalingDirection
    scale_factor: float
    trigger_metric: str
    trigger_value: float
    threshold: float
    instances_before: int
    instances_after: int
    cost_impact: float
    performance_impact: Dict[str, float]
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class PerformanceOptimization:
    """Performance optimization recommendation"""
    id: str
    timestamp: datetime
    region: str
    optimization_type: str
    current_performance: Dict[str, float]
    target_performance: Dict[str, float]
    recommended_actions: List[str]
    estimated_improvement: Dict[str, float]
    implementation_cost: float
    priority: int
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class CostOptimization:
    """Cost optimization recommendation"""
    id: str
    timestamp: datetime
    optimization_category: str
    current_cost: float
    optimized_cost: float
    savings_potential: float
    savings_percentage: float
    recommended_actions: List[str]
    risk_assessment: str
    implementation_effort: str
    payback_period_days: int
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class InfrastructureAlert:
    """Infrastructure monitoring alert"""
    id: str
    timestamp: datetime
    severity: str
    alert_type: str
    region: str
    resource_type: ResourceType
    metric_name: str
    current_value: float
    threshold: float
    description: str
    recommended_actions: List[str]
    auto_resolved: bool
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class CapacityPlan:
    """Billion-user capacity planning"""
    id: str
    target_users: int
    target_rps: int
    regions: List[str]
    resource_requirements: Dict[ResourceType, int]
    estimated_cost: float
    implementation_timeline: Dict[str, datetime]
    risk_factors: List[str]
    contingency_plans: List[str]
    created_at: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class GlobalLoadBalancingConfig:
    """Global load balancing configuration"""
    id: str
    algorithm: str
    health_check_interval: int
    failover_threshold: float
    traffic_distribution: Dict[str, float]
    geo_routing_enabled: bool
    latency_based_routing: bool
    cost_based_routing: bool
    updated_at: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())