"""
Innovation Pipeline Management Models

This module defines the data models for managing innovation pipelines,
including pipeline stages, optimization metrics, and resource allocation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import uuid4


class PipelineStage(Enum):
    """Innovation pipeline stages"""
    IDEATION = "ideation"
    RESEARCH = "research"
    EXPERIMENTATION = "experimentation"
    PROTOTYPING = "prototyping"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class InnovationPriority(Enum):
    """Innovation priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResourceType(Enum):
    """Types of resources for innovation development"""
    COMPUTE = "compute"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    RESEARCH_TIME = "research_time"
    DEVELOPMENT_TIME = "development_time"
    TESTING_TIME = "testing_time"
    BUDGET = "budget"


class PipelineStatus(Enum):
    """Pipeline status indicators"""
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: ResourceType
    amount: float
    unit: str
    duration: Optional[float] = None
    priority: InnovationPriority = InnovationPriority.MEDIUM


@dataclass
class ResourceAllocation:
    """Resource allocation tracking"""
    id: str = field(default_factory=lambda: str(uuid4()))
    innovation_id: str = ""
    resource_type: ResourceType = ResourceType.COMPUTE
    allocated_amount: float = 0.0
    used_amount: float = 0.0
    allocation_time: datetime = field(default_factory=datetime.utcnow)
    expected_completion: Optional[datetime] = None
    efficiency_score: float = 0.0


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    stage: PipelineStage
    throughput: float  # innovations per time unit
    cycle_time: float  # average time in stage
    success_rate: float  # percentage of successful transitions
    resource_utilization: float  # percentage of resources used
    bottleneck_score: float  # bottleneck severity (0-1)
    quality_score: float  # output quality metric
    cost_efficiency: float  # value per resource unit


@dataclass
class InnovationPipelineItem:
    """Individual innovation in the pipeline"""
    id: str = field(default_factory=lambda: str(uuid4()))
    innovation_id: str = ""
    current_stage: PipelineStage = PipelineStage.IDEATION
    priority: InnovationPriority = InnovationPriority.MEDIUM
    status: PipelineStatus = PipelineStatus.ACTIVE
    
    # Timeline tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    stage_entered_at: datetime = field(default_factory=datetime.utcnow)
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    
    # Resource tracking
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    resource_allocations: List[ResourceAllocation] = field(default_factory=list)
    
    # Performance tracking
    stage_metrics: Dict[PipelineStage, PipelineMetrics] = field(default_factory=dict)
    success_probability: float = 0.0
    risk_score: float = 0.0
    impact_score: float = 0.0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Other innovation IDs
    blocking_issues: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineOptimizationConfig:
    """Configuration for pipeline optimization"""
    max_concurrent_innovations: int = 100
    resource_buffer_percentage: float = 0.2
    priority_weight_critical: float = 1.0
    priority_weight_high: float = 0.8
    priority_weight_medium: float = 0.6
    priority_weight_low: float = 0.4
    
    # Optimization objectives
    maximize_throughput: bool = True
    minimize_cycle_time: bool = True
    maximize_success_rate: bool = True
    minimize_resource_waste: bool = True
    
    # Thresholds
    bottleneck_threshold: float = 0.7
    resource_utilization_target: float = 0.85
    quality_threshold: float = 0.8
    
    # Rebalancing settings
    rebalance_frequency_minutes: int = 30
    emergency_rebalance_threshold: float = 0.9


@dataclass
class PipelineOptimizationResult:
    """Result of pipeline optimization"""
    optimization_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Optimization actions
    resource_reallocations: List[ResourceAllocation] = field(default_factory=list)
    priority_adjustments: Dict[str, InnovationPriority] = field(default_factory=dict)
    stage_transitions: Dict[str, PipelineStage] = field(default_factory=dict)
    
    # Performance improvements
    expected_throughput_improvement: float = 0.0
    expected_cycle_time_reduction: float = 0.0
    expected_resource_savings: float = 0.0
    
    # Metrics
    optimization_score: float = 0.0
    confidence_level: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelinePerformanceReport:
    """Comprehensive pipeline performance report"""
    report_id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = field(default_factory=datetime.utcnow)
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    
    # Overall metrics
    total_innovations: int = 0
    completed_innovations: int = 0
    failed_innovations: int = 0
    active_innovations: int = 0
    
    # Stage-wise metrics
    stage_metrics: Dict[PipelineStage, PipelineMetrics] = field(default_factory=dict)
    
    # Resource metrics
    total_resources_allocated: Dict[ResourceType, float] = field(default_factory=dict)
    total_resources_used: Dict[ResourceType, float] = field(default_factory=dict)
    resource_efficiency: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Performance indicators
    overall_throughput: float = 0.0
    average_cycle_time: float = 0.0
    overall_success_rate: float = 0.0
    cost_per_innovation: float = 0.0
    
    # Bottleneck analysis
    identified_bottlenecks: List[PipelineStage] = field(default_factory=list)
    bottleneck_severity: Dict[PipelineStage, float] = field(default_factory=dict)
    
    # Trends
    throughput_trend: List[float] = field(default_factory=list)
    quality_trend: List[float] = field(default_factory=list)
    efficiency_trend: List[float] = field(default_factory=list)
    
    # Recommendations
    optimization_recommendations: List[str] = field(default_factory=list)
    capacity_recommendations: List[str] = field(default_factory=list)
    process_improvements: List[str] = field(default_factory=list)