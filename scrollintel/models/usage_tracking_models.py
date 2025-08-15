"""
Usage tracking models for visual content generation billing system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid


class GenerationType(Enum):
    """Types of content generation."""
    IMAGE = "image"
    VIDEO = "video"
    ENHANCEMENT = "enhancement"
    BATCH = "batch"


class ResourceType(Enum):
    """Types of computational resources."""
    GPU_SECONDS = "gpu_seconds"
    CPU_SECONDS = "cpu_seconds"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    API_CALLS = "api_calls"


@dataclass
class ResourceUsage:
    """Individual resource usage record."""
    resource_type: ResourceType
    amount: float
    unit_cost: float
    total_cost: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationUsage:
    """Usage record for a single generation request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    generation_type: GenerationType = GenerationType.IMAGE
    model_used: str = ""
    prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Resource consumption
    resources: List[ResourceUsage] = field(default_factory=list)
    total_cost: float = 0.0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Quality metrics
    quality_score: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # Billing
    billed: bool = False
    billing_period: Optional[str] = None


@dataclass
class UserUsageSummary:
    """Summary of user usage for a time period."""
    user_id: str
    period_start: datetime
    period_end: datetime
    
    # Generation counts
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    
    # By type
    image_generations: int = 0
    video_generations: int = 0
    enhancement_operations: int = 0
    batch_operations: int = 0
    
    # Resource usage
    total_gpu_seconds: float = 0.0
    total_cpu_seconds: float = 0.0
    total_storage_gb: float = 0.0
    total_bandwidth_gb: float = 0.0
    total_api_calls: int = 0
    
    # Costs
    total_cost: float = 0.0
    average_cost_per_generation: float = 0.0
    
    # Quality metrics
    average_quality_score: Optional[float] = None
    average_generation_time: float = 0.0


@dataclass
class BudgetAlert:
    """Budget monitoring alert."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    alert_type: str = "budget_threshold"
    threshold_percentage: float = 80.0
    current_usage: float = 0.0
    budget_limit: float = 0.0
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False


@dataclass
class UsageForecast:
    """Usage forecasting data."""
    user_id: str
    forecast_period_days: int = 30
    
    # Historical data points
    historical_usage: List[float] = field(default_factory=list)
    historical_dates: List[datetime] = field(default_factory=list)
    
    # Forecast results
    predicted_usage: float = 0.0
    predicted_cost: float = 0.0
    confidence_interval: tuple = (0.0, 0.0)
    
    # Trend analysis
    usage_trend: str = "stable"  # increasing, decreasing, stable
    seasonal_pattern: bool = False
    
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    recommendation_type: str = ""
    title: str = ""
    description: str = ""
    potential_savings: float = 0.0
    implementation_effort: str = "low"  # low, medium, high
    priority: str = "medium"  # low, medium, high
    
    # Supporting data
    current_cost: float = 0.0
    optimized_cost: float = 0.0
    affected_operations: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    implemented: bool = False
    implemented_at: Optional[datetime] = None