"""
Transformation Roadmap Models

Data models for systematic transformation journey planning, milestone tracking,
and roadmap optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum


class MilestoneType(Enum):
    """Type of transformation milestone"""
    FOUNDATION = "foundation"
    AWARENESS = "awareness"
    ADOPTION = "adoption"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"
    SUSTAINABILITY = "sustainability"


class MilestoneStatus(Enum):
    """Status of milestone"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    AT_RISK = "at_risk"


class RoadmapPhase(Enum):
    """Phase of transformation roadmap"""
    PREPARATION = "preparation"
    LAUNCH = "launch"
    IMPLEMENTATION = "implementation"
    REINFORCEMENT = "reinforcement"
    EVALUATION = "evaluation"


class DependencyType(Enum):
    """Type of dependency between milestones"""
    PREREQUISITE = "prerequisite"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    OPTIONAL = "optional"


@dataclass
class TransformationMilestone:
    """Represents a transformation milestone"""
    id: str
    name: str
    description: str
    milestone_type: MilestoneType
    target_date: datetime
    estimated_duration: timedelta
    success_criteria: List[str]
    deliverables: List[str]
    responsible_parties: List[str]
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED
    progress_percentage: float = 0.0
    actual_start_date: Optional[datetime] = None
    actual_completion_date: Optional[datetime] = None
    risks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # IDs of dependent milestones
    
    def __post_init__(self):
        if not 0 <= self.progress_percentage <= 100:
            raise ValueError("Progress percentage must be between 0 and 100")


@dataclass
class MilestoneDependency:
    """Represents dependency between milestones"""
    id: str
    predecessor_id: str
    successor_id: str
    dependency_type: DependencyType
    lag_time: timedelta = timedelta(days=0)
    description: str = ""


@dataclass
class RoadmapPhaseDefinition:
    """Definition of a roadmap phase"""
    phase: RoadmapPhase
    name: str
    description: str
    objectives: List[str]
    key_activities: List[str]
    success_metrics: List[str]
    typical_duration: timedelta
    critical_success_factors: List[str]


@dataclass
class TransformationRoadmap:
    """Comprehensive transformation roadmap"""
    id: str
    organization_id: str
    vision_id: str
    name: str
    description: str
    start_date: datetime
    target_completion_date: datetime
    phases: List[RoadmapPhaseDefinition]
    milestones: List[TransformationMilestone]
    dependencies: List[MilestoneDependency]
    overall_progress: float = 0.0
    current_phase: RoadmapPhase = RoadmapPhase.PREPARATION
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not 0 <= self.overall_progress <= 100:
            raise ValueError("Overall progress must be between 0 and 100")


@dataclass
class ProgressTrackingMetric:
    """Metric for tracking transformation progress"""
    id: str
    name: str
    description: str
    measurement_method: str
    target_value: float
    current_value: float = 0.0
    unit: str = ""
    frequency: str = "monthly"
    data_source: str = ""
    
    def calculate_progress_percentage(self) -> float:
        """Calculate progress as percentage of target"""
        if self.target_value == 0:
            return 0.0
        return min(100.0, (self.current_value / self.target_value) * 100)


@dataclass
class RoadmapOptimization:
    """Optimization recommendations for roadmap"""
    roadmap_id: str
    optimization_type: str
    current_issue: str
    recommended_action: str
    expected_benefit: str
    implementation_effort: str
    priority: int
    estimated_impact: float  # 0-1 scale
    
    def __post_init__(self):
        if not 0 <= self.estimated_impact <= 1:
            raise ValueError("Estimated impact must be between 0 and 1")


@dataclass
class RoadmapPlanningRequest:
    """Request for roadmap planning"""
    organization_id: str
    vision_id: str
    current_culture_state: Dict[str, Any]
    target_culture_state: Dict[str, Any]
    available_resources: Dict[str, Any]
    constraints: List[str]
    timeline_preferences: Dict[str, Any]
    stakeholder_priorities: Dict[str, List[str]]
    risk_tolerance: float = 0.5  # 0 = risk-averse, 1 = risk-tolerant
    
    def __post_init__(self):
        if not 0 <= self.risk_tolerance <= 1:
            raise ValueError("Risk tolerance must be between 0 and 1")


@dataclass
class RoadmapPlanningResult:
    """Result of roadmap planning process"""
    roadmap: TransformationRoadmap
    critical_path: List[str]  # Milestone IDs in critical path
    resource_requirements: Dict[str, Any]
    risk_assessment: List[str]
    success_probability: float
    alternative_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0 <= self.success_probability <= 1:
            raise ValueError("Success probability must be between 0 and 1")


@dataclass
class ProgressUpdate:
    """Update on milestone or roadmap progress"""
    milestone_id: str
    update_date: datetime
    progress_percentage: float
    status: MilestoneStatus
    achievements: List[str]
    challenges: List[str]
    next_steps: List[str]
    updated_by: str
    notes: str = ""
    
    def __post_init__(self):
        if not 0 <= self.progress_percentage <= 100:
            raise ValueError("Progress percentage must be between 0 and 100")


@dataclass
class RoadmapAdjustment:
    """Adjustment to roadmap based on progress and feedback"""
    roadmap_id: str
    adjustment_type: str
    reason: str
    changes_made: List[str]
    impact_assessment: str
    approval_required: bool
    approved_by: Optional[str] = None
    adjustment_date: datetime = field(default_factory=datetime.now)


@dataclass
class TransformationScenario:
    """Alternative transformation scenario"""
    id: str
    name: str
    description: str
    duration: timedelta
    resource_requirements: Dict[str, Any]
    success_probability: float
    key_differences: List[str]
    trade_offs: List[str]
    
    def __post_init__(self):
        if not 0 <= self.success_probability <= 1:
            raise ValueError("Success probability must be between 0 and 1")


@dataclass
class Transformation:
    """Represents a cultural transformation initiative"""
    id: str
    organization_id: str
    name: str
    description: str
    current_culture: Dict[str, Any]
    target_culture: Dict[str, Any]
    roadmap: Optional[TransformationRoadmap] = None
    status: str = "planning"
    progress: float = 0.0
    start_date: Optional[datetime] = None
    target_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not 0 <= self.progress <= 100:
            raise ValueError("Progress must be between 0 and 100")