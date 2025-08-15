"""
Progress Tracking Models for Cultural Transformation Leadership

This module defines data models for tracking cultural transformation progress,
milestones, and reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ProgressStatus(Enum):
    """Status of progress tracking items"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    DELAYED = "delayed"


class MilestoneType(Enum):
    """Types of transformation milestones"""
    ASSESSMENT = "assessment"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"


@dataclass
class ProgressMetric:
    """Individual progress metric"""
    id: str
    name: str
    description: str
    current_value: float
    target_value: float
    unit: str
    category: str
    last_updated: datetime
    trend: str  # "improving", "declining", "stable"
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.target_value == 0:
            return 100.0 if self.current_value > 0 else 0.0
        return min(100.0, (self.current_value / self.target_value) * 100)


@dataclass
class TransformationMilestone:
    """Cultural transformation milestone"""
    id: str
    transformation_id: str
    name: str
    description: str
    milestone_type: MilestoneType
    target_date: datetime
    completion_date: Optional[datetime] = None
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    success_criteria: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    
    @property
    def is_overdue(self) -> bool:
        """Check if milestone is overdue"""
        return (self.status != ProgressStatus.COMPLETED and 
                datetime.now() > self.target_date)


@dataclass
class ProgressReport:
    """Comprehensive progress report"""
    id: str
    transformation_id: str
    report_date: datetime
    overall_progress: float
    milestones: List[TransformationMilestone]
    metrics: List[ProgressMetric]
    achievements: List[str]
    challenges: List[str]
    next_steps: List[str]
    risk_indicators: Dict[str, float]
    recommendations: List[str]
    
    @property
    def completed_milestones(self) -> List[TransformationMilestone]:
        """Get completed milestones"""
        return [m for m in self.milestones if m.status == ProgressStatus.COMPLETED]
    
    @property
    def overdue_milestones(self) -> List[TransformationMilestone]:
        """Get overdue milestones"""
        return [m for m in self.milestones if m.is_overdue]


@dataclass
class ProgressDashboard:
    """Progress visualization dashboard data"""
    transformation_id: str
    dashboard_date: datetime
    overall_health_score: float
    progress_charts: Dict[str, Any]
    milestone_timeline: List[Dict[str, Any]]
    metric_trends: Dict[str, List[float]]
    alert_indicators: List[Dict[str, Any]]
    executive_summary: str


@dataclass
class ProgressAlert:
    """Progress tracking alert"""
    id: str
    transformation_id: str
    alert_type: str  # "milestone_delay", "metric_decline", "risk_threshold"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    created_date: datetime
    resolved_date: Optional[datetime] = None
    action_required: bool = True
    assigned_to: Optional[str] = None