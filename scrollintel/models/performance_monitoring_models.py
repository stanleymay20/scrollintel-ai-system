"""
Performance Monitoring Models for Crisis Leadership Excellence

This module defines data models for real-time team performance tracking,
issue identification, and optimization during crisis situations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class PerformanceStatus(Enum):
    """Performance status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    CRITICAL = "critical"


class InterventionType(Enum):
    """Types of performance interventions"""
    COACHING = "coaching"
    RESOURCE_ALLOCATION = "resource_allocation"
    ROLE_REASSIGNMENT = "role_reassignment"
    ADDITIONAL_SUPPORT = "additional_support"
    TRAINING = "training"
    WORKLOAD_ADJUSTMENT = "workload_adjustment"


class SupportType(Enum):
    """Types of support provision"""
    TECHNICAL_SUPPORT = "technical_support"
    EMOTIONAL_SUPPORT = "emotional_support"
    RESOURCE_SUPPORT = "resource_support"
    MENTORING = "mentoring"
    SKILL_DEVELOPMENT = "skill_development"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    is_critical: bool = False


@dataclass
class TeamMemberPerformance:
    """Performance data for individual team member"""
    member_id: str
    member_name: str
    role: str
    crisis_id: str
    performance_status: PerformanceStatus
    overall_score: float
    metrics: List[PerformanceMetric]
    task_completion_rate: float
    response_time_avg: float
    quality_score: float
    stress_level: float
    collaboration_score: float
    last_updated: datetime
    issues_identified: List[str]
    interventions_needed: List[InterventionType]


@dataclass
class PerformanceIssue:
    """Identified performance issue"""
    issue_id: str
    member_id: str
    crisis_id: str
    issue_type: str
    severity: str
    description: str
    impact_assessment: str
    identified_at: datetime
    resolved_at: Optional[datetime] = None
    intervention_applied: Optional[InterventionType] = None
    resolution_notes: Optional[str] = None


@dataclass
class PerformanceIntervention:
    """Performance intervention action"""
    intervention_id: str
    member_id: str
    crisis_id: str
    intervention_type: InterventionType
    description: str
    expected_outcome: str
    implemented_at: datetime
    effectiveness_score: Optional[float] = None
    completion_status: str = "pending"
    follow_up_required: bool = True


@dataclass
class SupportProvision:
    """Support provided to team members"""
    support_id: str
    member_id: str
    crisis_id: str
    support_type: SupportType
    description: str
    provider: str
    provided_at: datetime
    duration_minutes: Optional[int] = None
    effectiveness_rating: Optional[float] = None
    member_feedback: Optional[str] = None


@dataclass
class TeamPerformanceOverview:
    """Overall team performance summary"""
    crisis_id: str
    team_id: str
    overall_performance_score: float
    team_efficiency: float
    collaboration_index: float
    stress_level_avg: float
    task_completion_rate: float
    response_time_avg: float
    member_performances: List[TeamMemberPerformance]
    critical_issues_count: int
    interventions_active: int
    support_provisions_active: int
    last_updated: datetime


@dataclass
class PerformanceOptimization:
    """Performance optimization recommendation"""
    optimization_id: str
    crisis_id: str
    target_area: str
    current_performance: float
    target_performance: float
    optimization_strategy: str
    implementation_steps: List[str]
    expected_impact: str
    priority_level: str
    estimated_completion_time: int
    resources_required: List[str]


@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    crisis_id: str
    member_id: Optional[str]
    alert_type: str
    severity: str
    message: str
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    action_taken: Optional[str] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    crisis_id: str
    report_type: str
    generated_at: datetime
    time_period_start: datetime
    time_period_end: datetime
    team_overview: TeamPerformanceOverview
    key_insights: List[str]
    performance_trends: Dict[str, Any]
    recommendations: List[PerformanceOptimization]
    success_metrics: Dict[str, float]