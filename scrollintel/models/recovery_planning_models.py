"""
Recovery Planning Models for Crisis Leadership Excellence System

This module defines data models for post-crisis recovery strategy development,
milestone tracking, and success measurement.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum


class RecoveryPhase(Enum):
    """Recovery phases for structured recovery planning"""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class RecoveryStatus(Enum):
    """Status of recovery activities"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    BLOCKED = "blocked"


class RecoveryPriority(Enum):
    """Priority levels for recovery activities"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RecoveryMilestone:
    """Individual recovery milestone with tracking capabilities"""
    id: str
    name: str
    description: str
    phase: RecoveryPhase
    priority: RecoveryPriority
    target_date: datetime
    completion_date: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.PLANNED
    success_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    assigned_team: Optional[str] = None
    progress_percentage: float = 0.0
    resources_required: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class RecoveryStrategy:
    """Comprehensive recovery strategy for post-crisis situations"""
    id: str
    crisis_id: str
    strategy_name: str
    created_at: datetime
    updated_at: datetime
    recovery_objectives: List[str]
    success_metrics: Dict[str, float]
    milestones: List[RecoveryMilestone]
    resource_allocation: Dict[str, Any]
    timeline: Dict[RecoveryPhase, timedelta]
    stakeholder_communication_plan: Dict[str, Any]
    risk_mitigation_measures: List[str]
    contingency_plans: Dict[str, Any]


@dataclass
class RecoveryProgress:
    """Progress tracking for recovery activities"""
    strategy_id: str
    overall_progress: float
    phase_progress: Dict[RecoveryPhase, float]
    milestone_completion_rate: float
    timeline_adherence: float
    resource_utilization: Dict[str, float]
    success_metric_achievement: Dict[str, float]
    identified_issues: List[str]
    recommended_adjustments: List[str]
    last_updated: datetime


@dataclass
class RecoveryOptimization:
    """Optimization recommendations for recovery processes"""
    strategy_id: str
    optimization_type: str
    current_performance: Dict[str, float]
    target_performance: Dict[str, float]
    recommended_actions: List[str]
    expected_impact: Dict[str, float]
    implementation_effort: str
    priority_score: float
    created_at: datetime