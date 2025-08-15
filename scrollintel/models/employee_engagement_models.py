"""
Employee Engagement Models for Cultural Transformation Leadership System
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class EngagementLevel(Enum):
    """Employee engagement levels"""
    DISENGAGED = "disengaged"
    SOMEWHAT_ENGAGED = "somewhat_engaged"
    ENGAGED = "engaged"
    HIGHLY_ENGAGED = "highly_engaged"


class EngagementActivityType(Enum):
    """Types of engagement activities"""
    SURVEY = "survey"
    WORKSHOP = "workshop"
    FEEDBACK_SESSION = "feedback_session"
    TEAM_BUILDING = "team_building"
    RECOGNITION = "recognition"
    DEVELOPMENT = "development"
    COMMUNICATION = "communication"
    COLLABORATION = "collaboration"


class EngagementMetricType(Enum):
    """Types of engagement metrics"""
    PARTICIPATION_RATE = "participation_rate"
    SATISFACTION_SCORE = "satisfaction_score"
    FEEDBACK_QUALITY = "feedback_quality"
    RETENTION_RATE = "retention_rate"
    PRODUCTIVITY_SCORE = "productivity_score"
    CULTURAL_ALIGNMENT = "cultural_alignment"


@dataclass
class Employee:
    """Employee information for engagement tracking"""
    id: str
    name: str
    department: str
    role: str
    manager_id: Optional[str]
    hire_date: datetime
    engagement_level: EngagementLevel
    cultural_alignment_score: float
    last_engagement_date: Optional[datetime] = None
    engagement_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.engagement_history is None:
            self.engagement_history = []


@dataclass
class EngagementActivity:
    """Engagement activity definition"""
    id: str
    name: str
    activity_type: EngagementActivityType
    description: str
    target_audience: List[str]  # departments, roles, or specific employee IDs
    objectives: List[str]
    duration_minutes: int
    facilitator: str
    materials_needed: List[str]
    success_criteria: List[str]
    cultural_values_addressed: List[str]
    created_date: datetime
    scheduled_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    status: str = "planned"


@dataclass
class EngagementStrategy:
    """Employee engagement strategy"""
    id: str
    organization_id: str
    name: str
    description: str
    target_groups: List[str]
    objectives: List[str]
    activities: List[EngagementActivity]
    timeline: Dict[str, datetime]
    success_metrics: List[str]
    budget_allocated: float
    owner: str
    created_date: datetime
    status: str = "draft"


@dataclass
class EngagementMetric:
    """Engagement measurement metric"""
    id: str
    metric_type: EngagementMetricType
    name: str
    description: str
    measurement_method: str
    target_value: float
    current_value: float
    trend: str  # "improving", "stable", "declining"
    measurement_date: datetime
    data_source: str


@dataclass
class EngagementAssessment:
    """Employee engagement assessment"""
    id: str
    employee_id: str
    assessment_date: datetime
    engagement_level: EngagementLevel
    satisfaction_scores: Dict[str, float]
    feedback_comments: List[str]
    improvement_areas: List[str]
    strengths: List[str]
    action_items: List[str]
    assessor: str
    next_assessment_date: datetime


@dataclass
class EngagementPlan:
    """Personalized employee engagement plan"""
    id: str
    employee_id: str
    current_engagement_level: EngagementLevel
    target_engagement_level: EngagementLevel
    strategies: List[EngagementStrategy]
    activities: List[EngagementActivity]
    milestones: List[Dict[str, Any]]
    timeline: Dict[str, datetime]
    success_metrics: List[EngagementMetric]
    created_date: datetime
    last_updated: datetime
    status: str = "active"


@dataclass
class EngagementFeedback:
    """Employee engagement feedback"""
    id: str
    employee_id: str
    activity_id: Optional[str]
    feedback_type: str  # "activity", "general", "suggestion"
    rating: Optional[float]
    comments: str
    suggestions: List[str]
    sentiment: str  # "positive", "neutral", "negative"
    themes: List[str]
    submitted_date: datetime
    processed: bool = False


@dataclass
class EngagementReport:
    """Engagement effectiveness report"""
    id: str
    organization_id: str
    report_period: Dict[str, datetime]
    overall_engagement_score: float
    engagement_by_department: Dict[str, float]
    engagement_trends: Dict[str, List[float]]
    activity_effectiveness: Dict[str, float]
    key_insights: List[str]
    recommendations: List[str]
    metrics: List[EngagementMetric]
    generated_date: datetime
    generated_by: str


@dataclass
class EngagementImprovementPlan:
    """Plan for improving employee engagement"""
    id: str
    organization_id: str
    current_state: Dict[str, Any]
    target_state: Dict[str, Any]
    improvement_strategies: List[EngagementStrategy]
    implementation_timeline: Dict[str, datetime]
    resource_requirements: Dict[str, Any]
    success_criteria: List[str]
    risk_mitigation: List[str]
    monitoring_plan: Dict[str, Any]
    created_date: datetime
    owner: str
    status: str = "draft"