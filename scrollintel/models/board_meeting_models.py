"""
Board Meeting Optimization Models

This module defines data models for board meeting preparation, facilitation,
and optimization functionality.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class MeetingType(Enum):
    """Types of board meetings"""
    REGULAR = "regular"
    SPECIAL = "special"
    ANNUAL = "annual"
    EMERGENCY = "emergency"
    COMMITTEE = "committee"


class MeetingStatus(Enum):
    """Meeting status options"""
    PLANNED = "planned"
    PREPARED = "prepared"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AgendaItemType(Enum):
    """Types of agenda items"""
    STRATEGIC_REVIEW = "strategic_review"
    FINANCIAL_REPORT = "financial_report"
    GOVERNANCE = "governance"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION_ITEM = "decision_item"
    INFORMATION_ITEM = "information_item"
    DISCUSSION = "discussion"


class PreparationStatus(Enum):
    """Preparation status for agenda items"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    NEEDS_REVIEW = "needs_review"


@dataclass
class AgendaItem:
    """Individual agenda item for board meetings"""
    id: str
    title: str
    type: AgendaItemType
    description: str
    presenter: str
    duration_minutes: int
    priority: int
    preparation_status: PreparationStatus
    materials: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    decision_required: bool = False
    background_context: str = ""
    success_criteria: List[str] = field(default_factory=list)
    potential_questions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class MeetingPreparation:
    """Comprehensive meeting preparation data"""
    meeting_id: str
    preparation_checklist: List[str]
    materials_prepared: List[str]
    stakeholder_briefings: Dict[str, str]
    risk_assessments: List[str]
    success_predictions: Dict[str, float]
    contingency_plans: List[str]
    preparation_score: float
    last_updated: datetime
    preparation_notes: str = ""


@dataclass
class BoardMeeting:
    """Board meeting data model"""
    id: str
    title: str
    meeting_type: MeetingType
    scheduled_date: datetime
    duration_minutes: int
    location: str
    status: MeetingStatus
    agenda_items: List[AgendaItem]
    attendees: List[str]
    preparation: Optional[MeetingPreparation]
    board_id: str
    chair_id: str
    secretary_id: str
    created_at: datetime
    updated_at: datetime
    meeting_objectives: List[str] = field(default_factory=list)
    pre_meeting_materials: List[str] = field(default_factory=list)
    post_meeting_actions: List[str] = field(default_factory=list)
    meeting_notes: str = ""
    decisions_made: List[str] = field(default_factory=list)


@dataclass
class MeetingOptimization:
    """Meeting optimization recommendations"""
    meeting_id: str
    agenda_optimization: Dict[str, Any]
    timing_recommendations: Dict[str, int]
    content_suggestions: List[str]
    flow_improvements: List[str]
    engagement_strategies: List[str]
    success_probability: float
    optimization_score: float
    recommendations: List[str]
    potential_issues: List[str]
    mitigation_strategies: List[str]


@dataclass
class MeetingFacilitation:
    """Meeting facilitation support data"""
    meeting_id: str
    facilitation_guide: List[str]
    discussion_prompts: Dict[str, List[str]]
    conflict_resolution_strategies: List[str]
    time_management_cues: List[str]
    engagement_techniques: List[str]
    decision_facilitation_tools: List[str]
    meeting_flow_checkpoints: List[str]
    real_time_adjustments: List[str]


@dataclass
class MeetingOutcome:
    """Meeting outcome tracking"""
    meeting_id: str
    objectives_achieved: List[str]
    decisions_made: List[str]
    action_items: List[str]
    follow_up_required: List[str]
    stakeholder_satisfaction: Dict[str, float]
    meeting_effectiveness: float
    areas_for_improvement: List[str]
    success_metrics: Dict[str, float]
    next_meeting_recommendations: List[str]


@dataclass
class MeetingAnalytics:
    """Meeting analytics and insights"""
    meeting_id: str
    attendance_rate: float
    engagement_score: float
    decision_efficiency: float
    time_utilization: float
    content_relevance: float
    stakeholder_feedback: Dict[str, str]
    improvement_opportunities: List[str]
    benchmark_comparisons: Dict[str, float]
    trend_analysis: Dict[str, Any]