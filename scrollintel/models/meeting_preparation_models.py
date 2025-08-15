"""
Meeting Preparation Models for Board Executive Mastery System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class MeetingType(Enum):
    BOARD_MEETING = "board_meeting"
    EXECUTIVE_COMMITTEE = "executive_committee"
    AUDIT_COMMITTEE = "audit_committee"
    COMPENSATION_COMMITTEE = "compensation_committee"
    STRATEGY_SESSION = "strategy_session"
    QUARTERLY_REVIEW = "quarterly_review"


class PreparationStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    APPROVED = "approved"


class ContentType(Enum):
    PRESENTATION = "presentation"
    FINANCIAL_REPORT = "financial_report"
    STRATEGIC_UPDATE = "strategic_update"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_METRICS = "performance_metrics"
    DECISION_PROPOSAL = "decision_proposal"


@dataclass
class BoardMember:
    id: str
    name: str
    role: str
    expertise_areas: List[str]
    communication_preferences: Dict[str, Any]
    influence_level: float
    typical_concerns: List[str]
    decision_patterns: Dict[str, Any]


@dataclass
class MeetingObjective:
    id: str
    title: str
    description: str
    priority: int
    success_criteria: List[str]
    required_decisions: List[str]
    stakeholders: List[str]


@dataclass
class AgendaItem:
    id: str
    title: str
    description: str
    presenter: str
    duration_minutes: int
    content_type: ContentType
    objectives: List[str]
    materials_required: List[str]
    key_messages: List[str]
    anticipated_questions: List[str]
    decision_required: bool
    priority: int


@dataclass
class MeetingContent:
    id: str
    agenda_item_id: str
    content_type: ContentType
    title: str
    summary: str
    detailed_content: str
    supporting_data: Dict[str, Any]
    visualizations: List[str]
    key_takeaways: List[str]
    recommendations: List[str]


@dataclass
class PreparationTask:
    id: str
    title: str
    description: str
    assignee: str
    due_date: datetime
    status: PreparationStatus
    dependencies: List[str]
    deliverables: List[str]
    completion_criteria: List[str]


@dataclass
class SuccessMetric:
    id: str
    name: str
    description: str
    target_value: float
    measurement_method: str
    importance_weight: float


@dataclass
class MeetingPreparation:
    id: str
    meeting_id: str
    meeting_type: MeetingType
    meeting_date: datetime
    board_members: List[BoardMember]
    objectives: List[MeetingObjective]
    agenda_items: List[AgendaItem]
    content_materials: List[MeetingContent]
    preparation_tasks: List[PreparationTask]
    success_metrics: List[SuccessMetric]
    status: PreparationStatus
    created_at: datetime
    updated_at: datetime
    preparation_score: Optional[float] = None
    success_prediction: Optional[float] = None
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class AgendaOptimization:
    id: str
    meeting_preparation_id: str
    original_agenda: List[AgendaItem]
    optimized_agenda: List[AgendaItem]
    optimization_rationale: str
    time_allocation: Dict[str, int]
    flow_improvements: List[str]
    engagement_enhancements: List[str]
    decision_optimization: List[str]


@dataclass
class ContentPreparation:
    id: str
    meeting_preparation_id: str
    content_id: str
    target_audience: List[str]
    key_messages: List[str]
    supporting_evidence: List[str]
    visual_aids: List[str]
    narrative_structure: str
    anticipated_reactions: Dict[str, str]
    response_strategies: Dict[str, str]


@dataclass
class MeetingSuccessPrediction:
    id: str
    meeting_preparation_id: str
    overall_success_probability: float
    objective_achievement_probabilities: Dict[str, float]
    engagement_prediction: float
    decision_quality_prediction: float
    stakeholder_satisfaction_prediction: Dict[str, float]
    risk_factors: List[Dict[str, Any]]
    enhancement_recommendations: List[str]
    confidence_interval: Dict[str, float]


@dataclass
class PreparationInsight:
    id: str
    meeting_preparation_id: str
    insight_type: str
    title: str
    description: str
    impact_level: str
    actionable_recommendations: List[str]
    supporting_data: Dict[str, Any]
    confidence_score: float