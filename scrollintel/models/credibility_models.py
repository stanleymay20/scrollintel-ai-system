"""
Credibility and Trust Management Models for Board Executive Mastery

This module defines data models for credibility building and trust management
in board and executive relationships.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class CredibilityLevel(Enum):
    """Credibility assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXCEPTIONAL = "exceptional"


class TrustLevel(Enum):
    """Trust assessment levels"""
    DISTRUST = "distrust"
    CAUTIOUS = "cautious"
    NEUTRAL = "neutral"
    TRUSTING = "trusting"
    COMPLETE_TRUST = "complete_trust"


class CredibilityFactor(Enum):
    """Factors that influence credibility"""
    EXPERTISE = "expertise"
    TRACK_RECORD = "track_record"
    TRANSPARENCY = "transparency"
    CONSISTENCY = "consistency"
    COMMUNICATION = "communication"
    RESULTS_DELIVERY = "results_delivery"
    STRATEGIC_INSIGHT = "strategic_insight"
    PROBLEM_SOLVING = "problem_solving"


@dataclass
class CredibilityMetric:
    """Individual credibility measurement"""
    factor: CredibilityFactor
    score: float  # 0.0 to 1.0
    evidence: List[str]
    last_updated: datetime
    trend: str  # "improving", "stable", "declining"


@dataclass
class CredibilityAssessment:
    """Comprehensive credibility assessment"""
    stakeholder_id: str
    overall_score: float
    level: CredibilityLevel
    metrics: List[CredibilityMetric]
    strengths: List[str]
    improvement_areas: List[str]
    assessment_date: datetime
    historical_scores: List[float]


@dataclass
class CredibilityAction:
    """Action to build or maintain credibility"""
    id: str
    title: str
    description: str
    target_factor: CredibilityFactor
    expected_impact: float
    timeline: str
    resources_required: List[str]
    success_metrics: List[str]
    status: str  # "planned", "in_progress", "completed", "cancelled"


@dataclass
class TrustMetric:
    """Individual trust measurement"""
    dimension: str  # "reliability", "competence", "benevolence", "integrity"
    score: float  # 0.0 to 1.0
    evidence: List[str]
    last_interaction: datetime
    trend: str


@dataclass
class TrustAssessment:
    """Comprehensive trust assessment"""
    stakeholder_id: str
    overall_score: float
    level: TrustLevel
    metrics: List[TrustMetric]
    trust_drivers: List[str]
    trust_barriers: List[str]
    assessment_date: datetime
    relationship_history: List[Dict[str, Any]]


@dataclass
class TrustBuildingStrategy:
    """Strategy for building trust with stakeholder"""
    id: str
    stakeholder_id: str
    current_trust_level: TrustLevel
    target_trust_level: TrustLevel
    key_actions: List[str]
    timeline: str
    milestones: List[Dict[str, Any]]
    risk_factors: List[str]
    success_indicators: List[str]


@dataclass
class RelationshipEvent:
    """Event that impacts credibility or trust"""
    id: str
    stakeholder_id: str
    event_type: str  # "meeting", "presentation", "decision", "crisis", "success"
    description: str
    date: datetime
    credibility_impact: float  # -1.0 to 1.0
    trust_impact: float  # -1.0 to 1.0
    lessons_learned: List[str]
    follow_up_actions: List[str]


@dataclass
class StakeholderProfile:
    """Profile of board member or executive stakeholder"""
    id: str
    name: str
    role: str
    background: str
    values: List[str]
    communication_preferences: Dict[str, str]
    decision_making_style: str
    influence_level: float
    credibility_assessment: Optional[CredibilityAssessment]
    trust_assessment: Optional[TrustAssessment]
    relationship_events: List[RelationshipEvent]


@dataclass
class CredibilityPlan:
    """Comprehensive plan for building credibility"""
    id: str
    stakeholder_id: str
    current_assessment: CredibilityAssessment
    target_level: CredibilityLevel
    timeline: str
    actions: List[CredibilityAction]
    milestones: List[Dict[str, Any]]
    monitoring_schedule: List[str]
    contingency_plans: List[str]


@dataclass
class TrustRecoveryPlan:
    """Plan for recovering damaged trust"""
    id: str
    stakeholder_id: str
    trust_breach_description: str
    current_trust_level: TrustLevel
    target_trust_level: TrustLevel
    recovery_strategy: str
    immediate_actions: List[str]
    long_term_actions: List[str]
    timeline: str
    success_metrics: List[str]
    monitoring_plan: str


@dataclass
class CredibilityReport:
    """Report on credibility status and progress"""
    id: str
    report_date: datetime
    stakeholder_assessments: List[CredibilityAssessment]
    overall_credibility_score: float
    key_achievements: List[str]
    areas_for_improvement: List[str]
    recommended_actions: List[str]
    trend_analysis: Dict[str, Any]
    next_review_date: datetime