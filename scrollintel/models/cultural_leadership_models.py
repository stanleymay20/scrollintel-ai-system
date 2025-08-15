"""
Cultural Leadership Models

Data models for cultural leadership assessment, training, and effectiveness measurement.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class LeadershipLevel(Enum):
    """Leadership levels for assessment"""
    EMERGING = "emerging"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CulturalCompetency(Enum):
    """Core cultural leadership competencies"""
    VISION_CREATION = "vision_creation"
    VALUES_ALIGNMENT = "values_alignment"
    CHANGE_LEADERSHIP = "change_leadership"
    COMMUNICATION = "communication"
    INFLUENCE = "influence"
    EMPATHY = "empathy"
    AUTHENTICITY = "authenticity"
    RESILIENCE = "resilience"
    ADAPTABILITY = "adaptability"
    SYSTEMS_THINKING = "systems_thinking"


@dataclass
class CompetencyScore:
    """Individual competency assessment score"""
    competency: CulturalCompetency
    current_level: LeadershipLevel
    target_level: LeadershipLevel
    score: float  # 0-100
    evidence: List[str]
    development_areas: List[str]
    strengths: List[str]


@dataclass
class CulturalLeadershipAssessment:
    """Comprehensive cultural leadership assessment"""
    id: str
    leader_id: str
    organization_id: str
    assessment_date: datetime
    competency_scores: List[CompetencyScore]
    overall_score: float
    leadership_level: LeadershipLevel
    cultural_impact_score: float
    vision_clarity_score: float
    communication_effectiveness: float
    change_readiness: float
    team_engagement_score: float
    assessment_method: str
    assessor_id: Optional[str] = None
    self_assessment: bool = False
    peer_feedback: List[Dict[str, Any]] = field(default_factory=list)
    direct_report_feedback: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    development_plan: Optional['LeadershipDevelopmentPlan'] = None


@dataclass
class LeadershipDevelopmentPlan:
    """Personalized leadership development plan"""
    id: str
    leader_id: str
    assessment_id: str
    created_date: datetime
    target_completion: datetime
    priority_competencies: List[CulturalCompetency]
    development_goals: List[str]
    learning_activities: List['LearningActivity']
    coaching_sessions: List['CoachingSession']
    progress_milestones: List['DevelopmentMilestone']
    success_metrics: List[str]
    status: str = "active"


@dataclass
class LearningActivity:
    """Individual learning activity in development plan"""
    id: str
    title: str
    description: str
    activity_type: str  # workshop, reading, project, mentoring, etc.
    target_competencies: List[CulturalCompetency]
    estimated_duration: int  # hours
    resources: List[str]
    completion_criteria: List[str]
    status: str = "not_started"
    completion_date: Optional[datetime] = None
    effectiveness_rating: Optional[float] = None


@dataclass
class CoachingSession:
    """Cultural leadership coaching session"""
    id: str
    leader_id: str
    coach_id: str
    session_date: datetime
    duration: int  # minutes
    focus_areas: List[CulturalCompetency]
    objectives: List[str]
    activities: List[str]
    insights: List[str]
    action_items: List[str]
    progress_notes: str
    effectiveness_rating: Optional[float] = None


@dataclass
class DevelopmentMilestone:
    """Development progress milestone"""
    id: str
    title: str
    description: str
    target_date: datetime
    completion_criteria: List[str]
    success_metrics: List[str]
    status: str = "pending"
    completion_date: Optional[datetime] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class CulturalLeadershipProfile:
    """Comprehensive cultural leadership profile"""
    leader_id: str
    name: str
    role: str
    organization_id: str
    current_assessment: Optional[CulturalLeadershipAssessment] = None
    assessment_history: List[CulturalLeadershipAssessment] = field(default_factory=list)
    development_plans: List[LeadershipDevelopmentPlan] = field(default_factory=list)
    coaching_history: List[CoachingSession] = field(default_factory=list)
    cultural_impact_metrics: Dict[str, float] = field(default_factory=dict)
    leadership_style: Optional[str] = None
    strengths: List[str] = field(default_factory=list)
    development_areas: List[str] = field(default_factory=list)
    career_aspirations: List[str] = field(default_factory=list)


@dataclass
class AssessmentFramework:
    """Cultural leadership assessment framework"""
    id: str
    name: str
    description: str
    competencies: List[CulturalCompetency]
    assessment_methods: List[str]
    scoring_rubric: Dict[str, Any]
    validity_metrics: Dict[str, float]
    reliability_score: float
    cultural_context: str
    target_roles: List[str]
    version: str = "1.0"
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class LeadershipEffectivenessMetrics:
    """Metrics for measuring cultural leadership effectiveness"""
    leader_id: str
    measurement_period: str
    team_engagement_score: float
    cultural_alignment_score: float
    change_success_rate: float
    vision_clarity_rating: float
    communication_effectiveness: float
    influence_reach: int
    retention_rate: float
    promotion_rate: float
    peer_leadership_rating: float
    direct_report_satisfaction: float
    cultural_initiative_success: float
    innovation_fostered: int
    conflict_resolution_success: float
    measurement_date: datetime = field(default_factory=datetime.now)