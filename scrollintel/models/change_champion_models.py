"""
Change Champion Models

Data models for change champion identification, development, and network management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ChampionLevel(Enum):
    """Change champion levels"""
    EMERGING = "emerging"
    DEVELOPING = "developing"
    ACTIVE = "active"
    SENIOR = "senior"
    MASTER = "master"


class ChampionRole(Enum):
    """Change champion roles"""
    ADVOCATE = "advocate"
    FACILITATOR = "facilitator"
    TRAINER = "trainer"
    MENTOR = "mentor"
    COORDINATOR = "coordinator"
    STRATEGIST = "strategist"


class ChangeCapability(Enum):
    """Core change champion capabilities"""
    CHANGE_ADVOCACY = "change_advocacy"
    INFLUENCE_BUILDING = "influence_building"
    COMMUNICATION = "communication"
    TRAINING_DELIVERY = "training_delivery"
    RESISTANCE_MANAGEMENT = "resistance_management"
    NETWORK_BUILDING = "network_building"
    FEEDBACK_COLLECTION = "feedback_collection"
    COACHING_MENTORING = "coaching_mentoring"
    PROJECT_COORDINATION = "project_coordination"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"


class NetworkStatus(Enum):
    """Change champion network status"""
    FORMING = "forming"
    STORMING = "storming"
    NORMING = "norming"
    PERFORMING = "performing"
    TRANSFORMING = "transforming"


@dataclass
class ChangeChampionProfile:
    """Individual change champion profile"""
    id: str
    employee_id: str
    name: str
    role: str
    department: str
    organization_id: str
    champion_level: ChampionLevel
    champion_roles: List[ChampionRole]
    capabilities: Dict[ChangeCapability, float]  # Capability scores 0-100
    influence_network: List[str]  # Employee IDs in their network
    credibility_score: float
    engagement_score: float
    availability_score: float
    motivation_score: float
    cultural_fit_score: float
    change_experience: List[str]
    training_completed: List[str]
    certifications: List[str]
    mentorship_relationships: List[str]
    success_metrics: Dict[str, float]
    status: str = "active"
    joined_date: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class ChampionIdentificationCriteria:
    """Criteria for identifying potential change champions"""
    id: str
    name: str
    description: str
    required_capabilities: List[ChangeCapability]
    minimum_scores: Dict[ChangeCapability, float]
    influence_requirements: Dict[str, Any]
    experience_requirements: List[str]
    role_preferences: List[str]
    department_coverage: List[str]
    cultural_factors: List[str]
    exclusion_criteria: List[str]
    weight_factors: Dict[str, float]


@dataclass
class ChampionDevelopmentProgram:
    """Change champion development program"""
    id: str
    name: str
    description: str
    target_level: ChampionLevel
    target_roles: List[ChampionRole]
    duration_weeks: int
    learning_modules: List['LearningModule']
    practical_assignments: List['PracticalAssignment']
    mentorship_component: bool
    peer_learning_groups: bool
    certification_available: bool
    success_criteria: List[str]
    prerequisites: List[str]
    resources_required: List[str]


@dataclass
class LearningModule:
    """Individual learning module in development program"""
    id: str
    title: str
    description: str
    target_capabilities: List[ChangeCapability]
    learning_objectives: List[str]
    content_type: str  # workshop, online, reading, simulation
    duration_hours: int
    delivery_method: str
    materials: List[str]
    assessments: List[str]
    completion_criteria: List[str]
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class PracticalAssignment:
    """Practical assignment for skill development"""
    id: str
    title: str
    description: str
    target_capabilities: List[ChangeCapability]
    assignment_type: str  # project, presentation, facilitation, coaching
    duration_weeks: int
    deliverables: List[str]
    success_metrics: List[str]
    support_provided: List[str]
    evaluation_criteria: List[str]
    mentor_involvement: bool


@dataclass
class ChampionTrainingSession:
    """Individual training session"""
    id: str
    program_id: str
    module_id: str
    champion_id: str
    session_date: datetime
    duration_hours: int
    trainer_id: str
    session_type: str  # workshop, coaching, peer_learning, simulation
    learning_objectives: List[str]
    activities: List[str]
    materials_used: List[str]
    attendance_status: str
    engagement_score: float
    learning_assessment: Dict[str, float]
    feedback: str
    action_items: List[str]
    completion_status: str = "scheduled"


@dataclass
class ChampionNetwork:
    """Change champion network structure"""
    id: str
    name: str
    organization_id: str
    network_type: str  # departmental, cross_functional, project_based
    champions: List[str]  # Champion IDs
    network_lead: str  # Lead champion ID
    coordinators: List[str]  # Coordinator champion IDs
    coverage_areas: List[str]
    network_status: NetworkStatus
    formation_date: datetime
    objectives: List[str]
    success_metrics: List[str]
    communication_channels: List[str]
    meeting_schedule: str
    governance_structure: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class NetworkActivity:
    """Change champion network activity"""
    id: str
    network_id: str
    activity_type: str  # meeting, training, project, communication
    title: str
    description: str
    organizer_id: str
    participants: List[str]
    activity_date: datetime
    duration_hours: int
    objectives: List[str]
    outcomes: List[str]
    engagement_metrics: Dict[str, float]
    feedback_collected: List[str]
    follow_up_actions: List[str]
    success_rating: float


@dataclass
class ChampionMentorship:
    """Mentorship relationship between champions"""
    id: str
    mentor_id: str
    mentee_id: str
    mentorship_type: str  # formal, informal, peer, reverse
    start_date: datetime
    planned_duration_months: int
    focus_areas: List[ChangeCapability]
    objectives: List[str]
    meeting_frequency: str
    sessions_completed: int
    progress_milestones: List[str]
    success_metrics: List[str]
    mentor_feedback: List[str]
    mentee_feedback: List[str]
    status: str = "active"
    end_date: Optional[datetime] = None


@dataclass
class ChampionPerformanceMetrics:
    """Performance metrics for change champions"""
    champion_id: str
    measurement_period: str
    change_initiatives_supported: int
    training_sessions_delivered: int
    employees_influenced: int
    resistance_cases_resolved: int
    feedback_sessions_conducted: int
    network_engagement_score: float
    peer_rating: float
    manager_rating: float
    change_success_contribution: float
    knowledge_sharing_score: float
    mentorship_effectiveness: float
    innovation_contributions: int
    cultural_alignment_score: float
    overall_performance_score: float
    recognition_received: List[str]
    development_areas: List[str]
    measurement_date: datetime = field(default_factory=datetime.now)


@dataclass
class ChampionRecognition:
    """Recognition and rewards for change champions"""
    id: str
    champion_id: str
    recognition_type: str  # award, certificate, promotion, bonus
    title: str
    description: str
    criteria_met: List[str]
    awarded_by: str
    award_date: datetime
    public_recognition: bool
    monetary_value: Optional[float] = None
    career_impact: Optional[str] = None


@dataclass
class NetworkCoordinationPlan:
    """Plan for coordinating change champion network"""
    id: str
    network_id: str
    coordination_period: str
    objectives: List[str]
    key_initiatives: List[str]
    resource_allocation: Dict[str, Any]
    communication_strategy: Dict[str, Any]
    training_schedule: List[Dict[str, Any]]
    performance_targets: Dict[str, float]
    risk_mitigation: List[str]
    success_metrics: List[str]
    review_schedule: str
    stakeholder_engagement: List[str]
    budget_requirements: Optional[float] = None