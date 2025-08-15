"""
Team Coordination Models for Crisis Leadership Excellence

Models for crisis team formation, role assignment, and performance monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Set
from uuid import uuid4


class SkillLevel(Enum):
    """Skill proficiency levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class AvailabilityStatus(Enum):
    """Personnel availability status"""
    AVAILABLE = "available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    ON_LEAVE = "on_leave"
    IN_CRISIS_RESPONSE = "in_crisis_response"


class TeamRole(Enum):
    """Crisis team roles"""
    CRISIS_LEADER = "crisis_leader"
    TECHNICAL_LEAD = "technical_lead"
    COMMUNICATIONS_LEAD = "communications_lead"
    OPERATIONS_LEAD = "operations_lead"
    SECURITY_LEAD = "security_lead"
    CUSTOMER_LIAISON = "customer_liaison"
    LEGAL_ADVISOR = "legal_advisor"
    EXECUTIVE_LIAISON = "executive_liaison"
    RESOURCE_COORDINATOR = "resource_coordinator"
    DOCUMENTATION_LEAD = "documentation_lead"


class PerformanceMetric(Enum):
    """Team performance metrics"""
    RESPONSE_TIME = "response_time"
    TASK_COMPLETION_RATE = "task_completion_rate"
    COMMUNICATION_EFFECTIVENESS = "communication_effectiveness"
    DECISION_QUALITY = "decision_quality"
    STRESS_MANAGEMENT = "stress_management"
    COLLABORATION_SCORE = "collaboration_score"


@dataclass
class Skill:
    """Individual skill representation"""
    name: str
    level: SkillLevel
    years_experience: float
    certifications: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None
    confidence_score: float = 0.0


@dataclass
class Person:
    """Personnel information for crisis team formation"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    email: str = ""
    phone: str = ""
    department: str = ""
    title: str = ""
    skills: List[Skill] = field(default_factory=list)
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE
    current_workload: float = 0.0  # 0.0 to 1.0
    crisis_experience: Dict[str, int] = field(default_factory=dict)  # crisis_type -> count
    performance_history: Dict[str, float] = field(default_factory=dict)
    preferred_roles: List[TeamRole] = field(default_factory=list)
    timezone: str = "UTC"
    languages: List[str] = field(default_factory=list)
    emergency_contact: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TeamComposition:
    """Optimal team composition for crisis type"""
    crisis_type: str
    required_roles: List[TeamRole]
    team_size_range: tuple  # (min, max)
    skill_requirements: Dict[str, SkillLevel]
    experience_requirements: Dict[str, int]
    availability_requirements: Dict[str, float]
    priority_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoleAssignment:
    """Assignment of person to role in crisis team"""
    person_id: str
    role: TeamRole
    assignment_confidence: float
    responsibilities: List[str]
    required_skills: List[str]
    backup_person_id: Optional[str] = None
    assignment_rationale: str = ""
    assigned_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CrisisTeam:
    """Complete crisis response team"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    team_name: str = ""
    crisis_type: str = ""
    team_lead_id: str = ""
    members: List[str] = field(default_factory=list)  # person IDs
    role_assignments: List[RoleAssignment] = field(default_factory=list)
    formation_time: datetime = field(default_factory=datetime.utcnow)
    activation_time: Optional[datetime] = None
    deactivation_time: Optional[datetime] = None
    team_status: str = "forming"  # forming, active, standby, disbanded
    communication_channels: Dict[str, str] = field(default_factory=dict)
    escalation_contacts: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class TeamFormationRequest:
    """Request for crisis team formation"""
    crisis_id: str
    crisis_type: str
    severity_level: int
    urgency: str = "high"
    required_skills: List[str] = field(default_factory=list)
    preferred_team_size: int = 5
    formation_deadline: Optional[datetime] = None
    special_requirements: Dict[str, Any] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SkillMatch:
    """Skill matching result for person-role assignment"""
    person_id: str
    role: TeamRole
    skill_match_score: float
    experience_match_score: float
    availability_score: float
    overall_match_score: float
    missing_skills: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    match_rationale: str = ""


@dataclass
class TeamPerformanceSnapshot:
    """Real-time team performance snapshot"""
    team_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_performance_score: float = 0.0
    individual_scores: Dict[str, float] = field(default_factory=dict)
    metric_scores: Dict[PerformanceMetric, float] = field(default_factory=dict)
    active_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    stress_indicators: Dict[str, float] = field(default_factory=dict)
    communication_health: float = 0.0
    task_completion_rate: float = 0.0


@dataclass
class PerformanceIssue:
    """Identified performance issue"""
    id: str = field(default_factory=lambda: str(uuid4()))
    team_id: str = ""
    person_id: Optional[str] = None
    issue_type: str = ""
    severity: str = "medium"
    description: str = ""
    impact_assessment: str = ""
    recommended_interventions: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""


@dataclass
class TeamOptimizationSuggestion:
    """Suggestion for team optimization"""
    team_id: str
    suggestion_type: str  # role_reassignment, skill_development, resource_addition
    description: str
    expected_impact: str
    implementation_effort: str = "medium"
    priority: str = "medium"
    suggested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CrisisTeamTemplate:
    """Template for common crisis team configurations"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    crisis_types: List[str] = field(default_factory=list)
    template_roles: List[TeamRole] = field(default_factory=list)
    role_requirements: Dict[TeamRole, Dict[str, Any]] = field(default_factory=dict)
    team_size_guidelines: Dict[str, int] = field(default_factory=dict)
    formation_checklist: List[str] = field(default_factory=list)
    communication_protocols: Dict[str, str] = field(default_factory=dict)
    escalation_procedures: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)