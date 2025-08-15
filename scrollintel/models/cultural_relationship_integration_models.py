"""
Cultural Relationship Integration Models

Data models for integrating cultural transformation with human relationship systems.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class RelationshipType(Enum):
    """Types of human relationships in organizational context"""
    MANAGER_EMPLOYEE = "manager_employee"
    PEER_TO_PEER = "peer_to_peer"
    CROSS_FUNCTIONAL = "cross_functional"
    MENTOR_MENTEE = "mentor_mentee"
    CLIENT_INTERNAL = "client_internal"
    VENDOR_PARTNER = "vendor_partner"
    TEAM_MEMBER = "team_member"
    STAKEHOLDER = "stakeholder"


class CulturalContext(Enum):
    """Cultural contexts that influence relationships"""
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    SUPPORTIVE = "supportive"
    FORMAL = "formal"
    INFORMAL = "informal"
    INNOVATIVE = "innovative"
    TRADITIONAL = "traditional"


class RelationshipHealth(Enum):
    """Health status of relationships"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class CommunicationStyle(Enum):
    """Communication styles influenced by culture"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    FORMAL = "formal"
    CASUAL = "casual"
    COLLABORATIVE = "collaborative"
    AUTHORITATIVE = "authoritative"
    SUPPORTIVE = "supportive"
    ANALYTICAL = "analytical"


@dataclass
class CulturalRelationshipProfile:
    """Profile of how culture influences a specific relationship"""
    id: str
    relationship_id: str
    person_a_id: str
    person_b_id: str
    relationship_type: RelationshipType
    cultural_context: CulturalContext
    communication_style: CommunicationStyle
    cultural_factors: Dict[str, float]
    interaction_patterns: List[str]
    cultural_barriers: List[str]
    cultural_enablers: List[str]
    optimization_opportunities: List[str]
    last_updated: datetime


@dataclass
class RelationshipOptimization:
    """Optimization recommendations for culture-aware relationships"""
    id: str
    relationship_profile_id: str
    optimization_type: str
    current_effectiveness: float
    target_effectiveness: float
    cultural_interventions: List[str]
    communication_adjustments: List[str]
    behavioral_recommendations: List[str]
    success_metrics: List[str]
    implementation_timeline: str
    priority_level: int
    created_at: datetime


@dataclass
class CulturalInteraction:
    """Record of culturally-aware interaction"""
    id: str
    interaction_type: str
    participants: List[str]
    cultural_context: CulturalContext
    communication_style: CommunicationStyle
    cultural_considerations: Dict[str, Any]
    interaction_outcome: str
    effectiveness_score: float
    cultural_alignment: float
    lessons_learned: List[str]
    timestamp: datetime


@dataclass
class RelationshipMetrics:
    """Metrics for measuring relationship effectiveness with cultural context"""
    id: str
    relationship_id: str
    measurement_period: str
    trust_level: float
    communication_effectiveness: float
    collaboration_quality: float
    cultural_alignment: float
    conflict_resolution_effectiveness: float
    mutual_respect_level: float
    overall_relationship_health: RelationshipHealth
    cultural_impact_score: float
    improvement_areas: List[str]
    measured_at: datetime


@dataclass
class CulturalCommunicationGuideline:
    """Guidelines for culturally appropriate communication"""
    id: str
    cultural_context: CulturalContext
    relationship_type: RelationshipType
    communication_do_list: List[str]
    communication_dont_list: List[str]
    preferred_channels: List[str]
    timing_considerations: List[str]
    tone_recommendations: List[str]
    cultural_sensitivities: List[str]
    example_scenarios: List[Dict[str, str]]
    effectiveness_indicators: List[str]


@dataclass
class RelationshipConflictResolution:
    """Cultural approach to relationship conflict resolution"""
    id: str
    conflict_id: str
    relationship_id: str
    conflict_type: str
    cultural_factors: Dict[str, Any]
    resolution_approach: str
    cultural_mediation_strategies: List[str]
    communication_protocols: List[str]
    resolution_timeline: str
    success_criteria: List[str]
    cultural_learning_outcomes: List[str]
    resolution_status: str
    resolved_at: Optional[datetime]


@dataclass
class TeamCulturalDynamics:
    """Analysis of cultural dynamics within teams"""
    id: str
    team_id: str
    team_members: List[str]
    cultural_diversity_score: float
    dominant_cultural_patterns: List[str]
    communication_patterns: Dict[str, Any]
    collaboration_effectiveness: float
    cultural_conflicts: List[str]
    cultural_synergies: List[str]
    optimization_recommendations: List[str]
    team_cultural_health: float
    assessment_date: datetime


@dataclass
class CulturalRelationshipInsight:
    """Insights derived from cultural relationship analysis"""
    id: str
    insight_type: str
    scope: str  # individual, team, department, organization
    cultural_pattern: str
    relationship_impact: str
    actionable_recommendations: List[str]
    potential_benefits: List[str]
    implementation_complexity: str
    confidence_level: float
    supporting_evidence: List[str]
    generated_at: datetime


@dataclass
class RelationshipIntegrationReport:
    """Report on cultural-relationship integration"""
    id: str
    report_type: str
    reporting_period: str
    scope: str
    relationship_profiles: List[CulturalRelationshipProfile]
    optimization_summary: Dict[str, Any]
    cultural_insights: List[CulturalRelationshipInsight]
    team_dynamics_analysis: List[TeamCulturalDynamics]
    success_metrics: Dict[str, float]
    recommendations: List[str]
    next_steps: List[str]
    generated_at: datetime
    generated_by: str