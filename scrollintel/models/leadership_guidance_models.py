"""
Models for Crisis Leadership Guidance System
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

class CrisisType(Enum):
    TECHNICAL_OUTAGE = "technical_outage"
    SECURITY_BREACH = "security_breach"
    FINANCIAL_CRISIS = "financial_crisis"
    REGULATORY_ISSUE = "regulatory_issue"
    REPUTATION_DAMAGE = "reputation_damage"
    OPERATIONAL_FAILURE = "operational_failure"
    MARKET_VOLATILITY = "market_volatility"
    LEADERSHIP_CRISIS = "leadership_crisis"

class LeadershipStyle(Enum):
    DIRECTIVE = "directive"
    COLLABORATIVE = "collaborative"
    SUPPORTIVE = "supportive"
    TRANSFORMATIONAL = "transformational"
    ADAPTIVE = "adaptive"

class DecisionUrgency(Enum):
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    MODERATE = "moderate"
    LOW = "low"

@dataclass
class LeadershipBestPractice:
    id: str
    crisis_type: CrisisType
    practice_name: str
    description: str
    implementation_steps: List[str]
    success_indicators: List[str]
    common_pitfalls: List[str]
    effectiveness_score: float
    applicable_scenarios: List[str]

@dataclass
class DecisionContext:
    crisis_id: str
    crisis_type: CrisisType
    severity_level: int  # 1-10 scale
    stakeholders_affected: List[str]
    time_pressure: DecisionUrgency
    available_information: Dict[str, Any]
    resource_constraints: List[str]
    regulatory_considerations: List[str]

@dataclass
class LeadershipRecommendation:
    id: str
    context: DecisionContext
    recommended_style: LeadershipStyle
    key_actions: List[str]
    communication_strategy: str
    stakeholder_priorities: List[str]
    risk_mitigation_steps: List[str]
    success_metrics: List[str]
    confidence_score: float
    rationale: str

@dataclass
class LeadershipAssessment:
    leader_id: str
    crisis_id: str
    assessment_time: datetime
    decision_quality_score: float
    communication_effectiveness: float
    stakeholder_confidence: float
    team_morale_impact: float
    crisis_resolution_speed: float
    overall_effectiveness: float
    strengths: List[str]
    improvement_areas: List[str]
    coaching_recommendations: List[str]

@dataclass
class CoachingGuidance:
    assessment_id: str
    focus_area: str
    current_performance: float
    target_performance: float
    improvement_strategies: List[str]
    practice_exercises: List[str]
    success_indicators: List[str]
    timeline: str
    resources: List[str]