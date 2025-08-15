"""
Crisis Preparedness Models

Data models for crisis preparedness assessment and enhancement system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class PreparednessLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    CRITICAL = "critical"


class SimulationType(Enum):
    SYSTEM_OUTAGE = "system_outage"
    SECURITY_BREACH = "security_breach"
    DATA_LOSS = "data_loss"
    NATURAL_DISASTER = "natural_disaster"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    REGULATORY_INVESTIGATION = "regulatory_investigation"
    REPUTATION_CRISIS = "reputation_crisis"
    FINANCIAL_CRISIS = "financial_crisis"


class TrainingType(Enum):
    TABLETOP_EXERCISE = "tabletop_exercise"
    SIMULATION_DRILL = "simulation_drill"
    ROLE_PLAYING = "role_playing"
    DECISION_MAKING_WORKSHOP = "decision_making_workshop"
    COMMUNICATION_TRAINING = "communication_training"
    LEADERSHIP_DEVELOPMENT = "leadership_development"


class CapabilityArea(Enum):
    CRISIS_DETECTION = "crisis_detection"
    DECISION_MAKING = "decision_making"
    COMMUNICATION = "communication"
    RESOURCE_MOBILIZATION = "resource_mobilization"
    TEAM_COORDINATION = "team_coordination"
    STAKEHOLDER_MANAGEMENT = "stakeholder_management"
    RECOVERY_PLANNING = "recovery_planning"


@dataclass
class PreparednessAssessment:
    """Crisis preparedness assessment"""
    id: str
    assessment_date: datetime
    assessor_id: str
    
    # Overall assessment
    overall_preparedness_level: PreparednessLevel
    overall_score: float
    
    # Capability assessments
    capability_scores: Dict[CapabilityArea, float]
    capability_levels: Dict[CapabilityArea, PreparednessLevel]
    
    # Detailed findings
    strengths: List[str]
    weaknesses: List[str]
    gaps_identified: List[str]
    
    # Risk assessment
    high_risk_scenarios: List[str]
    vulnerability_areas: List[str]
    
    # Recommendations
    improvement_priorities: List[str]
    recommended_actions: List[str]
    
    # Assessment metadata
    assessment_methodology: str
    data_sources: List[str]
    confidence_level: float


@dataclass
class CrisisSimulation:
    """Crisis simulation exercise"""
    id: str
    simulation_type: SimulationType
    title: str
    description: str
    
    # Simulation parameters
    scenario_details: str
    complexity_level: str
    duration_minutes: int
    participants: List[str]
    
    # Objectives
    learning_objectives: List[str]
    success_criteria: List[str]
    
    # Execution details
    facilitator_id: str
    scheduled_date: datetime
    actual_start_time: Optional[datetime]
    actual_end_time: Optional[datetime]
    
    # Results
    participant_performance: Dict[str, float]
    objectives_achieved: List[str]
    lessons_learned: List[str]
    improvement_areas: List[str]
    
    # Simulation metadata
    simulation_status: str
    feedback_collected: bool
    report_generated: bool


@dataclass
class TrainingProgram:
    """Crisis response training program"""
    id: str
    program_name: str
    training_type: TrainingType
    description: str
    
    # Program structure
    modules: List[str]
    duration_hours: float
    target_audience: List[str]
    prerequisites: List[str]
    
    # Learning outcomes
    learning_objectives: List[str]
    competencies_developed: List[str]
    assessment_methods: List[str]
    
    # Delivery details
    delivery_method: str
    instructor_requirements: List[str]
    materials_needed: List[str]
    
    # Program metadata
    created_date: datetime
    last_updated: datetime
    version: str
    approval_status: str


@dataclass
class CapabilityDevelopment:
    """Crisis response capability development plan"""
    id: str
    capability_area: CapabilityArea
    current_level: PreparednessLevel
    target_level: PreparednessLevel
    
    # Development plan
    development_objectives: List[str]
    improvement_actions: List[str]
    training_requirements: List[str]
    resource_needs: List[str]
    
    # Timeline
    start_date: datetime
    target_completion_date: datetime
    milestones: List[Dict[str, Any]]
    
    # Progress tracking
    current_progress: float
    completed_actions: List[str]
    pending_actions: List[str]
    
    # Success metrics
    success_indicators: List[str]
    measurement_methods: List[str]
    
    # Development metadata
    responsible_team: str
    budget_allocated: float
    status: str


@dataclass
class SimulationResult:
    """Results from crisis simulation"""
    simulation_id: str
    participant_id: str
    
    # Performance metrics
    overall_performance_score: float
    decision_making_score: float
    communication_score: float
    leadership_score: float
    teamwork_score: float
    
    # Specific assessments
    response_time: float
    decision_quality: float
    stress_management: float
    adaptability: float
    
    # Feedback
    strengths_demonstrated: List[str]
    areas_for_improvement: List[str]
    specific_feedback: str
    
    # Development recommendations
    recommended_training: List[str]
    skill_development_priorities: List[str]
    
    # Result metadata
    assessment_date: datetime
    assessor_id: str
    confidence_level: float


@dataclass
class PreparednessReport:
    """Comprehensive preparedness report"""
    id: str
    report_title: str
    report_type: str
    
    # Report content
    executive_summary: str
    current_state_assessment: str
    gap_analysis: str
    improvement_recommendations: str
    implementation_roadmap: str
    
    # Supporting data
    assessment_data: Dict[str, Any]
    simulation_results: List[Dict[str, Any]]
    training_outcomes: List[Dict[str, Any]]
    
    # Report metadata
    generated_date: datetime
    report_period: str
    author_id: str
    review_status: str