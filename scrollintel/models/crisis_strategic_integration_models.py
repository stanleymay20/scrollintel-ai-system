"""
Crisis-Strategic Integration Models

This module defines data models for integrating crisis leadership capabilities
with strategic planning systems.
"""

from datetime import datetime, date
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field


class CrisisImpactLevel(str, Enum):
    """Crisis impact levels on strategic plans"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


class AdjustmentType(str, Enum):
    """Types of strategic adjustments during crisis"""
    RESOURCE_REALLOCATION = "resource_reallocation"
    MILESTONE_TIMELINE = "milestone_timeline"
    RISK_REBALANCING = "risk_rebalancing"
    COMMUNICATION_STRATEGY = "communication_strategy"
    TECHNOLOGY_PIVOT = "technology_pivot"
    MARKET_STRATEGY = "market_strategy"
    STAKEHOLDER_MANAGEMENT = "stakeholder_management"


class RecoveryPhase(str, Enum):
    """Phases of crisis recovery integration"""
    IMMEDIATE_STABILIZATION = "immediate_stabilization"
    STRATEGIC_REALIGNMENT = "strategic_realignment"
    MOMENTUM_RESTORATION = "momentum_restoration"
    ENHANCED_RESILIENCE = "enhanced_resilience"


class IntegrationStatus(str, Enum):
    """Status of crisis-strategic integration"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    MONITORING = "monitoring"
    RECOVERING = "recovering"
    COMPLETED = "completed"


@dataclass
class CrisisStrategicImpact:
    """Assessment of crisis impact on strategic plans"""
    crisis_id: str
    strategic_plan_id: str
    impact_level: CrisisImpactLevel
    affected_milestones: List[str]
    affected_technology_bets: List[str]
    resource_reallocation_needed: float  # Percentage of resources to reallocate
    timeline_adjustments: Dict[str, int]  # Milestone delays in days
    risk_level_changes: Dict[str, float]  # Risk level adjustments
    strategic_recommendations: List[str]
    recovery_timeline: int  # Days to recover strategic momentum
    confidence_score: float = 0.85  # Confidence in assessment accuracy
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


@dataclass
class CrisisAwareAdjustment:
    """Strategic adjustment recommendations during crisis"""
    adjustment_id: str
    crisis_id: str
    adjustment_type: AdjustmentType
    description: str
    priority: int  # 1-5, 1 being highest
    implementation_timeline: int  # Days to implement
    resource_requirements: Dict[str, Any]
    expected_benefits: List[str]
    risks: List[str]
    success_metrics: List[str]
    dependencies: List[str]
    approval_status: str = "pending"
    implementation_status: str = "not_started"
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


@dataclass
class RecoveryPhaseDetail:
    """Detailed information about a recovery phase"""
    phase: RecoveryPhase
    duration_days: int
    objectives: List[str]
    strategic_focus: str
    resource_allocation: float  # Percentage of resources allocated to crisis response
    key_activities: List[str]
    success_criteria: List[str]
    risks: List[str]
    dependencies: List[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    completion_percentage: float = 0.0


@dataclass
class RecoveryIntegrationPlan:
    """Integration plan for crisis recovery with long-term planning"""
    plan_id: str
    crisis_id: str
    strategic_roadmap_id: str
    recovery_phases: List[RecoveryPhaseDetail]
    milestone_realignment: Dict[str, datetime]
    resource_rebalancing: Dict[str, float]
    technology_bet_adjustments: List[Dict[str, Any]]
    stakeholder_communication_plan: Dict[str, Any]
    success_criteria: List[str]
    monitoring_framework: Dict[str, Any]
    status: IntegrationStatus = IntegrationStatus.ACTIVE
    progress_percentage: float = 0.0
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class StrategicResilienceMetric(BaseModel):
    """Metrics for measuring strategic resilience to crises"""
    metric_id: str
    name: str
    description: str
    current_value: float
    target_value: float
    measurement_unit: str
    trend: str  # "improving", "stable", "declining"
    last_updated: datetime
    data_source: str


class CrisisLearning(BaseModel):
    """Lessons learned from crisis-strategic integration"""
    learning_id: str
    crisis_id: str
    category: str  # "process", "technology", "communication", "resource_management"
    lesson_description: str
    impact_on_strategy: str
    recommended_actions: List[str]
    implementation_priority: int
    status: str = "identified"
    created_at: datetime = Field(default_factory=datetime.now)


class IntegrationSimulation(BaseModel):
    """Simulation of crisis-strategic integration scenarios"""
    simulation_id: str
    name: str
    description: str
    crisis_scenarios: List[Dict[str, Any]]
    strategic_context: Dict[str, Any]
    simulation_parameters: Dict[str, Any]
    results: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence_level: float = Field(ge=0, le=1)
    created_at: datetime = Field(default_factory=datetime.now)


class StakeholderImpactAssessment(BaseModel):
    """Assessment of crisis impact on different stakeholders"""
    stakeholder_group: str
    impact_level: CrisisImpactLevel
    specific_concerns: List[str]
    communication_needs: List[str]
    engagement_strategy: str
    confidence_level: float = Field(ge=0, le=1)
    recovery_expectations: Dict[str, Any]


class StrategicContinuityPlan(BaseModel):
    """Plan for maintaining strategic continuity during crisis"""
    plan_id: str
    crisis_type: str
    critical_strategic_initiatives: List[str]
    minimum_viable_operations: List[str]
    resource_preservation_strategy: Dict[str, Any]
    stakeholder_priorities: List[str]
    decision_making_protocols: Dict[str, Any]
    communication_frameworks: Dict[str, Any]
    recovery_triggers: List[str]
    success_metrics: List[str]


class CrisisStrategicDashboard(BaseModel):
    """Dashboard data for crisis-strategic integration monitoring"""
    dashboard_id: str
    crisis_id: str
    strategic_plan_id: str
    current_status: IntegrationStatus
    key_metrics: Dict[str, float]
    active_adjustments: int
    recovery_progress: float
    stakeholder_confidence: float
    resource_utilization: Dict[str, float]
    timeline_variance: Dict[str, int]
    risk_indicators: List[Dict[str, Any]]
    recent_activities: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    last_updated: datetime = Field(default_factory=datetime.now)


class IntegrationPerformanceMetrics(BaseModel):
    """Performance metrics for crisis-strategic integration"""
    integration_id: str
    crisis_resolution_time: int  # days
    strategic_momentum_recovery_time: int  # days
    resource_reallocation_efficiency: float
    stakeholder_satisfaction_score: float
    milestone_achievement_rate: float
    risk_mitigation_effectiveness: float
    communication_effectiveness: float
    overall_integration_score: float
    benchmarking_data: Dict[str, float]
    improvement_areas: List[str]


class CrisisStrategicAlert(BaseModel):
    """Alert for crisis-strategic integration issues"""
    alert_id: str
    crisis_id: str
    alert_type: str  # "timeline_deviation", "resource_constraint", "stakeholder_concern"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_components: List[str]
    recommended_actions: List[str]
    escalation_required: bool
    assigned_to: Optional[str] = None
    status: str = "open"
    created_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


class StrategicRiskProfile(BaseModel):
    """Risk profile for strategic initiatives during crisis"""
    profile_id: str
    strategic_initiative_id: str
    crisis_context: str
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    contingency_plans: List[str]
    monitoring_indicators: List[str]
    risk_appetite: float = Field(ge=0, le=1)
    risk_tolerance: float = Field(ge=0, le=1)
    last_assessment: datetime = Field(default_factory=datetime.now)


class CrisisStrategicWorkflow(BaseModel):
    """Workflow for crisis-strategic integration processes"""
    workflow_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    workflow_steps: List[Dict[str, Any]]
    approval_requirements: List[str]
    automation_level: str  # "manual", "semi_automated", "fully_automated"
    execution_time: int  # estimated minutes
    success_rate: float = Field(ge=0, le=1)
    last_executed: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


# Request/Response Models for API

class CrisisImpactAssessmentRequest(BaseModel):
    """Request model for crisis impact assessment"""
    crisis_data: Dict[str, Any]
    strategic_roadmap_data: Dict[str, Any]
    assessment_parameters: Optional[Dict[str, Any]] = None


class CrisisImpactAssessmentResponse(BaseModel):
    """Response model for crisis impact assessment"""
    status: str
    data: Dict[str, Any]
    message: str
    assessment_id: str
    confidence_score: float


class CrisisAdjustmentRequest(BaseModel):
    """Request model for generating crisis adjustments"""
    crisis_data: Dict[str, Any]
    strategic_roadmap_data: Dict[str, Any]
    impact_assessment_data: Dict[str, Any]
    adjustment_preferences: Optional[Dict[str, Any]] = None


class CrisisAdjustmentResponse(BaseModel):
    """Response model for crisis adjustments"""
    status: str
    data: Dict[str, Any]
    message: str
    adjustments_count: int
    total_implementation_time: int


class RecoveryPlanRequest(BaseModel):
    """Request model for recovery integration plan"""
    crisis_data: Dict[str, Any]
    strategic_roadmap_data: Dict[str, Any]
    impact_assessment_data: Dict[str, Any]
    recovery_preferences: Optional[Dict[str, Any]] = None


class RecoveryPlanResponse(BaseModel):
    """Response model for recovery integration plan"""
    status: str
    data: Dict[str, Any]
    message: str
    plan_id: str
    estimated_recovery_time: int


class IntegrationStatusResponse(BaseModel):
    """Response model for integration status"""
    status: str
    data: Dict[str, Any]
    message: str
    last_updated: datetime


class SimulationRequest(BaseModel):
    """Request model for integration simulation"""
    simulation_name: str
    crisis_scenarios: List[Dict[str, Any]]
    strategic_context: Dict[str, Any]
    simulation_parameters: Dict[str, Any]


class SimulationResponse(BaseModel):
    """Response model for integration simulation"""
    status: str
    data: Dict[str, Any]
    message: str
    simulation_id: str
    confidence_level: float