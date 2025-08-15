"""
Crisis Leadership Excellence Data Models

Pydantic models for the crisis leadership excellence system API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum


class CrisisSignalType(str, Enum):
    SYSTEM_ALERT = "system_alert"
    SECURITY_INCIDENT = "security_incident"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USER_COMPLAINT = "user_complaint"
    MEDIA_ATTENTION = "media_attention"
    REGULATORY_INQUIRY = "regulatory_inquiry"
    FINANCIAL_ANOMALY = "financial_anomaly"
    OPERATIONAL_FAILURE = "operational_failure"
    EXTERNAL_THREAT = "external_threat"
    STAKEHOLDER_CONCERN = "stakeholder_concern"


class CrisisSeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class CrisisSignal(BaseModel):
    """Individual crisis signal"""
    signal_type: CrisisSignalType
    severity: CrisisSeverityLevel
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrisisSignalRequest(BaseModel):
    """Request to handle a crisis situation"""
    crisis_type: str
    description: str
    signals: List[CrisisSignal]
    priority: str = "high"
    requester: str
    context: Dict[str, Any] = Field(default_factory=dict)


class CrisisResponseModel(BaseModel):
    """Complete crisis response model"""
    crisis_id: str
    response_plan: Dict[str, Any]
    team_formation: Dict[str, Any]
    resource_allocation: Dict[str, Any]
    communication_strategy: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    contingency_plans: List[Dict[str, Any]]
    stakeholder_updates: List[Dict[str, Any]]
    status: str
    created_at: datetime


class ValidationScenario(BaseModel):
    """Crisis validation scenario"""
    scenario_id: str
    crisis_type: str
    signals: List[CrisisSignal]
    expected_outcomes: List[str]
    difficulty_level: str = "medium"


class ValidationRequest(BaseModel):
    """Request to validate crisis response capability"""
    scenarios: List[ValidationScenario]
    validation_type: str = "comprehensive"
    success_threshold: float = 0.8


class DeploymentRequest(BaseModel):
    """Request to deploy crisis leadership system"""
    deployment_environment: str = "production"
    enable_monitoring: bool = True
    enable_learning: bool = True
    configuration: Dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    """System status response model"""
    system_name: str
    status: str
    active_crises: int
    total_crises_handled: int
    average_response_time: float
    success_rate: float
    stakeholder_satisfaction: float
    system_readiness: float
    last_updated: datetime


class CrisisMetrics(BaseModel):
    """Crisis performance metrics"""
    crisis_id: str
    response_time: float
    resolution_time: Optional[float]
    effectiveness_score: float
    stakeholder_satisfaction: float
    team_performance: float
    resource_efficiency: float
    communication_quality: float
    learning_value: float


class LeadershipAction(BaseModel):
    """Crisis leadership action"""
    action_id: str
    action_type: str
    description: str
    priority: str
    assigned_to: str
    deadline: datetime
    status: str
    outcomes: List[str] = Field(default_factory=list)


class StakeholderUpdate(BaseModel):
    """Stakeholder communication update"""
    update_id: str
    stakeholder_group: str
    message: str
    channel: str
    timestamp: datetime
    delivery_status: str
    response_required: bool = False


class ResourceAllocation(BaseModel):
    """Crisis resource allocation"""
    resource_id: str
    resource_type: str
    allocated_amount: float
    allocation_priority: str
    assigned_team: str
    allocation_time: datetime
    expected_duration: int  # minutes
    utilization_rate: float


class TeamAssignment(BaseModel):
    """Crisis team assignment"""
    assignment_id: str
    team_member: str
    role: str
    responsibilities: List[str]
    skills_required: List[str]
    availability: str
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class CommunicationPlan(BaseModel):
    """Crisis communication plan"""
    plan_id: str
    target_audiences: List[str]
    key_messages: Dict[str, str]
    communication_channels: List[str]
    timeline: List[Dict[str, Any]]
    approval_workflow: List[str]
    effectiveness_metrics: Dict[str, float] = Field(default_factory=dict)


class CrisisLearning(BaseModel):
    """Crisis learning and improvement data"""
    crisis_id: str
    lessons_learned: List[str]
    improvement_recommendations: List[str]
    best_practices: List[str]
    areas_for_improvement: List[str]
    knowledge_updates: Dict[str, Any] = Field(default_factory=dict)


class SystemCapability(BaseModel):
    """System capability assessment"""
    capability_name: str
    current_level: float
    target_level: float
    improvement_plan: List[str]
    assessment_date: datetime
    next_review_date: datetime


class CrisisSimulation(BaseModel):
    """Crisis simulation configuration"""
    simulation_id: str
    simulation_name: str
    scenarios: List[ValidationScenario]
    objectives: List[str]
    success_criteria: Dict[str, float]
    participants: List[str]
    duration: int  # minutes


class PerformanceReport(BaseModel):
    """Crisis performance report"""
    report_id: str
    reporting_period: str
    total_crises: int
    successful_resolutions: int
    average_response_time: float
    stakeholder_satisfaction: float
    key_achievements: List[str]
    areas_for_improvement: List[str]
    recommendations: List[str]
    generated_at: datetime


class CrisisAlert(BaseModel):
    """Crisis alert notification"""
    alert_id: str
    alert_type: str
    severity: CrisisSeverityLevel
    message: str
    affected_systems: List[str]
    recommended_actions: List[str]
    escalation_required: bool
    created_at: datetime


class RecoveryPlan(BaseModel):
    """Crisis recovery plan"""
    plan_id: str
    crisis_id: str
    recovery_objectives: List[str]
    recovery_steps: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    contingency_measures: List[str]


class StakeholderConfidence(BaseModel):
    """Stakeholder confidence metrics"""
    stakeholder_group: str
    confidence_level: float
    confidence_factors: Dict[str, float]
    concerns: List[str]
    improvement_actions: List[str]
    measurement_date: datetime


class CrisisKnowledge(BaseModel):
    """Crisis knowledge base entry"""
    knowledge_id: str
    crisis_type: str
    best_practices: List[str]
    common_pitfalls: List[str]
    recommended_responses: List[Dict[str, Any]]
    success_patterns: List[str]
    learning_sources: List[str]
    last_updated: datetime


class IntegrationStatus(BaseModel):
    """System integration status"""
    component_name: str
    integration_status: str
    last_health_check: datetime
    performance_metrics: Dict[str, float]
    issues: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class CrisisLeadershipConfig(BaseModel):
    """Crisis leadership system configuration"""
    system_name: str = "Crisis Leadership Excellence"
    response_time_target: int = 300  # seconds
    success_rate_target: float = 0.95
    stakeholder_satisfaction_target: float = 0.9
    learning_enabled: bool = True
    monitoring_enabled: bool = True
    integration_endpoints: List[str] = Field(default_factory=list)
    notification_channels: List[str] = Field(default_factory=list)