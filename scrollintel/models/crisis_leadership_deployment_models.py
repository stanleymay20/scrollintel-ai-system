"""
Data Models for Crisis Leadership Excellence Deployment

Defines Pydantic models for deployment requests, responses, and system metrics.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class ValidationLevelEnum(str, Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    STRESS_TEST = "stress_test"
    PRODUCTION_READY = "production_ready"


class DeploymentStatusEnum(str, Enum):
    INITIALIZING = "initializing"
    VALIDATING = "validating"
    TESTING = "testing"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"


class DeploymentRequest(BaseModel):
    """Request model for crisis leadership system deployment"""
    validation_level: ValidationLevelEnum = Field(
        default=ValidationLevelEnum.COMPREHENSIVE,
        description="Level of validation to perform during deployment"
    )
    force_deployment: bool = Field(
        default=False,
        description="Force deployment even if validation fails"
    )
    skip_stress_testing: bool = Field(
        default=False,
        description="Skip stress testing for faster deployment"
    )
    custom_scenarios: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Custom crisis scenarios for testing"
    )


class ValidationRequest(BaseModel):
    """Request model for crisis leadership excellence validation"""
    validation_scenarios: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Custom validation scenarios to test"
    )
    include_stress_tests: bool = Field(
        default=True,
        description="Include stress testing scenarios"
    )
    target_success_rate: float = Field(
        default=0.8,
        description="Target success rate for validation",
        ge=0.0,
        le=1.0
    )


class ComponentHealthMetrics(BaseModel):
    """Component health metrics"""
    crisis_detector: float = Field(ge=0.0, le=1.0)
    decision_engine: float = Field(ge=0.0, le=1.0)
    info_synthesizer: float = Field(ge=0.0, le=1.0)
    risk_analyzer: float = Field(ge=0.0, le=1.0)
    stakeholder_notifier: float = Field(ge=0.0, le=1.0)
    message_coordinator: float = Field(ge=0.0, le=1.0)
    media_manager: float = Field(ge=0.0, le=1.0)
    resource_assessor: float = Field(ge=0.0, le=1.0)
    resource_allocator: float = Field(ge=0.0, le=1.0)
    external_coordinator: float = Field(ge=0.0, le=1.0)
    team_former: float = Field(ge=0.0, le=1.0)
    role_assigner: float = Field(ge=0.0, le=1.0)
    performance_monitor: float = Field(ge=0.0, le=1.0)
    post_crisis_analyzer: float = Field(ge=0.0, le=1.0)
    preparedness_engine: float = Field(ge=0.0, le=1.0)
    recovery_planner: float = Field(ge=0.0, le=1.0)
    resilience_engine: float = Field(ge=0.0, le=1.0)
    leadership_guide: float = Field(ge=0.0, le=1.0)
    confidence_manager: float = Field(ge=0.0, le=1.0)
    effectiveness_tester: float = Field(ge=0.0, le=1.0)
    strategic_integrator: float = Field(ge=0.0, le=1.0)
    communication_integrator: float = Field(ge=0.0, le=1.0)


class IntegrationScores(BaseModel):
    """Integration test scores"""
    crisis_detection_to_decision: float = Field(ge=0.0, le=1.0)
    decision_to_communication: float = Field(ge=0.0, le=1.0)
    communication_to_execution: float = Field(ge=0.0, le=1.0)
    execution_to_monitoring: float = Field(ge=0.0, le=1.0)
    monitoring_to_learning: float = Field(ge=0.0, le=1.0)


class CrisisResponseCapabilities(BaseModel):
    """Crisis response capability scores"""
    system_outage_response: float = Field(ge=0.0, le=1.0)
    security_breach_response: float = Field(ge=0.0, le=1.0)
    financial_crisis_response: float = Field(ge=0.0, le=1.0)
    multi_crisis_handling: float = Field(ge=0.0, le=1.0)
    stakeholder_management: float = Field(ge=0.0, le=1.0)
    communication_excellence: float = Field(ge=0.0, le=1.0)
    leadership_effectiveness: float = Field(ge=0.0, le=1.0)


class PerformanceBenchmarks(BaseModel):
    """Performance benchmark scores"""
    crisis_detection_speed: float = Field(ge=0.0, le=1.0)
    decision_making_speed: float = Field(ge=0.0, le=1.0)
    communication_speed: float = Field(ge=0.0, le=1.0)
    resource_allocation_speed: float = Field(ge=0.0, le=1.0)
    overall_response_time: float = Field(ge=0.0, le=1.0)
    system_throughput: float = Field(ge=0.0, le=1.0)
    memory_efficiency: float = Field(ge=0.0, le=1.0)
    scalability_score: float = Field(ge=0.0, le=1.0)


class ContinuousLearningMetrics(BaseModel):
    """Continuous learning system metrics"""
    learning_system_active: bool
    data_collection_rate: float = Field(ge=0.0, le=1.0)
    pattern_recognition_accuracy: float = Field(ge=0.0, le=1.0)
    improvement_identification: float = Field(ge=0.0, le=1.0)
    adaptation_speed: float = Field(ge=0.0, le=1.0)
    knowledge_retention: float = Field(ge=0.0, le=1.0)


class DeploymentResponse(BaseModel):
    """Response model for crisis leadership system deployment"""
    deployment_id: str
    status: DeploymentStatusEnum
    deployment_timestamp: datetime
    validation_level: ValidationLevelEnum
    overall_readiness_score: float = Field(ge=0.0, le=1.0)
    deployment_success: bool
    component_health: Dict[str, float]
    integration_scores: Dict[str, float]
    crisis_response_capabilities: Dict[str, float]
    performance_benchmarks: Dict[str, float]
    continuous_learning_metrics: Dict[str, Union[bool, float]]
    message: str


class ValidationScenarioResult(BaseModel):
    """Result of individual validation scenario"""
    scenario_id: str
    crisis_type: str
    response_time: float
    effectiveness_score: float = Field(ge=0.0, le=1.0)
    success: bool
    leadership_score: float = Field(ge=0.0, le=1.0)
    stakeholder_satisfaction: float = Field(ge=0.0, le=1.0)
    error: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for crisis leadership excellence validation"""
    validation_id: str
    validation_timestamp: datetime
    scenarios_tested: int
    overall_success_rate: float = Field(ge=0.0, le=1.0)
    average_response_time: float
    leadership_effectiveness: float = Field(ge=0.0, le=1.0)
    stakeholder_satisfaction: float = Field(ge=0.0, le=1.0)
    crisis_type_performance: Dict[str, float]
    detailed_results: List[ValidationScenarioResult]
    validation_success: bool
    message: str


class SystemHealthSummary(BaseModel):
    """System health summary"""
    overall_readiness: float = Field(ge=0.0, le=1.0)
    component_health_avg: float = Field(ge=0.0, le=1.0)
    integration_health_avg: float = Field(ge=0.0, le=1.0)
    crisis_capabilities_avg: float = Field(ge=0.0, le=1.0)
    performance_avg: float = Field(ge=0.0, le=1.0)
    learning_system_health: bool


class SystemHealthResponse(BaseModel):
    """Response model for system health status"""
    deployment_status: DeploymentStatusEnum
    deployment_metrics: Optional[Dict[str, Any]]
    validation_history_count: int
    learning_data_points: int
    system_health: Union[SystemHealthSummary, Dict[str, str]]
    last_updated: datetime
    message: str


class BestPerformingScenario(BaseModel):
    """Best performing crisis scenario"""
    scenario_id: str
    effectiveness_score: float = Field(ge=0.0, le=1.0)
    response_time: float


class LearningInsightsResponse(BaseModel):
    """Response model for continuous learning insights"""
    insights_timestamp: datetime
    total_crises_handled: int
    average_response_time: float
    average_effectiveness: float = Field(ge=0.0, le=1.0)
    improvement_trend: str
    best_performing_scenarios: List[BestPerformingScenario]
    improvement_opportunities: List[str]
    learning_system_active: bool
    message: str


class CrisisTestSignal(BaseModel):
    """Crisis test signal model"""
    type: str
    severity: str
    description: Optional[str] = None
    affected_systems: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class CrisisTestRequest(BaseModel):
    """Request model for crisis response testing"""
    test_name: str
    crisis_signals: List[CrisisTestSignal]
    expected_response_time: Optional[float] = None
    success_criteria: Optional[Dict[str, float]] = None


class CrisisTestResponse(BaseModel):
    """Response model for crisis response testing"""
    test_id: str
    test_name: str
    response_time: float
    crisis_id: str
    response_plan: Dict[str, Any]
    team_formation: Dict[str, Any]
    resource_allocation: Dict[str, Any]
    communication_strategy: Dict[str, Any]
    success_metrics: Dict[str, float]
    test_success: bool
    message: str


class StressTestResponse(BaseModel):
    """Response model for stress testing"""
    stress_test_id: str
    deployment_readiness: float = Field(ge=0.0, le=1.0)
    stress_test_success_rate: float = Field(ge=0.0, le=1.0)
    average_response_time_under_stress: float
    leadership_effectiveness_under_stress: float = Field(ge=0.0, le=1.0)
    scenarios_tested: int
    stress_test_passed: bool
    message: str


class DeploymentHistoryEntry(BaseModel):
    """Deployment history entry"""
    deployment_timestamp: datetime
    validation_level: ValidationLevelEnum
    overall_readiness_score: float = Field(ge=0.0, le=1.0)
    deployment_success: bool
    component_health_avg: float = Field(ge=0.0, le=1.0)
    crisis_capabilities_avg: float = Field(ge=0.0, le=1.0)


class DeploymentHistoryResponse(BaseModel):
    """Response model for deployment history"""
    deployment_history: List[DeploymentHistoryEntry]
    total_deployments: int
    latest_deployment: Optional[DeploymentHistoryEntry]
    message: str


class EmergencyDeploymentResponse(BaseModel):
    """Response model for emergency deployment"""
    emergency_deployment_id: str
    deployment_status: DeploymentStatusEnum
    readiness_score: float = Field(ge=0.0, le=1.0)
    deployment_success: bool
    warning: str
    recommendation: str
    message: str


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: datetime
    deployment_system_status: DeploymentStatusEnum
    system_health: Union[SystemHealthSummary, Dict[str, str]]
    error: Optional[str] = None
    message: str