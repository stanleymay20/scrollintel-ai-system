"""
Resource Mobilization Models for Crisis Leadership Excellence

Dataclass-based models for resource assessment, allocation, and coordination
during crisis situations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from uuid import uuid4


class ResourceType(Enum):
    """Types of resources that can be mobilized"""
    HUMAN_RESOURCES = "human_resources"
    TECHNICAL_INFRASTRUCTURE = "technical_infrastructure"
    FINANCIAL_CAPITAL = "financial_capital"
    EQUIPMENT_HARDWARE = "equipment_hardware"
    SOFTWARE_LICENSES = "software_licenses"
    EXTERNAL_SERVICES = "external_services"
    FACILITIES_SPACE = "facilities_space"
    COMMUNICATION_CHANNELS = "communication_channels"
    DATA_STORAGE = "data_storage"
    NETWORK_BANDWIDTH = "network_bandwidth"
    CLOUD_COMPUTE = "cloud_compute"
    EMERGENCY_SUPPLIES = "emergency_supplies"


class ResourceStatus(Enum):
    """Current status of resource availability"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"
    DEPLETED = "depleted"
    PENDING = "pending"


class ResourcePriority(Enum):
    """Priority levels for resource allocation"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class AllocationStatus(Enum):
    """Status of resource allocation"""
    REQUESTED = "requested"
    APPROVED = "approved"
    ALLOCATED = "allocated"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ResourceCapability:
    """Specific capability or skill of a resource"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    proficiency_level: float = 0.0  # 0.0 to 1.0
    certification_required: bool = False
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    """Individual resource that can be mobilized"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    resource_type: ResourceType = ResourceType.HUMAN_RESOURCES
    status: ResourceStatus = ResourceStatus.AVAILABLE
    capabilities: List[ResourceCapability] = field(default_factory=list)
    capacity: float = 1.0  # Maximum utilization (1.0 = 100%)
    current_utilization: float = 0.0  # Current utilization (0.0 to 1.0)
    location: str = ""
    cost_per_hour: float = 0.0
    availability_schedule: Dict[str, Any] = field(default_factory=dict)
    contact_info: Dict[str, str] = field(default_factory=dict)
    last_maintenance: Optional[datetime] = None
    next_maintenance: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourcePool:
    """Collection of related resources"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    resource_type: ResourceType = ResourceType.HUMAN_RESOURCES
    resources: List[Resource] = field(default_factory=list)
    total_capacity: float = 0.0
    available_capacity: float = 0.0
    reserved_capacity: float = 0.0
    pool_manager: str = ""
    access_restrictions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceRequirement:
    """Requirement for specific resources during crisis"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    resource_type: ResourceType = ResourceType.HUMAN_RESOURCES
    required_capabilities: List[str] = field(default_factory=list)
    quantity_needed: float = 1.0
    priority: ResourcePriority = ResourcePriority.MEDIUM
    duration_needed: timedelta = field(default_factory=lambda: timedelta(hours=8))
    location_preference: str = ""
    budget_limit: float = 0.0
    deadline: Optional[datetime] = None
    justification: str = ""
    requested_by: str = ""
    requested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceGap:
    """Identified gap between required and available resources"""
    id: str = field(default_factory=lambda: str(uuid4()))
    requirement_id: str = ""
    resource_type: ResourceType = ResourceType.HUMAN_RESOURCES
    gap_quantity: float = 0.0
    gap_capabilities: List[str] = field(default_factory=list)
    severity: ResourcePriority = ResourcePriority.MEDIUM
    impact_description: str = ""
    alternative_options: List[str] = field(default_factory=list)
    procurement_options: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    time_to_acquire: timedelta = field(default_factory=lambda: timedelta(hours=24))
    identified_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceAllocation:
    """Allocation of specific resources to crisis response"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    requirement_id: str = ""
    resource_id: str = ""
    allocated_quantity: float = 1.0
    allocation_priority: ResourcePriority = ResourcePriority.MEDIUM
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=8))
    actual_duration: Optional[timedelta] = None
    status: AllocationStatus = AllocationStatus.REQUESTED
    allocated_by: str = ""
    approved_by: str = ""
    cost_estimate: float = 0.0
    actual_cost: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceInventory:
    """Complete inventory of available resources and capabilities"""
    id: str = field(default_factory=lambda: str(uuid4()))
    assessment_time: datetime = field(default_factory=datetime.utcnow)
    total_resources: int = 0
    resources_by_type: Dict[ResourceType, int] = field(default_factory=dict)
    available_resources: List[Resource] = field(default_factory=list)
    allocated_resources: List[Resource] = field(default_factory=list)
    unavailable_resources: List[Resource] = field(default_factory=list)
    resource_pools: List[ResourcePool] = field(default_factory=list)
    total_capacity: Dict[ResourceType, float] = field(default_factory=dict)
    available_capacity: Dict[ResourceType, float] = field(default_factory=dict)
    utilization_rates: Dict[ResourceType, float] = field(default_factory=dict)
    critical_shortages: List[ResourceGap] = field(default_factory=list)
    upcoming_maintenance: List[Dict[str, Any]] = field(default_factory=list)
    cost_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class AllocationPlan:
    """Comprehensive plan for resource allocation during crisis"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    plan_name: str = ""
    requirements: List[ResourceRequirement] = field(default_factory=list)
    allocations: List[ResourceAllocation] = field(default_factory=list)
    identified_gaps: List[ResourceGap] = field(default_factory=list)
    total_cost_estimate: float = 0.0
    implementation_timeline: Dict[str, datetime] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    contingency_plans: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    created_by: str = ""
    approved_by: str = ""
    plan_status: str = "draft"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceUtilization:
    """Tracking of resource utilization and performance"""
    id: str = field(default_factory=lambda: str(uuid4()))
    resource_id: str = ""
    allocation_id: str = ""
    utilization_period: Dict[str, datetime] = field(default_factory=dict)
    planned_utilization: float = 0.0
    actual_utilization: float = 0.0
    efficiency_score: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    issues_encountered: List[str] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)
    cost_efficiency: float = 0.0
    recorded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExternalPartner:
    """External partner or vendor for resource coordination"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    partner_type: str = ""  # vendor, contractor, partner_organization, etc.
    contact_info: Dict[str, str] = field(default_factory=dict)
    available_resources: List[ResourceType] = field(default_factory=list)
    service_capabilities: List[str] = field(default_factory=list)
    response_time_sla: timedelta = field(default_factory=lambda: timedelta(hours=4))
    cost_structure: Dict[str, float] = field(default_factory=dict)
    reliability_score: float = 0.0
    last_engagement: Optional[datetime] = None
    contract_terms: Dict[str, Any] = field(default_factory=dict)
    emergency_contact: Dict[str, str] = field(default_factory=dict)
    activation_protocol: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExternalResourceRequest:
    """Request for resources from external partners"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    partner_id: str = ""
    resource_type: ResourceType = ResourceType.EXTERNAL_SERVICES
    requested_capabilities: List[str] = field(default_factory=list)
    quantity_requested: float = 1.0
    urgency_level: ResourcePriority = ResourcePriority.MEDIUM
    duration_needed: timedelta = field(default_factory=lambda: timedelta(hours=8))
    budget_approved: float = 0.0
    delivery_location: str = ""
    special_requirements: List[str] = field(default_factory=list)
    request_status: str = "pending"
    requested_by: str = ""
    approved_by: str = ""
    partner_response: Dict[str, Any] = field(default_factory=dict)
    estimated_delivery: Optional[datetime] = None
    actual_delivery: Optional[datetime] = None
    request_time: datetime = field(default_factory=datetime.utcnow)
    response_time: Optional[datetime] = None


@dataclass
class CoordinationProtocol:
    """Protocol for coordinating with external partners"""
    id: str = field(default_factory=lambda: str(uuid4()))
    partner_id: str = ""
    protocol_name: str = ""
    activation_triggers: List[str] = field(default_factory=list)
    communication_channels: List[str] = field(default_factory=list)
    escalation_procedures: List[str] = field(default_factory=list)
    resource_request_process: List[str] = field(default_factory=list)
    quality_standards: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: List[str] = field(default_factory=list)
    review_schedule: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""