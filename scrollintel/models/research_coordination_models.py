"""
Research Coordination Models for Autonomous Innovation Lab

This module defines the data models for autonomous research project management,
milestone tracking, resource coordination, and research collaboration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set
from enum import Enum
import uuid


@dataclass
class ResearchTopic:
    """Research topic dataclass for coordination system"""
    title: str = ""
    description: str = ""
    domain: str = ""
    research_questions: List[str] = field(default_factory=list)
    methodology: str = ""
    keywords: List[str] = field(default_factory=list)
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    impact_potential: float = 0.0


@dataclass
class Hypothesis:
    """Research hypothesis dataclass for coordination system"""
    statement: str = ""
    confidence: float = 0.0
    testable: bool = True
    variables: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""


class ProjectStatus(Enum):
    """Research project status enumeration"""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MilestoneStatus(Enum):
    """Research milestone status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    BLOCKED = "blocked"


class ResourceType(Enum):
    """Research resource type enumeration"""
    COMPUTATIONAL = "computational"
    DATA = "data"
    HUMAN = "human"
    EQUIPMENT = "equipment"
    BUDGET = "budget"


class CollaborationType(Enum):
    """Research collaboration type enumeration"""
    KNOWLEDGE_SHARING = "knowledge_sharing"
    RESOURCE_SHARING = "resource_sharing"
    JOINT_RESEARCH = "joint_research"
    PEER_REVIEW = "peer_review"


@dataclass
class ResearchResource:
    """Research resource allocation model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.COMPUTATIONAL
    name: str = ""
    description: str = ""
    capacity: float = 0.0
    allocated: float = 0.0
    available: float = 0.0
    cost_per_unit: float = 0.0
    allocation_start: Optional[datetime] = None
    allocation_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.resource_type.value}_resource_{self.id[:8]}"
        self.available = self.capacity - self.allocated


@dataclass
class ResearchMilestone:
    """Research milestone tracking model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    name: str = ""
    description: str = ""
    status: MilestoneStatus = MilestoneStatus.PENDING
    planned_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    risk_level: str = "low"
    assigned_resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_overdue(self) -> bool:
        """Check if milestone is overdue"""
        if self.planned_end and self.status not in [MilestoneStatus.COMPLETED]:
            return datetime.now() > self.planned_end
        return False
    
    def get_duration_days(self) -> Optional[int]:
        """Get planned duration in days"""
        if self.planned_start and self.planned_end:
            return (self.planned_end - self.planned_start).days
        return None


@dataclass
class ResearchProject:
    """Autonomous research project model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: ProjectStatus = ProjectStatus.PLANNING
    priority: int = 1  # 1-10 scale
    research_domain: str = ""
    objectives: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    methodology: str = ""
    
    # Timeline
    planned_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    
    # Resources
    allocated_resources: List[ResearchResource] = field(default_factory=list)
    budget_allocated: float = 0.0
    budget_used: float = 0.0
    
    # Milestones and tracking
    milestones: List[ResearchMilestone] = field(default_factory=list)
    progress_percentage: float = 0.0
    
    # Collaboration
    collaborating_projects: List[str] = field(default_factory=list)
    knowledge_dependencies: List[str] = field(default_factory=list)
    
    # Results and outputs
    research_outputs: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    patents: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Research_Project_{self.id[:8]}"
    
    def get_active_milestones(self) -> List[ResearchMilestone]:
        """Get currently active milestones"""
        return [m for m in self.milestones if m.status == MilestoneStatus.IN_PROGRESS]
    
    def get_overdue_milestones(self) -> List[ResearchMilestone]:
        """Get overdue milestones"""
        return [m for m in self.milestones if m.is_overdue()]
    
    def calculate_progress(self) -> float:
        """Calculate overall project progress"""
        if not self.milestones:
            return 0.0
        
        total_progress = sum(m.progress_percentage for m in self.milestones)
        return total_progress / len(self.milestones)
    
    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get resource utilization by type"""
        utilization = {}
        for resource in self.allocated_resources:
            if resource.resource_type not in utilization:
                utilization[resource.resource_type] = 0.0
            if resource.capacity > 0:
                utilization[resource.resource_type] += (resource.allocated / resource.capacity) * 100
        return utilization


@dataclass
class ResearchCollaboration:
    """Research collaboration model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    collaboration_type: CollaborationType = CollaborationType.KNOWLEDGE_SHARING
    primary_project_id: str = ""
    collaborating_project_ids: List[str] = field(default_factory=list)
    
    # Collaboration details
    shared_resources: List[str] = field(default_factory=list)
    shared_knowledge: List[str] = field(default_factory=list)
    joint_objectives: List[str] = field(default_factory=list)
    
    # Coordination
    coordination_frequency: str = "weekly"  # daily, weekly, monthly
    communication_channels: List[str] = field(default_factory=list)
    
    # Synergy tracking
    synergy_score: float = 0.0
    knowledge_transfer_rate: float = 0.0
    resource_efficiency_gain: float = 0.0
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeAsset:
    """Research knowledge asset model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    content: str = ""
    asset_type: str = "research_finding"  # research_finding, methodology, dataset, model
    
    # Source information
    source_project_id: str = ""
    authors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Knowledge classification
    domain: str = ""
    keywords: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_status: str = "pending"  # pending, validated, disputed
    
    # Usage tracking
    access_count: int = 0
    citation_count: int = 0
    reuse_count: int = 0
    
    # Relationships
    related_assets: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSynergy:
    """Research synergy identification model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_ids: List[str] = field(default_factory=list)
    synergy_type: str = "knowledge_complementarity"
    
    # Synergy metrics
    potential_score: float = 0.0
    feasibility_score: float = 0.0
    impact_score: float = 0.0
    overall_score: float = 0.0
    
    # Synergy details
    complementary_strengths: List[str] = field(default_factory=list)
    shared_challenges: List[str] = field(default_factory=list)
    collaboration_opportunities: List[str] = field(default_factory=list)
    
    # Implementation
    recommended_actions: List[str] = field(default_factory=list)
    estimated_benefits: Dict[str, float] = field(default_factory=dict)
    implementation_complexity: str = "medium"  # low, medium, high
    
    # Status
    is_exploited: bool = False
    exploitation_results: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchCoordinationMetrics:
    """Research coordination performance metrics"""
    total_projects: int = 0
    active_projects: int = 0
    completed_projects: int = 0
    
    # Resource metrics
    total_resources: int = 0
    resource_utilization_rate: float = 0.0
    resource_efficiency_score: float = 0.0
    
    # Milestone metrics
    total_milestones: int = 0
    completed_milestones: int = 0
    overdue_milestones: int = 0
    milestone_completion_rate: float = 0.0
    
    # Collaboration metrics
    active_collaborations: int = 0
    knowledge_sharing_rate: float = 0.0
    synergy_exploitation_rate: float = 0.0
    
    # Performance metrics
    average_project_duration: float = 0.0
    success_rate: float = 0.0
    innovation_output_rate: float = 0.0
    
    # Timestamp
    calculated_at: datetime = field(default_factory=datetime.now)