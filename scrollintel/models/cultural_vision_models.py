"""
Cultural Vision Development Models

Data models for cultural vision creation, alignment, and communication systems.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class VisionScope(Enum):
    """Scope of cultural vision"""
    ORGANIZATIONAL = "organizational"
    DEPARTMENTAL = "departmental"
    TEAM = "team"
    PROJECT = "project"


class AlignmentLevel(Enum):
    """Level of alignment with strategic objectives"""
    FULLY_ALIGNED = "fully_aligned"
    MOSTLY_ALIGNED = "mostly_aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    MISALIGNED = "misaligned"


class StakeholderType(Enum):
    """Type of stakeholder for buy-in"""
    EXECUTIVE = "executive"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    CUSTOMER = "customer"
    PARTNER = "partner"
    INVESTOR = "investor"


@dataclass
class CulturalValue:
    """Represents a cultural value"""
    name: str
    description: str
    behavioral_indicators: List[str]
    importance_score: float
    measurability: float
    
    def __post_init__(self):
        if not 0 <= self.importance_score <= 1:
            raise ValueError("Importance score must be between 0 and 1")
        if not 0 <= self.measurability <= 1:
            raise ValueError("Measurability must be between 0 and 1")


@dataclass
class StrategicObjective:
    """Strategic objective for alignment"""
    id: str
    title: str
    description: str
    priority: int
    target_date: datetime
    success_metrics: List[str]
    cultural_requirements: List[str] = field(default_factory=list)


@dataclass
class CulturalVision:
    """Comprehensive cultural vision"""
    id: str
    organization_id: str
    title: str
    vision_statement: str
    mission_alignment: str
    core_values: List[CulturalValue]
    scope: VisionScope
    target_behaviors: List[str]
    success_indicators: List[str]
    created_date: datetime
    target_implementation: datetime
    alignment_score: float = 0.0
    stakeholder_buy_in: Dict[StakeholderType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.alignment_score <= 1:
            raise ValueError("Alignment score must be between 0 and 1")


@dataclass
class VisionAlignment:
    """Vision alignment with strategic objectives"""
    vision_id: str
    objective_id: str
    alignment_level: AlignmentLevel
    alignment_score: float
    supporting_evidence: List[str]
    gaps_identified: List[str]
    recommendations: List[str]
    
    def __post_init__(self):
        if not 0 <= self.alignment_score <= 1:
            raise ValueError("Alignment score must be between 0 and 1")


@dataclass
class StakeholderBuyIn:
    """Stakeholder buy-in assessment"""
    vision_id: str
    stakeholder_type: StakeholderType
    stakeholder_id: str
    buy_in_level: float
    concerns: List[str]
    support_factors: List[str]
    engagement_strategy: str
    last_updated: datetime
    
    def __post_init__(self):
        if not 0 <= self.buy_in_level <= 1:
            raise ValueError("Buy-in level must be between 0 and 1")


@dataclass
class CommunicationStrategy:
    """Communication strategy for vision"""
    vision_id: str
    target_audience: StakeholderType
    key_messages: List[str]
    communication_channels: List[str]
    frequency: str
    success_metrics: List[str]
    personalization_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionDevelopmentRequest:
    """Request for vision development"""
    organization_id: str
    scope: VisionScope
    strategic_objectives: List[StrategicObjective]
    current_culture_assessment: Dict[str, Any]
    stakeholder_requirements: Dict[StakeholderType, List[str]]
    constraints: List[str] = field(default_factory=list)
    timeline: Optional[datetime] = None


@dataclass
class VisionDevelopmentResult:
    """Result of vision development process"""
    vision: CulturalVision
    alignment_analysis: List[VisionAlignment]
    communication_strategies: List[CommunicationStrategy]
    implementation_recommendations: List[str]
    risk_factors: List[str]
    success_probability: float
    
    def __post_init__(self):
        if not 0 <= self.success_probability <= 1:
            raise ValueError("Success probability must be between 0 and 1")