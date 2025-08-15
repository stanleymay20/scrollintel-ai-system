"""
Information Synthesis Models for Crisis Leadership Excellence

This module defines the data models for information synthesis during crisis situations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from uuid import uuid4


class InformationSource(Enum):
    """Types of information sources during crisis"""
    INTERNAL_SYSTEMS = "internal_systems"
    EXTERNAL_REPORTS = "external_reports"
    STAKEHOLDER_INPUT = "stakeholder_input"
    MEDIA_MONITORING = "media_monitoring"
    SENSOR_DATA = "sensor_data"
    EXPERT_ANALYSIS = "expert_analysis"
    SOCIAL_MEDIA = "social_media"
    REGULATORY_UPDATES = "regulatory_updates"


class InformationPriority(Enum):
    """Priority levels for information processing"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConflictType(Enum):
    """Types of information conflicts"""
    CONTRADICTORY_FACTS = "contradictory_facts"
    TIMING_DISCREPANCY = "timing_discrepancy"
    SOURCE_RELIABILITY = "source_reliability"
    SCOPE_MISMATCH = "scope_mismatch"
    INTERPRETATION_VARIANCE = "interpretation_variance"


@dataclass
class InformationItem:
    """Individual piece of information in crisis context"""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    source: InformationSource = InformationSource.INTERNAL_SYSTEMS
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0  # 0.0 to 1.0
    reliability_score: float = 0.0  # 0.0 to 1.0
    priority: InformationPriority = InformationPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification_status: str = "unverified"  # unverified, verified, disputed
    impact_assessment: Optional[str] = None


@dataclass
class InformationConflict:
    """Represents conflicts between information items"""
    id: str = field(default_factory=lambda: str(uuid4()))
    conflict_type: ConflictType = ConflictType.CONTRADICTORY_FACTS
    conflicting_items: List[str] = field(default_factory=list)  # Information item IDs
    description: str = ""
    severity: float = 0.0  # 0.0 to 1.0
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class SynthesizedInformation:
    """Result of information synthesis process"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    synthesis_timestamp: datetime = field(default_factory=datetime.now)
    key_findings: List[str] = field(default_factory=list)
    confidence_level: float = 0.0  # Overall confidence in synthesis
    uncertainty_factors: List[str] = field(default_factory=list)
    information_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    source_items: List[str] = field(default_factory=list)  # Source information item IDs
    conflicts_identified: List[str] = field(default_factory=list)  # Conflict IDs
    priority_score: float = 0.0


@dataclass
class FilterCriteria:
    """Criteria for filtering information during synthesis"""
    min_confidence: float = 0.0
    min_reliability: float = 0.0
    required_sources: List[InformationSource] = field(default_factory=list)
    excluded_sources: List[InformationSource] = field(default_factory=list)
    priority_threshold: InformationPriority = InformationPriority.LOW
    time_window_hours: Optional[int] = None
    required_tags: List[str] = field(default_factory=list)
    excluded_tags: List[str] = field(default_factory=list)


@dataclass
class UncertaintyAssessment:
    """Assessment of uncertainty in synthesized information"""
    overall_uncertainty: float = 0.0  # 0.0 to 1.0
    information_completeness: float = 0.0  # 0.0 to 1.0
    source_diversity: float = 0.0  # 0.0 to 1.0
    temporal_consistency: float = 0.0  # 0.0 to 1.0
    conflict_resolution_confidence: float = 0.0  # 0.0 to 1.0
    key_uncertainties: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class SynthesisRequest:
    """Request for information synthesis"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    requester: str = ""
    request_timestamp: datetime = field(default_factory=datetime.now)
    information_items: List[str] = field(default_factory=list)  # Information item IDs
    filter_criteria: Optional[FilterCriteria] = None
    synthesis_focus: List[str] = field(default_factory=list)  # Areas to focus on
    urgency_level: InformationPriority = InformationPriority.MEDIUM
    expected_completion: Optional[datetime] = None


@dataclass
class SynthesisMetrics:
    """Metrics for synthesis process performance"""
    processing_time_seconds: float = 0.0
    items_processed: int = 0
    items_filtered_out: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    confidence_improvement: float = 0.0
    uncertainty_reduction: float = 0.0
    synthesis_quality_score: float = 0.0