"""
Behavioral Analysis Models for Cultural Transformation Leadership

This module defines data models for analyzing organizational behaviors,
behavioral patterns, and culture-behavior alignment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class BehaviorType(Enum):
    """Types of organizational behaviors"""
    COMMUNICATION = "communication"
    COLLABORATION = "collaboration"
    DECISION_MAKING = "decision_making"
    LEADERSHIP = "leadership"
    INNOVATION = "innovation"
    PERFORMANCE = "performance"
    LEARNING = "learning"
    CONFLICT_RESOLUTION = "conflict_resolution"


class BehaviorFrequency(Enum):
    """Frequency of behavior occurrence"""
    RARE = "rare"
    OCCASIONAL = "occasional"
    FREQUENT = "frequent"
    CONSTANT = "constant"


class AlignmentLevel(Enum):
    """Level of behavior-culture alignment"""
    MISALIGNED = "misaligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    WELL_ALIGNED = "well_aligned"
    PERFECTLY_ALIGNED = "perfectly_aligned"


@dataclass
class BehaviorPattern:
    """Represents an identified behavioral pattern"""
    id: str
    name: str
    description: str
    behavior_type: BehaviorType
    frequency: BehaviorFrequency
    triggers: List[str]
    outcomes: List[str]
    participants: List[str]
    context: Dict[str, Any]
    strength: float  # 0.0 to 1.0
    identified_date: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0")


@dataclass
class BehavioralNorm:
    """Represents an organizational behavioral norm"""
    id: str
    name: str
    description: str
    behavior_type: BehaviorType
    expected_behaviors: List[str]
    discouraged_behaviors: List[str]
    enforcement_mechanisms: List[str]
    compliance_rate: float  # 0.0 to 1.0
    cultural_importance: float  # 0.0 to 1.0
    established_date: datetime
    last_updated: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.compliance_rate <= 1.0:
            raise ValueError("Compliance rate must be between 0.0 and 1.0")
        if not 0.0 <= self.cultural_importance <= 1.0:
            raise ValueError("Cultural importance must be between 0.0 and 1.0")


@dataclass
class BehaviorCultureAlignment:
    """Represents alignment between behavior and culture"""
    id: str
    behavior_pattern_id: str
    cultural_value: str
    alignment_level: AlignmentLevel
    alignment_score: float  # 0.0 to 1.0
    supporting_evidence: List[str]
    conflicting_evidence: List[str]
    impact_on_culture: str
    recommendations: List[str]
    analysis_date: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.alignment_score <= 1.0:
            raise ValueError("Alignment score must be between 0.0 and 1.0")


@dataclass
class BehaviorAnalysisResult:
    """Complete behavioral analysis result"""
    organization_id: str
    analysis_id: str
    behavior_patterns: List[BehaviorPattern]
    behavioral_norms: List[BehavioralNorm]
    culture_alignments: List[BehaviorCultureAlignment]
    overall_health_score: float  # 0.0 to 1.0
    key_insights: List[str]
    recommendations: List[str]
    analysis_date: datetime
    analyst: str
    
    def __post_init__(self):
        if not 0.0 <= self.overall_health_score <= 1.0:
            raise ValueError("Overall health score must be between 0.0 and 1.0")


@dataclass
class BehaviorObservation:
    """Individual behavior observation"""
    id: str
    observer_id: str
    observed_behavior: str
    behavior_type: BehaviorType
    context: Dict[str, Any]
    participants: List[str]
    timestamp: datetime
    impact_assessment: str
    cultural_relevance: float  # 0.0 to 1.0
    
    def __post_init__(self):
        if not 0.0 <= self.cultural_relevance <= 1.0:
            raise ValueError("Cultural relevance must be between 0.0 and 1.0")


@dataclass
class BehaviorMetrics:
    """Behavioral metrics for analysis"""
    behavior_diversity_index: float
    norm_compliance_average: float
    culture_alignment_score: float
    behavior_consistency_index: float
    positive_behavior_ratio: float
    improvement_trend: float
    calculated_date: datetime