"""
Knowledge Integration Models for Autonomous Innovation Lab

This module defines data models for knowledge synthesis, pattern recognition,
and learning optimization in the autonomous innovation lab system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class KnowledgeType(Enum):
    """Types of knowledge in the system"""
    RESEARCH_FINDING = "research_finding"
    EXPERIMENTAL_RESULT = "experimental_result"
    PROTOTYPE_INSIGHT = "prototype_insight"
    VALIDATION_OUTCOME = "validation_outcome"
    PATTERN_DISCOVERY = "pattern_discovery"
    INNOVATION_CONCEPT = "innovation_concept"


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge items"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PatternType(Enum):
    """Types of patterns that can be recognized"""
    CORRELATION = "correlation"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    EMERGENT = "emergent"


@dataclass
class KnowledgeItem:
    """Individual piece of knowledge in the system"""
    id: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    source: str
    timestamp: datetime
    confidence: ConfidenceLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)  # IDs of related knowledge items


@dataclass
class KnowledgeCorrelation:
    """Correlation between knowledge items"""
    id: str
    item_ids: List[str]
    correlation_type: str
    strength: float  # 0.0 to 1.0
    confidence: ConfidenceLevel
    description: str
    discovered_at: datetime
    validation_status: str = "pending"


@dataclass
class SynthesizedKnowledge:
    """Result of knowledge synthesis process"""
    id: str
    source_items: List[str]  # IDs of source knowledge items
    synthesis_method: str
    synthesized_content: Dict[str, Any]
    insights: List[str]
    confidence: ConfidenceLevel
    created_at: datetime
    validation_results: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass
class Pattern:
    """Recognized pattern across innovations"""
    id: str
    pattern_type: PatternType
    description: str
    evidence: List[str]  # IDs of supporting knowledge items
    strength: float  # 0.0 to 1.0
    confidence: ConfidenceLevel
    discovered_at: datetime
    applications: List[str] = field(default_factory=list)
    predictive_power: float = 0.0


@dataclass
class LearningMetric:
    """Metrics for learning effectiveness"""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any]
    improvement_rate: float = 0.0


@dataclass
class LearningOptimization:
    """Learning optimization configuration and results"""
    id: str
    optimization_target: str
    current_metrics: List[LearningMetric]
    optimization_strategy: str
    parameters: Dict[str, Any]
    effectiveness_score: float
    created_at: datetime
    last_updated: datetime
    improvements: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class KnowledgeValidationResult:
    """Result of knowledge validation process"""
    knowledge_id: str
    validation_method: str
    is_valid: bool
    confidence: ConfidenceLevel
    validation_score: float
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeGraph:
    """Graph representation of knowledge relationships"""
    nodes: List[KnowledgeItem]
    edges: List[KnowledgeCorrelation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SynthesisRequest:
    """Request for knowledge synthesis"""
    id: str
    source_knowledge_ids: List[str]
    synthesis_goal: str
    method_preferences: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"
    requested_at: datetime = field(default_factory=datetime.now)


@dataclass
class PatternRecognitionResult:
    """Result of pattern recognition process"""
    patterns_found: List[Pattern]
    analysis_method: str
    confidence: ConfidenceLevel
    processing_time: float
    recommendations: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)