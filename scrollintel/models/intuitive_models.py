"""
Intuitive reasoning models for AGI cognitive architecture.
Implements intuitive insights, pattern synthesis, and creative problem-solving structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime
import uuid


class InsightType(Enum):
    """Types of intuitive insights"""
    PATTERN_RECOGNITION = "pattern_recognition"
    CREATIVE_LEAP = "creative_leap"
    HOLISTIC_UNDERSTANDING = "holistic_understanding"
    EMERGENT_PROPERTY = "emergent_property"
    CROSS_DOMAIN_CONNECTION = "cross_domain_connection"
    GESTALT_INSIGHT = "gestalt_insight"


class PatternComplexity(Enum):
    """Complexity levels of patterns"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"
    EMERGENT = "emergent"


class CreativityLevel(Enum):
    """Levels of creativity in solutions"""
    CONVENTIONAL = "conventional"
    ADAPTIVE = "adaptive"
    INNOVATIVE = "innovative"
    BREAKTHROUGH = "breakthrough"
    REVOLUTIONARY = "revolutionary"


@dataclass
class DataPoint:
    """Represents a single data point for pattern analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    value: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)


@dataclass
class Pattern:
    """Represents a discovered pattern"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""
    description: str = ""
    complexity: PatternComplexity = PatternComplexity.SIMPLE
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    emergence_conditions: Dict[str, Any] = field(default_factory=dict)
    predictive_power: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntuitiveInsight:
    """Represents an intuitive insight or leap"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: InsightType = InsightType.PATTERN_RECOGNITION
    description: str = ""
    confidence: float = 0.0
    novelty_score: float = 0.0
    coherence_score: float = 0.0
    source_patterns: List[Pattern] = field(default_factory=list)
    cross_domain_connections: List[str] = field(default_factory=list)
    emergence_context: Dict[str, Any] = field(default_factory=dict)
    validation_criteria: List[str] = field(default_factory=list)
    potential_applications: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall insight quality score"""
        return (self.confidence * 0.4 + 
                self.novelty_score * 0.3 + 
                self.coherence_score * 0.3)


@dataclass
class PatternSynthesis:
    """Represents synthesis of multiple patterns"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_patterns: List[Pattern] = field(default_factory=list)
    synthesized_pattern: Optional[Pattern] = None
    synthesis_method: str = ""
    emergence_properties: List[str] = field(default_factory=list)
    synthesis_confidence: float = 0.0
    cross_domain_bridges: List[Tuple[str, str]] = field(default_factory=list)
    holistic_properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Problem:
    """Represents a problem to be solved"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    domain: str = "general"
    complexity_level: float = 0.0
    constraints: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    known_solutions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)


@dataclass
class CreativeSolution:
    """Represents a creative solution to a problem"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_id: str = ""
    solution_description: str = ""
    creativity_level: CreativityLevel = CreativityLevel.CONVENTIONAL
    feasibility_score: float = 0.0
    innovation_score: float = 0.0
    elegance_score: float = 0.0
    implementation_steps: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    inspiration_sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_solution_quality(self) -> float:
        """Calculate overall solution quality"""
        return (self.feasibility_score * 0.3 + 
                self.innovation_score * 0.4 + 
                self.elegance_score * 0.3)


@dataclass
class Challenge:
    """Represents a challenge requiring creative thinking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    challenge_type: str = "general"
    difficulty_level: float = 0.0
    time_constraints: Optional[datetime] = None
    resource_constraints: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    previous_attempts: List[str] = field(default_factory=list)


@dataclass
class HolisticInsight:
    """Represents holistic understanding of complex systems"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_description: str = ""
    emergent_properties: List[str] = field(default_factory=list)
    system_dynamics: Dict[str, Any] = field(default_factory=dict)
    interconnections: List[Tuple[str, str, str]] = field(default_factory=list)  # (from, to, relationship)
    leverage_points: List[str] = field(default_factory=list)
    system_archetypes: List[str] = field(default_factory=list)
    feedback_loops: List[Dict[str, Any]] = field(default_factory=list)
    boundary_conditions: List[str] = field(default_factory=list)
    holistic_understanding_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Context:
    """Represents context for intuitive reasoning"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    situation: str = ""
    domain: str = "general"
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    social_context: Dict[str, Any] = field(default_factory=dict)
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    technological_context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    uncertainty_level: float = 0.0
    ambiguity_level: float = 0.0


@dataclass
class IntuitiveLeap:
    """Represents a sudden intuitive leap or breakthrough"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    leap_type: str = ""
    from_state: str = ""
    to_state: str = ""
    leap_description: str = ""
    confidence: float = 0.0
    surprise_factor: float = 0.0
    coherence_with_prior_knowledge: float = 0.0
    potential_impact: float = 0.0
    validation_requirements: List[str] = field(default_factory=list)
    supporting_intuitions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NeuralArchitecture:
    """Represents advanced neural architecture configuration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    architecture_type: str = ""
    layers: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    activation_functions: List[str] = field(default_factory=list)
    learning_mechanisms: List[str] = field(default_factory=list)
    attention_mechanisms: List[str] = field(default_factory=list)
    memory_systems: List[str] = field(default_factory=list)
    plasticity_rules: List[str] = field(default_factory=list)
    emergence_properties: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Represents validation results for intuitive insights"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_id: str = ""
    validation_method: str = ""
    validation_score: float = 0.0
    evidence_strength: float = 0.0
    consistency_score: float = 0.0
    predictive_accuracy: float = 0.0
    peer_validation: List[str] = field(default_factory=list)
    empirical_support: List[str] = field(default_factory=list)
    theoretical_grounding: List[str] = field(default_factory=list)
    limitations_identified: List[str] = field(default_factory=list)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConfidenceMetrics:
    """Represents confidence scoring metrics"""
    overall_confidence: float = 0.0
    pattern_confidence: float = 0.0
    synthesis_confidence: float = 0.0
    creativity_confidence: float = 0.0
    validation_confidence: float = 0.0
    uncertainty_quantification: Dict[str, float] = field(default_factory=dict)
    confidence_sources: List[str] = field(default_factory=list)
    confidence_degradation_factors: List[str] = field(default_factory=list)
    
    def calculate_weighted_confidence(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted confidence score"""
        if weights is None:
            weights = {
                'pattern': 0.25,
                'synthesis': 0.25,
                'creativity': 0.25,
                'validation': 0.25
            }
        
        return (self.pattern_confidence * weights.get('pattern', 0.25) +
                self.synthesis_confidence * weights.get('synthesis', 0.25) +
                self.creativity_confidence * weights.get('creativity', 0.25) +
                self.validation_confidence * weights.get('validation', 0.25))