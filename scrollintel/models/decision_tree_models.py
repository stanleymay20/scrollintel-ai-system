"""
Decision Tree Models for Crisis Leadership Excellence System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


class CrisisType(Enum):
    SYSTEM_OUTAGE = "system_outage"
    SECURITY_BREACH = "security_breach"
    DATA_LOSS = "data_loss"
    FINANCIAL_CRISIS = "financial_crisis"
    REGULATORY_VIOLATION = "regulatory_violation"
    REPUTATION_DAMAGE = "reputation_damage"
    PERSONNEL_CRISIS = "personnel_crisis"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_ATTACK = "cyber_attack"


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class DecisionNodeType(Enum):
    CONDITION = "condition"
    ACTION = "action"
    ESCALATION = "escalation"
    INFORMATION_GATHERING = "information_gathering"


class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DecisionCondition:
    """Represents a condition that must be evaluated in a decision tree"""
    condition_id: str
    description: str
    evaluation_criteria: str
    required_information: List[str]
    evaluation_method: str  # "automatic", "manual", "hybrid"
    timeout_seconds: int = 300


@dataclass
class DecisionAction:
    """Represents an action that can be taken in response to a crisis"""
    action_id: str
    title: str
    description: str
    required_resources: List[str]
    estimated_duration: int  # in minutes
    success_probability: float
    risk_level: str
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)


@dataclass
class DecisionNode:
    """Represents a node in a decision tree"""
    node_id: str
    node_type: DecisionNodeType
    title: str
    description: str
    condition: Optional[DecisionCondition] = None
    action: Optional[DecisionAction] = None
    children: List['DecisionNode'] = field(default_factory=list)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTree:
    """Represents a complete decision tree for a crisis scenario"""
    tree_id: str
    name: str
    description: str
    crisis_types: List[CrisisType]
    severity_levels: List[SeverityLevel]
    root_node: DecisionNode
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    success_rate: float = 0.0
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class DecisionPath:
    """Represents a path taken through a decision tree"""
    path_id: str
    tree_id: str
    crisis_id: str
    nodes_traversed: List[str]
    decisions_made: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime] = None
    outcome: Optional[str] = None
    success: Optional[bool] = None
    confidence_score: float = 0.0


@dataclass
class DecisionRecommendation:
    """Represents a recommendation from the decision tree system"""
    recommendation_id: str
    crisis_id: str
    tree_id: str
    recommended_action: DecisionAction
    confidence_level: ConfidenceLevel
    reasoning: str
    alternative_actions: List[DecisionAction] = field(default_factory=list)
    required_approvals: List[str] = field(default_factory=list)
    estimated_impact: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DecisionTreeMetrics:
    """Metrics for decision tree performance"""
    tree_id: str
    total_usage: int
    success_rate: float
    average_decision_time: float
    confidence_distribution: Dict[str, int]
    most_common_paths: List[Dict[str, Any]]
    failure_points: List[Dict[str, Any]]
    optimization_suggestions: List[str]
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DecisionTreeLearning:
    """Learning data for decision tree optimization"""
    learning_id: str
    tree_id: str
    crisis_scenario: Dict[str, Any]
    actual_outcome: str
    predicted_outcome: str
    decision_path: List[str]
    feedback_score: float
    lessons_learned: List[str]
    suggested_improvements: List[str]
    timestamp: datetime = field(default_factory=datetime.now)