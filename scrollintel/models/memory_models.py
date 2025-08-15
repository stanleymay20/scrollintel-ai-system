"""
Advanced Memory Architecture Models for AGI Cognitive System

This module implements human-like memory systems including working memory,
long-term memory, and episodic memory with consolidation and retrieval mechanisms.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
from abc import ABC, abstractmethod


class MemoryType(Enum):
    """Types of memory in the cognitive architecture"""
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class ConsolidationState(Enum):
    """States of memory consolidation"""
    FRESH = "fresh"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    FORGETTING = "forgetting"


@dataclass
class MemoryTrace:
    """Individual memory trace with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.WORKING
    strength: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    consolidation_state: ConsolidationState = ConsolidationState.FRESH
    associations: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    emotional_valence: float = 0.0
    
    def access(self):
        """Update access metadata when memory is retrieved"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # Strengthen memory through retrieval
        self.strength = min(1.0, self.strength + 0.1)


@dataclass
class AssociativeLink:
    """Link between memory traces for associative networks"""
    source_id: str
    target_id: str
    strength: float = 0.5
    link_type: str = "semantic"
    created_at: datetime = field(default_factory=datetime.now)
    activation_count: int = 0
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen associative link"""
        self.strength = min(1.0, self.strength + amount)
        self.activation_count += 1


@dataclass
class EpisodicMemory:
    """Episodic memory representing specific experiences"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    location: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)
    sensory_details: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    related_memories: Set[str] = field(default_factory=set)
    vividness: float = 1.0
    
    def decay_vividness(self, decay_rate: float = 0.01):
        """Natural decay of memory vividness over time"""
        self.vividness = max(0.0, self.vividness - decay_rate)


@dataclass
class WorkingMemorySlot:
    """Individual slot in working memory with limited capacity"""
    content: Any = None
    activation_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    source_memory_id: Optional[str] = None
    
    def is_active(self, threshold: float = 0.1) -> bool:
        """Check if slot is actively maintained"""
        return self.activation_level > threshold
    
    def decay(self, decay_rate: float = 0.05):
        """Natural decay of activation"""
        self.activation_level = max(0.0, self.activation_level - decay_rate)


@dataclass
class MemoryConsolidationJob:
    """Job for consolidating memories from working to long-term"""
    memory_id: str
    priority: float = 0.5
    consolidation_type: str = "standard"
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


class MemoryRetrievalQuery:
    """Query for retrieving memories with various criteria"""
    
    def __init__(self):
        self.content_keywords: List[str] = []
        self.memory_types: List[MemoryType] = []
        self.time_range: Optional[Tuple[datetime, datetime]] = None
        self.min_strength: float = 0.0
        self.max_results: int = 10
        self.include_associations: bool = True
        self.context_filters: Dict[str, Any] = {}
        self.similarity_threshold: float = 0.3


@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval with relevance scoring"""
    memory: MemoryTrace
    relevance_score: float
    retrieval_path: List[str] = field(default_factory=list)
    associated_memories: List['MemoryRetrievalResult'] = field(default_factory=list)


class MemoryOptimizationStrategy(ABC):
    """Abstract base for memory optimization strategies"""
    
    @abstractmethod
    def should_forget(self, memory: MemoryTrace) -> bool:
        """Determine if memory should be forgotten"""
        pass
    
    @abstractmethod
    def calculate_importance(self, memory: MemoryTrace) -> float:
        """Calculate memory importance for consolidation"""
        pass


class RecencyImportanceStrategy(MemoryOptimizationStrategy):
    """Strategy based on recency and access frequency"""
    
    def __init__(self, recency_weight: float = 0.6, frequency_weight: float = 0.4):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
    
    def should_forget(self, memory: MemoryTrace) -> bool:
        """Forget old, rarely accessed memories"""
        days_old = (datetime.now() - memory.created_at).days
        if days_old > 30 and memory.access_count < 2 and memory.strength < 0.3:
            return True
        return False
    
    def calculate_importance(self, memory: MemoryTrace) -> float:
        """Calculate importance based on recency and frequency"""
        days_old = max(1, (datetime.now() - memory.created_at).days)
        recency_score = 1.0 / days_old
        frequency_score = min(1.0, memory.access_count / 10.0)
        
        return (self.recency_weight * recency_score + 
                self.frequency_weight * frequency_score)


@dataclass
class MemorySystemMetrics:
    """Metrics for monitoring memory system performance"""
    total_memories: int = 0
    working_memory_utilization: float = 0.0
    consolidation_queue_size: int = 0
    average_retrieval_time: float = 0.0
    memory_efficiency: float = 0.0
    forgetting_rate: float = 0.0
    association_density: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)