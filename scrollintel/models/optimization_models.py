"""
Data models for the Automated Optimization Engine in the Advanced Prompt Management System.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from enum import Enum

from .database import Base
from .prompt_models import AdvancedPromptTemplate


class OptimizationAlgorithm(Enum):
    """Types of optimization algorithms."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"


class JobStatus(Enum):
    """Status of optimization jobs."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationType(Enum):
    """Type of optimization objective."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    PARETO_OPTIMIZATION = "pareto_optimization"


class OptimizationJob(Base):
    """Optimization job model for automated prompt improvement."""
    __tablename__ = "optimization_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    prompt_id = Column(String, ForeignKey("advanced_prompt_templates.id"), nullable=False)
    algorithm = Column(String(50), nullable=False, default=OptimizationAlgorithm.GENETIC_ALGORITHM.value)
    optimization_type = Column(String(50), nullable=False, default=OptimizationType.SINGLE_OBJECTIVE.value)
    config = Column(JSON, nullable=False)  # Algorithm-specific configuration
    objectives = Column(JSON, nullable=False)  # List of optimization objectives
    constraints = Column(JSON, default=list)  # List of constraints
    status = Column(String(20), nullable=False, default=JobStatus.PENDING.value)
    progress = Column(Float, default=0.0)  # Progress percentage (0-100)
    current_generation = Column(Integer, default=0)
    max_generations = Column(Integer, nullable=False, default=100)
    population_size = Column(Integer, nullable=False, default=50)
    best_fitness = Column(Float)
    convergence_threshold = Column(Float, default=0.001)
    early_stopping_patience = Column(Integer, default=10)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("AdvancedPromptTemplate")
    results = relationship("OptimizationResults", back_populates="job", cascade="all, delete-orphan")
    candidates = relationship("OptimizationCandidate", back_populates="job", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt_id": self.prompt_id,
            "algorithm": self.algorithm,
            "optimization_type": self.optimization_type,
            "config": self.config,
            "objectives": self.objectives,
            "constraints": self.constraints or [],
            "status": self.status,
            "progress": self.progress,
            "current_generation": self.current_generation,
            "max_generations": self.max_generations,
            "population_size": self.population_size,
            "best_fitness": self.best_fitness,
            "convergence_threshold": self.convergence_threshold,
            "early_stopping_patience": self.early_stopping_patience,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class OptimizationResults(Base):
    """Results of optimization jobs."""
    __tablename__ = "optimization_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("optimization_jobs.id"), nullable=False)
    best_prompt_content = Column(Text, nullable=False)
    best_fitness_score = Column(Float, nullable=False)
    fitness_history = Column(JSON, default=list)  # History of fitness scores
    pareto_front = Column(JSON, default=list)  # For multi-objective optimization
    convergence_data = Column(JSON, default=dict)  # Convergence analysis data
    performance_metrics = Column(JSON, nullable=False)  # Detailed performance metrics
    improvement_percentage = Column(Float)  # Improvement over original prompt
    statistical_significance = Column(Float)  # P-value of improvement
    confidence_interval = Column(JSON)  # Confidence interval for improvement
    optimization_summary = Column(Text)  # Human-readable summary
    recommendations = Column(JSON, default=list)  # List of recommendations
    diagnostics = Column(JSON, default=dict)  # Diagnostic information
    execution_time_seconds = Column(Float)
    total_evaluations = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("OptimizationJob", back_populates="results")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "best_prompt_content": self.best_prompt_content,
            "best_fitness_score": self.best_fitness_score,
            "fitness_history": self.fitness_history or [],
            "pareto_front": self.pareto_front or [],
            "convergence_data": self.convergence_data or {},
            "performance_metrics": self.performance_metrics,
            "improvement_percentage": self.improvement_percentage,
            "statistical_significance": self.statistical_significance,
            "confidence_interval": self.confidence_interval,
            "optimization_summary": self.optimization_summary,
            "recommendations": self.recommendations or [],
            "diagnostics": self.diagnostics or {},
            "execution_time_seconds": self.execution_time_seconds,
            "total_evaluations": self.total_evaluations,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class OptimizationCandidate(Base):
    """Individual candidate solutions during optimization."""
    __tablename__ = "optimization_candidates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("optimization_jobs.id"), nullable=False)
    generation = Column(Integer, nullable=False)
    candidate_index = Column(Integer, nullable=False)
    prompt_content = Column(Text, nullable=False)
    fitness_score = Column(Float, nullable=False)
    objective_scores = Column(JSON, nullable=False)  # Scores for each objective
    genetic_info = Column(JSON, default=dict)  # Genetic algorithm specific info
    rl_info = Column(JSON, default=dict)  # Reinforcement learning specific info
    evaluation_metrics = Column(JSON, nullable=False)  # Detailed evaluation metrics
    is_pareto_optimal = Column(Boolean, default=False)
    dominance_rank = Column(Integer)  # For multi-objective optimization
    crowding_distance = Column(Float)  # For NSGA-II algorithm
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("OptimizationJob", back_populates="candidates")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "generation": self.generation,
            "candidate_index": self.candidate_index,
            "prompt_content": self.prompt_content,
            "fitness_score": self.fitness_score,
            "objective_scores": self.objective_scores,
            "genetic_info": self.genetic_info or {},
            "rl_info": self.rl_info or {},
            "evaluation_metrics": self.evaluation_metrics,
            "is_pareto_optimal": self.is_pareto_optimal,
            "dominance_rank": self.dominance_rank,
            "crowding_distance": self.crowding_distance,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PerformanceMetric(Base):
    """Custom performance metrics for optimization."""
    __tablename__ = "performance_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    metric_type = Column(String(50), nullable=False)  # accuracy, relevance, efficiency, custom
    evaluation_function = Column(Text, nullable=False)  # Python code for evaluation
    weight = Column(Float, default=1.0)  # Weight in multi-objective optimization
    higher_is_better = Column(Boolean, default=True)
    min_value = Column(Float)
    max_value = Column(Float)
    is_active = Column(Boolean, default=True)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metric_type": self.metric_type,
            "evaluation_function": self.evaluation_function,
            "weight": self.weight,
            "higher_is_better": self.higher_is_better,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class OptimizationConfig:
    """Configuration class for optimization algorithms."""
    
    def __init__(self, algorithm: OptimizationAlgorithm, **kwargs):
        self.algorithm = algorithm
        self.config = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "algorithm": self.algorithm.value,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary representation."""
        algorithm = OptimizationAlgorithm(data["algorithm"])
        return cls(algorithm, **data.get("config", {}))


class TestCase:
    """Test case for prompt evaluation."""
    
    def __init__(self, input_data: Dict[str, Any], expected_output: Optional[str] = None,
                 evaluation_criteria: Optional[Dict[str, Any]] = None):
        self.input_data = input_data
        self.expected_output = expected_output
        self.evaluation_criteria = evaluation_criteria or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "evaluation_criteria": self.evaluation_criteria
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create from dictionary representation."""
        return cls(
            input_data=data["input_data"],
            expected_output=data.get("expected_output"),
            evaluation_criteria=data.get("evaluation_criteria", {})
        )


class PerformanceMetrics:
    """Performance metrics for prompt evaluation."""
    
    def __init__(self, accuracy: float = 0.0, relevance: float = 0.0, 
                 efficiency: float = 0.0, custom_metrics: Optional[Dict[str, float]] = None):
        self.accuracy = accuracy
        self.relevance = relevance
        self.efficiency = efficiency
        self.custom_metrics = custom_metrics or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "efficiency": self.efficiency,
            "custom_metrics": self.custom_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create from dictionary representation."""
        return cls(
            accuracy=data.get("accuracy", 0.0),
            relevance=data.get("relevance", 0.0),
            efficiency=data.get("efficiency", 0.0),
            custom_metrics=data.get("custom_metrics", {})
        )
    
    def get_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted score across all metrics."""
        if weights is None:
            weights = {"accuracy": 0.4, "relevance": 0.4, "efficiency": 0.2}
        
        score = (
            self.accuracy * weights.get("accuracy", 0.0) +
            self.relevance * weights.get("relevance", 0.0) +
            self.efficiency * weights.get("efficiency", 0.0)
        )
        
        # Add custom metrics
        for metric_name, metric_value in self.custom_metrics.items():
            weight = weights.get(metric_name, 0.0)
            score += metric_value * weight
        
        return score


class OptimizationStatus:
    """Status information for optimization jobs."""
    
    def __init__(self, job_id: str, status: JobStatus, progress: float = 0.0,
                 current_generation: int = 0, best_fitness: Optional[float] = None,
                 estimated_time_remaining: Optional[float] = None,
                 message: Optional[str] = None):
        self.job_id = job_id
        self.status = status
        self.progress = progress
        self.current_generation = current_generation
        self.best_fitness = best_fitness
        self.estimated_time_remaining = estimated_time_remaining
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "current_generation": self.current_generation,
            "best_fitness": self.best_fitness,
            "estimated_time_remaining": self.estimated_time_remaining,
            "message": self.message
        }