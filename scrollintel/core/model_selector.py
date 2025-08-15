"""
Model Selection Engine for Visual Generation System

This module provides intelligent model selection based on performance metrics,
cost optimization, quality prediction, and A/B testing capabilities.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import statistics
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types."""
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    IMAGE_ENHANCEMENT = "image_enhancement"
    STYLE_TRANSFER = "style_transfer"


class QualityMetric(Enum):
    """Quality assessment metrics."""
    OVERALL_SCORE = "overall_score"
    TECHNICAL_QUALITY = "technical_quality"
    AESTHETIC_SCORE = "aesthetic_score"
    PROMPT_ADHERENCE = "prompt_adherence"
    PROCESSING_TIME = "processing_time"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class ModelCapabilities:
    """Model capabilities and specifications."""
    model_id: str
    model_type: ModelType
    supported_resolutions: List[Tuple[int, int]]
    supported_formats: List[str]
    max_prompt_length: int
    supports_negative_prompts: bool = False
    supports_style_control: bool = False
    supports_batch_processing: bool = False
    gpu_memory_required: float = 4.0  # GB
    estimated_processing_time: float = 30.0  # seconds
    cost_per_generation: float = 0.10  # USD


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model."""
    model_id: str
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    average_cost: float = 0.0
    user_satisfaction_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_generations == 0:
            return 0.0
        return (self.successful_generations / self.total_generations) * 100
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (quality/time ratio)."""
        if self.average_processing_time == 0:
            return 0.0
        return self.average_quality_score / self.average_processing_time
    
    @property
    def cost_effectiveness(self) -> float:
        """Calculate cost effectiveness (quality/cost ratio)."""
        if self.average_cost == 0:
            return 0.0
        return self.average_quality_score / self.average_cost


@dataclass
class GenerationRequest:
    """Request for content generation."""
    request_id: str
    model_type: ModelType
    prompt: str
    negative_prompt: Optional[str] = None
    resolution: Tuple[int, int] = (1024, 1024)
    format: str = "jpg"
    style: Optional[str] = None
    quality_preference: str = "balanced"  # "speed", "balanced", "quality"
    budget_limit: Optional[float] = None
    user_id: Optional[str] = None
    priority: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing and comparison."""
        return {
            "model_type": self.model_type.value,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "resolution": self.resolution,
            "format": self.format,
            "style": self.style,
            "quality_preference": self.quality_preference
        }


@dataclass
class ModelSelection:
    """Result of model selection process."""
    selected_model: str
    confidence_score: float
    estimated_cost: float
    estimated_time: float
    estimated_quality: float
    selection_reason: str
    alternative_models: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "selected_model": self.selected_model,
            "confidence_score": self.confidence_score,
            "estimated_cost": self.estimated_cost,
            "estimated_time": self.estimated_time,
            "estimated_quality": self.estimated_quality,
            "selection_reason": self.selection_reason,
            "alternative_models": self.alternative_models
        }


class SelectionStrategy(ABC):
    """Abstract base class for model selection strategies."""
    
    @abstractmethod
    async def select_model(
        self,
        request: GenerationRequest,
        available_models: List[str],
        model_metrics: Dict[str, ModelPerformanceMetrics],
        model_capabilities: Dict[str, ModelCapabilities]
    ) -> ModelSelection:
        """Select the best model for the given request."""
        pass


class PerformanceBasedStrategy(SelectionStrategy):
    """Select model based on historical performance metrics."""
    
    async def select_model(
        self,
        request: GenerationRequest,
        available_models: List[str],
        model_metrics: Dict[str, ModelPerformanceMetrics],
        model_capabilities: Dict[str, ModelCapabilities]
    ) -> ModelSelection:
        """Select model with best performance for request type."""
        
        # Filter models by compatibility
        compatible_models = []
        for model_id in available_models:
            if model_id in model_capabilities:
                caps = model_capabilities[model_id]
                if (caps.model_type == request.model_type and
                    request.resolution in caps.supported_resolutions and
                    request.format in caps.supported_formats):
                    compatible_models.append(model_id)
        
        if not compatible_models:
            raise ValueError("No compatible models found for request")
        
        # Score models based on performance
        model_scores = {}
        for model_id in compatible_models:
            metrics = model_metrics.get(model_id, ModelPerformanceMetrics(model_id))
            
            # Calculate composite score
            success_weight = 0.3
            quality_weight = 0.4
            efficiency_weight = 0.3
            
            score = (
                metrics.success_rate * success_weight +
                metrics.average_quality_score * quality_weight +
                metrics.efficiency_score * efficiency_weight
            )
            
            model_scores[model_id] = score
        
        # Select best model
        best_model = max(model_scores.keys(), key=lambda m: model_scores[m])
        best_metrics = model_metrics.get(best_model, ModelPerformanceMetrics(best_model))
        best_caps = model_capabilities[best_model]
        
        return ModelSelection(
            selected_model=best_model,
            confidence_score=min(model_scores[best_model] / 100, 1.0),
            estimated_cost=best_caps.cost_per_generation,
            estimated_time=best_caps.estimated_processing_time,
            estimated_quality=best_metrics.average_quality_score,
            selection_reason="Selected based on historical performance metrics",
            alternative_models=sorted(
                [m for m in compatible_models if m != best_model],
                key=lambda m: model_scores[m],
                reverse=True
            )[:3]
        )


class CostOptimizedStrategy(SelectionStrategy):
    """Select model optimized for cost while maintaining quality threshold."""
    
    def __init__(self, min_quality_threshold: float = 0.7):
        self.min_quality_threshold = min_quality_threshold
    
    async def select_model(
        self,
        request: GenerationRequest,
        available_models: List[str],
        model_metrics: Dict[str, ModelPerformanceMetrics],
        model_capabilities: Dict[str, ModelCapabilities]
    ) -> ModelSelection:
        """Select most cost-effective model above quality threshold."""
        
        # Filter compatible models
        compatible_models = []
        for model_id in available_models:
            if model_id in model_capabilities:
                caps = model_capabilities[model_id]
                metrics = model_metrics.get(model_id, ModelPerformanceMetrics(model_id))
                
                if (caps.model_type == request.model_type and
                    request.resolution in caps.supported_resolutions and
                    request.format in caps.supported_formats and
                    metrics.average_quality_score >= self.min_quality_threshold):
                    compatible_models.append(model_id)
        
        if not compatible_models:
            # Fallback to any compatible model if none meet quality threshold
            compatible_models = [
                m for m in available_models
                if m in model_capabilities and
                model_capabilities[m].model_type == request.model_type
            ]
        
        if not compatible_models:
            raise ValueError("No compatible models found for request")
        
        # Select most cost-effective model
        best_model = min(
            compatible_models,
            key=lambda m: model_capabilities[m].cost_per_generation
        )
        
        best_metrics = model_metrics.get(best_model, ModelPerformanceMetrics(best_model))
        best_caps = model_capabilities[best_model]
        
        return ModelSelection(
            selected_model=best_model,
            confidence_score=0.8,  # High confidence for cost optimization
            estimated_cost=best_caps.cost_per_generation,
            estimated_time=best_caps.estimated_processing_time,
            estimated_quality=best_metrics.average_quality_score,
            selection_reason=f"Selected for cost optimization (${best_caps.cost_per_generation:.3f})",
            alternative_models=[
                m for m in compatible_models if m != best_model
            ][:3]
        )


class QualityOptimizedStrategy(SelectionStrategy):
    """Select model optimized for highest quality regardless of cost."""
    
    async def select_model(
        self,
        request: GenerationRequest,
        available_models: List[str],
        model_metrics: Dict[str, ModelPerformanceMetrics],
        model_capabilities: Dict[str, ModelCapabilities]
    ) -> ModelSelection:
        """Select highest quality model."""
        
        # Filter compatible models
        compatible_models = []
        for model_id in available_models:
            if model_id in model_capabilities:
                caps = model_capabilities[model_id]
                if (caps.model_type == request.model_type and
                    request.resolution in caps.supported_resolutions and
                    request.format in caps.supported_formats):
                    compatible_models.append(model_id)
        
        if not compatible_models:
            raise ValueError("No compatible models found for request")
        
        # Select highest quality model
        best_model = max(
            compatible_models,
            key=lambda m: model_metrics.get(m, ModelPerformanceMetrics(m)).average_quality_score
        )
        
        best_metrics = model_metrics.get(best_model, ModelPerformanceMetrics(best_model))
        best_caps = model_capabilities[best_model]
        
        return ModelSelection(
            selected_model=best_model,
            confidence_score=0.9,  # High confidence for quality optimization
            estimated_cost=best_caps.cost_per_generation,
            estimated_time=best_caps.estimated_processing_time,
            estimated_quality=best_metrics.average_quality_score,
            selection_reason=f"Selected for highest quality (score: {best_metrics.average_quality_score:.2f})",
            alternative_models=[
                m for m in compatible_models if m != best_model
            ][:3]
        )


class ABTestingFramework:
    """A/B testing framework for model comparison."""
    
    def __init__(self, test_duration_hours: int = 24):
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.test_duration = timedelta(hours=test_duration_hours)
        self._lock = asyncio.Lock()
    
    async def create_test(
        self,
        test_id: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        success_metrics: List[QualityMetric] = None
    ) -> bool:
        """Create a new A/B test."""
        async with self._lock:
            if test_id in self.active_tests:
                return False
            
            self.active_tests[test_id] = {
                "model_a": model_a,
                "model_b": model_b,
                "traffic_split": traffic_split,
                "success_metrics": success_metrics or [QualityMetric.OVERALL_SCORE],
                "start_time": datetime.now(),
                "end_time": datetime.now() + self.test_duration,
                "results_a": defaultdict(list),
                "results_b": defaultdict(list),
                "request_count_a": 0,
                "request_count_b": 0
            }
            
            logger.info(f"Created A/B test {test_id}: {model_a} vs {model_b}")
            return True
    
    async def get_test_assignment(self, test_id: str, request: GenerationRequest) -> Optional[str]:
        """Get model assignment for A/B test."""
        async with self._lock:
            if test_id not in self.active_tests:
                return None
            
            test = self.active_tests[test_id]
            
            # Check if test is still active
            if datetime.now() > test["end_time"]:
                await self._finalize_test(test_id)
                return None
            
            # Assign model based on traffic split
            if random.random() < test["traffic_split"]:
                test["request_count_a"] += 1
                return test["model_a"]
            else:
                test["request_count_b"] += 1
                return test["model_b"]
    
    async def record_test_result(
        self,
        test_id: str,
        model_id: str,
        metrics: Dict[QualityMetric, float]
    ):
        """Record test result for analysis."""
        async with self._lock:
            if test_id not in self.active_tests:
                return
            
            test = self.active_tests[test_id]
            
            if model_id == test["model_a"]:
                results = test["results_a"]
            elif model_id == test["model_b"]:
                results = test["results_b"]
            else:
                return
            
            for metric, value in metrics.items():
                results[metric].append(value)
    
    async def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current test status and preliminary results."""
        async with self._lock:
            if test_id not in self.active_tests:
                return self.test_results.get(test_id)
            
            test = self.active_tests[test_id]
            
            # Calculate preliminary statistics
            stats_a = self._calculate_stats(test["results_a"])
            stats_b = self._calculate_stats(test["results_b"])
            
            return {
                "test_id": test_id,
                "status": "active",
                "model_a": test["model_a"],
                "model_b": test["model_b"],
                "start_time": test["start_time"].isoformat(),
                "end_time": test["end_time"].isoformat(),
                "request_count_a": test["request_count_a"],
                "request_count_b": test["request_count_b"],
                "preliminary_results": {
                    "model_a": stats_a,
                    "model_b": stats_b
                }
            }
    
    async def _finalize_test(self, test_id: str):
        """Finalize test and calculate final results."""
        test = self.active_tests[test_id]
        
        stats_a = self._calculate_stats(test["results_a"])
        stats_b = self._calculate_stats(test["results_b"])
        
        # Determine winner based on primary success metric
        primary_metric = test["success_metrics"][0]
        winner = None
        
        if (primary_metric in stats_a and primary_metric in stats_b):
            if stats_a[primary_metric]["mean"] > stats_b[primary_metric]["mean"]:
                winner = test["model_a"]
            else:
                winner = test["model_b"]
        
        self.test_results[test_id] = {
            "test_id": test_id,
            "status": "completed",
            "model_a": test["model_a"],
            "model_b": test["model_b"],
            "winner": winner,
            "start_time": test["start_time"].isoformat(),
            "end_time": test["end_time"].isoformat(),
            "request_count_a": test["request_count_a"],
            "request_count_b": test["request_count_b"],
            "final_results": {
                "model_a": stats_a,
                "model_b": stats_b
            }
        }
        
        del self.active_tests[test_id]
        logger.info(f"Finalized A/B test {test_id}, winner: {winner}")
    
    def _calculate_stats(self, results: Dict[QualityMetric, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for test results."""
        stats = {}
        
        for metric, values in results.items():
            if values:
                stats[metric.value] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return stats


class ModelSelector:
    """
    Intelligent model selection engine with performance-based routing,
    cost optimization, quality prediction, and A/B testing capabilities.
    """
    
    def __init__(self):
        self.model_capabilities: Dict[str, ModelCapabilities] = {}
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.ab_testing = ABTestingFramework()
        self.selection_strategies = {
            "performance": PerformanceBasedStrategy(),
            "cost": CostOptimizedStrategy(),
            "quality": QualityOptimizedStrategy()
        }
        self.selection_history: deque = deque(maxlen=1000)
        self._lock = asyncio.Lock()
    
    async def register_model(self, capabilities: ModelCapabilities):
        """Register a new model with its capabilities."""
        async with self._lock:
            self.model_capabilities[capabilities.model_id] = capabilities
            
            if capabilities.model_id not in self.model_metrics:
                self.model_metrics[capabilities.model_id] = ModelPerformanceMetrics(
                    model_id=capabilities.model_id
                )
            
            logger.info(f"Registered model: {capabilities.model_id}")
    
    async def select_model(
        self,
        request: GenerationRequest,
        strategy: str = "performance"
    ) -> ModelSelection:
        """Select the best model for a generation request."""
        
        # Check for active A/B tests
        for test_id in list(self.ab_testing.active_tests.keys()):
            assigned_model = await self.ab_testing.get_test_assignment(test_id, request)
            if assigned_model:
                # Return A/B test assignment
                caps = self.model_capabilities.get(assigned_model)
                metrics = self.model_metrics.get(assigned_model, ModelPerformanceMetrics(assigned_model))
                
                selection = ModelSelection(
                    selected_model=assigned_model,
                    confidence_score=0.5,  # Neutral for A/B testing
                    estimated_cost=caps.cost_per_generation if caps else 0.1,
                    estimated_time=caps.estimated_processing_time if caps else 30.0,
                    estimated_quality=metrics.average_quality_score,
                    selection_reason=f"A/B test assignment (test: {test_id})",
                    alternative_models=[]
                )
                
                await self._record_selection(request, selection, f"ab_test_{test_id}")
                return selection
        
        # Use strategy-based selection
        if strategy not in self.selection_strategies:
            strategy = "performance"
        
        selection_strategy = self.selection_strategies[strategy]
        available_models = list(self.model_capabilities.keys())
        
        if not available_models:
            raise ValueError("No models registered")
        
        selection = await selection_strategy.select_model(
            request, available_models, self.model_metrics, self.model_capabilities
        )
        
        await self._record_selection(request, selection, strategy)
        return selection
    
    async def update_model_metrics(
        self,
        model_id: str,
        processing_time: float,
        quality_score: float,
        cost: float,
        success: bool,
        user_satisfaction: Optional[float] = None
    ):
        """Update model performance metrics based on generation results."""
        async with self._lock:
            if model_id not in self.model_metrics:
                self.model_metrics[model_id] = ModelPerformanceMetrics(model_id)
            
            metrics = self.model_metrics[model_id]
            
            # Update counters
            metrics.total_generations += 1
            if success:
                metrics.successful_generations += 1
            else:
                metrics.failed_generations += 1
            
            # Update averages using exponential moving average
            alpha = 0.1  # Learning rate
            
            if metrics.average_processing_time == 0:
                metrics.average_processing_time = processing_time
            else:
                metrics.average_processing_time = (
                    (1 - alpha) * metrics.average_processing_time + alpha * processing_time
                )
            
            if success:  # Only update quality for successful generations
                if metrics.average_quality_score == 0:
                    metrics.average_quality_score = quality_score
                else:
                    metrics.average_quality_score = (
                        (1 - alpha) * metrics.average_quality_score + alpha * quality_score
                    )
            
            if metrics.average_cost == 0:
                metrics.average_cost = cost
            else:
                metrics.average_cost = (1 - alpha) * metrics.average_cost + alpha * cost
            
            if user_satisfaction is not None:
                if metrics.user_satisfaction_score == 0:
                    metrics.user_satisfaction_score = user_satisfaction
                else:
                    metrics.user_satisfaction_score = (
                        (1 - alpha) * metrics.user_satisfaction_score + alpha * user_satisfaction
                    )
            
            metrics.last_updated = datetime.now()
            
            # Record A/B test results if applicable
            for test_id in self.ab_testing.active_tests:
                await self.ab_testing.record_test_result(
                    test_id,
                    model_id,
                    {
                        QualityMetric.OVERALL_SCORE: quality_score,
                        QualityMetric.PROCESSING_TIME: processing_time,
                        QualityMetric.USER_SATISFACTION: user_satisfaction or 0.0
                    }
                )
    
    async def get_model_rankings(
        self,
        model_type: ModelType,
        metric: QualityMetric = QualityMetric.OVERALL_SCORE
    ) -> List[Tuple[str, float]]:
        """Get model rankings for a specific type and metric."""
        async with self._lock:
            rankings = []
            
            for model_id, capabilities in self.model_capabilities.items():
                if capabilities.model_type == model_type:
                    metrics = self.model_metrics.get(model_id, ModelPerformanceMetrics(model_id))
                    
                    if metric == QualityMetric.OVERALL_SCORE:
                        score = metrics.average_quality_score
                    elif metric == QualityMetric.PROCESSING_TIME:
                        score = 1.0 / max(metrics.average_processing_time, 0.1)  # Inverse for ranking
                    elif metric == QualityMetric.USER_SATISFACTION:
                        score = metrics.user_satisfaction_score
                    else:
                        score = metrics.efficiency_score
                    
                    rankings.append((model_id, score))
            
            return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    async def create_ab_test(
        self,
        test_id: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5
    ) -> bool:
        """Create a new A/B test between two models."""
        return await self.ab_testing.create_test(test_id, model_a, model_b, traffic_split)
    
    async def get_ab_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test status and results."""
        return await self.ab_testing.get_test_status(test_id)
    
    async def get_selection_analytics(self) -> Dict[str, Any]:
        """Get analytics on model selection patterns."""
        async with self._lock:
            if not self.selection_history:
                return {"total_selections": 0}
            
            # Analyze selection patterns
            model_counts = defaultdict(int)
            strategy_counts = defaultdict(int)
            
            for record in self.selection_history:
                model_counts[record["selected_model"]] += 1
                strategy_counts[record["strategy"]] += 1
            
            return {
                "total_selections": len(self.selection_history),
                "model_usage": dict(model_counts),
                "strategy_usage": dict(strategy_counts),
                "most_selected_model": max(model_counts.keys(), key=model_counts.get) if model_counts else None,
                "most_used_strategy": max(strategy_counts.keys(), key=strategy_counts.get) if strategy_counts else None
            }
    
    async def _record_selection(self, request: GenerationRequest, selection: ModelSelection, strategy: str):
        """Record selection for analytics."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request.request_id,
            "selected_model": selection.selected_model,
            "strategy": strategy,
            "confidence_score": selection.confidence_score,
            "estimated_cost": selection.estimated_cost,
            "model_type": request.model_type.value,
            "quality_preference": request.quality_preference
        }
        
        self.selection_history.append(record)


# Utility functions for model management

async def initialize_default_models() -> ModelSelector:
    """Initialize model selector with default models."""
    selector = ModelSelector()
    
    # Register default image generation models
    await selector.register_model(ModelCapabilities(
        model_id="dalle3",
        model_type=ModelType.IMAGE_GENERATION,
        supported_resolutions=[(1024, 1024), (1792, 1024), (1024, 1792)],
        supported_formats=["jpg", "png"],
        max_prompt_length=4000,
        supports_negative_prompts=False,
        supports_style_control=True,
        gpu_memory_required=6.0,
        estimated_processing_time=15.0,
        cost_per_generation=0.040
    ))
    
    await selector.register_model(ModelCapabilities(
        model_id="stable_diffusion_xl",
        model_type=ModelType.IMAGE_GENERATION,
        supported_resolutions=[(512, 512), (768, 768), (1024, 1024)],
        supported_formats=["jpg", "png"],
        max_prompt_length=2000,
        supports_negative_prompts=True,
        supports_style_control=True,
        supports_batch_processing=True,
        gpu_memory_required=8.0,
        estimated_processing_time=25.0,
        cost_per_generation=0.020
    ))
    
    await selector.register_model(ModelCapabilities(
        model_id="midjourney",
        model_type=ModelType.IMAGE_GENERATION,
        supported_resolutions=[(1024, 1024), (1456, 816), (816, 1456)],
        supported_formats=["jpg", "png"],
        max_prompt_length=1000,
        supports_style_control=True,
        gpu_memory_required=4.0,
        estimated_processing_time=45.0,
        cost_per_generation=0.025
    ))
    
    # Initialize with some baseline metrics
    models = ["dalle3", "stable_diffusion_xl", "midjourney"]
    for model_id in models:
        await selector.update_model_metrics(
            model_id=model_id,
            processing_time=random.uniform(15, 45),
            quality_score=random.uniform(0.7, 0.9),
            cost=random.uniform(0.02, 0.05),
            success=True,
            user_satisfaction=random.uniform(0.6, 0.9)
        )
    
    return selector