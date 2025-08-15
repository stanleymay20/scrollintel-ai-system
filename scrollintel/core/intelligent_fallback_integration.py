"""
Intelligent Fallback Integration System for ScrollIntel.
Integrates all fallback components: content generation, progressive loading,
smart caching, and workflow alternatives into a unified system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .intelligent_fallback_manager import (
    intelligent_fallback_manager, ContentContext, ContentType, FallbackContent
)
from .progressive_content_loader import (
    progressive_content_loader, ContentChunk, LoadingProgress, LoadingStage,
    ContentPriority, ProgressiveLoadRequest
)
from .smart_cache_manager import (
    smart_cache_manager, StalenessLevel, cache_get, cache_set
)
from .workflow_alternative_engine import (
    workflow_alternative_engine, WorkflowContext, WorkflowCategory, 
    DifficultyLevel, get_workflow_alternatives
)

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Strategies for handling content failures."""
    IMMEDIATE_FALLBACK = "immediate_fallback"
    PROGRESSIVE_LOADING = "progressive_loading"
    CACHED_CONTENT = "cached_content"
    WORKFLOW_ALTERNATIVE = "workflow_alternative"
    HYBRID = "hybrid"


@dataclass
class IntegratedFallbackRequest:
    """Request for integrated fallback handling."""
    request_id: str
    user_id: Optional[str]
    content_type: ContentType
    original_function: Callable
    original_args: tuple
    original_kwargs: dict
    failure_context: Optional[Exception] = None
    preferred_strategy: FallbackStrategy = FallbackStrategy.HYBRID
    max_wait_time_seconds: float = 30.0
    allow_stale_cache: bool = True
    require_workflow_alternatives: bool = False
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedFallbackResult:
    """Result from integrated fallback system."""
    success: bool
    content: Any
    strategy_used: FallbackStrategy
    fallback_quality: str  # high, medium, low, emergency
    cache_hit: bool
    staleness_level: Optional[StalenessLevel]
    loading_time_seconds: float
    workflow_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    user_message: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentFallbackIntegration:
    """Main integration system for all fallback components."""
    
    def __init__(self):
        self.active_requests: Dict[str, IntegratedFallbackRequest] = {}
        self.strategy_performance: Dict[FallbackStrategy, Dict[str, float]] = {}
        self.user_strategy_preferences: Dict[str, FallbackStrategy] = {}
        
        # Performance tracking
        self.strategy_success_rates: Dict[FallbackStrategy, List[bool]] = {
            strategy: [] for strategy in FallbackStrategy
        }
        self.strategy_response_times: Dict[FallbackStrategy, List[float]] = {
            strategy: [] for strategy in FallbackStrategy
        }
        
        # Configuration
        self.enable_adaptive_strategy_selection = True
        self.cache_fallback_results = True
        self.max_concurrent_requests = 10
        
        # Initialize strategy weights
        self._initialize_strategy_weights()
    
    def _initialize_strategy_weights(self):
        """Initialize strategy performance weights."""
        self.strategy_performance = {
            FallbackStrategy.IMMEDIATE_FALLBACK: {
                "speed": 0.9,
                "quality": 0.3,
                "reliability": 0.8,
                "user_satisfaction": 0.5
            },
            FallbackStrategy.PROGRESSIVE_LOADING: {
                "speed": 0.6,
                "quality": 0.8,
                "reliability": 0.7,
                "user_satisfaction": 0.8
            },
            FallbackStrategy.CACHED_CONTENT: {
                "speed": 0.95,
                "quality": 0.7,
                "reliability": 0.9,
                "user_satisfaction": 0.6
            },
            FallbackStrategy.WORKFLOW_ALTERNATIVE: {
                "speed": 0.3,
                "quality": 0.9,
                "reliability": 0.6,
                "user_satisfaction": 0.9
            },
            FallbackStrategy.HYBRID: {
                "speed": 0.7,
                "quality": 0.8,
                "reliability": 0.8,
                "user_satisfaction": 0.8
            }
        }
    
    async def handle_content_failure(self, request: IntegratedFallbackRequest) -> IntegratedFallbackResult:
        """Handle content failure with integrated fallback strategies."""
        start_time = asyncio.get_event_loop().time()
        self.active_requests[request.request_id] = request
        
        try:
            # Determine optimal strategy
            strategy = await self._select_optimal_strategy(request)
            
            # Execute strategy
            result = await self._execute_strategy(strategy, request)
            
            # Record performance
            execution_time = asyncio.get_event_loop().time() - start_time
            await self._record_strategy_performance(strategy, result.success, execution_time)
            
            # Cache result if successful and caching enabled
            if result.success and self.cache_fallback_results:
                await self._cache_fallback_result(request, result)
            
            result.loading_time_seconds = execution_time
            result.strategy_used = strategy
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated fallback failed for request {request.request_id}: {e}")
            
            # Emergency fallback
            return await self._emergency_fallback(request, e)
            
        finally:
            # Cleanup
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    async def _select_optimal_strategy(self, request: IntegratedFallbackRequest) -> FallbackStrategy:
        """Select the optimal fallback strategy based on context and performance."""
        if request.preferred_strategy != FallbackStrategy.HYBRID:
            return request.preferred_strategy
        
        if not self.enable_adaptive_strategy_selection:
            return FallbackStrategy.HYBRID
        
        # Score each strategy
        strategy_scores = {}
        
        for strategy in FallbackStrategy:
            if strategy == FallbackStrategy.HYBRID:
                continue
                
            score = await self._score_strategy(strategy, request)
            strategy_scores[strategy] = score
        
        # Select best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        logger.debug(f"Selected strategy {best_strategy.value} for request {request.request_id}")
        return best_strategy
    
    async def _score_strategy(self, strategy: FallbackStrategy, request: IntegratedFallbackRequest) -> float:
        """Score a strategy based on context and historical performance."""
        base_weights = self.strategy_performance[strategy]
        score = 0.0
        
        # Context-based scoring
        if request.max_wait_time_seconds < 5.0:
            # Prioritize speed for urgent requests
            score += base_weights["speed"] * 0.4
            score += base_weights["reliability"] * 0.3
            score += base_weights["quality"] * 0.2
            score += base_weights["user_satisfaction"] * 0.1
        elif request.max_wait_time_seconds > 30.0:
            # Prioritize quality for non-urgent requests
            score += base_weights["quality"] * 0.4
            score += base_weights["user_satisfaction"] * 0.3
            score += base_weights["reliability"] * 0.2
            score += base_weights["speed"] * 0.1
        else:
            # Balanced scoring
            score += base_weights["speed"] * 0.25
            score += base_weights["quality"] * 0.25
            score += base_weights["reliability"] * 0.25
            score += base_weights["user_satisfaction"] * 0.25
        
        # Historical performance adjustment
        if strategy in self.strategy_success_rates:
            recent_successes = self.strategy_success_rates[strategy][-20:]  # Last 20 attempts
            if recent_successes:
                success_rate = sum(recent_successes) / len(recent_successes)
                score *= (0.5 + success_rate * 0.5)  # Scale by success rate
        
        # User preference adjustment
        if request.user_id and request.user_id in self.user_strategy_preferences:
            preferred_strategy = self.user_strategy_preferences[request.user_id]
            if strategy == preferred_strategy:
                score *= 1.2  # 20% boost for user preference
        
        # Content type specific adjustments
        if request.content_type == ContentType.CHART and strategy == FallbackStrategy.PROGRESSIVE_LOADING:
            score *= 1.1  # Charts benefit from progressive loading
        elif request.content_type == ContentType.TEXT and strategy == FallbackStrategy.IMMEDIATE_FALLBACK:
            score *= 1.1  # Text content can use immediate fallbacks effectively
        
        return score
    
    async def _execute_strategy(self, strategy: FallbackStrategy, 
                              request: IntegratedFallbackRequest) -> IntegratedFallbackResult:
        """Execute the selected fallback strategy."""
        if strategy == FallbackStrategy.IMMEDIATE_FALLBACK:
            return await self._execute_immediate_fallback(request)
        elif strategy == FallbackStrategy.PROGRESSIVE_LOADING:
            return await self._execute_progressive_loading(request)
        elif strategy == FallbackStrategy.CACHED_CONTENT:
            return await self._execute_cached_content(request)
        elif strategy == FallbackStrategy.WORKFLOW_ALTERNATIVE:
            return await self._execute_workflow_alternative(request)
        elif strategy == FallbackStrategy.HYBRID:
            return await self._execute_hybrid_strategy(request)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    async def _execute_immediate_fallback(self, request: IntegratedFallbackRequest) -> IntegratedFallbackResult:
        """Execute immediate fallback strategy."""
        context = ContentContext(
            user_id=request.user_id,
            content_type=request.content_type,
            original_request={"args": request.original_args, "kwargs": request.original_kwargs},
            error_context=request.failure_context,
            user_preferences=request.user_preferences,
            system_state=request.system_context
        )
        
        fallback_content = await intelligent_fallback_manager.generate_fallback_content(context)
        
        return IntegratedFallbackResult(
            success=True,
            content=fallback_content.content,
            strategy_used=FallbackStrategy.IMMEDIATE_FALLBACK,
            fallback_quality=fallback_content.quality.value,
            cache_hit=False,
            staleness_level=None,
            loading_time_seconds=0.0,
            user_message=fallback_content.user_message,
            suggested_actions=fallback_content.suggested_actions,
            metadata={"confidence": fallback_content.confidence}
        )
    
    async def _execute_progressive_loading(self, request: IntegratedFallbackRequest) -> IntegratedFallbackResult:
        """Execute progressive loading strategy."""
        # Create content chunks for progressive loading
        chunks = [
            ContentChunk(
                chunk_id=f"{request.request_id}_main",
                priority=ContentPriority.CRITICAL,
                content_type=request.content_type,
                loader_function=lambda: request.original_function(*request.original_args, **request.original_kwargs)
            )
        ]
        
        # Create progressive loading request
        load_request = progressive_content_loader.create_loading_request(
            user_id=request.user_id,
            content_chunks=chunks,
            timeout_seconds=request.max_wait_time_seconds
        )
        
        # Execute progressive loading
        final_progress = None
        async for progress in progressive_content_loader.load_content_progressively(load_request):
            final_progress = progress
            if progress.stage == LoadingStage.COMPLETE:
                break
        
        if final_progress and final_progress.partial_results:
            main_result = final_progress.partial_results.get(f"{request.request_id}_main")
            if main_result and main_result.get("content"):
                return IntegratedFallbackResult(
                    success=True,
                    content=main_result["content"],
                    strategy_used=FallbackStrategy.PROGRESSIVE_LOADING,
                    fallback_quality="high",
                    cache_hit=False,
                    staleness_level=None,
                    loading_time_seconds=0.0,
                    metadata={"progress_stage": final_progress.stage.value}
                )
        
        # Progressive loading failed, fall back to immediate fallback
        return await self._execute_immediate_fallback(request)
    
    async def _execute_cached_content(self, request: IntegratedFallbackRequest) -> IntegratedFallbackResult:
        """Execute cached content strategy."""
        # Generate cache key
        cache_key = f"fallback_{request.content_type.value}_{hash(str(request.original_args))}"
        
        # Try to get from cache
        staleness_tolerance = StalenessLevel.VERY_STALE if request.allow_stale_cache else StalenessLevel.MODERATELY_STALE
        cached_content, staleness_level = await cache_get(cache_key, staleness_tolerance=staleness_tolerance)
        
        if cached_content is not None:
            return IntegratedFallbackResult(
                success=True,
                content=cached_content,
                strategy_used=FallbackStrategy.CACHED_CONTENT,
                fallback_quality="medium" if staleness_level == StalenessLevel.FRESH else "low",
                cache_hit=True,
                staleness_level=staleness_level,
                loading_time_seconds=0.0,
                user_message=f"Showing cached content ({staleness_level.value})",
                suggested_actions=["Refresh for latest content"]
            )
        
        # No cached content available, fall back to immediate fallback
        return await self._execute_immediate_fallback(request)
    
    async def _execute_workflow_alternative(self, request: IntegratedFallbackRequest) -> IntegratedFallbackResult:
        """Execute workflow alternative strategy."""
        # Create workflow context
        workflow_context = WorkflowContext(
            user_id=request.user_id,
            original_workflow=request.original_function.__name__,
            failure_reason=str(request.failure_context) if request.failure_context else None,
            user_preferences=request.user_preferences,
            system_capabilities=request.system_context
        )
        
        # Get workflow alternatives
        suggestion_result = await workflow_alternative_engine.suggest_alternatives(workflow_context)
        
        if suggestion_result.alternatives:
            # Generate fallback content with workflow alternatives
            context = ContentContext(
                user_id=request.user_id,
                content_type=request.content_type,
                original_request={"function": request.original_function.__name__},
                error_context=request.failure_context
            )
            
            fallback_content = await intelligent_fallback_manager.generate_fallback_content(context)
            
            # Convert alternatives to dict format
            alternatives_dict = [
                {
                    "id": alt.alternative_id,
                    "name": alt.name,
                    "description": alt.description,
                    "difficulty": alt.difficulty.value,
                    "estimated_time": alt.estimated_total_time_minutes,
                    "success_probability": alt.success_probability,
                    "steps": [
                        {
                            "title": step.title,
                            "description": step.description,
                            "time": step.estimated_time_minutes
                        }
                        for step in alt.steps
                    ]
                }
                for alt in suggestion_result.alternatives
            ]
            
            return IntegratedFallbackResult(
                success=True,
                content=fallback_content.content,
                strategy_used=FallbackStrategy.WORKFLOW_ALTERNATIVE,
                fallback_quality="high",
                cache_hit=False,
                staleness_level=None,
                loading_time_seconds=0.0,
                workflow_alternatives=alternatives_dict,
                user_message=f"Here are {len(alternatives_dict)} alternative approaches you can try",
                suggested_actions=["Choose an alternative workflow", "Try the recommended approach"],
                metadata={
                    "confidence": suggestion_result.confidence_score,
                    "reasoning": suggestion_result.reasoning
                }
            )
        
        # No alternatives found, fall back to immediate fallback
        return await self._execute_immediate_fallback(request)
    
    async def _execute_hybrid_strategy(self, request: IntegratedFallbackRequest) -> IntegratedFallbackResult:
        """Execute hybrid strategy combining multiple approaches."""
        # Start with immediate fallback for quick response
        immediate_task = asyncio.create_task(self._execute_immediate_fallback(request))
        
        # Try cached content in parallel
        cache_task = asyncio.create_task(self._execute_cached_content(request))
        
        # Wait for first successful result
        done, pending = await asyncio.wait(
            [immediate_task, cache_task],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=5.0  # Quick timeout for initial response
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Get result from completed task
        if done:
            result = await next(iter(done))
            if result.success:
                # If we have time, also get workflow alternatives
                if request.max_wait_time_seconds > 10.0:
                    try:
                        workflow_result = await asyncio.wait_for(
                            self._execute_workflow_alternative(request),
                            timeout=request.max_wait_time_seconds - 5.0
                        )
                        if workflow_result.workflow_alternatives:
                            result.workflow_alternatives = workflow_result.workflow_alternatives
                            result.metadata.update(workflow_result.metadata)
                    except asyncio.TimeoutError:
                        pass  # Continue with current result
                
                result.strategy_used = FallbackStrategy.HYBRID
                return result
        
        # If quick strategies failed, try progressive loading
        if request.max_wait_time_seconds > 15.0:
            try:
                return await asyncio.wait_for(
                    self._execute_progressive_loading(request),
                    timeout=request.max_wait_time_seconds - 5.0
                )
            except asyncio.TimeoutError:
                pass
        
        # Final fallback
        return await self._execute_immediate_fallback(request)
    
    async def _emergency_fallback(self, request: IntegratedFallbackRequest, 
                                error: Exception) -> IntegratedFallbackResult:
        """Emergency fallback when all strategies fail."""
        emergency_content = {
            "error": "Content temporarily unavailable",
            "message": "We're experiencing technical difficulties. Please try again in a moment.",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.request_id
        }
        
        return IntegratedFallbackResult(
            success=False,
            content=emergency_content,
            strategy_used=FallbackStrategy.IMMEDIATE_FALLBACK,
            fallback_quality="emergency",
            cache_hit=False,
            staleness_level=None,
            loading_time_seconds=0.0,
            user_message="Service temporarily unavailable. Our team has been notified.",
            suggested_actions=[
                "Try again in a few minutes",
                "Contact support if the issue persists",
                "Use alternative features"
            ],
            metadata={"emergency": True, "error": str(error)}
        )
    
    async def _cache_fallback_result(self, request: IntegratedFallbackRequest, 
                                   result: IntegratedFallbackResult):
        """Cache successful fallback result for future use."""
        if not result.success:
            return
        
        cache_key = f"fallback_{request.content_type.value}_{hash(str(request.original_args))}"
        
        # Cache with appropriate TTL based on quality
        ttl_seconds = 3600  # 1 hour default
        if result.fallback_quality == "high":
            ttl_seconds = 7200  # 2 hours
        elif result.fallback_quality == "low":
            ttl_seconds = 1800  # 30 minutes
        elif result.fallback_quality == "emergency":
            ttl_seconds = 300   # 5 minutes
        
        await cache_set(
            cache_key,
            result.content,
            ttl_seconds=ttl_seconds,
            tags=[f"fallback_{request.content_type.value}", f"user_{request.user_id}"]
        )
    
    async def _record_strategy_performance(self, strategy: FallbackStrategy, 
                                         success: bool, execution_time: float):
        """Record strategy performance for adaptive selection."""
        self.strategy_success_rates[strategy].append(success)
        self.strategy_response_times[strategy].append(execution_time)
        
        # Keep only recent performance data
        max_history = 100
        if len(self.strategy_success_rates[strategy]) > max_history:
            self.strategy_success_rates[strategy] = self.strategy_success_rates[strategy][-max_history:]
        if len(self.strategy_response_times[strategy]) > max_history:
            self.strategy_response_times[strategy] = self.strategy_response_times[strategy][-max_history:]
    
    def set_user_strategy_preference(self, user_id: str, preferred_strategy: FallbackStrategy):
        """Set user's preferred fallback strategy."""
        self.user_strategy_preferences[user_id] = preferred_strategy
        logger.info(f"Set strategy preference for user {user_id}: {preferred_strategy.value}")
    
    def get_strategy_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all strategies."""
        stats = {}
        
        for strategy in FallbackStrategy:
            success_rates = self.strategy_success_rates[strategy]
            response_times = self.strategy_response_times[strategy]
            
            stats[strategy.value] = {
                "success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
                "total_attempts": len(success_rates)
            }
        
        return stats
    
    async def test_all_strategies(self, request: IntegratedFallbackRequest) -> Dict[str, IntegratedFallbackResult]:
        """Test all strategies for comparison (useful for debugging/optimization)."""
        results = {}
        
        for strategy in FallbackStrategy:
            if strategy == FallbackStrategy.HYBRID:
                continue
                
            try:
                test_request = IntegratedFallbackRequest(
                    request_id=f"{request.request_id}_test_{strategy.value}",
                    user_id=request.user_id,
                    content_type=request.content_type,
                    original_function=request.original_function,
                    original_args=request.original_args,
                    original_kwargs=request.original_kwargs,
                    failure_context=request.failure_context,
                    preferred_strategy=strategy,
                    max_wait_time_seconds=10.0  # Shorter timeout for testing
                )
                
                result = await self._execute_strategy(strategy, test_request)
                results[strategy.value] = result
                
            except Exception as e:
                logger.error(f"Strategy test failed for {strategy.value}: {e}")
                results[strategy.value] = IntegratedFallbackResult(
                    success=False,
                    content=None,
                    strategy_used=strategy,
                    fallback_quality="emergency",
                    cache_hit=False,
                    staleness_level=None,
                    loading_time_seconds=0.0,
                    metadata={"test_error": str(e)}
                )
        
        return results


# Global instance
intelligent_fallback_integration = IntelligentFallbackIntegration()


# Convenience functions and decorators
def with_intelligent_fallback(content_type: ContentType, 
                            strategy: FallbackStrategy = FallbackStrategy.HYBRID,
                            max_wait_time: float = 30.0):
    """Decorator to add intelligent fallback to any function."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            try:
                # Try original function first
                return await func(*args, **kwargs)
            except Exception as e:
                # Create fallback request
                request = IntegratedFallbackRequest(
                    request_id=f"fallback_{func.__name__}_{id(args)}",
                    user_id=kwargs.get('user_id'),
                    content_type=content_type,
                    original_function=func,
                    original_args=args,
                    original_kwargs=kwargs,
                    failure_context=e,
                    preferred_strategy=strategy,
                    max_wait_time_seconds=max_wait_time
                )
                
                # Handle with integrated fallback
                result = await intelligent_fallback_integration.handle_content_failure(request)
                
                if result.success:
                    return result.content
                else:
                    raise e  # Re-raise original exception if fallback fails
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


async def get_intelligent_fallback(content_type: ContentType, original_function: Callable,
                                 args: tuple, kwargs: dict, error: Exception = None,
                                 user_id: str = None, strategy: FallbackStrategy = FallbackStrategy.HYBRID) -> IntegratedFallbackResult:
    """Get intelligent fallback for failed content."""
    request = IntegratedFallbackRequest(
        request_id=f"manual_fallback_{id(args)}",
        user_id=user_id,
        content_type=content_type,
        original_function=original_function,
        original_args=args,
        original_kwargs=kwargs,
        failure_context=error,
        preferred_strategy=strategy
    )
    
    return await intelligent_fallback_integration.handle_content_failure(request)