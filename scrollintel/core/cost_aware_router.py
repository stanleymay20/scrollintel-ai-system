"""
Cost-Aware Routing for ScrollIntel-G6.
Implements dynamic programming to choose optimal strategy based on utility/cost ratio.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .council_of_models import ModelType, council_deliberation
from .proof_of_workflow import create_workflow_attestation

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    SINGLE_PASS = "single_pass"
    BEST_OF_N = "best_of_n"
    COUNCIL_DELIBERATION = "council_deliberation"
    CACHED_RESPONSE = "cached_response"
    LIGHTWEIGHT_FILTER = "lightweight_filter"


@dataclass
class RoutingCost:
    """Cost structure for different routing strategies."""
    compute_cost: float
    latency_cost: float
    token_cost: float
    total_cost: float


@dataclass
class RoutingUtility:
    """Utility metrics for routing decisions."""
    accuracy_score: float
    confidence_score: float
    completeness_score: float
    alignment_score: float
    total_utility: float


@dataclass
class RoutingDecision:
    """Decision made by the cost-aware router."""
    strategy: RoutingStrategy
    model_type: Optional[ModelType]
    expected_cost: RoutingCost
    expected_utility: RoutingUtility
    utility_cost_ratio: float
    reasoning: str


class CostModel:
    """Model for estimating costs of different routing strategies."""
    
    def __init__(self):
        # Base costs per strategy (in normalized units)
        self.base_costs = {
            RoutingStrategy.SINGLE_PASS: RoutingCost(1.0, 0.5, 1.0, 2.5),
            RoutingStrategy.BEST_OF_N: RoutingCost(3.0, 1.0, 3.0, 7.0),
            RoutingStrategy.COUNCIL_DELIBERATION: RoutingCost(5.0, 2.0, 5.0, 12.0),
            RoutingStrategy.CACHED_RESPONSE: RoutingCost(0.1, 0.1, 0.0, 0.2),
            RoutingStrategy.LIGHTWEIGHT_FILTER: RoutingCost(0.5, 0.2, 0.5, 1.2),
        }
        
        # Model-specific cost multipliers
        self.model_multipliers = {
            ModelType.SCROLL_CORE_M: 1.0,
            ModelType.GPT_5: 1.5,
            ModelType.CLAUDE_3_5: 1.3,
            ModelType.GEMINI_PRO: 1.2,
            ModelType.DEEPSEEK_V3: 0.8,
            ModelType.LLAMA_3_1: 0.9,
        }
    
    def estimate_cost(
        self,
        strategy: RoutingStrategy,
        model_type: Optional[ModelType] = None,
        task_complexity: float = 1.0,
        context_length: int = 1000
    ) -> RoutingCost:
        """Estimate cost for a routing strategy."""
        
        base_cost = self.base_costs[strategy]
        
        # Apply model multiplier
        model_multiplier = 1.0
        if model_type:
            model_multiplier = self.model_multipliers.get(model_type, 1.0)
        
        # Apply complexity and context multipliers
        complexity_multiplier = max(0.5, min(2.0, task_complexity))
        context_multiplier = max(0.8, min(1.5, context_length / 1000))
        
        total_multiplier = model_multiplier * complexity_multiplier * context_multiplier
        
        return RoutingCost(
            compute_cost=base_cost.compute_cost * total_multiplier,
            latency_cost=base_cost.latency_cost * total_multiplier,
            token_cost=base_cost.token_cost * total_multiplier,
            total_cost=base_cost.total_cost * total_multiplier
        )


class UtilityModel:
    """Model for estimating utility of different routing strategies."""
    
    def __init__(self):
        # Base utility scores per strategy
        self.base_utilities = {
            RoutingStrategy.SINGLE_PASS: RoutingUtility(0.7, 0.6, 0.7, 0.6, 0.65),
            RoutingStrategy.BEST_OF_N: RoutingUtility(0.8, 0.7, 0.8, 0.7, 0.75),
            RoutingStrategy.COUNCIL_DELIBERATION: RoutingUtility(0.9, 0.9, 0.9, 0.95, 0.91),
            RoutingStrategy.CACHED_RESPONSE: RoutingUtility(0.8, 0.9, 0.8, 0.8, 0.82),
            RoutingStrategy.LIGHTWEIGHT_FILTER: RoutingUtility(0.6, 0.5, 0.6, 0.7, 0.6),
        }
        
        # Model-specific utility multipliers
        self.model_multipliers = {
            ModelType.SCROLL_CORE_M: 1.1,
            ModelType.GPT_5: 1.0,
            ModelType.CLAUDE_3_5: 0.95,
            ModelType.GEMINI_PRO: 0.9,
            ModelType.DEEPSEEK_V3: 0.85,
            ModelType.LLAMA_3_1: 0.88,
        }
    
    def estimate_utility(
        self,
        strategy: RoutingStrategy,
        model_type: Optional[ModelType] = None,
        task_importance: float = 1.0,
        risk_level: float = 0.5
    ) -> RoutingUtility:
        """Estimate utility for a routing strategy."""
        
        base_utility = self.base_utilities[strategy]
        
        # Apply model multiplier
        model_multiplier = 1.0
        if model_type:
            model_multiplier = self.model_multipliers.get(model_type, 1.0)
        
        # Apply importance and risk multipliers
        importance_multiplier = max(0.8, min(1.2, task_importance))
        risk_multiplier = 1.0 + (risk_level * 0.2)  # Higher risk increases utility need
        
        total_multiplier = model_multiplier * importance_multiplier * risk_multiplier
        
        return RoutingUtility(
            accuracy_score=min(1.0, base_utility.accuracy_score * total_multiplier),
            confidence_score=min(1.0, base_utility.confidence_score * total_multiplier),
            completeness_score=min(1.0, base_utility.completeness_score * total_multiplier),
            alignment_score=min(1.0, base_utility.alignment_score * total_multiplier),
            total_utility=min(1.0, base_utility.total_utility * total_multiplier)
        )


class SemanticCache:
    """Semantic cache for storing and retrieving similar responses."""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.hit_count = 0
        self.miss_count = 0
    
    def _compute_semantic_hash(self, prompt: str, context: Dict[str, Any]) -> str:
        """Compute semantic hash for caching."""
        import hashlib
        
        # Normalize prompt and context for semantic similarity
        normalized_prompt = prompt.lower().strip()
        context_str = str(sorted(context.items()))
        
        combined = f"{normalized_prompt}|{context_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, prompt: str, context: Dict[str, Any]) -> Optional[Any]:
        """Get cached response if available and not expired."""
        cache_key = self._compute_semantic_hash(prompt, context)
        
        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            
            # Check if expired
            if datetime.utcnow() - timestamp > self.ttl:
                del self.cache[cache_key]
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            logger.debug(f"Cache hit for key: {cache_key[:16]}...")
            return response
        
        self.miss_count += 1
        return None
    
    def put(self, prompt: str, context: Dict[str, Any], response: Any) -> None:
        """Store response in cache."""
        cache_key = self._compute_semantic_hash(prompt, context)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = (response, datetime.utcnow())
        logger.debug(f"Cached response for key: {cache_key[:16]}...")
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class CostAwareRouter:
    """Routes requests to optimal strategy based on utility/cost analysis."""
    
    def __init__(self):
        self.cost_model = CostModel()
        self.utility_model = UtilityModel()
        self.cache = SemanticCache()
        self.budget_tracker = BudgetTracker()
        
        # Routing thresholds
        self.high_risk_threshold = 0.8
        self.council_utility_threshold = 0.85
        self.cache_similarity_threshold = 0.9
    
    async def route_request(
        self,
        prompt: str,
        context: Dict[str, Any],
        user_id: str,
        task_importance: float = 1.0,
        risk_level: float = 0.5,
        budget_limit: Optional[float] = None
    ) -> Tuple[Any, RoutingDecision]:
        """Route request to optimal strategy."""
        
        logger.info(f"Routing request with importance={task_importance}, risk={risk_level}")
        
        # Check cache first
        cached_response = self.cache.get(prompt, context)
        if cached_response and risk_level < 0.7:
            decision = RoutingDecision(
                strategy=RoutingStrategy.CACHED_RESPONSE,
                model_type=None,
                expected_cost=self.cost_model.estimate_cost(RoutingStrategy.CACHED_RESPONSE),
                expected_utility=self.utility_model.estimate_utility(RoutingStrategy.CACHED_RESPONSE),
                utility_cost_ratio=4.1,  # High ratio for cached responses
                reasoning="High-quality cached response available"
            )
            
            # Create attestation
            create_workflow_attestation(
                action_type="cached_routing",
                agent_id="cost_aware_router",
                user_id=user_id,
                prompt=prompt,
                tools_used=["semantic_cache"],
                datasets_used=[],
                model_version="router-v1.0",
                verifier_evidence={"cache_hit": True, "hit_ratio": self.cache.get_hit_ratio()},
                content=cached_response
            )
            
            return cached_response, decision
        
        # Evaluate all possible strategies
        strategies = await self._evaluate_strategies(
            prompt, context, task_importance, risk_level, budget_limit
        )
        
        # Select optimal strategy
        optimal_strategy = max(strategies, key=lambda s: s.utility_cost_ratio)
        
        # Execute selected strategy
        response = await self._execute_strategy(optimal_strategy, prompt, context, user_id)
        
        # Cache response if appropriate
        if optimal_strategy.strategy != RoutingStrategy.CACHED_RESPONSE:
            self.cache.put(prompt, context, response)
        
        # Update budget tracker
        self.budget_tracker.record_usage(user_id, optimal_strategy.expected_cost.total_cost)
        
        logger.info(f"Selected strategy: {optimal_strategy.strategy.value} (ratio: {optimal_strategy.utility_cost_ratio:.2f})")
        
        return response, optimal_strategy
    
    async def _evaluate_strategies(
        self,
        prompt: str,
        context: Dict[str, Any],
        task_importance: float,
        risk_level: float,
        budget_limit: Optional[float]
    ) -> List[RoutingDecision]:
        """Evaluate all possible routing strategies."""
        
        strategies = []
        task_complexity = self._estimate_task_complexity(prompt, context)
        context_length = len(prompt) + len(str(context))
        
        # Evaluate each strategy
        for strategy in RoutingStrategy:
            if strategy == RoutingStrategy.CACHED_RESPONSE:
                continue  # Already handled
            
            # Select best model for this strategy
            best_model = self._select_best_model(strategy, task_importance, risk_level)
            
            # Estimate cost and utility
            cost = self.cost_model.estimate_cost(strategy, best_model, task_complexity, context_length)
            utility = self.utility_model.estimate_utility(strategy, best_model, task_importance, risk_level)
            
            # Check budget constraint
            if budget_limit and cost.total_cost > budget_limit:
                continue
            
            # Calculate utility/cost ratio
            ratio = utility.total_utility / max(0.1, cost.total_cost)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(strategy, best_model, cost, utility, ratio)
            
            decision = RoutingDecision(
                strategy=strategy,
                model_type=best_model,
                expected_cost=cost,
                expected_utility=utility,
                utility_cost_ratio=ratio,
                reasoning=reasoning
            )
            
            strategies.append(decision)
        
        return strategies
    
    def _estimate_task_complexity(self, prompt: str, context: Dict[str, Any]) -> float:
        """Estimate task complexity based on prompt and context."""
        
        complexity_indicators = [
            len(prompt) > 1000,  # Long prompts
            "analyze" in prompt.lower(),
            "complex" in prompt.lower(),
            "detailed" in prompt.lower(),
            len(context) > 5,  # Rich context
            any(isinstance(v, list) and len(v) > 10 for v in context.values()),  # Large data
        ]
        
        base_complexity = sum(complexity_indicators) / len(complexity_indicators)
        return max(0.5, min(2.0, base_complexity * 2))
    
    def _select_best_model(
        self,
        strategy: RoutingStrategy,
        task_importance: float,
        risk_level: float
    ) -> Optional[ModelType]:
        """Select the best model for a given strategy."""
        
        if strategy == RoutingStrategy.COUNCIL_DELIBERATION:
            return None  # Uses multiple models
        
        # For high-risk tasks, prefer ScrollCore-M
        if risk_level > self.high_risk_threshold:
            return ModelType.SCROLL_CORE_M
        
        # For important tasks, use high-quality models
        if task_importance > 1.5:
            return ModelType.GPT_5
        
        # Default to ScrollCore-M for alignment
        return ModelType.SCROLL_CORE_M
    
    def _generate_reasoning(
        self,
        strategy: RoutingStrategy,
        model_type: Optional[ModelType],
        cost: RoutingCost,
        utility: RoutingUtility,
        ratio: float
    ) -> str:
        """Generate human-readable reasoning for strategy selection."""
        
        model_name = model_type.value if model_type else "multiple"
        
        return f"Strategy {strategy.value} with {model_name}: " \
               f"Cost={cost.total_cost:.2f}, Utility={utility.total_utility:.2f}, " \
               f"Ratio={ratio:.2f}. " \
               f"Selected for optimal utility/cost balance."
    
    async def _execute_strategy(
        self,
        decision: RoutingDecision,
        prompt: str,
        context: Dict[str, Any],
        user_id: str
    ) -> Any:
        """Execute the selected routing strategy."""
        
        if decision.strategy == RoutingStrategy.COUNCIL_DELIBERATION:
            council_decision = await council_deliberation(prompt, context, user_id, high_risk=True)
            return council_decision.final_content
        
        elif decision.strategy == RoutingStrategy.SINGLE_PASS:
            # Execute single model pass
            return await self._execute_single_pass(decision.model_type, prompt, context)
        
        elif decision.strategy == RoutingStrategy.BEST_OF_N:
            # Execute best-of-N sampling
            return await self._execute_best_of_n(decision.model_type, prompt, context, n=3)
        
        elif decision.strategy == RoutingStrategy.LIGHTWEIGHT_FILTER:
            # Execute with lightweight pre-filtering
            return await self._execute_lightweight_filter(decision.model_type, prompt, context)
        
        else:
            raise ValueError(f"Unknown strategy: {decision.strategy}")
    
    async def _execute_single_pass(self, model_type: ModelType, prompt: str, context: Dict[str, Any]) -> str:
        """Execute single model pass."""
        # This would integrate with actual model clients
        return f"Single pass response from {model_type.value}: {prompt[:100]}..."
    
    async def _execute_best_of_n(self, model_type: ModelType, prompt: str, context: Dict[str, Any], n: int) -> str:
        """Execute best-of-N sampling."""
        # This would generate N responses and select the best
        return f"Best-of-{n} response from {model_type.value}: {prompt[:100]}..."
    
    async def _execute_lightweight_filter(self, model_type: ModelType, prompt: str, context: Dict[str, Any]) -> str:
        """Execute with lightweight pre-filtering."""
        # This would use a lightweight model to filter/enhance the prompt
        return f"Filtered response from {model_type.value}: {prompt[:100]}..."


class BudgetTracker:
    """Tracks and manages budget allocation for different users/tenants."""
    
    def __init__(self):
        self.usage: Dict[str, float] = {}
        self.limits: Dict[str, float] = {}
        self.reset_time = datetime.utcnow()
    
    def set_budget_limit(self, user_id: str, limit: float) -> None:
        """Set budget limit for a user."""
        self.limits[user_id] = limit
    
    def record_usage(self, user_id: str, cost: float) -> None:
        """Record usage for a user."""
        if user_id not in self.usage:
            self.usage[user_id] = 0.0
        self.usage[user_id] += cost
    
    def get_remaining_budget(self, user_id: str) -> Optional[float]:
        """Get remaining budget for a user."""
        if user_id not in self.limits:
            return None
        
        used = self.usage.get(user_id, 0.0)
        return max(0.0, self.limits[user_id] - used)
    
    def is_over_budget(self, user_id: str) -> bool:
        """Check if user is over budget."""
        remaining = self.get_remaining_budget(user_id)
        return remaining is not None and remaining <= 0


# Global router instance
router = CostAwareRouter()


async def route_request(
    prompt: str,
    context: Dict[str, Any],
    user_id: str,
    task_importance: float = 1.0,
    risk_level: float = 0.5,
    budget_limit: Optional[float] = None
) -> Tuple[Any, RoutingDecision]:
    """Route request using cost-aware routing (convenience function)."""
    return await router.route_request(
        prompt, context, user_id, task_importance, risk_level, budget_limit
    )


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "hit_ratio": router.cache.get_hit_ratio(),
        "cache_size": len(router.cache.cache),
        "hit_count": router.cache.hit_count,
        "miss_count": router.cache.miss_count
    }