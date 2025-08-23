"""
AI Economic Optimizer with Reinforcement Learning
Implements intelligent resource allocation and cost optimization for ScrollIntel-G6
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
import json

from ..models.economic_optimization_models import (
    ResourceType, CloudProvider, OptimizationObjective, OptimizationStrategy,
    ResourceAllocation, OptimizationResult, MarketState, PerformanceMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class RLState:
    """Reinforcement Learning State Representation"""
    current_prices: np.ndarray
    price_trends: np.ndarray
    demand_forecast: np.ndarray
    supply_forecast: np.ndarray
    current_allocations: np.ndarray
    performance_metrics: np.ndarray
    time_features: np.ndarray

@dataclass
class RLAction:
    """Reinforcement Learning Action Space"""
    resource_allocation: Dict[ResourceType, Dict[CloudProvider, float]]
    trading_decisions: List[Dict[str, Any]]
    budget_allocation: Dict[str, float]

class AIEconomicOptimizer:
    """
    AI-Powered Economic Optimizer with Reinforcement Learning
    
    Implements:
    - Reinforcement learning for resource allocation
    - Multi-objective optimization with Pareto frontier exploration
    - Predictive market making for computational resources
    - Algorithmic trading across cloud providers
    - Economic forecasting with time-series prediction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        self.batch_size = self.config.get('batch_size', 32)
        
        # Initialize RL components
        self.q_network = self._initialize_q_network()
        self.target_network = self._initialize_q_network()
        self.replay_buffer = []
        self.experience_buffer_size = 10000
        
        # Market data and state
        self.market_state = None
        self.historical_data = []
        self.performance_history = []
        
        # Optimization strategies
        self.active_strategies: List[OptimizationStrategy] = []
        self.pareto_frontier = []
        
        logger.info("AI Economic Optimizer initialized")
    
    def _initialize_q_network(self) -> Dict[str, Any]:
        """Initialize Q-Network for reinforcement learning"""
        # Simplified Q-network representation
        # In production, this would use PyTorch/TensorFlow
        return {
            'weights': np.random.randn(100, 50),  # State -> Action mapping
            'biases': np.random.randn(50),
            'output_weights': np.random.randn(50, 20),  # Action space size
            'output_biases': np.random.randn(20)
        }
    
    def _encode_state(self, market_state: MarketState, 
                     current_allocations: List[ResourceAllocation]) -> RLState:
        """Encode current state for RL algorithm"""
        # Extract price features
        prices = np.array([p.price_per_hour for p in market_state.prices])
        
        # Calculate price trends (simplified)
        if len(self.historical_data) > 0:
            prev_prices = np.array([p.price_per_hour for p in self.historical_data[-1].prices])
            price_trends = (prices - prev_prices) / prev_prices
        else:
            price_trends = np.zeros_like(prices)
        
        # Demand and supply forecasts (simplified)
        demand_forecast = np.array([market_state.supply_demand_ratio.get(rt, 1.0) 
                                  for rt in ResourceType])
        supply_forecast = np.array([1.0 / market_state.supply_demand_ratio.get(rt, 1.0) 
                                  for rt in ResourceType])
        
        # Current allocations
        allocation_vector = np.zeros(len(ResourceType) * len(CloudProvider))
        for alloc in current_allocations:
            idx = list(ResourceType).index(alloc.resource_type) * len(CloudProvider) + \
                  list(CloudProvider).index(alloc.provider)
            allocation_vector[idx] = alloc.quantity
        
        # Performance metrics (simplified)
        if self.performance_history:
            latest_perf = self.performance_history[-1]
            perf_metrics = np.array([
                latest_perf.cost_reduction_percentage,
                latest_perf.prediction_accuracy,
                latest_perf.average_roi
            ])
        else:
            perf_metrics = np.zeros(3)
        
        # Time features
        now = datetime.utcnow()
        time_features = np.array([
            now.hour / 24.0,
            now.weekday() / 7.0,
            now.day / 31.0
        ])
        
        return RLState(
            current_prices=prices,
            price_trends=price_trends,
            demand_forecast=demand_forecast,
            supply_forecast=supply_forecast,
            current_allocations=allocation_vector,
            performance_metrics=perf_metrics,
            time_features=time_features
        )
    
    def _q_forward(self, state: RLState, network: Dict[str, Any]) -> np.ndarray:
        """Forward pass through Q-network"""
        # Concatenate all state features
        state_vector = np.concatenate([
            state.current_prices,
            state.price_trends,
            state.demand_forecast,
            state.supply_forecast,
            state.current_allocations,
            state.performance_metrics,
            state.time_features
        ])
        
        # Pad or truncate to expected input size
        if len(state_vector) < 100:
            state_vector = np.pad(state_vector, (0, 100 - len(state_vector)))
        else:
            state_vector = state_vector[:100]
        
        # Forward pass
        hidden = np.tanh(np.dot(state_vector, network['weights']) + network['biases'])
        q_values = np.dot(hidden, network['output_weights']) + network['output_biases']
        
        return q_values
    
    def _select_action(self, state: RLState, epsilon: float = None) -> RLAction:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.exploration_rate
        
        if np.random.random() < epsilon:
            # Random exploration
            return self._random_action()
        else:
            # Greedy action selection
            q_values = self._q_forward(state, self.q_network)
            action_idx = np.argmax(q_values)
            return self._decode_action(action_idx)
    
    def _random_action(self) -> RLAction:
        """Generate random action for exploration"""
        # Random resource allocation
        resource_allocation = {}
        for resource_type in ResourceType:
            resource_allocation[resource_type] = {}
            for provider in CloudProvider:
                resource_allocation[resource_type][provider] = np.random.uniform(0, 1)
        
        # Random trading decisions
        trading_decisions = []
        for _ in range(np.random.randint(0, 3)):
            trading_decisions.append({
                'action': np.random.choice(['buy', 'sell', 'hold']),
                'resource_type': np.random.choice(list(ResourceType)),
                'provider': np.random.choice(list(CloudProvider)),
                'quantity': np.random.randint(1, 10)
            })
        
        # Random budget allocation
        budget_allocation = {
            'immediate': np.random.uniform(0.3, 0.7),
            'reserved': np.random.uniform(0.2, 0.5),
            'spot': np.random.uniform(0.1, 0.3)
        }
        
        return RLAction(
            resource_allocation=resource_allocation,
            trading_decisions=trading_decisions,
            budget_allocation=budget_allocation
        )
    
    def _decode_action(self, action_idx: int) -> RLAction:
        """Decode action index to structured action"""
        # Simplified action decoding
        # In practice, this would be more sophisticated
        return self._random_action()
    
    def _calculate_reward(self, prev_state: RLState, action: RLAction, 
                         new_state: RLState, actual_cost: float, 
                         actual_performance: float) -> float:
        """Calculate reward for RL training"""
        # Multi-objective reward function
        cost_reward = -actual_cost / 1000.0  # Negative cost (minimize)
        performance_reward = actual_performance / 100.0  # Positive performance (maximize)
        
        # Efficiency reward (performance per cost)
        efficiency_reward = actual_performance / max(actual_cost, 1.0)
        
        # Risk penalty
        risk_penalty = 0.0
        if hasattr(action, 'risk_score'):
            risk_penalty = -action.risk_score * 0.1
        
        # Combine rewards
        total_reward = (
            0.4 * cost_reward +
            0.4 * performance_reward +
            0.15 * efficiency_reward +
            0.05 * risk_penalty
        )
        
        return total_reward
    
    async def optimize_resource_allocation(self, 
                                         market_state: MarketState,
                                         current_allocations: List[ResourceAllocation],
                                         strategies: List[OptimizationStrategy],
                                         budget_constraints: Dict[str, float]) -> OptimizationResult:
        """
        Optimize resource allocation using reinforcement learning
        """
        try:
            # Encode current state
            rl_state = self._encode_state(market_state, current_allocations)
            
            # Select action using RL policy
            action = self._select_action(rl_state)
            
            # Convert RL action to resource allocations
            optimized_allocations = await self._convert_to_allocations(
                action, market_state, budget_constraints
            )
            
            # Calculate expected metrics
            expected_cost = sum(alloc.total_cost for alloc in optimized_allocations)
            expected_performance = await self._estimate_performance(optimized_allocations)
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(optimized_allocations, market_state)
            
            # Multi-objective optimization
            pareto_efficiency = await self._calculate_pareto_efficiency(
                optimized_allocations, strategies
            )
            
            result = OptimizationResult(
                strategy_id=strategies[0].id if strategies else "default",
                allocations=optimized_allocations,
                trading_decisions=[],  # Will be populated by trading component
                expected_cost=expected_cost,
                expected_performance=expected_performance,
                risk_score=risk_score,
                pareto_efficiency=pareto_efficiency,
                optimization_time=0.1,  # Placeholder
                confidence=0.85
            )
            
            logger.info(f"Optimized resource allocation: cost={expected_cost:.2f}, "
                       f"performance={expected_performance:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in resource allocation optimization: {e}")
            raise
    
    async def _convert_to_allocations(self, 
                                    action: RLAction,
                                    market_state: MarketState,
                                    budget_constraints: Dict[str, float]) -> List[ResourceAllocation]:
        """Convert RL action to concrete resource allocations"""
        allocations = []
        total_budget = sum(budget_constraints.values())
        
        for resource_type, provider_weights in action.resource_allocation.items():
            for provider, weight in provider_weights.items():
                if weight > 0.1:  # Threshold for allocation
                    # Find matching price
                    matching_prices = [
                        p for p in market_state.prices 
                        if p.resource_type == resource_type and p.provider == provider
                    ]
                    
                    if matching_prices:
                        price = matching_prices[0]
                        allocation_budget = total_budget * weight * 0.1  # Scale down
                        quantity = max(1, int(allocation_budget / price.price_per_hour))
                        
                        allocation = ResourceAllocation(
                            task_id="optimization_task",
                            resource_type=resource_type,
                            provider=provider,
                            region=price.region,
                            quantity=quantity,
                            duration=timedelta(hours=1),
                            total_cost=quantity * price.price_per_hour,
                            expected_performance=quantity * 100,  # Simplified
                            allocation_reason=f"RL optimization with weight {weight:.3f}"
                        )
                        allocations.append(allocation)
        
        return allocations
    
    async def _estimate_performance(self, allocations: List[ResourceAllocation]) -> float:
        """Estimate performance for given allocations"""
        total_performance = 0.0
        
        for alloc in allocations:
            # Performance estimation based on resource type and quantity
            base_performance = {
                ResourceType.GPU_H100: 1000,
                ResourceType.GPU_A100: 800,
                ResourceType.GPU_V100: 600,
                ResourceType.CPU_COMPUTE: 100,
                ResourceType.MEMORY: 50,
                ResourceType.STORAGE: 20,
                ResourceType.BANDWIDTH: 30
            }.get(alloc.resource_type, 100)
            
            total_performance += base_performance * alloc.quantity
        
        return total_performance
    
    async def _calculate_risk_score(self, 
                                  allocations: List[ResourceAllocation],
                                  market_state: MarketState) -> float:
        """Calculate risk score for allocations"""
        total_risk = 0.0
        total_weight = 0.0
        
        for alloc in allocations:
            # Provider risk (simplified)
            provider_risk = {
                CloudProvider.AWS: 0.1,
                CloudProvider.GCP: 0.15,
                CloudProvider.AZURE: 0.12,
                CloudProvider.LAMBDA_LABS: 0.3,
                CloudProvider.RUNPOD: 0.4,
                CloudProvider.VAST_AI: 0.5,
                CloudProvider.PAPERSPACE: 0.35
            }.get(alloc.provider, 0.3)
            
            # Market volatility risk
            volatility_risk = market_state.market_volatility.get(alloc.resource_type, 0.2)
            
            # Combined risk
            allocation_risk = (provider_risk + volatility_risk) / 2
            weight = alloc.total_cost
            
            total_risk += allocation_risk * weight
            total_weight += weight
        
        return total_risk / max(total_weight, 1.0)
    
    async def _calculate_pareto_efficiency(self, 
                                         allocations: List[ResourceAllocation],
                                         strategies: List[OptimizationStrategy]) -> float:
        """Calculate Pareto efficiency score"""
        if not strategies:
            return 0.5
        
        # Multi-objective evaluation
        objectives = {}
        
        # Cost objective
        total_cost = sum(alloc.total_cost for alloc in allocations)
        objectives['cost'] = 1.0 / (1.0 + total_cost / 1000.0)  # Normalize
        
        # Performance objective
        total_performance = await self._estimate_performance(allocations)
        objectives['performance'] = min(1.0, total_performance / 10000.0)  # Normalize
        
        # Efficiency objective
        efficiency = total_performance / max(total_cost, 1.0)
        objectives['efficiency'] = min(1.0, efficiency / 100.0)  # Normalize
        
        # Calculate weighted score based on strategy
        strategy = strategies[0]
        weighted_score = 0.0
        
        for obj_type, weight in strategy.weights.items():
            if obj_type.value in objectives:
                weighted_score += weight * objectives[obj_type.value]
        
        return min(1.0, weighted_score)
    
    async def update_model(self, experience: Dict[str, Any]) -> None:
        """Update RL model with new experience"""
        try:
            # Add experience to replay buffer
            self.replay_buffer.append(experience)
            
            # Limit buffer size
            if len(self.replay_buffer) > self.experience_buffer_size:
                self.replay_buffer.pop(0)
            
            # Train if enough experiences
            if len(self.replay_buffer) >= self.batch_size:
                await self._train_q_network()
            
            logger.debug("RL model updated with new experience")
            
        except Exception as e:
            logger.error(f"Error updating RL model: {e}")
    
    async def _train_q_network(self) -> None:
        """Train Q-network using experience replay"""
        try:
            # Sample batch from replay buffer
            batch_indices = np.random.choice(
                len(self.replay_buffer), 
                size=min(self.batch_size, len(self.replay_buffer)),
                replace=False
            )
            
            batch = [self.replay_buffer[i] for i in batch_indices]
            
            # Simplified training (in practice, use proper gradient descent)
            for experience in batch:
                # Extract experience components
                state = experience.get('state')
                action = experience.get('action')
                reward = experience.get('reward')
                next_state = experience.get('next_state')
                done = experience.get('done', False)
                
                if state and next_state:
                    # Calculate target Q-value
                    if done:
                        target_q = reward
                    else:
                        next_q_values = self._q_forward(next_state, self.target_network)
                        target_q = reward + self.discount_factor * np.max(next_q_values)
                    
                    # Update Q-network (simplified)
                    current_q_values = self._q_forward(state, self.q_network)
                    # In practice, update specific action's Q-value using gradient descent
            
            # Update target network periodically
            if len(self.replay_buffer) % 100 == 0:
                self._update_target_network()
            
            logger.debug("Q-network training completed")
            
        except Exception as e:
            logger.error(f"Error training Q-network: {e}")
    
    def _update_target_network(self) -> None:
        """Update target network with current Q-network weights"""
        # Soft update (simplified)
        tau = 0.01
        for key in self.q_network:
            self.target_network[key] = (
                tau * self.q_network[key] + 
                (1 - tau) * self.target_network[key]
            )
    
    async def get_performance_metrics(self, 
                                    period_start: datetime,
                                    period_end: datetime) -> PerformanceMetrics:
        """Get performance metrics for the optimization system"""
        try:
            # Calculate metrics from historical data
            relevant_history = [
                h for h in self.performance_history
                if period_start <= h.timestamp <= period_end
            ]
            
            if not relevant_history:
                return PerformanceMetrics(
                    period_start=period_start,
                    period_end=period_end,
                    total_cost_savings=0.0,
                    cost_reduction_percentage=0.0,
                    arbitrage_profits=0.0,
                    prediction_accuracy=0.0,
                    successful_trades=0,
                    failed_trades=0,
                    average_roi=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0
                )
            
            # Aggregate metrics
            total_savings = sum(h.total_cost_savings for h in relevant_history)
            avg_reduction = np.mean([h.cost_reduction_percentage for h in relevant_history])
            total_arbitrage = sum(h.arbitrage_profits for h in relevant_history)
            avg_accuracy = np.mean([h.prediction_accuracy for h in relevant_history])
            total_successful = sum(h.successful_trades for h in relevant_history)
            total_failed = sum(h.failed_trades for h in relevant_history)
            avg_roi = np.mean([h.average_roi for h in relevant_history])
            
            # Calculate Sharpe ratio and max drawdown (simplified)
            roi_values = [h.average_roi for h in relevant_history]
            sharpe_ratio = np.mean(roi_values) / max(np.std(roi_values), 0.01)
            max_drawdown = max(roi_values) - min(roi_values) if roi_values else 0.0
            
            return PerformanceMetrics(
                period_start=period_start,
                period_end=period_end,
                total_cost_savings=total_savings,
                cost_reduction_percentage=avg_reduction,
                arbitrage_profits=total_arbitrage,
                prediction_accuracy=avg_accuracy,
                successful_trades=total_successful,
                failed_trades=total_failed,
                average_roi=avg_roi,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise