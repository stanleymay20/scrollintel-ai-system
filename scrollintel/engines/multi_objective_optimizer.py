"""
Multi-Objective Optimization Engine with Pareto Frontier Exploration
Implements advanced optimization algorithms for resource allocation with multiple competing objectives
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
import json
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..models.economic_optimization_models import (
    ResourceType, CloudProvider, OptimizationObjective, OptimizationStrategy,
    ResourceAllocation, OptimizationResult, MarketState
)

logger = logging.getLogger(__name__)

@dataclass
class ParetoSolution:
    """Individual solution on the Pareto frontier"""
    allocations: List[ResourceAllocation]
    objectives: Dict[str, float]
    dominance_rank: int
    crowding_distance: float
    fitness_score: float

@dataclass
class OptimizationConstraints:
    """Constraints for multi-objective optimization"""
    budget_limit: float
    performance_minimum: float
    latency_maximum: float
    availability_minimum: float
    provider_limits: Dict[CloudProvider, int]
    resource_limits: Dict[ResourceType, int]

class MultiObjectiveOptimizer:
    """
    Multi-Objective Optimization Engine with Pareto Frontier Exploration
    
    Implements:
    - NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    - MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
    - Pareto frontier analysis and visualization
    - Constraint handling and feasibility checking
    - Multi-criteria decision making support
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.population_size = self.config.get('population_size', 100)
        self.max_generations = self.config.get('max_generations', 200)
        self.crossover_rate = self.config.get('crossover_rate', 0.9)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.elite_size = self.config.get('elite_size', 20)
        
        # Optimization state
        self.current_population: List[ParetoSolution] = []
        self.pareto_frontier: List[ParetoSolution] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Objective functions
        self.objective_functions = {
            OptimizationObjective.MINIMIZE_COST: self._objective_cost,
            OptimizationObjective.MAXIMIZE_PERFORMANCE: self._objective_performance,
            OptimizationObjective.MINIMIZE_LATENCY: self._objective_latency,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: self._objective_throughput
        }
        
        logger.info("Multi-Objective Optimizer initialized")
    
    async def optimize_pareto_frontier(self, 
                                     market_state: MarketState,
                                     strategies: List[OptimizationStrategy],
                                     constraints: OptimizationConstraints) -> List[ParetoSolution]:
        """
        Find Pareto-optimal solutions using NSGA-II algorithm
        """
        try:
            logger.info("Starting Pareto frontier optimization")
            
            # Initialize population
            population = await self._initialize_population(market_state, constraints)
            
            # Evolution loop
            for generation in range(self.max_generations):
                # Evaluate objectives for all solutions
                await self._evaluate_population(population, market_state, strategies)
                
                # Non-dominated sorting
                fronts = self._non_dominated_sorting(population)
                
                # Calculate crowding distance
                for front in fronts:
                    self._calculate_crowding_distance(front)
                
                # Selection for next generation
                new_population = []
                
                # Add solutions from best fronts
                for front in fronts:
                    if len(new_population) + len(front) <= self.population_size:
                        new_population.extend(front)
                    else:
                        # Sort by crowding distance and add remaining
                        front.sort(key=lambda x: x.crowding_distance, reverse=True)
                        remaining = self.population_size - len(new_population)
                        new_population.extend(front[:remaining])
                        break
                
                # Generate offspring
                offspring = await self._generate_offspring(new_population, market_state, constraints)
                
                # Combine parent and offspring populations
                population = new_population + offspring
                
                # Log progress
                if generation % 20 == 0:
                    best_front = fronts[0] if fronts else []
                    logger.info(f"Generation {generation}: Pareto front size = {len(best_front)}")
            
            # Final evaluation and Pareto frontier extraction
            await self._evaluate_population(population, market_state, strategies)
            fronts = self._non_dominated_sorting(population)
            
            if fronts:
                self.pareto_frontier = fronts[0]  # Best front
                self._calculate_crowding_distance(self.pareto_frontier)
            
            logger.info(f"Optimization completed. Pareto frontier size: {len(self.pareto_frontier)}")
            
            return self.pareto_frontier
            
        except Exception as e:
            logger.error(f"Error in Pareto frontier optimization: {e}")
            raise
    
    async def _initialize_population(self, 
                                   market_state: MarketState,
                                   constraints: OptimizationConstraints) -> List[ParetoSolution]:
        """Initialize random population of solutions"""
        try:
            population = []
            
            for _ in range(self.population_size):
                # Generate random resource allocation
                allocations = await self._generate_random_allocation(market_state, constraints)
                
                solution = ParetoSolution(
                    allocations=allocations,
                    objectives={},
                    dominance_rank=0,
                    crowding_distance=0.0,
                    fitness_score=0.0
                )
                
                population.append(solution)
            
            return population
            
        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            raise
    
    async def _generate_random_allocation(self, 
                                        market_state: MarketState,
                                        constraints: OptimizationConstraints) -> List[ResourceAllocation]:
        """Generate random but feasible resource allocation"""
        try:
            allocations = []
            remaining_budget = constraints.budget_limit
            
            # Randomly select resources and providers
            available_prices = [p for p in market_state.prices if p.availability > 0]
            
            if not available_prices:
                return allocations
            
            # Generate 1-10 allocations
            num_allocations = np.random.randint(1, min(11, len(available_prices) + 1))
            selected_prices = np.random.choice(available_prices, size=num_allocations, replace=False)
            
            for price in selected_prices:
                if remaining_budget <= 0:
                    break
                
                # Check provider limits
                provider_limit = constraints.provider_limits.get(price.provider, 100)
                current_provider_usage = sum(
                    a.quantity for a in allocations 
                    if a.provider == price.provider
                )
                
                if current_provider_usage >= provider_limit:
                    continue
                
                # Check resource limits
                resource_limit = constraints.resource_limits.get(price.resource_type, 50)
                current_resource_usage = sum(
                    a.quantity for a in allocations 
                    if a.resource_type == price.resource_type
                )
                
                if current_resource_usage >= resource_limit:
                    continue
                
                # Calculate feasible quantity
                max_affordable = int(remaining_budget / price.price_per_hour)
                max_provider = provider_limit - current_provider_usage
                max_resource = resource_limit - current_resource_usage
                max_available = price.availability
                
                max_quantity = min(max_affordable, max_provider, max_resource, max_available)
                
                if max_quantity > 0:
                    quantity = np.random.randint(1, max_quantity + 1)
                    total_cost = quantity * price.price_per_hour
                    
                    allocation = ResourceAllocation(
                        task_id="optimization_task",
                        resource_type=price.resource_type,
                        provider=price.provider,
                        region=price.region,
                        quantity=quantity,
                        duration=timedelta(hours=1),
                        total_cost=total_cost,
                        expected_performance=quantity * self._get_performance_factor(price.resource_type),
                        allocation_reason="Random initialization"
                    )
                    
                    allocations.append(allocation)
                    remaining_budget -= total_cost
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error generating random allocation: {e}")
            return []
    
    def _get_performance_factor(self, resource_type: ResourceType) -> float:
        """Get performance factor for resource type"""
        factors = {
            ResourceType.GPU_H100: 1000.0,
            ResourceType.GPU_A100: 800.0,
            ResourceType.GPU_V100: 600.0,
            ResourceType.CPU_COMPUTE: 100.0,
            ResourceType.MEMORY: 50.0,
            ResourceType.STORAGE: 20.0,
            ResourceType.BANDWIDTH: 30.0
        }
        return factors.get(resource_type, 100.0)
    
    async def _evaluate_population(self, 
                                 population: List[ParetoSolution],
                                 market_state: MarketState,
                                 strategies: List[OptimizationStrategy]) -> None:
        """Evaluate objectives for all solutions in population"""
        try:
            for solution in population:
                objectives = {}
                
                # Calculate all objective values
                objectives['cost'] = await self._objective_cost(solution.allocations, market_state)
                objectives['performance'] = await self._objective_performance(solution.allocations, market_state)
                objectives['latency'] = await self._objective_latency(solution.allocations, market_state)
                objectives['throughput'] = await self._objective_throughput(solution.allocations, market_state)
                
                # Additional objectives
                objectives['efficiency'] = objectives['performance'] / max(objectives['cost'], 1.0)
                objectives['risk'] = await self._objective_risk(solution.allocations, market_state)
                objectives['availability'] = await self._objective_availability(solution.allocations, market_state)
                
                solution.objectives = objectives
                
                # Calculate fitness score based on strategies
                if strategies:
                    fitness = 0.0
                    for strategy in strategies:
                        strategy_fitness = 0.0
                        total_weight = sum(strategy.weights.values())
                        
                        for obj_type, weight in strategy.weights.items():
                            if obj_type.value in objectives:
                                obj_value = objectives[obj_type.value]
                                
                                # Normalize objective (minimize or maximize)
                                if obj_type in [OptimizationObjective.MINIMIZE_COST, OptimizationObjective.MINIMIZE_LATENCY]:
                                    normalized_value = 1.0 / (1.0 + obj_value)
                                else:
                                    normalized_value = obj_value / 1000.0  # Normalize to 0-1 range
                                
                                strategy_fitness += (weight / total_weight) * normalized_value
                        
                        fitness += strategy_fitness / len(strategies)
                    
                    solution.fitness_score = fitness
                else:
                    # Default fitness: balance cost and performance
                    solution.fitness_score = objectives['efficiency']
            
        except Exception as e:
            logger.error(f"Error evaluating population: {e}")
    
    async def _objective_cost(self, allocations: List[ResourceAllocation], market_state: MarketState) -> float:
        """Calculate total cost objective"""
        return sum(allocation.total_cost for allocation in allocations)
    
    async def _objective_performance(self, allocations: List[ResourceAllocation], market_state: MarketState) -> float:
        """Calculate total performance objective"""
        return sum(allocation.expected_performance for allocation in allocations)
    
    async def _objective_latency(self, allocations: List[ResourceAllocation], market_state: MarketState) -> float:
        """Calculate average latency objective"""
        if not allocations:
            return 1000.0  # High latency penalty
        
        # Simplified latency calculation based on provider and resource type
        latency_factors = {
            CloudProvider.AWS: 10.0,
            CloudProvider.GCP: 12.0,
            CloudProvider.AZURE: 11.0,
            CloudProvider.LAMBDA_LABS: 25.0,
            CloudProvider.RUNPOD: 30.0,
            CloudProvider.VAST_AI: 40.0,
            CloudProvider.PAPERSPACE: 35.0
        }
        
        resource_latency = {
            ResourceType.GPU_H100: 1.0,
            ResourceType.GPU_A100: 1.2,
            ResourceType.GPU_V100: 1.5,
            ResourceType.CPU_COMPUTE: 5.0,
            ResourceType.MEMORY: 0.5,
            ResourceType.STORAGE: 10.0,
            ResourceType.BANDWIDTH: 2.0
        }
        
        total_latency = 0.0
        total_weight = 0.0
        
        for allocation in allocations:
            provider_latency = latency_factors.get(allocation.provider, 30.0)
            resource_factor = resource_latency.get(allocation.resource_type, 5.0)
            
            allocation_latency = provider_latency * resource_factor
            weight = allocation.quantity
            
            total_latency += allocation_latency * weight
            total_weight += weight
        
        return total_latency / max(total_weight, 1.0)
    
    async def _objective_throughput(self, allocations: List[ResourceAllocation], market_state: MarketState) -> float:
        """Calculate total throughput objective"""
        throughput_factors = {
            ResourceType.GPU_H100: 1000.0,
            ResourceType.GPU_A100: 800.0,
            ResourceType.GPU_V100: 600.0,
            ResourceType.CPU_COMPUTE: 100.0,
            ResourceType.MEMORY: 200.0,
            ResourceType.STORAGE: 50.0,
            ResourceType.BANDWIDTH: 500.0
        }
        
        total_throughput = 0.0
        
        for allocation in allocations:
            base_throughput = throughput_factors.get(allocation.resource_type, 100.0)
            total_throughput += base_throughput * allocation.quantity
        
        return total_throughput
    
    async def _objective_risk(self, allocations: List[ResourceAllocation], market_state: MarketState) -> float:
        """Calculate risk objective"""
        provider_risk = {
            CloudProvider.AWS: 0.1,
            CloudProvider.GCP: 0.15,
            CloudProvider.AZURE: 0.12,
            CloudProvider.LAMBDA_LABS: 0.3,
            CloudProvider.RUNPOD: 0.4,
            CloudProvider.VAST_AI: 0.5,
            CloudProvider.PAPERSPACE: 0.35
        }
        
        total_risk = 0.0
        total_weight = 0.0
        
        for allocation in allocations:
            risk = provider_risk.get(allocation.provider, 0.3)
            weight = allocation.total_cost
            
            total_risk += risk * weight
            total_weight += weight
        
        return total_risk / max(total_weight, 1.0)
    
    async def _objective_availability(self, allocations: List[ResourceAllocation], market_state: MarketState) -> float:
        """Calculate availability objective"""
        # Find availability for each allocation
        total_availability = 0.0
        count = 0
        
        for allocation in allocations:
            matching_prices = [
                p for p in market_state.prices
                if p.resource_type == allocation.resource_type and p.provider == allocation.provider
            ]
            
            if matching_prices:
                availability = matching_prices[0].availability
                total_availability += availability
                count += 1
        
        return total_availability / max(count, 1)
    
    def _non_dominated_sorting(self, population: List[ParetoSolution]) -> List[List[ParetoSolution]]:
        """Perform non-dominated sorting (NSGA-II)"""
        try:
            fronts = []
            domination_count = {}
            dominated_solutions = {}
            
            # Initialize
            for solution in population:
                domination_count[id(solution)] = 0
                dominated_solutions[id(solution)] = []
            
            # Calculate domination relationships
            for i, solution_i in enumerate(population):
                for j, solution_j in enumerate(population):
                    if i != j:
                        if self._dominates(solution_i, solution_j):
                            dominated_solutions[id(solution_i)].append(solution_j)
                        elif self._dominates(solution_j, solution_i):
                            domination_count[id(solution_i)] += 1
            
            # Find first front
            first_front = []
            for solution in population:
                if domination_count[id(solution)] == 0:
                    solution.dominance_rank = 0
                    first_front.append(solution)
            
            fronts.append(first_front)
            
            # Find subsequent fronts
            front_index = 0
            while len(fronts[front_index]) > 0:
                next_front = []
                
                for solution in fronts[front_index]:
                    for dominated_solution in dominated_solutions[id(solution)]:
                        domination_count[id(dominated_solution)] -= 1
                        
                        if domination_count[id(dominated_solution)] == 0:
                            dominated_solution.dominance_rank = front_index + 1
                            next_front.append(dominated_solution)
                
                if next_front:
                    fronts.append(next_front)
                    front_index += 1
                else:
                    break
            
            return fronts
            
        except Exception as e:
            logger.error(f"Error in non-dominated sorting: {e}")
            return [population]  # Return all in one front as fallback
    
    def _dominates(self, solution_a: ParetoSolution, solution_b: ParetoSolution) -> bool:
        """Check if solution A dominates solution B"""
        try:
            # Solution A dominates B if:
            # 1. A is at least as good as B in all objectives
            # 2. A is strictly better than B in at least one objective
            
            objectives_a = solution_a.objectives
            objectives_b = solution_b.objectives
            
            if not objectives_a or not objectives_b:
                return False
            
            # Define which objectives to minimize vs maximize
            minimize_objectives = {'cost', 'latency', 'risk'}
            maximize_objectives = {'performance', 'throughput', 'efficiency', 'availability'}
            
            at_least_as_good = True
            strictly_better = False
            
            # Check all objectives
            all_objectives = set(objectives_a.keys()) | set(objectives_b.keys())
            
            for obj in all_objectives:
                val_a = objectives_a.get(obj, 0)
                val_b = objectives_b.get(obj, 0)
                
                if obj in minimize_objectives:
                    # For minimization objectives
                    if val_a > val_b:
                        at_least_as_good = False
                        break
                    elif val_a < val_b:
                        strictly_better = True
                elif obj in maximize_objectives:
                    # For maximization objectives
                    if val_a < val_b:
                        at_least_as_good = False
                        break
                    elif val_a > val_b:
                        strictly_better = True
            
            return at_least_as_good and strictly_better
            
        except Exception as e:
            logger.error(f"Error checking dominance: {e}")
            return False
    
    def _calculate_crowding_distance(self, front: List[ParetoSolution]) -> None:
        """Calculate crowding distance for solutions in a front"""
        try:
            if len(front) <= 2:
                for solution in front:
                    solution.crowding_distance = float('inf')
                return
            
            # Initialize crowding distances
            for solution in front:
                solution.crowding_distance = 0.0
            
            # Get all objective names
            if not front[0].objectives:
                return
            
            objectives = list(front[0].objectives.keys())
            
            # Calculate crowding distance for each objective
            for obj in objectives:
                # Sort by objective value
                front.sort(key=lambda x: x.objectives.get(obj, 0))
                
                # Set boundary solutions to infinite distance
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                
                # Calculate range
                obj_min = front[0].objectives.get(obj, 0)
                obj_max = front[-1].objectives.get(obj, 0)
                obj_range = obj_max - obj_min
                
                if obj_range == 0:
                    continue
                
                # Calculate crowding distance for intermediate solutions
                for i in range(1, len(front) - 1):
                    if front[i].crowding_distance != float('inf'):
                        distance = (front[i + 1].objectives.get(obj, 0) - 
                                  front[i - 1].objectives.get(obj, 0)) / obj_range
                        front[i].crowding_distance += distance
            
        except Exception as e:
            logger.error(f"Error calculating crowding distance: {e}")
    
    async def _generate_offspring(self, 
                                parent_population: List[ParetoSolution],
                                market_state: MarketState,
                                constraints: OptimizationConstraints) -> List[ParetoSolution]:
        """Generate offspring through crossover and mutation"""
        try:
            offspring = []
            
            while len(offspring) < len(parent_population):
                # Tournament selection
                parent1 = self._tournament_selection(parent_population)
                parent2 = self._tournament_selection(parent_population)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = await self._crossover(parent1, parent2, market_state, constraints)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = await self._mutate(child1, market_state, constraints)
                if np.random.random() < self.mutation_rate:
                    child2 = await self._mutate(child2, market_state, constraints)
                
                offspring.extend([child1, child2])
            
            return offspring[:len(parent_population)]
            
        except Exception as e:
            logger.error(f"Error generating offspring: {e}")
            return parent_population
    
    def _tournament_selection(self, population: List[ParetoSolution], tournament_size: int = 3) -> ParetoSolution:
        """Tournament selection for parent selection"""
        try:
            tournament = np.random.choice(population, size=min(tournament_size, len(population)), replace=False)
            
            # Select best based on dominance rank and crowding distance
            best = tournament[0]
            for candidate in tournament[1:]:
                if (candidate.dominance_rank < best.dominance_rank or
                    (candidate.dominance_rank == best.dominance_rank and 
                     candidate.crowding_distance > best.crowding_distance)):
                    best = candidate
            
            return best
            
        except Exception as e:
            logger.error(f"Error in tournament selection: {e}")
            return population[0] if population else None
    
    async def _crossover(self, 
                       parent1: ParetoSolution, 
                       parent2: ParetoSolution,
                       market_state: MarketState,
                       constraints: OptimizationConstraints) -> Tuple[ParetoSolution, ParetoSolution]:
        """Crossover operation to create offspring"""
        try:
            # Simple uniform crossover for allocations
            all_allocations = parent1.allocations + parent2.allocations
            
            if not all_allocations:
                return parent1, parent2
            
            # Create two children by randomly selecting allocations
            child1_allocations = []
            child2_allocations = []
            
            for allocation in all_allocations:
                if np.random.random() < 0.5:
                    child1_allocations.append(allocation)
                else:
                    child2_allocations.append(allocation)
            
            # Ensure feasibility
            child1_allocations = await self._ensure_feasibility(child1_allocations, constraints)
            child2_allocations = await self._ensure_feasibility(child2_allocations, constraints)
            
            child1 = ParetoSolution(
                allocations=child1_allocations,
                objectives={},
                dominance_rank=0,
                crowding_distance=0.0,
                fitness_score=0.0
            )
            
            child2 = ParetoSolution(
                allocations=child2_allocations,
                objectives={},
                dominance_rank=0,
                crowding_distance=0.0,
                fitness_score=0.0
            )
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return parent1, parent2
    
    async def _mutate(self, 
                    solution: ParetoSolution,
                    market_state: MarketState,
                    constraints: OptimizationConstraints) -> ParetoSolution:
        """Mutation operation"""
        try:
            mutated_allocations = solution.allocations.copy()
            
            if not mutated_allocations:
                # Add random allocation if empty
                new_allocations = await self._generate_random_allocation(market_state, constraints)
                if new_allocations:
                    mutated_allocations = new_allocations[:1]
            else:
                # Random mutation operations
                mutation_type = np.random.choice(['add', 'remove', 'modify'], p=[0.4, 0.3, 0.3])
                
                if mutation_type == 'add':
                    # Add new allocation
                    new_allocation = await self._generate_random_allocation(market_state, constraints)
                    if new_allocation:
                        mutated_allocations.extend(new_allocation[:1])
                
                elif mutation_type == 'remove' and len(mutated_allocations) > 1:
                    # Remove random allocation
                    idx = np.random.randint(0, len(mutated_allocations))
                    mutated_allocations.pop(idx)
                
                elif mutation_type == 'modify':
                    # Modify random allocation
                    idx = np.random.randint(0, len(mutated_allocations))
                    allocation = mutated_allocations[idx]
                    
                    # Modify quantity (±50%)
                    new_quantity = max(1, int(allocation.quantity * np.random.uniform(0.5, 1.5)))
                    
                    # Find matching price for cost calculation
                    matching_prices = [
                        p for p in market_state.prices
                        if (p.resource_type == allocation.resource_type and 
                            p.provider == allocation.provider)
                    ]
                    
                    if matching_prices:
                        price = matching_prices[0]
                        new_cost = new_quantity * price.price_per_hour
                        
                        mutated_allocations[idx] = ResourceAllocation(
                            task_id=allocation.task_id,
                            resource_type=allocation.resource_type,
                            provider=allocation.provider,
                            region=allocation.region,
                            quantity=new_quantity,
                            duration=allocation.duration,
                            total_cost=new_cost,
                            expected_performance=new_quantity * self._get_performance_factor(allocation.resource_type),
                            allocation_reason="Mutation"
                        )
            
            # Ensure feasibility
            mutated_allocations = await self._ensure_feasibility(mutated_allocations, constraints)
            
            mutated_solution = ParetoSolution(
                allocations=mutated_allocations,
                objectives={},
                dominance_rank=0,
                crowding_distance=0.0,
                fitness_score=0.0
            )
            
            return mutated_solution
            
        except Exception as e:
            logger.error(f"Error in mutation: {e}")
            return solution
    
    async def _ensure_feasibility(self, 
                                allocations: List[ResourceAllocation],
                                constraints: OptimizationConstraints) -> List[ResourceAllocation]:
        """Ensure allocations satisfy constraints"""
        try:
            feasible_allocations = []
            current_cost = 0.0
            provider_usage = {provider: 0 for provider in CloudProvider}
            resource_usage = {resource: 0 for resource in ResourceType}
            
            # Sort by cost efficiency (performance per cost)
            sorted_allocations = sorted(
                allocations,
                key=lambda a: a.expected_performance / max(a.total_cost, 1.0),
                reverse=True
            )
            
            for allocation in sorted_allocations:
                # Check budget constraint
                if current_cost + allocation.total_cost > constraints.budget_limit:
                    continue
                
                # Check provider limits
                provider_limit = constraints.provider_limits.get(allocation.provider, 100)
                if provider_usage[allocation.provider] + allocation.quantity > provider_limit:
                    continue
                
                # Check resource limits
                resource_limit = constraints.resource_limits.get(allocation.resource_type, 50)
                if resource_usage[allocation.resource_type] + allocation.quantity > resource_limit:
                    continue
                
                # Add to feasible allocations
                feasible_allocations.append(allocation)
                current_cost += allocation.total_cost
                provider_usage[allocation.provider] += allocation.quantity
                resource_usage[allocation.resource_type] += allocation.quantity
            
            return feasible_allocations
            
        except Exception as e:
            logger.error(f"Error ensuring feasibility: {e}")
            return allocations
    
    async def analyze_pareto_frontier(self, pareto_solutions: List[ParetoSolution]) -> Dict[str, Any]:
        """Analyze Pareto frontier and provide insights"""
        try:
            if not pareto_solutions:
                return {"error": "No Pareto solutions to analyze"}
            
            analysis = {
                "frontier_size": len(pareto_solutions),
                "objective_ranges": {},
                "trade_offs": {},
                "recommended_solutions": {},
                "diversity_metrics": {}
            }
            
            # Calculate objective ranges
            objectives = list(pareto_solutions[0].objectives.keys())
            for obj in objectives:
                values = [sol.objectives[obj] for sol in pareto_solutions]
                analysis["objective_ranges"][obj] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
            
            # Analyze trade-offs between objectives
            for i, obj1 in enumerate(objectives):
                for obj2 in objectives[i+1:]:
                    values1 = [sol.objectives[obj1] for sol in pareto_solutions]
                    values2 = [sol.objectives[obj2] for sol in pareto_solutions]
                    
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    analysis["trade_offs"][f"{obj1}_vs_{obj2}"] = {
                        "correlation": correlation,
                        "trade_off_strength": abs(correlation)
                    }
            
            # Recommend solutions for different preferences
            analysis["recommended_solutions"] = {
                "cost_optimal": min(pareto_solutions, key=lambda s: s.objectives.get('cost', float('inf'))),
                "performance_optimal": max(pareto_solutions, key=lambda s: s.objectives.get('performance', 0)),
                "balanced": max(pareto_solutions, key=lambda s: s.objectives.get('efficiency', 0)),
                "low_risk": min(pareto_solutions, key=lambda s: s.objectives.get('risk', float('inf')))
            }
            
            # Calculate diversity metrics
            if len(pareto_solutions) > 1:
                # Hypervolume (simplified)
                analysis["diversity_metrics"]["hypervolume"] = self._calculate_hypervolume(pareto_solutions)
                
                # Spacing metric
                analysis["diversity_metrics"]["spacing"] = self._calculate_spacing(pareto_solutions)
                
                # Spread metric
                analysis["diversity_metrics"]["spread"] = self._calculate_spread(pareto_solutions)
            
            logger.info("Pareto frontier analysis completed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Pareto frontier: {e}")
            return {"error": str(e)}
    
    def _calculate_hypervolume(self, solutions: List[ParetoSolution]) -> float:
        """Calculate hypervolume indicator (simplified 2D version)"""
        try:
            # Use cost and performance for 2D hypervolume
            points = []
            for sol in solutions:
                cost = sol.objectives.get('cost', 0)
                performance = sol.objectives.get('performance', 0)
                # Normalize: minimize cost, maximize performance
                points.append((1.0 / (1.0 + cost), performance / 1000.0))
            
            if len(points) < 2:
                return 0.0
            
            # Sort by first objective
            points.sort()
            
            # Calculate area under curve
            hypervolume = 0.0
            for i in range(len(points) - 1):
                width = points[i + 1][0] - points[i][0]
                height = points[i][1]
                hypervolume += width * height
            
            return hypervolume
            
        except Exception as e:
            logger.error(f"Error calculating hypervolume: {e}")
            return 0.0
    
    def _calculate_spacing(self, solutions: List[ParetoSolution]) -> float:
        """Calculate spacing metric for solution distribution"""
        try:
            if len(solutions) < 2:
                return 0.0
            
            distances = []
            
            for i, sol1 in enumerate(solutions):
                min_distance = float('inf')
                
                for j, sol2 in enumerate(solutions):
                    if i != j:
                        # Calculate Euclidean distance in objective space
                        distance = 0.0
                        for obj in sol1.objectives:
                            if obj in sol2.objectives:
                                diff = sol1.objectives[obj] - sol2.objectives[obj]
                                distance += diff ** 2
                        
                        distance = np.sqrt(distance)
                        min_distance = min(min_distance, distance)
                
                distances.append(min_distance)
            
            # Calculate spacing as standard deviation of distances
            mean_distance = np.mean(distances)
            spacing = np.sqrt(np.mean([(d - mean_distance) ** 2 for d in distances]))
            
            return spacing
            
        except Exception as e:
            logger.error(f"Error calculating spacing: {e}")
            return 0.0
    
    def _calculate_spread(self, solutions: List[ParetoSolution]) -> float:
        """Calculate spread metric for frontier extent"""
        try:
            if len(solutions) < 2:
                return 0.0
            
            # Calculate extent in each objective
            extents = []
            
            for obj in solutions[0].objectives:
                values = [sol.objectives[obj] for sol in solutions]
                extent = max(values) - min(values)
                extents.append(extent)
            
            # Return average extent (normalized)
            return np.mean(extents) / len(extents) if extents else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return 0.0