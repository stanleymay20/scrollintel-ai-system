"""
Main Optimization Engine for the Advanced Prompt Management System.
Coordinates genetic algorithms, reinforcement learning, and multi-objective optimization.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from ..models.optimization_models import (
    OptimizationJob, OptimizationResults, OptimizationCandidate,
    OptimizationAlgorithm, JobStatus, OptimizationType,
    PerformanceMetrics, TestCase, OptimizationConfig, OptimizationStatus
)
from .genetic_optimizer import GeneticOptimizer, GeneticConfig
# from .rl_optimizer import RLOptimizer, RLConfig  # Temporarily disabled
from ..core.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class ParetoFront:
    """Manages Pareto-optimal solutions for multi-objective optimization."""
    
    def __init__(self):
        self.solutions: List[Dict[str, Any]] = []
    
    def add_solution(self, solution: Dict[str, Any]) -> bool:
        """Add a solution to the Pareto front if it's non-dominated."""
        objectives = solution["objective_scores"]
        
        # Check if solution is dominated by any existing solution
        for existing in self.solutions:
            if self._dominates(existing["objective_scores"], objectives):
                return False  # Solution is dominated, don't add
        
        # Remove any existing solutions dominated by the new solution
        self.solutions = [
            existing for existing in self.solutions
            if not self._dominates(objectives, existing["objective_scores"])
        ]
        
        # Add the new solution
        self.solutions.append(solution)
        return True
    
    def _dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 dominates obj2 (Pareto dominance)."""
        better_in_at_least_one = False
        
        for key in obj1.keys():
            if key in obj2:
                if obj1[key] < obj2[key]:  # Assuming higher is better
                    return False
                elif obj1[key] > obj2[key]:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def get_solutions(self) -> List[Dict[str, Any]]:
        """Get all Pareto-optimal solutions."""
        return self.solutions.copy()
    
    def get_best_solution(self, weights: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """Get the best solution based on weighted sum."""
        if not self.solutions:
            return None
        
        if weights is None:
            weights = {"accuracy": 0.4, "relevance": 0.4, "efficiency": 0.2}
        
        best_solution = None
        best_score = float('-inf')
        
        for solution in self.solutions:
            score = sum(
                solution["objective_scores"].get(obj, 0) * weight
                for obj, weight in weights.items()
            )
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return best_solution


class PerformanceEvaluator:
    """Evaluates prompt performance using custom metrics."""
    
    def __init__(self, custom_metrics: Optional[List[Dict[str, Any]]] = None):
        self.custom_metrics = custom_metrics or []
        self.evaluation_cache = {}
    
    def evaluate_prompt(self, prompt: str, test_cases: List[TestCase]) -> PerformanceMetrics:
        """Evaluate a prompt against test cases."""
        # Check cache first
        cache_key = hash(prompt + str(test_cases))
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # Basic metrics calculation (simplified for demo)
        accuracy = self._calculate_accuracy(prompt, test_cases)
        relevance = self._calculate_relevance(prompt, test_cases)
        efficiency = self._calculate_efficiency(prompt, test_cases)
        
        # Custom metrics
        custom_scores = {}
        for metric in self.custom_metrics:
            try:
                score = self._evaluate_custom_metric(prompt, test_cases, metric)
                custom_scores[metric["name"]] = score
            except Exception as e:
                logger.error(f"Error evaluating custom metric {metric['name']}: {e}")
                custom_scores[metric["name"]] = 0.0
        
        metrics = PerformanceMetrics(accuracy, relevance, efficiency, custom_scores)
        
        # Cache result
        self.evaluation_cache[cache_key] = metrics
        
        return metrics
    
    def _calculate_accuracy(self, prompt: str, test_cases: List[TestCase]) -> float:
        """Calculate accuracy metric."""
        if not test_cases:
            return 0.5  # Default neutral score
        
        # Simplified accuracy calculation
        # In practice, this would involve actual AI model evaluation
        score = 0.0
        
        # Length-based heuristic
        if 50 <= len(prompt) <= 500:
            score += 0.3
        
        # Instruction clarity
        instruction_words = ["analyze", "explain", "describe", "create", "solve", "evaluate"]
        if any(word in prompt.lower() for word in instruction_words):
            score += 0.3
        
        # Context provision
        if "context" in prompt.lower() or "background" in prompt.lower():
            score += 0.2
        
        # Structure
        if "step by step" in prompt.lower() or ":" in prompt:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_relevance(self, prompt: str, test_cases: List[TestCase]) -> float:
        """Calculate relevance metric."""
        if not test_cases:
            return 0.5
        
        score = 0.0
        
        # Check if prompt addresses test case requirements
        for test_case in test_cases:
            input_data = test_case.input_data
            
            # Simple keyword matching
            keywords = str(input_data).lower().split()
            prompt_words = prompt.lower().split()
            
            overlap = len(set(keywords) & set(prompt_words))
            if overlap > 0:
                score += overlap / len(keywords)
        
        return min(1.0, score / len(test_cases) if test_cases else 0.5)
    
    def _calculate_efficiency(self, prompt: str, test_cases: List[TestCase]) -> float:
        """Calculate efficiency metric."""
        # Efficiency based on prompt length and clarity
        length_score = 1.0 - (len(prompt) / 1000)  # Penalize very long prompts
        length_score = max(0.0, length_score)
        
        # Clarity score (fewer redundant words)
        redundant_words = ["please", "kindly", "if possible", "thank you"]
        redundancy_penalty = sum(1 for word in redundant_words if word in prompt.lower())
        clarity_score = max(0.0, 1.0 - (redundancy_penalty * 0.1))
        
        return (length_score + clarity_score) / 2
    
    def _evaluate_custom_metric(self, prompt: str, test_cases: List[TestCase], 
                               metric: Dict[str, Any]) -> float:
        """Evaluate a custom metric."""
        # This would execute the custom evaluation function
        # For security, this should be sandboxed in production
        evaluation_function = metric.get("evaluation_function", "return 0.5")
        
        # Simple evaluation (in practice, use safe execution environment)
        try:
            # Create a safe context for evaluation
            context = {
                "prompt": prompt,
                "test_cases": test_cases,
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs
            }
            
            result = eval(evaluation_function, {"__builtins__": {}}, context)
            return float(result) if isinstance(result, (int, float)) else 0.0
        except Exception as e:
            logger.error(f"Error in custom metric evaluation: {e}")
            return 0.0


class OptimizationEngine:
    """Main optimization engine coordinating all optimization algorithms."""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.evaluator = PerformanceEvaluator()
        self.active_jobs: Dict[str, OptimizationJob] = {}
        self.job_futures: Dict[str, asyncio.Task] = {}
    
    async def optimize_prompt(self, prompt_id: str, config: OptimizationConfig) -> OptimizationJob:
        """Start prompt optimization job."""
        # Get the prompt template
        prompt_template = await self.prompt_manager.get_prompt(prompt_id)
        if not prompt_template:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        # Create optimization job
        job = OptimizationJob(
            name=f"Optimization of {prompt_template.name}",
            prompt_id=prompt_id,
            algorithm=config.algorithm.value,
            config=config.to_dict(),
            objectives=config.config.get("objectives", ["accuracy", "relevance", "efficiency"]),
            created_by="system"  # In practice, get from auth context
        )
        
        # Store job
        self.active_jobs[job.id] = job
        
        # Start optimization task
        task = asyncio.create_task(self._run_optimization(job, prompt_template.content))
        self.job_futures[job.id] = task
        
        return job
    
    async def _run_optimization(self, job: OptimizationJob, base_prompt: str) -> OptimizationResults:
        """Run the optimization process."""
        try:
            job.status = JobStatus.RUNNING.value
            job.started_at = datetime.utcnow()
            
            # Get test cases (in practice, these would be provided or generated)
            test_cases = self._generate_test_cases(base_prompt)
            
            # Set up evaluation function
            evaluation_function = lambda prompt, cases: self.evaluator.evaluate_prompt(prompt, cases)
            
            # Run optimization based on algorithm
            if job.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM.value:
                results = await self._run_genetic_optimization(job, base_prompt, test_cases, evaluation_function)
            elif job.algorithm == OptimizationAlgorithm.REINFORCEMENT_LEARNING.value:
                # RL optimization temporarily disabled - use genetic algorithm as fallback
                results = await self._run_genetic_optimization(job, base_prompt, test_cases, evaluation_function)
            else:
                raise ValueError(f"Unsupported algorithm: {job.algorithm}")
            
            # Handle multi-objective optimization
            if job.optimization_type == OptimizationType.MULTI_OBJECTIVE.value:
                results = await self._handle_multi_objective_results(job, results)
            
            # Create optimization results
            optimization_results = OptimizationResults(
                job_id=job.id,
                best_prompt_content=results["best_prompt"],
                best_fitness_score=results["best_fitness"],
                fitness_history=results.get("fitness_history", []),
                performance_metrics=results.get("objective_scores", {}),
                improvement_percentage=self._calculate_improvement_percentage(
                    base_prompt, results["best_prompt"], test_cases
                ),
                execution_time_seconds=results.get("execution_time", 0),
                total_evaluations=results.get("total_evaluations", 0),
                optimization_summary=self._generate_summary(results)
            )
            
            job.status = JobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            job.best_fitness = results["best_fitness"]
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Optimization job {job.id} failed: {e}")
            job.status = JobStatus.FAILED.value
            job.completed_at = datetime.utcnow()
            raise
        finally:
            # Clean up
            if job.id in self.job_futures:
                del self.job_futures[job.id]
    
    async def _run_genetic_optimization(self, job: OptimizationJob, base_prompt: str,
                                      test_cases: List[TestCase], evaluation_function: Callable) -> Dict[str, Any]:
        """Run genetic algorithm optimization."""
        config = GeneticConfig(
            population_size=job.population_size,
            max_generations=job.max_generations,
            convergence_threshold=job.convergence_threshold,
            **job.config.get("genetic_config", {})
        )
        
        optimizer = GeneticOptimizer(config, evaluation_function)
        
        # Progress callback
        def progress_callback(stats):
            job.progress = stats.get("progress", 0)
            job.current_generation = stats.get("generation", 0)
            job.best_fitness = stats.get("best_fitness")
        
        # Run optimization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor, optimizer.optimize, base_prompt, test_cases, progress_callback
            )
        
        return results
    
    async def _run_rl_optimization(self, job: OptimizationJob, base_prompt: str,
                                 test_cases: List[TestCase], evaluation_function: Callable) -> Dict[str, Any]:
        """Run reinforcement learning optimization."""
        config = RLConfig(
            max_episodes=job.config.get("max_episodes", 1000),
            **job.config.get("rl_config", {})
        )
        
        optimizer = RLOptimizer(config, evaluation_function)
        
        # Progress callback
        def progress_callback(stats):
            job.progress = stats.get("progress", 0)
            job.current_generation = stats.get("episode", 0)
            job.best_fitness = stats.get("best_reward")
        
        # Run optimization in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor, optimizer.optimize, base_prompt, test_cases, progress_callback
            )
        
        return results
    
    async def _handle_multi_objective_results(self, job: OptimizationJob, results: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-objective optimization results."""
        # Create Pareto front
        pareto_front = ParetoFront()
        
        # Add solutions to Pareto front (simplified - in practice, collect from optimization process)
        solution = {
            "prompt": results["best_prompt"],
            "objective_scores": results.get("objective_scores", {}),
            "fitness": results["best_fitness"]
        }
        pareto_front.add_solution(solution)
        
        # Update results with Pareto front information
        results["pareto_front"] = [sol for sol in pareto_front.get_solutions()]
        results["pareto_optimal_count"] = len(pareto_front.solutions)
        
        return results
    
    def _generate_test_cases(self, base_prompt: str) -> List[TestCase]:
        """Generate test cases for prompt evaluation."""
        # In practice, these would be provided by the user or generated based on prompt analysis
        test_cases = [
            TestCase(
                input_data={"query": "sample input"},
                expected_output="expected response",
                evaluation_criteria={"accuracy": 0.8, "relevance": 0.9}
            )
        ]
        return test_cases
    
    def _calculate_improvement_percentage(self, original_prompt: str, optimized_prompt: str,
                                        test_cases: List[TestCase]) -> float:
        """Calculate improvement percentage."""
        original_metrics = self.evaluator.evaluate_prompt(original_prompt, test_cases)
        optimized_metrics = self.evaluator.evaluate_prompt(optimized_prompt, test_cases)
        
        original_score = original_metrics.get_weighted_score()
        optimized_score = optimized_metrics.get_weighted_score()
        
        if original_score == 0:
            return 0.0
        
        return ((optimized_score - original_score) / original_score) * 100
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable optimization summary."""
        summary_parts = [
            f"Optimization completed successfully.",
            f"Best fitness score: {results['best_fitness']:.4f}",
            f"Execution time: {results.get('execution_time', 0):.2f} seconds",
            f"Total evaluations: {results.get('total_evaluations', 0)}"
        ]
        
        if "generations_completed" in results:
            summary_parts.append(f"Generations completed: {results['generations_completed']}")
        
        if "episodes_completed" in results:
            summary_parts.append(f"Episodes completed: {results['episodes_completed']}")
        
        return " ".join(summary_parts)
    
    async def get_optimization_status(self, job_id: str) -> OptimizationStatus:
        """Get the status of an optimization job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        return OptimizationStatus(
            job_id=job_id,
            status=JobStatus(job.status),
            progress=job.progress,
            current_generation=job.current_generation,
            best_fitness=job.best_fitness,
            message=f"Job is {job.status}"
        )
    
    async def cancel_optimization(self, job_id: str) -> bool:
        """Cancel an optimization job."""
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        
        if job_id in self.job_futures:
            task = self.job_futures[job_id]
            task.cancel()
            del self.job_futures[job_id]
        
        job.status = JobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        
        return True
    
    def evaluate_performance(self, prompt: str, test_cases: List[TestCase]) -> PerformanceMetrics:
        """Evaluate prompt performance with custom metrics."""
        return self.evaluator.evaluate_prompt(prompt, test_cases)
    
    def add_custom_metric(self, metric: Dict[str, Any]):
        """Add a custom performance metric."""
        self.evaluator.custom_metrics.append(metric)
    
    def get_active_jobs(self) -> List[OptimizationJob]:
        """Get all active optimization jobs."""
        return list(self.active_jobs.values())