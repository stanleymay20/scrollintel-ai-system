"""
Component Orchestrator for ScrollIntel Modular Architecture

This module provides orchestration capabilities for managing modular components,
handling their lifecycle, dependencies, and interactions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid

from .modular_components import (
    ComponentInterface, ComponentRegistry, ComponentInstance, 
    ComponentMetadata, ComponentType, ComponentStatus, component_registry
)
from .enhanced_specialized_agent import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

class OrchestrationStrategy(Enum):
    """Strategies for component orchestration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    CONDITIONAL = "conditional"

@dataclass
class ComponentTask:
    """Task to be executed by a component"""
    task_id: str
    component_id: str
    operation: str
    input_data: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class TaskResult:
    """Result of component task execution"""
    task_id: str
    component_id: str
    status: str  # success, error, timeout
    output_data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestrationPlan:
    """Plan for orchestrating multiple components"""
    plan_id: str
    name: str
    strategy: OrchestrationStrategy
    tasks: List[ComponentTask]
    global_timeout: int = 1800
    error_handling: str = "fail_fast"  # fail_fast, continue, rollback
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComponentOrchestrator:
    """
    Orchestrates interactions between modular components
    """
    
    def __init__(self, registry: ComponentRegistry = None):
        self.registry = registry or component_registry
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._orchestration_metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0
        }
    
    async def execute_plan(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute an orchestration plan"""
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        execution_context = {
            "execution_id": execution_id,
            "plan": plan,
            "start_time": start_time,
            "status": "running",
            "task_results": {},
            "errors": []
        }
        
        self._active_executions[execution_id] = execution_context
        
        try:
            logger.info(f"Starting orchestration plan '{plan.name}' (ID: {execution_id})")
            
            # Execute based on strategy
            if plan.strategy == OrchestrationStrategy.SEQUENTIAL:
                results = await self._execute_sequential(plan, execution_context)
            elif plan.strategy == OrchestrationStrategy.PARALLEL:
                results = await self._execute_parallel(plan, execution_context)
            elif plan.strategy == OrchestrationStrategy.PIPELINE:
                results = await self._execute_pipeline(plan, execution_context)
            elif plan.strategy == OrchestrationStrategy.CONDITIONAL:
                results = await self._execute_conditional(plan, execution_context)
            else:
                raise ValueError(f"Unknown orchestration strategy: {plan.strategy}")
            
            # Update execution context
            execution_context["status"] = "completed"
            execution_context["task_results"] = results
            execution_context["end_time"] = datetime.now()
            execution_context["duration"] = (execution_context["end_time"] - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(execution_context, True)
            
            logger.info(f"Orchestration plan '{plan.name}' completed successfully")
            
            return {
                "execution_id": execution_id,
                "status": "success",
                "results": results,
                "duration": execution_context["duration"],
                "metadata": {
                    "plan_name": plan.name,
                    "strategy": plan.strategy.value,
                    "tasks_executed": len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"Orchestration plan '{plan.name}' failed: {e}")
            
            execution_context["status"] = "failed"
            execution_context["error"] = str(e)
            execution_context["end_time"] = datetime.now()
            execution_context["duration"] = (execution_context["end_time"] - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(execution_context, False)
            
            return {
                "execution_id": execution_id,
                "status": "error",
                "error": str(e),
                "duration": execution_context["duration"],
                "metadata": {
                    "plan_name": plan.name,
                    "strategy": plan.strategy.value
                }
            }
        
        finally:
            # Move to history and cleanup
            self._execution_history.append(execution_context)
            self._active_executions.pop(execution_id, None)
    
    async def _execute_sequential(self, plan: OrchestrationPlan, 
                                context: Dict[str, Any]) -> Dict[str, TaskResult]:
        """Execute tasks sequentially"""
        results = {}
        
        # Build execution order based on dependencies
        execution_order = self._build_execution_order(plan.tasks)
        
        for task_id in execution_order:
            task = next(t for t in plan.tasks if t.task_id == task_id)
            
            # Check if dependencies are satisfied
            if not self._check_dependencies(task, results):
                error_msg = f"Dependencies not satisfied for task {task_id}"
                results[task_id] = TaskResult(
                    task_id=task_id,
                    component_id=task.component_id,
                    status="error",
                    error=error_msg
                )
                
                if plan.error_handling == "fail_fast":
                    break
                continue
            
            # Execute task
            result = await self._execute_task(task, results, context)
            results[task_id] = result
            
            # Handle task failure
            if result.status == "error":
                context["errors"].append(f"Task {task_id} failed: {result.error}")
                
                if plan.error_handling == "fail_fast":
                    break
                elif plan.error_handling == "rollback":
                    await self._rollback_tasks(results)
                    break
        
        return results
    
    async def _execute_parallel(self, plan: OrchestrationPlan, 
                              context: Dict[str, Any]) -> Dict[str, TaskResult]:
        """Execute tasks in parallel where possible"""
        results = {}
        remaining_tasks = plan.tasks.copy()
        
        while remaining_tasks:
            # Find tasks that can be executed (dependencies satisfied)
            ready_tasks = []
            for task in remaining_tasks:
                if self._check_dependencies(task, results):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # No tasks ready, check if we're stuck
                if remaining_tasks:
                    error_msg = f"Circular dependencies or unsatisfied dependencies in tasks: {[t.task_id for t in remaining_tasks]}"
                    context["errors"].append(error_msg)
                    break
                else:
                    break
            
            # Execute ready tasks in parallel
            tasks_to_execute = []
            for task in ready_tasks:
                tasks_to_execute.append(self._execute_task(task, results, context))
                remaining_tasks.remove(task)
            
            # Wait for all parallel tasks to complete
            parallel_results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(parallel_results):
                task = ready_tasks[i]
                
                if isinstance(result, Exception):
                    task_result = TaskResult(
                        task_id=task.task_id,
                        component_id=task.component_id,
                        status="error",
                        error=str(result)
                    )
                else:
                    task_result = result
                
                results[task.task_id] = task_result
                
                # Handle failures
                if task_result.status == "error":
                    context["errors"].append(f"Task {task.task_id} failed: {task_result.error}")
                    
                    if plan.error_handling == "fail_fast":
                        return results
        
        return results
    
    async def _execute_pipeline(self, plan: OrchestrationPlan, 
                              context: Dict[str, Any]) -> Dict[str, TaskResult]:
        """Execute tasks as a pipeline with data flowing between them"""
        results = {}
        pipeline_data = None
        
        # Execute tasks in order, passing output to next task
        for task in plan.tasks:
            # Use output from previous task as input
            if pipeline_data is not None:
                task.input_data = pipeline_data
            
            result = await self._execute_task(task, results, context)
            results[task.task_id] = result
            
            if result.status == "error":
                context["errors"].append(f"Pipeline task {task.task_id} failed: {result.error}")
                if plan.error_handling == "fail_fast":
                    break
            else:
                # Pass output to next task
                pipeline_data = result.output_data
        
        return results
    
    async def _execute_conditional(self, plan: OrchestrationPlan, 
                                 context: Dict[str, Any]) -> Dict[str, TaskResult]:
        """Execute tasks with conditional logic"""
        results = {}
        
        for task in plan.tasks:
            # Check if task should be executed based on conditions
            should_execute = self._evaluate_task_condition(task, results, context)
            
            if should_execute:
                result = await self._execute_task(task, results, context)
                results[task.task_id] = result
                
                if result.status == "error":
                    context["errors"].append(f"Conditional task {task.task_id} failed: {result.error}")
                    if plan.error_handling == "fail_fast":
                        break
            else:
                # Task skipped due to condition
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    component_id=task.component_id,
                    status="skipped",
                    metadata={"reason": "condition_not_met"}
                )
        
        return results
    
    async def _execute_task(self, task: ComponentTask, previous_results: Dict[str, TaskResult], 
                          context: Dict[str, Any]) -> TaskResult:
        """Execute a single component task"""
        start_time = datetime.now()
        
        try:
            # Get component instance
            component_instance = self.registry.get_component(task.component_id)
            if not component_instance:
                return TaskResult(
                    task_id=task.task_id,
                    component_id=task.component_id,
                    status="error",
                    error=f"Component {task.component_id} not found"
                )
            
            if component_instance.status != ComponentStatus.READY:
                return TaskResult(
                    task_id=task.task_id,
                    component_id=task.component_id,
                    status="error",
                    error=f"Component {task.component_id} not ready (status: {component_instance.status.value})"
                )
            
            # Prepare input data (may include outputs from previous tasks)
            input_data = self._prepare_task_input(task, previous_results)
            
            # Execute task with timeout
            try:
                result = await asyncio.wait_for(
                    self._call_component_operation(
                        component_instance.component, 
                        task.operation, 
                        input_data, 
                        task.parameters
                    ),
                    timeout=task.timeout
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return TaskResult(
                    task_id=task.task_id,
                    component_id=task.component_id,
                    status="success",
                    output_data=result,
                    processing_time=processing_time,
                    metadata={"operation": task.operation}
                )
                
            except asyncio.TimeoutError:
                return TaskResult(
                    task_id=task.task_id,
                    component_id=task.component_id,
                    status="timeout",
                    error=f"Task timed out after {task.timeout} seconds",
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} execution failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                component_id=task.component_id,
                status="error",
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _call_component_operation(self, component: ComponentInterface, 
                                      operation: str, input_data: Any, 
                                      parameters: Dict[str, Any]) -> Any:
        """Call a specific operation on a component"""
        # Check if component has the requested operation
        if hasattr(component, operation):
            method = getattr(component, operation)
            
            # Call method with appropriate parameters
            if asyncio.iscoroutinefunction(method):
                if parameters:
                    return await method(input_data, **parameters)
                else:
                    return await method(input_data)
            else:
                if parameters:
                    return method(input_data, **parameters)
                else:
                    return method(input_data)
        else:
            raise AttributeError(f"Component {component.component_id} does not have operation '{operation}'")
    
    def _prepare_task_input(self, task: ComponentTask, 
                          previous_results: Dict[str, TaskResult]) -> Any:
        """Prepare input data for a task, potentially combining with previous results"""
        input_data = task.input_data
        
        # If task has dependencies, combine their outputs with input
        if task.dependencies:
            dependency_outputs = {}
            for dep_id in task.dependencies:
                if dep_id in previous_results and previous_results[dep_id].status == "success":
                    dependency_outputs[dep_id] = previous_results[dep_id].output_data
            
            # Combine with original input
            if dependency_outputs:
                if isinstance(input_data, dict):
                    input_data = {**input_data, "dependency_outputs": dependency_outputs}
                else:
                    input_data = {
                        "original_input": input_data,
                        "dependency_outputs": dependency_outputs
                    }
        
        return input_data
    
    def _check_dependencies(self, task: ComponentTask, 
                          results: Dict[str, TaskResult]) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in results or results[dep_id].status != "success":
                return False
        return True
    
    def _build_execution_order(self, tasks: List[ComponentTask]) -> List[str]:
        """Build execution order based on task dependencies"""
        # Topological sort
        in_degree = {task.task_id: 0 for task in tasks}
        graph = {task.task_id: [] for task in tasks}
        
        # Build dependency graph
        for task in tasks:
            for dep in task.dependencies:
                if dep in graph:
                    graph[dep].append(task.task_id)
                    in_degree[task.task_id] += 1
        
        # Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _evaluate_task_condition(self, task: ComponentTask, 
                                results: Dict[str, TaskResult], 
                                context: Dict[str, Any]) -> bool:
        """Evaluate whether a task should be executed based on conditions"""
        # Simple condition evaluation based on task parameters
        condition = task.parameters.get("condition")
        if not condition:
            return True
        
        # Example condition evaluation (can be extended)
        if condition.get("type") == "result_check":
            dep_id = condition.get("dependency")
            expected_status = condition.get("status", "success")
            
            if dep_id in results:
                return results[dep_id].status == expected_status
        
        return True
    
    async def _rollback_tasks(self, results: Dict[str, TaskResult]):
        """Rollback completed tasks (if components support rollback)"""
        logger.info("Attempting to rollback completed tasks")
        
        for task_id, result in results.items():
            if result.status == "success":
                try:
                    component_instance = self.registry.get_component(result.component_id)
                    if component_instance and hasattr(component_instance.component, "rollback"):
                        await component_instance.component.rollback(task_id)
                        logger.info(f"Rolled back task {task_id}")
                except Exception as e:
                    logger.error(f"Failed to rollback task {task_id}: {e}")
    
    def _update_metrics(self, execution_context: Dict[str, Any], success: bool):
        """Update orchestration metrics"""
        self._orchestration_metrics["total_executions"] += 1
        
        if success:
            self._orchestration_metrics["successful_executions"] += 1
        else:
            self._orchestration_metrics["failed_executions"] += 1
        
        # Update average execution time
        if "duration" in execution_context:
            total_executions = self._orchestration_metrics["total_executions"]
            current_avg = self._orchestration_metrics["average_execution_time"]
            new_duration = execution_context["duration"]
            
            self._orchestration_metrics["average_execution_time"] = (
                (current_avg * (total_executions - 1) + new_duration) / total_executions
            )
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running execution"""
        return self._active_executions.get(execution_id)
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get orchestration performance metrics"""
        return self._orchestration_metrics.copy()
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self._execution_history[-limit:]

# Global orchestrator instance
component_orchestrator = ComponentOrchestrator()

# Utility functions
async def execute_orchestration_plan(plan: OrchestrationPlan) -> Dict[str, Any]:
    """Execute an orchestration plan using the global orchestrator"""
    return await component_orchestrator.execute_plan(plan)

def create_simple_plan(name: str, tasks: List[ComponentTask], 
                      strategy: OrchestrationStrategy = OrchestrationStrategy.SEQUENTIAL) -> OrchestrationPlan:
    """Create a simple orchestration plan"""
    return OrchestrationPlan(
        plan_id=str(uuid.uuid4()),
        name=name,
        strategy=strategy,
        tasks=tasks
    )

def create_analysis_generation_plan(analysis_data: Any, generation_prompt: str) -> OrchestrationPlan:
    """Create a plan that combines analysis and generation components"""
    tasks = [
        ComponentTask(
            task_id="validate_data",
            component_id="data_validator",
            operation="validate_data",
            input_data=analysis_data,
            parameters={"data_type": "numerical"}
        ),
        ComponentTask(
            task_id="analyze_data",
            component_id="statistical_analyzer",
            operation="analyze",
            input_data={
                "request_id": str(uuid.uuid4()),
                "analysis_type": "descriptive",
                "data": analysis_data,
                "parameters": {},
                "metadata": {}
            },
            dependencies=["validate_data"]
        ),
        ComponentTask(
            task_id="generate_report",
            component_id="text_generator",
            operation="generate",
            input_data={
                "request_id": str(uuid.uuid4()),
                "generation_type": "report",
                "prompt": generation_prompt,
                "parameters": {"format": "markdown"},
                "constraints": {"max_length": 1000},
                "metadata": {}
            },
            dependencies=["analyze_data"]
        )
    ]
    
    return OrchestrationPlan(
        plan_id=str(uuid.uuid4()),
        name="Analysis and Report Generation",
        strategy=OrchestrationStrategy.SEQUENTIAL,
        tasks=tasks
    )