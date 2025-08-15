"""
ScrollConductor - Master Orchestration System for ScrollIntel

This module implements the central orchestration engine that manages complex
workflows, coordinates sub-agents, handles failures, and provides unified
responses for multi-agent operations.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import threading
from collections import defaultdict, deque
import traceback

from .enhanced_specialized_agent import (
    EnhancedSpecializedAgent, AgentRequest, AgentResponse, 
    AgentCapability, RequestPriority, enhanced_agent_registry
)
from .schema_validation import schema_validator, ValidationResult
from .agent_monitoring import agent_monitor, record_agent_request

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"

class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"

class RetryPolicy(Enum):
    """Retry policy types"""
    NONE = "none"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"

@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    id: str
    name: str
    agent_type: str
    capabilities_required: List[str]
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None
    timeout: int = 300
    retry_policy: RetryPolicy = RetryPolicy.FIXED_DELAY
    max_retries: int = 3
    retry_delay: int = 5
    depends_on: List[str] = field(default_factory=list)
    parallel_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout: int = 1800  # 30 minutes default
    retry_policy: RetryPolicy = RetryPolicy.FIXED_DELAY
    max_retries: int = 3
    error_handling: str = "fail_fast"  # fail_fast, continue, rollback
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class StepExecution:
    """Runtime step execution state"""
    step_id: str
    status: StepStatus = StepStatus.PENDING
    agent_id: Optional[str] = None
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    retry_count: int = 0
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Runtime workflow execution state"""
    id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    steps: Dict[str, StepExecution] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    """Workflow execution metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    average_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    step_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

class WorkflowRegistry:
    """
    Registry for managing workflow definitions.
    """
    
    def __init__(self):
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._workflow_versions: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Register a workflow definition"""
        try:
            with self._lock:
                # Validate workflow definition
                if not self._validate_workflow(workflow):
                    return False
                
                self._workflows[workflow.id] = workflow
                
                # Track versions
                base_name = workflow.name
                if workflow.version not in self._workflow_versions[base_name]:
                    self._workflow_versions[base_name].append(workflow.version)
                    self._workflow_versions[base_name].sort(reverse=True)
                
                logger.info(f"Registered workflow {workflow.name} v{workflow.version} ({workflow.id})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register workflow {workflow.id}: {e}")
            return False
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow by ID"""
        with self._lock:
            return self._workflows.get(workflow_id)
    
    def get_workflow_by_name(self, name: str, version: Optional[str] = None) -> Optional[WorkflowDefinition]:
        """Get workflow by name and version"""
        with self._lock:
            if version is None:
                # Get latest version
                versions = self._workflow_versions.get(name, [])
                if not versions:
                    return None
                version = versions[0]
            
            # Find workflow with matching name and version
            for workflow in self._workflows.values():
                if workflow.name == name and workflow.version == version:
                    return workflow
            
            return None
    
    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all registered workflows"""
        with self._lock:
            return list(self._workflows.values())
    
    def unregister_workflow(self, workflow_id: str) -> bool:
        """Unregister a workflow"""
        with self._lock:
            if workflow_id in self._workflows:
                workflow = self._workflows[workflow_id]
                del self._workflows[workflow_id]
                
                # Update version tracking
                base_name = workflow.name
                if workflow.version in self._workflow_versions[base_name]:
                    self._workflow_versions[base_name].remove(workflow.version)
                    if not self._workflow_versions[base_name]:
                        del self._workflow_versions[base_name]
                
                logger.info(f"Unregistered workflow {workflow_id}")
                return True
            
            return False
    
    def _validate_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Validate workflow definition"""
        try:
            # Basic validation
            if not workflow.id or not workflow.name or not workflow.steps:
                logger.error("Workflow missing required fields")
                return False
            
            # Validate step dependencies
            step_ids = {step.id for step in workflow.steps}
            for step in workflow.steps:
                for dep in step.depends_on:
                    if dep not in step_ids:
                        logger.error(f"Step {step.id} depends on non-existent step {dep}")
                        return False
            
            # Check for circular dependencies
            if self._has_circular_dependencies(workflow.steps):
                logger.error("Workflow has circular dependencies")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow validation error: {e}")
            return False
    
    def _has_circular_dependencies(self, steps: List[WorkflowStep]) -> bool:
        """Check for circular dependencies in workflow steps"""
        # Build dependency graph
        graph = {step.id: step.depends_on for step in steps}
        
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True
        
        return False

class ScrollConductor:
    """
    Master orchestration engine for ScrollIntel workflows.
    """
    
    def __init__(self):
        self.workflow_registry = WorkflowRegistry()
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_history: deque = deque(maxlen=1000)
        self._execution_metrics: Dict[str, ExecutionMetrics] = defaultdict(ExecutionMetrics)
        self._lock = threading.RLock()
        self._executor_pool = {}
        self._shutdown_event = asyncio.Event()
        
        # Event handlers
        self._step_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._workflow_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Initialize built-in workflows
        self._register_builtin_workflows()
    
    def _register_builtin_workflows(self):
        """Register built-in workflow templates"""
        # Simple analysis workflow
        analysis_workflow = WorkflowDefinition(
            id="simple_analysis",
            name="Simple Analysis",
            description="Basic data analysis workflow",
            steps=[
                WorkflowStep(
                    id="validate_input",
                    name="Validate Input Data",
                    agent_type="validation",
                    capabilities_required=["validation"],
                    timeout=60
                ),
                WorkflowStep(
                    id="analyze_data",
                    name="Analyze Data",
                    agent_type="analysis",
                    capabilities_required=["analysis"],
                    depends_on=["validate_input"],
                    timeout=300
                ),
                WorkflowStep(
                    id="generate_report",
                    name="Generate Report",
                    agent_type="generation",
                    capabilities_required=["generation"],
                    depends_on=["analyze_data"],
                    timeout=120
                )
            ]
        )
        
        self.workflow_registry.register_workflow(analysis_workflow)
        
        # Parallel processing workflow
        parallel_workflow = WorkflowDefinition(
            id="parallel_processing",
            name="Parallel Processing",
            description="Process multiple data streams in parallel",
            execution_mode=ExecutionMode.PARALLEL,
            steps=[
                WorkflowStep(
                    id="split_data",
                    name="Split Data",
                    agent_type="transformation",
                    capabilities_required=["transformation"],
                    timeout=60
                ),
                WorkflowStep(
                    id="process_stream_1",
                    name="Process Stream 1",
                    agent_type="analysis",
                    capabilities_required=["analysis"],
                    depends_on=["split_data"],
                    parallel_group="processing",
                    timeout=300
                ),
                WorkflowStep(
                    id="process_stream_2",
                    name="Process Stream 2",
                    agent_type="analysis",
                    capabilities_required=["analysis"],
                    depends_on=["split_data"],
                    parallel_group="processing",
                    timeout=300
                ),
                WorkflowStep(
                    id="merge_results",
                    name="Merge Results",
                    agent_type="transformation",
                    capabilities_required=["transformation"],
                    depends_on=["process_stream_1", "process_stream_2"],
                    timeout=120
                )
            ]
        )
        
        self.workflow_registry.register_workflow(parallel_workflow)
    
    async def execute_workflow(self, workflow_id: str, input_data: Any,
                             user_id: Optional[str] = None,
                             session_id: Optional[str] = None,
                             priority: RequestPriority = RequestPriority.NORMAL,
                             metadata: Optional[Dict[str, Any]] = None) -> WorkflowExecution:
        """Execute a workflow"""
        
        # Get workflow definition
        workflow_def = self.workflow_registry.get_workflow(workflow_id)
        if not workflow_def:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Create execution instance
        execution = WorkflowExecution(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Initialize step executions
        for step in workflow_def.steps:
            execution.steps[step.id] = StepExecution(
                step_id=step.id,
                trace_id=execution.trace_id
            )
        
        # Store active execution
        with self._lock:
            self._active_executions[execution.id] = execution
        
        try:
            # Validate input
            if workflow_def.input_schema:
                validation_result = schema_validator.validate_request(
                    input_data, "workflow_input"
                )
                if not validation_result.valid:
                    raise ValueError(f"Input validation failed: {validation_result.errors}")
            
            # Start execution
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()
            
            await self._emit_workflow_event("started", execution)
            
            # Execute workflow based on mode
            if workflow_def.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(workflow_def, execution)
            elif workflow_def.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(workflow_def, execution)
            elif workflow_def.execution_mode == ExecutionMode.PIPELINE:
                await self._execute_pipeline(workflow_def, execution)
            else:
                await self._execute_conditional(workflow_def, execution)
            
            # Finalize execution
            execution.end_time = datetime.now()
            execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Determine final status
            failed_steps = [s for s in execution.steps.values() if s.status == StepStatus.FAILED]
            if failed_steps and workflow_def.error_handling == "fail_fast":
                execution.status = WorkflowStatus.FAILED
                execution.error = f"Workflow failed due to {len(failed_steps)} failed steps"
            else:
                execution.status = WorkflowStatus.COMPLETED
            
            # Update metrics
            self._update_execution_metrics(workflow_id, execution)
            
            await self._emit_workflow_event("completed", execution)
            
            return execution
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            await self._emit_workflow_event("failed", execution)
            
            return execution
        
        finally:
            # Move to history and cleanup
            with self._lock:
                self._active_executions.pop(execution.id, None)
                self._execution_history.append(execution)
    
    async def _execute_sequential(self, workflow_def: WorkflowDefinition, 
                                execution: WorkflowExecution):
        """Execute workflow steps sequentially"""
        # Build execution order based on dependencies
        execution_order = self._build_execution_order(workflow_def.steps)
        
        for step_id in execution_order:
            step_def = next(s for s in workflow_def.steps if s.id == step_id)
            step_exec = execution.steps[step_id]
            
            # Check if step should be skipped due to condition
            if step_def.condition and not self._evaluate_condition(step_def.condition, execution.context):
                step_exec.status = StepStatus.SKIPPED
                continue
            
            # Execute step
            await self._execute_step(step_def, step_exec, execution)
            
            # Handle step failure
            if step_exec.status == StepStatus.FAILED:
                if workflow_def.error_handling == "fail_fast":
                    break
                elif workflow_def.error_handling == "continue":
                    continue
                elif workflow_def.error_handling == "rollback":
                    await self._rollback_execution(workflow_def, execution, step_id)
                    break
    
    async def _execute_parallel(self, workflow_def: WorkflowDefinition,
                              execution: WorkflowExecution):
        """Execute workflow steps in parallel where possible"""
        # Group steps by parallel groups and dependencies
        parallel_groups = self._group_parallel_steps(workflow_def.steps)
        
        for group in parallel_groups:
            if len(group) == 1:
                # Single step
                step_id = group[0]
                step_def = next(s for s in workflow_def.steps if s.id == step_id)
                step_exec = execution.steps[step_id]
                await self._execute_step(step_def, step_exec, execution)
            else:
                # Parallel execution
                tasks = []
                for step_id in group:
                    step_def = next(s for s in workflow_def.steps if s.id == step_id)
                    step_exec = execution.steps[step_id]
                    task = asyncio.create_task(
                        self._execute_step(step_def, step_exec, execution)
                    )
                    tasks.append(task)
                
                # Wait for all parallel steps to complete
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_pipeline(self, workflow_def: WorkflowDefinition,
                              execution: WorkflowExecution):
        """Execute workflow as a pipeline with streaming data"""
        # For now, implement as sequential with data passing
        await self._execute_sequential(workflow_def, execution)
    
    async def _execute_conditional(self, workflow_def: WorkflowDefinition,
                                 execution: WorkflowExecution):
        """Execute workflow with conditional branching"""
        # For now, implement as sequential with condition checking
        await self._execute_sequential(workflow_def, execution)
    
    async def _execute_step(self, step_def: WorkflowStep, step_exec: StepExecution,
                          workflow_exec: WorkflowExecution):
        """Execute a single workflow step"""
        step_exec.status = StepStatus.RUNNING
        step_exec.start_time = datetime.now()
        
        await self._emit_step_event("started", step_def, step_exec, workflow_exec)
        
        try:
            # Prepare input data
            input_data = self._prepare_step_input(step_def, workflow_exec)
            step_exec.input_data = input_data
            
            # Find suitable agent
            suitable_agents = enhanced_agent_registry.get_agents_by_capability(
                step_def.capabilities_required[0] if step_def.capabilities_required else "analysis"
            )
            
            if not suitable_agents:
                raise ValueError(f"No agents available for capabilities: {step_def.capabilities_required}")
            
            # Select best agent (simple selection for now)
            selected_agent = suitable_agents[0]
            step_exec.agent_id = selected_agent.agent_id
            
            # Create agent request
            agent_request = AgentRequest(
                id=str(uuid.uuid4()),
                type=step_def.agent_type,
                payload=input_data,
                capabilities_required=step_def.capabilities_required,
                timeout=step_def.timeout,
                correlation_id=workflow_exec.trace_id,
                user_id=workflow_exec.user_id,
                session_id=workflow_exec.session_id,
                metadata={
                    "workflow_id": workflow_exec.workflow_id,
                    "workflow_execution_id": workflow_exec.id,
                    "step_id": step_def.id,
                    **step_def.metadata
                }
            )
            
            # Execute with retry logic
            response = await self._execute_with_retry(
                selected_agent, agent_request, step_def.retry_policy,
                step_def.max_retries, step_def.retry_delay
            )
            
            # Process response
            if response.status == "success":
                step_exec.status = StepStatus.COMPLETED
                step_exec.output_data = response.data
                
                # Update workflow context with output
                self._update_workflow_context(step_def, response.data, workflow_exec)
                
            else:
                step_exec.status = StepStatus.FAILED
                step_exec.error = response.error
                step_exec.error_code = response.error_code
            
            step_exec.end_time = datetime.now()
            step_exec.duration = (step_exec.end_time - step_exec.start_time).total_seconds()
            
            await self._emit_step_event("completed", step_def, step_exec, workflow_exec)
            
        except Exception as e:
            logger.error(f"Step {step_def.id} execution failed: {e}")
            step_exec.status = StepStatus.FAILED
            step_exec.error = str(e)
            step_exec.end_time = datetime.now()
            if step_exec.start_time:
                step_exec.duration = (step_exec.end_time - step_exec.start_time).total_seconds()
            
            await self._emit_step_event("failed", step_def, step_exec, workflow_exec)
    
    async def _execute_with_retry(self, agent: EnhancedSpecializedAgent,
                                request: AgentRequest, retry_policy: RetryPolicy,
                                max_retries: int, base_delay: int) -> AgentResponse:
        """Execute agent request with retry logic"""
        last_response = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await agent.handle_request(request)
                
                if response.status == "success":
                    return response
                
                last_response = response
                
                if attempt < max_retries:
                    # Calculate delay based on retry policy
                    if retry_policy == RetryPolicy.FIXED_DELAY:
                        delay = base_delay
                    elif retry_policy == RetryPolicy.EXPONENTIAL_BACKOFF:
                        delay = base_delay * (2 ** attempt)
                    elif retry_policy == RetryPolicy.LINEAR_BACKOFF:
                        delay = base_delay * (attempt + 1)
                    else:
                        delay = 0
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                
            except Exception as e:
                if attempt == max_retries:
                    # Create error response for final attempt
                    return AgentResponse(
                        request_id=request.id,
                        agent_id=agent.agent_id,
                        status="error",
                        data=None,
                        confidence=0.0,
                        processing_time=0.0,
                        error=str(e),
                        error_code="execution_exception"
                    )
                
                # Wait before retry
                if retry_policy != RetryPolicy.NONE:
                    await asyncio.sleep(base_delay)
        
        return last_response or AgentResponse(
            request_id=request.id,
            agent_id=agent.agent_id,
            status="error",
            data=None,
            confidence=0.0,
            processing_time=0.0,
            error="Max retries exceeded",
            error_code="max_retries_exceeded"
        )
    
    def _prepare_step_input(self, step_def: WorkflowStep, 
                          workflow_exec: WorkflowExecution) -> Dict[str, Any]:
        """Prepare input data for a step based on input mapping"""
        input_data = {}
        
        # Start with workflow input
        if not step_def.input_mapping:
            input_data = workflow_exec.input_data
        else:
            # Apply input mapping
            for output_key, input_path in step_def.input_mapping.items():
                if input_path.startswith("workflow.input"):
                    # Map from workflow input
                    path_parts = input_path.split(".")[2:]  # Skip "workflow.input"
                    value = workflow_exec.input_data
                    for part in path_parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    input_data[output_key] = value
                
                elif input_path.startswith("context."):
                    # Map from workflow context
                    context_key = input_path[8:]  # Remove "context."
                    input_data[output_key] = workflow_exec.context.get(context_key)
                
                elif "." in input_path:
                    # Map from previous step output
                    step_id, output_key_path = input_path.split(".", 1)
                    if step_id in workflow_exec.steps:
                        step_output = workflow_exec.steps[step_id].output_data
                        if isinstance(step_output, dict) and output_key_path in step_output:
                            input_data[output_key] = step_output[output_key_path]
        
        return input_data
    
    def _update_workflow_context(self, step_def: WorkflowStep, output_data: Any,
                               workflow_exec: WorkflowExecution):
        """Update workflow context with step output"""
        if step_def.output_mapping:
            for context_key, output_path in step_def.output_mapping.items():
                if output_path == "output":
                    workflow_exec.context[context_key] = output_data
                elif isinstance(output_data, dict) and output_path in output_data:
                    workflow_exec.context[context_key] = output_data[output_path]
        else:
            # Default: store entire output under step name
            workflow_exec.context[step_def.id] = output_data
    
    def _build_execution_order(self, steps: List[WorkflowStep]) -> List[str]:
        """Build execution order based on step dependencies"""
        # Topological sort
        in_degree = {step.id: 0 for step in steps}
        graph = {step.id: [] for step in steps}
        
        # Build graph and calculate in-degrees
        for step in steps:
            for dep in step.depends_on:
                graph[dep].append(step.id)
                in_degree[step.id] += 1
        
        # Kahn's algorithm
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _group_parallel_steps(self, steps: List[WorkflowStep]) -> List[List[str]]:
        """Group steps that can be executed in parallel"""
        # For now, simple grouping by parallel_group attribute
        groups = []
        processed = set()
        
        for step in steps:
            if step.id in processed:
                continue
            
            if step.parallel_group:
                # Find all steps in the same parallel group
                group = [s.id for s in steps if s.parallel_group == step.parallel_group]
                groups.append(group)
                processed.update(group)
            else:
                groups.append([step.id])
                processed.add(step.id)
        
        return groups
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a step condition"""
        # Simple condition evaluation (in production, use a proper expression evaluator)
        try:
            # Replace context variables
            for key, value in context.items():
                condition = condition.replace(f"${key}", str(value))
            
            # Evaluate simple conditions
            return eval(condition)
        except:
            return True  # Default to true if condition can't be evaluated
    
    async def _rollback_execution(self, workflow_def: WorkflowDefinition,
                                execution: WorkflowExecution, failed_step_id: str):
        """Rollback workflow execution"""
        logger.info(f"Rolling back workflow {execution.id} from step {failed_step_id}")
        
        # For now, just mark as failed
        # In a full implementation, this would undo completed steps
        execution.status = WorkflowStatus.FAILED
        execution.error = f"Workflow rolled back due to failure in step {failed_step_id}"
    
    def _update_execution_metrics(self, workflow_id: str, execution: WorkflowExecution):
        """Update execution metrics"""
        metrics = self._execution_metrics[workflow_id]
        
        metrics.total_executions += 1
        
        if execution.status == WorkflowStatus.COMPLETED:
            metrics.successful_executions += 1
        elif execution.status == WorkflowStatus.FAILED:
            metrics.failed_executions += 1
        elif execution.status == WorkflowStatus.CANCELLED:
            metrics.cancelled_executions += 1
        
        # Update duration metrics
        if execution.duration > 0:
            metrics.min_duration = min(metrics.min_duration, execution.duration)
            metrics.max_duration = max(metrics.max_duration, execution.duration)
            
            # Update average
            total_successful = metrics.successful_executions
            if total_successful > 0:
                metrics.average_duration = (
                    (metrics.average_duration * (total_successful - 1) + execution.duration) / total_successful
                )
        
        # Update error patterns
        if execution.error:
            error_key = execution.error_code or "unknown_error"
            metrics.error_patterns[error_key] = metrics.error_patterns.get(error_key, 0) + 1
        
        metrics.last_updated = datetime.now()
    
    async def _emit_workflow_event(self, event_type: str, execution: WorkflowExecution):
        """Emit workflow event to registered handlers"""
        handlers = self._workflow_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(execution)
                else:
                    handler(execution)
            except Exception as e:
                logger.error(f"Workflow event handler error: {e}")
    
    async def _emit_step_event(self, event_type: str, step_def: WorkflowStep,
                             step_exec: StepExecution, workflow_exec: WorkflowExecution):
        """Emit step event to registered handlers"""
        handlers = self._step_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(step_def, step_exec, workflow_exec)
                else:
                    handler(step_def, step_exec, workflow_exec)
            except Exception as e:
                logger.error(f"Step event handler error: {e}")
    
    def add_workflow_handler(self, event_type: str, handler: Callable):
        """Add workflow event handler"""
        self._workflow_handlers[event_type].append(handler)
    
    def add_step_handler(self, event_type: str, handler: Callable):
        """Add step event handler"""
        self._step_handlers[event_type].append(handler)
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status"""
        with self._lock:
            # Check active executions first
            if execution_id in self._active_executions:
                return self._active_executions[execution_id]
            
            # Check history
            for execution in self._execution_history:
                if execution.id == execution_id:
                    return execution
            
            return None
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all active executions"""
        with self._lock:
            return list(self._active_executions.values())
    
    def get_execution_metrics(self, workflow_id: str) -> Optional[ExecutionMetrics]:
        """Get execution metrics for a workflow"""
        return self._execution_metrics.get(workflow_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        with self._lock:
            if execution_id in self._active_executions:
                execution = self._active_executions[execution_id]
                execution.status = WorkflowStatus.CANCELLED
                execution.end_time = datetime.now()
                if execution.start_time:
                    execution.duration = (execution.end_time - execution.start_time).total_seconds()
                
                await self._emit_workflow_event("cancelled", execution)
                return True
            
            return False

# Global ScrollConductor instance
scroll_conductor = ScrollConductor()

# Utility functions
async def execute_workflow(workflow_id: str, input_data: Any, **kwargs) -> WorkflowExecution:
    """Execute a workflow using the global conductor"""
    return await scroll_conductor.execute_workflow(workflow_id, input_data, **kwargs)

def register_workflow(workflow: WorkflowDefinition) -> bool:
    """Register a workflow definition"""
    return scroll_conductor.workflow_registry.register_workflow(workflow)

def get_workflow_status(execution_id: str) -> Optional[WorkflowExecution]:
    """Get workflow execution status"""
    return scroll_conductor.get_execution_status(execution_id)

def get_active_workflows() -> List[WorkflowExecution]:
    """Get all active workflow executions"""
    return scroll_conductor.get_active_executions()

async def cancel_workflow(execution_id: str) -> bool:
    """Cancel a workflow execution"""
    return await scroll_conductor.cancel_execution(execution_id)