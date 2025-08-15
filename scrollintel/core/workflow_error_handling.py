"""
Workflow Error Handling System for ScrollIntel

This module provides comprehensive error handling, recovery mechanisms,
and rollback capabilities for workflow executions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
import traceback
from collections import defaultdict

from .scroll_conductor import (
    WorkflowExecution, StepExecution, WorkflowDefinition, WorkflowStep,
    WorkflowStatus, StepStatus, RetryPolicy
)

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    NETWORK = "network"
    AGENT = "agent"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Available recovery actions"""
    RETRY = "retry"
    SKIP = "skip"
    ROLLBACK = "rollback"
    COMPENSATE = "compensate"
    ESCALATE = "escalate"
    FAIL = "fail"
    CONTINUE = "continue"

class CompensationStatus(Enum):
    """Status of compensation actions"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ErrorContext:
    """Context information for an error"""
    workflow_id: str
    execution_id: str
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    error_message: str = ""
    error_code: Optional[str] = None
    exception_type: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorClassification:
    """Classification of an error"""
    category: ErrorCategory
    severity: ErrorSeverity
    is_retryable: bool
    is_transient: bool
    recovery_suggestions: List[RecoveryAction] = field(default_factory=list)
    estimated_recovery_time: Optional[int] = None  # seconds
    confidence: float = 0.0

@dataclass
class CompensationAction:
    """Definition of a compensation action"""
    id: str
    step_id: str
    action_type: str
    action_data: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3
    status: CompensationStatus = CompensationStatus.PENDING
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryPlan:
    """Recovery plan for handling errors"""
    id: str
    error_context: ErrorContext
    classification: ErrorClassification
    primary_action: RecoveryAction
    fallback_actions: List[RecoveryAction] = field(default_factory=list)
    compensation_actions: List[CompensationAction] = field(default_factory=list)
    timeout: int = 600
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class ErrorClassifier:
    """
    Classifies errors and suggests recovery actions.
    """
    
    def __init__(self):
        self._classification_rules: List[Callable] = []
        self._error_patterns: Dict[str, ErrorClassification] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default error classification rules"""
        
        # Timeout errors
        self.add_classification_rule(
            lambda ctx: "timeout" in ctx.error_message.lower() or ctx.error_code == "timeout",
            ErrorClassification(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                is_transient=True,
                recovery_suggestions=[RecoveryAction.RETRY, RecoveryAction.SKIP],
                estimated_recovery_time=30,
                confidence=0.9
            )
        )
        
        # Validation errors
        self.add_classification_rule(
            lambda ctx: "validation" in ctx.error_message.lower() or ctx.error_code == "validation_error",
            ErrorClassification(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                is_retryable=False,
                is_transient=False,
                recovery_suggestions=[RecoveryAction.FAIL, RecoveryAction.ESCALATE],
                confidence=0.95
            )
        )
        
        # Resource errors
        self.add_classification_rule(
            lambda ctx: any(keyword in ctx.error_message.lower() 
                          for keyword in ["memory", "disk", "cpu", "resource"]),
            ErrorClassification(
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                is_retryable=True,
                is_transient=True,
                recovery_suggestions=[RecoveryAction.RETRY, RecoveryAction.ESCALATE],
                estimated_recovery_time=60,
                confidence=0.8
            )
        )
        
        # Network errors
        self.add_classification_rule(
            lambda ctx: any(keyword in ctx.error_message.lower() 
                          for keyword in ["connection", "network", "unreachable", "dns"]),
            ErrorClassification(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                is_transient=True,
                recovery_suggestions=[RecoveryAction.RETRY, RecoveryAction.SKIP],
                estimated_recovery_time=15,
                confidence=0.85
            )
        )
        
        # Agent errors
        self.add_classification_rule(
            lambda ctx: ctx.error_code in ["agent_unavailable", "capacity_exceeded"],
            ErrorClassification(
                category=ErrorCategory.AGENT,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                is_transient=True,
                recovery_suggestions=[RecoveryAction.RETRY, RecoveryAction.CONTINUE],
                estimated_recovery_time=30,
                confidence=0.9
            )
        )
        
        # System errors
        self.add_classification_rule(
            lambda ctx: ctx.exception_type in ["SystemError", "RuntimeError", "OSError"],
            ErrorClassification(
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                is_retryable=True,
                is_transient=False,
                recovery_suggestions=[RecoveryAction.ESCALATE, RecoveryAction.ROLLBACK],
                estimated_recovery_time=120,
                confidence=0.7
            )
        )
    
    def add_classification_rule(self, condition: Callable[[ErrorContext], bool],
                              classification: ErrorClassification):
        """Add a custom classification rule"""
        self._classification_rules.append((condition, classification))
    
    def classify_error(self, error_context: ErrorContext) -> ErrorClassification:
        """Classify an error and suggest recovery actions"""
        
        # Check cached patterns first
        error_key = f"{error_context.error_code}:{error_context.exception_type}"
        if error_key in self._error_patterns:
            cached = self._error_patterns[error_key]
            logger.debug(f"Using cached classification for {error_key}")
            return cached
        
        # Apply classification rules
        for condition, classification in self._classification_rules:
            try:
                if condition(error_context):
                    self._error_patterns[error_key] = classification
                    return classification
            except Exception as e:
                logger.warning(f"Error in classification rule: {e}")
                continue
        
        # Default classification for unknown errors
        default_classification = ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            is_transient=True,
            recovery_suggestions=[RecoveryAction.RETRY, RecoveryAction.ESCALATE],
            estimated_recovery_time=60,
            confidence=0.5
        )
        
        self._error_patterns[error_key] = default_classification
        return default_classification

class CompensationManager:
    """
    Manages compensation actions for workflow rollbacks.
    """
    
    def __init__(self):
        self._compensation_handlers: Dict[str, Callable] = {}
        self._active_compensations: Dict[str, List[CompensationAction]] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default compensation handlers"""
        
        # File operations compensation
        self.register_handler("file_create", self._compensate_file_create)
        self.register_handler("file_delete", self._compensate_file_delete)
        self.register_handler("file_modify", self._compensate_file_modify)
        
        # Database operations compensation
        self.register_handler("db_insert", self._compensate_db_insert)
        self.register_handler("db_update", self._compensate_db_update)
        self.register_handler("db_delete", self._compensate_db_delete)
        
        # API calls compensation
        self.register_handler("api_call", self._compensate_api_call)
        
        # Resource allocation compensation
        self.register_handler("resource_allocate", self._compensate_resource_allocate)
    
    def register_handler(self, action_type: str, handler: Callable):
        """Register a compensation handler"""
        self._compensation_handlers[action_type] = handler
        logger.debug(f"Registered compensation handler for {action_type}")
    
    async def execute_compensation(self, execution_id: str, 
                                 compensation_actions: List[CompensationAction]) -> bool:
        """Execute compensation actions"""
        if not compensation_actions:
            return True
        
        self._active_compensations[execution_id] = compensation_actions
        success = True
        
        try:
            # Execute compensations in reverse order
            for action in reversed(compensation_actions):
                action.status = CompensationStatus.RUNNING
                action.start_time = datetime.now()
                
                try:
                    if action.action_type in self._compensation_handlers:
                        handler = self._compensation_handlers[action.action_type]
                        
                        # Execute with timeout
                        await asyncio.wait_for(
                            handler(action),
                            timeout=action.timeout
                        )
                        
                        action.status = CompensationStatus.COMPLETED
                    else:
                        logger.warning(f"No compensation handler for {action.action_type}")
                        action.status = CompensationStatus.SKIPPED
                
                except asyncio.TimeoutError:
                    action.status = CompensationStatus.FAILED
                    action.error = "Compensation timeout"
                    success = False
                
                except Exception as e:
                    action.status = CompensationStatus.FAILED
                    action.error = str(e)
                    success = False
                    logger.error(f"Compensation failed for {action.id}: {e}")
                
                finally:
                    action.end_time = datetime.now()
            
            return success
            
        finally:
            self._active_compensations.pop(execution_id, None)
    
    async def _compensate_file_create(self, action: CompensationAction):
        """Compensate file creation by deleting the file"""
        import os
        file_path = action.action_data.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Compensated file creation: deleted {file_path}")
    
    async def _compensate_file_delete(self, action: CompensationAction):
        """Compensate file deletion by restoring from backup"""
        # This would restore from backup if available
        logger.info(f"File deletion compensation not fully implemented")
    
    async def _compensate_file_modify(self, action: CompensationAction):
        """Compensate file modification by restoring original content"""
        import os
        file_path = action.action_data.get("file_path")
        original_content = action.action_data.get("original_content")
        
        if file_path and original_content and os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(original_content)
            logger.info(f"Compensated file modification: restored {file_path}")
    
    async def _compensate_db_insert(self, action: CompensationAction):
        """Compensate database insert by deleting the record"""
        # This would delete the inserted record
        logger.info(f"Database insert compensation not fully implemented")
    
    async def _compensate_db_update(self, action: CompensationAction):
        """Compensate database update by restoring original values"""
        # This would restore original values
        logger.info(f"Database update compensation not fully implemented")
    
    async def _compensate_db_delete(self, action: CompensationAction):
        """Compensate database delete by restoring the record"""
        # This would restore the deleted record
        logger.info(f"Database delete compensation not fully implemented")
    
    async def _compensate_api_call(self, action: CompensationAction):
        """Compensate API call by making reverse call if possible"""
        # This would make a compensating API call
        logger.info(f"API call compensation not fully implemented")
    
    async def _compensate_resource_allocate(self, action: CompensationAction):
        """Compensate resource allocation by releasing resources"""
        # This would release allocated resources
        logger.info(f"Resource allocation compensation not fully implemented")

class WorkflowErrorHandler:
    """
    Main error handling system for workflows.
    """
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.compensation_manager = CompensationManager()
        self._recovery_plans: Dict[str, RecoveryPlan] = {}
        self._error_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._escalation_handlers: List[Callable] = []
        
        # Statistics
        self._error_stats: Dict[str, int] = defaultdict(int)
        self._recovery_stats: Dict[str, int] = defaultdict(int)
    
    def add_error_handler(self, error_type: str, handler: Callable):
        """Add custom error handler"""
        self._error_handlers[error_type].append(handler)
    
    def add_escalation_handler(self, handler: Callable):
        """Add escalation handler"""
        self._escalation_handlers.append(handler)
    
    async def handle_workflow_error(self, workflow_execution: WorkflowExecution,
                                  step_execution: Optional[StepExecution] = None,
                                  error: Optional[Exception] = None) -> RecoveryPlan:
        """Handle workflow error and create recovery plan"""
        
        # Create error context
        error_context = ErrorContext(
            workflow_id=workflow_execution.workflow_id,
            execution_id=workflow_execution.id,
            step_id=step_execution.step_id if step_execution else None,
            agent_id=step_execution.agent_id if step_execution else None,
            error_message=str(error) if error else workflow_execution.error or "",
            error_code=step_execution.error_code if step_execution else workflow_execution.error_code,
            exception_type=type(error).__name__ if error else None,
            stack_trace=traceback.format_exc() if error else None,
            metadata={
                "workflow_status": workflow_execution.status.value,
                "step_status": step_execution.status.value if step_execution else None,
                "retry_count": step_execution.retry_count if step_execution else workflow_execution.retry_count
            }
        )
        
        # Classify error
        classification = self.classifier.classify_error(error_context)
        
        # Update statistics
        self._error_stats[classification.category.value] += 1
        
        # Create recovery plan
        recovery_plan = RecoveryPlan(
            id=str(uuid.uuid4()),
            error_context=error_context,
            classification=classification,
            primary_action=classification.recovery_suggestions[0] if classification.recovery_suggestions else RecoveryAction.FAIL,
            fallback_actions=classification.recovery_suggestions[1:] if len(classification.recovery_suggestions) > 1 else [],
            timeout=classification.estimated_recovery_time or 300
        )
        
        # Generate compensation actions if needed
        if recovery_plan.primary_action in [RecoveryAction.ROLLBACK, RecoveryAction.COMPENSATE]:
            recovery_plan.compensation_actions = await self._generate_compensation_actions(
                workflow_execution, step_execution
            )
        
        # Store recovery plan
        self._recovery_plans[recovery_plan.id] = recovery_plan
        
        logger.info(f"Created recovery plan {recovery_plan.id} for error in {error_context.workflow_id}")
        
        return recovery_plan
    
    async def execute_recovery_plan(self, recovery_plan: RecoveryPlan,
                                  workflow_execution: WorkflowExecution,
                                  step_execution: Optional[StepExecution] = None) -> bool:
        """Execute a recovery plan"""
        
        recovery_plan.executed_at = datetime.now()
        
        try:
            # Try primary action first
            success = await self._execute_recovery_action(
                recovery_plan.primary_action,
                recovery_plan,
                workflow_execution,
                step_execution
            )
            
            if success:
                recovery_plan.success = True
                self._recovery_stats[recovery_plan.primary_action.value] += 1
                logger.info(f"Recovery plan {recovery_plan.id} succeeded with primary action")
                return True
            
            # Try fallback actions
            for fallback_action in recovery_plan.fallback_actions:
                logger.info(f"Trying fallback action: {fallback_action.value}")
                
                success = await self._execute_recovery_action(
                    fallback_action,
                    recovery_plan,
                    workflow_execution,
                    step_execution
                )
                
                if success:
                    recovery_plan.success = True
                    self._recovery_stats[fallback_action.value] += 1
                    logger.info(f"Recovery plan {recovery_plan.id} succeeded with fallback action")
                    return True
            
            # All recovery actions failed
            recovery_plan.success = False
            logger.error(f"Recovery plan {recovery_plan.id} failed - all actions exhausted")
            return False
            
        except Exception as e:
            logger.error(f"Recovery plan execution failed: {e}")
            recovery_plan.success = False
            return False
        
        finally:
            recovery_plan.completed_at = datetime.now()
    
    async def _execute_recovery_action(self, action: RecoveryAction,
                                     recovery_plan: RecoveryPlan,
                                     workflow_execution: WorkflowExecution,
                                     step_execution: Optional[StepExecution] = None) -> bool:
        """Execute a specific recovery action"""
        
        try:
            if action == RecoveryAction.RETRY:
                return await self._retry_action(recovery_plan, workflow_execution, step_execution)
            
            elif action == RecoveryAction.SKIP:
                return await self._skip_action(recovery_plan, workflow_execution, step_execution)
            
            elif action == RecoveryAction.ROLLBACK:
                return await self._rollback_action(recovery_plan, workflow_execution)
            
            elif action == RecoveryAction.COMPENSATE:
                return await self._compensate_action(recovery_plan, workflow_execution)
            
            elif action == RecoveryAction.ESCALATE:
                return await self._escalate_action(recovery_plan, workflow_execution)
            
            elif action == RecoveryAction.CONTINUE:
                return await self._continue_action(recovery_plan, workflow_execution, step_execution)
            
            elif action == RecoveryAction.FAIL:
                return await self._fail_action(recovery_plan, workflow_execution)
            
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery action {action.value} failed: {e}")
            return False
    
    async def _retry_action(self, recovery_plan: RecoveryPlan,
                          workflow_execution: WorkflowExecution,
                          step_execution: Optional[StepExecution] = None) -> bool:
        """Retry the failed operation"""
        if step_execution:
            # Reset step status for retry
            step_execution.status = StepStatus.PENDING
            step_execution.error = None
            step_execution.error_code = None
            step_execution.retry_count += 1
            
            logger.info(f"Retrying step {step_execution.step_id} (attempt {step_execution.retry_count})")
            return True
        else:
            # Retry entire workflow
            workflow_execution.status = WorkflowStatus.PENDING
            workflow_execution.error = None
            workflow_execution.error_code = None
            workflow_execution.retry_count += 1
            
            logger.info(f"Retrying workflow {workflow_execution.workflow_id} (attempt {workflow_execution.retry_count})")
            return True
    
    async def _skip_action(self, recovery_plan: RecoveryPlan,
                         workflow_execution: WorkflowExecution,
                         step_execution: Optional[StepExecution] = None) -> bool:
        """Skip the failed step and continue"""
        if step_execution:
            step_execution.status = StepStatus.SKIPPED
            logger.info(f"Skipped failed step {step_execution.step_id}")
            return True
        return False
    
    async def _rollback_action(self, recovery_plan: RecoveryPlan,
                             workflow_execution: WorkflowExecution) -> bool:
        """Rollback the workflow execution"""
        logger.info(f"Rolling back workflow {workflow_execution.id}")
        
        # Execute compensation actions
        if recovery_plan.compensation_actions:
            success = await self.compensation_manager.execute_compensation(
                workflow_execution.id,
                recovery_plan.compensation_actions
            )
            
            if success:
                workflow_execution.status = WorkflowStatus.CANCELLED
                logger.info(f"Workflow {workflow_execution.id} rolled back successfully")
                return True
            else:
                logger.error(f"Rollback failed for workflow {workflow_execution.id}")
                return False
        
        # Simple rollback without compensation
        workflow_execution.status = WorkflowStatus.CANCELLED
        return True
    
    async def _compensate_action(self, recovery_plan: RecoveryPlan,
                               workflow_execution: WorkflowExecution) -> bool:
        """Execute compensation actions"""
        if recovery_plan.compensation_actions:
            return await self.compensation_manager.execute_compensation(
                workflow_execution.id,
                recovery_plan.compensation_actions
            )
        return True
    
    async def _escalate_action(self, recovery_plan: RecoveryPlan,
                             workflow_execution: WorkflowExecution) -> bool:
        """Escalate the error to human operators"""
        logger.warning(f"Escalating error for workflow {workflow_execution.id}")
        
        # Notify escalation handlers
        for handler in self._escalation_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(recovery_plan, workflow_execution)
                else:
                    handler(recovery_plan, workflow_execution)
            except Exception as e:
                logger.error(f"Escalation handler failed: {e}")
        
        # Mark workflow as requiring manual intervention
        workflow_execution.status = WorkflowStatus.PAUSED
        workflow_execution.metadata["escalated"] = True
        workflow_execution.metadata["escalation_time"] = datetime.now().isoformat()
        
        return True
    
    async def _continue_action(self, recovery_plan: RecoveryPlan,
                             workflow_execution: WorkflowExecution,
                             step_execution: Optional[StepExecution] = None) -> bool:
        """Continue execution despite the error"""
        logger.info(f"Continuing workflow {workflow_execution.id} despite error")
        
        if step_execution:
            step_execution.status = StepStatus.COMPLETED
            step_execution.output_data = {"error_ignored": True}
        
        return True
    
    async def _fail_action(self, recovery_plan: RecoveryPlan,
                         workflow_execution: WorkflowExecution) -> bool:
        """Fail the workflow execution"""
        logger.info(f"Failing workflow {workflow_execution.id}")
        workflow_execution.status = WorkflowStatus.FAILED
        return True
    
    async def _generate_compensation_actions(self, workflow_execution: WorkflowExecution,
                                           step_execution: Optional[StepExecution] = None) -> List[CompensationAction]:
        """Generate compensation actions for completed steps"""
        compensation_actions = []
        
        # Generate compensation for completed steps
        for step_id, step_exec in workflow_execution.steps.items():
            if step_exec.status == StepStatus.COMPLETED and step_exec.output_data:
                # Extract compensation info from step output
                output_data = step_exec.output_data
                
                if isinstance(output_data, dict) and "compensation" in output_data:
                    comp_info = output_data["compensation"]
                    
                    compensation_action = CompensationAction(
                        id=str(uuid.uuid4()),
                        step_id=step_id,
                        action_type=comp_info.get("type", "generic"),
                        action_data=comp_info.get("data", {}),
                        timeout=comp_info.get("timeout", 300)
                    )
                    
                    compensation_actions.append(compensation_action)
        
        return compensation_actions
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return {
            "error_counts": dict(self._error_stats),
            "recovery_counts": dict(self._recovery_stats),
            "total_errors": sum(self._error_stats.values()),
            "total_recoveries": sum(self._recovery_stats.values()),
            "recovery_rate": (
                sum(self._recovery_stats.values()) / sum(self._error_stats.values())
                if sum(self._error_stats.values()) > 0 else 0
            )
        }
    
    def get_recovery_plan(self, plan_id: str) -> Optional[RecoveryPlan]:
        """Get recovery plan by ID"""
        return self._recovery_plans.get(plan_id)
    
    def list_active_recovery_plans(self) -> List[RecoveryPlan]:
        """List all active recovery plans"""
        return [
            plan for plan in self._recovery_plans.values()
            if plan.executed_at is None or plan.completed_at is None
        ]

# Global error handler instance
workflow_error_handler = WorkflowErrorHandler()

# Utility functions
async def handle_workflow_error(workflow_execution: WorkflowExecution,
                              step_execution: Optional[StepExecution] = None,
                              error: Optional[Exception] = None) -> RecoveryPlan:
    """Handle workflow error using global handler"""
    return await workflow_error_handler.handle_workflow_error(
        workflow_execution, step_execution, error
    )

async def execute_recovery_plan(recovery_plan: RecoveryPlan,
                              workflow_execution: WorkflowExecution,
                              step_execution: Optional[StepExecution] = None) -> bool:
    """Execute recovery plan using global handler"""
    return await workflow_error_handler.execute_recovery_plan(
        recovery_plan, workflow_execution, step_execution
    )

def add_error_handler(error_type: str, handler: Callable):
    """Add custom error handler"""
    workflow_error_handler.add_error_handler(error_type, handler)

def add_escalation_handler(handler: Callable):
    """Add escalation handler"""
    workflow_error_handler.add_escalation_handler(handler)

def get_error_statistics() -> Dict[str, Any]:
    """Get error handling statistics"""
    return workflow_error_handler.get_error_statistics()

# Error handling decorators
def handle_step_errors(recovery_action: RecoveryAction = RecoveryAction.RETRY):
    """Decorator for automatic step error handling"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Step error in {func.__name__}: {e}")
                # In a full implementation, this would integrate with the workflow context
                raise e
        return wrapper
    return decorator

def compensate_action(action_type: str, compensation_data: Dict[str, Any]):
    """Decorator to mark actions that need compensation"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Add compensation info to result
            if isinstance(result, dict):
                result["compensation"] = {
                    "type": action_type,
                    "data": compensation_data
                }
            
            return result
        return wrapper
    return decorator