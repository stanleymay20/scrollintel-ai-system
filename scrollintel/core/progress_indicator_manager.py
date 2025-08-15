"""
Progress Indicator Manager - Real-time Progress Tracking System

This module provides comprehensive progress tracking and indication capabilities:
- Real-time progress indicators with detailed step information
- Estimated completion times and remaining duration
- Cancellable operations with cleanup handling
- Partial results display during long operations
- Progress history and analytics

Requirements: 6.1, 6.2
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import json
import time

logger = logging.getLogger(__name__)


class ProgressState(Enum):
    """Progress operation states"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ProgressType(Enum):
    """Types of progress indicators"""
    DETERMINATE = "determinate"  # Known total steps
    INDETERMINATE = "indeterminate"  # Unknown duration
    STREAMING = "streaming"  # Continuous data flow
    BATCH = "batch"  # Batch processing


@dataclass
class ProgressStep:
    """Individual progress step information"""
    step_id: str
    name: str
    description: str
    estimated_duration: Optional[float] = None  # seconds
    actual_duration: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: ProgressState = ProgressState.PENDING
    error_message: Optional[str] = None
    partial_results: Optional[Dict[str, Any]] = None


@dataclass
class ProgressOperation:
    """Complete progress operation tracking"""
    operation_id: str
    name: str
    description: str
    progress_type: ProgressType
    state: ProgressState = ProgressState.PENDING
    
    # Step tracking
    steps: List[ProgressStep] = field(default_factory=list)
    current_step_index: int = 0
    
    # Progress metrics
    progress_percentage: float = 0.0
    items_processed: int = 0
    total_items: Optional[int] = None
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # User interaction
    can_cancel: bool = False
    can_pause: bool = False
    show_partial_results: bool = False
    
    # Results and context
    partial_results: Dict[str, Any] = field(default_factory=dict)
    final_results: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    update_callbacks: List[Callable] = field(default_factory=list)
    completion_callbacks: List[Callable] = field(default_factory=list)


class ProgressIndicatorManager:
    """
    Manages real-time progress indicators and tracking for all operations
    """
    
    def __init__(self):
        self.active_operations: Dict[str, ProgressOperation] = {}
        self.completed_operations: Dict[str, ProgressOperation] = {}
        self.global_callbacks: List[Callable] = []
        self._operation_counter = 0
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Start cleanup task
        self._start_cleanup_task()
    
    async def create_operation(
        self,
        name: str,
        description: str,
        progress_type: ProgressType = ProgressType.DETERMINATE,
        steps: Optional[List[Dict[str, Any]]] = None,
        total_items: Optional[int] = None,
        can_cancel: bool = False,
        can_pause: bool = False,
        show_partial_results: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new progress operation"""
        try:
            operation_id = self._generate_operation_id()
            
            # Create progress steps
            progress_steps = []
            if steps:
                for i, step_data in enumerate(steps):
                    step = ProgressStep(
                        step_id=f"{operation_id}_step_{i}",
                        name=step_data.get("name", f"Step {i+1}"),
                        description=step_data.get("description", ""),
                        estimated_duration=step_data.get("estimated_duration")
                    )
                    progress_steps.append(step)
            
            operation = ProgressOperation(
                operation_id=operation_id,
                name=name,
                description=description,
                progress_type=progress_type,
                steps=progress_steps,
                total_items=total_items,
                can_cancel=can_cancel,
                can_pause=can_pause,
                show_partial_results=show_partial_results,
                context=context or {}
            )
            
            self.active_operations[operation_id] = operation
            
            logger.info(f"Created progress operation: {name} ({operation_id})")
            return operation_id
            
        except Exception as e:
            logger.error(f"Error creating progress operation: {e}")
            raise
    
    async def start_operation(self, operation_id: str) -> None:
        """Start a progress operation"""
        try:
            if operation_id not in self.active_operations:
                raise ValueError(f"Operation {operation_id} not found")
            
            operation = self.active_operations[operation_id]
            operation.state = ProgressState.RUNNING
            operation.start_time = datetime.utcnow()
            operation.last_update = datetime.utcnow()
            
            # Start first step if available
            if operation.steps:
                operation.steps[0].status = ProgressState.RUNNING
                operation.steps[0].start_time = datetime.utcnow()
            
            # Calculate initial estimated completion
            await self._update_estimated_completion(operation)
            
            # Notify callbacks
            await self._notify_callbacks(operation, "started")
            
            logger.info(f"Started operation: {operation.name}")
            
        except Exception as e:
            logger.error(f"Error starting operation: {e}")
            raise
    
    async def update_progress(
        self,
        operation_id: str,
        progress_percentage: Optional[float] = None,
        items_processed: Optional[int] = None,
        current_step: Optional[int] = None,
        step_name: Optional[str] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> None:
        """Update progress for an operation"""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found for progress update")
                return
            
            operation = self.active_operations[operation_id]
            
            if operation.state not in [ProgressState.RUNNING, ProgressState.PAUSED]:
                logger.warning(f"Cannot update progress for operation in state: {operation.state}")
                return
            
            # Update progress metrics
            if progress_percentage is not None:
                operation.progress_percentage = min(100.0, max(0.0, progress_percentage))
            
            if items_processed is not None:
                operation.items_processed = items_processed
                if operation.total_items:
                    operation.progress_percentage = (items_processed / operation.total_items) * 100
            
            # Update current step
            if current_step is not None and current_step < len(operation.steps):
                # Complete previous step
                if operation.current_step_index < len(operation.steps):
                    prev_step = operation.steps[operation.current_step_index]
                    if prev_step.status == ProgressState.RUNNING:
                        prev_step.status = ProgressState.COMPLETED
                        prev_step.end_time = datetime.utcnow()
                        if prev_step.start_time:
                            prev_step.actual_duration = (prev_step.end_time - prev_step.start_time).total_seconds()
                
                # Start new step
                operation.current_step_index = current_step
                new_step = operation.steps[current_step]
                new_step.status = ProgressState.RUNNING
                new_step.start_time = datetime.utcnow()
                
                if step_name:
                    new_step.name = step_name
            
            # Update partial results
            if partial_results:
                operation.partial_results.update(partial_results)
            
            # Update timing
            operation.last_update = datetime.utcnow()
            await self._update_estimated_completion(operation)
            
            # Notify callbacks
            await self._notify_callbacks(operation, "updated", {"message": message})
            
            logger.debug(f"Updated progress for {operation.name}: {operation.progress_percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    async def complete_operation(
        self,
        operation_id: str,
        final_results: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> None:
        """Complete a progress operation"""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found for completion")
                return
            
            operation = self.active_operations[operation_id]
            operation.state = ProgressState.COMPLETED
            operation.end_time = datetime.utcnow()
            operation.progress_percentage = 100.0
            operation.final_results = final_results
            
            # Complete current step
            if operation.current_step_index < len(operation.steps):
                current_step = operation.steps[operation.current_step_index]
                current_step.status = ProgressState.COMPLETED
                current_step.end_time = datetime.utcnow()
                if current_step.start_time:
                    current_step.actual_duration = (current_step.end_time - current_step.start_time).total_seconds()
            
            # Complete all remaining steps
            for i in range(operation.current_step_index + 1, len(operation.steps)):
                operation.steps[i].status = ProgressState.COMPLETED
            
            # Move to completed operations
            self.completed_operations[operation_id] = operation
            del self.active_operations[operation_id]
            
            # Notify callbacks
            await self._notify_callbacks(operation, "completed", {"message": message})
            
            logger.info(f"Completed operation: {operation.name}")
            
        except Exception as e:
            logger.error(f"Error completing operation: {e}")
    
    async def cancel_operation(
        self,
        operation_id: str,
        reason: Optional[str] = None
    ) -> None:
        """Cancel a progress operation"""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found for cancellation")
                return
            
            operation = self.active_operations[operation_id]
            
            if not operation.can_cancel:
                raise ValueError(f"Operation {operation_id} cannot be cancelled")
            
            operation.state = ProgressState.CANCELLED
            operation.end_time = datetime.utcnow()
            
            # Cancel current step
            if operation.current_step_index < len(operation.steps):
                current_step = operation.steps[operation.current_step_index]
                current_step.status = ProgressState.CANCELLED
                current_step.end_time = datetime.utcnow()
            
            # Move to completed operations
            self.completed_operations[operation_id] = operation
            del self.active_operations[operation_id]
            
            # Notify callbacks
            await self._notify_callbacks(operation, "cancelled", {"reason": reason})
            
            logger.info(f"Cancelled operation: {operation.name}")
            
        except Exception as e:
            logger.error(f"Error cancelling operation: {e}")
    
    async def fail_operation(
        self,
        operation_id: str,
        error: Exception,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark operation as failed"""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found for failure")
                return
            
            operation = self.active_operations[operation_id]
            operation.state = ProgressState.FAILED
            operation.end_time = datetime.utcnow()
            operation.error_details = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "details": error_details or {}
            }
            
            # Fail current step
            if operation.current_step_index < len(operation.steps):
                current_step = operation.steps[operation.current_step_index]
                current_step.status = ProgressState.FAILED
                current_step.end_time = datetime.utcnow()
                current_step.error_message = str(error)
            
            # Move to completed operations
            self.completed_operations[operation_id] = operation
            del self.active_operations[operation_id]
            
            # Notify callbacks
            await self._notify_callbacks(operation, "failed", {"error": str(error)})
            
            logger.error(f"Failed operation: {operation.name} - {error}")
            
        except Exception as e:
            logger.error(f"Error failing operation: {e}")
    
    async def pause_operation(self, operation_id: str) -> None:
        """Pause a progress operation"""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found for pausing")
                return
            
            operation = self.active_operations[operation_id]
            
            if not operation.can_pause:
                raise ValueError(f"Operation {operation_id} cannot be paused")
            
            if operation.state != ProgressState.RUNNING:
                raise ValueError(f"Operation {operation_id} is not running")
            
            operation.state = ProgressState.PAUSED
            
            # Pause current step
            if operation.current_step_index < len(operation.steps):
                current_step = operation.steps[operation.current_step_index]
                current_step.status = ProgressState.PAUSED
            
            # Notify callbacks
            await self._notify_callbacks(operation, "paused")
            
            logger.info(f"Paused operation: {operation.name}")
            
        except Exception as e:
            logger.error(f"Error pausing operation: {e}")
    
    async def resume_operation(self, operation_id: str) -> None:
        """Resume a paused operation"""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found for resuming")
                return
            
            operation = self.active_operations[operation_id]
            
            if operation.state != ProgressState.PAUSED:
                raise ValueError(f"Operation {operation_id} is not paused")
            
            operation.state = ProgressState.RUNNING
            
            # Resume current step
            if operation.current_step_index < len(operation.steps):
                current_step = operation.steps[operation.current_step_index]
                current_step.status = ProgressState.RUNNING
            
            # Notify callbacks
            await self._notify_callbacks(operation, "resumed")
            
            logger.info(f"Resumed operation: {operation.name}")
            
        except Exception as e:
            logger.error(f"Error resuming operation: {e}")
    
    def get_operation(self, operation_id: str) -> Optional[ProgressOperation]:
        """Get operation details"""
        return (self.active_operations.get(operation_id) or 
                self.completed_operations.get(operation_id))
    
    def get_active_operations(self) -> List[ProgressOperation]:
        """Get all active operations"""
        return list(self.active_operations.values())
    
    def get_operation_summary(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get operation summary for UI display"""
        operation = self.get_operation(operation_id)
        if not operation:
            return None
        
        current_step_name = ""
        if operation.current_step_index < len(operation.steps):
            current_step_name = operation.steps[operation.current_step_index].name
        
        duration = None
        if operation.start_time:
            end_time = operation.end_time or datetime.utcnow()
            duration = (end_time - operation.start_time).total_seconds()
        
        return {
            "operation_id": operation.operation_id,
            "name": operation.name,
            "description": operation.description,
            "state": operation.state.value,
            "progress_percentage": operation.progress_percentage,
            "current_step": current_step_name,
            "items_processed": operation.items_processed,
            "total_items": operation.total_items,
            "estimated_completion": operation.estimated_completion.isoformat() if operation.estimated_completion else None,
            "duration": duration,
            "can_cancel": operation.can_cancel,
            "can_pause": operation.can_pause,
            "show_partial_results": operation.show_partial_results,
            "has_partial_results": bool(operation.partial_results),
            "last_update": operation.last_update.isoformat() if operation.last_update else None
        }
    
    async def add_callback(
        self,
        operation_id: str,
        callback: Callable,
        callback_type: str = "update"
    ) -> None:
        """Add callback for operation updates"""
        operation = self.get_operation(operation_id)
        if operation:
            if callback_type == "completion":
                operation.completion_callbacks.append(callback)
            else:
                operation.update_callbacks.append(callback)
    
    def add_global_callback(self, callback: Callable) -> None:
        """Add global callback for all operations"""
        self.global_callbacks.append(callback)
    
    # Private methods
    
    async def _update_estimated_completion(self, operation: ProgressOperation) -> None:
        """Update estimated completion time"""
        try:
            if not operation.start_time or operation.progress_percentage <= 0:
                return
            
            elapsed = (datetime.utcnow() - operation.start_time).total_seconds()
            
            if operation.progress_percentage > 0:
                total_estimated = elapsed * (100 / operation.progress_percentage)
                remaining = total_estimated - elapsed
                operation.estimated_completion = datetime.utcnow() + timedelta(seconds=remaining)
            
        except Exception as e:
            logger.error(f"Error updating estimated completion: {e}")
    
    async def _notify_callbacks(
        self,
        operation: ProgressOperation,
        event_type: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify all callbacks about operation updates"""
        try:
            event_data = {
                "operation_id": operation.operation_id,
                "event_type": event_type,
                "operation": operation,
                "summary": self.get_operation_summary(operation.operation_id)
            }
            
            if extra_data:
                event_data.update(extra_data)
            
            # Notify operation-specific callbacks
            callbacks = operation.update_callbacks
            if event_type in ["completed", "cancelled", "failed"]:
                callbacks.extend(operation.completion_callbacks)
            
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(f"Error in operation callback: {e}")
            
            # Notify global callbacks
            for callback in self.global_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(f"Error in global callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying callbacks: {e}")
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID"""
        self._operation_counter += 1
        timestamp = int(time.time() * 1000)
        return f"op_{timestamp}_{self._operation_counter}"
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        async def cleanup_completed_operations():
            while True:
                try:
                    # Clean up completed operations older than 1 hour
                    cutoff_time = datetime.utcnow() - timedelta(hours=1)
                    to_remove = []
                    
                    for op_id, operation in self.completed_operations.items():
                        if operation.end_time and operation.end_time < cutoff_time:
                            to_remove.append(op_id)
                    
                    for op_id in to_remove:
                        del self.completed_operations[op_id]
                    
                    if to_remove:
                        logger.info(f"Cleaned up {len(to_remove)} completed operations")
                    
                    # Sleep for 10 minutes
                    await asyncio.sleep(600)
                    
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                    await asyncio.sleep(60)  # Retry after 1 minute
        
        self._cleanup_task = asyncio.create_task(cleanup_completed_operations())


# Global instance
progress_indicator_manager = ProgressIndicatorManager()