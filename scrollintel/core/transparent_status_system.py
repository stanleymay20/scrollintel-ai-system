"""
Transparent Status System - Unified Status Communication

This module integrates all status communication components into a unified system:
- Status Communication Manager
- Progress Indicator Manager  
- Intelligent Notification System
- Contextual Help System

Provides a single interface for transparent user communication.

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import json

from .status_communication_manager import (
    StatusCommunicationManager, SystemStatus, StatusLevel, 
    status_communication_manager
)
from .progress_indicator_manager import (
    ProgressIndicatorManager, ProgressType, ProgressState,
    progress_indicator_manager
)
from .intelligent_notification_system import (
    IntelligentNotificationSystem, NotificationChannel, NotificationPriority,
    UserActivityState, intelligent_notification_system
)
from .contextual_help_system import (
    ContextualHelpSystem, HelpTrigger, UserExpertiseLevel,
    contextual_help_system
)

logger = logging.getLogger(__name__)


class CommunicationEvent(Enum):
    """Types of communication events"""
    STATUS_CHANGE = "status_change"
    PROGRESS_UPDATE = "progress_update"
    OPERATION_START = "operation_start"
    OPERATION_COMPLETE = "operation_complete"
    ERROR_OCCURRED = "error_occurred"
    HELP_REQUESTED = "help_requested"
    USER_STUCK = "user_stuck"
    SYSTEM_MAINTENANCE = "system_maintenance"


@dataclass
class UnifiedStatusUpdate:
    """Unified status update combining all communication types"""
    event_type: CommunicationEvent
    user_id: Optional[str]
    component: str
    title: str
    message: str
    priority: str = "medium"
    data: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, str]]] = None
    progress_info: Optional[Dict[str, Any]] = None
    help_suggestions: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class TransparentStatusSystem:
    """
    Unified transparent status communication system that coordinates
    all status, progress, notification, and help components
    """
    
    def __init__(self):
        self.status_manager = status_communication_manager
        self.progress_manager = progress_indicator_manager
        self.notification_system = intelligent_notification_system
        self.help_system = contextual_help_system
        
        # Event handlers and subscribers
        self.event_handlers: Dict[CommunicationEvent, List[Callable]] = {}
        self.user_subscribers: Dict[str, List[Callable]] = {}
        self.global_subscribers: List[Callable] = []
        
        # Integration state
        self.active_communications: Dict[str, UnifiedStatusUpdate] = {}
        self.user_communication_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Initialize integrations
        self._setup_integrations()
    
    async def communicate_status_change(
        self,
        component: str,
        status_level: StatusLevel,
        message: str,
        affected_users: Optional[List[str]] = None,
        affected_features: Optional[List[str]] = None,
        alternatives: Optional[List[str]] = None,
        estimated_resolution: Optional[datetime] = None
    ) -> str:
        """Communicate system status changes to users"""
        try:
            # Update system status
            await self.status_manager.update_system_status(
                component=component,
                level=status_level,
                message=message,
                affected_features=affected_features or [],
                estimated_resolution=estimated_resolution,
                alternatives=alternatives or []
            )
            
            # Determine priority based on status level
            priority_map = {
                StatusLevel.OPERATIONAL: "low",
                StatusLevel.DEGRADED: "medium", 
                StatusLevel.MAINTENANCE: "medium",
                StatusLevel.OUTAGE: "critical",
                StatusLevel.RECOVERING: "medium"
            }
            priority = priority_map.get(status_level, "medium")
            
            # Create unified status update
            status_update = UnifiedStatusUpdate(
                event_type=CommunicationEvent.STATUS_CHANGE,
                user_id=None,  # Broadcast to all users
                component=component,
                title=f"{component} Status Update",
                message=message,
                priority=priority,
                data={
                    "status_level": status_level.value,
                    "affected_features": affected_features or [],
                    "alternatives": alternatives or [],
                    "estimated_resolution": estimated_resolution.isoformat() if estimated_resolution else None
                }
            )
            
            # Send notifications to affected users
            if affected_users:
                for user_id in affected_users:
                    await self._send_user_communication(user_id, status_update)
            else:
                # Broadcast to all users
                await self._broadcast_communication(status_update)
            
            # Provide contextual help if status is degraded or outage
            if status_level in [StatusLevel.DEGRADED, StatusLevel.OUTAGE]:
                await self._provide_status_help(component, status_level, affected_users)
            
            communication_id = f"status_{component}_{int(datetime.utcnow().timestamp())}"
            self.active_communications[communication_id] = status_update
            
            return communication_id
            
        except Exception as e:
            logger.error(f"Error communicating status change: {e}")
            raise
    
    async def start_operation_tracking(
        self,
        user_id: str,
        operation_name: str,
        description: str,
        total_steps: Optional[int] = None,
        estimated_duration: Optional[int] = None,
        can_cancel: bool = False,
        show_progress: bool = True
    ) -> str:
        """Start tracking a user operation with progress"""
        try:
            # Create progress operation
            operation_id = await self.progress_manager.create_operation(
                name=operation_name,
                description=description,
                progress_type=ProgressType.DETERMINATE if total_steps else ProgressType.INDETERMINATE,
                total_items=total_steps,
                can_cancel=can_cancel,
                show_partial_results=True
            )
            
            # Start the operation
            await self.progress_manager.start_operation(operation_id)
            
            # Create unified status update
            status_update = UnifiedStatusUpdate(
                event_type=CommunicationEvent.OPERATION_START,
                user_id=user_id,
                component="operation_tracker",
                title=f"Started: {operation_name}",
                message=description,
                priority="low",
                progress_info={
                    "operation_id": operation_id,
                    "total_steps": total_steps,
                    "estimated_duration": estimated_duration,
                    "can_cancel": can_cancel
                }
            )
            
            # Send to user
            if show_progress:
                await self._send_user_communication(user_id, status_update)
            
            return operation_id
            
        except Exception as e:
            logger.error(f"Error starting operation tracking: {e}")
            raise
    
    async def update_operation_progress(
        self,
        operation_id: str,
        user_id: str,
        progress_percentage: Optional[float] = None,
        current_step: Optional[int] = None,
        step_name: Optional[str] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> None:
        """Update operation progress"""
        try:
            # Update progress
            await self.progress_manager.update_progress(
                operation_id=operation_id,
                progress_percentage=progress_percentage,
                current_step=current_step,
                step_name=step_name,
                partial_results=partial_results,
                message=message
            )
            
            # Get operation summary
            operation_summary = self.progress_manager.get_operation_summary(operation_id)
            
            if operation_summary:
                # Create unified status update
                status_update = UnifiedStatusUpdate(
                    event_type=CommunicationEvent.PROGRESS_UPDATE,
                    user_id=user_id,
                    component="operation_tracker",
                    title=f"Progress: {operation_summary['name']}",
                    message=message or f"{step_name or 'Processing'} ({progress_percentage or 0:.1f}%)",
                    priority="low",
                    progress_info=operation_summary
                )
                
                # Send to user
                await self._send_user_communication(user_id, status_update)
            
        except Exception as e:
            logger.error(f"Error updating operation progress: {e}")
    
    async def complete_operation(
        self,
        operation_id: str,
        user_id: str,
        success: bool = True,
        final_results: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> None:
        """Complete operation tracking"""
        try:
            if success:
                await self.progress_manager.complete_operation(
                    operation_id=operation_id,
                    final_results=final_results,
                    message=message
                )
            else:
                error = Exception(message or "Operation failed")
                await self.progress_manager.fail_operation(
                    operation_id=operation_id,
                    error=error
                )
            
            # Get operation summary
            operation_summary = self.progress_manager.get_operation_summary(operation_id)
            
            if operation_summary:
                # Create unified status update
                status_update = UnifiedStatusUpdate(
                    event_type=CommunicationEvent.OPERATION_COMPLETE,
                    user_id=user_id,
                    component="operation_tracker",
                    title=f"{'Completed' if success else 'Failed'}: {operation_summary['name']}",
                    message=message or f"Operation {'completed successfully' if success else 'failed'}",
                    priority="medium" if success else "high",
                    data={"success": success, "final_results": final_results},
                    progress_info=operation_summary
                )
                
                # Send to user
                await self._send_user_communication(user_id, status_update)
                
                # Provide help if operation failed
                if not success:
                    await self._provide_operation_help(user_id, operation_id, message)
            
        except Exception as e:
            logger.error(f"Error completing operation: {e}")
    
    async def handle_error_with_help(
        self,
        user_id: str,
        error: Exception,
        context: Dict[str, Any],
        component: str = "system"
    ) -> List[Dict[str, Any]]:
        """Handle error with integrated help and recovery suggestions"""
        try:
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "context": context
            }
            
            # Get contextual help for error
            help_suggestions = await self.help_system.handle_error_help(
                user_id=user_id,
                error=error_info,
                context=context
            )
            
            # Create unified status update
            status_update = UnifiedStatusUpdate(
                event_type=CommunicationEvent.ERROR_OCCURRED,
                user_id=user_id,
                component=component,
                title="Error Occurred",
                message=str(error),
                priority="high",
                data=error_info,
                help_suggestions=[
                    {
                        "id": suggestion.suggestion_id,
                        "title": suggestion.content.title,
                        "content": suggestion.content.content,
                        "relevance": suggestion.relevance_score
                    }
                    for suggestion in help_suggestions
                ],
                actions=[
                    {"type": "view_help", "label": "Get Help"},
                    {"type": "retry", "label": "Try Again"},
                    {"type": "dismiss", "label": "Dismiss"}
                ]
            )
            
            # Send to user
            await self._send_user_communication(user_id, status_update)
            
            # Send notification
            await self.notification_system.send_notification(
                user_id=user_id,
                notification_type="error",
                title="Error Occurred",
                message=str(error),
                priority="high",
                data=error_info,
                actions=[
                    {"type": "view_help", "label": "Get Help"},
                    {"type": "retry", "label": "Try Again"}
                ]
            )
            
            return [
                {
                    "id": suggestion.suggestion_id,
                    "title": suggestion.content.title,
                    "content": suggestion.content.content,
                    "relevance": suggestion.relevance_score
                }
                for suggestion in help_suggestions
            ]
            
        except Exception as e:
            logger.error(f"Error handling error with help: {e}")
            return []
    
    async def provide_proactive_help(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Provide proactive help based on user behavior"""
        try:
            # Get proactive help suggestion
            help_suggestion = await self.help_system.provide_proactive_help(
                user_id=user_id,
                current_context=context
            )
            
            if help_suggestion:
                # Create unified status update
                status_update = UnifiedStatusUpdate(
                    event_type=CommunicationEvent.HELP_REQUESTED,
                    user_id=user_id,
                    component="help_system",
                    title="Helpful Suggestion",
                    message=help_suggestion.content.title,
                    priority="low",
                    help_suggestions=[{
                        "id": help_suggestion.suggestion_id,
                        "title": help_suggestion.content.title,
                        "content": help_suggestion.content.content,
                        "relevance": help_suggestion.relevance_score
                    }],
                    actions=[
                        {"type": "view_help", "label": "View Help"},
                        {"type": "dismiss", "label": "Not Now"}
                    ]
                )
                
                # Send to user (low priority, non-intrusive)
                await self._send_user_communication(user_id, status_update)
                
                return {
                    "id": help_suggestion.suggestion_id,
                    "title": help_suggestion.content.title,
                    "content": help_suggestion.content.content,
                    "relevance": help_suggestion.relevance_score
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error providing proactive help: {e}")
            return None
    
    async def update_user_activity(
        self,
        user_id: str,
        activity_state: UserActivityState,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update user activity state across all systems"""
        try:
            # Update notification system
            await self.notification_system.update_user_activity(user_id, activity_state)
            
            # Update help system context if provided
            if context:
                await self.help_system._update_user_context(user_id, context)
            
            # Check for proactive help if user becomes active
            if activity_state == UserActivityState.ACTIVE and context:
                await self.provide_proactive_help(user_id, context)
                
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
    
    async def set_user_communication_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> None:
        """Set user communication preferences across all systems"""
        try:
            self.user_communication_preferences[user_id] = preferences
            
            # Update notification rules if provided
            if "notification_rules" in preferences:
                await self.notification_system.set_user_notification_rules(
                    user_id, preferences["notification_rules"]
                )
            
            # Update help system preferences
            if "expertise_level" in preferences:
                expertise_level = UserExpertiseLevel(preferences["expertise_level"])
                await self.help_system.update_user_expertise(user_id, expertise_level)
            
        except Exception as e:
            logger.error(f"Error setting user communication preferences: {e}")
    
    async def get_user_status_overview(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive status overview for user"""
        try:
            # Get system overview
            system_overview = await self.status_manager.get_system_overview()
            
            # Get user's active operations
            active_operations = []
            for operation in self.progress_manager.get_active_operations():
                if operation.context.get("user_id") == user_id:
                    active_operations.append(
                        self.progress_manager.get_operation_summary(operation.operation_id)
                    )
            
            # Get notification stats
            notification_stats = await self.notification_system.get_user_notification_stats(user_id)
            
            # Get help analytics
            help_analytics = await self.help_system.get_help_analytics()
            
            return {
                "user_id": user_id,
                "system_status": system_overview,
                "active_operations": active_operations,
                "notification_stats": notification_stats,
                "help_available": len(self.help_system.help_content),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user status overview: {e}")
            return {"error": "Unable to retrieve status overview"}
    
    async def subscribe_to_user_communications(
        self,
        user_id: str,
        callback: Callable[[UnifiedStatusUpdate], None]
    ) -> None:
        """Subscribe to communications for specific user"""
        if user_id not in self.user_subscribers:
            self.user_subscribers[user_id] = []
        self.user_subscribers[user_id].append(callback)
    
    async def subscribe_to_all_communications(
        self,
        callback: Callable[[UnifiedStatusUpdate], None]
    ) -> None:
        """Subscribe to all communications (global)"""
        self.global_subscribers.append(callback)
    
    # Private methods
    
    def _setup_integrations(self) -> None:
        """Setup integrations between components"""
        # Add progress callbacks to create notifications
        self.progress_manager.add_global_callback(self._handle_progress_event)
        
        # Setup status manager callbacks
        # (In a real implementation, you'd add callback support to status manager)
    
    async def _handle_progress_event(self, event_data: Dict[str, Any]) -> None:
        """Handle progress events from progress manager"""
        try:
            event_type = event_data.get("event_type")
            operation = event_data.get("operation")
            
            if not operation:
                return
            
            # Extract user ID from operation context
            user_id = operation.context.get("user_id")
            if not user_id:
                return
            
            # Create appropriate notifications based on event type
            if event_type == "completed":
                await self.notification_system.send_notification(
                    user_id=user_id,
                    notification_type="operation_complete",
                    title=f"Completed: {operation.name}",
                    message=f"Your operation '{operation.name}' has completed successfully.",
                    priority="medium",
                    data={"operation_id": operation.operation_id}
                )
            elif event_type == "failed":
                await self.notification_system.send_notification(
                    user_id=user_id,
                    notification_type="operation_failed",
                    title=f"Failed: {operation.name}",
                    message=f"Your operation '{operation.name}' has failed. Click for help.",
                    priority="high",
                    data={"operation_id": operation.operation_id},
                    actions=[
                        {"type": "get_help", "label": "Get Help"},
                        {"type": "retry", "label": "Retry"}
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error handling progress event: {e}")
    
    async def _send_user_communication(
        self,
        user_id: str,
        status_update: UnifiedStatusUpdate
    ) -> None:
        """Send communication to specific user"""
        try:
            # Send through notification system
            await self.notification_system.send_notification(
                user_id=user_id,
                notification_type=status_update.event_type.value,
                title=status_update.title,
                message=status_update.message,
                priority=status_update.priority,
                data=status_update.data,
                actions=status_update.actions
            )
            
            # Notify user subscribers
            if user_id in self.user_subscribers:
                for callback in self.user_subscribers[user_id]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(status_update)
                        else:
                            callback(status_update)
                    except Exception as e:
                        logger.error(f"Error in user subscriber callback: {e}")
            
            # Notify global subscribers
            await self._notify_global_subscribers(status_update)
            
        except Exception as e:
            logger.error(f"Error sending user communication: {e}")
    
    async def _broadcast_communication(self, status_update: UnifiedStatusUpdate) -> None:
        """Broadcast communication to all users"""
        try:
            # This would typically integrate with your WebSocket or real-time system
            # For now, just notify global subscribers
            await self._notify_global_subscribers(status_update)
            
        except Exception as e:
            logger.error(f"Error broadcasting communication: {e}")
    
    async def _notify_global_subscribers(self, status_update: UnifiedStatusUpdate) -> None:
        """Notify global subscribers"""
        for callback in self.global_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status_update)
                else:
                    callback(status_update)
            except Exception as e:
                logger.error(f"Error in global subscriber callback: {e}")
    
    async def _provide_status_help(
        self,
        component: str,
        status_level: StatusLevel,
        affected_users: Optional[List[str]]
    ) -> None:
        """Provide contextual help for status issues"""
        try:
            context = {
                "component": component,
                "status_level": status_level.value,
                "page": "system_status"
            }
            
            # If specific users affected, provide targeted help
            if affected_users:
                for user_id in affected_users:
                    help_suggestions = await self.help_system.get_contextual_help(
                        user_id=user_id,
                        trigger=HelpTrigger.PROACTIVE,
                        context=context
                    )
                    
                    if help_suggestions:
                        await self.notification_system.send_notification(
                            user_id=user_id,
                            notification_type="help_available",
                            title="Help Available",
                            message=f"We've found some helpful resources for the {component} issue.",
                            priority="low",
                            actions=[{"type": "view_help", "label": "View Help"}]
                        )
            
        except Exception as e:
            logger.error(f"Error providing status help: {e}")
    
    async def _provide_operation_help(
        self,
        user_id: str,
        operation_id: str,
        error_message: Optional[str]
    ) -> None:
        """Provide help for failed operations"""
        try:
            context = {
                "operation_id": operation_id,
                "error_message": error_message,
                "page": "operation_failed"
            }
            
            help_suggestions = await self.help_system.get_contextual_help(
                user_id=user_id,
                trigger=HelpTrigger.ERROR_OCCURRED,
                context=context
            )
            
            if help_suggestions:
                await self.notification_system.send_notification(
                    user_id=user_id,
                    notification_type="operation_help",
                    title="Operation Failed - Help Available",
                    message="We can help you resolve this issue and retry your operation.",
                    priority="medium",
                    actions=[
                        {"type": "view_help", "label": "Get Help"},
                        {"type": "retry", "label": "Retry Operation"}
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error providing operation help: {e}")


# Global instance
transparent_status_system = TransparentStatusSystem()