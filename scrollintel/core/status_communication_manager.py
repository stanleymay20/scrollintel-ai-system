"""
Status Communication Manager - Transparent User Communication System

This module provides comprehensive status communication capabilities including:
- Real-time status indicators with progress tracking
- Intelligent user notification system
- Contextual help and guidance based on user actions
- Degradation explanation and alternative suggestion system

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StatusLevel(Enum):
    """System status levels"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OUTAGE = "outage"
    RECOVERING = "recovering"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageType(Enum):
    """Types of user messages"""
    STATUS_UPDATE = "status_update"
    PROGRESS_UPDATE = "progress_update"
    DEGRADATION_NOTICE = "degradation_notice"
    RECOVERY_NOTICE = "recovery_notice"
    HELP_SUGGESTION = "help_suggestion"
    ALTERNATIVE_ACTION = "alternative_action"
    MAINTENANCE_NOTICE = "maintenance_notice"


@dataclass
class SystemStatus:
    """System status information"""
    level: StatusLevel
    component: str
    message: str
    affected_features: List[str]
    estimated_resolution: Optional[datetime] = None
    alternatives: List[str] = None
    user_actions: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.alternatives is None:
            self.alternatives = []
        if self.user_actions is None:
            self.user_actions = []


@dataclass
class ProgressInfo:
    """Progress tracking information"""
    operation_id: str
    operation_name: str
    current_step: int
    total_steps: int
    step_name: str
    progress_percentage: float
    estimated_completion: Optional[datetime] = None
    can_cancel: bool = False
    partial_results: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class UserNotification:
    """User notification structure"""
    id: str
    type: MessageType
    priority: NotificationPriority
    title: str
    message: str
    actions: List[Dict[str, str]] = None
    auto_dismiss: bool = False
    dismiss_after: Optional[int] = None  # seconds
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.actions is None:
            self.actions = []


@dataclass
class ContextualHelp:
    """Contextual help and guidance"""
    context_id: str
    user_action: str
    help_text: str
    suggestions: List[str]
    related_docs: List[Dict[str, str]] = None
    video_tutorials: List[Dict[str, str]] = None
    confidence_score: float = 1.0
    
    def __post_init__(self):
        if self.related_docs is None:
            self.related_docs = []
        if self.video_tutorials is None:
            self.video_tutorials = []


class StatusCommunicationManager:
    """
    Manages transparent communication with users about system status,
    progress, and provides contextual help and guidance.
    """
    
    def __init__(self):
        self.status_subscribers: Dict[str, Set[Callable]] = {}
        self.progress_trackers: Dict[str, ProgressInfo] = {}
        self.active_notifications: Dict[str, UserNotification] = {}
        self.system_status: Dict[str, SystemStatus] = {}
        self.help_contexts: Dict[str, ContextualHelp] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self._notification_counter = 0
        
    async def update_system_status(
        self,
        component: str,
        level: StatusLevel,
        message: str,
        affected_features: List[str] = None,
        estimated_resolution: Optional[datetime] = None,
        alternatives: List[str] = None,
        user_actions: List[str] = None
    ) -> None:
        """Update system status and notify subscribers"""
        try:
            status = SystemStatus(
                level=level,
                component=component,
                message=message,
                affected_features=affected_features or [],
                estimated_resolution=estimated_resolution,
                alternatives=alternatives or [],
                user_actions=user_actions or []
            )
            
            self.system_status[component] = status
            
            # Create user notification based on status level
            await self._create_status_notification(status)
            
            # Notify subscribers
            await self._notify_status_subscribers(component, status)
            
            logger.info(f"System status updated for {component}: {level.value}")
            
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
    
    async def start_progress_tracking(
        self,
        operation_id: str,
        operation_name: str,
        total_steps: int,
        can_cancel: bool = False
    ) -> None:
        """Start tracking progress for a long-running operation"""
        try:
            progress = ProgressInfo(
                operation_id=operation_id,
                operation_name=operation_name,
                current_step=0,
                total_steps=total_steps,
                step_name="Starting...",
                progress_percentage=0.0,
                can_cancel=can_cancel
            )
            
            self.progress_trackers[operation_id] = progress
            
            # Notify users about operation start
            await self._create_progress_notification(progress, is_start=True)
            
            logger.info(f"Started progress tracking for {operation_name}")
            
        except Exception as e:
            logger.error(f"Error starting progress tracking: {e}")
    
    async def update_progress(
        self,
        operation_id: str,
        current_step: int,
        step_name: str,
        partial_results: Optional[Dict[str, Any]] = None,
        estimated_completion: Optional[datetime] = None
    ) -> None:
        """Update progress for a tracked operation"""
        try:
            if operation_id not in self.progress_trackers:
                logger.warning(f"Progress tracker not found for {operation_id}")
                return
            
            progress = self.progress_trackers[operation_id]
            progress.current_step = current_step
            progress.step_name = step_name
            progress.progress_percentage = (current_step / progress.total_steps) * 100
            progress.partial_results = partial_results
            progress.estimated_completion = estimated_completion
            progress.timestamp = datetime.utcnow()
            
            # Notify users about progress update
            await self._create_progress_notification(progress)
            
            logger.debug(f"Progress updated for {operation_id}: {progress.progress_percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    async def complete_progress(
        self,
        operation_id: str,
        final_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete progress tracking for an operation"""
        try:
            if operation_id not in self.progress_trackers:
                logger.warning(f"Progress tracker not found for {operation_id}")
                return
            
            progress = self.progress_trackers[operation_id]
            progress.current_step = progress.total_steps
            progress.step_name = "Completed"
            progress.progress_percentage = 100.0
            progress.partial_results = final_results
            progress.timestamp = datetime.utcnow()
            
            # Notify users about completion
            await self._create_progress_notification(progress, is_complete=True)
            
            # Clean up tracker after a delay
            asyncio.create_task(self._cleanup_progress_tracker(operation_id, delay=30))
            
            logger.info(f"Progress completed for {operation_id}")
            
        except Exception as e:
            logger.error(f"Error completing progress: {e}")
    
    async def notify_degradation(
        self,
        component: str,
        degradation_level: str,
        affected_functionality: List[str],
        alternatives: List[str],
        explanation: str,
        estimated_recovery: Optional[datetime] = None
    ) -> None:
        """Notify users about system degradation with alternatives"""
        try:
            notification = UserNotification(
                id=self._generate_notification_id(),
                type=MessageType.DEGRADATION_NOTICE,
                priority=NotificationPriority.HIGH,
                title=f"{component} Service Degraded",
                message=explanation,
                actions=[
                    {"type": "view_alternatives", "label": "View Alternatives"},
                    {"type": "dismiss", "label": "Dismiss"}
                ],
                context={
                    "component": component,
                    "degradation_level": degradation_level,
                    "affected_functionality": affected_functionality,
                    "alternatives": alternatives,
                    "estimated_recovery": estimated_recovery.isoformat() if estimated_recovery else None
                }
            )
            
            await self._send_notification(notification)
            
            logger.info(f"Degradation notification sent for {component}")
            
        except Exception as e:
            logger.error(f"Error sending degradation notification: {e}")
    
    async def provide_contextual_help(
        self,
        user_id: str,
        user_action: str,
        context: Dict[str, Any]
    ) -> Optional[ContextualHelp]:
        """Provide contextual help based on user actions"""
        try:
            context_id = f"{user_action}_{hash(str(context))}"
            
            # Check if we have cached help for this context
            if context_id in self.help_contexts:
                return self.help_contexts[context_id]
            
            # Generate contextual help
            help_info = await self._generate_contextual_help(user_action, context)
            
            if help_info:
                self.help_contexts[context_id] = help_info
                
                # Send help notification if confidence is high
                if help_info.confidence_score > 0.7:
                    await self._send_help_notification(user_id, help_info)
            
            return help_info
            
        except Exception as e:
            logger.error(f"Error providing contextual help: {e}")
            return None
    
    async def suggest_alternatives(
        self,
        user_id: str,
        blocked_action: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest alternative actions when primary action is blocked"""
        try:
            alternatives = await self._generate_alternatives(blocked_action, context)
            
            if alternatives:
                notification = UserNotification(
                    id=self._generate_notification_id(),
                    type=MessageType.ALTERNATIVE_ACTION,
                    priority=NotificationPriority.MEDIUM,
                    title="Alternative Actions Available",
                    message=f"The action '{blocked_action}' is currently unavailable. Here are some alternatives:",
                    actions=[
                        {"type": "view_alternatives", "label": "View Alternatives"},
                        {"type": "dismiss", "label": "Not Now"}
                    ],
                    context={
                        "blocked_action": blocked_action,
                        "alternatives": alternatives,
                        "user_context": context
                    }
                )
                
                await self._send_notification_to_user(user_id, notification)
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Error suggesting alternatives: {e}")
            return []
    
    async def subscribe_to_status(
        self,
        component: str,
        callback: Callable[[SystemStatus], None]
    ) -> None:
        """Subscribe to status updates for a component"""
        if component not in self.status_subscribers:
            self.status_subscribers[component] = set()
        
        self.status_subscribers[component].add(callback)
    
    async def unsubscribe_from_status(
        self,
        component: str,
        callback: Callable[[SystemStatus], None]
    ) -> None:
        """Unsubscribe from status updates"""
        if component in self.status_subscribers:
            self.status_subscribers[component].discard(callback)
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system status overview"""
        try:
            overall_status = StatusLevel.OPERATIONAL
            degraded_components = []
            active_operations = []
            
            # Determine overall status
            for component, status in self.system_status.items():
                if status.level in [StatusLevel.OUTAGE, StatusLevel.MAINTENANCE]:
                    overall_status = StatusLevel.OUTAGE
                elif status.level == StatusLevel.DEGRADED and overall_status == StatusLevel.OPERATIONAL:
                    overall_status = StatusLevel.DEGRADED
                    degraded_components.append(component)
            
            # Get active operations
            for op_id, progress in self.progress_trackers.items():
                if progress.progress_percentage < 100:
                    active_operations.append({
                        "id": op_id,
                        "name": progress.operation_name,
                        "progress": progress.progress_percentage,
                        "step": progress.step_name
                    })
            
            return {
                "overall_status": overall_status.value,
                "degraded_components": degraded_components,
                "active_operations": active_operations,
                "active_notifications": len(self.active_notifications),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
            return {"error": "Unable to retrieve system overview"}
    
    # Private methods
    
    async def _create_status_notification(self, status: SystemStatus) -> None:
        """Create notification based on system status"""
        priority_map = {
            StatusLevel.OPERATIONAL: NotificationPriority.LOW,
            StatusLevel.DEGRADED: NotificationPriority.MEDIUM,
            StatusLevel.MAINTENANCE: NotificationPriority.MEDIUM,
            StatusLevel.OUTAGE: NotificationPriority.CRITICAL,
            StatusLevel.RECOVERING: NotificationPriority.MEDIUM
        }
        
        notification = UserNotification(
            id=self._generate_notification_id(),
            type=MessageType.STATUS_UPDATE,
            priority=priority_map.get(status.level, NotificationPriority.MEDIUM),
            title=f"{status.component} Status Update",
            message=status.message,
            actions=self._generate_status_actions(status),
            context=asdict(status)
        )
        
        await self._send_notification(notification)
    
    async def _create_progress_notification(
        self,
        progress: ProgressInfo,
        is_start: bool = False,
        is_complete: bool = False
    ) -> None:
        """Create progress update notification"""
        if is_start:
            title = f"Started: {progress.operation_name}"
            message = f"Operation started with {progress.total_steps} steps"
        elif is_complete:
            title = f"Completed: {progress.operation_name}"
            message = "Operation completed successfully"
        else:
            title = f"Progress: {progress.operation_name}"
            message = f"{progress.step_name} ({progress.progress_percentage:.1f}%)"
        
        actions = []
        if progress.can_cancel and not is_complete:
            actions.append({"type": "cancel", "label": "Cancel Operation"})
        if progress.partial_results:
            actions.append({"type": "view_results", "label": "View Partial Results"})
        
        notification = UserNotification(
            id=self._generate_notification_id(),
            type=MessageType.PROGRESS_UPDATE,
            priority=NotificationPriority.LOW,
            title=title,
            message=message,
            actions=actions,
            auto_dismiss=is_complete,
            dismiss_after=5 if is_complete else None,
            context=asdict(progress)
        )
        
        await self._send_notification(notification)
    
    async def _generate_contextual_help(
        self,
        user_action: str,
        context: Dict[str, Any]
    ) -> Optional[ContextualHelp]:
        """Generate contextual help based on user action and context"""
        # This would typically use ML models or rule-based systems
        # For now, implementing basic rule-based help
        
        help_map = {
            "upload_file": {
                "help_text": "Upload files by dragging and dropping or clicking the upload button. Supported formats include CSV, JSON, Excel, and more.",
                "suggestions": [
                    "Ensure your file is under 100MB",
                    "Check that column headers are in the first row",
                    "Remove any special characters from column names"
                ],
                "related_docs": [
                    {"title": "File Upload Guide", "url": "/docs/file-upload"},
                    {"title": "Supported Formats", "url": "/docs/formats"}
                ]
            },
            "create_visualization": {
                "help_text": "Create visualizations by selecting your data source and choosing a chart type. The system will suggest the best visualization based on your data.",
                "suggestions": [
                    "Start with a simple chart type like bar or line",
                    "Ensure your data has clear categories or time series",
                    "Use filters to focus on specific data subsets"
                ],
                "related_docs": [
                    {"title": "Visualization Guide", "url": "/docs/visualizations"},
                    {"title": "Chart Types", "url": "/docs/chart-types"}
                ]
            },
            "run_analysis": {
                "help_text": "Run analysis by selecting your dataset and choosing an analysis type. The system will guide you through the process.",
                "suggestions": [
                    "Clean your data before running analysis",
                    "Start with descriptive statistics",
                    "Consider the size of your dataset for complex analyses"
                ],
                "related_docs": [
                    {"title": "Analysis Guide", "url": "/docs/analysis"},
                    {"title": "Data Preparation", "url": "/docs/data-prep"}
                ]
            }
        }
        
        if user_action in help_map:
            help_data = help_map[user_action]
            return ContextualHelp(
                context_id=f"{user_action}_{hash(str(context))}",
                user_action=user_action,
                help_text=help_data["help_text"],
                suggestions=help_data["suggestions"],
                related_docs=help_data.get("related_docs", []),
                confidence_score=0.8
            )
        
        return None
    
    async def _generate_alternatives(
        self,
        blocked_action: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate alternative actions when primary action is blocked"""
        alternatives_map = {
            "upload_large_file": [
                {
                    "action": "split_file",
                    "title": "Split File",
                    "description": "Split your large file into smaller chunks",
                    "difficulty": "easy"
                },
                {
                    "action": "use_cloud_storage",
                    "title": "Use Cloud Storage",
                    "description": "Upload to cloud storage and import via URL",
                    "difficulty": "medium"
                }
            ],
            "create_complex_visualization": [
                {
                    "action": "create_simple_chart",
                    "title": "Create Simple Chart",
                    "description": "Start with a basic chart and add complexity later",
                    "difficulty": "easy"
                },
                {
                    "action": "use_template",
                    "title": "Use Template",
                    "description": "Choose from pre-built visualization templates",
                    "difficulty": "easy"
                }
            ],
            "run_heavy_analysis": [
                {
                    "action": "sample_data",
                    "title": "Use Data Sample",
                    "description": "Run analysis on a representative sample",
                    "difficulty": "easy"
                },
                {
                    "action": "schedule_analysis",
                    "title": "Schedule Analysis",
                    "description": "Schedule the analysis to run during off-peak hours",
                    "difficulty": "medium"
                }
            ]
        }
        
        return alternatives_map.get(blocked_action, [])
    
    def _generate_status_actions(self, status: SystemStatus) -> List[Dict[str, str]]:
        """Generate appropriate actions based on system status"""
        actions = []
        
        if status.alternatives:
            actions.append({"type": "view_alternatives", "label": "View Alternatives"})
        
        if status.user_actions:
            actions.append({"type": "view_actions", "label": "What Can I Do?"})
        
        if status.level == StatusLevel.DEGRADED:
            actions.append({"type": "check_status", "label": "Check Status"})
        
        actions.append({"type": "dismiss", "label": "Dismiss"})
        
        return actions
    
    async def _notify_status_subscribers(self, component: str, status: SystemStatus) -> None:
        """Notify all subscribers about status changes"""
        if component in self.status_subscribers:
            for callback in self.status_subscribers[component]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(status)
                    else:
                        callback(status)
                except Exception as e:
                    logger.error(f"Error notifying status subscriber: {e}")
    
    async def _send_notification(self, notification: UserNotification) -> None:
        """Send notification to all users (broadcast)"""
        self.active_notifications[notification.id] = notification
        
        # Here you would integrate with your WebSocket or real-time communication system
        logger.info(f"Broadcasting notification: {notification.title}")
        
        # Auto-dismiss if configured
        if notification.auto_dismiss and notification.dismiss_after:
            asyncio.create_task(
                self._auto_dismiss_notification(notification.id, notification.dismiss_after)
            )
    
    async def _send_notification_to_user(self, user_id: str, notification: UserNotification) -> None:
        """Send notification to specific user"""
        self.active_notifications[notification.id] = notification
        
        # Here you would send to specific user via WebSocket or similar
        logger.info(f"Sending notification to user {user_id}: {notification.title}")
    
    async def _send_help_notification(self, user_id: str, help_info: ContextualHelp) -> None:
        """Send contextual help as notification"""
        notification = UserNotification(
            id=self._generate_notification_id(),
            type=MessageType.HELP_SUGGESTION,
            priority=NotificationPriority.LOW,
            title="Helpful Suggestion",
            message=help_info.help_text,
            actions=[
                {"type": "view_help", "label": "View Full Help"},
                {"type": "dismiss", "label": "Dismiss"}
            ],
            auto_dismiss=True,
            dismiss_after=10,
            context=asdict(help_info)
        )
        
        await self._send_notification_to_user(user_id, notification)
    
    async def _cleanup_progress_tracker(self, operation_id: str, delay: int = 30) -> None:
        """Clean up progress tracker after delay"""
        await asyncio.sleep(delay)
        self.progress_trackers.pop(operation_id, None)
    
    async def _auto_dismiss_notification(self, notification_id: str, delay: int) -> None:
        """Auto-dismiss notification after delay"""
        await asyncio.sleep(delay)
        self.active_notifications.pop(notification_id, None)
    
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        self._notification_counter += 1
        return f"notif_{int(datetime.utcnow().timestamp())}_{self._notification_counter}"


# Global instance
status_communication_manager = StatusCommunicationManager()