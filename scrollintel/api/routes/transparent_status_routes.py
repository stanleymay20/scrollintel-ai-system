"""
Transparent Status API Routes

API endpoints for the transparent status communication system.
Provides REST endpoints for status updates, progress tracking,
notifications, and contextual help.

Requirements: 6.1, 6.2, 6.3, 6.5
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import asyncio
import logging

from ...core.transparent_status_system import (
    transparent_status_system, CommunicationEvent, UnifiedStatusUpdate
)
from ...core.status_communication_manager import StatusLevel
from ...core.progress_indicator_manager import ProgressType
from ...core.intelligent_notification_system import (
    NotificationChannel, NotificationPriority, UserActivityState
)
from ...core.contextual_help_system import HelpTrigger, UserExpertiseLevel, HelpFormat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/status", tags=["transparent-status"])


# Pydantic models for request/response

class StatusUpdateRequest(BaseModel):
    component: str
    status_level: str = Field(..., regex="^(operational|degraded|maintenance|outage|recovering)$")
    message: str
    affected_users: Optional[List[str]] = None
    affected_features: Optional[List[str]] = None
    alternatives: Optional[List[str]] = None
    estimated_resolution: Optional[datetime] = None


class OperationStartRequest(BaseModel):
    operation_name: str
    description: str
    total_steps: Optional[int] = None
    estimated_duration: Optional[int] = None  # seconds
    can_cancel: bool = False
    show_progress: bool = True


class ProgressUpdateRequest(BaseModel):
    progress_percentage: Optional[float] = None
    current_step: Optional[int] = None
    step_name: Optional[str] = None
    partial_results: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class OperationCompleteRequest(BaseModel):
    success: bool = True
    final_results: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class NotificationRequest(BaseModel):
    notification_type: str
    title: str
    message: str
    priority: str = Field("medium", regex="^(low|medium|high|critical)$")
    data: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, str]]] = None
    channels: Optional[List[str]] = None


class UserPreferencesRequest(BaseModel):
    notification_rules: Optional[List[Dict[str, Any]]] = None
    expertise_level: Optional[str] = Field(None, regex="^(beginner|intermediate|advanced|expert)$")
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    preferred_channels: Optional[List[str]] = None


class HelpContentRequest(BaseModel):
    title: str
    content: str
    format: str = Field("text", regex="^(text|html|markdown|video|interactive|step_by_step)$")
    expertise_level: str = Field("beginner", regex="^(beginner|intermediate|advanced|expert)$")
    tags: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None
    interactive_elements: Optional[List[Dict[str, Any]]] = None


class UserActivityRequest(BaseModel):
    activity_state: str = Field(..., regex="^(active|idle|away|do_not_disturb|offline)$")
    context: Optional[Dict[str, Any]] = None


# Status Management Endpoints

@router.post("/system/status")
async def update_system_status(
    request: StatusUpdateRequest,
    background_tasks: BackgroundTasks
):
    """Update system status and notify users"""
    try:
        status_level = StatusLevel(request.status_level)
        
        communication_id = await transparent_status_system.communicate_status_change(
            component=request.component,
            status_level=status_level,
            message=request.message,
            affected_users=request.affected_users,
            affected_features=request.affected_features,
            alternatives=request.alternatives,
            estimated_resolution=request.estimated_resolution
        )
        
        return {
            "success": True,
            "communication_id": communication_id,
            "message": "Status update sent successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/overview")
async def get_system_overview():
    """Get comprehensive system status overview"""
    try:
        overview = await transparent_status_system.status_manager.get_system_overview()
        return overview
        
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Progress Tracking Endpoints

@router.post("/users/{user_id}/operations")
async def start_operation(
    user_id: str,
    request: OperationStartRequest
):
    """Start tracking a user operation"""
    try:
        operation_id = await transparent_status_system.start_operation_tracking(
            user_id=user_id,
            operation_name=request.operation_name,
            description=request.description,
            total_steps=request.total_steps,
            estimated_duration=request.estimated_duration,
            can_cancel=request.can_cancel,
            show_progress=request.show_progress
        )
        
        return {
            "success": True,
            "operation_id": operation_id,
            "message": "Operation tracking started"
        }
        
    except Exception as e:
        logger.error(f"Error starting operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/{user_id}/operations/{operation_id}/progress")
async def update_operation_progress(
    user_id: str,
    operation_id: str,
    request: ProgressUpdateRequest
):
    """Update operation progress"""
    try:
        await transparent_status_system.update_operation_progress(
            operation_id=operation_id,
            user_id=user_id,
            progress_percentage=request.progress_percentage,
            current_step=request.current_step,
            step_name=request.step_name,
            partial_results=request.partial_results,
            message=request.message
        )
        
        return {
            "success": True,
            "message": "Progress updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/operations/{operation_id}/complete")
async def complete_operation(
    user_id: str,
    operation_id: str,
    request: OperationCompleteRequest
):
    """Complete operation tracking"""
    try:
        await transparent_status_system.complete_operation(
            operation_id=operation_id,
            user_id=user_id,
            success=request.success,
            final_results=request.final_results,
            message=request.message
        )
        
        return {
            "success": True,
            "message": "Operation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error completing operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/operations")
async def get_user_operations(user_id: str):
    """Get user's active operations"""
    try:
        active_operations = []
        for operation in transparent_status_system.progress_manager.get_active_operations():
            if operation.context.get("user_id") == user_id:
                summary = transparent_status_system.progress_manager.get_operation_summary(
                    operation.operation_id
                )
                if summary:
                    active_operations.append(summary)
        
        return {
            "user_id": user_id,
            "active_operations": active_operations,
            "total_active": len(active_operations)
        }
        
    except Exception as e:
        logger.error(f"Error getting user operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Notification Endpoints

@router.post("/users/{user_id}/notifications")
async def send_notification(
    user_id: str,
    request: NotificationRequest
):
    """Send notification to user"""
    try:
        channels = None
        if request.channels:
            channels = [NotificationChannel(ch) for ch in request.channels]
        
        result = await transparent_status_system.notification_system.send_notification(
            user_id=user_id,
            notification_type=request.notification_type,
            title=request.title,
            message=request.message,
            priority=request.priority,
            data=request.data,
            actions=request.actions,
            channels=channels
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/notifications/stats")
async def get_notification_stats(user_id: str):
    """Get user notification statistics"""
    try:
        stats = await transparent_status_system.notification_system.get_user_notification_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/{user_id}/preferences")
async def set_user_preferences(
    user_id: str,
    request: UserPreferencesRequest
):
    """Set user communication preferences"""
    try:
        preferences = request.dict(exclude_none=True)
        
        await transparent_status_system.set_user_communication_preferences(
            user_id=user_id,
            preferences=preferences
        )
        
        return {
            "success": True,
            "message": "User preferences updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error setting user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/{user_id}/activity")
async def update_user_activity(
    user_id: str,
    request: UserActivityRequest
):
    """Update user activity state"""
    try:
        activity_state = UserActivityState(request.activity_state)
        
        await transparent_status_system.update_user_activity(
            user_id=user_id,
            activity_state=activity_state,
            context=request.context
        )
        
        return {
            "success": True,
            "message": "User activity updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating user activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Help System Endpoints

@router.get("/users/{user_id}/help")
async def get_contextual_help(
    user_id: str,
    trigger: str = "user_request",
    context: Optional[str] = None,
    max_suggestions: int = 5
):
    """Get contextual help for user"""
    try:
        help_trigger = HelpTrigger(trigger)
        context_dict = json.loads(context) if context else None
        
        suggestions = await transparent_status_system.help_system.get_contextual_help(
            user_id=user_id,
            trigger=help_trigger,
            context=context_dict,
            max_suggestions=max_suggestions
        )
        
        return {
            "user_id": user_id,
            "suggestions": [
                {
                    "id": suggestion.suggestion_id,
                    "title": suggestion.content.title,
                    "content": suggestion.content.content,
                    "format": suggestion.content.format.value,
                    "relevance_score": suggestion.relevance_score,
                    "estimated_time": suggestion.content.estimated_time,
                    "tags": suggestion.content.tags
                }
                for suggestion in suggestions
            ],
            "total_suggestions": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error getting contextual help: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/help/proactive")
async def get_proactive_help(
    user_id: str,
    context: Optional[str] = None
):
    """Get proactive help suggestion"""
    try:
        context_dict = json.loads(context) if context else {}
        
        suggestion = await transparent_status_system.provide_proactive_help(
            user_id=user_id,
            context=context_dict
        )
        
        if suggestion:
            return {
                "user_id": user_id,
                "suggestion": suggestion,
                "has_suggestion": True
            }
        else:
            return {
                "user_id": user_id,
                "suggestion": None,
                "has_suggestion": False
            }
        
    except Exception as e:
        logger.error(f"Error getting proactive help: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/help/content")
async def add_help_content(
    content_id: str,
    request: HelpContentRequest
):
    """Add new help content"""
    try:
        help_format = HelpFormat(request.format)
        expertise_level = UserExpertiseLevel(request.expertise_level)
        
        await transparent_status_system.help_system.add_help_content(
            content_id=content_id,
            title=request.title,
            content=request.content,
            format=help_format,
            expertise_level=expertise_level,
            tags=request.tags,
            prerequisites=request.prerequisites,
            interactive_elements=request.interactive_elements
        )
        
        return {
            "success": True,
            "content_id": content_id,
            "message": "Help content added successfully"
        }
        
    except Exception as e:
        logger.error(f"Error adding help content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/help/{content_id}/feedback")
async def record_help_feedback(
    user_id: str,
    content_id: str,
    helpful: bool,
    feedback: Optional[str] = None
):
    """Record user feedback on help content"""
    try:
        await transparent_status_system.help_system.record_help_feedback(
            user_id=user_id,
            content_id=content_id,
            helpful=helpful,
            feedback=feedback
        )
        
        return {
            "success": True,
            "message": "Feedback recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error recording help feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error Handling Endpoints

@router.post("/users/{user_id}/errors")
async def handle_error_with_help(
    user_id: str,
    error_type: str,
    error_message: str,
    context: Optional[str] = None,
    component: str = "system"
):
    """Handle error with integrated help"""
    try:
        context_dict = json.loads(context) if context else {}
        error = Exception(error_message)
        error.__class__.__name__ = error_type
        
        help_suggestions = await transparent_status_system.handle_error_with_help(
            user_id=user_id,
            error=error,
            context=context_dict,
            component=component
        )
        
        return {
            "user_id": user_id,
            "error_handled": True,
            "help_suggestions": help_suggestions,
            "total_suggestions": len(help_suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error handling error with help: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics and Overview Endpoints

@router.get("/users/{user_id}/overview")
async def get_user_status_overview(user_id: str):
    """Get comprehensive status overview for user"""
    try:
        overview = await transparent_status_system.get_user_status_overview(user_id)
        return overview
        
    except Exception as e:
        logger.error(f"Error getting user status overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/notifications")
async def get_notification_analytics():
    """Get system-wide notification analytics"""
    try:
        analytics = await transparent_status_system.notification_system.get_notification_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting notification analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/help")
async def get_help_analytics():
    """Get help system analytics"""
    try:
        analytics = await transparent_status_system.help_system.get_help_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting help analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Real-time Communication Endpoints

@router.get("/users/{user_id}/stream")
async def stream_user_communications(user_id: str):
    """Stream real-time communications for user"""
    async def event_stream():
        """Generate server-sent events for user communications"""
        try:
            # Create a queue for this user's events
            event_queue = asyncio.Queue()
            
            # Subscribe to user communications
            async def communication_handler(status_update: UnifiedStatusUpdate):
                if status_update.user_id == user_id or status_update.user_id is None:
                    await event_queue.put(status_update)
            
            await transparent_status_system.subscribe_to_user_communications(
                user_id, communication_handler
            )
            
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'user_id': user_id})}\n\n"
            
            # Stream events
            while True:
                try:
                    # Wait for event with timeout
                    status_update = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    
                    event_data = {
                        "type": "status_update",
                        "event_type": status_update.event_type.value,
                        "component": status_update.component,
                        "title": status_update.title,
                        "message": status_update.message,
                        "priority": status_update.priority,
                        "data": status_update.data,
                        "actions": status_update.actions,
                        "progress_info": status_update.progress_info,
                        "help_suggestions": status_update.help_suggestions,
                        "timestamp": status_update.timestamp.isoformat()
                    }
                    
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                    
        except Exception as e:
            logger.error(f"Error in event stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )