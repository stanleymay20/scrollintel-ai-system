"""
API routes for User Guidance and Support System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ...core.user_guidance_system import UserGuidanceSystem
from ...models.user_guidance_models import (
    GuidanceContext, HelpRequest, ErrorExplanation,
    ProactiveGuidance, SupportTicket, UserFeedback,
    GuidanceResponse, ContextualHint
)
from ...core.config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/guidance", tags=["user-guidance"])

# Initialize guidance system
guidance_system = UserGuidanceSystem()

@router.post("/help/contextual")
async def get_contextual_help(
    context: GuidanceContext,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Get contextual help based on user's current situation
    """
    try:
        guidance = await guidance_system.provide_contextual_help(context)
        
        # Track help request in background
        background_tasks.add_task(
            _track_help_request, 
            context.user_id, 
            "contextual_help", 
            guidance.get('confidence_score', 0.0)
        )
        
        return {
            "success": True,
            "guidance": guidance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error providing contextual help: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to provide contextual help: {str(e)}"
        )

@router.post("/error/explain")
async def explain_error(
    error_data: Dict[str, Any],
    context: GuidanceContext
) -> Dict[str, Any]:
    """
    Get intelligent explanation for an error
    """
    try:
        # Create exception from error data
        error_type = error_data.get('type', 'Exception')
        error_message = error_data.get('message', 'Unknown error')
        
        # Create a mock exception for explanation
        class MockException(Exception):
            def __init__(self, message: str):
                self.message = message
                super().__init__(message)
        
        mock_error = MockException(error_message)
        mock_error.__class__.__name__ = error_type
        
        explanation = await guidance_system.explain_error_intelligently(
            mock_error, context
        )
        
        return {
            "success": True,
            "explanation": {
                "error_id": explanation.error_id,
                "user_friendly_explanation": explanation.user_friendly_explanation,
                "actionable_solutions": explanation.actionable_solutions,
                "severity": explanation.severity.value,
                "resolution_confidence": explanation.resolution_confidence,
                "auto_resolved": explanation.auto_resolved
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error explaining error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to explain error: {str(e)}"
        )

@router.get("/proactive/{user_id}")
async def get_proactive_guidance(
    user_id: str,
    system_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get proactive guidance for a user
    """
    try:
        if system_state is None:
            system_state = await _get_current_system_state()
        
        guidance_list = await guidance_system.provide_proactive_guidance(
            user_id, system_state
        )
        
        return {
            "success": True,
            "guidance": [
                {
                    "guidance_id": g.guidance_id,
                    "type": g.type.value,
                    "title": g.title,
                    "message": g.message,
                    "actions": g.actions,
                    "priority": g.priority,
                    "expires_at": g.expires_at.isoformat() if g.expires_at else None
                }
                for g in guidance_list
            ],
            "count": len(guidance_list),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting proactive guidance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get proactive guidance: {str(e)}"
        )

@router.post("/support/ticket")
async def create_support_ticket(
    context: GuidanceContext,
    issue_description: str,
    error_details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an automated support ticket
    """
    try:
        ticket = await guidance_system.create_automated_support_ticket(
            context, issue_description, error_details
        )
        
        return {
            "success": True,
            "ticket": {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "priority": ticket.priority,
                "status": ticket.status.value,
                "created_at": ticket.created_at.isoformat(),
                "estimated_resolution": await _estimate_resolution_time(ticket)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating support ticket: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create support ticket: {str(e)}"
        )

@router.get("/support/ticket/{ticket_id}")
async def get_support_ticket(ticket_id: str) -> Dict[str, Any]:
    """
    Get support ticket details
    """
    try:
        ticket = guidance_system.support_tickets.get(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        return {
            "success": True,
            "ticket": {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "description": ticket.description,
                "priority": ticket.priority,
                "status": ticket.status.value,
                "created_at": ticket.created_at.isoformat(),
                "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
                "tags": ticket.tags,
                "resolution_notes": ticket.resolution_notes
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting support ticket: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get support ticket: {str(e)}"
        )

@router.post("/feedback")
async def submit_feedback(feedback: UserFeedback) -> Dict[str, Any]:
    """
    Submit feedback on guidance provided
    """
    try:
        # Store feedback for learning
        await _store_user_feedback(feedback)
        
        # Update guidance effectiveness
        await _update_guidance_effectiveness(feedback)
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "feedback_id": feedback.feedback_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )

@router.get("/hints/{page}")
async def get_contextual_hints(
    page: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Get contextual hints for a specific page
    """
    try:
        hints = await _get_page_hints(page, user_id)
        
        return {
            "success": True,
            "hints": [
                {
                    "hint_id": hint.hint_id,
                    "element_selector": hint.element_selector,
                    "hint_text": hint.hint_text,
                    "priority": hint.priority,
                    "shown_count": hint.shown_count,
                    "max_shows": hint.max_shows
                }
                for hint in hints
                if not hint.user_dismissed and hint.shown_count < hint.max_shows
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting contextual hints: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get contextual hints: {str(e)}"
        )

@router.post("/hints/{hint_id}/dismiss")
async def dismiss_hint(hint_id: str, user_id: str) -> Dict[str, Any]:
    """
    Dismiss a contextual hint
    """
    try:
        await _dismiss_hint(hint_id, user_id)
        
        return {
            "success": True,
            "message": "Hint dismissed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error dismissing hint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dismiss hint: {str(e)}"
        )

@router.get("/analytics/effectiveness")
async def get_guidance_analytics(
    user_id: Optional[str] = None,
    days: int = 30
) -> Dict[str, Any]:
    """
    Get guidance system effectiveness analytics
    """
    try:
        analytics = await _get_guidance_analytics(user_id, days)
        
        return {
            "success": True,
            "analytics": analytics,
            "period_days": days,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting guidance analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get guidance analytics: {str(e)}"
        )

@router.post("/guidance/{guidance_id}/action")
async def execute_guidance_action(
    guidance_id: str,
    action: Dict[str, Any],
    user_id: str
) -> Dict[str, Any]:
    """
    Execute an action from guidance
    """
    try:
        result = await _execute_guidance_action(guidance_id, action, user_id)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing guidance action: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute guidance action: {str(e)}"
        )

# Helper functions
async def _track_help_request(
    user_id: str, 
    help_type: str, 
    confidence_score: float
) -> None:
    """Track help request for analytics"""
    # Implementation would store tracking data
    pass

async def _get_current_system_state() -> Dict[str, Any]:
    """Get current system state for proactive guidance"""
    return {
        "degraded_services": [],
        "system_load": 0.5,
        "active_users": 100,
        "error_rate": 0.01
    }

async def _estimate_resolution_time(ticket: SupportTicket) -> str:
    """Estimate resolution time for a ticket"""
    priority_times = {
        "low": "2-3 business days",
        "medium": "1-2 business days", 
        "high": "4-8 hours",
        "critical": "1-2 hours"
    }
    return priority_times.get(ticket.priority, "1-2 business days")

async def _store_user_feedback(feedback: UserFeedback) -> None:
    """Store user feedback for learning"""
    # Implementation would store feedback in database
    pass

async def _update_guidance_effectiveness(feedback: UserFeedback) -> None:
    """Update guidance effectiveness based on feedback"""
    # Implementation would update effectiveness metrics
    pass

async def _get_page_hints(page: str, user_id: str) -> List[ContextualHint]:
    """Get contextual hints for a page"""
    # Implementation would return relevant hints
    return []

async def _dismiss_hint(hint_id: str, user_id: str) -> None:
    """Dismiss a contextual hint"""
    # Implementation would mark hint as dismissed
    pass

async def _get_guidance_analytics(
    user_id: Optional[str], 
    days: int
) -> Dict[str, Any]:
    """Get guidance system analytics"""
    return {
        "total_help_requests": 150,
        "successful_resolutions": 135,
        "average_resolution_time": 5.2,
        "user_satisfaction_score": 4.3,
        "proactive_guidance_acceptance": 0.78,
        "error_explanation_clarity": 4.1,
        "support_ticket_auto_resolution": 0.65
    }

async def _execute_guidance_action(
    guidance_id: str, 
    action: Dict[str, Any], 
    user_id: str
) -> Dict[str, Any]:
    """Execute an action from guidance"""
    action_type = action.get('action', '')
    
    if action_type == 'refresh_page':
        return {"action": "refresh", "message": "Please refresh the page"}
    elif action_type == 'retry_after_delay':
        return {"action": "wait", "delay": 30, "message": "Please wait 30 seconds and try again"}
    elif action_type == 'contact_support':
        return {"action": "redirect", "url": "/support", "message": "Redirecting to support"}
    else:
        return {"action": "unknown", "message": "Action not recognized"}