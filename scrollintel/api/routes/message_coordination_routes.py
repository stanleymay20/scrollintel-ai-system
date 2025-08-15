"""
Message Coordination API Routes

REST API endpoints for managing coordinated crisis communications across all channels.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.message_coordination_engine import MessageCoordinationEngine
from ...models.crisis_communication_models import NotificationChannel, NotificationPriority, StakeholderType
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/crisis/message-coordination", tags=["Crisis Message Coordination"])
logger = logging.getLogger(__name__)

# Global engine instance
coordination_engine = MessageCoordinationEngine()


@router.post("/messages")
async def create_coordinated_message(
    message_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a new coordinated message
    
    Args:
        message_data: Message creation data
        
    Returns:
        Created message information
    """
    try:
        # Extract and validate required fields
        crisis_id = message_data.get("crisis_id")
        message_type = message_data.get("message_type")
        master_content = message_data.get("master_content")
        master_subject = message_data.get("master_subject")
        target_channels = [NotificationChannel(ch) for ch in message_data.get("target_channels", [])]
        target_stakeholders = [StakeholderType(st) for st in message_data.get("target_stakeholders", [])]
        
        if not all([crisis_id, message_type, master_content, master_subject]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: crisis_id, message_type, master_content, master_subject"
            )
        
        # Optional fields
        priority = NotificationPriority(message_data.get("priority", "medium"))
        requires_approval = message_data.get("requires_approval", True)
        created_by = current_user.get("username", "unknown")
        
        # Create coordinated message
        message = await coordination_engine.create_coordinated_message(
            crisis_id=crisis_id,
            message_type=message_type,
            master_content=master_content,
            master_subject=master_subject,
            target_channels=target_channels,
            target_stakeholders=target_stakeholders,
            priority=priority,
            requires_approval=requires_approval,
            created_by=created_by
        )
        
        return {
            "status": "success",
            "message": "Coordinated message created successfully",
            "data": {
                "message_id": message.id,
                "status": message.status.value,
                "target_channels": [ch.value for ch in message.target_channels],
                "requires_approval": message.requires_approval,
                "approval_workflow": [
                    {
                        "step_order": step.step_order,
                        "approver_role": step.approver_role,
                        "status": step.status.value
                    }
                    for step in message.approval_workflow
                ],
                "channel_adaptations": message.channel_adaptations,
                "created_at": message.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating coordinated message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/{message_id}/submit-approval")
async def submit_message_for_approval(
    message_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Submit message for approval workflow
    
    Args:
        message_id: ID of message to submit
        
    Returns:
        Submission status
    """
    try:
        success = await coordination_engine.submit_for_approval(message_id)
        
        if success:
            return {
                "status": "success",
                "message": "Message submitted for approval",
                "message_id": message_id
            }
        else:
            raise HTTPException(status_code=404, detail="Message not found")
            
    except Exception as e:
        logger.error(f"Error submitting message for approval: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/{message_id}/approve")
async def approve_message(
    message_id: str,
    approval_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Approve message at current workflow step
    
    Args:
        message_id: ID of message to approve
        approval_data: Approval information
        
    Returns:
        Approval status
    """
    try:
        approver_id = current_user.get("user_id", "unknown")
        approver_role = approval_data.get("approver_role")
        comments = approval_data.get("comments", "")
        
        if not approver_role:
            raise HTTPException(status_code=400, detail="approver_role is required")
        
        success = await coordination_engine.approve_message(
            message_id=message_id,
            approver_id=approver_id,
            approver_role=approver_role,
            comments=comments
        )
        
        if success:
            return {
                "status": "success",
                "message": "Message approved successfully",
                "message_id": message_id,
                "approver_role": approver_role
            }
        else:
            raise HTTPException(status_code=404, detail="Message not found or approval step not valid")
            
    except Exception as e:
        logger.error(f"Error approving message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/{message_id}/reject")
async def reject_message(
    message_id: str,
    rejection_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Reject message
    
    Args:
        message_id: ID of message to reject
        rejection_data: Rejection information
        
    Returns:
        Rejection status
    """
    try:
        approver_id = current_user.get("user_id", "unknown")
        approver_role = rejection_data.get("approver_role")
        rejection_reason = rejection_data.get("rejection_reason", "")
        
        if not approver_role or not rejection_reason:
            raise HTTPException(
                status_code=400, 
                detail="approver_role and rejection_reason are required"
            )
        
        success = await coordination_engine.reject_message(
            message_id=message_id,
            approver_id=approver_id,
            approver_role=approver_role,
            rejection_reason=rejection_reason
        )
        
        if success:
            return {
                "status": "success",
                "message": "Message rejected",
                "message_id": message_id,
                "approver_role": approver_role,
                "rejection_reason": rejection_reason
            }
        else:
            raise HTTPException(status_code=404, detail="Message not found or rejection step not valid")
            
    except Exception as e:
        logger.error(f"Error rejecting message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/{message_id}/publish")
async def publish_message(
    message_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Publish approved message to all channels
    
    Args:
        message_id: ID of message to publish
        
    Returns:
        Publication results
    """
    try:
        result = await coordination_engine.publish_message(message_id)
        
        if result["success"]:
            return {
                "status": "success",
                "message": "Message published successfully",
                "data": result
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to publish message: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error publishing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/messages/{message_id}/version")
async def update_message_version(
    message_id: str,
    version_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Create new version of message
    
    Args:
        message_id: ID of message to update
        version_data: New version data
        
    Returns:
        Version update status
    """
    try:
        new_content = version_data.get("content")
        new_subject = version_data.get("subject")
        changes_summary = version_data.get("changes_summary", "")
        author = current_user.get("username", "unknown")
        
        if not new_content or not new_subject:
            raise HTTPException(
                status_code=400,
                detail="content and subject are required"
            )
        
        success = await coordination_engine.update_message_version(
            message_id=message_id,
            new_content=new_content,
            new_subject=new_subject,
            author=author,
            changes_summary=changes_summary
        )
        
        if success:
            return {
                "status": "success",
                "message": "Message version updated successfully",
                "message_id": message_id
            }
        else:
            raise HTTPException(status_code=404, detail="Message not found")
            
    except Exception as e:
        logger.error(f"Error updating message version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages/{message_id}/status")
async def get_message_status(
    message_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get current status of message
    
    Args:
        message_id: ID of message
        
    Returns:
        Message status information
    """
    try:
        status = coordination_engine.get_message_status(message_id)
        
        if status:
            return {
                "status": "success",
                "data": status
            }
        else:
            raise HTTPException(status_code=404, detail="Message not found")
            
    except Exception as e:
        logger.error(f"Error getting message status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages/{message_id}/effectiveness")
async def get_message_effectiveness(
    message_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get message effectiveness metrics
    
    Args:
        message_id: ID of message
        
    Returns:
        Effectiveness metrics
    """
    try:
        metrics = await coordination_engine.track_message_effectiveness(message_id)
        
        return {
            "status": "success",
            "data": {
                "message_id": metrics.message_id,
                "total_sent": metrics.total_sent,
                "total_delivered": metrics.total_delivered,
                "total_read": metrics.total_read,
                "total_responded": metrics.total_responded,
                "delivery_rate": metrics.delivery_rate,
                "read_rate": metrics.read_rate,
                "response_rate": metrics.response_rate,
                "overall_effectiveness_score": metrics.overall_effectiveness_score,
                "channel_performance": metrics.channel_performance,
                "stakeholder_engagement": metrics.stakeholder_engagement,
                "calculated_at": metrics.calculated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting message effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages")
async def list_messages(
    crisis_id: Optional[str] = None,
    status: Optional[str] = None,
    message_type: Optional[str] = None,
    limit: int = 50,
    current_user: Dict = Depends(get_current_user)
):
    """
    List coordinated messages with optional filtering
    
    Args:
        crisis_id: Filter by crisis ID
        status: Filter by message status
        message_type: Filter by message type
        limit: Maximum number of messages to return
        
    Returns:
        List of messages
    """
    try:
        messages = list(coordination_engine.messages.values())
        
        # Apply filters
        if crisis_id:
            messages = [m for m in messages if m.crisis_id == crisis_id]
        
        if status:
            messages = [m for m in messages if m.status.value == status]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        # Limit results
        messages = messages[:limit]
        
        return {
            "status": "success",
            "data": {
                "messages": [
                    {
                        "message_id": m.id,
                        "crisis_id": m.crisis_id,
                        "message_type": m.message_type,
                        "status": m.status.value,
                        "priority": m.priority.value,
                        "master_subject": m.master_subject,
                        "target_channels": [ch.value for ch in m.target_channels],
                        "requires_approval": m.requires_approval,
                        "created_by": m.created_by,
                        "created_at": m.created_at.isoformat(),
                        "published_time": m.published_time.isoformat() if m.published_time else None
                    }
                    for m in messages
                ],
                "total": len(messages)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_coordination_metrics(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get overall message coordination metrics
    
    Returns:
        Coordination system metrics
    """
    try:
        metrics = coordination_engine.get_coordination_metrics()
        
        return {
            "status": "success",
            "data": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting coordination metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/channels/adapters")
async def get_channel_adapters(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get channel adapter configurations
    
    Returns:
        Channel adapter information
    """
    try:
        return {
            "status": "success",
            "data": {
                "adapters": coordination_engine.channel_adapters,
                "supported_channels": list(coordination_engine.channel_adapters.keys())
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting channel adapters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/adaptation")
async def test_content_adaptation(
    test_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Test content adaptation for different channels
    
    Args:
        test_data: Test content and channels
        
    Returns:
        Adapted content for each channel
    """
    try:
        master_content = test_data.get("content", "")
        master_subject = test_data.get("subject", "")
        channels = [NotificationChannel(ch) for ch in test_data.get("channels", [])]
        
        if not master_content or not channels:
            raise HTTPException(
                status_code=400,
                detail="content and channels are required"
            )
        
        adaptations = await coordination_engine._adapt_content_for_channels(
            master_content, master_subject, channels
        )
        
        return {
            "status": "success",
            "data": {
                "original_content": master_content,
                "original_subject": master_subject,
                "adaptations": adaptations
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing content adaptation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))