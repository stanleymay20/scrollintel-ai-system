"""
Stakeholder Notification API Routes

REST API endpoints for managing stakeholder notifications during crisis situations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.stakeholder_notification_engine import StakeholderNotificationEngine
from ...models.crisis_communication_models import (
    Stakeholder, StakeholderType, NotificationTemplate, NotificationMessage,
    NotificationBatch, StakeholderGroup, NotificationPriority, NotificationChannel
)
from ...models.crisis_models_simple import Crisis
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/crisis/stakeholder-notification", tags=["Crisis Stakeholder Notification"])
logger = logging.getLogger(__name__)

# Global engine instance
notification_engine = StakeholderNotificationEngine()


@router.post("/notify/immediate")
async def send_immediate_notifications(
    crisis_data: Dict[str, Any],
    stakeholder_ids: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Dict = Depends(get_current_user)
):
    """
    Send immediate notifications to stakeholders about crisis
    
    Args:
        crisis_data: Crisis information
        stakeholder_ids: Specific stakeholders to notify (optional)
        
    Returns:
        Notification results and delivery status
    """
    try:
        # Convert crisis data to Crisis object
        crisis = Crisis(**crisis_data)
        
        # Send notifications
        result = await notification_engine.notify_stakeholders_immediate(
            crisis=crisis,
            stakeholder_ids=stakeholder_ids
        )
        
        if result["success"]:
            logger.info(f"Immediate notifications sent for crisis {crisis.id}")
            return {
                "status": "success",
                "message": "Immediate notifications sent successfully",
                "data": result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send notifications: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error sending immediate notifications: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stakeholders")
async def add_stakeholder(
    stakeholder_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Add new stakeholder to the notification system
    
    Args:
        stakeholder_data: Stakeholder information
        
    Returns:
        Success status and stakeholder ID
    """
    try:
        # Convert to Stakeholder object
        stakeholder = Stakeholder(**stakeholder_data)
        
        success = notification_engine.add_stakeholder(stakeholder)
        
        if success:
            return {
                "status": "success",
                "message": "Stakeholder added successfully",
                "stakeholder_id": stakeholder.id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add stakeholder")
            
    except Exception as e:
        logger.error(f"Error adding stakeholder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/stakeholders/{stakeholder_id}")
async def update_stakeholder(
    stakeholder_id: str,
    updates: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Update stakeholder information
    
    Args:
        stakeholder_id: ID of stakeholder to update
        updates: Fields to update
        
    Returns:
        Success status
    """
    try:
        success = notification_engine.update_stakeholder(stakeholder_id, updates)
        
        if success:
            return {
                "status": "success",
                "message": "Stakeholder updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Stakeholder not found")
            
    except Exception as e:
        logger.error(f"Error updating stakeholder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stakeholders")
async def list_stakeholders(
    stakeholder_type: Optional[StakeholderType] = None,
    priority: Optional[NotificationPriority] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    List stakeholders with optional filtering
    
    Args:
        stakeholder_type: Filter by stakeholder type
        priority: Filter by priority level
        
    Returns:
        List of stakeholders
    """
    try:
        stakeholders = list(notification_engine.stakeholders.values())
        
        # Apply filters
        if stakeholder_type:
            stakeholders = [s for s in stakeholders if s.stakeholder_type == stakeholder_type]
        
        if priority:
            stakeholders = [s for s in stakeholders if s.priority_level == priority]
        
        return {
            "status": "success",
            "data": {
                "stakeholders": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "stakeholder_type": s.stakeholder_type.value,
                        "priority_level": s.priority_level.value,
                        "role": s.role,
                        "department": s.department,
                        "influence_level": s.influence_level,
                        "preferred_channels": [c.value for c in s.preferred_channels]
                    }
                    for s in stakeholders
                ],
                "total": len(stakeholders)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing stakeholders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates")
async def add_notification_template(
    template_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Add notification template
    
    Args:
        template_data: Template information
        
    Returns:
        Success status and template ID
    """
    try:
        # Convert to NotificationTemplate object
        template = NotificationTemplate(**template_data)
        
        success = notification_engine.add_notification_template(template)
        
        if success:
            return {
                "status": "success",
                "message": "Notification template added successfully",
                "template_id": template.id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add template")
            
    except Exception as e:
        logger.error(f"Error adding notification template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_notification_templates(
    crisis_type: Optional[str] = None,
    stakeholder_type: Optional[StakeholderType] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    List notification templates with optional filtering
    
    Args:
        crisis_type: Filter by crisis type
        stakeholder_type: Filter by stakeholder type
        
    Returns:
        List of templates
    """
    try:
        templates = list(notification_engine.templates.values())
        
        # Apply filters
        if crisis_type:
            templates = [t for t in templates if t.crisis_type == crisis_type]
        
        if stakeholder_type:
            templates = [t for t in templates if t.stakeholder_type == stakeholder_type]
        
        return {
            "status": "success",
            "data": {
                "templates": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "crisis_type": t.crisis_type,
                        "stakeholder_type": t.stakeholder_type.value,
                        "channel": t.channel.value,
                        "subject_template": t.subject_template,
                        "approval_required": t.approval_required,
                        "auto_send": t.auto_send
                    }
                    for t in templates
                ],
                "total": len(templates)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notifications/status/{batch_id}")
async def get_notification_batch_status(
    batch_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get status of notification batch
    
    Args:
        batch_id: ID of notification batch
        
    Returns:
        Batch status and delivery metrics
    """
    try:
        # This would typically query a database
        # For now, return simulated status
        return {
            "status": "success",
            "data": {
                "batch_id": batch_id,
                "status": "completed",
                "total_notifications": 25,
                "successful_deliveries": 23,
                "failed_deliveries": 2,
                "delivery_rate": 0.92,
                "completion_time": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/notification")
async def test_notification_delivery(
    test_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Test notification delivery to specific stakeholder
    
    Args:
        test_data: Test notification data
        
    Returns:
        Test delivery results
    """
    try:
        stakeholder_id = test_data.get("stakeholder_id")
        channel = NotificationChannel(test_data.get("channel", "email"))
        message = test_data.get("message", "Test notification")
        
        if not stakeholder_id or stakeholder_id not in notification_engine.stakeholders:
            raise HTTPException(status_code=404, detail="Stakeholder not found")
        
        # Create test notification
        test_notification = NotificationMessage(
            crisis_id="test",
            stakeholder_id=stakeholder_id,
            channel=channel,
            priority=NotificationPriority.LOW,
            subject="Test Notification",
            content=message
        )
        
        # Send test notification
        result = await notification_engine._send_single_notification(test_notification)
        
        return {
            "status": "success",
            "message": "Test notification sent",
            "data": {
                "notification_id": test_notification.id,
                "delivery_result": result,
                "channel": channel.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error sending test notification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/delivery")
async def get_delivery_metrics(
    crisis_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get notification delivery metrics
    
    Args:
        crisis_id: Filter by specific crisis
        start_date: Start date for metrics
        end_date: End date for metrics
        
    Returns:
        Delivery metrics and analytics
    """
    try:
        # This would typically query metrics from database
        # For now, return simulated metrics
        metrics = {
            "total_notifications": 150,
            "successful_deliveries": 142,
            "failed_deliveries": 8,
            "overall_delivery_rate": 0.947,
            "channel_performance": {
                "email": {"sent": 80, "delivered": 78, "rate": 0.975},
                "sms": {"sent": 40, "delivered": 38, "rate": 0.95},
                "slack": {"sent": 20, "delivered": 20, "rate": 1.0},
                "push": {"sent": 10, "delivered": 6, "rate": 0.6}
            },
            "stakeholder_engagement": {
                "board_member": {"notified": 5, "responded": 5, "rate": 1.0},
                "executive": {"notified": 15, "responded": 14, "rate": 0.933},
                "employee": {"notified": 100, "responded": 85, "rate": 0.85},
                "customer": {"notified": 30, "responded": 12, "rate": 0.4}
            },
            "average_delivery_time_seconds": 2.3,
            "peak_delivery_hour": "09:00",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "data": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting delivery metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stakeholders/bulk-import")
async def bulk_import_stakeholders(
    stakeholders_data: List[Dict[str, Any]],
    current_user: Dict = Depends(get_current_user)
):
    """
    Bulk import stakeholders
    
    Args:
        stakeholders_data: List of stakeholder data
        
    Returns:
        Import results
    """
    try:
        results = {
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for i, stakeholder_data in enumerate(stakeholders_data):
            try:
                stakeholder = Stakeholder(**stakeholder_data)
                success = notification_engine.add_stakeholder(stakeholder)
                
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Row {i+1}: Failed to add stakeholder")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Row {i+1}: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Imported {results['successful']} stakeholders successfully",
            "data": results
        }
        
    except Exception as e:
        logger.error(f"Error bulk importing stakeholders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))