"""
API Routes for Crisis Communication Integration

Provides REST API endpoints for managing crisis communication integration
with all communication channels and systems.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ...core.crisis_communication_integration import (
    CrisisCommunicationIntegrator,
    CommunicationChannelType,
    CommunicationIntegrationConfig,
    CrisisContext
)
from ...models.crisis_models_simple import Crisis, CrisisStatus, CrisisType
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/crisis-communication", tags=["crisis-communication"])


class CommunicationRequest(BaseModel):
    """Request model for processing communication"""
    channel: str = Field(..., description="Communication channel type")
    message: str = Field(..., description="Message content")
    sender: str = Field(..., description="Message sender")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class BroadcastRequest(BaseModel):
    """Request model for broadcasting crisis updates"""
    crisis_id: str = Field(..., description="Crisis ID")
    message: str = Field(..., description="Update message")
    target_channels: Optional[List[str]] = Field(default=None, description="Target channels")


class CrisisRegistrationRequest(BaseModel):
    """Request model for registering a crisis"""
    crisis_id: str = Field(..., description="Crisis ID")
    crisis_type: str = Field(..., description="Crisis type")
    severity_level: int = Field(..., ge=1, le=5, description="Severity level (1-5)")
    affected_areas: List[str] = Field(..., description="Affected areas/systems")
    stakeholders_impacted: List[str] = Field(..., description="Impacted stakeholders")


class ConfigUpdateRequest(BaseModel):
    """Request model for updating integration configuration"""
    enabled_channels: List[str] = Field(..., description="Enabled communication channels")
    crisis_aware_filtering: bool = Field(default=True, description="Enable crisis-aware filtering")
    auto_context_injection: bool = Field(default=True, description="Enable auto context injection")
    priority_routing: bool = Field(default=True, description="Enable priority routing")


# Global integrator instance
integrator_config = CommunicationIntegrationConfig(
    enabled_channels=[
        CommunicationChannelType.EMAIL,
        CommunicationChannelType.SLACK,
        CommunicationChannelType.DASHBOARD,
        CommunicationChannelType.SMS,
        CommunicationChannelType.INTERNAL_CHAT
    ]
)
integrator = CrisisCommunicationIntegrator(integrator_config)


@router.post("/register-crisis")
async def register_crisis(
    request: CrisisRegistrationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Register a crisis for communication integration"""
    try:
        # Create crisis object
        crisis = Crisis(
            id=request.crisis_id,
            crisis_type=CrisisType(request.crisis_type),
            severity_level=request.severity_level,
            start_time=datetime.now(),
            affected_areas=request.affected_areas,
            stakeholders_impacted=request.stakeholders_impacted,
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        # Register with integrator
        success = await integrator.register_crisis(crisis)
        
        if success:
            return {
                "success": True,
                "message": f"Crisis {request.crisis_id} registered successfully",
                "crisis_id": request.crisis_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to register crisis")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid crisis type: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/process-communication")
async def process_communication(
    request: CommunicationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process communication with crisis awareness"""
    try:
        # Validate channel type
        try:
            channel = CommunicationChannelType(request.channel)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid channel type: {request.channel}")
        
        # Process communication
        result = await integrator.process_communication(
            channel=channel,
            message=request.message,
            sender=request.sender,
            context=request.context
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Communication processing failed: {str(e)}")


@router.post("/broadcast-update")
async def broadcast_crisis_update(
    request: BroadcastRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Broadcast crisis update across communication channels"""
    try:
        # Validate target channels if provided
        target_channels = None
        if request.target_channels:
            try:
                target_channels = [CommunicationChannelType(ch) for ch in request.target_channels]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid channel type: {str(e)}")
        
        # Broadcast update
        result = await integrator.broadcast_crisis_update(
            crisis_id=request.crisis_id,
            update_message=request.message,
            target_channels=target_channels
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broadcast failed: {str(e)}")


@router.get("/active-crises")
async def get_active_crises(current_user: dict = Depends(get_current_user)):
    """Get list of active crises"""
    try:
        active_crises = integrator.get_active_crises()
        
        return {
            "success": True,
            "active_crises": active_crises,
            "count": len(active_crises),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active crises: {str(e)}")


@router.put("/crisis/{crisis_id}/status")
async def update_crisis_status(
    crisis_id: str,
    status: str,
    current_user: dict = Depends(get_current_user)
):
    """Update crisis status"""
    try:
        # Validate status
        try:
            new_status = CrisisStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Update status
        await integrator.update_crisis_status(crisis_id, new_status)
        
        return {
            "success": True,
            "message": f"Crisis {crisis_id} status updated to {status}",
            "crisis_id": crisis_id,
            "new_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")


@router.post("/crisis/{crisis_id}/resolve")
async def resolve_crisis(
    crisis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Resolve a crisis and cleanup communication integration"""
    try:
        await integrator.resolve_crisis(crisis_id)
        
        return {
            "success": True,
            "message": f"Crisis {crisis_id} resolved successfully",
            "crisis_id": crisis_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crisis resolution failed: {str(e)}")


@router.get("/configuration")
async def get_integration_configuration(current_user: dict = Depends(get_current_user)):
    """Get current integration configuration"""
    try:
        config = {
            "enabled_channels": [ch.value for ch in integrator.config.enabled_channels],
            "crisis_aware_filtering": integrator.config.crisis_aware_filtering,
            "auto_context_injection": integrator.config.auto_context_injection,
            "priority_routing": integrator.config.priority_routing,
            "message_templates": integrator.config.message_templates,
            "escalation_triggers": integrator.config.escalation_triggers
        }
        
        return {
            "success": True,
            "configuration": config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.put("/configuration")
async def update_integration_configuration(
    request: ConfigUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update integration configuration"""
    try:
        # Validate channels
        try:
            enabled_channels = [CommunicationChannelType(ch) for ch in request.enabled_channels]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid channel type: {str(e)}")
        
        # Update configuration
        integrator.config.enabled_channels = enabled_channels
        integrator.config.crisis_aware_filtering = request.crisis_aware_filtering
        integrator.config.auto_context_injection = request.auto_context_injection
        integrator.config.priority_routing = request.priority_routing
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


@router.get("/channels")
async def get_available_channels(current_user: dict = Depends(get_current_user)):
    """Get list of available communication channels"""
    try:
        channels = [
            {
                "type": ch.value,
                "name": ch.value.replace("_", " ").title(),
                "enabled": ch in integrator.config.enabled_channels
            }
            for ch in CommunicationChannelType
        ]
        
        return {
            "success": True,
            "channels": channels,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get channels: {str(e)}")


@router.get("/crisis/{crisis_id}/context")
async def get_crisis_context(
    crisis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get crisis context for communication integration"""
    try:
        if crisis_id not in integrator.active_crises:
            raise HTTPException(status_code=404, detail=f"Crisis {crisis_id} not found")
        
        crisis_context = integrator.active_crises[crisis_id]
        
        context_data = {
            "crisis_id": crisis_context.crisis_id,
            "crisis_type": crisis_context.crisis_type.value,
            "severity_level": crisis_context.severity_level,
            "status": crisis_context.status.value,
            "affected_systems": crisis_context.affected_systems,
            "stakeholders": crisis_context.stakeholders,
            "communication_protocols": crisis_context.communication_protocols,
            "escalation_level": crisis_context.escalation_level,
            "start_time": crisis_context.start_time.isoformat(),
            "last_update": crisis_context.last_update.isoformat()
        }
        
        return {
            "success": True,
            "crisis_context": context_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get crisis context: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for crisis communication integration"""
    try:
        active_count = len(integrator.active_crises)
        enabled_channels = len(integrator.config.enabled_channels)
        
        return {
            "status": "healthy",
            "active_crises": active_count,
            "enabled_channels": enabled_channels,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")