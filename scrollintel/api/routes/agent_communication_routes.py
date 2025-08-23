"""
Agent Communication API Routes

REST API endpoints for agent-to-agent communication, collaboration sessions,
distributed state synchronization, and resource management.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ...core.agent_communication_framework import AgentCommunicationFramework
from ...models.agent_communication_models import (
    AgentMessageCreate, AgentMessageResponse, CollaborationSessionCreate,
    CollaborationSessionResponse, ResourceLockRequest, ResourceLockResponse,
    StateUpdateRequest, StateResponse, MessageDeliveryStatus,
    SessionInviteRequest, MessageType, MessagePriority
)
from ...models.database import get_db
from ...core.config import get_settings

router = APIRouter(prefix="/api/v1/agent-communication", tags=["Agent Communication"])
settings = get_settings()


def get_communication_framework(db: Session = Depends(get_db)) -> AgentCommunicationFramework:
    """Get agent communication framework instance"""
    return AgentCommunicationFramework(db, settings.ENCRYPTION_KEY)


class MessageFilter(BaseModel):
    """Filter for retrieving messages"""
    message_types: Optional[List[MessageType]] = None
    from_agent_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: Optional[MessagePriority] = None
    limit: int = 50
    offset: int = 0


class SessionFilter(BaseModel):
    """Filter for retrieving sessions"""
    status: Optional[str] = None
    initiator_agent_id: Optional[str] = None
    participant_agent_id: Optional[str] = None
    limit: int = 20
    offset: int = 0


# Message Management Endpoints
@router.post("/messages/send", response_model=AgentMessageResponse)
async def send_message(
    from_agent_id: str,
    message_request: AgentMessageCreate,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Send encrypted message between agents"""
    try:
        return await framework.send_message(from_agent_id, message_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.get("/messages/{agent_id}", response_model=List[Dict[str, Any]])
async def receive_messages(
    agent_id: str,
    message_types: Optional[str] = None,
    timeout: float = 5.0,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Receive and decrypt messages for an agent"""
    try:
        # Parse message types if provided
        types = None
        if message_types:
            types = [MessageType(t.strip()) for t in message_types.split(",")]
        
        return await framework.receive_messages(agent_id, types, timeout)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to receive messages: {str(e)}")


@router.post("/messages/{message_id}/acknowledge")
async def acknowledge_message(
    message_id: str,
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Acknowledge receipt of a message"""
    try:
        success = await framework.acknowledge_message(agent_id, message_id)
        if not success:
            raise HTTPException(status_code=404, detail="Message not found or not for this agent")
        
        return {"status": "acknowledged", "message_id": message_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge message: {str(e)}")


@router.post("/messages/broadcast")
async def broadcast_message(
    from_agent_id: str,
    message_content: Dict[str, Any],
    recipient_agents: List[str],
    message_type: MessageType = MessageType.NOTIFICATION,
    priority: MessagePriority = MessagePriority.NORMAL,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Broadcast message to multiple agents"""
    try:
        results = []
        for agent_id in recipient_agents:
            message_request = AgentMessageCreate(
                to_agent_id=agent_id,
                message_type=message_type,
                content=message_content,
                priority=priority
            )
            
            result = await framework.send_message(from_agent_id, message_request)
            results.append({
                "recipient": agent_id,
                "message_id": result.id,
                "status": "sent"
            })
        
        return {
            "broadcast_id": f"broadcast_{datetime.utcnow().timestamp()}",
            "recipients": len(recipient_agents),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast message: {str(e)}")


# Collaboration Session Endpoints
@router.post("/sessions", response_model=CollaborationSessionResponse)
async def create_collaboration_session(
    initiator_agent_id: str,
    session_request: CollaborationSessionCreate,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Create a new collaboration session"""
    try:
        return await framework.create_collaboration_session(initiator_agent_id, session_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.post("/sessions/{session_id}/join")
async def join_collaboration_session(
    session_id: str,
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Join a collaboration session"""
    try:
        success = await framework.join_collaboration_session(agent_id, session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or agent not invited")
        
        return {"status": "joined", "session_id": session_id, "agent_id": agent_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to join session: {str(e)}")


@router.post("/sessions/{session_id}/leave")
async def leave_collaboration_session(
    session_id: str,
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Leave a collaboration session"""
    try:
        success = await framework.leave_collaboration_session(agent_id, session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or agent not in session")
        
        return {"status": "left", "session_id": session_id, "agent_id": agent_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to leave session: {str(e)}")


@router.post("/sessions/{session_id}/message")
async def send_session_message(
    session_id: str,
    agent_id: str,
    message_content: Dict[str, Any],
    message_type: MessageType = MessageType.NOTIFICATION,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Send message to all participants in a collaboration session"""
    try:
        success = await framework.send_session_message(
            agent_id, session_id, message_content, message_type
        )
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or agent not in session")
        
        return {
            "status": "sent",
            "session_id": session_id,
            "sender": agent_id,
            "message_type": message_type.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send session message: {str(e)}")


@router.get("/sessions/{agent_id}", response_model=List[CollaborationSessionResponse])
async def get_agent_sessions(
    agent_id: str,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Get collaboration sessions for an agent"""
    try:
        # This would need to be implemented in the framework
        # For now, return empty list
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


# Distributed State Endpoints
@router.put("/state", response_model=StateResponse)
async def update_shared_state(
    agent_id: str,
    state_request: StateUpdateRequest,
    session_id: Optional[str] = None,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Update distributed shared state"""
    try:
        return await framework.update_shared_state(agent_id, state_request, session_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update state: {str(e)}")


@router.get("/state/{state_namespace}/{state_key}", response_model=StateResponse)
async def get_shared_state(
    state_namespace: str,
    state_key: str,
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Get distributed shared state"""
    try:
        state = await framework.get_shared_state(agent_id, state_key, state_namespace)
        if not state:
            raise HTTPException(status_code=404, detail="State not found")
        
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")


@router.post("/state/synchronize/{session_id}")
async def synchronize_session_state(
    session_id: str,
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Synchronize all state for a session"""
    try:
        synchronized_state = await framework.synchronize_state(agent_id, session_id)
        return {
            "session_id": session_id,
            "synchronized_at": datetime.utcnow().isoformat(),
            "state_count": len(synchronized_state),
            "state": synchronized_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to synchronize state: {str(e)}")


# Resource Locking Endpoints
@router.post("/locks", response_model=ResourceLockResponse)
async def request_resource_lock(
    agent_id: str,
    lock_request: ResourceLockRequest,
    session_id: Optional[str] = None,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Request a lock on a resource"""
    try:
        return await framework.request_resource_lock(agent_id, lock_request, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to request lock: {str(e)}")


@router.delete("/locks/{lock_id}")
async def release_resource_lock(
    lock_id: str,
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Release a resource lock"""
    try:
        success = await framework.release_resource_lock(agent_id, lock_id)
        if not success:
            raise HTTPException(status_code=404, detail="Lock not found or not owned by agent")
        
        return {"status": "released", "lock_id": lock_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to release lock: {str(e)}")


@router.get("/locks/{agent_id}/{resource_id}", response_model=ResourceLockResponse)
async def check_resource_lock(
    agent_id: str,
    resource_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Check if agent has a lock on a resource"""
    try:
        lock = await framework.check_resource_lock(agent_id, resource_id)
        if not lock:
            raise HTTPException(status_code=404, detail="No active lock found")
        
        return lock
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check lock: {str(e)}")


@router.get("/locks/{agent_id}")
async def get_agent_locks(
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Get all locks held by an agent"""
    try:
        # This would need to be implemented in the framework
        # For now, return empty list
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get locks: {str(e)}")


# Status and Monitoring Endpoints
@router.get("/status/{agent_id}")
async def get_agent_communication_status(
    agent_id: str,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Get comprehensive communication status for an agent"""
    try:
        return await framework.get_agent_status(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/cleanup")
async def cleanup_expired_resources(
    background_tasks: BackgroundTasks,
    framework: AgentCommunicationFramework = Depends(get_communication_framework)
):
    """Clean up expired messages, locks, and sessions"""
    try:
        background_tasks.add_task(framework.cleanup_expired_resources)
        return {"status": "cleanup_scheduled", "scheduled_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule cleanup: {str(e)}")


# Health Check Endpoint
@router.get("/health")
async def health_check():
    """Health check for agent communication system"""
    return {
        "status": "healthy",
        "service": "agent_communication",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# WebSocket endpoint for real-time communication (placeholder)
@router.get("/ws/{agent_id}")
async def websocket_endpoint(agent_id: str):
    """WebSocket endpoint for real-time agent communication"""
    # This would be implemented with FastAPI WebSocket support
    # For now, return information about the endpoint
    return {
        "message": "WebSocket endpoint for real-time communication",
        "agent_id": agent_id,
        "endpoint": f"/api/v1/agent-communication/ws/{agent_id}",
        "status": "not_implemented"
    }