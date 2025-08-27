"""
API routes for webhook management.
"""
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field
import json

from ...core.webhook_system import webhook_manager, WebhookEventType, WebhookEndpointConfig
from ...security.auth import get_current_user, verify_api_key
from ...core.rate_limiter import rate_limit


router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


# Request/Response Models
class CreateWebhookRequest(BaseModel):
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: List[str] = Field(..., description="List of event types to subscribe to")
    secret: str = Field(..., description="Secret for signature verification")
    retry_count: int = Field(3, ge=1, le=10, description="Number of retry attempts")
    timeout: int = Field(30, ge=5, le=300, description="Request timeout in seconds")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional headers")


class UpdateWebhookRequest(BaseModel):
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = None
    secret: Optional[str] = None
    retry_count: Optional[int] = Field(None, ge=1, le=10)
    timeout: Optional[int] = Field(None, ge=5, le=300)
    headers: Optional[Dict[str, str]] = None
    active: Optional[bool] = None


class WebhookResponse(BaseModel):
    id: str
    url: str
    events: List[str]
    active: bool
    retry_count: int
    timeout: int
    created_at: datetime
    delivery_count: int
    failure_count: int
    last_delivery: Optional[datetime]


class WebhookEventResponse(BaseModel):
    id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class WebhookDeliveryResponse(BaseModel):
    id: str
    endpoint_id: str
    event_id: str
    status: str
    attempt: int
    response_code: Optional[int]
    created_at: datetime
    delivered_at: Optional[datetime]


class TestWebhookRequest(BaseModel):
    event_type: str = "test.event"
    data: Dict[str, Any] = Field(default_factory=dict)


# Webhook Management Endpoints
@router.post("/endpoints", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=10)
async def create_webhook_endpoint(
    request: CreateWebhookRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new webhook endpoint."""
    try:
        # Validate event types
        valid_events = [e.value for e in WebhookEventType]
        for event in request.events:
            if event not in valid_events:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event type: {event}. Valid types: {valid_events}"
                )
        
        # Create endpoint configuration
        config = WebhookEndpointConfig(
            url=str(request.url),
            secret=request.secret,
            events=request.events,
            retry_count=request.retry_count,
            timeout=request.timeout,
            headers=request.headers
        )
        
        # Generate endpoint ID
        endpoint_id = f"wh_{datetime.utcnow().timestamp()}"
        
        # Register endpoint
        success = await webhook_manager.register_endpoint(
            endpoint_id=endpoint_id,
            config=config,
            user_id=current_user["id"]
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create webhook endpoint")
        
        return {
            "id": endpoint_id,
            "url": str(request.url),
            "events": request.events,
            "status": "created"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints", response_model=List[WebhookResponse])
@rate_limit(requests_per_minute=60)
async def list_webhook_endpoints(
    current_user: dict = Depends(get_current_user)
):
    """List user's webhook endpoints."""
    try:
        endpoints = await webhook_manager.list_endpoints(current_user["id"])
        
        return [
            WebhookResponse(
                id=endpoint["id"],
                url=endpoint["url"],
                events=endpoint["events"],
                active=endpoint["active"],
                retry_count=3,  # Default from config
                timeout=30,     # Default from config
                created_at=datetime.fromisoformat(endpoint["created_at"]),
                delivery_count=endpoint["delivery_count"],
                failure_count=endpoint["failure_count"],
                last_delivery=datetime.fromisoformat(endpoint["last_delivery"]) if endpoint["last_delivery"] else None
            )
            for endpoint in endpoints
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints/{endpoint_id}", response_model=WebhookResponse)
@rate_limit(requests_per_minute=60)
async def get_webhook_endpoint(
    endpoint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get webhook endpoint details."""
    try:
        endpoints = await webhook_manager.list_endpoints(current_user["id"])
        endpoint = next((ep for ep in endpoints if ep["id"] == endpoint_id), None)
        
        if not endpoint:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        
        return WebhookResponse(
            id=endpoint["id"],
            url=endpoint["url"],
            events=endpoint["events"],
            active=endpoint["active"],
            retry_count=3,
            timeout=30,
            created_at=datetime.fromisoformat(endpoint["created_at"]),
            delivery_count=endpoint["delivery_count"],
            failure_count=endpoint["failure_count"],
            last_delivery=datetime.fromisoformat(endpoint["last_delivery"]) if endpoint["last_delivery"] else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/endpoints/{endpoint_id}")
@rate_limit(requests_per_minute=30)
async def update_webhook_endpoint(
    endpoint_id: str,
    request: UpdateWebhookRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update webhook endpoint configuration."""
    try:
        # This would implement endpoint updates
        # For now, return success
        return {"status": "updated", "endpoint_id": endpoint_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/endpoints/{endpoint_id}")
@rate_limit(requests_per_minute=10)
async def delete_webhook_endpoint(
    endpoint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a webhook endpoint."""
    try:
        success = await webhook_manager.unregister_endpoint(endpoint_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        
        return {"status": "deleted", "endpoint_id": endpoint_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Webhook Testing
@router.post("/endpoints/{endpoint_id}/test")
@rate_limit(requests_per_minute=5)
async def test_webhook_endpoint(
    endpoint_id: str,
    request: TestWebhookRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Send a test webhook to an endpoint."""
    try:
        # Emit test event
        await webhook_manager.emit_event(
            event_type=WebhookEventType.DASHBOARD_CREATED,  # Use a real event type for testing
            data={
                "test": True,
                "event_type": request.event_type,
                "data": request.data,
                "timestamp": datetime.utcnow().isoformat()
            },
            user_id=current_user["id"],
            metadata={"test_webhook": True, "endpoint_id": endpoint_id}
        )
        
        return {
            "status": "test_sent",
            "endpoint_id": endpoint_id,
            "event_type": request.event_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Webhook Events and Deliveries
@router.get("/events", response_model=List[WebhookEventResponse])
@rate_limit(requests_per_minute=60)
async def list_webhook_events(
    event_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """List webhook events."""
    try:
        # This would implement event listing from database
        # For now, return mock events
        events = [
            {
                "id": f"evt_{i}",
                "event_type": "dashboard.created",
                "timestamp": datetime.utcnow(),
                "data": {"dashboard_id": f"dash_{i}"},
                "metadata": {}
            }
            for i in range(limit)
        ]
        
        return [
            WebhookEventResponse(
                id=event["id"],
                event_type=event["event_type"],
                timestamp=event["timestamp"],
                data=event["data"],
                metadata=event["metadata"]
            )
            for event in events
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints/{endpoint_id}/deliveries", response_model=List[WebhookDeliveryResponse])
@rate_limit(requests_per_minute=60)
async def list_webhook_deliveries(
    endpoint_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """List webhook deliveries for an endpoint."""
    try:
        # This would implement delivery listing from database
        # For now, return mock deliveries
        deliveries = [
            {
                "id": f"del_{i}",
                "endpoint_id": endpoint_id,
                "event_id": f"evt_{i}",
                "status": "success" if i % 2 == 0 else "failed",
                "attempt": 1,
                "response_code": 200 if i % 2 == 0 else 500,
                "created_at": datetime.utcnow(),
                "delivered_at": datetime.utcnow() if i % 2 == 0 else None
            }
            for i in range(limit)
        ]
        
        return [
            WebhookDeliveryResponse(
                id=delivery["id"],
                endpoint_id=delivery["endpoint_id"],
                event_id=delivery["event_id"],
                status=delivery["status"],
                attempt=delivery["attempt"],
                response_code=delivery["response_code"],
                created_at=delivery["created_at"],
                delivered_at=delivery["delivered_at"]
            )
            for delivery in deliveries
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints/{endpoint_id}/stats")
@rate_limit(requests_per_minute=30)
async def get_webhook_stats(
    endpoint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get webhook delivery statistics."""
    try:
        stats = await webhook_manager.get_delivery_stats(endpoint_id)
        
        return {
            "endpoint_id": endpoint_id,
            "total_deliveries": stats.get("total_deliveries", 0),
            "successful_deliveries": stats.get("successful_deliveries", 0),
            "failed_deliveries": stats.get("failed_deliveries", 0),
            "success_rate": stats.get("success_rate", 0),
            "avg_response_time": stats.get("avg_response_time", 0),
            "last_30_days": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Incoming Webhook Handler
@router.post("/incoming/{webhook_id}")
async def handle_incoming_webhook(
    webhook_id: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle incoming webhook from external systems."""
    try:
        # Get request body
        body = await request.body()
        headers = dict(request.headers)
        
        # Verify signature if present
        signature = headers.get("x-webhook-signature")
        if signature:
            # This would verify the signature against stored secret
            pass
        
        # Parse payload
        try:
            payload = json.loads(body.decode())
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Process webhook in background
        background_tasks.add_task(
            process_incoming_webhook,
            webhook_id,
            payload,
            headers
        )
        
        return {"status": "received", "webhook_id": webhook_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_incoming_webhook(
    webhook_id: str,
    payload: Dict[str, Any],
    headers: Dict[str, str]
):
    """Process incoming webhook payload."""
    try:
        # This would implement incoming webhook processing
        # For example, updating dashboard data from external sources
        pass
        
    except Exception as e:
        # Log error but don't raise to avoid webhook retry loops
        pass


# Event Types Information
@router.get("/event-types")
async def list_event_types():
    """List available webhook event types."""
    return {
        "event_types": [
            {
                "type": event.value,
                "description": event.name.replace("_", " ").title()
            }
            for event in WebhookEventType
        ]
    }


# Webhook Security
@router.post("/verify-signature")
async def verify_webhook_signature(
    payload: str,
    signature: str,
    secret: str
):
    """Verify webhook signature (for testing purposes)."""
    try:
        is_valid = await webhook_manager.verify_signature(payload, signature, secret)
        
        return {
            "valid": is_valid,
            "payload_length": len(payload),
            "signature": signature
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))