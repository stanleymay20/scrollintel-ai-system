"""
Webhook System for ScrollIntel - Real-time notifications for prompt management events.
"""
import asyncio
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

from ..models.database import Base
from ..core.config import get_config


class WebhookEventType(Enum):
    """Types of webhook events."""
    PROMPT_CREATED = "prompt.created"
    PROMPT_UPDATED = "prompt.updated"
    PROMPT_DELETED = "prompt.deleted"
    PROMPT_VERSION_CREATED = "prompt.version.created"
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_COMPLETED = "experiment.completed"
    OPTIMIZATION_STARTED = "optimization.started"
    OPTIMIZATION_COMPLETED = "optimization.completed"
    USAGE_THRESHOLD_EXCEEDED = "usage.threshold.exceeded"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"


class WebhookStatus(Enum):
    """Status of webhook delivery."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"


@dataclass
class WebhookEvent:
    """Webhook event data."""
    id: str
    event_type: WebhookEventType
    resource_type: str
    resource_id: str
    action: str
    timestamp: datetime
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "data": self.data
        }


class WebhookEndpoint(Base):
    """Database model for webhook endpoints."""
    __tablename__ = "webhook_endpoints"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    url = Column(String(2048), nullable=False)
    secret = Column(String(255))  # For signature verification
    events = Column(JSON, default=list)  # List of event types to subscribe to
    active = Column(Boolean, default=True)
    verify_ssl = Column(Boolean, default=True)
    timeout_seconds = Column(Integer, default=30)
    max_retries = Column(Integer, default=3)
    retry_backoff = Column(Integer, default=60)  # Seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_success = Column(DateTime)
    last_failure = Column(DateTime)
    failure_count = Column(Integer, default=0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "url": self.url,
            "events": self.events or [],
            "active": self.active,
            "verify_ssl": self.verify_ssl,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "failure_count": self.failure_count
        }


class WebhookDelivery(Base):
    """Database model for webhook delivery attempts."""
    __tablename__ = "webhook_deliveries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    endpoint_id = Column(String, nullable=False, index=True)
    event_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    status = Column(String, default=WebhookStatus.PENDING.value)
    attempt_count = Column(Integer, default=0)
    response_status = Column(Integer)
    response_body = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime)
    next_retry = Column(DateTime)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "endpoint_id": self.endpoint_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "response_status": self.response_status,
            "response_body": self.response_body,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "next_retry": self.next_retry.isoformat() if self.next_retry else None
        }


class WebhookManager:
    """Manages webhook endpoints and event delivery."""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.config = get_config()
        self.db = db_session
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.retry_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.workers: List[asyncio.Task] = []
        
        # Event handlers
        self.event_handlers: Dict[WebhookEventType, List[Callable]] = {}
    
    async def start(self):
        """Start the webhook delivery system."""
        if self.running:
            return
        
        self.running = True
        
        # Start delivery workers
        for i in range(3):  # 3 concurrent workers
            worker = asyncio.create_task(self._delivery_worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start retry worker
        retry_worker = asyncio.create_task(self._retry_worker())
        self.workers.append(retry_worker)
    
    async def stop(self):
        """Stop the webhook delivery system."""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
    
    async def register_endpoint(
        self,
        user_id: str,
        name: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        verify_ssl: bool = True,
        timeout_seconds: int = 30,
        max_retries: int = 3
    ) -> str:
        """Register a new webhook endpoint."""
        if not self.db:
            raise ValueError("Database session required")
        
        # Validate events
        valid_events = [event.value for event in WebhookEventType]
        invalid_events = [event for event in events if event not in valid_events]
        if invalid_events:
            raise ValueError(f"Invalid event types: {invalid_events}")
        
        endpoint = WebhookEndpoint(
            user_id=user_id,
            name=name,
            url=url,
            secret=secret,
            events=events,
            verify_ssl=verify_ssl,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        
        self.db.add(endpoint)
        self.db.commit()
        
        return endpoint.id
    
    async def update_endpoint(
        self,
        endpoint_id: str,
        user_id: str,
        **updates
    ) -> bool:
        """Update a webhook endpoint."""
        if not self.db:
            return False
        
        endpoint = self.db.query(WebhookEndpoint).filter(
            WebhookEndpoint.id == endpoint_id,
            WebhookEndpoint.user_id == user_id
        ).first()
        
        if not endpoint:
            return False
        
        # Update allowed fields
        allowed_fields = [
            'name', 'url', 'secret', 'events', 'active',
            'verify_ssl', 'timeout_seconds', 'max_retries', 'retry_backoff'
        ]
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(endpoint, field, value)
        
        endpoint.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True
    
    async def delete_endpoint(self, endpoint_id: str, user_id: str) -> bool:
        """Delete a webhook endpoint."""
        if not self.db:
            return False
        
        endpoint = self.db.query(WebhookEndpoint).filter(
            WebhookEndpoint.id == endpoint_id,
            WebhookEndpoint.user_id == user_id
        ).first()
        
        if not endpoint:
            return False
        
        self.db.delete(endpoint)
        self.db.commit()
        
        return True
    
    async def get_endpoints(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all webhook endpoints for a user."""
        if not self.db:
            return []
        
        endpoints = self.db.query(WebhookEndpoint).filter(
            WebhookEndpoint.user_id == user_id
        ).all()
        
        return [endpoint.to_dict() for endpoint in endpoints]
    
    async def trigger_event(self, event: WebhookEvent):
        """Trigger a webhook event."""
        if not self.running:
            await self.start()
        
        # Add to delivery queue
        await self.delivery_queue.put(event)
        
        # Call registered event handlers
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                # Log error but don't fail the webhook delivery
                print(f"Event handler error: {e}")
    
    def register_event_handler(self, event_type: WebhookEventType, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _delivery_worker(self, worker_name: str):
        """Worker for delivering webhook events."""
        while self.running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.delivery_queue.get(),
                    timeout=1.0
                )
                
                await self._deliver_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Delivery worker {worker_name} error: {e}")
    
    async def _deliver_event(self, event: WebhookEvent):
        """Deliver event to all subscribed endpoints."""
        if not self.db:
            return
        
        # Find endpoints subscribed to this event type
        endpoints = self.db.query(WebhookEndpoint).filter(
            WebhookEndpoint.active == True,
            WebhookEndpoint.events.contains([event.event_type.value])
        ).all()
        
        for endpoint in endpoints:
            # Create delivery record
            delivery = WebhookDelivery(
                endpoint_id=endpoint.id,
                event_id=event.id,
                event_type=event.event_type.value
            )
            
            self.db.add(delivery)
            self.db.commit()
            
            # Attempt delivery
            await self._attempt_delivery(endpoint, event, delivery)
    
    async def _attempt_delivery(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
        delivery: WebhookDelivery
    ):
        """Attempt to deliver event to endpoint."""
        delivery.attempt_count += 1
        delivery.status = WebhookStatus.PENDING.value
        
        try:
            # Prepare payload
            payload = {
                "event": event.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
                "delivery_id": delivery.id
            }
            
            # Create signature if secret is provided
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "ScrollIntel-Webhooks/1.0",
                "X-ScrollIntel-Event": event.event_type.value,
                "X-ScrollIntel-Delivery": delivery.id
            }
            
            if endpoint.secret:
                signature = self._create_signature(
                    json.dumps(payload, sort_keys=True),
                    endpoint.secret
                )
                headers["X-ScrollIntel-Signature"] = signature
            
            # Make HTTP request
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            connector = aiohttp.TCPConnector(verify_ssl=endpoint.verify_ssl)
            
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            ) as session:
                async with session.post(
                    endpoint.url,
                    json=payload,
                    headers=headers
                ) as response:
                    delivery.response_status = response.status
                    delivery.response_body = await response.text()
                    
                    if 200 <= response.status < 300:
                        # Success
                        delivery.status = WebhookStatus.DELIVERED.value
                        delivery.delivered_at = datetime.utcnow()
                        endpoint.last_success = datetime.utcnow()
                        endpoint.failure_count = 0
                    else:
                        # HTTP error
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}"
                        )
        
        except Exception as e:
            # Delivery failed
            delivery.status = WebhookStatus.FAILED.value
            delivery.error_message = str(e)
            endpoint.last_failure = datetime.utcnow()
            endpoint.failure_count += 1
            
            # Schedule retry if attempts remaining
            if delivery.attempt_count < endpoint.max_retries:
                delivery.status = WebhookStatus.RETRYING.value
                retry_delay = endpoint.retry_backoff * (2 ** (delivery.attempt_count - 1))
                delivery.next_retry = datetime.utcnow() + timedelta(seconds=retry_delay)
                
                # Add to retry queue
                await self.retry_queue.put((endpoint, event, delivery))
            
            # Disable endpoint if too many failures
            if endpoint.failure_count >= 10:
                endpoint.active = False
        
        finally:
            self.db.commit()
    
    def _create_signature(self, payload: str, secret: str) -> str:
        """Create HMAC signature for webhook payload."""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def _retry_worker(self):
        """Worker for retrying failed webhook deliveries."""
        while self.running:
            try:
                # Get retry item from queue with timeout
                endpoint, event, delivery = await asyncio.wait_for(
                    self.retry_queue.get(),
                    timeout=5.0
                )
                
                # Check if it's time to retry
                if delivery.next_retry and datetime.utcnow() >= delivery.next_retry:
                    await self._attempt_delivery(endpoint, event, delivery)
                else:
                    # Put back in queue for later
                    await self.retry_queue.put((endpoint, event, delivery))
                    await asyncio.sleep(1)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Retry worker error: {e}")
    
    async def get_delivery_status(
        self,
        endpoint_id: str,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get delivery status for an endpoint."""
        if not self.db:
            return []
        
        # Verify endpoint ownership
        endpoint = self.db.query(WebhookEndpoint).filter(
            WebhookEndpoint.id == endpoint_id,
            WebhookEndpoint.user_id == user_id
        ).first()
        
        if not endpoint:
            return []
        
        deliveries = self.db.query(WebhookDelivery).filter(
            WebhookDelivery.endpoint_id == endpoint_id
        ).order_by(
            WebhookDelivery.created_at.desc()
        ).limit(limit).all()
        
        return [delivery.to_dict() for delivery in deliveries]
    
    async def test_endpoint(self, endpoint_id: str, user_id: str) -> Dict[str, Any]:
        """Test a webhook endpoint with a test event."""
        if not self.db:
            return {"success": False, "error": "Database not available"}
        
        endpoint = self.db.query(WebhookEndpoint).filter(
            WebhookEndpoint.id == endpoint_id,
            WebhookEndpoint.user_id == user_id
        ).first()
        
        if not endpoint:
            return {"success": False, "error": "Endpoint not found"}
        
        # Create test event
        test_event = WebhookEvent(
            id=f"test_{int(time.time())}",
            event_type=WebhookEventType.PROMPT_CREATED,
            resource_type="prompt",
            resource_id="test-prompt-id",
            action="test",
            timestamp=datetime.utcnow(),
            user_id=user_id,
            data={"test": True, "message": "This is a test webhook"}
        )
        
        # Create delivery record
        delivery = WebhookDelivery(
            endpoint_id=endpoint.id,
            event_id=test_event.id,
            event_type=test_event.event_type.value
        )
        
        self.db.add(delivery)
        self.db.commit()
        
        # Attempt delivery
        await self._attempt_delivery(endpoint, test_event, delivery)
        
        return {
            "success": delivery.status == WebhookStatus.DELIVERED.value,
            "status": delivery.status,
            "response_status": delivery.response_status,
            "error_message": delivery.error_message,
            "delivery_id": delivery.id
        }


# Global webhook manager instance
webhook_manager = WebhookManager()


# Convenience functions
async def trigger_webhook_event(
    event_type: WebhookEventType,
    resource_type: str,
    resource_id: str,
    action: str,
    user_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
):
    """Trigger a webhook event."""
    event = WebhookEvent(
        id=f"event_{int(time.time() * 1000000)}",
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        timestamp=datetime.utcnow(),
        user_id=user_id,
        data=data or {}
    )
    
    await webhook_manager.trigger_event(event)


async def register_webhook_endpoint(
    user_id: str,
    name: str,
    url: str,
    events: List[str],
    secret: Optional[str] = None
) -> str:
    """Register a webhook endpoint."""
    return await webhook_manager.register_endpoint(
        user_id=user_id,
        name=name,
        url=url,
        events=events,
        secret=secret
    )