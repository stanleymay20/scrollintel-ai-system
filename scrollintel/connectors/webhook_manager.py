"""
Webhook Management System for Real-time Data Updates
Handles webhook registration, validation, and processing.
"""

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, validator


class WebhookStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    PAUSED = "paused"


class WebhookEvent(Enum):
    DATA_UPDATED = "data.updated"
    DATA_CREATED = "data.created"
    DATA_DELETED = "data.deleted"
    SYSTEM_ALERT = "system.alert"
    USER_ACTION = "user.action"
    CUSTOM = "custom"


@dataclass
class WebhookConfig:
    """Configuration for webhook endpoints"""
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    signature_header: str = "X-Webhook-Signature"
    timestamp_header: str = "X-Webhook-Timestamp"
    event_header: str = "X-Webhook-Event"
    
    def __post_init__(self):
        # Validate URL
        parsed = urlparse(self.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid webhook URL: {self.url}")


@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event: WebhookEvent
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    webhook_id: Optional[str] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event': self.event.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'webhook_id': self.webhook_id,
            'source': self.source
        }


@dataclass
class WebhookDelivery:
    """Webhook delivery tracking"""
    webhook_id: str
    payload: WebhookPayload
    config: WebhookConfig
    attempt: int = 1
    status: str = "pending"
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None


class WebhookValidator:
    """Validates incoming webhook requests"""
    
    @staticmethod
    def validate_signature(
        payload: bytes,
        signature: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> bool:
        """Validate webhook signature"""
        if not secret or not signature:
            return False
        
        # Remove algorithm prefix if present (e.g., "sha256=")
        if "=" in signature:
            signature = signature.split("=", 1)[1]
        
        # Calculate expected signature
        expected = hmac.new(
            secret.encode(),
            payload,
            getattr(hashlib, algorithm)
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)
    
    @staticmethod
    def validate_timestamp(
        timestamp_str: str,
        tolerance_seconds: int = 300
    ) -> bool:
        """Validate webhook timestamp to prevent replay attacks"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.utcnow()
            age = (now - timestamp.replace(tzinfo=None)).total_seconds()
            return abs(age) <= tolerance_seconds
        except (ValueError, TypeError):
            return False


class WebhookRegistry:
    """Registry for managing webhook subscriptions"""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_subscriptions: Dict[WebhookEvent, Set[str]] = {}
        self.delivery_history: List[WebhookDelivery] = []
        self.failed_deliveries: List[WebhookDelivery] = []
    
    def register_webhook(
        self,
        webhook_id: str,
        config: WebhookConfig
    ) -> None:
        """Register a new webhook"""
        self.webhooks[webhook_id] = config
        
        # Update event subscriptions
        for event in config.events:
            if event not in self.event_subscriptions:
                self.event_subscriptions[event] = set()
            self.event_subscriptions[event].add(webhook_id)
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook"""
        if webhook_id not in self.webhooks:
            return False
        
        config = self.webhooks[webhook_id]
        
        # Remove from event subscriptions
        for event in config.events:
            if event in self.event_subscriptions:
                self.event_subscriptions[event].discard(webhook_id)
                if not self.event_subscriptions[event]:
                    del self.event_subscriptions[event]
        
        del self.webhooks[webhook_id]
        return True
    
    def get_subscribers(self, event: WebhookEvent) -> List[str]:
        """Get webhook IDs subscribed to an event"""
        return list(self.event_subscriptions.get(event, set()))
    
    def get_webhook_config(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration"""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self) -> Dict[str, WebhookConfig]:
        """List all registered webhooks"""
        return self.webhooks.copy()


class WebhookDeliveryService:
    """Service for delivering webhooks"""
    
    def __init__(self, registry: WebhookRegistry):
        self.registry = registry
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the delivery service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._delivery_worker())
    
    async def stop(self):
        """Stop the delivery service"""
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
    
    async def send_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> List[WebhookDelivery]:
        """Send webhook to all subscribers"""
        payload = WebhookPayload(
            event=event,
            data=data,
            source=source
        )
        
        subscribers = self.registry.get_subscribers(event)
        deliveries = []
        
        for webhook_id in subscribers:
            config = self.registry.get_webhook_config(webhook_id)
            if config:
                delivery = WebhookDelivery(
                    webhook_id=webhook_id,
                    payload=payload,
                    config=config
                )
                deliveries.append(delivery)
                await self.delivery_queue.put(delivery)
        
        return deliveries
    
    async def _delivery_worker(self):
        """Background worker for processing webhook deliveries"""
        while self.is_running:
            try:
                delivery = await asyncio.wait_for(
                    self.delivery_queue.get(),
                    timeout=1.0
                )
                await self._deliver_webhook(delivery)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Webhook delivery worker error: {e}")
    
    async def _deliver_webhook(self, delivery: WebhookDelivery):
        """Deliver a single webhook"""
        config = delivery.config
        payload_dict = delivery.payload.to_dict()
        payload_dict['webhook_id'] = delivery.webhook_id
        
        headers = config.headers.copy()
        headers['Content-Type'] = 'application/json'
        headers[config.event_header] = delivery.payload.event.value
        headers[config.timestamp_header] = delivery.payload.timestamp.isoformat()
        
        # Add signature if secret is configured
        payload_bytes = json.dumps(payload_dict).encode()
        if config.secret:
            signature = hmac.new(
                config.secret.encode(),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            headers[config.signature_header] = f"sha256={signature}"
        
        for attempt in range(1, config.max_retries + 1):
            delivery.attempt = attempt
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        config.url,
                        data=payload_bytes,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=config.timeout)
                    ) as response:
                        delivery.response_code = response.status
                        delivery.response_body = await response.text()
                        
                        if 200 <= response.status < 300:
                            delivery.status = "delivered"
                            delivery.delivered_at = datetime.utcnow()
                            self.registry.delivery_history.append(delivery)
                            return
                        else:
                            delivery.error_message = f"HTTP {response.status}: {delivery.response_body}"
            
            except Exception as e:
                delivery.error_message = str(e)
            
            # Retry with delay if not the last attempt
            if attempt < config.max_retries:
                await asyncio.sleep(config.retry_delay * attempt)
        
        # All attempts failed
        delivery.status = "failed"
        self.registry.failed_deliveries.append(delivery)
    
    async def retry_failed_deliveries(self) -> int:
        """Retry all failed deliveries"""
        failed_count = len(self.registry.failed_deliveries)
        
        for delivery in self.registry.failed_deliveries.copy():
            delivery.attempt = 1  # Reset attempt counter
            delivery.status = "pending"
            delivery.error_message = None
            await self.delivery_queue.put(delivery)
        
        self.registry.failed_deliveries.clear()
        return failed_count


class WebhookReceiver:
    """Handles incoming webhook requests"""
    
    def __init__(self, app: FastAPI, registry: WebhookRegistry):
        self.app = app
        self.registry = registry
        self.handlers: Dict[WebhookEvent, List[Callable]] = {}
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes for webhook endpoints"""
        
        @self.app.post("/webhooks/{webhook_id}")
        async def receive_webhook(webhook_id: str, request: Request):
            """Receive and process incoming webhook"""
            config = self.registry.get_webhook_config(webhook_id)
            if not config:
                raise HTTPException(status_code=404, detail="Webhook not found")
            
            # Get request data
            body = await request.body()
            headers = dict(request.headers)
            
            # Validate signature if secret is configured
            if config.secret:
                signature = headers.get(config.signature_header.lower())
                if not signature or not WebhookValidator.validate_signature(
                    body, signature, config.secret
                ):
                    raise HTTPException(status_code=401, detail="Invalid signature")
            
            # Validate timestamp
            timestamp = headers.get(config.timestamp_header.lower())
            if timestamp and not WebhookValidator.validate_timestamp(timestamp):
                raise HTTPException(status_code=400, detail="Invalid timestamp")
            
            # Parse payload
            try:
                payload_data = json.loads(body)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON payload")
            
            # Extract event type
            event_str = headers.get(config.event_header.lower()) or payload_data.get('event')
            if not event_str:
                raise HTTPException(status_code=400, detail="Missing event type")
            
            try:
                event = WebhookEvent(event_str)
            except ValueError:
                event = WebhookEvent.CUSTOM
            
            # Process webhook
            await self._process_webhook(webhook_id, event, payload_data)
            
            return {"status": "received", "webhook_id": webhook_id}
    
    async def _process_webhook(
        self,
        webhook_id: str,
        event: WebhookEvent,
        data: Dict[str, Any]
    ):
        """Process received webhook"""
        handlers = self.handlers.get(event, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(webhook_id, event, data)
                else:
                    handler(webhook_id, event, data)
            except Exception as e:
                print(f"Webhook handler error: {e}")
    
    def register_handler(
        self,
        event: WebhookEvent,
        handler: Callable[[str, WebhookEvent, Dict[str, Any]], None]
    ):
        """Register a webhook event handler"""
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
    
    def unregister_handler(
        self,
        event: WebhookEvent,
        handler: Callable
    ):
        """Unregister a webhook event handler"""
        if event in self.handlers:
            try:
                self.handlers[event].remove(handler)
            except ValueError:
                pass


class WebhookManager:
    """Main webhook management interface"""
    
    def __init__(self, app: Optional[FastAPI] = None):
        self.registry = WebhookRegistry()
        self.delivery_service = WebhookDeliveryService(self.registry)
        self.receiver = WebhookReceiver(app, self.registry) if app else None
    
    async def start(self):
        """Start webhook services"""
        await self.delivery_service.start()
    
    async def stop(self):
        """Stop webhook services"""
        await self.delivery_service.stop()
    
    def register_webhook(
        self,
        webhook_id: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        **kwargs
    ) -> None:
        """Register a new webhook"""
        config = WebhookConfig(
            url=url,
            events=events,
            secret=secret,
            **kwargs
        )
        self.registry.register_webhook(webhook_id, config)
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook"""
        return self.registry.unregister_webhook(webhook_id)
    
    async def send_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> List[WebhookDelivery]:
        """Send webhook to subscribers"""
        return await self.delivery_service.send_webhook(event, data, source)
    
    def register_handler(
        self,
        event: WebhookEvent,
        handler: Callable[[str, WebhookEvent, Dict[str, Any]], None]
    ):
        """Register webhook event handler"""
        if self.receiver:
            self.receiver.register_handler(event, handler)
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        total_deliveries = len(self.registry.delivery_history)
        failed_deliveries = len(self.registry.failed_deliveries)
        
        return {
            'total_deliveries': total_deliveries,
            'successful_deliveries': total_deliveries,
            'failed_deliveries': failed_deliveries,
            'success_rate': (total_deliveries / (total_deliveries + failed_deliveries)) * 100 
                           if (total_deliveries + failed_deliveries) > 0 else 0
        }