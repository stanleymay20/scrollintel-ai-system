"""
Webhook system for external integrations.
Handles incoming webhooks and outgoing webhook notifications.
"""
import asyncio
import json
import hmac
import hashlib
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import aiohttp
import logging
from urllib.parse import urlparse

from ..models.database import get_db_session
from ..models.webhook_models import WebhookEndpoint, WebhookEvent, WebhookDelivery
from ..core.rate_limiter import rate_limit


logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """Webhook event types."""
    DASHBOARD_CREATED = "dashboard.created"
    DASHBOARD_UPDATED = "dashboard.updated"
    DASHBOARD_DELETED = "dashboard.deleted"
    METRIC_UPDATED = "metric.updated"
    INSIGHT_GENERATED = "insight.generated"
    ROI_CALCULATED = "roi.calculated"
    FORECAST_CREATED = "forecast.created"
    DATA_SOURCE_CONNECTED = "data_source.connected"
    DATA_SOURCE_DISCONNECTED = "data_source.disconnected"
    ALERT_TRIGGERED = "alert.triggered"


@dataclass
class WebhookPayload:
    """Webhook payload structure."""
    event_type: str
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class WebhookEndpointConfig:
    """Webhook endpoint configuration."""
    url: str
    secret: str
    events: List[str]
    active: bool = True
    retry_count: int = 3
    timeout: int = 30
    headers: Dict[str, str] = None


class WebhookManager:
    """Manages webhook endpoints and deliveries."""
    
    def __init__(self):
        self.endpoints: Dict[str, WebhookEndpointConfig] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.delivery_queue = asyncio.Queue()
        self.retry_queue = asyncio.Queue()
        self._running = False
        
    async def start(self):
        """Start webhook delivery workers."""
        if self._running:
            return
            
        self._running = True
        
        # Start delivery workers
        asyncio.create_task(self._delivery_worker())
        asyncio.create_task(self._retry_worker())
        
        logger.info("Webhook manager started")
    
    async def stop(self):
        """Stop webhook delivery workers."""
        self._running = False
        logger.info("Webhook manager stopped")
    
    async def register_endpoint(
        self,
        endpoint_id: str,
        config: WebhookEndpointConfig,
        user_id: str
    ) -> bool:
        """Register a new webhook endpoint."""
        try:
            # Validate URL
            parsed_url = urlparse(config.url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid webhook URL")
            
            # Validate events
            valid_events = [e.value for e in WebhookEventType]
            for event in config.events:
                if event not in valid_events:
                    raise ValueError(f"Invalid event type: {event}")
            
            # Store endpoint configuration
            async with get_db_session() as session:
                endpoint = WebhookEndpoint(
                    id=endpoint_id,
                    user_id=user_id,
                    url=config.url,
                    secret=config.secret,
                    events=config.events,
                    active=config.active,
                    retry_count=config.retry_count,
                    timeout=config.timeout,
                    headers=config.headers or {},
                    created_at=datetime.utcnow()
                )
                
                session.add(endpoint)
                await session.commit()
            
            # Cache endpoint
            self.endpoints[endpoint_id] = config
            
            logger.info(f"Registered webhook endpoint: {endpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register webhook endpoint: {e}")
            return False
    
    async def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister a webhook endpoint."""
        try:
            async with get_db_session() as session:
                endpoint = await session.get(WebhookEndpoint, endpoint_id)
                if endpoint:
                    await session.delete(endpoint)
                    await session.commit()
            
            # Remove from cache
            self.endpoints.pop(endpoint_id, None)
            
            logger.info(f"Unregistered webhook endpoint: {endpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister webhook endpoint: {e}")
            return False
    
    async def list_endpoints(self, user_id: str) -> List[Dict[str, Any]]:
        """List webhook endpoints for a user."""
        try:
            async with get_db_session() as session:
                endpoints = await session.execute(
                    "SELECT * FROM webhook_endpoints WHERE user_id = ?",
                    (user_id,)
                )
                
                return [
                    {
                        "id": endpoint.id,
                        "url": endpoint.url,
                        "events": endpoint.events,
                        "active": endpoint.active,
                        "created_at": endpoint.created_at.isoformat(),
                        "last_delivery": endpoint.last_delivery.isoformat() if endpoint.last_delivery else None,
                        "delivery_count": endpoint.delivery_count,
                        "failure_count": endpoint.failure_count
                    }
                    for endpoint in endpoints
                ]
                
        except Exception as e:
            logger.error(f"Failed to list webhook endpoints: {e}")
            return []
    
    async def emit_event(
        self,
        event_type: WebhookEventType,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Emit a webhook event."""
        try:
            payload = WebhookPayload(
                event_type=event_type.value,
                event_id=f"evt_{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                data=data,
                metadata=metadata or {}
            )
            
            # Store event
            async with get_db_session() as session:
                event = WebhookEvent(
                    id=payload.event_id,
                    event_type=payload.event_type,
                    user_id=user_id,
                    payload=payload.data,
                    metadata=payload.metadata,
                    created_at=payload.timestamp
                )
                
                session.add(event)
                await session.commit()
            
            # Queue for delivery
            await self.delivery_queue.put((payload, user_id))
            
            # Call local event handlers
            handlers = self.event_handlers.get(event_type.value, [])
            for handler in handlers:
                try:
                    await handler(payload)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
            
            logger.debug(f"Emitted webhook event: {event_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to emit webhook event: {e}")
    
    def register_event_handler(
        self,
        event_type: WebhookEventType,
        handler: Callable[[WebhookPayload], None]
    ):
        """Register a local event handler."""
        if event_type.value not in self.event_handlers:
            self.event_handlers[event_type.value] = []
        
        self.event_handlers[event_type.value].append(handler)
        logger.info(f"Registered event handler for: {event_type.value}")
    
    async def _delivery_worker(self):
        """Worker to deliver webhook events."""
        while self._running:
            try:
                # Get next event from queue
                payload, user_id = await asyncio.wait_for(
                    self.delivery_queue.get(),
                    timeout=1.0
                )
                
                # Find matching endpoints
                endpoints = await self._get_matching_endpoints(payload.event_type, user_id)
                
                # Deliver to each endpoint
                for endpoint_id, config in endpoints.items():
                    await self._deliver_webhook(endpoint_id, config, payload)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Delivery worker error: {e}")
    
    async def _retry_worker(self):
        """Worker to retry failed webhook deliveries."""
        while self._running:
            try:
                # Get next retry from queue
                delivery_id, attempt = await asyncio.wait_for(
                    self.retry_queue.get(),
                    timeout=5.0
                )
                
                # Wait before retry (exponential backoff)
                wait_time = min(2 ** attempt, 300)  # Max 5 minutes
                await asyncio.sleep(wait_time)
                
                # Retry delivery
                await self._retry_delivery(delivery_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Retry worker error: {e}")
    
    async def _get_matching_endpoints(
        self,
        event_type: str,
        user_id: Optional[str] = None
    ) -> Dict[str, WebhookEndpointConfig]:
        """Get endpoints that should receive this event."""
        matching = {}
        
        try:
            async with get_db_session() as session:
                query = """
                    SELECT * FROM webhook_endpoints 
                    WHERE active = true 
                    AND ? = ANY(events)
                """
                params = [event_type]
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                endpoints = await session.execute(query, params)
                
                for endpoint in endpoints:
                    config = WebhookEndpointConfig(
                        url=endpoint.url,
                        secret=endpoint.secret,
                        events=endpoint.events,
                        active=endpoint.active,
                        retry_count=endpoint.retry_count,
                        timeout=endpoint.timeout,
                        headers=endpoint.headers
                    )
                    matching[endpoint.id] = config
                    
        except Exception as e:
            logger.error(f"Failed to get matching endpoints: {e}")
        
        return matching
    
    async def _deliver_webhook(
        self,
        endpoint_id: str,
        config: WebhookEndpointConfig,
        payload: WebhookPayload
    ):
        """Deliver webhook to a specific endpoint."""
        delivery_id = f"del_{datetime.utcnow().timestamp()}"
        
        try:
            # Prepare payload
            webhook_payload = {
                "event_type": payload.event_type,
                "event_id": payload.event_id,
                "timestamp": payload.timestamp.isoformat(),
                "data": payload.data,
                "metadata": payload.metadata
            }
            
            # Create signature
            signature = self._create_signature(
                json.dumps(webhook_payload, sort_keys=True),
                config.secret
            )
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-Event": payload.event_type,
                "X-Webhook-ID": payload.event_id,
                "User-Agent": "ScrollIntel-Webhooks/1.0"
            }
            
            if config.headers:
                headers.update(config.headers)
            
            # Store delivery attempt
            async with get_db_session() as session:
                delivery = WebhookDelivery(
                    id=delivery_id,
                    endpoint_id=endpoint_id,
                    event_id=payload.event_id,
                    status="pending",
                    attempt=1,
                    created_at=datetime.utcnow()
                )
                
                session.add(delivery)
                await session.commit()
            
            # Make HTTP request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                async with session.post(
                    config.url,
                    json=webhook_payload,
                    headers=headers
                ) as response:
                    success = 200 <= response.status < 300
                    
                    # Update delivery status
                    await self._update_delivery_status(
                        delivery_id,
                        "success" if success else "failed",
                        response.status,
                        await response.text() if not success else None
                    )
                    
                    if success:
                        logger.debug(f"Webhook delivered successfully: {endpoint_id}")
                    else:
                        logger.warning(f"Webhook delivery failed: {endpoint_id}, status: {response.status}")
                        
                        # Queue for retry if not too many attempts
                        if delivery.attempt < config.retry_count:
                            await self.retry_queue.put((delivery_id, delivery.attempt))
        
        except Exception as e:
            logger.error(f"Webhook delivery error: {e}")
            
            # Update delivery status
            await self._update_delivery_status(
                delivery_id,
                "failed",
                0,
                str(e)
            )
            
            # Queue for retry
            await self.retry_queue.put((delivery_id, 1))
    
    async def _retry_delivery(self, delivery_id: str):
        """Retry a failed webhook delivery."""
        try:
            async with get_db_session() as session:
                delivery = await session.get(WebhookDelivery, delivery_id)
                if not delivery or delivery.status == "success":
                    return
                
                # Get endpoint and event
                endpoint = await session.get(WebhookEndpoint, delivery.endpoint_id)
                event = await session.get(WebhookEvent, delivery.event_id)
                
                if not endpoint or not event:
                    return
                
                # Recreate payload
                payload = WebhookPayload(
                    event_type=event.event_type,
                    event_id=event.id,
                    timestamp=event.created_at,
                    data=event.payload,
                    metadata=event.metadata
                )
                
                # Create config
                config = WebhookEndpointConfig(
                    url=endpoint.url,
                    secret=endpoint.secret,
                    events=endpoint.events,
                    retry_count=endpoint.retry_count,
                    timeout=endpoint.timeout,
                    headers=endpoint.headers
                )
                
                # Update attempt count
                delivery.attempt += 1
                await session.commit()
                
                # Retry delivery
                await self._deliver_webhook(endpoint.id, config, payload)
                
        except Exception as e:
            logger.error(f"Retry delivery error: {e}")
    
    async def _update_delivery_status(
        self,
        delivery_id: str,
        status: str,
        response_code: int,
        response_body: Optional[str] = None
    ):
        """Update webhook delivery status."""
        try:
            async with get_db_session() as session:
                delivery = await session.get(WebhookDelivery, delivery_id)
                if delivery:
                    delivery.status = status
                    delivery.response_code = response_code
                    delivery.response_body = response_body
                    delivery.delivered_at = datetime.utcnow()
                    
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update delivery status: {e}")
    
    def _create_signature(self, payload: str, secret: str) -> str:
        """Create HMAC signature for webhook payload."""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    async def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """Verify webhook signature."""
        try:
            expected_signature = self._create_signature(payload, secret)
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
    
    async def get_delivery_stats(self, endpoint_id: str) -> Dict[str, Any]:
        """Get delivery statistics for an endpoint."""
        try:
            async with get_db_session() as session:
                stats = await session.execute("""
                    SELECT 
                        COUNT(*) as total_deliveries,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_deliveries,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_deliveries,
                        AVG(CASE WHEN status = 'success' THEN response_code END) as avg_response_time
                    FROM webhook_deliveries 
                    WHERE endpoint_id = ?
                    AND created_at > ?
                """, (endpoint_id, datetime.utcnow() - timedelta(days=30)))
                
                result = stats.fetchone()
                
                return {
                    "total_deliveries": result.total_deliveries or 0,
                    "successful_deliveries": result.successful_deliveries or 0,
                    "failed_deliveries": result.failed_deliveries or 0,
                    "success_rate": (result.successful_deliveries / result.total_deliveries * 100) if result.total_deliveries else 0,
                    "avg_response_time": result.avg_response_time or 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get delivery stats: {e}")
            return {}


# Global webhook manager instance
webhook_manager = WebhookManager()