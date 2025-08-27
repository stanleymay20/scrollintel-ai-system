"""
Webhook Manager for Advanced Analytics Dashboard API

This module manages webhook registration, delivery, and monitoring
for external integrations.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import hashlib
import hmac
from urllib.parse import urlparse

from ...models.webhook_models import (
    Webhook, WebhookEvent, WebhookDelivery, WebhookStatus,
    DeliveryStatus, EventType, WebhookConfig
)
from ...core.database import get_database_session

logger = logging.getLogger(__name__)

class WebhookEventType(Enum):
    """Webhook event types"""
    DASHBOARD_CREATED = "dashboard.created"
    DASHBOARD_UPDATED = "dashboard.updated"
    DASHBOARD_DELETED = "dashboard.deleted"
    WIDGET_ADDED = "widget.added"
    WIDGET_UPDATED = "widget.updated"
    WIDGET_REMOVED = "widget.removed"
    INSIGHT_GENERATED = "insight.generated"
    INSIGHT_UPDATED = "insight.updated"
    FORECAST_CREATED = "forecast.created"
    FORECAST_UPDATED = "forecast.updated"
    ROI_ANALYSIS_COMPLETED = "roi.analysis.completed"
    DATA_SOURCE_CONNECTED = "data_source.connected"
    DATA_SOURCE_DISCONNECTED = "data_source.disconnected"
    DATA_SYNC_COMPLETED = "data_sync.completed"
    DATA_SYNC_FAILED = "data_sync.failed"
    ALERT_TRIGGERED = "alert.triggered"
    REPORT_GENERATED = "report.generated"
    SCHEDULE_EXECUTED = "schedule.executed"
    THRESHOLD_BREACHED = "threshold.breached"
    ANOMALY_DETECTED = "anomaly.detected"

@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event_type: str
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class WebhookManager:
    """Manages webhook registration and delivery"""
    
    def __init__(self):
        self.webhooks: Dict[str, Webhook] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.delivery_queue = asyncio.Queue()
        self.retry_queue = asyncio.Queue()
        self.delivery_workers = []
        self.retry_workers = []
        self.is_running = False
        
        # Configuration
        self.max_retries = 5
        self.retry_delays = [1, 5, 15, 60, 300]  # seconds
        self.delivery_timeout = 30  # seconds
        self.max_payload_size = 1024 * 1024  # 1MB
        self.signature_header = "X-Webhook-Signature"
        self.timestamp_header = "X-Webhook-Timestamp"
        
    async def initialize(self):
        """Initialize webhook manager"""
        logger.info("Initializing webhook manager...")
        
        # Load existing webhooks from database
        await self._load_webhooks()
        
        # Start delivery workers
        await self._start_workers()
        
        self.is_running = True
        logger.info("Webhook manager initialized successfully")
    
    async def cleanup(self):
        """Cleanup webhook manager"""
        logger.info("Cleaning up webhook manager...")
        
        self.is_running = False
        
        # Stop workers
        await self._stop_workers()
        
        # Save webhook state
        await self._save_webhooks()
        
        logger.info("Webhook manager cleanup complete")
    
    async def register_webhook(self, config: Dict[str, Any]) -> str:
        """Register a new webhook"""
        try:
            # Validate configuration
            self._validate_webhook_config(config)
            
            # Create webhook
            webhook_id = str(uuid.uuid4())
            webhook = Webhook(
                id=webhook_id,
                url=config['url'],
                events=config['events'],
                secret=config.get('secret'),
                headers=config.get('headers', {}),
                filters=config.get('filters', {}),
                retry_config=config.get('retry_config', {}),
                status=WebhookStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store webhook
            self.webhooks[webhook_id] = webhook
            
            # Save to database
            await self._save_webhook(webhook)
            
            logger.info(f"Webhook registered: {webhook_id} -> {config['url']}")
            return webhook_id
            
        except Exception as e:
            logger.error(f"Error registering webhook: {str(e)}")
            raise
    
    async def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> bool:
        """Update webhook configuration"""
        try:
            if webhook_id not in self.webhooks:
                return False
            
            webhook = self.webhooks[webhook_id]
            
            # Update fields
            if 'url' in updates:
                webhook.url = updates['url']
            if 'events' in updates:
                webhook.events = updates['events']
            if 'secret' in updates:
                webhook.secret = updates['secret']
            if 'headers' in updates:
                webhook.headers = updates['headers']
            if 'filters' in updates:
                webhook.filters = updates['filters']
            if 'retry_config' in updates:
                webhook.retry_config = updates['retry_config']
            if 'status' in updates:
                webhook.status = WebhookStatus(updates['status'])
            
            webhook.updated_at = datetime.utcnow()
            
            # Save to database
            await self._save_webhook(webhook)
            
            logger.info(f"Webhook updated: {webhook_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating webhook {webhook_id}: {str(e)}")
            raise
    
    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook"""
        try:
            if webhook_id not in self.webhooks:
                return False
            
            # Remove from memory
            del self.webhooks[webhook_id]
            
            # Remove from database
            await self._delete_webhook(webhook_id)
            
            logger.info(f"Webhook deleted: {webhook_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting webhook {webhook_id}: {str(e)}")
            raise
    
    async def get_webhook(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook details"""
        try:
            if webhook_id not in self.webhooks:
                return None
            
            webhook = self.webhooks[webhook_id]
            return {
                "id": webhook.id,
                "url": webhook.url,
                "events": webhook.events,
                "headers": webhook.headers,
                "filters": webhook.filters,
                "retry_config": webhook.retry_config,
                "status": webhook.status.value,
                "created_at": webhook.created_at.isoformat(),
                "updated_at": webhook.updated_at.isoformat(),
                "last_delivery": webhook.last_delivery.isoformat() if webhook.last_delivery else None,
                "delivery_count": webhook.delivery_count,
                "failure_count": webhook.failure_count
            }
            
        except Exception as e:
            logger.error(f"Error getting webhook {webhook_id}: {str(e)}")
            raise
    
    async def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all webhooks"""
        try:
            webhooks = []
            for webhook in self.webhooks.values():
                webhooks.append({
                    "id": webhook.id,
                    "url": webhook.url,
                    "events": webhook.events,
                    "status": webhook.status.value,
                    "created_at": webhook.created_at.isoformat(),
                    "last_delivery": webhook.last_delivery.isoformat() if webhook.last_delivery else None,
                    "delivery_count": webhook.delivery_count,
                    "failure_count": webhook.failure_count
                })
            
            return webhooks
            
        except Exception as e:
            logger.error(f"Error listing webhooks: {str(e)}")
            raise
    
    async def trigger_event(self, event_type: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Trigger a webhook event"""
        try:
            # Create event payload
            payload = WebhookPayload(
                event_type=event_type,
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                data=data,
                metadata=metadata
            )
            
            # Find matching webhooks
            matching_webhooks = []
            for webhook in self.webhooks.values():
                if webhook.status == WebhookStatus.ACTIVE and self._matches_webhook(webhook, payload):
                    matching_webhooks.append(webhook)
            
            # Queue deliveries
            for webhook in matching_webhooks:
                await self.delivery_queue.put((webhook, payload))
            
            logger.debug(f"Event triggered: {event_type} -> {len(matching_webhooks)} webhooks")
            
        except Exception as e:
            logger.error(f"Error triggering event {event_type}: {str(e)}")
            raise
    
    async def test_webhook(self, webhook_id: str, test_payload: Optional[Dict] = None) -> Dict[str, Any]:
        """Test a webhook with a test payload"""
        try:
            if webhook_id not in self.webhooks:
                raise ValueError("Webhook not found")
            
            webhook = self.webhooks[webhook_id]
            
            # Create test payload
            if test_payload is None:
                test_payload = {
                    "test": True,
                    "message": "This is a test webhook delivery",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            payload = WebhookPayload(
                event_type="webhook.test",
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                data=test_payload
            )
            
            # Attempt delivery
            delivery_result = await self._deliver_webhook(webhook, payload)
            
            return {
                "webhook_id": webhook_id,
                "test_successful": delivery_result.status == DeliveryStatus.SUCCESS,
                "status_code": delivery_result.status_code,
                "response_time": delivery_result.response_time,
                "error": delivery_result.error_message
            }
            
        except Exception as e:
            logger.error(f"Error testing webhook {webhook_id}: {str(e)}")
            raise
    
    async def get_delivery_history(self, webhook_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get delivery history for a webhook"""
        try:
            # Load from database
            deliveries = await self._load_delivery_history(webhook_id, limit)
            
            return [
                {
                    "id": delivery.id,
                    "event_type": delivery.event_type,
                    "status": delivery.status.value,
                    "status_code": delivery.status_code,
                    "response_time": delivery.response_time,
                    "attempt": delivery.attempt,
                    "delivered_at": delivery.delivered_at.isoformat(),
                    "error_message": delivery.error_message
                }
                for delivery in deliveries
            ]
            
        except Exception as e:
            logger.error(f"Error getting delivery history for {webhook_id}: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for webhook manager"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "webhooks_count": len(self.webhooks),
            "active_webhooks": len([w for w in self.webhooks.values() if w.status == WebhookStatus.ACTIVE]),
            "delivery_queue_size": self.delivery_queue.qsize(),
            "retry_queue_size": self.retry_queue.qsize(),
            "workers_running": len(self.delivery_workers) + len(self.retry_workers)
        }
    
    # Private Methods
    
    def _validate_webhook_config(self, config: Dict[str, Any]):
        """Validate webhook configuration"""
        required_fields = ['url', 'events']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate URL
        parsed_url = urlparse(config['url'])
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid webhook URL")
        
        # Validate events
        if not isinstance(config['events'], list) or not config['events']:
            raise ValueError("Events must be a non-empty list")
        
        # Validate event types
        valid_events = [e.value for e in WebhookEventType]
        for event in config['events']:
            if event not in valid_events and event != "*":
                raise ValueError(f"Invalid event type: {event}")
    
    def _matches_webhook(self, webhook: Webhook, payload: WebhookPayload) -> bool:
        """Check if payload matches webhook criteria"""
        # Check event type
        if "*" not in webhook.events and payload.event_type not in webhook.events:
            return False
        
        # Apply filters
        if webhook.filters:
            for filter_key, filter_value in webhook.filters.items():
                if filter_key in payload.data:
                    if payload.data[filter_key] != filter_value:
                        return False
        
        return True
    
    async def _deliver_webhook(self, webhook: Webhook, payload: WebhookPayload) -> 'WebhookDelivery':
        """Deliver webhook payload"""
        delivery_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Prepare payload
            webhook_payload = {
                "event_type": payload.event_type,
                "event_id": payload.event_id,
                "timestamp": payload.timestamp.isoformat(),
                "data": payload.data
            }
            
            if payload.metadata:
                webhook_payload["metadata"] = payload.metadata
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "ScrollIntel-Webhooks/1.0",
                self.timestamp_header: str(int(start_time.timestamp()))
            }
            
            # Add custom headers
            if webhook.headers:
                headers.update(webhook.headers)
            
            # Add signature if secret is provided
            if webhook.secret:
                payload_str = json.dumps(webhook_payload, sort_keys=True)
                signature = self._generate_signature(payload_str, webhook.secret)
                headers[self.signature_header] = signature
            
            # Make HTTP request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.delivery_timeout)) as session:
                async with session.post(
                    webhook.url,
                    json=webhook_payload,
                    headers=headers
                ) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Create delivery record
                    delivery = WebhookDelivery(
                        id=delivery_id,
                        webhook_id=webhook.id,
                        event_type=payload.event_type,
                        event_id=payload.event_id,
                        status=DeliveryStatus.SUCCESS if response.status < 400 else DeliveryStatus.FAILED,
                        status_code=response.status,
                        response_time=response_time,
                        attempt=1,
                        delivered_at=datetime.utcnow(),
                        error_message=None if response.status < 400 else f"HTTP {response.status}"
                    )
                    
                    # Update webhook stats
                    webhook.delivery_count += 1
                    webhook.last_delivery = datetime.utcnow()
                    
                    if delivery.status == DeliveryStatus.FAILED:
                        webhook.failure_count += 1
                    
                    # Save delivery record
                    await self._save_delivery(delivery)
                    
                    return delivery
        
        except asyncio.TimeoutError:
            response_time = (datetime.utcnow() - start_time).total_seconds()
            delivery = WebhookDelivery(
                id=delivery_id,
                webhook_id=webhook.id,
                event_type=payload.event_type,
                event_id=payload.event_id,
                status=DeliveryStatus.FAILED,
                status_code=0,
                response_time=response_time,
                attempt=1,
                delivered_at=datetime.utcnow(),
                error_message="Request timeout"
            )
            
            webhook.delivery_count += 1
            webhook.failure_count += 1
            webhook.last_delivery = datetime.utcnow()
            
            await self._save_delivery(delivery)
            return delivery
        
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds()
            delivery = WebhookDelivery(
                id=delivery_id,
                webhook_id=webhook.id,
                event_type=payload.event_type,
                event_id=payload.event_id,
                status=DeliveryStatus.FAILED,
                status_code=0,
                response_time=response_time,
                attempt=1,
                delivered_at=datetime.utcnow(),
                error_message=str(e)
            )
            
            webhook.delivery_count += 1
            webhook.failure_count += 1
            webhook.last_delivery = datetime.utcnow()
            
            await self._save_delivery(delivery)
            return delivery
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate webhook signature"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def _start_workers(self):
        """Start delivery workers"""
        # Start delivery workers
        for i in range(3):  # 3 delivery workers
            worker = asyncio.create_task(self._delivery_worker(f"delivery-{i}"))
            self.delivery_workers.append(worker)
        
        # Start retry workers
        for i in range(2):  # 2 retry workers
            worker = asyncio.create_task(self._retry_worker(f"retry-{i}"))
            self.retry_workers.append(worker)
        
        logger.info(f"Started {len(self.delivery_workers)} delivery workers and {len(self.retry_workers)} retry workers")
    
    async def _stop_workers(self):
        """Stop delivery workers"""
        # Cancel all workers
        for worker in self.delivery_workers + self.retry_workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.delivery_workers or self.retry_workers:
            await asyncio.gather(
                *self.delivery_workers,
                *self.retry_workers,
                return_exceptions=True
            )
        
        self.delivery_workers.clear()
        self.retry_workers.clear()
        
        logger.info("All webhook workers stopped")
    
    async def _delivery_worker(self, worker_name: str):
        """Delivery worker coroutine"""
        logger.info(f"Delivery worker {worker_name} started")
        
        try:
            while self.is_running:
                try:
                    # Get delivery task
                    webhook, payload = await asyncio.wait_for(
                        self.delivery_queue.get(),
                        timeout=1.0
                    )
                    
                    # Deliver webhook
                    delivery = await self._deliver_webhook(webhook, payload)
                    
                    # If delivery failed and retries are enabled, queue for retry
                    if delivery.status == DeliveryStatus.FAILED and self.max_retries > 0:
                        await self.retry_queue.put((webhook, payload, delivery))
                    
                    # Mark task as done
                    self.delivery_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in delivery worker {worker_name}: {str(e)}")
                    continue
        
        except asyncio.CancelledError:
            logger.info(f"Delivery worker {worker_name} cancelled")
        except Exception as e:
            logger.error(f"Delivery worker {worker_name} error: {str(e)}")
    
    async def _retry_worker(self, worker_name: str):
        """Retry worker coroutine"""
        logger.info(f"Retry worker {worker_name} started")
        
        try:
            while self.is_running:
                try:
                    # Get retry task
                    webhook, payload, original_delivery = await asyncio.wait_for(
                        self.retry_queue.get(),
                        timeout=1.0
                    )
                    
                    # Calculate retry delay
                    attempt = original_delivery.attempt
                    if attempt <= self.max_retries:
                        delay = self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                        await asyncio.sleep(delay)
                        
                        # Retry delivery
                        delivery = await self._deliver_webhook(webhook, payload)
                        delivery.attempt = attempt + 1
                        
                        # If still failed and more retries available, queue again
                        if delivery.status == DeliveryStatus.FAILED and attempt < self.max_retries:
                            await self.retry_queue.put((webhook, payload, delivery))
                    
                    # Mark task as done
                    self.retry_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in retry worker {worker_name}: {str(e)}")
                    continue
        
        except asyncio.CancelledError:
            logger.info(f"Retry worker {worker_name} cancelled")
        except Exception as e:
            logger.error(f"Retry worker {worker_name} error: {str(e)}")
    
    async def _load_webhooks(self):
        """Load webhooks from database"""
        try:
            async with get_database_session() as session:
                # Load webhooks from database
                # This would be implemented based on your database schema
                pass
        except Exception as e:
            logger.error(f"Error loading webhooks: {str(e)}")
    
    async def _save_webhooks(self):
        """Save webhooks to database"""
        try:
            async with get_database_session() as session:
                # Save webhooks to database
                # This would be implemented based on your database schema
                pass
        except Exception as e:
            logger.error(f"Error saving webhooks: {str(e)}")
    
    async def _save_webhook(self, webhook: Webhook):
        """Save single webhook to database"""
        try:
            async with get_database_session() as session:
                # Save webhook to database
                # This would be implemented based on your database schema
                pass
        except Exception as e:
            logger.error(f"Error saving webhook: {str(e)}")
    
    async def _delete_webhook(self, webhook_id: str):
        """Delete webhook from database"""
        try:
            async with get_database_session() as session:
                # Delete webhook from database
                # This would be implemented based on your database schema
                pass
        except Exception as e:
            logger.error(f"Error deleting webhook: {str(e)}")
    
    async def _save_delivery(self, delivery: WebhookDelivery):
        """Save delivery record to database"""
        try:
            async with get_database_session() as session:
                # Save delivery to database
                # This would be implemented based on your database schema
                pass
        except Exception as e:
            logger.error(f"Error saving delivery: {str(e)}")
    
    async def _load_delivery_history(self, webhook_id: str, limit: int) -> List[WebhookDelivery]:
        """Load delivery history from database"""
        try:
            async with get_database_session() as session:
                # Load delivery history from database
                # This would be implemented based on your database schema
                return []
        except Exception as e:
            logger.error(f"Error loading delivery history: {str(e)}")
            return []