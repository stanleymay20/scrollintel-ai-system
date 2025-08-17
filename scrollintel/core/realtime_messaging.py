"""
Real-Time Message Queuing and Event Streaming Infrastructure

This module implements enterprise-grade real-time messaging and event streaming
capabilities using Apache Kafka-like patterns for high-throughput, low-latency
communication between agents and system components.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the streaming system"""
    AGENT_REGISTERED = "agent_registered"
    AGENT_DEREGISTERED = "agent_deregistered"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    PERFORMANCE_METRIC = "performance_metric"
    SYSTEM_ALERT = "system_alert"
    COORDINATION_REQUEST = "coordination_request"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority levels for queue management"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class StreamEvent:
    """Event structure for the streaming system"""
    id: str
    event_type: EventType
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "correlation_id": self.correlation_id,
            "priority": self.priority.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Create event from dictionary"""
        return cls(
            id=data["id"],
            event_type=EventType(data["event_type"]),
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            correlation_id=data.get("correlation_id"),
            priority=MessagePriority(data.get("priority", MessagePriority.NORMAL.value)),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )


@dataclass
class QueuedMessage:
    """Message structure for priority queue"""
    priority: int
    timestamp: datetime
    event: StreamEvent
    
    def __lt__(self, other):
        """Comparison for priority queue (higher priority first, then FIFO)"""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.timestamp < other.timestamp  # FIFO for same priority


class EventSubscription:
    """Subscription configuration for event consumers"""
    
    def __init__(
        self,
        subscriber_id: str,
        event_types: List[EventType],
        callback: Callable[[StreamEvent], None],
        filter_func: Optional[Callable[[StreamEvent], bool]] = None,
        batch_size: int = 1,
        max_batch_wait: float = 1.0
    ):
        self.subscriber_id = subscriber_id
        self.event_types = set(event_types)
        self.callback = callback
        self.filter_func = filter_func
        self.batch_size = batch_size
        self.max_batch_wait = max_batch_wait
        self.created_at = datetime.utcnow()
        self.last_processed = datetime.utcnow()
        self.processed_count = 0
        self.error_count = 0


class PriorityMessageQueue:
    """
    High-performance priority message queue with real-time processing
    
    Implements enterprise-grade message queuing with priority handling,
    dead letter queues, and comprehensive monitoring.
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.queue = []
        self.dead_letter_queue = deque(maxlen=10000)
        self.processing_queue = {}
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stats = {
            "messages_queued": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "queue_size": 0,
            "processing_count": 0
        }
        self.running = False
        self.worker_threads = []
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def start(self, num_workers: int = 5):
        """Start queue processing workers"""
        self.running = True
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(f"worker-{i}",))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
        logger.info(f"Started {num_workers} queue workers")
    
    def stop(self):
        """Stop queue processing"""
        self.running = False
        with self.condition:
            self.condition.notify_all()
        
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Queue processing stopped")
    
    def enqueue(self, event: StreamEvent) -> bool:
        """
        Add event to priority queue
        
        Args:
            event: Event to queue
            
        Returns:
            bool: True if queued successfully, False if queue is full
        """
        with self.lock:
            if len(self.queue) >= self.max_size:
                logger.warning("Queue is full, dropping message")
                return False
            
            queued_message = QueuedMessage(
                priority=event.priority.value,
                timestamp=event.timestamp,
                event=event
            )
            
            heapq.heappush(self.queue, queued_message)
            self.stats["messages_queued"] += 1
            self.stats["queue_size"] = len(self.queue)
            
            # Notify workers
            self.condition.notify()
            
            return True
    
    def dequeue(self, timeout: float = 1.0) -> Optional[StreamEvent]:
        """
        Get next event from queue
        
        Args:
            timeout: Maximum time to wait for event
            
        Returns:
            StreamEvent or None if timeout
        """
        with self.condition:
            end_time = time.time() + timeout
            
            while self.running and not self.queue:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            
            if not self.running or not self.queue:
                return None
            
            queued_message = heapq.heappop(self.queue)
            event = queued_message.event
            
            # Track processing
            self.processing_queue[event.id] = {
                "event": event,
                "started_at": datetime.utcnow()
            }
            
            self.stats["queue_size"] = len(self.queue)
            self.stats["processing_count"] = len(self.processing_queue)
            
            return event
    
    def mark_processed(self, event_id: str, success: bool = True):
        """Mark event as processed"""
        with self.lock:
            if event_id in self.processing_queue:
                processing_info = self.processing_queue.pop(event_id)
                
                if success:
                    self.stats["messages_processed"] += 1
                else:
                    self.stats["messages_failed"] += 1
                    # Move to dead letter queue
                    self.dead_letter_queue.append({
                        "event": processing_info["event"],
                        "failed_at": datetime.utcnow(),
                        "processing_time": (datetime.utcnow() - processing_info["started_at"]).total_seconds()
                    })
                
                self.stats["processing_count"] = len(self.processing_queue)
    
    def _worker_loop(self, worker_id: str):
        """Worker thread loop for processing messages"""
        logger.info(f"Queue worker {worker_id} started")
        
        while self.running:
            try:
                event = self.dequeue(timeout=1.0)
                if event:
                    # Process event (placeholder - actual processing done by subscribers)
                    self.mark_processed(event.id, success=True)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return self.stats.copy()
    
    def get_dead_letter_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages from dead letter queue"""
        return list(self.dead_letter_queue)[-limit:]


class EventStreamProcessor:
    """
    High-throughput event stream processor
    
    Processes events in real-time with support for batching,
    filtering, and parallel processing.
    """
    
    def __init__(self):
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_handlers: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        self.message_queue = PriorityMessageQueue()
        self.batch_buffers: Dict[str, List[StreamEvent]] = defaultdict(list)
        self.batch_timers: Dict[str, float] = {}
        self.running = False
        self.processor_thread = None
        self.stats = {
            "events_processed": 0,
            "events_filtered": 0,
            "batch_deliveries": 0,
            "processing_errors": 0
        }
    
    def start(self):
        """Start event stream processing"""
        self.running = True
        self.message_queue.start()
        self.processor_thread = threading.Thread(target=self._processing_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        logger.info("Event stream processor started")
    
    def stop(self):
        """Stop event stream processing"""
        self.running = False
        self.message_queue.stop()
        
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        
        logger.info("Event stream processor stopped")
    
    def subscribe(
        self,
        subscriber_id: str,
        event_types: List[EventType],
        callback: Callable[[StreamEvent], None],
        filter_func: Optional[Callable[[StreamEvent], bool]] = None,
        batch_size: int = 1,
        max_batch_wait: float = 1.0
    ) -> bool:
        """
        Subscribe to event stream
        
        Args:
            subscriber_id: Unique subscriber identifier
            event_types: List of event types to subscribe to
            callback: Function to call when events are received
            filter_func: Optional filter function for events
            batch_size: Number of events to batch before delivery
            max_batch_wait: Maximum time to wait for batch completion
            
        Returns:
            bool: True if subscription successful
        """
        try:
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                event_types=event_types,
                callback=callback,
                filter_func=filter_func,
                batch_size=batch_size,
                max_batch_wait=max_batch_wait
            )
            
            self.subscriptions[subscriber_id] = subscription
            
            # Register for event types
            for event_type in event_types:
                self.event_handlers[event_type].append(subscription)
            
            logger.info(f"Subscription created for {subscriber_id}: {[et.value for et in event_types]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create subscription for {subscriber_id}: {e}")
            return False
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove subscription"""
        try:
            if subscriber_id not in self.subscriptions:
                return False
            
            subscription = self.subscriptions.pop(subscriber_id)
            
            # Remove from event handlers
            for event_type in subscription.event_types:
                if event_type in self.event_handlers:
                    self.event_handlers[event_type] = [
                        sub for sub in self.event_handlers[event_type]
                        if sub.subscriber_id != subscriber_id
                    ]
            
            # Clear batch buffer
            if subscriber_id in self.batch_buffers:
                del self.batch_buffers[subscriber_id]
            if subscriber_id in self.batch_timers:
                del self.batch_timers[subscriber_id]
            
            logger.info(f"Subscription removed for {subscriber_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove subscription for {subscriber_id}: {e}")
            return False
    
    def publish_event(self, event: StreamEvent) -> bool:
        """
        Publish event to stream
        
        Args:
            event: Event to publish
            
        Returns:
            bool: True if published successfully
        """
        return self.message_queue.enqueue(event)
    
    def _processing_loop(self):
        """Main processing loop for handling events"""
        logger.info("Event processing loop started")
        
        while self.running:
            try:
                # Process events from queue
                event = self.message_queue.dequeue(timeout=0.1)
                if event:
                    self._process_event(event)
                
                # Process batch timeouts
                self._process_batch_timeouts()
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.stats["processing_errors"] += 1
    
    def _process_event(self, event: StreamEvent):
        """Process individual event"""
        try:
            # Find matching subscriptions
            matching_subscriptions = self.event_handlers.get(event.event_type, [])
            
            for subscription in matching_subscriptions:
                try:
                    # Apply filter if present
                    if subscription.filter_func and not subscription.filter_func(event):
                        self.stats["events_filtered"] += 1
                        continue
                    
                    # Handle batching
                    if subscription.batch_size > 1:
                        self._add_to_batch(subscription, event)
                    else:
                        # Immediate delivery
                        subscription.callback(event)
                        subscription.processed_count += 1
                    
                    subscription.last_processed = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(f"Error processing event for {subscription.subscriber_id}: {e}")
                    subscription.error_count += 1
            
            self.stats["events_processed"] += 1
            self.message_queue.mark_processed(event.id, success=True)
            
        except Exception as e:
            logger.error(f"Event processing error: {e}")
            self.message_queue.mark_processed(event.id, success=False)
    
    def _add_to_batch(self, subscription: EventSubscription, event: StreamEvent):
        """Add event to batch buffer"""
        subscriber_id = subscription.subscriber_id
        
        # Initialize batch if needed
        if subscriber_id not in self.batch_buffers:
            self.batch_buffers[subscriber_id] = []
            self.batch_timers[subscriber_id] = time.time()
        
        self.batch_buffers[subscriber_id].append(event)
        
        # Deliver batch if full
        if len(self.batch_buffers[subscriber_id]) >= subscription.batch_size:
            self._deliver_batch(subscription)
    
    def _deliver_batch(self, subscription: EventSubscription):
        """Deliver batch of events to subscriber"""
        subscriber_id = subscription.subscriber_id
        
        if subscriber_id not in self.batch_buffers or not self.batch_buffers[subscriber_id]:
            return
        
        try:
            batch = self.batch_buffers[subscriber_id].copy()
            self.batch_buffers[subscriber_id].clear()
            
            if subscriber_id in self.batch_timers:
                del self.batch_timers[subscriber_id]
            
            # Deliver batch
            for event in batch:
                subscription.callback(event)
            
            subscription.processed_count += len(batch)
            self.stats["batch_deliveries"] += 1
            
        except Exception as e:
            logger.error(f"Batch delivery error for {subscriber_id}: {e}")
            subscription.error_count += 1
    
    def _process_batch_timeouts(self):
        """Process batch timeouts and deliver partial batches"""
        current_time = time.time()
        
        for subscriber_id, start_time in list(self.batch_timers.items()):
            if subscriber_id in self.subscriptions:
                subscription = self.subscriptions[subscriber_id]
                
                if current_time - start_time >= subscription.max_batch_wait:
                    self._deliver_batch(subscription)
    
    def get_subscription_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all subscriptions"""
        stats = {}
        for subscriber_id, subscription in self.subscriptions.items():
            stats[subscriber_id] = {
                "event_types": [et.value for et in subscription.event_types],
                "created_at": subscription.created_at.isoformat(),
                "last_processed": subscription.last_processed.isoformat(),
                "processed_count": subscription.processed_count,
                "error_count": subscription.error_count,
                "batch_size": subscription.batch_size,
                "pending_batch_size": len(self.batch_buffers.get(subscriber_id, []))
            }
        return stats
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics"""
        queue_stats = self.message_queue.get_stats()
        return {
            **self.stats,
            **queue_stats,
            "active_subscriptions": len(self.subscriptions),
            "pending_batches": len(self.batch_buffers)
        }


class RealTimeMessagingSystem:
    """
    Enterprise Real-Time Messaging System
    
    Provides high-performance, scalable messaging infrastructure for
    agent orchestration with comprehensive monitoring and management.
    """
    
    def __init__(self):
        self.event_processor = EventStreamProcessor()
        self.system_id = str(uuid.uuid4())
        self.started_at = None
        
    def start(self):
        """Start the messaging system"""
        self.started_at = datetime.utcnow()
        self.event_processor.start()
        logger.info(f"Real-time messaging system started: {self.system_id}")
    
    def stop(self):
        """Stop the messaging system"""
        self.event_processor.stop()
        logger.info(f"Real-time messaging system stopped: {self.system_id}")
    
    def create_event(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> StreamEvent:
        """Create a new stream event"""
        return StreamEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            timestamp=datetime.utcnow(),
            data=data,
            priority=priority,
            correlation_id=correlation_id
        )
    
    def publish(self, event: StreamEvent) -> bool:
        """Publish event to the stream"""
        return self.event_processor.publish_event(event)
    
    def subscribe(
        self,
        subscriber_id: str,
        event_types: List[EventType],
        callback: Callable[[StreamEvent], None],
        **kwargs
    ) -> bool:
        """Subscribe to event stream"""
        return self.event_processor.subscribe(
            subscriber_id, event_types, callback, **kwargs
        )
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from event stream"""
        return self.event_processor.unsubscribe(subscriber_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        processing_stats = self.event_processor.get_processing_stats()
        subscription_stats = self.event_processor.get_subscription_stats()
        
        return {
            "system_id": self.system_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": (datetime.utcnow() - self.started_at).total_seconds() if self.started_at else 0,
            "processing_stats": processing_stats,
            "subscription_count": len(subscription_stats),
            "subscription_stats": subscription_stats
        }
    
    # Convenience methods for common event types
    
    def publish_agent_registered(self, agent_id: str, agent_info: Dict[str, Any]):
        """Publish agent registration event"""
        event = self.create_event(
            EventType.AGENT_REGISTERED,
            "orchestration_system",
            {"agent_id": agent_id, "agent_info": agent_info}
        )
        return self.publish(event)
    
    def publish_task_assigned(self, task_id: str, agent_id: str, task_info: Dict[str, Any]):
        """Publish task assignment event"""
        event = self.create_event(
            EventType.TASK_ASSIGNED,
            "orchestration_system",
            {"task_id": task_id, "agent_id": agent_id, "task_info": task_info},
            priority=MessagePriority.HIGH
        )
        return self.publish(event)
    
    def publish_performance_metric(self, source: str, metrics: Dict[str, Any]):
        """Publish performance metrics"""
        event = self.create_event(
            EventType.PERFORMANCE_METRIC,
            source,
            metrics
        )
        return self.publish(event)
    
    def publish_system_alert(self, source: str, alert_data: Dict[str, Any], priority: MessagePriority = MessagePriority.HIGH):
        """Publish system alert"""
        event = self.create_event(
            EventType.SYSTEM_ALERT,
            source,
            alert_data,
            priority=priority
        )
        return self.publish(event)