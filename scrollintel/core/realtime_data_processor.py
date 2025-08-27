"""
Real-time Data Processing Pipeline for Advanced Analytics Dashboard
Handles streaming data ingestion, processing, and real-time analytics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.dashboard_models import BusinessMetric, Dashboard, Alert
from ..core.websocket_manager import WebSocketManager
from ..engines.predictive_engine import PredictiveEngine
from ..engines.insight_generator import InsightGenerator

logger = logging.getLogger(__name__)

class StreamType(Enum):
    METRICS = "metrics"
    EVENTS = "events"
    ALERTS = "alerts"
    INSIGHTS = "insights"

@dataclass
class StreamMessage:
    """Real-time stream message structure"""
    id: str
    stream_type: StreamType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'stream_type': self.stream_type.value
        }

class RealTimeDataProcessor:
    """
    Real-time data processing pipeline with streaming capabilities
    """
    
    def __init__(self, redis_client: redis.Redis, websocket_manager: WebSocketManager,
                 predictive_engine: PredictiveEngine, insight_generator: InsightGenerator):
        self.redis_client = redis_client
        self.websocket_manager = websocket_manager
        self.predictive_engine = predictive_engine
        self.insight_generator = insight_generator
        
        # Stream configurations
        self.stream_configs = {
            StreamType.METRICS: {
                'buffer_size': 1000,
                'batch_interval': 5,  # seconds
                'retention_hours': 24
            },
            StreamType.EVENTS: {
                'buffer_size': 500,
                'batch_interval': 2,
                'retention_hours': 72
            },
            StreamType.ALERTS: {
                'buffer_size': 100,
                'batch_interval': 1,
                'retention_hours': 168  # 1 week
            }
        }
        
        # Processing queues
        self.processing_queues = {
            stream_type: asyncio.Queue(maxsize=config['buffer_size'])
            for stream_type, config in self.stream_configs.items()
        }
        
        # Event handlers
        self.event_handlers: Dict[StreamType, List[Callable]] = {
            StreamType.METRICS: [],
            StreamType.EVENTS: [],
            StreamType.ALERTS: [],
            StreamType.INSIGHTS: []
        }
        
        self._running = False
        self._tasks = []
    
    async def start(self):
        """Start the real-time processing pipeline"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting real-time data processor")
        
        # Start processing tasks for each stream type
        for stream_type in StreamType:
            task = asyncio.create_task(self._process_stream(stream_type))
            self._tasks.append(task)
        
        # Start batch processing task
        batch_task = asyncio.create_task(self._batch_processor())
        self._tasks.append(batch_task)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_expired_data())
        self._tasks.append(cleanup_task)
    
    async def stop(self):
        """Stop the real-time processing pipeline"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("Real-time data processor stopped")
    
    async def ingest_data(self, message: StreamMessage) -> bool:
        """
        Ingest data into the real-time processing pipeline
        
        Args:
            message: Stream message to process
            
        Returns:
            bool: Success status
        """
        try:
            # Validate message
            if not self._validate_message(message):
                logger.warning(f"Invalid message format: {message.id}")
                return False
            
            # Add to processing queue
            queue = self.processing_queues.get(message.stream_type)
            if queue:
                await queue.put(message)
                
                # Store in Redis for persistence
                await self._store_in_redis(message)
                
                # Trigger immediate processing for high-priority messages
                if message.priority >= 5:
                    await self._process_high_priority_message(message)
                
                return True
            else:
                logger.error(f"No queue found for stream type: {message.stream_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting data: {str(e)}")
            return False
    
    async def _process_stream(self, stream_type: StreamType):
        """Process messages from a specific stream"""
        queue = self.processing_queues[stream_type]
        config = self.stream_configs[stream_type]
        
        batch = []
        last_batch_time = datetime.now()
        
        while self._running:
            try:
                # Get message with timeout
                try:
                    message = await asyncio.wait_for(
                        queue.get(), 
                        timeout=config['batch_interval']
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if conditions are met
                current_time = datetime.now()
                time_elapsed = (current_time - last_batch_time).total_seconds()
                
                if (len(batch) >= 10 or 
                    time_elapsed >= config['batch_interval'] or
                    any(msg.priority >= 5 for msg in batch)):
                    
                    if batch:
                        await self._process_batch(stream_type, batch)
                        batch.clear()
                        last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Error processing {stream_type.value} stream: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, stream_type: StreamType, messages: List[StreamMessage]):
        """Process a batch of messages"""
        try:
            logger.debug(f"Processing batch of {len(messages)} {stream_type.value} messages")
            
            if stream_type == StreamType.METRICS:
                await self._process_metrics_batch(messages)
            elif stream_type == StreamType.EVENTS:
                await self._process_events_batch(messages)
            elif stream_type == StreamType.ALERTS:
                await self._process_alerts_batch(messages)
            
            # Trigger event handlers
            for handler in self.event_handlers.get(stream_type, []):
                try:
                    await handler(messages)
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")
            
            # Send real-time updates via WebSocket
            await self._broadcast_updates(stream_type, messages)
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    
    async def _process_metrics_batch(self, messages: List[StreamMessage]):
        """Process metrics batch for analytics and predictions"""
        metrics_data = []
        
        for message in messages:
            try:
                metric_data = message.data
                metrics_data.append({
                    'timestamp': message.timestamp,
                    'source': message.source,
                    'value': metric_data.get('value'),
                    'metric_name': metric_data.get('name'),
                    'category': metric_data.get('category'),
                    'context': metric_data.get('context', {})
                })
            except Exception as e:
                logger.error(f"Error processing metric message {message.id}: {str(e)}")
        
        if metrics_data:
            # Update predictive models
            await self._update_predictions(metrics_data)
            
            # Generate insights
            await self._generate_real_time_insights(metrics_data)
    
    async def _process_events_batch(self, messages: List[StreamMessage]):
        """Process events batch for pattern detection"""
        events_data = []
        
        for message in messages:
            events_data.append({
                'timestamp': message.timestamp,
                'source': message.source,
                'event_type': message.data.get('type'),
                'data': message.data
            })
        
        # Detect patterns and anomalies
        await self._detect_event_patterns(events_data)
    
    async def _process_alerts_batch(self, messages: List[StreamMessage]):
        """Process alerts batch for escalation and notification"""
        for message in messages:
            alert_data = message.data
            
            # Create alert record
            await self._create_alert_record(
                alert_type=alert_data.get('type'),
                severity=alert_data.get('severity'),
                message=alert_data.get('message'),
                source=message.source,
                timestamp=message.timestamp,
                context=alert_data.get('context', {})
            )
    
    async def _process_high_priority_message(self, message: StreamMessage):
        """Process high-priority messages immediately"""
        try:
            if message.stream_type == StreamType.ALERTS:
                # Immediate alert processing
                await self._handle_critical_alert(message)
            elif message.stream_type == StreamType.METRICS:
                # Real-time metric analysis
                await self._analyze_critical_metric(message)
            
            # Immediate WebSocket broadcast
            await self.websocket_manager.broadcast_to_dashboards({
                'type': 'high_priority_update',
                'data': message.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Error processing high-priority message: {str(e)}")
    
    async def _update_predictions(self, metrics_data: List[Dict[str, Any]]):
        """Update predictive models with new metrics data"""
        try:
            # Group metrics by type for batch prediction updates
            metrics_by_type = {}
            for metric in metrics_data:
                metric_name = metric.get('metric_name')
                if metric_name:
                    if metric_name not in metrics_by_type:
                        metrics_by_type[metric_name] = []
                    metrics_by_type[metric_name].append(metric)
            
            # Update predictions for each metric type
            for metric_name, metric_list in metrics_by_type.items():
                await self.predictive_engine.update_real_time_predictions(
                    metric_name, metric_list
                )
                
        except Exception as e:
            logger.error(f"Error updating predictions: {str(e)}")
    
    async def _generate_real_time_insights(self, metrics_data: List[Dict[str, Any]]):
        """Generate real-time insights from metrics data"""
        try:
            insights = await self.insight_generator.analyze_real_time_data(metrics_data)
            
            if insights:
                # Create insight messages
                for insight in insights:
                    insight_message = StreamMessage(
                        id=f"insight_{datetime.now().timestamp()}",
                        stream_type=StreamType.INSIGHTS,
                        timestamp=datetime.now(),
                        source="insight_generator",
                        data=insight,
                        priority=insight.get('significance', 1)
                    )
                    
                    # Broadcast insight
                    await self.websocket_manager.broadcast_to_dashboards({
                        'type': 'new_insight',
                        'data': insight_message.to_dict()
                    })
                    
        except Exception as e:
            logger.error(f"Error generating real-time insights: {str(e)}")
    
    async def _detect_event_patterns(self, events_data: List[Dict[str, Any]]):
        """Detect patterns in event data"""
        try:
            # Simple pattern detection - can be enhanced with ML
            event_counts = {}
            for event in events_data:
                event_type = event.get('event_type')
                if event_type:
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Check for unusual patterns
            for event_type, count in event_counts.items():
                if count > 10:  # Threshold for unusual activity
                    await self._create_pattern_alert(event_type, count, events_data)
                    
        except Exception as e:
            logger.error(f"Error detecting event patterns: {str(e)}")
    
    async def _create_alert_record(self, alert_type: str, severity: str, 
                                 message: str, source: str, timestamp: datetime,
                                 context: Dict[str, Any]):
        """Create alert record in database"""
        try:
            # This would typically use a database session
            # For now, we'll store in Redis and broadcast
            alert_data = {
                'type': alert_type,
                'severity': severity,
                'message': message,
                'source': source,
                'timestamp': timestamp.isoformat(),
                'context': context
            }
            
            # Store in Redis
            alert_key = f"alert:{datetime.now().timestamp()}"
            await self.redis_client.setex(
                alert_key, 
                timedelta(days=7).total_seconds(),
                json.dumps(alert_data)
            )
            
            # Broadcast alert
            await self.websocket_manager.broadcast_to_dashboards({
                'type': 'new_alert',
                'data': alert_data
            })
            
        except Exception as e:
            logger.error(f"Error creating alert record: {str(e)}")
    
    async def _handle_critical_alert(self, message: StreamMessage):
        """Handle critical alerts with immediate escalation"""
        try:
            alert_data = message.data
            
            # Log critical alert
            logger.critical(f"Critical alert: {alert_data.get('message')} from {message.source}")
            
            # Immediate notification to all connected dashboards
            await self.websocket_manager.broadcast_to_dashboards({
                'type': 'critical_alert',
                'data': message.to_dict(),
                'requires_acknowledgment': True
            })
            
        except Exception as e:
            logger.error(f"Error handling critical alert: {str(e)}")
    
    async def _analyze_critical_metric(self, message: StreamMessage):
        """Analyze critical metrics for immediate insights"""
        try:
            metric_data = message.data
            
            # Quick analysis for threshold breaches
            value = metric_data.get('value')
            thresholds = metric_data.get('thresholds', {})
            
            if value and thresholds:
                if value > thresholds.get('critical_high', float('inf')):
                    await self._create_threshold_alert('critical_high', metric_data, message)
                elif value < thresholds.get('critical_low', float('-inf')):
                    await self._create_threshold_alert('critical_low', metric_data, message)
                    
        except Exception as e:
            logger.error(f"Error analyzing critical metric: {str(e)}")
    
    async def _create_threshold_alert(self, threshold_type: str, 
                                    metric_data: Dict[str, Any], 
                                    original_message: StreamMessage):
        """Create alert for threshold breach"""
        alert_message = StreamMessage(
            id=f"threshold_alert_{datetime.now().timestamp()}",
            stream_type=StreamType.ALERTS,
            timestamp=datetime.now(),
            source="threshold_monitor",
            data={
                'type': 'threshold_breach',
                'severity': 'critical' if 'critical' in threshold_type else 'warning',
                'message': f"Metric {metric_data.get('name')} breached {threshold_type} threshold",
                'metric_data': metric_data,
                'threshold_type': threshold_type
            },
            priority=5
        )
        
        await self.ingest_data(alert_message)
    
    async def _create_pattern_alert(self, event_type: str, count: int, 
                                  events_data: List[Dict[str, Any]]):
        """Create alert for detected patterns"""
        alert_message = StreamMessage(
            id=f"pattern_alert_{datetime.now().timestamp()}",
            stream_type=StreamType.ALERTS,
            timestamp=datetime.now(),
            source="pattern_detector",
            data={
                'type': 'unusual_pattern',
                'severity': 'warning',
                'message': f"Unusual activity detected: {count} {event_type} events",
                'event_type': event_type,
                'count': count,
                'time_window': '5_minutes'
            },
            priority=3
        )
        
        await self.ingest_data(alert_message)
    
    async def _broadcast_updates(self, stream_type: StreamType, messages: List[StreamMessage]):
        """Broadcast real-time updates to connected dashboards"""
        try:
            update_data = {
                'type': 'batch_update',
                'stream_type': stream_type.value,
                'messages': [msg.to_dict() for msg in messages],
                'timestamp': datetime.now().isoformat()
            }
            
            await self.websocket_manager.broadcast_to_dashboards(update_data)
            
        except Exception as e:
            logger.error(f"Error broadcasting updates: {str(e)}")
    
    async def _store_in_redis(self, message: StreamMessage):
        """Store message in Redis for persistence"""
        try:
            key = f"stream:{message.stream_type.value}:{message.id}"
            config = self.stream_configs[message.stream_type]
            ttl = timedelta(hours=config['retention_hours']).total_seconds()
            
            await self.redis_client.setex(
                key, 
                ttl,
                json.dumps(message.to_dict())
            )
            
        except Exception as e:
            logger.error(f"Error storing message in Redis: {str(e)}")
    
    async def _batch_processor(self):
        """Background task for batch processing operations"""
        while self._running:
            try:
                # Aggregate metrics for reporting
                await self._aggregate_metrics()
                
                # Clean up old data
                await self._cleanup_old_streams()
                
                # Update dashboard caches
                await self._update_dashboard_caches()
                
                # Sleep for batch interval
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in batch processor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _cleanup_expired_data(self):
        """Clean up expired data from Redis"""
        while self._running:
            try:
                # This runs every hour
                await asyncio.sleep(3600)
                
                # Redis handles TTL automatically, but we can do additional cleanup
                current_time = datetime.now()
                
                for stream_type, config in self.stream_configs.items():
                    cutoff_time = current_time - timedelta(hours=config['retention_hours'])
                    
                    # Clean up any manual tracking structures
                    # (Redis keys with TTL will expire automatically)
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
    
    async def _aggregate_metrics(self):
        """Aggregate metrics for dashboard display"""
        try:
            # Get recent metrics from Redis
            pattern = "stream:metrics:*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Aggregate by metric type and time window
                aggregations = {}
                
                for key in keys[-100:]:  # Process last 100 metrics
                    data = await self.redis_client.get(key)
                    if data:
                        message_data = json.loads(data)
                        metric_name = message_data.get('data', {}).get('name')
                        
                        if metric_name:
                            if metric_name not in aggregations:
                                aggregations[metric_name] = {
                                    'count': 0,
                                    'sum': 0,
                                    'min': float('inf'),
                                    'max': float('-inf'),
                                    'latest': None
                                }
                            
                            value = message_data.get('data', {}).get('value', 0)
                            agg = aggregations[metric_name]
                            
                            agg['count'] += 1
                            agg['sum'] += value
                            agg['min'] = min(agg['min'], value)
                            agg['max'] = max(agg['max'], value)
                            agg['latest'] = value
                
                # Store aggregations
                for metric_name, agg in aggregations.items():
                    agg['average'] = agg['sum'] / agg['count'] if agg['count'] > 0 else 0
                    
                    await self.redis_client.setex(
                        f"aggregation:{metric_name}",
                        300,  # 5 minutes TTL
                        json.dumps(agg)
                    )
                    
        except Exception as e:
            logger.error(f"Error aggregating metrics: {str(e)}")
    
    async def _cleanup_old_streams(self):
        """Clean up old stream data"""
        # Redis TTL handles most cleanup automatically
        pass
    
    async def _update_dashboard_caches(self):
        """Update dashboard caches with latest data"""
        try:
            # Update cached dashboard data for faster loading
            cache_data = {
                'last_updated': datetime.now().isoformat(),
                'active_alerts': await self._get_active_alerts_count(),
                'metrics_processed': await self._get_metrics_processed_count(),
                'system_health': 'healthy'
            }
            
            await self.redis_client.setex(
                "dashboard:cache:summary",
                60,  # 1 minute TTL
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Error updating dashboard caches: {str(e)}")
    
    async def _get_active_alerts_count(self) -> int:
        """Get count of active alerts"""
        try:
            pattern = "alert:*"
            keys = await self.redis_client.keys(pattern)
            return len(keys)
        except:
            return 0
    
    async def _get_metrics_processed_count(self) -> int:
        """Get count of processed metrics in last hour"""
        try:
            pattern = "stream:metrics:*"
            keys = await self.redis_client.keys(pattern)
            return len(keys)
        except:
            return 0
    
    def _validate_message(self, message: StreamMessage) -> bool:
        """Validate stream message format"""
        try:
            return (
                message.id and
                message.stream_type in StreamType and
                message.timestamp and
                message.source and
                isinstance(message.data, dict)
            )
        except:
            return False
    
    def register_event_handler(self, stream_type: StreamType, handler: Callable):
        """Register event handler for stream type"""
        if stream_type not in self.event_handlers:
            self.event_handlers[stream_type] = []
        
        self.event_handlers[stream_type].append(handler)
    
    async def get_stream_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics"""
        stats = {}
        
        for stream_type in StreamType:
            queue = self.processing_queues.get(stream_type)
            stats[stream_type.value] = {
                'queue_size': queue.qsize() if queue else 0,
                'config': self.stream_configs.get(stream_type, {}),
                'handlers_count': len(self.event_handlers.get(stream_type, []))
            }
        
        return {
            'streams': stats,
            'running': self._running,
            'tasks_count': len(self._tasks)
        }