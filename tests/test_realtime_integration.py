"""
Integration Tests for Real-time Data Processing and Alerts
Tests the complete integration of real-time processing, alerting, and notification systems
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock
import redis.asyncio as redis

from scrollintel.core.realtime_data_processor import (
    RealTimeDataProcessor, StreamMessage, StreamType
)
from scrollintel.core.intelligent_alerting_system import (
    IntelligentAlertingSystem, ThresholdRule, AlertSeverity, AlertStatus
)
from scrollintel.core.notification_system import (
    NotificationSystem, NotificationTemplate, NotificationRecipient, 
    NotificationRule, NotificationPriority, NotificationChannel
)
from scrollintel.core.data_quality_monitoring import (
    DataQualityMonitor, QualityRule, DataQualityDimension, QualityCheckType
)
from scrollintel.core.websocket_manager import WebSocketManager

class IntegrationTestEnvironment:
    """Test environment for integration testing"""
    
    def __init__(self):
        self.redis_client = None
        self.websocket_manager = None
        self.data_processor = None
        self.alerting_system = None
        self.notification_system = None
        self.quality_monitor = None
        
        # Test data collectors
        self.received_notifications = []
        self.received_alerts = []
        self.websocket_messages = []
    
    async def setup(self):
        """Setup the test environment"""
        # Mock Redis client
        self.redis_client = Mock(spec=redis.Redis)
        self.redis_client.setex = AsyncMock(return_value=True)
        self.redis_client.hset = AsyncMock(return_value=True)
        self.redis_client.hgetall = AsyncMock(return_value={})
        self.redis_client.keys = AsyncMock(return_value=[])
        self.redis_client.get = AsyncMock(return_value=None)
        self.redis_client.hdel = AsyncMock(return_value=True)
        
        # Mock WebSocket manager with message capture
        self.websocket_manager = Mock(spec=WebSocketManager)
        self.websocket_manager.broadcast_to_dashboards = AsyncMock(
            side_effect=self._capture_websocket_message
        )
        self.websocket_manager.send_to_user = AsyncMock(
            side_effect=self._capture_websocket_message
        )
        
        # Initialize systems
        await self._initialize_systems()
        
        # Setup test configurations
        await self._setup_test_configurations()
    
    async def _initialize_systems(self):
        """Initialize all systems"""
        # Create mock engines for data processor
        mock_predictive_engine = Mock()
        mock_predictive_engine.update_real_time_predictions = AsyncMock()
        
        mock_insight_generator = Mock()
        mock_insight_generator.analyze_real_time_data = AsyncMock(return_value=[])
        
        # Initialize data processor
        self.data_processor = RealTimeDataProcessor(
            redis_client=self.redis_client,
            websocket_manager=self.websocket_manager,
            predictive_engine=mock_predictive_engine,
            insight_generator=mock_insight_generator
        )
        
        # Initialize alerting system
        self.alerting_system = IntelligentAlertingSystem(
            redis_client=self.redis_client,
            websocket_manager=self.websocket_manager
        )
        
        # Initialize notification system
        self.notification_system = NotificationSystem(
            redis_client=self.redis_client,
            websocket_manager=self.websocket_manager
        )
        
        # Initialize quality monitor
        self.quality_monitor = DataQualityMonitor(
            redis_client=self.redis_client,
            alerting_system=self.alerting_system,
            notification_system=self.notification_system
        )
        
        # Register custom notification handler
        self.notification_system.register_channel_handler(
            NotificationChannel.WEBSOCKET,
            self._capture_notification
        )
        
        # Start all systems
        await self.data_processor.start()
        await self.alerting_system.start()
        await self.notification_system.start()
        await self.quality_monitor.start()
    
    async def _setup_test_configurations(self):
        """Setup test configurations"""
        # Add notification template
        template = NotificationTemplate(
            id="test_template",
            name="Test Template",
            channel=NotificationChannel.WEBSOCKET,
            subject_template="Alert: {title}",
            body_template="Message: {message}",
            priority=NotificationPriority.MEDIUM
        )
        await self.notification_system.add_template(template)
        
        # Add notification recipient
        recipient = NotificationRecipient(
            id="test_recipient",
            name="Test User",
            email="test@example.com"
        )
        await self.notification_system.add_recipient(recipient)
        
        # Add notification rule
        rule = NotificationRule(
            id="test_notification_rule",
            name="Test Notification Rule",
            conditions=[
                {"type": "event_type", "value": "threshold_breach"},
                {"type": "priority", "min_priority": 2}
            ],
            recipients=["test_recipient"],
            channels=[NotificationChannel.WEBSOCKET],
            template_id="test_template",
            priority=NotificationPriority.MEDIUM
        )
        await self.notification_system.add_rule(rule)
    
    async def _capture_websocket_message(self, message):
        """Capture WebSocket messages for testing"""
        self.websocket_messages.append({
            'timestamp': datetime.now(),
            'message': message
        })
    
    async def _capture_notification(self, notification):
        """Capture notifications for testing"""
        self.received_notifications.append({
            'timestamp': datetime.now(),
            'notification': notification
        })
        return True
    
    async def teardown(self):
        """Cleanup the test environment"""
        if self.data_processor:
            await self.data_processor.stop()
        if self.alerting_system:
            await self.alerting_system.stop()
        if self.notification_system:
            await self.notification_system.stop()
        if self.quality_monitor:
            await self.quality_monitor.stop()

@pytest.fixture
async def integration_env():
    """Fixture for integration test environment"""
    env = IntegrationTestEnvironment()
    await env.setup()
    yield env
    await env.teardown()

class TestRealTimeIntegration:
    """Integration tests for real-time processing system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_metric_processing_with_alerts(self, integration_env):
        """Test complete flow from metric ingestion to alert generation"""
        env = integration_env
        
        # Add threshold rule
        threshold_rule = ThresholdRule(
            id="test_threshold",
            metric_name="cpu_usage",
            operator=">",
            value=80.0,
            severity=AlertSeverity.HIGH,
            description="CPU usage too high",
            consecutive_breaches=1,
            cooldown_minutes=1
        )
        
        await env.alerting_system.add_threshold_rule(threshold_rule)
        
        # Create metric message that will breach threshold
        metric_message = StreamMessage(
            id="test_metric_1",
            stream_type=StreamType.METRICS,
            timestamp=datetime.now(),
            source="test_server",
            data={
                'name': 'cpu_usage',
                'value': 85.0,  # Above threshold
                'category': 'system',
                'thresholds': {
                    'critical_high': 90.0,
                    'warning_high': 80.0
                }
            },
            priority=3
        )
        
        # Ingest the metric
        success = await env.data_processor.ingest_data(metric_message)
        assert success, "Metric ingestion should succeed"
        
        # Allow processing time
        await asyncio.sleep(2)
        
        # Check that threshold was evaluated
        alerts = await env.alerting_system.check_thresholds(
            metric_name="cpu_usage",
            value=85.0,
            timestamp=datetime.now(),
            context={'source': 'test_server'}
        )
        
        # Verify alert was created
        active_alerts = await env.alerting_system.get_active_alerts()
        assert len(active_alerts) > 0, "Should have created an alert"
        
        # Verify WebSocket message was sent
        assert len(env.websocket_messages) > 0, "Should have sent WebSocket messages"
        
        # Check for alert-related messages
        alert_messages = [
            msg for msg in env.websocket_messages 
            if 'alert' in str(msg['message']).lower()
        ]
        assert len(alert_messages) > 0, "Should have sent alert-related WebSocket messages"
    
    @pytest.mark.asyncio
    async def test_data_quality_monitoring_with_notifications(self, integration_env):
        """Test data quality monitoring with notification integration"""
        env = integration_env
        
        # Add quality rule
        quality_rule = QualityRule(
            id="test_quality_rule",
            name="Test Completeness Check",
            description="Check for null values in user table",
            dimension=DataQualityDimension.COMPLETENESS,
            check_type=QualityCheckType.NULL_CHECK,
            table_name="users",
            column_name="email",
            threshold_warning=0.95,
            threshold_critical=0.85,
            schedule_minutes=60
        )
        
        await env.quality_monitor.add_quality_rule(quality_rule)
        
        # Run quality check
        metric = await env.quality_monitor.run_quality_check(quality_rule.id)
        
        assert metric is not None, "Quality check should return a metric"
        
        # Allow processing time for notifications
        await asyncio.sleep(1)
        
        # Check if notifications were triggered (depends on simulated quality score)
        if metric.score < quality_rule.threshold_critical:
            # Should have triggered notifications
            assert len(env.received_notifications) > 0, "Should have sent notifications for quality issues"
        
        # Verify quality summary is available
        summary = await env.quality_monitor.get_quality_summary()
        assert 'overall_score' in summary, "Should have quality summary"
    
    @pytest.mark.asyncio
    async def test_high_priority_message_processing(self, integration_env):
        """Test high-priority message processing and immediate alerts"""
        env = integration_env
        
        # Create high-priority alert message
        alert_message = StreamMessage(
            id="critical_alert_1",
            stream_type=StreamType.ALERTS,
            timestamp=datetime.now(),
            source="security_system",
            data={
                'type': 'security_breach',
                'severity': 'critical',
                'message': 'Unauthorized access detected',
                'context': {
                    'ip_address': '192.168.1.100',
                    'user_agent': 'malicious_bot'
                }
            },
            priority=5  # Maximum priority
        )
        
        # Ingest high-priority message
        success = await env.data_processor.ingest_data(alert_message)
        assert success, "High-priority message ingestion should succeed"
        
        # Allow minimal processing time (should be immediate)
        await asyncio.sleep(0.5)
        
        # Verify immediate WebSocket broadcast
        high_priority_messages = [
            msg for msg in env.websocket_messages
            if 'high_priority' in str(msg['message']).lower() or 'critical' in str(msg['message']).lower()
        ]
        
        assert len(high_priority_messages) > 0, "Should have sent high-priority WebSocket messages"
        
        # Verify message was processed quickly
        latest_message = env.websocket_messages[-1]
        time_diff = (datetime.now() - latest_message['timestamp']).total_seconds()
        assert time_diff < 1.0, "High-priority messages should be processed within 1 second"
    
    @pytest.mark.asyncio
    async def test_concurrent_stream_processing(self, integration_env):
        """Test concurrent processing of multiple stream types"""
        env = integration_env
        
        # Create messages for different streams
        messages = []
        
        # Metrics stream
        for i in range(10):
            messages.append(StreamMessage(
                id=f"metric_{i}",
                stream_type=StreamType.METRICS,
                timestamp=datetime.now(),
                source="test_metrics",
                data={'name': f'metric_{i}', 'value': 50 + i},
                priority=1
            ))
        
        # Events stream
        for i in range(5):
            messages.append(StreamMessage(
                id=f"event_{i}",
                stream_type=StreamType.EVENTS,
                timestamp=datetime.now(),
                source="test_events",
                data={'type': f'event_{i}', 'data': {'count': i}},
                priority=2
            ))
        
        # Alerts stream
        for i in range(3):
            messages.append(StreamMessage(
                id=f"alert_{i}",
                stream_type=StreamType.ALERTS,
                timestamp=datetime.now(),
                source="test_alerts",
                data={'type': 'warning', 'message': f'Alert {i}'},
                priority=4
            ))
        
        # Ingest all messages concurrently
        tasks = []
        for message in messages:
            task = asyncio.create_task(env.data_processor.ingest_data(message))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify all messages were processed successfully
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.95, f"Should have 95%+ success rate, got {success_rate:.2%}"
        
        # Allow processing time
        await asyncio.sleep(2)
        
        # Verify processing statistics
        stats = await env.data_processor.get_stream_stats()
        assert stats['running'], "Data processor should be running"
        
        # Verify WebSocket messages were sent
        assert len(env.websocket_messages) > 0, "Should have sent WebSocket messages"
    
    @pytest.mark.asyncio
    async def test_alert_acknowledgment_and_resolution_flow(self, integration_env):
        """Test complete alert lifecycle from creation to resolution"""
        env = integration_env
        
        # Create an alert
        alert = await env.alerting_system.create_alert(
            rule_id="test_rule",
            title="Test Alert",
            message="This is a test alert",
            severity=AlertSeverity.MEDIUM,
            source="integration_test",
            context={'test': True}
        )
        
        assert alert is not None, "Should create alert successfully"
        assert alert.status == AlertStatus.ACTIVE, "Alert should be active"
        
        # Acknowledge the alert
        ack_success = await env.alerting_system.acknowledge_alert(
            alert.id, 
            "test_user"
        )
        
        assert ack_success, "Should acknowledge alert successfully"
        
        # Verify acknowledgment
        active_alerts = await env.alerting_system.get_active_alerts()
        acknowledged_alert = next(
            (a for a in active_alerts if a['id'] == alert.id), 
            None
        )
        
        assert acknowledged_alert is not None, "Alert should still be active"
        assert acknowledged_alert['status'] == AlertStatus.ACKNOWLEDGED.value, "Alert should be acknowledged"
        
        # Resolve the alert
        resolve_success = await env.alerting_system.resolve_alert(
            alert.id, 
            "test_user"
        )
        
        assert resolve_success, "Should resolve alert successfully"
        
        # Verify resolution
        active_alerts_after = await env.alerting_system.get_active_alerts()
        resolved_alert = next(
            (a for a in active_alerts_after if a['id'] == alert.id), 
            None
        )
        
        assert resolved_alert is None, "Alert should no longer be active"
        
        # Verify WebSocket messages for acknowledgment and resolution
        ack_messages = [
            msg for msg in env.websocket_messages
            if 'acknowledged' in str(msg['message']).lower()
        ]
        
        resolve_messages = [
            msg for msg in env.websocket_messages
            if 'resolved' in str(msg['message']).lower()
        ]
        
        assert len(ack_messages) > 0, "Should have sent acknowledgment messages"
        assert len(resolve_messages) > 0, "Should have sent resolution messages"
    
    @pytest.mark.asyncio
    async def test_notification_system_integration(self, integration_env):
        """Test notification system integration with other components"""
        env = integration_env
        
        # Send a critical alert notification
        notification_ids = await env.notification_system.send_critical_alert(
            title="Critical System Alert",
            message="System is experiencing critical issues",
            context={
                'severity': 'critical',
                'affected_systems': ['database', 'api'],
                'estimated_impact': 'high'
            }
        )
        
        assert len(notification_ids) > 0, "Should create notifications"
        
        # Allow processing time
        await asyncio.sleep(1)
        
        # Verify notifications were processed
        assert len(env.received_notifications) > 0, "Should have processed notifications"
        
        # Send insight notification
        insight_data = {
            'title': 'Performance Insight',
            'message': 'Database query performance has improved by 25%',
            'significance': 0.8,
            'confidence': 0.9,
            'recommendations': ['Continue current optimization strategy']
        }
        
        insight_notification_ids = await env.notification_system.send_insight_notification(insight_data)
        
        assert len(insight_notification_ids) > 0, "Should create insight notifications"
        
        # Verify notification statistics
        stats = await env.notification_system.get_notification_statistics()
        assert stats['total_sent'] > 0, "Should have sent notifications"
    
    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, integration_env):
        """Test system performance under sustained load"""
        env = integration_env
        
        # Generate sustained load
        message_batches = []
        
        for batch in range(5):  # 5 batches
            batch_messages = []
            
            for i in range(100):  # 100 messages per batch
                message = StreamMessage(
                    id=f"load_test_{batch}_{i}",
                    stream_type=StreamType.METRICS,
                    timestamp=datetime.now(),
                    source="load_test",
                    data={
                        'name': f'load_metric_{i % 10}',
                        'value': 50 + (i % 50),
                        'batch': batch
                    },
                    priority=1 + (i % 3)
                )
                batch_messages.append(message)
            
            message_batches.append(batch_messages)
        
        # Process batches with timing
        start_time = datetime.now()
        
        for batch_messages in message_batches:
            # Process batch
            tasks = []
            for message in batch_messages:
                task = asyncio.create_task(env.data_processor.ingest_data(message))
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks)
            batch_success_rate = sum(batch_results) / len(batch_results)
            
            assert batch_success_rate >= 0.95, f"Batch should have 95%+ success rate"
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate performance metrics
        total_messages = sum(len(batch) for batch in message_batches)
        throughput = total_messages / total_duration
        
        print(f"Load test: {total_messages} messages in {total_duration:.2f}s")
        print(f"Throughput: {throughput:.2f} messages/second")
        
        # Performance assertions
        assert throughput >= 100, f"Should handle at least 100 msg/s, got {throughput:.2f}"
        
        # Verify system is still responsive
        stats = await env.data_processor.get_stream_stats()
        assert stats['running'], "System should still be running after load test"
        
        # Verify WebSocket messages were sent
        assert len(env.websocket_messages) > 0, "Should have sent WebSocket messages during load test"
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, integration_env):
        """Test system error recovery and resilience"""
        env = integration_env
        
        # Test with invalid messages
        invalid_messages = [
            StreamMessage(
                id="",  # Invalid empty ID
                stream_type=StreamType.METRICS,
                timestamp=datetime.now(),
                source="error_test",
                data={},
                priority=1
            ),
            StreamMessage(
                id="valid_after_error",
                stream_type=StreamType.METRICS,
                timestamp=datetime.now(),
                source="",  # Invalid empty source
                data={'name': 'test', 'value': 100},
                priority=1
            )
        ]
        
        # Process invalid messages
        error_tasks = []
        for message in invalid_messages:
            task = asyncio.create_task(env.data_processor.ingest_data(message))
            error_tasks.append(task)
        
        error_results = await asyncio.gather(*error_tasks)
        
        # Should handle errors gracefully
        assert not all(error_results), "Invalid messages should be rejected"
        
        # Verify system is still functional with valid messages
        valid_message = StreamMessage(
            id="recovery_test",
            stream_type=StreamType.METRICS,
            timestamp=datetime.now(),
            source="recovery_test",
            data={'name': 'recovery_metric', 'value': 75},
            priority=1
        )
        
        recovery_success = await env.data_processor.ingest_data(valid_message)
        assert recovery_success, "System should recover and process valid messages"
        
        # Verify system statistics
        stats = await env.data_processor.get_stream_stats()
        assert stats['running'], "System should still be running after errors"

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])