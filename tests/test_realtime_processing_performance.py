"""
Performance Tests for Real-time Data Processing and Alerts
Tests throughput, latency, and scalability of real-time processing components
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
import redis.asyncio as redis
from unittest.mock import Mock, AsyncMock

from scrollintel.core.realtime_data_processor import (
    RealTimeDataProcessor, StreamMessage, StreamType
)
from scrollintel.core.intelligent_alerting_system import (
    IntelligentAlertingSystem, ThresholdRule, AlertSeverity
)
from scrollintel.core.notification_system import (
    NotificationSystem, NotificationPriority
)
from scrollintel.core.data_quality_monitoring import (
    DataQualityMonitor, QualityRule, DataQualityDimension, QualityCheckType
)
from scrollintel.core.websocket_manager import WebSocketManager

class PerformanceTestSuite:
    """Performance test suite for real-time processing"""
    
    def __init__(self):
        self.redis_client = None
        self.websocket_manager = None
        self.data_processor = None
        self.alerting_system = None
        self.notification_system = None
        self.quality_monitor = None
        
        # Performance metrics
        self.metrics = {
            'throughput': [],
            'latency': [],
            'memory_usage': [],
            'cpu_usage': [],
            'error_rate': []
        }
    
    async def setup(self):
        """Setup test environment"""
        # Mock Redis client
        self.redis_client = Mock(spec=redis.Redis)
        self.redis_client.setex = AsyncMock(return_value=True)
        self.redis_client.hset = AsyncMock(return_value=True)
        self.redis_client.hgetall = AsyncMock(return_value={})
        self.redis_client.keys = AsyncMock(return_value=[])
        self.redis_client.get = AsyncMock(return_value=None)
        self.redis_client.hdel = AsyncMock(return_value=True)
        
        # Mock WebSocket manager
        self.websocket_manager = Mock(spec=WebSocketManager)
        self.websocket_manager.broadcast_to_dashboards = AsyncMock()
        self.websocket_manager.send_to_user = AsyncMock()
        
        # Initialize components
        self.data_processor = RealTimeDataProcessor(
            redis_client=self.redis_client,
            websocket_manager=self.websocket_manager,
            predictive_engine=Mock(),
            insight_generator=Mock()
        )
        
        self.alerting_system = IntelligentAlertingSystem(
            redis_client=self.redis_client,
            websocket_manager=self.websocket_manager
        )
        
        self.notification_system = NotificationSystem(
            redis_client=self.redis_client,
            websocket_manager=self.websocket_manager
        )
        
        self.quality_monitor = DataQualityMonitor(
            redis_client=self.redis_client,
            alerting_system=self.alerting_system,
            notification_system=self.notification_system
        )
        
        # Start systems
        await self.data_processor.start()
        await self.alerting_system.start()
        await self.notification_system.start()
        await self.quality_monitor.start()
    
    async def teardown(self):
        """Cleanup test environment"""
        if self.data_processor:
            await self.data_processor.stop()
        if self.alerting_system:
            await self.alerting_system.stop()
        if self.notification_system:
            await self.notification_system.stop()
        if self.quality_monitor:
            await self.quality_monitor.stop()

@pytest.fixture
async def performance_suite():
    """Fixture for performance test suite"""
    suite = PerformanceTestSuite()
    await suite.setup()
    yield suite
    await suite.teardown()

class TestRealTimeProcessingPerformance:
    """Performance tests for real-time data processing"""
    
    @pytest.mark.asyncio
    async def test_message_ingestion_throughput(self, performance_suite):
        """Test message ingestion throughput"""
        processor = performance_suite.data_processor
        
        # Test parameters
        message_counts = [100, 500, 1000, 5000, 10000]
        results = {}
        
        for count in message_counts:
            messages = self._generate_test_messages(count)
            
            start_time = time.time()
            
            # Ingest messages
            tasks = []
            for message in messages:
                task = asyncio.create_task(processor.ingest_data(message))
                tasks.append(task)
            
            results_list = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate throughput
            throughput = count / duration  # messages per second
            success_rate = sum(results_list) / len(results_list)
            
            results[count] = {
                'throughput': throughput,
                'duration': duration,
                'success_rate': success_rate
            }
            
            print(f"Messages: {count}, Throughput: {throughput:.2f} msg/s, Success: {success_rate:.2%}")
        
        # Verify performance requirements
        assert results[1000]['throughput'] >= 500, "Should handle at least 500 messages/second"
        assert results[1000]['success_rate'] >= 0.99, "Should have 99%+ success rate"
        
        # Check scalability
        throughput_1k = results[1000]['throughput']
        throughput_5k = results[5000]['throughput']
        scalability_ratio = throughput_5k / throughput_1k
        
        assert scalability_ratio >= 0.8, "Throughput should scale reasonably with load"
    
    @pytest.mark.asyncio
    async def test_processing_latency(self, performance_suite):
        """Test end-to-end processing latency"""
        processor = performance_suite.data_processor
        
        latencies = []
        
        # Test with different message priorities
        for priority in range(1, 6):
            for _ in range(100):  # 100 samples per priority
                message = StreamMessage(
                    id=f"latency_test_{time.time()}",
                    stream_type=StreamType.METRICS,
                    timestamp=datetime.now(),
                    source="performance_test",
                    data={'value': 100, 'name': 'test_metric'},
                    priority=priority
                )
                
                start_time = time.time()
                
                # Ingest and wait for processing
                success = await processor.ingest_data(message)
                
                # Simulate processing completion check
                await asyncio.sleep(0.001)  # Minimal processing time
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                
                if success:
                    latencies.append({
                        'priority': priority,
                        'latency_ms': latency
                    })
        
        # Analyze latency by priority
        priority_latencies = {}
        for data in latencies:
            priority = data['priority']
            if priority not in priority_latencies:
                priority_latencies[priority] = []
            priority_latencies[priority].append(data['latency_ms'])
        
        # Calculate statistics
        for priority, latency_list in priority_latencies.items():
            avg_latency = statistics.mean(latency_list)
            p95_latency = statistics.quantiles(latency_list, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latency_list, n=100)[98]  # 99th percentile
            
            print(f"Priority {priority}: Avg={avg_latency:.2f}ms, P95={p95_latency:.2f}ms, P99={p99_latency:.2f}ms")
            
            # Performance requirements
            if priority >= 4:  # High priority messages
                assert avg_latency <= 10, f"High priority messages should have <10ms average latency"
                assert p95_latency <= 50, f"High priority messages should have <50ms P95 latency"
            else:
                assert avg_latency <= 100, f"Normal priority messages should have <100ms average latency"
    
    @pytest.mark.asyncio
    async def test_concurrent_stream_processing(self, performance_suite):
        """Test concurrent processing of multiple stream types"""
        processor = performance_suite.data_processor
        
        # Create concurrent streams
        stream_configs = [
            (StreamType.METRICS, 1000, 1),
            (StreamType.EVENTS, 500, 2),
            (StreamType.ALERTS, 100, 5),
            (StreamType.INSIGHTS, 200, 3)
        ]
        
        start_time = time.time()
        
        # Start concurrent ingestion
        tasks = []
        for stream_type, count, priority in stream_configs:
            task = asyncio.create_task(
                self._ingest_stream_batch(processor, stream_type, count, priority)
            )
            tasks.append(task)
        
        # Wait for all streams to complete
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate overall throughput
        total_messages = sum(config[1] for config in stream_configs)
        overall_throughput = total_messages / total_duration
        
        print(f"Concurrent processing: {total_messages} messages in {total_duration:.2f}s")
        print(f"Overall throughput: {overall_throughput:.2f} msg/s")
        
        # Verify all streams processed successfully
        for i, (stream_type, count, priority) in enumerate(stream_configs):
            success_rate = results[i]
            assert success_rate >= 0.95, f"{stream_type.value} stream should have 95%+ success rate"
        
        # Performance requirement
        assert overall_throughput >= 400, "Should handle 400+ messages/second across all streams"
    
    @pytest.mark.asyncio
    async def test_alerting_system_performance(self, performance_suite):
        """Test alerting system performance under load"""
        alerting_system = performance_suite.alerting_system
        
        # Add threshold rules
        rules = []
        for i in range(100):  # 100 threshold rules
            rule = ThresholdRule(
                id=f"perf_rule_{i}",
                metric_name=f"test_metric_{i % 10}",  # 10 different metrics
                operator=">",
                value=50.0,
                severity=AlertSeverity.MEDIUM,
                description=f"Performance test rule {i}",
                cooldown_minutes=1
            )
            rules.append(rule)
            await alerting_system.add_threshold_rule(rule)
        
        # Test threshold checking performance
        check_times = []
        
        for _ in range(1000):  # 1000 threshold checks
            metric_name = f"test_metric_{_ % 10}"
            value = 75.0  # Above threshold
            
            start_time = time.time()
            
            alerts = await alerting_system.check_thresholds(
                metric_name=metric_name,
                value=value,
                timestamp=datetime.now(),
                context={'test': True}
            )
            
            end_time = time.time()
            check_time = (end_time - start_time) * 1000  # milliseconds
            check_times.append(check_time)
        
        # Analyze performance
        avg_check_time = statistics.mean(check_times)
        p95_check_time = statistics.quantiles(check_times, n=20)[18]
        
        print(f"Threshold checking: Avg={avg_check_time:.2f}ms, P95={p95_check_time:.2f}ms")
        
        # Performance requirements
        assert avg_check_time <= 5, "Average threshold check should be <5ms"
        assert p95_check_time <= 20, "P95 threshold check should be <20ms"
    
    @pytest.mark.asyncio
    async def test_notification_system_performance(self, performance_suite):
        """Test notification system performance"""
        notification_system = performance_suite.notification_system
        
        # Test notification creation and queuing
        notification_times = []
        
        for i in range(500):  # 500 notifications
            start_time = time.time()
            
            notification_ids = await notification_system.send_notification(
                event_type='performance_test',
                data={
                    'message': f'Test notification {i}',
                    'value': i,
                    'timestamp': datetime.now().isoformat()
                },
                priority=NotificationPriority.MEDIUM
            )
            
            end_time = time.time()
            notification_time = (end_time - start_time) * 1000  # milliseconds
            notification_times.append(notification_time)
        
        # Analyze performance
        avg_notification_time = statistics.mean(notification_times)
        p95_notification_time = statistics.quantiles(notification_times, n=20)[18]
        
        print(f"Notification creation: Avg={avg_notification_time:.2f}ms, P95={p95_notification_time:.2f}ms")
        
        # Performance requirements
        assert avg_notification_time <= 10, "Average notification creation should be <10ms"
        assert p95_notification_time <= 50, "P95 notification creation should be <50ms"
    
    @pytest.mark.asyncio
    async def test_data_quality_monitoring_performance(self, performance_suite):
        """Test data quality monitoring performance"""
        quality_monitor = performance_suite.quality_monitor
        
        # Add quality rules
        rules = []
        for i in range(50):  # 50 quality rules
            rule = QualityRule(
                id=f"perf_quality_rule_{i}",
                name=f"Performance Test Rule {i}",
                description=f"Performance test quality rule {i}",
                dimension=DataQualityDimension.COMPLETENESS,
                check_type=QualityCheckType.NULL_CHECK,
                table_name=f"test_table_{i % 5}",  # 5 different tables
                column_name=f"test_column_{i % 3}",  # 3 different columns
                schedule_minutes=60
            )
            rules.append(rule)
            await quality_monitor.add_quality_rule(rule)
        
        # Test quality check performance
        check_times = []
        
        for rule in rules:
            start_time = time.time()
            
            metric = await quality_monitor.run_quality_check(rule.id)
            
            end_time = time.time()
            check_time = (end_time - start_time) * 1000  # milliseconds
            check_times.append(check_time)
        
        # Analyze performance
        avg_check_time = statistics.mean(check_times)
        p95_check_time = statistics.quantiles(check_times, n=20)[18]
        
        print(f"Quality checks: Avg={avg_check_time:.2f}ms, P95={p95_check_time:.2f}ms")
        
        # Performance requirements
        assert avg_check_time <= 100, "Average quality check should be <100ms"
        assert p95_check_time <= 500, "P95 quality check should be <500ms"
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_suite):
        """Test memory usage under sustained load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        processor = performance_suite.data_processor
        
        # Sustained load test
        for batch in range(10):  # 10 batches
            messages = self._generate_test_messages(1000)
            
            # Ingest batch
            tasks = []
            for message in messages:
                task = asyncio.create_task(processor.ingest_data(message))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # Allow processing
            await asyncio.sleep(1)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"Batch {batch + 1}: Memory usage = {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Memory should not grow excessively
            assert memory_increase <= 100, f"Memory increase should be <100MB, got {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, performance_suite):
        """Test performance under error conditions"""
        processor = performance_suite.data_processor
        
        # Test with invalid messages
        invalid_messages = []
        for i in range(100):
            # Create invalid message (missing required fields)
            invalid_message = StreamMessage(
                id="",  # Invalid empty ID
                stream_type=StreamType.METRICS,
                timestamp=datetime.now(),
                source="",  # Invalid empty source
                data={}  # Empty data
            )
            invalid_messages.append(invalid_message)
        
        start_time = time.time()
        
        # Process invalid messages
        tasks = []
        for message in invalid_messages:
            task = asyncio.create_task(processor.ingest_data(message))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate error handling performance
        error_rate = 1 - (sum(results) / len(results))
        error_throughput = len(invalid_messages) / duration
        
        print(f"Error handling: {error_rate:.2%} error rate, {error_throughput:.2f} errors/s")
        
        # Performance requirements
        assert error_throughput >= 100, "Should handle 100+ errors/second"
        assert duration <= 2, "Error handling should complete within 2 seconds"
    
    def _generate_test_messages(self, count: int) -> List[StreamMessage]:
        """Generate test messages for performance testing"""
        messages = []
        
        for i in range(count):
            message = StreamMessage(
                id=f"perf_test_{i}_{time.time()}",
                stream_type=StreamType.METRICS,
                timestamp=datetime.now(),
                source="performance_test",
                data={
                    'name': f'test_metric_{i % 10}',
                    'value': 50 + (i % 100),
                    'category': 'performance',
                    'context': {'test_id': i}
                },
                priority=1 + (i % 5)
            )
            messages.append(message)
        
        return messages
    
    async def _ingest_stream_batch(self, processor: RealTimeDataProcessor,
                                 stream_type: StreamType, count: int, priority: int) -> float:
        """Ingest a batch of messages for a specific stream type"""
        messages = []
        
        for i in range(count):
            message = StreamMessage(
                id=f"{stream_type.value}_batch_{i}_{time.time()}",
                stream_type=stream_type,
                timestamp=datetime.now(),
                source="batch_test",
                data={
                    'type': stream_type.value,
                    'value': i,
                    'batch_id': f"batch_{stream_type.value}"
                },
                priority=priority
            )
            messages.append(message)
        
        # Ingest messages
        tasks = []
        for message in messages:
            task = asyncio.create_task(processor.ingest_data(message))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Return success rate
        return sum(results) / len(results) if results else 0

class TestScalabilityBenchmarks:
    """Scalability benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_horizontal_scaling_simulation(self, performance_suite):
        """Simulate horizontal scaling scenarios"""
        processor = performance_suite.data_processor
        
        # Simulate multiple processor instances
        scaling_factors = [1, 2, 4, 8]  # Number of simulated instances
        results = {}
        
        for factor in scaling_factors:
            # Simulate distributed load
            messages_per_instance = 1000
            total_messages = messages_per_instance * factor
            
            # Create message batches for each "instance"
            instance_tasks = []
            
            for instance in range(factor):
                messages = self._generate_distributed_messages(
                    messages_per_instance, instance, factor
                )
                
                task = asyncio.create_task(
                    self._process_instance_batch(processor, messages)
                )
                instance_tasks.append(task)
            
            start_time = time.time()
            
            # Process all instances concurrently
            instance_results = await asyncio.gather(*instance_tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate metrics
            total_throughput = total_messages / duration
            avg_success_rate = sum(instance_results) / len(instance_results)
            
            results[factor] = {
                'throughput': total_throughput,
                'duration': duration,
                'success_rate': avg_success_rate,
                'efficiency': total_throughput / factor  # Throughput per instance
            }
            
            print(f"Scaling factor {factor}: {total_throughput:.2f} msg/s, "
                  f"Efficiency: {results[factor]['efficiency']:.2f} msg/s per instance")
        
        # Analyze scaling efficiency
        baseline_efficiency = results[1]['efficiency']
        
        for factor in scaling_factors[1:]:
            efficiency_ratio = results[factor]['efficiency'] / baseline_efficiency
            print(f"Scaling factor {factor}: {efficiency_ratio:.2%} efficiency retention")
            
            # Should maintain at least 80% efficiency
            assert efficiency_ratio >= 0.8, f"Scaling efficiency should be >=80%, got {efficiency_ratio:.2%}"
    
    def _generate_distributed_messages(self, count: int, instance_id: int, 
                                     total_instances: int) -> List[StreamMessage]:
        """Generate messages for distributed processing simulation"""
        messages = []
        
        for i in range(count):
            message = StreamMessage(
                id=f"distributed_{instance_id}_{i}_{time.time()}",
                stream_type=StreamType.METRICS,
                timestamp=datetime.now(),
                source=f"instance_{instance_id}",
                data={
                    'name': f'distributed_metric_{i % 5}',
                    'value': 50 + (i % 100),
                    'instance_id': instance_id,
                    'total_instances': total_instances
                },
                priority=1 + (i % 3)
            )
            messages.append(message)
        
        return messages
    
    async def _process_instance_batch(self, processor: RealTimeDataProcessor,
                                    messages: List[StreamMessage]) -> float:
        """Process a batch of messages for one instance"""
        tasks = []
        for message in messages:
            task = asyncio.create_task(processor.ingest_data(message))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return sum(results) / len(results) if results else 0

# Integration test for complete real-time pipeline
@pytest.mark.asyncio
async def test_end_to_end_realtime_pipeline_performance():
    """Test complete real-time pipeline performance"""
    # This would test the entire pipeline from data ingestion
    # through processing, alerting, and notification
    
    suite = PerformanceTestSuite()
    await suite.setup()
    
    try:
        # Simulate realistic workload
        start_time = time.time()
        
        # Generate mixed workload
        metrics_messages = suite._generate_test_messages(2000)
        
        # Add some threshold rules
        for i in range(10):
            rule = ThresholdRule(
                id=f"e2e_rule_{i}",
                metric_name=f"test_metric_{i}",
                operator=">",
                value=75.0,
                severity=AlertSeverity.MEDIUM,
                description=f"End-to-end test rule {i}"
            )
            await suite.alerting_system.add_threshold_rule(rule)
        
        # Process messages
        tasks = []
        for message in metrics_messages:
            task = asyncio.create_task(suite.data_processor.ingest_data(message))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Allow processing to complete
        await asyncio.sleep(2)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate end-to-end performance
        throughput = len(metrics_messages) / duration
        success_rate = sum(results) / len(results)
        
        print(f"End-to-end pipeline: {throughput:.2f} msg/s, {success_rate:.2%} success rate")
        
        # Performance requirements
        assert throughput >= 300, "End-to-end pipeline should handle 300+ msg/s"
        assert success_rate >= 0.95, "End-to-end pipeline should have 95%+ success rate"
        
    finally:
        await suite.teardown()

if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])