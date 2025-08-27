"""
Tests for visual generation monitoring and observability system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from scrollintel.engines.visual_generation.monitoring import (
    MetricsCollector,
    DistributedTracer,
    VisualGenerationObservabilityDashboard,
    MetricType,
    SpanKind,
    SpanStatus,
    AlertSeverity
)


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.fixture
    async def metrics_collector(self):
        """Create metrics collector for testing."""
        collector = MetricsCollector(collection_interval=1)
        await collector.start()
        yield collector
        await collector.stop()
    
    async def test_counter_metrics(self, metrics_collector):
        """Test counter metric recording."""
        # Record counter metrics
        metrics_collector.increment_counter("test_counter", 1.0, {"type": "test"})
        metrics_collector.increment_counter("test_counter", 2.0, {"type": "test"})
        
        # Check counter value
        key = "test_counter[type=test]"
        assert metrics_collector.counters[key] == 3.0
        
        # Check metrics buffer
        assert len(metrics_collector.metrics_buffer) >= 2
    
    async def test_gauge_metrics(self, metrics_collector):
        """Test gauge metric recording."""
        # Set gauge metrics
        metrics_collector.set_gauge("test_gauge", 42.0, {"component": "test"})
        metrics_collector.set_gauge("test_gauge", 84.0, {"component": "test"})
        
        # Check gauge value (should be latest)
        key = "test_gauge[component=test]"
        assert metrics_collector.gauges[key] == 84.0
    
    async def test_histogram_metrics(self, metrics_collector):
        """Test histogram metric recording."""
        # Record histogram values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            metrics_collector.record_histogram("test_histogram", value)
        
        # Check histogram values
        assert "test_histogram" in metrics_collector.histograms
        assert len(metrics_collector.histograms["test_histogram"]) == 5
        assert sum(metrics_collector.histograms["test_histogram"]) == 15.0
    
    async def test_timer_metrics(self, metrics_collector):
        """Test timer functionality."""
        # Start timer
        timer_key = metrics_collector.start_timer("test_operation", "req_123")
        assert timer_key in metrics_collector.active_requests
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        # End timer
        duration = metrics_collector.end_timer(timer_key)
        assert duration is not None
        assert duration >= 0.1
        assert timer_key not in metrics_collector.active_requests
    
    async def test_request_tracking(self, metrics_collector):
        """Test request tracking functionality."""
        request_id = "test_request_123"
        request_type = "image"
        
        # Record request start
        metrics_collector.record_request_start(request_id, request_type, "user123")
        
        # Simulate processing time
        await asyncio.sleep(0.05)
        
        # Record completion
        metrics_collector.record_request_completion(
            request_id, request_type, True, quality_score=0.95
        )
        
        # Check completed requests
        assert len(metrics_collector.completed_requests) >= 1
        
        completed_request = metrics_collector.completed_requests[-1]
        assert completed_request["request_id"] == request_id
        assert completed_request["success"] is True
        assert completed_request["quality_score"] == 0.95
    
    async def test_performance_metrics_calculation(self, metrics_collector):
        """Test performance metrics calculation."""
        # Record some test requests
        for i in range(10):
            request_id = f"test_request_{i}"
            metrics_collector.record_request_start(request_id, "image")
            await asyncio.sleep(0.01)
            
            success = i < 8  # 80% success rate
            quality_score = 0.9 if success else None
            error_type = None if success else "test_error"
            
            metrics_collector.record_request_completion(
                request_id, "image", success, quality_score, error_type
            )
        
        # Get performance metrics
        performance = metrics_collector.get_performance_metrics()
        
        assert performance.total_requests >= 10
        assert performance.successful_requests >= 8
        assert performance.failed_requests >= 2
        assert 0.0 <= performance.error_rate <= 1.0
        assert performance.average_response_time > 0.0
    
    async def test_cache_operation_tracking(self, metrics_collector):
        """Test cache operation tracking."""
        # Record cache operations
        metrics_collector.record_cache_operation("get", True, "image")  # Hit
        metrics_collector.record_cache_operation("get", False, "image")  # Miss
        metrics_collector.record_cache_operation("put", True, "video")  # Store
        
        # Check counters
        hit_key = "visual_generation_cache_operations[content_type=image,operation=get,status=hit]"
        miss_key = "visual_generation_cache_operations[content_type=image,operation=get,status=miss]"
        
        assert metrics_collector.counters[hit_key] == 1.0
        assert metrics_collector.counters[miss_key] == 1.0
    
    async def test_worker_metrics_recording(self, metrics_collector):
        """Test worker metrics recording."""
        worker_id = "worker_123"
        worker_metrics = {
            "gpu_utilization": 0.75,
            "memory_usage": 0.60,
            "cpu_usage": 0.45,
            "active_jobs": 3,
            "queue_length": 5
        }
        
        metrics_collector.record_worker_metrics(worker_id, worker_metrics)
        
        # Check gauge metrics
        gpu_key = f"visual_generation_worker_gpu_utilization[worker_id={worker_id}]"
        memory_key = f"visual_generation_worker_memory_usage[worker_id={worker_id}]"
        
        assert metrics_collector.gauges[gpu_key] == 0.75
        assert metrics_collector.gauges[memory_key] == 0.60


class TestDistributedTracer:
    """Test distributed tracing functionality."""
    
    @pytest.fixture
    async def tracer(self):
        """Create tracer for testing."""
        tracer = DistributedTracer("test_service")
        await tracer.start()
        yield tracer
        await tracer.stop()
    
    async def test_trace_creation(self, tracer):
        """Test trace and span creation."""
        # Start a new trace
        context = tracer.start_trace("test_operation", user_id="123", request_type="image")
        
        assert context.trace_id is not None
        assert context.span_id is not None
        assert context.parent_span_id is None
        
        # Get current span
        current_span = tracer._get_current_span()
        assert current_span is not None
        assert current_span.operation_name == "test_operation"
        assert current_span.tags["user_id"] == "123"
    
    async def test_child_span_creation(self, tracer):
        """Test child span creation."""
        # Start parent trace
        parent_context = tracer.start_trace("parent_operation")
        
        # Start child span
        child_context = tracer.start_span("child_operation", parent_context)
        
        assert child_context.trace_id == parent_context.trace_id
        assert child_context.parent_span_id == parent_context.span_id
        assert child_context.span_id != parent_context.span_id
    
    async def test_span_finishing(self, tracer):
        """Test span finishing and recording."""
        # Start trace
        context = tracer.start_trace("test_operation")
        
        # Simulate some work
        await asyncio.sleep(0.05)
        
        # Finish span
        tracer.finish_span(context)
        
        # Check that span was recorded
        spans = tracer.collector.get_trace(context.trace_id)
        assert len(spans) == 1
        
        span = spans[0]
        assert span.operation_name == "test_operation"
        assert span.duration is not None
        assert span.duration >= 0.05
        assert span.status == SpanStatus.OK
    
    async def test_span_error_handling(self, tracer):
        """Test span error handling."""
        context = tracer.start_trace("error_operation")
        
        # Simulate error
        test_error = ValueError("Test error")
        tracer.finish_span(context, SpanStatus.ERROR, test_error)
        
        # Check error recording
        spans = tracer.collector.get_trace(context.trace_id)
        span = spans[0]
        
        assert span.status == SpanStatus.ERROR
        assert span.error == "Test error"
        assert span.tags["error"] is True
        assert span.tags["error.type"] == "ValueError"
    
    async def test_context_propagation(self, tracer):
        """Test context propagation through headers."""
        # Create context
        context = tracer.start_trace("test_operation")
        context.baggage["user_id"] = "123"
        
        # Inject into headers
        headers = tracer.inject_context(context)
        
        assert "X-Trace-Id" in headers
        assert "X-Span-Id" in headers
        assert headers["X-Trace-Id"] == context.trace_id
        
        # Extract from headers
        extracted_context = tracer.extract_context(headers)
        
        assert extracted_context is not None
        assert extracted_context.trace_id == context.trace_id
        assert extracted_context.baggage["user_id"] == "123"
    
    async def test_trace_context_manager(self, tracer):
        """Test trace context manager."""
        trace_id = None
        
        async with tracer.trace("context_manager_operation") as span:
            trace_id = span.context.trace_id
            span.set_tag("test_tag", "test_value")
            span.log("Test log message")
            
            # Simulate some work
            await asyncio.sleep(0.01)
        
        # Check recorded span
        spans = tracer.collector.get_trace(trace_id)
        assert len(spans) == 1
        
        span = spans[0]
        assert span.operation_name == "context_manager_operation"
        assert span.tags["test_tag"] == "test_value"
        assert len(span.logs) >= 1
        assert span.status == SpanStatus.OK
    
    async def test_trace_search(self, tracer):
        """Test trace search functionality."""
        # Create multiple traces
        for i in range(5):
            context = tracer.start_trace(f"operation_{i}", service="test_service")
            
            if i % 2 == 0:
                # Make some traces have errors
                tracer.finish_span(context, SpanStatus.ERROR, ValueError("Test error"))
            else:
                tracer.finish_span(context, SpanStatus.OK)
        
        # Search for error traces
        error_traces = tracer.collector.search_traces(has_errors=True, limit=10)
        assert len(error_traces) >= 2  # Should find error traces
        
        # Search by service name
        service_traces = tracer.collector.search_traces(service_name="test_service", limit=10)
        assert len(service_traces) >= 5


class TestObservabilityDashboard:
    """Test observability dashboard functionality."""
    
    @pytest.fixture
    async def dashboard(self):
        """Create dashboard for testing."""
        metrics_collector = MetricsCollector(collection_interval=1)
        tracer = DistributedTracer("test_service")
        
        await metrics_collector.start()
        await tracer.start()
        
        dashboard = VisualGenerationObservabilityDashboard(metrics_collector, tracer)
        await dashboard.initialize()
        
        yield dashboard
        
        await dashboard.shutdown()
        await tracer.stop()
        await metrics_collector.stop()
    
    async def test_dashboard_data_collection(self, dashboard):
        """Test dashboard data collection."""
        # Generate some test data
        dashboard.metrics_collector.increment_counter("test_requests", 10)
        dashboard.metrics_collector.set_gauge("test_cpu", 75.0)
        
        # Get dashboard data
        data = await dashboard.get_dashboard_data()
        
        assert "timestamp" in data
        assert "performance" in data
        assert "system_overview" in data
        assert "active_alerts" in data
        assert "health_status" in data
    
    async def test_alert_rule_management(self, dashboard):
        """Test alert rule management."""
        # Add custom alert rule
        dashboard.add_alert_rule(
            "test_alert",
            "error_rate",
            0.1,  # 10%
            AlertSeverity.WARNING,
            "Test alert description"
        )
        
        assert "test_alert" in dashboard.alert_rules
        
        rule = dashboard.alert_rules["test_alert"]
        assert rule["metric_name"] == "error_rate"
        assert rule["threshold"] == 0.1
        assert rule["severity"] == AlertSeverity.WARNING
    
    async def test_health_check_registration(self, dashboard):
        """Test health check registration."""
        async def test_health_check():
            from scrollintel.engines.visual_generation.monitoring.observability_dashboard import HealthCheck
            return HealthCheck(
                component="test_component",
                status="healthy",
                message="Test component is healthy",
                timestamp=datetime.now()
            )
        
        dashboard.add_health_check("test_component", test_health_check)
        
        assert "test_component" in dashboard.health_checks
        
        # Run health checks
        await dashboard._run_health_checks()
        
        assert "test_component" in dashboard.health_status
        health_check = dashboard.health_status["test_component"]
        assert health_check.status == "healthy"
    
    async def test_capacity_planning(self, dashboard):
        """Test capacity planning functionality."""
        # Set up some performance data
        dashboard.metrics_collector.set_gauge("cpu_usage", 70.0)
        dashboard.metrics_collector.set_gauge("memory_usage", 60.0)
        dashboard.metrics_collector.increment_counter("requests_total", 100)
        
        # Get capacity planning data
        capacity_data = await dashboard.get_capacity_planning_data()
        
        assert "current_metrics" in capacity_data
        assert "capacity_estimates" in capacity_data
        assert "recommendations" in capacity_data
        
        current = capacity_data["current_metrics"]
        assert "cpu_usage_percent" in current
        assert "memory_usage_percent" in current
    
    async def test_error_analysis(self, dashboard):
        """Test error analysis functionality."""
        # Create some error traces
        for i in range(3):
            context = dashboard.tracer.start_trace(f"error_operation_{i}")
            error = ValueError(f"Test error {i}")
            dashboard.tracer.finish_span(context, SpanStatus.ERROR, error)
        
        # Get error analysis
        error_analysis = await dashboard.get_error_analysis(hours=1)
        
        assert "total_error_traces" in error_analysis
        assert "error_patterns" in error_analysis
        assert "errors_by_service" in error_analysis
        assert error_analysis["total_error_traces"] >= 3
    
    async def test_trace_analysis(self, dashboard):
        """Test individual trace analysis."""
        # Create a test trace with multiple spans
        parent_context = dashboard.tracer.start_trace("parent_operation")
        
        child_context = dashboard.tracer.start_span("child_operation", parent_context)
        dashboard.tracer.finish_span(child_context)
        
        dashboard.tracer.finish_span(parent_context)
        
        # Analyze the trace
        analysis = await dashboard.get_trace_analysis(parent_context.trace_id)
        
        assert analysis is not None
        assert analysis["trace_id"] == parent_context.trace_id
        assert analysis["total_spans"] == 2
        assert len(analysis["spans"]) == 2
        assert "critical_path" in analysis


@pytest.mark.asyncio
async def test_monitoring_integration():
    """Test integration between monitoring components."""
    # Initialize all components
    metrics_collector = MetricsCollector(collection_interval=1)
    tracer = DistributedTracer("integration_test")
    dashboard = VisualGenerationObservabilityDashboard(metrics_collector, tracer)
    
    await metrics_collector.start()
    await tracer.start()
    await dashboard.initialize()
    
    try:
        # Simulate a complete request workflow
        request_id = "integration_test_request"
        
        # Start trace
        context = tracer.start_trace("image_generation_request", 
                                   request_id=request_id, 
                                   user_id="test_user")
        
        # Record metrics
        metrics_collector.record_request_start(request_id, "image", "test_user")
        
        # Simulate processing with child spans
        async with tracer.trace("prompt_enhancement") as span:
            span.set_tag("original_prompt", "test prompt")
            await asyncio.sleep(0.01)
        
        async with tracer.trace("model_inference") as span:
            span.set_tag("model", "stable_diffusion_xl")
            await asyncio.sleep(0.02)
        
        async with tracer.trace("post_processing") as span:
            span.set_tag("enhancement", "upscaling")
            await asyncio.sleep(0.01)
        
        # Complete request
        tracer.finish_span(context)
        metrics_collector.record_request_completion(request_id, "image", True, 0.95)
        
        # Verify integration
        dashboard_data = await dashboard.get_dashboard_data()
        
        assert dashboard_data["performance"]["total_requests"] >= 1
        assert len(dashboard_data["recent_traces"]) >= 1
        
        # Verify trace was recorded
        spans = tracer.collector.get_trace(context.trace_id)
        assert len(spans) == 4  # Parent + 3 child spans
        
    finally:
        await dashboard.shutdown()
        await tracer.stop()
        await metrics_collector.stop()


if __name__ == "__main__":
    pytest.main([__file__])