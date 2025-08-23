"""
Comprehensive test suite for production hardening components
Tests security, monitoring, caching, error handling, and database optimization.
"""
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from scrollintel.core.pipeline_monitoring import (
    MetricsCollector, StructuredLogger, HealthChecker, 
    monitor_performance, metrics_collector, structured_logger, health_checker
)
from scrollintel.core.pipeline_error_handling import (
    PipelineErrorHandler, CircuitBreaker, RetryPolicy, ErrorSeverity,
    RecoveryStrategy, with_error_handling, error_handler
)
from scrollintel.core.pipeline_cache import (
    IntelligentCache, QueryOptimizer, PerformanceMonitor,
    cached, intelligent_cache, query_optimizer, performance_monitor
)
from scrollintel.core.pipeline_security import (
    InputValidator, AccessController, SecurityAuditor, SecurityContext,
    require_permission, validate_input, input_validator, access_controller, security_auditor
)

class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def test_record_metric(self):
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 42.0, {"tag": "value"}, "units")
        
        metrics = collector.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0].name == "test_metric"
        assert metrics[0].value == 42.0
        assert metrics[0].labels == {"tag": "value"}
        assert metrics[0].unit == "units"
    
    def test_record_event(self):
        collector = MetricsCollector()
        
        collector.record_event("test_event", "pipeline_123", "user_456", 
                              {"detail": "test"}, success=True)
        
        events = collector.get_events("pipeline_123")
        assert len(events) == 1
        assert events[0].event_type == "test_event"
        assert events[0].pipeline_id == "pipeline_123"
        assert events[0].user_id == "user_456"
        assert events[0].success is True
    
    def test_get_summary_stats(self):
        collector = MetricsCollector()
        
        collector.record_metric("metric1", 1.0)
        collector.record_metric("metric2", 2.0)
        collector.record_event("event1", "pipeline1", "user1", {})
        
        stats = collector.get_summary_stats()
        assert stats['total_metrics'] == 2
        assert stats['total_events'] == 1
        assert "metric1" in stats['metric_types']
        assert "metric2" in stats['metric_types']

class TestStructuredLogger:
    """Test structured logging functionality"""
    
    def test_log_pipeline_operation(self):
        logger = StructuredLogger("test_logger")
        
        with patch.object(logger.logger, 'log') as mock_log:
            logger.log_pipeline_operation(
                "test_operation", "pipeline_123", "user_456",
                {"key": "value"}, success=True, duration=1.5
            )
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[1] == "Pipeline test_operation"
            assert kwargs['extra']['operation'] == "test_operation"
            assert kwargs['extra']['pipeline_id'] == "pipeline_123"
            assert kwargs['extra']['success'] is True
    
    def test_log_validation_result(self):
        logger = StructuredLogger("test_logger")
        
        with patch.object(logger.logger, 'log') as mock_log:
            logger.log_validation_result("pipeline_123", False, 
                                       ["error1", "error2"], ["warning1"])
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert kwargs['extra']['is_valid'] is False
            assert kwargs['extra']['error_count'] == 2
            assert kwargs['extra']['warning_count'] == 1

class TestHealthChecker:
    """Test health checking functionality"""
    
    def test_register_and_run_checks(self):
        checker = HealthChecker()
        
        # Register mock health checks
        checker.register_check("test_check_pass", lambda: True)
        checker.register_check("test_check_fail", lambda: False)
        checker.register_check("test_check_error", lambda: 1/0)  # Will raise exception
        
        results = checker.run_health_checks()
        
        assert results['overall_status'] == 'unhealthy'  # Due to failed checks
        assert results['checks']['test_check_pass']['status'] == 'healthy'
        assert results['checks']['test_check_fail']['status'] == 'unhealthy'
        assert results['checks']['test_check_error']['status'] == 'error'
        assert 'system' in results

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_closed_state(self):
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Should work normally when closed
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state.value == "closed"
    
    def test_circuit_breaker_opens_after_failures(self):
        breaker = CircuitBreaker(failure_threshold=2)
        
        # Cause failures to open circuit
        for _ in range(2):
            try:
                breaker.call(lambda: 1/0)  # Will raise exception
            except:
                pass
        
        assert breaker.state.value == "open"
        
        # Should raise exception when open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            breaker.call(lambda: "success")

class TestRetryPolicy:
    """Test retry policy functionality"""
    
    def test_get_delay_exponential_backoff(self):
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
    
    def test_get_delay_with_max_delay(self):
        policy = RetryPolicy(base_delay=10.0, max_delay=15.0, exponential_base=2.0, jitter=False)
        
        assert policy.get_delay(0) == 10.0
        assert policy.get_delay(1) == 15.0  # Capped at max_delay
        assert policy.get_delay(2) == 15.0  # Still capped

class TestPipelineErrorHandler:
    """Test error handling functionality"""
    
    def test_classify_error(self):
        handler = PipelineErrorHandler()
        
        # Test connection error classification
        connection_error = Exception("Connection timeout occurred")
        severity, strategy = handler.classify_error(connection_error, {})
        assert severity == ErrorSeverity.MEDIUM
        assert strategy == RecoveryStrategy.RETRY
        
        # Test validation error classification
        validation_error = Exception("Schema validation failed")
        severity, strategy = handler.classify_error(validation_error, {})
        assert severity == ErrorSeverity.HIGH
        assert strategy == RecoveryStrategy.SKIP
    
    def test_handle_error(self):
        handler = PipelineErrorHandler()
        
        error = Exception("Test error")
        pipeline_error = handler.handle_error(error, "pipeline_123", "node_456", {"context": "test"})
        
        assert pipeline_error.pipeline_id == "pipeline_123"
        assert pipeline_error.node_id == "node_456"
        assert pipeline_error.message == "Test error"
        assert pipeline_error.error_type == "Exception"
        assert pipeline_error.context == {"context": "test"}
    
    @pytest.mark.asyncio
    async def test_execute_with_recovery_success(self):
        handler = PipelineErrorHandler()
        
        async def test_func():
            return "success"
        
        result = await handler.execute_with_recovery(
            test_func, "test_operation", "pipeline_123"
        )
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_with_recovery_retry(self):
        handler = PipelineErrorHandler()
        handler.retry_policies["test_operation"] = RetryPolicy(max_attempts=3, base_delay=0.01)
        
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await handler.execute_with_recovery(
            test_func, "test_operation", "pipeline_123"
        )
        
        assert result == "success"
        assert call_count == 3

class TestIntelligentCache:
    """Test caching functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        cache = IntelligentCache(max_size_mb=1)
        
        await cache.set("test_operation", {"param": "value"}, "cached_result", ttl=3600)
        
        result = await cache.get("test_operation", {"param": "value"})
        assert result == "cached_result"
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        cache = IntelligentCache(max_size_mb=1)
        
        result = await cache.get("nonexistent_operation", {"param": "value"})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        cache = IntelligentCache(max_size_mb=1)
        
        await cache.set("test_operation", {"param": "value"}, "cached_result", ttl=1)
        
        # Should be available immediately
        result = await cache.get("test_operation", {"param": "value"})
        assert result == "cached_result"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        result = await cache.get("test_operation", {"param": "value"})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        cache = IntelligentCache(max_size_mb=1)
        
        await cache.set("test_operation", {"param": "value"}, "cached_result", 
                       ttl=3600, tags=["tag1", "tag2"])
        
        # Should be available
        result = await cache.get("test_operation", {"param": "value"})
        assert result == "cached_result"
        
        # Invalidate by tag
        await cache.invalidate(tags=["tag1"])
        
        # Should be gone now
        result = await cache.get("test_operation", {"param": "value"})
        assert result is None
    
    def test_cache_stats(self):
        cache = IntelligentCache(max_size_mb=1)
        
        stats = cache.get_stats()
        assert 'hit_count' in stats
        assert 'miss_count' in stats
        assert 'hit_rate_percent' in stats
        assert 'cache_size_entries' in stats

class TestQueryOptimizer:
    """Test query optimization functionality"""
    
    @pytest.mark.asyncio
    async def test_execute_optimized_query(self):
        optimizer = QueryOptimizer()
        
        # Mock database session
        mock_db = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [{'id': 1, 'name': 'test'}]
        mock_db.execute.return_value = mock_result
        
        result = await optimizer.execute_optimized_query(
            mock_db, "SELECT * FROM test", {"param": "value"}
        )
        
        assert result == [{'id': 1, 'name': 'test'}]
        mock_db.execute.assert_called_once()
    
    def test_get_query_stats(self):
        optimizer = QueryOptimizer()
        
        # Add some mock stats
        optimizer.query_stats["query1"] = {
            'query': 'SELECT * FROM table1',
            'execution_count': 5,
            'total_duration': 2.5,
            'avg_duration': 0.5,
            'max_duration': 1.0
        }
        
        stats = optimizer.get_query_stats()
        assert len(stats) == 1
        assert stats[0]['query'] == 'SELECT * FROM table1'
        assert stats[0]['avg_duration'] == 0.5

class TestInputValidator:
    """Test input validation functionality"""
    
    def test_validate_sql_injection(self):
        validator = InputValidator()
        
        # Safe input
        result = validator.validate_sql_injection("normal text")
        assert result.is_valid is True
        
        # SQL injection attempt
        result = validator.validate_sql_injection("'; DROP TABLE users; --")
        assert result.is_valid is False
        assert result.code == "SQL_INJECTION"
    
    def test_validate_xss(self):
        validator = InputValidator()
        
        # Safe input
        result = validator.validate_xss("normal text")
        assert result.is_valid is True
        
        # XSS attempt
        result = validator.validate_xss("<script>alert('xss')</script>")
        assert result.is_valid is False
        assert result.code == "XSS_ATTEMPT"
    
    def test_validate_path_traversal(self):
        validator = InputValidator()
        
        # Safe input
        result = validator.validate_path_traversal("normal/path")
        assert result.is_valid is True
        
        # Path traversal attempt
        result = validator.validate_path_traversal("../../../etc/passwd")
        assert result.is_valid is False
        assert result.code == "PATH_TRAVERSAL"
    
    def test_validate_command_injection(self):
        validator = InputValidator()
        
        # Safe input
        result = validator.validate_command_injection("normal text")
        assert result.is_valid is True
        
        # Command injection attempt
        result = validator.validate_command_injection("test; rm -rf /")
        assert result.is_valid is False
        assert result.code == "COMMAND_INJECTION"
    
    def test_validate_pipeline_config(self):
        validator = InputValidator()
        
        # Valid config
        valid_config = {
            "name": "Test Pipeline",
            "nodes": [
                {"id": "node1", "type": "source", "config": {"url": "http://example.com"}}
            ],
            "connections": [
                {"source": "node1", "target": "node2"}
            ]
        }
        
        results = validator.validate_pipeline_config(valid_config)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0
        
        # Invalid config - missing required fields
        invalid_config = {"name": "Test"}
        
        results = validator.validate_pipeline_config(invalid_config)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any(r.code == "MISSING_FIELD" for r in errors)

class TestAccessController:
    """Test access control functionality"""
    
    def test_check_permission(self):
        controller = AccessController()
        
        # Create security context
        context = Mock()
        context.roles = ['editor']
        context.permissions = set()
        
        # Editor should have read permission
        assert controller.check_permission(context, 'pipeline.read') is True
        
        # Editor should not have admin permission
        assert controller.check_permission(context, 'pipeline.admin') is False
    
    def test_get_user_permissions(self):
        controller = AccessController()
        
        permissions = controller.get_user_permissions(['editor', 'executor'])
        
        assert 'pipeline.read' in permissions
        assert 'pipeline.create' in permissions
        assert 'pipeline.execute' in permissions
        assert 'pipeline.admin' not in permissions

class TestSecurityAuditor:
    """Test security auditing functionality"""
    
    def test_log_security_event(self):
        auditor = SecurityAuditor()
        
        auditor.log_security_event(
            'access_attempt', 'user123', 'pipeline456', 'read', True,
            {'ip_address': '192.168.1.1'}
        )
        
        assert len(auditor.audit_log) == 1
        event = auditor.audit_log[0]
        assert event['event_type'] == 'access_attempt'
        assert event['user_id'] == 'user123'
        assert event['success'] is True
    
    def test_get_audit_log(self):
        auditor = SecurityAuditor()
        
        # Add some events
        auditor.log_security_event('event1', 'user1', 'resource1', 'action1', True)
        auditor.log_security_event('event2', 'user2', 'resource2', 'action2', False)
        
        # Get all events
        all_events = auditor.get_audit_log()
        assert len(all_events) == 2
        
        # Get events for specific user
        user1_events = auditor.get_audit_log(user_id='user1')
        assert len(user1_events) == 1
        assert user1_events[0]['user_id'] == 'user1'

class TestDecorators:
    """Test decorator functionality"""
    
    @pytest.mark.asyncio
    async def test_monitor_performance_decorator(self):
        @monitor_performance("test_operation")
        async def test_function(pipeline_id="test_pipeline"):
            await asyncio.sleep(0.01)  # Small delay
            return "success"
        
        with patch.object(structured_logger, 'log_pipeline_operation') as mock_log:
            result = await test_function()
            
            assert result == "success"
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[0] == "test_operation"  # operation name
    
    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        call_count = 0
        
        @cached(ttl=3600)
        async def expensive_function(param1, param2="default"):
            nonlocal call_count
            call_count += 1
            return f"result_{param1}_{param2}"
        
        # First call should execute function
        result1 = await expensive_function("test", param2="value")
        assert result1 == "result_test_value"
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_function("test", param2="value")
        assert result2 == "result_test_value"
        assert call_count == 1  # Should not increment
    
    def test_require_permission_decorator(self):
        @require_permission("pipeline.read")
        def protected_function(security_context=None):
            return "success"
        
        # Test with valid permission
        context = Mock()
        context.user_id = "user123"
        
        with patch.object(access_controller, 'check_permission', return_value=True):
            result = protected_function(security_context=context)
            assert result == "success"
        
        # Test without permission
        with patch.object(access_controller, 'check_permission', return_value=False):
            with pytest.raises(PermissionError):
                protected_function(security_context=context)
    
    def test_validate_input_decorator(self):
        @validate_input()
        def test_function(user_input="safe input"):
            return "success"
        
        # Test with safe input
        result = test_function(user_input="normal text")
        assert result == "success"
        
        # Test with dangerous input
        with pytest.raises(ValueError):
            test_function(user_input="'; DROP TABLE users; --")

class TestIntegration:
    """Integration tests for production hardening components"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_security_flow(self):
        """Test complete security flow for pipeline operations"""
        
        # Create security context
        context = Mock()
        context.user_id = "test_user"
        context.roles = ["editor"]
        context.permissions = set(["pipeline.read", "pipeline.create"])
        
        # Mock pipeline configuration
        pipeline_config = {
            "name": "Test Pipeline",
            "nodes": [
                {"id": "source1", "type": "source", "config": {"url": "http://example.com"}}
            ],
            "connections": []
        }
        
        # Validate configuration
        validation_results = input_validator.validate_pipeline_config(pipeline_config)
        errors = [r for r in validation_results if not r.is_valid]
        assert len(errors) == 0
        
        # Check permissions
        assert access_controller.check_permission(context, "pipeline.create") is True
        
        # Log security event
        security_auditor.log_security_event(
            "pipeline_create", context.user_id, "pipeline_123", "create", True
        )
        
        # Verify audit log
        events = security_auditor.get_audit_log(user_id=context.user_id)
        assert len(events) == 1
        assert events[0]["action"] == "create"
    
    @pytest.mark.asyncio
    async def test_error_handling_with_monitoring(self):
        """Test error handling integration with monitoring"""
        
        handler = PipelineErrorHandler()
        
        # Function that fails then succeeds
        attempt_count = 0
        
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        # Set up retry policy
        handler.retry_policies["test_op"] = RetryPolicy(max_attempts=5, base_delay=0.01)
        
        # Execute with recovery
        result = await handler.execute_with_recovery(
            flaky_function, "test_op", "pipeline_123"
        )
        
        assert result == "success"
        assert attempt_count == 3
        
        # Check that errors were recorded
        pipeline_errors = [e for e in handler.errors.values() if e.pipeline_id == "pipeline_123"]
        assert len(pipeline_errors) == 2  # Two failures before success
    
    @pytest.mark.asyncio
    async def test_caching_with_performance_monitoring(self):
        """Test caching integration with performance monitoring"""
        
        cache = IntelligentCache(max_size_mb=1)
        monitor = PerformanceMonitor()
        
        # Simulate expensive operation
        async def expensive_operation(param):
            await asyncio.sleep(0.01)  # Simulate work
            return f"result_{param}"
        
        # First call - cache miss
        start_time = time.time()
        result1 = await expensive_operation("test")
        duration1 = (time.time() - start_time) * 1000
        
        # Cache the result
        await cache.set("expensive_op", {"param": "test"}, result1)
        
        # Record performance
        monitor.record_performance("expensive_op", duration1, cache_hit=False)
        
        # Second call - cache hit
        start_time = time.time()
        cached_result = await cache.get("expensive_op", {"param": "test"})
        duration2 = (time.time() - start_time) * 1000
        
        # Record performance
        monitor.record_performance("expensive_op", duration2, cache_hit=True)
        
        assert cached_result == result1
        assert duration2 < duration1  # Cache should be faster
        
        # Check performance summary
        summary = monitor.get_performance_summary("expensive_op")
        assert summary["total_operations"] == 2
        assert summary["cache_hit_rate"] == 50.0  # 1 out of 2 operations was cached

if __name__ == "__main__":
    pytest.main([__file__, "-v"])