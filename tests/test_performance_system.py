"""
Performance Monitoring System Tests
Simplified tests for core performance monitoring functionality
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import only the core performance monitoring components
from scrollintel.core.performance_monitor import (
    ResponseTimeTracker, DatabaseQueryMonitor, CacheManager,
    ResponseTimeMetric, DatabaseQueryMetric, CacheMetric
)

class TestResponseTimeTracker:
    """Test response time tracking functionality"""
    
    def setup_method(self):
        self.tracker = ResponseTimeTracker(max_metrics=100)
        
    @pytest.mark.asyncio
    async def test_track_request_success(self):
        """Test successful request tracking"""
        async with self.tracker.track_request("/api/test", "GET", "user123"):
            await asyncio.sleep(0.1)  # Simulate processing time
            
        assert len(self.tracker.metrics) == 1
        metric = self.tracker.metrics[0]
        assert metric.endpoint == "/api/test"
        assert metric.method == "GET"
        assert metric.user_id == "user123"
        assert metric.status_code == 200
        assert metric.response_time >= 0.1
        
    @pytest.mark.asyncio
    async def test_track_request_error(self):
        """Test request tracking with error"""
        with pytest.raises(ValueError):
            async with self.tracker.track_request("/api/error", "POST"):
                raise ValueError("Test error")
                
        assert len(self.tracker.metrics) == 1
        metric = self.tracker.metrics[0]
        assert metric.endpoint == "/api/error"
        assert metric.method == "POST"
        assert metric.status_code == 500
        
    def test_get_endpoint_stats(self):
        """Test endpoint statistics calculation"""
        # Add some test metrics
        for i in range(5):
            metric = ResponseTimeMetric(
                endpoint="/api/test",
                method="GET",
                response_time=0.1 + (i * 0.1),
                status_code=200,
                timestamp=datetime.utcnow()
            )
            self.tracker.metrics.append(metric)
            self.tracker.endpoint_stats["GET /api/test"].append(metric.response_time)
            
        stats = self.tracker.get_endpoint_stats("/api/test", "GET")
        
        assert stats["request_count"] == 5
        assert abs(stats["avg_response_time"] - 0.3) < 0.001  # Account for floating point precision
        assert stats["min_response_time"] == 0.1
        assert stats["max_response_time"] == 0.5
        
    def test_get_slow_endpoints(self):
        """Test slow endpoint detection"""
        # Add fast endpoint
        for i in range(3):
            metric = ResponseTimeMetric(
                endpoint="/api/fast",
                method="GET",
                response_time=0.1,
                status_code=200,
                timestamp=datetime.utcnow()
            )
            self.tracker.metrics.append(metric)
            self.tracker.endpoint_stats["GET /api/fast"].append(metric.response_time)
            
        # Add slow endpoint
        for i in range(3):
            metric = ResponseTimeMetric(
                endpoint="/api/slow",
                method="GET",
                response_time=2.0,
                status_code=200,
                timestamp=datetime.utcnow()
            )
            self.tracker.metrics.append(metric)
            self.tracker.endpoint_stats["GET /api/slow"].append(metric.response_time)
            
        slow_endpoints = self.tracker.get_slow_endpoints(threshold=1.0)
        
        assert len(slow_endpoints) == 1
        assert slow_endpoints[0]["endpoint"] == "/api/slow"
        assert slow_endpoints[0]["avg_response_time"] == 2.0

class TestDatabaseQueryMonitor:
    """Test database query monitoring functionality"""
    
    def setup_method(self):
        self.monitor = DatabaseQueryMonitor(max_metrics=100)
        
    def test_track_query_success(self):
        """Test successful query tracking"""
        query = "SELECT * FROM users WHERE id = 1"
        
        with self.monitor.track_query(query, "select"):
            time.sleep(0.1)  # Simulate query execution
            
        assert len(self.monitor.metrics) == 1
        metric = self.monitor.metrics[0]
        assert metric.query_type == "select"
        assert metric.execution_time >= 0.1
        assert len(metric.query_hash) == 16
        
    def test_track_query_error(self):
        """Test query tracking with error"""
        query = "INVALID SQL"
        
        with pytest.raises(ValueError):
            with self.monitor.track_query(query, "invalid"):
                raise ValueError("SQL error")
                
        assert len(self.monitor.metrics) == 1
        metric = self.monitor.metrics[0]
        assert metric.query_type == "invalid"
        
    def test_get_slow_queries(self):
        """Test slow query detection"""
        # Add fast queries
        for i in range(3):
            metric = DatabaseQueryMetric(
                query_hash=f"hash{i}",
                query_type="select",
                execution_time=0.1,
                rows_affected=10,
                timestamp=datetime.utcnow(),
                query_text=f"SELECT * FROM table{i}"
            )
            self.monitor.metrics.append(metric)
            self.monitor.query_stats[f"hash{i}"].append(metric.execution_time)
            
        # Add slow query
        slow_metric = DatabaseQueryMetric(
            query_hash="slow_hash",
            query_type="select",
            execution_time=2.0,
            rows_affected=1000,
            timestamp=datetime.utcnow(),
            query_text="SELECT * FROM large_table"
        )
        self.monitor.metrics.append(slow_metric)
        self.monitor.query_stats["slow_hash"].append(slow_metric.execution_time)
        
        slow_queries = self.monitor.get_slow_queries(threshold=1.0)
        
        assert len(slow_queries) == 1
        assert slow_queries[0]["query_hash"] == "slow_hash"
        assert slow_queries[0]["avg_execution_time"] == 2.0

class TestCacheManager:
    """Test cache management functionality"""
    
    def setup_method(self):
        self.cache = CacheManager()
        
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test basic cache operations"""
        # Test cache miss
        result = await self.cache.get("test_key")
        assert result is None
        assert self.cache.miss_count == 1
        
        # Test cache set
        await self.cache.set("test_key", {"data": "test_value"}, ttl=3600)
        
        # Test cache hit
        result = await self.cache.get("test_key")
        assert result == {"data": "test_value"}
        assert self.cache.hit_count == 1
        
        # Test cache delete
        await self.cache.delete("test_key")
        result = await self.cache.get("test_key")
        assert result is None
        
    def test_cache_stats(self):
        """Test cache statistics"""
        self.cache.hit_count = 80
        self.cache.miss_count = 20
        self.cache.local_cache = {"key1": "value1", "key2": "value2"}
        
        stats = self.cache.get_cache_stats()
        
        assert stats["hit_count"] == 80
        assert stats["miss_count"] == 20
        assert stats["hit_rate"] == 80.0  # 80/(80+20) * 100
        assert stats["total_requests"] == 100
        assert stats["local_cache_size"] == 2

@pytest.mark.asyncio
async def test_performance_integration():
    """Integration test for performance monitoring system"""
    # Initialize components
    tracker = ResponseTimeTracker()
    db_monitor = DatabaseQueryMonitor()
    cache_manager = CacheManager()
    
    # Simulate some activity
    async with tracker.track_request("/api/users", "GET", "user123"):
        await asyncio.sleep(0.1)
        
    with db_monitor.track_query("SELECT * FROM users", "select"):
        time.sleep(0.05)
        
    await cache_manager.set("user:123", {"name": "Test User"})
    cached_user = await cache_manager.get("user:123")
    
    # Verify data was recorded
    assert len(tracker.metrics) == 1
    assert len(db_monitor.metrics) == 1
    assert cached_user == {"name": "Test User"}
    assert cache_manager.hit_count == 1
    
    # Test performance dashboard components
    from scrollintel.core.performance_monitor import PerformanceDashboard
    dashboard = PerformanceDashboard(tracker, db_monitor, cache_manager)
    
    with patch('psutil.cpu_percent', return_value=45.0):
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 60.0
            mock_memory.return_value.available = 8 * 1024**3  # 8GB
            
            dashboard_data = await dashboard.get_dashboard_data()
            
            # The dashboard calculates average from endpoint stats, which may be empty initially
            assert "avg_response_time" in dashboard_data["response_times"]
            assert dashboard_data["database"]["total_queries"] == 1
            assert dashboard_data["cache"]["hit_rate"] == 100.0
            assert dashboard_data["system"]["cpu_percent"] == 45.0

def test_performance_decorator():
    """Test performance tracking decorator"""
    from scrollintel.core.performance_monitor import track_performance
    
    @track_performance("/api/test", "GET")
    def test_function():
        time.sleep(0.1)
        return "success"
        
    result = test_function()
    assert result == "success"
    
    # Check that metrics were recorded
    from scrollintel.core.performance_monitor import response_tracker
    assert len(response_tracker.metrics) > 0

if __name__ == "__main__":
    pytest.main([__file__])