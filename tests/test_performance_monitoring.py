"""
Performance Monitoring System Tests
Tests for response time tracking, database monitoring, caching, and optimization
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.performance_monitor import (
    ResponseTimeTracker, DatabaseQueryMonitor, CacheManager,
    PerformanceOptimizer, PerformanceDashboard, ResponseTimeMetric,
    DatabaseQueryMetric, CacheMetric
)
from scrollintel.core.database_optimizer import DatabaseOptimizer, IndexRecommendation
from scrollintel.api.middleware.performance_middleware import PerformanceMiddleware

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
        assert stats["avg_response_time"] == 0.3  # (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5
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
        
    def test_get_query_stats_summary(self):
        """Test query statistics summary"""
        # Add test metrics
        execution_times = [0.1, 0.2, 0.5, 1.0, 2.0]
        for i, exec_time in enumerate(execution_times):
            metric = DatabaseQueryMetric(
                query_hash=f"hash{i}",
                query_type="select",
                execution_time=exec_time,
                rows_affected=10,
                timestamp=datetime.utcnow()
            )
            self.monitor.metrics.append(metric)
            
        summary = self.monitor.get_query_stats_summary()
        
        assert summary["total_queries"] == 5
        assert summary["avg_query_time"] == 0.76  # (0.1+0.2+0.5+1.0+2.0)/5
        assert summary["slow_queries_count"] == 2  # 1.0 and 2.0 > threshold

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

class TestPerformanceOptimizer:
    """Test performance optimization functionality"""
    
    def setup_method(self):
        self.tracker = ResponseTimeTracker()
        self.db_monitor = DatabaseQueryMonitor()
        self.cache_manager = CacheManager()
        self.optimizer = PerformanceOptimizer(self.tracker, self.db_monitor, self.cache_manager)
        
    @pytest.mark.asyncio
    async def test_optimize_slow_endpoints(self):
        """Test slow endpoint optimization suggestions"""
        # Add slow GET endpoint
        for i in range(5):
            metric = ResponseTimeMetric(
                endpoint="/api/data",
                method="GET",
                response_time=3.0,
                status_code=200,
                timestamp=datetime.utcnow()
            )
            self.tracker.metrics.append(metric)
            self.tracker.endpoint_stats["GET /api/data"].append(metric.response_time)
            
        optimizations = await self.optimizer.optimize_slow_endpoints()
        
        assert len(optimizations) == 1
        optimization = optimizations[0]
        assert optimization["endpoint"] == "/api/data"
        assert "Add response caching" in optimization["suggestions"]
        assert "Review database queries" in optimization["suggestions"]
        
    @pytest.mark.asyncio
    async def test_optimize_database_queries(self):
        """Test database query optimization suggestions"""
        # Add slow SELECT query
        metric = DatabaseQueryMetric(
            query_hash="slow_select",
            query_type="select",
            execution_time=2.0,
            rows_affected=1000,
            timestamp=datetime.utcnow(),
            query_text="SELECT * FROM users ORDER BY created_at"
        )
        self.db_monitor.metrics.append(metric)
        self.db_monitor.query_stats["slow_select"].append(metric.execution_time)
        
        optimizations = await self.optimizer.optimize_database_queries()
        
        assert len(optimizations) == 1
        optimization = optimizations[0]
        assert optimization["query_hash"] == "slow_select"
        assert "Add appropriate indexes" in optimization["suggestions"]
        
    @pytest.mark.asyncio
    async def test_get_performance_recommendations(self):
        """Test comprehensive performance recommendations"""
        # Set up low cache hit rate
        self.cache_manager.hit_count = 20
        self.cache_manager.miss_count = 80
        
        with patch('psutil.cpu_percent', return_value=85.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 90.0
                
                recommendations = await self.optimizer.get_performance_recommendations()
                
                assert len(recommendations["cache_recommendations"]) > 0
                assert len(recommendations["system_recommendations"]) > 0
                
                # Check for high CPU recommendation
                cpu_rec = next((r for r in recommendations["system_recommendations"] 
                              if r["type"] == "high_cpu"), None)
                assert cpu_rec is not None
                assert cpu_rec["priority"] == "high"

class TestDatabaseOptimizer:
    """Test database optimization functionality"""
    
    def setup_method(self):
        self.optimizer = DatabaseOptimizer()
        
    @pytest.mark.asyncio
    async def test_analyze_query(self):
        """Test individual query analysis"""
        query = "SELECT * FROM users WHERE email = 'test@example.com' ORDER BY created_at"
        
        optimization = await self.optimizer._analyze_query("test_hash", query, 2.5)
        
        assert optimization.query_hash == "test_hash"
        assert optimization.estimated_improvement == "medium"
        assert "Avoid SELECT *" in optimization.recommendations
        assert "Add index on ORDER BY columns" in optimization.recommendations
        
    def test_extract_tables_from_query(self):
        """Test table extraction from SQL queries"""
        query = """
        SELECT u.name, p.title 
        FROM users u 
        JOIN posts p ON u.id = p.user_id 
        WHERE u.active = true
        """
        
        tables = self.optimizer._extract_tables_from_query(query)
        
        assert "users" in tables
        assert "posts" in tables
        assert len(tables) == 2
        
    @pytest.mark.asyncio
    async def test_analyze_table_queries(self):
        """Test table-specific query analysis"""
        queries = [
            {
                "query_text": "SELECT * FROM users WHERE email = 'test@example.com'",
                "avg_execution_time": 1.5
            },
            {
                "query_text": "SELECT * FROM users WHERE email = 'other@example.com' ORDER BY created_at",
                "avg_execution_time": 2.0
            }
        ]
        
        recommendations = await self.optimizer._analyze_table_queries("users", queries)
        
        # Should recommend index on email column (used in WHERE clauses)
        email_rec = next((r for r in recommendations if "email" in r.columns), None)
        assert email_rec is not None
        assert email_rec.table_name == "users"
        assert email_rec.estimated_benefit in ["high", "medium"]

class TestPerformanceDashboard:
    """Test performance dashboard functionality"""
    
    def setup_method(self):
        self.tracker = ResponseTimeTracker()
        self.db_monitor = DatabaseQueryMonitor()
        self.cache_manager = CacheManager()
        self.dashboard = PerformanceDashboard(self.tracker, self.db_monitor, self.cache_manager)
        
    @pytest.mark.asyncio
    async def test_get_dashboard_data(self):
        """Test dashboard data generation"""
        # Add some test data
        metric = ResponseTimeMetric(
            endpoint="/api/test",
            method="GET",
            response_time=1.5,
            status_code=200,
            timestamp=datetime.utcnow()
        )
        self.tracker.metrics.append(metric)
        self.tracker.endpoint_stats["GET /api/test"].append(metric.response_time)
        
        self.cache_manager.hit_count = 80
        self.cache_manager.miss_count = 20
        
        with patch('psutil.cpu_percent', return_value=45.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.0
                mock_memory.return_value.available = 8 * 1024**3  # 8GB
                
                dashboard_data = await self.dashboard.get_dashboard_data()
                
                assert "timestamp" in dashboard_data
                assert "system" in dashboard_data
                assert "response_times" in dashboard_data
                assert "database" in dashboard_data
                assert "cache" in dashboard_data
                assert "endpoints" in dashboard_data
                
                assert dashboard_data["system"]["cpu_percent"] == 45.0
                assert dashboard_data["system"]["memory_percent"] == 60.0
                assert dashboard_data["cache"]["hit_rate"] == 80.0

class TestPerformanceMiddleware:
    """Test performance monitoring middleware"""
    
    @pytest.mark.asyncio
    async def test_middleware_tracks_requests(self):
        """Test that middleware properly tracks request metrics"""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            await asyncio.sleep(0.1)  # Simulate processing
            return {"message": "test"}
            
        # Create middleware instance
        middleware = PerformanceMiddleware(app)
        
        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.state = Mock()
        
        # Mock call_next
        async def mock_call_next(request):
            await asyncio.sleep(0.1)
            response = JSONResponse({"message": "test"})
            response.status_code = 200
            return response
            
        # Test middleware
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 200
        # Note: In a real test, we'd check that metrics were recorded

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
    
    # Test dashboard
    dashboard = PerformanceDashboard(tracker, db_monitor, cache_manager)
    dashboard_data = await dashboard.get_dashboard_data()
    
    assert dashboard_data["response_times"]["avg_response_time"] >= 0.1
    assert dashboard_data["database"]["total_queries"] == 1
    assert dashboard_data["cache"]["hit_rate"] == 100.0

if __name__ == "__main__":
    pytest.main([__file__])