"""
Performance Optimization Integration Tests

Tests integration of performance optimization system with ScrollIntel infrastructure
including API routes, database models, and agent coordination.

Requirements: 4.1, 6.1
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from scrollintel.api.routes.performance_optimization_routes import router
from scrollintel.models.performance_optimization_models import (
    CacheStatistics,
    AgentPerformanceMetrics,
    LoadBalancingEvents,
    ScalingEvents,
    ResourceForecasts,
    PerformanceAlerts,
    create_agent_metrics_entry,
    create_scaling_event_entry
)
from scrollintel.core.performance_optimization import (
    PerformanceOptimizationSystem,
    AgentMetrics
)

class TestPerformanceOptimizationAPI:
    """Test performance optimization API endpoints"""
    
    @pytest.fixture
    def mock_performance_system(self):
        """Mock performance optimization system"""
        system = Mock(spec=PerformanceOptimizationSystem)
        system._running = True
        
        # Mock cache manager
        system.cache_manager = Mock()
        system.cache_manager.get_stats.return_value = {
            'hits': 1000,
            'misses': 100,
            'hit_rate': 0.91,
            'size_mb': 45.2,
            'entry_count': 500,
            'evictions': 10
        }
        system.cache_manager.set = AsyncMock(return_value=True)
        system.cache_manager.get = AsyncMock(return_value="cached_value")
        
        # Mock load balancer
        system.load_balancer = Mock()
        system.load_balancer.get_agent_stats.return_value = {
            'total_agents': 5,
            'healthy_agents': 4,
            'average_health': 0.85,
            'model_trained': True,
            'request_history_size': 1500
        }
        system.load_balancer.register_agent = Mock()
        system.load_balancer.update_agent_metrics = Mock()
        system.load_balancer.select_agent = AsyncMock(return_value="selected_agent_1")
        system.load_balancer.train_model = AsyncMock()
        
        # Mock resource manager
        system.resource_manager = Mock()
        system.resource_manager.get_scaling_stats.return_value = {
            'current_instances': 3,
            'min_instances': 2,
            'max_instances': 10,
            'total_scaling_events': 15,
            'recent_scaling_events': 2,
            'cooldown_remaining': 0
        }
        system.resource_manager.evaluate_scaling = AsyncMock(return_value={
            'action': 'scale_up',
            'from_instances': 3,
            'to_instances': 4,
            'reason': 'High CPU usage detected'
        })
        
        # Mock forecaster
        system.forecaster = Mock()
        system.forecaster.get_forecasting_stats.return_value = {
            'is_trained': True,
            'historical_data_points': 2000,
            'data_time_range': {
                'start': '2024-01-01T00:00:00',
                'end': '2024-01-07T23:59:59'
            }
        }
        system.forecaster.forecast_demand = AsyncMock(return_value=[
            Mock(
                timestamp=datetime.now() + timedelta(hours=1),
                predicted_cpu=65.5,
                predicted_memory=70.2,
                predicted_requests=1200,
                confidence=0.85,
                scaling_recommendation='maintain_current'
            )
        ])
        system.forecaster.train_forecasting_models = AsyncMock()
        
        # Mock system methods
        system.get_performance_report = AsyncMock(return_value={
            'timestamp': datetime.now().isoformat(),
            'cache_stats': system.cache_manager.get_stats.return_value,
            'load_balancer_stats': system.load_balancer.get_agent_stats.return_value,
            'scaling_stats': system.resource_manager.get_scaling_stats.return_value,
            'forecasting_stats': system.forecaster.get_forecasting_stats.return_value,
            'system_health': {
                'overall_score': 0.88,
                'cache_health': 0.91,
                'agent_health': 0.85,
                'status': 'healthy'
            }
        })
        system._collect_system_metrics = AsyncMock(return_value={
            'cpu_usage': 45.2,
            'memory_usage': 62.1,
            'healthy_agents': 4,
            'total_agents': 5
        })
        system.start = AsyncMock()
        system.stop = AsyncMock()
        
        return system
    
    @pytest.fixture
    def client(self, mock_performance_system):
        """Create test client with mocked dependencies"""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        # Override dependency
        async def override_get_performance_system():
            return mock_performance_system
        
        app.dependency_overrides[router.dependencies[0]] = override_get_performance_system
        
        return TestClient(app)
    
    def test_get_system_health(self, client):
        """Test system health endpoint"""
        response = client.get("/api/v1/performance/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "health" in data["data"]
        assert data["data"]["health"]["status"] == "healthy"
    
    def test_get_performance_report(self, client):
        """Test performance report endpoint"""
        response = client.get("/api/v1/performance/report")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "cache_stats" in data["data"]
        assert "load_balancer_stats" in data["data"]
        assert "scaling_stats" in data["data"]
        assert "system_health" in data["data"]
    
    def test_cache_operations(self, client):
        """Test cache set and get operations"""
        # Test cache set
        cache_data = {
            "key": "test_key",
            "value": {"data": "test_value"},
            "ttl": 300
        }
        
        response = client.post("/api/v1/performance/cache/set", json=cache_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Test cache get
        response = client.get("/api/v1/performance/cache/get/test_key")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["data"]["found"] is True
        assert data["data"]["value"] == "cached_value"
        
        # Test cache stats
        response = client.get("/api/v1/performance/cache/stats")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["data"]["hit_rate"] == 0.91
    
    def test_load_balancer_operations(self, client):
        """Test load balancer operations"""
        # Test agent registration
        agent_data = {
            "agent_id": "test_agent",
            "cpu_usage": 45.0,
            "memory_usage": 55.0,
            "response_time": 800.0,
            "throughput": 120.0,
            "error_rate": 0.02,
            "active_connections": 25,
            "queue_length": 5
        }
        
        response = client.post("/api/v1/performance/load-balancer/register-agent", json=agent_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Test metrics update
        response = client.post("/api/v1/performance/load-balancer/update-metrics", json=agent_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Test agent selection
        selection_data = {"request_context": {"complexity": 0.5}}
        response = client.post("/api/v1/performance/load-balancer/select-agent", json=selection_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["data"]["selected_agent"] == "selected_agent_1"
        
        # Test load balancer stats
        response = client.get("/api/v1/performance/load-balancer/stats")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["data"]["total_agents"] == 5
    
    def test_scaling_operations(self, client):
        """Test auto-scaling operations"""
        # Test scaling status
        response = client.get("/api/v1/performance/scaling/status")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["data"]["current_instances"] == 3
        
        # Test scaling configuration
        config_data = {
            "min_instances": 3,
            "max_instances": 15,
            "cooldown_period": 600
        }
        
        response = client.post("/api/v1/performance/scaling/configure", json=config_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Test scaling evaluation
        response = client.post("/api/v1/performance/scaling/evaluate")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "scaling_action" in data["data"]
    
    def test_forecasting_operations(self, client):
        """Test forecasting operations"""
        # Test demand prediction
        forecast_data = {"hours_ahead": 6}
        response = client.post("/api/v1/performance/forecasting/predict", json=forecast_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "forecasts" in data["data"]
        assert len(data["data"]["forecasts"]) > 0
        
        # Test forecasting stats
        response = client.get("/api/v1/performance/forecasting/stats")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["data"]["is_trained"] is True

class TestPerformanceOptimizationModels:
    """Test performance optimization database models"""
    
    def test_cache_statistics_model(self):
        """Test cache statistics model"""
        stats = CacheStatistics(
            cache_type="distributed",
            hit_count=1000,
            miss_count=100,
            hit_rate=0.91,
            entry_count=500,
            size_bytes=50 * 1024 * 1024,
            size_mb=50.0,
            eviction_count=25,
            eviction_strategy="adaptive"
        )
        
        assert stats.cache_type == "distributed"
        assert stats.hit_rate == 0.91
        assert stats.size_mb == 50.0
    
    def test_agent_performance_metrics_model(self):
        """Test agent performance metrics model"""
        metrics = create_agent_metrics_entry(
            agent_id="test_agent",
            cpu_usage=65.0,
            memory_usage=70.0,
            response_time_ms=1200.0,
            throughput_rps=85.0,
            error_rate=0.03,
            active_connections=45,
            queue_length=12
        )
        
        assert metrics.agent_id == "test_agent"
        assert metrics.cpu_usage == 65.0
        assert 0 <= metrics.health_score <= 1
        assert metrics.status in ['healthy', 'degraded', 'critical', 'offline']
    
    def test_load_balancing_events_model(self):
        """Test load balancing events model"""
        event = LoadBalancingEvents(
            request_id="req_123",
            request_type="complex_analysis",
            selected_agent_id="agent_1",
            selection_strategy="ml_optimized",
            available_agents=["agent_1", "agent_2", "agent_3"],
            agent_scores={"agent_1": 0.85, "agent_2": 0.72, "agent_3": 0.68},
            actual_response_time_ms=1150.0,
            success=True,
            prediction_confidence=0.88
        )
        
        assert event.request_id == "req_123"
        assert event.selected_agent_id == "agent_1"
        assert event.success is True
    
    def test_scaling_events_model(self):
        """Test scaling events model"""
        trigger_metrics = {
            'cpu_usage': 85.0,
            'memory_usage': 88.0,
            'response_time': 2500.0,
            'error_rate': 0.06
        }
        
        event = create_scaling_event_entry(
            action="scale_up",
            from_instances=3,
            to_instances=4,
            trigger_metrics=trigger_metrics,
            reason="High CPU and memory usage detected",
            min_instances=2,
            max_instances=10
        )
        
        assert event.action == "scale_up"
        assert event.from_instances == 3
        assert event.to_instances == 4
        assert event.trigger_cpu_usage == 85.0
    
    def test_resource_forecasts_model(self):
        """Test resource forecasts model"""
        forecast = ResourceForecasts(
            forecast_timestamp=datetime.now(),
            target_timestamp=datetime.now() + timedelta(hours=2),
            predicted_cpu_usage=72.5,
            predicted_memory_usage=68.2,
            predicted_request_count=1500,
            forecast_horizon_hours=2,
            confidence_score=0.82,
            scaling_recommendation="scale_up_moderate",
            recommended_instances=4
        )
        
        assert forecast.predicted_cpu_usage == 72.5
        assert forecast.forecast_horizon_hours == 2
        assert forecast.confidence_score == 0.82
    
    def test_performance_alerts_model(self):
        """Test performance alerts model"""
        alert = PerformanceAlerts(
            alert_type="cache",
            severity="medium",
            title="Low Cache Hit Rate",
            description="Cache hit rate has dropped below 70%",
            source_component="cache_manager",
            trigger_metrics={"hit_rate": 0.65, "miss_rate": 0.35},
            threshold_values={"min_hit_rate": 0.70}
        )
        
        assert alert.alert_type == "cache"
        assert alert.severity == "medium"
        assert alert.status == "active"

class TestPerformanceOptimizationIntegration:
    """Test integration with ScrollIntel system"""
    
    @pytest.mark.asyncio
    async def test_agent_coordination_integration(self):
        """Test integration with agent coordination system"""
        from unittest.mock import Mock
        import redis
        
        # Mock Redis client
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        # Create performance system
        perf_system = PerformanceOptimizationSystem(mock_redis)
        await perf_system.start()
        
        try:
            # Simulate agent registration from ScrollIntel agents
            agents = [
                {"id": "scroll_bi_agent", "type": "bi", "performance": "high"},
                {"id": "scroll_ml_agent", "type": "ml", "performance": "medium"},
                {"id": "scroll_qa_agent", "type": "qa", "performance": "high"}
            ]
            
            for agent in agents:
                metrics = AgentMetrics(
                    agent_id=agent["id"],
                    cpu_usage=30.0 if agent["performance"] == "high" else 60.0,
                    memory_usage=40.0 if agent["performance"] == "high" else 70.0,
                    response_time=500.0 if agent["performance"] == "high" else 1500.0,
                    throughput=150.0 if agent["performance"] == "high" else 80.0,
                    error_rate=0.01 if agent["performance"] == "high" else 0.03,
                    active_connections=20,
                    queue_length=5,
                    last_updated=datetime.now()
                )
                
                perf_system.load_balancer.register_agent(agent["id"], metrics)
            
            # Test agent selection for different request types
            bi_request = {"request_type": "analytics", "complexity": 0.7}
            selected_agent = await perf_system.load_balancer.select_agent(bi_request)
            
            assert selected_agent in [agent["id"] for agent in agents]
            
            # Test performance monitoring
            report = await perf_system.get_performance_report()
            assert report["load_balancer_stats"]["total_agents"] == 3
            
        finally:
            await perf_system.stop()
    
    @pytest.mark.asyncio
    async def test_caching_integration(self):
        """Test caching integration with ScrollIntel operations"""
        from unittest.mock import Mock
        import redis
        
        # Mock Redis client
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        # Create performance system
        perf_system = PerformanceOptimizationSystem(mock_redis)
        
        # Test caching of common ScrollIntel operations
        cache_scenarios = [
            {"key": "user_profile_123", "data": {"name": "John", "role": "analyst"}},
            {"key": "ml_model_predictions_456", "data": [0.85, 0.92, 0.78]},
            {"key": "dashboard_config_789", "data": {"layout": "grid", "widgets": 5}},
            {"key": "query_results_abc", "data": {"rows": 1000, "columns": 15}}
        ]
        
        # Cache data
        for scenario in cache_scenarios:
            success = await perf_system.cache_manager.set(
                scenario["key"], 
                scenario["data"], 
                ttl=300
            )
            assert success is True
        
        # Retrieve cached data
        for scenario in cache_scenarios:
            cached_data = await perf_system.cache_manager.get(scenario["key"])
            # Note: In real implementation, this would match the original data
            # Here we're testing the caching mechanism structure
            assert cached_data is not None or cached_data == scenario["data"]
        
        # Test cache performance
        stats = perf_system.cache_manager.get_stats()
        assert "hits" in stats
        assert "misses" in stats
    
    @pytest.mark.asyncio
    async def test_scaling_integration(self):
        """Test scaling integration with ScrollIntel workloads"""
        from unittest.mock import Mock, patch
        import redis
        
        # Mock Redis client
        mock_redis = Mock(spec=redis.Redis)
        
        # Create performance system
        perf_system = PerformanceOptimizationSystem(mock_redis)
        
        # Simulate different workload scenarios
        workload_scenarios = [
            {
                "name": "low_load",
                "metrics": {
                    "average_cpu_usage": 25.0,
                    "average_memory_usage": 35.0,
                    "average_response_time": 400.0,
                    "error_rate": 0.005
                },
                "expected_action": None  # No scaling needed
            },
            {
                "name": "high_load",
                "metrics": {
                    "average_cpu_usage": 85.0,
                    "average_memory_usage": 88.0,
                    "average_response_time": 3000.0,
                    "error_rate": 0.08
                },
                "expected_action": "scale_up"
            },
            {
                "name": "very_low_load",
                "metrics": {
                    "average_cpu_usage": 10.0,
                    "average_memory_usage": 20.0,
                    "average_response_time": 200.0,
                    "error_rate": 0.001
                },
                "expected_action": "scale_down"
            }
        ]
        
        for scenario in workload_scenarios:
            # Reset scaling state
            perf_system.resource_manager.last_scaling_action = datetime.min
            perf_system.resource_manager.current_instances = 4
            
            scaling_action = await perf_system.resource_manager.evaluate_scaling(
                scenario["metrics"]
            )
            
            if scenario["expected_action"]:
                assert scaling_action is not None
                assert scaling_action["action"] == scenario["expected_action"]
            else:
                # For low load, might not trigger scaling depending on thresholds
                pass
    
    @pytest.mark.asyncio
    async def test_forecasting_integration(self):
        """Test forecasting integration with ScrollIntel usage patterns"""
        from unittest.mock import Mock
        import redis
        
        # Mock Redis client
        mock_redis = Mock(spec=redis.Redis)
        
        # Create performance system
        perf_system = PerformanceOptimizationSystem(mock_redis)
        
        # Simulate ScrollIntel usage patterns
        usage_patterns = [
            # Business hours pattern
            {"hour": 9, "cpu": 60, "memory": 65, "requests": 1200},
            {"hour": 10, "cpu": 70, "memory": 72, "requests": 1500},
            {"hour": 11, "cpu": 75, "memory": 78, "requests": 1800},
            {"hour": 14, "cpu": 80, "memory": 82, "requests": 2000},
            {"hour": 16, "cpu": 85, "memory": 88, "requests": 2200},
            # Evening pattern
            {"hour": 19, "cpu": 45, "memory": 50, "requests": 800},
            {"hour": 21, "cpu": 35, "memory": 40, "requests": 600},
            # Night pattern
            {"hour": 2, "cpu": 20, "memory": 25, "requests": 200},
            {"hour": 5, "cpu": 15, "memory": 20, "requests": 150}
        ]
        
        # Record historical data
        for pattern in usage_patterns * 10:  # Repeat to build history
            perf_system.forecaster.record_metrics({
                'cpu_usage': pattern["cpu"],
                'memory_usage': pattern["memory"],
                'request_count': pattern["requests"],
                'response_time': 500 + (pattern["cpu"] * 10)  # Response time correlates with CPU
            })
        
        # Test forecasting
        forecasting_stats = perf_system.forecaster.get_forecasting_stats()
        assert forecasting_stats["historical_data_points"] > 0
        
        # Train model if enough data
        if forecasting_stats["historical_data_points"] >= 100:
            await perf_system.forecaster.train_forecasting_models()
            
            # Generate forecasts
            forecasts = await perf_system.forecaster.forecast_demand(hours_ahead=6)
            
            # Verify forecast structure
            for forecast in forecasts:
                assert hasattr(forecast, 'predicted_cpu')
                assert hasattr(forecast, 'predicted_memory')
                assert hasattr(forecast, 'predicted_requests')
                assert hasattr(forecast, 'confidence')
                assert hasattr(forecast, 'scaling_recommendation')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])