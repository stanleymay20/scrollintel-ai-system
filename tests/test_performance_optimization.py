"""
Tests for Performance Optimization System

Comprehensive tests for intelligent caching, ML load balancing,
auto-scaling, and predictive forecasting components.

Requirements: 4.1, 6.1
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import redis

from scrollintel.core.performance_optimization import (
    PerformanceOptimizationSystem,
    IntelligentCacheManager,
    MLLoadBalancer,
    AutoScalingResourceManager,
    PredictiveResourceForecaster,
    AgentMetrics,
    CacheEntry,
    ResourceDemand,
    CacheStrategy,
    LoadBalancingStrategy
)

class TestIntelligentCacheManager:
    """Test intelligent cache management"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.setex.return_value = True
        return mock_redis
    
    @pytest.fixture
    def cache_manager(self, mock_redis):
        """Create cache manager instance"""
        return IntelligentCacheManager(mock_redis, max_size_mb=10)
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache_manager):
        """Test basic cache set and get operations"""
        # Test setting a value
        success = await cache_manager.set("test_key", {"data": "test_value"}, ttl=300)
        assert success is True
        
        # Test getting the value
        value = await cache_manager.get("test_key")
        assert value == {"data": "test_value"}
        
        # Test cache hit statistics
        stats = cache_manager.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_manager):
        """Test cache miss behavior"""
        value = await cache_manager.get("nonexistent_key")
        assert value is None
        
        stats = cache_manager.get_stats()
        assert stats['misses'] == 1
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, cache_manager):
        """Test TTL-based cache expiration"""
        # Set value with short TTL
        await cache_manager.set("ttl_key", "ttl_value", ttl=1)
        
        # Should be available immediately
        value = await cache_manager.get("ttl_key")
        assert value == "ttl_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        value = await cache_manager.get("ttl_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache_manager):
        """Test cache eviction when capacity is exceeded"""
        # Fill cache to capacity
        large_value = "x" * 1024 * 1024  # 1MB
        
        for i in range(12):  # Exceed 10MB limit
            await cache_manager.set(f"key_{i}", large_value)
        
        # Check that eviction occurred
        stats = cache_manager.get_stats()
        assert stats['evictions'] > 0
        assert stats['size_mb'] <= 10
    
    def test_eviction_strategies(self, cache_manager):
        """Test different cache eviction strategies"""
        # Test LRU strategy
        cache_manager.strategy = CacheStrategy.LRU
        
        # Add entries with different access times
        now = datetime.now()
        cache_manager.local_cache = {
            "old": CacheEntry("old", "value", now - timedelta(hours=2), now - timedelta(hours=1)),
            "new": CacheEntry("new", "value", now - timedelta(minutes=30), now - timedelta(minutes=10))
        }
        
        evict_key = cache_manager._select_eviction_candidate()
        assert evict_key == "old"  # Should evict least recently used
        
        # Test LFU strategy
        cache_manager.strategy = CacheStrategy.LFU
        cache_manager.local_cache["old"].access_count = 1
        cache_manager.local_cache["new"].access_count = 10
        
        evict_key = cache_manager._select_eviction_candidate()
        assert evict_key == "old"  # Should evict least frequently used

class TestMLLoadBalancer:
    """Test ML-based load balancer"""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer instance"""
        return MLLoadBalancer()
    
    @pytest.fixture
    def sample_agent_metrics(self):
        """Create sample agent metrics"""
        return AgentMetrics(
            agent_id="agent_1",
            cpu_usage=50.0,
            memory_usage=60.0,
            response_time=1000.0,
            throughput=100.0,
            error_rate=0.01,
            active_connections=50,
            queue_length=10,
            last_updated=datetime.now()
        )
    
    def test_agent_registration(self, load_balancer, sample_agent_metrics):
        """Test agent registration"""
        load_balancer.register_agent("agent_1", sample_agent_metrics)
        
        assert "agent_1" in load_balancer.agents
        assert load_balancer.agents["agent_1"].agent_id == "agent_1"
    
    def test_agent_metrics_update(self, load_balancer, sample_agent_metrics):
        """Test agent metrics updates"""
        load_balancer.register_agent("agent_1", sample_agent_metrics)
        
        # Update metrics
        updated_metrics = sample_agent_metrics
        updated_metrics.cpu_usage = 70.0
        load_balancer.update_agent_metrics("agent_1", updated_metrics)
        
        assert load_balancer.agents["agent_1"].cpu_usage == 70.0
        assert len(load_balancer.request_history) > 0
    
    def test_health_score_calculation(self, sample_agent_metrics):
        """Test agent health score calculation"""
        health_score = sample_agent_metrics.calculate_health_score()
        
        assert 0 <= health_score <= 1
        assert health_score > 0  # Should be positive for reasonable metrics
    
    @pytest.mark.asyncio
    async def test_agent_selection_heuristic(self, load_balancer):
        """Test heuristic-based agent selection"""
        # Register multiple agents with different performance
        good_agent = AgentMetrics(
            agent_id="good_agent",
            cpu_usage=30.0,
            memory_usage=40.0,
            response_time=500.0,
            throughput=200.0,
            error_rate=0.001,
            active_connections=20,
            queue_length=5,
            last_updated=datetime.now()
        )
        good_agent.calculate_health_score()
        
        bad_agent = AgentMetrics(
            agent_id="bad_agent",
            cpu_usage=90.0,
            memory_usage=95.0,
            response_time=5000.0,
            throughput=10.0,
            error_rate=0.1,
            active_connections=100,
            queue_length=50,
            last_updated=datetime.now()
        )
        bad_agent.calculate_health_score()
        
        load_balancer.register_agent("good_agent", good_agent)
        load_balancer.register_agent("bad_agent", bad_agent)
        
        # Select agent
        selected = await load_balancer.select_agent({"request_complexity": 1.0})
        
        # Should select the better performing agent
        assert selected == "good_agent"
    
    @pytest.mark.asyncio
    async def test_no_healthy_agents(self, load_balancer):
        """Test behavior when no healthy agents are available"""
        # Register unhealthy agent
        unhealthy_agent = AgentMetrics(
            agent_id="unhealthy_agent",
            cpu_usage=99.0,
            memory_usage=99.0,
            response_time=10000.0,
            throughput=1.0,
            error_rate=0.5,
            active_connections=1000,
            queue_length=500,
            last_updated=datetime.now()
        )
        unhealthy_agent.calculate_health_score()
        
        load_balancer.register_agent("unhealthy_agent", unhealthy_agent)
        
        selected = await load_balancer.select_agent({})
        assert selected is None  # No healthy agents available
    
    @pytest.mark.asyncio
    async def test_ml_model_training(self, load_balancer, sample_agent_metrics):
        """Test ML model training"""
        # Add training data
        for i in range(150):  # Need sufficient data for training
            metrics = sample_agent_metrics
            metrics.agent_id = f"agent_{i % 3}"
            metrics.cpu_usage = 30 + (i % 50)
            metrics.response_time = 500 + (i % 1000)
            load_balancer.update_agent_metrics(metrics.agent_id, metrics)
        
        # Train model
        await load_balancer.train_model()
        
        assert load_balancer.is_trained is True
    
    def test_load_balancer_stats(self, load_balancer, sample_agent_metrics):
        """Test load balancer statistics"""
        load_balancer.register_agent("agent_1", sample_agent_metrics)
        
        stats = load_balancer.get_agent_stats()
        
        assert stats['total_agents'] == 1
        assert 'healthy_agents' in stats
        assert 'average_health' in stats

class TestAutoScalingResourceManager:
    """Test auto-scaling resource manager"""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager instance"""
        return AutoScalingResourceManager(min_instances=2, max_instances=10)
    
    @pytest.mark.asyncio
    async def test_scale_up_decision(self, resource_manager):
        """Test scale up decision logic"""
        # Metrics indicating need to scale up
        metrics = {
            'average_cpu_usage': 80.0,
            'average_memory_usage': 85.0,
            'average_response_time': 3000.0,
            'error_rate': 0.06
        }
        
        scaling_action = await resource_manager.evaluate_scaling(metrics)
        
        assert scaling_action is not None
        assert scaling_action['action'] == 'scale_up'
        assert scaling_action['to_instances'] > scaling_action['from_instances']
    
    @pytest.mark.asyncio
    async def test_scale_down_decision(self, resource_manager):
        """Test scale down decision logic"""
        # Start with more instances
        resource_manager.current_instances = 5
        
        # Metrics indicating can scale down
        metrics = {
            'average_cpu_usage': 20.0,
            'average_memory_usage': 30.0,
            'average_response_time': 300.0,
            'error_rate': 0.005
        }
        
        scaling_action = await resource_manager.evaluate_scaling(metrics)
        
        assert scaling_action is not None
        assert scaling_action['action'] == 'scale_down'
        assert scaling_action['to_instances'] < scaling_action['from_instances']
    
    @pytest.mark.asyncio
    async def test_no_scaling_needed(self, resource_manager):
        """Test when no scaling is needed"""
        # Balanced metrics
        metrics = {
            'average_cpu_usage': 50.0,
            'average_memory_usage': 55.0,
            'average_response_time': 1000.0,
            'error_rate': 0.02
        }
        
        scaling_action = await resource_manager.evaluate_scaling(metrics)
        
        assert scaling_action is None
    
    @pytest.mark.asyncio
    async def test_cooldown_period(self, resource_manager):
        """Test cooldown period enforcement"""
        # First scaling action
        metrics = {
            'average_cpu_usage': 80.0,
            'average_memory_usage': 85.0,
            'average_response_time': 3000.0,
            'error_rate': 0.06
        }
        
        first_action = await resource_manager.evaluate_scaling(metrics)
        assert first_action is not None
        
        # Immediate second attempt should be blocked by cooldown
        second_action = await resource_manager.evaluate_scaling(metrics)
        assert second_action is None
    
    @pytest.mark.asyncio
    async def test_instance_limits(self, resource_manager):
        """Test instance limit enforcement"""
        # Test max limit
        resource_manager.current_instances = resource_manager.max_instances
        
        metrics = {
            'average_cpu_usage': 95.0,
            'average_memory_usage': 95.0,
            'average_response_time': 5000.0,
            'error_rate': 0.1
        }
        
        scaling_action = await resource_manager.evaluate_scaling(metrics)
        assert scaling_action is None  # Can't scale beyond max
        
        # Test min limit
        resource_manager.current_instances = resource_manager.min_instances
        resource_manager.last_scaling_action = datetime.min  # Reset cooldown
        
        metrics = {
            'average_cpu_usage': 5.0,
            'average_memory_usage': 10.0,
            'average_response_time': 100.0,
            'error_rate': 0.001
        }
        
        scaling_action = await resource_manager.evaluate_scaling(metrics)
        assert scaling_action is None  # Can't scale below min
    
    def test_scaling_stats(self, resource_manager):
        """Test scaling statistics"""
        stats = resource_manager.get_scaling_stats()
        
        assert 'current_instances' in stats
        assert 'min_instances' in stats
        assert 'max_instances' in stats
        assert 'total_scaling_events' in stats

class TestPredictiveResourceForecaster:
    """Test predictive resource forecasting"""
    
    @pytest.fixture
    def forecaster(self):
        """Create forecaster instance"""
        return PredictiveResourceForecaster()
    
    def test_metrics_recording(self, forecaster):
        """Test metrics recording"""
        metrics = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'request_count': 1000,
            'response_time': 500.0
        }
        
        forecaster.record_metrics(metrics)
        
        assert len(forecaster.historical_data) == 1
        assert forecaster.historical_data[0]['cpu_usage'] == 50.0
    
    @pytest.mark.asyncio
    async def test_model_training_insufficient_data(self, forecaster):
        """Test model training with insufficient data"""
        # Add minimal data
        for i in range(50):
            forecaster.record_metrics({
                'cpu_usage': 50 + i,
                'memory_usage': 60 + i,
                'request_count': 1000 + i,
                'response_time': 500 + i
            })
        
        await forecaster.train_forecasting_models()
        
        # Should not be trained with insufficient data
        assert forecaster.is_trained is False
    
    @pytest.mark.asyncio
    async def test_model_training_sufficient_data(self, forecaster):
        """Test model training with sufficient data"""
        # Add sufficient training data
        for i in range(200):
            forecaster.record_metrics({
                'cpu_usage': 30 + (i % 50),
                'memory_usage': 40 + (i % 60),
                'request_count': 1000 + (i % 500),
                'response_time': 500 + (i % 1000)
            })
        
        await forecaster.train_forecasting_models()
        
        assert forecaster.is_trained is True
    
    @pytest.mark.asyncio
    async def test_demand_forecasting(self, forecaster):
        """Test resource demand forecasting"""
        # Add training data and train model
        for i in range(200):
            forecaster.record_metrics({
                'cpu_usage': 30 + (i % 50),
                'memory_usage': 40 + (i % 60),
                'request_count': 1000 + (i % 500),
                'response_time': 500 + (i % 1000)
            })
        
        await forecaster.train_forecasting_models()
        
        # Generate forecasts
        forecasts = await forecaster.forecast_demand(hours_ahead=6)
        
        assert len(forecasts) == 6
        for forecast in forecasts:
            assert isinstance(forecast, ResourceDemand)
            assert forecast.predicted_cpu >= 0
            assert forecast.predicted_memory >= 0
            assert forecast.predicted_requests >= 0
            assert 0 <= forecast.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_forecasting_without_training(self, forecaster):
        """Test forecasting without trained model"""
        forecasts = await forecaster.forecast_demand(hours_ahead=3)
        
        assert len(forecasts) == 0  # No forecasts without training
    
    def test_scaling_recommendations(self, forecaster):
        """Test scaling recommendation generation"""
        # Test scale up recommendation
        recommendation = forecaster._generate_scaling_recommendation(85.0, 90.0, 2000)
        assert "scale_up" in recommendation
        
        # Test scale down recommendation
        recommendation = forecaster._generate_scaling_recommendation(15.0, 25.0, 100)
        assert "scale_down" in recommendation
        
        # Test maintain recommendation
        recommendation = forecaster._generate_scaling_recommendation(50.0, 55.0, 1000)
        assert recommendation == "maintain_current"
    
    def test_forecasting_stats(self, forecaster):
        """Test forecasting statistics"""
        # Add some data
        for i in range(10):
            forecaster.record_metrics({
                'cpu_usage': 50.0,
                'memory_usage': 60.0,
                'request_count': 1000,
                'response_time': 500.0
            })
        
        stats = forecaster.get_forecasting_stats()
        
        assert 'is_trained' in stats
        assert 'historical_data_points' in stats
        assert stats['historical_data_points'] == 10

class TestPerformanceOptimizationSystem:
    """Test complete performance optimization system"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        return mock_redis
    
    @pytest.fixture
    def performance_system(self, mock_redis):
        """Create performance optimization system"""
        return PerformanceOptimizationSystem(mock_redis)
    
    @pytest.mark.asyncio
    async def test_system_startup_shutdown(self, performance_system):
        """Test system startup and shutdown"""
        await performance_system.start()
        assert performance_system._running is True
        
        await performance_system.stop()
        assert performance_system._running is False
    
    @pytest.mark.asyncio
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    async def test_metrics_collection(self, mock_memory, mock_cpu, performance_system):
        """Test system metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(percent=55.0, available=8*1024**3)
        
        metrics = await performance_system._collect_system_metrics()
        
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert metrics['cpu_usage'] == 45.0
        assert metrics['memory_usage'] == 55.0
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self, performance_system):
        """Test performance report generation"""
        report = await performance_system.get_performance_report()
        
        assert 'timestamp' in report
        assert 'cache_stats' in report
        assert 'load_balancer_stats' in report
        assert 'scaling_stats' in report
        assert 'forecasting_stats' in report
        assert 'system_health' in report
    
    @pytest.mark.asyncio
    async def test_system_health_calculation(self, performance_system):
        """Test system health calculation"""
        health = await performance_system._calculate_system_health()
        
        assert 'overall_score' in health
        assert 'cache_health' in health
        assert 'agent_health' in health
        assert 'status' in health
        assert 0 <= health['overall_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_integrated_workflow(self, performance_system):
        """Test integrated performance optimization workflow"""
        # Start system
        await performance_system.start()
        
        # Register agents
        agent_metrics = AgentMetrics(
            agent_id="test_agent",
            cpu_usage=60.0,
            memory_usage=70.0,
            response_time=1200.0,
            throughput=80.0,
            error_rate=0.02,
            active_connections=40,
            queue_length=8,
            last_updated=datetime.now()
        )
        
        performance_system.load_balancer.register_agent("test_agent", agent_metrics)
        
        # Test cache operations
        await performance_system.cache_manager.set("test_key", "test_value")
        cached_value = await performance_system.cache_manager.get("test_key")
        assert cached_value == "test_value"
        
        # Test agent selection
        selected_agent = await performance_system.load_balancer.select_agent({})
        assert selected_agent == "test_agent"
        
        # Record metrics for forecasting
        for i in range(10):
            performance_system.forecaster.record_metrics({
                'cpu_usage': 50 + i,
                'memory_usage': 60 + i,
                'request_count': 1000 + i * 10,
                'response_time': 500 + i * 50
            })
        
        # Generate performance report
        report = await performance_system.get_performance_report()
        assert report['load_balancer_stats']['total_agents'] == 1
        
        # Stop system
        await performance_system.stop()

class TestPerformanceOptimizationIntegration:
    """Integration tests for performance optimization"""
    
    @pytest.mark.asyncio
    async def test_cache_load_balancer_integration(self):
        """Test integration between cache and load balancer"""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        system = PerformanceOptimizationSystem(mock_redis)
        
        # Cache agent selection results
        agent_selection_key = "lb_selection_agent_1"
        await system.cache_manager.set(agent_selection_key, "selected_agent_1")
        
        # Verify cached result
        cached_selection = await system.cache_manager.get(agent_selection_key)
        assert cached_selection == "selected_agent_1"
    
    @pytest.mark.asyncio
    async def test_forecasting_scaling_integration(self):
        """Test integration between forecasting and scaling"""
        mock_redis = Mock(spec=redis.Redis)
        system = PerformanceOptimizationSystem(mock_redis)
        
        # Add historical data showing increasing load
        for i in range(100):
            system.forecaster.record_metrics({
                'cpu_usage': 30 + i * 0.5,  # Gradually increasing
                'memory_usage': 40 + i * 0.4,
                'request_count': 1000 + i * 10,
                'response_time': 500 + i * 5
            })
        
        # Train forecasting model
        await system.forecaster.train_forecasting_models()
        
        # Generate forecast
        forecasts = await system.forecaster.forecast_demand(hours_ahead=2)
        
        if forecasts:
            # Check if forecast indicates need for scaling
            next_forecast = forecasts[0]
            if next_forecast.predicted_cpu > 70:
                # Simulate scaling decision based on forecast
                scaling_metrics = {
                    'average_cpu_usage': next_forecast.predicted_cpu,
                    'average_memory_usage': next_forecast.predicted_memory,
                    'average_response_time': 1000.0,
                    'error_rate': 0.02
                }
                
                scaling_action = await system.resource_manager.evaluate_scaling(scaling_metrics)
                
                if scaling_action:
                    assert scaling_action['action'] in ['scale_up', 'scale_down', 'maintain']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])