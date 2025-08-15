"""
Tests for the intelligent performance optimization engine.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from scrollintel.core.intelligent_performance_optimizer import (
    IntelligentPerformanceOptimizer,
    DeviceCapabilityDetector,
    LoadPredictor,
    IntelligentCacheManager,
    ProgressiveEnhancementManager,
    DeviceCapability,
    OptimizationStrategy,
    DeviceProfile,
    LoadPrediction,
    CacheItem,
    ProgressiveEnhancement,
    get_intelligent_optimizer,
    optimize_performance_for_request,
    predict_system_load
)
from scrollintel.core.performance_optimizer import PerformanceMetrics


class TestDeviceCapabilityDetector:
    """Test device capability detection."""
    
    @pytest.fixture
    def detector(self):
        return DeviceCapabilityDetector()
    
    @pytest.mark.asyncio
    async def test_detect_mobile_device(self, detector):
        """Test mobile device detection."""
        user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        
        profile = await detector.detect_device_capabilities(user_agent)
        
        assert profile.device_type == DeviceCapability.MOBILE
        assert profile.is_mobile is True
        assert profile.cpu_cores == 4
        assert profile.memory_gb == 4.0
        assert 0.0 <= profile.performance_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_detect_desktop_device(self, detector):
        """Test desktop device detection."""
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        
        profile = await detector.detect_device_capabilities(user_agent)
        
        assert profile.device_type == DeviceCapability.DESKTOP
        assert profile.is_mobile is False
        assert profile.cpu_cores == 8
        assert profile.memory_gb == 16.0
        assert 0.0 <= profile.performance_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_detect_with_client_hints(self, detector):
        """Test device detection with client hints."""
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        client_hints = {
            'cpu_cores': 16,
            'memory_gb': 32.0,
            'network_speed': 1000.0,
            'screen_resolution': (3840, 2160),
            'webgl': True,
            'webworkers': True
        }
        
        profile = await detector.detect_device_capabilities(user_agent, client_hints)
        
        assert profile.cpu_cores == 16
        assert profile.memory_gb == 32.0
        assert profile.network_speed == 1000.0
        assert profile.screen_resolution == (3840, 2160)
        assert profile.supports_webgl is True
        assert profile.supports_webworkers is True
    
    def test_performance_score_calculation(self, detector):
        """Test performance score calculation."""
        # High-end device
        score = detector._calculate_performance_score(16, 32.0, 1000.0, DeviceCapability.DESKTOP)
        assert score > 0.8
        
        # Low-end device
        score = detector._calculate_performance_score(2, 2.0, 10.0, DeviceCapability.MOBILE)
        assert score < 0.5


class TestLoadPredictor:
    """Test load prediction functionality."""
    
    @pytest.fixture
    def predictor(self):
        return LoadPredictor()
    
    @pytest.mark.asyncio
    async def test_predict_load_insufficient_data(self, predictor):
        """Test load prediction with insufficient historical data."""
        prediction = await predictor.predict_load()
        
        assert isinstance(prediction, LoadPrediction)
        assert prediction.confidence_score == 0.3
        assert 'insufficient_data' in prediction.factors
    
    @pytest.mark.asyncio
    async def test_predict_load_with_data(self, predictor):
        """Test load prediction with historical data."""
        # Add some historical metrics
        for i in range(10):
            metrics = PerformanceMetrics(
                operation_type="test",
                execution_time=1.0,
                memory_usage=0.5 + i * 0.01,
                cpu_usage=0.4 + i * 0.02,
                throughput=1.0,
                error_rate=0.0,
                timestamp=datetime.utcnow().timestamp()
            )
            predictor.record_metrics(metrics)
        
        prediction = await predictor.predict_load(timedelta(minutes=15))
        
        assert isinstance(prediction, LoadPrediction)
        assert prediction.confidence_score > 0.3
        assert 0.0 <= prediction.predicted_cpu_usage <= 100.0
        assert 0.0 <= prediction.predicted_memory_usage <= 100.0
        assert prediction.predicted_concurrent_users >= 0
    
    def test_predict_metric(self, predictor):
        """Test individual metric prediction."""
        # Create mock recent metrics
        recent_metrics = []
        for i in range(5):
            metric = Mock()
            metric.cpu_usage = 50.0 + i * 2.0
            recent_metrics.append(metric)
        
        predicted_value = predictor._predict_metric(
            'cpu_usage', recent_metrics, timedelta(minutes=30)
        )
        
        assert 0.0 <= predicted_value <= 100.0
    
    def test_calculate_prediction_confidence(self, predictor):
        """Test prediction confidence calculation."""
        # Low variance metrics (high confidence)
        stable_metrics = []
        for i in range(10):
            metric = Mock()
            metric.cpu_usage = 50.0  # Constant value
            stable_metrics.append(metric)
        
        confidence = predictor._calculate_prediction_confidence(stable_metrics)
        assert confidence > 0.8
        
        # High variance metrics (low confidence)
        volatile_metrics = []
        for i in range(10):
            metric = Mock()
            metric.cpu_usage = 50.0 + (i % 2) * 40.0  # Alternating values
            volatile_metrics.append(metric)
        
        confidence = predictor._calculate_prediction_confidence(volatile_metrics)
        assert confidence < 0.8


class TestIntelligentCacheManager:
    """Test intelligent caching functionality."""
    
    @pytest.fixture
    def cache_manager(self):
        return IntelligentCacheManager(max_cache_size=100)
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache_manager):
        """Test basic cache set and get operations."""
        await cache_manager.set("test_key", "test_value", user_id="user1")
        
        value = await cache_manager.get("test_key", user_id="user1")
        assert value == "test_value"
        
        # Check cache stats
        stats = cache_manager.get_cache_stats()
        assert stats['hits'] == 1
        assert stats['total_items'] == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_manager):
        """Test cache miss handling."""
        value = await cache_manager.get("nonexistent_key")
        assert value is None
        
        stats = cache_manager.get_cache_stats()
        assert stats['misses'] == 1
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, cache_manager):
        """Test cache item expiry."""
        # Set item with short TTL
        await cache_manager.set(
            "expiring_key", 
            "expiring_value", 
            ttl=timedelta(milliseconds=1)
        )
        
        # Wait for expiry
        await asyncio.sleep(0.01)
        
        value = await cache_manager.get("expiring_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache_manager):
        """Test intelligent cache eviction."""
        # Fill cache beyond capacity
        for i in range(150):  # More than max_cache_size
            await cache_manager.set(f"key_{i}", f"value_{i}")
        
        # Check that cache size is within limits
        assert len(cache_manager.cache) <= cache_manager.max_cache_size
        
        # Check eviction stats
        stats = cache_manager.get_cache_stats()
        assert stats['evictions'] > 0
    
    def test_predict_next_access(self, cache_manager):
        """Test next access prediction."""
        key = "test_key"
        
        # Add access pattern
        now = datetime.utcnow()
        cache_manager.access_patterns[key] = [
            now - timedelta(minutes=10),
            now - timedelta(minutes=5),
            now
        ]
        
        predicted_time = cache_manager._predict_next_access(key)
        assert predicted_time is not None
        assert predicted_time > now
    
    @pytest.mark.asyncio
    async def test_user_pattern_tracking(self, cache_manager):
        """Test user access pattern tracking."""
        user_id = "test_user"
        
        # Simulate user accessing multiple keys
        await cache_manager.get("key1", user_id=user_id)
        await cache_manager.get("key2", user_id=user_id)
        await cache_manager.get("key3", user_id=user_id)
        
        # Check that user patterns are recorded
        assert user_id in cache_manager.user_patterns
        assert 'access_sequence' in cache_manager.user_patterns[user_id]
        assert 'key_preferences' in cache_manager.user_patterns[user_id]


class TestProgressiveEnhancementManager:
    """Test progressive enhancement functionality."""
    
    @pytest.fixture
    def enhancement_manager(self):
        return ProgressiveEnhancementManager()
    
    @pytest.mark.asyncio
    async def test_adapt_features_high_end_device(self, enhancement_manager):
        """Test feature adaptation for high-end device."""
        device_profile = DeviceProfile(
            device_id="high_end_device",
            device_type=DeviceCapability.HIGH_END,
            cpu_cores=16,
            memory_gb=32.0,
            network_speed=1000.0,
            screen_resolution=(3840, 2160),
            supports_webgl=True,
            supports_webworkers=True,
            performance_score=0.9
        )
        
        features = await enhancement_manager.adapt_features_for_device(
            'dashboard', device_profile
        )
        
        assert 'basic_charts' in features  # Base features
        assert 'interactive_charts' in features  # Enhanced features
        assert 'webgl_visualizations' in features  # High-end features
    
    @pytest.mark.asyncio
    async def test_adapt_features_mobile_device(self, enhancement_manager):
        """Test feature adaptation for mobile device."""
        device_profile = DeviceProfile(
            device_id="mobile_device",
            device_type=DeviceCapability.MOBILE,
            cpu_cores=4,
            memory_gb=4.0,
            network_speed=50.0,
            screen_resolution=(1080, 1920),
            supports_webgl=True,
            supports_webworkers=True,
            is_mobile=True,
            performance_score=0.6
        )
        
        features = await enhancement_manager.adapt_features_for_device(
            'dashboard', device_profile
        )
        
        assert 'basic_charts' in features  # Base features
        assert 'mobile_optimized_charts' in features  # Mobile features
        assert 'webgl_visualizations' not in features  # No high-end features
    
    @pytest.mark.asyncio
    async def test_performance_based_filtering(self, enhancement_manager):
        """Test feature filtering based on performance metrics."""
        device_profile = DeviceProfile(
            device_id="test_device",
            device_type=DeviceCapability.HIGH_END,
            cpu_cores=16,
            memory_gb=32.0,
            network_speed=1000.0,
            screen_resolution=(3840, 2160),
            supports_webgl=True,
            supports_webworkers=True,
            performance_score=0.9
        )
        
        # Simulate poor performance
        poor_performance = {
            'cpu_usage': 0.95,  # 95% CPU usage
            'memory_usage': 0.90,  # 90% memory usage
            'network_speed': 0.5   # Slow network
        }
        
        features = await enhancement_manager.adapt_features_for_device(
            'dashboard', device_profile, poor_performance
        )
        
        # Resource-intensive features should be filtered out
        assert 'webgl_visualizations' not in features
        assert 'real_time_updates' not in features
    
    @pytest.mark.asyncio
    async def test_record_feature_performance(self, enhancement_manager):
        """Test recording feature performance for learning."""
        await enhancement_manager.record_feature_performance(
            'dashboard', 'webgl_visualizations', DeviceCapability.HIGH_END, 0.3
        )
        
        # Check that poorly performing feature is removed
        config = enhancement_manager.enhancement_configs['dashboard']
        assert 'webgl_visualizations' not in config.enhanced_features[DeviceCapability.HIGH_END]


class TestIntelligentPerformanceOptimizer:
    """Test the main intelligent performance optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        return IntelligentPerformanceOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimize_for_device(self, optimizer):
        """Test device-specific optimization."""
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        client_hints = {'cpu_cores': 8, 'memory_gb': 16.0}
        
        config = await optimizer.optimize_for_device(user_agent, client_hints, "user1")
        
        assert 'device_profile' in config
        assert 'enabled_features' in config
        assert 'performance_settings' in config
        assert 'resource_limits' in config
        
        # Check device profile
        device_profile = config['device_profile']
        assert device_profile['device_type'] == 'desktop'
        assert device_profile['performance_score'] > 0.0
        
        # Check enabled features
        enabled_features = config['enabled_features']
        assert 'dashboard' in enabled_features
        assert 'data_processing' in enabled_features
        assert 'visualization' in enabled_features
    
    @pytest.mark.asyncio
    async def test_predict_and_allocate_resources(self, optimizer):
        """Test load prediction and resource allocation."""
        result = await optimizer.predict_and_allocate_resources(timedelta(minutes=15))
        
        assert 'prediction' in result
        assert 'allocation' in result
        assert 'optimization_strategy' in result
        
        # Check prediction
        prediction = result['prediction']
        assert 'predicted_cpu_usage' in prediction
        assert 'predicted_memory_usage' in prediction
        assert 'predicted_users' in prediction
        assert 'confidence' in prediction
        
        # Check allocation
        allocation = result['allocation']
        assert 'cpu_allocation' in allocation
        assert 'memory_allocation' in allocation
        assert 'network_allocation' in allocation
    
    @pytest.mark.asyncio
    async def test_optimize_caching_strategy(self, optimizer):
        """Test caching strategy optimization."""
        optimization = await optimizer.optimize_caching_strategy("user1", {"context": "test"})
        
        assert 'current_performance' in optimization
        assert 'recommendations' in optimization
        assert 'preload_candidates' in optimization
        assert 'eviction_candidates' in optimization
        
        # Check performance metrics
        performance = optimization['current_performance']
        assert 'hit_rate' in performance
        assert 'total_items' in performance
    
    def test_calculate_optimal_settings(self, optimizer):
        """Test calculation of optimal settings."""
        device_profile = DeviceProfile(
            device_id="test_device",
            device_type=DeviceCapability.HIGH_END,
            cpu_cores=16,
            memory_gb=32.0,
            network_speed=1000.0,
            screen_resolution=(3840, 2160),
            supports_webgl=True,
            supports_webworkers=True,
            performance_score=0.9
        )
        
        # Test chunk size calculation
        chunk_size = optimizer._calculate_optimal_chunk_size(device_profile)
        assert chunk_size == 10000  # High-end device should get large chunks
        
        # Test cache size calculation
        cache_size = optimizer._calculate_optimal_cache_size(device_profile)
        assert cache_size > 1000  # Should be substantial for high-end device
        
        # Test update frequency calculation
        update_freq = optimizer._calculate_optimal_update_frequency(device_profile)
        assert update_freq == 1.0  # High-end device should get frequent updates
        
        # Test quality settings calculation
        quality = optimizer._calculate_quality_settings(device_profile)
        assert quality['chart_quality'] == 'high'
        assert quality['enable_effects'] is True
    
    def test_resource_allocation_calculations(self, optimizer):
        """Test resource allocation calculations."""
        prediction = LoadPrediction(
            timestamp=datetime.utcnow(),
            predicted_cpu_usage=70.0,
            predicted_memory_usage=60.0,
            predicted_network_usage=50.0,
            predicted_concurrent_users=100,
            confidence_score=0.8,
            prediction_horizon=timedelta(minutes=30)
        )
        
        # Test CPU allocation
        cpu_alloc = optimizer._calculate_cpu_allocation('dashboard', prediction)
        assert 0.0 <= cpu_alloc <= 0.8
        
        # Test memory allocation
        memory_alloc = optimizer._calculate_memory_allocation('cache', prediction)
        assert 0.0 <= memory_alloc <= 0.8
        
        # Test network allocation
        network_alloc = optimizer._calculate_network_allocation('real_time_updates', prediction)
        assert 0.0 <= network_alloc <= 0.9
    
    def test_get_optimization_stats(self, optimizer):
        """Test optimization statistics retrieval."""
        stats = optimizer.get_optimization_stats()
        
        assert 'current_strategy' in stats
        assert 'cache_performance' in stats
        assert 'device_adaptations' in stats
        assert 'resource_allocations' in stats
        assert 'optimization_history' in stats
        assert 'performance_targets' in stats
        
        # Check that strategy is valid
        assert stats['current_strategy'] in ['aggressive', 'balanced', 'conservative', 'adaptive']


class TestAPIEndpoints:
    """Test API endpoint functions."""
    
    @pytest.mark.asyncio
    async def test_optimize_performance_for_request(self):
        """Test performance optimization API endpoint."""
        result = await optimize_performance_for_request(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            client_hints={'cpu_cores': 8},
            user_id="test_user"
        )
        
        assert result['optimized'] is True
        assert 'device_optimization' in result
        assert 'cache_optimization' in result
        assert 'timestamp' in result
    
    @pytest.mark.asyncio
    async def test_predict_system_load(self):
        """Test system load prediction API endpoint."""
        result = await predict_system_load(horizon_minutes=30)
        
        assert result['predicted'] is True
        assert 'result' in result
        assert 'timestamp' in result
        
        # Check result structure
        prediction_result = result['result']
        assert 'prediction' in prediction_result
        assert 'allocation' in prediction_result
        assert 'optimization_strategy' in prediction_result


class TestGlobalOptimizer:
    """Test global optimizer instance management."""
    
    def test_get_intelligent_optimizer_singleton(self):
        """Test that get_intelligent_optimizer returns singleton instance."""
        optimizer1 = get_intelligent_optimizer()
        optimizer2 = get_intelligent_optimizer()
        
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, IntelligentPerformanceOptimizer)


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete optimization scenarios."""
    
    @pytest.mark.asyncio
    async def test_mobile_user_optimization_flow(self):
        """Test complete optimization flow for mobile user."""
        optimizer = IntelligentPerformanceOptimizer()
        
        # Simulate mobile user
        user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        user_id = "mobile_user_123"
        
        # Step 1: Optimize for device
        device_config = await optimizer.optimize_for_device(user_agent, {}, user_id)
        
        assert device_config['device_profile']['device_type'] == 'mobile'
        assert device_config['device_profile']['is_mobile'] is True
        
        # Step 2: Predict load and allocate resources
        resource_result = await optimizer.predict_and_allocate_resources()
        
        assert 'prediction' in resource_result
        assert 'allocation' in resource_result
        
        # Step 3: Optimize caching
        cache_result = await optimizer.optimize_caching_strategy(user_id)
        
        assert 'current_performance' in cache_result
        assert 'recommendations' in cache_result
        
        # Step 4: Get optimization stats
        stats = optimizer.get_optimization_stats()
        
        assert stats['current_strategy'] in ['aggressive', 'balanced', 'conservative', 'adaptive']
        assert 'cache_performance' in stats
    
    @pytest.mark.asyncio
    async def test_high_load_adaptation_scenario(self):
        """Test system adaptation under high load conditions."""
        optimizer = IntelligentPerformanceOptimizer()
        
        # Simulate high load metrics
        high_load_metrics = []
        for i in range(20):
            metrics = PerformanceMetrics(
                operation_type="high_load_test",
                execution_time=2.0 + i * 0.1,
                memory_usage=0.8 + i * 0.01,  # Increasing memory usage
                cpu_usage=0.7 + i * 0.015,   # Increasing CPU usage
                throughput=1.0 / (1.0 + i * 0.1),  # Decreasing throughput
                error_rate=i * 0.005,  # Increasing error rate
                timestamp=datetime.utcnow().timestamp()
            )
            optimizer.load_predictor.record_metrics(metrics)
            high_load_metrics.append(metrics)
        
        # Predict load under high load conditions
        prediction = await optimizer.load_predictor.predict_load()
        
        # Should predict high resource usage (values are in 0-1 range, not 0-100)
        assert prediction.predicted_cpu_usage > 0.7
        assert prediction.predicted_memory_usage > 0.8
        
        # System should adapt optimization strategy
        current_performance = {
            'cpu_usage': 0.9,
            'memory_usage': 0.85,
            'network_speed': 50.0,
            'response_time': 3.0
        }
        
        await optimizer._adjust_optimization_strategy(current_performance)
        
        # Should switch to aggressive optimization
        assert optimizer.current_strategy == OptimizationStrategy.AGGRESSIVE
    
    @pytest.mark.asyncio
    async def test_cache_learning_scenario(self):
        """Test cache learning and adaptation scenario."""
        cache_manager = IntelligentCacheManager(max_cache_size=50)
        user_id = "learning_user"
        
        # Simulate user access patterns
        access_sequence = [
            "dashboard_data", "user_profile", "recent_files",
            "dashboard_data", "analytics_report", "user_profile",
            "dashboard_data", "recent_files", "analytics_report"
        ]
        
        # Simulate multiple access cycles
        for cycle in range(3):
            for key in access_sequence:
                # Set data if not in cache
                cached_value = await cache_manager.get(key, user_id)
                if cached_value is None:
                    await cache_manager.set(key, f"data_for_{key}_{cycle}", user_id)
                else:
                    # Just access the cached data
                    await cache_manager.get(key, user_id)
        
        # Check that user patterns are learned
        assert user_id in cache_manager.user_patterns
        user_pattern = cache_manager.user_patterns[user_id]
        
        assert 'access_sequence' in user_pattern
        assert 'key_preferences' in user_pattern
        
        # Most accessed keys should have higher preference scores
        preferences = user_pattern['key_preferences']
        assert preferences['dashboard_data'] >= preferences['analytics_report']
        
        # Cache should have good hit rate due to learned patterns
        stats = cache_manager.get_cache_stats()
        assert stats['hit_rate'] > 0.5


if __name__ == "__main__":
    pytest.main([__file__])