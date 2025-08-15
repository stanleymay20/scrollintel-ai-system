"""
Demo script for the Intelligent Performance Optimization Engine.
Showcases adaptive performance optimization, dynamic resource allocation,
intelligent caching, and progressive enhancement capabilities.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.intelligent_performance_optimizer import (
    IntelligentPerformanceOptimizer,
    DeviceCapability,
    OptimizationStrategy,
    get_intelligent_optimizer,
    optimize_performance_for_request,
    predict_system_load
)
from scrollintel.core.performance_optimizer import PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_device_capability_detection():
    """Demonstrate device capability detection and adaptation."""
    print("\n" + "="*80)
    print("DEVICE CAPABILITY DETECTION AND ADAPTATION DEMO")
    print("="*80)
    
    optimizer = get_intelligent_optimizer()
    
    # Test different device types
    test_devices = [
        {
            'name': 'High-End Desktop',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'client_hints': {
                'cpu_cores': 16,
                'memory_gb': 32.0,
                'network_speed': 1000.0,
                'screen_resolution': (3840, 2160),
                'webgl': True,
                'webworkers': True
            }
        },
        {
            'name': 'Mobile Device',
            'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
            'client_hints': {
                'cpu_cores': 6,
                'memory_gb': 4.0,
                'network_speed': 50.0,
                'screen_resolution': (1080, 1920),
                'webgl': True,
                'webworkers': True
            }
        },
        {
            'name': 'Low-End Laptop',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'client_hints': {
                'cpu_cores': 2,
                'memory_gb': 4.0,
                'network_speed': 25.0,
                'screen_resolution': (1366, 768),
                'webgl': False,
                'webworkers': True
            }
        }
    ]
    
    for device in test_devices:
        print(f"\n📱 Testing {device['name']}:")
        print("-" * 50)
        
        # Optimize for device
        config = await optimizer.optimize_for_device(
            device['user_agent'],
            device['client_hints'],
            f"user_{device['name'].lower().replace(' ', '_')}"
        )
        
        device_profile = config['device_profile']
        enabled_features = config['enabled_features']
        performance_settings = config['performance_settings']
        
        print(f"Device Type: {device_profile['device_type']}")
        print(f"Performance Score: {device_profile['performance_score']:.2f}")
        print(f"Is Mobile: {device_profile['is_mobile']}")
        
        print(f"\n🎛️ Performance Settings:")
        print(f"  Chunk Size: {performance_settings['chunk_size']:,}")
        print(f"  Cache Size: {performance_settings['cache_size']:,}")
        print(f"  Update Frequency: {performance_settings['update_frequency']}s")
        print(f"  Quality: {performance_settings['quality_settings']['chart_quality']}")
        
        print(f"\n✨ Enabled Features:")
        for feature_set, features in enabled_features.items():
            print(f"  {feature_set.title()}: {len(features)} features")
            for feature in features[:3]:  # Show first 3 features
                print(f"    • {feature}")
            if len(features) > 3:
                print(f"    • ... and {len(features) - 3} more")


async def demo_load_prediction_and_resource_allocation():
    """Demonstrate load prediction and dynamic resource allocation."""
    print("\n" + "="*80)
    print("LOAD PREDICTION AND RESOURCE ALLOCATION DEMO")
    print("="*80)
    
    optimizer = get_intelligent_optimizer()
    
    # Simulate historical load data
    print("\n📊 Simulating historical system metrics...")
    base_time = datetime.utcnow()
    
    for i in range(50):
        # Simulate varying load patterns
        hour_of_day = (base_time.hour + i) % 24
        
        # Business hours have higher load
        if 9 <= hour_of_day <= 17:
            cpu_base = 60.0
            memory_base = 55.0
            users_base = 80
        elif 18 <= hour_of_day <= 22:
            cpu_base = 45.0
            memory_base = 40.0
            users_base = 50
        else:
            cpu_base = 25.0
            memory_base = 30.0
            users_base = 15
        
        # Add some randomness
        import random
        cpu_usage = cpu_base + random.uniform(-10, 15)
        memory_usage = memory_base + random.uniform(-5, 10)
        concurrent_users = users_base + random.randint(-20, 30)
        
        metrics = PerformanceMetrics(
            operation_type="system_monitoring",
            execution_time=1.0,
            memory_usage=max(0, min(100, memory_usage)) / 100.0,
            cpu_usage=max(0, min(100, cpu_usage)) / 100.0,
            throughput=1.0,
            error_rate=random.uniform(0, 0.05),
            timestamp=(base_time - timedelta(hours=50-i)).timestamp()
        )
        
        optimizer.load_predictor.record_metrics(metrics)
    
    print("✅ Historical data loaded (50 data points)")
    
    # Test different prediction horizons
    horizons = [15, 30, 60, 120]  # minutes
    
    for horizon in horizons:
        print(f"\n🔮 Predicting load for next {horizon} minutes:")
        print("-" * 40)
        
        result = await optimizer.predict_and_allocate_resources(
            timedelta(minutes=horizon)
        )
        
        prediction = result['prediction']
        allocation = result['allocation']
        
        print(f"Predicted CPU Usage: {prediction['predicted_cpu_usage']:.1f}%")
        print(f"Predicted Memory Usage: {prediction['predicted_memory_usage']:.1f}%")
        print(f"Predicted Concurrent Users: {prediction['predicted_users']}")
        print(f"Confidence Score: {prediction['confidence']:.2f}")
        
        print(f"\n🎯 Resource Allocation:")
        cpu_alloc = allocation['cpu_allocation']
        memory_alloc = allocation['memory_allocation']
        
        print(f"  CPU Allocation:")
        for service, percentage in cpu_alloc.items():
            print(f"    {service}: {percentage:.1%}")
        
        print(f"  Memory Allocation:")
        for component, percentage in memory_alloc.items():
            print(f"    {component}: {percentage:.1%}")
        
        print(f"\n📈 Key Factors:")
        for factor, weight in prediction['factors'].items():
            print(f"  {factor}: {weight:.2f}")


async def demo_intelligent_caching():
    """Demonstrate intelligent caching with predictive pre-loading."""
    print("\n" + "="*80)
    print("INTELLIGENT CACHING WITH PREDICTIVE PRE-LOADING DEMO")
    print("="*80)
    
    optimizer = get_intelligent_optimizer()
    cache_manager = optimizer.cache_manager
    
    # Simulate user access patterns
    users = ['alice', 'bob', 'charlie']
    data_types = [
        'dashboard_data', 'user_profile', 'recent_files', 'analytics_report',
        'project_summary', 'team_metrics', 'financial_data', 'system_logs'
    ]
    
    print("\n👥 Simulating user access patterns...")
    
    # Simulate realistic access patterns
    for cycle in range(5):
        print(f"  Cycle {cycle + 1}/5")
        
        for user in users:
            # Each user has preferred access patterns
            if user == 'alice':
                # Alice frequently accesses dashboard and analytics
                sequence = ['dashboard_data', 'analytics_report', 'team_metrics', 'dashboard_data']
            elif user == 'bob':
                # Bob works with files and projects
                sequence = ['recent_files', 'project_summary', 'user_profile', 'recent_files']
            else:
                # Charlie is an admin who checks everything
                sequence = ['system_logs', 'financial_data', 'team_metrics', 'user_profile']
            
            for key in sequence:
                # Try to get from cache
                cached_value = await cache_manager.get(key, user_id=user)
                
                if cached_value is None:
                    # Cache miss - simulate data loading and caching
                    data = f"Generated data for {key} (cycle {cycle})"
                    await cache_manager.set(
                        key, data, user_id=user, 
                        priority=1.0 + (cycle * 0.1)  # Increase priority over time
                    )
                    print(f"    {user}: MISS - Loaded and cached {key}")
                else:
                    print(f"    {user}: HIT - Retrieved {key} from cache")
        
        # Small delay between cycles
        await asyncio.sleep(0.1)
    
    # Show cache statistics
    print(f"\n📊 Cache Performance Statistics:")
    print("-" * 40)
    
    stats = cache_manager.get_cache_stats()
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"Total Items: {stats['total_items']}")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Cache Size: {stats['cache_size_mb']:.2f} MB")
    print(f"Evictions: {stats['evictions']}")
    
    # Show user patterns learned
    print(f"\n🧠 Learned User Patterns:")
    print("-" * 30)
    
    for user in users:
        if user in cache_manager.user_patterns:
            pattern = cache_manager.user_patterns[user]
            preferences = pattern.get('key_preferences', {})
            
            print(f"{user.title()}:")
            # Show top 3 most accessed items
            sorted_prefs = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
            for key, count in sorted_prefs[:3]:
                print(f"  • {key}: {count} accesses")
    
    # Demonstrate cache optimization
    print(f"\n🎯 Cache Optimization Recommendations:")
    print("-" * 45)
    
    cache_optimization = await optimizer.optimize_caching_strategy('alice')
    
    for recommendation in cache_optimization['recommendations']:
        print(f"• {recommendation['type']}: {recommendation['reason']}")
        print(f"  Action: {recommendation['suggested_action']}")


async def demo_progressive_enhancement():
    """Demonstrate progressive enhancement with automatic adaptation."""
    print("\n" + "="*80)
    print("PROGRESSIVE ENHANCEMENT WITH AUTOMATIC ADAPTATION DEMO")
    print("="*80)
    
    optimizer = get_intelligent_optimizer()
    enhancement_manager = optimizer.enhancement_manager
    
    # Test different performance scenarios
    scenarios = [
        {
            'name': 'Optimal Performance',
            'performance': {
                'cpu_usage': 0.3,
                'memory_usage': 0.4,
                'network_speed': 1000.0,
                'response_time': 0.5
            }
        },
        {
            'name': 'High Load',
            'performance': {
                'cpu_usage': 0.85,
                'memory_usage': 0.80,
                'network_speed': 100.0,
                'response_time': 2.5
            }
        },
        {
            'name': 'Critical Load',
            'performance': {
                'cpu_usage': 0.95,
                'memory_usage': 0.92,
                'network_speed': 10.0,
                'response_time': 8.0
            }
        }
    ]
    
    # Test with different device types
    from scrollintel.core.intelligent_performance_optimizer import DeviceProfile
    
    test_devices = [
        DeviceProfile(
            device_id="high_end_desktop",
            device_type=DeviceCapability.HIGH_END,
            cpu_cores=16,
            memory_gb=32.0,
            network_speed=1000.0,
            screen_resolution=(3840, 2160),
            supports_webgl=True,
            supports_webworkers=True,
            performance_score=0.9
        ),
        DeviceProfile(
            device_id="mobile_device",
            device_type=DeviceCapability.MOBILE,
            cpu_cores=6,
            memory_gb=4.0,
            network_speed=50.0,
            screen_resolution=(1080, 1920),
            supports_webgl=True,
            supports_webworkers=True,
            is_mobile=True,
            performance_score=0.6
        )
    ]
    
    feature_sets = ['dashboard', 'visualization', 'data_processing']
    
    for device in test_devices:
        print(f"\n📱 Device: {device.device_type.value.title()} (Score: {device.performance_score:.2f})")
        print("=" * 60)
        
        for scenario in scenarios:
            print(f"\n🎭 Scenario: {scenario['name']}")
            print("-" * 30)
            
            for feature_set in feature_sets:
                features = await enhancement_manager.adapt_features_for_device(
                    feature_set, device, scenario['performance']
                )
                
                print(f"{feature_set.title()}: {len(features)} features enabled")
                
                # Show some example features
                base_features = [f for f in features if f in ['basic_charts', 'data_tables', 'simple_filters']]
                enhanced_features = [f for f in features if f not in base_features]
                
                if base_features:
                    print(f"  Base: {', '.join(base_features[:2])}")
                if enhanced_features:
                    print(f"  Enhanced: {', '.join(enhanced_features[:3])}")
                    if len(enhanced_features) > 3:
                        print(f"  ... and {len(enhanced_features) - 3} more")


async def demo_performance_feedback_learning():
    """Demonstrate performance feedback learning and adaptation."""
    print("\n" + "="*80)
    print("PERFORMANCE FEEDBACK LEARNING AND ADAPTATION DEMO")
    print("="*80)
    
    optimizer = get_intelligent_optimizer()
    enhancement_manager = optimizer.enhancement_manager
    
    print("\n🎓 Simulating performance feedback learning...")
    
    # Simulate feedback for different features on different devices
    feedback_scenarios = [
        {
            'feature_set': 'visualization',
            'feature': 'webgl_visualizations',
            'device_capability': DeviceCapability.HIGH_END,
            'performance_score': 0.9,
            'description': 'WebGL works great on high-end devices'
        },
        {
            'feature_set': 'visualization',
            'feature': 'webgl_visualizations',
            'device_capability': DeviceCapability.MEDIUM_END,
            'performance_score': 0.6,
            'description': 'WebGL has mixed performance on medium devices'
        },
        {
            'feature_set': 'visualization',
            'feature': 'webgl_visualizations',
            'device_capability': DeviceCapability.LOW_END,
            'performance_score': 0.2,
            'description': 'WebGL performs poorly on low-end devices'
        },
        {
            'feature_set': 'dashboard',
            'feature': 'real_time_updates',
            'device_capability': DeviceCapability.MOBILE,
            'performance_score': 0.3,
            'description': 'Real-time updates drain mobile battery'
        }
    ]
    
    for scenario in feedback_scenarios:
        print(f"\n📝 Recording feedback: {scenario['description']}")
        print(f"   Feature: {scenario['feature']}")
        print(f"   Device: {scenario['device_capability'].value}")
        print(f"   Score: {scenario['performance_score']:.1f}")
        
        await enhancement_manager.record_feature_performance(
            scenario['feature_set'],
            scenario['feature'],
            scenario['device_capability'],
            scenario['performance_score']
        )
    
    # Show adaptation statistics
    print(f"\n📊 Adaptation Statistics:")
    print("-" * 30)
    
    stats = enhancement_manager.get_adaptation_stats()
    print(f"Total Adaptations: {stats['total_adaptations']}")
    print(f"Feature Sets: {', '.join(stats['feature_sets'])}")
    
    if stats['device_capabilities']:
        print(f"Device Capability Distribution:")
        for capability, count in stats['device_capabilities'].items():
            print(f"  {capability}: {count}")
    
    # Show how features are now adapted based on learning
    print(f"\n🧠 Learned Adaptations:")
    print("-" * 25)
    
    # Check if poorly performing features were removed
    viz_config = enhancement_manager.enhancement_configs.get('visualization')
    if viz_config:
        low_end_features = viz_config.enhanced_features.get(DeviceCapability.LOW_END, [])
        if 'webgl_visualizations' not in low_end_features:
            print("✅ WebGL visualizations removed from low-end devices (learned from feedback)")
        
        mobile_features = viz_config.enhanced_features.get(DeviceCapability.MOBILE, [])
        print(f"📱 Mobile visualization features: {len(mobile_features)} enabled")


async def demo_api_endpoints():
    """Demonstrate API endpoint functionality."""
    print("\n" + "="*80)
    print("API ENDPOINTS DEMONSTRATION")
    print("="*80)
    
    print("\n🌐 Testing API endpoint functions...")
    
    # Test device optimization endpoint
    print("\n1. Device Optimization Endpoint:")
    print("-" * 35)
    
    result = await optimize_performance_for_request(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        client_hints={'cpu_cores': 8, 'memory_gb': 16.0},
        user_id="api_test_user"
    )
    
    print(f"✅ Optimization Status: {result['optimized']}")
    print(f"📱 Device Type: {result['device_optimization']['device_profile']['device_type']}")
    print(f"⚡ Performance Score: {result['device_optimization']['device_profile']['performance_score']:.2f}")
    print(f"💾 Cache Hit Rate: {result['cache_optimization']['current_performance']['hit_rate']:.1%}")
    
    # Test load prediction endpoint
    print("\n2. Load Prediction Endpoint:")
    print("-" * 30)
    
    prediction_result = await predict_system_load(horizon_minutes=60)
    
    print(f"✅ Prediction Status: {prediction_result['predicted']}")
    
    if prediction_result['predicted']:
        prediction_data = prediction_result['result']['prediction']
        print(f"🔮 Predicted CPU: {prediction_data['predicted_cpu_usage']:.1f}%")
        print(f"🔮 Predicted Memory: {prediction_data['predicted_memory_usage']:.1f}%")
        print(f"👥 Predicted Users: {prediction_data['predicted_users']}")
        print(f"🎯 Confidence: {prediction_data['confidence']:.2f}")


async def demo_optimization_strategies():
    """Demonstrate different optimization strategies."""
    print("\n" + "="*80)
    print("OPTIMIZATION STRATEGIES DEMONSTRATION")
    print("="*80)
    
    optimizer = get_intelligent_optimizer()
    
    strategies = [
        OptimizationStrategy.CONSERVATIVE,
        OptimizationStrategy.BALANCED,
        OptimizationStrategy.AGGRESSIVE,
        OptimizationStrategy.ADAPTIVE
    ]
    
    print("\n🎛️ Testing different optimization strategies...")
    
    for strategy in strategies:
        print(f"\n📊 Strategy: {strategy.value.upper()}")
        print("-" * 40)
        
        # Set strategy
        optimizer.current_strategy = strategy
        
        # Simulate performance adjustment
        test_performance = {
            'cpu_usage': 0.75,
            'memory_usage': 0.68,
            'network_speed': 100.0,
            'response_time': 1.5
        }
        
        await optimizer._adjust_optimization_strategy(test_performance)
        
        print(f"Current Strategy: {optimizer.current_strategy.value}")
        
        # Get optimization stats
        stats = optimizer.get_optimization_stats()
        print(f"Performance Targets:")
        for metric, target in stats['performance_targets'].items():
            print(f"  {metric}: {target}")


async def main():
    """Run all demonstration scenarios."""
    print("🚀 SCROLLINTEL INTELLIGENT PERFORMANCE OPTIMIZATION ENGINE DEMO")
    print("=" * 80)
    print("This demo showcases the comprehensive performance optimization capabilities")
    print("including device adaptation, load prediction, intelligent caching, and")
    print("progressive enhancement with machine learning-based optimization.")
    
    try:
        # Run all demo scenarios
        await demo_device_capability_detection()
        await demo_load_prediction_and_resource_allocation()
        await demo_intelligent_caching()
        await demo_progressive_enhancement()
        await demo_performance_feedback_learning()
        await demo_api_endpoints()
        await demo_optimization_strategies()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nThe Intelligent Performance Optimization Engine demonstrates:")
        print("• 🎯 Adaptive device capability detection and optimization")
        print("• 🔮 Predictive load forecasting and resource allocation")
        print("• 🧠 Machine learning-based caching with pattern recognition")
        print("• ⚡ Progressive enhancement with automatic adaptation")
        print("• 📊 Performance feedback learning and continuous improvement")
        print("• 🌐 Comprehensive API endpoints for integration")
        print("• 🎛️ Multiple optimization strategies for different scenarios")
        
        print(f"\n🎉 ScrollIntel now provides bulletproof performance optimization")
        print(f"   that adapts to any device, predicts system needs, and learns")
        print(f"   from user behavior to continuously improve performance!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo encountered an error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())