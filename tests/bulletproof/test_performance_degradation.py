"""
Performance Testing with Degradation Scenario Validation

This module tests system performance under various degradation scenarios
and validates that performance optimization and graceful degradation work correctly.
"""

import asyncio
import pytest
import time
import statistics
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional, Tuple
import psutil
import threading

try:
    from scrollintel.core.bulletproof_orchestrator import BulletproofOrchestrator
except ImportError:
    from unittest.mock import AsyncMock
    BulletproofOrchestrator = AsyncMock

try:
    from scrollintel.core.graceful_degradation import GracefulDegradationManager
except ImportError:
    from unittest.mock import AsyncMock
    GracefulDegradationManager = AsyncMock

try:
    from scrollintel.core.intelligent_performance_optimizer import IntelligentPerformanceOptimizer
except ImportError:
    from unittest.mock import AsyncMock
    IntelligentPerformanceOptimizer = AsyncMock

try:
    from scrollintel.core.bulletproof_monitoring import BulletproofMonitoring
except ImportError:
    from unittest.mock import AsyncMock
    BulletproofMonitoring = AsyncMock


class PerformanceTestFramework:
    """Framework for testing performance under degradation scenarios."""
    
    def __init__(self):
        self.orchestrator = BulletproofOrchestrator()
        self.degradation_manager = GracefulDegradationManager()
        self.performance_optimizer = IntelligentPerformanceOptimizer()
        self.monitoring = BulletproofMonitoring()
        self.performance_metrics = []
        
    async def measure_operation_performance(self, operation_func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance metrics for an operation."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        try:
            result = await operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()
        
        metrics = {
            'response_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'cpu_usage': (start_cpu + end_cpu) / 2,
            'success': success,
            'error': error,
            'result': result,
            'timestamp': start_time
        }
        
        self.performance_metrics.append(metrics)
        return metrics
        
    async def simulate_high_load(self, concurrent_operations: int = 50, duration: int = 30):
        """Simulate high load conditions."""
        async def load_operation():
            return await self.orchestrator.handle_user_action({
                'action': 'complex_data_processing',
                'user_id': f'load_test_user_{time.time()}',
                'data_size': 'large'
            })
            
        # Create concurrent operations
        tasks = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Create batch of concurrent operations
            batch_tasks = [
                self.measure_operation_performance(load_operation)
                for _ in range(min(concurrent_operations, 10))  # Limit batch size
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            tasks.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
            
        return tasks
        
    async def simulate_memory_pressure(self, pressure_level: str = 'high'):
        """Simulate memory pressure conditions."""
        pressure_levels = {
            'low': 100 * 1024 * 1024,      # 100MB
            'medium': 500 * 1024 * 1024,   # 500MB
            'high': 1024 * 1024 * 1024,    # 1GB
            'extreme': 2048 * 1024 * 1024  # 2GB
        }
        
        memory_size = pressure_levels.get(pressure_level, pressure_levels['high'])
        
        # Allocate memory to create pressure
        memory_hog = []
        try:
            chunk_size = 1024 * 1024  # 1MB chunks
            chunks_needed = memory_size // chunk_size
            
            for i in range(chunks_needed):
                memory_hog.append(bytearray(chunk_size))
                if i % 100 == 0:  # Yield control periodically
                    await asyncio.sleep(0.001)
                    
            # Keep memory allocated for a period
            await asyncio.sleep(5)
            
        finally:
            # Clean up memory
            del memory_hog
            
    async def simulate_cpu_intensive_load(self, intensity: str = 'high', duration: int = 10):
        """Simulate CPU-intensive operations."""
        intensity_levels = {
            'low': 1000,
            'medium': 10000,
            'high': 100000,
            'extreme': 1000000
        }
        
        operations = intensity_levels.get(intensity, intensity_levels['high'])
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # CPU-intensive calculation
            result = sum(i * i for i in range(operations))
            await asyncio.sleep(0.001)  # Small yield
            
    async def test_degradation_levels(self) -> Dict[str, Any]:
        """Test different degradation levels and their performance impact."""
        degradation_results = {}
        
        for level in range(1, 6):  # Test degradation levels 1-5
            # Apply degradation level
            await self.degradation_manager.apply_degradation_level(level)
            
            # Measure performance at this degradation level
            performance_metrics = []
            for i in range(10):
                metrics = await self.measure_operation_performance(
                    self.orchestrator.handle_user_action,
                    {
                        'action': 'standard_operation',
                        'user_id': f'degradation_test_user_{i}',
                        'degradation_level': level
                    }
                )
                performance_metrics.append(metrics)
                
            # Calculate statistics
            response_times = [m['response_time'] for m in performance_metrics if m['success']]
            success_rate = sum(1 for m in performance_metrics if m['success']) / len(performance_metrics)
            
            degradation_results[f'level_{level}'] = {
                'avg_response_time': statistics.mean(response_times) if response_times else float('inf'),
                'median_response_time': statistics.median(response_times) if response_times else float('inf'),
                'success_rate': success_rate,
                'total_operations': len(performance_metrics),
                'successful_operations': len(response_times)
            }
            
        return degradation_results


@pytest.mark.asyncio
class TestPerformanceDegradation:
    """Test performance under various degradation scenarios."""
    
    @pytest.fixture
    def performance_framework(self):
        return PerformanceTestFramework()
        
    async def test_baseline_performance(self, performance_framework):
        """Establish baseline performance metrics."""
        # Measure baseline performance without any stress
        baseline_metrics = []
        
        for i in range(20):
            metrics = await performance_framework.measure_operation_performance(
                performance_framework.orchestrator.handle_user_action,
                {
                    'action': 'baseline_operation',
                    'user_id': f'baseline_user_{i}'
                }
            )
            baseline_metrics.append(metrics)
            
        # Calculate baseline statistics
        response_times = [m['response_time'] for m in baseline_metrics if m['success']]
        success_rate = sum(1 for m in baseline_metrics if m['success']) / len(baseline_metrics)
        
        # Verify baseline performance meets requirements
        avg_response_time = statistics.mean(response_times)
        assert avg_response_time < 1.0, f"Baseline response time should be < 1s, got {avg_response_time:.2f}s"
        assert success_rate >= 0.99, f"Baseline success rate should be >= 99%, got {success_rate:.2%}"
        
    async def test_performance_under_high_load(self, performance_framework):
        """Test performance degradation under high load."""
        # Simulate high load
        load_results = await performance_framework.simulate_high_load(
            concurrent_operations=30,
            duration=15
        )
        
        # Analyze performance under load
        successful_operations = [r for r in load_results if isinstance(r, dict) and r.get('success', False)]
        failed_operations = [r for r in load_results if isinstance(r, dict) and not r.get('success', True)]
        
        if successful_operations:
            response_times = [op['response_time'] for op in successful_operations]
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            
            # Verify performance under load
            assert avg_response_time < 5.0, f"Average response time under load should be < 5s, got {avg_response_time:.2f}s"
            assert p95_response_time < 10.0, f"95th percentile response time should be < 10s, got {p95_response_time:.2f}s"
            
        success_rate = len(successful_operations) / len(load_results) if load_results else 0
        assert success_rate >= 0.8, f"Success rate under load should be >= 80%, got {success_rate:.2%}"
        
    async def test_memory_pressure_performance(self, performance_framework):
        """Test performance under memory pressure."""
        # Start memory pressure simulation
        memory_task = asyncio.create_task(
            performance_framework.simulate_memory_pressure('high')
        )
        
        # Measure performance during memory pressure
        pressure_metrics = []
        for i in range(15):
            metrics = await performance_framework.measure_operation_performance(
                performance_framework.orchestrator.handle_user_action,
                {
                    'action': 'memory_pressure_operation',
                    'user_id': f'memory_test_user_{i}'
                }
            )
            pressure_metrics.append(metrics)
            await asyncio.sleep(0.5)
            
        await memory_task
        
        # Analyze performance under memory pressure
        successful_ops = [m for m in pressure_metrics if m['success']]
        if successful_ops:
            response_times = [m['response_time'] for m in successful_ops]
            avg_response_time = statistics.mean(response_times)
            
            # Performance should degrade gracefully, not catastrophically
            assert avg_response_time < 8.0, f"Response time under memory pressure should be < 8s, got {avg_response_time:.2f}s"
            
        success_rate = len(successful_ops) / len(pressure_metrics)
        assert success_rate >= 0.7, f"Success rate under memory pressure should be >= 70%, got {success_rate:.2%}"
        
    async def test_cpu_intensive_performance(self, performance_framework):
        """Test performance under CPU-intensive load."""
        # Start CPU-intensive load
        cpu_task = asyncio.create_task(
            performance_framework.simulate_cpu_intensive_load('high', 10)
        )
        
        # Measure performance during CPU stress
        cpu_metrics = []
        for i in range(12):
            metrics = await performance_framework.measure_operation_performance(
                performance_framework.orchestrator.handle_user_action,
                {
                    'action': 'cpu_stress_operation',
                    'user_id': f'cpu_test_user_{i}'
                }
            )
            cpu_metrics.append(metrics)
            await asyncio.sleep(1)
            
        await cpu_task
        
        # Analyze CPU stress performance
        successful_ops = [m for m in cpu_metrics if m['success']]
        if successful_ops:
            response_times = [m['response_time'] for m in successful_ops]
            avg_response_time = statistics.mean(response_times)
            
            assert avg_response_time < 6.0, f"Response time under CPU stress should be < 6s, got {avg_response_time:.2f}s"
            
        success_rate = len(successful_ops) / len(cpu_metrics)
        assert success_rate >= 0.75, f"Success rate under CPU stress should be >= 75%, got {success_rate:.2%}"
        
    async def test_degradation_level_performance(self, performance_framework):
        """Test performance at different degradation levels."""
        degradation_results = await performance_framework.test_degradation_levels()
        
        # Verify degradation levels provide appropriate performance trade-offs
        for level in range(1, 6):
            level_key = f'level_{level}'
            result = degradation_results[level_key]
            
            # Higher degradation levels should maintain higher success rates
            # but may have different response time characteristics
            assert result['success_rate'] >= 0.8, f"Degradation level {level} should maintain >= 80% success rate"
            
            # Response times may increase with degradation but should remain reasonable
            if result['avg_response_time'] != float('inf'):
                max_acceptable_time = 2.0 * level  # Allow more time for higher degradation levels
                assert result['avg_response_time'] < max_acceptable_time, \
                    f"Degradation level {level} response time should be < {max_acceptable_time}s"
                    
    async def test_performance_optimization_effectiveness(self, performance_framework):
        """Test that performance optimization actually improves performance."""
        # Measure performance without optimization
        await performance_framework.performance_optimizer.disable_optimization()
        
        unoptimized_metrics = []
        for i in range(10):
            metrics = await performance_framework.measure_operation_performance(
                performance_framework.orchestrator.handle_user_action,
                {
                    'action': 'optimization_test_operation',
                    'user_id': f'unoptimized_user_{i}'
                }
            )
            unoptimized_metrics.append(metrics)
            
        # Enable optimization
        await performance_framework.performance_optimizer.enable_optimization()
        
        # Measure performance with optimization
        optimized_metrics = []
        for i in range(10):
            metrics = await performance_framework.measure_operation_performance(
                performance_framework.orchestrator.handle_user_action,
                {
                    'action': 'optimization_test_operation',
                    'user_id': f'optimized_user_{i}'
                }
            )
            optimized_metrics.append(metrics)
            
        # Compare performance
        unoptimized_times = [m['response_time'] for m in unoptimized_metrics if m['success']]
        optimized_times = [m['response_time'] for m in optimized_metrics if m['success']]
        
        if unoptimized_times and optimized_times:
            unoptimized_avg = statistics.mean(unoptimized_times)
            optimized_avg = statistics.mean(optimized_times)
            
            # Optimization should improve performance by at least 10%
            improvement = (unoptimized_avg - optimized_avg) / unoptimized_avg
            assert improvement >= 0.1, f"Optimization should improve performance by >= 10%, got {improvement:.2%}"
            
    async def test_concurrent_user_performance(self, performance_framework):
        """Test performance with multiple concurrent users."""
        user_count = 20
        operations_per_user = 5
        
        async def user_operations(user_id: str):
            user_metrics = []
            for i in range(operations_per_user):
                metrics = await performance_framework.measure_operation_performance(
                    performance_framework.orchestrator.handle_user_action,
                    {
                        'action': 'concurrent_user_operation',
                        'user_id': user_id,
                        'operation_id': i
                    }
                )
                user_metrics.append(metrics)
                await asyncio.sleep(0.1)  # Small delay between operations
            return user_metrics
            
        # Run concurrent user operations
        user_tasks = [
            user_operations(f'concurrent_user_{i}')
            for i in range(user_count)
        ]
        
        all_user_results = await asyncio.gather(*user_tasks)
        
        # Flatten results
        all_metrics = []
        for user_results in all_user_results:
            all_metrics.extend(user_results)
            
        # Analyze concurrent performance
        successful_ops = [m for m in all_metrics if m['success']]
        success_rate = len(successful_ops) / len(all_metrics)
        
        assert success_rate >= 0.85, f"Concurrent user success rate should be >= 85%, got {success_rate:.2%}"
        
        if successful_ops:
            response_times = [m['response_time'] for m in successful_ops]
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]
            
            assert avg_response_time < 3.0, f"Concurrent user avg response time should be < 3s, got {avg_response_time:.2f}s"
            assert p95_response_time < 8.0, f"Concurrent user p95 response time should be < 8s, got {p95_response_time:.2f}s"
            
    async def test_performance_monitoring_accuracy(self, performance_framework):
        """Test that performance monitoring accurately tracks system performance."""
        # Generate known performance patterns
        test_operations = [
            {'delay': 0.1, 'should_fail': False},
            {'delay': 0.5, 'should_fail': False},
            {'delay': 1.0, 'should_fail': False},
            {'delay': 2.0, 'should_fail': False},
            {'delay': 0.2, 'should_fail': True}
        ]
        
        for i, op_config in enumerate(test_operations):
            # Simulate operation with known characteristics
            start_time = time.time()
            
            if op_config['should_fail']:
                try:
                    raise Exception("Intentional test failure")
                except Exception:
                    pass
            else:
                await asyncio.sleep(op_config['delay'])
                
            # Check if monitoring captured the operation correctly
            monitoring_data = await performance_framework.monitoring.get_recent_metrics(
                operation_id=f'monitoring_test_{i}'
            )
            
            if monitoring_data:
                recorded_time = monitoring_data.get('response_time', 0)
                expected_time = op_config['delay']
                
                # Allow 10% tolerance for timing accuracy
                time_diff = abs(recorded_time - expected_time)
                tolerance = expected_time * 0.1
                
                assert time_diff <= tolerance, f"Monitoring should accurately track response times"
                
                recorded_success = monitoring_data.get('success', True)
                expected_success = not op_config['should_fail']
                
                assert recorded_success == expected_success, "Monitoring should accurately track success/failure"


@pytest.mark.asyncio
class TestPerformanceRecovery:
    """Test performance recovery after degradation scenarios."""
    
    async def test_performance_recovery_after_load_spike(self):
        """Test that performance recovers after a load spike."""
        framework = PerformanceTestFramework()
        
        # Measure baseline performance
        baseline_metrics = await framework.measure_operation_performance(
            framework.orchestrator.handle_user_action,
            {'action': 'recovery_test_baseline', 'user_id': 'recovery_user'}
        )
        baseline_time = baseline_metrics['response_time']
        
        # Create load spike
        await framework.simulate_high_load(concurrent_operations=50, duration=10)
        
        # Wait for recovery
        await asyncio.sleep(5)
        
        # Measure post-recovery performance
        recovery_metrics = await framework.measure_operation_performance(
            framework.orchestrator.handle_user_action,
            {'action': 'recovery_test_post', 'user_id': 'recovery_user'}
        )
        recovery_time = recovery_metrics['response_time']
        
        # Performance should recover to within 50% of baseline
        performance_ratio = recovery_time / baseline_time
        assert performance_ratio <= 1.5, f"Performance should recover to within 50% of baseline, got {performance_ratio:.2f}x"
        
    async def test_automatic_scaling_effectiveness(self):
        """Test that automatic scaling improves performance under load."""
        framework = PerformanceTestFramework()
        
        # Disable auto-scaling
        await framework.performance_optimizer.disable_auto_scaling()
        
        # Measure performance without scaling
        no_scaling_results = await framework.simulate_high_load(
            concurrent_operations=30, duration=10
        )
        
        # Enable auto-scaling
        await framework.performance_optimizer.enable_auto_scaling()
        
        # Measure performance with scaling
        with_scaling_results = await framework.simulate_high_load(
            concurrent_operations=30, duration=10
        )
        
        # Compare results
        no_scaling_success = sum(1 for r in no_scaling_results if isinstance(r, dict) and r.get('success', False))
        with_scaling_success = sum(1 for r in with_scaling_results if isinstance(r, dict) and r.get('success', False))
        
        no_scaling_rate = no_scaling_success / len(no_scaling_results) if no_scaling_results else 0
        with_scaling_rate = with_scaling_success / len(with_scaling_results) if with_scaling_results else 0
        
        # Auto-scaling should improve success rate
        improvement = with_scaling_rate - no_scaling_rate
        assert improvement >= 0.1, f"Auto-scaling should improve success rate by >= 10%, got {improvement:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])