"""
Performance Testing Suite
Tests system performance under enterprise-scale loads
"""
import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from scrollintel.core.realtime_orchestration_engine import RealtimeOrchestrationEngine
from scrollintel.core.agent_registry import AgentRegistry
from scrollintel.core.data_pipeline import DataPipeline
from scrollintel.engines.intelligence_engine import IntelligenceEngine
from scrollintel.models.agent_steering_models import BusinessTask, TaskPriority


class TestPerformanceScaling:
    """Test system performance under various load conditions"""
    
    @pytest.fixture
    def performance_test_config(self):
        """Configuration for performance tests"""
        return {
            'concurrent_users': [10, 50, 100, 500, 1000],
            'data_volumes': [1000, 10000, 100000, 1000000],
            'task_complexities': ['simple', 'medium', 'complex', 'enterprise'],
            'response_time_thresholds': {
                'simple': 1.0,      # 1 second
                'medium': 5.0,      # 5 seconds
                'complex': 30.0,    # 30 seconds
                'enterprise': 300.0  # 5 minutes
            },
            'throughput_targets': {
                'simple': 100,      # tasks per second
                'medium': 20,       # tasks per second
                'complex': 5,       # tasks per second
                'enterprise': 1     # tasks per second
            }
        }
    
    @pytest.fixture
    def load_test_data(self):
        """Generate large datasets for load testing"""
        sizes = {
            'small': 1000,
            'medium': 10000,
            'large': 100000,
            'enterprise': 1000000
        }
        
        datasets = {}
        for size_name, size in sizes.items():
            datasets[size_name] = {
                'customers': pd.DataFrame({
                    'customer_id': range(1, size + 1),
                    'name': [f'Customer_{i}' for i in range(1, size + 1)],
                    'email': [f'customer{i}@company.com' for i in range(1, size + 1)],
                    'signup_date': pd.date_range('2020-01-01', periods=size),
                    'total_spent': np.random.uniform(100, 50000, size),
                    'last_activity': pd.date_range('2024-01-01', periods=size),
                    'segment': np.random.choice(['premium', 'standard', 'basic'], size)
                }),
                'transactions': pd.DataFrame({
                    'transaction_id': range(1, size * 10 + 1),
                    'customer_id': np.random.randint(1, size + 1, size * 10),
                    'amount': np.random.uniform(10, 5000, size * 10),
                    'timestamp': pd.date_range('2024-01-01', periods=size * 10, freq='5min'),
                    'product_category': np.random.choice(['A', 'B', 'C', 'D'], size * 10),
                    'payment_method': np.random.choice(['credit', 'debit', 'cash'], size * 10)
                })
            }
        
        return datasets
    
    @pytest.mark.asyncio
    async def test_concurrent_user_scaling(self, performance_test_config):
        """Test system performance with increasing concurrent users"""
        
        orchestration_engine = RealtimeOrchestrationEngine()
        
        # Mock agent processing
        mock_agent = Mock()
        mock_agent.process_task = AsyncMock(return_value={
            'status': 'completed',
            'results': 'test_results',
            'execution_time': 0.1
        })
        
        orchestration_engine.agent_registry = Mock()
        orchestration_engine.agent_registry.get_available_agents.return_value = [mock_agent] * 10
        
        results = {}
        
        for user_count in performance_test_config['concurrent_users']:
            # Create concurrent tasks
            tasks = []
            for i in range(user_count):
                task = BusinessTask(
                    id=f'concurrent_task_{i}',
                    title=f'Concurrent Task {i}',
                    description='Performance test task',
                    priority=TaskPriority.MEDIUM
                )
                tasks.append(task)
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Execute concurrent tasks
            task_results = await asyncio.gather(*[
                orchestration_engine.execute_business_workflow(task) for task in tasks
            ])
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            total_time = end_time - start_time
            throughput = user_count / total_time
            memory_usage = end_memory - start_memory
            
            results[user_count] = {
                'total_time': total_time,
                'throughput': throughput,
                'memory_usage': memory_usage,
                'success_rate': sum(1 for r in task_results if r['status'] == 'completed') / len(task_results),
                'average_response_time': total_time / user_count
            }
            
            # Validate performance thresholds
            assert results[user_count]['success_rate'] >= 0.95  # 95% success rate
            assert results[user_count]['average_response_time'] <= 10.0  # 10 second max response
            
            print(f"Users: {user_count}, Throughput: {throughput:.2f} tasks/sec, "
                  f"Memory: {memory_usage:.2f} MB, Success Rate: {results[user_count]['success_rate']:.2%}")
        
        # Validate scaling characteristics
        assert results[10]['throughput'] > 0  # Basic functionality
        assert results[1000]['success_rate'] >= 0.90  # Maintains quality under load
    
    @pytest.mark.asyncio
    async def test_data_volume_scaling(self, performance_test_config, load_test_data):
        """Test performance with increasing data volumes"""
        
        data_pipeline = DataPipeline()
        
        # Mock data connectors
        data_pipeline.connectors = {
            'database': Mock(),
            'api': Mock(),
            'file': Mock()
        }
        
        results = {}
        
        for volume_name, dataset in load_test_data.items():
            customer_data = dataset['customers']
            transaction_data = dataset['transactions']
            
            # Mock data extraction
            data_pipeline.connectors['database'].extract_data = AsyncMock(
                return_value=customer_data
            )
            
            # Measure data processing performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Process data
            processed_customers = await data_pipeline.process_customer_data(customer_data)
            processed_transactions = await data_pipeline.process_transaction_data(transaction_data)
            
            # Perform analytics
            analytics_results = await data_pipeline.compute_analytics(
                processed_customers, processed_transactions
            )
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            processing_time = end_time - start_time
            records_per_second = len(customer_data) / processing_time
            memory_usage = end_memory - start_memory
            
            results[volume_name] = {
                'record_count': len(customer_data),
                'processing_time': processing_time,
                'records_per_second': records_per_second,
                'memory_usage': memory_usage,
                'memory_per_record': memory_usage / len(customer_data) if len(customer_data) > 0 else 0
            }
            
            # Validate performance requirements
            assert processing_time <= 300  # 5 minutes max for any dataset
            assert records_per_second >= 100  # Minimum throughput
            
            print(f"Volume: {volume_name} ({len(customer_data):,} records), "
                  f"Time: {processing_time:.2f}s, "
                  f"Throughput: {records_per_second:.0f} records/sec, "
                  f"Memory: {memory_usage:.2f} MB")
        
        # Validate linear scaling characteristics
        small_throughput = results['small']['records_per_second']
        large_throughput = results['large']['records_per_second']
        
        # Throughput should not degrade significantly with volume
        assert large_throughput >= small_throughput * 0.5  # At least 50% of small dataset throughput
    
    @pytest.mark.asyncio
    async def test_complex_workflow_performance(self, performance_test_config):
        """Test performance of complex multi-agent workflows"""
        
        orchestration_engine = RealtimeOrchestrationEngine()
        
        # Create workflows of varying complexity
        workflow_configs = {
            'simple': {
                'agents': 1,
                'data_sources': 1,
                'processing_steps': 3,
                'expected_time': 1.0
            },
            'medium': {
                'agents': 3,
                'data_sources': 2,
                'processing_steps': 8,
                'expected_time': 5.0
            },
            'complex': {
                'agents': 5,
                'data_sources': 4,
                'processing_steps': 15,
                'expected_time': 30.0
            },
            'enterprise': {
                'agents': 10,
                'data_sources': 8,
                'processing_steps': 25,
                'expected_time': 300.0
            }
        }
        
        results = {}
        
        for complexity, config in workflow_configs.items():
            # Mock agents with realistic processing times
            mock_agents = []
            for i in range(config['agents']):
                agent = Mock()
                # Simulate realistic processing time based on complexity
                processing_time = config['expected_time'] / config['agents'] * np.random.uniform(0.8, 1.2)
                
                async def mock_process(task, pt=processing_time):
                    await asyncio.sleep(pt / 10)  # Scale down for testing
                    return {
                        'status': 'completed',
                        'results': f'agent_{i}_results',
                        'processing_time': pt
                    }
                
                agent.process_task = mock_process
                mock_agents.append(agent)
            
            orchestration_engine.agent_registry = Mock()
            orchestration_engine.agent_registry.get_available_agents.return_value = mock_agents
            
            # Create complex task
            task = BusinessTask(
                id=f'complex_task_{complexity}',
                title=f'Complex {complexity.title()} Workflow',
                description=f'Performance test for {complexity} workflow',
                priority=TaskPriority.HIGH
            )
            
            # Measure workflow performance
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            
            result = await orchestration_engine.execute_complex_workflow(
                task, 
                agent_count=config['agents'],
                data_sources=config['data_sources'],
                processing_steps=config['processing_steps']
            )
            
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            cpu_usage = end_cpu - start_cpu
            
            results[complexity] = {
                'execution_time': execution_time,
                'cpu_usage': cpu_usage,
                'agents_used': config['agents'],
                'success': result['status'] == 'completed',
                'efficiency': config['expected_time'] / 10 / execution_time  # Adjusted for test scaling
            }
            
            # Validate performance thresholds
            max_time = performance_test_config['response_time_thresholds'][complexity] / 10  # Scale for testing
            assert execution_time <= max_time, f"{complexity} workflow exceeded time limit"
            assert result['status'] == 'completed', f"{complexity} workflow failed"
            
            print(f"Complexity: {complexity}, Time: {execution_time:.2f}s, "
                  f"CPU: {cpu_usage:.1f}%, Agents: {config['agents']}, "
                  f"Efficiency: {results[complexity]['efficiency']:.2f}")
        
        # Validate that complex workflows scale appropriately
        assert results['simple']['success']
        assert results['enterprise']['success']
        assert results['enterprise']['execution_time'] > results['simple']['execution_time']
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, load_test_data):
        """Test memory usage optimization under load"""
        
        orchestration_engine = RealtimeOrchestrationEngine()
        
        # Test memory usage with different data sizes
        memory_results = {}
        
        for size_name, dataset in load_test_data.items():
            if size_name == 'enterprise':  # Skip largest dataset for memory test
                continue
                
            # Measure baseline memory
            baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Load and process data
            start_time = time.time()
            
            # Simulate data processing
            processed_data = await orchestration_engine.process_large_dataset(
                dataset['customers'],
                chunk_size=1000,  # Process in chunks to optimize memory
                enable_streaming=True
            )
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Clean up and measure final memory
            del processed_data
            import gc
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            processing_time = time.time() - start_time
            
            memory_results[size_name] = {
                'baseline_memory': baseline_memory,
                'peak_memory': peak_memory,
                'final_memory': final_memory,
                'memory_increase': peak_memory - baseline_memory,
                'memory_efficiency': len(dataset['customers']) / (peak_memory - baseline_memory),
                'processing_time': processing_time,
                'memory_cleaned': peak_memory - final_memory > 0
            }
            
            # Validate memory efficiency
            assert memory_results[size_name]['memory_cleaned'], "Memory not properly cleaned up"
            assert memory_results[size_name]['memory_efficiency'] > 100, "Poor memory efficiency"
            
            print(f"Size: {size_name}, Records: {len(dataset['customers']):,}, "
                  f"Peak Memory: {peak_memory:.1f} MB, "
                  f"Efficiency: {memory_results[size_name]['memory_efficiency']:.0f} records/MB")
        
        # Validate memory scaling
        small_efficiency = memory_results['small']['memory_efficiency']
        large_efficiency = memory_results['large']['memory_efficiency']
        
        # Memory efficiency should not degrade significantly with size
        assert large_efficiency >= small_efficiency * 0.7, "Memory efficiency degrades too much with scale"
    
    @pytest.mark.asyncio
    async def test_real_time_processing_performance(self):
        """Test real-time data processing performance"""
        
        orchestration_engine = RealtimeOrchestrationEngine()
        
        # Simulate real-time data stream
        async def generate_real_time_data(duration_seconds=10, events_per_second=100):
            start_time = time.time()
            event_count = 0
            
            while time.time() - start_time < duration_seconds:
                # Generate batch of events
                batch_size = min(events_per_second, 10)  # Process in small batches
                
                for _ in range(batch_size):
                    event = {
                        'timestamp': datetime.now(),
                        'event_id': event_count,
                        'user_id': np.random.randint(1, 10000),
                        'action': np.random.choice(['click', 'view', 'purchase', 'search']),
                        'value': np.random.uniform(1, 1000)
                    }
                    yield event
                    event_count += 1
                
                # Control event rate
                await asyncio.sleep(1.0 / events_per_second * batch_size)
        
        # Test real-time processing
        processed_events = 0
        processing_times = []
        start_time = time.time()
        
        async for event in generate_real_time_data(duration_seconds=5, events_per_second=50):
            event_start = time.time()
            
            # Process event
            result = await orchestration_engine.process_real_time_event(event)
            
            event_end = time.time()
            processing_time = event_end - event_start
            processing_times.append(processing_time)
            processed_events += 1
            
            # Validate real-time requirements
            assert processing_time <= 0.1, f"Event processing too slow: {processing_time:.3f}s"
            assert result['status'] == 'processed', "Event processing failed"
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        average_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        throughput = processed_events / total_time
        
        # Validate real-time performance
        assert average_processing_time <= 0.05, f"Average processing time too high: {average_processing_time:.3f}s"
        assert max_processing_time <= 0.1, f"Max processing time too high: {max_processing_time:.3f}s"
        assert throughput >= 40, f"Throughput too low: {throughput:.1f} events/sec"
        
        print(f"Processed {processed_events} events in {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} events/sec")
        print(f"Avg processing time: {average_processing_time*1000:.1f}ms")
        print(f"Max processing time: {max_processing_time*1000:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_database_performance(self, load_test_data):
        """Test database performance under load"""
        
        # Mock database operations
        class MockDatabase:
            def __init__(self):
                self.query_times = []
                self.connection_pool_size = 10
                self.active_connections = 0
            
            async def execute_query(self, query, params=None):
                # Simulate database query time
                self.active_connections += 1
                
                # Simulate connection pool contention
                if self.active_connections > self.connection_pool_size:
                    await asyncio.sleep(0.1)  # Connection wait time
                
                # Simulate query execution time based on complexity
                if 'JOIN' in query.upper():
                    query_time = np.random.uniform(0.05, 0.2)  # Complex query
                else:
                    query_time = np.random.uniform(0.01, 0.05)  # Simple query
                
                await asyncio.sleep(query_time)
                self.query_times.append(query_time)
                self.active_connections -= 1
                
                return {'status': 'success', 'execution_time': query_time}
        
        db = MockDatabase()
        
        # Test concurrent database operations
        queries = [
            "SELECT * FROM customers WHERE id = ?",
            "SELECT c.*, t.* FROM customers c JOIN transactions t ON c.id = t.customer_id",
            "INSERT INTO analytics (customer_id, metric, value) VALUES (?, ?, ?)",
            "UPDATE customers SET last_activity = ? WHERE id = ?",
            "SELECT COUNT(*) FROM transactions WHERE date >= ?",
        ]
        
        # Execute concurrent queries
        concurrent_queries = 50
        start_time = time.time()
        
        tasks = []
        for i in range(concurrent_queries):
            query = np.random.choice(queries)
            task = db.execute_query(query, params=[i])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Calculate database performance metrics
        successful_queries = sum(1 for r in results if r['status'] == 'success')
        average_query_time = np.mean(db.query_times)
        max_query_time = np.max(db.query_times)
        queries_per_second = concurrent_queries / total_time
        
        # Validate database performance
        assert successful_queries == concurrent_queries, "Some database queries failed"
        assert average_query_time <= 0.1, f"Average query time too high: {average_query_time:.3f}s"
        assert max_query_time <= 0.5, f"Max query time too high: {max_query_time:.3f}s"
        assert queries_per_second >= 20, f"Database throughput too low: {queries_per_second:.1f} queries/sec"
        
        print(f"Database Performance:")
        print(f"  Concurrent queries: {concurrent_queries}")
        print(f"  Success rate: {successful_queries/concurrent_queries:.2%}")
        print(f"  Throughput: {queries_per_second:.1f} queries/sec")
        print(f"  Avg query time: {average_query_time*1000:.1f}ms")
        print(f"  Max query time: {max_query_time*1000:.1f}ms")


class TestStressTests:
    """Stress tests for extreme load conditions"""
    
    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self):
        """Test system behavior under extreme concurrent load"""
        
        orchestration_engine = RealtimeOrchestrationEngine()
        
        # Mock lightweight agent
        mock_agent = Mock()
        mock_agent.process_task = AsyncMock(return_value={
            'status': 'completed',
            'results': 'stress_test_result'
        })
        
        orchestration_engine.agent_registry = Mock()
        orchestration_engine.agent_registry.get_available_agents.return_value = [mock_agent] * 20
        
        # Create extreme load (1000 concurrent tasks)
        extreme_load = 1000
        tasks = []
        
        for i in range(extreme_load):
            task = BusinessTask(
                id=f'stress_task_{i}',
                title=f'Stress Test Task {i}',
                description='Extreme load stress test',
                priority=TaskPriority.LOW
            )
            tasks.append(task)
        
        # Execute under extreme load
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            results = await asyncio.gather(*[
                orchestration_engine.execute_business_workflow(task) for task in tasks
            ], return_exceptions=True)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Analyze results
            successful_tasks = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'completed')
            failed_tasks = len(results) - successful_tasks
            success_rate = successful_tasks / len(results)
            
            total_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = successful_tasks / total_time
            
            # Validate stress test results
            assert success_rate >= 0.80, f"Success rate too low under stress: {success_rate:.2%}"
            assert memory_usage <= 1000, f"Memory usage too high: {memory_usage:.1f} MB"
            assert throughput >= 10, f"Throughput too low under stress: {throughput:.1f} tasks/sec"
            
            print(f"Stress Test Results:")
            print(f"  Tasks: {extreme_load}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} tasks/sec")
            print(f"  Memory usage: {memory_usage:.1f} MB")
            
        except Exception as e:
            # System should handle extreme load gracefully
            pytest.fail(f"System failed under extreme load: {e}")
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(self):
        """Test system recovery from resource exhaustion"""
        
        orchestration_engine = RealtimeOrchestrationEngine()
        
        # Simulate resource exhaustion
        class ResourceExhaustionSimulator:
            def __init__(self):
                self.memory_pressure = False
                self.cpu_pressure = False
                self.connection_exhaustion = False
            
            async def simulate_memory_pressure(self):
                self.memory_pressure = True
                # Simulate high memory usage
                await asyncio.sleep(2)
                self.memory_pressure = False
            
            async def simulate_cpu_pressure(self):
                self.cpu_pressure = True
                # Simulate high CPU usage
                await asyncio.sleep(2)
                self.cpu_pressure = False
            
            async def simulate_connection_exhaustion(self):
                self.connection_exhaustion = True
                # Simulate connection pool exhaustion
                await asyncio.sleep(2)
                self.connection_exhaustion = False
        
        simulator = ResourceExhaustionSimulator()
        
        # Test recovery from each type of resource exhaustion
        recovery_tests = [
            ('memory_pressure', simulator.simulate_memory_pressure),
            ('cpu_pressure', simulator.simulate_cpu_pressure),
            ('connection_exhaustion', simulator.simulate_connection_exhaustion)
        ]
        
        for test_name, simulation_func in recovery_tests:
            # Start resource exhaustion simulation
            simulation_task = asyncio.create_task(simulation_func())
            
            # Try to execute tasks during resource pressure
            task = BusinessTask(
                id=f'recovery_test_{test_name}',
                title=f'Recovery Test - {test_name}',
                description='Test recovery from resource exhaustion',
                priority=TaskPriority.HIGH
            )
            
            start_time = time.time()
            
            try:
                result = await orchestration_engine.execute_with_resource_monitoring(task)
                execution_time = time.time() - start_time
                
                # Validate recovery behavior
                assert result['status'] in ['completed', 'degraded'], f"Task failed during {test_name}"
                assert execution_time <= 30, f"Recovery took too long for {test_name}: {execution_time:.2f}s"
                
                if result['status'] == 'degraded':
                    assert 'recovery_actions' in result, f"No recovery actions for {test_name}"
                
                print(f"Recovery test {test_name}: {result['status']} in {execution_time:.2f}s")
                
            except Exception as e:
                pytest.fail(f"System failed to recover from {test_name}: {e}")
            
            finally:
                await simulation_task
    
    @pytest.mark.asyncio
    async def test_long_running_stability(self):
        """Test system stability over extended periods"""
        
        orchestration_engine = RealtimeOrchestrationEngine()
        
        # Mock agent with slight variability
        mock_agent = Mock()
        
        async def variable_processing(task):
            # Simulate variable processing times and occasional failures
            processing_time = np.random.uniform(0.1, 0.5)
            await asyncio.sleep(processing_time)
            
            # 5% chance of failure to test error handling
            if np.random.random() < 0.05:
                raise Exception("Random processing failure")
            
            return {
                'status': 'completed',
                'results': 'long_running_result',
                'processing_time': processing_time
            }
        
        mock_agent.process_task = variable_processing
        
        orchestration_engine.agent_registry = Mock()
        orchestration_engine.agent_registry.get_available_agents.return_value = [mock_agent] * 5
        
        # Run continuous load for extended period
        duration_minutes = 2  # 2 minutes for testing (would be hours in production)
        tasks_per_minute = 30
        
        start_time = time.time()
        total_tasks = 0
        successful_tasks = 0
        failed_tasks = 0
        
        while time.time() - start_time < duration_minutes * 60:
            # Create batch of tasks
            batch_size = tasks_per_minute // 6  # 6 batches per minute
            batch_tasks = []
            
            for i in range(batch_size):
                task = BusinessTask(
                    id=f'stability_task_{total_tasks + i}',
                    title=f'Stability Test Task {total_tasks + i}',
                    description='Long-running stability test',
                    priority=TaskPriority.MEDIUM
                )
                batch_tasks.append(task)
            
            # Execute batch
            try:
                batch_results = await asyncio.gather(*[
                    orchestration_engine.execute_business_workflow(task) for task in batch_tasks
                ], return_exceptions=True)
                
                # Count results
                for result in batch_results:
                    total_tasks += 1
                    if isinstance(result, dict) and result.get('status') == 'completed':
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
                
            except Exception as e:
                failed_tasks += len(batch_tasks)
                total_tasks += len(batch_tasks)
            
            # Wait before next batch
            await asyncio.sleep(10)  # 10 seconds between batches
        
        total_time = time.time() - start_time
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        # Validate long-running stability
        assert total_tasks > 0, "No tasks were executed"
        assert success_rate >= 0.90, f"Success rate degraded over time: {success_rate:.2%}"
        
        print(f"Long-running Stability Test:")
        print(f"  Duration: {total_time/60:.1f} minutes")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average throughput: {total_tasks/total_time*60:.1f} tasks/minute")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])