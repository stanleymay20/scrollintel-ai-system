"""
Tests for Distributed Data Processing Engine
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch

from ai_data_readiness.engines.distributed_processor import (
    DistributedDataProcessor,
    ProcessingConfig,
    ProcessingTask,
    ResourceMonitor,
    LoadBalancer
)


class TestDistributedDataProcessor:
    """Test cases for DistributedDataProcessor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ProcessingConfig(
            max_workers=4,
            min_workers=2,
            chunk_size=1000,
            monitoring_interval=1.0
        )
        self.processor = DistributedDataProcessor(self.config)
    
    def teardown_method(self):
        """Clean up after tests"""
        if hasattr(self, 'processor'):
            self.processor.stop()
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.config.max_workers == 4
        assert self.processor.config.min_workers == 2
        assert self.processor.current_workers == 2
        assert len(self.processor.active_tasks) == 0
    
    def test_task_submission(self):
        """Test task submission"""
        # Create test data
        df = pd.DataFrame({
            'col1': range(100),
            'col2': np.random.randn(100)
        })
        
        def simple_transform(data):
            return data * 2
        
        task = ProcessingTask(
            task_id="test_task_1",
            data_chunk=df,
            transformation_func=simple_transform,
            priority=1
        )
        
        task_id = self.processor.submit_task(task)
        assert task_id == "test_task_1"
        assert self.processor.processing_stats['total_tasks'] == 1
    
    def test_dataframe_processing(self):
        """Test DataFrame processing with chunking"""
        # Create test DataFrame
        df = pd.DataFrame({
            'numeric_column': range(5000),
            'text_column': ['test'] * 5000
        })
        
        def multiply_numeric(chunk_df):
            result = chunk_df.copy()
            if 'numeric_column' in result.columns:
                result['numeric_column'] = result['numeric_column'] * 2
            return result
        
        # Start processor
        self.processor.start()
        
        # Process DataFrame
        task_ids = self.processor.process_dataframe(
            df=df,
            transformation_func=multiply_numeric,
            chunk_size=1000
        )
        
        # Should create 5 chunks (5000 / 1000)
        assert len(task_ids) == 5
        
        # Wait for results
        results = self.processor.get_results(task_ids, timeout=10.0)
        assert len(results) == 5
        
        # Combine results
        combined_df = self.processor.combine_chunk_results(results)
        assert len(combined_df) == 5000
        assert combined_df['numeric_column'].iloc[0] == 0  # 0 * 2
        assert combined_df['numeric_column'].iloc[1] == 2  # 1 * 2
    
    def test_auto_scaling(self):
        """Test auto-scaling functionality"""
        # Mock resource monitor to trigger scaling
        with patch.object(self.processor.resource_monitor, 'should_scale_up', return_value=True):
            initial_workers = self.processor.current_workers
            self.processor._auto_scale()
            assert self.processor.current_workers == initial_workers + 1
        
        with patch.object(self.processor.resource_monitor, 'should_scale_down', return_value=True):
            initial_workers = self.processor.current_workers
            self.processor._auto_scale()
            assert self.processor.current_workers == initial_workers - 1
    
    def test_performance_optimization(self):
        """Test performance optimization"""
        # Set up some processing stats
        self.processor.processing_stats['completed_tasks'] = 20
        self.processor.processing_stats['total_processing_time'] = 100.0
        self.processor.processing_stats['average_processing_time'] = 5.0
        
        initial_chunk_size = self.processor.adaptive_chunk_size
        
        # Should reduce chunk size due to high average time
        self.processor._optimize_performance()
        
        # Chunk size should be reduced
        assert self.processor.adaptive_chunk_size <= initial_chunk_size
    
    def test_chunk_size_optimization(self):
        """Test chunk size optimization"""
        # Small dataset
        small_chunk = self.processor.optimize_chunk_size(1000)
        assert small_chunk <= self.config.chunk_size
        
        # Large dataset
        large_chunk = self.processor.optimize_chunk_size(2000000)
        assert large_chunk >= self.config.chunk_size
        
        # Complex processing
        complex_chunk = self.processor.optimize_chunk_size(100000, complexity_factor=2.0)
        normal_chunk = self.processor.optimize_chunk_size(100000, complexity_factor=1.0)
        assert complex_chunk < normal_chunk
    
    def test_processing_stats(self):
        """Test processing statistics"""
        stats = self.processor.get_processing_stats()
        
        required_keys = [
            'total_tasks', 'completed_tasks', 'failed_tasks',
            'total_processing_time', 'average_processing_time',
            'throughput_per_second', 'current_workers', 'queue_size',
            'active_tasks', 'adaptive_chunk_size'
        ]
        
        for key in required_keys:
            assert key in stats


class TestResourceMonitor:
    """Test cases for ResourceMonitor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ProcessingConfig(monitoring_interval=0.1)
        self.monitor = ResourceMonitor(self.config)
    
    def teardown_method(self):
        """Clean up after tests"""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        assert len(self.monitor.metrics_history) == 0
        assert not self.monitor._monitoring
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        self.monitor.start_monitoring()
        assert self.monitor._monitoring
        
        # Wait a bit for metrics to be collected
        time.sleep(0.2)
        
        self.monitor.stop_monitoring()
        assert not self.monitor._monitoring
        assert len(self.monitor.metrics_history) > 0
    
    def test_scaling_decisions(self):
        """Test scaling decision logic"""
        # Mock high resource usage
        with patch('psutil.cpu_percent', return_value=90.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 85.0
                
                # Should scale up with high usage and queued tasks
                assert self.monitor.should_scale_up(active_tasks=2, queue_size=5)
                
                # Should not scale down with high usage
                assert not self.monitor.should_scale_down(active_tasks=2, queue_size=0)


class TestLoadBalancer:
    """Test cases for LoadBalancer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.balancer = LoadBalancer()
    
    def test_balancer_initialization(self):
        """Test balancer initialization"""
        assert len(self.balancer.worker_loads) == 0
        assert len(self.balancer.task_history) == 0
        assert self.balancer.balancing_strategy == 'weighted_round_robin'
    
    def test_task_assignment_strategies(self):
        """Test different task assignment strategies"""
        workers = ['worker1', 'worker2', 'worker3']
        task = ProcessingTask(
            task_id="test_task",
            data_chunk=pd.DataFrame({'col': [1, 2, 3]}),
            transformation_func=lambda x: x
        )
        
        # Test different strategies
        strategies = ['round_robin', 'least_loaded', 'weighted_round_robin', 'performance_based']
        
        for strategy in strategies:
            self.balancer.set_balancing_strategy(strategy)
            assigned_worker = self.balancer.assign_task(task, workers)
            assert assigned_worker in workers
    
    def test_worker_performance_tracking(self):
        """Test worker performance tracking"""
        worker_id = 'test_worker'
        
        # Simulate task completions
        execution_times = [1.0, 1.5, 2.0, 1.2, 1.8]
        for exec_time in execution_times:
            self.balancer.task_completed(worker_id, exec_time)
        
        # Check performance tracking
        assert worker_id in self.balancer.task_history
        assert len(self.balancer.task_history[worker_id]) == 5
        assert worker_id in self.balancer.worker_capabilities
        
        capabilities = self.balancer.worker_capabilities[worker_id]
        assert capabilities['task_count'] == 5
        assert capabilities['avg_execution_time'] == sum(execution_times) / len(execution_times)
    
    def test_worker_stats(self):
        """Test worker statistics"""
        # Add some test data
        self.balancer.task_completed('worker1', 1.0)
        self.balancer.task_completed('worker2', 2.0)
        
        stats = self.balancer.get_worker_stats()
        assert 'worker1' in stats
        assert 'worker2' in stats
        
        for worker_stats in stats.values():
            required_keys = ['current_load', 'task_history_count', 'capabilities', 'recent_avg_time']
            for key in required_keys:
                assert key in worker_stats


class TestProcessingTask:
    """Test cases for ProcessingTask"""
    
    def test_task_creation(self):
        """Test task creation"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        task = ProcessingTask(
            task_id="test_task",
            data_chunk=df,
            transformation_func=lambda x: x * 2,
            priority=1,
            metadata={'test': 'value'}
        )
        
        assert task.task_id == "test_task"
        assert len(task.data_chunk) == 3
        assert task.priority == 1
        assert task.metadata['test'] == 'value'
        assert task.created_at > 0


# Integration tests
class TestDistributedProcessingIntegration:
    """Integration tests for distributed processing"""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing workflow"""
        # Create test data
        df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        def complex_transformation(chunk_df):
            """Complex transformation for testing"""
            result = chunk_df.copy()
            
            # Normalize values
            if 'value' in result.columns:
                mean_val = result['value'].mean()
                std_val = result['value'].std()
                if std_val > 0:
                    result['value_normalized'] = (result['value'] - mean_val) / std_val
            
            # Add derived features
            result['value_squared'] = result['value'] ** 2
            result['value_abs'] = result['value'].abs()
            
            return result
        
        # Set up processor
        config = ProcessingConfig(
            max_workers=4,
            min_workers=2,
            chunk_size=2000
        )
        processor = DistributedDataProcessor(config)
        
        try:
            # Start processing
            processor.start()
            
            # Process data
            task_ids = processor.process_dataframe(
                df=df,
                transformation_func=complex_transformation,
                chunk_size=2000
            )
            
            # Wait for results
            results = processor.get_results(task_ids, timeout=30.0)
            
            # Verify results
            assert len(results) == 5  # 10000 / 2000 = 5 chunks
            
            # Combine and verify
            combined_df = processor.combine_chunk_results(results)
            assert len(combined_df) == 10000
            assert 'value_normalized' in combined_df.columns
            assert 'value_squared' in combined_df.columns
            assert 'value_abs' in combined_df.columns
            
            # Check processing stats
            stats = processor.get_processing_stats()
            assert stats['completed_tasks'] == 5
            assert stats['failed_tasks'] == 0
            assert stats['total_processing_time'] > 0
            
        finally:
            processor.stop()


if __name__ == "__main__":
    pytest.main([__file__])