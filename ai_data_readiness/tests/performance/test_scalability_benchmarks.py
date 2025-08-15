"""
Performance and scalability benchmarks for AI Data Readiness Platform.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from ai_data_readiness.core.data_ingestion_service import DataIngestionService
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
from ai_data_readiness.engines.drift_monitor import DriftMonitor


class TestScalabilityBenchmarks:
    """Performance and scalability benchmark tests."""
    
    @pytest.fixture
    def benchmark_components(self, test_config):
        """Set up components for benchmarking."""
        return {
            'ingestion': DataIngestionService(test_config),
            'quality': QualityAssessmentEngine(test_config),
            'bias': BiasAnalysisEngine(test_config),
            'features': FeatureEngineeringEngine(test_config),
            'drift': DriftMonitor(test_config)
        }
    
    @pytest.fixture
    def performance_thresholds(self):
        """Define performance thresholds for different operations."""
        return {
            'ingestion': {
                'small_dataset': {'max_time': 2.0, 'min_throughput': 1000},  # rows/sec
                'medium_dataset': {'max_time': 10.0, 'min_throughput': 2000},
                'large_dataset': {'max_time': 60.0, 'min_throughput': 5000}
            },
            'quality_assessment': {
                'small_dataset': {'max_time': 5.0},
                'medium_dataset': {'max_time': 20.0},
                'large_dataset': {'max_time': 120.0}
            },
            'bias_analysis': {
                'small_dataset': {'max_time': 3.0},
                'medium_dataset': {'max_time': 15.0},
                'large_dataset': {'max_time': 90.0}
            },
            'feature_engineering': {
                'small_dataset': {'max_time': 8.0},
                'medium_dataset': {'max_time': 30.0},
                'large_dataset': {'max_time': 180.0}
            }
        }
    
    def generate_benchmark_dataset(self, n_rows, n_features, complexity='medium'):
        """Generate benchmark dataset with specified characteristics."""
        np.random.seed(42)
        
        data = {}
        
        # Generate features based on complexity
        if complexity == 'simple':
            # Simple numerical and categorical features
            for i in range(n_features):
                if i % 2 == 0:
                    data[f'num_feature_{i}'] = np.random.normal(0, 1, n_rows)
                else:
                    data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], n_rows)
        
        elif complexity == 'medium':
            # Mixed data types with some correlations
            for i in range(n_features):
                if i % 4 == 0:
                    data[f'num_feature_{i}'] = np.random.normal(0, 1, n_rows)
                elif i % 4 == 1:
                    data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows)
                elif i % 4 == 2:
                    data[f'bool_feature_{i}'] = np.random.choice([True, False], n_rows)
                else:
                    # Correlated feature
                    base_feature = f'num_feature_{i-3}' if f'num_feature_{i-3}' in data else f'num_feature_0'
                    if base_feature in data:
                        data[f'corr_feature_{i}'] = data[base_feature] + np.random.normal(0, 0.5, n_rows)
                    else:
                        data[f'corr_feature_{i}'] = np.random.normal(0, 1, n_rows)
        
        elif complexity == 'complex':
            # Complex data with multiple data types, missing values, and relationships
            for i in range(n_features):
                if i % 6 == 0:
                    # Numerical with outliers
                    feature = np.random.normal(0, 1, n_rows)
                    outlier_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
                    feature[outlier_indices] = np.random.uniform(-10, 10, len(outlier_indices))
                    data[f'num_outlier_feature_{i}'] = feature
                elif i % 6 == 1:
                    # High cardinality categorical
                    data[f'high_card_cat_{i}'] = np.random.choice([f'cat_{j}' for j in range(50)], n_rows)
                elif i % 6 == 2:
                    # Time series feature
                    data[f'time_feature_{i}'] = pd.date_range('2020-01-01', periods=n_rows, freq='H')
                elif i % 6 == 3:
                    # Text-like feature
                    data[f'text_feature_{i}'] = [f'text_value_{np.random.randint(0, 1000)}' for _ in range(n_rows)]
                elif i % 6 == 4:
                    # Feature with missing values
                    feature = np.random.normal(0, 1, n_rows)
                    missing_indices = np.random.choice(n_rows, size=int(n_rows * 0.1), replace=False)
                    feature = feature.astype(object)
                    feature[missing_indices] = None
                    data[f'missing_feature_{i}'] = feature
                else:
                    # Skewed numerical feature
                    data[f'skewed_feature_{i}'] = np.random.exponential(2, n_rows)
        
        # Add target variable
        data['target'] = np.random.choice([0, 1], n_rows)
        
        return pd.DataFrame(data)
    
    @pytest.mark.parametrize("dataset_size,expected_category", [
        (1000, 'small_dataset'),
        (10000, 'medium_dataset'),
        (50000, 'large_dataset')
    ])
    def test_ingestion_performance(self, benchmark_components, performance_thresholds, 
                                 dataset_size, expected_category, temp_directory):
        """Test data ingestion performance across different dataset sizes."""
        # Generate test dataset
        n_features = min(20, max(5, dataset_size // 1000))  # Scale features with size
        test_data = self.generate_benchmark_dataset(dataset_size, n_features, 'medium')
        
        # Save to file
        csv_file = temp_directory / f"benchmark_ingestion_{dataset_size}.csv"
        test_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        # Measure ingestion performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = benchmark_components['ingestion'].ingest_batch_data(source_config)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        ingestion_time = end_time - start_time
        memory_used = end_memory - start_memory
        throughput = dataset_size / ingestion_time if ingestion_time > 0 else float('inf')
        
        # Verify results
        assert result['rows'] == dataset_size
        assert result['columns'] == len(test_data.columns)
        
        # Check performance thresholds
        thresholds = performance_thresholds['ingestion'][expected_category]
        assert ingestion_time <= thresholds['max_time'], f"Ingestion took {ingestion_time:.2f}s, expected <= {thresholds['max_time']}s"
        assert throughput >= thresholds['min_throughput'], f"Throughput {throughput:.0f} rows/s, expected >= {thresholds['min_throughput']} rows/s"
        
        # Log performance metrics
        print(f"\nIngestion Performance ({dataset_size} rows):")
        print(f"  Time: {ingestion_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} rows/s")
        print(f"  Memory used: {memory_used:.1f} MB")
    
    @pytest.mark.parametrize("dataset_size,expected_category", [
        (1000, 'small_dataset'),
        (10000, 'medium_dataset'),
        (25000, 'large_dataset')  # Reduced for unit tests
    ])
    def test_quality_assessment_performance(self, benchmark_components, performance_thresholds,
                                          dataset_size, expected_category, temp_directory):
        """Test quality assessment performance across different dataset sizes."""
        # Generate test dataset with quality issues
        test_data = self.generate_benchmark_dataset(dataset_size, 15, 'complex')
        
        # Save and ingest
        csv_file = temp_directory / f"benchmark_quality_{dataset_size}.csv"
        test_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        ingestion_result = benchmark_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        # Measure quality assessment performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        quality_report = benchmark_components['quality'].assess_quality(dataset_id)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        assessment_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Verify results
        assert 'overall_score' in quality_report
        assert 0 <= quality_report['overall_score'] <= 1
        
        # Check performance thresholds
        thresholds = performance_thresholds['quality_assessment'][expected_category]
        assert assessment_time <= thresholds['max_time'], f"Quality assessment took {assessment_time:.2f}s, expected <= {thresholds['max_time']}s"
        
        # Log performance metrics
        print(f"\nQuality Assessment Performance ({dataset_size} rows):")
        print(f"  Time: {assessment_time:.2f}s")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Quality score: {quality_report['overall_score']:.3f}")
    
    @pytest.mark.parametrize("dataset_size,expected_category", [
        (1000, 'small_dataset'),
        (5000, 'medium_dataset'),
        (15000, 'large_dataset')  # Reduced for unit tests
    ])
    def test_bias_analysis_performance(self, benchmark_components, performance_thresholds,
                                     dataset_size, expected_category, temp_directory):
        """Test bias analysis performance across different dataset sizes."""
        # Generate biased dataset
        np.random.seed(42)
        
        gender = np.random.choice(['M', 'F'], dataset_size)
        age = np.random.randint(18, 80, dataset_size)
        income = np.random.normal(50000, 15000, dataset_size)
        
        # Introduce bias
        approval_prob = np.where(gender == 'M', 0.7, 0.5)
        approved = np.random.binomial(1, approval_prob)
        
        test_data = pd.DataFrame({
            'gender': gender,
            'age': age,
            'income': income,
            'approved': approved
        })
        
        # Add more features to increase complexity
        for i in range(10):
            test_data[f'feature_{i}'] = np.random.normal(0, 1, dataset_size)
        
        # Save and ingest
        csv_file = temp_directory / f"benchmark_bias_{dataset_size}.csv"
        test_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        ingestion_result = benchmark_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        # Measure bias analysis performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        bias_report = benchmark_components['bias'].detect_bias(dataset_id, ['gender'])
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        analysis_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Verify results
        assert 'bias_detected' in bias_report
        assert 'bias_score' in bias_report
        
        # Check performance thresholds
        thresholds = performance_thresholds['bias_analysis'][expected_category]
        assert analysis_time <= thresholds['max_time'], f"Bias analysis took {analysis_time:.2f}s, expected <= {thresholds['max_time']}s"
        
        # Log performance metrics
        print(f"\nBias Analysis Performance ({dataset_size} rows):")
        print(f"  Time: {analysis_time:.2f}s")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Bias detected: {bias_report['bias_detected']}")
        print(f"  Bias score: {bias_report['bias_score']:.3f}")
    
    def test_concurrent_processing_performance(self, benchmark_components, temp_directory):
        """Test performance under concurrent processing load."""
        # Create multiple datasets for concurrent processing
        datasets = []
        dataset_size = 2000
        
        for i in range(5):
            test_data = self.generate_benchmark_dataset(dataset_size, 10, 'medium')
            csv_file = temp_directory / f"concurrent_test_{i}.csv"
            test_data.to_csv(csv_file, index=False)
            datasets.append(csv_file)
        
        def process_dataset_sequential(file_path):
            """Process a single dataset sequentially."""
            source_config = {
                'source_type': 'file',
                'file_path': str(file_path),
                'format': 'csv'
            }
            
            start_time = time.time()
            
            # Ingestion
            ingestion_result = benchmark_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            # Quality assessment
            quality_report = benchmark_components['quality'].assess_quality(dataset_id)
            
            end_time = time.time()
            
            return {
                'file_path': str(file_path),
                'processing_time': end_time - start_time,
                'quality_score': quality_report['overall_score']
            }
        
        # Sequential processing
        sequential_start = time.time()
        sequential_results = []
        
        for dataset in datasets:
            result = process_dataset_sequential(dataset)
            sequential_results.append(result)
        
        sequential_total_time = time.time() - sequential_start
        
        # Concurrent processing
        concurrent_start = time.time()
        concurrent_results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_dataset = {executor.submit(process_dataset_sequential, dataset): dataset 
                               for dataset in datasets}
            
            for future in future_to_dataset:
                result = future.result()
                concurrent_results.append(result)
        
        concurrent_total_time = time.time() - concurrent_start
        
        # Verify results
        assert len(sequential_results) == len(datasets)
        assert len(concurrent_results) == len(datasets)
        
        # Concurrent processing should be faster
        speedup = sequential_total_time / concurrent_total_time
        assert speedup > 1.2, f"Expected speedup > 1.2x, got {speedup:.2f}x"
        
        # Log performance metrics
        print(f"\nConcurrent Processing Performance:")
        print(f"  Sequential time: {sequential_total_time:.2f}s")
        print(f"  Concurrent time: {concurrent_total_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    def test_memory_usage_scaling(self, benchmark_components, temp_directory):
        """Test memory usage scaling with dataset size."""
        dataset_sizes = [1000, 5000, 10000]
        memory_usage = []
        
        for size in dataset_sizes:
            # Generate dataset
            test_data = self.generate_benchmark_dataset(size, 15, 'medium')
            csv_file = temp_directory / f"memory_test_{size}.csv"
            test_data.to_csv(csv_file, index=False)
            
            # Measure memory before processing
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Process dataset
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            ingestion_result = benchmark_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            quality_report = benchmark_components['quality'].assess_quality(dataset_id)
            
            # Measure memory after processing
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            
            memory_usage.append({
                'dataset_size': size,
                'memory_used_mb': memory_used,
                'memory_per_row_kb': (memory_used * 1024) / size
            })
            
            # Clean up to free memory
            del test_data
            import gc
            gc.collect()
        
        # Verify memory usage is reasonable and scales appropriately
        for i, usage in enumerate(memory_usage):
            print(f"\nMemory Usage ({usage['dataset_size']} rows):")
            print(f"  Total memory used: {usage['memory_used_mb']:.1f} MB")
            print(f"  Memory per row: {usage['memory_per_row_kb']:.2f} KB")
            
            # Memory per row should be reasonable (< 10 KB per row)
            assert usage['memory_per_row_kb'] < 10, f"Memory per row too high: {usage['memory_per_row_kb']:.2f} KB"
            
            # Memory usage should scale sub-linearly (due to fixed overhead)
            if i > 0:
                prev_usage = memory_usage[i-1]
                size_ratio = usage['dataset_size'] / prev_usage['dataset_size']
                memory_ratio = usage['memory_used_mb'] / prev_usage['memory_used_mb']
                
                # Memory should scale less than linearly
                assert memory_ratio < size_ratio * 1.2, f"Memory scaling too high: {memory_ratio:.2f}x vs {size_ratio:.2f}x size increase"
    
    def test_cpu_utilization_efficiency(self, benchmark_components, temp_directory):
        """Test CPU utilization efficiency during processing."""
        # Generate medium-sized dataset
        test_data = self.generate_benchmark_dataset(10000, 20, 'complex')
        csv_file = temp_directory / "cpu_test.csv"
        test_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        # Monitor CPU usage during processing
        cpu_usage = []
        
        def monitor_cpu():
            while not stop_monitoring:
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        stop_monitoring = False
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            # Process dataset
            start_time = time.time()
            
            ingestion_result = benchmark_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            quality_report = benchmark_components['quality'].assess_quality(dataset_id)
            bias_report = benchmark_components['bias'].detect_bias(dataset_id, ['target'])
            
            processing_time = time.time() - start_time
            
        finally:
            stop_monitoring = True
            monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_usage:
            avg_cpu = np.mean(cpu_usage)
            max_cpu = np.max(cpu_usage)
            
            print(f"\nCPU Utilization:")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Average CPU usage: {avg_cpu:.1f}%")
            print(f"  Peak CPU usage: {max_cpu:.1f}%")
            
            # CPU usage should be reasonable (not too low, indicating inefficiency)
            assert avg_cpu > 10, f"CPU usage too low: {avg_cpu:.1f}% (possible inefficiency)"
            
            # CPU usage should not be constantly at 100% (indicating bottleneck)
            high_cpu_samples = sum(1 for usage in cpu_usage if usage > 95)
            high_cpu_ratio = high_cpu_samples / len(cpu_usage)
            assert high_cpu_ratio < 0.8, f"CPU usage too high for too long: {high_cpu_ratio:.1%} of time > 95%"
    
    def test_drift_monitoring_performance(self, benchmark_components, temp_directory):
        """Test drift monitoring performance with large datasets."""
        dataset_size = 10000
        
        # Create reference dataset
        reference_data = self.generate_benchmark_dataset(dataset_size, 15, 'medium')
        reference_file = temp_directory / "drift_reference.csv"
        reference_data.to_csv(reference_file, index=False)
        
        # Create current dataset with drift
        current_data = reference_data.copy()
        # Introduce drift in numerical features
        for col in current_data.select_dtypes(include=[np.number]).columns:
            if col != 'target':
                current_data[col] = current_data[col] + np.random.normal(0.5, 0.2, len(current_data))
        
        current_file = temp_directory / "drift_current.csv"
        current_data.to_csv(current_file, index=False)
        
        # Ingest both datasets
        reference_config = {
            'source_type': 'file',
            'file_path': str(reference_file),
            'format': 'csv'
        }
        
        current_config = {
            'source_type': 'file',
            'file_path': str(current_file),
            'format': 'csv'
        }
        
        reference_result = benchmark_components['ingestion'].ingest_batch_data(reference_config)
        current_result = benchmark_components['ingestion'].ingest_batch_data(current_config)
        
        reference_dataset_id = reference_result['dataset_id']
        current_dataset_id = current_result['dataset_id']
        
        # Measure drift monitoring performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        drift_report = benchmark_components['drift'].monitor_drift(
            current_dataset_id, reference_dataset_id
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        drift_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Verify results
        assert 'drift_detected' in drift_report
        assert 'drift_score' in drift_report
        assert 'feature_drift_scores' in drift_report
        
        # Performance should be reasonable
        assert drift_time < 30.0, f"Drift monitoring took {drift_time:.2f}s, expected < 30s"
        
        # Should detect drift
        assert drift_report['drift_detected'] is True
        assert drift_report['drift_score'] > 0.1
        
        print(f"\nDrift Monitoring Performance ({dataset_size} rows):")
        print(f"  Time: {drift_time:.2f}s")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Drift detected: {drift_report['drift_detected']}")
        print(f"  Drift score: {drift_report['drift_score']:.3f}")
    
    def test_batch_processing_efficiency(self, benchmark_components, temp_directory):
        """Test efficiency of batch processing vs single large processing."""
        total_rows = 20000
        batch_sizes = [1000, 5000, 10000, 20000]  # Last one is single batch
        
        results = {}
        
        for batch_size in batch_sizes:
            # Generate dataset
            test_data = self.generate_benchmark_dataset(total_rows, 10, 'medium')
            csv_file = temp_directory / f"batch_test_{batch_size}.csv"
            test_data.to_csv(csv_file, index=False)
            
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv',
                'batch_size': batch_size
            }
            
            # Measure processing time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            ingestion_result = benchmark_components['ingestion'].ingest_batch_data(source_config)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            processing_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            results[batch_size] = {
                'processing_time': processing_time,
                'memory_used': memory_used,
                'throughput': total_rows / processing_time
            }
            
            print(f"\nBatch Processing (batch_size={batch_size}):")
            print(f"  Time: {processing_time:.2f}s")
            print(f"  Memory: {memory_used:.1f} MB")
            print(f"  Throughput: {results[batch_size]['throughput']:.0f} rows/s")
        
        # Analyze batch processing efficiency
        # Smaller batches should use less memory but may have lower throughput
        assert results[1000]['memory_used'] <= results[20000]['memory_used'] * 1.5
        
        # Throughput should be reasonable across all batch sizes
        for batch_size, result in results.items():
            assert result['throughput'] > 500, f"Throughput too low for batch_size {batch_size}: {result['throughput']:.0f} rows/s"