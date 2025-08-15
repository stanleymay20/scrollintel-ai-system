"""
Performance benchmark tests for AI Data Readiness Platform.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import gc
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile

from ai_data_readiness.core.data_ingestion_service import DataIngestionService
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def performance_components(self, test_config):
        """Set up components for performance testing."""
        return {
            'ingestion': DataIngestionService(test_config),
            'quality': QualityAssessmentEngine(test_config),
            'bias': BiasAnalysisEngine(test_config),
            'features': FeatureEngineeringEngine(test_config)
        }
    
    def test_data_ingestion_performance(self, performance_components, temp_directory):
        """Test data ingestion performance with various dataset sizes."""
        sizes = [1000, 10000, 50000]  # Reduced for CI/CD
        results = {}
        
        for size in sizes:
            # Generate dataset
            data = self._generate_performance_dataset(size, 20)
            csv_file = temp_directory / f"perf_data_{size}.csv"
            data.to_csv(csv_file, index=False)
            
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            # Measure ingestion time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = performance_components['ingestion'].ingest_batch_data(source_config)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            ingestion_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            results[size] = {
                'time': ingestion_time,
                'memory': memory_used,
                'rows_per_second': size / ingestion_time if ingestion_time > 0 else float('inf'),
                'dataset_id': result['dataset_id']
            }
            
            # Performance assertions
            assert ingestion_time < size * 0.001  # Max 1ms per row
            assert memory_used < size * 0.01  # Max 10KB per row
            
            # Cleanup
            gc.collect()
        
        # Test scalability
        if len(results) >= 2:
            sizes_list = sorted(results.keys())
            small_size = sizes_list[0]
            large_size = sizes_list[-1]
            
            # Time should scale sub-linearly (better than O(n))
            time_ratio = results[large_size]['time'] / results[small_size]['time']
            size_ratio = large_size / small_size
            
            assert time_ratio < size_ratio * 1.5  # Allow 50% overhead for larger datasets
    
    def test_quality_assessment_performance(self, performance_components, temp_directory):
        """Test quality assessment performance."""
        sizes = [1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            # Generate dataset with quality issues
            data = self._generate_quality_test_dataset(size)
            csv_file = temp_directory / f"quality_perf_{size}.csv"
            data.to_csv(csv_file, index=False)
            
            # Ingest data first
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            ingestion_result = performance_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            # Measure quality assessment time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            quality_report = performance_components['quality'].assess_quality(dataset_id)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            assessment_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            results[size] = {
                'time': assessment_time,
                'memory': memory_used,
                'overall_score': quality_report['overall_score']
            }
            
            # Performance assertions
            assert assessment_time < 30.0  # Max 30 seconds
            assert memory_used < 500  # Max 500MB additional memory
            
            gc.collect()
        
        # Verify quality assessment accuracy is maintained
        for size, result in results.items():
            assert 0 <= result['overall_score'] <= 1
    
    def test_bias_analysis_performance(self, performance_components, temp_directory):
        """Test bias analysis performance."""
        sizes = [1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            # Generate biased dataset
            data = self._generate_biased_dataset(size)
            csv_file = temp_directory / f"bias_perf_{size}.csv"
            data.to_csv(csv_file, index=False)
            
            # Ingest data
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            ingestion_result = performance_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            # Measure bias analysis time
            start_time = time.time()
            
            bias_report = performance_components['bias'].detect_bias(
                dataset_id, ['gender', 'age_group']
            )
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            results[size] = {
                'time': analysis_time,
                'bias_detected': bias_report['bias_detected'],
                'bias_score': bias_report['bias_score']
            }
            
            # Performance assertions
            assert analysis_time < 60.0  # Max 1 minute
            
            gc.collect()
        
        # Verify bias detection accuracy
        for result in results.values():
            assert result['bias_detected'] is True  # Should detect bias in generated data
            assert 0 <= result['bias_score'] <= 1
    
    def test_feature_engineering_performance(self, performance_components, temp_directory):
        """Test feature engineering performance."""
        sizes = [1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            # Generate dataset for feature engineering
            data = self._generate_feature_dataset(size)
            csv_file = temp_directory / f"feature_perf_{size}.csv"
            data.to_csv(csv_file, index=False)
            
            # Ingest data
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            ingestion_result = performance_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            # Measure feature engineering time
            start_time = time.time()
            
            recommendations = performance_components['features'].recommend_features(
                dataset_id, 'classification'
            )
            
            end_time = time.time()
            engineering_time = end_time - start_time
            
            results[size] = {
                'time': engineering_time,
                'num_recommendations': len(recommendations['recommended_features']),
                'num_transformations': len(recommendations['transformations'])
            }
            
            # Performance assertions
            assert engineering_time < 45.0  # Max 45 seconds
            assert results[size]['num_recommendations'] > 0
            
            gc.collect()
    
    def test_concurrent_processing_performance(self, performance_components, temp_directory):
        """Test concurrent processing performance."""
        num_datasets = 5
        dataset_size = 2000
        
        # Generate multiple datasets
        datasets = []
        for i in range(num_datasets):
            data = self._generate_performance_dataset(dataset_size, 10)
            csv_file = temp_directory / f"concurrent_{i}.csv"
            data.to_csv(csv_file, index=False)
            datasets.append(csv_file)
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = []
        
        for csv_file in datasets:
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            result = performance_components['ingestion'].ingest_batch_data(source_config)
            sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        
        def process_dataset(csv_file):
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            return performance_components['ingestion'].ingest_batch_data(source_config)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            concurrent_results = list(executor.map(process_dataset, datasets))
        
        concurrent_time = time.time() - start_time
        
        # Concurrent processing should be faster
        assert concurrent_time < sequential_time
        assert len(concurrent_results) == num_datasets
        
        # Results should be equivalent
        for seq_result, conc_result in zip(sequential_results, concurrent_results):
            assert seq_result['rows'] == conc_result['rows']
            assert seq_result['columns'] == conc_result['columns']
    
    def test_memory_usage_optimization(self, performance_components, temp_directory):
        """Test memory usage optimization."""
        # Generate large dataset
        large_data = self._generate_performance_dataset(20000, 50)
        csv_file = temp_directory / "memory_test.csv"
        large_data.to_csv(csv_file, index=False)
        
        # Monitor memory usage during processing
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'batch_size': 1000  # Process in batches
        }
        
        # Process with memory monitoring
        memory_samples = []
        
        def memory_monitor():
            while True:
                memory_samples.append(psutil.Process().memory_info().rss / 1024 / 1024)
                time.sleep(0.1)
        
        import threading
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        
        result = performance_components['ingestion'].ingest_batch_data(source_config)
        
        time.sleep(1)  # Let monitor collect final samples
        
        max_memory = max(memory_samples) if memory_samples else initial_memory
        memory_increase = max_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 1000  # Max 1GB increase
        assert result['rows'] == 20000
        
        # Force garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_cleanup = max_memory - final_memory
        
        # Should clean up most memory
        assert memory_cleanup > memory_increase * 0.5
    
    def test_cpu_utilization_efficiency(self, performance_components, temp_directory):
        """Test CPU utilization efficiency."""
        # Generate CPU-intensive dataset
        data = self._generate_complex_dataset(5000)
        csv_file = temp_directory / "cpu_test.csv"
        data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        # Monitor CPU usage
        cpu_samples = []
        
        def cpu_monitor():
            while True:
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        import threading
        monitor_thread = threading.Thread(target=cpu_monitor, daemon=True)
        monitor_thread.start()
        
        start_time = time.time()
        
        # Ingest and analyze
        ingestion_result = performance_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        quality_report = performance_components['quality'].assess_quality(dataset_id)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        time.sleep(1)  # Let monitor collect final samples
        
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0
        
        # CPU utilization should be reasonable
        assert avg_cpu < 80  # Average CPU usage under 80%
        assert max_cpu < 95   # Peak CPU usage under 95%
        assert processing_time < 30  # Complete within 30 seconds
    
    def test_disk_io_performance(self, performance_components, temp_directory):
        """Test disk I/O performance."""
        # Generate multiple files of different sizes
        file_sizes = [1000, 5000, 10000]
        io_results = {}
        
        for size in file_sizes:
            data = self._generate_performance_dataset(size, 20)
            csv_file = temp_directory / f"io_test_{size}.csv"
            
            # Measure write time
            write_start = time.time()
            data.to_csv(csv_file, index=False)
            write_time = time.time() - write_start
            
            # Measure read time
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            read_start = time.time()
            result = performance_components['ingestion'].ingest_batch_data(source_config)
            read_time = time.time() - read_start
            
            file_size_mb = csv_file.stat().st_size / 1024 / 1024
            
            io_results[size] = {
                'write_time': write_time,
                'read_time': read_time,
                'file_size_mb': file_size_mb,
                'write_speed_mbps': file_size_mb / write_time if write_time > 0 else float('inf'),
                'read_speed_mbps': file_size_mb / read_time if read_time > 0 else float('inf')
            }
            
            # Performance assertions
            assert io_results[size]['write_speed_mbps'] > 10  # Min 10 MB/s write
            assert io_results[size]['read_speed_mbps'] > 20   # Min 20 MB/s read
    
    def test_cache_performance(self, performance_components, temp_directory):
        """Test caching performance improvements."""
        # Generate dataset
        data = self._generate_performance_dataset(5000, 15)
        csv_file = temp_directory / "cache_test.csv"
        data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        # First run (cold cache)
        start_time = time.time()
        result1 = performance_components['ingestion'].ingest_batch_data(source_config)
        first_run_time = time.time() - start_time
        
        dataset_id = result1['dataset_id']
        
        # First quality assessment (cold cache)
        start_time = time.time()
        quality1 = performance_components['quality'].assess_quality(dataset_id)
        first_quality_time = time.time() - start_time
        
        # Second quality assessment (warm cache)
        start_time = time.time()
        quality2 = performance_components['quality'].assess_quality(dataset_id)
        second_quality_time = time.time() - start_time
        
        # Cached run should be faster
        assert second_quality_time < first_quality_time * 0.8  # At least 20% faster
        
        # Results should be identical
        assert quality1['overall_score'] == quality2['overall_score']
    
    # Helper methods for generating test datasets
    
    def _generate_performance_dataset(self, n_rows, n_features):
        """Generate dataset for performance testing."""
        np.random.seed(42)
        
        data = {}
        for i in range(n_features):
            if i % 3 == 0:
                data[f'num_{i}'] = np.random.normal(0, 1, n_rows)
            elif i % 3 == 1:
                data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
            else:
                data[f'bool_{i}'] = np.random.choice([True, False], n_rows)
        
        return pd.DataFrame(data)
    
    def _generate_quality_test_dataset(self, n_rows):
        """Generate dataset with quality issues."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'complete_col': range(n_rows),
            'missing_col': [i if i % 10 != 0 else None for i in range(n_rows)],
            'outlier_col': np.concatenate([
                np.random.normal(50, 10, n_rows - 10),
                [1000, -1000] * 5  # Outliers
            ]),
            'inconsistent_col': [
                f'Value_{i}' if i % 100 != 0 else f'value_{i}' 
                for i in range(n_rows)
            ]
        })
        
        return data
    
    def _generate_biased_dataset(self, n_rows):
        """Generate dataset with bias."""
        np.random.seed(42)
        
        gender = np.random.choice(['M', 'F'], n_rows)
        age = np.random.randint(18, 80, n_rows)
        age_group = np.where(age < 30, 'Young', np.where(age < 50, 'Middle', 'Senior'))
        
        # Introduce bias
        approval_prob = np.where(
            gender == 'M', 0.7,
            np.where(age_group == 'Young', 0.4, 0.5)
        )
        
        approved = np.random.binomial(1, approval_prob)
        
        return pd.DataFrame({
            'gender': gender,
            'age': age,
            'age_group': age_group,
            'approved': approved
        })
    
    def _generate_feature_dataset(self, n_rows):
        """Generate dataset for feature engineering."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'numerical_1': np.random.normal(0, 1, n_rows),
            'numerical_2': np.random.exponential(2, n_rows),
            'categorical_1': np.random.choice(['A', 'B', 'C'], n_rows),
            'categorical_2': np.random.choice([f'Cat_{i}' for i in range(10)], n_rows),
            'date_col': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
            'target': np.random.choice([0, 1], n_rows)
        })
    
    def _generate_complex_dataset(self, n_rows):
        """Generate computationally complex dataset."""
        np.random.seed(42)
        
        # Create correlated features
        base = np.random.normal(0, 1, n_rows)
        
        return pd.DataFrame({
            'feature_1': base,
            'feature_2': base + np.random.normal(0, 0.1, n_rows),
            'feature_3': base ** 2 + np.random.normal(0, 0.5, n_rows),
            'feature_4': np.sin(base) + np.random.normal(0, 0.2, n_rows),
            'feature_5': np.exp(base / 2) + np.random.normal(0, 1, n_rows),
            'categorical': np.random.choice([f'Complex_{i}' for i in range(50)], n_rows),
            'target': (base > 0).astype(int)
        })