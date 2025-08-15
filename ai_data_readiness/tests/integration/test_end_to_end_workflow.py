"""
End-to-end integration tests for AI Data Readiness Platform workflows.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ai_data_readiness.core.data_ingestion_service import DataIngestionService
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
from ai_data_readiness.engines.drift_monitor import DriftMonitor
from ai_data_readiness.engines.ai_readiness_reporting_engine import AIReadinessReportingEngine


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def workflow_components(self, test_config, test_session):
        """Set up all workflow components."""
        return {
            'ingestion': DataIngestionService(test_config),
            'quality': QualityAssessmentEngine(test_config),
            'bias': BiasAnalysisEngine(test_config),
            'features': FeatureEngineeringEngine(test_config),
            'drift': DriftMonitor(test_config),
            'reporting': AIReadinessReportingEngine(test_config)
        }
    
    def test_complete_data_preparation_workflow(self, workflow_components, sample_csv_data, temp_directory):
        """Test complete data preparation workflow from ingestion to AI readiness."""
        # Step 1: Data Ingestion
        csv_file = temp_directory / "workflow_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        ingestion_result = workflow_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        assert dataset_id is not None
        assert ingestion_result['rows'] == len(sample_csv_data)
        
        # Step 2: Quality Assessment
        quality_report = workflow_components['quality'].assess_quality(dataset_id)
        
        assert quality_report['dataset_id'] == dataset_id
        assert 'overall_score' in quality_report
        assert quality_report['overall_score'] > 0
        
        # Step 3: Bias Analysis (if applicable)
        if 'gender' in sample_csv_data.columns:
            bias_report = workflow_components['bias'].detect_bias(dataset_id, ['gender'])
            assert bias_report['dataset_id'] == dataset_id
            assert 'bias_detected' in bias_report
        
        # Step 4: Feature Engineering Recommendations
        feature_recommendations = workflow_components['features'].recommend_features(
            dataset_id, 'classification'
        )
        
        assert 'recommended_features' in feature_recommendations
        assert 'transformations' in feature_recommendations
        
        # Step 5: Apply Feature Transformations
        if feature_recommendations['transformations']:
            transformation_result = workflow_components['features'].apply_transformations(
                dataset_id, feature_recommendations['transformations'][:2]  # Apply first 2
            )
            
            assert 'transformed_dataset_id' in transformation_result
            transformed_dataset_id = transformation_result['transformed_dataset_id']
        else:
            transformed_dataset_id = dataset_id
        
        # Step 6: Final AI Readiness Assessment
        ai_readiness = workflow_components['reporting'].generate_ai_readiness_report(
            transformed_dataset_id
        )
        
        assert 'overall_ai_readiness_score' in ai_readiness
        assert 'dimension_scores' in ai_readiness
        assert 'recommendations' in ai_readiness
        assert 'improvement_roadmap' in ai_readiness
        
        # Verify workflow coherence
        assert 0 <= ai_readiness['overall_ai_readiness_score'] <= 1
        
        # Check that all major dimensions are covered
        dimensions = ai_readiness['dimension_scores']
        expected_dimensions = ['data_quality', 'feature_quality', 'bias_fairness', 'compliance']
        
        for dimension in expected_dimensions:
            assert dimension in dimensions
    
    def test_streaming_data_workflow(self, workflow_components):
        """Test streaming data processing workflow."""
        # Mock streaming data setup
        stream_config = {
            'source_type': 'kafka',
            'topic': 'test_stream',
            'bootstrap_servers': 'localhost:9092'
        }
        
        # This would normally set up real streaming, but we'll mock it
        with pytest.raises(Exception):  # Expected to fail without real Kafka
            workflow_components['ingestion'].ingest_streaming_data(stream_config)
    
    def test_batch_processing_large_dataset_workflow(self, workflow_components, temp_directory):
        """Test batch processing workflow with large dataset."""
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'feature1': np.random.normal(0, 1, 10000),
            'feature2': np.random.exponential(2, 10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000),
            'target': np.random.choice([0, 1], 10000)
        })
        
        csv_file = temp_directory / "large_workflow_data.csv"
        large_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'batch_size': 1000
        }
        
        # Test batch ingestion
        ingestion_result = workflow_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        assert ingestion_result['rows'] == 10000
        assert 'processing_batches' in ingestion_result
        
        # Test quality assessment on large dataset
        quality_report = workflow_components['quality'].assess_quality(dataset_id)
        assert quality_report['overall_score'] > 0
        
        # Test feature recommendations
        feature_recs = workflow_components['features'].recommend_features(dataset_id, 'classification')
        assert len(feature_recs['recommended_features']) > 0
    
    def test_data_drift_monitoring_workflow(self, workflow_components, sample_csv_data, temp_directory):
        """Test data drift monitoring workflow."""
        # Create reference dataset
        reference_file = temp_directory / "reference_data.csv"
        sample_csv_data.to_csv(reference_file, index=False)
        
        reference_config = {
            'source_type': 'file',
            'file_path': str(reference_file),
            'format': 'csv'
        }
        
        reference_result = workflow_components['ingestion'].ingest_batch_data(reference_config)
        reference_dataset_id = reference_result['dataset_id']
        
        # Create current dataset with some drift
        drifted_data = sample_csv_data.copy()
        drifted_data['age'] = drifted_data['age'] + 10  # Age drift
        drifted_data['income'] = drifted_data['income'] * 1.2  # Income drift
        
        current_file = temp_directory / "current_data.csv"
        drifted_data.to_csv(current_file, index=False)
        
        current_config = {
            'source_type': 'file',
            'file_path': str(current_file),
            'format': 'csv'
        }
        
        current_result = workflow_components['ingestion'].ingest_batch_data(current_config)
        current_dataset_id = current_result['dataset_id']
        
        # Monitor drift
        drift_report = workflow_components['drift'].monitor_drift(
            current_dataset_id, reference_dataset_id
        )
        
        assert 'drift_detected' in drift_report
        assert 'drift_score' in drift_report
        assert 'feature_drift_scores' in drift_report
        
        # Should detect drift in age and income
        feature_drift = drift_report['feature_drift_scores']
        assert 'age' in feature_drift
        assert 'income' in feature_drift
        
        # Drift scores should be significant
        assert feature_drift['age'] > 0.1
        assert feature_drift['income'] > 0.1
    
    def test_model_performance_correlation_workflow(self, workflow_components, sample_csv_data, temp_directory):
        """Test correlation between data quality and model performance."""
        # Create datasets with different quality levels
        
        # High quality dataset
        high_quality_data = sample_csv_data.copy()
        
        # Low quality dataset (introduce issues)
        low_quality_data = sample_csv_data.copy()
        # Add missing values
        low_quality_data.loc[low_quality_data.index[:20], 'age'] = None
        low_quality_data.loc[low_quality_data.index[10:30], 'income'] = None
        # Add outliers
        low_quality_data.loc[low_quality_data.index[:5], 'income'] = 1000000
        # Add inconsistencies
        low_quality_data.loc[low_quality_data.index[:10], 'gender'] = 'Unknown'
        
        datasets = {
            'high_quality': high_quality_data,
            'low_quality': low_quality_data
        }
        
        results = {}
        
        for quality_level, data in datasets.items():
            # Ingest data
            csv_file = temp_directory / f"{quality_level}_data.csv"
            data.to_csv(csv_file, index=False)
            
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            ingestion_result = workflow_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            # Assess quality
            quality_report = workflow_components['quality'].assess_quality(dataset_id)
            
            # Get AI readiness score
            ai_readiness = workflow_components['reporting'].generate_ai_readiness_report(dataset_id)
            
            results[quality_level] = {
                'quality_score': quality_report['overall_score'],
                'ai_readiness_score': ai_readiness['overall_ai_readiness_score']
            }
        
        # High quality data should have better scores
        assert results['high_quality']['quality_score'] > results['low_quality']['quality_score']
        assert results['high_quality']['ai_readiness_score'] > results['low_quality']['ai_readiness_score']
    
    def test_compliance_validation_workflow(self, workflow_components, temp_directory):
        """Test compliance validation workflow."""
        # Create dataset with PII and sensitive information
        sensitive_data = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@email.com', 'jane@email.com', 'bob@email.com'],
            'ssn': ['123-45-6789', '987-65-4321', '555-44-3333'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000],
            'medical_condition': ['Diabetes', 'Hypertension', 'None']
        })
        
        csv_file = temp_directory / "sensitive_data.csv"
        sensitive_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        # Ingest data
        ingestion_result = workflow_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        # This would normally include compliance analysis
        # For now, we'll just verify the workflow structure
        quality_report = workflow_components['quality'].assess_quality(dataset_id)
        
        # Should identify potential compliance issues
        assert 'compliance_issues' in quality_report or 'privacy_concerns' in quality_report
    
    def test_automated_remediation_workflow(self, workflow_components, sample_missing_data, temp_directory):
        """Test automated data remediation workflow."""
        # Use data with missing values
        csv_file = temp_directory / "missing_data.csv"
        sample_missing_data.to_csv(csv_file, index=False)
        
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        # Ingest data
        ingestion_result = workflow_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        # Assess quality
        quality_report = workflow_components['quality'].assess_quality(dataset_id)
        
        # Get improvement recommendations
        recommendations = workflow_components['quality'].recommend_improvements(quality_report)
        
        assert 'recommendations' in recommendations
        assert len(recommendations['recommendations']) > 0
        
        # Should recommend handling missing values
        missing_value_recs = [
            r for r in recommendations['recommendations'] 
            if 'missing' in r['description'].lower()
        ]
        assert len(missing_value_recs) > 0
    
    def test_multi_format_data_integration_workflow(self, workflow_components, sample_csv_data, temp_directory):
        """Test integration of multiple data formats."""
        # Create data in different formats
        csv_file = temp_directory / "data.csv"
        json_file = temp_directory / "data.json"
        parquet_file = temp_directory / "data.parquet"
        
        sample_csv_data.to_csv(csv_file, index=False)
        sample_csv_data.to_json(json_file, orient='records')
        sample_csv_data.to_parquet(parquet_file, index=False)
        
        formats = [
            {'file': csv_file, 'format': 'csv'},
            {'file': json_file, 'format': 'json'},
            {'file': parquet_file, 'format': 'parquet'}
        ]
        
        dataset_ids = []
        
        for fmt in formats:
            source_config = {
                'source_type': 'file',
                'file_path': str(fmt['file']),
                'format': fmt['format']
            }
            
            ingestion_result = workflow_components['ingestion'].ingest_batch_data(source_config)
            dataset_ids.append(ingestion_result['dataset_id'])
            
            # Verify consistent ingestion across formats
            assert ingestion_result['rows'] == len(sample_csv_data)
            assert ingestion_result['columns'] == len(sample_csv_data.columns)
        
        # All datasets should have similar quality scores
        quality_scores = []
        for dataset_id in dataset_ids:
            quality_report = workflow_components['quality'].assess_quality(dataset_id)
            quality_scores.append(quality_report['overall_score'])
        
        # Quality scores should be similar (within 0.1)
        max_score = max(quality_scores)
        min_score = min(quality_scores)
        assert max_score - min_score < 0.1
    
    def test_scalability_stress_test(self, workflow_components, temp_directory, scalability_test_config):
        """Test system scalability with increasing data sizes."""
        sizes = [
            scalability_test_config['small_dataset_size'],
            scalability_test_config['medium_dataset_size']
            # Skip large dataset in unit tests to avoid timeout
        ]
        
        performance_results = {}
        
        for size in sizes:
            # Generate dataset of specified size
            data = generate_synthetic_dataset(n_rows=size, n_features=10)
            
            csv_file = temp_directory / f"scalability_data_{size}.csv"
            data.to_csv(csv_file, index=False)
            
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            # Measure ingestion time
            import time
            start_time = time.time()
            
            ingestion_result = workflow_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            ingestion_time = time.time() - start_time
            
            # Measure quality assessment time
            start_time = time.time()
            quality_report = workflow_components['quality'].assess_quality(dataset_id)
            quality_time = time.time() - start_time
            
            performance_results[size] = {
                'ingestion_time': ingestion_time,
                'quality_time': quality_time,
                'ingestion_rate': size / ingestion_time if ingestion_time > 0 else float('inf')
            }
        
        # Verify performance scales reasonably
        small_size = scalability_test_config['small_dataset_size']
        medium_size = scalability_test_config['medium_dataset_size']
        
        if medium_size in performance_results and small_size in performance_results:
            # Ingestion rate should not degrade too much
            small_rate = performance_results[small_size]['ingestion_rate']
            medium_rate = performance_results[medium_size]['ingestion_rate']
            
            # Allow up to 50% degradation in rate for larger datasets
            assert medium_rate > small_rate * 0.5


def generate_synthetic_dataset(n_rows=1000, n_features=10, missing_rate=0.1, bias_factor=0.0):
    """Generate synthetic dataset for testing."""
    np.random.seed(42)
    
    data = {}
    for i in range(n_features):
        if i % 3 == 0:  # Numerical features
            data[f'num_feature_{i}'] = np.random.normal(0, 1, n_rows)
        elif i % 3 == 1:  # Categorical features
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
        else:  # Boolean features
            data[f'bool_feature_{i}'] = np.random.choice([True, False], n_rows)
    
    # Add target variable
    data['target'] = np.random.choice([0, 1], n_rows)
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    if missing_rate > 0:
        for col in df.columns:
            if col != 'target':
                missing_mask = np.random.random(n_rows) < missing_rate
                df.loc[missing_mask, col] = None
    
    return df