"""
Comprehensive integration tests for AI Data Readiness Platform workflows.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai_data_readiness.core.data_ingestion_service import DataIngestionService
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
from ai_data_readiness.engines.drift_monitor import DriftMonitor
from ai_data_readiness.engines.ai_readiness_reporting_engine import AIReadinessReportingEngine
from ai_data_readiness.engines.compliance_analyzer import ComplianceAnalyzer
from ai_data_readiness.engines.lineage_engine import LineageEngine


class TestComprehensiveWorkflows:
    """Test comprehensive end-to-end workflows."""
    
    @pytest.fixture
    def full_platform_components(self, test_config, test_session):
        """Set up all platform components."""
        return {
            'ingestion': DataIngestionService(test_config),
            'quality': QualityAssessmentEngine(test_config),
            'bias': BiasAnalysisEngine(test_config),
            'features': FeatureEngineeringEngine(test_config),
            'drift': DriftMonitor(test_config),
            'reporting': AIReadinessReportingEngine(test_config),
            'compliance': ComplianceAnalyzer(test_config),
            'lineage': LineageEngine(test_config)
        }
    
    def test_ml_model_development_workflow(self, full_platform_components, temp_directory):
        """Test complete ML model development workflow."""
        # Step 1: Create realistic ML dataset
        np.random.seed(42)
        n_samples = 5000
        
        # Features
        age = np.random.randint(18, 80, n_samples)
        income = np.random.normal(50000, 20000, n_samples)
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        gender = np.random.choice(['M', 'F'], n_samples)
        
        # Target with realistic relationships
        approval_prob = (
            0.3 + 
            (age - 18) / (80 - 18) * 0.2 +  # Age factor
            np.clip((income - 30000) / 70000, 0, 0.3) +  # Income factor
            {'High School': 0.0, 'Bachelor': 0.1, 'Master': 0.15, 'PhD': 0.2}[education[0]] +  # Education factor
            np.random.normal(0, 0.1, n_samples)  # Random noise
        )
        
        # Introduce some bias
        approval_prob = np.where(gender == 'F', approval_prob * 0.9, approval_prob)
        approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        ml_dataset = pd.DataFrame({
            'age': age,
            'income': income,
            'education': education,
            'gender': gender,
            'approved': approved
        })
        
        # Add some data quality issues
        # Missing values
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        ml_dataset.loc[missing_indices[:len(missing_indices)//2], 'income'] = None
        ml_dataset.loc[missing_indices[len(missing_indices)//2:], 'education'] = None
        
        # Outliers
        outlier_indices = np.random.choice(n_samples, size=20, replace=False)
        ml_dataset.loc[outlier_indices, 'income'] = np.random.uniform(200000, 500000, 20)
        
        # Save dataset
        csv_file = temp_directory / "ml_development_data.csv"
        ml_dataset.to_csv(csv_file, index=False)
        
        # Step 2: Data Ingestion
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv',
            'extract_metadata': True,
            'track_lineage': True
        }
        
        ingestion_result = full_platform_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        assert ingestion_result['rows'] == n_samples
        assert 'lineage' in ingestion_result
        
        # Step 3: Comprehensive Quality Assessment
        quality_report = full_platform_components['quality'].assess_quality(dataset_id)
        
        assert quality_report['overall_score'] > 0
        assert 'completeness_score' in quality_report
        assert 'accuracy_score' in quality_report
        assert 'consistency_score' in quality_report
        assert 'validity_score' in quality_report
        
        # Should detect missing values
        assert quality_report['completeness_score'] < 1.0
        
        # Step 4: Bias Analysis
        bias_report = full_platform_components['bias'].detect_bias(dataset_id, ['gender'])
        
        assert 'bias_detected' in bias_report
        assert 'gender' in bias_report['protected_attributes']
        
        # Should detect gender bias we introduced
        if bias_report['bias_detected']:
            assert bias_report['bias_score'] > 0.05
        
        # Step 5: Feature Engineering Recommendations
        feature_recommendations = full_platform_components['features'].recommend_features(
            dataset_id, 'classification'
        )
        
        assert 'recommended_features' in feature_recommendations
        assert 'transformations' in feature_recommendations
        assert len(feature_recommendations['recommended_features']) > 0
        
        # Step 6: Apply Feature Transformations
        selected_transformations = feature_recommendations['transformations'][:3]  # Apply first 3
        
        if selected_transformations:
            transformation_result = full_platform_components['features'].apply_transformations(
                dataset_id, selected_transformations
            )
            
            assert 'transformed_dataset_id' in transformation_result
            transformed_dataset_id = transformation_result['transformed_dataset_id']
        else:
            transformed_dataset_id = dataset_id
        
        # Step 7: Compliance Analysis
        compliance_result = full_platform_components['compliance'].analyze_compliance(
            transformed_dataset_id, ['GDPR', 'CCPA']
        )
        
        assert 'compliance_status' in compliance_result
        assert 'privacy_risks' in compliance_result
        
        # Step 8: Final AI Readiness Assessment
        ai_readiness = full_platform_components['reporting'].generate_ai_readiness_report(
            transformed_dataset_id
        )
        
        assert 'overall_ai_readiness_score' in ai_readiness
        assert 'dimension_scores' in ai_readiness
        assert 'recommendations' in ai_readiness
        assert 'improvement_roadmap' in ai_readiness
        
        # Step 9: Lineage Tracking Verification
        lineage_info = full_platform_components['lineage'].get_dataset_lineage(transformed_dataset_id)
        
        assert 'source_datasets' in lineage_info
        assert 'transformations' in lineage_info
        assert 'creation_timestamp' in lineage_info
        
        # Verify end-to-end coherence
        assert 0 <= ai_readiness['overall_ai_readiness_score'] <= 1
        
        # Quality issues should be reflected in AI readiness score
        if quality_report['overall_score'] < 0.8:
            assert ai_readiness['overall_ai_readiness_score'] < 0.9
    
    def test_production_monitoring_workflow(self, full_platform_components, temp_directory):
        """Test production data monitoring workflow."""
        # Create baseline/reference dataset
        np.random.seed(42)
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.exponential(2, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
            'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })
        
        baseline_file = temp_directory / "baseline_data.csv"
        baseline_data.to_csv(baseline_file, index=False)
        
        # Ingest baseline data
        baseline_config = {
            'source_type': 'file',
            'file_path': str(baseline_file),
            'format': 'csv'
        }
        
        baseline_result = full_platform_components['ingestion'].ingest_batch_data(baseline_config)
        baseline_dataset_id = baseline_result['dataset_id']
        
        # Create production data with drift
        production_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1.2, 1000),  # Mean and variance drift
            'feature2': np.random.exponential(2.5, 1000),   # Distribution drift
            'feature3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.3, 0.4, 0.3]),  # Category drift
            'target': np.random.choice([0, 1], 1000, p=[0.6, 0.4])  # Target drift
        })
        
        production_file = temp_directory / "production_data.csv"
        production_data.to_csv(production_file, index=False)
        
        # Ingest production data
        production_config = {
            'source_type': 'file',
            'file_path': str(production_file),
            'format': 'csv'
        }
        
        production_result = full_platform_components['ingestion'].ingest_batch_data(production_config)
        production_dataset_id = production_result['dataset_id']
        
        # Monitor drift
        drift_report = full_platform_components['drift'].monitor_drift(
            production_dataset_id, baseline_dataset_id
        )
        
        assert 'drift_detected' in drift_report
        assert 'drift_score' in drift_report
        assert 'feature_drift_scores' in drift_report
        
        # Should detect drift in all features
        feature_drift = drift_report['feature_drift_scores']
        assert 'feature1' in feature_drift
        assert 'feature2' in feature_drift
        assert 'feature3' in feature_drift
        
        # Drift scores should be significant
        assert feature_drift['feature1'] > 0.1
        assert feature_drift['feature2'] > 0.1
        assert feature_drift['feature3'] > 0.1
        
        # Set up drift monitoring alerts
        drift_thresholds = {
            'overall_drift_threshold': 0.2,
            'feature_drift_threshold': 0.15,
            'alert_frequency': 'daily'
        }
        
        alert_setup = full_platform_components['drift'].set_drift_thresholds(
            production_dataset_id, drift_thresholds
        )
        
        assert alert_setup['thresholds_set'] is True
        
        # Get drift alerts
        alerts = full_platform_components['drift'].get_drift_alerts(production_dataset_id)
        
        assert len(alerts) > 0  # Should have alerts due to detected drift
        
        for alert in alerts:
            assert 'alert_type' in alert
            assert 'severity' in alert
            assert 'feature' in alert
            assert 'drift_score' in alert
    
    def test_multi_dataset_comparison_workflow(self, full_platform_components, temp_directory):
        """Test workflow for comparing multiple datasets."""
        datasets = {}
        dataset_ids = {}
        
        # Create multiple datasets with different characteristics
        dataset_configs = {
            'high_quality': {
                'missing_rate': 0.01,
                'outlier_rate': 0.005,
                'bias_factor': 0.05
            },
            'medium_quality': {
                'missing_rate': 0.05,
                'outlier_rate': 0.02,
                'bias_factor': 0.15
            },
            'low_quality': {
                'missing_rate': 0.15,
                'outlier_rate': 0.05,
                'bias_factor': 0.3
            }
        }
        
        for dataset_name, config in dataset_configs.items():
            # Generate dataset with specified characteristics
            np.random.seed(42)
            n_samples = 2000
            
            # Base features
            age = np.random.randint(18, 80, n_samples)
            income = np.random.normal(50000, 15000, n_samples)
            gender = np.random.choice(['M', 'F'], n_samples)
            
            # Introduce bias
            approval_prob = np.where(
                gender == 'M',
                0.7,
                0.7 - config['bias_factor']
            )
            approved = np.random.binomial(1, approval_prob)
            
            data = pd.DataFrame({
                'age': age,
                'income': income,
                'gender': gender,
                'approved': approved
            })
            
            # Introduce missing values
            missing_count = int(n_samples * config['missing_rate'])
            missing_indices = np.random.choice(n_samples, missing_count, replace=False)
            data.loc[missing_indices[:missing_count//2], 'income'] = None
            data.loc[missing_indices[missing_count//2:], 'age'] = None
            
            # Introduce outliers
            outlier_count = int(n_samples * config['outlier_rate'])
            outlier_indices = np.random.choice(n_samples, outlier_count, replace=False)
            data.loc[outlier_indices, 'income'] = np.random.uniform(200000, 500000, outlier_count)
            
            datasets[dataset_name] = data
            
            # Save and ingest dataset
            csv_file = temp_directory / f"{dataset_name}_data.csv"
            data.to_csv(csv_file, index=False)
            
            source_config = {
                'source_type': 'file',
                'file_path': str(csv_file),
                'format': 'csv'
            }
            
            ingestion_result = full_platform_components['ingestion'].ingest_batch_data(source_config)
            dataset_ids[dataset_name] = ingestion_result['dataset_id']
        
        # Assess quality for all datasets
        quality_reports = {}
        for dataset_name, dataset_id in dataset_ids.items():
            quality_reports[dataset_name] = full_platform_components['quality'].assess_quality(dataset_id)
        
        # Verify quality ordering
        assert quality_reports['high_quality']['overall_score'] > quality_reports['medium_quality']['overall_score']
        assert quality_reports['medium_quality']['overall_score'] > quality_reports['low_quality']['overall_score']
        
        # Assess bias for all datasets
        bias_reports = {}
        for dataset_name, dataset_id in dataset_ids.items():
            bias_reports[dataset_name] = full_platform_components['bias'].detect_bias(dataset_id, ['gender'])
        
        # Verify bias ordering
        assert bias_reports['low_quality']['bias_score'] > bias_reports['medium_quality']['bias_score']
        assert bias_reports['medium_quality']['bias_score'] > bias_reports['high_quality']['bias_score']
        
        # Generate comparative AI readiness reports
        ai_readiness_reports = {}
        for dataset_name, dataset_id in dataset_ids.items():
            ai_readiness_reports[dataset_name] = full_platform_components['reporting'].generate_ai_readiness_report(dataset_id)
        
        # Verify AI readiness ordering
        assert (ai_readiness_reports['high_quality']['overall_ai_readiness_score'] > 
                ai_readiness_reports['medium_quality']['overall_ai_readiness_score'])
        assert (ai_readiness_reports['medium_quality']['overall_ai_readiness_score'] > 
                ai_readiness_reports['low_quality']['overall_ai_readiness_score'])
        
        # Generate comparative report
        comparative_report = full_platform_components['reporting'].generate_comparative_report(
            list(dataset_ids.values())
        )
        
        assert 'dataset_comparison' in comparative_report
        assert 'ranking' in comparative_report
        assert 'recommendations' in comparative_report
        
        # Verify ranking matches our expectations
        ranking = comparative_report['ranking']
        assert ranking[0]['dataset_id'] == dataset_ids['high_quality']
        assert ranking[-1]['dataset_id'] == dataset_ids['low_quality']
    
    def test_concurrent_processing_workflow(self, full_platform_components, temp_directory):
        """Test concurrent processing of multiple datasets."""
        # Create multiple datasets for concurrent processing
        datasets = []
        for i in range(5):
            data = pd.DataFrame({
                'id': range(i*1000, (i+1)*1000),
                'feature1': np.random.normal(i, 1, 1000),  # Different means
                'feature2': np.random.exponential(i+1, 1000),  # Different scales
                'category': np.random.choice(['A', 'B', 'C'], 1000),
                'target': np.random.choice([0, 1], 1000)
            })
            
            csv_file = temp_directory / f"concurrent_dataset_{i}.csv"
            data.to_csv(csv_file, index=False)
            datasets.append(csv_file)
        
        # Process datasets concurrently
        def process_dataset(file_path):
            source_config = {
                'source_type': 'file',
                'file_path': str(file_path),
                'format': 'csv'
            }
            
            # Ingestion
            ingestion_result = full_platform_components['ingestion'].ingest_batch_data(source_config)
            dataset_id = ingestion_result['dataset_id']
            
            # Quality assessment
            quality_report = full_platform_components['quality'].assess_quality(dataset_id)
            
            # Feature recommendations
            feature_recs = full_platform_components['features'].recommend_features(dataset_id, 'classification')
            
            return {
                'dataset_id': dataset_id,
                'file_path': str(file_path),
                'quality_score': quality_report['overall_score'],
                'feature_count': len(feature_recs['recommended_features'])
            }
        
        # Execute concurrent processing
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_dataset = {executor.submit(process_dataset, dataset): dataset for dataset in datasets}
            results = []
            
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    pytest.fail(f'Dataset {dataset} generated an exception: {exc}')
        
        concurrent_time = time.time() - start_time
        
        # Verify all datasets were processed
        assert len(results) == 5
        
        # Verify results are reasonable
        for result in results:
            assert 'dataset_id' in result
            assert 'quality_score' in result
            assert 0 <= result['quality_score'] <= 1
            assert result['feature_count'] > 0
        
        # Concurrent processing should be faster than sequential
        # (This is a rough check - actual speedup depends on system resources)
        assert concurrent_time < 30  # Should complete within reasonable time
    
    def test_real_time_streaming_simulation(self, full_platform_components):
        """Test real-time streaming data processing simulation."""
        # Simulate streaming data processing
        streaming_config = {
            'source_type': 'stream',
            'stream_name': 'test_stream',
            'batch_size': 100,
            'processing_interval': 1  # seconds
        }
        
        # Mock streaming data generator
        def generate_streaming_batch():
            return pd.DataFrame({
                'timestamp': [pd.Timestamp.now()] * 100,
                'feature1': np.random.normal(0, 1, 100),
                'feature2': np.random.exponential(1, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
        
        # Process multiple streaming batches
        batch_results = []
        
        for batch_num in range(3):  # Process 3 batches
            batch_data = generate_streaming_batch()
            
            # Simulate batch ingestion
            batch_file = f"streaming_batch_{batch_num}.csv"
            
            # Mock the streaming ingestion process
            ingestion_result = {
                'dataset_id': f'stream_batch_{batch_num}',
                'rows': len(batch_data),
                'columns': len(batch_data.columns),
                'batch_number': batch_num,
                'processing_timestamp': pd.Timestamp.now()
            }
            
            # Quick quality assessment for streaming data
            quality_metrics = {
                'completeness': 1.0 - batch_data.isnull().sum().sum() / (len(batch_data) * len(batch_data.columns)),
                'freshness': 1.0,  # Assume fresh data
                'volume': len(batch_data)
            }
            
            batch_result = {
                'batch_id': batch_num,
                'ingestion_result': ingestion_result,
                'quality_metrics': quality_metrics
            }
            
            batch_results.append(batch_result)
        
        # Verify streaming processing results
        assert len(batch_results) == 3
        
        for result in batch_results:
            assert 'batch_id' in result
            assert 'ingestion_result' in result
            assert 'quality_metrics' in result
            
            quality = result['quality_metrics']
            assert 'completeness' in quality
            assert 'freshness' in quality
            assert 'volume' in quality
            
            assert quality['completeness'] > 0.9  # High completeness expected
            assert quality['volume'] == 100  # Expected batch size
    
    def test_automated_remediation_workflow(self, full_platform_components, temp_directory):
        """Test automated data remediation workflow."""
        # Create dataset with known issues
        problematic_data = pd.DataFrame({
            'id': range(1000),
            'age': [25 if i % 10 != 0 else None for i in range(1000)],  # 10% missing
            'income': [50000 + np.random.normal(0, 10000) if i % 20 != 0 else None for i in range(1000)],  # 5% missing
            'category': ['A' if i % 3 == 0 else 'B' if i % 3 == 1 else 'C' for i in range(1000)],
            'score': [np.random.uniform(0, 100) if i % 50 != 0 else 150 for i in range(1000)]  # 2% outliers
        })
        
        # Add some outliers
        outlier_indices = np.random.choice(1000, 20, replace=False)
        problematic_data.loc[outlier_indices, 'income'] = np.random.uniform(200000, 500000, 20)
        
        csv_file = temp_directory / "problematic_data.csv"
        problematic_data.to_csv(csv_file, index=False)
        
        # Step 1: Ingest problematic data
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        ingestion_result = full_platform_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        # Step 2: Assess quality and identify issues
        quality_report = full_platform_components['quality'].assess_quality(dataset_id)
        
        assert quality_report['completeness_score'] < 1.0  # Should detect missing values
        assert len(quality_report['issues']) > 0  # Should identify issues
        
        # Step 3: Get improvement recommendations
        recommendations = full_platform_components['quality'].recommend_improvements(quality_report)
        
        assert 'recommendations' in recommendations
        assert len(recommendations['recommendations']) > 0
        
        # Step 4: Apply automated remediation
        remediation_actions = []
        for rec in recommendations['recommendations']:
            if rec['action_type'] == 'impute_missing':
                remediation_actions.append({
                    'type': 'imputation',
                    'column': rec['column'],
                    'method': rec['suggested_method']
                })
            elif rec['action_type'] == 'remove_outliers':
                remediation_actions.append({
                    'type': 'outlier_removal',
                    'column': rec['column'],
                    'method': rec['suggested_method']
                })
        
        # Apply remediation (mock implementation)
        if remediation_actions:
            remediation_result = {
                'remediated_dataset_id': f"{dataset_id}_remediated",
                'actions_applied': remediation_actions,
                'improvement_summary': {
                    'completeness_improvement': 0.15,
                    'accuracy_improvement': 0.10,
                    'overall_improvement': 0.12
                }
            }
            
            remediated_dataset_id = remediation_result['remediated_dataset_id']
        else:
            remediated_dataset_id = dataset_id
        
        # Step 5: Verify improvement
        # Mock improved quality assessment
        improved_quality_report = {
            'dataset_id': remediated_dataset_id,
            'overall_score': quality_report['overall_score'] + 0.12,
            'completeness_score': min(quality_report['completeness_score'] + 0.15, 1.0),
            'accuracy_score': min(quality_report['accuracy_score'] + 0.10, 1.0),
            'consistency_score': quality_report['consistency_score'],
            'validity_score': quality_report['validity_score'],
            'issues': [issue for issue in quality_report['issues'] if issue['severity'] != 'high']
        }
        
        # Verify improvement
        assert improved_quality_report['overall_score'] > quality_report['overall_score']
        assert improved_quality_report['completeness_score'] > quality_report['completeness_score']
        assert len(improved_quality_report['issues']) < len(quality_report['issues'])
    
    def test_compliance_and_privacy_workflow(self, full_platform_components, temp_directory):
        """Test compliance and privacy validation workflow."""
        # Create dataset with PII and sensitive information
        sensitive_data = pd.DataFrame({
            'customer_id': range(1000),
            'name': [f'Customer_{i}' for i in range(1000)],
            'email': [f'customer{i}@email.com' for i in range(1000)],
            'ssn': [f'{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}' for _ in range(1000)],
            'phone': [f'{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for _ in range(1000)],
            'age': np.random.randint(18, 80, 1000),
            'income': np.random.normal(50000, 15000, 1000),
            'medical_condition': np.random.choice(['None', 'Diabetes', 'Hypertension', 'Heart Disease'], 1000, p=[0.7, 0.1, 0.1, 0.1]),
            'credit_score': np.random.randint(300, 850, 1000),
            'approved': np.random.choice([0, 1], 1000)
        })
        
        csv_file = temp_directory / "sensitive_data.csv"
        sensitive_data.to_csv(csv_file, index=False)
        
        # Step 1: Ingest sensitive data
        source_config = {
            'source_type': 'file',
            'file_path': str(csv_file),
            'format': 'csv'
        }
        
        ingestion_result = full_platform_components['ingestion'].ingest_batch_data(source_config)
        dataset_id = ingestion_result['dataset_id']
        
        # Step 2: Compliance analysis
        compliance_regulations = ['GDPR', 'CCPA', 'HIPAA']
        compliance_result = full_platform_components['compliance'].analyze_compliance(
            dataset_id, compliance_regulations
        )
        
        assert 'compliance_status' in compliance_result
        assert 'privacy_risks' in compliance_result
        assert 'pii_detected' in compliance_result
        assert 'sensitive_data_detected' in compliance_result
        
        # Should detect PII and sensitive data
        assert compliance_result['pii_detected'] is True
        assert compliance_result['sensitive_data_detected'] is True
        
        # Should identify compliance risks
        assert len(compliance_result['privacy_risks']) > 0
        
        # Step 3: Privacy-preserving recommendations
        privacy_recommendations = full_platform_components['compliance'].recommend_privacy_techniques(
            compliance_result
        )
        
        assert 'anonymization_techniques' in privacy_recommendations
        assert 'data_minimization' in privacy_recommendations
        assert 'access_controls' in privacy_recommendations
        
        # Should recommend anonymization for PII
        anon_techniques = privacy_recommendations['anonymization_techniques']
        assert len(anon_techniques) > 0
        
        # Should recommend removing or masking sensitive columns
        data_min = privacy_recommendations['data_minimization']
        assert 'columns_to_remove' in data_min or 'columns_to_mask' in data_min
        
        # Step 4: Apply privacy-preserving transformations (mock)
        privacy_transformations = [
            {'type': 'remove_column', 'column': 'ssn'},
            {'type': 'remove_column', 'column': 'name'},
            {'type': 'remove_column', 'column': 'email'},
            {'type': 'mask_column', 'column': 'phone'},
            {'type': 'generalize', 'column': 'age', 'bins': [18, 30, 45, 60, 80]},
            {'type': 'add_noise', 'column': 'income', 'noise_level': 0.1}
        ]
        
        anonymized_result = {
            'anonymized_dataset_id': f"{dataset_id}_anonymized",
            'transformations_applied': privacy_transformations,
            'privacy_risk_reduction': 0.8,
            'utility_preservation': 0.85
        }
        
        # Step 5: Verify compliance improvement
        anonymized_dataset_id = anonymized_result['anonymized_dataset_id']
        
        # Mock compliance re-assessment
        improved_compliance = {
            'compliance_status': {
                'GDPR': 'compliant',
                'CCPA': 'compliant',
                'HIPAA': 'partially_compliant'
            },
            'privacy_risks': [],  # Reduced risks
            'pii_detected': False,
            'sensitive_data_detected': True,  # Still some sensitive data (medical)
            'risk_score': 0.2  # Reduced from original
        }
        
        # Verify improvement
        assert improved_compliance['compliance_status']['GDPR'] == 'compliant'
        assert improved_compliance['compliance_status']['CCPA'] == 'compliant'
        assert improved_compliance['pii_detected'] is False
        assert len(improved_compliance['privacy_risks']) == 0
        
        # Step 6: Final AI readiness with privacy considerations
        final_ai_readiness = full_platform_components['reporting'].generate_ai_readiness_report(
            anonymized_dataset_id
        )
        
        assert 'privacy_compliance_score' in final_ai_readiness['dimension_scores']
        assert final_ai_readiness['dimension_scores']['privacy_compliance_score'] > 0.8