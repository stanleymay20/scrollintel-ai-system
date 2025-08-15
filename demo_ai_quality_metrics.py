"""Demo script for AI-specific quality metrics and scoring functionality."""

import pandas as pd
import numpy as np
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.ai_quality_metrics import AIQualityMetrics
from ai_data_readiness.models.base_models import Dataset, Schema, DatasetMetadata
from ai_data_readiness.core.config import Config


def create_sample_datasets():
    """Create sample datasets with different AI readiness characteristics."""
    np.random.seed(42)
    
    # Dataset 1: High-quality AI-ready data
    good_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.randint(1, 10, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.4, 0.35, 0.25]),  # Balanced
        'outcome': np.random.randint(0, 2, 1000)
    })
    
    # Dataset 2: Poor quality data with AI-specific issues
    poor_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),  # Highly correlated with feature1
        'feature3': np.ones(1000),  # Zero variance
        'category': np.random.choice(['A', 'B'], 1000, p=[0.95, 0.05]),  # Highly imbalanced
        'target': np.random.randint(0, 2, 1000),  # Suspicious target column
        'email_address': [f'user{i}@example.com' for i in range(1000)],  # PII
    })
    
    # Make feature2 highly correlated with feature1
    poor_data['feature2'] = poor_data['feature1'] * 0.9 + np.random.normal(0, 0.1, 1000)
    
    # Add missing values
    poor_data.loc[np.random.choice(poor_data.index, 200, replace=False), 'feature1'] = np.nan
    
    return good_data, poor_data


def create_datasets_metadata():
    """Create dataset metadata objects."""
    good_schema = Schema(columns={
        'feature1': 'float', 'feature2': 'float', 'feature3': 'integer',
        'category': 'categorical', 'outcome': 'integer'
    })
    
    poor_schema = Schema(columns={
        'feature1': 'float', 'feature2': 'float', 'feature3': 'integer',
        'category': 'categorical', 'target': 'integer', 'email_address': 'string'
    })
    
    good_dataset = Dataset(
        id='good_dataset_001',
        name='High Quality Dataset',
        schema=good_schema,
        metadata=DatasetMetadata(name='good_data', row_count=1000, column_count=5)
    )
    
    poor_dataset = Dataset(
        id='poor_dataset_001',
        name='Poor Quality Dataset',
        schema=poor_schema,
        metadata=DatasetMetadata(name='poor_data', row_count=1000, column_count=6)
    )
    
    return good_dataset, poor_dataset


def analyze_dataset(name, dataset, data, engine, ai_metrics):
    """Analyze a dataset and display results."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    
    # Get base quality report
    quality_report = engine.assess_quality(dataset, data)
    
    # Calculate AI readiness score
    ai_score = ai_metrics.calculate_ai_readiness_score(dataset, data, quality_report)
    
    # Display overall scores
    print(f"\n📊 OVERALL SCORES:")
    print(f"   AI Readiness Score:    {ai_score.overall_score:.3f}")
    print(f"   Data Quality Score:    {ai_score.data_quality_score:.3f}")
    print(f"   Feature Quality Score: {ai_score.feature_quality_score:.3f}")
    print(f"   Bias Score:           {ai_score.bias_score:.3f}")
    print(f"   Compliance Score:     {ai_score.compliance_score:.3f}")
    print(f"   Scalability Score:    {ai_score.scalability_score:.3f}")
    
    # Display dimension details
    print(f"\n🔍 DIMENSION DETAILS:")
    for dim_name, dim_score in ai_score.dimensions.items():
        print(f"   {dim_name.replace('_', ' ').title()}: {dim_score.score:.3f} (weight: {dim_score.weight})")
        if 'details' in dim_score.details:
            for key, value in dim_score.details.items():
                if isinstance(value, (int, float)):
                    print(f"     - {key}: {value:.3f}")
                else:
                    print(f"     - {key}: {value}")
    
    # Display improvement areas
    if ai_score.improvement_areas:
        print(f"\n⚠️  IMPROVEMENT AREAS ({len(ai_score.improvement_areas)}):")
        for area in ai_score.improvement_areas:
            print(f"   {area.area.replace('_', ' ').title()}: {area.current_score:.3f} → {area.target_score:.3f} ({area.priority} priority)")
            for action in area.actions[:2]:  # Show first 2 actions
                print(f"     • {action}")
    else:
        print(f"\n✅ NO IMPROVEMENT AREAS IDENTIFIED")
    
    # Detect anomalies
    anomaly_result = ai_metrics.detect_anomalies(data)
    print(f"\n🚨 ANOMALY DETECTION:")
    print(f"   Anomaly Score:     {anomaly_result['anomaly_score']:.3f}")
    print(f"   Total Anomalies:   {anomaly_result['total_anomalies']}")
    print(f"   Anomalous Rows:    {anomaly_result['anomalous_rows']}")
    
    if anomaly_result['anomalies']:
        print(f"   Anomaly Types:")
        anomaly_types = {}
        for anomaly in anomaly_result['anomalies']:
            anomaly_type = anomaly['type']
            if anomaly_type not in anomaly_types:
                anomaly_types[anomaly_type] = 0
            anomaly_types[anomaly_type] += 1
        
        for anomaly_type, count in anomaly_types.items():
            print(f"     - {anomaly_type.replace('_', ' ').title()}: {count}")


def main():
    """Main demo function."""
    print("🤖 AI DATA READINESS PLATFORM - QUALITY METRICS DEMO")
    print("=" * 60)
    
    # Initialize engines
    config = Config()
    engine = QualityAssessmentEngine(config)
    ai_metrics = AIQualityMetrics()
    
    # Create sample datasets
    good_data, poor_data = create_sample_datasets()
    good_dataset, poor_dataset = create_datasets_metadata()
    
    # Analyze datasets
    analyze_dataset("HIGH QUALITY DATASET", good_dataset, good_data, engine, ai_metrics)
    analyze_dataset("POOR QUALITY DATASET", poor_dataset, poor_data, engine, ai_metrics)
    
    print(f"\n{'='*60}")
    print("🎯 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\n📋 KEY FEATURES DEMONSTRATED:")
    print(f"   ✅ AI readiness scoring algorithm")
    print(f"   ✅ Feature correlation analysis")
    print(f"   ✅ Target leakage detection")
    print(f"   ✅ Statistical anomaly detection")
    print(f"   ✅ Bias and fairness assessment")
    print(f"   ✅ Compliance and PII detection")
    print(f"   ✅ Scalability assessment")
    print(f"   ✅ Improvement recommendations")
    
    print(f"\n🔧 REQUIREMENTS IMPLEMENTED:")
    print(f"   • Requirement 1.3: Feature correlation and target leakage detection")
    print(f"   • Requirement 1.4: Statistical anomaly detection capabilities")
    print(f"   • Requirement 5.1: AI readiness scoring algorithm")


if __name__ == "__main__":
    main()