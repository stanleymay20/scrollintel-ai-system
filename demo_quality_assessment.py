"""Demo script for the Quality Assessment Engine."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.models.base_models import Dataset, DatasetMetadata, Schema
from ai_data_readiness.core.config import Config


def create_sample_data():
    """Create sample data with various quality issues."""
    np.random.seed(42)
    
    # Create data with quality issues
    data = pd.DataFrame({
        'customer_id': [1, 2, None, 4, 5, 5, 7, 8, 9, 10],  # Missing value and duplicate
        'name': ['John Doe', '', 'JANE SMITH', 'bob johnson', None, 'Alice Brown', 'charlie wilson', 'DAVID CLARK', 'eve adams', 'Frank Miller'],  # Mixed case, empty, missing
        'email': ['john@email', 'invalid-email', 'jane@example.com', 'bob@test.com', None, 'alice@example.com', 'charlie@test', 'david@example.com', 'eve@test.com', 'frank@example.com'],  # Invalid formats
        'age': [25, -5, 150, 30, None, 35, 40, 999, 45, 50],  # Negative, unrealistic values
        'salary': [50000, None, 1000000, 60000, 55000, None, 70000, -10000, 80000, 90000],  # Missing, extreme values
        'registration_date': ['2023-01-01', None, '2025-12-31', '2023-03-01', '2023-04-01', 'invalid-date', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01'],  # Future date, invalid format
        'is_premium': [True, 'yes', False, 1, None, True, 'no', False, True, 0]  # Mixed types
    })
    
    return data


def create_dataset_schema():
    """Create dataset schema definition."""
    schema = Schema(
        columns={
            'customer_id': 'integer',
            'name': 'string',
            'email': 'string',
            'age': 'integer',
            'salary': 'float',
            'registration_date': 'datetime',
            'is_premium': 'boolean'
        },
        primary_key='customer_id'
    )
    
    metadata = DatasetMetadata(
        name="customer_data",
        description="Customer registration and profile data",
        row_count=10,
        column_count=7
    )
    
    return Dataset(
        id="customer-dataset-001",
        name="Customer Data",
        schema=schema,
        metadata=metadata
    )


def main():
    """Run quality assessment demo."""
    print("=== AI Data Readiness Platform - Quality Assessment Demo ===\n")
    
    # Initialize the quality assessment engine
    config = Config()
    engine = QualityAssessmentEngine(config)
    
    # Create sample data and dataset
    data = create_sample_data()
    dataset = create_dataset_schema()
    
    print("Sample Data:")
    print(data.to_string())
    print(f"\nData Shape: {data.shape}")
    print(f"Dataset ID: {dataset.id}")
    print(f"Schema: {dataset.schema.columns}")
    
    print("\n" + "="*60)
    print("RUNNING QUALITY ASSESSMENT...")
    print("="*60)
    
    # Perform quality assessment
    report = engine.assess_quality(dataset, data)
    
    # Display results
    print(f"\n📊 QUALITY ASSESSMENT RESULTS")
    print(f"{'='*40}")
    print(f"Overall Quality Score: {report.overall_score:.3f}")
    print(f"Generated at: {report.generated_at}")
    
    print(f"\n📈 DIMENSION SCORES:")
    print(f"  • Completeness: {report.completeness_score:.3f}")
    print(f"  • Accuracy:     {report.accuracy_score:.3f}")
    print(f"  • Consistency:  {report.consistency_score:.3f}")
    print(f"  • Validity:     {report.validity_score:.3f}")
    print(f"  • Uniqueness:   {report.uniqueness_score:.3f}")
    print(f"  • Timeliness:   {report.timeliness_score:.3f}")
    
    print(f"\n⚠️  QUALITY ISSUES IDENTIFIED ({len(report.issues)}):")
    for i, issue in enumerate(report.issues, 1):
        print(f"  {i}. [{issue.severity.upper()}] {issue.dimension.value.title()}")
        print(f"     {issue.description}")
        print(f"     Affected columns: {', '.join(issue.affected_columns)}")
        print(f"     Affected rows: {issue.affected_rows}")
        print(f"     Recommendation: {issue.recommendation}")
        print()
    
    print(f"💡 IMPROVEMENT RECOMMENDATIONS ({len(report.recommendations)}):")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. [{rec.priority.upper()}] {rec.type.replace('_', ' ').title()}")
        print(f"     {rec.description}")
        print(f"     Implementation: {rec.implementation}")
        print(f"     Estimated Impact: {rec.estimated_impact:.1%}")
        print(f"     Estimated Effort: {rec.estimated_effort}")
        print()
    
    # Quality assessment summary
    print(f"📋 ASSESSMENT SUMMARY:")
    print(f"{'='*40}")
    
    if report.overall_score >= 0.9:
        quality_level = "EXCELLENT"
        emoji = "🟢"
    elif report.overall_score >= 0.7:
        quality_level = "GOOD"
        emoji = "🟡"
    elif report.overall_score >= 0.5:
        quality_level = "FAIR"
        emoji = "🟠"
    else:
        quality_level = "POOR"
        emoji = "🔴"
    
    print(f"{emoji} Data Quality Level: {quality_level}")
    print(f"   Overall Score: {report.overall_score:.1%}")
    
    if report.overall_score < config.quality.ai_readiness_threshold:
        print(f"❌ Data does not meet AI readiness threshold ({config.quality.ai_readiness_threshold:.1%})")
        print("   Recommend addressing quality issues before using for AI/ML")
    else:
        print(f"✅ Data meets AI readiness threshold ({config.quality.ai_readiness_threshold:.1%})")
        print("   Data is suitable for AI/ML applications")
    
    print(f"\n🔧 NEXT STEPS:")
    if report.issues:
        print("   1. Address the identified quality issues")
        print("   2. Implement the recommended improvements")
        print("   3. Re-run quality assessment to verify improvements")
        print("   4. Proceed with AI/ML model development")
    else:
        print("   1. Data quality is excellent!")
        print("   2. Proceed with AI/ML model development")
        print("   3. Set up continuous monitoring for production")
    
    print(f"\n{'='*60}")
    print("Quality assessment completed successfully! 🎉")


if __name__ == "__main__":
    main()