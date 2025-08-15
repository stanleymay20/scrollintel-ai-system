#!/usr/bin/env python3
"""
Demo script for AI Data Readiness Platform - Anonymization Engine

This script demonstrates the data anonymization and privacy protection
capabilities including various anonymization techniques and privacy risk assessment.
"""

print("🚀 Starting Anonymization Engine Demo")

import pandas as pd
import numpy as np
from datetime import datetime

# Import the anonymization engine components
from ai_data_readiness.engines.anonymization_engine import (
    AnonymizationEngine, AnonymizationTechnique, PrivacyRiskLevel,
    AnonymizationConfig, create_anonymization_engine
)

print("✅ Imports successful")


def create_sample_datasets():
    """Create sample datasets for demonstration"""
    
    print("\n📊 Creating sample datasets...")
    
    # Customer dataset with various sensitive information
    customer_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana'],
        'last_name': ['Doe', 'Smith', 'Wilson', 'Brown', 'Davis', 'Miller'],
        'email': [
            'john.doe@email.com', 
            'jane.smith@company.org', 
            'bob.wilson@test.com',
            'alice.brown@example.net',
            'charlie.davis@domain.co',
            'diana.miller@sample.org'
        ],
        'phone': ['555-123-4567', '555-987-6543', '555-555-5555', '555-111-2222', '555-999-8888', '555-777-6666'],
        'ssn': ['123-45-6789', '987-65-4321', '555-44-3333', '111-22-3333', '777-88-9999', '444-55-6666'],
        'age': [25, 34, 45, 28, 52, 31],
        'salary': [65000, 85000, 95000, 72000, 110000, 78000],
        'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Executive', 'Marketing']
    })
    
    # Healthcare dataset with PHI
    healthcare_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'medical_record_number': ['MRN12345', 'MRN67890', 'MRN11111', 'MRN22222', 'MRN33333'],
        'diagnosis': ['Diabetes Type 2', 'Hypertension', 'Asthma', 'Arthritis', 'Migraine'],
        'treatment': ['Metformin', 'Lisinopril', 'Albuterol', 'Ibuprofen', 'Sumatriptan'],
        'doctor_notes': [
            'Patient shows good compliance',
            'Blood pressure well controlled',
            'Asthma symptoms improved',
            'Joint pain manageable',
            'Headache frequency reduced'
        ],
        'insurance_id': ['INS001', 'INS002', 'INS003', 'INS004', 'INS005']
    })
    
    # Financial dataset
    financial_data = pd.DataFrame({
        'account_id': ['ACC001', 'ACC002', 'ACC003', 'ACC004'],
        'account_holder': ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown'],
        'credit_card': ['4532-1234-5678-9012', '5555-4444-3333-2222', '4111-1111-1111-1111', '3782-822463-10005'],
        'bank_account': ['123456789', '987654321', '555444333', '111222333'],
        'transaction_amount': [1250.50, 89.99, 2340.75, 567.25],
        'merchant': ['Amazon', 'Starbucks', 'Best Buy', 'Target']
    })
    
    return {
        'customer': customer_data,
        'healthcare': healthcare_data,
        'financial': financial_data
    }


def demonstrate_privacy_risk_assessment():
    """Demonstrate privacy risk assessment capabilities"""
    
    print("\n" + "="*80)
    print("PRIVACY RISK ASSESSMENT DEMO")
    print("="*80)
    
    engine = create_anonymization_engine()
    datasets = create_sample_datasets()
    
    for dataset_name, data in datasets.items():
        print(f"\n📋 ASSESSING PRIVACY RISKS: {dataset_name.upper()} DATASET")
        print("-" * 60)
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Perform privacy risk assessment
        assessments = engine.assess_privacy_risk(data)
        
        print(f"\n🔍 PRIVACY RISK ANALYSIS RESULTS:")
        for assessment in assessments:
            risk_emoji = {
                'low': '🟢',
                'medium': '🟡', 
                'high': '🟠',
                'critical': '🔴'
            }
            
            print(f"\n  Column: {assessment.column_name}")
            print(f"  Risk Level: {risk_emoji.get(assessment.risk_level.value, '❓')} {assessment.risk_level.value.upper()}")
            print(f"  Risk Score: {assessment.risk_score:.2f}")
            print(f"  Vulnerability Factors: {', '.join(assessment.vulnerability_factors) if assessment.vulnerability_factors else 'None'}")
            print(f"  Recommended Techniques: {', '.join([t.value for t in assessment.recommended_techniques])}")


def demonstrate_anonymization_techniques():
    """Demonstrate various anonymization techniques"""
    
    print("\n" + "="*80)
    print("ANONYMIZATION TECHNIQUES DEMO")
    print("="*80)
    
    engine = create_anonymization_engine()
    datasets = create_sample_datasets()
    
    # Use customer data for demonstration
    customer_data = datasets['customer']
    
    print(f"\n📊 ORIGINAL CUSTOMER DATA:")
    print(customer_data.head(3).to_string())
    
    # Demonstrate different anonymization techniques
    techniques_to_demo = [
        {
            'name': 'PSEUDONYMIZATION',
            'technique': AnonymizationTechnique.PSEUDONYMIZATION,
            'columns': ['customer_id', 'email'],
            'parameters': {}
        },
        {
            'name': 'DATA MASKING',
            'technique': AnonymizationTechnique.MASKING,
            'columns': ['phone', 'ssn'],
            'parameters': {
                'mask_char': '*',
                'preserve_length': True,
                'preserve_format': True
            }
        },
        {
            'name': 'K-ANONYMITY',
            'technique': AnonymizationTechnique.K_ANONYMITY,
            'columns': ['age', 'salary'],
            'parameters': {'k': 3}
        },
        {
            'name': 'GENERALIZATION',
            'technique': AnonymizationTechnique.GENERALIZATION,
            'columns': ['department'],
            'parameters': {'generalization_levels': 2}
        },
        {
            'name': 'SUPPRESSION',
            'technique': AnonymizationTechnique.SUPPRESSION,
            'columns': ['first_name', 'last_name'],
            'parameters': {'suppression_rate': 0.3}
        }
    ]
    
    for demo in techniques_to_demo:
        print(f"\n🔧 APPLYING {demo['name']}:")
        print("-" * 50)
        
        config = AnonymizationConfig(
            technique=demo['technique'],
            parameters=demo['parameters'],
            target_columns=demo['columns']
        )
        
        result = engine.anonymize_data(customer_data, config)
        
        print(f"Technique: {result.technique_applied.value}")
        print(f"Processing Time: {result.processing_time:.3f} seconds")
        print(f"Privacy Gain: {result.privacy_gain:.1%}")
        print(f"Utility Loss: {result.utility_loss:.1%}")
        print(f"Columns Processed: {', '.join(demo['columns'])}")
        
        # Show sample of anonymized data
        print(f"\nSample Anonymized Data:")
        sample_columns = demo['columns'][:2]  # Show first 2 columns
        if sample_columns:
            for col in sample_columns:
                if col in result.anonymized_data.columns:
                    original_sample = customer_data[col].head(3).tolist()
                    anonymized_sample = result.anonymized_data[col].head(3).tolist()
                    print(f"  {col}:")
                    print(f"    Original:   {original_sample}")
                    print(f"    Anonymized: {anonymized_sample}")


def demonstrate_synthetic_data_generation():
    """Demonstrate synthetic data generation"""
    
    print("\n" + "="*80)
    print("SYNTHETIC DATA GENERATION DEMO")
    print("="*80)
    
    engine = create_anonymization_engine()
    datasets = create_sample_datasets()
    
    # Use healthcare data for synthetic generation
    healthcare_data = datasets['healthcare']
    
    print(f"\n📊 ORIGINAL HEALTHCARE DATA:")
    print(healthcare_data.to_string())
    
    config = AnonymizationConfig(
        technique=AnonymizationTechnique.SYNTHETIC_DATA,
        parameters={},
        target_columns=list(healthcare_data.columns)
    )
    
    result = engine.anonymize_data(healthcare_data, config)
    
    print(f"\n🤖 SYNTHETIC HEALTHCARE DATA:")
    print(result.anonymized_data.to_string())
    
    print(f"\n📈 SYNTHETIC DATA METRICS:")
    print(f"Original Shape: {result.original_data_shape}")
    print(f"Synthetic Shape: {result.anonymized_data.shape}")
    print(f"Privacy Gain: {result.privacy_gain:.1%}")
    print(f"Utility Loss: {result.utility_loss:.1%}")
    print(f"Processing Time: {result.processing_time:.3f} seconds")


def demonstrate_anonymization_recommendations():
    """Demonstrate anonymization strategy recommendations"""
    
    print("\n" + "="*80)
    print("ANONYMIZATION STRATEGY RECOMMENDATIONS")
    print("="*80)
    
    engine = create_anonymization_engine()
    datasets = create_sample_datasets()
    
    # Use financial data for recommendations
    financial_data = datasets['financial']
    
    print(f"\n💳 ANALYZING FINANCIAL DATA:")
    print(financial_data.to_string())
    
    # Assess privacy risks
    assessments = engine.assess_privacy_risk(financial_data)
    
    # Get recommendations
    recommendations = engine.recommend_anonymization_strategy(assessments)
    
    print(f"\n💡 ANONYMIZATION RECOMMENDATIONS:")
    print("-" * 50)
    
    for i, config in enumerate(recommendations, 1):
        print(f"\n{i}. Column: {', '.join(config.target_columns)}")
        print(f"   Recommended Technique: {config.technique.value.upper()}")
        print(f"   Parameters: {config.parameters}")
        print(f"   Preserve Utility: {config.preserve_utility}")
        print(f"   Risk Threshold: {config.risk_threshold}")
        
        # Apply the recommendation
        result = engine.anonymize_data(financial_data, config)
        print(f"   Expected Privacy Gain: {result.privacy_gain:.1%}")
        print(f"   Expected Utility Loss: {result.utility_loss:.1%}")


def demonstrate_differential_privacy():
    """Demonstrate differential privacy technique"""
    
    print("\n" + "="*80)
    print("DIFFERENTIAL PRIVACY DEMO")
    print("="*80)
    
    engine = create_anonymization_engine()
    
    # Create numerical dataset for differential privacy
    numerical_data = pd.DataFrame({
        'age': [25, 34, 45, 28, 52, 31, 38, 42],
        'salary': [65000, 85000, 95000, 72000, 110000, 78000, 88000, 92000],
        'years_experience': [3, 8, 15, 5, 20, 7, 12, 18]
    })
    
    print(f"\n📊 ORIGINAL NUMERICAL DATA:")
    print(numerical_data.to_string())
    
    # Apply differential privacy with different epsilon values
    epsilon_values = [0.1, 1.0, 10.0]
    
    for epsilon in epsilon_values:
        print(f"\n🔒 DIFFERENTIAL PRIVACY (ε = {epsilon}):")
        print("-" * 40)
        
        config = AnonymizationConfig(
            technique=AnonymizationTechnique.DIFFERENTIAL_PRIVACY,
            parameters={
                'epsilon': epsilon,
                'sensitivity': 1.0
            },
            target_columns=['age', 'salary', 'years_experience']
        )
        
        result = engine.anonymize_data(numerical_data, config)
        
        print(f"Privacy Budget (ε): {epsilon}")
        print(f"Privacy Gain: {result.privacy_gain:.1%}")
        print(f"Utility Loss: {result.utility_loss:.1%}")
        
        # Show comparison of means
        print(f"\nMean Comparison:")
        for col in ['age', 'salary', 'years_experience']:
            original_mean = numerical_data[col].mean()
            dp_mean = result.anonymized_data[col].mean()
            print(f"  {col}: {original_mean:.1f} → {dp_mean:.1f}")


def main():
    """Main demo function"""
    
    print("=" * 80)
    print("AI DATA READINESS PLATFORM - ANONYMIZATION ENGINE DEMO")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demonstrate_privacy_risk_assessment()
        demonstrate_anonymization_techniques()
        demonstrate_synthetic_data_generation()
        demonstrate_anonymization_recommendations()
        demonstrate_differential_privacy()
        
        print("\n" + "="*80)
        print("✅ ANONYMIZATION ENGINE DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("• Privacy risk assessment")
        print("• Multiple anonymization techniques:")
        print("  - Pseudonymization")
        print("  - Data masking")
        print("  - K-anonymity")
        print("  - Generalization")
        print("  - Suppression")
        print("  - Differential privacy")
        print("  - Synthetic data generation")
        print("• Automated strategy recommendations")
        print("• Privacy and utility metrics")
        print("• Comprehensive privacy protection")
        
        print("\n📋 Task 6.2 Implementation Summary:")
        print("✅ AnonymizationEngine class")
        print("✅ Multiple anonymization techniques")
        print("✅ Privacy risk assessment algorithms")
        print("✅ Automated recommendation system")
        print("✅ Privacy and utility metrics")
        print("✅ Comprehensive test coverage")
        print("✅ Working demonstration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()