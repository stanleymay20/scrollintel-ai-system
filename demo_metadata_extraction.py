"""
Demo script for AI Data Readiness Platform - Metadata Extraction

This script demonstrates the comprehensive metadata extraction and dataset profiling
capabilities with automatic cataloging and versioning support.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from ai_data_readiness.core.metadata_extractor import (
    MetadataExtractor, 
    ProfileLevel
)


async def create_sample_datasets():
    """Create sample datasets for demonstration"""
    
    # Dataset 1: Customer data with various data quality issues
    np.random.seed(42)
    customer_data = {
        'customer_id': range(1, 1001),
        'first_name': [f'Customer_{i}' for i in range(1, 1001)],
        'last_name': [f'Lastname_{i}' for i in range(1, 1001)],
        'email': [f'customer{i}@example.com' if i % 10 != 0 else None for i in range(1, 1001)],
        'phone': [f'555-{i:04d}' if i % 15 != 0 else None for i in range(1, 1001)],
        'age': np.random.randint(18, 80, 1000),
        'annual_income': np.random.normal(50000, 20000, 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'account_balance': np.random.normal(5000, 10000, 1000),
        'registration_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'is_premium': np.random.choice([True, False], 1000, p=[0.3, 0.7]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 1000)
    }
    
    customer_df = pd.DataFrame(customer_data)
    
    # Add some data quality issues
    # Missing values
    customer_df.loc[50:60, 'annual_income'] = None
    customer_df.loc[100:110, 'credit_score'] = None
    
    # Duplicate rows
    customer_df = pd.concat([customer_df, customer_df.iloc[0:5]], ignore_index=True)
    
    # Outliers
    customer_df.loc[0, 'annual_income'] = 1000000  # Outlier
    customer_df.loc[1, 'age'] = 150  # Invalid age
    
    # Dataset 2: Product data with different characteristics
    product_data = {
        'product_id': [f'PROD_{i:05d}' for i in range(1, 501)],
        'product_name': [f'Product {i}' for i in range(1, 501)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], 500),
        'price': np.random.exponential(50, 500),
        'cost': np.random.exponential(30, 500),
        'weight': np.random.gamma(2, 2, 500),
        'dimensions': [f'{np.random.randint(10, 100)}x{np.random.randint(10, 100)}x{np.random.randint(5, 50)}' for _ in range(500)],
        'in_stock': np.random.choice([True, False], 500, p=[0.8, 0.2]),
        'supplier_id': np.random.randint(1, 50, 500),
        'launch_date': pd.date_range('2019-01-01', periods=500, freq='3D'),
        'rating': np.random.uniform(1, 5, 500),
        'review_count': np.random.poisson(100, 500)
    }
    
    product_df = pd.DataFrame(product_data)
    
    # Add some missing values
    product_df.loc[20:30, 'rating'] = None
    product_df.loc[40:45, 'weight'] = None
    
    return customer_df, product_df


async def demonstrate_basic_profiling():
    """Demonstrate basic dataset profiling"""
    print("=" * 80)
    print("BASIC DATASET PROFILING DEMONSTRATION")
    print("=" * 80)
    
    extractor = MetadataExtractor()
    customer_df, product_df = await create_sample_datasets()
    
    # Basic profiling of customer dataset
    print("\n1. Basic Customer Dataset Profile:")
    print("-" * 40)
    
    customer_profile = await extractor.extract_comprehensive_metadata(
        customer_df, 
        'customer_dataset_v1', 
        ProfileLevel.BASIC
    )
    
    print(f"Dataset ID: {customer_profile.dataset_id}")
    print(f"Rows: {customer_profile.row_count:,}")
    print(f"Columns: {customer_profile.column_count}")
    print(f"Memory Usage: {customer_profile.memory_usage / 1024 / 1024:.2f} MB")
    print(f"Missing Values: {customer_profile.missing_values_total:,} ({customer_profile.missing_values_percentage:.2f}%)")
    print(f"Duplicate Rows: {customer_profile.duplicate_rows} ({customer_profile.duplicate_rows_percentage:.2f}%)")
    print(f"Data Types: {customer_profile.data_types_distribution}")
    
    print("\nColumn Profiles (First 5):")
    for i, col_profile in enumerate(customer_profile.column_profiles[:5]):
        print(f"  {col_profile.name} ({col_profile.data_type}):")
        print(f"    - Null: {col_profile.null_count} ({col_profile.null_percentage:.1f}%)")
        print(f"    - Unique: {col_profile.unique_count} ({col_profile.unique_percentage:.1f}%)")
        print(f"    - Most Frequent: {col_profile.most_frequent_value} (count: {col_profile.most_frequent_count})")


async def demonstrate_standard_profiling():
    """Demonstrate standard dataset profiling with correlations"""
    print("\n\n" + "=" * 80)
    print("STANDARD DATASET PROFILING WITH CORRELATIONS")
    print("=" * 80)
    
    extractor = MetadataExtractor()
    customer_df, product_df = await create_sample_datasets()
    
    # Standard profiling of product dataset
    print("\n2. Standard Product Dataset Profile:")
    print("-" * 40)
    
    product_profile = await extractor.extract_comprehensive_metadata(
        product_df, 
        'product_dataset_v1', 
        ProfileLevel.STANDARD
    )
    
    print(f"Dataset ID: {product_profile.dataset_id}")
    print(f"Profile Level: {product_profile.profile_level.value}")
    print(f"Rows: {product_profile.row_count:,}")
    print(f"Columns: {product_profile.column_count}")
    
    # Show statistical summary
    if product_profile.statistical_summary:
        print(f"\nStatistical Summary:")
        print(f"  - Numeric columns: {product_profile.statistical_summary['numeric_columns']}")
        print(f"  - Categorical columns: {product_profile.statistical_summary['categorical_columns']}")
        print(f"  - DateTime columns: {product_profile.statistical_summary['datetime_columns']}")
        print(f"  - Boolean columns: {product_profile.statistical_summary['boolean_columns']}")
    
    # Show correlations
    if product_profile.correlations:
        print(f"\nTop Correlations:")
        correlations = []
        for col1, corr_dict in product_profile.correlations.items():
            for col2, corr_value in corr_dict.items():
                if col1 != col2 and abs(corr_value) > 0.1:
                    correlations.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for col1, col2, corr in correlations[:5]:
            print(f"  - {col1} ↔ {col2}: {corr:.3f}")


async def demonstrate_comprehensive_profiling():
    """Demonstrate comprehensive dataset profiling with quality scoring"""
    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE DATASET PROFILING WITH QUALITY SCORING")
    print("=" * 80)
    
    extractor = MetadataExtractor()
    customer_df, product_df = await create_sample_datasets()
    
    # Comprehensive profiling of customer dataset
    print("\n3. Comprehensive Customer Dataset Profile:")
    print("-" * 40)
    
    customer_profile = await extractor.extract_comprehensive_metadata(
        customer_df, 
        'customer_dataset_comprehensive', 
        ProfileLevel.COMPREHENSIVE
    )
    
    print(f"Dataset ID: {customer_profile.dataset_id}")
    print(f"Data Quality Score: {customer_profile.data_quality_score}/100")
    
    # Show detailed column analysis
    print(f"\nDetailed Column Analysis:")
    for col_profile in customer_profile.column_profiles:
        print(f"\n  {col_profile.name} ({col_profile.data_type}):")
        print(f"    - Completeness: {100 - col_profile.null_percentage:.1f}%")
        print(f"    - Uniqueness: {col_profile.unique_percentage:.1f}%")
        
        if col_profile.min_value is not None:
            print(f"    - Range: {col_profile.min_value} to {col_profile.max_value}")
            print(f"    - Mean: {col_profile.mean_value:.2f}")
            print(f"    - Std Dev: {col_profile.std_deviation:.2f}")
        
        if col_profile.pattern_analysis:
            print(f"    - Pattern Analysis:")
            for key, value in col_profile.pattern_analysis.items():
                if isinstance(value, (int, float)):
                    print(f"      • {key}: {value}")
        
        if col_profile.value_distribution:
            print(f"    - Top Values: {dict(list(col_profile.value_distribution.items())[:3])}")


async def demonstrate_schema_versioning():
    """Demonstrate schema cataloging and versioning"""
    print("\n\n" + "=" * 80)
    print("SCHEMA CATALOGING AND VERSIONING")
    print("=" * 80)
    
    extractor = MetadataExtractor()
    customer_df, product_df = await create_sample_datasets()
    
    print("\n4. Schema Versioning Demo:")
    print("-" * 40)
    
    # Create initial schema version
    print("Creating initial schema version...")
    try:
        schema_v1 = await extractor.create_schema_catalog_entry(
            customer_df,
            'customer_schema_demo',
            'demo_user',
            'Initial customer schema with all fields'
        )
        
        print(f"✓ Created schema version: {schema_v1.version_id}")
        print(f"  - Version number: {schema_v1.version_number}")
        print(f"  - Columns: {len(schema_v1.schema.columns)}")
        print(f"  - Created by: {schema_v1.created_by}")
        print(f"  - Change summary: {schema_v1.change_summary}")
        
    except Exception as e:
        print(f"Note: Schema versioning requires database setup. Error: {e}")
    
    # Demonstrate schema change detection
    print("\nDetecting schema changes...")
    
    # Modify the dataset
    modified_df = customer_df.copy()
    modified_df['new_loyalty_score'] = np.random.randint(1, 10, len(modified_df))
    modified_df = modified_df.drop('phone', axis=1)
    modified_df['age'] = modified_df['age'].astype(float)
    
    try:
        has_changes, changes = await extractor.detect_schema_changes(
            modified_df,
            'customer_schema_demo'
        )
        
        print(f"Schema changes detected: {has_changes}")
        if changes:
            print("Changes found:")
            for change in changes:
                print(f"  - {change}")
                
    except Exception as e:
        print(f"Note: Change detection requires database setup. Error: {e}")


async def demonstrate_text_pattern_analysis():
    """Demonstrate advanced text pattern analysis"""
    print("\n\n" + "=" * 80)
    print("ADVANCED TEXT PATTERN ANALYSIS")
    print("=" * 80)
    
    extractor = MetadataExtractor()
    
    # Create dataset with various text patterns
    text_data = {
        'emails': [
            'john.doe@company.com', 'jane.smith@university.edu', 'invalid-email',
            'user123@domain.org', None, 'another@test.co.uk', 'bad_format',
            'support@service.com', 'admin@localhost', 'test@example.com'
        ],
        'phone_numbers': [
            '555-123-4567', '+1-800-555-0199', '(555) 987-6543',
            '555.123.4567', 'not-a-phone', None, '1234567890',
            '+44 20 7946 0958', '555-CALL-NOW', '123-45-6789'
        ],
        'urls': [
            'https://www.example.com', 'http://test.org/page',
            'https://secure.site.com/login', 'ftp://files.server.com',
            'not-a-url', None, 'www.missing-protocol.com',
            'https://api.service.com/v1/endpoint', 'http://localhost:8080',
            'https://subdomain.domain.co.uk/path'
        ],
        'mixed_content': [
            'Product ABC-123', 'Order #ORD-456789', 'Customer ID: CUST_001',
            'Serial: SN123456789', 'Code: XYZ-999', None,
            'Reference: REF_2023_001', 'Batch: B20230815001',
            'Item: ITEM-SKU-789', 'Transaction: TXN_20230815_001'
        ]
    }
    
    text_df = pd.DataFrame(text_data)
    
    print("\n5. Text Pattern Analysis:")
    print("-" * 40)
    
    text_profile = await extractor.extract_comprehensive_metadata(
        text_df,
        'text_patterns_demo',
        ProfileLevel.COMPREHENSIVE
    )
    
    for col_profile in text_profile.column_profiles:
        if col_profile.pattern_analysis:
            print(f"\n{col_profile.name.upper()} Analysis:")
            print(f"  - Average length: {col_profile.pattern_analysis['avg_length']:.1f}")
            print(f"  - Length range: {col_profile.pattern_analysis['min_length']} - {col_profile.pattern_analysis['max_length']}")
            print(f"  - Email patterns: {col_profile.pattern_analysis['email_pattern_matches']}")
            print(f"  - Phone patterns: {col_profile.pattern_analysis['phone_pattern_matches']}")
            print(f"  - URL patterns: {col_profile.pattern_analysis['url_pattern_matches']}")
            print(f"  - Contains numbers: {col_profile.pattern_analysis['contains_numbers']}")
            print(f"  - Contains special chars: {col_profile.pattern_analysis['contains_special_chars']}")


async def demonstrate_data_quality_assessment():
    """Demonstrate comprehensive data quality assessment"""
    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    extractor = MetadataExtractor()
    
    # Create datasets with different quality levels
    
    # High quality dataset
    high_quality_data = {
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'age': np.random.randint(18, 65, 100),
        'salary': np.random.normal(50000, 10000, 100)
    }
    high_quality_df = pd.DataFrame(high_quality_data)
    
    # Low quality dataset
    low_quality_data = {
        'id': [1, 2, 3, 1, 2, None, 7, 8, 9, 10],  # Duplicates and nulls
        'name': ['User1', None, 'User3', 'User1', None, 'User6', '', 'User8', None, 'User10'],
        'email': ['user1@test', None, 'invalid', 'user1@test', 'user5@', None, 'user7@example.com', '', None, 'user10@test.com'],
        'age': [25, None, -5, 25, 150, None, 30, 35, None, 40],  # Invalid values
        'salary': [50000, None, -1000, 50000, 999999, None, 60000, 55000, None, 65000]  # Outliers
    }
    low_quality_df = pd.DataFrame(low_quality_data)
    
    print("\n6. Data Quality Comparison:")
    print("-" * 40)
    
    # Profile high quality dataset
    high_quality_profile = await extractor.extract_comprehensive_metadata(
        high_quality_df,
        'high_quality_dataset',
        ProfileLevel.COMPREHENSIVE
    )
    
    # Profile low quality dataset
    low_quality_profile = await extractor.extract_comprehensive_metadata(
        low_quality_df,
        'low_quality_dataset',
        ProfileLevel.COMPREHENSIVE
    )
    
    print(f"HIGH QUALITY DATASET:")
    print(f"  - Data Quality Score: {high_quality_profile.data_quality_score}/100")
    print(f"  - Missing Values: {high_quality_profile.missing_values_percentage:.1f}%")
    print(f"  - Duplicate Rows: {high_quality_profile.duplicate_rows_percentage:.1f}%")
    print(f"  - Rows: {high_quality_profile.row_count}")
    
    print(f"\nLOW QUALITY DATASET:")
    print(f"  - Data Quality Score: {low_quality_profile.data_quality_score}/100")
    print(f"  - Missing Values: {low_quality_profile.missing_values_percentage:.1f}%")
    print(f"  - Duplicate Rows: {low_quality_profile.duplicate_rows_percentage:.1f}%")
    print(f"  - Rows: {low_quality_profile.row_count}")
    
    print(f"\nQUALITY IMPROVEMENT POTENTIAL:")
    improvement = high_quality_profile.data_quality_score - low_quality_profile.data_quality_score
    print(f"  - Score difference: {improvement:.1f} points")
    print(f"  - Improvement potential: {improvement/high_quality_profile.data_quality_score*100:.1f}%")


async def main():
    """Main demonstration function"""
    print("AI DATA READINESS PLATFORM")
    print("Metadata Extraction & Dataset Profiling Demo")
    print("=" * 80)
    
    try:
        await demonstrate_basic_profiling()
        await demonstrate_standard_profiling()
        await demonstrate_comprehensive_profiling()
        await demonstrate_schema_versioning()
        await demonstrate_text_pattern_analysis()
        await demonstrate_data_quality_assessment()
        
        print("\n\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("✓ Multi-level dataset profiling (Basic, Standard, Comprehensive)")
        print("✓ Comprehensive column analysis with statistics")
        print("✓ Advanced text pattern recognition")
        print("✓ Correlation analysis for numeric data")
        print("✓ Data quality scoring and assessment")
        print("✓ Schema cataloging and versioning")
        print("✓ Schema change detection")
        print("✓ Missing value and duplicate detection")
        print("✓ Statistical summaries and distributions")
        
        print("\nNext Steps:")
        print("- Set up database for schema versioning")
        print("- Integrate with data ingestion pipeline")
        print("- Add quality assessment rules engine")
        print("- Implement automated remediation suggestions")
        
    except Exception as e:
        print(f"\nDemo encountered an error: {e}")
        print("This is expected if database components are not set up.")


if __name__ == "__main__":
    asyncio.run(main())