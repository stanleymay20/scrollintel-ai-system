"""Simple test for metadata extraction"""

import asyncio
import pandas as pd
import numpy as np
from ai_data_readiness.core.metadata_extractor import MetadataExtractor, ProfileLevel

async def test_basic_extraction():
    """Test basic metadata extraction"""
    # Create sample data
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'is_active': np.random.choice([True, False], 100)
    }
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[5:10, 'name'] = None
    df.loc[15:20, 'salary'] = None
    
    # Create extractor and extract metadata
    extractor = MetadataExtractor()
    profile = await extractor.extract_comprehensive_metadata(
        df, 'test_dataset', ProfileLevel.COMPREHENSIVE
    )
    
    print(f"Dataset ID: {profile.dataset_id}")
    print(f"Row count: {profile.row_count}")
    print(f"Column count: {profile.column_count}")
    print(f"Missing values: {profile.missing_values_total} ({profile.missing_values_percentage:.2f}%)")
    print(f"Duplicate rows: {profile.duplicate_rows} ({profile.duplicate_rows_percentage:.2f}%)")
    print(f"Data quality score: {profile.data_quality_score}")
    
    print("\nColumn profiles:")
    for col_profile in profile.column_profiles:
        print(f"  {col_profile.name}: {col_profile.data_type}, "
              f"null: {col_profile.null_percentage:.1f}%, "
              f"unique: {col_profile.unique_percentage:.1f}%")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_basic_extraction())