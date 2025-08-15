"""Test the metadata extraction methods directly"""

import asyncio
import pandas as pd
import numpy as np
from ai_data_readiness.core.metadata_extractor import MetadataExtractor, ProfileLevel

async def test_methods():
    """Test individual methods"""
    # Create sample data
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'name': [f'User_{i}' for i in range(1, 101)]
    }
    df = pd.DataFrame(data)
    
    extractor = MetadataExtractor()
    
    print("Testing correlation calculation...")
    correlations = await extractor._calculate_correlations(df)
    print(f"Correlations: {correlations}")
    
    print("\nTesting statistical summary...")
    summary = await extractor._generate_statistical_summary(df)
    print(f"Summary: {summary}")
    
    print("\nTesting data quality score...")
    # Create dummy column profiles
    column_profiles = []
    for col in df.columns:
        from ai_data_readiness.core.metadata_extractor import ColumnProfile
        profile = ColumnProfile(
            name=col,
            data_type=str(df[col].dtype),
            null_count=0,
            null_percentage=0.0,
            unique_count=df[col].nunique(),
            unique_percentage=100.0,
            most_frequent_value=None,
            most_frequent_count=0
        )
        column_profiles.append(profile)
    
    quality_score = await extractor._calculate_data_quality_score(df, column_profiles)
    print(f"Quality score: {quality_score}")

if __name__ == "__main__":
    asyncio.run(test_methods())