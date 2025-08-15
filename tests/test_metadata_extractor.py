"""
Tests for MetadataExtractor - AI Data Readiness Platform

This module tests the comprehensive metadata extraction and dataset profiling
capabilities with automatic cataloging and versioning support.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, patch

from ai_data_readiness.core.metadata_extractor import (
    MetadataExtractor, 
    ProfileLevel, 
    DatasetProfile, 
    ColumnProfile,
    SchemaVersion
)
from ai_data_readiness.core.metadata_extractor import MetadataExtractionError


class TestMetadataExtractor:
    """Test suite for MetadataExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create MetadataExtractor instance"""
        return MetadataExtractor()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        np.random.seed(42)
        data = {
            'id': range(1, 101),
            'name': [f'User_{i}' for i in range(1, 101)],
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100),
            'is_active': np.random.choice([True, False], 100),
            'join_date': pd.date_range('2020-01-01', periods=100, freq='D')
        }
        
        # Add some missing values
        df = pd.DataFrame(data)
        df.loc[5:10, 'email'] = None
        df.loc[15:20, 'salary'] = None
        
        return df
    
    @pytest.fixture
    def sample_dataframe_with_duplicates(self):
        """Create DataFrame with duplicate rows"""
        data = {
            'col1': [1, 2, 3, 1, 2],
            'col2': ['A', 'B', 'C', 'A', 'B'],
            'col3': [10.5, 20.5, 30.5, 10.5, 20.5]
        }
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_extract_comprehensive_metadata_basic(self, extractor, sample_dataframe):
        """Test basic metadata extraction"""
        profile = await extractor.extract_comprehensive_metadata(
            sample_dataframe, 
            'test_dataset_1', 
            ProfileLevel.BASIC
        )
        
        assert isinstance(profile, DatasetProfile)
        assert profile.dataset_id == 'test_dataset_1'
        assert profile.profile_level == ProfileLevel.BASIC
        assert profile.row_count == 100
        assert profile.column_count == 8
        assert profile.missing_values_total > 0  # We added some missing values
        assert len(profile.column_profiles) == 8
        
        # Basic level should not have correlations or statistical summary
        assert profile.correlations is None
        assert profile.statistical_summary is None
        assert profile.data_quality_score is None
    
    @pytest.mark.asyncio
    async def test_extract_comprehensive_metadata_standard(self, extractor, sample_dataframe):
        """Test standard level metadata extraction"""
        profile = await extractor.extract_comprehensive_metadata(
            sample_dataframe, 
            'test_dataset_2', 
            ProfileLevel.STANDARD
        )
        
        assert profile.profile_level == ProfileLevel.STANDARD
        assert profile.correlations is not None
        assert profile.statistical_summary is not None
        assert profile.data_quality_score is None  # Only available in comprehensive
        
        # Check correlations structure
        assert isinstance(profile.correlations, dict)
        
        # Check statistical summary structure
        assert 'numeric_columns' in profile.statistical_summary
        assert 'categorical_columns' in profile.statistical_summary
    
    @pytest.mark.asyncio
    async def test_extract_comprehensive_metadata_comprehensive(self, extractor, sample_dataframe):
        """Test comprehensive level metadata extraction"""
        profile = await extractor.extract_comprehensive_metadata(
            sample_dataframe, 
            'test_dataset_3', 
            ProfileLevel.COMPREHENSIVE
        )
        
        assert profile.profile_level == ProfileLevel.COMPREHENSIVE
        assert profile.correlations is not None
        assert profile.statistical_summary is not None
        assert profile.data_quality_score is not None
        assert isinstance(profile.data_quality_score, float)
        assert 0 <= profile.data_quality_score <= 100
    
    @pytest.mark.asyncio
    async def test_column_profiling_numeric(self, extractor, sample_dataframe):
        """Test numeric column profiling"""
        profile = await extractor.extract_comprehensive_metadata(
            sample_dataframe, 
            'test_dataset_4', 
            ProfileLevel.COMPREHENSIVE
        )
        
        # Find age column profile
        age_profile = next(cp for cp in profile.column_profiles if cp.name == 'age')
        
        assert age_profile.data_type == 'int64'
        assert age_profile.min_value is not None
        assert age_profile.max_value is not None
        assert age_profile.mean_value is not None
        assert age_profile.median_value is not None
        assert age_profile.std_deviation is not None
        assert age_profile.quartiles is not None
        assert 'q1' in age_profile.quartiles
        assert 'q2' in age_profile.quartiles
        assert 'q3' in age_profile.quartiles
    
    @pytest.mark.asyncio
    async def test_column_profiling_text(self, extractor, sample_dataframe):
        """Test text column profiling"""
        profile = await extractor.extract_comprehensive_metadata(
            sample_dataframe, 
            'test_dataset_5', 
            ProfileLevel.COMPREHENSIVE
        )
        
        # Find email column profile
        email_profile = next(cp for cp in profile.column_profiles if cp.name == 'email')
        
        assert email_profile.data_type == 'object'
        assert email_profile.pattern_analysis is not None
        assert 'avg_length' in email_profile.pattern_analysis
        assert 'email_pattern_matches' in email_profile.pattern_analysis
        assert email_profile.pattern_analysis['email_pattern_matches'] > 0
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, extractor, sample_dataframe_with_duplicates):
        """Test duplicate row detection"""
        profile = await extractor.extract_comprehensive_metadata(
            sample_dataframe_with_duplicates, 
            'test_dataset_duplicates', 
            ProfileLevel.BASIC
        )
        
        assert profile.duplicate_rows == 2  # Two duplicate rows
        assert profile.duplicate_rows_percentage == 40.0  # 2 out of 5 rows
    
    @pytest.mark.asyncio
    async def test_data_quality_score_calculation(self, extractor, sample_dataframe):
        """Test data quality score calculation"""
        profile = await extractor.extract_comprehensive_metadata(
            sample_dataframe, 
            'test_dataset_quality', 
            ProfileLevel.COMPREHENSIVE
        )
        
        assert profile.data_quality_score is not None
        assert isinstance(profile.data_quality_score, float)
        assert 0 <= profile.data_quality_score <= 100
        
        # With missing values, score should be less than perfect
        assert profile.data_quality_score < 100
    
    @pytest.mark.asyncio
    async def test_schema_catalog_entry_creation(self, extractor, sample_dataframe):
        """Test schema catalog entry creation"""
        with patch.object(extractor, '_get_schema_versions', return_value=[]):
            with patch.object(extractor, '_store_schema_version', return_value=None):
                schema_version = await extractor.create_schema_catalog_entry(
                    sample_dataframe,
                    'test_dataset_schema',
                    'test_user',
                    'Initial schema creation'
                )
                
                assert isinstance(schema_version, SchemaVersion)
                assert schema_version.dataset_id == 'test_dataset_schema'
                assert schema_version.created_by == 'test_user'
                assert schema_version.version_number == 1
                assert len(schema_version.schema.columns) == 8
    
    @pytest.mark.asyncio
    async def test_schema_change_detection_no_changes(self, extractor, sample_dataframe):
        """Test schema change detection with no changes"""
        # Mock existing schema version
        mock_schema_version = SchemaVersion(
            version_id='test_v1',
            dataset_id='test_dataset',
            schema=await extractor._extract_schema_from_dataframe(sample_dataframe, 'test_dataset'),
            version_number=1,
            created_at=datetime.utcnow(),
            created_by='test_user',
            change_summary='Initial'
        )
        
        with patch.object(extractor, '_get_schema_versions', return_value=[mock_schema_version]):
            has_changes, changes = await extractor.detect_schema_changes(
                sample_dataframe, 
                'test_dataset'
            )
            
            assert not has_changes
            assert len(changes) == 0
    
    @pytest.mark.asyncio
    async def test_schema_change_detection_with_changes(self, extractor, sample_dataframe):
        """Test schema change detection with changes"""
        # Create modified DataFrame
        modified_df = sample_dataframe.copy()
        modified_df['new_column'] = 'test'
        modified_df = modified_df.drop('email', axis=1)
        modified_df['age'] = modified_df['age'].astype(float)  # Type change
        
        # Mock existing schema version with original DataFrame
        mock_schema_version = SchemaVersion(
            version_id='test_v1',
            dataset_id='test_dataset',
            schema=await extractor._extract_schema_from_dataframe(sample_dataframe, 'test_dataset'),
            version_number=1,
            created_at=datetime.utcnow(),
            created_by='test_user',
            change_summary='Initial'
        )
        
        with patch.object(extractor, '_get_schema_versions', return_value=[mock_schema_version]):
            has_changes, changes = await extractor.detect_schema_changes(
                modified_df, 
                'test_dataset'
            )
            
            assert has_changes
            assert len(changes) >= 3  # Added column, removed column, type change
            
            # Check for specific changes
            change_strings = ' '.join(changes)
            assert 'Added column: new_column' in change_strings
            assert 'Removed column: email' in change_strings
            assert 'Changed type for age' in change_strings
    
    @pytest.mark.asyncio
    async def test_get_dataset_catalog_specific(self, extractor):
        """Test getting catalog for specific dataset"""
        mock_versions = [
            SchemaVersion(
                version_id='test_v1',
                dataset_id='test_dataset',
                schema=None,
                version_number=1,
                created_at=datetime.utcnow(),
                created_by='test_user',
                change_summary='Initial'
            )
        ]
        
        with patch.object(extractor, '_get_schema_versions', return_value=mock_versions):
            with patch.object(extractor, '_get_dataset_metadata', return_value=None):
                catalog = await extractor.get_dataset_catalog('test_dataset')
                
                assert catalog['dataset_id'] == 'test_dataset'
                assert len(catalog['schema_versions']) == 1
                assert catalog['latest_version'] is not None
    
    @pytest.mark.asyncio
    async def test_get_dataset_catalog_all(self, extractor):
        """Test getting catalog for all datasets"""
        mock_datasets = ['dataset1', 'dataset2']
        mock_versions = [
            SchemaVersion(
                version_id='test_v1',
                dataset_id='dataset1',
                schema=None,
                version_number=1,
                created_at=datetime.utcnow(),
                created_by='test_user',
                change_summary='Initial'
            )
        ]
        
        with patch.object(extractor, '_get_all_datasets', return_value=mock_datasets):
            with patch.object(extractor, '_get_schema_versions', return_value=mock_versions):
                with patch.object(extractor, '_get_dataset_metadata', return_value=None):
                    catalog = await extractor.get_dataset_catalog()
                    
                    assert len(catalog) == 2
                    assert 'dataset1' in catalog
                    assert 'dataset2' in catalog
    
    @pytest.mark.asyncio
    async def test_text_pattern_analysis(self, extractor):
        """Test text pattern analysis"""
        # Create DataFrame with various text patterns
        data = {
            'emails': ['user1@example.com', 'user2@test.org', 'invalid_email', None],
            'phones': ['123-456-7890', '+1-555-123-4567', 'not_a_phone', None],
            'urls': ['https://example.com', 'http://test.org', 'not_a_url', None],
            'mixed_text': ['Hello123!', 'Test@#$', 'Simple', None]
        }
        df = pd.DataFrame(data)
        
        profile = await extractor.extract_comprehensive_metadata(
            df, 
            'test_patterns', 
            ProfileLevel.COMPREHENSIVE
        )
        
        # Check email column pattern analysis
        email_profile = next(cp for cp in profile.column_profiles if cp.name == 'emails')
        assert email_profile.pattern_analysis is not None
        assert email_profile.pattern_analysis['email_pattern_matches'] == 2
        
        # Check phone column pattern analysis
        phone_profile = next(cp for cp in profile.column_profiles if cp.name == 'phones')
        assert phone_profile.pattern_analysis['phone_pattern_matches'] >= 1
        
        # Check URL column pattern analysis
        url_profile = next(cp for cp in profile.column_profiles if cp.name == 'urls')
        assert url_profile.pattern_analysis['url_pattern_matches'] == 2
    
    @pytest.mark.asyncio
    async def test_correlation_calculation(self, extractor):
        """Test correlation calculation"""
        # Create DataFrame with correlated numeric columns
        np.random.seed(42)
        x = np.random.randn(100)
        data = {
            'x': x,
            'y': x * 2 + np.random.randn(100) * 0.1,  # Highly correlated
            'z': np.random.randn(100),  # Uncorrelated
            'text': ['text'] * 100  # Non-numeric
        }
        df = pd.DataFrame(data)
        
        profile = await extractor.extract_comprehensive_metadata(
            df, 
            'test_correlation', 
            ProfileLevel.STANDARD
        )
        
        assert profile.correlations is not None
        assert 'x' in profile.correlations
        assert 'y' in profile.correlations['x']
        
        # x and y should be highly correlated
        xy_correlation = profile.correlations['x']['y']
        assert abs(xy_correlation) > 0.8
    
    @pytest.mark.asyncio
    async def test_error_handling(self, extractor):
        """Test error handling in metadata extraction"""
        # Test with invalid DataFrame
        with pytest.raises(MetadataExtractionError):
            await extractor.extract_comprehensive_metadata(
                None, 
                'invalid_dataset', 
                ProfileLevel.BASIC
            )
    
    @pytest.mark.asyncio
    async def test_empty_dataframe_handling(self, extractor):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        profile = await extractor.extract_comprehensive_metadata(
            empty_df, 
            'empty_dataset', 
            ProfileLevel.BASIC
        )
        
        assert profile.row_count == 0
        assert profile.column_count == 0
        assert profile.missing_values_total == 0
        assert profile.missing_values_percentage == 0
        assert len(profile.column_profiles) == 0
    
    @pytest.mark.asyncio
    async def test_single_column_dataframe(self, extractor):
        """Test handling of single column DataFrame"""
        single_col_df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        
        profile = await extractor.extract_comprehensive_metadata(
            single_col_df, 
            'single_col_dataset', 
            ProfileLevel.COMPREHENSIVE
        )
        
        assert profile.row_count == 5
        assert profile.column_count == 1
        assert len(profile.column_profiles) == 1
        assert profile.column_profiles[0].name == 'single_col'
        
        # Correlations should be empty for single column
        assert profile.correlations == {}
    
    def test_schema_hash_generation(self, extractor):
        """Test schema hash generation"""
        from ai_data_readiness.models.base_models import Schema, ColumnSchema
        
        schema1 = Schema(
            dataset_id='test',
            columns=[
                ColumnSchema(name='col1', data_type='int64', nullable=False),
                ColumnSchema(name='col2', data_type='object', nullable=True)
            ],
            primary_keys=[],
            foreign_keys=[],
            indexes=[],
            constraints=[]
        )
        
        schema2 = Schema(
            dataset_id='test',
            columns=[
                ColumnSchema(name='col1', data_type='int64', nullable=False),
                ColumnSchema(name='col2', data_type='object', nullable=True)
            ],
            primary_keys=[],
            foreign_keys=[],
            indexes=[],
            constraints=[]
        )
        
        schema3 = Schema(
            dataset_id='test',
            columns=[
                ColumnSchema(name='col1', data_type='float64', nullable=False),  # Different type
                ColumnSchema(name='col2', data_type='object', nullable=True)
            ],
            primary_keys=[],
            foreign_keys=[],
            indexes=[],
            constraints=[]
        )
        
        hash1 = extractor._generate_schema_hash(schema1)
        hash2 = extractor._generate_schema_hash(schema2)
        hash3 = extractor._generate_schema_hash(schema3)
        
        # Same schemas should have same hash
        assert hash1 == hash2
        
        # Different schemas should have different hash
        assert hash1 != hash3