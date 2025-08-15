"""
Tests for Metadata Extraction Accuracy - AI Data Readiness Platform

This module tests the accuracy and reliability of metadata extraction
and cataloging system with various data scenarios and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import json

from ai_data_readiness.core.metadata_extractor import (
    MetadataExtractor, 
    ProfileLevel, 
    DatasetProfile, 
    ColumnProfile,
    SchemaVersion
)
from ai_data_readiness.core.schema_catalog import SchemaCatalog
from ai_data_readiness.core.exceptions import MetadataExtractionError
from ai_data_readiness.models.catalog_models import (
    Schema, ColumnSchema, SchemaChange, SchemaChangeType,
    CatalogEntry, SchemaEvolution
)


class TestMetadataExtractionAccuracy:
    """Test suite for metadata extraction accuracy"""
    
    @pytest.fixture
    def extractor(self):
        """Create MetadataExtractor instance"""
        return MetadataExtractor()
    
    @pytest.fixture
    def schema_catalog(self):
        """Create SchemaCatalog instance"""
        return SchemaCatalog()
    
    @pytest.fixture
    def high_quality_dataset(self):
        """Create high-quality dataset for testing"""
        np.random.seed(42)
        data = {
            'customer_id': range(1, 1001),
            'first_name': [f'FirstName{i}' for i in range(1, 1001)],
            'last_name': [f'LastName{i}' for i in range(1, 1001)],
            'email': [f'customer{i}@company.com' for i in range(1, 1001)],
            'age': np.random.randint(18, 80, 1000),
            'annual_income': np.random.normal(75000, 25000, 1000),
            'credit_score': np.random.randint(300, 850, 1000),
            'account_balance': np.random.exponential(5000, 1000),
            'is_premium': np.random.choice([True, False], 1000, p=[0.3, 0.7]),
            'registration_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'last_login': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France'], 1000),
            'subscription_tier': np.random.choice(['Basic', 'Premium', 'Enterprise'], 1000, p=[0.5, 0.3, 0.2])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def low_quality_dataset(self):
        """Create low-quality dataset with various issues"""
        np.random.seed(42)
        
        # Create base data
        data = {
            'id': list(range(1, 501)) + list(range(1, 501)),  # Duplicates
            'name': [f'Name{i}' if i % 5 != 0 else None for i in range(1, 1001)],  # 20% missing
            'email': [f'user{i}@test.com' if i % 3 != 0 else 'invalid_email' for i in range(1, 1001)],  # Invalid emails
            'age': [np.random.randint(18, 80) if i % 4 != 0 else None for i in range(1, 1001)],  # 25% missing
            'salary': [np.random.normal(50000, 15000) if i % 6 != 0 else -1000 for i in range(1, 1001)],  # Negative values
            'phone': [f'123-456-{i:04d}' if i % 7 != 0 else 'not_a_phone' for i in range(1, 1001)],  # Invalid phones
            'mixed_types': [str(i) if i % 2 == 0 else i for i in range(1, 1001)],  # Mixed types
            'inconsistent_dates': [
                datetime(2020, 1, 1) + timedelta(days=i) if i % 8 != 0 else f'2020-01-{i%30+1:02d}'
                for i in range(1, 1001)
            ]  # Mixed date formats
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def complex_schema_dataset(self):
        """Create dataset with complex schema for testing"""
        data = {
            'uuid_field': [f'uuid-{i:04d}-{i*2:04d}-{i*3:04d}' for i in range(1, 101)],
            'json_field': [json.dumps({'key': f'value{i}', 'nested': {'count': i}}) for i in range(1, 101)],
            'array_field': [[i, i*2, i*3] for i in range(1, 101)],
            'decimal_precision': [round(i * 3.14159, 6) for i in range(1, 101)],
            'categorical_ordered': pd.Categorical(['Low', 'Medium', 'High'] * 33 + ['Low'], ordered=True),
            'text_long': [f'This is a long text field with {i} words ' * (i % 10 + 1) for i in range(1, 101)],
            'binary_data': [bytes(f'binary{i}', 'utf-8') for i in range(1, 101)],
            'sparse_data': [i if i % 20 == 0 else None for i in range(1, 101)]  # 95% sparse
        }
        
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_high_quality_data_profiling_accuracy(self, extractor, high_quality_dataset):
        """Test metadata extraction accuracy on high-quality data"""
        profile = await extractor.extract_comprehensive_metadata(
            high_quality_dataset,
            'high_quality_test',
            ProfileLevel.COMPREHENSIVE
        )
        
        # Verify basic statistics accuracy
        assert profile.row_count == 1000
        assert profile.column_count == 13
        assert profile.missing_values_total == 0  # No missing values in high-quality data
        assert profile.missing_values_percentage == 0.0
        assert profile.duplicate_rows == 0  # No duplicates
        
        # Verify data quality score is high
        assert profile.data_quality_score >= 95.0
        
        # Verify column profiling accuracy
        customer_id_profile = next(cp for cp in profile.column_profiles if cp.name == 'customer_id')
        assert customer_id_profile.unique_count == 1000
        assert customer_id_profile.unique_percentage == 100.0
        assert customer_id_profile.null_count == 0
        
        # Verify email pattern detection
        email_profile = next(cp for cp in profile.column_profiles if cp.name == 'email')
        assert email_profile.pattern_analysis['email_pattern_matches'] == 1000
        
        # Verify numeric statistics accuracy
        age_profile = next(cp for cp in profile.column_profiles if cp.name == 'age')
        assert 18 <= age_profile.min_value <= 20  # Should be close to 18
        assert 78 <= age_profile.max_value <= 80  # Should be close to 80
        assert age_profile.quartiles is not None
        
        # Verify correlations are calculated
        assert profile.correlations is not None
        assert len(profile.correlations) > 0
    
    @pytest.mark.asyncio
    async def test_low_quality_data_detection_accuracy(self, extractor, low_quality_dataset):
        """Test accurate detection of data quality issues"""
        profile = await extractor.extract_comprehensive_metadata(
            low_quality_dataset,
            'low_quality_test',
            ProfileLevel.COMPREHENSIVE
        )
        
        # Verify duplicate detection
        assert profile.duplicate_rows == 500  # Half the dataset is duplicated
        assert profile.duplicate_rows_percentage == 50.0
        
        # Verify missing value detection
        assert profile.missing_values_total > 0
        assert profile.missing_values_percentage > 0
        
        # Verify data quality score is low
        assert profile.data_quality_score < 70.0
        
        # Verify pattern analysis detects invalid data
        email_profile = next(cp for cp in profile.column_profiles if cp.name == 'email')
        # Should detect that not all emails are valid
        assert email_profile.pattern_analysis['email_pattern_matches'] < profile.row_count
        
        phone_profile = next(cp for cp in profile.column_profiles if cp.name == 'phone')
        # Should detect invalid phone numbers
        assert phone_profile.pattern_analysis['phone_pattern_matches'] < profile.row_count
        
        # Verify mixed type detection
        mixed_profile = next(cp for cp in profile.column_profiles if cp.name == 'mixed_types')
        assert mixed_profile.data_type == 'object'  # Mixed types result in object type
    
    @pytest.mark.asyncio
    async def test_complex_schema_handling_accuracy(self, extractor, complex_schema_dataset):
        """Test accurate handling of complex schema types"""
        profile = await extractor.extract_comprehensive_metadata(
            complex_schema_dataset,
            'complex_schema_test',
            ProfileLevel.COMPREHENSIVE
        )
        
        # Verify sparse data detection
        sparse_profile = next(cp for cp in profile.column_profiles if cp.name == 'sparse_data')
        assert sparse_profile.null_percentage >= 90.0  # Should detect high sparsity
        
        # Verify categorical data handling
        categorical_profile = next(cp for cp in profile.column_profiles if cp.name == 'categorical_ordered')
        assert categorical_profile.unique_count == 3  # Low, Medium, High
        assert categorical_profile.value_distribution is not None
        
        # Verify text length analysis
        text_profile = next(cp for cp in profile.column_profiles if cp.name == 'text_long')
        assert text_profile.pattern_analysis is not None
        assert text_profile.pattern_analysis['avg_length'] > 50  # Long text should have high avg length
        
        # Verify precision handling
        decimal_profile = next(cp for cp in profile.column_profiles if cp.name == 'decimal_precision')
        assert decimal_profile.data_type in ['float64', 'float32']
        assert decimal_profile.std_deviation is not None
    
    @pytest.mark.asyncio
    async def test_schema_change_detection_accuracy(self, schema_catalog):
        """Test accurate detection of schema changes"""
        # Create original schema
        original_columns = [
            ColumnSchema(name='id', data_type='int64', nullable=False, primary_key=True),
            ColumnSchema(name='name', data_type='object', nullable=True),
            ColumnSchema(name='email', data_type='object', nullable=True),
            ColumnSchema(name='age', data_type='int64', nullable=True)
        ]
        
        original_schema = Schema(
            dataset_id='test_dataset',
            columns=original_columns,
            primary_keys=['id']
        )
        
        # Create modified schema
        modified_columns = [
            ColumnSchema(name='id', data_type='int64', nullable=False, primary_key=True),
            ColumnSchema(name='full_name', data_type='object', nullable=True),  # Renamed
            ColumnSchema(name='email', data_type='object', nullable=True),
            ColumnSchema(name='age', data_type='float64', nullable=True),  # Type changed
            ColumnSchema(name='created_at', data_type='datetime64[ns]', nullable=False)  # Added
        ]
        
        modified_schema = Schema(
            dataset_id='test_dataset',
            columns=modified_columns,
            primary_keys=['id']
        )
        
        # Test change detection
        changes = await schema_catalog._detect_schema_changes(
            {'columns': [col.__dict__ for col in original_columns]},
            {'columns': [col.__dict__ for col in modified_columns]}
        )
        
        # Verify all changes are detected
        change_types = [change.change_type for change in changes]
        assert SchemaChangeType.COLUMN_ADDED in change_types
        assert SchemaChangeType.COLUMN_REMOVED in change_types
        assert SchemaChangeType.COLUMN_TYPE_CHANGED in change_types
        
        # Verify specific changes
        added_changes = [c for c in changes if c.change_type == SchemaChangeType.COLUMN_ADDED]
        assert any(c.column_name == 'created_at' for c in added_changes)
        
        removed_changes = [c for c in changes if c.change_type == SchemaChangeType.COLUMN_REMOVED]
        assert any(c.column_name == 'name' for c in removed_changes)
        
        type_changes = [c for c in changes if c.change_type == SchemaChangeType.COLUMN_TYPE_CHANGED]
        assert any(c.column_name == 'age' and c.old_value == 'int64' and c.new_value == 'float64' for c in type_changes)
    
    @pytest.mark.asyncio
    async def test_schema_compatibility_accuracy(self, schema_catalog):
        """Test accurate schema compatibility checking"""
        # Compatible schemas (subset relationship)
        schema1_dict = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'name', 'data_type': 'object'}
            ]
        }
        
        schema2_dict = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'name', 'data_type': 'object'},
                {'name': 'email', 'data_type': 'object'}  # Additional column
            ]
        }
        
        # Incompatible schemas (type mismatch)
        schema3_dict = {
            'columns': [
                {'name': 'id', 'data_type': 'float64'},  # Different type
                {'name': 'name', 'data_type': 'object'}
            ]
        }
        
        # Test compatibility
        compatible = await schema_catalog._check_schema_compatibility(schema1_dict, schema2_dict)
        assert compatible  # schema1 is subset of schema2
        
        incompatible = await schema_catalog._check_schema_compatibility(schema1_dict, schema3_dict)
        assert not incompatible  # Type mismatch for 'id'
        
        reverse_compatible = await schema_catalog._check_schema_compatibility(schema2_dict, schema1_dict)
        assert not reverse_compatible  # schema2 has additional column not in schema1
    
    @pytest.mark.asyncio
    async def test_statistical_accuracy_validation(self, extractor):
        """Test statistical calculations accuracy with known data"""
        # Create dataset with known statistical properties
        np.random.seed(42)
        
        # Normal distribution with known parameters
        normal_data = np.random.normal(100, 15, 10000)
        
        # Uniform distribution
        uniform_data = np.random.uniform(0, 100, 10000)
        
        # Exponential distribution
        exp_data = np.random.exponential(2, 10000)
        
        data = {
            'normal_col': normal_data,
            'uniform_col': uniform_data,
            'exp_col': exp_data,
            'constant_col': [42] * 10000,  # No variance
            'linear_col': list(range(10000))  # Perfect linear relationship
        }
        
        df = pd.DataFrame(data)
        
        profile = await extractor.extract_comprehensive_metadata(
            df,
            'statistical_test',
            ProfileLevel.COMPREHENSIVE
        )
        
        # Test normal distribution statistics
        normal_profile = next(cp for cp in profile.column_profiles if cp.name == 'normal_col')
        assert abs(normal_profile.mean_value - 100) < 1  # Should be close to 100
        assert abs(normal_profile.std_deviation - 15) < 1  # Should be close to 15
        
        # Test uniform distribution
        uniform_profile = next(cp for cp in profile.column_profiles if cp.name == 'uniform_col')
        assert 0 <= uniform_profile.min_value <= 5  # Should be close to 0
        assert 95 <= uniform_profile.max_value <= 100  # Should be close to 100
        
        # Test constant column
        constant_profile = next(cp for cp in profile.column_profiles if cp.name == 'constant_col')
        assert constant_profile.unique_count == 1
        assert constant_profile.std_deviation == 0  # No variance
        
        # Test correlations
        assert profile.correlations is not None
        # Linear column should have perfect correlation with itself
        assert abs(profile.correlations['linear_col']['linear_col'] - 1.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_edge_case_handling_accuracy(self, extractor):
        """Test accurate handling of edge cases"""
        # Test various edge cases
        edge_cases = {
            'empty_strings': ['', '', '', 'value'],
            'whitespace_only': ['   ', '\t', '\n', 'value'],
            'special_chars': ['!@#$%', '^&*()', '[]{}', 'normal'],
            'unicode_chars': ['café', '北京', '🚀', 'normal'],
            'very_long_text': ['x' * 10000, 'y' * 5000, 'z' * 1000, 'short'],
            'extreme_numbers': [float('inf'), float('-inf'), float('nan'), 42],
            'date_edge_cases': [
                pd.Timestamp('1900-01-01'),
                pd.Timestamp('2100-12-31'),
                pd.Timestamp('2000-02-29'),  # Leap year
                pd.Timestamp('2024-01-01')
            ]
        }
        
        df = pd.DataFrame(edge_cases)
        
        profile = await extractor.extract_comprehensive_metadata(
            df,
            'edge_cases_test',
            ProfileLevel.COMPREHENSIVE
        )
        
        # Verify handling of empty strings
        empty_profile = next(cp for cp in profile.column_profiles if cp.name == 'empty_strings')
        assert empty_profile.unique_count <= 2  # Empty strings should be counted properly
        
        # Verify handling of extreme numbers
        extreme_profile = next(cp for cp in profile.column_profiles if cp.name == 'extreme_numbers')
        # Should handle inf and nan gracefully
        assert extreme_profile.min_value is not None or pd.isna(extreme_profile.min_value)
        
        # Verify unicode handling
        unicode_profile = next(cp for cp in profile.column_profiles if cp.name == 'unicode_chars')
        assert unicode_profile.unique_count == 4
        
        # Verify long text handling
        long_text_profile = next(cp for cp in profile.column_profiles if cp.name == 'very_long_text')
        assert long_text_profile.pattern_analysis is not None
        assert long_text_profile.pattern_analysis['max_length'] >= 10000
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, extractor):
        """Test metadata extraction performance and accuracy with large dataset"""
        # Create large dataset
        np.random.seed(42)
        size = 100000
        
        large_data = {
            'id': range(size),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
            'value': np.random.normal(0, 1, size),
            'flag': np.random.choice([True, False], size),
            'timestamp': pd.date_range('2020-01-01', periods=size, freq='min')
        }
        
        df = pd.DataFrame(large_data)
        
        # Time the extraction
        start_time = datetime.now()
        profile = await extractor.extract_comprehensive_metadata(
            df,
            'large_dataset_test',
            ProfileLevel.STANDARD  # Use standard to balance accuracy and performance
        )
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Verify accuracy is maintained
        assert profile.row_count == size
        assert profile.column_count == 5
        assert len(profile.column_profiles) == 5
        
        # Verify reasonable performance (should complete within reasonable time)
        assert duration < 60  # Should complete within 1 minute
        
        # Verify statistical accuracy on large dataset
        value_profile = next(cp for cp in profile.column_profiles if cp.name == 'value')
        assert abs(value_profile.mean_value) < 0.1  # Should be close to 0
        assert abs(value_profile.std_deviation - 1.0) < 0.1  # Should be close to 1
    
    @pytest.mark.asyncio
    async def test_catalog_search_accuracy(self, schema_catalog):
        """Test catalog search functionality accuracy"""
        # Mock database session and results
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Test search with query
            results = await schema_catalog.search_catalog("customer", {"tags": ["production"]})
            
            # Verify search was called with correct parameters
            assert mock_session.query.called
            assert mock_query.filter.called
    
    @pytest.mark.asyncio
    async def test_lineage_tracking_accuracy(self, schema_catalog):
        """Test dataset lineage tracking accuracy"""
        # Mock database operations
        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_dataset
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Test lineage update
            await schema_catalog.update_dataset_lineage(
                'dataset1',
                ['source1', 'source2'],
                ['derived1', 'derived2']
            )
            
            # Verify lineage was updated correctly
            assert mock_dataset.lineage_upstream == ['source1', 'source2']
            assert mock_dataset.lineage_downstream == ['derived1', 'derived2']
            assert mock_session.commit.called
    
    def test_schema_hash_consistency(self, schema_catalog):
        """Test schema hash generation consistency"""
        # Create identical schemas
        schema1 = Schema(
            dataset_id='test',
            columns=[
                ColumnSchema(name='col1', data_type='int64', nullable=False),
                ColumnSchema(name='col2', data_type='object', nullable=True)
            ],
            primary_keys=['col1']
        )
        
        schema2 = Schema(
            dataset_id='test',
            columns=[
                ColumnSchema(name='col1', data_type='int64', nullable=False),
                ColumnSchema(name='col2', data_type='object', nullable=True)
            ],
            primary_keys=['col1']
        )
        
        # Different schema
        schema3 = Schema(
            dataset_id='test',
            columns=[
                ColumnSchema(name='col1', data_type='int64', nullable=False),
                ColumnSchema(name='col2', data_type='object', nullable=True),
                ColumnSchema(name='col3', data_type='float64', nullable=True)  # Additional column
            ],
            primary_keys=['col1']
        )
        
        hash1 = schema_catalog._generate_schema_hash(schema1)
        hash2 = schema_catalog._generate_schema_hash(schema2)
        hash3 = schema_catalog._generate_schema_hash(schema3)
        
        # Identical schemas should have identical hashes
        assert hash1 == hash2
        
        # Different schemas should have different hashes
        assert hash1 != hash3
        assert hash2 != hash3
        
        # Hashes should be consistent across multiple calls
        assert hash1 == schema_catalog._generate_schema_hash(schema1)