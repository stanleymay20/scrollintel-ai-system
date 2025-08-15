"""
Tests for Schema Catalog - AI Data Readiness Platform

This module tests the schema catalog functionality including versioning,
change tracking, and catalog management capabilities.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.orm import Session

from ai_data_readiness.core.schema_catalog import SchemaCatalog
from ai_data_readiness.core.exceptions import MetadataExtractionError
from ai_data_readiness.models.catalog_models import (
    Schema, ColumnSchema, SchemaChange, SchemaChangeType,
    CatalogEntry, SchemaEvolution, SchemaCatalogModel,
    DatasetCatalogModel, DatasetUsageModel, SchemaChangeLogModel
)


class TestSchemaCatalog:
    """Test suite for SchemaCatalog"""
    
    @pytest.fixture
    def catalog(self):
        """Create SchemaCatalog instance"""
        return SchemaCatalog()
    
    @pytest.fixture
    def sample_schema(self):
        """Create sample schema for testing"""
        columns = [
            ColumnSchema(
                name='id',
                data_type='int64',
                nullable=False,
                unique=True,
                primary_key=True,
                description='Primary key'
            ),
            ColumnSchema(
                name='name',
                data_type='object',
                nullable=True,
                description='Customer name'
            ),
            ColumnSchema(
                name='email',
                data_type='object',
                nullable=True,
                unique=True,
                description='Customer email'
            ),
            ColumnSchema(
                name='created_at',
                data_type='datetime64[ns]',
                nullable=False,
                description='Creation timestamp'
            )
        ]
        
        return Schema(
            dataset_id='test_dataset',
            columns=columns,
            primary_keys=['id'],
            foreign_keys={},
            indexes=[{'name': 'idx_email', 'columns': ['email']}],
            constraints=[],
            description='Test dataset schema'
        )
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        session = MagicMock(spec=Session)
        session.query.return_value = session
        session.filter.return_value = session
        session.order_by.return_value = session
        session.first.return_value = None
        session.all.return_value = []
        session.add = MagicMock()
        session.commit = MagicMock()
        return session
    
    @pytest.mark.asyncio
    async def test_create_schema_version_new_dataset(self, catalog, sample_schema, mock_db_session):
        """Test creating first schema version for new dataset"""
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock no existing versions
            mock_db_session.all.return_value = []
            
            version_id = await catalog.create_schema_version(
                dataset_id='test_dataset',
                schema=sample_schema,
                created_by='test_user',
                change_summary='Initial schema creation',
                tags=['production', 'customer_data']
            )
            
            # Verify version ID format
            assert version_id.startswith('test_dataset_v1_')
            assert len(version_id.split('_')) == 4  # dataset_id_v1_hash
            
            # Verify database operations
            assert mock_db_session.add.called
            assert mock_db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_create_schema_version_existing_dataset(self, catalog, sample_schema, mock_db_session):
        """Test creating new schema version for existing dataset"""
        # Mock existing version
        existing_version = MagicMock()
        existing_version.version_id = 'test_dataset_v1_abcd1234'
        existing_version.version_number = 1
        existing_version.schema_definition = {
            'dataset_id': 'test_dataset',
            'columns': [
                {'name': 'id', 'data_type': 'int64', 'nullable': False},
                {'name': 'name', 'data_type': 'object', 'nullable': True}
            ]
        }
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock existing versions
            mock_db_session.all.return_value = [existing_version]
            
            version_id = await catalog.create_schema_version(
                dataset_id='test_dataset',
                schema=sample_schema,
                created_by='test_user',
                change_summary='Added email and created_at columns'
            )
            
            # Verify version number incremented
            assert 'v2_' in version_id
            
            # Verify parent version is set
            call_args = mock_db_session.add.call_args[0][0]
            assert call_args.parent_version_id == 'test_dataset_v1_abcd1234'
    
    @pytest.mark.asyncio
    async def test_get_schema_version(self, catalog, mock_db_session):
        """Test retrieving specific schema version"""
        # Mock schema version in database
        mock_version = MagicMock()
        mock_version.schema_definition = {
            'dataset_id': 'test_dataset',
            'columns': [
                {
                    'name': 'id',
                    'data_type': 'int64',
                    'nullable': False,
                    'unique': True,
                    'primary_key': True,
                    'foreign_key': None,
                    'constraints': [],
                    'description': 'Primary key'
                }
            ],
            'primary_keys': ['id'],
            'foreign_keys': {},
            'indexes': [],
            'constraints': [],
            'description': 'Test schema'
        }
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.first.return_value = mock_version
            
            schema = await catalog.get_schema_version('test_dataset_v1_abcd1234')
            
            assert schema is not None
            assert schema.dataset_id == 'test_dataset'
            assert len(schema.columns) == 1
            assert schema.columns[0].name == 'id'
            assert schema.columns[0].primary_key is True
    
    @pytest.mark.asyncio
    async def test_get_latest_schema_version(self, catalog, mock_db_session):
        """Test retrieving latest schema version"""
        mock_version = MagicMock()
        mock_version.schema_definition = {
            'dataset_id': 'test_dataset',
            'columns': [{'name': 'id', 'data_type': 'int64', 'nullable': False}],
            'primary_keys': [],
            'foreign_keys': {},
            'indexes': [],
            'constraints': []
        }
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.first.return_value = mock_version
            
            schema = await catalog.get_latest_schema_version('test_dataset')
            
            assert schema is not None
            assert schema.dataset_id == 'test_dataset'
            
            # Verify query used descending order
            mock_db_session.order_by.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_schema_evolution(self, catalog, mock_db_session):
        """Test retrieving complete schema evolution"""
        # Mock versions
        version1 = MagicMock()
        version1.version_id = 'test_v1'
        version1.version_number = 1
        version1.created_at = datetime(2024, 1, 1)
        version1.created_by = 'user1'
        version1.change_summary = 'Initial'
        version1.schema_hash = 'hash1'
        
        version2 = MagicMock()
        version2.version_id = 'test_v2'
        version2.version_number = 2
        version2.created_at = datetime(2024, 1, 2)
        version2.created_by = 'user2'
        version2.change_summary = 'Added column'
        version2.schema_hash = 'hash2'
        
        # Mock changes
        change1 = MagicMock()
        change1.change_type = 'column_added'
        change1.column_name = 'email'
        change1.old_value = None
        change1.new_value = {'name': 'email', 'data_type': 'object'}
        change1.description = 'Added email column'
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock query results
            mock_db_session.all.side_effect = [
                [version1, version2],  # versions query
                [change1]  # changes query
            ]
            
            with patch.object(catalog, '_build_compatibility_matrix', return_value={}):
                evolution = await catalog.get_schema_evolution('test_dataset')
            
            assert evolution is not None
            assert evolution.dataset_id == 'test_dataset'
            assert len(evolution.versions) == 2
            assert len(evolution.changes) == 1
            assert len(evolution.evolution_timeline) == 2
            
            # Verify timeline order
            assert evolution.evolution_timeline[0]['version_id'] == 'test_v1'
            assert evolution.evolution_timeline[1]['version_id'] == 'test_v2'
    
    @pytest.mark.asyncio
    async def test_search_catalog(self, catalog, mock_db_session):
        """Test catalog search functionality"""
        # Mock dataset catalog entry
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = 'test_dataset'
        mock_dataset.name = 'Test Dataset'
        mock_dataset.description = 'A test dataset'
        mock_dataset.tags = ['production', 'customer']
        mock_dataset.owner = 'test_user'
        mock_dataset.data_classification = 'internal'
        mock_dataset.usage_statistics = {'access_count': 100}
        mock_dataset.lineage_upstream = ['source1']
        mock_dataset.lineage_downstream = ['derived1']
        mock_dataset.created_at = datetime(2024, 1, 1)
        mock_dataset.updated_at = datetime(2024, 1, 2)
        mock_dataset.last_accessed_at = datetime(2024, 1, 3)
        mock_dataset.access_count = 50
        
        # Mock schema version
        mock_schema_version = MagicMock()
        mock_schema_version.version_id = 'test_v1'
        
        # Mock profile
        mock_profile = MagicMock()
        mock_profile.row_count = 1000
        mock_profile.column_count = 5
        mock_profile.data_quality_score = 95.5
        mock_profile.missing_values_percentage = 2.1
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock query chain
            mock_query = MagicMock()
            mock_db_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [mock_dataset]
            
            # Mock additional queries for schema versions and profiles
            mock_db_session.query.side_effect = [
                mock_query,  # Main search query
                MagicMock(filter=MagicMock(return_value=MagicMock(order_by=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_schema_version])))))),  # Schema versions
                MagicMock(filter=MagicMock(return_value=MagicMock(order_by=MagicMock(return_value=MagicMock(first=MagicMock(return_value=mock_profile))))))  # Profile
            ]
            
            results = await catalog.search_catalog(
                query='test',
                filters={'tags': ['production'], 'owner': 'test_user'}
            )
            
            assert len(results) == 1
            result = results[0]
            
            assert result.dataset_id == 'test_dataset'
            assert result.name == 'Test Dataset'
            assert 'production' in result.tags
            assert result.profile_summary['row_count'] == 1000
            assert result.profile_summary['data_quality_score'] == 95.5
    
    @pytest.mark.asyncio
    async def test_register_dataset(self, catalog, mock_db_session):
        """Test dataset registration"""
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock no existing dataset
            mock_db_session.first.return_value = None
            
            dataset_id = await catalog.register_dataset(
                dataset_id='new_dataset',
                name='New Dataset',
                description='A new dataset for testing',
                owner='test_user',
                source='test_source',
                format='CSV',
                tags=['test', 'development'],
                data_classification='internal',
                business_glossary={'customer': 'A person who buys products'}
            )
            
            assert dataset_id == 'new_dataset'
            assert mock_db_session.add.called
            assert mock_db_session.commit.called
            
            # Verify the dataset object passed to add
            added_dataset = mock_db_session.add.call_args[0][0]
            assert added_dataset.name == 'New Dataset'
            assert added_dataset.owner == 'test_user'
            assert 'test' in added_dataset.tags
    
    @pytest.mark.asyncio
    async def test_register_existing_dataset(self, catalog, mock_db_session):
        """Test registering dataset that already exists"""
        existing_dataset = MagicMock()
        existing_dataset.dataset_id = 'existing_dataset'
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.first.return_value = existing_dataset
            
            dataset_id = await catalog.register_dataset(
                dataset_id='existing_dataset',
                name='Existing Dataset',
                description='Already exists',
                owner='test_user',
                source='test_source',
                format='CSV'
            )
            
            assert dataset_id == 'existing_dataset'
            # Should not add new entry
            assert not mock_db_session.add.called
    
    @pytest.mark.asyncio
    async def test_update_dataset_lineage(self, catalog, mock_db_session):
        """Test updating dataset lineage"""
        mock_dataset = MagicMock()
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.first.return_value = mock_dataset
            
            await catalog.update_dataset_lineage(
                dataset_id='test_dataset',
                upstream_datasets=['source1', 'source2'],
                downstream_datasets=['derived1', 'derived2']
            )
            
            assert mock_dataset.lineage_upstream == ['source1', 'source2']
            assert mock_dataset.lineage_downstream == ['derived1', 'derived2']
            assert mock_dataset.updated_at is not None
            assert mock_db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_update_lineage_nonexistent_dataset(self, catalog, mock_db_session):
        """Test updating lineage for non-existent dataset"""
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.first.return_value = None
            
            with pytest.raises(MetadataExtractionError, match="Dataset .* not found"):
                await catalog.update_dataset_lineage(
                    dataset_id='nonexistent',
                    upstream_datasets=[],
                    downstream_datasets=[]
                )
    
    @pytest.mark.asyncio
    async def test_track_dataset_usage(self, catalog, mock_db_session):
        """Test tracking dataset usage"""
        mock_dataset = MagicMock()
        mock_dataset.access_count = 10
        
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.first.return_value = mock_dataset
            
            await catalog.track_dataset_usage(
                dataset_id='test_dataset',
                user_id='test_user',
                operation='read',
                duration_seconds=5.2,
                rows_processed=1000,
                bytes_processed=50000,
                success=True,
                metadata={'query_type': 'select'}
            )
            
            # Verify usage record was added
            assert mock_db_session.add.called
            usage_record = mock_db_session.add.call_args[0][0]
            assert usage_record.dataset_id == 'test_dataset'
            assert usage_record.user_id == 'test_user'
            assert usage_record.operation == 'read'
            assert usage_record.duration_seconds == 5.2
            assert usage_record.success is True
            
            # Verify dataset access count was incremented
            assert mock_dataset.access_count == 11
            assert mock_dataset.last_accessed_at is not None
            
            assert mock_db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_detect_schema_changes(self, catalog):
        """Test schema change detection"""
        old_schema = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'name', 'data_type': 'object'},
                {'name': 'email', 'data_type': 'object'}
            ]
        }
        
        new_schema = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'full_name', 'data_type': 'object'},  # Renamed from 'name'
                {'name': 'email', 'data_type': 'object'},
                {'name': 'phone', 'data_type': 'object'},  # Added
                {'name': 'age', 'data_type': 'int64'}  # Added
            ]
        }
        
        changes = await catalog._detect_schema_changes(old_schema, new_schema)
        
        # Should detect: removed 'name', added 'full_name', added 'phone', added 'age'
        change_types = [change.change_type for change in changes]
        
        assert SchemaChangeType.COLUMN_REMOVED in change_types
        assert SchemaChangeType.COLUMN_ADDED in change_types
        
        # Check specific changes
        removed_changes = [c for c in changes if c.change_type == SchemaChangeType.COLUMN_REMOVED]
        added_changes = [c for c in changes if c.change_type == SchemaChangeType.COLUMN_ADDED]
        
        assert any(c.column_name == 'name' for c in removed_changes)
        assert any(c.column_name == 'full_name' for c in added_changes)
        assert any(c.column_name == 'phone' for c in added_changes)
        assert any(c.column_name == 'age' for c in added_changes)
    
    @pytest.mark.asyncio
    async def test_detect_type_changes(self, catalog):
        """Test detection of column type changes"""
        old_schema = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'amount', 'data_type': 'int64'},
                {'name': 'status', 'data_type': 'object'}
            ]
        }
        
        new_schema = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'amount', 'data_type': 'float64'},  # Type changed
                {'name': 'status', 'data_type': 'category'}  # Type changed
            ]
        }
        
        changes = await catalog._detect_schema_changes(old_schema, new_schema)
        
        type_changes = [c for c in changes if c.change_type == SchemaChangeType.COLUMN_TYPE_CHANGED]
        
        assert len(type_changes) == 2
        
        amount_change = next(c for c in type_changes if c.column_name == 'amount')
        assert amount_change.old_value == 'int64'
        assert amount_change.new_value == 'float64'
        
        status_change = next(c for c in type_changes if c.column_name == 'status')
        assert status_change.old_value == 'object'
        assert status_change.new_value == 'category'
    
    @pytest.mark.asyncio
    async def test_check_schema_compatibility(self, catalog):
        """Test schema compatibility checking"""
        # Compatible schemas (subset)
        schema1 = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'name', 'data_type': 'object'}
            ]
        }
        
        schema2 = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'name', 'data_type': 'object'},
                {'name': 'email', 'data_type': 'object'}
            ]
        }
        
        # Incompatible schemas (type mismatch)
        schema3 = {
            'columns': [
                {'name': 'id', 'data_type': 'float64'},  # Different type
                {'name': 'name', 'data_type': 'object'}
            ]
        }
        
        # Test compatibility
        assert await catalog._check_schema_compatibility(schema1, schema2) is True
        assert await catalog._check_schema_compatibility(schema1, schema3) is False
        assert await catalog._check_schema_compatibility(schema2, schema1) is False  # schema2 has extra column
    
    def test_schema_hash_generation(self, catalog, sample_schema):
        """Test schema hash generation"""
        hash1 = catalog._generate_schema_hash(sample_schema)
        hash2 = catalog._generate_schema_hash(sample_schema)
        
        # Same schema should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Different schema should produce different hash
        modified_schema = Schema(
            dataset_id='test_dataset',
            columns=sample_schema.columns + [
                ColumnSchema(name='new_col', data_type='object', nullable=True)
            ],
            primary_keys=sample_schema.primary_keys
        )
        
        hash3 = catalog._generate_schema_hash(modified_schema)
        assert hash1 != hash3
    
    @pytest.mark.asyncio
    async def test_build_compatibility_matrix(self, catalog):
        """Test building compatibility matrix"""
        # Mock schema versions
        version1 = MagicMock()
        version1.version_id = 'v1'
        version1.schema_definition = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'name', 'data_type': 'object'}
            ]
        }
        
        version2 = MagicMock()
        version2.version_id = 'v2'
        version2.schema_definition = {
            'columns': [
                {'name': 'id', 'data_type': 'int64'},
                {'name': 'name', 'data_type': 'object'},
                {'name': 'email', 'data_type': 'object'}
            ]
        }
        
        version3 = MagicMock()
        version3.version_id = 'v3'
        version3.schema_definition = {
            'columns': [
                {'name': 'id', 'data_type': 'float64'},  # Incompatible type change
                {'name': 'name', 'data_type': 'object'}
            ]
        }
        
        versions = [version1, version2, version3]
        
        matrix = await catalog._build_compatibility_matrix(versions)
        
        # Self-compatibility
        assert matrix['v1']['v1'] is True
        assert matrix['v2']['v2'] is True
        assert matrix['v3']['v3'] is True
        
        # v1 compatible with v2 (subset)
        assert matrix['v1']['v2'] is True
        
        # v2 not compatible with v1 (superset)
        assert matrix['v2']['v1'] is False
        
        # v1 not compatible with v3 (type mismatch)
        assert matrix['v1']['v3'] is False
        assert matrix['v3']['v1'] is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self, catalog):
        """Test error handling in catalog operations"""
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            # Mock database error
            mock_get_session.side_effect = Exception("Database connection failed")
            
            with pytest.raises(MetadataExtractionError):
                await catalog.create_schema_version(
                    dataset_id='test',
                    schema=MagicMock(),
                    created_by='user',
                    change_summary='test'
                )
    
    @pytest.mark.asyncio
    async def test_usage_tracking_error_handling(self, catalog, mock_db_session):
        """Test that usage tracking errors don't raise exceptions"""
        with patch('ai_data_readiness.core.schema_catalog.get_db_session') as mock_get_session:
            # Mock database error
            mock_get_session.side_effect = Exception("Database error")
            
            # Should not raise exception
            await catalog.track_dataset_usage(
                dataset_id='test',
                user_id='user',
                operation='read'
            )
    
    @pytest.mark.asyncio
    async def test_convert_to_schema(self, catalog):
        """Test conversion from dictionary to Schema object"""
        schema_dict = {
            'dataset_id': 'test_dataset',
            'columns': [
                {
                    'name': 'id',
                    'data_type': 'int64',
                    'nullable': False,
                    'unique': True,
                    'primary_key': True,
                    'foreign_key': None,
                    'constraints': [],
                    'description': 'Primary key'
                }
            ],
            'primary_keys': ['id'],
            'foreign_keys': {},
            'indexes': [],
            'constraints': [],
            'description': 'Test schema'
        }
        
        schema = await catalog._convert_to_schema(schema_dict)
        
        assert isinstance(schema, Schema)
        assert schema.dataset_id == 'test_dataset'
        assert len(schema.columns) == 1
        assert isinstance(schema.columns[0], ColumnSchema)
        assert schema.columns[0].name == 'id'
        assert schema.columns[0].primary_key is True
        assert schema.primary_keys == ['id']
        assert schema.description == 'Test schema'