"""
Schema Catalog Service for AI Data Readiness Platform

This module provides comprehensive schema cataloging, versioning, and management
capabilities with automatic change detection and lineage tracking.
"""

import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from .config import get_settings
from .exceptions import MetadataExtractionError
from ..models.catalog_models import (
    SchemaCatalogModel, DatasetProfileModel, SchemaChangeLogModel,
    DatasetCatalogModel, DatasetUsageModel, Schema, ColumnSchema,
    SchemaChange, SchemaChangeType, CatalogEntry, SchemaEvolution
)
from ..models.database import get_db_session


class SchemaCatalog:
    """
    Comprehensive schema catalog service with versioning and change tracking.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
    
    async def create_schema_version(
        self,
        dataset_id: str,
        schema: Schema,
        created_by: str,
        change_summary: str = "Schema version created",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new schema version in the catalog.
        
        Args:
            dataset_id: Unique identifier for the dataset
            schema: Schema definition
            created_by: User who created this version
            change_summary: Description of changes
            tags: Optional tags for this version
            
        Returns:
            str: Version ID of the created schema version
            
        Raises:
            MetadataExtractionError: If schema creation fails
        """
        try:
            async with get_db_session() as session:
                # Get existing versions
                existing_versions = await self._get_schema_versions_from_db(session, dataset_id)
                version_number = len(existing_versions) + 1
                
                # Generate schema hash for change detection
                schema_hash = self._generate_schema_hash(schema)
                
                # Check if this exact schema already exists
                existing_with_hash = session.query(SchemaCatalogModel).filter(
                    and_(
                        SchemaCatalogModel.dataset_id == dataset_id,
                        SchemaCatalogModel.schema_hash == schema_hash
                    )
                ).first()
                
                if existing_with_hash:
                    self.logger.info(f"Schema with hash {schema_hash} already exists for dataset {dataset_id}")
                    return existing_with_hash.version_id
                
                # Generate version ID
                version_id = f"{dataset_id}_v{version_number}_{schema_hash[:8]}"
                
                # Detect changes from previous version
                parent_version_id = None
                if existing_versions:
                    parent_version_id = existing_versions[-1].version_id
                    await self._log_schema_changes(
                        session, dataset_id, parent_version_id, version_id,
                        existing_versions[-1].schema_definition, asdict(schema), created_by
                    )
                
                # Create schema catalog entry
                schema_entry = SchemaCatalogModel(
                    dataset_id=dataset_id,
                    version_id=version_id,
                    version_number=version_number,
                    schema_definition=asdict(schema),
                    schema_hash=schema_hash,
                    created_by=created_by,
                    change_summary=change_summary,
                    parent_version_id=parent_version_id,
                    tags=tags or []
                )
                
                session.add(schema_entry)
                
                # Deactivate previous versions if this is not the first
                if existing_versions:
                    for version in existing_versions:
                        version.is_active = False
                
                session.commit()
                
                self.logger.info(f"Created schema version {version_id} for dataset {dataset_id}")
                return version_id
                
        except Exception as e:
            self.logger.error(f"Failed to create schema version: {str(e)}")
            raise MetadataExtractionError(f"Schema version creation failed: {str(e)}")
    
    async def get_schema_version(self, version_id: str) -> Optional[Schema]:
        """
        Get a specific schema version.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Schema object or None if not found
        """
        try:
            async with get_db_session() as session:
                version = session.query(SchemaCatalogModel).filter(
                    SchemaCatalogModel.version_id == version_id
                ).first()
                
                if not version:
                    return None
                
                # Convert from dict to Schema object
                schema_dict = version.schema_definition
                columns = [ColumnSchema(**col) for col in schema_dict['columns']]
                
                return Schema(
                    dataset_id=schema_dict['dataset_id'],
                    columns=columns,
                    primary_keys=schema_dict.get('primary_keys', []),
                    foreign_keys=schema_dict.get('foreign_keys', {}),
                    indexes=schema_dict.get('indexes', []),
                    constraints=schema_dict.get('constraints', []),
                    description=schema_dict.get('description')
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get schema version {version_id}: {str(e)}")
            return None
    
    async def get_latest_schema_version(self, dataset_id: str) -> Optional[Schema]:
        """
        Get the latest schema version for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Latest Schema object or None if not found
        """
        try:
            async with get_db_session() as session:
                latest_version = session.query(SchemaCatalogModel).filter(
                    SchemaCatalogModel.dataset_id == dataset_id
                ).order_by(desc(SchemaCatalogModel.version_number)).first()
                
                if not latest_version:
                    return None
                
                return await self._convert_to_schema(latest_version.schema_definition)
                
        except Exception as e:
            self.logger.error(f"Failed to get latest schema version for {dataset_id}: {str(e)}")
            return None
    
    async def get_schema_evolution(self, dataset_id: str) -> Optional[SchemaEvolution]:
        """
        Get complete schema evolution history for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            SchemaEvolution object with complete history
        """
        try:
            async with get_db_session() as session:
                # Get all versions
                versions = session.query(SchemaCatalogModel).filter(
                    SchemaCatalogModel.dataset_id == dataset_id
                ).order_by(SchemaCatalogModel.version_number).all()
                
                if not versions:
                    return None
                
                # Get all changes
                changes = session.query(SchemaChangeLogModel).filter(
                    SchemaChangeLogModel.dataset_id == dataset_id
                ).order_by(SchemaChangeLogModel.created_at).all()
                
                # Build evolution timeline
                timeline = []
                for version in versions:
                    timeline.append({
                        'version_id': version.version_id,
                        'version_number': version.version_number,
                        'created_at': version.created_at,
                        'created_by': version.created_by,
                        'change_summary': version.change_summary,
                        'schema_hash': version.schema_hash
                    })
                
                # Build compatibility matrix
                compatibility_matrix = await self._build_compatibility_matrix(versions)
                
                # Convert changes to SchemaChange objects
                schema_changes = []
                for change in changes:
                    schema_changes.append(SchemaChange(
                        change_type=SchemaChangeType(change.change_type),
                        column_name=change.column_name,
                        old_value=change.old_value,
                        new_value=change.new_value,
                        description=change.description
                    ))
                
                return SchemaEvolution(
                    dataset_id=dataset_id,
                    versions=[asdict(v) for v in versions],
                    changes=schema_changes,
                    evolution_timeline=timeline,
                    compatibility_matrix=compatibility_matrix
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get schema evolution for {dataset_id}: {str(e)}")
            return None
    
    async def search_catalog(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CatalogEntry]:
        """
        Search the schema catalog.
        
        Args:
            query: Search query string
            filters: Optional filters (tags, created_by, etc.)
            
        Returns:
            List of matching catalog entries
        """
        try:
            async with get_db_session() as session:
                # Build base query
                base_query = session.query(DatasetCatalogModel)
                
                # Apply text search
                if query:
                    base_query = base_query.filter(
                        DatasetCatalogModel.name.ilike(f'%{query}%') |
                        DatasetCatalogModel.description.ilike(f'%{query}%')
                    )
                
                # Apply filters
                if filters:
                    if 'tags' in filters:
                        # JSON contains filter for tags
                        for tag in filters['tags']:
                            base_query = base_query.filter(
                                DatasetCatalogModel.tags.contains([tag])
                            )
                    
                    if 'owner' in filters:
                        base_query = base_query.filter(
                            DatasetCatalogModel.owner == filters['owner']
                        )
                    
                    if 'data_classification' in filters:
                        base_query = base_query.filter(
                            DatasetCatalogModel.data_classification == filters['data_classification']
                        )
                
                results = base_query.all()
                
                # Convert to CatalogEntry objects
                catalog_entries = []
                for result in results:
                    # Get schema versions for this dataset
                    schema_versions = session.query(SchemaCatalogModel).filter(
                        SchemaCatalogModel.dataset_id == result.dataset_id
                    ).order_by(SchemaCatalogModel.version_number).all()
                    
                    # Get latest profile
                    latest_profile = session.query(DatasetProfileModel).filter(
                        DatasetProfileModel.dataset_id == result.dataset_id
                    ).order_by(desc(DatasetProfileModel.created_at)).first()
                    
                    profile_summary = {}
                    if latest_profile:
                        profile_summary = {
                            'row_count': latest_profile.row_count,
                            'column_count': latest_profile.column_count,
                            'data_quality_score': latest_profile.data_quality_score,
                            'missing_values_percentage': latest_profile.missing_values_percentage
                        }
                    
                    catalog_entry = CatalogEntry(
                        dataset_id=str(result.dataset_id),
                        name=result.name,
                        description=result.description or "",
                        current_schema_version=schema_versions[-1].version_id if schema_versions else "",
                        schema_versions=[v.version_id for v in schema_versions],
                        profile_summary=profile_summary,
                        usage_statistics=result.usage_statistics or {},
                        lineage={
                            'upstream': result.lineage_upstream or [],
                            'downstream': result.lineage_downstream or []
                        },
                        tags=result.tags or [],
                        created_at=result.created_at,
                        updated_at=result.updated_at,
                        last_accessed_at=result.last_accessed_at,
                        access_count=result.access_count
                    )
                    
                    catalog_entries.append(catalog_entry)
                
                return catalog_entries
                
        except Exception as e:
            self.logger.error(f"Catalog search failed: {str(e)}")
            return []
    
    async def register_dataset(
        self,
        dataset_id: str,
        name: str,
        description: str,
        owner: str,
        source: str,
        format: str,
        tags: Optional[List[str]] = None,
        data_classification: str = "internal",
        business_glossary: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a new dataset in the catalog.
        
        Args:
            dataset_id: Unique dataset identifier
            name: Human-readable dataset name
            description: Dataset description
            owner: Dataset owner
            source: Data source information
            format: Data format (CSV, JSON, etc.)
            tags: Optional tags
            data_classification: Data classification level
            business_glossary: Business terms and definitions
            
        Returns:
            Dataset ID of registered dataset
        """
        try:
            async with get_db_session() as session:
                # Check if dataset already exists
                existing = session.query(DatasetCatalogModel).filter(
                    DatasetCatalogModel.dataset_id == dataset_id
                ).first()
                
                if existing:
                    self.logger.warning(f"Dataset {dataset_id} already registered")
                    return str(existing.dataset_id)
                
                # Create new catalog entry
                catalog_entry = DatasetCatalogModel(
                    dataset_id=dataset_id,
                    name=name,
                    description=description,
                    source=source,
                    format=format,
                    owner=owner,
                    tags=tags or [],
                    business_glossary=business_glossary or {},
                    data_classification=data_classification,
                    retention_policy={},
                    access_permissions={},
                    usage_statistics={},
                    lineage_upstream=[],
                    lineage_downstream=[]
                )
                
                session.add(catalog_entry)
                session.commit()
                
                self.logger.info(f"Registered dataset {dataset_id} in catalog")
                return str(catalog_entry.dataset_id)
                
        except Exception as e:
            self.logger.error(f"Failed to register dataset {dataset_id}: {str(e)}")
            raise MetadataExtractionError(f"Dataset registration failed: {str(e)}")
    
    async def update_dataset_lineage(
        self,
        dataset_id: str,
        upstream_datasets: List[str],
        downstream_datasets: List[str]
    ):
        """
        Update dataset lineage information.
        
        Args:
            dataset_id: Dataset identifier
            upstream_datasets: List of upstream dataset IDs
            downstream_datasets: List of downstream dataset IDs
        """
        try:
            async with get_db_session() as session:
                dataset = session.query(DatasetCatalogModel).filter(
                    DatasetCatalogModel.dataset_id == dataset_id
                ).first()
                
                if not dataset:
                    raise MetadataExtractionError(f"Dataset {dataset_id} not found")
                
                dataset.lineage_upstream = upstream_datasets
                dataset.lineage_downstream = downstream_datasets
                dataset.updated_at = datetime.utcnow()
                
                session.commit()
                
                self.logger.info(f"Updated lineage for dataset {dataset_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to update lineage for {dataset_id}: {str(e)}")
            raise MetadataExtractionError(f"Lineage update failed: {str(e)}")
    
    async def track_dataset_usage(
        self,
        dataset_id: str,
        user_id: str,
        operation: str,
        duration_seconds: Optional[float] = None,
        rows_processed: Optional[int] = None,
        bytes_processed: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track dataset usage for analytics and governance.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User performing the operation
            operation: Type of operation (read, write, transform, analyze)
            duration_seconds: Operation duration
            rows_processed: Number of rows processed
            bytes_processed: Number of bytes processed
            success: Whether operation was successful
            error_message: Error message if operation failed
            metadata: Additional operation metadata
        """
        try:
            async with get_db_session() as session:
                # Record usage
                usage_record = DatasetUsageModel(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    operation=operation,
                    duration_seconds=duration_seconds,
                    rows_processed=rows_processed,
                    bytes_processed=bytes_processed,
                    success=success,
                    error_message=error_message,
                    metadata=metadata or {}
                )
                
                session.add(usage_record)
                
                # Update dataset access statistics
                dataset = session.query(DatasetCatalogModel).filter(
                    DatasetCatalogModel.dataset_id == dataset_id
                ).first()
                
                if dataset:
                    dataset.increment_access()
                
                session.commit()
                
                self.logger.debug(f"Tracked usage for dataset {dataset_id} by user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to track usage for {dataset_id}: {str(e)}")
            # Don't raise exception for usage tracking failures
    
    def _generate_schema_hash(self, schema: Schema) -> str:
        """Generate hash for schema to detect changes"""
        schema_dict = asdict(schema)
        schema_json = json.dumps(schema_dict, sort_keys=True)
        return hashlib.md5(schema_json.encode()).hexdigest()
    
    async def _get_schema_versions_from_db(self, session: Session, dataset_id: str) -> List[SchemaCatalogModel]:
        """Get existing schema versions for a dataset from database"""
        return session.query(SchemaCatalogModel).filter(
            SchemaCatalogModel.dataset_id == dataset_id
        ).order_by(SchemaCatalogModel.version_number).all()
    
    async def _log_schema_changes(
        self,
        session: Session,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any],
        created_by: str
    ):
        """Log schema changes between versions"""
        try:
            changes = await self._detect_schema_changes(old_schema, new_schema)
            
            for change in changes:
                change_log = SchemaChangeLogModel(
                    dataset_id=dataset_id,
                    from_version_id=from_version_id,
                    to_version_id=to_version_id,
                    change_type=change.change_type.value,
                    column_name=change.column_name,
                    old_value=change.old_value,
                    new_value=change.new_value,
                    description=change.description,
                    created_by=created_by
                )
                
                session.add(change_log)
                
        except Exception as e:
            self.logger.error(f"Failed to log schema changes: {str(e)}")
    
    async def _detect_schema_changes(
        self, 
        old_schema: Dict[str, Any], 
        new_schema: Dict[str, Any]
    ) -> List[SchemaChange]:
        """Detect changes between two schema versions"""
        changes = []
        
        old_columns = {col['name']: col for col in old_schema.get('columns', [])}
        new_columns = {col['name']: col for col in new_schema.get('columns', [])}
        
        # Check for new columns
        for col_name in set(new_columns.keys()) - set(old_columns.keys()):
            changes.append(SchemaChange(
                change_type=SchemaChangeType.COLUMN_ADDED,
                column_name=col_name,
                new_value=new_columns[col_name],
                description=f"Added column {col_name}"
            ))
        
        # Check for removed columns
        for col_name in set(old_columns.keys()) - set(new_columns.keys()):
            changes.append(SchemaChange(
                change_type=SchemaChangeType.COLUMN_REMOVED,
                column_name=col_name,
                old_value=old_columns[col_name],
                description=f"Removed column {col_name}"
            ))
        
        # Check for type changes
        for col_name in set(old_columns.keys()) & set(new_columns.keys()):
            old_col = old_columns[col_name]
            new_col = new_columns[col_name]
            
            if old_col['data_type'] != new_col['data_type']:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
                    column_name=col_name,
                    old_value=old_col['data_type'],
                    new_value=new_col['data_type'],
                    description=f"Changed type for {col_name}: {old_col['data_type']} -> {new_col['data_type']}"
                ))
        
        return changes
    
    async def _build_compatibility_matrix(self, versions: List[SchemaCatalogModel]) -> Dict[str, Dict[str, bool]]:
        """Build compatibility matrix between schema versions"""
        matrix = {}
        
        for i, version1 in enumerate(versions):
            matrix[version1.version_id] = {}
            
            for j, version2 in enumerate(versions):
                if i == j:
                    matrix[version1.version_id][version2.version_id] = True
                else:
                    # Simple compatibility check - same columns and types
                    compatible = await self._check_schema_compatibility(
                        version1.schema_definition,
                        version2.schema_definition
                    )
                    matrix[version1.version_id][version2.version_id] = compatible
        
        return matrix
    
    async def _check_schema_compatibility(
        self, 
        schema1: Dict[str, Any], 
        schema2: Dict[str, Any]
    ) -> bool:
        """Check if two schemas are compatible"""
        try:
            cols1 = {col['name']: col['data_type'] for col in schema1.get('columns', [])}
            cols2 = {col['name']: col['data_type'] for col in schema2.get('columns', [])}
            
            # Schemas are compatible if all columns in schema1 exist in schema2 with same types
            for col_name, data_type in cols1.items():
                if col_name not in cols2 or cols2[col_name] != data_type:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _convert_to_schema(self, schema_dict: Dict[str, Any]) -> Schema:
        """Convert schema dictionary to Schema object"""
        columns = [ColumnSchema(**col) for col in schema_dict['columns']]
        
        return Schema(
            dataset_id=schema_dict['dataset_id'],
            columns=columns,
            primary_keys=schema_dict.get('primary_keys', []),
            foreign_keys=schema_dict.get('foreign_keys', {}),
            indexes=schema_dict.get('indexes', []),
            constraints=schema_dict.get('constraints', []),
            description=schema_dict.get('description')
        )
             