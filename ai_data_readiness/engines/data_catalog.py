"""Data catalog engine for governance metadata management."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from ..models.governance_models import (
    DataCatalogEntry, DataClassification, GovernanceMetrics
)
from ..models.governance_database import (
    DataCatalogEntryModel, DatasetModel, UserModel
)
from ..models.database import get_db_session
from ..core.exceptions import AIDataReadinessError


logger = logging.getLogger(__name__)


class DataCatalogError(AIDataReadinessError):
    """Exception raised for data catalog errors."""
    pass


class DataCatalog:
    """Data catalog for managing dataset metadata and governance information."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def register_dataset(
        self,
        dataset_id: str,
        name: str,
        description: str,
        owner: str,
        classification: DataClassification = DataClassification.INTERNAL,
        steward: Optional[str] = None,
        tags: Optional[List[str]] = None,
        business_terms: Optional[List[str]] = None
    ) -> DataCatalogEntry:
        """Register a dataset in the catalog."""
        try:
            with get_db_session() as session:
                # Check if dataset exists
                dataset = session.query(DatasetModel).filter(
                    DatasetModel.id == dataset_id
                ).first()
                
                if not dataset:
                    raise DataCatalogError(f"Dataset {dataset_id} not found")
                
                # Check if catalog entry already exists
                existing_entry = session.query(DataCatalogEntryModel).filter(
                    DataCatalogEntryModel.dataset_id == dataset_id
                ).first()
                
                if existing_entry:
                    raise DataCatalogError(f"Dataset {dataset_id} already registered in catalog")
                
                # Create catalog entry
                catalog_entry = DataCatalogEntryModel(
                    dataset_id=dataset_id,
                    name=name,
                    description=description,
                    classification=classification.value,
                    owner=owner,
                    steward=steward,
                    business_glossary_terms=business_terms or [],
                    tags=tags or [],
                    schema_info=self._extract_schema_info(dataset),
                    quality_metrics={}
                )
                
                session.add(catalog_entry)
                session.commit()
                session.refresh(catalog_entry)
                
                self.logger.info(f"Dataset {dataset_id} registered in catalog")
                
                return self._model_to_dataclass(catalog_entry)
                
        except Exception as e:
            self.logger.error(f"Error registering dataset in catalog: {str(e)}")
            raise DataCatalogError(f"Failed to register dataset: {str(e)}")
    
    def update_catalog_entry(
        self,
        dataset_id: str,
        updates: Dict[str, Any]
    ) -> DataCatalogEntry:
        """Update a catalog entry."""
        try:
            with get_db_session() as session:
                catalog_entry = session.query(DataCatalogEntryModel).filter(
                    DataCatalogEntryModel.dataset_id == dataset_id
                ).first()
                
                if not catalog_entry:
                    raise DataCatalogError(f"Catalog entry for dataset {dataset_id} not found")
                
                # Update allowed fields
                allowed_fields = {
                    'name', 'description', 'classification', 'steward',
                    'business_glossary_terms', 'tags', 'retention_policy',
                    'compliance_requirements'
                }
                
                for field, value in updates.items():
                    if field in allowed_fields:
                        setattr(catalog_entry, field, value)
                
                catalog_entry.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(catalog_entry)
                
                self.logger.info(f"Catalog entry for dataset {dataset_id} updated")
                
                return self._model_to_dataclass(catalog_entry)
                
        except Exception as e:
            self.logger.error(f"Error updating catalog entry: {str(e)}")
            raise DataCatalogError(f"Failed to update catalog entry: {str(e)}")
    
    def get_catalog_entry(self, dataset_id: str) -> Optional[DataCatalogEntry]:
        """Get a catalog entry by dataset ID."""
        try:
            with get_db_session() as session:
                catalog_entry = session.query(DataCatalogEntryModel).filter(
                    DataCatalogEntryModel.dataset_id == dataset_id
                ).first()
                
                if not catalog_entry:
                    return None
                
                return self._model_to_dataclass(catalog_entry)
                
        except Exception as e:
            self.logger.error(f"Error retrieving catalog entry: {str(e)}")
            raise DataCatalogError(f"Failed to retrieve catalog entry: {str(e)}")
    
    def search_catalog(
        self,
        query: Optional[str] = None,
        classification: Optional[DataClassification] = None,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
        limit: int = 100
    ) -> List[DataCatalogEntry]:
        """Search the data catalog."""
        try:
            with get_db_session() as session:
                query_obj = session.query(DataCatalogEntryModel)
                
                # Apply filters
                if query:
                    query_obj = query_obj.filter(
                        DataCatalogEntryModel.name.ilike(f"%{query}%") |
                        DataCatalogEntryModel.description.ilike(f"%{query}%")
                    )
                
                if classification:
                    query_obj = query_obj.filter(
                        DataCatalogEntryModel.classification == classification.value
                    )
                
                if owner:
                    query_obj = query_obj.filter(
                        DataCatalogEntryModel.owner == owner
                    )
                
                if tags:
                    for tag in tags:
                        query_obj = query_obj.filter(
                            DataCatalogEntryModel.tags.contains([tag])
                        )
                
                entries = query_obj.limit(limit).all()
                
                return [self._model_to_dataclass(entry) for entry in entries]
                
        except Exception as e:
            self.logger.error(f"Error searching catalog: {str(e)}")
            raise DataCatalogError(f"Failed to search catalog: {str(e)}")
    
    def update_quality_metrics(
        self,
        dataset_id: str,
        quality_metrics: Dict[str, float]
    ) -> None:
        """Update quality metrics for a catalog entry."""
        try:
            with get_db_session() as session:
                catalog_entry = session.query(DataCatalogEntryModel).filter(
                    DataCatalogEntryModel.dataset_id == dataset_id
                ).first()
                
                if not catalog_entry:
                    raise DataCatalogError(f"Catalog entry for dataset {dataset_id} not found")
                
                catalog_entry.quality_metrics = quality_metrics
                catalog_entry.updated_at = datetime.utcnow()
                session.commit()
                
                self.logger.info(f"Quality metrics updated for dataset {dataset_id}")
                
        except Exception as e:
            self.logger.error(f"Error updating quality metrics: {str(e)}")
            raise DataCatalogError(f"Failed to update quality metrics: {str(e)}")
    
    def update_usage_statistics(
        self,
        dataset_id: str,
        usage_stats: Dict[str, Any]
    ) -> None:
        """Update usage statistics for a catalog entry."""
        try:
            with get_db_session() as session:
                catalog_entry = session.query(DataCatalogEntryModel).filter(
                    DataCatalogEntryModel.dataset_id == dataset_id
                ).first()
                
                if not catalog_entry:
                    raise DataCatalogError(f"Catalog entry for dataset {dataset_id} not found")
                
                catalog_entry.usage_statistics = usage_stats
                catalog_entry.last_accessed = datetime.utcnow()
                catalog_entry.updated_at = datetime.utcnow()
                session.commit()
                
                self.logger.info(f"Usage statistics updated for dataset {dataset_id}")
                
        except Exception as e:
            self.logger.error(f"Error updating usage statistics: {str(e)}")
            raise DataCatalogError(f"Failed to update usage statistics: {str(e)}")
    
    def get_governance_metrics(self) -> GovernanceMetrics:
        """Calculate governance metrics."""
        try:
            with get_db_session() as session:
                # Count total datasets
                total_datasets = session.query(DatasetModel).count()
                
                # Count classified datasets
                classified_datasets = session.query(DataCatalogEntryModel).count()
                
                # Count active users
                active_users = session.query(UserModel).filter(
                    UserModel.is_active == True
                ).count()
                
                # Count data stewards
                data_stewards = session.query(DataCatalogEntryModel.steward).distinct().count()
                
                # Calculate average quality score
                quality_scores = []
                catalog_entries = session.query(DataCatalogEntryModel).all()
                for entry in catalog_entries:
                    if entry.quality_metrics and 'overall_score' in entry.quality_metrics:
                        quality_scores.append(entry.quality_metrics['overall_score'])
                
                avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                
                return GovernanceMetrics(
                    total_datasets=total_datasets,
                    classified_datasets=classified_datasets,
                    policy_violations=0,  # TODO: Calculate from policy engine
                    compliance_score=0.0,  # TODO: Calculate from compliance reports
                    data_quality_score=avg_quality_score,
                    access_requests_pending=0,  # TODO: Calculate from access control
                    audit_events_count=0,  # TODO: Calculate from audit events
                    active_users=active_users,
                    data_stewards=data_stewards,
                    calculated_at=datetime.utcnow()
                )
                
        except Exception as e:
            self.logger.error(f"Error calculating governance metrics: {str(e)}")
            raise DataCatalogError(f"Failed to calculate governance metrics: {str(e)}")
    
    def _extract_schema_info(self, dataset: DatasetModel) -> Dict[str, Any]:
        """Extract schema information from dataset."""
        if dataset.schema_definition:
            return {
                'columns': dataset.schema_definition.get('columns', {}),
                'primary_key': dataset.schema_definition.get('primary_key'),
                'foreign_keys': dataset.schema_definition.get('foreign_keys', {}),
                'constraints': dataset.schema_definition.get('constraints', [])
            }
        return {}
    
    def _model_to_dataclass(self, model: DataCatalogEntryModel) -> DataCatalogEntry:
        """Convert database model to dataclass."""
        return DataCatalogEntry(
            id=str(model.id),
            dataset_id=str(model.dataset_id),
            name=model.name,
            description=model.description or "",
            classification=DataClassification(model.classification),
            owner=str(model.owner) if model.owner else "",
            steward=str(model.steward) if model.steward else "",
            business_glossary_terms=model.business_glossary_terms or [],
            tags=model.tags or [],
            schema_info=model.schema_info or {},
            lineage_info=model.lineage_info or {},
            quality_metrics=model.quality_metrics or {},
            usage_statistics=model.usage_statistics or {},
            retention_policy=model.retention_policy,
            compliance_requirements=model.compliance_requirements or [],
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_accessed=model.last_accessed
        )