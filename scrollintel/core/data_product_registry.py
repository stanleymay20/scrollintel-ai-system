"""
Data Product Registry Core

Core functionality for managing data products including CRUD operations,
versioning, search capabilities, and governance enforcement.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from scrollintel.models.data_product_models import (
    DataProduct, DataProductVersion, DataProvenance, QualityMetrics,
    BiasAssessment, DataSchema, GovernancePolicy, ComplianceTag,
    AccessLevel, VerificationStatus
)


class DataProductRegistry:
    """Core data product registry with CRUD operations and versioning"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_data_product(
        self,
        name: str,
        schema_definition: Dict[str, Any],
        owner: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        access_level: AccessLevel = AccessLevel.INTERNAL,
        compliance_tags: Optional[List[str]] = None
    ) -> DataProduct:
        """Create a new data product with initial version"""
        
        # Generate version hash based on name, schema and metadata
        version_hash = self._generate_version_hash(name, schema_definition, metadata or {})
        initial_version = "1.0.0"
        
        # Create data product
        data_product = DataProduct(
            name=name,
            version=initial_version,
            description=description,
            schema_definition=schema_definition,
            product_metadata=metadata or {},
            owner=owner,
            access_level=access_level.value,
            compliance_tags=compliance_tags or [],
            verification_status=VerificationStatus.PENDING.value
        )
        
        self.db.add(data_product)
        self.db.flush()  # Get the ID
        
        # Create initial version record
        version_record = DataProductVersion(
            data_product_id=data_product.id,
            version_hash=version_hash,
            version_number=initial_version,
            schema_hash=self._generate_schema_hash(schema_definition),
            change_description="Initial version",
            change_type="major",
            created_by=owner
        )
        
        self.db.add(version_record)
        self.db.commit()
        
        return data_product
    
    def get_data_product(
        self,
        product_id: str,
        version: Optional[str] = None
    ) -> Optional[DataProduct]:
        """Get data product by ID and optional version"""
        
        query = self.db.query(DataProduct).filter(DataProduct.id == product_id)
        
        if version:
            query = query.filter(DataProduct.version == version)
        
        return query.first()
    
    def get_data_product_by_name(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[DataProduct]:
        """Get data product by name and optional version"""
        
        query = self.db.query(DataProduct).filter(DataProduct.name == name)
        
        if version:
            query = query.filter(DataProduct.version == version)
        else:
            # Get latest version
            query = query.order_by(desc(DataProduct.created_at))
        
        return query.first()
    
    def update_data_product(
        self,
        product_id: str,
        schema_definition: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        updated_by: str = None
    ) -> DataProduct:
        """Update data product and create new version if schema changed"""
        
        data_product = self.get_data_product(product_id)
        if not data_product:
            raise ValueError(f"Data product {product_id} not found")
        
        # Check if schema changed
        schema_changed = False
        if schema_definition and schema_definition != data_product.schema_definition:
            schema_changed = True
            new_version = self._increment_version(data_product.version, "minor")
            new_version_hash = self._generate_version_hash(data_product.name, schema_definition, metadata or data_product.product_metadata)
            
            # Create new version record
            version_record = DataProductVersion(
                data_product_id=data_product.id,
                version_hash=new_version_hash,
                version_number=new_version,
                schema_hash=self._generate_schema_hash(schema_definition),
                change_description="Schema update",
                change_type="minor",
                created_by=updated_by or data_product.owner
            )
            self.db.add(version_record)
            
            # Update data product
            data_product.schema_definition = schema_definition
            data_product.version = new_version
        
        # Update other fields
        if metadata is not None:
            data_product.product_metadata = metadata
        if description is not None:
            data_product.description = description
        
        data_product.updated_at = datetime.utcnow()
        self.db.commit()
        
        return data_product
    
    def delete_data_product(self, product_id: str) -> bool:
        """Soft delete data product (mark as inactive)"""
        
        data_product = self.get_data_product(product_id)
        if not data_product:
            return False
        
        # Instead of hard delete, we could add an is_active field
        # For now, we'll do hard delete but preserve versions
        self.db.delete(data_product)
        self.db.commit()
        
        return True
    
    def search_data_products(
        self,
        query: Optional[str] = None,
        owner: Optional[str] = None,
        access_level: Optional[AccessLevel] = None,
        verification_status: Optional[VerificationStatus] = None,
        compliance_tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[DataProduct], int]:
        """Search data products with filters"""
        
        db_query = self.db.query(DataProduct)
        
        # Text search in name and description
        if query:
            db_query = db_query.filter(
                or_(
                    DataProduct.name.ilike(f"%{query}%"),
                    DataProduct.description.ilike(f"%{query}%")
                )
            )
        
        # Filter by owner
        if owner:
            db_query = db_query.filter(DataProduct.owner == owner)
        
        # Filter by access level
        if access_level:
            db_query = db_query.filter(DataProduct.access_level == access_level.value)
        
        # Filter by verification status
        if verification_status:
            db_query = db_query.filter(DataProduct.verification_status == verification_status.value)
        
        # Filter by compliance tags
        if compliance_tags:
            for tag in compliance_tags:
                db_query = db_query.filter(DataProduct.compliance_tags.contains([tag]))
        
        # Get total count
        total_count = db_query.count()
        
        # Apply pagination and ordering
        results = db_query.order_by(desc(DataProduct.updated_at)).offset(offset).limit(limit).all()
        
        return results, total_count
    
    def get_data_product_versions(self, product_id: str) -> List[DataProductVersion]:
        """Get all versions of a data product"""
        
        return self.db.query(DataProductVersion).filter(
            DataProductVersion.data_product_id == product_id
        ).order_by(desc(DataProductVersion.created_at)).all()
    
    def create_provenance_record(
        self,
        product_id: str,
        source_systems: List[str],
        transformations: List[Dict[str, Any]],
        lineage_graph: Dict[str, Any]
    ) -> DataProvenance:
        """Create provenance record for data product"""
        
        provenance_data = {
            "source_systems": source_systems,
            "transformations": transformations,
            "lineage_graph": lineage_graph
        }
        
        provenance = DataProvenance(
            data_product_id=product_id,
            source_systems=source_systems,
            transformations=transformations,
            lineage_graph=lineage_graph,
            provenance_hash=self._generate_provenance_hash(provenance_data)
        )
        
        self.db.add(provenance)
        self.db.commit()
        
        return provenance
    
    def update_quality_metrics(
        self,
        product_id: str,
        completeness_score: float,
        accuracy_score: float,
        consistency_score: float,
        timeliness_score: float,
        issues: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[str]] = None,
        assessed_by: Optional[str] = None
    ) -> QualityMetrics:
        """Update quality metrics for data product"""
        
        overall_score = (completeness_score + accuracy_score + consistency_score + timeliness_score) / 4
        
        # Check if metrics already exist
        existing_metrics = self.db.query(QualityMetrics).filter(
            QualityMetrics.data_product_id == product_id
        ).order_by(desc(QualityMetrics.assessed_at)).first()
        
        metrics = QualityMetrics(
            data_product_id=product_id,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            issues=issues or [],
            recommendations=recommendations or [],
            assessed_by=assessed_by
        )
        
        self.db.add(metrics)
        
        # Update data product quality score
        data_product = self.get_data_product(product_id)
        if data_product:
            data_product.quality_score = overall_score
        
        self.db.commit()
        
        return metrics
    
    def update_bias_assessment(
        self,
        product_id: str,
        protected_attributes: List[str],
        statistical_parity: float,
        equalized_odds: float,
        demographic_parity: float,
        individual_fairness: float,
        bias_issues: Optional[List[Dict[str, Any]]] = None,
        mitigation_strategies: Optional[List[str]] = None,
        assessed_by: Optional[str] = None
    ) -> BiasAssessment:
        """Update bias assessment for data product"""
        
        assessment = BiasAssessment(
            data_product_id=product_id,
            protected_attributes=protected_attributes,
            statistical_parity=statistical_parity,
            equalized_odds=equalized_odds,
            demographic_parity=demographic_parity,
            individual_fairness=individual_fairness,
            bias_issues=bias_issues or [],
            mitigation_strategies=mitigation_strategies or [],
            assessed_by=assessed_by
        )
        
        self.db.add(assessment)
        
        # Update data product bias score (average of fairness metrics)
        bias_score = (statistical_parity + equalized_odds + demographic_parity + individual_fairness) / 4
        data_product = self.get_data_product(product_id)
        if data_product:
            data_product.bias_score = bias_score
        
        self.db.commit()
        
        return assessment
    
    def verify_data_product(
        self,
        product_id: str,
        verification_status: VerificationStatus,
        verified_by: Optional[str] = None
    ) -> DataProduct:
        """Update verification status of data product"""
        
        data_product = self.get_data_product(product_id)
        if not data_product:
            raise ValueError(f"Data product {product_id} not found")
        
        data_product.verification_status = verification_status.value
        data_product.updated_at = datetime.utcnow()
        
        self.db.commit()
        
        return data_product
    
    def _generate_version_hash(self, name: str, schema: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Generate hash for version tracking"""
        content = json.dumps({"name": name, "schema": schema, "metadata": metadata}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_schema_hash(self, schema: Dict[str, Any]) -> str:
        """Generate hash for schema tracking"""
        content = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_provenance_hash(self, provenance_data: Dict[str, Any]) -> str:
        """Generate hash for provenance verification"""
        content = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _increment_version(self, current_version: str, change_type: str) -> str:
        """Increment version number based on change type"""
        parts = current_version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        if change_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif change_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"


class DataProductSearchEngine:
    """Enhanced search capabilities for data products"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def semantic_search(
        self,
        query: str,
        limit: int = 20
    ) -> List[DataProduct]:
        """Semantic search using metadata and descriptions"""
        
        # For now, implement basic text search
        # In production, this would use Elasticsearch or vector embeddings
        
        return self.db.query(DataProduct).filter(
            or_(
                DataProduct.name.ilike(f"%{query}%"),
                DataProduct.description.ilike(f"%{query}%"),
                func.cast(DataProduct.product_metadata, str).ilike(f"%{query}%")
            )
        ).limit(limit).all()
    
    def faceted_search(
        self,
        facets: Dict[str, List[str]],
        limit: int = 50
    ) -> List[DataProduct]:
        """Faceted search with multiple filter dimensions"""
        
        query = self.db.query(DataProduct)
        
        for facet_name, facet_values in facets.items():
            if facet_name == "owner":
                query = query.filter(DataProduct.owner.in_(facet_values))
            elif facet_name == "access_level":
                query = query.filter(DataProduct.access_level.in_(facet_values))
            elif facet_name == "verification_status":
                query = query.filter(DataProduct.verification_status.in_(facet_values))
            elif facet_name == "compliance_tags":
                for tag in facet_values:
                    query = query.filter(DataProduct.compliance_tags.contains([tag]))
        
        return query.limit(limit).all()
    
    def get_related_products(
        self,
        product_id: str,
        limit: int = 10
    ) -> List[DataProduct]:
        """Find related data products based on metadata similarity"""
        
        # Get the source product
        source_product = self.db.query(DataProduct).filter(
            DataProduct.id == product_id
        ).first()
        
        if not source_product:
            return []
        
        # Simple implementation: find products with similar owners or tags
        related = self.db.query(DataProduct).filter(
            and_(
                DataProduct.id != product_id,
                or_(
                    DataProduct.owner == source_product.owner,
                    DataProduct.compliance_tags.overlap(source_product.compliance_tags)
                )
            )
        ).limit(limit).all()
        
        return related