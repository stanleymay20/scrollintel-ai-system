"""
Unit tests for Data Product Registry

Tests for CRUD operations, versioning, search capabilities,
and governance enforcement.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.data_product_models import (
    Base, DataProduct, DataProductVersion, DataProvenance,
    QualityMetrics, BiasAssessment, AccessLevel, VerificationStatus
)
from scrollintel.core.data_product_registry import DataProductRegistry, DataProductSearchEngine


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def registry(db_session):
    """Create data product registry instance"""
    return DataProductRegistry(db_session)


@pytest.fixture
def search_engine(db_session):
    """Create search engine instance"""
    return DataProductSearchEngine(db_session)


@pytest.fixture
def sample_schema():
    """Sample data product schema"""
    return {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "age": {"type": "integer", "minimum": 0},
            "created_at": {"type": "string", "format": "date-time"}
        },
        "required": ["user_id", "email"]
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata"""
    return {
        "domain": "user_management",
        "category": "customer_data",
        "tags": ["pii", "customer", "profile"],
        "data_source": "user_service_db",
        "update_frequency": "daily"
    }


class TestDataProductRegistry:
    """Test data product registry operations"""
    
    def test_create_data_product(self, registry, sample_schema, sample_metadata):
        """Test creating a new data product"""
        
        product = registry.create_data_product(
            name="user_profiles",
            schema_definition=sample_schema,
            owner="data_team",
            description="User profile data from main application",
            metadata=sample_metadata,
            access_level=AccessLevel.INTERNAL,
            compliance_tags=["GDPR", "CCPA"]
        )
        
        assert product.id is not None
        assert product.name == "user_profiles"
        assert product.version == "1.0.0"
        assert product.owner == "data_team"
        assert product.access_level == AccessLevel.INTERNAL.value
        assert product.compliance_tags == ["GDPR", "CCPA"]
        assert product.verification_status == VerificationStatus.PENDING.value
        assert product.schema_definition == sample_schema
        assert product.product_metadata == sample_metadata
        
        # Check that initial version was created
        versions = registry.get_data_product_versions(str(product.id))
        assert len(versions) == 1
        assert versions[0].version_number == "1.0.0"
        assert versions[0].change_type == "major"
    
    def test_get_data_product_by_id(self, registry, sample_schema):
        """Test retrieving data product by ID"""
        
        product = registry.create_data_product(
            name="test_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        retrieved = registry.get_data_product(str(product.id))
        assert retrieved is not None
        assert retrieved.id == product.id
        assert retrieved.name == "test_product"
    
    def test_get_data_product_by_name(self, registry, sample_schema):
        """Test retrieving data product by name"""
        
        product = registry.create_data_product(
            name="unique_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        retrieved = registry.get_data_product_by_name("unique_product")
        assert retrieved is not None
        assert retrieved.id == product.id
        assert retrieved.name == "unique_product"
    
    def test_update_data_product_schema(self, registry, sample_schema):
        """Test updating data product schema creates new version"""
        
        product = registry.create_data_product(
            name="evolving_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        # Update schema
        updated_schema = sample_schema.copy()
        updated_schema["properties"]["phone"] = {"type": "string"}
        
        updated_product = registry.update_data_product(
            str(product.id),
            schema_definition=updated_schema,
            updated_by="test_owner"
        )
        
        assert updated_product.version == "1.1.0"  # Minor version increment
        assert updated_product.schema_definition == updated_schema
        
        # Check versions
        versions = registry.get_data_product_versions(str(product.id))
        assert len(versions) == 2
        assert versions[0].version_number == "1.1.0"  # Latest first
        assert versions[1].version_number == "1.0.0"
    
    def test_update_data_product_metadata_only(self, registry, sample_schema, sample_metadata):
        """Test updating only metadata doesn't create new version"""
        
        product = registry.create_data_product(
            name="metadata_product",
            schema_definition=sample_schema,
            owner="test_owner",
            metadata=sample_metadata
        )
        
        original_version = product.version
        
        # Update only metadata
        new_metadata = sample_metadata.copy()
        new_metadata["tags"].append("updated")
        
        updated_product = registry.update_data_product(
            str(product.id),
            metadata=new_metadata
        )
        
        assert updated_product.version == original_version  # No version change
        assert updated_product.product_metadata == new_metadata
        
        # Check versions - should still be only one
        versions = registry.get_data_product_versions(str(product.id))
        assert len(versions) == 1
    
    def test_delete_data_product(self, registry, sample_schema):
        """Test deleting data product"""
        
        product = registry.create_data_product(
            name="deletable_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        product_id = str(product.id)
        
        # Delete product
        success = registry.delete_data_product(product_id)
        assert success is True
        
        # Verify it's deleted
        retrieved = registry.get_data_product(product_id)
        assert retrieved is None
    
    def test_search_data_products_by_text(self, registry, sample_schema):
        """Test text search functionality"""
        
        # Create test products
        registry.create_data_product(
            name="user_data",
            schema_definition=sample_schema,
            owner="team_a",
            description="User profile information"
        )
        
        registry.create_data_product(
            name="order_data",
            schema_definition=sample_schema,
            owner="team_b",
            description="Customer order history"
        )
        
        # Search by name
        results, count = registry.search_data_products(query="user")
        assert count == 1
        assert results[0].name == "user_data"
        
        # Search by description
        results, count = registry.search_data_products(query="customer")
        assert count == 1
        assert results[0].name == "order_data"
    
    def test_search_data_products_with_filters(self, registry, sample_schema):
        """Test search with various filters"""
        
        # Create test products with different attributes
        registry.create_data_product(
            name="product_a",
            schema_definition=sample_schema,
            owner="team_a",
            access_level=AccessLevel.PUBLIC,
            compliance_tags=["GDPR"]
        )
        
        registry.create_data_product(
            name="product_b",
            schema_definition=sample_schema,
            owner="team_b",
            access_level=AccessLevel.RESTRICTED,
            compliance_tags=["HIPAA"]
        )
        
        # Filter by owner
        results, count = registry.search_data_products(owner="team_a")
        assert count == 1
        assert results[0].name == "product_a"
        
        # Filter by access level
        results, count = registry.search_data_products(access_level=AccessLevel.PUBLIC)
        assert count == 1
        assert results[0].name == "product_a"
        
        # Filter by compliance tags
        results, count = registry.search_data_products(compliance_tags=["HIPAA"])
        assert count == 1
        assert results[0].name == "product_b"
    
    def test_create_provenance_record(self, registry, sample_schema):
        """Test creating provenance record"""
        
        product = registry.create_data_product(
            name="tracked_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        provenance = registry.create_provenance_record(
            str(product.id),
            source_systems=["user_db", "analytics_db"],
            transformations=[
                {"type": "join", "tables": ["users", "user_analytics"]},
                {"type": "filter", "condition": "active = true"}
            ],
            lineage_graph={
                "nodes": ["user_db", "analytics_db", "tracked_product"],
                "edges": [
                    {"from": "user_db", "to": "tracked_product"},
                    {"from": "analytics_db", "to": "tracked_product"}
                ]
            }
        )
        
        assert provenance.data_product_id == product.id
        assert provenance.source_systems == ["user_db", "analytics_db"]
        assert len(provenance.transformations) == 2
        assert provenance.provenance_hash is not None
    
    def test_update_quality_metrics(self, registry, sample_schema):
        """Test updating quality metrics"""
        
        product = registry.create_data_product(
            name="quality_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        metrics = registry.update_quality_metrics(
            str(product.id),
            completeness_score=0.95,
            accuracy_score=0.88,
            consistency_score=0.92,
            timeliness_score=0.85,
            issues=[{"type": "missing_values", "count": 50}],
            recommendations=["Implement data validation"],
            assessed_by="quality_team"
        )
        
        assert metrics.completeness_score == 0.95
        assert metrics.accuracy_score == 0.88
        assert metrics.overall_score == 0.9  # Average of all scores
        assert len(metrics.issues) == 1
        assert len(metrics.recommendations) == 1
        
        # Check that product quality score was updated
        updated_product = registry.get_data_product(str(product.id))
        assert updated_product.quality_score == 0.9
    
    def test_update_bias_assessment(self, registry, sample_schema):
        """Test updating bias assessment"""
        
        product = registry.create_data_product(
            name="bias_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        assessment = registry.update_bias_assessment(
            str(product.id),
            protected_attributes=["gender", "age"],
            statistical_parity=0.85,
            equalized_odds=0.82,
            demographic_parity=0.88,
            individual_fairness=0.90,
            bias_issues=[{"attribute": "gender", "severity": "medium"}],
            mitigation_strategies=["Rebalance training data"],
            assessed_by="fairness_team"
        )
        
        assert assessment.protected_attributes == ["gender", "age"]
        assert assessment.statistical_parity == 0.85
        assert len(assessment.bias_issues) == 1
        assert len(assessment.mitigation_strategies) == 1
        
        # Check that product bias score was updated
        updated_product = registry.get_data_product(str(product.id))
        expected_bias_score = (0.85 + 0.82 + 0.88 + 0.90) / 4
        assert updated_product.bias_score == expected_bias_score
    
    def test_verify_data_product(self, registry, sample_schema):
        """Test verification status updates"""
        
        product = registry.create_data_product(
            name="verifiable_product",
            schema_definition=sample_schema,
            owner="test_owner"
        )
        
        # Initially pending
        assert product.verification_status == VerificationStatus.PENDING.value
        
        # Verify product
        verified_product = registry.verify_data_product(
            str(product.id),
            VerificationStatus.VERIFIED
        )
        
        assert verified_product.verification_status == VerificationStatus.VERIFIED.value
        
        # Quarantine product
        quarantined_product = registry.verify_data_product(
            str(product.id),
            VerificationStatus.QUARANTINED
        )
        
        assert quarantined_product.verification_status == VerificationStatus.QUARANTINED.value
    
    def test_version_hash_generation(self, registry, sample_schema):
        """Test that version hashes are generated correctly"""
        
        product1 = registry.create_data_product(
            name="hash_test_1",
            schema_definition=sample_schema,
            owner="test_owner",
            metadata={"key": "value1"}
        )
        
        product2 = registry.create_data_product(
            name="hash_test_2",
            schema_definition=sample_schema,
            owner="test_owner",
            metadata={"key": "value1"}  # Same metadata
        )
        
        versions1 = registry.get_data_product_versions(str(product1.id))
        versions2 = registry.get_data_product_versions(str(product2.id))
        
        # Same schema and metadata should produce same hash
        assert versions1[0].version_hash == versions2[0].version_hash
        
        # Update one product with different metadata
        registry.update_data_product(
            str(product1.id),
            metadata={"key": "value2"}
        )
        
        updated_versions1 = registry.get_data_product_versions(str(product1.id))
        
        # Should have different hash now
        assert updated_versions1[0].version_hash != versions2[0].version_hash


class TestDataProductSearchEngine:
    """Test search engine functionality"""
    
    def test_semantic_search(self, search_engine, registry, sample_schema):
        """Test semantic search capabilities"""
        
        # Create test products
        registry.create_data_product(
            name="customer_profiles",
            schema_definition=sample_schema,
            owner="team_a",
            description="Customer demographic and behavioral data"
        )
        
        registry.create_data_product(
            name="user_analytics",
            schema_definition=sample_schema,
            owner="team_b",
            description="User interaction and engagement metrics"
        )
        
        # Search for customer-related products
        results = search_engine.semantic_search("customer")
        assert len(results) >= 1
        assert any("customer" in result.name.lower() or 
                 "customer" in (result.description or "").lower() 
                 for result in results)
    
    def test_faceted_search(self, search_engine, registry, sample_schema):
        """Test faceted search functionality"""
        
        # Create products with different facet values
        registry.create_data_product(
            name="product_1",
            schema_definition=sample_schema,
            owner="team_a",
            access_level=AccessLevel.PUBLIC
        )
        
        registry.create_data_product(
            name="product_2",
            schema_definition=sample_schema,
            owner="team_b",
            access_level=AccessLevel.RESTRICTED
        )
        
        # Search with facets
        facets = {
            "owner": ["team_a"],
            "access_level": [AccessLevel.PUBLIC.value]
        }
        
        results = search_engine.faceted_search(facets)
        assert len(results) == 1
        assert results[0].name == "product_1"
    
    def test_get_related_products(self, search_engine, registry, sample_schema):
        """Test finding related products"""
        
        # Create products with similar attributes
        product1 = registry.create_data_product(
            name="user_data_1",
            schema_definition=sample_schema,
            owner="data_team",
            compliance_tags=["GDPR", "CCPA"]
        )
        
        registry.create_data_product(
            name="user_data_2",
            schema_definition=sample_schema,
            owner="data_team",  # Same owner
            compliance_tags=["GDPR"]  # Overlapping tags
        )
        
        registry.create_data_product(
            name="order_data",
            schema_definition=sample_schema,
            owner="commerce_team",  # Different owner
            compliance_tags=["PCI"]  # Different tags
        )
        
        # Find related products
        related = search_engine.get_related_products(str(product1.id))
        
        # Should find user_data_2 (same owner and overlapping tags)
        # but not order_data (different owner and tags)
        assert len(related) >= 1
        related_names = [p.name for p in related]
        assert "user_data_2" in related_names
        assert "order_data" not in related_names


@pytest.mark.integration
class TestDataProductRegistryIntegration:
    """Integration tests for the complete registry system"""
    
    def test_complete_data_product_lifecycle(self, registry, sample_schema, sample_metadata):
        """Test complete lifecycle from creation to verification"""
        
        # 1. Create data product
        product = registry.create_data_product(
            name="lifecycle_product",
            schema_definition=sample_schema,
            owner="data_team",
            description="Test product for lifecycle",
            metadata=sample_metadata,
            compliance_tags=["GDPR"]
        )
        
        assert product.verification_status == VerificationStatus.PENDING.value
        
        # 2. Add provenance
        provenance = registry.create_provenance_record(
            str(product.id),
            source_systems=["source_db"],
            transformations=[{"type": "etl", "process": "daily_batch"}],
            lineage_graph={"nodes": ["source_db", "lifecycle_product"]}
        )
        
        assert provenance is not None
        
        # 3. Update quality metrics
        metrics = registry.update_quality_metrics(
            str(product.id),
            completeness_score=0.95,
            accuracy_score=0.90,
            consistency_score=0.88,
            timeliness_score=0.92
        )
        
        assert metrics.overall_score > 0.9
        
        # 4. Update bias assessment
        assessment = registry.update_bias_assessment(
            str(product.id),
            protected_attributes=["age"],
            statistical_parity=0.85,
            equalized_odds=0.87,
            demographic_parity=0.89,
            individual_fairness=0.91
        )
        
        assert assessment is not None
        
        # 5. Verify product
        verified_product = registry.verify_data_product(
            str(product.id),
            VerificationStatus.VERIFIED
        )
        
        assert verified_product.verification_status == VerificationStatus.VERIFIED.value
        
        # 6. Update schema (should create new version)
        updated_schema = sample_schema.copy()
        updated_schema["properties"]["new_field"] = {"type": "string"}
        
        updated_product = registry.update_data_product(
            str(product.id),
            schema_definition=updated_schema
        )
        
        assert updated_product.version == "1.1.0"
        
        # 7. Verify all components are linked
        versions = registry.get_data_product_versions(str(product.id))
        assert len(versions) == 2
        
        final_product = registry.get_data_product(str(product.id))
        assert final_product.quality_score > 0.9
        assert final_product.bias_score > 0.8
        assert final_product.verification_status == VerificationStatus.VERIFIED.value