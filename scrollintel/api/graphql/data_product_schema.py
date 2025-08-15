"""
GraphQL Schema for Data Product Registry

Enhanced GraphQL interface for complex data product queries with
relationships, nested filtering, advanced search capabilities, and real-time subscriptions.
"""

import graphene
from graphene import ObjectType, String, Int, Float, Boolean, List, Field, DateTime, JSONString, Enum
from graphene_sqlalchemy import SQLAlchemyObjectType
from sqlalchemy.orm import Session
from typing import Optional, List as TypingList, Dict, Any
import asyncio
from datetime import datetime

from scrollintel.models.data_product_models import (
    DataProduct as DataProductModel,
    DataProductVersion as DataProductVersionModel,
    DataProvenance as DataProvenanceModel,
    QualityMetrics as QualityMetricsModel,
    BiasAssessment as BiasAssessmentModel,
    AccessLevel,
    VerificationStatus
)
from scrollintel.core.data_product_registry import DataProductRegistry, DataProductSearchEngine
from scrollintel.core.elasticsearch_indexer import DataProductIndexer


# GraphQL Enums
class AccessLevelEnum(Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    RESTRICTED = "RESTRICTED"
    CONFIDENTIAL = "CONFIDENTIAL"


class VerificationStatusEnum(Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"


# GraphQL Types
class QualityMetricsType(SQLAlchemyObjectType):
    """GraphQL type for QualityMetrics"""
    class Meta:
        model = QualityMetricsModel
    
    issues = JSONString()
    recommendations = List(String)


class BiasAssessmentType(SQLAlchemyObjectType):
    """GraphQL type for BiasAssessment"""
    class Meta:
        model = BiasAssessmentModel
    
    protected_attributes = List(String)
    bias_issues = JSONString()
    mitigation_strategies = List(String)


class DataProvenanceType(SQLAlchemyObjectType):
    """GraphQL type for DataProvenance"""
    class Meta:
        model = DataProvenanceModel
    
    source_systems = List(String)
    transformations = JSONString()
    lineage_graph = JSONString()


class DataProductVersionType(SQLAlchemyObjectType):
    """GraphQL type for DataProductVersion"""
    class Meta:
        model = DataProductVersionModel


class DataProductType(SQLAlchemyObjectType):
    """Enhanced GraphQL type for DataProduct"""
    class Meta:
        model = DataProductModel
        exclude_fields = ('product_metadata',)
    
    metadata = JSONString()
    versions = List(DataProductVersionType)
    provenance = Field(DataProvenanceType)
    quality_metrics = Field(QualityMetricsType)
    bias_assessment = Field(BiasAssessmentType)
    related_products = List(lambda: DataProductType)
    lineage_upstream = List(lambda: DataProductType)
    lineage_downstream = List(lambda: DataProductType)
    
    def resolve_metadata(self, info):
        return self.product_metadata
    
    def resolve_versions(self, info):
        return self.versions
    
    def resolve_provenance(self, info):
        return self.provenance
    
    def resolve_quality_metrics(self, info):
        return self.quality_metrics
    
    def resolve_bias_assessment(self, info):
        return self.bias_assessment
    
    def resolve_related_products(self, info):
        # Get related products based on similarity
        search_engine = DataProductSearchEngine(info.context.get('db'))
        return search_engine.get_related_products(str(self.id), limit=10)
    
    def resolve_lineage_upstream(self, info):
        # Get upstream data products in lineage
        registry = DataProductRegistry(info.context.get('db'))
        return registry.get_upstream_products(str(self.id))
    
    def resolve_lineage_downstream(self, info):
        # Get downstream data products in lineage
        registry = DataProductRegistry(info.context.get('db'))
        return registry.get_downstream_products(str(self.id))


# Search Result Types
class DataProductSearchResult(ObjectType):
    """Search result with metadata"""
    product = Field(DataProductType)
    score = Float()
    highlights = JSONString()


class DataProductConnection(ObjectType):
    """Paginated connection for data products"""
    edges = List(DataProductType)
    total_count = Int()
    has_next_page = Boolean()
    has_previous_page = Boolean()


class FacetValue(ObjectType):
    """Facet value with count"""
    value = String()
    count = Int()


class Facet(ObjectType):
    """Search facet"""
    field = String()
    values = List(FacetValue)


class SearchResponse(ObjectType):
    """Enhanced search response with facets"""
    products = Field(DataProductConnection)
    facets = List(Facet)
    total_count = Int()
    query_time_ms = Int()


# Input Types
class DataProductFilterInput(graphene.InputObjectType):
    """Filter input for data product queries"""
    owner = String()
    access_level = AccessLevelEnum()
    verification_status = VerificationStatusEnum()
    compliance_tags = List(String)
    quality_score_min = Float()
    quality_score_max = Float()
    bias_score_min = Float()
    bias_score_max = Float()
    created_after = DateTime()
    created_before = DateTime()
    updated_after = DateTime()
    updated_before = DateTime()


class SearchInput(graphene.InputObjectType):
    """Search input with advanced options"""
    query = String()
    filters = Field(DataProductFilterInput)
    facets = List(String)
    sort_by = String()
    sort_order = String()
    limit = Int()
    offset = Int()


class DataProductCreateInput(graphene.InputObjectType):
    """Input for creating data products"""
    name = String(required=True)
    description = String()
    schema_definition = JSONString(required=True)
    metadata = JSONString()
    access_level = AccessLevelEnum()
    compliance_tags = List(String)
    owner = String(required=True)


class DataProductUpdateInput(graphene.InputObjectType):
    """Input for updating data products"""
    description = String()
    schema_definition = JSONString()
    metadata = JSONString()
    access_level = AccessLevelEnum()
    compliance_tags = List(String)
    


# Query Class
class Query(ObjectType):
    """GraphQL Query root"""
    
    # Single product queries
    data_product = Field(
        DataProductType,
        id=String(required=True),
        version=String(),
        description="Get data product by ID"
    )
    
    # Search and list queries
    search_data_products = Field(
        SearchResponse,
        input=SearchInput(required=True),
        description="Advanced search for data products"
    )
    
    data_products = Field(
        DataProductConnection,
        filters=Field(DataProductFilterInput),
        limit=Int(default_value=50),
        offset=Int(default_value=0),
        description="List data products with filters"
    )
    
    # Semantic search
    semantic_search = Field(
        List(DataProductSearchResult),
        query=String(required=True),
        limit=Int(default_value=20),
        description="Semantic search using embeddings"
    )
    
    # Faceted search
    faceted_search = Field(
        SearchResponse,
        query=String(),
        facets=List(String, required=True),
        filters=JSONString(),
        limit=Int(default_value=50),
        description="Faceted search with aggregations"
    )
    
    # Lineage queries
    data_lineage = Field(
        JSONString,
        product_id=String(required=True),
        depth=Int(default_value=3),
        direction=String(default_value="both"),  # upstream, downstream, both
        description="Get data lineage graph"
    )
    
    # Quality and governance queries
    quality_report = Field(
        JSONString,
        product_id=String(),
        owner=String(),
        access_level=AccessLevelEnum(),
        description="Generate quality report"
    )
    
    compliance_report = Field(
        JSONString,
        compliance_tags=List(String),
        verification_status=VerificationStatusEnum(),
        description="Generate compliance report"
    )
    
    # Analytics queries
    product_analytics = Field(
        JSONString,
        product_id=String(),
        time_range=String(default_value="30d"),
        description="Get product usage analytics"
    )
    
    registry_stats = Field(
        JSONString,
        description="Get registry statistics"
    )
    
    # Resolvers
    def resolve_data_product(self, info, id, version=None):
        registry = DataProductRegistry(info.context.get('db'))
        return registry.get_data_product(id, version)
    
    def resolve_search_data_products(self, info, input):
        start_time = datetime.now()
        
        search_engine = DataProductSearchEngine(info.context.get('db'))
        indexer = DataProductIndexer()
        
        # Perform search
        if input.get('query'):
            results, total_count = indexer.search_data_products(
                query=input['query'],
                filters=input.get('filters', {}),
                sort_by=input.get('sort_by'),
                sort_order=input.get('sort_order', 'desc'),
                limit=input.get('limit', 50),
                offset=input.get('offset', 0)
            )
        else:
            registry = DataProductRegistry(info.context.get('db'))
            results, total_count = registry.search_data_products(
                **input.get('filters', {}),
                limit=input.get('limit', 50),
                offset=input.get('offset', 0)
            )
        
        # Get facets if requested
        facets = []
        if input.get('facets'):
            facet_results = indexer.faceted_search(
                query=input.get('query'),
                facets=input['facets'],
                filters=input.get('filters', {}),
                limit=0  # Only get facets
            )
            facets = [
                Facet(
                    field=field,
                    values=[
                        FacetValue(value=value, count=count)
                        for value, count in values.items()
                    ]
                )
                for field, values in facet_results.get('facets', {}).items()
            ]
        
        query_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return SearchResponse(
            products=DataProductConnection(
                edges=results,
                total_count=total_count,
                has_next_page=input.get('offset', 0) + input.get('limit', 50) < total_count,
                has_previous_page=input.get('offset', 0) > 0
            ),
            facets=facets,
            total_count=total_count,
            query_time_ms=query_time
        )
    
    def resolve_data_products(self, info, filters=None, limit=50, offset=0):
        registry = DataProductRegistry(info.context.get('db'))
        
        filter_dict = {}
        if filters:
            if filters.owner:
                filter_dict['owner'] = filters.owner
            if filters.access_level:
                filter_dict['access_level'] = AccessLevel(filters.access_level)
            if filters.verification_status:
                filter_dict['verification_status'] = VerificationStatus(filters.verification_status)
            if filters.compliance_tags:
                filter_dict['compliance_tags'] = filters.compliance_tags
        
        results, total_count = registry.search_data_products(
            **filter_dict,
            limit=limit,
            offset=offset
        )
        
        return DataProductConnection(
            edges=results,
            total_count=total_count,
            has_next_page=offset + limit < total_count,
            has_previous_page=offset > 0
        )
    
    def resolve_semantic_search(self, info, query, limit=20):
        search_engine = DataProductSearchEngine(info.context.get('db'))
        results = search_engine.semantic_search(query, limit)
        
        return [
            DataProductSearchResult(
                product=product,
                score=0.95,  # Placeholder score
                highlights={}
            )
            for product in results
        ]
    
    def resolve_faceted_search(self, info, facets, query=None, filters=None, limit=50):
        indexer = DataProductIndexer()
        
        result = indexer.faceted_search(
            query=query,
            facets=facets,
            filters=filters or {},
            limit=limit
        )
        
        facet_objects = [
            Facet(
                field=field,
                values=[
                    FacetValue(value=value, count=count)
                    for value, count in values.items()
                ]
            )
            for field, values in result.get('facets', {}).items()
        ]
        
        return SearchResponse(
            products=DataProductConnection(
                edges=result.get('results', []),
                total_count=result.get('total_count', 0),
                has_next_page=False,
                has_previous_page=False
            ),
            facets=facet_objects,
            total_count=result.get('total_count', 0),
            query_time_ms=result.get('query_time_ms', 0)
        )
    
    def resolve_data_lineage(self, info, product_id, depth=3, direction="both"):
        registry = DataProductRegistry(info.context.get('db'))
        return registry.get_data_lineage_graph(product_id, depth, direction)
    
    def resolve_quality_report(self, info, product_id=None, owner=None, access_level=None):
        registry = DataProductRegistry(info.context.get('db'))
        return registry.generate_quality_report(
            product_id=product_id,
            owner=owner,
            access_level=access_level
        )
    
    def resolve_compliance_report(self, info, compliance_tags=None, verification_status=None):
        registry = DataProductRegistry(info.context.get('db'))
        return registry.generate_compliance_report(
            compliance_tags=compliance_tags,
            verification_status=verification_status
        )
    
    def resolve_product_analytics(self, info, product_id=None, time_range="30d"):
        registry = DataProductRegistry(info.context.get('db'))
        return registry.get_product_analytics(product_id, time_range)
    
    def resolve_registry_stats(self, info):
        registry = DataProductRegistry(info.context.get('db'))
        return registry.get_registry_statistics()


# Mutation Class
class Mutation(ObjectType):
    """GraphQL Mutation root"""
    
    create_data_product = Field(
        DataProductType,
        input=DataProductCreateInput(required=True),
        description="Create a new data product"
    )
    
    update_data_product = Field(
        DataProductType,
        id=String(required=True),
        input=DataProductUpdateInput(required=True),
        description="Update an existing data product"
    )
    
    delete_data_product = Field(
        Boolean,
        id=String(required=True),
        description="Delete a data product"
    )
    
    verify_data_product = Field(
        DataProductType,
        id=String(required=True),
        status=VerificationStatusEnum(required=True),
        verified_by=String(),
        description="Update verification status"
    )
    
    def resolve_create_data_product(self, info, input):
        registry = DataProductRegistry(info.context.get('db'))
        indexer = DataProductIndexer()
        
        product = registry.create_data_product(
            name=input['name'],
            schema_definition=input['schema_definition'],
            owner=input['owner'],
            description=input.get('description'),
            metadata=input.get('metadata'),
            access_level=AccessLevel(input.get('access_level', 'INTERNAL')),
            compliance_tags=input.get('compliance_tags')
        )
        
        # Index in Elasticsearch
        indexer.index_data_product(product)
        
        return product
    
    def resolve_update_data_product(self, info, id, input):
        registry = DataProductRegistry(info.context.get('db'))
        indexer = DataProductIndexer()
        
        product = registry.update_data_product(
            product_id=id,
            schema_definition=input.get('schema_definition'),
            metadata=input.get('metadata'),
            description=input.get('description'),
            updated_by=info.context.get('user_id', 'system')
        )
        
        # Update index
        indexer.index_data_product(product)
        
        return product
    
    def resolve_delete_data_product(self, info, id):
        registry = DataProductRegistry(info.context.get('db'))
        indexer = DataProductIndexer()
        
        success = registry.delete_data_product(id)
        
        if success:
            indexer.delete_data_product(id)
        
        return success
    
    def resolve_verify_data_product(self, info, id, status, verified_by=None):
        registry = DataProductRegistry(info.context.get('db'))
        indexer = DataProductIndexer()
        
        product = registry.verify_data_product(
            product_id=id,
            verification_status=VerificationStatus(status),
            verified_by=verified_by or info.context.get('user_id', 'system')
        )
        
        # Update index
        indexer.index_data_product(product)
        
        return product


# Subscription Class for real-time updates
class Subscription(ObjectType):
    """GraphQL Subscription root for real-time updates"""
    
    data_product_updated = Field(
        DataProductType,
        product_id=String(),
        description="Subscribe to data product updates"
    )
    
    quality_alert = Field(
        JSONString,
        threshold=Float(default_value=0.8),
        description="Subscribe to quality alerts"
    )
    
    verification_status_changed = Field(
        DataProductType,
        description="Subscribe to verification status changes"
    )
    
    async def resolve_data_product_updated(self, info, product_id=None):
        # In a real implementation, this would use a message broker like Redis
        # For now, we'll simulate with a simple async generator
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            registry = DataProductRegistry(info.context.get('db'))
            if product_id:
                product = registry.get_data_product(product_id)
                if product:
                    yield product
            else:
                # Yield recently updated products
                recent_products = registry.get_recently_updated_products(limit=1)
                if recent_products:
                    yield recent_products[0]
    
    async def resolve_quality_alert(self, info, threshold=0.8):
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            registry = DataProductRegistry(info.context.get('db'))
            alerts = registry.get_quality_alerts(threshold)
            
            for alert in alerts:
                yield alert
    
    async def resolve_verification_status_changed(self, info):
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            registry = DataProductRegistry(info.context.get('db'))
            recently_verified = registry.get_recently_verified_products(limit=1)
            
            if recently_verified:
                yield recently_verified[0]


# Schema definition
schema = graphene.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)