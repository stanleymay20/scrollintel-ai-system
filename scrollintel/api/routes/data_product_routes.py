"""
Data Product API Routes

Enhanced REST API endpoints for data product management including CRUD operations,
search, versioning, governance features, rate limiting, and authentication.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import jwt
from functools import wraps
import time
from collections import defaultdict

from scrollintel.core.data_product_registry import DataProductRegistry, DataProductSearchEngine
from scrollintel.core.elasticsearch_indexer import DataProductIndexer
from scrollintel.models.data_product_models import AccessLevel, VerificationStatus
from scrollintel.models.database import get_db

router = APIRouter(prefix="/api/v1/data-products", tags=["data-products"])

# Security and rate limiting
security = HTTPBearer()

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)

# JWT configuration
JWT_SECRET = "your-secret-key"  # In production, use environment variable
JWT_ALGORITHM = "HS256"

class RateLimitExceeded(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or next((arg for arg in args if isinstance(arg, Request)), None)
            if not request:
                return await func(*args, **kwargs)
            
            client_ip = request.client.host
            current_time = time.time()
            
            # Clean old requests
            rate_limit_storage[client_ip] = [
                req_time for req_time in rate_limit_storage[client_ip]
                if current_time - req_time < window_seconds
            ]
            
            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= max_requests:
                raise RateLimitExceeded()
            
            # Add current request
            rate_limit_storage[client_ip].append(current_time)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id, "permissions": payload.get("permissions", [])}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_permission(permission: str):
    """Require specific permission"""
    def dependency(user_info: dict = Depends(verify_token)):
        if permission not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
        return user_info
    return dependency


# Pydantic models for request/response
class DataProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    schema_definition: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    access_level: AccessLevel = AccessLevel.INTERNAL
    compliance_tags: Optional[List[str]] = None


class DataProductUpdate(BaseModel):
    description: Optional[str] = None
    schema_definition: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DataProductResponse(BaseModel):
    id: str
    name: str
    version: str
    description: Optional[str]
    schema_definition: Dict[str, Any]
    product_metadata: Dict[str, Any]
    owner: str
    access_level: str
    verification_status: str
    quality_score: float
    bias_score: float
    compliance_tags: List[str]
    created_at: datetime
    updated_at: datetime
    freshness_timestamp: datetime

    class Config:
        from_attributes = True


class DataProductSearchResponse(BaseModel):
    results: List[DataProductResponse]
    total_count: int
    limit: int
    offset: int


class ProvenanceCreate(BaseModel):
    source_systems: List[str]
    transformations: List[Dict[str, Any]]
    lineage_graph: Dict[str, Any]


class QualityMetricsUpdate(BaseModel):
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    timeliness_score: float = Field(..., ge=0.0, le=1.0)
    issues: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[str]] = None
    assessed_by: Optional[str] = None


class BiasAssessmentUpdate(BaseModel):
    protected_attributes: List[str]
    statistical_parity: float = Field(..., ge=0.0, le=1.0)
    equalized_odds: float = Field(..., ge=0.0, le=1.0)
    demographic_parity: float = Field(..., ge=0.0, le=1.0)
    individual_fairness: float = Field(..., ge=0.0, le=1.0)
    bias_issues: Optional[List[Dict[str, Any]]] = None
    mitigation_strategies: Optional[List[str]] = None
    assessed_by: Optional[str] = None


class VerificationUpdate(BaseModel):
    verification_status: VerificationStatus
    verified_by: Optional[str] = None


# Dependency to get registry instance
def get_registry(db: Session = Depends(get_db)) -> DataProductRegistry:
    return DataProductRegistry(db)


def get_search_engine(db: Session = Depends(get_db)) -> DataProductSearchEngine:
    return DataProductSearchEngine(db)


def get_indexer() -> DataProductIndexer:
    return DataProductIndexer()


@router.post("/", response_model=DataProductResponse, status_code=status.HTTP_201_CREATED)
@rate_limit(max_requests=50, window_seconds=3600)
async def create_data_product(
    request: Request,
    data_product: DataProductCreate,
    owner: str = Query(..., description="Owner of the data product"),
    registry: DataProductRegistry = Depends(get_registry),
    indexer: DataProductIndexer = Depends(get_indexer),
    user_info: dict = Depends(require_permission("data_product:create"))
):
    """Create a new data product"""
    try:
        product = registry.create_data_product(
            name=data_product.name,
            schema_definition=data_product.schema_definition,
            owner=owner,
            description=data_product.description,
            metadata=data_product.metadata,
            access_level=data_product.access_level,
            compliance_tags=data_product.compliance_tags
        )
        
        # Index in Elasticsearch
        indexer.index_data_product(product)
        
        return DataProductResponse.from_orm(product)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create data product: {str(e)}")


@router.get("/{product_id}", response_model=DataProductResponse)
@rate_limit(max_requests=200, window_seconds=3600)
async def get_data_product(
    request: Request,
    product_id: str,
    version: Optional[str] = Query(None, description="Specific version to retrieve"),
    registry: DataProductRegistry = Depends(get_registry),
    user_info: dict = Depends(require_permission("data_product:read"))
):
    """Get data product by ID"""
    product = registry.get_data_product(product_id, version)
    
    if not product:
        raise HTTPException(status_code=404, detail="Data product not found")
    
    return DataProductResponse.from_orm(product)


@router.get("/", response_model=DataProductSearchResponse)
@rate_limit(max_requests=100, window_seconds=3600)
async def search_data_products(
    request: Request,
    query: Optional[str] = Query(None, description="Search query"),
    owner: Optional[str] = Query(None, description="Filter by owner"),
    access_level: Optional[AccessLevel] = Query(None, description="Filter by access level"),
    verification_status: Optional[VerificationStatus] = Query(None, description="Filter by verification status"),
    compliance_tags: Optional[List[str]] = Query(None, description="Filter by compliance tags"),
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    registry: DataProductRegistry = Depends(get_registry),
    user_info: dict = Depends(require_permission("data_product:search"))
):
    """Search data products with filters"""
    try:
        results, total_count = registry.search_data_products(
            query=query,
            owner=owner,
            access_level=access_level,
            verification_status=verification_status,
            compliance_tags=compliance_tags,
            limit=limit,
            offset=offset
        )
        
        return DataProductSearchResponse(
            results=[DataProductResponse.from_orm(product) for product in results],
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.put("/{product_id}", response_model=DataProductResponse)
@rate_limit(max_requests=30, window_seconds=3600)
async def update_data_product(
    request: Request,
    product_id: str,
    update_data: DataProductUpdate,
    updated_by: str = Query(..., description="User making the update"),
    registry: DataProductRegistry = Depends(get_registry),
    indexer: DataProductIndexer = Depends(get_indexer),
    user_info: dict = Depends(require_permission("data_product:update"))
):
    """Update data product"""
    try:
        product = registry.update_data_product(
            product_id=product_id,
            schema_definition=update_data.schema_definition,
            metadata=update_data.metadata,
            description=update_data.description,
            updated_by=updated_by
        )
        
        # Update index
        indexer.index_data_product(product)
        
        return DataProductResponse.from_orm(product)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update data product: {str(e)}")


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
@rate_limit(max_requests=10, window_seconds=3600)
async def delete_data_product(
    request: Request,
    product_id: str,
    registry: DataProductRegistry = Depends(get_registry),
    indexer: DataProductIndexer = Depends(get_indexer),
    user_info: dict = Depends(require_permission("data_product:delete"))
):
    """Delete data product"""
    try:
        success = registry.delete_data_product(product_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Data product not found")
        
        # Remove from index
        indexer.delete_data_product(product_id)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete data product: {str(e)}")


@router.get("/{product_id}/versions")
async def get_data_product_versions(
    product_id: str,
    registry: DataProductRegistry = Depends(get_registry)
):
    """Get all versions of a data product"""
    try:
        versions = registry.get_data_product_versions(product_id)
        
        return {
            "product_id": product_id,
            "versions": [
                {
                    "id": str(version.id),
                    "version_number": version.version_number,
                    "version_hash": version.version_hash,
                    "schema_hash": version.schema_hash,
                    "change_description": version.change_description,
                    "change_type": version.change_type,
                    "is_active": version.is_active,
                    "created_at": version.created_at,
                    "created_by": version.created_by
                }
                for version in versions
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get versions: {str(e)}")


@router.post("/{product_id}/provenance")
async def create_provenance_record(
    product_id: str,
    provenance_data: ProvenanceCreate,
    registry: DataProductRegistry = Depends(get_registry)
):
    """Create provenance record for data product"""
    try:
        provenance = registry.create_provenance_record(
            product_id=product_id,
            source_systems=provenance_data.source_systems,
            transformations=provenance_data.transformations,
            lineage_graph=provenance_data.lineage_graph
        )
        
        return {
            "id": str(provenance.id),
            "data_product_id": str(provenance.data_product_id),
            "source_systems": provenance.source_systems,
            "transformations": provenance.transformations,
            "lineage_graph": provenance.lineage_graph,
            "provenance_hash": provenance.provenance_hash,
            "is_verified": provenance.is_verified,
            "creation_timestamp": provenance.creation_timestamp,
            "last_modified": provenance.last_modified
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create provenance record: {str(e)}")


@router.put("/{product_id}/quality-metrics")
async def update_quality_metrics(
    product_id: str,
    metrics_data: QualityMetricsUpdate,
    registry: DataProductRegistry = Depends(get_registry)
):
    """Update quality metrics for data product"""
    try:
        metrics = registry.update_quality_metrics(
            product_id=product_id,
            completeness_score=metrics_data.completeness_score,
            accuracy_score=metrics_data.accuracy_score,
            consistency_score=metrics_data.consistency_score,
            timeliness_score=metrics_data.timeliness_score,
            issues=metrics_data.issues,
            recommendations=metrics_data.recommendations,
            assessed_by=metrics_data.assessed_by
        )
        
        return {
            "id": str(metrics.id),
            "data_product_id": str(metrics.data_product_id),
            "completeness_score": metrics.completeness_score,
            "accuracy_score": metrics.accuracy_score,
            "consistency_score": metrics.consistency_score,
            "timeliness_score": metrics.timeliness_score,
            "overall_score": metrics.overall_score,
            "issues": metrics.issues,
            "recommendations": metrics.recommendations,
            "assessed_at": metrics.assessed_at,
            "assessed_by": metrics.assessed_by
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update quality metrics: {str(e)}")


@router.put("/{product_id}/bias-assessment")
async def update_bias_assessment(
    product_id: str,
    assessment_data: BiasAssessmentUpdate,
    registry: DataProductRegistry = Depends(get_registry)
):
    """Update bias assessment for data product"""
    try:
        assessment = registry.update_bias_assessment(
            product_id=product_id,
            protected_attributes=assessment_data.protected_attributes,
            statistical_parity=assessment_data.statistical_parity,
            equalized_odds=assessment_data.equalized_odds,
            demographic_parity=assessment_data.demographic_parity,
            individual_fairness=assessment_data.individual_fairness,
            bias_issues=assessment_data.bias_issues,
            mitigation_strategies=assessment_data.mitigation_strategies,
            assessed_by=assessment_data.assessed_by
        )
        
        return {
            "id": str(assessment.id),
            "data_product_id": str(assessment.data_product_id),
            "protected_attributes": assessment.protected_attributes,
            "statistical_parity": assessment.statistical_parity,
            "equalized_odds": assessment.equalized_odds,
            "demographic_parity": assessment.demographic_parity,
            "individual_fairness": assessment.individual_fairness,
            "bias_issues": assessment.bias_issues,
            "mitigation_strategies": assessment.mitigation_strategies,
            "assessed_at": assessment.assessed_at,
            "assessed_by": assessment.assessed_by
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update bias assessment: {str(e)}")


@router.put("/{product_id}/verification")
async def update_verification_status(
    product_id: str,
    verification_data: VerificationUpdate,
    registry: DataProductRegistry = Depends(get_registry),
    indexer: DataProductIndexer = Depends(get_indexer)
):
    """Update verification status of data product"""
    try:
        product = registry.verify_data_product(
            product_id=product_id,
            verification_status=verification_data.verification_status,
            verified_by=verification_data.verified_by
        )
        
        # Update index
        indexer.index_data_product(product)
        
        return {
            "id": str(product.id),
            "verification_status": product.verification_status,
            "updated_at": product.updated_at
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update verification status: {str(e)}")


# Advanced search endpoints
@router.get("/search/semantic")
async def semantic_search(
    query: str = Query(..., description="Semantic search query"),
    limit: int = Query(20, ge=1, le=100),
    search_engine: DataProductSearchEngine = Depends(get_search_engine)
):
    """Semantic search for data products"""
    try:
        results = search_engine.semantic_search(query, limit)
        
        return {
            "query": query,
            "results": [DataProductResponse.from_orm(product) for product in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.get("/search/faceted")
async def faceted_search(
    facets: Dict[str, List[str]] = Query(..., description="Facet filters"),
    limit: int = Query(50, ge=1, le=100),
    search_engine: DataProductSearchEngine = Depends(get_search_engine)
):
    """Faceted search with multiple filter dimensions"""
    try:
        results = search_engine.faceted_search(facets, limit)
        
        return {
            "facets": facets,
            "results": [DataProductResponse.from_orm(product) for product in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Faceted search failed: {str(e)}")


@router.get("/{product_id}/related")
async def get_related_products(
    product_id: str,
    limit: int = Query(10, ge=1, le=50),
    search_engine: DataProductSearchEngine = Depends(get_search_engine)
):
    """Find related data products"""
    try:
        results = search_engine.get_related_products(product_id, limit)
        
        return {
            "product_id": product_id,
            "related_products": [DataProductResponse.from_orm(product) for product in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find related products: {str(e)}")


# Elasticsearch-powered search endpoints
@router.get("/search/elasticsearch")
async def elasticsearch_search(
    query: Optional[str] = Query(None, description="Search query"),
    filters: Optional[Dict[str, Any]] = Query(None, description="Search filters"),
    sort_by: Optional[str] = Query(None, description="Sort field"),
    sort_order: str = Query("desc", description="Sort order"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    indexer: DataProductIndexer = Depends(get_indexer)
):
    """Full-text search using Elasticsearch"""
    try:
        results, total_count = indexer.search_data_products(
            query=query,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        return {
            "query": query,
            "filters": filters,
            "results": results,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {str(e)}")


@router.get("/search/faceted-elasticsearch")
async def elasticsearch_faceted_search(
    query: Optional[str] = Query(None, description="Search query"),
    facets: Optional[List[str]] = Query(None, description="Facets to include"),
    filters: Optional[Dict[str, Any]] = Query(None, description="Search filters"),
    limit: int = Query(50, ge=1, le=100),
    indexer: DataProductIndexer = Depends(get_indexer)
):
    """Faceted search using Elasticsearch"""
    try:
        result = indexer.faceted_search(
            query=query,
            facets=facets,
            filters=filters,
            limit=limit
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Elasticsearch faceted search failed: {str(e)}")


@router.get("/suggest")
async def suggest_data_products(
    partial_query: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(10, ge=1, le=20),
    indexer: DataProductIndexer = Depends(get_indexer)
):
    """Auto-suggest data product names"""
    try:
        suggestions = indexer.suggest_data_products(partial_query, limit)
        
        return {
            "partial_query": partial_query,
            "suggestions": suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for data product service"""
    return {"status": "healthy", "service": "data-product-registry"}