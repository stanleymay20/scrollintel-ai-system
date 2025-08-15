"""
API routes for Rapid Prototyping System

This module provides REST API endpoints for the autonomous innovation lab's
rapid prototyping capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ...engines.rapid_prototyper import RapidPrototyper
from ...models.prototype_models import (
    Concept, Prototype, PrototypeType, PrototypeStatus,
    ConceptCategory, TechnologyStack, QualityMetrics,
    ValidationResult, PrototypingSession, create_concept_from_description
)
from ...core.config import get_settings
from ...security.auth import get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/rapid-prototyping", tags=["Rapid Prototyping"])

# Global rapid prototyper instance
rapid_prototyper = RapidPrototyper()


# Request/Response models
from pydantic import BaseModel, Field

class ConceptCreateRequest(BaseModel):
    name: str = Field(..., description="Name of the concept")
    description: str = Field(..., description="Detailed description of the concept")
    category: ConceptCategory = Field(default=ConceptCategory.TECHNOLOGY)
    requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    target_users: List[str] = Field(default_factory=list)
    business_value: str = Field(default="")
    technical_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    innovation_potential: float = Field(default=0.5, ge=0.0, le=1.0)
    market_readiness: float = Field(default=0.5, ge=0.0, le=1.0)


class PrototypeCreateRequest(BaseModel):
    concept_id: str = Field(..., description="ID of the concept to prototype")
    priority: str = Field(default="normal", description="Priority level")
    custom_requirements: List[str] = Field(default_factory=list)


class PrototypeOptimizeRequest(BaseModel):
    optimization_targets: List[str] = Field(default_factory=list)
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class PrototypeResponse(BaseModel):
    id: str
    concept_id: str
    name: str
    description: str
    prototype_type: PrototypeType
    status: PrototypeStatus
    development_progress: float
    creation_timestamp: datetime
    completion_timestamp: Optional[datetime]
    quality_score: Optional[float]
    is_deployable: bool
    
    class Config:
        from_attributes = True


class ConceptResponse(BaseModel):
    id: str
    name: str
    description: str
    category: ConceptCategory
    technical_complexity: float
    innovation_potential: float
    market_readiness: float
    creation_timestamp: datetime
    
    class Config:
        from_attributes = True


# Concept management endpoints

@router.post("/concepts", response_model=ConceptResponse)
async def create_concept(
    request: ConceptCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new innovation concept"""
    try:
        concept = Concept(
            name=request.name,
            description=request.description,
            category=request.category,
            requirements=request.requirements,
            constraints=request.constraints,
            success_criteria=request.success_criteria,
            target_users=request.target_users,
            business_value=request.business_value,
            technical_complexity=request.technical_complexity,
            innovation_potential=request.innovation_potential,
            market_readiness=request.market_readiness
        )
        
        logger.info(f"Created concept: {concept.name} by user {current_user.get('username')}")
        
        return ConceptResponse(
            id=concept.id,
            name=concept.name,
            description=concept.description,
            category=concept.category,
            technical_complexity=concept.technical_complexity,
            innovation_potential=concept.innovation_potential,
            market_readiness=concept.market_readiness,
            creation_timestamp=concept.creation_timestamp
        )
        
    except Exception as e:
        logger.error(f"Error creating concept: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create concept: {str(e)}")


@router.get("/concepts", response_model=List[ConceptResponse])
async def list_concepts(
    category: Optional[ConceptCategory] = None,
    min_innovation_potential: Optional[float] = None,
    current_user: dict = Depends(get_current_user)
):
    """List all innovation concepts with optional filtering"""
    try:
        # In a real implementation, this would query a database
        # For now, return empty list as concepts are created on-demand
        concepts = []
        
        logger.info(f"Listed {len(concepts)} concepts for user {current_user.get('username')}")
        return concepts
        
    except Exception as e:
        logger.error(f"Error listing concepts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list concepts: {str(e)}")


@router.get("/concepts/{concept_id}", response_model=ConceptResponse)
async def get_concept(
    concept_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific concept by ID"""
    try:
        # In a real implementation, this would query a database
        # For now, create a sample concept
        concept = create_concept_from_description(
            name="Sample Concept",
            description="This is a sample concept for demonstration",
            category=ConceptCategory.TECHNOLOGY
        )
        concept.id = concept_id
        
        return ConceptResponse(
            id=concept.id,
            name=concept.name,
            description=concept.description,
            category=concept.category,
            technical_complexity=concept.technical_complexity,
            innovation_potential=concept.innovation_potential,
            market_readiness=concept.market_readiness,
            creation_timestamp=concept.creation_timestamp
        )
        
    except Exception as e:
        logger.error(f"Error getting concept {concept_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Concept not found: {concept_id}")


# Prototype management endpoints

@router.post("/prototypes", response_model=PrototypeResponse)
async def create_prototype(
    request: PrototypeCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a rapid prototype from a concept"""
    try:
        # Create or get the concept
        concept = create_concept_from_description(
            name="Innovation Concept",
            description="Concept for rapid prototyping",
            category=ConceptCategory.TECHNOLOGY
        )
        concept.id = request.concept_id
        
        # Start prototype creation in background
        background_tasks.add_task(
            _create_prototype_background,
            concept,
            current_user.get('username', 'unknown')
        )
        
        # Return immediate response with placeholder
        prototype_response = PrototypeResponse(
            id="pending",
            concept_id=concept.id,
            name=f"{concept.name} Prototype",
            description=f"Rapid prototype for {concept.description}",
            prototype_type=PrototypeType.PROOF_OF_CONCEPT,
            status=PrototypeStatus.PLANNED,
            development_progress=0.0,
            creation_timestamp=datetime.utcnow(),
            completion_timestamp=None,
            quality_score=None,
            is_deployable=False
        )
        
        logger.info(f"Started prototype creation for concept {request.concept_id}")
        return prototype_response
        
    except Exception as e:
        logger.error(f"Error creating prototype: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create prototype: {str(e)}")


async def _create_prototype_background(concept: Concept, username: str):
    """Background task for prototype creation"""
    try:
        prototype = await rapid_prototyper.create_rapid_prototype(concept)
        logger.info(f"Prototype {prototype.id} created successfully for user {username}")
    except Exception as e:
        logger.error(f"Background prototype creation failed: {str(e)}")


@router.get("/prototypes", response_model=List[PrototypeResponse])
async def list_prototypes(
    status: Optional[PrototypeStatus] = None,
    prototype_type: Optional[PrototypeType] = None,
    current_user: dict = Depends(get_current_user)
):
    """List all prototypes with optional filtering"""
    try:
        prototypes = await rapid_prototyper.list_active_prototypes()
        
        # Apply filters
        if status:
            prototypes = [p for p in prototypes if p.status == status]
        if prototype_type:
            prototypes = [p for p in prototypes if p.prototype_type == prototype_type]
        
        # Convert to response format
        prototype_responses = []
        for prototype in prototypes:
            quality_score = None
            if prototype.validation_result:
                quality_score = prototype.validation_result.overall_score
            
            prototype_responses.append(PrototypeResponse(
                id=prototype.id,
                concept_id=prototype.concept_id,
                name=prototype.name,
                description=prototype.description,
                prototype_type=prototype.prototype_type,
                status=prototype.status,
                development_progress=prototype.development_progress,
                creation_timestamp=prototype.creation_timestamp,
                completion_timestamp=prototype.completion_timestamp,
                quality_score=quality_score,
                is_deployable=prototype.is_deployable
            ))
        
        logger.info(f"Listed {len(prototype_responses)} prototypes for user {current_user.get('username')}")
        return prototype_responses
        
    except Exception as e:
        logger.error(f"Error listing prototypes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list prototypes: {str(e)}")


@router.get("/prototypes/{prototype_id}", response_model=PrototypeResponse)
async def get_prototype(
    prototype_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific prototype by ID"""
    try:
        prototype = await rapid_prototyper.get_prototype_status(prototype_id)
        
        if not prototype:
            raise HTTPException(status_code=404, detail=f"Prototype not found: {prototype_id}")
        
        quality_score = None
        if prototype.validation_result:
            quality_score = prototype.validation_result.overall_score
        
        return PrototypeResponse(
            id=prototype.id,
            concept_id=prototype.concept_id,
            name=prototype.name,
            description=prototype.description,
            prototype_type=prototype.prototype_type,
            status=prototype.status,
            development_progress=prototype.development_progress,
            creation_timestamp=prototype.creation_timestamp,
            completion_timestamp=prototype.completion_timestamp,
            quality_score=quality_score,
            is_deployable=prototype.is_deployable
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prototype {prototype_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get prototype: {str(e)}")


@router.post("/prototypes/{prototype_id}/optimize", response_model=PrototypeResponse)
async def optimize_prototype(
    prototype_id: str,
    request: PrototypeOptimizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Optimize an existing prototype"""
    try:
        prototype = await rapid_prototyper.optimize_prototype(prototype_id)
        
        quality_score = None
        if prototype.validation_result:
            quality_score = prototype.validation_result.overall_score
        
        logger.info(f"Optimized prototype {prototype_id} for user {current_user.get('username')}")
        
        return PrototypeResponse(
            id=prototype.id,
            concept_id=prototype.concept_id,
            name=prototype.name,
            description=prototype.description,
            prototype_type=prototype.prototype_type,
            status=prototype.status,
            development_progress=prototype.development_progress,
            creation_timestamp=prototype.creation_timestamp,
            completion_timestamp=prototype.completion_timestamp,
            quality_score=quality_score,
            is_deployable=prototype.is_deployable
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing prototype {prototype_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize prototype: {str(e)}")


@router.get("/prototypes/{prototype_id}/code")
async def get_prototype_code(
    prototype_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the generated code for a prototype"""
    try:
        prototype = await rapid_prototyper.get_prototype_status(prototype_id)
        
        if not prototype:
            raise HTTPException(status_code=404, detail=f"Prototype not found: {prototype_id}")
        
        return {
            "prototype_id": prototype_id,
            "generated_code": prototype.generated_code,
            "file_structure": prototype.file_structure,
            "technology_stack": prototype.technology_stack.__dict__ if prototype.technology_stack else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prototype code {prototype_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get prototype code: {str(e)}")


@router.get("/prototypes/{prototype_id}/validation")
async def get_prototype_validation(
    prototype_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get validation results for a prototype"""
    try:
        prototype = await rapid_prototyper.get_prototype_status(prototype_id)
        
        if not prototype:
            raise HTTPException(status_code=404, detail=f"Prototype not found: {prototype_id}")
        
        if not prototype.validation_result:
            return {"message": "Prototype has not been validated yet"}
        
        return {
            "prototype_id": prototype_id,
            "validation_result": {
                "overall_score": prototype.validation_result.overall_score,
                "category_scores": prototype.validation_result.category_scores,
                "passes_validation": prototype.validation_result.passes_validation,
                "recommendations": prototype.validation_result.recommendations,
                "validation_timestamp": prototype.validation_result.validation_timestamp
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prototype validation {prototype_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get prototype validation: {str(e)}")


# Analytics and reporting endpoints

@router.get("/analytics/prototypes")
async def get_prototype_analytics(
    current_user: dict = Depends(get_current_user)
):
    """Get analytics data for all prototypes"""
    try:
        prototypes = await rapid_prototyper.list_active_prototypes()
        
        # Calculate analytics
        total_prototypes = len(prototypes)
        status_distribution = {}
        type_distribution = {}
        avg_development_time = 0
        success_rate = 0
        
        if prototypes:
            # Status distribution
            for prototype in prototypes:
                status = prototype.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1
            
            # Type distribution
            for prototype in prototypes:
                ptype = prototype.prototype_type.value
                type_distribution[ptype] = type_distribution.get(ptype, 0) + 1
            
            # Average development time
            completed_prototypes = [p for p in prototypes if p.completion_timestamp]
            if completed_prototypes:
                total_time = sum(p.actual_development_time for p in completed_prototypes)
                avg_development_time = total_time / len(completed_prototypes)
            
            # Success rate
            validated_prototypes = [p for p in prototypes if p.status == PrototypeStatus.VALIDATED]
            success_rate = len(validated_prototypes) / total_prototypes if total_prototypes > 0 else 0
        
        return {
            "total_prototypes": total_prototypes,
            "status_distribution": status_distribution,
            "type_distribution": type_distribution,
            "average_development_time_hours": avg_development_time,
            "success_rate": success_rate,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting prototype analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for rapid prototyping service"""
    try:
        # Check if rapid prototyper is working
        prototypes = await rapid_prototyper.list_active_prototypes()
        
        return {
            "status": "healthy",
            "service": "rapid-prototyping",
            "active_prototypes": len(prototypes),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "rapid-prototyping",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )