"""
API Routes for Knowledge Synthesis Framework

This module provides REST API endpoints for the knowledge synthesis framework,
enabling integration of research findings, experimental results, and knowledge validation.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.knowledge_synthesis_framework import KnowledgeSynthesisFramework
from ...models.knowledge_integration_models import (
    KnowledgeItem, SynthesizedKnowledge, KnowledgeValidationResult,
    KnowledgeGraph, SynthesisRequest, KnowledgeCorrelation
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge-synthesis", tags=["Knowledge Synthesis"])

# Global instance (in production, use dependency injection)
knowledge_synthesis = KnowledgeSynthesisFramework()


@router.post("/integrate/research-findings", response_model=List[Dict[str, Any]])
async def integrate_research_findings(
    findings: List[Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Integrate research findings into the knowledge base
    
    Args:
        findings: List of research findings to integrate
        
    Returns:
        List of created knowledge items
    """
    try:
        knowledge_items = await knowledge_synthesis.integrate_research_findings(findings)
        
        # Schedule background correlation analysis
        background_tasks.add_task(
            knowledge_synthesis.identify_knowledge_correlations,
            [item.id for item in knowledge_items]
        )
        
        return [
            {
                "id": item.id,
                "knowledge_type": item.knowledge_type.value,
                "source": item.source,
                "confidence": item.confidence.value,
                "timestamp": item.timestamp.isoformat(),
                "tags": item.tags
            }
            for item in knowledge_items
        ]
        
    except Exception as e:
        logger.error(f"Error integrating research findings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrate/experimental-results", response_model=List[Dict[str, Any]])
async def integrate_experimental_results(
    results: List[Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Integrate experimental results into the knowledge base
    
    Args:
        results: List of experimental results to integrate
        
    Returns:
        List of created knowledge items
    """
    try:
        knowledge_items = await knowledge_synthesis.integrate_experimental_results(results)
        
        # Schedule background correlation analysis
        background_tasks.add_task(
            knowledge_synthesis.identify_knowledge_correlations,
            [item.id for item in knowledge_items]
        )
        
        return [
            {
                "id": item.id,
                "knowledge_type": item.knowledge_type.value,
                "source": item.source,
                "confidence": item.confidence.value,
                "timestamp": item.timestamp.isoformat(),
                "tags": item.tags
            }
            for item in knowledge_items
        ]
        
    except Exception as e:
        logger.error(f"Error integrating experimental results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlations", response_model=List[Dict[str, Any]])
async def get_knowledge_correlations(
    knowledge_ids: Optional[List[str]] = None,
    min_strength: float = 0.3
):
    """
    Get knowledge correlations
    
    Args:
        knowledge_ids: Optional list of specific knowledge IDs to analyze
        min_strength: Minimum correlation strength threshold
        
    Returns:
        List of knowledge correlations
    """
    try:
        correlations = await knowledge_synthesis.identify_knowledge_correlations(knowledge_ids)
        
        # Filter by minimum strength
        filtered_correlations = [
            corr for corr in correlations 
            if corr.strength >= min_strength
        ]
        
        return [
            {
                "id": corr.id,
                "item_ids": corr.item_ids,
                "correlation_type": corr.correlation_type,
                "strength": corr.strength,
                "confidence": corr.confidence.value,
                "description": corr.description,
                "discovered_at": corr.discovered_at.isoformat()
            }
            for corr in filtered_correlations
        ]
        
    except Exception as e:
        logger.error(f"Error getting correlations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize", response_model=Dict[str, Any])
async def synthesize_knowledge(synthesis_request: Dict[str, Any]):
    """
    Synthesize knowledge from multiple sources
    
    Args:
        synthesis_request: Synthesis request parameters
        
    Returns:
        Synthesized knowledge result
    """
    try:
        # Create synthesis request object
        request = SynthesisRequest(
            id=f"synthesis_request_{datetime.now().timestamp()}",
            source_knowledge_ids=synthesis_request["source_knowledge_ids"],
            synthesis_goal=synthesis_request["synthesis_goal"],
            method_preferences=synthesis_request.get("method_preferences", []),
            constraints=synthesis_request.get("constraints", {}),
            priority=synthesis_request.get("priority", "medium")
        )
        
        synthesized = await knowledge_synthesis.synthesize_knowledge(request)
        
        return {
            "id": synthesized.id,
            "source_items": synthesized.source_items,
            "synthesis_method": synthesized.synthesis_method,
            "synthesized_content": synthesized.synthesized_content,
            "insights": synthesized.insights,
            "confidence": synthesized.confidence.value,
            "quality_score": synthesized.quality_score,
            "created_at": synthesized.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error synthesizing knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/{knowledge_id}", response_model=Dict[str, Any])
async def validate_knowledge(
    knowledge_id: str,
    validation_methods: Optional[List[str]] = None
):
    """
    Validate knowledge item for quality assurance
    
    Args:
        knowledge_id: ID of knowledge item to validate
        validation_methods: List of validation methods to use
        
    Returns:
        Validation result
    """
    try:
        validation_result = await knowledge_synthesis.validate_knowledge(
            knowledge_id, validation_methods
        )
        
        return {
            "knowledge_id": validation_result.knowledge_id,
            "validation_method": validation_result.validation_method,
            "is_valid": validation_result.is_valid,
            "confidence": validation_result.confidence.value,
            "validation_score": validation_result.validation_score,
            "issues_found": validation_result.issues_found,
            "recommendations": validation_result.recommendations,
            "validated_at": validation_result.validated_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph", response_model=Dict[str, Any])
async def get_knowledge_graph(knowledge_ids: Optional[List[str]] = None):
    """
    Get knowledge graph representation
    
    Args:
        knowledge_ids: Optional list of specific knowledge IDs to include
        
    Returns:
        Knowledge graph data
    """
    try:
        knowledge_graph = await knowledge_synthesis.create_knowledge_graph(knowledge_ids)
        
        return {
            "nodes": [
                {
                    "id": node.id,
                    "knowledge_type": node.knowledge_type.value,
                    "source": node.source,
                    "confidence": node.confidence.value,
                    "tags": node.tags,
                    "relationships": node.relationships
                }
                for node in knowledge_graph.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "item_ids": edge.item_ids,
                    "correlation_type": edge.correlation_type,
                    "strength": edge.strength,
                    "confidence": edge.confidence.value
                }
                for edge in knowledge_graph.edges
            ],
            "metadata": knowledge_graph.metadata,
            "created_at": knowledge_graph.created_at.isoformat(),
            "last_updated": knowledge_graph.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/{knowledge_id}", response_model=Dict[str, Any])
async def get_knowledge_item(knowledge_id: str):
    """
    Get specific knowledge item
    
    Args:
        knowledge_id: ID of knowledge item to retrieve
        
    Returns:
        Knowledge item data
    """
    try:
        if knowledge_id not in knowledge_synthesis.knowledge_store:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        knowledge_item = knowledge_synthesis.knowledge_store[knowledge_id]
        
        return {
            "id": knowledge_item.id,
            "knowledge_type": knowledge_item.knowledge_type.value,
            "content": knowledge_item.content,
            "source": knowledge_item.source,
            "timestamp": knowledge_item.timestamp.isoformat(),
            "confidence": knowledge_item.confidence.value,
            "metadata": knowledge_item.metadata,
            "tags": knowledge_item.tags,
            "relationships": knowledge_item.relationships
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge", response_model=List[Dict[str, Any]])
async def list_knowledge_items(
    knowledge_type: Optional[str] = None,
    source: Optional[str] = None,
    confidence: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List knowledge items with optional filtering
    
    Args:
        knowledge_type: Filter by knowledge type
        source: Filter by source
        confidence: Filter by confidence level
        limit: Maximum number of items to return
        offset: Number of items to skip
        
    Returns:
        List of knowledge items
    """
    try:
        items = list(knowledge_synthesis.knowledge_store.values())
        
        # Apply filters
        if knowledge_type:
            items = [item for item in items if item.knowledge_type.value == knowledge_type]
        
        if source:
            items = [item for item in items if item.source == source]
        
        if confidence:
            items = [item for item in items if item.confidence.value == confidence]
        
        # Apply pagination
        items = items[offset:offset + limit]
        
        return [
            {
                "id": item.id,
                "knowledge_type": item.knowledge_type.value,
                "source": item.source,
                "timestamp": item.timestamp.isoformat(),
                "confidence": item.confidence.value,
                "tags": item.tags,
                "relationships_count": len(item.relationships)
            }
            for item in items
        ]
        
    except Exception as e:
        logger.error(f"Error listing knowledge items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/synthesized", response_model=List[Dict[str, Any]])
async def list_synthesized_knowledge(
    limit: int = 50,
    offset: int = 0
):
    """
    List synthesized knowledge items
    
    Args:
        limit: Maximum number of items to return
        offset: Number of items to skip
        
    Returns:
        List of synthesized knowledge items
    """
    try:
        items = list(knowledge_synthesis.synthesized_knowledge.values())
        
        # Sort by creation date (newest first)
        items.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        items = items[offset:offset + limit]
        
        return [
            {
                "id": item.id,
                "source_items_count": len(item.source_items),
                "synthesis_method": item.synthesis_method,
                "insights_count": len(item.insights),
                "confidence": item.confidence.value,
                "quality_score": item.quality_score,
                "created_at": item.created_at.isoformat()
            }
            for item in items
        ]
        
    except Exception as e:
        logger.error(f"Error listing synthesized knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_knowledge_stats():
    """
    Get knowledge synthesis statistics
    
    Returns:
        Statistics about the knowledge base
    """
    try:
        knowledge_items = list(knowledge_synthesis.knowledge_store.values())
        correlations = list(knowledge_synthesis.correlations.values())
        synthesized_items = list(knowledge_synthesis.synthesized_knowledge.values())
        
        # Calculate statistics
        stats = {
            "total_knowledge_items": len(knowledge_items),
            "total_correlations": len(correlations),
            "total_synthesized_items": len(synthesized_items),
            "knowledge_types": {},
            "confidence_distribution": {},
            "average_correlation_strength": 0.0,
            "average_quality_score": 0.0
        }
        
        # Knowledge type distribution
        for item in knowledge_items:
            type_name = item.knowledge_type.value
            stats["knowledge_types"][type_name] = stats["knowledge_types"].get(type_name, 0) + 1
        
        # Confidence distribution
        for item in knowledge_items:
            confidence_name = item.confidence.value
            stats["confidence_distribution"][confidence_name] = stats["confidence_distribution"].get(confidence_name, 0) + 1
        
        # Average correlation strength
        if correlations:
            stats["average_correlation_strength"] = sum(corr.strength for corr in correlations) / len(correlations)
        
        # Average quality score
        if synthesized_items:
            stats["average_quality_score"] = sum(item.quality_score for item in synthesized_items) / len(synthesized_items)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))