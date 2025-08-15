"""
API Routes for Pattern Recognition Engine

This module provides REST API endpoints for the pattern recognition engine,
enabling pattern recognition, analysis, interpretation, and innovation optimization.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.pattern_recognition_engine import PatternRecognitionEngine
from ...engines.knowledge_synthesis_framework import KnowledgeSynthesisFramework
from ...models.knowledge_integration_models import (
    Pattern, PatternRecognitionResult, PatternType, KnowledgeItem
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pattern-recognition", tags=["Pattern Recognition"])

# Global instances (in production, use dependency injection)
pattern_engine = PatternRecognitionEngine()
knowledge_synthesis = KnowledgeSynthesisFramework()


@router.post("/recognize", response_model=Dict[str, Any])
async def recognize_patterns(
    knowledge_item_ids: List[str],
    pattern_types: Optional[List[str]] = None
):
    """
    Recognize patterns across knowledge items
    
    Args:
        knowledge_item_ids: List of knowledge item IDs to analyze
        pattern_types: Optional list of specific pattern types to look for
        
    Returns:
        Pattern recognition result
    """
    try:
        # Get knowledge items
        knowledge_items = []
        for item_id in knowledge_item_ids:
            if item_id in knowledge_synthesis.knowledge_store:
                knowledge_items.append(knowledge_synthesis.knowledge_store[item_id])
        
        if not knowledge_items:
            raise HTTPException(status_code=404, detail="No valid knowledge items found")
        
        # Convert pattern type strings to enums
        pattern_type_enums = None
        if pattern_types:
            try:
                pattern_type_enums = [PatternType(pt) for pt in pattern_types]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid pattern type: {str(e)}")
        
        # Recognize patterns
        result = await pattern_engine.recognize_patterns(knowledge_items, pattern_type_enums)
        
        return {
            "patterns_found": [
                {
                    "id": pattern.id,
                    "pattern_type": pattern.pattern_type.value,
                    "description": pattern.description,
                    "evidence": pattern.evidence,
                    "strength": pattern.strength,
                    "confidence": pattern.confidence.value,
                    "discovered_at": pattern.discovered_at.isoformat(),
                    "predictive_power": pattern.predictive_power,
                    "applications": pattern.applications
                }
                for pattern in result.patterns_found
            ],
            "analysis_method": result.analysis_method,
            "confidence": result.confidence.value,
            "processing_time": result.processing_time,
            "recommendations": result.recommendations,
            "discovered_at": result.discovered_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recognizing patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-significance/{pattern_id}", response_model=Dict[str, Any])
async def analyze_pattern_significance(
    pattern_id: str,
    knowledge_item_ids: Optional[List[str]] = None
):
    """
    Analyze the significance of a specific pattern
    
    Args:
        pattern_id: ID of pattern to analyze
        knowledge_item_ids: Optional context knowledge item IDs
        
    Returns:
        Pattern significance analysis
    """
    try:
        # Get pattern
        if pattern_id not in pattern_engine.patterns:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        pattern = pattern_engine.patterns[pattern_id]
        
        # Get context knowledge items
        knowledge_items = []
        if knowledge_item_ids:
            for item_id in knowledge_item_ids:
                if item_id in knowledge_synthesis.knowledge_store:
                    knowledge_items.append(knowledge_synthesis.knowledge_store[item_id])
        else:
            # Use all available knowledge items as context
            knowledge_items = list(knowledge_synthesis.knowledge_store.values())
        
        if not knowledge_items:
            raise HTTPException(status_code=404, detail="No context knowledge items found")
        
        # Analyze significance
        significance_analysis = await pattern_engine.analyze_pattern_significance(pattern, knowledge_items)
        
        return significance_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing pattern significance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret", response_model=Dict[str, Any])
async def interpret_patterns(
    pattern_ids: List[str],
    context: Optional[Dict[str, Any]] = None
):
    """
    Interpret patterns to extract meaningful insights
    
    Args:
        pattern_ids: List of pattern IDs to interpret
        context: Optional context information
        
    Returns:
        Pattern interpretation results
    """
    try:
        # Get patterns
        patterns = []
        for pattern_id in pattern_ids:
            if pattern_id in pattern_engine.patterns:
                patterns.append(pattern_engine.patterns[pattern_id])
        
        if not patterns:
            raise HTTPException(status_code=404, detail="No valid patterns found")
        
        # Interpret patterns
        interpretation_result = await pattern_engine.interpret_patterns(patterns, context)
        
        return interpretation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error interpreting patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-innovation", response_model=Dict[str, Any])
async def optimize_innovation_based_on_patterns(
    pattern_ids: List[str],
    innovation_context: Dict[str, Any]
):
    """
    Optimize innovation based on recognized patterns
    
    Args:
        pattern_ids: List of pattern IDs to use for optimization
        innovation_context: Context about the innovation to optimize
        
    Returns:
        Innovation optimization recommendations
    """
    try:
        # Get patterns
        patterns = []
        for pattern_id in pattern_ids:
            if pattern_id in pattern_engine.patterns:
                patterns.append(pattern_engine.patterns[pattern_id])
        
        if not patterns:
            raise HTTPException(status_code=404, detail="No valid patterns found")
        
        # Optimize innovation
        optimization_result = await pattern_engine.optimize_innovation_based_on_patterns(
            patterns, innovation_context
        )
        
        return optimization_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing innovation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhance-pipeline", response_model=Dict[str, Any])
async def enhance_innovation_pipeline(
    pattern_ids: List[str],
    pipeline_context: Dict[str, Any]
):
    """
    Enhance innovation pipeline based on pattern insights
    
    Args:
        pattern_ids: List of pattern IDs to use for enhancement
        pipeline_context: Context about the innovation pipeline
        
    Returns:
        Pipeline enhancement recommendations
    """
    try:
        # Get patterns
        patterns = []
        for pattern_id in pattern_ids:
            if pattern_id in pattern_engine.patterns:
                patterns.append(pattern_engine.patterns[pattern_id])
        
        if not patterns:
            raise HTTPException(status_code=404, detail="No valid patterns found")
        
        # Enhance pipeline
        enhancement_result = await pattern_engine.enhance_innovation_pipeline(
            patterns, pipeline_context
        )
        
        return enhancement_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns", response_model=List[Dict[str, Any]])
async def list_patterns(
    pattern_type: Optional[str] = None,
    min_strength: float = 0.0,
    min_confidence: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List recognized patterns with optional filtering
    
    Args:
        pattern_type: Filter by pattern type
        min_strength: Minimum pattern strength threshold
        min_confidence: Minimum confidence level
        limit: Maximum number of patterns to return
        offset: Number of patterns to skip
        
    Returns:
        List of patterns
    """
    try:
        patterns = list(pattern_engine.patterns.values())
        
        # Apply filters
        if pattern_type:
            try:
                pattern_type_enum = PatternType(pattern_type)
                patterns = [p for p in patterns if p.pattern_type == pattern_type_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid pattern type: {pattern_type}")
        
        if min_strength > 0:
            patterns = [p for p in patterns if p.strength >= min_strength]
        
        if min_confidence:
            try:
                from ...models.knowledge_integration_models import ConfidenceLevel
                min_conf_enum = ConfidenceLevel(min_confidence)
                confidence_order = {
                    ConfidenceLevel.LOW: 1,
                    ConfidenceLevel.MEDIUM: 2,
                    ConfidenceLevel.HIGH: 3,
                    ConfidenceLevel.VERY_HIGH: 4
                }
                min_conf_value = confidence_order[min_conf_enum]
                patterns = [p for p in patterns if confidence_order[p.confidence] >= min_conf_value]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid confidence level: {min_confidence}")
        
        # Sort by strength (descending)
        patterns.sort(key=lambda x: x.strength, reverse=True)
        
        # Apply pagination
        patterns = patterns[offset:offset + limit]
        
        return [
            {
                "id": pattern.id,
                "pattern_type": pattern.pattern_type.value,
                "description": pattern.description,
                "evidence_count": len(pattern.evidence),
                "strength": pattern.strength,
                "confidence": pattern.confidence.value,
                "discovered_at": pattern.discovered_at.isoformat(),
                "predictive_power": pattern.predictive_power,
                "applications_count": len(pattern.applications)
            }
            for pattern in patterns
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{pattern_id}", response_model=Dict[str, Any])
async def get_pattern(pattern_id: str):
    """
    Get specific pattern details
    
    Args:
        pattern_id: ID of pattern to retrieve
        
    Returns:
        Pattern details
    """
    try:
        if pattern_id not in pattern_engine.patterns:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        pattern = pattern_engine.patterns[pattern_id]
        
        return {
            "id": pattern.id,
            "pattern_type": pattern.pattern_type.value,
            "description": pattern.description,
            "evidence": pattern.evidence,
            "strength": pattern.strength,
            "confidence": pattern.confidence.value,
            "discovered_at": pattern.discovered_at.isoformat(),
            "predictive_power": pattern.predictive_power,
            "applications": pattern.applications
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_pattern_stats():
    """
    Get pattern recognition statistics
    
    Returns:
        Statistics about recognized patterns
    """
    try:
        patterns = list(pattern_engine.patterns.values())
        
        if not patterns:
            return {
                "total_patterns": 0,
                "pattern_types": {},
                "confidence_distribution": {},
                "average_strength": 0.0,
                "average_predictive_power": 0.0
            }
        
        # Calculate statistics
        stats = {
            "total_patterns": len(patterns),
            "pattern_types": {},
            "confidence_distribution": {},
            "average_strength": 0.0,
            "average_predictive_power": 0.0,
            "strength_distribution": {
                "high": 0,  # > 0.7
                "medium": 0,  # 0.4 - 0.7
                "low": 0  # < 0.4
            }
        }
        
        # Pattern type distribution
        for pattern in patterns:
            type_name = pattern.pattern_type.value
            stats["pattern_types"][type_name] = stats["pattern_types"].get(type_name, 0) + 1
        
        # Confidence distribution
        for pattern in patterns:
            confidence_name = pattern.confidence.value
            stats["confidence_distribution"][confidence_name] = stats["confidence_distribution"].get(confidence_name, 0) + 1
        
        # Average strength
        stats["average_strength"] = sum(p.strength for p in patterns) / len(patterns)
        
        # Average predictive power
        predictive_powers = [p.predictive_power for p in patterns if p.predictive_power > 0]
        if predictive_powers:
            stats["average_predictive_power"] = sum(predictive_powers) / len(predictive_powers)
        
        # Strength distribution
        for pattern in patterns:
            if pattern.strength > 0.7:
                stats["strength_distribution"]["high"] += 1
            elif pattern.strength > 0.4:
                stats["strength_distribution"]["medium"] += 1
            else:
                stats["strength_distribution"]["low"] += 1
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting pattern stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/patterns/{pattern_id}")
async def delete_pattern(pattern_id: str):
    """
    Delete a specific pattern
    
    Args:
        pattern_id: ID of pattern to delete
        
    Returns:
        Success message
    """
    try:
        if pattern_id not in pattern_engine.patterns:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        del pattern_engine.patterns[pattern_id]
        
        return {"message": f"Pattern {pattern_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-recognize", response_model=Dict[str, Any])
async def batch_recognize_patterns(
    batch_requests: List[Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Batch recognize patterns for multiple knowledge item sets
    
    Args:
        batch_requests: List of pattern recognition requests
        background_tasks: Background task manager
        
    Returns:
        Batch processing results
    """
    try:
        results = []
        
        for i, request in enumerate(batch_requests):
            knowledge_item_ids = request.get("knowledge_item_ids", [])
            pattern_types = request.get("pattern_types")
            
            # Get knowledge items
            knowledge_items = []
            for item_id in knowledge_item_ids:
                if item_id in knowledge_synthesis.knowledge_store:
                    knowledge_items.append(knowledge_synthesis.knowledge_store[item_id])
            
            if knowledge_items:
                # Convert pattern type strings to enums
                pattern_type_enums = None
                if pattern_types:
                    try:
                        pattern_type_enums = [PatternType(pt) for pt in pattern_types]
                    except ValueError:
                        pattern_type_enums = None
                
                # Recognize patterns
                result = await pattern_engine.recognize_patterns(knowledge_items, pattern_type_enums)
                
                results.append({
                    "request_index": i,
                    "patterns_found_count": len(result.patterns_found),
                    "processing_time": result.processing_time,
                    "confidence": result.confidence.value,
                    "status": "success"
                })
            else:
                results.append({
                    "request_index": i,
                    "patterns_found_count": 0,
                    "status": "no_valid_knowledge_items"
                })
        
        return {
            "batch_results": results,
            "total_requests": len(batch_requests),
            "successful_requests": len([r for r in results if r["status"] == "success"]),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch pattern recognition: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))