"""
Information Synthesis API Routes for Crisis Leadership Excellence

This module provides REST API endpoints for information synthesis operations
during crisis situations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
import logging
from datetime import datetime

from ...engines.information_synthesis_engine import InformationSynthesisEngine
from ...models.information_synthesis_models import (
    InformationItem, SynthesisRequest, SynthesizedInformation,
    FilterCriteria, UncertaintyAssessment, SynthesisMetrics,
    InformationSource, InformationPriority
)
from ...core.auth import get_current_user
from ...core.monitoring import track_api_call

router = APIRouter(prefix="/api/v1/information-synthesis", tags=["information-synthesis"])
logger = logging.getLogger(__name__)

# Global engine instance
synthesis_engine = InformationSynthesisEngine()


@router.post("/synthesize", response_model=SynthesizedInformation)
@track_api_call
async def synthesize_information(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Synthesize information for crisis decision-making
    """
    try:
        logger.info(f"Starting information synthesis for crisis {request.crisis_id}")
        
        # Perform synthesis
        synthesis = await synthesis_engine.synthesize_information(request)
        
        # Log synthesis completion in background
        background_tasks.add_task(
            log_synthesis_completion,
            synthesis.id,
            current_user.get("user_id"),
            len(request.information_items)
        )
        
        return synthesis
        
    except Exception as e:
        logger.error(f"Information synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@router.post("/information-items", response_model=dict)
@track_api_call
async def add_information_item(
    item: InformationItem,
    current_user: dict = Depends(get_current_user)
):
    """
    Add a new information item for synthesis
    """
    try:
        item_id = await synthesis_engine.add_information_item(item)
        
        logger.info(f"Added information item {item_id} from source {item.source.value}")
        
        return {
            "item_id": item_id,
            "status": "added",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to add information item: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add item: {str(e)}")


@router.get("/synthesis/{synthesis_id}", response_model=SynthesizedInformation)
@track_api_call
async def get_synthesis(
    synthesis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve a specific synthesis result
    """
    try:
        if synthesis_id not in synthesis_engine.synthesis_cache:
            raise HTTPException(status_code=404, detail="Synthesis not found")
        
        synthesis = synthesis_engine.synthesis_cache[synthesis_id]
        
        logger.info(f"Retrieved synthesis {synthesis_id}")
        
        return synthesis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@router.get("/synthesis/{synthesis_id}/metrics", response_model=SynthesisMetrics)
@track_api_call
async def get_synthesis_metrics(
    synthesis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get performance metrics for a synthesis process
    """
    try:
        metrics = await synthesis_engine.get_synthesis_metrics(synthesis_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Synthesis metrics not found")
        
        logger.info(f"Retrieved metrics for synthesis {synthesis_id}")
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve synthesis metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@router.post("/rapid-synthesis", response_model=SynthesizedInformation)
@track_api_call
async def rapid_information_synthesis(
    crisis_id: str,
    information_items: List[InformationItem],
    urgency_level: InformationPriority = InformationPriority.HIGH,
    filter_criteria: Optional[FilterCriteria] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform rapid information synthesis for immediate crisis response
    """
    try:
        logger.info(f"Starting rapid synthesis for crisis {crisis_id} with {len(information_items)} items")
        
        # Add information items to engine
        item_ids = []
        for item in information_items:
            item_id = await synthesis_engine.add_information_item(item)
            item_ids.append(item_id)
        
        # Create synthesis request
        request = SynthesisRequest(
            crisis_id=crisis_id,
            requester=current_user.get("user_id", "system"),
            information_items=item_ids,
            filter_criteria=filter_criteria,
            urgency_level=urgency_level,
            expected_completion=datetime.now()
        )
        
        # Perform synthesis
        synthesis = await synthesis_engine.synthesize_information(request)
        
        logger.info(f"Completed rapid synthesis {synthesis.id} with confidence {synthesis.confidence_level:.2f}")
        
        return synthesis
        
    except Exception as e:
        logger.error(f"Rapid synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rapid synthesis failed: {str(e)}")


@router.get("/information-items/{item_id}", response_model=InformationItem)
@track_api_call
async def get_information_item(
    item_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve a specific information item
    """
    try:
        if item_id not in synthesis_engine.information_store:
            raise HTTPException(status_code=404, detail="Information item not found")
        
        item = synthesis_engine.information_store[item_id]
        
        logger.info(f"Retrieved information item {item_id}")
        
        return item
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve information item: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Item retrieval failed: {str(e)}")


@router.get("/crisis/{crisis_id}/syntheses", response_model=List[SynthesizedInformation])
@track_api_call
async def get_crisis_syntheses(
    crisis_id: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """
    Get all syntheses for a specific crisis
    """
    try:
        # Filter syntheses by crisis ID
        crisis_syntheses = [
            synthesis for synthesis in synthesis_engine.synthesis_cache.values()
            if synthesis.crisis_id == crisis_id
        ]
        
        # Sort by timestamp (most recent first) and limit
        crisis_syntheses.sort(key=lambda x: x.synthesis_timestamp, reverse=True)
        crisis_syntheses = crisis_syntheses[:limit]
        
        logger.info(f"Retrieved {len(crisis_syntheses)} syntheses for crisis {crisis_id}")
        
        return crisis_syntheses
        
    except Exception as e:
        logger.error(f"Failed to retrieve crisis syntheses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Crisis syntheses retrieval failed: {str(e)}")


@router.post("/validate-synthesis", response_model=dict)
@track_api_call
async def validate_synthesis_quality(
    synthesis_id: str,
    validation_feedback: dict,
    current_user: dict = Depends(get_current_user)
):
    """
    Validate and provide feedback on synthesis quality
    """
    try:
        if synthesis_id not in synthesis_engine.synthesis_cache:
            raise HTTPException(status_code=404, detail="Synthesis not found")
        
        synthesis = synthesis_engine.synthesis_cache[synthesis_id]
        
        # Process validation feedback (simplified implementation)
        quality_score = validation_feedback.get("quality_score", 0.0)
        accuracy_score = validation_feedback.get("accuracy_score", 0.0)
        usefulness_score = validation_feedback.get("usefulness_score", 0.0)
        
        # Calculate overall validation score
        overall_score = (quality_score + accuracy_score + usefulness_score) / 3
        
        # Store validation results (in a real implementation, this would go to a database)
        validation_result = {
            "synthesis_id": synthesis_id,
            "validator": current_user.get("user_id"),
            "validation_timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
            "accuracy_score": accuracy_score,
            "usefulness_score": usefulness_score,
            "overall_score": overall_score,
            "feedback": validation_feedback.get("comments", "")
        }
        
        logger.info(f"Validated synthesis {synthesis_id} with overall score {overall_score:.2f}")
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint for information synthesis service
    """
    try:
        # Check engine status
        engine_status = "healthy"
        
        # Check cache sizes
        information_items_count = len(synthesis_engine.information_store)
        syntheses_count = len(synthesis_engine.synthesis_cache)
        conflicts_count = len(synthesis_engine.conflict_store)
        
        return {
            "status": "healthy",
            "engine_status": engine_status,
            "information_items": information_items_count,
            "syntheses_cached": syntheses_count,
            "conflicts_tracked": conflicts_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def log_synthesis_completion(synthesis_id: str, user_id: str, item_count: int):
    """
    Background task to log synthesis completion
    """
    try:
        logger.info(f"Synthesis {synthesis_id} completed by user {user_id} with {item_count} items")
        
        # In a real implementation, this would write to an audit log or metrics system
        
    except Exception as e:
        logger.error(f"Failed to log synthesis completion: {str(e)}")


# Error handlers
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Value error in information synthesis: {str(exc)}")
    raise HTTPException(status_code=400, detail=str(exc))


@router.exception_handler(TimeoutError)
async def timeout_error_handler(request, exc):
    logger.error(f"Timeout in information synthesis: {str(exc)}")
    raise HTTPException(status_code=408, detail="Information synthesis timed out")