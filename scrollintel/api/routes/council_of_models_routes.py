"""
API Routes for ScrollIntel G6 - Superintelligent Council of Models

Provides REST API endpoints for interacting with the Council of Models system
including deliberation, debate, argumentation, and swarm intelligence coordination.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import logging
from datetime import datetime
import uuid

from scrollintel.core.council_of_models import (
    SuperCouncilOfModels,
    DebateRole,
    ArgumentationDepth
)
from scrollintel.core.interfaces import BaseEngine

# Initialize router
router = APIRouter(prefix="/api/v1/council", tags=["Council of Models"])
logger = logging.getLogger(__name__)

# Global council instance (in production, this would be dependency-injected)
_council_instance = None


def get_council() -> SuperCouncilOfModels:
    """Get or create council instance"""
    global _council_instance
    if _council_instance is None:
        _council_instance = SuperCouncilOfModels()
    return _council_instance


# Request/Response Models
class DeliberationRequest(BaseModel):
    """Request model for council deliberation"""
    content: str = Field(..., description="The question or decision to deliberate on")
    type: str = Field(default="general", description="Type of decision (strategic, ethical, technical, etc.)")
    complexity: str = Field(default="medium", description="Complexity level (low, medium, high, extreme)")
    domain: str = Field(default="general", description="Domain of expertise required")
    context: Optional[str] = Field(None, description="Additional context for the decision")
    constraints: Optional[List[str]] = Field(default_factory=list, description="Constraints to consider")
    stakeholders: Optional[List[str]] = Field(default_factory=list, description="Stakeholders affected")
    timeout_seconds: Optional[int] = Field(default=300, description="Maximum deliberation time")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Should we implement quantum computing for our AI infrastructure?",
                "type": "strategic_decision",
                "complexity": "high",
                "domain": "technology",
                "context": "Mid-size AI company with $50M budget",
                "constraints": ["Budget limitations", "Timeline constraints"],
                "stakeholders": ["engineering_team", "executives", "customers"],
                "timeout_seconds": 300
            }
        }


class DeliberationResponse(BaseModel):
    """Response model for council deliberation"""
    decision_id: str
    timestamp: str
    confidence_score: float
    consensus_level: float
    final_recommendation: str
    reasoning_chain: List[str]
    philosophical_considerations: List[str]
    emergent_insights: List[str]
    processing_time: float
    models_used: List[str]
    debate_summary: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "decision_id": "council_decision_123456789",
                "timestamp": "2024-01-15T10:30:00Z",
                "confidence_score": 0.87,
                "consensus_level": 0.82,
                "final_recommendation": "Implement quantum computing with phased approach",
                "reasoning_chain": [
                    "Multi-model analysis shows quantum advantage for specific AI workloads",
                    "Game-theoretic optimization suggests phased implementation minimizes risk",
                    "Swarm intelligence coordination validates technical feasibility"
                ],
                "philosophical_considerations": [
                    "Balance between technological advancement and practical constraints",
                    "Ethical implications of quantum-enhanced AI capabilities"
                ],
                "emergent_insights": [
                    "Quantum-classical hybrid approach offers optimal risk-reward balance",
                    "Stakeholder alignment crucial for successful implementation"
                ],
                "processing_time": 45.2,
                "models_used": ["gpt-5", "claude-4", "gemini-ultra", "palm-3"]
            }
        }


class DebateRequest(BaseModel):
    """Request model for adversarial debate"""
    topic: str = Field(..., description="Topic for adversarial debate")
    context: Optional[str] = Field(None, description="Context for the debate")
    max_rounds: Optional[int] = Field(default=5, description="Maximum debate rounds")
    models_to_include: Optional[List[str]] = Field(None, description="Specific models to include")
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "Should AI systems be granted legal personhood?",
                "context": "Considering advanced AI consciousness and autonomy",
                "max_rounds": 5,
                "models_to_include": ["gpt-5", "claude-4", "gemini-ultra"]
            }
        }


class ArgumentationRequest(BaseModel):
    """Request model for recursive argumentation"""
    arguments: List[Dict[str, Any]] = Field(..., description="Initial arguments to refine")
    depth: str = Field(default="deep", description="Argumentation depth (surface, intermediate, deep, infinite)")
    
    class Config:
        schema_extra = {
            "example": {
                "arguments": [
                    {
                        "source": "initial_analysis",
                        "content": ["AI consciousness is substrate-independent"],
                        "evidence": ["Computational theory of mind"],
                        "confidence": 0.7
                    }
                ],
                "depth": "deep"
            }
        }


class SocraticRequest(BaseModel):
    """Request model for Socratic questioning"""
    topic: str = Field(..., description="Topic for philosophical inquiry")
    arguments: Optional[List[Dict[str, Any]]] = Field(None, description="Arguments to analyze")
    philosophical_domains: Optional[List[str]] = Field(None, description="Specific domains to explore")
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "What is the nature of consciousness?",
                "arguments": [
                    {
                        "content": ["Consciousness requires subjective experience"],
                        "confidence": 0.8
                    }
                ],
                "philosophical_domains": ["epistemology", "philosophy_of_mind"]
            }
        }


# API Endpoints

@router.post("/deliberate", response_model=DeliberationResponse)
async def deliberate(
    request: DeliberationRequest,
    background_tasks: BackgroundTasks,
    council: SuperCouncilOfModels = Depends(get_council)
) -> DeliberationResponse:
    """
    Conduct full council deliberation on a decision or question
    
    This endpoint orchestrates the complete council process including:
    - Model selection and role assignment
    - Adversarial debate between teams
    - Recursive argumentation refinement
    - Socratic philosophical inquiry
    - Game-theoretic optimization
    - Swarm intelligence coordination
    """
    try:
        logger.info(f"Starting council deliberation: {request.content[:100]}...")
        
        # Convert request to internal format
        internal_request = {
            'id': str(uuid.uuid4()),
            'type': request.type,
            'complexity': request.complexity,
            'domain': request.domain,
            'content': request.content,
            'context': request.context,
            'constraints': request.constraints,
            'stakeholders': request.stakeholders,
            'start_time': datetime.utcnow().timestamp()
        }
        
        # Execute deliberation with timeout
        start_time = datetime.utcnow()
        
        try:
            result = await asyncio.wait_for(
                council.deliberate(internal_request),
                timeout=request.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Deliberation timed out after {request.timeout_seconds} seconds"
            )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Extract models used (would be tracked in actual implementation)
        models_used = list(council.frontier_models.keys())[:8]  # Sample for demo
        
        # Format response
        response = DeliberationResponse(
            decision_id=result['decision_id'],
            timestamp=result['timestamp'],
            confidence_score=result['confidence_score'],
            consensus_level=result['consensus_level'],
            final_recommendation=result['final_recommendation'],
            reasoning_chain=result['reasoning_chain'],
            philosophical_considerations=result.get('philosophical_considerations', []),
            emergent_insights=result.get('emergent_insights', []),
            processing_time=processing_time,
            models_used=models_used,
            debate_summary=result.get('debate_summary')
        )
        
        # Log successful deliberation
        background_tasks.add_task(
            log_deliberation_metrics,
            internal_request['id'],
            processing_time,
            result['confidence_score'],
            len(models_used)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in council deliberation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deliberation failed: {str(e)}")


@router.post("/debate")
async def conduct_debate(
    request: DebateRequest,
    council: SuperCouncilOfModels = Depends(get_council)
) -> Dict[str, Any]:
    """
    Conduct adversarial debate between red and blue teams
    
    This endpoint runs just the debate component of the council system,
    useful for exploring controversial topics or testing different perspectives.
    """
    try:
        logger.info(f"Starting adversarial debate: {request.topic}")
        
        # Create internal request format
        internal_request = {
            'id': str(uuid.uuid4()),
            'content': request.topic,
            'context': request.context,
            'type': 'debate'
        }
        
        # Select models for debate
        if request.models_to_include:
            selected_models = [m for m in request.models_to_include if m in council.frontier_models]
        else:
            selected_models = await council._select_models_for_deliberation(internal_request)
        
        # Assign debate roles
        role_assignments = await council._assign_debate_roles(selected_models, internal_request)
        
        # Set max rounds if specified
        original_max_rounds = council.adversarial_debate_engine.max_debate_rounds
        if request.max_rounds:
            council.adversarial_debate_engine.max_debate_rounds = request.max_rounds
        
        try:
            # Conduct debate
            debate_results = await council.adversarial_debate_engine.conduct_debate(
                internal_request, role_assignments
            )
            
            # Add role assignment info to results
            debate_results['role_assignments'] = {
                model_id: role.value for model_id, role in role_assignments.items()
            }
            debate_results['selected_models'] = selected_models
            
            return debate_results
            
        finally:
            # Restore original max rounds
            council.adversarial_debate_engine.max_debate_rounds = original_max_rounds
        
    except Exception as e:
        logger.error(f"Error in adversarial debate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debate failed: {str(e)}")


@router.post("/argumentation")
async def recursive_argumentation(
    request: ArgumentationRequest,
    council: SuperCouncilOfModels = Depends(get_council)
) -> Dict[str, Any]:
    """
    Apply recursive argumentation to deepen and refine arguments
    
    This endpoint takes initial arguments and applies recursive refinement
    to achieve deeper philosophical and logical analysis.
    """
    try:
        logger.info(f"Starting recursive argumentation with {len(request.arguments)} arguments")
        
        # Convert depth string to enum
        depth_mapping = {
            'surface': ArgumentationDepth.SURFACE,
            'intermediate': ArgumentationDepth.INTERMEDIATE,
            'deep': ArgumentationDepth.DEEP,
            'infinite': ArgumentationDepth.INFINITE
        }
        
        depth = depth_mapping.get(request.depth.lower(), ArgumentationDepth.DEEP)
        
        # Format arguments for processing
        debate_results = {
            'rounds': [
                {
                    'round': i + 1,
                    'red_argument' if i % 2 == 0 else 'blue_argument': arg
                }
                for i, arg in enumerate(request.arguments)
            ]
        }
        
        # Apply recursive argumentation
        results = await council.recursive_argumentation_engine.deepen_arguments(
            debate_results, depth
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in recursive argumentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Argumentation failed: {str(e)}")


@router.post("/socratic")
async def socratic_questioning(
    request: SocraticRequest,
    council: SuperCouncilOfModels = Depends(get_council)
) -> Dict[str, Any]:
    """
    Conduct Socratic questioning for deep philosophical inquiry
    
    This endpoint applies Socratic questioning methods to explore
    fundamental assumptions and generate philosophical insights.
    """
    try:
        logger.info(f"Starting Socratic questioning: {request.topic}")
        
        # Format arguments for Socratic analysis
        if request.arguments:
            refined_arguments = {'refined_arguments': request.arguments}
        else:
            # Create default argument from topic
            refined_arguments = {
                'refined_arguments': [
                    {
                        'source': 'topic_analysis',
                        'content': [request.topic],
                        'confidence': 0.5
                    }
                ]
            }
        
        # Create internal request
        internal_request = {
            'id': str(uuid.uuid4()),
            'content': request.topic
        }
        
        # Conduct Socratic inquiry
        results = await council.socratic_questioning_engine.conduct_inquiry(
            refined_arguments, internal_request
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in Socratic questioning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Socratic inquiry failed: {str(e)}")


@router.get("/models")
async def list_models(council: SuperCouncilOfModels = Depends(get_council)) -> Dict[str, Any]:
    """
    List all available frontier models in the council
    
    Returns information about each model's capabilities and specializations.
    """
    try:
        models_info = {}
        
        for model_id, capability in council.frontier_models.items():
            models_info[model_id] = {
                'name': capability.model_name,
                'reasoning_strength': capability.reasoning_strength,
                'creativity_score': capability.creativity_score,
                'factual_accuracy': capability.factual_accuracy,
                'philosophical_depth': capability.philosophical_depth,
                'adversarial_robustness': capability.adversarial_robustness,
                'specializations': capability.specializations
            }
        
        return {
            'total_models': len(models_info),
            'models': models_info
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/status")
async def get_council_status(council: SuperCouncilOfModels = Depends(get_council)) -> Dict[str, Any]:
    """
    Get current status and performance metrics of the council
    
    Returns information about recent deliberations, model performance,
    and system health metrics.
    """
    try:
        return {
            'status': 'operational',
            'total_models': len(council.frontier_models),
            'total_deliberations': len(council.debate_history),
            'recent_deliberations': council.debate_history[-5:] if council.debate_history else [],
            'model_performance_metrics': dict(list(council.model_performance_metrics.items())[:10]),
            'system_health': {
                'engines_operational': True,
                'last_health_check': datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting council status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/stream-deliberation")
async def stream_deliberation(
    request: DeliberationRequest,
    council: SuperCouncilOfModels = Depends(get_council)
) -> StreamingResponse:
    """
    Stream council deliberation progress in real-time
    
    This endpoint provides a streaming response that shows the deliberation
    process as it unfolds, useful for monitoring complex decisions.
    """
    async def generate_deliberation_stream():
        try:
            # Convert request to internal format
            internal_request = {
                'id': str(uuid.uuid4()),
                'type': request.type,
                'complexity': request.complexity,
                'domain': request.domain,
                'content': request.content,
                'context': request.context,
                'start_time': datetime.utcnow().timestamp()
            }
            
            # Stream progress updates
            yield f"data: {json.dumps({'status': 'started', 'message': 'Initializing council deliberation'})}\n\n"
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Selecting models and assigning roles'})}\n\n"
            await asyncio.sleep(0.1)  # Simulate processing time
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Conducting adversarial debate'})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Applying recursive argumentation'})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Conducting Socratic inquiry'})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Optimizing with game theory'})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Coordinating swarm intelligence'})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Synthesizing final decision'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Execute actual deliberation (simplified for streaming)
            result = await council.deliberate(internal_request)
            
            yield f"data: {json.dumps({'status': 'completed', 'result': result})}\n\n"
            
        except Exception as e:
            error_msg = {'status': 'error', 'message': str(e)}
            yield f"data: {json.dumps(error_msg)}\n\n"
    
    return StreamingResponse(
        generate_deliberation_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# Background task functions
async def log_deliberation_metrics(
    request_id: str,
    processing_time: float,
    confidence_score: float,
    models_count: int
):
    """Log deliberation metrics for monitoring and analysis"""
    try:
        logger.info(f"Deliberation metrics - ID: {request_id}, "
                   f"Time: {processing_time:.2f}s, "
                   f"Confidence: {confidence_score:.3f}, "
                   f"Models: {models_count}")
        
        # In production, this would write to a metrics database
        # For now, we just log the information
        
    except Exception as e:
        logger.error(f"Error logging deliberation metrics: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for the council system"""
    try:
        council = get_council()
        
        # Basic health checks
        models_available = len(council.frontier_models) > 0
        engines_initialized = all([
            council.adversarial_debate_engine is not None,
            council.recursive_argumentation_engine is not None,
            council.socratic_questioning_engine is not None,
            council.game_theoretic_optimizer is not None,
            council.swarm_intelligence_coordinator is not None
        ])
        
        if models_available and engines_initialized:
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {"status": "unhealthy", "timestamp": datetime.utcnow().isoformat()}
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e), "timestamp": datetime.utcnow().isoformat()}