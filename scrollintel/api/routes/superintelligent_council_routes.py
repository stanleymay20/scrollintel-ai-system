"""
API routes for the Superintelligent Council of Models
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from scrollintel.core.super_council_of_models import SuperCouncilOfModels, DebateRole, ArgumentationDepth

# Initialize router and logger
router = APIRouter(prefix="/api/v1/superintelligent-council", tags=["Superintelligent Council"])
logger = logging.getLogger(__name__)

# Global council instance
council_instance = None


def get_council() -> SuperCouncilOfModels:
    """Get or create the global council instance"""
    global council_instance
    if council_instance is None:
        council_instance = SuperCouncilOfModels()
    return council_instance


# Pydantic models for API
class DeliberationRequest(BaseModel):
    """Request model for council deliberation"""
    id: Optional[str] = Field(None, description="Unique request identifier")
    type: str = Field(..., description="Type of decision/question")
    complexity: str = Field("medium", description="Complexity level: low, medium, high")
    domain: str = Field("general", description="Domain of expertise required")
    content: str = Field(..., description="The question or decision to deliberate")
    context: Optional[str] = Field(None, description="Additional context")
    priority: str = Field("normal", description="Priority level: low, normal, high, urgent")
    timeout_seconds: Optional[int] = Field(300, description="Maximum deliberation time")


class DeliberationResponse(BaseModel):
    """Response model for council deliberation"""
    decision_id: str
    timestamp: str
    confidence_score: float
    reasoning_chain: List[str]
    supporting_evidence: List[str]
    potential_risks: List[str]
    alternative_approaches: List[str]
    philosophical_considerations: List[str]
    emergent_insights: List[str]
    consensus_level: float
    dissenting_opinions: List[str]
    final_recommendation: str
    processing_time_seconds: Optional[float] = None
    models_used: Optional[List[str]] = None
    debate_rounds: Optional[int] = None
    is_fallback: Optional[bool] = False


class CouncilStatus(BaseModel):
    """Status model for the council"""
    status: str
    total_models: int
    deliberations_completed: int
    engines_status: Dict[str, str]
    performance_metrics: Optional[Dict[str, Any]] = None


class ModelCapabilityInfo(BaseModel):
    """Model capability information"""
    model_name: str
    reasoning_strength: float
    creativity_score: float
    factual_accuracy: float
    philosophical_depth: float
    adversarial_robustness: float
    specializations: List[str]


class DebateHistoryEntry(BaseModel):
    """Debate history entry"""
    request_id: str
    timestamp: str
    models_used: List[str]
    decision_quality: float
    consensus_level: float
    process_duration: float
    debate_rounds: int
    socratic_questions: int


@router.post("/deliberate", response_model=DeliberationResponse)
async def deliberate(request: DeliberationRequest, background_tasks: BackgroundTasks):
    """
    Submit a request for council deliberation
    
    This endpoint orchestrates the full superintelligent council process:
    - Model selection and role assignment
    - Adversarial debate with red-team vs blue-team dynamics
    - Recursive argumentation with infinite depth reasoning
    - Socratic questioning for philosophical inquiry
    - Game-theoretic optimization with Nash equilibrium
    - Swarm intelligence coordination for emergent behavior
    """
    try:
        council = get_council()
        
        # Prepare request for council
        council_request = {
            'id': request.id or f"req_{int(datetime.utcnow().timestamp())}",
            'type': request.type,
            'complexity': request.complexity,
            'domain': request.domain,
            'content': request.content,
            'context': request.context,
            'priority': request.priority,
            'start_time': datetime.utcnow().timestamp()
        }
        
        logger.info(f"Starting council deliberation for request: {council_request['id']}")
        
        # Execute deliberation
        start_time = datetime.utcnow()
        result = await council.deliberate(council_request)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare response
        response = DeliberationResponse(
            decision_id=result['decision_id'],
            timestamp=result['timestamp'],
            confidence_score=result['confidence_score'],
            reasoning_chain=result['reasoning_chain'],
            supporting_evidence=result.get('supporting_evidence', []),
            potential_risks=result.get('potential_risks', []),
            alternative_approaches=result.get('alternative_approaches', []),
            philosophical_considerations=result.get('philosophical_considerations', []),
            emergent_insights=result.get('emergent_insights', []),
            consensus_level=result['consensus_level'],
            dissenting_opinions=result.get('dissenting_opinions', []),
            final_recommendation=result['final_recommendation'],
            processing_time_seconds=processing_time,
            is_fallback=result.get('is_fallback', False)
        )
        
        logger.info(f"Council deliberation completed in {processing_time:.2f}s with confidence {result['confidence_score']:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in council deliberation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deliberation failed: {str(e)}")


@router.get("/status", response_model=CouncilStatus)
async def get_council_status():
    """
    Get the current status of the superintelligent council
    """
    try:
        council = get_council()
        status = council.get_status()
        
        # Add performance metrics if available
        performance_metrics = {}
        if council.model_performance_metrics:
            performance_metrics = {
                'total_models_used': len(council.model_performance_metrics),
                'average_deliberations_per_model': sum(
                    metrics['deliberations_participated'] 
                    for metrics in council.model_performance_metrics.values()
                ) / len(council.model_performance_metrics) if council.model_performance_metrics else 0
            }
        
        return CouncilStatus(
            status=status['status'],
            total_models=status['total_models'],
            deliberations_completed=status['deliberations_completed'],
            engines_status=status['engines_status'],
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting council status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/models", response_model=List[ModelCapabilityInfo])
async def get_frontier_models():
    """
    Get information about all frontier models in the council
    """
    try:
        council = get_council()
        
        models_info = []
        for model_id, capability in council.frontier_models.items():
            models_info.append(ModelCapabilityInfo(
                model_name=capability.model_name,
                reasoning_strength=capability.reasoning_strength,
                creativity_score=capability.creativity_score,
                factual_accuracy=capability.factual_accuracy,
                philosophical_depth=capability.philosophical_depth,
                adversarial_robustness=capability.adversarial_robustness,
                specializations=capability.specializations
            ))
        
        return models_info
        
    except Exception as e:
        logger.error(f"Error getting frontier models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/history", response_model=List[DebateHistoryEntry])
async def get_debate_history(limit: int = 50):
    """
    Get the debate history from the council
    """
    try:
        council = get_council()
        
        # Get recent history (limited)
        recent_history = council.debate_history[-limit:] if council.debate_history else []
        
        history_entries = []
        for entry in recent_history:
            history_entries.append(DebateHistoryEntry(
                request_id=entry['request_id'],
                timestamp=entry['timestamp'],
                models_used=entry['models_used'],
                decision_quality=entry['decision_quality'],
                consensus_level=entry['consensus_level'],
                process_duration=entry['process_duration'],
                debate_rounds=entry['debate_rounds'],
                socratic_questions=entry['socratic_questions']
            ))
        
        return history_entries
        
    except Exception as e:
        logger.error(f"Error getting debate history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/models/{model_id}/performance")
async def get_model_performance(model_id: str):
    """
    Get performance metrics for a specific model
    """
    try:
        council = get_council()
        
        if model_id not in council.model_performance_metrics:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found in performance metrics")
        
        metrics = council.model_performance_metrics[model_id]
        
        return {
            'model_id': model_id,
            'deliberations_participated': metrics['deliberations_participated'],
            'average_contribution_score': metrics['average_contribution_score'],
            'consensus_rate': metrics['consensus_rate'],
            'accuracy_rate': metrics['accuracy_rate']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.post("/test-debate")
async def test_adversarial_debate(
    question: str,
    complexity: str = "medium",
    max_rounds: int = 5
):
    """
    Test the adversarial debate system with a specific question
    """
    try:
        council = get_council()
        
        # Create a test request
        test_request = {
            'id': f"test_debate_{int(datetime.utcnow().timestamp())}",
            'type': 'test_debate',
            'complexity': complexity,
            'domain': 'general',
            'content': question,
            'start_time': datetime.utcnow().timestamp()
        }
        
        # Select models and assign roles
        selected_models = await council._select_models_for_deliberation(test_request)
        role_assignments = await council._assign_debate_roles(selected_models, test_request)
        
        # Set max rounds for test
        original_max_rounds = council.adversarial_debate_engine.max_debate_rounds
        council.adversarial_debate_engine.max_debate_rounds = max_rounds
        
        try:
            # Conduct debate
            debate_results = await council.adversarial_debate_engine.conduct_debate(
                test_request, role_assignments
            )
            
            return {
                'question': question,
                'selected_models': selected_models,
                'role_assignments': {model: role.value for model, role in role_assignments.items()},
                'debate_results': debate_results,
                'test_completed': True
            }
            
        finally:
            # Restore original max rounds
            council.adversarial_debate_engine.max_debate_rounds = original_max_rounds
        
    except Exception as e:
        logger.error(f"Error in test debate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test debate failed: {str(e)}")


@router.post("/test-socratic")
async def test_socratic_questioning(
    topic: str,
    philosophical_domain: str = "epistemology"
):
    """
    Test the Socratic questioning engine with a specific topic
    """
    try:
        council = get_council()
        
        # Create mock refined arguments for testing
        refined_arguments = {
            'reasoning_chains': [f"Initial reasoning about {topic}"],
            'philosophical_insights': [f"Basic insight about {topic}"]
        }
        
        test_request = {
            'content': topic,
            'domain': philosophical_domain
        }
        
        # Conduct Socratic inquiry
        socratic_results = await council.socratic_questioning_engine.conduct_inquiry(
            refined_arguments, test_request
        )
        
        return {
            'topic': topic,
            'philosophical_domain': philosophical_domain,
            'socratic_results': socratic_results,
            'test_completed': True
        }
        
    except Exception as e:
        logger.error(f"Error in test Socratic questioning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test Socratic questioning failed: {str(e)}")


@router.post("/test-swarm")
async def test_swarm_intelligence(
    problem: str,
    swarm_size: int = 20
):
    """
    Test the swarm intelligence coordination with a specific problem
    """
    try:
        council = get_council()
        
        # Create mock optimal strategy for testing
        optimal_strategy = {
            'optimal_strategy': 'collaborative_problem_solving',
            'expected_utility': 0.8
        }
        
        selected_models = ['gpt-5', 'claude-4', 'gemini-ultra', 'palm-3']
        
        # Set swarm size for test
        original_swarm_size = council.swarm_intelligence_coordinator.swarm_size
        council.swarm_intelligence_coordinator.swarm_size = swarm_size
        
        try:
            # Coordinate swarm intelligence
            swarm_results = await council.swarm_intelligence_coordinator.coordinate_emergence(
                optimal_strategy, selected_models
            )
            
            return {
                'problem': problem,
                'swarm_size': swarm_size,
                'selected_models': selected_models,
                'swarm_results': swarm_results,
                'test_completed': True
            }
            
        finally:
            # Restore original swarm size
            council.swarm_intelligence_coordinator.swarm_size = original_swarm_size
        
    except Exception as e:
        logger.error(f"Error in test swarm intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test swarm intelligence failed: {str(e)}")


@router.delete("/history")
async def clear_debate_history():
    """
    Clear the debate history (admin function)
    """
    try:
        council = get_council()
        council.debate_history.clear()
        council.model_performance_metrics.clear()
        
        return {
            'message': 'Debate history and performance metrics cleared',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing debate history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check for the superintelligent council
    """
    try:
        council = get_council()
        status = council.get_status()
        
        return {
            'status': 'healthy',
            'council_status': status['status'],
            'total_models': status['total_models'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }