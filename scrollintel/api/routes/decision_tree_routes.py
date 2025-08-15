"""
API Routes for Decision Tree System
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from ...engines.decision_tree_engine import DecisionTreeEngine
from ...models.decision_tree_models import (
    CrisisType, SeverityLevel, DecisionTree, DecisionPath,
    DecisionRecommendation, DecisionTreeMetrics
)


# Pydantic models for API requests/responses
class CrisisContextRequest(BaseModel):
    crisis_id: str
    crisis_type: str
    severity: str
    context: Dict[str, Any]


class DecisionPathRequest(BaseModel):
    crisis_id: str
    tree_id: str


class NavigationRequest(BaseModel):
    path_id: str
    context: Dict[str, Any]


class PathCompletionRequest(BaseModel):
    path_id: str
    outcome: str
    success: bool


class LearningRequest(BaseModel):
    path_id: str
    feedback_score: float
    lessons_learned: List[str]


class DecisionTreeResponse(BaseModel):
    tree_id: str
    name: str
    description: str
    crisis_types: List[str]
    severity_levels: List[str]
    success_rate: float
    usage_count: int
    tags: List[str]


class DecisionRecommendationResponse(BaseModel):
    recommendation_id: str
    crisis_id: str
    tree_id: str
    recommended_action: Dict[str, Any]
    confidence_level: str
    reasoning: str
    alternative_actions: List[Dict[str, Any]]
    estimated_impact: Dict[str, float]
    risk_assessment: Dict[str, Any]


router = APIRouter(prefix="/api/v1/decision-tree", tags=["decision-tree"])
logger = logging.getLogger(__name__)


def get_decision_tree_engine() -> DecisionTreeEngine:
    """Dependency to get decision tree engine instance"""
    return DecisionTreeEngine()


@router.get("/trees", response_model=List[DecisionTreeResponse])
async def get_available_trees(engine: DecisionTreeEngine = Depends(get_decision_tree_engine)):
    """Get all available decision trees"""
    try:
        trees = []
        for tree in engine.decision_trees.values():
            trees.append(DecisionTreeResponse(
                tree_id=tree.tree_id,
                name=tree.name,
                description=tree.description,
                crisis_types=[ct.value for ct in tree.crisis_types],
                severity_levels=[sl.value for sl in tree.severity_levels],
                success_rate=tree.success_rate,
                usage_count=tree.usage_count,
                tags=tree.tags
            ))
        
        return trees
    
    except Exception as e:
        logger.error(f"Error getting decision trees: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve decision trees")


@router.post("/find-applicable")
async def find_applicable_trees(
    request: CrisisContextRequest,
    engine: DecisionTreeEngine = Depends(get_decision_tree_engine)
):
    """Find decision trees applicable to a crisis scenario"""
    try:
        # Convert string values to enums
        crisis_type = CrisisType(request.crisis_type)
        severity = SeverityLevel(request.severity)
        
        applicable_trees = engine.find_applicable_trees(crisis_type, severity)
        
        response_trees = []
        for tree in applicable_trees:
            response_trees.append(DecisionTreeResponse(
                tree_id=tree.tree_id,
                name=tree.name,
                description=tree.description,
                crisis_types=[ct.value for ct in tree.crisis_types],
                severity_levels=[sl.value for sl in tree.severity_levels],
                success_rate=tree.success_rate,
                usage_count=tree.usage_count,
                tags=tree.tags
            ))
        
        return {
            "crisis_id": request.crisis_id,
            "applicable_trees": response_trees,
            "recommendation": response_trees[0] if response_trees else None
        }
    
    except ValueError as e:
        logger.error(f"Invalid crisis type or severity: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid crisis type or severity: {str(e)}")
    except Exception as e:
        logger.error(f"Error finding applicable trees: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find applicable decision trees")


@router.post("/start-path")
async def start_decision_path(
    request: DecisionPathRequest,
    engine: DecisionTreeEngine = Depends(get_decision_tree_engine)
):
    """Start a new decision path through a decision tree"""
    try:
        path = engine.start_decision_path(request.crisis_id, request.tree_id)
        
        return {
            "path_id": path.path_id,
            "tree_id": path.tree_id,
            "crisis_id": path.crisis_id,
            "start_time": path.start_time.isoformat(),
            "status": "active"
        }
    
    except ValueError as e:
        logger.error(f"Invalid tree ID: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting decision path: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start decision path")


@router.post("/navigate", response_model=DecisionRecommendationResponse)
async def navigate_decision_tree(
    request: NavigationRequest,
    engine: DecisionTreeEngine = Depends(get_decision_tree_engine)
):
    """Navigate through a decision tree and get recommendations"""
    try:
        recommendation = engine.navigate_tree(request.path_id, request.context)
        
        return DecisionRecommendationResponse(
            recommendation_id=recommendation.recommendation_id,
            crisis_id=recommendation.crisis_id,
            tree_id=recommendation.tree_id,
            recommended_action={
                "action_id": recommendation.recommended_action.action_id,
                "title": recommendation.recommended_action.title,
                "description": recommendation.recommended_action.description,
                "required_resources": recommendation.recommended_action.required_resources,
                "estimated_duration": recommendation.recommended_action.estimated_duration,
                "success_probability": recommendation.recommended_action.success_probability,
                "risk_level": recommendation.recommended_action.risk_level
            },
            confidence_level=recommendation.confidence_level.value,
            reasoning=recommendation.reasoning,
            alternative_actions=[
                {
                    "action_id": action.action_id,
                    "title": action.title,
                    "description": action.description,
                    "success_probability": action.success_probability,
                    "risk_level": action.risk_level
                }
                for action in recommendation.alternative_actions
            ],
            estimated_impact=recommendation.estimated_impact,
            risk_assessment=recommendation.risk_assessment
        )
    
    except ValueError as e:
        logger.error(f"Invalid path ID: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error navigating decision tree: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to navigate decision tree")


@router.post("/complete-path")
async def complete_decision_path(
    request: PathCompletionRequest,
    engine: DecisionTreeEngine = Depends(get_decision_tree_engine)
):
    """Complete a decision path and record the outcome"""
    try:
        engine.complete_path(request.path_id, request.outcome, request.success)
        
        return {
            "path_id": request.path_id,
            "outcome": request.outcome,
            "success": request.success,
            "status": "completed"
        }
    
    except ValueError as e:
        logger.error(f"Invalid path ID: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error completing decision path: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to complete decision path")


@router.post("/learn")
async def record_learning(
    request: LearningRequest,
    engine: DecisionTreeEngine = Depends(get_decision_tree_engine)
):
    """Record learning data from a completed decision path"""
    try:
        engine.learn_from_outcome(request.path_id, request.feedback_score, request.lessons_learned)
        
        return {
            "path_id": request.path_id,
            "feedback_score": request.feedback_score,
            "lessons_count": len(request.lessons_learned),
            "status": "recorded"
        }
    
    except Exception as e:
        logger.error(f"Error recording learning data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record learning data")


@router.get("/metrics/{tree_id}")
async def get_tree_metrics(
    tree_id: str,
    engine: DecisionTreeEngine = Depends(get_decision_tree_engine)
):
    """Get performance metrics for a decision tree"""
    try:
        metrics = engine.get_tree_metrics(tree_id)
        
        return {
            "tree_id": metrics.tree_id,
            "total_usage": metrics.total_usage,
            "success_rate": metrics.success_rate,
            "average_decision_time": metrics.average_decision_time,
            "confidence_distribution": metrics.confidence_distribution,
            "most_common_paths": metrics.most_common_paths,
            "failure_points": metrics.failure_points,
            "optimization_suggestions": metrics.optimization_suggestions,
            "last_updated": metrics.last_updated.isoformat()
        }
    
    except ValueError as e:
        logger.error(f"Invalid tree ID: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting tree metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get tree metrics")


@router.post("/optimize/{tree_id}")
async def optimize_decision_tree(
    tree_id: str,
    engine: DecisionTreeEngine = Depends(get_decision_tree_engine)
):
    """Optimize a decision tree based on learning data"""
    try:
        suggestions = engine.optimize_tree(tree_id)
        
        return {
            "tree_id": tree_id,
            "optimization_suggestions": suggestions,
            "status": "optimized"
        }
    
    except Exception as e:
        logger.error(f"Error optimizing decision tree: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to optimize decision tree")


@router.get("/active-paths")
async def get_active_paths(engine: DecisionTreeEngine = Depends(get_decision_tree_engine)):
    """Get all currently active decision paths"""
    try:
        active_paths = []
        for path in engine.active_paths.values():
            active_paths.append({
                "path_id": path.path_id,
                "tree_id": path.tree_id,
                "crisis_id": path.crisis_id,
                "start_time": path.start_time.isoformat(),
                "nodes_traversed": len(path.nodes_traversed),
                "decisions_made": len(path.decisions_made)
            })
        
        return {
            "active_paths": active_paths,
            "total_count": len(active_paths)
        }
    
    except Exception as e:
        logger.error(f"Error getting active paths: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get active paths")