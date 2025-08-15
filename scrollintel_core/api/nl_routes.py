"""
Natural Language Interface API Routes
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from ..agents.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/nl", tags=["Natural Language Interface"])

# Global orchestrator instance (will be injected in main app)
orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Dependency to get orchestrator instance"""
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    return orchestrator


class QueryRequest(BaseModel):
    """Request model for query parsing"""
    query: str
    session_id: Optional[str] = None
    context: Dict[str, Any] = {}


class ConversationalRequest(BaseModel):
    """Request model for conversational interactions"""
    query: str
    session_id: str
    context: Dict[str, Any] = {}


class QueryParseResponse(BaseModel):
    """Response model for query parsing"""
    original_query: str
    intent: str
    confidence: float
    suggested_agent: str
    entities: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    context_needed: List[str]


class ConversationalResponse(BaseModel):
    """Response model for conversational interactions"""
    success: bool
    agent: str
    result: Any = None
    error: Optional[str] = None
    nl_response: str
    session_id: str
    parsing: Dict[str, Any]
    processing_time: float


@router.post("/parse", response_model=QueryParseResponse)
async def parse_query(
    request: QueryRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Parse a user query and extract intent, entities, and parameters"""
    try:
        result = await orch.parse_query(
            request.query, 
            request.session_id, 
            request.context
        )
        
        return QueryParseResponse(**result)
        
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ConversationalResponse)
async def conversational_chat(
    request: ConversationalRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Process a conversational request with full NL capabilities"""
    try:
        result = await orch.process_conversational_request(
            request.query,
            request.session_id,
            request.context
        )
        
        return ConversationalResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing conversational request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions")
async def get_suggestions(
    query: str,
    session_id: Optional[str] = None,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get suggestions for improving a query"""
    try:
        suggestions = await orch.get_nl_suggestions(query, session_id)
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{session_id}")
async def get_conversation_history(
    session_id: str,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get conversation history for a session"""
    try:
        history = await orch.get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{session_id}")
async def clear_conversation(
    session_id: str,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Clear conversation history for a session"""
    try:
        success = await orch.clear_conversation(session_id)
        return {"session_id": session_id, "cleared": success}
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intents")
async def get_available_intents():
    """Get list of available intents and their descriptions"""
    return {
        "intents": {
            "cto": {
                "description": "Technology architecture, scaling, and infrastructure decisions",
                "keywords": ["architecture", "technology", "stack", "scaling", "infrastructure", "system", "design", "cto"]
            },
            "data_scientist": {
                "description": "Data analysis, statistics, and insights generation",
                "keywords": ["analyze", "analysis", "statistics", "correlation", "insights", "explore", "data", "scientist"]
            },
            "ml_engineer": {
                "description": "Machine learning model building and deployment",
                "keywords": ["machine", "learning", "model", "predict", "train", "algorithm", "ml", "engineer"]
            },
            "bi": {
                "description": "Business intelligence, dashboards, and reporting",
                "keywords": ["dashboard", "report", "kpi", "business", "intelligence", "bi", "visualization", "metrics"]
            },
            "ai_engineer": {
                "description": "AI strategy and implementation guidance",
                "keywords": ["ai", "artificial", "intelligence", "strategy", "implementation", "roadmap", "engineer"]
            },
            "qa": {
                "description": "Question answering and data querying",
                "keywords": ["what", "how", "when", "where", "why", "show", "find", "search", "query", "question"]
            },
            "forecast": {
                "description": "Time series forecasting and trend prediction",
                "keywords": ["forecast", "predict", "future", "trend", "projection", "time", "series", "seasonal"]
            }
        }
    }


@router.get("/entities")
async def get_entity_types():
    """Get list of entity types that can be extracted"""
    return {
        "entity_types": {
            "dataset": {
                "description": "Dataset or file names",
                "examples": ["sales_data.csv", "customer_data", "my_dataset"]
            },
            "column": {
                "description": "Column or field names",
                "examples": ["revenue", "customer_id", "date"]
            },
            "metric": {
                "description": "Statistical or performance metrics",
                "examples": ["accuracy", "precision", "mean", "sum", "count"]
            },
            "time_period": {
                "description": "Time periods and durations",
                "examples": ["last 30 days", "monthly", "next quarter"]
            },
            "number": {
                "description": "Numerical values and percentages",
                "examples": ["100", "25%", "3.14"]
            },
            "model_type": {
                "description": "Machine learning model types",
                "examples": ["classification", "regression", "random forest", "neural network"]
            }
        }
    }


@router.get("/health")
async def nl_health_check(
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Health check for NL interface"""
    try:
        # Test basic NL processing
        test_result = await orch.parse_query("test query")
        
        return {
            "status": "healthy",
            "nl_processor": "active",
            "test_parsing": test_result.get("intent") is not None,
            "conversation_memory": "active"
        }
        
    except Exception as e:
        logger.error(f"NL health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def set_orchestrator(orch: AgentOrchestrator):
    """Set the orchestrator instance for the routes"""
    global orchestrator
    orchestrator = orch