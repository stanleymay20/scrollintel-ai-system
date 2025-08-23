"""
API Routes for Real-Time Orchestration Engine

Provides REST API endpoints for submitting tasks, monitoring orchestration,
and managing real-time agent coordination.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ...core.realtime_orchestration_engine import (
    RealTimeOrchestrationEngine,
    TaskPriority,
    CollaborationMode,
    OrchestrationTask
)
from ...core.agent_registry import AgentRegistry
from ...core.message_bus import get_message_bus

router = APIRouter(prefix="/api/v1/orchestration", tags=["Real-Time Orchestration"])


class TaskSubmissionRequest(BaseModel):
    """Request model for task submission"""
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    required_capabilities: List[str] = Field(..., description="Required agent capabilities")
    preferred_capabilities: Optional[List[str]] = Field(None, description="Preferred capabilities")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload data")
    priority: str = Field("NORMAL", description="Task priority (LOW, NORMAL, HIGH, CRITICAL, EMERGENCY)")
    collaboration_mode: str = Field("SEQUENTIAL", description="Collaboration mode")
    collaboration_agents: Optional[List[str]] = Field(None, description="Specific agents for collaboration")
    timeout_seconds: float = Field(300.0, description="Task timeout in seconds")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    id: str
    name: str
    description: str
    status: str
    priority: str
    assigned_agent_id: Optional[str]
    collaboration_mode: str
    collaboration_agents: List[str]
    created_at: str
    queued_at: Optional[str]
    assigned_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    retry_count: int
    error_message: Optional[str]
    execution_metrics: Dict[str, Any]


class EngineStatsResponse(BaseModel):
    """Response model for engine statistics"""
    engine_status: Dict[str, Any]
    task_statistics: Dict[str, Any]
    distribution_stats: Dict[str, Any]
    load_balancing_stats: Dict[str, Any]
    coordination_stats: Dict[str, Any]
    component_status: Dict[str, str]


# Global orchestration engine instance
_orchestration_engine: Optional[RealTimeOrchestrationEngine] = None


async def get_orchestration_engine() -> RealTimeOrchestrationEngine:
    """Get or create the orchestration engine instance"""
    global _orchestration_engine
    
    if _orchestration_engine is None:
        # Import here to avoid circular imports
        from ...core.agent_registry import AgentRegistry
        from ...core.message_bus import get_message_bus
        
        # Create mock message bus for testing
        message_bus = get_message_bus()
        
        # Create mock agent registry
        agent_registry = AgentRegistry(message_bus)
        
        _orchestration_engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
        await _orchestration_engine.start()
    
    return _orchestration_engine


@router.post("/tasks", response_model=Dict[str, str])
async def submit_task(
    request: TaskSubmissionRequest,
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Submit a new task for orchestration"""
    try:
        # Validate priority
        try:
            priority = TaskPriority[request.priority.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")
        
        # Validate collaboration mode
        try:
            collaboration_mode = CollaborationMode[request.collaboration_mode.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid collaboration mode: {request.collaboration_mode}")
        
        # Submit task
        task_id = await engine.submit_task(
            name=request.name,
            description=request.description,
            required_capabilities=request.required_capabilities,
            payload=request.payload,
            priority=priority,
            collaboration_mode=collaboration_mode,
            collaboration_agents=request.collaboration_agents,
            timeout_seconds=request.timeout_seconds,
            context=request.context
        )
        
        return {"task_id": task_id, "status": "submitted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Get the status of a specific task"""
    try:
        task_status = await engine.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return TaskStatusResponse(**task_status)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Cancel a task"""
    try:
        success = await engine.cancel_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")
        
        return {"task_id": task_id, "status": "cancelled"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.get("/tasks", response_model=List[Dict[str, Any]])
async def get_active_tasks(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Get all currently active tasks"""
    try:
        return engine.get_active_tasks()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active tasks: {str(e)}")


@router.get("/stats", response_model=EngineStatsResponse)
async def get_engine_stats(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Get comprehensive orchestration engine statistics"""
    try:
        stats = engine.get_engine_stats()
        return EngineStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get engine stats: {str(e)}")


@router.post("/engine/start")
async def start_engine(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Start the orchestration engine"""
    try:
        await engine.start()
        return {"status": "started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}")


@router.post("/engine/stop")
async def stop_engine(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Stop the orchestration engine"""
    try:
        await engine.stop()
        return {"status": "stopped"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop engine: {str(e)}")


@router.get("/load-balancer/stats")
async def get_load_balancer_stats(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Get load balancer statistics"""
    try:
        return engine.load_balancer.get_load_balance_stats()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get load balancer stats: {str(e)}")


@router.get("/coordinator/stats")
async def get_coordinator_stats(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Get multi-agent coordinator statistics"""
    try:
        return engine.coordinator.get_coordination_stats()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get coordinator stats: {str(e)}")


@router.post("/test/submit-sample-task")
async def submit_sample_task(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Submit a sample task for testing"""
    try:
        task_id = await engine.submit_task(
            name="Sample Analysis Task",
            description="Analyze sample data and generate insights",
            required_capabilities=["data_analysis", "reporting"],
            payload={"data": "sample_dataset.csv", "analysis_type": "descriptive"},
            priority=TaskPriority.NORMAL,
            collaboration_mode=CollaborationMode.SEQUENTIAL,
            timeout_seconds=180.0,
            context={"source": "api_test", "user_id": "test_user"}
        )
        
        return {"task_id": task_id, "status": "submitted", "type": "sample_task"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit sample task: {str(e)}")


@router.post("/test/submit-collaborative-task")
async def submit_collaborative_task(
    engine: RealTimeOrchestrationEngine = Depends(get_orchestration_engine)
):
    """Submit a collaborative task for testing"""
    try:
        task_id = await engine.submit_task(
            name="Collaborative Analysis",
            description="Multi-agent collaborative data analysis",
            required_capabilities=["data_analysis", "machine_learning", "visualization"],
            payload={"dataset": "large_dataset.csv", "analysis_depth": "comprehensive"},
            priority=TaskPriority.HIGH,
            collaboration_mode=CollaborationMode.PARALLEL,
            collaboration_agents=["agent-1", "agent-2", "agent-3"],
            timeout_seconds=600.0,
            context={"collaboration_test": True, "expected_agents": 3}
        )
        
        return {"task_id": task_id, "status": "submitted", "type": "collaborative_task"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit collaborative task: {str(e)}")