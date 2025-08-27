"""
API routes for FederatedEngine
Provides REST endpoints for federated learning operations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from scrollintel.engines.federated_engine import (
    get_federated_engine, 
    FederatedEngine,
    EdgeDeviceType,
    PrivacyLevel
)
from scrollintel.core.config import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/federated", tags=["federated"])

# Pydantic models for request/response
class DeviceConfig(BaseModel):
    device_name: str = Field(..., description="Name of the device")
    device_type: EdgeDeviceType = Field(default=EdgeDeviceType.DESKTOP, description="Type of device")
    ip_address: Optional[str] = Field(None, description="IP address of the device")
    port: Optional[int] = Field(None, description="Port number")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Device capabilities")
    data_size: int = Field(default=1000, description="Size of local dataset")
    compute_power: float = Field(default=1.0, description="Relative compute capability")
    bandwidth: float = Field(default=10.0, description="Bandwidth in MB/s")
    battery_level: Optional[float] = Field(None, description="Battery level (0-100)")
    privacy_level: PrivacyLevel = Field(default=PrivacyLevel.MEDIUM, description="Privacy level")

class TaskConfig(BaseModel):
    task_name: str = Field(..., description="Name of the federated learning task")
    model_architecture: Dict[str, Any] = Field(..., description="Model architecture configuration")
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    privacy_config: Dict[str, Any] = Field(default_factory=dict, description="Privacy configuration")
    participating_devices: List[str] = Field(default_factory=list, description="List of device IDs")
    target_rounds: int = Field(default=10, description="Target number of training rounds")
    convergence_threshold: float = Field(default=0.001, description="Convergence threshold")

class TrainingRequest(BaseModel):
    task_id: str = Field(..., description="Task ID to start training for")
    framework: str = Field(default="pytorch", description="Framework to use (pytorch, pysyft, tff)")

class DeviceResponse(BaseModel):
    device_id: str
    device_name: str
    device_type: str
    status: str
    compute_power: float
    bandwidth: float
    battery_level: Optional[float]
    privacy_level: str
    last_seen: str
    model_version: int
    data_size: int

class TaskResponse(BaseModel):
    task_id: str
    task_name: str
    status: str
    rounds_completed: int
    target_rounds: int
    participating_devices: int
    created_at: str
    convergence_threshold: float

class FederationStatusResponse(BaseModel):
    engine_status: str
    active_task: Optional[str]
    total_tasks: int
    total_devices: int
    online_devices: int
    privacy_budget_remaining: float
    config: Dict[str, Any]

def get_federated_engine_dependency() -> FederatedEngine:
    """Dependency to get federated engine instance"""
    redis_client = get_redis_client()
    return get_federated_engine(redis_client)

@router.post("/devices", response_model=Dict[str, str])
async def add_edge_device(
    device_config: DeviceConfig,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Add a new edge device to the federation"""
    try:
        device_id = await engine.add_edge_device(device_config.dict())
        return {"device_id": device_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to add edge device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/devices/{device_id}")
async def remove_edge_device(
    device_id: str,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Remove an edge device from the federation"""
    try:
        success = await engine.remove_edge_device(device_id)
        if success:
            return {"device_id": device_id, "status": "removed"}
        else:
            raise HTTPException(status_code=404, detail="Device not found")
    except Exception as e:
        logger.error(f"Failed to remove edge device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices", response_model=List[DeviceResponse])
async def list_edge_devices(
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """List all edge devices in the federation"""
    try:
        devices = engine.device_simulator.get_all_devices_status()
        return [DeviceResponse(**device) for device in devices]
    except Exception as e:
        logger.error(f"Failed to list edge devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices/{device_id}", response_model=DeviceResponse)
async def get_device_status(
    device_id: str,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Get status of a specific edge device"""
    try:
        device_status = engine.device_simulator.get_device_status(device_id)
        if device_status:
            return DeviceResponse(**device_status)
        else:
            raise HTTPException(status_code=404, detail="Device not found")
    except Exception as e:
        logger.error(f"Failed to get device status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/devices/{device_id}/simulate-failure")
async def simulate_device_failure(
    device_id: str,
    failure_type: str = "network",
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Simulate a failure on an edge device"""
    try:
        engine.device_simulator.simulate_device_failure(device_id, failure_type)
        return {"device_id": device_id, "failure_type": failure_type, "status": "simulated"}
    except Exception as e:
        logger.error(f"Failed to simulate device failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks", response_model=Dict[str, str])
async def create_federated_task(
    task_config: TaskConfig,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Create a new federated learning task"""
    try:
        task_id = await engine.create_federated_task(task_config.dict())
        return {"task_id": task_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create federated task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/{task_id}/start")
async def start_federated_training(
    task_id: str,
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Start federated training for a task"""
    try:
        # Start training in background
        background_tasks.add_task(
            engine.start_federated_training,
            task_id,
            training_request.framework
        )
        return {"task_id": task_id, "status": "training_started"}
    except Exception as e:
        logger.error(f"Failed to start federated training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/{task_id}/pause")
async def pause_federated_training(
    task_id: str,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Pause federated training for a task"""
    try:
        success = await engine.pause_training(task_id)
        if success:
            return {"task_id": task_id, "status": "paused"}
        else:
            raise HTTPException(status_code=404, detail="Task not found or cannot be paused")
    except Exception as e:
        logger.error(f"Failed to pause federated training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/{task_id}/resume")
async def resume_federated_training(
    task_id: str,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Resume federated training for a task"""
    try:
        success = await engine.resume_training(task_id)
        if success:
            return {"task_id": task_id, "status": "resumed"}
        else:
            raise HTTPException(status_code=404, detail="Task not found or cannot be resumed")
    except Exception as e:
        logger.error(f"Failed to resume federated training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks", response_model=List[TaskResponse])
async def list_federated_tasks(
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """List all federated learning tasks"""
    try:
        tasks = []
        for task_id in engine.tasks.keys():
            task_status = await engine.get_task_status(task_id)
            if task_status:
                tasks.append(TaskResponse(**task_status))
        return tasks
    except Exception as e:
        logger.error(f"Failed to list federated tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Get status of a specific federated learning task"""
    try:
        task_status = await engine.get_task_status(task_id)
        if task_status:
            return TaskResponse(**task_status)
        else:
            raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tasks/{task_id}")
async def delete_federated_task(
    task_id: str,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Delete a federated learning task"""
    try:
        if task_id in engine.tasks:
            del engine.tasks[task_id]
            
            # Remove from Redis if available
            if engine.redis_client:
                await engine.redis_client.hdel("federated_tasks", task_id)
                await engine.redis_client.delete(f"training_rounds:{task_id}")
            
            return {"task_id": task_id, "status": "deleted"}
        else:
            raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        logger.error(f"Failed to delete federated task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=FederationStatusResponse)
async def get_federation_status(
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Get overall federation status"""
    try:
        status = await engine.get_federation_status()
        return FederationStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get federation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_completed_tasks(
    older_than_hours: int = 24,
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Clean up completed tasks older than specified hours"""
    try:
        cleaned_count = await engine.cleanup_completed_tasks(older_than_hours)
        return {"cleaned_tasks": cleaned_count, "status": "completed"}
    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/privacy/budget")
async def get_privacy_budget(
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Get current privacy budget status"""
    try:
        remaining_budget = engine.privacy_engine.get_remaining_budget()
        used_budget = engine.privacy_engine.privacy_budget_used
        total_budget = engine.privacy_engine.epsilon + used_budget
        
        return {
            "total_budget": total_budget,
            "used_budget": used_budget,
            "remaining_budget": remaining_budget,
            "budget_percentage": (remaining_budget / total_budget * 100) if total_budget > 0 else 0
        }
    except Exception as e:
        logger.error(f"Failed to get privacy budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/privacy/reset-budget")
async def reset_privacy_budget(
    engine: FederatedEngine = Depends(get_federated_engine_dependency)
):
    """Reset privacy budget"""
    try:
        engine.privacy_engine.reset_budget()
        return {"status": "privacy_budget_reset"}
    except Exception as e:
        logger.error(f"Failed to reset privacy budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/frameworks/support")
async def get_framework_support():
    """Get information about supported federated learning frameworks"""
    try:
        from scrollintel.engines.federated_engine import HAS_TORCH, HAS_TENSORFLOW, HAS_SYFT, HAS_TFF
        
        return {
            "pytorch": HAS_TORCH,
            "tensorflow": HAS_TENSORFLOW,
            "pysyft": HAS_SYFT,
            "tensorflow_federated": HAS_TFF,
            "custom": True  # Always available
        }
    except Exception as e:
        logger.error(f"Failed to get framework support: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint for federated engine"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "federated_engine"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))