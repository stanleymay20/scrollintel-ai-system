"""Health check routes."""

from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import os

from ..models.responses import HealthResponse
from ...core.config import Config

router = APIRouter()
config = Config()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    # Check system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check component health
    components = {
        "database": "healthy",  # Would check actual DB connection
        "storage": "healthy",   # Would check storage availability
        "processing": "healthy" # Would check processing queue
    }
    
    # Determine overall status
    status = "healthy"
    if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
        status = "degraded"
    
    # Check if any components are unhealthy
    if any(comp_status != "healthy" for comp_status in components.values()):
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        timestamp=datetime.utcnow(),
        components={
            **components,
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "disk_usage": f"{disk.percent}%"
        }
    )


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics."""
    # System information
    system_info = {
        "platform": os.name,
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "disk_total": psutil.disk_usage('/').total,
        "uptime": datetime.utcnow().isoformat()
    }
    
    # Process information
    process = psutil.Process()
    process_info = {
        "pid": process.pid,
        "memory_usage": process.memory_info().rss,
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads(),
        "open_files": len(process.open_files())
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "system": system_info,
        "process": process_info,
        "config": {
            "max_workers": config.processing.max_workers,
            "batch_size": config.processing.batch_size,
            "memory_limit_gb": config.processing.memory_limit_gb
        }
    }


@router.get("/health/readiness")
async def readiness_check():
    """Readiness check for load balancers."""
    # Check if service is ready to accept requests
    # This would typically check database connectivity, etc.
    
    try:
        # Simulate database check
        # await check_database_connection()
        
        return {"status": "ready", "timestamp": datetime.utcnow()}
    except Exception as e:
        return {"status": "not_ready", "error": str(e), "timestamp": datetime.utcnow()}


@router.get("/health/liveness")
async def liveness_check():
    """Liveness check for container orchestration."""
    # Simple check to verify the service is alive
    return {"status": "alive", "timestamp": datetime.utcnow()}