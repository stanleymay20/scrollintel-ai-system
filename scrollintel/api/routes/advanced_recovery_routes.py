"""
API routes for Advanced Recovery and Self-Healing System.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...core.advanced_recovery_system import advanced_recovery_system

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advanced-recovery", tags=["advanced-recovery"])


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    try:
        return advanced_recovery_system.get_system_status()
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/{node_name}")
async def check_node_health(node_name: str) -> Dict[str, Any]:
    """Check health of a specific node."""
    try:
        health_score = await advanced_recovery_system.perform_health_check(node_name)
        return {
            "node_name": node_name,
            "health_score": health_score,
            "status": advanced_recovery_system._get_health_status(health_score),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed for {node_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/repair/{node_name}")
async def trigger_autonomous_repair(node_name: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger autonomous repair for a specific node."""
    try:
        # Run repair in background
        background_tasks.add_task(advanced_recovery_system.autonomous_system_repair, node_name)
        
        return {
            "message": f"Autonomous repair initiated for {node_name}",
            "node_name": node_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to trigger repair for {node_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dependencies")
async def get_dependency_status() -> Dict[str, Any]:
    """Get status of all dependencies."""
    try:
        return await advanced_recovery_system.intelligent_dependency_management()
    except Exception as e:
        logger.error(f"Failed to get dependency status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_status() -> Dict[str, Any]:
    """Get performance optimization status."""
    try:
        return await advanced_recovery_system.self_optimizing_performance_tuning()
    except Exception as e:
        logger.error(f"Failed to get performance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def trigger_performance_optimization(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger performance optimization."""
    try:
        background_tasks.add_task(advanced_recovery_system.self_optimizing_performance_tuning)
        
        return {
            "message": "Performance optimization initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to trigger performance optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/maintenance")
async def get_maintenance_status() -> Dict[str, Any]:
    """Get predictive maintenance status."""
    try:
        return await advanced_recovery_system.predictive_maintenance()
    except Exception as e:
        logger.error(f"Failed to get maintenance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/schedule")
async def schedule_maintenance(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Schedule predictive maintenance tasks."""
    try:
        background_tasks.add_task(advanced_recovery_system.predictive_maintenance)
        
        return {
            "message": "Predictive maintenance scheduled",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to schedule maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/auto-recovery")
async def toggle_auto_recovery(enabled: bool) -> Dict[str, Any]:
    """Enable or disable auto-recovery."""
    try:
        advanced_recovery_system.auto_recovery_enabled = enabled
        
        return {
            "message": f"Auto-recovery {'enabled' if enabled else 'disabled'}",
            "auto_recovery_enabled": enabled,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to toggle auto-recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/performance-history")
async def get_performance_history(limit: int = 100) -> Dict[str, Any]:
    """Get performance metrics history."""
    try:
        history = list(advanced_recovery_system.performance_history)[-limit:]
        
        return {
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_usage": m.cpu_usage,
                    "memory_usage": m.memory_usage,
                    "disk_io": m.disk_io,
                    "network_io": m.network_io,
                    "response_time": m.response_time,
                    "throughput": m.throughput,
                    "error_rate": m.error_rate
                }
                for m in history
            ],
            "total_samples": len(advanced_recovery_system.performance_history),
            "returned_samples": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recovery-patterns")
async def get_recovery_patterns() -> Dict[str, Any]:
    """Get available recovery patterns."""
    try:
        return {
            "recovery_patterns": advanced_recovery_system.recovery_patterns,
            "optimization_rules": list(advanced_recovery_system.optimization_rules.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get recovery patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))