"""
API Routes for Learning Optimization System

This module provides REST API endpoints for the learning optimization system,
enabling continuous learning optimization, effectiveness measurement, and adaptive learning.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ...engines.learning_optimization_system import LearningOptimizationSystem
from ...models.knowledge_integration_models import LearningMetric, LearningOptimization

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/learning-optimization", tags=["Learning Optimization"])

# Global instance (in production, use dependency injection)
learning_system = LearningOptimizationSystem()


@router.post("/optimize-continuous-learning", response_model=Dict[str, Any])
async def optimize_continuous_learning(
    learning_context: Dict[str, Any],
    current_metrics: List[Dict[str, Any]],
    optimization_targets: List[str]
):
    """
    Optimize continuous learning processes
    
    Args:
        learning_context: Context about the learning environment
        current_metrics: Current learning performance metrics
        optimization_targets: List of metrics to optimize
        
    Returns:
        Learning optimization configuration and results
    """
    try:
        # Convert metric dictionaries to LearningMetric objects
        metric_objects = []
        for metric_data in current_metrics:
            metric = LearningMetric(
                metric_name=metric_data["metric_name"],
                value=metric_data["value"],
                timestamp=datetime.fromisoformat(metric_data["timestamp"]) if "timestamp" in metric_data else datetime.now(),
                context=metric_data.get("context", {}),
                improvement_rate=metric_data.get("improvement_rate", 0.0)
            )
            metric_objects.append(metric)
        
        # Optimize continuous learning
        optimization = await learning_system.optimize_continuous_learning(
            learning_context, metric_objects, optimization_targets
        )
        
        return {
            "optimization_id": optimization.id,
            "optimization_target": optimization.optimization_target,
            "optimization_strategy": optimization.optimization_strategy,
            "parameters": optimization.parameters,
            "effectiveness_score": optimization.effectiveness_score,
            "created_at": optimization.created_at.isoformat(),
            "current_metrics_count": len(optimization.current_metrics),
            "improvements_count": len(optimization.improvements)
        }
        
    except Exception as e:
        logger.error(f"Error optimizing continuous learning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/measure-effectiveness", response_model=Dict[str, Any])
async def measure_learning_effectiveness(
    learning_activities: List[Dict[str, Any]],
    time_window_days: int = 30
):
    """
    Measure learning effectiveness across different activities
    
    Args:
        learning_activities: List of learning activities to measure
        time_window_days: Time window for measurement in days
        
    Returns:
        Learning effectiveness measurements
    """
    try:
        time_window = timedelta(days=time_window_days)
        
        # Measure learning effectiveness
        effectiveness = await learning_system.measure_learning_effectiveness(
            learning_activities, time_window
        )
        
        return effectiveness
        
    except Exception as e:
        logger.error(f"Error measuring learning effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/implement-adaptive-learning", response_model=Dict[str, Any])
async def implement_adaptive_learning(
    learning_context: Dict[str, Any],
    performance_feedback: List[Dict[str, Any]],
    adaptation_goals: List[str]
):
    """
    Implement adaptive learning and process enhancement
    
    Args:
        learning_context: Context about the learning environment
        performance_feedback: Feedback on learning performance
        adaptation_goals: Goals for adaptation
        
    Returns:
        Adaptive learning implementation results
    """
    try:
        # Implement adaptive learning
        adaptive_result = await learning_system.implement_adaptive_learning(
            learning_context, performance_feedback, adaptation_goals
        )
        
        return adaptive_result
        
    except Exception as e:
        logger.error(f"Error implementing adaptive learning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhance-processes", response_model=Dict[str, Any])
async def enhance_innovation_processes(
    process_data: List[Dict[str, Any]],
    enhancement_objectives: List[str]
):
    """
    Enhance innovation processes based on learning insights
    
    Args:
        process_data: Data about current innovation processes
        enhancement_objectives: Objectives for process enhancement
        
    Returns:
        Process enhancement recommendations and results
    """
    try:
        # Enhance innovation processes
        enhancement_result = await learning_system.enhance_innovation_processes(
            process_data, enhancement_objectives
        )
        
        return enhancement_result
        
    except Exception as e:
        logger.error(f"Error enhancing innovation processes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-parameters/{optimization_id}", response_model=Dict[str, Any])
async def optimize_learning_parameters(
    optimization_id: str,
    performance_data: List[Dict[str, Any]]
):
    """
    Optimize learning parameters based on performance data
    
    Args:
        optimization_id: ID of learning optimization to update
        performance_data: Recent performance data
        
    Returns:
        Parameter optimization results
    """
    try:
        # Optimize learning parameters
        optimization_result = await learning_system.optimize_learning_parameters(
            optimization_id, performance_data
        )
        
        return optimization_result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing learning parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimizations", response_model=List[Dict[str, Any]])
async def list_learning_optimizations(
    limit: int = 50,
    offset: int = 0
):
    """
    List learning optimizations
    
    Args:
        limit: Maximum number of optimizations to return
        offset: Number of optimizations to skip
        
    Returns:
        List of learning optimizations
    """
    try:
        optimizations = list(learning_system.learning_optimizations.values())
        
        # Sort by creation date (newest first)
        optimizations.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        optimizations = optimizations[offset:offset + limit]
        
        return [
            {
                "optimization_id": opt.id,
                "optimization_target": opt.optimization_target,
                "optimization_strategy": opt.optimization_strategy,
                "effectiveness_score": opt.effectiveness_score,
                "created_at": opt.created_at.isoformat(),
                "last_updated": opt.last_updated.isoformat(),
                "improvements_count": len(opt.improvements)
            }
            for opt in optimizations
        ]
        
    except Exception as e:
        logger.error(f"Error listing learning optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimizations/{optimization_id}", response_model=Dict[str, Any])
async def get_learning_optimization(optimization_id: str):
    """
    Get specific learning optimization details
    
    Args:
        optimization_id: ID of optimization to retrieve
        
    Returns:
        Learning optimization details
    """
    try:
        if optimization_id not in learning_system.learning_optimizations:
            raise HTTPException(status_code=404, detail="Learning optimization not found")
        
        optimization = learning_system.learning_optimizations[optimization_id]
        
        return {
            "optimization_id": optimization.id,
            "optimization_target": optimization.optimization_target,
            "optimization_strategy": optimization.optimization_strategy,
            "parameters": optimization.parameters,
            "effectiveness_score": optimization.effectiveness_score,
            "created_at": optimization.created_at.isoformat(),
            "last_updated": optimization.last_updated.isoformat(),
            "current_metrics": [
                {
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "context": metric.context,
                    "improvement_rate": metric.improvement_rate
                }
                for metric in optimization.current_metrics
            ],
            "improvements": optimization.improvements
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting learning optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics-history/{metric_name}", response_model=List[Dict[str, Any]])
async def get_metrics_history(
    metric_name: str,
    limit: int = 100
):
    """
    Get metrics history for a specific metric
    
    Args:
        metric_name: Name of metric to retrieve history for
        limit: Maximum number of metrics to return
        
    Returns:
        List of historical metrics
    """
    try:
        if metric_name not in learning_system.learning_metrics_history:
            return []
        
        metrics = list(learning_system.learning_metrics_history[metric_name])
        
        # Sort by timestamp (newest first)
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        metrics = metrics[:limit]
        
        return [
            {
                "metric_name": metric.metric_name,
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "context": metric.context,
                "improvement_rate": metric.improvement_rate
            }
            for metric in metrics
        ]
        
    except Exception as e:
        logger.error(f"Error getting metrics history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptive-parameters", response_model=Dict[str, Any])
async def get_adaptive_parameters():
    """
    Get current adaptive parameters
    
    Returns:
        Current adaptive parameters
    """
    try:
        if not learning_system.adaptive_parameters:
            return {"message": "No adaptive parameters available"}
        
        # Get latest parameters
        latest_timestamp = max(learning_system.adaptive_parameters.keys())
        latest_parameters = learning_system.adaptive_parameters[latest_timestamp]
        
        return {
            "timestamp": latest_timestamp,
            "parameters": latest_parameters,
            "parameter_count": len(latest_parameters)
        }
        
    except Exception as e:
        logger.error(f"Error getting adaptive parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-baselines", response_model=Dict[str, float])
async def get_performance_baselines():
    """
    Get performance baselines
    
    Returns:
        Performance baselines for different metrics
    """
    try:
        return learning_system.performance_baselines
        
    except Exception as e:
        logger.error(f"Error getting performance baselines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance-baselines", response_model=Dict[str, str])
async def set_performance_baseline(
    metric_name: str,
    baseline_value: float
):
    """
    Set performance baseline for a metric
    
    Args:
        metric_name: Name of metric to set baseline for
        baseline_value: Baseline value to set
        
    Returns:
        Success message
    """
    try:
        learning_system.performance_baselines[metric_name] = baseline_value
        
        return {"message": f"Baseline set for {metric_name}: {baseline_value}"}
        
    except Exception as e:
        logger.error(f"Error setting performance baseline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_learning_optimization_stats():
    """
    Get learning optimization statistics
    
    Returns:
        Statistics about learning optimizations
    """
    try:
        optimizations = list(learning_system.learning_optimizations.values())
        
        if not optimizations:
            return {
                "total_optimizations": 0,
                "average_effectiveness": 0.0,
                "optimization_strategies": {},
                "metrics_tracked": 0
            }
        
        # Calculate statistics
        stats = {
            "total_optimizations": len(optimizations),
            "average_effectiveness": sum(opt.effectiveness_score for opt in optimizations) / len(optimizations),
            "optimization_strategies": {},
            "metrics_tracked": len(learning_system.learning_metrics_history),
            "adaptive_parameters_count": len(learning_system.adaptive_parameters),
            "performance_baselines_count": len(learning_system.performance_baselines)
        }
        
        # Strategy distribution
        for optimization in optimizations:
            strategy = optimization.optimization_strategy
            stats["optimization_strategies"][strategy] = stats["optimization_strategies"].get(strategy, 0) + 1
        
        # Effectiveness distribution
        high_effectiveness = len([opt for opt in optimizations if opt.effectiveness_score > 0.8])
        medium_effectiveness = len([opt for opt in optimizations if 0.5 <= opt.effectiveness_score <= 0.8])
        low_effectiveness = len([opt for opt in optimizations if opt.effectiveness_score < 0.5])
        
        stats["effectiveness_distribution"] = {
            "high": high_effectiveness,
            "medium": medium_effectiveness,
            "low": low_effectiveness
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting learning optimization stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/optimizations/{optimization_id}")
async def delete_learning_optimization(optimization_id: str):
    """
    Delete a learning optimization
    
    Args:
        optimization_id: ID of optimization to delete
        
    Returns:
        Success message
    """
    try:
        if optimization_id not in learning_system.learning_optimizations:
            raise HTTPException(status_code=404, detail="Learning optimization not found")
        
        del learning_system.learning_optimizations[optimization_id]
        
        return {"message": f"Learning optimization {optimization_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting learning optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-optimize", response_model=Dict[str, Any])
async def batch_optimize_learning(
    batch_requests: List[Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Batch optimize learning for multiple contexts
    
    Args:
        batch_requests: List of optimization requests
        background_tasks: Background task manager
        
    Returns:
        Batch optimization results
    """
    try:
        results = []
        
        for i, request in enumerate(batch_requests):
            try:
                learning_context = request.get("learning_context", {})
                current_metrics_data = request.get("current_metrics", [])
                optimization_targets = request.get("optimization_targets", [])
                
                # Convert metric dictionaries to LearningMetric objects
                metric_objects = []
                for metric_data in current_metrics_data:
                    metric = LearningMetric(
                        metric_name=metric_data["metric_name"],
                        value=metric_data["value"],
                        timestamp=datetime.fromisoformat(metric_data["timestamp"]) if "timestamp" in metric_data else datetime.now(),
                        context=metric_data.get("context", {}),
                        improvement_rate=metric_data.get("improvement_rate", 0.0)
                    )
                    metric_objects.append(metric)
                
                # Optimize continuous learning
                optimization = await learning_system.optimize_continuous_learning(
                    learning_context, metric_objects, optimization_targets
                )
                
                results.append({
                    "request_index": i,
                    "optimization_id": optimization.id,
                    "effectiveness_score": optimization.effectiveness_score,
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "request_index": i,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "batch_results": results,
            "total_requests": len(batch_requests),
            "successful_requests": len([r for r in results if r["status"] == "success"]),
            "failed_requests": len([r for r in results if r["status"] == "failed"]),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch learning optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))