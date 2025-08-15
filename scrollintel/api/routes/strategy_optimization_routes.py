"""
API routes for Strategy Optimization Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from scrollintel.engines.strategy_optimization_engine import (
    StrategyOptimizationEngine,
    OptimizationContext,
    OptimizationMetric,
    OptimizationRecommendation,
    StrategyAdjustment,
    OptimizationType,
    OptimizationPriority
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/strategy-optimization", tags=["strategy-optimization"])

# Global engine instance
optimization_engine = StrategyOptimizationEngine()


@router.post("/optimize")
async def optimize_transformation_strategy(
    transformation_id: str,
    context_data: Dict[str, Any],
    metrics_data: List[Dict[str, Any]]
):
    """
    Generate optimization recommendations for a transformation strategy
    """
    try:
        # Convert input data to engine objects
        context = OptimizationContext(
            transformation_id=transformation_id,
            current_progress=context_data.get("current_progress", 0.0),
            timeline_status=context_data.get("timeline_status", "unknown"),
            budget_utilization=context_data.get("budget_utilization", 0.0),
            resistance_level=context_data.get("resistance_level", 0.0),
            engagement_score=context_data.get("engagement_score", 0.0),
            performance_metrics=context_data.get("performance_metrics", {}),
            external_factors=context_data.get("external_factors", [])
        )
        
        metrics = []
        for metric_data in metrics_data:
            metric = OptimizationMetric(
                name=metric_data["name"],
                current_value=metric_data["current_value"],
                target_value=metric_data["target_value"],
                weight=metric_data.get("weight", 1.0),
                trend=metric_data.get("trend", "stable"),
                last_updated=datetime.fromisoformat(metric_data.get("last_updated", datetime.now().isoformat()))
            )
            metrics.append(metric)
        
        # Generate optimization recommendations
        recommendations = optimization_engine.optimize_strategy(context, metrics)
        
        # Convert recommendations to response format
        response_recommendations = []
        for rec in recommendations:
            response_recommendations.append({
                "id": rec.id,
                "optimization_type": rec.optimization_type.value,
                "priority": rec.priority.value,
                "title": rec.title,
                "description": rec.description,
                "expected_impact": rec.expected_impact,
                "implementation_effort": rec.implementation_effort,
                "timeline": rec.timeline,
                "success_probability": rec.success_probability,
                "dependencies": rec.dependencies,
                "risks": rec.risks,
                "created_at": rec.created_at.isoformat()
            })
        
        return {
            "transformation_id": transformation_id,
            "optimization_context": {
                "current_progress": context.current_progress,
                "timeline_status": context.timeline_status,
                "resistance_level": context.resistance_level,
                "engagement_score": context.engagement_score
            },
            "recommendations": response_recommendations,
            "total_recommendations": len(response_recommendations),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error optimizing transformation strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/adjustments")
async def create_strategy_adjustment(
    transformation_id: str,
    recommendation_id: str,
    original_strategy: Dict[str, Any]
):
    """
    Create a strategy adjustment based on optimization recommendation
    """
    try:
        # Find the recommendation (in real implementation, this would be stored)
        # For now, create a mock recommendation
        mock_recommendation = OptimizationRecommendation(
            id=recommendation_id,
            optimization_type=OptimizationType.PERFORMANCE_BASED,
            priority=OptimizationPriority.HIGH,
            title="Mock Optimization",
            description="Mock optimization for testing",
            expected_impact=0.5,
            implementation_effort="Medium",
            timeline="1-2 months",
            success_probability=0.75
        )
        
        # Create strategy adjustment
        adjustment = optimization_engine.create_strategy_adjustment(
            transformation_id, mock_recommendation, original_strategy
        )
        
        return {
            "adjustment_id": adjustment.id,
            "transformation_id": adjustment.transformation_id,
            "adjustment_type": adjustment.adjustment_type,
            "rationale": adjustment.rationale,
            "expected_outcomes": adjustment.expected_outcomes,
            "implementation_date": adjustment.implementation_date.isoformat(),
            "status": adjustment.status,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating strategy adjustment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Adjustment creation failed: {str(e)}")


@router.post("/adjustments/{adjustment_id}/implement")
async def implement_strategy_adjustment(
    adjustment_id: str,
    background_tasks: BackgroundTasks
):
    """
    Implement a strategy adjustment
    """
    try:
        # Implement the adjustment
        success = optimization_engine.implement_adjustment(adjustment_id)
        
        if success:
            return {
                "adjustment_id": adjustment_id,
                "status": "implementation_started",
                "message": "Strategy adjustment implementation initiated",
                "implemented_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to implement adjustment")
        
    except Exception as e:
        logger.error(f"Error implementing strategy adjustment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Implementation failed: {str(e)}")


@router.get("/adjustments/{adjustment_id}/effectiveness")
async def monitor_adjustment_effectiveness(
    adjustment_id: str,
    post_metrics: Optional[List[Dict[str, Any]]] = None
):
    """
    Monitor the effectiveness of an implemented adjustment
    """
    try:
        # Convert post metrics if provided
        metrics = []
        if post_metrics:
            for metric_data in post_metrics:
                metric = OptimizationMetric(
                    name=metric_data["name"],
                    current_value=metric_data["current_value"],
                    target_value=metric_data["target_value"],
                    weight=metric_data.get("weight", 1.0),
                    trend=metric_data.get("trend", "stable"),
                    last_updated=datetime.fromisoformat(metric_data.get("last_updated", datetime.now().isoformat()))
                )
                metrics.append(metric)
        
        # Monitor effectiveness
        effectiveness_data = optimization_engine.monitor_adjustment_effectiveness(adjustment_id, metrics)
        
        return {
            "adjustment_id": adjustment_id,
            "effectiveness_analysis": effectiveness_data,
            "monitored_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error monitoring adjustment effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")


@router.post("/continuous-improvement")
async def generate_continuous_improvement_plan(
    transformation_id: str,
    optimization_history: List[Dict[str, Any]]
):
    """
    Generate continuous improvement plan based on optimization history
    """
    try:
        # Generate improvement plan
        improvement_plan = optimization_engine.generate_continuous_improvement_plan(
            transformation_id, optimization_history
        )
        
        return {
            "transformation_id": transformation_id,
            "improvement_plan": improvement_plan,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating continuous improvement plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")


@router.get("/transformations/{transformation_id}/optimization-history")
async def get_optimization_history(transformation_id: str):
    """
    Get optimization history for a transformation
    """
    try:
        # In real implementation, this would fetch from database
        # For now, return mock data
        history = [
            {
                "optimization_id": "opt_001",
                "optimization_type": "performance_based",
                "applied_at": "2024-01-15T10:00:00",
                "effectiveness_score": 0.8,
                "status": "completed"
            },
            {
                "optimization_id": "opt_002",
                "optimization_type": "engagement_based",
                "applied_at": "2024-01-20T14:30:00",
                "effectiveness_score": 0.75,
                "status": "completed"
            }
        ]
        
        return {
            "transformation_id": transformation_id,
            "optimization_history": history,
            "total_optimizations": len(history),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving optimization history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@router.get("/optimization-types")
async def get_optimization_types():
    """
    Get available optimization types and their descriptions
    """
    try:
        optimization_types = {
            "performance_based": {
                "name": "Performance-Based Optimization",
                "description": "Focuses on improving transformation performance metrics",
                "typical_use_cases": ["Low performance scores", "Missed targets", "Underperforming initiatives"]
            },
            "timeline_based": {
                "name": "Timeline-Based Optimization",
                "description": "Focuses on accelerating transformation timeline",
                "typical_use_cases": ["Behind schedule", "Tight deadlines", "Resource constraints"]
            },
            "resistance_based": {
                "name": "Resistance-Based Optimization",
                "description": "Focuses on reducing cultural resistance",
                "typical_use_cases": ["High resistance levels", "Stakeholder pushback", "Change fatigue"]
            },
            "engagement_based": {
                "name": "Engagement-Based Optimization",
                "description": "Focuses on improving employee engagement",
                "typical_use_cases": ["Low engagement scores", "Poor participation", "Motivation issues"]
            },
            "resource_based": {
                "name": "Resource-Based Optimization",
                "description": "Focuses on optimizing resource utilization",
                "typical_use_cases": ["Budget constraints", "Resource conflicts", "Efficiency issues"]
            }
        }
        
        return {
            "optimization_types": optimization_types,
            "total_types": len(optimization_types),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving optimization types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Type retrieval failed: {str(e)}")


@router.post("/recommendations/{recommendation_id}/feedback")
async def provide_recommendation_feedback(
    recommendation_id: str,
    feedback_data: Dict[str, Any]
):
    """
    Provide feedback on optimization recommendation
    """
    try:
        feedback = {
            "recommendation_id": recommendation_id,
            "usefulness_score": feedback_data.get("usefulness_score", 0),
            "implementation_difficulty": feedback_data.get("implementation_difficulty", "unknown"),
            "actual_impact": feedback_data.get("actual_impact", 0),
            "comments": feedback_data.get("comments", ""),
            "would_recommend": feedback_data.get("would_recommend", False),
            "feedback_provided_at": datetime.now().isoformat()
        }
        
        # In real implementation, this would be stored for learning
        logger.info(f"Received feedback for recommendation {recommendation_id}: {feedback}")
        
        return {
            "recommendation_id": recommendation_id,
            "feedback_recorded": True,
            "message": "Thank you for your feedback. It will help improve future recommendations.",
            "recorded_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording recommendation feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback recording failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for strategy optimization service"""
    return {
        "service": "strategy-optimization",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }