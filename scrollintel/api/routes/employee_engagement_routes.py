"""
Employee Engagement API Routes for Cultural Transformation Leadership System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.employee_engagement_engine import EmployeeEngagementEngine
from ...models.employee_engagement_models import (
    Employee, EngagementActivity, EngagementStrategy, EngagementMetric,
    EngagementAssessment, EngagementPlan, EngagementFeedback, EngagementReport,
    EngagementImprovementPlan, EngagementLevel, EngagementActivityType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/employee-engagement", tags=["employee-engagement"])

# Global engine instance
engagement_engine = EmployeeEngagementEngine()


@router.post("/strategies", response_model=Dict[str, Any])
async def develop_engagement_strategy(
    organization_id: str,
    target_groups: List[str],
    cultural_objectives: List[str],
    current_engagement_data: Dict[str, Any]
):
    """
    Develop comprehensive employee engagement strategy
    """
    try:
        strategy = engagement_engine.develop_engagement_strategy(
            organization_id=organization_id,
            target_groups=target_groups,
            cultural_objectives=cultural_objectives,
            current_engagement_data=current_engagement_data
        )
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "target_groups": strategy.target_groups,
                "objectives": strategy.objectives,
                "timeline": {k: v.isoformat() for k, v in strategy.timeline.items()},
                "success_metrics": strategy.success_metrics,
                "budget_allocated": strategy.budget_allocated,
                "status": strategy.status
            },
            "message": "Employee engagement strategy developed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error developing engagement strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{strategy_id}/activities", response_model=Dict[str, Any])
async def design_engagement_activities(
    strategy_id: str,
    employee_profiles: List[Dict[str, Any]]
):
    """
    Design engagement activities for a strategy
    """
    try:
        # Get strategy
        strategy = engagement_engine.engagement_strategies.get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Convert employee profile dicts to Employee objects
        employees = []
        for profile in employee_profiles:
            employee = Employee(
                id=profile["id"],
                name=profile["name"],
                department=profile["department"],
                role=profile["role"],
                manager_id=profile.get("manager_id"),
                hire_date=datetime.fromisoformat(profile["hire_date"]),
                engagement_level=EngagementLevel(profile["engagement_level"]),
                cultural_alignment_score=profile["cultural_alignment_score"]
            )
            employees.append(employee)
        
        activities = engagement_engine.design_engagement_activities(
            strategy=strategy,
            employee_profiles=employees
        )
        
        return {
            "success": True,
            "activities": [
                {
                    "id": activity.id,
                    "name": activity.name,
                    "activity_type": activity.activity_type.value,
                    "description": activity.description,
                    "target_audience": activity.target_audience,
                    "objectives": activity.objectives,
                    "duration_minutes": activity.duration_minutes,
                    "facilitator": activity.facilitator,
                    "materials_needed": activity.materials_needed,
                    "success_criteria": activity.success_criteria,
                    "cultural_values_addressed": activity.cultural_values_addressed,
                    "status": activity.status
                }
                for activity in activities
            ],
            "message": f"Designed {len(activities)} engagement activities"
        }
        
    except Exception as e:
        logger.error(f"Error designing engagement activities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/activities/{activity_id}/execute", response_model=Dict[str, Any])
async def execute_engagement_activity(
    activity_id: str,
    participants: List[str],
    execution_context: Dict[str, Any]
):
    """
    Execute an engagement activity
    """
    try:
        execution_report = engagement_engine.execute_engagement_activity(
            activity_id=activity_id,
            participants=participants,
            execution_context=execution_context
        )
        
        return {
            "success": True,
            "execution_report": {
                "activity_id": execution_report["activity_id"],
                "participants": execution_report["participants"],
                "execution_date": execution_report["execution_date"].isoformat(),
                "preparation_results": execution_report["preparation_results"],
                "execution_results": execution_report["execution_results"],
                "followup_results": execution_report["followup_results"],
                "success_indicators": execution_report["success_indicators"]
            },
            "message": "Engagement activity executed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error executing engagement activity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/measurement", response_model=Dict[str, Any])
async def measure_engagement_effectiveness(
    organization_id: str,
    measurement_period: Dict[str, str]  # ISO format dates
):
    """
    Measure engagement effectiveness and generate report
    """
    try:
        # Convert date strings to datetime objects
        period = {
            k: datetime.fromisoformat(v) for k, v in measurement_period.items()
        }
        
        report = engagement_engine.measure_engagement_effectiveness(
            organization_id=organization_id,
            measurement_period=period
        )
        
        return {
            "success": True,
            "report": {
                "id": report.id,
                "organization_id": report.organization_id,
                "report_period": {k: v.isoformat() for k, v in report.report_period.items()},
                "overall_engagement_score": report.overall_engagement_score,
                "engagement_by_department": report.engagement_by_department,
                "engagement_trends": report.engagement_trends,
                "activity_effectiveness": report.activity_effectiveness,
                "key_insights": report.key_insights,
                "recommendations": report.recommendations,
                "metrics": [
                    {
                        "id": metric.id,
                        "metric_type": metric.metric_type.value,
                        "name": metric.name,
                        "current_value": metric.current_value,
                        "target_value": metric.target_value,
                        "trend": metric.trend
                    }
                    for metric in report.metrics
                ],
                "generated_date": report.generated_date.isoformat()
            },
            "message": "Engagement effectiveness measured successfully"
        }
        
    except Exception as e:
        logger.error(f"Error measuring engagement effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/improvement-plans", response_model=Dict[str, Any])
async def create_improvement_plan(
    report_id: str,
    improvement_goals: List[str]
):
    """
    Create engagement improvement plan based on report
    """
    try:
        # For this demo, we'll create a mock report
        # In a real implementation, you'd retrieve the actual report
        mock_report = EngagementReport(
            id=report_id,
            organization_id="demo_org",
            report_period={"start": datetime.now(), "end": datetime.now()},
            overall_engagement_score=3.5,
            engagement_by_department={"engineering": 3.8, "sales": 3.2},
            engagement_trends={"overall": [3.2, 3.3, 3.4, 3.5]},
            activity_effectiveness={"workshops": 0.75, "surveys": 0.80},
            key_insights=["Engagement below target", "Sales team needs attention"],
            recommendations=["Increase feedback frequency", "Improve recognition"],
            metrics=[],
            generated_date=datetime.now(),
            generated_by="engagement_engine"
        )
        
        improvement_plan = engagement_engine.create_improvement_plan(
            engagement_report=mock_report,
            improvement_goals=improvement_goals
        )
        
        return {
            "success": True,
            "improvement_plan": {
                "id": improvement_plan.id,
                "organization_id": improvement_plan.organization_id,
                "current_state": improvement_plan.current_state,
                "target_state": improvement_plan.target_state,
                "improvement_strategies": [
                    {
                        "id": strategy.id,
                        "name": strategy.name,
                        "description": strategy.description,
                        "objectives": strategy.objectives,
                        "budget_allocated": strategy.budget_allocated
                    }
                    for strategy in improvement_plan.improvement_strategies
                ],
                "implementation_timeline": {
                    k: v.isoformat() for k, v in improvement_plan.implementation_timeline.items()
                },
                "resource_requirements": improvement_plan.resource_requirements,
                "success_criteria": improvement_plan.success_criteria,
                "risk_mitigation": improvement_plan.risk_mitigation,
                "monitoring_plan": improvement_plan.monitoring_plan,
                "status": improvement_plan.status
            },
            "message": "Engagement improvement plan created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating improvement plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=Dict[str, Any])
async def process_engagement_feedback(
    employee_id: str,
    feedback_type: str,
    comments: str,
    rating: Optional[float] = None,
    activity_id: Optional[str] = None,
    suggestions: List[str] = []
):
    """
    Process employee engagement feedback
    """
    try:
        feedback = EngagementFeedback(
            id=f"feedback_{employee_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            employee_id=employee_id,
            activity_id=activity_id,
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            suggestions=suggestions,
            sentiment="neutral",  # Will be analyzed by engine
            themes=[],  # Will be extracted by engine
            submitted_date=datetime.now(),
            processed=False
        )
        
        processing_results = engagement_engine.process_engagement_feedback(feedback)
        
        return {
            "success": True,
            "processing_results": {
                "feedback_id": processing_results["feedback_id"],
                "sentiment_analysis": processing_results["sentiment_analysis"],
                "themes": processing_results["themes"],
                "insights": processing_results["insights"],
                "response_recommendations": processing_results["response_recommendations"],
                "processed_date": processing_results["processed_date"].isoformat()
            },
            "message": "Engagement feedback processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing engagement feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=Dict[str, Any])
async def list_engagement_strategies(organization_id: Optional[str] = None):
    """
    List all engagement strategies
    """
    try:
        strategies = list(engagement_engine.engagement_strategies.values())
        
        if organization_id:
            strategies = [s for s in strategies if s.organization_id == organization_id]
        
        return {
            "success": True,
            "strategies": [
                {
                    "id": strategy.id,
                    "name": strategy.name,
                    "organization_id": strategy.organization_id,
                    "target_groups": strategy.target_groups,
                    "objectives": strategy.objectives,
                    "status": strategy.status,
                    "created_date": strategy.created_date.isoformat()
                }
                for strategy in strategies
            ],
            "total": len(strategies)
        }
        
    except Exception as e:
        logger.error(f"Error listing engagement strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activities", response_model=Dict[str, Any])
async def list_engagement_activities(
    strategy_id: Optional[str] = None,
    activity_type: Optional[str] = None,
    status: Optional[str] = None
):
    """
    List engagement activities with optional filters
    """
    try:
        activities = list(engagement_engine.engagement_activities.values())
        
        # Apply filters
        if activity_type:
            activities = [a for a in activities if a.activity_type.value == activity_type]
        
        if status:
            activities = [a for a in activities if a.status == status]
        
        return {
            "success": True,
            "activities": [
                {
                    "id": activity.id,
                    "name": activity.name,
                    "activity_type": activity.activity_type.value,
                    "description": activity.description,
                    "duration_minutes": activity.duration_minutes,
                    "facilitator": activity.facilitator,
                    "status": activity.status,
                    "created_date": activity.created_date.isoformat(),
                    "scheduled_date": activity.scheduled_date.isoformat() if activity.scheduled_date else None
                }
                for activity in activities
            ],
            "total": len(activities)
        }
        
    except Exception as e:
        logger.error(f"Error listing engagement activities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback", response_model=Dict[str, Any])
async def list_engagement_feedback(
    employee_id: Optional[str] = None,
    feedback_type: Optional[str] = None,
    processed: Optional[bool] = None
):
    """
    List engagement feedback with optional filters
    """
    try:
        feedback_list = list(engagement_engine.feedback_data.values())
        
        # Apply filters
        if employee_id:
            feedback_list = [f for f in feedback_list if f.employee_id == employee_id]
        
        if feedback_type:
            feedback_list = [f for f in feedback_list if f.feedback_type == feedback_type]
        
        if processed is not None:
            feedback_list = [f for f in feedback_list if f.processed == processed]
        
        return {
            "success": True,
            "feedback": [
                {
                    "id": feedback.id,
                    "employee_id": feedback.employee_id,
                    "feedback_type": feedback.feedback_type,
                    "rating": feedback.rating,
                    "comments": feedback.comments,
                    "sentiment": feedback.sentiment,
                    "themes": feedback.themes,
                    "processed": feedback.processed,
                    "submitted_date": feedback.submitted_date.isoformat()
                }
                for feedback in feedback_list
            ],
            "total": len(feedback_list)
        }
        
    except Exception as e:
        logger.error(f"Error listing engagement feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{organization_id}", response_model=Dict[str, Any])
async def get_engagement_metrics(organization_id: str):
    """
    Get current engagement metrics for organization
    """
    try:
        # Generate current metrics report
        current_period = {
            "start": datetime.now().replace(day=1),
            "end": datetime.now()
        }
        
        report = engagement_engine.measure_engagement_effectiveness(
            organization_id=organization_id,
            measurement_period=current_period
        )
        
        return {
            "success": True,
            "metrics": {
                "overall_engagement_score": report.overall_engagement_score,
                "engagement_by_department": report.engagement_by_department,
                "activity_effectiveness": report.activity_effectiveness,
                "key_insights": report.key_insights,
                "recommendations": report.recommendations
            },
            "last_updated": report.generated_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting engagement metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/strategies/{strategy_id}", response_model=Dict[str, Any])
async def delete_engagement_strategy(strategy_id: str):
    """
    Delete an engagement strategy
    """
    try:
        if strategy_id not in engagement_engine.engagement_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        del engagement_engine.engagement_strategies[strategy_id]
        
        return {
            "success": True,
            "message": "Engagement strategy deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting engagement strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/activities/{activity_id}", response_model=Dict[str, Any])
async def delete_engagement_activity(activity_id: str):
    """
    Delete an engagement activity
    """
    try:
        if activity_id not in engagement_engine.engagement_activities:
            raise HTTPException(status_code=404, detail="Activity not found")
        
        del engagement_engine.engagement_activities[activity_id]
        
        return {
            "success": True,
            "message": "Engagement activity deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting engagement activity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint for employee engagement system
    """
    return {
        "success": True,
        "service": "employee_engagement",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }