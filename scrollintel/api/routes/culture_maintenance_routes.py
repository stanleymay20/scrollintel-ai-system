"""
Culture Maintenance API Routes

API endpoints for cultural sustainability assessment and maintenance framework.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from ...engines.culture_maintenance_engine import CultureMaintenanceEngine
from ...models.culture_maintenance_models import (
    SustainabilityAssessment, MaintenanceStrategy, CultureMaintenancePlan,
    LongTermMonitoringResult
)
from ...models.cultural_assessment_models import CultureMap, CulturalTransformation

router = APIRouter(prefix="/api/culture-maintenance", tags=["culture-maintenance"])
logger = logging.getLogger(__name__)

# Initialize engine
maintenance_engine = CultureMaintenanceEngine()


@router.post("/assess-sustainability")
async def assess_cultural_sustainability(
    organization_id: str,
    transformation_data: Dict[str, Any],
    current_culture_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Assess sustainability of cultural changes"""
    try:
        # Convert input data to models (simplified for demo)
        transformation = CulturalTransformation(
            id=transformation_data.get("id", "demo_transformation"),
            organization_id=organization_id,
            current_culture={},
            target_culture={},
            vision={},
            roadmap={},
            interventions=[],
            progress=transformation_data.get("progress", 0.8),
            start_date=None,
            target_completion=None
        )
        
        current_culture = CultureMap(
            organization_id=organization_id,
            assessment_date=None,
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=current_culture_data.get("health_score", 0.75),
            assessment_confidence=0.8,
            data_sources=["api"]
        )
        
        assessment = maintenance_engine.assess_cultural_sustainability(
            organization_id, transformation, current_culture
        )
        
        return {
            "success": True,
            "assessment": {
                "assessment_id": assessment.assessment_id,
                "sustainability_level": assessment.sustainability_level.value,
                "overall_score": assessment.overall_score,
                "risk_factors": assessment.risk_factors,
                "protective_factors": assessment.protective_factors,
                "health_indicators": [
                    {
                        "name": indicator.name,
                        "current_value": indicator.current_value,
                        "target_value": indicator.target_value,
                        "trend": indicator.trend
                    }
                    for indicator in assessment.health_indicators
                ],
                "next_assessment_due": assessment.next_assessment_due.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error assessing cultural sustainability: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/develop-maintenance-strategy")
async def develop_maintenance_strategy(
    organization_id: str,
    assessment_data: Dict[str, Any],
    target_culture_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Develop comprehensive culture maintenance strategies"""
    try:
        # Create sustainability assessment from data
        sustainability_assessment = SustainabilityAssessment(
            assessment_id=assessment_data.get("assessment_id", "demo_assessment"),
            organization_id=organization_id,
            transformation_id=assessment_data.get("transformation_id", "demo_transformation"),
            sustainability_level=assessment_data.get("sustainability_level", "medium"),
            risk_factors=assessment_data.get("risk_factors", []),
            protective_factors=assessment_data.get("protective_factors", []),
            health_indicators=[],
            overall_score=assessment_data.get("overall_score", 0.7),
            assessment_date=None,
            next_assessment_due=None
        )
        
        target_culture = CultureMap(
            organization_id=organization_id,
            assessment_date=None,
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_metrics=[],
            overall_health_score=target_culture_data.get("health_score", 0.85),
            assessment_confidence=0.8,
            data_sources=["api"]
        )
        
        strategies = maintenance_engine.develop_maintenance_strategy(
            organization_id, sustainability_assessment, target_culture
        )
        
        return {
            "success": True,
            "strategies": [
                {
                    "strategy_id": strategy.strategy_id,
                    "target_culture_elements": strategy.target_culture_elements,
                    "maintenance_activities": strategy.maintenance_activities,
                    "success_metrics": strategy.success_metrics,
                    "review_frequency": strategy.review_frequency,
                    "resource_requirements": strategy.resource_requirements
                }
                for strategy in strategies
            ]
        }
        
    except Exception as e:
        logger.error(f"Error developing maintenance strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-maintenance-plan")
async def create_maintenance_plan(
    organization_id: str,
    assessment_data: Dict[str, Any],
    strategies_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create comprehensive culture maintenance plan"""
    try:
        # Create sustainability assessment from data
        sustainability_assessment = SustainabilityAssessment(
            assessment_id=assessment_data.get("assessment_id", "demo_assessment"),
            organization_id=organization_id,
            transformation_id=assessment_data.get("transformation_id", "demo_transformation"),
            sustainability_level=assessment_data.get("sustainability_level", "medium"),
            risk_factors=assessment_data.get("risk_factors", []),
            protective_factors=assessment_data.get("protective_factors", []),
            health_indicators=[],
            overall_score=assessment_data.get("overall_score", 0.7),
            assessment_date=None,
            next_assessment_due=None
        )
        
        # Create maintenance strategies from data
        maintenance_strategies = []
        for strategy_data in strategies_data:
            strategy = MaintenanceStrategy(
                strategy_id=strategy_data.get("strategy_id", "demo_strategy"),
                organization_id=organization_id,
                target_culture_elements=strategy_data.get("target_culture_elements", []),
                maintenance_activities=strategy_data.get("maintenance_activities", []),
                monitoring_schedule=strategy_data.get("monitoring_schedule", {}),
                resource_requirements=strategy_data.get("resource_requirements", {}),
                success_metrics=strategy_data.get("success_metrics", []),
                review_frequency=strategy_data.get("review_frequency", "monthly"),
                created_date=None
            )
            maintenance_strategies.append(strategy)
        
        plan = maintenance_engine.create_maintenance_plan(
            organization_id, sustainability_assessment, maintenance_strategies
        )
        
        return {
            "success": True,
            "plan": {
                "plan_id": plan.plan_id,
                "status": plan.status.value,
                "monitoring_framework": plan.monitoring_framework,
                "intervention_triggers": plan.intervention_triggers,
                "resource_allocation": plan.resource_allocation,
                "timeline": plan.timeline,
                "created_date": plan.created_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating maintenance plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor-long-term-health")
async def monitor_long_term_health(
    organization_id: str,
    plan_data: Dict[str, Any],
    monitoring_period_days: int = 90
) -> Dict[str, Any]:
    """Monitor long-term culture health and optimization"""
    try:
        # Create maintenance plan from data (simplified)
        maintenance_plan = CultureMaintenancePlan(
            plan_id=plan_data.get("plan_id", "demo_plan"),
            organization_id=organization_id,
            sustainability_assessment=None,
            maintenance_strategies=[],
            monitoring_framework=plan_data.get("monitoring_framework", {}),
            intervention_triggers=plan_data.get("intervention_triggers", []),
            resource_allocation=plan_data.get("resource_allocation", {}),
            timeline=plan_data.get("timeline", {}),
            status=plan_data.get("status", "stable"),
            created_date=None,
            last_updated=None
        )
        
        monitoring_result = maintenance_engine.monitor_long_term_health(
            organization_id, maintenance_plan, monitoring_period_days
        )
        
        return {
            "success": True,
            "monitoring_result": {
                "monitoring_id": monitoring_result.monitoring_id,
                "monitoring_period": {
                    "start": monitoring_result.monitoring_period["start"].isoformat(),
                    "end": monitoring_result.monitoring_period["end"].isoformat()
                },
                "health_trends": monitoring_result.health_trends,
                "sustainability_metrics": monitoring_result.sustainability_metrics,
                "risk_indicators": monitoring_result.risk_indicators,
                "recommendations": monitoring_result.recommendations,
                "next_actions": monitoring_result.next_actions
            }
        }
        
    except Exception as e:
        logger.error(f"Error monitoring long-term health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-status/{organization_id}")
async def get_culture_health_status(organization_id: str) -> Dict[str, Any]:
    """Get current culture health status"""
    try:
        # Simulate getting current health status
        health_status = {
            "organization_id": organization_id,
            "overall_health_score": 0.78,
            "sustainability_level": "high",
            "last_assessment": "2024-01-15T10:00:00Z",
            "next_assessment_due": "2024-04-15T10:00:00Z",
            "active_maintenance_strategies": 4,
            "recent_trends": {
                "engagement": "improving",
                "alignment": "stable",
                "behavior_consistency": "improving"
            },
            "risk_indicators": [],
            "protective_factors": [
                "Strong leadership commitment",
                "Clear value system",
                "Effective communication channels"
            ]
        }
        
        return {
            "success": True,
            "health_status": health_status
        }
        
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/maintenance-recommendations/{organization_id}")
async def get_maintenance_recommendations(organization_id: str) -> Dict[str, Any]:
    """Get current maintenance recommendations"""
    try:
        recommendations = {
            "organization_id": organization_id,
            "immediate_actions": [
                "Continue current monitoring schedule",
                "Reinforce positive behavioral trends"
            ],
            "short_term_recommendations": [
                "Implement additional engagement initiatives",
                "Strengthen communication channels"
            ],
            "long_term_strategies": [
                "Develop culture champion network",
                "Establish continuous improvement processes"
            ],
            "resource_needs": {
                "time_investment": "15 hours per week",
                "budget_requirement": "medium",
                "personnel": "2-3 dedicated team members"
            }
        }
        
        return {
            "success": True,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting maintenance recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))