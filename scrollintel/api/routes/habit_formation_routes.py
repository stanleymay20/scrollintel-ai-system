"""
API Routes for Habit Formation Engine

Provides REST endpoints for organizational habit formation functionality
including habit design, formation strategies, and sustainability mechanisms.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from datetime import datetime
import logging

from ...engines.habit_formation_engine import HabitFormationEngine
from ...models.habit_formation_models import (
    OrganizationalHabit, HabitFormationStrategy, HabitSustainability,
    HabitProgress, HabitFormationMetrics, HabitType, HabitFrequency,
    HabitStage, SustainabilityLevel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/habit-formation", tags=["habit-formation"])

# Global engine instance
habit_engine = HabitFormationEngine()


@router.post("/habits/design")
async def design_organizational_habit(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create new positive organizational habit identification and design
    
    Requirements: 3.3, 3.4 - Create new positive organizational habit identification and design
    """
    try:
        name = request_data.get("name", "")
        description = request_data.get("description", "")
        habit_type_str = request_data.get("habit_type", "")
        target_behavior = request_data.get("target_behavior", "")
        participants = request_data.get("participants", [])
        cultural_values = request_data.get("cultural_values", [])
        business_objectives = request_data.get("business_objectives", [])
        
        if not all([name, description, habit_type_str, target_behavior, participants]):
            raise HTTPException(
                status_code=400,
                detail="name, description, habit_type, target_behavior, and participants are required"
            )
        
        try:
            habit_type = HabitType(habit_type_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid habit_type. Must be one of: {[t.value for t in HabitType]}"
            )
        
        habit = habit_engine.design_organizational_habit(
            name=name,
            description=description,
            habit_type=habit_type,
            target_behavior=target_behavior,
            participants=participants,
            cultural_values=cultural_values,
            business_objectives=business_objectives
        )
        
        return {
            "success": True,
            "habit_id": habit.id,
            "habit": {
                "id": habit.id,
                "name": habit.name,
                "description": habit.description,
                "habit_type": habit.habit_type.value,
                "target_behavior": habit.target_behavior,
                "trigger_conditions": habit.trigger_conditions,
                "execution_steps": habit.execution_steps,
                "success_indicators": habit.success_indicators,
                "frequency": habit.frequency.value,
                "duration_minutes": habit.duration_minutes,
                "participants": habit.participants,
                "facilitators": habit.facilitators,
                "resources_required": habit.resources_required,
                "cultural_alignment": habit.cultural_alignment,
                "business_impact": habit.business_impact,
                "stage": habit.stage.value,
                "created_date": habit.created_date.isoformat(),
                "created_by": habit.created_by
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error designing organizational habit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Habit design failed: {str(e)}")


@router.post("/habits/{habit_id}/strategy")
async def create_habit_formation_strategy(
    habit_id: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build habit formation strategy and implementation framework
    
    Requirements: 3.3, 3.4 - Build habit formation strategy and implementation framework
    """
    try:
        habit = habit_engine.get_habit(habit_id)
        if not habit:
            raise HTTPException(status_code=404, detail="Habit not found")
        
        organizational_context = request_data.get("organizational_context", {})
        
        strategy = habit_engine.create_habit_formation_strategy(
            habit=habit,
            organizational_context=organizational_context
        )
        
        return {
            "success": True,
            "strategy_id": strategy.id,
            "strategy": {
                "id": strategy.id,
                "habit_id": strategy.habit_id,
                "strategy_name": strategy.strategy_name,
                "description": strategy.description,
                "formation_phases": strategy.formation_phases,
                "timeline_weeks": strategy.timeline_weeks,
                "key_milestones": strategy.key_milestones,
                "success_metrics": strategy.success_metrics,
                "reinforcement_mechanisms": strategy.reinforcement_mechanisms,
                "barrier_mitigation": strategy.barrier_mitigation,
                "stakeholder_engagement": strategy.stakeholder_engagement,
                "resource_allocation": strategy.resource_allocation,
                "risk_assessment": strategy.risk_assessment,
                "created_date": strategy.created_date.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating habit formation strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Strategy creation failed: {str(e)}")


@router.post("/habits/{habit_id}/sustainability")
async def implement_sustainability_mechanisms(
    habit_id: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Implement habit sustainability and reinforcement mechanisms
    
    Requirements: 3.3, 3.4 - Implement habit sustainability and reinforcement mechanisms
    """
    try:
        habit = habit_engine.get_habit(habit_id)
        if not habit:
            raise HTTPException(status_code=404, detail="Habit not found")
        
        formation_progress = request_data.get("formation_progress", {})
        
        sustainability = habit_engine.implement_habit_sustainability_mechanisms(
            habit=habit,
            formation_progress=formation_progress
        )
        
        return {
            "success": True,
            "sustainability_id": sustainability.id,
            "sustainability": {
                "id": sustainability.id,
                "habit_id": sustainability.habit_id,
                "sustainability_level": sustainability.sustainability_level.value,
                "reinforcement_systems": sustainability.reinforcement_systems,
                "monitoring_mechanisms": sustainability.monitoring_mechanisms,
                "feedback_loops": sustainability.feedback_loops,
                "adaptation_triggers": sustainability.adaptation_triggers,
                "renewal_strategies": sustainability.renewal_strategies,
                "institutional_support": sustainability.institutional_support,
                "cultural_integration": sustainability.cultural_integration,
                "resilience_factors": sustainability.resilience_factors,
                "vulnerability_points": sustainability.vulnerability_points,
                "mitigation_plans": sustainability.mitigation_plans,
                "sustainability_score": sustainability.sustainability_score,
                "last_assessment": sustainability.last_assessment.isoformat(),
                "next_review": sustainability.next_review.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error implementing sustainability mechanisms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sustainability implementation failed: {str(e)}")


@router.post("/habits/{habit_id}/progress")
async def track_habit_progress(
    habit_id: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Track progress of habit formation and execution"""
    try:
        habit = habit_engine.get_habit(habit_id)
        if not habit:
            raise HTTPException(status_code=404, detail="Habit not found")
        
        participant_id = request_data.get("participant_id", "")
        tracking_period = request_data.get("tracking_period", "")
        execution_count = request_data.get("execution_count", 0)
        target_count = request_data.get("target_count", 1)
        quality_score = request_data.get("quality_score", 0.8)
        engagement_level = request_data.get("engagement_level", 0.8)
        
        if not participant_id or not tracking_period:
            raise HTTPException(
                status_code=400,
                detail="participant_id and tracking_period are required"
            )
        
        # Validate scores
        for score_name, score_value in [("quality_score", quality_score), ("engagement_level", engagement_level)]:
            if not 0.0 <= score_value <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"{score_name} must be between 0.0 and 1.0"
                )
        
        progress = habit_engine.track_habit_progress(
            habit_id=habit_id,
            participant_id=participant_id,
            tracking_period=tracking_period,
            execution_count=execution_count,
            target_count=target_count,
            quality_score=quality_score,
            engagement_level=engagement_level
        )
        
        return {
            "success": True,
            "progress_id": progress.id,
            "progress": {
                "id": progress.id,
                "habit_id": progress.habit_id,
                "participant_id": progress.participant_id,
                "tracking_period": progress.tracking_period,
                "execution_count": progress.execution_count,
                "target_count": progress.target_count,
                "consistency_rate": progress.consistency_rate,
                "quality_score": progress.quality_score,
                "engagement_level": progress.engagement_level,
                "barriers_encountered": progress.barriers_encountered,
                "support_received": progress.support_received,
                "improvements_noted": progress.improvements_noted,
                "feedback_provided": progress.feedback_provided,
                "next_period_goals": progress.next_period_goals,
                "recorded_date": progress.recorded_date.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking habit progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Progress tracking failed: {str(e)}")


@router.get("/habits/{habit_id}")
async def get_organizational_habit(habit_id: str) -> Dict[str, Any]:
    """Get organizational habit details"""
    try:
        habit = habit_engine.get_habit(habit_id)
        
        if not habit:
            raise HTTPException(status_code=404, detail="Habit not found")
        
        return {
            "success": True,
            "habit": {
                "id": habit.id,
                "name": habit.name,
                "description": habit.description,
                "habit_type": habit.habit_type.value,
                "target_behavior": habit.target_behavior,
                "trigger_conditions": habit.trigger_conditions,
                "execution_steps": habit.execution_steps,
                "success_indicators": habit.success_indicators,
                "frequency": habit.frequency.value,
                "duration_minutes": habit.duration_minutes,
                "participants": habit.participants,
                "facilitators": habit.facilitators,
                "resources_required": habit.resources_required,
                "cultural_alignment": habit.cultural_alignment,
                "business_impact": habit.business_impact,
                "stage": habit.stage.value,
                "created_date": habit.created_date.isoformat(),
                "created_by": habit.created_by
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving habit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve habit: {str(e)}")


@router.get("/organization/{organization_id}/habits")
async def get_organization_habits(organization_id: str) -> Dict[str, Any]:
    """Get all organizational habits for an organization"""
    try:
        habits = habit_engine.get_organization_habits(organization_id)
        
        return {
            "success": True,
            "organization_id": organization_id,
            "habits_count": len(habits),
            "habits": [
                {
                    "id": habit.id,
                    "name": habit.name,
                    "habit_type": habit.habit_type.value,
                    "target_behavior": habit.target_behavior,
                    "frequency": habit.frequency.value,
                    "participants_count": len(habit.participants),
                    "cultural_alignment": habit.cultural_alignment,
                    "stage": habit.stage.value,
                    "created_date": habit.created_date.isoformat()
                }
                for habit in habits
            ]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving organization habits: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve habits: {str(e)}")


@router.post("/organization/{organization_id}/metrics")
async def calculate_habit_formation_metrics(organization_id: str) -> Dict[str, Any]:
    """Calculate comprehensive habit formation metrics for an organization"""
    try:
        metrics = habit_engine.calculate_habit_formation_metrics(organization_id)
        
        return {
            "success": True,
            "organization_id": organization_id,
            "metrics": {
                "total_habits_designed": metrics.total_habits_designed,
                "habits_in_formation": metrics.habits_in_formation,
                "habits_established": metrics.habits_established,
                "habits_institutionalized": metrics.habits_institutionalized,
                "average_formation_time_weeks": metrics.average_formation_time_weeks,
                "overall_success_rate": metrics.overall_success_rate,
                "participant_engagement_average": metrics.participant_engagement_average,
                "sustainability_index": metrics.sustainability_index,
                "cultural_integration_score": metrics.cultural_integration_score,
                "business_impact_score": metrics.business_impact_score,
                "roi_achieved": metrics.roi_achieved,
                "calculated_date": metrics.calculated_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating habit formation metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")


@router.get("/strategies/{strategy_id}")
async def get_formation_strategy(strategy_id: str) -> Dict[str, Any]:
    """Get habit formation strategy details"""
    try:
        strategy = habit_engine.get_formation_strategy(strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Formation strategy not found")
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "habit_id": strategy.habit_id,
                "strategy_name": strategy.strategy_name,
                "description": strategy.description,
                "formation_phases": strategy.formation_phases,
                "timeline_weeks": strategy.timeline_weeks,
                "key_milestones": strategy.key_milestones,
                "success_metrics": strategy.success_metrics,
                "reinforcement_mechanisms": strategy.reinforcement_mechanisms,
                "barrier_mitigation": strategy.barrier_mitigation,
                "stakeholder_engagement": strategy.stakeholder_engagement,
                "resource_allocation": strategy.resource_allocation,
                "risk_assessment": strategy.risk_assessment,
                "created_date": strategy.created_date.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving formation strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve strategy: {str(e)}")


@router.get("/sustainability/{sustainability_id}")
async def get_sustainability_mechanism(sustainability_id: str) -> Dict[str, Any]:
    """Get habit sustainability mechanism details"""
    try:
        sustainability = habit_engine.get_sustainability_mechanism(sustainability_id)
        
        if not sustainability:
            raise HTTPException(status_code=404, detail="Sustainability mechanism not found")
        
        return {
            "success": True,
            "sustainability": {
                "id": sustainability.id,
                "habit_id": sustainability.habit_id,
                "sustainability_level": sustainability.sustainability_level.value,
                "reinforcement_systems": sustainability.reinforcement_systems,
                "monitoring_mechanisms": sustainability.monitoring_mechanisms,
                "feedback_loops": sustainability.feedback_loops,
                "adaptation_triggers": sustainability.adaptation_triggers,
                "renewal_strategies": sustainability.renewal_strategies,
                "institutional_support": sustainability.institutional_support,
                "cultural_integration": sustainability.cultural_integration,
                "resilience_factors": sustainability.resilience_factors,
                "vulnerability_points": sustainability.vulnerability_points,
                "mitigation_plans": sustainability.mitigation_plans,
                "sustainability_score": sustainability.sustainability_score,
                "last_assessment": sustainability.last_assessment.isoformat(),
                "next_review": sustainability.next_review.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sustainability mechanism: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sustainability mechanism: {str(e)}")


@router.get("/habit-types")
async def get_habit_types() -> Dict[str, Any]:
    """Get available habit types"""
    return {
        "success": True,
        "habit_types": [
            {
                "value": habit_type.value,
                "name": habit_type.value.replace("_", " ").title(),
                "description": f"Habits focused on {habit_type.value.replace('_', ' ')}"
            }
            for habit_type in HabitType
        ]
    }


@router.get("/frequencies")
async def get_habit_frequencies() -> Dict[str, Any]:
    """Get available habit frequencies"""
    return {
        "success": True,
        "frequencies": [
            {
                "value": frequency.value,
                "name": frequency.value.replace("_", " ").title()
            }
            for frequency in HabitFrequency
        ]
    }


@router.get("/stages")
async def get_habit_stages() -> Dict[str, Any]:
    """Get available habit stages"""
    return {
        "success": True,
        "stages": [
            {
                "value": stage.value,
                "name": stage.value.replace("_", " ").title()
            }
            for stage in HabitStage
        ]
    }


@router.get("/sustainability-levels")
async def get_sustainability_levels() -> Dict[str, Any]:
    """Get available sustainability levels"""
    return {
        "success": True,
        "sustainability_levels": [
            {
                "value": level.value,
                "name": level.value.replace("_", " ").title()
            }
            for level in SustainabilityLevel
        ]
    }


@router.post("/validate/habit")
async def validate_habit_data(habit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate organizational habit data"""
    try:
        required_fields = ["name", "description", "habit_type", "target_behavior", "participants"]
        missing_fields = [field for field in required_fields if not habit_data.get(field)]
        
        if missing_fields:
            return {
                "success": True,
                "valid": False,
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }
        
        # Validate habit type
        habit_type_str = habit_data.get("habit_type")
        try:
            HabitType(habit_type_str)
        except ValueError:
            return {
                "success": True,
                "valid": False,
                "message": f"Invalid habit_type. Must be one of: {[t.value for t in HabitType]}"
            }
        
        # Validate participants list
        participants = habit_data.get("participants", [])
        if not isinstance(participants, list) or len(participants) == 0:
            return {
                "success": True,
                "valid": False,
                "message": "participants must be a non-empty list"
            }
        
        # Validate cultural alignment if provided
        cultural_alignment = habit_data.get("cultural_alignment")
        if cultural_alignment is not None and not 0.0 <= cultural_alignment <= 1.0:
            return {
                "success": True,
                "valid": False,
                "message": "cultural_alignment must be between 0.0 and 1.0"
            }
        
        return {
            "success": True,
            "valid": True,
            "message": "Habit data is valid"
        }
        
    except Exception as e:
        return {
            "success": True,
            "valid": False,
            "message": f"Validation failed: {str(e)}"
        }