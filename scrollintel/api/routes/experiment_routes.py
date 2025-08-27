"""
API routes for A/B Testing Engine in the Advanced Prompt Management System.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from scrollintel.models.database import Base
from scrollintel.models.experiment_models import (
    Experiment, ExperimentVariant, VariantMetric, ExperimentResult,
    ExperimentSchedule, ExperimentStatus
)
from scrollintel.engines.experiment_engine import ExperimentEngine, ExperimentConfig

# Mock dependencies for demo
def get_db():
    """Mock database session for demo."""
    return None

def get_current_user():
    """Mock current user for demo."""
    return "demo-user"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])

# Initialize experiment engine
experiment_engine = ExperimentEngine()


@router.post("/", response_model=Dict[str, Any])
async def create_experiment(
    config: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new A/B test experiment."""
    try:
        # Convert dict to ExperimentConfig
        experiment_config = ExperimentConfig(
            name=config["name"],
            prompt_id=config["prompt_id"],
            hypothesis=config["hypothesis"],
            variants=config["variants"],
            success_metrics=config["success_metrics"],
            target_sample_size=config.get("target_sample_size", 1000),
            confidence_level=config.get("confidence_level", 0.95),
            minimum_effect_size=config.get("minimum_effect_size", 0.05),
            traffic_allocation=config.get("traffic_allocation", 1.0),
            duration_hours=config.get("duration_hours"),
            auto_start=config.get("auto_start", False),
            auto_stop=config.get("auto_stop", False),
            auto_promote_winner=config.get("auto_promote_winner", False),
            schedule_config=config.get("schedule_config")
        )
        
        result = await experiment_engine.create_experiment(experiment_config)
        return result
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[Dict[str, Any]])
async def list_experiments(
    status: Optional[str] = None,
    prompt_id: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all experiments with optional filtering."""
    try:
        query = db.query(Experiment)
        
        if status:
            query = query.filter(Experiment.status == status)
        if prompt_id:
            query = query.filter(Experiment.prompt_id == prompt_id)
            
        experiments = query.all()
        return [exp.to_dict() for exp in experiments]
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=Dict[str, Any])
async def get_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get experiment details by ID."""
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        return experiment.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/start", response_model=Dict[str, Any])
async def start_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start an A/B test experiment."""
    try:
        result = await experiment_engine.start_experiment(experiment_id)
        return result
        
    except Exception as e:
        logger.error(f"Error starting experiment {experiment_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{experiment_id}/stop", response_model=Dict[str, Any])
async def stop_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stop an A/B test experiment."""
    try:
        result = await experiment_engine.stop_experiment(experiment_id)
        return result
        
    except Exception as e:
        logger.error(f"Error stopping experiment {experiment_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{experiment_id}/pause", response_model=Dict[str, Any])
async def pause_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Pause an A/B test experiment."""
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        if experiment.status != ExperimentStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="Can only pause running experiments")
            
        experiment.status = ExperimentStatus.PAUSED.value
        db.commit()
        
        return {"experiment_id": experiment_id, "status": "paused"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/variants", response_model=List[Dict[str, Any]])
async def get_experiment_variants(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all variants for an experiment."""
    try:
        variants = db.query(ExperimentVariant).filter(
            ExperimentVariant.experiment_id == experiment_id
        ).all()
        
        return [variant.to_dict() for variant in variants]
        
    except Exception as e:
        logger.error(f"Error getting variants for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/variants/{variant_id}/metrics", response_model=Dict[str, Any])
async def record_variant_metric(
    experiment_id: str,
    variant_id: str,
    metric_data: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Record a metric value for an experiment variant."""
    try:
        # Verify variant exists and belongs to experiment
        variant = db.query(ExperimentVariant).filter(
            ExperimentVariant.id == variant_id,
            ExperimentVariant.experiment_id == experiment_id
        ).first()
        
        if not variant:
            raise HTTPException(status_code=404, detail="Variant not found")
            
        # Create metric record
        metric = VariantMetric(
            variant_id=variant_id,
            metric_name=metric_data["metric_name"],
            metric_value=metric_data["metric_value"],
            sample_size=metric_data.get("sample_size", 1),
            session_id=metric_data.get("session_id"),
            user_feedback=metric_data.get("user_feedback")
        )
        
        db.add(metric)
        db.commit()
        
        return {"status": "recorded", "metric_id": metric.id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording metric for variant {variant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/analysis", response_model=Dict[str, Any])
async def analyze_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform statistical analysis of experiment results."""
    try:
        results = await experiment_engine.analyze_experiment_results(experiment_id)
        return results.__dict__
        
    except Exception as e:
        logger.error(f"Error analyzing experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/promote-winner", response_model=Dict[str, Any])
async def promote_winner(
    experiment_id: str,
    winner_data: Optional[Dict[str, Any]] = None,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Promote the winning variant to production."""
    try:
        result = await experiment_engine.promote_winner(experiment_id)
        return result
        
    except Exception as e:
        logger.error(f"Error promoting winner for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{experiment_id}/schedule", response_model=Dict[str, Any])
async def schedule_experiment(
    experiment_id: str,
    schedule_config: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Schedule experiment automation."""
    try:
        schedule = ExperimentSchedule(
            experiment_id=experiment_id,
            schedule_type=schedule_config["schedule_type"],
            cron_expression=schedule_config.get("cron_expression"),
            auto_start=schedule_config.get("auto_start", False),
            auto_stop=schedule_config.get("auto_stop", False),
            auto_promote_winner=schedule_config.get("auto_promote_winner", False),
            promotion_threshold=schedule_config.get("promotion_threshold", 0.05),
            max_duration_hours=schedule_config.get("max_duration_hours"),
            created_by=current_user
        )
        
        db.add(schedule)
        db.commit()
        
        return schedule.to_dict()
        
    except Exception as e:
        logger.error(f"Error scheduling experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/status", response_model=Dict[str, Any])
async def get_experiment_status(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed experiment status and progress."""
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        # Get metrics count
        total_metrics = db.query(VariantMetric).join(ExperimentVariant).filter(
            ExperimentVariant.experiment_id == experiment_id
        ).count()
        
        # Calculate progress
        progress = min(total_metrics / experiment.target_sample_size, 1.0) if experiment.target_sample_size > 0 else 0
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "target_sample_size": experiment.target_sample_size,
            "total_metrics_collected": total_metrics,
            "progress": progress,
            "variants": [v.to_dict() for v in experiment.variants],
            "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
            "end_date": experiment.end_date.isoformat() if experiment.end_date else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-scheduled", response_model=List[Dict[str, Any]])
async def run_scheduled_experiments(
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run scheduled experiments (typically called by a cron job)."""
    try:
        # Get due schedules
        now = datetime.utcnow()
        due_schedules = db.query(ExperimentSchedule).filter(
            ExperimentSchedule.next_run <= now,
            ExperimentSchedule.is_active == True
        ).all()
        
        results = []
        for schedule in due_schedules:
            try:
                experiment = schedule.experiment
                actions_taken = []
                
                # Auto-start if configured
                if schedule.auto_start and experiment.status == ExperimentStatus.DRAFT.value:
                    await experiment_engine.start_experiment(experiment.id)
                    actions_taken.append("started")
                
                # Auto-stop if configured and duration exceeded
                if (schedule.auto_stop and schedule.max_duration_hours and 
                    experiment.status == ExperimentStatus.RUNNING.value and
                    experiment.start_date):
                    
                    duration = now - experiment.start_date
                    if duration.total_seconds() / 3600 >= schedule.max_duration_hours:
                        await experiment_engine.stop_experiment(experiment.id)
                        actions_taken.append("stopped")
                
                # Auto-promote winner if configured
                if (schedule.auto_promote_winner and 
                    experiment.status == ExperimentStatus.COMPLETED.value):
                    
                    result = await experiment_engine.promote_winner(experiment.id)
                    if result.get("status") == "promoted":
                        actions_taken.append("promoted_winner")
                
                # Update schedule
                schedule.last_run = now
                # Calculate next run based on schedule type
                if schedule.schedule_type == "daily":
                    from datetime import timedelta
                    schedule.next_run = now + timedelta(days=1)
                elif schedule.schedule_type == "weekly":
                    from datetime import timedelta
                    schedule.next_run = now + timedelta(weeks=1)
                
                results.append({
                    "experiment_id": experiment.id,
                    "schedule_id": schedule.id,
                    "actions_taken": actions_taken
                })
                
            except Exception as e:
                logger.error(f"Error processing schedule {schedule.id}: {e}")
                results.append({
                    "experiment_id": schedule.experiment_id,
                    "schedule_id": schedule.id,
                    "error": str(e)
                })
        
        db.commit()
        return results
        
    except Exception as e:
        logger.error(f"Error running scheduled experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{experiment_id}", response_model=Dict[str, Any])
async def delete_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an experiment and all its data."""
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        # Can only delete draft or completed experiments
        if experiment.status in [ExperimentStatus.RUNNING.value, ExperimentStatus.PAUSED.value]:
            raise HTTPException(status_code=400, detail="Cannot delete running or paused experiments")
            
        db.delete(experiment)
        db.commit()
        
        return {"status": "deleted", "experiment_id": experiment_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))