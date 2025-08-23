"""
Continuous Improvement API Routes

This module provides REST API endpoints for the continuous improvement system,
including feedback collection, A/B testing, model retraining, and feature enhancement.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ...core.database import get_db_session
from ...core.auth import get_current_user, require_permissions
from ...engines.continuous_improvement_engine import ContinuousImprovementEngine
from ...models.continuous_improvement_models import (
    FeedbackCreate, FeedbackResponse, ABTestCreate, ABTestResponse,
    ModelRetrainingCreate, ModelRetrainingResponse, FeatureEnhancementCreate,
    FeatureEnhancementResponse, ImprovementMetrics, ImprovementRecommendation,
    UserFeedback, ABTest, ModelRetrainingJob, FeatureEnhancement,
    FeedbackType, FeedbackPriority, ABTestStatus, ModelRetrainingStatus,
    FeatureEnhancementStatus
)

router = APIRouter(prefix="/api/v1/continuous-improvement", tags=["continuous-improvement"])

# Initialize improvement engine
improvement_engine = ContinuousImprovementEngine()

@router.post("/feedback", response_model=FeedbackResponse)
async def create_feedback(
    feedback: FeedbackCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Create new user feedback for continuous improvement.
    
    This endpoint collects real user feedback and automatically triggers
    improvement analysis and recommendations.
    """
    try:
        feedback_data = feedback.dict()
        
        result = await improvement_engine.collect_user_feedback(
            user_id=current_user["user_id"],
            feedback_data=feedback_data,
            db=db
        )
        
        return FeedbackResponse.from_orm(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback", response_model=List[FeedbackResponse])
async def get_feedback(
    feedback_type: Optional[FeedbackType] = None,
    priority: Optional[FeedbackPriority] = None,
    feature_area: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get feedback records with filtering and pagination.
    """
    try:
        query = db.query(UserFeedback)
        
        # Apply filters
        if feedback_type:
            query = query.filter(UserFeedback.feedback_type == feedback_type)
        
        if priority:
            query = query.filter(UserFeedback.priority == priority)
        
        if feature_area:
            query = query.filter(UserFeedback.feature_area == feature_area)
        
        if start_date:
            query = query.filter(UserFeedback.created_at >= start_date)
        
        if end_date:
            query = query.filter(UserFeedback.created_at <= end_date)
        
        # Apply pagination and ordering
        feedback_records = query.order_by(desc(UserFeedback.created_at)).offset(offset).limit(limit).all()
        
        return [FeedbackResponse.from_orm(record) for record in feedback_records]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests", response_model=ABTestResponse)
async def create_ab_test(
    test_config: ABTestCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_permissions(["admin", "data_scientist"])),
    db: Session = Depends(get_db_session)
):
    """
    Create new A/B test for system improvements.
    
    This endpoint creates A/B tests to validate system enhancements
    and measure their impact on business metrics.
    """
    try:
        test_data = test_config.dict()
        
        result = await improvement_engine.create_ab_test(
            test_config=test_data,
            db=db
        )
        
        return ABTestResponse.from_orm(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests/{test_id}/start")
async def start_ab_test(
    test_id: int,
    current_user: dict = Depends(require_permissions(["admin", "data_scientist"])),
    db: Session = Depends(get_db_session)
):
    """
    Start an A/B test and begin collecting results.
    """
    try:
        success = await improvement_engine.start_ab_test(test_id, db)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start A/B test")
        
        return {"message": f"A/B test {test_id} started successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests/{test_id}/results")
async def record_ab_test_result(
    test_id: int,
    variant_name: str,
    metrics: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Record A/B test result for a user interaction.
    """
    try:
        result = await improvement_engine.record_ab_test_result(
            test_id=test_id,
            user_id=current_user["user_id"],
            variant_name=variant_name,
            metrics=metrics,
            db=db
        )
        
        return {"message": "A/B test result recorded successfully", "result_id": result.id}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests/{test_id}/analysis")
async def analyze_ab_test(
    test_id: int,
    current_user: dict = Depends(require_permissions(["admin", "data_scientist"])),
    db: Session = Depends(get_db_session)
):
    """
    Analyze A/B test results and get statistical significance.
    """
    try:
        analysis = await improvement_engine.analyze_ab_test_results(test_id, db)
        return analysis
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests", response_model=List[ABTestResponse])
async def get_ab_tests(
    status: Optional[ABTestStatus] = None,
    feature_area: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get A/B tests with filtering and pagination.
    """
    try:
        query = db.query(ABTest)
        
        if status:
            query = query.filter(ABTest.status == status)
        
        if feature_area:
            query = query.filter(ABTest.feature_area == feature_area)
        
        tests = query.order_by(desc(ABTest.created_at)).offset(offset).limit(limit).all()
        
        return [ABTestResponse.from_orm(test) for test in tests]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))@router.
post("/model-retraining", response_model=ModelRetrainingResponse)
async def schedule_model_retraining(
    retraining_config: ModelRetrainingCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_permissions(["admin", "ml_engineer"])),
    db: Session = Depends(get_db_session)
):
    """
    Schedule model retraining based on business outcomes.
    
    This endpoint schedules ML model retraining using real business feedback
    and performance data to improve model accuracy and business impact.
    """
    try:
        config_data = retraining_config.dict()
        
        result = await improvement_engine.schedule_model_retraining(
            model_config=config_data,
            db=db
        )
        
        # Schedule background execution
        background_tasks.add_task(
            improvement_engine.execute_model_retraining,
            result.id,
            db
        )
        
        return ModelRetrainingResponse.from_orm(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-retraining/{job_id}/status")
async def get_retraining_status(
    job_id: int,
    current_user: dict = Depends(require_permissions(["admin", "ml_engineer"])),
    db: Session = Depends(get_db_session)
):
    """
    Get model retraining job status and progress.
    """
    try:
        job = db.query(ModelRetrainingJob).filter(ModelRetrainingJob.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Retraining job not found")
        
        return ModelRetrainingResponse.from_orm(job)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-retraining", response_model=List[ModelRetrainingResponse])
async def get_retraining_jobs(
    model_name: Optional[str] = None,
    status: Optional[ModelRetrainingStatus] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(require_permissions(["admin", "ml_engineer"])),
    db: Session = Depends(get_db_session)
):
    """
    Get model retraining jobs with filtering and pagination.
    """
    try:
        query = db.query(ModelRetrainingJob)
        
        if model_name:
            query = query.filter(ModelRetrainingJob.model_name == model_name)
        
        if status:
            query = query.filter(ModelRetrainingJob.status == status)
        
        jobs = query.order_by(desc(ModelRetrainingJob.created_at)).offset(offset).limit(limit).all()
        
        return [ModelRetrainingResponse.from_orm(job) for job in jobs]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature-enhancements", response_model=FeatureEnhancementResponse)
async def create_feature_enhancement(
    enhancement: FeatureEnhancementCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Create feature enhancement request based on user requirements.
    
    This endpoint allows users to submit feature enhancement requests
    that are automatically prioritized based on business value and impact.
    """
    try:
        enhancement_data = enhancement.dict()
        
        result = await improvement_engine.create_feature_enhancement(
            requester_id=current_user["user_id"],
            enhancement_data=enhancement_data,
            db=db
        )
        
        return FeatureEnhancementResponse.from_orm(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-enhancements", response_model=List[FeatureEnhancementResponse])
async def get_feature_enhancements(
    status: Optional[FeatureEnhancementStatus] = None,
    feature_area: Optional[str] = None,
    priority: Optional[FeedbackPriority] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get feature enhancement requests with filtering and pagination.
    """
    try:
        query = db.query(FeatureEnhancement)
        
        if status:
            query = query.filter(FeatureEnhancement.status == status)
        
        if feature_area:
            query = query.filter(FeatureEnhancement.feature_area == feature_area)
        
        if priority:
            query = query.filter(FeatureEnhancement.priority == priority)
        
        enhancements = query.order_by(desc(FeatureEnhancement.created_at)).offset(offset).limit(limit).all()
        
        return [FeatureEnhancementResponse.from_orm(enhancement) for enhancement in enhancements]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/feature-enhancements/{enhancement_id}/status")
async def update_enhancement_status(
    enhancement_id: int,
    status: FeatureEnhancementStatus,
    notes: Optional[str] = None,
    current_user: dict = Depends(require_permissions(["admin", "product_manager"])),
    db: Session = Depends(get_db_session)
):
    """
    Update feature enhancement status and progress.
    """
    try:
        enhancement = db.query(FeatureEnhancement).filter(
            FeatureEnhancement.id == enhancement_id
        ).first()
        
        if not enhancement:
            raise HTTPException(status_code=404, detail="Feature enhancement not found")
        
        # Update status and timestamp
        enhancement.status = status
        enhancement.updated_at = datetime.utcnow()
        
        if status == FeatureEnhancementStatus.APPROVED:
            enhancement.approved_at = datetime.utcnow()
        elif status == FeatureEnhancementStatus.IN_DEVELOPMENT:
            enhancement.development_started_at = datetime.utcnow()
        elif status == FeatureEnhancementStatus.DEPLOYED:
            enhancement.deployed_at = datetime.utcnow()
        
        if notes:
            enhancement.implementation_notes = notes
        
        db.commit()
        
        return {"message": f"Enhancement {enhancement_id} status updated to {status}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations", response_model=List[ImprovementRecommendation])
async def get_improvement_recommendations(
    time_window_days: int = Query(default=30, ge=1, le=365),
    current_user: dict = Depends(require_permissions(["admin", "product_manager"])),
    db: Session = Depends(get_db_session)
):
    """
    Get data-driven improvement recommendations based on real business outcomes.
    
    This endpoint analyzes feedback, A/B test results, model performance,
    and feature usage to generate actionable improvement recommendations.
    """
    try:
        recommendations = await improvement_engine.generate_improvement_recommendations(
            db=db,
            time_window_days=time_window_days
        )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=ImprovementMetrics)
async def get_improvement_metrics(
    time_window_days: int = Query(default=30, ge=1, le=365),
    current_user: dict = Depends(require_permissions(["admin", "analyst"])),
    db: Session = Depends(get_db_session)
):
    """
    Get comprehensive improvement metrics and analytics.
    
    This endpoint provides detailed metrics on feedback trends, A/B test results,
    model performance, feature adoption, and business impact.
    """
    try:
        metrics = await improvement_engine.get_improvement_metrics(
            db=db,
            time_window_days=time_window_days
        )
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_improvement_dashboard(
    time_window_days: int = Query(default=30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get improvement dashboard data with key metrics and trends.
    """
    try:
        # Get summary metrics
        metrics = await improvement_engine.get_improvement_metrics(db, time_window_days)
        
        # Get recent feedback
        recent_feedback = db.query(UserFeedback).filter(
            UserFeedback.created_at >= datetime.utcnow() - timedelta(days=7)
        ).order_by(desc(UserFeedback.created_at)).limit(10).all()
        
        # Get active A/B tests
        active_tests = db.query(ABTest).filter(
            ABTest.status == ABTestStatus.RUNNING
        ).count()
        
        # Get recent enhancements
        recent_enhancements = db.query(FeatureEnhancement).filter(
            FeatureEnhancement.created_at >= datetime.utcnow() - timedelta(days=30)
        ).count()
        
        return {
            "metrics": metrics,
            "recent_feedback_count": len(recent_feedback),
            "active_ab_tests": active_tests,
            "recent_enhancements": recent_enhancements,
            "time_window_days": time_window_days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback/{feedback_id}/resolve")
async def resolve_feedback(
    feedback_id: int,
    resolution_notes: str,
    current_user: dict = Depends(require_permissions(["admin", "support"])),
    db: Session = Depends(get_db_session)
):
    """
    Mark feedback as resolved with resolution notes.
    """
    try:
        feedback = db.query(UserFeedback).filter(UserFeedback.id == feedback_id).first()
        
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        feedback.resolved_at = datetime.utcnow()
        feedback.resolution_notes = resolution_notes
        feedback.updated_at = datetime.utcnow()
        
        db.commit()
        
        return {"message": f"Feedback {feedback_id} resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))