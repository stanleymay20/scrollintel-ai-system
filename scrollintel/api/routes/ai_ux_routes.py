"""
AI UX Optimization API Routes

This module provides REST API endpoints for AI-powered user experience optimization,
including failure prediction, user behavior analysis, personalized degradation,
and interface optimization.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field

from scrollintel.engines.ai_ux_optimizer import AIUXOptimizer
from scrollintel.models.ai_ux_models import (
    UserInteractionModel, SystemMetricsModel, UserFeedbackModel,
    create_user_interaction_from_dict, create_system_metrics_from_dict,
    aggregate_user_interactions
)
from scrollintel.core.config import get_database_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ai-ux", tags=["AI UX Optimization"])

# Initialize AI UX Optimizer
ai_ux_optimizer = AIUXOptimizer()

# Pydantic models for request/response validation

class UserInteractionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    action_type: str = Field(..., description="Type of action performed")
    feature_used: Optional[str] = Field(None, description="Feature that was used")
    page_visited: Optional[str] = Field(None, description="Page that was visited")
    duration: Optional[float] = Field(None, description="Duration of action in seconds")
    success: bool = Field(True, description="Whether the action was successful")
    error_encountered: Optional[str] = Field(None, description="Error message if any")
    help_requested: bool = Field(False, description="Whether help was requested")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SystemMetricsRequest(BaseModel):
    cpu_usage: Optional[float] = Field(None, ge=0, le=1, description="CPU usage (0-1)")
    memory_usage: Optional[float] = Field(None, ge=0, le=1, description="Memory usage (0-1)")
    disk_usage: Optional[float] = Field(None, ge=0, le=1, description="Disk usage (0-1)")
    network_latency: Optional[float] = Field(None, ge=0, description="Network latency in ms")
    error_rate: Optional[float] = Field(None, ge=0, le=1, description="Error rate (0-1)")
    response_time: Optional[float] = Field(None, ge=0, description="Response time in ms")
    active_users: Optional[int] = Field(None, ge=0, description="Number of active users")
    request_rate: Optional[float] = Field(None, ge=0, description="Requests per second")
    system_load: Optional[float] = Field(None, ge=0, description="Overall system load")
    additional_metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")

class UserFeedbackRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    optimization_type: str = Field(..., description="Type of optimization")
    satisfaction_score: float = Field(..., ge=1, le=5, description="Satisfaction score (1-5)")
    feedback_text: Optional[str] = Field(None, description="Feedback text")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    would_recommend: bool = Field(True, description="Would recommend to others")

class FailurePredictionResponse(BaseModel):
    prediction_type: str
    probability: float
    confidence: float
    time_to_failure: Optional[int]
    contributing_factors: List[str]
    recommended_actions: List[str]
    timestamp: datetime

class UserBehaviorResponse(BaseModel):
    user_id: str
    behavior_pattern: str
    engagement_score: float
    frustration_indicators: List[str]
    preferred_features: List[str]
    usage_patterns: Dict[str, Any]
    assistance_needs: List[str]
    timestamp: datetime

class PersonalizedDegradationResponse(BaseModel):
    user_id: str
    strategy: str
    feature_priorities: Dict[str, int]
    acceptable_delays: Dict[str, float]
    fallback_preferences: Dict[str, str]
    communication_style: str
    timestamp: datetime

class InterfaceOptimizationResponse(BaseModel):
    user_id: str
    layout_preferences: Dict[str, Any]
    interaction_patterns: Dict[str, float]
    performance_requirements: Dict[str, float]
    accessibility_needs: List[str]
    optimization_suggestions: List[str]
    timestamp: datetime

# API Endpoints

@router.post("/interactions", response_model=Dict[str, str])
async def record_user_interaction(
    interaction: UserInteractionRequest,
    background_tasks: BackgroundTasks,
    db_session=Depends(get_database_session)
):
    """Record a user interaction for behavior analysis"""
    try:
        # Store interaction in database
        interaction_model = create_user_interaction_from_dict(interaction.dict())
        db_session.add(interaction_model)
        db_session.commit()
        
        # Trigger behavior analysis in background
        background_tasks.add_task(
            analyze_user_behavior_background,
            interaction.user_id,
            interaction.dict()
        )
        
        return {"status": "success", "message": "Interaction recorded"}
        
    except Exception as e:
        logger.error(f"Error recording user interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to record interaction")

@router.post("/system-metrics", response_model=Dict[str, str])
async def record_system_metrics(
    metrics: SystemMetricsRequest,
    background_tasks: BackgroundTasks,
    db_session=Depends(get_database_session)
):
    """Record system metrics for failure prediction"""
    try:
        # Store metrics in database
        metrics_model = create_system_metrics_from_dict(metrics.dict())
        db_session.add(metrics_model)
        db_session.commit()
        
        # Trigger failure prediction in background
        background_tasks.add_task(
            predict_failures_background,
            metrics.dict()
        )
        
        return {"status": "success", "message": "Metrics recorded"}
        
    except Exception as e:
        logger.error(f"Error recording system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to record metrics")

@router.get("/failure-predictions", response_model=List[FailurePredictionResponse])
async def get_failure_predictions(
    system_metrics: Optional[Dict[str, Any]] = None
):
    """Get failure predictions based on current system metrics"""
    try:
        if system_metrics is None:
            # Use default/current metrics
            system_metrics = {
                'cpu_usage': 0.5,
                'memory_usage': 0.6,
                'disk_usage': 0.3,
                'network_latency': 100,
                'error_rate': 0.01,
                'response_time': 500,
                'active_users': 100,
                'request_rate': 10
            }
        
        predictions = await ai_ux_optimizer.predict_failures(system_metrics)
        
        return [
            FailurePredictionResponse(
                prediction_type=p.prediction_type.value,
                probability=p.probability,
                confidence=p.confidence,
                time_to_failure=p.time_to_failure,
                contributing_factors=p.contributing_factors,
                recommended_actions=p.recommended_actions,
                timestamp=p.timestamp
            )
            for p in predictions
        ]
        
    except Exception as e:
        logger.error(f"Error getting failure predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions")

@router.get("/user-behavior/{user_id}", response_model=UserBehaviorResponse)
async def get_user_behavior_analysis(user_id: str):
    """Get behavior analysis for a specific user"""
    try:
        # Get user profile from optimizer
        user_profile = ai_ux_optimizer.user_profiles.get(user_id)
        
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return UserBehaviorResponse(
            user_id=user_profile.user_id,
            behavior_pattern=user_profile.behavior_pattern.value,
            engagement_score=user_profile.engagement_score,
            frustration_indicators=user_profile.frustration_indicators,
            preferred_features=user_profile.preferred_features,
            usage_patterns=user_profile.usage_patterns,
            assistance_needs=user_profile.assistance_needs,
            timestamp=user_profile.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user behavior analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get behavior analysis")

@router.post("/user-behavior/{user_id}/analyze", response_model=UserBehaviorResponse)
async def analyze_user_behavior(
    user_id: str,
    interaction_data: Dict[str, Any]
):
    """Analyze user behavior based on interaction data"""
    try:
        analysis = await ai_ux_optimizer.analyze_user_behavior(user_id, interaction_data)
        
        return UserBehaviorResponse(
            user_id=analysis.user_id,
            behavior_pattern=analysis.behavior_pattern.value,
            engagement_score=analysis.engagement_score,
            frustration_indicators=analysis.frustration_indicators,
            preferred_features=analysis.preferred_features,
            usage_patterns=analysis.usage_patterns,
            assistance_needs=analysis.assistance_needs,
            timestamp=analysis.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error analyzing user behavior: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze behavior")

@router.get("/personalized-degradation/{user_id}", response_model=PersonalizedDegradationResponse)
async def get_personalized_degradation(
    user_id: str,
    system_conditions: Optional[Dict[str, Any]] = None
):
    """Get personalized degradation strategy for a user"""
    try:
        if system_conditions is None:
            system_conditions = {'system_load': 0.5}
        
        degradation = await ai_ux_optimizer.create_personalized_degradation(
            user_id, system_conditions
        )
        
        return PersonalizedDegradationResponse(
            user_id=degradation.user_id,
            strategy=degradation.strategy.value,
            feature_priorities=degradation.feature_priorities,
            acceptable_delays=degradation.acceptable_delays,
            fallback_preferences=degradation.fallback_preferences,
            communication_style=degradation.communication_style,
            timestamp=degradation.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error getting personalized degradation: {e}")
        raise HTTPException(status_code=500, detail="Failed to get degradation strategy")

@router.get("/interface-optimization/{user_id}", response_model=InterfaceOptimizationResponse)
async def get_interface_optimization(
    user_id: str,
    current_interface: Optional[Dict[str, Any]] = None
):
    """Get interface optimization recommendations for a user"""
    try:
        if current_interface is None:
            current_interface = {}
        
        optimization = await ai_ux_optimizer.optimize_interface(user_id, current_interface)
        
        return InterfaceOptimizationResponse(
            user_id=optimization.user_id,
            layout_preferences=optimization.layout_preferences,
            interaction_patterns=optimization.interaction_patterns,
            performance_requirements=optimization.performance_requirements,
            accessibility_needs=optimization.accessibility_needs,
            optimization_suggestions=optimization.optimization_suggestions,
            timestamp=optimization.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error getting interface optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to get optimization")

@router.post("/feedback", response_model=Dict[str, str])
async def submit_user_feedback(
    feedback: UserFeedbackRequest,
    db_session=Depends(get_database_session)
):
    """Submit user feedback on AI UX optimizations"""
    try:
        # Store feedback in database
        feedback_model = UserFeedbackModel(
            user_id=feedback.user_id,
            optimization_type=feedback.optimization_type,
            satisfaction_score=feedback.satisfaction_score,
            feedback_text=feedback.feedback_text,
            improvement_suggestions=feedback.improvement_suggestions,
            would_recommend=feedback.would_recommend
        )
        
        db_session.add(feedback_model)
        db_session.commit()
        
        return {"status": "success", "message": "Feedback submitted"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_optimization_metrics():
    """Get AI UX optimization performance metrics"""
    try:
        metrics = await ai_ux_optimizer.get_optimization_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting optimization metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@router.post("/train-models", response_model=Dict[str, str])
async def train_ai_models(
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Train AI models with new data"""
    try:
        # Trigger model training in background
        background_tasks.add_task(
            train_models_background,
            training_data
        )
        
        return {"status": "success", "message": "Model training started"}
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start training")

@router.get("/user-sessions/{user_id}", response_model=Dict[str, Any])
async def get_user_session_analysis(
    user_id: str,
    days: int = 7,
    db_session=Depends(get_database_session)
):
    """Get user session analysis for the past N days"""
    try:
        # Get user interactions from database
        since_date = datetime.utcnow() - timedelta(days=days)
        
        interactions = db_session.query(UserInteractionModel).filter(
            UserInteractionModel.user_id == user_id,
            UserInteractionModel.timestamp >= since_date
        ).all()
        
        if not interactions:
            return {"message": "No interactions found for user"}
        
        # Aggregate interactions by session
        sessions = {}
        for interaction in interactions:
            session_id = interaction.session_id
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(interaction)
        
        # Analyze each session
        session_analyses = []
        for session_id, session_interactions in sessions.items():
            analysis = aggregate_user_interactions(session_interactions)
            analysis['session_id'] = session_id
            analysis['start_time'] = min(i.timestamp for i in session_interactions).isoformat()
            analysis['end_time'] = max(i.timestamp for i in session_interactions).isoformat()
            session_analyses.append(analysis)
        
        # Overall statistics
        total_interactions = len(interactions)
        total_sessions = len(sessions)
        avg_session_duration = sum(s['session_duration'] for s in session_analyses) / len(session_analyses)
        overall_success_rate = sum(1 for i in interactions if i.success) / total_interactions
        
        return {
            'user_id': user_id,
            'analysis_period_days': days,
            'total_interactions': total_interactions,
            'total_sessions': total_sessions,
            'average_session_duration': avg_session_duration,
            'overall_success_rate': overall_success_rate,
            'sessions': session_analyses
        }
        
    except Exception as e:
        logger.error(f"Error getting user session analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session analysis")

# Background task functions

async def analyze_user_behavior_background(user_id: str, interaction_data: Dict[str, Any]):
    """Background task to analyze user behavior"""
    try:
        await ai_ux_optimizer.analyze_user_behavior(user_id, interaction_data)
        logger.info(f"Completed behavior analysis for user {user_id}")
    except Exception as e:
        logger.error(f"Error in background behavior analysis: {e}")

async def predict_failures_background(system_metrics: Dict[str, Any]):
    """Background task to predict failures"""
    try:
        predictions = await ai_ux_optimizer.predict_failures(system_metrics)
        if predictions:
            logger.info(f"Generated {len(predictions)} failure predictions")
    except Exception as e:
        logger.error(f"Error in background failure prediction: {e}")

async def train_models_background(training_data: Dict[str, Any]):
    """Background task to train AI models"""
    try:
        await ai_ux_optimizer.train_models(training_data)
        logger.info("Completed AI model training")
    except Exception as e:
        logger.error(f"Error in background model training: {e}")

# Health check endpoint
@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check for AI UX optimization service"""
    try:
        # Check if optimizer is initialized
        if ai_ux_optimizer.failure_predictor is None:
            return {"status": "initializing", "message": "AI models are still initializing"}
        
        return {"status": "healthy", "message": "AI UX optimization service is running"}
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}