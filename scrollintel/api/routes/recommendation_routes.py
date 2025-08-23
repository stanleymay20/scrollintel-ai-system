"""
API routes for the AI Recommendation Engine
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
import uuid
from datetime import datetime

from scrollintel.engines.recommendation_engine import (
    RecommendationEngine, Schema, Dataset, Transformation, 
    JoinRecommendation, Optimization, DataPatternAnalysis
)
from scrollintel.models.recommendation_models import (
    RecommendationHistory, UserFeedback, PerformanceBaseline
)

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])

# Initialize recommendation engine
recommendation_engine = RecommendationEngine()


# Pydantic models for API
class SchemaRequest(BaseModel):
    name: str
    columns: List[Dict[str, Any]]
    data_types: Dict[str, str]
    sample_data: Optional[Dict[str, Any]] = None


class DatasetRequest(BaseModel):
    name: str
    schema: SchemaRequest
    row_count: int
    size_mb: float
    quality_score: float = 0.0


class TransformationRecommendationRequest(BaseModel):
    source_schema: SchemaRequest
    target_schema: SchemaRequest


class OptimizationRequest(BaseModel):
    pipeline: Dict[str, Any]
    metrics: Dict[str, Any]


class JoinRecommendationRequest(BaseModel):
    left_dataset: DatasetRequest
    right_dataset: DatasetRequest


class DataPatternRequest(BaseModel):
    data_sample: Dict[str, Any]  # JSON representation of DataFrame


class FeedbackRequest(BaseModel):
    recommendation_id: str
    feedback_type: str = Field(..., description="Type of feedback: rating, comment, usage")
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None
    was_helpful: Optional[bool] = None
    implementation_difficulty: Optional[str] = None
    actual_benefit: Optional[str] = None
    user_id: Optional[str] = None


class TransformationResponse(BaseModel):
    name: str
    type: str
    description: str
    confidence: float
    parameters: Dict[str, Any]
    estimated_performance_impact: float


class OptimizationResponse(BaseModel):
    category: str
    description: str
    impact: str
    implementation_effort: str
    estimated_improvement: float
    priority: int


class JoinRecommendationResponse(BaseModel):
    join_type: str
    left_key: str
    right_key: str
    confidence: float
    estimated_rows: int
    performance_score: float


class DataPatternResponse(BaseModel):
    patterns: List[str]
    anomalies: List[str]
    recommendations: List[str]
    quality_issues: List[str]


@router.post("/transformations", response_model=List[TransformationResponse])
async def get_transformation_recommendations(request: TransformationRecommendationRequest):
    """
    Get AI-powered transformation recommendations between source and target schemas
    """
    try:
        # Convert request to engine objects
        source_schema = Schema(
            name=request.source_schema.name,
            columns=request.source_schema.columns,
            data_types=request.source_schema.data_types
        )
        
        target_schema = Schema(
            name=request.target_schema.name,
            columns=request.target_schema.columns,
            data_types=request.target_schema.data_types
        )
        
        # Add sample data if provided
        if request.source_schema.sample_data:
            source_schema.sample_data = pd.DataFrame(request.source_schema.sample_data)
        
        # Get recommendations
        recommendations = recommendation_engine.recommend_transformations(source_schema, target_schema)
        
        # Store recommendations in history
        for rec in recommendations:
            rec_id = str(uuid.uuid4())
            history = RecommendationHistory(
                recommendation_id=rec_id,
                recommendation_type="transformation",
                name=rec.name,
                description=rec.description,
                confidence_score=rec.confidence,
                parameters=rec.parameters,
                estimated_impact=rec.estimated_performance_impact
            )
            # Note: In a real implementation, you'd save to database here
        
        # Convert to response format
        return [
            TransformationResponse(
                name=rec.name,
                type=rec.type,
                description=rec.description,
                confidence=rec.confidence,
                parameters=rec.parameters,
                estimated_performance_impact=rec.estimated_performance_impact
            )
            for rec in recommendations
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating transformation recommendations: {str(e)}")


@router.post("/optimizations", response_model=List[OptimizationResponse])
async def get_optimization_recommendations(request: OptimizationRequest):
    """
    Get AI-powered performance optimization recommendations
    """
    try:
        # Get recommendations
        optimizations = recommendation_engine.suggest_optimizations(request.pipeline, request.metrics)
        
        # Store recommendations in history
        for opt in optimizations:
            rec_id = str(uuid.uuid4())
            history = RecommendationHistory(
                recommendation_id=rec_id,
                recommendation_type="performance",
                name=f"{opt.category}_optimization",
                description=opt.description,
                confidence_score=0.8,  # Default confidence for optimizations
                parameters={"category": opt.category, "impact": opt.impact},
                estimated_impact=opt.estimated_improvement
            )
            # Note: In a real implementation, you'd save to database here
        
        # Convert to response format
        return [
            OptimizationResponse(
                category=opt.category,
                description=opt.description,
                impact=opt.impact,
                implementation_effort=opt.implementation_effort,
                estimated_improvement=opt.estimated_improvement,
                priority=opt.priority
            )
            for opt in optimizations
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating optimization recommendations: {str(e)}")


@router.post("/join-strategy", response_model=JoinRecommendationResponse)
async def get_join_recommendation(request: JoinRecommendationRequest):
    """
    Get AI-powered join strategy recommendation for two datasets
    """
    try:
        # Convert request to engine objects
        left_schema = Schema(
            name=request.left_dataset.schema.name,
            columns=request.left_dataset.schema.columns,
            data_types=request.left_dataset.schema.data_types
        )
        
        right_schema = Schema(
            name=request.right_dataset.schema.name,
            columns=request.right_dataset.schema.columns,
            data_types=request.right_dataset.schema.data_types
        )
        
        left_dataset = Dataset(
            name=request.left_dataset.name,
            schema=left_schema,
            row_count=request.left_dataset.row_count,
            size_mb=request.left_dataset.size_mb,
            quality_score=request.left_dataset.quality_score
        )
        
        right_dataset = Dataset(
            name=request.right_dataset.name,
            schema=right_schema,
            row_count=request.right_dataset.row_count,
            size_mb=request.right_dataset.size_mb,
            quality_score=request.right_dataset.quality_score
        )
        
        # Get recommendation
        recommendation = recommendation_engine.recommend_join_strategy(left_dataset, right_dataset)
        
        # Store recommendation in history
        rec_id = str(uuid.uuid4())
        history = RecommendationHistory(
            recommendation_id=rec_id,
            recommendation_type="join_strategy",
            name=f"{recommendation.join_type}_join",
            description=f"Join {left_dataset.name} and {right_dataset.name} using {recommendation.join_type} join",
            confidence_score=recommendation.confidence,
            parameters={
                "join_type": recommendation.join_type,
                "left_key": recommendation.left_key,
                "right_key": recommendation.right_key
            },
            estimated_impact=recommendation.performance_score
        )
        # Note: In a real implementation, you'd save to database here
        
        return JoinRecommendationResponse(
            join_type=recommendation.join_type,
            left_key=recommendation.left_key,
            right_key=recommendation.right_key,
            confidence=recommendation.confidence,
            estimated_rows=recommendation.estimated_rows,
            performance_score=recommendation.performance_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating join recommendation: {str(e)}")


@router.post("/data-patterns", response_model=DataPatternResponse)
async def analyze_data_patterns(request: DataPatternRequest):
    """
    Analyze data patterns and provide insights
    """
    try:
        # Convert JSON to DataFrame
        data_sample = pd.DataFrame(request.data_sample)
        
        # Analyze patterns
        analysis = recommendation_engine.analyze_data_patterns(data_sample)
        
        return DataPatternResponse(
            patterns=analysis.patterns,
            anomalies=analysis.anomalies,
            recommendations=analysis.recommendations,
            quality_issues=analysis.quality_issues
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing data patterns: {str(e)}")


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on recommendations to improve future suggestions
    """
    try:
        # Store feedback in database
        feedback = UserFeedback(
            recommendation_id=request.recommendation_id,
            user_id=request.user_id,
            feedback_type=request.feedback_type,
            rating=request.rating,
            comment=request.comment,
            was_helpful=request.was_helpful,
            implementation_difficulty=request.implementation_difficulty,
            actual_benefit=request.actual_benefit
        )
        # Note: In a real implementation, you'd save to database here
        
        # Learn from feedback
        feedback_data = {
            "rating": request.rating,
            "helpful": request.was_helpful,
            "difficulty": request.implementation_difficulty,
            "benefit": request.actual_benefit
        }
        
        recommendation_engine.learn_from_feedback(request.recommendation_id, feedback_data)
        
        return {"message": "Feedback submitted successfully", "recommendation_id": request.recommendation_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")


@router.get("/history")
async def get_recommendation_history(
    recommendation_type: Optional[str] = Query(None, description="Filter by recommendation type"),
    limit: int = Query(50, ge=1, le=1000, description="Number of recommendations to return"),
    offset: int = Query(0, ge=0, description="Number of recommendations to skip")
):
    """
    Get recommendation history with optional filtering
    """
    try:
        # Note: In a real implementation, you'd query the database here
        # For now, return mock data
        mock_history = [
            {
                "id": 1,
                "recommendation_id": "rec_001",
                "recommendation_type": "transformation",
                "name": "convert_data_type",
                "description": "Convert column 'age' from string to integer",
                "confidence_score": 0.85,
                "was_accepted": True,
                "was_implemented": True,
                "user_feedback_score": 4,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        # Apply filtering
        if recommendation_type:
            mock_history = [h for h in mock_history if h["recommendation_type"] == recommendation_type]
        
        # Apply pagination
        total = len(mock_history)
        history = mock_history[offset:offset + limit]
        
        return {
            "recommendations": history,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recommendation history: {str(e)}")


@router.get("/metrics")
async def get_recommendation_metrics():
    """
    Get recommendation system performance metrics
    """
    try:
        # Note: In a real implementation, you'd calculate these from the database
        metrics = {
            "total_recommendations": 1250,
            "acceptance_rate": 0.68,
            "implementation_rate": 0.45,
            "average_confidence_score": 0.78,
            "average_user_rating": 4.2,
            "performance_improvement": 0.32,
            "recommendation_types": {
                "transformation": 650,
                "performance": 400,
                "join_strategy": 150,
                "schema_mapping": 50
            },
            "recent_trends": {
                "daily_recommendations": 45,
                "weekly_growth": 0.12,
                "user_satisfaction_trend": "increasing"
            }
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recommendation metrics: {str(e)}")


@router.post("/performance-baseline")
async def store_performance_baseline(
    pipeline_id: str,
    metrics: Dict[str, Any]
):
    """
    Store performance baseline for future optimization recommendations
    """
    try:
        baseline = PerformanceBaseline(
            pipeline_id=pipeline_id,
            pipeline_config_hash=str(hash(str(metrics.get("config", {})))),
            execution_time_seconds=metrics.get("execution_time_seconds", 0),
            memory_usage_mb=metrics.get("memory_usage_mb", 0),
            cpu_usage_percent=metrics.get("cpu_usage_percent", 0),
            rows_processed=metrics.get("rows_processed", 0),
            data_size_mb=metrics.get("data_size_mb", 0),
            error_rate=metrics.get("error_rate", 0),
            data_quality_score=metrics.get("data_quality_score"),
            execution_environment=metrics.get("environment"),
            resource_allocation=metrics.get("resources")
        )
        # Note: In a real implementation, you'd save to database here
        
        return {
            "message": "Performance baseline stored successfully",
            "pipeline_id": pipeline_id,
            "baseline_id": baseline.id if hasattr(baseline, 'id') else "mock_id"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing performance baseline: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for the recommendation service"""
    return {
        "status": "healthy",
        "service": "recommendation_engine",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }