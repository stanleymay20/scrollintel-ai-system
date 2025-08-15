"""
API routes for AI Insight Generation Engine.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from ...engines.insight_generator import InsightGenerator, AnalyticsData
from ...models.insight_models import (
    Pattern, Insight, ActionRecommendation, BusinessContext, Anomaly,
    InsightType, PatternType, SignificanceLevel, ActionPriority
)
from ...models.dashboard_models import BusinessMetric
from ...core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/insights", tags=["insights"])

# Global insight generator instance
insight_generator = None


async def get_insight_generator() -> InsightGenerator:
    """Get or create insight generator instance."""
    global insight_generator
    if insight_generator is None:
        insight_generator = InsightGenerator()
        await insight_generator.start()
    return insight_generator


# Request/Response Models
class AnalyticsDataRequest(BaseModel):
    """Request model for analytics data."""
    metrics: List[Dict[str, Any]]
    time_range: Optional[Dict[str, str]] = None
    context: Optional[Dict[str, Any]] = None


class PatternDetectionRequest(BaseModel):
    """Request model for pattern detection."""
    metrics: List[Dict[str, Any]]
    pattern_types: Optional[List[str]] = None
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class InsightGenerationRequest(BaseModel):
    """Request model for insight generation."""
    patterns: List[Dict[str, Any]]
    business_context: Optional[Dict[str, Any]] = None


class AnomalyExplanationRequest(BaseModel):
    """Request model for anomaly explanation."""
    anomaly: Dict[str, Any]
    business_context: Optional[Dict[str, Any]] = None


class InsightResponse(BaseModel):
    """Response model for insights."""
    patterns: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]


class PatternResponse(BaseModel):
    """Response model for patterns."""
    patterns: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@router.post("/analyze", response_model=InsightResponse)
async def analyze_data(
    request: AnalyticsDataRequest,
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """
    Analyze business metrics data and generate comprehensive insights.
    
    This endpoint performs complete insight analysis including:
    - Pattern detection (trends, correlations, anomalies)
    - Natural language insight generation
    - Actionable recommendations
    - Business impact assessment
    """
    try:
        # Convert request data to BusinessMetric objects
        metrics = []
        for metric_data in request.metrics:
            metric = BusinessMetric(
                name=metric_data.get("name", ""),
                category=metric_data.get("category", "general"),
                value=float(metric_data.get("value", 0)),
                unit=metric_data.get("unit", ""),
                timestamp=datetime.fromisoformat(metric_data.get("timestamp", datetime.utcnow().isoformat())),
                source=metric_data.get("source", "api"),
                context=metric_data.get("context", {})
            )
            metrics.append(metric)
        
        # Parse time range
        time_range = None
        if request.time_range:
            start_time = datetime.fromisoformat(request.time_range.get("start", datetime.utcnow().isoformat()))
            end_time = datetime.fromisoformat(request.time_range.get("end", datetime.utcnow().isoformat()))
            time_range = (start_time, end_time)
        else:
            # Default to last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            time_range = (start_time, end_time)
        
        # Create analytics data
        analytics_data = AnalyticsData(
            metrics=metrics,
            time_range=time_range,
            context=request.context or {}
        )
        
        # Process insights
        result = await generator.process(analytics_data)
        
        return InsightResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/patterns/detect", response_model=PatternResponse)
async def detect_patterns(
    request: PatternDetectionRequest,
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """
    Detect patterns in business metrics data.
    
    Identifies various types of patterns including:
    - Trends (increasing/decreasing)
    - Correlations between metrics
    - Seasonal patterns
    - Cyclical behaviors
    """
    try:
        # Convert request data to BusinessMetric objects
        metrics = []
        for metric_data in request.metrics:
            metric = BusinessMetric(
                name=metric_data.get("name", ""),
                category=metric_data.get("category", "general"),
                value=float(metric_data.get("value", 0)),
                unit=metric_data.get("unit", ""),
                timestamp=datetime.fromisoformat(metric_data.get("timestamp", datetime.utcnow().isoformat())),
                source=metric_data.get("source", "api"),
                context=metric_data.get("context", {})
            )
            metrics.append(metric)
        
        # Create analytics data
        analytics_data = AnalyticsData(
            metrics=metrics,
            time_range=(datetime.utcnow() - timedelta(days=30), datetime.utcnow()),
            context={}
        )
        
        # Detect patterns
        patterns = await generator.analyze_data_patterns(analytics_data)
        
        # Filter by confidence threshold
        filtered_patterns = [
            p for p in patterns 
            if p.confidence >= request.confidence_threshold
        ]
        
        # Filter by pattern types if specified
        if request.pattern_types:
            filtered_patterns = [
                p for p in filtered_patterns
                if p.type in request.pattern_types
            ]
        
        return PatternResponse(
            patterns=[generator._pattern_to_dict(p) for p in filtered_patterns],
            metadata={
                "total_patterns": len(patterns),
                "filtered_patterns": len(filtered_patterns),
                "confidence_threshold": request.confidence_threshold,
                "pattern_types_filter": request.pattern_types,
                "processed_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {str(e)}")


@router.post("/insights/generate")
async def generate_insights(
    request: InsightGenerationRequest,
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """
    Generate natural language insights from detected patterns.
    
    Converts technical patterns into business-friendly insights with:
    - Clear explanations of what the pattern means
    - Business impact assessment
    - Confidence and significance scoring
    """
    try:
        # Convert pattern data to Pattern objects
        patterns = []
        for pattern_data in request.patterns:
            pattern = Pattern(
                type=pattern_data.get("type", ""),
                metric_name=pattern_data.get("metric_name", ""),
                metric_category=pattern_data.get("metric_category", ""),
                description=pattern_data.get("description", ""),
                confidence=float(pattern_data.get("confidence", 0)),
                significance=pattern_data.get("significance", "medium"),
                statistical_measures=pattern_data.get("statistical_measures", {})
            )
            patterns.append(pattern)
        
        # Generate insights
        insights = await generator.generate_insights(patterns)
        
        return {
            "insights": [generator._insight_to_dict(i) for i in insights],
            "metadata": {
                "patterns_processed": len(patterns),
                "insights_generated": len(insights),
                "processed_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")


@router.post("/recommendations/generate", response_model=RecommendationResponse)
async def generate_recommendations(
    insights: List[Dict[str, Any]],
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """
    Generate actionable recommendations based on insights.
    
    Creates specific, prioritized action items including:
    - Implementation steps
    - Timeline and effort estimates
    - Responsible roles
    - Success metrics
    """
    try:
        # Convert insight data to Insight objects
        insight_objects = []
        for insight_data in insights:
            insight = Insight(
                type=insight_data.get("type", ""),
                title=insight_data.get("title", ""),
                description=insight_data.get("description", ""),
                explanation=insight_data.get("explanation", ""),
                business_impact=insight_data.get("business_impact", ""),
                confidence=float(insight_data.get("confidence", 0)),
                significance=float(insight_data.get("significance", 0)),
                priority=insight_data.get("priority", "medium"),
                tags=insight_data.get("tags", []),
                affected_metrics=insight_data.get("affected_metrics", [])
            )
            insight_objects.append(insight)
        
        # Generate recommendations
        recommendations = await generator.suggest_actions(insight_objects)
        
        return RecommendationResponse(
            recommendations=[generator._recommendation_to_dict(r) for r in recommendations],
            metadata={
                "insights_processed": len(insight_objects),
                "recommendations_generated": len(recommendations),
                "processed_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@router.post("/anomalies/explain")
async def explain_anomaly(
    request: AnomalyExplanationRequest,
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """
    Explain detected anomalies with business context.
    
    Provides detailed explanations of anomalies including:
    - What makes it anomalous
    - Potential business causes
    - Recommended investigation steps
    """
    try:
        # Convert anomaly data to Anomaly object
        anomaly_data = request.anomaly
        anomaly = Anomaly(
            metric_name=anomaly_data.get("metric_name", ""),
            metric_value=float(anomaly_data.get("metric_value", 0)),
            expected_value=float(anomaly_data.get("expected_value", 0)),
            deviation_score=float(anomaly_data.get("deviation_score", 0)),
            anomaly_type=anomaly_data.get("anomaly_type", "unknown"),
            severity=anomaly_data.get("severity", "medium"),
            context=anomaly_data.get("context", {})
        )
        
        # Convert business context if provided
        business_context = None
        if request.business_context:
            context_data = request.business_context
            business_context = BusinessContext(
                context_type=context_data.get("context_type", "general"),
                name=context_data.get("name", ""),
                description=context_data.get("description", ""),
                context_metadata=context_data.get("metadata", {}),
                thresholds=context_data.get("thresholds", {}),
                kpis=context_data.get("kpis", []),
                benchmarks=context_data.get("benchmarks", {})
            )
        
        # Generate explanation
        explanation = await generator.explain_anomaly(anomaly, business_context)
        
        return {
            "explanation": explanation,
            "anomaly_details": {
                "metric_name": anomaly.metric_name,
                "severity": anomaly.severity,
                "deviation_score": anomaly.deviation_score,
                "anomaly_type": anomaly.anomaly_type
            },
            "metadata": {
                "explained_at": datetime.utcnow().isoformat(),
                "has_business_context": business_context is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error explaining anomaly: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly explanation failed: {str(e)}")


@router.get("/status")
async def get_engine_status(
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """Get the current status of the insight generation engine."""
    try:
        status = generator.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/health")
async def health_check(
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """Perform health check on the insight generation engine."""
    try:
        is_healthy = await generator.health_check()
        return {
            "healthy": is_healthy,
            "engine_id": generator.engine_id,
            "status": generator.status.value,
            "checked_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "checked_at": datetime.utcnow().isoformat()
        }


@router.get("/metrics")
async def get_engine_metrics(
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """Get performance metrics for the insight generation engine."""
    try:
        metrics = generator.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


# Background task endpoints
@router.post("/analyze/async")
async def analyze_data_async(
    request: AnalyticsDataRequest,
    background_tasks: BackgroundTasks,
    generator: InsightGenerator = Depends(get_insight_generator)
):
    """
    Start asynchronous analysis of business metrics data.
    
    Returns immediately with a task ID that can be used to check status
    and retrieve results when complete.
    """
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # Add background task
        background_tasks.add_task(
            _process_async_analysis,
            task_id,
            request,
            generator
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Analysis started in background",
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting async analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")


async def _process_async_analysis(
    task_id: str,
    request: AnalyticsDataRequest,
    generator: InsightGenerator
):
    """Process analysis in background task."""
    try:
        logger.info(f"Starting async analysis task {task_id}")
        
        # Convert request data (same as synchronous endpoint)
        metrics = []
        for metric_data in request.metrics:
            metric = BusinessMetric(
                name=metric_data.get("name", ""),
                category=metric_data.get("category", "general"),
                value=float(metric_data.get("value", 0)),
                unit=metric_data.get("unit", ""),
                timestamp=datetime.fromisoformat(metric_data.get("timestamp", datetime.utcnow().isoformat())),
                source=metric_data.get("source", "api"),
                context=metric_data.get("context", {})
            )
            metrics.append(metric)
        
        # Parse time range
        time_range = None
        if request.time_range:
            start_time = datetime.fromisoformat(request.time_range.get("start", datetime.utcnow().isoformat()))
            end_time = datetime.fromisoformat(request.time_range.get("end", datetime.utcnow().isoformat()))
            time_range = (start_time, end_time)
        else:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            time_range = (start_time, end_time)
        
        # Create analytics data
        analytics_data = AnalyticsData(
            metrics=metrics,
            time_range=time_range,
            context=request.context or {}
        )
        
        # Process insights
        result = await generator.process(analytics_data)
        
        # Store result (in a real implementation, this would be stored in a database or cache)
        # For now, we'll just log completion
        logger.info(f"Async analysis task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Async analysis task {task_id} failed: {e}")


# Utility endpoints
@router.get("/types/insights")
async def get_insight_types():
    """Get available insight types."""
    return {
        "insight_types": [t.value for t in InsightType],
        "pattern_types": [t.value for t in PatternType],
        "significance_levels": [s.value for s in SignificanceLevel],
        "action_priorities": [p.value for p in ActionPriority]
    }


@router.get("/templates/business-contexts")
async def get_business_context_templates():
    """Get templates for business contexts."""
    return {
        "templates": {
            "industry": {
                "context_type": "industry",
                "name": "Technology",
                "description": "Technology industry context",
                "thresholds": {
                    "revenue_growth": {"critical": 0.2, "high": 0.15, "medium": 0.1},
                    "customer_acquisition_cost": {"critical": 0.3, "high": 0.2, "medium": 0.15}
                },
                "kpis": ["revenue", "customer_acquisition_cost", "churn_rate", "user_engagement"],
                "benchmarks": {}
            },
            "department": {
                "context_type": "department",
                "name": "Engineering",
                "description": "Engineering department context",
                "thresholds": {
                    "deployment_frequency": {"critical": 0.5, "high": 0.3, "medium": 0.2},
                    "lead_time": {"critical": 0.4, "high": 0.3, "medium": 0.2}
                },
                "kpis": ["deployment_frequency", "lead_time", "mttr", "change_failure_rate"],
                "benchmarks": {}
            }
        }
    }