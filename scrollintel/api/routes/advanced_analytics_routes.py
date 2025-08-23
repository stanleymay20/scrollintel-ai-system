"""
Advanced Analytics API Routes

This module provides REST API endpoints for all advanced analytics capabilities
including graph analytics, semantic search, pattern recognition, and predictive analytics.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ...models.advanced_analytics_models import (
    AdvancedAnalyticsRequest, AdvancedAnalyticsResponse, AnalyticsType,
    GraphAnalysisRequest, GraphAnalysisResult,
    SemanticQuery, SemanticSearchResponse,
    PatternRecognitionRequest, PatternRecognitionResult,
    PredictiveAnalyticsRequest, PredictiveAnalyticsResult,
    AnalyticsInsight
)
from ...engines.advanced_analytics_engine import advanced_analytics_engine
from ...core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/advanced-analytics", tags=["Advanced Analytics"])


@router.post("/execute", response_model=AdvancedAnalyticsResponse)
async def execute_analytics(
    request: AdvancedAnalyticsRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Execute advanced analytics operations.
    
    Supports all analytics types:
    - Graph Analysis
    - Semantic Search
    - Pattern Recognition
    - Predictive Analytics
    """
    try:
        logger.info(f"Executing {request.analytics_type.value} analytics for user {current_user.get('user_id')}")
        
        # Execute analytics
        result = await advanced_analytics_engine.execute_analytics(request)
        
        # Track usage in background
        background_tasks.add_task(
            _track_analytics_usage,
            current_user.get('user_id'),
            request.analytics_type,
            result.execution_metrics
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics execution failed: {str(e)}")


@router.post("/comprehensive-analysis")
async def execute_comprehensive_analysis(
    data_sources: List[str],
    business_context: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Execute comprehensive analysis using all analytics engines.
    
    This endpoint runs graph analytics, pattern recognition, and predictive analytics
    in parallel to provide a complete business intelligence overview.
    """
    try:
        logger.info(f"Starting comprehensive analysis for user {current_user.get('user_id')}")
        
        result = await advanced_analytics_engine.execute_comprehensive_analysis(
            data_sources, business_context
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Comprehensive analysis completed for {len(data_sources)} data sources"
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


@router.get("/business-intelligence-summary")
async def get_business_intelligence_summary(
    time_period: int = Query(30, description="Number of days to look back"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get business intelligence summary from recent analytics.
    
    Provides executive-level insights, trends, and recommendations based on
    recent analytics operations and their results.
    """
    try:
        summary = await advanced_analytics_engine.get_business_intelligence_summary(time_period)
        
        return {
            "status": "success",
            "data": summary,
            "time_period_days": time_period,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error generating BI summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"BI summary generation failed: {str(e)}")


# Graph Analytics Endpoints
@router.post("/graph/build", tags=["Graph Analytics"])
async def build_enterprise_graph(
    data_sources: List[str],
    current_user: dict = Depends(get_current_user)
):
    """Build enterprise graph from multiple data sources."""
    try:
        result = await advanced_analytics_engine.graph_engine.build_enterprise_graph(data_sources)
        
        return {
            "status": "success",
            "data": result,
            "message": f"Enterprise graph built from {len(data_sources)} data sources"
        }
        
    except Exception as e:
        logger.error(f"Error building enterprise graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph building failed: {str(e)}")


@router.post("/graph/analyze", response_model=GraphAnalysisResult, tags=["Graph Analytics"])
async def analyze_graph_relationships(
    request: GraphAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze complex relationships in the enterprise graph."""
    try:
        result = await advanced_analytics_engine.graph_engine.analyze_complex_relationships(request)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing graph relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph analysis failed: {str(e)}")


@router.get("/graph/opportunities", tags=["Graph Analytics"])
async def detect_graph_opportunities(
    business_context: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """Detect business opportunities through graph analysis."""
    try:
        opportunities = await advanced_analytics_engine.graph_engine.detect_business_opportunities(
            business_context or {}
        )
        
        return {
            "status": "success",
            "data": opportunities,
            "count": len(opportunities)
        }
        
    except Exception as e:
        logger.error(f"Error detecting graph opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Opportunity detection failed: {str(e)}")


# Semantic Search Endpoints
@router.post("/search/index", tags=["Semantic Search"])
async def index_enterprise_data(
    data_sources: List[str],
    current_user: dict = Depends(get_current_user)
):
    """Index enterprise data for semantic search."""
    try:
        result = await advanced_analytics_engine.search_engine.index_enterprise_data(data_sources)
        
        return {
            "status": "success",
            "data": result,
            "message": f"Indexed data from {len(data_sources)} sources"
        }
        
    except Exception as e:
        logger.error(f"Error indexing enterprise data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data indexing failed: {str(e)}")


@router.post("/search/query", response_model=SemanticSearchResponse, tags=["Semantic Search"])
async def semantic_search(
    query: SemanticQuery,
    current_user: dict = Depends(get_current_user)
):
    """Perform semantic search across enterprise data."""
    try:
        result = await advanced_analytics_engine.search_engine.semantic_search(query)
        return result
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.get("/search/related/{content_id}", tags=["Semantic Search"])
async def discover_related_content(
    content_id: str,
    max_results: int = Query(20, description="Maximum number of related items"),
    current_user: dict = Depends(get_current_user)
):
    """Discover content related to a specific document or entity."""
    try:
        related_items = await advanced_analytics_engine.search_engine.discover_related_content(
            content_id, max_results
        )
        
        return {
            "status": "success",
            "data": related_items,
            "content_id": content_id,
            "count": len(related_items)
        }
        
    except Exception as e:
        logger.error(f"Error discovering related content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Related content discovery failed: {str(e)}")


# Pattern Recognition Endpoints
@router.post("/patterns/recognize", response_model=PatternRecognitionResult, tags=["Pattern Recognition"])
async def recognize_patterns(
    request: PatternRecognitionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Recognize patterns in business data."""
    try:
        result = await advanced_analytics_engine.pattern_engine.recognize_patterns(request)
        return result
        
    except Exception as e:
        logger.error(f"Error recognizing patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pattern recognition failed: {str(e)}")


@router.get("/patterns/opportunities", tags=["Pattern Recognition"])
async def detect_emerging_opportunities(
    data_sources: List[str] = Query(..., description="Data sources to analyze"),
    lookback_days: int = Query(90, description="Number of days to look back"),
    current_user: dict = Depends(get_current_user)
):
    """Detect emerging business opportunities through pattern analysis."""
    try:
        opportunities = await advanced_analytics_engine.pattern_engine.detect_emerging_opportunities(
            data_sources, lookback_days
        )
        
        return {
            "status": "success",
            "data": opportunities,
            "data_sources": data_sources,
            "lookback_days": lookback_days,
            "count": len(opportunities)
        }
        
    except Exception as e:
        logger.error(f"Error detecting emerging opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Opportunity detection failed: {str(e)}")


@router.post("/patterns/monitor", tags=["Pattern Recognition"])
async def monitor_pattern_changes(
    baseline_patterns: List[Dict[str, Any]],
    current_data_source: str,
    current_user: dict = Depends(get_current_user)
):
    """Monitor changes in patterns compared to a baseline."""
    try:
        # Convert baseline patterns (simplified for API)
        from ...models.advanced_analytics_models import RecognizedPattern, PatternType
        
        baseline_pattern_objects = []
        for pattern_data in baseline_patterns:
            pattern = RecognizedPattern(
                pattern_type=PatternType(pattern_data.get("pattern_type", "trend")),
                description=pattern_data.get("description", ""),
                strength=pattern_data.get("strength", 0.5),
                confidence=pattern_data.get("confidence", 0.5),
                data_points=pattern_data.get("data_points", [])
            )
            baseline_pattern_objects.append(pattern)
        
        changes = await advanced_analytics_engine.pattern_engine.monitor_pattern_changes(
            baseline_pattern_objects, current_data_source
        )
        
        return {
            "status": "success",
            "data": changes,
            "baseline_patterns_count": len(baseline_patterns),
            "current_data_source": current_data_source
        }
        
    except Exception as e:
        logger.error(f"Error monitoring pattern changes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pattern monitoring failed: {str(e)}")


# Predictive Analytics Endpoints
@router.post("/predictions/forecast", response_model=PredictiveAnalyticsResult, tags=["Predictive Analytics"])
async def predict_business_outcomes(
    request: PredictiveAnalyticsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate comprehensive business outcome predictions."""
    try:
        result = await advanced_analytics_engine.predictive_engine.predict_business_outcomes(request)
        return result
        
    except Exception as e:
        logger.error(f"Error predicting business outcomes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predictions/revenue", tags=["Predictive Analytics"])
async def forecast_revenue(
    data_sources: List[str],
    forecast_horizon: int = Query(90, description="Number of days to forecast"),
    current_user: dict = Depends(get_current_user)
):
    """Generate revenue forecasts with confidence intervals."""
    try:
        prediction = await advanced_analytics_engine.predictive_engine.forecast_revenue(
            data_sources, forecast_horizon
        )
        
        return {
            "status": "success",
            "data": prediction,
            "forecast_horizon_days": forecast_horizon,
            "data_sources": data_sources
        }
        
    except Exception as e:
        logger.error(f"Error forecasting revenue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Revenue forecasting failed: {str(e)}")


@router.post("/predictions/churn", tags=["Predictive Analytics"])
async def predict_customer_churn(
    customer_data_source: str,
    current_user: dict = Depends(get_current_user)
):
    """Predict customer churn probabilities."""
    try:
        predictions = await advanced_analytics_engine.predictive_engine.predict_customer_churn(
            customer_data_source
        )
        
        return {
            "status": "success",
            "data": predictions,
            "customer_data_source": customer_data_source,
            "predictions_count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error predicting customer churn: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Churn prediction failed: {str(e)}")


@router.get("/predictions/growth-opportunities", tags=["Predictive Analytics"])
async def identify_growth_opportunities(
    data_sources: List[str] = Query(..., description="Data sources to analyze"),
    current_user: dict = Depends(get_current_user)
):
    """Identify growth opportunities through predictive analysis."""
    try:
        opportunities = await advanced_analytics_engine.predictive_engine.identify_growth_opportunities(
            data_sources
        )
        
        return {
            "status": "success",
            "data": opportunities,
            "data_sources": data_sources,
            "opportunities_count": len(opportunities)
        }
        
    except Exception as e:
        logger.error(f"Error identifying growth opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Growth opportunity identification failed: {str(e)}")


# Performance and Monitoring Endpoints
@router.get("/performance/summary")
async def get_performance_summary(
    current_user: dict = Depends(get_current_user)
):
    """Get performance summary across all analytics engines."""
    try:
        summary = await advanced_analytics_engine.get_performance_summary()
        
        return {
            "status": "success",
            "data": summary,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance summary failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for advanced analytics services."""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "engines": {
                "graph_analytics": "operational",
                "semantic_search": "operational", 
                "pattern_recognition": "operational",
                "predictive_analytics": "operational"
            },
            "version": "1.0.0"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }


# Utility Functions
async def _track_analytics_usage(user_id: str, analytics_type: AnalyticsType, 
                                execution_metrics: Dict[str, Any]):
    """Track analytics usage for billing and monitoring."""
    try:
        # Log usage metrics
        logger.info(f"Analytics usage - User: {user_id}, Type: {analytics_type.value}, "
                   f"Execution time: {execution_metrics.get('execution_time_ms', 0):.2f}ms")
        
        # Here you would typically store usage data in a database
        # for billing, monitoring, and analytics purposes
        
    except Exception as e:
        logger.warning(f"Error tracking analytics usage: {str(e)}")


# Export router
__all__ = ["router"]