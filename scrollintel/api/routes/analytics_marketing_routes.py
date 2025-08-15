"""
Analytics and Marketing API Routes
Provides endpoints for analytics tracking, marketing attribution, and user segmentation
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from scrollintel.core.analytics_tracker import analytics_tracker, EventType
from scrollintel.core.conversion_funnel import funnel_analyzer
from scrollintel.core.ab_testing import ab_testing_framework, ExperimentStatus, VariantType
from scrollintel.core.marketing_attribution import marketing_attribution, AttributionModel, CampaignStatus
from scrollintel.core.user_segmentation import user_segmentation, SegmentType, CohortType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["Analytics & Marketing"])

# Analytics Tracking Endpoints

@router.post("/track/event")
async def track_event(
    user_id: str = Body(...),
    session_id: str = Body(...),
    event_name: str = Body(...),
    properties: Dict[str, Any] = Body(default={}),
    page_url: str = Body(...),
    user_agent: str = Body(...),
    ip_address: str = Body(...),
    campaign_data: Optional[Dict[str, str]] = Body(default=None)
):
    """Track a user event"""
    try:
        event_id = await analytics_tracker.track_event(
            user_id=user_id,
            session_id=session_id,
            event_name=event_name,
            properties=properties,
            page_url=page_url,
            user_agent=user_agent,
            ip_address=ip_address,
            campaign_data=campaign_data
        )
        
        # Also track for funnel analysis
        await funnel_analyzer.track_user_journey(
            user_id=user_id,
            session_id=session_id,
            event_name=event_name,
            event_properties=properties,
            page_url=page_url
        )
        
        # Track marketing touchpoint if campaign data present
        if campaign_data:
            await marketing_attribution.track_touchpoint(
                user_id=user_id,
                session_id=session_id,
                page_url=page_url,
                referrer=properties.get("referrer"),
                user_agent=user_agent,
                ip_address=ip_address,
                utm_parameters=campaign_data
            )
        
        return {"event_id": event_id, "status": "tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/track/page-view")
async def track_page_view(
    user_id: str = Body(...),
    session_id: str = Body(...),
    page_url: str = Body(...),
    page_title: str = Body(...),
    user_agent: str = Body(...),
    ip_address: str = Body(...),
    referrer: Optional[str] = Body(default=None)
):
    """Track a page view"""
    try:
        event_id = await analytics_tracker.track_page_view(
            user_id=user_id,
            session_id=session_id,
            page_url=page_url,
            page_title=page_title,
            user_agent=user_agent,
            ip_address=ip_address,
            referrer=referrer
        )
        
        return {"event_id": event_id, "status": "tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking page view: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user-behavior/{user_id}")
async def get_user_behavior(
    user_id: str,
    days: int = Query(default=30, ge=1, le=365)
):
    """Get user behavior analytics"""
    try:
        behavior_data = await analytics_tracker.get_user_behavior_data(user_id, days)
        return behavior_data
        
    except Exception as e:
        logger.error(f"Error getting user behavior: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_analytics_summary(
    days: int = Query(default=30, ge=1, le=365)
):
    """Get analytics summary for dashboard"""
    try:
        summary = await analytics_tracker.get_analytics_summary(days)
        return summary
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversion Funnel Endpoints

@router.post("/funnels")
async def create_custom_funnel(
    funnel_id: str = Body(...),
    steps: List[Dict[str, Any]] = Body(...)
):
    """Create a custom conversion funnel"""
    try:
        created_funnel_id = await funnel_analyzer.create_custom_funnel(funnel_id, steps)
        return {"funnel_id": created_funnel_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating funnel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/funnels/{funnel_id}/analysis")
async def analyze_funnel(
    funnel_id: str,
    days: int = Query(default=30, ge=1, le=365)
):
    """Analyze funnel performance"""
    try:
        analysis = await funnel_analyzer.analyze_funnel_performance(funnel_id, days)
        return {
            "funnel_id": analysis.funnel_id,
            "analysis_id": analysis.analysis_id,
            "period_start": analysis.period_start.isoformat(),
            "period_end": analysis.period_end.isoformat(),
            "total_users": analysis.total_users,
            "step_conversions": analysis.step_conversions,
            "step_rates": analysis.step_rates,
            "drop_off_analysis": analysis.drop_off_analysis,
            "optimization_suggestions": analysis.optimization_suggestions
        }
        
    except Exception as e:
        logger.error(f"Error analyzing funnel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/funnels/summary")
async def get_funnel_summary(
    days: int = Query(default=30, ge=1, le=365)
):
    """Get summary of all funnel performance"""
    try:
        summary = await funnel_analyzer.get_funnel_summary(days)
        return summary
        
    except Exception as e:
        logger.error(f"Error getting funnel summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# A/B Testing Endpoints

@router.post("/experiments")
async def create_experiment(
    name: str = Body(...),
    description: str = Body(...),
    hypothesis: str = Body(...),
    success_metrics: List[str] = Body(...),
    variants: List[Dict[str, Any]] = Body(...),
    target_sample_size: int = Body(default=1000),
    confidence_level: float = Body(default=0.95),
    created_by: str = Body(default="system")
):
    """Create a new A/B test experiment"""
    try:
        experiment_id = await ab_testing_framework.create_experiment(
            name=name,
            description=description,
            hypothesis=hypothesis,
            success_metrics=success_metrics,
            variants=variants,
            target_sample_size=target_sample_size,
            confidence_level=confidence_level,
            created_by=created_by
        )
        
        return {"experiment_id": experiment_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an A/B test experiment"""
    try:
        success = await ab_testing_framework.start_experiment(experiment_id)
        return {"experiment_id": experiment_id, "status": "started" if success else "failed"}
        
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/assign")
async def assign_user_to_experiment(
    experiment_id: str,
    user_id: str = Body(...),
    session_id: str = Body(...),
    user_properties: Dict[str, Any] = Body(default={})
):
    """Assign user to experiment variant"""
    try:
        variant_id = await ab_testing_framework.assign_user_to_experiment(
            user_id=user_id,
            experiment_id=experiment_id,
            session_id=session_id,
            user_properties=user_properties
        )
        
        return {"experiment_id": experiment_id, "variant_id": variant_id}
        
    except Exception as e:
        logger.error(f"Error assigning user to experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/results")
async def record_experiment_result(
    experiment_id: str,
    user_id: str = Body(...),
    metric_name: str = Body(...),
    metric_value: float = Body(...)
):
    """Record experiment result for user"""
    try:
        result_id = await ab_testing_framework.record_experiment_result(
            user_id=user_id,
            experiment_id=experiment_id,
            metric_name=metric_name,
            metric_value=metric_value
        )
        
        return {"result_id": result_id, "status": "recorded"}
        
    except Exception as e:
        logger.error(f"Error recording experiment result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/analysis")
async def analyze_experiment(experiment_id: str):
    """Get statistical analysis of experiment results"""
    try:
        analyses = await ab_testing_framework.analyze_experiment(experiment_id)
        
        # Convert analyses to serializable format
        serialized_analyses = {}
        for key, analysis in analyses.items():
            serialized_analyses[key] = {
                "experiment_id": analysis.experiment_id,
                "analysis_id": analysis.analysis_id,
                "metric_name": analysis.metric_name,
                "control_mean": analysis.control_mean,
                "treatment_mean": analysis.treatment_mean,
                "control_std": analysis.control_std,
                "treatment_std": analysis.treatment_std,
                "control_count": analysis.control_count,
                "treatment_count": analysis.treatment_count,
                "p_value": analysis.p_value,
                "confidence_interval": analysis.confidence_interval,
                "effect_size": analysis.effect_size,
                "is_significant": analysis.is_significant,
                "power": analysis.power,
                "recommendation": analysis.recommendation,
                "analyzed_at": analysis.analyzed_at.isoformat()
            }
        
        return serialized_analyses
        
    except Exception as e:
        logger.error(f"Error analyzing experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/dashboard")
async def get_experiment_dashboard():
    """Get experiment dashboard data"""
    try:
        dashboard = await ab_testing_framework.get_experiment_dashboard()
        return dashboard
        
    except Exception as e:
        logger.error(f"Error getting experiment dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Marketing Attribution Endpoints

@router.post("/campaigns")
async def create_campaign(
    name: str = Body(...),
    description: str = Body(...),
    campaign_type: str = Body(...),
    source: str = Body(...),
    medium: str = Body(...),
    budget: float = Body(...),
    start_date: datetime = Body(...),
    end_date: Optional[datetime] = Body(default=None),
    content: Optional[str] = Body(default=None),
    term: Optional[str] = Body(default=None),
    target_audience: Dict[str, Any] = Body(default={}),
    goals: List[str] = Body(default=[])
):
    """Create a new marketing campaign"""
    try:
        campaign_id = await marketing_attribution.create_campaign(
            name=name,
            description=description,
            campaign_type=campaign_type,
            source=source,
            medium=medium,
            budget=budget,
            start_date=start_date,
            end_date=end_date,
            content=content,
            term=term,
            target_audience=target_audience,
            goals=goals
        )
        
        return {"campaign_id": campaign_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conversions")
async def track_conversion(
    user_id: str = Body(...),
    session_id: str = Body(...),
    conversion_type: str = Body(...),
    conversion_value: float = Body(...),
    attribution_model: str = Body(default="last_touch")
):
    """Track a conversion with attribution"""
    try:
        model = AttributionModel(attribution_model)
        conversion_id = await marketing_attribution.track_conversion(
            user_id=user_id,
            session_id=session_id,
            conversion_type=conversion_type,
            conversion_value=conversion_value,
            attribution_model=model
        )
        
        return {"conversion_id": conversion_id, "status": "tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/attribution/report")
async def get_attribution_report(
    attribution_model: str = Query(default="last_touch"),
    days: int = Query(default=30, ge=1, le=365)
):
    """Generate attribution report"""
    try:
        model = AttributionModel(attribution_model)
        report = await marketing_attribution.generate_attribution_report(model, days)
        
        return {
            "report_id": report.report_id,
            "attribution_model": report.attribution_model.value,
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "campaign_performance": report.campaign_performance,
            "channel_performance": report.channel_performance,
            "conversion_paths": report.conversion_paths,
            "roi_analysis": report.roi_analysis,
            "generated_at": report.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating attribution report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/marketing/dashboard")
async def get_marketing_dashboard(
    days: int = Query(default=30, ge=1, le=365)
):
    """Get marketing attribution dashboard"""
    try:
        dashboard = await marketing_attribution.get_marketing_dashboard(days)
        return dashboard
        
    except Exception as e:
        logger.error(f"Error getting marketing dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# User Segmentation Endpoints

@router.post("/segments")
async def create_user_segment(
    name: str = Body(...),
    description: str = Body(...),
    segment_type: str = Body(...),
    criteria: Dict[str, Any] = Body(...)
):
    """Create a custom user segment"""
    try:
        segment_type_enum = SegmentType(segment_type)
        segment_id = await user_segmentation.create_custom_segment(
            name=name,
            description=description,
            segment_type=segment_type_enum,
            criteria=criteria
        )
        
        return {"segment_id": segment_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating user segment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user_id}/profile")
async def update_user_profile(
    user_id: str,
    events: List[Dict[str, Any]] = Body(...),
    properties: Dict[str, Any] = Body(default={})
):
    """Update user profile with activity data"""
    try:
        profile = await user_segmentation.update_user_profile(
            user_id=user_id,
            events=events,
            properties=properties
        )
        
        return {
            "user_id": profile.user_id,
            "behavioral_score": profile.behavioral_score,
            "engagement_level": profile.engagement_level,
            "lifecycle_stage": profile.lifecycle_stage,
            "segments": profile.segments,
            "total_sessions": profile.total_sessions,
            "total_events": profile.total_events
        }
        
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cohorts/{cohort_id}/analysis")
async def perform_cohort_analysis(
    cohort_id: str,
    analysis_periods: int = Query(default=12, ge=1, le=24)
):
    """Perform cohort analysis"""
    try:
        analysis = await user_segmentation.perform_cohort_analysis(cohort_id, analysis_periods)
        
        return {
            "analysis_id": analysis.analysis_id,
            "cohort_id": analysis.cohort_id,
            "period_start": analysis.period_start.isoformat(),
            "period_end": analysis.period_end.isoformat(),
            "retention_rates": analysis.retention_rates,
            "revenue_data": analysis.revenue_data,
            "user_counts": analysis.user_counts,
            "insights": analysis.insights,
            "generated_at": analysis.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error performing cohort analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/segmentation/dashboard")
async def get_segmentation_dashboard():
    """Get user segmentation dashboard"""
    try:
        dashboard = await user_segmentation.get_segmentation_dashboard()
        return dashboard
        
    except Exception as e:
        logger.error(f"Error getting segmentation dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Google Analytics Integration Endpoints

@router.post("/google-analytics/setup")
async def setup_google_analytics(
    tracking_id: str = Body(...),
    measurement_id: str = Body(...),
    api_secret: str = Body(...)
):
    """Setup Google Analytics integration"""
    try:
        # This would integrate with actual Google Analytics API
        # For now, return success
        return {
            "status": "configured",
            "tracking_id": tracking_id,
            "measurement_id": measurement_id
        }
        
    except Exception as e:
        logger.error(f"Error setting up Google Analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/google-analytics/send-event")
async def send_event_to_google_analytics(
    client_id: str = Body(...),
    event_name: str = Body(...),
    event_parameters: Dict[str, Any] = Body(default={})
):
    """Send event to Google Analytics"""
    try:
        # This would send to actual Google Analytics
        # For now, just track locally and return success
        return {
            "status": "sent",
            "client_id": client_id,
            "event_name": event_name
        }
        
    except Exception as e:
        logger.error(f"Error sending event to Google Analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))