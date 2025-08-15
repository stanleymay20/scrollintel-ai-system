"""
Media Management API Routes for Crisis Leadership Excellence

This module provides REST API endpoints for media management during crisis situations,
including media inquiry handling, PR strategy coordination, and sentiment monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from ...engines.media_management_engine import MediaManagementEngine
from ...models.media_management_models import (
    MediaInquiry, MediaResponse, PRStrategy, SentimentAnalysis,
    MediaManagementMetrics, MediaMonitoringAlert, MediaOutlet,
    MediaInquiryType, InquiryPriority, SentimentScore
)
from ...core.auth import get_current_user
from ...core.error_handling import handle_api_error

router = APIRouter(prefix="/api/v1/media-management", tags=["media-management"])
logger = logging.getLogger(__name__)

# Global engine instance
media_engine = MediaManagementEngine()


@router.post("/inquiries", response_model=Dict[str, Any])
async def handle_media_inquiry(
    inquiry: MediaInquiry,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Handle incoming media inquiry with professional response
    """
    try:
        logger.info(f"Processing media inquiry {inquiry.id} from {inquiry.outlet.name}")
        
        # Process inquiry
        response = await media_engine.handle_media_inquiry(inquiry)
        
        # Schedule follow-up tasks
        background_tasks.add_task(
            _schedule_inquiry_follow_up,
            inquiry.id,
            inquiry.deadline
        )
        
        return {
            "status": "success",
            "inquiry_id": inquiry.id,
            "response": {
                "id": response.id,
                "type": response.response_type,
                "key_messages": response.key_messages,
                "assigned_spokesperson": inquiry.assigned_spokesperson,
                "priority": inquiry.priority.value,
                "estimated_response_time": inquiry.received_at + timedelta(
                    minutes=media_engine.response_time_targets.get(inquiry.priority, 60)
                )
            },
            "next_steps": [
                "Review and approve response content",
                "Coordinate with assigned spokesperson",
                "Monitor for follow-up questions"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error handling media inquiry: {str(e)}")
        raise handle_api_error(e, "Failed to process media inquiry")


@router.get("/inquiries/{inquiry_id}", response_model=Dict[str, Any])
async def get_media_inquiry(
    inquiry_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get details of a specific media inquiry
    """
    try:
        if inquiry_id not in media_engine.active_inquiries:
            raise HTTPException(status_code=404, detail="Media inquiry not found")
        
        inquiry = media_engine.active_inquiries[inquiry_id]
        
        return {
            "inquiry": {
                "id": inquiry.id,
                "crisis_id": inquiry.crisis_id,
                "outlet": {
                    "name": inquiry.outlet.name,
                    "type": inquiry.outlet.outlet_type.value,
                    "reach": inquiry.outlet.reach,
                    "influence_score": inquiry.outlet.influence_score
                },
                "reporter": inquiry.reporter_name,
                "subject": inquiry.subject,
                "questions": inquiry.questions,
                "priority": inquiry.priority.value,
                "status": inquiry.response_status.value,
                "deadline": inquiry.deadline,
                "assigned_spokesperson": inquiry.assigned_spokesperson,
                "received_at": inquiry.received_at
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving media inquiry: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve media inquiry")


@router.get("/inquiries", response_model=Dict[str, Any])
async def list_media_inquiries(
    crisis_id: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """
    List media inquiries with optional filtering
    """
    try:
        inquiries = list(media_engine.active_inquiries.values())
        
        # Apply filters
        if crisis_id:
            inquiries = [inq for inq in inquiries if inq.crisis_id == crisis_id]
        
        if priority:
            try:
                priority_enum = InquiryPriority(priority)
                inquiries = [inq for inq in inquiries if inq.priority == priority_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid priority value")
        
        if status:
            inquiries = [inq for inq in inquiries if inq.response_status.value == status]
        
        # Limit results
        inquiries = inquiries[:limit]
        
        # Format response
        inquiry_list = []
        for inquiry in inquiries:
            inquiry_list.append({
                "id": inquiry.id,
                "crisis_id": inquiry.crisis_id,
                "outlet_name": inquiry.outlet.name,
                "subject": inquiry.subject,
                "priority": inquiry.priority.value,
                "status": inquiry.response_status.value,
                "deadline": inquiry.deadline,
                "received_at": inquiry.received_at
            })
        
        return {
            "inquiries": inquiry_list,
            "total": len(inquiry_list),
            "filters_applied": {
                "crisis_id": crisis_id,
                "priority": priority,
                "status": status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing media inquiries: {str(e)}")
        raise handle_api_error(e, "Failed to list media inquiries")


@router.post("/pr-strategy", response_model=Dict[str, Any])
async def coordinate_pr_strategy(
    strategy: PRStrategy,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Coordinate public relations strategy for crisis management
    """
    try:
        logger.info(f"Coordinating PR strategy {strategy.id} for crisis {strategy.crisis_id}")
        
        # Coordinate strategy
        coordination_result = await media_engine.coordinate_pr_strategy(
            strategy.crisis_id, 
            strategy
        )
        
        # Schedule monitoring tasks
        background_tasks.add_task(
            _schedule_strategy_monitoring,
            strategy.id,
            strategy.crisis_id
        )
        
        return {
            "status": "success",
            "strategy_id": strategy.id,
            "coordination_result": coordination_result,
            "message_consistency": {
                "score": coordination_result["consistency_score"],
                "status": "good" if coordination_result["consistency_score"] > 80 else "needs_review"
            },
            "timeline": coordination_result["timeline"],
            "monitoring": {
                "active": True,
                "next_review": coordination_result["next_review"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error coordinating PR strategy: {str(e)}")
        raise handle_api_error(e, "Failed to coordinate PR strategy")


@router.get("/pr-strategy/{strategy_id}", response_model=Dict[str, Any])
async def get_pr_strategy(
    strategy_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get details of a specific PR strategy
    """
    try:
        if strategy_id not in media_engine.pr_strategies:
            raise HTTPException(status_code=404, detail="PR strategy not found")
        
        strategy = media_engine.pr_strategies[strategy_id]
        
        return {
            "strategy": {
                "id": strategy.id,
                "crisis_id": strategy.crisis_id,
                "name": strategy.strategy_name,
                "objectives": strategy.objectives,
                "target_audiences": strategy.target_audiences,
                "key_messages": strategy.key_messages,
                "communication_channels": strategy.communication_channels,
                "timeline": strategy.timeline,
                "success_metrics": strategy.success_metrics,
                "spokesperson_assignments": strategy.spokesperson_assignments,
                "created_at": strategy.created_at,
                "updated_at": strategy.updated_at
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving PR strategy: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve PR strategy")


@router.post("/sentiment-analysis/{crisis_id}", response_model=Dict[str, Any])
async def analyze_media_sentiment(
    crisis_id: str,
    time_period_hours: Optional[int] = 24,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform media sentiment analysis for crisis
    """
    try:
        logger.info(f"Analyzing media sentiment for crisis {crisis_id}")
        
        time_period = timedelta(hours=time_period_hours)
        analysis = await media_engine.monitor_media_sentiment(crisis_id, time_period)
        
        return {
            "status": "success",
            "analysis": {
                "id": analysis.id,
                "crisis_id": analysis.crisis_id,
                "period": {
                    "start": analysis.analysis_period["start"],
                    "end": analysis.analysis_period["end"],
                    "hours": time_period_hours
                },
                "overall_sentiment": analysis.overall_sentiment.value,
                "sentiment_trend": analysis.sentiment_trend,
                "metrics": {
                    "total_mentions": analysis.mention_volume,
                    "positive_mentions": analysis.positive_mentions,
                    "negative_mentions": analysis.negative_mentions,
                    "neutral_mentions": analysis.neutral_mentions,
                    "positive_percentage": round(
                        (analysis.positive_mentions / analysis.mention_volume * 100) 
                        if analysis.mention_volume > 0 else 0, 1
                    ),
                    "negative_percentage": round(
                        (analysis.negative_mentions / analysis.mention_volume * 100) 
                        if analysis.mention_volume > 0 else 0, 1
                    )
                },
                "key_drivers": analysis.key_sentiment_drivers,
                "outlet_breakdown": analysis.outlet_breakdown,
                "recommendations": analysis.recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing media sentiment: {str(e)}")
        raise handle_api_error(e, "Failed to analyze media sentiment")


@router.get("/sentiment-history/{crisis_id}", response_model=Dict[str, Any])
async def get_sentiment_history(
    crisis_id: str,
    days: int = 7,
    current_user: dict = Depends(get_current_user)
):
    """
    Get historical sentiment analysis data for crisis
    """
    try:
        # Filter sentiment history for crisis
        crisis_sentiment = [
            analysis for analysis in media_engine.sentiment_history.values()
            if analysis.crisis_id == crisis_id
        ]
        
        # Sort by creation time
        crisis_sentiment.sort(key=lambda x: x.created_at, reverse=True)
        
        # Limit to requested days
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sentiment = [
            analysis for analysis in crisis_sentiment
            if analysis.created_at >= cutoff_date
        ]
        
        # Format response
        history_data = []
        for analysis in recent_sentiment:
            history_data.append({
                "timestamp": analysis.created_at,
                "overall_sentiment": analysis.overall_sentiment.value,
                "mention_volume": analysis.mention_volume,
                "positive_mentions": analysis.positive_mentions,
                "negative_mentions": analysis.negative_mentions,
                "sentiment_trend": analysis.sentiment_trend
            })
        
        return {
            "crisis_id": crisis_id,
            "period_days": days,
            "data_points": len(history_data),
            "history": history_data,
            "summary": {
                "average_sentiment": _calculate_average_sentiment(recent_sentiment),
                "total_mentions": sum(a.mention_volume for a in recent_sentiment),
                "trend_direction": _determine_overall_trend(recent_sentiment)
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving sentiment history: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve sentiment history")


@router.get("/alerts/{crisis_id}", response_model=Dict[str, Any])
async def get_media_alerts(
    crisis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get active media monitoring alerts for crisis
    """
    try:
        # Filter alerts for crisis
        crisis_alerts = [
            alert for alert in media_engine.monitoring_alerts.values()
            if alert.crisis_id == crisis_id and alert.resolved_at is None
        ]
        
        # Sort by severity and creation time
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        crisis_alerts.sort(
            key=lambda x: (severity_order.get(x.severity, 4), x.created_at),
            reverse=True
        )
        
        # Format alerts
        alert_list = []
        for alert in crisis_alerts:
            alert_list.append({
                "id": alert.id,
                "type": alert.alert_type,
                "severity": alert.severity,
                "description": alert.description,
                "recommended_actions": alert.recommended_actions,
                "stakeholders_to_notify": alert.stakeholders_to_notify,
                "created_at": alert.created_at,
                "acknowledged": alert.acknowledged_at is not None
            })
        
        return {
            "crisis_id": crisis_id,
            "active_alerts": alert_list,
            "alert_count": len(alert_list),
            "severity_breakdown": {
                "critical": len([a for a in crisis_alerts if a.severity == "critical"]),
                "high": len([a for a in crisis_alerts if a.severity == "high"]),
                "medium": len([a for a in crisis_alerts if a.severity == "medium"]),
                "low": len([a for a in crisis_alerts if a.severity == "low"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving media alerts: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve media alerts")


@router.post("/alerts/{alert_id}/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Acknowledge a media monitoring alert
    """
    try:
        if alert_id not in media_engine.monitoring_alerts:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert = media_engine.monitoring_alerts[alert_id]
        alert.acknowledged_at = datetime.now()
        
        logger.info(f"Alert {alert_id} acknowledged by user {current_user.get('username', 'unknown')}")
        
        return {
            "status": "success",
            "alert_id": alert_id,
            "acknowledged_at": alert.acknowledged_at,
            "acknowledged_by": current_user.get('username', 'unknown')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise handle_api_error(e, "Failed to acknowledge alert")


@router.get("/metrics/{crisis_id}", response_model=Dict[str, Any])
async def get_media_management_metrics(
    crisis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive media management effectiveness metrics
    """
    try:
        metrics = await media_engine.get_media_management_metrics(crisis_id)
        
        return {
            "crisis_id": crisis_id,
            "metrics": {
                "inquiry_management": {
                    "total_inquiries": metrics.total_inquiries,
                    "response_rate": round(metrics.response_rate * 100, 1),
                    "average_response_time_minutes": round(metrics.average_response_time, 1),
                    "response_rate_status": "excellent" if metrics.response_rate > 0.9 else "good" if metrics.response_rate > 0.7 else "needs_improvement"
                },
                "media_coverage": {
                    "positive_coverage_percentage": round(metrics.positive_coverage_percentage, 1),
                    "total_media_reach": metrics.media_reach,
                    "coverage_quality": "positive" if metrics.positive_coverage_percentage > 60 else "mixed" if metrics.positive_coverage_percentage > 40 else "challenging"
                },
                "message_consistency": {
                    "consistency_score": round(metrics.message_consistency_score, 1),
                    "status": "excellent" if metrics.message_consistency_score > 90 else "good" if metrics.message_consistency_score > 75 else "needs_review"
                },
                "spokesperson_performance": metrics.spokesperson_effectiveness,
                "crisis_control": {
                    "narrative_control_score": round(metrics.crisis_narrative_control, 1),
                    "reputation_impact_score": round(metrics.reputation_impact_score, 1),
                    "overall_effectiveness": "high" if metrics.crisis_narrative_control > 80 else "moderate" if metrics.crisis_narrative_control > 60 else "low"
                }
            },
            "calculated_at": metrics.calculated_at,
            "recommendations": _generate_metrics_recommendations(metrics)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving media management metrics: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve media management metrics")


# Helper functions

async def _schedule_inquiry_follow_up(inquiry_id: str, deadline: datetime):
    """
    Schedule follow-up tasks for media inquiry
    """
    # Implementation for scheduling follow-up reminders
    logger.info(f"Scheduled follow-up for inquiry {inquiry_id} with deadline {deadline}")


async def _schedule_strategy_monitoring(strategy_id: str, crisis_id: str):
    """
    Schedule ongoing monitoring for PR strategy
    """
    # Implementation for continuous strategy monitoring
    logger.info(f"Scheduled monitoring for PR strategy {strategy_id} in crisis {crisis_id}")


def _calculate_average_sentiment(sentiment_analyses: List[SentimentAnalysis]) -> str:
    """
    Calculate average sentiment from multiple analyses
    """
    if not sentiment_analyses:
        return "neutral"
    
    sentiment_scores = {
        SentimentScore.VERY_POSITIVE: 2,
        SentimentScore.POSITIVE: 1,
        SentimentScore.NEUTRAL: 0,
        SentimentScore.NEGATIVE: -1,
        SentimentScore.VERY_NEGATIVE: -2
    }
    
    total_score = sum(sentiment_scores.get(analysis.overall_sentiment, 0) for analysis in sentiment_analyses)
    average_score = total_score / len(sentiment_analyses)
    
    if average_score > 0.5:
        return "positive"
    elif average_score < -0.5:
        return "negative"
    else:
        return "neutral"


def _determine_overall_trend(sentiment_analyses: List[SentimentAnalysis]) -> str:
    """
    Determine overall sentiment trend
    """
    if len(sentiment_analyses) < 2:
        return "stable"
    
    # Simple trend analysis based on recent vs older sentiment
    recent = sentiment_analyses[:len(sentiment_analyses)//2]
    older = sentiment_analyses[len(sentiment_analyses)//2:]
    
    recent_avg = _calculate_average_sentiment(recent)
    older_avg = _calculate_average_sentiment(older)
    
    if recent_avg == "positive" and older_avg != "positive":
        return "improving"
    elif recent_avg == "negative" and older_avg != "negative":
        return "declining"
    else:
        return "stable"


def _generate_metrics_recommendations(metrics: MediaManagementMetrics) -> List[str]:
    """
    Generate recommendations based on metrics
    """
    recommendations = []
    
    if metrics.response_rate < 0.8:
        recommendations.append("Improve media inquiry response rate by streamlining approval processes")
    
    if metrics.average_response_time > 90:
        recommendations.append("Reduce average response time by pre-approving standard responses")
    
    if metrics.positive_coverage_percentage < 50:
        recommendations.append("Enhance proactive media outreach to improve coverage sentiment")
    
    if metrics.message_consistency_score < 80:
        recommendations.append("Strengthen message coordination across all spokespersons")
    
    if metrics.crisis_narrative_control < 70:
        recommendations.append("Increase proactive communication to maintain narrative control")
    
    return recommendations