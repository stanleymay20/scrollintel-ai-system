"""
API routes for AI-Enhanced Security Operations Center
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ...ai_soc.ai_soc_orchestrator import AISOCOrchestrator
from ...ai_soc.ml_siem_engine import SecurityEvent, EventType, ThreatLevel
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Initialize AI SOC
ai_soc = AISOCOrchestrator()

router = APIRouter(prefix="/api/v1/ai-soc", tags=["AI Security Operations Center"])

# Pydantic models for API
class SecurityEventRequest(BaseModel):
    event_type: str
    source_ip: str
    user_id: Optional[str] = None
    resource: str
    raw_data: Dict[str, Any] = {}
    risk_score: float = 0.0

class ThreatHuntingRequest(BaseModel):
    query_name: str
    time_window_hours: int = 24
    indicators: List[str] = []

class RiskForecastRequest(BaseModel):
    entity_type: str
    entity_id: str
    forecast_days: int = 30

class ThreatPredictionRequest(BaseModel):
    threat_type: str
    timeframe_days: int = 30

@router.on_event("startup")
async def startup_ai_soc():
    """Initialize AI SOC on startup"""
    try:
        await ai_soc.initialize()
        logger.info("AI SOC initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI SOC: {e}")

@router.post("/events/process")
async def process_security_event(event_request: SecurityEventRequest):
    """Process a security event through the AI SOC pipeline"""
    try:
        # Convert request to SecurityEvent
        event = SecurityEvent(
            event_id=f"api_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            event_type=EventType(event_request.event_type),
            source_ip=event_request.source_ip,
            user_id=event_request.user_id,
            resource=event_request.resource,
            raw_data=event_request.raw_data,
            risk_score=event_request.risk_score
        )
        
        # Process through AI SOC
        results = await ai_soc.process_security_event(event)
        
        return {
            "status": "success",
            "event_id": event.event_id,
            "processing_results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing security event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_soc_dashboard():
    """Get comprehensive SOC dashboard"""
    try:
        dashboard = await ai_soc.get_soc_dashboard()
        
        return {
            "status": "success",
            "dashboard": {
                "timestamp": dashboard.timestamp.isoformat(),
                "overall_risk_score": dashboard.overall_risk_score,
                "active_incidents": dashboard.active_incidents,
                "recent_alerts_count": len(dashboard.recent_alerts),
                "top_threats": dashboard.top_threats,
                "performance_metrics": {
                    "events_processed": dashboard.performance_metrics.events_processed,
                    "alerts_generated": dashboard.performance_metrics.alerts_generated,
                    "incidents_created": dashboard.performance_metrics.incidents_created,
                    "false_positive_rate": dashboard.performance_metrics.false_positive_rate,
                    "automation_rate": dashboard.performance_metrics.automation_rate
                },
                "system_health": dashboard.system_health,
                "recommendations": dashboard.recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting SOC dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_comprehensive_metrics():
    """Get comprehensive metrics from all AI SOC components"""
    try:
        metrics = ai_soc.get_comprehensive_metrics()
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_active_alerts(limit: int = 50):
    """Get active security alerts"""
    try:
        alerts = list(ai_soc.active_alerts.values())
        
        # Sort by creation time (most recent first)
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        
        # Limit results
        alerts = alerts[:limit]
        
        alert_data = []
        for alert in alerts:
            alert_data.append({
                "alert_id": alert.alert_id,
                "threat_type": alert.threat_type,
                "severity": alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity),
                "confidence": alert.confidence,
                "created_at": alert.created_at.isoformat(),
                "event_id": alert.event.event_id,
                "source_ip": alert.event.source_ip,
                "user_id": alert.event.user_id,
                "recommended_actions": alert.recommended_actions
            })
        
        return {
            "status": "success",
            "alerts": alert_data,
            "total_count": len(ai_soc.active_alerts)
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/incidents")
async def get_active_incidents(limit: int = 50):
    """Get active security incidents"""
    try:
        incidents = list(ai_soc.active_incidents.values())
        
        # Sort by creation time (most recent first)
        incidents.sort(key=lambda i: i.created_at, reverse=True)
        
        # Limit results
        incidents = incidents[:limit]
        
        incident_data = []
        for incident in incidents:
            incident_data.append({
                "incident_id": incident.incident_id,
                "title": incident.title,
                "category": incident.category.value if hasattr(incident.category, 'value') else str(incident.category),
                "severity": incident.severity.value if hasattr(incident.severity, 'value') else str(incident.severity),
                "status": incident.status.value if hasattr(incident.status, 'value') else str(incident.status),
                "created_at": incident.created_at.isoformat(),
                "affected_assets": incident.affected_assets,
                "classification_confidence": incident.classification_confidence
            })
        
        return {
            "status": "success",
            "incidents": incident_data,
            "total_count": len(ai_soc.active_incidents)
        }
        
    except Exception as e:
        logger.error(f"Error getting incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/threat-hunting/execute")
async def execute_threat_hunting(background_tasks: BackgroundTasks):
    """Execute threat hunting queries"""
    try:
        # Execute threat hunting in background
        background_tasks.add_task(ai_soc.behavioral_analytics.execute_threat_hunting)
        
        return {
            "status": "success",
            "message": "Threat hunting execution started"
        }
        
    except Exception as e:
        logger.error(f"Error executing threat hunting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecasting/risk")
async def generate_risk_forecast(request: RiskForecastRequest):
    """Generate risk forecast for entity"""
    try:
        forecast = await ai_soc.predictive_analytics.generate_risk_forecast(
            request.entity_type,
            request.entity_id,
            request.forecast_days
        )
        
        return {
            "status": "success",
            "forecast": {
                "forecast_id": forecast.forecast_id,
                "entity_type": forecast.entity_type,
                "entity_id": forecast.entity_id,
                "forecast_date": forecast.forecast_date.isoformat(),
                "predicted_risk_score": forecast.predicted_risk_score,
                "confidence_interval": forecast.confidence_interval,
                "risk_factors": forecast.risk_factors,
                "recommendations": forecast.recommendations,
                "created_at": forecast.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating risk forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecasting/threat")
async def predict_threat_likelihood(request: ThreatPredictionRequest):
    """Predict likelihood of specific threat type"""
    try:
        prediction = await ai_soc.predictive_analytics.predict_threat_likelihood(
            request.threat_type,
            request.timeframe_days
        )
        
        return {
            "status": "success",
            "prediction": {
                "prediction_id": prediction.prediction_id,
                "threat_type": prediction.threat_type,
                "probability": prediction.probability,
                "expected_timeframe_days": prediction.expected_timeframe.days,
                "affected_entities": prediction.affected_entities,
                "contributing_factors": prediction.contributing_factors,
                "mitigation_strategies": prediction.mitigation_strategies,
                "created_at": prediction.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error predicting threat likelihood: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/behavioral-analytics/user-risk/{user_id}")
async def get_user_risk_assessment(user_id: str):
    """Get comprehensive risk assessment for user"""
    try:
        risk_assessment = await ai_soc.behavioral_analytics.get_user_risk_assessment(user_id)
        
        return {
            "status": "success",
            "risk_assessment": risk_assessment
        }
        
    except Exception as e:
        logger.error(f"Error getting user risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml-siem/performance")
async def get_ml_siem_performance():
    """Get ML SIEM performance metrics"""
    try:
        metrics = ai_soc.ml_siem.get_performance_metrics()
        
        return {
            "status": "success",
            "ml_siem_metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting ML SIEM metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/correlation/performance")
async def get_correlation_performance():
    """Get threat correlation system performance metrics"""
    try:
        metrics = ai_soc.correlation_system.get_performance_metrics()
        
        return {
            "status": "success",
            "correlation_metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting correlation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/incident-response/performance")
async def get_incident_response_performance():
    """Get incident response orchestrator performance metrics"""
    try:
        metrics = ai_soc.incident_orchestrator.get_performance_metrics()
        
        return {
            "status": "success",
            "incident_response_metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting incident response metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/incidents/{incident_id}/response")
async def execute_incident_response(incident_id: str):
    """Execute automated response for specific incident"""
    try:
        if incident_id not in ai_soc.active_incidents:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        incident = ai_soc.active_incidents[incident_id]
        response = await ai_soc.incident_orchestrator.execute_incident_response(incident)
        
        return {
            "status": "success",
            "response": {
                "response_id": response.response_id,
                "incident_id": response.incident_id,
                "playbook_name": response.playbook.name,
                "status": response.status.value if hasattr(response.status, 'value') else str(response.status),
                "success_rate": response.success_rate,
                "human_intervention_required": response.human_intervention_required,
                "executed_steps": len(response.executed_steps)
            }
        }
        
    except Exception as e:
        logger.error(f"Error executing incident response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_ai_soc_health():
    """Get AI SOC system health status"""
    try:
        system_health = await ai_soc._get_system_health()
        
        return {
            "status": "success",
            "system_health": system_health,
            "initialized": ai_soc.is_initialized,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))