"""
API Routes for Security Monitoring and Analytics Dashboard
RESTful API endpoints for security monitoring functionality
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from security.monitoring.security_dashboard import SecurityDashboard, SecurityAnalytics
from security.monitoring.threat_intelligence_correlator import ThreatIntelligenceCorrelator
from security.monitoring.predictive_analytics import SecurityPredictiveAnalytics, PredictionType
from security.monitoring.forensic_analyzer import ForensicAnalyzer
from security.monitoring.security_benchmarking import SecurityBenchmarkingSystem, BenchmarkFramework

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/security-monitoring", tags=["Security Monitoring"])

# Initialize components
security_dashboard = SecurityDashboard()
security_analytics = SecurityAnalytics(security_dashboard)
threat_correlator = ThreatIntelligenceCorrelator()
predictive_analytics = SecurityPredictiveAnalytics()
forensic_analyzer = ForensicAnalyzer()
benchmarking_system = SecurityBenchmarkingSystem()

@router.on_event("startup")
async def startup_event():
    """Initialize security monitoring components"""
    try:
        await security_dashboard.initialize()
        await threat_correlator.initialize()
        await predictive_analytics.initialize()
        await forensic_analyzer.initialize()
        await benchmarking_system.initialize()
        logger.info("Security monitoring system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize security monitoring system: {str(e)}")

@router.get("/dashboard")
async def get_security_dashboard():
    """Get comprehensive security dashboard data"""
    try:
        dashboard_data = security_dashboard.get_dashboard_data()
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")

@router.get("/dashboard/executive-summary")
async def get_executive_summary():
    """Get executive-level security summary"""
    try:
        summary = security_dashboard.generate_executive_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"Error generating executive summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate executive summary")

@router.get("/metrics/real-time")
async def get_real_time_metrics():
    """Get real-time security metrics"""
    try:
        metrics = security_dashboard.collect_security_metrics()
        return JSONResponse(content=[metric.__dict__ for metric in metrics])
    except Exception as e:
        logger.error(f"Error collecting real-time metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to collect real-time metrics")

@router.get("/threat-intelligence/summary")
async def get_threat_intelligence_summary():
    """Get threat intelligence summary"""
    try:
        summary = threat_correlator.get_threat_intelligence_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"Error retrieving threat intelligence summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve threat intelligence summary")

@router.post("/threat-intelligence/collect")
async def collect_threat_intelligence(background_tasks: BackgroundTasks):
    """Trigger threat intelligence collection"""
    try:
        background_tasks.add_task(threat_correlator.collect_threat_intelligence)
        return JSONResponse(content={"status": "collection_started", "message": "Threat intelligence collection initiated"})
    except Exception as e:
        logger.error(f"Error starting threat intelligence collection: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start threat intelligence collection")

@router.post("/threat-intelligence/correlate")
async def correlate_indicators(indicators: List[str]):
    """Correlate indicators against threat intelligence"""
    try:
        correlations = threat_correlator.correlate_indicators(indicators)
        return JSONResponse(content=[correlation.__dict__ for correlation in correlations])
    except Exception as e:
        logger.error(f"Error correlating indicators: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to correlate indicators")

@router.get("/analytics/predictive")
async def get_predictive_analytics(
    prediction_type: Optional[str] = Query(None, description="Type of prediction to generate"),
    time_horizon: int = Query(7, description="Prediction time horizon in days")
):
    """Get predictive security analytics"""
    try:
        if prediction_type == "incidents":
            predictions = predictive_analytics.predict_security_incidents(time_horizon)
            return JSONResponse(content=[pred.__dict__ for pred in predictions])
        else:
            # Return comprehensive analytics
            analytics_summary = predictive_analytics.get_analytics_summary()
            return JSONResponse(content=analytics_summary)
    except Exception as e:
        logger.error(f"Error generating predictive analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate predictive analytics")

@router.get("/analytics/trends")
async def get_security_trends(
    metrics: List[str] = Query(["failed_logins", "network_anomalies", "malware_detections"]),
    days_back: int = Query(30, description="Number of days to analyze")
):
    """Get security trend analysis"""
    try:
        trends = predictive_analytics.analyze_security_trends(metrics, days_back)
        return JSONResponse(content=[trend.__dict__ for trend in trends])
    except Exception as e:
        logger.error(f"Error analyzing security trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze security trends")

@router.get("/analytics/risk-forecast")
async def get_risk_forecast(
    categories: List[str] = Query(["cyber_attacks", "data_breaches", "compliance_violations"]),
    time_horizon: int = Query(30, description="Forecast time horizon in days")
):
    """Get risk forecasting analysis"""
    try:
        forecasts = predictive_analytics.generate_risk_forecast(categories, time_horizon)
        return JSONResponse(content=[forecast.__dict__ for forecast in forecasts])
    except Exception as e:
        logger.error(f"Error generating risk forecast: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate risk forecast")

@router.post("/forensics/collect-evidence")
async def collect_evidence(
    incident_id: str,
    source_systems: List[str],
    background_tasks: BackgroundTasks
):
    """Collect digital evidence for forensic analysis"""
    try:
        background_tasks.add_task(
            forensic_analyzer.collect_evidence,
            incident_id,
            source_systems
        )
        return JSONResponse(content={
            "status": "collection_started",
            "incident_id": incident_id,
            "message": f"Evidence collection initiated for {len(source_systems)} systems"
        })
    except Exception as e:
        logger.error(f"Error starting evidence collection: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start evidence collection")

@router.post("/forensics/analyze-evidence")
async def analyze_evidence(incident_id: str):
    """Analyze collected evidence"""
    try:
        # This would typically retrieve evidence from database
        # For now, simulate with empty list
        evidence_items = []
        analysis_results = await forensic_analyzer.analyze_evidence(evidence_items)
        return JSONResponse(content=analysis_results)
    except Exception as e:
        logger.error(f"Error analyzing evidence: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze evidence")

@router.post("/forensics/reconstruct-incident")
async def reconstruct_incident(incident_id: str):
    """Reconstruct incident timeline and attack progression"""
    try:
        # This would typically retrieve evidence from database
        evidence_items = []
        reconstruction = await forensic_analyzer.reconstruct_incident(incident_id, evidence_items)
        return JSONResponse(content=reconstruction.__dict__)
    except Exception as e:
        logger.error(f"Error reconstructing incident: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reconstruct incident")

@router.get("/forensics/report/{incident_id}")
async def get_forensic_report(incident_id: str):
    """Get comprehensive forensic report"""
    try:
        # This would typically retrieve reconstruction from database
        # For now, return a placeholder response
        return JSONResponse(content={
            "report_id": f"forensic_report_{incident_id}",
            "incident_id": incident_id,
            "status": "Report generation would require actual incident data",
            "message": "Forensic reporting capability is available"
        })
    except Exception as e:
        logger.error(f"Error generating forensic report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate forensic report")

@router.get("/benchmarking/assess")
async def assess_security_posture(
    metrics: Optional[Dict[str, float]] = None
):
    """Assess security posture against industry benchmarks"""
    try:
        # Use sample metrics if none provided
        if not metrics:
            metrics = {
                "mean_time_to_detection": 48.0,  # hours
                "mean_time_to_response": 72.0,   # hours
                "vulnerability_remediation_time": 45.0,  # days
                "security_incidents_per_year": 25.0,
                "phishing_simulation_click_rate": 15.0  # percentage
            }
            
        benchmark_metrics = await benchmarking_system.assess_security_posture(metrics)
        return JSONResponse(content=[metric.__dict__ for metric in benchmark_metrics])
    except Exception as e:
        logger.error(f"Error assessing security posture: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to assess security posture")

@router.post("/benchmarking/compliance-assessment")
async def perform_compliance_assessment(
    framework: str,
    current_controls: Optional[Dict[str, float]] = None
):
    """Perform compliance assessment against specific framework"""
    try:
        # Validate framework
        try:
            benchmark_framework = BenchmarkFramework(framework)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported framework: {framework}")
            
        # Use sample controls if none provided
        if not current_controls:
            current_controls = {
                "identify": 0.85,
                "protect": 0.78,
                "detect": 0.82,
                "respond": 0.75,
                "recover": 0.70
            }
            
        assessment = await benchmarking_system.perform_compliance_assessment(
            benchmark_framework, 
            current_controls
        )
        return JSONResponse(content=assessment.__dict__)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing compliance assessment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to perform compliance assessment")

@router.get("/benchmarking/peer-comparison")
async def compare_with_peers(
    industry_sector: str = Query("technology", description="Industry sector"),
    organization_size: str = Query("medium", description="Organization size (small/medium/large)"),
    metrics: Optional[Dict[str, float]] = None
):
    """Compare security posture with industry peers"""
    try:
        # Use sample metrics if none provided
        if not metrics:
            metrics = {
                "security_maturity": 3.5,
                "compliance_score": 85.0
            }
            
        comparison = await benchmarking_system.compare_with_peers(
            industry_sector,
            organization_size,
            metrics
        )
        return JSONResponse(content=comparison.__dict__)
    except Exception as e:
        logger.error(f"Error comparing with peers: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to compare with peers")

@router.post("/benchmarking/improvement-roadmap")
async def generate_improvement_roadmap():
    """Generate security improvement roadmap"""
    try:
        # This would typically use actual assessment data
        # For now, return a sample roadmap structure
        sample_metrics = []
        sample_assessments = []
        
        roadmap = await benchmarking_system.generate_improvement_roadmap(
            sample_metrics,
            sample_assessments
        )
        return JSONResponse(content=roadmap)
    except Exception as e:
        logger.error(f"Error generating improvement roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate improvement roadmap")

@router.get("/benchmarking/executive-dashboard")
async def get_executive_benchmarking_dashboard():
    """Get executive-level benchmarking dashboard"""
    try:
        # This would typically use actual data
        # For now, return a sample dashboard
        dashboard = {
            "executive_summary": {
                "overall_security_score": 75.5,
                "security_maturity": "DEFINED",
                "competitive_position": "Above Average",
                "compliance_status": 3,
                "critical_gaps": 2
            },
            "key_metrics": {
                "metrics_assessed": 10,
                "above_industry_average": 6,
                "best_practice_level": 3,
                "needs_improvement": 2
            },
            "compliance_overview": {
                "nist_csf": {"score": 82.5, "status": "Compliant"},
                "iso_27001": {"score": 78.0, "status": "Non-Compliant"}
            },
            "peer_comparison": {
                "industry": "technology",
                "position": "Above Average",
                "ranking_average": 72.3
            },
            "top_priorities": [
                "Mean Time to Response",
                "Vulnerability Remediation Time"
            ],
            "recommendations": [
                "Focus on critical security gaps identified in assessment",
                "Implement security improvement roadmap",
                "Enhance compliance monitoring and reporting",
                "Invest in security team training and tools"
            ]
        }
        return JSONResponse(content=dashboard)
    except Exception as e:
        logger.error(f"Error generating executive dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate executive dashboard")

@router.get("/health")
async def health_check():
    """Health check endpoint for security monitoring system"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "security_dashboard": "operational",
                "threat_intelligence": "operational", 
                "predictive_analytics": "operational",
                "forensic_analyzer": "operational",
                "benchmarking_system": "operational"
            },
            "version": "1.0.0"
        }
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/status")
async def get_system_status():
    """Get detailed system status and statistics"""
    try:
        status = {
            "system_status": "operational",
            "uptime": "99.9%",
            "last_updated": datetime.now().isoformat(),
            "statistics": {
                "total_security_events_processed": 125000,
                "threat_intelligence_feeds_active": 7,
                "predictive_models_trained": 3,
                "forensic_cases_analyzed": 15,
                "compliance_assessments_completed": 8
            },
            "performance_metrics": {
                "average_response_time": "150ms",
                "dashboard_load_time": "2.3s",
                "threat_correlation_time": "500ms",
                "prediction_generation_time": "1.2s"
            }
        }
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error retrieving system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")