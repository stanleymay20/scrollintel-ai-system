"""
Advanced Analytics API Routes
Provides endpoints for comprehensive reporting, statistical analysis, and executive summaries
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import io
import pandas as pd
from pydantic import BaseModel

from scrollintel.engines.comprehensive_reporting_engine import (
    ComprehensiveReportingEngine, ReportConfig, ReportFormat, ReportType
)
from scrollintel.core.automated_report_scheduler import (
    AutomatedReportScheduler, ScheduledReportConfig, ScheduleFrequency, DeliveryMethod
)
from scrollintel.engines.advanced_statistical_analytics import AdvancedStatisticalAnalytics
from scrollintel.engines.executive_summary_generator import (
    ExecutiveSummaryGenerator, SummaryType
)

router = APIRouter(prefix="/api/advanced-analytics", tags=["advanced-analytics"])

# Initialize engines
reporting_engine = ComprehensiveReportingEngine()
scheduler = AutomatedReportScheduler(reporting_engine)
statistical_engine = AdvancedStatisticalAnalytics()
summary_generator = ExecutiveSummaryGenerator()

# Pydantic models
class ReportGenerationRequest(BaseModel):
    report_type: str
    format: str
    title: str
    description: str
    date_range: Dict[str, str]
    filters: Dict[str, Any] = {}
    sections: List[str] = []
    visualizations: List[str] = []
    recipients: List[str] = []

class ScheduledReportRequest(BaseModel):
    name: str
    description: str
    report_config: ReportGenerationRequest
    frequency: str
    schedule_time: str
    delivery_methods: List[str]
    delivery_config: Dict[str, Any]
    data_source_config: Dict[str, Any]

class StatisticalAnalysisRequest(BaseModel):
    data: Dict[str, Any]
    target_column: Optional[str] = None
    analysis_types: List[str] = ["comprehensive"]

class ExecutiveSummaryRequest(BaseModel):
    data: Dict[str, Any]
    summary_type: str = "comprehensive"
    focus_areas: Optional[List[str]] = None

@router.post("/reports/generate")
async def generate_report(request: ReportGenerationRequest):
    """Generate a comprehensive report"""
    try:
        # Convert request to ReportConfig
        config = ReportConfig(
            report_type=ReportType(request.report_type),
            format=ReportFormat(request.format),
            title=request.title,
            description=request.description,
            date_range={
                'start_date': datetime.fromisoformat(request.date_range['start_date']),
                'end_date': datetime.fromisoformat(request.date_range['end_date'])
            },
            filters=request.filters,
            sections=request.sections,
            visualizations=request.visualizations,
            recipients=request.recipients
        )
        
        # Fetch data (mock implementation)
        data = await _fetch_report_data(config)
        
        # Generate report
        report = reporting_engine.generate_report(config, data)
        
        # Return report metadata and download link
        return {
            "report_id": report.id,
            "status": "completed",
            "generated_at": report.generated_at.isoformat(),
            "format": report.config.format.value,
            "download_url": f"/api/advanced-analytics/reports/{report.id}/download",
            "metadata": report.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/reports/{report_id}/download")
async def download_report(report_id: str):
    """Download a generated report"""
    try:
        # In a real implementation, you would retrieve the report from storage
        # For now, we'll generate a sample report
        sample_data = await _get_sample_data()
        
        config = ReportConfig(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.PDF,
            title="Sample Executive Report",
            description="Comprehensive business analytics report",
            date_range={
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now()
            },
            filters={},
            sections=[],
            visualizations=[],
            recipients=[]
        )
        
        report = reporting_engine.generate_report(config, sample_data)
        
        # Create streaming response
        def generate_file():
            yield report.file_data
        
        media_type = "application/pdf" if config.format == ReportFormat.PDF else "application/octet-stream"
        filename = f"{report_id}.{config.format.value}"
        
        return StreamingResponse(
            io.BytesIO(report.file_data),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report download failed: {str(e)}")

@router.post("/reports/schedule")
async def schedule_report(request: ScheduledReportRequest):
    """Schedule automated report generation"""
    try:
        # Convert request to ScheduledReportConfig
        report_config = ReportConfig(
            report_type=ReportType(request.report_config.report_type),
            format=ReportFormat(request.report_config.format),
            title=request.report_config.title,
            description=request.report_config.description,
            date_range={
                'start_date': datetime.fromisoformat(request.report_config.date_range['start_date']),
                'end_date': datetime.fromisoformat(request.report_config.date_range['end_date'])
            },
            filters=request.report_config.filters,
            sections=request.report_config.sections,
            visualizations=request.report_config.visualizations,
            recipients=request.report_config.recipients
        )
        
        scheduled_config = ScheduledReportConfig(
            id=f"scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=request.name,
            description=request.description,
            report_config=report_config,
            frequency=ScheduleFrequency(request.frequency),
            schedule_time=request.schedule_time,
            delivery_methods=[DeliveryMethod(method) for method in request.delivery_methods],
            delivery_config=request.delivery_config,
            data_source_config=request.data_source_config
        )
        
        # Add to scheduler
        success = scheduler.add_scheduled_report(scheduled_config)
        
        if success:
            return {
                "scheduled_report_id": scheduled_config.id,
                "status": "scheduled",
                "next_run": scheduled_config.next_run.isoformat() if scheduled_config.next_run else None,
                "frequency": scheduled_config.frequency.value
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to schedule report")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report scheduling failed: {str(e)}")

@router.get("/reports/scheduled")
async def get_scheduled_reports():
    """Get all scheduled reports"""
    try:
        reports = scheduler.get_scheduled_reports()
        
        return {
            "scheduled_reports": [
                {
                    "id": report.id,
                    "name": report.name,
                    "description": report.description,
                    "frequency": report.frequency.value,
                    "is_active": report.is_active,
                    "last_run": report.last_run.isoformat() if report.last_run else None,
                    "next_run": report.next_run.isoformat() if report.next_run else None,
                    "delivery_methods": [method.value for method in report.delivery_methods]
                }
                for report in reports
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled reports: {str(e)}")

@router.delete("/reports/scheduled/{report_id}")
async def delete_scheduled_report(report_id: str):
    """Delete a scheduled report"""
    try:
        success = scheduler.remove_scheduled_report(report_id)
        
        if success:
            return {"status": "deleted", "report_id": report_id}
        else:
            raise HTTPException(status_code=404, detail="Scheduled report not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete scheduled report: {str(e)}")

@router.post("/analysis/statistical")
async def perform_statistical_analysis(request: StatisticalAnalysisRequest):
    """Perform advanced statistical analysis"""
    try:
        # Convert data to DataFrame
        if isinstance(request.data, dict) and 'records' in request.data:
            df = pd.DataFrame(request.data['records'])
        else:
            # Assume data is in a format that can be converted to DataFrame
            df = pd.DataFrame(request.data)
        
        # Perform comprehensive analysis
        results = statistical_engine.comprehensive_analysis(df, request.target_column)
        
        # Generate insights
        insights = statistical_engine.generate_insights(results)
        
        return {
            "analysis_results": results,
            "insights": insights,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include=['number']).columns)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical analysis failed: {str(e)}")

@router.post("/analysis/outliers")
async def detect_outliers(
    data: Dict[str, Any],
    method: str = Query("isolation_forest", description="Outlier detection method")
):
    """Detect outliers in data"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Detect outliers
        result = statistical_engine.detect_outliers(df, method)
        
        return {
            "outliers": result.anomalies,
            "anomaly_scores": result.anomaly_scores,
            "total_anomalies": result.total_anomalies,
            "anomaly_percentage": result.anomaly_percentage,
            "method": result.method,
            "threshold": result.threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outlier detection failed: {str(e)}")

@router.post("/analysis/clustering")
async def perform_clustering(
    data: Dict[str, Any],
    method: str = Query("kmeans", description="Clustering method"),
    n_clusters: Optional[int] = Query(None, description="Number of clusters")
):
    """Perform cluster analysis"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Perform clustering
        result = statistical_engine.cluster_analysis(df, method, n_clusters)
        
        return {
            "cluster_labels": result.cluster_labels,
            "cluster_centers": result.cluster_centers,
            "n_clusters": result.n_clusters,
            "silhouette_score": result.silhouette_score,
            "method": result.method,
            "inertia": result.inertia
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@router.post("/summaries/executive")
async def generate_executive_summary(request: ExecutiveSummaryRequest):
    """Generate executive summary with key findings and recommendations"""
    try:
        # Generate executive summary
        summary = summary_generator.generate_executive_summary(
            request.data,
            SummaryType(request.summary_type),
            request.focus_areas
        )
        
        # Format for API response
        return {
            "summary": {
                "title": summary.title,
                "summary_type": summary.summary_type.value,
                "executive_overview": summary.executive_overview,
                "key_findings": [
                    {
                        "title": finding.title,
                        "description": finding.description,
                        "impact": finding.impact,
                        "priority": finding.priority.value,
                        "category": finding.category,
                        "metrics": finding.metrics,
                        "trend": finding.trend,
                        "confidence": finding.confidence
                    }
                    for finding in summary.key_findings
                ],
                "recommendations": [
                    {
                        "title": rec.title,
                        "description": rec.description,
                        "rationale": rec.rationale,
                        "expected_impact": rec.expected_impact,
                        "implementation_effort": rec.implementation_effort,
                        "timeline": rec.timeline,
                        "priority": rec.priority.value,
                        "success_metrics": rec.success_metrics,
                        "risks": rec.risks
                    }
                    for rec in summary.recommendations
                ],
                "performance_highlights": summary.performance_highlights,
                "risk_alerts": summary.risk_alerts,
                "next_steps": summary.next_steps,
                "confidence_score": summary.confidence_score,
                "generated_at": summary.generated_at.isoformat(),
                "data_period": {
                    "start_date": summary.data_period['start_date'].isoformat(),
                    "end_date": summary.data_period['end_date'].isoformat()
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executive summary generation failed: {str(e)}")

@router.get("/summaries/executive/{summary_id}/formatted")
async def get_formatted_executive_summary(summary_id: str):
    """Get formatted executive summary for presentation"""
    try:
        # In a real implementation, you would retrieve the summary from storage
        # For now, we'll generate a sample summary
        sample_data = await _get_sample_data()
        
        summary = summary_generator.generate_executive_summary(
            sample_data,
            SummaryType.COMPREHENSIVE
        )
        
        formatted_summary = summary_generator.format_for_presentation(summary)
        
        return {
            "formatted_summary": formatted_summary,
            "summary_id": summary_id,
            "format": "markdown"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get formatted summary: {str(e)}")

@router.get("/dashboard/mobile")
async def get_mobile_dashboard_data():
    """Get mobile-optimized dashboard data"""
    try:
        # Generate sample data for mobile dashboard
        sample_data = await _get_sample_data()
        
        # Generate executive summary
        summary = summary_generator.generate_executive_summary(
            sample_data,
            SummaryType.COMPREHENSIVE
        )
        
        # Format for mobile dashboard
        mobile_data = {
            "metrics": [
                {
                    "id": "1",
                    "title": "Revenue",
                    "value": "$2.4M",
                    "change": "+12.5%",
                    "trend": "up",
                    "category": "financial",
                    "priority": "high"
                },
                {
                    "id": "2",
                    "title": "Active Users",
                    "value": "45.2K",
                    "change": "+8.3%",
                    "trend": "up",
                    "category": "engagement",
                    "priority": "medium"
                },
                {
                    "id": "3",
                    "title": "System Uptime",
                    "value": "98.7%",
                    "change": "-0.3%",
                    "trend": "down",
                    "category": "performance",
                    "priority": "high"
                },
                {
                    "id": "4",
                    "title": "ROI",
                    "value": "24.8%",
                    "change": "+3.2%",
                    "trend": "up",
                    "category": "financial",
                    "priority": "high"
                }
            ],
            "key_findings": [
                {
                    "id": str(i),
                    "title": finding.title,
                    "description": finding.description,
                    "impact": finding.impact,
                    "priority": finding.priority.value,
                    "category": finding.category
                }
                for i, finding in enumerate(summary.key_findings[:5], 1)
            ],
            "recommendations": [
                {
                    "id": str(i),
                    "title": rec.title,
                    "description": rec.description,
                    "timeline": rec.timeline,
                    "priority": rec.priority.value,
                    "implementation_effort": rec.implementation_effort
                }
                for i, rec in enumerate(summary.recommendations[:5], 1)
            ],
            "risk_alerts": summary.risk_alerts,
            "last_updated": datetime.now().isoformat(),
            "confidence_score": summary.confidence_score
        }
        
        return mobile_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get mobile dashboard data: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "reporting_engine": "operational",
            "scheduler": "operational" if scheduler.is_running else "stopped",
            "statistical_engine": "operational",
            "summary_generator": "operational"
        }
    }

# Helper functions
async def _fetch_report_data(config: ReportConfig) -> Dict[str, Any]:
    """Fetch data for report generation"""
    # Mock implementation - in reality, this would fetch from various data sources
    return await _get_sample_data()

async def _get_sample_data() -> Dict[str, Any]:
    """Get sample data for testing"""
    return {
        "metrics": {
            "revenue": 2400000,
            "revenue_growth": "12.5%",
            "active_users": 45200,
            "user_growth": "8.3%",
            "roi": "24.8%"
        },
        "system_metrics": {
            "uptime": "98.7%",
            "response_time": "150ms",
            "throughput": "1000 req/s",
            "error_rate": "0.1%"
        },
        "financial_metrics": {
            "revenue": 2400000,
            "revenue_growth": "12.5%",
            "profit_margin": "18.5%",
            "customer_acquisition_cost": 125
        },
        "engagement_metrics": {
            "active_users": 45200,
            "user_growth": "8.3%",
            "session_duration": "8.5 minutes",
            "bounce_rate": "32%"
        },
        "operational_metrics": {
            "process_efficiency": "87%",
            "automation_rate": "65%",
            "cost_per_transaction": "$0.45"
        },
        "roi_analysis": {
            "total_roi": "24.8%",
            "payback_period": "18 months",
            "net_present_value": 1250000
        }
    }