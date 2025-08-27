"""
API Routes for Advanced Reporting and Analytics

This module provides REST API endpoints for the advanced reporting and analytics
functionality including comprehensive reporting, scheduling, and interactive reports.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, File, UploadFile
from fastapi.responses import Response, HTMLResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from pydantic import BaseModel, Field

from ...engines.comprehensive_reporting_engine import (
    ComprehensiveReportingEngine, ReportConfig, ReportType, ReportFormat, GeneratedReport
)
from ...core.automated_report_scheduler import (
    AutomatedReportScheduler, ReportSchedule, ScheduleConfig, DeliveryConfig,
    ScheduleFrequency, DeliveryMethod, ScheduleStatus
)
from ...engines.advanced_statistical_analytics import (
    AdvancedStatisticalAnalytics, AnalysisConfig, AnalysisType
)
from ...engines.executive_summary_generator import (
    ExecutiveSummaryGenerator, SummaryType, ExecutiveSummary
)
from ...engines.interactive_report_builder import (
    InteractiveReportBuilder, ComponentType, ChartType, LayoutType,
    ComponentConfig, ChartConfig, ReportLayout
)

logger = logging.getLogger(__name__)

# Initialize engines
reporting_engine = ComprehensiveReportingEngine()
scheduler = AutomatedReportScheduler(reporting_engine)
analytics_engine = AdvancedStatisticalAnalytics()
summary_generator = ExecutiveSummaryGenerator()
report_builder = InteractiveReportBuilder()

router = APIRouter(prefix="/api/v1/advanced-analytics", tags=["Advanced Analytics"])


# Pydantic models for request/response
class ReportGenerationRequest(BaseModel):
    report_type: str = Field(..., description="Type of report to generate")
    format: str = Field(..., description="Output format (pdf, excel, web, json, csv)")
    title: str = Field(..., description="Report title")
    description: str = Field(..., description="Report description")
    data_sources: List[str] = Field(default=[], description="Data sources to include")
    filters: Dict[str, Any] = Field(default={}, description="Filters to apply")
    date_range: Dict[str, str] = Field(default={}, description="Date range for data")
    template_id: Optional[str] = Field(None, description="Template ID to use")
    custom_sections: Optional[List[Dict]] = Field(None, description="Custom sections")


class ScheduleCreationRequest(BaseModel):
    name: str = Field(..., description="Schedule name")
    description: str = Field(..., description="Schedule description")
    report_config: Dict[str, Any] = Field(..., description="Report configuration")
    frequency: str = Field(..., description="Schedule frequency")
    start_date: str = Field(..., description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")
    time_of_day: Optional[str] = Field(None, description="Time of day (HH:MM)")
    day_of_week: Optional[int] = Field(None, description="Day of week (0=Monday)")
    day_of_month: Optional[int] = Field(None, description="Day of month")
    delivery_method: str = Field(..., description="Delivery method")
    recipients: List[str] = Field(..., description="Delivery recipients")
    delivery_settings: Dict[str, Any] = Field(default={}, description="Delivery settings")


class AnalysisRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Data to analyze")
    analysis_types: List[str] = Field(..., description="Types of analysis to perform")
    confidence_level: float = Field(0.95, description="Confidence level")
    significance_threshold: float = Field(0.05, description="Significance threshold")
    min_data_points: int = Field(30, description="Minimum data points required")


class SummaryGenerationRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Analytics data and insights")
    summary_type: str = Field("performance_overview", description="Type of summary")
    target_audience: str = Field("executive", description="Target audience")
    custom_focus: Optional[List[str]] = Field(None, description="Custom focus areas")


class InteractiveReportRequest(BaseModel):
    name: str = Field(..., description="Report name")
    description: str = Field(..., description="Report description")
    layout_type: str = Field("single_column", description="Layout type")
    template_id: Optional[str] = Field(None, description="Template ID")


class ComponentRequest(BaseModel):
    component_type: str = Field(..., description="Component type")
    title: str = Field(..., description="Component title")
    position: Dict[str, int] = Field(..., description="Position and size")
    properties: Optional[Dict[str, Any]] = Field(None, description="Component properties")


# Report Generation Endpoints

@router.post("/reports/generate")
async def generate_report(request: ReportGenerationRequest):
    """Generate a comprehensive report"""
    try:
        # Parse date range
        date_range = {}
        if request.date_range.get('start'):
            date_range['start'] = datetime.fromisoformat(request.date_range['start'])
        if request.date_range.get('end'):
            date_range['end'] = datetime.fromisoformat(request.date_range['end'])
        
        # Create report configuration
        config = ReportConfig(
            report_type=ReportType(request.report_type),
            format=ReportFormat(request.format),
            title=request.title,
            description=request.description,
            data_sources=request.data_sources,
            filters=request.filters,
            date_range=date_range,
            template_id=request.template_id,
            custom_sections=request.custom_sections
        )
        
        # Generate report
        report = await reporting_engine.generate_report(config)
        
        # Return appropriate response based on format
        if config.format in [ReportFormat.PDF, ReportFormat.EXCEL]:
            return Response(
                content=report.content,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={report.report_id}.{config.format.value}"}
            )
        elif config.format == ReportFormat.WEB:
            return HTMLResponse(content=report.content.decode('utf-8'))
        else:
            return {
                "report_id": report.report_id,
                "content": report.content.decode('utf-8'),
                "metadata": report.metadata,
                "generated_at": report.generated_at.isoformat(),
                "file_size": report.file_size
            }
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get a generated report by ID"""
    try:
        report = await reporting_engine.get_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "report_id": report.report_id,
            "title": report.config.title,
            "type": report.config.report_type.value,
            "format": report.format.value,
            "generated_at": report.generated_at.isoformat(),
            "file_size": report.file_size,
            "metadata": report.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports")
async def list_reports(
    report_type: Optional[str] = Query(None, description="Filter by report type"),
    format: Optional[str] = Query(None, description="Filter by format")
):
    """List all generated reports"""
    try:
        filters = {}
        if report_type:
            filters['type'] = report_type
        if format:
            filters['format'] = format
        
        reports = await reporting_engine.list_reports(filters)
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Report Scheduling Endpoints

@router.post("/schedules")
async def create_schedule(request: ScheduleCreationRequest):
    """Create a new report schedule"""
    try:
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date) if request.end_date else None
        
        # Create schedule configuration
        schedule_config = ScheduleConfig(
            frequency=ScheduleFrequency(request.frequency),
            start_date=start_date,
            end_date=end_date,
            time_of_day=request.time_of_day,
            day_of_week=request.day_of_week,
            day_of_month=request.day_of_month
        )
        
        # Create delivery configuration
        delivery_config = DeliveryConfig(
            method=DeliveryMethod(request.delivery_method),
            recipients=request.recipients,
            settings=request.delivery_settings
        )
        
        # Create schedule
        schedule = ReportSchedule(
            schedule_id="",  # Will be generated
            name=request.name,
            description=request.description,
            report_config=request.report_config,
            schedule_config=schedule_config,
            delivery_config=delivery_config,
            status=ScheduleStatus.ACTIVE,
            created_at=datetime.utcnow()
        )
        
        schedule_id = await scheduler.create_schedule(schedule)
        
        return {
            "schedule_id": schedule_id,
            "message": "Schedule created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules")
async def list_schedules(status: Optional[str] = Query(None, description="Filter by status")):
    """List all report schedules"""
    try:
        schedule_status = ScheduleStatus(status) if status else None
        schedules = await scheduler.list_schedules(schedule_status)
        return {"schedules": schedules}
        
    except Exception as e:
        logger.error(f"Error listing schedules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules/{schedule_id}")
async def get_schedule(schedule_id: str):
    """Get a schedule by ID"""
    try:
        schedule = await scheduler.get_schedule(schedule_id)
        
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {
            "schedule_id": schedule.schedule_id,
            "name": schedule.name,
            "description": schedule.description,
            "status": schedule.status.value,
            "frequency": schedule.schedule_config.frequency.value,
            "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
            "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
            "run_count": schedule.run_count,
            "failure_count": schedule.failure_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/schedules/{schedule_id}/pause")
async def pause_schedule(schedule_id: str):
    """Pause a schedule"""
    try:
        success = await scheduler.pause_schedule(schedule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"message": "Schedule paused successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/schedules/{schedule_id}/resume")
async def resume_schedule(schedule_id: str):
    """Resume a paused schedule"""
    try:
        success = await scheduler.resume_schedule(schedule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"message": "Schedule resumed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a schedule"""
    try:
        success = await scheduler.delete_schedule(schedule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"message": "Schedule deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules/{schedule_id}/executions")
async def get_execution_history(schedule_id: str):
    """Get execution history for a schedule"""
    try:
        executions = await scheduler.get_execution_history(schedule_id)
        return {"executions": executions}
        
    except Exception as e:
        logger.error(f"Error getting execution history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistical Analysis Endpoints

@router.post("/analysis/statistical")
async def perform_statistical_analysis(request: AnalysisRequest):
    """Perform comprehensive statistical analysis"""
    try:
        # Convert data to pandas DataFrame if needed
        import pandas as pd
        
        if isinstance(request.data, dict) and 'dataframe' in request.data:
            df = pd.DataFrame(request.data['dataframe'])
        else:
            df = pd.DataFrame(request.data)
        
        # Create analysis configuration
        config = AnalysisConfig(
            analysis_types=[AnalysisType(at) for at in request.analysis_types],
            confidence_level=request.confidence_level,
            significance_threshold=request.significance_threshold,
            min_data_points=request.min_data_points
        )
        
        # Perform analysis
        results = await analytics_engine.perform_comprehensive_analysis(df, config)
        
        # Convert results to JSON-serializable format
        json_results = {}
        for analysis_type, analysis_results in results.items():
            json_results[analysis_type] = [
                {
                    "analysis_type": result.analysis_type.value,
                    "metric_name": result.metric_name,
                    "result": result.result,
                    "confidence_level": result.confidence_level,
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "interpretation": result.interpretation,
                    "recommendations": result.recommendations
                }
                for result in analysis_results
            ]
        
        return {"analysis_results": json_results}
        
    except Exception as e:
        logger.error(f"Error performing statistical analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/ml-insights")
async def generate_ml_insights(request: AnalysisRequest):
    """Generate ML-powered insights from data"""
    try:
        import pandas as pd
        
        # Convert data to DataFrame
        if isinstance(request.data, dict) and 'dataframe' in request.data:
            df = pd.DataFrame(request.data['dataframe'])
        else:
            df = pd.DataFrame(request.data)
        
        # Create analysis configuration
        config = AnalysisConfig(
            analysis_types=[AnalysisType(at) for at in request.analysis_types],
            confidence_level=request.confidence_level,
            significance_threshold=request.significance_threshold,
            min_data_points=request.min_data_points
        )
        
        # Perform statistical analysis first
        statistical_results = await analytics_engine.perform_comprehensive_analysis(df, config)
        
        # Generate ML insights
        ml_insights = await analytics_engine.generate_ml_insights(df, statistical_results)
        
        # Convert to JSON-serializable format
        insights_json = [
            {
                "insight_type": insight.insight_type.value,
                "title": insight.title,
                "description": insight.description,
                "confidence_score": insight.confidence_score,
                "impact_score": insight.impact_score,
                "data_points": insight.data_points,
                "visualizations": insight.visualizations,
                "action_items": insight.action_items,
                "metadata": insight.metadata
            }
            for insight in ml_insights
        ]
        
        return {"ml_insights": insights_json}
        
    except Exception as e:
        logger.error(f"Error generating ML insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Executive Summary Endpoints

@router.post("/summaries/generate")
async def generate_executive_summary(request: SummaryGenerationRequest):
    """Generate executive summary with natural language explanations"""
    try:
        # Generate summary
        summary = await summary_generator.generate_executive_summary(
            data=request.data,
            summary_type=SummaryType(request.summary_type),
            target_audience=request.target_audience,
            custom_focus=request.custom_focus
        )
        
        # Convert to JSON-serializable format
        summary_json = {
            "title": summary.title,
            "executive_overview": summary.executive_overview,
            "key_highlights": summary.key_highlights,
            "critical_insights": [
                {
                    "title": insight.title,
                    "summary": insight.summary,
                    "detailed_explanation": insight.detailed_explanation,
                    "key_metrics": insight.key_metrics,
                    "urgency": insight.urgency.value,
                    "confidence_level": insight.confidence_level,
                    "business_impact": insight.business_impact,
                    "recommended_actions": insight.recommended_actions,
                    "visualization_suggestions": insight.visualization_suggestions
                }
                for insight in summary.critical_insights
            ],
            "performance_metrics": summary.performance_metrics,
            "trends_and_patterns": summary.trends_and_patterns,
            "risks_and_opportunities": summary.risks_and_opportunities,
            "strategic_recommendations": summary.strategic_recommendations,
            "next_steps": summary.next_steps,
            "generated_at": summary.generated_at.isoformat(),
            "data_period": {
                "start": summary.data_period.get("start").isoformat() if summary.data_period.get("start") else None,
                "end": summary.data_period.get("end").isoformat() if summary.data_period.get("end") else None
            },
            "confidence_score": summary.confidence_score
        }
        
        return {"executive_summary": summary_json}
        
    except Exception as e:
        logger.error(f"Error generating executive summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Interactive Report Builder Endpoints

@router.post("/interactive-reports")
async def create_interactive_report(request: InteractiveReportRequest, created_by: str = "user"):
    """Create a new interactive report"""
    try:
        report_id = await report_builder.create_report(
            name=request.name,
            description=request.description,
            created_by=created_by,
            layout_type=LayoutType(request.layout_type),
            template_id=request.template_id
        )
        
        return {
            "report_id": report_id,
            "message": "Interactive report created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating interactive report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interactive-reports")
async def list_interactive_reports(created_by: Optional[str] = Query(None)):
    """List all interactive reports"""
    try:
        reports = await report_builder.list_reports(created_by)
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Error listing interactive reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interactive-reports/{report_id}")
async def get_interactive_report(report_id: str):
    """Get an interactive report by ID"""
    try:
        report = await report_builder.get_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        report_json = await report_builder.generate_report_json(report_id)
        return {"report": report_json}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting interactive report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interactive-reports/{report_id}/html")
async def get_interactive_report_html(report_id: str):
    """Get HTML representation of interactive report"""
    try:
        html_content = await report_builder.generate_report_html(report_id)
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error generating report HTML: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interactive-reports/{report_id}/components")
async def add_report_component(report_id: str, request: ComponentRequest):
    """Add a component to an interactive report"""
    try:
        component_id = await report_builder.add_component(
            report_id=report_id,
            component_type=ComponentType(request.component_type),
            title=request.title,
            position=request.position,
            properties=request.properties
        )
        
        return {
            "component_id": component_id,
            "message": "Component added successfully"
        }
        
    except Exception as e:
        logger.error(f"Error adding component: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/interactive-reports/{report_id}/components/{component_id}")
async def update_report_component(report_id: str, component_id: str, updates: Dict[str, Any]):
    """Update a component in an interactive report"""
    try:
        success = await report_builder.update_component(
            report_id=report_id,
            component_id=component_id,
            updates=updates
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Component not found")
        
        return {"message": "Component updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating component: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/interactive-reports/{report_id}/components/{component_id}")
async def remove_report_component(report_id: str, component_id: str):
    """Remove a component from an interactive report"""
    try:
        success = await report_builder.remove_component(
            report_id=report_id,
            component_id=component_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Component not found")
        
        return {"message": "Component removed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing component: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/interactive-reports/{report_id}")
async def delete_interactive_report(report_id: str):
    """Delete an interactive report"""
    try:
        success = await report_builder.delete_report(report_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {"message": "Report deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility Endpoints

@router.get("/formats")
async def get_supported_formats():
    """Get list of supported report formats"""
    try:
        formats = reporting_engine.get_supported_formats()
        return {"supported_formats": formats}
        
    except Exception as e:
        logger.error(f"Error getting supported formats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report-types")
async def get_report_types():
    """Get list of available report types"""
    try:
        types = reporting_engine.get_report_types()
        return {"report_types": types}
        
    except Exception as e:
        logger.error(f"Error getting report types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/component-library")
async def get_component_library():
    """Get the interactive report component library"""
    try:
        library = report_builder.get_component_library()
        return {"component_library": library}
        
    except Exception as e:
        logger.error(f"Error getting component library: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart-types")
async def get_chart_types():
    """Get list of supported chart types"""
    try:
        chart_types = report_builder.get_supported_chart_types()
        return {"chart_types": chart_types}
        
    except Exception as e:
        logger.error(f"Error getting chart types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Start scheduler when module is imported
@router.on_event("startup")
async def startup_event():
    """Start the report scheduler on startup"""
    scheduler.start_scheduler()
    logger.info("Report scheduler started")


@router.on_event("shutdown")
async def shutdown_event():
    """Stop the report scheduler on shutdown"""
    scheduler.stop_scheduler()
    logger.info("Report scheduler stopped")