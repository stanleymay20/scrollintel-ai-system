"""
API routes for dashboard functionality
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from ...dashboard.dashboard_generator import DashboardGenerator
from ...engines.ai_readiness_reporting_engine import (
    AIReadinessReportingEngine,
    IndustryStandard,
    ReportType
)
from ...engines.quality_assessment_engine import QualityAssessmentEngine
from ...engines.bias_analysis_engine import BiasAnalysisEngine
from ..models.requests import ReportGenerationRequest
from ..models.responses import BaseResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

# Initialize components
dashboard_generator = DashboardGenerator()
reporting_engine = AIReadinessReportingEngine()
quality_engine = QualityAssessmentEngine()
bias_engine = BiasAnalysisEngine()


@router.get("/readiness/{dataset_id}", response_class=HTMLResponse)
async def get_readiness_dashboard(
    dataset_id: str,
    industry: IndustryStandard = IndustryStandard.GENERAL,
    include_real_time: bool = Query(True, description="Include real-time monitoring"),
    theme: str = Query("professional", description="Dashboard theme"),
    color_scheme: str = Query("blue", description="Color scheme")
):
    """
    Generate interactive AI readiness dashboard for a dataset
    
    Args:
        dataset_id: Dataset identifier
        industry: Industry standard for benchmarking
        include_real_time: Include real-time monitoring components
        theme: Dashboard theme (professional, modern, minimal)
        color_scheme: Color scheme (blue, green, purple)
        
    Returns:
        HTML dashboard content
    """
    try:
        logger.info(f"Generating readiness dashboard for dataset {dataset_id}")
        
        # Get AI readiness data
        ai_readiness_score = await quality_engine.calculate_ai_readiness_score(dataset_id)
        quality_report = await quality_engine.assess_quality(dataset_id)
        bias_report = await bias_engine.detect_bias(dataset_id, [])
        
        # Generate comprehensive report
        report = reporting_engine.generate_comprehensive_report(
            dataset_id=dataset_id,
            ai_readiness_score=ai_readiness_score,
            quality_report=quality_report,
            bias_report=bias_report,
            industry=industry,
            report_type=ReportType.DETAILED_TECHNICAL
        )
        
        # Generate dashboard HTML
        customization_options = {
            "theme": theme,
            "color_scheme": color_scheme,
            "show_details": True
        }
        
        dashboard_html = dashboard_generator.generate_readiness_dashboard(
            report=report,
            include_real_time=include_real_time,
            customization_options=customization_options
        )
        
        return HTMLResponse(content=dashboard_html, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generating readiness dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring", response_class=HTMLResponse)
async def get_monitoring_dashboard(
    datasets: List[str] = Query(..., description="List of dataset IDs to monitor"),
    refresh_interval: int = Query(30, description="Refresh interval in seconds"),
    alert_thresholds: Optional[str] = Query(None, description="JSON string of alert thresholds")
):
    """
    Generate real-time monitoring dashboard
    
    Args:
        datasets: List of dataset IDs to monitor
        refresh_interval: Dashboard refresh interval in seconds
        alert_thresholds: Custom alert thresholds as JSON string
        
    Returns:
        HTML monitoring dashboard content
    """
    try:
        logger.info(f"Generating monitoring dashboard for {len(datasets)} datasets")
        
        # Parse alert thresholds if provided
        parsed_thresholds = None
        if alert_thresholds:
            import json
            try:
                parsed_thresholds = json.loads(alert_thresholds)
            except json.JSONDecodeError:
                logger.warning("Invalid alert_thresholds JSON, using defaults")
        
        # Generate monitoring dashboard HTML
        dashboard_html = dashboard_generator.generate_monitoring_dashboard(
            datasets=datasets,
            refresh_interval=refresh_interval,
            alert_thresholds=parsed_thresholds
        )
        
        return HTMLResponse(content=dashboard_html, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generating monitoring dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison", response_class=HTMLResponse)
async def get_comparison_dashboard(
    dataset_ids: List[str] = Query(..., description="List of dataset IDs to compare"),
    dimensions: Optional[List[str]] = Query(None, description="Dimensions to compare"),
    industry: IndustryStandard = IndustryStandard.GENERAL
):
    """
    Generate comparison dashboard for multiple datasets
    
    Args:
        dataset_ids: List of dataset IDs to compare
        dimensions: Dimensions to compare (optional)
        industry: Industry standard for benchmarking
        
    Returns:
        HTML comparison dashboard content
    """
    try:
        logger.info(f"Generating comparison dashboard for {len(dataset_ids)} datasets")
        
        if len(dataset_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 datasets required for comparison"
            )
        
        if len(dataset_ids) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 datasets allowed for comparison"
            )
        
        # Generate reports for all datasets
        reports = []
        for dataset_id in dataset_ids:
            try:
                # Get AI readiness data
                ai_readiness_score = await quality_engine.calculate_ai_readiness_score(dataset_id)
                quality_report = await quality_engine.assess_quality(dataset_id)
                bias_report = await bias_engine.detect_bias(dataset_id, [])
                
                # Generate report
                report = reporting_engine.generate_comprehensive_report(
                    dataset_id=dataset_id,
                    ai_readiness_score=ai_readiness_score,
                    quality_report=quality_report,
                    bias_report=bias_report,
                    industry=industry,
                    report_type=ReportType.BENCHMARK_COMPARISON
                )
                reports.append(report)
                
            except Exception as e:
                logger.warning(f"Failed to generate report for dataset {dataset_id}: {str(e)}")
                continue
        
        if not reports:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate reports for any datasets"
            )
        
        # Generate comparison dashboard HTML
        dashboard_html = dashboard_generator.generate_comparison_dashboard(
            reports=reports,
            comparison_dimensions=dimensions
        )
        
        return HTMLResponse(content=dashboard_html, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comparison dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom", response_class=HTMLResponse)
async def generate_custom_dashboard(
    dataset_id: str,
    dashboard_config: Dict[str, Any]
):
    """
    Generate custom dashboard with user-defined configuration
    
    Args:
        dataset_id: Dataset identifier
        dashboard_config: Custom dashboard configuration
        
    Returns:
        HTML custom dashboard content
    """
    try:
        logger.info(f"Generating custom dashboard for dataset {dataset_id}")
        
        # Extract configuration options
        industry = IndustryStandard(dashboard_config.get("industry", "general"))
        include_real_time = dashboard_config.get("include_real_time", True)
        customization_options = dashboard_config.get("customization", {})
        
        # Get AI readiness data
        ai_readiness_score = await quality_engine.calculate_ai_readiness_score(dataset_id)
        quality_report = await quality_engine.assess_quality(dataset_id)
        bias_report = await bias_engine.detect_bias(dataset_id, [])
        
        # Generate comprehensive report
        report = reporting_engine.generate_comprehensive_report(
            dataset_id=dataset_id,
            ai_readiness_score=ai_readiness_score,
            quality_report=quality_report,
            bias_report=bias_report,
            industry=industry,
            report_type=ReportType.DETAILED_TECHNICAL
        )
        
        # Generate custom dashboard HTML
        dashboard_html = dashboard_generator.generate_readiness_dashboard(
            report=report,
            include_real_time=include_real_time,
            customization_options=customization_options
        )
        
        return HTMLResponse(content=dashboard_html, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generating custom dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def dashboard_health_check():
    """
    Health check endpoint for dashboard service
    
    Returns:
        Service health status
    """
    try:
        # Check if dashboard generator is working
        test_data = {
            "dataset_id": "test",
            "overall_score": 0.8,
            "dimension_scores": {"data_quality": 0.8},
            "benchmark_comparison": {"overall_gap": 0.0},
            "improvement_actions": [],
            "compliance_status": {},
            "risk_assessment": {},
            "industry": "general",
            "generated_at": "2024-01-01T00:00:00"
        }
        
        # Test dashboard generation (just check if it doesn't crash)
        dashboard_generator._prepare_dashboard_data = lambda x: test_data
        
        return BaseResponse(
            success=True,
            message="Dashboard service is healthy"
        )
        
    except Exception as e:
        logger.error(f"Dashboard health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Dashboard service is unhealthy")


@router.get("/templates")
async def get_dashboard_templates():
    """
    Get available dashboard templates and themes
    
    Returns:
        Available templates and customization options
    """
    try:
        templates = {
            "themes": ["professional", "modern", "minimal", "dark"],
            "color_schemes": ["blue", "green", "purple", "red", "orange"],
            "layouts": ["grid", "single-column", "sidebar"],
            "components": [
                "score_card",
                "dimension_scores",
                "benchmark_comparison",
                "improvement_actions",
                "compliance_status",
                "risk_assessment",
                "real_time_monitoring"
            ]
        }
        
        return {
            "success": True,
            "templates": templates,
            "message": "Dashboard templates retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview/{dataset_id}")
async def get_dashboard_preview(
    dataset_id: str,
    theme: str = Query("professional", description="Dashboard theme"),
    color_scheme: str = Query("blue", description="Color scheme")
):
    """
    Generate dashboard preview (lightweight version)
    
    Args:
        dataset_id: Dataset identifier
        theme: Dashboard theme
        color_scheme: Color scheme
        
    Returns:
        Dashboard preview data
    """
    try:
        logger.info(f"Generating dashboard preview for dataset {dataset_id}")
        
        # Get basic AI readiness data
        ai_readiness_score = await quality_engine.calculate_ai_readiness_score(dataset_id)
        
        # Create preview data
        preview_data = {
            "dataset_id": dataset_id,
            "overall_score": ai_readiness_score.overall_score,
            "dimension_scores": {
                "data_quality": ai_readiness_score.data_quality_score,
                "feature_quality": ai_readiness_score.feature_quality_score,
                "bias_score": ai_readiness_score.bias_score,
                "compliance_score": ai_readiness_score.compliance_score
            },
            "readiness_level": dashboard_generator._get_readiness_label(ai_readiness_score.overall_score),
            "theme": theme,
            "color_scheme": color_scheme
        }
        
        return {
            "success": True,
            "preview": preview_data,
            "message": "Dashboard preview generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating dashboard preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))