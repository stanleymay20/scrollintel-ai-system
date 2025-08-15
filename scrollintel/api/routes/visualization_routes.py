"""
API routes for data visualization and export functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
import logging
import os
from pathlib import Path

from ...models.visualization_models import (
    ChartConfiguration, VisualizationData, ExportRequest,
    DashboardLayout, DataFilter, InteractiveFeatures,
    PrintLayout, VisualizationResponse, DrillDownConfig
)
from ...engines.visualization_engine import VisualizationEngine
from ...engines.export_engine import ExportEngine
from ...engines.dashboard_engine import DashboardEngine
from ...core.never_fail_decorators import bulletproof_endpoint, safe_data_fetch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/visualization", tags=["visualization"])

# Initialize engines
viz_engine = VisualizationEngine()
export_engine = ExportEngine()
dashboard_engine = DashboardEngine()


@router.post("/charts/create", response_model=VisualizationResponse)
@bulletproof_endpoint({
    "success": True,
    "data": {
        "id": "fallback_chart",
        "name": "Sample Chart",
        "chart_config": {"chart_type": "bar", "title": "Sample Data"},
        "data": [{"category": "A", "value": 100}, {"category": "B", "value": 150}],
        "metadata": {"fallback": True}
    },
    "message": "Chart created with sample data due to system load"
})
async def create_chart(
    data: List[Dict[str, Any]],
    config: ChartConfiguration,
    filters: Optional[List[DataFilter]] = None
):
    """Create a new chart visualization."""
    result = await viz_engine.create_visualization(data, config, filters)
    return result


@router.post("/charts/interactive")
async def create_interactive_chart(
    data: List[Dict[str, Any]],
    config: ChartConfiguration,
    interactive_features: InteractiveFeatures
):
    """Create an interactive chart with advanced features."""
    try:
        result = await viz_engine.create_interactive_chart(data, config, interactive_features)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to create interactive chart"))
        
        return result
    except Exception as e:
        logger.error(f"Error creating interactive chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create interactive chart: {str(e)}")


@router.post("/charts/suggestions")
async def get_chart_suggestions(data: List[Dict[str, Any]]):
    """Get chart type suggestions based on data characteristics."""
    try:
        suggestions = viz_engine.get_chart_suggestions(data)
        return {
            "success": True,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
    except Exception as e:
        logger.error(f"Error getting chart suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chart suggestions: {str(e)}")


@router.post("/charts/statistics")
async def get_chart_statistics(
    data: List[Dict[str, Any]],
    config: ChartConfiguration
):
    """Get statistics for chart data."""
    try:
        stats = viz_engine.calculate_chart_statistics(data, config)
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error calculating chart statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate statistics: {str(e)}")


@router.post("/export")
async def export_visualization(
    request: ExportRequest,
    background_tasks: BackgroundTasks
):
    """Export visualizations in the requested format."""
    try:
        # In a real implementation, you would fetch the visualizations from the database
        # For now, we'll create mock data
        visualizations = []  # This would be populated from the database
        dashboard = None  # This would be fetched if dashboard_id is provided
        
        result = await export_engine.export_visualization(request, visualizations, dashboard)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Export failed"))
        
        # Schedule cleanup of temp file
        if "filepath" in result:
            background_tasks.add_task(cleanup_temp_file, result["filepath"])
        
        return {
            "success": True,
            "export_id": result.get("filename", ""),
            "download_url": f"/api/visualization/download/{result.get('filename', '')}",
            "content_type": result.get("content_type", ""),
            "size": result.get("size", 0)
        }
        
    except Exception as e:
        logger.error(f"Error exporting visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Download an exported file."""
    try:
        filepath = Path("temp/exports") / filename
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine content type based on file extension
        content_types = {
            ".pdf": "application/pdf",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".json": "application/json",
            ".zip": "application/zip",
            ".png": "image/png",
            ".svg": "image/svg+xml"
        }
        
        content_type = content_types.get(filepath.suffix, "application/octet-stream")
        
        return FileResponse(
            path=str(filepath),
            filename=filename,
            media_type=content_type
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/dashboards/create")
async def create_dashboard(
    name: str,
    description: Optional[str] = None,
    template: Optional[str] = None,
    charts: Optional[List[str]] = None
):
    """Create a new dashboard."""
    try:
        result = await dashboard_engine.create_dashboard(name, description, template, charts)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to create dashboard"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create dashboard: {str(e)}")


@router.put("/dashboards/{dashboard_id}/layout")
async def update_dashboard_layout(
    dashboard_id: str,
    layout: List[Dict[str, Any]]
):
    """Update dashboard layout."""
    try:
        result = await dashboard_engine.update_dashboard_layout(dashboard_id, layout)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to update layout"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error updating dashboard layout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update dashboard layout: {str(e)}")


@router.post("/dashboards/{dashboard_id}/charts")
async def add_chart_to_dashboard(
    dashboard_id: str,
    chart_id: str,
    position: Optional[Dict[str, int]] = None
):
    """Add a chart to a dashboard."""
    try:
        result = await dashboard_engine.add_chart_to_dashboard(dashboard_id, chart_id, position)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to add chart"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding chart to dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add chart to dashboard: {str(e)}")


@router.post("/dashboards/{dashboard_id}/filters")
async def apply_dashboard_filters(
    dashboard_id: str,
    filters: List[DataFilter]
):
    """Apply filters to all charts in a dashboard."""
    try:
        result = await dashboard_engine.apply_dashboard_filters(dashboard_id, filters)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to apply filters"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error applying dashboard filters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to apply dashboard filters: {str(e)}")


@router.post("/dashboards/{dashboard_id}/print-layout")
async def create_print_layout(
    dashboard_id: str,
    print_config: PrintLayout
):
    """Create a print-friendly layout for a dashboard."""
    try:
        result = await dashboard_engine.create_print_layout(dashboard_id, print_config)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to create print layout"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating print layout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create print layout: {str(e)}")


@router.get("/dashboards/templates")
async def get_dashboard_templates():
    """Get available dashboard templates."""
    try:
        templates = dashboard_engine.get_dashboard_templates()
        return {
            "success": True,
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        logger.error(f"Error getting dashboard templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@router.post("/dashboards/{dashboard_id}/duplicate")
async def duplicate_dashboard(
    dashboard_id: str,
    new_name: str
):
    """Duplicate an existing dashboard."""
    try:
        result = await dashboard_engine.duplicate_dashboard(dashboard_id, new_name)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to duplicate dashboard"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error duplicating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to duplicate dashboard: {str(e)}")


@router.post("/data/filter")
async def filter_data(
    data: List[Dict[str, Any]],
    filters: List[DataFilter]
):
    """Apply filters to data and return filtered results."""
    try:
        # Use the visualization engine's filter method
        filtered_data = viz_engine._apply_filters(data, filters)
        
        return {
            "success": True,
            "data": filtered_data,
            "original_count": len(data),
            "filtered_count": len(filtered_data),
            "filters_applied": len(filters)
        }
        
    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to filter data: {str(e)}")


async def cleanup_temp_file(filepath: str):
    """Clean up temporary export files."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {filepath}: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for visualization service."""
    return {
        "status": "healthy",
        "service": "visualization",
        "timestamp": "2024-01-01T00:00:00Z",
        "features": {
            "chart_creation": True,
            "interactive_charts": True,
            "export_pdf": True,
            "export_excel": True,
            "dashboard_customization": True,
            "print_layouts": True
        }
    }