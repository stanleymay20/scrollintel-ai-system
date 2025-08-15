"""
API routes for ScrollViz visualization engine.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json
import io
from datetime import datetime

from scrollintel.engines.scroll_viz_engine import ScrollVizEngine, ChartType, ExportFormat
from scrollintel.security.auth import get_current_user
from scrollintel.security.audit import audit_logger
from scrollintel.models.database import User

router = APIRouter(prefix="/api/v1/viz", tags=["visualization"])

# Global engine instance
viz_engine = None


async def get_viz_engine() -> ScrollVizEngine:
    """Get or create ScrollViz engine instance."""
    global viz_engine
    if viz_engine is None:
        viz_engine = ScrollVizEngine()
        await viz_engine.start()
    return viz_engine


@router.post("/recommend-chart-type")
async def recommend_chart_type(
    data: Dict[str, Any],
    columns: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    engine: ScrollVizEngine = Depends(get_viz_engine)
):
    """
    Recommend chart type based on data characteristics.
    
    Args:
        data: Dataset as dictionary
        columns: Specific columns to analyze (optional)
        
    Returns:
        Chart type recommendations with reasoning
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Prepare parameters
        params = {
            "action": "recommend_chart_type",
            "columns": columns or df.columns.tolist()
        }
        
        # Get recommendation
        result = await engine.execute(df, params)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="chart_type_recommendation",
            resource_type="visualization",
            details={"columns": params["columns"], "recommendation": result["recommended_type"]}
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart recommendation failed: {str(e)}")


@router.post("/generate-chart")
async def generate_chart(
    data: Dict[str, Any],
    chart_type: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    title: Optional[str] = None,
    theme: str = "plotly_white",
    width: int = 800,
    height: int = 600,
    show_legend: bool = True,
    include_recharts: bool = False,
    include_vega: bool = False,
    current_user: User = Depends(get_current_user),
    engine: ScrollVizEngine = Depends(get_viz_engine)
):
    """
    Generate a chart from data.
    
    Args:
        data: Dataset as dictionary
        chart_type: Type of chart to generate (optional, will auto-recommend if not provided)
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Column for color encoding (optional)
        size_column: Column for size encoding (optional)
        title: Chart title
        theme: Plotly theme
        width: Chart width in pixels
        height: Chart height in pixels
        show_legend: Whether to show legend
        include_recharts: Include Recharts configuration
        include_vega: Include Vega-Lite specification
        
    Returns:
        Generated chart with multiple format options
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Prepare parameters
        params = {
            "action": "generate_chart",
            "chart_type": chart_type,
            "x_column": x_column or (df.columns[0] if len(df.columns) > 0 else None),
            "y_column": y_column or (df.columns[1] if len(df.columns) > 1 else None),
            "color_column": color_column,
            "size_column": size_column,
            "title": title or f"{chart_type or 'Auto'} Chart",
            "theme": theme,
            "width": width,
            "height": height,
            "show_legend": show_legend,
            "include_recharts": include_recharts,
            "include_vega": include_vega
        }
        
        # Generate chart
        result = await engine.execute(df, params)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="chart_generation",
            resource_type="visualization",
            details={
                "chart_type": result["chart_type"],
                "columns": [x_column, y_column, color_column, size_column],
                "data_rows": len(df)
            }
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@router.post("/create-dashboard")
async def create_dashboard(
    dashboard_request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    engine: ScrollVizEngine = Depends(get_viz_engine)
):
    """
    Create an interactive dashboard with multiple visualizations.
    
    Args:
        dashboard_request: Dashboard configuration with data and chart specifications
        
    Returns:
        Dashboard with multiple charts and real-time configuration
    """
    try:
        data = dashboard_request.get("data")
        dashboard_config = dashboard_request.get("config", {})
        
        # Prepare parameters
        params = {
            "action": "create_dashboard",
            "dashboard_config": dashboard_config
        }
        
        # Create dashboard
        result = await engine.execute(data, params)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="dashboard_creation",
            resource_type="dashboard",
            details={
                "dashboard_id": result["id"],
                "chart_count": len(result["charts"]),
                "title": result["title"]
            }
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard creation failed: {str(e)}")


@router.post("/export-visualization")
async def export_visualization(
    chart_data: Dict[str, Any],
    format: str = "png",
    width: int = 800,
    height: int = 600,
    current_user: User = Depends(get_current_user),
    engine: ScrollVizEngine = Depends(get_viz_engine)
):
    """
    Export visualization in various formats (PNG, SVG, PDF, HTML, JSON).
    
    Args:
        chart_data: Chart data from previous generation
        format: Export format (png, svg, pdf, html, json)
        width: Export width in pixels
        height: Export height in pixels
        
    Returns:
        Exported visualization data
    """
    try:
        # Validate format
        if format not in [f.value for f in ExportFormat]:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Prepare parameters
        params = {
            "action": "export_visualization",
            "format": format,
            "width": width,
            "height": height
        }
        
        # Export visualization
        result = await engine.execute(chart_data, params)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="visualization_export",
            resource_type="visualization",
            details={
                "format": format,
                "dimensions": f"{width}x{height}"
            }
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization export failed: {str(e)}")


@router.get("/templates")
async def get_visualization_templates(
    current_user: User = Depends(get_current_user),
    engine: ScrollVizEngine = Depends(get_viz_engine)
):
    """
    Get available visualization templates.
    
    Returns:
        List of available visualization templates
    """
    try:
        # Get templates
        result = await engine.execute(None, {"action": "get_templates"})
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@router.get("/chart-types")
async def get_supported_chart_types():
    """
    Get list of supported chart types.
    
    Returns:
        List of supported chart types with descriptions
    """
    chart_types = [
        {"type": ChartType.BAR.value, "description": "Bar chart for categorical data comparison"},
        {"type": ChartType.LINE.value, "description": "Line chart for trends over time"},
        {"type": ChartType.SCATTER.value, "description": "Scatter plot for relationship analysis"},
        {"type": ChartType.PIE.value, "description": "Pie chart for proportional data"},
        {"type": ChartType.HISTOGRAM.value, "description": "Histogram for data distribution"},
        {"type": ChartType.BOX.value, "description": "Box plot for statistical distribution"},
        {"type": ChartType.VIOLIN.value, "description": "Violin plot for distribution shape"},
        {"type": ChartType.HEATMAP.value, "description": "Heatmap for matrix data visualization"},
        {"type": ChartType.AREA.value, "description": "Area chart for cumulative data"},
        {"type": ChartType.BUBBLE.value, "description": "Bubble chart for multi-dimensional data"},
        {"type": ChartType.TREEMAP.value, "description": "Treemap for hierarchical data"},
        {"type": ChartType.SUNBURST.value, "description": "Sunburst for nested categorical data"},
        {"type": ChartType.FUNNEL.value, "description": "Funnel chart for process flow"},
        {"type": ChartType.WATERFALL.value, "description": "Waterfall chart for cumulative effects"},
        {"type": ChartType.CANDLESTICK.value, "description": "Candlestick chart for financial data"},
        {"type": ChartType.RADAR.value, "description": "Radar chart for multi-variable comparison"},
        {"type": ChartType.PARALLEL_COORDINATES.value, "description": "Parallel coordinates for high-dimensional data"},
        {"type": ChartType.SANKEY.value, "description": "Sankey diagram for flow visualization"}
    ]
    
    return {
        "success": True,
        "data": {
            "chart_types": chart_types,
            "total_count": len(chart_types)
        }
    }


@router.get("/export-formats")
async def get_supported_export_formats():
    """
    Get list of supported export formats.
    
    Returns:
        List of supported export formats
    """
    export_formats = [
        {"format": ExportFormat.PNG.value, "description": "PNG image format", "mime_type": "image/png"},
        {"format": ExportFormat.SVG.value, "description": "SVG vector format", "mime_type": "image/svg+xml"},
        {"format": ExportFormat.PDF.value, "description": "PDF document format", "mime_type": "application/pdf"},
        {"format": ExportFormat.HTML.value, "description": "HTML interactive format", "mime_type": "text/html"},
        {"format": ExportFormat.JSON.value, "description": "JSON data format", "mime_type": "application/json"}
    ]
    
    return {
        "success": True,
        "data": {
            "export_formats": export_formats,
            "total_count": len(export_formats)
        }
    }


@router.post("/upload-and-visualize")
async def upload_and_visualize(
    file: UploadFile = File(...),
    chart_type: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    engine: ScrollVizEngine = Depends(get_viz_engine)
):
    """
    Upload a file and automatically create visualization.
    
    Args:
        file: Uploaded data file (CSV, Excel, JSON)
        chart_type: Preferred chart type (optional)
        x_column: Column for x-axis (optional)
        y_column: Column for y-axis (optional)
        
    Returns:
        Generated visualization from uploaded data
    """
    try:
        # Read file content
        content = await file.read()
        
        # Parse based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate chart
        params = {
            "action": "generate_chart",
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "title": f"Visualization from {file.filename}"
        }
        
        result = await engine.execute(df, params)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="file_upload_visualization",
            resource_type="visualization",
            details={
                "filename": file.filename,
                "chart_type": result["chart_type"],
                "data_rows": len(df),
                "data_columns": len(df.columns)
            }
        )
        
        return {
            "success": True,
            "data": {
                "visualization": result,
                "data_info": {
                    "filename": file.filename,
                    "rows": len(df),
                    "columns": df.columns.tolist(),
                    "data_types": df.dtypes.to_dict()
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload and visualization failed: {str(e)}")


@router.get("/status")
async def get_viz_engine_status(
    current_user: User = Depends(get_current_user),
    engine: ScrollVizEngine = Depends(get_viz_engine)
):
    """
    Get ScrollViz engine status and metrics.
    
    Returns:
        Engine status and performance metrics
    """
    try:
        status = engine.get_status()
        metrics = engine.get_metrics()
        
        return {
            "success": True,
            "data": {
                "status": status,
                "metrics": metrics
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get engine status: {str(e)}")


@router.post("/shutdown")
async def shutdown_viz_engine(
    current_user: User = Depends(get_current_user)
):
    """
    Shutdown ScrollViz engine (admin only).
    
    Returns:
        Shutdown confirmation
    """
    try:
        global viz_engine
        if viz_engine:
            await viz_engine.stop()
            viz_engine = None
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="engine_shutdown",
            resource_type="engine",
            details={"engine": "scroll_viz"}
        )
        
        return {
            "success": True,
            "message": "ScrollViz engine shutdown successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine shutdown failed: {str(e)}")