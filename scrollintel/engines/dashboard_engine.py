"""
Dashboard customization and layout engine for ScrollIntel.
"""

import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..models.visualization_models import (
    DashboardLayout, VisualizationData, ChartConfiguration,
    DataFilter, PrintLayout
)

logger = logging.getLogger(__name__)


class DashboardEngine:
    """Advanced dashboard customization and layout engine."""
    
    def __init__(self):
        self.default_layouts = {
            "executive": {
                "name": "Executive Dashboard",
                "description": "High-level KPIs and metrics for executives",
                "layout": [
                    {"i": "kpi-1", "x": 0, "y": 0, "w": 3, "h": 2, "type": "kpi"},
                    {"i": "kpi-2", "x": 3, "y": 0, "w": 3, "h": 2, "type": "kpi"},
                    {"i": "kpi-3", "x": 6, "y": 0, "w": 3, "h": 2, "type": "kpi"},
                    {"i": "kpi-4", "x": 9, "y": 0, "w": 3, "h": 2, "type": "kpi"},
                    {"i": "chart-1", "x": 0, "y": 2, "w": 6, "h": 4, "type": "chart"},
                    {"i": "chart-2", "x": 6, "y": 2, "w": 6, "h": 4, "type": "chart"},
                    {"i": "table-1", "x": 0, "y": 6, "w": 12, "h": 3, "type": "table"}
                ]
            },
            "analytical": {
                "name": "Analytical Dashboard",
                "description": "Detailed analysis and drill-down capabilities",
                "layout": [
                    {"i": "filter-panel", "x": 0, "y": 0, "w": 3, "h": 6, "type": "filters"},
                    {"i": "main-chart", "x": 3, "y": 0, "w": 9, "h": 4, "type": "chart"},
                    {"i": "detail-chart-1", "x": 3, "y": 4, "w": 4, "h": 3, "type": "chart"},
                    {"i": "detail-chart-2", "x": 7, "y": 4, "w": 5, "h": 3, "type": "chart"},
                    {"i": "data-table", "x": 0, "y": 7, "w": 12, "h": 4, "type": "table"}
                ]
            },
            "operational": {
                "name": "Operational Dashboard",
                "description": "Real-time monitoring and alerts",
                "layout": [
                    {"i": "status-1", "x": 0, "y": 0, "w": 2, "h": 2, "type": "status"},
                    {"i": "status-2", "x": 2, "y": 0, "w": 2, "h": 2, "type": "status"},
                    {"i": "status-3", "x": 4, "y": 0, "w": 2, "h": 2, "type": "status"},
                    {"i": "alerts", "x": 6, "y": 0, "w": 6, "h": 2, "type": "alerts"},
                    {"i": "trend-1", "x": 0, "y": 2, "w": 6, "h": 3, "type": "chart"},
                    {"i": "trend-2", "x": 6, "y": 2, "w": 6, "h": 3, "type": "chart"},
                    {"i": "metrics", "x": 0, "y": 5, "w": 12, "h": 3, "type": "metrics"}
                ]
            }
        }
        
        self.widget_types = {
            "chart": {"min_w": 3, "min_h": 3, "max_w": 12, "max_h": 8},
            "kpi": {"min_w": 2, "min_h": 1, "max_w": 4, "max_h": 3},
            "table": {"min_w": 4, "min_h": 3, "max_w": 12, "max_h": 6},
            "filters": {"min_w": 2, "min_h": 4, "max_w": 4, "max_h": 8},
            "status": {"min_w": 1, "min_h": 1, "max_w": 3, "max_h": 2},
            "alerts": {"min_w": 3, "min_h": 2, "max_w": 8, "max_h": 4},
            "metrics": {"min_w": 6, "min_h": 2, "max_w": 12, "max_h": 4}
        }
    
    async def create_dashboard(
        self, 
        name: str,
        description: Optional[str] = None,
        template: Optional[str] = None,
        charts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new dashboard."""
        try:
            dashboard_id = str(uuid.uuid4())
            
            # Use template if provided
            if template and template in self.default_layouts:
                layout_config = self.default_layouts[template]
                layout = layout_config["layout"]
                if not description:
                    description = layout_config["description"]
            else:
                # Create empty layout
                layout = []
            
            dashboard = DashboardLayout(
                id=dashboard_id,
                name=name,
                description=description,
                layout=layout,
                charts=charts or [],
                filters=[],
                refresh_interval=300,
                is_public=False
            )
            
            return {
                "success": True,
                "dashboard": dashboard,
                "message": "Dashboard created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to create dashboard: {str(e)}"
            }
    
    async def update_dashboard_layout(
        self, 
        dashboard_id: str,
        layout: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update dashboard layout."""
        try:
            # Validate layout
            validation_result = self._validate_layout(layout)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"Invalid layout: {validation_result['error']}"
                }
            
            # Update dashboard (in a real implementation, this would update the database)
            updated_dashboard = {
                "id": dashboard_id,
                "layout": layout,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return {
                "success": True,
                "dashboard": updated_dashboard,
                "message": "Dashboard layout updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error updating dashboard layout: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to update dashboard layout: {str(e)}"
            }
    
    def _validate_layout(self, layout: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate dashboard layout configuration."""
        try:
            for item in layout:
                # Check required fields
                required_fields = ["i", "x", "y", "w", "h"]
                for field in required_fields:
                    if field not in item:
                        return {
                            "valid": False,
                            "error": f"Missing required field '{field}' in layout item"
                        }
                
                # Check widget type constraints
                widget_type = item.get("type", "chart")
                if widget_type in self.widget_types:
                    constraints = self.widget_types[widget_type]
                    
                    if item["w"] < constraints["min_w"] or item["w"] > constraints["max_w"]:
                        return {
                            "valid": False,
                            "error": f"Widget width {item['w']} outside allowed range for type {widget_type}"
                        }
                    
                    if item["h"] < constraints["min_h"] or item["h"] > constraints["max_h"]:
                        return {
                            "valid": False,
                            "error": f"Widget height {item['h']} outside allowed range for type {widget_type}"
                        }
                
                # Check for overlaps (simplified check)
                for other_item in layout:
                    if other_item["i"] != item["i"]:
                        if self._items_overlap(item, other_item):
                            return {
                                "valid": False,
                                "error": f"Layout items {item['i']} and {other_item['i']} overlap"
                            }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Layout validation error: {str(e)}"
            }
    
    def _items_overlap(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Check if two layout items overlap."""
        x1, y1, w1, h1 = item1["x"], item1["y"], item1["w"], item1["h"]
        x2, y2, w2, h2 = item2["x"], item2["y"], item2["w"], item2["h"]
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
    
    async def add_chart_to_dashboard(
        self, 
        dashboard_id: str,
        chart_id: str,
        position: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Add a chart to a dashboard."""
        try:
            # Find available position if not provided
            if not position:
                position = self._find_available_position("chart")
            
            # Create layout item
            layout_item = {
                "i": f"chart-{chart_id}",
                "x": position["x"],
                "y": position["y"],
                "w": position.get("w", 6),
                "h": position.get("h", 4),
                "type": "chart",
                "chart_id": chart_id
            }
            
            return {
                "success": True,
                "layout_item": layout_item,
                "message": "Chart added to dashboard successfully"
            }
            
        except Exception as e:
            logger.error(f"Error adding chart to dashboard: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to add chart to dashboard: {str(e)}"
            }
    
    def _find_available_position(self, widget_type: str) -> Dict[str, int]:
        """Find an available position for a new widget."""
        constraints = self.widget_types.get(widget_type, self.widget_types["chart"])
        
        # Simple algorithm: place at bottom
        return {
            "x": 0,
            "y": 0,  # In a real implementation, this would calculate the next available row
            "w": constraints["min_w"],
            "h": constraints["min_h"]
        }
    
    async def apply_dashboard_filters(
        self, 
        dashboard_id: str,
        filters: List[DataFilter]
    ) -> Dict[str, Any]:
        """Apply filters to all charts in a dashboard."""
        try:
            # In a real implementation, this would:
            # 1. Get all charts in the dashboard
            # 2. Apply filters to each chart's data
            # 3. Return updated chart data
            
            return {
                "success": True,
                "filters_applied": len(filters),
                "message": "Filters applied to dashboard successfully"
            }
            
        except Exception as e:
            logger.error(f"Error applying dashboard filters: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to apply dashboard filters: {str(e)}"
            }
    
    async def create_print_layout(
        self, 
        dashboard_id: str,
        print_config: PrintLayout
    ) -> Dict[str, Any]:
        """Create a print-friendly layout for a dashboard."""
        try:
            # Calculate optimal layout for printing
            print_layout = self._optimize_for_print(print_config)
            
            return {
                "success": True,
                "print_layout": print_layout,
                "message": "Print layout created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating print layout: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to create print layout: {str(e)}"
            }
    
    def _optimize_for_print(self, print_config: PrintLayout) -> Dict[str, Any]:
        """Optimize dashboard layout for printing."""
        # Calculate page dimensions based on page size and margins
        page_dimensions = self._get_page_dimensions(print_config.page_size, print_config.orientation)
        
        # Adjust for margins
        printable_width = page_dimensions["width"] - print_config.margins["left"] - print_config.margins["right"]
        printable_height = page_dimensions["height"] - print_config.margins["top"] - print_config.margins["bottom"]
        
        # Reserve space for header/footer
        if print_config.header:
            printable_height -= 0.5  # inches
        if print_config.footer:
            printable_height -= 0.5  # inches
        
        return {
            "page_size": print_config.page_size,
            "orientation": print_config.orientation,
            "printable_area": {
                "width": printable_width,
                "height": printable_height
            },
            "charts_per_page": print_config.charts_per_page,
            "include_summary": print_config.include_summary,
            "include_filters": print_config.include_filters
        }
    
    def _get_page_dimensions(self, page_size: str, orientation: str) -> Dict[str, float]:
        """Get page dimensions in inches."""
        dimensions = {
            "A4": {"width": 8.27, "height": 11.69},
            "Letter": {"width": 8.5, "height": 11.0},
            "Legal": {"width": 8.5, "height": 14.0}
        }
        
        page_dims = dimensions.get(page_size, dimensions["A4"])
        
        if orientation == "landscape":
            return {"width": page_dims["height"], "height": page_dims["width"]}
        else:
            return page_dims
    
    def get_dashboard_templates(self) -> List[Dict[str, Any]]:
        """Get available dashboard templates."""
        templates = []
        
        for template_id, template_config in self.default_layouts.items():
            templates.append({
                "id": template_id,
                "name": template_config["name"],
                "description": template_config["description"],
                "preview_image": f"/templates/{template_id}.png",  # Placeholder
                "widget_count": len(template_config["layout"]),
                "category": "built-in"
            })
        
        return templates
    
    async def duplicate_dashboard(
        self, 
        dashboard_id: str,
        new_name: str
    ) -> Dict[str, Any]:
        """Duplicate an existing dashboard."""
        try:
            # In a real implementation, this would:
            # 1. Get the original dashboard
            # 2. Create a new dashboard with the same layout and charts
            # 3. Return the new dashboard
            
            new_dashboard_id = str(uuid.uuid4())
            
            return {
                "success": True,
                "dashboard_id": new_dashboard_id,
                "message": f"Dashboard duplicated as '{new_name}'"
            }
            
        except Exception as e:
            logger.error(f"Error duplicating dashboard: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to duplicate dashboard: {str(e)}"
            }