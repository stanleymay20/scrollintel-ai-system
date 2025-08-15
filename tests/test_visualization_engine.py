"""
Tests for the visualization engine.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.visualization_engine import VisualizationEngine
from scrollintel.models.visualization_models import (
    ChartConfiguration, ChartType, DataFilter, FilterOperator,
    InteractiveFeatures, DrillDownConfig
)


@pytest.fixture
def viz_engine():
    """Create a visualization engine instance."""
    return VisualizationEngine()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {"date": "2024-01-01", "sales": 1000, "category": "A", "region": "North"},
        {"date": "2024-01-02", "sales": 1200, "category": "B", "region": "South"},
        {"date": "2024-01-03", "sales": 800, "category": "A", "region": "East"},
        {"date": "2024-01-04", "sales": 1500, "category": "C", "region": "West"},
        {"date": "2024-01-05", "sales": 900, "category": "B", "region": "North"},
    ]


@pytest.fixture
def chart_config():
    """Create a sample chart configuration."""
    return ChartConfiguration(
        chart_type=ChartType.BAR,
        title="Sales by Category",
        x_axis="category",
        y_axis="sales",
        color_scheme="default",
        width=800,
        height=400,
        interactive=True,
        show_legend=True,
        show_grid=True,
        show_tooltip=True
    )


class TestVisualizationEngine:
    """Test cases for VisualizationEngine."""
    
    @pytest.mark.asyncio
    async def test_create_visualization_success(self, viz_engine, sample_data, chart_config):
        """Test successful visualization creation."""
        result = await viz_engine.create_visualization(sample_data, chart_config)
        
        assert result.success is True
        assert result.data is not None
        assert result.data.name == chart_config.title
        assert result.data.chart_config.chart_type == chart_config.chart_type
        assert len(result.data.data) == len(sample_data)
        assert result.message == "Visualization created successfully"
    
    @pytest.mark.asyncio
    async def test_create_visualization_with_filters(self, viz_engine, sample_data, chart_config):
        """Test visualization creation with data filters."""
        filters = [
            DataFilter(
                field="category",
                operator=FilterOperator.EQUALS,
                value="A"
            )
        ]
        
        result = await viz_engine.create_visualization(sample_data, chart_config, filters)
        
        assert result.success is True
        assert len(result.data.data) == 2  # Only category A items
        assert all(item["category"] == "A" for item in result.data.data)
    
    @pytest.mark.asyncio
    async def test_create_visualization_with_aggregation(self, viz_engine, sample_data):
        """Test visualization creation with data aggregation."""
        config = ChartConfiguration(
            chart_type=ChartType.BAR,
            title="Sales by Category",
            x_axis="category",
            y_axis="sales",
            aggregation="sum",
            group_by="category"
        )
        
        result = await viz_engine.create_visualization(sample_data, config)
        
        assert result.success is True
        # Should have aggregated data by category
        categories = [item["category"] for item in result.data.data]
        assert len(set(categories)) == len(categories)  # No duplicates
    
    @pytest.mark.asyncio
    async def test_create_interactive_chart(self, viz_engine, sample_data, chart_config):
        """Test interactive chart creation."""
        interactive_features = InteractiveFeatures(
            zoom=True,
            pan=True,
            brush=False,
            crossfilter=False,
            drill_down=DrillDownConfig(
                enabled=True,
                levels=["category", "region"],
                default_level=0,
                show_breadcrumbs=True
            )
        )
        
        result = await viz_engine.create_interactive_chart(
            sample_data, chart_config, interactive_features
        )
        
        assert result["success"] is True
        assert "chart" in result
        assert "interactiveConfig" in result
        assert result["interactiveConfig"]["zoom"] is True
        assert result["interactiveConfig"]["drillDown"]["enabled"] is True
    
    def test_apply_filters_equals(self, viz_engine, sample_data):
        """Test applying equals filter."""
        filters = [
            DataFilter(
                field="category",
                operator=FilterOperator.EQUALS,
                value="A"
            )
        ]
        
        filtered_data = viz_engine._apply_filters(sample_data, filters)
        
        assert len(filtered_data) == 2
        assert all(item["category"] == "A" for item in filtered_data)
    
    def test_apply_filters_greater_than(self, viz_engine, sample_data):
        """Test applying greater than filter."""
        filters = [
            DataFilter(
                field="sales",
                operator=FilterOperator.GREATER_THAN,
                value=1000
            )
        ]
        
        filtered_data = viz_engine._apply_filters(sample_data, filters)
        
        assert len(filtered_data) == 2
        assert all(item["sales"] > 1000 for item in filtered_data)
    
    def test_apply_filters_contains(self, viz_engine, sample_data):
        """Test applying contains filter."""
        filters = [
            DataFilter(
                field="region",
                operator=FilterOperator.CONTAINS,
                value="orth"
            )
        ]
        
        filtered_data = viz_engine._apply_filters(sample_data, filters)
        
        assert len(filtered_data) == 2
        assert all("orth" in item["region"] for item in filtered_data)
    
    def test_apply_filters_in(self, viz_engine, sample_data):
        """Test applying 'in' filter."""
        filters = [
            DataFilter(
                field="category",
                operator=FilterOperator.IN,
                value=["A", "B"]
            )
        ]
        
        filtered_data = viz_engine._apply_filters(sample_data, filters)
        
        assert len(filtered_data) == 4
        assert all(item["category"] in ["A", "B"] for item in filtered_data)
    
    def test_apply_filters_between(self, viz_engine, sample_data):
        """Test applying between filter."""
        filters = [
            DataFilter(
                field="sales",
                operator=FilterOperator.BETWEEN,
                value=[900, 1200]
            )
        ]
        
        filtered_data = viz_engine._apply_filters(sample_data, filters)
        
        assert len(filtered_data) == 3
        assert all(900 <= item["sales"] <= 1200 for item in filtered_data)
    
    def test_process_pie_data(self, viz_engine, sample_data):
        """Test pie chart data processing."""
        df = pd.DataFrame(sample_data)
        config = ChartConfiguration(
            chart_type=ChartType.PIE,
            title="Sales by Category",
            x_axis="category",
            y_axis="sales",
            group_by="category"
        )
        
        processed_df = viz_engine._process_pie_data(df, config)
        
        # Should group by category and sum sales
        assert len(processed_df) <= len(df)
        assert "category" in processed_df.columns
        assert "sales" in processed_df.columns
    
    def test_get_chart_suggestions(self, viz_engine, sample_data):
        """Test chart type suggestions."""
        suggestions = viz_engine.get_chart_suggestions(sample_data)
        
        assert len(suggestions) > 0
        assert all("type" in suggestion for suggestion in suggestions)
        assert all("confidence" in suggestion for suggestion in suggestions)
        assert all(0 <= suggestion["confidence"] <= 1 for suggestion in suggestions)
        
        # Should be sorted by confidence
        confidences = [s["confidence"] for s in suggestions]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_get_chart_suggestions_empty_data(self, viz_engine):
        """Test chart suggestions with empty data."""
        suggestions = viz_engine.get_chart_suggestions([])
        
        assert suggestions == []
    
    def test_calculate_chart_statistics(self, viz_engine, sample_data, chart_config):
        """Test chart statistics calculation."""
        stats = viz_engine.calculate_chart_statistics(sample_data, chart_config)
        
        assert "total_records" in stats
        assert stats["total_records"] == len(sample_data)
        assert "columns" in stats
        assert "data_types" in stats
        assert "numeric_stats" in stats
        assert "categorical_stats" in stats
        
        # Check numeric statistics
        assert "sales" in stats["numeric_stats"]
        sales_stats = stats["numeric_stats"]["sales"]
        assert "mean" in sales_stats
        assert "std" in sales_stats
        assert "min" in sales_stats
        assert "max" in sales_stats
        
        # Check categorical statistics
        assert "category" in stats["categorical_stats"]
        category_stats = stats["categorical_stats"]["category"]
        assert "unique_values" in category_stats
        assert "most_common" in category_stats
    
    def test_generate_chart_config(self, viz_engine, sample_data, chart_config):
        """Test chart configuration generation."""
        chart_config_dict = viz_engine._generate_chart_config(sample_data, chart_config)
        
        assert chart_config_dict["type"] == chart_config.chart_type
        assert chart_config_dict["title"] == chart_config.title
        assert chart_config_dict["width"] == chart_config.width
        assert chart_config_dict["height"] == chart_config.height
        assert chart_config_dict["interactive"] == chart_config.interactive
        assert chart_config_dict["showLegend"] == chart_config.show_legend
        assert chart_config_dict["showGrid"] == chart_config.show_grid
        assert chart_config_dict["showTooltip"] == chart_config.show_tooltip
        
        # Check axis configuration
        assert "xAxis" in chart_config_dict
        assert "yAxis" in chart_config_dict
        assert chart_config_dict["xAxis"]["dataKey"] == chart_config.x_axis
        assert chart_config_dict["yAxis"]["dataKey"] == chart_config.y_axis
    
    def test_color_schemes(self, viz_engine):
        """Test color scheme availability."""
        assert "default" in viz_engine.color_schemes
        assert "professional" in viz_engine.color_schemes
        assert "modern" in viz_engine.color_schemes
        assert "scrollintel" in viz_engine.color_schemes
        
        # Each scheme should have colors
        for scheme_name, colors in viz_engine.color_schemes.items():
            assert isinstance(colors, list)
            assert len(colors) > 0
            assert all(isinstance(color, str) for color in colors)
            assert all(color.startswith("#") for color in colors)
    
    @pytest.mark.asyncio
    async def test_create_visualization_error_handling(self, viz_engine):
        """Test error handling in visualization creation."""
        # Test with invalid data
        invalid_config = ChartConfiguration(
            chart_type=ChartType.BAR,
            title="Test",
            x_axis="nonexistent_column",
            y_axis="another_nonexistent_column"
        )
        
        result = await viz_engine.create_visualization([], invalid_config)
        
        # Should handle gracefully and still return a response
        assert result.success is True  # Engine should handle empty data gracefully
    
    def test_process_data_with_grouping(self, viz_engine, sample_data):
        """Test data processing with grouping and aggregation."""
        config = ChartConfiguration(
            chart_type=ChartType.BAR,
            title="Test",
            x_axis="category",
            y_axis="sales",
            group_by="category",
            aggregation="sum"
        )
        
        processed_data = viz_engine._process_data(sample_data, config)
        
        # Should have grouped data
        categories = [item["category"] for item in processed_data]
        assert len(set(categories)) == len(categories)  # No duplicates
        
        # Should have aggregated sales values
        total_sales = sum(item["sales"] for item in processed_data)
        original_total = sum(item["sales"] for item in sample_data)
        assert total_sales == original_total