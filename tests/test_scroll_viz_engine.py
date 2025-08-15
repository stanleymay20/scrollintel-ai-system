"""
Unit tests for ScrollViz Engine.
"""

import pytest
import pandas as pd
import numpy as np
import json
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.scroll_viz_engine import (
    ScrollVizEngine, ChartType, DataType, ExportFormat, VisualizationTemplate
)
from scrollintel.engines.base_engine import EngineStatus


@pytest.fixture
def viz_engine():
    """Create a ScrollViz engine instance."""
    return ScrollVizEngine()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'profit': np.random.normal(100, 50, 100),
        'quantity': np.random.randint(1, 100, 100)
    })


@pytest.fixture
def numerical_data():
    """Create numerical data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'z': np.random.normal(0, 1, 100)
    })


@pytest.fixture
def categorical_data():
    """Create categorical data for testing."""
    return pd.DataFrame({
        'category': ['A', 'B', 'C', 'D', 'E'],
        'count': [10, 25, 15, 30, 20],
        'percentage': [20, 50, 30, 60, 40]
    })


class TestScrollVizEngine:
    """Test ScrollViz engine functionality."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, viz_engine):
        """Test engine initialization."""
        await viz_engine.start()
        
        assert viz_engine.status == EngineStatus.READY
        assert viz_engine.engine_id == "scroll_viz"
        assert viz_engine.name == "ScrollViz Visualization Engine"
        assert len(viz_engine.templates) > 0
        assert len(viz_engine.chart_recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_engine_status(self, viz_engine):
        """Test engine status reporting."""
        await viz_engine.start()
        status = viz_engine.get_status()
        
        assert status["healthy"] is True
        assert status["status"] == "ready"
        assert "templates_loaded" in status
        assert "supported_chart_types" in status
        assert "supported_export_formats" in status
    
    @pytest.mark.asyncio
    async def test_chart_type_recommendation_numerical(self, viz_engine, numerical_data):
        """Test chart type recommendation for numerical data."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            numerical_data,
            {"action": "recommend_chart_type", "columns": ["x"]}
        )
        
        assert "recommended_type" in result
        assert "alternative_types" in result
        assert "data_types" in result
        assert "reasoning" in result
        assert result["data_types"][0] == DataType.NUMERICAL.value
    
    @pytest.mark.asyncio
    async def test_chart_type_recommendation_categorical(self, viz_engine, categorical_data):
        """Test chart type recommendation for categorical data."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            categorical_data,
            {"action": "recommend_chart_type", "columns": ["category", "count"]}
        )
        
        assert "recommended_type" in result
        assert result["data_types"][0] == DataType.CATEGORICAL.value
        assert result["data_types"][1] == DataType.NUMERICAL.value
    
    @pytest.mark.asyncio
    async def test_chart_type_recommendation_datetime(self, viz_engine, sample_data):
        """Test chart type recommendation for datetime data."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            sample_data,
            {"action": "recommend_chart_type", "columns": ["date", "sales"]}
        )
        
        assert "recommended_type" in result
        assert result["data_types"][0] == DataType.DATETIME.value
        assert result["data_types"][1] == DataType.NUMERICAL.value
        assert result["recommended_type"] in [ChartType.LINE.value, ChartType.AREA.value]
    
    @pytest.mark.asyncio
    async def test_generate_bar_chart(self, viz_engine, categorical_data):
        """Test bar chart generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count",
                "title": "Test Bar Chart"
            }
        )
        
        assert result["chart_type"] == ChartType.BAR.value
        assert "plotly_json" in result
        assert "html" in result
        assert "config" in result
        
        # Verify plotly JSON is valid
        plotly_data = json.loads(result["plotly_json"])
        assert "data" in plotly_data
        assert "layout" in plotly_data
    
    @pytest.mark.asyncio
    async def test_generate_line_chart(self, viz_engine, sample_data):
        """Test line chart generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            sample_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.LINE.value,
                "x_column": "date",
                "y_column": "sales",
                "title": "Sales Trend"
            }
        )
        
        assert result["chart_type"] == ChartType.LINE.value
        assert "plotly_json" in result
        
        plotly_data = json.loads(result["plotly_json"])
        assert plotly_data["layout"]["title"]["text"] == "Sales Trend"
    
    @pytest.mark.asyncio
    async def test_generate_scatter_plot(self, viz_engine, numerical_data):
        """Test scatter plot generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            numerical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.SCATTER.value,
                "x_column": "x",
                "y_column": "y",
                "title": "Scatter Plot"
            }
        )
        
        assert result["chart_type"] == ChartType.SCATTER.value
        assert "plotly_json" in result
    
    @pytest.mark.asyncio
    async def test_generate_pie_chart(self, viz_engine, categorical_data):
        """Test pie chart generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.PIE.value,
                "x_column": "category",
                "y_column": "count",
                "title": "Category Distribution"
            }
        )
        
        assert result["chart_type"] == ChartType.PIE.value
        assert "plotly_json" in result
    
    @pytest.mark.asyncio
    async def test_generate_histogram(self, viz_engine, numerical_data):
        """Test histogram generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            numerical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.HISTOGRAM.value,
                "x_column": "x",
                "title": "Distribution"
            }
        )
        
        assert result["chart_type"] == ChartType.HISTOGRAM.value
        assert "plotly_json" in result
    
    @pytest.mark.asyncio
    async def test_generate_heatmap(self, viz_engine, numerical_data):
        """Test heatmap generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            numerical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.HEATMAP.value,
                "correlation": True,
                "title": "Correlation Heatmap"
            }
        )
        
        assert result["chart_type"] == ChartType.HEATMAP.value
        assert "plotly_json" in result
    
    @pytest.mark.asyncio
    async def test_auto_chart_recommendation(self, viz_engine, sample_data):
        """Test automatic chart type recommendation during generation."""
        await viz_engine.start()
        
        # Don't specify chart_type, let it auto-recommend
        result = await viz_engine.process(
            sample_data,
            {
                "action": "generate_chart",
                "x_column": "date",
                "y_column": "sales"
            }
        )
        
        assert "chart_type" in result
        assert "plotly_json" in result
        # Should recommend line chart for datetime + numerical
        assert result["chart_type"] in [ChartType.LINE.value, ChartType.AREA.value]
    
    @pytest.mark.asyncio
    async def test_recharts_config_generation(self, viz_engine, categorical_data):
        """Test Recharts configuration generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count",
                "include_recharts": True
            }
        )
        
        assert "recharts_config" in result
        recharts_config = result["recharts_config"]
        assert recharts_config["type"] == ChartType.BAR.value
        assert "data" in recharts_config
        assert "xAxis" in recharts_config
        assert "yAxis" in recharts_config
        assert "series" in recharts_config
    
    @pytest.mark.asyncio
    async def test_vega_lite_spec_generation(self, viz_engine, categorical_data):
        """Test Vega-Lite specification generation."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count",
                "include_vega": True
            }
        )
        
        assert "vega_lite_spec" in result
        vega_spec = result["vega_lite_spec"]
        assert vega_spec["$schema"].startswith("https://vega.github.io/schema/vega-lite")
        assert "data" in vega_spec
        assert "mark" in vega_spec
        assert "encoding" in vega_spec
    
    @pytest.mark.asyncio
    async def test_create_dashboard_single_dataset(self, viz_engine, sample_data):
        """Test dashboard creation with single dataset."""
        await viz_engine.start()
        
        dashboard_config = {
            "title": "Sales Dashboard",
            "description": "Sales performance overview",
            "charts": [
                {
                    "chart_type": ChartType.LINE.value,
                    "x_column": "date",
                    "y_column": "sales",
                    "title": "Sales Trend"
                },
                {
                    "chart_type": ChartType.BAR.value,
                    "x_column": "category",
                    "y_column": "sales",
                    "title": "Sales by Category"
                }
            ]
        }
        
        result = await viz_engine.process(
            sample_data,
            {
                "action": "create_dashboard",
                "dashboard_config": dashboard_config
            }
        )
        
        assert "id" in result
        assert result["title"] == "Sales Dashboard"
        assert len(result["charts"]) == 2
        assert "created_at" in result
    
    @pytest.mark.asyncio
    async def test_create_dashboard_multiple_datasets(self, viz_engine, sample_data, categorical_data):
        """Test dashboard creation with multiple datasets."""
        await viz_engine.start()
        
        data_configs = [
            {
                "data": sample_data,
                "config": {
                    "chart_type": ChartType.LINE.value,
                    "x_column": "date",
                    "y_column": "sales",
                    "title": "Sales Trend"
                }
            },
            {
                "data": categorical_data,
                "config": {
                    "chart_type": ChartType.PIE.value,
                    "x_column": "category",
                    "y_column": "count",
                    "title": "Category Distribution"
                }
            }
        ]
        
        result = await viz_engine.process(
            data_configs,
            {"action": "create_dashboard"}
        )
        
        assert len(result["charts"]) == 2
        assert result["charts"][0]["chart_type"] == ChartType.LINE.value
        assert result["charts"][1]["chart_type"] == ChartType.PIE.value
    
    @pytest.mark.asyncio
    @patch('plotly.graph_objects.Figure.to_image')
    async def test_export_png(self, mock_to_image, viz_engine, categorical_data):
        """Test PNG export functionality."""
        await viz_engine.start()
        
        # Mock the image generation
        mock_image_bytes = b"fake_png_data"
        mock_to_image.return_value = mock_image_bytes
        
        # First generate a chart
        chart_result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count"
            }
        )
        
        # Then export it
        export_result = await viz_engine.process(
            chart_result,
            {
                "action": "export_visualization",
                "format": ExportFormat.PNG.value,
                "width": 800,
                "height": 600
            }
        )
        
        assert export_result["format"] == ExportFormat.PNG.value
        assert export_result["mime_type"] == "image/png"
        assert "data" in export_result
        # Verify base64 encoding
        decoded_data = base64.b64decode(export_result["data"])
        assert decoded_data == mock_image_bytes
    
    @pytest.mark.asyncio
    @patch('plotly.graph_objects.Figure.to_image')
    async def test_export_svg(self, mock_to_image, viz_engine, categorical_data):
        """Test SVG export functionality."""
        await viz_engine.start()
        
        mock_svg_data = "<svg>fake svg content</svg>"
        mock_to_image.return_value = mock_svg_data.encode()
        
        chart_result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count"
            }
        )
        
        export_result = await viz_engine.process(
            chart_result,
            {
                "action": "export_visualization",
                "format": ExportFormat.SVG.value
            }
        )
        
        assert export_result["format"] == ExportFormat.SVG.value
        assert export_result["mime_type"] == "image/svg+xml"
        assert export_result["data"] == mock_svg_data
    
    @pytest.mark.asyncio
    async def test_export_html(self, viz_engine, categorical_data):
        """Test HTML export functionality."""
        await viz_engine.start()
        
        chart_result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count"
            }
        )
        
        export_result = await viz_engine.process(
            chart_result,
            {
                "action": "export_visualization",
                "format": ExportFormat.HTML.value
            }
        )
        
        assert export_result["format"] == ExportFormat.HTML.value
        assert export_result["mime_type"] == "text/html"
        assert "<html>" in export_result["data"]
        assert "plotly" in export_result["data"].lower()
    
    @pytest.mark.asyncio
    async def test_export_json(self, viz_engine, categorical_data):
        """Test JSON export functionality."""
        await viz_engine.start()
        
        chart_result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count"
            }
        )
        
        export_result = await viz_engine.process(
            chart_result,
            {
                "action": "export_visualization",
                "format": ExportFormat.JSON.value
            }
        )
        
        assert export_result["format"] == ExportFormat.JSON.value
        assert export_result["mime_type"] == "application/json"
        # Should be valid JSON
        json.loads(export_result["data"])
    
    @pytest.mark.asyncio
    async def test_get_templates(self, viz_engine):
        """Test template retrieval."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            None,
            {"action": "get_templates"}
        )
        
        assert "templates" in result
        assert "total_count" in result
        assert len(result["templates"]) > 0
        
        # Check template structure
        template = result["templates"][0]
        assert "name" in template
        assert "chart_type" in template
        assert "description" in template
        assert "data_requirements" in template
        assert "config" in template
    
    @pytest.mark.asyncio
    async def test_data_type_analysis(self, viz_engine, sample_data):
        """Test data type analysis functionality."""
        await viz_engine.start()
        
        # Test different column types
        date_type = viz_engine._analyze_column_type(sample_data['date'])
        assert date_type == DataType.DATETIME.value
        
        numerical_type = viz_engine._analyze_column_type(sample_data['sales'])
        assert numerical_type == DataType.NUMERICAL.value
        
        categorical_type = viz_engine._analyze_column_type(sample_data['category'])
        assert categorical_type == DataType.CATEGORICAL.value
    
    @pytest.mark.asyncio
    async def test_context_recommendations(self, viz_engine, sample_data):
        """Test context-based recommendations."""
        await viz_engine.start()
        
        recommendations = viz_engine._get_context_recommendations(sample_data, {})
        
        # Should detect time series data
        assert any("time series" in rec.lower() for rec in recommendations)
        
        # Should suggest correlation analysis for multiple numeric columns
        assert any("correlation" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_action(self, viz_engine):
        """Test error handling for invalid actions."""
        await viz_engine.start()
        
        with pytest.raises(ValueError, match="Unknown action"):
            await viz_engine.process(
                {},
                {"action": "invalid_action"}
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_data(self, viz_engine):
        """Test error handling for missing data."""
        await viz_engine.start()
        
        with pytest.raises(Exception):
            await viz_engine.process(
                None,
                {"action": "generate_chart"}
            )
    
    @pytest.mark.asyncio
    async def test_cleanup(self, viz_engine):
        """Test engine cleanup."""
        await viz_engine.start()
        
        # Verify templates are loaded
        assert len(viz_engine.templates) > 0
        assert len(viz_engine.chart_recommendations) > 0
        
        # Cleanup
        await viz_engine.cleanup()
        
        # Verify cleanup
        assert len(viz_engine.templates) == 0
        assert len(viz_engine.chart_recommendations) == 0
    
    @pytest.mark.asyncio
    async def test_chart_styling_parameters(self, viz_engine, categorical_data):
        """Test chart styling and customization parameters."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            categorical_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.BAR.value,
                "x_column": "category",
                "y_column": "count",
                "title": "Custom Styled Chart",
                "theme": "plotly_dark",
                "show_legend": False,
                "width": 1000,
                "height": 800
            }
        )
        
        plotly_data = json.loads(result["plotly_json"])
        layout = plotly_data["layout"]
        
        assert layout["title"]["text"] == "Custom Styled Chart"
        # Template is applied as an object, not just the name
        assert "template" in layout
        assert layout["showlegend"] is False
        assert layout["width"] == 1000
        assert layout["height"] == 800
    
    @pytest.mark.asyncio
    async def test_color_and_size_encoding(self, viz_engine, sample_data):
        """Test color and size encoding in charts."""
        await viz_engine.start()
        
        result = await viz_engine.process(
            sample_data,
            {
                "action": "generate_chart",
                "chart_type": ChartType.SCATTER.value,
                "x_column": "sales",
                "y_column": "profit",
                "color_column": "category",
                "size_column": "quantity",
                "title": "Sales vs Profit"
            }
        )
        
        assert result["chart_type"] == ChartType.SCATTER.value
        plotly_data = json.loads(result["plotly_json"])
        
        # Verify that color and size encodings are applied
        assert "data" in plotly_data
        assert len(plotly_data["data"]) > 0


class TestVisualizationTemplate:
    """Test VisualizationTemplate class."""
    
    def test_template_creation(self):
        """Test template creation."""
        template = VisualizationTemplate(
            name="test_template",
            chart_type=ChartType.BAR,
            description="Test template",
            data_requirements={"x": "categorical", "y": "numerical"},
            config={"color": "blue"}
        )
        
        assert template.name == "test_template"
        assert template.chart_type == ChartType.BAR
        assert template.description == "Test template"
        assert template.data_requirements == {"x": "categorical", "y": "numerical"}
        assert template.config == {"color": "blue"}


class TestEnums:
    """Test enum definitions."""
    
    def test_chart_type_enum(self):
        """Test ChartType enum."""
        assert ChartType.BAR.value == "bar"
        assert ChartType.LINE.value == "line"
        assert ChartType.SCATTER.value == "scatter"
        assert ChartType.PIE.value == "pie"
    
    def test_data_type_enum(self):
        """Test DataType enum."""
        assert DataType.NUMERICAL.value == "numerical"
        assert DataType.CATEGORICAL.value == "categorical"
        assert DataType.DATETIME.value == "datetime"
        assert DataType.BOOLEAN.value == "boolean"
    
    def test_export_format_enum(self):
        """Test ExportFormat enum."""
        assert ExportFormat.PNG.value == "png"
        assert ExportFormat.SVG.value == "svg"
        assert ExportFormat.PDF.value == "pdf"
        assert ExportFormat.HTML.value == "html"
        assert ExportFormat.JSON.value == "json"
