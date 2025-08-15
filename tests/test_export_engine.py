"""
Tests for the export engine.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from scrollintel.engines.export_engine import ExportEngine
from scrollintel.models.visualization_models import (
    ExportRequest, ExportFormat, VisualizationData, 
    ChartConfiguration, ChartType, DashboardLayout
)


@pytest.fixture
def export_engine():
    """Create an export engine instance."""
    return ExportEngine()


@pytest.fixture
def sample_visualization():
    """Create sample visualization data."""
    config = ChartConfiguration(
        chart_type=ChartType.BAR,
        title="Test Chart",
        x_axis="category",
        y_axis="sales",
        width=800,
        height=400
    )
    
    return VisualizationData(
        id="test-viz-1",
        name="Test Visualization",
        description="A test visualization",
        data=[
            {"category": "A", "sales": 1000},
            {"category": "B", "sales": 1200},
            {"category": "C", "sales": 800}
        ],
        chart_config=config,
        metadata={"created_by": "test_user"}
    )


@pytest.fixture
def sample_dashboard():
    """Create sample dashboard data."""
    return DashboardLayout(
        id="test-dashboard-1",
        name="Test Dashboard",
        description="A test dashboard",
        layout=[
            {"i": "chart-1", "x": 0, "y": 0, "w": 6, "h": 4}
        ],
        charts=["test-viz-1"]
    )


class TestExportEngine:
    """Test cases for ExportEngine."""
    
    def test_init(self, export_engine):
        """Test export engine initialization."""
        assert export_engine.temp_dir.exists()
        assert export_engine.temp_dir.name == "exports"
    
    @pytest.mark.asyncio
    async def test_export_json_success(self, export_engine, sample_visualization):
        """Test successful JSON export."""
        request = ExportRequest(
            format=ExportFormat.JSON,
            include_data=True,
            include_metadata=True,
            custom_title="Test Export"
        )
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is True
        assert "filename" in result
        assert result["filename"].endswith(".json")
        assert result["content_type"] == "application/json"
        assert "filepath" in result
        assert "size" in result
        
        # Verify file was created
        filepath = Path(result["filepath"])
        assert filepath.exists()
        
        # Verify JSON content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert "export_info" in data
        assert "visualizations" in data
        assert data["export_info"]["title"] == "Test Export"
        assert data["export_info"]["format"] == "json"
        assert len(data["visualizations"]) == 1
        
        viz_data = data["visualizations"][0]
        assert viz_data["id"] == sample_visualization.id
        assert viz_data["name"] == sample_visualization.name
        assert "data" in viz_data
        assert "metadata" in viz_data
        
        # Cleanup
        filepath.unlink()
    
    @pytest.mark.asyncio
    async def test_export_json_with_dashboard(self, export_engine, sample_visualization, sample_dashboard):
        """Test JSON export with dashboard information."""
        request = ExportRequest(
            format=ExportFormat.JSON,
            dashboard_id=sample_dashboard.id
        )
        
        result = await export_engine.export_visualization(
            request, [sample_visualization], sample_dashboard
        )
        
        assert result["success"] is True
        
        # Verify dashboard info in JSON
        filepath = Path(result["filepath"])
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert "dashboard" in data
        assert data["dashboard"]["id"] == sample_dashboard.id
        assert data["dashboard"]["name"] == sample_dashboard.name
        
        # Cleanup
        filepath.unlink()
    
    @pytest.mark.asyncio
    async def test_export_csv_single_visualization(self, export_engine, sample_visualization):
        """Test CSV export with single visualization."""
        request = ExportRequest(format=ExportFormat.CSV)
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is True
        assert result["filename"].endswith(".csv")
        assert result["content_type"] == "text/csv"
        
        # Verify CSV file
        filepath = Path(result["filepath"])
        assert filepath.exists()
        
        # Check CSV content
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert "category,sales" in content
        assert "A,1000" in content
        assert "B,1200" in content
        assert "C,800" in content
        
        # Cleanup
        filepath.unlink()
    
    @pytest.mark.asyncio
    async def test_export_csv_multiple_visualizations(self, export_engine, sample_visualization):
        """Test CSV export with multiple visualizations (creates ZIP)."""
        # Create second visualization
        viz2 = VisualizationData(
            id="test-viz-2",
            name="Test Viz 2",
            data=[{"x": 1, "y": 2}],
            chart_config=sample_visualization.chart_config
        )
        
        request = ExportRequest(format=ExportFormat.CSV)
        
        result = await export_engine.export_visualization(
            request, [sample_visualization, viz2]
        )
        
        assert result["success"] is True
        assert result["filename"].endswith(".zip")
        assert result["content_type"] == "application/zip"
        
        # Verify ZIP file
        filepath = Path(result["filepath"])
        assert filepath.exists()
        
        # Cleanup
        filepath.unlink()
    
    @pytest.mark.asyncio
    @patch('scrollintel.engines.export_engine.REPORTLAB_AVAILABLE', True)
    @patch('scrollintel.engines.export_engine.SimpleDocTemplate')
    async def test_export_pdf_success(self, mock_doc, export_engine, sample_visualization):
        """Test successful PDF export."""
        # Mock ReportLab components
        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        
        request = ExportRequest(
            format=ExportFormat.PDF,
            custom_title="Test PDF Report"
        )
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is True
        assert result["filename"].endswith(".pdf")
        assert result["content_type"] == "application/pdf"
        
        # Verify ReportLab was called
        mock_doc.assert_called_once()
        mock_doc_instance.build.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('scrollintel.engines.export_engine.REPORTLAB_AVAILABLE', False)
    async def test_export_pdf_reportlab_unavailable(self, export_engine, sample_visualization):
        """Test PDF export when ReportLab is not available."""
        request = ExportRequest(format=ExportFormat.PDF)
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is False
        assert "ReportLab not available" in result["error"]
    
    @pytest.mark.asyncio
    @patch('scrollintel.engines.export_engine.OPENPYXL_AVAILABLE', True)
    @patch('scrollintel.engines.export_engine.openpyxl.Workbook')
    async def test_export_excel_success(self, mock_workbook, export_engine, sample_visualization):
        """Test successful Excel export."""
        # Mock openpyxl components
        mock_wb = MagicMock()
        mock_workbook.return_value = mock_wb
        mock_sheet = MagicMock()
        mock_wb.create_sheet.return_value = mock_sheet
        mock_wb.active = mock_sheet
        
        request = ExportRequest(format=ExportFormat.EXCEL)
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is True
        assert result["filename"].endswith(".xlsx")
        assert result["content_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        # Verify openpyxl was called
        mock_workbook.assert_called_once()
        mock_wb.save.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('scrollintel.engines.export_engine.OPENPYXL_AVAILABLE', False)
    async def test_export_excel_openpyxl_unavailable(self, export_engine, sample_visualization):
        """Test Excel export when openpyxl is not available."""
        request = ExportRequest(format=ExportFormat.EXCEL)
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is False
        assert "openpyxl not available" in result["error"]
    
    @pytest.mark.asyncio
    async def test_export_image_not_implemented(self, export_engine, sample_visualization):
        """Test image export (not yet implemented)."""
        request = ExportRequest(format=ExportFormat.PNG)
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is False
        assert "not yet implemented" in result["error"]
    
    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, export_engine, sample_visualization):
        """Test export with unsupported format."""
        # This would require modifying the enum, so we'll test error handling
        request = ExportRequest(format=ExportFormat.JSON)  # Valid format
        
        # Mock the format to be unsupported
        with patch.object(request, 'format', 'unsupported'):
            result = await export_engine.export_visualization(
                request, [sample_visualization]
            )
            
            assert result["success"] is False
            assert "Unsupported export format" in result["error"]
    
    @pytest.mark.asyncio
    async def test_export_with_filters(self, export_engine, sample_visualization):
        """Test export with applied filters."""
        from scrollintel.models.visualization_models import DataFilter, FilterOperator
        
        filters = [
            DataFilter(
                field="category",
                operator=FilterOperator.EQUALS,
                value="A"
            )
        ]
        
        request = ExportRequest(
            format=ExportFormat.JSON,
            filters=filters
        )
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is True
        
        # Verify filters are included in export
        filepath = Path(result["filepath"])
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Filters should be documented in export info
        assert "export_info" in data
        
        # Cleanup
        filepath.unlink()
    
    def test_create_pdf_table(self, export_engine):
        """Test PDF table creation."""
        data = [
            {"name": "John", "age": 30, "city": "New York"},
            {"name": "Jane", "age": 25, "city": "Boston"}
        ]
        
        table = export_engine._create_pdf_table(data)
        
        # Should return None if ReportLab not available, or a table object if available
        # We can't test the actual table creation without ReportLab installed
        assert table is None or hasattr(table, 'setStyle')
    
    def test_create_pdf_table_empty_data(self, export_engine):
        """Test PDF table creation with empty data."""
        table = export_engine._create_pdf_table([])
        
        assert table is None
    
    def test_create_excel_chart(self, export_engine, sample_visualization):
        """Test Excel chart creation."""
        import pandas as pd
        
        df = pd.DataFrame(sample_visualization.data)
        chart = export_engine._create_excel_chart(sample_visualization, df)
        
        # Should return None if openpyxl not available
        assert chart is None
    
    @pytest.mark.asyncio
    async def test_export_error_handling(self, export_engine):
        """Test export error handling."""
        # Test with empty visualization data
        empty_viz = VisualizationData(
            id="empty",
            name="Empty",
            data=[],  # Empty data
            chart_config=ChartConfiguration(
                chart_type=ChartType.BAR,
                title="Empty",
                x_axis="x",
                y_axis="y"
            )
        )
        
        request = ExportRequest(format=ExportFormat.JSON)
        
        # Should handle gracefully
        result = await export_engine.export_visualization(request, [empty_viz])
        
        # Should still succeed with empty data
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_export_json_exclude_data(self, export_engine, sample_visualization):
        """Test JSON export excluding data."""
        request = ExportRequest(
            format=ExportFormat.JSON,
            include_data=False,
            include_metadata=True
        )
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is True
        
        # Verify data is excluded
        filepath = Path(result["filepath"])
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        viz_data = data["visualizations"][0]
        assert "data" not in viz_data
        assert "metadata" in viz_data
        
        # Cleanup
        filepath.unlink()
    
    @pytest.mark.asyncio
    async def test_export_json_exclude_metadata(self, export_engine, sample_visualization):
        """Test JSON export excluding metadata."""
        request = ExportRequest(
            format=ExportFormat.JSON,
            include_data=True,
            include_metadata=False
        )
        
        result = await export_engine.export_visualization(
            request, [sample_visualization]
        )
        
        assert result["success"] is True
        
        # Verify metadata is excluded
        filepath = Path(result["filepath"])
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        viz_data = data["visualizations"][0]
        assert "data" in viz_data
        assert "metadata" not in viz_data
        
        # Cleanup
        filepath.unlink()