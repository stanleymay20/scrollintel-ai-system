"""
Tests for BI and Analytics Tool Integration Engine
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import tempfile
import os

from ai_data_readiness.engines.bi_analytics_integrator import (
    BIAnalyticsIntegrator,
    BIToolConfig,
    DataExportConfig,
    ReportDistributionInfo,
    DataSourceInfo,
    TableauConnector,
    PowerBIConnector,
    GenericBIConnector,
    DataExporter
)


class TestBIToolConfig:
    """Test BI Tool Configuration"""
    
    def test_config_creation(self):
        """Test creating BI tool configuration"""
        config = BIToolConfig(
            tool_type="tableau",
            server_url="https://tableau.company.com",
            credentials={"username": "test_user", "password": "test_pass"},
            workspace_id="project123",
            metadata={"environment": "production"}
        )
        
        assert config.tool_type == "tableau"
        assert config.server_url == "https://tableau.company.com"
        assert config.credentials["username"] == "test_user"
        assert config.workspace_id == "project123"
        assert config.metadata["environment"] == "production"


class TestDataExportConfig:
    """Test Data Export Configuration"""
    
    def test_export_config_creation(self):
        """Test creating data export configuration"""
        config = DataExportConfig(
            export_format="csv",
            include_metadata=True,
            compression="gzip",
            filters={"category": {"in": ["A", "B"]}}
        )
        
        assert config.export_format == "csv"
        assert config.include_metadata is True
        assert config.compression == "gzip"
        assert config.filters["category"]["in"] == ["A", "B"]


class TestGenericBIConnector:
    """Test Generic BI Connector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = BIToolConfig(
            tool_type="generic",
            server_url="https://bi-tool.company.com",
            credentials={"api_key": "test_key"}
        )
        self.connector = GenericBIConnector(self.config)
    
    @patch('requests.Session.get')
    def test_connect_success(self, mock_get):
        """Test successful connection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        assert result is True
        mock_get.assert_called_once_with("https://bi-tool.company.com/api/health")
    
    @patch('requests.Session.get')
    def test_connect_failure(self, mock_get):
        """Test connection failure"""
        mock_get.side_effect = Exception("Connection failed")
        
        result = self.connector.connect()
        assert result is False
    
    @patch('requests.Session.post')
    def test_create_data_source(self, mock_post):
        """Test creating data source"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "ds123",
            "name": "Test Dataset"
        }
        mock_post.return_value = mock_response
        
        dataset_info = {
            "id": "dataset1",
            "name": "Test Dataset",
            "format": "json"
        }
        
        result = self.connector.create_data_source(dataset_info)
        
        assert isinstance(result, DataSourceInfo)
        assert result.source_id == "ds123"
        assert result.source_name == "Test Dataset"
        assert result.dataset_id == "dataset1"
    
    @patch('requests.Session.get')
    def test_get_data_sources(self, mock_get):
        """Test getting data sources"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "ds1", "name": "Dataset 1", "status": "active"},
            {"id": "ds2", "name": "Dataset 2", "status": "inactive"}
        ]
        mock_get.return_value = mock_response
        
        sources = self.connector.get_data_sources()
        
        assert len(sources) == 2
        assert sources[0].source_id == "ds1"
        assert sources[0].source_name == "Dataset 1"
        assert sources[0].refresh_status == "active"
    
    @patch('requests.Session.post')
    def test_create_report(self, mock_post):
        """Test creating report"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "report123"}
        mock_post.return_value = mock_response
        
        report_config = {
            "name": "Test Report",
            "data_source_id": "ds123"
        }
        
        report_id = self.connector.create_report(report_config)
        assert report_id == "report123"
    
    @patch('requests.Session.post')
    def test_distribute_report(self, mock_post):
        """Test report distribution"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        distribution_info = ReportDistributionInfo(
            report_id="report123",
            report_name="Test Report",
            recipients=["user1@company.com", "user2@company.com"],
            distribution_schedule="daily",
            format="pdf",
            status="pending"
        )
        
        result = self.connector.distribute_report(distribution_info)
        assert result is True


@patch('ai_data_readiness.engines.bi_analytics_integrator.TABLEAU_AVAILABLE', True)
class TestTableauConnector:
    """Test Tableau Connector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = BIToolConfig(
            tool_type="tableau",
            server_url="https://tableau.company.com",
            credentials={"username": "test_user", "password": "test_pass"}
        )
    
    @patch('tableauserverclient.Server')
    @patch('tableauserverclient.TableauAuth')
    def test_connect_success(self, mock_auth, mock_server):
        """Test successful Tableau connection"""
        mock_server_instance = Mock()
        mock_server.return_value = mock_server_instance
        
        mock_auth_instance = Mock()
        mock_auth.return_value = mock_auth_instance
        
        connector = TableauConnector(self.config)
        result = connector.connect()
        
        assert result is True
        mock_server.assert_called_once()
        mock_auth.assert_called_once_with("test_user", "test_pass", site_id="")
    
    @patch('tableauserverclient.Server')
    @patch('tableauserverclient.TableauAuth')
    def test_create_data_source(self, mock_auth, mock_server):
        """Test creating Tableau data source"""
        connector = TableauConnector(self.config)
        
        dataset_info = {
            "id": "dataset1",
            "name": "Test Dataset",
            "row_count": 1000
        }
        
        result = connector.create_data_source(dataset_info)
        
        assert isinstance(result, DataSourceInfo)
        assert result.source_name == "Test Dataset"
        assert result.dataset_id == "dataset1"
        assert result.row_count == 1000


class TestPowerBIConnector:
    """Test Power BI Connector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = BIToolConfig(
            tool_type="powerbi",
            server_url="https://api.powerbi.com",
            credentials={
                "client_id": "test_client_id",
                "client_secret": "test_client_secret"
            },
            workspace_id="workspace123"
        )
    
    @patch('requests.post')
    def test_connect_success(self, mock_post):
        """Test successful Power BI connection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "test_token"}
        mock_post.return_value = mock_response
        
        connector = PowerBIConnector(self.config)
        result = connector.connect()
        
        assert result is True
        assert connector.access_token == "test_token"
    
    @patch('requests.post')
    def test_connect_failure(self, mock_post):
        """Test Power BI connection failure"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        connector = PowerBIConnector(self.config)
        result = connector.connect()
        
        assert result is False


class TestDataExporter:
    """Test Data Exporter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.exporter = DataExporter()
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10.5, 20.3, 30.1, 40.8, 50.2],
            'category': ['X', 'Y', 'X', 'Y', 'X']
        })
    
    def test_export_csv(self):
        """Test CSV export"""
        config = DataExportConfig(export_format="csv")
        
        result = self.exporter.export_dataset(self.sample_data, config)
        
        assert isinstance(result, str)
        assert "id,name,value,category" in result
        assert "1,A,10.5,X" in result
    
    def test_export_json(self):
        """Test JSON export"""
        config = DataExportConfig(export_format="json", include_metadata=True)
        
        result = self.exporter.export_dataset(self.sample_data, config)
        
        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data" in result
        assert result["metadata"]["row_count"] == 5
        assert result["metadata"]["column_count"] == 4
        assert len(result["data"]) == 5
    
    def test_export_json_no_metadata(self):
        """Test JSON export without metadata"""
        config = DataExportConfig(export_format="json", include_metadata=False)
        
        result = self.exporter.export_dataset(self.sample_data, config)
        
        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0]["id"] == 1
        assert result[0]["name"] == "A"
    
    def test_export_with_filters(self):
        """Test export with data filters"""
        config = DataExportConfig(
            export_format="json",
            include_metadata=False,
            filters={"category": {"equals": "X"}}
        )
        
        result = self.exporter.export_dataset(self.sample_data, config)
        
        assert isinstance(result, list)
        assert len(result) == 3  # Only records with category 'X'
        assert all(record["category"] == "X" for record in result)
    
    def test_export_with_range_filter(self):
        """Test export with range filter"""
        config = DataExportConfig(
            export_format="json",
            include_metadata=False,
            filters={"value": {"range": [20, 40]}}
        )
        
        result = self.exporter.export_dataset(self.sample_data, config)
        
        assert isinstance(result, list)
        assert len(result) == 2  # Values 20.3 and 30.1
        assert all(20 <= record["value"] <= 40 for record in result)
    
    def test_export_parquet(self):
        """Test Parquet export"""
        config = DataExportConfig(export_format="parquet")
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            try:
                result = self.exporter.export_dataset(self.sample_data, config, tmp_file.name)
                
                assert result == tmp_file.name
                assert os.path.exists(tmp_file.name)
                
                # Verify the file can be read back
                loaded_data = pd.read_parquet(tmp_file.name)
                assert len(loaded_data) == 5
                assert list(loaded_data.columns) == ['id', 'name', 'value', 'category']
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_export_excel(self):
        """Test Excel export"""
        config = DataExportConfig(export_format="excel", include_metadata=True)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                result = self.exporter.export_dataset(self.sample_data, config, tmp_file.name)
                
                assert result == tmp_file.name
                assert os.path.exists(tmp_file.name)
                
                # Verify the file can be read back
                loaded_data = pd.read_excel(tmp_file.name, sheet_name='Data')
                assert len(loaded_data) == 5
                assert list(loaded_data.columns) == ['id', 'name', 'value', 'category']
                
                # Check metadata sheet
                metadata = pd.read_excel(tmp_file.name, sheet_name='Metadata')
                assert len(metadata) == 4  # 4 metadata properties
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_unsupported_format(self):
        """Test unsupported export format"""
        config = DataExportConfig(export_format="unsupported")
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.exporter.export_dataset(self.sample_data, config)


class TestBIAnalyticsIntegrator:
    """Test BI Analytics Integrator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.integrator = BIAnalyticsIntegrator()
    
    def test_register_bi_tool_generic(self):
        """Test registering a generic BI tool"""
        config = BIToolConfig(
            tool_type="generic",
            server_url="https://bi-tool.company.com",
            credentials={"api_key": "test_key"}
        )
        
        with patch.object(GenericBIConnector, 'connect', return_value=True):
            result = self.integrator.register_bi_tool("test_bi_tool", config)
            assert result is True
            assert "test_bi_tool" in self.integrator.connectors
    
    def test_register_bi_tool_connection_failure(self):
        """Test registering BI tool with connection failure"""
        config = BIToolConfig(
            tool_type="generic",
            server_url="https://invalid-bi-tool.com",
            credentials={}
        )
        
        with patch.object(GenericBIConnector, 'connect', return_value=False):
            result = self.integrator.register_bi_tool("invalid_bi_tool", config)
            assert result is False
            assert "invalid_bi_tool" not in self.integrator.connectors
    
    def test_create_data_sources(self):
        """Test creating data sources across BI tools"""
        # Register mock BI tools
        mock_connector1 = Mock()
        mock_source1 = DataSourceInfo(
            source_id="ds1",
            source_name="Test Dataset",
            connection_type="api",
            dataset_id="dataset1"
        )
        mock_connector1.create_data_source.return_value = mock_source1
        
        mock_connector2 = Mock()
        mock_source2 = DataSourceInfo(
            source_id="ds2",
            source_name="Test Dataset",
            connection_type="tableau",
            dataset_id="dataset1"
        )
        mock_connector2.create_data_source.return_value = mock_source2
        
        self.integrator.connectors = {
            "generic_tool": mock_connector1,
            "tableau_tool": mock_connector2
        }
        
        dataset_info = {
            "id": "dataset1",
            "name": "Test Dataset",
            "description": "Test dataset description"
        }
        
        results = self.integrator.create_data_sources(dataset_info)
        
        assert len(results) == 2
        assert "generic_tool" in results
        assert "tableau_tool" in results
        assert results["generic_tool"].source_id == "ds1"
        assert results["tableau_tool"].source_id == "ds2"
    
    def test_export_data(self):
        """Test data export functionality"""
        sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        
        export_configs = [
            DataExportConfig(export_format="csv"),
            DataExportConfig(export_format="json", include_metadata=True)
        ]
        
        results = self.integrator.export_data(sample_data, export_configs, "test_data")
        
        assert len(results) == 2
        assert "csv" in results
        assert "json" in results
        assert results["csv"] is not None
        assert results["json"] is not None
    
    def test_distribute_reports(self):
        """Test report distribution"""
        # Register mock BI tools
        mock_connector1 = Mock()
        mock_connector1.distribute_report.return_value = True
        
        mock_connector2 = Mock()
        mock_connector2.distribute_report.return_value = False
        
        self.integrator.connectors = {
            "tool1": mock_connector1,
            "tool2": mock_connector2
        }
        
        distributions = [
            ReportDistributionInfo(
                report_id="report1",
                report_name="Test Report",
                recipients=["user@company.com"],
                distribution_schedule="daily",
                format="pdf",
                status="pending"
            )
        ]
        
        results = self.integrator.distribute_reports(distributions)
        
        assert len(results) == 1
        assert "report1" in results
        assert results["report1"] is True  # At least one tool succeeded
    
    def test_sync_data_sources(self):
        """Test data source synchronization"""
        # Mock connector with existing data sources
        mock_connector = Mock()
        existing_source = DataSourceInfo(
            source_id="ds1",
            source_name="Existing Dataset",
            connection_type="api",
            dataset_id="dataset1"
        )
        mock_connector.get_data_sources.return_value = [existing_source]
        mock_connector.update_data_source.return_value = True
        
        self.integrator.connectors = {"test_tool": mock_connector}
        
        dataset_updates = {
            "dataset1": {
                "id": "dataset1",
                "name": "Updated Dataset",
                "description": "Updated description"
            }
        }
        
        results = self.integrator.sync_data_sources(dataset_updates)
        
        assert "dataset1" in results
        assert "test_tool_ds1" in results["dataset1"]
        assert results["dataset1"]["test_tool_ds1"] is True
    
    def test_get_integration_status(self):
        """Test getting integration status"""
        mock_connector = Mock()
        mock_connector.connect.return_value = True
        mock_connector.get_data_sources.return_value = [Mock(), Mock()]  # 2 data sources
        mock_connector.config.tool_type = "generic"
        mock_connector.config.server_url = "https://test.com"
        mock_connector.config.workspace_id = "workspace123"
        
        self.integrator.connectors = {"test_tool": mock_connector}
        
        status = self.integrator.get_integration_status()
        
        assert "test_tool" in status
        assert status["test_tool"]["connected"] is True
        assert status["test_tool"]["data_source_count"] == 2
        assert status["test_tool"]["tool_type"] == "generic"


class TestReportDistributionInfo:
    """Test ReportDistributionInfo dataclass"""
    
    def test_creation(self):
        """Test creating ReportDistributionInfo"""
        distribution = ReportDistributionInfo(
            report_id="report123",
            report_name="Monthly Sales Report",
            recipients=["manager@company.com", "analyst@company.com"],
            distribution_schedule="monthly",
            format="pdf",
            status="active",
            last_sent=datetime.now(),
            metadata={"priority": "high"}
        )
        
        assert distribution.report_id == "report123"
        assert distribution.report_name == "Monthly Sales Report"
        assert len(distribution.recipients) == 2
        assert distribution.distribution_schedule == "monthly"
        assert distribution.format == "pdf"
        assert distribution.metadata["priority"] == "high"


class TestDataSourceInfo:
    """Test DataSourceInfo dataclass"""
    
    def test_creation(self):
        """Test creating DataSourceInfo"""
        source_info = DataSourceInfo(
            source_id="ds123",
            source_name="Customer Data",
            connection_type="database",
            dataset_id="dataset456",
            last_refresh=datetime.now(),
            refresh_status="success",
            row_count=10000,
            metadata={"table": "customers"}
        )
        
        assert source_info.source_id == "ds123"
        assert source_info.source_name == "Customer Data"
        assert source_info.connection_type == "database"
        assert source_info.dataset_id == "dataset456"
        assert source_info.refresh_status == "success"
        assert source_info.row_count == 10000


if __name__ == '__main__':
    pytest.main([__file__])