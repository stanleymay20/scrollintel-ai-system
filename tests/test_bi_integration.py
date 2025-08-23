"""
Tests for BI Integration System
Comprehensive test suite for all BI tool integrations
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from scrollintel.models.bi_integration_models import (
    BIToolType, ExportFormat, EmbedType, ConnectionStatus,
    BIConnectionConfig, DashboardExportRequest, EmbedTokenRequest,
    DataSyncRequest
)
from scrollintel.connectors.bi_connector_base import bi_connector_registry
from scrollintel.connectors.tableau_connector import TableauConnector
from scrollintel.connectors.power_bi_connector import PowerBIConnector
from scrollintel.connectors.looker_connector import LookerConnector
from scrollintel.engines.bi_integration_engine import BIIntegrationEngine


class TestBIConnectorBase:
    """Test the base BI connector framework"""
    
    def test_connector_registry(self):
        """Test connector registration"""
        # Check that connectors are registered
        assert bi_connector_registry.is_supported(BIToolType.TABLEAU)
        assert bi_connector_registry.is_supported(BIToolType.POWER_BI)
        assert bi_connector_registry.is_supported(BIToolType.LOOKER)
        
        # Check available tools
        available_tools = bi_connector_registry.get_available_tools()
        assert BIToolType.TABLEAU in available_tools
        assert BIToolType.POWER_BI in available_tools
        assert BIToolType.LOOKER in available_tools
    
    def test_get_connector(self):
        """Test getting connector instances"""
        config = {
            'id': 'test',
            'name': 'Test Connection',
            'bi_tool_type': BIToolType.TABLEAU,
            'server_url': 'https://test.tableau.com',
            'username': 'test_user',
            'password': 'test_pass',
            'site_id': 'default'
        }
        
        connector = bi_connector_registry.get_connector(BIToolType.TABLEAU, config)
        assert isinstance(connector, TableauConnector)
        assert connector.tool_type == BIToolType.TABLEAU


class TestTableauConnector:
    """Test Tableau connector implementation"""
    
    @pytest.fixture
    def tableau_config(self):
        return {
            'id': 'tableau-test',
            'name': 'Test Tableau',
            'bi_tool_type': BIToolType.TABLEAU,
            'server_url': 'https://test.tableau.com',
            'username': 'test_user',
            'password': 'test_pass',
            'site_id': 'default',
            'api_version': '3.19'
        }
    
    @pytest.fixture
    def tableau_connector(self, tableau_config):
        return TableauConnector(tableau_config)
    
    def test_tableau_initialization(self, tableau_connector):
        """Test Tableau connector initialization"""
        assert tableau_connector.tool_type == BIToolType.TABLEAU
        assert tableau_connector.server_url == 'https://test.tableau.com'
        assert tableau_connector.username == 'test_user'
        assert tableau_connector.site_id == 'default'
    
    def test_required_config_fields(self, tableau_connector):
        """Test required configuration fields"""
        required_fields = tableau_connector.get_required_config_fields()
        assert 'server_url' in required_fields
        assert 'username' in required_fields
        assert 'password' in required_fields
        assert 'site_id' in required_fields
    
    def test_supported_formats(self, tableau_connector):
        """Test supported export formats"""
        formats = tableau_connector.get_supported_export_formats()
        assert ExportFormat.PDF in formats
        assert ExportFormat.PNG in formats
        assert ExportFormat.CSV in formats
    
    @pytest.mark.asyncio
    async def test_validate_config(self, tableau_connector):
        """Test configuration validation"""
        errors = await tableau_connector.validate_config()
        assert len(errors) == 0  # Should be valid
        
        # Test with missing field
        tableau_connector.config['server_url'] = None
        errors = await tableau_connector.validate_config()
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_authentication(self, mock_post, tableau_connector):
        """Test Tableau authentication"""
        # Mock successful authentication response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = '''
        <tsResponse>
            <credentials token="test-token">
                <site id="site-uuid" />
            </credentials>
            <user id="user-id" />
        </tsResponse>
        '''
        mock_post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession'):
            result = await tableau_connector.authenticate()
            assert result is True
            assert tableau_connector.auth_token == "test-token"
            assert tableau_connector._authenticated is True


class TestPowerBIConnector:
    """Test Power BI connector implementation"""
    
    @pytest.fixture
    def powerbi_config(self):
        return {
            'id': 'powerbi-test',
            'name': 'Test Power BI',
            'bi_tool_type': BIToolType.POWER_BI,
            'tenant_id': 'test-tenant',
            'client_id': 'test-client',
            'client_secret': 'test-secret',
            'workspace_id': 'test-workspace'
        }
    
    @pytest.fixture
    def powerbi_connector(self, powerbi_config):
        return PowerBIConnector(powerbi_config)
    
    def test_powerbi_initialization(self, powerbi_connector):
        """Test Power BI connector initialization"""
        assert powerbi_connector.tool_type == BIToolType.POWER_BI
        assert powerbi_connector.tenant_id == 'test-tenant'
        assert powerbi_connector.client_id == 'test-client'
        assert powerbi_connector.workspace_id == 'test-workspace'
    
    def test_supported_formats(self, powerbi_connector):
        """Test supported export formats"""
        formats = powerbi_connector.get_supported_export_formats()
        assert ExportFormat.PDF in formats
        assert ExportFormat.PNG in formats
        assert ExportFormat.JPEG in formats
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_authentication(self, mock_post, powerbi_connector):
        """Test Power BI authentication"""
        # Mock successful authentication response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            'access_token': 'test-access-token',
            'expires_in': 3600
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession'):
            result = await powerbi_connector.authenticate()
            assert result is True
            assert powerbi_connector.access_token == "test-access-token"
            assert powerbi_connector._authenticated is True


class TestLookerConnector:
    """Test Looker connector implementation"""
    
    @pytest.fixture
    def looker_config(self):
        return {
            'id': 'looker-test',
            'name': 'Test Looker',
            'bi_tool_type': BIToolType.LOOKER,
            'base_url': 'https://test.looker.com',
            'client_id': 'test-client',
            'client_secret': 'test-secret',
            'embed_secret': 'embed-secret'
        }
    
    @pytest.fixture
    def looker_connector(self, looker_config):
        return LookerConnector(looker_config)
    
    def test_looker_initialization(self, looker_connector):
        """Test Looker connector initialization"""
        assert looker_connector.tool_type == BIToolType.LOOKER
        assert looker_connector.base_url == 'https://test.looker.com'
        assert looker_connector.client_id == 'test-client'
        assert looker_connector.embed_secret == 'embed-secret'
    
    def test_supported_formats(self, looker_connector):
        """Test supported export formats"""
        formats = looker_connector.get_supported_export_formats()
        assert ExportFormat.PDF in formats
        assert ExportFormat.PNG in formats
        assert ExportFormat.JPEG in formats
        assert ExportFormat.CSV in formats
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_authentication(self, mock_post, looker_connector):
        """Test Looker authentication"""
        # Mock successful authentication response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            'access_token': 'test-access-token',
            'expires_in': 3600
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession'):
            result = await looker_connector.authenticate()
            assert result is True
            assert looker_connector.access_token == "test-access-token"
            assert looker_connector._authenticated is True


class TestBIIntegrationEngine:
    """Test the main BI integration engine"""
    
    @pytest.fixture
    def bi_engine(self):
        return BIIntegrationEngine()
    
    @pytest.fixture
    def mock_db(self):
        db = Mock()
        db.add = Mock()
        db.commit = Mock()
        db.query = Mock()
        return db
    
    @pytest.fixture
    def sample_config(self):
        return BIConnectionConfig(
            name="Test Connection",
            bi_tool_type=BIToolType.TABLEAU,
            server_url="https://test.tableau.com",
            username="test_user",
            password="test_pass",
            site_id="default"
        )
    
    @pytest.mark.asyncio
    async def test_get_supported_tools(self, bi_engine):
        """Test getting supported BI tools"""
        tools = await bi_engine.get_supported_tools()
        assert len(tools) >= 3  # At least Tableau, Power BI, Looker
        
        tool_types = [tool['tool_type'] for tool in tools]
        assert 'tableau' in tool_types
        assert 'power_bi' in tool_types
        assert 'looker' in tool_types
    
    @pytest.mark.asyncio
    @patch('scrollintel.connectors.tableau_connector.TableauConnector.authenticate')
    @patch('scrollintel.connectors.tableau_connector.TableauConnector.test_connection')
    @patch('scrollintel.connectors.tableau_connector.TableauConnector.validate_config')
    async def test_create_connection(self, mock_validate, mock_test, mock_auth, bi_engine, mock_db, sample_config):
        """Test creating a BI connection"""
        # Mock successful validation and connection
        mock_validate.return_value = []
        mock_auth.return_value = True
        mock_test.return_value = {"status": "connected"}
        
        result = await bi_engine.create_connection(sample_config, mock_db)
        
        assert result.name == "Test Connection"
        assert result.bi_tool_type == BIToolType.TABLEAU
        assert result.status == ConnectionStatus.ACTIVE
        assert mock_db.add.called
        assert mock_db.commit.called
    
    @pytest.mark.asyncio
    async def test_export_dashboard(self, bi_engine, mock_db):
        """Test dashboard export functionality"""
        # Mock connection
        mock_connector = AsyncMock()
        mock_connector.export_dashboard.return_value = b"fake-pdf-content"
        
        bi_engine.active_connections["test-conn"] = mock_connector
        
        export_request = DashboardExportRequest(
            dashboard_id="test-dashboard",
            format=ExportFormat.PDF
        )
        
        with patch('scrollintel.engines.bi_integration_engine.BIExportJob'):
            job_id = await bi_engine.export_dashboard("test-conn", export_request, mock_db)
            assert job_id is not None
            assert mock_db.add.called
    
    @pytest.mark.asyncio
    async def test_create_embed_token(self, bi_engine):
        """Test embed token creation"""
        # Mock connection
        mock_connector = AsyncMock()
        mock_connector.create_embed_token.return_value = Mock(
            token="test-token",
            embed_url="https://test.com/embed",
            expires_at=datetime.utcnow(),
            embed_code="<iframe>...</iframe>"
        )
        
        bi_engine.active_connections["test-conn"] = mock_connector
        
        embed_request = EmbedTokenRequest(
            dashboard_id="test-dashboard",
            user_id="test-user",
            embed_type=EmbedType.IFRAME
        )
        
        result = await bi_engine.create_embed_token("test-conn", embed_request, Mock())
        assert result.token == "test-token"
        assert "iframe" in result.embed_code.lower()
    
    @pytest.mark.asyncio
    async def test_sync_data_source(self, bi_engine):
        """Test data source synchronization"""
        # Mock connection
        mock_connector = AsyncMock()
        mock_connector.sync_data_source.return_value = Mock(
            success=True,
            records_processed=100,
            records_updated=50,
            records_created=25,
            errors=[],
            sync_duration=5.0,
            last_sync_time=datetime.utcnow()
        )
        
        bi_engine.active_connections["test-conn"] = mock_connector
        
        sync_request = DataSyncRequest(
            connection_id="test-conn",
            data_source_id="test-datasource",
            incremental=True
        )
        
        result = await bi_engine.sync_data_source("test-conn", sync_request, Mock())
        assert result.success is True
        assert result.records_processed == 100


class TestBIIntegrationAPI:
    """Test BI integration API endpoints"""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from scrollintel.api.routes.bi_integration_routes import router
        
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_get_supported_tools(self, client):
        """Test getting supported tools endpoint"""
        with patch('scrollintel.engines.bi_integration_engine.bi_integration_engine.get_supported_tools') as mock_get:
            mock_get.return_value = [
                {
                    'tool_type': 'tableau',
                    'name': 'Tableau',
                    'supported_export_formats': ['pdf', 'png', 'csv'],
                    'supported_embed_types': ['iframe', 'javascript'],
                    'required_config_fields': ['server_url', 'username', 'password']
                }
            ]
            
            response = client.get("/api/v1/bi-integration/supported-tools")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['tool_type'] == 'tableau'
    
    def test_create_connection_validation(self, client):
        """Test connection creation with validation"""
        # Test with missing required fields
        invalid_config = {
            "name": "Test Connection",
            "bi_tool_type": "tableau"
            # Missing server_url, username, password
        }
        
        with patch('scrollintel.models.database.get_db'):
            response = client.post("/api/v1/bi-integration/connections", json=invalid_config)
            assert response.status_code == 422  # Validation error


class TestWhiteLabelEmbedding:
    """Test white-label embedding capabilities"""
    
    @pytest.mark.asyncio
    async def test_white_label_embed_customization(self):
        """Test white-label embed customization"""
        from scrollintel.models.bi_integration_models import EmbedTokenResponse, EmbedType
        
        # Mock embed response
        embed_response = EmbedTokenResponse(
            token="test-token",
            embed_url="https://test.com/embed/dashboard/123",
            expires_at=datetime.utcnow(),
            embed_code=None
        )
        
        # Test white-label customization
        white_label_config = {
            "embed_url": embed_response.embed_url,
            "token": embed_response.token,
            "theme": "custom",
            "branding": {
                "hide_toolbar": True,
                "hide_tabs": True,
                "custom_css": ".bi-toolbar { display: none !important; }"
            }
        }
        
        assert white_label_config["branding"]["hide_toolbar"] is True
        assert "display: none" in white_label_config["branding"]["custom_css"]


class TestRealTimeDataFeeds:
    """Test real-time data feed capabilities"""
    
    @pytest.mark.asyncio
    async def test_power_bi_real_time_feed(self):
        """Test Power BI real-time data feed setup"""
        config = {
            'tenant_id': 'test-tenant',
            'client_id': 'test-client',
            'client_secret': 'test-secret',
            'workspace_id': 'test-workspace'
        }
        
        connector = PowerBIConnector(config)
        
        # Mock authentication
        connector._authenticated = True
        connector.access_token = "test-token"
        
        result = await connector.get_real_time_data_feed("test-dataset")
        
        assert result["supported"] is True
        assert "streaming_endpoint" in result
        assert "POST" in result["method"]
    
    @pytest.mark.asyncio
    async def test_tableau_real_time_limitations(self):
        """Test Tableau real-time limitations"""
        config = {
            'server_url': 'https://test.tableau.com',
            'username': 'test',
            'password': 'test',
            'site_id': 'default'
        }
        
        connector = TableauConnector(config)
        
        result = await connector.get_real_time_data_feed("test-datasource")
        
        assert result["supported"] is False
        assert "alternatives" in result


if __name__ == "__main__":
    pytest.main([__file__])