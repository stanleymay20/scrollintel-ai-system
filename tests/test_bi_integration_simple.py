"""
Simple BI Integration Tests
Basic tests to verify BI integration functionality
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from scrollintel.models.bi_integration_models import (
    BIToolType, ExportFormat, EmbedType, 
    BIConnectionConfig
)
from scrollintel.connectors.bi_connector_base import bi_connector_registry
from scrollintel.connectors.tableau_connector import TableauConnector
from scrollintel.connectors.power_bi_connector import PowerBIConnector
from scrollintel.connectors.looker_connector import LookerConnector


def test_bi_connector_registry():
    """Test that all BI connectors are registered"""
    # Check that connectors are registered
    assert bi_connector_registry.is_supported(BIToolType.TABLEAU)
    assert bi_connector_registry.is_supported(BIToolType.POWER_BI)
    assert bi_connector_registry.is_supported(BIToolType.LOOKER)
    
    # Check available tools
    available_tools = bi_connector_registry.get_available_tools()
    assert BIToolType.TABLEAU in available_tools
    assert BIToolType.POWER_BI in available_tools
    assert BIToolType.LOOKER in available_tools


def test_tableau_connector_creation():
    """Test creating Tableau connector"""
    config = {
        'id': 'test',
        'name': 'Test Tableau',
        'bi_tool_type': BIToolType.TABLEAU,
        'server_url': 'https://test.tableau.com',
        'username': 'test_user',
        'password': 'test_pass',
        'site_id': 'default'
    }
    
    connector = bi_connector_registry.get_connector(BIToolType.TABLEAU, config)
    assert isinstance(connector, TableauConnector)
    assert connector.tool_type == BIToolType.TABLEAU
    assert connector.server_url == 'https://test.tableau.com'


def test_power_bi_connector_creation():
    """Test creating Power BI connector"""
    config = {
        'id': 'test',
        'name': 'Test Power BI',
        'bi_tool_type': BIToolType.POWER_BI,
        'tenant_id': 'test-tenant',
        'client_id': 'test-client',
        'client_secret': 'test-secret',
        'workspace_id': 'test-workspace'
    }
    
    connector = bi_connector_registry.get_connector(BIToolType.POWER_BI, config)
    assert isinstance(connector, PowerBIConnector)
    assert connector.tool_type == BIToolType.POWER_BI
    assert connector.tenant_id == 'test-tenant'


def test_looker_connector_creation():
    """Test creating Looker connector"""
    config = {
        'id': 'test',
        'name': 'Test Looker',
        'bi_tool_type': BIToolType.LOOKER,
        'base_url': 'https://test.looker.com',
        'client_id': 'test-client',
        'client_secret': 'test-secret'
    }
    
    connector = bi_connector_registry.get_connector(BIToolType.LOOKER, config)
    assert isinstance(connector, LookerConnector)
    assert connector.tool_type == BIToolType.LOOKER
    assert connector.base_url == 'https://test.looker.com'


def test_supported_export_formats():
    """Test that connectors support expected export formats"""
    tableau_config = {
        'server_url': 'https://test.tableau.com',
        'username': 'test', 'password': 'test', 'site_id': 'default'
    }
    tableau_connector = TableauConnector(tableau_config)
    tableau_formats = tableau_connector.get_supported_export_formats()
    assert ExportFormat.PDF in tableau_formats
    assert ExportFormat.PNG in tableau_formats
    
    powerbi_config = {
        'tenant_id': 'test', 'client_id': 'test', 
        'client_secret': 'test', 'workspace_id': 'test'
    }
    powerbi_connector = PowerBIConnector(powerbi_config)
    powerbi_formats = powerbi_connector.get_supported_export_formats()
    assert ExportFormat.PDF in powerbi_formats
    assert ExportFormat.PNG in powerbi_formats
    
    looker_config = {
        'base_url': 'https://test.looker.com',
        'client_id': 'test', 'client_secret': 'test'
    }
    looker_connector = LookerConnector(looker_config)
    looker_formats = looker_connector.get_supported_export_formats()
    assert ExportFormat.PDF in looker_formats
    assert ExportFormat.CSV in looker_formats


def test_supported_embed_types():
    """Test that connectors support expected embed types"""
    tableau_config = {
        'server_url': 'https://test.tableau.com',
        'username': 'test', 'password': 'test', 'site_id': 'default'
    }
    tableau_connector = TableauConnector(tableau_config)
    embed_types = tableau_connector.get_supported_embed_types()
    assert EmbedType.IFRAME in embed_types
    
    powerbi_config = {
        'tenant_id': 'test', 'client_id': 'test', 
        'client_secret': 'test', 'workspace_id': 'test'
    }
    powerbi_connector = PowerBIConnector(powerbi_config)
    embed_types = powerbi_connector.get_supported_embed_types()
    assert EmbedType.IFRAME in embed_types
    assert EmbedType.JAVASCRIPT in embed_types


def test_required_config_fields():
    """Test that connectors specify required configuration fields"""
    tableau_config = {'server_url': 'test', 'username': 'test', 'password': 'test', 'site_id': 'test'}
    tableau_connector = TableauConnector(tableau_config)
    required_fields = tableau_connector.get_required_config_fields()
    assert 'server_url' in required_fields
    assert 'username' in required_fields
    assert 'password' in required_fields
    
    powerbi_config = {'tenant_id': 'test', 'client_id': 'test', 'client_secret': 'test', 'workspace_id': 'test'}
    powerbi_connector = PowerBIConnector(powerbi_config)
    required_fields = powerbi_connector.get_required_config_fields()
    assert 'tenant_id' in required_fields
    assert 'client_id' in required_fields
    assert 'client_secret' in required_fields


@pytest.mark.asyncio
async def test_config_validation():
    """Test configuration validation"""
    # Valid config
    valid_config = {
        'server_url': 'https://test.tableau.com',
        'username': 'test_user',
        'password': 'test_pass',
        'site_id': 'default'
    }
    tableau_connector = TableauConnector(valid_config)
    errors = await tableau_connector.validate_config()
    assert len(errors) == 0
    
    # Invalid config (missing required field)
    invalid_config = {
        'server_url': 'https://test.tableau.com',
        'username': 'test_user'
        # Missing password and site_id
    }
    tableau_connector = TableauConnector(invalid_config)
    errors = await tableau_connector.validate_config()
    assert len(errors) > 0


def test_bi_connection_config_model():
    """Test BIConnectionConfig Pydantic model"""
    config = BIConnectionConfig(
        name="Test Connection",
        bi_tool_type=BIToolType.TABLEAU,
        server_url="https://test.tableau.com",
        username="test_user",
        password="test_pass",
        site_id="default"
    )
    
    assert config.name == "Test Connection"
    assert config.bi_tool_type == BIToolType.TABLEAU
    assert config.server_url == "https://test.tableau.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])