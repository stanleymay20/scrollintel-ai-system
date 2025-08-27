"""
Comprehensive integration tests for all data connector implementations.
Tests error handling, retry logic, and connector functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from scrollintel.core.data_connector import (
    DataSourceConfig, DataSourceType, DataIntegrationManager,
    ConnectionStatus, ConnectorError, ConnectionError, 
    AuthenticationError, DataFetchError, TimeoutError
)
from scrollintel.core.data_integration_setup import setup_data_integration, create_sample_configurations
from scrollintel.connectors.erp_connectors import SAPConnector, OracleERPConnector, MicrosoftDynamicsConnector
from scrollintel.connectors.crm_connectors import SalesforceConnector, HubSpotConnector, MicrosoftCRMConnector
from scrollintel.connectors.bi_connectors import TableauConnector, PowerBIConnector, LookerConnector, QlikConnector
from scrollintel.connectors.cloud_connectors import AWSConnector, AzureConnector, GCPConnector


class TestERPConnectors:
    """Test ERP system connectors"""
    
    @pytest.fixture
    def sap_config(self):
        """SAP connector configuration"""
        return DataSourceConfig(
            source_id='test_sap',
            source_type=DataSourceType.ERP,
            name='Test SAP System',
            connection_params={
                'host': 'test-sap.company.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.fixture
    def oracle_config(self):
        """Oracle ERP connector configuration"""
        return DataSourceConfig(
            source_id='test_oracle',
            source_type=DataSourceType.ERP,
            name='Test Oracle ERP',
            connection_params={
                'base_url': 'https://test-oracle.company.com',
                'username': 'test_user',
                'password': 'test_password'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.fixture
    def dynamics_config(self):
        """Microsoft Dynamics connector configuration"""
        return DataSourceConfig(
            source_id='test_dynamics',
            source_type=DataSourceType.ERP,
            name='Test Dynamics 365',
            connection_params={
                'org_url': 'https://test.crm.dynamics.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_sap_connector_connection(self, sap_config):
        """Test SAP connector connection and basic operations"""
        connector = SAPConnector(sap_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        assert connector.status == ConnectionStatus.CONNECTED
        
        # Test connection validation
        await connector.test_connection()
        
        # Test data fetching
        query = {
            'table': 'MARA',
            'fields': ['MATNR', 'MAKTX'],
            'limit': 10
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        assert all(record.source_id == sap_config.source_id for record in data)
        
        # Test schema retrieval
        schema = await connector.get_schema()
        assert 'tables' in schema
        assert 'MARA' in schema['tables']
        
        # Test disconnect
        await connector.disconnect()
        assert connector.status == ConnectionStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_oracle_connector_connection(self, oracle_config):
        """Test Oracle ERP connector connection and operations"""
        connector = OracleERPConnector(oracle_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        assert connector.status == ConnectionStatus.CONNECTED
        
        # Test data fetching
        query = {
            'resource': 'items',
            'limit': 5
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        # Test schema retrieval
        schema = await connector.get_schema()
        assert 'resources' in schema
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_dynamics_connector_connection(self, dynamics_config):
        """Test Microsoft Dynamics connector connection and operations"""
        connector = MicrosoftDynamicsConnector(dynamics_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        assert connector.status == ConnectionStatus.CONNECTED
        
        # Test data fetching
        query = {
            'entity': 'products',
            'limit': 5
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_missing_parameters_validation(self):
        """Test validation of missing required parameters"""
        # Test SAP with missing parameters
        invalid_config = DataSourceConfig(
            source_id='invalid_sap',
            source_type=DataSourceType.ERP,
            name='Invalid SAP',
            connection_params={'host': 'test-host'}  # Missing required params
        )
        
        connector = SAPConnector(invalid_config)
        
        with pytest.raises(ValueError, match="Missing required parameter"):
            await connector.validate_connection_params()


class TestCRMConnectors:
    """Test CRM system connectors"""
    
    @pytest.fixture
    def salesforce_config(self):
        """Salesforce connector configuration"""
        return DataSourceConfig(
            source_id='test_salesforce',
            source_type=DataSourceType.CRM,
            name='Test Salesforce',
            connection_params={
                'instance_url': 'https://test.salesforce.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'username': 'test@company.com',
                'password': 'test_password',
                'security_token': 'test_token'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.fixture
    def hubspot_config(self):
        """HubSpot connector configuration"""
        return DataSourceConfig(
            source_id='test_hubspot',
            source_type=DataSourceType.CRM,
            name='Test HubSpot',
            connection_params={
                'access_token': 'test_access_token'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_salesforce_connector(self, salesforce_config):
        """Test Salesforce connector functionality"""
        connector = SalesforceConnector(salesforce_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        # Test data fetching
        query = {
            'sobject': 'Account',
            'fields': ['Id', 'Name'],
            'limit': 5
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        # Test schema retrieval
        schema = await connector.get_schema()
        assert 'sobjects' in schema
        assert 'Account' in schema['sobjects']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_hubspot_connector(self, hubspot_config):
        """Test HubSpot connector functionality"""
        connector = HubSpotConnector(hubspot_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        # Test data fetching
        query = {
            'object_type': 'companies',
            'limit': 5
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_hubspot_parameter_validation(self):
        """Test HubSpot custom parameter validation"""
        # Test with no API key or access token
        invalid_config = DataSourceConfig(
            source_id='invalid_hubspot',
            source_type=DataSourceType.CRM,
            name='Invalid HubSpot',
            connection_params={}  # Missing required params
        )
        
        connector = HubSpotConnector(invalid_config)
        
        with pytest.raises(ValueError, match="Either api_key or access_token is required"):
            await connector.validate_connection_params()


class TestBIConnectors:
    """Test BI tool connectors"""
    
    @pytest.fixture
    def tableau_config(self):
        """Tableau connector configuration"""
        return DataSourceConfig(
            source_id='test_tableau',
            source_type=DataSourceType.BI_TOOL,
            name='Test Tableau',
            connection_params={
                'server_url': 'https://tableau.company.com',
                'username': 'test_user',
                'password': 'test_password',
                'site_id': 'default'
            },
            timeout=60,
            retry_attempts=2
        )
    
    @pytest.fixture
    def powerbi_config(self):
        """Power BI connector configuration"""
        return DataSourceConfig(
            source_id='test_powerbi',
            source_type=DataSourceType.BI_TOOL,
            name='Test Power BI',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            },
            timeout=60,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_tableau_connector(self, tableau_config):
        """Test Tableau connector functionality"""
        connector = TableauConnector(tableau_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        # Test data fetching
        query = {
            'resource_type': 'workbooks',
            'limit': 5
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        # Test schema retrieval
        schema = await connector.get_schema()
        assert 'resources' in schema
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_powerbi_connector(self, powerbi_config):
        """Test Power BI connector functionality"""
        connector = PowerBIConnector(powerbi_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        # Test data fetching
        query = {
            'resource_type': 'reports',
            'limit': 5
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        await connector.disconnect()


class TestCloudConnectors:
    """Test cloud platform connectors"""
    
    @pytest.fixture
    def aws_config(self):
        """AWS connector configuration"""
        return DataSourceConfig(
            source_id='test_aws',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test AWS',
            connection_params={
                'access_key_id': 'test_access_key',
                'secret_access_key': 'test_secret_key',
                'region': 'us-east-1'
            },
            timeout=120,
            retry_attempts=3
        )
    
    @pytest.fixture
    def azure_config(self):
        """Azure connector configuration"""
        return DataSourceConfig(
            source_id='test_azure',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test Azure',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id',
                'subscription_id': 'test_subscription_id'
            },
            timeout=120,
            retry_attempts=3
        )
    
    @pytest.fixture
    def gcp_config(self):
        """GCP connector configuration"""
        return DataSourceConfig(
            source_id='test_gcp',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test GCP',
            connection_params={
                'service_account_key': '{"type": "service_account"}',
                'project_id': 'test-project',
                'billing_account_id': 'test-billing-account'
            },
            timeout=120,
            retry_attempts=3
        )
    
    @pytest.mark.asyncio
    async def test_aws_connector(self, aws_config):
        """Test AWS connector functionality"""
        connector = AWSConnector(aws_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        # Test cost data fetching
        query = {
            'service': 'cost-explorer',
            'granularity': 'DAILY',
            'metrics': ['BlendedCost']
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        # Test CloudWatch data fetching
        query = {
            'service': 'cloudwatch'
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_azure_connector(self, azure_config):
        """Test Azure connector functionality"""
        connector = AzureConnector(azure_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        # Test cost management data
        query = {
            'service': 'cost-management',
            'granularity': 'Daily'
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_gcp_connector(self, gcp_config):
        """Test GCP connector functionality"""
        connector = GCPConnector(gcp_config)
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        # Test billing data
        query = {
            'service': 'billing'
        }
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        await connector.disconnect()


class TestDataIntegrationManager:
    """Test the data integration manager with all connectors"""
    
    @pytest.fixture
    def integration_manager(self):
        """Setup integration manager with all connectors"""
        return setup_data_integration()
    
    @pytest.mark.asyncio
    async def test_connector_registration(self, integration_manager):
        """Test that all connectors are properly registered"""
        registry = integration_manager.registry
        
        # Check ERP connectors
        assert 'sap' in registry.connector_classes
        assert 'oracle_erp' in registry.connector_classes
        assert 'microsoft_dynamics' in registry.connector_classes
        
        # Check CRM connectors
        assert 'salesforce' in registry.connector_classes
        assert 'hubspot' in registry.connector_classes
        assert 'microsoft_crm' in registry.connector_classes
        
        # Check BI connectors
        assert 'tableau' in registry.connector_classes
        assert 'powerbi' in registry.connector_classes
        assert 'looker' in registry.connector_classes
        assert 'qlik' in registry.connector_classes
        
        # Check cloud connectors
        assert 'aws' in registry.connector_classes
        assert 'azure' in registry.connector_classes
        assert 'gcp' in registry.connector_classes
    
    @pytest.mark.asyncio
    async def test_sample_configurations(self, integration_manager):
        """Test sample configurations for all connector types"""
        configurations = create_sample_configurations()
        
        # Test that we have configurations for all connector types
        expected_connectors = [
            'sap', 'oracle_erp', 'salesforce', 'hubspot', 
            'tableau', 'powerbi', 'aws', 'azure', 'gcp'
        ]
        
        for connector_type in expected_connectors:
            assert connector_type in configurations
            config = configurations[connector_type]
            assert config.source_id is not None
            assert config.source_type is not None
            assert config.connection_params is not None
    
    @pytest.mark.asyncio
    async def test_connector_creation_and_health_check(self, integration_manager):
        """Test connector creation and health monitoring"""
        # Create a test SAP connector
        sap_config = DataSourceConfig(
            source_id='health_test_sap',
            source_type=DataSourceType.ERP,
            name='Health Test SAP',
            connection_params={
                'host': 'test-sap.company.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
        
        # Add data source
        success = await integration_manager.add_data_source(sap_config)
        assert success is True
        
        # Check health status
        health_status = await integration_manager.get_health_status()
        assert 'health_test_sap' in health_status
        
        health = health_status['health_test_sap']
        assert health.source_id == 'health_test_sap'
        assert health.status == ConnectionStatus.CONNECTED
        
        # Test connection
        test_results = await integration_manager.test_all_connections()
        assert 'health_test_sap' in test_results
        assert test_results['health_test_sap'] is True
        
        # Remove data source
        removed = await integration_manager.remove_data_source('health_test_sap')
        assert removed is True


class TestErrorHandlingAndRetry:
    """Test error handling and retry logic across all connectors"""
    
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test connection timeout handling"""
        config = DataSourceConfig(
            source_id='timeout_test',
            source_type=DataSourceType.ERP,
            name='Timeout Test',
            connection_params={
                'host': 'test-sap.company.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            },
            timeout=0.1  # Very short timeout to trigger timeout error
        )
        
        connector = SAPConnector(config)
        
        # This should eventually succeed due to retry logic, but may timeout
        try:
            result = await connector.connect_with_retry()
            # If it succeeds, that's fine too
            assert result is True
        except TimeoutError:
            # Timeout is expected with very short timeout
            assert connector.status == ConnectionStatus.ERROR
            assert "timeout" in connector.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_data_fetch_error_handling(self):
        """Test data fetch error handling and retry"""
        config = DataSourceConfig(
            source_id='fetch_error_test',
            source_type=DataSourceType.ERP,
            name='Fetch Error Test',
            connection_params={
                'host': 'test-sap.company.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
        
        connector = SAPConnector(config)
        await connector.connect_with_retry()
        
        # Test with invalid query parameters
        invalid_query = {
            'table': 'INVALID_TABLE',
            'limit': 999999  # Exceeds maximum
        }
        
        try:
            await connector.fetch_data_with_retry(invalid_query)
        except (ValueError, DataFetchError):
            # Expected error for invalid parameters
            pass
    
    @pytest.mark.asyncio
    async def test_health_monitoring_after_errors(self):
        """Test health monitoring captures error states"""
        config = DataSourceConfig(
            source_id='health_error_test',
            source_type=DataSourceType.ERP,
            name='Health Error Test',
            connection_params={
                'host': 'test-sap.company.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
        
        connector = SAPConnector(config)
        
        # Force an error by trying to fetch data without connecting
        try:
            await connector.fetch_data_with_retry({'table': 'TEST'})
        except ConnectionError:
            pass  # Expected
        
        # Check health reflects the error
        health = connector.get_health()
        assert health.status == ConnectionStatus.DISCONNECTED
        assert health.last_successful_sync is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])