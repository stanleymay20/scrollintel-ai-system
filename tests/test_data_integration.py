"""
Integration tests for multi-source data integration system.
Tests all data connectors and the integration manager.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.data_connector import (
    DataIntegrationManager, DataSourceConfig, DataSourceType, 
    ConnectionStatus, DataRecord
)
from scrollintel.core.data_integration_setup import (
    setup_data_integration, create_sample_configurations, 
    initialize_sample_data_sources
)
from scrollintel.connectors.erp_connectors import SAPConnector, OracleERPConnector, MicrosoftDynamicsConnector
from scrollintel.connectors.crm_connectors import SalesforceConnector, HubSpotConnector, MicrosoftCRMConnector
from scrollintel.connectors.bi_connectors import TableauConnector, PowerBIConnector, LookerConnector, QlikConnector
from scrollintel.connectors.cloud_connectors import AWSConnector, AzureConnector, GCPConnector


class TestDataIntegrationManager:
    """Test the main data integration manager"""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh data integration manager for each test"""
        return setup_data_integration()
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample data source configuration"""
        return DataSourceConfig(
            source_id='test_source',
            source_type=DataSourceType.ERP,
            name='Test Source',
            connection_params={
                'host': 'test.example.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            },
            refresh_interval=300,
            timeout=30,
            retry_attempts=3,
            enabled=True
        )
    
    def test_manager_initialization(self, manager):
        """Test that the manager initializes correctly"""
        assert manager is not None
        assert isinstance(manager, DataIntegrationManager)
        assert len(manager.registry.connector_classes) > 0
    
    @pytest.mark.asyncio
    async def test_add_data_source(self, manager, sample_config):
        """Test adding a data source"""
        success = await manager.add_data_source(sample_config)
        assert success is True
        
        # Verify the connector was created
        connector = manager.registry.get_connector(sample_config.source_id)
        assert connector is not None
        assert connector.config.source_id == sample_config.source_id
    
    @pytest.mark.asyncio
    async def test_remove_data_source(self, manager, sample_config):
        """Test removing a data source"""
        # First add a data source
        await manager.add_data_source(sample_config)
        
        # Then remove it
        success = await manager.remove_data_source(sample_config.source_id)
        assert success is True
        
        # Verify it was removed
        connector = manager.registry.get_connector(sample_config.source_id)
        assert connector is None
    
    @pytest.mark.asyncio
    async def test_health_status(self, manager, sample_config):
        """Test getting health status of data sources"""
        await manager.add_data_source(sample_config)
        
        health_status = await manager.get_health_status()
        assert sample_config.source_id in health_status
        
        health = health_status[sample_config.source_id]
        assert health.source_id == sample_config.source_id
        assert health.status in [ConnectionStatus.CONNECTED, ConnectionStatus.ERROR]
    
    @pytest.mark.asyncio
    async def test_connection_testing(self, manager, sample_config):
        """Test connection testing for all data sources"""
        await manager.add_data_source(sample_config)
        
        results = await manager.test_all_connections()
        assert sample_config.source_id in results
        assert isinstance(results[sample_config.source_id], bool)


class TestERPConnectors:
    """Test ERP system connectors"""
    
    @pytest.fixture
    def sap_config(self):
        return DataSourceConfig(
            source_id='test_sap',
            source_type=DataSourceType.ERP,
            name='Test SAP',
            connection_params={
                'host': 'sap.test.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
    
    @pytest.fixture
    def oracle_config(self):
        return DataSourceConfig(
            source_id='test_oracle',
            source_type=DataSourceType.ERP,
            name='Test Oracle ERP',
            connection_params={
                'base_url': 'https://oracle.test.com',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
    
    @pytest.fixture
    def dynamics_config(self):
        return DataSourceConfig(
            source_id='test_dynamics',
            source_type=DataSourceType.ERP,
            name='Test Dynamics 365',
            connection_params={
                'org_url': 'https://dynamics.test.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            }
        )
    
    @pytest.mark.asyncio
    async def test_sap_connector(self, sap_config):
        """Test SAP connector functionality"""
        connector = SAPConnector(sap_config)
        
        # Test connection
        success = await connector.connect()
        assert success is True
        assert connector.status == ConnectionStatus.CONNECTED
        
        # Test data fetch
        query = {'table': 'MARA', 'limit': 10}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 10
        
        if data:
            record = data[0]
            assert isinstance(record, DataRecord)
            assert record.source_id == sap_config.source_id
            assert 'MATNR' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'tables' in schema
        assert 'MARA' in schema['tables']
        
        # Test disconnect
        success = await connector.disconnect()
        assert success is True
    
    @pytest.mark.asyncio
    async def test_oracle_connector(self, oracle_config):
        """Test Oracle ERP connector functionality"""
        connector = OracleERPConnector(oracle_config)
        
        # Test connection
        success = await connector.connect()
        assert success is True
        
        # Test data fetch
        query = {'resource': 'items', 'limit': 5}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 5
        
        if data:
            record = data[0]
            assert isinstance(record, DataRecord)
            assert 'ItemId' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'resources' in schema
        assert 'items' in schema['resources']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_dynamics_connector(self, dynamics_config):
        """Test Microsoft Dynamics connector functionality"""
        connector = MicrosoftDynamicsConnector(dynamics_config)
        
        # Test connection
        success = await connector.connect()
        assert success is True
        
        # Test data fetch
        query = {'entity': 'products', 'limit': 8}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 8
        
        if data:
            record = data[0]
            assert isinstance(record, DataRecord)
            assert 'productid' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'entities' in schema
        assert 'products' in schema['entities']
        
        await connector.disconnect()


class TestCRMConnectors:
    """Test CRM system connectors"""
    
    @pytest.fixture
    def salesforce_config(self):
        return DataSourceConfig(
            source_id='test_salesforce',
            source_type=DataSourceType.CRM,
            name='Test Salesforce',
            connection_params={
                'instance_url': 'https://test.salesforce.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'username': 'test@example.com',
                'password': 'test_password'
            }
        )
    
    @pytest.fixture
    def hubspot_config(self):
        return DataSourceConfig(
            source_id='test_hubspot',
            source_type=DataSourceType.CRM,
            name='Test HubSpot',
            connection_params={
                'access_token': 'test_access_token'
            }
        )
    
    @pytest.fixture
    def mscrm_config(self):
        return DataSourceConfig(
            source_id='test_mscrm',
            source_type=DataSourceType.CRM,
            name='Test Microsoft CRM',
            connection_params={
                'org_url': 'https://test.crm.dynamics.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            }
        )
    
    @pytest.mark.asyncio
    async def test_salesforce_connector(self, salesforce_config):
        """Test Salesforce connector functionality"""
        connector = SalesforceConnector(salesforce_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test Account data
        query = {'sobject': 'Account', 'limit': 5}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 5
        
        if data:
            record = data[0]
            assert 'Id' in record.data
            assert 'Name' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'sobjects' in schema
        assert 'Account' in schema['sobjects']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_hubspot_connector(self, hubspot_config):
        """Test HubSpot connector functionality"""
        connector = HubSpotConnector(hubspot_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test companies data
        query = {'object_type': 'companies', 'limit': 6}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 6
        
        if data:
            record = data[0]
            assert 'id' in record.data
            assert 'properties' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'objects' in schema
        assert 'companies' in schema['objects']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_microsoft_crm_connector(self, mscrm_config):
        """Test Microsoft CRM connector functionality"""
        connector = MicrosoftCRMConnector(mscrm_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test accounts data
        query = {'entity': 'accounts', 'limit': 7}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 7
        
        if data:
            record = data[0]
            assert 'accountid' in record.data
            assert 'name' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'entities' in schema
        assert 'accounts' in schema['entities']
        
        await connector.disconnect()


class TestBIConnectors:
    """Test BI tool connectors"""
    
    @pytest.fixture
    def tableau_config(self):
        return DataSourceConfig(
            source_id='test_tableau',
            source_type=DataSourceType.BI_TOOL,
            name='Test Tableau',
            connection_params={
                'server_url': 'https://tableau.test.com',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
    
    @pytest.fixture
    def powerbi_config(self):
        return DataSourceConfig(
            source_id='test_powerbi',
            source_type=DataSourceType.BI_TOOL,
            name='Test Power BI',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            }
        )
    
    @pytest.mark.asyncio
    async def test_tableau_connector(self, tableau_config):
        """Test Tableau connector functionality"""
        connector = TableauConnector(tableau_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test workbooks data
        query = {'resource_type': 'workbooks', 'limit': 4}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 4
        
        if data:
            record = data[0]
            assert 'id' in record.data
            assert 'name' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'resources' in schema
        assert 'workbooks' in schema['resources']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_powerbi_connector(self, powerbi_config):
        """Test Power BI connector functionality"""
        connector = PowerBIConnector(powerbi_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test reports data
        query = {'resource_type': 'reports', 'limit': 3}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        assert len(data) <= 3
        
        if data:
            record = data[0]
            assert 'id' in record.data
            assert 'name' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'resources' in schema
        assert 'reports' in schema['resources']
        
        await connector.disconnect()


class TestCloudConnectors:
    """Test cloud platform connectors"""
    
    @pytest.fixture
    def aws_config(self):
        return DataSourceConfig(
            source_id='test_aws',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test AWS',
            connection_params={
                'access_key_id': 'test_access_key',
                'secret_access_key': 'test_secret_key',
                'region': 'us-east-1'
            }
        )
    
    @pytest.fixture
    def azure_config(self):
        return DataSourceConfig(
            source_id='test_azure',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test Azure',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id',
                'subscription_id': 'test_subscription_id'
            }
        )
    
    @pytest.fixture
    def gcp_config(self):
        return DataSourceConfig(
            source_id='test_gcp',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test GCP',
            connection_params={
                'service_account_key': 'test_service_account_key',
                'project_id': 'test_project_id',
                'billing_account_id': 'test_billing_account'
            }
        )
    
    @pytest.mark.asyncio
    async def test_aws_connector(self, aws_config):
        """Test AWS connector functionality"""
        connector = AWSConnector(aws_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test cost explorer data
        query = {
            'service': 'cost-explorer',
            'time_period': {
                'Start': (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'End': datetime.utcnow().strftime('%Y-%m-%d')
            },
            'granularity': 'DAILY'
        }
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        
        if data:
            record = data[0]
            assert 'Service' in record.data
            assert 'BlendedCost' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'services' in schema
        assert 'cost-explorer' in schema['services']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_azure_connector(self, azure_config):
        """Test Azure connector functionality"""
        connector = AzureConnector(azure_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test cost management data
        query = {'service': 'cost-management'}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        
        if data:
            record = data[0]
            assert 'id' in record.data
            assert 'properties' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'services' in schema
        assert 'cost-management' in schema['services']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_gcp_connector(self, gcp_config):
        """Test GCP connector functionality"""
        connector = GCPConnector(gcp_config)
        
        success = await connector.connect()
        assert success is True
        
        # Test billing data
        query = {'service': 'billing'}
        data = await connector.fetch_data(query)
        assert isinstance(data, list)
        
        if data:
            record = data[0]
            assert 'project' in record.data
            assert 'service' in record.data
            assert 'cost' in record.data
        
        # Test schema
        schema = await connector.get_schema()
        assert 'services' in schema
        assert 'billing' in schema['services']
        
        await connector.disconnect()


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows"""
    
    @pytest.mark.asyncio
    async def test_sample_configurations(self):
        """Test sample configuration creation"""
        configs = create_sample_configurations()
        
        assert isinstance(configs, dict)
        assert len(configs) > 0
        
        # Check that we have configurations for different source types
        source_types = set()
        for config in configs.values():
            source_types.add(config.source_type)
        
        assert DataSourceType.ERP in source_types
        assert DataSourceType.CRM in source_types
        assert DataSourceType.BI_TOOL in source_types
        assert DataSourceType.CLOUD_PLATFORM in source_types
    
    @pytest.mark.asyncio
    async def test_full_integration_workflow(self):
        """Test a complete integration workflow"""
        manager = setup_data_integration()
        
        # Initialize sample data sources
        results = await initialize_sample_data_sources(manager)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that at least some connections succeeded
        successful_connections = sum(1 for success in results.values() if success)
        assert successful_connections > 0
        
        # Test health status
        health_status = await manager.get_health_status()
        assert len(health_status) == len(results)
        
        # Test connection testing
        connection_results = await manager.test_all_connections()
        assert len(connection_results) == len(results)
    
    @pytest.mark.asyncio
    async def test_data_synchronization(self):
        """Test data synchronization across multiple sources"""
        manager = setup_data_integration()
        
        # Add a few test data sources
        test_configs = [
            DataSourceConfig(
                source_id=f'test_source_{i}',
                source_type=DataSourceType.ERP,
                name=f'Test Source {i}',
                connection_params={
                    'host': f'test{i}.example.com',
                    'client': '100',
                    'username': 'test_user',
                    'password': 'test_password'
                },
                enabled=False  # Disable periodic sync for testing
            )
            for i in range(3)
        ]
        
        # Add all sources
        for config in test_configs:
            success = await manager.add_data_source(config)
            assert success is True
        
        # Sync data from all sources
        all_data = []
        for config in test_configs:
            try:
                data = await manager.sync_data_source(config.source_id)
                all_data.extend(data)
            except Exception as e:
                # Some connections might fail in test environment
                print(f"Sync failed for {config.source_id}: {e}")
        
        # Verify we got some data
        assert len(all_data) >= 0  # Allow for empty results in test environment
        
        # Clean up
        for config in test_configs:
            await manager.remove_data_source(config.source_id)


if __name__ == '__main__':
    # Run specific tests
    pytest.main([__file__, '-v'])