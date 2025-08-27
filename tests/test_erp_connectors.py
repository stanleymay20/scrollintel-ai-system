"""
Specific tests for ERP connectors (SAP, Oracle, Microsoft Dynamics).
"""

import pytest
import asyncio
from datetime import datetime

from scrollintel.core.data_connector import DataSourceConfig, DataSourceType, ConnectionStatus
from scrollintel.connectors.erp_connectors import SAPConnector, OracleERPConnector, MicrosoftDynamicsConnector


class TestSAPConnector:
    """Detailed tests for SAP connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_sap_detailed',
            source_type=DataSourceType.ERP,
            name='Test SAP Detailed',
            connection_params={
                'host': 'sap-test.company.com',
                'client': '100',
                'username': 'sap_test_user',
                'password': 'sap_test_password'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_sap_connection_lifecycle(self, valid_config):
        """Test complete SAP connection lifecycle"""
        connector = SAPConnector(valid_config)
        
        # Initial state
        assert connector.status == ConnectionStatus.DISCONNECTED
        assert connector.connection_attempts == 0
        
        # Connect
        result = await connector.connect_with_retry()
        assert result is True
        assert connector.status == ConnectionStatus.CONNECTED
        assert connector.connection_attempts > 0
        
        # Test connection
        test_result = await connector.test_connection()
        assert test_result is True
        
        # Disconnect
        disconnect_result = await connector.disconnect()
        assert disconnect_result is True
        assert connector.status == ConnectionStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_sap_data_fetching_variations(self, valid_config):
        """Test different SAP data fetching scenarios"""
        connector = SAPConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different table queries
        test_queries = [
            {
                'table': 'MARA',
                'fields': ['MATNR', 'MAKTX'],
                'limit': 5
            },
            {
                'table': 'VBAK',
                'fields': ['VBELN', 'KUNNR'],
                'limit': 10
            },
            {
                'table': 'KNA1',  # Customer master
                'limit': 3
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify data structure
            for record in data:
                assert record.source_id == valid_config.source_id
                assert record.record_id.startswith('SAP_')
                assert isinstance(record.data, dict)
                assert isinstance(record.timestamp, datetime)
                assert 'table' in record.metadata
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_sap_schema_retrieval(self, valid_config):
        """Test SAP schema information retrieval"""
        connector = SAPConnector(valid_config)
        await connector.connect_with_retry()
        
        schema = await connector.get_schema()
        
        # Verify schema structure
        assert 'tables' in schema
        assert isinstance(schema['tables'], dict)
        
        # Check specific tables
        expected_tables = ['MARA', 'VBAK']
        for table in expected_tables:
            assert table in schema['tables']
            table_schema = schema['tables'][table]
            assert 'description' in table_schema
            assert 'fields' in table_schema
            assert isinstance(table_schema['fields'], dict)
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_sap_parameter_validation(self):
        """Test SAP parameter validation"""
        # Test missing host
        invalid_config = DataSourceConfig(
            source_id='invalid_sap',
            source_type=DataSourceType.ERP,
            name='Invalid SAP',
            connection_params={
                'client': '100',
                'username': 'user',
                'password': 'pass'
                # Missing 'host'
            }
        )
        
        connector = SAPConnector(invalid_config)
        with pytest.raises(ValueError, match="Missing required parameter: host"):
            await connector.validate_connection_params()
        
        # Test empty password
        invalid_config.connection_params['host'] = 'test-host'
        invalid_config.connection_params['password'] = ''
        
        with pytest.raises(ValueError, match="Empty value for required parameter: password"):
            await connector.validate_connection_params()


class TestOracleERPConnector:
    """Detailed tests for Oracle ERP connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_oracle_detailed',
            source_type=DataSourceType.ERP,
            name='Test Oracle ERP Detailed',
            connection_params={
                'base_url': 'https://oracle-test.company.com',
                'username': 'oracle_test_user',
                'password': 'oracle_test_password'
            },
            timeout=45,
            retry_attempts=3
        )
    
    @pytest.mark.asyncio
    async def test_oracle_resource_queries(self, valid_config):
        """Test Oracle ERP resource queries"""
        connector = OracleERPConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different resource types
        test_queries = [
            {
                'resource': 'items',
                'filters': {'ItemNumber': 'ITEM-*'},
                'limit': 5
            },
            {
                'resource': 'purchaseOrders',
                'limit': 3
            },
            {
                'resource': 'suppliers',
                'limit': 8
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            
            # Verify Oracle-specific data structure
            for record in data:
                assert record.record_id.startswith('ORACLE_')
                assert 'resource' in record.metadata
                assert record.metadata['resource'] == query['resource']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_oracle_schema_structure(self, valid_config):
        """Test Oracle ERP schema structure"""
        connector = OracleERPConnector(valid_config)
        await connector.connect_with_retry()
        
        schema = await connector.get_schema()
        
        # Verify Oracle schema structure
        assert 'resources' in schema
        
        # Check specific resources
        expected_resources = ['items', 'purchaseOrders']
        for resource in expected_resources:
            assert resource in schema['resources']
            resource_schema = schema['resources'][resource]
            assert 'description' in resource_schema
            assert 'fields' in resource_schema
        
        await connector.disconnect()


class TestMicrosoftDynamicsConnector:
    """Detailed tests for Microsoft Dynamics connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_dynamics_detailed',
            source_type=DataSourceType.ERP,
            name='Test Dynamics Detailed',
            connection_params={
                'org_url': 'https://test-dynamics.crm.dynamics.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            },
            timeout=60,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_dynamics_entity_queries(self, valid_config):
        """Test Dynamics 365 entity queries"""
        connector = MicrosoftDynamicsConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different entity types
        test_queries = [
            {
                'entity': 'products',
                'select': ['productid', 'name', 'price'],
                'limit': 5
            },
            {
                'entity': 'accounts',
                'select': ['accountid', 'name', 'revenue'],
                'filter': "revenue gt 100000",
                'limit': 3
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            
            # Verify Dynamics-specific data structure
            for record in data:
                assert record.record_id.startswith('D365_')
                assert 'entity' in record.metadata
                assert record.metadata['entity'] == query['entity']
                
                # Check GUID format for IDs
                if query['entity'] == 'products':
                    assert 'productid' in record.data
                    # Verify GUID-like format
                    product_id = record.data['productid']
                    assert len(product_id.split('-')) == 5  # GUID format
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_dynamics_oauth_parameters(self, valid_config):
        """Test Dynamics OAuth parameter validation"""
        connector = MicrosoftDynamicsConnector(valid_config)
        
        # Test required OAuth parameters
        required_params = connector.get_required_params()
        expected_oauth_params = ['org_url', 'client_id', 'client_secret', 'tenant_id']
        
        for param in expected_oauth_params:
            assert param in required_params
        
        # Test validation passes with all parameters
        await connector.validate_connection_params()
        
        # Test validation fails with missing tenant_id
        invalid_config = DataSourceConfig(
            source_id='invalid_dynamics',
            source_type=DataSourceType.ERP,
            name='Invalid Dynamics',
            connection_params={
                'org_url': 'https://test.crm.dynamics.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret'
                # Missing 'tenant_id'
            }
        )
        
        invalid_connector = MicrosoftDynamicsConnector(invalid_config)
        with pytest.raises(ValueError, match="Missing required parameter: tenant_id"):
            await invalid_connector.validate_connection_params()


class TestERPConnectorErrorHandling:
    """Test error handling across all ERP connectors"""
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self):
        """Test connection retry logic with simulated failures"""
        config = DataSourceConfig(
            source_id='retry_test_sap',
            source_type=DataSourceType.ERP,
            name='Retry Test SAP',
            connection_params={
                'host': 'unreliable-sap.company.com',
                'client': '100',
                'username': 'test_user',
                'password': 'test_password'
            },
            retry_attempts=3
        )
        
        connector = SAPConnector(config)
        
        # The connector should eventually succeed due to retry logic
        # (our mock implementation has a 10% failure rate)
        result = await connector.connect_with_retry()
        assert result is True
        assert connector.connection_attempts >= 1
    
    @pytest.mark.asyncio
    async def test_data_fetch_timeout_handling(self):
        """Test data fetch timeout handling"""
        config = DataSourceConfig(
            source_id='timeout_test_oracle',
            source_type=DataSourceType.ERP,
            name='Timeout Test Oracle',
            connection_params={
                'base_url': 'https://slow-oracle.company.com',
                'username': 'test_user',
                'password': 'test_password'
            },
            timeout=0.1  # Very short timeout
        )
        
        connector = OracleERPConnector(config)
        
        try:
            await connector.connect_with_retry()
            
            # This should timeout due to short timeout setting
            query = {'resource': 'items', 'limit': 1000}
            await connector.fetch_data_with_retry(query)
            
        except Exception as e:
            # Timeout or other error is expected
            assert connector.error_message is not None
    
    @pytest.mark.asyncio
    async def test_health_status_tracking(self):
        """Test health status tracking across operations"""
        config = DataSourceConfig(
            source_id='health_track_dynamics',
            source_type=DataSourceType.ERP,
            name='Health Track Dynamics',
            connection_params={
                'org_url': 'https://test.crm.dynamics.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            }
        )
        
        connector = MicrosoftDynamicsConnector(config)
        
        # Initial health
        health = connector.get_health()
        assert health.status == ConnectionStatus.DISCONNECTED
        assert health.records_synced == 0
        
        # After connection
        await connector.connect_with_retry()
        health = connector.get_health()
        assert health.status == ConnectionStatus.CONNECTED
        
        # After data fetch
        query = {'entity': 'products', 'limit': 5}
        data = await connector.fetch_data_with_retry(query)
        
        health = connector.get_health()
        assert health.records_synced == len(data)
        assert health.last_successful_sync is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])