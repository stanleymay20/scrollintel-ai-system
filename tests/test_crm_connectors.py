"""
Specific tests for CRM connectors (Salesforce, HubSpot, Microsoft CRM).
"""

import pytest
import asyncio
from datetime import datetime

from scrollintel.core.data_connector import DataSourceConfig, DataSourceType, ConnectionStatus
from scrollintel.connectors.crm_connectors import SalesforceConnector, HubSpotConnector, MicrosoftCRMConnector


class TestSalesforceConnector:
    """Detailed tests for Salesforce connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_salesforce_detailed',
            source_type=DataSourceType.CRM,
            name='Test Salesforce Detailed',
            connection_params={
                'instance_url': 'https://test-company.salesforce.com',
                'client_id': 'salesforce_client_id_test',
                'client_secret': 'salesforce_client_secret_test',
                'username': 'test.user@company.com',
                'password': 'salesforce_password_test',
                'security_token': 'salesforce_security_token_test'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_salesforce_sobject_queries(self, valid_config):
        """Test Salesforce sObject queries"""
        connector = SalesforceConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different sObject types
        test_queries = [
            {
                'sobject': 'Account',
                'fields': ['Id', 'Name', 'Type', 'Industry'],
                'where': "Type = 'Customer'",
                'limit': 5
            },
            {
                'sobject': 'Opportunity',
                'fields': ['Id', 'Name', 'StageName', 'Amount'],
                'where': "StageName = 'Prospecting'",
                'limit': 10
            },
            {
                'sobject': 'Contact',
                'fields': ['Id', 'FirstName', 'LastName', 'Email'],
                'limit': 3
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify Salesforce-specific data structure
            for record in data:
                assert record.source_id == valid_config.source_id
                assert record.record_id.startswith('SF_')
                assert query['sobject'] in record.record_id
                
                # Check Salesforce ID format (18 characters)
                assert 'Id' in record.data
                sf_id = record.data['Id']
                assert len(sf_id) == 18  # Salesforce ID format
                
                # Verify metadata
                assert 'sobject' in record.metadata
                assert record.metadata['sobject'] == query['sobject']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_salesforce_schema_metadata(self, valid_config):
        """Test Salesforce schema metadata retrieval"""
        connector = SalesforceConnector(valid_config)
        await connector.connect_with_retry()
        
        schema = await connector.get_schema()
        
        # Verify schema structure
        assert 'sobjects' in schema
        assert isinstance(schema['sobjects'], dict)
        
        # Check specific sObjects
        expected_sobjects = ['Account', 'Opportunity', 'Contact']
        for sobject in expected_sobjects:
            assert sobject in schema['sobjects']
            sobject_schema = schema['sobjects'][sobject]
            assert 'description' in sobject_schema
            assert 'fields' in sobject_schema
            
            # Check field definitions
            fields = sobject_schema['fields']
            assert 'Id' in fields
            assert fields['Id']['type'] == 'id'
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_salesforce_oauth_validation(self, valid_config):
        """Test Salesforce OAuth parameter validation"""
        connector = SalesforceConnector(valid_config)
        
        # Test required parameters
        required_params = connector.get_required_params()
        expected_params = ['instance_url', 'client_id', 'client_secret', 'username', 'password']
        
        for param in expected_params:
            assert param in required_params
        
        # Test validation passes
        await connector.validate_connection_params()


class TestHubSpotConnector:
    """Detailed tests for HubSpot connector"""
    
    @pytest.fixture
    def access_token_config(self):
        return DataSourceConfig(
            source_id='test_hubspot_token',
            source_type=DataSourceType.CRM,
            name='Test HubSpot Token',
            connection_params={
                'access_token': 'hubspot_access_token_test'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.fixture
    def api_key_config(self):
        return DataSourceConfig(
            source_id='test_hubspot_apikey',
            source_type=DataSourceType.CRM,
            name='Test HubSpot API Key',
            connection_params={
                'api_key': 'hubspot_api_key_test'
            },
            timeout=30,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_hubspot_object_queries(self, access_token_config):
        """Test HubSpot object queries"""
        connector = HubSpotConnector(access_token_config)
        await connector.connect_with_retry()
        
        # Test different object types
        test_queries = [
            {
                'object_type': 'companies',
                'properties': ['name', 'domain', 'industry'],
                'filters': [{'propertyName': 'industry', 'operator': 'EQ', 'value': 'Technology'}],
                'limit': 5
            },
            {
                'object_type': 'deals',
                'properties': ['dealname', 'amount', 'dealstage'],
                'limit': 8
            },
            {
                'object_type': 'contacts',
                'properties': ['firstname', 'lastname', 'email'],
                'limit': 3
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify HubSpot-specific data structure
            for record in data:
                assert record.record_id.startswith('HS_')
                assert query['object_type'].upper() in record.record_id
                
                # Check HubSpot data format
                assert 'id' in record.data
                assert 'properties' in record.data
                
                # Verify metadata
                assert 'object_type' in record.metadata
                assert record.metadata['object_type'] == query['object_type']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_hubspot_authentication_methods(self, access_token_config, api_key_config):
        """Test both HubSpot authentication methods"""
        # Test access token authentication
        token_connector = HubSpotConnector(access_token_config)
        await token_connector.validate_connection_params()
        result = await token_connector.connect_with_retry()
        assert result is True
        await token_connector.disconnect()
        
        # Test API key authentication
        key_connector = HubSpotConnector(api_key_config)
        await key_connector.validate_connection_params()
        result = await key_connector.connect_with_retry()
        assert result is True
        await key_connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_hubspot_custom_validation(self):
        """Test HubSpot custom parameter validation"""
        # Test with neither API key nor access token
        invalid_config = DataSourceConfig(
            source_id='invalid_hubspot',
            source_type=DataSourceType.CRM,
            name='Invalid HubSpot',
            connection_params={}
        )
        
        connector = HubSpotConnector(invalid_config)
        with pytest.raises(ValueError, match="Either api_key or access_token is required"):
            await connector.validate_connection_params()
        
        # Test with both (should still work)
        both_config = DataSourceConfig(
            source_id='both_hubspot',
            source_type=DataSourceType.CRM,
            name='Both Auth HubSpot',
            connection_params={
                'api_key': 'test_api_key',
                'access_token': 'test_access_token'
            }
        )
        
        both_connector = HubSpotConnector(both_config)
        await both_connector.validate_connection_params()  # Should not raise


class TestMicrosoftCRMConnector:
    """Detailed tests for Microsoft CRM connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_mscrm_detailed',
            source_type=DataSourceType.CRM,
            name='Test Microsoft CRM Detailed',
            connection_params={
                'org_url': 'https://test-org.crm.dynamics.com',
                'client_id': 'mscrm_client_id_test',
                'client_secret': 'mscrm_client_secret_test',
                'tenant_id': 'mscrm_tenant_id_test'
            },
            timeout=45,
            retry_attempts=3
        )
    
    @pytest.mark.asyncio
    async def test_mscrm_entity_queries(self, valid_config):
        """Test Microsoft CRM entity queries"""
        connector = MicrosoftCRMConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different entity types
        test_queries = [
            {
                'entity': 'accounts',
                'select': ['accountid', 'name', 'revenue'],
                'filter': 'revenue gt 500000',
                'limit': 5
            },
            {
                'entity': 'opportunities',
                'select': ['opportunityid', 'name', 'estimatedvalue'],
                'limit': 10
            },
            {
                'entity': 'contacts',
                'select': ['contactid', 'firstname', 'lastname', 'emailaddress1'],
                'limit': 3
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify Microsoft CRM-specific data structure
            for record in data:
                assert record.record_id.startswith('MSCRM_')
                assert query['entity'].upper() in record.record_id
                
                # Check GUID format for entity IDs
                entity_id_field = f"{query['entity'][:-1]}id"  # Remove 's' and add 'id'
                if entity_id_field in record.data:
                    entity_id = record.data[entity_id_field]
                    # Verify GUID format (8-4-4-4-12 characters)
                    guid_parts = entity_id.split('-')
                    assert len(guid_parts) == 5
                    assert len(guid_parts[0]) == 8
                    assert len(guid_parts[1]) == 4
                
                # Verify metadata
                assert 'entity' in record.metadata
                assert record.metadata['entity'] == query['entity']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_mscrm_web_api_structure(self, valid_config):
        """Test Microsoft CRM Web API data structure"""
        connector = MicrosoftCRMConnector(valid_config)
        await connector.connect_with_retry()
        
        schema = await connector.get_schema()
        
        # Verify Web API schema structure
        assert 'entities' in schema
        
        # Check specific entities
        expected_entities = ['accounts', 'opportunities', 'contacts']
        for entity in expected_entities:
            assert entity in schema['entities']
            entity_schema = schema['entities'][entity]
            assert 'description' in entity_schema
            assert 'fields' in entity_schema
            
            # Check Dynamics-specific field types
            fields = entity_schema['fields']
            for field_name, field_info in fields.items():
                assert 'type' in field_info
                # Should have Edm types (OData standard)
                assert field_info['type'].startswith('Edm.')
        
        await connector.disconnect()


class TestCRMConnectorIntegration:
    """Integration tests across all CRM connectors"""
    
    @pytest.mark.asyncio
    async def test_crm_data_consistency(self):
        """Test data consistency across CRM connectors"""
        # Create configs for all CRM connectors
        sf_config = DataSourceConfig(
            source_id='consistency_sf',
            source_type=DataSourceType.CRM,
            name='Consistency Salesforce',
            connection_params={
                'instance_url': 'https://test.salesforce.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'username': 'test@company.com',
                'password': 'test_password'
            }
        )
        
        hs_config = DataSourceConfig(
            source_id='consistency_hs',
            source_type=DataSourceType.CRM,
            name='Consistency HubSpot',
            connection_params={
                'access_token': 'test_access_token'
            }
        )
        
        ms_config = DataSourceConfig(
            source_id='consistency_ms',
            source_type=DataSourceType.CRM,
            name='Consistency Microsoft',
            connection_params={
                'org_url': 'https://test.crm.dynamics.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            }
        )
        
        connectors = [
            SalesforceConnector(sf_config),
            HubSpotConnector(hs_config),
            MicrosoftCRMConnector(ms_config)
        ]
        
        # Test that all connectors follow the same interface
        for connector in connectors:
            # Connect
            result = await connector.connect_with_retry()
            assert result is True
            
            # Test connection
            test_result = await connector.test_connection()
            assert test_result is True
            
            # Get health
            health = connector.get_health()
            assert health.status == ConnectionStatus.CONNECTED
            
            # Get schema
            schema = await connector.get_schema()
            assert isinstance(schema, dict)
            
            # Disconnect
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_crm_error_handling_consistency(self):
        """Test consistent error handling across CRM connectors"""
        # Test with invalid configurations
        invalid_configs = [
            # Salesforce with missing parameters
            DataSourceConfig(
                source_id='invalid_sf',
                source_type=DataSourceType.CRM,
                name='Invalid Salesforce',
                connection_params={'instance_url': 'https://test.salesforce.com'}
            ),
            # HubSpot with no authentication
            DataSourceConfig(
                source_id='invalid_hs',
                source_type=DataSourceType.CRM,
                name='Invalid HubSpot',
                connection_params={}
            ),
            # Microsoft CRM with missing tenant
            DataSourceConfig(
                source_id='invalid_ms',
                source_type=DataSourceType.CRM,
                name='Invalid Microsoft',
                connection_params={
                    'org_url': 'https://test.crm.dynamics.com',
                    'client_id': 'test_client_id',
                    'client_secret': 'test_client_secret'
                }
            )
        ]
        
        connector_classes = [SalesforceConnector, HubSpotConnector, MicrosoftCRMConnector]
        
        for config, connector_class in zip(invalid_configs, connector_classes):
            connector = connector_class(config)
            
            # All should fail validation
            with pytest.raises(ValueError):
                await connector.validate_connection_params()
    
    @pytest.mark.asyncio
    async def test_crm_performance_characteristics(self):
        """Test performance characteristics of CRM connectors"""
        config = DataSourceConfig(
            source_id='perf_test_sf',
            source_type=DataSourceType.CRM,
            name='Performance Test Salesforce',
            connection_params={
                'instance_url': 'https://test.salesforce.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'username': 'test@company.com',
                'password': 'test_password'
            },
            timeout=60
        )
        
        connector = SalesforceConnector(config)
        await connector.connect_with_retry()
        
        # Test different query sizes
        query_sizes = [1, 5, 10, 50]
        
        for size in query_sizes:
            start_time = datetime.utcnow()
            
            query = {
                'sobject': 'Account',
                'fields': ['Id', 'Name'],
                'limit': size
            }
            
            data = await connector.fetch_data_with_retry(query)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Verify results
            assert len(data) <= size
            assert duration < 30  # Should complete within 30 seconds
            
            # Verify health tracking
            health = connector.get_health()
            assert health.records_synced >= len(data)
        
        await connector.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])