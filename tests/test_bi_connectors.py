"""
Specific tests for BI tool connectors (Tableau, Power BI, Looker, Qlik).
"""

import pytest
import asyncio
from datetime import datetime

from scrollintel.core.data_connector import DataSourceConfig, DataSourceType, ConnectionStatus
from scrollintel.connectors.bi_connectors import TableauConnector, PowerBIConnector, LookerConnector, QlikConnector


class TestTableauConnector:
    """Detailed tests for Tableau connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_tableau_detailed',
            source_type=DataSourceType.BI_TOOL,
            name='Test Tableau Detailed',
            connection_params={
                'server_url': 'https://tableau-test.company.com',
                'username': 'tableau_test_user',
                'password': 'tableau_test_password',
                'site_id': 'test_site'
            },
            timeout=60,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_tableau_resource_queries(self, valid_config):
        """Test Tableau resource queries"""
        connector = TableauConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different resource types
        test_queries = [
            {
                'resource_type': 'workbooks',
                'filters': {'projectName': 'Analytics'},
                'limit': 5
            },
            {
                'resource_type': 'datasources',
                'filters': {'type': 'sqlserver'},
                'limit': 3
            },
            {
                'resource_type': 'views',
                'limit': 8
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify Tableau-specific data structure
            for record in data:
                assert record.source_id == valid_config.source_id
                assert record.record_id.startswith('TABLEAU_')
                assert query['resource_type'].upper() in record.record_id
                
                # Check Tableau ID format
                assert 'id' in record.data
                tableau_id = record.data['id']
                assert tableau_id.startswith('tableau-')
                
                # Verify metadata
                assert 'resource_type' in record.metadata
                assert record.metadata['resource_type'] == query['resource_type']
                assert 'tableau_server' in record.metadata
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_tableau_workbook_details(self, valid_config):
        """Test Tableau workbook-specific data"""
        connector = TableauConnector(valid_config)
        await connector.connect_with_retry()
        
        query = {
            'resource_type': 'workbooks',
            'limit': 5
        }
        
        data = await connector.fetch_data_with_retry(query)
        
        for record in data:
            workbook_data = record.data
            
            # Check workbook-specific fields
            required_fields = ['id', 'name', 'projectId', 'size', 'viewCount']
            for field in required_fields:
                assert field in workbook_data
            
            # Verify data types
            assert isinstance(workbook_data['size'], int)
            assert isinstance(workbook_data['viewCount'], int)
            assert isinstance(workbook_data['tags'], list)
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_tableau_schema_structure(self, valid_config):
        """Test Tableau schema structure"""
        connector = TableauConnector(valid_config)
        await connector.connect_with_retry()
        
        schema = await connector.get_schema()
        
        # Verify schema structure
        assert 'resources' in schema
        
        # Check specific resources
        expected_resources = ['workbooks', 'datasources', 'views']
        for resource in expected_resources:
            assert resource in schema['resources']
            resource_schema = schema['resources'][resource]
            assert 'description' in resource_schema
            assert 'fields' in resource_schema
        
        await connector.disconnect()


class TestPowerBIConnector:
    """Detailed tests for Power BI connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_powerbi_detailed',
            source_type=DataSourceType.BI_TOOL,
            name='Test Power BI Detailed',
            connection_params={
                'client_id': 'powerbi_client_id_test',
                'client_secret': 'powerbi_client_secret_test',
                'tenant_id': 'powerbi_tenant_id_test'
            },
            timeout=60,
            retry_attempts=3
        )
    
    @pytest.mark.asyncio
    async def test_powerbi_resource_queries(self, valid_config):
        """Test Power BI resource queries"""
        connector = PowerBIConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different resource types
        test_queries = [
            {
                'resource_type': 'reports',
                'workspace_id': 'test-workspace-id',
                'limit': 5
            },
            {
                'resource_type': 'datasets',
                'limit': 3
            },
            {
                'resource_type': 'dashboards',
                'limit': 8
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify Power BI-specific data structure
            for record in data:
                assert record.record_id.startswith('PBI_')
                assert query['resource_type'].upper() in record.record_id
                
                # Check Power BI GUID format
                assert 'id' in record.data
                pbi_id = record.data['id']
                # Power BI uses GUID format
                guid_parts = pbi_id.split('-')
                assert len(guid_parts) == 5
                
                # Verify metadata
                assert 'resource_type' in record.metadata
                assert record.metadata['resource_type'] == query['resource_type']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_powerbi_oauth_authentication(self, valid_config):
        """Test Power BI OAuth authentication"""
        connector = PowerBIConnector(valid_config)
        
        # Test required OAuth parameters
        required_params = connector.get_required_params()
        expected_oauth_params = ['client_id', 'client_secret', 'tenant_id']
        
        for param in expected_oauth_params:
            assert param in required_params
        
        # Test validation passes
        await connector.validate_connection_params()
        
        # Test connection
        result = await connector.connect_with_retry()
        assert result is True
        
        await connector.disconnect()


class TestLookerConnector:
    """Detailed tests for Looker connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_looker_detailed',
            source_type=DataSourceType.BI_TOOL,
            name='Test Looker Detailed',
            connection_params={
                'base_url': 'https://looker-test.company.com',
                'client_id': 'looker_client_id_test',
                'client_secret': 'looker_client_secret_test'
            },
            timeout=60,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_looker_resource_queries(self, valid_config):
        """Test Looker resource queries"""
        connector = LookerConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different resource types
        test_queries = [
            {
                'resource_type': 'looks',
                'fields': ['id', 'title', 'public'],
                'limit': 5
            },
            {
                'resource_type': 'dashboards',
                'fields': ['id', 'title', 'space_id'],
                'limit': 3
            },
            {
                'resource_type': 'queries',
                'limit': 8
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify Looker-specific data structure
            for record in data:
                assert record.record_id.startswith('LOOKER_')
                assert query['resource_type'].upper() in record.record_id
                
                # Check Looker data structure
                assert 'id' in record.data
                
                if query['resource_type'] == 'looks':
                    # Looks have integer IDs
                    assert isinstance(record.data['id'], int)
                    assert 'title' in record.data
                    assert 'public' in record.data
                elif query['resource_type'] == 'dashboards':
                    # Dashboards have string IDs
                    assert isinstance(record.data['id'], str)
                    assert record.data['id'].startswith('dashboard-')
                elif query['resource_type'] == 'queries':
                    # Queries have specific structure
                    assert 'model' in record.data
                    assert 'explore' in record.data
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_looker_query_structure(self, valid_config):
        """Test Looker query-specific data structure"""
        connector = LookerConnector(valid_config)
        await connector.connect_with_retry()
        
        query = {
            'resource_type': 'queries',
            'limit': 3
        }
        
        data = await connector.fetch_data_with_retry(query)
        
        for record in data:
            query_data = record.data
            
            # Check query-specific fields
            required_fields = ['id', 'model', 'explore', 'dimensions', 'measures']
            for field in required_fields:
                assert field in query_data
            
            # Verify data types
            assert isinstance(query_data['dimensions'], list)
            assert isinstance(query_data['measures'], list)
            assert isinstance(query_data['filters'], dict)
        
        await connector.disconnect()


class TestQlikConnector:
    """Detailed tests for Qlik Sense connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_qlik_detailed',
            source_type=DataSourceType.BI_TOOL,
            name='Test Qlik Detailed',
            connection_params={
                'server_url': 'https://qlik-test.company.com',
                'username': 'qlik_test_user',
                'password': 'qlik_test_password',
                'virtual_proxy': 'test_proxy'
            },
            timeout=90,
            retry_attempts=2
        )
    
    @pytest.mark.asyncio
    async def test_qlik_resource_queries(self, valid_config):
        """Test Qlik Sense resource queries"""
        connector = QlikConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different resource types
        test_queries = [
            {
                'resource_type': 'apps',
                'filters': {'published': True},
                'limit': 5
            },
            {
                'resource_type': 'sheets',
                'limit': 3
            },
            {
                'resource_type': 'dataconnections',
                'limit': 8
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            assert len(data) <= query['limit']
            
            # Verify Qlik-specific data structure
            for record in data:
                assert record.record_id.startswith('QLIK_')
                assert query['resource_type'].upper() in record.record_id
                
                # Check Qlik GUID format
                assert 'id' in record.data
                qlik_id = record.data['id']
                # Qlik uses GUID format
                guid_parts = qlik_id.split('-')
                assert len(guid_parts) == 5
                
                # Verify metadata
                assert 'resource_type' in record.metadata
                assert record.metadata['resource_type'] == query['resource_type']
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_qlik_app_structure(self, valid_config):
        """Test Qlik app-specific data structure"""
        connector = QlikConnector(valid_config)
        await connector.connect_with_retry()
        
        query = {
            'resource_type': 'apps',
            'limit': 3
        }
        
        data = await connector.fetch_data_with_retry(query)
        
        for record in data:
            app_data = record.data
            
            # Check app-specific fields
            required_fields = ['id', 'name', 'stream', 'owner', 'published']
            for field in required_fields:
                assert field in app_data
            
            # Verify nested objects
            assert isinstance(app_data['stream'], dict)
            assert 'name' in app_data['stream']
            assert isinstance(app_data['owner'], dict)
            assert 'name' in app_data['owner']
            
            # Verify data types
            assert isinstance(app_data['published'], bool)
            assert isinstance(app_data['fileSize'], int)
        
        await connector.disconnect()


class TestBIConnectorIntegration:
    """Integration tests across all BI connectors"""
    
    @pytest.mark.asyncio
    async def test_bi_connector_consistency(self):
        """Test consistency across BI connectors"""
        # Create configs for all BI connectors
        tableau_config = DataSourceConfig(
            source_id='consistency_tableau',
            source_type=DataSourceType.BI_TOOL,
            name='Consistency Tableau',
            connection_params={
                'server_url': 'https://tableau.company.com',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
        
        powerbi_config = DataSourceConfig(
            source_id='consistency_powerbi',
            source_type=DataSourceType.BI_TOOL,
            name='Consistency Power BI',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            }
        )
        
        looker_config = DataSourceConfig(
            source_id='consistency_looker',
            source_type=DataSourceType.BI_TOOL,
            name='Consistency Looker',
            connection_params={
                'base_url': 'https://looker.company.com',
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret'
            }
        )
        
        qlik_config = DataSourceConfig(
            source_id='consistency_qlik',
            source_type=DataSourceType.BI_TOOL,
            name='Consistency Qlik',
            connection_params={
                'server_url': 'https://qlik.company.com',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
        
        connectors = [
            TableauConnector(tableau_config),
            PowerBIConnector(powerbi_config),
            LookerConnector(looker_config),
            QlikConnector(qlik_config)
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
    async def test_bi_metadata_extraction(self):
        """Test metadata extraction across BI tools"""
        config = DataSourceConfig(
            source_id='metadata_test_tableau',
            source_type=DataSourceType.BI_TOOL,
            name='Metadata Test Tableau',
            connection_params={
                'server_url': 'https://tableau.company.com',
                'username': 'test_user',
                'password': 'test_password'
            }
        )
        
        connector = TableauConnector(config)
        await connector.connect_with_retry()
        
        # Test metadata extraction for different resource types
        resource_types = ['workbooks', 'datasources', 'views']
        
        for resource_type in resource_types:
            query = {
                'resource_type': resource_type,
                'limit': 2
            }
            
            data = await connector.fetch_data_with_retry(query)
            
            for record in data:
                # Verify common metadata fields
                assert 'resource_type' in record.metadata
                assert 'tableau_server' in record.metadata
                
                # Verify timestamp is recent
                time_diff = datetime.utcnow() - record.timestamp
                assert time_diff.total_seconds() < 60  # Within last minute
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_bi_performance_monitoring(self):
        """Test performance monitoring for BI connectors"""
        config = DataSourceConfig(
            source_id='perf_test_powerbi',
            source_type=DataSourceType.BI_TOOL,
            name='Performance Test Power BI',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id'
            },
            timeout=120
        )
        
        connector = PowerBIConnector(config)
        await connector.connect_with_retry()
        
        # Test different query sizes and monitor performance
        query_sizes = [1, 5, 10]
        
        for size in query_sizes:
            start_time = datetime.utcnow()
            
            query = {
                'resource_type': 'reports',
                'limit': size
            }
            
            data = await connector.fetch_data_with_retry(query)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Verify results
            assert len(data) <= size
            assert duration < 60  # Should complete within 60 seconds
            
            # Verify health tracking
            health = connector.get_health()
            assert health.records_synced >= len(data)
        
        await connector.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])