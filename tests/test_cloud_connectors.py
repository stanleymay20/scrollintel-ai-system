"""
Specific tests for cloud platform connectors (AWS, Azure, GCP).
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from scrollintel.core.data_connector import DataSourceConfig, DataSourceType, ConnectionStatus
from scrollintel.connectors.cloud_connectors import AWSConnector, AzureConnector, GCPConnector


class TestAWSConnector:
    """Detailed tests for AWS connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_aws_detailed',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test AWS Detailed',
            connection_params={
                'access_key_id': 'AKIATEST123456789012',
                'secret_access_key': 'test_secret_access_key_1234567890123456789012',
                'region': 'us-east-1',
                'session_token': 'optional_session_token'
            },
            timeout=120,
            retry_attempts=3
        )
    
    @pytest.mark.asyncio
    async def test_aws_cost_explorer_queries(self, valid_config):
        """Test AWS Cost Explorer queries"""
        connector = AWSConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test different cost explorer queries
        test_queries = [
            {
                'service': 'cost-explorer',
                'time_period': {
                    'Start': (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'End': datetime.utcnow().strftime('%Y-%m-%d')
                },
                'granularity': 'DAILY',
                'metrics': ['BlendedCost', 'UsageQuantity']
            },
            {
                'service': 'cost-explorer',
                'time_period': {
                    'Start': (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    'End': datetime.utcnow().strftime('%Y-%m-%d')
                },
                'granularity': 'MONTHLY',
                'metrics': ['BlendedCost']
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            
            # Verify AWS Cost Explorer data structure
            for record in data:
                assert record.source_id == valid_config.source_id
                assert record.record_id.startswith('AWS_COST_')
                
                # Check cost data structure
                cost_data = record.data
                assert 'TimePeriod' in cost_data
                assert 'Service' in cost_data
                assert 'BlendedCost' in cost_data
                
                # Verify cost format
                blended_cost = cost_data['BlendedCost']
                assert 'Amount' in blended_cost
                assert 'Unit' in blended_cost
                assert blended_cost['Unit'] == 'USD'
                
                # Verify metadata
                assert 'service' in record.metadata
                assert record.metadata['service'] == 'cost-explorer'
                assert 'aws_region' in record.metadata
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_aws_cloudwatch_queries(self, valid_config):
        """Test AWS CloudWatch queries"""
        connector = AWSConnector(valid_config)
        await connector.connect_with_retry()
        
        query = {
            'service': 'cloudwatch'
        }
        
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        # Verify CloudWatch data structure
        for record in data:
            assert record.record_id.startswith('AWS_CLOUDWATCH_')
            
            metric_data = record.data
            assert 'MetricName' in metric_data
            assert 'Namespace' in metric_data
            assert 'Dimensions' in metric_data
            assert 'Value' in metric_data
            assert 'Unit' in metric_data
            
            # Verify dimensions structure
            dimensions = metric_data['Dimensions']
            assert isinstance(dimensions, list)
            if dimensions:
                assert 'Name' in dimensions[0]
                assert 'Value' in dimensions[0]
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_aws_credential_validation(self, valid_config):
        """Test AWS credential validation"""
        connector = AWSConnector(valid_config)
        
        # Test required parameters
        required_params = connector.get_required_params()
        expected_params = ['access_key_id', 'secret_access_key']
        
        for param in expected_params:
            assert param in required_params
        
        # Test validation passes
        await connector.validate_connection_params()
        
        # Test with missing secret key
        invalid_config = DataSourceConfig(
            source_id='invalid_aws',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Invalid AWS',
            connection_params={
                'access_key_id': 'AKIATEST123456789012'
                # Missing secret_access_key
            }
        )
        
        invalid_connector = AWSConnector(invalid_config)
        with pytest.raises(ValueError, match="Missing required parameter: secret_access_key"):
            await invalid_connector.validate_connection_params()


class TestAzureConnector:
    """Detailed tests for Azure connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_azure_detailed',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test Azure Detailed',
            connection_params={
                'client_id': 'azure_client_id_test_12345678',
                'client_secret': 'azure_client_secret_test_1234567890',
                'tenant_id': 'azure_tenant_id_test_12345678',
                'subscription_id': 'azure_subscription_id_test_12345678'
            },
            timeout=120,
            retry_attempts=3
        )
    
    @pytest.mark.asyncio
    async def test_azure_cost_management_queries(self, valid_config):
        """Test Azure Cost Management queries"""
        connector = AzureConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test cost management queries
        test_queries = [
            {
                'service': 'cost-management',
                'time_period': {
                    'from': (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'to': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                },
                'granularity': 'Daily'
            },
            {
                'service': 'cost-management',
                'time_period': {
                    'from': (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'to': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                },
                'granularity': 'Monthly'
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            
            # Verify Azure Cost Management data structure
            for record in data:
                assert record.record_id.startswith('AZURE_COST_')
                
                cost_data = record.data
                assert 'id' in cost_data
                assert 'name' in cost_data
                assert 'type' in cost_data
                assert 'properties' in cost_data
                
                # Verify properties structure
                properties = cost_data['properties']
                assert 'columns' in properties
                assert 'rows' in properties
                
                # Verify columns structure
                columns = properties['columns']
                assert isinstance(columns, list)
                if columns:
                    assert 'name' in columns[0]
                    assert 'type' in columns[0]
                
                # Verify metadata
                assert 'service' in record.metadata
                assert record.metadata['service'] == 'cost-management'
                assert 'subscription_id' in record.metadata
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_azure_monitor_queries(self, valid_config):
        """Test Azure Monitor queries"""
        connector = AzureConnector(valid_config)
        await connector.connect_with_retry()
        
        query = {
            'service': 'monitor'
        }
        
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        # Verify Azure Monitor data structure
        for record in data:
            assert record.record_id.startswith('AZURE_MONITOR_')
            
            metric_data = record.data
            assert 'id' in metric_data
            assert 'type' in metric_data
            assert 'name' in metric_data
            assert 'unit' in metric_data
            assert 'timeseries' in metric_data
            
            # Verify name structure
            name = metric_data['name']
            assert 'value' in name
            assert 'localizedValue' in name
            
            # Verify timeseries structure
            timeseries = metric_data['timeseries']
            assert isinstance(timeseries, list)
            if timeseries:
                assert 'data' in timeseries[0]
                data_points = timeseries[0]['data']
                if data_points:
                    assert 'timeStamp' in data_points[0]
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_azure_oauth_validation(self, valid_config):
        """Test Azure OAuth parameter validation"""
        connector = AzureConnector(valid_config)
        
        # Test required OAuth parameters
        required_params = connector.get_required_params()
        expected_oauth_params = ['client_id', 'client_secret', 'tenant_id', 'subscription_id']
        
        for param in expected_oauth_params:
            assert param in required_params
        
        # Test validation passes
        await connector.validate_connection_params()


class TestGCPConnector:
    """Detailed tests for GCP connector"""
    
    @pytest.fixture
    def valid_config(self):
        return DataSourceConfig(
            source_id='test_gcp_detailed',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Test GCP Detailed',
            connection_params={
                'service_account_key': '{"type": "service_account", "project_id": "test-project"}',
                'project_id': 'test-project-12345',
                'billing_account_id': 'billing-account-12345'
            },
            timeout=120,
            retry_attempts=3
        )
    
    @pytest.mark.asyncio
    async def test_gcp_billing_queries(self, valid_config):
        """Test GCP Cloud Billing queries"""
        connector = GCPConnector(valid_config)
        await connector.connect_with_retry()
        
        # Test billing queries
        test_queries = [
            {
                'service': 'billing',
                'time_range': {
                    'start_time': (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'end_time': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
                }
            },
            {
                'service': 'billing',
                'time_range': {
                    'start_time': (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'end_time': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
                }
            }
        ]
        
        for query in test_queries:
            data = await connector.fetch_data_with_retry(query)
            assert len(data) > 0
            
            # Verify GCP Billing data structure
            for record in data:
                assert record.record_id.startswith('GCP_BILLING_')
                
                billing_data = record.data
                assert 'account_id' in billing_data
                assert 'project' in billing_data
                assert 'service' in billing_data
                assert 'sku' in billing_data
                assert 'cost' in billing_data
                assert 'currency' in billing_data
                assert 'usage' in billing_data
                
                # Verify project structure
                project = billing_data['project']
                assert 'id' in project
                assert 'name' in project
                
                # Verify service structure
                service = billing_data['service']
                assert 'id' in service
                assert 'description' in service
                
                # Verify usage structure
                usage = billing_data['usage']
                assert 'amount' in usage
                assert 'unit' in usage
                
                # Verify metadata
                assert 'service' in record.metadata
                assert record.metadata['service'] == 'billing'
                assert 'project_id' in record.metadata
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_gcp_monitoring_queries(self, valid_config):
        """Test GCP Cloud Monitoring queries"""
        connector = GCPConnector(valid_config)
        await connector.connect_with_retry()
        
        query = {
            'service': 'monitoring'
        }
        
        data = await connector.fetch_data_with_retry(query)
        assert len(data) > 0
        
        # Verify GCP Monitoring data structure
        for record in data:
            assert record.record_id.startswith('GCP_MONITORING_')
            
            monitoring_data = record.data
            assert 'metric' in monitoring_data
            assert 'resource' in monitoring_data
            assert 'metricKind' in monitoring_data
            assert 'valueType' in monitoring_data
            assert 'points' in monitoring_data
            
            # Verify metric structure
            metric = monitoring_data['metric']
            assert 'type' in metric
            assert 'labels' in metric
            
            # Verify resource structure
            resource = monitoring_data['resource']
            assert 'type' in resource
            assert 'labels' in resource
            
            # Verify points structure
            points = monitoring_data['points']
            assert isinstance(points, list)
            if points:
                assert 'interval' in points[0]
                assert 'value' in points[0]
        
        await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_gcp_service_account_validation(self, valid_config):
        """Test GCP service account validation"""
        connector = GCPConnector(valid_config)
        
        # Test required parameters
        required_params = connector.get_required_params()
        expected_params = ['service_account_key', 'project_id']
        
        for param in expected_params:
            assert param in required_params
        
        # Test validation passes
        await connector.validate_connection_params()


class TestCloudConnectorIntegration:
    """Integration tests across all cloud connectors"""
    
    @pytest.mark.asyncio
    async def test_cloud_connector_consistency(self):
        """Test consistency across cloud connectors"""
        # Create configs for all cloud connectors
        aws_config = DataSourceConfig(
            source_id='consistency_aws',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Consistency AWS',
            connection_params={
                'access_key_id': 'AKIATEST123456789012',
                'secret_access_key': 'test_secret_key',
                'region': 'us-east-1'
            }
        )
        
        azure_config = DataSourceConfig(
            source_id='consistency_azure',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Consistency Azure',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id',
                'subscription_id': 'test_subscription_id'
            }
        )
        
        gcp_config = DataSourceConfig(
            source_id='consistency_gcp',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Consistency GCP',
            connection_params={
                'service_account_key': '{"type": "service_account"}',
                'project_id': 'test-project'
            }
        )
        
        connectors = [
            AWSConnector(aws_config),
            AzureConnector(azure_config),
            GCPConnector(gcp_config)
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
            assert 'services' in schema
            
            # Disconnect
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_cloud_cost_data_consistency(self):
        """Test cost data consistency across cloud providers"""
        # Test AWS cost data
        aws_config = DataSourceConfig(
            source_id='cost_test_aws',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Cost Test AWS',
            connection_params={
                'access_key_id': 'AKIATEST123456789012',
                'secret_access_key': 'test_secret_key',
                'region': 'us-east-1'
            }
        )
        
        aws_connector = AWSConnector(aws_config)
        await aws_connector.connect_with_retry()
        
        aws_query = {
            'service': 'cost-explorer',
            'granularity': 'DAILY',
            'metrics': ['BlendedCost']
        }
        
        aws_data = await aws_connector.fetch_data_with_retry(aws_query)
        
        # Verify AWS cost data has required fields
        for record in aws_data:
            cost_data = record.data
            assert 'BlendedCost' in cost_data
            assert 'Amount' in cost_data['BlendedCost']
            assert 'Unit' in cost_data['BlendedCost']
        
        await aws_connector.disconnect()
        
        # Test Azure cost data
        azure_config = DataSourceConfig(
            source_id='cost_test_azure',
            source_type=DataSourceType.CLOUD_PLATFORM,
            name='Cost Test Azure',
            connection_params={
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id',
                'subscription_id': 'test_subscription_id'
            }
        )
        
        azure_connector = AzureConnector(azure_config)
        await azure_connector.connect_with_retry()
        
        azure_query = {
            'service': 'cost-management',
            'granularity': 'Daily'
        }
        
        azure_data = await azure_connector.fetch_data_with_retry(azure_query)
        
        # Verify Azure cost data has required structure
        for record in azure_data:
            cost_data = record.data
            assert 'properties' in cost_data
            assert 'columns' in cost_data['properties']
            assert 'rows' in cost_data['properties']
        
        await azure_connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_cloud_monitoring_data_consistency(self):
        """Test monitoring data consistency across cloud providers"""
        # Test different cloud monitoring services
        configs = [
            ('aws', AWSConnector, {
                'access_key_id': 'AKIATEST123456789012',
                'secret_access_key': 'test_secret_key',
                'region': 'us-east-1'
            }),
            ('azure', AzureConnector, {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'tenant_id': 'test_tenant_id',
                'subscription_id': 'test_subscription_id'
            }),
            ('gcp', GCPConnector, {
                'service_account_key': '{"type": "service_account"}',
                'project_id': 'test-project'
            })
        ]
        
        for provider, connector_class, params in configs:
            config = DataSourceConfig(
                source_id=f'monitor_test_{provider}',
                source_type=DataSourceType.CLOUD_PLATFORM,
                name=f'Monitor Test {provider.upper()}',
                connection_params=params
            )
            
            connector = connector_class(config)
            await connector.connect_with_retry()
            
            # Query monitoring data
            if provider == 'aws':
                query = {'service': 'cloudwatch'}
            elif provider == 'azure':
                query = {'service': 'monitor'}
            else:  # gcp
                query = {'service': 'monitoring'}
            
            data = await connector.fetch_data_with_retry(query)
            
            # Verify all providers return monitoring data
            assert len(data) > 0
            
            for record in data:
                # All should have timestamp and metadata
                assert record.timestamp is not None
                assert 'service' in record.metadata
                
                # All should have some form of metric data
                assert isinstance(record.data, dict)
                assert len(record.data) > 0
            
            await connector.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])