"""
Enterprise Connector Integration Tests
Tests all enterprise system connectors with real-world scenarios
"""
import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from scrollintel.connectors.enterprise_data_connectors import EnterpriseDataConnector
from scrollintel.connectors.erp_connectors import SAPConnector, OracleConnector
from scrollintel.connectors.crm_connectors import SalesforceConnector, HubSpotConnector
from scrollintel.connectors.data_lake_connectors import SnowflakeConnector, DatabricksConnector
from scrollintel.core.data_pipeline import DataPipeline
from scrollintel.models.agent_steering_models import ConnectionConfig, DataSource


class TestEnterpriseConnectors:
    """Test suite for enterprise system connectors"""
    
    @pytest.fixture
    def connector_configs(self):
        """Sample connector configurations"""
        return {
            'sap': ConnectionConfig(
                name='sap_test',
                type='sap',
                host='test-sap.company.com',
                port=3300,
                username='test_user',
                password='test_pass',
                database='TEST_DB',
                timeout=30
            ),
            'salesforce': ConnectionConfig(
                name='sf_test',
                type='salesforce',
                instance_url='https://test.salesforce.com',
                username='test@company.com',
                password='test_pass',
                security_token='test_token',
                api_version='58.0'
            ),
            'snowflake': ConnectionConfig(
                name='snowflake_test',
                type='snowflake',
                account='test_account',
                username='test_user',
                password='test_pass',
                warehouse='TEST_WH',
                database='TEST_DB',
                schema='PUBLIC'
            )
        }
    
    @pytest.fixture
    def sample_enterprise_data(self):
        """Generate sample enterprise data for testing"""
        return {
            'customers': pd.DataFrame({
                'customer_id': range(1, 1001),
                'name': [f'Customer_{i}' for i in range(1, 1001)],
                'email': [f'customer{i}@company.com' for i in range(1, 1001)],
                'created_date': pd.date_range('2020-01-01', periods=1000),
                'revenue': np.random.uniform(1000, 100000, 1000),
                'status': np.random.choice(['active', 'inactive', 'prospect'], 1000)
            }),
            'transactions': pd.DataFrame({
                'transaction_id': range(1, 5001),
                'customer_id': np.random.randint(1, 1001, 5000),
                'amount': np.random.uniform(10, 5000, 5000),
                'transaction_date': pd.date_range('2023-01-01', periods=5000, freq='H'),
                'product_category': np.random.choice(['A', 'B', 'C', 'D'], 5000),
                'payment_method': np.random.choice(['credit', 'debit', 'cash'], 5000)
            }),
            'inventory': pd.DataFrame({
                'product_id': range(1, 501),
                'product_name': [f'Product_{i}' for i in range(1, 501)],
                'category': np.random.choice(['Electronics', 'Clothing', 'Books'], 500),
                'stock_level': np.random.randint(0, 1000, 500),
                'unit_price': np.random.uniform(5, 500, 500),
                'last_updated': pd.date_range('2024-01-01', periods=500)
            })
        }
    
    @pytest.mark.asyncio
    async def test_sap_connector_integration(self, connector_configs, sample_enterprise_data):
        """Test SAP connector with real-world scenarios"""
        config = connector_configs['sap']
        
        with patch('scrollintel.connectors.erp_connectors.pyrfc') as mock_pyrfc:
            # Mock SAP connection
            mock_connection = Mock()
            mock_pyrfc.Connection.return_value = mock_connection
            
            # Mock SAP table data
            mock_connection.call.return_value = {
                'ET_DATA': [
                    {'CUSTOMER_ID': '001', 'NAME': 'Test Customer', 'REVENUE': 50000},
                    {'CUSTOMER_ID': '002', 'NAME': 'Another Customer', 'REVENUE': 75000}
                ]
            }
            
            connector = SAPConnector(config)
            
            # Test connection establishment
            await connector.connect()
            assert connector.is_connected()
            
            # Test data extraction
            customers = await connector.extract_table_data('CUSTOMERS', limit=1000)
            assert len(customers) > 0
            assert 'CUSTOMER_ID' in customers[0]
            
            # Test real-time data streaming
            stream = await connector.setup_streaming('SALES_ORDERS')
            assert stream is not None
            
            # Test connection cleanup
            await connector.disconnect()
            assert not connector.is_connected()
    
    @pytest.mark.asyncio
    async def test_salesforce_connector_integration(self, connector_configs, sample_enterprise_data):
        """Test Salesforce connector with CRM scenarios"""
        config = connector_configs['salesforce']
        
        with patch('scrollintel.connectors.crm_connectors.simple_salesforce') as mock_sf:
            # Mock Salesforce connection
            mock_sf_instance = Mock()
            mock_sf.Salesforce.return_value = mock_sf_instance
            
            # Mock SOQL query results
            mock_sf_instance.query.return_value = {
                'records': [
                    {'Id': '001', 'Name': 'Test Account', 'AnnualRevenue': 1000000},
                    {'Id': '002', 'Name': 'Another Account', 'AnnualRevenue': 2000000}
                ],
                'totalSize': 2
            }
            
            connector = SalesforceConnector(config)
            
            # Test connection
            await connector.connect()
            assert connector.is_connected()
            
            # Test SOQL queries
            accounts = await connector.query_records('Account', ['Id', 'Name', 'AnnualRevenue'])
            assert len(accounts) == 2
            assert accounts[0]['Name'] == 'Test Account'
            
            # Test bulk data extraction
            opportunities = await connector.bulk_extract('Opportunity', 
                                                       fields=['Id', 'Name', 'Amount', 'StageName'])
            assert opportunities is not None
            
            # Test real-time change tracking
            changes = await connector.get_recent_changes('Account', hours=24)
            assert isinstance(changes, list)
            
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_snowflake_connector_integration(self, connector_configs, sample_enterprise_data):
        """Test Snowflake data lake connector"""
        config = connector_configs['snowflake']
        
        with patch('scrollintel.connectors.data_lake_connectors.snowflake.connector') as mock_sf:
            # Mock Snowflake connection
            mock_connection = Mock()
            mock_sf.connect.return_value = mock_connection
            
            mock_cursor = Mock()
            mock_connection.cursor.return_value = mock_cursor
            
            # Mock query results
            mock_cursor.fetchall.return_value = [
                ('001', 'Customer A', 50000.0),
                ('002', 'Customer B', 75000.0)
            ]
            mock_cursor.description = [
                ('CUSTOMER_ID', 'VARCHAR'), 
                ('NAME', 'VARCHAR'), 
                ('REVENUE', 'NUMBER')
            ]
            
            connector = SnowflakeConnector(config)
            
            # Test connection
            await connector.connect()
            assert connector.is_connected()
            
            # Test SQL queries
            result = await connector.execute_query(
                "SELECT customer_id, name, revenue FROM customers LIMIT 1000"
            )
            assert len(result) == 2
            
            # Test data loading
            success = await connector.load_dataframe(
                sample_enterprise_data['customers'], 
                'TEST_CUSTOMERS'
            )
            assert success
            
            # Test streaming data
            stream = await connector.create_stream('SALES_STREAM', 'SALES_TABLE')
            assert stream is not None
            
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_multi_connector_data_pipeline(self, connector_configs, sample_enterprise_data):
        """Test coordinated data pipeline across multiple enterprise systems"""
        
        # Mock all connectors
        with patch.multiple(
            'scrollintel.connectors',
            SAPConnector=Mock(),
            SalesforceConnector=Mock(),
            SnowflakeConnector=Mock()
        ) as mocks:
            
            # Configure mock connectors
            for connector_mock in mocks.values():
                connector_instance = Mock()
                connector_instance.connect = AsyncMock()
                connector_instance.is_connected.return_value = True
                connector_instance.extract_data = AsyncMock(return_value=sample_enterprise_data['customers'])
                connector_mock.return_value = connector_instance
            
            pipeline = DataPipeline()
            
            # Add data sources
            sources = [
                DataSource(name='sap_customers', connector_type='sap', config=connector_configs['sap']),
                DataSource(name='sf_accounts', connector_type='salesforce', config=connector_configs['salesforce']),
                DataSource(name='snowflake_analytics', connector_type='snowflake', config=connector_configs['snowflake'])
            ]
            
            for source in sources:
                await pipeline.add_source(source)
            
            # Test pipeline execution
            results = await pipeline.execute_extraction()
            assert len(results) == 3
            
            # Test data transformation
            transformed_data = await pipeline.transform_data(results)
            assert transformed_data is not None
            
            # Test data validation
            validation_results = await pipeline.validate_data(transformed_data)
            assert validation_results['is_valid']
            
            # Test pipeline monitoring
            metrics = await pipeline.get_performance_metrics()
            assert 'execution_time' in metrics
            assert 'data_quality_score' in metrics
    
    @pytest.mark.asyncio
    async def test_connector_error_handling(self, connector_configs):
        """Test error handling and recovery in connectors"""
        
        # Test connection failures
        with patch('scrollintel.connectors.erp_connectors.pyrfc.Connection') as mock_conn:
            mock_conn.side_effect = Exception("Connection failed")
            
            connector = SAPConnector(connector_configs['sap'])
            
            with pytest.raises(Exception):
                await connector.connect()
            
            # Test retry mechanism
            connector.max_retries = 3
            retry_count = 0
            
            async def mock_connect_with_retry():
                nonlocal retry_count
                retry_count += 1
                if retry_count < 3:
                    raise Exception("Connection failed")
                return True
            
            connector.connect = mock_connect_with_retry
            
            result = await connector.connect_with_retry()
            assert result is True
            assert retry_count == 3
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, sample_enterprise_data):
        """Test data quality validation across enterprise connectors"""
        
        connector = EnterpriseDataConnector()
        
        # Test data completeness
        completeness_score = await connector.validate_completeness(
            sample_enterprise_data['customers']
        )
        assert 0 <= completeness_score <= 1
        
        # Test data consistency
        consistency_results = await connector.validate_consistency([
            sample_enterprise_data['customers'],
            sample_enterprise_data['transactions']
        ])
        assert 'customer_id_consistency' in consistency_results
        
        # Test data freshness
        freshness_score = await connector.validate_freshness(
            sample_enterprise_data['inventory'],
            timestamp_column='last_updated',
            max_age_hours=24
        )
        assert 0 <= freshness_score <= 1
        
        # Test data accuracy
        accuracy_results = await connector.validate_accuracy(
            sample_enterprise_data['transactions'],
            business_rules={
                'amount': {'min': 0, 'max': 1000000},
                'customer_id': {'exists_in': 'customers'}
            }
        )
        assert 'accuracy_score' in accuracy_results
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, connector_configs, sample_enterprise_data):
        """Test connector performance under various loads"""
        
        # Test single connector performance
        start_time = time.time()
        
        with patch('scrollintel.connectors.data_lake_connectors.snowflake.connector') as mock_sf:
            mock_connection = Mock()
            mock_sf.connect.return_value = mock_connection
            mock_cursor = Mock()
            mock_connection.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [(i, f'Record_{i}') for i in range(10000)]
            
            connector = SnowflakeConnector(connector_configs['snowflake'])
            await connector.connect()
            
            # Test large data extraction
            large_dataset = await connector.execute_query(
                "SELECT * FROM large_table LIMIT 10000"
            )
            
            extraction_time = time.time() - start_time
            assert extraction_time < 30  # Should complete within 30 seconds
            assert len(large_dataset) == 10000
        
        # Test concurrent connector operations
        async def concurrent_extraction(connector_config, query):
            with patch('scrollintel.connectors.data_lake_connectors.snowflake.connector') as mock_sf:
                mock_connection = Mock()
                mock_sf.connect.return_value = mock_connection
                mock_cursor = Mock()
                mock_connection.cursor.return_value = mock_cursor
                mock_cursor.fetchall.return_value = [(1, 'test')]
                
                connector = SnowflakeConnector(connector_config)
                await connector.connect()
                return await connector.execute_query(query)
        
        # Run 10 concurrent extractions
        start_time = time.time()
        tasks = [
            concurrent_extraction(connector_configs['snowflake'], f"SELECT * FROM table_{i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        assert len(results) == 10
        assert concurrent_time < 60  # Should complete within 60 seconds
    
    @pytest.mark.asyncio
    async def test_security_compliance(self, connector_configs):
        """Test security and compliance features of connectors"""
        
        # Test encryption in transit
        connector = EnterpriseDataConnector()
        
        # Test SSL/TLS configuration
        ssl_config = await connector.validate_ssl_config(connector_configs['snowflake'])
        assert ssl_config['ssl_enabled']
        assert ssl_config['certificate_valid']
        
        # Test authentication mechanisms
        auth_result = await connector.test_authentication(connector_configs['salesforce'])
        assert auth_result['authenticated']
        assert auth_result['token_valid']
        
        # Test data masking
        sensitive_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'ssn': ['123-45-6789', '987-65-4321', '555-12-3456'],
            'email': ['test@example.com', 'user@company.com', 'admin@test.org']
        })
        
        masked_data = await connector.apply_data_masking(
            sensitive_data,
            masking_rules={
                'ssn': 'full_mask',
                'email': 'partial_mask'
            }
        )
        
        assert masked_data['ssn'].iloc[0] == '***-**-****'
        assert '@' in masked_data['email'].iloc[0]  # Partial masking preserves domain
        
        # Test audit logging
        audit_logs = await connector.get_audit_logs(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert isinstance(audit_logs, list)
        if audit_logs:
            assert 'timestamp' in audit_logs[0]
            assert 'action' in audit_logs[0]
            assert 'user' in audit_logs[0]


class TestConnectorResilience:
    """Test connector resilience and fault tolerance"""
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, connector_configs):
        """Test connector behavior during network failures"""
        
        with patch('scrollintel.connectors.erp_connectors.pyrfc.Connection') as mock_conn:
            # Simulate intermittent network failures
            call_count = 0
            
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Network timeout")
                return Mock()
            
            mock_conn.side_effect = side_effect
            
            connector = SAPConnector(connector_configs['sap'])
            connector.retry_attempts = 3
            connector.retry_delay = 0.1  # Fast retry for testing
            
            # Should succeed after retries
            await connector.connect_with_retry()
            assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_data_corruption_detection(self, sample_enterprise_data):
        """Test detection and handling of corrupted data"""
        
        connector = EnterpriseDataConnector()
        
        # Introduce data corruption
        corrupted_data = sample_enterprise_data['customers'].copy()
        corrupted_data.loc[10:20, 'revenue'] = -999999  # Invalid negative revenue
        corrupted_data.loc[30:40, 'email'] = 'invalid_email'  # Invalid email format
        
        # Test corruption detection
        corruption_report = await connector.detect_data_corruption(
            corrupted_data,
            validation_rules={
                'revenue': {'min': 0, 'max': 1000000},
                'email': {'format': 'email'}
            }
        )
        
        assert corruption_report['corrupted_records'] > 0
        assert 'revenue' in corruption_report['corrupted_fields']
        assert 'email' in corruption_report['corrupted_fields']
        
        # Test automatic data cleaning
        cleaned_data = await connector.clean_corrupted_data(
            corrupted_data,
            corruption_report
        )
        
        assert len(cleaned_data) < len(corrupted_data)  # Corrupted records removed
        assert all(cleaned_data['revenue'] >= 0)  # No negative revenue
    
    @pytest.mark.asyncio
    async def test_schema_evolution_handling(self, connector_configs):
        """Test handling of schema changes in enterprise systems"""
        
        with patch('scrollintel.connectors.data_lake_connectors.snowflake.connector') as mock_sf:
            mock_connection = Mock()
            mock_sf.connect.return_value = mock_connection
            mock_cursor = Mock()
            mock_connection.cursor.return_value = mock_cursor
            
            connector = SnowflakeConnector(connector_configs['snowflake'])
            await connector.connect()
            
            # Test schema detection
            original_schema = {
                'customer_id': 'VARCHAR',
                'name': 'VARCHAR',
                'revenue': 'NUMBER'
            }
            
            new_schema = {
                'customer_id': 'VARCHAR',
                'name': 'VARCHAR',
                'revenue': 'NUMBER',
                'created_date': 'TIMESTAMP',  # New field
                'status': 'VARCHAR'  # New field
            }
            
            schema_changes = await connector.detect_schema_changes(
                'customers',
                original_schema,
                new_schema
            )
            
            assert len(schema_changes['added_fields']) == 2
            assert 'created_date' in schema_changes['added_fields']
            assert 'status' in schema_changes['added_fields']
            
            # Test automatic schema adaptation
            adapted_query = await connector.adapt_query_to_schema(
                "SELECT customer_id, name, revenue FROM customers",
                schema_changes
            )
            
            assert 'created_date' in adapted_query or 'status' in adapted_query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])