"""
Comprehensive test suite for Enterprise Data Integration Layer.
Tests connectors, pipelines, validation, and real-time streaming.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from scrollintel.core.data_connector import DataSourceConfig, DataSourceType, DataRecord
from scrollintel.connectors.enterprise_data_connectors import (
    SQLServerConnector, SnowflakeConnector, DatabricksConnector,
    EnterpriseDataValidator, EnterpriseDataEnricher, RealTimeDataStreamer,
    StreamingConfig, StreamingMode, DataValidationLevel, ValidationRule, EnrichmentRule
)
from scrollintel.connectors.data_lake_connectors import BigQueryConnector, RedshiftConnector
from scrollintel.connectors.erp_connectors import SAPConnector, OracleERPConnector
from scrollintel.connectors.crm_connectors import SalesforceConnector, HubSpotConnector
from scrollintel.core.data_pipeline import (
    create_enterprise_pipeline, SchemaValidator, DataCleaner, 
    DataEnricher, QualityAssessment, ProcessingMode
)


class TestEnterpriseDataConnectors:
    """Test enterprise data connectors"""
    
    @pytest.fixture
    def sql_server_config(self):
        return DataSourceConfig(
            source_id="test_sql_server",
            source_type=DataSourceType.DATABASE,
            name="Test SQL Server",
            connection_params={
                "server": "localhost",
                "database": "testdb",
                "username": "testuser",
                "password": "testpass",
                "driver": "ODBC Driver 17 for SQL Server",
                "encrypt": True
            }
        )
    
    @pytest.fixture
    def snowflake_config(self):
        return DataSourceConfig(
            source_id="test_snowflake",
            source_type=DataSourceType.CLOUD_PLATFORM,
            name="Test Snowflake",
            connection_params={
                "account": "test_account",
                "user": "testuser",
                "password": "testpass",
                "warehouse": "COMPUTE_WH",
                "database": "TEST_DB",
                "schema": "PUBLIC"
            }
        )
    
    @pytest.fixture
    def databricks_config(self):
        return DataSourceConfig(
            source_id="test_databricks",
            source_type=DataSourceType.CLOUD_PLATFORM,
            name="Test Databricks",
            connection_params={
                "server_hostname": "test.cloud.databricks.com",
                "http_path": "/sql/1.0/warehouses/test",
                "access_token": "test_token",
                "catalog": "main",
                "schema": "default"
            }
        )
    
    @pytest.mark.asyncio
    async def test_sql_server_connector(self, sql_server_config):
        """Test SQL Server connector functionality"""
        connector = SQLServerConnector(sql_server_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected is True
        assert connector.status.value == "connected"
        
        # Test connection test
        test_result = await connector.test_connection()
        assert test_result is True
        
        # Test data fetch
        query = {
            "table": "dbo.Customers",
            "columns": ["CustomerID", "CompanyName", "ContactName"],
            "limit": 10
        }
        records = await connector.fetch_data(query)
        
        assert len(records) > 0
        assert all(isinstance(record, DataRecord) for record in records)
        assert all(record.source_id == sql_server_config.source_id for record in records)
        
        # Verify data validation and enrichment
        for record in records:
            assert "validation_passed" in record.metadata
            assert "enrichment_applied" in record.metadata
            assert record.metadata["validation_passed"] is True
        
        # Test schema retrieval
        schema = await connector.get_schema()
        assert "database" in schema
        assert "tables" in schema
        assert "dbo.Customers" in schema["tables"]
        
        # Test disconnect
        disconnected = await connector.disconnect()
        assert disconnected is True
    
    @pytest.mark.asyncio
    async def test_snowflake_connector(self, snowflake_config):
        """Test Snowflake connector functionality"""
        connector = SnowflakeConnector(snowflake_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected is True
        
        # Test data fetch with analytics
        query = {
            "sql": "SELECT * FROM TRANSACTIONS LIMIT 100",
            "warehouse": "COMPUTE_WH"
        }
        records = await connector.fetch_data(query)
        
        assert len(records) > 0
        
        # Verify analytical enrichments
        for record in records[:5]:  # Check first 5 records
            data = record.data
            assert "sales_metrics" in data
            assert "customer_analytics" in data
            assert "seasonal_trends" in data
            assert "forecasting_features" in data
            
            # Verify specific metrics
            assert "profit_margin_percent" in data["sales_metrics"]
            assert "customer_value_tier" in data["customer_analytics"]
            assert "seasonal_factor" in data["seasonal_trends"]
        
        # Test schema
        schema = await connector.get_schema()
        assert "account" in schema
        assert "database" in schema
        assert "tables" in schema
    
    @pytest.mark.asyncio
    async def test_databricks_connector(self, databricks_config):
        """Test Databricks connector functionality"""
        connector = DatabricksConnector(databricks_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected is True
        
        # Test data fetch with ML features
        query = {
            "sql": "SELECT * FROM user_events LIMIT 50",
            "catalog": "main",
            "schema": "default"
        }
        records = await connector.fetch_data(query)
        
        assert len(records) > 0
        
        # Verify ML enrichments
        for record in records[:3]:  # Check first 3 records
            data = record.data
            assert "user_lifetime_value" in data
            assert "churn_probability" in data
            assert "next_best_action" in data
            assert "anomaly_score" in data
            assert "engagement_score" in data
            
            # Verify Delta Lake metadata
            assert "_delta_version" in data
            assert "_commit_timestamp" in data
        
        # Test schema with Delta Lake features
        schema = await connector.get_schema()
        assert "catalog" in schema
        assert "tables" in schema
        assert "user_events" in schema["tables"]
        assert schema["tables"]["user_events"]["table_format"] == "DELTA"


class TestDataLakeConnectors:
    """Test data lake connectors"""
    
    @pytest.fixture
    def bigquery_config(self):
        return DataSourceConfig(
            source_id="test_bigquery",
            source_type=DataSourceType.CLOUD_PLATFORM,
            name="Test BigQuery",
            connection_params={
                "project_id": "test-project",
                "dataset_id": "test_dataset",
                "location": "US"
            }
        )
    
    @pytest.fixture
    def redshift_config(self):
        return DataSourceConfig(
            source_id="test_redshift",
            source_type=DataSourceType.CLOUD_PLATFORM,
            name="Test Redshift",
            connection_params={
                "host": "test-cluster.redshift.amazonaws.com",
                "port": 5439,
                "database": "testdb",
                "user": "testuser",
                "password": "testpass",
                "cluster_identifier": "test-cluster"
            }
        )
    
    @pytest.mark.asyncio
    async def test_bigquery_connector(self, bigquery_config):
        """Test BigQuery connector with advanced analytics"""
        connector = BigQueryConnector(bigquery_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected is True
        
        # Test data fetch
        query = {
            "sql": "SELECT * FROM `dataset.events_*` WHERE _TABLE_SUFFIX BETWEEN '20240101' AND '20240131'",
            "limit": 100
        }
        records = await connector.fetch_data(query)
        
        assert len(records) > 0
        
        # Verify BigQuery-specific analytics
        for record in records[:3]:
            data = record.data
            assert "audience_segments" in data
            assert "attribution_model" in data
            assert "funnel_stage" in data
            assert "cohort_analysis" in data
            assert "anomaly_detection" in data
            
            # Verify audience segments
            assert isinstance(data["audience_segments"], list)
            
            # Verify attribution model
            attribution = data["attribution_model"]
            assert "first_touch" in attribution
            assert "conversion_credit" in attribution
            
            # Verify anomaly detection
            anomaly_data = data["anomaly_detection"]
            assert "anomalies_detected" in anomaly_data
            assert "anomaly_score" in anomaly_data
        
        # Test schema with partitioning info
        schema = await connector.get_schema()
        assert "project_id" in schema
        assert "tables" in schema
        events_table = schema["tables"]["events_*"]
        assert "partitioning" in events_table
        assert events_table["partitioning"]["type"] == "TIME"
    
    @pytest.mark.asyncio
    async def test_redshift_connector(self, redshift_config):
        """Test Redshift connector with columnar analytics"""
        connector = RedshiftConnector(redshift_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected is True
        
        # Test data fetch
        query = {
            "sql": "SELECT * FROM sales_fact WHERE transaction_date >= '2024-01-01'",
            "limit": 50
        }
        records = await connector.fetch_data(query)
        
        assert len(records) > 0
        
        # Verify Redshift-specific analytics
        for record in records[:3]:
            data = record.data
            assert "sales_metrics" in data
            assert "customer_analytics" in data
            assert "product_performance" in data
            assert "seasonal_trends" in data
            assert "forecasting_features" in data
            
            # Verify sales metrics
            sales_metrics = data["sales_metrics"]
            assert "gross_profit" in sales_metrics
            assert "profit_margin_percent" in sales_metrics
            
            # Verify customer analytics
            customer_analytics = data["customer_analytics"]
            assert "customer_value_tier" in customer_analytics
            assert "churn_risk" in customer_analytics
            
            # Verify forecasting features
            forecasting = data["forecasting_features"]
            assert "day_of_week" in forecasting
            assert "is_weekend" in forecasting
        
        # Test schema with Redshift-specific features
        schema = await connector.get_schema()
        assert "cluster_identifier" in schema
        sales_fact = schema["schemas"]["public"]["tables"]["sales_fact"]
        assert "distribution_style" in sales_fact
        assert "sort_keys" in sales_fact
        assert "compression" in sales_fact


class TestDataValidationAndEnrichment:
    """Test data validation and enrichment components"""
    
    @pytest.fixture
    def sample_record(self):
        return {
            "id": 12345,
            "name": "  Test Customer  ",
            "email": "test@example.com",
            "phone": "(555) 123-4567",
            "amount": 150.75,
            "created_date": "2024-01-15T10:30:00Z",
            "status": "active"
        }
    
    @pytest.mark.asyncio
    async def test_enterprise_data_validator(self, sample_record):
        """Test enterprise data validation"""
        validator = EnterpriseDataValidator(DataValidationLevel.ENTERPRISE)
        
        # Add validation rules
        validator.add_validation_rule(ValidationRule(
            field_name="id",
            rule_type="required"
        ))
        validator.add_validation_rule(ValidationRule(
            field_name="email",
            rule_type="pattern",
            parameters={"pattern": r"^[^@]+@[^@]+\.[^@]+$"}
        ))
        validator.add_validation_rule(ValidationRule(
            field_name="amount",
            rule_type="range",
            parameters={"min": 0, "max": 10000}
        ))
        
        # Test validation
        result = await validator.validate_record(sample_record)
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert "transformed_record" in result
        
        # Test with invalid record
        invalid_record = sample_record.copy()
        invalid_record["email"] = "invalid-email"
        invalid_record["amount"] = -50
        
        invalid_result = await validator.validate_record(invalid_record)
        assert invalid_result["is_valid"] is False
        assert len(invalid_result["errors"]) > 0
        
        # Check validation stats
        stats = validator.get_validation_stats()
        assert stats["total_records"] == 2
        assert stats["valid_records"] == 1
        assert stats["invalid_records"] == 1
    
    @pytest.mark.asyncio
    async def test_enterprise_data_enricher(self, sample_record):
        """Test enterprise data enrichment"""
        enricher = EnterpriseDataEnricher()
        
        # Add enrichment rules
        enricher.add_enrichment_rule(EnrichmentRule(
            source_field="amount",
            target_field="amount_category",
            enrichment_type="calculation",
            parameters={"formula": "high if {amount} > 100 else low"}
        ))
        enricher.add_enrichment_rule(EnrichmentRule(
            source_field="name",
            target_field="name_length",
            enrichment_type="calculation",
            parameters={"formula": "len({name})"}
        ))
        
        # Test enrichment
        enriched_record = await enricher.enrich_record(sample_record)
        
        assert "processing_timestamp" in enriched_record
        assert "record_hash" in enriched_record
        assert "enrichment_version" in enriched_record
        
        # Verify enrichment was applied
        assert enriched_record != sample_record
    
    @pytest.mark.asyncio
    async def test_real_time_data_streamer(self):
        """Test real-time data streaming"""
        config = StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            batch_size=10,
            flush_interval=1,
            compression=True,
            encryption=True
        )
        
        streamer = RealTimeDataStreamer(config)
        
        # Create test records
        records = []
        for i in range(25):
            record = DataRecord(
                source_id="test_source",
                record_id=f"record_{i}",
                data={"id": i, "value": f"test_value_{i}"},
                timestamp=datetime.utcnow(),
                metadata={"test": True}
            )
            records.append(record)
        
        # Add records to streamer
        for record in records:
            await streamer.add_record(record)
        
        # Verify streaming stats
        stats = streamer.streaming_stats
        assert stats["records_streamed"] > 0
        assert stats["flush_count"] > 0


class TestDataPipeline:
    """Test data processing pipeline"""
    
    @pytest.fixture
    def sample_schema(self):
        return {
            "type": "object",
            "required": ["id", "name", "email"],
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
                "email": {"type": "string", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
                "amount": {"type": "number", "minimum": 0, "maximum": 10000},
                "created_date": {"type": "string"}
            }
        }
    
    @pytest.fixture
    def pipeline_config(self):
        return {
            "processing_mode": ProcessingMode.STREAMING.value,
            "batch_size": 100,
            "validation": {
                "strict_mode": True
            },
            "cleaning": {
                "cleaning_rules": {
                    "name": {
                        "trim": True,
                        "normalize_whitespace": True
                    }
                },
                "null_strategies": {
                    "email": "remove_record"
                }
            },
            "enrichment": {
                "enrichment_rules": [
                    {
                        "type": "calculation",
                        "source_field": "amount",
                        "target_field": "amount_tier",
                        "formula": "premium if {amount} > 1000 else standard"
                    }
                ]
            },
            "quality": {
                "quality_thresholds": {
                    "minimum_score": 0.7
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_enterprise_pipeline_creation(self, sample_schema, pipeline_config):
        """Test enterprise pipeline creation and configuration"""
        pipeline = create_enterprise_pipeline(sample_schema, pipeline_config)
        
        assert len(pipeline.processors) == 4  # validator, cleaner, enricher, quality
        assert pipeline.processing_mode.value == "streaming"
        
        # Verify processor types
        processor_names = [p.name for p in pipeline.processors]
        assert "schema_validator" in processor_names
        assert "data_cleaner" in processor_names
        assert "data_enricher" in processor_names
        assert "quality_assessment" in processor_names
    
    @pytest.mark.asyncio
    async def test_pipeline_record_processing(self, sample_schema, pipeline_config):
        """Test pipeline record processing"""
        pipeline = create_enterprise_pipeline(sample_schema, pipeline_config)
        
        # Create test record
        test_record = DataRecord(
            source_id="test_source",
            record_id="test_record_1",
            data={
                "id": 1,
                "name": "  Test Customer  ",
                "email": "test@example.com",
                "amount": 1500.00,
                "created_date": "2024-01-15T10:30:00Z"
            },
            timestamp=datetime.utcnow(),
            metadata={"source": "test"}
        )
        
        # Process record
        processed_record = await pipeline.process_record(test_record)
        
        assert processed_record is not None
        assert processed_record.record_id == test_record.record_id
        
        # Verify processing metadata
        metadata = processed_record.metadata
        assert "cleaning_applied" in metadata
        assert "enrichments_applied" in metadata
        assert "quality_metrics" in metadata
        
        # Verify data transformations
        data = processed_record.data
        assert data["name"] == "Test Customer"  # Trimmed
        assert "amount_tier" in data  # Enriched
        assert "processing_timestamp" in data  # Standard enrichment
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_processing(self, sample_schema, pipeline_config):
        """Test pipeline batch processing"""
        pipeline = create_enterprise_pipeline(sample_schema, pipeline_config)
        
        # Create test records
        test_records = []
        for i in range(10):
            record = DataRecord(
                source_id="test_source",
                record_id=f"test_record_{i}",
                data={
                    "id": i,
                    "name": f"Customer {i}",
                    "email": f"customer{i}@example.com",
                    "amount": 100.0 + i * 50,
                    "created_date": datetime.utcnow().isoformat()
                },
                timestamp=datetime.utcnow(),
                metadata={"batch": True}
            )
            test_records.append(record)
        
        # Process batch
        processed_records = await pipeline.process_batch(test_records)
        
        assert len(processed_records) <= len(test_records)  # Some may be filtered
        
        # Verify all processed records have required metadata
        for record in processed_records:
            assert "quality_metrics" in record.metadata
            assert "processing_timestamp" in record.data
    
    @pytest.mark.asyncio
    async def test_pipeline_statistics(self, sample_schema, pipeline_config):
        """Test pipeline statistics collection"""
        pipeline = create_enterprise_pipeline(sample_schema, pipeline_config)
        
        # Process some records
        for i in range(5):
            record = DataRecord(
                source_id="test_source",
                record_id=f"stats_record_{i}",
                data={
                    "id": i,
                    "name": f"Stats Customer {i}",
                    "email": f"stats{i}@example.com",
                    "amount": 200.0 + i * 25
                },
                timestamp=datetime.utcnow(),
                metadata={}
            )
            await pipeline.process_record(record)
        
        # Get statistics
        stats = pipeline.get_pipeline_stats()
        
        assert "pipeline_stats" in stats
        assert "processor_stats" in stats
        assert "processing_mode" in stats
        assert "active_processors" in stats
        
        pipeline_stats = stats["pipeline_stats"]
        assert pipeline_stats["records_processed"] == 5
        assert pipeline_stats["throughput_per_second"] > 0


class TestERPConnectors:
    """Test ERP system connectors"""
    
    @pytest.fixture
    def sap_config(self):
        return DataSourceConfig(
            source_id="test_sap",
            source_type=DataSourceType.ERP,
            name="Test SAP",
            connection_params={
                "host": "sap.example.com",
                "client": "100",
                "username": "testuser",
                "password": "testpass"
            }
        )
    
    @pytest.mark.asyncio
    async def test_sap_connector(self, sap_config):
        """Test SAP ERP connector"""
        connector = SAPConnector(sap_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected is True
        
        # Test data fetch
        query = {
            "table": "MARA",
            "fields": ["MATNR", "MAKTX", "MTART"],
            "limit": 20
        }
        records = await connector.fetch_data(query)
        
        assert len(records) > 0
        assert all(record.source_id == sap_config.source_id for record in records)
        
        # Verify SAP-specific data structure
        for record in records[:3]:
            data = record.data
            assert "MATNR" in data  # Material number
            assert "MAKTX" in data  # Material description
            assert record.metadata["table"] == "MARA"
        
        # Test schema
        schema = await connector.get_schema()
        assert "tables" in schema
        assert "MARA" in schema["tables"]


class TestCRMConnectors:
    """Test CRM system connectors"""
    
    @pytest.fixture
    def salesforce_config(self):
        return DataSourceConfig(
            source_id="test_salesforce",
            source_type=DataSourceType.CRM,
            name="Test Salesforce",
            connection_params={
                "instance_url": "https://test.salesforce.com",
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "username": "test@example.com",
                "password": "testpass"
            }
        )
    
    @pytest.mark.asyncio
    async def test_salesforce_connector(self, salesforce_config):
        """Test Salesforce CRM connector"""
        connector = SalesforceConnector(salesforce_config)
        
        # Test connection
        connected = await connector.connect()
        assert connected is True
        
        # Test data fetch
        query = {
            "sobject": "Account",
            "fields": ["Id", "Name", "Type", "Industry"],
            "limit": 15
        }
        records = await connector.fetch_data(query)
        
        assert len(records) > 0
        
        # Verify Salesforce-specific data structure
        for record in records[:3]:
            data = record.data
            assert "Id" in data
            assert "Name" in data
            assert len(data["Id"]) == 18  # Salesforce ID length
            assert record.metadata["sobject"] == "Account"
        
        # Test schema
        schema = await connector.get_schema()
        assert "sobjects" in schema
        assert "Account" in schema["sobjects"]


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])