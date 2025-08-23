"""
Data Lake Connectors for Snowflake, Databricks, BigQuery, and other cloud data warehouses.
Implements advanced streaming, partitioning, and real-time analytics capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import gzip
import base64
from urllib.parse import urljoin

from ..core.data_connector import BaseDataConnector, DataRecord, ConnectionStatus, DataSourceConfig
from .enterprise_data_connectors import (
    EnterpriseDataValidator, EnterpriseDataEnricher, RealTimeDataStreamer,
    StreamingConfig, StreamingMode, DataValidationLevel
)

logger = logging.getLogger(__name__)


class DataLakeFormat(Enum):
    """Supported data lake formats"""
    PARQUET = "parquet"
    DELTA = "delta"
    ICEBERG = "iceberg"
    HUDI = "hudi"
    AVRO = "avro"
    ORC = "orc"


class CompressionType(Enum):
    """Compression types for data lake storage"""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class PartitionConfig:
    """Configuration for data partitioning"""
    partition_columns: List[str]
    partition_strategy: str = "hive"  # hive, range, hash
    partition_granularity: str = "daily"  # hourly, daily, monthly
    max_partitions: int = 1000


@dataclass
class DataLakeConfig:
    """Configuration for data lake operations"""
    format: DataLakeFormat = DataLakeFormat.PARQUET
    compression: CompressionType = CompressionType.SNAPPY
    partition_config: Optional[PartitionConfig] = None
    schema_evolution: bool = True
    time_travel: bool = False
    change_data_capture: bool = False
    auto_optimize: bool = True
    vacuum_retention_hours: int = 168  # 7 days


class BigQueryConnector(BaseDataConnector):
    """Google BigQuery data warehouse connector"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.validator = EnterpriseDataValidator(DataValidationLevel.ENTERPRISE)
        self.enricher = EnterpriseDataEnricher()
        self.streamer = RealTimeDataStreamer(StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            batch_size=1000,
            flush_interval=30,
            compression=True,
            encryption=True
        ))
        self.data_lake_config = DataLakeConfig(
            format=DataLakeFormat.PARQUET,
            partition_config=PartitionConfig(
                partition_columns=["event_date"],
                partition_strategy="range",
                partition_granularity="daily"
            ),
            time_travel=True,
            change_data_capture=True
        )
    
    async def connect(self) -> bool:
        """Connect to BigQuery with service account authentication"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # BigQuery connection parameters
            project_id = self.config.connection_params.get('project_id')
            dataset_id = self.config.connection_params.get('dataset_id')
            service_account_key = self.config.connection_params.get('service_account_key')
            location = self.config.connection_params.get('location', 'US')
            
            if not all([project_id, dataset_id]):
                raise ValueError("Missing required BigQuery connection parameters")
            
            if not service_account_key:
                # Try to use default credentials
                logger.info("Using default Google Cloud credentials")
            
            # Simulate BigQuery authentication
            await asyncio.sleep(1.5)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to BigQuery: {project_id}.{dataset_id}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"BigQuery connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from BigQuery"""
        try:
            await self.streamer.flush_buffer()
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from BigQuery")
            return True
        except Exception as e:
            logger.error(f"BigQuery disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test BigQuery connection"""
        try:
            await asyncio.sleep(0.8)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from BigQuery with advanced analytics"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to BigQuery")
        
        try:
            sql_query = query.get('sql', 'SELECT * FROM `dataset.table` LIMIT 1000')
            use_legacy_sql = query.get('use_legacy_sql', False)
            job_config = query.get('job_config', {})
            
            # Simulate BigQuery job execution
            await asyncio.sleep(3.5)
            
            # Mock BigQuery analytics data
            raw_records = []
            for i in range(min(query.get('limit', 1000), 400)):
                raw_record = {
                    'row_id': f'bq_{i:012d}',
                    'event_timestamp': (datetime.utcnow() - timedelta(hours=i % 24)).isoformat(),
                    'event_date': (datetime.utcnow() - timedelta(days=i % 30)).date().isoformat(),
                    'user_pseudo_id': f'user_{(i % 50000):08d}',
                    'session_id': f'session_{(i % 5000):06d}',
                    'event_name': ['page_view', 'scroll', 'click', 'purchase', 'sign_up'][i % 5],
                    'event_params': json.dumps({
                        'page_title': f'Page {i % 100}',
                        'page_location': f'https://example.com/page/{i % 100}',
                        'engagement_time_msec': (i % 300) * 1000,
                        'value': round((i % 50) * 2.5, 2) if i % 10 == 0 else None
                    }),
                    'user_properties': json.dumps({
                        'user_category': ['new', 'returning', 'loyal'][i % 3],
                        'subscription_tier': ['free', 'premium', 'enterprise'][i % 3],
                        'registration_date': (datetime.utcnow() - timedelta(days=i % 365)).date().isoformat()
                    }),
                    'device': json.dumps({
                        'category': ['desktop', 'mobile', 'tablet'][i % 3],
                        'operating_system': ['Windows', 'macOS', 'iOS', 'Android', 'Linux'][i % 5],
                        'browser': ['Chrome', 'Safari', 'Firefox', 'Edge'][i % 4],
                        'language': ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'ja-JP'][i % 5]
                    }),
                    'geo': json.dumps({
                        'country': ['US', 'CA', 'GB', 'DE', 'FR', 'JP', 'AU'][i % 7],
                        'region': f'Region-{i % 20}',
                        'city': f'City-{i % 100}',
                        'continent': ['Americas', 'Europe', 'Asia', 'Oceania'][i % 4]
                    }),
                    'traffic_source': json.dumps({
                        'medium': ['organic', 'cpc', 'email', 'social', 'referral'][i % 5],
                        'source': ['google', 'facebook', 'twitter', 'email', 'direct'][i % 5],
                        'campaign': f'campaign_{i % 10}' if i % 3 == 0 else None
                    }),
                    'ecommerce': json.dumps({
                        'transaction_id': f'txn_{i:010d}' if i % 20 == 0 else None,
                        'purchase_revenue': round((i % 100) * 12.5, 2) if i % 20 == 0 else None,
                        'items': [
                            {
                                'item_id': f'item_{(i % 1000):04d}',
                                'item_name': f'Product {i % 1000}',
                                'item_category': ['Electronics', 'Clothing', 'Books', 'Home'][i % 4],
                                'price': round(10 + (i % 50) * 5, 2),
                                'quantity': 1 + (i % 3)
                            }
                        ] if i % 20 == 0 else []
                    }),
                    'custom_dimensions': json.dumps({
                        'experiment_id': f'exp_{i % 5}',
                        'variant': ['A', 'B', 'C'][i % 3],
                        'cohort': f'cohort_{(i // 100) % 10}',
                        'ltv_segment': ['high', 'medium', 'low'][i % 3]
                    }),
                    'ml_predictions': json.dumps({
                        'churn_probability': round(0.1 + (i % 80) * 0.01, 3),
                        'conversion_probability': round(0.05 + (i % 90) * 0.01, 3),
                        'lifetime_value': round(50 + (i % 500) * 2, 2),
                        'next_purchase_days': 7 + (i % 30)
                    }),
                    '_table_suffix': f'{(datetime.utcnow() - timedelta(days=i % 30)).strftime("%Y%m%d")}',
                    '_bq_load_timestamp': datetime.utcnow().isoformat(),
                    '_partition_date': (datetime.utcnow() - timedelta(days=i % 30)).date().isoformat()
                }
                raw_records.append(raw_record)
            
            # Advanced processing with BigQuery-specific features
            processed_records = []
            for raw_record in raw_records:
                # Validate with BigQuery schema constraints
                validation_result = await self.validator.validate_record(raw_record)
                if not validation_result["is_valid"]:
                    continue
                
                # Enrich with BigQuery ML predictions
                enriched_record = await self.enricher.enrich_record(validation_result["transformed_record"])
                
                # Add BigQuery-specific analytics
                enriched_record.update({
                    'audience_segments': self._calculate_audience_segments(enriched_record),
                    'attribution_model': self._apply_attribution_model(enriched_record),
                    'funnel_stage': self._determine_funnel_stage(enriched_record),
                    'cohort_analysis': self._perform_cohort_analysis(enriched_record),
                    'anomaly_detection': self._detect_anomalies(enriched_record)
                })
                
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"BIGQUERY_{enriched_record['row_id']}",
                    data=enriched_record,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'bigquery_project': self.config.connection_params.get('project_id'),
                        'dataset': self.config.connection_params.get('dataset_id'),
                        'table_suffix': enriched_record.get('_table_suffix'),
                        'partition_date': enriched_record.get('_partition_date'),
                        'job_id': f'job_{hash(sql_query) % 1000000:06d}',
                        'slot_ms': (i % 1000) * 100,
                        'bytes_processed': (i % 10000) * 1024,
                        'cache_hit': i % 5 == 0
                    }
                )
                
                processed_records.append(record)
                await self.streamer.add_record(record)
            
            logger.info(f"Fetched {len(processed_records)} records from BigQuery")
            return processed_records
            
        except Exception as e:
            logger.error(f"BigQuery data fetch failed: {e}")
            raise
    
    def _calculate_audience_segments(self, record: Dict[str, Any]) -> List[str]:
        """Calculate audience segments for the user"""
        segments = []
        
        try:
            user_props = json.loads(record.get('user_properties', '{}'))
            device_info = json.loads(record.get('device', '{}'))
            
            # Subscription-based segments
            tier = user_props.get('subscription_tier', 'free')
            segments.append(f'{tier}_users')
            
            # Device-based segments
            category = device_info.get('category', 'unknown')
            segments.append(f'{category}_users')
            
            # Behavioral segments
            event_name = record.get('event_name', '')
            if event_name == 'purchase':
                segments.append('purchasers')
            elif event_name == 'sign_up':
                segments.append('new_signups')
            
        except Exception:
            segments = ['unknown_segment']
        
        return segments
    
    def _apply_attribution_model(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply attribution model to the event"""
        try:
            traffic_source = json.loads(record.get('traffic_source', '{}'))
            event_name = record.get('event_name', '')
            
            attribution = {
                'first_touch': traffic_source.get('source', 'unknown'),
                'last_touch': traffic_source.get('source', 'unknown'),
                'linear_weight': 1.0,
                'time_decay_weight': 0.8,
                'position_based_weight': 0.6
            }
            
            # Adjust weights based on event type
            if event_name == 'purchase':
                attribution['conversion_credit'] = 1.0
            else:
                attribution['conversion_credit'] = 0.1
            
            return attribution
        except Exception:
            return {'model': 'unknown'}
    
    def _determine_funnel_stage(self, record: Dict[str, Any]) -> str:
        """Determine user's position in the conversion funnel"""
        event_name = record.get('event_name', '')
        
        funnel_mapping = {
            'page_view': 'awareness',
            'scroll': 'interest',
            'click': 'consideration',
            'sign_up': 'intent',
            'purchase': 'conversion'
        }
        
        return funnel_mapping.get(event_name, 'unknown')
    
    def _perform_cohort_analysis(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cohort analysis for the user"""
        try:
            user_props = json.loads(record.get('user_properties', '{}'))
            reg_date = user_props.get('registration_date', '')
            event_date = record.get('event_date', '')
            
            if reg_date and event_date:
                reg_dt = datetime.fromisoformat(reg_date)
                event_dt = datetime.fromisoformat(event_date)
                days_since_reg = (event_dt - reg_dt).days
                
                return {
                    'cohort_month': reg_dt.strftime('%Y-%m'),
                    'days_since_registration': days_since_reg,
                    'cohort_week': max(0, days_since_reg // 7),
                    'retention_bucket': self._get_retention_bucket(days_since_reg)
                }
        except Exception:
            pass
        
        return {'cohort_month': 'unknown'}
    
    def _get_retention_bucket(self, days: int) -> str:
        """Get retention bucket based on days since registration"""
        if days <= 1:
            return 'day_1'
        elif days <= 7:
            return 'week_1'
        elif days <= 30:
            return 'month_1'
        elif days <= 90:
            return 'quarter_1'
        else:
            return 'long_term'
    
    def _detect_anomalies(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in the event data"""
        anomalies = []
        
        try:
            # Check for unusual engagement time
            event_params = json.loads(record.get('event_params', '{}'))
            engagement_time = event_params.get('engagement_time_msec', 0)
            
            if engagement_time > 300000:  # > 5 minutes
                anomalies.append('high_engagement_time')
            elif engagement_time == 0:
                anomalies.append('zero_engagement_time')
            
            # Check for unusual purchase values
            ecommerce = json.loads(record.get('ecommerce', '{}'))
            revenue = ecommerce.get('purchase_revenue', 0)
            
            if revenue > 10000:  # > $10,000
                anomalies.append('high_value_purchase')
            
            # Check for bot-like behavior
            device_info = json.loads(record.get('device', '{}'))
            user_agent = device_info.get('browser', '')
            
            if 'bot' in user_agent.lower():
                anomalies.append('potential_bot')
            
        except Exception:
            anomalies.append('parsing_error')
        
        return {
            'anomalies_detected': anomalies,
            'anomaly_score': len(anomalies) * 0.25,
            'requires_review': len(anomalies) > 2
        }
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get BigQuery dataset schema"""
        return {
            'project_id': self.config.connection_params.get('project_id'),
            'dataset_id': self.config.connection_params.get('dataset_id'),
            'location': self.config.connection_params.get('location', 'US'),
            'tables': {
                'events_*': {
                    'description': 'Google Analytics 4 Events (Sharded)',
                    'table_type': 'TABLE',
                    'partitioning': {
                        'type': 'TIME',
                        'field': 'event_timestamp',
                        'granularity': 'DAY'
                    },
                    'clustering': ['event_name', 'user_pseudo_id'],
                    'schema': [
                        {'name': 'event_timestamp', 'type': 'INTEGER', 'mode': 'NULLABLE'},
                        {'name': 'event_name', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'event_params', 'type': 'RECORD', 'mode': 'REPEATED'},
                        {'name': 'user_pseudo_id', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'user_properties', 'type': 'RECORD', 'mode': 'REPEATED'},
                        {'name': 'device', 'type': 'RECORD', 'mode': 'NULLABLE'},
                        {'name': 'geo', 'type': 'RECORD', 'mode': 'NULLABLE'},
                        {'name': 'traffic_source', 'type': 'RECORD', 'mode': 'NULLABLE'},
                        {'name': 'ecommerce', 'type': 'RECORD', 'mode': 'NULLABLE'}
                    ]
                },
                'user_dimensions': {
                    'description': 'User Dimension Table',
                    'table_type': 'TABLE',
                    'partitioning': {
                        'type': 'TIME',
                        'field': 'last_updated',
                        'granularity': 'DAY'
                    },
                    'schema': [
                        {'name': 'user_pseudo_id', 'type': 'STRING', 'mode': 'REQUIRED'},
                        {'name': 'first_seen_date', 'type': 'DATE', 'mode': 'NULLABLE'},
                        {'name': 'last_seen_date', 'type': 'DATE', 'mode': 'NULLABLE'},
                        {'name': 'lifetime_value', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                        {'name': 'user_segment', 'type': 'STRING', 'mode': 'NULLABLE'}
                    ]
                }
            }
        }


class RedshiftConnector(BaseDataConnector):
    """Amazon Redshift data warehouse connector"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.validator = EnterpriseDataValidator(DataValidationLevel.ENTERPRISE)
        self.enricher = EnterpriseDataEnricher()
        self.streamer = RealTimeDataStreamer(StreamingConfig(
            mode=StreamingMode.MICRO_BATCH,
            batch_size=2000,
            flush_interval=45,
            compression=True,
            encryption=True
        ))
        self.data_lake_config = DataLakeConfig(
            format=DataLakeFormat.PARQUET,
            compression=CompressionType.GZIP,
            partition_config=PartitionConfig(
                partition_columns=["year", "month", "day"],
                partition_strategy="hive"
            )
        )
    
    async def connect(self) -> bool:
        """Connect to Amazon Redshift"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Redshift connection parameters
            host = self.config.connection_params.get('host')
            port = self.config.connection_params.get('port', 5439)
            database = self.config.connection_params.get('database')
            user = self.config.connection_params.get('user')
            password = self.config.connection_params.get('password')
            cluster_identifier = self.config.connection_params.get('cluster_identifier')
            
            # Support for IAM authentication
            iam_role = self.config.connection_params.get('iam_role')
            aws_access_key_id = self.config.connection_params.get('aws_access_key_id')
            aws_secret_access_key = self.config.connection_params.get('aws_secret_access_key')
            
            if not all([host, database, user]):
                raise ValueError("Missing required Redshift connection parameters")
            
            if not password and not (iam_role or aws_access_key_id):
                raise ValueError("Either password or AWS credentials required")
            
            # Simulate Redshift connection
            await asyncio.sleep(2)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Redshift: {host}:{port}/{database}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Redshift connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Redshift"""
        try:
            await self.streamer.flush_buffer()
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Redshift")
            return True
        except Exception as e:
            logger.error(f"Redshift disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Redshift connection"""
        try:
            await asyncio.sleep(1)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from Redshift with columnar analytics"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Redshift")
        
        try:
            sql_query = query.get('sql', 'SELECT * FROM sales_fact LIMIT 1000')
            workload_management = query.get('wlm_queue', 'default')
            
            # Simulate Redshift query execution with columnar processing
            await asyncio.sleep(4)
            
            # Mock Redshift data warehouse data
            raw_records = []
            for i in range(min(query.get('limit', 1000), 250)):
                raw_record = {
                    'sale_id': i + 1000000,
                    'transaction_date': (datetime.utcnow() - timedelta(days=i % 365)).date().isoformat(),
                    'transaction_timestamp': (datetime.utcnow() - timedelta(hours=i % 8760)).isoformat(),
                    'customer_id': f'CUST_{(i % 100000):08d}',
                    'product_id': f'PROD_{(i % 10000):06d}',
                    'store_id': f'STORE_{(i % 500):04d}',
                    'sales_rep_id': f'REP_{(i % 1000):05d}',
                    'quantity': 1 + (i % 10),
                    'unit_price': round(5.0 + (i % 200) * 2.5, 2),
                    'discount_amount': round((i % 20) * 0.5, 2),
                    'tax_amount': round((5.0 + (i % 200) * 2.5) * 0.08, 2),
                    'total_amount': round((5.0 + (i % 200) * 2.5) * (1 + (i % 10)) * 1.08, 2),
                    'cost_of_goods': round((5.0 + (i % 200) * 2.5) * 0.6, 2),
                    'profit_margin': round((5.0 + (i % 200) * 2.5) * 0.4, 2),
                    'payment_method': ['CREDIT', 'DEBIT', 'CASH', 'CHECK', 'GIFT_CARD'][i % 5],
                    'channel': ['ONLINE', 'RETAIL', 'MOBILE', 'PHONE', 'CATALOG'][i % 5],
                    'region': ['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL'][i % 5],
                    'country': ['US', 'CA', 'MX', 'UK', 'DE', 'FR', 'JP', 'AU'][i % 8],
                    'currency': 'USD',
                    'exchange_rate': 1.0,
                    'promotion_id': f'PROMO_{i % 50:03d}' if i % 10 == 0 else None,
                    'campaign_id': f'CAMP_{i % 20:03d}' if i % 15 == 0 else None,
                    'customer_segment': ['PREMIUM', 'STANDARD', 'BASIC', 'VIP'][i % 4],
                    'product_category': ['ELECTRONICS', 'CLOTHING', 'HOME', 'BOOKS', 'SPORTS'][i % 5],
                    'product_subcategory': f'SUBCAT_{i % 25:03d}',
                    'brand': f'BRAND_{i % 100:03d}',
                    'supplier_id': f'SUPP_{i % 200:04d}',
                    'warehouse_id': f'WH_{i % 50:03d}',
                    'shipping_method': ['STANDARD', 'EXPRESS', 'OVERNIGHT', 'PICKUP'][i % 4],
                    'shipping_cost': round((i % 20) * 2.5, 2),
                    'order_priority': ['LOW', 'MEDIUM', 'HIGH', 'URGENT'][i % 4],
                    'return_flag': i % 50 == 0,
                    'return_reason': 'DEFECTIVE' if i % 50 == 0 else None,
                    'satisfaction_score': 3.5 + (i % 15) * 0.1,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat(),
                    'etl_batch_id': f'batch_{(i // 1000):06d}',
                    'data_source': 'pos_system',
                    'year': (datetime.utcnow() - timedelta(days=i % 365)).year,
                    'month': (datetime.utcnow() - timedelta(days=i % 365)).month,
                    'day': (datetime.utcnow() - timedelta(days=i % 365)).day,
                    'quarter': ((datetime.utcnow() - timedelta(days=i % 365)).month - 1) // 3 + 1,
                    'week_of_year': (datetime.utcnow() - timedelta(days=i % 365)).isocalendar()[1]
                }
                raw_records.append(raw_record)
            
            # Process with enterprise data warehouse features
            processed_records = []
            for raw_record in raw_records:
                # Validate with data warehouse constraints
                validation_result = await self.validator.validate_record(raw_record)
                if not validation_result["is_valid"]:
                    continue
                
                # Enrich with analytical calculations
                enriched_record = await self.enricher.enrich_record(validation_result["transformed_record"])
                
                # Add Redshift-specific analytics
                enriched_record.update({
                    'sales_metrics': self._calculate_sales_metrics(enriched_record),
                    'customer_analytics': self._analyze_customer_behavior(enriched_record),
                    'product_performance': self._analyze_product_performance(enriched_record),
                    'seasonal_trends': self._calculate_seasonal_trends(enriched_record),
                    'forecasting_features': self._generate_forecasting_features(enriched_record)
                })
                
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"REDSHIFT_{enriched_record['sale_id']}",
                    data=enriched_record,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'redshift_cluster': self.config.connection_params.get('cluster_identifier'),
                        'database': self.config.connection_params.get('database'),
                        'wlm_queue': workload_management,
                        'query_id': f'query_{hash(sql_query) % 1000000:06d}',
                        'compression_encoding': 'LZO',
                        'sort_key': 'transaction_date',
                        'dist_key': 'customer_id',
                        'table_stats': {
                            'rows_scanned': (i % 1000000) + 1000000,
                            'bytes_scanned': (i % 10000000) + 10000000,
                            'query_time_ms': 2000 + (i % 3000)
                        }
                    }
                )
                
                processed_records.append(record)
                await self.streamer.add_record(record)
            
            logger.info(f"Fetched {len(processed_records)} records from Redshift")
            return processed_records
            
        except Exception as e:
            logger.error(f"Redshift data fetch failed: {e}")
            raise
    
    def _calculate_sales_metrics(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive sales metrics"""
        total_amount = record.get('total_amount', 0)
        cost_of_goods = record.get('cost_of_goods', 0)
        quantity = record.get('quantity', 1)
        
        return {
            'gross_profit': total_amount - cost_of_goods,
            'profit_margin_percent': ((total_amount - cost_of_goods) / total_amount * 100) if total_amount > 0 else 0,
            'average_selling_price': total_amount / quantity if quantity > 0 else 0,
            'revenue_per_unit': total_amount / quantity if quantity > 0 else 0,
            'markup_percent': ((total_amount - cost_of_goods) / cost_of_goods * 100) if cost_of_goods > 0 else 0
        }
    
    def _analyze_customer_behavior(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer behavior patterns"""
        customer_segment = record.get('customer_segment', 'STANDARD')
        total_amount = record.get('total_amount', 0)
        channel = record.get('channel', 'RETAIL')
        
        # Simulate customer analytics
        behavior_score = hash(record.get('customer_id', '')) % 100
        
        return {
            'customer_value_tier': self._get_value_tier(total_amount),
            'channel_preference': channel,
            'purchase_frequency_score': (behavior_score % 10) + 1,
            'brand_loyalty_score': (behavior_score % 5) * 0.2,
            'price_sensitivity': ['LOW', 'MEDIUM', 'HIGH'][behavior_score % 3],
            'cross_sell_opportunity': behavior_score % 4 == 0,
            'churn_risk': 'HIGH' if behavior_score % 10 < 2 else 'LOW'
        }
    
    def _get_value_tier(self, amount: float) -> str:
        """Determine customer value tier based on purchase amount"""
        if amount > 1000:
            return 'HIGH_VALUE'
        elif amount > 500:
            return 'MEDIUM_VALUE'
        elif amount > 100:
            return 'STANDARD_VALUE'
        else:
            return 'LOW_VALUE'
    
    def _analyze_product_performance(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product performance metrics"""
        product_category = record.get('product_category', 'UNKNOWN')
        profit_margin = record.get('profit_margin', 0)
        quantity = record.get('quantity', 1)
        
        return {
            'category_performance': self._get_category_performance(product_category),
            'inventory_velocity': 'FAST' if quantity > 5 else 'SLOW',
            'margin_category': 'HIGH' if profit_margin > 50 else 'STANDARD',
            'demand_indicator': quantity * 10,  # Simplified demand score
            'seasonality_factor': self._get_seasonality_factor(product_category),
            'competitive_position': ['LEADER', 'CHALLENGER', 'FOLLOWER'][hash(record.get('product_id', '')) % 3]
        }
    
    def _get_category_performance(self, category: str) -> str:
        """Get category performance rating"""
        performance_map = {
            'ELECTRONICS': 'HIGH',
            'CLOTHING': 'MEDIUM',
            'HOME': 'MEDIUM',
            'BOOKS': 'LOW',
            'SPORTS': 'HIGH'
        }
        return performance_map.get(category, 'MEDIUM')
    
    def _get_seasonality_factor(self, category: str) -> float:
        """Get seasonality factor for product category"""
        seasonality_map = {
            'ELECTRONICS': 1.2,  # Higher during holidays
            'CLOTHING': 1.5,     # Seasonal variations
            'HOME': 1.1,         # Moderate seasonality
            'BOOKS': 0.9,        # Lower seasonality
            'SPORTS': 1.3        # Seasonal sports equipment
        }
        return seasonality_map.get(category, 1.0)
    
    def _calculate_seasonal_trends(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate seasonal trend indicators"""
        month = record.get('month', 1)
        quarter = record.get('quarter', 1)
        
        # Seasonal factors by month
        seasonal_factors = {
            1: 0.8, 2: 0.7, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.3, 8: 1.2, 9: 1.0, 10: 1.1, 11: 1.4, 12: 1.6
        }
        
        return {
            'seasonal_factor': seasonal_factors.get(month, 1.0),
            'quarter_trend': 'UP' if quarter in [2, 4] else 'DOWN',
            'holiday_proximity': month in [11, 12, 1],
            'back_to_school_season': month in [8, 9],
            'summer_season': month in [6, 7, 8],
            'year_end_effect': month == 12
        }
    
    def _generate_forecasting_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features for sales forecasting models"""
        transaction_date = record.get('transaction_date', '')
        total_amount = record.get('total_amount', 0)
        
        try:
            date_obj = datetime.fromisoformat(transaction_date)
            day_of_week = date_obj.weekday()
            day_of_month = date_obj.day
            
            return {
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5,
                'is_month_end': day_of_month >= 28,
                'is_month_start': day_of_month <= 3,
                'week_of_month': (day_of_month - 1) // 7 + 1,
                'sales_momentum': min(total_amount / 100, 10),  # Capped momentum score
                'trend_indicator': 1 if total_amount > 500 else -1,
                'volatility_score': abs(hash(str(total_amount)) % 100) / 100
            }
        except Exception:
            return {'forecasting_error': True}
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Redshift database schema"""
        return {
            'cluster_identifier': self.config.connection_params.get('cluster_identifier'),
            'database': self.config.connection_params.get('database'),
            'schemas': {
                'public': {
                    'tables': {
                        'sales_fact': {
                            'description': 'Sales Fact Table',
                            'table_type': 'BASE TABLE',
                            'distribution_style': 'KEY',
                            'distribution_key': 'customer_id',
                            'sort_keys': ['transaction_date', 'store_id'],
                            'compression': 'LZO',
                            'columns': {
                                'sale_id': {'type': 'BIGINT', 'nullable': False, 'encoding': 'DELTA'},
                                'transaction_date': {'type': 'DATE', 'nullable': False, 'encoding': 'DELTA32K'},
                                'customer_id': {'type': 'VARCHAR(20)', 'nullable': False, 'encoding': 'LZO'},
                                'product_id': {'type': 'VARCHAR(15)', 'nullable': False, 'encoding': 'LZO'},
                                'quantity': {'type': 'INTEGER', 'nullable': False, 'encoding': 'DELTA'},
                                'total_amount': {'type': 'DECIMAL(10,2)', 'nullable': False, 'encoding': 'DELTA32K'}
                            }
                        },
                        'customer_dim': {
                            'description': 'Customer Dimension Table',
                            'table_type': 'BASE TABLE',
                            'distribution_style': 'ALL',
                            'sort_keys': ['customer_id'],
                            'columns': {
                                'customer_id': {'type': 'VARCHAR(20)', 'nullable': False},
                                'customer_name': {'type': 'VARCHAR(100)', 'nullable': False},
                                'customer_segment': {'type': 'VARCHAR(20)', 'nullable': True},
                                'registration_date': {'type': 'DATE', 'nullable': True}
                            }
                        }
                    }
                }
            }
        }