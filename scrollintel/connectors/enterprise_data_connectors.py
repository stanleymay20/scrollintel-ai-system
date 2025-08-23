"""
Enterprise Data Integration Layer - Real-time connectors for enterprise systems.
Implements secure data streaming, validation, and enrichment pipelines.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import ssl
import aiohttp
from urllib.parse import urljoin
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet

from ..core.data_connector import BaseDataConnector, DataRecord, ConnectionStatus, DataSourceConfig

logger = logging.getLogger(__name__)


class DataValidationLevel(Enum):
    """Data validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class StreamingMode(Enum):
    """Real-time streaming modes"""
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    REAL_TIME = "real_time"
    CHANGE_DATA_CAPTURE = "cdc"


@dataclass
class ValidationRule:
    """Data validation rule definition"""
    field_name: str
    rule_type: str  # required, type, range, pattern, custom
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_action: str = "reject"  # reject, warn, transform


@dataclass
class EnrichmentRule:
    """Data enrichment rule definition"""
    source_field: str
    target_field: str
    enrichment_type: str  # lookup, calculation, external_api, ml_inference
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingConfig:
    """Configuration for real-time streaming"""
    mode: StreamingMode
    batch_size: int = 1000
    flush_interval: int = 30  # seconds
    buffer_size: int = 10000
    compression: bool = True
    encryption: bool = True
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2,
        "max_delay": 300
    })


class EnterpriseDataValidator:
    """Enterprise-grade data validation engine"""
    
    def __init__(self, validation_level: DataValidationLevel = DataValidationLevel.ENTERPRISE):
        self.validation_level = validation_level
        self.validation_rules: List[ValidationRule] = []
        self.validation_stats = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "warnings": 0
        }
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.validation_rules.append(rule)
        logger.info(f"Added validation rule for field: {rule.field_name}")
    
    async def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single record"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "transformed_record": record.copy()
        }
        
        self.validation_stats["total_records"] += 1
        
        for rule in self.validation_rules:
            field_value = record.get(rule.field_name)
            
            try:
                if rule.rule_type == "required":
                    if field_value is None or field_value == "":
                        validation_result["errors"].append(f"Required field '{rule.field_name}' is missing")
                        validation_result["is_valid"] = False
                
                elif rule.rule_type == "type":
                    expected_type = rule.parameters.get("type")
                    if field_value is not None and not isinstance(field_value, expected_type):
                        if rule.error_action == "transform":
                            try:
                                validation_result["transformed_record"][rule.field_name] = expected_type(field_value)
                                validation_result["warnings"].append(f"Transformed '{rule.field_name}' to {expected_type.__name__}")
                            except (ValueError, TypeError):
                                validation_result["errors"].append(f"Cannot convert '{rule.field_name}' to {expected_type.__name__}")
                                validation_result["is_valid"] = False
                        else:
                            validation_result["errors"].append(f"Field '{rule.field_name}' must be of type {expected_type.__name__}")
                            validation_result["is_valid"] = False
                
                elif rule.rule_type == "range":
                    if field_value is not None:
                        min_val = rule.parameters.get("min")
                        max_val = rule.parameters.get("max")
                        if min_val is not None and field_value < min_val:
                            validation_result["errors"].append(f"Field '{rule.field_name}' below minimum value {min_val}")
                            validation_result["is_valid"] = False
                        if max_val is not None and field_value > max_val:
                            validation_result["errors"].append(f"Field '{rule.field_name}' above maximum value {max_val}")
                            validation_result["is_valid"] = False
                
                elif rule.rule_type == "pattern":
                    import re
                    if field_value is not None:
                        pattern = rule.parameters.get("pattern")
                        if pattern and not re.match(pattern, str(field_value)):
                            validation_result["errors"].append(f"Field '{rule.field_name}' does not match required pattern")
                            validation_result["is_valid"] = False
                
            except Exception as e:
                logger.error(f"Validation error for rule {rule.field_name}: {e}")
                validation_result["errors"].append(f"Validation rule error: {e}")
                validation_result["is_valid"] = False
        
        if validation_result["is_valid"]:
            self.validation_stats["valid_records"] += 1
        else:
            self.validation_stats["invalid_records"] += 1
        
        if validation_result["warnings"]:
            self.validation_stats["warnings"] += len(validation_result["warnings"])
        
        return validation_result
    
    async def validate_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of records"""
        validation_results = []
        for record in records:
            result = await self.validate_record(record)
            validation_results.append(result)
        return validation_results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        if stats["total_records"] > 0:
            stats["validation_rate"] = stats["valid_records"] / stats["total_records"]
            stats["error_rate"] = stats["invalid_records"] / stats["total_records"]
        return stats


class EnterpriseDataEnricher:
    """Enterprise data enrichment engine"""
    
    def __init__(self):
        self.enrichment_rules: List[EnrichmentRule] = []
        self.lookup_cache: Dict[str, Any] = {}
        self.enrichment_stats = {
            "total_records": 0,
            "enriched_records": 0,
            "enrichment_failures": 0
        }
    
    def add_enrichment_rule(self, rule: EnrichmentRule):
        """Add an enrichment rule"""
        self.enrichment_rules.append(rule)
        logger.info(f"Added enrichment rule: {rule.source_field} -> {rule.target_field}")
    
    async def enrich_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single record"""
        enriched_record = record.copy()
        self.enrichment_stats["total_records"] += 1
        
        for rule in self.enrichment_rules:
            try:
                source_value = record.get(rule.source_field)
                if source_value is None:
                    continue
                
                if rule.enrichment_type == "lookup":
                    # Lookup enrichment from cache or external source
                    lookup_key = f"{rule.target_field}:{source_value}"
                    if lookup_key in self.lookup_cache:
                        enriched_record[rule.target_field] = self.lookup_cache[lookup_key]
                    else:
                        # Simulate lookup
                        enriched_value = await self._perform_lookup(source_value, rule.parameters)
                        self.lookup_cache[lookup_key] = enriched_value
                        enriched_record[rule.target_field] = enriched_value
                
                elif rule.enrichment_type == "calculation":
                    # Perform calculation-based enrichment
                    formula = rule.parameters.get("formula")
                    if formula:
                        enriched_record[rule.target_field] = await self._perform_calculation(record, formula)
                
                elif rule.enrichment_type == "external_api":
                    # External API enrichment
                    api_config = rule.parameters.get("api_config", {})
                    enriched_value = await self._call_external_api(source_value, api_config)
                    enriched_record[rule.target_field] = enriched_value
                
                elif rule.enrichment_type == "ml_inference":
                    # ML model inference
                    model_config = rule.parameters.get("model_config", {})
                    enriched_value = await self._ml_inference(record, model_config)
                    enriched_record[rule.target_field] = enriched_value
                
                self.enrichment_stats["enriched_records"] += 1
                
            except Exception as e:
                logger.error(f"Enrichment error for rule {rule.source_field}: {e}")
                self.enrichment_stats["enrichment_failures"] += 1
        
        return enriched_record
    
    async def _perform_lookup(self, value: Any, parameters: Dict[str, Any]) -> Any:
        """Perform lookup enrichment"""
        # Simulate lookup table or database query
        lookup_table = parameters.get("lookup_table", {})
        return lookup_table.get(str(value), f"enriched_{value}")
    
    async def _perform_calculation(self, record: Dict[str, Any], formula: str) -> Any:
        """Perform calculation-based enrichment"""
        # Simple calculation engine - in production, use a proper expression evaluator
        try:
            # Replace field references in formula
            for field, value in record.items():
                if isinstance(value, (int, float)):
                    formula = formula.replace(f"{{{field}}}", str(value))
            
            # Evaluate simple mathematical expressions
            if all(c in "0123456789+-*/.() " for c in formula):
                return eval(formula)
            else:
                return f"calculated_from_{formula}"
        except Exception:
            return None
    
    async def _call_external_api(self, value: Any, api_config: Dict[str, Any]) -> Any:
        """Call external API for enrichment"""
        # Simulate external API call
        await asyncio.sleep(0.1)  # Simulate network delay
        return f"api_enriched_{value}"
    
    async def _ml_inference(self, record: Dict[str, Any], model_config: Dict[str, Any]) -> Any:
        """Perform ML model inference"""
        # Simulate ML model inference
        await asyncio.sleep(0.05)  # Simulate inference time
        return f"ml_predicted_{hash(str(record)) % 1000}"


class RealTimeDataStreamer:
    """Real-time data streaming engine"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.buffer: List[DataRecord] = []
        self.last_flush = datetime.utcnow()
        self.encryption_key = Fernet.generate_key() if config.encryption else None
        self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
        self.streaming_stats = {
            "records_streamed": 0,
            "bytes_streamed": 0,
            "flush_count": 0,
            "errors": 0
        }
    
    async def add_record(self, record: DataRecord):
        """Add a record to the streaming buffer"""
        self.buffer.append(record)
        
        # Check if we need to flush
        should_flush = (
            len(self.buffer) >= self.config.batch_size or
            (datetime.utcnow() - self.last_flush).total_seconds() >= self.config.flush_interval
        )
        
        if should_flush:
            await self.flush_buffer()
    
    async def flush_buffer(self):
        """Flush the current buffer"""
        if not self.buffer:
            return
        
        try:
            # Prepare batch for streaming
            batch_data = []
            for record in self.buffer:
                record_data = {
                    "source_id": record.source_id,
                    "record_id": record.record_id,
                    "data": record.data,
                    "timestamp": record.timestamp.isoformat(),
                    "metadata": record.metadata
                }
                batch_data.append(record_data)
            
            # Serialize and optionally compress/encrypt
            serialized_data = json.dumps(batch_data)
            
            if self.config.compression:
                import gzip
                serialized_data = gzip.compress(serialized_data.encode()).decode('latin1')
            
            if self.config.encryption and self.cipher:
                serialized_data = self.cipher.encrypt(serialized_data.encode()).decode('latin1')
            
            # Stream the data (simulate streaming to data lake/warehouse)
            await self._stream_to_destination(serialized_data)
            
            # Update statistics
            self.streaming_stats["records_streamed"] += len(self.buffer)
            self.streaming_stats["bytes_streamed"] += len(serialized_data)
            self.streaming_stats["flush_count"] += 1
            
            # Clear buffer
            self.buffer.clear()
            self.last_flush = datetime.utcnow()
            
            logger.info(f"Flushed {len(batch_data)} records to streaming destination")
            
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            self.streaming_stats["errors"] += 1
            
            # Implement retry logic
            await self._handle_flush_error(e)
    
    async def _stream_to_destination(self, data: str):
        """Stream data to the configured destination"""
        # Simulate streaming to various destinations
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # In production, this would stream to:
        # - Apache Kafka
        # - Amazon Kinesis
        # - Azure Event Hubs
        # - Google Cloud Pub/Sub
        # - Data lakes (S3, ADLS, GCS)
        # - Data warehouses (Snowflake, BigQuery, Redshift)
        
        logger.debug(f"Streamed {len(data)} bytes to destination")
    
    async def _handle_flush_error(self, error: Exception):
        """Handle flush errors with retry logic"""
        retry_policy = self.config.retry_policy
        max_retries = retry_policy.get("max_retries", 3)
        backoff_factor = retry_policy.get("backoff_factor", 2)
        max_delay = retry_policy.get("max_delay", 300)
        
        for attempt in range(max_retries):
            try:
                delay = min(backoff_factor ** attempt, max_delay)
                await asyncio.sleep(delay)
                await self.flush_buffer()
                return
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")
        
        logger.error(f"Failed to flush buffer after {max_retries} retries")


class SQLServerConnector(BaseDataConnector):
    """Enterprise SQL Server connector with real-time streaming"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.validator = EnterpriseDataValidator()
        self.enricher = EnterpriseDataEnricher()
        self.streamer = RealTimeDataStreamer(StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            batch_size=500,
            flush_interval=15
        ))
        self._setup_validation_rules()
        self._setup_enrichment_rules()
    
    def _setup_validation_rules(self):
        """Setup default validation rules for SQL Server data"""
        self.validator.add_validation_rule(ValidationRule(
            field_name="id",
            rule_type="required"
        ))
        self.validator.add_validation_rule(ValidationRule(
            field_name="created_date",
            rule_type="type",
            parameters={"type": str}
        ))
    
    def _setup_enrichment_rules(self):
        """Setup default enrichment rules"""
        self.enricher.add_enrichment_rule(EnrichmentRule(
            source_field="id",
            target_field="record_hash",
            enrichment_type="calculation",
            parameters={"formula": "hash({id})"}
        ))
    
    async def connect(self) -> bool:
        """Connect to SQL Server with enterprise security"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # SQL Server connection parameters
            server = self.config.connection_params.get('server')
            database = self.config.connection_params.get('database')
            username = self.config.connection_params.get('username')
            password = self.config.connection_params.get('password')
            driver = self.config.connection_params.get('driver', 'ODBC Driver 17 for SQL Server')
            encrypt = self.config.connection_params.get('encrypt', True)
            trust_cert = self.config.connection_params.get('trust_server_certificate', False)
            
            if not all([server, database, username, password]):
                raise ValueError("Missing required SQL Server connection parameters")
            
            # Simulate secure connection with encryption
            await asyncio.sleep(1.5)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to SQL Server: {server}/{database}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"SQL Server connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from SQL Server"""
        try:
            # Flush any remaining data
            await self.streamer.flush_buffer()
            
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from SQL Server")
            return True
        except Exception as e:
            logger.error(f"SQL Server disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test SQL Server connection"""
        try:
            await asyncio.sleep(0.5)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from SQL Server with validation and enrichment"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to SQL Server")
        
        try:
            table_name = query.get('table', 'dbo.Customers')
            columns = query.get('columns', ['*'])
            where_clause = query.get('where', '')
            limit = query.get('limit', 1000)
            
            # Simulate SQL Server query execution
            await asyncio.sleep(2.5)
            
            # Mock SQL Server data
            raw_records = []
            for i in range(min(limit, 200)):
                raw_record = {
                    'CustomerID': i + 1,
                    'CompanyName': f'SQL Server Company {i}',
                    'ContactName': f'Contact {i}',
                    'ContactTitle': 'Manager',
                    'Address': f'{100 + i} Main St',
                    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'][i % 4],
                    'Region': ['NY', 'CA', 'IL', 'TX'][i % 4],
                    'PostalCode': f'{10000 + i:05d}',
                    'Country': 'USA',
                    'Phone': f'(555) {100 + i:03d}-{1000 + i:04d}',
                    'Fax': f'(555) {200 + i:03d}-{2000 + i:04d}',
                    'created_date': datetime.utcnow().isoformat(),
                    'last_modified': datetime.utcnow().isoformat(),
                    'annual_revenue': 100000 + i * 5000,
                    'employee_count': 10 + i * 2
                }
                raw_records.append(raw_record)
            
            # Validate and enrich data
            processed_records = []
            for raw_record in raw_records:
                # Validate
                validation_result = await self.validator.validate_record(raw_record)
                if not validation_result["is_valid"]:
                    logger.warning(f"Invalid record: {validation_result['errors']}")
                    continue
                
                # Enrich
                enriched_record = await self.enricher.enrich_record(validation_result["transformed_record"])
                
                # Create DataRecord
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"SQLSERVER_{table_name}_{enriched_record['CustomerID']}",
                    data=enriched_record,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'table': table_name,
                        'sql_server': self.config.connection_params.get('server'),
                        'database': self.config.connection_params.get('database'),
                        'validation_passed': True,
                        'enrichment_applied': True
                    }
                )
                
                processed_records.append(record)
                
                # Stream in real-time
                await self.streamer.add_record(record)
            
            logger.info(f"Fetched and processed {len(processed_records)} records from SQL Server table {table_name}")
            return processed_records
            
        except Exception as e:
            logger.error(f"SQL Server data fetch failed: {e}")
            raise
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get SQL Server database schema"""
        return {
            'database': self.config.connection_params.get('database'),
            'tables': {
                'dbo.Customers': {
                    'description': 'Customer Master Table',
                    'columns': {
                        'CustomerID': {'type': 'int', 'nullable': False, 'primary_key': True},
                        'CompanyName': {'type': 'nvarchar', 'length': 40, 'nullable': False},
                        'ContactName': {'type': 'nvarchar', 'length': 30, 'nullable': True},
                        'ContactTitle': {'type': 'nvarchar', 'length': 30, 'nullable': True},
                        'Address': {'type': 'nvarchar', 'length': 60, 'nullable': True},
                        'City': {'type': 'nvarchar', 'length': 15, 'nullable': True},
                        'Region': {'type': 'nvarchar', 'length': 15, 'nullable': True},
                        'PostalCode': {'type': 'nvarchar', 'length': 10, 'nullable': True},
                        'Country': {'type': 'nvarchar', 'length': 15, 'nullable': True},
                        'Phone': {'type': 'nvarchar', 'length': 24, 'nullable': True},
                        'Fax': {'type': 'nvarchar', 'length': 24, 'nullable': True}
                    }
                },
                'dbo.Orders': {
                    'description': 'Order Transaction Table',
                    'columns': {
                        'OrderID': {'type': 'int', 'nullable': False, 'primary_key': True},
                        'CustomerID': {'type': 'int', 'nullable': True, 'foreign_key': 'dbo.Customers.CustomerID'},
                        'EmployeeID': {'type': 'int', 'nullable': True},
                        'OrderDate': {'type': 'datetime', 'nullable': True},
                        'RequiredDate': {'type': 'datetime', 'nullable': True},
                        'ShippedDate': {'type': 'datetime', 'nullable': True},
                        'Freight': {'type': 'money', 'nullable': True}
                    }
                }
            }
        }


class SnowflakeConnector(BaseDataConnector):
    """Enterprise Snowflake data warehouse connector"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.validator = EnterpriseDataValidator(DataValidationLevel.ENTERPRISE)
        self.enricher = EnterpriseDataEnricher()
        self.streamer = RealTimeDataStreamer(StreamingConfig(
            mode=StreamingMode.MICRO_BATCH,
            batch_size=1000,
            flush_interval=30,
            compression=True,
            encryption=True
        ))
    
    async def connect(self) -> bool:
        """Connect to Snowflake with enterprise authentication"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Snowflake connection parameters
            account = self.config.connection_params.get('account')
            user = self.config.connection_params.get('user')
            password = self.config.connection_params.get('password')
            warehouse = self.config.connection_params.get('warehouse')
            database = self.config.connection_params.get('database')
            schema = self.config.connection_params.get('schema', 'PUBLIC')
            role = self.config.connection_params.get('role')
            
            # Support for key-pair authentication
            private_key = self.config.connection_params.get('private_key')
            private_key_passphrase = self.config.connection_params.get('private_key_passphrase')
            
            if not account:
                raise ValueError("Snowflake account is required")
            
            if not user:
                raise ValueError("Snowflake user is required")
            
            if not (password or private_key):
                raise ValueError("Either password or private key is required")
            
            # Simulate Snowflake connection with JWT authentication
            await asyncio.sleep(2)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Snowflake: {account}/{database}/{schema}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Snowflake connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Snowflake"""
        try:
            await self.streamer.flush_buffer()
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Snowflake")
            return True
        except Exception as e:
            logger.error(f"Snowflake disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Snowflake connection"""
        try:
            await asyncio.sleep(0.8)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from Snowflake with advanced analytics"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Snowflake")
        
        try:
            sql_query = query.get('sql', 'SELECT * FROM SAMPLE_DATA LIMIT 1000')
            warehouse = query.get('warehouse', self.config.connection_params.get('warehouse'))
            
            # Simulate Snowflake query execution with warehouse scaling
            await asyncio.sleep(3)
            
            # Mock Snowflake analytical data
            raw_records = []
            for i in range(min(query.get('limit', 1000), 500)):
                raw_record = {
                    'TRANSACTION_ID': f'TXN_{i:010d}',
                    'CUSTOMER_ID': f'CUST_{(i % 1000):06d}',
                    'PRODUCT_ID': f'PROD_{(i % 100):04d}',
                    'TRANSACTION_DATE': (datetime.utcnow() - timedelta(days=i % 365)).isoformat(),
                    'AMOUNT': round(10.0 + i * 2.5 + (i % 50) * 10, 2),
                    'QUANTITY': 1 + (i % 10),
                    'DISCOUNT': round((i % 20) * 0.05, 2),
                    'TAX_AMOUNT': round((10.0 + i * 2.5) * 0.08, 2),
                    'TOTAL_AMOUNT': round((10.0 + i * 2.5) * 1.08, 2),
                    'PAYMENT_METHOD': ['CREDIT_CARD', 'DEBIT_CARD', 'CASH', 'CHECK'][i % 4],
                    'STORE_ID': f'STORE_{(i % 50):03d}',
                    'REGION': ['NORTH', 'SOUTH', 'EAST', 'WEST'][i % 4],
                    'CHANNEL': ['ONLINE', 'RETAIL', 'MOBILE', 'PHONE'][i % 4],
                    'CREATED_TIMESTAMP': datetime.utcnow().isoformat(),
                    'UPDATED_TIMESTAMP': datetime.utcnow().isoformat()
                }
                raw_records.append(raw_record)
            
            # Process with enterprise-grade validation and enrichment
            processed_records = []
            for raw_record in raw_records:
                # Advanced validation
                validation_result = await self.validator.validate_record(raw_record)
                if not validation_result["is_valid"]:
                    continue
                
                # Business intelligence enrichment
                enriched_record = await self.enricher.enrich_record(validation_result["transformed_record"])
                
                # Add analytical metadata
                enriched_record.update({
                    'PROFIT_MARGIN': round(enriched_record['AMOUNT'] * 0.25, 2),
                    'CUSTOMER_SEGMENT': self._determine_customer_segment(enriched_record),
                    'SEASONAL_FACTOR': self._calculate_seasonal_factor(enriched_record['TRANSACTION_DATE']),
                    'RISK_SCORE': self._calculate_risk_score(enriched_record)
                })
                
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"SNOWFLAKE_{enriched_record['TRANSACTION_ID']}",
                    data=enriched_record,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'snowflake_account': self.config.connection_params.get('account'),
                        'warehouse': warehouse,
                        'query_id': f'query_{hash(sql_query) % 1000000:06d}',
                        'data_quality_score': 0.95 + (i % 10) * 0.005,
                        'processing_time_ms': 150 + (i % 100)
                    }
                )
                
                processed_records.append(record)
                await self.streamer.add_record(record)
            
            logger.info(f"Fetched {len(processed_records)} records from Snowflake")
            return processed_records
            
        except Exception as e:
            logger.error(f"Snowflake data fetch failed: {e}")
            raise
    
    def _determine_customer_segment(self, record: Dict[str, Any]) -> str:
        """Determine customer segment based on transaction data"""
        amount = record.get('AMOUNT', 0)
        if amount > 1000:
            return 'PREMIUM'
        elif amount > 500:
            return 'STANDARD'
        else:
            return 'BASIC'
    
    def _calculate_seasonal_factor(self, transaction_date: str) -> float:
        """Calculate seasonal factor for the transaction"""
        try:
            date_obj = datetime.fromisoformat(transaction_date.replace('Z', '+00:00'))
            month = date_obj.month
            # Simple seasonal factors
            seasonal_factors = {
                12: 1.3, 1: 1.1, 2: 0.9,  # Winter
                3: 1.0, 4: 1.1, 5: 1.2,   # Spring
                6: 1.3, 7: 1.4, 8: 1.3,   # Summer
                9: 1.1, 10: 1.2, 11: 1.4  # Fall
            }
            return seasonal_factors.get(month, 1.0)
        except Exception:
            return 1.0
    
    def _calculate_risk_score(self, record: Dict[str, Any]) -> float:
        """Calculate transaction risk score"""
        amount = record.get('AMOUNT', 0)
        payment_method = record.get('PAYMENT_METHOD', '')
        
        # Simple risk scoring
        risk_score = 0.1  # Base risk
        
        if amount > 5000:
            risk_score += 0.3
        elif amount > 1000:
            risk_score += 0.1
        
        if payment_method == 'CASH':
            risk_score += 0.2
        elif payment_method == 'CHECK':
            risk_score += 0.15
        
        return min(risk_score, 1.0)
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Snowflake database schema"""
        return {
            'account': self.config.connection_params.get('account'),
            'database': self.config.connection_params.get('database'),
            'schema': self.config.connection_params.get('schema'),
            'tables': {
                'TRANSACTIONS': {
                    'description': 'Transaction Fact Table',
                    'columns': {
                        'TRANSACTION_ID': {'type': 'VARCHAR', 'nullable': False, 'primary_key': True},
                        'CUSTOMER_ID': {'type': 'VARCHAR', 'nullable': False},
                        'PRODUCT_ID': {'type': 'VARCHAR', 'nullable': False},
                        'TRANSACTION_DATE': {'type': 'TIMESTAMP_NTZ', 'nullable': False},
                        'AMOUNT': {'type': 'NUMBER', 'precision': 10, 'scale': 2},
                        'QUANTITY': {'type': 'NUMBER', 'precision': 5, 'scale': 0},
                        'TOTAL_AMOUNT': {'type': 'NUMBER', 'precision': 10, 'scale': 2}
                    },
                    'clustering_keys': ['TRANSACTION_DATE', 'REGION'],
                    'partitioning': 'MONTHLY'
                },
                'CUSTOMERS': {
                    'description': 'Customer Dimension Table',
                    'columns': {
                        'CUSTOMER_ID': {'type': 'VARCHAR', 'nullable': False, 'primary_key': True},
                        'CUSTOMER_NAME': {'type': 'VARCHAR', 'nullable': False},
                        'EMAIL': {'type': 'VARCHAR', 'nullable': True},
                        'REGISTRATION_DATE': {'type': 'DATE', 'nullable': False},
                        'CUSTOMER_SEGMENT': {'type': 'VARCHAR', 'nullable': True}
                    }
                }
            }
        }


class DatabricksConnector(BaseDataConnector):
    """Enterprise Databricks lakehouse connector"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.validator = EnterpriseDataValidator(DataValidationLevel.ENTERPRISE)
        self.enricher = EnterpriseDataEnricher()
        self.streamer = RealTimeDataStreamer(StreamingConfig(
            mode=StreamingMode.CHANGE_DATA_CAPTURE,
            batch_size=2000,
            flush_interval=60,
            compression=True,
            encryption=True
        ))
    
    async def connect(self) -> bool:
        """Connect to Databricks with token authentication"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Databricks connection parameters
            server_hostname = self.config.connection_params.get('server_hostname')
            http_path = self.config.connection_params.get('http_path')
            access_token = self.config.connection_params.get('access_token')
            catalog = self.config.connection_params.get('catalog', 'main')
            schema = self.config.connection_params.get('schema', 'default')
            
            if not all([server_hostname, http_path, access_token]):
                raise ValueError("Missing required Databricks connection parameters")
            
            # Simulate Databricks connection
            await asyncio.sleep(2.5)
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Databricks: {server_hostname}/{catalog}/{schema}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Databricks connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Databricks"""
        try:
            await self.streamer.flush_buffer()
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Databricks")
            return True
        except Exception as e:
            logger.error(f"Databricks disconnect failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Databricks connection"""
        try:
            await asyncio.sleep(1)
            return self.status == ConnectionStatus.CONNECTED
        except Exception:
            return False
    
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from Databricks Delta Lake"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Databricks")
        
        try:
            sql_query = query.get('sql', 'SELECT * FROM delta_table LIMIT 1000')
            catalog = query.get('catalog', self.config.connection_params.get('catalog'))
            schema = query.get('schema', self.config.connection_params.get('schema'))
            
            # Simulate Databricks Spark SQL execution
            await asyncio.sleep(4)
            
            # Mock Databricks Delta Lake data with ML features
            raw_records = []
            for i in range(min(query.get('limit', 1000), 300)):
                raw_record = {
                    'event_id': f'evt_{i:012d}',
                    'user_id': f'user_{(i % 10000):08d}',
                    'session_id': f'sess_{(i % 1000):06d}',
                    'event_timestamp': (datetime.utcnow() - timedelta(minutes=i % 1440)).isoformat(),
                    'event_type': ['page_view', 'click', 'purchase', 'signup', 'logout'][i % 5],
                    'page_url': f'/page/{(i % 100):03d}',
                    'referrer': ['google.com', 'facebook.com', 'direct', 'email'][i % 4],
                    'device_type': ['desktop', 'mobile', 'tablet'][i % 3],
                    'browser': ['chrome', 'firefox', 'safari', 'edge'][i % 4],
                    'os': ['windows', 'macos', 'linux', 'ios', 'android'][i % 5],
                    'country': ['US', 'CA', 'UK', 'DE', 'FR', 'JP', 'AU'][i % 7],
                    'city': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'][i % 5],
                    'ip_address': f'192.168.{(i % 255):03d}.{((i * 7) % 255):03d}',
                    'user_agent': f'Mozilla/5.0 (compatible; Bot/{i % 100})',
                    'conversion_value': round((i % 50) * 12.5, 2) if i % 10 == 0 else 0,
                    'ab_test_variant': ['A', 'B', 'C'][i % 3],
                    'feature_flags': json.dumps({
                        'new_ui': i % 2 == 0,
                        'premium_features': i % 5 == 0,
                        'beta_access': i % 10 == 0
                    }),
                    'ml_prediction_score': round(0.1 + (i % 90) * 0.01, 3),
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat(),
                    '_delta_version': i % 1000,
                    '_commit_timestamp': datetime.utcnow().isoformat()
                }
                raw_records.append(raw_record)
            
            # Advanced processing with ML enrichment
            processed_records = []
            for raw_record in raw_records:
                # Validate with strict enterprise rules
                validation_result = await self.validator.validate_record(raw_record)
                if not validation_result["is_valid"]:
                    continue
                
                # ML-powered enrichment
                enriched_record = await self.enricher.enrich_record(validation_result["transformed_record"])
                
                # Add advanced analytics features
                enriched_record.update({
                    'user_lifetime_value': self._calculate_ltv(enriched_record),
                    'churn_probability': self._predict_churn(enriched_record),
                    'next_best_action': self._recommend_action(enriched_record),
                    'anomaly_score': self._detect_anomaly(enriched_record),
                    'engagement_score': self._calculate_engagement(enriched_record)
                })
                
                record = DataRecord(
                    source_id=self.config.source_id,
                    record_id=f"DATABRICKS_{enriched_record['event_id']}",
                    data=enriched_record,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'databricks_cluster': self.config.connection_params.get('server_hostname'),
                        'catalog': catalog,
                        'schema': schema,
                        'delta_version': enriched_record.get('_delta_version'),
                        'spark_version': '3.4.0',
                        'processing_engine': 'Delta Lake',
                        'ml_model_version': '2.1.0'
                    }
                )
                
                processed_records.append(record)
                await self.streamer.add_record(record)
            
            logger.info(f"Fetched {len(processed_records)} records from Databricks Delta Lake")
            return processed_records
            
        except Exception as e:
            logger.error(f"Databricks data fetch failed: {e}")
            raise
    
    def _calculate_ltv(self, record: Dict[str, Any]) -> float:
        """Calculate user lifetime value using ML model"""
        conversion_value = record.get('conversion_value', 0)
        ml_score = record.get('ml_prediction_score', 0.5)
        return round(conversion_value * ml_score * 12, 2)  # Annualized LTV
    
    def _predict_churn(self, record: Dict[str, Any]) -> float:
        """Predict churn probability using behavioral features"""
        event_type = record.get('event_type', '')
        device_type = record.get('device_type', '')
        
        # Simple churn model
        churn_prob = 0.1  # Base churn rate
        
        if event_type == 'logout':
            churn_prob += 0.2
        elif event_type == 'purchase':
            churn_prob -= 0.15
        
        if device_type == 'mobile':
            churn_prob -= 0.05
        
        return max(0, min(1, churn_prob))
    
    def _recommend_action(self, record: Dict[str, Any]) -> str:
        """Recommend next best action for user"""
        event_type = record.get('event_type', '')
        conversion_value = record.get('conversion_value', 0)
        
        if conversion_value > 0:
            return 'upsell_premium'
        elif event_type == 'page_view':
            return 'show_product_recommendation'
        elif event_type == 'click':
            return 'offer_discount'
        else:
            return 'send_engagement_email'
    
    def _detect_anomaly(self, record: Dict[str, Any]) -> float:
        """Detect anomalous behavior"""
        ip_parts = record.get('ip_address', '0.0.0.0').split('.')
        try:
            ip_sum = sum(int(part) for part in ip_parts)
            # Simple anomaly detection based on IP pattern
            return min(1.0, abs(ip_sum - 500) / 1000)
        except Exception:
            return 0.5
    
    def _calculate_engagement(self, record: Dict[str, Any]) -> float:
        """Calculate user engagement score"""
        event_type = record.get('event_type', '')
        ml_score = record.get('ml_prediction_score', 0.5)
        
        engagement_weights = {
            'page_view': 0.1,
            'click': 0.3,
            'purchase': 1.0,
            'signup': 0.8,
            'logout': -0.2
        }
        
        base_engagement = engagement_weights.get(event_type, 0.1)
        return max(0, min(1, base_engagement * ml_score))
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Databricks catalog schema"""
        return {
            'catalog': self.config.connection_params.get('catalog'),
            'schema': self.config.connection_params.get('schema'),
            'tables': {
                'user_events': {
                    'description': 'User Event Stream Table (Delta Lake)',
                    'table_format': 'DELTA',
                    'columns': {
                        'event_id': {'type': 'STRING', 'nullable': False, 'primary_key': True},
                        'user_id': {'type': 'STRING', 'nullable': False},
                        'event_timestamp': {'type': 'TIMESTAMP', 'nullable': False},
                        'event_type': {'type': 'STRING', 'nullable': False},
                        'conversion_value': {'type': 'DECIMAL', 'precision': 10, 'scale': 2},
                        'ml_prediction_score': {'type': 'DOUBLE', 'nullable': True},
                        '_delta_version': {'type': 'BIGINT', 'nullable': False}
                    },
                    'partitioned_by': ['event_timestamp'],
                    'z_order_by': ['user_id', 'event_type'],
                    'delta_features': {
                        'change_data_feed': True,
                        'time_travel': True,
                        'vacuum_retention': '7 DAYS'
                    }
                },
                'ml_features': {
                    'description': 'ML Feature Store Table',
                    'table_format': 'DELTA',
                    'columns': {
                        'user_id': {'type': 'STRING', 'nullable': False, 'primary_key': True},
                        'feature_timestamp': {'type': 'TIMESTAMP', 'nullable': False},
                        'user_lifetime_value': {'type': 'DOUBLE', 'nullable': True},
                        'churn_probability': {'type': 'DOUBLE', 'nullable': True},
                        'engagement_score': {'type': 'DOUBLE', 'nullable': True}
                    }
                }
            }
        }