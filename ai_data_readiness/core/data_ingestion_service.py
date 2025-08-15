"""
Data Ingestion Service for AI Data Readiness Platform

This module provides comprehensive data ingestion capabilities with multi-format support,
automatic schema detection, and validation for both batch and streaming data processing.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.orm import Session

from .config import get_settings
from .exceptions import DataIngestionError, SchemaValidationError
from ..models.base_models import DatasetMetadata, Schema, ValidationResult
from ..models.database import get_db_session


class DataFormat(Enum):
    """Supported data formats for ingestion"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    AVRO = "avro"
    EXCEL = "excel"
    XML = "xml"


class ProcessingMode(Enum):
    """Data processing modes"""
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class SourceConfig:
    """Configuration for data source"""
    source_type: str
    connection_params: Dict[str, Any]
    data_format: DataFormat
    sampling_config: Optional[Dict[str, Any]] = None
    validation_rules: Optional[List[Dict[str, Any]]] = None
    batch_size: int = 1000
    timeout: int = 300


@dataclass
class StreamConfig:
    """Configuration for streaming data"""
    stream_name: str
    source_config: SourceConfig
    buffer_size: int = 10000
    checkpoint_interval: int = 60
    error_handling: str = "continue"  # continue, stop, retry


class StreamHandler:
    """Handler for streaming data processing"""
    
    def __init__(self, config: StreamConfig, ingestion_service: 'DataIngestionService'):
        self.config = config
        self.ingestion_service = ingestion_service
        self.is_running = False
        self.buffer = []
        self.last_checkpoint = datetime.utcnow()
        
    async def start(self):
        """Start streaming data processing"""
        self.is_running = True
        logging.info(f"Starting stream handler for {self.config.stream_name}")
        
    async def stop(self):
        """Stop streaming data processing"""
        self.is_running = False
        logging.info(f"Stopping stream handler for {self.config.stream_name}")
        
    async def process_batch(self, data_batch: List[Dict[str, Any]]) -> bool:
        """Process a batch of streaming data"""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(data_batch)
            
            # Apply validation rules
            validation_result = await self.ingestion_service._validate_data(
                df, self.config.source_config.validation_rules or []
            )
            
            if not validation_result.is_valid:
                logging.warning(f"Validation failed for stream batch: {validation_result.errors}")
                if self.config.error_handling == "stop":
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error processing stream batch: {str(e)}")
            if self.config.error_handling == "stop":
                return False
            return True


class DataIngestionService:
    """
    Comprehensive data ingestion service with multi-format support,
    automatic schema detection, and validation capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {
            DataFormat.CSV: self._read_csv,
            DataFormat.JSON: self._read_json,
            DataFormat.PARQUET: self._read_parquet,
            DataFormat.EXCEL: self._read_excel,
        }
        
    async def ingest_batch_data(self, source_config: SourceConfig) -> DatasetMetadata:
        """
        Ingest batch data from various sources with automatic schema detection.
        
        Args:
            source_config: Configuration for the data source
            
        Returns:
            DatasetMetadata: Metadata about the ingested dataset
            
        Raises:
            DataIngestionError: If ingestion fails
        """
        try:
            self.logger.info(f"Starting batch ingestion for {source_config.source_type}")
            
            # Read data based on format
            df = await self._read_data(source_config)
            
            # Apply sampling if configured
            if source_config.sampling_config:
                df = self._apply_sampling(df, source_config.sampling_config)
            
            # Detect schema automatically
            schema = await self._detect_schema(df)
            
            # Validate data against rules
            validation_result = await self._validate_data(df, source_config.validation_rules or [])
            
            # Create dataset metadata
            metadata = DatasetMetadata(
                dataset_id=self._generate_dataset_id(),
                name=f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                description=f"Ingested from {source_config.source_type}",
                schema=schema,
                row_count=len(df),
                column_count=len(df.columns),
                file_size=df.memory_usage(deep=True).sum(),
                created_at=datetime.utcnow(),
                source_config=asdict(source_config),
                validation_result=validation_result,
                processing_mode=ProcessingMode.BATCH.value
            )
            
            # Store dataset and metadata
            await self._store_dataset(df, metadata)
            
            self.logger.info(f"Successfully ingested dataset {metadata.dataset_id}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Batch ingestion failed: {str(e)}")
            raise DataIngestionError(f"Failed to ingest batch data: {str(e)}")
    
    async def ingest_streaming_data(self, stream_config: StreamConfig) -> StreamHandler:
        """
        Set up streaming data ingestion with continuous processing.
        
        Args:
            stream_config: Configuration for the streaming data source
            
        Returns:
            StreamHandler: Handler for managing the stream
            
        Raises:
            DataIngestionError: If stream setup fails
        """
        try:
            self.logger.info(f"Setting up streaming ingestion for {stream_config.stream_name}")
            
            # Create stream handler
            handler = StreamHandler(stream_config, self)
            
            # Start the stream processing
            await handler.start()
            
            return handler
            
        except Exception as e:
            self.logger.error(f"Streaming ingestion setup failed: {str(e)}")
            raise DataIngestionError(f"Failed to setup streaming ingestion: {str(e)}")
    
    async def validate_schema(self, dataset_id: str, expected_schema: Schema) -> ValidationResult:
        """
        Validate dataset schema against expected schema.
        
        Args:
            dataset_id: ID of the dataset to validate
            expected_schema: Expected schema definition
            
        Returns:
            ValidationResult: Result of schema validation
        """
        try:
            # Load dataset metadata
            metadata = await self._load_dataset_metadata(dataset_id)
            if not metadata:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Dataset {dataset_id} not found"],
                    warnings=[]
                )
            
            # Compare schemas
            actual_schema = metadata.schema
            errors = []
            warnings = []
            
            # Check column count
            if len(actual_schema.columns) != len(expected_schema.columns):
                errors.append(f"Column count mismatch: expected {len(expected_schema.columns)}, got {len(actual_schema.columns)}")
            
            # Check individual columns
            expected_cols = {col.name: col for col in expected_schema.columns}
            actual_cols = {col.name: col for col in actual_schema.columns}
            
            for col_name, expected_col in expected_cols.items():
                if col_name not in actual_cols:
                    errors.append(f"Missing column: {col_name}")
                else:
                    actual_col = actual_cols[col_name]
                    if actual_col.data_type != expected_col.data_type:
                        warnings.append(f"Type mismatch for {col_name}: expected {expected_col.data_type}, got {actual_col.data_type}")
            
            # Check for extra columns
            for col_name in actual_cols:
                if col_name not in expected_cols:
                    warnings.append(f"Unexpected column: {col_name}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            raise SchemaValidationError(f"Schema validation failed: {str(e)}")
    
    async def extract_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Extract comprehensive metadata for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            DatasetMetadata: Extracted metadata or None if not found
        """
        try:
            return await self._load_dataset_metadata(dataset_id)
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}")
            return None
    
    # Private helper methods
    
    async def _read_data(self, source_config: SourceConfig) -> pd.DataFrame:
        """Read data based on the configured format"""
        if source_config.data_format not in self.supported_formats:
            raise DataIngestionError(f"Unsupported data format: {source_config.data_format}")
        
        reader_func = self.supported_formats[source_config.data_format]
        return await reader_func(source_config)
    
    async def _read_csv(self, source_config: SourceConfig) -> pd.DataFrame:
        """Read CSV data"""
        file_path = source_config.connection_params.get('file_path')
        if not file_path:
            raise DataIngestionError("file_path required for CSV format")
        
        return pd.read_csv(file_path, **source_config.connection_params.get('read_options', {}))
    
    async def _read_json(self, source_config: SourceConfig) -> pd.DataFrame:
        """Read JSON data"""
        file_path = source_config.connection_params.get('file_path')
        if not file_path:
            raise DataIngestionError("file_path required for JSON format")
        
        return pd.read_json(file_path, **source_config.connection_params.get('read_options', {}))
    
    async def _read_parquet(self, source_config: SourceConfig) -> pd.DataFrame:
        """Read Parquet data"""
        file_path = source_config.connection_params.get('file_path')
        if not file_path:
            raise DataIngestionError("file_path required for Parquet format")
        
        return pd.read_parquet(file_path, **source_config.connection_params.get('read_options', {}))
    
    async def _read_excel(self, source_config: SourceConfig) -> pd.DataFrame:
        """Read Excel data"""
        file_path = source_config.connection_params.get('file_path')
        if not file_path:
            raise DataIngestionError("file_path required for Excel format")
        
        return pd.read_excel(file_path, **source_config.connection_params.get('read_options', {}))
    
    def _apply_sampling(self, df: pd.DataFrame, sampling_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply sampling to the dataset"""
        method = sampling_config.get('method', 'random')
        sample_size = sampling_config.get('sample_size', 1000)
        
        if method == 'random':
            if sample_size < 1:
                # Treat as fraction
                return df.sample(frac=sample_size)
            else:
                # Treat as absolute count
                return df.sample(n=min(sample_size, len(df)))
        elif method == 'head':
            return df.head(sample_size)
        elif method == 'tail':
            return df.tail(sample_size)
        else:
            self.logger.warning(f"Unknown sampling method: {method}, using random")
            return df.sample(n=min(sample_size, len(df)))
    
    async def _detect_schema(self, df: pd.DataFrame) -> Schema:
        """Automatically detect schema from DataFrame"""
        from ..models.base_models import ColumnSchema
        
        columns = []
        for col_name in df.columns:
            col_data = df[col_name]
            
            # Detect data type
            if pd.api.types.is_numeric_dtype(col_data):
                if pd.api.types.is_integer_dtype(col_data):
                    data_type = "integer"
                else:
                    data_type = "float"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                data_type = "datetime"
            elif pd.api.types.is_bool_dtype(col_data):
                data_type = "boolean"
            else:
                data_type = "string"
            
            # Calculate statistics
            null_count = col_data.isnull().sum()
            unique_count = col_data.nunique()
            
            column_schema = ColumnSchema(
                name=col_name,
                data_type=data_type,
                nullable=null_count > 0,
                unique_count=unique_count,
                null_count=int(null_count),
                min_value=col_data.min() if pd.api.types.is_numeric_dtype(col_data) else None,
                max_value=col_data.max() if pd.api.types.is_numeric_dtype(col_data) else None
            )
            columns.append(column_schema)
        
        return Schema(
            columns=columns,
            primary_key=None,  # Auto-detection of primary key would require more analysis
            foreign_keys=[],
            indexes=[]
        )
    
    async def _validate_data(self, df: pd.DataFrame, validation_rules: List[Dict[str, Any]]) -> ValidationResult:
        """Validate data against specified rules"""
        errors = []
        warnings = []
        
        for rule in validation_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'not_null':
                columns = rule.get('columns', [])
                for col in columns:
                    if col in df.columns:
                        null_count = df[col].isnull().sum()
                        if null_count > 0:
                            errors.append(f"Column {col} has {null_count} null values")
            
            elif rule_type == 'unique':
                columns = rule.get('columns', [])
                for col in columns:
                    if col in df.columns:
                        duplicate_count = df[col].duplicated().sum()
                        if duplicate_count > 0:
                            errors.append(f"Column {col} has {duplicate_count} duplicate values")
            
            elif rule_type == 'range':
                column = rule.get('column')
                min_val = rule.get('min')
                max_val = rule.get('max')
                
                if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                    if min_val is not None:
                        below_min = (df[column] < min_val).sum()
                        if below_min > 0:
                            errors.append(f"Column {column} has {below_min} values below minimum {min_val}")
                    
                    if max_val is not None:
                        above_max = (df[column] > max_val).sum()
                        if above_max > 0:
                            errors.append(f"Column {column} has {above_max} values above maximum {max_val}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _generate_dataset_id(self) -> str:
        """Generate unique dataset ID"""
        import uuid
        return f"dataset_{uuid.uuid4().hex[:8]}"
    
    async def _store_dataset(self, df: pd.DataFrame, metadata: DatasetMetadata):
        """Store dataset and its metadata"""
        # In a real implementation, this would store to a database or file system
        # For now, we'll just log the operation
        self.logger.info(f"Storing dataset {metadata.dataset_id} with {len(df)} rows")
        
        # Store metadata in database
        with get_db_session() as session:
            # This would involve actual database operations
            pass
    
    async def _load_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Load dataset metadata from storage"""
        # In a real implementation, this would load from database
        # For now, return None
        return None