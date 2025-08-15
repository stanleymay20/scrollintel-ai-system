"""
Data Normalizer for Advanced Analytics Dashboard System

Provides schema mapping, data transformation, and normalization capabilities
for multi-source data integration with quality assurance.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DataType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"


class TransformationType(Enum):
    DIRECT_MAPPING = "direct_mapping"
    VALUE_MAPPING = "value_mapping"
    CALCULATION = "calculation"
    AGGREGATION = "aggregation"
    CONCATENATION = "concatenation"


@dataclass
class SchemaField:
    """Represents a field in a data schema"""
    name: str
    data_type: DataType
    required: bool = True
    default_value: Any = None
    validation_rules: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class SchemaMapping:
    """Defines mapping between source and target schemas"""
    source_field: str
    target_field: str
    transformation_type: TransformationType
    transformation_config: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class DataSchema:
    """Represents a complete data schema"""
    name: str
    version: str
    fields: List[SchemaField]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizationResult:
    """Result of data normalization process"""
    success: bool
    normalized_data: pd.DataFrame
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    transformation_log: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


class DataNormalizer:
    """
    Advanced data normalizer with schema mapping and transformation capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.schemas: Dict[str, DataSchema] = {}
        self.mappings: Dict[str, List[SchemaMapping]] = {}
        self.transformation_functions = self._initialize_transformations()
        
    def register_schema(self, schema: DataSchema) -> bool:
        """Register a data schema for normalization"""
        try:
            self.schemas[schema.name] = schema
            logger.info(f"Registered schema: {schema.name} v{schema.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to register schema {schema.name}: {str(e)}")
            return False
    
    def register_mapping(self, source_schema: str, target_schema: str, 
                        mappings: List[SchemaMapping]) -> bool:
        """Register field mappings between schemas"""
        try:
            mapping_key = f"{source_schema}->{target_schema}"
            self.mappings[mapping_key] = mappings
            logger.info(f"Registered mapping: {mapping_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to register mapping {mapping_key}: {str(e)}")
            return False
    
    def normalize_data(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], 
                      source_schema: str, target_schema: str) -> NormalizationResult:
        """
        Normalize data from source schema to target schema
        """
        try:
            # Convert input data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Get schemas and mappings
            source_schema_obj = self.schemas.get(source_schema)
            target_schema_obj = self.schemas.get(target_schema)
            mapping_key = f"{source_schema}->{target_schema}"
            mappings = self.mappings.get(mapping_key, [])
            
            if not source_schema_obj or not target_schema_obj:
                return NormalizationResult(
                    success=False,
                    normalized_data=pd.DataFrame(),
                    errors=[f"Schema not found: {source_schema} or {target_schema}"]
                )
            
            # Initialize result tracking
            errors = []
            warnings = []
            transformation_log = []
            
            # Create target DataFrame structure
            target_columns = [field.name for field in target_schema_obj.fields]
            normalized_df = pd.DataFrame(columns=target_columns)
            
            # Apply transformations
            for mapping in mappings:
                try:
                    transformed_data = self._apply_transformation(
                        df, mapping, transformation_log
                    )
                    
                    if transformed_data is not None:
                        normalized_df[mapping.target_field] = transformed_data
                    
                except Exception as e:
                    error_msg = f"Transformation failed for {mapping.source_field}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Apply default values for missing fields
            for field in target_schema_obj.fields:
                if field.name not in normalized_df.columns:
                    if field.default_value is not None:
                        normalized_df[field.name] = field.default_value
                    elif not field.required:
                        normalized_df[field.name] = None
                    else:
                        warnings.append(f"Required field {field.name} has no mapping or default")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(df, normalized_df)
            
            return NormalizationResult(
                success=len(errors) == 0,
                normalized_data=normalized_df,
                errors=errors,
                warnings=warnings,
                transformation_log=transformation_log,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Data normalization failed: {str(e)}")
            return NormalizationResult(
                success=False,
                normalized_data=pd.DataFrame(),
                errors=[f"Normalization failed: {str(e)}"]
            )
    
    def _apply_transformation(self, df: pd.DataFrame, mapping: SchemaMapping, 
                            transformation_log: List[Dict[str, Any]]) -> Optional[pd.Series]:
        """Apply a single field transformation"""
        try:
            transformation_func = self.transformation_functions.get(mapping.transformation_type)
            if not transformation_func:
                raise ValueError(f"Unknown transformation type: {mapping.transformation_type}")
            
            result = transformation_func(df, mapping)
            
            # Log transformation
            transformation_log.append({
                "source_field": mapping.source_field,
                "target_field": mapping.target_field,
                "transformation_type": mapping.transformation_type.value,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            })
            
            return result
            
        except Exception as e:
            transformation_log.append({
                "source_field": mapping.source_field,
                "target_field": mapping.target_field,
                "transformation_type": mapping.transformation_type.value,
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "error": str(e)
            })
            raise
    
    def _initialize_transformations(self) -> Dict[TransformationType, callable]:
        """Initialize transformation functions"""
        return {
            TransformationType.DIRECT_MAPPING: self._direct_mapping,
            TransformationType.VALUE_MAPPING: self._value_mapping,
            TransformationType.CALCULATION: self._calculation,
            TransformationType.AGGREGATION: self._aggregation,
            TransformationType.CONCATENATION: self._concatenation
        }
    
    def _direct_mapping(self, df: pd.DataFrame, mapping: SchemaMapping) -> pd.Series:
        """Direct field mapping"""
        if mapping.source_field not in df.columns:
            raise ValueError(f"Source field {mapping.source_field} not found")
        return df[mapping.source_field]
    
    def _value_mapping(self, df: pd.DataFrame, mapping: SchemaMapping) -> pd.Series:
        """Value mapping with lookup table"""
        source_data = df[mapping.source_field]
        value_map = mapping.transformation_config.get("value_map", {})
        default_value = mapping.transformation_config.get("default_value")
        
        return source_data.map(value_map).fillna(default_value)
    
    def _calculation(self, df: pd.DataFrame, mapping: SchemaMapping) -> pd.Series:
        """Mathematical calculation transformation"""
        formula = mapping.transformation_config.get("formula", "")
        if not formula:
            raise ValueError("Formula required for calculation transformation")
        
        # Simple formula evaluation (extend as needed)
        # This is a basic implementation - in production, use a safer expression evaluator
        try:
            return df.eval(formula)
        except Exception as e:
            raise ValueError(f"Formula evaluation failed: {str(e)}")
    
    def _aggregation(self, df: pd.DataFrame, mapping: SchemaMapping) -> pd.Series:
        """Aggregation transformation"""
        source_fields = mapping.transformation_config.get("source_fields", [])
        agg_function = mapping.transformation_config.get("function", "sum")
        
        if not source_fields:
            source_fields = [mapping.source_field]
        
        data_subset = df[source_fields]
        
        if agg_function == "sum":
            return data_subset.sum(axis=1)
        elif agg_function == "mean":
            return data_subset.mean(axis=1)
        elif agg_function == "max":
            return data_subset.max(axis=1)
        elif agg_function == "min":
            return data_subset.min(axis=1)
        else:
            raise ValueError(f"Unknown aggregation function: {agg_function}")
    
    def _concatenation(self, df: pd.DataFrame, mapping: SchemaMapping) -> pd.Series:
        """String concatenation transformation"""
        source_fields = mapping.transformation_config.get("source_fields", [])
        separator = mapping.transformation_config.get("separator", " ")
        
        if not source_fields:
            source_fields = [mapping.source_field]
        
        result = df[source_fields[0]].astype(str)
        for field in source_fields[1:]:
            result = result + separator + df[field].astype(str)
        
        return result
    
    def _calculate_quality_metrics(self, source_df: pd.DataFrame, 
                                 target_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        try:
            source_rows = len(source_df)
            target_rows = len(target_df)
            
            # Basic metrics
            metrics = {
                "source_rows": source_rows,
                "target_rows": target_rows,
                "transformation_success_rate": target_rows / source_rows if source_rows > 0 else 0,
                "completeness": {},
                "data_types": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Completeness metrics
            for column in target_df.columns:
                non_null_count = target_df[column].notna().sum()
                metrics["completeness"][column] = non_null_count / target_rows if target_rows > 0 else 0
            
            # Data type distribution
            for column in target_df.columns:
                dtype_info = str(target_df[column].dtype)
                metrics["data_types"][column] = dtype_info
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {str(e)}")
            return {"error": str(e)}
    
    def validate_schema_compatibility(self, source_schema: str, 
                                    target_schema: str) -> Dict[str, Any]:
        """Validate compatibility between schemas"""
        try:
            source_schema_obj = self.schemas.get(source_schema)
            target_schema_obj = self.schemas.get(target_schema)
            
            if not source_schema_obj or not target_schema_obj:
                return {
                    "compatible": False,
                    "errors": ["One or both schemas not found"]
                }
            
            mapping_key = f"{source_schema}->{target_schema}"
            mappings = self.mappings.get(mapping_key, [])
            
            errors = []
            warnings = []
            
            # Check if all required target fields have mappings
            target_field_names = {field.name for field in target_schema_obj.fields if field.required}
            mapped_fields = {mapping.target_field for mapping in mappings}
            
            missing_mappings = target_field_names - mapped_fields
            if missing_mappings:
                errors.extend([f"No mapping for required field: {field}" for field in missing_mappings])
            
            # Check for unmapped source fields
            source_field_names = {field.name for field in source_schema_obj.fields}
            mapped_source_fields = {mapping.source_field for mapping in mappings}
            
            unmapped_source = source_field_names - mapped_source_fields
            if unmapped_source:
                warnings.extend([f"Unmapped source field: {field}" for field in unmapped_source])
            
            return {
                "compatible": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "mapping_coverage": len(mapped_fields) / len(target_field_names) if target_field_names else 1.0
            }
            
        except Exception as e:
            return {
                "compatible": False,
                "errors": [f"Validation failed: {str(e)}"]
            }
    
    def get_schema_info(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a schema"""
        schema = self.schemas.get(schema_name)
        if not schema:
            return None
        
        return {
            "name": schema.name,
            "version": schema.version,
            "field_count": len(schema.fields),
            "fields": [
                {
                    "name": field.name,
                    "type": field.data_type.value,
                    "required": field.required,
                    "default": field.default_value,
                    "description": field.description
                }
                for field in schema.fields
            ],
            "metadata": schema.metadata
        }