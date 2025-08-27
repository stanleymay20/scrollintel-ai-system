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
    Enhanced with automated schema discovery and intelligent mapping suggestions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.schemas: Dict[str, DataSchema] = {}
        self.mappings: Dict[str, List[SchemaMapping]] = {}
        self.transformation_functions = self._initialize_transformations()
        self.schema_registry = {}
        self.mapping_suggestions = {}
        self.transformation_cache = {}
        self.validation_rules: Dict[str, List[callable]] = {}
        self.quality_thresholds = self.config.get("quality_thresholds", {
            "completeness": 0.95,
            "validity": 0.98,
            "consistency": 0.99
        })
        
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
    
    def auto_discover_schema(self, data: pd.DataFrame, schema_name: str) -> DataSchema:
        """Automatically discover schema from data"""
        try:
            fields = []
            
            for column in data.columns:
                # Infer data type
                dtype = data[column].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    data_type = DataType.INTEGER
                elif pd.api.types.is_float_dtype(dtype):
                    data_type = DataType.FLOAT
                elif pd.api.types.is_bool_dtype(dtype):
                    data_type = DataType.BOOLEAN
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    data_type = DataType.DATETIME
                else:
                    data_type = DataType.STRING
                
                # Check if field is required (has non-null values)
                required = data[column].notna().all()
                
                # Get default value if not required
                default_value = None
                if not required:
                    mode_values = data[column].mode()
                    if not mode_values.empty:
                        default_value = mode_values.iloc[0]
                
                field = SchemaField(
                    name=column,
                    data_type=data_type,
                    required=required,
                    default_value=default_value,
                    description=f"Auto-discovered field from {schema_name}"
                )
                fields.append(field)
            
            schema = DataSchema(
                name=schema_name,
                version="1.0",
                fields=fields,
                metadata={
                    "auto_discovered": True,
                    "discovery_timestamp": datetime.utcnow().isoformat(),
                    "source_records": len(data)
                }
            )
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema auto-discovery failed: {str(e)}")
            raise
    
    def suggest_field_mappings(self, source_schema: str, target_schema: str) -> List[Dict[str, Any]]:
        """Suggest field mappings between schemas using similarity analysis"""
        try:
            source_schema_obj = self.schemas.get(source_schema)
            target_schema_obj = self.schemas.get(target_schema)
            
            if not source_schema_obj or not target_schema_obj:
                return []
            
            suggestions = []
            
            for target_field in target_schema_obj.fields:
                best_matches = []
                
                for source_field in source_schema_obj.fields:
                    # Calculate similarity score
                    name_similarity = self._calculate_name_similarity(
                        source_field.name, target_field.name
                    )
                    type_compatibility = self._check_type_compatibility(
                        source_field.data_type, target_field.data_type
                    )
                    
                    overall_score = (name_similarity * 0.7) + (type_compatibility * 0.3)
                    
                    if overall_score > 0.3:  # Threshold for suggestions
                        best_matches.append({
                            "source_field": source_field.name,
                            "target_field": target_field.name,
                            "similarity_score": overall_score,
                            "name_similarity": name_similarity,
                            "type_compatibility": type_compatibility,
                            "suggested_transformation": self._suggest_transformation(
                                source_field, target_field
                            )
                        })
                
                # Sort by similarity score
                best_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
                suggestions.extend(best_matches[:3])  # Top 3 suggestions per target field
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest mappings: {str(e)}")
            return []
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between field names"""
        try:
            # Simple similarity based on common substrings and edit distance
            name1_lower = name1.lower().replace("_", "").replace("-", "")
            name2_lower = name2.lower().replace("_", "").replace("-", "")
            
            if name1_lower == name2_lower:
                return 1.0
            
            # Check for substring matches
            if name1_lower in name2_lower or name2_lower in name1_lower:
                return 0.8
            
            # Simple edit distance approximation
            common_chars = set(name1_lower) & set(name2_lower)
            total_chars = set(name1_lower) | set(name2_lower)
            
            if total_chars:
                return len(common_chars) / len(total_chars)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _check_type_compatibility(self, source_type: DataType, target_type: DataType) -> float:
        """Check compatibility between data types"""
        if source_type == target_type:
            return 1.0
        
        # Define compatibility matrix
        compatibility_matrix = {
            (DataType.INTEGER, DataType.FLOAT): 0.9,
            (DataType.INTEGER, DataType.STRING): 0.7,
            (DataType.FLOAT, DataType.STRING): 0.7,
            (DataType.BOOLEAN, DataType.STRING): 0.6,
            (DataType.DATETIME, DataType.STRING): 0.8,
        }
        
        # Check both directions
        score = compatibility_matrix.get((source_type, target_type), 0.0)
        if score == 0.0:
            score = compatibility_matrix.get((target_type, source_type), 0.0)
        
        return score
    
    def _suggest_transformation(self, source_field: SchemaField, target_field: SchemaField) -> Dict[str, Any]:
        """Suggest appropriate transformation for field mapping"""
        if source_field.data_type == target_field.data_type:
            return {
                "type": TransformationType.DIRECT_MAPPING.value,
                "confidence": 0.9
            }
        
        # Type conversion suggestions
        if (source_field.data_type == DataType.INTEGER and 
            target_field.data_type == DataType.FLOAT):
            return {
                "type": TransformationType.CALCULATION.value,
                "formula": f"float({source_field.name})",
                "confidence": 0.8
            }
        
        if target_field.data_type == DataType.STRING:
            return {
                "type": TransformationType.CALCULATION.value,
                "formula": f"str({source_field.name})",
                "confidence": 0.7
            }
        
        return {
            "type": TransformationType.DIRECT_MAPPING.value,
            "confidence": 0.5
        }
    
    def register_validation_rule(self, field_name: str, validation_func: callable) -> bool:
        """Register a custom validation rule for a field"""
        try:
            if field_name not in self.validation_rules:
                self.validation_rules[field_name] = []
            
            self.validation_rules[field_name].append(validation_func)
            logger.info(f"Registered validation rule for field: {field_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register validation rule for {field_name}: {str(e)}")
            return False
    
    def validate_normalized_data(self, data: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
        """Validate normalized data against schema and custom rules"""
        try:
            schema = self.schemas.get(schema_name)
            if not schema:
                return {"error": f"Schema {schema_name} not found"}
            
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "quality_metrics": {}
            }
            
            # Schema validation
            for field in schema.fields:
                if field.name not in data.columns:
                    if field.required:
                        validation_results["errors"].append(f"Required field {field.name} missing")
                        validation_results["valid"] = False
                    else:
                        validation_results["warnings"].append(f"Optional field {field.name} missing")
                    continue
                
                # Data type validation
                column_data = data[field.name]
                
                # Completeness check
                completeness = column_data.notna().sum() / len(data) if len(data) > 0 else 1.0
                if completeness < self.quality_thresholds.get("completeness", 0.95):
                    validation_results["warnings"].append(
                        f"Field {field.name} completeness ({completeness:.2%}) below threshold"
                    )
                
                # Custom validation rules
                if field.name in self.validation_rules:
                    for rule in self.validation_rules[field.name]:
                        try:
                            rule_result = rule(column_data)
                            if not rule_result.get("valid", True):
                                validation_results["errors"].append(
                                    f"Validation failed for {field.name}: {rule_result.get('message', 'Unknown error')}"
                                )
                                validation_results["valid"] = False
                        except Exception as e:
                            validation_results["warnings"].append(
                                f"Validation rule error for {field.name}: {str(e)}"
                            )
                
                # Calculate quality metrics
                validation_results["quality_metrics"][field.name] = {
                    "completeness": completeness,
                    "null_count": column_data.isna().sum(),
                    "unique_count": column_data.nunique(),
                    "data_type": str(column_data.dtype)
                }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {"error": str(e)}
    
    def create_data_quality_profile(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Create comprehensive data quality profile"""
        try:
            profile = {
                "dataset_name": dataset_name,
                "profiling_timestamp": datetime.utcnow().isoformat(),
                "basic_statistics": {
                    "row_count": len(data),
                    "column_count": len(data.columns),
                    "memory_usage": data.memory_usage(deep=True).sum(),
                    "duplicate_rows": data.duplicated().sum()
                },
                "column_profiles": {},
                "data_quality_score": 0.0,
                "recommendations": []
            }
            
            total_quality_score = 0
            
            for column in data.columns:
                column_data = data[column]
                
                # Basic statistics
                column_profile = {
                    "data_type": str(column_data.dtype),
                    "null_count": column_data.isna().sum(),
                    "null_percentage": column_data.isna().sum() / len(data) * 100,
                    "unique_count": column_data.nunique(),
                    "unique_percentage": column_data.nunique() / len(data) * 100 if len(data) > 0 else 0
                }
                
                # Type-specific statistics
                if pd.api.types.is_numeric_dtype(column_data):
                    column_profile.update({
                        "min": column_data.min(),
                        "max": column_data.max(),
                        "mean": column_data.mean(),
                        "median": column_data.median(),
                        "std": column_data.std(),
                        "zeros_count": (column_data == 0).sum(),
                        "negative_count": (column_data < 0).sum()
                    })
                elif pd.api.types.is_string_dtype(column_data) or column_data.dtype == 'object':
                    non_null_data = column_data.dropna()
                    if not non_null_data.empty:
                        column_profile.update({
                            "min_length": non_null_data.astype(str).str.len().min(),
                            "max_length": non_null_data.astype(str).str.len().max(),
                            "avg_length": non_null_data.astype(str).str.len().mean(),
                            "empty_strings": (non_null_data.astype(str) == "").sum(),
                            "whitespace_only": non_null_data.astype(str).str.strip().eq("").sum()
                        })
                
                # Quality score for this column
                completeness_score = (1 - column_profile["null_percentage"] / 100) * 100
                uniqueness_score = min(100, column_profile["unique_percentage"])
                
                column_quality_score = (completeness_score + uniqueness_score) / 2
                column_profile["quality_score"] = column_quality_score
                
                total_quality_score += column_quality_score
                
                profile["column_profiles"][column] = column_profile
            
            # Overall quality score
            profile["data_quality_score"] = total_quality_score / len(data.columns) if data.columns.size > 0 else 0
            
            # Generate recommendations
            profile["recommendations"] = self._generate_quality_recommendations(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to create data quality profile: {str(e)}")
            return {"error": str(e)}
    
    def _generate_quality_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        try:
            # Overall recommendations
            if profile["data_quality_score"] < 70:
                recommendations.append("Overall data quality is poor. Consider comprehensive data cleansing.")
            
            # Duplicate rows
            if profile["basic_statistics"]["duplicate_rows"] > 0:
                recommendations.append(f"Remove {profile['basic_statistics']['duplicate_rows']} duplicate rows.")
            
            # Column-specific recommendations
            for column, column_profile in profile["column_profiles"].items():
                if column_profile["null_percentage"] > 20:
                    recommendations.append(f"Column '{column}' has high null percentage ({column_profile['null_percentage']:.1f}%). Consider imputation or removal.")
                
                if column_profile["unique_count"] == 1:
                    recommendations.append(f"Column '{column}' has only one unique value. Consider removing if not needed.")
                
                if column_profile.get("empty_strings", 0) > 0:
                    recommendations.append(f"Column '{column}' contains empty strings. Consider standardizing null representation.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return ["Error generating recommendations"]