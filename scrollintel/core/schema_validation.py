"""
Schema Validation Framework for ScrollIntel

This module provides comprehensive JSON schema validation for all agent
communications with error reporting, versioning, and validation middleware.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import jsonschema
from jsonschema import Draft7Validator, validators
from datetime import datetime
import uuid
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SchemaType(Enum):
    """Schema types for different validation contexts"""
    REQUEST = "request"
    RESPONSE = "response"
    CONFIG = "config"
    EVENT = "event"
    METRIC = "metric"

@dataclass
class ValidationError:
    """Detailed validation error information"""
    path: str
    message: str
    code: str
    value: Any
    severity: ValidationSeverity = ValidationSeverity.ERROR
    schema_path: str = ""
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationWarning:
    """Validation warning information"""
    path: str
    message: str
    code: str
    value: Any
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    schema_version: str = "1.0.0"
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SchemaInfo:
    """Schema metadata and information"""
    name: str
    version: str
    schema_type: SchemaType
    schema: Dict[str, Any]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    deprecated: bool = False
    migration_path: Optional[str] = None

class SchemaRegistry:
    """
    Registry for managing JSON schemas with versioning and validation.
    """
    
    def __init__(self):
        self._schemas: Dict[str, Dict[str, SchemaInfo]] = defaultdict(dict)
        self._default_versions: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._cache_max_size = 1000
        
        # Initialize with built-in schemas
        self._register_builtin_schemas()
    
    def register_schema(self, name: str, version: str, schema: Dict[str, Any], 
                       schema_type: SchemaType = SchemaType.REQUEST,
                       description: str = "", tags: List[str] = None,
                       dependencies: List[str] = None) -> bool:
        """
        Register a new schema with version management.
        """
        try:
            with self._lock:
                # Validate the schema itself
                Draft7Validator.check_schema(schema)
                
                schema_info = SchemaInfo(
                    name=name,
                    version=version,
                    schema_type=schema_type,
                    schema=schema,
                    description=description,
                    tags=tags or [],
                    dependencies=dependencies or []
                )
                
                self._schemas[name][version] = schema_info
                
                # Set as default version if it's the first or higher version
                if name not in self._default_versions or self._is_higher_version(version, self._default_versions[name]):
                    self._default_versions[name] = version
                
                logger.info(f"Registered schema {name} version {version}")
                return True
                
        except jsonschema.SchemaError as e:
            logger.error(f"Invalid schema {name} version {version}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to register schema {name} version {version}: {e}")
            return False
    
    def get_schema(self, name: str, version: Optional[str] = None) -> Optional[SchemaInfo]:
        """
        Get schema by name and version.
        """
        with self._lock:
            if name not in self._schemas:
                return None
            
            target_version = version or self._default_versions.get(name)
            if not target_version:
                return None
            
            return self._schemas[name].get(target_version)
    
    def list_schemas(self, schema_type: Optional[SchemaType] = None, 
                    tags: Optional[List[str]] = None) -> List[SchemaInfo]:
        """
        List all schemas with optional filtering.
        """
        with self._lock:
            schemas = []
            
            for name_versions in self._schemas.values():
                for schema_info in name_versions.values():
                    # Filter by type
                    if schema_type and schema_info.schema_type != schema_type:
                        continue
                    
                    # Filter by tags
                    if tags and not any(tag in schema_info.tags for tag in tags):
                        continue
                    
                    schemas.append(schema_info)
            
            return schemas
    
    def deprecate_schema(self, name: str, version: str, migration_path: Optional[str] = None) -> bool:
        """
        Mark a schema version as deprecated.
        """
        with self._lock:
            schema_info = self.get_schema(name, version)
            if schema_info:
                schema_info.deprecated = True
                schema_info.migration_path = migration_path
                schema_info.updated_at = datetime.now()
                logger.info(f"Deprecated schema {name} version {version}")
                return True
            return False
    
    def get_schema_versions(self, name: str) -> List[str]:
        """
        Get all versions of a schema.
        """
        with self._lock:
            return list(self._schemas.get(name, {}).keys())
    
    def _is_higher_version(self, version1: str, version2: str) -> bool:
        """
        Compare version strings (simple semantic versioning).
        """
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad with zeros to make same length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            return v1_parts > v2_parts
        except ValueError:
            # Fallback to string comparison
            return version1 > version2
    
    def _register_builtin_schemas(self):
        """
        Register built-in schemas for common data types.
        """
        # Agent request schema
        agent_request_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "type": {"type": "string", "minLength": 1},
                "payload": {"type": "object"},
                "capabilities_required": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                "timeout": {"type": "integer", "minimum": 1},
                "retry_count": {"type": "integer", "minimum": 0},
                "max_retries": {"type": "integer", "minimum": 0},
                "correlation_id": {"type": ["string", "null"]},
                "user_id": {"type": ["string", "null"]},
                "session_id": {"type": ["string", "null"]},
                "metadata": {"type": "object"},
                "timestamp": {"type": "string", "format": "date-time"},
                "deadline": {"type": ["string", "null"], "format": "date-time"}
            },
            "required": ["id", "type", "payload"],
            "additionalProperties": False
        }
        
        self.register_schema(
            "agent_request", "1.0.0", agent_request_schema,
            SchemaType.REQUEST, "Standard agent request format"
        )
        
        # Agent response schema
        agent_response_schema = {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "minLength": 1},
                "agent_id": {"type": "string", "minLength": 1},
                "status": {"type": "string", "enum": ["success", "error", "partial", "timeout"]},
                "data": {},  # Any type allowed
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "processing_time": {"type": "number", "minimum": 0},
                "error": {"type": ["string", "null"]},
                "error_code": {"type": ["string", "null"]},
                "warnings": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "metadata": {"type": "object"},
                "timestamp": {"type": "string", "format": "date-time"},
                "trace_id": {"type": ["string", "null"]}
            },
            "required": ["request_id", "agent_id", "status", "confidence", "processing_time"],
            "additionalProperties": False
        }
        
        self.register_schema(
            "agent_response", "1.0.0", agent_response_schema,
            SchemaType.RESPONSE, "Standard agent response format"
        )
        
        # Health status schema
        health_status_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "status": {"type": "string", "enum": ["pass", "warn", "fail"]},
                            "details": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "duration": {"type": "number", "minimum": 0},
                            "metadata": {"type": "object"}
                        },
                        "required": ["name", "status", "details"]
                    }
                },
                "last_updated": {"type": "string", "format": "date-time"},
                "uptime": {"type": "string"},
                "version": {"type": "string"},
                "build_info": {"type": "object"}
            },
            "required": ["status", "checks", "last_updated"],
            "additionalProperties": False
        }
        
        self.register_schema(
            "health_status", "1.0.0", health_status_schema,
            SchemaType.RESPONSE, "Agent health status format"
        )

class SchemaValidator:
    """
    Advanced JSON schema validator with enhanced error reporting and caching.
    """
    
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry
        self._validator_cache: Dict[str, Draft7Validator] = {}
        self._cache_lock = threading.RLock()
    
    def validate_request(self, data: Any, schema_name: str, 
                        schema_version: Optional[str] = None) -> ValidationResult:
        """
        Validate request data against a schema.
        """
        return self._validate(data, schema_name, schema_version, SchemaType.REQUEST)
    
    def validate_response(self, data: Any, schema_name: str,
                         schema_version: Optional[str] = None) -> ValidationResult:
        """
        Validate response data against a schema.
        """
        return self._validate(data, schema_name, schema_version, SchemaType.RESPONSE)
    
    def validate_config(self, data: Any, schema_name: str,
                       schema_version: Optional[str] = None) -> ValidationResult:
        """
        Validate configuration data against a schema.
        """
        return self._validate(data, schema_name, schema_version, SchemaType.CONFIG)
    
    def _validate(self, data: Any, schema_name: str, schema_version: Optional[str],
                 expected_type: SchemaType) -> ValidationResult:
        """
        Internal validation method with comprehensive error handling.
        """
        start_time = datetime.now()
        
        try:
            # Get schema
            schema_info = self.registry.get_schema(schema_name, schema_version)
            if not schema_info:
                return ValidationResult(
                    valid=False,
                    errors=[ValidationError(
                        path="",
                        message=f"Schema {schema_name} version {schema_version or 'default'} not found",
                        code="schema_not_found",
                        value=None,
                        severity=ValidationSeverity.CRITICAL
                    )],
                    validation_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Check if schema is deprecated
            warnings = []
            if schema_info.deprecated:
                warnings.append(ValidationWarning(
                    path="",
                    message=f"Schema {schema_name} version {schema_info.version} is deprecated",
                    code="deprecated_schema",
                    value=None,
                    suggestion=schema_info.migration_path
                ))
            
            # Get or create validator
            validator = self._get_validator(schema_info)
            
            # Perform validation
            errors = []
            for error in validator.iter_errors(data):
                validation_error = self._convert_jsonschema_error(error)
                errors.append(validation_error)
            
            # Additional custom validations
            custom_errors, custom_warnings = self._custom_validations(data, schema_info)
            errors.extend(custom_errors)
            warnings.extend(custom_warnings)
            
            validation_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                schema_version=schema_info.version,
                validation_time=validation_time,
                metadata={
                    "schema_name": schema_name,
                    "schema_type": schema_info.schema_type.value,
                    "data_type": type(data).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Validation error for schema {schema_name}: {e}")
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    path="",
                    message=f"Validation failed: {str(e)}",
                    code="validation_exception",
                    value=None,
                    severity=ValidationSeverity.CRITICAL
                )],
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _get_validator(self, schema_info: SchemaInfo) -> Draft7Validator:
        """
        Get or create a cached validator for the schema.
        """
        cache_key = f"{schema_info.name}:{schema_info.version}"
        
        with self._cache_lock:
            if cache_key not in self._validator_cache:
                # Create custom validator with enhanced error reporting
                validator_class = validators.create(
                    meta_schema=Draft7Validator.META_SCHEMA,
                    validators=Draft7Validator.VALIDATORS
                )
                
                self._validator_cache[cache_key] = validator_class(schema_info.schema)
            
            return self._validator_cache[cache_key]
    
    def _convert_jsonschema_error(self, error: jsonschema.ValidationError) -> ValidationError:
        """
        Convert jsonschema ValidationError to our ValidationError format.
        """
        path = ".".join(str(p) for p in error.absolute_path)
        
        # Determine severity based on error type
        severity = ValidationSeverity.ERROR
        if error.validator in ["required", "type"]:
            severity = ValidationSeverity.CRITICAL
        elif error.validator in ["format", "pattern"]:
            severity = ValidationSeverity.WARNING
        
        # Generate helpful suggestions
        suggestion = self._generate_suggestion(error)
        
        return ValidationError(
            path=path,
            message=error.message,
            code=error.validator,
            value=error.instance,
            severity=severity,
            schema_path=".".join(str(p) for p in error.schema_path),
            suggestion=suggestion
        )
    
    def _generate_suggestion(self, error: jsonschema.ValidationError) -> Optional[str]:
        """
        Generate helpful suggestions for common validation errors.
        """
        if error.validator == "required":
            missing_props = error.validator_value
            return f"Add required properties: {', '.join(missing_props)}"
        
        elif error.validator == "type":
            expected_type = error.validator_value
            actual_type = type(error.instance).__name__
            return f"Expected {expected_type}, got {actual_type}"
        
        elif error.validator == "enum":
            valid_values = error.validator_value
            return f"Valid values are: {', '.join(map(str, valid_values))}"
        
        elif error.validator == "minLength":
            min_len = error.validator_value
            return f"Minimum length is {min_len} characters"
        
        elif error.validator == "maxLength":
            max_len = error.validator_value
            return f"Maximum length is {max_len} characters"
        
        elif error.validator == "minimum":
            min_val = error.validator_value
            return f"Minimum value is {min_val}"
        
        elif error.validator == "maximum":
            max_val = error.validator_value
            return f"Maximum value is {max_val}"
        
        return None
    
    def _custom_validations(self, data: Any, schema_info: SchemaInfo) -> Tuple[List[ValidationError], List[ValidationWarning]]:
        """
        Perform custom validations beyond JSON schema.
        """
        errors = []
        warnings = []
        
        # Example: Check for common security issues
        if isinstance(data, dict):
            # Check for potential SQL injection patterns
            for key, value in data.items():
                if isinstance(value, str) and any(pattern in value.lower() for pattern in ['drop table', 'delete from', 'union select']):
                    errors.append(ValidationError(
                        path=key,
                        message="Potential SQL injection detected",
                        code="security_sql_injection",
                        value=value,
                        severity=ValidationSeverity.CRITICAL,
                        suggestion="Sanitize input data"
                    ))
                
                # Check for excessively long strings
                if isinstance(value, str) and len(value) > 10000:
                    warnings.append(ValidationWarning(
                        path=key,
                        message="Very long string detected",
                        code="performance_long_string",
                        value=f"Length: {len(value)}",
                        suggestion="Consider truncating or using references for large data"
                    ))
        
        return errors, warnings

class ValidationMiddleware:
    """
    Middleware for automatic request/response validation.
    """
    
    def __init__(self, validator: SchemaValidator):
        self.validator = validator
        self.validation_stats = defaultdict(int)
        self._stats_lock = threading.Lock()
    
    async def validate_request_middleware(self, request: Any, schema_name: str,
                                        schema_version: Optional[str] = None) -> ValidationResult:
        """
        Middleware for validating incoming requests.
        """
        result = self.validator.validate_request(request, schema_name, schema_version)
        
        with self._stats_lock:
            self.validation_stats["total_requests"] += 1
            if result.valid:
                self.validation_stats["valid_requests"] += 1
            else:
                self.validation_stats["invalid_requests"] += 1
        
        if not result.valid:
            logger.warning(f"Request validation failed for schema {schema_name}: {len(result.errors)} errors")
            for error in result.errors:
                logger.warning(f"  {error.path}: {error.message}")
        
        return result
    
    async def validate_response_middleware(self, response: Any, schema_name: str,
                                         schema_version: Optional[str] = None) -> ValidationResult:
        """
        Middleware for validating outgoing responses.
        """
        result = self.validator.validate_response(response, schema_name, schema_version)
        
        with self._stats_lock:
            self.validation_stats["total_responses"] += 1
            if result.valid:
                self.validation_stats["valid_responses"] += 1
            else:
                self.validation_stats["invalid_responses"] += 1
        
        if not result.valid:
            logger.error(f"Response validation failed for schema {schema_name}: {len(result.errors)} errors")
            for error in result.errors:
                logger.error(f"  {error.path}: {error.message}")
        
        return result
    
    def get_validation_stats(self) -> Dict[str, int]:
        """
        Get validation statistics.
        """
        with self._stats_lock:
            return dict(self.validation_stats)
    
    def reset_stats(self):
        """
        Reset validation statistics.
        """
        with self._stats_lock:
            self.validation_stats.clear()

# Global instances
schema_registry = SchemaRegistry()
schema_validator = SchemaValidator(schema_registry)
validation_middleware = ValidationMiddleware(schema_validator)

# Utility functions
def validate_agent_request(data: Any) -> ValidationResult:
    """Validate agent request data."""
    return schema_validator.validate_request(data, "agent_request")

def validate_agent_response(data: Any) -> ValidationResult:
    """Validate agent response data."""
    return schema_validator.validate_response(data, "agent_response")

def validate_health_status(data: Any) -> ValidationResult:
    """Validate health status data."""
    return schema_validator.validate_response(data, "health_status")

def register_custom_schema(name: str, schema: Dict[str, Any], 
                          schema_type: SchemaType = SchemaType.REQUEST,
                          version: str = "1.0.0") -> bool:
    """Register a custom schema."""
    return schema_registry.register_schema(name, version, schema, schema_type)

def get_validation_stats() -> Dict[str, int]:
    """Get global validation statistics."""
    return validation_middleware.get_validation_stats()

# Schema validation decorators
def validate_input(schema_name: str, schema_version: Optional[str] = None):
    """Decorator for automatic input validation."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Assume first argument is the data to validate
            if args:
                result = schema_validator.validate_request(args[0], schema_name, schema_version)
                if not result.valid:
                    raise ValueError(f"Input validation failed: {result.errors}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_output(schema_name: str, schema_version: Optional[str] = None):
    """Decorator for automatic output validation."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            validation_result = schema_validator.validate_response(result, schema_name, schema_version)
            if not validation_result.valid:
                logger.error(f"Output validation failed: {validation_result.errors}")
                # In production, you might want to handle this differently
            return result
        return wrapper
    return decorator