"""
API Generation Models for Automated Code Generation System

This module defines the data models for API specifications, endpoints,
and related components used in the API code generation system.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class HTTPMethod(Enum):
    """HTTP methods supported by the API generator."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class APIType(Enum):
    """Types of APIs that can be generated."""
    REST = "REST"
    GRAPHQL = "GRAPHQL"
    HYBRID = "HYBRID"


class ValidationRule(Enum):
    """Validation rules for API parameters."""
    REQUIRED = "required"
    EMAIL = "email"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    PATTERN = "pattern"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    UNIQUE = "unique"


@dataclass
class Parameter:
    """Represents an API parameter with validation rules."""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False
    default_value: Optional[Any] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    example: Optional[Any] = None
    
    def add_validation_rule(self, rule: ValidationRule, value: Any = None):
        """Add a validation rule to the parameter."""
        rule_dict = {"rule": rule.value}
        if value is not None:
            rule_dict["value"] = value
        self.validation_rules.append(rule_dict)


@dataclass
class Response:
    """Represents an API response specification."""
    status_code: int
    description: str
    schema: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    examples: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Endpoint:
    """Represents a single API endpoint."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    path: str = ""
    method: HTTPMethod = HTTPMethod.GET
    name: str = ""
    description: str = ""
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: List[Response] = field(default_factory=list)
    security_requirements: List[str] = field(default_factory=list)
    deprecated: bool = False
    version: str = "1.0.0"
    
    def add_parameter(self, param: Parameter):
        """Add a parameter to the endpoint."""
        self.parameters.append(param)
    
    def add_response(self, response: Response):
        """Add a response specification to the endpoint."""
        self.responses.append(response)


@dataclass
class GraphQLField:
    """Represents a GraphQL field definition."""
    name: str
    type: str
    description: Optional[str] = None
    arguments: List[Parameter] = field(default_factory=list)
    resolver: Optional[str] = None
    nullable: bool = True
    list_type: bool = False


@dataclass
class GraphQLType:
    """Represents a GraphQL type definition."""
    name: str
    description: Optional[str] = None
    fields: List[GraphQLField] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    is_input: bool = False
    is_enum: bool = False
    enum_values: List[str] = field(default_factory=list)


@dataclass
class GraphQLSchema:
    """Represents a complete GraphQL schema."""
    types: List[GraphQLType] = field(default_factory=list)
    queries: List[GraphQLField] = field(default_factory=list)
    mutations: List[GraphQLField] = field(default_factory=list)
    subscriptions: List[GraphQLField] = field(default_factory=list)


@dataclass
class SecurityScheme:
    """Represents API security scheme configuration."""
    name: str
    type: str  # apiKey, http, oauth2, openIdConnect
    description: Optional[str] = None
    in_location: Optional[str] = None  # query, header, cookie
    scheme: Optional[str] = None  # bearer, basic
    bearer_format: Optional[str] = None
    flows: Optional[Dict[str, Any]] = None


@dataclass
class APIVersion:
    """Represents an API version with backward compatibility info."""
    version: str
    release_date: datetime
    deprecated: bool = False
    sunset_date: Optional[datetime] = None
    breaking_changes: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None
    changelog: List[str] = field(default_factory=list)


@dataclass
class APISpec:
    """Complete API specification for code generation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    api_type: APIType = APIType.REST
    base_url: str = ""
    
    # REST API components
    endpoints: List[Endpoint] = field(default_factory=list)
    
    # GraphQL components
    graphql_schema: Optional[GraphQLSchema] = None
    
    # Common components
    security_schemes: List[SecurityScheme] = field(default_factory=list)
    global_parameters: List[Parameter] = field(default_factory=list)
    global_headers: Dict[str, str] = field(default_factory=dict)
    
    # Versioning and compatibility
    versions: List[APIVersion] = field(default_factory=list)
    backward_compatible: bool = True
    
    # Documentation
    documentation: Dict[str, Any] = field(default_factory=dict)
    examples: Dict[str, Any] = field(default_factory=dict)
    
    # Generation metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    generated_code: Dict[str, str] = field(default_factory=dict)
    
    def add_endpoint(self, endpoint: Endpoint):
        """Add an endpoint to the API specification."""
        self.endpoints.append(endpoint)
        self.updated_at = datetime.now()
    
    def add_security_scheme(self, scheme: SecurityScheme):
        """Add a security scheme to the API specification."""
        self.security_schemes.append(scheme)
    
    def add_version(self, version: APIVersion):
        """Add a new version to the API specification."""
        self.versions.append(version)
        self.version = version.version
        self.updated_at = datetime.now()
    
    def get_current_version(self) -> Optional[APIVersion]:
        """Get the current version of the API."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.release_date)


@dataclass
class GeneratedAPICode:
    """Represents generated API code with metadata."""
    api_spec_id: str
    language: str
    framework: str
    code_files: Dict[str, str] = field(default_factory=dict)
    test_files: Dict[str, str] = field(default_factory=dict)
    documentation_files: Dict[str, str] = field(default_factory=dict)
    configuration_files: Dict[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    generator_version: str = "1.0.0"
    
    def add_code_file(self, filename: str, content: str):
        """Add a generated code file."""
        self.code_files[filename] = content
    
    def add_test_file(self, filename: str, content: str):
        """Add a generated test file."""
        self.test_files[filename] = content
    
    def add_documentation_file(self, filename: str, content: str):
        """Add a generated documentation file."""
        self.documentation_files[filename] = content


@dataclass
class CRUDOperation:
    """Represents a CRUD operation specification."""
    entity_name: str
    operation: str  # create, read, update, delete, list
    endpoint_path: str
    method: HTTPMethod
    parameters: List[Parameter] = field(default_factory=list)
    request_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)


@dataclass
class APIGenerationRequest:
    """Request for API code generation."""
    api_spec: APISpec
    target_language: str = "python"
    target_framework: str = "fastapi"
    include_tests: bool = True
    include_documentation: bool = True
    include_validation: bool = True
    include_authentication: bool = True
    database_type: str = "postgresql"
    orm_type: str = "sqlalchemy"
    additional_options: Dict[str, Any] = field(default_factory=dict)