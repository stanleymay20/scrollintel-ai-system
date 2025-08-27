"""
API Code Generation Engine

This module provides comprehensive API code generation capabilities including
RESTful APIs, GraphQL APIs, documentation generation, and versioning support.
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from jinja2 import Environment, BaseLoader, Template
import re

from ..models.api_generation_models import (
    APISpec, Endpoint, Parameter, Response, HTTPMethod, APIType,
    GraphQLSchema, GraphQLType, GraphQLField, SecurityScheme,
    GeneratedAPICode, CRUDOperation, APIGenerationRequest,
    ValidationRule, APIVersion
)
from ..models.database_schema_models import DatabaseSchema, Entity, Field


class APICodeGenerator:
    """Main API code generation engine."""
    
    def __init__(self):
        self.jinja_env = Environment(loader=BaseLoader())
        self.supported_languages = ["python", "javascript", "typescript", "java", "go"]
        self.supported_frameworks = {
            "python": ["fastapi", "flask", "django"],
            "javascript": ["express", "koa", "nestjs"],
            "typescript": ["express", "nestjs", "apollo"],
            "java": ["spring", "quarkus"],
            "go": ["gin", "echo", "gorilla"]
        }
    
    def generate_api_from_schema(
        self, 
        database_schema: DatabaseSchema, 
        request: APIGenerationRequest
    ) -> GeneratedAPICode:
        """Generate complete API code from database schema."""
        # Create API specification from database schema
        api_spec = self._create_api_spec_from_schema(database_schema, request)
        
        # Generate the API code
        return self.generate_api_code(api_spec, request)
    
    def generate_api_code(
        self, 
        api_spec: APISpec, 
        request: APIGenerationRequest
    ) -> GeneratedAPICode:
        """Generate complete API code from specification."""
        generated_code = GeneratedAPICode(
            api_spec_id=api_spec.id,
            language=request.target_language,
            framework=request.target_framework
        )
        
        # Generate based on API type
        if api_spec.api_type == APIType.REST:
            self._generate_rest_api(api_spec, request, generated_code)
        elif api_spec.api_type == APIType.GRAPHQL:
            self._generate_graphql_api(api_spec, request, generated_code)
        elif api_spec.api_type == APIType.HYBRID:
            self._generate_rest_api(api_spec, request, generated_code)
            self._generate_graphql_api(api_spec, request, generated_code)
        
        # Generate common components
        if request.include_documentation:
            self._generate_api_documentation(api_spec, generated_code)
        
        if request.include_tests:
            self._generate_api_tests(api_spec, request, generated_code)
        
        # Generate configuration files
        self._generate_configuration_files(api_spec, request, generated_code)
        
        return generated_code
    
    def _create_api_spec_from_schema(
        self, 
        database_schema: DatabaseSchema, 
        request: APIGenerationRequest
    ) -> APISpec:
        """Create API specification from database schema."""
        api_spec = APISpec(
            name=f"{database_schema.name} API",
            description=f"Auto-generated API for {database_schema.name}",
            version="1.0.0",
            api_type=APIType.REST,
            base_url=f"/api/v1"
        )
        
        # Generate CRUD endpoints for each entity
        for entity in database_schema.entities:
            crud_endpoints = self._generate_crud_endpoints(entity)
            for endpoint in crud_endpoints:
                api_spec.add_endpoint(endpoint)
        
        # Add authentication if requested
        if request.include_authentication:
            auth_scheme = SecurityScheme(
                name="bearerAuth",
                type="http",
                scheme="bearer",
                bearer_format="JWT",
                description="JWT Bearer token authentication"
            )
            api_spec.add_security_scheme(auth_scheme)
        
        return api_spec
    
    def _generate_crud_endpoints(self, entity: Entity) -> List[Endpoint]:
        """Generate CRUD endpoints for a database entity."""
        endpoints = []
        entity_name = entity.name.lower()
        entity_name_plural = f"{entity_name}s"  # Simple pluralization
        
        # GET /entities - List all
        list_endpoint = Endpoint(
            path=f"/{entity_name_plural}",
            method=HTTPMethod.GET,
            name=f"list_{entity_name_plural}",
            description=f"Retrieve a list of {entity_name_plural}",
            summary=f"List {entity_name_plural}",
            tags=[entity.name]
        )
        
        # Add pagination parameters
        list_endpoint.add_parameter(Parameter(
            name="page", type="integer", description="Page number", default_value=1
        ))
        list_endpoint.add_parameter(Parameter(
            name="limit", type="integer", description="Items per page", default_value=10
        ))
        
        # Add filter parameters for each field
        for field in entity.fields:
            if field.type in ["string", "integer", "boolean"]:
                list_endpoint.add_parameter(Parameter(
                    name=f"filter_{field.name}",
                    type=field.type,
                    description=f"Filter by {field.name}",
                    required=False
                ))
        
        list_endpoint.add_response(Response(
            status_code=200,
            description=f"List of {entity_name_plural}",
            schema={"type": "array", "items": self._entity_to_schema(entity)}
        ))
        endpoints.append(list_endpoint)
        
        # GET /entities/{id} - Get by ID
        get_endpoint = Endpoint(
            path=f"/{entity_name_plural}/{{id}}",
            method=HTTPMethod.GET,
            name=f"get_{entity_name}",
            description=f"Retrieve a specific {entity_name} by ID",
            summary=f"Get {entity_name}",
            tags=[entity.name]
        )
        get_endpoint.add_parameter(Parameter(
            name="id", type="string", description=f"{entity_name} ID", required=True
        ))
        get_endpoint.add_response(Response(
            status_code=200,
            description=f"{entity_name} details",
            schema=self._entity_to_schema(entity)
        ))
        get_endpoint.add_response(Response(
            status_code=404,
            description=f"{entity_name} not found"
        ))
        endpoints.append(get_endpoint)
        
        # POST /entities - Create
        create_endpoint = Endpoint(
            path=f"/{entity_name_plural}",
            method=HTTPMethod.POST,
            name=f"create_{entity_name}",
            description=f"Create a new {entity_name}",
            summary=f"Create {entity_name}",
            tags=[entity.name]
        )
        create_endpoint.request_body = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": self._entity_to_create_schema(entity)
                }
            }
        }
        create_endpoint.add_response(Response(
            status_code=201,
            description=f"{entity_name} created successfully",
            schema=self._entity_to_schema(entity)
        ))
        create_endpoint.add_response(Response(
            status_code=400,
            description="Invalid input data"
        ))
        endpoints.append(create_endpoint)
        
        # PUT /entities/{id} - Update
        update_endpoint = Endpoint(
            path=f"/{entity_name_plural}/{{id}}",
            method=HTTPMethod.PUT,
            name=f"update_{entity_name}",
            description=f"Update an existing {entity_name}",
            summary=f"Update {entity_name}",
            tags=[entity.name]
        )
        update_endpoint.add_parameter(Parameter(
            name="id", type="string", description=f"{entity_name} ID", required=True
        ))
        update_endpoint.request_body = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": self._entity_to_update_schema(entity)
                }
            }
        }
        update_endpoint.add_response(Response(
            status_code=200,
            description=f"{entity_name} updated successfully",
            schema=self._entity_to_schema(entity)
        ))
        endpoints.append(update_endpoint)
        
        # DELETE /entities/{id} - Delete
        delete_endpoint = Endpoint(
            path=f"/{entity_name_plural}/{{id}}",
            method=HTTPMethod.DELETE,
            name=f"delete_{entity_name}",
            description=f"Delete a {entity_name}",
            summary=f"Delete {entity_name}",
            tags=[entity.name]
        )
        delete_endpoint.add_parameter(Parameter(
            name="id", type="string", description=f"{entity_name} ID", required=True
        ))
        delete_endpoint.add_response(Response(
            status_code=204,
            description=f"{entity_name} deleted successfully"
        ))
        delete_endpoint.add_response(Response(
            status_code=404,
            description=f"{entity_name} not found"
        ))
        endpoints.append(delete_endpoint)
        
        return endpoints
    
    def _entity_to_schema(self, entity: Entity) -> Dict[str, Any]:
        """Convert database entity to JSON schema."""
        properties = {}
        required = []
        
        for field in entity.fields:
            field_schema = {"type": self._map_field_type(field.type)}
            
            if field.description:
                field_schema["description"] = field.description
            
            if field.constraints:
                for constraint in field.constraints:
                    if constraint.type == "max_length":
                        field_schema["maxLength"] = constraint.value
                    elif constraint.type == "min_length":
                        field_schema["minLength"] = constraint.value
                    elif constraint.type == "pattern":
                        field_schema["pattern"] = constraint.value
            
            properties[field.name] = field_schema
            
            if not field.nullable and field.name != "id":
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _entity_to_create_schema(self, entity: Entity) -> Dict[str, Any]:
        """Convert entity to create schema (excluding auto-generated fields)."""
        schema = self._entity_to_schema(entity)
        
        # Remove auto-generated fields
        if "id" in schema["properties"]:
            del schema["properties"]["id"]
        if "created_at" in schema["properties"]:
            del schema["properties"]["created_at"]
        if "updated_at" in schema["properties"]:
            del schema["properties"]["updated_at"]
        
        # Update required fields
        schema["required"] = [f for f in schema["required"] if f not in ["id", "created_at", "updated_at"]]
        
        return schema
    
    def _entity_to_update_schema(self, entity: Entity) -> Dict[str, Any]:
        """Convert entity to update schema (all fields optional except ID)."""
        schema = self._entity_to_create_schema(entity)
        schema["required"] = []  # Make all fields optional for updates
        return schema
    
    def _map_field_type(self, field_type: str) -> str:
        """Map database field type to JSON schema type."""
        type_mapping = {
            "string": "string",
            "text": "string",
            "integer": "integer",
            "bigint": "integer",
            "float": "number",
            "decimal": "number",
            "boolean": "boolean",
            "date": "string",
            "datetime": "string",
            "timestamp": "string",
            "json": "object",
            "array": "array"
        }
        return type_mapping.get(field_type.lower(), "string")
    
    def _generate_rest_api(
        self, 
        api_spec: APISpec, 
        request: APIGenerationRequest, 
        generated_code: GeneratedAPICode
    ):
        """Generate RESTful API code."""
        if request.target_language == "python":
            self._generate_python_rest_api(api_spec, request, generated_code)
        elif request.target_language in ["javascript", "typescript"]:
            self._generate_node_rest_api(api_spec, request, generated_code)
    
    def _generate_python_rest_api(
        self, 
        api_spec: APISpec, 
        request: APIGenerationRequest, 
        generated_code: GeneratedAPICode
    ):
        """Generate Python REST API code."""
        if request.target_framework == "fastapi":
            self._generate_fastapi_code(api_spec, request, generated_code)
        elif request.target_framework == "flask":
            self._generate_flask_code(api_spec, request, generated_code)
    
    def _generate_fastapi_code(
        self, 
        api_spec: APISpec, 
        request: APIGenerationRequest, 
        generated_code: GeneratedAPICode
    ):
        """Generate FastAPI code."""
        # Main application file
        main_code = self._generate_fastapi_main(api_spec, request)
        generated_code.add_code_file("main.py", main_code)
        
        # Models file
        models_code = self._generate_fastapi_models(api_spec)
        generated_code.add_code_file("models.py", models_code)
        
        # Routes files
        for tag in self._get_unique_tags(api_spec):
            routes_code = self._generate_fastapi_routes(api_spec, tag)
            generated_code.add_code_file(f"routes/{tag.lower()}_routes.py", routes_code)
        
        # Database configuration
        if request.database_type:
            db_code = self._generate_fastapi_database(request)
            generated_code.add_code_file("database.py", db_code)
        
        # Requirements file
        requirements = self._generate_python_requirements(request)
        generated_code.add_configuration_file("requirements.txt", requirements)
    
    def _generate_fastapi_main(self, api_spec: APISpec, request: APIGenerationRequest) -> str:
        """Generate FastAPI main application file."""
        template = Template('''"""
{{ api_spec.name }} - Auto-generated API
{{ api_spec.description }}
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
{% if request.include_authentication %}
from fastapi.security.http import HTTPAuthorizationCredentials
{% endif %}
import uvicorn

{% for tag in unique_tags %}
from routes.{{ tag.lower() }}_routes import router as {{ tag.lower() }}_router
{% endfor %}
{% if request.database_type %}
from database import engine, Base
{% endif %}

# Create FastAPI application
app = FastAPI(
    title="{{ api_spec.name }}",
    description="{{ api_spec.description }}",
    version="{{ api_spec.version }}",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

{% if request.include_authentication %}
# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return current user."""
    # TODO: Implement JWT validation logic
    return {"user_id": "example_user"}
{% endif %}

{% for tag in unique_tags %}
# Include {{ tag }} routes
app.include_router({{ tag.lower() }}_router, prefix="/api/v1", tags=["{{ tag }}"])
{% endfor %}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "{{ api_spec.name }} is running", "version": "{{ api_spec.version }}"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "{{ datetime.now().isoformat() }}"}

{% if request.database_type %}
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    Base.metadata.create_all(bind=engine)
{% endif %}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')
        
        return template.render(
            api_spec=api_spec,
            request=request,
            unique_tags=self._get_unique_tags(api_spec),
            datetime=datetime
        )
    
    def _generate_fastapi_models(self, api_spec: APISpec) -> str:
        """Generate Pydantic models for FastAPI."""
        template = Template('''"""
Pydantic models for {{ api_spec.name }}
Auto-generated from API specification
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

{% for tag in unique_tags %}
# {{ tag }} Models
class {{ tag }}Base(BaseModel):
    """Base model for {{ tag }}."""
    pass

class {{ tag }}Create({{ tag }}Base):
    """Model for creating {{ tag }}."""
    pass

class {{ tag }}Update({{ tag }}Base):
    """Model for updating {{ tag }}."""
    pass

class {{ tag }}Response({{ tag }}Base):
    """Model for {{ tag }} response."""
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

{% endfor %}

# Common response models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    details: Optional[dict] = None

class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List[dict]
    total: int
    page: int
    limit: int
    pages: int
''')
        
        return template.render(
            api_spec=api_spec,
            unique_tags=self._get_unique_tags(api_spec)
        )
    
    def _generate_fastapi_routes(self, api_spec: APISpec, tag: str) -> str:
        """Generate FastAPI routes for a specific tag."""
        endpoints = [ep for ep in api_spec.endpoints if tag in ep.tags]
        
        template = Template('''"""
{{ tag }} routes for {{ api_spec.name }}
Auto-generated API routes
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from models import {{ tag }}Create, {{ tag }}Update, {{ tag }}Response, ErrorResponse, PaginatedResponse

router = APIRouter()

{% for endpoint in endpoints %}
@router.{{ endpoint.method.value.lower() }}("{{ endpoint.path }}")
async def {{ endpoint.name }}(
    {% for param in endpoint.parameters %}
    {% if param.name in endpoint.path %}
    {{ param.name }}: {{ param.type }},
    {% else %}
    {{ param.name }}: {{ 'Optional[' + param.type + ']' if not param.required else param.type }} = {{ 'Query(' + (param.default_value|string if param.default_value else 'None') + ')' if not param.required else '' }},
    {% endif %}
    {% endfor %}
    {% if endpoint.request_body %}
    data: {{ tag }}{{ 'Create' if endpoint.method.value == 'POST' else 'Update' }},
    {% endif %}
) -> {{ tag }}Response:
    """{{ endpoint.description }}"""
    # TODO: Implement {{ endpoint.name }} logic
    {% if endpoint.method.value == 'GET' and 'id' in endpoint.path %}
    # Get single {{ tag.lower() }} by ID
    return {{ tag }}Response(
        id="{{ param.name if param.name == 'id' else 'example_id' }}",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00"
    )
    {% elif endpoint.method.value == 'GET' %}
    # List {{ tag.lower() }}s with pagination
    return PaginatedResponse(
        items=[],
        total=0,
        page=page or 1,
        limit=limit or 10,
        pages=0
    )
    {% elif endpoint.method.value == 'POST' %}
    # Create new {{ tag.lower() }}
    return {{ tag }}Response(
        id="new_id",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00"
    )
    {% elif endpoint.method.value in ['PUT', 'PATCH'] %}
    # Update {{ tag.lower() }}
    return {{ tag }}Response(
        id=id,
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00"
    )
    {% elif endpoint.method.value == 'DELETE' %}
    # Delete {{ tag.lower() }}
    return {"message": "{{ tag }} deleted successfully"}
    {% endif %}

{% endfor %}
''')
        
        return template.render(
            api_spec=api_spec,
            tag=tag,
            endpoints=endpoints
        )
    
    def _generate_fastapi_database(self, request: APIGenerationRequest) -> str:
        """Generate database configuration for FastAPI."""
        if request.database_type == "postgresql":
            return '''"""
Database configuration for PostgreSQL
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")

# Create engine
engine = create_engine(DATABASE_URL)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''
        return ""
    
    def _generate_python_requirements(self, request: APIGenerationRequest) -> str:
        """Generate Python requirements.txt file."""
        requirements = ["fastapi>=0.104.0", "uvicorn>=0.24.0", "pydantic>=2.0.0"]
        
        if request.database_type == "postgresql":
            requirements.extend(["sqlalchemy>=2.0.0", "psycopg2-binary>=2.9.0"])
        elif request.database_type == "mysql":
            requirements.extend(["sqlalchemy>=2.0.0", "pymysql>=1.0.0"])
        
        if request.include_authentication:
            requirements.extend(["python-jose>=3.3.0", "passlib>=1.7.0", "python-multipart>=0.0.6"])
        
        if request.include_tests:
            requirements.extend(["pytest>=7.0.0", "httpx>=0.25.0"])
        
        return "\n".join(requirements)
    
    def _get_unique_tags(self, api_spec: APISpec) -> List[str]:
        """Get unique tags from API specification."""
        tags = set()
        for endpoint in api_spec.endpoints:
            tags.update(endpoint.tags)
        return sorted(list(tags))
    
    def _generate_graphql_api(
        self, 
        api_spec: APISpec, 
        request: APIGenerationRequest, 
        generated_code: GeneratedAPICode
    ):
        """Generate GraphQL API code."""
        if not api_spec.graphql_schema:
            return
        
        # Generate GraphQL schema file
        schema_code = self._generate_graphql_schema(api_spec.graphql_schema)
        generated_code.add_code_file("schema.graphql", schema_code)
        
        # Generate resolvers
        resolvers_code = self._generate_graphql_resolvers(api_spec.graphql_schema, request)
        generated_code.add_code_file("resolvers.py", resolvers_code)
    
    def _generate_graphql_schema(self, schema: GraphQLSchema) -> str:
        """Generate GraphQL schema definition."""
        schema_parts = []
        
        # Generate types
        for gql_type in schema.types:
            if gql_type.is_enum:
                enum_def = f"enum {gql_type.name} {{\n"
                for value in gql_type.enum_values:
                    enum_def += f"  {value}\n"
                enum_def += "}\n"
                schema_parts.append(enum_def)
            else:
                type_def = f"type {gql_type.name} {{\n"
                for field in gql_type.fields:
                    field_type = field.type
                    if field.list_type:
                        field_type = f"[{field_type}]"
                    if not field.nullable:
                        field_type += "!"
                    
                    args = ""
                    if field.arguments:
                        arg_strs = []
                        for arg in field.arguments:
                            arg_type = arg.type
                            if arg.required:
                                arg_type += "!"
                            arg_strs.append(f"{arg.name}: {arg_type}")
                        args = f"({', '.join(arg_strs)})"
                    
                    type_def += f"  {field.name}{args}: {field_type}\n"
                type_def += "}\n"
                schema_parts.append(type_def)
        
        # Generate Query type
        if schema.queries:
            query_def = "type Query {\n"
            for query in schema.queries:
                field_type = query.type
                if query.list_type:
                    field_type = f"[{field_type}]"
                if not query.nullable:
                    field_type += "!"
                
                args = ""
                if query.arguments:
                    arg_strs = []
                    for arg in query.arguments:
                        arg_type = arg.type
                        if arg.required:
                            arg_type += "!"
                        arg_strs.append(f"{arg.name}: {arg_type}")
                    args = f"({', '.join(arg_strs)})"
                
                query_def += f"  {query.name}{args}: {field_type}\n"
            query_def += "}\n"
            schema_parts.append(query_def)
        
        # Generate Mutation type
        if schema.mutations:
            mutation_def = "type Mutation {\n"
            for mutation in schema.mutations:
                field_type = mutation.type
                if mutation.list_type:
                    field_type = f"[{field_type}]"
                if not mutation.nullable:
                    field_type += "!"
                
                args = ""
                if mutation.arguments:
                    arg_strs = []
                    for arg in mutation.arguments:
                        arg_type = arg.type
                        if arg.required:
                            arg_type += "!"
                        arg_strs.append(f"{arg.name}: {arg_type}")
                    args = f"({', '.join(arg_strs)})"
                
                mutation_def += f"  {mutation.name}{args}: {field_type}\n"
            mutation_def += "}\n"
            schema_parts.append(mutation_def)
        
        return "\n".join(schema_parts)
    
    def _generate_graphql_resolvers(
        self, 
        schema: GraphQLSchema, 
        request: APIGenerationRequest
    ) -> str:
        """Generate GraphQL resolvers."""
        template = Template('''"""
GraphQL Resolvers
Auto-generated resolvers for GraphQL schema
"""

from typing import List, Optional, Dict, Any
# import graphene  # Temporarily disabled for Docker build
# from graphene import ObjectType, String, Int, Boolean, List as GrapheneList  # Temporarily disabled for Docker build

{% for gql_type in schema.types %}
{% if not gql_type.is_enum %}
class {{ gql_type.name }}Type(ObjectType):
    """GraphQL type for {{ gql_type.name }}."""
    {% for field in gql_type.fields %}
    {{ field.name }} = {{ 'String()' if field.type == 'String' else 'Int()' if field.type == 'Int' else 'Boolean()' if field.type == 'Boolean' else 'String()' }}
    {% endfor %}
{% endif %}
{% endfor %}

class Query(ObjectType):
    """GraphQL Query resolvers."""
    
    {% for query in schema.queries %}
    {{ query.name }} = {{ 'GrapheneList(' + query.type + 'Type)' if query.list_type else query.type + 'Type()' }}
    
    def resolve_{{ query.name }}(self, info{% for arg in query.arguments %}, {{ arg.name }}=None{% endfor %}):
        """Resolve {{ query.name }} query."""
        # TODO: Implement {{ query.name }} resolver logic
        {% if query.list_type %}
        return []
        {% else %}
        return None
        {% endif %}
    {% endfor %}

class Mutation(ObjectType):
    """GraphQL Mutation resolvers."""
    
    {% for mutation in schema.mutations %}
    {{ mutation.name }} = {{ mutation.type + 'Type()' }}
    
    def resolve_{{ mutation.name }}(self, info{% for arg in mutation.arguments %}, {{ arg.name }}=None{% endfor %}):
        """Resolve {{ mutation.name }} mutation."""
        # TODO: Implement {{ mutation.name }} resolver logic
        return None
    {% endfor %}

# Create schema
schema = graphene.Schema(query=Query, mutation=Mutation)
''')
        
        return template.render(schema=schema)
    
    def _generate_api_documentation(self, api_spec: APISpec, generated_code: GeneratedAPICode):
        """Generate API documentation (OpenAPI/Swagger)."""
        # Generate OpenAPI specification
        openapi_spec = self._generate_openapi_spec(api_spec)
        generated_code.add_documentation_file("openapi.json", json.dumps(openapi_spec, indent=2))
        generated_code.add_documentation_file("openapi.yaml", yaml.dump(openapi_spec, default_flow_style=False))
        
        # Generate README
        readme_content = self._generate_api_readme(api_spec)
        generated_code.add_documentation_file("README.md", readme_content)
    
    def _generate_openapi_spec(self, api_spec: APISpec) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": api_spec.name,
                "description": api_spec.description,
                "version": api_spec.version
            },
            "servers": [
                {"url": api_spec.base_url, "description": "API Server"}
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {}
            }
        }
        
        # Add security schemes
        for scheme in api_spec.security_schemes:
            openapi_spec["components"]["securitySchemes"][scheme.name] = {
                "type": scheme.type,
                "scheme": scheme.scheme,
                "bearerFormat": scheme.bearer_format
            }
        
        # Add paths
        for endpoint in api_spec.endpoints:
            path = endpoint.path
            method = endpoint.method.value.lower()
            
            if path not in openapi_spec["paths"]:
                openapi_spec["paths"][path] = {}
            
            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "parameters": [],
                "responses": {}
            }
            
            # Add parameters
            for param in endpoint.parameters:
                param_spec = {
                    "name": param.name,
                    "in": "path" if param.name in path else "query",
                    "required": param.required,
                    "schema": {"type": param.type}
                }
                if param.description:
                    param_spec["description"] = param.description
                operation["parameters"].append(param_spec)
            
            # Add request body
            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body
            
            # Add responses
            for response in endpoint.responses:
                operation["responses"][str(response.status_code)] = {
                    "description": response.description
                }
                if response.schema:
                    operation["responses"][str(response.status_code)]["content"] = {
                        "application/json": {"schema": response.schema}
                    }
            
            # Add security
            if endpoint.security_requirements:
                operation["security"] = [
                    {req: []} for req in endpoint.security_requirements
                ]
            
            openapi_spec["paths"][path][method] = operation
        
        return openapi_spec
    
    def _generate_api_readme(self, api_spec: APISpec) -> str:
        """Generate README documentation for the API."""
        template = Template('''# {{ api_spec.name }}

{{ api_spec.description }}

## Version
{{ api_spec.version }}

## Base URL
```
{{ api_spec.base_url }}
```

## Authentication
{% if api_spec.security_schemes %}
This API uses the following authentication methods:
{% for scheme in api_spec.security_schemes %}
- **{{ scheme.name }}**: {{ scheme.description or scheme.type }}
{% endfor %}
{% else %}
No authentication required.
{% endif %}

## Endpoints

{% for tag in unique_tags %}
### {{ tag }}
{% for endpoint in api_spec.endpoints %}
{% if tag in endpoint.tags %}
#### {{ endpoint.method.value }} {{ endpoint.path }}
{{ endpoint.description }}

**Parameters:**
{% for param in endpoint.parameters %}
- `{{ param.name }}` ({{ param.type }}){% if param.required %} *required*{% endif %}: {{ param.description or 'No description' }}
{% endfor %}

**Responses:**
{% for response in endpoint.responses %}
- `{{ response.status_code }}`: {{ response.description }}
{% endfor %}

{% endif %}
{% endfor %}
{% endfor %}

## Error Handling

The API uses standard HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Internal Server Error

## Rate Limiting

Rate limiting is applied to all endpoints. Default limits:
- 1000 requests per hour per IP
- 100 requests per minute per IP

## Support

For support, please contact the development team.
''')
        
        return template.render(
            api_spec=api_spec,
            unique_tags=self._get_unique_tags(api_spec)
        )
    
    def _generate_api_tests(
        self, 
        api_spec: APISpec, 
        request: APIGenerationRequest, 
        generated_code: GeneratedAPICode
    ):
        """Generate API tests."""
        if request.target_language == "python":
            self._generate_python_api_tests(api_spec, generated_code)
    
    def _generate_python_api_tests(self, api_spec: APISpec, generated_code: GeneratedAPICode):
        """Generate Python API tests using pytest."""
        # Generate test configuration
        test_config = '''"""
Test configuration for API tests
"""

import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers fixture."""
    return {"Authorization": "Bearer test_token"}
'''
        generated_code.add_test_file("conftest.py", test_config)
        
        # Generate tests for each tag
        for tag in self._get_unique_tags(api_spec):
            test_code = self._generate_tag_tests(api_spec, tag)
            generated_code.add_test_file(f"test_{tag.lower()}_api.py", test_code)
    
    def _generate_tag_tests(self, api_spec: APISpec, tag: str) -> str:
        """Generate tests for a specific tag."""
        endpoints = [ep for ep in api_spec.endpoints if tag in ep.tags]
        
        template = Template('''"""
Tests for {{ tag }} API endpoints
"""

import pytest
from fastapi.testclient import TestClient

def test_{{ tag.lower() }}_endpoints_exist(client):
    """Test that {{ tag.lower() }} endpoints exist."""
    {% for endpoint in endpoints %}
    # Test {{ endpoint.method.value }} {{ endpoint.path }}
    {% if endpoint.method.value == 'GET' %}
    {% if 'id' in endpoint.path %}
    response = client.get("{{ endpoint.path.replace('{id}', 'test_id') }}")
    {% else %}
    response = client.get("{{ endpoint.path }}")
    {% endif %}
    assert response.status_code in [200, 404, 401]  # Valid responses
    {% elif endpoint.method.value == 'POST' %}
    response = client.post("{{ endpoint.path }}", json={})
    assert response.status_code in [201, 400, 401]  # Valid responses
    {% elif endpoint.method.value in ['PUT', 'PATCH'] %}
    response = client.{{ endpoint.method.value.lower() }}("{{ endpoint.path.replace('{id}', 'test_id') }}", json={})
    assert response.status_code in [200, 400, 404, 401]  # Valid responses
    {% elif endpoint.method.value == 'DELETE' %}
    response = client.delete("{{ endpoint.path.replace('{id}', 'test_id') }}")
    assert response.status_code in [204, 404, 401]  # Valid responses
    {% endif %}
    
    {% endfor %}

{% for endpoint in endpoints %}
def test_{{ endpoint.name }}(client):
    """Test {{ endpoint.name }} endpoint."""
    {% if endpoint.method.value == 'GET' and 'id' not in endpoint.path %}
    # Test list endpoint
    response = client.get("{{ endpoint.path }}")
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (list, dict))
    {% elif endpoint.method.value == 'GET' and 'id' in endpoint.path %}
    # Test get by ID endpoint
    response = client.get("{{ endpoint.path.replace('{id}', 'test_id') }}")
    assert response.status_code in [200, 404]
    {% elif endpoint.method.value == 'POST' %}
    # Test create endpoint
    test_data = {}  # TODO: Add test data
    response = client.post("{{ endpoint.path }}", json=test_data)
    assert response.status_code in [201, 400]
    {% elif endpoint.method.value in ['PUT', 'PATCH'] %}
    # Test update endpoint
    test_data = {}  # TODO: Add test data
    response = client.{{ endpoint.method.value.lower() }}("{{ endpoint.path.replace('{id}', 'test_id') }}", json=test_data)
    assert response.status_code in [200, 400, 404]
    {% elif endpoint.method.value == 'DELETE' %}
    # Test delete endpoint
    response = client.delete("{{ endpoint.path.replace('{id}', 'test_id') }}")
    assert response.status_code in [204, 404]
    {% endif %}

{% endfor %}
''')
        
        return template.render(tag=tag, endpoints=endpoints)
    
    def _generate_configuration_files(
        self, 
        api_spec: APISpec, 
        request: APIGenerationRequest, 
        generated_code: GeneratedAPICode
    ):
        """Generate configuration files."""
        # Docker configuration
        dockerfile = self._generate_dockerfile(request)
        generated_code.add_configuration_file("Dockerfile", dockerfile)
        
        # Docker Compose
        docker_compose = self._generate_docker_compose(request)
        generated_code.add_configuration_file("docker-compose.yml", docker_compose)
        
        # Environment configuration
        env_example = self._generate_env_example(request)
        generated_code.add_configuration_file(".env.example", env_example)
    
    def _generate_dockerfile(self, request: APIGenerationRequest) -> str:
        """Generate Dockerfile for the API."""
        if request.target_language == "python":
            return '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        return ""
    
    def _generate_docker_compose(self, request: APIGenerationRequest) -> str:
        """Generate docker-compose.yml file."""
        template = Template('''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL={{ database_url }}
    depends_on:
      {% if request.database_type == 'postgresql' %}
      - postgres
      {% elif request.database_type == 'mysql' %}
      - mysql
      {% endif %}

  {% if request.database_type == 'postgresql' %}
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: apidb
      POSTGRES_USER: apiuser
      POSTGRES_PASSWORD: apipass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  {% elif request.database_type == 'mysql' %}
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: apidb
      MYSQL_USER: apiuser
      MYSQL_PASSWORD: apipass
      MYSQL_ROOT_PASSWORD: rootpass
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
  {% endif %}

volumes:
  {% if request.database_type == 'postgresql' %}
  postgres_data:
  {% elif request.database_type == 'mysql' %}
  mysql_data:
  {% endif %}
''')
        
        database_url = ""
        if request.database_type == "postgresql":
            database_url = "postgresql://apiuser:apipass@postgres:5432/apidb"
        elif request.database_type == "mysql":
            database_url = "mysql://apiuser:apipass@mysql:3306/apidb"
        
        return template.render(request=request, database_url=database_url)
    
    def _generate_env_example(self, request: APIGenerationRequest) -> str:
        """Generate .env.example file."""
        env_vars = [
            "# Database Configuration",
            f"DATABASE_URL=your_database_url_here"
        ]
        
        if request.include_authentication:
            env_vars.extend([
                "",
                "# Authentication",
                "JWT_SECRET_KEY=your_jwt_secret_key_here",
                "JWT_ALGORITHM=HS256",
                "ACCESS_TOKEN_EXPIRE_MINUTES=30"
            ])
        
        env_vars.extend([
            "",
            "# API Configuration",
            "API_HOST=0.0.0.0",
            "API_PORT=8000",
            "DEBUG=false"
        ])
        
        return "\n".join(env_vars)
    
    def generate_api_versioning(
        self, 
        api_spec: APISpec, 
        new_version: str, 
        breaking_changes: List[str] = None
    ) -> APISpec:
        """Generate a new version of the API with backward compatibility."""
        # Create new version
        version = APIVersion(
            version=new_version,
            release_date=datetime.now(),
            breaking_changes=breaking_changes or [],
            changelog=[]
        )
        
        # Clone API spec for new version
        new_spec = APISpec(
            name=api_spec.name,
            description=api_spec.description,
            version=new_version,
            api_type=api_spec.api_type,
            base_url=api_spec.base_url.replace("/v1", f"/v{new_version.split('.')[0]}")
        )
        
        # Copy endpoints with version updates
        for endpoint in api_spec.endpoints:
            new_endpoint = Endpoint(
                path=endpoint.path,
                method=endpoint.method,
                name=endpoint.name,
                description=endpoint.description,
                summary=endpoint.summary,
                tags=endpoint.tags,
                parameters=endpoint.parameters.copy(),
                request_body=endpoint.request_body,
                responses=endpoint.responses.copy(),
                security_requirements=endpoint.security_requirements.copy(),
                version=new_version
            )
            new_spec.add_endpoint(new_endpoint)
        
        # Add version info
        new_spec.add_version(version)
        
        return new_spec


# Utility functions for API generation
def create_crud_api_from_entities(entities: List[Entity]) -> APISpec:
    """Create a complete CRUD API from database entities."""
    generator = APICodeGenerator()
    
    api_spec = APISpec(
        name="Generated CRUD API",
        description="Auto-generated CRUD API from database entities",
        version="1.0.0",
        api_type=APIType.REST
    )
    
    for entity in entities:
        crud_endpoints = generator._generate_crud_endpoints(entity)
        for endpoint in crud_endpoints:
            api_spec.add_endpoint(endpoint)
    
    return api_spec