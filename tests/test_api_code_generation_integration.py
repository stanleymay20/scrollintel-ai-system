"""
Integration tests for API Code Generation System

This module contains comprehensive integration tests for the API code generation
system, including REST API generation, GraphQL generation, documentation, and versioning.
"""

import pytest
import json
import yaml
from typing import Dict, List, Any
from datetime import datetime

from scrollintel.models.api_generation_models import (
    APISpec, Endpoint, Parameter, Response, HTTPMethod, APIType,
    SecurityScheme, APIGenerationRequest, GeneratedAPICode,
    GraphQLSchema, GraphQLType, GraphQLField
)
from scrollintel.models.database_schema_models import (
    DatabaseSchema, Entity, Field, Relationship, Constraint
)
from scrollintel.engines.api_code_generator import APICodeGenerator
from scrollintel.engines.graphql_generator import GraphQLGenerator
from scrollintel.engines.api_documentation_generator import APIDocumentationGenerator
from scrollintel.engines.api_versioning_manager import APIVersioningManager, ChangeType, APIChange


class TestAPICodeGenerationIntegration:
    """Integration tests for API code generation."""
    
    @pytest.fixture
    def sample_database_schema(self):
        """Create a sample database schema for testing."""
        schema = DatabaseSchema(
            name="BlogSystem",
            description="A simple blog system database"
        )
        
        # User entity
        user_entity = Entity(
            name="User",
            description="Blog user entity"
        )
        user_entity.fields = [
            Field(name="id", type="uuid", nullable=False, primary_key=True),
            Field(name="username", type="string", nullable=False),
            Field(name="email", type="string", nullable=False),
            Field(name="password_hash", type="string", nullable=False),
            Field(name="created_at", type="datetime", nullable=False),
            Field(name="updated_at", type="datetime", nullable=False)
        ]
        
        # Add constraints
        user_entity.fields[1].constraints = [
            Constraint(type="max_length", value=50),
            Constraint(type="unique", value=True)
        ]
        user_entity.fields[2].constraints = [
            Constraint(type="unique", value=True)
        ]
        
        # Post entity
        post_entity = Entity(
            name="Post",
            description="Blog post entity"
        )
        post_entity.fields = [
            Field(name="id", type="uuid", nullable=False, primary_key=True),
            Field(name="title", type="string", nullable=False),
            Field(name="content", type="text", nullable=False),
            Field(name="author_id", type="uuid", nullable=False),
            Field(name="published", type="boolean", nullable=False, default_value=False),
            Field(name="created_at", type="datetime", nullable=False),
            Field(name="updated_at", type="datetime", nullable=False)
        ]
        
        # Add entities to schema
        schema.entities = [user_entity, post_entity]
        
        # Add relationships
        schema.relationships = [
            Relationship(
                name="user_posts",
                type="one_to_many",
                source_entity="User",
                target_entity="Post",
                source_field="id",
                target_field="author_id"
            )
        ]
        
        return schema
    
    @pytest.fixture
    def api_generation_request(self):
        """Create a sample API generation request."""
        return APIGenerationRequest(
            target_language="python",
            target_framework="fastapi",
            include_tests=True,
            include_documentation=True,
            include_validation=True,
            include_authentication=True,
            database_type="postgresql",
            orm_type="sqlalchemy"
        )
    
    def test_complete_api_generation_workflow(
        self, 
        sample_database_schema, 
        api_generation_request
    ):
        """Test complete API generation workflow from database schema."""
        generator = APICodeGenerator()
        
        # Generate API from database schema
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Verify generated code structure
        assert isinstance(generated_code, GeneratedAPICode)
        assert generated_code.language == "python"
        assert generated_code.framework == "fastapi"
        
        # Check that main files are generated
        assert "main.py" in generated_code.code_files
        assert "models.py" in generated_code.code_files
        assert "database.py" in generated_code.code_files
        
        # Check route files
        assert "routes/user_routes.py" in generated_code.code_files
        assert "routes/post_routes.py" in generated_code.code_files
        
        # Check configuration files
        assert "requirements.txt" in generated_code.configuration_files
        assert "Dockerfile" in generated_code.configuration_files
        assert "docker-compose.yml" in generated_code.configuration_files
        
        # Check test files
        assert "conftest.py" in generated_code.test_files
        assert "test_user_api.py" in generated_code.test_files
        assert "test_post_api.py" in generated_code.test_files
        
        # Check documentation files
        assert "README.md" in generated_code.documentation_files
        assert "openapi.json" in generated_code.documentation_files
    
    def test_fastapi_code_generation_quality(
        self, 
        sample_database_schema, 
        api_generation_request
    ):
        """Test quality of generated FastAPI code."""
        generator = APICodeGenerator()
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Check main.py content
        main_code = generated_code.code_files["main.py"]
        assert "from fastapi import FastAPI" in main_code
        assert "app = FastAPI(" in main_code
        assert "include_router" in main_code
        assert "uvicorn.run" in main_code
        
        # Check models.py content
        models_code = generated_code.code_files["models.py"]
        assert "from pydantic import BaseModel" in models_code
        assert "class UserBase(BaseModel)" in models_code
        assert "class PostBase(BaseModel)" in models_code
        
        # Check route files
        user_routes = generated_code.code_files["routes/user_routes.py"]
        assert "@router.get" in user_routes
        assert "@router.post" in user_routes
        assert "@router.put" in user_routes
        assert "@router.delete" in user_routes
        assert "async def" in user_routes
        
        # Check database configuration
        db_code = generated_code.code_files["database.py"]
        assert "from sqlalchemy import create_engine" in db_code
        assert "postgresql://" in db_code
        assert "SessionLocal" in db_code
    
    def test_graphql_generation_integration(self, sample_database_schema):
        """Test GraphQL API generation integration."""
        generator = GraphQLGenerator()
        request = APIGenerationRequest(
            target_language="python",
            target_framework="graphene",
            additional_options={"include_subscriptions": True}
        )
        
        # Generate GraphQL schema
        graphql_schema = generator.generate_graphql_from_schema(
            sample_database_schema, 
            request
        )
        
        # Verify schema structure
        assert isinstance(graphql_schema, GraphQLSchema)
        assert len(graphql_schema.types) >= 4  # User, Post, UserCreateInput, PostCreateInput
        assert len(graphql_schema.queries) >= 4  # user, users, post, posts
        assert len(graphql_schema.mutations) >= 6  # create, update, delete for each entity
        assert len(graphql_schema.subscriptions) >= 6  # created, updated, deleted for each entity
        
        # Check type generation
        type_names = [t.name for t in graphql_schema.types]
        assert "User" in type_names
        assert "Post" in type_names
        assert "UserCreateInput" in type_names
        assert "PostCreateInput" in type_names
        
        # Check query generation
        query_names = [q.name for q in graphql_schema.queries]
        assert "user" in query_names
        assert "users" in query_names
        assert "post" in query_names
        assert "posts" in query_names
        
        # Check mutation generation
        mutation_names = [m.name for m in graphql_schema.mutations]
        assert "create_user" in mutation_names
        assert "update_user" in mutation_names
        assert "delete_user" in mutation_names
    
    def test_api_documentation_generation(self, sample_database_schema, api_generation_request):
        """Test API documentation generation."""
        generator = APICodeGenerator()
        doc_generator = APIDocumentationGenerator()
        
        # Generate API specification
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Create API spec for documentation
        api_spec = generator._create_api_spec_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Generate OpenAPI specification
        openapi_spec = doc_generator.generate_openapi_specification(api_spec)
        
        # Verify OpenAPI structure
        assert openapi_spec["openapi"] == "3.0.3"
        assert openapi_spec["info"]["title"] == api_spec.name
        assert "paths" in openapi_spec
        assert "components" in openapi_spec
        
        # Check paths
        paths = openapi_spec["paths"]
        assert "/users" in paths
        assert "/users/{id}" in paths
        assert "/posts" in paths
        assert "/posts/{id}" in paths
        
        # Check HTTP methods
        assert "get" in paths["/users"]
        assert "post" in paths["/users"]
        assert "get" in paths["/users/{id}"]
        assert "put" in paths["/users/{id}"]
        assert "delete" in paths["/users/{id}"]
        
        # Generate Swagger UI HTML
        swagger_html = doc_generator.generate_swagger_ui_html(api_spec, openapi_spec)
        assert "swagger-ui" in swagger_html
        assert api_spec.name in swagger_html
        
        # Generate Postman collection
        postman_collection = doc_generator.generate_postman_collection(api_spec)
        assert postman_collection["info"]["name"] == api_spec.name
        assert len(postman_collection["item"]) >= 2  # User and Post folders
    
    def test_api_versioning_integration(self, sample_database_schema, api_generation_request):
        """Test API versioning and backward compatibility."""
        generator = APICodeGenerator()
        versioning_manager = APIVersioningManager()
        
        # Generate initial API version
        api_spec_v1 = generator._create_api_spec_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        api_spec_v1.version = "1.0.0"
        
        # Create modified schema for v2
        modified_schema = sample_database_schema
        
        # Add a new field to User entity (non-breaking change)
        user_entity = next(e for e in modified_schema.entities if e.name == "User")
        user_entity.fields.append(
            Field(name="bio", type="text", nullable=True, description="User biography")
        )
        
        # Generate v2 API
        api_spec_v2 = generator._create_api_spec_from_schema(
            modified_schema, 
            api_generation_request
        )
        api_spec_v2.version = "2.0.0"
        
        # Compare versions
        compatibility_report = versioning_manager.compare_api_versions(
            api_spec_v1, 
            api_spec_v2
        )
        
        # Verify compatibility report
        assert compatibility_report.old_version == "1.0.0"
        assert compatibility_report.new_version == "2.0.0"
        # Should be backward compatible since we only added optional fields
        assert compatibility_report.is_backward_compatible
        
        # Check migration guide generation
        migration_guide = compatibility_report.migration_guide
        assert "Migration Guide" in migration_guide
        assert "1.0.0 → 2.0.0" in migration_guide
    
    def test_crud_operations_generation(self, sample_database_schema, api_generation_request):
        """Test CRUD operations generation for entities."""
        generator = APICodeGenerator()
        
        # Generate API specification
        api_spec = generator._create_api_spec_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Check User CRUD endpoints
        user_endpoints = [ep for ep in api_spec.endpoints if "User" in ep.tags]
        user_paths = [ep.path for ep in user_endpoints]
        user_methods = [ep.method for ep in user_endpoints]
        
        assert "/users" in user_paths
        assert "/users/{id}" in user_paths
        assert HTTPMethod.GET in user_methods
        assert HTTPMethod.POST in user_methods
        assert HTTPMethod.PUT in user_methods
        assert HTTPMethod.DELETE in user_methods
        
        # Check Post CRUD endpoints
        post_endpoints = [ep for ep in api_spec.endpoints if "Post" in ep.tags]
        post_paths = [ep.path for ep in post_endpoints]
        post_methods = [ep.method for ep in post_endpoints]
        
        assert "/posts" in post_paths
        assert "/posts/{id}" in post_paths
        assert HTTPMethod.GET in post_methods
        assert HTTPMethod.POST in post_methods
        assert HTTPMethod.PUT in post_methods
        assert HTTPMethod.DELETE in post_methods
    
    def test_authentication_integration(self, sample_database_schema):
        """Test authentication integration in generated APIs."""
        generator = APICodeGenerator()
        
        # Request with authentication
        request_with_auth = APIGenerationRequest(
            target_language="python",
            target_framework="fastapi",
            include_authentication=True
        )
        
        # Generate API
        api_spec = generator._create_api_spec_from_schema(
            sample_database_schema, 
            request_with_auth
        )
        
        # Check security schemes
        assert len(api_spec.security_schemes) > 0
        auth_scheme = api_spec.security_schemes[0]
        assert auth_scheme.type == "http"
        assert auth_scheme.scheme == "bearer"
        
        # Generate code
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            request_with_auth
        )
        
        # Check authentication in main.py
        main_code = generated_code.code_files["main.py"]
        assert "HTTPBearer" in main_code
        assert "get_current_user" in main_code
        assert "Authorization" in main_code
    
    def test_validation_rules_integration(self, sample_database_schema, api_generation_request):
        """Test validation rules integration in generated APIs."""
        generator = APICodeGenerator()
        
        # Generate API with validation
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Check models for validation
        models_code = generated_code.code_files["models.py"]
        assert "from pydantic import BaseModel, Field" in models_code
        
        # Check that constraints are applied
        # Username should have max length constraint
        user_routes = generated_code.code_files["routes/user_routes.py"]
        assert "HTTPException" in user_routes  # For validation errors
    
    def test_error_handling_integration(self, sample_database_schema, api_generation_request):
        """Test error handling in generated APIs."""
        generator = APICodeGenerator()
        
        # Generate API
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Check error handling in routes
        user_routes = generated_code.code_files["routes/user_routes.py"]
        assert "HTTPException" in user_routes
        
        # Check error models
        models_code = generated_code.code_files["models.py"]
        assert "ErrorResponse" in models_code
    
    def test_database_integration_options(self, sample_database_schema):
        """Test different database integration options."""
        generator = APICodeGenerator()
        
        # Test PostgreSQL
        pg_request = APIGenerationRequest(
            target_language="python",
            target_framework="fastapi",
            database_type="postgresql",
            orm_type="sqlalchemy"
        )
        
        pg_code = generator.generate_api_from_schema(sample_database_schema, pg_request)
        pg_db_code = pg_code.code_files["database.py"]
        assert "postgresql://" in pg_db_code
        assert "psycopg2" in pg_code.configuration_files["requirements.txt"]
        
        # Test MySQL
        mysql_request = APIGenerationRequest(
            target_language="python",
            target_framework="fastapi",
            database_type="mysql",
            orm_type="sqlalchemy"
        )
        
        mysql_code = generator.generate_api_from_schema(sample_database_schema, mysql_request)
        mysql_requirements = mysql_code.configuration_files["requirements.txt"]
        assert "pymysql" in mysql_requirements
    
    def test_api_client_generation(self, sample_database_schema, api_generation_request):
        """Test API client code generation."""
        generator = APICodeGenerator()
        doc_generator = APIDocumentationGenerator()
        
        # Generate API specification
        api_spec = generator._create_api_spec_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Generate Python client
        python_client = doc_generator.generate_api_client_code(api_spec, "python")
        
        assert "client.py" in python_client
        client_code = python_client["client.py"]
        
        # Check client structure
        assert "class" in client_code
        assert "Client" in client_code
        assert "def __init__" in client_code
        assert "requests" in client_code
        assert "Bearer" in client_code
        
        # Check that methods are generated for endpoints
        assert "def list_users" in client_code or "def get_users" in client_code
        assert "def create_user" in client_code
    
    def test_comprehensive_test_generation(self, sample_database_schema, api_generation_request):
        """Test comprehensive test generation for APIs."""
        generator = APICodeGenerator()
        
        # Generate API with tests
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Check test configuration
        assert "conftest.py" in generated_code.test_files
        conftest_code = generated_code.test_files["conftest.py"]
        assert "TestClient" in conftest_code
        assert "@pytest.fixture" in conftest_code
        
        # Check entity-specific tests
        user_test_code = generated_code.test_files["test_user_api.py"]
        assert "def test_" in user_test_code
        assert "client.get" in user_test_code
        assert "client.post" in user_test_code
        assert "assert response.status_code" in user_test_code
        
        post_test_code = generated_code.test_files["test_post_api.py"]
        assert "def test_" in post_test_code
        assert "client" in post_test_code
    
    def test_docker_configuration_generation(self, sample_database_schema, api_generation_request):
        """Test Docker configuration generation."""
        generator = APICodeGenerator()
        
        # Generate API with Docker config
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Check Dockerfile
        assert "Dockerfile" in generated_code.configuration_files
        dockerfile = generated_code.configuration_files["Dockerfile"]
        assert "FROM python:" in dockerfile
        assert "COPY requirements.txt" in dockerfile
        assert "pip install" in dockerfile
        assert "uvicorn" in dockerfile
        
        # Check docker-compose.yml
        assert "docker-compose.yml" in generated_code.configuration_files
        docker_compose = generated_code.configuration_files["docker-compose.yml"]
        assert "version:" in docker_compose
        assert "services:" in docker_compose
        assert "postgres:" in docker_compose  # Database service
        
        # Check environment configuration
        assert ".env.example" in generated_code.configuration_files
        env_example = generated_code.configuration_files[".env.example"]
        assert "DATABASE_URL" in env_example
        assert "JWT_SECRET_KEY" in env_example  # Since auth is enabled
    
    def test_api_specification_validation(self, sample_database_schema, api_generation_request):
        """Test API specification validation."""
        generator = APICodeGenerator()
        
        # Generate API specification
        api_spec = generator._create_api_spec_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Validate basic structure
        assert api_spec.name
        assert api_spec.version
        assert api_spec.base_url
        assert len(api_spec.endpoints) > 0
        
        # Validate endpoints
        for endpoint in api_spec.endpoints:
            assert endpoint.path
            assert endpoint.method
            assert endpoint.name
            assert len(endpoint.responses) > 0
            
            # Check that all path parameters are defined
            path_params = [p for p in endpoint.parameters if f"{{{p.name}}}" in endpoint.path]
            for param in path_params:
                assert param.required  # Path parameters must be required
        
        # Validate responses
        for endpoint in api_spec.endpoints:
            for response in endpoint.responses:
                assert response.status_code
                assert response.description
    
    def test_performance_considerations(self, sample_database_schema, api_generation_request):
        """Test performance considerations in generated code."""
        generator = APICodeGenerator()
        
        # Generate API
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Check for pagination in list endpoints
        user_routes = generated_code.code_files["routes/user_routes.py"]
        assert "limit" in user_routes  # Pagination parameter
        assert "page" in user_routes or "offset" in user_routes
        
        # Check for database session management
        db_code = generated_code.code_files["database.py"]
        assert "get_db" in db_code  # Database dependency
        assert "SessionLocal" in db_code
        assert "try:" in db_code and "finally:" in db_code  # Proper session cleanup
    
    @pytest.mark.asyncio
    async def test_generated_api_execution(self, sample_database_schema, api_generation_request):
        """Test that generated API code can be executed (syntax validation)."""
        generator = APICodeGenerator()
        
        # Generate API
        generated_code = generator.generate_api_from_schema(
            sample_database_schema, 
            api_generation_request
        )
        
        # Test that main.py can be compiled
        main_code = generated_code.code_files["main.py"]
        try:
            compile(main_code, "main.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated main.py has syntax errors: {e}")
        
        # Test that models.py can be compiled
        models_code = generated_code.code_files["models.py"]
        try:
            compile(models_code, "models.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated models.py has syntax errors: {e}")
        
        # Test route files
        for filename, code in generated_code.code_files.items():
            if filename.startswith("routes/"):
                try:
                    compile(code, filename, "exec")
                except SyntaxError as e:
                    pytest.fail(f"Generated {filename} has syntax errors: {e}")


class TestAPIGenerationEdgeCases:
    """Test edge cases and error conditions in API generation."""
    
    def test_empty_database_schema(self):
        """Test API generation with empty database schema."""
        generator = APICodeGenerator()
        empty_schema = DatabaseSchema(name="Empty", description="Empty schema")
        
        request = APIGenerationRequest(
            target_language="python",
            target_framework="fastapi"
        )
        
        # Should not fail, but generate minimal API
        generated_code = generator.generate_api_from_schema(empty_schema, request)
        
        assert isinstance(generated_code, GeneratedAPICode)
        assert "main.py" in generated_code.code_files
        
        # Main should still have basic structure
        main_code = generated_code.code_files["main.py"]
        assert "FastAPI" in main_code
        assert "app = FastAPI" in main_code
    
    def test_complex_relationships(self):
        """Test API generation with complex entity relationships."""
        schema = DatabaseSchema(name="Complex", description="Complex relationships")
        
        # Create entities with many-to-many relationships
        user_entity = Entity(name="User")
        user_entity.fields = [
            Field(name="id", type="uuid", primary_key=True),
            Field(name="name", type="string")
        ]
        
        role_entity = Entity(name="Role")
        role_entity.fields = [
            Field(name="id", type="uuid", primary_key=True),
            Field(name="name", type="string")
        ]
        
        schema.entities = [user_entity, role_entity]
        schema.relationships = [
            Relationship(
                name="user_roles",
                type="many_to_many",
                source_entity="User",
                target_entity="Role"
            )
        ]
        
        generator = APICodeGenerator()
        request = APIGenerationRequest(target_language="python", target_framework="fastapi")
        
        # Should handle complex relationships
        generated_code = generator.generate_api_from_schema(schema, request)
        assert isinstance(generated_code, GeneratedAPICode)
    
    def test_unsupported_language_framework(self):
        """Test handling of unsupported language/framework combinations."""
        generator = APICodeGenerator()
        schema = DatabaseSchema(name="Test")
        
        request = APIGenerationRequest(
            target_language="unsupported_language",
            target_framework="unsupported_framework"
        )
        
        # Should handle gracefully or raise appropriate error
        try:
            generated_code = generator.generate_api_from_schema(schema, request)
            # If it doesn't raise an error, it should at least return something
            assert isinstance(generated_code, GeneratedAPICode)
        except ValueError:
            # Expected for unsupported combinations
            pass
    
    def test_large_schema_performance(self):
        """Test performance with large database schemas."""
        schema = DatabaseSchema(name="Large", description="Large schema")
        
        # Create many entities
        for i in range(50):  # 50 entities
            entity = Entity(name=f"Entity{i}")
            entity.fields = [
                Field(name="id", type="uuid", primary_key=True),
                Field(name="name", type="string"),
                Field(name="description", type="text"),
                Field(name="created_at", type="datetime")
            ]
            schema.entities.append(entity)
        
        generator = APICodeGenerator()
        request = APIGenerationRequest(target_language="python", target_framework="fastapi")
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        generated_code = generator.generate_api_from_schema(schema, request)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Should complete within 30 seconds for 50 entities
        assert generation_time < 30
        assert isinstance(generated_code, GeneratedAPICode)
        assert len(generated_code.code_files) > 50  # Should have many route files


if __name__ == "__main__":
    pytest.main([__file__, "-v"])