"""
API Generation Routes

This module provides REST API endpoints for the automated code generation system,
allowing users to generate APIs from database schemas and natural language requirements.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from typing import Dict, List, Optional, Any
import json
import zipfile
import io
import tempfile
import os
from datetime import datetime

from ...models.api_generation_models import (
    APISpec, APIGenerationRequest, GeneratedAPICode,
    APIType, HTTPMethod
)
from ...models.database_schema_models import DatabaseSchema
from ...engines.api_code_generator import APICodeGenerator
from ...engines.graphql_generator import GraphQLGenerator
from ...engines.api_documentation_generator import (
    APIDocumentationGenerator, generate_complete_documentation
)
from ...engines.api_versioning_manager import APIVersioningManager
from ...core.auth import get_current_user


router = APIRouter(prefix="/api/v1/code-generation", tags=["API Code Generation"])


# Initialize generators
api_generator = APICodeGenerator()
graphql_generator = GraphQLGenerator()
doc_generator = APIDocumentationGenerator()
versioning_manager = APIVersioningManager()


@router.post("/generate-from-schema")
async def generate_api_from_schema(
    request: APIGenerationRequest,
    database_schema: DatabaseSchema,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Generate API code from database schema."""
    try:
        # Generate API code
        generated_code = api_generator.generate_api_from_schema(
            database_schema, 
            request
        )
        
        # Store generation metadata
        generation_metadata = {
            "user_id": current_user.get("user_id"),
            "generated_at": datetime.now().isoformat(),
            "language": request.target_language,
            "framework": request.target_framework,
            "database_type": request.database_type,
            "schema_name": database_schema.name,
            "entity_count": len(database_schema.entities),
            "file_count": len(generated_code.code_files) + len(generated_code.test_files) + len(generated_code.documentation_files)
        }
        
        return {
            "success": True,
            "message": "API code generated successfully",
            "generation_id": generated_code.api_spec_id,
            "metadata": generation_metadata,
            "files": {
                "code_files": list(generated_code.code_files.keys()),
                "test_files": list(generated_code.test_files.keys()),
                "documentation_files": list(generated_code.documentation_files.keys()),
                "configuration_files": list(generated_code.configuration_files.keys())
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate API code: {str(e)}"
        )


@router.post("/generate-from-requirements")
async def generate_api_from_requirements(
    requirements_text: str,
    request: APIGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Generate API code from natural language requirements."""
    try:
        # TODO: Implement NLP processing to convert requirements to database schema
        # For now, return a placeholder response
        
        return {
            "success": False,
            "message": "Natural language processing for requirements is not yet implemented",
            "suggestion": "Please use the generate-from-schema endpoint with a structured database schema"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate API from requirements: {str(e)}"
        )


@router.get("/download/{generation_id}")
async def download_generated_code(
    generation_id: str,
    format: str = "zip",
    current_user: dict = Depends(get_current_user)
):
    """Download generated API code as a zip file."""
    try:
        # TODO: Implement storage and retrieval of generated code
        # For now, create a sample zip file
        
        # Create in-memory zip file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add sample files
            zip_file.writestr("README.md", "# Generated API\n\nThis is a generated API.")
            zip_file.writestr("main.py", "# Generated main.py\nprint('Hello, World!')")
            zip_file.writestr("requirements.txt", "fastapi>=0.104.0\nuvicorn>=0.24.0")
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=api-{generation_id}.zip"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download generated code: {str(e)}"
        )


@router.post("/generate-graphql")
async def generate_graphql_api(
    database_schema: DatabaseSchema,
    request: APIGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate GraphQL API from database schema."""
    try:
        # Generate GraphQL schema
        graphql_schema = graphql_generator.generate_graphql_from_schema(
            database_schema, 
            request
        )
        
        # Generate schema definition
        schema_definition = graphql_generator._generate_graphql_schema(graphql_schema)
        
        # Generate resolvers
        resolvers_code = graphql_generator._generate_graphql_resolvers(
            graphql_schema, 
            request
        )
        
        # Generate advanced features
        advanced_features = graphql_generator.generate_advanced_graphql_features(
            graphql_schema,
            database_schema
        )
        
        return {
            "success": True,
            "message": "GraphQL API generated successfully",
            "schema_definition": schema_definition,
            "resolvers_code": resolvers_code,
            "advanced_features": advanced_features,
            "types_count": len(graphql_schema.types),
            "queries_count": len(graphql_schema.queries),
            "mutations_count": len(graphql_schema.mutations),
            "subscriptions_count": len(graphql_schema.subscriptions)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate GraphQL API: {str(e)}"
        )


@router.post("/generate-documentation")
async def generate_api_documentation(
    api_spec: APISpec,
    formats: List[str] = ["openapi", "swagger", "postman", "guide"],
    current_user: dict = Depends(get_current_user)
):
    """Generate comprehensive API documentation."""
    try:
        # Generate documentation in requested formats
        documentation = generate_complete_documentation(api_spec, formats)
        
        return {
            "success": True,
            "message": "API documentation generated successfully",
            "formats": formats,
            "files": list(documentation.keys()),
            "documentation": documentation
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate API documentation: {str(e)}"
        )


@router.post("/generate-client")
async def generate_api_client(
    api_spec: APISpec,
    language: str = "python",
    current_user: dict = Depends(get_current_user)
):
    """Generate API client code in specified language."""
    try:
        # Generate client code
        client_code = doc_generator.generate_api_client_code(api_spec, language)
        
        return {
            "success": True,
            "message": f"API client generated successfully for {language}",
            "language": language,
            "files": client_code
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate API client: {str(e)}"
        )


@router.post("/compare-versions")
async def compare_api_versions(
    old_spec: APISpec,
    new_spec: APISpec,
    current_user: dict = Depends(get_current_user)
):
    """Compare two API versions and generate compatibility report."""
    try:
        # Compare API versions
        compatibility_report = versioning_manager.compare_api_versions(
            old_spec, 
            new_spec
        )
        
        return {
            "success": True,
            "message": "API versions compared successfully",
            "old_version": compatibility_report.old_version,
            "new_version": compatibility_report.new_version,
            "is_backward_compatible": compatibility_report.is_backward_compatible,
            "breaking_changes_count": len(compatibility_report.breaking_changes),
            "non_breaking_changes_count": len(compatibility_report.non_breaking_changes),
            "additions_count": len(compatibility_report.additions),
            "deprecations_count": len(compatibility_report.deprecations),
            "breaking_changes": [
                {
                    "component": change.component,
                    "component_name": change.component_name,
                    "description": change.description,
                    "migration_notes": change.migration_notes
                }
                for change in compatibility_report.breaking_changes
            ],
            "migration_guide": compatibility_report.migration_guide
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare API versions: {str(e)}"
        )


@router.post("/create-version")
async def create_api_version(
    base_spec: APISpec,
    version: str,
    changes: List[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """Create a new API version with specified changes."""
    try:
        # Convert changes to APIChange objects if provided
        api_changes = []
        if changes:
            from ...engines.api_versioning_manager import APIChange, ChangeType
            for change_data in changes:
                api_changes.append(APIChange(
                    change_type=ChangeType(change_data.get("type", "non_breaking")),
                    component=change_data.get("component", ""),
                    component_name=change_data.get("component_name", ""),
                    description=change_data.get("description", ""),
                    old_value=change_data.get("old_value"),
                    new_value=change_data.get("new_value"),
                    migration_notes=change_data.get("migration_notes")
                ))
        
        # Create new version
        new_spec = versioning_manager.create_versioned_api(
            base_spec, 
            version, 
            api_changes
        )
        
        # Suggest version number if not provided
        if not version and api_changes:
            suggested_version = versioning_manager.suggest_version_number(
                base_spec.version, 
                api_changes
            )
        else:
            suggested_version = version
        
        return {
            "success": True,
            "message": "New API version created successfully",
            "base_version": base_spec.version,
            "new_version": new_spec.version,
            "suggested_version": suggested_version,
            "changes_applied": len(api_changes),
            "endpoints_count": len(new_spec.endpoints)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create API version: {str(e)}"
        )


@router.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported programming languages and frameworks."""
    return {
        "success": True,
        "supported_languages": api_generator.supported_languages,
        "supported_frameworks": api_generator.supported_frameworks,
        "database_types": ["postgresql", "mysql", "sqlite", "mongodb"],
        "orm_types": ["sqlalchemy", "django", "prisma", "mongoose"]
    }


@router.get("/templates")
async def get_api_templates():
    """Get available API templates and patterns."""
    templates = {
        "rest_crud": {
            "name": "REST CRUD API",
            "description": "Standard REST API with CRUD operations",
            "features": ["CRUD operations", "Pagination", "Filtering", "Validation"]
        },
        "graphql_full": {
            "name": "Full GraphQL API",
            "description": "Complete GraphQL API with queries, mutations, and subscriptions",
            "features": ["Queries", "Mutations", "Subscriptions", "DataLoader", "Custom Scalars"]
        },
        "microservice": {
            "name": "Microservice API",
            "description": "Microservice-ready API with health checks and metrics",
            "features": ["Health checks", "Metrics", "Logging", "Circuit breaker"]
        },
        "serverless": {
            "name": "Serverless API",
            "description": "Serverless-optimized API for cloud functions",
            "features": ["Cold start optimization", "Event-driven", "Auto-scaling"]
        }
    }
    
    return {
        "success": True,
        "templates": templates
    }


@router.post("/validate-schema")
async def validate_database_schema(
    database_schema: DatabaseSchema,
    current_user: dict = Depends(get_current_user)
):
    """Validate database schema for API generation."""
    try:
        validation_errors = []
        validation_warnings = []
        
        # Basic validation
        if not database_schema.name:
            validation_errors.append("Database schema name is required")
        
        if not database_schema.entities:
            validation_errors.append("Database schema must contain at least one entity")
        
        # Entity validation
        for entity in database_schema.entities:
            if not entity.name:
                validation_errors.append(f"Entity name is required")
                continue
            
            if not entity.fields:
                validation_errors.append(f"Entity '{entity.name}' must have at least one field")
                continue
            
            # Check for primary key
            has_primary_key = any(field.primary_key for field in entity.fields)
            if not has_primary_key:
                validation_warnings.append(f"Entity '{entity.name}' should have a primary key field")
            
            # Field validation
            for field in entity.fields:
                if not field.name:
                    validation_errors.append(f"Field name is required in entity '{entity.name}'")
                
                if not field.type:
                    validation_errors.append(f"Field type is required for '{field.name}' in entity '{entity.name}'")
        
        # Relationship validation
        entity_names = {entity.name for entity in database_schema.entities}
        for relationship in database_schema.relationships:
            if relationship.source_entity not in entity_names:
                validation_errors.append(f"Source entity '{relationship.source_entity}' not found in schema")
            
            if relationship.target_entity not in entity_names:
                validation_errors.append(f"Target entity '{relationship.target_entity}' not found in schema")
        
        is_valid = len(validation_errors) == 0
        
        return {
            "success": True,
            "is_valid": is_valid,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "entity_count": len(database_schema.entities),
            "relationship_count": len(database_schema.relationships)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate database schema: {str(e)}"
        )


@router.post("/preview")
async def preview_generated_api(
    database_schema: DatabaseSchema,
    request: APIGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Preview what will be generated without actually generating the code."""
    try:
        # Create API specification
        api_spec = api_generator._create_api_spec_from_schema(
            database_schema, 
            request
        )
        
        # Analyze what will be generated
        endpoints_by_entity = {}
        for endpoint in api_spec.endpoints:
            for tag in endpoint.tags:
                if tag not in endpoints_by_entity:
                    endpoints_by_entity[tag] = []
                endpoints_by_entity[tag].append({
                    "method": endpoint.method.value,
                    "path": endpoint.path,
                    "name": endpoint.name,
                    "description": endpoint.description
                })
        
        # Estimate file counts
        estimated_files = {
            "code_files": 2 + len(database_schema.entities),  # main.py, models.py, + route files
            "test_files": 1 + len(database_schema.entities),  # conftest.py + test files
            "documentation_files": 3,  # README.md, openapi.json, openapi.yaml
            "configuration_files": 4   # requirements.txt, Dockerfile, docker-compose.yml, .env.example
        }
        
        return {
            "success": True,
            "message": "API preview generated successfully",
            "api_name": api_spec.name,
            "api_version": api_spec.version,
            "base_url": api_spec.base_url,
            "endpoints_count": len(api_spec.endpoints),
            "endpoints_by_entity": endpoints_by_entity,
            "estimated_files": estimated_files,
            "features": {
                "authentication": request.include_authentication,
                "validation": request.include_validation,
                "tests": request.include_tests,
                "documentation": request.include_documentation
            },
            "technology_stack": {
                "language": request.target_language,
                "framework": request.target_framework,
                "database": request.database_type,
                "orm": request.orm_type
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate API preview: {str(e)}"
        )


@router.get("/generation-history")
async def get_generation_history(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get user's API generation history."""
    try:
        # TODO: Implement actual history retrieval from database
        # For now, return mock data
        
        mock_history = [
            {
                "generation_id": "gen_123",
                "schema_name": "BlogSystem",
                "language": "python",
                "framework": "fastapi",
                "generated_at": "2024-01-15T10:30:00Z",
                "entity_count": 3,
                "file_count": 12
            },
            {
                "generation_id": "gen_124",
                "schema_name": "ECommerceAPI",
                "language": "typescript",
                "framework": "express",
                "generated_at": "2024-01-14T15:45:00Z",
                "entity_count": 8,
                "file_count": 25
            }
        ]
        
        return {
            "success": True,
            "history": mock_history[offset:offset + limit],
            "total": len(mock_history),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve generation history: {str(e)}"
        )


@router.delete("/generation/{generation_id}")
async def delete_generated_code(
    generation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete generated API code and associated files."""
    try:
        # TODO: Implement actual deletion from storage
        
        return {
            "success": True,
            "message": f"Generated code {generation_id} deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete generated code: {str(e)}"
        )


@router.post("/upload-schema")
async def upload_database_schema(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload database schema file (JSON, YAML, or SQL)."""
    try:
        # Read file content
        content = await file.read()
        
        # Parse based on file extension
        if file.filename.endswith('.json'):
            schema_data = json.loads(content.decode('utf-8'))
        elif file.filename.endswith(('.yml', '.yaml')):
            import yaml
            schema_data = yaml.safe_load(content.decode('utf-8'))
        elif file.filename.endswith('.sql'):
            # TODO: Implement SQL parsing to extract schema
            raise HTTPException(
                status_code=400,
                detail="SQL file parsing is not yet implemented"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please use JSON, YAML, or SQL files."
            )
        
        # TODO: Convert schema_data to DatabaseSchema object
        
        return {
            "success": True,
            "message": "Database schema uploaded successfully",
            "filename": file.filename,
            "size": len(content),
            "format": file.filename.split('.')[-1]
        }
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in uploaded file"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload database schema: {str(e)}"
        )