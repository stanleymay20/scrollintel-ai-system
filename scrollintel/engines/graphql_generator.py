"""
GraphQL API Generator

This module provides specialized GraphQL API generation capabilities,
including schema generation, resolver creation, and complex relationship handling.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from ..models.api_generation_models import (
    GraphQLSchema, GraphQLType, GraphQLField, Parameter,
    APISpec, GeneratedAPICode, APIGenerationRequest
)
from ..models.database_schema_models import DatabaseSchema, Entity, Field, Relationship


class GraphQLGenerator:
    """Specialized GraphQL API generator."""
    
    def __init__(self):
        self.scalar_mappings = {
            "string": "String",
            "text": "String", 
            "integer": "Int",
            "bigint": "Int",
            "float": "Float",
            "decimal": "Float",
            "boolean": "Boolean",
            "date": "String",
            "datetime": "String",
            "timestamp": "String",
            "json": "JSON",
            "uuid": "ID"
        }
    
    def generate_graphql_from_schema(
        self, 
        database_schema: DatabaseSchema,
        request: APIGenerationRequest
    ) -> GraphQLSchema:
        """Generate GraphQL schema from database schema."""
        graphql_schema = GraphQLSchema()
        
        # Generate types from entities
        for entity in database_schema.entities:
            gql_type = self._entity_to_graphql_type(entity, database_schema)
            graphql_schema.types.append(gql_type)
            
            # Generate input types
            create_input = self._generate_create_input_type(entity)
            update_input = self._generate_update_input_type(entity)
            graphql_schema.types.extend([create_input, update_input])
        
        # Generate queries
        queries = self._generate_queries(database_schema)
        graphql_schema.queries.extend(queries)
        
        # Generate mutations
        mutations = self._generate_mutations(database_schema)
        graphql_schema.mutations.extend(mutations)
        
        # Generate subscriptions if needed
        if request.additional_options.get("include_subscriptions", False):
            subscriptions = self._generate_subscriptions(database_schema)
            graphql_schema.subscriptions.extend(subscriptions)
        
        return graphql_schema
    
    def _entity_to_graphql_type(
        self, 
        entity: Entity, 
        database_schema: DatabaseSchema
    ) -> GraphQLType:
        """Convert database entity to GraphQL type."""
        gql_type = GraphQLType(
            name=entity.name,
            description=entity.description or f"GraphQL type for {entity.name}"
        )
        
        # Add fields from entity
        for field in entity.fields:
            gql_field = GraphQLField(
                name=field.name,
                type=self._map_field_type(field),
                description=field.description,
                nullable=field.nullable
            )
            gql_type.fields.append(gql_field)
        
        # Add relationship fields
        relationships = self._get_entity_relationships(entity, database_schema)
        for relationship in relationships:
            rel_field = self._relationship_to_field(relationship, entity)
            if rel_field:
                gql_type.fields.append(rel_field)
        
        return gql_type
    
    def _generate_create_input_type(self, entity: Entity) -> GraphQLType:
        """Generate create input type for entity."""
        input_type = GraphQLType(
            name=f"{entity.name}CreateInput",
            description=f"Input type for creating {entity.name}",
            is_input=True
        )
        
        # Add fields excluding auto-generated ones
        for field in entity.fields:
            if field.name not in ["id", "created_at", "updated_at"]:
                gql_field = GraphQLField(
                    name=field.name,
                    type=self._map_field_type(field),
                    description=field.description,
                    nullable=field.nullable
                )
                input_type.fields.append(gql_field)
        
        return input_type
    
    def _generate_update_input_type(self, entity: Entity) -> GraphQLType:
        """Generate update input type for entity."""
        input_type = GraphQLType(
            name=f"{entity.name}UpdateInput",
            description=f"Input type for updating {entity.name}",
            is_input=True
        )
        
        # Add fields excluding auto-generated ones, all optional
        for field in entity.fields:
            if field.name not in ["id", "created_at", "updated_at"]:
                gql_field = GraphQLField(
                    name=field.name,
                    type=self._map_field_type(field),
                    description=field.description,
                    nullable=True  # All fields optional for updates
                )
                input_type.fields.append(gql_field)
        
        return input_type
    
    def _generate_queries(self, database_schema: DatabaseSchema) -> List[GraphQLField]:
        """Generate GraphQL queries for all entities."""
        queries = []
        
        for entity in database_schema.entities:
            entity_name = entity.name
            entity_name_lower = entity_name.lower()
            entity_name_plural = f"{entity_name_lower}s"
            
            # Single entity query
            single_query = GraphQLField(
                name=entity_name_lower,
                type=entity_name,
                description=f"Get a single {entity_name} by ID",
                arguments=[
                    Parameter(name="id", type="ID", required=True, description=f"{entity_name} ID")
                ],
                resolver=f"resolve_{entity_name_lower}",
                nullable=True
            )
            queries.append(single_query)
            
            # List query with filtering and pagination
            list_query = GraphQLField(
                name=entity_name_plural,
                type=entity_name,
                description=f"Get a list of {entity_name_plural}",
                list_type=True,
                arguments=self._generate_list_arguments(entity),
                resolver=f"resolve_{entity_name_plural}",
                nullable=False
            )
            queries.append(list_query)
            
            # Search query if entity has searchable fields
            searchable_fields = [f for f in entity.fields if f.type in ["string", "text"]]
            if searchable_fields:
                search_query = GraphQLField(
                    name=f"search_{entity_name_plural}",
                    type=entity_name,
                    description=f"Search {entity_name_plural} by text",
                    list_type=True,
                    arguments=[
                        Parameter(name="query", type="String", required=True, description="Search query"),
                        Parameter(name="limit", type="Int", description="Maximum results", default_value=10),
                        Parameter(name="offset", type="Int", description="Results offset", default_value=0)
                    ],
                    resolver=f"resolve_search_{entity_name_plural}",
                    nullable=False
                )
                queries.append(search_query)
        
        return queries
    
    def _generate_mutations(self, database_schema: DatabaseSchema) -> List[GraphQLField]:
        """Generate GraphQL mutations for all entities."""
        mutations = []
        
        for entity in database_schema.entities:
            entity_name = entity.name
            entity_name_lower = entity_name.lower()
            
            # Create mutation
            create_mutation = GraphQLField(
                name=f"create_{entity_name_lower}",
                type=entity_name,
                description=f"Create a new {entity_name}",
                arguments=[
                    Parameter(
                        name="input", 
                        type=f"{entity_name}CreateInput", 
                        required=True, 
                        description=f"Data for creating {entity_name}"
                    )
                ],
                resolver=f"resolve_create_{entity_name_lower}",
                nullable=False
            )
            mutations.append(create_mutation)
            
            # Update mutation
            update_mutation = GraphQLField(
                name=f"update_{entity_name_lower}",
                type=entity_name,
                description=f"Update an existing {entity_name}",
                arguments=[
                    Parameter(name="id", type="ID", required=True, description=f"{entity_name} ID"),
                    Parameter(
                        name="input", 
                        type=f"{entity_name}UpdateInput", 
                        required=True, 
                        description=f"Data for updating {entity_name}"
                    )
                ],
                resolver=f"resolve_update_{entity_name_lower}",
                nullable=False
            )
            mutations.append(update_mutation)
            
            # Delete mutation
            delete_mutation = GraphQLField(
                name=f"delete_{entity_name_lower}",
                type="Boolean",
                description=f"Delete a {entity_name}",
                arguments=[
                    Parameter(name="id", type="ID", required=True, description=f"{entity_name} ID")
                ],
                resolver=f"resolve_delete_{entity_name_lower}",
                nullable=False
            )
            mutations.append(delete_mutation)
            
            # Bulk operations
            bulk_create_mutation = GraphQLField(
                name=f"bulk_create_{entity_name_lower}s",
                type=entity_name,
                description=f"Create multiple {entity_name}s",
                list_type=True,
                arguments=[
                    Parameter(
                        name="input", 
                        type=f"[{entity_name}CreateInput!]", 
                        required=True, 
                        description=f"Array of data for creating {entity_name}s"
                    )
                ],
                resolver=f"resolve_bulk_create_{entity_name_lower}s",
                nullable=False
            )
            mutations.append(bulk_create_mutation)
        
        return mutations
    
    def _generate_subscriptions(self, database_schema: DatabaseSchema) -> List[GraphQLField]:
        """Generate GraphQL subscriptions for real-time updates."""
        subscriptions = []
        
        for entity in database_schema.entities:
            entity_name = entity.name
            entity_name_lower = entity_name.lower()
            
            # Entity created subscription
            created_subscription = GraphQLField(
                name=f"{entity_name_lower}_created",
                type=entity_name,
                description=f"Subscribe to {entity_name} creation events",
                resolver=f"resolve_{entity_name_lower}_created",
                nullable=False
            )
            subscriptions.append(created_subscription)
            
            # Entity updated subscription
            updated_subscription = GraphQLField(
                name=f"{entity_name_lower}_updated",
                type=entity_name,
                description=f"Subscribe to {entity_name} update events",
                arguments=[
                    Parameter(name="id", type="ID", description=f"Filter by {entity_name} ID")
                ],
                resolver=f"resolve_{entity_name_lower}_updated",
                nullable=False
            )
            subscriptions.append(updated_subscription)
            
            # Entity deleted subscription
            deleted_subscription = GraphQLField(
                name=f"{entity_name_lower}_deleted",
                type="ID",
                description=f"Subscribe to {entity_name} deletion events",
                resolver=f"resolve_{entity_name_lower}_deleted",
                nullable=False
            )
            subscriptions.append(deleted_subscription)
        
        return subscriptions
    
    def _generate_list_arguments(self, entity: Entity) -> List[Parameter]:
        """Generate arguments for list queries."""
        arguments = [
            Parameter(name="limit", type="Int", description="Maximum results", default_value=10),
            Parameter(name="offset", type="Int", description="Results offset", default_value=0),
            Parameter(name="orderBy", type="String", description="Field to order by"),
            Parameter(name="orderDirection", type="String", description="Order direction (ASC/DESC)", default_value="ASC")
        ]
        
        # Add filter arguments for each field
        for field in entity.fields:
            if field.type in ["string", "integer", "float", "boolean", "date", "datetime"]:
                filter_arg = Parameter(
                    name=f"filter_{field.name}",
                    type=self._map_field_type(field),
                    description=f"Filter by {field.name}",
                    required=False
                )
                arguments.append(filter_arg)
                
                # Add range filters for numeric and date fields
                if field.type in ["integer", "float", "date", "datetime"]:
                    arguments.extend([
                        Parameter(
                            name=f"filter_{field.name}_gte",
                            type=self._map_field_type(field),
                            description=f"Filter {field.name} greater than or equal",
                            required=False
                        ),
                        Parameter(
                            name=f"filter_{field.name}_lte",
                            type=self._map_field_type(field),
                            description=f"Filter {field.name} less than or equal",
                            required=False
                        )
                    ])
                
                # Add text search for string fields
                if field.type in ["string", "text"]:
                    arguments.append(Parameter(
                        name=f"filter_{field.name}_contains",
                        type="String",
                        description=f"Filter {field.name} containing text",
                        required=False
                    ))
        
        return arguments
    
    def _map_field_type(self, field: Field) -> str:
        """Map database field type to GraphQL scalar type."""
        base_type = self.scalar_mappings.get(field.type.lower(), "String")
        
        # Handle arrays
        if field.type.lower() == "array":
            return f"[String]"
        
        return base_type
    
    def _get_entity_relationships(
        self, 
        entity: Entity, 
        database_schema: DatabaseSchema
    ) -> List[Relationship]:
        """Get all relationships for an entity."""
        relationships = []
        
        for relationship in database_schema.relationships:
            if relationship.source_entity == entity.name or relationship.target_entity == entity.name:
                relationships.append(relationship)
        
        return relationships
    
    def _relationship_to_field(
        self, 
        relationship: Relationship, 
        entity: Entity
    ) -> Optional[GraphQLField]:
        """Convert relationship to GraphQL field."""
        if relationship.source_entity == entity.name:
            # This entity is the source
            target_name = relationship.target_entity
            field_name = target_name.lower()
            
            if relationship.type == "one_to_many":
                return GraphQLField(
                    name=f"{field_name}s",
                    type=target_name,
                    list_type=True,
                    description=f"Related {target_name}s",
                    resolver=f"resolve_{field_name}s",
                    nullable=False
                )
            elif relationship.type == "one_to_one":
                return GraphQLField(
                    name=field_name,
                    type=target_name,
                    description=f"Related {target_name}",
                    resolver=f"resolve_{field_name}",
                    nullable=True
                )
            elif relationship.type == "many_to_many":
                return GraphQLField(
                    name=f"{field_name}s",
                    type=target_name,
                    list_type=True,
                    description=f"Related {target_name}s",
                    resolver=f"resolve_{field_name}s",
                    nullable=False
                )
        
        elif relationship.target_entity == entity.name:
            # This entity is the target
            source_name = relationship.source_entity
            field_name = source_name.lower()
            
            if relationship.type == "one_to_many":
                return GraphQLField(
                    name=field_name,
                    type=source_name,
                    description=f"Related {source_name}",
                    resolver=f"resolve_{field_name}",
                    nullable=True
                )
            elif relationship.type == "many_to_many":
                return GraphQLField(
                    name=f"{field_name}s",
                    type=source_name,
                    list_type=True,
                    description=f"Related {source_name}s",
                    resolver=f"resolve_{field_name}s",
                    nullable=False
                )
        
        return None
    
    def generate_advanced_graphql_features(
        self, 
        graphql_schema: GraphQLSchema,
        database_schema: DatabaseSchema
    ) -> Dict[str, str]:
        """Generate advanced GraphQL features like custom scalars, directives, etc."""
        features = {}
        
        # Custom scalars
        custom_scalars = self._generate_custom_scalars(database_schema)
        if custom_scalars:
            features["custom_scalars.py"] = custom_scalars
        
        # Directives
        directives = self._generate_directives()
        if directives:
            features["directives.py"] = directives
        
        # DataLoader for N+1 problem resolution
        dataloaders = self._generate_dataloaders(database_schema)
        if dataloaders:
            features["dataloaders.py"] = dataloaders
        
        # Middleware
        middleware = self._generate_graphql_middleware()
        if middleware:
            features["middleware.py"] = middleware
        
        return features
    
    def _generate_custom_scalars(self, database_schema: DatabaseSchema) -> str:
        """Generate custom GraphQL scalars."""
        scalars_needed = set()
        
        # Check what custom scalars we need
        for entity in database_schema.entities:
            for field in entity.fields:
                if field.type.lower() in ["datetime", "date", "json", "uuid"]:
                    scalars_needed.add(field.type.lower())
        
        if not scalars_needed:
            return ""
        
        scalar_code = '''"""
Custom GraphQL Scalars
"""

import json
from datetime import datetime, date
from uuid import UUID
# from graphene import Scalar  # Temporarily disabled for Docker build
from graphql.language import ast

'''
        
        if "datetime" in scalars_needed:
            scalar_code += '''
class DateTime(Scalar):
    """DateTime scalar type."""
    
    @staticmethod
    def serialize(dt):
        return dt.isoformat() if dt else None
    
    @staticmethod
    def parse_literal(node):
        if isinstance(node, ast.StringValue):
            return datetime.fromisoformat(node.value)
    
    @staticmethod
    def parse_value(value):
        return datetime.fromisoformat(value) if value else None

'''
        
        if "date" in scalars_needed:
            scalar_code += '''
class Date(Scalar):
    """Date scalar type."""
    
    @staticmethod
    def serialize(dt):
        return dt.isoformat() if dt else None
    
    @staticmethod
    def parse_literal(node):
        if isinstance(node, ast.StringValue):
            return date.fromisoformat(node.value)
    
    @staticmethod
    def parse_value(value):
        return date.fromisoformat(value) if value else None

'''
        
        if "json" in scalars_needed:
            scalar_code += '''
class JSON(Scalar):
    """JSON scalar type."""
    
    @staticmethod
    def serialize(value):
        return value
    
    @staticmethod
    def parse_literal(node):
        if isinstance(node, ast.StringValue):
            return json.loads(node.value)
    
    @staticmethod
    def parse_value(value):
        return json.loads(value) if isinstance(value, str) else value

'''
        
        if "uuid" in scalars_needed:
            scalar_code += '''
class UUID(Scalar):
    """UUID scalar type."""
    
    @staticmethod
    def serialize(uuid_val):
        return str(uuid_val) if uuid_val else None
    
    @staticmethod
    def parse_literal(node):
        if isinstance(node, ast.StringValue):
            return UUID(node.value)
    
    @staticmethod
    def parse_value(value):
        return UUID(value) if value else None

'''
        
        return scalar_code
    
    def _generate_directives(self) -> str:
        """Generate GraphQL directives."""
        return '''"""
GraphQL Directives
"""

# from graphene import Directive, String, Boolean, Int  # Temporarily disabled for Docker build

class AuthDirective(Directive):
    """Authentication directive."""
    
    class Meta:
        locations = ["FIELD_DEFINITION"]
    
    requires = String(description="Required permission")

class RateLimitDirective(Directive):
    """Rate limiting directive."""
    
    class Meta:
        locations = ["FIELD_DEFINITION"]
    
    max_requests = Int(description="Maximum requests per window")
    window = Int(description="Time window in seconds")

class CacheDirective(Directive):
    """Caching directive."""
    
    class Meta:
        locations = ["FIELD_DEFINITION"]
    
    max_age = Int(description="Cache max age in seconds")
    scope = String(description="Cache scope (PUBLIC, PRIVATE)")
'''
    
    def _generate_dataloaders(self, database_schema: DatabaseSchema) -> str:
        """Generate DataLoader classes for efficient data fetching."""
        dataloader_code = '''"""
DataLoaders for efficient GraphQL data fetching
"""

from typing import List, Dict, Any
from aiodataloader import DataLoader

'''
        
        for entity in database_schema.entities:
            entity_name = entity.name
            entity_name_lower = entity_name.lower()
            
            dataloader_code += f'''
class {entity_name}Loader(DataLoader):
    """DataLoader for {entity_name} entities."""
    
    async def batch_load_fn(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Batch load {entity_name}s by IDs."""
        # TODO: Implement batch loading logic
        # This should fetch all {entity_name}s with IDs in keys
        # and return them in the same order as keys
        results = []
        for key in keys:
            # Placeholder - replace with actual database query
            result = {{"id": key, "name": f"{entity_name} {{key}}"}}
            results.append(result)
        return results

'''
        
        # Add relationship loaders
        for relationship in database_schema.relationships:
            source_entity = relationship.source_entity
            target_entity = relationship.target_entity
            
            dataloader_code += f'''
class {source_entity}{target_entity}Loader(DataLoader):
    """DataLoader for {source_entity} -> {target_entity} relationship."""
    
    async def batch_load_fn(self, source_ids: List[str]) -> List[List[Dict[str, Any]]]:
        """Batch load {target_entity}s for {source_entity}s."""
        # TODO: Implement relationship batch loading
        results = []
        for source_id in source_ids:
            # Placeholder - replace with actual relationship query
            related_items = []
            results.append(related_items)
        return results

'''
        
        return dataloader_code
    
    def _generate_graphql_middleware(self) -> str:
        """Generate GraphQL middleware."""
        return '''"""
GraphQL Middleware
"""

from typing import Any, Dict
from graphql import GraphQLResolveInfo

class AuthenticationMiddleware:
    """Authentication middleware for GraphQL."""
    
    def resolve(self, next, root, info: GraphQLResolveInfo, **args):
        """Check authentication before resolving."""
        # TODO: Implement authentication logic
        # Check if user is authenticated
        # Extract user from context
        return next(root, info, **args)

class AuthorizationMiddleware:
    """Authorization middleware for GraphQL."""
    
    def resolve(self, next, root, info: GraphQLResolveInfo, **args):
        """Check authorization before resolving."""
        # TODO: Implement authorization logic
        # Check if user has permission for this field
        return next(root, info, **args)

class LoggingMiddleware:
    """Logging middleware for GraphQL."""
    
    def resolve(self, next, root, info: GraphQLResolveInfo, **args):
        """Log GraphQL operations."""
        field_name = info.field_name
        parent_type = info.parent_type.name
        
        # Log the operation
        print(f"Resolving {parent_type}.{field_name}")
        
        result = next(root, info, **args)
        
        # Log completion
        print(f"Resolved {parent_type}.{field_name}")
        
        return result

class ErrorHandlingMiddleware:
    """Error handling middleware for GraphQL."""
    
    def resolve(self, next, root, info: GraphQLResolveInfo, **args):
        """Handle errors in GraphQL resolvers."""
        try:
            return next(root, info, **args)
        except Exception as e:
            # Log the error
            print(f"Error in {info.parent_type.name}.{info.field_name}: {str(e)}")
            
            # Return appropriate error or None
            raise e

class CachingMiddleware:
    """Caching middleware for GraphQL."""
    
    def __init__(self):
        self.cache = {}
    
    def resolve(self, next, root, info: GraphQLResolveInfo, **args):
        """Cache GraphQL resolver results."""
        # Create cache key
        cache_key = f"{info.parent_type.name}.{info.field_name}:{hash(str(args))}"
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Resolve and cache
        result = next(root, info, **args)
        self.cache[cache_key] = result
        
        return result
'''