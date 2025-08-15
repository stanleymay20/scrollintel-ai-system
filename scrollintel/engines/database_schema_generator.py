"""
Database schema generation engine for automated code generation.
"""
import uuid
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text

from ..models.database_schema_models import (
    DatabaseSchema, Entity, Field, Index, Constraint, EntityRelationship,
    Migration, MigrationOperation, PerformanceOptimization,
    SchemaGenerationRequest, SchemaGenerationResult, SchemaValidationResult,
    DatabaseType, FieldType, RelationshipType, IndexType, ConstraintType
)
from ..models.code_generation_models import Requirements, Entity as RequirementEntity
from ..core.config import get_settings


class DatabaseSchemaGenerator:
    """Generates database schemas from business requirements."""
    
    def __init__(self):
        self.settings = get_settings()
        self.type_mappings = self._initialize_type_mappings()
        self.naming_conventions = self._initialize_naming_conventions()
    
    def _initialize_type_mappings(self) -> Dict[DatabaseType, Dict[str, str]]:
        """Initialize field type mappings for different databases."""
        return {
            DatabaseType.POSTGRESQL: {
                FieldType.STRING: "VARCHAR",
                FieldType.INTEGER: "INTEGER",
                FieldType.FLOAT: "REAL",
                FieldType.BOOLEAN: "BOOLEAN",
                FieldType.DATE: "DATE",
                FieldType.DATETIME: "TIMESTAMP",
                FieldType.TEXT: "TEXT",
                FieldType.JSON: "JSONB",
                FieldType.UUID: "UUID",
                FieldType.DECIMAL: "DECIMAL",
                FieldType.BINARY: "BYTEA",
                FieldType.ARRAY: "ARRAY"
            },
            DatabaseType.MYSQL: {
                FieldType.STRING: "VARCHAR",
                FieldType.INTEGER: "INT",
                FieldType.FLOAT: "FLOAT",
                FieldType.BOOLEAN: "BOOLEAN",
                FieldType.DATE: "DATE",
                FieldType.DATETIME: "DATETIME",
                FieldType.TEXT: "TEXT",
                FieldType.JSON: "JSON",
                FieldType.UUID: "CHAR(36)",
                FieldType.DECIMAL: "DECIMAL",
                FieldType.BINARY: "BLOB",
                FieldType.ARRAY: "JSON"
            },
            DatabaseType.MONGODB: {
                FieldType.STRING: "String",
                FieldType.INTEGER: "Number",
                FieldType.FLOAT: "Number",
                FieldType.BOOLEAN: "Boolean",
                FieldType.DATE: "Date",
                FieldType.DATETIME: "Date",
                FieldType.TEXT: "String",
                FieldType.JSON: "Object",
                FieldType.UUID: "String",
                FieldType.DECIMAL: "Decimal128",
                FieldType.BINARY: "BinData",
                FieldType.ARRAY: "Array",
                FieldType.OBJECT: "Object"
            }
        }
    
    def _initialize_naming_conventions(self) -> Dict[str, str]:
        """Initialize default naming conventions."""
        return {
            "table_case": "snake_case",
            "column_case": "snake_case",
            "index_prefix": "idx_",
            "foreign_key_prefix": "fk_",
            "unique_prefix": "uk_",
            "check_prefix": "ck_"
        }
    
    def generate_schema(self, request: SchemaGenerationRequest, requirements: Requirements) -> SchemaGenerationResult:
        """Generate database schema from requirements."""
        try:
            start_time = datetime.utcnow()
            
            # Extract entities from requirements
            entities = self._extract_entities_from_requirements(requirements)
            
            # Generate database entities
            db_entities = self._generate_database_entities(entities, request.database_type)
            
            # Establish relationships
            relationships = self._establish_relationships(db_entities, requirements)
            
            # Apply optimizations
            optimizations = self._generate_optimizations(db_entities, request)
            
            # Create schema
            schema = DatabaseSchema(
                id=str(uuid.uuid4()),
                name=f"{requirements.project_name}_schema",
                database_type=request.database_type,
                entities=db_entities,
                relationships=relationships,
                optimizations=optimizations,
                performance_requirements=request.performance_requirements,
                security_settings=request.security_requirements
            )
            
            # Generate SQL scripts
            sql_scripts = self._generate_sql_scripts(schema)
            
            # Generate documentation
            documentation = self._generate_documentation(schema)
            
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            return SchemaGenerationResult(
                success=True,
                schema=schema,
                generation_time=generation_time,
                sql_scripts=sql_scripts,
                documentation=documentation
            )
            
        except Exception as e:
            return SchemaGenerationResult(
                success=False,
                errors=[f"Schema generation failed: {str(e)}"],
                generation_time=0.0
            )
    
    def _extract_entities_from_requirements(self, requirements: Requirements) -> List[RequirementEntity]:
        """Extract data entities from requirements."""
        entities = []
        
        # Get entities marked as data entities
        for entity in requirements.entities:
            if entity.type.value == "data_entity":
                entities.append(entity)
        
        # Extract additional entities from requirement text
        for req in requirements.parsed_requirements:
            # Look for common data entity patterns
            entity_patterns = [
                r'\b(user|customer|order|product|invoice|payment|account|profile)\b',
                r'\b(\w+)\s+(?:table|entity|model|record)\b',
                r'\bstore\s+(\w+)\s+(?:data|information)\b'
            ]
            
            for pattern in entity_patterns:
                matches = re.findall(pattern, req.original_text.lower())
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    # Create entity if not already exists
                    if not any(e.name.lower() == match.lower() for e in entities):
                        entity = RequirementEntity(
                            id=str(uuid.uuid4()),
                            name=match.title(),
                            type="data_entity",
                            description=f"Data entity extracted from requirements",
                            confidence=0.7,
                            source_text=req.original_text,
                            position=(0, len(req.original_text))
                        )
                        entities.append(entity)
        
        return entities
    
    def _generate_database_entities(self, entities: List[RequirementEntity], db_type: DatabaseType) -> List[Entity]:
        """Generate database entities from requirement entities."""
        db_entities = []
        
        for entity in entities:
            # Generate fields for entity
            fields = self._generate_entity_fields(entity, db_type)
            
            # Generate indexes
            indexes = self._generate_entity_indexes(entity, fields)
            
            # Generate constraints
            constraints = self._generate_entity_constraints(entity, fields)
            
            db_entity = Entity(
                name=entity.name,
                table_name=self._to_table_name(entity.name),
                description=entity.description,
                fields=fields,
                indexes=indexes,
                constraints=constraints
            )
            
            db_entities.append(db_entity)
        
        return db_entities
    
    def _generate_entity_fields(self, entity: RequirementEntity, db_type: DatabaseType) -> List[Field]:
        """Generate fields for a database entity."""
        fields = []
        
        # Always add ID field
        id_field = Field(
            name="id",
            type=FieldType.UUID if db_type == DatabaseType.POSTGRESQL else FieldType.STRING,
            nullable=False,
            description="Primary key"
        )
        fields.append(id_field)
        
        # Generate fields based on entity attributes and common patterns
        common_fields = self._get_common_fields_for_entity(entity.name.lower())
        
        for field_name, field_type in common_fields.items():
            field = Field(
                name=field_name,
                type=field_type,
                nullable=field_name not in ["name", "title", "email"],
                description=f"{field_name.replace('_', ' ').title()} field"
            )
            
            # Set appropriate constraints
            if field_name == "email":
                field.validation_rules = ["email_format"]
            elif field_name in ["created_at", "updated_at"]:
                field.nullable = False
                if field_name == "created_at":
                    field.default_value = "CURRENT_TIMESTAMP"
            elif field_type == FieldType.STRING:
                field.max_length = 255
            
            fields.append(field)
        
        # Add audit fields
        audit_fields = [
            Field(name="created_at", type=FieldType.DATETIME, nullable=False, default_value="CURRENT_TIMESTAMP"),
            Field(name="updated_at", type=FieldType.DATETIME, nullable=False, default_value="CURRENT_TIMESTAMP"),
            Field(name="created_by", type=FieldType.UUID, nullable=True),
            Field(name="updated_by", type=FieldType.UUID, nullable=True)
        ]
        
        fields.extend(audit_fields)
        
        return fields
    
    def _get_common_fields_for_entity(self, entity_name: str) -> Dict[str, FieldType]:
        """Get common fields based on entity type."""
        field_patterns = {
            "user": {
                "username": FieldType.STRING,
                "email": FieldType.STRING,
                "first_name": FieldType.STRING,
                "last_name": FieldType.STRING,
                "password_hash": FieldType.STRING,
                "is_active": FieldType.BOOLEAN,
                "last_login": FieldType.DATETIME
            },
            "customer": {
                "name": FieldType.STRING,
                "email": FieldType.STRING,
                "phone": FieldType.STRING,
                "address": FieldType.TEXT,
                "company": FieldType.STRING,
                "status": FieldType.STRING
            },
            "order": {
                "order_number": FieldType.STRING,
                "customer_id": FieldType.UUID,
                "status": FieldType.STRING,
                "total_amount": FieldType.DECIMAL,
                "currency": FieldType.STRING,
                "order_date": FieldType.DATETIME
            },
            "product": {
                "name": FieldType.STRING,
                "description": FieldType.TEXT,
                "price": FieldType.DECIMAL,
                "currency": FieldType.STRING,
                "sku": FieldType.STRING,
                "category": FieldType.STRING,
                "is_active": FieldType.BOOLEAN
            },
            "default": {
                "name": FieldType.STRING,
                "description": FieldType.TEXT,
                "status": FieldType.STRING,
                "is_active": FieldType.BOOLEAN
            }
        }
        
        return field_patterns.get(entity_name, field_patterns["default"])
    
    def _generate_entity_indexes(self, entity: RequirementEntity, fields: List[Field]) -> List[Index]:
        """Generate indexes for entity."""
        indexes = []
        
        # Primary key index
        pk_index = Index(
            name=f"pk_{self._to_table_name(entity.name)}",
            type=IndexType.PRIMARY,
            fields=["id"],
            unique=True
        )
        indexes.append(pk_index)
        
        # Generate indexes for common lookup fields
        lookup_fields = ["email", "username", "name", "sku", "order_number"]
        for field in fields:
            if field.name in lookup_fields:
                index = Index(
                    name=f"idx_{self._to_table_name(entity.name)}_{field.name}",
                    type=IndexType.INDEX,
                    fields=[field.name],
                    unique=field.name in ["email", "username", "sku", "order_number"]
                )
                indexes.append(index)
        
        # Add composite indexes for common query patterns
        if any(f.name == "status" for f in fields) and any(f.name == "created_at" for f in fields):
            composite_index = Index(
                name=f"idx_{self._to_table_name(entity.name)}_status_created",
                type=IndexType.COMPOSITE,
                fields=["status", "created_at"]
            )
            indexes.append(composite_index)
        
        return indexes
    
    def _generate_entity_constraints(self, entity: RequirementEntity, fields: List[Field]) -> List[Constraint]:
        """Generate constraints for entity."""
        constraints = []
        
        # Primary key constraint
        pk_constraint = Constraint(
            name=f"pk_{self._to_table_name(entity.name)}",
            type=ConstraintType.PRIMARY_KEY,
            fields=["id"]
        )
        constraints.append(pk_constraint)
        
        # Not null constraints
        for field in fields:
            if not field.nullable:
                nn_constraint = Constraint(
                    name=f"nn_{self._to_table_name(entity.name)}_{field.name}",
                    type=ConstraintType.NOT_NULL,
                    fields=[field.name]
                )
                constraints.append(nn_constraint)
        
        # Unique constraints
        unique_fields = ["email", "username", "sku", "order_number"]
        for field in fields:
            if field.name in unique_fields:
                unique_constraint = Constraint(
                    name=f"uk_{self._to_table_name(entity.name)}_{field.name}",
                    type=ConstraintType.UNIQUE,
                    fields=[field.name]
                )
                constraints.append(unique_constraint)
        
        # Check constraints
        for field in fields:
            if field.name == "email":
                check_constraint = Constraint(
                    name=f"ck_{self._to_table_name(entity.name)}_email_format",
                    type=ConstraintType.CHECK,
                    fields=[field.name],
                    check_condition="email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'"
                )
                constraints.append(check_constraint)
        
        return constraints
    
    def _establish_relationships(self, entities: List[Entity], requirements: Requirements) -> List[EntityRelationship]:
        """Establish relationships between entities."""
        relationships = []
        
        # Analyze requirement relationships
        for relationship in requirements.relationships:
            source_entity = self._find_entity_by_name(entities, relationship.source_entity_id)
            target_entity = self._find_entity_by_name(entities, relationship.target_entity_id)
            
            if source_entity and target_entity:
                rel_type = self._determine_relationship_type(relationship.relationship_type)
                
                entity_relationship = EntityRelationship(
                    id=str(uuid.uuid4()),
                    source_entity=source_entity.name,
                    target_entity=target_entity.name,
                    relationship_type=rel_type,
                    source_fields=[f"{target_entity.name.lower()}_id"],
                    target_fields=["id"],
                    description=relationship.description
                )
                relationships.append(entity_relationship)
                
                # Add foreign key field to source entity
                fk_field = Field(
                    name=f"{target_entity.name.lower()}_id",
                    type=FieldType.UUID,
                    nullable=rel_type != RelationshipType.ONE_TO_ONE,
                    description=f"Foreign key to {target_entity.name}"
                )
                source_entity.fields.append(fk_field)
                
                # Add foreign key constraint
                fk_constraint = Constraint(
                    name=f"fk_{source_entity.table_name}_{target_entity.table_name}",
                    type=ConstraintType.FOREIGN_KEY,
                    fields=[f"{target_entity.name.lower()}_id"],
                    reference_table=target_entity.table_name,
                    reference_fields=["id"],
                    on_delete="CASCADE" if rel_type == RelationshipType.ONE_TO_MANY else "SET NULL"
                )
                source_entity.constraints.append(fk_constraint)
        
        return relationships
    
    def _find_entity_by_name(self, entities: List[Entity], name: str) -> Optional[Entity]:
        """Find entity by name."""
        for entity in entities:
            if entity.name.lower() == name.lower():
                return entity
        return None
    
    def _determine_relationship_type(self, rel_type: str) -> RelationshipType:
        """Determine relationship type from string."""
        type_mapping = {
            "one_to_one": RelationshipType.ONE_TO_ONE,
            "one_to_many": RelationshipType.ONE_TO_MANY,
            "many_to_one": RelationshipType.MANY_TO_ONE,
            "many_to_many": RelationshipType.MANY_TO_MANY,
            "belongs_to": RelationshipType.MANY_TO_ONE,
            "has_many": RelationshipType.ONE_TO_MANY,
            "has_one": RelationshipType.ONE_TO_ONE
        }
        return type_mapping.get(rel_type.lower(), RelationshipType.MANY_TO_ONE)
    
    def _generate_optimizations(self, entities: List[Entity], request: SchemaGenerationRequest) -> List[PerformanceOptimization]:
        """Generate performance optimizations."""
        optimizations = []
        
        for entity in entities:
            # Index optimizations
            if len(entity.fields) > 10:
                opt = PerformanceOptimization(
                    entity_name=entity.name,
                    optimization_type="indexing",
                    description=f"Consider partitioning {entity.name} table for better performance",
                    impact_level="HIGH",
                    implementation_sql=f"-- Consider table partitioning for {entity.table_name}"
                )
                optimizations.append(opt)
            
            # Query optimization suggestions
            if any(f.name == "status" for f in entity.fields):
                opt = PerformanceOptimization(
                    entity_name=entity.name,
                    optimization_type="query",
                    description=f"Add covering index for status-based queries on {entity.name}",
                    impact_level="MEDIUM",
                    implementation_sql=f"CREATE INDEX idx_{entity.table_name}_status_covering ON {entity.table_name} (status) INCLUDE (id, created_at);"
                )
                optimizations.append(opt)
        
        return optimizations
    
    def _generate_sql_scripts(self, schema: DatabaseSchema) -> Dict[str, str]:
        """Generate SQL scripts for schema."""
        scripts = {}
        
        if schema.database_type == DatabaseType.POSTGRESQL:
            scripts["create_schema.sql"] = self._generate_postgresql_schema(schema)
        elif schema.database_type == DatabaseType.MYSQL:
            scripts["create_schema.sql"] = self._generate_mysql_schema(schema)
        elif schema.database_type == DatabaseType.MONGODB:
            scripts["create_schema.js"] = self._generate_mongodb_schema(schema)
        
        return scripts
    
    def _generate_postgresql_schema(self, schema: DatabaseSchema) -> str:
        """Generate PostgreSQL schema SQL."""
        sql_parts = []
        
        # Create tables
        for entity in schema.entities:
            table_sql = f"CREATE TABLE {entity.table_name} (\n"
            
            field_definitions = []
            for field in entity.fields:
                field_def = f"    {field.name} {self._get_postgresql_type(field)}"
                if not field.nullable:
                    field_def += " NOT NULL"
                if field.default_value:
                    field_def += f" DEFAULT {field.default_value}"
                field_definitions.append(field_def)
            
            table_sql += ",\n".join(field_definitions)
            table_sql += "\n);\n\n"
            sql_parts.append(table_sql)
            
            # Create indexes
            for index in entity.indexes:
                if index.type != IndexType.PRIMARY:
                    index_sql = f"CREATE {'UNIQUE ' if index.unique else ''}INDEX {index.name} ON {entity.table_name} ({', '.join(index.fields)});\n"
                    sql_parts.append(index_sql)
        
        return "".join(sql_parts)
    
    def _generate_mysql_schema(self, schema: DatabaseSchema) -> str:
        """Generate MySQL schema SQL."""
        sql_parts = []
        
        for entity in schema.entities:
            table_sql = f"CREATE TABLE {entity.table_name} (\n"
            
            field_definitions = []
            for field in entity.fields:
                field_def = f"    {field.name} {self._get_mysql_type(field)}"
                if not field.nullable:
                    field_def += " NOT NULL"
                if field.default_value:
                    field_def += f" DEFAULT {field.default_value}"
                field_definitions.append(field_def)
            
            # Add primary key
            field_definitions.append("    PRIMARY KEY (id)")
            
            table_sql += ",\n".join(field_definitions)
            table_sql += "\n) ENGINE=InnoDB;\n\n"
            sql_parts.append(table_sql)
        
        return "".join(sql_parts)
    
    def _generate_mongodb_schema(self, schema: DatabaseSchema) -> str:
        """Generate MongoDB schema JavaScript."""
        js_parts = []
        
        for entity in schema.entities:
            # Create collection
            js_parts.append(f"db.createCollection('{entity.table_name}');\n")
            
            # Create indexes
            for index in entity.indexes:
                if index.type != IndexType.PRIMARY:
                    index_spec = "{" + ", ".join([f'"{field}": 1' for field in index.fields]) + "}"
                    options = '{"unique": true}' if index.unique else '{}'
                    js_parts.append(f"db.{entity.table_name}.createIndex({index_spec}, {options});\n")
            
            js_parts.append("\n")
        
        return "".join(js_parts)
    
    def _get_postgresql_type(self, field: Field) -> str:
        """Get PostgreSQL type for field."""
        type_map = self.type_mappings[DatabaseType.POSTGRESQL]
        base_type = type_map.get(field.type, "VARCHAR")
        
        if field.type == FieldType.STRING and field.max_length:
            return f"{base_type}({field.max_length})"
        elif field.type == FieldType.DECIMAL and field.precision:
            scale = field.scale or 2
            return f"{base_type}({field.precision},{scale})"
        
        return base_type
    
    def _get_mysql_type(self, field: Field) -> str:
        """Get MySQL type for field."""
        type_map = self.type_mappings[DatabaseType.MYSQL]
        base_type = type_map.get(field.type, "VARCHAR")
        
        if field.type == FieldType.STRING and field.max_length:
            return f"{base_type}({field.max_length})"
        elif field.type == FieldType.DECIMAL and field.precision:
            scale = field.scale or 2
            return f"{base_type}({field.precision},{scale})"
        
        return base_type
    
    def _generate_documentation(self, schema: DatabaseSchema) -> str:
        """Generate schema documentation."""
        doc_parts = []
        
        doc_parts.append(f"# Database Schema Documentation\n\n")
        doc_parts.append(f"**Schema Name:** {schema.name}\n")
        doc_parts.append(f"**Database Type:** {schema.database_type.value}\n")
        doc_parts.append(f"**Version:** {schema.version}\n")
        doc_parts.append(f"**Generated:** {schema.created_at.isoformat()}\n\n")
        
        if schema.description:
            doc_parts.append(f"**Description:** {schema.description}\n\n")
        
        doc_parts.append("## Entities\n\n")
        
        for entity in schema.entities:
            doc_parts.append(f"### {entity.name}\n\n")
            if entity.description:
                doc_parts.append(f"{entity.description}\n\n")
            
            doc_parts.append("**Fields:**\n\n")
            doc_parts.append("| Field | Type | Nullable | Description |\n")
            doc_parts.append("|-------|------|----------|-------------|\n")
            
            for field in entity.fields:
                nullable = "Yes" if field.nullable else "No"
                description = field.description or ""
                doc_parts.append(f"| {field.name} | {field.type.value} | {nullable} | {description} |\n")
            
            doc_parts.append("\n")
            
            if entity.indexes:
                doc_parts.append("**Indexes:**\n\n")
                for index in entity.indexes:
                    doc_parts.append(f"- {index.name}: {', '.join(index.fields)} ({'unique' if index.unique else 'non-unique'})\n")
                doc_parts.append("\n")
        
        return "".join(doc_parts)
    
    def _to_table_name(self, entity_name: str) -> str:
        """Convert entity name to table name using naming conventions."""
        # Convert to snake_case and pluralize
        table_name = re.sub(r'(?<!^)(?=[A-Z])', '_', entity_name).lower()
        
        # Simple pluralization
        if table_name.endswith('y'):
            table_name = table_name[:-1] + 'ies'
        elif table_name.endswith(('s', 'sh', 'ch', 'x', 'z')):
            table_name += 'es'
        else:
            table_name += 's'
        
        return table_name
    
    def validate_schema(self, schema: DatabaseSchema) -> SchemaValidationResult:
        """Validate generated schema."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for missing primary keys
        for entity in schema.entities:
            if not any(c.type == ConstraintType.PRIMARY_KEY for c in entity.constraints):
                errors.append(f"Entity {entity.name} is missing a primary key")
        
        # Check for orphaned foreign keys
        entity_names = {e.name for e in schema.entities}
        for entity in schema.entities:
            for constraint in entity.constraints:
                if constraint.type == ConstraintType.FOREIGN_KEY:
                    if constraint.reference_table not in [e.table_name for e in schema.entities]:
                        errors.append(f"Foreign key in {entity.name} references non-existent table {constraint.reference_table}")
        
        # Performance suggestions
        for entity in schema.entities:
            if len(entity.fields) > 20:
                suggestions.append(f"Consider normalizing {entity.name} - it has many fields")
            
            if not any(i.type == IndexType.INDEX for i in entity.indexes if len(i.fields) > 1):
                suggestions.append(f"Consider adding composite indexes to {entity.name} for common query patterns")
        
        # Calculate scores
        performance_score = max(0.0, 1.0 - (len(warnings) * 0.1))
        normalization_score = 1.0  # Simplified calculation
        security_score = 1.0 if not errors else 0.5
        
        return SchemaValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            performance_score=performance_score,
            normalization_score=normalization_score,
            security_score=security_score
        )
    
    def generate_migration(self, old_schema: DatabaseSchema, new_schema: DatabaseSchema) -> Migration:
        """Generate migration between schema versions."""
        operations = []
        
        # Find new entities
        old_entity_names = {e.name for e in old_schema.entities}
        new_entity_names = {e.name for e in new_schema.entities}
        
        # Create new tables
        for entity in new_schema.entities:
            if entity.name not in old_entity_names:
                operation = MigrationOperation(
                    operation_type="CREATE_TABLE",
                    entity_name=entity.name,
                    details={"entity": entity.dict()},
                    sql_statement=self._generate_create_table_sql(entity, new_schema.database_type),
                    rollback_statement=f"DROP TABLE {entity.table_name};"
                )
                operations.append(operation)
        
        # Drop removed tables
        for entity in old_schema.entities:
            if entity.name not in new_entity_names:
                operation = MigrationOperation(
                    operation_type="DROP_TABLE",
                    entity_name=entity.name,
                    details={"table_name": entity.table_name},
                    sql_statement=f"DROP TABLE {entity.table_name};",
                    rollback_statement=self._generate_create_table_sql(entity, old_schema.database_type)
                )
                operations.append(operation)
        
        # Generate migration
        migration_id = str(uuid.uuid4())
        version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        up_sql = "\n".join([op.sql_statement for op in operations])
        down_sql = "\n".join([op.rollback_statement for op in reversed(operations) if op.rollback_statement])
        
        checksum = hashlib.md5(up_sql.encode()).hexdigest()
        
        return Migration(
            id=migration_id,
            version=version,
            description=f"Migration from {old_schema.version} to {new_schema.version}",
            database_type=new_schema.database_type,
            operations=operations,
            up_sql=up_sql,
            down_sql=down_sql,
            checksum=checksum
        )
    
    def _generate_create_table_sql(self, entity: Entity, db_type: DatabaseType) -> str:
        """Generate CREATE TABLE SQL for entity."""
        if db_type == DatabaseType.POSTGRESQL:
            return self._generate_postgresql_create_table(entity)
        elif db_type == DatabaseType.MYSQL:
            return self._generate_mysql_create_table(entity)
        else:
            return f"-- CREATE TABLE {entity.table_name} not implemented for {db_type}"
    
    def _generate_postgresql_create_table(self, entity: Entity) -> str:
        """Generate PostgreSQL CREATE TABLE statement."""
        sql = f"CREATE TABLE {entity.table_name} (\n"
        
        field_definitions = []
        for field in entity.fields:
            field_def = f"    {field.name} {self._get_postgresql_type(field)}"
            if not field.nullable:
                field_def += " NOT NULL"
            if field.default_value:
                field_def += f" DEFAULT {field.default_value}"
            field_definitions.append(field_def)
        
        sql += ",\n".join(field_definitions)
        sql += "\n);"
        
        return sql
    
    def _generate_mysql_create_table(self, entity: Entity) -> str:
        """Generate MySQL CREATE TABLE statement."""
        sql = f"CREATE TABLE {entity.table_name} (\n"
        
        field_definitions = []
        for field in entity.fields:
            field_def = f"    {field.name} {self._get_mysql_type(field)}"
            if not field.nullable:
                field_def += " NOT NULL"
            if field.default_value:
                field_def += f" DEFAULT {field.default_value}"
            field_definitions.append(field_def)
        
        # Add primary key
        field_definitions.append("    PRIMARY KEY (id)")
        
        sql += ",\n".join(field_definitions)
        sql += "\n) ENGINE=InnoDB;"
        
        return sql