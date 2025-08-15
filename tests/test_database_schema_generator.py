"""
Unit tests for database schema generation engine.
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.models.database_schema_models import (
    DatabaseSchema, Entity, Field, Index, Constraint, EntityRelationship,
    SchemaGenerationRequest, SchemaGenerationResult, SchemaValidationResult,
    DatabaseType, FieldType, RelationshipType, IndexType, ConstraintType
)
from scrollintel.models.code_generation_models import (
    Requirements, ParsedRequirement, Entity as RequirementEntity,
    Relationship, RequirementType, Intent, EntityType
)
from scrollintel.engines.database_schema_generator import DatabaseSchemaGenerator


class TestDatabaseSchemaGenerator:
    """Test cases for DatabaseSchemaGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create a DatabaseSchemaGenerator instance."""
        return DatabaseSchemaGenerator()
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample requirements for testing."""
        user_entity = RequirementEntity(
            id=str(uuid.uuid4()),
            name="User",
            type=EntityType.DATA_ENTITY,
            description="User entity for authentication",
            confidence=0.9,
            source_text="Users need to authenticate",
            position=(0, 25)
        )
        
        order_entity = RequirementEntity(
            id=str(uuid.uuid4()),
            name="Order",
            type=EntityType.DATA_ENTITY,
            description="Order entity for e-commerce",
            confidence=0.8,
            source_text="Orders need to be tracked",
            position=(26, 50)
        )
        
        relationship = Relationship(
            id=str(uuid.uuid4()),
            source_entity_id="Order",
            target_entity_id="User",
            relationship_type="many_to_one",
            description="Orders belong to users",
            confidence=0.9
        )
        
        return Requirements(
            id=str(uuid.uuid4()),
            project_name="Test Project",
            raw_text="Users need to authenticate. Orders need to be tracked.",
            entities=[user_entity, order_entity],
            relationships=[relationship]
        )
    
    @pytest.fixture
    def sample_request(self):
        """Create sample schema generation request."""
        return SchemaGenerationRequest(
            requirements_id=str(uuid.uuid4()),
            database_type=DatabaseType.POSTGRESQL,
            performance_requirements={"max_query_time": "100ms"},
            scalability_requirements={"max_concurrent_users": 10000},
            optimization_level="medium"
        )
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert generator.type_mappings is not None
        assert generator.naming_conventions is not None
        assert DatabaseType.POSTGRESQL in generator.type_mappings
        assert DatabaseType.MYSQL in generator.type_mappings
        assert DatabaseType.MONGODB in generator.type_mappings
    
    def test_extract_entities_from_requirements(self, generator, sample_requirements):
        """Test entity extraction from requirements."""
        entities = generator._extract_entities_from_requirements(sample_requirements)
        
        assert len(entities) >= 2
        entity_names = [e.name for e in entities]
        assert "User" in entity_names
        assert "Order" in entity_names
    
    def test_generate_database_entities(self, generator, sample_requirements):
        """Test database entity generation."""
        req_entities = generator._extract_entities_from_requirements(sample_requirements)
        db_entities = generator._generate_database_entities(req_entities, DatabaseType.POSTGRESQL)
        
        assert len(db_entities) >= 2
        
        # Check user entity
        user_entity = next((e for e in db_entities if e.name == "User"), None)
        assert user_entity is not None
        assert user_entity.table_name == "users"
        
        # Check required fields
        field_names = [f.name for f in user_entity.fields]
        assert "id" in field_names
        assert "created_at" in field_names
        assert "updated_at" in field_names
        
        # Check indexes
        assert len(user_entity.indexes) > 0
        pk_index = next((i for i in user_entity.indexes if i.type == IndexType.PRIMARY), None)
        assert pk_index is not None
        
        # Check constraints
        assert len(user_entity.constraints) > 0
        pk_constraint = next((c for c in user_entity.constraints if c.type == ConstraintType.PRIMARY_KEY), None)
        assert pk_constraint is not None
    
    def test_generate_entity_fields(self, generator):
        """Test entity field generation."""
        user_entity = RequirementEntity(
            id=str(uuid.uuid4()),
            name="User",
            type=EntityType.DATA_ENTITY,
            description="User entity",
            confidence=0.9,
            source_text="User entity",
            position=(0, 10)
        )
        
        fields = generator._generate_entity_fields(user_entity, DatabaseType.POSTGRESQL)
        
        assert len(fields) > 0
        field_names = [f.name for f in fields]
        
        # Check required fields
        assert "id" in field_names
        assert "created_at" in field_names
        assert "updated_at" in field_names
        
        # Check user-specific fields
        assert "username" in field_names or "email" in field_names
        
        # Check field properties
        id_field = next(f for f in fields if f.name == "id")
        assert id_field.type == FieldType.UUID
        assert not id_field.nullable
    
    def test_get_common_fields_for_entity(self, generator):
        """Test common field generation for different entity types."""
        # Test user entity
        user_fields = generator._get_common_fields_for_entity("user")
        assert "username" in user_fields
        assert "email" in user_fields
        assert "password_hash" in user_fields
        
        # Test order entity
        order_fields = generator._get_common_fields_for_entity("order")
        assert "order_number" in order_fields
        assert "total_amount" in order_fields
        assert "status" in order_fields
        
        # Test default entity
        default_fields = generator._get_common_fields_for_entity("unknown")
        assert "name" in default_fields
        assert "status" in default_fields
    
    def test_establish_relationships(self, generator, sample_requirements):
        """Test relationship establishment."""
        req_entities = generator._extract_entities_from_requirements(sample_requirements)
        db_entities = generator._generate_database_entities(req_entities, DatabaseType.POSTGRESQL)
        relationships = generator._establish_relationships(db_entities, sample_requirements)
        
        assert len(relationships) > 0
        
        # Check relationship properties
        rel = relationships[0]
        assert rel.source_entity in ["User", "Order"]
        assert rel.target_entity in ["User", "Order"]
        assert rel.relationship_type in [RelationshipType.ONE_TO_MANY, RelationshipType.MANY_TO_ONE]
        
        # Check foreign key was added
        source_entity = generator._find_entity_by_name(db_entities, rel.source_entity)
        if source_entity:
            fk_fields = [f for f in source_entity.fields if f.name.endswith("_id")]
            assert len(fk_fields) > 0
    
    def test_generate_optimizations(self, generator, sample_request):
        """Test optimization generation."""
        # Create sample entities
        entities = [
            Entity(
                name="User",
                table_name="users",
                fields=[Field(name=f"field_{i}", type=FieldType.STRING) for i in range(15)],
                indexes=[],
                constraints=[]
            ),
            Entity(
                name="Order",
                table_name="orders",
                fields=[
                    Field(name="id", type=FieldType.UUID),
                    Field(name="status", type=FieldType.STRING),
                    Field(name="created_at", type=FieldType.DATETIME)
                ],
                indexes=[],
                constraints=[]
            )
        ]
        
        optimizations = generator._generate_optimizations(entities, sample_request)
        
        assert len(optimizations) > 0
        
        # Check optimization properties
        opt = optimizations[0]
        assert opt.entity_name in ["User", "Order"]
        assert opt.optimization_type in ["indexing", "query", "partitioning"]
        assert opt.impact_level in ["HIGH", "MEDIUM", "LOW"]
        assert opt.implementation_sql is not None
    
    def test_generate_postgresql_schema(self, generator):
        """Test PostgreSQL schema generation."""
        entity = Entity(
            name="User",
            table_name="users",
            fields=[
                Field(name="id", type=FieldType.UUID, nullable=False),
                Field(name="username", type=FieldType.STRING, nullable=False, max_length=50),
                Field(name="email", type=FieldType.STRING, nullable=True, max_length=255),
                Field(name="created_at", type=FieldType.DATETIME, nullable=False, default_value="CURRENT_TIMESTAMP")
            ],
            indexes=[
                Index(name="pk_users", type=IndexType.PRIMARY, fields=["id"], unique=True),
                Index(name="idx_users_username", type=IndexType.INDEX, fields=["username"], unique=True)
            ],
            constraints=[]
        )
        
        schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="test_schema",
            database_type=DatabaseType.POSTGRESQL,
            entities=[entity]
        )
        
        sql = generator._generate_postgresql_schema(schema)
        
        assert "CREATE TABLE users" in sql
        assert "id UUID NOT NULL" in sql
        assert "username VARCHAR(50) NOT NULL" in sql
        assert "email VARCHAR(255)" in sql
        assert "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP" in sql
        assert "CREATE UNIQUE INDEX idx_users_username" in sql
    
    def test_generate_mysql_schema(self, generator):
        """Test MySQL schema generation."""
        entity = Entity(
            name="User",
            table_name="users",
            fields=[
                Field(name="id", type=FieldType.STRING, nullable=False, max_length=36),
                Field(name="username", type=FieldType.STRING, nullable=False, max_length=50)
            ],
            indexes=[],
            constraints=[]
        )
        
        schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="test_schema",
            database_type=DatabaseType.MYSQL,
            entities=[entity]
        )
        
        sql = generator._generate_mysql_schema(schema)
        
        assert "CREATE TABLE users" in sql
        assert "id VARCHAR(36) NOT NULL" in sql
        assert "username VARCHAR(50) NOT NULL" in sql
        assert "PRIMARY KEY (id)" in sql
        assert "ENGINE=InnoDB" in sql
    
    def test_generate_mongodb_schema(self, generator):
        """Test MongoDB schema generation."""
        entity = Entity(
            name="User",
            table_name="users",
            fields=[
                Field(name="_id", type=FieldType.OBJECT),
                Field(name="username", type=FieldType.STRING)
            ],
            indexes=[
                Index(name="idx_users_username", type=IndexType.INDEX, fields=["username"], unique=True)
            ],
            constraints=[]
        )
        
        schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="test_schema",
            database_type=DatabaseType.MONGODB,
            entities=[entity]
        )
        
        js = generator._generate_mongodb_schema(schema)
        
        assert "db.createCollection('users')" in js
        assert "db.users.createIndex" in js
        assert '"username": 1' in js
        assert '"unique": true' in js
    
    def test_validate_schema(self, generator):
        """Test schema validation."""
        # Valid schema
        valid_entity = Entity(
            name="User",
            table_name="users",
            fields=[Field(name="id", type=FieldType.UUID, nullable=False)],
            indexes=[Index(name="pk_users", type=IndexType.PRIMARY, fields=["id"])],
            constraints=[Constraint(name="pk_users", type=ConstraintType.PRIMARY_KEY, fields=["id"])]
        )
        
        valid_schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="valid_schema",
            database_type=DatabaseType.POSTGRESQL,
            entities=[valid_entity]
        )
        
        result = generator.validate_schema(valid_schema)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.performance_score > 0
        assert result.normalization_score > 0
        assert result.security_score > 0
        
        # Invalid schema (missing primary key)
        invalid_entity = Entity(
            name="User",
            table_name="users",
            fields=[Field(name="id", type=FieldType.UUID, nullable=False)],
            indexes=[],
            constraints=[]
        )
        
        invalid_schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="invalid_schema",
            database_type=DatabaseType.POSTGRESQL,
            entities=[invalid_entity]
        )
        
        result = generator.validate_schema(invalid_schema)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "missing a primary key" in result.errors[0]
    
    def test_generate_migration(self, generator):
        """Test migration generation."""
        # Old schema
        old_entity = Entity(
            name="User",
            table_name="users",
            fields=[
                Field(name="id", type=FieldType.UUID, nullable=False),
                Field(name="username", type=FieldType.STRING, nullable=False)
            ],
            indexes=[],
            constraints=[]
        )
        
        old_schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="old_schema",
            database_type=DatabaseType.POSTGRESQL,
            version="1.0.0",
            entities=[old_entity]
        )
        
        # New schema with additional entity
        new_entity = Entity(
            name="Order",
            table_name="orders",
            fields=[
                Field(name="id", type=FieldType.UUID, nullable=False),
                Field(name="order_number", type=FieldType.STRING, nullable=False)
            ],
            indexes=[],
            constraints=[]
        )
        
        new_schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="new_schema",
            database_type=DatabaseType.POSTGRESQL,
            version="2.0.0",
            entities=[old_entity, new_entity]
        )
        
        migration = generator.generate_migration(old_schema, new_schema)
        
        assert migration is not None
        assert migration.database_type == DatabaseType.POSTGRESQL
        assert len(migration.operations) > 0
        assert migration.up_sql is not None
        assert migration.down_sql is not None
        assert migration.checksum is not None
        
        # Check for CREATE_TABLE operation
        create_ops = [op for op in migration.operations if op.operation_type == "CREATE_TABLE"]
        assert len(create_ops) > 0
        assert create_ops[0].entity_name == "Order"
    
    def test_to_table_name(self, generator):
        """Test table name conversion."""
        assert generator._to_table_name("User") == "users"
        assert generator._to_table_name("OrderItem") == "order_items"
        assert generator._to_table_name("Company") == "companies"
        assert generator._to_table_name("Address") == "addresses"
    
    def test_get_postgresql_type(self, generator):
        """Test PostgreSQL type mapping."""
        # String with length
        string_field = Field(name="test", type=FieldType.STRING, max_length=100)
        assert generator._get_postgresql_type(string_field) == "VARCHAR(100)"
        
        # Decimal with precision
        decimal_field = Field(name="test", type=FieldType.DECIMAL, precision=10, scale=2)
        assert generator._get_postgresql_type(decimal_field) == "DECIMAL(10,2)"
        
        # Basic types
        uuid_field = Field(name="test", type=FieldType.UUID)
        assert generator._get_postgresql_type(uuid_field) == "UUID"
        
        boolean_field = Field(name="test", type=FieldType.BOOLEAN)
        assert generator._get_postgresql_type(boolean_field) == "BOOLEAN"
    
    def test_get_mysql_type(self, generator):
        """Test MySQL type mapping."""
        # String with length
        string_field = Field(name="test", type=FieldType.STRING, max_length=100)
        assert generator._get_mysql_type(string_field) == "VARCHAR(100)"
        
        # UUID (mapped to CHAR in MySQL)
        uuid_field = Field(name="test", type=FieldType.UUID)
        assert generator._get_mysql_type(uuid_field) == "CHAR(36)"
        
        # Integer
        int_field = Field(name="test", type=FieldType.INTEGER)
        assert generator._get_mysql_type(int_field) == "INT"
    
    def test_generate_schema_full_workflow(self, generator, sample_request, sample_requirements):
        """Test complete schema generation workflow."""
        result = generator.generate_schema(sample_request, sample_requirements)
        
        assert result.success
        assert result.schema is not None
        assert result.generation_time > 0
        assert len(result.sql_scripts) > 0
        assert result.documentation is not None
        
        # Check schema properties
        schema = result.schema
        assert schema.name == f"{sample_requirements.project_name}_schema"
        assert schema.database_type == sample_request.database_type
        assert len(schema.entities) >= 2
        
        # Check SQL scripts
        assert "create_schema.sql" in result.sql_scripts
        sql_content = result.sql_scripts["create_schema.sql"]
        assert "CREATE TABLE" in sql_content
        
        # Check documentation
        assert "# Database Schema Documentation" in result.documentation
        assert schema.name in result.documentation
    
    def test_error_handling(self, generator):
        """Test error handling in schema generation."""
        # Invalid request
        invalid_request = SchemaGenerationRequest(
            requirements_id="invalid_id",
            database_type=DatabaseType.POSTGRESQL
        )
        
        invalid_requirements = Requirements(
            id="invalid_id",
            project_name="",
            raw_text="",
            entities=[],
            relationships=[]
        )
        
        # Should handle gracefully
        result = generator.generate_schema(invalid_request, invalid_requirements)
        assert result is not None
        # May succeed with empty entities or fail gracefully
    
    @pytest.mark.parametrize("db_type", [DatabaseType.POSTGRESQL, DatabaseType.MYSQL, DatabaseType.MONGODB])
    def test_multi_database_support(self, generator, sample_request, sample_requirements, db_type):
        """Test schema generation for different database types."""
        sample_request.database_type = db_type
        result = generator.generate_schema(sample_request, sample_requirements)
        
        assert result.success
        assert result.schema.database_type == db_type
        assert len(result.sql_scripts) > 0
        
        # Check database-specific SQL
        if db_type == DatabaseType.MONGODB:
            assert any(".js" in filename for filename in result.sql_scripts.keys())
        else:
            assert any(".sql" in filename for filename in result.sql_scripts.keys())


class TestDatabaseSchemaModels:
    """Test cases for database schema models."""
    
    def test_field_model(self):
        """Test Field model."""
        field = Field(
            name="username",
            type=FieldType.STRING,
            nullable=False,
            max_length=50,
            description="User's username"
        )
        
        assert field.name == "username"
        assert field.type == FieldType.STRING
        assert not field.nullable
        assert field.max_length == 50
        assert field.description == "User's username"
    
    def test_entity_model(self):
        """Test Entity model."""
        field = Field(name="id", type=FieldType.UUID, nullable=False)
        index = Index(name="pk_users", type=IndexType.PRIMARY, fields=["id"])
        constraint = Constraint(name="pk_users", type=ConstraintType.PRIMARY_KEY, fields=["id"])
        
        entity = Entity(
            name="User",
            table_name="users",
            description="User entity",
            fields=[field],
            indexes=[index],
            constraints=[constraint]
        )
        
        assert entity.name == "User"
        assert entity.table_name == "users"
        assert len(entity.fields) == 1
        assert len(entity.indexes) == 1
        assert len(entity.constraints) == 1
    
    def test_database_schema_model(self):
        """Test DatabaseSchema model."""
        entity = Entity(
            name="User",
            table_name="users",
            fields=[Field(name="id", type=FieldType.UUID, nullable=False)],
            indexes=[],
            constraints=[]
        )
        
        schema = DatabaseSchema(
            id=str(uuid.uuid4()),
            name="test_schema",
            database_type=DatabaseType.POSTGRESQL,
            entities=[entity]
        )
        
        assert schema.name == "test_schema"
        assert schema.database_type == DatabaseType.POSTGRESQL
        assert len(schema.entities) == 1
        assert schema.version == "1.0.0"  # default value
    
    def test_schema_generation_request(self):
        """Test SchemaGenerationRequest model."""
        request = SchemaGenerationRequest(
            requirements_id=str(uuid.uuid4()),
            database_type=DatabaseType.MYSQL,
            performance_requirements={"max_query_time": "50ms"},
            optimization_level="high"
        )
        
        assert request.database_type == DatabaseType.MYSQL
        assert request.performance_requirements["max_query_time"] == "50ms"
        assert request.optimization_level == "high"
    
    def test_schema_generation_result(self):
        """Test SchemaGenerationResult model."""
        result = SchemaGenerationResult(
            success=True,
            generation_time=1.5,
            sql_scripts={"create.sql": "CREATE TABLE test;"},
            documentation="Test documentation"
        )
        
        assert result.success
        assert result.generation_time == 1.5
        assert "create.sql" in result.sql_scripts
        assert result.documentation == "Test documentation"


if __name__ == "__main__":
    pytest.main([__file__])