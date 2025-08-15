"""
Data models for database schema generation system.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    SQLITE = "sqlite"
    ORACLE = "oracle"
    MSSQL = "mssql"


class FieldType(str, Enum):
    """Database field types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TEXT = "text"
    JSON = "json"
    UUID = "uuid"
    DECIMAL = "decimal"
    BINARY = "binary"
    ARRAY = "array"
    OBJECT = "object"  # For MongoDB


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class IndexType(str, Enum):
    """Types of database indexes."""
    PRIMARY = "primary"
    UNIQUE = "unique"
    INDEX = "index"
    COMPOSITE = "composite"
    PARTIAL = "partial"
    FULL_TEXT = "full_text"
    SPATIAL = "spatial"


class ConstraintType(str, Enum):
    """Types of database constraints."""
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    CHECK = "check"
    DEFAULT = "default"


class Field(BaseModel):
    """Represents a database field/column."""
    name: str
    type: FieldType
    nullable: bool = True
    default_value: Optional[Union[str, int, float, bool]] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    description: Optional[str] = None
    validation_rules: List[str] = []
    metadata: Dict[str, Any] = {}


class Index(BaseModel):
    """Represents a database index."""
    name: str
    type: IndexType
    fields: List[str]
    unique: bool = False
    partial_condition: Optional[str] = None
    description: Optional[str] = None


class Constraint(BaseModel):
    """Represents a database constraint."""
    name: str
    type: ConstraintType
    fields: List[str]
    reference_table: Optional[str] = None
    reference_fields: List[str] = []
    check_condition: Optional[str] = None
    on_delete: Optional[str] = None
    on_update: Optional[str] = None


class EntityRelationship(BaseModel):
    """Represents a relationship between database entities."""
    id: str
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    source_fields: List[str]
    target_fields: List[str]
    cascade_delete: bool = False
    cascade_update: bool = False
    description: Optional[str] = None


class Entity(BaseModel):
    """Represents a database entity/table."""
    name: str
    table_name: str
    description: Optional[str] = None
    fields: List[Field]
    indexes: List[Index] = []
    constraints: List[Constraint] = []
    relationships: List[EntityRelationship] = []
    estimated_rows: Optional[int] = None
    growth_rate: Optional[float] = None
    access_patterns: List[str] = []
    metadata: Dict[str, Any] = {}


class MigrationOperation(BaseModel):
    """Represents a database migration operation."""
    operation_type: str
    entity_name: str
    details: Dict[str, Any]
    sql_statement: str
    rollback_statement: Optional[str] = None
    dependencies: List[str] = []


class Migration(BaseModel):
    """Represents a database migration."""
    id: str
    version: str
    description: str
    database_type: DatabaseType
    operations: List[MigrationOperation]
    up_sql: str
    down_sql: str
    checksum: str
    created_at: datetime = None

    def __init__(self, **data):
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)


class PerformanceOptimization(BaseModel):
    """Represents performance optimization recommendations."""
    entity_name: str
    optimization_type: str
    description: str
    impact_level: str
    implementation_sql: str
    estimated_improvement: Optional[str] = None


class DatabaseSchema(BaseModel):
    """Represents a complete database schema."""
    id: str
    name: str
    database_type: DatabaseType
    version: str = "1.0.0"
    description: Optional[str] = None
    entities: List[Entity]
    relationships: List[EntityRelationship] = []
    migrations: List[Migration] = []
    optimizations: List[PerformanceOptimization] = []
    configuration: Dict[str, Any] = {}
    estimated_size: Optional[str] = None
    performance_requirements: Dict[str, Any] = {}
    security_settings: Dict[str, Any] = {}
    backup_strategy: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.utcnow()
        super().__init__(**data)


class SchemaGenerationRequest(BaseModel):
    """Request for schema generation."""
    requirements_id: str
    database_type: DatabaseType
    performance_requirements: Dict[str, Any] = {}
    scalability_requirements: Dict[str, Any] = {}
    security_requirements: Dict[str, Any] = {}
    naming_conventions: Dict[str, str] = {}
    optimization_level: str = "medium"


class SchemaGenerationResult(BaseModel):
    """Result of schema generation."""
    success: bool
    schema: Optional[DatabaseSchema] = None
    warnings: List[str] = []
    errors: List[str] = []
    generation_time: float
    sql_scripts: Dict[str, str] = {}
    documentation: Optional[str] = None


class SchemaValidationResult(BaseModel):
    """Result of schema validation."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []
    performance_score: float = 0.0
    normalization_score: float = 0.0
    security_score: float = 0.0