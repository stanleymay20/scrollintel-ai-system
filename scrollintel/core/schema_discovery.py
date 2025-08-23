"""
Schema Discovery and Metadata Extraction System
Discovers database schemas, extracts metadata, and maps data types
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy import text, inspect, MetaData, Table, Column
from sqlalchemy.engine.reflection import Inspector

from ..models.enterprise_connection_models import (
    DataSchema, TableMetadata, ConnectionType
)
from ..connectors.database_connectors import DatabaseConnector

logger = logging.getLogger(__name__)

@dataclass
class ColumnInfo:
    """Column information structure"""
    name: str
    data_type: str
    nullable: bool
    default_value: Any = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None

@dataclass
class IndexInfo:
    """Index information structure"""
    name: str
    columns: List[str]
    is_unique: bool
    is_primary: bool
    index_type: str

class SchemaDiscoveryEngine:
    """Engine for discovering and extracting database schema information"""
    
    def __init__(self, connector: DatabaseConnector, connection_type: ConnectionType):
        self.connector = connector
        self.connection_type = connection_type
        
    async def discover_full_schema(self, schema_name: Optional[str] = None) -> DataSchema:
        """Discover complete database schema"""
        try:
            logger.info(f"Starting schema discovery for {self.connection_type}")
            
            # Get basic schema info
            schema_info = await self.connector.get_schema_info()
            
            # Enhance with detailed metadata
            enhanced_tables = []
            for table_info in schema_info.tables:
                table_name = table_info.get('table_name') or table_info.get('TABLE_NAME')
                if table_name:
                    metadata = await self.get_table_metadata(table_name, schema_name)
                    enhanced_tables.append({
                        **table_info,
                        'metadata': metadata.__dict__ if metadata else None
                    })
            
            schema_info.tables = enhanced_tables
            return schema_info
            
        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")
            raise
    
    async def get_table_metadata(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> Optional[TableMetadata]:
        """Get detailed metadata for a specific table"""
        try:
            # Get column information
            columns = await self._get_column_info(table_name, schema_name)
            
            # Get primary keys
            primary_keys = await self._get_primary_keys(table_name, schema_name)
            
            # Get foreign keys
            foreign_keys = await self._get_foreign_keys(table_name, schema_name)
            
            # Get indexes
            indexes = await self._get_indexes(table_name, schema_name)
            
            # Get table statistics
            row_count, size_mb = await self._get_table_statistics(table_name, schema_name)
            
            return TableMetadata(
                table_name=table_name,
                schema_name=schema_name or 'default',
                columns=[col.__dict__ for col in columns],
                primary_keys=primary_keys,
                foreign_keys=[fk.__dict__ for fk in foreign_keys],
                indexes=[idx.__dict__ for idx in indexes],
                row_count=row_count,
                size_mb=size_mb
            )
            
        except Exception as e:
            logger.error(f"Failed to get metadata for table {table_name}: {e}")
            return None
    
    async def _get_column_info(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[ColumnInfo]:
        """Get column information for a table"""
        try:
            if self.connection_type == ConnectionType.POSTGRESQL:
                return await self._get_postgresql_columns(table_name, schema_name)
            elif self.connection_type == ConnectionType.MYSQL:
                return await self._get_mysql_columns(table_name, schema_name)
            elif self.connection_type == ConnectionType.SQL_SERVER:
                return await self._get_sqlserver_columns(table_name, schema_name)
            elif self.connection_type == ConnectionType.ORACLE:
                return await self._get_oracle_columns(table_name, schema_name)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get column info for {table_name}: {e}")
            return []
    
    async def _get_postgresql_columns(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[ColumnInfo]:
        """Get PostgreSQL column information"""
        query = text("""
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                CASE WHEN fk.column_name IS NOT NULL THEN true ELSE false END as is_foreign_key,
                fk.foreign_table_name,
                fk.foreign_column_name
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku 
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_name = :table_name
                    AND tc.table_schema = :schema_name
            ) pk ON c.column_name = pk.column_name
            LEFT JOIN (
                SELECT 
                    ku.column_name,
                    ccu.table_name as foreign_table_name,
                    ccu.column_name as foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku 
                    ON tc.constraint_name = ku.constraint_name
                JOIN information_schema.constraint_column_usage ccu 
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_name = :table_name
                    AND tc.table_schema = :schema_name
            ) fk ON c.column_name = fk.column_name
            WHERE c.table_name = :table_name 
                AND c.table_schema = :schema_name
            ORDER BY c.ordinal_position
        """)
        
        schema = schema_name or 'public'
        result = await self.connector.execute_query(
            str(query), 
            {"table_name": table_name, "schema_name": schema}
        )
        
        columns = []
        for row in result:
            columns.append(ColumnInfo(
                name=row['column_name'],
                data_type=row['data_type'],
                nullable=row['is_nullable'] == 'YES',
                default_value=row['column_default'],
                max_length=row['character_maximum_length'],
                precision=row['numeric_precision'],
                scale=row['numeric_scale'],
                is_primary_key=row['is_primary_key'],
                is_foreign_key=row['is_foreign_key'],
                foreign_key_table=row.get('foreign_table_name'),
                foreign_key_column=row.get('foreign_column_name')
            ))
        
        return columns
    
    async def _get_mysql_columns(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[ColumnInfo]:
        """Get MySQL column information"""
        query = text("""
            SELECT 
                COLUMN_NAME as column_name,
                DATA_TYPE as data_type,
                IS_NULLABLE as is_nullable,
                COLUMN_DEFAULT as column_default,
                CHARACTER_MAXIMUM_LENGTH as character_maximum_length,
                NUMERIC_PRECISION as numeric_precision,
                NUMERIC_SCALE as numeric_scale,
                CASE WHEN COLUMN_KEY = 'PRI' THEN true ELSE false END as is_primary_key,
                CASE WHEN COLUMN_KEY = 'MUL' THEN true ELSE false END as is_foreign_key
            FROM information_schema.COLUMNS
            WHERE TABLE_NAME = :table_name 
                AND TABLE_SCHEMA = :schema_name
            ORDER BY ORDINAL_POSITION
        """)
        
        result = await self.connector.execute_query(
            str(query), 
            {"table_name": table_name, "schema_name": schema_name}
        )
        
        columns = []
        for row in result:
            columns.append(ColumnInfo(
                name=row['column_name'],
                data_type=row['data_type'],
                nullable=row['is_nullable'] == 'YES',
                default_value=row['column_default'],
                max_length=row['character_maximum_length'],
                precision=row['numeric_precision'],
                scale=row['numeric_scale'],
                is_primary_key=row['is_primary_key'],
                is_foreign_key=row['is_foreign_key']
            ))
        
        return columns
    
    async def _get_sqlserver_columns(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[ColumnInfo]:
        """Get SQL Server column information"""
        query = text("""
            SELECT 
                c.COLUMN_NAME as column_name,
                c.DATA_TYPE as data_type,
                c.IS_NULLABLE as is_nullable,
                c.COLUMN_DEFAULT as column_default,
                c.CHARACTER_MAXIMUM_LENGTH as character_maximum_length,
                c.NUMERIC_PRECISION as numeric_precision,
                c.NUMERIC_SCALE as numeric_scale,
                CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as is_primary_key,
                CASE WHEN fk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as is_foreign_key
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN (
                SELECT ku.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku 
                    ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                    AND tc.TABLE_NAME = :table_name
                    AND tc.TABLE_SCHEMA = :schema_name
            ) pk ON c.COLUMN_NAME = pk.COLUMN_NAME
            LEFT JOIN (
                SELECT ku.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku 
                    ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                    AND tc.TABLE_NAME = :table_name
                    AND tc.TABLE_SCHEMA = :schema_name
            ) fk ON c.COLUMN_NAME = fk.COLUMN_NAME
            WHERE c.TABLE_NAME = :table_name 
                AND c.TABLE_SCHEMA = :schema_name
            ORDER BY c.ORDINAL_POSITION
        """)
        
        schema = schema_name or 'dbo'
        result = await self.connector.execute_query(
            str(query), 
            {"table_name": table_name, "schema_name": schema}
        )
        
        columns = []
        for row in result:
            columns.append(ColumnInfo(
                name=row['column_name'],
                data_type=row['data_type'],
                nullable=row['is_nullable'] == 'YES',
                default_value=row['column_default'],
                max_length=row['character_maximum_length'],
                precision=row['numeric_precision'],
                scale=row['numeric_scale'],
                is_primary_key=bool(row['is_primary_key']),
                is_foreign_key=bool(row['is_foreign_key'])
            ))
        
        return columns
    
    async def _get_oracle_columns(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[ColumnInfo]:
        """Get Oracle column information"""
        query = text("""
            SELECT 
                COLUMN_NAME as column_name,
                DATA_TYPE as data_type,
                NULLABLE as is_nullable,
                DATA_DEFAULT as column_default,
                DATA_LENGTH as character_maximum_length,
                DATA_PRECISION as numeric_precision,
                DATA_SCALE as numeric_scale
            FROM ALL_TAB_COLUMNS
            WHERE TABLE_NAME = :table_name 
                AND OWNER = :schema_name
            ORDER BY COLUMN_ID
        """)
        
        result = await self.connector.execute_query(
            str(query), 
            {"table_name": table_name, "schema_name": schema_name}
        )
        
        columns = []
        for row in result:
            columns.append(ColumnInfo(
                name=row['column_name'],
                data_type=row['data_type'],
                nullable=row['is_nullable'] == 'Y',
                default_value=row['column_default'],
                max_length=row['character_maximum_length'],
                precision=row['numeric_precision'],
                scale=row['numeric_scale']
            ))
        
        return columns
    
    async def _get_primary_keys(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[str]:
        """Get primary key columns"""
        try:
            if self.connection_type == ConnectionType.POSTGRESQL:
                query = text("""
                    SELECT ku.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage ku 
                        ON tc.constraint_name = ku.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                        AND tc.table_name = :table_name
                        AND tc.table_schema = :schema_name
                    ORDER BY ku.ordinal_position
                """)
                schema = schema_name or 'public'
            elif self.connection_type == ConnectionType.MYSQL:
                query = text("""
                    SELECT COLUMN_NAME as column_name
                    FROM information_schema.KEY_COLUMN_USAGE
                    WHERE CONSTRAINT_NAME = 'PRIMARY'
                        AND TABLE_NAME = :table_name
                        AND TABLE_SCHEMA = :schema_name
                    ORDER BY ORDINAL_POSITION
                """)
                schema = schema_name
            else:
                return []
            
            result = await self.connector.execute_query(
                str(query), 
                {"table_name": table_name, "schema_name": schema}
            )
            
            return [row['column_name'] for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get primary keys for {table_name}: {e}")
            return []
    
    async def _get_foreign_keys(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get foreign key information"""
        try:
            # Implementation varies by database type
            return []  # Simplified for now
        except Exception as e:
            logger.error(f"Failed to get foreign keys for {table_name}: {e}")
            return []
    
    async def _get_indexes(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> List[IndexInfo]:
        """Get index information"""
        try:
            # Implementation varies by database type
            return []  # Simplified for now
        except Exception as e:
            logger.error(f"Failed to get indexes for {table_name}: {e}")
            return []
    
    async def _get_table_statistics(
        self, 
        table_name: str, 
        schema_name: Optional[str] = None
    ) -> Tuple[Optional[int], Optional[float]]:
        """Get table statistics (row count, size)"""
        try:
            if self.connection_type == ConnectionType.POSTGRESQL:
                query = text(f"""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        most_common_vals,
                        most_common_freqs,
                        histogram_bounds
                    FROM pg_stats 
                    WHERE tablename = :table_name 
                        AND schemaname = :schema_name
                    LIMIT 1
                """)
                
                # Get row count
                count_query = text(f'SELECT COUNT(*) as row_count FROM "{schema_name or "public"}"."{table_name}"')
                count_result = await self.connector.execute_query(str(count_query))
                row_count = count_result[0]['row_count'] if count_result else None
                
                return row_count, None  # Size calculation would need pg_relation_size
            
            return None, None
            
        except Exception as e:
            logger.error(f"Failed to get statistics for {table_name}: {e}")
            return None, None

class DataTypeMapper:
    """Maps database-specific data types to standard types"""
    
    TYPE_MAPPINGS = {
        ConnectionType.POSTGRESQL: {
            'integer': 'int',
            'bigint': 'bigint',
            'smallint': 'smallint',
            'numeric': 'decimal',
            'real': 'float',
            'double precision': 'double',
            'character varying': 'varchar',
            'character': 'char',
            'text': 'text',
            'boolean': 'boolean',
            'date': 'date',
            'timestamp without time zone': 'datetime',
            'timestamp with time zone': 'datetimetz',
            'time': 'time',
            'json': 'json',
            'jsonb': 'json',
            'uuid': 'uuid'
        },
        ConnectionType.MYSQL: {
            'int': 'int',
            'bigint': 'bigint',
            'smallint': 'smallint',
            'tinyint': 'tinyint',
            'decimal': 'decimal',
            'float': 'float',
            'double': 'double',
            'varchar': 'varchar',
            'char': 'char',
            'text': 'text',
            'longtext': 'longtext',
            'boolean': 'boolean',
            'date': 'date',
            'datetime': 'datetime',
            'timestamp': 'timestamp',
            'time': 'time',
            'json': 'json'
        },
        ConnectionType.SQL_SERVER: {
            'int': 'int',
            'bigint': 'bigint',
            'smallint': 'smallint',
            'tinyint': 'tinyint',
            'decimal': 'decimal',
            'numeric': 'decimal',
            'float': 'float',
            'real': 'float',
            'varchar': 'varchar',
            'char': 'char',
            'text': 'text',
            'nvarchar': 'nvarchar',
            'nchar': 'nchar',
            'ntext': 'ntext',
            'bit': 'boolean',
            'date': 'date',
            'datetime': 'datetime',
            'datetime2': 'datetime',
            'time': 'time',
            'uniqueidentifier': 'uuid'
        },
        ConnectionType.ORACLE: {
            'NUMBER': 'decimal',
            'VARCHAR2': 'varchar',
            'CHAR': 'char',
            'CLOB': 'text',
            'DATE': 'datetime',
            'TIMESTAMP': 'timestamp',
            'RAW': 'binary',
            'BLOB': 'blob'
        }
    }
    
    @classmethod
    def map_type(cls, db_type: str, connection_type: ConnectionType) -> str:
        """Map database-specific type to standard type"""
        mappings = cls.TYPE_MAPPINGS.get(connection_type, {})
        return mappings.get(db_type.lower(), db_type)