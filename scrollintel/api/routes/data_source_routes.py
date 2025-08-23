"""
API routes for data source connectivity management.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from scrollintel.core.connection_manager import ConnectionManager
from scrollintel.models.data_source_models import (
    DataSourceConfig, ConnectionTest, DataSchema,
    DataSourceType, ConnectionStatus
)
from scrollintel.models.database import get_db

router = APIRouter(prefix="/api/data-sources", tags=["data-sources"])

# Pydantic models for API
class DataSourceConfigCreate(BaseModel):
    name: str = Field(..., description="Name of the data source")
    description: Optional[str] = Field(None, description="Description of the data source")
    source_type: str = Field(..., description="Type of data source (database, rest_api, graphql, file_system, streaming)")
    connection_config: Dict[str, Any] = Field(..., description="Connection configuration")
    auth_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Authentication configuration")

class DataSourceConfigUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    connection_config: Optional[Dict[str, Any]] = None
    auth_config: Optional[Dict[str, Any]] = None

class DataSourceConfigResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    source_type: str
    status: str
    last_tested: Optional[datetime]
    last_connected: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]

    class Config:
        from_attributes = True

class ConnectionTestResponse(BaseModel):
    id: str
    data_source_id: str
    test_timestamp: datetime
    success: bool
    error_message: Optional[str]
    response_time_ms: Optional[int]
    test_details: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True

class DataSchemaResponse(BaseModel):
    id: str
    data_source_id: str
    schema_name: Optional[str]
    table_name: Optional[str]
    columns: List[Dict[str, Any]]
    discovered_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class DataQueryRequest(BaseModel):
    query_config: Dict[str, Any] = Field(..., description="Query configuration specific to data source type")
    limit: Optional[int] = Field(1000, description="Maximum number of records to return")

def get_connection_manager(db: Session = Depends(get_db)) -> ConnectionManager:
    """Get connection manager instance."""
    return ConnectionManager(db)

@router.post("/", response_model=DataSourceConfigResponse)
async def create_data_source(
    config_data: DataSourceConfigCreate,
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    current_user: str = "system"  # TODO: Get from auth
):
    """Create a new data source configuration."""
    try:
        config_dict = config_data.dict()
        config_dict["created_by"] = current_user
        
        config = await connection_manager.create_data_source(config_dict)
        return DataSourceConfigResponse.from_orm(config)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[DataSourceConfigResponse])
async def list_data_sources(
    source_type: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all data source configurations."""
    query = db.query(DataSourceConfig)
    
    if source_type:
        try:
            source_type_enum = DataSourceType(source_type)
            query = query.filter(DataSourceConfig.source_type == source_type_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type}")
    
    if status:
        try:
            status_enum = ConnectionStatus(status)
            query = query.filter(DataSourceConfig.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    configs = query.all()
    return [DataSourceConfigResponse.from_orm(config) for config in configs]

@router.get("/{data_source_id}", response_model=DataSourceConfigResponse)
async def get_data_source(
    data_source_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific data source configuration."""
    config = db.query(DataSourceConfig).filter_by(id=data_source_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    return DataSourceConfigResponse.from_orm(config)

@router.put("/{data_source_id}", response_model=DataSourceConfigResponse)
async def update_data_source(
    data_source_id: str,
    update_data: DataSourceConfigUpdate,
    db: Session = Depends(get_db)
):
    """Update a data source configuration."""
    config = db.query(DataSourceConfig).filter_by(id=data_source_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    update_dict = update_data.dict(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(config, field, value)
    
    config.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(config)
    
    return DataSourceConfigResponse.from_orm(config)

@router.delete("/{data_source_id}")
async def delete_data_source(
    data_source_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Delete a data source configuration."""
    config = connection_manager.db_session.query(DataSourceConfig).filter_by(id=data_source_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    # Close any active connections
    connection_manager.close_connection(data_source_id)
    
    # Delete from database
    connection_manager.db_session.delete(config)
    connection_manager.db_session.commit()
    
    return {"message": "Data source deleted successfully"}

@router.post("/{data_source_id}/test", response_model=ConnectionTestResponse)
async def test_data_source_connection(
    data_source_id: str,
    background_tasks: BackgroundTasks,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Test connection to a data source."""
    try:
        success = await connection_manager.test_connection(data_source_id)
        
        # Get the latest test result
        test_result = connection_manager.db_session.query(ConnectionTest).filter_by(
            data_source_id=data_source_id
        ).order_by(ConnectionTest.test_timestamp.desc()).first()
        
        if not test_result:
            raise HTTPException(status_code=500, detail="Test result not found")
        
        return ConnectionTestResponse.from_orm(test_result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{data_source_id}/tests", response_model=List[ConnectionTestResponse])
async def get_connection_test_history(
    data_source_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get connection test history for a data source."""
    tests = db.query(ConnectionTest).filter_by(
        data_source_id=data_source_id
    ).order_by(ConnectionTest.test_timestamp.desc()).limit(limit).all()
    
    return [ConnectionTestResponse.from_orm(test) for test in tests]

@router.post("/{data_source_id}/discover-schema", response_model=List[DataSchemaResponse])
async def discover_data_source_schema(
    data_source_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Discover schema information from a data source."""
    try:
        schemas = await connection_manager.discover_schema(data_source_id)
        return [DataSchemaResponse.from_orm(schema) for schema in schemas]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{data_source_id}/schemas", response_model=List[DataSchemaResponse])
async def get_data_source_schemas(
    data_source_id: str,
    db: Session = Depends(get_db)
):
    """Get discovered schemas for a data source."""
    schemas = db.query(DataSchema).filter_by(
        data_source_id=data_source_id,
        is_active=True
    ).all()
    
    return [DataSchemaResponse.from_orm(schema) for schema in schemas]

@router.post("/{data_source_id}/query")
async def query_data_source(
    data_source_id: str,
    query_request: DataQueryRequest,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Query data from a data source."""
    try:
        # Get connection
        connection = await connection_manager.get_connection(data_source_id)
        
        # Get data source config to determine type
        config = connection_manager.db_session.query(DataSourceConfig).filter_by(id=data_source_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Get appropriate connector
        connector = connection_manager.connectors[config.source_type]
        
        # Add limit to query config
        query_config = query_request.query_config.copy()
        query_config["limit"] = query_request.limit
        
        # Execute query
        data = await connector.read_data(connection, query_config)
        
        return {
            "data": data,
            "count": len(data),
            "data_source_id": data_source_id,
            "query_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
async def health_check_all_sources(
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Perform health check on all active data sources."""
    try:
        results = await connection_manager.health_check()
        
        healthy_count = sum(1 for result in results.values() if result["status"] == "healthy")
        total_count = len(results)
        
        return {
            "overall_status": "healthy" if healthy_count == total_count else "degraded",
            "healthy_sources": healthy_count,
            "total_sources": total_count,
            "sources": results,
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types/supported")
async def get_supported_data_source_types():
    """Get list of supported data source types and their configurations."""
    return {
        "database": {
            "supported_types": ["postgresql", "mysql", "sql_server", "oracle"],
            "required_config": ["database_type", "host", "database"],
            "optional_config": ["port", "schema"],
            "auth_types": ["username_password", "connection_string"]
        },
        "rest_api": {
            "required_config": ["base_url"],
            "optional_config": ["health_endpoint", "timeout"],
            "auth_types": ["bearer", "api_key", "basic", "none"]
        },
        "graphql": {
            "required_config": ["endpoint"],
            "optional_config": ["timeout"],
            "auth_types": ["bearer", "api_key", "none"]
        },
        "file_system": {
            "supported_formats": ["csv", "json", "parquet", "excel"],
            "required_config": ["file_path", "file_format"],
            "optional_config": ["delimiter", "encoding", "has_header", "sheet_name"]
        },
        "streaming": {
            "supported_types": ["kafka", "kinesis", "pubsub"],
            "kafka_config": ["bootstrap_servers", "topic"],
            "kinesis_config": ["region", "stream_name"],
            "pubsub_config": ["project_id", "subscription_name"]
        }
    }

@router.post("/{data_source_id}/validate")
async def validate_data_source_config(
    data_source_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Validate data source configuration."""
    try:
        config = connection_manager.db_session.query(DataSourceConfig).filter_by(id=data_source_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        connector = connection_manager.connectors[config.source_type]
        validation_errors = connector.validate_config(config.connection_config)
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "data_source_id": data_source_id,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))