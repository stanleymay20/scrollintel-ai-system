"""
API routes for Enterprise Data Integration Layer.
Provides endpoints for managing data connectors, pipelines, and real-time streaming.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio
import json
import logging

from ...core.data_connector import DataSourceConfig, DataSourceType, data_integration_manager
from ...connectors.enterprise_data_connectors import (
    SQLServerConnector, SnowflakeConnector, DatabricksConnector,
    StreamingConfig, StreamingMode, DataValidationLevel
)
from ...connectors.data_lake_connectors import BigQueryConnector, RedshiftConnector
from ...connectors.erp_connectors import SAPConnector, OracleERPConnector, MicrosoftDynamicsConnector
from ...connectors.crm_connectors import SalesforceConnector, HubSpotConnector, MicrosoftCRMConnector
from ...core.data_pipeline import create_enterprise_pipeline, ProcessingMode
from ...core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/enterprise-data", tags=["Enterprise Data Integration"])


# Pydantic models for request/response
class DataSourceConfigModel(BaseModel):
    source_id: str = Field(..., description="Unique identifier for the data source")
    source_type: DataSourceType = Field(..., description="Type of data source")
    name: str = Field(..., description="Human-readable name for the data source")
    connection_params: Dict[str, Any] = Field(..., description="Connection parameters")
    refresh_interval: int = Field(300, description="Refresh interval in seconds")
    timeout: int = Field(30, description="Connection timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    enabled: bool = Field(True, description="Whether the data source is enabled")


class StreamingConfigModel(BaseModel):
    mode: StreamingMode = Field(StreamingMode.REAL_TIME, description="Streaming mode")
    batch_size: int = Field(1000, description="Batch size for processing")
    flush_interval: int = Field(30, description="Flush interval in seconds")
    buffer_size: int = Field(10000, description="Buffer size")
    compression: bool = Field(True, description="Enable compression")
    encryption: bool = Field(True, description="Enable encryption")


class PipelineConfigModel(BaseModel):
    processing_mode: ProcessingMode = Field(ProcessingMode.STREAMING, description="Processing mode")
    batch_size: int = Field(1000, description="Batch size")
    validation_config: Dict[str, Any] = Field(default_factory=dict, description="Validation configuration")
    cleaning_config: Dict[str, Any] = Field(default_factory=dict, description="Cleaning configuration")
    enrichment_config: Dict[str, Any] = Field(default_factory=dict, description="Enrichment configuration")
    quality_config: Dict[str, Any] = Field(default_factory=dict, description="Quality assessment configuration")


class DataQueryModel(BaseModel):
    source_id: str = Field(..., description="Data source identifier")
    query_params: Dict[str, Any] = Field(..., description="Query parameters")
    limit: Optional[int] = Field(1000, description="Maximum number of records")
    streaming: bool = Field(False, description="Enable streaming response")


class DataSourceResponse(BaseModel):
    source_id: str
    name: str
    source_type: str
    status: str
    last_sync: Optional[datetime]
    error_message: Optional[str]
    health_metrics: Dict[str, Any]


class PipelineStatsResponse(BaseModel):
    pipeline_stats: Dict[str, Any]
    processor_stats: Dict[str, Any]
    processing_mode: str
    active_processors: int


# Register all connector classes
def register_connectors():
    """Register all available connector classes"""
    # ERP Connectors
    data_integration_manager.register_connector_class('sap', SAPConnector)
    data_integration_manager.register_connector_class('oracle_erp', OracleERPConnector)
    data_integration_manager.register_connector_class('microsoft_dynamics', MicrosoftDynamicsConnector)
    
    # CRM Connectors
    data_integration_manager.register_connector_class('salesforce', SalesforceConnector)
    data_integration_manager.register_connector_class('hubspot', HubSpotConnector)
    data_integration_manager.register_connector_class('microsoft_crm', MicrosoftCRMConnector)
    
    # Enterprise Database Connectors
    data_integration_manager.register_connector_class('sql_server', SQLServerConnector)
    
    # Data Lake Connectors
    data_integration_manager.register_connector_class('snowflake', SnowflakeConnector)
    data_integration_manager.register_connector_class('databricks', DatabricksConnector)
    data_integration_manager.register_connector_class('bigquery', BigQueryConnector)
    data_integration_manager.register_connector_class('redshift', RedshiftConnector)


# Initialize connectors on startup
register_connectors()


@router.post("/data-sources", response_model=Dict[str, str])
async def create_data_source(
    config: DataSourceConfigModel,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new data source connection"""
    try:
        # Convert Pydantic model to DataSourceConfig
        data_source_config = DataSourceConfig(
            source_id=config.source_id,
            source_type=config.source_type,
            name=config.name,
            connection_params=config.connection_params,
            refresh_interval=config.refresh_interval,
            timeout=config.timeout,
            retry_attempts=config.retry_attempts,
            enabled=config.enabled
        )
        
        # Add data source
        success = await data_integration_manager.add_data_source(data_source_config)
        
        if success:
            logger.info(f"Created data source: {config.source_id}")
            return {"message": f"Data source {config.source_id} created successfully", "source_id": config.source_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to create data source")
    
    except Exception as e:
        logger.error(f"Error creating data source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-sources", response_model=List[DataSourceResponse])
async def list_data_sources(current_user: Dict = Depends(get_current_user)):
    """List all configured data sources"""
    try:
        health_status = await data_integration_manager.get_health_status()
        
        data_sources = []
        for source_id, health in health_status.items():
            connector = data_integration_manager.registry.get_connector(source_id)
            if connector:
                data_sources.append(DataSourceResponse(
                    source_id=source_id,
                    name=connector.config.name,
                    source_type=connector.config.source_type.value,
                    status=health.status.value,
                    last_sync=health.last_successful_sync,
                    error_message=health.error_message,
                    health_metrics={
                        "latency_ms": health.latency_ms,
                        "records_synced": health.records_synced
                    }
                ))
        
        return data_sources
    
    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-sources/{source_id}", response_model=DataSourceResponse)
async def get_data_source(
    source_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get details of a specific data source"""
    try:
        connector = data_integration_manager.registry.get_connector(source_id)
        if not connector:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        health = connector.get_health()
        
        return DataSourceResponse(
            source_id=source_id,
            name=connector.config.name,
            source_type=connector.config.source_type.value,
            status=health.status.value,
            last_sync=health.last_successful_sync,
            error_message=health.error_message,
            health_metrics={
                "latency_ms": health.latency_ms,
                "records_synced": health.records_synced
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data-sources/{source_id}")
async def delete_data_source(
    source_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a data source"""
    try:
        success = await data_integration_manager.remove_data_source(source_id)
        
        if success:
            return {"message": f"Data source {source_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Data source not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-sources/{source_id}/test")
async def test_data_source(
    source_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Test connection to a data source"""
    try:
        connector = data_integration_manager.registry.get_connector(source_id)
        if not connector:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        success = await connector.test_connection()
        
        return {
            "source_id": source_id,
            "connection_test": "passed" if success else "failed",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing data source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-sources/{source_id}/sync")
async def sync_data_source(
    source_id: str,
    query_params: Optional[Dict[str, Any]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Manually trigger data synchronization"""
    try:
        records = await data_integration_manager.sync_data_source(source_id, query_params)
        
        return {
            "source_id": source_id,
            "records_synced": len(records),
            "sync_timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }
    
    except Exception as e:
        logger.error(f"Error syncing data source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_data(
    query: DataQueryModel,
    current_user: Dict = Depends(get_current_user)
):
    """Query data from a specific source"""
    try:
        connector = data_integration_manager.registry.get_connector(query.source_id)
        if not connector:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Add limit to query params
        query_params = query.query_params.copy()
        if query.limit:
            query_params['limit'] = query.limit
        
        records = await connector.fetch_data(query_params)
        
        if query.streaming:
            # Return streaming response
            async def generate_records():
                for record in records:
                    yield f"data: {json.dumps(record.__dict__, default=str)}\n\n"
            
            return StreamingResponse(
                generate_records(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return all records
            return {
                "source_id": query.source_id,
                "records": [record.__dict__ for record in records],
                "count": len(records),
                "query_timestamp": datetime.utcnow().isoformat()
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying data from {query.source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-sources/{source_id}/schema")
async def get_data_source_schema(
    source_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get schema information for a data source"""
    try:
        connector = data_integration_manager.registry.get_connector(source_id)
        if not connector:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        schema = await connector.get_schema()
        
        return {
            "source_id": source_id,
            "schema": schema,
            "retrieved_at": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting schema for {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipelines/create")
async def create_data_pipeline(
    pipeline_config: PipelineConfigModel,
    schema: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Create a new data processing pipeline"""
    try:
        # Convert Pydantic model to dict
        config = {
            'processing_mode': pipeline_config.processing_mode.value,
            'batch_size': pipeline_config.batch_size,
            'validation': pipeline_config.validation_config,
            'cleaning': pipeline_config.cleaning_config,
            'enrichment': pipeline_config.enrichment_config,
            'quality': pipeline_config.quality_config
        }
        
        # Create pipeline
        pipeline = create_enterprise_pipeline(schema, config)
        
        # Store pipeline (in production, use a proper pipeline registry)
        pipeline_id = f"pipeline_{hash(str(config)) % 1000000:06d}"
        
        return {
            "pipeline_id": pipeline_id,
            "message": "Data pipeline created successfully",
            "processors": len(pipeline.processors),
            "processing_mode": pipeline_config.processing_mode.value
        }
    
    except Exception as e:
        logger.error(f"Error creating data pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines/{pipeline_id}/stats", response_model=PipelineStatsResponse)
async def get_pipeline_stats(
    pipeline_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get statistics for a data pipeline"""
    try:
        # In production, retrieve pipeline from registry
        # For now, return mock stats
        mock_stats = {
            "pipeline_stats": {
                "records_processed": 10000,
                "records_passed": 9500,
                "records_failed": 500,
                "processing_time_ms": 45000,
                "throughput_per_second": 222.2,
                "error_rate": 0.05
            },
            "processor_stats": {
                "schema_validator": {
                    "records_processed": 10000,
                    "records_passed": 9800,
                    "records_failed": 200
                },
                "data_cleaner": {
                    "records_processed": 9800,
                    "records_passed": 9700,
                    "records_failed": 100
                },
                "data_enricher": {
                    "records_processed": 9700,
                    "records_passed": 9600,
                    "records_enriched": 9600
                },
                "quality_assessment": {
                    "records_processed": 9600,
                    "records_passed": 9500,
                    "records_failed": 100
                }
            },
            "processing_mode": "streaming",
            "active_processors": 4
        }
        
        return PipelineStatsResponse(**mock_stats)
    
    except Exception as e:
        logger.error(f"Error getting pipeline stats for {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_system_health(current_user: Dict = Depends(get_current_user)):
    """Get overall system health status"""
    try:
        health_status = await data_integration_manager.get_health_status()
        connection_tests = await data_integration_manager.test_all_connections()
        
        total_sources = len(health_status)
        healthy_sources = sum(1 for health in health_status.values() if health.status.value == 'connected')
        failed_connections = sum(1 for result in connection_tests.values() if not result)
        
        return {
            "system_status": "healthy" if failed_connections == 0 else "degraded",
            "total_data_sources": total_sources,
            "healthy_sources": healthy_sources,
            "failed_connections": failed_connections,
            "uptime_percentage": (healthy_sources / total_sources * 100) if total_sources > 0 else 100,
            "last_check": datetime.utcnow().isoformat(),
            "data_sources": [
                {
                    "source_id": source_id,
                    "status": health.status.value,
                    "last_sync": health.last_successful_sync.isoformat() if health.last_successful_sync else None,
                    "error": health.error_message
                }
                for source_id, health in health_status.items()
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_system_metrics(
    time_range: str = Query("1h", description="Time range for metrics (1h, 24h, 7d)"),
    current_user: Dict = Depends(get_current_user)
):
    """Get system performance metrics"""
    try:
        # Parse time range
        time_ranges = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        if time_range not in time_ranges:
            raise HTTPException(status_code=400, detail="Invalid time range")
        
        # Mock metrics (in production, retrieve from monitoring system)
        end_time = datetime.utcnow()
        start_time = end_time - time_ranges[time_range]
        
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration": time_range
            },
            "metrics": {
                "total_records_processed": 1000000,
                "average_throughput_per_second": 278.5,
                "peak_throughput_per_second": 450.2,
                "average_latency_ms": 125.3,
                "error_rate_percentage": 2.1,
                "data_quality_score": 0.94,
                "storage_usage_gb": 1250.7,
                "network_throughput_mbps": 45.2
            },
            "by_source": {
                "salesforce": {
                    "records_processed": 250000,
                    "error_rate": 0.015,
                    "avg_latency_ms": 95.2
                },
                "sap": {
                    "records_processed": 180000,
                    "error_rate": 0.032,
                    "avg_latency_ms": 156.8
                },
                "snowflake": {
                    "records_processed": 320000,
                    "error_rate": 0.008,
                    "avg_latency_ms": 78.4
                }
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streaming/start")
async def start_streaming_session(
    source_id: str,
    streaming_config: StreamingConfigModel,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Start a real-time streaming session"""
    try:
        connector = data_integration_manager.registry.get_connector(source_id)
        if not connector:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Create streaming session
        session_id = f"stream_{source_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Start background streaming task
        async def streaming_task():
            try:
                logger.info(f"Starting streaming session {session_id} for {source_id}")
                
                # Simulate streaming (in production, implement actual streaming)
                for i in range(100):  # Stream 100 records
                    await asyncio.sleep(streaming_config.flush_interval / 100)
                    
                    # Fetch and stream data
                    records = await connector.fetch_data({'limit': streaming_config.batch_size})
                    
                    # Process through pipeline if configured
                    # ... pipeline processing ...
                    
                    logger.debug(f"Streamed batch {i+1} with {len(records)} records")
                
                logger.info(f"Streaming session {session_id} completed")
                
            except Exception as e:
                logger.error(f"Streaming session {session_id} failed: {e}")
        
        background_tasks.add_task(streaming_task)
        
        return {
            "session_id": session_id,
            "source_id": source_id,
            "status": "started",
            "streaming_config": streaming_config.dict(),
            "started_at": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting streaming session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connectors/available")
async def get_available_connectors(current_user: Dict = Depends(get_current_user)):
    """Get list of available connector types"""
    return {
        "erp_connectors": [
            {
                "name": "sap",
                "display_name": "SAP ERP",
                "description": "Connect to SAP ERP systems via RFC",
                "required_params": ["host", "client", "username", "password"]
            },
            {
                "name": "oracle_erp",
                "display_name": "Oracle ERP Cloud",
                "description": "Connect to Oracle ERP Cloud via REST API",
                "required_params": ["base_url", "username", "password"]
            },
            {
                "name": "microsoft_dynamics",
                "display_name": "Microsoft Dynamics 365",
                "description": "Connect to Dynamics 365 via Web API",
                "required_params": ["org_url", "client_id", "client_secret", "tenant_id"]
            }
        ],
        "crm_connectors": [
            {
                "name": "salesforce",
                "display_name": "Salesforce CRM",
                "description": "Connect to Salesforce via REST API and SOQL",
                "required_params": ["instance_url", "client_id", "client_secret", "username", "password"]
            },
            {
                "name": "hubspot",
                "display_name": "HubSpot CRM",
                "description": "Connect to HubSpot via REST API",
                "required_params": ["api_key"]
            },
            {
                "name": "microsoft_crm",
                "display_name": "Microsoft Dynamics CRM",
                "description": "Connect to Dynamics CRM via Web API",
                "required_params": ["org_url", "client_id", "client_secret", "tenant_id"]
            }
        ],
        "database_connectors": [
            {
                "name": "sql_server",
                "display_name": "Microsoft SQL Server",
                "description": "Connect to SQL Server databases",
                "required_params": ["server", "database", "username", "password"]
            }
        ],
        "data_lake_connectors": [
            {
                "name": "snowflake",
                "display_name": "Snowflake",
                "description": "Connect to Snowflake data warehouse",
                "required_params": ["account", "user", "warehouse", "database"]
            },
            {
                "name": "databricks",
                "display_name": "Databricks",
                "description": "Connect to Databricks lakehouse platform",
                "required_params": ["server_hostname", "http_path", "access_token"]
            },
            {
                "name": "bigquery",
                "display_name": "Google BigQuery",
                "description": "Connect to Google BigQuery data warehouse",
                "required_params": ["project_id", "dataset_id"]
            },
            {
                "name": "redshift",
                "display_name": "Amazon Redshift",
                "description": "Connect to Amazon Redshift data warehouse",
                "required_params": ["host", "database", "user"]
            }
        ]
    }