"""
API Integration Routes
Provides REST endpoints for managing API connections, testing, and monitoring
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.database_connection_manager import get_async_session

# Simple dependency for database session
async def get_db():
    """Get database session for API routes"""
    async with get_async_session() as session:
        yield session
from ...models.api_integration_models import (
    APIConnection, APIEndpoint, WebhookConfig, APIRequestLog, 
    APISchema, APIDataSync, APIMetrics
)
from ...connectors.api_connectors import (
    APIConnectorFactory, APIType, AuthType, AuthConfig, 
    RateLimitConfig, APIEndpoint as ConnectorEndpoint,
    APISchemaDiscovery, WebhookManager
)

router = APIRouter(prefix="/api/integration", tags=["API Integration"])

# Pydantic models for request/response
class APIConnectionCreate(BaseModel):
    name: str = Field(..., description="Connection name")
    description: Optional[str] = None
    api_type: str = Field(..., description="API type: rest, graphql, soap")
    base_url: str = Field(..., description="Base URL of the API")
    auth_type: str = Field(default="none", description="Authentication type")
    auth_config: Dict[str, Any] = Field(default_factory=dict)
    rate_limit_config: Optional[Dict[str, Any]] = None


class APIConnectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    base_url: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    rate_limit_config: Optional[Dict[str, Any]] = None


class APIEndpointCreate(BaseModel):
    name: str
    description: Optional[str] = None
    endpoint_path: str
    http_method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None
    request_body_template: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3


class APITestRequest(BaseModel):
    endpoint_path: Optional[str] = None
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None


class WebhookConfigCreate(BaseModel):
    name: str
    description: Optional[str] = None
    webhook_url: str
    secret_token: Optional[str] = None
    event_types: List[str] = Field(default_factory=list)
    handler_function: Optional[str] = None


class DataSyncCreate(BaseModel):
    sync_name: str
    sync_type: str = "full"  # full, incremental, real_time
    schedule_cron: Optional[str] = None
    sync_query: Optional[str] = None
    target_table: Optional[str] = None
    data_transformation: Optional[Dict[str, Any]] = None


# Connection Management Endpoints

@router.post("/connections", response_model=Dict[str, Any])
async def create_api_connection(
    connection_data: APIConnectionCreate,
    db: Session = Depends(get_db)
):
    """Create a new API connection"""
    try:
        # Validate API type
        api_type = APIType(connection_data.api_type.lower())
        auth_type = AuthType(connection_data.auth_type.lower())
        
        # Create database record
        db_connection = APIConnection(
            name=connection_data.name,
            description=connection_data.description,
            api_type=connection_data.api_type.lower(),
            base_url=connection_data.base_url,
            auth_type=connection_data.auth_type.lower(),
            auth_config=connection_data.auth_config,
            rate_limit_config=connection_data.rate_limit_config or {},
            status="inactive"
        )
        
        db.add(db_connection)
        db.commit()
        db.refresh(db_connection)
        
        return {
            "id": str(db_connection.id),
            "name": db_connection.name,
            "api_type": db_connection.api_type,
            "status": db_connection.status,
            "created_at": db_connection.created_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid API or auth type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create connection: {e}")


@router.get("/connections", response_model=List[Dict[str, Any]])
async def list_api_connections(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    api_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List API connections with optional filtering"""
    query = db.query(APIConnection)
    
    if api_type:
        query = query.filter(APIConnection.api_type == api_type.lower())
    if status:
        query = query.filter(APIConnection.status == status.lower())
    
    connections = query.offset(skip).limit(limit).all()
    
    return [
        {
            "id": str(conn.id),
            "name": conn.name,
            "description": conn.description,
            "api_type": conn.api_type,
            "base_url": conn.base_url,
            "status": conn.status,
            "last_tested": conn.last_tested.isoformat() if conn.last_tested else None,
            "created_at": conn.created_at.isoformat(),
            "endpoint_count": len(conn.endpoints)
        }
        for conn in connections
    ]


@router.get("/connections/{connection_id}", response_model=Dict[str, Any])
async def get_api_connection(connection_id: str, db: Session = Depends(get_db)):
    """Get detailed information about an API connection"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    return {
        "id": str(connection.id),
        "name": connection.name,
        "description": connection.description,
        "api_type": connection.api_type,
        "base_url": connection.base_url,
        "auth_type": connection.auth_type,
        "status": connection.status,
        "last_tested": connection.last_tested.isoformat() if connection.last_tested else None,
        "last_success": connection.last_success.isoformat() if connection.last_success else None,
        "error_message": connection.error_message,
        "created_at": connection.created_at.isoformat(),
        "endpoints": [
            {
                "id": str(ep.id),
                "name": ep.name,
                "endpoint_path": ep.endpoint_path,
                "http_method": ep.http_method,
                "is_active": ep.is_active,
                "success_count": ep.success_count,
                "error_count": ep.error_count
            }
            for ep in connection.endpoints
        ],
        "webhooks": [
            {
                "id": str(wh.id),
                "name": wh.name,
                "webhook_url": wh.webhook_url,
                "is_active": wh.is_active,
                "total_received": wh.total_received
            }
            for wh in connection.webhooks
        ]
    }


@router.put("/connections/{connection_id}", response_model=Dict[str, Any])
async def update_api_connection(
    connection_id: str,
    update_data: APIConnectionUpdate,
    db: Session = Depends(get_db)
):
    """Update an API connection"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Update fields
    update_dict = update_data.dict(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(connection, field, value)
    
    connection.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(connection)
    
    return {
        "id": str(connection.id),
        "name": connection.name,
        "status": connection.status,
        "updated_at": connection.updated_at.isoformat()
    }


@router.delete("/connections/{connection_id}")
async def delete_api_connection(connection_id: str, db: Session = Depends(get_db)):
    """Delete an API connection and all related data"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    db.delete(connection)
    db.commit()
    
    return {"message": "Connection deleted successfully"}


# Connection Testing Endpoints

@router.post("/connections/{connection_id}/test", response_model=Dict[str, Any])
async def test_api_connection(
    connection_id: str,
    test_request: Optional[APITestRequest] = None,
    db: Session = Depends(get_db)
):
    """Test an API connection"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    try:
        # Update connection status
        connection.status = "testing"
        connection.last_tested = datetime.utcnow()
        db.commit()
        
        # Create connector
        api_type = APIType(connection.api_type)
        auth_config = AuthConfig(
            auth_type=AuthType(connection.auth_type),
            credentials=connection.auth_config
        )
        rate_config = RateLimitConfig(**connection.rate_limit_config) if connection.rate_limit_config else None
        
        connector = APIConnectorFactory.create_connector(
            api_type, connection.base_url, auth_config, rate_config
        )
        
        # Perform test request
        async with connector:
            if test_request:
                endpoint = ConnectorEndpoint(
                    url=test_request.endpoint_path or "/",
                    method=test_request.method,
                    headers=test_request.headers or {},
                    params=test_request.parameters or {}
                )
                result = await connector.make_request(endpoint, test_request.body)
            else:
                # Default health check
                endpoint = ConnectorEndpoint(url="/", method="GET")
                result = await connector.make_request(endpoint)
        
        # Update connection status
        connection.status = "active"
        connection.last_success = datetime.utcnow()
        connection.error_message = None
        db.commit()
        
        return {
            "success": True,
            "status": "active",
            "response": result,
            "tested_at": connection.last_tested.isoformat()
        }
        
    except Exception as e:
        # Update connection with error
        connection.status = "error"
        connection.error_message = str(e)
        db.commit()
        
        return {
            "success": False,
            "status": "error",
            "error": str(e),
            "tested_at": connection.last_tested.isoformat()
        }


# Endpoint Management

@router.post("/connections/{connection_id}/endpoints", response_model=Dict[str, Any])
async def create_api_endpoint(
    connection_id: str,
    endpoint_data: APIEndpointCreate,
    db: Session = Depends(get_db)
):
    """Create a new API endpoint for a connection"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    db_endpoint = APIEndpoint(
        connection_id=connection_id,
        name=endpoint_data.name,
        description=endpoint_data.description,
        endpoint_path=endpoint_data.endpoint_path,
        http_method=endpoint_data.http_method.upper(),
        headers=endpoint_data.headers or {},
        parameters=endpoint_data.parameters or {},
        request_body_template=endpoint_data.request_body_template,
        timeout=endpoint_data.timeout,
        retry_count=endpoint_data.retry_count
    )
    
    db.add(db_endpoint)
    db.commit()
    db.refresh(db_endpoint)
    
    return {
        "id": str(db_endpoint.id),
        "name": db_endpoint.name,
        "endpoint_path": db_endpoint.endpoint_path,
        "http_method": db_endpoint.http_method,
        "created_at": db_endpoint.created_at.isoformat()
    }


@router.post("/connections/{connection_id}/endpoints/{endpoint_id}/call", response_model=Dict[str, Any])
async def call_api_endpoint(
    connection_id: str,
    endpoint_id: str,
    request_data: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Call a specific API endpoint"""
    endpoint = db.query(APIEndpoint).filter(
        APIEndpoint.id == endpoint_id,
        APIEndpoint.connection_id == connection_id
    ).first()
    
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    connection = endpoint.connection
    
    try:
        # Create connector
        api_type = APIType(connection.api_type)
        auth_config = AuthConfig(
            auth_type=AuthType(connection.auth_type),
            credentials=connection.auth_config
        )
        rate_config = RateLimitConfig(**connection.rate_limit_config) if connection.rate_limit_config else None
        
        connector = APIConnectorFactory.create_connector(
            api_type, connection.base_url, auth_config, rate_config
        )
        
        # Make request
        start_time = datetime.utcnow()
        async with connector:
            connector_endpoint = ConnectorEndpoint(
                url=endpoint.endpoint_path,
                method=endpoint.http_method,
                headers=endpoint.headers,
                params=endpoint.parameters,
                timeout=endpoint.timeout
            )
            result = await connector.make_request(connector_endpoint, request_data)
        
        end_time = datetime.utcnow()
        response_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Log request
        log_entry = APIRequestLog(
            connection_id=connection_id,
            endpoint_id=endpoint_id,
            request_method=endpoint.http_method,
            request_url=f"{connection.base_url}{endpoint.endpoint_path}",
            request_headers=endpoint.headers,
            request_body=json.dumps(request_data) if request_data else None,
            response_status=200,  # Assume success if no exception
            response_body=json.dumps(result),
            response_time_ms=response_time,
            success=True,
            timestamp=start_time
        )
        db.add(log_entry)
        
        # Update endpoint statistics
        endpoint.last_called = start_time
        endpoint.success_count += 1
        
        db.commit()
        
        return {
            "success": True,
            "response": result,
            "response_time_ms": response_time,
            "called_at": start_time.isoformat()
        }
        
    except Exception as e:
        # Log error
        log_entry = APIRequestLog(
            connection_id=connection_id,
            endpoint_id=endpoint_id,
            request_method=endpoint.http_method,
            request_url=f"{connection.base_url}{endpoint.endpoint_path}",
            request_headers=endpoint.headers,
            request_body=json.dumps(request_data) if request_data else None,
            response_status=0,
            success=False,
            error_message=str(e),
            timestamp=datetime.utcnow()
        )
        db.add(log_entry)
        
        # Update endpoint statistics
        endpoint.error_count += 1
        
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"API call failed: {e}")


# Schema Discovery

@router.post("/connections/{connection_id}/discover-schema", response_model=Dict[str, Any])
async def discover_api_schema(
    connection_id: str,
    openapi_url: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Discover API schema for documentation and validation"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    try:
        # Create connector
        api_type = APIType(connection.api_type)
        auth_config = AuthConfig(
            auth_type=AuthType(connection.auth_type),
            credentials=connection.auth_config
        )
        
        connector = APIConnectorFactory.create_connector(
            api_type, connection.base_url, auth_config
        )
        
        # Discover schema
        async with connector:
            discovery = APISchemaDiscovery(connector)
            
            if api_type == APIType.REST:
                schema_data = await discovery.discover_rest_schema(openapi_url)
            elif api_type == APIType.GRAPHQL:
                schema_data = await discovery.discover_graphql_schema()
            else:
                schema_data = {"error": "Schema discovery not supported for SOAP APIs"}
        
        # Save schema to database
        if "error" not in schema_data:
            db_schema = APISchema(
                connection_id=connection_id,
                schema_type=api_type.value,
                schema_data=schema_data,
                endpoints_discovered=schema_data.get("endpoints", []),
                discovery_method="introspection" if api_type == APIType.GRAPHQL else "openapi",
                schema_url=openapi_url
            )
            db.add(db_schema)
            db.commit()
        
        return {
            "success": "error" not in schema_data,
            "schema": schema_data,
            "discovered_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "discovered_at": datetime.utcnow().isoformat()
        }


# Webhook Management

@router.post("/connections/{connection_id}/webhooks", response_model=Dict[str, Any])
async def create_webhook_config(
    connection_id: str,
    webhook_data: WebhookConfigCreate,
    db: Session = Depends(get_db)
):
    """Create webhook configuration for real-time updates"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    db_webhook = WebhookConfig(
        connection_id=connection_id,
        name=webhook_data.name,
        description=webhook_data.description,
        webhook_url=webhook_data.webhook_url,
        secret_token=webhook_data.secret_token,
        event_types=webhook_data.event_types,
        handler_function=webhook_data.handler_function
    )
    
    db.add(db_webhook)
    db.commit()
    db.refresh(db_webhook)
    
    return {
        "id": str(db_webhook.id),
        "name": db_webhook.name,
        "webhook_url": db_webhook.webhook_url,
        "is_active": db_webhook.is_active,
        "created_at": db_webhook.created_at.isoformat()
    }


# Monitoring and Metrics

@router.get("/connections/{connection_id}/metrics", response_model=Dict[str, Any])
async def get_connection_metrics(
    connection_id: str,
    hours: int = Query(24, ge=1, le=168),  # Last 1-168 hours
    db: Session = Depends(get_db)
):
    """Get performance metrics for an API connection"""
    connection = db.query(APIConnection).filter(APIConnection.id == connection_id).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Get metrics from the last N hours
    since = datetime.utcnow() - timedelta(hours=hours)
    
    metrics = db.query(APIMetrics).filter(
        APIMetrics.connection_id == connection_id,
        APIMetrics.metric_date >= since
    ).all()
    
    # Get recent request logs for detailed analysis
    recent_logs = db.query(APIRequestLog).filter(
        APIRequestLog.connection_id == connection_id,
        APIRequestLog.timestamp >= since
    ).all()
    
    # Calculate summary statistics
    total_requests = sum(m.total_requests for m in metrics)
    successful_requests = sum(m.successful_requests for m in metrics)
    failed_requests = sum(m.failed_requests for m in metrics)
    
    avg_response_times = [m.avg_response_time_ms for m in metrics if m.avg_response_time_ms]
    avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
    
    return {
        "connection_id": connection_id,
        "period_hours": hours,
        "summary": {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time_ms": round(avg_response_time, 2)
        },
        "hourly_metrics": [
            {
                "hour": m.metric_date.isoformat(),
                "requests": m.total_requests,
                "success_rate": (m.successful_requests / m.total_requests * 100) if m.total_requests > 0 else 0,
                "avg_response_time_ms": m.avg_response_time_ms,
                "rate_limited": m.rate_limited_requests
            }
            for m in metrics
        ],
        "recent_errors": [
            {
                "timestamp": log.timestamp.isoformat(),
                "endpoint": log.request_url,
                "error": log.error_message,
                "status": log.response_status
            }
            for log in recent_logs
            if not log.success
        ][-10:]  # Last 10 errors
    }


@router.get("/connections/{connection_id}/logs", response_model=List[Dict[str, Any]])
async def get_connection_logs(
    connection_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    success_only: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    """Get request logs for an API connection"""
    query = db.query(APIRequestLog).filter(APIRequestLog.connection_id == connection_id)
    
    if success_only is not None:
        query = query.filter(APIRequestLog.success == success_only)
    
    logs = query.order_by(APIRequestLog.timestamp.desc()).offset(skip).limit(limit).all()
    
    return [
        {
            "id": str(log.id),
            "timestamp": log.timestamp.isoformat(),
            "method": log.request_method,
            "url": log.request_url,
            "status": log.response_status,
            "response_time_ms": log.response_time_ms,
            "success": log.success,
            "error": log.error_message,
            "retry_attempt": log.retry_attempt
        }
        for log in logs
    ]