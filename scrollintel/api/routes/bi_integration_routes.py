"""
BI Integration API Routes
REST API endpoints for BI tool integration functionality
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from scrollintel.core.database import get_db
from scrollintel.models.bi_integration_models import (
    BIConnectionConfig, BIConnectionResponse, DashboardExportRequest,
    EmbedTokenRequest, EmbedTokenResponse, DataSyncRequest, DataSyncResult,
    BIDashboardInfo, BIIntegrationStatus, BIToolType
)
from scrollintel.engines.bi_integration_engine import bi_integration_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bi-integration", tags=["BI Integration"])


@router.post("/connections", response_model=BIConnectionResponse)
async def create_bi_connection(
    config: BIConnectionConfig,
    db: Session = Depends(get_db)
):
    """
    Create a new BI tool connection
    
    Supports:
    - Tableau Server/Online
    - Microsoft Power BI
    - Looker
    """
    try:
        return await bi_integration_engine.create_connection(config, db)
    except Exception as e:
        logger.error(f"Error creating BI connection: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/connections/{connection_id}/test")
async def test_bi_connection(
    connection_id: str,
    db: Session = Depends(get_db)
):
    """Test a BI tool connection"""
    try:
        result = await bi_integration_engine.test_connection(connection_id, db)
        return result
    except Exception as e:
        logger.error(f"Error testing BI connection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections/{connection_id}/dashboards", response_model=List[BIDashboardInfo])
async def get_dashboards(
    connection_id: str,
    project_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get list of dashboards from a BI tool"""
    try:
        return await bi_integration_engine.get_dashboards(connection_id, project_id, db)
    except Exception as e:
        logger.error(f"Error getting dashboards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections/{connection_id}/dashboards/{dashboard_id}", response_model=BIDashboardInfo)
async def get_dashboard_info(
    connection_id: str,
    dashboard_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific dashboard"""
    try:
        connector = await bi_integration_engine.get_connection(connection_id, db)
        if not connector:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        return await connector.get_dashboard_info(dashboard_id)
    except Exception as e:
        logger.error(f"Error getting dashboard info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/export")
async def export_dashboard(
    connection_id: str,
    request: DashboardExportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Export a dashboard in specified format
    
    Supported formats:
    - PDF
    - PNG
    - JPEG
    - CSV (where supported)
    
    Returns a job ID for tracking export progress
    """
    try:
        job_id = await bi_integration_engine.export_dashboard(connection_id, request, db)
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Export job started successfully"
        }
    except Exception as e:
        logger.error(f"Error starting dashboard export: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export-jobs/{job_id}")
async def get_export_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Get the status of an export job"""
    try:
        from scrollintel.models.bi_integration_models import BIExportJob
        
        job = db.query(BIExportJob).filter(BIExportJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Export job not found")
        
        return {
            "job_id": job.id,
            "status": job.status,
            "dashboard_id": job.dashboard_id,
            "export_format": job.export_format,
            "file_path": job.file_path,
            "error_message": job.error_message,
            "created_at": job.created_at,
            "completed_at": job.completed_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting export job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/embed-token", response_model=EmbedTokenResponse)
async def create_embed_token(
    connection_id: str,
    request: EmbedTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Create an embed token for dashboard embedding
    
    Supports:
    - iframe embedding
    - JavaScript SDK embedding
    - White-label embedding
    """
    try:
        return await bi_integration_engine.create_embed_token(connection_id, request, db)
    except Exception as e:
        logger.error(f"Error creating embed token: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/sync", response_model=DataSyncResult)
async def sync_data_source(
    connection_id: str,
    request: DataSyncRequest,
    db: Session = Depends(get_db)
):
    """
    Synchronize a data source
    
    Triggers data refresh in the BI tool
    """
    try:
        return await bi_integration_engine.sync_data_source(connection_id, request, db)
    except Exception as e:
        logger.error(f"Error syncing data source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections/{connection_id}/status", response_model=BIIntegrationStatus)
async def get_integration_status(
    connection_id: str,
    db: Session = Depends(get_db)
):
    """Get integration status for a connection"""
    try:
        return await bi_integration_engine.get_integration_status(connection_id, db)
    except Exception as e:
        logger.error(f"Error getting integration status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-tools")
async def get_supported_tools():
    """Get list of supported BI tools and their capabilities"""
    try:
        return await bi_integration_engine.get_supported_tools()
    except Exception as e:
        logger.error(f"Error getting supported tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/connections/{connection_id}")
async def delete_bi_connection(
    connection_id: str,
    db: Session = Depends(get_db)
):
    """Delete a BI tool connection"""
    try:
        success = await bi_integration_engine.delete_connection(connection_id, db)
        if success:
            return {"message": "Connection deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Connection not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting BI connection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/data-sources")
async def create_data_source(
    connection_id: str,
    name: str,
    connection_string: str,
    source_type: str,
    config: Optional[dict] = None,
    db: Session = Depends(get_db)
):
    """Create a new data source in the BI tool"""
    try:
        connector = await bi_integration_engine.get_connection(connection_id, db)
        if not connector:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        data_source_id = await connector.create_data_source(name, connection_string, source_type, config)
        
        # Save to database
        from scrollintel.models.bi_integration_models import BIDataSource
        import uuid
        
        db_data_source = BIDataSource(
            id=str(uuid.uuid4()),
            connection_id=connection_id,
            name=name,
            source_type=source_type,
            connection_string=connection_string
        )
        db.add(db_data_source)
        db.commit()
        
        return {
            "data_source_id": data_source_id,
            "message": "Data source created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating data source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections/{connection_id}/real-time-feed/{data_source_id}")
async def get_real_time_data_feed(
    connection_id: str,
    data_source_id: str,
    callback_url: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Set up real-time data feed for a data source"""
    try:
        connector = await bi_integration_engine.get_connection(connection_id, db)
        if not connector:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        return await connector.get_real_time_data_feed(data_source_id, callback_url)
    except Exception as e:
        logger.error(f"Error setting up real-time feed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections/{connection_id}/capabilities")
async def get_connection_capabilities(
    connection_id: str,
    db: Session = Depends(get_db)
):
    """Get capabilities of a specific BI connection"""
    try:
        connector = await bi_integration_engine.get_connection(connection_id, db)
        if not connector:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        return {
            "tool_type": connector.tool_type.value,
            "supported_export_formats": [fmt.value for fmt in connector.get_supported_export_formats()],
            "supported_embed_types": [embed.value for embed in connector.get_supported_embed_types()],
            "required_config_fields": connector.get_required_config_fields()
        }
    except Exception as e:
        logger.error(f"Error getting connection capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# White-label embedding endpoints
@router.get("/embed/{connection_id}/{dashboard_id}")
async def get_white_label_embed(
    connection_id: str,
    dashboard_id: str,
    user_id: str,
    theme: Optional[str] = "default",
    db: Session = Depends(get_db)
):
    """
    Get white-label embed configuration
    
    Returns customized embed code with branding removed
    """
    try:
        from scrollintel.models.bi_integration_models import EmbedTokenRequest, EmbedType
        
        request = EmbedTokenRequest(
            dashboard_id=dashboard_id,
            user_id=user_id,
            embed_type=EmbedType.WHITE_LABEL,
            expiry_minutes=120
        )
        
        embed_response = await bi_integration_engine.create_embed_token(connection_id, request, db)
        
        # Customize for white-label
        white_label_config = {
            "embed_url": embed_response.embed_url,
            "token": embed_response.token,
            "expires_at": embed_response.expires_at,
            "theme": theme,
            "branding": {
                "hide_toolbar": True,
                "hide_tabs": True,
                "custom_css": f"""
                    .bi-toolbar {{ display: none !important; }}
                    .bi-branding {{ display: none !important; }}
                    .bi-theme-{theme} {{ 
                        --primary-color: #your-brand-color;
                        --background-color: #your-bg-color;
                    }}
                """
            }
        }
        
        return white_label_config
    except Exception as e:
        logger.error(f"Error creating white-label embed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))