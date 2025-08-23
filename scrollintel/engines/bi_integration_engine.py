"""
BI Integration Engine
Main engine that coordinates all BI tool integrations and provides unified interface
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging

from sqlalchemy.orm import Session
from scrollintel.core.database import get_db
from scrollintel.models.bi_integration_models import (
    BIConnection, BIDashboard, BIDataSource, BIExportJob,
    BIToolType, ExportFormat, EmbedType, ConnectionStatus,
    BIConnectionConfig, BIConnectionResponse, DashboardExportRequest,
    EmbedTokenRequest, EmbedTokenResponse, DataSyncRequest, DataSyncResult,
    BIDashboardInfo, BIIntegrationStatus
)
from scrollintel.connectors.bi_connector_base import (
    bi_connector_registry, BaseBIConnector, BIConnectorError
)

# Import all connectors to register them
from scrollintel.connectors.tableau_connector import TableauConnector
from scrollintel.connectors.power_bi_connector import PowerBIConnector
from scrollintel.connectors.looker_connector import LookerConnector

logger = logging.getLogger(__name__)


class BIIntegrationEngine:
    """
    Main BI Integration Engine
    Provides unified interface for all BI tool integrations
    """
    
    def __init__(self):
        self.active_connections: Dict[str, BaseBIConnector] = {}
    
    async def create_connection(
        self, 
        config: BIConnectionConfig,
        db: Session
    ) -> BIConnectionResponse:
        """Create a new BI tool connection"""
        try:
            # Validate configuration
            if not bi_connector_registry.is_supported(config.bi_tool_type):
                raise ValueError(f"Unsupported BI tool type: {config.bi_tool_type}")
            
            # Create connection record
            connection_id = str(uuid.uuid4())
            
            # Prepare connection configuration
            connection_config = {
                'id': connection_id,
                'name': config.name,
                'bi_tool_type': config.bi_tool_type,
                'server_url': config.server_url,
                'username': config.username,
                'password': config.password,
                'api_key': config.api_key,
                'site_id': config.site_id,
                'project_id': config.project_id,
                **config.additional_config
            }
            
            # Test connection
            connector = bi_connector_registry.get_connector(config.bi_tool_type, connection_config)
            
            # Validate configuration
            validation_errors = await connector.validate_config()
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")
            
            # Test authentication
            await connector.authenticate()
            connection_test = await connector.test_connection()
            
            if connection_test.get('status') != 'connected':
                raise ConnectionError(f"Connection test failed: {connection_test.get('error', 'Unknown error')}")
            
            # Save to database
            db_connection = BIConnection(
                id=connection_id,
                name=config.name,
                bi_tool_type=config.bi_tool_type.value,
                connection_config=connection_config,
                credentials={
                    'username': config.username,
                    'password': config.password,  # Should be encrypted in production
                    'api_key': config.api_key
                },
                status=ConnectionStatus.ACTIVE,
                last_sync=datetime.utcnow()
            )
            
            db.add(db_connection)
            db.commit()
            
            # Store active connector
            self.active_connections[connection_id] = connector
            
            logger.info(f"Created BI connection: {config.name} ({config.bi_tool_type})")
            
            return BIConnectionResponse(
                id=connection_id,
                name=config.name,
                bi_tool_type=config.bi_tool_type,
                status=ConnectionStatus.ACTIVE,
                created_at=datetime.utcnow(),
                last_sync=datetime.utcnow()
            )
        
        except Exception as e:
            logger.error(f"Error creating BI connection: {str(e)}")
            raise BIConnectorError(f"Failed to create connection: {str(e)}")
    
    async def get_connection(self, connection_id: str, db: Session) -> Optional[BaseBIConnector]:
        """Get an active BI connector"""
        # Check if already loaded
        if connection_id in self.active_connections:
            return self.active_connections[connection_id]
        
        # Load from database
        db_connection = db.query(BIConnection).filter(BIConnection.id == connection_id).first()
        if not db_connection:
            return None
        
        # Create connector
        try:
            bi_tool_type = BIToolType(db_connection.bi_tool_type)
            connector = bi_connector_registry.get_connector(bi_tool_type, db_connection.connection_config)
            
            # Cache the connector
            self.active_connections[connection_id] = connector
            return connector
        
        except Exception as e:
            logger.error(f"Error loading BI connection {connection_id}: {str(e)}")
            return None
    
    async def test_connection(self, connection_id: str, db: Session) -> Dict[str, Any]:
        """Test a BI connection"""
        try:
            connector = await self.get_connection(connection_id, db)
            if not connector:
                return {"status": "error", "error": "Connection not found"}
            
            return await connector.test_connection()
        
        except Exception as e:
            logger.error(f"Error testing connection {connection_id}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def get_dashboards(
        self, 
        connection_id: str, 
        project_id: Optional[str] = None,
        db: Session = None
    ) -> List[BIDashboardInfo]:
        """Get dashboards from a BI tool"""
        try:
            connector = await self.get_connection(connection_id, db)
            if not connector:
                raise BIConnectorError("Connection not found")
            
            dashboards = await connector.get_dashboards(project_id)
            
            # Update database with dashboard info
            if db:
                for dashboard in dashboards:
                    existing = db.query(BIDashboard).filter(
                        BIDashboard.connection_id == connection_id,
                        BIDashboard.external_id == dashboard.id
                    ).first()
                    
                    if not existing:
                        db_dashboard = BIDashboard(
                            id=str(uuid.uuid4()),
                            connection_id=connection_id,
                            external_id=dashboard.id,
                            name=dashboard.name,
                            description=dashboard.description,
                            url=dashboard.url,
                            is_public=dashboard.is_public
                        )
                        db.add(db_dashboard)
                
                db.commit()
            
            return dashboards
        
        except Exception as e:
            logger.error(f"Error getting dashboards from {connection_id}: {str(e)}")
            raise BIConnectorError(f"Failed to get dashboards: {str(e)}")
    
    async def export_dashboard(
        self, 
        connection_id: str,
        request: DashboardExportRequest,
        db: Session
    ) -> str:
        """Export a dashboard and return job ID"""
        try:
            connector = await self.get_connection(connection_id, db)
            if not connector:
                raise BIConnectorError("Connection not found")
            
            # Create export job
            job_id = str(uuid.uuid4())
            export_job = BIExportJob(
                id=job_id,
                dashboard_id=request.dashboard_id,
                export_format=request.format.value,
                status="pending"
            )
            db.add(export_job)
            db.commit()
            
            # Start export in background
            asyncio.create_task(self._export_dashboard_async(
                connector, request, job_id, db
            ))
            
            return job_id
        
        except Exception as e:
            logger.error(f"Error starting dashboard export: {str(e)}")
            raise BIConnectorError(f"Failed to start export: {str(e)}")
    
    async def _export_dashboard_async(
        self,
        connector: BaseBIConnector,
        request: DashboardExportRequest,
        job_id: str,
        db: Session
    ):
        """Async dashboard export worker"""
        try:
            # Update job status
            export_job = db.query(BIExportJob).filter(BIExportJob.id == job_id).first()
            export_job.status = "running"
            db.commit()
            
            # Perform export
            content = await connector.export_dashboard(
                request.dashboard_id,
                request.format,
                request.filters,
                request.parameters
            )
            
            # Save file (in production, save to cloud storage)
            file_path = f"/tmp/exports/{job_id}.{request.format.value}"
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Update job
            export_job.status = "completed"
            export_job.file_path = file_path
            export_job.completed_at = datetime.utcnow()
            db.commit()
            
        except Exception as e:
            logger.error(f"Export job {job_id} failed: {str(e)}")
            export_job = db.query(BIExportJob).filter(BIExportJob.id == job_id).first()
            if export_job:
                export_job.status = "failed"
                export_job.error_message = str(e)
                export_job.completed_at = datetime.utcnow()
                db.commit()
    
    async def create_embed_token(
        self,
        connection_id: str,
        request: EmbedTokenRequest,
        db: Session
    ) -> EmbedTokenResponse:
        """Create an embed token for a dashboard"""
        try:
            connector = await self.get_connection(connection_id, db)
            if not connector:
                raise BIConnectorError("Connection not found")
            
            return await connector.create_embed_token(
                request.dashboard_id,
                request.user_id,
                request.embed_type,
                request.permissions,
                request.expiry_minutes
            )
        
        except Exception as e:
            logger.error(f"Error creating embed token: {str(e)}")
            raise BIConnectorError(f"Failed to create embed token: {str(e)}")
    
    async def sync_data_source(
        self,
        connection_id: str,
        request: DataSyncRequest,
        db: Session
    ) -> DataSyncResult:
        """Synchronize a data source"""
        try:
            connector = await self.get_connection(connection_id, db)
            if not connector:
                raise BIConnectorError("Connection not found")
            
            result = await connector.sync_data_source(
                request.data_source_id,
                request.incremental,
                request.filters
            )
            
            # Update last sync time in database
            data_source = db.query(BIDataSource).filter(
                BIDataSource.connection_id == connection_id,
                BIDataSource.id == request.data_source_id
            ).first()
            
            if data_source:
                data_source.last_refresh = result.last_sync_time
                db.commit()
            
            return result
        
        except Exception as e:
            logger.error(f"Error syncing data source: {str(e)}")
            return DataSyncResult(
                success=False,
                records_processed=0,
                records_updated=0,
                records_created=0,
                errors=[str(e)],
                sync_duration=0,
                last_sync_time=datetime.utcnow()
            )
    
    async def get_integration_status(
        self,
        connection_id: str,
        db: Session
    ) -> BIIntegrationStatus:
        """Get integration status for a connection"""
        try:
            # Get connection info
            db_connection = db.query(BIConnection).filter(BIConnection.id == connection_id).first()
            if not db_connection:
                raise BIConnectorError("Connection not found")
            
            # Count dashboards and data sources
            dashboards_count = db.query(BIDashboard).filter(BIDashboard.connection_id == connection_id).count()
            data_sources_count = db.query(BIDataSource).filter(BIDataSource.connection_id == connection_id).count()
            
            # Get health check
            connector = await self.get_connection(connection_id, db)
            health_check = {}
            if connector:
                health_check = await connector.health_check()
            
            return BIIntegrationStatus(
                connection_id=connection_id,
                status=ConnectionStatus(db_connection.status),
                dashboards_count=dashboards_count,
                data_sources_count=data_sources_count,
                last_activity=db_connection.last_sync,
                health_check=health_check
            )
        
        except Exception as e:
            logger.error(f"Error getting integration status: {str(e)}")
            raise BIConnectorError(f"Failed to get status: {str(e)}")
    
    async def get_supported_tools(self) -> List[Dict[str, Any]]:
        """Get list of supported BI tools"""
        supported_tools = []
        
        for tool_type in bi_connector_registry.get_available_tools():
            # Get a sample connector to check capabilities
            sample_config = {'id': 'sample', 'name': 'sample', 'bi_tool_type': tool_type}
            connector = bi_connector_registry.get_connector(tool_type, sample_config)
            
            supported_tools.append({
                'tool_type': tool_type.value,
                'name': tool_type.value.replace('_', ' ').title(),
                'supported_export_formats': [fmt.value for fmt in connector.get_supported_export_formats()],
                'supported_embed_types': [embed.value for embed in connector.get_supported_embed_types()],
                'required_config_fields': connector.get_required_config_fields()
            })
        
        return supported_tools
    
    async def delete_connection(self, connection_id: str, db: Session) -> bool:
        """Delete a BI connection"""
        try:
            # Remove from active connections
            if connection_id in self.active_connections:
                connector = self.active_connections[connection_id]
                await connector.__aexit__(None, None, None)  # Cleanup
                del self.active_connections[connection_id]
            
            # Delete from database
            db_connection = db.query(BIConnection).filter(BIConnection.id == connection_id).first()
            if db_connection:
                # Delete related records
                db.query(BIDashboard).filter(BIDashboard.connection_id == connection_id).delete()
                db.query(BIDataSource).filter(BIDataSource.connection_id == connection_id).delete()
                db.query(BIExportJob).filter(BIExportJob.dashboard_id.in_(
                    db.query(BIDashboard.external_id).filter(BIDashboard.connection_id == connection_id)
                )).delete()
                
                db.delete(db_connection)
                db.commit()
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error deleting connection {connection_id}: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup all active connections"""
        for connection_id, connector in self.active_connections.items():
            try:
                await connector.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up connection {connection_id}: {str(e)}")
        
        self.active_connections.clear()


# Global instance
bi_integration_engine = BIIntegrationEngine()