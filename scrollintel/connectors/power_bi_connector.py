"""
Microsoft Power BI Connector
Implements Power BI integration with real-time data feeds and embedding
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode
import logging

from scrollintel.connectors.bi_connector_base import (
    BaseBIConnector, register_bi_connector, AuthenticationError, 
    ConnectionError, ExportError, EmbedError
)
from scrollintel.models.bi_integration_models import (
    BIToolType, ExportFormat, EmbedType, BIDashboardInfo,
    DataSyncResult, EmbedTokenResponse
)

logger = logging.getLogger(__name__)


@register_bi_connector(BIToolType.POWER_BI)
class PowerBIConnector(BaseBIConnector):
    """
    Microsoft Power BI connector implementation
    Supports Power BI REST API and embedding
    """
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__(connection_config)
        self.tenant_id = connection_config.get('tenant_id')
        self.client_id = connection_config.get('client_id')
        self.client_secret = connection_config.get('client_secret')
        self.username = connection_config.get('username')
        self.password = connection_config.get('password')
        self.workspace_id = connection_config.get('workspace_id')
        
        self.access_token = None
        self.token_expires_at = None
        self.session = None
        
        # Power BI API endpoints
        self.auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        self.api_base = "https://api.powerbi.com/v1.0/myorg"
    
    @property
    def tool_type(self) -> BIToolType:
        return BIToolType.POWER_BI
    
    def get_required_config_fields(self) -> List[str]:
        return ['tenant_id', 'client_id', 'client_secret', 'workspace_id']
    
    async def authenticate(self) -> bool:
        """Authenticate with Power BI using OAuth2"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Prepare authentication request
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'https://analysis.windows.net/powerbi/api/.default'
            }
            
            # If username/password provided, use resource owner password credentials
            if self.username and self.password:
                auth_data.update({
                    'grant_type': 'password',
                    'username': self.username,
                    'password': self.password,
                    'scope': 'https://analysis.windows.net/powerbi/api/Report.ReadWrite.All Dataset.ReadWrite.All Dashboard.Read.All'
                })
            
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            
            async with self.session.post(self.auth_url, data=urlencode(auth_data), headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AuthenticationError(f"Power BI authentication failed: {error_text}")
                
                auth_response = await response.json()
                self.access_token = auth_response.get('access_token')
                expires_in = auth_response.get('expires_in', 3600)
                self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            
            if not self.access_token:
                raise AuthenticationError("Failed to obtain access token")
            
            self._authenticated = True
            self._update_activity()
            logger.info("Successfully authenticated with Power BI")
            return True
            
        except Exception as e:
            logger.error(f"Power BI authentication error: {str(e)}")
            raise AuthenticationError(f"Power BI authentication failed: {str(e)}")
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid authentication token"""
        if not self._authenticated or (self.token_expires_at and datetime.utcnow() >= self.token_expires_at):
            await self.authenticate()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Power BI"""
        try:
            await self._ensure_authenticated()
            
            # Test by getting workspace info
            workspace_url = f"{self.api_base}/groups/{self.workspace_id}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.get(workspace_url, headers=headers) as response:
                if response.status == 200:
                    workspace_info = await response.json()
                    return {
                        "status": "connected",
                        "workspace_name": workspace_info.get('name'),
                        "workspace_id": self.workspace_id,
                        "api_version": "v1.0"
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_dashboards(self, project_id: Optional[str] = None) -> List[BIDashboardInfo]:
        """Get list of dashboards from Power BI workspace"""
        await self._ensure_authenticated()
        
        try:
            # Get dashboards from workspace
            dashboards_url = f"{self.api_base}/groups/{self.workspace_id}/dashboards"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            dashboards = []
            
            async with self.session.get(dashboards_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to get dashboards: {error_text}")
                
                dashboards_data = await response.json()
                
                for dashboard in dashboards_data.get('value', []):
                    dashboard_info = BIDashboardInfo(
                        id=dashboard['id'],
                        name=dashboard['displayName'],
                        description=f"Power BI Dashboard: {dashboard['displayName']}",
                        url=dashboard.get('webUrl', ''),
                        is_public=dashboard.get('isReadOnly', False),
                        embed_available=True,
                        last_updated=datetime.utcnow()  # Power BI doesn't provide this directly
                    )
                    dashboards.append(dashboard_info)
            
            self._update_activity()
            return dashboards
            
        except Exception as e:
            logger.error(f"Error getting Power BI dashboards: {str(e)}")
            raise ConnectionError(f"Failed to get dashboards: {str(e)}")
    
    async def get_dashboard_info(self, dashboard_id: str) -> BIDashboardInfo:
        """Get detailed information about a specific dashboard"""
        await self._ensure_authenticated()
        
        try:
            dashboard_url = f"{self.api_base}/groups/{self.workspace_id}/dashboards/{dashboard_id}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.get(dashboard_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectionError(f"Dashboard not found: {error_text}")
                
                dashboard = await response.json()
                
                return BIDashboardInfo(
                    id=dashboard['id'],
                    name=dashboard['displayName'],
                    description=f"Power BI Dashboard: {dashboard['displayName']}",
                    url=dashboard.get('webUrl', ''),
                    is_public=dashboard.get('isReadOnly', False),
                    embed_available=True,
                    last_updated=datetime.utcnow()
                )
        
        except Exception as e:
            logger.error(f"Error getting Power BI dashboard info: {str(e)}")
            raise ConnectionError(f"Failed to get dashboard info: {str(e)}")
    
    async def export_dashboard(
        self, 
        dashboard_id: str, 
        format: ExportFormat,
        filters: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Export Power BI dashboard in specified format"""
        await self._ensure_authenticated()
        
        try:
            # Power BI export is typically done through reports, not dashboards
            # First, get the reports in the dashboard
            tiles_url = f"{self.api_base}/groups/{self.workspace_id}/dashboards/{dashboard_id}/tiles"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.get(tiles_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ExportError(f"Failed to get dashboard tiles: {error_text}")
                
                tiles_data = await response.json()
                
                # Find the first report tile to export
                report_id = None
                for tile in tiles_data.get('value', []):
                    if tile.get('reportId'):
                        report_id = tile['reportId']
                        break
                
                if not report_id:
                    raise ExportError("No exportable reports found in dashboard")
            
            # Export the report
            format_mapping = {
                ExportFormat.PDF: 'PDF',
                ExportFormat.PNG: 'PNG',
                ExportFormat.JPEG: 'IMAGE'
            }
            
            if format not in format_mapping:
                raise ExportError(f"Unsupported export format: {format}")
            
            powerbi_format = format_mapping[format]
            
            # Start export job
            export_url = f"{self.api_base}/groups/{self.workspace_id}/reports/{report_id}/ExportTo"
            export_request = {
                "format": powerbi_format,
                "powerBIReportConfiguration": {
                    "settings": {
                        "includeHiddenPages": False
                    }
                }
            }
            
            if filters:
                export_request["powerBIReportConfiguration"]["reportLevelFilters"] = [
                    {"filter": f"{key} eq '{value}'"} for key, value in filters.items()
                ]
            
            async with self.session.post(export_url, json=export_request, headers=headers) as response:
                if response.status != 202:
                    error_text = await response.text()
                    raise ExportError(f"Export job failed to start: {error_text}")
                
                export_job = await response.json()
                export_id = export_job['id']
            
            # Poll for completion
            status_url = f"{self.api_base}/groups/{self.workspace_id}/reports/{report_id}/exports/{export_id}"
            
            for _ in range(30):  # Wait up to 5 minutes
                await asyncio.sleep(10)
                
                async with self.session.get(status_url, headers=headers) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        if status_data['status'] == 'Succeeded':
                            # Download the file
                            file_url = f"{status_url}/file"
                            async with self.session.get(file_url, headers=headers) as file_response:
                                if file_response.status == 200:
                                    content = await file_response.read()
                                    self._update_activity()
                                    return content
                                else:
                                    raise ExportError("Failed to download exported file")
                        elif status_data['status'] == 'Failed':
                            raise ExportError(f"Export failed: {status_data.get('error', 'Unknown error')}")
            
            raise ExportError("Export timeout - job did not complete in time")
        
        except Exception as e:
            logger.error(f"Error exporting Power BI dashboard: {str(e)}")
            raise ExportError(f"Failed to export dashboard: {str(e)}")
    
    async def create_embed_token(
        self,
        dashboard_id: str,
        user_id: str,
        embed_type: EmbedType = EmbedType.IFRAME,
        permissions: Optional[List[str]] = None,
        expiry_minutes: int = 60
    ) -> EmbedTokenResponse:
        """Create embed token for Power BI dashboard"""
        await self._ensure_authenticated()
        
        try:
            # Generate embed token
            token_url = f"{self.api_base}/groups/{self.workspace_id}/dashboards/{dashboard_id}/GenerateToken"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            token_request = {
                "accessLevel": "View",
                "allowSaveAs": False,
                "identities": [
                    {
                        "username": user_id,
                        "roles": permissions or ["Viewer"]
                    }
                ] if permissions else []
            }
            
            async with self.session.post(token_url, json=token_request, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise EmbedError(f"Failed to generate embed token: {error_text}")
                
                token_data = await response.json()
                embed_token = token_data['token']
                expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
            
            # Build embed URL
            embed_url = f"https://app.powerbi.com/dashboardEmbed?dashboardId={dashboard_id}&groupId={self.workspace_id}"
            
            # Generate embed code
            embed_code = None
            if embed_type == EmbedType.IFRAME:
                embed_code = f'''
                <iframe 
                    width="100%" 
                    height="600" 
                    src="{embed_url}" 
                    frameborder="0" 
                    allowFullScreen="true">
                </iframe>
                <script>
                    // Power BI JavaScript API integration would go here
                    // Requires powerbi-client library
                </script>
                '''
            elif embed_type == EmbedType.JAVASCRIPT:
                embed_code = f'''
                <div id="powerbi-dashboard"></div>
                <script src="https://cdn.jsdelivr.net/npm/powerbi-client@2.19.1/dist/powerbi.min.js"></script>
                <script>
                    const embedConfig = {{
                        type: 'dashboard',
                        id: '{dashboard_id}',
                        embedUrl: '{embed_url}',
                        accessToken: '{embed_token}',
                        tokenType: models.TokenType.Embed,
                        settings: {{
                            panes: {{
                                filters: {{ expanded: false, visible: true }},
                                pageNavigation: {{ visible: true }}
                            }}
                        }}
                    }};
                    
                    const dashboardContainer = document.getElementById('powerbi-dashboard');
                    const dashboard = powerbi.embed(dashboardContainer, embedConfig);
                </script>
                '''
            
            return EmbedTokenResponse(
                token=embed_token,
                embed_url=embed_url,
                expires_at=expires_at,
                embed_code=embed_code
            )
        
        except Exception as e:
            logger.error(f"Error creating Power BI embed token: {str(e)}")
            raise EmbedError(f"Failed to create embed token: {str(e)}")
    
    async def sync_data_source(
        self,
        data_source_id: str,
        incremental: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> DataSyncResult:
        """Refresh Power BI dataset"""
        await self._ensure_authenticated()
        
        try:
            start_time = datetime.utcnow()
            
            # Trigger dataset refresh
            refresh_url = f"{self.api_base}/groups/{self.workspace_id}/datasets/{data_source_id}/refreshes"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            refresh_request = {
                "type": "full" if not incremental else "automatic",
                "commitMode": "transactional"
            }
            
            async with self.session.post(refresh_url, json=refresh_request, headers=headers) as response:
                if response.status not in [200, 202]:
                    error_text = await response.text()
                    return DataSyncResult(
                        success=False,
                        records_processed=0,
                        records_updated=0,
                        records_created=0,
                        errors=[f"Refresh failed: {error_text}"],
                        sync_duration=0,
                        last_sync_time=start_time
                    )
                
                # Power BI refresh is asynchronous, we could poll for completion
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                return DataSyncResult(
                    success=True,
                    records_processed=1,  # Placeholder - Power BI doesn't provide exact counts
                    records_updated=1,
                    records_created=0,
                    errors=[],
                    sync_duration=duration,
                    last_sync_time=end_time
                )
        
        except Exception as e:
            logger.error(f"Error syncing Power BI dataset: {str(e)}")
            return DataSyncResult(
                success=False,
                records_processed=0,
                records_updated=0,
                records_created=0,
                errors=[str(e)],
                sync_duration=0,
                last_sync_time=datetime.utcnow()
            )
    
    async def create_data_source(
        self,
        name: str,
        connection_string: str,
        source_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new dataset in Power BI"""
        await self._ensure_authenticated()
        
        try:
            # Create dataset schema
            dataset_schema = {
                "name": name,
                "tables": [
                    {
                        "name": "MainTable",
                        "columns": [
                            {"name": "ID", "dataType": "Int64"},
                            {"name": "Name", "dataType": "string"},
                            {"name": "Value", "dataType": "Double"},
                            {"name": "Date", "dataType": "DateTime"}
                        ]
                    }
                ]
            }
            
            # Add custom schema if provided
            if config and 'schema' in config:
                dataset_schema.update(config['schema'])
            
            create_url = f"{self.api_base}/groups/{self.workspace_id}/datasets"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.post(create_url, json=dataset_schema, headers=headers) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to create dataset: {error_text}")
                
                dataset = await response.json()
                return dataset['id']
        
        except Exception as e:
            logger.error(f"Error creating Power BI dataset: {str(e)}")
            raise ConnectionError(f"Failed to create dataset: {str(e)}")
    
    async def update_data_source(
        self,
        data_source_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """Update Power BI dataset configuration"""
        await self._ensure_authenticated()
        
        try:
            # Update dataset properties
            update_url = f"{self.api_base}/groups/{self.workspace_id}/datasets/{data_source_id}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.patch(update_url, json=config, headers=headers) as response:
                return response.status == 200
        
        except Exception as e:
            logger.error(f"Error updating Power BI dataset: {str(e)}")
            return False
    
    async def delete_data_source(self, data_source_id: str) -> bool:
        """Delete Power BI dataset"""
        await self._ensure_authenticated()
        
        try:
            delete_url = f"{self.api_base}/groups/{self.workspace_id}/datasets/{data_source_id}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.delete(delete_url, headers=headers) as response:
                return response.status == 200
        
        except Exception as e:
            logger.error(f"Error deleting Power BI dataset: {str(e)}")
            return False
    
    async def get_real_time_data_feed(
        self,
        data_source_id: str,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set up real-time data feed for Power BI"""
        await self._ensure_authenticated()
        
        try:
            # Power BI supports real-time streaming through push datasets
            streaming_url = f"{self.api_base}/groups/{self.workspace_id}/datasets/{data_source_id}/rows"
            
            return {
                "supported": True,
                "streaming_endpoint": streaming_url,
                "method": "POST",
                "authentication": "Bearer token required",
                "data_format": "JSON array of objects",
                "max_rows_per_request": 10000,
                "rate_limit": "120 requests per minute"
            }
        
        except Exception as e:
            logger.error(f"Error setting up Power BI real-time feed: {str(e)}")
            return {
                "supported": False,
                "error": str(e)
            }
    
    def get_supported_export_formats(self) -> List[ExportFormat]:
        """Return supported export formats for Power BI"""
        return [
            ExportFormat.PDF,
            ExportFormat.PNG,
            ExportFormat.JPEG
        ]
    
    def get_supported_embed_types(self) -> List[EmbedType]:
        """Return supported embed types for Power BI"""
        return [
            EmbedType.IFRAME,
            EmbedType.JAVASCRIPT
        ]
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up Power BI session"""
        if self.session:
            await self.session.close()
            self.session = None
            self.access_token = None
            self._authenticated = False