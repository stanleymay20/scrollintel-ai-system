"""
Tableau Server/Online Connector
Implements Tableau integration with embedding and export capabilities
"""

import asyncio
import aiohttp
import jwt
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin
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


@register_bi_connector(BIToolType.TABLEAU)
class TableauConnector(BaseBIConnector):
    """
    Tableau Server/Online connector implementation
    Supports REST API v3.x and embedding API
    """
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__(connection_config)
        self.server_url = connection_config.get('server_url')
        self.username = connection_config.get('username')
        self.password = connection_config.get('password')
        self.site_id = connection_config.get('site_id', 'default')
        self.api_version = connection_config.get('api_version', '3.19')
        self.connected_app_client_id = connection_config.get('connected_app_client_id')
        self.connected_app_secret = connection_config.get('connected_app_secret')
        
        self.auth_token = None
        self.site_uuid = None
        self.user_id = None
        self.session = None
        
        # API endpoints
        self.api_base = f"{self.server_url}/api/{self.api_version}"
    
    @property
    def tool_type(self) -> BIToolType:
        return BIToolType.TABLEAU
    
    def get_required_config_fields(self) -> List[str]:
        return ['server_url', 'username', 'password', 'site_id']
    
    async def authenticate(self) -> bool:
        """Authenticate with Tableau Server using REST API"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Sign in request
            signin_url = f"{self.api_base}/auth/signin"
            
            # Create XML payload for authentication
            credentials_xml = f"""
            <tsRequest>
                <credentials name="{self.username}" password="{self.password}">
                    <site contentUrl="{self.site_id}" />
                </credentials>
            </tsRequest>
            """
            
            headers = {
                'Content-Type': 'application/xml',
                'Accept': 'application/xml'
            }
            
            async with self.session.post(signin_url, data=credentials_xml, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AuthenticationError(f"Tableau authentication failed: {error_text}")
                
                # Parse response XML
                response_xml = await response.text()
                root = ET.fromstring(response_xml)
                
                # Extract authentication details
                credentials = root.find('.//credentials')
                if credentials is not None:
                    self.auth_token = credentials.get('token')
                    site = credentials.find('site')
                    if site is not None:
                        self.site_uuid = site.get('id')
                
                user = root.find('.//user')
                if user is not None:
                    self.user_id = user.get('id')
            
            if not self.auth_token:
                raise AuthenticationError("Failed to obtain authentication token")
            
            self._authenticated = True
            self._update_activity()
            logger.info(f"Successfully authenticated with Tableau Server: {self.server_url}")
            return True
            
        except Exception as e:
            logger.error(f"Tableau authentication error: {str(e)}")
            raise AuthenticationError(f"Tableau authentication failed: {str(e)}")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Tableau Server"""
        try:
            if not self._authenticated:
                await self.authenticate()
            
            # Test by getting server info
            server_info_url = f"{self.api_base}/serverinfo"
            headers = {'X-Tableau-Auth': self.auth_token}
            
            async with self.session.get(server_info_url, headers=headers) as response:
                if response.status == 200:
                    server_info_xml = await response.text()
                    root = ET.fromstring(server_info_xml)
                    server_info = root.find('.//serverInfo')
                    
                    return {
                        "status": "connected",
                        "server_version": server_info.get('productVersion') if server_info is not None else "unknown",
                        "api_version": self.api_version,
                        "site_id": self.site_id,
                        "authenticated_user": self.username
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}"
                    }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_dashboards(self, project_id: Optional[str] = None) -> List[BIDashboardInfo]:
        """Get list of dashboards from Tableau Server"""
        if not self._authenticated:
            await self.authenticate()
        
        try:
            # Get workbooks (which contain dashboards)
            workbooks_url = f"{self.api_base}/sites/{self.site_uuid}/workbooks"
            if project_id:
                workbooks_url += f"?filter=projectId:eq:{project_id}"
            
            headers = {'X-Tableau-Auth': self.auth_token}
            
            dashboards = []
            
            async with self.session.get(workbooks_url, headers=headers) as response:
                if response.status != 200:
                    raise ConnectionError(f"Failed to get workbooks: HTTP {response.status}")
                
                workbooks_xml = await response.text()
                root = ET.fromstring(workbooks_xml)
                
                for workbook in root.findall('.//workbook'):
                    workbook_id = workbook.get('id')
                    workbook_name = workbook.get('name')
                    
                    # Get views (dashboards) for this workbook
                    views_url = f"{self.api_base}/sites/{self.site_uuid}/workbooks/{workbook_id}/views"
                    
                    async with self.session.get(views_url, headers=headers) as views_response:
                        if views_response.status == 200:
                            views_xml = await views_response.text()
                            views_root = ET.fromstring(views_xml)
                            
                            for view in views_root.findall('.//view'):
                                dashboard_info = BIDashboardInfo(
                                    id=view.get('id'),
                                    name=f"{workbook_name} - {view.get('name')}",
                                    description=f"Dashboard from workbook: {workbook_name}",
                                    url=f"{self.server_url}/#/views/{view.get('contentUrl')}",
                                    is_public=False,  # Tableau dashboards are typically private
                                    embed_available=True,
                                    last_updated=datetime.utcnow()  # Would need to parse from XML
                                )
                                dashboards.append(dashboard_info)
            
            self._update_activity()
            return dashboards
            
        except Exception as e:
            logger.error(f"Error getting Tableau dashboards: {str(e)}")
            raise ConnectionError(f"Failed to get dashboards: {str(e)}")
    
    async def get_dashboard_info(self, dashboard_id: str) -> BIDashboardInfo:
        """Get detailed information about a specific dashboard"""
        if not self._authenticated:
            await self.authenticate()
        
        try:
            view_url = f"{self.api_base}/sites/{self.site_uuid}/views/{dashboard_id}"
            headers = {'X-Tableau-Auth': self.auth_token}
            
            async with self.session.get(view_url, headers=headers) as response:
                if response.status != 200:
                    raise ConnectionError(f"Dashboard not found: {dashboard_id}")
                
                view_xml = await response.text()
                root = ET.fromstring(view_xml)
                view = root.find('.//view')
                
                if view is None:
                    raise ConnectionError(f"Invalid dashboard response for: {dashboard_id}")
                
                return BIDashboardInfo(
                    id=view.get('id'),
                    name=view.get('name'),
                    description=f"Tableau dashboard: {view.get('name')}",
                    url=f"{self.server_url}/#/views/{view.get('contentUrl')}",
                    is_public=False,
                    embed_available=True,
                    last_updated=datetime.utcnow()
                )
        
        except Exception as e:
            logger.error(f"Error getting Tableau dashboard info: {str(e)}")
            raise ConnectionError(f"Failed to get dashboard info: {str(e)}")
    
    async def export_dashboard(
        self, 
        dashboard_id: str, 
        format: ExportFormat,
        filters: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Export Tableau dashboard in specified format"""
        if not self._authenticated:
            await self.authenticate()
        
        try:
            # Map export formats to Tableau API formats
            format_mapping = {
                ExportFormat.PDF: 'pdf',
                ExportFormat.PNG: 'png',
                ExportFormat.CSV: 'csv'
            }
            
            if format not in format_mapping:
                raise ExportError(f"Unsupported export format: {format}")
            
            tableau_format = format_mapping[format]
            
            # Build export URL
            export_url = f"{self.api_base}/sites/{self.site_uuid}/views/{dashboard_id}/{tableau_format}"
            
            # Add filters and parameters as query string
            params = {}
            if filters:
                for key, value in filters.items():
                    params[f"vf_{key}"] = value
            
            if parameters:
                params.update(parameters)
            
            headers = {'X-Tableau-Auth': self.auth_token}
            
            async with self.session.get(export_url, headers=headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ExportError(f"Export failed: {error_text}")
                
                content = await response.read()
                self._update_activity()
                return content
        
        except Exception as e:
            logger.error(f"Error exporting Tableau dashboard: {str(e)}")
            raise ExportError(f"Failed to export dashboard: {str(e)}")
    
    async def create_embed_token(
        self,
        dashboard_id: str,
        user_id: str,
        embed_type: EmbedType = EmbedType.IFRAME,
        permissions: Optional[List[str]] = None,
        expiry_minutes: int = 60
    ) -> EmbedTokenResponse:
        """Create embed token for Tableau dashboard"""
        try:
            if not self.connected_app_client_id or not self.connected_app_secret:
                raise EmbedError("Connected App credentials required for embedding")
            
            # Create JWT token for Tableau embedding
            now = datetime.utcnow()
            exp = now + timedelta(minutes=expiry_minutes)
            
            payload = {
                "iss": self.connected_app_client_id,
                "exp": int(exp.timestamp()),
                "jti": f"{user_id}_{dashboard_id}_{int(now.timestamp())}",
                "aud": "tableau",
                "sub": user_id,
                "scp": permissions or ["tableau:views:embed"]
            }
            
            token = jwt.encode(payload, self.connected_app_secret, algorithm="HS256")
            
            # Build embed URL
            embed_url = f"{self.server_url}/trusted/{token}/views/{dashboard_id}"
            
            # Generate iframe embed code
            embed_code = None
            if embed_type == EmbedType.IFRAME:
                embed_code = f'''
                <iframe 
                    src="{embed_url}" 
                    width="100%" 
                    height="600"
                    frameborder="0">
                </iframe>
                '''
            
            return EmbedTokenResponse(
                token=token,
                embed_url=embed_url,
                expires_at=exp,
                embed_code=embed_code
            )
        
        except Exception as e:
            logger.error(f"Error creating Tableau embed token: {str(e)}")
            raise EmbedError(f"Failed to create embed token: {str(e)}")
    
    async def sync_data_source(
        self,
        data_source_id: str,
        incremental: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> DataSyncResult:
        """Refresh Tableau data source"""
        if not self._authenticated:
            await self.authenticate()
        
        try:
            start_time = datetime.utcnow()
            
            # Trigger data source refresh
            refresh_url = f"{self.api_base}/sites/{self.site_uuid}/datasources/{data_source_id}/refresh"
            headers = {'X-Tableau-Auth': self.auth_token}
            
            async with self.session.post(refresh_url, headers=headers) as response:
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
                
                # For Tableau, we can't get exact record counts from refresh
                # This would typically be an asynchronous operation
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                return DataSyncResult(
                    success=True,
                    records_processed=1,  # Placeholder - Tableau doesn't provide this
                    records_updated=1,
                    records_created=0,
                    errors=[],
                    sync_duration=duration,
                    last_sync_time=end_time
                )
        
        except Exception as e:
            logger.error(f"Error syncing Tableau data source: {str(e)}")
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
        """Create new data source in Tableau"""
        # This would require publishing a data source file (.tds or .tdsx)
        # Implementation depends on specific requirements
        raise NotImplementedError("Data source creation requires file upload - not implemented in this version")
    
    async def update_data_source(
        self,
        data_source_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """Update Tableau data source configuration"""
        # This would require updating connection details
        # Implementation depends on specific requirements
        raise NotImplementedError("Data source update not implemented in this version")
    
    async def delete_data_source(self, data_source_id: str) -> bool:
        """Delete Tableau data source"""
        if not self._authenticated:
            await self.authenticate()
        
        try:
            delete_url = f"{self.api_base}/sites/{self.site_uuid}/datasources/{data_source_id}"
            headers = {'X-Tableau-Auth': self.auth_token}
            
            async with self.session.delete(delete_url, headers=headers) as response:
                return response.status == 204
        
        except Exception as e:
            logger.error(f"Error deleting Tableau data source: {str(e)}")
            return False
    
    async def get_real_time_data_feed(
        self,
        data_source_id: str,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set up real-time data feed for Tableau"""
        # Tableau doesn't have native real-time streaming
        # This would typically involve setting up webhooks or scheduled refreshes
        return {
            "supported": False,
            "message": "Tableau requires scheduled refreshes or manual triggers for data updates",
            "alternatives": ["scheduled_refresh", "manual_refresh", "extract_refresh"]
        }
    
    def get_supported_export_formats(self) -> List[ExportFormat]:
        """Return supported export formats for Tableau"""
        return [
            ExportFormat.PDF,
            ExportFormat.PNG,
            ExportFormat.CSV
        ]
    
    def get_supported_embed_types(self) -> List[EmbedType]:
        """Return supported embed types for Tableau"""
        return [
            EmbedType.IFRAME,
            EmbedType.JAVASCRIPT
        ]
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up Tableau session"""
        if self.session and self.auth_token:
            try:
                # Sign out
                signout_url = f"{self.api_base}/auth/signout"
                headers = {'X-Tableau-Auth': self.auth_token}
                await self.session.post(signout_url, headers=headers)
            except Exception:
                pass  # Ignore signout errors
            
            await self.session.close()
            self.session = None
            self.auth_token = None
            self._authenticated = False