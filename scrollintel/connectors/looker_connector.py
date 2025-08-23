"""
Looker Connector
Implements Looker integration with API access and embedding
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


@register_bi_connector(BIToolType.LOOKER)
class LookerConnector(BaseBIConnector):
    """
    Looker connector implementation
    Supports Looker API 4.0 and embedding
    """
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__(connection_config)
        self.base_url = connection_config.get('base_url')  # e.g., https://company.looker.com
        self.client_id = connection_config.get('client_id')
        self.client_secret = connection_config.get('client_secret')
        self.embed_secret = connection_config.get('embed_secret')  # For SSO embedding
        
        self.access_token = None
        self.token_expires_at = None
        self.session = None
        
        # Looker API endpoints
        self.api_base = f"{self.base_url}/api/4.0"
    
    @property
    def tool_type(self) -> BIToolType:
        return BIToolType.LOOKER
    
    def get_required_config_fields(self) -> List[str]:
        return ['base_url', 'client_id', 'client_secret']
    
    async def authenticate(self) -> bool:
        """Authenticate with Looker using API credentials"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Looker authentication
            auth_url = f"{self.api_base}/login"
            auth_data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            
            async with self.session.post(auth_url, data=urlencode(auth_data), headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AuthenticationError(f"Looker authentication failed: {error_text}")
                
                auth_response = await response.json()
                self.access_token = auth_response.get('access_token')
                expires_in = auth_response.get('expires_in', 3600)
                self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            
            if not self.access_token:
                raise AuthenticationError("Failed to obtain access token")
            
            self._authenticated = True
            self._update_activity()
            logger.info(f"Successfully authenticated with Looker: {self.base_url}")
            return True
            
        except Exception as e:
            logger.error(f"Looker authentication error: {str(e)}")
            raise AuthenticationError(f"Looker authentication failed: {str(e)}")
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid authentication token"""
        if not self._authenticated or (self.token_expires_at and datetime.utcnow() >= self.token_expires_at):
            await self.authenticate()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Looker"""
        try:
            await self._ensure_authenticated()
            
            # Test by getting user info
            user_url = f"{self.api_base}/user"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.get(user_url, headers=headers) as response:
                if response.status == 200:
                    user_info = await response.json()
                    return {
                        "status": "connected",
                        "user_id": user_info.get('id'),
                        "user_name": user_info.get('display_name'),
                        "api_version": "4.0",
                        "looker_version": user_info.get('looker_version')
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
        """Get list of dashboards from Looker"""
        await self._ensure_authenticated()
        
        try:
            # Get dashboards
            dashboards_url = f"{self.api_base}/dashboards"
            if project_id:
                dashboards_url += f"?space_id={project_id}"
            
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            dashboards = []
            
            async with self.session.get(dashboards_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to get dashboards: {error_text}")
                
                dashboards_data = await response.json()
                
                for dashboard in dashboards_data:
                    dashboard_info = BIDashboardInfo(
                        id=dashboard['id'],
                        name=dashboard.get('title', 'Untitled Dashboard'),
                        description=dashboard.get('description', ''),
                        url=f"{self.base_url}/dashboards/{dashboard['id']}",
                        is_public=dashboard.get('public', False),
                        embed_available=True,
                        last_updated=datetime.fromisoformat(dashboard['updated_at'].replace('Z', '+00:00')) if dashboard.get('updated_at') else datetime.utcnow()
                    )
                    dashboards.append(dashboard_info)
            
            self._update_activity()
            return dashboards
            
        except Exception as e:
            logger.error(f"Error getting Looker dashboards: {str(e)}")
            raise ConnectionError(f"Failed to get dashboards: {str(e)}")
    
    async def get_dashboard_info(self, dashboard_id: str) -> BIDashboardInfo:
        """Get detailed information about a specific dashboard"""
        await self._ensure_authenticated()
        
        try:
            dashboard_url = f"{self.api_base}/dashboards/{dashboard_id}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.get(dashboard_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectionError(f"Dashboard not found: {error_text}")
                
                dashboard = await response.json()
                
                return BIDashboardInfo(
                    id=dashboard['id'],
                    name=dashboard.get('title', 'Untitled Dashboard'),
                    description=dashboard.get('description', ''),
                    url=f"{self.base_url}/dashboards/{dashboard['id']}",
                    is_public=dashboard.get('public', False),
                    embed_available=True,
                    last_updated=datetime.fromisoformat(dashboard['updated_at'].replace('Z', '+00:00')) if dashboard.get('updated_at') else datetime.utcnow()
                )
        
        except Exception as e:
            logger.error(f"Error getting Looker dashboard info: {str(e)}")
            raise ConnectionError(f"Failed to get dashboard info: {str(e)}")
    
    async def export_dashboard(
        self, 
        dashboard_id: str, 
        format: ExportFormat,
        filters: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Export Looker dashboard in specified format"""
        await self._ensure_authenticated()
        
        try:
            # Map export formats to Looker formats
            format_mapping = {
                ExportFormat.PDF: 'pdf',
                ExportFormat.PNG: 'png',
                ExportFormat.JPEG: 'jpg',
                ExportFormat.CSV: 'csv_zip'  # Looker exports CSV as zip
            }
            
            if format not in format_mapping:
                raise ExportError(f"Unsupported export format: {format}")
            
            looker_format = format_mapping[format]
            
            # Create render task
            render_url = f"{self.api_base}/render_tasks/dashboards/{dashboard_id}/{looker_format}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            render_request = {
                "result_format": looker_format,
                "width": 1024,
                "height": 768
            }
            
            # Add filters if provided
            if filters:
                filter_params = []
                for key, value in filters.items():
                    filter_params.append(f"{key}={value}")
                render_request["filters"] = "&".join(filter_params)
            
            async with self.session.post(render_url, json=render_request, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ExportError(f"Render task failed: {error_text}")
                
                render_task = await response.json()
                task_id = render_task['id']
            
            # Poll for completion
            task_url = f"{self.api_base}/render_tasks/{task_id}"
            
            for _ in range(30):  # Wait up to 5 minutes
                await asyncio.sleep(10)
                
                async with self.session.get(task_url, headers=headers) as response:
                    if response.status == 200:
                        task_status = await response.json()
                        if task_status['status'] == 'success':
                            # Download the result
                            result_url = f"{self.api_base}/render_tasks/{task_id}/results"
                            async with self.session.get(result_url, headers=headers) as result_response:
                                if result_response.status == 200:
                                    content = await result_response.read()
                                    self._update_activity()
                                    return content
                                else:
                                    raise ExportError("Failed to download rendered content")
                        elif task_status['status'] == 'failure':
                            raise ExportError(f"Render failed: {task_status.get('status_detail', 'Unknown error')}")
            
            raise ExportError("Export timeout - render task did not complete in time")
        
        except Exception as e:
            logger.error(f"Error exporting Looker dashboard: {str(e)}")
            raise ExportError(f"Failed to export dashboard: {str(e)}")
    
    async def create_embed_token(
        self,
        dashboard_id: str,
        user_id: str,
        embed_type: EmbedType = EmbedType.IFRAME,
        permissions: Optional[List[str]] = None,
        expiry_minutes: int = 60
    ) -> EmbedTokenResponse:
        """Create embed token for Looker dashboard"""
        try:
            if not self.embed_secret:
                raise EmbedError("Embed secret required for Looker embedding")
            
            # Create SSO embed URL
            import hmac
            import hashlib
            import urllib.parse
            import time
            
            # Embed parameters
            embed_params = {
                'external_user_id': user_id,
                'models': permissions or ['*'],  # Models the user can access
                'permissions': ['access_data', 'see_looks', 'see_user_dashboards'],
                'session_length': expiry_minutes * 60,
                'force_logout_login': True
            }
            
            # Create the embed URL
            embed_path = f"/embed/dashboards/{dashboard_id}"
            
            # Generate signature
            string_to_sign = f"{embed_path}\n{json.dumps(embed_params, sort_keys=True)}"
            signature = hmac.new(
                self.embed_secret.encode('utf-8'),
                string_to_sign.encode('utf-8'),
                hashlib.sha1
            ).hexdigest()
            
            # Build final URL
            query_params = {
                'embed_domain': self.base_url,
                'signature': signature,
                **embed_params
            }
            
            embed_url = f"{self.base_url}{embed_path}?{urllib.parse.urlencode(query_params)}"
            expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
            
            # Generate embed code
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
            elif embed_type == EmbedType.JAVASCRIPT:
                embed_code = f'''
                <div id="looker-dashboard"></div>
                <script src="https://cdn.jsdelivr.net/npm/@looker/embed-sdk@1.8.0/lib/embed.js"></script>
                <script>
                    LookerEmbedSDK.init('{self.base_url}');
                    
                    LookerEmbedSDK.createDashboardWithId({dashboard_id})
                        .appendTo('#looker-dashboard')
                        .build()
                        .connect()
                        .then(dashboard => {{
                            console.log('Looker dashboard loaded');
                        }})
                        .catch(error => {{
                            console.error('Error loading dashboard:', error);
                        }});
                </script>
                '''
            
            return EmbedTokenResponse(
                token=signature,  # Using signature as token
                embed_url=embed_url,
                expires_at=expires_at,
                embed_code=embed_code
            )
        
        except Exception as e:
            logger.error(f"Error creating Looker embed token: {str(e)}")
            raise EmbedError(f"Failed to create embed token: {str(e)}")
    
    async def sync_data_source(
        self,
        data_source_id: str,
        incremental: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> DataSyncResult:
        """Refresh Looker connection/model"""
        await self._ensure_authenticated()
        
        try:
            start_time = datetime.utcnow()
            
            # Test the connection
            connection_url = f"{self.api_base}/connections/{data_source_id}/test"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.put(connection_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return DataSyncResult(
                        success=False,
                        records_processed=0,
                        records_updated=0,
                        records_created=0,
                        errors=[f"Connection test failed: {error_text}"],
                        sync_duration=0,
                        last_sync_time=start_time
                    )
                
                test_result = await response.json()
                
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                return DataSyncResult(
                    success=test_result.get('status') == 'success',
                    records_processed=1,  # Placeholder - Looker doesn't provide record counts
                    records_updated=1,
                    records_created=0,
                    errors=[test_result.get('message')] if test_result.get('status') != 'success' else [],
                    sync_duration=duration,
                    last_sync_time=end_time
                )
        
        except Exception as e:
            logger.error(f"Error syncing Looker connection: {str(e)}")
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
        """Create new connection in Looker"""
        await self._ensure_authenticated()
        
        try:
            # Parse connection string to extract database details
            # This is a simplified implementation
            connection_config = {
                "name": name,
                "dialect": {
                    "name": source_type.lower()
                },
                "host": config.get('host', 'localhost') if config else 'localhost',
                "port": config.get('port', '5432') if config else '5432',
                "database": config.get('database', name) if config else name,
                "username": config.get('username', '') if config else '',
                "password": config.get('password', '') if config else '',
                "ssl": config.get('ssl', False) if config else False
            }
            
            create_url = f"{self.api_base}/connections"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.post(create_url, json=connection_config, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to create connection: {error_text}")
                
                connection = await response.json()
                return connection['name']  # Looker uses name as ID
        
        except Exception as e:
            logger.error(f"Error creating Looker connection: {str(e)}")
            raise ConnectionError(f"Failed to create connection: {str(e)}")
    
    async def update_data_source(
        self,
        data_source_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """Update Looker connection configuration"""
        await self._ensure_authenticated()
        
        try:
            update_url = f"{self.api_base}/connections/{data_source_id}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.patch(update_url, json=config, headers=headers) as response:
                return response.status == 200
        
        except Exception as e:
            logger.error(f"Error updating Looker connection: {str(e)}")
            return False
    
    async def delete_data_source(self, data_source_id: str) -> bool:
        """Delete Looker connection"""
        await self._ensure_authenticated()
        
        try:
            delete_url = f"{self.api_base}/connections/{data_source_id}"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.delete(delete_url, headers=headers) as response:
                return response.status == 204
        
        except Exception as e:
            logger.error(f"Error deleting Looker connection: {str(e)}")
            return False
    
    async def get_real_time_data_feed(
        self,
        data_source_id: str,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set up real-time data feed for Looker"""
        # Looker doesn't have native real-time streaming like Power BI
        # But it can work with real-time databases and refresh frequently
        return {
            "supported": False,
            "message": "Looker works with real-time databases but doesn't have native streaming APIs",
            "alternatives": [
                "Use real-time database connections (e.g., BigQuery streaming)",
                "Set up frequent scheduled refreshes",
                "Use Looker's persistent derived tables (PDTs) with short rebuild times"
            ]
        }
    
    def get_supported_export_formats(self) -> List[ExportFormat]:
        """Return supported export formats for Looker"""
        return [
            ExportFormat.PDF,
            ExportFormat.PNG,
            ExportFormat.JPEG,
            ExportFormat.CSV
        ]
    
    def get_supported_embed_types(self) -> List[EmbedType]:
        """Return supported embed types for Looker"""
        return [
            EmbedType.IFRAME,
            EmbedType.JAVASCRIPT
        ]
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up Looker session"""
        if self.session and self.access_token:
            try:
                # Logout
                logout_url = f"{self.api_base}/logout"
                headers = {'Authorization': f'Bearer {self.access_token}'}
                await self.session.delete(logout_url, headers=headers)
            except Exception:
                pass  # Ignore logout errors
            
            await self.session.close()
            self.session = None
            self.access_token = None
            self._authenticated = False