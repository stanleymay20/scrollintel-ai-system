"""
Base BI Connector Framework with Plugin Architecture
Provides abstract base class for all BI tool integrations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from scrollintel.models.bi_integration_models import (
    BIToolType, ExportFormat, EmbedType, ConnectionStatus,
    BIDashboardInfo, DataSyncResult, EmbedTokenResponse
)

logger = logging.getLogger(__name__)


class BIConnectorError(Exception):
    """Base exception for BI connector errors"""
    pass


class AuthenticationError(BIConnectorError):
    """Authentication failed"""
    pass


class ConnectionError(BIConnectorError):
    """Connection failed"""
    pass


class ExportError(BIConnectorError):
    """Export operation failed"""
    pass


class EmbedError(BIConnectorError):
    """Embedding operation failed"""
    pass


class BaseBIConnector(ABC):
    """
    Abstract base class for BI tool connectors
    Defines the interface that all BI connectors must implement
    """
    
    def __init__(self, connection_config: Dict[str, Any]):
        """Initialize connector with configuration"""
        self.config = connection_config
        self.connection_id = connection_config.get('id')
        self.name = connection_config.get('name')
        self.bi_tool_type = connection_config.get('bi_tool_type')
        self._authenticated = False
        self._last_activity = None
    
    @property
    @abstractmethod
    def tool_type(self) -> BIToolType:
        """Return the BI tool type this connector supports"""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the BI tool
        Returns True if successful, raises AuthenticationError if failed
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the BI tool
        Returns connection status and health information
        """
        pass
    
    @abstractmethod
    async def get_dashboards(self, project_id: Optional[str] = None) -> List[BIDashboardInfo]:
        """
        Get list of available dashboards
        """
        pass
    
    @abstractmethod
    async def get_dashboard_info(self, dashboard_id: str) -> BIDashboardInfo:
        """
        Get detailed information about a specific dashboard
        """
        pass
    
    @abstractmethod
    async def export_dashboard(
        self, 
        dashboard_id: str, 
        format: ExportFormat,
        filters: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Export dashboard in specified format
        Returns the exported content as bytes
        """
        pass
    
    @abstractmethod
    async def create_embed_token(
        self,
        dashboard_id: str,
        user_id: str,
        embed_type: EmbedType = EmbedType.IFRAME,
        permissions: Optional[List[str]] = None,
        expiry_minutes: int = 60
    ) -> EmbedTokenResponse:
        """
        Create an embed token for dashboard embedding
        """
        pass
    
    @abstractmethod
    async def sync_data_source(
        self,
        data_source_id: str,
        incremental: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> DataSyncResult:
        """
        Synchronize data from external source to BI tool
        """
        pass
    
    @abstractmethod
    async def create_data_source(
        self,
        name: str,
        connection_string: str,
        source_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new data source in the BI tool
        Returns the data source ID
        """
        pass
    
    @abstractmethod
    async def update_data_source(
        self,
        data_source_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Update an existing data source configuration
        """
        pass
    
    @abstractmethod
    async def delete_data_source(self, data_source_id: str) -> bool:
        """
        Delete a data source from the BI tool
        """
        pass
    
    @abstractmethod
    async def get_real_time_data_feed(
        self,
        data_source_id: str,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set up real-time data feed for live dashboards
        """
        pass
    
    # Common utility methods
    async def validate_config(self) -> List[str]:
        """
        Validate the connector configuration
        Returns list of validation errors (empty if valid)
        """
        errors = []
        
        required_fields = self.get_required_config_fields()
        for field in required_fields:
            if field not in self.config or not self.config[field]:
                errors.append(f"Missing required field: {field}")
        
        return errors
    
    @abstractmethod
    def get_required_config_fields(self) -> List[str]:
        """
        Return list of required configuration fields for this connector
        """
        pass
    
    def get_supported_export_formats(self) -> List[ExportFormat]:
        """
        Return list of supported export formats
        Default implementation returns common formats
        """
        return [
            ExportFormat.PDF,
            ExportFormat.PNG,
            ExportFormat.CSV,
            ExportFormat.JSON
        ]
    
    def get_supported_embed_types(self) -> List[EmbedType]:
        """
        Return list of supported embed types
        Default implementation returns iframe
        """
        return [EmbedType.IFRAME]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the connector
        """
        try:
            connection_result = await self.test_connection()
            return {
                "status": "healthy",
                "authenticated": self._authenticated,
                "last_activity": self._last_activity,
                "connection_test": connection_result
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "authenticated": self._authenticated,
                "last_activity": self._last_activity
            }
    
    def _update_activity(self):
        """Update last activity timestamp"""
        self._last_activity = datetime.utcnow()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cleanup if needed
        pass


class BIConnectorRegistry:
    """
    Registry for BI connector plugins
    Manages available connectors and their instantiation
    """
    
    def __init__(self):
        self._connectors: Dict[BIToolType, type] = {}
    
    def register(self, tool_type: BIToolType, connector_class: type):
        """Register a connector class for a BI tool type"""
        if not issubclass(connector_class, BaseBIConnector):
            raise ValueError(f"Connector class must inherit from BaseBIConnector")
        
        self._connectors[tool_type] = connector_class
        logger.info(f"Registered BI connector for {tool_type}: {connector_class.__name__}")
    
    def get_connector(self, tool_type: BIToolType, config: Dict[str, Any]) -> BaseBIConnector:
        """Get a connector instance for the specified tool type"""
        if tool_type not in self._connectors:
            raise ValueError(f"No connector registered for BI tool type: {tool_type}")
        
        connector_class = self._connectors[tool_type]
        return connector_class(config)
    
    def get_available_tools(self) -> List[BIToolType]:
        """Get list of available BI tool types"""
        return list(self._connectors.keys())
    
    def is_supported(self, tool_type: BIToolType) -> bool:
        """Check if a BI tool type is supported"""
        return tool_type in self._connectors


# Global registry instance
bi_connector_registry = BIConnectorRegistry()


def register_bi_connector(tool_type: BIToolType):
    """
    Decorator to register a BI connector class
    
    Usage:
    @register_bi_connector(BIToolType.TABLEAU)
    class TableauConnector(BaseBIConnector):
        ...
    """
    def decorator(connector_class):
        bi_connector_registry.register(tool_type, connector_class)
        return connector_class
    return decorator