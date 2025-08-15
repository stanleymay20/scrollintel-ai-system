"""
Multi-source data integration framework for the Advanced Analytics Dashboard.
Provides unified interface for connecting to various enterprise data sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Supported data source types"""
    ERP = "erp"
    CRM = "crm"
    BI_TOOL = "bi_tool"
    CLOUD_PLATFORM = "cloud_platform"
    DATABASE = "database"
    API = "api"


class ConnectionStatus(Enum):
    """Connection status states"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"


@dataclass
class DataSourceConfig:
    """Configuration for data source connections"""
    source_id: str
    source_type: DataSourceType
    name: str
    connection_params: Dict[str, Any]
    refresh_interval: int = 300  # seconds
    timeout: int = 30
    retry_attempts: int = 3
    enabled: bool = True


@dataclass
class DataRecord:
    """Standardized data record format"""
    source_id: str
    record_id: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ConnectionHealth:
    """Health status of a data connection"""
    source_id: str
    status: ConnectionStatus
    last_successful_sync: Optional[datetime]
    error_message: Optional[str]
    latency_ms: Optional[float]
    records_synced: int


class BaseDataConnector(ABC):
    """Abstract base class for all data connectors"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.last_sync = None
        self.error_message = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the data source"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if connection is working"""
        pass
    
    @abstractmethod
    async def fetch_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Fetch data from the source"""
        pass
    
    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """Get data schema/structure from the source"""
        pass
    
    def get_health(self) -> ConnectionHealth:
        """Get current health status"""
        return ConnectionHealth(
            source_id=self.config.source_id,
            status=self.status,
            last_successful_sync=self.last_sync,
            error_message=self.error_message,
            latency_ms=None,
            records_synced=0
        )


class DataConnectorRegistry:
    """Registry for managing data connectors"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseDataConnector] = {}
        self.connector_classes: Dict[str, type] = {}  # Use specific connector names instead of types
    
    def register_connector_class(self, connector_name: str, connector_class: type):
        """Register a connector class with a specific name"""
        self.connector_classes[connector_name] = connector_class
        logger.info(f"Registered connector class: {connector_name}")
    
    def create_connector(self, config: DataSourceConfig) -> BaseDataConnector:
        """Create a new connector instance based on connection parameters"""
        # Determine connector type from config
        connector_name = self._determine_connector_type(config)
        
        if connector_name not in self.connector_classes:
            raise ValueError(f"No connector class registered for {connector_name}")
        
        connector_class = self.connector_classes[connector_name]
        connector = connector_class(config)
        self.connectors[config.source_id] = connector
        
        logger.info(f"Created {connector_name} connector for {config.source_id}")
        return connector
    
    def _determine_connector_type(self, config: DataSourceConfig) -> str:
        """Determine the specific connector type from configuration"""
        params = config.connection_params
        
        # ERP Systems
        if config.source_type == DataSourceType.ERP:
            if 'client' in params and 'host' in params:
                return 'sap'
            elif 'base_url' in params and 'oracle' in params.get('base_url', '').lower():
                return 'oracle_erp'
            elif 'org_url' in params and 'dynamics' in params.get('org_url', '').lower():
                return 'microsoft_dynamics'
            else:
                return 'sap'  # Default ERP connector
        
        # CRM Systems
        elif config.source_type == DataSourceType.CRM:
            if 'instance_url' in params and 'salesforce' in params.get('instance_url', '').lower():
                return 'salesforce'
            elif 'access_token' in params or 'api_key' in params:
                return 'hubspot'
            elif 'org_url' in params and 'crm' in params.get('org_url', '').lower():
                return 'microsoft_crm'
            else:
                return 'salesforce'  # Default CRM connector
        
        # BI Tools
        elif config.source_type == DataSourceType.BI_TOOL:
            if 'server_url' in params and 'tableau' in params.get('server_url', '').lower():
                return 'tableau'
            elif 'tenant_id' in params and 'client_id' in params:
                return 'powerbi'
            elif 'base_url' in params and 'looker' in params.get('base_url', '').lower():
                return 'looker'
            elif 'virtual_proxy' in params or 'qlik' in str(params).lower():
                return 'qlik'
            else:
                return 'tableau'  # Default BI connector
        
        # Cloud Platforms
        elif config.source_type == DataSourceType.CLOUD_PLATFORM:
            if 'access_key_id' in params and 'secret_access_key' in params:
                return 'aws'
            elif 'subscription_id' in params and 'tenant_id' in params:
                return 'azure'
            elif 'service_account_key' in params and 'project_id' in params:
                return 'gcp'
            else:
                return 'aws'  # Default cloud connector
        
        # Default fallback
        return 'sap'
    
    def get_connector(self, source_id: str) -> Optional[BaseDataConnector]:
        """Get connector by source ID"""
        return self.connectors.get(source_id)
    
    def remove_connector(self, source_id: str) -> bool:
        """Remove connector from registry"""
        if source_id in self.connectors:
            del self.connectors[source_id]
            logger.info(f"Removed connector {source_id}")
            return True
        return False
    
    def list_connectors(self) -> List[str]:
        """List all registered connector IDs"""
        return list(self.connectors.keys())


class DataIntegrationManager:
    """Main manager for data integration operations"""
    
    def __init__(self):
        self.registry = DataConnectorRegistry()
        self.active_syncs: Dict[str, asyncio.Task] = {}
    
    def register_connector_class(self, connector_name: str, connector_class: type):
        """Register a connector class"""
        self.registry.register_connector_class(connector_name, connector_class)
    
    async def add_data_source(self, config: DataSourceConfig) -> bool:
        """Add and connect to a new data source"""
        try:
            connector = self.registry.create_connector(config)
            success = await connector.connect()
            
            if success and config.enabled:
                # Start periodic sync if enabled
                await self._start_periodic_sync(config.source_id)
            
            return success
        except Exception as e:
            logger.error(f"Failed to add data source {config.source_id}: {e}")
            return False
    
    async def remove_data_source(self, source_id: str) -> bool:
        """Remove a data source"""
        try:
            # Stop periodic sync
            await self._stop_periodic_sync(source_id)
            
            # Disconnect and remove connector
            connector = self.registry.get_connector(source_id)
            if connector:
                await connector.disconnect()
            
            return self.registry.remove_connector(source_id)
        except Exception as e:
            logger.error(f"Failed to remove data source {source_id}: {e}")
            return False
    
    async def sync_data_source(self, source_id: str, query: Optional[Dict[str, Any]] = None) -> List[DataRecord]:
        """Manually sync data from a specific source"""
        connector = self.registry.get_connector(source_id)
        if not connector:
            raise ValueError(f"Data source {source_id} not found")
        
        if query is None:
            query = {}
        
        try:
            data = await connector.fetch_data(query)
            connector.last_sync = datetime.utcnow()
            connector.status = ConnectionStatus.CONNECTED
            connector.error_message = None
            
            logger.info(f"Synced {len(data)} records from {source_id}")
            return data
        except Exception as e:
            connector.status = ConnectionStatus.ERROR
            connector.error_message = str(e)
            logger.error(f"Failed to sync data from {source_id}: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, ConnectionHealth]:
        """Get health status of all data sources"""
        health_status = {}
        
        for source_id, connector in self.registry.connectors.items():
            health_status[source_id] = connector.get_health()
        
        return health_status
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test all data source connections"""
        results = {}
        
        for source_id, connector in self.registry.connectors.items():
            try:
                results[source_id] = await connector.test_connection()
            except Exception as e:
                logger.error(f"Connection test failed for {source_id}: {e}")
                results[source_id] = False
        
        return results
    
    async def _start_periodic_sync(self, source_id: str):
        """Start periodic data synchronization"""
        if source_id in self.active_syncs:
            return  # Already running
        
        connector = self.registry.get_connector(source_id)
        if not connector:
            return
        
        async def sync_loop():
            while source_id in self.active_syncs:
                try:
                    await self.sync_data_source(source_id)
                    await asyncio.sleep(connector.config.refresh_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Periodic sync error for {source_id}: {e}")
                    await asyncio.sleep(60)  # Wait before retry
        
        task = asyncio.create_task(sync_loop())
        self.active_syncs[source_id] = task
        logger.info(f"Started periodic sync for {source_id}")
    
    async def _stop_periodic_sync(self, source_id: str):
        """Stop periodic data synchronization"""
        if source_id in self.active_syncs:
            task = self.active_syncs[source_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.active_syncs[source_id]
            logger.info(f"Stopped periodic sync for {source_id}")


# Global instance
data_integration_manager = DataIntegrationManager()