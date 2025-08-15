"""
Offline-first architecture implementation for ScrollIntel.
Provides seamless online/offline integration with intelligent data management.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import threading
from pathlib import Path

from .offline_data_manager import OfflineDataManager, DataOperation, SyncStatus
from .offline_sync_engine import IntelligentSyncEngine, get_sync_engine

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Network connection status."""
    ONLINE = "online"
    OFFLINE = "offline"
    UNSTABLE = "unstable"
    RECONNECTING = "reconnecting"


class OperationMode(Enum):
    """Application operation modes."""
    ONLINE_FIRST = "online_first"
    OFFLINE_FIRST = "offline_first"
    HYBRID = "hybrid"
    OFFLINE_ONLY = "offline_only"


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    latency_ms: float
    bandwidth_mbps: float
    packet_loss_percent: float
    connection_stability: float  # 0-1 scale
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class OfflineCapability:
    """Defines offline capabilities for a feature."""
    feature_name: str
    read_offline: bool = True
    write_offline: bool = True
    sync_priority: str = "normal"
    conflict_resolution: str = "intelligent_merge"
    cache_duration_hours: int = 24
    max_offline_entries: int = 1000
    requires_auth: bool = True


class OfflineFirstArchitecture:
    """Offline-first architecture manager for ScrollIntel."""
    
    def __init__(self, db_path: str = "data/offline_architecture.db"):
        self.db_path = db_path
        
        # Core components
        self.offline_manager = OfflineDataManager(db_path)
        self.sync_engine = get_sync_engine(self.offline_manager)
        
        # Connection management
        self.connection_status = ConnectionStatus.ONLINE
        self.operation_mode = OperationMode.OFFLINE_FIRST
        self.network_metrics = NetworkMetrics(0.0, 0.0, 0.0, 1.0)
        
        # Feature capabilities
        self.feature_capabilities: Dict[str, OfflineCapability] = {}
        self.offline_handlers: Dict[str, Callable] = {}
        self.online_handlers: Dict[str, Callable] = {}
        
        # Request routing
        self.request_queue: List[Dict[str, Any]] = []
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self.request_cache: Dict[str, Any] = {}
        
        # Background tasks
        self.connection_monitor_task: Optional[asyncio.Task] = None
        self.request_processor_task: Optional[asyncio.Task] = None
        self.cache_cleanup_task: Optional[asyncio.Task] = None
        
        # Callbacks and hooks
        self.connection_callbacks: List[Callable] = []
        self.mode_change_callbacks: List[Callable] = []
        self.sync_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_metrics = {
            'requests_served_offline': 0,
            'requests_served_online': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'sync_operations': 0,
            'failed_requests': 0,
            'avg_response_time_offline': 0.0,
            'avg_response_time_online': 0.0
        }
        
        # Initialize architecture
        self._setup_default_capabilities()
        self._start_background_tasks()
    
    def _setup_default_capabilities(self):
        """Setup default offline capabilities for core features."""
        
        # Data visualization
        self.register_feature_capability(OfflineCapability(
            feature_name="data_visualization",
            read_offline=True,
            write_offline=True,
            sync_priority="high",
            conflict_resolution="intelligent_merge",
            cache_duration_hours=48,
            max_offline_entries=500
        ))
        
        # File processing
        self.register_feature_capability(OfflineCapability(
            feature_name="file_processing",
            read_offline=True,
            write_offline=True,
            sync_priority="normal",
            conflict_resolution="timestamp_based",
            cache_duration_hours=24,
            max_offline_entries=100
        ))
        
        # User preferences
        self.register_feature_capability(OfflineCapability(
            feature_name="user_preferences",
            read_offline=True,
            write_offline=True,
            sync_priority="high",
            conflict_resolution="client_wins",
            cache_duration_hours=168,  # 1 week
            max_offline_entries=50
        ))
        
        # Analytics data
        self.register_feature_capability(OfflineCapability(
            feature_name="analytics",
            read_offline=True,
            write_offline=False,  # Analytics are read-only offline
            sync_priority="low",
            conflict_resolution="server_wins",
            cache_duration_hours=12,
            max_offline_entries=1000
        ))
        
        # AI interactions
        self.register_feature_capability(OfflineCapability(
            feature_name="ai_interactions",
            read_offline=True,
            write_offline=True,
            sync_priority="normal",
            conflict_resolution="intelligent_merge",
            cache_duration_hours=6,
            max_offline_entries=200
        ))
        
        # Dashboard configurations
        self.register_feature_capability(OfflineCapability(
            feature_name="dashboard_config",
            read_offline=True,
            write_offline=True,
            sync_priority="high",
            conflict_resolution="intelligent_merge",
            cache_duration_hours=72,
            max_offline_entries=100
        ))
    
    def _start_background_tasks(self):
        """Start background monitoring and processing tasks."""
        self.connection_monitor_task = asyncio.create_task(self._monitor_connection())
        self.request_processor_task = asyncio.create_task(self._process_request_queue())
        self.cache_cleanup_task = asyncio.create_task(self._cleanup_cache())
    
    async def _monitor_connection(self):
        """Monitor network connection status."""
        while True:
            try:
                # Test connection (simplified implementation)
                previous_status = self.connection_status
                new_status = await self._test_connection()
                
                if new_status != previous_status:
                    await self._handle_connection_change(previous_status, new_status)
                
                # Update network metrics
                await self._update_network_metrics()
                
                # Adjust operation mode based on connection quality
                await self._adjust_operation_mode()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _test_connection(self) -> ConnectionStatus:
        """Test network connection and return status."""
        try:
            # Simulate connection test (replace with actual implementation)
            import random
            
            # Simulate occasional offline periods
            if random.random() < 0.05:  # 5% chance of being offline
                return ConnectionStatus.OFFLINE
            
            # Simulate unstable connection
            if random.random() < 0.1:  # 10% chance of unstable
                return ConnectionStatus.UNSTABLE
            
            return ConnectionStatus.ONLINE
            
        except Exception:
            return ConnectionStatus.OFFLINE
    
    async def _update_network_metrics(self):
        """Update network performance metrics."""
        try:
            # Simulate network metrics (replace with actual measurement)
            import random
            
            if self.connection_status == ConnectionStatus.ONLINE:
                self.network_metrics.latency_ms = random.uniform(10, 100)
                self.network_metrics.bandwidth_mbps = random.uniform(1, 100)
                self.network_metrics.packet_loss_percent = random.uniform(0, 2)
                self.network_metrics.connection_stability = random.uniform(0.8, 1.0)
            elif self.connection_status == ConnectionStatus.UNSTABLE:
                self.network_metrics.latency_ms = random.uniform(100, 1000)
                self.network_metrics.bandwidth_mbps = random.uniform(0.1, 5)
                self.network_metrics.packet_loss_percent = random.uniform(2, 10)
                self.network_metrics.connection_stability = random.uniform(0.3, 0.7)
            else:  # OFFLINE
                self.network_metrics.latency_ms = float('inf')
                self.network_metrics.bandwidth_mbps = 0.0
                self.network_metrics.packet_loss_percent = 100.0
                self.network_metrics.connection_stability = 0.0
            
            self.network_metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Network metrics update error: {e}")
    
    async def _adjust_operation_mode(self):
        """Adjust operation mode based on connection quality."""
        if self.connection_status == ConnectionStatus.OFFLINE:
            new_mode = OperationMode.OFFLINE_ONLY
        elif self.connection_status == ConnectionStatus.UNSTABLE:
            new_mode = OperationMode.OFFLINE_FIRST
        elif self.network_metrics.latency_ms > 500 or self.network_metrics.connection_stability < 0.7:
            new_mode = OperationMode.OFFLINE_FIRST
        else:
            new_mode = OperationMode.HYBRID
        
        if new_mode != self.operation_mode:
            await self._change_operation_mode(new_mode)
    
    async def _handle_connection_change(self, previous: ConnectionStatus, current: ConnectionStatus):
        """Handle connection status changes."""
        self.connection_status = current
        
        logger.info(f"Connection status changed: {previous.value} -> {current.value}")
        
        # Notify callbacks
        for callback in self.connection_callbacks:
            try:
                await callback(previous, current)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
        
        # Handle specific transitions
        if previous == ConnectionStatus.OFFLINE and current in [ConnectionStatus.ONLINE, ConnectionStatus.UNSTABLE]:
            # Coming back online - trigger sync
            await self._trigger_reconnection_sync()
        elif current == ConnectionStatus.OFFLINE:
            # Going offline - ensure offline mode is ready
            await self._prepare_offline_mode()
    
    async def _change_operation_mode(self, new_mode: OperationMode):
        """Change operation mode."""
        previous_mode = self.operation_mode
        self.operation_mode = new_mode
        
        logger.info(f"Operation mode changed: {previous_mode.value} -> {new_mode.value}")
        
        # Notify callbacks
        for callback in self.mode_change_callbacks:
            try:
                await callback(previous_mode, new_mode)
            except Exception as e:
                logger.error(f"Mode change callback error: {e}")
    
    async def _trigger_reconnection_sync(self):
        """Trigger synchronization when reconnecting."""
        try:
            logger.info("Triggering reconnection sync...")
            
            # Force sync of pending data
            sync_result = await self.offline_manager.force_sync()
            
            logger.info(f"Reconnection sync completed: {sync_result}")
            
            # Notify sync callbacks
            for callback in self.sync_callbacks:
                try:
                    await callback("reconnection_sync", sync_result)
                except Exception as e:
                    logger.error(f"Sync callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Reconnection sync failed: {e}")
    
    async def _prepare_offline_mode(self):
        """Prepare for offline operation."""
        try:
            logger.info("Preparing offline mode...")
            
            # Ensure critical data is cached
            await self._cache_critical_data()
            
            # Update offline manager status
            self.offline_manager.set_online_status(False)
            
            logger.info("Offline mode prepared")
            
        except Exception as e:
            logger.error(f"Offline mode preparation failed: {e}")
    
    async def _cache_critical_data(self):
        """Cache critical data for offline operation."""
        try:
            # Cache user preferences
            await self._cache_feature_data("user_preferences")
            
            # Cache dashboard configurations
            await self._cache_feature_data("dashboard_config")
            
            # Cache recent analytics data
            await self._cache_feature_data("analytics")
            
        except Exception as e:
            logger.error(f"Critical data caching failed: {e}")
    
    async def _cache_feature_data(self, feature_name: str):
        """Cache data for a specific feature."""
        capability = self.feature_capabilities.get(feature_name)
        if not capability:
            return
        
        try:
            # This would fetch and cache recent data for the feature
            # For now, just log the action
            logger.debug(f"Caching data for feature: {feature_name}")
            
        except Exception as e:
            logger.error(f"Feature data caching failed for {feature_name}: {e}")
    
    async def _process_request_queue(self):
        """Process queued requests."""
        while True:
            try:
                if self.request_queue:
                    # Process up to 10 requests at a time
                    requests_to_process = self.request_queue[:10]
                    self.request_queue = self.request_queue[10:]
                    
                    for request in requests_to_process:
                        await self._process_queued_request(request)
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Request queue processing error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _process_queued_request(self, request: Dict[str, Any]):
        """Process a single queued request."""
        try:
            request_id = request.get('id')
            feature_name = request.get('feature')
            operation = request.get('operation')
            data = request.get('data')
            
            # Execute the request
            result = await self._execute_request(feature_name, operation, data)
            
            # Store result if request is still pending
            if request_id in self.pending_requests:
                self.pending_requests[request_id]['result'] = result
                self.pending_requests[request_id]['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Queued request processing failed: {e}")
            
            request_id = request.get('id')
            if request_id in self.pending_requests:
                self.pending_requests[request_id]['error'] = str(e)
                self.pending_requests[request_id]['status'] = 'failed'
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up request cache
                current_time = time.time()
                expired_keys = [
                    key for key, value in self.request_cache.items()
                    if current_time - value.get('timestamp', 0) > 3600  # 1 hour expiry
                ]
                
                for key in expired_keys:
                    del self.request_cache[key]
                
                # Clean up pending requests
                expired_requests = [
                    req_id for req_id, req_data in self.pending_requests.items()
                    if current_time - req_data.get('created_at', 0) > 300  # 5 minute expiry
                ]
                
                for req_id in expired_requests:
                    del self.pending_requests[req_id]
                
                # Clean up offline storage
                await self.offline_manager.cleanup_storage(days_old=7)
                
                logger.debug(f"Cache cleanup completed: {len(expired_keys)} cache entries, {len(expired_requests)} pending requests")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    # Public API methods
    def register_feature_capability(self, capability: OfflineCapability):
        """Register offline capability for a feature."""
        self.feature_capabilities[capability.feature_name] = capability
        logger.info(f"Registered offline capability for feature: {capability.feature_name}")
    
    def register_offline_handler(self, feature_name: str, handler: Callable):
        """Register offline handler for a feature."""
        self.offline_handlers[feature_name] = handler
        logger.info(f"Registered offline handler for feature: {feature_name}")
    
    def register_online_handler(self, feature_name: str, handler: Callable):
        """Register online handler for a feature."""
        self.online_handlers[feature_name] = handler
        logger.info(f"Registered online handler for feature: {feature_name}")
    
    async def execute_request(self, feature_name: str, operation: str, 
                            data: Optional[Dict[str, Any]] = None,
                            user_id: Optional[str] = None) -> Any:
        """Execute a request with offline-first architecture."""
        start_time = time.time()
        
        try:
            # Check if feature has offline capability
            capability = self.feature_capabilities.get(feature_name)
            if not capability:
                # No offline capability - try online only
                return await self._execute_online_request(feature_name, operation, data, user_id)
            
            # Determine execution strategy based on operation mode and capability
            if self.operation_mode == OperationMode.OFFLINE_ONLY:
                result = await self._execute_offline_request(feature_name, operation, data, user_id, capability)
            elif self.operation_mode == OperationMode.OFFLINE_FIRST:
                result = await self._execute_offline_first_request(feature_name, operation, data, user_id, capability)
            elif self.operation_mode == OperationMode.ONLINE_FIRST:
                result = await self._execute_online_first_request(feature_name, operation, data, user_id, capability)
            else:  # HYBRID
                result = await self._execute_hybrid_request(feature_name, operation, data, user_id, capability)
            
            # Update performance metrics
            response_time = time.time() - start_time
            if self.connection_status == ConnectionStatus.OFFLINE:
                self.performance_metrics['requests_served_offline'] += 1
                self._update_avg_response_time('offline', response_time)
            else:
                self.performance_metrics['requests_served_online'] += 1
                self._update_avg_response_time('online', response_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Request execution failed for {feature_name}.{operation}: {e}")
            self.performance_metrics['failed_requests'] += 1
            
            # Try fallback execution
            return await self._execute_fallback_request(feature_name, operation, data, user_id)
    
    async def _execute_offline_request(self, feature_name: str, operation: str, 
                                     data: Optional[Dict[str, Any]], user_id: Optional[str],
                                     capability: OfflineCapability) -> Any:
        """Execute request in offline mode."""
        # Check if operation is supported offline
        if operation in ['create', 'update', 'delete'] and not capability.write_offline:
            raise Exception(f"Write operations not supported offline for {feature_name}")
        
        if operation == 'read' and not capability.read_offline:
            raise Exception(f"Read operations not supported offline for {feature_name}")
        
        # Use offline handler if available
        offline_handler = self.offline_handlers.get(feature_name)
        if offline_handler:
            return await offline_handler(operation, data, user_id)
        
        # Use offline data manager
        if operation == 'create':
            entry_id = await self.offline_manager.store_data(
                feature_name, data, DataOperation.CREATE, user_id
            )
            return {'id': entry_id, 'status': 'created_offline'}
        
        elif operation == 'read':
            if data and 'id' in data:
                result = await self.offline_manager.get_data(feature_name, data['id'])
                return result
            else:
                results = await self.offline_manager.query_data(
                    feature_name, data, limit=capability.max_offline_entries
                )
                return results
        
        elif operation == 'update':
            if not data or 'id' not in data:
                raise Exception("Update operation requires ID")
            
            success = await self.offline_manager.update_data(
                feature_name, data['id'], data, user_id
            )
            return {'success': success, 'status': 'updated_offline'}
        
        elif operation == 'delete':
            if not data or 'id' not in data:
                raise Exception("Delete operation requires ID")
            
            success = await self.offline_manager.delete_data(
                feature_name, data['id'], user_id
            )
            return {'success': success, 'status': 'deleted_offline'}
        
        else:
            raise Exception(f"Unsupported operation: {operation}")
    
    async def _execute_online_request(self, feature_name: str, operation: str,
                                    data: Optional[Dict[str, Any]], user_id: Optional[str]) -> Any:
        """Execute request in online mode."""
        online_handler = self.online_handlers.get(feature_name)
        if online_handler:
            return await online_handler(operation, data, user_id)
        
        # Simulate online API call
        await asyncio.sleep(0.1)  # Simulate network delay
        return {
            'feature': feature_name,
            'operation': operation,
            'data': data,
            'user_id': user_id,
            'status': 'completed_online',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_offline_first_request(self, feature_name: str, operation: str,
                                           data: Optional[Dict[str, Any]], user_id: Optional[str],
                                           capability: OfflineCapability) -> Any:
        """Execute request with offline-first strategy."""
        try:
            # Try offline first
            return await self._execute_offline_request(feature_name, operation, data, user_id, capability)
        except Exception as offline_error:
            # Fall back to online if available
            if self.connection_status != ConnectionStatus.OFFLINE:
                try:
                    return await self._execute_online_request(feature_name, operation, data, user_id)
                except Exception as online_error:
                    logger.error(f"Both offline and online execution failed: offline={offline_error}, online={online_error}")
                    raise offline_error
            else:
                raise offline_error
    
    async def _execute_online_first_request(self, feature_name: str, operation: str,
                                          data: Optional[Dict[str, Any]], user_id: Optional[str],
                                          capability: OfflineCapability) -> Any:
        """Execute request with online-first strategy."""
        if self.connection_status != ConnectionStatus.OFFLINE:
            try:
                # Try online first
                result = await self._execute_online_request(feature_name, operation, data, user_id)
                
                # Cache result for offline use
                if operation == 'read' and result:
                    await self._cache_request_result(feature_name, operation, data, result)
                
                return result
            except Exception as online_error:
                # Fall back to offline
                try:
                    return await self._execute_offline_request(feature_name, operation, data, user_id, capability)
                except Exception as offline_error:
                    logger.error(f"Both online and offline execution failed: online={online_error}, offline={offline_error}")
                    raise online_error
        else:
            # Offline only
            return await self._execute_offline_request(feature_name, operation, data, user_id, capability)
    
    async def _execute_hybrid_request(self, feature_name: str, operation: str,
                                    data: Optional[Dict[str, Any]], user_id: Optional[str],
                                    capability: OfflineCapability) -> Any:
        """Execute request with hybrid strategy."""
        # For read operations, check cache first
        if operation == 'read':
            cache_key = self._generate_cache_key(feature_name, operation, data)
            cached_result = self.request_cache.get(cache_key)
            
            if cached_result and time.time() - cached_result['timestamp'] < capability.cache_duration_hours * 3600:
                self.performance_metrics['cache_hits'] += 1
                return cached_result['data']
            else:
                self.performance_metrics['cache_misses'] += 1
        
        # Decide based on connection quality and operation type
        if self.connection_status == ConnectionStatus.ONLINE and self.network_metrics.latency_ms < 200:
            # Good connection - try online first
            return await self._execute_online_first_request(feature_name, operation, data, user_id, capability)
        else:
            # Poor connection or offline - try offline first
            return await self._execute_offline_first_request(feature_name, operation, data, user_id, capability)
    
    async def _execute_fallback_request(self, feature_name: str, operation: str,
                                      data: Optional[Dict[str, Any]], user_id: Optional[str]) -> Any:
        """Execute fallback request when all else fails."""
        # Return cached data if available
        cache_key = self._generate_cache_key(feature_name, operation, data)
        cached_result = self.request_cache.get(cache_key)
        
        if cached_result:
            logger.warning(f"Using stale cached data for {feature_name}.{operation}")
            return {
                **cached_result['data'],
                '_fallback': True,
                '_stale': True,
                '_cached_at': cached_result['timestamp']
            }
        
        # Return empty/default response
        return {
            'error': 'Service temporarily unavailable',
            'feature': feature_name,
            'operation': operation,
            'fallback': True,
            'message': 'Please try again when connection is restored'
        }
    
    async def _cache_request_result(self, feature_name: str, operation: str,
                                  request_data: Optional[Dict[str, Any]], result: Any):
        """Cache request result for offline use."""
        try:
            cache_key = self._generate_cache_key(feature_name, operation, request_data)
            self.request_cache[cache_key] = {
                'data': result,
                'timestamp': time.time(),
                'feature': feature_name,
                'operation': operation
            }
            
            # Also store in offline manager for persistence
            if operation == 'read' and isinstance(result, dict):
                await self.offline_manager.store_data(
                    f"{feature_name}_cache",
                    {
                        'cache_key': cache_key,
                        'data': result,
                        'cached_at': datetime.now().isoformat()
                    },
                    DataOperation.CREATE
                )
            
        except Exception as e:
            logger.error(f"Request result caching failed: {e}")
    
    def _generate_cache_key(self, feature_name: str, operation: str, 
                          data: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for request."""
        key_data = {
            'feature': feature_name,
            'operation': operation,
            'data': data
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _update_avg_response_time(self, mode: str, response_time: float):
        """Update average response time metrics."""
        metric_key = f'avg_response_time_{mode}'
        count_key = f'requests_served_{mode}'
        
        current_avg = self.performance_metrics[metric_key]
        current_count = self.performance_metrics[count_key]
        
        # Calculate new average
        new_avg = ((current_avg * (current_count - 1)) + response_time) / current_count
        self.performance_metrics[metric_key] = new_avg
    
    # Callback registration methods
    def register_connection_callback(self, callback: Callable):
        """Register callback for connection status changes."""
        self.connection_callbacks.append(callback)
    
    def register_mode_change_callback(self, callback: Callable):
        """Register callback for operation mode changes."""
        self.mode_change_callbacks.append(callback)
    
    def register_sync_callback(self, callback: Callable):
        """Register callback for sync events."""
        self.sync_callbacks.append(callback)
    
    # Status and metrics methods
    def get_architecture_status(self) -> Dict[str, Any]:
        """Get current architecture status."""
        return {
            'connection_status': self.connection_status.value,
            'operation_mode': self.operation_mode.value,
            'network_metrics': {
                'latency_ms': self.network_metrics.latency_ms,
                'bandwidth_mbps': self.network_metrics.bandwidth_mbps,
                'packet_loss_percent': self.network_metrics.packet_loss_percent,
                'connection_stability': self.network_metrics.connection_stability,
                'last_updated': self.network_metrics.last_updated.isoformat()
            },
            'offline_manager_status': self.offline_manager.get_sync_status(),
            'performance_metrics': self.performance_metrics.copy(),
            'feature_capabilities': {
                name: {
                    'read_offline': cap.read_offline,
                    'write_offline': cap.write_offline,
                    'sync_priority': cap.sync_priority,
                    'cache_duration_hours': cap.cache_duration_hours
                }
                for name, cap in self.feature_capabilities.items()
            },
            'cache_info': {
                'request_cache_size': len(self.request_cache),
                'pending_requests': len(self.pending_requests),
                'queued_requests': len(self.request_queue)
            }
        }
    
    def get_offline_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get offline capabilities for all features."""
        return {
            name: {
                'read_offline': cap.read_offline,
                'write_offline': cap.write_offline,
                'sync_priority': cap.sync_priority,
                'conflict_resolution': cap.conflict_resolution,
                'cache_duration_hours': cap.cache_duration_hours,
                'max_offline_entries': cap.max_offline_entries,
                'requires_auth': cap.requires_auth
            }
            for name, cap in self.feature_capabilities.items()
        }
    
    async def force_sync_all(self) -> Dict[str, Any]:
        """Force synchronization of all offline data."""
        return await self.offline_manager.force_sync()
    
    async def get_sync_conflicts(self) -> List[Dict[str, Any]]:
        """Get list of sync conflicts."""
        return self.offline_manager.get_conflicts()
    
    async def resolve_sync_conflict(self, conflict_id: str, resolution: str,
                                  custom_data: Optional[Dict[str, Any]] = None) -> bool:
        """Resolve a sync conflict."""
        from .offline_data_manager import ConflictResolutionStrategy
        
        try:
            strategy = ConflictResolutionStrategy(resolution)
            return await self.offline_manager.resolve_conflict(conflict_id, strategy, custom_data)
        except ValueError:
            logger.error(f"Invalid resolution strategy: {resolution}")
            return False
    
    def set_operation_mode(self, mode: OperationMode):
        """Manually set operation mode."""
        asyncio.create_task(self._change_operation_mode(mode))
    
    def close(self):
        """Close the offline-first architecture."""
        # Cancel background tasks
        if self.connection_monitor_task:
            self.connection_monitor_task.cancel()
        if self.request_processor_task:
            self.request_processor_task.cancel()
        if self.cache_cleanup_task:
            self.cache_cleanup_task.cancel()
        
        # Close offline manager
        self.offline_manager.close()
        
        logger.info("Offline-first architecture closed")


# Global instance
offline_first_architecture = OfflineFirstArchitecture()


# Convenience decorators and context managers
def offline_capable(feature_name: str, read_offline: bool = True, write_offline: bool = True):
    """Decorator to mark a function as offline-capable."""
    def decorator(func):
        # Register the function as an offline handler
        offline_first_architecture.register_offline_handler(feature_name, func)
        return func
    return decorator


@asynccontextmanager
async def offline_context(feature_name: str):
    """Context manager for offline operations."""
    try:
        # Ensure offline capability is available
        capability = offline_first_architecture.feature_capabilities.get(feature_name)
        if not capability:
            raise Exception(f"No offline capability registered for {feature_name}")
        
        yield offline_first_architecture
        
    except Exception as e:
        logger.error(f"Offline context error for {feature_name}: {e}")
        raise


async def execute_offline_first(feature_name: str, operation: str, 
                              data: Optional[Dict[str, Any]] = None,
                              user_id: Optional[str] = None) -> Any:
    """Execute a request with offline-first architecture."""
    return await offline_first_architecture.execute_request(feature_name, operation, data, user_id)