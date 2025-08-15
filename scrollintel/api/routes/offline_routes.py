"""
API routes for offline capabilities management.
Provides endpoints for offline data management, sync operations, and PWA functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import logging

from ...core.offline_data_manager import (
    offline_data_manager, DataOperation, SyncStatus, ConflictResolutionStrategy
)
from ...core.offline_first_architecture import (
    offline_first_architecture, OperationMode, ConnectionStatus, OfflineCapability
)
from ...core.progressive_web_app import (
    progressive_web_app, CacheStrategy, ResourceType, CacheRule
)
from ...core.offline_sync_engine import get_sync_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/offline", tags=["offline"])


# Offline Data Management Routes

@router.post("/data/{collection}")
async def store_offline_data(
    collection: str,
    data: Dict[str, Any],
    operation: str = "create",
    user_id: Optional[str] = None
):
    """Store data in offline storage."""
    try:
        operation_enum = DataOperation(operation.lower())
        entry_id = await offline_data_manager.store_data(
            collection, data, operation_enum, user_id
        )
        
        return {
            "success": True,
            "entry_id": entry_id,
            "collection": collection,
            "operation": operation,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid operation: {operation}")
    except Exception as e:
        logger.error(f"Failed to store offline data: {e}")
        raise HTTPException(status_code=500, detail="Failed to store data")


@router.get("/data/{collection}/{entry_id}")
async def get_offline_data(collection: str, entry_id: str):
    """Retrieve data from offline storage."""
    try:
        data = await offline_data_manager.get_data(collection, entry_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Data not found")
        
        return {
            "success": True,
            "data": data,
            "collection": collection,
            "entry_id": entry_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve offline data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data")


@router.get("/data/{collection}")
async def query_offline_data(
    collection: str,
    user_id: Optional[str] = None,
    sync_status: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0
):
    """Query data from offline storage."""
    try:
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if sync_status:
            filters["sync_status"] = sync_status
        
        results = await offline_data_manager.query_data(
            collection, filters, limit, offset
        )
        
        return {
            "success": True,
            "results": results,
            "collection": collection,
            "count": len(results),
            "filters": filters
        }
        
    except Exception as e:
        logger.error(f"Failed to query offline data: {e}")
        raise HTTPException(status_code=500, detail="Failed to query data")


@router.put("/data/{collection}/{entry_id}")
async def update_offline_data(
    collection: str,
    entry_id: str,
    updates: Dict[str, Any],
    user_id: Optional[str] = None
):
    """Update data in offline storage."""
    try:
        success = await offline_data_manager.update_data(
            collection, entry_id, updates, user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Data not found")
        
        return {
            "success": True,
            "entry_id": entry_id,
            "collection": collection,
            "updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update offline data: {e}")
        raise HTTPException(status_code=500, detail="Failed to update data")


@router.delete("/data/{collection}/{entry_id}")
async def delete_offline_data(
    collection: str,
    entry_id: str,
    user_id: Optional[str] = None
):
    """Delete data from offline storage."""
    try:
        success = await offline_data_manager.delete_data(
            collection, entry_id, user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Data not found")
        
        return {
            "success": True,
            "entry_id": entry_id,
            "collection": collection,
            "deleted": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete offline data: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete data")


# Sync Management Routes

@router.post("/sync/force")
async def force_sync(
    collection: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Force synchronization of offline data."""
    try:
        if collection:
            sync_result = await offline_data_manager.force_sync(collection)
        else:
            sync_result = await offline_data_manager.force_sync()
        
        return {
            "success": True,
            "sync_result": sync_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to force sync: {e}")
        raise HTTPException(status_code=500, detail="Sync failed")


@router.get("/sync/status")
async def get_sync_status():
    """Get current synchronization status."""
    try:
        status = offline_data_manager.get_sync_status()
        
        # Add sync engine metrics if available
        sync_engine = get_sync_engine()
        if sync_engine:
            status["sync_metrics"] = sync_engine.get_sync_metrics()
            status["conflict_insights"] = sync_engine.get_conflict_insights()
        
        return {
            "success": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync status")


@router.get("/sync/conflicts")
async def get_sync_conflicts():
    """Get list of synchronization conflicts."""
    try:
        conflicts = offline_data_manager.get_conflicts()
        
        return {
            "success": True,
            "conflicts": conflicts,
            "count": len(conflicts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get sync conflicts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conflicts")


@router.post("/sync/conflicts/{conflict_id}/resolve")
async def resolve_sync_conflict(
    conflict_id: str,
    resolution: Dict[str, Any]
):
    """Resolve a synchronization conflict."""
    try:
        strategy = resolution.get("strategy", "intelligent_merge")
        custom_data = resolution.get("custom_data")
        
        try:
            strategy_enum = ConflictResolutionStrategy(strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid resolution strategy: {strategy}")
        
        success = await offline_data_manager.resolve_conflict(
            conflict_id, strategy_enum, custom_data
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        return {
            "success": True,
            "conflict_id": conflict_id,
            "resolution_strategy": strategy,
            "resolved": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve conflict: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve conflict")


# Offline-First Architecture Routes

@router.post("/architecture/execute")
async def execute_offline_first_request(
    request_data: Dict[str, Any]
):
    """Execute request using offline-first architecture."""
    try:
        feature_name = request_data.get("feature_name")
        operation = request_data.get("operation")
        data = request_data.get("data")
        user_id = request_data.get("user_id")
        
        if not feature_name or not operation:
            raise HTTPException(status_code=400, detail="feature_name and operation are required")
        
        result = await offline_first_architecture.execute_request(
            feature_name, operation, data, user_id
        )
        
        return {
            "success": True,
            "result": result,
            "feature_name": feature_name,
            "operation": operation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute offline-first request: {e}")
        raise HTTPException(status_code=500, detail="Request execution failed")


@router.get("/architecture/status")
async def get_architecture_status():
    """Get offline-first architecture status."""
    try:
        status = offline_first_architecture.get_architecture_status()
        
        return {
            "success": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get architecture status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@router.get("/architecture/capabilities")
async def get_offline_capabilities():
    """Get offline capabilities for all features."""
    try:
        capabilities = offline_first_architecture.get_offline_capabilities()
        
        return {
            "success": True,
            "capabilities": capabilities
        }
        
    except Exception as e:
        logger.error(f"Failed to get offline capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to get capabilities")


@router.post("/architecture/capabilities")
async def register_offline_capability(
    capability_data: Dict[str, Any]
):
    """Register offline capability for a feature."""
    try:
        capability = OfflineCapability(
            feature_name=capability_data["feature_name"],
            read_offline=capability_data.get("read_offline", True),
            write_offline=capability_data.get("write_offline", True),
            sync_priority=capability_data.get("sync_priority", "normal"),
            conflict_resolution=capability_data.get("conflict_resolution", "intelligent_merge"),
            cache_duration_hours=capability_data.get("cache_duration_hours", 24),
            max_offline_entries=capability_data.get("max_offline_entries", 1000),
            requires_auth=capability_data.get("requires_auth", True)
        )
        
        offline_first_architecture.register_feature_capability(capability)
        
        return {
            "success": True,
            "feature_name": capability.feature_name,
            "registered": True
        }
        
    except Exception as e:
        logger.error(f"Failed to register offline capability: {e}")
        raise HTTPException(status_code=500, detail="Failed to register capability")


@router.post("/architecture/mode")
async def set_operation_mode(
    mode_data: Dict[str, str]
):
    """Set operation mode for offline-first architecture."""
    try:
        mode = mode_data.get("mode")
        if not mode:
            raise HTTPException(status_code=400, detail="mode is required")
        
        try:
            mode_enum = OperationMode(mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid operation mode: {mode}")
        
        offline_first_architecture.set_operation_mode(mode_enum)
        
        return {
            "success": True,
            "operation_mode": mode,
            "updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set operation mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to set mode")


# Progressive Web App Routes

@router.get("/pwa/manifest.json")
async def get_pwa_manifest():
    """Get PWA manifest."""
    try:
        manifest = progressive_web_app.generate_manifest()
        return JSONResponse(content=manifest)
        
    except Exception as e:
        logger.error(f"Failed to generate PWA manifest: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate manifest")


@router.get("/pwa/sw.js")
async def get_service_worker():
    """Get service worker JavaScript."""
    try:
        sw_code = progressive_web_app.generate_service_worker()
        return PlainTextResponse(content=sw_code, media_type="application/javascript")
        
    except Exception as e:
        logger.error(f"Failed to generate service worker: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate service worker")


@router.get("/pwa/offline.html")
async def get_offline_page():
    """Get offline fallback page."""
    try:
        offline_html = progressive_web_app.generate_offline_page()
        return HTMLResponse(content=offline_html)
        
    except Exception as e:
        logger.error(f"Failed to generate offline page: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate offline page")


@router.get("/pwa/status")
async def get_pwa_status():
    """Get PWA status and metrics."""
    try:
        status = progressive_web_app.get_pwa_status()
        
        return {
            "success": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get PWA status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get PWA status")


@router.get("/pwa/cache-info")
async def get_cache_info():
    """Get PWA cache information."""
    try:
        cache_info = progressive_web_app.get_cache_info()
        
        return {
            "success": True,
            "cache_info": cache_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache info")


@router.post("/pwa/cache-rules")
async def add_cache_rule(
    rule_data: Dict[str, Any]
):
    """Add PWA cache rule."""
    try:
        rule = CacheRule(
            pattern=rule_data["pattern"],
            strategy=CacheStrategy(rule_data["strategy"]),
            resource_type=ResourceType(rule_data["resource_type"]),
            max_age_hours=rule_data.get("max_age_hours", 24),
            max_entries=rule_data.get("max_entries", 100),
            network_timeout_ms=rule_data.get("network_timeout_ms", 3000),
            cache_name=rule_data.get("cache_name", "default")
        )
        
        progressive_web_app.add_cache_rule(rule)
        
        return {
            "success": True,
            "rule": {
                "pattern": rule.pattern,
                "strategy": rule.strategy.value,
                "resource_type": rule.resource_type.value
            },
            "added": True
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid rule data: {e}")
    except Exception as e:
        logger.error(f"Failed to add cache rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to add cache rule")


@router.post("/pwa/install-prompt")
async def handle_install_prompt(
    prompt_data: Dict[str, Any]
):
    """Handle PWA install prompt."""
    try:
        user_id = prompt_data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        result = await progressive_web_app.handle_install_prompt(user_id)
        
        return {
            "success": True,
            "prompt_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to handle install prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle install prompt")


@router.post("/pwa/install-result")
async def handle_install_result(
    result_data: Dict[str, Any]
):
    """Handle PWA install result."""
    try:
        user_id = result_data.get("user_id")
        event_id = result_data.get("event_id")
        result = result_data.get("result")
        user_choice = result_data.get("user_choice")
        
        if not all([user_id, event_id is not None, result]):
            raise HTTPException(status_code=400, detail="user_id, event_id, and result are required")
        
        install_result = await progressive_web_app.handle_install_result(
            user_id, event_id, result, user_choice
        )
        
        return {
            "success": True,
            "install_result": install_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to handle install result: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle install result")


@router.post("/pwa/track-engagement")
async def track_user_engagement(
    engagement_data: Dict[str, Any]
):
    """Track user engagement for PWA."""
    try:
        user_id = engagement_data.get("user_id")
        action = engagement_data.get("action")
        duration = engagement_data.get("duration", 0)
        
        if not user_id or not action:
            raise HTTPException(status_code=400, detail="user_id and action are required")
        
        progressive_web_app.track_user_engagement(user_id, action, duration)
        
        return {
            "success": True,
            "user_id": user_id,
            "action": action,
            "tracked": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track user engagement: {e}")
        raise HTTPException(status_code=500, detail="Failed to track engagement")


@router.post("/pwa/push-notification")
async def send_push_notification(
    notification_data: Dict[str, Any]
):
    """Send push notification."""
    try:
        user_id = notification_data.get("user_id")
        title = notification_data.get("title")
        body = notification_data.get("body")
        data = notification_data.get("data")
        actions = notification_data.get("actions")
        
        if not all([user_id, title, body]):
            raise HTTPException(status_code=400, detail="user_id, title, and body are required")
        
        success = await progressive_web_app.send_push_notification(
            user_id, title, body, data, actions
        )
        
        return {
            "success": success,
            "notification_sent": success,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send push notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")


@router.post("/pwa/background-sync")
async def trigger_background_sync(
    sync_data: Dict[str, Any]
):
    """Trigger background sync."""
    try:
        tag = sync_data.get("tag")
        data = sync_data.get("data")
        
        if not tag:
            raise HTTPException(status_code=400, detail="tag is required")
        
        success = await progressive_web_app.trigger_background_sync(tag, data)
        
        return {
            "success": success,
            "sync_triggered": success,
            "tag": tag
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger background sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger sync")


# Storage Management Routes

@router.get("/storage/info")
async def get_storage_info():
    """Get storage information."""
    try:
        storage_info = offline_data_manager.get_storage_info()
        
        return {
            "success": True,
            "storage_info": storage_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get storage info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get storage info")


@router.post("/storage/cleanup")
async def cleanup_storage(
    cleanup_data: Dict[str, Any] = None
):
    """Clean up old storage data."""
    try:
        days_old = cleanup_data.get("days_old", 30) if cleanup_data else 30
        
        cleanup_result = await offline_data_manager.cleanup_storage(days_old)
        
        return {
            "success": True,
            "cleanup_result": cleanup_result
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup storage: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup storage")


# Health Check Routes

@router.get("/health")
async def health_check():
    """Health check for offline capabilities."""
    try:
        # Check offline data manager
        offline_status = offline_data_manager.get_sync_status()
        
        # Check architecture
        arch_status = offline_first_architecture.get_architecture_status()
        
        # Check PWA
        pwa_status = progressive_web_app.get_pwa_status()
        
        return {
            "success": True,
            "healthy": True,
            "components": {
                "offline_data_manager": {
                    "healthy": offline_status.get("is_online", False),
                    "pending_operations": offline_status.get("pending_sync_operations", 0)
                },
                "offline_first_architecture": {
                    "healthy": True,
                    "connection_status": arch_status.get("connection_status"),
                    "operation_mode": arch_status.get("operation_mode")
                },
                "progressive_web_app": {
                    "healthy": True,
                    "service_worker_version": pwa_status.get("service_worker_version"),
                    "cache_rules": pwa_status.get("cache_rules_count", 0)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Utility Routes

@router.post("/test/simulate-offline")
async def simulate_offline_mode():
    """Simulate offline mode for testing."""
    try:
        offline_data_manager.set_online_status(False)
        offline_first_architecture.connection_status = ConnectionStatus.OFFLINE
        offline_first_architecture.operation_mode = OperationMode.OFFLINE_ONLY
        
        return {
            "success": True,
            "mode": "offline",
            "simulated": True
        }
        
    except Exception as e:
        logger.error(f"Failed to simulate offline mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to simulate offline mode")


@router.post("/test/simulate-online")
async def simulate_online_mode():
    """Simulate online mode for testing."""
    try:
        offline_data_manager.set_online_status(True)
        offline_first_architecture.connection_status = ConnectionStatus.ONLINE
        offline_first_architecture.operation_mode = OperationMode.HYBRID
        
        return {
            "success": True,
            "mode": "online",
            "simulated": True
        }
        
    except Exception as e:
        logger.error(f"Failed to simulate online mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to simulate online mode")


@router.get("/test/generate-test-data")
async def generate_test_data():
    """Generate test data for offline capabilities testing."""
    try:
        test_collections = ["test_charts", "test_reports", "test_dashboards"]
        generated_entries = []
        
        for collection in test_collections:
            for i in range(5):
                test_data = {
                    "id": f"{collection}_item_{i}",
                    "name": f"Test {collection.replace('test_', '').title()} {i}",
                    "type": collection.replace("test_", ""),
                    "data": {"values": list(range(10))},
                    "created_at": datetime.now().isoformat()
                }
                
                entry_id = await offline_data_manager.store_data(
                    collection, test_data, DataOperation.CREATE, "test_user"
                )
                
                generated_entries.append({
                    "collection": collection,
                    "entry_id": entry_id,
                    "data": test_data
                })
        
        return {
            "success": True,
            "generated_entries": generated_entries,
            "total_entries": len(generated_entries)
        }
        
    except Exception as e:
        logger.error(f"Failed to generate test data: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate test data")