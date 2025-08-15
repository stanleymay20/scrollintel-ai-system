"""
ScrollIntel Status Page API Routes
Provides public status information and uptime data
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from ...core.uptime_monitor import uptime_monitor, ServiceStatus
from ...core.monitoring import metrics_collector
from ...core.log_aggregation import log_aggregator

router = APIRouter(prefix="/status", tags=["status"])

@router.get("/", response_model=Dict[str, Any])
async def get_status_page():
    """Get comprehensive status page data"""
    try:
        status_data = uptime_monitor.get_status_page_data()
        
        # Add additional metrics
        status_data["metrics"] = {
            "total_requests_24h": await _get_request_count_24h(),
            "avg_response_time": await _get_avg_response_time(),
            "active_users": await _get_active_user_count(),
            "system_load": await _get_system_load()
        }
        
        return status_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def get_health_status():
    """Get basic health status for monitoring"""
    try:
        overall_status = uptime_monitor.get_overall_status()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",  # Would come from app config
            "uptime": await _get_uptime_seconds()
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@router.get("/services", response_model=Dict[str, Any])
async def get_services_status():
    """Get detailed status for all services"""
    try:
        status_data = uptime_monitor.get_status_page_data()
        return {
            "services": status_data["services"],
            "last_updated": status_data["last_updated"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving services status: {str(e)}")

@router.get("/incidents", response_model=List[Dict[str, Any]])
async def get_recent_incidents(days: int = 7):
    """Get recent incidents"""
    try:
        if days > 30:
            days = 30  # Limit to 30 days max
            
        status_data = uptime_monitor.get_status_page_data()
        
        # Filter incidents by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_incidents = [
            incident for incident in status_data["incidents"]
            if datetime.fromisoformat(incident["started_at"]) > cutoff_date
        ]
        
        return recent_incidents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving incidents: {str(e)}")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_status_metrics():
    """Get status page metrics"""
    try:
        return {
            "uptime_stats": uptime_monitor._calculate_uptime_stats(),
            "performance_metrics": {
                "avg_response_time_24h": await _get_avg_response_time(),
                "request_count_24h": await _get_request_count_24h(),
                "error_rate_24h": await _get_error_rate_24h(),
                "active_users": await _get_active_user_count()
            },
            "system_metrics": metrics_collector.get_metrics_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

@router.get("/history/{service_name}")
async def get_service_history(service_name: str, hours: int = 24):
    """Get historical data for a specific service"""
    try:
        if hours > 168:  # Limit to 7 days
            hours = 168
            
        if service_name not in uptime_monitor.status_history:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        # Filter history by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        history = [
            entry for entry in uptime_monitor.status_history[service_name]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
        
        return {
            "service": service_name,
            "time_range": f"{hours} hours",
            "data_points": len(history),
            "history": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving service history: {str(e)}")

@router.post("/webhook/alert")
async def receive_alert_webhook(alert_data: Dict[str, Any]):
    """Receive alerts from monitoring systems"""
    try:
        # Process incoming alert
        alert_type = alert_data.get("type", "unknown")
        severity = alert_data.get("severity", "info")
        message = alert_data.get("message", "Alert received")
        
        # Log the alert
        print(f"ALERT WEBHOOK: {alert_type} - {severity} - {message}")
        
        # Here you would typically:
        # 1. Create or update incidents
        # 2. Send notifications
        # 3. Update status page
        
        return {"status": "received", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing alert: {str(e)}")

@router.get("/export")
async def export_status_data(format: str = "json", days: int = 7):
    """Export status data for external analysis"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        if days > 30:
            days = 30
            
        # Get comprehensive status data
        status_data = uptime_monitor.get_status_page_data()
        
        # Add historical data
        export_data = {
            "export_info": {
                "generated_at": datetime.utcnow().isoformat(),
                "time_range_days": days,
                "format": format
            },
            "current_status": status_data,
            "service_history": {}
        }
        
        # Add service history
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        for service_name, history in uptime_monitor.status_history.items():
            filtered_history = [
                entry for entry in history
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
            ]
            export_data["service_history"][service_name] = filtered_history
        
        if format == "json":
            return export_data
        else:
            # Convert to CSV format (simplified)
            csv_data = "timestamp,service,status,response_time\n"
            for service_name, history in export_data["service_history"].items():
                for entry in history:
                    csv_data += f"{entry['timestamp']},{service_name},{entry['success']},{entry['response_time']}\n"
            
            return {"csv_data": csv_data}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

# Helper functions
async def _get_request_count_24h() -> int:
    """Get total request count for last 24 hours"""
    try:
        # This would query your metrics system
        # For now, return a mock value
        return 12500
    except:
        return 0

async def _get_avg_response_time() -> float:
    """Get average response time"""
    try:
        # This would calculate from metrics
        return 0.25  # 250ms
    except:
        return 0.0

async def _get_error_rate_24h() -> float:
    """Get error rate for last 24 hours"""
    try:
        # Calculate error rate from logs
        error_summary = await log_aggregator.get_error_summary(hours=24)
        total_requests = await _get_request_count_24h()
        
        if total_requests > 0:
            return (error_summary["total_errors"] / total_requests) * 100
        return 0.0
    except:
        return 0.0

async def _get_active_user_count() -> int:
    """Get current active user count"""
    try:
        # This would query your session store
        return 45  # Mock value
    except:
        return 0

async def _get_system_load() -> Dict[str, float]:
    """Get current system load"""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except:
        return {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0}

async def _get_uptime_seconds() -> int:
    """Get application uptime in seconds"""
    try:
        # This would track application start time
        return 86400  # Mock: 24 hours
    except:
        return 0