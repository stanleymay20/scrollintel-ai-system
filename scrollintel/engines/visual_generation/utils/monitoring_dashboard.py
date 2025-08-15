"""
Monitoring dashboard and health check endpoints for visual generation system.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import redis.asyncio as redis

from .metrics_collector import MetricsCollector, SystemMetrics, GenerationMetrics
from .alerting_system import AlertingSystem
from .auto_scaling_manager import AutoScalingManager
from ..config import InfrastructureConfig


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]


class MetricsResponse(BaseModel):
    """Metrics response model."""
    timestamp: float
    system: Dict[str, Any]
    generation: Dict[str, Any]


class AlertsResponse(BaseModel):
    """Alerts response model."""
    total_alerts: int
    active_alerts: List[Dict[str, Any]]
    alert_statistics: Dict[str, Any]


class MonitoringDashboard:
    """Monitoring dashboard and API endpoints."""
    
    def __init__(
        self, 
        config: InfrastructureConfig,
        metrics_collector: MetricsCollector,
        alerting_system: AlertingSystem,
        auto_scaling_manager: Optional[AutoScalingManager] = None
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.alerting_system = alerting_system
        self.auto_scaling_manager = auto_scaling_manager
        
        self.app = FastAPI(title="ScrollIntel Visual Generation Monitoring")
        self.start_time = time.time()
        self.version = "1.0.0"
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Comprehensive health check endpoint."""
            return await self._perform_health_check()
        
        @self.app.get("/health/live")
        async def liveness_probe():
            """Kubernetes liveness probe endpoint."""
            return {"status": "alive", "timestamp": time.time()}
        
        @self.app.get("/health/ready")
        async def readiness_probe():
            """Kubernetes readiness probe endpoint."""
            return await self._check_readiness()
        
        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_current_metrics():
            """Get current system and generation metrics."""
            return await self._get_current_metrics()
        
        @self.app.get("/metrics/history")
        async def get_metrics_history(hours: int = 24):
            """Get metrics history for specified hours."""
            return await self._get_metrics_history(hours)
        
        @self.app.get("/metrics/prometheus")
        async def get_prometheus_metrics():
            """Get Prometheus-formatted metrics."""
            metrics_text = await self.metrics_collector.get_prometheus_metrics()
            return Response(content=metrics_text, media_type="text/plain")
        
        @self.app.get("/alerts", response_model=AlertsResponse)
        async def get_alerts():
            """Get current alerts and statistics."""
            return await self._get_alerts()
        
        @self.app.get("/alerts/test")
        async def test_alerts():
            """Test all notification channels."""
            return await self.alerting_system.test_notification_channels()
        
        @self.app.get("/scaling/status")
        async def get_scaling_status():
            """Get auto-scaling status."""
            if not self.auto_scaling_manager:
                raise HTTPException(status_code=404, detail="Auto-scaling not configured")
            return await self.auto_scaling_manager.get_scaling_status()
        
        @self.app.post("/scaling/manual")
        async def manual_scale(worker_type: str, target_count: int):
            """Manually scale workers."""
            if not self.auto_scaling_manager:
                raise HTTPException(status_code=404, detail="Auto-scaling not configured")
            
            from .auto_scaling_manager import WorkerType
            try:
                worker_type_enum = WorkerType(worker_type)
                await self.auto_scaling_manager.manual_scale(worker_type_enum, target_count)
                return {"status": "success", "message": f"Scaled {worker_type} to {target_count} workers"}
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid worker type: {worker_type}")
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def monitoring_dashboard():
            """Serve monitoring dashboard HTML."""
            return await self._generate_dashboard_html()
        
        @self.app.get("/dashboard/data")
        async def dashboard_data():
            """Get dashboard data for frontend."""
            return await self._get_dashboard_data()
    
    async def _perform_health_check(self) -> HealthCheckResponse:
        """Perform comprehensive health check."""
        components = {}
        overall_status = "healthy"
        
        # Check metrics collector
        try:
            current_metrics = await self.metrics_collector.get_current_metrics()
            components["metrics_collector"] = {
                "status": "healthy",
                "last_collection": current_metrics.get("timestamp", 0),
                "details": "Collecting metrics successfully"
            }
        except Exception as e:
            components["metrics_collector"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "Failed to collect metrics"
            }
            overall_status = "unhealthy"
        
        # Check alerting system
        try:
            recent_alerts = await self.alerting_system.get_recent_alerts(1)  # Last hour
            components["alerting_system"] = {
                "status": "healthy",
                "recent_alerts_count": len(recent_alerts),
                "details": "Alerting system operational"
            }
        except Exception as e:
            components["alerting_system"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "Alerting system error"
            }
            overall_status = "unhealthy"
        
        # Check auto-scaling manager
        if self.auto_scaling_manager:
            try:
                scaling_status = await self.auto_scaling_manager.get_scaling_status()
                components["auto_scaling"] = {
                    "status": "healthy" if scaling_status["monitoring_active"] else "warning",
                    "monitoring_active": scaling_status["monitoring_active"],
                    "total_workers": sum(scaling_status["active_workers"].values()),
                    "details": "Auto-scaling operational"
                }
            except Exception as e:
                components["auto_scaling"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "details": "Auto-scaling error"
                }
                overall_status = "unhealthy"
        
        # Check Redis connectivity
        if self.metrics_collector.redis_client:
            try:
                await self.metrics_collector.redis_client.ping()
                components["redis"] = {
                    "status": "healthy",
                    "details": "Redis connection active"
                }
            except Exception as e:
                components["redis"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "details": "Redis connection failed"
                }
                overall_status = "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=time.time(),
            version=self.version,
            uptime_seconds=time.time() - self.start_time,
            components=components
        )
    
    async def _check_readiness(self) -> Dict[str, Any]:
        """Check if service is ready to handle requests."""
        try:
            # Check if metrics collector is running
            if not self.metrics_collector.collection_task:
                return {"status": "not_ready", "reason": "Metrics collection not started"}
            
            # Check if we have recent metrics
            current_metrics = await self.metrics_collector.get_current_metrics()
            if not current_metrics:
                return {"status": "not_ready", "reason": "No metrics available"}
            
            # Check if metrics are recent (within last 2 minutes)
            last_collection = current_metrics.get("timestamp", 0)
            if time.time() - last_collection > 120:
                return {"status": "not_ready", "reason": "Metrics are stale"}
            
            return {"status": "ready", "timestamp": time.time()}
            
        except Exception as e:
            return {"status": "not_ready", "reason": str(e)}
    
    async def _get_current_metrics(self) -> MetricsResponse:
        """Get current metrics."""
        try:
            metrics = await self.metrics_collector.get_current_metrics()
            return MetricsResponse(
                timestamp=metrics.get("timestamp", time.time()),
                system=metrics.get("system", {}),
                generation=metrics.get("generation", {})
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
    
    async def _get_metrics_history(self, hours: int) -> Dict[str, Any]:
        """Get metrics history."""
        try:
            history = await self.metrics_collector.get_metrics_history(hours)
            return {
                "time_period_hours": hours,
                "system_metrics": history.get("system", []),
                "generation_metrics": history.get("generation", [])
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get metrics history: {str(e)}")
    
    async def _get_alerts(self) -> AlertsResponse:
        """Get current alerts."""
        try:
            active_alerts = await self.alerting_system.get_recent_alerts(24)
            alert_stats = await self.alerting_system.get_alert_statistics(24)
            
            return AlertsResponse(
                total_alerts=len(active_alerts),
                active_alerts=active_alerts,
                alert_statistics=alert_stats
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")
    
    async def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            # Get current metrics
            current_metrics = await self.metrics_collector.get_current_metrics()
            
            # Get recent alerts
            recent_alerts = await self.alerting_system.get_recent_alerts(24)
            
            # Get scaling status
            scaling_status = {}
            if self.auto_scaling_manager:
                scaling_status = await self.auto_scaling_manager.get_scaling_status()
            
            # Get health status
            health = await self._perform_health_check()
            
            return {
                "timestamp": time.time(),
                "health": health.dict(),
                "metrics": current_metrics,
                "alerts": {
                    "recent_count": len(recent_alerts),
                    "recent_alerts": recent_alerts[:10]  # Last 10 alerts
                },
                "scaling": scaling_status,
                "uptime_seconds": time.time() - self.start_time
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")
    
    async def _generate_dashboard_html(self) -> str:
        """Generate monitoring dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ScrollIntel Visual Generation Monitoring</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .card {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #333;
                }
                .metric-label {
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 5px;
                }
                .status-healthy { color: #28a745; }
                .status-warning { color: #ffc107; }
                .status-unhealthy { color: #dc3545; }
                .alert-item {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                    border-left: 4px solid;
                }
                .alert-critical { border-left-color: #dc3545; background-color: #f8d7da; }
                .alert-warning { border-left-color: #ffc107; background-color: #fff3cd; }
                .alert-info { border-left-color: #17a2b8; background-color: #d1ecf1; }
                .refresh-btn {
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 0;
                }
                .refresh-btn:hover {
                    background: #0056b3;
                }
                .chart-container {
                    position: relative;
                    height: 300px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ScrollIntel Visual Generation Monitoring</h1>
                    <p>Real-time performance monitoring and alerting dashboard</p>
                    <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
                    <span id="last-updated"></span>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>System Health</h3>
                        <div id="health-status" class="metric-value">Loading...</div>
                        <div class="metric-label">Overall Status</div>
                        <div id="uptime" class="metric-label"></div>
                    </div>
                    
                    <div class="card">
                        <h3>CPU Usage</h3>
                        <div id="cpu-usage" class="metric-value">--%</div>
                        <div class="metric-label">Current CPU Utilization</div>
                    </div>
                    
                    <div class="card">
                        <h3>Memory Usage</h3>
                        <div id="memory-usage" class="metric-value">--%</div>
                        <div class="metric-label">Current Memory Utilization</div>
                    </div>
                    
                    <div class="card">
                        <h3>GPU Usage</h3>
                        <div id="gpu-usage" class="metric-value">--%</div>
                        <div class="metric-label">Current GPU Utilization</div>
                    </div>
                    
                    <div class="card">
                        <h3>Queue Length</h3>
                        <div id="queue-length" class="metric-value">--</div>
                        <div class="metric-label">Pending Requests</div>
                    </div>
                    
                    <div class="card">
                        <h3>Success Rate</h3>
                        <div id="success-rate" class="metric-value">--%</div>
                        <div class="metric-label">Request Success Rate</div>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>Recent Alerts</h3>
                        <div id="alerts-container">Loading alerts...</div>
                    </div>
                    
                    <div class="card">
                        <h3>Active Workers</h3>
                        <div id="workers-container">Loading worker status...</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Performance Trends</h3>
                    <div class="chart-container">
                        <canvas id="performance-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <script>
                let performanceChart;
                
                async function refreshData() {
                    try {
                        const response = await fetch('/dashboard/data');
                        const data = await response.json();
                        updateDashboard(data);
                        document.getElementById('last-updated').textContent = 
                            'Last updated: ' + new Date().toLocaleTimeString();
                    } catch (error) {
                        console.error('Failed to refresh data:', error);
                    }
                }
                
                function updateDashboard(data) {
                    // Update health status
                    const healthElement = document.getElementById('health-status');
                    healthElement.textContent = data.health.status.toUpperCase();
                    healthElement.className = 'metric-value status-' + 
                        (data.health.status === 'healthy' ? 'healthy' : 
                         data.health.status === 'degraded' ? 'warning' : 'unhealthy');
                    
                    // Update uptime
                    const uptimeHours = Math.floor(data.uptime_seconds / 3600);
                    const uptimeMinutes = Math.floor((data.uptime_seconds % 3600) / 60);
                    document.getElementById('uptime').textContent = 
                        `Uptime: ${uptimeHours}h ${uptimeMinutes}m`;
                    
                    // Update system metrics
                    if (data.metrics.system) {
                        document.getElementById('cpu-usage').textContent = 
                            Math.round(data.metrics.system.cpu_usage_percent || 0) + '%';
                        document.getElementById('memory-usage').textContent = 
                            Math.round(data.metrics.system.memory_usage_percent || 0) + '%';
                        document.getElementById('gpu-usage').textContent = 
                            Math.round(data.metrics.system.gpu_usage_percent || 0) + '%';
                    }
                    
                    // Update generation metrics
                    if (data.metrics.generation) {
                        document.getElementById('queue-length').textContent = 
                            data.metrics.generation.queue_length || 0;
                        document.getElementById('success-rate').textContent = 
                            Math.round((data.metrics.generation.success_rate || 0) * 100) + '%';
                    }
                    
                    // Update alerts
                    updateAlerts(data.alerts);
                    
                    // Update workers
                    updateWorkers(data.scaling);
                }
                
                function updateAlerts(alertsData) {
                    const container = document.getElementById('alerts-container');
                    if (!alertsData.recent_alerts || alertsData.recent_alerts.length === 0) {
                        container.innerHTML = '<p>No recent alerts</p>';
                        return;
                    }
                    
                    container.innerHTML = alertsData.recent_alerts.map(alert => `
                        <div class="alert-item alert-${alert.severity}">
                            <strong>${alert.rule_name}</strong><br>
                            <small>${alert.message}</small><br>
                            <small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
                
                function updateWorkers(scalingData) {
                    const container = document.getElementById('workers-container');
                    if (!scalingData.active_workers) {
                        container.innerHTML = '<p>No scaling data available</p>';
                        return;
                    }
                    
                    container.innerHTML = Object.entries(scalingData.active_workers).map(([type, count]) => `
                        <div style="margin: 10px 0;">
                            <strong>${type.replace('_', ' ').toUpperCase()}:</strong> ${count} workers
                        </div>
                    `).join('');
                }
                
                // Initialize dashboard
                refreshData();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
            </script>
        </body>
        </html>
        """


async def create_monitoring_app(
    config: InfrastructureConfig,
    metrics_collector: MetricsCollector,
    alerting_system: AlertingSystem,
    auto_scaling_manager: Optional[AutoScalingManager] = None
) -> FastAPI:
    """Create and configure the monitoring FastAPI application."""
    
    dashboard = MonitoringDashboard(
        config=config,
        metrics_collector=metrics_collector,
        alerting_system=alerting_system,
        auto_scaling_manager=auto_scaling_manager
    )
    
    return dashboard.app