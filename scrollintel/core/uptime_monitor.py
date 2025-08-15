"""
ScrollIntel Uptime Monitoring System
Monitors service availability and generates status page data
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from pathlib import Path

from ..core.config import get_settings
from ..core.monitoring import metrics_collector

settings = get_settings()

class ServiceStatus(Enum):
    """Service status enumeration"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    url: str
    method: str = "GET"
    timeout: int = 10
    expected_status: int = 200
    expected_content: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    interval: int = 60  # seconds

@dataclass
class ServiceMetrics:
    """Service availability metrics"""
    name: str
    status: ServiceStatus
    response_time: float
    uptime_percentage: float
    last_check: datetime
    last_incident: Optional[datetime] = None
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0

@dataclass
class Incident:
    """Service incident record"""
    id: str
    service: str
    status: ServiceStatus
    title: str
    description: str
    started_at: datetime
    resolved_at: Optional[datetime] = None
    duration: Optional[int] = None  # seconds
    impact: str = "minor"  # minor, major, critical

class UptimeMonitor:
    """Monitors service uptime and availability"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checks: List[HealthCheck] = []
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.incidents: List[Incident] = []
        self.status_history: Dict[str, List[Dict]] = {}
        self.data_file = Path("data/uptime_data.json")
        
        # Ensure data directory exists
        self.data_file.parent.mkdir(exist_ok=True)
        
        # Initialize default health checks
        self._setup_default_health_checks()
        
        # Load existing data
        self._load_data()
    
    def _setup_default_health_checks(self):
        """Setup default health checks for ScrollIntel services"""
        base_url = getattr(settings, 'BASE_URL', 'http://localhost:8000')
        
        self.health_checks = [
            HealthCheck(
                name="API Health",
                url=f"{base_url}/health",
                timeout=5,
                interval=30
            ),
            HealthCheck(
                name="Database Connection",
                url=f"{base_url}/health/db",
                timeout=10,
                interval=60
            ),
            HealthCheck(
                name="Redis Cache",
                url=f"{base_url}/health/redis",
                timeout=5,
                interval=60
            ),
            HealthCheck(
                name="AI Agents",
                url=f"{base_url}/health/agents",
                timeout=15,
                interval=120
            ),
            HealthCheck(
                name="File Processing",
                url=f"{base_url}/health/files",
                timeout=10,
                interval=300
            ),
            HealthCheck(
                name="Authentication",
                url=f"{base_url}/health/auth",
                timeout=5,
                interval=60
            )
        ]
    
    async def perform_health_check(self, check: HealthCheck) -> tuple[bool, float, Optional[str]]:
        """Perform a single health check"""
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=check.timeout)
            headers = check.headers or {}
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(check.method, check.url, headers=headers) as response:
                    response_time = time.time() - start_time
                    
                    # Check status code
                    if response.status != check.expected_status:
                        return False, response_time, f"Expected status {check.expected_status}, got {response.status}"
                    
                    # Check content if specified
                    if check.expected_content:
                        content = await response.text()
                        if check.expected_content not in content:
                            return False, response_time, f"Expected content '{check.expected_content}' not found"
                    
                    return True, response_time, None
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return False, response_time, "Request timeout"
        except aiohttp.ClientError as e:
            response_time = time.time() - start_time
            return False, response_time, f"Client error: {str(e)}"
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, f"Unexpected error: {str(e)}"
    
    async def monitor_service(self, check: HealthCheck):
        """Monitor a single service continuously"""
        while True:
            try:
                success, response_time, error_message = await self.perform_health_check(check)
                
                # Update metrics
                if check.name not in self.service_metrics:
                    self.service_metrics[check.name] = ServiceMetrics(
                        name=check.name,
                        status=ServiceStatus.OPERATIONAL,
                        response_time=response_time,
                        uptime_percentage=100.0,
                        last_check=datetime.utcnow(),
                        total_checks=0,
                        successful_checks=0,
                        failed_checks=0
                    )
                
                metrics = self.service_metrics[check.name]
                metrics.total_checks += 1
                metrics.response_time = response_time
                metrics.last_check = datetime.utcnow()
                
                if success:
                    metrics.successful_checks += 1
                    # Update status if it was down
                    if metrics.status != ServiceStatus.OPERATIONAL:
                        await self._resolve_incident(check.name)
                        metrics.status = ServiceStatus.OPERATIONAL
                else:
                    metrics.failed_checks += 1
                    self.logger.warning(f"Health check failed for {check.name}: {error_message}")
                    
                    # Determine new status based on failure rate
                    failure_rate = metrics.failed_checks / metrics.total_checks
                    if failure_rate > 0.5:
                        new_status = ServiceStatus.MAJOR_OUTAGE
                    elif failure_rate > 0.2:
                        new_status = ServiceStatus.PARTIAL_OUTAGE
                    else:
                        new_status = ServiceStatus.DEGRADED
                    
                    # Create incident if status changed to worse
                    if metrics.status == ServiceStatus.OPERATIONAL:
                        await self._create_incident(check.name, new_status, error_message)
                    
                    metrics.status = new_status
                
                # Calculate uptime percentage
                metrics.uptime_percentage = (metrics.successful_checks / metrics.total_checks) * 100
                
                # Record status history
                await self._record_status_history(check.name, success, response_time)
                
                # Update Prometheus metrics
                metrics_collector.record_uptime_check(check.name, success, response_time)
                
                # Save data periodically
                if metrics.total_checks % 10 == 0:
                    await self._save_data()
                
            except Exception as e:
                self.logger.error(f"Error monitoring {check.name}: {e}")
            
            # Wait for next check
            await asyncio.sleep(check.interval)
    
    async def _create_incident(self, service: str, status: ServiceStatus, description: str):
        """Create a new incident"""
        incident_id = f"{service}_{int(time.time())}"
        
        # Determine impact level
        impact = "minor"
        if status == ServiceStatus.MAJOR_OUTAGE:
            impact = "critical"
        elif status == ServiceStatus.PARTIAL_OUTAGE:
            impact = "major"
        
        incident = Incident(
            id=incident_id,
            service=service,
            status=status,
            title=f"{service} experiencing issues",
            description=description,
            started_at=datetime.utcnow(),
            impact=impact
        )
        
        self.incidents.append(incident)
        
        # Update service metrics
        if service in self.service_metrics:
            self.service_metrics[service].last_incident = incident.started_at
        
        self.logger.error(f"Incident created: {incident.title}")
        
        # Trigger alert
        await self._trigger_incident_alert(incident)
    
    async def _resolve_incident(self, service: str):
        """Resolve the latest incident for a service"""
        # Find the latest unresolved incident for this service
        for incident in reversed(self.incidents):
            if incident.service == service and incident.resolved_at is None:
                incident.resolved_at = datetime.utcnow()
                incident.duration = int((incident.resolved_at - incident.started_at).total_seconds())
                
                self.logger.info(f"Incident resolved: {incident.title} (Duration: {incident.duration}s)")
                
                # Trigger resolution alert
                await self._trigger_resolution_alert(incident)
                break
    
    async def _trigger_incident_alert(self, incident: Incident):
        """Trigger alert for new incident"""
        # This would integrate with your alerting system
        alert_data = {
            "type": "incident_created",
            "incident": asdict(incident),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log for now (would send to alerting system)
        self.logger.critical(f"INCIDENT ALERT: {json.dumps(alert_data)}")
    
    async def _trigger_resolution_alert(self, incident: Incident):
        """Trigger alert for incident resolution"""
        alert_data = {
            "type": "incident_resolved",
            "incident": asdict(incident),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log for now (would send to alerting system)
        self.logger.info(f"RESOLUTION ALERT: {json.dumps(alert_data)}")
    
    async def _record_status_history(self, service: str, success: bool, response_time: float):
        """Record status history for trending"""
        if service not in self.status_history:
            self.status_history[service] = []
        
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "response_time": response_time
        }
        
        self.status_history[service].append(history_entry)
        
        # Keep only last 24 hours of history
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.status_history[service] = [
            entry for entry in self.status_history[service]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
    
    def get_overall_status(self) -> ServiceStatus:
        """Get overall system status"""
        if not self.service_metrics:
            return ServiceStatus.OPERATIONAL
        
        statuses = [metrics.status for metrics in self.service_metrics.values()]
        
        # Determine overall status based on individual service statuses
        if ServiceStatus.MAJOR_OUTAGE in statuses:
            return ServiceStatus.MAJOR_OUTAGE
        elif ServiceStatus.PARTIAL_OUTAGE in statuses:
            return ServiceStatus.PARTIAL_OUTAGE
        elif ServiceStatus.DEGRADED in statuses:
            return ServiceStatus.DEGRADED
        else:
            return ServiceStatus.OPERATIONAL
    
    def get_status_page_data(self) -> Dict[str, Any]:
        """Get data for status page"""
        overall_status = self.get_overall_status()
        
        # Get recent incidents (last 7 days)
        recent_incidents = [
            asdict(incident) for incident in self.incidents
            if incident.started_at > datetime.utcnow() - timedelta(days=7)
        ]
        
        # Sort incidents by start time (newest first)
        recent_incidents.sort(key=lambda x: x['started_at'], reverse=True)
        
        return {
            "overall_status": overall_status.value,
            "last_updated": datetime.utcnow().isoformat(),
            "services": {
                name: {
                    "status": metrics.status.value,
                    "uptime_percentage": round(metrics.uptime_percentage, 2),
                    "response_time": round(metrics.response_time * 1000, 2),  # Convert to ms
                    "last_check": metrics.last_check.isoformat()
                }
                for name, metrics in self.service_metrics.items()
            },
            "incidents": recent_incidents[:10],  # Last 10 incidents
            "uptime_stats": self._calculate_uptime_stats()
        }
    
    def _calculate_uptime_stats(self) -> Dict[str, float]:
        """Calculate uptime statistics"""
        if not self.service_metrics:
            return {"24h": 100.0, "7d": 100.0, "30d": 100.0}
        
        # For now, return current uptime percentages
        # In a real implementation, you'd calculate based on historical data
        avg_uptime = sum(metrics.uptime_percentage for metrics in self.service_metrics.values()) / len(self.service_metrics)
        
        return {
            "24h": round(avg_uptime, 2),
            "7d": round(avg_uptime, 2),
            "30d": round(avg_uptime, 2)
        }
    
    async def _save_data(self):
        """Save uptime data to file"""
        try:
            data = {
                "service_metrics": {
                    name: asdict(metrics) for name, metrics in self.service_metrics.items()
                },
                "incidents": [asdict(incident) for incident in self.incidents],
                "status_history": self.status_history,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Convert datetime objects to strings for JSON serialization
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, default=datetime_converter, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving uptime data: {e}")
    
    def _load_data(self):
        """Load uptime data from file"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load service metrics
                for name, metrics_data in data.get("service_metrics", {}).items():
                    # Convert string timestamps back to datetime
                    metrics_data["last_check"] = datetime.fromisoformat(metrics_data["last_check"])
                    if metrics_data.get("last_incident"):
                        metrics_data["last_incident"] = datetime.fromisoformat(metrics_data["last_incident"])
                    
                    metrics_data["status"] = ServiceStatus(metrics_data["status"])
                    self.service_metrics[name] = ServiceMetrics(**metrics_data)
                
                # Load incidents
                for incident_data in data.get("incidents", []):
                    incident_data["started_at"] = datetime.fromisoformat(incident_data["started_at"])
                    if incident_data.get("resolved_at"):
                        incident_data["resolved_at"] = datetime.fromisoformat(incident_data["resolved_at"])
                    
                    incident_data["status"] = ServiceStatus(incident_data["status"])
                    self.incidents.append(Incident(**incident_data))
                
                # Load status history
                self.status_history = data.get("status_history", {})
                
        except Exception as e:
            self.logger.error(f"Error loading uptime data: {e}")
    
    async def start_monitoring(self):
        """Start monitoring all services"""
        self.logger.info("Starting uptime monitoring...")
        
        # Create monitoring tasks for each health check
        tasks = []
        for check in self.health_checks:
            task = asyncio.create_task(self.monitor_service(check))
            tasks.append(task)
        
        # Wait for all tasks (they run indefinitely)
        await asyncio.gather(*tasks)

# Global uptime monitor instance
uptime_monitor = UptimeMonitor()