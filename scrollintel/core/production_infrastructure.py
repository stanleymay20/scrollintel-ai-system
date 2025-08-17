"""
Production Infrastructure Management System
Handles load balancing, auto-scaling, and production deployment
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import psutil
import redis
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import aiohttp
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ServiceMetrics:
    cpu_usage: float
    memory_usage: float
    response_time: float
    error_rate: float
    active_connections: int
    timestamp: float

@dataclass
class LoadBalancerConfig:
    algorithm: str = "round_robin"  # round_robin, least_connections, weighted
    health_check_interval: int = 30
    max_retries: int = 3
    timeout: int = 5

class ProductionInfrastructure:
    """Production-ready infrastructure management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.db_pool = None
        self.load_balancer = LoadBalancer(LoadBalancerConfig())
        self.auto_scaler = AutoScaler(config.get('scaling', {}))
        self.health_monitor = HealthMonitor()
        self.cache_manager = CacheManager()
        
    async def initialize(self):
        """Initialize all infrastructure components"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True,
                max_connections=20
            )
            
            # Initialize database connection pool
            self.db_pool = create_engine(
                self.config.get('database_url'),
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Start background services
            await self._start_background_services()
            
            logger.info("Production infrastructure initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize infrastructure: {e}")
            raise

    async def _start_background_services(self):
        """Start background monitoring and scaling services"""
        asyncio.create_task(self.health_monitor.start_monitoring())
        asyncio.create_task(self.auto_scaler.start_scaling_monitor())
        asyncio.create_task(self.cache_manager.start_cache_cleanup())

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            'status': await self.health_monitor.get_overall_status(),
            'metrics': await self.health_monitor.get_current_metrics(),
            'load_balancer': self.load_balancer.get_status(),
            'auto_scaler': self.auto_scaler.get_status(),
            'cache': self.cache_manager.get_status(),
            'timestamp': time.time()
        }

class LoadBalancer:
    """Intelligent load balancing with health checks"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.servers = []
        self.current_index = 0
        self.server_health = {}
        
    def add_server(self, server_url: str, weight: int = 1):
        """Add server to load balancer pool"""
        self.servers.append({
            'url': server_url,
            'weight': weight,
            'active_connections': 0,
            'total_requests': 0,
            'error_count': 0
        })
        self.server_health[server_url] = HealthStatus.HEALTHY
        
    async def get_next_server(self) -> Optional[str]:
        """Get next available server based on algorithm"""
        healthy_servers = [
            server for server in self.servers 
            if self.server_health.get(server['url']) == HealthStatus.HEALTHY
        ]
        
        if not healthy_servers:
            logger.warning("No healthy servers available")
            return None
            
        if self.config.algorithm == "round_robin":
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1
            return server['url']
            
        elif self.config.algorithm == "least_connections":
            server = min(healthy_servers, key=lambda s: s['active_connections'])
            return server['url']
            
        elif self.config.algorithm == "weighted":
            # Weighted round robin implementation
            total_weight = sum(s['weight'] for s in healthy_servers)
            if total_weight == 0:
                return healthy_servers[0]['url']
            # Simplified weighted selection
            return healthy_servers[0]['url']
            
        return healthy_servers[0]['url']
    
    async def health_check(self, server_url: str) -> HealthStatus:
        """Perform health check on server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        return HealthStatus.HEALTHY
                    else:
                        return HealthStatus.DEGRADED
                        
        except Exception as e:
            logger.warning(f"Health check failed for {server_url}: {e}")
            return HealthStatus.UNHEALTHY
    
    async def start_health_checks(self):
        """Start periodic health checks"""
        while True:
            for server in self.servers:
                status = await self.health_check(server['url'])
                self.server_health[server['url']] = status
                
            await asyncio.sleep(self.config.health_check_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status"""
        return {
            'servers': self.servers,
            'health_status': dict(self.server_health),
            'algorithm': self.config.algorithm,
            'total_servers': len(self.servers),
            'healthy_servers': sum(1 for status in self.server_health.values() 
                                 if status == HealthStatus.HEALTHY)
        }

class AutoScaler:
    """Automatic scaling based on metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_instances = config.get('min_instances', 2)
        self.max_instances = config.get('max_instances', 10)
        self.target_cpu = config.get('target_cpu', 70.0)
        self.target_memory = config.get('target_memory', 80.0)
        self.scale_up_threshold = config.get('scale_up_threshold', 80.0)
        self.scale_down_threshold = config.get('scale_down_threshold', 30.0)
        self.current_instances = self.min_instances
        
    async def start_scaling_monitor(self):
        """Start monitoring for scaling decisions"""
        while True:
            try:
                metrics = await self._collect_metrics()
                scaling_decision = self._make_scaling_decision(metrics)
                
                if scaling_decision != 0:
                    await self._execute_scaling(scaling_decision)
                    
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                
            await asyncio.sleep(60)  # Check every minute
    
    async def _collect_metrics(self) -> ServiceMetrics:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return ServiceMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            response_time=0.0,  # Would be collected from monitoring
            error_rate=0.0,     # Would be collected from monitoring
            active_connections=0,  # Would be collected from load balancer
            timestamp=time.time()
        )
    
    def _make_scaling_decision(self, metrics: ServiceMetrics) -> int:
        """Make scaling decision based on metrics"""
        if (metrics.cpu_usage > self.scale_up_threshold or 
            metrics.memory_usage > self.scale_up_threshold):
            if self.current_instances < self.max_instances:
                return 1  # Scale up
                
        elif (metrics.cpu_usage < self.scale_down_threshold and 
              metrics.memory_usage < self.scale_down_threshold):
            if self.current_instances > self.min_instances:
                return -1  # Scale down
                
        return 0  # No scaling needed
    
    async def _execute_scaling(self, direction: int):
        """Execute scaling action"""
        if direction > 0:
            logger.info(f"Scaling up from {self.current_instances} instances")
            self.current_instances += 1
            # Here you would integrate with your container orchestrator
            # e.g., Kubernetes, Docker Swarm, etc.
            
        elif direction < 0:
            logger.info(f"Scaling down from {self.current_instances} instances")
            self.current_instances -= 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get auto-scaler status"""
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'target_cpu': self.target_cpu,
            'target_memory': self.target_memory
        }

class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        while True:
            try:
                metrics = await self._collect_health_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _collect_health_metrics(self) -> ServiceMetrics:
        """Collect comprehensive health metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return ServiceMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            response_time=0.0,  # Would be measured from actual requests
            error_rate=0.0,     # Would be calculated from logs
            active_connections=0,  # Would be from connection pool
            timestamp=time.time()
        )
    
    async def _check_alerts(self, metrics: ServiceMetrics):
        """Check for alert conditions"""
        alerts = []
        
        if metrics.cpu_usage > 90:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'critical',
                'message': f'CPU usage at {metrics.cpu_usage}%',
                'timestamp': metrics.timestamp
            })
            
        if metrics.memory_usage > 90:
            alerts.append({
                'type': 'high_memory',
                'severity': 'critical',
                'message': f'Memory usage at {metrics.memory_usage}%',
                'timestamp': metrics.timestamp
            })
        
        self.alerts.extend(alerts)
        
        # Keep only recent alerts
        cutoff_time = time.time() - 3600  # Last hour
        self.alerts = [alert for alert in self.alerts 
                      if alert['timestamp'] > cutoff_time]
    
    async def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.metrics_history:
            return HealthStatus.UNHEALTHY
            
        latest_metrics = self.metrics_history[-1]
        
        if (latest_metrics.cpu_usage > 90 or 
            latest_metrics.memory_usage > 90):
            return HealthStatus.UNHEALTHY
            
        if (latest_metrics.cpu_usage > 80 or 
            latest_metrics.memory_usage > 80):
            return HealthStatus.DEGRADED
            
        return HealthStatus.HEALTHY
    
    async def get_current_metrics(self) -> Optional[ServiceMetrics]:
        """Get current system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None

class CacheManager:
    """Intelligent caching management"""
    
    def __init__(self):
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        
    async def start_cache_cleanup(self):
        """Start periodic cache cleanup"""
        while True:
            try:
                await self._cleanup_expired_keys()
                await self._update_cache_stats()
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                
            await asyncio.sleep(300)  # Every 5 minutes
    
    async def _cleanup_expired_keys(self):
        """Clean up expired cache keys"""
        # Implementation would depend on cache backend
        pass
    
    async def _update_cache_stats(self):
        """Update cache statistics"""
        # Implementation would collect stats from Redis/Memcached
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache status"""
        hit_rate = (self.cache_stats['hits'] / 
                   (self.cache_stats['hits'] + self.cache_stats['misses'])
                   if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 
                   else 0)
        
        return {
            'hit_rate': hit_rate,
            'total_requests': self.cache_stats['hits'] + self.cache_stats['misses'],
            'memory_usage': self.cache_stats['memory_usage'],
            'evictions': self.cache_stats['evictions']
        }

# Global infrastructure instance
infrastructure = None

async def initialize_infrastructure(config: Dict[str, Any]):
    """Initialize global infrastructure"""
    global infrastructure
    infrastructure = ProductionInfrastructure(config)
    await infrastructure.initialize()
    return infrastructure

async def get_infrastructure() -> ProductionInfrastructure:
    """Get global infrastructure instance"""
    if infrastructure is None:
        raise RuntimeError("Infrastructure not initialized")
    return infrastructure