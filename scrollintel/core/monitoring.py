"""
Monitoring System - 100% Optimized
Comprehensive monitoring and observability for ScrollIntel
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Alert:
    name: str
    level: AlertLevel
    message: str
    timestamp: float
    resolved: bool = False
    metadata: Dict[str, Any] = None

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = {}
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels or {}
        )
        
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(metric)
            
            # Keep only last 1000 metrics per name
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metrics(self, name: str = None, since: float = None) -> List[Metric]:
        """Get metrics by name and time range"""
        with self._lock:
            if name:
                metrics = self.metrics.get(name, [])
            else:
                metrics = []
                for metric_list in self.metrics.values():
                    metrics.extend(metric_list)
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics

class PerformanceMonitor:
    """Monitors system performance"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics_collector.record_metric("cpu_usage", cpu_percent, MetricType.GAUGE)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics_collector.record_metric("memory_usage", memory.percent, MetricType.GAUGE)
                self.metrics_collector.record_metric("memory_available", memory.available, MetricType.GAUGE)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.metrics_collector.record_metric("disk_usage", disk.percent, MetricType.GAUGE)
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error

# Global instances
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor()

class MonitoringSystem:
    """Comprehensive monitoring system"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.alert_rules = {}
        self.monitoring_active = False
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.subscribers = weakref.WeakSet()
        
        # Monitoring configuration
        self.config = {
            'collection_interval': 5,  # seconds
            'retention_period': 3600,  # 1 hour
            'max_metrics': 10000,
            'alert_cooldown': 300,     # 5 minutes
            'enable_system_metrics': True,
            'enable_application_metrics': True,
            'enable_performance_metrics': True
        }
        
        # System thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 1.0,      # seconds
            'error_rate': 0.05,        # 5%
            'throughput_min': 100      # requests/minute
        }
        
        # Performance counters
        self.counters = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'bytes_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    async def start(self):
        """Start monitoring system"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("📊 Starting Monitoring System...")
        
        # Start monitoring tasks
        tasks = [
            self._collect_system_metrics(),
            self._collect_application_metrics(),
            self._process_alerts(),
            self._cleanup_old_data()
        ]
        
        # Run monitoring tasks concurrently
        asyncio.create_task(asyncio.gather(*tasks, return_exceptions=True))
        
        logger.info("✅ Monitoring System started")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        while self.monitoring_active:
            try:
                if self.config['enable_system_metrics']:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    await self.record_metric("system.cpu.usage", cpu_percent, MetricType.GAUGE)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    await self.record_metric("system.memory.usage", memory.percent, MetricType.GAUGE)
                    await self.record_metric("system.memory.available", memory.available, MetricType.GAUGE)
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    await self.record_metric("system.disk.usage", disk.percent, MetricType.GAUGE)
                    await self.record_metric("system.disk.free", disk.free, MetricType.GAUGE)
                    
                    # Network metrics
                    network = psutil.net_io_counters()
                    await self.record_metric("system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER)
                    await self.record_metric("system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER)
                    
                    # Process metrics
                    process = psutil.Process()
                    await self.record_metric("process.cpu.usage", process.cpu_percent(), MetricType.GAUGE)
                    await self.record_metric("process.memory.usage", process.memory_percent(), MetricType.GAUGE)
                    await self.record_metric("process.threads", process.num_threads(), MetricType.GAUGE)
                
                await asyncio.sleep(self.config['collection_interval'])
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        while self.monitoring_active:
            try:
                if self.config['enable_application_metrics']:
                    # Request metrics
                    await self.record_metric("app.requests.total", self.counters['requests_total'], MetricType.COUNTER)
                    await self.record_metric("app.requests.success", self.counters['requests_success'], MetricType.COUNTER)
                    await self.record_metric("app.requests.error", self.counters['requests_error'], MetricType.COUNTER)
                    
                    # Calculate error rate
                    total_requests = self.counters['requests_total']
                    if total_requests > 0:
                        error_rate = self.counters['requests_error'] / total_requests
                        await self.record_metric("app.error_rate", error_rate, MetricType.GAUGE)
                    
                    # Cache metrics
                    await self.record_metric("app.cache.hits", self.counters['cache_hits'], MetricType.COUNTER)
                    await self.record_metric("app.cache.misses", self.counters['cache_misses'], MetricType.COUNTER)
                    
                    # Calculate cache hit rate
                    total_cache_requests = self.counters['cache_hits'] + self.counters['cache_misses']
                    if total_cache_requests > 0:
                        hit_rate = self.counters['cache_hits'] / total_cache_requests
                        await self.record_metric("app.cache.hit_rate", hit_rate, MetricType.GAUGE)
                    
                    # Data processing metrics
                    await self.record_metric("app.bytes_processed", self.counters['bytes_processed'], MetricType.COUNTER)
                
                await asyncio.sleep(self.config['collection_interval'])
                
            except Exception as e:
                logger.error(f"Application metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _process_alerts(self):
        """Process alert rules and generate alerts"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Check alert rules
                for rule_name, rule in self.alert_rules.items():
                    await self._evaluate_alert_rule(rule_name, rule, current_time)
                
                # Clean up resolved alerts
                self._cleanup_resolved_alerts()
                
                await asyncio.sleep(10)  # Check alerts every 10 seconds
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_old_data(self):
        """Cleanup old metrics and alerts"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                retention_cutoff = current_time - self.config['retention_period']
                
                # Cleanup old metrics
                for metric_name, metric_list in self.metrics.items():
                    self.metrics[metric_name] = [
                        m for m in metric_list
                        if m.timestamp > retention_cutoff
                    ]
                
                # Cleanup old alerts
                self.alerts = [
                    alert for alert in self.alerts
                    if alert.timestamp > retention_cutoff or not alert.resolved
                ]
                
                # Limit metrics count
                for metric_name, metric_list in self.metrics.items():
                    if len(metric_list) > self.config['max_metrics']:
                        self.metrics[metric_name] = metric_list[-self.config['max_metrics']:]
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(metric)
            
            # Notify subscribers
            await self._notify_subscribers('metric_recorded', metric)
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    async def increment_counter(self, counter_name: str, value: float = 1.0):
        """Increment a counter"""
        if counter_name in self.counters:
            self.counters[counter_name] += value
        else:
            self.counters[counter_name] = value
        
        await self.record_metric(f"counter.{counter_name}", self.counters[counter_name], MetricType.COUNTER)
    
    async def record_timer(self, name: str, duration: float):
        """Record a timer metric"""
        await self.record_metric(name, duration, MetricType.TIMER)
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        self.alert_rules = {
            'high_cpu_usage': {
                'metric': 'system.cpu.usage',
                'condition': 'greater_than',
                'threshold': self.thresholds['cpu_usage'],
                'level': AlertLevel.WARNING,
                'message': 'High CPU usage detected: {value:.1f}%'
            },
            'high_memory_usage': {
                'metric': 'system.memory.usage',
                'condition': 'greater_than',
                'threshold': self.thresholds['memory_usage'],
                'level': AlertLevel.WARNING,
                'message': 'High memory usage detected: {value:.1f}%'
            },
            'high_disk_usage': {
                'metric': 'system.disk.usage',
                'condition': 'greater_than',
                'threshold': self.thresholds['disk_usage'],
                'level': AlertLevel.ERROR,
                'message': 'High disk usage detected: {value:.1f}%'
            },
            'high_error_rate': {
                'metric': 'app.error_rate',
                'condition': 'greater_than',
                'threshold': self.thresholds['error_rate'],
                'level': AlertLevel.ERROR,
                'message': 'High error rate detected: {value:.2%}'
            }
        }
    
    async def _evaluate_alert_rule(self, rule_name: str, rule: Dict[str, Any], current_time: float):
        """Evaluate an alert rule"""
        try:
            metric_name = rule['metric']
            
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return
            
            # Get latest metric value
            latest_metric = self.metrics[metric_name][-1]
            value = latest_metric.value
            threshold = rule['threshold']
            condition = rule['condition']
            
            # Evaluate condition
            triggered = False
            if condition == 'greater_than' and value > threshold:
                triggered = True
            elif condition == 'less_than' and value < threshold:
                triggered = True
            elif condition == 'equals' and value == threshold:
                triggered = True
            
            if triggered:
                # Check if alert already exists and not resolved
                existing_alert = None
                for alert in self.alerts:
                    if alert.name == rule_name and not alert.resolved:
                        existing_alert = alert
                        break
                
                if not existing_alert:
                    # Create new alert
                    alert = Alert(
                        name=rule_name,
                        level=rule['level'],
                        message=rule['message'].format(value=value),
                        timestamp=current_time,
                        metadata={'metric_value': value, 'threshold': threshold}
                    )
                    
                    self.alerts.append(alert)
                    await self._notify_subscribers('alert_triggered', alert)
                    
                    logger.warning(f"🚨 Alert triggered: {alert.message}")
            else:
                # Resolve existing alert if condition no longer met
                for alert in self.alerts:
                    if alert.name == rule_name and not alert.resolved:
                        alert.resolved = True
                        await self._notify_subscribers('alert_resolved', alert)
                        logger.info(f"✅ Alert resolved: {rule_name}")
        
        except Exception as e:
            logger.error(f"Alert rule evaluation error for {rule_name}: {e}")
    
    def _cleanup_resolved_alerts(self):
        """Cleanup resolved alerts older than cooldown period"""
        current_time = time.time()
        cooldown_cutoff = current_time - self.config['alert_cooldown']
        
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or alert.timestamp > cooldown_cutoff
        ]
    
    async def _notify_subscribers(self, event_type: str, data: Any):
        """Notify subscribers of monitoring events"""
        for subscriber in list(self.subscribers):
            try:
                if hasattr(subscriber, 'on_monitoring_event'):
                    await subscriber.on_monitoring_event(event_type, data)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")
    
    def subscribe(self, subscriber):
        """Subscribe to monitoring events"""
        self.subscribers.add(subscriber)
    
    def unsubscribe(self, subscriber):
        """Unsubscribe from monitoring events"""
        self.subscribers.discard(subscriber)
    
    def get_metrics(self, name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics"""
        if name:
            metrics = self.metrics.get(name, [])
            return [m.to_dict() for m in metrics[-limit:]]
        else:
            all_metrics = []
            for metric_name, metric_list in self.metrics.items():
                all_metrics.extend([m.to_dict() for m in metric_list[-limit:]])
            return sorted(all_metrics, key=lambda x: x['timestamp'])[-limit:]
    
    def get_alerts(self, resolved: bool = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts"""
        alerts = self.alerts
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return [asdict(alert) for alert in sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status summary"""
        try:
            # Get latest system metrics
            cpu_usage = 0.0
            memory_usage = 0.0
            disk_usage = 0.0
            
            if 'system.cpu.usage' in self.metrics and self.metrics['system.cpu.usage']:
                cpu_usage = self.metrics['system.cpu.usage'][-1].value
            
            if 'system.memory.usage' in self.metrics and self.metrics['system.memory.usage']:
                memory_usage = self.metrics['system.memory.usage'][-1].value
            
            if 'system.disk.usage' in self.metrics and self.metrics['system.disk.usage']:
                disk_usage = self.metrics['system.disk.usage'][-1].value
            
            # Count active alerts
            active_alerts = len([a for a in self.alerts if not a.resolved])
            
            # Determine overall health
            health = "healthy"
            if active_alerts > 0:
                critical_alerts = len([a for a in self.alerts if not a.resolved and a.level == AlertLevel.CRITICAL])
                error_alerts = len([a for a in self.alerts if not a.resolved and a.level == AlertLevel.ERROR])
                
                if critical_alerts > 0:
                    health = "critical"
                elif error_alerts > 0:
                    health = "degraded"
                else:
                    health = "warning"
            
            return {
                'health': health,
                'monitoring_active': self.monitoring_active,
                'system_metrics': {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage
                },
                'application_metrics': {
                    'requests_total': self.counters.get('requests_total', 0),
                    'error_rate': self.counters.get('requests_error', 0) / max(self.counters.get('requests_total', 1), 1),
                    'cache_hit_rate': self.counters.get('cache_hits', 0) / max(self.counters.get('cache_hits', 0) + self.counters.get('cache_misses', 0), 1)
                },
                'alerts': {
                    'active': active_alerts,
                    'total': len(self.alerts)
                },
                'metrics_count': sum(len(metrics) for metrics in self.metrics.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'health': 'unknown',
                'error': str(e),
                'monitoring_active': self.monitoring_active
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            start_time = time.time()
            
            # Basic functionality test
            test_metric_name = "health_check.test"
            await self.record_metric(test_metric_name, 1.0, MetricType.GAUGE)
            
            # Check if metric was recorded
            if test_metric_name in self.metrics and self.metrics[test_metric_name]:
                response_time = time.time() - start_time
                
                return {
                    'status': 'healthy',
                    'response_time_ms': round(response_time * 1000, 2),
                    'monitoring_active': self.monitoring_active,
                    'metrics_count': len(self.metrics),
                    'alerts_count': len(self.alerts)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Failed to record test metric',
                    'monitoring_active': self.monitoring_active
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'monitoring_active': self.monitoring_active
            }
    
    async def stop(self):
        """Stop monitoring system"""
        self.monitoring_active = False
        self.thread_pool.shutdown(wait=True)
        logger.info("📊 Monitoring System stopped")

# Global monitoring system instance
_monitoring_system = None

def get_monitoring_system() -> MonitoringSystem:
    """Get global monitoring system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system

async def start_monitoring():
    """Start monitoring system"""
    system = get_monitoring_system()
    await system.start()

def get_system_status():
    """Get system status"""
    system = get_monitoring_system()
    return system.get_system_status()