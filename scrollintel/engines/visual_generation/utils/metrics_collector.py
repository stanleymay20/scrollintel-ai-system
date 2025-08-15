"""
Real-time performance metrics collection for visual generation system.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import aiohttp
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import GPUtil

from ..config import InfrastructureConfig
from ..exceptions import MonitoringError


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_available_gb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    gpu_temperature: float = 0.0
    load_average: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))


@dataclass
class GenerationMetrics:
    """Generation-specific performance metrics."""
    timestamp: float = field(default_factory=time.time)
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    queue_length: int = 0
    average_generation_time: float = 0.0
    average_queue_wait_time: float = 0.0
    requests_per_second: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0
    
    # Model-specific metrics
    model_usage: Dict[str, int] = field(default_factory=dict)
    model_performance: Dict[str, float] = field(default_factory=dict)
    model_costs: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    average_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Configuration for performance alerts."""
    name: str
    metric_path: str  # e.g., "system.cpu_usage_percent"
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    duration: int  # seconds - how long condition must persist
    severity: str  # "critical", "warning", "info"
    enabled: bool = True
    last_triggered: Optional[float] = None
    cooldown: int = 300  # seconds between alerts


class MetricsCollector:
    """Collects real-time performance metrics for visual generation system."""
    
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.generation_metrics_history: List[GenerationMetrics] = []
        
        # Redis for distributed metrics
        self.redis_client: Optional[redis.Redis] = None
        
        # Prometheus metrics
        self.prometheus_registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Alert system
        self.alert_rules: List[AlertRule] = []
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.request_start_times: Dict[str, float] = {}
        self.model_usage_counters: Dict[str, int] = {}
        
        self._initialize_redis()
        self._setup_default_alerts()
    
    def _initialize_redis(self):
        """Initialize Redis connection for distributed metrics."""
        if hasattr(self.config, 'redis_url') and self.config.redis_url:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis for metrics: {e}")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # System metrics
        self.cpu_usage_gauge = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.prometheus_registry
        )
        
        self.memory_usage_gauge = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.prometheus_registry
        )
        
        self.gpu_usage_gauge = Gauge(
            'system_gpu_usage_percent',
            'GPU usage percentage',
            registry=self.prometheus_registry
        )
        
        # Generation metrics
        self.generation_requests_total = Counter(
            'generation_requests_total',
            'Total number of generation requests',
            ['model', 'status'],
            registry=self.prometheus_registry
        )
        
        self.generation_duration_histogram = Histogram(
            'generation_duration_seconds',
            'Generation duration in seconds',
            ['model'],
            registry=self.prometheus_registry
        )
        
        self.queue_length_gauge = Gauge(
            'generation_queue_length',
            'Current queue length',
            registry=self.prometheus_registry
        )
        
        self.quality_score_histogram = Histogram(
            'generation_quality_score',
            'Generation quality scores',
            ['model'],
            registry=self.prometheus_registry
        )
        
        self.cost_gauge = Gauge(
            'generation_cost_per_hour',
            'Current cost per hour',
            registry=self.prometheus_registry
        )
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        self.alert_rules = [
            AlertRule(
                name="High CPU Usage",
                metric_path="system.cpu_usage_percent",
                threshold=90.0,
                comparison="gt",
                duration=300,
                severity="warning"
            ),
            AlertRule(
                name="Critical CPU Usage",
                metric_path="system.cpu_usage_percent",
                threshold=95.0,
                comparison="gt",
                duration=60,
                severity="critical"
            ),
            AlertRule(
                name="High Memory Usage",
                metric_path="system.memory_usage_percent",
                threshold=85.0,
                comparison="gt",
                duration=300,
                severity="warning"
            ),
            AlertRule(
                name="Critical Memory Usage",
                metric_path="system.memory_usage_percent",
                threshold=95.0,
                comparison="gt",
                duration=60,
                severity="critical"
            ),
            AlertRule(
                name="High GPU Usage",
                metric_path="system.gpu_usage_percent",
                threshold=95.0,
                comparison="gt",
                duration=300,
                severity="warning"
            ),
            AlertRule(
                name="High Error Rate",
                metric_path="generation.error_rate",
                threshold=0.1,  # 10%
                comparison="gt",
                duration=300,
                severity="warning"
            ),
            AlertRule(
                name="Critical Error Rate",
                metric_path="generation.error_rate",
                threshold=0.25,  # 25%
                comparison="gt",
                duration=60,
                severity="critical"
            ),
            AlertRule(
                name="Long Queue",
                metric_path="generation.queue_length",
                threshold=50,
                comparison="gt",
                duration=600,
                severity="warning"
            ),
            AlertRule(
                name="High Cost",
                metric_path="generation.cost_per_hour",
                threshold=100.0,
                comparison="gt",
                duration=300,
                severity="warning"
            )
        ]
    
    async def start_monitoring(self):
        """Start the metrics collection and alerting."""
        if self.collection_task is None:
            self.collection_task = asyncio.create_task(self._collection_loop())
        
        if self.alert_task is None:
            self.alert_task = asyncio.create_task(self._alert_loop())
        
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop the metrics collection and alerting."""
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None
        
        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
            self.alert_task = None
        
        self.logger.info("Performance monitoring stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect generation metrics
                generation_metrics = await self._collect_generation_metrics()
                self.generation_metrics_history.append(generation_metrics)
                
                # Update Prometheus metrics
                await self._update_prometheus_metrics(system_metrics, generation_metrics)
                
                # Store in Redis for distributed access
                if self.redis_client:
                    await self._store_metrics_in_redis(system_metrics, generation_metrics)
                
                # Cleanup old metrics (keep last 24 hours)
                await self._cleanup_old_metrics()
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_available_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_recv = network.bytes_recv
            
            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0.0, 0.0, 0.0)
            
            # GPU metrics
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            gpu_temperature = 0.0
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory_usage = gpu.memoryUtil * 100
                    gpu_temperature = gpu.temperature
            except Exception as e:
                self.logger.debug(f"GPU metrics not available: {e}")
            
            return SystemMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_percent,
                disk_available_gb=disk_available_gb,
                network_bytes_sent=network_sent,
                network_bytes_recv=network_recv,
                gpu_usage_percent=gpu_usage,
                gpu_memory_usage_percent=gpu_memory_usage,
                gpu_temperature=gpu_temperature,
                load_average=load_avg
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()
    
    async def _collect_generation_metrics(self) -> GenerationMetrics:
        """Collect generation-specific performance metrics."""
        try:
            # Get metrics from Redis or local storage
            if self.redis_client:
                metrics_data = await self._get_generation_metrics_from_redis()
            else:
                metrics_data = await self._get_generation_metrics_from_local()
            
            return GenerationMetrics(**metrics_data)
            
        except Exception as e:
            self.logger.error(f"Failed to collect generation metrics: {e}")
            return GenerationMetrics()
    
    async def _get_generation_metrics_from_redis(self) -> Dict[str, Any]:
        """Get generation metrics from Redis."""
        try:
            # Get basic counters
            total_requests = int(await self.redis_client.get('metrics:total_requests') or 0)
            completed_requests = int(await self.redis_client.get('metrics:completed_requests') or 0)
            failed_requests = int(await self.redis_client.get('metrics:failed_requests') or 0)
            
            # Get queue length
            queue_length = await self.redis_client.llen('generation_queue')
            
            # Calculate rates and averages
            success_rate = completed_requests / total_requests if total_requests > 0 else 0.0
            error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
            
            # Get timing metrics
            generation_times = await self.redis_client.lrange('metrics:generation_times', 0, -1)
            avg_generation_time = 0.0
            if generation_times:
                times = [float(t) for t in generation_times]
                avg_generation_time = sum(times) / len(times)
            
            # Get model usage
            model_usage = {}
            model_keys = await self.redis_client.keys('metrics:model_usage:*')
            for key in model_keys:
                model_name = key.split(':')[-1]
                usage_count = int(await self.redis_client.get(key) or 0)
                model_usage[model_name] = usage_count
            
            return {
                'total_requests': total_requests,
                'completed_requests': completed_requests,
                'failed_requests': failed_requests,
                'queue_length': queue_length,
                'average_generation_time': avg_generation_time,
                'success_rate': success_rate,
                'error_rate': error_rate,
                'model_usage': model_usage
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics from Redis: {e}")
            return {}
    
    async def _get_generation_metrics_from_local(self) -> Dict[str, Any]:
        """Get generation metrics from local storage."""
        # This would integrate with your local job tracking system
        return {
            'total_requests': len(self.request_start_times),
            'completed_requests': 0,
            'failed_requests': 0,
            'queue_length': 0,
            'average_generation_time': 0.0,
            'success_rate': 0.0,
            'error_rate': 0.0,
            'model_usage': self.model_usage_counters.copy()
        }
    
    async def _update_prometheus_metrics(self, system_metrics: SystemMetrics, generation_metrics: GenerationMetrics):
        """Update Prometheus metrics."""
        try:
            # Update system metrics
            self.cpu_usage_gauge.set(system_metrics.cpu_usage_percent)
            self.memory_usage_gauge.set(system_metrics.memory_usage_percent)
            self.gpu_usage_gauge.set(system_metrics.gpu_usage_percent)
            
            # Update generation metrics
            self.queue_length_gauge.set(generation_metrics.queue_length)
            self.cost_gauge.set(generation_metrics.cost_per_hour)
            
        except Exception as e:
            self.logger.error(f"Failed to update Prometheus metrics: {e}")
    
    async def _store_metrics_in_redis(self, system_metrics: SystemMetrics, generation_metrics: GenerationMetrics):
        """Store metrics in Redis for distributed access."""
        try:
            # Store system metrics
            system_key = f"metrics:system:{int(system_metrics.timestamp)}"
            await self.redis_client.hset(system_key, mapping={
                'cpu_usage': system_metrics.cpu_usage_percent,
                'memory_usage': system_metrics.memory_usage_percent,
                'gpu_usage': system_metrics.gpu_usage_percent,
                'disk_usage': system_metrics.disk_usage_percent
            })
            await self.redis_client.expire(system_key, 86400)  # 24 hours
            
            # Store generation metrics
            generation_key = f"metrics:generation:{int(generation_metrics.timestamp)}"
            await self.redis_client.hset(generation_key, mapping={
                'total_requests': generation_metrics.total_requests,
                'completed_requests': generation_metrics.completed_requests,
                'failed_requests': generation_metrics.failed_requests,
                'queue_length': generation_metrics.queue_length,
                'success_rate': generation_metrics.success_rate,
                'error_rate': generation_metrics.error_rate
            })
            await self.redis_client.expire(generation_key, 86400)  # 24 hours
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics in Redis: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat."""
        cutoff_time = time.time() - 86400  # 24 hours ago
        
        # Clean up system metrics
        self.system_metrics_history = [
            m for m in self.system_metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        # Clean up generation metrics
        self.generation_metrics_history = [
            m for m in self.generation_metrics_history 
            if m.timestamp > cutoff_time
        ]
    
    async def _alert_loop(self):
        """Main alerting loop."""
        while True:
            try:
                await self._check_alerts()
                await asyncio.sleep(60)  # Check alerts every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_alerts(self):
        """Check all alert rules and trigger alerts if needed."""
        if not self.system_metrics_history or not self.generation_metrics_history:
            return
        
        current_system = self.system_metrics_history[-1]
        current_generation = self.generation_metrics_history[-1]
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                # Get metric value
                metric_value = self._get_metric_value(rule.metric_path, current_system, current_generation)
                
                # Check if condition is met
                condition_met = self._evaluate_condition(metric_value, rule.threshold, rule.comparison)
                
                if condition_met:
                    # Check if condition has persisted for required duration
                    if await self._check_alert_duration(rule, current_system.timestamp):
                        await self._trigger_alert(rule, metric_value)
                
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    def _get_metric_value(self, metric_path: str, system_metrics: SystemMetrics, generation_metrics: GenerationMetrics) -> float:
        """Get metric value from metrics objects."""
        parts = metric_path.split('.')
        
        if parts[0] == 'system':
            obj = system_metrics
        elif parts[0] == 'generation':
            obj = generation_metrics
        else:
            raise ValueError(f"Unknown metric category: {parts[0]}")
        
        # Navigate to the metric value
        for part in parts[1:]:
            obj = getattr(obj, part)
        
        return float(obj)
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition."""
        if comparison == 'gt':
            return value > threshold
        elif comparison == 'lt':
            return value < threshold
        elif comparison == 'gte':
            return value >= threshold
        elif comparison == 'lte':
            return value <= threshold
        elif comparison == 'eq':
            return value == threshold
        else:
            raise ValueError(f"Unknown comparison operator: {comparison}")
    
    async def _check_alert_duration(self, rule: AlertRule, current_time: float) -> bool:
        """Check if alert condition has persisted for required duration."""
        # For simplicity, we'll just check if enough time has passed since last trigger
        if rule.last_triggered is None:
            return True
        
        return current_time - rule.last_triggered > rule.cooldown
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """Trigger an alert."""
        rule.last_triggered = time.time()
        
        alert_data = {
            'rule_name': rule.name,
            'severity': rule.severity,
            'metric_path': rule.metric_path,
            'current_value': metric_value,
            'threshold': rule.threshold,
            'timestamp': rule.last_triggered,
            'message': f"{rule.name}: {rule.metric_path} is {metric_value} (threshold: {rule.threshold})"
        }
        
        self.logger.warning(f"ALERT: {alert_data['message']}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Store alert in Redis
        if self.redis_client:
            try:
                alert_key = f"alerts:{int(rule.last_triggered)}"
                await self.redis_client.hset(alert_key, mapping=alert_data)
                await self.redis_client.expire(alert_key, 604800)  # 7 days
            except Exception as e:
                self.logger.error(f"Failed to store alert in Redis: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules.append(rule)
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule by name."""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
    
    async def record_request_start(self, request_id: str, model_name: str):
        """Record the start of a generation request."""
        self.request_start_times[request_id] = time.time()
        
        # Update model usage counter
        self.model_usage_counters[model_name] = self.model_usage_counters.get(model_name, 0) + 1
        
        # Update Prometheus counter
        self.generation_requests_total.labels(model=model_name, status='started').inc()
        
        # Store in Redis
        if self.redis_client:
            await self.redis_client.incr('metrics:total_requests')
            await self.redis_client.incr(f'metrics:model_usage:{model_name}')
    
    async def record_request_completion(self, request_id: str, model_name: str, success: bool, quality_score: Optional[float] = None):
        """Record the completion of a generation request."""
        if request_id in self.request_start_times:
            duration = time.time() - self.request_start_times[request_id]
            del self.request_start_times[request_id]
            
            # Update Prometheus metrics
            status = 'completed' if success else 'failed'
            self.generation_requests_total.labels(model=model_name, status=status).inc()
            self.generation_duration_histogram.labels(model=model_name).observe(duration)
            
            if quality_score is not None:
                self.quality_score_histogram.labels(model=model_name).observe(quality_score)
            
            # Store in Redis
            if self.redis_client:
                if success:
                    await self.redis_client.incr('metrics:completed_requests')
                else:
                    await self.redis_client.incr('metrics:failed_requests')
                
                # Store generation time
                await self.redis_client.lpush('metrics:generation_times', duration)
                await self.redis_client.ltrim('metrics:generation_times', 0, 999)  # Keep last 1000
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.system_metrics_history or not self.generation_metrics_history:
            return {}
        
        current_system = self.system_metrics_history[-1]
        current_generation = self.generation_metrics_history[-1]
        
        return {
            'timestamp': current_system.timestamp,
            'system': {
                'cpu_usage_percent': current_system.cpu_usage_percent,
                'memory_usage_percent': current_system.memory_usage_percent,
                'gpu_usage_percent': current_system.gpu_usage_percent,
                'disk_usage_percent': current_system.disk_usage_percent,
                'load_average': current_system.load_average
            },
            'generation': {
                'total_requests': current_generation.total_requests,
                'completed_requests': current_generation.completed_requests,
                'failed_requests': current_generation.failed_requests,
                'queue_length': current_generation.queue_length,
                'success_rate': current_generation.success_rate,
                'error_rate': current_generation.error_rate,
                'average_generation_time': current_generation.average_generation_time,
                'cost_per_hour': current_generation.cost_per_hour
            }
        }
    
    async def get_metrics_history(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        system_history = [
            {
                'timestamp': m.timestamp,
                'cpu_usage_percent': m.cpu_usage_percent,
                'memory_usage_percent': m.memory_usage_percent,
                'gpu_usage_percent': m.gpu_usage_percent,
                'disk_usage_percent': m.disk_usage_percent
            }
            for m in self.system_metrics_history
            if m.timestamp > cutoff_time
        ]
        
        generation_history = [
            {
                'timestamp': m.timestamp,
                'total_requests': m.total_requests,
                'completed_requests': m.completed_requests,
                'failed_requests': m.failed_requests,
                'queue_length': m.queue_length,
                'success_rate': m.success_rate,
                'error_rate': m.error_rate
            }
            for m in self.generation_metrics_history
            if m.timestamp > cutoff_time
        ]
        
        return {
            'system': system_history,
            'generation': generation_history
        }
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.prometheus_registry).decode('utf-8')
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        active_alerts = []
        
        if self.redis_client:
            try:
                # Get recent alerts from Redis
                alert_keys = await self.redis_client.keys('alerts:*')
                for key in alert_keys[-10:]:  # Get last 10 alerts
                    alert_data = await self.redis_client.hgetall(key)
                    if alert_data:
                        active_alerts.append(alert_data)
            except Exception as e:
                self.logger.error(f"Failed to get alerts from Redis: {e}")
        
        return active_alerts
    
    async def cleanup(self):
        """Clean up metrics collector resources."""
        await self.stop_monitoring()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.system_metrics_history.clear()
        self.generation_metrics_history.clear()
        self.request_start_times.clear()
        self.model_usage_counters.clear()
        
        self.logger.info("Metrics collector cleaned up")