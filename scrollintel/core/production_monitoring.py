"""
Production Monitoring and Alerting System for Bulletproof User Experience

This module provides comprehensive production monitoring with predictive alerts,
user experience quality monitoring, failure pattern detection, and automated
optimization capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque
import psutil
import time

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MetricType(Enum):
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    USER_EXPERIENCE = "user_experience"
    SYSTEM_HEALTH = "system_health"
    FAILURE_PATTERN = "failure_pattern"

@dataclass
class SystemMetric:
    """System metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class FailurePattern:
    """Failure pattern detection result"""
    pattern_id: str
    pattern_type: str
    frequency: int
    last_occurrence: datetime
    confidence: float
    description: str
    suggested_actions: List[str]

class ProductionMonitor:
    """
    Production monitoring system with predictive alerts and user experience tracking
    """
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)
        self.alerts = {}
        self.alert_rules = {}
        self.failure_patterns = {}
        self.user_sessions = {}
        self.system_baselines = {}
        self.prediction_models = {}
        self.monitoring_active = False
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
    def _setup_default_alert_rules(self):
        """Setup default monitoring and alert rules"""
        self.alert_rules = {
            "response_time_high": {
                "metric": "response_time",
                "threshold": 2000,  # 2 seconds
                "operator": ">",
                "severity": AlertSeverity.HIGH,
                "description": "Response time exceeding 2 seconds"
            },
            "error_rate_high": {
                "metric": "error_rate",
                "threshold": 0.05,  # 5%
                "operator": ">",
                "severity": AlertSeverity.CRITICAL,
                "description": "Error rate exceeding 5%"
            },
            "cpu_usage_high": {
                "metric": "cpu_usage",
                "threshold": 80,  # 80%
                "operator": ">",
                "severity": AlertSeverity.MEDIUM,
                "description": "CPU usage exceeding 80%"
            },
            "memory_usage_high": {
                "metric": "memory_usage",
                "threshold": 85,  # 85%
                "operator": ">",
                "severity": AlertSeverity.MEDIUM,
                "description": "Memory usage exceeding 85%"
            },
            "user_satisfaction_low": {
                "metric": "user_satisfaction",
                "threshold": 0.8,  # 80%
                "operator": "<",
                "severity": AlertSeverity.HIGH,
                "description": "User satisfaction below 80%"
            }
        }
    
    async def start_monitoring(self):
        """Start the production monitoring system"""
        self.monitoring_active = True
        logger.info("Production monitoring system started")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._monitor_user_experience()),
            asyncio.create_task(self._detect_failure_patterns()),
            asyncio.create_task(self._process_alerts()),
            asyncio.create_task(self._generate_predictions())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """Stop the production monitoring system"""
        self.monitoring_active = False
        logger.info("Production monitoring system stopped")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self._add_metric("cpu_usage", cpu_percent, "%", timestamp, MetricType.SYSTEM_HEALTH)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self._add_metric("memory_usage", memory.percent, "%", timestamp, MetricType.SYSTEM_HEALTH)
                self._add_metric("memory_available", memory.available / (1024**3), "GB", timestamp, MetricType.SYSTEM_HEALTH)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self._add_metric("disk_usage", disk_percent, "%", timestamp, MetricType.SYSTEM_HEALTH)
                
                # Network metrics (if available)
                try:
                    network = psutil.net_io_counters()
                    self._add_metric("network_bytes_sent", network.bytes_sent, "bytes", timestamp, MetricType.SYSTEM_HEALTH)
                    self._add_metric("network_bytes_recv", network.bytes_recv, "bytes", timestamp, MetricType.SYSTEM_HEALTH)
                except:
                    pass
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_user_experience(self):
        """Monitor user experience quality metrics"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # Calculate user experience metrics
                active_sessions = len(self.user_sessions)
                self._add_metric("active_sessions", active_sessions, "count", timestamp, MetricType.USER_EXPERIENCE)
                
                # Calculate average session satisfaction
                if self.user_sessions:
                    satisfactions = [session.get('satisfaction', 0.8) for session in self.user_sessions.values()]
                    avg_satisfaction = statistics.mean(satisfactions)
                    self._add_metric("user_satisfaction", avg_satisfaction, "score", timestamp, MetricType.USER_EXPERIENCE)
                
                # Monitor response times from recent metrics
                recent_response_times = self._get_recent_metrics("response_time", minutes=5)
                if recent_response_times:
                    avg_response_time = statistics.mean([m.value for m in recent_response_times])
                    self._add_metric("avg_response_time", avg_response_time, "ms", timestamp, MetricType.PERFORMANCE)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error monitoring user experience: {e}")
                await asyncio.sleep(120)
    
    async def _detect_failure_patterns(self):
        """Detect failure patterns and anomalies"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # Analyze error patterns
                error_metrics = self._get_recent_metrics("error_rate", hours=1)
                if len(error_metrics) >= 10:
                    error_rates = [m.value for m in error_metrics]
                    
                    # Detect increasing error trend
                    if self._detect_increasing_trend(error_rates):
                        pattern = FailurePattern(
                            pattern_id=f"error_trend_{int(timestamp.timestamp())}",
                            pattern_type="increasing_errors",
                            frequency=len(error_metrics),
                            last_occurrence=timestamp,
                            confidence=0.8,
                            description="Increasing error rate trend detected",
                            suggested_actions=[
                                "Check recent deployments",
                                "Review system logs",
                                "Monitor resource usage"
                            ]
                        )
                        self.failure_patterns[pattern.pattern_id] = pattern
                
                # Detect performance degradation patterns
                response_time_metrics = self._get_recent_metrics("response_time", hours=2)
                if len(response_time_metrics) >= 20:
                    response_times = [m.value for m in response_time_metrics]
                    
                    # Detect performance degradation
                    if self._detect_performance_degradation(response_times):
                        pattern = FailurePattern(
                            pattern_id=f"perf_degradation_{int(timestamp.timestamp())}",
                            pattern_type="performance_degradation",
                            frequency=len(response_time_metrics),
                            last_occurrence=timestamp,
                            confidence=0.75,
                            description="Performance degradation pattern detected",
                            suggested_actions=[
                                "Scale resources",
                                "Optimize database queries",
                                "Check for memory leaks"
                            ]
                        )
                        self.failure_patterns[pattern.pattern_id] = pattern
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error detecting failure patterns: {e}")
                await asyncio.sleep(600)
    
    async def _process_alerts(self):
        """Process and trigger alerts based on rules"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for rule_name, rule in self.alert_rules.items():
                    # Get recent metrics for this rule
                    recent_metrics = self._get_recent_metrics(rule["metric"], minutes=5)
                    
                    if recent_metrics:
                        latest_metric = recent_metrics[-1]
                        
                        # Check if alert condition is met
                        if self._evaluate_alert_condition(latest_metric.value, rule):
                            # Check if alert already exists and is not resolved
                            existing_alert = None
                            for alert in self.alerts.values():
                                if (alert.metric_name == rule["metric"] and 
                                    not alert.resolved and
                                    (current_time - alert.timestamp).seconds < 3600):  # Within last hour
                                    existing_alert = alert
                                    break
                            
                            if not existing_alert:
                                # Create new alert
                                alert_id = f"{rule_name}_{int(current_time.timestamp())}"
                                alert = Alert(
                                    id=alert_id,
                                    title=f"{rule['description']}",
                                    description=f"Metric {rule['metric']} value {latest_metric.value} {rule['operator']} threshold {rule['threshold']}",
                                    severity=rule["severity"],
                                    metric_name=rule["metric"],
                                    current_value=latest_metric.value,
                                    threshold=rule["threshold"],
                                    timestamp=current_time
                                )
                                
                                self.alerts[alert_id] = alert
                                await self._trigger_alert(alert)
                        
                        else:
                            # Check if we should resolve existing alerts
                            for alert in self.alerts.values():
                                if (alert.metric_name == rule["metric"] and 
                                    not alert.resolved and
                                    not self._evaluate_alert_condition(latest_metric.value, rule)):
                                    alert.resolved = True
                                    alert.resolution_time = current_time
                                    await self._resolve_alert(alert)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(120)
    
    async def _generate_predictions(self):
        """Generate predictive alerts based on trends"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # Predict resource exhaustion
                await self._predict_resource_exhaustion()
                
                # Predict performance issues
                await self._predict_performance_issues()
                
                # Predict user experience degradation
                await self._predict_ux_degradation()
                
                await asyncio.sleep(600)  # Generate predictions every 10 minutes
                
            except Exception as e:
                logger.error(f"Error generating predictions: {e}")
                await asyncio.sleep(1200)
    
    def _add_metric(self, name: str, value: float, unit: str, timestamp: datetime, metric_type: MetricType, tags: Dict[str, str] = None):
        """Add a metric to the buffer"""
        metric = SystemMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            metric_type=metric_type,
            tags=tags or {}
        )
        self.metrics_buffer.append(metric)
    
    def _get_recent_metrics(self, metric_name: str, minutes: int = None, hours: int = None) -> List[SystemMetric]:
        """Get recent metrics for a specific metric name"""
        if minutes:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
        elif hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
        else:
            cutoff_time = datetime.now() - timedelta(minutes=10)
        
        return [
            metric for metric in self.metrics_buffer
            if metric.name == metric_name and metric.timestamp >= cutoff_time
        ]
    
    def _evaluate_alert_condition(self, value: float, rule: Dict) -> bool:
        """Evaluate if an alert condition is met"""
        operator = rule["operator"]
        threshold = rule["threshold"]
        
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        
        return False
    
    def _detect_increasing_trend(self, values: List[float], threshold: float = 0.1) -> bool:
        """Detect if there's an increasing trend in values"""
        if len(values) < 5:
            return False
        
        # Simple trend detection using linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return False
        
        slope = numerator / denominator
        return slope > threshold
    
    def _detect_performance_degradation(self, response_times: List[float]) -> bool:
        """Detect performance degradation pattern"""
        if len(response_times) < 10:
            return False
        
        # Compare recent average with historical average
        recent_avg = statistics.mean(response_times[-5:])
        historical_avg = statistics.mean(response_times[:-5])
        
        # Consider degradation if recent average is 50% higher
        return recent_avg > historical_avg * 1.5
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        logger.warning(f"ALERT TRIGGERED: {alert.title} - {alert.description}")
        
        # Here you would integrate with your alerting system
        # (email, Slack, PagerDuty, etc.)
        
        # For now, just log the alert
        alert_data = {
            "alert_id": alert.id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "metric": alert.metric_name,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "timestamp": alert.timestamp.isoformat()
        }
        
        logger.info(f"Alert data: {json.dumps(alert_data, indent=2)}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        logger.info(f"ALERT RESOLVED: {alert.title}")
        
        # Log resolution
        resolution_data = {
            "alert_id": alert.id,
            "title": alert.title,
            "resolved_at": alert.resolution_time.isoformat(),
            "duration": (alert.resolution_time - alert.timestamp).total_seconds()
        }
        
        logger.info(f"Alert resolution: {json.dumps(resolution_data, indent=2)}")
    
    async def _predict_resource_exhaustion(self):
        """Predict when resources might be exhausted"""
        try:
            # Predict memory exhaustion
            memory_metrics = self._get_recent_metrics("memory_usage", hours=2)
            if len(memory_metrics) >= 10:
                memory_values = [m.value for m in memory_metrics]
                if self._detect_increasing_trend(memory_values, threshold=0.05):
                    # Predict when memory will reach 95%
                    current_memory = memory_values[-1]
                    if current_memory > 70:  # Only predict if already above 70%
                        alert = Alert(
                            id=f"memory_exhaustion_prediction_{int(time.time())}",
                            title="Predicted Memory Exhaustion",
                            description=f"Memory usage trending upward. Current: {current_memory:.1f}%. Predicted to reach 95% soon.",
                            severity=AlertSeverity.HIGH,
                            metric_name="memory_usage",
                            current_value=current_memory,
                            threshold=95.0,
                            timestamp=datetime.now(),
                            tags={"type": "prediction"}
                        )
                        
                        self.alerts[alert.id] = alert
                        await self._trigger_alert(alert)
            
            # Similar predictions for CPU and disk
            cpu_metrics = self._get_recent_metrics("cpu_usage", hours=1)
            if len(cpu_metrics) >= 10:
                cpu_values = [m.value for m in cpu_metrics]
                if self._detect_increasing_trend(cpu_values, threshold=0.1):
                    current_cpu = cpu_values[-1]
                    if current_cpu > 60:
                        alert = Alert(
                            id=f"cpu_exhaustion_prediction_{int(time.time())}",
                            title="Predicted CPU Exhaustion",
                            description=f"CPU usage trending upward. Current: {current_cpu:.1f}%. Consider scaling resources.",
                            severity=AlertSeverity.MEDIUM,
                            metric_name="cpu_usage",
                            current_value=current_cpu,
                            threshold=90.0,
                            timestamp=datetime.now(),
                            tags={"type": "prediction"}
                        )
                        
                        self.alerts[alert.id] = alert
                        await self._trigger_alert(alert)
        
        except Exception as e:
            logger.error(f"Error predicting resource exhaustion: {e}")
    
    async def _predict_performance_issues(self):
        """Predict potential performance issues"""
        try:
            response_time_metrics = self._get_recent_metrics("avg_response_time", hours=1)
            if len(response_time_metrics) >= 10:
                response_times = [m.value for m in response_time_metrics]
                
                if self._detect_increasing_trend(response_times, threshold=10):  # 10ms increase trend
                    current_response_time = response_times[-1]
                    
                    alert = Alert(
                        id=f"performance_degradation_prediction_{int(time.time())}",
                        title="Predicted Performance Degradation",
                        description=f"Response time trending upward. Current: {current_response_time:.0f}ms. Consider optimization.",
                        severity=AlertSeverity.MEDIUM,
                        metric_name="avg_response_time",
                        current_value=current_response_time,
                        threshold=2000.0,
                        timestamp=datetime.now(),
                        tags={"type": "prediction"}
                    )
                    
                    self.alerts[alert.id] = alert
                    await self._trigger_alert(alert)
        
        except Exception as e:
            logger.error(f"Error predicting performance issues: {e}")
    
    async def _predict_ux_degradation(self):
        """Predict user experience degradation"""
        try:
            satisfaction_metrics = self._get_recent_metrics("user_satisfaction", hours=2)
            if len(satisfaction_metrics) >= 5:
                satisfaction_values = [m.value for m in satisfaction_metrics]
                
                # Check for declining satisfaction
                if len(satisfaction_values) >= 3:
                    recent_trend = satisfaction_values[-3:]
                    if all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                        current_satisfaction = satisfaction_values[-1]
                        
                        alert = Alert(
                            id=f"ux_degradation_prediction_{int(time.time())}",
                            title="Predicted User Experience Degradation",
                            description=f"User satisfaction declining. Current: {current_satisfaction:.2f}. Review user feedback.",
                            severity=AlertSeverity.HIGH,
                            metric_name="user_satisfaction",
                            current_value=current_satisfaction,
                            threshold=0.8,
                            timestamp=datetime.now(),
                            tags={"type": "prediction"}
                        )
                        
                        self.alerts[alert.id] = alert
                        await self._trigger_alert(alert)
        
        except Exception as e:
            logger.error(f"Error predicting UX degradation: {e}")
    
    # Public API methods
    
    def record_user_action(self, user_id: str, action: str, response_time: float, success: bool, satisfaction: float = None):
        """Record a user action for monitoring"""
        timestamp = datetime.now()
        
        # Update user session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'start_time': timestamp,
                'actions': [],
                'satisfaction': satisfaction or 0.8
            }
        
        self.user_sessions[user_id]['actions'].append({
            'action': action,
            'response_time': response_time,
            'success': success,
            'timestamp': timestamp
        })
        
        if satisfaction:
            self.user_sessions[user_id]['satisfaction'] = satisfaction
        
        # Record metrics
        self._add_metric("response_time", response_time, "ms", timestamp, MetricType.PERFORMANCE, {"user_id": user_id})
        self._add_metric("action_success", 1.0 if success else 0.0, "bool", timestamp, MetricType.USER_EXPERIENCE, {"user_id": user_id})
        
        # Calculate error rate
        recent_actions = self._get_recent_metrics("action_success", minutes=5)
        if recent_actions:
            error_rate = 1.0 - (sum(m.value for m in recent_actions) / len(recent_actions))
            self._add_metric("error_rate", error_rate, "rate", timestamp, MetricType.AVAILABILITY)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        current_time = datetime.now()
        
        # Get recent metrics
        recent_metrics = {}
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage", "error_rate", "avg_response_time", "user_satisfaction"]:
            metrics = self._get_recent_metrics(metric_name, minutes=5)
            if metrics:
                recent_metrics[metric_name] = {
                    "current": metrics[-1].value,
                    "average": statistics.mean([m.value for m in metrics]),
                    "trend": "increasing" if len(metrics) > 1 and metrics[-1].value > metrics[0].value else "stable"
                }
        
        # Count active alerts
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        alert_counts = defaultdict(int)
        for alert in active_alerts:
            alert_counts[alert.severity.value] += 1
        
        # Overall health score (0-100)
        health_score = self._calculate_health_score(recent_metrics)
        
        return {
            "timestamp": current_time.isoformat(),
            "health_score": health_score,
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 60 else "critical",
            "metrics": recent_metrics,
            "active_alerts": len(active_alerts),
            "alert_breakdown": dict(alert_counts),
            "failure_patterns": len(self.failure_patterns),
            "active_sessions": len(self.user_sessions)
        }
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        score = 100.0
        
        # Deduct points for high resource usage
        if "cpu_usage" in metrics and metrics["cpu_usage"]["current"] > 80:
            score -= (metrics["cpu_usage"]["current"] - 80) * 0.5
        
        if "memory_usage" in metrics and metrics["memory_usage"]["current"] > 85:
            score -= (metrics["memory_usage"]["current"] - 85) * 0.7
        
        # Deduct points for high error rate
        if "error_rate" in metrics and metrics["error_rate"]["current"] > 0.01:
            score -= metrics["error_rate"]["current"] * 1000  # 1% error = 10 points
        
        # Deduct points for slow response times
        if "avg_response_time" in metrics and metrics["avg_response_time"]["current"] > 1000:
            score -= (metrics["avg_response_time"]["current"] - 1000) * 0.01
        
        # Deduct points for low user satisfaction
        if "user_satisfaction" in metrics and metrics["user_satisfaction"]["current"] < 0.8:
            score -= (0.8 - metrics["user_satisfaction"]["current"]) * 50
        
        return max(0.0, min(100.0, score))
    
    def get_alerts(self, resolved: bool = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by resolution status"""
        alerts = list(self.alerts.values())
        
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        return [asdict(alert) for alert in sorted(alerts, key=lambda x: x.timestamp, reverse=True)]
    
    def get_failure_patterns(self) -> List[Dict[str, Any]]:
        """Get detected failure patterns"""
        return [asdict(pattern) for pattern in self.failure_patterns.values()]
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
        
        summary = {}
        for metric_type in MetricType:
            type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
            
            if type_metrics:
                # Group by metric name
                by_name = defaultdict(list)
                for metric in type_metrics:
                    by_name[metric.name].append(metric.value)
                
                summary[metric_type.value] = {}
                for name, values in by_name.items():
                    summary[metric_type.value][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "average": statistics.mean(values),
                        "latest": values[-1] if values else None
                    }
        
        return summary

# Global production monitor instance
production_monitor = ProductionMonitor()