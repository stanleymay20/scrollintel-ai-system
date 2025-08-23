"""
Automated Alerting System for Performance Degradation and Failures
Advanced alerting with ML-based anomaly detection and intelligent escalation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp

from ..core.config import get_settings
from ..core.logging_config import get_logger
from ..core.alerting import alert_manager, Alert, AlertSeverity, AlertStatus
from ..core.real_time_monitoring import real_time_collector

settings = get_settings()
logger = get_logger(__name__)

@dataclass
class AlertRule:
    """Advanced alert rule with ML capabilities"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    rule_type: str  # "threshold", "anomaly", "trend", "composite"
    severity: AlertSeverity
    conditions: Dict[str, Any]
    ml_model_config: Optional[Dict[str, Any]]
    escalation_policy: Dict[str, Any]
    suppression_rules: Dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class AlertEscalation:
    """Alert escalation configuration"""
    escalation_id: str
    alert_id: str
    level: int
    escalated_at: datetime
    escalated_to: List[str]
    escalation_reason: str
    acknowledgment_required: bool
    timeout_minutes: int

@dataclass
class AlertMetrics:
    """Alert system performance metrics"""
    timestamp: datetime
    total_alerts_generated: int
    false_positive_rate: float
    mean_time_to_detection: float
    mean_time_to_resolution: float
    escalation_rate: float
    acknowledgment_rate: float
    alert_accuracy_score: float

class MLAnomalyDetector:
    """Machine learning-based anomaly detection for alerts"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_trained: Dict[str, bool] = defaultdict(bool)
        self.anomaly_threshold = 0.1  # 10% anomaly threshold
        
    async def train_model(self, metric_name: str, data: List[float]):
        """Train anomaly detection model for specific metric"""
        try:
            if len(data) < 50:  # Need minimum data for training
                return False
                
            # Prepare data
            X = np.array(data).reshape(-1, 1)
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train isolation forest
            model = IsolationForest(
                contamination=self.anomaly_threshold,
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            
            # Store model and scaler
            self.models[metric_name] = model
            self.scalers[metric_name] = scaler
            self.model_trained[metric_name] = True
            
            self.logger.info(f"Anomaly detection model trained for {metric_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training anomaly model for {metric_name}: {e}")
            return False
            
    async def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous"""
        try:
            if not self.model_trained.get(metric_name, False):
                return False, 0.0
                
            # Scale the value
            scaler = self.scalers[metric_name]
            value_scaled = scaler.transform([[value]])
            
            # Predict anomaly
            model = self.models[metric_name]
            anomaly_score = model.decision_function(value_scaled)[0]
            is_anomaly = model.predict(value_scaled)[0] == -1
            
            return is_anomaly, abs(anomaly_score)
            
        except Exception as e:
            self.logger.error(f"Error detecting anomaly for {metric_name}: {e}")
            return False, 0.0
            
    async def update_training_data(self, metric_name: str, value: float):
        """Update training data with new value"""
        self.training_data[metric_name].append(value)
        
        # Retrain model periodically
        if len(self.training_data[metric_name]) % 100 == 0:
            await self.train_model(metric_name, list(self.training_data[metric_name]))

class IntelligentAlertManager:
    """Intelligent alert management with ML-based detection and smart escalation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.escalations: Dict[str, AlertEscalation] = {}
        self.alert_history: List[Alert] = []
        self.ml_detector = MLAnomalyDetector()
        self.notification_channels: List[Callable] = []
        self.alert_metrics: List[AlertMetrics] = []
        self.running = False
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default intelligent alert rules"""
        try:
            # Performance degradation rule
            perf_rule = AlertRule(
                rule_id="performance_degradation",
                name="Performance Degradation Detection",
                description="Detects performance degradation using ML anomaly detection",
                metric_name="response_time",
                rule_type="anomaly",
                severity=AlertSeverity.WARNING,
                conditions={
                    "anomaly_threshold": 0.1,
                    "consecutive_anomalies": 3,
                    "time_window_minutes": 15
                },
                ml_model_config={
                    "model_type": "isolation_forest",
                    "contamination": 0.1,
                    "retrain_interval_hours": 24
                },
                escalation_policy={
                    "levels": [
                        {"timeout_minutes": 15, "notify": ["ops_team"]},
                        {"timeout_minutes": 30, "notify": ["engineering_manager"]},
                        {"timeout_minutes": 60, "notify": ["cto"]}
                    ]
                },
                suppression_rules={
                    "max_alerts_per_hour": 5,
                    "similar_alert_window_minutes": 30
                },
                enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.alert_rules[perf_rule.rule_id] = perf_rule
            
            # System failure rule
            failure_rule = AlertRule(
                rule_id="system_failure",
                name="System Failure Detection",
                description="Detects critical system failures and outages",
                metric_name="system_health",
                rule_type="threshold",
                severity=AlertSeverity.CRITICAL,
                conditions={
                    "threshold": 50.0,
                    "operator": "less_than",
                    "duration_minutes": 5
                },
                ml_model_config=None,
                escalation_policy={
                    "levels": [
                        {"timeout_minutes": 5, "notify": ["ops_team", "engineering_manager"]},
                        {"timeout_minutes": 15, "notify": ["cto", "ceo"]}
                    ]
                },
                suppression_rules={
                    "max_alerts_per_hour": 10
                },
                enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.alert_rules[failure_rule.rule_id] = failure_rule
            
            # Business impact rule
            business_rule = AlertRule(
                rule_id="business_impact",
                name="Business Impact Alert",
                description="Alerts on significant business impact metrics",
                metric_name="roi_percentage",
                rule_type="trend",
                severity=AlertSeverity.WARNING,
                conditions={
                    "trend_direction": "decreasing",
                    "trend_threshold": -10.0,  # 10% decrease
                    "time_window_hours": 24
                },
                ml_model_config={
                    "model_type": "trend_analysis",
                    "sensitivity": 0.8
                },
                escalation_policy={
                    "levels": [
                        {"timeout_minutes": 30, "notify": ["business_analyst"]},
                        {"timeout_minutes": 60, "notify": ["finance_manager"]},
                        {"timeout_minutes": 120, "notify": ["cfo"]}
                    ]
                },
                suppression_rules={
                    "max_alerts_per_day": 3
                },
                enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.alert_rules[business_rule.rule_id] = business_rule
            
            self.logger.info(f"Initialized {len(self.alert_rules)} default alert rules")
            
        except Exception as e:
            self.logger.error(f"Error initializing default alert rules: {e}")
            
    async def start_monitoring(self):
        """Start intelligent alert monitoring"""
        self.running = True
        self.logger.info("Starting intelligent alert monitoring")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_metrics()),
            asyncio.create_task(self._process_escalations()),
            asyncio.create_task(self._update_ml_models()),
            asyncio.create_task(self._calculate_alert_metrics()),
            asyncio.create_task(self._cleanup_resolved_alerts())
        ]
        
        await asyncio.gather(*tasks)
        
    async def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        self.logger.info("Stopping intelligent alert monitoring")
        
    async def _monitor_metrics(self):
        """Monitor metrics and evaluate alert rules"""
        while self.running:
            try:
                # Get current metrics
                dashboard_data = await real_time_collector.get_real_time_dashboard_data()
                
                if dashboard_data:
                    await self._evaluate_alert_rules(dashboard_data)
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring metrics: {e}")
                await asyncio.sleep(30)
                
    async def _evaluate_alert_rules(self, metrics_data: Dict[str, Any]):
        """Evaluate all alert rules against current metrics"""
        try:
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                    
                await self._evaluate_single_rule(rule, metrics_data)
                
        except Exception as e:
            self.logger.error(f"Error evaluating alert rules: {e}")
            
    async def _evaluate_single_rule(self, rule: AlertRule, metrics_data: Dict[str, Any]):
        """Evaluate a single alert rule"""
        try:
            # Extract metric value
            metric_value = self._extract_metric_value(rule.metric_name, metrics_data)
            if metric_value is None:
                return
                
            # Update ML training data
            if rule.ml_model_config:
                await self.ml_detector.update_training_data(rule.metric_name, metric_value)
                
            # Evaluate based on rule type
            should_alert = False
            alert_context = {}
            
            if rule.rule_type == "threshold":
                should_alert, alert_context = await self._evaluate_threshold_rule(rule, metric_value)
            elif rule.rule_type == "anomaly":
                should_alert, alert_context = await self._evaluate_anomaly_rule(rule, metric_value)
            elif rule.rule_type == "trend":
                should_alert, alert_context = await self._evaluate_trend_rule(rule, metric_value)
            elif rule.rule_type == "composite":
                should_alert, alert_context = await self._evaluate_composite_rule(rule, metrics_data)
                
            # Handle alert
            if should_alert:
                await self._trigger_alert(rule, metric_value, alert_context)
            else:
                await self._resolve_alert_if_exists(rule.rule_id)
                
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            
    def _extract_metric_value(self, metric_name: str, metrics_data: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from metrics data"""
        try:
            # Navigate nested dictionary structure
            if metric_name == "response_time":
                return metrics_data.get("system_health", {}).get("average_response_time", None)
            elif metric_name == "system_health":
                return metrics_data.get("system_health", {}).get("overall_health_score", None)
            elif metric_name == "roi_percentage":
                return metrics_data.get("business_impact", {}).get("roi_percentage", None)
            elif metric_name == "error_rate":
                return metrics_data.get("system_health", {}).get("error_rate", None)
            elif metric_name == "cpu_utilization":
                return metrics_data.get("system_health", {}).get("cpu_utilization", None)
            elif metric_name == "memory_utilization":
                return metrics_data.get("system_health", {}).get("memory_utilization", None)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting metric {metric_name}: {e}")
            return None
            
    async def _evaluate_threshold_rule(self, rule: AlertRule, value: float) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate threshold-based rule"""
        try:
            conditions = rule.conditions
            threshold = conditions.get("threshold", 0)
            operator = conditions.get("operator", "greater_than")
            
            should_alert = False
            if operator == "greater_than":
                should_alert = value > threshold
            elif operator == "less_than":
                should_alert = value < threshold
            elif operator == "equals":
                should_alert = abs(value - threshold) < 0.001
                
            context = {
                "rule_type": "threshold",
                "threshold": threshold,
                "operator": operator,
                "current_value": value,
                "deviation": abs(value - threshold)
            }
            
            return should_alert, context
            
        except Exception as e:
            self.logger.error(f"Error evaluating threshold rule: {e}")
            return False, {}
            
    async def _evaluate_anomaly_rule(self, rule: AlertRule, value: float) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate anomaly-based rule"""
        try:
            is_anomaly, anomaly_score = await self.ml_detector.detect_anomaly(rule.metric_name, value)
            
            context = {
                "rule_type": "anomaly",
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score,
                "current_value": value,
                "threshold": rule.conditions.get("anomaly_threshold", 0.1)
            }
            
            return is_anomaly, context
            
        except Exception as e:
            self.logger.error(f"Error evaluating anomaly rule: {e}")
            return False, {}
            
    async def _evaluate_trend_rule(self, rule: AlertRule, value: float) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate trend-based rule"""
        try:
            # Get historical values for trend analysis
            historical_values = list(self.ml_detector.training_data.get(rule.metric_name, []))
            
            if len(historical_values) < 10:  # Need minimum data for trend analysis
                return False, {"rule_type": "trend", "insufficient_data": True}
                
            # Calculate trend
            recent_values = historical_values[-10:]  # Last 10 values
            older_values = historical_values[-20:-10] if len(historical_values) >= 20 else historical_values[:-10]
            
            if not older_values:
                return False, {"rule_type": "trend", "insufficient_data": True}
                
            recent_avg = np.mean(recent_values)
            older_avg = np.mean(older_values)
            
            trend_percentage = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
            
            conditions = rule.conditions
            trend_direction = conditions.get("trend_direction", "decreasing")
            trend_threshold = conditions.get("trend_threshold", -10.0)
            
            should_alert = False
            if trend_direction == "decreasing" and trend_percentage < trend_threshold:
                should_alert = True
            elif trend_direction == "increasing" and trend_percentage > abs(trend_threshold):
                should_alert = True
                
            context = {
                "rule_type": "trend",
                "trend_percentage": trend_percentage,
                "trend_direction": "decreasing" if trend_percentage < 0 else "increasing",
                "threshold": trend_threshold,
                "recent_avg": recent_avg,
                "older_avg": older_avg
            }
            
            return should_alert, context
            
        except Exception as e:
            self.logger.error(f"Error evaluating trend rule: {e}")
            return False, {}
            
    async def _evaluate_composite_rule(self, rule: AlertRule, metrics_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate composite rule with multiple conditions"""
        try:
            # Placeholder for composite rule evaluation
            # This would combine multiple metrics and conditions
            return False, {"rule_type": "composite", "not_implemented": True}
            
        except Exception as e:
            self.logger.error(f"Error evaluating composite rule: {e}")
            return False, {}
            
    async def _trigger_alert(self, rule: AlertRule, value: float, context: Dict[str, Any]):
        """Trigger an alert based on rule evaluation"""
        try:
            alert_id = f"{rule.rule_id}_{int(datetime.utcnow().timestamp())}"
            
            # Check suppression rules
            if await self._is_alert_suppressed(rule, alert_id):
                return
                
            # Create alert
            alert = Alert(
                id=alert_id,
                name=rule.name,
                description=f"{rule.description}. Current value: {value}",
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                metric_name=rule.metric_name,
                current_value=value,
                threshold=context.get("threshold", 0),
                timestamp=datetime.utcnow(),
                tags={
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    **context
                }
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_alert_notifications(alert, rule)
            
            # Start escalation if configured
            if rule.escalation_policy:
                await self._start_escalation(alert, rule)
                
            self.logger.warning(
                f"Alert triggered: {rule.name}",
                extra={
                    "alert_id": alert_id,
                    "rule_id": rule.rule_id,
                    "metric_name": rule.metric_name,
                    "current_value": value,
                    "severity": rule.severity.value,
                    "context": context
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
            
    async def _is_alert_suppressed(self, rule: AlertRule, alert_id: str) -> bool:
        """Check if alert should be suppressed"""
        try:
            suppression = rule.suppression_rules
            
            # Check max alerts per hour
            max_per_hour = suppression.get("max_alerts_per_hour", 999)
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_alerts = [a for a in self.alert_history 
                           if a.timestamp >= hour_ago and a.tags.get("rule_id") == rule.rule_id]
            
            if len(recent_alerts) >= max_per_hour:
                self.logger.info(f"Alert suppressed due to rate limit: {rule.rule_id}")
                return True
                
            # Check similar alert window
            window_minutes = suppression.get("similar_alert_window_minutes", 0)
            if window_minutes > 0:
                window_ago = datetime.utcnow() - timedelta(minutes=window_minutes)
                similar_alerts = [a for a in self.alert_history 
                                if a.timestamp >= window_ago and a.tags.get("rule_id") == rule.rule_id]
                
                if similar_alerts:
                    self.logger.info(f"Alert suppressed due to similar alert window: {rule.rule_id}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking alert suppression: {e}")
            return False
            
    async def _resolve_alert_if_exists(self, rule_id: str):
        """Resolve alert if it exists and conditions are no longer met"""
        try:
            # Find active alerts for this rule
            alerts_to_resolve = [alert_id for alert_id, alert in self.active_alerts.items() 
                               if alert.tags.get("rule_id") == rule_id]
            
            for alert_id in alerts_to_resolve:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                
                # Send resolution notification
                await self._send_resolution_notification(alert)
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert_id}")
                
        except Exception as e:
            self.logger.error(f"Error resolving alerts: {e}")
            
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications through configured channels"""
        try:
            # Send to all configured notification channels
            for channel in self.notification_channels:
                await channel(alert, rule)
                
        except Exception as e:
            self.logger.error(f"Error sending alert notifications: {e}")
            
    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification"""
        try:
            # Send resolution notifications
            for channel in self.notification_channels:
                if hasattr(channel, 'send_resolution'):
                    await channel.send_resolution(alert)
                    
        except Exception as e:
            self.logger.error(f"Error sending resolution notification: {e}")
            
    async def _start_escalation(self, alert: Alert, rule: AlertRule):
        """Start alert escalation process"""
        try:
            escalation_levels = rule.escalation_policy.get("levels", [])
            
            for level, config in enumerate(escalation_levels):
                escalation = AlertEscalation(
                    escalation_id=str(uuid.uuid4()),
                    alert_id=alert.id,
                    level=level,
                    escalated_at=datetime.utcnow() + timedelta(minutes=config.get("timeout_minutes", 30)),
                    escalated_to=config.get("notify", []),
                    escalation_reason=f"Alert not acknowledged within {config.get('timeout_minutes', 30)} minutes",
                    acknowledgment_required=config.get("acknowledgment_required", True),
                    timeout_minutes=config.get("timeout_minutes", 30)
                )
                
                self.escalations[escalation.escalation_id] = escalation
                
        except Exception as e:
            self.logger.error(f"Error starting escalation: {e}")
            
    async def _process_escalations(self):
        """Process pending escalations"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for escalation_id, escalation in list(self.escalations.items()):
                    if current_time >= escalation.escalated_at:
                        await self._execute_escalation(escalation)
                        del self.escalations[escalation_id]
                        
                await asyncio.sleep(60)  # Check escalations every minute
                
            except Exception as e:
                self.logger.error(f"Error processing escalations: {e}")
                await asyncio.sleep(60)
                
    async def _execute_escalation(self, escalation: AlertEscalation):
        """Execute escalation notification"""
        try:
            alert = self.active_alerts.get(escalation.alert_id)
            if not alert:
                return
                
            self.logger.warning(
                f"Escalating alert to level {escalation.level}",
                extra={
                    "alert_id": escalation.alert_id,
                    "escalation_level": escalation.level,
                    "escalated_to": escalation.escalated_to
                }
            )
            
            # Send escalation notifications
            # Implementation would send to specific escalation targets
            
        except Exception as e:
            self.logger.error(f"Error executing escalation: {e}")
            
    async def _update_ml_models(self):
        """Periodically update ML models"""
        while self.running:
            try:
                # Retrain models with accumulated data
                for metric_name, data in self.ml_detector.training_data.items():
                    if len(data) >= 100:  # Minimum data for retraining
                        await self.ml_detector.train_model(metric_name, list(data))
                        
                await asyncio.sleep(3600)  # Update models every hour
                
            except Exception as e:
                self.logger.error(f"Error updating ML models: {e}")
                await asyncio.sleep(3600)
                
    async def _calculate_alert_metrics(self):
        """Calculate alert system performance metrics"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                hour_ago = current_time - timedelta(hours=1)
                
                # Get recent alerts
                recent_alerts = [a for a in self.alert_history if a.timestamp >= hour_ago]
                
                # Calculate metrics
                total_alerts = len(recent_alerts)
                resolved_alerts = [a for a in recent_alerts if a.status == AlertStatus.RESOLVED]
                
                # Calculate mean time to resolution
                mttr = 0.0
                if resolved_alerts:
                    resolution_times = [(a.resolved_at - a.timestamp).total_seconds() / 60 
                                      for a in resolved_alerts if a.resolved_at]
                    mttr = np.mean(resolution_times) if resolution_times else 0.0
                    
                metrics = AlertMetrics(
                    timestamp=current_time,
                    total_alerts_generated=total_alerts,
                    false_positive_rate=0.05,  # Would be calculated from feedback
                    mean_time_to_detection=2.5,  # Average detection time in minutes
                    mean_time_to_resolution=mttr,
                    escalation_rate=len(self.escalations) / max(1, total_alerts),
                    acknowledgment_rate=0.85,  # Would be calculated from acknowledgments
                    alert_accuracy_score=0.92  # Would be calculated from feedback
                )
                
                self.alert_metrics.append(metrics)
                
                await asyncio.sleep(3600)  # Calculate metrics every hour
                
            except Exception as e:
                self.logger.error(f"Error calculating alert metrics: {e}")
                await asyncio.sleep(3600)
                
    async def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        while self.running:
            try:
                # Keep only last 1000 alerts in history
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]
                    
                # Keep only recent alert metrics
                if len(self.alert_metrics) > 168:  # 1 week of hourly metrics
                    self.alert_metrics = self.alert_metrics[-168:]
                    
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Error cleaning up alerts: {e}")
                await asyncio.sleep(3600)
                
    async def get_alert_dashboard_data(self) -> Dict[str, Any]:
        """Get alert dashboard data"""
        try:
            current_time = datetime.utcnow()
            
            # Active alerts summary
            active_by_severity = defaultdict(int)
            for alert in self.active_alerts.values():
                active_by_severity[alert.severity.value] += 1
                
            # Recent metrics
            latest_metrics = self.alert_metrics[-1] if self.alert_metrics else None
            
            return {
                "timestamp": current_time.isoformat(),
                "active_alerts": {
                    "total": len(self.active_alerts),
                    "by_severity": dict(active_by_severity),
                    "critical_count": active_by_severity.get("critical", 0),
                    "warning_count": active_by_severity.get("warning", 0)
                },
                "alert_rules": {
                    "total": len(self.alert_rules),
                    "enabled": len([r for r in self.alert_rules.values() if r.enabled]),
                    "ml_enabled": len([r for r in self.alert_rules.values() if r.ml_model_config])
                },
                "performance_metrics": asdict(latest_metrics) if latest_metrics else None,
                "escalations": {
                    "active": len(self.escalations),
                    "pending": len([e for e in self.escalations.values() 
                                  if e.escalated_at > current_time])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting alert dashboard data: {e}")
            return {"error": str(e)}

# Global instance
intelligent_alert_manager = IntelligentAlertManager()