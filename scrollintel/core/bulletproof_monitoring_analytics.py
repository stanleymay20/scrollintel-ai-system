"""
Comprehensive monitoring and analytics system for bulletproof user experience.
Provides real-time monitoring, failure pattern analysis, user satisfaction tracking,
and system health visualization with predictive alerts.
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
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_HEALTH = "system_health"
    FAILURE_RATE = "failure_rate"

@dataclass
class UserExperienceMetric:
    """Represents a user experience metric data point."""
    timestamp: datetime
    user_id: str
    metric_type: MetricType
    value: float
    context: Dict[str, Any]
    session_id: Optional[str] = None
    component: Optional[str] = None

@dataclass
class FailurePattern:
    """Represents a detected failure pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    severity: AlertSeverity
    components_affected: List[str]
    first_occurrence: datetime
    last_occurrence: datetime
    root_cause: Optional[str] = None
    mitigation_applied: bool = False

@dataclass
class SystemHealthSnapshot:
    """Represents a system health snapshot."""
    timestamp: datetime
    overall_health_score: float
    component_health: Dict[str, float]
    active_issues: List[str]
    performance_metrics: Dict[str, float]
    user_satisfaction_score: float
    failure_rate: float
    recovery_time: float

@dataclass
class PredictiveAlert:
    """Represents a predictive alert."""
    alert_id: str
    severity: AlertSeverity
    predicted_issue: str
    probability: float
    time_to_occurrence: timedelta
    affected_components: List[str]
    recommended_actions: List[str]
    created_at: datetime

class BulletproofMonitoringAnalytics:
    """
    Comprehensive monitoring and analytics system for bulletproof user experience.
    """
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)
        self.failure_patterns = {}
        self.health_history = deque(maxlen=1000)
        self.user_satisfaction_scores = defaultdict(list)
        self.active_alerts = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.error_rates = deque(maxlen=1000)
        self.throughput_metrics = deque(maxlen=1000)
        
        # User experience tracking
        self.user_journeys = defaultdict(list)
        self.satisfaction_feedback = deque(maxlen=5000)
        
        # System health tracking
        self.component_status = {}
        self.resource_utilization = defaultdict(deque)
        
    async def record_metric(self, metric: UserExperienceMetric) -> None:
        """Record a user experience metric."""
        try:
            self.metrics_buffer.append(metric)
            
            # Update specific metric collections
            if metric.metric_type == MetricType.PERFORMANCE:
                self.response_times.append(metric.value)
            elif metric.metric_type == MetricType.FAILURE_RATE:
                self.error_rates.append(metric.value)
            elif metric.metric_type == MetricType.USER_SATISFACTION:
                self.user_satisfaction_scores[metric.user_id].append(metric.value)
                self.satisfaction_feedback.append(metric)
            
            # Analyze for patterns and anomalies
            await self._analyze_metric_patterns(metric)
            
            logger.debug(f"Recorded metric: {metric.metric_type.value} = {metric.value}")
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    async def _analyze_metric_patterns(self, metric: UserExperienceMetric) -> None:
        """Analyze metrics for patterns and anomalies."""
        try:
            # Check for anomalies if we have enough data
            if len(self.metrics_buffer) >= 100:
                await self._detect_anomalies()
            
            # Check for failure patterns
            if metric.metric_type == MetricType.FAILURE_RATE and metric.value > 0.1:
                await self._analyze_failure_patterns(metric)
            
            # Check for performance degradation
            if metric.metric_type == MetricType.PERFORMANCE and metric.value > 2000:  # 2 seconds
                await self._analyze_performance_degradation(metric)
                
        except Exception as e:
            logger.error(f"Error analyzing metric patterns: {e}")
    
    async def _detect_anomalies(self) -> None:
        """Detect anomalies in metrics using machine learning."""
        try:
            # Prepare data for anomaly detection
            recent_metrics = list(self.metrics_buffer)[-100:]
            if len(recent_metrics) < 50:
                return
            
            # Extract features
            features = []
            for metric in recent_metrics:
                feature_vector = [
                    metric.value,
                    len(metric.context),
                    hash(metric.metric_type.value) % 1000,
                    metric.timestamp.hour,
                    metric.timestamp.weekday()
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Train or update anomaly detector
            if not self.is_trained:
                scaled_features = self.scaler.fit_transform(features_array)
                self.anomaly_detector.fit(scaled_features)
                self.is_trained = True
            else:
                scaled_features = self.scaler.transform(features_array)
                anomalies = self.anomaly_detector.predict(scaled_features)
                
                # Process detected anomalies
                for i, is_anomaly in enumerate(anomalies):
                    if is_anomaly == -1:  # Anomaly detected
                        metric = recent_metrics[i]
                        await self._handle_anomaly(metric)
                        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
    
    async def _analyze_failure_patterns(self, metric: UserExperienceMetric) -> None:
        """Analyze failure patterns and trends."""
        try:
            component = metric.component or "unknown"
            pattern_key = f"{component}_{metric.metric_type.value}"
            
            current_time = datetime.now()
            
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = FailurePattern(
                    pattern_id=pattern_key,
                    pattern_type="failure_rate_spike",
                    frequency=1,
                    severity=AlertSeverity.MEDIUM,
                    components_affected=[component],
                    first_occurrence=current_time,
                    last_occurrence=current_time
                )
            else:
                pattern = self.failure_patterns[pattern_key]
                pattern.frequency += 1
                pattern.last_occurrence = current_time
                
                # Escalate severity based on frequency
                if pattern.frequency > 10:
                    pattern.severity = AlertSeverity.HIGH
                if pattern.frequency > 20:
                    pattern.severity = AlertSeverity.CRITICAL
            
            # Generate alert if pattern is significant
            pattern = self.failure_patterns[pattern_key]
            if pattern.frequency > 5 and not pattern.mitigation_applied:
                await self._generate_failure_alert(pattern)
                
        except Exception as e:
            logger.error(f"Error analyzing failure patterns: {e}")
    
    async def _analyze_performance_degradation(self, metric: UserExperienceMetric) -> None:
        """Analyze performance degradation patterns."""
        try:
            if len(self.response_times) < 10:
                return
            
            recent_avg = statistics.mean(list(self.response_times)[-10:])
            overall_avg = statistics.mean(self.response_times)
            
            # Check for significant performance degradation
            if recent_avg > overall_avg * 1.5:  # 50% worse than average
                alert = PredictiveAlert(
                    alert_id=f"perf_degradation_{datetime.now().isoformat()}",
                    severity=AlertSeverity.HIGH,
                    predicted_issue="Performance degradation detected",
                    probability=0.8,
                    time_to_occurrence=timedelta(minutes=5),
                    affected_components=[metric.component or "system"],
                    recommended_actions=[
                        "Scale resources",
                        "Check system load",
                        "Review recent deployments"
                    ],
                    created_at=datetime.now()
                )
                
                await self._handle_predictive_alert(alert)
                
        except Exception as e:
            logger.error(f"Error analyzing performance degradation: {e}")
    
    async def _handle_anomaly(self, metric: UserExperienceMetric) -> None:
        """Handle detected anomaly."""
        try:
            alert = PredictiveAlert(
                alert_id=f"anomaly_{datetime.now().isoformat()}",
                severity=AlertSeverity.MEDIUM,
                predicted_issue=f"Anomaly detected in {metric.metric_type.value}",
                probability=0.7,
                time_to_occurrence=timedelta(minutes=1),
                affected_components=[metric.component or "system"],
                recommended_actions=[
                    "Investigate metric anomaly",
                    "Check system logs",
                    "Monitor user impact"
                ],
                created_at=datetime.now()
            )
            
            await self._handle_predictive_alert(alert)
            
        except Exception as e:
            logger.error(f"Error handling anomaly: {e}")
    
    async def _generate_failure_alert(self, pattern: FailurePattern) -> None:
        """Generate alert for failure pattern."""
        try:
            alert = PredictiveAlert(
                alert_id=f"failure_pattern_{pattern.pattern_id}",
                severity=pattern.severity,
                predicted_issue=f"Failure pattern detected: {pattern.pattern_type}",
                probability=0.9,
                time_to_occurrence=timedelta(minutes=2),
                affected_components=pattern.components_affected,
                recommended_actions=[
                    "Apply automatic mitigation",
                    "Scale affected components",
                    "Investigate root cause"
                ],
                created_at=datetime.now()
            )
            
            await self._handle_predictive_alert(alert)
            
        except Exception as e:
            logger.error(f"Error generating failure alert: {e}")
    
    async def _handle_predictive_alert(self, alert: PredictiveAlert) -> None:
        """Handle predictive alert."""
        try:
            self.active_alerts[alert.alert_id] = alert
            
            # Log alert
            logger.warning(f"Predictive alert: {alert.predicted_issue} "
                         f"(Severity: {alert.severity.value}, "
                         f"Probability: {alert.probability})")
            
            # Trigger automated responses based on severity
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                await self._trigger_automated_response(alert)
                
        except Exception as e:
            logger.error(f"Error handling predictive alert: {e}")
    
    async def _trigger_automated_response(self, alert: PredictiveAlert) -> None:
        """Trigger automated response to alert."""
        try:
            # Implement automated responses based on alert type
            if "performance" in alert.predicted_issue.lower():
                # Trigger performance optimization
                logger.info("Triggering performance optimization")
            elif "failure" in alert.predicted_issue.lower():
                # Trigger failure mitigation
                logger.info("Triggering failure mitigation")
            
            # Mark as handled
            alert.recommended_actions.append("Automated response triggered")
            
        except Exception as e:
            logger.error(f"Error triggering automated response: {e}")
    
    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data."""
        try:
            current_time = datetime.now()
            
            # Calculate current metrics
            recent_metrics = [m for m in self.metrics_buffer 
                            if (current_time - m.timestamp).seconds < 300]  # Last 5 minutes
            
            performance_metrics = [m for m in recent_metrics 
                                 if m.metric_type == MetricType.PERFORMANCE]
            
            avg_response_time = (statistics.mean([m.value for m in performance_metrics]) 
                               if performance_metrics else 0)
            
            # User satisfaction
            recent_satisfaction = [m for m in recent_metrics 
                                 if m.metric_type == MetricType.USER_SATISFACTION]
            avg_satisfaction = (statistics.mean([m.value for m in recent_satisfaction]) 
                              if recent_satisfaction else 0)
            
            # System health
            health_score = await self._calculate_system_health_score()
            
            # Active issues
            active_issues = len([a for a in self.active_alerts.values() 
                               if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]])
            
            return {
                "timestamp": current_time.isoformat(),
                "metrics": {
                    "avg_response_time": avg_response_time,
                    "user_satisfaction": avg_satisfaction,
                    "system_health_score": health_score,
                    "active_critical_issues": active_issues,
                    "total_users_active": len(set(m.user_id for m in recent_metrics)),
                    "requests_per_minute": len(recent_metrics)
                },
                "alerts": [asdict(alert) for alert in list(self.active_alerts.values())[-10:]],
                "trends": await self._get_trend_data(),
                "component_health": await self._get_component_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    async def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score."""
        try:
            scores = []
            
            # Performance score (0-100)
            if self.response_times:
                avg_response = statistics.mean(self.response_times)
                perf_score = max(0, 100 - (avg_response / 50))  # 50ms = 100 points
                scores.append(perf_score)
            
            # Reliability score (0-100)
            if self.error_rates:
                avg_error_rate = statistics.mean(self.error_rates)
                reliability_score = max(0, 100 - (avg_error_rate * 1000))
                scores.append(reliability_score)
            
            # User satisfaction score (0-100)
            if self.satisfaction_feedback:
                recent_satisfaction = [f.value for f in list(self.satisfaction_feedback)[-100:]]
                satisfaction_score = statistics.mean(recent_satisfaction) * 20  # Scale to 0-100
                scores.append(satisfaction_score)
            
            # Alert penalty
            critical_alerts = len([a for a in self.active_alerts.values() 
                                 if a.severity == AlertSeverity.CRITICAL])
            high_alerts = len([a for a in self.active_alerts.values() 
                             if a.severity == AlertSeverity.HIGH])
            
            alert_penalty = (critical_alerts * 20) + (high_alerts * 10)
            
            overall_score = statistics.mean(scores) if scores else 100
            overall_score = max(0, overall_score - alert_penalty)
            
            return round(overall_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating system health score: {e}")
            return 0.0
    
    async def _get_trend_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get trend data for visualization."""
        try:
            current_time = datetime.now()
            
            # Get hourly trends for the last 24 hours
            trends = {
                "response_time": [],
                "error_rate": [],
                "user_satisfaction": [],
                "system_health": []
            }
            
            for hour in range(24):
                hour_start = current_time - timedelta(hours=hour+1)
                hour_end = current_time - timedelta(hours=hour)
                
                hour_metrics = [m for m in self.metrics_buffer 
                              if hour_start <= m.timestamp < hour_end]
                
                # Response time trend
                perf_metrics = [m.value for m in hour_metrics 
                              if m.metric_type == MetricType.PERFORMANCE]
                avg_response = statistics.mean(perf_metrics) if perf_metrics else 0
                
                trends["response_time"].append({
                    "timestamp": hour_start.isoformat(),
                    "value": avg_response
                })
                
                # Error rate trend
                error_metrics = [m.value for m in hour_metrics 
                               if m.metric_type == MetricType.FAILURE_RATE]
                avg_error = statistics.mean(error_metrics) if error_metrics else 0
                
                trends["error_rate"].append({
                    "timestamp": hour_start.isoformat(),
                    "value": avg_error
                })
                
                # User satisfaction trend
                satisfaction_metrics = [m.value for m in hour_metrics 
                                      if m.metric_type == MetricType.USER_SATISFACTION]
                avg_satisfaction = statistics.mean(satisfaction_metrics) if satisfaction_metrics else 0
                
                trends["user_satisfaction"].append({
                    "timestamp": hour_start.isoformat(),
                    "value": avg_satisfaction
                })
            
            # Reverse to get chronological order
            for trend_type in trends:
                trends[trend_type].reverse()
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting trend data: {e}")
            return {}
    
    async def _get_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Get component health status."""
        try:
            components = {}
            
            # Analyze metrics by component
            for metric in list(self.metrics_buffer)[-1000:]:  # Last 1000 metrics
                component = metric.component or "system"
                
                if component not in components:
                    components[component] = {
                        "health_score": 100,
                        "response_time": [],
                        "error_rate": [],
                        "last_updated": metric.timestamp.isoformat()
                    }
                
                if metric.metric_type == MetricType.PERFORMANCE:
                    components[component]["response_time"].append(metric.value)
                elif metric.metric_type == MetricType.FAILURE_RATE:
                    components[component]["error_rate"].append(metric.value)
                
                components[component]["last_updated"] = metric.timestamp.isoformat()
            
            # Calculate health scores
            for component, data in components.items():
                health_score = 100
                
                if data["response_time"]:
                    avg_response = statistics.mean(data["response_time"])
                    health_score -= min(50, avg_response / 40)  # Penalty for slow response
                
                if data["error_rate"]:
                    avg_error = statistics.mean(data["error_rate"])
                    health_score -= min(50, avg_error * 500)  # Penalty for errors
                
                data["health_score"] = max(0, round(health_score, 2))
                
                # Simplify data for response
                data["avg_response_time"] = (statistics.mean(data["response_time"]) 
                                           if data["response_time"] else 0)
                data["avg_error_rate"] = (statistics.mean(data["error_rate"]) 
                                        if data["error_rate"] else 0)
                
                # Remove raw data arrays
                del data["response_time"]
                del data["error_rate"]
            
            return components
            
        except Exception as e:
            logger.error(f"Error getting component health: {e}")
            return {}
    
    async def analyze_user_satisfaction_patterns(self) -> Dict[str, Any]:
        """Analyze user satisfaction patterns and trends."""
        try:
            if not self.satisfaction_feedback:
                return {"message": "No satisfaction data available"}
            
            # Convert to DataFrame for analysis
            satisfaction_data = []
            for feedback in self.satisfaction_feedback:
                satisfaction_data.append({
                    "timestamp": feedback.timestamp,
                    "user_id": feedback.user_id,
                    "score": feedback.value,
                    "context": feedback.context
                })
            
            df = pd.DataFrame(satisfaction_data)
            
            # Overall statistics
            overall_stats = {
                "mean_satisfaction": df["score"].mean(),
                "median_satisfaction": df["score"].median(),
                "std_satisfaction": df["score"].std(),
                "total_responses": len(df)
            }
            
            # Satisfaction by time of day
            df["hour"] = df["timestamp"].dt.hour
            hourly_satisfaction = df.groupby("hour")["score"].mean().to_dict()
            
            # Satisfaction trends over time
            df["date"] = df["timestamp"].dt.date
            daily_satisfaction = df.groupby("date")["score"].mean().to_dict()
            
            # User satisfaction distribution
            satisfaction_distribution = df["score"].value_counts().sort_index().to_dict()
            
            # Identify patterns
            patterns = []
            
            # Low satisfaction periods
            low_satisfaction_hours = [hour for hour, score in hourly_satisfaction.items() 
                                    if score < overall_stats["mean_satisfaction"] - overall_stats["std_satisfaction"]]
            if low_satisfaction_hours:
                patterns.append({
                    "type": "low_satisfaction_hours",
                    "description": f"Lower satisfaction during hours: {low_satisfaction_hours}",
                    "recommendation": "Investigate system performance during these hours"
                })
            
            # Declining satisfaction trend
            recent_scores = df.tail(100)["score"].tolist()
            if len(recent_scores) >= 20:
                recent_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                if recent_trend < -0.01:  # Declining trend
                    patterns.append({
                        "type": "declining_satisfaction",
                        "description": "User satisfaction is declining",
                        "recommendation": "Immediate investigation required"
                    })
            
            return {
                "overall_stats": overall_stats,
                "hourly_satisfaction": hourly_satisfaction,
                "daily_satisfaction": {str(k): v for k, v in daily_satisfaction.items()},
                "satisfaction_distribution": satisfaction_distribution,
                "patterns": patterns,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user satisfaction patterns: {e}")
            return {"error": str(e)}
    
    async def get_failure_pattern_analysis(self) -> Dict[str, Any]:
        """Get comprehensive failure pattern analysis."""
        try:
            if not self.failure_patterns:
                return {"message": "No failure patterns detected"}
            
            # Analyze failure patterns
            patterns_analysis = []
            
            for pattern_id, pattern in self.failure_patterns.items():
                pattern_data = asdict(pattern)
                
                # Calculate pattern metrics
                duration = (pattern.last_occurrence - pattern.first_occurrence).total_seconds()
                frequency_per_hour = pattern.frequency / max(1, duration / 3600)
                
                pattern_data.update({
                    "duration_hours": duration / 3600,
                    "frequency_per_hour": frequency_per_hour,
                    "is_active": (datetime.now() - pattern.last_occurrence).total_seconds() < 3600,
                    "impact_score": self._calculate_pattern_impact_score(pattern)
                })
                
                patterns_analysis.append(pattern_data)
            
            # Sort by impact score
            patterns_analysis.sort(key=lambda x: x["impact_score"], reverse=True)
            
            # Generate insights
            insights = []
            
            # Most frequent patterns
            most_frequent = max(self.failure_patterns.values(), key=lambda p: p.frequency)
            insights.append({
                "type": "most_frequent_pattern",
                "description": f"Most frequent failure pattern: {most_frequent.pattern_type}",
                "frequency": most_frequent.frequency,
                "components": most_frequent.components_affected
            })
            
            # Critical patterns
            critical_patterns = [p for p in self.failure_patterns.values() 
                               if p.severity == AlertSeverity.CRITICAL]
            if critical_patterns:
                insights.append({
                    "type": "critical_patterns",
                    "description": f"{len(critical_patterns)} critical failure patterns detected",
                    "patterns": [p.pattern_id for p in critical_patterns]
                })
            
            # Recent patterns
            recent_patterns = [p for p in self.failure_patterns.values() 
                             if (datetime.now() - p.last_occurrence).total_seconds() < 3600]
            if recent_patterns:
                insights.append({
                    "type": "recent_patterns",
                    "description": f"{len(recent_patterns)} failure patterns active in the last hour",
                    "patterns": [p.pattern_id for p in recent_patterns]
                })
            
            return {
                "patterns": patterns_analysis,
                "insights": insights,
                "total_patterns": len(self.failure_patterns),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing failure patterns: {e}")
            return {"error": str(e)}
    
    def _calculate_pattern_impact_score(self, pattern: FailurePattern) -> float:
        """Calculate impact score for a failure pattern."""
        try:
            # Base score from frequency
            frequency_score = min(100, pattern.frequency * 2)
            
            # Severity multiplier
            severity_multipliers = {
                AlertSeverity.LOW: 1.0,
                AlertSeverity.MEDIUM: 1.5,
                AlertSeverity.HIGH: 2.0,
                AlertSeverity.CRITICAL: 3.0
            }
            severity_multiplier = severity_multipliers.get(pattern.severity, 1.0)
            
            # Component count multiplier
            component_multiplier = 1 + (len(pattern.components_affected) * 0.2)
            
            # Recency multiplier (more recent = higher impact)
            hours_since_last = (datetime.now() - pattern.last_occurrence).total_seconds() / 3600
            recency_multiplier = max(0.1, 2.0 - (hours_since_last / 24))
            
            impact_score = frequency_score * severity_multiplier * component_multiplier * recency_multiplier
            
            return round(impact_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating pattern impact score: {e}")
            return 0.0
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        try:
            current_time = datetime.now()
            
            # System overview
            system_health_score = await self._calculate_system_health_score()
            
            # Performance metrics
            performance_summary = {
                "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0,
                "p95_response_time": np.percentile(self.response_times, 95) if self.response_times else 0,
                "p99_response_time": np.percentile(self.response_times, 99) if self.response_times else 0,
                "avg_error_rate": statistics.mean(self.error_rates) if self.error_rates else 0
            }
            
            # User satisfaction summary
            satisfaction_summary = {
                "avg_satisfaction": 0,
                "total_feedback": len(self.satisfaction_feedback),
                "satisfaction_trend": "stable"
            }
            
            if self.satisfaction_feedback:
                recent_satisfaction = [f.value for f in list(self.satisfaction_feedback)[-100:]]
                satisfaction_summary["avg_satisfaction"] = statistics.mean(recent_satisfaction)
                
                # Calculate trend
                if len(recent_satisfaction) >= 20:
                    trend = np.polyfit(range(len(recent_satisfaction)), recent_satisfaction, 1)[0]
                    if trend > 0.01:
                        satisfaction_summary["satisfaction_trend"] = "improving"
                    elif trend < -0.01:
                        satisfaction_summary["satisfaction_trend"] = "declining"
            
            # Alert summary
            alert_summary = {
                "total_active_alerts": len(self.active_alerts),
                "critical_alerts": len([a for a in self.active_alerts.values() 
                                      if a.severity == AlertSeverity.CRITICAL]),
                "high_alerts": len([a for a in self.active_alerts.values() 
                                  if a.severity == AlertSeverity.HIGH]),
                "medium_alerts": len([a for a in self.active_alerts.values() 
                                    if a.severity == AlertSeverity.MEDIUM])
            }
            
            # Component health summary
            component_health = await self._get_component_health()
            
            # Recommendations
            recommendations = []
            
            if system_health_score < 80:
                recommendations.append("System health is below optimal. Investigate performance issues.")
            
            if performance_summary["avg_response_time"] > 1000:
                recommendations.append("Average response time is high. Consider performance optimization.")
            
            if satisfaction_summary["satisfaction_trend"] == "declining":
                recommendations.append("User satisfaction is declining. Immediate attention required.")
            
            if alert_summary["critical_alerts"] > 0:
                recommendations.append(f"{alert_summary['critical_alerts']} critical alerts require immediate attention.")
            
            return {
                "report_timestamp": current_time.isoformat(),
                "system_health_score": system_health_score,
                "performance_summary": performance_summary,
                "satisfaction_summary": satisfaction_summary,
                "alert_summary": alert_summary,
                "component_health": component_health,
                "recommendations": recommendations,
                "data_points_analyzed": len(self.metrics_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {"error": str(e)}

# Global instance
bulletproof_analytics = BulletproofMonitoringAnalytics()