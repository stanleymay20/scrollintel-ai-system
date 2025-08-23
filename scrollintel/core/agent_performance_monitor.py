"""
Real-Time Agent Performance Monitoring System

This module provides comprehensive real-time monitoring of agent performance,
business impact tracking, and automated alerting capabilities for the
Agent Steering System.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics being tracked"""
    PERFORMANCE = "performance"
    BUSINESS_IMPACT = "business_impact"
    SYSTEM_HEALTH = "system_health"
    AGENT_ACTIVITY = "agent_activity"


@dataclass
class AgentMetrics:
    """Real-time agent performance metrics"""
    agent_id: str
    timestamp: datetime
    response_time: float
    throughput: float
    accuracy: float
    reliability: float
    resource_utilization: Dict[str, float]
    business_impact: Dict[str, float]
    error_rate: float
    success_rate: float
    active_tasks: int
    completed_tasks: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class BusinessImpactMetrics:
    """Business impact tracking metrics"""
    timestamp: datetime
    cost_savings: float
    revenue_increase: float
    risk_reduction: float
    productivity_gain: float
    customer_satisfaction: float
    compliance_score: float
    roi_percentage: float
    time_to_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Alert:
    """System alert definition"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    agent_id: Optional[str]
    metric_type: MetricType
    threshold_value: float
    actual_value: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['metric_type'] = self.metric_type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolution_time:
            data['resolution_time'] = self.resolution_time.isoformat()
        return data


class RealTimeMetricsCollector:
    """Collects and processes real-time metrics from agents"""
    
    def __init__(self, redis_client: redis.Redis, db_connection: psycopg2.extensions.connection):
        self.redis_client = redis_client
        self.db_connection = db_connection
        self.metrics_buffer = defaultdict(deque)
        self.prometheus_registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        self.agent_response_time = Histogram(
            'agent_response_time_seconds',
            'Agent response time in seconds',
            ['agent_id'],
            registry=self.prometheus_registry
        )
        
        self.agent_throughput = Gauge(
            'agent_throughput_requests_per_second',
            'Agent throughput in requests per second',
            ['agent_id'],
            registry=self.prometheus_registry
        )
        
        self.agent_accuracy = Gauge(
            'agent_accuracy_percentage',
            'Agent accuracy percentage',
            ['agent_id'],
            registry=self.prometheus_registry
        )
        
        self.business_cost_savings = Gauge(
            'business_cost_savings_dollars',
            'Business cost savings in dollars',
            registry=self.prometheus_registry
        )
        
        self.business_roi = Gauge(
            'business_roi_percentage',
            'Business ROI percentage',
            registry=self.prometheus_registry
        )
        
        self.active_alerts = Gauge(
            'active_alerts_count',
            'Number of active alerts',
            ['severity'],
            registry=self.prometheus_registry
        )
    
    async def collect_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Collect real-time metrics for a specific agent"""
        try:
            # Get metrics from Redis cache (real-time data)
            metrics_key = f"agent_metrics:{agent_id}"
            cached_metrics = await self._get_cached_metrics(metrics_key)
            
            if cached_metrics:
                # Update Prometheus metrics
                self.agent_response_time.labels(agent_id=agent_id).observe(cached_metrics.response_time)
                self.agent_throughput.labels(agent_id=agent_id).set(cached_metrics.throughput)
                self.agent_accuracy.labels(agent_id=agent_id).set(cached_metrics.accuracy)
                
                # Store in buffer for batch processing
                self.metrics_buffer[agent_id].append(cached_metrics)
                
                # Keep only last 1000 metrics per agent
                if len(self.metrics_buffer[agent_id]) > 1000:
                    self.metrics_buffer[agent_id].popleft()
                
                return cached_metrics
            
            # If no cached metrics, collect from agent directly
            return await self._collect_direct_metrics(agent_id)
            
        except Exception as e:
            logger.error(f"Error collecting metrics for agent {agent_id}: {e}")
            raise
    
    async def _get_cached_metrics(self, metrics_key: str) -> Optional[AgentMetrics]:
        """Get metrics from Redis cache"""
        try:
            cached_data = self.redis_client.get(metrics_key)
            if cached_data:
                data = json.loads(cached_data)
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                return AgentMetrics(**data)
            return None
        except Exception as e:
            logger.error(f"Error getting cached metrics: {e}")
            return None
    
    async def _collect_direct_metrics(self, agent_id: str) -> AgentMetrics:
        """Collect metrics directly from agent"""
        # This would integrate with actual agent monitoring
        # For now, return sample metrics
        return AgentMetrics(
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            response_time=0.5,
            throughput=100.0,
            accuracy=95.0,
            reliability=99.5,
            resource_utilization={"cpu": 45.0, "memory": 60.0, "disk": 30.0},
            business_impact={"cost_savings": 1000.0, "productivity": 25.0},
            error_rate=0.5,
            success_rate=99.5,
            active_tasks=5,
            completed_tasks=150
        )
    
    async def store_metrics(self, metrics: AgentMetrics):
        """Store metrics in database and cache"""
        try:
            # Store in Redis for real-time access
            metrics_key = f"agent_metrics:{metrics.agent_id}"
            self.redis_client.setex(
                metrics_key,
                300,  # 5 minutes TTL
                json.dumps(metrics.to_dict())
            )
            
            # Store in database for historical analysis
            await self._store_metrics_db(metrics)
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
            raise
    
    async def _store_metrics_db(self, metrics: AgentMetrics):
        """Store metrics in PostgreSQL database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO agent_metrics (
                        agent_id, timestamp, response_time, throughput, accuracy,
                        reliability, resource_utilization, business_impact,
                        error_rate, success_rate, active_tasks, completed_tasks
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    metrics.agent_id,
                    metrics.timestamp,
                    metrics.response_time,
                    metrics.throughput,
                    metrics.accuracy,
                    metrics.reliability,
                    json.dumps(metrics.resource_utilization),
                    json.dumps(metrics.business_impact),
                    metrics.error_rate,
                    metrics.success_rate,
                    metrics.active_tasks,
                    metrics.completed_tasks
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing metrics in database: {e}")
            self.db_connection.rollback()
            raise


class BusinessImpactTracker:
    """Tracks and calculates business impact metrics and ROI"""
    
    def __init__(self, redis_client: redis.Redis, db_connection: psycopg2.extensions.connection):
        self.redis_client = redis_client
        self.db_connection = db_connection
        self.impact_history = deque(maxlen=10000)
    
    async def calculate_roi(self, time_period: timedelta = timedelta(days=30)) -> Dict[str, float]:
        """Calculate ROI for specified time period"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_period
            
            # Get business impact data from database
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        SUM(cost_savings) as total_cost_savings,
                        SUM(revenue_increase) as total_revenue_increase,
                        AVG(productivity_gain) as avg_productivity_gain,
                        AVG(customer_satisfaction) as avg_customer_satisfaction,
                        COUNT(*) as measurement_count
                    FROM business_impact_metrics 
                    WHERE timestamp >= %s AND timestamp <= %s
                """, (start_time, end_time))
                
                result = cursor.fetchone()
                
                if result and result['measurement_count'] > 0:
                    # Calculate total benefits
                    total_benefits = (result['total_cost_savings'] or 0) + (result['total_revenue_increase'] or 0)
                    
                    # Estimate system costs (this would come from actual cost tracking)
                    estimated_costs = await self._estimate_system_costs(time_period)
                    
                    # Calculate ROI
                    roi_percentage = ((total_benefits - estimated_costs) / estimated_costs) * 100 if estimated_costs > 0 else 0
                    
                    return {
                        "roi_percentage": roi_percentage,
                        "total_benefits": total_benefits,
                        "total_costs": estimated_costs,
                        "cost_savings": result['total_cost_savings'] or 0,
                        "revenue_increase": result['total_revenue_increase'] or 0,
                        "productivity_gain": result['avg_productivity_gain'] or 0,
                        "customer_satisfaction": result['avg_customer_satisfaction'] or 0,
                        "measurement_period_days": time_period.days
                    }
                
                return {"roi_percentage": 0, "total_benefits": 0, "total_costs": 0}
                
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            raise
    
    async def _estimate_system_costs(self, time_period: timedelta) -> float:
        """Estimate system operational costs for the time period"""
        # This would integrate with actual cost tracking systems
        # For now, return estimated costs based on infrastructure
        daily_cost = 500.0  # Estimated daily operational cost
        return daily_cost * time_period.days
    
    async def track_business_impact(self, impact_metrics: BusinessImpactMetrics):
        """Track and store business impact metrics"""
        try:
            # Store in Redis for real-time access
            impact_key = f"business_impact:{impact_metrics.timestamp.isoformat()}"
            self.redis_client.setex(
                impact_key,
                3600,  # 1 hour TTL
                json.dumps(impact_metrics.to_dict())
            )
            
            # Store in database
            await self._store_impact_db(impact_metrics)
            
            # Add to history buffer
            self.impact_history.append(impact_metrics)
            
        except Exception as e:
            logger.error(f"Error tracking business impact: {e}")
            raise
    
    async def _store_impact_db(self, impact_metrics: BusinessImpactMetrics):
        """Store business impact metrics in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO business_impact_metrics (
                        timestamp, cost_savings, revenue_increase, risk_reduction,
                        productivity_gain, customer_satisfaction, compliance_score,
                        roi_percentage, time_to_value
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    impact_metrics.timestamp,
                    impact_metrics.cost_savings,
                    impact_metrics.revenue_increase,
                    impact_metrics.risk_reduction,
                    impact_metrics.productivity_gain,
                    impact_metrics.customer_satisfaction,
                    impact_metrics.compliance_score,
                    impact_metrics.roi_percentage,
                    impact_metrics.time_to_value
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing business impact in database: {e}")
            self.db_connection.rollback()
            raise
    
    async def get_real_time_impact(self) -> Dict[str, float]:
        """Get real-time business impact summary"""
        try:
            if not self.impact_history:
                return {"cost_savings": 0, "revenue_increase": 0, "productivity_gain": 0}
            
            # Calculate rolling averages from recent data
            recent_metrics = list(self.impact_history)[-100:]  # Last 100 measurements
            
            total_cost_savings = sum(m.cost_savings for m in recent_metrics)
            total_revenue_increase = sum(m.revenue_increase for m in recent_metrics)
            avg_productivity_gain = np.mean([m.productivity_gain for m in recent_metrics])
            avg_customer_satisfaction = np.mean([m.customer_satisfaction for m in recent_metrics])
            
            return {
                "cost_savings": total_cost_savings,
                "revenue_increase": total_revenue_increase,
                "productivity_gain": float(avg_productivity_gain),
                "customer_satisfaction": float(avg_customer_satisfaction),
                "measurement_count": len(recent_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time impact: {e}")
            return {"cost_savings": 0, "revenue_increase": 0, "productivity_gain": 0}


class AlertingSystem:
    """Automated alerting system for performance degradation and failures"""
    
    def __init__(self, redis_client: redis.Redis, db_connection: psycopg2.extensions.connection):
        self.redis_client = redis_client
        self.db_connection = db_connection
        self.alert_rules = {}
        self.active_alerts = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        self.alert_rules = {
            "response_time_high": {
                "metric": "response_time",
                "threshold": 2.0,
                "operator": ">",
                "severity": AlertSeverity.HIGH,
                "description": "Agent response time exceeds 2 seconds"
            },
            "accuracy_low": {
                "metric": "accuracy",
                "threshold": 90.0,
                "operator": "<",
                "severity": AlertSeverity.MEDIUM,
                "description": "Agent accuracy below 90%"
            },
            "error_rate_high": {
                "metric": "error_rate",
                "threshold": 5.0,
                "operator": ">",
                "severity": AlertSeverity.HIGH,
                "description": "Agent error rate exceeds 5%"
            },
            "roi_negative": {
                "metric": "roi_percentage",
                "threshold": 0.0,
                "operator": "<",
                "severity": AlertSeverity.CRITICAL,
                "description": "Negative ROI detected"
            }
        }
    
    async def check_metrics_for_alerts(self, metrics: AgentMetrics) -> List[Alert]:
        """Check metrics against alert rules and generate alerts"""
        alerts = []
        
        try:
            for rule_name, rule in self.alert_rules.items():
                metric_value = getattr(metrics, rule["metric"], None)
                
                if metric_value is not None:
                    should_alert = self._evaluate_rule(metric_value, rule)
                    
                    if should_alert:
                        alert = Alert(
                            id=f"{rule_name}_{metrics.agent_id}_{int(time.time())}",
                            severity=rule["severity"],
                            title=f"Agent {metrics.agent_id}: {rule['description']}",
                            description=f"Metric {rule['metric']} value {metric_value} {rule['operator']} threshold {rule['threshold']}",
                            timestamp=datetime.utcnow(),
                            agent_id=metrics.agent_id,
                            metric_type=MetricType.PERFORMANCE,
                            threshold_value=rule["threshold"],
                            actual_value=metric_value
                        )
                        
                        alerts.append(alert)
                        await self._store_alert(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking metrics for alerts: {e}")
            return []
    
    def _evaluate_rule(self, value: float, rule: Dict[str, Any]) -> bool:
        """Evaluate if a metric value triggers an alert rule"""
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
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database and cache"""
        try:
            # Store in Redis for real-time access
            alert_key = f"alert:{alert.id}"
            self.redis_client.setex(
                alert_key,
                86400,  # 24 hours TTL
                json.dumps(alert.to_dict())
            )
            
            # Add to active alerts
            self.active_alerts[alert.id] = alert
            
            # Store in database
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO alerts (
                        id, severity, title, description, timestamp, agent_id,
                        metric_type, threshold_value, actual_value, resolved
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    alert.id,
                    alert.severity.value,
                    alert.title,
                    alert.description,
                    alert.timestamp,
                    alert.agent_id,
                    alert.metric_type.value,
                    alert.threshold_value,
                    alert.actual_value,
                    alert.resolved
                ))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            self.db_connection.rollback()
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
                
                # Update in database
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        UPDATE alerts 
                        SET resolved = %s, resolution_time = %s 
                        WHERE id = %s
                    """, (True, alert.resolution_time, alert_id))
                    self.db_connection.commit()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts, optionally filtered by severity"""
        try:
            alerts = list(self.active_alerts.values())
            
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []


class ExecutiveReportingEngine:
    """Generates executive reports with quantified business value metrics"""
    
    def __init__(self, 
                 metrics_collector: RealTimeMetricsCollector,
                 impact_tracker: BusinessImpactTracker,
                 alerting_system: AlertingSystem):
        self.metrics_collector = metrics_collector
        self.impact_tracker = impact_tracker
        self.alerting_system = alerting_system
    
    async def generate_executive_summary(self, time_period: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Generate comprehensive executive summary report"""
        try:
            # Get ROI and business impact data
            roi_data = await self.impact_tracker.calculate_roi(time_period)
            real_time_impact = await self.impact_tracker.get_real_time_impact()
            
            # Get system health summary
            health_summary = await self._get_system_health_summary()
            
            # Get alert summary
            alert_summary = await self._get_alert_summary()
            
            # Calculate key performance indicators
            kpis = await self._calculate_kpis(time_period)
            
            return {
                "report_generated": datetime.utcnow().isoformat(),
                "time_period_days": time_period.days,
                "executive_summary": {
                    "roi_percentage": roi_data.get("roi_percentage", 0),
                    "total_cost_savings": roi_data.get("cost_savings", 0),
                    "total_revenue_increase": roi_data.get("revenue_increase", 0),
                    "system_health_score": health_summary.get("overall_score", 0),
                    "active_critical_alerts": alert_summary.get("critical_count", 0),
                    "agent_performance_score": kpis.get("avg_performance_score", 0)
                },
                "business_impact": {
                    "roi_analysis": roi_data,
                    "real_time_metrics": real_time_impact,
                    "productivity_improvements": kpis.get("productivity_metrics", {}),
                    "cost_optimization": kpis.get("cost_metrics", {})
                },
                "system_performance": {
                    "health_summary": health_summary,
                    "agent_metrics": kpis.get("agent_metrics", {}),
                    "infrastructure_utilization": kpis.get("infrastructure_metrics", {})
                },
                "alerts_and_issues": {
                    "alert_summary": alert_summary,
                    "resolved_issues": alert_summary.get("resolved_count", 0),
                    "mean_resolution_time": alert_summary.get("mean_resolution_time", 0)
                },
                "recommendations": await self._generate_recommendations(roi_data, health_summary, alert_summary)
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            raise
    
    async def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        try:
            # This would integrate with actual system monitoring
            # For now, return sample health data
            return {
                "overall_score": 95.5,
                "agent_availability": 99.2,
                "response_time_avg": 0.8,
                "error_rate": 0.3,
                "resource_utilization": {
                    "cpu": 65.0,
                    "memory": 70.0,
                    "disk": 45.0,
                    "network": 30.0
                }
            }
        except Exception as e:
            logger.error(f"Error getting system health summary: {e}")
            return {"overall_score": 0}
    
    async def _get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts and their resolution status"""
        try:
            active_alerts = await self.alerting_system.get_active_alerts()
            
            # Count alerts by severity
            severity_counts = {
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                "low": len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            }
            
            return {
                "total_active": len(active_alerts),
                "critical_count": severity_counts["critical"],
                "high_count": severity_counts["high"],
                "medium_count": severity_counts["medium"],
                "low_count": severity_counts["low"],
                "severity_distribution": severity_counts,
                "resolved_count": 0,  # Would come from database query
                "mean_resolution_time": 0  # Would be calculated from resolved alerts
            }
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {"total_active": 0, "critical_count": 0}
    
    async def _calculate_kpis(self, time_period: timedelta) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        try:
            # This would calculate actual KPIs from collected metrics
            # For now, return sample KPI data
            return {
                "avg_performance_score": 92.5,
                "productivity_metrics": {
                    "tasks_completed": 1250,
                    "avg_completion_time": 45.2,
                    "efficiency_improvement": 23.5
                },
                "cost_metrics": {
                    "infrastructure_costs": 15000.0,
                    "operational_savings": 45000.0,
                    "cost_per_transaction": 2.35
                },
                "agent_metrics": {
                    "total_agents": 12,
                    "active_agents": 11,
                    "avg_utilization": 78.5,
                    "peak_concurrent_tasks": 156
                },
                "infrastructure_metrics": {
                    "uptime_percentage": 99.95,
                    "avg_response_time": 0.65,
                    "throughput_rps": 2500.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating KPIs: {e}")
            return {"avg_performance_score": 0}
    
    async def _generate_recommendations(self, 
                                     roi_data: Dict[str, Any], 
                                     health_summary: Dict[str, Any], 
                                     alert_summary: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on current metrics"""
        recommendations = []
        
        try:
            # ROI-based recommendations
            roi_percentage = roi_data.get("roi_percentage", 0)
            if roi_percentage < 10:
                recommendations.append("Consider optimizing agent workflows to improve ROI - current ROI is below target of 10%")
            elif roi_percentage > 50:
                recommendations.append("Excellent ROI performance - consider scaling successful agent configurations")
            
            # Health-based recommendations
            health_score = health_summary.get("overall_score", 0)
            if health_score < 90:
                recommendations.append("System health below optimal - investigate performance bottlenecks")
            
            # Alert-based recommendations
            critical_alerts = alert_summary.get("critical_count", 0)
            if critical_alerts > 0:
                recommendations.append(f"Address {critical_alerts} critical alerts immediately to prevent system degradation")
            
            # Resource utilization recommendations
            resource_util = health_summary.get("resource_utilization", {})
            cpu_util = resource_util.get("cpu", 0)
            if cpu_util > 80:
                recommendations.append("High CPU utilization detected - consider scaling infrastructure")
            elif cpu_util < 30:
                recommendations.append("Low CPU utilization - opportunity to optimize resource allocation")
            
            if not recommendations:
                recommendations.append("System performing optimally - continue monitoring for sustained performance")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to system error"]


class AgentSteeringMonitoringSystem:
    """Main monitoring system that coordinates all monitoring components"""
    
    def __init__(self, redis_url: str, database_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.db_connection = psycopg2.connect(database_url)
        
        # Initialize components
        self.metrics_collector = RealTimeMetricsCollector(self.redis_client, self.db_connection)
        self.impact_tracker = BusinessImpactTracker(self.redis_client, self.db_connection)
        self.alerting_system = AlertingSystem(self.redis_client, self.db_connection)
        self.reporting_engine = ExecutiveReportingEngine(
            self.metrics_collector, 
            self.impact_tracker, 
            self.alerting_system
        )
        
        self._monitoring_active = False
        self._monitoring_task = None
    
    async def start_monitoring(self, collection_interval: int = 30):
        """Start the real-time monitoring system"""
        if self._monitoring_active:
            logger.warning("Monitoring system already active")
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(collection_interval)
        )
        logger.info("Agent Steering monitoring system started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent Steering monitoring system stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Get list of active agents (this would come from agent registry)
                active_agents = await self._get_active_agents()
                
                # Collect metrics for each agent
                for agent_id in active_agents:
                    try:
                        metrics = await self.metrics_collector.collect_agent_metrics(agent_id)
                        await self.metrics_collector.store_metrics(metrics)
                        
                        # Check for alerts
                        alerts = await self.alerting_system.check_metrics_for_alerts(metrics)
                        if alerts:
                            logger.info(f"Generated {len(alerts)} alerts for agent {agent_id}")
                        
                    except Exception as e:
                        logger.error(f"Error monitoring agent {agent_id}: {e}")
                
                # Calculate and track business impact
                await self._update_business_impact()
                
                # Wait for next collection interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _get_active_agents(self) -> List[str]:
        """Get list of active agent IDs"""
        # This would integrate with the actual agent registry
        # For now, return sample agent IDs
        return ["agent_001", "agent_002", "agent_003", "agent_004"]
    
    async def _update_business_impact(self):
        """Update business impact metrics"""
        try:
            # Calculate current business impact
            impact_metrics = BusinessImpactMetrics(
                timestamp=datetime.utcnow(),
                cost_savings=1500.0,  # Would be calculated from actual data
                revenue_increase=2500.0,
                risk_reduction=15.0,
                productivity_gain=25.0,
                customer_satisfaction=4.2,
                compliance_score=98.5,
                roi_percentage=35.0,
                time_to_value=2.5
            )
            
            await self.impact_tracker.track_business_impact(impact_metrics)
            
        except Exception as e:
            logger.error(f"Error updating business impact: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            # Get real-time metrics
            real_time_impact = await self.impact_tracker.get_real_time_impact()
            active_alerts = await self.alerting_system.get_active_alerts()
            
            # Get recent ROI data
            roi_data = await self.impact_tracker.calculate_roi(timedelta(days=7))
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "business_impact": real_time_impact,
                "roi_metrics": roi_data,
                "active_alerts": [alert.to_dict() for alert in active_alerts[:10]],  # Latest 10 alerts
                "alert_counts": {
                    "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                    "high": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                    "medium": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                    "low": len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
                },
                "system_status": "operational"  # Would be calculated from actual health checks
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    async def generate_executive_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate executive report for specified number of days"""
        try:
            time_period = timedelta(days=days)
            return await self.reporting_engine.generate_executive_summary(time_period)
        except Exception as e:
            logger.error(f"Error generating executive report: {e}")
            return {"error": str(e)}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        try:
            return generate_latest(self.metrics_collector.prometheus_registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Error getting Prometheus metrics: {e}")
            return ""
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.stop_monitoring()
            self.redis_client.close()
            self.db_connection.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")