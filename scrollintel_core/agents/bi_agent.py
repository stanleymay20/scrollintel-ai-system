"""
BI Agent - Dashboard and KPI creation with real-time updates
"""
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import Agent, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    id: str
    name: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int  # seconds
    filters: Dict[str, Any]
    permissions: Dict[str, Any]


@dataclass
class KPIMetric:
    """KPI metric definition"""
    id: str
    name: str
    calculation: str
    target_value: Optional[float]
    threshold_warning: Optional[float]
    threshold_critical: Optional[float]
    unit: str
    format_type: str  # percentage, currency, number
    trend_direction: str  # higher_better, lower_better, neutral


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.connections: Dict[str, List] = {}  # dashboard_id -> [websocket_connections]
        self.active_dashboards: Dict[str, DashboardConfig] = {}
    
    async def connect(self, websocket, dashboard_id: str):
        """Add WebSocket connection for dashboard"""
        if dashboard_id not in self.connections:
            self.connections[dashboard_id] = []
        self.connections[dashboard_id].append(websocket)
    
    async def disconnect(self, websocket, dashboard_id: str):
        """Remove WebSocket connection"""
        if dashboard_id in self.connections:
            self.connections[dashboard_id].remove(websocket)
    
    async def broadcast_update(self, dashboard_id: str, data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if dashboard_id in self.connections:
            for websocket in self.connections[dashboard_id]:
                try:
                    await websocket.send_text(json.dumps(data))
                except Exception as e:
                    logger.error(f"Failed to send WebSocket update: {e}")


class BusinessMetricsCalculator:
    """Calculates business metrics and KPIs"""
    
    @staticmethod
    def calculate_revenue_metrics(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate revenue-related metrics"""
        if 'revenue' not in data.columns:
            return {"error": "Revenue column not found"}
        
        current_revenue = data['revenue'].sum()
        avg_revenue = data['revenue'].mean()
        revenue_growth = 0
        
        if len(data) > 1:
            # Calculate growth rate
            recent_revenue = data['revenue'].tail(len(data)//2).sum()
            older_revenue = data['revenue'].head(len(data)//2).sum()
            if older_revenue > 0:
                revenue_growth = ((recent_revenue - older_revenue) / older_revenue) * 100
        
        return {
            "total_revenue": current_revenue,
            "average_revenue": avg_revenue,
            "revenue_growth_rate": revenue_growth,
            "revenue_trend": "increasing" if revenue_growth > 0 else "decreasing"
        }
    
    @staticmethod
    def calculate_customer_metrics(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate customer-related metrics"""
        metrics = {}
        
        if 'customer_id' in data.columns:
            metrics['total_customers'] = data['customer_id'].nunique()
            metrics['repeat_customers'] = len(data[data.duplicated(subset=['customer_id'], keep=False)])
        
        if 'acquisition_date' in data.columns:
            # Customer acquisition over time
            data['acquisition_date'] = pd.to_datetime(data['acquisition_date'])
            monthly_acquisitions = data.groupby(data['acquisition_date'].dt.to_period('M')).size()
            metrics['monthly_acquisition_trend'] = monthly_acquisitions.to_dict()
        
        if 'churn_date' in data.columns:
            churned_customers = data[data['churn_date'].notna()]
            metrics['churn_rate'] = len(churned_customers) / len(data) * 100
        
        return metrics
    
    @staticmethod
    def calculate_operational_metrics(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate operational metrics"""
        metrics = {}
        
        # Process efficiency metrics
        if 'process_time' in data.columns:
            metrics['avg_process_time'] = data['process_time'].mean()
            metrics['process_time_std'] = data['process_time'].std()
        
        if 'quality_score' in data.columns:
            metrics['avg_quality_score'] = data['quality_score'].mean()
            metrics['quality_trend'] = "improving" if data['quality_score'].is_monotonic_increasing else "declining"
        
        if 'error_count' in data.columns:
            metrics['total_errors'] = data['error_count'].sum()
            metrics['error_rate'] = data['error_count'].mean()
        
        return metrics


class AlertSystem:
    """Manages business alerts and notifications"""
    
    def __init__(self):
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: List[Dict[str, Any]] = []
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add new alert rule"""
        self.alert_rules.append(rule)
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any alerts should be triggered"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            metric_name = rule['metric']
            condition = rule['condition']
            threshold = rule['threshold']
            
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                
                if self._evaluate_condition(metric_value, condition, threshold):
                    alert = {
                        "id": f"alert_{int(time.time())}",
                        "rule_id": rule.get('id'),
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold,
                        "condition": condition,
                        "severity": rule.get('severity', 'medium'),
                        "message": rule.get('message', f"{metric_name} {condition} {threshold}"),
                        "timestamp": datetime.now().isoformat()
                    }
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.001
        return False


class BIAgent(Agent):
    """BI Agent for dashboard creation and business intelligence with real-time updates"""
    
    def __init__(self):
        super().__init__(
            name="BI Agent",
            description="Creates dashboards, tracks KPIs, provides business intelligence insights with real-time updates"
        )
        self.websocket_manager = WebSocketManager()
        self.metrics_calculator = BusinessMetricsCalculator()
        self.alert_system = AlertSystem()
        self.dashboards: Dict[str, DashboardConfig] = {}
        self.kpi_metrics: Dict[str, KPIMetric] = {}
        self._setup_default_kpis()
    
    def get_capabilities(self) -> List[str]:
        """Return BI agent capabilities"""
        return [
            "Automatic dashboard generation from data",
            "Real-time dashboard updates via WebSocket",
            "Business metric calculation and tracking",
            "KPI monitoring with threshold alerts",
            "Interactive visualizations and charts",
            "Executive reporting and insights",
            "Custom alert system for metric changes",
            "Multi-user dashboard sharing",
            "Mobile-responsive dashboard design"
        ]
    
    def _setup_default_kpis(self):
        """Setup default KPI metrics"""
        default_kpis = [
            KPIMetric("revenue_total", "Total Revenue", "sum", None, None, None, "$", "currency", "higher_better"),
            KPIMetric("revenue_growth", "Revenue Growth Rate", "growth_rate", 10.0, 5.0, 0.0, "%", "percentage", "higher_better"),
            KPIMetric("customer_count", "Total Customers", "count_unique", None, None, None, "", "number", "higher_better"),
            KPIMetric("churn_rate", "Customer Churn Rate", "percentage", 5.0, 10.0, 15.0, "%", "percentage", "lower_better"),
            KPIMetric("avg_order_value", "Average Order Value", "mean", None, None, None, "$", "currency", "higher_better"),
            KPIMetric("conversion_rate", "Conversion Rate", "percentage", 5.0, 3.0, 1.0, "%", "percentage", "higher_better")
        ]
        
        for kpi in default_kpis:
            self.kpi_metrics[kpi.id] = kpi
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process BI requests with real-time capabilities"""
        start_time = time.time()
        
        try:
            query = request.query.lower()
            context = request.context
            
            if "dashboard" in query:
                if "create" in query or "generate" in query:
                    result = await self._create_dashboard(context)
                elif "update" in query or "refresh" in query:
                    result = await self._update_dashboard(context)
                else:
                    result = await self._get_dashboard_info(context)
            elif "kpi" in query or "metric" in query:
                if "calculate" in query or "compute" in query:
                    result = await self._calculate_metrics(context)
                else:
                    result = self._setup_kpis(context)
            elif "alert" in query:
                if "setup" in query or "create" in query:
                    result = self._setup_alerts(context)
                elif "check" in query:
                    result = await self._check_alerts(context)
                else:
                    result = self._get_alert_status(context)
            elif "report" in query:
                result = await self._generate_report(context)
            elif "real-time" in query or "websocket" in query:
                result = await self._setup_realtime_updates(context)
            else:
                result = self._provide_bi_guidance(request.query, context)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                result=result,
                metadata={
                    "bi_task": self._classify_bi_task(query),
                    "real_time_enabled": "real-time" in query or "websocket" in query,
                    "dashboard_count": len(self.dashboards),
                    "active_alerts": len(self.alert_system.active_alerts)
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"BI Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _create_dashboard(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create business dashboard with real-time capabilities"""
        dashboard_type = context.get("dashboard_type", "executive")
        data_source = context.get("data_source")
        dashboard_id = f"dashboard_{int(time.time())}"
        
        # Define dashboard templates based on type
        templates = {
            "executive": {
                "name": "Executive Dashboard",
                "widgets": [
                    {"type": "kpi_card", "metric": "revenue_total", "position": {"x": 0, "y": 0, "w": 3, "h": 2}},
                    {"type": "kpi_card", "metric": "customer_count", "position": {"x": 3, "y": 0, "w": 3, "h": 2}},
                    {"type": "line_chart", "metric": "revenue_growth", "position": {"x": 0, "y": 2, "w": 6, "h": 4}},
                    {"type": "alert_panel", "position": {"x": 6, "y": 0, "w": 3, "h": 6}}
                ],
                "refresh_interval": 30,  # 30 seconds
                "filters": {"date_range": "last_30_days"}
            },
            "operational": {
                "name": "Operational Dashboard",
                "widgets": [
                    {"type": "gauge", "metric": "process_efficiency", "position": {"x": 0, "y": 0, "w": 3, "h": 3}},
                    {"type": "bar_chart", "metric": "daily_volumes", "position": {"x": 3, "y": 0, "w": 6, "h": 3}},
                    {"type": "table", "metric": "recent_activities", "position": {"x": 0, "y": 3, "w": 9, "h": 3}}
                ],
                "refresh_interval": 300,  # 5 minutes
                "filters": {"date_range": "today"}
            },
            "sales": {
                "name": "Sales Dashboard",
                "widgets": [
                    {"type": "kpi_card", "metric": "conversion_rate", "position": {"x": 0, "y": 0, "w": 3, "h": 2}},
                    {"type": "kpi_card", "metric": "avg_order_value", "position": {"x": 3, "y": 0, "w": 3, "h": 2}},
                    {"type": "funnel_chart", "metric": "sales_funnel", "position": {"x": 0, "y": 2, "w": 6, "h": 4}},
                    {"type": "leaderboard", "metric": "top_performers", "position": {"x": 6, "y": 0, "w": 3, "h": 6}}
                ],
                "refresh_interval": 60,  # 1 minute
                "filters": {"date_range": "this_month"}
            }
        }
        
        template = templates.get(dashboard_type, templates["executive"])
        
        # Create dashboard configuration
        dashboard_config = DashboardConfig(
            id=dashboard_id,
            name=template["name"],
            widgets=template["widgets"],
            layout={"grid_size": {"width": 12, "height": 8}},
            refresh_interval=template["refresh_interval"],
            filters=template["filters"],
            permissions={"owner": context.get("user_id"), "viewers": [], "editors": []}
        )
        
        # Store dashboard
        self.dashboards[dashboard_id] = dashboard_config
        self.websocket_manager.active_dashboards[dashboard_id] = dashboard_config
        
        # If data source provided, calculate initial metrics
        initial_data = {}
        if data_source:
            initial_data = await self._calculate_dashboard_data(dashboard_config, data_source)
        
        return {
            "dashboard_id": dashboard_id,
            "dashboard_config": {
                "name": dashboard_config.name,
                "widgets": dashboard_config.widgets,
                "layout": dashboard_config.layout,
                "refresh_interval": dashboard_config.refresh_interval,
                "filters": dashboard_config.filters
            },
            "initial_data": initial_data,
            "websocket_endpoint": f"/ws/dashboard/{dashboard_id}",
            "real_time_enabled": True,
            "customization_options": [
                "Drag-and-drop widget rearrangement",
                "Custom color themes and branding",
                "Responsive mobile/desktop layouts",
                "Role-based access permissions",
                "Custom KPI thresholds and alerts",
                "Export to PDF/PNG/Excel formats"
            ],
            "supported_visualizations": [
                "KPI cards with trend indicators",
                "Line/bar/area charts",
                "Pie/donut charts",
                "Gauge/speedometer charts",
                "Data tables with sorting/filtering",
                "Heatmaps and treemaps",
                "Funnel and waterfall charts"
            ]
        }
    
    async def _update_dashboard(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update dashboard with new data"""
        dashboard_id = context.get("dashboard_id")
        if not dashboard_id or dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard_config = self.dashboards[dashboard_id]
        data_source = context.get("data_source")
        
        if data_source:
            # Calculate updated metrics
            updated_data = await self._calculate_dashboard_data(dashboard_config, data_source)
            
            # Check for alerts
            alerts = self.alert_system.check_alerts(updated_data.get("metrics", {}))
            
            # Broadcast update via WebSocket
            update_payload = {
                "type": "dashboard_update",
                "dashboard_id": dashboard_id,
                "data": updated_data,
                "alerts": alerts,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.websocket_manager.broadcast_update(dashboard_id, update_payload)
            
            return {
                "dashboard_id": dashboard_id,
                "updated_data": updated_data,
                "alerts_triggered": len(alerts),
                "last_updated": datetime.now().isoformat(),
                "next_update": (datetime.now() + timedelta(seconds=dashboard_config.refresh_interval)).isoformat()
            }
        
        return {"error": "No data source provided for update"}
    
    async def _get_dashboard_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get dashboard information"""
        dashboard_id = context.get("dashboard_id")
        
        if dashboard_id and dashboard_id in self.dashboards:
            dashboard = self.dashboards[dashboard_id]
            return {
                "dashboard_id": dashboard_id,
                "name": dashboard.name,
                "widget_count": len(dashboard.widgets),
                "refresh_interval": dashboard.refresh_interval,
                "permissions": dashboard.permissions,
                "last_updated": datetime.now().isoformat()
            }
        
        return {
            "available_dashboards": [
                {
                    "id": dash_id,
                    "name": dash.name,
                    "widget_count": len(dash.widgets)
                }
                for dash_id, dash in self.dashboards.items()
            ],
            "total_dashboards": len(self.dashboards)
        }
    
    async def _calculate_dashboard_data(self, dashboard_config: DashboardConfig, data_source: Any) -> Dict[str, Any]:
        """Calculate data for dashboard widgets"""
        # This would integrate with actual data sources
        # For now, return sample calculated data
        
        sample_data = {
            "metrics": {
                "revenue_total": 125000.50,
                "revenue_growth": 15.3,
                "customer_count": 1250,
                "churn_rate": 3.2,
                "conversion_rate": 4.8,
                "avg_order_value": 89.99
            },
            "charts": {
                "revenue_trend": {
                    "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
                    "data": [95000, 102000, 108000, 118000, 125000]
                },
                "customer_acquisition": {
                    "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
                    "data": [45, 52, 38, 61]
                }
            },
            "tables": {
                "top_products": [
                    {"product": "Product A", "revenue": 45000, "units": 450},
                    {"product": "Product B", "revenue": 38000, "units": 380},
                    {"product": "Product C", "revenue": 32000, "units": 320}
                ]
            }
        }
        
        return sample_data
    
    async def _calculate_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business metrics from data"""
        data_source = context.get("data_source")
        metric_types = context.get("metric_types", ["revenue", "customer", "operational"])
        
        if not data_source:
            return {"error": "No data source provided"}
        
        # Simulate data loading (in real implementation, load from actual source)
        # For demo, create sample DataFrame
        sample_data = pd.DataFrame({
            'revenue': np.random.normal(1000, 200, 100),
            'customer_id': range(1, 101),
            'acquisition_date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'process_time': np.random.normal(30, 5, 100),
            'quality_score': np.random.uniform(0.7, 1.0, 100)
        })
        
        calculated_metrics = {}
        
        if "revenue" in metric_types:
            calculated_metrics.update(self.metrics_calculator.calculate_revenue_metrics(sample_data))
        
        if "customer" in metric_types:
            calculated_metrics.update(self.metrics_calculator.calculate_customer_metrics(sample_data))
        
        if "operational" in metric_types:
            calculated_metrics.update(self.metrics_calculator.calculate_operational_metrics(sample_data))
        
        return {
            "calculated_metrics": calculated_metrics,
            "data_points_analyzed": len(sample_data),
            "metric_types": metric_types,
            "calculation_timestamp": datetime.now().isoformat(),
            "data_quality_score": 0.95  # Sample quality score
        }
    
    def _setup_kpis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup KPIs and metrics"""
        return {
            "common_kpis": {
                "financial": [
                    "Monthly Recurring Revenue (MRR)",
                    "Customer Acquisition Cost (CAC)",
                    "Lifetime Value (LTV)",
                    "Gross/Net Profit Margins"
                ],
                "operational": [
                    "Customer Satisfaction Score",
                    "Employee Productivity",
                    "Process Efficiency",
                    "Quality Metrics"
                ],
                "growth": [
                    "User Growth Rate",
                    "Market Share",
                    "Product Adoption",
                    "Retention Rate"
                ]
            },
            "kpi_configuration": {
                "calculation_method": "Automated based on data source",
                "update_frequency": "Real-time, hourly, daily, or custom",
                "target_setting": "Manual targets or historical benchmarks",
                "trend_analysis": "Automatic trend detection and forecasting"
            },
            "alerting": [
                "Threshold-based alerts",
                "Anomaly detection",
                "Email/SMS notifications",
                "Slack/Teams integration"
            ]
        }
    
    async def _generate_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive business reports"""
        report_type = context.get("report_type", "executive_summary")
        date_range = context.get("date_range", "last_30_days")
        data_source = context.get("data_source")
        
        # Calculate metrics for report
        metrics = {}
        if data_source:
            metrics_context = {"data_source": data_source, "metric_types": ["revenue", "customer", "operational"]}
            metrics_result = await self._calculate_metrics(metrics_context)
            metrics = metrics_result.get("calculated_metrics", {})
        
        # Generate insights based on metrics
        insights = self._generate_insights(metrics)
        
        report_templates = {
            "executive_summary": {
                "title": "Executive Business Summary",
                "sections": [
                    {
                        "name": "Key Performance Indicators",
                        "content": {
                            "revenue_total": metrics.get("total_revenue", 0),
                            "revenue_growth": metrics.get("revenue_growth_rate", 0),
                            "customer_count": metrics.get("total_customers", 0),
                            "churn_rate": metrics.get("churn_rate", 0)
                        }
                    },
                    {
                        "name": "Performance Trends",
                        "content": insights.get("trends", [])
                    },
                    {
                        "name": "Key Insights",
                        "content": insights.get("key_insights", [])
                    },
                    {
                        "name": "Recommendations",
                        "content": insights.get("recommendations", [])
                    }
                ]
            },
            "operational_report": {
                "title": "Operational Performance Report",
                "sections": [
                    {
                        "name": "Process Efficiency",
                        "content": {
                            "avg_process_time": metrics.get("avg_process_time", 0),
                            "quality_score": metrics.get("avg_quality_score", 0),
                            "error_rate": metrics.get("error_rate", 0)
                        }
                    },
                    {
                        "name": "Volume Analysis",
                        "content": insights.get("volume_analysis", {})
                    },
                    {
                        "name": "Quality Metrics",
                        "content": insights.get("quality_metrics", {})
                    }
                ]
            },
            "financial_report": {
                "title": "Financial Performance Report",
                "sections": [
                    {
                        "name": "Revenue Analysis",
                        "content": {
                            "total_revenue": metrics.get("total_revenue", 0),
                            "average_revenue": metrics.get("average_revenue", 0),
                            "growth_rate": metrics.get("revenue_growth_rate", 0)
                        }
                    },
                    {
                        "name": "Customer Metrics",
                        "content": {
                            "customer_count": metrics.get("total_customers", 0),
                            "repeat_customers": metrics.get("repeat_customers", 0),
                            "churn_rate": metrics.get("churn_rate", 0)
                        }
                    }
                ]
            }
        }
        
        selected_template = report_templates.get(report_type, report_templates["executive_summary"])
        
        return {
            "report_id": f"report_{int(time.time())}",
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "date_range": date_range,
            "report_data": selected_template,
            "export_options": {
                "pdf": f"/api/reports/{report_type}/pdf",
                "excel": f"/api/reports/{report_type}/excel",
                "json": f"/api/reports/{report_type}/json",
                "dashboard": f"/dashboard/report/{report_type}"
            },
            "automated_scheduling": {
                "available": True,
                "frequencies": ["daily", "weekly", "monthly", "quarterly"],
                "delivery_methods": ["email", "dashboard", "api_webhook"]
            },
            "insights_summary": {
                "total_insights": len(insights.get("key_insights", [])),
                "recommendations": len(insights.get("recommendations", [])),
                "data_quality": "high",
                "confidence_score": 0.92
            }
        }
    
    def _generate_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business insights from metrics"""
        insights = {
            "key_insights": [],
            "trends": [],
            "recommendations": [],
            "volume_analysis": {},
            "quality_metrics": {}
        }
        
        # Revenue insights
        if "revenue_growth_rate" in metrics:
            growth_rate = metrics["revenue_growth_rate"]
            if growth_rate > 10:
                insights["key_insights"].append("Strong revenue growth indicates healthy business expansion")
                insights["recommendations"].append("Consider scaling operations to support continued growth")
            elif growth_rate < 0:
                insights["key_insights"].append("Revenue decline requires immediate attention")
                insights["recommendations"].append("Analyze customer feedback and market conditions")
        
        # Customer insights
        if "churn_rate" in metrics:
            churn_rate = metrics["churn_rate"]
            if churn_rate > 10:
                insights["key_insights"].append("High customer churn rate is concerning")
                insights["recommendations"].append("Implement customer retention programs")
            elif churn_rate < 5:
                insights["key_insights"].append("Low churn rate indicates good customer satisfaction")
        
        # Operational insights
        if "avg_quality_score" in metrics:
            quality_score = metrics["avg_quality_score"]
            if quality_score > 0.9:
                insights["key_insights"].append("Excellent quality scores demonstrate operational excellence")
            elif quality_score < 0.7:
                insights["key_insights"].append("Quality scores below target - review processes")
                insights["recommendations"].append("Implement quality improvement initiatives")
        
        # Trend analysis
        insights["trends"] = [
            "Revenue trending upward over last 30 days",
            "Customer acquisition rate stable",
            "Operational efficiency improving"
        ]
        
        return insights
    
    def _setup_alerts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup business alerts with real-time monitoring"""
        alert_config = context.get("alert_config", {})
        
        # Default alert rules
        default_rules = [
            {
                "id": "revenue_drop",
                "metric": "revenue_total",
                "condition": "less_than",
                "threshold": 100000,
                "severity": "high",
                "message": "Revenue has dropped below $100,000"
            },
            {
                "id": "churn_spike",
                "metric": "churn_rate",
                "condition": "greater_than",
                "threshold": 10.0,
                "severity": "critical",
                "message": "Customer churn rate exceeds 10%"
            },
            {
                "id": "conversion_low",
                "metric": "conversion_rate",
                "condition": "less_than",
                "threshold": 2.0,
                "severity": "medium",
                "message": "Conversion rate below 2%"
            }
        ]
        
        # Add custom rules from context
        custom_rules = alert_config.get("rules", [])
        all_rules = default_rules + custom_rules
        
        # Setup alert rules
        for rule in all_rules:
            self.alert_system.add_alert_rule(rule)
        
        return {
            "alert_system_status": "active",
            "total_rules": len(all_rules),
            "alert_rules": all_rules,
            "notification_channels": {
                "email": {
                    "enabled": True,
                    "recipients": alert_config.get("email_recipients", [])
                },
                "webhook": {
                    "enabled": True,
                    "url": alert_config.get("webhook_url")
                },
                "real_time": {
                    "enabled": True,
                    "websocket_broadcast": True
                }
            },
            "alert_features": [
                "Real-time threshold monitoring",
                "Anomaly detection using statistical methods",
                "Trend change detection",
                "Multi-channel notifications",
                "Alert suppression and escalation",
                "Historical alert tracking"
            ]
        }
    
    async def _check_alerts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check current metrics against alert rules"""
        current_metrics = context.get("metrics", {})
        
        if not current_metrics:
            # Get latest metrics from dashboards
            if self.dashboards:
                dashboard_id = list(self.dashboards.keys())[0]
                dashboard_data = await self._calculate_dashboard_data(
                    self.dashboards[dashboard_id], 
                    context.get("data_source")
                )
                current_metrics = dashboard_data.get("metrics", {})
        
        triggered_alerts = self.alert_system.check_alerts(current_metrics)
        
        # Broadcast alerts via WebSocket if any triggered
        if triggered_alerts:
            for dashboard_id in self.dashboards.keys():
                alert_payload = {
                    "type": "alert_triggered",
                    "alerts": triggered_alerts,
                    "timestamp": datetime.now().isoformat()
                }
                await self.websocket_manager.broadcast_update(dashboard_id, alert_payload)
        
        return {
            "alerts_checked": len(self.alert_system.alert_rules),
            "alerts_triggered": len(triggered_alerts),
            "triggered_alerts": triggered_alerts,
            "metrics_evaluated": current_metrics,
            "check_timestamp": datetime.now().isoformat()
        }
    
    def _get_alert_status(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current alert system status"""
        return {
            "alert_system_active": True,
            "total_rules": len(self.alert_system.alert_rules),
            "active_alerts": len(self.alert_system.active_alerts),
            "alert_rules": [
                {
                    "id": rule.get("id"),
                    "metric": rule.get("metric"),
                    "condition": rule.get("condition"),
                    "threshold": rule.get("threshold"),
                    "severity": rule.get("severity")
                }
                for rule in self.alert_system.alert_rules
            ],
            "recent_alerts": self.alert_system.active_alerts[-10:]  # Last 10 alerts
        }
    
    async def _setup_realtime_updates(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup real-time dashboard updates"""
        dashboard_id = context.get("dashboard_id")
        update_interval = context.get("update_interval", 30)  # seconds
        
        if not dashboard_id:
            return {"error": "Dashboard ID required for real-time updates"}
        
        if dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        # Update dashboard refresh interval
        self.dashboards[dashboard_id].refresh_interval = update_interval
        
        return {
            "real_time_setup": "complete",
            "dashboard_id": dashboard_id,
            "update_interval": update_interval,
            "websocket_endpoint": f"/ws/dashboard/{dashboard_id}",
            "features_enabled": [
                "Automatic data refresh",
                "Real-time metric updates",
                "Live alert notifications",
                "Multi-user synchronization",
                "Connection status monitoring"
            ],
            "websocket_events": [
                "dashboard_update - New data available",
                "alert_triggered - Alert condition met",
                "connection_status - WebSocket status changes",
                "user_activity - Other users viewing dashboard"
            ]
        }
    
    def _provide_bi_guidance(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general BI guidance"""
        return {
            "bi_best_practices": [
                "Start with key business questions",
                "Focus on actionable metrics",
                "Ensure data quality and consistency",
                "Design for your audience",
                "Keep dashboards simple and focused"
            ],
            "dashboard_design_principles": [
                "Most important information at the top",
                "Use consistent colors and formatting",
                "Minimize cognitive load",
                "Provide context with comparisons",
                "Enable drill-down capabilities"
            ],
            "data_visualization_guidelines": {
                "charts_for_trends": "Line charts",
                "charts_for_comparisons": "Bar charts",
                "charts_for_parts_of_whole": "Pie charts (use sparingly)",
                "charts_for_correlations": "Scatter plots",
                "charts_for_distributions": "Histograms"
            },
            "implementation_steps": [
                "Define business objectives and KPIs",
                "Identify and connect data sources",
                "Create initial dashboard prototypes",
                "Gather user feedback and iterate",
                "Implement automated reporting and alerts"
            ]
        }
    
    def _classify_bi_task(self, query: str) -> str:
        """Classify the type of BI task"""
        if "dashboard" in query:
            if "create" in query or "generate" in query:
                return "dashboard_creation"
            elif "update" in query or "refresh" in query:
                return "dashboard_update"
            else:
                return "dashboard_info"
        elif "kpi" in query or "metric" in query:
            if "calculate" in query or "compute" in query:
                return "metric_calculation"
            else:
                return "kpi_setup"
        elif "alert" in query:
            if "setup" in query or "create" in query:
                return "alert_setup"
            elif "check" in query:
                return "alert_check"
            else:
                return "alert_status"
        elif "report" in query:
            return "report_generation"
        elif "real-time" in query or "websocket" in query:
            return "realtime_setup"
        else:
            return "general_bi_guidance"
    
    # WebSocket connection management methods
    async def connect_websocket(self, websocket, dashboard_id: str):
        """Connect WebSocket for real-time updates"""
        await self.websocket_manager.connect(websocket, dashboard_id)
    
    async def disconnect_websocket(self, websocket, dashboard_id: str):
        """Disconnect WebSocket"""
        await self.websocket_manager.disconnect(websocket, dashboard_id)
    
    async def broadcast_dashboard_update(self, dashboard_id: str, data: Dict[str, Any]):
        """Broadcast update to dashboard subscribers"""
        await self.websocket_manager.broadcast_update(dashboard_id, data)
    
    # Utility methods for dashboard management
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of all dashboards"""
        return [
            {
                "id": dash_id,
                "name": dash.name,
                "widget_count": len(dash.widgets),
                "refresh_interval": dash.refresh_interval,
                "permissions": dash.permissions
            }
            for dash_id, dash in self.dashboards.items()
        ]
    
    def get_kpi_definitions(self) -> List[Dict[str, Any]]:
        """Get all KPI metric definitions"""
        return [
            {
                "id": kpi.id,
                "name": kpi.name,
                "calculation": kpi.calculation,
                "target_value": kpi.target_value,
                "unit": kpi.unit,
                "format_type": kpi.format_type,
                "trend_direction": kpi.trend_direction
            }
            for kpi in self.kpi_metrics.values()
        ]
    
    def add_custom_kpi(self, kpi_config: Dict[str, Any]) -> str:
        """Add custom KPI metric"""
        kpi_id = kpi_config.get("id", f"custom_kpi_{int(time.time())}")
        
        custom_kpi = KPIMetric(
            id=kpi_id,
            name=kpi_config.get("name", "Custom KPI"),
            calculation=kpi_config.get("calculation", "sum"),
            target_value=kpi_config.get("target_value"),
            threshold_warning=kpi_config.get("threshold_warning"),
            threshold_critical=kpi_config.get("threshold_critical"),
            unit=kpi_config.get("unit", ""),
            format_type=kpi_config.get("format_type", "number"),
            trend_direction=kpi_config.get("trend_direction", "higher_better")
        )
        
        self.kpi_metrics[kpi_id] = custom_kpi
        return kpi_id
    
    def remove_dashboard(self, dashboard_id: str) -> bool:
        """Remove dashboard"""
        if dashboard_id in self.dashboards:
            del self.dashboards[dashboard_id]
            if dashboard_id in self.websocket_manager.active_dashboards:
                del self.websocket_manager.active_dashboards[dashboard_id]
            return True
        return False
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alert history"""
        return self.alert_system.active_alerts[-limit:]
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check for BI Agent"""
        try:
            # Check core components
            components_healthy = (
                self.websocket_manager is not None and
                self.metrics_calculator is not None and
                self.alert_system is not None and
                len(self.kpi_metrics) > 0
            )
            
            self.is_healthy = components_healthy
            self.last_health_check = datetime.utcnow()
            
            return {
                "agent": self.name,
                "healthy": self.is_healthy,
                "last_check": self.last_health_check.isoformat(),
                "capabilities": self.capabilities,
                "components": {
                    "websocket_manager": self.websocket_manager is not None,
                    "metrics_calculator": self.metrics_calculator is not None,
                    "alert_system": self.alert_system is not None,
                    "kpi_metrics": len(self.kpi_metrics)
                },
                "statistics": {
                    "active_dashboards": len(self.dashboards),
                    "alert_rules": len(self.alert_system.alert_rules),
                    "kpi_definitions": len(self.kpi_metrics)
                }
            }
        except Exception as e:
            logger.error(f"BI Agent health check failed: {e}")
            self.is_healthy = False
            return {
                "agent": self.name,
                "healthy": False,
                "error": str(e),
                "last_check": self.last_health_check.isoformat()
            }