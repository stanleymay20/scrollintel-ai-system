"""
Role-based dashboard templates for different executive roles.
"""
from typing import Dict, List, Any
from ..models.dashboard_models import ExecutiveRole, WidgetType


class DashboardTemplates:
    """Predefined dashboard templates for different executive roles."""
    
    @staticmethod
    def get_cto_template() -> Dict[str, Any]:
        """CTO Executive Dashboard Template."""
        return {
            "name": "CTO Executive Dashboard",
            "description": "Comprehensive technology leadership dashboard with AI/ML initiatives, ROI tracking, and system performance",
            "category": "executive",
            "role": ExecutiveRole.CTO.value,
            "layout": {
                "grid_columns": 12,
                "grid_rows": 8,
                "theme": "tech"
            },
            "widgets": [
                {
                    "name": "Technology ROI Overview",
                    "type": WidgetType.KPI.value,
                    "position_x": 0,
                    "position_y": 0,
                    "width": 12,
                    "height": 2,
                    "config": {
                        "metrics": ["tech_roi", "ai_investment_return", "infrastructure_savings"],
                        "display_mode": "horizontal",
                        "show_trends": True
                    },
                    "data_source": "roi_calculator",
                    "refresh_interval": 300
                },
                {
                    "name": "AI Initiative Performance",
                    "type": WidgetType.CHART.value,
                    "position_x": 0,
                    "position_y": 2,
                    "width": 8,
                    "height": 3,
                    "config": {
                        "chart_type": "line",
                        "metrics": ["model_accuracy", "deployment_success_rate", "user_adoption"],
                        "time_range": "30d",
                        "show_predictions": True
                    },
                    "data_source": "ai_projects",
                    "refresh_interval": 600
                },
                {
                    "name": "System Health",
                    "type": WidgetType.METRIC.value,
                    "position_x": 8,
                    "position_y": 2,
                    "width": 4,
                    "height": 3,
                    "config": {
                        "metric": "system_uptime",
                        "threshold_warning": 95,
                        "threshold_critical": 90,
                        "show_alerts": True
                    },
                    "data_source": "monitoring",
                    "refresh_interval": 60
                },
                {
                    "name": "Technology Investment Pipeline",
                    "type": WidgetType.TABLE.value,
                    "position_x": 0,
                    "position_y": 5,
                    "width": 6,
                    "height": 3,
                    "config": {
                        "columns": ["project", "status", "budget", "expected_roi", "timeline"],
                        "sortable": True,
                        "filterable": True
                    },
                    "data_source": "project_pipeline",
                    "refresh_interval": 1800
                },
                {
                    "name": "Team Productivity Metrics",
                    "type": WidgetType.CHART.value,
                    "position_x": 6,
                    "position_y": 5,
                    "width": 6,
                    "height": 3,
                    "config": {
                        "chart_type": "bar",
                        "metrics": ["velocity", "code_quality", "deployment_frequency"],
                        "comparison_period": "previous_month"
                    },
                    "data_source": "team_metrics",
                    "refresh_interval": 3600
                }
            ]
        }
    
    @staticmethod
    def get_cfo_template() -> Dict[str, Any]:
        """CFO Executive Dashboard Template."""
        return {
            "name": "CFO Executive Dashboard",
            "description": "Financial leadership dashboard with ROI analysis, cost optimization, and budget tracking",
            "category": "executive",
            "role": ExecutiveRole.CFO.value,
            "layout": {
                "grid_columns": 12,
                "grid_rows": 8,
                "theme": "financial"
            },
            "widgets": [
                {
                    "name": "Financial KPIs",
                    "type": WidgetType.KPI.value,
                    "position_x": 0,
                    "position_y": 0,
                    "width": 12,
                    "height": 2,
                    "config": {
                        "metrics": ["total_revenue", "cost_savings", "profit_margin", "cash_flow"],
                        "display_mode": "grid",
                        "show_variance": True
                    },
                    "data_source": "financial_data",
                    "refresh_interval": 1800
                },
                {
                    "name": "Technology Investment ROI",
                    "type": WidgetType.CHART.value,
                    "position_x": 0,
                    "position_y": 2,
                    "width": 8,
                    "height": 3,
                    "config": {
                        "chart_type": "line",
                        "metrics": ["cumulative_roi", "monthly_savings", "investment_payback"],
                        "time_range": "12m",
                        "show_projections": True
                    },
                    "data_source": "roi_calculator",
                    "refresh_interval": 3600
                },
                {
                    "name": "Budget vs Actual",
                    "type": WidgetType.CHART.value,
                    "position_x": 8,
                    "position_y": 2,
                    "width": 4,
                    "height": 3,
                    "config": {
                        "chart_type": "bar",
                        "comparison": ["budget", "actual", "forecast"],
                        "breakdown": "department"
                    },
                    "data_source": "budget_tracking",
                    "refresh_interval": 3600
                },
                {
                    "name": "Cost Center Analysis",
                    "type": WidgetType.TABLE.value,
                    "position_x": 0,
                    "position_y": 5,
                    "width": 8,
                    "height": 3,
                    "config": {
                        "columns": ["department", "budget", "actual", "variance", "efficiency"],
                        "highlight_variances": True,
                        "show_trends": True
                    },
                    "data_source": "cost_centers",
                    "refresh_interval": 3600
                },
                {
                    "name": "Risk Indicators",
                    "type": WidgetType.METRIC.value,
                    "position_x": 8,
                    "position_y": 5,
                    "width": 4,
                    "height": 3,
                    "config": {
                        "metrics": ["budget_risk", "cash_flow_risk", "investment_risk"],
                        "alert_thresholds": {"high": 80, "medium": 60},
                        "show_recommendations": True
                    },
                    "data_source": "risk_analysis",
                    "refresh_interval": 1800
                }
            ]
        }
    
    @staticmethod
    def get_ceo_template() -> Dict[str, Any]:
        """CEO Executive Dashboard Template."""
        return {
            "name": "CEO Executive Dashboard",
            "description": "Strategic leadership dashboard with business performance, growth metrics, and strategic initiatives",
            "category": "executive",
            "role": ExecutiveRole.CEO.value,
            "layout": {
                "grid_columns": 12,
                "grid_rows": 10,
                "theme": "executive"
            },
            "widgets": [
                {
                    "name": "Business Performance Overview",
                    "type": WidgetType.KPI.value,
                    "position_x": 0,
                    "position_y": 0,
                    "width": 12,
                    "height": 2,
                    "config": {
                        "metrics": ["revenue_growth", "market_share", "customer_satisfaction", "employee_engagement"],
                        "display_mode": "executive",
                        "show_targets": True
                    },
                    "data_source": "business_metrics",
                    "refresh_interval": 3600
                },
                {
                    "name": "Strategic Initiative Progress",
                    "type": WidgetType.CHART.value,
                    "position_x": 0,
                    "position_y": 2,
                    "width": 6,
                    "height": 4,
                    "config": {
                        "chart_type": "gantt",
                        "show_milestones": True,
                        "show_dependencies": True,
                        "time_range": "12m"
                    },
                    "data_source": "strategic_initiatives",
                    "refresh_interval": 3600
                },
                {
                    "name": "Market Position",
                    "type": WidgetType.CHART.value,
                    "position_x": 6,
                    "position_y": 2,
                    "width": 6,
                    "height": 4,
                    "config": {
                        "chart_type": "radar",
                        "metrics": ["innovation", "efficiency", "customer_focus", "market_presence"],
                        "benchmark": "industry_average"
                    },
                    "data_source": "market_analysis",
                    "refresh_interval": 86400
                },
                {
                    "name": "Financial Summary",
                    "type": WidgetType.TABLE.value,
                    "position_x": 0,
                    "position_y": 6,
                    "width": 8,
                    "height": 2,
                    "config": {
                        "columns": ["metric", "current", "target", "variance", "trend"],
                        "financial_format": True,
                        "highlight_critical": True
                    },
                    "data_source": "executive_financials",
                    "refresh_interval": 3600
                },
                {
                    "name": "Risk Dashboard",
                    "type": WidgetType.METRIC.value,
                    "position_x": 8,
                    "position_y": 6,
                    "width": 4,
                    "height": 2,
                    "config": {
                        "risk_categories": ["operational", "financial", "strategic", "compliance"],
                        "severity_levels": ["low", "medium", "high", "critical"],
                        "show_mitigation": True
                    },
                    "data_source": "risk_management",
                    "refresh_interval": 1800
                },
                {
                    "name": "Competitive Intelligence",
                    "type": WidgetType.CHART.value,
                    "position_x": 0,
                    "position_y": 8,
                    "width": 12,
                    "height": 2,
                    "config": {
                        "chart_type": "comparison",
                        "competitors": ["competitor_a", "competitor_b", "competitor_c"],
                        "metrics": ["market_share", "growth_rate", "innovation_index"]
                    },
                    "data_source": "competitive_analysis",
                    "refresh_interval": 86400
                }
            ]
        }
    
    @staticmethod
    def get_vp_engineering_template() -> Dict[str, Any]:
        """VP Engineering Dashboard Template."""
        return {
            "name": "VP Engineering Dashboard",
            "description": "Engineering leadership dashboard with team performance, delivery metrics, and technical health",
            "category": "executive",
            "role": ExecutiveRole.VP_ENGINEERING.value,
            "layout": {
                "grid_columns": 12,
                "grid_rows": 8,
                "theme": "engineering"
            },
            "widgets": [
                {
                    "name": "Engineering KPIs",
                    "type": WidgetType.KPI.value,
                    "position_x": 0,
                    "position_y": 0,
                    "width": 12,
                    "height": 2,
                    "config": {
                        "metrics": ["deployment_frequency", "lead_time", "mttr", "change_failure_rate"],
                        "dora_metrics": True,
                        "show_benchmarks": True
                    },
                    "data_source": "engineering_metrics",
                    "refresh_interval": 1800
                },
                {
                    "name": "Team Velocity Trends",
                    "type": WidgetType.CHART.value,
                    "position_x": 0,
                    "position_y": 2,
                    "width": 8,
                    "height": 3,
                    "config": {
                        "chart_type": "line",
                        "metrics": ["story_points", "cycle_time", "throughput"],
                        "team_breakdown": True,
                        "time_range": "3m"
                    },
                    "data_source": "agile_metrics",
                    "refresh_interval": 3600
                },
                {
                    "name": "Code Quality",
                    "type": WidgetType.METRIC.value,
                    "position_x": 8,
                    "position_y": 2,
                    "width": 4,
                    "height": 3,
                    "config": {
                        "metrics": ["code_coverage", "technical_debt", "security_score"],
                        "quality_gates": True,
                        "show_trends": True
                    },
                    "data_source": "code_quality",
                    "refresh_interval": 3600
                },
                {
                    "name": "Sprint Performance",
                    "type": WidgetType.TABLE.value,
                    "position_x": 0,
                    "position_y": 5,
                    "width": 12,
                    "height": 3,
                    "config": {
                        "columns": ["team", "sprint", "planned", "completed", "velocity", "burndown"],
                        "show_capacity": True,
                        "highlight_risks": True
                    },
                    "data_source": "sprint_data",
                    "refresh_interval": 3600
                }
            ]
        }
    
    @staticmethod
    def get_department_head_template() -> Dict[str, Any]:
        """Department Head Dashboard Template."""
        return {
            "name": "Department Head Dashboard",
            "description": "Departmental leadership dashboard with team metrics, budget tracking, and operational KPIs",
            "category": "departmental",
            "role": ExecutiveRole.DEPARTMENT_HEAD.value,
            "layout": {
                "grid_columns": 12,
                "grid_rows": 6,
                "theme": "departmental"
            },
            "widgets": [
                {
                    "name": "Department KPIs",
                    "type": WidgetType.KPI.value,
                    "position_x": 0,
                    "position_y": 0,
                    "width": 12,
                    "height": 2,
                    "config": {
                        "metrics": ["productivity", "efficiency", "quality", "satisfaction"],
                        "department_specific": True,
                        "show_targets": True
                    },
                    "data_source": "department_metrics",
                    "refresh_interval": 3600
                },
                {
                    "name": "Team Performance",
                    "type": WidgetType.CHART.value,
                    "position_x": 0,
                    "position_y": 2,
                    "width": 8,
                    "height": 2,
                    "config": {
                        "chart_type": "bar",
                        "metrics": ["individual_performance", "team_collaboration", "goal_achievement"],
                        "show_comparisons": True
                    },
                    "data_source": "team_performance",
                    "refresh_interval": 3600
                },
                {
                    "name": "Budget Status",
                    "type": WidgetType.METRIC.value,
                    "position_x": 8,
                    "position_y": 2,
                    "width": 4,
                    "height": 2,
                    "config": {
                        "metrics": ["budget_utilization", "cost_per_outcome", "forecast_accuracy"],
                        "alert_thresholds": {"warning": 80, "critical": 95}
                    },
                    "data_source": "department_budget",
                    "refresh_interval": 3600
                },
                {
                    "name": "Operational Metrics",
                    "type": WidgetType.TABLE.value,
                    "position_x": 0,
                    "position_y": 4,
                    "width": 12,
                    "height": 2,
                    "config": {
                        "columns": ["metric", "current", "target", "trend", "action_required"],
                        "operational_focus": True,
                        "show_recommendations": True
                    },
                    "data_source": "operational_data",
                    "refresh_interval": 1800
                }
            ]
        }
    
    @staticmethod
    def get_all_templates() -> Dict[str, Dict[str, Any]]:
        """Get all available dashboard templates."""
        return {
            "cto": DashboardTemplates.get_cto_template(),
            "cfo": DashboardTemplates.get_cfo_template(),
            "ceo": DashboardTemplates.get_ceo_template(),
            "vp_engineering": DashboardTemplates.get_vp_engineering_template(),
            "department_head": DashboardTemplates.get_department_head_template()
        }
    
    @staticmethod
    def get_template_by_role(role: str) -> Dict[str, Any]:
        """Get template for a specific role."""
        templates = DashboardTemplates.get_all_templates()
        return templates.get(role.lower(), templates["department_head"])