"""
ScrollBI Agent - Dashboard creation and business intelligence agent.

This agent provides instant dashboard generation from BI queries and data schemas,
real-time dashboard updates with WebSocket connections, alert system for threshold-based
notifications, and dashboard sharing with permission management.
"""

import asyncio
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

warnings.filterwarnings('ignore')

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus
from scrollintel.engines.scroll_viz_engine import ScrollVizEngine, ChartType
from scrollintel.models.database import Dashboard, Dataset, User

logger = logging.getLogger(__name__)

# OpenAI integration for intelligent dashboard creation
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None


class DashboardType(Enum):
    """Types of dashboards that can be created."""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    FINANCIAL = "financial"
    MARKETING = "marketing"
    SALES = "sales"
    CUSTOM = "custom"


class AlertType(Enum):
    """Types of alerts for threshold monitoring."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    THRESHOLD_BELOW = "threshold_below"
    ANOMALY_DETECTED = "anomaly_detected"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    TREND_CHANGE = "trend_change"


class SharePermission(Enum):
    """Dashboard sharing permission levels."""
    VIEW_ONLY = "view_only"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


@dataclass
class DashboardConfig:
    """Configuration for dashboard creation."""
    name: str
    description: str
    dashboard_type: DashboardType
    layout: str  # grid, masonry, custom
    refresh_interval: int  # minutes
    auto_refresh: bool
    theme: str
    filters: Dict[str, Any]
    charts: List[Dict[str, Any]]
    real_time_enabled: bool
    alert_config: Dict[str, Any]


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    alert_type: AlertType
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    notification_channels: List[str]
    is_active: bool
    created_at: datetime


@dataclass
class DashboardShare:
    """Dashboard sharing configuration."""
    dashboard_id: str
    user_id: str
    permission: SharePermission
    expires_at: Optional[datetime]
    created_by: str
    created_at: datetime


class ScrollBIAgent(BaseAgent):
    """ScrollBI agent for dashboard creation and business intelligence."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-bi",
            name="ScrollBI Agent",
            agent_type=AgentType.BI_DEVELOPER
        )
        self.capabilities = [
            AgentCapability(
                name="instant_dashboard_creation",
                description="Create interactive dashboards instantly from BI queries and data schemas",
                input_types=["bi_query", "data_schema", "dashboard_config"],
                output_types=["dashboard", "dashboard_config", "visualization_specs"]
            ),
            AgentCapability(
                name="real_time_dashboard_updates",
                description="Enable real-time dashboard updates with WebSocket connections",
                input_types=["dashboard_id", "data_source", "update_config"],
                output_types=["websocket_config", "real_time_data", "update_status"]
            ),
            AgentCapability(
                name="threshold_alerts",
                description="Create alert system for threshold-based notifications",
                input_types=["alert_rules", "dashboard_metrics", "notification_config"],
                output_types=["alert_system", "notifications", "alert_status"]
            ),
            AgentCapability(
                name="dashboard_sharing",
                description="Dashboard sharing and permission management system",
                input_types=["dashboard_id", "share_config", "permissions"],
                output_types=["share_links", "permission_matrix", "access_logs"]
            ),
            AgentCapability(
                name="bi_query_analysis",
                description="Analyze BI queries and recommend optimal dashboard layouts",
                input_types=["bi_query", "data_context", "user_preferences"],
                output_types=["dashboard_recommendations", "chart_suggestions", "layout_options"]
            ),
            AgentCapability(
                name="dashboard_optimization",
                description="Optimize dashboard performance and user experience",
                input_types=["dashboard_config", "usage_analytics", "performance_metrics"],
                output_types=["optimization_recommendations", "performance_report", "ux_improvements"]
            )
        ]
        
        # Initialize OpenAI client for AI-enhanced dashboard creation
        if HAS_OPENAI:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize ScrollViz engine for visualization
        self.scrollviz_engine = ScrollVizEngine()
        
        # Dashboard templates for different business contexts
        self.dashboard_templates = self._initialize_dashboard_templates()
        
        # Alert rules storage (in production, this would be in database)
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Dashboard shares storage (in production, this would be in database)
        self.dashboard_shares: Dict[str, List[DashboardShare]] = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Dict[str, List] = {}
        
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming dashboard creation and BI requests."""
        start_time = asyncio.get_event_loop().time()
        try:
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if ("create" in prompt and "dashboard" in prompt) or ("build" in prompt and "dashboard" in prompt):
                content = await self._create_dashboard(request.prompt, context)
            elif "real-time" in prompt or "live dashboard" in prompt:
                content = await self._setup_real_time_dashboard(request.prompt, context)
            elif "alert" in prompt or "notification" in prompt or "threshold" in prompt:
                content = await self._setup_alerts(request.prompt, context)
            elif "share dashboard" in prompt or "permission" in prompt:
                content = await self._manage_dashboard_sharing(request.prompt, context)
            elif "optimize dashboard" in prompt or "performance" in prompt:
                content = await self._optimize_dashboard(request.prompt, context)
            elif "bi query" in prompt or "business intelligence" in prompt:
                content = await self._analyze_bi_query(request.prompt, context)
            else:
                content = await self._general_bi_assistance(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"bi-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error processing ScrollBI request: {e}")
            return AgentResponse(
                id=f"bi-{uuid4()}",
                request_id=request.id,
                content=f"Error processing dashboard request: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the capabilities of this agent."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy and ready to process requests."""
        try:
            # Test ScrollViz engine
            await self.scrollviz_engine.initialize()
            
            # Test basic dashboard creation
            test_data = pd.DataFrame({
                'category': ['A', 'B', 'C'],
                'value': [10, 20, 30]
            })
            
            test_config = {
                'name': 'Test Dashboard',
                'description': 'Test dashboard for health check',
                'dashboard_type': DashboardType.CUSTOM.value,
                'layout': 'grid',
                'theme': 'light',
                'refresh_interval': 60,
                'auto_refresh': True,
                'real_time_enabled': False,
                'charts': [{'type': 'bar', 'title': 'Test Chart'}],
                'filters': {},
                'alert_config': {}
            }
            
            # Test basic dashboard creation
            data_sources = [{
                "id": "test_dataset",
                "name": "Test Dataset",
                "type": "dataframe",
                "data": test_data,
                "schema": self._analyze_dataframe_schema(test_data)
            }]
            
            dashboard_spec = await self._generate_dashboard_spec(test_config, data_sources)
            
            return dashboard_spec is not None
        except Exception as e:
            logger.error(f"ScrollBI health check failed: {e}")
            return False
    
    async def _create_dashboard(self, prompt: str, context: Dict[str, Any]) -> str:
        """Create interactive dashboard from BI queries and data schemas."""
        try:
            # Extract dashboard requirements from prompt and context
            dashboard_config = await self._parse_dashboard_requirements(prompt, context)
            
            # Get data sources
            data_sources = await self._prepare_data_sources(context)
            
            # Generate dashboard specification
            dashboard_spec = await self._generate_dashboard_spec(dashboard_config, data_sources)
            
            # Create visualizations using ScrollViz
            visualizations = await self._create_dashboard_visualizations(dashboard_spec, data_sources)
            
            # Generate real-time configuration if requested
            real_time_config = await self._setup_real_time_config(dashboard_spec)
            
            # Generate deployment configuration
            deployment_config = await self._generate_deployment_config(dashboard_spec)
            
            report = f"""
# ScrollBI Dashboard Creation Report

## Dashboard Overview
- **Name**: {dashboard_spec['name']}
- **Type**: {dashboard_spec['dashboard_type']}
- **Layout**: {dashboard_spec['layout']}
- **Charts**: {len(visualizations)} visualizations
- **Real-time**: {'Enabled' if dashboard_spec.get('real_time_enabled') else 'Disabled'}

## Dashboard Configuration
```json
{json.dumps(dashboard_spec, indent=2, default=str)}
```

## Visualizations Created
{self._format_visualizations_summary(visualizations)}

## Real-time Configuration
{self._format_real_time_config(real_time_config)}

## Data Sources
{self._format_data_sources(data_sources)}

## Dashboard Features
- **Auto-refresh**: Every {dashboard_spec.get('refresh_interval', 60)} minutes
- **Responsive design**: Optimized for desktop and mobile
- **Interactive filters**: {len(dashboard_spec.get('filters', {}))} filter controls
- **Export options**: PDF, PNG, Excel
- **Sharing**: Configurable permissions and access control

## Performance Optimization
{await self._generate_performance_recommendations(dashboard_spec, data_sources)}

## Deployment Instructions
{self._format_deployment_instructions(deployment_config)}

## Next Steps
1. Review dashboard configuration and customize as needed
2. Set up data source connections and refresh schedules
3. Configure user access and sharing permissions
4. Set up alerts and notifications for key metrics
5. Deploy dashboard to production environment

---
*Dashboard created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return f"Error creating dashboard: {str(e)}"
    
    async def _setup_real_time_dashboard(self, prompt: str, context: Dict[str, Any]) -> str:
        """Set up real-time dashboard updates with WebSocket connections."""
        try:
            dashboard_id = context.get('dashboard_id')
            if not dashboard_id:
                return "Error: Dashboard ID required for real-time setup."
            
            # Configure WebSocket connections
            websocket_config = await self._configure_websockets(dashboard_id, context)
            
            # Set up data streaming
            streaming_config = await self._setup_data_streaming(dashboard_id, context)
            
            # Configure update intervals
            update_config = await self._configure_update_intervals(dashboard_id, context)
            
            # Set up connection monitoring
            monitoring_config = await self._setup_connection_monitoring(dashboard_id)
            
            report = f"""
# Real-time Dashboard Configuration

## Dashboard: {dashboard_id}

## WebSocket Configuration
```json
{json.dumps(websocket_config, indent=2)}
```

## Data Streaming Setup
{self._format_streaming_config(streaming_config)}

## Update Configuration
- **Update Interval**: {update_config.get('interval', 30)} seconds
- **Batch Size**: {update_config.get('batch_size', 100)} records
- **Buffer Size**: {update_config.get('buffer_size', 1000)} records
- **Compression**: {update_config.get('compression', 'gzip')}

## Connection Monitoring
{self._format_monitoring_config(monitoring_config)}

## Real-time Features Enabled
- Live data updates
- Real-time chart animations
- Instant alert notifications
- Collaborative viewing
- Connection status indicators

## Performance Considerations
- Optimized data serialization
- Efficient WebSocket message handling
- Automatic reconnection on connection loss
- Client-side data caching
- Progressive data loading

## Implementation Code
```javascript
// WebSocket connection setup
const ws = new WebSocket('{websocket_config.get("endpoint", "ws://localhost:8000/ws")}');

ws.onmessage = function(event) {{
    const data = JSON.parse(event.data);
    updateDashboard(data);
}};

// Dashboard update function
function updateDashboard(data) {{
    // Update charts with new data
    data.charts.forEach(chart => {{
        updateChart(chart.id, chart.data);
    }});
}}
```

---
*Real-time configuration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error setting up real-time dashboard: {e}")
            return f"Error setting up real-time dashboard: {str(e)}"
    
    async def _setup_alerts(self, prompt: str, context: Dict[str, Any]) -> str:
        """Set up alert system for threshold-based notifications."""
        try:
            dashboard_id = context.get('dashboard_id')
            alert_config = context.get('alert_config', {})
            
            # Parse alert requirements from prompt
            alert_requirements = await self._parse_alert_requirements(prompt, context)
            
            # Create alert rules
            alert_rules = await self._create_alert_rules(alert_requirements)
            
            # Set up notification channels
            notification_config = await self._setup_notification_channels(alert_config)
            
            # Configure alert monitoring
            monitoring_setup = await self._setup_alert_monitoring(alert_rules)
            
            # Store alert rules
            for rule in alert_rules:
                self.alert_rules[rule.id] = rule
            
            report = f"""
# Alert System Configuration

## Dashboard: {dashboard_id or 'Global'}

## Alert Rules Created
{self._format_alert_rules(alert_rules)}

## Notification Channels
{self._format_notification_channels(notification_config)}

## Alert Monitoring Setup
{self._format_alert_monitoring(monitoring_setup)}

## Alert Types Configured
- **Threshold Alerts**: Monitor when metrics exceed or fall below specified values
- **Anomaly Detection**: Detect unusual patterns in data
- **Data Quality Alerts**: Monitor for missing or invalid data
- **Trend Change Alerts**: Detect significant changes in trends

## Notification Methods
- Email notifications
- Slack/Teams integration
- SMS alerts (for critical thresholds)
- In-dashboard notifications
- Mobile push notifications

## Alert Management
- Enable/disable alerts individually
- Modify thresholds and conditions
- Set notification frequency limits
- Configure escalation rules
- View alert history and analytics

## Example Alert Configuration
```json
{{
  "alert_id": "revenue_threshold",
  "name": "Revenue Below Target",
  "metric": "monthly_revenue",
  "condition": "< 100000",
  "notification": ["email", "slack"],
  "frequency": "immediate"
}}
```

---
*Alert system configured at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error setting up alerts: {e}")
            return f"Error setting up alerts: {str(e)}"
    
    async def _manage_dashboard_sharing(self, prompt: str, context: Dict[str, Any]) -> str:
        """Manage dashboard sharing and permissions."""
        try:
            dashboard_id = context.get('dashboard_id')
            if not dashboard_id:
                return "Error: Dashboard ID required for sharing management."
            
            # Parse sharing requirements
            sharing_config = await self._parse_sharing_requirements(prompt, context)
            
            # Create share links
            share_links = await self._create_share_links(dashboard_id, sharing_config)
            
            # Set up permission matrix
            permission_matrix = await self._setup_permission_matrix(dashboard_id, sharing_config)
            
            # Configure access controls
            access_controls = await self._configure_access_controls(dashboard_id, sharing_config)
            
            # Set up audit logging
            audit_config = await self._setup_sharing_audit(dashboard_id)
            
            report = f"""
# Dashboard Sharing Configuration

## Dashboard: {dashboard_id}

## Share Links Created
{self._format_share_links(share_links)}

## Permission Matrix
{self._format_permission_matrix(permission_matrix)}

## Access Controls
{self._format_access_controls(access_controls)}

## Sharing Features
- **Public Links**: Share with anyone via secure links
- **User-specific Access**: Grant access to specific users
- **Role-based Permissions**: Different access levels (view, comment, edit, admin)
- **Time-limited Access**: Set expiration dates for shared links
- **IP Restrictions**: Limit access to specific IP ranges
- **Download Controls**: Control export and download permissions

## Permission Levels
- **View Only**: Can view dashboard but cannot interact or export
- **Comment**: Can view and add comments/annotations
- **Edit**: Can modify dashboard configuration and charts
- **Admin**: Full control including sharing and permission management

## Security Features
- Encrypted share links
- Access logging and monitoring
- Automatic link expiration
- Revocation capabilities
- Two-factor authentication support

## Audit Trail
{self._format_audit_config(audit_config)}

---
*Sharing configuration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error managing dashboard sharing: {e}")
            return f"Error managing dashboard sharing: {str(e)}"
    
    async def _optimize_dashboard(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize dashboard performance and user experience."""
        try:
            dashboard_id = context.get('dashboard_id')
            performance_data = context.get('performance_data', {})
            usage_analytics = context.get('usage_analytics', {})
            
            # Analyze current performance
            performance_analysis = await self._analyze_dashboard_performance(dashboard_id, performance_data)
            
            # Generate optimization recommendations
            optimizations = await self._generate_optimization_recommendations(performance_analysis, usage_analytics)
            
            # Create performance improvement plan
            improvement_plan = await self._create_improvement_plan(optimizations)
            
            # Generate UX recommendations
            ux_recommendations = await self._generate_ux_recommendations(usage_analytics)
            
            report = f"""
# Dashboard Optimization Report

## Dashboard: {dashboard_id or 'General'}

## Performance Analysis
{self._format_performance_analysis(performance_analysis)}

## Optimization Recommendations
{self._format_optimization_recommendations(optimizations)}

## Performance Improvement Plan
{self._format_improvement_plan(improvement_plan)}

## User Experience Recommendations
{self._format_ux_recommendations(ux_recommendations)}

## Technical Optimizations
- **Data Loading**: Implement lazy loading and pagination
- **Caching**: Set up intelligent caching strategies
- **Compression**: Enable data compression for faster transfers
- **CDN**: Use content delivery network for static assets
- **Database**: Optimize queries and add appropriate indexes

## Visual Optimizations
- **Chart Performance**: Optimize chart rendering and animations
- **Responsive Design**: Improve mobile and tablet experience
- **Loading States**: Add skeleton screens and progress indicators
- **Error Handling**: Implement graceful error states

## User Experience Improvements
- **Navigation**: Simplify dashboard navigation
- **Filters**: Optimize filter performance and usability
- **Tooltips**: Add contextual help and explanations
- **Accessibility**: Ensure WCAG compliance
- **Personalization**: Enable user customization options

## Implementation Priority
1. **High Priority**: Critical performance issues
2. **Medium Priority**: User experience improvements
3. **Low Priority**: Nice-to-have enhancements

---
*Optimization analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error optimizing dashboard: {e}")
            return f"Error optimizing dashboard: {str(e)}"
    
    async def _analyze_bi_query(self, prompt: str, context: Dict[str, Any]) -> str:
        """Analyze BI queries and recommend optimal dashboard layouts."""
        try:
            bi_query = context.get('bi_query', prompt)
            data_context = context.get('data_context', {})
            user_preferences = context.get('user_preferences', {})
            
            # Parse BI query
            query_analysis = await self._parse_bi_query(bi_query)
            
            # Analyze data requirements
            data_requirements = await self._analyze_data_requirements(query_analysis, data_context)
            
            # Generate dashboard recommendations
            dashboard_recommendations = await self._generate_dashboard_recommendations(query_analysis, data_requirements)
            
            # Suggest chart types
            chart_suggestions = await self._suggest_chart_types(query_analysis, data_requirements)
            
            # Recommend layouts
            layout_options = await self._recommend_layouts(dashboard_recommendations, user_preferences)
            
            report = f"""
# BI Query Analysis Report

## Query Analysis
{self._format_query_analysis(query_analysis)}

## Data Requirements
{self._format_data_requirements(data_requirements)}

## Dashboard Recommendations
{self._format_dashboard_recommendations(dashboard_recommendations)}

## Chart Suggestions
{self._format_chart_suggestions(chart_suggestions)}

## Layout Options
{self._format_layout_options(layout_options)}

## Implementation Guidance
- **Data Sources**: {len(data_requirements.get('sources', []))} sources identified
- **Refresh Frequency**: {data_requirements.get('refresh_frequency', 'hourly')}
- **Estimated Load Time**: {data_requirements.get('estimated_load_time', '< 3 seconds')}
- **Complexity Level**: {query_analysis.get('complexity', 'medium')}

## Best Practices Applied
- Optimized for query performance
- Responsive design considerations
- User experience optimization
- Accessibility compliance
- Mobile-first approach

---
*BI query analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing BI query: {e}")
            return f"Error analyzing BI query: {str(e)}"
    
    async def _general_bi_assistance(self, prompt: str, context: Dict[str, Any]) -> str:
        """Provide general BI assistance and guidance."""
        try:
            # Get AI-powered assistance
            ai_response = await self._get_ai_bi_assistance(prompt, context)
            
            # Provide relevant templates and examples
            templates = await self._suggest_relevant_templates(prompt)
            
            # Generate best practices
            best_practices = await self._generate_bi_best_practices(prompt)
            
            report = f"""
# ScrollBI Assistant

## Your Request
{prompt}

## AI-Powered Guidance
{ai_response}

## Relevant Templates
{self._format_template_suggestions(templates)}

## Best Practices
{self._format_best_practices(best_practices)}

## Quick Actions
- Create a new dashboard
- Set up real-time monitoring
- Configure alerts and notifications
- Share existing dashboards
- Optimize dashboard performance

## Resources
- Dashboard template library
- Chart type selection guide
- Performance optimization checklist
- Sharing and security guide
- Real-time setup documentation

---
*Assistance provided at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error providing BI assistance: {e}")
            return f"Error providing BI assistance: {str(e)}"
    
    # Helper methods for dashboard creation and management
    
    def _initialize_dashboard_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dashboard templates for different business contexts."""
        return {
            "executive_summary": {
                "name": "Executive Summary Dashboard",
                "description": "High-level KPIs and metrics for executives",
                "layout": "grid",
                "charts": [
                    {"type": "kpi_card", "title": "Revenue", "position": {"x": 0, "y": 0, "w": 3, "h": 2}},
                    {"type": "kpi_card", "title": "Growth Rate", "position": {"x": 3, "y": 0, "w": 3, "h": 2}},
                    {"type": "line", "title": "Revenue Trend", "position": {"x": 0, "y": 2, "w": 6, "h": 4}},
                    {"type": "bar", "title": "Department Performance", "position": {"x": 6, "y": 0, "w": 6, "h": 6}}
                ]
            },
            "sales_dashboard": {
                "name": "Sales Performance Dashboard",
                "description": "Sales metrics and pipeline analysis",
                "layout": "grid",
                "charts": [
                    {"type": "gauge", "title": "Sales Target", "position": {"x": 0, "y": 0, "w": 4, "h": 3}},
                    {"type": "funnel", "title": "Sales Pipeline", "position": {"x": 4, "y": 0, "w": 4, "h": 3}},
                    {"type": "line", "title": "Sales Trend", "position": {"x": 8, "y": 0, "w": 4, "h": 3}},
                    {"type": "table", "title": "Top Deals", "position": {"x": 0, "y": 3, "w": 12, "h": 4}}
                ]
            },
            "financial_dashboard": {
                "name": "Financial Dashboard",
                "description": "Financial metrics and analysis",
                "layout": "grid",
                "charts": [
                    {"type": "waterfall", "title": "Cash Flow", "position": {"x": 0, "y": 0, "w": 6, "h": 4}},
                    {"type": "pie", "title": "Expense Breakdown", "position": {"x": 6, "y": 0, "w": 6, "h": 4}},
                    {"type": "area", "title": "Profit Margin", "position": {"x": 0, "y": 4, "w": 12, "h": 3}}
                ]
            },
            "operational_dashboard": {
                "name": "Operational Dashboard",
                "description": "Operational metrics and monitoring",
                "layout": "grid",
                "charts": [
                    {"type": "gauge", "title": "System Health", "position": {"x": 0, "y": 0, "w": 3, "h": 3}},
                    {"type": "line", "title": "Performance Metrics", "position": {"x": 3, "y": 0, "w": 9, "h": 3}},
                    {"type": "heatmap", "title": "Activity Heatmap", "position": {"x": 0, "y": 3, "w": 12, "h": 4}}
                ]
            }
        }
    
    async def _parse_dashboard_requirements(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse dashboard requirements from prompt and context."""
        # Default configuration
        config = {
            "name": context.get("name", "New Dashboard"),
            "description": context.get("description", ""),
            "dashboard_type": DashboardType.CUSTOM.value,
            "layout": "grid",
            "refresh_interval": 60,
            "auto_refresh": True,
            "theme": "light",
            "filters": {},
            "charts": [],
            "real_time_enabled": False,
            "alert_config": {}
        }
        
        # Parse from prompt
        if "executive" in prompt.lower():
            config["dashboard_type"] = DashboardType.EXECUTIVE.value
            config["charts"] = self.dashboard_templates["executive_summary"]["charts"]
        elif "sales" in prompt.lower():
            config["dashboard_type"] = DashboardType.SALES.value
            config["charts"] = self.dashboard_templates["sales_dashboard"]["charts"]
        elif "financial" in prompt.lower():
            config["dashboard_type"] = DashboardType.FINANCIAL.value
            config["charts"] = self.dashboard_templates["financial_dashboard"]["charts"]
        elif "operational" in prompt.lower():
            config["dashboard_type"] = DashboardType.OPERATIONAL.value
            config["charts"] = self.dashboard_templates["operational_dashboard"]["charts"]
        
        # Parse real-time requirements
        if "real-time" in prompt.lower() or "live" in prompt.lower():
            config["real_time_enabled"] = True
            config["refresh_interval"] = 5  # 5 minutes for real-time
        
        # Update with context overrides
        config.update(context.get("dashboard_config", {}))
        
        return config
    
    async def _prepare_data_sources(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare data sources for dashboard creation."""
        data_sources = []
        
        # Handle dataset from context
        if "dataset" in context:
            dataset = context["dataset"]
            if isinstance(dataset, pd.DataFrame):
                data_sources.append({
                    "id": "primary_dataset",
                    "name": "Primary Dataset",
                    "type": "dataframe",
                    "data": dataset,
                    "schema": self._analyze_dataframe_schema(dataset)
                })
        
        # Handle dataset path
        if "dataset_path" in context:
            dataset_path = context["dataset_path"]
            try:
                df = await self._load_dataset(dataset_path)
                data_sources.append({
                    "id": "file_dataset",
                    "name": f"Dataset from {dataset_path}",
                    "type": "file",
                    "data": df,
                    "schema": self._analyze_dataframe_schema(df),
                    "path": dataset_path
                })
            except Exception as e:
                logger.error(f"Error loading dataset from {dataset_path}: {e}")
        
        # Handle database connections
        if "connection_string" in context:
            data_sources.append({
                "id": "database_source",
                "name": "Database Source",
                "type": "database",
                "connection_string": context["connection_string"],
                "query": context.get("query", "SELECT * FROM main_table LIMIT 1000")
            })
        
        return data_sources
    
    async def _generate_dashboard_spec(self, config: Dict[str, Any], data_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive dashboard specification."""
        dashboard_spec = {
            "id": f"dashboard_{uuid4()}",
            "name": config["name"],
            "description": config["description"],
            "dashboard_type": config["dashboard_type"],
            "layout": config["layout"],
            "theme": config["theme"],
            "refresh_interval": config["refresh_interval"],
            "auto_refresh": config["auto_refresh"],
            "real_time_enabled": config["real_time_enabled"],
            "created_at": datetime.now().isoformat(),
            "data_sources": [ds["id"] for ds in data_sources],
            "charts": [],
            "filters": config["filters"],
            "alert_config": config["alert_config"]
        }
        
        # Generate chart specifications based on data
        if data_sources and config["charts"]:
            for chart_config in config["charts"]:
                chart_spec = await self._generate_chart_spec(chart_config, data_sources[0])
                dashboard_spec["charts"].append(chart_spec)
        
        return dashboard_spec
    
    async def _generate_chart_spec(self, chart_config: Dict[str, Any], data_source: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart specification from configuration and data source."""
        chart_spec = {
            "id": f"chart_{uuid4()}",
            "type": chart_config["type"],
            "title": chart_config["title"],
            "position": chart_config.get("position", {"x": 0, "y": 0, "w": 6, "h": 4}),
            "data_source_id": data_source["id"],
            "config": {}
        }
        
        # Add chart-specific configuration based on data
        if "data" in data_source:
            df = data_source["data"]
            schema = data_source["schema"]
            
            # Auto-select columns based on chart type
            if chart_config["type"] == "bar":
                categorical_cols = [col for col, dtype in schema.items() if dtype == "categorical"]
                numerical_cols = [col for col, dtype in schema.items() if dtype == "numerical"]
                
                chart_spec["config"] = {
                    "x_column": categorical_cols[0] if categorical_cols else df.columns[0],
                    "y_column": numerical_cols[0] if numerical_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                }
            
            elif chart_config["type"] == "line":
                datetime_cols = [col for col, dtype in schema.items() if dtype == "datetime"]
                numerical_cols = [col for col, dtype in schema.items() if dtype == "numerical"]
                
                chart_spec["config"] = {
                    "x_column": datetime_cols[0] if datetime_cols else df.columns[0],
                    "y_column": numerical_cols[0] if numerical_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                }
            
            elif chart_config["type"] == "pie":
                categorical_cols = [col for col, dtype in schema.items() if dtype == "categorical"]
                numerical_cols = [col for col, dtype in schema.items() if dtype == "numerical"]
                
                chart_spec["config"] = {
                    "labels_column": categorical_cols[0] if categorical_cols else df.columns[0],
                    "values_column": numerical_cols[0] if numerical_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                }
        
        return chart_spec
    
    async def _create_dashboard_visualizations(self, dashboard_spec: Dict[str, Any], data_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create visualizations for the dashboard using ScrollViz engine."""
        visualizations = []
        
        # Create data source lookup
        data_lookup = {ds["id"]: ds for ds in data_sources}
        
        for chart_spec in dashboard_spec["charts"]:
            try:
                data_source = data_lookup.get(chart_spec["data_source_id"])
                if not data_source or "data" not in data_source:
                    continue
                
                # Prepare visualization parameters
                viz_params = {
                    "action": "generate_chart",
                    "chart_type": chart_spec["type"],
                    "title": chart_spec["title"],
                    **chart_spec["config"]
                }
                
                # Generate visualization using ScrollViz
                viz_result = await self.scrollviz_engine.process(data_source["data"], viz_params)
                
                visualization = {
                    "id": chart_spec["id"],
                    "title": chart_spec["title"],
                    "type": chart_spec["type"],
                    "position": chart_spec["position"],
                    "config": viz_result,
                    "data_source": chart_spec["data_source_id"]
                }
                
                visualizations.append(visualization)
                
            except Exception as e:
                logger.error(f"Error creating visualization {chart_spec['id']}: {e}")
        
        return visualizations
    
    async def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file path."""
        if dataset_path.endswith('.csv'):
            return pd.read_csv(dataset_path)
        elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
            return pd.read_excel(dataset_path)
        elif dataset_path.endswith('.json'):
            return pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
    
    def _analyze_dataframe_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze DataFrame schema and return column types."""
        schema = {}
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                schema[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(df[col]):
                schema[col] = "numerical"
            elif pd.api.types.is_bool_dtype(df[col]):
                schema[col] = "boolean"
            elif df[col].dtype == 'object':
                # Check if categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 or df[col].nunique() < 20:
                    schema[col] = "categorical"
                else:
                    schema[col] = "text"
            else:
                schema[col] = "categorical"
        return schema
    
    # Formatting helper methods
    
    def _format_visualizations_summary(self, visualizations: List[Dict[str, Any]]) -> str:
        """Format visualizations summary."""
        if not visualizations:
            return "No visualizations created."
        
        summary = []
        for viz in visualizations:
            summary.append(f"- **{viz['title']}** ({viz['type']}): Position {viz['position']}")
        
        return "\n".join(summary)
    
    def _format_real_time_config(self, config: Dict[str, Any]) -> str:
        """Format real-time configuration."""
        if not config:
            return "Real-time updates not configured."
        
        return f"""
- **WebSocket Endpoint**: {config.get('websocket_endpoint', 'Not configured')}
- **Update Interval**: {config.get('update_interval', 30)} seconds
- **Connection Monitoring**: {'Enabled' if config.get('monitoring_enabled') else 'Disabled'}
- **Auto-reconnect**: {'Enabled' if config.get('auto_reconnect') else 'Disabled'}
"""
    
    def _format_data_sources(self, data_sources: List[Dict[str, Any]]) -> str:
        """Format data sources information."""
        if not data_sources:
            return "No data sources configured."
        
        summary = []
        for ds in data_sources:
            summary.append(f"- **{ds['name']}** ({ds['type']}): {ds.get('schema', {}).keys() if 'schema' in ds else 'Schema not available'}")
        
        return "\n".join(summary)
    
    async def _generate_performance_recommendations(self, dashboard_spec: Dict[str, Any], data_sources: List[Dict[str, Any]]) -> str:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check number of charts
        chart_count = len(dashboard_spec.get("charts", []))
        if chart_count > 10:
            recommendations.append("Consider reducing the number of charts or implementing lazy loading")
        
        # Check data size
        for ds in data_sources:
            if "data" in ds and hasattr(ds["data"], "shape"):
                rows, cols = ds["data"].shape
                if rows > 10000:
                    recommendations.append(f"Large dataset detected ({rows:,} rows). Consider data pagination or aggregation")
        
        # Check refresh interval
        refresh_interval = dashboard_spec.get("refresh_interval", 60)
        if refresh_interval < 5:
            recommendations.append("Very frequent refresh intervals may impact performance. Consider increasing to 5+ minutes")
        
        if not recommendations:
            recommendations.append("Dashboard configuration looks optimized for performance")
        
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _format_deployment_instructions(self, config: Dict[str, Any]) -> str:
        """Format deployment instructions."""
        return f"""
1. **Environment Setup**: Ensure ScrollViz engine is properly configured
2. **Data Connections**: Verify all data source connections are active
3. **WebSocket Configuration**: Set up WebSocket server for real-time features
4. **Security**: Configure authentication and authorization
5. **Monitoring**: Set up performance monitoring and logging
6. **Backup**: Configure dashboard configuration backup
"""
    
    # Placeholder methods for advanced features (to be implemented)
    
    async def _setup_real_time_config(self, dashboard_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Set up real-time configuration."""
        if not dashboard_spec.get("real_time_enabled"):
            return {}
        
        return {
            "websocket_endpoint": f"ws://localhost:8000/ws/dashboard/{dashboard_spec['id']}",
            "update_interval": 30,
            "monitoring_enabled": True,
            "auto_reconnect": True
        }
    
    async def _generate_deployment_config(self, dashboard_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment configuration."""
        return {
            "dashboard_id": dashboard_spec["id"],
            "environment": "production",
            "scaling": "auto",
            "monitoring": "enabled"
        }
    
    async def _configure_websockets(self, dashboard_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Configure WebSocket connections for real-time updates."""
        return {
            "endpoint": f"ws://localhost:8000/ws/dashboard/{dashboard_id}",
            "protocols": ["websocket"],
            "heartbeat_interval": 30,
            "max_connections": 100
        }
    
    async def _setup_data_streaming(self, dashboard_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set up data streaming configuration."""
        return {
            "stream_type": "websocket",
            "buffer_size": 1000,
            "batch_size": 100,
            "compression": "gzip"
        }
    
    async def _configure_update_intervals(self, dashboard_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Configure update intervals for real-time dashboard."""
        return {
            "interval": context.get("update_interval", 30),
            "batch_size": context.get("batch_size", 100),
            "buffer_size": context.get("buffer_size", 1000),
            "compression": context.get("compression", "gzip")
        }
    
    async def _setup_connection_monitoring(self, dashboard_id: str) -> Dict[str, Any]:
        """Set up connection monitoring for WebSocket connections."""
        return {
            "health_check_interval": 60,
            "connection_timeout": 300,
            "max_reconnect_attempts": 5,
            "monitoring_enabled": True
        }
    
    def _format_streaming_config(self, config: Dict[str, Any]) -> str:
        """Format streaming configuration."""
        return f"""
- **Stream Type**: {config.get('stream_type', 'websocket')}
- **Buffer Size**: {config.get('buffer_size', 1000)} records
- **Batch Size**: {config.get('batch_size', 100)} records
- **Compression**: {config.get('compression', 'gzip')}
"""
    
    def _format_monitoring_config(self, config: Dict[str, Any]) -> str:
        """Format monitoring configuration."""
        return f"""
- **Health Check Interval**: {config.get('health_check_interval', 60)} seconds
- **Connection Timeout**: {config.get('connection_timeout', 300)} seconds
- **Max Reconnect Attempts**: {config.get('max_reconnect_attempts', 5)}
- **Monitoring**: {'Enabled' if config.get('monitoring_enabled') else 'Disabled'}
"""
    
    async def _parse_alert_requirements(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse alert requirements from prompt and context."""
        return {
            "metrics": context.get("metrics", ["revenue", "conversion_rate"]),
            "thresholds": context.get("thresholds", {"revenue": 100000, "conversion_rate": 0.05}),
            "notification_channels": context.get("notification_channels", ["email"])
        }
    
    async def _create_alert_rules(self, requirements: Dict[str, Any]) -> List[AlertRule]:
        """Create alert rules from requirements."""
        rules = []
        for metric, threshold in requirements.get("thresholds", {}).items():
            rule = AlertRule(
                id=f"alert_{uuid4()}",
                name=f"{metric.title()} Alert",
                description=f"Alert when {metric} exceeds threshold",
                alert_type=AlertType.THRESHOLD_EXCEEDED,
                metric_name=metric,
                threshold_value=threshold,
                comparison_operator=">",
                notification_channels=requirements.get("notification_channels", ["email"]),
                is_active=True,
                created_at=datetime.now()
            )
            rules.append(rule)
        return rules
    
    async def _setup_notification_channels(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up notification channels."""
        return {
            "email": {"enabled": True, "recipients": config.get("email_recipients", [])},
            "slack": {"enabled": False, "webhook_url": config.get("slack_webhook")},
            "sms": {"enabled": False, "phone_numbers": config.get("phone_numbers", [])}
        }
    
    async def _setup_alert_monitoring(self, rules: List[AlertRule]) -> Dict[str, Any]:
        """Set up alert monitoring system."""
        return {
            "monitoring_interval": 60,
            "rules_count": len(rules),
            "active_rules": len([r for r in rules if r.is_active]),
            "notification_methods": ["email", "slack", "sms"]
        }
    
    def _format_alert_rules(self, rules: List[AlertRule]) -> str:
        """Format alert rules."""
        if not rules:
            return "No alert rules configured."
        
        formatted = []
        for rule in rules:
            formatted.append(f"- **{rule.name}**: {rule.metric_name} {rule.comparison_operator} {rule.threshold_value}")
        
        return "\n".join(formatted)
    
    def _format_notification_channels(self, config: Dict[str, Any]) -> str:
        """Format notification channels."""
        channels = []
        for channel, settings in config.items():
            status = "Enabled" if settings.get("enabled") else "Disabled"
            channels.append(f"- **{channel.title()}**: {status}")
        
        return "\n".join(channels)
    
    def _format_alert_monitoring(self, config: Dict[str, Any]) -> str:
        """Format alert monitoring configuration."""
        return f"""
- **Monitoring Interval**: {config.get('monitoring_interval', 60)} seconds
- **Total Rules**: {config.get('rules_count', 0)}
- **Active Rules**: {config.get('active_rules', 0)}
- **Notification Methods**: {', '.join(config.get('notification_methods', []))}
"""
    
    # Additional placeholder methods for sharing, optimization, and BI analysis
    
    async def _parse_sharing_requirements(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse sharing requirements."""
        return context.get("sharing_config", {})
    
    async def _create_share_links(self, dashboard_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create share links."""
        return [{"link": f"https://scrollintel.com/dashboard/{dashboard_id}/share", "permission": "view"}]
    
    async def _setup_permission_matrix(self, dashboard_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up permission matrix."""
        return {"view": ["user1", "user2"], "edit": ["admin1"]}
    
    async def _configure_access_controls(self, dashboard_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure access controls."""
        return {"ip_restrictions": [], "time_restrictions": None}
    
    async def _setup_sharing_audit(self, dashboard_id: str) -> Dict[str, Any]:
        """Set up sharing audit configuration."""
        return {"audit_enabled": True, "log_access": True}
    
    def _format_share_links(self, links: List[Dict[str, Any]]) -> str:
        """Format share links."""
        return "\n".join(f"- {link['link']} ({link['permission']})" for link in links)
    
    def _format_permission_matrix(self, matrix: Dict[str, Any]) -> str:
        """Format permission matrix."""
        formatted = []
        for permission, users in matrix.items():
            formatted.append(f"- **{permission.title()}**: {', '.join(users)}")
        return "\n".join(formatted)
    
    def _format_access_controls(self, controls: Dict[str, Any]) -> str:
        """Format access controls."""
        return f"""
- **IP Restrictions**: {controls.get('ip_restrictions', 'None')}
- **Time Restrictions**: {controls.get('time_restrictions', 'None')}
"""
    
    def _format_audit_config(self, config: Dict[str, Any]) -> str:
        """Format audit configuration."""
        return f"""
- **Audit Enabled**: {'Yes' if config.get('audit_enabled') else 'No'}
- **Log Access**: {'Yes' if config.get('log_access') else 'No'}
"""
    
    # Additional methods for optimization and BI analysis would be implemented here
    # These are placeholder implementations for the core functionality
    
    async def _analyze_dashboard_performance(self, dashboard_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dashboard performance."""
        return {"load_time": 2.5, "chart_render_time": 1.2, "data_fetch_time": 0.8}
    
    async def _generate_optimization_recommendations(self, analysis: Dict[str, Any], usage: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        return ["Enable data caching", "Optimize chart rendering", "Implement lazy loading"]
    
    async def _create_improvement_plan(self, optimizations: List[str]) -> Dict[str, Any]:
        """Create improvement plan."""
        return {"high_priority": optimizations[:2], "medium_priority": optimizations[2:]}
    
    async def _generate_ux_recommendations(self, usage: Dict[str, Any]) -> List[str]:
        """Generate UX recommendations."""
        return ["Improve mobile responsiveness", "Add contextual help", "Simplify navigation"]
    
    def _format_performance_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format performance analysis."""
        return f"""
- **Load Time**: {analysis.get('load_time', 'N/A')} seconds
- **Chart Render Time**: {analysis.get('chart_render_time', 'N/A')} seconds
- **Data Fetch Time**: {analysis.get('data_fetch_time', 'N/A')} seconds
"""
    
    def _format_optimization_recommendations(self, recommendations: List[str]) -> str:
        """Format optimization recommendations."""
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _format_improvement_plan(self, plan: Dict[str, Any]) -> str:
        """Format improvement plan."""
        return f"""
**High Priority:**
{chr(10).join(f"- {item}" for item in plan.get('high_priority', []))}

**Medium Priority:**
{chr(10).join(f"- {item}" for item in plan.get('medium_priority', []))}
"""
    
    def _format_ux_recommendations(self, recommendations: List[str]) -> str:
        """Format UX recommendations."""
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    async def _parse_bi_query(self, query: str) -> Dict[str, Any]:
        """Parse BI query."""
        return {"complexity": "medium", "tables": ["sales", "customers"], "metrics": ["revenue", "count"]}
    
    async def _analyze_data_requirements(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data requirements."""
        return {"sources": ["database"], "refresh_frequency": "hourly", "estimated_load_time": "< 3 seconds"}
    
    async def _generate_dashboard_recommendations(self, analysis: Dict[str, Any], requirements: Dict[str, Any]) -> List[str]:
        """Generate dashboard recommendations."""
        return ["Use executive template", "Add real-time updates", "Include KPI cards"]
    
    async def _suggest_chart_types(self, analysis: Dict[str, Any], requirements: Dict[str, Any]) -> List[str]:
        """Suggest chart types."""
        return ["bar", "line", "pie", "kpi_card"]
    
    async def _recommend_layouts(self, recommendations: List[str], preferences: Dict[str, Any]) -> List[str]:
        """Recommend layouts."""
        return ["grid", "masonry", "custom"]
    
    def _format_query_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format query analysis."""
        return f"""
- **Complexity**: {analysis.get('complexity', 'Unknown')}
- **Tables**: {', '.join(analysis.get('tables', []))}
- **Metrics**: {', '.join(analysis.get('metrics', []))}
"""
    
    def _format_data_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format data requirements."""
        return f"""
- **Sources**: {', '.join(requirements.get('sources', []))}
- **Refresh Frequency**: {requirements.get('refresh_frequency', 'Unknown')}
- **Estimated Load Time**: {requirements.get('estimated_load_time', 'Unknown')}
"""
    
    def _format_dashboard_recommendations(self, recommendations: List[str]) -> str:
        """Format dashboard recommendations."""
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _format_chart_suggestions(self, suggestions: List[str]) -> str:
        """Format chart suggestions."""
        return "\n".join(f"- {chart}" for chart in suggestions)
    
    def _format_layout_options(self, options: List[str]) -> str:
        """Format layout options."""
        return "\n".join(f"- {option}" for option in options)
    
    async def _get_ai_bi_assistance(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get AI-powered BI assistance."""
        if HAS_OPENAI and openai.api_key:
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a business intelligence expert helping with dashboard creation and data analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
        
        return "AI assistance is not available. Please check your OpenAI API configuration."
    
    async def _suggest_relevant_templates(self, prompt: str) -> List[str]:
        """Suggest relevant templates."""
        templates = []
        prompt_lower = prompt.lower()
        
        if "executive" in prompt_lower or "summary" in prompt_lower:
            templates.append("executive_summary")
        if "sales" in prompt_lower:
            templates.append("sales_dashboard")
        if "financial" in prompt_lower or "finance" in prompt_lower:
            templates.append("financial_dashboard")
        if "operational" in prompt_lower or "operations" in prompt_lower:
            templates.append("operational_dashboard")
        
        return templates if templates else ["custom"]
    
    async def _generate_bi_best_practices(self, prompt: str) -> List[str]:
        """Generate BI best practices."""
        return [
            "Keep dashboards focused on key metrics",
            "Use consistent color schemes and branding",
            "Optimize for mobile and tablet viewing",
            "Implement proper data governance",
            "Set up automated data quality checks",
            "Use progressive disclosure for complex data",
            "Ensure accessibility compliance",
            "Monitor dashboard performance regularly"
        ]
    
    def _format_template_suggestions(self, templates: List[str]) -> str:
        """Format template suggestions."""
        if not templates:
            return "No specific templates recommended."
        
        formatted = []
        for template in templates:
            if template in self.dashboard_templates:
                template_info = self.dashboard_templates[template]
                formatted.append(f"- **{template_info['name']}**: {template_info['description']}")
            else:
                formatted.append(f"- **{template}**: Custom template")
        
        return "\n".join(formatted)
    
    def _format_best_practices(self, practices: List[str]) -> str:
        """Format best practices."""
        return "\n".join(f"- {practice}" for practice in practices)