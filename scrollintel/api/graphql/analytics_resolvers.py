"""
GraphQL Resolvers for Advanced Analytics Dashboard API

This module implements the GraphQL resolvers for flexible data querying
across all dashboard and analytics functionality.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import asdict

from ...core.dashboard_manager import DashboardManager
from ...engines.roi_calculator import ROICalculator
from ...engines.insight_generator import InsightGenerator
from ...engines.predictive_engine import PredictiveEngine
from ...core.data_connector import DataConnector
from ...core.template_engine import TemplateEngine
from ...engines.reporting_engine import ReportingEngine

logger = logging.getLogger(__name__)

class AnalyticsResolvers:
    """GraphQL resolvers for Advanced Analytics Dashboard API"""
    
    def __init__(self):
        self.dashboard_manager = DashboardManager()
        self.roi_calculator = ROICalculator()
        self.insight_generator = InsightGenerator()
        self.predictive_engine = PredictiveEngine()
        self.data_connector = DataConnector()
        self.template_engine = TemplateEngine()
        self.reporting_engine = ReportingEngine()
    
    # Query Resolvers
    
    async def resolve_dashboard(self, info, id: str) -> Optional[Dict]:
        """Resolve single dashboard by ID"""
        try:
            dashboard = await self.dashboard_manager.get_dashboard(id)
            if dashboard:
                return self._serialize_dashboard(dashboard)
            return None
        except Exception as e:
            logger.error(f"Error resolving dashboard {id}: {str(e)}")
            raise
    
    async def resolve_dashboards(
        self, 
        info, 
        type: Optional[str] = None,
        owner: Optional[str] = None,
        pagination: Optional[Dict] = None,
        sort: Optional[Dict] = None
    ) -> List[Dict]:
        """Resolve multiple dashboards with filtering"""
        try:
            filters = {}
            if type:
                filters['type'] = type
            if owner:
                filters['owner'] = owner
            
            # Apply pagination
            page = pagination.get('page', 1) if pagination else 1
            limit = pagination.get('limit', 20) if pagination else 20
            offset = (page - 1) * limit
            
            # Apply sorting
            sort_field = sort.get('field', 'created_at') if sort else 'created_at'
            sort_direction = sort.get('direction', 'desc') if sort else 'desc'
            
            dashboards = await self.dashboard_manager.list_dashboards(
                filters=filters,
                offset=offset,
                limit=limit,
                sort_field=sort_field,
                sort_direction=sort_direction
            )
            
            return [self._serialize_dashboard(d) for d in dashboards]
        except Exception as e:
            logger.error(f"Error resolving dashboards: {str(e)}")
            raise
    
    async def resolve_dashboard_metrics(
        self, 
        info, 
        id: str, 
        time_range: Optional[str] = None
    ) -> Optional[Dict]:
        """Resolve dashboard metrics"""
        try:
            # Parse time range
            start_date, end_date = self._parse_time_range(time_range)
            
            metrics = await self.dashboard_manager.get_dashboard_metrics(
                id, start_date, end_date
            )
            
            if metrics:
                return self._serialize_dashboard_metrics(metrics)
            return None
        except Exception as e:
            logger.error(f"Error resolving dashboard metrics {id}: {str(e)}")
            raise
    
    async def resolve_roi_analysis(self, info, id: str) -> Optional[Dict]:
        """Resolve single ROI analysis by ID"""
        try:
            analysis = await self.roi_calculator.get_analysis(id)
            if analysis:
                return self._serialize_roi_analysis(analysis)
            return None
        except Exception as e:
            logger.error(f"Error resolving ROI analysis {id}: {str(e)}")
            raise
    
    async def resolve_roi_analyses(
        self,
        info,
        project_id: Optional[str] = None,
        date_range: Optional[str] = None,
        pagination: Optional[Dict] = None
    ) -> List[Dict]:
        """Resolve multiple ROI analyses"""
        try:
            filters = {}
            if project_id:
                filters['project_id'] = project_id
            if date_range:
                start_date, end_date = self._parse_time_range(date_range)
                filters['date_range'] = (start_date, end_date)
            
            # Apply pagination
            page = pagination.get('page', 1) if pagination else 1
            limit = pagination.get('limit', 20) if pagination else 20
            offset = (page - 1) * limit
            
            analyses = await self.roi_calculator.list_analyses(
                filters=filters,
                offset=offset,
                limit=limit
            )
            
            return [self._serialize_roi_analysis(a) for a in analyses]
        except Exception as e:
            logger.error(f"Error resolving ROI analyses: {str(e)}")
            raise
    
    async def resolve_roi_trends(
        self, 
        info, 
        project_id: str, 
        time_range: str
    ) -> List[Dict]:
        """Resolve ROI trends for a project"""
        try:
            start_date, end_date = self._parse_time_range(time_range)
            trends = await self.roi_calculator.get_roi_trends(
                project_id, start_date, end_date
            )
            return [self._serialize_roi_trend(t) for t in trends]
        except Exception as e:
            logger.error(f"Error resolving ROI trends: {str(e)}")
            raise
    
    async def resolve_roi_comparison(
        self, 
        info, 
        project_ids: List[str]
    ) -> List[Dict]:
        """Resolve ROI comparison across projects"""
        try:
            comparison = await self.roi_calculator.compare_projects(project_ids)
            return [self._serialize_roi_analysis(a) for a in comparison]
        except Exception as e:
            logger.error(f"Error resolving ROI comparison: {str(e)}")
            raise
    
    async def resolve_insight(self, info, id: str) -> Optional[Dict]:
        """Resolve single insight by ID"""
        try:
            insight = await self.insight_generator.get_insight(id)
            if insight:
                return self._serialize_insight(insight)
            return None
        except Exception as e:
            logger.error(f"Error resolving insight {id}: {str(e)}")
            raise
    
    async def resolve_insights(
        self,
        info,
        type: Optional[str] = None,
        significance: Optional[float] = None,
        date_range: Optional[str] = None,
        pagination: Optional[Dict] = None,
        sort: Optional[Dict] = None
    ) -> List[Dict]:
        """Resolve multiple insights with filtering"""
        try:
            filters = {}
            if type:
                filters['type'] = type
            if significance:
                filters['significance'] = significance
            if date_range:
                start_date, end_date = self._parse_time_range(date_range)
                filters['date_range'] = (start_date, end_date)
            
            # Apply pagination
            page = pagination.get('page', 1) if pagination else 1
            limit = pagination.get('limit', 20) if pagination else 20
            offset = (page - 1) * limit
            
            # Apply sorting
            sort_field = sort.get('field', 'created_at') if sort else 'created_at'
            sort_direction = sort.get('direction', 'desc') if sort else 'desc'
            
            insights = await self.insight_generator.list_insights(
                filters=filters,
                offset=offset,
                limit=limit,
                sort_field=sort_field,
                sort_direction=sort_direction
            )
            
            return [self._serialize_insight(i) for i in insights]
        except Exception as e:
            logger.error(f"Error resolving insights: {str(e)}")
            raise
    
    async def resolve_insight_trends(
        self, 
        info, 
        time_range: str
    ) -> List[Dict]:
        """Resolve insight trends over time"""
        try:
            start_date, end_date = self._parse_time_range(time_range)
            trends = await self.insight_generator.get_insight_trends(
                start_date, end_date
            )
            return [self._serialize_insight(i) for i in trends]
        except Exception as e:
            logger.error(f"Error resolving insight trends: {str(e)}")
            raise
    
    async def resolve_forecast(self, info, id: str) -> Optional[Dict]:
        """Resolve single forecast by ID"""
        try:
            forecast = await self.predictive_engine.get_forecast(id)
            if forecast:
                return self._serialize_forecast(forecast)
            return None
        except Exception as e:
            logger.error(f"Error resolving forecast {id}: {str(e)}")
            raise
    
    async def resolve_forecasts(
        self,
        info,
        metric: Optional[str] = None,
        horizon: Optional[int] = None,
        pagination: Optional[Dict] = None
    ) -> List[Dict]:
        """Resolve multiple forecasts"""
        try:
            filters = {}
            if metric:
                filters['metric'] = metric
            if horizon:
                filters['horizon'] = horizon
            
            # Apply pagination
            page = pagination.get('page', 1) if pagination else 1
            limit = pagination.get('limit', 20) if pagination else 20
            offset = (page - 1) * limit
            
            forecasts = await self.predictive_engine.list_forecasts(
                filters=filters,
                offset=offset,
                limit=limit
            )
            
            return [self._serialize_forecast(f) for f in forecasts]
        except Exception as e:
            logger.error(f"Error resolving forecasts: {str(e)}")
            raise
    
    async def resolve_scenario_analysis(
        self, 
        info, 
        forecast_id: str, 
        scenarios: List[Dict]
    ) -> List[Dict]:
        """Resolve scenario analysis for a forecast"""
        try:
            results = await self.predictive_engine.analyze_scenarios(
                forecast_id, scenarios
            )
            return [self._serialize_scenario(s) for s in results]
        except Exception as e:
            logger.error(f"Error resolving scenario analysis: {str(e)}")
            raise
    
    async def resolve_data_source(self, info, id: str) -> Optional[Dict]:
        """Resolve single data source by ID"""
        try:
            source = await self.data_connector.get_data_source(id)
            if source:
                return self._serialize_data_source(source)
            return None
        except Exception as e:
            logger.error(f"Error resolving data source {id}: {str(e)}")
            raise
    
    async def resolve_data_sources(
        self,
        info,
        type: Optional[str] = None,
        status: Optional[str] = None,
        pagination: Optional[Dict] = None
    ) -> List[Dict]:
        """Resolve multiple data sources"""
        try:
            filters = {}
            if type:
                filters['type'] = type
            if status:
                filters['status'] = status
            
            # Apply pagination
            page = pagination.get('page', 1) if pagination else 1
            limit = pagination.get('limit', 20) if pagination else 20
            offset = (page - 1) * limit
            
            sources = await self.data_connector.list_data_sources(
                filters=filters,
                offset=offset,
                limit=limit
            )
            
            return [self._serialize_data_source(s) for s in sources]
        except Exception as e:
            logger.error(f"Error resolving data sources: {str(e)}")
            raise
    
    async def resolve_data_quality(
        self, 
        info, 
        source_id: str, 
        time_range: Optional[str] = None
    ) -> Dict:
        """Resolve data quality metrics for a source"""
        try:
            start_date, end_date = self._parse_time_range(time_range)
            quality = await self.data_connector.get_data_quality(
                source_id, start_date, end_date
            )
            return quality
        except Exception as e:
            logger.error(f"Error resolving data quality: {str(e)}")
            raise
    
    async def resolve_dashboard_template(self, info, id: str) -> Optional[Dict]:
        """Resolve single dashboard template by ID"""
        try:
            template = await self.template_engine.get_template(id)
            if template:
                return self._serialize_dashboard_template(template)
            return None
        except Exception as e:
            logger.error(f"Error resolving dashboard template {id}: {str(e)}")
            raise
    
    async def resolve_dashboard_templates(
        self,
        info,
        category: Optional[str] = None,
        industry: Optional[str] = None,
        pagination: Optional[Dict] = None,
        sort: Optional[Dict] = None
    ) -> List[Dict]:
        """Resolve multiple dashboard templates"""
        try:
            filters = {}
            if category:
                filters['category'] = category
            if industry:
                filters['industry'] = industry
            
            # Apply pagination
            page = pagination.get('page', 1) if pagination else 1
            limit = pagination.get('limit', 20) if pagination else 20
            offset = (page - 1) * limit
            
            # Apply sorting
            sort_field = sort.get('field', 'popularity') if sort else 'popularity'
            sort_direction = sort.get('direction', 'desc') if sort else 'desc'
            
            templates = await self.template_engine.list_templates(
                filters=filters,
                offset=offset,
                limit=limit,
                sort_field=sort_field,
                sort_direction=sort_direction
            )
            
            return [self._serialize_dashboard_template(t) for t in templates]
        except Exception as e:
            logger.error(f"Error resolving dashboard templates: {str(e)}")
            raise
    
    async def resolve_analytics_report(self, info, id: str) -> Optional[Dict]:
        """Resolve single analytics report by ID"""
        try:
            report = await self.reporting_engine.get_report(id)
            if report:
                return self._serialize_analytics_report(report)
            return None
        except Exception as e:
            logger.error(f"Error resolving analytics report {id}: {str(e)}")
            raise
    
    async def resolve_analytics_reports(
        self,
        info,
        type: Optional[str] = None,
        date_range: Optional[str] = None,
        pagination: Optional[Dict] = None
    ) -> List[Dict]:
        """Resolve multiple analytics reports"""
        try:
            filters = {}
            if type:
                filters['type'] = type
            if date_range:
                start_date, end_date = self._parse_time_range(date_range)
                filters['date_range'] = (start_date, end_date)
            
            # Apply pagination
            page = pagination.get('page', 1) if pagination else 1
            limit = pagination.get('limit', 20) if pagination else 20
            offset = (page - 1) * limit
            
            reports = await self.reporting_engine.list_reports(
                filters=filters,
                offset=offset,
                limit=limit
            )
            
            return [self._serialize_analytics_report(r) for r in reports]
        except Exception as e:
            logger.error(f"Error resolving analytics reports: {str(e)}")
            raise
    
    async def resolve_search(
        self, 
        info, 
        query: str, 
        types: Optional[List[str]] = None, 
        limit: Optional[int] = None
    ) -> Dict:
        """Resolve search across all entities"""
        try:
            search_limit = limit or 50
            search_types = types or ['dashboard', 'insight', 'forecast', 'template']
            
            results = {}
            
            if 'dashboard' in search_types:
                dashboards = await self.dashboard_manager.search_dashboards(
                    query, limit=search_limit
                )
                results['dashboards'] = [self._serialize_dashboard(d) for d in dashboards]
            
            if 'insight' in search_types:
                insights = await self.insight_generator.search_insights(
                    query, limit=search_limit
                )
                results['insights'] = [self._serialize_insight(i) for i in insights]
            
            if 'forecast' in search_types:
                forecasts = await self.predictive_engine.search_forecasts(
                    query, limit=search_limit
                )
                results['forecasts'] = [self._serialize_forecast(f) for f in forecasts]
            
            if 'template' in search_types:
                templates = await self.template_engine.search_templates(
                    query, limit=search_limit
                )
                results['templates'] = [self._serialize_dashboard_template(t) for t in templates]
            
            return results
        except Exception as e:
            logger.error(f"Error resolving search: {str(e)}")
            raise
    
    async def resolve_recommendations(
        self, 
        info, 
        user_id: str, 
        context: Optional[Dict] = None
    ) -> Dict:
        """Resolve personalized recommendations"""
        try:
            # Get recommendations from various engines
            dashboard_recs = await self.dashboard_manager.get_recommendations(
                user_id, context
            )
            template_recs = await self.template_engine.get_recommendations(
                user_id, context
            )
            insight_recs = await self.insight_generator.get_recommendations(
                user_id, context
            )
            
            return {
                'dashboards': [self._serialize_dashboard(d) for d in dashboard_recs],
                'templates': [self._serialize_dashboard_template(t) for t in template_recs],
                'insights': [self._serialize_insight(i) for i in insight_recs]
            }
        except Exception as e:
            logger.error(f"Error resolving recommendations: {str(e)}")
            raise
    
    # Mutation Resolvers
    
    async def resolve_create_dashboard(self, info, input: Dict) -> Dict:
        """Create a new dashboard"""
        try:
            dashboard = await self.dashboard_manager.create_dashboard(input)
            return self._serialize_dashboard(dashboard)
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise
    
    async def resolve_update_dashboard(self, info, id: str, input: Dict) -> Dict:
        """Update an existing dashboard"""
        try:
            dashboard = await self.dashboard_manager.update_dashboard(id, input)
            return self._serialize_dashboard(dashboard)
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
            raise
    
    async def resolve_delete_dashboard(self, info, id: str) -> bool:
        """Delete a dashboard"""
        try:
            return await self.dashboard_manager.delete_dashboard(id)
        except Exception as e:
            logger.error(f"Error deleting dashboard: {str(e)}")
            raise
    
    # Helper Methods
    
    def _parse_time_range(self, time_range: Optional[str]) -> tuple:
        """Parse time range string into start and end dates"""
        if not time_range:
            return None, None
        
        now = datetime.utcnow()
        
        if time_range == "1d":
            return now - timedelta(days=1), now
        elif time_range == "7d":
            return now - timedelta(days=7), now
        elif time_range == "30d":
            return now - timedelta(days=30), now
        elif time_range == "90d":
            return now - timedelta(days=90), now
        elif time_range == "1y":
            return now - timedelta(days=365), now
        else:
            # Try to parse as ISO format range
            try:
                parts = time_range.split("/")
                if len(parts) == 2:
                    start = datetime.fromisoformat(parts[0])
                    end = datetime.fromisoformat(parts[1])
                    return start, end
            except:
                pass
        
        return None, None
    
    def _serialize_dashboard(self, dashboard) -> Dict:
        """Serialize dashboard object to GraphQL format"""
        return {
            "id": dashboard.id,
            "name": dashboard.name,
            "type": dashboard.type.value if hasattr(dashboard.type, 'value') else dashboard.type,
            "owner": dashboard.owner_id,
            "config": dashboard.config,
            "widgets": [self._serialize_widget(w) for w in dashboard.widgets],
            "permissions": [self._serialize_permission(p) for p in dashboard.permissions],
            "createdAt": dashboard.created_at.isoformat(),
            "updatedAt": dashboard.updated_at.isoformat(),
            "isActive": getattr(dashboard, 'is_active', True)
        }
    
    def _serialize_widget(self, widget) -> Dict:
        """Serialize widget object to GraphQL format"""
        return {
            "id": widget.id,
            "type": widget.type.value if hasattr(widget.type, 'value') else widget.type,
            "title": widget.title,
            "config": widget.config,
            "position": {
                "x": widget.position.get('x', 0),
                "y": widget.position.get('y', 0),
                "width": widget.position.get('width', 1),
                "height": widget.position.get('height', 1)
            },
            "data": getattr(widget, 'data', None),
            "lastUpdated": getattr(widget, 'last_updated', datetime.utcnow()).isoformat()
        }
    
    def _serialize_permission(self, permission) -> Dict:
        """Serialize permission object to GraphQL format"""
        return {
            "userId": permission.user_id,
            "role": permission.role.value if hasattr(permission.role, 'value') else permission.role,
            "grantedAt": permission.granted_at.isoformat(),
            "grantedBy": permission.granted_by
        }
    
    def _serialize_dashboard_metrics(self, metrics) -> Dict:
        """Serialize dashboard metrics to GraphQL format"""
        return {
            "totalViews": metrics.total_views,
            "uniqueUsers": metrics.unique_users,
            "avgSessionDuration": metrics.avg_session_duration,
            "lastAccessed": metrics.last_accessed.isoformat() if metrics.last_accessed else None,
            "popularWidgets": metrics.popular_widgets
        }
    
    def _serialize_roi_analysis(self, analysis) -> Dict:
        """Serialize ROI analysis to GraphQL format"""
        return {
            "id": analysis.id,
            "projectId": analysis.project_id,
            "projectName": analysis.project_name,
            "totalInvestment": analysis.total_investment,
            "totalBenefits": analysis.total_benefits,
            "roiPercentage": analysis.roi_percentage,
            "paybackPeriod": analysis.payback_period,
            "npv": analysis.npv,
            "irr": analysis.irr,
            "analysisDate": analysis.analysis_date.isoformat(),
            "breakdown": self._serialize_roi_breakdown(analysis.breakdown),
            "trends": [self._serialize_roi_trend(t) for t in getattr(analysis, 'trends', [])]
        }
    
    def _serialize_roi_breakdown(self, breakdown) -> Dict:
        """Serialize ROI breakdown to GraphQL format"""
        return {
            "directCosts": breakdown.direct_costs,
            "indirectCosts": breakdown.indirect_costs,
            "operationalSavings": breakdown.operational_savings,
            "productivityGains": breakdown.productivity_gains,
            "revenueIncrease": breakdown.revenue_increase,
            "riskMitigation": breakdown.risk_mitigation
        }
    
    def _serialize_roi_trend(self, trend) -> Dict:
        """Serialize ROI trend to GraphQL format"""
        return {
            "period": trend.period,
            "investment": trend.investment,
            "benefits": trend.benefits,
            "cumulativeROI": trend.cumulative_roi
        }
    
    def _serialize_insight(self, insight) -> Dict:
        """Serialize insight to GraphQL format"""
        return {
            "id": insight.id,
            "type": insight.type.value if hasattr(insight.type, 'value') else insight.type,
            "title": insight.title,
            "description": insight.description,
            "significance": insight.significance,
            "confidence": insight.confidence,
            "recommendations": insight.recommendations,
            "createdAt": insight.created_at.isoformat(),
            "dataPoints": [self._serialize_data_point(dp) for dp in getattr(insight, 'data_points', [])],
            "visualizations": [self._serialize_visualization(v) for v in getattr(insight, 'visualizations', [])],
            "actionItems": [self._serialize_action_item(ai) for ai in getattr(insight, 'action_items', [])],
            "businessImpact": self._serialize_business_impact(getattr(insight, 'business_impact', {}))
        }
    
    def _serialize_data_point(self, data_point) -> Dict:
        """Serialize data point to GraphQL format"""
        return {
            "metric": data_point.metric,
            "value": data_point.value,
            "unit": data_point.unit,
            "timestamp": data_point.timestamp.isoformat(),
            "source": data_point.source,
            "context": getattr(data_point, 'context', {})
        }
    
    def _serialize_visualization(self, visualization) -> Dict:
        """Serialize visualization to GraphQL format"""
        return {
            "type": visualization.type.value if hasattr(visualization.type, 'value') else visualization.type,
            "title": visualization.title,
            "config": visualization.config,
            "data": visualization.data
        }
    
    def _serialize_action_item(self, action_item) -> Dict:
        """Serialize action item to GraphQL format"""
        return {
            "id": action_item.id,
            "title": action_item.title,
            "description": action_item.description,
            "priority": action_item.priority.value if hasattr(action_item.priority, 'value') else action_item.priority,
            "estimatedImpact": action_item.estimated_impact,
            "estimatedEffort": action_item.estimated_effort,
            "assignee": getattr(action_item, 'assignee', None),
            "dueDate": action_item.due_date.isoformat() if getattr(action_item, 'due_date') else None,
            "status": action_item.status.value if hasattr(action_item.status, 'value') else action_item.status
        }
    
    def _serialize_business_impact(self, business_impact) -> Dict:
        """Serialize business impact to GraphQL format"""
        if not business_impact:
            return {
                "category": "EFFICIENCY",
                "magnitude": 0.0,
                "timeframe": "unknown",
                "affectedMetrics": [],
                "riskLevel": "LOW"
            }
        
        return {
            "category": business_impact.get('category', 'EFFICIENCY'),
            "magnitude": business_impact.get('magnitude', 0.0),
            "timeframe": business_impact.get('timeframe', 'unknown'),
            "affectedMetrics": business_impact.get('affected_metrics', []),
            "riskLevel": business_impact.get('risk_level', 'LOW')
        }
    
    def _serialize_forecast(self, forecast) -> Dict:
        """Serialize forecast to GraphQL format"""
        return {
            "id": forecast.id,
            "metric": forecast.metric,
            "horizon": forecast.horizon,
            "predictions": [self._serialize_prediction(p) for p in forecast.predictions],
            "confidence": forecast.confidence,
            "model": forecast.model,
            "accuracy": getattr(forecast, 'accuracy', 0.0),
            "generatedAt": forecast.generated_at.isoformat(),
            "scenarios": [self._serialize_scenario(s) for s in getattr(forecast, 'scenarios', [])]
        }
    
    def _serialize_prediction(self, prediction) -> Dict:
        """Serialize prediction to GraphQL format"""
        return {
            "timestamp": prediction.timestamp.isoformat(),
            "value": prediction.value,
            "lowerBound": prediction.lower_bound,
            "upperBound": prediction.upper_bound,
            "confidence": prediction.confidence
        }
    
    def _serialize_scenario(self, scenario) -> Dict:
        """Serialize scenario to GraphQL format"""
        return {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "assumptions": scenario.assumptions,
            "predictions": [self._serialize_prediction(p) for p in scenario.predictions],
            "impact": self._serialize_scenario_impact(scenario.impact)
        }
    
    def _serialize_scenario_impact(self, impact) -> Dict:
        """Serialize scenario impact to GraphQL format"""
        return {
            "revenueChange": impact.revenue_change,
            "costChange": impact.cost_change,
            "riskChange": impact.risk_change,
            "timeToRealization": impact.time_to_realization
        }
    
    def _serialize_data_source(self, source) -> Dict:
        """Serialize data source to GraphQL format"""
        return {
            "id": source.id,
            "name": source.name,
            "type": source.type.value if hasattr(source.type, 'value') else source.type,
            "status": source.status.value if hasattr(source.status, 'value') else source.status,
            "config": source.config,
            "lastSync": source.last_sync.isoformat() if source.last_sync else None,
            "metrics": self._serialize_data_source_metrics(source.metrics),
            "schema": self._serialize_data_schema(source.schema)
        }
    
    def _serialize_data_source_metrics(self, metrics) -> Dict:
        """Serialize data source metrics to GraphQL format"""
        return {
            "recordCount": metrics.record_count,
            "syncFrequency": metrics.sync_frequency,
            "errorRate": metrics.error_rate,
            "avgSyncTime": metrics.avg_sync_time,
            "dataQuality": metrics.data_quality
        }
    
    def _serialize_data_schema(self, schema) -> Dict:
        """Serialize data schema to GraphQL format"""
        return {
            "tables": [self._serialize_table_schema(t) for t in schema.tables],
            "relationships": [self._serialize_relationship(r) for r in schema.relationships],
            "lastUpdated": schema.last_updated.isoformat()
        }
    
    def _serialize_table_schema(self, table) -> Dict:
        """Serialize table schema to GraphQL format"""
        return {
            "name": table.name,
            "columns": [self._serialize_column_schema(c) for c in table.columns],
            "recordCount": table.record_count
        }
    
    def _serialize_column_schema(self, column) -> Dict:
        """Serialize column schema to GraphQL format"""
        return {
            "name": column.name,
            "type": column.type,
            "nullable": column.nullable,
            "unique": column.unique,
            "description": getattr(column, 'description', None)
        }
    
    def _serialize_relationship(self, relationship) -> Dict:
        """Serialize relationship to GraphQL format"""
        return {
            "fromTable": relationship.from_table,
            "toTable": relationship.to_table,
            "type": relationship.type.value if hasattr(relationship.type, 'value') else relationship.type,
            "columns": relationship.columns
        }
    
    def _serialize_dashboard_template(self, template) -> Dict:
        """Serialize dashboard template to GraphQL format"""
        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "category": template.category.value if hasattr(template.category, 'value') else template.category,
            "industry": getattr(template, 'industry', None),
            "config": template.config,
            "widgets": [self._serialize_template_widget(w) for w in template.widgets],
            "popularity": getattr(template, 'popularity', 0),
            "rating": getattr(template, 'rating', 0.0),
            "createdBy": template.created_by,
            "createdAt": template.created_at.isoformat(),
            "isPublic": getattr(template, 'is_public', True)
        }
    
    def _serialize_template_widget(self, widget) -> Dict:
        """Serialize template widget to GraphQL format"""
        return {
            "type": widget.type.value if hasattr(widget.type, 'value') else widget.type,
            "title": widget.title,
            "config": widget.config,
            "position": {
                "x": widget.position.get('x', 0),
                "y": widget.position.get('y', 0),
                "width": widget.position.get('width', 1),
                "height": widget.position.get('height', 1)
            },
            "dataSources": getattr(widget, 'data_sources', [])
        }
    
    def _serialize_analytics_report(self, report) -> Dict:
        """Serialize analytics report to GraphQL format"""
        return {
            "id": report.id,
            "title": report.title,
            "type": report.type.value if hasattr(report.type, 'value') else report.type,
            "format": report.format.value if hasattr(report.format, 'value') else report.format,
            "generatedAt": report.generated_at.isoformat(),
            "fileSize": report.file_size,
            "downloadUrl": getattr(report, 'download_url', ''),
            "metadata": report.metadata,
            "schedule": self._serialize_report_schedule(getattr(report, 'schedule', None))
        }
    
    def _serialize_report_schedule(self, schedule) -> Optional[Dict]:
        """Serialize report schedule to GraphQL format"""
        if not schedule:
            return None
        
        return {
            "id": schedule.id,
            "frequency": schedule.frequency.value if hasattr(schedule.frequency, 'value') else schedule.frequency,
            "nextRun": schedule.next_run.isoformat() if schedule.next_run else None,
            "lastRun": schedule.last_run.isoformat() if schedule.last_run else None,
            "recipients": schedule.recipients,
            "isActive": schedule.is_active
        }