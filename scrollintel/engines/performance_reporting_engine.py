"""
Performance Reporting Framework

This engine creates executive-level performance reporting and analysis,
builds performance insight generation and communication, and implements
performance reporting optimization and customization.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    MARKET = "market"
    TECHNOLOGY = "technology"
    RISK = "risk"

class ReportingPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class TrendDirection(Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"

class PerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    current_value: float
    target_value: float
    previous_value: Optional[float]
    unit: str
    data_source: str
    calculation_method: str
    owner: str
    last_updated: datetime
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceTrend:
    metric_id: str
    trend_direction: TrendDirection
    trend_strength: float  # 0-1, how strong the trend is
    trend_duration: int  # days
    variance: float
    confidence_level: float
    historical_data: List[Tuple[datetime, float]]
    forecast_values: List[Tuple[datetime, float]]

@dataclass
class PerformanceInsight:
    insight_id: str
    metric_id: str
    insight_type: str  # anomaly, trend, correlation, forecast
    title: str
    description: str
    significance_score: float  # 0-1
    actionable: bool
    recommended_actions: List[str]
    impact_assessment: str
    confidence_level: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceReport:
    report_id: str
    title: str
    reporting_period: ReportingPeriod
    period_start: datetime
    period_end: datetime
    audience: str  # board, executive, department
    metrics: List[PerformanceMetric]
    insights: List[PerformanceInsight]
    executive_summary: str
    key_findings: List[str]
    recommendations: List[str]
    customizations: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    generated_by: str = "ScrollIntel Performance Engine"c
lass PerformanceReportingEngine:
    """Engine for executive-level performance reporting and analysis"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.trends: List[PerformanceTrend] = []
        self.insights: List[PerformanceInsight] = []
        self.reports: List[PerformanceReport] = []
        self.report_templates = self._initialize_report_templates()
        self.insight_generators = self._initialize_insight_generators()
    
    def _initialize_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize report templates for different audiences"""
        return {
            'board': {
                'focus_areas': ['strategic', 'financial', 'risk'],
                'detail_level': 'high_level',
                'visualization_style': 'executive_dashboard',
                'key_sections': ['executive_summary', 'strategic_metrics', 'financial_performance', 'risk_indicators'],
                'max_metrics': 12,
                'insight_types': ['strategic_trends', 'performance_alerts', 'forecasts']
            },
            'executive': {
                'focus_areas': ['operational', 'financial', 'strategic', 'customer'],
                'detail_level': 'detailed',
                'visualization_style': 'comprehensive_charts',
                'key_sections': ['performance_overview', 'operational_metrics', 'financial_analysis', 'customer_insights'],
                'max_metrics': 25,
                'insight_types': ['operational_trends', 'anomaly_detection', 'correlation_analysis']
            },
            'department': {
                'focus_areas': ['operational', 'employee', 'technology'],
                'detail_level': 'granular',
                'visualization_style': 'detailed_analytics',
                'key_sections': ['departmental_kpis', 'team_performance', 'process_metrics', 'improvement_areas'],
                'max_metrics': 50,
                'insight_types': ['process_optimization', 'team_analytics', 'efficiency_insights']
            }
        }
    
    def _initialize_insight_generators(self) -> Dict[str, callable]:
        """Initialize insight generation functions"""
        return {
            'anomaly_detection': self._detect_anomalies,
            'trend_analysis': self._analyze_trends,
            'correlation_analysis': self._analyze_correlations,
            'forecast_generation': self._generate_forecasts,
            'performance_alerts': self._generate_performance_alerts,
            'comparative_analysis': self._perform_comparative_analysis
        }
    
    def add_performance_metric(self, metric: PerformanceMetric) -> None:
        """Add a performance metric to the system"""
        # Check if metric already exists and update or add new
        existing_metric = next((m for m in self.metrics if m.metric_id == metric.metric_id), None)
        if existing_metric:
            existing_metric.current_value = metric.current_value
            existing_metric.previous_value = existing_metric.current_value
            existing_metric.last_updated = datetime.now()
        else:
            self.metrics.append(metric)
        
        logger.info(f"Added/updated performance metric: {metric.name}")
    
    def calculate_performance_level(self, metric: PerformanceMetric) -> PerformanceLevel:
        """Calculate performance level based on current vs target value"""
        if metric.target_value == 0:
            return PerformanceLevel.SATISFACTORY
        
        performance_ratio = metric.current_value / metric.target_value
        
        if performance_ratio >= 1.1:
            return PerformanceLevel.EXCELLENT
        elif performance_ratio >= 1.0:
            return PerformanceLevel.GOOD
        elif performance_ratio >= 0.9:
            return PerformanceLevel.SATISFACTORY
        elif performance_ratio >= 0.8:
            return PerformanceLevel.NEEDS_IMPROVEMENT
        else:
            return PerformanceLevel.CRITICAL
    
    def analyze_metric_trend(self, metric_id: str, historical_data: List[Tuple[datetime, float]]) -> PerformanceTrend:
        """Analyze trend for a specific metric"""
        if len(historical_data) < 3:
            return PerformanceTrend(
                metric_id=metric_id,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                trend_duration=0,
                variance=0.0,
                confidence_level=0.0,
                historical_data=historical_data,
                forecast_values=[]
            )
        
        # Extract values and calculate trend
        values = [point[1] for point in historical_data]
        dates = [point[0] for point in historical_data]
        
        # Calculate linear regression for trend
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction and strength
        if abs(slope) < 0.01:
            trend_direction = TrendDirection.STABLE
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = TrendDirection.IMPROVING
            trend_strength = min(abs(slope) / (max(values) - min(values)) * 10, 1.0)
        else:
            trend_direction = TrendDirection.DECLINING
            trend_strength = min(abs(slope) / (max(values) - min(values)) * 10, 1.0)
        
        # Calculate variance and confidence
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        confidence_level = max(0.5, 1.0 - (variance / (y_mean ** 2)) if y_mean != 0 else 0.5)
        
        # Generate simple forecast (next 3 periods)
        forecast_values = []
        last_date = dates[-1]
        last_value = values[-1]
        
        for i in range(1, 4):
            forecast_date = last_date + timedelta(days=30 * i)  # Assume monthly periods
            forecast_value = last_value + (slope * i)
            forecast_values.append((forecast_date, forecast_value))
        
        trend = PerformanceTrend(
            metric_id=metric_id,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            trend_duration=len(historical_data),
            variance=variance,
            confidence_level=confidence_level,
            historical_data=historical_data,
            forecast_values=forecast_values
        )
        
        # Update or add trend
        existing_trend = next((t for t in self.trends if t.metric_id == metric_id), None)
        if existing_trend:
            self.trends.remove(existing_trend)
        self.trends.append(trend)
        
        return trend
    
    def generate_performance_insights(self, metric_ids: Optional[List[str]] = None) -> List[PerformanceInsight]:
        """Generate performance insights for specified metrics or all metrics"""
        target_metrics = self.metrics if not metric_ids else [m for m in self.metrics if m.metric_id in metric_ids]
        new_insights = []
        
        for metric in target_metrics:
            # Generate different types of insights
            for insight_type, generator in self.insight_generators.items():
                try:
                    insights = generator(metric)
                    new_insights.extend(insights)
                except Exception as e:
                    logger.warning(f"Failed to generate {insight_type} for {metric.name}: {str(e)}")
        
        # Add new insights to collection
        self.insights.extend(new_insights)
        
        # Sort by significance score
        new_insights.sort(key=lambda x: x.significance_score, reverse=True)
        
        logger.info(f"Generated {len(new_insights)} performance insights")
        return new_insights
    
    def _detect_anomalies(self, metric: PerformanceMetric) -> List[PerformanceInsight]:
        """Detect anomalies in metric performance"""
        insights = []
        
        # Check for significant deviation from target
        if metric.target_value > 0:
            deviation = abs(metric.current_value - metric.target_value) / metric.target_value
            
            if deviation > 0.2:  # 20% deviation threshold
                insight = PerformanceInsight(
                    insight_id=f"anomaly_{metric.metric_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metric_id=metric.metric_id,
                    insight_type="anomaly",
                    title=f"Significant Deviation in {metric.name}",
                    description=f"{metric.name} is {deviation:.1%} away from target value",
                    significance_score=min(deviation, 1.0),
                    actionable=True,
                    recommended_actions=[
                        "Investigate root cause of deviation",
                        "Review data collection process",
                        "Consider target adjustment if justified"
                    ],
                    impact_assessment="High" if deviation > 0.5 else "Medium",
                    confidence_level=0.8
                )
                insights.append(insight)
        
        # Check for sudden change from previous value
        if metric.previous_value is not None and metric.previous_value > 0:
            change_rate = abs(metric.current_value - metric.previous_value) / metric.previous_value
            
            if change_rate > 0.3:  # 30% change threshold
                insight = PerformanceInsight(
                    insight_id=f"change_{metric.metric_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metric_id=metric.metric_id,
                    insight_type="anomaly",
                    title=f"Sudden Change in {metric.name}",
                    description=f"{metric.name} changed by {change_rate:.1%} from previous period",
                    significance_score=min(change_rate, 1.0),
                    actionable=True,
                    recommended_actions=[
                        "Analyze factors contributing to change",
                        "Validate data accuracy",
                        "Assess impact on related metrics"
                    ],
                    impact_assessment="High" if change_rate > 0.5 else "Medium",
                    confidence_level=0.7
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_trends(self, metric: PerformanceMetric) -> List[PerformanceInsight]:
        """Analyze trends for metric"""
        insights = []
        
        # Find trend for this metric
        trend = next((t for t in self.trends if t.metric_id == metric.metric_id), None)
        
        if trend and trend.trend_strength > 0.3:  # Significant trend threshold
            direction_text = "improving" if trend.trend_direction == TrendDirection.IMPROVING else "declining"
            
            insight = PerformanceInsight(
                insight_id=f"trend_{metric.metric_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metric_id=metric.metric_id,
                insight_type="trend",
                title=f"{metric.name} Shows {direction_text.title()} Trend",
                description=f"{metric.name} has been {direction_text} with {trend.trend_strength:.1%} strength over {trend.trend_duration} periods",
                significance_score=trend.trend_strength * trend.confidence_level,
                actionable=True,
                recommended_actions=self._get_trend_recommendations(trend.trend_direction, metric),
                impact_assessment=self._assess_trend_impact(trend, metric),
                confidence_level=trend.confidence_level
            )
            insights.append(insight)
        
        return insights
    
    def _get_trend_recommendations(self, trend_direction: TrendDirection, metric: PerformanceMetric) -> List[str]:
        """Get recommendations based on trend direction"""
        if trend_direction == TrendDirection.IMPROVING:
            return [
                "Continue current strategies that are driving improvement",
                "Consider scaling successful initiatives",
                "Document best practices for replication"
            ]
        elif trend_direction == TrendDirection.DECLINING:
            return [
                "Investigate root causes of decline",
                "Implement corrective action plan",
                "Increase monitoring frequency",
                "Consider resource reallocation"
            ]
        else:
            return [
                "Monitor for emerging patterns",
                "Consider process optimization opportunities"
            ]
    
    def _assess_trend_impact(self, trend: PerformanceTrend, metric: PerformanceMetric) -> str:
        """Assess the impact of a trend"""
        if metric.metric_type in [MetricType.FINANCIAL, MetricType.STRATEGIC]:
            if trend.trend_strength > 0.7:
                return "Critical"
            elif trend.trend_strength > 0.5:
                return "High"
            else:
                return "Medium"
        else:
            if trend.trend_strength > 0.8:
                return "High"
            elif trend.trend_strength > 0.5:
                return "Medium"
            else:
                return "Low"
    
    def _analyze_correlations(self, metric: PerformanceMetric) -> List[PerformanceInsight]:
        """Analyze correlations between metrics"""
        insights = []
        
        # Simple correlation analysis with other metrics of same type
        related_metrics = [m for m in self.metrics if m.metric_type == metric.metric_type and m.metric_id != metric.metric_id]
        
        for related_metric in related_metrics[:3]:  # Limit to top 3 related metrics
            # Simple correlation based on performance levels
            current_level = self.calculate_performance_level(metric)
            related_level = self.calculate_performance_level(related_metric)
            
            # If both metrics are performing similarly (both good or both poor)
            if (current_level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD] and 
                related_level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]) or \
               (current_level in [PerformanceLevel.NEEDS_IMPROVEMENT, PerformanceLevel.CRITICAL] and 
                related_level in [PerformanceLevel.NEEDS_IMPROVEMENT, PerformanceLevel.CRITICAL]):
                
                insight = PerformanceInsight(
                    insight_id=f"correlation_{metric.metric_id}_{related_metric.metric_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metric_id=metric.metric_id,
                    insight_type="correlation",
                    title=f"Performance Correlation: {metric.name} and {related_metric.name}",
                    description=f"{metric.name} and {related_metric.name} show similar performance patterns",
                    significance_score=0.6,
                    actionable=True,
                    recommended_actions=[
                        "Investigate common factors affecting both metrics",
                        "Consider integrated improvement strategies",
                        "Monitor both metrics together"
                    ],
                    impact_assessment="Medium",
                    confidence_level=0.6
                )
                insights.append(insight)
        
        return insights
    
    def _generate_forecasts(self, metric: PerformanceMetric) -> List[PerformanceInsight]:
        """Generate forecast insights"""
        insights = []
        
        # Find trend for forecasting
        trend = next((t for t in self.trends if t.metric_id == metric.metric_id), None)
        
        if trend and trend.forecast_values:
            next_period_forecast = trend.forecast_values[0][1]
            
            # Check if forecast indicates potential issues
            if metric.target_value > 0:
                forecast_vs_target = next_period_forecast / metric.target_value
                
                if forecast_vs_target < 0.8:  # Forecast below 80% of target
                    insight = PerformanceInsight(
                        insight_id=f"forecast_{metric.metric_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        metric_id=metric.metric_id,
                        insight_type="forecast",
                        title=f"Forecast Alert: {metric.name} May Miss Target",
                        description=f"Based on current trends, {metric.name} is forecasted to be {forecast_vs_target:.1%} of target next period",
                        significance_score=1.0 - forecast_vs_target,
                        actionable=True,
                        recommended_actions=[
                            "Implement proactive improvement measures",
                            "Adjust resource allocation",
                            "Review and update action plans"
                        ],
                        impact_assessment="High" if forecast_vs_target < 0.7 else "Medium",
                        confidence_level=trend.confidence_level
                    )
                    insights.append(insight)
        
        return insights
    
    def _generate_performance_alerts(self, metric: PerformanceMetric) -> List[PerformanceInsight]:
        """Generate performance alerts"""
        insights = []
        
        performance_level = self.calculate_performance_level(metric)
        
        if performance_level == PerformanceLevel.CRITICAL:
            insight = PerformanceInsight(
                insight_id=f"alert_{metric.metric_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metric_id=metric.metric_id,
                insight_type="alert",
                title=f"Critical Performance Alert: {metric.name}",
                description=f"{metric.name} is performing at critical level ({metric.current_value} vs target {metric.target_value})",
                significance_score=1.0,
                actionable=True,
                recommended_actions=[
                    "Immediate intervention required",
                    "Escalate to senior management",
                    "Implement emergency action plan"
                ],
                impact_assessment="Critical",
                confidence_level=0.9
            )
            insights.append(insight)
        
        return insights
    
    def _perform_comparative_analysis(self, metric: PerformanceMetric) -> List[PerformanceInsight]:
        """Perform comparative analysis"""
        insights = []
        
        # Compare with similar metrics or historical performance
        similar_metrics = [m for m in self.metrics if m.metric_type == metric.metric_type and m.metric_id != metric.metric_id]
        
        if similar_metrics:
            avg_performance = sum(m.current_value for m in similar_metrics) / len(similar_metrics)
            
            if metric.current_value > avg_performance * 1.2:  # 20% above average
                insight = PerformanceInsight(
                    insight_id=f"comparative_{metric.metric_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metric_id=metric.metric_id,
                    insight_type="comparative",
                    title=f"{metric.name} Outperforming Peer Metrics",
                    description=f"{metric.name} is {((metric.current_value / avg_performance) - 1):.1%} above average for {metric.metric_type.value} metrics",
                    significance_score=0.7,
                    actionable=True,
                    recommended_actions=[
                        "Document success factors",
                        "Share best practices with other areas",
                        "Consider scaling successful approaches"
                    ],
                    impact_assessment="Medium",
                    confidence_level=0.7
                )
                insights.append(insight)
        
        return insights 
   def create_performance_report(
        self,
        title: str,
        reporting_period: ReportingPeriod,
        period_start: datetime,
        period_end: datetime,
        audience: str,
        metric_ids: Optional[List[str]] = None,
        customizations: Optional[Dict[str, Any]] = None
    ) -> PerformanceReport:
        """Create comprehensive performance report"""
        
        # Get template for audience
        template = self.report_templates.get(audience, self.report_templates['executive'])
        
        # Select metrics based on template focus areas and limits
        selected_metrics = self._select_metrics_for_report(template, metric_ids)
        
        # Generate insights for selected metrics
        report_insights = self._generate_report_insights(selected_metrics, template)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(selected_metrics, report_insights, template)
        
        # Extract key findings
        key_findings = self._extract_key_findings(selected_metrics, report_insights)
        
        # Generate recommendations
        recommendations = self._generate_report_recommendations(selected_metrics, report_insights, template)
        
        # Apply customizations
        final_customizations = customizations or {}
        final_customizations.update(template)
        
        report = PerformanceReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            reporting_period=reporting_period,
            period_start=period_start,
            period_end=period_end,
            audience=audience,
            metrics=selected_metrics,
            insights=report_insights,
            executive_summary=executive_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            customizations=final_customizations
        )
        
        self.reports.append(report)
        logger.info(f"Created performance report: {title}")
        
        return report
    
    def _select_metrics_for_report(self, template: Dict[str, Any], metric_ids: Optional[List[str]]) -> List[PerformanceMetric]:
        """Select metrics for report based on template and constraints"""
        
        if metric_ids:
            # Use specified metrics
            selected = [m for m in self.metrics if m.metric_id in metric_ids]
        else:
            # Select based on template focus areas
            focus_types = [MetricType(area) for area in template['focus_areas'] if area in [t.value for t in MetricType]]
            selected = [m for m in self.metrics if m.metric_type in focus_types]
        
        # Limit to template max metrics
        max_metrics = template.get('max_metrics', 25)
        if len(selected) > max_metrics:
            # Prioritize by performance level and metric type importance
            selected.sort(key=lambda m: (
                self._get_metric_priority(m, template),
                self._get_performance_urgency(m)
            ), reverse=True)
            selected = selected[:max_metrics]
        
        return selected
    
    def _get_metric_priority(self, metric: PerformanceMetric, template: Dict[str, Any]) -> float:
        """Get metric priority based on template focus areas"""
        focus_areas = template['focus_areas']
        
        # Higher priority for metrics in focus areas
        if metric.metric_type.value in focus_areas:
            priority = 1.0
            # Extra priority for strategic and financial metrics
            if metric.metric_type in [MetricType.STRATEGIC, MetricType.FINANCIAL]:
                priority += 0.5
        else:
            priority = 0.5
        
        return priority
    
    def _get_performance_urgency(self, metric: PerformanceMetric) -> float:
        """Get performance urgency score"""
        performance_level = self.calculate_performance_level(metric)
        
        urgency_scores = {
            PerformanceLevel.CRITICAL: 1.0,
            PerformanceLevel.NEEDS_IMPROVEMENT: 0.8,
            PerformanceLevel.SATISFACTORY: 0.4,
            PerformanceLevel.GOOD: 0.2,
            PerformanceLevel.EXCELLENT: 0.1
        }
        
        return urgency_scores.get(performance_level, 0.5)
    
    def _generate_report_insights(self, metrics: List[PerformanceMetric], template: Dict[str, Any]) -> List[PerformanceInsight]:
        """Generate insights for report based on template requirements"""
        
        all_insights = []
        insight_types = template.get('insight_types', ['anomaly_detection', 'trend_analysis'])
        
        for metric in metrics:
            for insight_type in insight_types:
                if insight_type in self.insight_generators:
                    try:
                        insights = self.insight_generators[insight_type](metric)
                        all_insights.extend(insights)
                    except Exception as e:
                        logger.warning(f"Failed to generate {insight_type} for {metric.name}: {str(e)}")
        
        # Sort by significance and limit
        all_insights.sort(key=lambda x: x.significance_score, reverse=True)
        return all_insights[:20]  # Limit to top 20 insights
    
    def _create_executive_summary(
        self,
        metrics: List[PerformanceMetric],
        insights: List[PerformanceInsight],
        template: Dict[str, Any]
    ) -> str:
        """Create executive summary for the report"""
        
        # Calculate overall performance statistics
        total_metrics = len(metrics)
        performance_levels = [self.calculate_performance_level(m) for m in metrics]
        
        excellent_count = performance_levels.count(PerformanceLevel.EXCELLENT)
        good_count = performance_levels.count(PerformanceLevel.GOOD)
        satisfactory_count = performance_levels.count(PerformanceLevel.SATISFACTORY)
        needs_improvement_count = performance_levels.count(PerformanceLevel.NEEDS_IMPROVEMENT)
        critical_count = performance_levels.count(PerformanceLevel.CRITICAL)
        
        # Count insights by type
        critical_insights = len([i for i in insights if i.significance_score > 0.8])
        actionable_insights = len([i for i in insights if i.actionable])
        
        # Create summary based on audience
        if template.get('detail_level') == 'high_level':
            # Board-level summary
            summary = f"""
            Performance Overview: {total_metrics} key metrics reviewed for this reporting period.
            
            Performance Distribution:
            • {excellent_count + good_count} metrics performing at or above target ({((excellent_count + good_count) / total_metrics * 100):.0f}%)
            • {satisfactory_count} metrics meeting baseline expectations
            • {needs_improvement_count + critical_count} metrics requiring attention ({((needs_improvement_count + critical_count) / total_metrics * 100):.0f}%)
            
            Key Insights: {critical_insights} high-significance insights identified, with {actionable_insights} actionable recommendations.
            
            Strategic Focus: {"Strong performance across strategic metrics" if excellent_count > needs_improvement_count else "Strategic metrics require focused attention"}.
            """
        else:
            # Detailed summary
            summary = f"""
            Comprehensive Performance Analysis: {total_metrics} metrics analyzed across {len(set(m.metric_type for m in metrics))} performance categories.
            
            Performance Breakdown:
            • Excellent: {excellent_count} metrics ({(excellent_count / total_metrics * 100):.1f}%)
            • Good: {good_count} metrics ({(good_count / total_metrics * 100):.1f}%)
            • Satisfactory: {satisfactory_count} metrics ({(satisfactory_count / total_metrics * 100):.1f}%)
            • Needs Improvement: {needs_improvement_count} metrics ({(needs_improvement_count / total_metrics * 100):.1f}%)
            • Critical: {critical_count} metrics ({(critical_count / total_metrics * 100):.1f}%)
            
            Insight Analysis: {len(insights)} total insights generated, including {critical_insights} high-priority items requiring immediate attention.
            
            Trend Analysis: {"Positive trends identified in key performance areas" if excellent_count >= needs_improvement_count else "Mixed performance trends require strategic intervention"}.
            """
        
        return summary.strip()
    
    def _extract_key_findings(self, metrics: List[PerformanceMetric], insights: List[PerformanceInsight]) -> List[str]:
        """Extract key findings from metrics and insights"""
        findings = []
        
        # Top performing metrics
        excellent_metrics = [m for m in metrics if self.calculate_performance_level(m) == PerformanceLevel.EXCELLENT]
        if excellent_metrics:
            top_metric = max(excellent_metrics, key=lambda m: m.current_value / m.target_value if m.target_value > 0 else 0)
            findings.append(f"{top_metric.name} is significantly exceeding targets, demonstrating strong performance")
        
        # Critical issues
        critical_metrics = [m for m in metrics if self.calculate_performance_level(m) == PerformanceLevel.CRITICAL]
        if critical_metrics:
            findings.append(f"{len(critical_metrics)} metrics are performing at critical levels requiring immediate attention")
        
        # Trend findings
        declining_trends = [i for i in insights if i.insight_type == "trend" and "declining" in i.description.lower()]
        if declining_trends:
            findings.append(f"Declining performance trends identified in {len(declining_trends)} key areas")
        
        # Anomaly findings
        anomalies = [i for i in insights if i.insight_type == "anomaly" and i.significance_score > 0.7]
        if anomalies:
            findings.append(f"Significant performance anomalies detected requiring investigation")
        
        # Correlation findings
        correlations = [i for i in insights if i.insight_type == "correlation"]
        if correlations:
            findings.append(f"Performance correlations identified between related metrics")
        
        return findings[:5]  # Limit to top 5 findings
    
    def _generate_report_recommendations(
        self,
        metrics: List[PerformanceMetric],
        insights: List[PerformanceInsight],
        template: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on performance analysis"""
        recommendations = []
        
        # Critical metric recommendations
        critical_metrics = [m for m in metrics if self.calculate_performance_level(m) == PerformanceLevel.CRITICAL]
        if critical_metrics:
            recommendations.append(f"Immediate action required for {len(critical_metrics)} critical metrics - implement emergency improvement plans")
        
        # Trend-based recommendations
        declining_insights = [i for i in insights if i.insight_type == "trend" and "declining" in i.description.lower()]
        if declining_insights:
            recommendations.append("Address declining performance trends through root cause analysis and corrective action plans")
        
        # Opportunity recommendations
        excellent_metrics = [m for m in metrics if self.calculate_performance_level(m) == PerformanceLevel.EXCELLENT]
        if excellent_metrics:
            recommendations.append("Leverage high-performing areas to drive improvements in underperforming metrics")
        
        # Insight-based recommendations
        high_significance_insights = [i for i in insights if i.significance_score > 0.8 and i.actionable]
        if high_significance_insights:
            recommendations.append("Implement actionable insights from high-significance performance analysis")
        
        # Audience-specific recommendations
        if template.get('detail_level') == 'high_level':
            recommendations.append("Schedule detailed performance review sessions with operational teams")
            recommendations.append("Consider strategic resource reallocation based on performance patterns")
        else:
            recommendations.append("Increase monitoring frequency for underperforming metrics")
            recommendations.append("Implement process improvements in identified opportunity areas")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def optimize_report_for_audience(self, report_id: str, audience_feedback: Dict[str, Any]) -> PerformanceReport:
        """Optimize report based on audience feedback"""
        
        report = next((r for r in self.reports if r.report_id == report_id), None)
        if not report:
            raise ValueError(f"Report {report_id} not found")
        
        # Apply optimizations based on feedback
        optimizations = {}
        
        # Adjust detail level
        if 'detail_preference' in audience_feedback:
            if audience_feedback['detail_preference'] == 'more_detail':
                optimizations['max_metrics'] = min(report.customizations.get('max_metrics', 25) + 10, 50)
                optimizations['insight_types'] = report.customizations.get('insight_types', []) + ['correlation_analysis']
            elif audience_feedback['detail_preference'] == 'less_detail':
                optimizations['max_metrics'] = max(report.customizations.get('max_metrics', 25) - 5, 10)
        
        # Adjust focus areas
        if 'focus_preferences' in audience_feedback:
            optimizations['focus_areas'] = audience_feedback['focus_preferences']
        
        # Adjust visualization style
        if 'visualization_preference' in audience_feedback:
            optimizations['visualization_style'] = audience_feedback['visualization_preference']
        
        # Update customizations
        report.customizations.update(optimizations)
        
        # Regenerate report with new customizations
        optimized_report = self.create_performance_report(
            title=f"{report.title} (Optimized)",
            reporting_period=report.reporting_period,
            period_start=report.period_start,
            period_end=report.period_end,
            audience=report.audience,
            customizations=report.customizations
        )
        
        logger.info(f"Optimized report {report_id} based on audience feedback")
        return optimized_report
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        
        if not self.metrics:
            return {
                'total_metrics': 0,
                'performance_distribution': {},
                'trend_analysis': {},
                'insight_summary': {},
                'recommendations': []
            }
        
        # Performance distribution
        performance_levels = [self.calculate_performance_level(m) for m in self.metrics]
        performance_distribution = {
            level.value: performance_levels.count(level)
            for level in PerformanceLevel
        }
        
        # Trend analysis
        trend_analysis = {
            'improving_trends': len([t for t in self.trends if t.trend_direction == TrendDirection.IMPROVING]),
            'declining_trends': len([t for t in self.trends if t.trend_direction == TrendDirection.DECLINING]),
            'stable_trends': len([t for t in self.trends if t.trend_direction == TrendDirection.STABLE]),
            'volatile_trends': len([t for t in self.trends if t.trend_direction == TrendDirection.VOLATILE])
        }
        
        # Insight summary
        insight_summary = {
            'total_insights': len(self.insights),
            'high_significance': len([i for i in self.insights if i.significance_score > 0.8]),
            'actionable_insights': len([i for i in self.insights if i.actionable]),
            'by_type': {}
        }
        
        # Group insights by type
        for insight in self.insights:
            insight_type = insight.insight_type
            if insight_type not in insight_summary['by_type']:
                insight_summary['by_type'][insight_type] = 0
            insight_summary['by_type'][insight_type] += 1
        
        # Generate system-level recommendations
        system_recommendations = self._generate_system_recommendations(
            performance_distribution, trend_analysis, insight_summary
        )
        
        return {
            'total_metrics': len(self.metrics),
            'performance_distribution': performance_distribution,
            'trend_analysis': trend_analysis,
            'insight_summary': insight_summary,
            'metric_types': {mt.value: len([m for m in self.metrics if m.metric_type == mt]) for mt in MetricType},
            'recent_reports': len([r for r in self.reports if r.created_at > datetime.now() - timedelta(days=30)]),
            'recommendations': system_recommendations
        }
    
    def _generate_system_recommendations(
        self,
        performance_dist: Dict[str, int],
        trend_analysis: Dict[str, int],
        insight_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        critical_count = performance_dist.get('critical', 0)
        needs_improvement_count = performance_dist.get('needs_improvement', 0)
        
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical performance metrics immediately")
        
        if needs_improvement_count > critical_count:
            recommendations.append("Focus on systematic improvement of underperforming metrics")
        
        # Trend-based recommendations
        declining_trends = trend_analysis.get('declining_trends', 0)
        improving_trends = trend_analysis.get('improving_trends', 0)
        
        if declining_trends > improving_trends:
            recommendations.append("Implement trend reversal strategies for declining performance areas")
        
        # Insight-based recommendations
        actionable_insights = insight_summary.get('actionable_insights', 0)
        total_insights = insight_summary.get('total_insights', 0)
        
        if actionable_insights > 0:
            recommendations.append(f"Act on {actionable_insights} actionable performance insights")
        
        if total_insights < len(self.metrics) * 0.5:
            recommendations.append("Increase insight generation frequency for better performance understanding")
        
        return recommendations
    
    def export_report_data(self, report_id: str, format_type: str = 'json') -> Dict[str, Any]:
        """Export report data in specified format"""
        
        report = next((r for r in self.reports if r.report_id == report_id), None)
        if not report:
            raise ValueError(f"Report {report_id} not found")
        
        export_data = {
            'report_metadata': {
                'id': report.report_id,
                'title': report.title,
                'period': report.reporting_period.value,
                'period_start': report.period_start.isoformat(),
                'period_end': report.period_end.isoformat(),
                'audience': report.audience,
                'generated_at': report.created_at.isoformat(),
                'generated_by': report.generated_by
            },
            'executive_summary': report.executive_summary,
            'key_findings': report.key_findings,
            'recommendations': report.recommendations,
            'metrics': [
                {
                    'id': m.metric_id,
                    'name': m.name,
                    'type': m.metric_type.value,
                    'current_value': m.current_value,
                    'target_value': m.target_value,
                    'performance_level': self.calculate_performance_level(m).value,
                    'unit': m.unit,
                    'owner': m.owner
                }
                for m in report.metrics
            ],
            'insights': [
                {
                    'id': i.insight_id,
                    'type': i.insight_type,
                    'title': i.title,
                    'description': i.description,
                    'significance_score': i.significance_score,
                    'actionable': i.actionable,
                    'recommended_actions': i.recommended_actions,
                    'impact_assessment': i.impact_assessment,
                    'confidence_level': i.confidence_level
                }
                for i in report.insights
            ],
            'customizations': report.customizations
        }
        
        if format_type == 'json':
            return export_data
        else:
            # Could extend to support other formats (CSV, PDF, etc.)
            return export_data