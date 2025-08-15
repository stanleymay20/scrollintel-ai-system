"""
Executive Data Visualization Engine

This engine creates executive-level data presentations and insight communication
optimized for board consumption with impact measurement capabilities.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json

from ..models.board_presentation_models import ContentSection


class VisualizationType(Enum):
    """Types of executive visualizations"""
    EXECUTIVE_DASHBOARD = "executive_dashboard"
    KPI_SCORECARD = "kpi_scorecard"
    TREND_ANALYSIS = "trend_analysis"
    PERFORMANCE_COMPARISON = "performance_comparison"
    STRATEGIC_METRICS = "strategic_metrics"
    RISK_HEATMAP = "risk_heatmap"
    FINANCIAL_SUMMARY = "financial_summary"
    MARKET_POSITION = "market_position"


class ChartType(Enum):
    """Chart types optimized for executive consumption"""
    EXECUTIVE_SUMMARY_CHART = "executive_summary_chart"
    KPI_GAUGE = "kpi_gauge"
    TREND_LINE = "trend_line"
    COMPARISON_BAR = "comparison_bar"
    PERFORMANCE_WATERFALL = "performance_waterfall"
    STRATEGIC_BUBBLE = "strategic_bubble"
    RISK_MATRIX = "risk_matrix"
    GROWTH_TRAJECTORY = "growth_trajectory"


class ExecutiveVisualization:
    """Executive-level visualization with metadata"""
    
    def __init__(
        self,
        id: str,
        title: str,
        visualization_type: VisualizationType,
        chart_type: ChartType,
        data: Dict[str, Any],
        insights: List[str],
        executive_summary: str,
        impact_score: float,
        board_relevance: Dict[str, float]
    ):
        self.id = id
        self.title = title
        self.visualization_type = visualization_type
        self.chart_type = chart_type
        self.data = data
        self.insights = insights
        self.executive_summary = executive_summary
        self.impact_score = impact_score
        self.board_relevance = board_relevance
        self.created_at = datetime.now()


class ExecutiveDataVisualizer:
    """Core data visualization component for executives"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.visualization_templates = self._load_visualization_templates()
        self.executive_guidelines = self._load_executive_guidelines()
    
    def create_executive_visualizations(
        self, 
        data: Dict[str, Any], 
        board_context: Optional[Dict[str, Any]] = None
    ) -> List[ExecutiveVisualization]:
        """Create executive-level data presentations and insight communication"""
        try:
            visualizations = []
            
            # Analyze data structure and content
            data_analysis = self._analyze_data_structure(data)
            
            # Determine optimal visualizations
            viz_recommendations = self._recommend_visualizations(data_analysis, board_context)
            
            # Create each recommended visualization
            for recommendation in viz_recommendations:
                viz = self._create_visualization(
                    data=data,
                    viz_type=recommendation['type'],
                    chart_type=recommendation['chart_type'],
                    focus_area=recommendation['focus_area'],
                    board_context=board_context
                )
                if viz:
                    visualizations.append(viz)
            
            # Optimize visualizations for board consumption
            optimized_visualizations = self._optimize_for_board_consumption(
                visualizations, board_context
            )
            
            self.logger.info(f"Created {len(optimized_visualizations)} executive visualizations")
            return optimized_visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating executive visualizations: {str(e)}")
            raise
    
    def _analyze_data_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data structure to determine visualization opportunities"""
        analysis = {
            "data_types": {},
            "metrics_count": 0,
            "time_series": False,
            "comparative_data": False,
            "hierarchical_data": False,
            "financial_data": False,
            "performance_data": False,
            "strategic_data": False
        }
        
        for key, value in data.items():
            # Analyze data types
            if isinstance(value, (int, float)):
                analysis["data_types"][key] = "numeric"
                analysis["metrics_count"] += 1
            elif isinstance(value, list):
                analysis["data_types"][key] = "list"
                if len(value) > 1 and all(isinstance(item, (int, float)) for item in value):
                    analysis["time_series"] = True
            elif isinstance(value, dict):
                analysis["data_types"][key] = "dict"
                analysis["hierarchical_data"] = True
                if len(value) > 1:
                    analysis["comparative_data"] = True
            
            # Identify data categories
            key_lower = key.lower()
            if any(term in key_lower for term in ['revenue', 'profit', 'cost', 'financial', 'budget']):
                analysis["financial_data"] = True
            if any(term in key_lower for term in ['performance', 'kpi', 'metric', 'score']):
                analysis["performance_data"] = True
            if any(term in key_lower for term in ['strategic', 'goal', 'objective', 'initiative']):
                analysis["strategic_data"] = True
        
        return analysis
    
    def _recommend_visualizations(
        self, 
        data_analysis: Dict[str, Any], 
        board_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Recommend optimal visualizations based on data analysis"""
        recommendations = []
        
        # Financial data visualizations
        if data_analysis["financial_data"]:
            recommendations.append({
                "type": VisualizationType.FINANCIAL_SUMMARY,
                "chart_type": ChartType.EXECUTIVE_SUMMARY_CHART,
                "focus_area": "financial_performance",
                "priority": 1
            })
        
        # Performance data visualizations
        if data_analysis["performance_data"]:
            recommendations.append({
                "type": VisualizationType.KPI_SCORECARD,
                "chart_type": ChartType.KPI_GAUGE,
                "focus_area": "performance_metrics",
                "priority": 1
            })
        
        # Time series data visualizations
        if data_analysis["time_series"]:
            recommendations.append({
                "type": VisualizationType.TREND_ANALYSIS,
                "chart_type": ChartType.TREND_LINE,
                "focus_area": "trend_analysis",
                "priority": 2
            })
        
        # Comparative data visualizations
        if data_analysis["comparative_data"]:
            recommendations.append({
                "type": VisualizationType.PERFORMANCE_COMPARISON,
                "chart_type": ChartType.COMPARISON_BAR,
                "focus_area": "comparative_analysis",
                "priority": 2
            })
        
        # Strategic data visualizations
        if data_analysis["strategic_data"]:
            recommendations.append({
                "type": VisualizationType.STRATEGIC_METRICS,
                "chart_type": ChartType.STRATEGIC_BUBBLE,
                "focus_area": "strategic_initiatives",
                "priority": 1
            })
        
        # Always include executive dashboard if sufficient metrics
        if data_analysis["metrics_count"] >= 3:
            recommendations.append({
                "type": VisualizationType.EXECUTIVE_DASHBOARD,
                "chart_type": ChartType.EXECUTIVE_SUMMARY_CHART,
                "focus_area": "overall_performance",
                "priority": 1
            })
        
        # Sort by priority and limit to top recommendations
        recommendations.sort(key=lambda x: x["priority"])
        return recommendations[:5]  # Limit to 5 visualizations for executive consumption
    
    def _create_visualization(
        self,
        data: Dict[str, Any],
        viz_type: VisualizationType,
        chart_type: ChartType,
        focus_area: str,
        board_context: Optional[Dict[str, Any]]
    ) -> Optional[ExecutiveVisualization]:
        """Create individual visualization"""
        try:
            viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{focus_area}"
            
            # Extract relevant data for focus area
            focused_data = self._extract_focused_data(data, focus_area)
            
            # Generate insights
            insights = self._generate_insights(focused_data, viz_type)
            
            # Create executive summary
            executive_summary = self._create_visualization_summary(focused_data, insights, viz_type)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(focused_data, viz_type, board_context)
            
            # Calculate board relevance
            board_relevance = self._calculate_board_relevance(focused_data, board_context)
            
            # Create title
            title = self._generate_visualization_title(viz_type, focus_area)
            
            visualization = ExecutiveVisualization(
                id=viz_id,
                title=title,
                visualization_type=viz_type,
                chart_type=chart_type,
                data=focused_data,
                insights=insights,
                executive_summary=executive_summary,
                impact_score=impact_score,
                board_relevance=board_relevance
            )
            
            return visualization
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _extract_focused_data(self, data: Dict[str, Any], focus_area: str) -> Dict[str, Any]:
        """Extract data relevant to specific focus area"""
        focused_data = {}
        
        focus_keywords = {
            "financial_performance": ["revenue", "profit", "cost", "margin", "financial"],
            "performance_metrics": ["performance", "kpi", "metric", "score", "efficiency"],
            "trend_analysis": ["growth", "trend", "change", "rate", "trajectory"],
            "comparative_analysis": ["comparison", "vs", "versus", "benchmark", "target"],
            "strategic_initiatives": ["strategic", "initiative", "goal", "objective", "plan"],
            "overall_performance": []  # Include all data for overall view
        }
        
        keywords = focus_keywords.get(focus_area, [])
        
        if focus_area == "overall_performance":
            # Include all numeric data for overall performance
            for key, value in data.items():
                if isinstance(value, (int, float, dict, list)):
                    focused_data[key] = value
        else:
            # Filter data based on keywords
            for key, value in data.items():
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in keywords):
                    focused_data[key] = value
        
        # Ensure we have some data
        if not focused_data and data:
            # Fallback to first few items if no matches
            focused_data = dict(list(data.items())[:3])
        
        return focused_data
    
    def _generate_insights(self, data: Dict[str, Any], viz_type: VisualizationType) -> List[str]:
        """Generate executive insights from data"""
        insights = []
        
        try:
            if viz_type == VisualizationType.FINANCIAL_SUMMARY:
                insights.extend(self._generate_financial_insights(data))
            elif viz_type == VisualizationType.KPI_SCORECARD:
                insights.extend(self._generate_performance_insights(data))
            elif viz_type == VisualizationType.TREND_ANALYSIS:
                insights.extend(self._generate_trend_insights(data))
            elif viz_type == VisualizationType.PERFORMANCE_COMPARISON:
                insights.extend(self._generate_comparison_insights(data))
            elif viz_type == VisualizationType.STRATEGIC_METRICS:
                insights.extend(self._generate_strategic_insights(data))
            elif viz_type == VisualizationType.EXECUTIVE_DASHBOARD:
                insights.extend(self._generate_dashboard_insights(data))
            
            # Ensure we have at least some insights
            if not insights:
                insights = self._generate_generic_insights(data)
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            insights = ["Data analysis completed", "Key metrics identified"]
        
        return insights[:5]  # Limit to 5 key insights
    
    def _generate_financial_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate financial-specific insights"""
        insights = []
        
        # Revenue insights
        if "revenue" in data:
            revenue = data["revenue"]
            if isinstance(revenue, (int, float)):
                if revenue > 1000000:
                    insights.append(f"Revenue of ${revenue/1000000:.1f}M demonstrates strong market position")
                else:
                    insights.append(f"Revenue of ${revenue:,.0f} shows current business scale")
        
        # Growth insights
        if "growth_rate" in data or "growth" in data:
            growth = data.get("growth_rate", data.get("growth", 0))
            if isinstance(growth, (int, float)):
                if growth > 20:
                    insights.append(f"Exceptional growth rate of {growth}% indicates strong market traction")
                elif growth > 10:
                    insights.append(f"Solid growth rate of {growth}% shows healthy business expansion")
                else:
                    insights.append(f"Growth rate of {growth}% requires strategic attention")
        
        # Profitability insights
        if "profit_margin" in data or "margin" in data:
            margin = data.get("profit_margin", data.get("margin", 0))
            if isinstance(margin, (int, float)):
                if margin > 20:
                    insights.append(f"Strong profit margin of {margin}% indicates efficient operations")
                elif margin > 10:
                    insights.append(f"Healthy profit margin of {margin}% shows good cost management")
        
        return insights
    
    def _generate_performance_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate performance-specific insights"""
        insights = []
        
        # Look for performance indicators
        performance_keys = [k for k in data.keys() if any(term in k.lower() 
                          for term in ['performance', 'score', 'rating', 'efficiency'])]
        
        for key in performance_keys:
            value = data[key]
            if isinstance(value, (int, float)):
                if value > 80:
                    insights.append(f"Excellent {key.replace('_', ' ')}: {value}")
                elif value > 60:
                    insights.append(f"Good {key.replace('_', ' ')}: {value}")
                else:
                    insights.append(f"{key.replace('_', ' ')} needs improvement: {value}")
        
        return insights
    
    def _generate_trend_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate trend-specific insights"""
        insights = []
        
        # Look for time-series or trend data
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 1:
                if all(isinstance(item, (int, float)) for item in value):
                    # Calculate trend
                    if value[-1] > value[0]:
                        change = ((value[-1] - value[0]) / value[0]) * 100
                        insights.append(f"{key.replace('_', ' ')} shows positive trend: +{change:.1f}%")
                    elif value[-1] < value[0]:
                        change = ((value[0] - value[-1]) / value[0]) * 100
                        insights.append(f"{key.replace('_', ' ')} shows declining trend: -{change:.1f}%")
        
        return insights
    
    def _generate_comparison_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate comparison-specific insights"""
        insights = []
        
        # Look for comparative data
        for key, value in data.items():
            if isinstance(value, dict):
                values = [v for v in value.values() if isinstance(v, (int, float))]
                if len(values) > 1:
                    max_val = max(values)
                    min_val = min(values)
                    max_key = [k for k, v in value.items() if v == max_val][0]
                    insights.append(f"In {key.replace('_', ' ')}, {max_key} leads with {max_val}")
                    
                    if max_val > min_val * 2:
                        insights.append(f"Significant performance gap identified in {key.replace('_', ' ')}")
        
        return insights
    
    def _generate_strategic_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate strategic-specific insights"""
        insights = []
        
        # Look for strategic indicators
        strategic_keys = [k for k in data.keys() if any(term in k.lower() 
                        for term in ['strategic', 'goal', 'objective', 'initiative', 'target'])]
        
        for key in strategic_keys:
            value = data[key]
            if isinstance(value, str):
                insights.append(f"Strategic focus: {value}")
            elif isinstance(value, list):
                insights.append(f"{len(value)} strategic initiatives in {key.replace('_', ' ')}")
            elif isinstance(value, (int, float)):
                insights.append(f"Strategic target for {key.replace('_', ' ')}: {value}")
        
        return insights
    
    def _generate_dashboard_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate dashboard-level insights"""
        insights = []
        
        # Overall performance summary
        numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
        if numeric_values:
            avg_value = sum(numeric_values) / len(numeric_values)
            insights.append(f"Overall metrics show average performance level of {avg_value:.1f}")
        
        # Key areas identification
        high_performers = []
        for key, value in data.items():
            if isinstance(value, (int, float)) and value > 80:
                high_performers.append(key.replace('_', ' '))
        
        if high_performers:
            insights.append(f"Strong performance in: {', '.join(high_performers[:3])}")
        
        return insights
    
    def _generate_generic_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate generic insights when specific ones aren't available"""
        insights = []
        
        insights.append(f"Analysis covers {len(data)} key data points")
        
        numeric_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        if numeric_data:
            insights.append(f"Quantitative metrics available for {len(numeric_data)} areas")
        
        complex_data = {k: v for k, v in data.items() if isinstance(v, (dict, list))}
        if complex_data:
            insights.append(f"Detailed analysis available for {len(complex_data)} categories")
        
        return insights
    
    def _create_visualization_summary(
        self, 
        data: Dict[str, Any], 
        insights: List[str], 
        viz_type: VisualizationType
    ) -> str:
        """Create executive summary for visualization"""
        summary_parts = []
        
        # Type-specific summary
        type_summaries = {
            VisualizationType.FINANCIAL_SUMMARY: "Financial performance analysis",
            VisualizationType.KPI_SCORECARD: "Key performance indicators overview",
            VisualizationType.TREND_ANALYSIS: "Trend analysis and trajectory assessment",
            VisualizationType.PERFORMANCE_COMPARISON: "Comparative performance evaluation",
            VisualizationType.STRATEGIC_METRICS: "Strategic initiatives progress review",
            VisualizationType.EXECUTIVE_DASHBOARD: "Comprehensive executive overview"
        }
        
        summary_parts.append(type_summaries.get(viz_type, "Data analysis summary"))
        
        # Add key insight
        if insights:
            summary_parts.append(f"Key finding: {insights[0]}")
        
        # Add data scope
        summary_parts.append(f"Based on {len(data)} data elements")
        
        return " | ".join(summary_parts)
    
    def _calculate_impact_score(
        self, 
        data: Dict[str, Any], 
        viz_type: VisualizationType, 
        board_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate visualization impact score"""
        score = 0.5  # Base score
        
        # Data richness factor
        if len(data) > 5:
            score += 0.2
        elif len(data) > 2:
            score += 0.1
        
        # Visualization type importance
        high_impact_types = [
            VisualizationType.FINANCIAL_SUMMARY,
            VisualizationType.EXECUTIVE_DASHBOARD,
            VisualizationType.STRATEGIC_METRICS
        ]
        if viz_type in high_impact_types:
            score += 0.2
        
        # Financial data bonus
        financial_keywords = ['revenue', 'profit', 'cost', 'financial']
        if any(keyword in str(data).lower() for keyword in financial_keywords):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_board_relevance(
        self, 
        data: Dict[str, Any], 
        board_context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate relevance to board members"""
        relevance = {}
        
        # Default relevance for all board members
        base_relevance = 0.7
        
        if board_context and "board_members" in board_context:
            for member_id in board_context["board_members"]:
                relevance[member_id] = base_relevance
        else:
            # Generic board member relevance
            relevance["all_members"] = base_relevance
        
        return relevance
    
    def _generate_visualization_title(self, viz_type: VisualizationType, focus_area: str) -> str:
        """Generate appropriate title for visualization"""
        titles = {
            VisualizationType.FINANCIAL_SUMMARY: "Financial Performance Summary",
            VisualizationType.KPI_SCORECARD: "Key Performance Indicators",
            VisualizationType.TREND_ANALYSIS: "Performance Trends",
            VisualizationType.PERFORMANCE_COMPARISON: "Comparative Analysis",
            VisualizationType.STRATEGIC_METRICS: "Strategic Initiative Progress",
            VisualizationType.EXECUTIVE_DASHBOARD: "Executive Dashboard"
        }
        
        base_title = titles.get(viz_type, "Data Analysis")
        
        # Add focus area if specific
        if focus_area and focus_area != "overall_performance":
            focus_formatted = focus_area.replace('_', ' ').title()
            return f"{base_title} - {focus_formatted}"
        
        return base_title
    
    def _optimize_for_board_consumption(
        self, 
        visualizations: List[ExecutiveVisualization], 
        board_context: Optional[Dict[str, Any]]
    ) -> List[ExecutiveVisualization]:
        """Optimize visualizations for board consumption"""
        # Sort by impact score
        visualizations.sort(key=lambda v: v.impact_score, reverse=True)
        
        # Limit to most impactful visualizations
        max_visualizations = 4  # Optimal for board attention
        optimized = visualizations[:max_visualizations]
        
        # Enhance titles for executive clarity
        for viz in optimized:
            viz.title = self._enhance_executive_title(viz.title)
        
        return optimized
    
    def _enhance_executive_title(self, title: str) -> str:
        """Enhance title for executive clarity"""
        # Add executive context
        if "Performance" in title and "Executive" not in title:
            title = f"Executive {title}"
        
        return title
    
    def _load_visualization_templates(self) -> Dict[str, Any]:
        """Load visualization templates"""
        return {
            "executive_dashboard": {
                "layout": "grid",
                "max_metrics": 6,
                "emphasis": "high_level"
            },
            "kpi_scorecard": {
                "layout": "scorecard",
                "max_kpis": 8,
                "emphasis": "performance"
            },
            "financial_summary": {
                "layout": "financial",
                "max_items": 5,
                "emphasis": "financial_health"
            }
        }
    
    def _load_executive_guidelines(self) -> Dict[str, Any]:
        """Load executive visualization guidelines"""
        return {
            "clarity": {
                "max_data_points": 10,
                "font_size": "large",
                "color_scheme": "high_contrast"
            },
            "engagement": {
                "interaction_points": True,
                "drill_down": "limited",
                "animation": "subtle"
            },
            "timing": {
                "load_time": "fast",
                "update_frequency": "real_time"
            }
        }


class VisualizationOptimizer:
    """Optimizes data visualizations for board consumption"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_board_consumption(
        self, 
        visualizations: List[ExecutiveVisualization],
        board_preferences: Optional[Dict[str, Any]] = None
    ) -> List[ExecutiveVisualization]:
        """Optimize visualizations for board consumption"""
        try:
            # Apply board-specific optimizations
            if board_preferences:
                visualizations = self._apply_board_preferences(visualizations, board_preferences)
            
            # Optimize data density
            visualizations = self._optimize_data_density(visualizations)
            
            # Enhance visual hierarchy
            visualizations = self._enhance_visual_hierarchy(visualizations)
            
            # Optimize for time constraints
            visualizations = self._optimize_for_time_constraints(visualizations)
            
            self.logger.info(f"Optimized {len(visualizations)} visualizations for board consumption")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error optimizing visualizations: {str(e)}")
            return visualizations
    
    def _apply_board_preferences(
        self, 
        visualizations: List[ExecutiveVisualization],
        preferences: Dict[str, Any]
    ) -> List[ExecutiveVisualization]:
        """Apply board-specific preferences"""
        # Adjust based on detail preference
        detail_level = preferences.get("detail_level", "medium")
        
        if detail_level == "low":
            # Simplify visualizations
            for viz in visualizations:
                viz.insights = viz.insights[:3]  # Limit insights
        elif detail_level == "high":
            # Keep full detail
            pass
        
        return visualizations
    
    def _optimize_data_density(self, visualizations: List[ExecutiveVisualization]) -> List[ExecutiveVisualization]:
        """Optimize data density for executive consumption"""
        for viz in visualizations:
            # Limit data points for clarity
            if isinstance(viz.data, dict) and len(viz.data) > 8:
                # Keep most important data points
                sorted_items = sorted(viz.data.items(), 
                                    key=lambda x: isinstance(x[1], (int, float)) and x[1] or 0, 
                                    reverse=True)
                viz.data = dict(sorted_items[:8])
        
        return visualizations
    
    def _enhance_visual_hierarchy(self, visualizations: List[ExecutiveVisualization]) -> List[ExecutiveVisualization]:
        """Enhance visual hierarchy for executive focus"""
        for viz in visualizations:
            # Ensure key insights are prominent
            if len(viz.insights) > 3:
                viz.insights = viz.insights[:3]  # Top 3 insights only
        
        return visualizations
    
    def _optimize_for_time_constraints(self, visualizations: List[ExecutiveVisualization]) -> List[ExecutiveVisualization]:
        """Optimize for board meeting time constraints"""
        # Prioritize high-impact visualizations
        visualizations.sort(key=lambda v: v.impact_score, reverse=True)
        
        # Limit to optimal number for board attention
        return visualizations[:4]


class VisualizationImpactMeasurer:
    """Measures and improves visualization impact"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def measure_visualization_impact(
        self, 
        visualizations: List[ExecutiveVisualization],
        board_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Measure visualization impact and effectiveness"""
        try:
            impact_metrics = {
                "total_visualizations": len(visualizations),
                "average_impact_score": 0.0,
                "high_impact_count": 0,
                "board_relevance_score": 0.0,
                "insight_quality_score": 0.0,
                "recommendations": []
            }
            
            if not visualizations:
                return impact_metrics
            
            # Calculate average impact score
            total_impact = sum(viz.impact_score for viz in visualizations)
            impact_metrics["average_impact_score"] = total_impact / len(visualizations)
            
            # Count high-impact visualizations
            impact_metrics["high_impact_count"] = sum(1 for viz in visualizations if viz.impact_score > 0.8)
            
            # Calculate board relevance score
            relevance_scores = []
            for viz in visualizations:
                if viz.board_relevance:
                    avg_relevance = sum(viz.board_relevance.values()) / len(viz.board_relevance)
                    relevance_scores.append(avg_relevance)
            
            if relevance_scores:
                impact_metrics["board_relevance_score"] = sum(relevance_scores) / len(relevance_scores)
            
            # Calculate insight quality score
            total_insights = sum(len(viz.insights) for viz in visualizations)
            if total_insights > 0:
                impact_metrics["insight_quality_score"] = min(1.0, total_insights / (len(visualizations) * 3))
            
            # Generate recommendations
            impact_metrics["recommendations"] = self._generate_improvement_recommendations(
                visualizations, impact_metrics
            )
            
            self.logger.info(f"Measured impact for {len(visualizations)} visualizations")
            return impact_metrics
            
        except Exception as e:
            self.logger.error(f"Error measuring visualization impact: {str(e)}")
            return {"error": str(e)}
    
    def improve_visualization_impact(
        self, 
        visualizations: List[ExecutiveVisualization],
        impact_metrics: Dict[str, Any]
    ) -> List[ExecutiveVisualization]:
        """Improve visualization impact based on measurements"""
        try:
            improved_visualizations = visualizations.copy()
            
            # Improve low-impact visualizations
            for viz in improved_visualizations:
                if viz.impact_score < 0.6:
                    viz = self._enhance_low_impact_visualization(viz)
            
            # Enhance insights for better quality
            if impact_metrics.get("insight_quality_score", 0) < 0.7:
                improved_visualizations = self._enhance_insights(improved_visualizations)
            
            # Improve board relevance
            if impact_metrics.get("board_relevance_score", 0) < 0.7:
                improved_visualizations = self._improve_board_relevance(improved_visualizations)
            
            self.logger.info(f"Improved {len(improved_visualizations)} visualizations")
            return improved_visualizations
            
        except Exception as e:
            self.logger.error(f"Error improving visualization impact: {str(e)}")
            return visualizations
    
    def _generate_improvement_recommendations(
        self, 
        visualizations: List[ExecutiveVisualization],
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if metrics["average_impact_score"] < 0.7:
            recommendations.append("Increase data relevance and strategic focus")
        
        if metrics["high_impact_count"] < len(visualizations) * 0.5:
            recommendations.append("Enhance visualization types for greater executive impact")
        
        if metrics["board_relevance_score"] < 0.7:
            recommendations.append("Improve alignment with board member interests and expertise")
        
        if metrics["insight_quality_score"] < 0.7:
            recommendations.append("Strengthen analytical insights and actionable recommendations")
        
        if len(visualizations) > 5:
            recommendations.append("Consider reducing number of visualizations for better focus")
        
        return recommendations
    
    def _enhance_low_impact_visualization(self, viz: ExecutiveVisualization) -> ExecutiveVisualization:
        """Enhance low-impact visualization"""
        # Add more strategic context
        if "strategic" not in viz.executive_summary.lower():
            viz.executive_summary = f"Strategic insight: {viz.executive_summary}"
        
        # Enhance insights
        if len(viz.insights) < 3:
            viz.insights.append("Requires board attention and strategic consideration")
        
        # Boost impact score
        viz.impact_score = min(1.0, viz.impact_score + 0.2)
        
        return viz
    
    def _enhance_insights(self, visualizations: List[ExecutiveVisualization]) -> List[ExecutiveVisualization]:
        """Enhance insights across visualizations"""
        for viz in visualizations:
            if len(viz.insights) < 3:
                # Add strategic insight
                viz.insights.append(f"Strategic implications identified in {viz.title.lower()}")
        
        return visualizations
    
    def _improve_board_relevance(self, visualizations: List[ExecutiveVisualization]) -> List[ExecutiveVisualization]:
        """Improve board relevance of visualizations"""
        for viz in visualizations:
            # Increase relevance scores
            for member_id in viz.board_relevance:
                viz.board_relevance[member_id] = min(1.0, viz.board_relevance[member_id] + 0.1)
        
        return visualizations


class ExecutiveDataVisualizationEngine:
    """Main engine for executive data visualization"""
    
    def __init__(self):
        self.visualizer = ExecutiveDataVisualizer()
        self.optimizer = VisualizationOptimizer()
        self.impact_measurer = VisualizationImpactMeasurer()
        self.logger = logging.getLogger(__name__)
    
    def create_optimized_visualizations(
        self, 
        data: Dict[str, Any],
        board_context: Optional[Dict[str, Any]] = None,
        board_preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[ExecutiveVisualization], Dict[str, Any]]:
        """Create and optimize executive visualizations with impact measurement"""
        try:
            # Create initial visualizations
            visualizations = self.visualizer.create_executive_visualizations(data, board_context)
            
            # Optimize for board consumption
            optimized_visualizations = self.optimizer.optimize_for_board_consumption(
                visualizations, board_preferences
            )
            
            # Measure impact
            impact_metrics = self.impact_measurer.measure_visualization_impact(optimized_visualizations)
            
            # Improve if needed
            if impact_metrics.get("average_impact_score", 0) < 0.8:
                optimized_visualizations = self.impact_measurer.improve_visualization_impact(
                    optimized_visualizations, impact_metrics
                )
                # Re-measure after improvements
                impact_metrics = self.impact_measurer.measure_visualization_impact(optimized_visualizations)
            
            self.logger.info(f"Created {len(optimized_visualizations)} optimized executive visualizations")
            return optimized_visualizations, impact_metrics
            
        except Exception as e:
            self.logger.error(f"Error creating optimized visualizations: {str(e)}")
            raise