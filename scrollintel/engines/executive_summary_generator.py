"""
Executive Summary Generator with Natural Language Explanations

This module generates executive summaries with natural language explanations
for complex analytics and insights, making data accessible to business leaders.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of executive summaries"""
    PERFORMANCE_OVERVIEW = "performance_overview"
    FINANCIAL_SUMMARY = "financial_summary"
    OPERATIONAL_METRICS = "operational_metrics"
    STRATEGIC_INSIGHTS = "strategic_insights"
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY_ANALYSIS = "opportunity_analysis"


class Urgency(Enum):
    """Urgency levels for insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutiveInsight:
    """Individual executive insight with natural language explanation"""
    title: str
    summary: str
    detailed_explanation: str
    key_metrics: Dict[str, Any]
    urgency: Urgency
    confidence_level: float
    business_impact: str
    recommended_actions: List[str]
    supporting_data: Dict[str, Any]
    visualization_suggestions: List[str]


@dataclass
class ExecutiveSummary:
    """Complete executive summary"""
    title: str
    executive_overview: str
    key_highlights: List[str]
    critical_insights: List[ExecutiveInsight]
    performance_metrics: Dict[str, Any]
    trends_and_patterns: List[str]
    risks_and_opportunities: Dict[str, List[str]]
    strategic_recommendations: List[str]
    next_steps: List[str]
    generated_at: datetime
    data_period: Dict[str, datetime]
    confidence_score: float


class ExecutiveSummaryGenerator:
    """
    Advanced executive summary generator with natural language processing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.business_context = {}
        
    def set_business_context(self, context: Dict[str, Any]):
        """Set business context for more relevant summaries"""
        self.business_context = context
        self.logger.info("Business context updated")
    
    async def generate_executive_summary(
        self,
        data: Dict[str, Any],
        summary_type: SummaryType = SummaryType.PERFORMANCE_OVERVIEW,
        target_audience: str = "executive",
        custom_focus: Optional[List[str]] = None
    ) -> ExecutiveSummary:
        """
        Generate comprehensive executive summary with natural language explanations
        
        Args:
            data: Analytics data and insights
            summary_type: Type of summary to generate
            target_audience: Target audience (executive, board, department_head)
            custom_focus: Custom focus areas for the summary
            
        Returns:
            Complete executive summary
        """
        try:
            self.logger.info(f"Generating {summary_type.value} executive summary for {target_audience}")
            
            # Extract key insights from data
            insights = await self._extract_key_insights(data, summary_type, custom_focus)
            
            # Generate executive overview
            overview = await self._generate_executive_overview(insights, summary_type, target_audience)
            
            # Create key highlights
            highlights = await self._create_key_highlights(insights)
            
            # Generate performance metrics summary
            metrics = await self._summarize_performance_metrics(data)
            
            # Identify trends and patterns
            trends = await self._identify_trends_and_patterns(data, insights)
            
            # Assess risks and opportunities
            risks_opportunities = await self._assess_risks_and_opportunities(insights)
            
            # Generate strategic recommendations
            recommendations = await self._generate_strategic_recommendations(insights, summary_type)
            
            # Create next steps
            next_steps = await self._create_next_steps(insights, recommendations)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(insights)
            
            # Create executive summary
            summary = ExecutiveSummary(
                title=self._generate_summary_title(summary_type, target_audience),
                executive_overview=overview,
                key_highlights=highlights,
                critical_insights=insights,
                performance_metrics=metrics,
                trends_and_patterns=trends,
                risks_and_opportunities=risks_opportunities,
                strategic_recommendations=recommendations,
                next_steps=next_steps,
                generated_at=datetime.utcnow(),
                data_period=data.get('data_period', {
                    'start': datetime.utcnow() - timedelta(days=30),
                    'end': datetime.utcnow()
                }),
                confidence_score=confidence_score
            )
            
            self.logger.info("Executive summary generated successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            raise
    
    async def _extract_key_insights(
        self, 
        data: Dict[str, Any], 
        summary_type: SummaryType,
        custom_focus: Optional[List[str]]
    ) -> List[ExecutiveInsight]:
        """Extract and prioritize key insights from analytics data"""
        insights = []
        
        try:
            # Process statistical analysis results
            if 'statistical_analysis' in data:
                stat_insights = await self._process_statistical_insights(
                    data['statistical_analysis'], summary_type
                )
                insights.extend(stat_insights)
            
            # Process ML insights
            if 'ml_insights' in data:
                ml_insights = await self._process_ml_insights(
                    data['ml_insights'], summary_type
                )
                insights.extend(ml_insights)
            
            # Process ROI analysis
            if 'roi_analysis' in data:
                roi_insights = await self._process_roi_insights(
                    data['roi_analysis'], summary_type
                )
                insights.extend(roi_insights)
            
            # Apply custom focus if provided
            if custom_focus:
                insights = self._apply_custom_focus(insights, custom_focus)
            
            # Sort by urgency and business impact
            insights.sort(key=lambda x: (x.urgency.value, -x.confidence_level), reverse=True)
            
            # Limit to top insights
            return insights[:10]
            
        except Exception as e:
            self.logger.error(f"Error extracting key insights: {str(e)}")
            return []
    
    async def _process_statistical_insights(
        self, 
        statistical_data: Dict[str, Any], 
        summary_type: SummaryType
    ) -> List[ExecutiveInsight]:
        """Process statistical analysis results into executive insights"""
        insights = []
        
        try:
            # Process correlation insights
            if 'correlation' in statistical_data:
                for result in statistical_data['correlation']:
                    correlation = result.get('result', {}).get('correlation', 0)
                    if abs(correlation) > 0.7:
                        insight = ExecutiveInsight(
                            title=f"Strong Correlation: {result.get('metric_name', 'Unknown')}",
                            summary=f"Identified {abs(correlation):.1%} correlation between key metrics",
                            detailed_explanation=self._explain_correlation(result),
                            key_metrics={"correlation": correlation, "p_value": result.get('p_value')},
                            urgency=Urgency.MEDIUM,
                            confidence_level=result.get('confidence_level', 0.95),
                            business_impact=self._assess_correlation_impact(correlation),
                            recommended_actions=["Investigate causal relationships", "Develop predictive models"],
                            supporting_data=result,
                            visualization_suggestions=["scatter_plot", "correlation_matrix"]
                        )
                        insights.append(insight)
            
            # Process trend insights
            if 'trend_analysis' in statistical_data:
                for result in statistical_data['trend_analysis']:
                    trend_strength = result.get('result', {}).get('trend_strength', 0)
                    if trend_strength > 0.5:
                        insight = ExecutiveInsight(
                            title=f"Trend in {result.get('metric_name', 'Metric')}",
                            summary=f"Detected {result.get('result', {}).get('trend_direction', 'unknown')} trend",
                            detailed_explanation=self._explain_trend(result),
                            key_metrics=result.get('result', {}),
                            urgency=self._assess_trend_urgency(result),
                            confidence_level=result.get('confidence_level', 0.95),
                            business_impact=self._assess_trend_impact(result),
                            recommended_actions=["Monitor trend continuation", "Adjust strategies"],
                            supporting_data=result,
                            visualization_suggestions=["line_chart", "trend_analysis"]
                        )
                        insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error processing statistical insights: {str(e)}")
        
        return insights
    
    async def _process_ml_insights(
        self, 
        ml_data: List[Dict[str, Any]], 
        summary_type: SummaryType
    ) -> List[ExecutiveInsight]:
        """Process ML insights into executive format"""
        insights = []
        
        try:
            for ml_insight in ml_data:
                insight = ExecutiveInsight(
                    title=ml_insight.get('title', 'ML Insight'),
                    summary=ml_insight.get('description', ''),
                    detailed_explanation=self._expand_ml_explanation(ml_insight),
                    key_metrics={
                        "confidence_score": ml_insight.get('confidence_score', 0),
                        "impact_score": ml_insight.get('impact_score', 0)
                    },
                    urgency=self._map_ml_urgency(ml_insight),
                    confidence_level=ml_insight.get('confidence_score', 0.5),
                    business_impact=self._assess_ml_impact(ml_insight),
                    recommended_actions=ml_insight.get('action_items', []),
                    supporting_data=ml_insight,
                    visualization_suggestions=["chart", "dashboard"]
                )
                insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error processing ML insights: {str(e)}")
        
        return insights
    
    async def _process_roi_insights(
        self, 
        roi_data: Dict[str, Any], 
        summary_type: SummaryType
    ) -> List[ExecutiveInsight]:
        """Process ROI analysis into executive insights"""
        insights = []
        
        try:
            # Overall ROI performance
            if 'total_roi' in roi_data:
                total_roi = roi_data['total_roi']
                insight = ExecutiveInsight(
                    title="Return on Investment Performance",
                    summary=f"Overall ROI of {total_roi:.1%}",
                    detailed_explanation=self._explain_roi_performance(roi_data),
                    key_metrics=roi_data,
                    urgency=Urgency.HIGH if total_roi < 0 else Urgency.MEDIUM,
                    confidence_level=0.9,
                    business_impact=self._assess_roi_impact(total_roi),
                    recommended_actions=["Review investment strategy", "Optimize resource allocation"],
                    supporting_data=roi_data,
                    visualization_suggestions=["roi_chart", "waterfall_chart"]
                )
                insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error processing ROI insights: {str(e)}")
        
        return insights
    
    async def _generate_executive_overview(
        self, 
        insights: List[ExecutiveInsight], 
        summary_type: SummaryType,
        target_audience: str
    ) -> str:
        """Generate executive overview narrative"""
        try:
            # Count insights by urgency
            critical_count = sum(1 for i in insights if i.urgency == Urgency.CRITICAL)
            high_count = sum(1 for i in insights if i.urgency == Urgency.HIGH)
            
            # Generate opening based on urgency
            if critical_count > 0:
                opening = f"This report identifies {critical_count} critical issue{'s' if critical_count > 1 else ''} requiring immediate attention."
            elif high_count > 0:
                opening = f"Analysis reveals {high_count} high-priority item{'s' if high_count > 1 else ''} for consideration."
            else:
                opening = "Overall performance indicators show stable operations with optimization opportunities."
            
            # Add confidence statement
            avg_confidence = sum(i.confidence_level for i in insights) / len(insights) if insights else 0.5
            confidence_text = f" Analysis confidence level: {avg_confidence:.0%}."
            
            return f"{opening}{confidence_text}"
            
        except Exception as e:
            self.logger.error(f"Error generating executive overview: {str(e)}")
            return "Executive summary analysis completed."
    
    async def _create_key_highlights(self, insights: List[ExecutiveInsight]) -> List[str]:
        """Create key highlights from insights"""
        highlights = []
        
        try:
            # Top 5 most impactful insights
            top_insights = sorted(insights, key=lambda x: x.confidence_level, reverse=True)[:5]
            
            for insight in top_insights:
                highlight = f"{insight.title}: {insight.summary}"
                highlights.append(highlight)
            
            return highlights
            
        except Exception as e:
            self.logger.error(f"Error creating key highlights: {str(e)}")
            return ["Analysis completed with actionable insights identified."]
    
    async def _summarize_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize key performance metrics"""
        try:
            metrics_summary = {}
            
            # Extract key metrics from various data sources
            if 'dashboard_metrics' in data:
                metrics_summary.update(data['dashboard_metrics'])
            
            if 'roi_analysis' in data:
                metrics_summary['roi_metrics'] = data['roi_analysis']
            
            return metrics_summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing performance metrics: {str(e)}")
            return {}
    
    async def _identify_trends_and_patterns(
        self, 
        data: Dict[str, Any], 
        insights: List[ExecutiveInsight]
    ) -> List[str]:
        """Identify key trends and patterns"""
        trends = []
        
        try:
            # Extract trends from insights
            trend_insights = [i for i in insights if 'trend' in i.title.lower()]
            
            for insight in trend_insights:
                trend_desc = f"{insight.title}: {insight.summary}"
                trends.append(trend_desc)
            
            # Add general patterns
            if len(insights) > 5:
                trends.append(f"Analysis identified {len(insights)} key areas requiring attention")
            
            return trends[:10]  # Limit to top 10 trends
            
        except Exception as e:
            self.logger.error(f"Error identifying trends and patterns: {str(e)}")
            return ["Multiple data patterns identified."]
    
    async def _assess_risks_and_opportunities(self, insights: List[ExecutiveInsight]) -> Dict[str, List[str]]:
        """Assess risks and opportunities from insights"""
        try:
            risks = []
            opportunities = []
            
            for insight in insights:
                # Classify as risk or opportunity based on content
                if any(word in insight.business_impact.lower() for word in ['risk', 'decline', 'negative']):
                    risks.append(f"{insight.title}: {insight.business_impact}")
                elif any(word in insight.business_impact.lower() for word in ['opportunity', 'growth', 'positive']):
                    opportunities.append(f"{insight.title}: {insight.business_impact}")
                
                # High urgency items are typically risks
                if insight.urgency in [Urgency.HIGH, Urgency.CRITICAL]:
                    if insight.title not in [r.split(':')[0] for r in risks]:
                        risks.append(f"{insight.title}: Requires immediate attention")
            
            return {
                'risks': risks[:5],  # Top 5 risks
                'opportunities': opportunities[:5]  # Top 5 opportunities
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing risks and opportunities: {str(e)}")
            return {'risks': [], 'opportunities': []}
    
    async def _generate_strategic_recommendations(
        self, 
        insights: List[ExecutiveInsight], 
        summary_type: SummaryType
    ) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        try:
            # Collect all recommended actions
            all_actions = []
            for insight in insights:
                all_actions.extend(insight.recommended_actions)
            
            # Prioritize and deduplicate
            unique_actions = list(set(all_actions))
            
            return unique_actions[:10]  # Top 10 actions
            
        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {str(e)}")
            return ["Implement data-driven decision making", "Establish regular monitoring"]
    
    async def _create_next_steps(
        self, 
        insights: List[ExecutiveInsight], 
        recommendations: List[str]
    ) -> List[str]:
        """Create actionable next steps"""
        next_steps = []
        
        try:
            # Immediate actions for critical insights
            critical_insights = [i for i in insights if i.urgency == Urgency.CRITICAL]
            if critical_insights:
                next_steps.append("Address critical issues within 24-48 hours")
            
            # High priority actions
            high_priority_insights = [i for i in insights if i.urgency == Urgency.HIGH]
            if high_priority_insights:
                next_steps.append("Develop action plans for high-priority items within one week")
            
            # Strategic planning
            next_steps.append("Schedule follow-up analysis in 30 days")
            next_steps.append("Implement recommended monitoring systems")
            
            # Top recommendations as next steps
            next_steps.extend(recommendations[:3])
            
            return next_steps[:8]  # Limit to 8 next steps
            
        except Exception as e:
            self.logger.error(f"Error creating next steps: {str(e)}")
            return ["Review analysis results", "Implement monitoring systems"]
    
    # Helper methods for natural language generation
    
    def _explain_correlation(self, result: Dict[str, Any]) -> str:
        """Generate detailed explanation for correlation"""
        correlation = result.get('result', {}).get('correlation', 0)
        metric_name = result.get('metric_name', 'metrics')
        
        strength = "strong" if abs(correlation) > 0.7 else "moderate"
        direction = "positive" if correlation > 0 else "negative"
        
        explanation = f"A {strength} {direction} correlation of {correlation:.3f} was discovered between {metric_name}. "
        
        if correlation > 0:
            explanation += "This indicates that as one metric increases, the other tends to increase as well."
        else:
            explanation += "This indicates that as one metric increases, the other tends to decrease."
        
        return explanation
    
    def _explain_trend(self, result: Dict[str, Any]) -> str:
        """Generate detailed explanation for trend"""
        trend_data = result.get('result', {})
        metric_name = result.get('metric_name', 'metric')
        direction = trend_data.get('trend_direction', 'unknown')
        strength = trend_data.get('trend_strength', 0)
        
        explanation = f"Analysis of {metric_name} reveals a {direction} trend with {strength:.1%} correlation strength. "
        
        if direction == 'increasing':
            explanation += "The metric shows consistent upward movement."
        elif direction == 'decreasing':
            explanation += "The metric shows consistent downward movement."
        
        return explanation
    
    def _expand_ml_explanation(self, ml_insight: Dict[str, Any]) -> str:
        """Expand ML insight with detailed explanation"""
        base_description = ml_insight.get('description', '')
        confidence = ml_insight.get('confidence_score', 0)
        impact = ml_insight.get('impact_score', 0)
        
        explanation = f"{base_description} "
        explanation += f"This insight has a confidence level of {confidence:.0%} and "
        explanation += f"potential business impact score of {impact:.1f}/1.0."
        
        return explanation
    
    def _explain_roi_performance(self, roi_data: Dict[str, Any]) -> str:
        """Generate detailed ROI explanation"""
        total_roi = roi_data.get('total_roi', 0)
        investment = roi_data.get('total_investment', 0)
        returns = roi_data.get('total_returns', 0)
        
        explanation = f"ROI analysis shows {total_roi:.1%} overall return "
        explanation += f"on ${investment:,.0f} investment, generating ${returns:,.0f} in returns. "
        
        if total_roi > 0.2:
            explanation += "This represents excellent performance."
        elif total_roi > 0.1:
            explanation += "This represents good performance."
        elif total_roi > 0:
            explanation += "This represents positive but modest performance."
        else:
            explanation += "This represents negative performance requiring review."
        
        return explanation
    
    # Assessment methods
    
    def _assess_correlation_impact(self, correlation: float) -> str:
        """Assess business impact of correlation"""
        if abs(correlation) > 0.8:
            return "High impact: Strong correlation enables predictive modeling"
        elif abs(correlation) > 0.6:
            return "Medium impact: Moderate correlation provides strategic insights"
        else:
            return "Low impact: Weak correlation with limited actionable value"
    
    def _assess_trend_impact(self, result: Dict[str, Any]) -> str:
        """Assess business impact of trend"""
        direction = result.get('result', {}).get('trend_direction', 'unknown')
        strength = result.get('result', {}).get('trend_strength', 0)
        
        if direction == 'increasing' and strength > 0.7:
            return "Positive impact: Strong upward trend indicates improving performance"
        elif direction == 'decreasing' and strength > 0.7:
            return "Negative impact: Strong downward trend requires intervention"
        else:
            return "Moderate impact: Trend provides directional insight"
    
    def _assess_ml_impact(self, ml_insight: Dict[str, Any]) -> str:
        """Assess business impact of ML insight"""
        impact_score = ml_insight.get('impact_score', 0)
        
        if impact_score > 0.8:
            return "High impact: Significant potential to improve outcomes"
        elif impact_score > 0.5:
            return "Medium impact: Moderate potential for improvement"
        else:
            return "Low impact: Limited but measurable effect"
    
    def _assess_roi_impact(self, total_roi: float) -> str:
        """Assess business impact of ROI"""
        if total_roi > 0.2:
            return "Excellent impact: ROI significantly exceeds expectations"
        elif total_roi > 0.1:
            return "Good impact: ROI meets expectations"
        elif total_roi > 0:
            return "Positive impact: ROI is positive but modest"
        else:
            return "Negative impact: Investments not generating returns"
    
    # Utility methods
    
    def _assess_trend_urgency(self, result: Dict[str, Any]) -> Urgency:
        """Assess urgency of trend finding"""
        direction = result.get('result', {}).get('trend_direction', 'unknown')
        strength = result.get('result', {}).get('trend_strength', 0)
        
        if direction == 'decreasing' and strength > 0.7:
            return Urgency.HIGH
        elif strength > 0.8:
            return Urgency.MEDIUM
        else:
            return Urgency.LOW
    
    def _map_ml_urgency(self, ml_insight: Dict[str, Any]) -> Urgency:
        """Map ML insight to urgency level"""
        impact_score = ml_insight.get('impact_score', 0)
        confidence_score = ml_insight.get('confidence_score', 0)
        
        combined_score = impact_score * confidence_score
        
        if combined_score > 0.8:
            return Urgency.HIGH
        elif combined_score > 0.5:
            return Urgency.MEDIUM
        else:
            return Urgency.LOW
    
    def _apply_custom_focus(self, insights: List[ExecutiveInsight], custom_focus: List[str]) -> List[ExecutiveInsight]:
        """Apply custom focus areas to filter insights"""
        if not custom_focus:
            return insights
        
        focused_insights = []
        for insight in insights:
            for focus_area in custom_focus:
                if focus_area.lower() in insight.title.lower() or focus_area.lower() in insight.summary.lower():
                    focused_insights.append(insight)
                    break
        
        return focused_insights
    
    def _calculate_confidence_score(self, insights: List[ExecutiveInsight]) -> float:
        """Calculate overall confidence score"""
        if not insights:
            return 0.5
        
        return sum(insight.confidence_level for insight in insights) / len(insights)
    
    def _generate_summary_title(self, summary_type: SummaryType, target_audience: str) -> str:
        """Generate appropriate summary title"""
        type_titles = {
            SummaryType.PERFORMANCE_OVERVIEW: "Performance Overview",
            SummaryType.FINANCIAL_SUMMARY: "Financial Summary",
            SummaryType.OPERATIONAL_METRICS: "Operational Metrics Report",
            SummaryType.STRATEGIC_INSIGHTS: "Strategic Insights",
            SummaryType.RISK_ASSESSMENT: "Risk Assessment Report",
            SummaryType.OPPORTUNITY_ANALYSIS: "Opportunity Analysis"
        }
        
        base_title = type_titles.get(summary_type, "Executive Summary")
        
        if target_audience == "board":
            return f"Board Report: {base_title}"
        elif target_audience == "department_head":
            return f"Department Summary: {base_title}"
        else:
            return f"Executive Summary: {base_title}"