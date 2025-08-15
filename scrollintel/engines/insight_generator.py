"""
AI Insight Generation Engine for Advanced Analytics Dashboard System.

This engine provides automated insight generation with natural language processing,
pattern detection, anomaly detection, and business context understanding.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import re

from .base_engine import BaseEngine, EngineCapability, EngineStatus
from ..models.insight_models import (
    Pattern, Insight, ActionRecommendation, BusinessContext, Anomaly,
    InsightType, PatternType, SignificanceLevel, ActionPriority
)
from ..models.dashboard_models import BusinessMetric

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsData:
    """Container for analytics data to be processed."""
    metrics: List[BusinessMetric]
    time_range: Tuple[datetime, datetime]
    context: Dict[str, Any]


@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    slope: float
    r_squared: float
    p_value: float
    trend_strength: str
    direction: str
    confidence: float


class PatternDetector:
    """Detects significant patterns in business metrics data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    async def detect_trends(self, data: List[BusinessMetric]) -> List[Pattern]:
        """Detect trend patterns in time series data."""
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        # Group by metric name
        metric_groups = {}
        for metric in data:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric)
        
        for metric_name, metrics in metric_groups.items():
            if len(metrics) < 3:
                continue
                
            # Sort by timestamp
            metrics.sort(key=lambda x: x.timestamp)
            
            # Extract values and timestamps
            values = [m.value for m in metrics]
            timestamps = [m.timestamp.timestamp() for m in metrics]
            
            # Perform trend analysis
            trend_analysis = self._analyze_trend(timestamps, values)
            
            if trend_analysis.confidence > 0.7:  # High confidence threshold
                pattern_type = (PatternType.INCREASING_TREND 
                              if trend_analysis.direction == "increasing" 
                              else PatternType.DECREASING_TREND)
                
                significance = self._determine_significance(trend_analysis.confidence, abs(trend_analysis.slope))
                
                pattern = Pattern(
                    type=pattern_type.value,
                    metric_name=metric_name,
                    metric_category=metrics[0].category,
                    description=f"{trend_analysis.direction.title()} trend detected with {trend_analysis.confidence:.2%} confidence",
                    confidence=trend_analysis.confidence,
                    significance=significance.value,
                    start_time=metrics[0].timestamp,
                    end_time=metrics[-1].timestamp,
                    data_points=[{"timestamp": m.timestamp.isoformat(), "value": m.value} for m in metrics],
                    statistical_measures={
                        "slope": trend_analysis.slope,
                        "r_squared": trend_analysis.r_squared,
                        "p_value": trend_analysis.p_value,
                        "trend_strength": trend_analysis.trend_strength
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def detect_anomalies(self, data: List[BusinessMetric]) -> List[Anomaly]:
        """Detect anomalies in business metrics."""
        anomalies = []
        
        # Group by metric name
        metric_groups = {}
        for metric in data:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric)
        
        for metric_name, metrics in metric_groups.items():
            if len(metrics) < 10:  # Need sufficient data for anomaly detection
                continue
            
            values = np.array([m.value for m in metrics]).reshape(-1, 1)
            
            # Fit anomaly detector
            self.anomaly_detector.fit(values)
            anomaly_scores = self.anomaly_detector.decision_function(values)
            anomaly_labels = self.anomaly_detector.predict(values)
            
            # Calculate statistical thresholds
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            for i, (metric, score, label) in enumerate(zip(metrics, anomaly_scores, anomaly_labels)):
                if label == -1:  # Anomaly detected
                    deviation = abs(metric.value - mean_val) / std_val if std_val > 0 else 0
                    
                    anomaly_type = "spike" if metric.value > mean_val else "drop"
                    severity = self._determine_anomaly_severity(deviation)
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        metric_value=metric.value,
                        expected_value=float(mean_val),
                        deviation_score=deviation,
                        anomaly_type=anomaly_type,
                        severity=severity.value,
                        detected_at=datetime.utcnow(),
                        context={"isolation_score": float(score)}
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_correlations(self, data: List[BusinessMetric]) -> List[Pattern]:
        """Detect correlations between different metrics."""
        patterns = []
        
        # Create DataFrame for correlation analysis
        df_data = {}
        for metric in data:
            if metric.name not in df_data:
                df_data[metric.name] = []
            df_data[metric.name].append({
                'timestamp': metric.timestamp,
                'value': metric.value
            })
        
        # Convert to time-aligned DataFrame
        dfs = {}
        for metric_name, values in df_data.items():
            if len(values) < 5:
                continue
            df = pd.DataFrame(values)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').resample('1H').mean().fillna(method='ffill')
            dfs[metric_name] = df['value']
        
        if len(dfs) < 2:
            return patterns
        
        # Create combined DataFrame
        combined_df = pd.DataFrame(dfs).dropna()
        
        if combined_df.shape[0] < 5 or combined_df.shape[1] < 2:
            return patterns
        
        # Calculate correlations
        correlation_matrix = combined_df.corr()
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                metric1 = correlation_matrix.columns[i]
                metric2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                
                if abs(correlation) > 0.7:  # Strong correlation threshold
                    significance = self._determine_significance(abs(correlation), abs(correlation))
                    
                    pattern = Pattern(
                        type=PatternType.CORRELATION.value,
                        metric_name=f"{metric1} vs {metric2}",
                        description=f"Strong {'positive' if correlation > 0 else 'negative'} correlation ({correlation:.3f}) between {metric1} and {metric2}",
                        confidence=abs(correlation),
                        significance=significance.value,
                        statistical_measures={
                            "correlation_coefficient": correlation,
                            "metric1": metric1,
                            "metric2": metric2
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_trend(self, timestamps: List[float], values: List[float]) -> TrendAnalysis:
        """Analyze trend in time series data."""
        if len(timestamps) < 2:
            return TrendAnalysis(0, 0, 1, "none", "stable", 0)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
        
        r_squared = r_value ** 2
        
        # Determine trend strength and direction
        if abs(slope) < 1e-10:
            trend_strength = "none"
            direction = "stable"
        elif r_squared > 0.8:
            trend_strength = "strong"
        elif r_squared > 0.5:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"
        
        if slope > 0:
            direction = "increasing"
        elif slope < 0:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # Calculate confidence based on R-squared and p-value
        confidence = r_squared * (1 - p_value) if p_value < 0.05 else r_squared * 0.5
        
        return TrendAnalysis(
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            trend_strength=trend_strength,
            direction=direction,
            confidence=min(confidence, 1.0)
        )
    
    def _determine_significance(self, confidence: float, magnitude: float) -> SignificanceLevel:
        """Determine significance level based on confidence and magnitude."""
        score = confidence * magnitude
        
        if score > 0.8:
            return SignificanceLevel.CRITICAL
        elif score > 0.6:
            return SignificanceLevel.HIGH
        elif score > 0.4:
            return SignificanceLevel.MEDIUM
        else:
            return SignificanceLevel.LOW
    
    def _determine_anomaly_severity(self, deviation: float) -> SignificanceLevel:
        """Determine anomaly severity based on deviation score."""
        if deviation > 3:
            return SignificanceLevel.CRITICAL
        elif deviation > 2:
            return SignificanceLevel.HIGH
        elif deviation > 1.5:
            return SignificanceLevel.MEDIUM
        else:
            return SignificanceLevel.LOW


class InsightEngine:
    """Generates natural language insights from detected patterns."""
    
    def __init__(self):
        self.insight_templates = {
            InsightType.TREND: {
                "increasing": "The {metric} has shown a {strength} upward trend over the past {period}, increasing by {change}. This indicates {business_impact}.",
                "decreasing": "The {metric} has shown a {strength} downward trend over the past {period}, decreasing by {change}. This suggests {business_impact}."
            },
            InsightType.ANOMALY: {
                "spike": "An unusual spike in {metric} was detected on {date}, reaching {value} which is {deviation}x higher than normal. This could indicate {potential_cause}.",
                "drop": "An unusual drop in {metric} was detected on {date}, falling to {value} which is {deviation}x lower than normal. This may suggest {potential_cause}."
            },
            InsightType.CORRELATION: {
                "positive": "A strong positive correlation ({correlation}) was found between {metric1} and {metric2}. When {metric1} increases, {metric2} tends to increase as well.",
                "negative": "A strong negative correlation ({correlation}) was found between {metric1} and {metric2}. When {metric1} increases, {metric2} tends to decrease."
            }
        }
    
    async def generate_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate natural language insights from patterns."""
        insights = []
        
        for pattern in patterns:
            insight = await self._create_insight_from_pattern(pattern)
            if insight:
                insights.append(insight)
        
        return insights
    
    async def _create_insight_from_pattern(self, pattern: Pattern) -> Optional[Insight]:
        """Create an insight from a detected pattern."""
        try:
            if pattern.type == PatternType.INCREASING_TREND.value:
                return await self._create_trend_insight(pattern, "increasing")
            elif pattern.type == PatternType.DECREASING_TREND.value:
                return await self._create_trend_insight(pattern, "decreasing")
            elif pattern.type == PatternType.CORRELATION.value:
                return await self._create_correlation_insight(pattern)
            else:
                return await self._create_generic_insight(pattern)
        except Exception as e:
            logger.error(f"Error creating insight from pattern {pattern.id}: {e}")
            return None
    
    async def _create_trend_insight(self, pattern: Pattern, direction: str) -> Insight:
        """Create insight for trend patterns."""
        stats = pattern.statistical_measures
        
        # Calculate time period
        period = self._calculate_period(pattern.start_time, pattern.end_time)
        
        # Calculate change percentage
        data_points = pattern.data_points
        if len(data_points) >= 2:
            start_value = data_points[0]["value"]
            end_value = data_points[-1]["value"]
            change_pct = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0
        else:
            change_pct = 0
        
        # Generate description
        template = self.insight_templates[InsightType.TREND][direction]
        description = template.format(
            metric=pattern.metric_name,
            strength=stats.get("trend_strength", "moderate"),
            period=period,
            change=f"{abs(change_pct):.1f}%",
            business_impact=self._get_business_impact(pattern.metric_name, direction, change_pct)
        )
        
        # Generate title
        title = f"{direction.title()} trend in {pattern.metric_name}"
        
        # Determine insight type and priority
        insight_type = InsightType.TREND
        priority = self._determine_priority(pattern.significance, abs(change_pct))
        
        return Insight(
            pattern_id=pattern.id,
            type=insight_type.value,
            title=title,
            description=description,
            explanation=self._generate_explanation(pattern, direction),
            business_impact=self._get_business_impact(pattern.metric_name, direction, change_pct),
            confidence=pattern.confidence,
            significance=abs(change_pct) / 100,  # Normalize to 0-1
            priority=priority.value,
            tags=["trend", direction, pattern.metric_category or "general"],
            affected_metrics=[pattern.metric_name]
        )
    
    async def _create_correlation_insight(self, pattern: Pattern) -> Insight:
        """Create insight for correlation patterns."""
        stats = pattern.statistical_measures
        correlation = stats.get("correlation_coefficient", 0)
        metric1 = stats.get("metric1", "")
        metric2 = stats.get("metric2", "")
        
        direction = "positive" if correlation > 0 else "negative"
        
        template = self.insight_templates[InsightType.CORRELATION][direction]
        description = template.format(
            correlation=f"{correlation:.3f}",
            metric1=metric1,
            metric2=metric2
        )
        
        title = f"{direction.title()} correlation between {metric1} and {metric2}"
        
        return Insight(
            pattern_id=pattern.id,
            type=InsightType.CORRELATION.value,
            title=title,
            description=description,
            explanation=f"Statistical analysis reveals a {direction} correlation coefficient of {correlation:.3f}, indicating that these metrics tend to move {'together' if correlation > 0 else 'in opposite directions'}.",
            business_impact=self._get_correlation_business_impact(metric1, metric2, correlation),
            confidence=abs(correlation),
            significance=abs(correlation),
            priority=self._determine_priority(pattern.significance, abs(correlation) * 100).value,
            tags=["correlation", direction, "relationship"],
            affected_metrics=[metric1, metric2]
        )
    
    async def _create_generic_insight(self, pattern: Pattern) -> Insight:
        """Create generic insight for other pattern types."""
        return Insight(
            pattern_id=pattern.id,
            type=InsightType.OPPORTUNITY.value,
            title=f"Pattern detected in {pattern.metric_name}",
            description=pattern.description,
            explanation=f"A {pattern.type} pattern was detected with {pattern.confidence:.2%} confidence.",
            business_impact="Further analysis recommended to understand business implications.",
            confidence=pattern.confidence,
            significance=pattern.confidence,
            priority=ActionPriority.MEDIUM.value,
            tags=[pattern.type, pattern.metric_category or "general"],
            affected_metrics=[pattern.metric_name]
        )
    
    def _calculate_period(self, start_time: datetime, end_time: datetime) -> str:
        """Calculate human-readable time period."""
        if not start_time or not end_time:
            return "recent period"
        
        delta = end_time - start_time
        days = delta.days
        
        if days < 1:
            return "past few hours"
        elif days == 1:
            return "past day"
        elif days < 7:
            return f"past {days} days"
        elif days < 30:
            weeks = days // 7
            return f"past {weeks} week{'s' if weeks > 1 else ''}"
        elif days < 365:
            months = days // 30
            return f"past {months} month{'s' if months > 1 else ''}"
        else:
            years = days // 365
            return f"past {years} year{'s' if years > 1 else ''}"
    
    def _get_business_impact(self, metric_name: str, direction: str, change_pct: float) -> str:
        """Generate business impact explanation based on metric and change."""
        metric_lower = metric_name.lower()
        
        # Define positive and negative indicators for different metrics
        positive_metrics = ["revenue", "profit", "sales", "conversion", "efficiency", "satisfaction", "performance"]
        negative_metrics = ["cost", "error", "latency", "churn", "downtime", "complaints"]
        
        is_positive_metric = any(pos in metric_lower for pos in positive_metrics)
        is_negative_metric = any(neg in metric_lower for neg in negative_metrics)
        
        if is_positive_metric:
            if direction == "increasing":
                return "positive business performance and growth opportunities"
            else:
                return "potential concerns that may require attention"
        elif is_negative_metric:
            if direction == "increasing":
                return "potential issues that may need immediate attention"
            else:
                return "improvements in operational efficiency"
        else:
            return "changes in business operations that warrant further analysis"
    
    def _get_correlation_business_impact(self, metric1: str, metric2: str, correlation: float) -> str:
        """Generate business impact for correlation insights."""
        strength = "strong" if abs(correlation) > 0.8 else "moderate"
        direction = "positive" if correlation > 0 else "negative"
        
        return f"This {strength} {direction} relationship can be leveraged for predictive analytics and strategic decision-making. Changes in {metric1} can be used to anticipate changes in {metric2}."
    
    def _determine_priority(self, significance: str, magnitude: float) -> ActionPriority:
        """Determine action priority based on significance and magnitude."""
        if significance == SignificanceLevel.CRITICAL.value or magnitude > 50:
            return ActionPriority.URGENT
        elif significance == SignificanceLevel.HIGH.value or magnitude > 25:
            return ActionPriority.HIGH
        elif significance == SignificanceLevel.MEDIUM.value or magnitude > 10:
            return ActionPriority.MEDIUM
        else:
            return ActionPriority.LOW
    
    def _generate_explanation(self, pattern: Pattern, context: str) -> str:
        """Generate detailed explanation for the pattern."""
        stats = pattern.statistical_measures
        
        explanation = f"This pattern was detected using statistical analysis with {pattern.confidence:.2%} confidence. "
        
        if "r_squared" in stats:
            explanation += f"The trend explains {stats['r_squared']:.2%} of the variance in the data. "
        
        if "p_value" in stats and stats["p_value"] < 0.05:
            explanation += "The pattern is statistically significant. "
        
        explanation += f"The analysis covers the period from {pattern.start_time} to {pattern.end_time}."
        
        return explanation


class RecommendationEngine:
    """Generates actionable recommendations based on insights."""
    
    def __init__(self):
        self.recommendation_templates = {
            InsightType.TREND: {
                "increasing_positive": [
                    "Continue current strategies that are driving this positive trend",
                    "Allocate additional resources to accelerate growth",
                    "Document and replicate successful practices across other areas"
                ],
                "increasing_negative": [
                    "Investigate root causes of this concerning trend",
                    "Implement immediate corrective measures",
                    "Set up monitoring alerts for early detection"
                ],
                "decreasing_positive": [
                    "Investigate factors causing this decline",
                    "Implement improvement initiatives",
                    "Review and adjust current strategies"
                ],
                "decreasing_negative": [
                    "Continue efforts that are reducing this negative metric",
                    "Share best practices with other teams",
                    "Monitor to ensure sustained improvement"
                ]
            }
        }
    
    async def generate_recommendations(self, insights: List[Insight]) -> List[ActionRecommendation]:
        """Generate actionable recommendations for insights."""
        recommendations = []
        
        for insight in insights:
            recs = await self._create_recommendations_for_insight(insight)
            recommendations.extend(recs)
        
        return recommendations
    
    async def _create_recommendations_for_insight(self, insight: Insight) -> List[ActionRecommendation]:
        """Create recommendations for a specific insight."""
        recommendations = []
        
        if insight.type == InsightType.TREND.value:
            recommendations = await self._create_trend_recommendations(insight)
        elif insight.type == InsightType.ANOMALY.value:
            recommendations = await self._create_anomaly_recommendations(insight)
        elif insight.type == InsightType.CORRELATION.value:
            recommendations = await self._create_correlation_recommendations(insight)
        else:
            recommendations = await self._create_generic_recommendations(insight)
        
        return recommendations
    
    async def _create_trend_recommendations(self, insight: Insight) -> List[ActionRecommendation]:
        """Create recommendations for trend insights."""
        recommendations = []
        
        # Determine if trend is positive or negative based on metric type
        metric_name = insight.affected_metrics[0] if insight.affected_metrics else ""
        is_positive_trend = self._is_positive_trend(metric_name, insight.description)
        
        if "increasing" in insight.description.lower():
            trend_type = "increasing_positive" if is_positive_trend else "increasing_negative"
        else:
            trend_type = "decreasing_positive" if is_positive_trend else "decreasing_negative"
        
        templates = self.recommendation_templates[InsightType.TREND].get(trend_type, [])
        
        for i, template in enumerate(templates):
            priority = ActionPriority.HIGH if i == 0 else ActionPriority.MEDIUM
            
            recommendation = ActionRecommendation(
                insight_id=insight.id,
                title=f"Action {i+1}: {template}",
                description=self._expand_recommendation(template, insight),
                action_type="trend_response",
                priority=priority.value,
                estimated_impact=insight.significance,
                effort_required=self._estimate_effort(template),
                timeline=self._suggest_timeline(priority),
                responsible_role=self._suggest_responsible_role(metric_name),
                success_metrics=self._suggest_success_metrics(metric_name, trend_type),
                implementation_steps=self._create_implementation_steps(template, insight)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _create_anomaly_recommendations(self, insight: Insight) -> List[ActionRecommendation]:
        """Create recommendations for anomaly insights."""
        recommendations = []
        
        # Immediate investigation recommendation
        investigation_rec = ActionRecommendation(
            insight_id=insight.id,
            title="Immediate Investigation Required",
            description=f"Investigate the root cause of the anomaly in {insight.affected_metrics[0] if insight.affected_metrics else 'the metric'}. This unusual pattern requires immediate attention to prevent potential business impact.",
            action_type="investigation",
            priority=ActionPriority.URGENT.value,
            estimated_impact=insight.significance,
            effort_required="medium",
            timeline="immediate",
            responsible_role="data_analyst",
            success_metrics=["root_cause_identified", "corrective_action_implemented"],
            implementation_steps=[
                "Review data sources for data quality issues",
                "Check for system changes or external factors",
                "Analyze historical patterns for similar anomalies",
                "Document findings and recommended actions"
            ]
        )
        recommendations.append(investigation_rec)
        
        return recommendations
    
    async def _create_correlation_recommendations(self, insight: Insight) -> List[ActionRecommendation]:
        """Create recommendations for correlation insights."""
        recommendations = []
        
        # Leverage correlation for predictive analytics
        predictive_rec = ActionRecommendation(
            insight_id=insight.id,
            title="Leverage Correlation for Predictive Analytics",
            description=f"Use the discovered correlation between {' and '.join(insight.affected_metrics)} to improve forecasting and decision-making processes.",
            action_type="predictive_modeling",
            priority=ActionPriority.MEDIUM.value,
            estimated_impact=insight.significance * 0.8,
            effort_required="high",
            timeline="2-4 weeks",
            responsible_role="data_scientist",
            success_metrics=["predictive_model_accuracy", "decision_making_improvement"],
            implementation_steps=[
                "Develop predictive model using correlation",
                "Validate model accuracy with historical data",
                "Integrate predictions into dashboard",
                "Train stakeholders on using predictions"
            ]
        )
        recommendations.append(predictive_rec)
        
        return recommendations
    
    async def _create_generic_recommendations(self, insight: Insight) -> List[ActionRecommendation]:
        """Create generic recommendations for other insight types."""
        recommendations = []
        
        generic_rec = ActionRecommendation(
            insight_id=insight.id,
            title="Further Analysis Recommended",
            description=f"Conduct deeper analysis of the pattern detected in {insight.affected_metrics[0] if insight.affected_metrics else 'the data'} to understand its business implications and determine appropriate actions.",
            action_type="analysis",
            priority=ActionPriority.MEDIUM.value,
            estimated_impact=insight.significance,
            effort_required="medium",
            timeline="1-2 weeks",
            responsible_role="business_analyst",
            success_metrics=["pattern_understanding", "action_plan_created"],
            implementation_steps=[
                "Gather additional context and data",
                "Consult with domain experts",
                "Analyze business impact",
                "Develop action plan"
            ]
        )
        recommendations.append(generic_rec)
        
        return recommendations
    
    def _is_positive_trend(self, metric_name: str, description: str) -> bool:
        """Determine if a trend is positive based on metric name and description."""
        metric_lower = metric_name.lower()
        positive_indicators = ["revenue", "profit", "sales", "conversion", "efficiency", "satisfaction", "performance"]
        negative_indicators = ["cost", "error", "latency", "churn", "downtime", "complaints"]
        
        is_positive_metric = any(pos in metric_lower for pos in positive_indicators)
        is_negative_metric = any(neg in metric_lower for neg in negative_indicators)
        
        if is_positive_metric:
            return "increasing" in description.lower()
        elif is_negative_metric:
            return "decreasing" in description.lower()
        else:
            return True  # Default to positive
    
    def _expand_recommendation(self, template: str, insight: Insight) -> str:
        """Expand recommendation template with specific details."""
        metric_name = insight.affected_metrics[0] if insight.affected_metrics else "the metric"
        
        expansions = {
            "Continue current strategies": f"The positive trend in {metric_name} indicates that current strategies are effective. Maintain and strengthen these approaches.",
            "Investigate root causes": f"The concerning trend in {metric_name} requires immediate investigation to identify underlying causes and prevent further deterioration.",
            "Implement immediate corrective measures": f"Take swift action to address the negative trend in {metric_name} before it impacts business performance significantly."
        }
        
        for key, expansion in expansions.items():
            if key in template:
                return expansion
        
        return template
    
    def _estimate_effort(self, recommendation: str) -> str:
        """Estimate effort required for recommendation."""
        if any(word in recommendation.lower() for word in ["immediate", "urgent", "swift"]):
            return "high"
        elif any(word in recommendation.lower() for word in ["investigate", "analyze", "review"]):
            return "medium"
        else:
            return "low"
    
    def _suggest_timeline(self, priority: ActionPriority) -> str:
        """Suggest timeline based on priority."""
        if priority == ActionPriority.URGENT:
            return "immediate"
        elif priority == ActionPriority.HIGH:
            return "1-3 days"
        elif priority == ActionPriority.MEDIUM:
            return "1-2 weeks"
        else:
            return "2-4 weeks"
    
    def _suggest_responsible_role(self, metric_name: str) -> str:
        """Suggest responsible role based on metric type."""
        metric_lower = metric_name.lower()
        
        if any(word in metric_lower for word in ["revenue", "sales", "profit"]):
            return "sales_manager"
        elif any(word in metric_lower for word in ["cost", "budget", "financial"]):
            return "finance_manager"
        elif any(word in metric_lower for word in ["performance", "latency", "error"]):
            return "engineering_manager"
        elif any(word in metric_lower for word in ["satisfaction", "churn", "retention"]):
            return "customer_success_manager"
        else:
            return "department_head"
    
    def _suggest_success_metrics(self, metric_name: str, trend_type: str) -> List[str]:
        """Suggest success metrics for measuring recommendation effectiveness."""
        base_metrics = [f"{metric_name}_improvement", "action_completion_rate"]
        
        if "increasing_negative" in trend_type or "decreasing_positive" in trend_type:
            base_metrics.append("trend_reversal")
        else:
            base_metrics.append("trend_acceleration")
        
        return base_metrics
    
    def _create_implementation_steps(self, recommendation: str, insight: Insight) -> List[str]:
        """Create implementation steps for recommendation."""
        if "investigate" in recommendation.lower():
            return [
                "Gather relevant data and context",
                "Identify potential root causes",
                "Analyze contributing factors",
                "Develop corrective action plan",
                "Implement and monitor results"
            ]
        elif "continue" in recommendation.lower():
            return [
                "Document current successful practices",
                "Identify key success factors",
                "Allocate additional resources",
                "Monitor continued performance",
                "Scale successful approaches"
            ]
        else:
            return [
                "Define specific objectives",
                "Develop implementation plan",
                "Allocate necessary resources",
                "Execute planned actions",
                "Monitor and adjust as needed"
            ]


class InsightGenerator(BaseEngine):
    """Main AI Insight Generation Engine."""
    
    def __init__(self):
        super().__init__(
            engine_id="insight_generator",
            name="AI Insight Generation Engine",
            capabilities=[
                EngineCapability.DATA_ANALYSIS,
                EngineCapability.COGNITIVE_REASONING,
                EngineCapability.REPORT_GENERATION
            ]
        )
        
        self.pattern_detector = PatternDetector()
        self.insight_engine = InsightEngine()
        self.recommendation_engine = RecommendationEngine()
        self.business_contexts = {}
    
    async def initialize(self) -> None:
        """Initialize the insight generation engine."""
        logger.info("Initializing AI Insight Generation Engine")
        
        # Load business contexts and thresholds
        await self._load_business_contexts()
        
        logger.info("AI Insight Generation Engine initialized successfully")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process analytics data and generate insights."""
        if not isinstance(input_data, AnalyticsData):
            raise ValueError("Input data must be AnalyticsData instance")
        
        parameters = parameters or {}
        
        try:
            # Step 1: Detect patterns
            logger.info("Detecting patterns in analytics data")
            trend_patterns = await self.pattern_detector.detect_trends(input_data.metrics)
            correlation_patterns = await self.pattern_detector.detect_correlations(input_data.metrics)
            anomalies = await self.pattern_detector.detect_anomalies(input_data.metrics)
            
            all_patterns = trend_patterns + correlation_patterns
            
            # Step 2: Generate insights
            logger.info(f"Generating insights from {len(all_patterns)} patterns")
            insights = await self.insight_engine.generate_insights(all_patterns)
            
            # Step 3: Generate recommendations
            logger.info(f"Generating recommendations for {len(insights)} insights")
            recommendations = await self.recommendation_engine.generate_recommendations(insights)
            
            # Step 4: Rank and prioritize insights
            ranked_insights = await self._rank_insights(insights)
            
            # Step 5: Generate summary
            summary = await self._generate_summary(ranked_insights, anomalies)
            
            return {
                "patterns": [self._pattern_to_dict(p) for p in all_patterns],
                "insights": [self._insight_to_dict(i) for i in ranked_insights],
                "recommendations": [self._recommendation_to_dict(r) for r in recommendations],
                "anomalies": [self._anomaly_to_dict(a) for a in anomalies],
                "summary": summary,
                "metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "metrics_count": len(input_data.metrics),
                    "patterns_found": len(all_patterns),
                    "insights_generated": len(insights),
                    "recommendations_created": len(recommendations),
                    "anomalies_detected": len(anomalies)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing insights: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up AI Insight Generation Engine")
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status.value,
            "healthy": self.status == EngineStatus.READY,
            "capabilities": [cap.value for cap in self.capabilities],
            "business_contexts_loaded": len(self.business_contexts),
            "last_processing": self.last_used.isoformat() if self.last_used else None
        }
    
    async def analyze_data_patterns(self, data: AnalyticsData) -> List[Pattern]:
        """Analyze data patterns (public interface)."""
        trend_patterns = await self.pattern_detector.detect_trends(data.metrics)
        correlation_patterns = await self.pattern_detector.detect_correlations(data.metrics)
        return trend_patterns + correlation_patterns
    
    async def generate_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate insights from patterns (public interface)."""
        return await self.insight_engine.generate_insights(patterns)
    
    async def explain_anomaly(self, anomaly: Anomaly, context: BusinessContext) -> str:
        """Explain anomaly with business context (public interface)."""
        explanation = f"An anomaly was detected in {anomaly.metric_name} with a deviation score of {anomaly.deviation_score:.2f}. "
        
        if context:
            explanation += f"In the context of {context.name}, this could indicate {self._get_contextual_explanation(anomaly, context)}."
        
        if anomaly.root_causes:
            explanation += f" Potential causes include: {', '.join(anomaly.root_causes)}."
        
        return explanation
    
    async def suggest_actions(self, insights: List[Insight]) -> List[ActionRecommendation]:
        """Suggest actions based on insights (public interface)."""
        return await self.recommendation_engine.generate_recommendations(insights)
    
    async def _load_business_contexts(self) -> None:
        """Load business contexts and thresholds."""
        # This would typically load from database
        # For now, we'll use default contexts
        self.business_contexts = {
            "default": {
                "thresholds": {
                    "revenue": {"critical": 0.2, "high": 0.15, "medium": 0.1},
                    "cost": {"critical": 0.25, "high": 0.2, "medium": 0.15},
                    "performance": {"critical": 0.3, "high": 0.2, "medium": 0.1}
                },
                "kpis": ["revenue", "cost", "efficiency", "satisfaction"],
                "benchmarks": {}
            }
        }
    
    async def _rank_insights(self, insights: List[Insight]) -> List[Insight]:
        """Rank insights by business impact and priority."""
        def insight_score(insight: Insight) -> float:
            priority_weights = {
                ActionPriority.URGENT.value: 1.0,
                ActionPriority.HIGH.value: 0.8,
                ActionPriority.MEDIUM.value: 0.6,
                ActionPriority.LOW.value: 0.4
            }
            
            priority_weight = priority_weights.get(insight.priority, 0.5)
            return insight.significance * insight.confidence * priority_weight
        
        return sorted(insights, key=insight_score, reverse=True)
    
    async def _generate_summary(self, insights: List[Insight], anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Generate executive summary of insights."""
        if not insights and not anomalies:
            return {
                "overview": "No significant patterns or anomalies detected in the current data.",
                "key_findings": [],
                "urgent_actions": [],
                "recommendations_count": 0
            }
        
        # Count insights by type and priority
        insight_counts = {}
        urgent_actions = []
        
        for insight in insights:
            insight_counts[insight.type] = insight_counts.get(insight.type, 0) + 1
            if insight.priority == ActionPriority.URGENT.value:
                urgent_actions.append(insight.title)
        
        # Generate overview
        total_insights = len(insights)
        total_anomalies = len(anomalies)
        
        overview = f"Analysis identified {total_insights} insights"
        if total_anomalies > 0:
            overview += f" and {total_anomalies} anomalies"
        overview += " in your business metrics."
        
        # Key findings
        key_findings = []
        if insights:
            top_insight = insights[0]
            key_findings.append(f"Most significant finding: {top_insight.title}")
        
        if anomalies:
            critical_anomalies = [a for a in anomalies if a.severity == SignificanceLevel.CRITICAL.value]
            if critical_anomalies:
                key_findings.append(f"{len(critical_anomalies)} critical anomalies require immediate attention")
        
        return {
            "overview": overview,
            "key_findings": key_findings,
            "urgent_actions": urgent_actions,
            "insight_breakdown": insight_counts,
            "recommendations_count": sum(len(i.recommendations) for i in insights if hasattr(i, 'recommendations')),
            "anomalies_count": total_anomalies
        }
    
    def _get_contextual_explanation(self, anomaly: Anomaly, context: BusinessContext) -> str:
        """Get contextual explanation for anomaly."""
        if context.context_type == "industry":
            return f"unusual behavior for the {context.name} industry"
        elif context.context_type == "department":
            return f"unexpected performance in the {context.name} department"
        else:
            return "deviation from normal business patterns"
    
    def _pattern_to_dict(self, pattern: Pattern) -> Dict[str, Any]:
        """Convert Pattern object to dictionary."""
        return {
            "id": pattern.id,
            "type": pattern.type,
            "metric_name": pattern.metric_name,
            "description": pattern.description,
            "confidence": pattern.confidence,
            "significance": pattern.significance,
            "start_time": pattern.start_time.isoformat() if pattern.start_time else None,
            "end_time": pattern.end_time.isoformat() if pattern.end_time else None,
            "statistical_measures": pattern.statistical_measures or {}
        }
    
    def _insight_to_dict(self, insight: Insight) -> Dict[str, Any]:
        """Convert Insight object to dictionary."""
        return {
            "id": insight.id,
            "type": insight.type,
            "title": insight.title,
            "description": insight.description,
            "explanation": insight.explanation,
            "business_impact": insight.business_impact,
            "confidence": insight.confidence,
            "significance": insight.significance,
            "priority": insight.priority,
            "tags": insight.tags,
            "affected_metrics": insight.affected_metrics,
            "created_at": insight.created_at.isoformat() if insight.created_at else datetime.utcnow().isoformat()
        }
    
    def _recommendation_to_dict(self, recommendation: ActionRecommendation) -> Dict[str, Any]:
        """Convert ActionRecommendation object to dictionary."""
        return {
            "id": recommendation.id,
            "title": recommendation.title,
            "description": recommendation.description,
            "action_type": recommendation.action_type,
            "priority": recommendation.priority,
            "estimated_impact": recommendation.estimated_impact,
            "effort_required": recommendation.effort_required,
            "timeline": recommendation.timeline,
            "responsible_role": recommendation.responsible_role,
            "success_metrics": recommendation.success_metrics,
            "implementation_steps": recommendation.implementation_steps
        }
    
    def _anomaly_to_dict(self, anomaly: Anomaly) -> Dict[str, Any]:
        """Convert Anomaly object to dictionary."""
        return {
            "id": anomaly.id,
            "metric_name": anomaly.metric_name,
            "metric_value": anomaly.metric_value,
            "expected_value": anomaly.expected_value,
            "deviation_score": anomaly.deviation_score,
            "anomaly_type": anomaly.anomaly_type,
            "severity": anomaly.severity,
            "detected_at": anomaly.detected_at.isoformat() if anomaly.detected_at else datetime.utcnow().isoformat(),
            "context": anomaly.context or {}
        }