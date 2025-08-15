"""
Team-wide analytics dashboard and insights system.
Provides comprehensive analytics visualization and team insights.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

from ..models.analytics_models import (
    PromptMetrics, UsageAnalytics, AnalyticsReport, AlertRule
)
from ..models.database import get_db_session
from .prompt_analytics import PromptPerformanceTracker, TrendAnalyzer, PatternRecognizer

logger = logging.getLogger(__name__)

class TeamAnalyticsDashboard:
    """Comprehensive team analytics dashboard."""
    
    def __init__(self):
        self.performance_tracker = PromptPerformanceTracker()
        self.trend_analyzer = TrendAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
    
    async def get_team_dashboard_data(
        self,
        team_id: str,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data for a team."""
        try:
            if not date_range:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)
                date_range = (start_date, end_date)
            
            dashboard_data = {
                "team_id": team_id,
                "date_range": {
                    "start": date_range[0].isoformat(),
                    "end": date_range[1].isoformat()
                },
                "overview": await self._get_team_overview(team_id, date_range),
                "performance_metrics": await self._get_team_performance_metrics(team_id, date_range),
                "usage_analytics": await self._get_team_usage_analytics(team_id, date_range),
                "top_prompts": await self._get_top_performing_prompts(team_id, date_range),
                "improvement_opportunities": await self._get_improvement_opportunities(team_id, date_range),
                "trends": await self._get_team_trends(team_id, date_range),
                "alerts": await self._get_active_alerts(team_id),
                "recommendations": await self._generate_team_recommendations(team_id, date_range)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting team dashboard data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_team_overview(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get high-level team overview metrics."""
        try:
            with get_db_session() as db:
                # Get all metrics for team in date range
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.team_id == team_id,
                        PromptMetrics.created_at >= date_range[0],
                        PromptMetrics.created_at <= date_range[1]
                    )
                ).all()
                
                # Get unique prompts and users
                unique_prompts = set(m.prompt_id for m in metrics)
                unique_users = set(m.user_id for m in metrics if m.user_id)
                
                # Calculate total costs
                total_cost = sum(m.cost_per_request for m in metrics if m.cost_per_request)
                
                # Calculate average performance scores
                accuracy_scores = [m.accuracy_score for m in metrics if m.accuracy_score is not None]
                avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else None
                
                satisfaction_scores = [m.user_satisfaction for m in metrics if m.user_satisfaction is not None]
                avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else None
                
                return {
                    "total_requests": len(metrics),
                    "unique_prompts": len(unique_prompts),
                    "active_users": len(unique_users),
                    "total_cost": round(total_cost, 2) if total_cost else 0,
                    "avg_accuracy": round(avg_accuracy, 3) if avg_accuracy else None,
                    "avg_satisfaction": round(avg_satisfaction, 2) if avg_satisfaction else None,
                    "cost_per_request": round(total_cost / len(metrics), 4) if total_cost and metrics else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting team overview: {str(e)}")
            return {}
    
    async def _get_team_performance_metrics(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get detailed performance metrics for team."""
        try:
            with get_db_session() as db:
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.team_id == team_id,
                        PromptMetrics.created_at >= date_range[0],
                        PromptMetrics.created_at <= date_range[1]
                    )
                ).all()
                
                if not metrics:
                    return {}
                
                performance_data = {}
                
                # Calculate metrics for each performance dimension
                metric_names = [
                    'accuracy_score', 'relevance_score', 'efficiency_score',
                    'user_satisfaction', 'response_time_ms', 'token_usage'
                ]
                
                for metric_name in metric_names:
                    values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
                    
                    if values:
                        performance_data[metric_name] = {
                            "current_avg": round(np.mean(values), 3),
                            "median": round(np.median(values), 3),
                            "std_dev": round(np.std(values), 3),
                            "min": round(min(values), 3),
                            "max": round(max(values), 3),
                            "percentile_25": round(np.percentile(values, 25), 3),
                            "percentile_75": round(np.percentile(values, 75), 3),
                            "trend": self._calculate_metric_trend(values),
                            "distribution": self._calculate_distribution(values)
                        }
                
                return performance_data
                
        except Exception as e:
            logger.error(f"Error getting team performance metrics: {str(e)}")
            return {}
    
    async def _get_team_usage_analytics(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get usage analytics for team."""
        try:
            with get_db_session() as db:
                # Get usage analytics for team prompts
                team_prompts = db.query(PromptMetrics.prompt_id).filter(
                    PromptMetrics.team_id == team_id
                ).distinct().all()
                
                prompt_ids = [p[0] for p in team_prompts]
                
                analytics = db.query(UsageAnalytics).filter(
                    and_(
                        UsageAnalytics.prompt_id.in_(prompt_ids),
                        UsageAnalytics.analysis_period_start >= date_range[0],
                        UsageAnalytics.analysis_period_end <= date_range[1]
                    )
                ).all()
                
                if not analytics:
                    return {}
                
                # Aggregate usage data
                total_requests = sum(a.total_requests for a in analytics)
                successful_requests = sum(a.successful_requests for a in analytics)
                failed_requests = sum(a.failed_requests for a in analytics)
                
                # Calculate success rate
                success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
                
                # Aggregate hourly patterns
                combined_hourly = defaultdict(int)
                for a in analytics:
                    if a.hourly_patterns:
                        for hour, count in a.hourly_patterns.items():
                            combined_hourly[int(hour)] += count
                
                # Find peak usage times
                peak_hours = sorted(combined_hourly.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Aggregate daily usage
                combined_daily = defaultdict(int)
                for a in analytics:
                    if a.daily_usage:
                        for date, count in a.daily_usage.items():
                            combined_daily[date] += count
                
                return {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "success_rate": round(success_rate, 2),
                    "peak_usage_hours": [{"hour": h, "requests": c} for h, c in peak_hours],
                    "hourly_distribution": dict(combined_hourly),
                    "daily_usage": dict(combined_daily),
                    "usage_trend": self._calculate_usage_trend(list(combined_daily.values()))
                }
                
        except Exception as e:
            logger.error(f"Error getting team usage analytics: {str(e)}")
            return {}
    
    async def _get_top_performing_prompts(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top performing prompts for team."""
        try:
            with get_db_session() as db:
                # Get all prompts with their performance metrics
                prompt_performance = {}
                
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.team_id == team_id,
                        PromptMetrics.created_at >= date_range[0],
                        PromptMetrics.created_at <= date_range[1]
                    )
                ).all()
                
                # Group by prompt
                for metric in metrics:
                    if metric.prompt_id not in prompt_performance:
                        prompt_performance[metric.prompt_id] = []
                    prompt_performance[metric.prompt_id].append(metric)
                
                # Calculate performance scores
                prompt_scores = []
                for prompt_id, prompt_metrics in prompt_performance.items():
                    score = self._calculate_overall_performance_score(prompt_metrics)
                    usage_count = len(prompt_metrics)
                    
                    # Get latest metrics for additional info
                    latest_metric = max(prompt_metrics, key=lambda x: x.created_at)
                    
                    prompt_scores.append({
                        "prompt_id": prompt_id,
                        "performance_score": score,
                        "usage_count": usage_count,
                        "latest_accuracy": latest_metric.accuracy_score,
                        "latest_satisfaction": latest_metric.user_satisfaction,
                        "avg_response_time": np.mean([m.response_time_ms for m in prompt_metrics if m.response_time_ms]),
                        "total_cost": sum(m.cost_per_request for m in prompt_metrics if m.cost_per_request)
                    })
                
                # Sort by performance score and return top performers
                top_prompts = sorted(prompt_scores, key=lambda x: x["performance_score"], reverse=True)[:limit]
                
                return top_prompts
                
        except Exception as e:
            logger.error(f"Error getting top performing prompts: {str(e)}")
            return []
    
    async def _get_improvement_opportunities(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get prompts with improvement opportunities."""
        try:
            with get_db_session() as db:
                # Get prompts with declining performance or low scores
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.team_id == team_id,
                        PromptMetrics.created_at >= date_range[0],
                        PromptMetrics.created_at <= date_range[1]
                    )
                ).all()
                
                # Group by prompt and analyze
                prompt_analysis = {}
                for metric in metrics:
                    if metric.prompt_id not in prompt_analysis:
                        prompt_analysis[metric.prompt_id] = []
                    prompt_analysis[metric.prompt_id].append(metric)
                
                improvement_opportunities = []
                
                for prompt_id, prompt_metrics in prompt_analysis.items():
                    # Calculate improvement potential
                    issues = self._identify_performance_issues(prompt_metrics)
                    
                    if issues:
                        opportunity = {
                            "prompt_id": prompt_id,
                            "usage_count": len(prompt_metrics),
                            "issues": issues,
                            "improvement_potential": self._calculate_improvement_potential(issues),
                            "recommended_actions": self._generate_improvement_actions(issues)
                        }
                        improvement_opportunities.append(opportunity)
                
                # Sort by improvement potential
                opportunities = sorted(
                    improvement_opportunities,
                    key=lambda x: x["improvement_potential"],
                    reverse=True
                )[:limit]
                
                return opportunities
                
        except Exception as e:
            logger.error(f"Error getting improvement opportunities: {str(e)}")
            return []
    
    async def _get_team_trends(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get trend analysis for team."""
        try:
            trends = {}
            
            # Get team prompts
            with get_db_session() as db:
                team_prompts = db.query(PromptMetrics.prompt_id).filter(
                    PromptMetrics.team_id == team_id
                ).distinct().all()
                
                prompt_ids = [p[0] for p in team_prompts]
            
            # Analyze trends for key metrics
            key_metrics = ['accuracy_score', 'user_satisfaction', 'response_time_ms']
            
            for metric_name in key_metrics:
                # Get aggregated trend for all team prompts
                trend_data = await self._get_aggregated_trend(prompt_ids, metric_name, date_range)
                if trend_data:
                    trends[metric_name] = trend_data
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting team trends: {str(e)}")
            return {}
    
    async def _get_active_alerts(self, team_id: str) -> List[Dict[str, Any]]:
        """Get active alerts for team."""
        try:
            with get_db_session() as db:
                alerts = db.query(AlertRule).filter(
                    and_(
                        AlertRule.active == True,
                        AlertRule.team_ids.contains([team_id])
                    )
                ).all()
                
                active_alerts = []
                for alert in alerts:
                    # Check if alert should be triggered
                    should_trigger = await self._check_alert_condition(alert)
                    
                    if should_trigger:
                        active_alerts.append({
                            "id": alert.id,
                            "name": alert.name,
                            "severity": alert.severity,
                            "description": alert.description,
                            "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
                            "trigger_count": alert.trigger_count
                        })
                
                return active_alerts
                
        except Exception as e:
            logger.error(f"Error getting active alerts: {str(e)}")
            return []
    
    async def _generate_team_recommendations(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered recommendations for team."""
        try:
            recommendations = []
            
            # Get team performance data
            overview = await self._get_team_overview(team_id, date_range)
            performance = await self._get_team_performance_metrics(team_id, date_range)
            usage = await self._get_team_usage_analytics(team_id, date_range)
            
            # Generate recommendations based on data
            if overview.get("avg_accuracy", 0) < 0.7:
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "title": "Improve Prompt Accuracy",
                    "description": "Team average accuracy is below 70%. Consider prompt optimization.",
                    "actions": [
                        "Review low-performing prompts",
                        "Implement A/B testing",
                        "Gather user feedback"
                    ]
                })
            
            if usage.get("success_rate", 0) < 95:
                recommendations.append({
                    "type": "reliability",
                    "priority": "medium",
                    "title": "Improve Success Rate",
                    "description": f"Success rate is {usage.get('success_rate', 0)}%. Investigate failures.",
                    "actions": [
                        "Analyze error patterns",
                        "Implement better error handling",
                        "Review prompt complexity"
                    ]
                })
            
            if overview.get("cost_per_request", 0) > 0.01:  # Threshold for high cost
                recommendations.append({
                    "type": "cost",
                    "priority": "medium",
                    "title": "Optimize Costs",
                    "description": "Cost per request is high. Consider optimization.",
                    "actions": [
                        "Review token usage",
                        "Optimize prompt length",
                        "Consider model alternatives"
                    ]
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating team recommendations: {str(e)}")
            return []
    
    def _calculate_metric_trend(self, values: List[float]) -> str:
        """Calculate trend for a metric."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend calculation
        recent_avg = np.mean(values[-len(values)//3:])
        older_avg = np.mean(values[:len(values)//3])
        
        if abs(recent_avg - older_avg) < 0.05:  # 5% threshold
            return "stable"
        elif recent_avg > older_avg:
            return "improving"
        else:
            return "declining"
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate distribution of values."""
        if not values:
            return {}
        
        # Create bins for distribution
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return {"single_value": len(values)}
        
        bins = np.linspace(min_val, max_val, 6)  # 5 bins
        hist, _ = np.histogram(values, bins=bins)
        
        return {f"bin_{i}": int(count) for i, count in enumerate(hist)}
    
    def _calculate_usage_trend(self, daily_values: List[int]) -> str:
        """Calculate usage trend from daily values."""
        if len(daily_values) < 7:
            return "insufficient_data"
        
        # Compare recent week to previous week
        if len(daily_values) >= 14:
            recent_week = sum(daily_values[-7:])
            previous_week = sum(daily_values[-14:-7])
            
            if abs(recent_week - previous_week) < previous_week * 0.1:  # 10% threshold
                return "stable"
            elif recent_week > previous_week:
                return "increasing"
            else:
                return "decreasing"
        
        return "stable"
    
    def _calculate_overall_performance_score(self, metrics: List[PromptMetrics]) -> float:
        """Calculate overall performance score for a prompt."""
        scores = []
        weights = {
            'accuracy_score': 0.3,
            'relevance_score': 0.25,
            'efficiency_score': 0.2,
            'user_satisfaction': 0.25
        }
        
        for metric_name, weight in weights.items():
            values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
            if values:
                avg_value = np.mean(values)
                scores.append(avg_value * weight)
        
        return sum(scores) if scores else 0.0
    
    def _identify_performance_issues(self, metrics: List[PromptMetrics]) -> List[Dict[str, Any]]:
        """Identify performance issues in prompt metrics."""
        issues = []
        
        # Check accuracy
        accuracy_scores = [m.accuracy_score for m in metrics if m.accuracy_score is not None]
        if accuracy_scores and np.mean(accuracy_scores) < 0.7:
            issues.append({
                "type": "low_accuracy",
                "severity": "high",
                "description": f"Average accuracy is {np.mean(accuracy_scores):.2f}"
            })
        
        # Check response time
        response_times = [m.response_time_ms for m in metrics if m.response_time_ms is not None]
        if response_times and np.mean(response_times) > 2000:  # 2 seconds
            issues.append({
                "type": "slow_response",
                "severity": "medium",
                "description": f"Average response time is {np.mean(response_times):.0f}ms"
            })
        
        # Check user satisfaction
        satisfaction_scores = [m.user_satisfaction for m in metrics if m.user_satisfaction is not None]
        if satisfaction_scores and np.mean(satisfaction_scores) < 3.5:
            issues.append({
                "type": "low_satisfaction",
                "severity": "high",
                "description": f"Average satisfaction is {np.mean(satisfaction_scores):.1f}/5"
            })
        
        return issues
    
    def _calculate_improvement_potential(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate improvement potential based on issues."""
        if not issues:
            return 0.0
        
        severity_weights = {"high": 1.0, "medium": 0.6, "low": 0.3}
        total_weight = sum(severity_weights.get(issue["severity"], 0.3) for issue in issues)
        
        return min(total_weight, 1.0)
    
    def _generate_improvement_actions(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement actions based on issues."""
        actions = []
        
        for issue in issues:
            if issue["type"] == "low_accuracy":
                actions.extend([
                    "Review and refine prompt structure",
                    "Add more specific examples",
                    "Test with different models"
                ])
            elif issue["type"] == "slow_response":
                actions.extend([
                    "Optimize prompt length",
                    "Reduce complexity",
                    "Consider faster model variants"
                ])
            elif issue["type"] == "low_satisfaction":
                actions.extend([
                    "Gather detailed user feedback",
                    "Improve output formatting",
                    "Add context-specific instructions"
                ])
        
        return list(set(actions))  # Remove duplicates
    
    async def _get_aggregated_trend(
        self,
        prompt_ids: List[str],
        metric_name: str,
        date_range: Tuple[datetime, datetime]
    ) -> Optional[Dict[str, Any]]:
        """Get aggregated trend for multiple prompts."""
        try:
            with get_db_session() as db:
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.prompt_id.in_(prompt_ids),
                        PromptMetrics.created_at >= date_range[0],
                        PromptMetrics.created_at <= date_range[1]
                    )
                ).order_by(PromptMetrics.created_at).all()
                
                values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
                
                if len(values) < 3:
                    return None
                
                # Calculate trend
                trend_direction = self._calculate_metric_trend(values)
                
                # Calculate daily averages for visualization
                daily_data = defaultdict(list)
                for metric in metrics:
                    if getattr(metric, metric_name) is not None:
                        date_key = metric.created_at.date().isoformat()
                        daily_data[date_key].append(getattr(metric, metric_name))
                
                daily_averages = {
                    date: np.mean(values) for date, values in daily_data.items()
                }
                
                return {
                    "trend_direction": trend_direction,
                    "current_value": values[-1] if values else None,
                    "average_value": np.mean(values),
                    "daily_data": daily_averages,
                    "data_points": len(values)
                }
                
        except Exception as e:
            logger.error(f"Error getting aggregated trend: {str(e)}")
            return None
    
    async def _check_alert_condition(self, alert: AlertRule) -> bool:
        """Check if alert condition is met."""
        try:
            # This is a simplified implementation
            # In practice, you'd implement specific logic for each alert type
            return False  # Placeholder
            
        except Exception as e:
            logger.error(f"Error checking alert condition: {str(e)}")
            return False

class InsightsGenerator:
    """Generates actionable insights from analytics data."""
    
    def __init__(self):
        self.insight_templates = {
            "performance_decline": "Performance for {prompt_id} has declined by {percentage}% over the last {days} days",
            "usage_spike": "Usage for {prompt_id} increased by {percentage}% - consider scaling resources",
            "cost_optimization": "Optimizing {prompt_id} could save ${amount} per month",
            "user_satisfaction": "User satisfaction for {prompt_id} is {rating}/5 - investigate feedback"
        }
    
    async def generate_insights(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights for a team."""
        insights = []
        
        try:
            # Get team data
            dashboard = TeamAnalyticsDashboard()
            team_data = await dashboard.get_team_dashboard_data(team_id, date_range)
            
            # Generate insights based on patterns
            insights.extend(await self._generate_performance_insights(team_data))
            insights.extend(await self._generate_usage_insights(team_data))
            insights.extend(await self._generate_cost_insights(team_data))
            insights.extend(await self._generate_satisfaction_insights(team_data))
            
            # Sort by priority
            insights.sort(key=lambda x: self._get_priority_score(x), reverse=True)
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    async def _generate_performance_insights(self, team_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance-related insights."""
        insights = []
        
        performance_metrics = team_data.get("performance_metrics", {})
        
        for metric_name, metric_data in performance_metrics.items():
            if metric_data.get("trend") == "declining":
                insights.append({
                    "type": "performance_decline",
                    "priority": "high",
                    "title": f"{metric_name.replace('_', ' ').title()} Declining",
                    "description": f"Team {metric_name} has been declining",
                    "impact": "high",
                    "actionable": True,
                    "data": metric_data
                })
        
        return insights
    
    async def _generate_usage_insights(self, team_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate usage-related insights."""
        insights = []
        
        usage_analytics = team_data.get("usage_analytics", {})
        
        if usage_analytics.get("usage_trend") == "increasing":
            insights.append({
                "type": "usage_spike",
                "priority": "medium",
                "title": "Usage Increasing",
                "description": "Team prompt usage is trending upward",
                "impact": "medium",
                "actionable": True,
                "data": usage_analytics
            })
        
        return insights
    
    async def _generate_cost_insights(self, team_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cost-related insights."""
        insights = []
        
        overview = team_data.get("overview", {})
        
        if overview.get("cost_per_request", 0) > 0.005:  # 0.5 cents threshold
            insights.append({
                "type": "cost_optimization",
                "priority": "medium",
                "title": "High Cost Per Request",
                "description": f"Cost per request is ${overview.get('cost_per_request', 0):.4f}",
                "impact": "medium",
                "actionable": True,
                "data": overview
            })
        
        return insights
    
    async def _generate_satisfaction_insights(self, team_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate satisfaction-related insights."""
        insights = []
        
        overview = team_data.get("overview", {})
        
        if overview.get("avg_satisfaction") and overview.get("avg_satisfaction") < 3.5:
            insights.append({
                "type": "user_satisfaction",
                "priority": "high",
                "title": "Low User Satisfaction",
                "description": f"Average satisfaction is {overview.get('avg_satisfaction'):.1f}/5",
                "impact": "high",
                "actionable": True,
                "data": overview
            })
        
        return insights
    
    def _get_priority_score(self, insight: Dict[str, Any]) -> int:
        """Calculate priority score for insight."""
        priority_scores = {"high": 3, "medium": 2, "low": 1}
        impact_scores = {"high": 3, "medium": 2, "low": 1}
        
        priority_score = priority_scores.get(insight.get("priority", "low"), 1)
        impact_score = impact_scores.get(insight.get("impact", "low"), 1)
        actionable_score = 2 if insight.get("actionable", False) else 1
        
        return priority_score * impact_score * actionable_score