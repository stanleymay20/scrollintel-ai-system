"""
Analytics dashboard system for prompt management.
Provides comprehensive dashboards and insights for teams and organizations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
import logging

from ..core.config import get_settings
from ..core.logging_config import get_logger
from .prompt_analytics import prompt_performance_tracker

settings = get_settings()
logger = get_logger(__name__)

@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    widget_type: str  # 'chart', 'metric', 'table', 'alert'
    title: str
    data_source: str
    configuration: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 300  # seconds

@dataclass
class TeamDashboard:
    """Team dashboard configuration."""
    dashboard_id: str
    team_id: str
    name: str
    description: Optional[str]
    widgets: List[DashboardWidget]
    layout: Dict[str, Any]
    created_by: str
    created_at: datetime
    updated_at: datetime

class TeamAnalyticsDashboard:
    """Comprehensive team analytics dashboard system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.dashboards: Dict[str, TeamDashboard] = {}
        self.widget_data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def get_team_dashboard_data(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data for a team."""
        try:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            # Get team analytics
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            if 'error' in team_analytics:
                return team_analytics
            
            # Get additional dashboard metrics
            dashboard_data = {
                'team_id': team_id,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'summary_metrics': await self._get_summary_metrics(team_id, date_range),
                'performance_trends': await self._get_performance_trends(team_id, date_range),
                'usage_patterns': await self._get_usage_patterns(team_id, date_range),
                'cost_analysis': await self._get_cost_analysis(team_id, date_range),
                'quality_metrics': await self._get_quality_metrics(team_id, date_range),
                'user_activity': await self._get_user_activity(team_id, date_range),
                'alerts': await self._get_active_alerts(team_id),
                'recommendations': await self._generate_recommendations(team_id, date_range),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Merge with team analytics
            dashboard_data.update(team_analytics)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error getting team dashboard data: {e}")
            return {'error': str(e)}
    
    async def _get_summary_metrics(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get high-level summary metrics."""
        try:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            # Get current period data
            current_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            # Get previous period for comparison
            prev_start = start_date - timedelta(days=days)
            prev_end = start_date
            prev_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            current_summary = current_analytics.get('summary', {})
            prev_summary = prev_analytics.get('summary', {})
            
            # Calculate changes
            def calculate_change(current, previous):
                if previous and previous != 0:
                    return ((current - previous) / previous) * 100
                return 0
            
            metrics = {
                'total_requests': {
                    'value': current_summary.get('total_requests', 0),
                    'change': calculate_change(
                        current_summary.get('total_requests', 0),
                        prev_summary.get('total_requests', 0)
                    )
                },
                'success_rate': {
                    'value': current_summary.get('success_rate', 0),
                    'change': calculate_change(
                        current_summary.get('success_rate', 0),
                        prev_summary.get('success_rate', 0)
                    )
                },
                'avg_accuracy': {
                    'value': current_summary.get('avg_accuracy_score'),
                    'change': calculate_change(
                        current_summary.get('avg_accuracy_score', 0) or 0,
                        prev_summary.get('avg_accuracy_score', 0) or 0
                    )
                },
                'avg_response_time': {
                    'value': current_summary.get('avg_response_time_ms'),
                    'change': calculate_change(
                        current_summary.get('avg_response_time_ms', 0) or 0,
                        prev_summary.get('avg_response_time_ms', 0) or 0
                    )
                },
                'total_cost': {
                    'value': current_summary.get('total_cost'),
                    'change': calculate_change(
                        current_summary.get('total_cost', 0) or 0,
                        prev_summary.get('total_cost', 0) or 0
                    )
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting summary metrics: {e}")
            return {}
    
    async def _get_performance_trends(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get performance trend data for charts."""
        try:
            # This would typically query historical data from database
            # For now, generate sample trend data
            
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            # Generate daily trend data
            daily_trends = []
            for i in range(days):
                date = start_date + timedelta(days=i)
                daily_trends.append({
                    'date': date.date().isoformat(),
                    'requests': 100 + (i * 5) + (i % 7) * 20,  # Sample data
                    'success_rate': 95 + (i % 10),
                    'avg_accuracy': 0.85 + (i % 20) * 0.01,
                    'avg_response_time': 1500 + (i % 15) * 100
                })
            
            # Calculate trend directions
            if len(daily_trends) >= 2:
                first_half = daily_trends[:len(daily_trends)//2]
                second_half = daily_trends[len(daily_trends)//2:]
                
                trends = {}
                for metric in ['requests', 'success_rate', 'avg_accuracy', 'avg_response_time']:
                    first_avg = statistics.mean([d[metric] for d in first_half])
                    second_avg = statistics.mean([d[metric] for d in second_half])
                    
                    if second_avg > first_avg * 1.05:
                        trends[metric] = 'increasing'
                    elif second_avg < first_avg * 0.95:
                        trends[metric] = 'decreasing'
                    else:
                        trends[metric] = 'stable'
            else:
                trends = {}
            
            return {
                'daily_trends': daily_trends,
                'trend_directions': trends,
                'period_days': days
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {e}")
            return {}
    
    async def _get_usage_patterns(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get usage pattern analysis."""
        try:
            # Get team analytics for usage patterns
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            daily_usage = team_analytics.get('daily_usage', {})
            
            # Analyze patterns
            patterns = {
                'daily_usage': daily_usage,
                'peak_day': None,
                'avg_daily_usage': 0,
                'usage_distribution': {},
                'weekday_patterns': {}
            }
            
            if daily_usage:
                # Find peak day
                peak_day = max(daily_usage.items(), key=lambda x: x[1])
                patterns['peak_day'] = {
                    'date': peak_day[0],
                    'requests': peak_day[1]
                }
                
                # Calculate average
                patterns['avg_daily_usage'] = statistics.mean(daily_usage.values())
                
                # Analyze weekday patterns
                weekday_usage = defaultdict(list)
                for date_str, usage in daily_usage.items():
                    try:
                        date_obj = datetime.fromisoformat(date_str).date()
                        weekday = date_obj.strftime('%A')
                        weekday_usage[weekday].append(usage)
                    except:
                        continue
                
                for weekday, usage_list in weekday_usage.items():
                    patterns['weekday_patterns'][weekday] = {
                        'avg_usage': statistics.mean(usage_list),
                        'total_days': len(usage_list)
                    }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting usage patterns: {e}")
            return {}
    
    async def _get_cost_analysis(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get cost analysis data."""
        try:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            total_cost = team_analytics.get('summary', {}).get('total_cost', 0) or 0
            total_requests = team_analytics.get('summary', {}).get('total_requests', 0)
            
            cost_analysis = {
                'total_cost': total_cost,
                'avg_cost_per_request': total_cost / total_requests if total_requests > 0 else 0,
                'daily_cost_breakdown': {},
                'cost_by_prompt': {},
                'cost_trends': {},
                'cost_optimization_opportunities': []
            }
            
            # Generate cost optimization recommendations
            if total_cost > 100:  # If significant cost
                cost_analysis['cost_optimization_opportunities'] = [
                    "Consider prompt optimization to reduce token usage",
                    "Review high-cost prompts for efficiency improvements",
                    "Implement caching for frequently used prompts"
                ]
            
            return cost_analysis
            
        except Exception as e:
            self.logger.error(f"Error getting cost analysis: {e}")
            return {}
    
    async def _get_quality_metrics(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get quality metrics analysis."""
        try:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            summary = team_analytics.get('summary', {})
            
            quality_metrics = {
                'overall_quality_score': 0,
                'accuracy_distribution': {},
                'quality_trends': {},
                'quality_issues': [],
                'improvement_suggestions': []
            }
            
            # Calculate overall quality score
            accuracy = summary.get('avg_accuracy_score', 0) or 0
            success_rate = summary.get('success_rate', 0) / 100
            
            quality_score = (accuracy * 0.6 + success_rate * 0.4) * 100
            quality_metrics['overall_quality_score'] = round(quality_score, 2)
            
            # Generate quality improvement suggestions
            if accuracy < 0.8:
                quality_metrics['improvement_suggestions'].append(
                    "Accuracy scores are below 80%. Consider prompt refinement."
                )
            
            if success_rate < 0.95:
                quality_metrics['improvement_suggestions'].append(
                    "Success rate is below 95%. Review error patterns."
                )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting quality metrics: {e}")
            return {}
    
    async def _get_user_activity(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get user activity analysis."""
        try:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            top_users = team_analytics.get('top_users', [])
            unique_users = team_analytics.get('summary', {}).get('unique_users', 0)
            
            user_activity = {
                'total_active_users': unique_users,
                'top_users': top_users,
                'user_engagement_score': 0,
                'new_users': 0,  # Would need historical data
                'user_retention': {},
                'activity_patterns': {}
            }
            
            # Calculate engagement score
            if top_users:
                total_requests = sum(user['request_count'] for user in top_users)
                avg_requests_per_user = total_requests / len(top_users) if top_users else 0
                user_activity['user_engagement_score'] = round(avg_requests_per_user, 2)
            
            return user_activity
            
        except Exception as e:
            self.logger.error(f"Error getting user activity: {e}")
            return {}
    
    async def _get_active_alerts(self, team_id: str) -> List[Dict[str, Any]]:
        """Get active alerts for the team."""
        try:
            # This would typically query alert rules and check conditions
            # For now, return sample alerts
            
            alerts = []
            
            # Sample alert conditions (would be real in production)
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=1
            )
            
            summary = team_analytics.get('summary', {})
            success_rate = summary.get('success_rate', 100)
            
            if success_rate < 95:
                alerts.append({
                    'alert_id': 'alert_001',
                    'type': 'performance',
                    'severity': 'high',
                    'title': 'Low Success Rate',
                    'message': f'Team success rate is {success_rate}%, below threshold of 95%',
                    'triggered_at': datetime.utcnow().isoformat(),
                    'status': 'active'
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def _generate_recommendations(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered recommendations for the team."""
        try:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            recommendations = []
            summary = team_analytics.get('summary', {})
            
            # Performance recommendations
            success_rate = summary.get('success_rate', 100)
            if success_rate < 95:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'title': 'Improve Success Rate',
                    'description': 'Success rate is below optimal. Review error patterns and improve prompt reliability.',
                    'action_items': [
                        'Analyze failed requests for common patterns',
                        'Review and update error-prone prompts',
                        'Implement better error handling'
                    ]
                })
            
            # Cost optimization recommendations
            total_cost = summary.get('total_cost', 0) or 0
            if total_cost > 500:  # Threshold for cost optimization
                recommendations.append({
                    'type': 'cost',
                    'priority': 'medium',
                    'title': 'Cost Optimization Opportunity',
                    'description': f'Total cost of ${total_cost:.2f} suggests optimization opportunities.',
                    'action_items': [
                        'Review high-cost prompts for efficiency',
                        'Implement prompt caching where appropriate',
                        'Consider model selection optimization'
                    ]
                })
            
            # Usage pattern recommendations
            top_prompts = team_analytics.get('top_prompts', [])
            if len(top_prompts) > 0:
                top_prompt = top_prompts[0]
                if top_prompt['usage_count'] > summary.get('total_requests', 0) * 0.5:
                    recommendations.append({
                        'type': 'usage',
                        'priority': 'low',
                        'title': 'Prompt Diversification',
                        'description': 'Heavy reliance on a single prompt detected.',
                        'action_items': [
                            'Consider creating variations of popular prompts',
                            'Implement A/B testing for optimization',
                            'Develop backup prompts for reliability'
                        ]
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []

class InsightsGenerator:
    """AI-powered insights generator for prompt analytics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def generate_insights(
        self,
        team_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive insights for a team."""
        try:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            
            # Get team analytics
            team_analytics = await prompt_performance_tracker.get_team_analytics(
                team_id=team_id,
                days=days
            )
            
            insights = []
            
            if 'error' in team_analytics:
                return insights
            
            summary = team_analytics.get('summary', {})
            
            # Performance insights
            success_rate = summary.get('success_rate', 0)
            if success_rate > 98:
                insights.append({
                    'type': 'performance',
                    'category': 'positive',
                    'title': 'Excellent Reliability',
                    'description': f'Your team maintains an outstanding {success_rate}% success rate.',
                    'impact': 'high',
                    'confidence': 0.95
                })
            elif success_rate < 90:
                insights.append({
                    'type': 'performance',
                    'category': 'concern',
                    'title': 'Reliability Issues Detected',
                    'description': f'Success rate of {success_rate}% indicates potential reliability issues.',
                    'impact': 'high',
                    'confidence': 0.90
                })
            
            # Usage insights
            total_requests = summary.get('total_requests', 0)
            unique_prompts = summary.get('unique_prompts', 0)
            
            if unique_prompts > 0:
                avg_requests_per_prompt = total_requests / unique_prompts
                if avg_requests_per_prompt > 100:
                    insights.append({
                        'type': 'usage',
                        'category': 'optimization',
                        'title': 'High Prompt Utilization',
                        'description': f'Average of {avg_requests_per_prompt:.0f} requests per prompt indicates good reusability.',
                        'impact': 'medium',
                        'confidence': 0.85
                    })
            
            # Cost insights
            total_cost = summary.get('total_cost', 0) or 0
            if total_cost > 0 and total_requests > 0:
                cost_per_request = total_cost / total_requests
                if cost_per_request < 0.001:
                    insights.append({
                        'type': 'cost',
                        'category': 'positive',
                        'title': 'Cost Efficient Operations',
                        'description': f'Low cost per request (${cost_per_request:.4f}) indicates efficient prompt usage.',
                        'impact': 'medium',
                        'confidence': 0.80
                    })
            
            # Team collaboration insights
            unique_users = summary.get('unique_users', 0)
            if unique_users > 5:
                insights.append({
                    'type': 'collaboration',
                    'category': 'positive',
                    'title': 'Active Team Collaboration',
                    'description': f'{unique_users} team members are actively using prompts.',
                    'impact': 'medium',
                    'confidence': 0.75
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return []

# Global instances
team_analytics_dashboard = TeamAnalyticsDashboard()
insights_generator = InsightsGenerator()