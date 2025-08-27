"""
Real-time prompt performance tracking and analytics system.
Provides comprehensive tracking of prompt usage, performance metrics, and analytics.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import uuid
import statistics
import logging

from ..core.config import get_settings
from ..core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

@dataclass
class PromptUsageEvent:
    """Individual prompt usage event."""
    event_id: str
    prompt_id: str
    version_id: Optional[str]
    user_id: Optional[str]
    team_id: Optional[str]
    
    # Performance metrics
    accuracy_score: Optional[float] = None
    relevance_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    user_satisfaction: Optional[float] = None
    response_time_ms: Optional[int] = None
    token_usage: Optional[int] = None
    cost_per_request: Optional[float] = None
    
    # Context information
    use_case: Optional[str] = None
    model_used: Optional[str] = None
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    
    # Metadata
    timestamp: datetime = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class PromptPerformanceSummary:
    """Summary of prompt performance over a time period."""
    prompt_id: str
    version_id: Optional[str]
    period_start: datetime
    period_end: datetime
    
    # Usage statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    unique_users: int = 0
    
    # Performance metrics (averages)
    avg_accuracy_score: Optional[float] = None
    avg_relevance_score: Optional[float] = None
    avg_efficiency_score: Optional[float] = None
    avg_user_satisfaction: Optional[float] = None
    avg_response_time_ms: Optional[float] = None
    avg_token_usage: Optional[float] = None
    avg_cost_per_request: Optional[float] = None
    
    # Performance trends
    accuracy_trend: Optional[str] = None  # 'improving', 'declining', 'stable'
    usage_trend: Optional[str] = None
    cost_trend: Optional[str] = None
    
    # Time-based patterns
    hourly_usage: Dict[int, int] = None
    daily_usage: Dict[str, int] = None
    peak_usage_hour: Optional[int] = None
    
    def __post_init__(self):
        if self.hourly_usage is None:
            self.hourly_usage = {}
        if self.daily_usage is None:
            self.daily_usage = {}

class PromptPerformanceTracker:
    """Real-time prompt performance tracking system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.events_buffer: List[PromptUsageEvent] = []
        self.performance_cache: Dict[str, PromptPerformanceSummary] = {}
        self.real_time_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.running = False
        
    async def record_prompt_usage(
        self,
        prompt_id: str,
        version_id: Optional[str] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a prompt usage event with performance metrics."""
        try:
            event = PromptUsageEvent(
                event_id=str(uuid.uuid4()),
                prompt_id=prompt_id,
                version_id=version_id,
                user_id=user_id,
                team_id=team_id
            )
            
            # Add performance metrics if provided
            if performance_metrics:
                event.accuracy_score = performance_metrics.get('accuracy_score')
                event.relevance_score = performance_metrics.get('relevance_score')
                event.efficiency_score = performance_metrics.get('efficiency_score')
                event.user_satisfaction = performance_metrics.get('user_satisfaction')
                event.response_time_ms = performance_metrics.get('response_time_ms')
                event.token_usage = performance_metrics.get('token_usage')
                event.cost_per_request = performance_metrics.get('cost_per_request')
            
            # Add context information if provided
            if context:
                event.use_case = context.get('use_case')
                event.model_used = context.get('model_used')
                event.input_length = context.get('input_length')
                event.output_length = context.get('output_length')
                event.success = context.get('success', True)
                event.error_message = context.get('error_message')
            
            # Add to buffer
            self.events_buffer.append(event)
            
            # Update real-time metrics
            await self._update_real_time_metrics(event)
            
            self.logger.info(
                f"Recorded prompt usage: {prompt_id}",
                prompt_id=prompt_id,
                version_id=version_id,
                user_id=user_id,
                team_id=team_id,
                success=event.success
            )
            
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Error recording prompt usage: {e}")
            raise
    
    async def _update_real_time_metrics(self, event: PromptUsageEvent):
        """Update real-time metrics for immediate access."""
        key = f"{event.prompt_id}:{event.version_id or 'latest'}"
        
        if key not in self.real_time_metrics:
            self.real_time_metrics[key] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'response_times': [],
                'accuracy_scores': [],
                'last_updated': datetime.utcnow()
            }
        
        metrics = self.real_time_metrics[key]
        metrics['total_requests'] += 1
        
        if event.success:
            metrics['successful_requests'] += 1
        else:
            metrics['failed_requests'] += 1
        
        if event.response_time_ms is not None:
            metrics['response_times'].append(event.response_time_ms)
            # Keep only last 100 measurements for real-time calculations
            if len(metrics['response_times']) > 100:
                metrics['response_times'] = metrics['response_times'][-100:]
        
        if event.accuracy_score is not None:
            metrics['accuracy_scores'].append(event.accuracy_score)
            if len(metrics['accuracy_scores']) > 100:
                metrics['accuracy_scores'] = metrics['accuracy_scores'][-100:]
        
        metrics['last_updated'] = datetime.utcnow()
    
    async def get_real_time_metrics(self, prompt_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Get real-time metrics for a prompt."""
        key = f"{prompt_id}:{version_id or 'latest'}"
        
        if key not in self.real_time_metrics:
            return {
                'prompt_id': prompt_id,
                'version_id': version_id,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0,
                'avg_response_time': None,
                'avg_accuracy_score': None,
                'last_updated': None
            }
        
        metrics = self.real_time_metrics[key]
        
        # Calculate averages
        avg_response_time = None
        if metrics['response_times']:
            avg_response_time = statistics.mean(metrics['response_times'])
        
        avg_accuracy_score = None
        if metrics['accuracy_scores']:
            avg_accuracy_score = statistics.mean(metrics['accuracy_scores'])
        
        success_rate = 0.0
        if metrics['total_requests'] > 0:
            success_rate = metrics['successful_requests'] / metrics['total_requests'] * 100
        
        return {
            'prompt_id': prompt_id,
            'version_id': version_id,
            'total_requests': metrics['total_requests'],
            'successful_requests': metrics['successful_requests'],
            'failed_requests': metrics['failed_requests'],
            'success_rate': round(success_rate, 2),
            'avg_response_time': round(avg_response_time, 2) if avg_response_time else None,
            'avg_accuracy_score': round(avg_accuracy_score, 3) if avg_accuracy_score else None,
            'last_updated': metrics['last_updated']
        }
    
    async def get_prompt_performance_summary(
        self,
        prompt_id: str,
        version_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive performance summary for a prompt."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Filter events for the specified prompt and time period
            relevant_events = [
                event for event in self.events_buffer
                if event.prompt_id == prompt_id
                and (version_id is None or event.version_id == version_id)
                and start_date <= event.timestamp <= end_date
            ]
            
            if not relevant_events:
                return {
                    'prompt_id': prompt_id,
                    'version_id': version_id,
                    'period_start': start_date,
                    'period_end': end_date,
                    'total_requests': 0,
                    'message': 'No usage data found for the specified period'
                }
            
            # Calculate summary statistics
            total_requests = len(relevant_events)
            successful_requests = sum(1 for e in relevant_events if e.success)
            failed_requests = total_requests - successful_requests
            unique_users = len(set(e.user_id for e in relevant_events if e.user_id))
            
            # Calculate performance metrics averages
            accuracy_scores = [e.accuracy_score for e in relevant_events if e.accuracy_score is not None]
            relevance_scores = [e.relevance_score for e in relevant_events if e.relevance_score is not None]
            efficiency_scores = [e.efficiency_score for e in relevant_events if e.efficiency_score is not None]
            satisfaction_scores = [e.user_satisfaction for e in relevant_events if e.user_satisfaction is not None]
            response_times = [e.response_time_ms for e in relevant_events if e.response_time_ms is not None]
            token_usage = [e.token_usage for e in relevant_events if e.token_usage is not None]
            costs = [e.cost_per_request for e in relevant_events if e.cost_per_request is not None]
            
            # Calculate hourly and daily usage patterns
            hourly_usage = defaultdict(int)
            daily_usage = defaultdict(int)
            
            for event in relevant_events:
                hour = event.timestamp.hour
                day = event.timestamp.date().isoformat()
                hourly_usage[hour] += 1
                daily_usage[day] += 1
            
            # Find peak usage hour
            peak_usage_hour = max(hourly_usage.items(), key=lambda x: x[1])[0] if hourly_usage else None
            
            # Calculate trends (simplified)
            accuracy_trend = self._calculate_trend(accuracy_scores) if accuracy_scores else None
            usage_trend = self._calculate_usage_trend(daily_usage)
            
            summary = {
                'prompt_id': prompt_id,
                'version_id': version_id,
                'period_start': start_date,
                'period_end': end_date,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': round(successful_requests / total_requests * 100, 2) if total_requests > 0 else 0,
                'unique_users': unique_users,
                'avg_accuracy_score': round(statistics.mean(accuracy_scores), 3) if accuracy_scores else None,
                'avg_relevance_score': round(statistics.mean(relevance_scores), 3) if relevance_scores else None,
                'avg_efficiency_score': round(statistics.mean(efficiency_scores), 3) if efficiency_scores else None,
                'avg_user_satisfaction': round(statistics.mean(satisfaction_scores), 3) if satisfaction_scores else None,
                'avg_response_time_ms': round(statistics.mean(response_times), 2) if response_times else None,
                'avg_token_usage': round(statistics.mean(token_usage), 2) if token_usage else None,
                'avg_cost_per_request': round(statistics.mean(costs), 4) if costs else None,
                'total_cost': round(sum(costs), 2) if costs else None,
                'accuracy_trend': accuracy_trend,
                'usage_trend': usage_trend,
                'hourly_usage': dict(hourly_usage),
                'daily_usage': dict(daily_usage),
                'peak_usage_hour': peak_usage_hour
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting prompt performance summary: {e}")
            return {'error': str(e)}
    
    async def get_team_analytics(self, team_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics for a team."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Filter events for the team
            team_events = [
                event for event in self.events_buffer
                if event.team_id == team_id
                and start_date <= event.timestamp <= end_date
            ]
            
            if not team_events:
                return {
                    'team_id': team_id,
                    'period_start': start_date,
                    'period_end': end_date,
                    'total_requests': 0,
                    'message': 'No usage data found for the specified team and period'
                }
            
            # Calculate team-wide statistics
            total_requests = len(team_events)
            successful_requests = sum(1 for e in team_events if e.success)
            unique_prompts = len(set(e.prompt_id for e in team_events))
            unique_users = len(set(e.user_id for e in team_events if e.user_id))
            
            # Prompt usage breakdown
            prompt_usage = Counter(e.prompt_id for e in team_events)
            top_prompts = [
                {'prompt_id': prompt_id, 'usage_count': count}
                for prompt_id, count in prompt_usage.most_common(10)
            ]
            
            # User activity breakdown
            user_activity = Counter(e.user_id for e in team_events if e.user_id)
            top_users = [
                {'user_id': user_id, 'request_count': count}
                for user_id, count in user_activity.most_common(10)
            ]
            
            # Performance metrics
            accuracy_scores = [e.accuracy_score for e in team_events if e.accuracy_score is not None]
            response_times = [e.response_time_ms for e in team_events if e.response_time_ms is not None]
            costs = [e.cost_per_request for e in team_events if e.cost_per_request is not None]
            
            # Daily usage pattern
            daily_usage = defaultdict(int)
            for event in team_events:
                day = event.timestamp.date().isoformat()
                daily_usage[day] += 1
            
            analytics = {
                'team_id': team_id,
                'period_start': start_date,
                'period_end': end_date,
                'summary': {
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'success_rate': round(successful_requests / total_requests * 100, 2) if total_requests > 0 else 0,
                    'unique_prompts': unique_prompts,
                    'unique_users': unique_users,
                    'avg_accuracy_score': round(statistics.mean(accuracy_scores), 3) if accuracy_scores else None,
                    'avg_response_time_ms': round(statistics.mean(response_times), 2) if response_times else None,
                    'total_cost': round(sum(costs), 2) if costs else None
                },
                'top_prompts': top_prompts,
                'top_users': top_users,
                'daily_usage': dict(daily_usage),
                'usage_trend': self._calculate_usage_trend(daily_usage)
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting team analytics: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple trend calculation using first and last quartiles
        n = len(values)
        first_quarter = values[:n//4] if n >= 4 else values[:1]
        last_quarter = values[-n//4:] if n >= 4 else values[-1:]
        
        if not first_quarter or not last_quarter:
            return 'stable'
        
        first_avg = statistics.mean(first_quarter)
        last_avg = statistics.mean(last_quarter)
        
        change_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
        
        if change_percent > 5:
            return 'improving'
        elif change_percent < -5:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_usage_trend(self, daily_usage: Dict[str, int]) -> str:
        """Calculate usage trend from daily usage data."""
        if len(daily_usage) < 2:
            return 'stable'
        
        # Sort by date and get values
        sorted_days = sorted(daily_usage.keys())
        values = [daily_usage[day] for day in sorted_days]
        
        return self._calculate_trend(values)
    
    async def flush_events_to_storage(self):
        """Flush events buffer to persistent storage."""
        if not self.events_buffer:
            return
        
        try:
            # In a real implementation, this would save to database
            # For now, we'll just log the flush operation
            self.logger.info(f"Flushing {len(self.events_buffer)} events to storage")
            
            # Keep only recent events in memory (last 10000)
            if len(self.events_buffer) > 10000:
                self.events_buffer = self.events_buffer[-10000:]
            
        except Exception as e:
            self.logger.error(f"Error flushing events to storage: {e}")
    
    async def start_background_tasks(self):
        """Start background tasks for analytics processing."""
        self.running = True
        
        # Start periodic flush task
        asyncio.create_task(self._periodic_flush())
        
        self.logger.info("Prompt performance tracker started")
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        self.running = False
        self.logger.info("Prompt performance tracker stopped")
    
    async def _periodic_flush(self):
        """Periodically flush events to storage."""
        while self.running:
            try:
                await self.flush_events_to_storage()
                await asyncio.sleep(300)  # Flush every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in periodic flush: {e}")
                await asyncio.sleep(60)

class AnalyticsEngine:
    """Advanced analytics engine for prompt performance analysis."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.performance_tracker = PromptPerformanceTracker()
    
    async def generate_performance_insights(
        self,
        prompt_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Generate AI-powered insights about prompt performance."""
        try:
            # Get performance summary
            summary = await self.performance_tracker.get_prompt_performance_summary(
                prompt_id=prompt_id,
                days=days
            )
            
            if 'error' in summary:
                return summary
            
            insights = []
            recommendations = []
            
            # Analyze success rate
            success_rate = summary.get('success_rate', 0)
            if success_rate < 90:
                insights.append(f"Success rate is {success_rate}%, which is below optimal (>95%)")
                recommendations.append("Review error patterns and improve prompt reliability")
            elif success_rate > 98:
                insights.append(f"Excellent success rate of {success_rate}%")
            
            # Analyze response time
            avg_response_time = summary.get('avg_response_time_ms')
            if avg_response_time and avg_response_time > 5000:
                insights.append(f"Average response time is {avg_response_time}ms, which may impact user experience")
                recommendations.append("Consider optimizing prompt length or model selection")
            
            # Analyze accuracy trend
            accuracy_trend = summary.get('accuracy_trend')
            if accuracy_trend == 'declining':
                insights.append("Accuracy scores are declining over time")
                recommendations.append("Review recent prompt changes and consider A/B testing")
            elif accuracy_trend == 'improving':
                insights.append("Accuracy scores are improving over time")
            
            # Analyze usage patterns
            hourly_usage = summary.get('hourly_usage', {})
            if hourly_usage:
                peak_hour = max(hourly_usage.items(), key=lambda x: x[1])[0]
                insights.append(f"Peak usage occurs at hour {peak_hour}")
            
            # Cost analysis
            total_cost = summary.get('total_cost')
            avg_cost = summary.get('avg_cost_per_request')
            if total_cost and avg_cost:
                if avg_cost > 0.01:  # Threshold for expensive requests
                    insights.append(f"Average cost per request is ${avg_cost:.4f}")
                    recommendations.append("Consider cost optimization strategies")
            
            return {
                'prompt_id': prompt_id,
                'analysis_period': f"{days} days",
                'insights': insights,
                'recommendations': recommendations,
                'performance_summary': summary,
                'generated_at': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance insights: {e}")
            return {'error': str(e)}
    
    async def compare_prompt_versions(
        self,
        prompt_id: str,
        version_ids: List[str],
        days: int = 30
    ) -> Dict[str, Any]:
        """Compare performance between different prompt versions."""
        try:
            comparisons = {}
            
            for version_id in version_ids:
                summary = await self.performance_tracker.get_prompt_performance_summary(
                    prompt_id=prompt_id,
                    version_id=version_id,
                    days=days
                )
                comparisons[version_id] = summary
            
            # Find best performing version
            best_version = None
            best_score = 0
            
            for version_id, summary in comparisons.items():
                if 'error' not in summary:
                    # Calculate composite score
                    accuracy = summary.get('avg_accuracy_score', 0) or 0
                    success_rate = summary.get('success_rate', 0) / 100
                    response_time_score = 1 - min(summary.get('avg_response_time_ms', 1000) / 10000, 1)
                    
                    composite_score = (accuracy * 0.4 + success_rate * 0.4 + response_time_score * 0.2)
                    
                    if composite_score > best_score:
                        best_score = composite_score
                        best_version = version_id
            
            return {
                'prompt_id': prompt_id,
                'comparison_period': f"{days} days",
                'version_comparisons': comparisons,
                'best_version': best_version,
                'best_score': round(best_score, 3),
                'generated_at': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing prompt versions: {e}")
            return {'error': str(e)}

# Global instances
prompt_performance_tracker = PromptPerformanceTracker()
analytics_engine = AnalyticsEngine()