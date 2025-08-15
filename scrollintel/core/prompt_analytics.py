"""
Comprehensive prompt performance tracking system.
Provides real-time analytics, performance monitoring, and usage insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

from ..models.analytics_models import (
    PromptMetrics, UsageAnalytics, AnalyticsReport, AlertRule,
    TrendAnalysis, PatternRecognition
)
from ..models.database import get_db_session

logger = logging.getLogger(__name__)

class PromptPerformanceTracker:
    """Tracks and analyzes prompt performance in real-time."""
    
    def __init__(self):
        self.metrics_cache = {}
        self.performance_thresholds = {
            'accuracy_score': {'good': 0.8, 'poor': 0.6},
            'relevance_score': {'good': 0.85, 'poor': 0.7},
            'efficiency_score': {'good': 0.9, 'poor': 0.7},
            'response_time_ms': {'good': 1000, 'poor': 3000},
            'user_satisfaction': {'good': 4.0, 'poor': 3.0}
        }
    
    async def record_prompt_usage(
        self,
        prompt_id: str,
        version_id: Optional[str] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a prompt usage event with performance metrics."""
        try:
            with get_db_session() as db:
                # Create metrics record
                metrics = PromptMetrics(
                    prompt_id=prompt_id,
                    version_id=version_id,
                    accuracy_score=performance_metrics.get('accuracy_score') if performance_metrics else None,
                    relevance_score=performance_metrics.get('relevance_score') if performance_metrics else None,
                    efficiency_score=performance_metrics.get('efficiency_score') if performance_metrics else None,
                    user_satisfaction=performance_metrics.get('user_satisfaction') if performance_metrics else None,
                    response_time_ms=performance_metrics.get('response_time_ms') if performance_metrics else None,
                    token_usage=performance_metrics.get('token_usage') if performance_metrics else None,
                    cost_per_request=performance_metrics.get('cost_per_request') if performance_metrics else None,
                    use_case=context.get('use_case') if context else None,
                    model_used=context.get('model_used') if context else None,
                    user_id=context.get('user_id') if context else None,
                    team_id=context.get('team_id') if context else None
                )
                
                db.add(metrics)
                db.commit()
                
                # Update usage analytics asynchronously
                asyncio.create_task(self._update_usage_analytics(prompt_id, metrics.id))
                
                logger.info(f"Recorded prompt usage for {prompt_id}")
                return metrics.id
                
        except Exception as e:
            logger.error(f"Error recording prompt usage: {str(e)}")
            raise
    
    async def _update_usage_analytics(self, prompt_id: str, metrics_id: str):
        """Update usage analytics for a prompt."""
        try:
            with get_db_session() as db:
                # Get or create analytics record for current period
                today = datetime.utcnow().date()
                period_start = datetime.combine(today, datetime.min.time())
                period_end = period_start + timedelta(days=1)
                
                analytics = db.query(UsageAnalytics).filter(
                    and_(
                        UsageAnalytics.prompt_id == prompt_id,
                        UsageAnalytics.analysis_period_start == period_start
                    )
                ).first()
                
                if not analytics:
                    analytics = UsageAnalytics(
                        prompt_id=prompt_id,
                        analysis_period_start=period_start,
                        analysis_period_end=period_end,
                        total_requests=0,
                        successful_requests=0,
                        failed_requests=0,
                        daily_usage={},
                        hourly_patterns={str(i): 0 for i in range(24)},
                        team_usage={},
                        user_adoption={}
                    )
                    db.add(analytics)
                
                # Update counters
                analytics.total_requests += 1
                analytics.successful_requests += 1  # Assume success for now
                
                # Update hourly patterns
                current_hour = datetime.utcnow().hour
                hourly_patterns = analytics.hourly_patterns or {str(i): 0 for i in range(24)}
                hourly_patterns[str(current_hour)] = hourly_patterns.get(str(current_hour), 0) + 1
                analytics.hourly_patterns = hourly_patterns
                
                # Update daily usage
                today_str = today.isoformat()
                daily_usage = analytics.daily_usage or {}
                daily_usage[today_str] = daily_usage.get(today_str, 0) + 1
                analytics.daily_usage = daily_usage
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Error updating usage analytics: {str(e)}")
    
    async def get_prompt_performance_summary(
        self,
        prompt_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive performance summary for a prompt."""
        try:
            with get_db_session() as db:
                # Get metrics for the specified period
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.prompt_id == prompt_id,
                        PromptMetrics.created_at >= cutoff_date
                    )
                ).all()
                
                if not metrics:
                    return {"error": "No metrics found for this prompt"}
                
                # Calculate summary statistics
                summary = {
                    "prompt_id": prompt_id,
                    "period_days": days,
                    "total_usage": len(metrics),
                    "performance_metrics": {},
                    "trends": {},
                    "recommendations": []
                }
                
                # Calculate performance metrics
                for metric_name in ['accuracy_score', 'relevance_score', 'efficiency_score', 
                                  'user_satisfaction', 'response_time_ms', 'token_usage', 'cost_per_request']:
                    values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
                    
                    if values:
                        summary["performance_metrics"][metric_name] = {
                            "average": np.mean(values),
                            "median": np.median(values),
                            "std_dev": np.std(values),
                            "min": min(values),
                            "max": max(values),
                            "trend": self._calculate_trend(values)
                        }
                
                # Add usage patterns
                summary["usage_patterns"] = await self._analyze_usage_patterns(prompt_id, days)
                
                # Generate recommendations
                summary["recommendations"] = self._generate_performance_recommendations(summary)
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:  # Threshold for "stable"
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"
    
    async def _analyze_usage_patterns(self, prompt_id: str, days: int) -> Dict[str, Any]:
        """Analyze usage patterns for a prompt."""
        try:
            with get_db_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                analytics = db.query(UsageAnalytics).filter(
                    and_(
                        UsageAnalytics.prompt_id == prompt_id,
                        UsageAnalytics.analysis_period_start >= cutoff_date
                    )
                ).all()
                
                if not analytics:
                    return {}
                
                # Aggregate patterns
                total_requests = sum(a.total_requests for a in analytics)
                avg_response_time = np.mean([a.avg_response_time for a in analytics if a.avg_response_time])
                
                # Combine hourly patterns
                combined_hourly = defaultdict(int)
                for a in analytics:
                    if a.hourly_patterns:
                        for hour, count in a.hourly_patterns.items():
                            combined_hourly[hour] += count
                
                # Find peak usage hours
                peak_hours = sorted(combined_hourly.items(), key=lambda x: x[1], reverse=True)[:3]
                
                return {
                    "total_requests": total_requests,
                    "avg_response_time": avg_response_time,
                    "peak_usage_hours": [{"hour": int(h), "requests": c} for h, c in peak_hours],
                    "usage_distribution": dict(combined_hourly)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {str(e)}")
            return {}
    
    def _generate_performance_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate AI-powered performance recommendations."""
        recommendations = []
        
        performance_metrics = summary.get("performance_metrics", {})
        
        # Check accuracy
        if "accuracy_score" in performance_metrics:
            avg_accuracy = performance_metrics["accuracy_score"]["average"]
            if avg_accuracy < self.performance_thresholds["accuracy_score"]["poor"]:
                recommendations.append("Consider revising prompt structure to improve accuracy")
            elif performance_metrics["accuracy_score"]["trend"] == "declining":
                recommendations.append("Accuracy is declining - review recent changes")
        
        # Check response time
        if "response_time_ms" in performance_metrics:
            avg_time = performance_metrics["response_time_ms"]["average"]
            if avg_time > self.performance_thresholds["response_time_ms"]["poor"]:
                recommendations.append("Response time is high - consider prompt optimization")
        
        # Check user satisfaction
        if "user_satisfaction" in performance_metrics:
            avg_satisfaction = performance_metrics["user_satisfaction"]["average"]
            if avg_satisfaction < self.performance_thresholds["user_satisfaction"]["poor"]:
                recommendations.append("User satisfaction is low - gather feedback for improvements")
        
        # Usage-based recommendations
        usage_patterns = summary.get("usage_patterns", {})
        if usage_patterns.get("total_requests", 0) > 1000:
            recommendations.append("High usage detected - consider A/B testing for optimization")
        
        return recommendations
    
    async def get_team_analytics(
        self,
        team_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for a team."""
        try:
            with get_db_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get all metrics for team
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.team_id == team_id,
                        PromptMetrics.created_at >= cutoff_date
                    )
                ).all()
                
                if not metrics:
                    return {"error": "No metrics found for this team"}
                
                # Group by prompt
                prompt_metrics = defaultdict(list)
                for metric in metrics:
                    prompt_metrics[metric.prompt_id].append(metric)
                
                # Calculate team summary
                team_summary = {
                    "team_id": team_id,
                    "period_days": days,
                    "total_prompts": len(prompt_metrics),
                    "total_usage": len(metrics),
                    "prompt_performance": {},
                    "top_performers": [],
                    "improvement_opportunities": []
                }
                
                # Analyze each prompt
                for prompt_id, prompt_metrics_list in prompt_metrics.items():
                    performance = self._calculate_prompt_performance_score(prompt_metrics_list)
                    team_summary["prompt_performance"][prompt_id] = performance
                
                # Identify top performers and improvement opportunities
                sorted_prompts = sorted(
                    team_summary["prompt_performance"].items(),
                    key=lambda x: x[1]["overall_score"],
                    reverse=True
                )
                
                team_summary["top_performers"] = sorted_prompts[:5]
                team_summary["improvement_opportunities"] = sorted_prompts[-5:]
                
                return team_summary
                
        except Exception as e:
            logger.error(f"Error getting team analytics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_prompt_performance_score(self, metrics: List[PromptMetrics]) -> Dict[str, Any]:
        """Calculate overall performance score for a prompt."""
        scores = []
        
        # Weight different metrics
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
        
        overall_score = sum(scores) if scores else 0
        
        return {
            "overall_score": overall_score,
            "usage_count": len(metrics),
            "performance_category": self._categorize_performance(overall_score)
        }
    
    def _categorize_performance(self, score: float) -> str:
        """Categorize performance based on score."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "fair"
        else:
            return "needs_improvement"

class AnalyticsEngine:
    """Main analytics engine for prompt management system."""
    
    def __init__(self):
        self.performance_tracker = PromptPerformanceTracker()
        self.trend_analyzer = TrendAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
    
    async def generate_comprehensive_report(
        self,
        report_type: str,
        scope: Dict[str, Any],
        date_range: Tuple[datetime, datetime]
    ) -> str:
        """Generate comprehensive analytics report."""
        try:
            report_data = {
                "report_type": report_type,
                "scope": scope,
                "date_range": date_range,
                "generated_at": datetime.utcnow(),
                "summary": {},
                "detailed_analysis": {},
                "recommendations": []
            }
            
            if report_type == "performance":
                report_data = await self._generate_performance_report(report_data)
            elif report_type == "usage":
                report_data = await self._generate_usage_report(report_data)
            elif report_type == "team":
                report_data = await self._generate_team_report(report_data)
            elif report_type == "trend":
                report_data = await self._generate_trend_report(report_data)
            
            # Save report to database
            with get_db_session() as db:
                report = AnalyticsReport(
                    report_type=report_type,
                    title=f"{report_type.title()} Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
                    description=f"Comprehensive {report_type} analysis",
                    prompt_ids=scope.get("prompt_ids"),
                    team_ids=scope.get("team_ids"),
                    date_range_start=date_range[0],
                    date_range_end=date_range[1],
                    summary=report_data["summary"],
                    detailed_data=report_data["detailed_analysis"],
                    recommendations=report_data["recommendations"],
                    status="generated"
                )
                
                db.add(report)
                db.commit()
                
                logger.info(f"Generated {report_type} report: {report.id}")
                return report.id
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    async def _generate_performance_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance-focused report."""
        # Implementation for performance report
        report_data["summary"]["report_focus"] = "Performance Analysis"
        return report_data
    
    async def _generate_usage_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate usage-focused report."""
        # Implementation for usage report
        report_data["summary"]["report_focus"] = "Usage Analysis"
        return report_data
    
    async def _generate_team_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate team-focused report."""
        # Implementation for team report
        report_data["summary"]["report_focus"] = "Team Analysis"
        return report_data
    
    async def _generate_trend_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trend-focused report."""
        # Implementation for trend report
        report_data["summary"]["report_focus"] = "Trend Analysis"
        return report_data

class TrendAnalyzer:
    """Analyzes trends in prompt performance and usage."""
    
    def __init__(self):
        self.trend_algorithms = ['linear', 'polynomial', 'seasonal']
    
    async def analyze_trends(
        self,
        prompt_id: str,
        metric_name: str,
        days: int = 30
    ) -> TrendAnalysis:
        """Analyze trends for a specific metric."""
        try:
            with get_db_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                metrics = db.query(PromptMetrics).filter(
                    and_(
                        PromptMetrics.prompt_id == prompt_id,
                        PromptMetrics.created_at >= cutoff_date
                    )
                ).order_by(PromptMetrics.created_at).all()
                
                values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
                timestamps = [m.created_at for m in metrics if getattr(m, metric_name) is not None]
                
                if len(values) < 3:
                    return TrendAnalysis(
                        metric_name=metric_name,
                        trend_direction="insufficient_data",
                        trend_strength=0.0,
                        confidence_level=0.0,
                        data_points=[],
                        forecast=None
                    )
                
                # Calculate trend
                trend_direction, trend_strength, confidence = self._calculate_advanced_trend(values)
                
                # Prepare data points
                data_points = [
                    {"timestamp": ts.isoformat(), "value": val}
                    for ts, val in zip(timestamps, values)
                ]
                
                # Generate forecast
                forecast = self._generate_forecast(values, timestamps)
                
                return TrendAnalysis(
                    metric_name=metric_name,
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    confidence_level=confidence,
                    data_points=data_points,
                    forecast=forecast
                )
                
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            raise
    
    def _calculate_advanced_trend(self, values: List[float]) -> Tuple[str, float, float]:
        """Calculate advanced trend analysis."""
        if len(values) < 3:
            return "insufficient_data", 0.0, 0.0
        
        # Linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate R-squared for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction and strength
        trend_strength = abs(slope) / (max(values) - min(values)) if max(values) != min(values) else 0
        
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        return trend_direction, min(trend_strength, 1.0), max(0.0, min(r_squared, 1.0))
    
    def _generate_forecast(self, values: List[float], timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Generate simple forecast based on trend."""
        if len(values) < 3:
            return []
        
        # Simple linear extrapolation
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Forecast next 7 days
        forecast_points = []
        last_timestamp = timestamps[-1]
        
        for i in range(1, 8):
            future_timestamp = last_timestamp + timedelta(days=i)
            future_value = slope * (len(values) + i - 1) + intercept
            
            forecast_points.append({
                "timestamp": future_timestamp.isoformat(),
                "predicted_value": max(0, future_value),  # Ensure non-negative
                "confidence": max(0, 1 - (i * 0.1))  # Decreasing confidence
            })
        
        return forecast_points

class PatternRecognizer:
    """Recognizes patterns in prompt usage and performance."""
    
    def __init__(self):
        self.pattern_types = ['seasonal', 'cyclical', 'anomaly', 'usage_spike', 'performance_drop']
    
    async def recognize_patterns(
        self,
        prompt_ids: List[str],
        days: int = 30
    ) -> List[PatternRecognition]:
        """Recognize patterns across multiple prompts."""
        patterns = []
        
        try:
            with get_db_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                for prompt_id in prompt_ids:
                    # Get usage analytics
                    analytics = db.query(UsageAnalytics).filter(
                        and_(
                            UsageAnalytics.prompt_id == prompt_id,
                            UsageAnalytics.analysis_period_start >= cutoff_date
                        )
                    ).all()
                    
                    if analytics:
                        prompt_patterns = self._analyze_prompt_patterns(prompt_id, analytics)
                        patterns.extend(prompt_patterns)
                
                return patterns
                
        except Exception as e:
            logger.error(f"Error recognizing patterns: {str(e)}")
            return []
    
    def _analyze_prompt_patterns(
        self,
        prompt_id: str,
        analytics: List[UsageAnalytics]
    ) -> List[PatternRecognition]:
        """Analyze patterns for a specific prompt."""
        patterns = []
        
        # Analyze hourly usage patterns
        hourly_pattern = self._detect_hourly_patterns(analytics)
        if hourly_pattern:
            patterns.append(PatternRecognition(
                pattern_type="cyclical",
                pattern_description=f"Daily usage cycle detected for prompt {prompt_id}",
                confidence_score=hourly_pattern["confidence"],
                affected_prompts=[prompt_id],
                recommendations=[
                    "Consider scheduling optimization during low-usage hours",
                    "Prepare for peak usage periods"
                ]
            ))
        
        # Analyze usage anomalies
        anomaly_pattern = self._detect_usage_anomalies(analytics)
        if anomaly_pattern:
            patterns.append(PatternRecognition(
                pattern_type="anomaly",
                pattern_description=f"Usage anomaly detected for prompt {prompt_id}",
                confidence_score=anomaly_pattern["confidence"],
                affected_prompts=[prompt_id],
                recommendations=[
                    "Investigate cause of usage anomaly",
                    "Monitor for continued unusual patterns"
                ]
            ))
        
        return patterns
    
    def _detect_hourly_patterns(self, analytics: List[UsageAnalytics]) -> Optional[Dict[str, Any]]:
        """Detect hourly usage patterns."""
        if not analytics:
            return None
        
        # Combine hourly patterns
        combined_hourly = defaultdict(int)
        for a in analytics:
            if a.hourly_patterns:
                for hour, count in a.hourly_patterns.items():
                    combined_hourly[int(hour)] += count
        
        if len(combined_hourly) < 12:  # Need sufficient data
            return None
        
        # Calculate pattern strength
        hourly_values = [combined_hourly.get(i, 0) for i in range(24)]
        pattern_strength = np.std(hourly_values) / np.mean(hourly_values) if np.mean(hourly_values) > 0 else 0
        
        if pattern_strength > 0.5:  # Threshold for significant pattern
            return {
                "confidence": min(pattern_strength, 1.0),
                "peak_hours": sorted(combined_hourly.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        
        return None
    
    def _detect_usage_anomalies(self, analytics: List[UsageAnalytics]) -> Optional[Dict[str, Any]]:
        """Detect usage anomalies."""
        if len(analytics) < 7:  # Need at least a week of data
            return None
        
        # Get daily usage counts
        daily_usage = []
        for a in analytics:
            if a.daily_usage:
                daily_total = sum(a.daily_usage.values())
                daily_usage.append(daily_total)
        
        if len(daily_usage) < 7:
            return None
        
        # Calculate z-scores to detect anomalies
        mean_usage = np.mean(daily_usage)
        std_usage = np.std(daily_usage)
        
        if std_usage == 0:
            return None
        
        z_scores = [(usage - mean_usage) / std_usage for usage in daily_usage]
        anomaly_threshold = 2.0  # Standard threshold for anomalies
        
        anomalies = [abs(z) > anomaly_threshold for z in z_scores]
        
        if any(anomalies):
            anomaly_count = sum(anomalies)
            confidence = min(anomaly_count / len(daily_usage), 1.0)
            
            return {
                "confidence": confidence,
                "anomaly_days": anomaly_count,
                "total_days": len(daily_usage)
            }
        
        return None