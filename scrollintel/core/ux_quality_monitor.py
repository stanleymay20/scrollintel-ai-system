"""
User Experience Quality Monitoring System

This module provides comprehensive user experience monitoring with automatic
optimization, satisfaction tracking, and proactive user assistance.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)

class UXMetricType(Enum):
    PERFORMANCE = "performance"
    USABILITY = "usability"
    SATISFACTION = "satisfaction"
    ACCESSIBILITY = "accessibility"
    ENGAGEMENT = "engagement"

class OptimizationAction(Enum):
    PRELOAD_CONTENT = "preload_content"
    REDUCE_COMPLEXITY = "reduce_complexity"
    IMPROVE_FEEDBACK = "improve_feedback"
    ENHANCE_GUIDANCE = "enhance_guidance"
    OPTIMIZE_LAYOUT = "optimize_layout"

@dataclass
class UXMetric:
    """User experience metric data structure"""
    id: str
    user_id: str
    session_id: str
    metric_type: UXMetricType
    metric_name: str
    value: float
    context: Dict[str, Any]
    timestamp: datetime
    
@dataclass
class UserSession:
    """User session tracking"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    page_views: int
    actions_completed: int
    errors_encountered: int
    satisfaction_score: float
    engagement_score: float
    context: Dict[str, Any]

@dataclass
class UXOptimization:
    """UX optimization recommendation"""
    id: str
    action: OptimizationAction
    target_metric: str
    current_value: float
    expected_improvement: float
    confidence: float
    description: str
    implementation_priority: int
    estimated_impact: str

class UXQualityMonitor:
    """
    User Experience Quality Monitoring System with automatic optimization
    """
    
    def __init__(self):
        self.ux_metrics = deque(maxlen=50000)
        self.user_sessions = {}
        self.optimization_rules = {}
        self.active_optimizations = {}
        self.satisfaction_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.6
        }
        self.monitoring_active = False
        
        # Initialize optimization rules
        self._setup_optimization_rules()
    
    def _setup_optimization_rules(self):
        """Setup UX optimization rules"""
        self.optimization_rules = {
            "slow_page_load": {
                "condition": lambda metrics: any(m.metric_name == "page_load_time" and m.value > 3000 for m in metrics),
                "action": OptimizationAction.PRELOAD_CONTENT,
                "description": "Preload critical content to improve page load times",
                "priority": 1
            },
            "high_error_rate": {
                "condition": lambda metrics: any(m.metric_name == "error_rate" and m.value > 0.05 for m in metrics),
                "action": OptimizationAction.IMPROVE_FEEDBACK,
                "description": "Improve error messages and user feedback",
                "priority": 1
            },
            "low_task_completion": {
                "condition": lambda metrics: any(m.metric_name == "task_completion_rate" and m.value < 0.8 for m in metrics),
                "action": OptimizationAction.ENHANCE_GUIDANCE,
                "description": "Enhance user guidance and help systems",
                "priority": 2
            },
            "high_bounce_rate": {
                "condition": lambda metrics: any(m.metric_name == "bounce_rate" and m.value > 0.6 for m in metrics),
                "action": OptimizationAction.OPTIMIZE_LAYOUT,
                "description": "Optimize page layout and content presentation",
                "priority": 2
            },
            "complex_navigation": {
                "condition": lambda metrics: any(m.metric_name == "navigation_complexity" and m.value > 0.7 for m in metrics),
                "action": OptimizationAction.REDUCE_COMPLEXITY,
                "description": "Simplify navigation and reduce cognitive load",
                "priority": 3
            }
        }
    
    async def start_monitoring(self):
        """Start UX quality monitoring"""
        self.monitoring_active = True
        logger.info("UX Quality Monitor started")
        
        tasks = [
            asyncio.create_task(self._monitor_user_sessions()),
            asyncio.create_task(self._analyze_ux_patterns()),
            asyncio.create_task(self._generate_optimizations()),
            asyncio.create_task(self._track_satisfaction_trends())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """Stop UX quality monitoring"""
        self.monitoring_active = False
        logger.info("UX Quality Monitor stopped")
    
    def record_ux_metric(self, user_id: str, session_id: str, metric_type: UXMetricType, 
                        metric_name: str, value: float, context: Dict[str, Any] = None):
        """Record a UX metric"""
        metric = UXMetric(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            context=context or {},
            timestamp=datetime.now()
        )
        
        self.ux_metrics.append(metric)
        
        # Update session if exists
        if session_id in self.user_sessions:
            self.user_sessions[session_id].last_activity = datetime.now()
            
            # Update session metrics based on the recorded metric
            if metric_name == "page_view":
                self.user_sessions[session_id].page_views += 1
            elif metric_name == "action_completed":
                self.user_sessions[session_id].actions_completed += 1
            elif metric_name == "error_encountered":
                self.user_sessions[session_id].errors_encountered += 1
            elif metric_name == "satisfaction_rating":
                self.user_sessions[session_id].satisfaction_score = value
    
    def start_user_session(self, user_id: str, session_id: str = None, context: Dict[str, Any] = None) -> str:
        """Start tracking a user session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            page_views=0,
            actions_completed=0,
            errors_encountered=0,
            satisfaction_score=0.8,  # Default neutral satisfaction
            engagement_score=0.5,   # Default engagement
            context=context or {}
        )
        
        self.user_sessions[session_id] = session
        return session_id
    
    def end_user_session(self, session_id: str, final_satisfaction: float = None):
        """End a user session"""
        if session_id in self.user_sessions:
            session = self.user_sessions[session_id]
            
            if final_satisfaction is not None:
                session.satisfaction_score = final_satisfaction
            
            # Calculate final engagement score
            session_duration = (datetime.now() - session.start_time).total_seconds()
            if session_duration > 0:
                # Engagement based on actions per minute and session duration
                actions_per_minute = (session.actions_completed / session_duration) * 60
                engagement = min(1.0, (actions_per_minute * 0.1) + (min(session_duration, 1800) / 1800) * 0.5)
                session.engagement_score = engagement
            
            # Record final session metrics
            self.record_ux_metric(
                session.user_id, session_id, UXMetricType.SATISFACTION,
                "final_satisfaction", session.satisfaction_score
            )
            self.record_ux_metric(
                session.user_id, session_id, UXMetricType.ENGAGEMENT,
                "final_engagement", session.engagement_score
            )
    
    async def _monitor_user_sessions(self):
        """Monitor active user sessions"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                inactive_threshold = timedelta(minutes=30)
                
                # Check for inactive sessions
                inactive_sessions = []
                for session_id, session in self.user_sessions.items():
                    if current_time - session.last_activity > inactive_threshold:
                        inactive_sessions.append(session_id)
                
                # End inactive sessions
                for session_id in inactive_sessions:
                    self.end_user_session(session_id)
                    del self.user_sessions[session_id]
                
                # Calculate real-time UX metrics
                active_sessions = len(self.user_sessions)
                if active_sessions > 0:
                    avg_satisfaction = statistics.mean([s.satisfaction_score for s in self.user_sessions.values()])
                    avg_engagement = statistics.mean([s.engagement_score for s in self.user_sessions.values()])
                    
                    # Record aggregate metrics
                    self.record_ux_metric(
                        "system", "aggregate", UXMetricType.SATISFACTION,
                        "average_satisfaction", avg_satisfaction
                    )
                    self.record_ux_metric(
                        "system", "aggregate", UXMetricType.ENGAGEMENT,
                        "average_engagement", avg_engagement
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring user sessions: {e}")
                await asyncio.sleep(120)
    
    async def _analyze_ux_patterns(self):
        """Analyze UX patterns and identify issues"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                recent_cutoff = current_time - timedelta(hours=1)
                
                # Get recent metrics
                recent_metrics = [m for m in self.ux_metrics if m.timestamp >= recent_cutoff]
                
                if len(recent_metrics) < 10:
                    await asyncio.sleep(300)
                    continue
                
                # Analyze patterns by metric type
                patterns = {}
                
                for metric_type in UXMetricType:
                    type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
                    
                    if type_metrics:
                        # Group by metric name
                        by_name = defaultdict(list)
                        for metric in type_metrics:
                            by_name[metric.metric_name].append(metric.value)
                        
                        patterns[metric_type.value] = {}
                        for name, values in by_name.items():
                            if len(values) >= 3:
                                patterns[metric_type.value][name] = {
                                    "average": statistics.mean(values),
                                    "trend": self._calculate_trend(values),
                                    "variance": statistics.variance(values) if len(values) > 1 else 0,
                                    "latest": values[-1]
                                }
                
                # Identify concerning patterns
                concerns = self._identify_ux_concerns(patterns)
                
                # Log significant patterns
                if concerns:
                    logger.info(f"UX concerns identified: {json.dumps(concerns, indent=2)}")
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error analyzing UX patterns: {e}")
                await asyncio.sleep(600)
    
    async def _generate_optimizations(self):
        """Generate UX optimization recommendations"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                recent_cutoff = current_time - timedelta(hours=2)
                
                # Get recent metrics for optimization analysis
                recent_metrics = [m for m in self.ux_metrics if m.timestamp >= recent_cutoff]
                
                if len(recent_metrics) < 20:
                    await asyncio.sleep(600)
                    continue
                
                # Check optimization rules
                for rule_name, rule in self.optimization_rules.items():
                    if rule["condition"](recent_metrics):
                        # Generate optimization recommendation
                        optimization_id = f"{rule_name}_{int(current_time.timestamp())}"
                        
                        if optimization_id not in self.active_optimizations:
                            # Calculate expected improvement
                            current_performance = self._get_current_performance(rule["action"])
                            expected_improvement = self._estimate_improvement(rule["action"], current_performance)
                            
                            optimization = UXOptimization(
                                id=optimization_id,
                                action=rule["action"],
                                target_metric=rule_name,
                                current_value=current_performance,
                                expected_improvement=expected_improvement,
                                confidence=0.8,  # Default confidence
                                description=rule["description"],
                                implementation_priority=rule["priority"],
                                estimated_impact="medium"
                            )
                            
                            self.active_optimizations[optimization_id] = optimization
                            await self._trigger_optimization(optimization)
                
                await asyncio.sleep(600)  # Generate optimizations every 10 minutes
                
            except Exception as e:
                logger.error(f"Error generating optimizations: {e}")
                await asyncio.sleep(1200)
    
    async def _track_satisfaction_trends(self):
        """Track user satisfaction trends"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                # Calculate satisfaction trends over different time periods
                periods = [
                    ("1h", timedelta(hours=1)),
                    ("6h", timedelta(hours=6)),
                    ("24h", timedelta(hours=24))
                ]
                
                trends = {}
                
                for period_name, period_delta in periods:
                    cutoff_time = current_time - period_delta
                    period_metrics = [
                        m for m in self.ux_metrics 
                        if m.timestamp >= cutoff_time and m.metric_name in ["satisfaction_rating", "final_satisfaction"]
                    ]
                    
                    if period_metrics:
                        satisfaction_values = [m.value for m in period_metrics]
                        trends[period_name] = {
                            "average": statistics.mean(satisfaction_values),
                            "count": len(satisfaction_values),
                            "trend": self._calculate_trend(satisfaction_values),
                            "distribution": self._calculate_satisfaction_distribution(satisfaction_values)
                        }
                
                # Log satisfaction trends
                if trends:
                    logger.info(f"Satisfaction trends: {json.dumps(trends, indent=2)}")
                
                # Check for concerning trends
                if "1h" in trends and trends["1h"]["average"] < 0.7:
                    logger.warning(f"Low satisfaction detected: {trends['1h']['average']:.2f}")
                
                await asyncio.sleep(1800)  # Track trends every 30 minutes
                
            except Exception as e:
                logger.error(f"Error tracking satisfaction trends: {e}")
                await asyncio.sleep(3600)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 3:
            return "stable"
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        diff_threshold = 0.05  # 5% change threshold
        
        if second_avg > first_avg * (1 + diff_threshold):
            return "increasing"
        elif second_avg < first_avg * (1 - diff_threshold):
            return "decreasing"
        else:
            return "stable"
    
    def _identify_ux_concerns(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify UX concerns from patterns"""
        concerns = []
        
        # Check satisfaction patterns
        if "satisfaction" in patterns:
            for metric_name, data in patterns["satisfaction"].items():
                if data["average"] < 0.7:
                    concerns.append(f"Low {metric_name}: {data['average']:.2f}")
                if data["trend"] == "decreasing":
                    concerns.append(f"Declining {metric_name}")
        
        # Check performance patterns
        if "performance" in patterns:
            for metric_name, data in patterns["performance"].items():
                if "load_time" in metric_name and data["average"] > 3000:
                    concerns.append(f"Slow {metric_name}: {data['average']:.0f}ms")
                if data["trend"] == "increasing" and "time" in metric_name:
                    concerns.append(f"Increasing {metric_name}")
        
        # Check usability patterns
        if "usability" in patterns:
            for metric_name, data in patterns["usability"].items():
                if "error" in metric_name and data["average"] > 0.05:
                    concerns.append(f"High {metric_name}: {data['average']:.1%}")
                if "completion" in metric_name and data["average"] < 0.8:
                    concerns.append(f"Low {metric_name}: {data['average']:.1%}")
        
        return concerns
    
    def _get_current_performance(self, action: OptimizationAction) -> float:
        """Get current performance metric for optimization action"""
        # This would be more sophisticated in a real implementation
        recent_metrics = list(self.ux_metrics)[-100:]  # Last 100 metrics
        
        if action == OptimizationAction.PRELOAD_CONTENT:
            load_times = [m.value for m in recent_metrics if "load_time" in m.metric_name]
            return statistics.mean(load_times) if load_times else 2000.0
        
        elif action == OptimizationAction.IMPROVE_FEEDBACK:
            error_rates = [m.value for m in recent_metrics if "error_rate" in m.metric_name]
            return statistics.mean(error_rates) if error_rates else 0.03
        
        elif action == OptimizationAction.ENHANCE_GUIDANCE:
            completion_rates = [m.value for m in recent_metrics if "completion" in m.metric_name]
            return statistics.mean(completion_rates) if completion_rates else 0.75
        
        return 0.5  # Default value
    
    def _estimate_improvement(self, action: OptimizationAction, current_value: float) -> float:
        """Estimate improvement from optimization action"""
        # Conservative improvement estimates
        improvement_factors = {
            OptimizationAction.PRELOAD_CONTENT: 0.3,  # 30% improvement in load times
            OptimizationAction.IMPROVE_FEEDBACK: 0.5,  # 50% reduction in error impact
            OptimizationAction.ENHANCE_GUIDANCE: 0.2,  # 20% improvement in completion
            OptimizationAction.OPTIMIZE_LAYOUT: 0.15,  # 15% improvement in usability
            OptimizationAction.REDUCE_COMPLEXITY: 0.25  # 25% improvement in navigation
        }
        
        factor = improvement_factors.get(action, 0.1)
        return current_value * factor
    
    def _calculate_satisfaction_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate satisfaction distribution"""
        distribution = {level: 0 for level in self.satisfaction_thresholds.keys()}
        
        for value in values:
            if value >= self.satisfaction_thresholds['excellent']:
                distribution['excellent'] += 1
            elif value >= self.satisfaction_thresholds['good']:
                distribution['good'] += 1
            elif value >= self.satisfaction_thresholds['acceptable']:
                distribution['acceptable'] += 1
            else:
                distribution['poor'] += 1
        
        # Convert to percentages
        total = len(values)
        if total > 0:
            distribution = {k: (v / total) * 100 for k, v in distribution.items()}
        
        return distribution
    
    async def _trigger_optimization(self, optimization: UXOptimization):
        """Trigger a UX optimization"""
        logger.info(f"UX OPTIMIZATION TRIGGERED: {optimization.description}")
        
        optimization_data = {
            "optimization_id": optimization.id,
            "action": optimization.action.value,
            "description": optimization.description,
            "priority": optimization.implementation_priority,
            "expected_improvement": optimization.expected_improvement,
            "confidence": optimization.confidence
        }
        
        logger.info(f"Optimization data: {json.dumps(optimization_data, indent=2)}")
    
    # Public API methods
    
    def get_ux_dashboard(self) -> Dict[str, Any]:
        """Get UX quality dashboard data"""
        current_time = datetime.now()
        
        # Active sessions
        active_sessions = len(self.user_sessions)
        
        # Recent satisfaction
        recent_satisfaction_metrics = [
            m for m in self.ux_metrics 
            if m.timestamp >= current_time - timedelta(hours=1) and 
               m.metric_name in ["satisfaction_rating", "final_satisfaction"]
        ]
        
        avg_satisfaction = 0.8  # Default
        if recent_satisfaction_metrics:
            avg_satisfaction = statistics.mean([m.value for m in recent_satisfaction_metrics])
        
        # Recent performance
        recent_performance_metrics = [
            m for m in self.ux_metrics 
            if m.timestamp >= current_time - timedelta(hours=1) and 
               "load_time" in m.metric_name
        ]
        
        avg_load_time = 2000  # Default
        if recent_performance_metrics:
            avg_load_time = statistics.mean([m.value for m in recent_performance_metrics])
        
        # Active optimizations
        active_optimizations = len(self.active_optimizations)
        
        return {
            "timestamp": current_time.isoformat(),
            "active_sessions": active_sessions,
            "average_satisfaction": avg_satisfaction,
            "satisfaction_level": self._get_satisfaction_level(avg_satisfaction),
            "average_load_time": avg_load_time,
            "active_optimizations": active_optimizations,
            "total_metrics_collected": len(self.ux_metrics),
            "recent_metrics_count": len([m for m in self.ux_metrics if m.timestamp >= current_time - timedelta(hours=1)])
        }
    
    def _get_satisfaction_level(self, score: float) -> str:
        """Get satisfaction level description"""
        if score >= self.satisfaction_thresholds['excellent']:
            return "excellent"
        elif score >= self.satisfaction_thresholds['good']:
            return "good"
        elif score >= self.satisfaction_thresholds['acceptable']:
            return "acceptable"
        else:
            return "poor"
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations"""
        return [asdict(opt) for opt in sorted(
            self.active_optimizations.values(),
            key=lambda x: x.implementation_priority
        )]
    
    def get_user_session_summary(self, user_id: str = None) -> Dict[str, Any]:
        """Get user session summary"""
        sessions = list(self.user_sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        if not sessions:
            return {"message": "No active sessions found"}
        
        return {
            "total_sessions": len(sessions),
            "average_satisfaction": statistics.mean([s.satisfaction_score for s in sessions]),
            "average_engagement": statistics.mean([s.engagement_score for s in sessions]),
            "total_page_views": sum(s.page_views for s in sessions),
            "total_actions": sum(s.actions_completed for s in sessions),
            "total_errors": sum(s.errors_encountered for s in sessions)
        }

# Global UX quality monitor instance
ux_quality_monitor = UXQualityMonitor()