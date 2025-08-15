"""
Comprehensive Feedback System for ScrollIntel

This module provides user feedback collection, analysis, and real-time
performance monitoring with audit trail and compliance reporting.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
import statistics
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback"""
    RATING = "rating"
    COMMENT = "comment"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    PERFORMANCE_ISSUE = "performance_issue"
    USABILITY_FEEDBACK = "usability_feedback"

class FeedbackCategory(Enum):
    """Feedback categories"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    SPEED = "speed"
    USABILITY = "usability"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"

class FeedbackSentiment(Enum):
    """Sentiment analysis results"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UserFeedback:
    """Individual user feedback entry"""
    feedback_id: str
    session_id: str
    user_id: Optional[str]
    agent_id: str
    request_id: str
    feedback_type: FeedbackType
    rating: Optional[int] = None  # 1-5 scale
    categories: List[FeedbackCategory] = field(default_factory=list)
    comment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    sentiment: Optional[FeedbackSentiment] = None

@dataclass
class FeedbackAnalysis:
    """Analysis results for feedback data"""
    analysis_id: str
    time_period: Dict[str, datetime]
    total_feedback_count: int
    average_rating: float
    rating_distribution: Dict[int, int]
    category_breakdown: Dict[str, Dict[str, Any]]
    sentiment_distribution: Dict[str, int]
    trends: List[Dict[str, Any]]
    recommendations: List[str]
    quality_score: float
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceAlert:
    """Performance alert based on feedback analysis"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    agent_id: Optional[str]
    metric_name: str
    current_value: float
    threshold_value: float
    description: str
    recommendations: List[str]
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class QualityMetrics:
    """Quality metrics derived from feedback"""
    metric_id: str
    agent_id: str
    time_window: timedelta
    accuracy_score: float
    relevance_score: float
    speed_score: float
    usability_score: float
    overall_quality_score: float
    feedback_volume: int
    confidence_level: float
    calculated_at: datetime = field(default_factory=datetime.now)

class SentimentAnalyzer:
    """Simple sentiment analysis for feedback comments"""
    
    def __init__(self):
        # Simple keyword-based sentiment analysis
        self.positive_keywords = {
            'excellent', 'great', 'good', 'amazing', 'perfect', 'helpful',
            'useful', 'fast', 'accurate', 'clear', 'easy', 'love', 'like',
            'satisfied', 'impressed', 'wonderful', 'fantastic'
        }
        
        self.negative_keywords = {
            'bad', 'terrible', 'awful', 'horrible', 'useless', 'slow',
            'confusing', 'unclear', 'wrong', 'inaccurate', 'hate', 'dislike',
            'frustrated', 'disappointed', 'broken', 'failed', 'error'
        }
        
        self.neutral_keywords = {
            'okay', 'fine', 'average', 'normal', 'standard', 'typical'
        }
    
    def analyze_sentiment(self, text: str) -> FeedbackSentiment:
        """Analyze sentiment of feedback text"""
        if not text:
            return FeedbackSentiment.NEUTRAL
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        neutral_count = sum(1 for word in words if word in self.neutral_keywords)
        
        if positive_count > negative_count and positive_count > 0:
            return FeedbackSentiment.POSITIVE
        elif negative_count > positive_count and negative_count > 0:
            return FeedbackSentiment.NEGATIVE
        elif positive_count == negative_count and both > 0:
            return FeedbackSentiment.MIXED
        else:
            return FeedbackSentiment.NEUTRAL

class FeedbackCollector:
    """Collects and stores user feedback"""
    
    def __init__(self):
        self._feedback_storage: Dict[str, UserFeedback] = {}
        self._feedback_by_agent: Dict[str, List[str]] = defaultdict(list)
        self._feedback_by_session: Dict[str, List[str]] = defaultdict(list)
        self._feedback_queue: deque = deque()
        self._lock = threading.RLock()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    async def collect_feedback(self, feedback: UserFeedback) -> bool:
        """Collect user feedback"""
        try:
            with self._lock:
                # Analyze sentiment if comment provided
                if feedback.comment:
                    feedback.sentiment = self.sentiment_analyzer.analyze_sentiment(feedback.comment)
                
                # Store feedback
                self._feedback_storage[feedback.feedback_id] = feedback
                self._feedback_by_agent[feedback.agent_id].append(feedback.feedback_id)
                self._feedback_by_session[feedback.session_id].append(feedback.feedback_id)
                
                # Add to processing queue
                self._feedback_queue.append(feedback.feedback_id)
                
                logger.info(f"Collected feedback {feedback.feedback_id} for agent {feedback.agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            return False
    
    def get_feedback(self, feedback_id: str) -> Optional[UserFeedback]:
        """Get feedback by ID"""
        with self._lock:
            return self._feedback_storage.get(feedback_id)
    
    def get_feedback_by_agent(self, agent_id: str, 
                            time_window: Optional[timedelta] = None) -> List[UserFeedback]:
        """Get feedback for a specific agent"""
        with self._lock:
            feedback_ids = self._feedback_by_agent.get(agent_id, [])
            feedback_list = []
            
            cutoff_time = datetime.now() - time_window if time_window else None
            
            for feedback_id in feedback_ids:
                feedback = self._feedback_storage.get(feedback_id)
                if feedback and (not cutoff_time or feedback.timestamp >= cutoff_time):
                    feedback_list.append(feedback)
            
            return feedback_list
    
    def get_feedback_by_session(self, session_id: str) -> List[UserFeedback]:
        """Get feedback for a specific session"""
        with self._lock:
            feedback_ids = self._feedback_by_session.get(session_id, [])
            return [self._feedback_storage[fid] for fid in feedback_ids 
                   if fid in self._feedback_storage]
    
    def get_recent_feedback(self, limit: int = 100, 
                          time_window: Optional[timedelta] = None) -> List[UserFeedback]:
        """Get recent feedback entries"""
        with self._lock:
            all_feedback = list(self._feedback_storage.values())
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                all_feedback = [f for f in all_feedback if f.timestamp >= cutoff_time]
            
            # Sort by timestamp and limit
            all_feedback.sort(key=lambda x: x.timestamp, reverse=True)
            return all_feedback[:limit]
    
    def get_unprocessed_feedback(self) -> List[str]:
        """Get IDs of unprocessed feedback"""
        with self._lock:
            return list(self._feedback_queue)
    
    def mark_processed(self, feedback_id: str):
        """Mark feedback as processed"""
        with self._lock:
            if feedback_id in self._feedback_storage:
                self._feedback_storage[feedback_id].processed = True
                
                # Remove from queue
                try:
                    self._feedback_queue.remove(feedback_id)
                except ValueError:
                    pass  # Already removed

class FeedbackAnalyzer:
    """Analyzes feedback data and generates insights"""
    
    def __init__(self, collector: FeedbackCollector):
        self.collector = collector
        self._analysis_cache: Dict[str, FeedbackAnalysis] = {}
        self._cache_ttl = timedelta(minutes=15)
    
    async def analyze_feedback(self, agent_id: Optional[str] = None,
                             time_window: timedelta = timedelta(hours=24)) -> FeedbackAnalysis:
        """Analyze feedback data"""
        
        # Create cache key
        cache_key = f"{agent_id or 'all'}_{time_window.total_seconds()}"
        
        # Check cache
        if cache_key in self._analysis_cache:
            cached_analysis = self._analysis_cache[cache_key]
            if datetime.now() - cached_analysis.generated_at < self._cache_ttl:
                return cached_analysis
        
        # Get feedback data
        if agent_id:
            feedback_data = self.collector.get_feedback_by_agent(agent_id, time_window)
        else:
            feedback_data = self.collector.get_recent_feedback(time_window=time_window)
        
        if not feedback_data:
            return FeedbackAnalysis(
                analysis_id=str(uuid.uuid4()),
                time_period={
                    "start": datetime.now() - time_window,
                    "end": datetime.now()
                },
                total_feedback_count=0,
                average_rating=0.0,
                rating_distribution={},
                category_breakdown={},
                sentiment_distribution={},
                trends=[],
                recommendations=["No feedback data available for analysis"],
                quality_score=0.0
            )
        
        # Perform analysis
        analysis = await self._perform_analysis(feedback_data, time_window)
        
        # Cache result
        self._analysis_cache[cache_key] = analysis
        
        return analysis
    
    async def _perform_analysis(self, feedback_data: List[UserFeedback], 
                              time_window: timedelta) -> FeedbackAnalysis:
        """Perform detailed feedback analysis"""
        
        # Basic statistics
        total_count = len(feedback_data)
        ratings = [f.rating for f in feedback_data if f.rating is not None]
        average_rating = statistics.mean(ratings) if ratings else 0.0
        
        # Rating distribution
        rating_distribution = defaultdict(int)
        for rating in ratings:
            rating_distribution[rating] += 1
        
        # Category breakdown
        category_breakdown = {}
        for category in FeedbackCategory:
            category_feedback = [f for f in feedback_data if category in f.categories]
            category_ratings = [f.rating for f in category_feedback if f.rating is not None]
            
            category_breakdown[category.value] = {
                "count": len(category_feedback),
                "average_rating": statistics.mean(category_ratings) if category_ratings else 0.0,
                "percentage": (len(category_feedback) / total_count * 100) if total_count > 0 else 0.0
            }
        
        # Sentiment distribution
        sentiment_distribution = defaultdict(int)
        for feedback in feedback_data:
            if feedback.sentiment:
                sentiment_distribution[feedback.sentiment.value] += 1
        
        # Trend analysis
        trends = await self._analyze_trends(feedback_data, time_window)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            average_rating, category_breakdown, sentiment_distribution
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            average_rating, category_breakdown, sentiment_distribution
        )
        
        return FeedbackAnalysis(
            analysis_id=str(uuid.uuid4()),
            time_period={
                "start": datetime.now() - time_window,
                "end": datetime.now()
            },
            total_feedback_count=total_count,
            average_rating=average_rating,
            rating_distribution=dict(rating_distribution),
            category_breakdown=category_breakdown,
            sentiment_distribution=dict(sentiment_distribution),
            trends=trends,
            recommendations=recommendations,
            quality_score=quality_score
        )
    
    async def _analyze_trends(self, feedback_data: List[UserFeedback], 
                            time_window: timedelta) -> List[Dict[str, Any]]:
        """Analyze trends in feedback data"""
        trends = []
        
        if len(feedback_data) < 10:  # Need minimum data for trend analysis
            return trends
        
        # Sort by timestamp
        sorted_feedback = sorted(feedback_data, key=lambda x: x.timestamp)
        
        # Split into time buckets for trend analysis
        bucket_count = min(10, len(sorted_feedback) // 2)
        bucket_size = len(sorted_feedback) // bucket_count
        
        bucket_ratings = []
        for i in range(bucket_count):
            start_idx = i * bucket_size
            end_idx = start_idx + bucket_size if i < bucket_count - 1 else len(sorted_feedback)
            
            bucket_feedback = sorted_feedback[start_idx:end_idx]
            bucket_rating_values = [f.rating for f in bucket_feedback if f.rating is not None]
            
            if bucket_rating_values:
                bucket_ratings.append(statistics.mean(bucket_rating_values))
        
        # Analyze rating trend
        if len(bucket_ratings) >= 3:
            recent_avg = statistics.mean(bucket_ratings[-3:])
            earlier_avg = statistics.mean(bucket_ratings[:3])
            
            if recent_avg > earlier_avg + 0.2:
                trends.append({
                    "type": "rating_improvement",
                    "description": "User ratings have been improving over time",
                    "confidence": 0.8,
                    "change": recent_avg - earlier_avg
                })
            elif recent_avg < earlier_avg - 0.2:
                trends.append({
                    "type": "rating_decline",
                    "description": "User ratings have been declining over time",
                    "confidence": 0.8,
                    "change": recent_avg - earlier_avg
                })
        
        # Analyze sentiment trends
        recent_feedback = sorted_feedback[-len(sorted_feedback)//3:]
        earlier_feedback = sorted_feedback[:len(sorted_feedback)//3]
        
        recent_negative = sum(1 for f in recent_feedback 
                            if f.sentiment == FeedbackSentiment.NEGATIVE)
        earlier_negative = sum(1 for f in earlier_feedback 
                             if f.sentiment == FeedbackSentiment.NEGATIVE)
        
        recent_negative_pct = recent_negative / len(recent_feedback) if recent_feedback else 0
        earlier_negative_pct = earlier_negative / len(earlier_feedback) if earlier_feedback else 0
        
        if recent_negative_pct > earlier_negative_pct + 0.1:
            trends.append({
                "type": "sentiment_decline",
                "description": "Negative sentiment has increased recently",
                "confidence": 0.7,
                "change": recent_negative_pct - earlier_negative_pct
            })
        
        return trends
    
    def _generate_recommendations(self, average_rating: float, 
                                category_breakdown: Dict[str, Dict[str, Any]],
                                sentiment_distribution: Dict[str, int]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Rating-based recommendations
        if average_rating < 3.0:
            recommendations.append("Overall user satisfaction is low. Consider reviewing system performance and user experience.")
        elif average_rating < 4.0:
            recommendations.append("User satisfaction is moderate. Focus on addressing specific pain points.")
        
        # Category-based recommendations
        for category, data in category_breakdown.items():
            if data["average_rating"] < 3.0 and data["count"] > 5:
                recommendations.append(f"Users are dissatisfied with {category}. This area needs immediate attention.")
        
        # Sentiment-based recommendations
        negative_count = sentiment_distribution.get("negative", 0)
        total_sentiment = sum(sentiment_distribution.values())
        
        if total_sentiment > 0 and negative_count / total_sentiment > 0.3:
            recommendations.append("High negative sentiment detected. Review recent user comments for specific issues.")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Continue monitoring user feedback and maintain current quality levels.")
        
        return recommendations
    
    def _calculate_quality_score(self, average_rating: float,
                               category_breakdown: Dict[str, Dict[str, Any]],
                               sentiment_distribution: Dict[str, int]) -> float:
        """Calculate overall quality score"""
        
        # Rating component (40% weight)
        rating_score = (average_rating / 5.0) * 0.4
        
        # Category component (40% weight)
        category_scores = []
        for category_data in category_breakdown.values():
            if category_data["count"] > 0:
                category_scores.append(category_data["average_rating"] / 5.0)
        
        category_score = (statistics.mean(category_scores) if category_scores else 0.5) * 0.4
        
        # Sentiment component (20% weight)
        total_sentiment = sum(sentiment_distribution.values())
        if total_sentiment > 0:
            positive_pct = sentiment_distribution.get("positive", 0) / total_sentiment
            negative_pct = sentiment_distribution.get("negative", 0) / total_sentiment
            sentiment_score = (positive_pct - negative_pct + 1) / 2 * 0.2
        else:
            sentiment_score = 0.1
        
        return min(1.0, max(0.0, rating_score + category_score + sentiment_score))

class AlertManager:
    """Manages performance alerts based on feedback analysis"""
    
    def __init__(self, analyzer: FeedbackAnalyzer):
        self.analyzer = analyzer
        self._alerts: Dict[str, PerformanceAlert] = {}
        self._alert_handlers: List[Callable] = []
        self._thresholds = {
            "average_rating": {"warning": 3.5, "critical": 2.5},
            "quality_score": {"warning": 0.6, "critical": 0.4},
            "negative_sentiment_pct": {"warning": 0.3, "critical": 0.5}
        }
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self._alert_handlers.append(handler)
    
    async def check_alerts(self, agent_id: Optional[str] = None):
        """Check for alert conditions"""
        analysis = await self.analyzer.analyze_feedback(agent_id)
        
        alerts_triggered = []
        
        # Check average rating
        if analysis.average_rating <= self._thresholds["average_rating"]["critical"]:
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="low_rating",
                severity=AlertSeverity.CRITICAL,
                agent_id=agent_id,
                metric_name="average_rating",
                current_value=analysis.average_rating,
                threshold_value=self._thresholds["average_rating"]["critical"],
                description=f"Average user rating ({analysis.average_rating:.2f}) is critically low",
                recommendations=["Review recent user feedback", "Investigate system performance issues"]
            )
            alerts_triggered.append(alert)
        
        elif analysis.average_rating <= self._thresholds["average_rating"]["warning"]:
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="low_rating",
                severity=AlertSeverity.MEDIUM,
                agent_id=agent_id,
                metric_name="average_rating",
                current_value=analysis.average_rating,
                threshold_value=self._thresholds["average_rating"]["warning"],
                description=f"Average user rating ({analysis.average_rating:.2f}) is below warning threshold",
                recommendations=["Monitor user feedback closely", "Consider user experience improvements"]
            )
            alerts_triggered.append(alert)
        
        # Check quality score
        if analysis.quality_score <= self._thresholds["quality_score"]["critical"]:
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="low_quality",
                severity=AlertSeverity.CRITICAL,
                agent_id=agent_id,
                metric_name="quality_score",
                current_value=analysis.quality_score,
                threshold_value=self._thresholds["quality_score"]["critical"],
                description=f"Overall quality score ({analysis.quality_score:.2f}) is critically low",
                recommendations=["Immediate quality review required", "Check all system components"]
            )
            alerts_triggered.append(alert)
        
        # Check negative sentiment
        total_sentiment = sum(analysis.sentiment_distribution.values())
        if total_sentiment > 0:
            negative_pct = analysis.sentiment_distribution.get("negative", 0) / total_sentiment
            
            if negative_pct >= self._thresholds["negative_sentiment_pct"]["critical"]:
                alert = PerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="high_negative_sentiment",
                    severity=AlertSeverity.HIGH,
                    agent_id=agent_id,
                    metric_name="negative_sentiment_percentage",
                    current_value=negative_pct,
                    threshold_value=self._thresholds["negative_sentiment_pct"]["critical"],
                    description=f"High negative sentiment detected ({negative_pct:.1%})",
                    recommendations=["Review user comments", "Address common complaints"]
                )
                alerts_triggered.append(alert)
        
        # Store and notify about new alerts
        for alert in alerts_triggered:
            self._alerts[alert.alert_id] = alert
            await self._notify_alert_handlers(alert)
        
        return alerts_triggered
    
    async def _notify_alert_handlers(self, alert: PerformanceAlert):
        """Notify alert handlers"""
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return [alert for alert in self._alerts.values() if not alert.resolved]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self._alerts:
            self._alerts[alert_id].resolved = True
            return True
        return False

class FeedbackSystem:
    """Main feedback system coordinator"""
    
    def __init__(self):
        self.collector = FeedbackCollector()
        self.analyzer = FeedbackAnalyzer(self.collector)
        self.alert_manager = AlertManager(self.analyzer)
        self._processing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the feedback system"""
        self._processing_task = asyncio.create_task(self._feedback_processing_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Feedback system started")
    
    async def stop(self):
        """Stop the feedback system"""
        self._shutdown_event.set()
        
        if self._processing_task:
            self._processing_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._processing_task, self._monitoring_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Feedback system stopped")
    
    async def submit_feedback(self, session_id: str, agent_id: str, request_id: str,
                            rating: Optional[int] = None, categories: List[FeedbackCategory] = None,
                            comment: Optional[str] = None, user_id: Optional[str] = None,
                            feedback_type: FeedbackType = FeedbackType.RATING,
                            metadata: Dict[str, Any] = None) -> str:
        """Submit user feedback"""
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            request_id=request_id,
            feedback_type=feedback_type,
            rating=rating,
            categories=categories or [],
            comment=comment,
            metadata=metadata or {}
        )
        
        success = await self.collector.collect_feedback(feedback)
        
        if success:
            return feedback.feedback_id
        else:
            raise Exception("Failed to submit feedback")
    
    async def get_feedback_analysis(self, agent_id: Optional[str] = None,
                                  time_window: timedelta = timedelta(hours=24)) -> FeedbackAnalysis:
        """Get feedback analysis"""
        return await self.analyzer.analyze_feedback(agent_id, time_window)
    
    async def _feedback_processing_loop(self):
        """Background feedback processing loop"""
        while not self._shutdown_event.is_set():
            try:
                unprocessed = self.collector.get_unprocessed_feedback()
                
                for feedback_id in unprocessed:
                    feedback = self.collector.get_feedback(feedback_id)
                    if feedback:
                        # Process feedback (could include additional analysis, storage, etc.)
                        await self._process_feedback(feedback)
                        self.collector.mark_processed(feedback_id)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Feedback processing error: {e}")
                await asyncio.sleep(10)
    
    async def _monitoring_loop(self):
        """Background monitoring loop for alerts"""
        while not self._shutdown_event.is_set():
            try:
                # Check for alerts
                await self.alert_manager.check_alerts()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(300)
    
    async def _process_feedback(self, feedback: UserFeedback):
        """Process individual feedback entry"""
        # This could include additional processing like:
        # - Storing in database
        # - Triggering immediate alerts for critical feedback
        # - Updating real-time dashboards
        # - Notifying relevant teams
        
        logger.debug(f"Processed feedback {feedback.feedback_id}")

# Global feedback system instance
feedback_system = FeedbackSystem()

# Utility functions
async def start_feedback_system():
    """Start the global feedback system"""
    await feedback_system.start()

async def stop_feedback_system():
    """Stop the global feedback system"""
    await feedback_system.stop()

async def submit_user_feedback(session_id: str, agent_id: str, request_id: str, **kwargs) -> str:
    """Submit user feedback using the global system"""
    return await feedback_system.submit_feedback(session_id, agent_id, request_id, **kwargs)

async def get_feedback_analysis(agent_id: Optional[str] = None, 
                              time_window: timedelta = timedelta(hours=24)) -> FeedbackAnalysis:
    """Get feedback analysis using the global system"""
    return await feedback_system.get_feedback_analysis(agent_id, time_window)

def add_alert_handler(handler: Callable):
    """Add alert handler to the global system"""
    feedback_system.alert_manager.add_alert_handler(handler)