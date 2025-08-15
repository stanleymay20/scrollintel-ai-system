"""
Targeted Education Content Delivery and Tracking System
Advanced system for tracking content effectiveness and audience engagement
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import json
from uuid import uuid4

logger = logging.getLogger(__name__)

class ContentFormat(Enum):
    ARTICLE = "article"
    VIDEO = "video"
    WEBINAR = "webinar"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    INFOGRAPHIC = "infographic"
    PODCAST = "podcast"
    INTERACTIVE_DEMO = "interactive_demo"
    ASSESSMENT = "assessment"
    COURSE = "course"

class LearningObjective(Enum):
    AWARENESS = "awareness"
    UNDERSTANDING = "understanding"
    EVALUATION = "evaluation"
    TRIAL = "trial"
    ADOPTION = "adoption"
    ADVOCACY = "advocacy"

class PersonalizationLevel(Enum):
    GENERIC = "generic"
    SEGMENT_BASED = "segment_based"
    ROLE_BASED = "role_based"
    INDIVIDUAL = "individual"
    BEHAVIORAL = "behavioral"

@dataclass
class LearningPath:
    id: str
    name: str
    description: str
    target_segment: str
    learning_objectives: List[LearningObjective]
    content_sequence: List[str]  # Content IDs in order
    estimated_duration_hours: float
    difficulty_level: str  # "beginner", "intermediate", "advanced"
    prerequisites: List[str] = field(default_factory=list)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ContentEngagement:
    content_id: str
    user_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int = 0
    completion_percentage: float = 0.0
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    engagement_score: float = 0.0
    learning_outcome_achieved: bool = False
    feedback_rating: Optional[float] = None
    feedback_comments: Optional[str] = None

@dataclass
class ContentPerformanceMetrics:
    content_id: str
    total_views: int = 0
    unique_viewers: int = 0
    average_engagement_time: float = 0.0
    completion_rate: float = 0.0
    satisfaction_score: float = 0.0
    learning_effectiveness_score: float = 0.0
    conversion_rate: float = 0.0
    share_rate: float = 0.0
    return_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class UserLearningProfile:
    user_id: str
    segment: str
    role: str
    experience_level: str
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    completed_content: List[str] = field(default_factory=list)
    in_progress_content: List[str] = field(default_factory=list)
    learning_paths: List[str] = field(default_factory=list)
    engagement_history: List[ContentEngagement] = field(default_factory=list)
    knowledge_level: Dict[str, float] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)

@dataclass
class ContentRecommendation:
    content_id: str
    user_id: str
    recommendation_score: float
    reasoning: List[str]
    personalization_factors: Dict[str, Any]
    optimal_delivery_time: datetime
    preferred_format: ContentFormat
    estimated_engagement_probability: float
    generated_at: datetime = field(default_factory=datetime.now)

class EducationContentTracker:
    """
    Advanced system for tracking education content delivery, engagement,
    and effectiveness with personalized recommendations and learning analytics.
    """
    
    def __init__(self):
        self.learning_paths: Dict[str, LearningPath] = {}
        self.content_engagements: Dict[str, List[ContentEngagement]] = {}
        self.performance_metrics: Dict[str, ContentPerformanceMetrics] = {}
        self.user_profiles: Dict[str, UserLearningProfile] = {}
        self.content_recommendations: Dict[str, List[ContentRecommendation]] = {}
        self.learning_analytics: Dict[str, Any] = {}
        self._initialize_tracking_system()
    
    def _initialize_tracking_system(self):
        """Initialize the content tracking system"""
        self._setup_learning_paths()
        self._setup_analytics_framework()
        logger.info("Education Content Tracker initialized")
    
    def _setup_learning_paths(self):
        """Setup predefined learning paths for different segments"""
        # CTO Learning Path
        cto_path = LearningPath(
            id="cto_ai_leadership_path",
            name="AI Leadership for CTOs",
            description="Comprehensive learning path for CTOs to understand AI leadership capabilities",
            target_segment="enterprise_ctos",
            learning_objectives=[
                LearningObjective.AWARENESS,
                LearningObjective.UNDERSTANDING,
                LearningObjective.EVALUATION,
                LearningObjective.TRIAL
            ],
            content_sequence=[
                "ai_cto_introduction",
                "technical_capabilities_overview",
                "business_case_analysis",
                "implementation_roadmap",
                "roi_calculator_demo",
                "case_study_enterprise_a",
                "interactive_demo_session",
                "pilot_program_guide"
            ],
            estimated_duration_hours=8.5,
            difficulty_level="intermediate",
            completion_criteria={
                "minimum_content_completion": 0.8,
                "assessment_score": 0.75,
                "demo_interaction": True
            },
            success_metrics={
                "knowledge_retention": 0.85,
                "engagement_score": 0.75,
                "conversion_probability": 0.25
            }
        )
        
        # Tech Leader Learning Path
        tech_leader_path = LearningPath(
            id="tech_leader_implementation_path",
            name="AI CTO Implementation for Tech Leaders",
            description="Technical implementation guide for technology leaders",
            target_segment="tech_leaders",
            learning_objectives=[
                LearningObjective.UNDERSTANDING,
                LearningObjective.EVALUATION,
                LearningObjective.TRIAL,
                LearningObjective.ADOPTION
            ],
            content_sequence=[
                "technical_deep_dive",
                "architecture_overview",
                "integration_patterns",
                "security_considerations",
                "performance_benchmarks",
                "implementation_best_practices",
                "troubleshooting_guide",
                "advanced_configuration"
            ],
            estimated_duration_hours=12.0,
            difficulty_level="advanced",
            completion_criteria={
                "minimum_content_completion": 0.9,
                "technical_assessment_score": 0.8,
                "hands_on_exercise": True
            },
            success_metrics={
                "technical_proficiency": 0.9,
                "implementation_confidence": 0.8,
                "adoption_likelihood": 0.4
            }
        )
        
        # Board Member Learning Path
        board_path = LearningPath(
            id="board_strategic_overview_path",
            name="AI CTO Strategic Overview for Board Members",
            description="Strategic overview and governance considerations for board members",
            target_segment="board_members",
            learning_objectives=[
                LearningObjective.AWARENESS,
                LearningObjective.UNDERSTANDING,
                LearningObjective.EVALUATION
            ],
            content_sequence=[
                "executive_summary_ai_cto",
                "strategic_value_proposition",
                "risk_assessment_framework",
                "governance_considerations",
                "financial_impact_analysis",
                "competitive_advantage_analysis",
                "regulatory_compliance_overview"
            ],
            estimated_duration_hours=4.0,
            difficulty_level="beginner",
            completion_criteria={
                "minimum_content_completion": 0.7,
                "strategic_understanding_assessment": 0.7
            },
            success_metrics={
                "strategic_alignment": 0.8,
                "approval_likelihood": 0.6,
                "investment_readiness": 0.5
            }
        )
        
        self.learning_paths = {
            cto_path.id: cto_path,
            tech_leader_path.id: tech_leader_path,
            board_path.id: board_path
        }
    
    def _setup_analytics_framework(self):
        """Setup analytics framework for learning insights"""
        self.learning_analytics = {
            "engagement_patterns": {},
            "learning_effectiveness": {},
            "content_performance": {},
            "user_journey_analysis": {},
            "predictive_models": {}
        }
    
    async def track_content_engagement(
        self, 
        content_id: str, 
        user_id: str, 
        engagement_data: Dict[str, Any]
    ) -> ContentEngagement:
        """Track detailed content engagement"""
        try:
            session_id = engagement_data.get("session_id", str(uuid4()))
            
            engagement = ContentEngagement(
                content_id=content_id,
                user_id=user_id,
                session_id=session_id,
                start_time=datetime.fromisoformat(engagement_data["start_time"]),
                end_time=datetime.fromisoformat(engagement_data["end_time"]) if engagement_data.get("end_time") else None,
                duration_seconds=engagement_data.get("duration_seconds", 0),
                completion_percentage=engagement_data.get("completion_percentage", 0.0),
                interactions=engagement_data.get("interactions", []),
                feedback_rating=engagement_data.get("feedback_rating"),
                feedback_comments=engagement_data.get("feedback_comments")
            )
            
            # Calculate engagement score
            engagement.engagement_score = await self._calculate_engagement_score(engagement)
            
            # Determine learning outcome achievement
            engagement.learning_outcome_achieved = await self._assess_learning_outcome(engagement)
            
            # Store engagement
            if content_id not in self.content_engagements:
                self.content_engagements[content_id] = []
            self.content_engagements[content_id].append(engagement)
            
            # Update user profile
            await self._update_user_learning_profile(user_id, engagement)
            
            # Update content performance metrics
            await self._update_content_performance_metrics(content_id, engagement)
            
            logger.info(f"Tracked engagement for content {content_id} by user {user_id}")
            return engagement
            
        except Exception as e:
            logger.error(f"Error tracking content engagement: {str(e)}")
            raise
    
    async def _calculate_engagement_score(self, engagement: ContentEngagement) -> float:
        """Calculate engagement score based on multiple factors"""
        score = 0.0
        
        # Time-based engagement (40% weight)
        if engagement.duration_seconds > 0:
            # Normalize duration (assume 10 minutes is optimal)
            optimal_duration = 600  # 10 minutes
            time_score = min(1.0, engagement.duration_seconds / optimal_duration)
            score += time_score * 0.4
        
        # Completion-based engagement (30% weight)
        score += engagement.completion_percentage * 0.3
        
        # Interaction-based engagement (20% weight)
        interaction_score = min(1.0, len(engagement.interactions) / 5)  # 5 interactions is optimal
        score += interaction_score * 0.2
        
        # Feedback-based engagement (10% weight)
        if engagement.feedback_rating:
            feedback_score = engagement.feedback_rating / 5.0  # Assuming 5-point scale
            score += feedback_score * 0.1
        
        return min(1.0, score)
    
    async def _assess_learning_outcome(self, engagement: ContentEngagement) -> bool:
        """Assess if learning outcome was achieved"""
        # Simple heuristic: high completion + good engagement + positive feedback
        return (
            engagement.completion_percentage >= 0.8 and
            engagement.engagement_score >= 0.7 and
            (engagement.feedback_rating is None or engagement.feedback_rating >= 4.0)
        )
    
    async def _update_user_learning_profile(self, user_id: str, engagement: ContentEngagement):
        """Update user learning profile based on engagement"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserLearningProfile(
                user_id=user_id,
                segment="unknown",
                role="unknown",
                experience_level="unknown"
            )
        
        profile = self.user_profiles[user_id]
        profile.engagement_history.append(engagement)
        profile.last_active = datetime.now()
        
        # Update completed content
        if engagement.learning_outcome_achieved:
            if engagement.content_id not in profile.completed_content:
                profile.completed_content.append(engagement.content_id)
            
            # Remove from in-progress if completed
            if engagement.content_id in profile.in_progress_content:
                profile.in_progress_content.remove(engagement.content_id)
        else:
            # Add to in-progress if not already there
            if (engagement.content_id not in profile.in_progress_content and 
                engagement.content_id not in profile.completed_content):
                profile.in_progress_content.append(engagement.content_id)
        
        # Update learning preferences based on engagement patterns
        await self._update_learning_preferences(profile, engagement)
    
    async def _update_learning_preferences(self, profile: UserLearningProfile, engagement: ContentEngagement):
        """Update user learning preferences based on engagement patterns"""
        # Analyze engagement patterns to infer preferences
        if engagement.engagement_score > 0.8:
            # High engagement indicates preference for this type of content
            content_format = "unknown"  # Would be determined from content metadata
            
            if "preferred_formats" not in profile.learning_preferences:
                profile.learning_preferences["preferred_formats"] = {}
            
            current_score = profile.learning_preferences["preferred_formats"].get(content_format, 0.0)
            profile.learning_preferences["preferred_formats"][content_format] = min(1.0, current_score + 0.1)
        
        # Update optimal engagement time
        engagement_hour = engagement.start_time.hour
        if "optimal_hours" not in profile.learning_preferences:
            profile.learning_preferences["optimal_hours"] = {}
        
        hour_key = str(engagement_hour)
        current_score = profile.learning_preferences["optimal_hours"].get(hour_key, 0.0)
        profile.learning_preferences["optimal_hours"][hour_key] = current_score + engagement.engagement_score
    
    async def _update_content_performance_metrics(self, content_id: str, engagement: ContentEngagement):
        """Update content performance metrics"""
        if content_id not in self.performance_metrics:
            self.performance_metrics[content_id] = ContentPerformanceMetrics(content_id=content_id)
        
        metrics = self.performance_metrics[content_id]
        
        # Update view counts
        metrics.total_views += 1
        
        # Update unique viewers (simplified - would need more sophisticated tracking)
        metrics.unique_viewers = len(set(
            eng.user_id for eng in self.content_engagements.get(content_id, [])
        ))
        
        # Update average engagement time
        all_engagements = self.content_engagements.get(content_id, [])
        total_time = sum(eng.duration_seconds for eng in all_engagements)
        metrics.average_engagement_time = total_time / len(all_engagements) if all_engagements else 0
        
        # Update completion rate
        completed_engagements = [eng for eng in all_engagements if eng.learning_outcome_achieved]
        metrics.completion_rate = len(completed_engagements) / len(all_engagements) if all_engagements else 0
        
        # Update satisfaction score
        rated_engagements = [eng for eng in all_engagements if eng.feedback_rating is not None]
        if rated_engagements:
            metrics.satisfaction_score = sum(eng.feedback_rating for eng in rated_engagements) / len(rated_engagements)
        
        # Update learning effectiveness score
        metrics.learning_effectiveness_score = sum(eng.engagement_score for eng in all_engagements) / len(all_engagements) if all_engagements else 0
        
        metrics.last_updated = datetime.now()
    
    async def generate_personalized_recommendations(self, user_id: str, limit: int = 5) -> List[ContentRecommendation]:
        """Generate personalized content recommendations for user"""
        try:
            if user_id not in self.user_profiles:
                # Create basic profile for new user
                self.user_profiles[user_id] = UserLearningProfile(
                    user_id=user_id,
                    segment="unknown",
                    role="unknown",
                    experience_level="beginner"
                )
            
            profile = self.user_profiles[user_id]
            recommendations = []
            
            # Find appropriate learning path
            suitable_paths = await self._find_suitable_learning_paths(profile)
            
            for path in suitable_paths[:2]:  # Consider top 2 paths
                # Get next content in path
                next_content = await self._get_next_content_in_path(profile, path)
                
                if next_content:
                    recommendation = await self._create_content_recommendation(
                        profile, next_content, path
                    )
                    recommendations.append(recommendation)
            
            # Add content based on interests and engagement patterns
            interest_based_content = await self._get_interest_based_recommendations(profile, limit - len(recommendations))
            recommendations.extend(interest_based_content)
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
            
            # Store recommendations
            self.content_recommendations[user_id] = recommendations[:limit]
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    async def _find_suitable_learning_paths(self, profile: UserLearningProfile) -> List[LearningPath]:
        """Find learning paths suitable for user profile"""
        suitable_paths = []
        
        for path in self.learning_paths.values():
            # Match by segment
            if profile.segment == path.target_segment or profile.segment == "unknown":
                # Check prerequisites
                prerequisites_met = all(
                    prereq in profile.completed_content 
                    for prereq in path.prerequisites
                )
                
                if prerequisites_met:
                    suitable_paths.append(path)
        
        return suitable_paths
    
    async def _get_next_content_in_path(self, profile: UserLearningProfile, path: LearningPath) -> Optional[str]:
        """Get next content item in learning path for user"""
        for content_id in path.content_sequence:
            if (content_id not in profile.completed_content and 
                content_id not in profile.in_progress_content):
                return content_id
        
        return None
    
    async def _create_content_recommendation(
        self, 
        profile: UserLearningProfile, 
        content_id: str, 
        path: LearningPath
    ) -> ContentRecommendation:
        """Create content recommendation with scoring and reasoning"""
        # Calculate recommendation score
        score = 0.8  # Base score for path-based recommendation
        
        # Adjust based on user engagement history
        avg_engagement = sum(eng.engagement_score for eng in profile.engagement_history) / len(profile.engagement_history) if profile.engagement_history else 0.5
        score *= (0.5 + avg_engagement * 0.5)
        
        # Determine optimal delivery time
        optimal_time = await self._calculate_optimal_delivery_time(profile)
        
        # Determine preferred format
        preferred_format = await self._determine_preferred_format(profile)
        
        # Calculate engagement probability
        engagement_probability = min(0.95, score * 0.8 + avg_engagement * 0.2)
        
        recommendation = ContentRecommendation(
            content_id=content_id,
            user_id=profile.user_id,
            recommendation_score=score,
            reasoning=[
                f"Next step in {path.name} learning path",
                f"Matches your {profile.segment} profile",
                f"Based on your {avg_engagement:.1%} average engagement"
            ],
            personalization_factors={
                "learning_path": path.id,
                "user_segment": profile.segment,
                "experience_level": profile.experience_level,
                "engagement_history": len(profile.engagement_history)
            },
            optimal_delivery_time=optimal_time,
            preferred_format=preferred_format,
            estimated_engagement_probability=engagement_probability
        )
        
        return recommendation
    
    async def _get_interest_based_recommendations(self, profile: UserLearningProfile, limit: int) -> List[ContentRecommendation]:
        """Get recommendations based on user interests and behavior"""
        recommendations = []
        
        # Analyze engagement patterns to infer interests
        if profile.engagement_history:
            # Find content types with high engagement
            high_engagement_content = [
                eng.content_id for eng in profile.engagement_history 
                if eng.engagement_score > 0.7
            ]
            
            # Recommend similar content (simplified - would use content similarity)
            for content_id in high_engagement_content[:limit]:
                similar_content_id = f"similar_to_{content_id}"  # Placeholder
                
                recommendation = ContentRecommendation(
                    content_id=similar_content_id,
                    user_id=profile.user_id,
                    recommendation_score=0.6,
                    reasoning=[
                        f"Similar to {content_id} which you engaged with highly",
                        "Based on your content preferences"
                    ],
                    personalization_factors={
                        "based_on_content": content_id,
                        "similarity_score": 0.8
                    },
                    optimal_delivery_time=await self._calculate_optimal_delivery_time(profile),
                    preferred_format=await self._determine_preferred_format(profile),
                    estimated_engagement_probability=0.65
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _calculate_optimal_delivery_time(self, profile: UserLearningProfile) -> datetime:
        """Calculate optimal delivery time based on user behavior"""
        # Analyze engagement history to find optimal times
        if profile.engagement_history:
            engagement_hours = [eng.start_time.hour for eng in profile.engagement_history]
            
            # Find most common hour
            hour_counts = {}
            for hour in engagement_hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            optimal_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 10
        else:
            optimal_hour = 10  # Default to 10 AM
        
        # Calculate next optimal delivery time
        now = datetime.now()
        optimal_time = now.replace(hour=optimal_hour, minute=0, second=0, microsecond=0)
        
        # If optimal time has passed today, schedule for tomorrow
        if optimal_time <= now:
            optimal_time += timedelta(days=1)
        
        return optimal_time
    
    async def _determine_preferred_format(self, profile: UserLearningProfile) -> ContentFormat:
        """Determine user's preferred content format"""
        if "preferred_formats" in profile.learning_preferences:
            formats = profile.learning_preferences["preferred_formats"]
            if formats:
                preferred_format_str = max(formats, key=formats.get)
                try:
                    return ContentFormat(preferred_format_str)
                except ValueError:
                    pass
        
        # Default based on segment
        segment_defaults = {
            "enterprise_ctos": ContentFormat.WHITEPAPER,
            "tech_leaders": ContentFormat.VIDEO,
            "board_members": ContentFormat.ARTICLE,
            "investors": ContentFormat.CASE_STUDY
        }
        
        return segment_defaults.get(profile.segment, ContentFormat.ARTICLE)
    
    async def analyze_learning_effectiveness(self, content_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze learning effectiveness across content or specific content"""
        try:
            if content_id:
                # Analyze specific content
                return await self._analyze_content_effectiveness(content_id)
            else:
                # Analyze overall effectiveness
                return await self._analyze_overall_effectiveness()
                
        except Exception as e:
            logger.error(f"Error analyzing learning effectiveness: {str(e)}")
            raise
    
    async def _analyze_content_effectiveness(self, content_id: str) -> Dict[str, Any]:
        """Analyze effectiveness of specific content"""
        if content_id not in self.content_engagements:
            return {"error": "No engagement data found for content"}
        
        engagements = self.content_engagements[content_id]
        metrics = self.performance_metrics.get(content_id)
        
        analysis = {
            "content_id": content_id,
            "total_engagements": len(engagements),
            "unique_users": len(set(eng.user_id for eng in engagements)),
            "average_engagement_score": sum(eng.engagement_score for eng in engagements) / len(engagements),
            "completion_rate": len([eng for eng in engagements if eng.learning_outcome_achieved]) / len(engagements),
            "average_duration_minutes": sum(eng.duration_seconds for eng in engagements) / len(engagements) / 60,
            "satisfaction_score": metrics.satisfaction_score if metrics else 0.0,
            "engagement_distribution": self._calculate_engagement_distribution(engagements),
            "user_segment_breakdown": self._analyze_segment_breakdown(engagements),
            "improvement_recommendations": await self._generate_content_improvement_recommendations(content_id, engagements)
        }
        
        return analysis
    
    async def _analyze_overall_effectiveness(self) -> Dict[str, Any]:
        """Analyze overall learning effectiveness across all content"""
        all_engagements = []
        for engagements in self.content_engagements.values():
            all_engagements.extend(engagements)
        
        if not all_engagements:
            return {"error": "No engagement data available"}
        
        analysis = {
            "total_engagements": len(all_engagements),
            "unique_users": len(set(eng.user_id for eng in all_engagements)),
            "average_engagement_score": sum(eng.engagement_score for eng in all_engagements) / len(all_engagements),
            "overall_completion_rate": len([eng for eng in all_engagements if eng.learning_outcome_achieved]) / len(all_engagements),
            "content_performance_ranking": await self._rank_content_by_performance(),
            "user_learning_progress": await self._analyze_user_learning_progress(),
            "learning_path_effectiveness": await self._analyze_learning_path_effectiveness(),
            "engagement_trends": await self._analyze_engagement_trends(),
            "recommendations": await self._generate_system_recommendations()
        }
        
        return analysis
    
    def _calculate_engagement_distribution(self, engagements: List[ContentEngagement]) -> Dict[str, int]:
        """Calculate distribution of engagement scores"""
        distribution = {"low": 0, "medium": 0, "high": 0}
        
        for engagement in engagements:
            if engagement.engagement_score < 0.4:
                distribution["low"] += 1
            elif engagement.engagement_score < 0.7:
                distribution["medium"] += 1
            else:
                distribution["high"] += 1
        
        return distribution
    
    def _analyze_segment_breakdown(self, engagements: List[ContentEngagement]) -> Dict[str, Dict[str, Any]]:
        """Analyze engagement by user segment"""
        segment_data = {}
        
        for engagement in engagements:
            user_profile = self.user_profiles.get(engagement.user_id)
            segment = user_profile.segment if user_profile else "unknown"
            
            if segment not in segment_data:
                segment_data[segment] = {
                    "count": 0,
                    "total_engagement": 0.0,
                    "completions": 0
                }
            
            segment_data[segment]["count"] += 1
            segment_data[segment]["total_engagement"] += engagement.engagement_score
            if engagement.learning_outcome_achieved:
                segment_data[segment]["completions"] += 1
        
        # Calculate averages
        for segment, data in segment_data.items():
            data["average_engagement"] = data["total_engagement"] / data["count"]
            data["completion_rate"] = data["completions"] / data["count"]
        
        return segment_data
    
    async def _generate_content_improvement_recommendations(
        self, 
        content_id: str, 
        engagements: List[ContentEngagement]
    ) -> List[str]:
        """Generate recommendations for improving content"""
        recommendations = []
        
        avg_engagement = sum(eng.engagement_score for eng in engagements) / len(engagements)
        completion_rate = len([eng for eng in engagements if eng.learning_outcome_achieved]) / len(engagements)
        avg_duration = sum(eng.duration_seconds for eng in engagements) / len(engagements)
        
        if avg_engagement < 0.5:
            recommendations.append("Improve content engagement through interactive elements")
        
        if completion_rate < 0.6:
            recommendations.append("Simplify content structure to improve completion rates")
        
        if avg_duration < 300:  # Less than 5 minutes
            recommendations.append("Content may be too short - consider adding more depth")
        elif avg_duration > 1800:  # More than 30 minutes
            recommendations.append("Content may be too long - consider breaking into smaller segments")
        
        # Analyze feedback
        feedback_ratings = [eng.feedback_rating for eng in engagements if eng.feedback_rating is not None]
        if feedback_ratings:
            avg_rating = sum(feedback_ratings) / len(feedback_ratings)
            if avg_rating < 3.5:
                recommendations.append("Address user feedback concerns to improve satisfaction")
        
        return recommendations
    
    async def _rank_content_by_performance(self) -> List[Dict[str, Any]]:
        """Rank content by performance metrics"""
        content_scores = []
        
        for content_id, metrics in self.performance_metrics.items():
            # Calculate composite performance score
            score = (
                metrics.completion_rate * 0.3 +
                metrics.satisfaction_score / 5.0 * 0.25 +
                metrics.learning_effectiveness_score * 0.25 +
                min(1.0, metrics.average_engagement_time / 600) * 0.2  # Normalize to 10 minutes
            )
            
            content_scores.append({
                "content_id": content_id,
                "performance_score": score,
                "completion_rate": metrics.completion_rate,
                "satisfaction_score": metrics.satisfaction_score,
                "effectiveness_score": metrics.learning_effectiveness_score,
                "total_views": metrics.total_views
            })
        
        # Sort by performance score
        content_scores.sort(key=lambda x: x["performance_score"], reverse=True)
        
        return content_scores
    
    async def _analyze_user_learning_progress(self) -> Dict[str, Any]:
        """Analyze user learning progress across the system"""
        progress_data = {
            "total_users": len(self.user_profiles),
            "active_learners": 0,
            "completed_paths": 0,
            "average_completion_rate": 0.0,
            "segment_progress": {}
        }
        
        total_completion_rate = 0.0
        
        for profile in self.user_profiles.values():
            # Check if user is active (engaged in last 30 days)
            if profile.last_active > datetime.now() - timedelta(days=30):
                progress_data["active_learners"] += 1
            
            # Calculate user completion rate
            total_content = len(profile.completed_content) + len(profile.in_progress_content)
            completion_rate = len(profile.completed_content) / total_content if total_content > 0 else 0
            total_completion_rate += completion_rate
            
            # Check for completed learning paths
            for path_id in profile.learning_paths:
                path = self.learning_paths.get(path_id)
                if path and all(content_id in profile.completed_content for content_id in path.content_sequence):
                    progress_data["completed_paths"] += 1
            
            # Segment progress
            segment = profile.segment
            if segment not in progress_data["segment_progress"]:
                progress_data["segment_progress"][segment] = {
                    "users": 0,
                    "total_completion": 0.0
                }
            
            progress_data["segment_progress"][segment]["users"] += 1
            progress_data["segment_progress"][segment]["total_completion"] += completion_rate
        
        # Calculate averages
        if progress_data["total_users"] > 0:
            progress_data["average_completion_rate"] = total_completion_rate / progress_data["total_users"]
        
        for segment_data in progress_data["segment_progress"].values():
            if segment_data["users"] > 0:
                segment_data["average_completion"] = segment_data["total_completion"] / segment_data["users"]
        
        return progress_data
    
    async def _analyze_learning_path_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of learning paths"""
        path_analysis = {}
        
        for path_id, path in self.learning_paths.items():
            users_on_path = [
                profile for profile in self.user_profiles.values()
                if path_id in profile.learning_paths
            ]
            
            if users_on_path:
                completed_users = [
                    profile for profile in users_on_path
                    if all(content_id in profile.completed_content for content_id in path.content_sequence)
                ]
                
                path_analysis[path_id] = {
                    "name": path.name,
                    "total_users": len(users_on_path),
                    "completed_users": len(completed_users),
                    "completion_rate": len(completed_users) / len(users_on_path),
                    "average_time_to_complete": await self._calculate_average_completion_time(path, completed_users),
                    "effectiveness_score": await self._calculate_path_effectiveness(path, users_on_path)
                }
        
        return path_analysis
    
    async def _calculate_average_completion_time(self, path: LearningPath, completed_users: List[UserLearningProfile]) -> float:
        """Calculate average time to complete learning path"""
        completion_times = []
        
        for profile in completed_users:
            # Find first and last engagement with path content
            path_engagements = [
                eng for eng in profile.engagement_history
                if eng.content_id in path.content_sequence
            ]
            
            if path_engagements:
                first_engagement = min(path_engagements, key=lambda x: x.start_time)
                last_engagement = max(path_engagements, key=lambda x: x.start_time)
                
                completion_time = (last_engagement.start_time - first_engagement.start_time).total_seconds() / 3600  # Hours
                completion_times.append(completion_time)
        
        return sum(completion_times) / len(completion_times) if completion_times else 0.0
    
    async def _calculate_path_effectiveness(self, path: LearningPath, users: List[UserLearningProfile]) -> float:
        """Calculate learning path effectiveness score"""
        if not users:
            return 0.0
        
        total_score = 0.0
        
        for profile in users:
            # Calculate user's progress through path
            completed_content = [
                content_id for content_id in path.content_sequence
                if content_id in profile.completed_content
            ]
            
            progress_score = len(completed_content) / len(path.content_sequence)
            
            # Calculate engagement quality for path content
            path_engagements = [
                eng for eng in profile.engagement_history
                if eng.content_id in path.content_sequence
            ]
            
            engagement_score = sum(eng.engagement_score for eng in path_engagements) / len(path_engagements) if path_engagements else 0
            
            # Combine progress and engagement
            user_effectiveness = (progress_score * 0.6 + engagement_score * 0.4)
            total_score += user_effectiveness
        
        return total_score / len(users)
    
    async def _analyze_engagement_trends(self) -> Dict[str, Any]:
        """Analyze engagement trends over time"""
        # Group engagements by time periods
        daily_engagement = {}
        weekly_engagement = {}
        monthly_engagement = {}
        
        all_engagements = []
        for engagements in self.content_engagements.values():
            all_engagements.extend(engagements)
        
        for engagement in all_engagements:
            date_key = engagement.start_time.date().isoformat()
            week_key = f"{engagement.start_time.year}-W{engagement.start_time.isocalendar()[1]}"
            month_key = f"{engagement.start_time.year}-{engagement.start_time.month:02d}"
            
            # Daily trends
            if date_key not in daily_engagement:
                daily_engagement[date_key] = {"count": 0, "total_score": 0.0}
            daily_engagement[date_key]["count"] += 1
            daily_engagement[date_key]["total_score"] += engagement.engagement_score
            
            # Weekly trends
            if week_key not in weekly_engagement:
                weekly_engagement[week_key] = {"count": 0, "total_score": 0.0}
            weekly_engagement[week_key]["count"] += 1
            weekly_engagement[week_key]["total_score"] += engagement.engagement_score
            
            # Monthly trends
            if month_key not in monthly_engagement:
                monthly_engagement[month_key] = {"count": 0, "total_score": 0.0}
            monthly_engagement[month_key]["count"] += 1
            monthly_engagement[month_key]["total_score"] += engagement.engagement_score
        
        # Calculate averages
        for period_data in [daily_engagement, weekly_engagement, monthly_engagement]:
            for data in period_data.values():
                data["average_score"] = data["total_score"] / data["count"] if data["count"] > 0 else 0
        
        return {
            "daily_trends": daily_engagement,
            "weekly_trends": weekly_engagement,
            "monthly_trends": monthly_engagement,
            "trend_analysis": {
                "engagement_growing": len(daily_engagement) > 7,  # Simplified trend detection
                "peak_engagement_day": max(daily_engagement.items(), key=lambda x: x[1]["count"])[0] if daily_engagement else None,
                "average_daily_engagements": sum(data["count"] for data in daily_engagement.values()) / len(daily_engagement) if daily_engagement else 0
            }
        }
    
    async def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations for improvement"""
        recommendations = []
        
        # Analyze overall performance
        all_engagements = []
        for engagements in self.content_engagements.values():
            all_engagements.extend(engagements)
        
        if all_engagements:
            avg_engagement = sum(eng.engagement_score for eng in all_engagements) / len(all_engagements)
            completion_rate = len([eng for eng in all_engagements if eng.learning_outcome_achieved]) / len(all_engagements)
            
            if avg_engagement < 0.6:
                recommendations.append("Focus on improving overall content engagement through interactivity")
            
            if completion_rate < 0.7:
                recommendations.append("Optimize content length and complexity to improve completion rates")
            
            # Analyze user retention
            active_users = len([
                profile for profile in self.user_profiles.values()
                if profile.last_active > datetime.now() - timedelta(days=30)
            ])
            
            total_users = len(self.user_profiles)
            retention_rate = active_users / total_users if total_users > 0 else 0
            
            if retention_rate < 0.5:
                recommendations.append("Implement user retention strategies and re-engagement campaigns")
            
            # Analyze learning path adoption
            users_on_paths = len([
                profile for profile in self.user_profiles.values()
                if profile.learning_paths
            ])
            
            path_adoption_rate = users_on_paths / total_users if total_users > 0 else 0
            
            if path_adoption_rate < 0.3:
                recommendations.append("Promote learning path adoption through better onboarding")
        
        return recommendations
    
    async def export_analytics_report(self, format: str = "json") -> Dict[str, Any]:
        """Export comprehensive analytics report"""
        try:
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "comprehensive_learning_analytics",
                    "format": format,
                    "data_period": {
                        "start_date": min(
                            eng.start_time for engagements in self.content_engagements.values()
                            for eng in engagements
                        ).isoformat() if self.content_engagements else None,
                        "end_date": datetime.now().isoformat()
                    }
                },
                "executive_summary": await self._generate_executive_summary(),
                "content_performance": await self._rank_content_by_performance(),
                "user_analytics": await self._analyze_user_learning_progress(),
                "learning_path_effectiveness": await self._analyze_learning_path_effectiveness(),
                "engagement_trends": await self._analyze_engagement_trends(),
                "recommendations": await self._generate_system_recommendations(),
                "detailed_metrics": {
                    "total_content_pieces": len(self.performance_metrics),
                    "total_engagements": sum(len(engagements) for engagements in self.content_engagements.values()),
                    "total_users": len(self.user_profiles),
                    "total_learning_paths": len(self.learning_paths)
                }
            }
            
            logger.info("Generated comprehensive analytics report")
            return report
            
        except Exception as e:
            logger.error(f"Error exporting analytics report: {str(e)}")
            raise
    
    async def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of learning analytics"""
        all_engagements = []
        for engagements in self.content_engagements.values():
            all_engagements.extend(engagements)
        
        if not all_engagements:
            return {"message": "No engagement data available"}
        
        return {
            "key_metrics": {
                "total_learning_sessions": len(all_engagements),
                "unique_learners": len(set(eng.user_id for eng in all_engagements)),
                "average_engagement_score": sum(eng.engagement_score for eng in all_engagements) / len(all_engagements),
                "overall_completion_rate": len([eng for eng in all_engagements if eng.learning_outcome_achieved]) / len(all_engagements),
                "total_learning_hours": sum(eng.duration_seconds for eng in all_engagements) / 3600
            },
            "performance_highlights": [
                f"Achieved {len([eng for eng in all_engagements if eng.learning_outcome_achieved])} successful learning outcomes",
                f"Average engagement score of {sum(eng.engagement_score for eng in all_engagements) / len(all_engagements):.2f}",
                f"Total of {sum(eng.duration_seconds for eng in all_engagements) / 3600:.1f} hours of learning delivered"
            ],
            "areas_for_improvement": [
                "Increase content completion rates",
                "Improve user retention and re-engagement",
                "Expand personalization capabilities"
            ]
        }