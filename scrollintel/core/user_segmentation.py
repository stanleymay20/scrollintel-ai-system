"""
User Segmentation and Cohort Analysis
Implements user segmentation, cohort analysis, and behavioral clustering
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
import logging
from enum import Enum
import math

logger = logging.getLogger(__name__)

class SegmentType(Enum):
    BEHAVIORAL = "behavioral"
    DEMOGRAPHIC = "demographic"
    GEOGRAPHIC = "geographic"
    PSYCHOGRAPHIC = "psychographic"
    TECHNOGRAPHIC = "technographic"
    CUSTOM = "custom"

class CohortType(Enum):
    ACQUISITION = "acquisition"
    BEHAVIORAL = "behavioral"
    REVENUE = "revenue"

@dataclass
class UserSegment:
    segment_id: str
    name: str
    description: str
    segment_type: SegmentType
    criteria: Dict[str, Any]
    user_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool
    metadata: Dict[str, Any]

@dataclass
class CohortDefinition:
    cohort_id: str
    name: str
    description: str
    cohort_type: CohortType
    period_type: str  # daily, weekly, monthly
    start_date: datetime
    end_date: Optional[datetime]
    criteria: Dict[str, Any]
    created_at: datetime

@dataclass
class CohortAnalysis:
    analysis_id: str
    cohort_id: str
    period_start: datetime
    period_end: datetime
    cohort_data: Dict[str, Any]
    retention_rates: Dict[str, float]
    revenue_data: Dict[str, float]
    user_counts: Dict[str, int]
    insights: List[str]
    generated_at: datetime

@dataclass
class UserProfile:
    user_id: str
    first_seen: datetime
    last_seen: datetime
    total_sessions: int
    total_page_views: int
    total_events: int
    segments: List[str]
    cohorts: List[str]
    properties: Dict[str, Any]
    behavioral_score: float
    engagement_level: str
    lifecycle_stage: str

class UserSegmentationEngine:
    """User segmentation and cohort analysis engine"""
    
    def __init__(self):
        self.segments: Dict[str, UserSegment] = {}
        self.cohorts: Dict[str, CohortDefinition] = {}
        self.cohort_analyses: List[CohortAnalysis] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Initialize default segments
        self._create_default_segments()
        
        # Initialize default cohorts
        self._create_default_cohorts()
    
    def _create_default_segments(self):
        """Create default user segments"""
        default_segments = [
            {
                "name": "New Users",
                "description": "Users who signed up in the last 7 days",
                "segment_type": SegmentType.BEHAVIORAL,
                "criteria": {
                    "days_since_signup": {"max": 7}
                }
            },
            {
                "name": "Active Users",
                "description": "Users who have been active in the last 30 days",
                "segment_type": SegmentType.BEHAVIORAL,
                "criteria": {
                    "days_since_last_activity": {"max": 30},
                    "total_sessions": {"min": 1}
                }
            },
            {
                "name": "Power Users",
                "description": "Highly engaged users with frequent activity",
                "segment_type": SegmentType.BEHAVIORAL,
                "criteria": {
                    "total_sessions": {"min": 10},
                    "engagement_level": "high",
                    "days_since_last_activity": {"max": 7}
                }
            },
            {
                "name": "At Risk Users",
                "description": "Previously active users who haven't been seen recently",
                "segment_type": SegmentType.BEHAVIORAL,
                "criteria": {
                    "days_since_last_activity": {"min": 14, "max": 60},
                    "total_sessions": {"min": 3}
                }
            },
            {
                "name": "Churned Users",
                "description": "Users who haven't been active for over 60 days",
                "segment_type": SegmentType.BEHAVIORAL,
                "criteria": {
                    "days_since_last_activity": {"min": 60}
                }
            }
        ]
        
        for segment_data in default_segments:
            segment_id = str(uuid.uuid4())
            segment = UserSegment(
                segment_id=segment_id,
                name=segment_data["name"],
                description=segment_data["description"],
                segment_type=segment_data["segment_type"],
                criteria=segment_data["criteria"],
                user_count=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True,
                metadata={}
            )
            self.segments[segment_id] = segment
    
    def _create_default_cohorts(self):
        """Create default cohort definitions"""
        # Weekly acquisition cohorts
        acquisition_cohort = CohortDefinition(
            cohort_id=str(uuid.uuid4()),
            name="Weekly Acquisition Cohorts",
            description="Users grouped by week of first signup",
            cohort_type=CohortType.ACQUISITION,
            period_type="weekly",
            start_date=datetime.utcnow() - timedelta(weeks=12),
            end_date=None,
            criteria={"event_name": "user_signup"},
            created_at=datetime.utcnow()
        )
        self.cohorts[acquisition_cohort.cohort_id] = acquisition_cohort
        
        # Monthly revenue cohorts
        revenue_cohort = CohortDefinition(
            cohort_id=str(uuid.uuid4()),
            name="Monthly Revenue Cohorts",
            description="Users grouped by month of first purchase",
            cohort_type=CohortType.REVENUE,
            period_type="monthly",
            start_date=datetime.utcnow() - timedelta(days=365),
            end_date=None,
            criteria={"event_name": "subscription_created"},
            created_at=datetime.utcnow()
        )
        self.cohorts[revenue_cohort.cohort_id] = revenue_cohort
    
    async def create_custom_segment(self,
                                   name: str,
                                   description: str,
                                   segment_type: SegmentType,
                                   criteria: Dict[str, Any]) -> str:
        """Create a custom user segment"""
        try:
            segment_id = str(uuid.uuid4())
            
            segment = UserSegment(
                segment_id=segment_id,
                name=name,
                description=description,
                segment_type=segment_type,
                criteria=criteria,
                user_count=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True,
                metadata={}
            )
            
            self.segments[segment_id] = segment
            
            # Calculate initial user count
            await self._update_segment_users(segment_id)
            
            logger.info(f"Created custom segment: {name} with {segment.user_count} users")
            return segment_id
            
        except Exception as e:
            logger.error(f"Error creating custom segment: {str(e)}")
            raise
    
    async def update_user_profile(self,
                                 user_id: str,
                                 events: List[Dict[str, Any]],
                                 properties: Dict[str, Any] = None) -> UserProfile:
        """Update user profile with new activity data"""
        try:
            if user_id not in self.user_profiles:
                # Create new profile
                first_event = min(events, key=lambda x: x.get("timestamp", datetime.utcnow()))
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    first_seen=first_event.get("timestamp", datetime.utcnow()),
                    last_seen=datetime.utcnow(),
                    total_sessions=0,
                    total_page_views=0,
                    total_events=0,
                    segments=[],
                    cohorts=[],
                    properties=properties or {},
                    behavioral_score=0.0,
                    engagement_level="low",
                    lifecycle_stage="new"
                )
            
            profile = self.user_profiles[user_id]
            
            # Update activity metrics
            profile.last_seen = datetime.utcnow()
            profile.total_events += len(events)
            
            # Count sessions and page views
            sessions = set()
            page_views = 0
            
            for event in events:
                if "session_id" in event:
                    sessions.add(event["session_id"])
                if event.get("event_type") == "page_view":
                    page_views += 1
            
            profile.total_sessions += len(sessions)
            profile.total_page_views += page_views
            
            # Update properties
            if properties:
                profile.properties.update(properties)
            
            # Calculate behavioral score
            profile.behavioral_score = await self._calculate_behavioral_score(profile)
            
            # Update engagement level
            profile.engagement_level = await self._calculate_engagement_level(profile)
            
            # Update lifecycle stage
            profile.lifecycle_stage = await self._calculate_lifecycle_stage(profile)
            
            # Update segment memberships
            await self._update_user_segments(user_id)
            
            logger.info(f"Updated profile for user {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            raise
    
    async def _calculate_behavioral_score(self, profile: UserProfile) -> float:
        """Calculate behavioral engagement score (0-100)"""
        try:
            score = 0.0
            
            # Recency score (0-30 points)
            days_since_last_activity = (datetime.utcnow() - profile.last_seen).days
            if days_since_last_activity == 0:
                recency_score = 30
            elif days_since_last_activity <= 7:
                recency_score = 25
            elif days_since_last_activity <= 30:
                recency_score = 15
            else:
                recency_score = 0
            
            score += recency_score
            
            # Frequency score (0-40 points)
            days_active = (datetime.utcnow() - profile.first_seen).days + 1
            session_frequency = profile.total_sessions / days_active
            
            if session_frequency >= 1:
                frequency_score = 40
            elif session_frequency >= 0.5:
                frequency_score = 30
            elif session_frequency >= 0.2:
                frequency_score = 20
            else:
                frequency_score = 10
            
            score += frequency_score
            
            # Volume score (0-30 points)
            if profile.total_events >= 100:
                volume_score = 30
            elif profile.total_events >= 50:
                volume_score = 20
            elif profile.total_events >= 10:
                volume_score = 10
            else:
                volume_score = 5
            
            score += volume_score
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating behavioral score: {str(e)}")
            return 0.0
    
    async def _calculate_engagement_level(self, profile: UserProfile) -> str:
        """Calculate engagement level based on behavioral score"""
        if profile.behavioral_score >= 80:
            return "high"
        elif profile.behavioral_score >= 50:
            return "medium"
        else:
            return "low"
    
    async def _calculate_lifecycle_stage(self, profile: UserProfile) -> str:
        """Calculate user lifecycle stage"""
        days_since_signup = (datetime.utcnow() - profile.first_seen).days
        days_since_last_activity = (datetime.utcnow() - profile.last_seen).days
        
        if days_since_signup <= 7:
            return "new"
        elif days_since_last_activity <= 7 and profile.total_sessions >= 5:
            return "active"
        elif days_since_last_activity <= 30:
            return "engaged"
        elif days_since_last_activity <= 60:
            return "at_risk"
        else:
            return "churned"
    
    async def _update_user_segments(self, user_id: str):
        """Update user's segment memberships"""
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        profile.segments = []
        
        for segment_id, segment in self.segments.items():
            if not segment.is_active:
                continue
            
            if await self._user_matches_segment_criteria(profile, segment.criteria):
                profile.segments.append(segment_id)
    
    async def _user_matches_segment_criteria(self,
                                           profile: UserProfile,
                                           criteria: Dict[str, Any]) -> bool:
        """Check if user matches segment criteria"""
        try:
            for criterion, condition in criteria.items():
                if criterion == "days_since_signup":
                    days = (datetime.utcnow() - profile.first_seen).days
                    if not self._check_numeric_condition(days, condition):
                        return False
                
                elif criterion == "days_since_last_activity":
                    days = (datetime.utcnow() - profile.last_seen).days
                    if not self._check_numeric_condition(days, condition):
                        return False
                
                elif criterion == "total_sessions":
                    if not self._check_numeric_condition(profile.total_sessions, condition):
                        return False
                
                elif criterion == "total_events":
                    if not self._check_numeric_condition(profile.total_events, condition):
                        return False
                
                elif criterion == "engagement_level":
                    if profile.engagement_level != condition:
                        return False
                
                elif criterion == "lifecycle_stage":
                    if profile.lifecycle_stage != condition:
                        return False
                
                elif criterion in profile.properties:
                    if profile.properties[criterion] != condition:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking segment criteria: {str(e)}")
            return False
    
    def _check_numeric_condition(self, value: float, condition: Any) -> bool:
        """Check if numeric value meets condition"""
        if isinstance(condition, dict):
            if "min" in condition and value < condition["min"]:
                return False
            if "max" in condition and value > condition["max"]:
                return False
            return True
        else:
            return value == condition
    
    async def _update_segment_users(self, segment_id: str):
        """Update user count for segment"""
        if segment_id not in self.segments:
            return
        
        segment = self.segments[segment_id]
        matching_users = 0
        
        for profile in self.user_profiles.values():
            if await self._user_matches_segment_criteria(profile, segment.criteria):
                matching_users += 1
        
        segment.user_count = matching_users
        segment.updated_at = datetime.utcnow()
    
    async def perform_cohort_analysis(self,
                                     cohort_id: str,
                                     analysis_periods: int = 12) -> CohortAnalysis:
        """Perform cohort analysis"""
        try:
            if cohort_id not in self.cohorts:
                raise ValueError(f"Cohort {cohort_id} not found")
            
            cohort_def = self.cohorts[cohort_id]
            
            # Generate cohort data
            cohort_data = await self._generate_cohort_data(cohort_def, analysis_periods)
            
            # Calculate retention rates
            retention_rates = await self._calculate_retention_rates(cohort_data)
            
            # Calculate revenue data if applicable
            revenue_data = {}
            if cohort_def.cohort_type == CohortType.REVENUE:
                revenue_data = await self._calculate_cohort_revenue(cohort_data)
            
            # Calculate user counts
            user_counts = await self._calculate_cohort_user_counts(cohort_data)
            
            # Generate insights
            insights = await self._generate_cohort_insights(retention_rates, revenue_data, user_counts)
            
            analysis = CohortAnalysis(
                analysis_id=str(uuid.uuid4()),
                cohort_id=cohort_id,
                period_start=cohort_def.start_date,
                period_end=datetime.utcnow(),
                cohort_data=cohort_data,
                retention_rates=retention_rates,
                revenue_data=revenue_data,
                user_counts=user_counts,
                insights=insights,
                generated_at=datetime.utcnow()
            )
            
            self.cohort_analyses.append(analysis)
            logger.info(f"Completed cohort analysis for {cohort_def.name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing cohort analysis: {str(e)}")
            raise
    
    async def _generate_cohort_data(self,
                                   cohort_def: CohortDefinition,
                                   periods: int) -> Dict[str, Any]:
        """Generate cohort data structure"""
        cohort_data = {}
        
        # Create time periods based on cohort type
        if cohort_def.period_type == "weekly":
            period_delta = timedelta(weeks=1)
        elif cohort_def.period_type == "monthly":
            period_delta = timedelta(days=30)
        else:
            period_delta = timedelta(days=1)
        
        current_date = cohort_def.start_date
        
        for period in range(periods):
            period_key = current_date.strftime("%Y-%m-%d")
            cohort_data[period_key] = {
                "period_start": current_date,
                "period_end": current_date + period_delta,
                "users": [],
                "retention_by_period": {}
            }
            current_date += period_delta
        
        # Assign users to cohorts based on their first activity
        for profile in self.user_profiles.values():
            cohort_period = self._find_user_cohort_period(profile, cohort_def, cohort_data)
            if cohort_period:
                cohort_data[cohort_period]["users"].append(profile.user_id)
        
        return cohort_data
    
    def _find_user_cohort_period(self,
                                profile: UserProfile,
                                cohort_def: CohortDefinition,
                                cohort_data: Dict[str, Any]) -> Optional[str]:
        """Find which cohort period a user belongs to"""
        user_cohort_date = profile.first_seen
        
        for period_key, period_data in cohort_data.items():
            if (period_data["period_start"] <= user_cohort_date < period_data["period_end"]):
                return period_key
        
        return None
    
    async def _calculate_retention_rates(self, cohort_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate retention rates for cohorts"""
        retention_rates = {}
        
        for period_key, period_data in cohort_data.items():
            if not period_data["users"]:
                continue
            
            initial_users = len(period_data["users"])
            
            # Calculate retention for subsequent periods
            for i in range(1, 13):  # 12 periods forward
                retained_users = 0
                retention_date = period_data["period_start"] + timedelta(weeks=i)
                
                for user_id in period_data["users"]:
                    if user_id in self.user_profiles:
                        profile = self.user_profiles[user_id]
                        if profile.last_seen >= retention_date:
                            retained_users += 1
                
                retention_rate = (retained_users / initial_users * 100) if initial_users > 0 else 0
                retention_rates[f"{period_key}_period_{i}"] = retention_rate
        
        return retention_rates
    
    async def _calculate_cohort_revenue(self, cohort_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate revenue data for cohorts"""
        # This would integrate with actual revenue/billing data
        # For now, return placeholder data
        revenue_data = {}
        
        for period_key, period_data in cohort_data.items():
            revenue_data[f"{period_key}_ltv"] = len(period_data["users"]) * 50.0  # Placeholder LTV
            revenue_data[f"{period_key}_arpu"] = 50.0  # Placeholder ARPU
        
        return revenue_data
    
    async def _calculate_cohort_user_counts(self, cohort_data: Dict[str, Any]) -> Dict[str, int]:
        """Calculate user counts for cohorts"""
        user_counts = {}
        
        for period_key, period_data in cohort_data.items():
            user_counts[period_key] = len(period_data["users"])
        
        return user_counts
    
    async def _generate_cohort_insights(self,
                                       retention_rates: Dict[str, float],
                                       revenue_data: Dict[str, float],
                                       user_counts: Dict[str, int]) -> List[str]:
        """Generate insights from cohort analysis"""
        insights = []
        
        if retention_rates:
            # Find average retention rates
            period_1_rates = [rate for key, rate in retention_rates.items() if "_period_1" in key]
            period_4_rates = [rate for key, rate in retention_rates.items() if "_period_4" in key]
            
            if period_1_rates:
                avg_period_1 = sum(period_1_rates) / len(period_1_rates)
                insights.append(f"Average 1-period retention rate: {avg_period_1:.1f}%")
                
                if avg_period_1 < 30:
                    insights.append("Low early retention - focus on onboarding improvements")
                elif avg_period_1 > 60:
                    insights.append("Strong early retention - good product-market fit")
            
            if period_4_rates:
                avg_period_4 = sum(period_4_rates) / len(period_4_rates)
                insights.append(f"Average 4-period retention rate: {avg_period_4:.1f}%")
        
        # User acquisition insights
        if user_counts:
            recent_cohorts = sorted(user_counts.items(), key=lambda x: x[0])[-4:]
            if len(recent_cohorts) >= 2:
                growth_rate = ((recent_cohorts[-1][1] - recent_cohorts[0][1]) / recent_cohorts[0][1] * 100) if recent_cohorts[0][1] > 0 else 0
                insights.append(f"User acquisition growth rate: {growth_rate:+.1f}%")
        
        return insights if insights else ["Insufficient data for meaningful insights"]
    
    async def get_segmentation_dashboard(self) -> Dict[str, Any]:
        """Get segmentation dashboard data"""
        try:
            # Update all segment user counts
            for segment_id in self.segments.keys():
                await self._update_segment_users(segment_id)
            
            # Get segment distribution
            segment_distribution = {}
            for segment_id, segment in self.segments.items():
                if segment.is_active:
                    segment_distribution[segment.name] = segment.user_count
            
            # Calculate engagement distribution
            engagement_distribution = {"high": 0, "medium": 0, "low": 0}
            lifecycle_distribution = {"new": 0, "active": 0, "engaged": 0, "at_risk": 0, "churned": 0}
            
            for profile in self.user_profiles.values():
                engagement_distribution[profile.engagement_level] += 1
                lifecycle_distribution[profile.lifecycle_stage] += 1
            
            return {
                "total_users": len(self.user_profiles),
                "total_segments": len([s for s in self.segments.values() if s.is_active]),
                "segment_distribution": segment_distribution,
                "engagement_distribution": engagement_distribution,
                "lifecycle_distribution": lifecycle_distribution,
                "avg_behavioral_score": sum(p.behavioral_score for p in self.user_profiles.values()) / len(self.user_profiles) if self.user_profiles else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting segmentation dashboard: {str(e)}")
            raise

# Global user segmentation engine instance
user_segmentation = UserSegmentationEngine()