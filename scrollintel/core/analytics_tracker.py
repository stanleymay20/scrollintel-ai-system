"""
Analytics and Marketing Infrastructure
Implements Google Analytics, user behavior tracking, and marketing attribution
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class EventType(Enum):
    PAGE_VIEW = "page_view"
    USER_ACTION = "user_action"
    CONVERSION = "conversion"
    EXPERIMENT = "experiment"
    CAMPAIGN = "campaign"
    COHORT = "cohort"

@dataclass
class AnalyticsEvent:
    event_id: str
    user_id: str
    session_id: str
    event_type: EventType
    event_name: str
    properties: Dict[str, Any]
    timestamp: datetime
    page_url: str
    user_agent: str
    ip_address: str
    campaign_data: Optional[Dict[str, str]] = None
    experiment_data: Optional[Dict[str, str]] = None

@dataclass
class ConversionFunnel:
    funnel_id: str
    name: str
    steps: List[str]
    conversion_rates: Dict[str, float]
    drop_off_points: List[str]
    total_users: int
    completed_users: int

@dataclass
class UserSegment:
    segment_id: str
    name: str
    criteria: Dict[str, Any]
    user_count: int
    properties: Dict[str, Any]

class AnalyticsTracker:
    """Core analytics tracking system"""
    
    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self.sessions: Dict[str, Dict] = {}
        self.user_segments: Dict[str, UserSegment] = {}
        self.conversion_funnels: Dict[str, ConversionFunnel] = {}
        
    async def track_event(self, 
                         user_id: str,
                         session_id: str,
                         event_name: str,
                         properties: Dict[str, Any],
                         page_url: str,
                         user_agent: str,
                         ip_address: str,
                         campaign_data: Optional[Dict[str, str]] = None) -> str:
        """Track a user event"""
        try:
            event = AnalyticsEvent(
                event_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                event_type=EventType.USER_ACTION,
                event_name=event_name,
                properties=properties,
                timestamp=datetime.utcnow(),
                page_url=page_url,
                user_agent=user_agent,
                ip_address=ip_address,
                campaign_data=campaign_data
            )
            
            self.events.append(event)
            await self._update_session(session_id, event)
            await self._process_conversion_tracking(event)
            
            logger.info(f"Tracked event: {event_name} for user: {user_id}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Error tracking event: {str(e)}")
            raise
    
    async def track_page_view(self,
                             user_id: str,
                             session_id: str,
                             page_url: str,
                             page_title: str,
                             user_agent: str,
                             ip_address: str,
                             referrer: Optional[str] = None) -> str:
        """Track a page view"""
        try:
            properties = {
                "page_title": page_title,
                "referrer": referrer,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            event = AnalyticsEvent(
                event_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                event_type=EventType.PAGE_VIEW,
                event_name="page_view",
                properties=properties,
                timestamp=datetime.utcnow(),
                page_url=page_url,
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            self.events.append(event)
            await self._update_session(session_id, event)
            
            logger.info(f"Tracked page view: {page_url} for user: {user_id}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Error tracking page view: {str(e)}")
            raise
    
    async def _update_session(self, session_id: str, event: AnalyticsEvent):
        """Update session data with new event"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "user_id": event.user_id,
                "start_time": event.timestamp,
                "last_activity": event.timestamp,
                "page_views": 0,
                "events": 0,
                "pages_visited": set()
            }
        
        session = self.sessions[session_id]
        session["last_activity"] = event.timestamp
        session["events"] += 1
        
        if event.event_type == EventType.PAGE_VIEW:
            session["page_views"] += 1
            session["pages_visited"].add(event.page_url)
    
    async def _process_conversion_tracking(self, event: AnalyticsEvent):
        """Process event for conversion tracking"""
        # Check if event matches any conversion goals
        conversion_events = [
            "user_signup",
            "subscription_created",
            "payment_completed",
            "trial_started",
            "feature_activated"
        ]
        
        if event.event_name in conversion_events:
            await self.track_conversion(
                user_id=event.user_id,
                conversion_type=event.event_name,
                value=event.properties.get("value", 0),
                properties=event.properties
            )
    
    async def track_conversion(self,
                              user_id: str,
                              conversion_type: str,
                              value: float = 0,
                              properties: Dict[str, Any] = None) -> str:
        """Track a conversion event"""
        try:
            conversion_event = AnalyticsEvent(
                event_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id="",  # Will be filled from current session
                event_type=EventType.CONVERSION,
                event_name=conversion_type,
                properties={
                    "conversion_value": value,
                    "conversion_type": conversion_type,
                    **(properties or {})
                },
                timestamp=datetime.utcnow(),
                page_url="",
                user_agent="",
                ip_address=""
            )
            
            self.events.append(conversion_event)
            logger.info(f"Tracked conversion: {conversion_type} for user: {user_id}")
            return conversion_event.event_id
            
        except Exception as e:
            logger.error(f"Error tracking conversion: {str(e)}")
            raise
    
    async def get_user_behavior_data(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user behavior analytics"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            user_events = [
                event for event in self.events
                if event.user_id == user_id and event.timestamp >= cutoff_date
            ]
            
            if not user_events:
                return {"user_id": user_id, "events": 0, "sessions": 0}
            
            # Calculate behavior metrics
            page_views = len([e for e in user_events if e.event_type == EventType.PAGE_VIEW])
            actions = len([e for e in user_events if e.event_type == EventType.USER_ACTION])
            conversions = len([e for e in user_events if e.event_type == EventType.CONVERSION])
            
            # Get unique sessions
            unique_sessions = len(set(e.session_id for e in user_events if e.session_id))
            
            # Calculate engagement metrics
            pages_visited = set()
            for event in user_events:
                if event.event_type == EventType.PAGE_VIEW:
                    pages_visited.add(event.page_url)
            
            return {
                "user_id": user_id,
                "total_events": len(user_events),
                "page_views": page_views,
                "actions": actions,
                "conversions": conversions,
                "sessions": unique_sessions,
                "unique_pages": len(pages_visited),
                "engagement_score": self._calculate_engagement_score(user_events),
                "last_activity": max(e.timestamp for e in user_events).isoformat(),
                "first_activity": min(e.timestamp for e in user_events).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user behavior data: {str(e)}")
            raise
    
    def _calculate_engagement_score(self, events: List[AnalyticsEvent]) -> float:
        """Calculate user engagement score"""
        if not events:
            return 0.0
        
        # Simple engagement scoring based on activity
        page_views = len([e for e in events if e.event_type == EventType.PAGE_VIEW])
        actions = len([e for e in events if e.event_type == EventType.USER_ACTION])
        conversions = len([e for e in events if e.event_type == EventType.CONVERSION])
        
        # Weight different actions
        score = (page_views * 1) + (actions * 2) + (conversions * 10)
        
        # Normalize by time period
        if len(events) > 1:
            time_span = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).days
            if time_span > 0:
                score = score / time_span
        
        return min(score, 100.0)  # Cap at 100
    
    async def create_user_segment(self,
                                 name: str,
                                 criteria: Dict[str, Any]) -> str:
        """Create a user segment based on criteria"""
        try:
            segment_id = str(uuid.uuid4())
            
            # Find users matching criteria
            matching_users = await self._find_users_by_criteria(criteria)
            
            segment = UserSegment(
                segment_id=segment_id,
                name=name,
                criteria=criteria,
                user_count=len(matching_users),
                properties={
                    "created_at": datetime.utcnow().isoformat(),
                    "matching_users": matching_users[:100]  # Store sample
                }
            )
            
            self.user_segments[segment_id] = segment
            logger.info(f"Created user segment: {name} with {len(matching_users)} users")
            return segment_id
            
        except Exception as e:
            logger.error(f"Error creating user segment: {str(e)}")
            raise
    
    async def _find_users_by_criteria(self, criteria: Dict[str, Any]) -> List[str]:
        """Find users matching segmentation criteria"""
        matching_users = set()
        
        for event in self.events:
            user_matches = True
            
            # Check event-based criteria
            if "event_name" in criteria:
                if event.event_name != criteria["event_name"]:
                    continue
            
            if "event_type" in criteria:
                if event.event_type.value != criteria["event_type"]:
                    continue
            
            # Check time-based criteria
            if "days_since_last_activity" in criteria:
                days_ago = (datetime.utcnow() - event.timestamp).days
                if days_ago > criteria["days_since_last_activity"]:
                    continue
            
            # Check property-based criteria
            if "properties" in criteria:
                for key, value in criteria["properties"].items():
                    if key not in event.properties or event.properties[key] != value:
                        user_matches = False
                        break
            
            if user_matches:
                matching_users.add(event.user_id)
        
        return list(matching_users)
    
    async def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics summary for dashboard"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            recent_events = [
                event for event in self.events
                if event.timestamp >= cutoff_date
            ]
            
            if not recent_events:
                return {"total_events": 0, "unique_users": 0, "sessions": 0}
            
            unique_users = len(set(e.user_id for e in recent_events))
            unique_sessions = len(set(e.session_id for e in recent_events if e.session_id))
            page_views = len([e for e in recent_events if e.event_type == EventType.PAGE_VIEW])
            conversions = len([e for e in recent_events if e.event_type == EventType.CONVERSION])
            
            # Calculate daily metrics
            daily_metrics = {}
            for event in recent_events:
                date_key = event.timestamp.date().isoformat()
                if date_key not in daily_metrics:
                    daily_metrics[date_key] = {"events": 0, "users": set(), "conversions": 0}
                
                daily_metrics[date_key]["events"] += 1
                daily_metrics[date_key]["users"].add(event.user_id)
                if event.event_type == EventType.CONVERSION:
                    daily_metrics[date_key]["conversions"] += 1
            
            # Convert sets to counts
            for date_key in daily_metrics:
                daily_metrics[date_key]["users"] = len(daily_metrics[date_key]["users"])
            
            return {
                "total_events": len(recent_events),
                "unique_users": unique_users,
                "sessions": unique_sessions,
                "page_views": page_views,
                "conversions": conversions,
                "conversion_rate": (conversions / unique_users * 100) if unique_users > 0 else 0,
                "daily_metrics": daily_metrics,
                "top_pages": self._get_top_pages(recent_events),
                "top_events": self._get_top_events(recent_events)
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {str(e)}")
            raise
    
    def _get_top_pages(self, events: List[AnalyticsEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top pages by views"""
        page_counts = {}
        for event in events:
            if event.event_type == EventType.PAGE_VIEW:
                page_counts[event.page_url] = page_counts.get(event.page_url, 0) + 1
        
        return [
            {"page": page, "views": count}
            for page, count in sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _get_top_events(self, events: List[AnalyticsEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top events by frequency"""
        event_counts = {}
        for event in events:
            if event.event_type == EventType.USER_ACTION:
                event_counts[event.event_name] = event_counts.get(event.event_name, 0) + 1
        
        return [
            {"event": event_name, "count": count}
            for event_name, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]

# Global analytics tracker instance
analytics_tracker = AnalyticsTracker()