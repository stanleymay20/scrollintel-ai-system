"""
Intelligent Notification System - Smart User Communication

This module provides intelligent notification capabilities including:
- Smart notification prioritization and filtering
- User preference-based notification delivery
- Context-aware notification timing
- Notification batching and aggregation
- Multi-channel notification delivery

Requirements: 6.1, 6.2, 6.3
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"


class NotificationFrequency(Enum):
    """Notification frequency preferences"""
    IMMEDIATE = "immediate"
    BATCHED_5MIN = "batched_5min"
    BATCHED_15MIN = "batched_15min"
    BATCHED_HOURLY = "batched_hourly"
    DAILY_DIGEST = "daily_digest"
    WEEKLY_DIGEST = "weekly_digest"


class UserActivityState(Enum):
    """User activity states for smart timing"""
    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    DO_NOT_DISTURB = "do_not_disturb"
    OFFLINE = "offline"


@dataclass
class NotificationRule:
    """User notification preferences and rules"""
    user_id: str
    notification_type: str
    channels: List[NotificationChannel]
    frequency: NotificationFrequency
    priority_threshold: str = "medium"  # low, medium, high, critical
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    keywords_filter: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class NotificationTemplate:
    """Notification message templates"""
    template_id: str
    notification_type: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    action_templates: List[Dict[str, str]] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)


@dataclass
class DeliveryAttempt:
    """Notification delivery attempt tracking"""
    attempt_id: str
    notification_id: str
    channel: NotificationChannel
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class NotificationBatch:
    """Batched notifications for digest delivery"""
    batch_id: str
    user_id: str
    frequency: NotificationFrequency
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_delivery: Optional[datetime] = None


class IntelligentNotificationSystem:
    """
    Intelligent notification system that learns user preferences and
    optimizes notification delivery timing and channels
    """
    
    def __init__(self):
        self.user_rules: Dict[str, List[NotificationRule]] = defaultdict(list)
        self.user_activity: Dict[str, UserActivityState] = {}
        self.user_last_seen: Dict[str, datetime] = {}
        self.notification_templates: Dict[str, NotificationTemplate] = {}
        self.delivery_history: Dict[str, List[DeliveryAttempt]] = defaultdict(list)
        self.pending_batches: Dict[str, List[NotificationBatch]] = defaultdict(list)
        self.notification_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Channel handlers
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Background tasks
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        
        # Start background tasks
        self._start_background_tasks()
    
    async def send_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        priority: str = "medium",
        data: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, str]]] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[str, Any]:
        """Send intelligent notification to user"""
        try:
            notification_id = self._generate_notification_id()
            
            # Get user rules for this notification type
            applicable_rules = self._get_applicable_rules(user_id, notification_type, priority)
            
            if not applicable_rules:
                logger.info(f"No applicable rules for user {user_id}, notification type {notification_type}")
                return {"notification_id": notification_id, "status": "filtered", "reason": "no_rules"}
            
            # Check if user is in quiet hours
            if self._is_quiet_hours(user_id):
                if priority not in ["high", "critical"]:
                    logger.info(f"User {user_id} in quiet hours, deferring notification")
                    return await self._defer_notification(user_id, notification_id, notification_type, 
                                                         title, message, priority, data, actions)
            
            # Check user activity state for smart timing
            activity_state = self.user_activity.get(user_id, UserActivityState.OFFLINE)
            if await self._should_defer_for_activity(user_id, activity_state, priority):
                return await self._defer_notification(user_id, notification_id, notification_type,
                                                     title, message, priority, data, actions)
            
            # Determine delivery channels and frequency
            delivery_plan = await self._create_delivery_plan(user_id, applicable_rules, channels)
            
            # Check if should batch
            if await self._should_batch_notification(user_id, notification_type, priority, delivery_plan):
                return await self._add_to_batch(user_id, notification_id, notification_type,
                                              title, message, priority, data, actions, delivery_plan)
            
            # Send immediately
            results = await self._deliver_notification(
                notification_id, user_id, notification_type, title, message,
                priority, data, actions, delivery_plan
            )
            
            # Update statistics
            await self._update_notification_stats(user_id, notification_type, priority, results)
            
            return {
                "notification_id": notification_id,
                "status": "sent",
                "delivery_results": results
            }
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return {"notification_id": notification_id, "status": "error", "error": str(e)}
    
    async def set_user_notification_rules(
        self,
        user_id: str,
        rules: List[Dict[str, Any]]
    ) -> None:
        """Set notification rules for a user"""
        try:
            user_rules = []
            for rule_data in rules:
                rule = NotificationRule(
                    user_id=user_id,
                    notification_type=rule_data["notification_type"],
                    channels=[NotificationChannel(ch) for ch in rule_data["channels"]],
                    frequency=NotificationFrequency(rule_data["frequency"]),
                    priority_threshold=rule_data.get("priority_threshold", "medium"),
                    quiet_hours_start=rule_data.get("quiet_hours_start"),
                    quiet_hours_end=rule_data.get("quiet_hours_end"),
                    keywords_filter=rule_data.get("keywords_filter", []),
                    exclude_keywords=rule_data.get("exclude_keywords", []),
                    enabled=rule_data.get("enabled", True)
                )
                user_rules.append(rule)
            
            self.user_rules[user_id] = user_rules
            logger.info(f"Updated notification rules for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error setting user notification rules: {e}")
            raise
    
    async def update_user_activity(
        self,
        user_id: str,
        activity_state: UserActivityState
    ) -> None:
        """Update user activity state for smart notification timing"""
        self.user_activity[user_id] = activity_state
        self.user_last_seen[user_id] = datetime.utcnow()
        
        # If user becomes active, check for deferred notifications
        if activity_state == UserActivityState.ACTIVE:
            await self._process_deferred_notifications(user_id)
    
    async def register_channel_handler(
        self,
        channel: NotificationChannel,
        handler: Callable
    ) -> None:
        """Register handler for notification channel"""
        self.channel_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel.value}")
    
    async def add_notification_template(
        self,
        template_id: str,
        notification_type: str,
        channel: NotificationChannel,
        subject_template: str,
        body_template: str,
        action_templates: Optional[List[Dict[str, str]]] = None,
        variables: Optional[List[str]] = None
    ) -> None:
        """Add notification template"""
        template = NotificationTemplate(
            template_id=template_id,
            notification_type=notification_type,
            channel=channel,
            subject_template=subject_template,
            body_template=body_template,
            action_templates=action_templates or [],
            variables=variables or []
        )
        
        self.notification_templates[template_id] = template
        logger.info(f"Added notification template: {template_id}")
    
    async def get_user_notification_stats(self, user_id: str) -> Dict[str, Any]:
        """Get notification statistics for user"""
        stats = self.notification_stats.get(user_id, {})
        
        # Add recent activity
        recent_deliveries = []
        if user_id in self.delivery_history:
            recent_deliveries = [
                {
                    "notification_id": attempt.notification_id,
                    "channel": attempt.channel.value,
                    "timestamp": attempt.timestamp.isoformat(),
                    "success": attempt.success
                }
                for attempt in self.delivery_history[user_id][-10:]  # Last 10
            ]
        
        return {
            "user_id": user_id,
            "total_sent": stats.get("total_sent", 0),
            "total_delivered": stats.get("total_delivered", 0),
            "total_failed": stats.get("total_failed", 0),
            "delivery_rate": stats.get("delivery_rate", 0.0),
            "preferred_channels": stats.get("preferred_channels", []),
            "recent_deliveries": recent_deliveries,
            "active_batches": len(self.pending_batches.get(user_id, [])),
            "last_activity": self.user_last_seen.get(user_id, datetime.min).isoformat()
        }
    
    async def get_notification_analytics(self) -> Dict[str, Any]:
        """Get system-wide notification analytics"""
        total_users = len(self.user_rules)
        total_notifications = sum(
            stats.get("total_sent", 0) 
            for stats in self.notification_stats.values()
        )
        
        # Channel performance
        channel_stats = defaultdict(lambda: {"sent": 0, "delivered": 0, "failed": 0})
        for user_attempts in self.delivery_history.values():
            for attempt in user_attempts:
                channel = attempt.channel.value
                channel_stats[channel]["sent"] += 1
                if attempt.success:
                    channel_stats[channel]["delivered"] += 1
                else:
                    channel_stats[channel]["failed"] += 1
        
        return {
            "total_users": total_users,
            "total_notifications": total_notifications,
            "channel_performance": dict(channel_stats),
            "active_batches": sum(len(batches) for batches in self.pending_batches.values()),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # Private methods
    
    def _get_applicable_rules(
        self,
        user_id: str,
        notification_type: str,
        priority: str
    ) -> List[NotificationRule]:
        """Get applicable notification rules for user and type"""
        user_rules = self.user_rules.get(user_id, [])
        applicable = []
        
        priority_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        
        for rule in user_rules:
            if not rule.enabled:
                continue
            
            # Check notification type (support wildcards)
            if rule.notification_type != "*" and rule.notification_type != notification_type:
                continue
            
            # Check priority threshold
            rule_priority_level = priority_levels.get(rule.priority_threshold, 1)
            notification_priority_level = priority_levels.get(priority, 1)
            
            if notification_priority_level >= rule_priority_level:
                applicable.append(rule)
        
        return applicable
    
    def _is_quiet_hours(self, user_id: str) -> bool:
        """Check if user is in quiet hours"""
        user_rules = self.user_rules.get(user_id, [])
        
        for rule in user_rules:
            if rule.quiet_hours_start and rule.quiet_hours_end:
                now = datetime.utcnow().time()
                start_time = datetime.strptime(rule.quiet_hours_start, "%H:%M").time()
                end_time = datetime.strptime(rule.quiet_hours_end, "%H:%M").time()
                
                if start_time <= end_time:
                    # Same day quiet hours
                    if start_time <= now <= end_time:
                        return True
                else:
                    # Overnight quiet hours
                    if now >= start_time or now <= end_time:
                        return True
        
        return False
    
    async def _should_defer_for_activity(
        self,
        user_id: str,
        activity_state: UserActivityState,
        priority: str
    ) -> bool:
        """Check if notification should be deferred based on user activity"""
        if priority in ["high", "critical"]:
            return False
        
        if activity_state == UserActivityState.DO_NOT_DISTURB:
            return True
        
        if activity_state == UserActivityState.AWAY:
            # Defer non-urgent notifications for away users
            return priority == "low"
        
        return False
    
    async def _create_delivery_plan(
        self,
        user_id: str,
        rules: List[NotificationRule],
        preferred_channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[str, Any]:
        """Create delivery plan based on rules and preferences"""
        channels = set()
        frequency = NotificationFrequency.IMMEDIATE
        
        # Collect channels and determine frequency
        for rule in rules:
            channels.update(rule.channels)
            # Use most restrictive frequency
            if rule.frequency != NotificationFrequency.IMMEDIATE:
                frequency = rule.frequency
        
        # Override with preferred channels if provided
        if preferred_channels:
            channels = set(preferred_channels)
        
        return {
            "channels": list(channels),
            "frequency": frequency,
            "rules": rules
        }
    
    async def _should_batch_notification(
        self,
        user_id: str,
        notification_type: str,
        priority: str,
        delivery_plan: Dict[str, Any]
    ) -> bool:
        """Check if notification should be batched"""
        frequency = delivery_plan["frequency"]
        
        # Never batch critical notifications
        if priority == "critical":
            return False
        
        # Batch based on frequency setting
        return frequency != NotificationFrequency.IMMEDIATE
    
    async def _add_to_batch(
        self,
        user_id: str,
        notification_id: str,
        notification_type: str,
        title: str,
        message: str,
        priority: str,
        data: Optional[Dict[str, Any]],
        actions: Optional[List[Dict[str, str]]],
        delivery_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add notification to batch for later delivery"""
        frequency = delivery_plan["frequency"]
        
        # Find or create batch
        batch = None
        for existing_batch in self.pending_batches[user_id]:
            if existing_batch.frequency == frequency:
                batch = existing_batch
                break
        
        if not batch:
            batch = NotificationBatch(
                batch_id=self._generate_batch_id(),
                user_id=user_id,
                frequency=frequency,
                scheduled_delivery=self._calculate_batch_delivery_time(frequency)
            )
            self.pending_batches[user_id].append(batch)
        
        # Add notification to batch
        batch.notifications.append({
            "notification_id": notification_id,
            "notification_type": notification_type,
            "title": title,
            "message": message,
            "priority": priority,
            "data": data,
            "actions": actions,
            "delivery_plan": delivery_plan,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Added notification {notification_id} to batch {batch.batch_id}")
        
        return {
            "notification_id": notification_id,
            "status": "batched",
            "batch_id": batch.batch_id,
            "scheduled_delivery": batch.scheduled_delivery.isoformat()
        }
    
    async def _deliver_notification(
        self,
        notification_id: str,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        priority: str,
        data: Optional[Dict[str, Any]],
        actions: Optional[List[Dict[str, str]]],
        delivery_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deliver notification through specified channels"""
        results = {}
        
        for channel in delivery_plan["channels"]:
            try:
                # Get template for channel
                template = self._get_template(notification_type, channel)
                
                # Format message using template
                formatted_message = await self._format_message(
                    template, title, message, data, actions
                )
                
                # Deliver through channel
                if channel in self.channel_handlers:
                    handler = self.channel_handlers[channel]
                    result = await handler(user_id, formatted_message)
                    success = result.get("success", False)
                else:
                    # Default in-app delivery
                    result = await self._deliver_in_app(user_id, formatted_message)
                    success = result.get("success", True)
                
                # Record delivery attempt
                attempt = DeliveryAttempt(
                    attempt_id=self._generate_attempt_id(),
                    notification_id=notification_id,
                    channel=channel,
                    timestamp=datetime.utcnow(),
                    success=success,
                    error_message=result.get("error"),
                    response_data=result.get("response_data")
                )
                
                self.delivery_history[user_id].append(attempt)
                results[channel.value] = result
                
            except Exception as e:
                logger.error(f"Error delivering notification via {channel.value}: {e}")
                results[channel.value] = {"success": False, "error": str(e)}
        
        return results
    
    async def _defer_notification(
        self,
        user_id: str,
        notification_id: str,
        notification_type: str,
        title: str,
        message: str,
        priority: str,
        data: Optional[Dict[str, Any]],
        actions: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """Defer notification for later delivery"""
        # For now, add to immediate batch - in production, you'd store in database
        return await self._add_to_batch(
            user_id, notification_id, notification_type, title, message,
            priority, data, actions, {"frequency": NotificationFrequency.BATCHED_15MIN, "channels": [NotificationChannel.IN_APP]}
        )
    
    async def _process_deferred_notifications(self, user_id: str) -> None:
        """Process deferred notifications when user becomes active"""
        # This would typically query deferred notifications from database
        # For now, just process any pending batches
        if user_id in self.pending_batches:
            for batch in self.pending_batches[user_id][:]:  # Copy list to avoid modification during iteration
                if batch.frequency in [NotificationFrequency.BATCHED_5MIN, NotificationFrequency.BATCHED_15MIN]:
                    await self._deliver_batch(batch)
                    self.pending_batches[user_id].remove(batch)
    
    def _get_template(
        self,
        notification_type: str,
        channel: NotificationChannel
    ) -> Optional[NotificationTemplate]:
        """Get notification template for type and channel"""
        for template in self.notification_templates.values():
            if template.notification_type == notification_type and template.channel == channel:
                return template
        
        # Return default template
        return NotificationTemplate(
            template_id="default",
            notification_type=notification_type,
            channel=channel,
            subject_template="{title}",
            body_template="{message}"
        )
    
    async def _format_message(
        self,
        template: Optional[NotificationTemplate],
        title: str,
        message: str,
        data: Optional[Dict[str, Any]],
        actions: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """Format message using template"""
        if not template:
            return {
                "title": title,
                "message": message,
                "data": data,
                "actions": actions
            }
        
        # Simple template formatting - in production, use proper template engine
        variables = {"title": title, "message": message}
        if data:
            variables.update(data)
        
        formatted_subject = template.subject_template.format(**variables)
        formatted_body = template.body_template.format(**variables)
        
        return {
            "title": formatted_subject,
            "message": formatted_body,
            "data": data,
            "actions": actions or template.action_templates
        }
    
    async def _deliver_in_app(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Default in-app notification delivery"""
        # This would integrate with your WebSocket or real-time system
        logger.info(f"In-app notification for user {user_id}: {message['title']}")
        return {"success": True, "delivery_method": "in_app"}
    
    def _calculate_batch_delivery_time(self, frequency: NotificationFrequency) -> datetime:
        """Calculate when batch should be delivered"""
        now = datetime.utcnow()
        
        if frequency == NotificationFrequency.BATCHED_5MIN:
            return now + timedelta(minutes=5)
        elif frequency == NotificationFrequency.BATCHED_15MIN:
            return now + timedelta(minutes=15)
        elif frequency == NotificationFrequency.BATCHED_HOURLY:
            return now + timedelta(hours=1)
        elif frequency == NotificationFrequency.DAILY_DIGEST:
            # Next day at 9 AM
            next_day = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return next_day
        elif frequency == NotificationFrequency.WEEKLY_DIGEST:
            # Next Monday at 9 AM
            days_ahead = 7 - now.weekday()
            next_monday = now + timedelta(days=days_ahead)
            return next_monday.replace(hour=9, minute=0, second=0, microsecond=0)
        
        return now
    
    async def _deliver_batch(self, batch: NotificationBatch) -> None:
        """Deliver batched notifications"""
        try:
            # Create digest message
            digest_title = f"Notification Digest ({len(batch.notifications)} items)"
            digest_message = self._create_digest_message(batch.notifications)
            
            # Deliver digest
            await self._deliver_in_app(batch.user_id, {
                "title": digest_title,
                "message": digest_message,
                "data": {"batch_id": batch.batch_id, "notifications": batch.notifications},
                "actions": [{"type": "view_all", "label": "View All"}]
            })
            
            logger.info(f"Delivered batch {batch.batch_id} with {len(batch.notifications)} notifications")
            
        except Exception as e:
            logger.error(f"Error delivering batch {batch.batch_id}: {e}")
    
    def _create_digest_message(self, notifications: List[Dict[str, Any]]) -> str:
        """Create digest message from batched notifications"""
        if len(notifications) == 1:
            return notifications[0]["message"]
        
        summary = f"You have {len(notifications)} new notifications:\n\n"
        for i, notif in enumerate(notifications[:5], 1):  # Show first 5
            summary += f"{i}. {notif['title']}\n"
        
        if len(notifications) > 5:
            summary += f"\n... and {len(notifications) - 5} more"
        
        return summary
    
    async def _update_notification_stats(
        self,
        user_id: str,
        notification_type: str,
        priority: str,
        results: Dict[str, Any]
    ) -> None:
        """Update notification statistics"""
        stats = self.notification_stats[user_id]
        
        stats["total_sent"] = stats.get("total_sent", 0) + 1
        
        delivered = any(result.get("success", False) for result in results.values())
        if delivered:
            stats["total_delivered"] = stats.get("total_delivered", 0) + 1
        else:
            stats["total_failed"] = stats.get("total_failed", 0) + 1
        
        # Update delivery rate
        if stats["total_sent"] > 0:
            stats["delivery_rate"] = stats["total_delivered"] / stats["total_sent"]
    
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        return f"notif_{int(datetime.utcnow().timestamp() * 1000)}"
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        return f"batch_{int(datetime.utcnow().timestamp() * 1000)}"
    
    def _generate_attempt_id(self) -> str:
        """Generate unique attempt ID"""
        return f"attempt_{int(datetime.utcnow().timestamp() * 1000)}"
    
    def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        async def process_batches():
            while True:
                try:
                    now = datetime.utcnow()
                    
                    # Process all pending batches
                    for user_id, batches in list(self.pending_batches.items()):
                        for batch in batches[:]:  # Copy to avoid modification during iteration
                            if batch.scheduled_delivery and batch.scheduled_delivery <= now:
                                await self._deliver_batch(batch)
                                batches.remove(batch)
                    
                    # Sleep for 1 minute
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in batch processor: {e}")
                    await asyncio.sleep(60)
        
        self._batch_processor_task = asyncio.create_task(process_batches())


# Global instance
intelligent_notification_system = IntelligentNotificationSystem()