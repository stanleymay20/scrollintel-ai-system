"""
Stakeholder Notification Engine

Handles immediate and appropriate stakeholder communication during crisis situations.
Implements stakeholder prioritization, message customization, and delivery tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

from ..models.crisis_communication_models import (
    Stakeholder, StakeholderType, NotificationMessage, NotificationTemplate,
    NotificationBatch, StakeholderGroup, CommunicationPlan, NotificationMetrics,
    NotificationPriority, NotificationChannel, NotificationStatus
)
from ..models.crisis_models_simple import Crisis, CrisisType, SeverityLevel


class StakeholderNotificationEngine:
    """Engine for managing stakeholder notifications during crisis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.groups: Dict[str, StakeholderGroup] = {}
        self.communication_plans: Dict[str, CommunicationPlan] = {}
        self.notification_queue: List[NotificationMessage] = []
        self.delivery_providers = self._initialize_delivery_providers()
        
    def _initialize_delivery_providers(self) -> Dict[str, Any]:
        """Initialize notification delivery providers"""
        return {
            "email": {"enabled": True, "rate_limit": 100},
            "sms": {"enabled": True, "rate_limit": 50},
            "phone": {"enabled": True, "rate_limit": 10},
            "slack": {"enabled": True, "rate_limit": 200},
            "teams": {"enabled": True, "rate_limit": 200},
            "push": {"enabled": True, "rate_limit": 500},
            "portal": {"enabled": True, "rate_limit": 1000}
        }
    
    async def notify_stakeholders_immediate(
        self, 
        crisis: Crisis, 
        stakeholder_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send immediate notifications to stakeholders about crisis
        
        Args:
            crisis: Crisis information
            stakeholder_ids: Specific stakeholders to notify (optional)
            
        Returns:
            Notification results and metrics
        """
        try:
            # Determine stakeholders to notify
            if stakeholder_ids:
                stakeholders = [self.stakeholders[sid] for sid in stakeholder_ids if sid in self.stakeholders]
            else:
                stakeholders = self._prioritize_stakeholders_for_crisis(crisis)
            
            # Create communication plan if not exists
            comm_plan = self._get_or_create_communication_plan(crisis)
            
            # Generate notifications
            notifications = []
            for stakeholder in stakeholders:
                notification = await self._create_stakeholder_notification(
                    crisis, stakeholder, comm_plan, immediate=True
                )
                if notification:
                    notifications.append(notification)
            
            # Send notifications in batches
            batch = NotificationBatch(
                crisis_id=crisis.id,
                name=f"Immediate Crisis Notification - {crisis.crisis_type.value}",
                description=f"Immediate stakeholder notification for {crisis.crisis_type.value}",
                stakeholder_groups=[s.stakeholder_type for s in stakeholders],
                messages=[n.id for n in notifications]
            )
            
            results = await self._send_notification_batch(batch, notifications)
            
            # Track metrics
            metrics = self._calculate_notification_metrics(notifications)
            
            self.logger.info(f"Sent immediate crisis notifications for crisis {crisis.id}")
            
            return {
                "success": True,
                "batch_id": batch.id,
                "notifications_sent": len(notifications),
                "stakeholders_notified": len(stakeholders),
                "delivery_results": results,
                "metrics": asdict(metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send immediate stakeholder notifications: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "notifications_sent": 0
            }
    
    def _prioritize_stakeholders_for_crisis(self, crisis: Crisis) -> List[Stakeholder]:
        """Prioritize stakeholders based on crisis type and severity"""
        relevant_stakeholders = []
        
        for stakeholder in self.stakeholders.values():
            # Calculate relevance score
            relevance_score = self._calculate_stakeholder_relevance(stakeholder, crisis)
            
            if relevance_score > 0:
                relevant_stakeholders.append((stakeholder, relevance_score))
        
        # Sort by relevance score (descending) and priority level
        relevant_stakeholders.sort(
            key=lambda x: (x[1], x[0].priority_level.value, x[0].influence_level),
            reverse=True
        )
        
        return [stakeholder for stakeholder, _ in relevant_stakeholders]
    
    def _calculate_stakeholder_relevance(self, stakeholder: Stakeholder, crisis: Crisis) -> float:
        """Calculate how relevant a stakeholder is to the crisis"""
        base_relevance = stakeholder.crisis_relevance.get(crisis.crisis_type.value, 0)
        
        # Adjust based on severity
        severity_multiplier = {
            SeverityLevel.LOW: 0.5,
            SeverityLevel.MEDIUM: 1.0,
            SeverityLevel.HIGH: 1.5,
            SeverityLevel.CRITICAL: 2.0
        }.get(crisis.severity_level, 1.0)
        
        # Adjust based on stakeholder type and crisis type
        type_relevance = self._get_stakeholder_type_relevance(
            stakeholder.stakeholder_type, crisis.crisis_type
        )
        
        return base_relevance * severity_multiplier * type_relevance
    
    def _get_stakeholder_type_relevance(
        self, 
        stakeholder_type: StakeholderType, 
        crisis_type: CrisisType
    ) -> float:
        """Get relevance multiplier based on stakeholder type and crisis type"""
        relevance_matrix = {
            CrisisType.SECURITY_BREACH: {
                StakeholderType.BOARD_MEMBER: 1.0,
                StakeholderType.EXECUTIVE: 1.0,
                StakeholderType.CUSTOMER: 0.9,
                StakeholderType.REGULATOR: 0.8,
                StakeholderType.MEDIA: 0.7,
                StakeholderType.EMPLOYEE: 0.6
            },
            CrisisType.SYSTEM_OUTAGE: {
                StakeholderType.CUSTOMER: 1.0,
                StakeholderType.EXECUTIVE: 0.9,
                StakeholderType.EMPLOYEE: 0.8,
                StakeholderType.PARTNER: 0.7,
                StakeholderType.BOARD_MEMBER: 0.6
            },
            CrisisType.FINANCIAL_CRISIS: {
                StakeholderType.BOARD_MEMBER: 1.0,
                StakeholderType.INVESTOR: 1.0,
                StakeholderType.EXECUTIVE: 0.9,
                StakeholderType.REGULATOR: 0.8,
                StakeholderType.EMPLOYEE: 0.7
            },
            CrisisType.REGULATORY_ISSUE: {
                StakeholderType.REGULATOR: 1.0,
                StakeholderType.BOARD_MEMBER: 0.9,
                StakeholderType.EXECUTIVE: 0.9,
                StakeholderType.MEDIA: 0.6,
                StakeholderType.CUSTOMER: 0.5
            },
            CrisisType.REPUTATION_DAMAGE: {
                StakeholderType.MEDIA: 1.0,
                StakeholderType.CUSTOMER: 0.9,
                StakeholderType.BOARD_MEMBER: 0.8,
                StakeholderType.EXECUTIVE: 0.8,
                StakeholderType.PUBLIC: 0.7
            }
        }
        
        return relevance_matrix.get(crisis_type, {}).get(stakeholder_type, 0.5)
    
    async def _create_stakeholder_notification(
        self,
        crisis: Crisis,
        stakeholder: Stakeholder,
        comm_plan: CommunicationPlan,
        immediate: bool = False
    ) -> Optional[NotificationMessage]:
        """Create customized notification for stakeholder"""
        try:
            # Select appropriate template
            template = self._select_notification_template(
                crisis.crisis_type.value, stakeholder.stakeholder_type
            )
            
            if not template:
                self.logger.warning(f"No template found for {stakeholder.stakeholder_type} and {crisis.crisis_type}")
                return None
            
            # Select best channel
            channel = self._select_optimal_channel(stakeholder, immediate)
            
            # Customize message content
            content = self._customize_message_content(template, crisis, stakeholder)
            
            # Determine priority
            priority = self._determine_notification_priority(crisis, stakeholder, immediate)
            
            notification = NotificationMessage(
                crisis_id=crisis.id,
                stakeholder_id=stakeholder.id,
                template_id=template.id,
                channel=channel,
                priority=priority,
                subject=content["subject"],
                content=content["body"],
                variables=content["variables"],
                scheduled_time=datetime.utcnow() if immediate else None
            )
            
            return notification
            
        except Exception as e:
            self.logger.error(f"Failed to create notification for stakeholder {stakeholder.id}: {str(e)}")
            return None
    
    def _select_notification_template(
        self, 
        crisis_type: str, 
        stakeholder_type: StakeholderType
    ) -> Optional[NotificationTemplate]:
        """Select appropriate notification template"""
        # Find exact match first
        for template in self.templates.values():
            if (template.crisis_type == crisis_type and 
                template.stakeholder_type == stakeholder_type):
                return template
        
        # Find generic template for stakeholder type
        for template in self.templates.values():
            if (template.crisis_type == "generic" and 
                template.stakeholder_type == stakeholder_type):
                return template
        
        # Find generic template for crisis type
        for template in self.templates.values():
            if (template.crisis_type == crisis_type and 
                template.stakeholder_type == StakeholderType.EMPLOYEE):
                return template
        
        return None
    
    def _select_optimal_channel(
        self, 
        stakeholder: Stakeholder, 
        immediate: bool = False
    ) -> NotificationChannel:
        """Select optimal communication channel for stakeholder"""
        if immediate:
            # For immediate notifications, prefer faster channels
            priority_channels = [
                NotificationChannel.SMS,
                NotificationChannel.PHONE,
                NotificationChannel.PUSH,
                NotificationChannel.SLACK,
                NotificationChannel.EMAIL
            ]
        else:
            # For regular notifications, use preferred channels
            priority_channels = stakeholder.preferred_channels or [NotificationChannel.EMAIL]
        
        # Select first available channel
        for channel in priority_channels:
            if self._is_channel_available(channel, stakeholder):
                return channel
        
        # Fallback to email
        return NotificationChannel.EMAIL
    
    def _is_channel_available(
        self, 
        channel: NotificationChannel, 
        stakeholder: Stakeholder
    ) -> bool:
        """Check if communication channel is available for stakeholder"""
        channel_contact_map = {
            NotificationChannel.EMAIL: "email",
            NotificationChannel.SMS: "phone",
            NotificationChannel.PHONE: "phone",
            NotificationChannel.SLACK: "slack",
            NotificationChannel.TEAMS: "teams"
        }
        
        required_contact = channel_contact_map.get(channel)
        if required_contact:
            return required_contact in stakeholder.contact_info
        
        return True  # Portal and push notifications don't require specific contact info
    
    def _customize_message_content(
        self,
        template: NotificationTemplate,
        crisis: Crisis,
        stakeholder: Stakeholder
    ) -> Dict[str, Any]:
        """Customize message content for specific stakeholder"""
        variables = {
            "stakeholder_name": stakeholder.name,
            "crisis_type": crisis.crisis_type.value.replace("_", " ").title(),
            "severity": crisis.severity_level.value.title(),
            "start_time": crisis.start_time.strftime("%Y-%m-%d %H:%M UTC"),
            "affected_areas": ", ".join(crisis.affected_areas),
            "current_status": crisis.current_status.value.replace("_", " ").title(),
            "crisis_id": crisis.id
        }
        
        # Add common variables that might be needed
        variables.update({
            "business_impact": self._get_business_impact_summary(crisis),
            "impact_description": self._get_customer_impact_description(crisis),
            "action_required": self._get_employee_action_required(crisis, stakeholder),
            "department": stakeholder.department,
            "sender_name": "Crisis Management System"
        })
        
        # Apply template variables with safe formatting
        try:
            subject = template.subject_template.format(**variables)
        except KeyError as e:
            self.logger.warning(f"Missing variable {e} in subject template, using fallback")
            subject = f"Crisis Alert: {crisis.crisis_type.value.replace('_', ' ').title()}"
        
        try:
            body = template.body_template.format(**variables)
        except KeyError as e:
            self.logger.warning(f"Missing variable {e} in body template, using fallback")
            body = f"Dear {stakeholder.name},\n\nWe are experiencing a {crisis.crisis_type.value.replace('_', ' ')} situation. Please standby for updates.\n\nCrisis Management Team"
        
        return {
            "subject": subject,
            "body": body,
            "variables": variables
        }
    
    def _determine_notification_priority(
        self,
        crisis: Crisis,
        stakeholder: Stakeholder,
        immediate: bool = False
    ) -> NotificationPriority:
        """Determine notification priority"""
        if immediate or crisis.severity_level == SeverityLevel.CRITICAL:
            return NotificationPriority.CRITICAL
        elif crisis.severity_level == SeverityLevel.HIGH:
            return NotificationPriority.HIGH
        elif stakeholder.priority_level == NotificationPriority.HIGH:
            return NotificationPriority.HIGH
        else:
            return NotificationPriority.MEDIUM
    
    async def _send_notification_batch(
        self,
        batch: NotificationBatch,
        notifications: List[NotificationMessage]
    ) -> Dict[str, Any]:
        """Send batch of notifications"""
        batch.status = "sending"
        batch.sent_time = datetime.utcnow()
        
        results = {
            "success": 0,
            "failed": 0,
            "details": []
        }
        
        # Send notifications concurrently
        tasks = []
        for notification in notifications:
            task = asyncio.create_task(self._send_single_notification(notification))
            tasks.append(task)
        
        # Wait for all notifications to complete
        notification_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(notification_results):
            notification = notifications[i]
            if isinstance(result, Exception):
                notification.status = NotificationStatus.FAILED
                notification.failure_reason = str(result)
                results["failed"] += 1
            elif result.get("success"):
                notification.status = NotificationStatus.SENT
                notification.sent_time = datetime.utcnow()
                results["success"] += 1
            else:
                notification.status = NotificationStatus.FAILED
                notification.failure_reason = result.get("error", "Unknown error")
                results["failed"] += 1
            
            results["details"].append({
                "notification_id": notification.id,
                "stakeholder_id": notification.stakeholder_id,
                "status": notification.status.value,
                "channel": notification.channel.value
            })
        
        batch.status = "completed"
        batch.completion_time = datetime.utcnow()
        batch.success_count = results["success"]
        batch.failure_count = results["failed"]
        
        return results
    
    async def _send_single_notification(self, notification: NotificationMessage) -> Dict[str, Any]:
        """Send individual notification"""
        try:
            # Simulate sending notification based on channel
            if notification.channel == NotificationChannel.EMAIL:
                await self._send_email_notification(notification)
            elif notification.channel == NotificationChannel.SMS:
                await self._send_sms_notification(notification)
            elif notification.channel == NotificationChannel.PHONE:
                await self._send_phone_notification(notification)
            elif notification.channel == NotificationChannel.SLACK:
                await self._send_slack_notification(notification)
            elif notification.channel == NotificationChannel.TEAMS:
                await self._send_teams_notification(notification)
            elif notification.channel == NotificationChannel.PUSH:
                await self._send_push_notification(notification)
            elif notification.channel == NotificationChannel.PORTAL:
                await self._send_portal_notification(notification)
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_email_notification(self, notification: NotificationMessage):
        """Send email notification (simulated)"""
        await asyncio.sleep(0.1)  # Simulate network delay
        self.logger.info(f"Email sent to stakeholder {notification.stakeholder_id}")
    
    async def _send_sms_notification(self, notification: NotificationMessage):
        """Send SMS notification (simulated)"""
        await asyncio.sleep(0.05)  # Simulate network delay
        self.logger.info(f"SMS sent to stakeholder {notification.stakeholder_id}")
    
    async def _send_phone_notification(self, notification: NotificationMessage):
        """Send phone notification (simulated)"""
        await asyncio.sleep(0.2)  # Simulate call setup
        self.logger.info(f"Phone call initiated to stakeholder {notification.stakeholder_id}")
    
    async def _send_slack_notification(self, notification: NotificationMessage):
        """Send Slack notification (simulated)"""
        await asyncio.sleep(0.05)  # Simulate API call
        self.logger.info(f"Slack message sent to stakeholder {notification.stakeholder_id}")
    
    async def _send_teams_notification(self, notification: NotificationMessage):
        """Send Teams notification (simulated)"""
        await asyncio.sleep(0.05)  # Simulate API call
        self.logger.info(f"Teams message sent to stakeholder {notification.stakeholder_id}")
    
    async def _send_push_notification(self, notification: NotificationMessage):
        """Send push notification (simulated)"""
        await asyncio.sleep(0.02)  # Simulate push service
        self.logger.info(f"Push notification sent to stakeholder {notification.stakeholder_id}")
    
    async def _send_portal_notification(self, notification: NotificationMessage):
        """Send portal notification (simulated)"""
        await asyncio.sleep(0.01)  # Simulate database write
        self.logger.info(f"Portal notification created for stakeholder {notification.stakeholder_id}")
    
    def _get_or_create_communication_plan(self, crisis: Crisis) -> CommunicationPlan:
        """Get or create communication plan for crisis"""
        plan_key = f"{crisis.crisis_type.value}_{crisis.severity_level.value}"
        
        if plan_key not in self.communication_plans:
            self.communication_plans[plan_key] = CommunicationPlan(
                crisis_id=crisis.id,
                crisis_type=crisis.crisis_type.value,
                phases=[
                    {"name": "immediate", "duration_minutes": 15},
                    {"name": "update", "duration_minutes": 60},
                    {"name": "resolution", "duration_minutes": 240}
                ]
            )
        
        return self.communication_plans[plan_key]
    
    def _calculate_notification_metrics(
        self, 
        notifications: List[NotificationMessage]
    ) -> NotificationMetrics:
        """Calculate metrics for notification batch"""
        total_sent = len([n for n in notifications if n.status == NotificationStatus.SENT])
        total_failed = len([n for n in notifications if n.status == NotificationStatus.FAILED])
        
        metrics = NotificationMetrics(
            total_sent=total_sent,
            total_failed=total_failed,
            delivery_rate=total_sent / len(notifications) if notifications else 0
        )
        
        return metrics
    
    def _get_customer_impact_description(self, crisis: Crisis) -> str:
        """Get customer-focused impact description"""
        impact_descriptions = {
            CrisisType.SYSTEM_OUTAGE: "Service may be temporarily unavailable",
            CrisisType.SECURITY_BREACH: "Your data security is our top priority",
            CrisisType.PERFORMANCE_DEGRADATION: "You may experience slower response times"
        }
        return impact_descriptions.get(crisis.crisis_type, "We are working to resolve this issue")
    
    def _get_employee_action_required(self, crisis: Crisis, stakeholder: Stakeholder) -> str:
        """Get employee-specific action requirements"""
        if stakeholder.department == "IT":
            return "Please report to the crisis response center immediately"
        elif stakeholder.department == "Customer Service":
            return "Please follow the customer communication script provided"
        else:
            return "Continue normal operations unless otherwise directed"
    
    def _get_business_impact_summary(self, crisis: Crisis) -> str:
        """Get business impact summary for executives"""
        return f"Estimated business impact: {crisis.severity_level.value} level disruption to operations"
    
    # Stakeholder management methods
    def add_stakeholder(self, stakeholder: Stakeholder) -> bool:
        """Add stakeholder to the system"""
        try:
            self.stakeholders[stakeholder.id] = stakeholder
            self.logger.info(f"Added stakeholder: {stakeholder.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add stakeholder: {str(e)}")
            return False
    
    def update_stakeholder(self, stakeholder_id: str, updates: Dict[str, Any]) -> bool:
        """Update stakeholder information"""
        try:
            if stakeholder_id in self.stakeholders:
                stakeholder = self.stakeholders[stakeholder_id]
                for key, value in updates.items():
                    if hasattr(stakeholder, key):
                        setattr(stakeholder, key, value)
                stakeholder.updated_at = datetime.utcnow()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to update stakeholder: {str(e)}")
            return False
    
    def add_notification_template(self, template: NotificationTemplate) -> bool:
        """Add notification template"""
        try:
            self.templates[template.id] = template
            self.logger.info(f"Added notification template: {template.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add template: {str(e)}")
            return False