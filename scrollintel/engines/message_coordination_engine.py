"""
Message Coordination Engine

Handles consistent messaging across all communication channels during crisis situations.
Implements message approval workflow, version control, and effectiveness tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict, dataclass, field
from enum import Enum
from uuid import uuid4

from ..models.crisis_communication_models import (
    NotificationChannel, NotificationPriority, StakeholderType
)


class MessageStatus(Enum):
    """Message status states"""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ApprovalStatus(Enum):
    """Approval workflow status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


@dataclass
class MessageVersion:
    """Message version for version control"""
    id: str = field(default_factory=lambda: str(uuid4()))
    version_number: str = "1.0"
    content: str = ""
    subject: str = ""
    author: str = ""
    changes_summary: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_current: bool = False


@dataclass
class ApprovalWorkflowStep:
    """Approval workflow step"""
    id: str = field(default_factory=lambda: str(uuid4()))
    step_order: int = 1
    approver_role: str = ""
    approver_id: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    comments: str = ""
    approved_at: Optional[datetime] = None
    escalation_timeout_minutes: int = 30


@dataclass
class CoordinatedMessage:
    """Coordinated message across channels"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    message_type: str = ""
    priority: NotificationPriority = NotificationPriority.MEDIUM
    target_channels: List[NotificationChannel] = field(default_factory=list)
    target_stakeholders: List[StakeholderType] = field(default_factory=list)
    
    # Content
    master_content: str = ""
    master_subject: str = ""
    channel_adaptations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Version control
    versions: List[MessageVersion] = field(default_factory=list)
    current_version: str = "1.0"
    
    # Approval workflow
    approval_workflow: List[ApprovalWorkflowStep] = field(default_factory=list)
    requires_approval: bool = True
    status: MessageStatus = MessageStatus.DRAFT
    
    # Scheduling
    scheduled_time: Optional[datetime] = None
    published_time: Optional[datetime] = None
    
    # Tracking
    delivery_tracking: Dict[str, Any] = field(default_factory=dict)
    effectiveness_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MessageTemplate:
    """Template for coordinated messages"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    message_type: str = ""
    crisis_type: str = ""
    
    # Template content
    master_template: str = ""
    subject_template: str = ""
    channel_templates: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Configuration
    default_channels: List[NotificationChannel] = field(default_factory=list)
    default_stakeholders: List[StakeholderType] = field(default_factory=list)
    requires_approval: bool = True
    approval_workflow_template: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class MessageEffectivenessMetrics:
    """Metrics for message effectiveness"""
    message_id: str = ""
    total_sent: int = 0
    total_delivered: int = 0
    total_read: int = 0
    total_responded: int = 0
    
    # Rates
    delivery_rate: float = 0.0
    read_rate: float = 0.0
    response_rate: float = 0.0
    
    # Channel performance
    channel_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Stakeholder engagement
    stakeholder_engagement: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Timing metrics
    average_delivery_time_seconds: float = 0.0
    average_read_time_seconds: float = 0.0
    
    # Effectiveness score
    overall_effectiveness_score: float = 0.0
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.utcnow)


class MessageCoordinationEngine:
    """Engine for coordinating messages across all communication channels"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.messages: Dict[str, CoordinatedMessage] = {}
        self.templates: Dict[str, MessageTemplate] = {}
        self.approval_workflows: Dict[str, List[ApprovalWorkflowStep]] = {}
        self.effectiveness_metrics: Dict[str, MessageEffectivenessMetrics] = {}
        self.channel_adapters = self._initialize_channel_adapters()
        
    def _initialize_channel_adapters(self) -> Dict[str, Any]:
        """Initialize channel-specific message adapters"""
        return {
            "email": {
                "max_subject_length": 100,
                "max_body_length": 10000,
                "supports_html": True,
                "supports_attachments": True
            },
            "sms": {
                "max_subject_length": 0,
                "max_body_length": 160,
                "supports_html": False,
                "supports_attachments": False
            },
            "slack": {
                "max_subject_length": 0,
                "max_body_length": 4000,
                "supports_html": False,
                "supports_attachments": True,
                "supports_markdown": True
            },
            "teams": {
                "max_subject_length": 0,
                "max_body_length": 4000,
                "supports_html": True,
                "supports_attachments": True
            },
            "portal": {
                "max_subject_length": 200,
                "max_body_length": 50000,
                "supports_html": True,
                "supports_attachments": True
            },
            "media_release": {
                "max_subject_length": 150,
                "max_body_length": 5000,
                "supports_html": False,
                "supports_attachments": False,
                "requires_approval": True
            }
        }
    
    async def create_coordinated_message(
        self,
        crisis_id: str,
        message_type: str,
        master_content: str,
        master_subject: str,
        target_channels: List[NotificationChannel],
        target_stakeholders: List[StakeholderType],
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        requires_approval: bool = True,
        created_by: str = "system"
    ) -> CoordinatedMessage:
        """
        Create a new coordinated message
        
        Args:
            crisis_id: ID of the crisis
            message_type: Type of message (e.g., "initial_alert", "update", "resolution")
            master_content: Master message content
            master_subject: Master message subject
            target_channels: Channels to send message to
            target_stakeholders: Stakeholder types to target
            priority: Message priority
            requires_approval: Whether message requires approval
            created_by: User who created the message
            
        Returns:
            Created coordinated message
        """
        try:
            # Create coordinated message
            message = CoordinatedMessage(
                crisis_id=crisis_id,
                message_type=message_type,
                priority=priority,
                target_channels=target_channels,
                target_stakeholders=target_stakeholders,
                master_content=master_content,
                master_subject=master_subject,
                requires_approval=requires_approval,
                created_by=created_by
            )
            
            # Create initial version
            initial_version = MessageVersion(
                version_number="1.0",
                content=master_content,
                subject=master_subject,
                author=created_by,
                changes_summary="Initial version",
                is_current=True
            )
            message.versions.append(initial_version)
            
            # Adapt content for each channel
            message.channel_adaptations = await self._adapt_content_for_channels(
                master_content, master_subject, target_channels
            )
            
            # Set up approval workflow if required
            if requires_approval:
                message.approval_workflow = self._create_approval_workflow(
                    message_type, priority, target_channels
                )
                message.status = MessageStatus.PENDING_APPROVAL
            else:
                message.status = MessageStatus.APPROVED
            
            # Store message
            self.messages[message.id] = message
            
            self.logger.info(f"Created coordinated message {message.id} for crisis {crisis_id}")
            
            return message
            
        except Exception as e:
            self.logger.error(f"Failed to create coordinated message: {str(e)}")
            raise
    
    async def _adapt_content_for_channels(
        self,
        master_content: str,
        master_subject: str,
        channels: List[NotificationChannel]
    ) -> Dict[str, Dict[str, str]]:
        """Adapt message content for specific channels"""
        adaptations = {}
        
        for channel in channels:
            channel_key = channel.value
            adapter_config = self.channel_adapters.get(channel_key, {})
            
            # Adapt subject
            adapted_subject = master_subject
            max_subject_length = adapter_config.get("max_subject_length", 1000)
            if max_subject_length > 0 and len(adapted_subject) > max_subject_length:
                adapted_subject = adapted_subject[:max_subject_length-3] + "..."
            
            # Adapt content
            adapted_content = master_content
            max_body_length = adapter_config.get("max_body_length", 10000)
            if len(adapted_content) > max_body_length:
                adapted_content = adapted_content[:max_body_length-3] + "..."
            
            # Channel-specific formatting
            if channel == NotificationChannel.SMS:
                # SMS: Remove formatting, keep essential info only
                adapted_content = self._format_for_sms(adapted_content)
                adapted_subject = ""  # SMS doesn't use subjects
            elif channel == NotificationChannel.SLACK:
                # Slack: Convert to markdown format
                adapted_content = self._format_for_slack(adapted_content)
                adapted_subject = ""  # Slack doesn't use subjects
            elif channel == NotificationChannel.EMAIL:
                # Email: Keep HTML formatting if supported
                adapted_content = self._format_for_email(adapted_content)
            elif channel == NotificationChannel.MEDIA_RELEASE:
                # Media: Professional press release format
                adapted_content = self._format_for_media(adapted_content)
            
            adaptations[channel_key] = {
                "subject": adapted_subject,
                "content": adapted_content,
                "metadata": {
                    "original_length": len(master_content),
                    "adapted_length": len(adapted_content),
                    "truncated": len(adapted_content) < len(master_content)
                }
            }
        
        return adaptations
    
    def _format_for_sms(self, content: str) -> str:
        """Format content for SMS"""
        # Remove HTML tags and excessive whitespace
        import re
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Keep only essential information
        lines = content.split('\n')
        essential_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('---') and not line.startswith('==='):
                essential_lines.append(line)
        
        return ' '.join(essential_lines)
    
    def _format_for_slack(self, content: str) -> str:
        """Format content for Slack with markdown"""
        # Convert basic HTML to markdown
        content = content.replace('<b>', '*').replace('</b>', '*')
        content = content.replace('<strong>', '*').replace('</strong>', '*')
        content = content.replace('<i>', '_').replace('</i>', '_')
        content = content.replace('<em>', '_').replace('</em>', '_')
        
        # Add Slack-specific formatting
        if "URGENT" in content.upper() or "CRITICAL" in content.upper():
            content = f"🚨 *URGENT* 🚨\n\n{content}"
        
        return content
    
    def _format_for_email(self, content: str) -> str:
        """Format content for email"""
        # Keep HTML formatting for email
        if not content.startswith('<'):
            # Convert plain text to basic HTML
            content = content.replace('\n\n', '</p><p>')
            content = content.replace('\n', '<br>')
            content = f'<p>{content}</p>'
        
        return content
    
    def _format_for_media(self, content: str) -> str:
        """Format content for media release"""
        # Professional press release format
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove internal jargon and technical details
                if not any(word in line.lower() for word in ['internal', 'crisis team', 'hotline']):
                    formatted_lines.append(line)
        
        # Add standard media release footer
        formatted_lines.append("")
        formatted_lines.append("For media inquiries, please contact:")
        formatted_lines.append("Press Relations Team")
        formatted_lines.append("press@company.com")
        
        return '\n'.join(formatted_lines)
    
    def _create_approval_workflow(
        self,
        message_type: str,
        priority: NotificationPriority,
        channels: List[NotificationChannel]
    ) -> List[ApprovalWorkflowStep]:
        """Create approval workflow based on message characteristics"""
        workflow_steps = []
        
        # Standard approval workflow
        if priority == NotificationPriority.CRITICAL:
            # Critical messages: CEO approval required
            workflow_steps.append(ApprovalWorkflowStep(
                step_order=1,
                approver_role="CEO",
                escalation_timeout_minutes=15
            ))
        elif priority == NotificationPriority.HIGH:
            # High priority: Department head approval
            workflow_steps.append(ApprovalWorkflowStep(
                step_order=1,
                approver_role="Department_Head",
                escalation_timeout_minutes=30
            ))
        else:
            # Medium/Low priority: Manager approval
            workflow_steps.append(ApprovalWorkflowStep(
                step_order=1,
                approver_role="Manager",
                escalation_timeout_minutes=60
            ))
        
        # Media releases require additional PR approval
        if NotificationChannel.MEDIA_RELEASE in channels:
            workflow_steps.append(ApprovalWorkflowStep(
                step_order=len(workflow_steps) + 1,
                approver_role="PR_Manager",
                escalation_timeout_minutes=30
            ))
        
        # Customer communications require customer success approval
        customer_channels = [NotificationChannel.EMAIL, NotificationChannel.PORTAL]
        if any(channel in channels for channel in customer_channels):
            workflow_steps.append(ApprovalWorkflowStep(
                step_order=len(workflow_steps) + 1,
                approver_role="Customer_Success_Manager",
                escalation_timeout_minutes=45
            ))
        
        return workflow_steps
    
    async def submit_for_approval(self, message_id: str) -> bool:
        """Submit message for approval"""
        try:
            if message_id not in self.messages:
                return False
            
            message = self.messages[message_id]
            
            if not message.requires_approval:
                message.status = MessageStatus.APPROVED
                return True
            
            message.status = MessageStatus.PENDING_APPROVAL
            
            # Start approval workflow
            if message.approval_workflow:
                first_step = message.approval_workflow[0]
                # In a real system, this would send notification to approver
                self.logger.info(f"Approval request sent to {first_step.approver_role} for message {message_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit message for approval: {str(e)}")
            return False
    
    async def approve_message(
        self,
        message_id: str,
        approver_id: str,
        approver_role: str,
        comments: str = ""
    ) -> bool:
        """Approve message at current workflow step"""
        try:
            if message_id not in self.messages:
                return False
            
            message = self.messages[message_id]
            
            # Find current approval step
            current_step = None
            for step in message.approval_workflow:
                if step.status == ApprovalStatus.PENDING and step.approver_role == approver_role:
                    current_step = step
                    break
            
            if not current_step:
                return False
            
            # Approve current step
            current_step.status = ApprovalStatus.APPROVED
            current_step.approver_id = approver_id
            current_step.comments = comments
            current_step.approved_at = datetime.utcnow()
            
            # Check if all steps are approved
            all_approved = all(
                step.status == ApprovalStatus.APPROVED 
                for step in message.approval_workflow
            )
            
            if all_approved:
                message.status = MessageStatus.APPROVED
                self.logger.info(f"Message {message_id} fully approved")
            else:
                # Move to next step
                next_step = next(
                    (step for step in message.approval_workflow 
                     if step.status == ApprovalStatus.PENDING), 
                    None
                )
                if next_step:
                    self.logger.info(f"Message {message_id} approved by {approver_role}, moving to {next_step.approver_role}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to approve message: {str(e)}")
            return False
    
    async def reject_message(
        self,
        message_id: str,
        approver_id: str,
        approver_role: str,
        rejection_reason: str
    ) -> bool:
        """Reject message"""
        try:
            if message_id not in self.messages:
                return False
            
            message = self.messages[message_id]
            
            # Find current approval step
            current_step = None
            for step in message.approval_workflow:
                if step.status == ApprovalStatus.PENDING and step.approver_role == approver_role:
                    current_step = step
                    break
            
            if not current_step:
                return False
            
            # Reject message
            current_step.status = ApprovalStatus.REJECTED
            current_step.approver_id = approver_id
            current_step.comments = rejection_reason
            current_step.approved_at = datetime.utcnow()
            
            message.status = MessageStatus.REJECTED
            
            self.logger.info(f"Message {message_id} rejected by {approver_role}: {rejection_reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reject message: {str(e)}")
            return False
    
    async def publish_message(self, message_id: str) -> Dict[str, Any]:
        """Publish approved message to all channels"""
        try:
            if message_id not in self.messages:
                return {"success": False, "error": "Message not found"}
            
            message = self.messages[message_id]
            
            if message.status != MessageStatus.APPROVED:
                return {"success": False, "error": "Message not approved"}
            
            # Publish to all target channels
            publication_results = {}
            
            for channel in message.target_channels:
                channel_key = channel.value
                adaptation = message.channel_adaptations.get(channel_key, {})
                
                result = await self._publish_to_channel(
                    channel, 
                    adaptation.get("subject", message.master_subject),
                    adaptation.get("content", message.master_content),
                    message
                )
                
                publication_results[channel_key] = result
            
            # Update message status
            message.status = MessageStatus.PUBLISHED
            message.published_time = datetime.utcnow()
            
            # Initialize delivery tracking
            message.delivery_tracking = {
                "published_at": message.published_time.isoformat(),
                "channels": publication_results,
                "total_sent": sum(1 for r in publication_results.values() if r.get("success")),
                "total_failed": sum(1 for r in publication_results.values() if not r.get("success"))
            }
            
            self.logger.info(f"Published message {message_id} to {len(message.target_channels)} channels")
            
            return {
                "success": True,
                "message_id": message_id,
                "published_at": message.published_time.isoformat(),
                "channels": publication_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _publish_to_channel(
        self,
        channel: NotificationChannel,
        subject: str,
        content: str,
        message: CoordinatedMessage
    ) -> Dict[str, Any]:
        """Publish message to specific channel"""
        try:
            # Simulate channel-specific publishing
            await asyncio.sleep(0.1)  # Simulate network delay
            
            self.logger.info(f"Published to {channel.value}: {subject[:50]}...")
            
            return {
                "success": True,
                "channel": channel.value,
                "sent_at": datetime.utcnow().isoformat(),
                "message_length": len(content)
            }
            
        except Exception as e:
            return {
                "success": False,
                "channel": channel.value,
                "error": str(e)
            }
    
    async def update_message_version(
        self,
        message_id: str,
        new_content: str,
        new_subject: str,
        author: str,
        changes_summary: str
    ) -> bool:
        """Create new version of message"""
        try:
            if message_id not in self.messages:
                return False
            
            message = self.messages[message_id]
            
            # Mark current version as not current
            for version in message.versions:
                version.is_current = False
            
            # Create new version
            version_number = f"{len(message.versions) + 1}.0"
            new_version = MessageVersion(
                version_number=version_number,
                content=new_content,
                subject=new_subject,
                author=author,
                changes_summary=changes_summary,
                is_current=True
            )
            
            message.versions.append(new_version)
            message.current_version = version_number
            message.master_content = new_content
            message.master_subject = new_subject
            message.updated_at = datetime.utcnow()
            
            # Re-adapt content for channels
            message.channel_adaptations = await self._adapt_content_for_channels(
                new_content, new_subject, message.target_channels
            )
            
            # Reset approval if message was already approved
            if message.status == MessageStatus.APPROVED:
                message.status = MessageStatus.PENDING_APPROVAL
                # Reset approval workflow
                for step in message.approval_workflow:
                    step.status = ApprovalStatus.PENDING
                    step.approved_at = None
                    step.comments = ""
            
            self.logger.info(f"Created version {version_number} for message {message_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update message version: {str(e)}")
            return False
    
    async def track_message_effectiveness(self, message_id: str) -> MessageEffectivenessMetrics:
        """Track and calculate message effectiveness metrics"""
        try:
            if message_id not in self.messages:
                return MessageEffectivenessMetrics(message_id=message_id)
            
            message = self.messages[message_id]
            
            # Simulate metrics collection (in real system, this would query actual data)
            metrics = MessageEffectivenessMetrics(
                message_id=message_id,
                total_sent=len(message.target_channels) * 10,  # Simulated
                total_delivered=len(message.target_channels) * 9,  # Simulated
                total_read=len(message.target_channels) * 7,  # Simulated
                total_responded=len(message.target_channels) * 3  # Simulated
            )
            
            # Calculate rates
            if metrics.total_sent > 0:
                metrics.delivery_rate = metrics.total_delivered / metrics.total_sent
                metrics.read_rate = metrics.total_read / metrics.total_sent
                metrics.response_rate = metrics.total_responded / metrics.total_sent
            
            # Channel performance (simulated)
            for channel in message.target_channels:
                channel_key = channel.value
                metrics.channel_performance[channel_key] = {
                    "delivery_rate": 0.9 + (hash(channel_key) % 10) / 100,
                    "read_rate": 0.7 + (hash(channel_key) % 20) / 100,
                    "response_rate": 0.3 + (hash(channel_key) % 15) / 100
                }
            
            # Calculate overall effectiveness score
            metrics.overall_effectiveness_score = (
                metrics.delivery_rate * 0.3 +
                metrics.read_rate * 0.4 +
                metrics.response_rate * 0.3
            )
            
            # Store metrics
            self.effectiveness_metrics[message_id] = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to track message effectiveness: {str(e)}")
            return MessageEffectivenessMetrics(message_id=message_id)
    
    def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of message"""
        if message_id not in self.messages:
            return None
        
        message = self.messages[message_id]
        
        return {
            "message_id": message_id,
            "status": message.status.value,
            "current_version": message.current_version,
            "created_at": message.created_at.isoformat(),
            "updated_at": message.updated_at.isoformat(),
            "published_time": message.published_time.isoformat() if message.published_time else None,
            "approval_workflow": [
                {
                    "step_order": step.step_order,
                    "approver_role": step.approver_role,
                    "status": step.status.value,
                    "approved_at": step.approved_at.isoformat() if step.approved_at else None,
                    "comments": step.comments
                }
                for step in message.approval_workflow
            ],
            "target_channels": [channel.value for channel in message.target_channels],
            "delivery_tracking": message.delivery_tracking
        }
    
    def add_message_template(self, template: MessageTemplate) -> bool:
        """Add message template"""
        try:
            self.templates[template.id] = template
            self.logger.info(f"Added message template: {template.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add message template: {str(e)}")
            return False
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get overall message coordination metrics"""
        total_messages = len(self.messages)
        
        if total_messages == 0:
            return {
                "total_messages": 0,
                "status_distribution": {},
                "average_approval_time": 0,
                "channel_usage": {},
                "effectiveness_summary": {}
            }
        
        # Status distribution
        status_counts = {}
        for message in self.messages.values():
            status = message.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Channel usage
        channel_counts = {}
        for message in self.messages.values():
            for channel in message.target_channels:
                channel_key = channel.value
                channel_counts[channel_key] = channel_counts.get(channel_key, 0) + 1
        
        # Effectiveness summary
        effectiveness_scores = [
            metrics.overall_effectiveness_score 
            for metrics in self.effectiveness_metrics.values()
            if metrics.overall_effectiveness_score > 0
        ]
        
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0
        
        return {
            "total_messages": total_messages,
            "status_distribution": status_counts,
            "average_approval_time": 25.5,  # Simulated
            "channel_usage": channel_counts,
            "effectiveness_summary": {
                "average_effectiveness_score": avg_effectiveness,
                "total_tracked_messages": len(self.effectiveness_metrics),
                "high_performing_messages": len([s for s in effectiveness_scores if s > 0.8])
            }
        }