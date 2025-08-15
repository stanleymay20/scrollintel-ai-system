"""
Tests for Message Coordination Engine

Test suite for crisis message coordination system including consistent messaging,
approval workflows, version control, and effectiveness tracking.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.message_coordination_engine import (
    MessageCoordinationEngine, MessageStatus, ApprovalStatus
)
from scrollintel.models.crisis_communication_models import (
    NotificationChannel, NotificationPriority, StakeholderType
)


class TestMessageCoordinationEngine:
    """Test cases for MessageCoordinationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create message coordination engine instance"""
        return MessageCoordinationEngine()
    
    @pytest.fixture
    def sample_message_data(self):
        """Create sample message data"""
        return {
            "crisis_id": "crisis_001",
            "message_type": "initial_alert",
            "master_content": "We are experiencing a system outage. Our team is working to resolve this issue.",
            "master_subject": "System Outage Alert",
            "target_channels": [NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.SLACK],
            "target_stakeholders": [StakeholderType.EMPLOYEE, StakeholderType.CUSTOMER],
            "priority": NotificationPriority.HIGH,
            "requires_approval": True,
            "created_by": "test_user"
        }
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert isinstance(engine.messages, dict)
        assert isinstance(engine.templates, dict)
        assert isinstance(engine.approval_workflows, dict)
        assert isinstance(engine.effectiveness_metrics, dict)
        assert isinstance(engine.channel_adapters, dict)
        
        # Check channel adapters are initialized
        assert "email" in engine.channel_adapters
        assert "sms" in engine.channel_adapters
        assert "slack" in engine.channel_adapters
    
    @pytest.mark.asyncio
    async def test_create_coordinated_message(self, engine, sample_message_data):
        """Test creating coordinated message"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        assert message is not None
        assert message.crisis_id == sample_message_data["crisis_id"]
        assert message.message_type == sample_message_data["message_type"]
        assert message.master_content == sample_message_data["master_content"]
        assert message.master_subject == sample_message_data["master_subject"]
        assert message.priority == sample_message_data["priority"]
        assert message.requires_approval == sample_message_data["requires_approval"]
        assert message.created_by == sample_message_data["created_by"]
        
        # Check message is stored
        assert message.id in engine.messages
        
        # Check initial version is created
        assert len(message.versions) == 1
        assert message.versions[0].is_current is True
        assert message.current_version == "1.0"
        
        # Check approval workflow is created
        assert len(message.approval_workflow) > 0
        assert message.status == MessageStatus.PENDING_APPROVAL
        
        # Check channel adaptations are created
        assert len(message.channel_adaptations) == len(sample_message_data["target_channels"])
    
    @pytest.mark.asyncio
    async def test_content_adaptation_for_channels(self, engine):
        """Test content adaptation for different channels"""
        master_content = "This is a test message with <b>bold text</b> and multiple lines.\n\nSecond paragraph here."
        master_subject = "Test Subject"
        channels = [NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.SLACK]
        
        adaptations = await engine._adapt_content_for_channels(
            master_content, master_subject, channels
        )
        
        assert len(adaptations) == 3
        
        # Check email adaptation
        email_adaptation = adaptations["email"]
        assert email_adaptation["subject"] == master_subject
        assert "<p>" in email_adaptation["content"]  # HTML formatting
        
        # Check SMS adaptation
        sms_adaptation = adaptations["sms"]
        assert sms_adaptation["subject"] == ""  # SMS doesn't use subjects
        assert len(sms_adaptation["content"]) <= 160  # SMS length limit
        assert "<b>" not in sms_adaptation["content"]  # HTML tags removed
        
        # Check Slack adaptation
        slack_adaptation = adaptations["slack"]
        assert slack_adaptation["subject"] == ""  # Slack doesn't use subjects
        assert "*" in slack_adaptation["content"]  # Markdown formatting
    
    def test_sms_formatting(self, engine):
        """Test SMS-specific formatting"""
        content = "<b>URGENT:</b> System outage detected.\n\nPlease standby for updates.\n\n---\nTechnical Team"
        
        formatted = engine._format_for_sms(content)
        
        assert "<b>" not in formatted
        assert "</b>" not in formatted
        assert "---" not in formatted
        assert len(formatted) <= 160
        assert "URGENT" in formatted
    
    def test_slack_formatting(self, engine):
        """Test Slack-specific formatting"""
        content = "<b>URGENT:</b> System outage detected.\n\nPlease standby for updates."
        
        formatted = engine._format_for_slack(content)
        
        assert "*URGENT*" in formatted
        assert "🚨" in formatted  # Urgent indicator added
        assert "<b>" not in formatted
    
    def test_media_formatting(self, engine):
        """Test media release formatting"""
        content = "System outage detected. Internal crisis team activated. Please contact hotline for updates."
        
        formatted = engine._format_for_media(content)
        
        assert "crisis team" not in formatted  # Internal jargon removed
        assert "hotline" not in formatted  # Internal references removed
        assert "press@company.com" in formatted  # Media contact added
    
    def test_create_approval_workflow(self, engine):
        """Test approval workflow creation"""
        # Test critical priority workflow
        workflow = engine._create_approval_workflow(
            "initial_alert", 
            NotificationPriority.CRITICAL, 
            [NotificationChannel.EMAIL]
        )
        
        assert len(workflow) >= 1
        assert any(step.approver_role == "CEO" for step in workflow)
        
        # Test media release workflow
        workflow = engine._create_approval_workflow(
            "media_statement", 
            NotificationPriority.HIGH, 
            [NotificationChannel.MEDIA_RELEASE]
        )
        
        assert any(step.approver_role == "PR_Manager" for step in workflow)
    
    @pytest.mark.asyncio
    async def test_submit_for_approval(self, engine, sample_message_data):
        """Test submitting message for approval"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        # Message should already be pending approval
        assert message.status == MessageStatus.PENDING_APPROVAL
        
        # Test submitting again
        success = await engine.submit_for_approval(message.id)
        assert success is True
        
        # Test with non-existent message
        success = await engine.submit_for_approval("non_existent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_approve_message(self, engine, sample_message_data):
        """Test message approval"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        # Get first approval step
        first_step = message.approval_workflow[0]
        
        # Approve message
        success = await engine.approve_message(
            message.id,
            "approver_001",
            first_step.approver_role,
            "Approved for publication"
        )
        
        assert success is True
        assert first_step.status == ApprovalStatus.APPROVED
        assert first_step.approver_id == "approver_001"
        assert first_step.comments == "Approved for publication"
        
        # If only one step, message should be approved
        if len(message.approval_workflow) == 1:
            assert message.status == MessageStatus.APPROVED
    
    @pytest.mark.asyncio
    async def test_reject_message(self, engine, sample_message_data):
        """Test message rejection"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        # Get first approval step
        first_step = message.approval_workflow[0]
        
        # Reject message
        success = await engine.reject_message(
            message.id,
            "approver_001",
            first_step.approver_role,
            "Content needs revision"
        )
        
        assert success is True
        assert first_step.status == ApprovalStatus.REJECTED
        assert message.status == MessageStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_publish_message(self, engine, sample_message_data):
        """Test message publication"""
        # Create and approve message
        message = await engine.create_coordinated_message(**sample_message_data)
        message.status = MessageStatus.APPROVED  # Simulate approval
        
        # Publish message
        result = await engine.publish_message(message.id)
        
        assert result["success"] is True
        assert message.status == MessageStatus.PUBLISHED
        assert message.published_time is not None
        assert "channels" in result
        
        # Check delivery tracking
        assert "delivery_tracking" in message.__dict__
        assert message.delivery_tracking["total_sent"] > 0
    
    @pytest.mark.asyncio
    async def test_publish_unapproved_message(self, engine, sample_message_data):
        """Test publishing unapproved message fails"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        # Try to publish without approval
        result = await engine.publish_message(message.id)
        
        assert result["success"] is False
        assert "not approved" in result["error"]
    
    @pytest.mark.asyncio
    async def test_update_message_version(self, engine, sample_message_data):
        """Test updating message version"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        # Update message
        new_content = "Updated message content with additional information."
        new_subject = "Updated Subject"
        
        success = await engine.update_message_version(
            message.id,
            new_content,
            new_subject,
            "editor_user",
            "Added more details"
        )
        
        assert success is True
        assert len(message.versions) == 2
        assert message.current_version == "2.0"
        assert message.master_content == new_content
        assert message.master_subject == new_subject
        
        # Check current version
        current_version = next(v for v in message.versions if v.is_current)
        assert current_version.content == new_content
        assert current_version.author == "editor_user"
        assert current_version.changes_summary == "Added more details"
        
        # Check approval is reset
        if message.requires_approval:
            assert message.status == MessageStatus.PENDING_APPROVAL
    
    @pytest.mark.asyncio
    async def test_track_message_effectiveness(self, engine, sample_message_data):
        """Test message effectiveness tracking"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        metrics = await engine.track_message_effectiveness(message.id)
        
        assert metrics.message_id == message.id
        assert metrics.total_sent > 0
        assert metrics.delivery_rate >= 0
        assert metrics.read_rate >= 0
        assert metrics.response_rate >= 0
        assert metrics.overall_effectiveness_score >= 0
        
        # Check channel performance
        for channel in message.target_channels:
            assert channel.value in metrics.channel_performance
        
        # Check metrics are stored
        assert message.id in engine.effectiveness_metrics
    
    def test_get_message_status(self, engine):
        """Test getting message status"""
        # Test non-existent message
        status = engine.get_message_status("non_existent")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_message_status_existing(self, engine, sample_message_data):
        """Test getting status of existing message"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        status = engine.get_message_status(message.id)
        
        assert status is not None
        assert status["message_id"] == message.id
        assert status["status"] == message.status.value
        assert status["current_version"] == message.current_version
        assert "approval_workflow" in status
        assert "target_channels" in status
    
    def test_get_coordination_metrics(self, engine):
        """Test getting coordination metrics"""
        metrics = engine.get_coordination_metrics()
        
        assert "total_messages" in metrics
        assert "status_distribution" in metrics
        assert "channel_usage" in metrics
        assert "effectiveness_summary" in metrics
        
        # With no messages, total should be 0
        assert metrics["total_messages"] == 0
    
    @pytest.mark.asyncio
    async def test_coordination_metrics_with_messages(self, engine, sample_message_data):
        """Test coordination metrics with messages"""
        # Create a few messages
        await engine.create_coordinated_message(**sample_message_data)
        
        sample_message_data["message_type"] = "update"
        await engine.create_coordinated_message(**sample_message_data)
        
        metrics = engine.get_coordination_metrics()
        
        assert metrics["total_messages"] == 2
        assert "pending_approval" in metrics["status_distribution"]
        assert "email" in metrics["channel_usage"]
    
    @pytest.mark.asyncio
    async def test_publish_to_channel(self, engine, sample_message_data):
        """Test publishing to specific channel"""
        message = await engine.create_coordinated_message(**sample_message_data)
        
        result = await engine._publish_to_channel(
            NotificationChannel.EMAIL,
            "Test Subject",
            "Test Content",
            message
        )
        
        assert result["success"] is True
        assert result["channel"] == "email"
        assert "sent_at" in result
    
    @pytest.mark.asyncio
    async def test_channel_adaptation_length_limits(self, engine):
        """Test channel adaptation respects length limits"""
        # Create very long content
        long_content = "A" * 1000
        long_subject = "B" * 200
        
        adaptations = await engine._adapt_content_for_channels(
            long_content, long_subject, [NotificationChannel.SMS]
        )
        
        sms_adaptation = adaptations["sms"]
        assert len(sms_adaptation["content"]) <= 160
        assert sms_adaptation["metadata"]["truncated"] is True
    
    @pytest.mark.asyncio
    async def test_approval_workflow_multiple_steps(self, engine):
        """Test approval workflow with multiple steps"""
        # Create message requiring media release (multiple approval steps)
        message_data = {
            "crisis_id": "crisis_001",
            "message_type": "media_statement",
            "master_content": "Official statement about the incident.",
            "master_subject": "Official Statement",
            "target_channels": [NotificationChannel.MEDIA_RELEASE, NotificationChannel.EMAIL],
            "target_stakeholders": [StakeholderType.MEDIA, StakeholderType.CUSTOMER],
            "priority": NotificationPriority.HIGH,
            "requires_approval": True,
            "created_by": "pr_team"
        }
        
        message = await engine.create_coordinated_message(**message_data)
        
        # Should have multiple approval steps
        assert len(message.approval_workflow) > 1
        
        # Approve first step
        first_step = message.approval_workflow[0]
        await engine.approve_message(message.id, "approver1", first_step.approver_role)
        
        # Message should still be pending (not fully approved)
        assert message.status == MessageStatus.PENDING_APPROVAL
        
        # Approve remaining steps
        for step in message.approval_workflow[1:]:
            if step.status == ApprovalStatus.PENDING:
                await engine.approve_message(message.id, "approver2", step.approver_role)
        
        # Now message should be fully approved
        assert message.status == MessageStatus.APPROVED
    
    @pytest.mark.asyncio
    async def test_message_without_approval(self, engine, sample_message_data):
        """Test creating message that doesn't require approval"""
        sample_message_data["requires_approval"] = False
        
        message = await engine.create_coordinated_message(**sample_message_data)
        
        assert message.requires_approval is False
        assert message.status == MessageStatus.APPROVED
        assert len(message.approval_workflow) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_message_id(self, engine):
        """Test error handling with invalid message ID"""
        # Test various operations with invalid message ID
        success = await engine.submit_for_approval("invalid_id")
        assert success is False
        
        success = await engine.approve_message("invalid_id", "user", "role")
        assert success is False
        
        success = await engine.reject_message("invalid_id", "user", "role", "reason")
        assert success is False
        
        result = await engine.publish_message("invalid_id")
        assert result["success"] is False
        
        success = await engine.update_message_version("invalid_id", "content", "subject", "user", "summary")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_concurrent_message_operations(self, engine, sample_message_data):
        """Test concurrent message operations"""
        # Create multiple messages concurrently
        tasks = []
        for i in range(5):
            message_data = sample_message_data.copy()
            message_data["crisis_id"] = f"crisis_{i}"
            task = asyncio.create_task(engine.create_coordinated_message(**message_data))
            tasks.append(task)
        
        messages = await asyncio.gather(*tasks)
        
        assert len(messages) == 5
        assert len(engine.messages) == 5
        
        # All messages should have unique IDs
        message_ids = [m.id for m in messages]
        assert len(set(message_ids)) == 5