"""
Tests for Stakeholder Notification Engine

Test suite for crisis stakeholder notification system including immediate notifications,
stakeholder prioritization, message customization, and delivery tracking.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.stakeholder_notification_engine import StakeholderNotificationEngine
from scrollintel.models.crisis_communication_models import (
    Stakeholder, StakeholderType, NotificationTemplate, NotificationMessage,
    NotificationPriority, NotificationChannel, NotificationStatus
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisType, SeverityLevel, CrisisStatus


class TestStakeholderNotificationEngine:
    """Test cases for StakeholderNotificationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create notification engine instance"""
        return StakeholderNotificationEngine()
    
    @pytest.fixture
    def sample_stakeholders(self):
        """Create sample stakeholders"""
        return [
            Stakeholder(
                id="stakeholder_1",
                name="John CEO",
                stakeholder_type=StakeholderType.EXECUTIVE,
                contact_info={"email": "ceo@company.com", "phone": "+1234567890"},
                preferred_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
                priority_level=NotificationPriority.CRITICAL,
                role="CEO",
                department="Executive",
                influence_level=10,
                crisis_relevance={"security_breach": 10, "system_outage": 8}
            ),
            Stakeholder(
                id="stakeholder_2",
                name="Jane CTO",
                stakeholder_type=StakeholderType.EXECUTIVE,
                contact_info={"email": "cto@company.com", "slack": "@jane.cto"},
                preferred_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
                priority_level=NotificationPriority.HIGH,
                role="CTO",
                department="Technology",
                influence_level=9,
                crisis_relevance={"security_breach": 10, "system_outage": 10}
            ),
            Stakeholder(
                id="stakeholder_3",
                name="Bob Employee",
                stakeholder_type=StakeholderType.EMPLOYEE,
                contact_info={"email": "bob@company.com"},
                preferred_channels=[NotificationChannel.EMAIL],
                priority_level=NotificationPriority.MEDIUM,
                role="Developer",
                department="IT",
                influence_level=3,
                crisis_relevance={"system_outage": 7}
            )
        ]
    
    @pytest.fixture
    def sample_templates(self):
        """Create sample notification templates"""
        return [
            NotificationTemplate(
                id="template_1",
                name="Security Breach - Executive",
                crisis_type="security_breach",
                stakeholder_type=StakeholderType.EXECUTIVE,
                channel=NotificationChannel.EMAIL,
                subject_template="URGENT: Security Incident - {crisis_type}",
                body_template="Dear {stakeholder_name}, we have detected a {crisis_type} at {start_time}. Severity: {severity}. Please join the crisis response immediately.",
                approval_required=False,
                auto_send=True
            ),
            NotificationTemplate(
                id="template_2",
                name="System Outage - Employee",
                crisis_type="system_outage",
                stakeholder_type=StakeholderType.EMPLOYEE,
                channel=NotificationChannel.EMAIL,
                subject_template="System Outage Notification",
                body_template="Hi {stakeholder_name}, we are experiencing a {crisis_type}. Status: {current_status}. {action_required}",
                approval_required=False,
                auto_send=True
            )
        ]
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis"""
        return Crisis(
            id="crisis_1",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.utcnow(),
            affected_areas=["user_database", "payment_system"],
            stakeholders_impacted=["customers", "employees"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[]
        )
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert isinstance(engine.stakeholders, dict)
        assert isinstance(engine.templates, dict)
        assert isinstance(engine.groups, dict)
        assert isinstance(engine.communication_plans, dict)
        assert isinstance(engine.notification_queue, list)
        assert isinstance(engine.delivery_providers, dict)
    
    def test_add_stakeholder(self, engine, sample_stakeholders):
        """Test adding stakeholder"""
        stakeholder = sample_stakeholders[0]
        
        result = engine.add_stakeholder(stakeholder)
        
        assert result is True
        assert stakeholder.id in engine.stakeholders
        assert engine.stakeholders[stakeholder.id] == stakeholder
    
    def test_update_stakeholder(self, engine, sample_stakeholders):
        """Test updating stakeholder"""
        stakeholder = sample_stakeholders[0]
        engine.add_stakeholder(stakeholder)
        
        updates = {
            "name": "John Updated CEO",
            "priority_level": NotificationPriority.CRITICAL
        }
        
        result = engine.update_stakeholder(stakeholder.id, updates)
        
        assert result is True
        assert engine.stakeholders[stakeholder.id].name == "John Updated CEO"
        assert engine.stakeholders[stakeholder.id].priority_level == NotificationPriority.CRITICAL
    
    def test_add_notification_template(self, engine, sample_templates):
        """Test adding notification template"""
        template = sample_templates[0]
        
        result = engine.add_notification_template(template)
        
        assert result is True
        assert template.id in engine.templates
        assert engine.templates[template.id] == template
    
    def test_prioritize_stakeholders_for_crisis(self, engine, sample_stakeholders, sample_crisis):
        """Test stakeholder prioritization for crisis"""
        # Add stakeholders to engine
        for stakeholder in sample_stakeholders:
            engine.add_stakeholder(stakeholder)
        
        prioritized = engine._prioritize_stakeholders_for_crisis(sample_crisis)
        
        assert len(prioritized) > 0
        # CEO should be first due to high relevance and priority
        assert prioritized[0].role == "CEO"
        # CTO should be second due to high relevance
        assert prioritized[1].role == "CTO"
    
    def test_calculate_stakeholder_relevance(self, engine, sample_stakeholders, sample_crisis):
        """Test stakeholder relevance calculation"""
        stakeholder = sample_stakeholders[0]  # CEO
        
        relevance = engine._calculate_stakeholder_relevance(stakeholder, sample_crisis)
        
        assert relevance > 0
        # Should be high due to security breach relevance and high severity
        assert relevance >= 10.0
    
    def test_select_notification_template(self, engine, sample_templates):
        """Test notification template selection"""
        # Add templates to engine
        for template in sample_templates:
            engine.add_notification_template(template)
        
        # Test exact match
        template = engine._select_notification_template(
            "security_breach", StakeholderType.EXECUTIVE
        )
        
        assert template is not None
        assert template.crisis_type == "security_breach"
        assert template.stakeholder_type == StakeholderType.EXECUTIVE
    
    def test_select_optimal_channel(self, engine, sample_stakeholders):
        """Test optimal channel selection"""
        stakeholder = sample_stakeholders[0]  # Has email and SMS
        
        # Test immediate notification (should prefer faster channels)
        channel = engine._select_optimal_channel(stakeholder, immediate=True)
        assert channel in [NotificationChannel.SMS, NotificationChannel.PHONE, NotificationChannel.PUSH]
        
        # Test regular notification (should use preferred channels)
        channel = engine._select_optimal_channel(stakeholder, immediate=False)
        assert channel in stakeholder.preferred_channels
    
    def test_customize_message_content(self, engine, sample_templates, sample_crisis, sample_stakeholders):
        """Test message content customization"""
        template = sample_templates[0]
        stakeholder = sample_stakeholders[0]
        
        content = engine._customize_message_content(template, sample_crisis, stakeholder)
        
        assert "subject" in content
        assert "body" in content
        assert "variables" in content
        
        # Check variable substitution
        assert stakeholder.name in content["body"]
        assert sample_crisis.crisis_type.value.replace("_", " ").title() in content["body"]
    
    def test_determine_notification_priority(self, engine, sample_crisis, sample_stakeholders):
        """Test notification priority determination"""
        stakeholder = sample_stakeholders[0]
        
        # Test immediate notification
        priority = engine._determine_notification_priority(sample_crisis, stakeholder, immediate=True)
        assert priority == NotificationPriority.CRITICAL
        
        # Test critical crisis
        critical_crisis = sample_crisis
        critical_crisis.severity_level = SeverityLevel.CRITICAL
        priority = engine._determine_notification_priority(critical_crisis, stakeholder, immediate=False)
        assert priority == NotificationPriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_send_single_notification(self, engine):
        """Test sending single notification"""
        notification = NotificationMessage(
            crisis_id="test_crisis",
            stakeholder_id="test_stakeholder",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.HIGH,
            subject="Test Subject",
            content="Test Content"
        )
        
        with patch.object(engine, '_send_email_notification', new_callable=AsyncMock) as mock_send:
            result = await engine._send_single_notification(notification)
            
            assert result["success"] is True
            mock_send.assert_called_once_with(notification)
    
    @pytest.mark.asyncio
    async def test_create_stakeholder_notification(self, engine, sample_crisis, sample_stakeholders, sample_templates):
        """Test creating stakeholder notification"""
        # Setup engine with stakeholder and template
        stakeholder = sample_stakeholders[0]
        template = sample_templates[0]
        engine.add_stakeholder(stakeholder)
        engine.add_notification_template(template)
        
        # Create communication plan
        comm_plan = engine._get_or_create_communication_plan(sample_crisis)
        
        notification = await engine._create_stakeholder_notification(
            sample_crisis, stakeholder, comm_plan, immediate=True
        )
        
        assert notification is not None
        assert notification.crisis_id == sample_crisis.id
        assert notification.stakeholder_id == stakeholder.id
        assert notification.priority == NotificationPriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_notify_stakeholders_immediate(self, engine, sample_crisis, sample_stakeholders, sample_templates):
        """Test immediate stakeholder notification"""
        # Setup engine
        for stakeholder in sample_stakeholders:
            engine.add_stakeholder(stakeholder)
        for template in sample_templates:
            engine.add_notification_template(template)
        
        # Mock notification sending
        with patch.object(engine, '_send_single_notification', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"success": True}
            
            result = await engine.notify_stakeholders_immediate(sample_crisis)
            
            assert result["success"] is True
            assert result["notifications_sent"] > 0
            assert result["stakeholders_notified"] > 0
            assert "batch_id" in result
            assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_notify_specific_stakeholders(self, engine, sample_crisis, sample_stakeholders, sample_templates):
        """Test notifying specific stakeholders"""
        # Setup engine
        for stakeholder in sample_stakeholders:
            engine.add_stakeholder(stakeholder)
        for template in sample_templates:
            engine.add_notification_template(template)
        
        # Notify only CEO
        stakeholder_ids = [sample_stakeholders[0].id]
        
        with patch.object(engine, '_send_single_notification', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"success": True}
            
            result = await engine.notify_stakeholders_immediate(
                sample_crisis, stakeholder_ids=stakeholder_ids
            )
            
            assert result["success"] is True
            assert result["stakeholders_notified"] == 1
    
    @pytest.mark.asyncio
    async def test_send_notification_batch(self, engine):
        """Test sending notification batch"""
        from scrollintel.models.crisis_communication_models import NotificationBatch
        
        # Create test notifications
        notifications = [
            NotificationMessage(
                crisis_id="test_crisis",
                stakeholder_id=f"stakeholder_{i}",
                channel=NotificationChannel.EMAIL,
                priority=NotificationPriority.MEDIUM,
                subject=f"Test Subject {i}",
                content=f"Test Content {i}"
            )
            for i in range(3)
        ]
        
        batch = NotificationBatch(
            crisis_id="test_crisis",
            name="Test Batch",
            messages=[n.id for n in notifications]
        )
        
        with patch.object(engine, '_send_single_notification', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"success": True}
            
            results = await engine._send_notification_batch(batch, notifications)
            
            assert results["success"] == 3
            assert results["failed"] == 0
            assert len(results["details"]) == 3
            assert batch.status == "completed"
    
    def test_get_or_create_communication_plan(self, engine, sample_crisis):
        """Test communication plan creation"""
        plan = engine._get_or_create_communication_plan(sample_crisis)
        
        assert plan is not None
        assert plan.crisis_id == sample_crisis.id
        assert plan.crisis_type == sample_crisis.crisis_type.value
        assert len(plan.phases) > 0
    
    def test_calculate_notification_metrics(self, engine):
        """Test notification metrics calculation"""
        notifications = [
            NotificationMessage(status=NotificationStatus.SENT),
            NotificationMessage(status=NotificationStatus.SENT),
            NotificationMessage(status=NotificationStatus.FAILED)
        ]
        
        metrics = engine._calculate_notification_metrics(notifications)
        
        assert metrics.total_sent == 2
        assert metrics.total_failed == 1
        assert metrics.delivery_rate == 2/3
    
    def test_stakeholder_type_relevance_matrix(self, engine):
        """Test stakeholder type relevance matrix"""
        # Test security breach relevance
        relevance = engine._get_stakeholder_type_relevance(
            StakeholderType.BOARD_MEMBER, CrisisType.SECURITY_BREACH
        )
        assert relevance == 1.0
        
        # Test system outage relevance
        relevance = engine._get_stakeholder_type_relevance(
            StakeholderType.CUSTOMER, CrisisType.SYSTEM_OUTAGE
        )
        assert relevance == 1.0
        
        # Test financial crisis relevance
        relevance = engine._get_stakeholder_type_relevance(
            StakeholderType.INVESTOR, CrisisType.FINANCIAL_CRISIS
        )
        assert relevance == 1.0
    
    def test_channel_availability_check(self, engine, sample_stakeholders):
        """Test channel availability checking"""
        stakeholder = sample_stakeholders[0]  # Has email and phone
        
        # Test available channels
        assert engine._is_channel_available(NotificationChannel.EMAIL, stakeholder) is True
        assert engine._is_channel_available(NotificationChannel.SMS, stakeholder) is True
        
        # Test unavailable channel
        assert engine._is_channel_available(NotificationChannel.SLACK, stakeholder) is False
    
    def test_impact_descriptions(self, engine, sample_crisis):
        """Test impact description generation"""
        # Test customer impact
        impact = engine._get_customer_impact_description(sample_crisis)
        assert isinstance(impact, str)
        assert len(impact) > 0
        
        # Test employee action
        stakeholder = Stakeholder(department="IT")
        action = engine._get_employee_action_required(sample_crisis, stakeholder)
        assert isinstance(action, str)
        assert "crisis response center" in action.lower()
        
        # Test business impact
        impact = engine._get_business_impact_summary(sample_crisis)
        assert isinstance(impact, str)
        assert sample_crisis.severity_level.value in impact
    
    @pytest.mark.asyncio
    async def test_notification_failure_handling(self, engine):
        """Test notification failure handling"""
        notification = NotificationMessage(
            crisis_id="test_crisis",
            stakeholder_id="test_stakeholder",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.HIGH,
            subject="Test Subject",
            content="Test Content"
        )
        
        with patch.object(engine, '_send_email_notification', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Network error")
            
            result = await engine._send_single_notification(notification)
            
            assert result["success"] is False
            assert "error" in result
    
    def test_error_handling_add_stakeholder(self, engine):
        """Test error handling when adding stakeholder"""
        # Test with invalid stakeholder data
        with patch.object(engine.logger, 'error') as mock_logger:
            result = engine.add_stakeholder(None)
            assert result is False
            mock_logger.assert_called_once()
    
    def test_error_handling_update_stakeholder(self, engine):
        """Test error handling when updating stakeholder"""
        # Test updating non-existent stakeholder
        result = engine.update_stakeholder("non_existent", {"name": "Test"})
        assert result is False
    
    @pytest.mark.asyncio
    async def test_concurrent_notification_sending(self, engine, sample_crisis, sample_stakeholders, sample_templates):
        """Test concurrent notification sending"""
        # Setup engine with multiple stakeholders
        for stakeholder in sample_stakeholders:
            engine.add_stakeholder(stakeholder)
        for template in sample_templates:
            engine.add_notification_template(template)
        
        # Mock notification sending with delay to test concurrency
        async def mock_send_with_delay(notification):
            await asyncio.sleep(0.1)  # Simulate network delay
            return {"success": True}
        
        with patch.object(engine, '_send_single_notification', side_effect=mock_send_with_delay):
            start_time = datetime.utcnow()
            result = await engine.notify_stakeholders_immediate(sample_crisis)
            end_time = datetime.utcnow()
            
            # Should complete faster than sequential sending
            duration = (end_time - start_time).total_seconds()
            assert duration < 0.5  # Should be much faster than 0.3 seconds (3 * 0.1)
            assert result["success"] is True