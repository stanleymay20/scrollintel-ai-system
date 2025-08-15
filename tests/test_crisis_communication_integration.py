"""
Tests for Crisis Communication Integration System

Tests the integration of crisis leadership capabilities with communication systems,
ensuring seamless crisis-aware communication across all channels.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.crisis_communication_integration import (
    CrisisCommunicationIntegrator,
    CommunicationChannelType,
    CommunicationIntegrationConfig,
    CrisisContext
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisStatus, CrisisType, SeverityLevel
from scrollintel.models.crisis_communication_integration_models import (
    CommunicationMessageModel,
    BroadcastModel,
    MessagePriority
)


class TestCrisisCommunicationIntegrator:
    """Test cases for CrisisCommunicationIntegrator"""
    
    @pytest.fixture
    def integration_config(self):
        """Create test integration configuration"""
        return CommunicationIntegrationConfig(
            enabled_channels=[
                CommunicationChannelType.EMAIL,
                CommunicationChannelType.SLACK,
                CommunicationChannelType.DASHBOARD
            ],
            crisis_aware_filtering=True,
            auto_context_injection=True,
            priority_routing=True
        )
    
    @pytest.fixture
    def integrator(self, integration_config):
        """Create test integrator instance"""
        return CrisisCommunicationIntegrator(integration_config)
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis for testing"""
        return Crisis(
            id="crisis_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now(),
            affected_areas=["api", "database"],
            stakeholders_impacted=["customers", "support_team"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
    
    @pytest.mark.asyncio
    async def test_register_crisis(self, integrator, sample_crisis):
        """Test crisis registration for communication integration"""
        # Register crisis
        success = await integrator.register_crisis(sample_crisis)
        
        # Verify registration
        assert success is True
        assert sample_crisis.id in integrator.active_crises
        
        # Verify crisis context
        crisis_context = integrator.active_crises[sample_crisis.id]
        assert crisis_context.crisis_id == sample_crisis.id
        assert crisis_context.crisis_type == sample_crisis.crisis_type
        assert crisis_context.severity_level == 3  # HIGH maps to 3
        assert crisis_context.escalation_level > 0
    
    @pytest.mark.asyncio
    async def test_process_communication_without_crisis(self, integrator):
        """Test communication processing when no crisis is active"""
        # Process normal communication
        result = await integrator.process_communication(
            channel=CommunicationChannelType.EMAIL,
            message="Hello, how are things?",
            sender="user@example.com"
        )
        
        # Verify normal processing
        assert result["success"] is True
        assert "response" in result
        assert result["crisis_context"] == {}
    
    @pytest.mark.asyncio
    async def test_process_communication_with_crisis(self, integrator, sample_crisis):
        """Test communication processing during active crisis"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Process communication during crisis
        result = await integrator.process_communication(
            channel=CommunicationChannelType.SLACK,
            message="What's the status of the outage?",
            sender="manager@example.com"
        )
        
        # Verify crisis-aware processing
        assert result["success"] is True
        assert "crisis_context" in result
        assert result["crisis_context"]["active_crisis"] is True
        assert result["crisis_context"]["crisis_type"] == "system_outage"
        assert result["crisis_context"]["severity"] == 3
    
    @pytest.mark.asyncio
    async def test_broadcast_crisis_update(self, integrator, sample_crisis):
        """Test broadcasting crisis updates across channels"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Broadcast update
        result = await integrator.broadcast_crisis_update(
            crisis_id=sample_crisis.id,
            update_message="Systems are being restored",
            target_channels=[CommunicationChannelType.EMAIL, CommunicationChannelType.SLACK]
        )
        
        # Verify broadcast
        assert result["success"] is True
        assert result["crisis_id"] == sample_crisis.id
        assert "broadcast_results" in result
        assert "email" in result["broadcast_results"]
        assert "slack" in result["broadcast_results"]
    
    @pytest.mark.asyncio
    async def test_crisis_context_injection(self, integrator, sample_crisis):
        """Test automatic crisis context injection"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Test context injection
        context = {"user_id": "123", "department": "engineering"}
        enhanced_context = await integrator._inject_crisis_context(context)
        
        # Verify context enhancement
        assert "crisis_info" in enhanced_context
        assert enhanced_context["crisis_info"]["active_crisis"] is True
        assert enhanced_context["crisis_info"]["crisis_id"] == sample_crisis.id
        assert enhanced_context["user_id"] == "123"  # Original context preserved
    
    @pytest.mark.asyncio
    async def test_crisis_aware_response_generation(self, integrator, sample_crisis):
        """Test crisis-aware response generation"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Generate crisis-aware response
        context = await integrator._inject_crisis_context({})
        response = await integrator._generate_crisis_aware_response(
            channel=CommunicationChannelType.EMAIL,
            message="Is everything working?",
            sender="user@example.com",
            context=context
        )
        
        # Verify crisis awareness in response
        assert response is not None
        assert len(response) > 0
        # Response should be different from normal response
        normal_response = await integrator._generate_normal_response(
            "Is everything working?", "user@example.com", {}
        )
        assert response != normal_response
    
    @pytest.mark.asyncio
    async def test_crisis_protocol_application(self, integrator, sample_crisis):
        """Test application of crisis communication protocols"""
        # Register high-severity crisis
        sample_crisis.severity_level = 4
        await integrator.register_crisis(sample_crisis)
        
        # Apply crisis protocols
        context = await integrator._inject_crisis_context({})
        response = "System status update"
        processed_response = await integrator._apply_crisis_protocols(
            response, CommunicationChannelType.EMAIL, context
        )
        
        # Verify protocol application
        assert processed_response != response  # Should be modified
        assert "CRISIS" in processed_response or "ALERT" in processed_response
    
    @pytest.mark.asyncio
    async def test_communication_routing(self, integrator, sample_crisis):
        """Test crisis communication routing"""
        # Register high-severity crisis
        sample_crisis.severity_level = 4
        await integrator.register_crisis(sample_crisis)
        
        # Test routing
        context = await integrator._inject_crisis_context({})
        routing_result = await integrator._route_crisis_communication(
            "Test message", CommunicationChannelType.SLACK, context
        )
        
        # Verify routing decisions
        assert routing_result["primary_channel"] == "slack"
        assert routing_result["escalated"] is True  # High severity should escalate
        assert len(routing_result["additional_channels"]) > 0
    
    @pytest.mark.asyncio
    async def test_crisis_relevance_calculation(self, integrator, sample_crisis):
        """Test crisis relevance calculation for context"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Test relevance with matching context
        context = {
            "affected_systems": ["api"],
            "stakeholders": ["customers"]
        }
        crisis_context = integrator.active_crises[sample_crisis.id]
        relevance = integrator._calculate_crisis_relevance(crisis_context, context)
        
        # Should have positive relevance due to overlaps
        assert relevance > 0
        
        # Test relevance with non-matching context
        unrelated_context = {
            "affected_systems": ["billing"],
            "stakeholders": ["vendors"]
        }
        unrelated_relevance = integrator._calculate_crisis_relevance(
            crisis_context, unrelated_context
        )
        
        # Should have lower relevance
        assert unrelated_relevance < relevance
    
    @pytest.mark.asyncio
    async def test_message_customization_for_channels(self, integrator, sample_crisis):
        """Test message customization for different channels"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        crisis_context = integrator.active_crises[sample_crisis.id]
        
        base_message = "System outage has been resolved"
        
        # Test SMS customization (should truncate)
        sms_message = await integrator._customize_message_for_channel(
            base_message * 10,  # Long message
            CommunicationChannelType.SMS,
            crisis_context
        )
        assert len(sms_message) <= 163  # SMS limit + "..."
        
        # Test email customization (should add subject)
        email_message = await integrator._customize_message_for_channel(
            base_message,
            CommunicationChannelType.EMAIL,
            crisis_context
        )
        assert "Subject:" in email_message
        
        # Test Slack customization (should add formatting)
        slack_message = await integrator._customize_message_for_channel(
            base_message,
            CommunicationChannelType.SLACK,
            crisis_context
        )
        assert "🚨" in slack_message or "*" in slack_message
    
    @pytest.mark.asyncio
    async def test_crisis_status_update(self, integrator, sample_crisis):
        """Test crisis status updates and communication"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Update status
        await integrator.update_crisis_status(sample_crisis.id, CrisisStatus.RESOLVING)
        
        # Verify status update
        crisis_context = integrator.active_crises[sample_crisis.id]
        assert crisis_context.status == CrisisStatus.RESOLVING
        assert crisis_context.last_update > crisis_context.start_time
    
    @pytest.mark.asyncio
    async def test_crisis_resolution(self, integrator, sample_crisis):
        """Test crisis resolution and cleanup"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        assert sample_crisis.id in integrator.active_crises
        
        # Resolve crisis
        await integrator.resolve_crisis(sample_crisis.id)
        
        # Verify cleanup
        assert sample_crisis.id not in integrator.active_crises
    
    @pytest.mark.asyncio
    async def test_multiple_active_crises(self, integrator):
        """Test handling multiple active crises"""
        # Create multiple crises
        crisis1 = Crisis(
            id="crisis_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now(),
            affected_areas=["api"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        crisis2 = Crisis(
            id="crisis_002",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL,
            start_time=datetime.now(),
            affected_areas=["database"],
            stakeholders_impacted=["all_users"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        # Register both crises
        await integrator.register_crisis(crisis1)
        await integrator.register_crisis(crisis2)
        
        # Verify both are active
        assert len(integrator.active_crises) == 2
        assert crisis1.id in integrator.active_crises
        assert crisis2.id in integrator.active_crises
        
        # Test communication processing finds most relevant crisis
        context = {"affected_systems": ["database"]}
        enhanced_context = await integrator._inject_crisis_context(context)
        
        # Should find crisis2 as more relevant due to database overlap
        assert enhanced_context["crisis_info"]["crisis_id"] == crisis2.id
    
    def test_escalation_level_calculation(self, integrator):
        """Test escalation level calculation"""
        # Test low severity crisis
        low_crisis = Crisis(
            id="low_crisis",
            crisis_type=CrisisType.PERFORMANCE_DEGRADATION,
            severity_level=SeverityLevel.MEDIUM,
            start_time=datetime.now(),
            affected_areas=["api"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        low_escalation = integrator._calculate_escalation_level(low_crisis)
        assert low_escalation >= 2
        
        # Test high severity security crisis
        high_crisis = Crisis(
            id="high_crisis",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL,
            start_time=datetime.now(),
            affected_areas=["api", "database", "auth", "billing"],
            stakeholders_impacted=["customers", "partners", "employees", "regulators", "media"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        high_escalation = integrator._calculate_escalation_level(high_crisis)
        assert high_escalation > low_escalation
        assert high_escalation <= 5  # Should be capped at 5
    
    def test_get_active_crises(self, integrator, sample_crisis):
        """Test getting list of active crises"""
        # Initially no active crises
        active_crises = integrator.get_active_crises()
        assert len(active_crises) == 0
        
        # Register crisis
        asyncio.run(integrator.register_crisis(sample_crisis))
        
        # Should have one active crisis
        active_crises = integrator.get_active_crises()
        assert len(active_crises) == 1
        assert active_crises[0]["crisis_id"] == sample_crisis.id
        assert active_crises[0]["crisis_type"] == sample_crisis.crisis_type.value
        assert active_crises[0]["severity"] == sample_crisis.severity_level
    
    @pytest.mark.asyncio
    async def test_communication_with_context(self, integrator, sample_crisis):
        """Test communication processing with additional context"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Process communication with context
        context = {
            "user_role": "admin",
            "department": "engineering",
            "affected_systems": ["api"]
        }
        
        result = await integrator.process_communication(
            channel=CommunicationChannelType.DASHBOARD,
            message="Need status update",
            sender="admin@example.com",
            context=context
        )
        
        # Verify context is preserved and enhanced
        assert result["success"] is True
        assert "crisis_context" in result
        assert result["crisis_context"]["active_crisis"] is True
        # Original context should be preserved in the processing
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all_channels(self, integrator, sample_crisis):
        """Test broadcasting to all enabled channels"""
        # Register crisis
        await integrator.register_crisis(sample_crisis)
        
        # Broadcast without specifying channels (should use all enabled)
        result = await integrator.broadcast_crisis_update(
            crisis_id=sample_crisis.id,
            update_message="All systems operational"
        )
        
        # Verify broadcast to all enabled channels
        assert result["success"] is True
        broadcast_results = result["broadcast_results"]
        
        # Should have results for all enabled channels
        for channel in integrator.config.enabled_channels:
            assert channel.value in broadcast_results
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_crisis(self, integrator):
        """Test error handling for invalid crisis operations"""
        # Try to broadcast to non-existent crisis
        result = await integrator.broadcast_crisis_update(
            crisis_id="non_existent_crisis",
            update_message="Test message"
        )
        
        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_communication_processing_error_handling(self, integrator):
        """Test error handling in communication processing"""
        # Mock an error in response generation
        with patch.object(integrator, '_generate_crisis_aware_response', 
                         side_effect=Exception("Test error")):
            result = await integrator.process_communication(
                channel=CommunicationChannelType.EMAIL,
                message="Test message",
                sender="test@example.com"
            )
            
            # Should handle error gracefully
            assert result["success"] is False
            assert "error" in result
            assert "Test error" in result["error"]


class TestCommunicationIntegrationConfig:
    """Test cases for CommunicationIntegrationConfig"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = CommunicationIntegrationConfig(
            enabled_channels=[CommunicationChannelType.EMAIL]
        )
        
        assert config.crisis_aware_filtering is True
        assert config.auto_context_injection is True
        assert config.priority_routing is True
        assert len(config.escalation_triggers) == 0
        assert len(config.message_templates) == 0
    
    def test_custom_configuration(self):
        """Test custom configuration values"""
        custom_templates = {
            "security_breach_4_email": "URGENT SECURITY ALERT: {message}"
        }
        custom_triggers = {
            "severity_4": "auto_escalate"
        }
        
        config = CommunicationIntegrationConfig(
            enabled_channels=[CommunicationChannelType.EMAIL, CommunicationChannelType.SLACK],
            crisis_aware_filtering=False,
            auto_context_injection=False,
            priority_routing=False,
            escalation_triggers=custom_triggers,
            message_templates=custom_templates
        )
        
        assert config.crisis_aware_filtering is False
        assert config.auto_context_injection is False
        assert config.priority_routing is False
        assert config.escalation_triggers == custom_triggers
        assert config.message_templates == custom_templates


@pytest.mark.integration
class TestCrisisCommunicationIntegration:
    """Integration tests for crisis communication system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_crisis_communication_flow(self):
        """Test complete crisis communication flow"""
        # Setup
        config = CommunicationIntegrationConfig(
            enabled_channels=[
                CommunicationChannelType.EMAIL,
                CommunicationChannelType.SLACK,
                CommunicationChannelType.DASHBOARD
            ]
        )
        integrator = CrisisCommunicationIntegrator(config)
        
        # Create and register crisis
        crisis = Crisis(
            id="integration_test_crisis",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now(),
            affected_areas=["api", "frontend"],
            stakeholders_impacted=["customers", "support"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        # Step 1: Register crisis
        registration_success = await integrator.register_crisis(crisis)
        assert registration_success is True
        
        # Step 2: Process incoming communication
        comm_result = await integrator.process_communication(
            channel=CommunicationChannelType.SLACK,
            message="What's happening with the system?",
            sender="manager@example.com"
        )
        assert comm_result["success"] is True
        assert comm_result["crisis_context"]["active_crisis"] is True
        
        # Step 3: Broadcast update
        broadcast_result = await integrator.broadcast_crisis_update(
            crisis_id=crisis.id,
            update_message="We are investigating the issue and will provide updates shortly"
        )
        assert broadcast_result["success"] is True
        
        # Step 4: Update crisis status
        await integrator.update_crisis_status(crisis.id, CrisisStatus.RESOLVING)
        
        # Step 5: Resolve crisis
        await integrator.resolve_crisis(crisis.id)
        
        # Verify crisis is no longer active
        active_crises = integrator.get_active_crises()
        assert len(active_crises) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_crisis_handling(self):
        """Test handling multiple concurrent crises"""
        config = CommunicationIntegrationConfig(
            enabled_channels=[CommunicationChannelType.EMAIL, CommunicationChannelType.SLACK]
        )
        integrator = CrisisCommunicationIntegrator(config)
        
        # Create multiple crises
        crises = []
        severity_levels = [SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        for i in range(3):
            crisis = Crisis(
                id=f"concurrent_crisis_{i}",
                crisis_type=CrisisType.SYSTEM_OUTAGE,
                severity_level=severity_levels[i],
                start_time=datetime.now(),
                affected_areas=[f"system_{i}"],
                stakeholders_impacted=[f"team_{i}"],
                current_status=CrisisStatus.ACTIVE,
                response_actions=[],
                resolution_time=None
            )
            crises.append(crisis)
        
        # Register all crises concurrently
        registration_tasks = [integrator.register_crisis(crisis) for crisis in crises]
        registration_results = await asyncio.gather(*registration_tasks)
        
        # Verify all registrations succeeded
        assert all(registration_results)
        assert len(integrator.get_active_crises()) == 3
        
        # Process communications for each crisis
        communication_tasks = []
        for i, crisis in enumerate(crises):
            task = integrator.process_communication(
                channel=CommunicationChannelType.EMAIL,
                message=f"Status update for crisis {i}",
                sender=f"user{i}@example.com",
                context={"affected_systems": [f"system_{i}"]}
            )
            communication_tasks.append(task)
        
        communication_results = await asyncio.gather(*communication_tasks)
        
        # Verify all communications processed successfully
        assert all(result["success"] for result in communication_results)
        
        # Verify each communication got the right crisis context
        for i, result in enumerate(communication_results):
            assert result["crisis_context"]["crisis_id"] == f"concurrent_crisis_{i}"
        
        # Resolve all crises
        resolution_tasks = [integrator.resolve_crisis(crisis.id) for crisis in crises]
        await asyncio.gather(*resolution_tasks)
        
        # Verify all crises resolved
        assert len(integrator.get_active_crises()) == 0