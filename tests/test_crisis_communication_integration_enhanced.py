"""
Tests for Enhanced Crisis Communication Integration

Tests the seamless integration with all communication channels,
crisis communication context in all interactions, and crisis-aware
response generation and messaging.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.crisis_communication_integration import (
    CrisisCommunicationIntegration,
    CrisisContext,
    CrisisLevel,
    CommunicationChannel,
    CommunicationSystemIntegration,
    CrisisAwareResponse,
    crisis_communication_integration
)


class TestEnhancedCrisisCommunicationIntegration:
    
    @pytest.fixture
    def integration_engine(self):
        """Create integration engine for testing"""
        return CrisisCommunicationIntegration()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis context"""
        return CrisisContext(
            crisis_id="crisis_001",
            crisis_type="system_outage",
            severity_level=CrisisLevel.HIGH,
            start_time=datetime.now(),
            affected_systems=["web_app", "api", "database"],
            stakeholders=["customers", "employees", "executives"],
            status="active"
        )
    
    def test_system_integrations_initialization(self, integration_engine):
        """Test that all communication systems are properly initialized"""
        expected_systems = [
            "message_coordination",
            "stakeholder_notification", 
            "media_management",
            "executive_communication",
            "chat_interface",
            "email_system",
            "collaboration_tools"
        ]
        
        for system in expected_systems:
            assert system in integration_engine.integrated_systems
            assert integration_engine.integrated_systems[system].is_active
            assert integration_engine.integrated_systems[system].crisis_context_support
    
    def test_response_enhancers_setup(self, integration_engine):
        """Test that response enhancers are properly configured"""
        expected_enhancers = [
            "chat", "email", "notification", "media", 
            "executive", "customer", "employee", "regulatory"
        ]
        
        for enhancer in expected_enhancers:
            assert enhancer in integration_engine.response_enhancers
            assert callable(integration_engine.response_enhancers[enhancer])
    
    def test_context_propagation_rules(self, integration_engine):
        """Test context propagation rules configuration"""
        expected_rule_types = [
            "all_systems", "customer_facing", "internal_systems", 
            "media_systems", "regulatory_systems"
        ]
        
        for rule_type in expected_rule_types:
            assert rule_type in integration_engine.context_propagation_rules
            assert isinstance(integration_engine.context_propagation_rules[rule_type], dict)
    
    @pytest.mark.asyncio
    async def test_crisis_registration_and_propagation(self, integration_engine, sample_crisis):
        """Test crisis registration and context propagation to all systems"""
        # Register crisis
        result = await integration_engine.register_crisis(sample_crisis)
        
        assert result is True
        assert sample_crisis.crisis_id in integration_engine.active_crises
        
        # Verify context was propagated to systems
        assert hasattr(integration_engine, 'system_contexts')
        assert len(integration_engine.system_contexts) > 0
    
    @pytest.mark.asyncio
    async def test_context_preparation_for_different_systems(self, integration_engine, sample_crisis):
        """Test context preparation for different system types"""
        # Test internal system context
        internal_context = integration_engine._prepare_context_for_system("chat", sample_crisis)
        assert "technical_details" in internal_context
        assert "action_items" in internal_context
        assert "escalation_path" in internal_context
        
        # Test customer-facing system context
        customer_context = integration_engine._prepare_context_for_system("notification", sample_crisis)
        assert "customer_friendly_description" in customer_context
        assert "eta" in customer_context
        assert "support_contact" in customer_context
        
        # Test media system context
        media_context = integration_engine._prepare_context_for_system("media", sample_crisis)
        assert "official_statement" in media_context
        assert "media_contact" in media_context
        assert "company_position" in media_context
    
    def test_crisis_aware_response_generation(self, integration_engine, sample_crisis):
        """Test crisis-aware response generation"""
        # Register crisis first
        integration_engine.active_crises[sample_crisis.crisis_id] = sample_crisis
        
        # Test chat response
        response = integration_engine.generate_crisis_aware_response(
            query="What's the current system status?",
            response_type="chat"
        )
        
        assert isinstance(response, CrisisAwareResponse)
        assert response.crisis_context is not None
        assert "crisis" in response.crisis_enhanced_response.lower()
        assert response.confidence_score > 0
    
    def test_chat_response_enhancement(self, integration_engine, sample_crisis):
        """Test chat response enhancement with crisis context"""
        base_response = "The system is currently being maintained."
        
        enhanced = integration_engine._enhance_chat_response(
            query="What's happening?",
            base_response=base_response,
            crisis_context=sample_crisis,
            user_context=None
        )
        
        assert "🚨" in enhanced
        assert "Crisis Alert" in enhanced
        assert sample_crisis.crisis_type.replace('_', ' ') in enhanced
        assert sample_crisis.severity_level.value.upper() in enhanced
    
    def test_email_response_enhancement(self, integration_engine, sample_crisis):
        """Test email response enhancement with crisis context"""
        base_response = "Thank you for your inquiry."
        
        enhanced = integration_engine._enhance_email_response(
            query="Need help",
            base_response=base_response,
            crisis_context=sample_crisis,
            user_context=None
        )
        
        assert f"[CRISIS-{sample_crisis.severity_level.value.upper()}]" in enhanced
        assert "CRISIS UPDATE" in enhanced
        assert sample_crisis.crisis_id in enhanced
    
    def test_executive_response_enhancement(self, integration_engine, sample_crisis):
        """Test executive response enhancement with strategic context"""
        base_response = "Here's the current situation."
        
        enhanced = integration_engine._enhance_executive_response(
            query="What's the business impact?",
            base_response=base_response,
            crisis_context=sample_crisis,
            user_context={"stakeholder_type": "executive"}
        )
        
        assert "EXECUTIVE BRIEFING" in enhanced
        assert "Business Impact" in enhanced
        assert "Next Actions" in enhanced
    
    def test_customer_response_enhancement(self, integration_engine, sample_crisis):
        """Test customer response enhancement with customer-friendly context"""
        base_response = "We're here to help."
        
        enhanced = integration_engine._enhance_customer_response(
            query="Is the service working?",
            base_response=base_response,
            crisis_context=sample_crisis,
            user_context={"stakeholder_type": "customer"}
        )
        
        assert "keep you informed" in enhanced
        assert "working to resolve" in enhanced
        assert "Thank you for your patience" in enhanced
    
    def test_escalation_detection(self, integration_engine, sample_crisis):
        """Test escalation detection during crisis"""
        # Test escalation for urgent query during high severity crisis
        should_escalate = integration_engine._should_escalate_query(
            query="URGENT: System is completely down!",
            crisis_context=sample_crisis,
            user_context=None
        )
        assert should_escalate is True
        
        # Test escalation for VIP stakeholder
        should_escalate = integration_engine._should_escalate_query(
            query="What's happening?",
            crisis_context=sample_crisis,
            user_context={"stakeholder_type": "executive"}
        )
        assert should_escalate is True
        
        # Test no escalation for normal query
        should_escalate = integration_engine._should_escalate_query(
            query="How are you?",
            crisis_context=sample_crisis,
            user_context={"stakeholder_type": "employee"}
        )
        assert should_escalate is False
    
    def test_additional_actions_generation(self, integration_engine, sample_crisis):
        """Test additional actions generation based on context"""
        actions = integration_engine._get_additional_actions(
            query="What's the status?",
            crisis_context=sample_crisis,
            user_context={"stakeholder_type": "customer"}
        )
        
        assert len(actions) > 0
        assert "Monitor crisis status updates" in actions
        assert "Check service status page" in actions
        assert "Contact customer support if needed" in actions
    
    def test_confidence_score_calculation(self, integration_engine, sample_crisis):
        """Test response confidence score calculation"""
        # Simple query should have higher confidence
        confidence = integration_engine._calculate_response_confidence(
            query="What's the status?",
            crisis_context=sample_crisis
        )
        assert 0.0 <= confidence <= 1.0
        
        # Complex query should have lower confidence
        complex_confidence = integration_engine._calculate_response_confidence(
            query="Why did this complex technical issue happen and how will you prevent it?",
            crisis_context=sample_crisis
        )
        assert complex_confidence < confidence
    
    @pytest.mark.asyncio
    async def test_communication_channel_integration(self, integration_engine):
        """Test integration with all communication channels"""
        result = await integration_engine.integrate_with_all_communication_channels()
        
        assert result["success"] is True
        assert result["integrated_systems"] > 0
        assert "results" in result
        
        # Verify each system was processed
        for system_name in integration_engine.integrated_systems.keys():
            assert system_name in result["results"]
    
    def test_integration_status_reporting(self, integration_engine):
        """Test integration status reporting"""
        status = integration_engine.get_integration_status()
        
        assert "total_systems" in status
        assert "active_systems" in status
        assert "crisis_aware_systems" in status
        assert "systems" in status
        
        assert status["total_systems"] > 0
        assert status["crisis_aware_systems"] > 0
    
    def test_helper_methods(self, integration_engine, sample_crisis):
        """Test various helper methods"""
        # Test customer-friendly description
        description = integration_engine._get_customer_friendly_description(sample_crisis)
        assert isinstance(description, str)
        assert len(description) > 0
        
        # Test ETA
        eta = integration_engine._get_estimated_resolution_time(sample_crisis)
        assert eta is None or isinstance(eta, str)
        
        # Test business impact
        impact = integration_engine._get_business_impact(sample_crisis)
        assert isinstance(impact, str)
        assert len(impact) > 0
        
        # Test crisis duration
        duration = integration_engine._get_crisis_duration(sample_crisis)
        assert isinstance(duration, str)
        assert "h" in duration and "m" in duration
    
    def test_most_relevant_crisis_selection(self, integration_engine):
        """Test selection of most relevant crisis for queries"""
        # Create multiple crises with different severities
        low_crisis = CrisisContext(
            crisis_id="low_001",
            crisis_type="minor_issue",
            severity_level=CrisisLevel.LOW,
            start_time=datetime.now()
        )
        
        critical_crisis = CrisisContext(
            crisis_id="critical_001", 
            crisis_type="major_outage",
            severity_level=CrisisLevel.CRITICAL,
            start_time=datetime.now()
        )
        
        integration_engine.active_crises["low_001"] = low_crisis
        integration_engine.active_crises["critical_001"] = critical_crisis
        
        # Should select the critical crisis
        relevant_crisis = integration_engine._get_most_relevant_crisis(
            query="What's happening?",
            user_context=None
        )
        
        assert relevant_crisis is not None
        assert relevant_crisis.crisis_id == "critical_001"
        assert relevant_crisis.severity_level == CrisisLevel.CRITICAL


@pytest.mark.asyncio
async def test_global_instance():
    """Test the global crisis communication integration instance"""
    assert crisis_communication_integration is not None
    assert isinstance(crisis_communication_integration, CrisisCommunicationIntegration)
    
    # Test basic functionality
    status = crisis_communication_integration.get_integration_status()
    assert isinstance(status, dict)
    assert "total_systems" in status


if __name__ == "__main__":
    pytest.main([__file__])