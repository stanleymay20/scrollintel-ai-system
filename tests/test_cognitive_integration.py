"""
Tests for the Cognitive Integration System
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from scrollintel.engines.cognitive_integrator import (
    CognitiveIntegrator, CognitiveState, AttentionAllocation, IntegratedDecision,
    CognitiveLoadLevel, AttentionFocus
)


class TestCognitiveIntegrator:
    """Test suite for CognitiveIntegrator"""
    
    @pytest_asyncio.fixture
    async def integrator(self):
        """Create a cognitive integrator instance for testing"""
        integrator = CognitiveIntegrator()
        await integrator.start()
        yield integrator
        await integrator.stop()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test cognitive integrator initialization"""
        integrator = CognitiveIntegrator()
        
        # Check that all subsystems are initialized
        assert integrator.consciousness_engine is not None
        assert integrator.awareness_engine is not None
        assert integrator.intuitive_reasoning is not None
        assert integrator.memory_system is not None
        assert integrator.meta_learning is not None
        assert integrator.emotion_simulator is not None
        assert integrator.personality_engine is not None
        
        # Check initial state
        assert isinstance(integrator.cognitive_state, CognitiveState)
        assert isinstance(integrator.attention_allocation, AttentionAllocation)
        assert integrator.decision_history == []
        assert not integrator._running
    
    @pytest.mark.asyncio
    async def test_start_stop_system(self):
        """Test starting and stopping the cognitive integration system"""
        integrator = CognitiveIntegrator()
        
        # Test start
        await integrator.start()
        assert integrator._running
        assert integrator._integration_task is not None
        assert integrator._monitoring_task is not None
        assert integrator._regulation_task is not None
        
        # Test stop
        await integrator.stop()
        assert not integrator._running
    
    @pytest.mark.asyncio
    async def test_process_complex_situation(self, integrator):
        """Test processing a complex situation through all cognitive systems"""
        situation = "We need to develop a new AI system that can handle multiple complex tasks simultaneously while maintaining ethical standards and user trust."
        context = {
            "complexity": 0.9,
            "stakeholders": ["users", "developers", "regulators"],
            "constraints": ["ethical_guidelines", "performance_requirements"],
            "time_pressure": 0.7
        }
        
        decision = await integrator.process_complex_situation(situation, context)
        
        # Verify decision structure
        assert isinstance(decision, IntegratedDecision)
        assert decision.decision_content is not None
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.reasoning_path) > 0
        assert len(decision.supporting_systems) > 0
        assert decision.integration_quality >= 0.0
        
        # Verify decision is stored
        assert len(integrator.decision_history) == 1
        assert integrator.decision_history[0] == decision
    
    @pytest.mark.asyncio
    async def test_attention_management(self, integrator):
        """Test attention management across cognitive systems"""
        # Test focusing on reasoning
        allocation = await integrator.manage_attention(AttentionFocus.REASONING, 0.8)
        
        assert isinstance(allocation, AttentionAllocation)
        assert allocation.reasoning_attention > allocation.consciousness_attention
        assert allocation.reasoning_attention > allocation.memory_attention
        assert abs(sum([
            allocation.consciousness_attention,
            allocation.reasoning_attention,
            allocation.memory_attention,
            allocation.learning_attention,
            allocation.emotion_attention,
            allocation.integration_attention
        ]) - 1.0) < 0.01  # Should sum to 1.0
        
        # Test focusing on emotion
        allocation = await integrator.manage_attention(AttentionFocus.EMOTION, 0.9)
        assert allocation.emotion_attention > allocation.reasoning_attention
        
        # Verify attention focus is updated
        assert integrator.cognitive_state.attention_focus == AttentionFocus.EMOTION
    
    @pytest.mark.asyncio
    async def test_cognitive_load_balancing(self, integrator):
        """Test cognitive load balancing across systems"""
        # Mock system loads with one overloaded system
        with patch.object(integrator, '_assess_system_loads') as mock_assess:
            mock_assess.return_value = {
                "consciousness": 0.6,
                "reasoning": 0.95,  # Overloaded
                "memory": 0.5,
                "learning": 0.4,
                "emotion": 0.3,
                "integration": 0.7
            }
            
            system_loads = await integrator.balance_cognitive_load()
            
            # Verify load balancing was attempted
            assert isinstance(system_loads, dict)
            assert "reasoning" in system_loads
            
            # Verify cognitive load level is updated
            assert integrator.cognitive_state.cognitive_load in [
                CognitiveLoadLevel.LOW, CognitiveLoadLevel.MODERATE, 
                CognitiveLoadLevel.HIGH, CognitiveLoadLevel.CRITICAL
            ]
    
    @pytest.mark.asyncio
    async def test_coherent_decision_making(self, integrator):
        """Test coherent decision making with multiple options"""
        decision_context = {
            "situation": "Choose the best approach for implementing a new feature",
            "urgency": 0.6,
            "stakeholders": ["product_team", "engineering_team", "users"]
        }
        options = [
            "Implement quickly with basic functionality",
            "Take time to implement comprehensive solution",
            "Implement in phases with iterative improvements"
        ]
        
        decision = await integrator.make_coherent_decision(decision_context, options)
        
        # Verify decision structure
        assert isinstance(decision, IntegratedDecision)
        assert any(option in decision.decision_content for option in options)
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.reasoning_path) > 0
        assert decision.integration_quality >= 0.0
    
    @pytest.mark.asyncio
    async def test_cognitive_health_monitoring(self, integrator):
        """Test cognitive health monitoring"""
        health_report = await integrator.monitor_cognitive_health()
        
        # Verify health report structure
        assert "system_health" in health_report
        assert "integration_health" in health_report
        assert "cognitive_state" in health_report
        assert "attention_allocation" in health_report
        assert "issues_identified" in health_report
        assert "recommendations" in health_report
        assert "overall_score" in health_report
        
        # Verify system health scores
        system_health = health_report["system_health"]
        expected_systems = ["consciousness", "reasoning", "memory", "learning", "emotion", "personality"]
        for system in expected_systems:
            assert system in system_health
            assert 0.0 <= system_health[system] <= 1.0
        
        # Verify overall score
        assert 0.0 <= health_report["overall_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_self_regulation(self, integrator):
        """Test cognitive self-regulation"""
        regulation_result = await integrator.self_regulate()
        
        # Verify regulation result structure
        assert "regulation_needs" in regulation_result
        assert "regulation_applied" in regulation_result
        assert "effectiveness" in regulation_result
        assert "new_cognitive_state" in regulation_result
        
        # Verify regulation needs assessment
        regulation_needs = regulation_result["regulation_needs"]
        assert isinstance(regulation_needs, dict)
        for need, severity in regulation_needs.items():
            assert 0.0 <= severity <= 1.0
        
        # Verify effectiveness score
        assert 0.0 <= regulation_result["effectiveness"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_system_integration_quality(self, integrator):
        """Test the quality of system integration"""
        # Process multiple situations to build history
        situations = [
            "Analyze market trends for strategic planning",
            "Resolve a technical conflict between team members",
            "Design a user-friendly interface for complex functionality"
        ]
        
        decisions = []
        for situation in situations:
            decision = await integrator.process_complex_situation(situation)
            decisions.append(decision)
        
        # Verify integration quality improves or maintains
        integration_qualities = [d.integration_quality for d in decisions]
        assert all(q >= 0.0 for q in integration_qualities)
        
        # Verify system learns from decisions
        assert len(integrator.decision_history) == len(situations)
    
    @pytest.mark.asyncio
    async def test_attention_reallocation_under_load(self, integrator):
        """Test automatic attention reallocation under high cognitive load"""
        # Set high cognitive load
        integrator.cognitive_state.cognitive_load = CognitiveLoadLevel.HIGH
        
        # Mock the auto-reallocation method to verify it's called
        with patch.object(integrator, '_auto_reallocate_attention') as mock_reallocate:
            mock_reallocate.return_value = None
            
            # Trigger the integration loop logic
            await integrator._update_integration_metrics()
            if integrator._needs_attention_reallocation():
                await integrator._auto_reallocate_attention()
            
            # Verify reallocation was considered
            assert integrator.cognitive_state.cognitive_load == CognitiveLoadLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_decision_coherence_validation(self, integrator):
        """Test decision coherence validation"""
        # Create a mock decision with various inputs
        decision = IntegratedDecision(
            decision_id="test_decision",
            decision_content="Test decision content",
            confidence=0.8,
            reasoning_path=["step1", "step2", "step3"],
            supporting_systems=["consciousness", "reasoning", "memory"],
            consciousness_input={"awareness": 0.8},
            intuitive_input={"insight": "test insight"},
            memory_input={"guidance": "test guidance"}
        )
        
        coherence_score = await integrator._validate_decision_coherence(decision)
        
        # Verify coherence score is reasonable
        assert 0.0 <= coherence_score <= 1.0
        
        # Test with minimal decision
        minimal_decision = IntegratedDecision(
            decision_id="minimal_decision",
            decision_content="Minimal decision",
            confidence=0.3,
            reasoning_path=["single_step"],
            supporting_systems=["consciousness"]
        )
        
        minimal_coherence = await integrator._validate_decision_coherence(minimal_decision)
        assert minimal_coherence < coherence_score  # Should be lower coherence
    
    @pytest.mark.asyncio
    async def test_system_status_reporting(self, integrator):
        """Test system status reporting"""
        status = integrator.get_system_status()
        
        # Verify status structure
        assert "running" in status
        assert "cognitive_state" in status
        assert "attention_allocation" in status
        assert "integration_metrics" in status
        assert "decision_history_count" in status
        assert "recent_decisions" in status
        
        # Verify status values
        assert isinstance(status["running"], bool)
        assert isinstance(status["cognitive_state"], CognitiveState)
        assert isinstance(status["attention_allocation"], AttentionAllocation)
        assert isinstance(status["decision_history_count"], int)
        assert isinstance(status["recent_decisions"], list)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_system_integration(self, integrator):
        """Test error handling when subsystems fail"""
        # Mock a subsystem to raise an exception
        with patch.object(integrator.consciousness_engine, 'simulate_awareness') as mock_awareness:
            mock_awareness.side_effect = Exception("Consciousness system error")
            
            # Process situation should still work with other systems
            decision = await integrator.process_complex_situation("Test situation with error")
            
            # Verify decision is still created despite error
            assert isinstance(decision, IntegratedDecision)
            assert decision.decision_content is not None
            
            # Verify error is handled gracefully (consciousness input should show error)
            assert decision.consciousness_input is not None
            assert "error" in decision.consciousness_input
    
    @pytest.mark.asyncio
    async def test_cognitive_state_updates(self, integrator):
        """Test cognitive state updates from decisions"""
        initial_consciousness = integrator.cognitive_state.consciousness_level
        initial_reasoning = integrator.cognitive_state.reasoning_capacity
        
        # Create decision with multiple system inputs
        decision = IntegratedDecision(
            decision_id="test_update",
            decision_content="Test decision",
            confidence=0.8,
            reasoning_path=["test"],
            supporting_systems=["consciousness", "reasoning"],
            consciousness_input={"awareness": 0.9},
            intuitive_input={"insight": "test"},
            integration_quality=0.85
        )
        
        await integrator._update_cognitive_state_from_decision(decision)
        
        # Verify state updates
        assert integrator.cognitive_state.consciousness_level >= initial_consciousness
        assert integrator.cognitive_state.reasoning_capacity >= initial_reasoning
        assert integrator.cognitive_state.integration_coherence == decision.integration_quality
    
    def test_attention_allocation_validation(self):
        """Test attention allocation validation and normalization"""
        integrator = CognitiveIntegrator()
        
        # Create allocation that doesn't sum to 1.0
        allocation = AttentionAllocation(
            consciousness_attention=0.3,
            reasoning_attention=0.4,
            memory_attention=0.2,
            learning_attention=0.1,
            emotion_attention=0.1,
            integration_attention=0.1  # Total = 1.2
        )
        
        validated_allocation = integrator._validate_attention_allocation(allocation)
        
        # Verify normalization
        total = (validated_allocation.consciousness_attention + 
                validated_allocation.reasoning_attention +
                validated_allocation.memory_attention + 
                validated_allocation.learning_attention +
                validated_allocation.emotion_attention + 
                validated_allocation.integration_attention)
        
        assert abs(total - 1.0) < 0.01
    
    def test_cognitive_load_level_determination(self):
        """Test cognitive load level determination"""
        integrator = CognitiveIntegrator()
        
        # Test different load levels
        assert integrator._determine_load_level(0.3) == CognitiveLoadLevel.LOW
        assert integrator._determine_load_level(0.6) == CognitiveLoadLevel.MODERATE
        assert integrator._determine_load_level(0.8) == CognitiveLoadLevel.HIGH
        assert integrator._determine_load_level(0.95) == CognitiveLoadLevel.CRITICAL
    
    def test_situation_complexity_assessment(self):
        """Test situation complexity assessment"""
        integrator = CognitiveIntegrator()
        
        # Simple situation
        simple_complexity = integrator._assess_situation_complexity("Simple task")
        assert 0.0 <= simple_complexity <= 1.0
        
        # Complex situation
        complex_situation = "This is a very complex and challenging situation that requires multiple difficult decisions and uncertain outcomes"
        complex_complexity = integrator._assess_situation_complexity(complex_situation)
        assert complex_complexity > simple_complexity
        
        # Situation with complexity keywords
        keyword_situation = "Complex challenging difficult uncertain multiple"
        keyword_complexity = integrator._assess_situation_complexity(keyword_situation)
        assert keyword_complexity > simple_complexity


class TestCognitiveState:
    """Test suite for CognitiveState"""
    
    def test_cognitive_state_initialization(self):
        """Test cognitive state initialization with defaults"""
        state = CognitiveState()
        
        assert 0.0 <= state.consciousness_level <= 1.0
        assert 0.0 <= state.reasoning_capacity <= 1.0
        assert 0.0 <= state.memory_utilization <= 1.0
        assert 0.0 <= state.learning_efficiency <= 1.0
        assert 0.0 <= state.emotional_stability <= 1.0
        assert 0.0 <= state.integration_coherence <= 1.0
        assert isinstance(state.attention_focus, AttentionFocus)
        assert isinstance(state.cognitive_load, CognitiveLoadLevel)
        assert isinstance(state.active_goals, list)
        assert isinstance(state.processing_queue, list)
        assert isinstance(state.last_updated, datetime)
    
    def test_cognitive_state_custom_values(self):
        """Test cognitive state with custom values"""
        state = CognitiveState(
            consciousness_level=0.9,
            reasoning_capacity=0.8,
            attention_focus=AttentionFocus.REASONING,
            cognitive_load=CognitiveLoadLevel.HIGH,
            active_goals=["goal1", "goal2"]
        )
        
        assert state.consciousness_level == 0.9
        assert state.reasoning_capacity == 0.8
        assert state.attention_focus == AttentionFocus.REASONING
        assert state.cognitive_load == CognitiveLoadLevel.HIGH
        assert state.active_goals == ["goal1", "goal2"]


class TestIntegratedDecision:
    """Test suite for IntegratedDecision"""
    
    def test_integrated_decision_creation(self):
        """Test integrated decision creation"""
        decision = IntegratedDecision(
            decision_id="test_decision",
            decision_content="Test decision content",
            confidence=0.85,
            reasoning_path=["step1", "step2"],
            supporting_systems=["consciousness", "reasoning"]
        )
        
        assert decision.decision_id == "test_decision"
        assert decision.decision_content == "Test decision content"
        assert decision.confidence == 0.85
        assert decision.reasoning_path == ["step1", "step2"]
        assert decision.supporting_systems == ["consciousness", "reasoning"]
        assert decision.integration_quality == 0.0  # Default value
        assert isinstance(decision.timestamp, datetime)
    
    def test_integrated_decision_with_all_inputs(self):
        """Test integrated decision with all system inputs"""
        decision = IntegratedDecision(
            decision_id="full_decision",
            decision_content="Full decision",
            confidence=0.9,
            reasoning_path=["comprehensive_analysis"],
            supporting_systems=["all_systems"],
            consciousness_input={"awareness": 0.8},
            intuitive_input={"insight": "creative_solution"},
            memory_input={"guidance": "past_experience"},
            learning_input={"skill": "new_capability"},
            emotional_input={"state": "confident"},
            personality_input={"trait": "systematic"},
            integration_quality=0.95
        )
        
        assert decision.consciousness_input["awareness"] == 0.8
        assert decision.intuitive_input["insight"] == "creative_solution"
        assert decision.memory_input["guidance"] == "past_experience"
        assert decision.learning_input["skill"] == "new_capability"
        assert decision.emotional_input["state"] == "confident"
        assert decision.personality_input["trait"] == "systematic"
        assert decision.integration_quality == 0.95


class TestAttentionAllocation:
    """Test suite for AttentionAllocation"""
    
    def test_attention_allocation_defaults(self):
        """Test attention allocation with default values"""
        allocation = AttentionAllocation()
        
        # Check all attention values are between 0 and 1
        assert 0.0 <= allocation.consciousness_attention <= 1.0
        assert 0.0 <= allocation.reasoning_attention <= 1.0
        assert 0.0 <= allocation.memory_attention <= 1.0
        assert 0.0 <= allocation.learning_attention <= 1.0
        assert 0.0 <= allocation.emotion_attention <= 1.0
        assert 0.0 <= allocation.integration_attention <= 1.0
        
        # Check total capacity
        assert allocation.total_capacity == 1.0
        assert isinstance(allocation.focus_priority, list)
    
    def test_attention_allocation_custom_values(self):
        """Test attention allocation with custom values"""
        allocation = AttentionAllocation(
            consciousness_attention=0.3,
            reasoning_attention=0.4,
            memory_attention=0.1,
            learning_attention=0.1,
            emotion_attention=0.05,
            integration_attention=0.05,
            focus_priority=[AttentionFocus.REASONING, AttentionFocus.CONSCIOUSNESS]
        )
        
        assert allocation.consciousness_attention == 0.3
        assert allocation.reasoning_attention == 0.4
        assert allocation.focus_priority == [AttentionFocus.REASONING, AttentionFocus.CONSCIOUSNESS]
        
        # Check total is 1.0
        total = (allocation.consciousness_attention + allocation.reasoning_attention +
                allocation.memory_attention + allocation.learning_attention +
                allocation.emotion_attention + allocation.integration_attention)
        assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])