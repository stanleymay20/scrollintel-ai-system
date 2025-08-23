"""
Tests for the Superintelligent Council of Models with Adversarial Collaboration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from datetime import datetime

from scrollintel.core.super_council_of_models import (
    SuperCouncilOfModels,
    AdversarialDebateEngine,
    DebateRole,
    ArgumentationDepth,
    ModelCapability,
    DebateArgument,
    SocraticQuestion
)
from scrollintel.core.super_council_engines import (
    RecursiveArgumentationEngine,
    SocraticQuestioningEngine,
    GameTheoreticOptimizer,
    SwarmIntelligenceCoordinator
)


class TestSuperCouncilOfModels:
    """Test the main SuperCouncilOfModels class"""
    
    @pytest.fixture
    def council(self):
        """Create a SuperCouncilOfModels instance for testing"""
        return SuperCouncilOfModels()
    
    def test_initialization(self, council):
        """Test that the council initializes correctly"""
        assert council.engine_id == "superintelligent_council"
        assert council.name == "Superintelligent Council of Models"
        assert len(council.frontier_models) >= 50  # Should have 50+ models
        assert isinstance(council.adversarial_debate_engine, AdversarialDebateEngine)
        assert isinstance(council.recursive_argumentation_engine, RecursiveArgumentationEngine)
        assert isinstance(council.socratic_questioning_engine, SocraticQuestioningEngine)
        assert isinstance(council.game_theoretic_optimizer, GameTheoreticOptimizer)
        assert isinstance(council.swarm_intelligence_coordinator, SwarmIntelligenceCoordinator)
    
    def test_frontier_models_initialization(self, council):
        """Test that frontier models are properly initialized"""
        # Check that we have the expected frontier models
        expected_models = [
            "gpt-5", "gpt-4-turbo", "claude-4", "claude-3-opus",
            "gemini-ultra", "palm-3", "llama-3-400b", "mistral-large",
            "deepseek-coder", "cohere-command-r"
        ]
        
        for model in expected_models:
            assert model in council.frontier_models
            capability = council.frontier_models[model]
            assert isinstance(capability, ModelCapability)
            assert 0.0 <= capability.reasoning_strength <= 1.0
            assert 0.0 <= capability.creativity_score <= 1.0
            assert 0.0 <= capability.factual_accuracy <= 1.0
            assert 0.0 <= capability.philosophical_depth <= 1.0
            assert 0.0 <= capability.adversarial_robustness <= 1.0
            assert len(capability.specializations) > 0
    
    def test_specialized_models_generation(self, council):
        """Test that specialized models are generated correctly"""
        specialized_models = [model for model in council.frontier_models.keys() 
                            if model.startswith('specialist_')]
        
        # Should have many specialized models
        assert len(specialized_models) >= 40
        
        # Check some specialized domains
        domains_found = set()
        for model_id in specialized_models:
            capability = council.frontier_models[model_id]
            domains_found.update(capability.specializations)
        
        expected_domains = [
            'scientific_reasoning', 'creative_synthesis', 'ethical_analysis',
            'mathematical_proof', 'philosophical_inquiry', 'strategic_planning',
            'quantum_reasoning', 'consciousness_studies', 'superintelligence'
        ]
        
        for domain in expected_domains:
            assert domain in domains_found
    
    @pytest.mark.asyncio
    async def test_model_selection_for_deliberation(self, council):
        """Test model selection based on request characteristics"""
        request = {
            'type': 'complex_reasoning',
            'complexity': 'high',
            'domain': 'scientific_reasoning'
        }
        
        selected_models = await council._select_models_for_deliberation(request)
        
        assert len(selected_models) >= 5
        assert len(selected_models) <= 12
        
        # Should include models with high reasoning strength for complex tasks
        for model_id in selected_models[:3]:  # Top 3 should be high performers
            capability = council.frontier_models[model_id]
            assert capability.reasoning_strength >= 0.85
    
    @pytest.mark.asyncio
    async def test_debate_role_assignment(self, council):
        """Test that debate roles are assigned correctly"""
        selected_models = ['gpt-5', 'claude-4', 'gemini-ultra', 'palm-3', 
                          'llama-3-400b', 'mistral-large', 'deepseek-coder']
        
        role_assignments = await council._assign_debate_roles(selected_models, {})
        
        # Check that all models are assigned roles
        assert len(role_assignments) == len(selected_models)
        
        # Check that we have the expected role distribution
        roles = list(role_assignments.values())
        assert DebateRole.RED_TEAM in roles
        assert DebateRole.BLUE_TEAM in roles
        
        # Should have at least one moderator or Socratic questioner
        special_roles = [DebateRole.MODERATOR, DebateRole.SOCRATIC_QUESTIONER, DebateRole.JUROR]
        assert any(role in special_roles for role in roles)
    
    @pytest.mark.asyncio
    async def test_deliberation_process(self, council):
        """Test the full deliberation process"""
        request = {
            'id': 'test_request_1',
            'type': 'strategic_decision',
            'complexity': 'medium',
            'domain': 'general',
            'content': 'Should we implement feature X?',
            'start_time': datetime.utcnow().timestamp()
        }
        
        # Mock the sub-engines to avoid complex async operations
        with patch.object(council.adversarial_debate_engine, 'conduct_debate', 
                         new_callable=AsyncMock) as mock_debate, \
             patch.object(council.recursive_argumentation_engine, 'deepen_arguments',
                         new_callable=AsyncMock) as mock_recursive, \
             patch.object(council.socratic_questioning_engine, 'conduct_inquiry',
                         new_callable=AsyncMock) as mock_socratic, \
             patch.object(council.game_theoretic_optimizer, 'find_nash_equilibrium',
                         new_callable=AsyncMock) as mock_game_theory, \
             patch.object(council.swarm_intelligence_coordinator, 'coordinate_emergence',
                         new_callable=AsyncMock) as mock_swarm:
            
            # Set up mock returns
            mock_debate.return_value = {
                'total_rounds': 3,
                'red_team_confidence': 0.8,
                'blue_team_confidence': 0.7,
                'agreement_level': 0.6,
                'debate_reasoning': ['Reason 1', 'Reason 2']
            }
            
            mock_recursive.return_value = {
                'recursive_depth_achieved': 5,
                'reasoning_chains': ['Deep reason 1', 'Deep reason 2']
            }
            
            mock_socratic.return_value = {
                'insights': ['Insight 1', 'Insight 2'],
                'clarity_level': 0.8,
                'socratic_reasoning': ['Socratic reason 1']
            }
            
            mock_game_theory.return_value = {
                'optimal_strategy': 'consensus_building',
                'expected_utility': 0.85
            }
            
            mock_swarm.return_value = {
                'emergent_confidence': 0.9,
                'coherence_level': 0.8,
                'recommendation': 'Implement feature X with modifications',
                'emergent_reasoning': ['Swarm reason 1'],
                'insights': ['Emergent insight 1']
            }
            
            # Execute deliberation
            result = await council.deliberate(request)
            
            # Verify the result structure
            assert 'decision_id' in result
            assert 'timestamp' in result
            assert 'confidence_score' in result
            assert 'reasoning_chain' in result
            assert 'consensus_level' in result
            assert 'final_recommendation' in result
            
            # Verify confidence score is reasonable
            assert 0.0 <= result['confidence_score'] <= 1.0
            assert 0.0 <= result['consensus_level'] <= 1.0
            
            # Verify all engines were called
            mock_debate.assert_called_once()
            mock_recursive.assert_called_once()
            mock_socratic.assert_called_once()
            mock_game_theory.assert_called_once()
            mock_swarm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, council):
        """Test fallback to single model when deliberation fails"""
        request = {
            'id': 'test_request_fallback',
            'type': 'simple_query',
            'start_time': datetime.utcnow().timestamp()
        }
        
        # Mock the debate engine to raise an exception
        with patch.object(council.adversarial_debate_engine, 'conduct_debate',
                         side_effect=Exception("Deliberation failed")):
            
            result = await council.deliberate(request)
            
            # Should return fallback result
            assert result['is_fallback'] is True
            assert 'model_used' in result
            assert result['confidence_score'] == 0.7  # Lower confidence for fallback
            assert 'Fallback decision' in result['final_recommendation']
    
    def test_get_status(self, council):
        """Test status reporting"""
        status = council.get_status()
        
        assert status['status'] == 'active'
        assert status['total_models'] == len(council.frontier_models)
        assert status['deliberations_completed'] == len(council.debate_history)
        assert 'engines_status' in status
        
        engines_status = status['engines_status']
        expected_engines = [
            'adversarial_debate', 'recursive_argumentation', 
            'socratic_questioning', 'game_theoretic_optimizer', 'swarm_intelligence'
        ]
        
        for engine in expected_engines:
            assert engine in engines_status
            assert engines_status[engine] == 'active'


class TestAdversarialDebateEngine:
    """Test the AdversarialDebateEngine"""
    
    @pytest.fixture
    def debate_engine(self):
        """Create an AdversarialDebateEngine for testing"""
        return AdversarialDebateEngine()
    
    @pytest.mark.asyncio
    async def test_conduct_debate(self, debate_engine):
        """Test conducting an adversarial debate"""
        request = {
            'id': 'debate_test',
            'content': 'Should AI systems have rights?'
        }
        
        role_assignments = {
            'gpt-5': DebateRole.RED_TEAM,
            'claude-4': DebateRole.RED_TEAM,
            'gemini-ultra': DebateRole.BLUE_TEAM,
            'palm-3': DebateRole.BLUE_TEAM,
            'llama-3-400b': DebateRole.MODERATOR
        }
        
        result = await debate_engine.conduct_debate(request, role_assignments)
        
        assert 'total_rounds' in result
        assert 'red_team_average_score' in result
        assert 'blue_team_average_score' in result
        assert 'final_convergence' in result
        assert 'agreement_level' in result
        assert 'rounds' in result
        
        # Should have conducted at least one round
        assert result['total_rounds'] >= 1
        assert len(result['rounds']) == result['total_rounds']
    
    @pytest.mark.asyncio
    async def test_team_argument_generation(self, debate_engine):
        """Test generating team arguments"""
        team_models = ['gpt-5', 'claude-4']
        request = {'content': 'Test question'}
        
        argument = await debate_engine._generate_team_argument(
            team_models, request, None, "challenge"
        )
        
        assert argument['stance'] == "challenge"
        assert 'main_points' in argument
        assert 'evidence' in argument
        assert 'reasoning_chain' in argument
        assert 'confidence' in argument
        assert 'individual_contributions' in argument
        
        # Should have contributions from both models
        assert len(argument['individual_contributions']) == len(team_models)
        
        # Confidence should be reasonable
        assert 0.0 <= argument['confidence'] <= 1.0


class TestRecursiveArgumentationEngine:
    """Test the RecursiveArgumentationEngine"""
    
    @pytest.fixture
    def recursive_engine(self):
        """Create a RecursiveArgumentationEngine for testing"""
        return RecursiveArgumentationEngine()
    
    @pytest.mark.asyncio
    async def test_deepen_arguments_infinite(self, recursive_engine):
        """Test recursive argument deepening with infinite depth"""
        debate_results = {
            'rounds': [
                {
                    'red_argument': {'main_points': ['Point 1', 'Point 2'], 'confidence': 0.8},
                    'blue_argument': {'main_points': ['Counter 1', 'Counter 2'], 'confidence': 0.7}
                }
            ]
        }
        
        result = await recursive_engine.deepen_arguments(
            debate_results, ArgumentationDepth.INFINITE
        )
        
        assert 'original_arguments' in result
        assert 'recursive_depth_achieved' in result
        assert 'reasoning_chains' in result
        assert 'philosophical_insights' in result
        
        # Should achieve some depth
        assert result['recursive_depth_achieved'] >= 0
        assert len(result['reasoning_chains']) > 0
    
    @pytest.mark.asyncio
    async def test_deepen_arguments_surface(self, recursive_engine):
        """Test recursive argument deepening with surface depth"""
        debate_results = {
            'rounds': [
                {
                    'red_argument': {'main_points': ['Point 1'], 'confidence': 0.8},
                    'blue_argument': {'main_points': ['Counter 1'], 'confidence': 0.7}
                }
            ]
        }
        
        result = await recursive_engine.deepen_arguments(
            debate_results, ArgumentationDepth.SURFACE
        )
        
        # Surface depth should be limited
        assert result['recursive_depth_achieved'] <= 2


class TestSocraticQuestioningEngine:
    """Test the SocraticQuestioningEngine"""
    
    @pytest.fixture
    def socratic_engine(self):
        """Create a SocraticQuestioningEngine for testing"""
        return SocraticQuestioningEngine()
    
    @pytest.mark.asyncio
    async def test_conduct_inquiry(self, socratic_engine):
        """Test conducting Socratic inquiry"""
        refined_arguments = {
            'reasoning_chains': ['Reason 1', 'Reason 2'],
            'philosophical_insights': ['Insight 1']
        }
        
        request = {'content': 'Philosophical question about consciousness'}
        
        result = await socratic_engine.conduct_inquiry(refined_arguments, request)
        
        assert 'questions_asked' in result
        assert 'insights' in result
        assert 'assumptions_challenged' in result
        assert 'clarity_level' in result
        assert 'socratic_reasoning' in result
        assert 'philosophical_depth' in result
        
        # Should generate questions and insights
        assert len(result['questions_asked']) > 0
        assert len(result['insights']) > 0
        
        # Metrics should be reasonable
        assert 0.0 <= result['clarity_level'] <= 1.0
        assert 0.0 <= result['philosophical_depth'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_socratic_questions(self, socratic_engine):
        """Test Socratic question generation"""
        refined_arguments = {'reasoning_chains': ['Test reasoning']}
        request = {'content': 'Test question'}
        
        questions = await socratic_engine._generate_socratic_questions(
            refined_arguments, request
        )
        
        assert len(questions) > 0
        
        for question in questions:
            assert isinstance(question, SocraticQuestion)
            assert question.question
            assert question.philosophical_domain in socratic_engine.philosophical_domains
            assert question.depth_level > 0
            assert len(question.target_assumptions) > 0


class TestGameTheoreticOptimizer:
    """Test the GameTheoreticOptimizer"""
    
    @pytest.fixture
    def game_optimizer(self):
        """Create a GameTheoreticOptimizer for testing"""
        return GameTheoreticOptimizer()
    
    @pytest.mark.asyncio
    async def test_find_nash_equilibrium(self, game_optimizer):
        """Test finding Nash equilibrium"""
        refined_arguments = {'reasoning_chains': ['Argument 1', 'Argument 2']}
        socratic_insights = {'insights': ['Insight 1', 'Insight 2']}
        
        result = await game_optimizer.find_nash_equilibrium(
            refined_arguments, socratic_insights
        )
        
        assert 'equilibrium_strategy' in result
        assert 'players' in result
        assert 'strategies' in result
        assert 'payoff_matrix' in result
        assert 'convergence_achieved' in result
        assert 'optimal_decision' in result
        assert 'expected_utility' in result
        
        # Should have the expected players
        expected_players = ['red_team', 'blue_team', 'moderator', 'socratic_questioner']
        assert result['players'] == expected_players
        
        # Utility should be reasonable
        assert 0.0 <= result['expected_utility'] <= 10.0  # Depends on payoff scale


class TestSwarmIntelligenceCoordinator:
    """Test the SwarmIntelligenceCoordinator"""
    
    @pytest.fixture
    def swarm_coordinator(self):
        """Create a SwarmIntelligenceCoordinator for testing"""
        return SwarmIntelligenceCoordinator()
    
    @pytest.mark.asyncio
    async def test_coordinate_emergence(self, swarm_coordinator):
        """Test swarm intelligence coordination"""
        optimal_strategy = {
            'optimal_strategy': 'consensus_building',
            'expected_utility': 0.8
        }
        
        selected_models = ['gpt-5', 'claude-4', 'gemini-ultra']
        
        result = await swarm_coordinator.coordinate_emergence(
            optimal_strategy, selected_models
        )
        
        assert 'swarm_agents' in result
        assert 'coordination_cycles' in result
        assert 'emergent_behaviors' in result
        assert 'collective_intelligence_score' in result
        assert 'emergent_confidence' in result
        assert 'coherence_level' in result
        assert 'recommendation' in result
        
        # Should have created swarm agents
        assert result['swarm_agents'] > 0
        
        # Should have run some coordination cycles
        assert result['coordination_cycles'] > 0
        
        # Scores should be reasonable
        assert 0.0 <= result['collective_intelligence_score'] <= 1.0
        assert 0.0 <= result['coherence_level'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_emergent_behavior_detection(self, swarm_coordinator):
        """Test detection of emergent behaviors"""
        emergence_results = {
            'intelligence_score': 0.85,
            'coherence': 0.75,
            'cycles': 12
        }
        
        behaviors = await swarm_coordinator._detect_emergent_behaviors(emergence_results)
        
        assert 'collective_decision_making' in behaviors
        assert 'swarm_consensus' in behaviors
        assert 'distributed_problem_solving' in behaviors
        assert 'adaptive_coordination' in behaviors
        assert 'emergent_intelligence' in behaviors
        assert 'insights' in behaviors
        
        # High intelligence score should trigger emergent intelligence
        assert behaviors['emergent_intelligence'] is True
        
        # High coherence should trigger swarm consensus
        assert behaviors['swarm_consensus'] is True
        
        # Should have insights
        assert len(behaviors['insights']) > 0


@pytest.mark.integration
class TestSuperCouncilIntegration:
    """Integration tests for the complete superintelligent council system"""
    
    @pytest.mark.asyncio
    async def test_full_deliberation_integration(self):
        """Test full integration of all council components"""
        council = SuperCouncilOfModels()
        
        request = {
            'id': 'integration_test_1',
            'type': 'complex_ethical_decision',
            'complexity': 'high',
            'domain': 'ethical_analysis',
            'content': 'Should autonomous AI systems be granted legal personhood?',
            'context': 'This is a complex philosophical and legal question with far-reaching implications.',
            'start_time': datetime.utcnow().timestamp()
        }
        
        # This test runs the actual deliberation process
        result = await council.deliberate(request)
        
        # Verify comprehensive result structure
        required_fields = [
            'decision_id', 'timestamp', 'confidence_score', 'reasoning_chain',
            'supporting_evidence', 'potential_risks', 'alternative_approaches',
            'philosophical_considerations', 'emergent_insights', 'consensus_level',
            'dissenting_opinions', 'final_recommendation'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Verify data quality
        assert result['confidence_score'] > 0.0
        assert result['consensus_level'] >= 0.0
        assert len(result['reasoning_chain']) > 0
        assert len(result['philosophical_considerations']) > 0
        assert len(result['emergent_insights']) > 0
        assert result['final_recommendation']
        
        # Verify deliberation was recorded
        assert len(council.debate_history) > 0
        latest_record = council.debate_history[-1]
        assert latest_record['request_id'] == request['id']
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test council performance with multiple concurrent requests"""
        council = SuperCouncilOfModels()
        
        # Create multiple requests
        requests = []
        for i in range(5):
            requests.append({
                'id': f'load_test_{i}',
                'type': 'decision_making',
                'complexity': 'medium',
                'domain': 'general',
                'content': f'Decision scenario {i}',
                'start_time': datetime.utcnow().timestamp()
            })
        
        # Process requests concurrently
        tasks = [council.deliberate(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed successfully
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Request {i} failed: {result}"
            assert 'decision_id' in result
            assert 'final_recommendation' in result
        
        # Verify all deliberations were recorded
        assert len(council.debate_history) >= len(requests)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])