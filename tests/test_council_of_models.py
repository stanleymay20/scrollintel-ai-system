"""
Tests for ScrollIntel G6 - Superintelligent Council of Models with Adversarial Collaboration
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from scrollintel.core.council_of_models import (
    SuperCouncilOfModels,
    AdversarialDebateEngine,
    RecursiveArgumentationEngine,
    SocraticQuestioningEngine,
    GameTheoreticOptimizer,
    SwarmIntelligenceCoordinator,
    DebateRole,
    ArgumentationDepth,
    ModelCapability,
    DebateArgument,
    SocraticQuestion
)


class TestSuperCouncilOfModels:
    """Test the main SuperCouncilOfModels class"""
    
    @pytest.fixture
    def council(self):
        return SuperCouncilOfModels()
    
    @pytest.fixture
    def sample_request(self):
        return {
            'id': 'test_request_123',
            'type': 'strategic_decision',
            'complexity': 'high',
            'domain': 'general',
            'content': 'Should we implement quantum computing for our AI infrastructure?',
            'start_time': 1234567890
        }
    
    def test_initialization(self, council):
        """Test council initialization"""
        assert council is not None
        assert len(council.frontier_models) >= 10  # Should have 50+ models
        assert 'gpt-5' in council.frontier_models
        assert 'claude-4' in council.frontier_models
        assert 'gemini-ultra' in council.frontier_models
        
        # Check that engines are initialized
        assert council.adversarial_debate_engine is not None
        assert council.recursive_argumentation_engine is not None
        assert council.socratic_questioning_engine is not None
        assert council.game_theoretic_optimizer is not None
        assert council.swarm_intelligence_coordinator is not None
    
    def test_model_capabilities(self, council):
        """Test that model capabilities are properly defined"""
        gpt5_capability = council.frontier_models['gpt-5']
        
        assert isinstance(gpt5_capability, ModelCapability)
        assert gpt5_capability.model_name == "GPT-5"
        assert 0.0 <= gpt5_capability.reasoning_strength <= 1.0
        assert 0.0 <= gpt5_capability.creativity_score <= 1.0
        assert 0.0 <= gpt5_capability.factual_accuracy <= 1.0
        assert 0.0 <= gpt5_capability.philosophical_depth <= 1.0
        assert 0.0 <= gpt5_capability.adversarial_robustness <= 1.0
        assert len(gpt5_capability.specializations) > 0
    
    @pytest.mark.asyncio
    async def test_select_models_for_deliberation(self, council, sample_request):
        """Test model selection for deliberation"""
        selected_models = await council._select_models_for_deliberation(sample_request)
        
        assert isinstance(selected_models, list)
        assert 5 <= len(selected_models) <= 12  # Optimal council size
        assert all(model_id in council.frontier_models for model_id in selected_models)
    
    @pytest.mark.asyncio
    async def test_assign_debate_roles(self, council):
        """Test debate role assignment"""
        selected_models = ['gpt-5', 'claude-4', 'gemini-ultra', 'palm-3', 'llama-3-400b', 'mistral-large']
        sample_request = {'type': 'general', 'complexity': 'medium'}
        
        role_assignments = await council._assign_debate_roles(selected_models, sample_request)
        
        assert isinstance(role_assignments, dict)
        assert len(role_assignments) == len(selected_models)
        
        # Check that all roles are valid
        for model_id, role in role_assignments.items():
            assert model_id in selected_models
            assert isinstance(role, DebateRole)
        
        # Check that we have red and blue teams
        roles = list(role_assignments.values())
        assert DebateRole.RED_TEAM in roles
        assert DebateRole.BLUE_TEAM in roles
    
    @pytest.mark.asyncio
    async def test_deliberate_full_process(self, council, sample_request):
        """Test the full deliberation process"""
        # Mock the sub-engines to avoid complex async operations
        with patch.object(council.adversarial_debate_engine, 'conduct_debate', new_callable=AsyncMock) as mock_debate, \
             patch.object(council.recursive_argumentation_engine, 'deepen_arguments', new_callable=AsyncMock) as mock_recursive, \
             patch.object(council.socratic_questioning_engine, 'conduct_inquiry', new_callable=AsyncMock) as mock_socratic, \
             patch.object(council.game_theoretic_optimizer, 'find_nash_equilibrium', new_callable=AsyncMock) as mock_game_theory, \
             patch.object(council.swarm_intelligence_coordinator, 'coordinate_emergence', new_callable=AsyncMock) as mock_swarm:
            
            # Set up mock returns
            mock_debate.return_value = {
                'total_rounds': 3,
                'overall_winner': 'blue',
                'debate_quality': 0.8,
                'debate_reasoning': ['Round 1: Analysis', 'Round 2: Counter-analysis']
            }
            
            mock_recursive.return_value = {
                'refined_arguments': [{'source': 'test', 'content': ['refined point']}],
                'depth_achieved': 5
            }
            
            mock_socratic.return_value = {
                'insights': ['Deep insight 1', 'Deep insight 2'],
                'socratic_reasoning': ['Socratic analysis step']
            }
            
            mock_game_theory.return_value = {
                'optimal_strategy': {'strategy_1': 0.7, 'strategy_2': 0.3},
                'expected_payoff': 0.75
            }
            
            mock_swarm.return_value = {
                'emergent_solution': {
                    'recommendation': 'Implement quantum computing with phased approach',
                    'confidence': 0.85,
                    'insights': ['Emergent insight'],
                    'reasoning_chain': ['Swarm reasoning'],
                    'coherence_level': 0.9
                },
                'collective_intelligence_score': 0.88
            }
            
            # Execute deliberation
            result = await council.deliberate(sample_request)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'decision_id' in result
            assert 'timestamp' in result
            assert 'confidence_score' in result
            assert 'reasoning_chain' in result
            assert 'final_recommendation' in result
            assert 'consensus_level' in result
            
            # Verify that all engines were called
            mock_debate.assert_called_once()
            mock_recursive.assert_called_once()
            mock_socratic.assert_called_once()
            mock_game_theory.assert_called_once()
            mock_swarm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_single_model_decision(self, council, sample_request):
        """Test fallback to single model decision"""
        result = await council._fallback_single_model_decision(sample_request)
        
        assert isinstance(result, dict)
        assert result['is_fallback'] is True
        assert 'model_used' in result
        assert 'confidence_score' in result
        assert result['confidence_score'] < 0.8  # Lower confidence for fallback


class TestAdversarialDebateEngine:
    """Test the AdversarialDebateEngine class"""
    
    @pytest.fixture
    def debate_engine(self):
        return AdversarialDebateEngine()
    
    @pytest.fixture
    def sample_role_assignments(self):
        return {
            'gpt-5': DebateRole.RED_TEAM,
            'claude-4': DebateRole.BLUE_TEAM,
            'gemini-ultra': DebateRole.MODERATOR,
            'palm-3': DebateRole.JUROR
        }
    
    @pytest.fixture
    def sample_request(self):
        return {
            'id': 'debate_test',
            'content': 'Should AI systems have consciousness?'
        }
    
    def test_initialization(self, debate_engine):
        """Test debate engine initialization"""
        assert debate_engine.max_debate_rounds == 5
        assert debate_engine.convergence_threshold == 0.1
    
    @pytest.mark.asyncio
    async def test_conduct_debate(self, debate_engine, sample_request, sample_role_assignments):
        """Test conducting adversarial debate"""
        with patch.object(debate_engine, '_generate_team_argument', new_callable=AsyncMock) as mock_generate, \
             patch.object(debate_engine, '_evaluate_debate_round', new_callable=AsyncMock) as mock_evaluate:
            
            # Mock team argument generation
            mock_generate.return_value = {
                'stance': 'test_stance',
                'main_points': ['Point 1', 'Point 2'],
                'confidence': 0.8,
                'evidence': ['Evidence 1'],
                'reasoning_chain': ['Reasoning step']
            }
            
            # Mock round evaluation
            mock_evaluate.return_value = {
                'winner': 'blue',
                'convergence_score': 0.15,  # Above threshold to stop early
                'quality_score': 0.8,
                'reasoning': 'Blue team provided stronger arguments'
            }
            
            result = await debate_engine.conduct_debate(sample_request, sample_role_assignments)
            
            assert isinstance(result, dict)
            assert 'total_rounds' in result
            assert 'overall_winner' in result
            assert 'debate_quality' in result
            assert 'rounds' in result
            
            # Should stop early due to convergence
            assert result['total_rounds'] <= debate_engine.max_debate_rounds
    
    @pytest.mark.asyncio
    async def test_generate_team_argument(self, debate_engine, sample_request):
        """Test team argument generation"""
        team_models = ['gpt-5', 'claude-4']
        
        with patch.object(debate_engine, '_get_model_contribution', new_callable=AsyncMock) as mock_contribution:
            mock_contribution.return_value = {
                'model_id': 'test_model',
                'points': ['Test point'],
                'evidence': ['Test evidence'],
                'reasoning': ['Test reasoning'],
                'confidence': 0.8
            }
            
            result = await debate_engine._generate_team_argument(
                team_models, sample_request, None, 'challenge'
            )
            
            assert isinstance(result, dict)
            assert result['stance'] == 'challenge'
            assert 'main_points' in result
            assert 'evidence' in result
            assert 'reasoning_chain' in result
            assert 'confidence' in result
            assert 0.0 <= result['confidence'] <= 1.0


class TestRecursiveArgumentationEngine:
    """Test the RecursiveArgumentationEngine class"""
    
    @pytest.fixture
    def recursive_engine(self):
        return RecursiveArgumentationEngine()
    
    @pytest.fixture
    def sample_debate_results(self):
        return {
            'rounds': [
                {
                    'round': 1,
                    'red_argument': {
                        'main_points': ['Red point 1', 'Red point 2'],
                        'evidence': ['Red evidence'],
                        'confidence': 0.7,
                        'reasoning_chain': ['Red reasoning']
                    },
                    'blue_argument': {
                        'main_points': ['Blue point 1'],
                        'evidence': ['Blue evidence'],
                        'confidence': 0.8,
                        'reasoning_chain': ['Blue reasoning']
                    }
                }
            ]
        }
    
    def test_initialization(self, recursive_engine):
        """Test recursive argumentation engine initialization"""
        assert recursive_engine.max_recursion_depth == 10
        assert recursive_engine.min_improvement_threshold == 0.05
    
    @pytest.mark.asyncio
    async def test_deepen_arguments(self, recursive_engine, sample_debate_results):
        """Test argument deepening process"""
        result = await recursive_engine.deepen_arguments(
            sample_debate_results, 
            ArgumentationDepth.DEEP
        )
        
        assert isinstance(result, dict)
        assert 'original_arguments' in result
        assert 'refined_arguments' in result
        assert 'depth_achieved' in result
        assert 'refinement_quality' in result
        
        # Check that arguments were extracted and refined
        assert len(result['original_arguments']) > 0
        assert len(result['refined_arguments']) == len(result['original_arguments'])
        assert result['depth_achieved'] == 5  # DEEP level
    
    @pytest.mark.asyncio
    async def test_recursive_refinement(self, recursive_engine):
        """Test recursive refinement of individual arguments"""
        argument = {
            'source': 'test',
            'content': ['Initial point'],
            'evidence': ['Initial evidence'],
            'confidence': 0.6,
            'reasoning_chain': ['Initial reasoning']
        }
        
        result = await recursive_engine._recursive_refinement(argument, 3, 0)
        
        assert isinstance(result, dict)
        assert 'reasoning_chain' in result
        assert 'addressed_counterarguments' in result
        assert 'counter_responses' in result
        assert 'refinement_depth' in result
        
        # Should have deeper reasoning than original
        assert len(result['reasoning_chain']) > len(argument['reasoning_chain'])


class TestSocraticQuestioningEngine:
    """Test the SocraticQuestioningEngine class"""
    
    @pytest.fixture
    def socratic_engine(self):
        return SocraticQuestioningEngine()
    
    @pytest.fixture
    def sample_refined_arguments(self):
        return {
            'refined_arguments': [
                {
                    'source': 'red_team',
                    'content': ['Consciousness requires subjective experience'],
                    'confidence': 0.8
                },
                {
                    'source': 'blue_team',
                    'content': ['Consciousness is emergent from complexity'],
                    'confidence': 0.7
                }
            ]
        }
    
    @pytest.fixture
    def sample_request(self):
        return {
            'id': 'socratic_test',
            'content': 'What is the nature of consciousness?'
        }
    
    def test_initialization(self, socratic_engine):
        """Test Socratic questioning engine initialization"""
        assert len(socratic_engine.philosophical_domains) >= 8
        assert 'epistemology' in socratic_engine.philosophical_domains
        assert 'ethics' in socratic_engine.philosophical_domains
    
    @pytest.mark.asyncio
    async def test_conduct_inquiry(self, socratic_engine, sample_refined_arguments, sample_request):
        """Test conducting Socratic inquiry"""
        result = await socratic_engine.conduct_inquiry(sample_refined_arguments, sample_request)
        
        assert isinstance(result, dict)
        assert 'socratic_sessions' in result
        assert 'synthesized_insights' in result
        assert 'philosophical_depth_achieved' in result
        assert 'key_assumptions_revealed' in result
        assert 'insights' in result
        
        # Should have sessions for each argument
        assert len(result['socratic_sessions']) == len(sample_refined_arguments['refined_arguments'])
    
    @pytest.mark.asyncio
    async def test_generate_question_sequence(self, socratic_engine, sample_request):
        """Test generation of Socratic question sequence"""
        argument = {'content': ['Test argument'], 'confidence': 0.8}
        
        questions = await socratic_engine._generate_question_sequence(argument, sample_request)
        
        assert isinstance(questions, list)
        assert len(questions) > 0
        
        for question in questions:
            assert isinstance(question, SocraticQuestion)
            assert question.question is not None
            assert question.philosophical_domain in socratic_engine.philosophical_domains
            assert question.depth_level > 0


class TestGameTheoreticOptimizer:
    """Test the GameTheoreticOptimizer class"""
    
    @pytest.fixture
    def game_optimizer(self):
        return GameTheoreticOptimizer()
    
    @pytest.fixture
    def sample_refined_arguments(self):
        return {
            'refined_arguments': [
                {
                    'source': 'red_team',
                    'content': ['Option A is optimal'],
                    'confidence': 0.8,
                    'evidence': ['Evidence A']
                },
                {
                    'source': 'blue_team',
                    'content': ['Option B is better'],
                    'confidence': 0.7,
                    'evidence': ['Evidence B']
                }
            ]
        }
    
    @pytest.fixture
    def sample_socratic_insights(self):
        return {
            'insights': ['Insight 1', 'Insight 2'],
            'assumptions_revealed': ['Assumption 1']
        }
    
    def test_initialization(self, game_optimizer):
        """Test game-theoretic optimizer initialization"""
        assert game_optimizer.convergence_tolerance == 1e-6
        assert game_optimizer.max_iterations == 1000
    
    @pytest.mark.asyncio
    async def test_find_nash_equilibrium(self, game_optimizer, sample_refined_arguments, sample_socratic_insights):
        """Test Nash equilibrium finding"""
        result = await game_optimizer.find_nash_equilibrium(
            sample_refined_arguments, 
            sample_socratic_insights
        )
        
        assert isinstance(result, dict)
        assert 'decision_options' in result
        assert 'payoff_matrix' in result
        assert 'nash_equilibrium' in result
        assert 'stability_analysis' in result
        assert 'optimal_strategy' in result
        
        # Check Nash equilibrium structure
        nash_eq = result['nash_equilibrium']
        assert 'strategy' in nash_eq
        assert 'expected_payoff' in nash_eq
        assert 'converged' in nash_eq
        
        # Strategy should sum to 1 (probability distribution)
        strategy_values = list(nash_eq['strategy'].values())
        assert abs(sum(strategy_values) - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_build_payoff_matrix(self, game_optimizer):
        """Test payoff matrix construction"""
        decision_options = [
            {'id': 'option_1', 'confidence': 0.8, 'risk_level': 'low'},
            {'id': 'option_2', 'confidence': 0.6, 'risk_level': 'high'}
        ]
        
        payoff_matrix = await game_optimizer._build_payoff_matrix(decision_options)
        
        assert isinstance(payoff_matrix, np.ndarray)
        assert payoff_matrix.shape == (2, 2)
        assert np.all(payoff_matrix >= 0.0)
        assert np.all(payoff_matrix <= 1.0)


class TestSwarmIntelligenceCoordinator:
    """Test the SwarmIntelligenceCoordinator class"""
    
    @pytest.fixture
    def swarm_coordinator(self):
        return SwarmIntelligenceCoordinator()
    
    @pytest.fixture
    def sample_optimal_strategy(self):
        return {
            'strategy': {'option_1': 0.7, 'option_2': 0.3},
            'expected_payoff': 0.8
        }
    
    @pytest.fixture
    def sample_selected_models(self):
        return ['gpt-5', 'claude-4', 'gemini-ultra']
    
    def test_initialization(self, swarm_coordinator):
        """Test swarm intelligence coordinator initialization"""
        assert swarm_coordinator.swarm_size == 50
        assert swarm_coordinator.emergence_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_coordinate_emergence(self, swarm_coordinator, sample_optimal_strategy, sample_selected_models):
        """Test swarm intelligence coordination"""
        result = await swarm_coordinator.coordinate_emergence(
            sample_optimal_strategy, 
            sample_selected_models
        )
        
        assert isinstance(result, dict)
        assert 'swarm_agents' in result
        assert 'coordination_cycles' in result
        assert 'emergent_behaviors' in result
        assert 'emergent_solution' in result
        assert 'collective_intelligence_score' in result
        
        # Check that swarm was properly sized
        assert result['swarm_agents'] > 0
        assert result['coordination_cycles'] > 0
        
        # Check emergent solution structure
        emergent_solution = result['emergent_solution']
        assert 'insights' in emergent_solution
        assert 'recommendation' in emergent_solution
        assert 'confidence' in emergent_solution
    
    @pytest.mark.asyncio
    async def test_initialize_swarm_agents(self, swarm_coordinator, sample_selected_models, sample_optimal_strategy):
        """Test swarm agent initialization"""
        agents = await swarm_coordinator._initialize_swarm_agents(
            sample_selected_models, 
            sample_optimal_strategy
        )
        
        assert isinstance(agents, list)
        assert len(agents) > 0
        
        for agent in agents:
            assert 'id' in agent
            assert 'base_model' in agent
            assert 'position' in agent
            assert 'velocity' in agent
            assert 'fitness' in agent
            
            # Check that position and velocity are proper arrays
            assert len(agent['position']) == 5
            assert len(agent['velocity']) == 5
            assert agent['base_model'] in sample_selected_models
    
    @pytest.mark.asyncio
    async def test_detect_emergent_behaviors(self, swarm_coordinator):
        """Test emergent behavior detection"""
        coordination_cycles = [
            {
                'cycle': 0,
                'emergent_patterns': ['clustering_behavior', 'velocity_alignment'],
                'collective_fitness': 0.6
            },
            {
                'cycle': 1,
                'emergent_patterns': ['clustering_behavior', 'fitness_convergence'],
                'collective_fitness': 0.7
            },
            {
                'cycle': 2,
                'emergent_patterns': ['velocity_alignment', 'fitness_convergence'],
                'collective_fitness': 0.8
            }
        ]
        
        emergent_behaviors = await swarm_coordinator._detect_emergent_behaviors(coordination_cycles)
        
        assert isinstance(emergent_behaviors, list)
        
        # Should detect patterns that appear frequently
        behavior_types = [behavior['behavior_type'] for behavior in emergent_behaviors]
        
        # Should include collective learning due to fitness improvement
        assert 'collective_learning' in behavior_types or len(emergent_behaviors) > 0


@pytest.mark.asyncio
async def test_integration_council_deliberation():
    """Integration test for full council deliberation process"""
    council = SuperCouncilOfModels()
    
    request = {
        'id': 'integration_test',
        'type': 'strategic_decision',
        'complexity': 'high',
        'domain': 'technology',
        'content': 'Should we adopt quantum computing for our AI infrastructure?',
        'start_time': 1234567890
    }
    
    # This is a simplified integration test - in practice, you'd want to mock
    # the actual model API calls to avoid external dependencies
    with patch.object(council, '_get_model_contribution', new_callable=AsyncMock) as mock_contrib:
        mock_contrib.return_value = {
            'model_id': 'test_model',
            'points': ['Quantum computing offers exponential speedup'],
            'evidence': ['Quantum algorithms show theoretical advantages'],
            'reasoning': ['Quantum superposition enables parallel computation'],
            'confidence': 0.85
        }
        
        result = await council.deliberate(request)
        
        # Verify the result has all expected components
        assert isinstance(result, dict)
        assert 'decision_id' in result
        assert 'confidence_score' in result
        assert 'final_recommendation' in result
        assert 'consensus_level' in result
        
        # Should have reasonable confidence and consensus
        assert 0.0 <= result['confidence_score'] <= 1.0
        assert 0.0 <= result['consensus_level'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])