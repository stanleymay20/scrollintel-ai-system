"""
Tests for ScrollRLAgent - Reinforcement Learning Agent
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
import torch
import gymnasium as gym

from scrollintel.agents.scroll_rl_agent import (
    ScrollRLAgent, DQNAgent, A2CAgent, ReplayBuffer, 
    DQNNetwork, A2CNetwork, RLConfig, MultiAgentEnvironment
)

class TestScrollRLAgent:
    """Test cases for ScrollRLAgent"""
    
    @pytest.fixture
    def rl_agent(self):
        """Create ScrollRLAgent instance for testing"""
        return ScrollRLAgent()
    
    @pytest.fixture
    def sample_config(self):
        """Sample RL configuration"""
        return {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "batch_size": 32,
            "num_episodes": 10,  # Small number for testing
            "max_steps_per_episode": 50
        }
    
    def test_agent_initialization(self, rl_agent):
        """Test agent initialization"""
        assert rl_agent.agent_type == "scroll_rl"
        assert "q_learning" in rl_agent.capabilities
        assert "a2c_training" in rl_agent.capabilities
        assert "multi_agent_rl" in rl_agent.capabilities
        assert isinstance(rl_agent.environments, dict)
        assert isinstance(rl_agent.trained_agents, dict)
    
    def test_get_capabilities(self, rl_agent):
        """Test get_capabilities method"""
        capabilities = rl_agent.get_capabilities()
        assert isinstance(capabilities, list)
        assert "q_learning" in capabilities
        assert "a2c_training" in capabilities
        assert "gym_integration" in capabilities
    
    def test_get_status(self, rl_agent):
        """Test get_status method"""
        status = rl_agent.get_status()
        assert status["agent_type"] == "scroll_rl"
        assert status["status"] == "active"
        assert "capabilities" in status
        assert "trained_models" in status

class TestDQNComponents:
    """Test DQN-related components"""
    
    def test_dqn_network(self):
        """Test DQN network creation and forward pass"""
        state_size = 4
        action_size = 2
        hidden_size = 64
        
        network = DQNNetwork(state_size, action_size, hidden_size)
        
        # Test forward pass
        sample_input = torch.randn(1, state_size)
        output = network(sample_input)
        
        assert output.shape == (1, action_size)
        assert not torch.isnan(output).any()
    
    def test_replay_buffer(self):
        """Test replay buffer functionality"""
        buffer = ReplayBuffer(capacity=100)
        
        # Test empty buffer
        assert len(buffer) == 0
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = np.random.choice([True, False])
            
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 50
        
        # Test sampling
        batch = buffer.sample(10)
        assert len(batch) == 10
        
        # Test capacity limit
        for i in range(60):
            buffer.push(np.random.randn(4), 0, 0, np.random.randn(4), False)
        
        assert len(buffer) == 100  # Should not exceed capacity
    
    def test_dqn_agent_creation(self):
        """Test DQN agent creation"""
        config = RLConfig(num_episodes=10, batch_size=16)
        agent = DQNAgent(state_size=4, action_size=2, config=config)
        
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.epsilon == config.epsilon
        assert isinstance(agent.q_network, DQNNetwork)
        assert isinstance(agent.target_network, DQNNetwork)
        assert isinstance(agent.memory, ReplayBuffer)
    
    def test_dqn_agent_action_selection(self):
        """Test DQN agent action selection"""
        config = RLConfig(epsilon=0.0)  # No exploration
        agent = DQNAgent(state_size=4, action_size=2, config=config)
        
        state = np.random.randn(4)
        
        # Test deterministic action (no exploration)
        action = agent.act(state, training=False)
        assert isinstance(action, int)
        assert 0 <= action < 2
        
        # Test with exploration
        agent.epsilon = 1.0  # Full exploration
        action = agent.act(state, training=True)
        assert isinstance(action, int)
        assert 0 <= action < 2

class TestA2CComponents:
    """Test A2C-related components"""
    
    def test_a2c_network(self):
        """Test A2C network creation and forward pass"""
        state_size = 4
        action_size = 2
        hidden_size = 64
        
        network = A2CNetwork(state_size, action_size, hidden_size)
        
        # Test forward pass
        sample_input = torch.randn(1, state_size)
        action_probs, state_value = network(sample_input)
        
        assert action_probs.shape == (1, action_size)
        assert state_value.shape == (1, 1)
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(1))  # Probabilities sum to 1
        assert not torch.isnan(action_probs).any()
        assert not torch.isnan(state_value).any()
    
    def test_a2c_agent_creation(self):
        """Test A2C agent creation"""
        config = RLConfig(num_episodes=10)
        agent = A2CAgent(state_size=4, action_size=2, config=config)
        
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert isinstance(agent.network, A2CNetwork)
    
    def test_a2c_agent_action_selection(self):
        """Test A2C agent action selection"""
        config = RLConfig()
        agent = A2CAgent(state_size=4, action_size=2, config=config)
        
        state = np.random.randn(4)
        action, log_prob, value = agent.act(state)
        
        assert isinstance(action, int)
        assert 0 <= action < 2
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)

class TestMultiAgentEnvironment:
    """Test multi-agent environment wrapper"""
    
    @patch('gymnasium.make')
    def test_multi_agent_environment_creation(self, mock_gym_make):
        """Test multi-agent environment creation"""
        # Mock gym environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_env.step.return_value = (np.array([0, 0, 0, 0]), 1.0, False, False, {})
        mock_gym_make.return_value = mock_env
        
        multi_env = MultiAgentEnvironment("CartPole-v1", num_agents=2)
        
        assert multi_env.num_agents == 2
        assert len(multi_env.envs) == 2
        assert mock_gym_make.call_count == 2
    
    @patch('gymnasium.make')
    def test_multi_agent_environment_reset(self, mock_gym_make):
        """Test multi-agent environment reset"""
        # Mock gym environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_gym_make.return_value = mock_env
        
        multi_env = MultiAgentEnvironment("CartPole-v1", num_agents=2)
        states = multi_env.reset()
        
        assert len(states) == 2
        assert mock_env.reset.call_count == 2
    
    @patch('gymnasium.make')
    def test_multi_agent_environment_step(self, mock_gym_make):
        """Test multi-agent environment step"""
        # Mock gym environment
        mock_env = Mock()
        mock_env.step.return_value = (np.array([0, 0, 0, 0]), 1.0, False, False, {})
        mock_gym_make.return_value = mock_env
        
        multi_env = MultiAgentEnvironment("CartPole-v1", num_agents=2)
        actions = [0, 1]
        results = multi_env.step(actions)
        
        assert len(results) == 2
        assert mock_env.step.call_count == 2

class TestScrollRLAgentRequests:
    """Test ScrollRLAgent request processing"""
    
    @pytest.fixture
    def rl_agent(self):
        """Create ScrollRLAgent instance for testing"""
        return ScrollRLAgent()
    
    @pytest.mark.asyncio
    async def test_unknown_task_type(self, rl_agent):
        """Test handling of unknown task type"""
        request = {"task_type": "unknown_task"}
        result = await rl_agent.process_request(request)
        
        assert result["success"] is False
        assert "Unknown task type" in result["error"]
        assert "supported_tasks" in result
    
    @pytest.mark.asyncio
    @patch('gymnasium.make')
    async def test_create_environment_request(self, mock_gym_make, rl_agent):
        """Test create environment request"""
        # Mock gym environment
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space = Mock()
        mock_env.action_space.n = 2
        mock_env.close = Mock()
        mock_gym_make.return_value = mock_env
        
        request = {
            "task_type": "create_environment",
            "environment_name": "CartPole-v1",
            "environment_type": "gym"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["environment_name"] == "CartPole-v1"
        assert "environment_info" in result
        assert mock_env.close.called
    
    @pytest.mark.asyncio
    async def test_get_training_status_empty(self, rl_agent):
        """Test get training status with no experiments"""
        request = {"task_type": "get_training_status"}
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["total_experiments"] == 0
        assert result["experiments"] == []
    
    @pytest.mark.asyncio
    @patch('gymnasium.make')
    async def test_dqn_training_request_mock(self, mock_gym_make, rl_agent, sample_config):
        """Test DQN training request with mocked environment"""
        # Mock gym environment
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space = Mock()
        mock_env.action_space.n = 2
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_env.step.return_value = (np.array([0, 0, 0, 0]), 1.0, True, False, {})
        mock_env.close = Mock()
        mock_gym_make.return_value = mock_env
        
        request = {
            "task_type": "train_dqn",
            "environment": "CartPole-v1",
            "config": sample_config,
            "experiment_name": "test_dqn"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["experiment_name"] == "test_dqn"
        assert result["algorithm"] == "DQN"
        assert "final_avg_score" in result
        assert mock_env.close.called
    
    @pytest.mark.asyncio
    @patch('gymnasium.make')
    async def test_a2c_training_request_mock(self, mock_gym_make, rl_agent, sample_config):
        """Test A2C training request with mocked environment"""
        # Mock gym environment
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space = Mock()
        mock_env.action_space.n = 2
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_env.step.return_value = (np.array([0, 0, 0, 0]), 1.0, True, False, {})
        mock_env.close = Mock()
        mock_gym_make.return_value = mock_env
        
        request = {
            "task_type": "train_a2c",
            "environment": "CartPole-v1",
            "config": sample_config,
            "experiment_name": "test_a2c"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["experiment_name"] == "test_a2c"
        assert result["algorithm"] == "A2C"
        assert "final_avg_score" in result
        assert mock_env.close.called
    
    @pytest.mark.asyncio
    @patch('scrollintel.agents.scroll_rl_agent.MultiAgentEnvironment')
    @patch('gymnasium.make')
    async def test_multi_agent_training_request(self, mock_gym_make, mock_multi_env_class, rl_agent, sample_config):
        """Test multi-agent training request"""
        # Mock gym environment for getting info
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space = Mock()
        mock_env.action_space.n = 2
        mock_env.close = Mock()
        mock_gym_make.return_value = mock_env
        
        # Mock multi-agent environment
        mock_multi_env = Mock()
        mock_multi_env.reset.return_value = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
        mock_multi_env.step.return_value = [
            (np.array([0, 0, 0, 0]), 1.0, True, {}),
            (np.array([0, 0, 0, 0]), 1.0, True, {})
        ]
        mock_multi_env.close = Mock()
        mock_multi_env_class.return_value = mock_multi_env
        
        request = {
            "task_type": "multi_agent_training",
            "environment": "CartPole-v1",
            "num_agents": 2,
            "scenario_type": "cooperative",
            "algorithm": "DQN",
            "config": sample_config,
            "experiment_name": "test_multi_agent"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["experiment_name"] == "test_multi_agent"
        assert result["scenario_type"] == "cooperative"
        assert result["num_agents"] == 2
        assert "final_avg_scores" in result
        assert mock_multi_env.close.called
    
    @pytest.mark.asyncio
    async def test_evaluate_policy_not_found(self, rl_agent):
        """Test policy evaluation with non-existent experiment"""
        request = {
            "task_type": "evaluate_policy",
            "experiment_name": "non_existent",
            "num_episodes": 10
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is False
        assert "not found" in result["error"]
        assert "available_experiments" in result
    
    @pytest.mark.asyncio
    @patch('gymnasium.make')
    async def test_reward_optimization_request(self, mock_gym_make, rl_agent):
        """Test reward optimization request"""
        # Mock gym environment
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space = Mock()
        mock_env.action_space.n = 2
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_env.step.return_value = (np.array([0, 0, 0, 0]), 1.0, True, False, {})
        mock_env.close = Mock()
        mock_gym_make.return_value = mock_env
        
        request = {
            "task_type": "optimize_rewards",
            "environment": "CartPole-v1",
            "reward_components": {
                "survival_bonus": 0.1,
                "progress_reward": 0.1,
                "stability_reward": 0.5
            },
            "method": "grid_search"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["optimization_method"] == "grid_search"
        assert "best_config" in result
        assert "best_score" in result
        assert "all_results" in result

class TestRLConfig:
    """Test RL configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RLConfig()
        
        assert config.learning_rate == 0.001
        assert config.gamma == 0.99
        assert config.epsilon == 1.0
        assert config.epsilon_min == 0.01
        assert config.batch_size == 32
        assert config.num_episodes == 1000
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = RLConfig(
            learning_rate=0.01,
            gamma=0.95,
            epsilon=0.5,
            batch_size=64,
            num_episodes=500
        )
        
        assert config.learning_rate == 0.01
        assert config.gamma == 0.95
        assert config.epsilon == 0.5
        assert config.batch_size == 64
        assert config.num_episodes == 500

if __name__ == "__main__":
    pytest.main([__file__])