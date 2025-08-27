"""
Integration tests for ScrollRLAgent with real environments
"""

import pytest
import asyncio
import numpy as np
import gymnasium as gym
from unittest.mock import patch

from scrollintel.agents.scroll_rl_agent import ScrollRLAgent

class TestScrollRLIntegration:
    """Integration tests for ScrollRLAgent"""
    
    @pytest.fixture
    def rl_agent(self):
        """Create ScrollRLAgent instance for testing"""
        return ScrollRLAgent()
    
    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for fast testing"""
        return {
            "learning_rate": 0.01,
            "gamma": 0.99,
            "epsilon": 0.5,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.99,
            "batch_size": 16,
            "num_episodes": 5,  # Very small for testing
            "max_steps_per_episode": 20
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_dqn_training(self, rl_agent, minimal_config):
        """Test complete DQN training workflow"""
        request = {
            "task_type": "train_dqn",
            "environment": "CartPole-v1",
            "config": minimal_config,
            "experiment_name": "integration_test_dqn"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["experiment_name"] == "integration_test_dqn"
        assert result["algorithm"] == "DQN"
        assert result["environment"] == "CartPole-v1"
        assert isinstance(result["final_avg_score"], float)
        assert result["total_episodes"] == minimal_config["num_episodes"]
        assert result["model_saved"] is True
        
        # Verify experiment is stored
        assert "integration_test_dqn" in rl_agent.trained_agents
        
        # Verify experiment data
        experiment = rl_agent.trained_agents["integration_test_dqn"]
        assert experiment["algorithm"] == "DQN"
        assert experiment["environment"] == "CartPole-v1"
        assert len(experiment["training_scores"]) == minimal_config["num_episodes"]
    
    @pytest.mark.asyncio
    async def test_end_to_end_a2c_training(self, rl_agent, minimal_config):
        """Test complete A2C training workflow"""
        request = {
            "task_type": "train_a2c",
            "environment": "CartPole-v1",
            "config": minimal_config,
            "experiment_name": "integration_test_a2c"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["experiment_name"] == "integration_test_a2c"
        assert result["algorithm"] == "A2C"
        assert result["environment"] == "CartPole-v1"
        assert isinstance(result["final_avg_score"], float)
        assert result["total_episodes"] == minimal_config["num_episodes"]
        assert result["model_saved"] is True
        
        # Verify experiment is stored
        assert "integration_test_a2c" in rl_agent.trained_agents
    
    @pytest.mark.asyncio
    async def test_policy_evaluation_workflow(self, rl_agent, minimal_config):
        """Test training followed by policy evaluation"""
        # First train an agent
        train_request = {
            "task_type": "train_dqn",
            "environment": "CartPole-v1",
            "config": minimal_config,
            "experiment_name": "eval_test_dqn"
        }
        
        train_result = await rl_agent.process_request(train_request)
        assert train_result["success"] is True
        
        # Then evaluate the policy
        eval_request = {
            "task_type": "evaluate_policy",
            "experiment_name": "eval_test_dqn",
            "num_episodes": 5,
            "render": False
        }
        
        eval_result = await rl_agent.process_request(eval_request)
        
        assert eval_result["success"] is True
        assert eval_result["experiment_name"] == "eval_test_dqn"
        assert eval_result["evaluation_episodes"] == 5
        assert isinstance(eval_result["avg_score"], float)
        assert isinstance(eval_result["std_score"], float)
        assert isinstance(eval_result["min_score"], float)
        assert isinstance(eval_result["max_score"], float)
        assert isinstance(eval_result["avg_episode_length"], float)
        assert isinstance(eval_result["success_rate"], float)
    
    @pytest.mark.asyncio
    async def test_multi_agent_training_workflow(self, rl_agent, minimal_config):
        """Test multi-agent training workflow"""
        request = {
            "task_type": "multi_agent_training",
            "environment": "CartPole-v1",
            "num_agents": 2,
            "scenario_type": "cooperative",
            "algorithm": "DQN",
            "config": minimal_config,
            "experiment_name": "integration_test_multi"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["experiment_name"] == "integration_test_multi"
        assert result["scenario_type"] == "cooperative"
        assert result["num_agents"] == 2
        assert len(result["final_avg_scores"]) == 2
        assert result["total_episodes"] == minimal_config["num_episodes"]
        assert result["models_saved"] is True
        
        # Verify experiment is stored
        assert "integration_test_multi" in rl_agent.trained_agents
        
        # Verify multi-agent specific data
        experiment = rl_agent.trained_agents["integration_test_multi"]
        assert experiment["num_agents"] == 2
        assert experiment["scenario_type"] == "cooperative"
        assert len(experiment["training_scores"]) == 2  # One list per agent
    
    @pytest.mark.asyncio
    async def test_multi_agent_evaluation_workflow(self, rl_agent, minimal_config):
        """Test multi-agent training followed by evaluation"""
        # First train multi-agent system
        train_request = {
            "task_type": "multi_agent_training",
            "environment": "CartPole-v1",
            "num_agents": 2,
            "scenario_type": "competitive",
            "algorithm": "A2C",
            "config": minimal_config,
            "experiment_name": "multi_eval_test"
        }
        
        train_result = await rl_agent.process_request(train_request)
        assert train_result["success"] is True
        
        # Then evaluate the multi-agent policy
        eval_request = {
            "task_type": "evaluate_policy",
            "experiment_name": "multi_eval_test",
            "num_episodes": 3,
            "render": False
        }
        
        eval_result = await rl_agent.process_request(eval_request)
        
        assert eval_result["success"] is True
        assert eval_result["experiment_name"] == "multi_eval_test"
        assert eval_result["evaluation_episodes"] == 3
        assert eval_result["total_agents"] == 2
        assert "agent_performance" in eval_result
        
        # Check individual agent performance
        agent_perf = eval_result["agent_performance"]
        assert "agent_0" in agent_perf
        assert "agent_1" in agent_perf
        
        for agent_id in ["agent_0", "agent_1"]:
            perf = agent_perf[agent_id]
            assert isinstance(perf["avg_score"], float)
            assert isinstance(perf["std_score"], float)
            assert isinstance(perf["min_score"], float)
            assert isinstance(perf["max_score"], float)
    
    @pytest.mark.asyncio
    async def test_reward_optimization_workflow(self, rl_agent):
        """Test reward function optimization workflow"""
        request = {
            "task_type": "optimize_rewards",
            "environment": "CartPole-v1",
            "reward_components": {
                "survival_bonus": 0.1,
                "progress_reward": 0.1
            },
            "method": "grid_search"
        }
        
        result = await rl_agent.process_request(request)
        
        assert result["success"] is True
        assert result["optimization_method"] == "grid_search"
        assert "best_config" in result
        assert "best_score" in result
        assert "all_results" in result
        assert isinstance(result["total_configurations_tested"], int)
        assert result["total_configurations_tested"] > 0
        
        # Verify results structure
        assert isinstance(result["best_config"], dict)
        assert isinstance(result["best_score"], float)
        assert isinstance(result["all_results"], list)
        
        # Check individual results
        for res in result["all_results"]:
            assert "config" in res
            assert "avg_score" in res
            assert isinstance(res["avg_score"], float)
    
    @pytest.mark.asyncio
    async def test_training_status_workflow(self, rl_agent, minimal_config):
        """Test training status tracking workflow"""
        # Initially no experiments
        status_request = {"task_type": "get_training_status"}
        result = await rl_agent.process_request(status_request)
        
        assert result["success"] is True
        assert result["total_experiments"] == 0
        assert result["experiments"] == []
        
        # Train some agents
        for i, algorithm in enumerate(["DQN", "A2C"]):
            train_request = {
                "task_type": f"train_{algorithm.lower()}",
                "environment": "CartPole-v1",
                "config": minimal_config,
                "experiment_name": f"status_test_{algorithm.lower()}_{i}"
            }
            
            train_result = await rl_agent.process_request(train_request)
            assert train_result["success"] is True
        
        # Check status after training
        result = await rl_agent.process_request(status_request)
        
        assert result["success"] is True
        assert result["total_experiments"] == 2
        assert len(result["experiments"]) == 2
        
        # Verify experiment details
        experiments = result["experiments"]
        algorithms = [exp["algorithm"] for exp in experiments]
        assert "DQN" in algorithms
        assert "A2C" in algorithms
        
        for exp in experiments:
            assert "experiment_name" in exp
            assert "algorithm" in exp
            assert "environment" in exp
            assert "status" in exp
            assert "training_episodes" in exp
            assert "final_performance" in exp
    
    @pytest.mark.asyncio
    async def test_environment_creation_workflow(self, rl_agent):
        """Test environment creation and usage workflow"""
        # Create environment
        env_request = {
            "task_type": "create_environment",
            "environment_name": "CartPole-v1",
            "environment_type": "gym"
        }
        
        result = await rl_agent.process_request(env_request)
        
        assert result["success"] is True
        assert result["environment_name"] == "CartPole-v1"
        assert "environment_info" in result
        
        env_info = result["environment_info"]
        assert env_info["name"] == "CartPole-v1"
        assert env_info["type"] == "gym"
        assert "observation_space" in env_info
        assert "action_space" in env_info
        assert "state_size" in env_info
        assert "action_size" in env_info
        
        # Verify environment is registered
        assert "CartPole-v1" in rl_agent.environments
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, rl_agent):
        """Test error handling in various scenarios"""
        # Test invalid environment
        invalid_env_request = {
            "task_type": "train_dqn",
            "environment": "NonExistentEnv-v1",
            "config": {"num_episodes": 1}
        }
        
        result = await rl_agent.process_request(invalid_env_request)
        assert result["success"] is False
        assert "error" in result
        
        # Test evaluation of non-existent experiment
        invalid_eval_request = {
            "task_type": "evaluate_policy",
            "experiment_name": "non_existent_experiment",
            "num_episodes": 5
        }
        
        result = await rl_agent.process_request(invalid_eval_request)
        assert result["success"] is False
        assert "not found" in result["error"]
        assert "available_experiments" in result
        
        # Test unsupported optimization method
        invalid_opt_request = {
            "task_type": "optimize_rewards",
            "environment": "CartPole-v1",
            "reward_components": {"survival_bonus": 0.1},
            "method": "unsupported_method"
        }
        
        result = await rl_agent.process_request(invalid_opt_request)
        assert result["success"] is False
        assert "Unsupported optimization method" in result["error"]
        assert "supported_methods" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_training_workflow(self, rl_agent, minimal_config):
        """Test handling of concurrent training requests"""
        # Create multiple training requests
        requests = []
        for i in range(3):
            request = {
                "task_type": "train_dqn",
                "environment": "CartPole-v1",
                "config": minimal_config,
                "experiment_name": f"concurrent_test_{i}"
            }
            requests.append(request)
        
        # Execute requests concurrently
        tasks = [rl_agent.process_request(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        for i, result in enumerate(results):
            assert result["success"] is True
            assert result["experiment_name"] == f"concurrent_test_{i}"
            assert f"concurrent_test_{i}" in rl_agent.trained_agents
        
        # Verify final status
        status_request = {"task_type": "get_training_status"}
        status_result = await rl_agent.process_request(status_request)
        
        assert status_result["success"] is True
        assert status_result["total_experiments"] >= 3

if __name__ == "__main__":
    pytest.main([__file__])