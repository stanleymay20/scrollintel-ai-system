"""
ScrollRLAgent - Reinforcement Learning Agent for ScrollIntel

This agent provides reinforcement learning capabilities including Q-Learning, A2C algorithms,
OpenAI Gym integration, and multi-agent RL scenarios.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import logging
from datetime import datetime
import pickle
from collections import deque, namedtuple
import random

from ..core.interfaces import BaseAgent
from ..models.rl_models import RLExperiment, RLModel, RLEnvironment, RLPolicy
from ..core.config import get_config

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class RLConfig:
    """Configuration for RL algorithms"""
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100
    hidden_size: int = 128
    num_episodes: int = 1000
    max_steps_per_episode: int = 500

class DQNNetwork(nn.Module):
    """Deep Q-Network for Q-Learning"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class A2CNetwork(nn.Module):
    """Actor-Critic Network for A2C algorithm"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(A2CNetwork, self).__init__()
        self.shared_fc = nn.Linear(state_size, hidden_size)
        
        # Actor network
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, action_size)
        
        # Critic network
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        shared = F.relu(self.shared_fc(x))
        
        # Actor output (action probabilities)
        actor = F.relu(self.actor_fc(shared))
        action_probs = F.softmax(self.actor_out(actor), dim=-1)
        
        # Critic output (state value)
        critic = F.relu(self.critic_fc(shared))
        state_value = self.critic_out(critic)
        
        return action_probs, state_value

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size: int, action_size: int, config: RLConfig):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size, config.hidden_size)
        self.target_network = DQNNetwork(state_size, action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(config.memory_size)
        
        # Exploration parameters
        self.epsilon = config.epsilon
        
        # Training step counter
        self.step_count = 0
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.config.batch_size:
            return
        
        batch = self.memory.sample(self.config.batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

class A2CAgent:
    """Advantage Actor-Critic Agent"""
    
    def __init__(self, state_size: int, action_size: int, config: RLConfig):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Neural network
        self.network = A2CNetwork(state_size, action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Choose action and return action, log probability, and state value"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.network(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, state_value.squeeze()
    
    def update(self, states, actions, rewards, log_probs, values, next_values):
        """Update the network using A2C algorithm"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        next_values = torch.FloatTensor(next_values)
        
        # Calculate advantages
        returns = []
        R = next_values[-1]
        for reward in reversed(rewards):
            R = reward + self.config.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        advantages = returns - values
        
        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

class MultiAgentEnvironment:
    """Multi-agent environment wrapper"""
    
    def __init__(self, env_name: str, num_agents: int):
        self.env_name = env_name
        self.num_agents = num_agents
        self.agents = []
        
        # Create individual environments for each agent
        self.envs = [gym.make(env_name) for _ in range(num_agents)]
        
    def reset(self):
        """Reset all environments"""
        return [env.reset()[0] for env in self.envs]
    
    def step(self, actions):
        """Step all environments with given actions"""
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            results.append((obs, reward, done, info))
        return results
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()

class ScrollRLAgent(BaseAgent):
    """
    ScrollRLAgent - Advanced Reinforcement Learning Agent
    
    Provides comprehensive RL capabilities including:
    - Q-Learning with Deep Q-Networks (DQN)
    - Advantage Actor-Critic (A2C) algorithms
    - OpenAI Gym integration
    - Multi-agent RL scenarios
    - Policy optimization and reward function design
    """
    
    def __init__(self):
        super().__init__()
        self.agent_type = "scroll_rl"
        self.capabilities = [
            "q_learning",
            "a2c_training",
            "gym_integration",
            "multi_agent_rl",
            "policy_optimization",
            "reward_design",
            "model_evaluation",
            "environment_simulation"
        ]
        self.config = get_config()
        
        # RL-specific attributes
        self.environments = {}
        self.trained_agents = {}
        self.experiments = {}
        
        logger.info("ScrollRLAgent initialized successfully")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process RL-related requests"""
        try:
            task_type = request.get("task_type")
            
            if task_type == "train_dqn":
                return await self._train_dqn_agent(request)
            elif task_type == "train_a2c":
                return await self._train_a2c_agent(request)
            elif task_type == "create_environment":
                return await self._create_environment(request)
            elif task_type == "multi_agent_training":
                return await self._train_multi_agent(request)
            elif task_type == "evaluate_policy":
                return await self._evaluate_policy(request)
            elif task_type == "optimize_rewards":
                return await self._optimize_reward_function(request)
            elif task_type == "get_training_status":
                return await self._get_training_status(request)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}",
                    "supported_tasks": [
                        "train_dqn", "train_a2c", "create_environment",
                        "multi_agent_training", "evaluate_policy", 
                        "optimize_rewards", "get_training_status"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error processing RL request: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": self.agent_type
            }
    
    async def _train_dqn_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Train a DQN agent on specified environment"""
        try:
            env_name = request.get("environment", "CartPole-v1")
            config_params = request.get("config", {})
            experiment_name = request.get("experiment_name", f"dqn_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Create environment
            env = gym.make(env_name)
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            
            # Create configuration
            config = RLConfig(**config_params)
            
            # Create DQN agent
            agent = DQNAgent(state_size, action_size, config)
            
            # Training loop
            scores = []
            training_log = []
            
            for episode in range(config.num_episodes):
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                
                for step in range(config.max_steps_per_episode):
                    action = agent.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                    
                    # Train the agent
                    if len(agent.memory) > config.batch_size:
                        agent.replay()
                
                scores.append(total_reward)
                
                # Log progress
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:])
                    log_entry = {
                        "episode": episode,
                        "score": total_reward,
                        "avg_score": avg_score,
                        "epsilon": agent.epsilon,
                        "steps": steps
                    }
                    training_log.append(log_entry)
                    logger.info(f"Episode {episode}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            env.close()
            
            # Save trained agent
            self.trained_agents[experiment_name] = {
                "agent": agent,
                "algorithm": "DQN",
                "environment": env_name,
                "config": config,
                "training_scores": scores,
                "training_log": training_log
            }
            
            # Create experiment record
            experiment = RLExperiment(
                name=experiment_name,
                algorithm="DQN",
                environment=env_name,
                config=config_params,
                final_score=float(np.mean(scores[-100:])),
                training_episodes=config.num_episodes,
                created_at=datetime.now()
            )
            
            self.experiments[experiment_name] = experiment
            
            return {
                "success": True,
                "experiment_name": experiment_name,
                "algorithm": "DQN",
                "environment": env_name,
                "final_avg_score": float(np.mean(scores[-100:])),
                "total_episodes": config.num_episodes,
                "training_log": training_log[-10:],  # Last 10 entries
                "model_saved": True
            }
            
        except Exception as e:
            logger.error(f"Error training DQN agent: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _train_a2c_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Train an A2C agent on specified environment"""
        try:
            env_name = request.get("environment", "CartPole-v1")
            config_params = request.get("config", {})
            experiment_name = request.get("experiment_name", f"a2c_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Create environment
            env = gym.make(env_name)
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            
            # Create configuration
            config = RLConfig(**config_params)
            
            # Create A2C agent
            agent = A2CAgent(state_size, action_size, config)
            
            # Training loop
            scores = []
            training_log = []
            
            for episode in range(config.num_episodes):
                state, _ = env.reset()
                
                states, actions, rewards, log_probs, values = [], [], [], [], []
                total_reward = 0
                steps = 0
                
                for step in range(config.max_steps_per_episode):
                    action, log_prob, value = agent.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                # Calculate next state value for bootstrapping
                if done:
                    next_value = 0
                else:
                    _, _, next_value = agent.act(state)
                    next_value = next_value.item()
                
                next_values = [next_value]
                
                # Update agent
                if len(states) > 0:
                    actor_loss, critic_loss = agent.update(
                        states, actions, rewards, log_probs, values, next_values
                    )
                
                scores.append(total_reward)
                
                # Log progress
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:])
                    log_entry = {
                        "episode": episode,
                        "score": total_reward,
                        "avg_score": avg_score,
                        "actor_loss": actor_loss if 'actor_loss' in locals() else 0,
                        "critic_loss": critic_loss if 'critic_loss' in locals() else 0,
                        "steps": steps
                    }
                    training_log.append(log_entry)
                    logger.info(f"Episode {episode}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}")
            
            env.close()
            
            # Save trained agent
            self.trained_agents[experiment_name] = {
                "agent": agent,
                "algorithm": "A2C",
                "environment": env_name,
                "config": config,
                "training_scores": scores,
                "training_log": training_log
            }
            
            # Create experiment record
            experiment = RLExperiment(
                name=experiment_name,
                algorithm="A2C",
                environment=env_name,
                config=config_params,
                final_score=float(np.mean(scores[-100:])),
                training_episodes=config.num_episodes,
                created_at=datetime.now()
            )
            
            self.experiments[experiment_name] = experiment
            
            return {
                "success": True,
                "experiment_name": experiment_name,
                "algorithm": "A2C",
                "environment": env_name,
                "final_avg_score": float(np.mean(scores[-100:])),
                "total_episodes": config.num_episodes,
                "training_log": training_log[-10:],  # Last 10 entries
                "model_saved": True
            }
            
        except Exception as e:
            logger.error(f"Error training A2C agent: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _create_environment(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create and register a custom environment"""
        try:
            env_name = request.get("environment_name")
            env_type = request.get("environment_type", "gym")
            
            if env_type == "gym":
                # Create standard Gym environment
                env = gym.make(env_name)
                
                env_info = {
                    "name": env_name,
                    "type": "gym",
                    "observation_space": str(env.observation_space),
                    "action_space": str(env.action_space),
                    "state_size": env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None,
                    "action_size": env.action_space.n if hasattr(env.action_space, 'n') else None
                }
                
                self.environments[env_name] = env_info
                env.close()
                
                return {
                    "success": True,
                    "environment_name": env_name,
                    "environment_info": env_info
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported environment type: {env_type}"
                }
                
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _train_multi_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Train multiple agents in cooperative or competitive scenarios"""
        try:
            env_name = request.get("environment", "CartPole-v1")
            num_agents = request.get("num_agents", 2)
            scenario_type = request.get("scenario_type", "cooperative")  # cooperative or competitive
            config_params = request.get("config", {})
            experiment_name = request.get("experiment_name", f"multi_agent_{scenario_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Create multi-agent environment
            multi_env = MultiAgentEnvironment(env_name, num_agents)
            
            # Get environment info
            sample_env = gym.make(env_name)
            state_size = sample_env.observation_space.shape[0]
            action_size = sample_env.action_space.n
            sample_env.close()
            
            # Create configuration
            config = RLConfig(**config_params)
            
            # Create agents
            agents = []
            for i in range(num_agents):
                if request.get("algorithm", "DQN") == "DQN":
                    agent = DQNAgent(state_size, action_size, config)
                else:
                    agent = A2CAgent(state_size, action_size, config)
                agents.append(agent)
            
            # Training loop
            all_scores = [[] for _ in range(num_agents)]
            training_log = []
            
            for episode in range(config.num_episodes):
                states = multi_env.reset()
                total_rewards = [0] * num_agents
                steps = 0
                
                for step in range(config.max_steps_per_episode):
                    # Get actions from all agents
                    actions = []
                    for i, agent in enumerate(agents):
                        if isinstance(agent, DQNAgent):
                            action = agent.act(states[i])
                        else:  # A2C
                            action, _, _ = agent.act(states[i])
                        actions.append(action)
                    
                    # Step environment
                    results = multi_env.step(actions)
                    next_states, rewards, dones, _ = zip(*results)
                    
                    # Modify rewards based on scenario type
                    if scenario_type == "cooperative":
                        # Shared reward - all agents get average reward
                        shared_reward = np.mean(rewards)
                        rewards = [shared_reward] * num_agents
                    elif scenario_type == "competitive":
                        # Zero-sum rewards - one agent's gain is another's loss
                        total_reward = sum(rewards)
                        rewards = [r - total_reward/num_agents for r in rewards]
                    
                    # Store experiences and update agents
                    for i, agent in enumerate(agents):
                        if isinstance(agent, DQNAgent):
                            agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                            if len(agent.memory) > config.batch_size:
                                agent.replay()
                        
                        total_rewards[i] += rewards[i]
                    
                    states = next_states
                    steps += 1
                    
                    if any(dones):
                        break
                
                # Record scores
                for i in range(num_agents):
                    all_scores[i].append(total_rewards[i])
                
                # Log progress
                if episode % 100 == 0:
                    avg_scores = [np.mean(scores[-100:]) for scores in all_scores]
                    log_entry = {
                        "episode": episode,
                        "agent_scores": total_rewards,
                        "avg_scores": avg_scores,
                        "steps": steps,
                        "scenario_type": scenario_type
                    }
                    training_log.append(log_entry)
                    logger.info(f"Episode {episode}, Avg Scores: {[f'{s:.2f}' for s in avg_scores]}")
            
            multi_env.close()
            
            # Save trained agents
            self.trained_agents[experiment_name] = {
                "agents": agents,
                "algorithm": request.get("algorithm", "DQN"),
                "environment": env_name,
                "scenario_type": scenario_type,
                "num_agents": num_agents,
                "config": config,
                "training_scores": all_scores,
                "training_log": training_log
            }
            
            return {
                "success": True,
                "experiment_name": experiment_name,
                "scenario_type": scenario_type,
                "num_agents": num_agents,
                "final_avg_scores": [float(np.mean(scores[-100:])) for scores in all_scores],
                "total_episodes": config.num_episodes,
                "training_log": training_log[-5:],  # Last 5 entries
                "models_saved": True
            }
            
        except Exception as e:
            logger.error(f"Error training multi-agent system: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _evaluate_policy(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a trained policy"""
        try:
            experiment_name = request.get("experiment_name")
            num_episodes = request.get("num_episodes", 100)
            render = request.get("render", False)
            
            if experiment_name not in self.trained_agents:
                return {
                    "success": False,
                    "error": f"Experiment '{experiment_name}' not found",
                    "available_experiments": list(self.trained_agents.keys())
                }
            
            experiment = self.trained_agents[experiment_name]
            env_name = experiment["environment"]
            
            # Create environment
            env = gym.make(env_name, render_mode="human" if render else None)
            
            # Evaluate policy
            scores = []
            episode_lengths = []
            
            if "agents" in experiment:  # Multi-agent
                agents = experiment["agents"]
                multi_env = MultiAgentEnvironment(env_name, len(agents))
                
                for episode in range(num_episodes):
                    states = multi_env.reset()
                    total_rewards = [0] * len(agents)
                    steps = 0
                    
                    for step in range(1000):  # Max steps per episode
                        actions = []
                        for i, agent in enumerate(agents):
                            if isinstance(agent, DQNAgent):
                                action = agent.act(states[i], training=False)
                            else:  # A2C
                                action, _, _ = agent.act(states[i])
                            actions.append(action)
                        
                        results = multi_env.step(actions)
                        next_states, rewards, dones, _ = zip(*results)
                        
                        for i in range(len(agents)):
                            total_rewards[i] += rewards[i]
                        
                        states = next_states
                        steps += 1
                        
                        if any(dones):
                            break
                    
                    scores.append(total_rewards)
                    episode_lengths.append(steps)
                
                multi_env.close()
                
                # Calculate statistics
                avg_scores = [np.mean([episode[i] for episode in scores]) for i in range(len(agents))]
                std_scores = [np.std([episode[i] for episode in scores]) for i in range(len(agents))]
                
                return {
                    "success": True,
                    "experiment_name": experiment_name,
                    "evaluation_episodes": num_episodes,
                    "agent_performance": {
                        f"agent_{i}": {
                            "avg_score": float(avg_scores[i]),
                            "std_score": float(std_scores[i]),
                            "min_score": float(min([episode[i] for episode in scores])),
                            "max_score": float(max([episode[i] for episode in scores]))
                        }
                        for i in range(len(agents))
                    },
                    "avg_episode_length": float(np.mean(episode_lengths)),
                    "total_agents": len(agents)
                }
                
            else:  # Single agent
                agent = experiment["agent"]
                
                for episode in range(num_episodes):
                    state, _ = env.reset()
                    total_reward = 0
                    steps = 0
                    
                    for step in range(1000):  # Max steps per episode
                        if isinstance(agent, DQNAgent):
                            action = agent.act(state, training=False)
                        else:  # A2C
                            action, _, _ = agent.act(state)
                        
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        state = next_state
                        total_reward += reward
                        steps += 1
                        
                        if done:
                            break
                    
                    scores.append(total_reward)
                    episode_lengths.append(steps)
                
                env.close()
                
                return {
                    "success": True,
                    "experiment_name": experiment_name,
                    "evaluation_episodes": num_episodes,
                    "avg_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "min_score": float(min(scores)),
                    "max_score": float(max(scores)),
                    "avg_episode_length": float(np.mean(episode_lengths)),
                    "success_rate": float(sum(1 for s in scores if s > 0) / len(scores))
                }
            
        except Exception as e:
            logger.error(f"Error evaluating policy: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _optimize_reward_function(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize reward function design"""
        try:
            env_name = request.get("environment")
            reward_components = request.get("reward_components", {})
            optimization_method = request.get("method", "grid_search")
            
            # Define reward function optimization
            def custom_reward_function(state, action, next_state, base_reward, components):
                """Custom reward function with multiple components"""
                total_reward = base_reward
                
                # Add custom reward components
                if "survival_bonus" in components:
                    total_reward += components["survival_bonus"]
                
                if "progress_reward" in components and len(state) > 0:
                    # Reward for making progress (e.g., moving right in CartPole)
                    if len(state) >= 1:
                        total_reward += components["progress_reward"] * abs(state[0])
                
                if "stability_reward" in components and len(state) > 1:
                    # Reward for stability (e.g., keeping pole upright)
                    if len(state) >= 2:
                        total_reward += components["stability_reward"] * (1 - abs(state[1]))
                
                return total_reward
            
            # Test different reward configurations
            if optimization_method == "grid_search":
                # Define parameter grid
                param_grid = {
                    "survival_bonus": [0.1, 0.5, 1.0],
                    "progress_reward": [0.0, 0.1, 0.2],
                    "stability_reward": [0.0, 0.5, 1.0]
                }
                
                best_config = None
                best_score = float('-inf')
                results = []
                
                # Grid search over parameters
                import itertools
                param_combinations = list(itertools.product(*param_grid.values()))
                
                for params in param_combinations[:9]:  # Limit to 9 combinations for demo
                    config = dict(zip(param_grid.keys(), params))
                    
                    # Quick training with this reward configuration
                    env = gym.make(env_name)
                    state_size = env.observation_space.shape[0]
                    action_size = env.action_space.n
                    
                    # Train for fewer episodes for optimization
                    quick_config = RLConfig(num_episodes=100, epsilon_decay=0.99)
                    agent = DQNAgent(state_size, action_size, quick_config)
                    
                    scores = []
                    for episode in range(quick_config.num_episodes):
                        state, _ = env.reset()
                        total_reward = 0
                        
                        for step in range(200):  # Shorter episodes
                            action = agent.act(state)
                            next_state, base_reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                            
                            # Apply custom reward function
                            custom_reward = custom_reward_function(
                                state, action, next_state, base_reward, config
                            )
                            
                            agent.remember(state, action, custom_reward, next_state, done)
                            state = next_state
                            total_reward += custom_reward
                            
                            if done:
                                break
                            
                            if len(agent.memory) > quick_config.batch_size:
                                agent.replay()
                        
                        scores.append(total_reward)
                    
                    avg_score = np.mean(scores[-20:])  # Average of last 20 episodes
                    
                    result = {
                        "config": config,
                        "avg_score": float(avg_score),
                        "final_scores": scores[-10:]
                    }
                    results.append(result)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_config = config
                    
                    env.close()
                
                return {
                    "success": True,
                    "optimization_method": optimization_method,
                    "best_config": best_config,
                    "best_score": float(best_score),
                    "all_results": results,
                    "total_configurations_tested": len(results)
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported optimization method: {optimization_method}",
                    "supported_methods": ["grid_search"]
                }
                
        except Exception as e:
            logger.error(f"Error optimizing reward function: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_training_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of training experiments"""
        try:
            experiment_name = request.get("experiment_name")
            
            if experiment_name:
                # Get specific experiment status
                if experiment_name not in self.trained_agents:
                    return {
                        "success": False,
                        "error": f"Experiment '{experiment_name}' not found"
                    }
                
                experiment = self.trained_agents[experiment_name]
                
                return {
                    "success": True,
                    "experiment_name": experiment_name,
                    "algorithm": experiment["algorithm"],
                    "environment": experiment["environment"],
                    "status": "completed",
                    "training_episodes": len(experiment["training_scores"]),
                    "final_performance": float(np.mean(experiment["training_scores"][-100:])) if experiment["training_scores"] else 0,
                    "training_log": experiment["training_log"][-5:] if experiment["training_log"] else []
                }
            
            else:
                # Get all experiments status
                experiments_status = []
                
                for name, experiment in self.trained_agents.items():
                    status = {
                        "experiment_name": name,
                        "algorithm": experiment["algorithm"],
                        "environment": experiment["environment"],
                        "status": "completed",
                        "training_episodes": len(experiment["training_scores"]),
                        "final_performance": float(np.mean(experiment["training_scores"][-100:])) if experiment["training_scores"] else 0
                    }
                    
                    if "num_agents" in experiment:
                        status["num_agents"] = experiment["num_agents"]
                        status["scenario_type"] = experiment["scenario_type"]
                    
                    experiments_status.append(status)
                
                return {
                    "success": True,
                    "total_experiments": len(experiments_status),
                    "experiments": experiments_status,
                    "available_environments": list(self.environments.keys())
                }
                
        except Exception as e:
            logger.error(f"Error getting training status: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return self.capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status"""
        return {
            "agent_type": self.agent_type,
            "status": "active",
            "capabilities": self.capabilities,
            "trained_models": len(self.trained_agents),
            "environments": len(self.environments),
            "experiments": len(self.experiments)
        }