"""
Reinforcement Learning Optimizer for prompt tuning in the Advanced Prompt Management System.
"""
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from collections import deque
import time

from ..models.optimization_models import PerformanceMetrics, TestCase

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for reinforcement learning optimizer."""
    learning_rate: float = 0.001
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    discount_factor: float = 0.95
    memory_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    max_episodes: int = 1000
    max_steps_per_episode: int = 50
    reward_threshold: float = 0.8
    exploration_bonus: float = 0.01


class PromptState:
    """Represents the state of a prompt for RL optimization."""
    
    def __init__(self, content: str, metrics: Optional[PerformanceMetrics] = None):
        self.content = content
        self.metrics = metrics or PerformanceMetrics()
        self.features = self._extract_features()
    
    def _extract_features(self) -> np.ndarray:
        """Extract numerical features from the prompt."""
        features = []
        
        # Length features
        features.append(len(self.content))
        features.append(len(self.content.split()))
        features.append(len(self.content.split('.')))
        
        # Complexity features
        features.append(self.content.count('?'))
        features.append(self.content.count('!'))
        features.append(self.content.count(','))
        features.append(self.content.count(';'))
        features.append(self.content.count(':'))
        
        # Instruction features
        instruction_words = ['analyze', 'explain', 'describe', 'create', 'solve', 'evaluate']
        features.append(sum(1 for word in instruction_words if word in self.content.lower()))
        
        # Quality indicators
        quality_words = ['carefully', 'thoroughly', 'detailed', 'comprehensive', 'specific']
        features.append(sum(1 for word in quality_words if word in self.content.lower()))
        
        # Structure features
        features.append(1 if 'step by step' in self.content.lower() else 0)
        features.append(1 if 'example' in self.content.lower() else 0)
        features.append(1 if 'context' in self.content.lower() else 0)
        
        # Performance metrics as features
        features.extend([
            self.metrics.accuracy,
            self.metrics.relevance,
            self.metrics.efficiency
        ])
        
        return np.array(features, dtype=np.float32)


class RLOptimizer:
    """Reinforcement Learning optimizer for prompt tuning."""
    
    def __init__(self, config: RLConfig, evaluation_function: Callable[[str, List[TestCase]], PerformanceMetrics]):
        self.config = config
        self.evaluation_function = evaluation_function
        self.best_prompt = None
        self.best_reward = float('-inf')
    
    def optimize(self, base_prompt: str, test_cases: List[TestCase],
                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Run RL optimization (simplified implementation)."""
        start_time = time.time()
        
        # Initialize state
        initial_metrics = self.evaluation_function(base_prompt, test_cases)
        self.best_prompt = base_prompt
        self.best_reward = initial_metrics.get_weighted_score()
        
        # Simple optimization loop (placeholder for full RL implementation)
        for episode in range(min(10, self.config.max_episodes)):  # Reduced for testing
            # Simple random modification for testing
            modified_prompt = self._simple_modify(base_prompt)
            metrics = self.evaluation_function(modified_prompt, test_cases)
            reward = metrics.get_weighted_score()
            
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_prompt = modified_prompt
            
            if progress_callback and episode % 5 == 0:
                progress_callback({
                    "episode": episode,
                    "progress": (episode / 10) * 100,
                    "best_reward": self.best_reward
                })
        
        execution_time = time.time() - start_time
        final_metrics = self.evaluation_function(self.best_prompt, test_cases)
        
        return {
            "best_prompt": self.best_prompt,
            "best_fitness": self.best_reward,
            "objective_scores": final_metrics.to_dict(),
            "episodes_completed": 10,
            "execution_time": execution_time,
            "reward_history": [self.best_reward],
            "final_metrics": final_metrics.to_dict(),
            "total_steps": 10
        }
    
    def _simple_modify(self, prompt: str) -> str:
        """Simple prompt modification for testing."""
        modifications = [
            f"Please {prompt.lower()}",
            f"Carefully {prompt.lower()}",
            f"{prompt} with detailed analysis",
            f"Step by step, {prompt.lower()}"
        ]
        return random.choice(modifications)