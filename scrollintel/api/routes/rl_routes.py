"""
RL Routes - API endpoints for reinforcement learning functionality
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging

from ...agents.scroll_rl_agent import ScrollRLAgent
from ...core.interfaces import get_agent
from ...security.auth import get_current_user
from ...models.rl_models import RLExperiment, RLModel, RLEnvironment, RLPolicy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rl", tags=["reinforcement_learning"])

# Pydantic models for request/response
class RLTrainingRequest(BaseModel):
    """Request model for RL training"""
    algorithm: str = Field(..., description="RL algorithm (DQN, A2C)")
    environment: str = Field(..., description="Environment name")
    experiment_name: Optional[str] = Field(None, description="Custom experiment name")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")

class MultiAgentTrainingRequest(BaseModel):
    """Request model for multi-agent RL training"""
    algorithm: str = Field(..., description="RL algorithm (DQN, A2C)")
    environment: str = Field(..., description="Environment name")
    num_agents: int = Field(2, description="Number of agents")
    scenario_type: str = Field("cooperative", description="Scenario type (cooperative, competitive)")
    experiment_name: Optional[str] = Field(None, description="Custom experiment name")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")

class PolicyEvaluationRequest(BaseModel):
    """Request model for policy evaluation"""
    experiment_name: str = Field(..., description="Experiment name to evaluate")
    num_episodes: int = Field(100, description="Number of evaluation episodes")
    render: bool = Field(False, description="Whether to render environment")

class RewardOptimizationRequest(BaseModel):
    """Request model for reward function optimization"""
    environment: str = Field(..., description="Environment name")
    reward_components: Dict[str, float] = Field(..., description="Reward components to optimize")
    method: str = Field("grid_search", description="Optimization method")

class EnvironmentCreationRequest(BaseModel):
    """Request model for environment creation"""
    environment_name: str = Field(..., description="Environment name")
    environment_type: str = Field("gym", description="Environment type")

class RLTrainingResponse(BaseModel):
    """Response model for RL training"""
    success: bool
    experiment_name: Optional[str] = None
    algorithm: Optional[str] = None
    environment: Optional[str] = None
    final_avg_score: Optional[float] = None
    total_episodes: Optional[int] = None
    training_log: Optional[List[Dict[str, Any]]] = None
    model_saved: Optional[bool] = None
    error: Optional[str] = None

class MultiAgentTrainingResponse(BaseModel):
    """Response model for multi-agent training"""
    success: bool
    experiment_name: Optional[str] = None
    scenario_type: Optional[str] = None
    num_agents: Optional[int] = None
    final_avg_scores: Optional[List[float]] = None
    total_episodes: Optional[int] = None
    training_log: Optional[List[Dict[str, Any]]] = None
    models_saved: Optional[bool] = None
    error: Optional[str] = None

class PolicyEvaluationResponse(BaseModel):
    """Response model for policy evaluation"""
    success: bool
    experiment_name: Optional[str] = None
    evaluation_episodes: Optional[int] = None
    avg_score: Optional[float] = None
    std_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    avg_episode_length: Optional[float] = None
    success_rate: Optional[float] = None
    agent_performance: Optional[Dict[str, Any]] = None
    total_agents: Optional[int] = None
    error: Optional[str] = None

class RewardOptimizationResponse(BaseModel):
    """Response model for reward optimization"""
    success: bool
    optimization_method: Optional[str] = None
    best_config: Optional[Dict[str, float]] = None
    best_score: Optional[float] = None
    all_results: Optional[List[Dict[str, Any]]] = None
    total_configurations_tested: Optional[int] = None
    error: Optional[str] = None

class TrainingStatusResponse(BaseModel):
    """Response model for training status"""
    success: bool
    total_experiments: Optional[int] = None
    experiments: Optional[List[Dict[str, Any]]] = None
    available_environments: Optional[List[str]] = None
    experiment_name: Optional[str] = None
    algorithm: Optional[str] = None
    environment: Optional[str] = None
    status: Optional[str] = None
    training_episodes: Optional[int] = None
    final_performance: Optional[float] = None
    training_log: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

# Helper function to get RL agent
async def get_rl_agent() -> ScrollRLAgent:
    """Get ScrollRLAgent instance"""
    agent = await get_agent("scroll_rl")
    if not isinstance(agent, ScrollRLAgent):
        raise HTTPException(status_code=500, detail="ScrollRLAgent not available")
    return agent

@router.post("/train/dqn", response_model=RLTrainingResponse)
async def train_dqn_agent(
    request: RLTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Train a DQN (Deep Q-Network) agent
    
    This endpoint trains a DQN agent on the specified environment with custom configuration.
    Training runs in the background and results are stored for later retrieval.
    """
    try:
        agent = await get_rl_agent()
        
        # Prepare request for agent
        agent_request = {
            "task_type": "train_dqn",
            "algorithm": request.algorithm,
            "environment": request.environment,
            "experiment_name": request.experiment_name,
            "config": request.config,
            "user_id": current_user.get("user_id")
        }
        
        # Process request
        result = await agent.process_request(agent_request)
        
        if result["success"]:
            logger.info(f"DQN training started for user {current_user.get('user_id')}: {result.get('experiment_name')}")
        
        return RLTrainingResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in DQN training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train/a2c", response_model=RLTrainingResponse)
async def train_a2c_agent(
    request: RLTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Train an A2C (Advantage Actor-Critic) agent
    
    This endpoint trains an A2C agent on the specified environment with custom configuration.
    A2C is often more stable than DQN for continuous control tasks.
    """
    try:
        agent = await get_rl_agent()
        
        # Prepare request for agent
        agent_request = {
            "task_type": "train_a2c",
            "algorithm": request.algorithm,
            "environment": request.environment,
            "experiment_name": request.experiment_name,
            "config": request.config,
            "user_id": current_user.get("user_id")
        }
        
        # Process request
        result = await agent.process_request(agent_request)
        
        if result["success"]:
            logger.info(f"A2C training started for user {current_user.get('user_id')}: {result.get('experiment_name')}")
        
        return RLTrainingResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in A2C training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train/multi-agent", response_model=MultiAgentTrainingResponse)
async def train_multi_agent_system(
    request: MultiAgentTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Train a multi-agent RL system
    
    This endpoint trains multiple agents in cooperative or competitive scenarios.
    Supports different interaction patterns and reward structures.
    """
    try:
        agent = await get_rl_agent()
        
        # Prepare request for agent
        agent_request = {
            "task_type": "multi_agent_training",
            "algorithm": request.algorithm,
            "environment": request.environment,
            "num_agents": request.num_agents,
            "scenario_type": request.scenario_type,
            "experiment_name": request.experiment_name,
            "config": request.config,
            "user_id": current_user.get("user_id")
        }
        
        # Process request
        result = await agent.process_request(agent_request)
        
        if result["success"]:
            logger.info(f"Multi-agent training started for user {current_user.get('user_id')}: {result.get('experiment_name')}")
        
        return MultiAgentTrainingResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in multi-agent training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=PolicyEvaluationResponse)
async def evaluate_policy(
    request: PolicyEvaluationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Evaluate a trained RL policy
    
    This endpoint evaluates the performance of a trained agent by running
    multiple episodes and collecting performance statistics.
    """
    try:
        agent = await get_rl_agent()
        
        # Prepare request for agent
        agent_request = {
            "task_type": "evaluate_policy",
            "experiment_name": request.experiment_name,
            "num_episodes": request.num_episodes,
            "render": request.render,
            "user_id": current_user.get("user_id")
        }
        
        # Process request
        result = await agent.process_request(agent_request)
        
        if result["success"]:
            logger.info(f"Policy evaluation completed for user {current_user.get('user_id')}: {request.experiment_name}")
        
        return PolicyEvaluationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in policy evaluation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-rewards", response_model=RewardOptimizationResponse)
async def optimize_reward_function(
    request: RewardOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Optimize reward function design
    
    This endpoint optimizes reward function parameters to improve agent performance.
    Uses various optimization methods to find the best reward configuration.
    """
    try:
        agent = await get_rl_agent()
        
        # Prepare request for agent
        agent_request = {
            "task_type": "optimize_rewards",
            "environment": request.environment,
            "reward_components": request.reward_components,
            "method": request.method,
            "user_id": current_user.get("user_id")
        }
        
        # Process request
        result = await agent.process_request(agent_request)
        
        if result["success"]:
            logger.info(f"Reward optimization completed for user {current_user.get('user_id')}: {request.environment}")
        
        return RewardOptimizationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in reward optimization endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/environments", response_model=Dict[str, Any])
async def create_environment(
    request: EnvironmentCreationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create and register a new RL environment
    
    This endpoint creates a new environment for RL training and registers
    it with the system for future use.
    """
    try:
        agent = await get_rl_agent()
        
        # Prepare request for agent
        agent_request = {
            "task_type": "create_environment",
            "environment_name": request.environment_name,
            "environment_type": request.environment_type,
            "user_id": current_user.get("user_id")
        }
        
        # Process request
        result = await agent.process_request(agent_request)
        
        if result["success"]:
            logger.info(f"Environment created for user {current_user.get('user_id')}: {request.environment_name}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in environment creation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status(
    experiment_name: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get training status for experiments
    
    This endpoint returns the status of RL training experiments.
    Can return status for a specific experiment or all experiments.
    """
    try:
        agent = await get_rl_agent()
        
        # Prepare request for agent
        agent_request = {
            "task_type": "get_training_status",
            "experiment_name": experiment_name,
            "user_id": current_user.get("user_id")
        }
        
        # Process request
        result = await agent.process_request(agent_request)
        
        return TrainingStatusResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in training status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments(
    algorithm: Optional[str] = None,
    environment: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    List all RL experiments
    
    This endpoint returns a list of all RL experiments with optional filtering
    by algorithm or environment.
    """
    try:
        agent = await get_rl_agent()
        
        # Get all experiments
        status_result = await agent.process_request({
            "task_type": "get_training_status",
            "user_id": current_user.get("user_id")
        })
        
        if not status_result["success"]:
            raise HTTPException(status_code=500, detail=status_result.get("error"))
        
        experiments = status_result.get("experiments", [])
        
        # Apply filters
        if algorithm:
            experiments = [exp for exp in experiments if exp.get("algorithm") == algorithm]
        
        if environment:
            experiments = [exp for exp in experiments if exp.get("environment") == environment]
        
        return experiments
        
    except Exception as e:
        logger.error(f"Error in list experiments endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environments", response_model=List[Dict[str, Any]])
async def list_environments(
    current_user: dict = Depends(get_current_user)
):
    """
    List all available RL environments
    
    This endpoint returns a list of all available RL environments
    that can be used for training.
    """
    try:
        agent = await get_rl_agent()
        
        # Get environment list
        status_result = await agent.process_request({
            "task_type": "get_training_status",
            "user_id": current_user.get("user_id")
        })
        
        if not status_result["success"]:
            raise HTTPException(status_code=500, detail=status_result.get("error"))
        
        # Return available environments
        environments = status_result.get("available_environments", [])
        
        # Add some default Gym environments
        default_environments = [
            {"name": "CartPole-v1", "type": "gym", "description": "Classic cart-pole balancing task"},
            {"name": "MountainCar-v0", "type": "gym", "description": "Mountain car climbing task"},
            {"name": "Acrobot-v1", "type": "gym", "description": "Acrobot swing-up task"},
            {"name": "LunarLander-v2", "type": "gym", "description": "Lunar lander control task"}
        ]
        
        return default_environments
        
    except Exception as e:
        logger.error(f"Error in list environments endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/experiments/{experiment_name}")
async def delete_experiment(
    experiment_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete an RL experiment
    
    This endpoint deletes an RL experiment and all associated data.
    """
    try:
        agent = await get_rl_agent()
        
        # Check if experiment exists
        if experiment_name not in agent.trained_agents:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
        
        # Delete experiment
        del agent.trained_agents[experiment_name]
        if experiment_name in agent.experiments:
            del agent.experiments[experiment_name]
        
        logger.info(f"Experiment deleted by user {current_user.get('user_id')}: {experiment_name}")
        
        return {"success": True, "message": f"Experiment '{experiment_name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete experiment endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent/capabilities")
async def get_agent_capabilities(
    current_user: dict = Depends(get_current_user)
):
    """
    Get RL agent capabilities
    
    This endpoint returns the capabilities of the ScrollRLAgent.
    """
    try:
        agent = await get_rl_agent()
        
        return {
            "success": True,
            "capabilities": agent.get_capabilities(),
            "status": agent.get_status()
        }
        
    except Exception as e:
        logger.error(f"Error in get capabilities endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))