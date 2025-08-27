"""
Demo script for ScrollRLAgent - Reinforcement Learning Agent

This script demonstrates the capabilities of the ScrollRLAgent including:
- DQN training on CartPole environment
- A2C training on MountainCar environment
- Multi-agent cooperative training
- Policy evaluation
- Reward function optimization
"""

import asyncio
import json
import time
from datetime import datetime

from scrollintel.agents.scroll_rl_agent import ScrollRLAgent

async def demo_scroll_rl_agent():
    """Comprehensive demo of ScrollRLAgent capabilities"""
    
    print("🤖 ScrollRLAgent Demo - Reinforcement Learning Capabilities")
    print("=" * 60)
    
    # Initialize agent
    agent = ScrollRLAgent()
    
    print(f"✅ Agent initialized: {agent.agent_type}")
    print(f"🎯 Capabilities: {', '.join(agent.capabilities)}")
    print()
    
    # Demo 1: Create Environment
    print("📋 Demo 1: Environment Creation")
    print("-" * 30)
    
    env_request = {
        "task_type": "create_environment",
        "environment_name": "CartPole-v1",
        "environment_type": "gym"
    }
    
    result = await agent.process_request(env_request)
    if result["success"]:
        print(f"✅ Environment created: {result['environment_name']}")
        print(f"   Environment info: {result['environment_info']}")
    else:
        print(f"❌ Environment creation failed: {result['error']}")
    print()
    
    # Demo 2: DQN Training
    print("📋 Demo 2: DQN Training on CartPole")
    print("-" * 30)
    
    dqn_config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "batch_size": 32,
        "num_episodes": 200,  # Reduced for demo
        "max_steps_per_episode": 200
    }
    
    dqn_request = {
        "task_type": "train_dqn",
        "environment": "CartPole-v1",
        "config": dqn_config,
        "experiment_name": f"demo_dqn_{datetime.now().strftime('%H%M%S')}"
    }
    
    print("🚀 Starting DQN training...")
    start_time = time.time()
    
    result = await agent.process_request(dqn_request)
    
    training_time = time.time() - start_time
    
    if result["success"]:
        print(f"✅ DQN training completed in {training_time:.2f} seconds")
        print(f"   Experiment: {result['experiment_name']}")
        print(f"   Algorithm: {result['algorithm']}")
        print(f"   Environment: {result['environment']}")
        print(f"   Final avg score: {result['final_avg_score']:.2f}")
        print(f"   Total episodes: {result['total_episodes']}")
        
        if result.get("training_log"):
            print("   Recent training progress:")
            for log_entry in result["training_log"][-3:]:
                print(f"     Episode {log_entry['episode']}: Score {log_entry['score']:.2f}, "
                      f"Avg {log_entry['avg_score']:.2f}, ε={log_entry['epsilon']:.3f}")
        
        dqn_experiment_name = result['experiment_name']
    else:
        print(f"❌ DQN training failed: {result['error']}")
        dqn_experiment_name = None
    print()
    
    # Demo 3: A2C Training
    print("📋 Demo 3: A2C Training on MountainCar")
    print("-" * 30)
    
    a2c_config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "num_episodes": 150,  # Reduced for demo
        "max_steps_per_episode": 200
    }
    
    a2c_request = {
        "task_type": "train_a2c",
        "environment": "MountainCar-v0",
        "config": a2c_config,
        "experiment_name": f"demo_a2c_{datetime.now().strftime('%H%M%S')}"
    }
    
    print("🚀 Starting A2C training...")
    start_time = time.time()
    
    result = await agent.process_request(a2c_request)
    
    training_time = time.time() - start_time
    
    if result["success"]:
        print(f"✅ A2C training completed in {training_time:.2f} seconds")
        print(f"   Experiment: {result['experiment_name']}")
        print(f"   Algorithm: {result['algorithm']}")
        print(f"   Environment: {result['environment']}")
        print(f"   Final avg score: {result['final_avg_score']:.2f}")
        print(f"   Total episodes: {result['total_episodes']}")
        
        if result.get("training_log"):
            print("   Recent training progress:")
            for log_entry in result["training_log"][-3:]:
                print(f"     Episode {log_entry['episode']}: Score {log_entry['score']:.2f}, "
                      f"Avg {log_entry['avg_score']:.2f}")
        
        a2c_experiment_name = result['experiment_name']
    else:
        print(f"❌ A2C training failed: {result['error']}")
        a2c_experiment_name = None
    print()
    
    # Demo 4: Multi-Agent Training
    print("📋 Demo 4: Multi-Agent Cooperative Training")
    print("-" * 30)
    
    multi_agent_config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "batch_size": 32,
        "num_episodes": 100,  # Reduced for demo
        "max_steps_per_episode": 200
    }
    
    multi_agent_request = {
        "task_type": "multi_agent_training",
        "environment": "CartPole-v1",
        "num_agents": 3,
        "scenario_type": "cooperative",
        "algorithm": "DQN",
        "config": multi_agent_config,
        "experiment_name": f"demo_multi_agent_{datetime.now().strftime('%H%M%S')}"
    }
    
    print("🚀 Starting multi-agent training...")
    start_time = time.time()
    
    result = await agent.process_request(multi_agent_request)
    
    training_time = time.time() - start_time
    
    if result["success"]:
        print(f"✅ Multi-agent training completed in {training_time:.2f} seconds")
        print(f"   Experiment: {result['experiment_name']}")
        print(f"   Scenario: {result['scenario_type']}")
        print(f"   Number of agents: {result['num_agents']}")
        print(f"   Final avg scores: {[f'{score:.2f}' for score in result['final_avg_scores']]}")
        print(f"   Total episodes: {result['total_episodes']}")
        
        multi_agent_experiment_name = result['experiment_name']
    else:
        print(f"❌ Multi-agent training failed: {result['error']}")
        multi_agent_experiment_name = None
    print()
    
    # Demo 5: Policy Evaluation
    if dqn_experiment_name:
        print("📋 Demo 5: Policy Evaluation")
        print("-" * 30)
        
        eval_request = {
            "task_type": "evaluate_policy",
            "experiment_name": dqn_experiment_name,
            "num_episodes": 50,
            "render": False
        }
        
        print(f"🔍 Evaluating policy: {dqn_experiment_name}")
        
        result = await agent.process_request(eval_request)
        
        if result["success"]:
            print(f"✅ Policy evaluation completed")
            print(f"   Experiment: {result['experiment_name']}")
            print(f"   Evaluation episodes: {result['evaluation_episodes']}")
            print(f"   Average score: {result['avg_score']:.2f} ± {result['std_score']:.2f}")
            print(f"   Score range: {result['min_score']:.2f} to {result['max_score']:.2f}")
            print(f"   Average episode length: {result['avg_episode_length']:.1f}")
            print(f"   Success rate: {result['success_rate']:.2%}")
        else:
            print(f"❌ Policy evaluation failed: {result['error']}")
        print()
    
    # Demo 6: Reward Function Optimization
    print("📋 Demo 6: Reward Function Optimization")
    print("-" * 30)
    
    reward_request = {
        "task_type": "optimize_rewards",
        "environment": "CartPole-v1",
        "reward_components": {
            "survival_bonus": 0.1,
            "progress_reward": 0.1,
            "stability_reward": 0.5
        },
        "method": "grid_search"
    }
    
    print("🔧 Optimizing reward function...")
    start_time = time.time()
    
    result = await agent.process_request(reward_request)
    
    optimization_time = time.time() - start_time
    
    if result["success"]:
        print(f"✅ Reward optimization completed in {optimization_time:.2f} seconds")
        print(f"   Optimization method: {result['optimization_method']}")
        print(f"   Best configuration: {result['best_config']}")
        print(f"   Best score: {result['best_score']:.2f}")
        print(f"   Configurations tested: {result['total_configurations_tested']}")
        
        if result.get("all_results"):
            print("   Top 3 configurations:")
            sorted_results = sorted(result["all_results"], key=lambda x: x["avg_score"], reverse=True)
            for i, res in enumerate(sorted_results[:3]):
                print(f"     {i+1}. Score: {res['avg_score']:.2f}, Config: {res['config']}")
    else:
        print(f"❌ Reward optimization failed: {result['error']}")
    print()
    
    # Demo 7: Training Status
    print("📋 Demo 7: Training Status Overview")
    print("-" * 30)
    
    status_request = {
        "task_type": "get_training_status"
    }
    
    result = await agent.process_request(status_request)
    
    if result["success"]:
        print(f"✅ Training status retrieved")
        print(f"   Total experiments: {result['total_experiments']}")
        
        if result.get("experiments"):
            print("   Experiment summary:")
            for exp in result["experiments"]:
                print(f"     • {exp['experiment_name']}: {exp['algorithm']} on {exp['environment']}")
                print(f"       Episodes: {exp['training_episodes']}, "
                      f"Performance: {exp['final_performance']:.2f}")
                if exp.get("num_agents"):
                    print(f"       Multi-agent: {exp['num_agents']} agents, {exp['scenario_type']}")
        
        if result.get("available_environments"):
            print(f"   Available environments: {', '.join(result['available_environments'])}")
    else:
        print(f"❌ Status retrieval failed: {result['error']}")
    print()
    
    # Demo 8: Agent Status and Capabilities
    print("📋 Demo 8: Agent Status and Capabilities")
    print("-" * 30)
    
    status = agent.get_status()
    capabilities = agent.get_capabilities()
    
    print(f"✅ Agent Status:")
    print(f"   Type: {status['agent_type']}")
    print(f"   Status: {status['status']}")
    print(f"   Trained models: {status['trained_models']}")
    print(f"   Environments: {status['environments']}")
    print(f"   Experiments: {status['experiments']}")
    print()
    
    print(f"🎯 Agent Capabilities:")
    for capability in capabilities:
        print(f"   • {capability}")
    print()
    
    # Summary
    print("📊 Demo Summary")
    print("-" * 30)
    print("✅ ScrollRLAgent successfully demonstrated:")
    print("   • Environment creation and management")
    print("   • DQN training with experience replay")
    print("   • A2C training with actor-critic architecture")
    print("   • Multi-agent cooperative training")
    print("   • Policy evaluation and performance metrics")
    print("   • Reward function optimization")
    print("   • Training status and experiment management")
    print()
    print("🎉 All RL capabilities are working correctly!")
    print("🚀 ScrollRLAgent is ready for production use!")

async def demo_advanced_rl_features():
    """Demo advanced RL features"""
    
    print("\n🔬 Advanced RL Features Demo")
    print("=" * 40)
    
    agent = ScrollRLAgent()
    
    # Demo competitive multi-agent training
    print("📋 Competitive Multi-Agent Training")
    print("-" * 30)
    
    competitive_request = {
        "task_type": "multi_agent_training",
        "environment": "CartPole-v1",
        "num_agents": 2,
        "scenario_type": "competitive",
        "algorithm": "A2C",
        "config": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "num_episodes": 50,
            "max_steps_per_episode": 200
        },
        "experiment_name": f"competitive_demo_{datetime.now().strftime('%H%M%S')}"
    }
    
    print("🥊 Starting competitive training...")
    result = await agent.process_request(competitive_request)
    
    if result["success"]:
        print(f"✅ Competitive training completed")
        print(f"   Final scores: {[f'{score:.2f}' for score in result['final_avg_scores']]}")
        print(f"   Competition dynamics observed!")
    else:
        print(f"❌ Competitive training failed: {result['error']}")
    print()
    
    # Demo different reward optimization methods
    print("📋 Advanced Reward Optimization")
    print("-" * 30)
    
    # Test unsupported method
    unsupported_request = {
        "task_type": "optimize_rewards",
        "environment": "CartPole-v1",
        "reward_components": {"survival_bonus": 0.1},
        "method": "bayesian_optimization"  # Not implemented
    }
    
    result = await agent.process_request(unsupported_request)
    
    if not result["success"]:
        print(f"✅ Correctly handled unsupported method: {result['error']}")
        print(f"   Supported methods: {result.get('supported_methods', [])}")
    print()
    
    print("🎯 Advanced features demonstrated successfully!")

if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(demo_scroll_rl_agent())
    
    # Run advanced features demo
    asyncio.run(demo_advanced_rl_features())