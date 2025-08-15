#!/usr/bin/env python3
"""
Demo script for ScrollMLEngineer Agent
Tests ML engineering workflows and pipeline management capabilities.
"""

import asyncio
import json
from datetime import datetime
from scrollintel.agents.scroll_ml_engineer import ScrollMLEngineer, MLFramework, DeploymentTarget
from scrollintel.core.interfaces import AgentRequest


async def demo_ml_pipeline_setup():
    """Demo ML pipeline setup."""
    print("🔧 Testing ML Pipeline Setup...")
    
    agent = ScrollMLEngineer()
    
    request = AgentRequest(
        id="demo-pipeline-1",
        user_id="demo-user",
        agent_id="scroll-ml-engineer",
        prompt="Set up ML pipeline for customer churn prediction",
        context={
            "action": "setup_pipeline",
            "dataset_path": "data/customers.csv",
            "target_column": "churn",
            "framework": MLFramework.SCIKIT_LEARN
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"Status: {response.status}")
    print(f"Execution time: {response.execution_time:.2f}s")
    print("Response preview:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    print()


async def demo_model_deployment():
    """Demo model deployment."""
    print("🚀 Testing Model Deployment...")
    
    agent = ScrollMLEngineer()
    
    request = AgentRequest(
        id="demo-deploy-1",
        user_id="demo-user",
        agent_id="scroll-ml-engineer",
        prompt="Deploy my trained churn prediction model",
        context={
            "action": "deploy_model",
            "model_name": "churn_predictor",
            "model_path": "models/churn_model.pkl",
            "deployment_target": DeploymentTarget.FASTAPI
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"Status: {response.status}")
    print(f"Execution time: {response.execution_time:.2f}s")
    print("Response preview:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    print()


async def demo_model_monitoring():
    """Demo model monitoring."""
    print("📊 Testing Model Monitoring...")
    
    agent = ScrollMLEngineer()
    
    request = AgentRequest(
        id="demo-monitor-1",
        user_id="demo-user",
        agent_id="scroll-ml-engineer",
        prompt="Monitor my deployed churn prediction model",
        context={
            "action": "monitor_model",
            "model_name": "churn_predictor",
            "monitoring_config": {
                "accuracy": 0.92,
                "drift_detected": False,
                "performance_degradation": False,
                "new_data_available": True
            }
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"Status: {response.status}")
    print(f"Execution time: {response.execution_time:.2f}s")
    print("Response preview:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    print()


async def demo_framework_integration():
    """Demo framework integration."""
    print("🔗 Testing Framework Integration...")
    
    agent = ScrollMLEngineer()
    
    request = AgentRequest(
        id="demo-framework-1",
        user_id="demo-user",
        agent_id="scroll-ml-engineer",
        prompt="Help me integrate TensorFlow for deep learning",
        context={
            "action": "framework_integration",
            "framework": MLFramework.TENSORFLOW,
            "integration_type": "deep_learning",
            "requirements": ["neural_networks", "gpu_support"]
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"Status: {response.status}")
    print(f"Execution time: {response.execution_time:.2f}s")
    print("Response preview:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    print()


async def demo_mlops_automation():
    """Demo MLOps automation."""
    print("⚙️ Testing MLOps Automation...")
    
    agent = ScrollMLEngineer()
    
    request = AgentRequest(
        id="demo-mlops-1",
        user_id="demo-user",
        agent_id="scroll-ml-engineer",
        prompt="Set up MLOps automation for my ML project",
        context={
            "action": "mlops_automation",
            "project_name": "churn_prediction_project",
            "config": {"ci_cd": True, "monitoring": True, "auto_retrain": True}
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"Status: {response.status}")
    print(f"Execution time: {response.execution_time:.2f}s")
    print("Response preview:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    print()


async def demo_general_advice():
    """Demo general ML engineering advice."""
    print("💡 Testing General ML Engineering Advice...")
    
    agent = ScrollMLEngineer()
    
    request = AgentRequest(
        id="demo-advice-1",
        user_id="demo-user",
        agent_id="scroll-ml-engineer",
        prompt="What's the best approach for handling model versioning in production?",
        context={},
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"Status: {response.status}")
    print(f"Execution time: {response.execution_time:.2f}s")
    print("Response preview:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    print()


async def main():
    """Run all demos."""
    print("=" * 60)
    print("ScrollMLEngineer Agent Demo")
    print("=" * 60)
    print()
    
    # Test agent initialization
    agent = ScrollMLEngineer()
    print(f"Agent ID: {agent.agent_id}")
    print(f"Agent Name: {agent.name}")
    print(f"Agent Type: {agent.agent_type}")
    print(f"Capabilities: {len(agent.get_capabilities())}")
    print(f"Health Check: {await agent.health_check()}")
    print()
    
    # Run demos
    await demo_ml_pipeline_setup()
    await demo_model_deployment()
    await demo_model_monitoring()
    await demo_framework_integration()
    await demo_mlops_automation()
    await demo_general_advice()
    
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())