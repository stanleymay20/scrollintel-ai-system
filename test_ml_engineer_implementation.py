#!/usr/bin/env python3
"""
Test ML Engineer Agent Implementation
"""
import asyncio
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.append('scrollintel_core')

from agents.ml_engineer_agent import MLEngineerAgent
from agents.base import AgentRequest

async def test_ml_engineer_agent():
    """Test the ML Engineer Agent functionality"""
    print("🤖 Testing ML Engineer Agent Implementation")
    print("=" * 50)
    
    # Initialize agent
    agent = MLEngineerAgent()
    
    # Test 1: Agent Info
    print("\n1. Testing Agent Info...")
    info = agent.get_info()
    print(f"Agent Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Capabilities: {len(info['capabilities'])} capabilities")
    for cap in info['capabilities']:
        print(f"  - {cap}")
    
    # Test 2: Health Check
    print("\n2. Testing Health Check...")
    health = await agent.health_check()
    print(f"Health Status: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
    
    # Test 3: General ML Guidance
    print("\n3. Testing General ML Guidance...")
    request = AgentRequest(
        query="What is machine learning?",
        context={}
    )
    response = await agent.process(request)
    print(f"Success: {response.success}")
    print(f"Task Type: {response.metadata.get('task_type')}")
    
    # Test 4: Model Building with Sample Data
    print("\n4. Testing Model Building...")
    
    # Create sample classification data
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    feature_names = [f'feature_{i}' for i in range(5)]
    
    # Create DataFrame
    data_dict = {}
    for i, name in enumerate(feature_names):
        data_dict[name] = X[:, i].tolist()
    data_dict['target'] = y.tolist()
    
    request = AgentRequest(
        query="Build a classification model",
        context={
            "data": data_dict,
            "target_column": "target",
            "problem_type": "classification",
            "test_size": 0.2
        }
    )
    
    response = await agent.process(request)
    print(f"Model Building Success: {response.success}")
    
    if response.success:
        result = response.result
        print(f"Model ID: {result.get('model_id')}")
        print(f"Best Model: {result.get('best_model')}")
        print(f"Problem Type: {result.get('problem_type')}")
        print(f"Performance: {result.get('performance')}")
        
        # Store model_id for next tests
        model_id = result.get('model_id')
        
        # Test 5: Model Evaluation
        print("\n5. Testing Model Evaluation...")
        request = AgentRequest(
            query="Evaluate model performance",
            context={"model_id": model_id}
        )
        response = await agent.process(request)
        print(f"Evaluation Success: {response.success}")
        
        # Test 6: Make Predictions
        print("\n6. Testing Predictions...")
        # Create sample prediction data
        sample_data = {feature_names[i]: [X[0, i]] for i in range(5)}
        
        request = AgentRequest(
            query="Make predictions",
            context={
                "model_id": model_id,
                "input_data": sample_data
            }
        )
        response = await agent.process(request)
        print(f"Prediction Success: {response.success}")
        if response.success:
            print(f"Predictions: {response.result.get('predictions')}")
        
        # Test 7: Model Deployment
        print("\n7. Testing Model Deployment...")
        request = AgentRequest(
            query="Deploy model",
            context={"model_id": model_id}
        )
        response = await agent.process(request)
        print(f"Deployment Success: {response.success}")
        if response.success:
            print(f"Deployment Status: {response.result.get('deployment_status')}")
    
    # Test 8: Hyperparameter Tuning
    print("\n8. Testing Hyperparameter Tuning...")
    request = AgentRequest(
        query="Tune hyperparameters",
        context={
            "data": data_dict,
            "target_column": "target",
            "model_type": "random_forest",
            "search_method": "grid",
            "cv_folds": 3  # Reduced for faster testing
        }
    )
    response = await agent.process(request)
    print(f"Hyperparameter Tuning Success: {response.success}")
    if response.success:
        result = response.result
        print(f"Tuned Model ID: {result.get('model_id')}")
        print(f"Best Score: {result.get('best_score')}")
        print(f"Best Params: {result.get('best_params')}")
    
    print("\n" + "=" * 50)
    print("✅ ML Engineer Agent Implementation Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_ml_engineer_agent())