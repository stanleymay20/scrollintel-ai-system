#!/usr/bin/env python3
"""
Focused Demo of ML Engineer Agent - Core Functionality
Showcases the working features: model building, hyperparameter tuning, predictions, and deployment
"""
import asyncio
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.append('scrollintel_core')

from agents.ml_engineer_agent import MLEngineerAgent
from agents.base import AgentRequest

async def main():
    """Demonstrate ML Engineer Agent core capabilities"""
    print("🤖 ML Engineer Agent - Core Capabilities Demo")
    print("=" * 60)
    
    agent = MLEngineerAgent()
    
    # Show agent info
    print("📋 Agent Information:")
    info = agent.get_info()
    print(f"   Name: {info['name']}")
    print(f"   Description: {info['description']}")
    print(f"   Capabilities: {len(info['capabilities'])} features")
    
    # Health check
    health = await agent.health_check()
    print(f"   Status: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
    
    # Create sample classification dataset
    print("\n📊 Creating Sample Dataset...")
    X, y = make_classification(
        n_samples=200, 
        n_features=6, 
        n_classes=2, 
        n_informative=4,
        n_redundant=2,
        random_state=42
    )
    
    # Convert to dictionary format
    feature_names = [f'feature_{i}' for i in range(6)]
    data_dict = {}
    for i, name in enumerate(feature_names):
        data_dict[name] = X[:, i].tolist()
    data_dict['target'] = y.tolist()
    
    print(f"   Dataset: {len(data_dict['target'])} samples, {len(feature_names)} features, 2 classes")
    
    # 1. Build initial model
    print("\n🔨 Step 1: Building Initial Model...")
    request = AgentRequest(
        query="Build a classification model",
        context={
            "data": data_dict,
            "target_column": "target",
            "problem_type": "classification"
        }
    )
    
    response = await agent.process(request)
    if response.success and 'error' not in response.result:
        result = response.result
        print(f"✅ Model built successfully!")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Best Algorithm: {result['best_model']}")
        print(f"   Accuracy: {result['performance']['accuracy']:.3f}")
        print(f"   F1-Score: {result['performance']['f1_score']:.3f}")
        
        initial_model_id = result['model_id']
        
        # 2. Hyperparameter tuning
        print("\n⚙️ Step 2: Hyperparameter Tuning...")
        request = AgentRequest(
            query="Tune hyperparameters",
            context={
                "data": data_dict,
                "target_column": "target",
                "model_type": "random_forest",
                "search_method": "grid",
                "cv_folds": 3  # Reduced for faster demo
            }
        )
        
        response = await agent.process(request)
        if response.success and 'error' not in response.result:
            result = response.result
            print(f"✅ Hyperparameter tuning completed!")
            print(f"   Tuned Model ID: {result['model_id']}")
            print(f"   Cross-validation Score: {result['best_score']:.3f}")
            print(f"   Best Parameters: {result['best_params']}")
            
            tuned_model_id = result['model_id']
            
            # 3. Make predictions
            print("\n🎯 Step 3: Making Predictions...")
            # Create a sample for prediction
            sample_data = {feature_names[i]: X[0, i] for i in range(6)}
            
            request = AgentRequest(
                query="Predict",
                context={
                    "model_id": tuned_model_id,
                    "input_data": sample_data
                }
            )
            
            response = await agent.process(request)
            if response.success and 'predictions' in response.result:
                result = response.result
                print(f"✅ Prediction made!")
                print(f"   Input sample: {sample_data}")
                print(f"   Predicted class: {result['predictions'][0]}")
                if 'probabilities' in result:
                    probs = result['probabilities'][0]
                    print(f"   Class probabilities: [Class 0: {probs[0]:.3f}, Class 1: {probs[1]:.3f}]")
            
            # 4. Model comparison
            print("\n📊 Step 4: Model Comparison...")
            request = AgentRequest(
                query="Compare models",
                context={"model_ids": [initial_model_id, tuned_model_id]}
            )
            
            response = await agent.process(request)
            if response.success:
                result = response.result
                if 'comparison_results' in result:
                    print(f"✅ Compared {len(result['comparison_results'])} models")
                    for model_id, info in result['comparison_results'].items():
                        print(f"   {model_id}: {info['status']}")
                else:
                    print("✅ Model comparison guidance provided")
            
            # 5. Deployment
            print("\n🚀 Step 5: Model Deployment...")
            request = AgentRequest(
                query="Deploy model",
                context={"model_id": tuned_model_id}
            )
            
            response = await agent.process(request)
            if response.success:
                result = response.result
                print(f"✅ Deployment code generated!")
                print(f"   Status: {result['deployment_status']}")
                print(f"   Available endpoints: {list(result['api_endpoints'].keys())}")
                print(f"   Local URL: {result['deployment_options']['local']['url']}")
        else:
            print("❌ Hyperparameter tuning failed, using initial model")
            tuned_model_id = initial_model_id
    else:
        print("❌ Initial model building failed")
        return
    
    # 6. General ML guidance
    print("\n💡 Step 6: ML Guidance...")
    request = AgentRequest(
        query="What are ML best practices?",
        context={}
    )
    
    response = await agent.process(request)
    if response.success:
        result = response.result
        print("✅ ML guidance provided!")
        print("   Available commands:")
        for cmd, desc in result['available_commands'].items():
            print(f"   - {cmd}: {desc}")
    
    print("\n" + "=" * 60)
    print("🎉 ML Engineer Agent Demo Complete!")
    print("✅ Successfully demonstrated:")
    print("   • Automated model building with multiple algorithms")
    print("   • Hyperparameter tuning with cross-validation")
    print("   • Model performance evaluation and comparison")
    print("   • Real-time predictions with probability scores")
    print("   • FastAPI deployment code generation")
    print("   • ML best practices guidance")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())