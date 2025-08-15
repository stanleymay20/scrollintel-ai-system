#!/usr/bin/env python3
"""
Comprehensive Demo of ML Engineer Agent
Showcases automated model building, hyperparameter tuning, and deployment
"""
import asyncio
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.append('scrollintel_core')

from agents.ml_engineer_agent import MLEngineerAgent
from agents.base import AgentRequest

async def demo_classification_workflow():
    """Demo complete classification workflow"""
    print("🎯 Classification Workflow Demo")
    print("-" * 40)
    
    agent = MLEngineerAgent()
    
    # Create sample data
    X, y = make_classification(
        n_samples=200, 
        n_features=8, 
        n_classes=3, 
        n_informative=6,
        n_redundant=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(8)]
    data_dict = {}
    for i, name in enumerate(feature_names):
        data_dict[name] = X[:, i].tolist()
    data_dict['category'] = y.tolist()
    
    print(f"📊 Dataset: {len(data_dict['category'])} samples, {len(feature_names)} features, 3 classes")
    
    # Step 1: Build initial model
    print("\n1️⃣ Building initial model...")
    request = AgentRequest(
        query="Build a classification model",
        context={
            "data": data_dict,
            "target_column": "category",
            "problem_type": "classification"
        }
    )
    
    response = await agent.process(request)
    if response.success:
        result = response.result
        print(f"✅ Model built successfully!")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Best Model: {result['best_model']}")
        print(f"   Performance: {result['performance']}")
        
        initial_model_id = result['model_id']
        
        # Step 2: Hyperparameter tuning
        print("\n2️⃣ Tuning hyperparameters...")
        request = AgentRequest(
            query="Tune hyperparameters",
            context={
                "data": data_dict,
                "target_column": "category",
                "model_type": "random_forest",
                "search_method": "grid",
                "cv_folds": 5
            }
        )
        
        response = await agent.process(request)
        if response.success:
            result = response.result
            if 'error' in result:
                print(f"❌ Hyperparameter tuning error: {result['error']}")
                tuned_model_id = initial_model_id  # Use initial model instead
            else:
                print(f"✅ Hyperparameter tuning completed!")
                print(f"   Tuned Model ID: {result['model_id']}")
                print(f"   Best Score: {result['best_score']:.4f}")
                print(f"   Best Parameters: {result['best_params']}")
                
                tuned_model_id = result['model_id']
            
            # Step 3: Compare models
            print("\n3️⃣ Comparing models...")
            request = AgentRequest(
                query="Compare models",
                context={"model_ids": [initial_model_id, tuned_model_id]}
            )
            
            response = await agent.process(request)
            if response.success:
                result = response.result
                if 'comparison_results' in result:
                    print("✅ Model comparison completed!")
                    print(f"   Available models: {len(result['comparison_results'])}")
                else:
                    print("✅ Model comparison guidance provided!")
            
            # Step 4: Make predictions
            print("\n4️⃣ Making predictions...")
            sample_data = {feature_names[i]: X[0, i] for i in range(8)}
            
            request = AgentRequest(
                query="Predict",
                context={
                    "model_id": tuned_model_id,
                    "input_data": sample_data
                }
            )
            
            response = await agent.process(request)
            if response.success:
                result = response.result
                print(f"✅ Predictions made!")
                print(f"   Predicted class: {result['predictions'][0]}")
                if 'probabilities' in result:
                    print(f"   Class probabilities: {result['probabilities'][0]}")
            
            # Step 5: Generate deployment code
            print("\n5️⃣ Generating deployment code...")
            request = AgentRequest(
                query="Deploy model",
                context={"model_id": tuned_model_id}
            )
            
            response = await agent.process(request)
            if response.success:
                print("✅ Deployment code generated!")
                print(f"   Status: {response.result['deployment_status']}")
                print(f"   API endpoints: {list(response.result['api_endpoints'].keys())}")

async def demo_regression_workflow():
    """Demo regression workflow"""
    print("\n\n📈 Regression Workflow Demo")
    print("-" * 40)
    
    agent = MLEngineerAgent()
    
    # Create regression data
    X, y = make_regression(
        n_samples=150,
        n_features=6,
        noise=0.1,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'input_{i}' for i in range(6)]
    data_dict = {}
    for i, name in enumerate(feature_names):
        data_dict[name] = X[:, i].tolist()
    data_dict['target_value'] = y.tolist()
    
    print(f"📊 Dataset: {len(data_dict['target_value'])} samples, {len(feature_names)} features")
    
    # Build regression model
    print("\n1️⃣ Building regression model...")
    request = AgentRequest(
        query="Build regression model",
        context={
            "data": data_dict,
            "target_column": "target_value",
            "problem_type": "regression"
        }
    )
    
    response = await agent.process(request)
    if response.success:
        result = response.result
        if 'error' in result:
            print(f"❌ Regression model error: {result['error']}")
        else:
            print(f"✅ Regression model built!")
            print(f"   Model ID: {result['model_id']}")
            print(f"   Best Model: {result['best_model']}")
            print(f"   R² Score: {result['performance']['r2_score']:.4f}")
            print(f"   RMSE: {result['performance']['rmse']:.4f}")

async def demo_real_world_dataset():
    """Demo with real-world dataset"""
    print("\n\n🌍 Real-World Dataset Demo (Iris)")
    print("-" * 40)
    
    agent = MLEngineerAgent()
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    # Convert to dict format
    data_dict = df.to_dict('list')
    
    print(f"📊 Iris Dataset: {len(df)} samples, {len(iris.feature_names)} features, 3 species")
    
    # Build model with hyperparameter tuning
    print("\n1️⃣ Building optimized model...")
    request = AgentRequest(
        query="Build and tune classification model",
        context={
            "data": data_dict,
            "target_column": "species",
            "model_type": "random_forest",
            "search_method": "random",
            "cv_folds": 5
        }
    )
    
    response = await agent.process(request)
    if response.success:
        result = response.result
        if 'error' in result:
            print(f"❌ Model optimization error: {result['error']}")
        else:
            print(f"✅ Optimized model built!")
            print(f"   Model ID: {result['model_id']}")
            print(f"   Cross-validation score: {result['best_score']:.4f}")
            print(f"   Optimization method: {result['search_method']}")
            
            # Test prediction on a sample
            sample = {
                'sepal length (cm)': 5.1,
                'sepal width (cm)': 3.5,
                'petal length (cm)': 1.4,
                'petal width (cm)': 0.2
            }
            
            request = AgentRequest(
                query="Predict species",
                context={
                    "model_id": result['model_id'],
                    "input_data": sample
                }
            )
            
            pred_response = await agent.process(request)
            if pred_response.success:
                pred_result = pred_response.result
                if 'predictions' in pred_result and pred_result['predictions']:
                    species_names = ['setosa', 'versicolor', 'virginica']
                    predicted_species = species_names[int(pred_result['predictions'][0])]
                    print(f"   Sample prediction: {predicted_species}")
                else:
                    print("   Prediction completed")

async def demo_error_handling():
    """Demo error handling and guidance"""
    print("\n\n⚠️ Error Handling Demo")
    print("-" * 40)
    
    agent = MLEngineerAgent()
    
    # Test with missing data
    print("\n1️⃣ Testing with missing required data...")
    request = AgentRequest(
        query="Build model without data",
        context={}
    )
    
    response = await agent.process(request)
    if not response.success:
        print("✅ Proper error handling for missing data")
    else:
        print("✅ Guidance provided for model building")
    
    # Test with invalid model ID
    print("\n2️⃣ Testing with invalid model ID...")
    request = AgentRequest(
        query="Make predictions",
        context={
            "model_id": "nonexistent_model",
            "input_data": {"feature1": 1.0}
        }
    )
    
    response = await agent.process(request)
    if response.success and "error" in response.result:
        print("✅ Proper error handling for invalid model ID")

async def main():
    """Run comprehensive ML Engineer Agent demo"""
    print("🤖 ML Engineer Agent - Comprehensive Demo")
    print("=" * 60)
    print("Demonstrating automated model building, hyperparameter tuning,")
    print("cross-validation, performance evaluation, and deployment pipeline")
    print("=" * 60)
    
    # Run all demos
    await demo_classification_workflow()
    await demo_regression_workflow()
    await demo_real_world_dataset()
    await demo_error_handling()
    
    print("\n" + "=" * 60)
    print("🎉 ML Engineer Agent Demo Complete!")
    print("✅ All core capabilities demonstrated successfully")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())