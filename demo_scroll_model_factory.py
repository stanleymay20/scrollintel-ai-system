#!/usr/bin/env python3
"""
Demo script for ScrollModelFactory engine.
Demonstrates custom model creation, validation, and deployment.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

from scrollintel.engines.scroll_model_factory import ScrollModelFactory, ModelAlgorithm, ModelTemplate, ValidationStrategy


async def demo_scroll_model_factory():
    """Demonstrate ScrollModelFactory capabilities."""
    print("🏭 ScrollModelFactory Demo")
    print("=" * 50)
    
    # Initialize engine
    print("\n1. Initializing ScrollModelFactory engine...")
    engine = ScrollModelFactory()
    await engine.start()
    
    try:
        # Get engine status
        print("\n2. Engine Status:")
        status = engine.get_status()
        print(f"   Engine ID: {status['engine_id']}")
        print(f"   Status: {status['status']}")
        print(f"   Available Templates: {status['available_templates']}")
        print(f"   Available Algorithms: {status['available_algorithms']}")
        
        # Get templates
        print("\n3. Available Templates:")
        templates_result = await engine.process(
            input_data=None,
            parameters={"action": "get_templates"}
        )
        
        for template_key, template_info in templates_result["templates"].items():
            print(f"   • {template_info['name']}: {template_info['description']}")
        
        # Get algorithms
        print("\n4. Available Algorithms:")
        algorithms_result = await engine.process(
            input_data=None,
            parameters={"action": "get_algorithms"}
        )
        
        for alg_key, alg_info in algorithms_result["algorithms"].items():
            supports = []
            if alg_info["supports_classification"]:
                supports.append("Classification")
            if alg_info["supports_regression"]:
                supports.append("Regression")
            print(f"   • {alg_info['name']}: {', '.join(supports)}")
        
        # Create sample data
        print("\n5. Creating sample dataset...")
        np.random.seed(42)
        n_samples = 200
        
        # Create classification dataset
        X = np.random.randn(n_samples, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        data = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
        data['target'] = y
        
        print(f"   Dataset shape: {data.shape}")
        print(f"   Target distribution: {data['target'].value_counts().to_dict()}")
        
        # Create classification model
        print("\n6. Creating classification model...")
        model_params = {
            "action": "create_model",
            "model_name": "demo_classification_model",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "template": ModelTemplate.BINARY_CLASSIFICATION.value,
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3", "feature4"],
            "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT.value,
            "hyperparameter_tuning": False,
            "custom_params": {"n_estimators": 50, "max_depth": 10}
        }
        
        model_result = await engine.process(input_data=data, parameters=model_params)
        
        print(f"   Model ID: {model_result['model_id']}")
        print(f"   Algorithm: {model_result['algorithm']}")
        print(f"   Is Classification: {model_result['is_classification']}")
        print(f"   Training Duration: {model_result['training_duration']:.2f}s")
        
        print("   Metrics:")
        for metric, value in model_result['metrics'].items():
            if isinstance(value, float):
                print(f"     {metric}: {value:.4f}")
            else:
                print(f"     {metric}: {value}")
        
        # Validate model
        print("\n7. Validating model...")
        validation_data = [[1.0, 2.0, -0.5, 0.8], [0.5, -1.5, 2.0, -0.3]]
        
        validation_params = {
            "action": "validate_model",
            "model_id": model_result['model_id'],
            "validation_data": validation_data
        }
        
        validation_result = await engine.process(input_data=None, parameters=validation_params)
        
        print(f"   Validation Status: {validation_result['validation_status']}")
        print(f"   Predictions: {validation_result['predictions']}")
        
        # Deploy model
        print("\n8. Deploying model...")
        deployment_params = {
            "action": "deploy_model",
            "model_id": model_result['model_id'],
            "endpoint_name": "demo_classification_endpoint"
        }
        
        deployment_result = await engine.process(input_data=None, parameters=deployment_params)
        
        print(f"   Deployment Status: {deployment_result['status']}")
        print(f"   API Endpoint: {deployment_result['api_endpoint']}")
        print(f"   Endpoint Name: {deployment_result['endpoint_name']}")
        
        # Create regression model
        print("\n9. Creating regression model...")
        
        # Create regression dataset
        X_reg = np.random.randn(n_samples, 3)
        y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] * 1.5 + X_reg[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1
        
        reg_data = pd.DataFrame(X_reg, columns=['x1', 'x2', 'x3'])
        reg_data['y'] = y_reg
        
        reg_params = {
            "action": "create_model",
            "model_name": "demo_regression_model",
            "algorithm": ModelAlgorithm.LINEAR_REGRESSION.value,
            "template": ModelTemplate.REGRESSION.value,
            "target_column": "y",
            "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT.value,
            "hyperparameter_tuning": False
        }
        
        reg_result = await engine.process(input_data=reg_data, parameters=reg_params)
        
        print(f"   Model ID: {reg_result['model_id']}")
        print(f"   Algorithm: {reg_result['algorithm']}")
        print(f"   Is Classification: {reg_result['is_classification']}")
        print(f"   Training Duration: {reg_result['training_duration']:.2f}s")
        
        print("   Metrics:")
        for metric, value in reg_result['metrics'].items():
            if isinstance(value, float):
                print(f"     {metric}: {value:.4f}")
            else:
                print(f"     {metric}: {value}")
        
        # Test hyperparameter tuning
        print("\n10. Creating model with hyperparameter tuning...")
        
        tuned_params = {
            "action": "create_model",
            "model_name": "demo_tuned_model",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "target_column": "target",
            "validation_strategy": ValidationStrategy.CROSS_VALIDATION.value,
            "hyperparameter_tuning": True
        }
        
        tuned_result = await engine.process(input_data=data, parameters=tuned_params)
        
        print(f"   Model ID: {tuned_result['model_id']}")
        print(f"   Training Duration: {tuned_result['training_duration']:.2f}s")
        print("   Tuned Metrics:")
        for metric, value in tuned_result['metrics'].items():
            if isinstance(value, float):
                print(f"     {metric}: {value:.4f}")
            else:
                print(f"     {metric}: {value}")
        
        print("\n✅ ScrollModelFactory demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        raise
    
    finally:
        # Clean up
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(demo_scroll_model_factory())