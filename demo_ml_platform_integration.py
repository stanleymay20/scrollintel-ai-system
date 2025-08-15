"""
Demo script for ML Platform Integration functionality

This script demonstrates the capabilities of the ML Platform Integration engine
including connecting to ML platforms, retrieving models, deploying models,
and analyzing data quality correlations with model performance.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from ai_data_readiness.engines.ml_platform_integrator import (
    MLPlatformIntegrator,
    MLPlatformConfig,
    ModelDeploymentInfo,
    DataQualityCorrelation
)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def demo_platform_registration():
    """Demonstrate ML platform registration"""
    print_section("ML Platform Registration Demo")
    
    integrator = MLPlatformIntegrator()
    
    # Register different types of ML platforms
    platforms = [
        {
            "name": "local_mlflow",
            "config": MLPlatformConfig(
                platform_type="mlflow",
                endpoint_url="http://localhost:5000",
                credentials={},
                metadata={"environment": "development"}
            )
        },
        {
            "name": "production_kubeflow",
            "config": MLPlatformConfig(
                platform_type="kubeflow",
                endpoint_url="https://kubeflow.company.com",
                credentials={"token": "fake_token_123"},
                metadata={"environment": "production", "region": "us-west-2"}
            )
        },
        {
            "name": "custom_ml_platform",
            "config": MLPlatformConfig(
                platform_type="generic",
                endpoint_url="https://ml-api.company.com",
                credentials={"api_key": "fake_api_key_456"},
                metadata={"vendor": "custom", "version": "2.1"}
            )
        }
    ]
    
    print("Registering ML platforms...")
    for platform in platforms:
        print(f"\nRegistering platform: {platform['name']}")
        print(f"  Type: {platform['config'].platform_type}")
        print(f"  Endpoint: {platform['config'].endpoint_url}")
        
        # In a real scenario, this would attempt actual connections
        # For demo purposes, we'll simulate the registration
        try:
            success = integrator.register_platform(platform['name'], platform['config'])
            if success:
                print(f"  ✓ Successfully registered {platform['name']}")
            else:
                print(f"  ✗ Failed to register {platform['name']} (connection failed)")
        except Exception as e:
            print(f"  ✗ Registration failed: {str(e)}")
    
    return integrator


def demo_model_discovery(integrator: MLPlatformIntegrator):
    """Demonstrate model discovery across platforms"""
    print_section("Model Discovery Demo")
    
    print("Discovering models across all registered platforms...")
    
    # Simulate model discovery results
    mock_models = {
        "local_mlflow": [
            {
                "id": "fraud_detection_v1",
                "name": "Fraud Detection Model",
                "description": "Credit card fraud detection using XGBoost",
                "version": "1.2.3",
                "stage": "Production",
                "accuracy": 0.94,
                "created_at": "2024-01-15T10:30:00Z"
            },
            {
                "id": "customer_churn_v2",
                "name": "Customer Churn Prediction",
                "description": "Predicts customer churn using ensemble methods",
                "version": "2.1.0",
                "stage": "Staging",
                "accuracy": 0.87,
                "created_at": "2024-02-01T14:20:00Z"
            }
        ],
        "production_kubeflow": [
            {
                "id": "recommendation_engine",
                "name": "Product Recommendation Engine",
                "description": "Collaborative filtering recommendation system",
                "version": "3.0.1",
                "status": "Active",
                "precision": 0.82,
                "recall": 0.79,
                "created_at": "2024-01-20T09:15:00Z"
            }
        ],
        "custom_ml_platform": [
            {
                "id": "sentiment_analyzer",
                "name": "Social Media Sentiment Analyzer",
                "description": "BERT-based sentiment analysis for social media",
                "version": "1.5.2",
                "f1_score": 0.91,
                "created_at": "2024-02-10T16:45:00Z"
            }
        ]
    }
    
    for platform_name, models in mock_models.items():
        print_subsection(f"Models from {platform_name}")
        print(f"Found {len(models)} models:")
        
        for model in models:
            print(f"\n  Model: {model['name']}")
            print(f"    ID: {model['id']}")
            print(f"    Version: {model['version']}")
            print(f"    Description: {model['description']}")
            
            # Print performance metrics
            metrics = {k: v for k, v in model.items() 
                      if k in ['accuracy', 'precision', 'recall', 'f1_score']}
            if metrics:
                print(f"    Performance: {metrics}")
    
    return mock_models


def demo_model_deployment(integrator: MLPlatformIntegrator, mock_models: Dict[str, Any]):
    """Demonstrate model deployment"""
    print_section("Model Deployment Demo")
    
    # Select a model for deployment
    model_to_deploy = mock_models["local_mlflow"][0]  # Fraud detection model
    platform_name = "local_mlflow"
    
    print(f"Deploying model: {model_to_deploy['name']}")
    print(f"Platform: {platform_name}")
    print(f"Model ID: {model_to_deploy['id']}")
    print(f"Version: {model_to_deploy['version']}")
    
    # Simulate deployment
    deployment_info = ModelDeploymentInfo(
        model_id=model_to_deploy['id'],
        model_name=model_to_deploy['name'],
        version=model_to_deploy['version'],
        deployment_status="deployed",
        endpoint_url=f"https://api.company.com/models/{model_to_deploy['id']}/predict",
        performance_metrics={
            "accuracy": model_to_deploy.get('accuracy', 0.0),
            "latency_ms": 45,
            "throughput_rps": 1200
        },
        created_at=datetime.now()
    )
    
    print(f"\n✓ Deployment successful!")
    print(f"  Status: {deployment_info.deployment_status}")
    print(f"  Endpoint: {deployment_info.endpoint_url}")
    print(f"  Performance Metrics: {deployment_info.performance_metrics}")
    print(f"  Deployed at: {deployment_info.created_at}")
    
    return deployment_info


def demo_data_quality_correlation(integrator: MLPlatformIntegrator, mock_models: Dict[str, Any]):
    """Demonstrate data quality and model performance correlation analysis"""
    print_section("Data Quality Correlation Analysis Demo")
    
    # Sample dataset quality metrics
    dataset_quality = {
        "dataset_id": "customer_transactions_2024",
        "overall_quality_score": 0.89,
        "quality_dimensions": {
            "completeness": 0.95,
            "accuracy": 0.87,
            "consistency": 0.91,
            "validity": 0.84,
            "uniqueness": 0.93,
            "timeliness": 0.88
        }
    }
    
    print(f"Analyzing correlation for dataset: {dataset_quality['dataset_id']}")
    print(f"Overall quality score: {dataset_quality['overall_quality_score']:.2f}")
    print("\nQuality dimensions:")
    for dimension, score in dataset_quality['quality_dimensions'].items():
        print(f"  {dimension.capitalize()}: {score:.2f}")
    
    print_subsection("Model Performance Correlation")
    
    # Simulate correlation analysis with multiple models
    correlations = []
    
    for platform_name, models in mock_models.items():
        for model in models:
            # Extract performance score
            performance_metrics = {k: v for k, v in model.items() 
                                 if k in ['accuracy', 'precision', 'recall', 'f1_score']}
            
            if performance_metrics:
                performance_score = sum(performance_metrics.values()) / len(performance_metrics)
                
                # Calculate correlation coefficient (simplified)
                quality_score = dataset_quality['overall_quality_score']
                correlation_coeff = 1.0 - abs(quality_score - performance_score)
                correlation_coeff = max(0.0, min(1.0, correlation_coeff))
                
                correlation = DataQualityCorrelation(
                    dataset_id=dataset_quality['dataset_id'],
                    model_id=model['id'],
                    quality_score=quality_score,
                    performance_score=performance_score,
                    correlation_coefficient=correlation_coeff,
                    quality_dimensions=dataset_quality['quality_dimensions'],
                    performance_metrics=performance_metrics,
                    timestamp=datetime.now()
                )
                
                correlations.append(correlation)
                
                print(f"\nModel: {model['name']} ({platform_name})")
                print(f"  Performance Score: {performance_score:.3f}")
                print(f"  Quality Score: {quality_score:.3f}")
                print(f"  Correlation Coefficient: {correlation_coeff:.3f}")
                
                # Provide insights
                if correlation_coeff > 0.8:
                    print(f"  📈 Strong positive correlation - High data quality supports good model performance")
                elif correlation_coeff > 0.6:
                    print(f"  📊 Moderate correlation - Data quality improvements may boost performance")
                else:
                    print(f"  ⚠️  Weak correlation - Model performance may be limited by other factors")
    
    # Summary statistics
    if correlations:
        avg_correlation = sum(c.correlation_coefficient for c in correlations) / len(correlations)
        high_corr_count = sum(1 for c in correlations if c.correlation_coefficient > 0.7)
        
        print_subsection("Correlation Analysis Summary")
        print(f"Total models analyzed: {len(correlations)}")
        print(f"Average correlation coefficient: {avg_correlation:.3f}")
        print(f"Models with high correlation (>0.7): {high_corr_count}")
        print(f"Dataset quality impact: {'High' if avg_correlation > 0.7 else 'Moderate' if avg_correlation > 0.5 else 'Low'}")
    
    return correlations


def demo_platform_monitoring(integrator: MLPlatformIntegrator):
    """Demonstrate platform monitoring and health checks"""
    print_section("Platform Monitoring Demo")
    
    print("Checking health status of all registered platforms...")
    
    # Simulate platform status
    platform_status = {
        "local_mlflow": {
            "connected": True,
            "platform_type": "mlflow",
            "endpoint_url": "http://localhost:5000",
            "model_count": 2,
            "last_sync": "2024-02-15T10:30:00Z",
            "response_time_ms": 120,
            "status": "healthy"
        },
        "production_kubeflow": {
            "connected": True,
            "platform_type": "kubeflow",
            "endpoint_url": "https://kubeflow.company.com",
            "model_count": 1,
            "last_sync": "2024-02-15T10:28:00Z",
            "response_time_ms": 250,
            "status": "healthy"
        },
        "custom_ml_platform": {
            "connected": False,
            "platform_type": "generic",
            "endpoint_url": "https://ml-api.company.com",
            "model_count": 0,
            "last_sync": "2024-02-15T09:45:00Z",
            "response_time_ms": None,
            "status": "disconnected",
            "error": "Connection timeout after 30 seconds"
        }
    }
    
    for platform_name, status in platform_status.items():
        print(f"\nPlatform: {platform_name}")
        print(f"  Status: {'🟢 Connected' if status['connected'] else '🔴 Disconnected'}")
        print(f"  Type: {status['platform_type']}")
        print(f"  Endpoint: {status['endpoint_url']}")
        print(f"  Models: {status['model_count']}")
        print(f"  Last Sync: {status['last_sync']}")
        
        if status['connected']:
            print(f"  Response Time: {status['response_time_ms']}ms")
        else:
            print(f"  Error: {status.get('error', 'Unknown error')}")
    
    # Overall health summary
    connected_count = sum(1 for s in platform_status.values() if s['connected'])
    total_count = len(platform_status)
    total_models = sum(s['model_count'] for s in platform_status.values())
    
    print_subsection("Overall Health Summary")
    print(f"Connected Platforms: {connected_count}/{total_count}")
    print(f"Total Models Available: {total_models}")
    print(f"System Health: {'🟢 Good' if connected_count == total_count else '🟡 Degraded' if connected_count > 0 else '🔴 Critical'}")


def demo_integration_recommendations():
    """Demonstrate integration recommendations and best practices"""
    print_section("Integration Recommendations")
    
    recommendations = [
        {
            "category": "Data Quality Impact",
            "items": [
                "Models with correlation coefficient > 0.8 show strong dependency on data quality",
                "Focus data quality improvements on completeness and accuracy dimensions",
                "Consider automated data quality monitoring for high-correlation models"
            ]
        },
        {
            "category": "Model Performance Optimization",
            "items": [
                "Fraud Detection Model: High correlation (0.92) - prioritize data quality maintenance",
                "Sentiment Analyzer: Moderate correlation (0.67) - investigate feature engineering opportunities",
                "Recommendation Engine: Consider A/B testing with improved data preprocessing"
            ]
        },
        {
            "category": "Platform Management",
            "items": [
                "Set up automated health checks for all ML platforms",
                "Implement alerting for platform connectivity issues",
                "Consider load balancing for high-traffic model endpoints"
            ]
        },
        {
            "category": "Deployment Best Practices",
            "items": [
                "Use blue-green deployments for production models",
                "Implement model versioning and rollback capabilities",
                "Monitor model drift and performance degradation"
            ]
        }
    ]
    
    for rec in recommendations:
        print_subsection(rec['category'])
        for item in rec['items']:
            print(f"  • {item}")


def main():
    """Run the complete ML Platform Integration demo"""
    print("🚀 AI Data Readiness Platform - ML Integration Demo")
    print("This demo showcases ML platform integration capabilities")
    
    try:
        # Demo platform registration
        integrator = demo_platform_registration()
        
        # Demo model discovery
        mock_models = demo_model_discovery(integrator)
        
        # Demo model deployment
        deployment_info = demo_model_deployment(integrator, mock_models)
        
        # Demo data quality correlation analysis
        correlations = demo_data_quality_correlation(integrator, mock_models)
        
        # Demo platform monitoring
        demo_platform_monitoring(integrator)
        
        # Demo recommendations
        demo_integration_recommendations()
        
        print_section("Demo Completed Successfully")
        print("✅ All ML platform integration features demonstrated")
        print("\nKey capabilities showcased:")
        print("  • Multi-platform ML system integration (MLflow, Kubeflow, Generic)")
        print("  • Automated model discovery and cataloging")
        print("  • Model deployment automation")
        print("  • Data quality and model performance correlation analysis")
        print("  • Platform health monitoring and status tracking")
        print("  • Intelligent recommendations for optimization")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()