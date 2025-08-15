"""
Demo script for Feature Engineering Engine.

This script demonstrates the intelligent feature recommendation system
as specified in requirements 2.1 and 2.2.
"""

import pandas as pd
import numpy as np
from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
from ai_data_readiness.models.feature_models import ModelType

def main():
    """Demonstrate feature engineering engine capabilities."""
    print("=== AI Data Readiness Platform - Feature Engineering Engine Demo ===\n")
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'numerical_feature': np.random.normal(100, 15, n_samples),
        'categorical_low': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_high': np.random.choice([f'cat_{i}' for i in range(50)], n_samples),
        'binary_feature': np.random.choice([0, 1], n_samples),
        'temporal_feature': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'target': np.random.normal(50, 10, n_samples)
    })
    
    # Add some missing values
    data.loc[np.random.choice(n_samples, 50, replace=False), 'numerical_feature'] = np.nan
    data.loc[np.random.choice(n_samples, 30, replace=False), 'categorical_low'] = np.nan
    
    print(f"Dataset shape: {data.shape}")
    print(f"Dataset columns: {list(data.columns)}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print()
    
    # Initialize feature engineering engine
    engine = FeatureEngineeringEngine()
    
    # Test 1: Feature recommendations for linear model (Requirement 2.1)
    print("=== Test 1: Feature Recommendations for Linear Model (Requirement 2.1) ===")
    
    recommendations = engine.recommend_features(
        dataset_id="demo_dataset",
        data=data,
        model_type=ModelType.LINEAR_REGRESSION,
        target_column='target'
    )
    
    print(f"Total recommendations: {len(recommendations.recommendations)}")
    print(f"Encoding strategies: {len(recommendations.encoding_strategies)}")
    print(f"Temporal features available: {recommendations.temporal_features is not None}")
    print()
    
    # Show top recommendations
    print("Top 5 Feature Recommendations:")
    for i, rec in enumerate(recommendations.recommendations[:5]):
        print(f"{i+1}. {rec.feature_name} - {rec.recommendation_type}")
        print(f"   Impact: {rec.expected_impact:.2f}, Confidence: {rec.confidence:.2f}")
        print(f"   Rationale: {rec.rationale}")
        print()
    
    # Test 2: Encoding strategies optimization (Requirement 2.2)
    print("=== Test 2: Encoding Strategies Optimization (Requirement 2.2) ===")
    
    print("Encoding Strategies:")
    for strategy in recommendations.encoding_strategies:
        print(f"- {strategy.feature_name}: {strategy.encoding_type}")
        print(f"  Expected dimensions: {strategy.expected_dimensions}")
        print(f"  Parameters: {strategy.parameters}")
        print()
    
    # Test 3: Different model types get different recommendations
    print("=== Test 3: Model-Specific Recommendations ===")
    
    models_to_test = [
        ModelType.LINEAR_REGRESSION,
        ModelType.RANDOM_FOREST,
        ModelType.NEURAL_NETWORK
    ]
    
    for model_type in models_to_test:
        model_recs = engine.recommend_features(
            dataset_id="demo_dataset",
            data=data,
            model_type=model_type,
            target_column='target'
        )
        
        rec_types = {r.recommendation_type for r in model_recs.recommendations}
        print(f"{model_type.value}: {len(model_recs.recommendations)} recommendations")
        print(f"  Types: {', '.join(rec_types)}")
        print()
    
    # Test 4: Feature analysis capabilities
    print("=== Test 4: Feature Analysis Capabilities ===")
    
    feature_analysis = engine._analyze_features(data, 'target')
    
    print("Feature Analysis Summary:")
    for feature_name, info in feature_analysis.items():
        if feature_name != 'target':
            print(f"- {feature_name}: {info.type.value}")
            print(f"  Missing rate: {info.missing_rate:.3f}")
            print(f"  Unique values: {info.unique_values}")
            if info.correlation_with_target:
                print(f"  Correlation with target: {info.correlation_with_target:.3f}")
            print()
    
    print("=== Demo Complete ===")
    print("\nThe Feature Engineering Engine successfully demonstrates:")
    print("✓ Requirement 2.1: Intelligent feature transformations based on data types and target variables")
    print("✓ Requirement 2.2: Optimal encoding strategies for categorical variables")
    print("✓ Model-specific recommendations for different ML algorithms")
    print("✓ Comprehensive feature analysis and quality assessment")

if __name__ == "__main__":
    main()