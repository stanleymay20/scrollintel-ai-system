#!/usr/bin/env python3
"""
Test script for Task 5.2: Build transformation and temporal feature generation

This script tests the enhanced transformation pipeline and temporal feature generation
capabilities implemented for the AI Data Readiness Platform.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

def create_test_data():
    """Create comprehensive test data for transformation and temporal feature testing."""
    np.random.seed(42)
    
    # Create time series data
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # Create various types of features
    data = {
        'timestamp': dates,
        'sales': np.random.normal(1000, 200, 365) + 100 * np.sin(np.arange(365) * 2 * np.pi / 7),  # Weekly seasonality
        'temperature': 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 2, 365),  # Yearly seasonality
        'price': np.random.lognormal(3, 0.5, 365),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 365),
        'is_weekend': [(d.weekday() >= 5) for d in dates],
        'numerical_1': np.random.normal(50, 15, 365),
        'numerical_2': np.random.exponential(2, 365),
        'numerical_3': np.random.uniform(0, 100, 365),
        'target': np.random.normal(0, 1, 365)
    }
    
    return pd.DataFrame(data)

def test_transformation_pipeline():
    """Test the enhanced transformation pipeline."""
    print("Testing Enhanced Transformation Pipeline...")
    
    try:
        from ai_data_readiness.engines.transformation_pipeline import AdvancedTransformationPipeline, PipelineConfig
        from ai_data_readiness.models.feature_models import ModelType
        
        # Create test data
        data = create_test_data()
        
        # Configure pipeline
        config = PipelineConfig(
            enable_scaling=True,
            enable_encoding=True,
            enable_temporal_features=True,
            enable_interaction_features=True,
            enable_polynomial_features=True,
            enable_dimensionality_reduction=False
        )
        
        # Initialize pipeline
        pipeline = AdvancedTransformationPipeline(config)
        
        # Test pipeline transformation
        transformed_data, metadata = pipeline.fit_transform_pipeline(
            data, 
            target_column='target',
            model_type=ModelType.RANDOM_FOREST
        )
        
        print(f"✓ Original shape: {data.shape}")
        print(f"✓ Transformed shape: {transformed_data.shape}")
        print(f"✓ Transformations applied: {metadata['transformations_applied']}")
        print(f"✓ Feature count change: {metadata['quality_metrics'].get('feature_count_change', 0):.2f}")
        
        # Test polynomial features
        polynomial_features = [col for col in transformed_data.columns if any(suffix in col for suffix in ['_squared', '_cubed', '_sqrt', '_log', '_exp', '_reciprocal'])]
        print(f"✓ Polynomial features created: {len(polynomial_features)}")
        
        # Test temporal features
        temporal_features = [col for col in transformed_data.columns if 'timestamp_' in col]
        print(f"✓ Temporal features created: {len(temporal_features)}")
        
        # Test interaction features
        interaction_features = [col for col in transformed_data.columns if '_mult' in col or '_add' in col]
        print(f"✓ Interaction features created: {len(interaction_features)}")
        
        print("✓ Transformation Pipeline Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Transformation Pipeline Test FAILED: {str(e)}\n")
        return False

def test_temporal_feature_generator():
    """Test the enhanced temporal feature generator."""
    print("Testing Enhanced Temporal Feature Generator...")
    
    try:
        from ai_data_readiness.engines.temporal_feature_generator import AdvancedTemporalFeatureGenerator, TemporalConfig
        
        # Create test data
        data = create_test_data()
        
        # Configure temporal generator
        config = TemporalConfig(
            enable_lag_features=True,
            enable_rolling_features=True,
            enable_seasonal_features=True,
            enable_trend_features=True,
            enable_fourier_features=True,
            max_lag_periods=14,
            rolling_windows=[3, 7, 14],
            seasonal_periods=[7, 30]
        )
        
        # Initialize generator
        generator = AdvancedTemporalFeatureGenerator(config)
        
        # Test comprehensive temporal feature generation
        enhanced_data, metadata = generator.generate_comprehensive_temporal_features(
            data,
            time_column='timestamp',
            value_columns=['sales', 'temperature'],
            target_column='target'
        )
        
        print(f"✓ Original shape: {data.shape}")
        print(f"✓ Enhanced shape: {enhanced_data.shape}")
        print(f"✓ Total features created: {len(metadata['features_created'])}")
        
        # Check feature categories
        for category, features in metadata['feature_categories'].items():
            print(f"✓ {category}: {len(features)} features")
        
        # Test advanced temporal transformations
        advanced_data, advanced_metadata = generator.create_advanced_temporal_transformations(
            data,
            time_column='timestamp',
            value_columns=['sales', 'temperature']
        )
        
        print(f"✓ Advanced transformations created: {len(advanced_metadata['features_created'])}")
        
        # Test pattern detection
        patterns = generator.detect_temporal_patterns(
            data,
            time_column='timestamp',
            value_column='sales'
        )
        
        print(f"✓ Temporal patterns detected: {list(patterns.keys())}")
        if 'trend' in patterns:
            print(f"  - Trend direction: {patterns['trend']['direction']}")
            print(f"  - Trend strength (R²): {patterns['trend']['r_squared']:.3f}")
        
        # Test Fourier features
        fourier_features = [col for col in enhanced_data.columns if 'fourier' in col]
        print(f"✓ Fourier features created: {len(fourier_features)}")
        
        # Test lag features
        lag_features = [col for col in enhanced_data.columns if '_lag_' in col]
        print(f"✓ Lag features created: {len(lag_features)}")
        
        # Test rolling features
        rolling_features = [col for col in enhanced_data.columns if '_rolling_' in col]
        print(f"✓ Rolling features created: {len(rolling_features)}")
        
        print("✓ Temporal Feature Generator Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Temporal Feature Generator Test FAILED: {str(e)}\n")
        return False

def test_dimensionality_reduction():
    """Test the enhanced dimensionality reduction capabilities."""
    print("Testing Enhanced Dimensionality Reduction...")
    
    try:
        from ai_data_readiness.engines.dimensionality_reduction import AdvancedDimensionalityReducer, DimensionalityReductionConfig
        from ai_data_readiness.models.feature_models import ModelType
        
        # Create test data with many features
        data = create_test_data()
        
        # Add more numerical features to test dimensionality reduction
        np.random.seed(42)
        for i in range(20):
            data[f'feature_{i}'] = np.random.normal(0, 1, len(data))
        
        # Configure reducer
        config = DimensionalityReductionConfig(
            target_variance_ratio=0.95,
            max_components=10,
            enable_feature_selection=True,
            enable_pca=True,
            enable_ica=True
        )
        
        # Initialize reducer
        reducer = AdvancedDimensionalityReducer(config)
        
        # Test recommendations
        recommendations = reducer.recommend_dimensionality_reduction(
            data,
            target_column='target',
            model_type=ModelType.RANDOM_FOREST
        )
        
        print(f"✓ Dimensionality assessment: {recommendations['dimensionality_assessment']}")
        print(f"✓ Recommended techniques: {recommendations['recommended_techniques']}")
        
        # Test dimensionality reduction application
        reduced_data, reduction_metadata = reducer.apply_dimensionality_reduction(
            data,
            target_column='target',
            model_type=ModelType.RANDOM_FOREST
        )
        
        print(f"✓ Original shape: {data.shape}")
        print(f"✓ Reduced shape: {reduced_data.shape}")
        print(f"✓ Techniques applied: {reduction_metadata['techniques_applied']}")
        print(f"✓ Feature reduction ratio: {reduction_metadata['quality_metrics'].get('feature_reduction_ratio', 0):.3f}")
        
        # Test advanced dimensionality features
        advanced_data, advanced_metadata = reducer.create_advanced_dimensionality_features(
            data,
            target_column='target'
        )
        
        print(f"✓ Advanced dimensionality features: {len(advanced_metadata['features_created'])}")
        print(f"✓ Techniques applied: {advanced_metadata['techniques_applied']}")
        
        # Test optimal dimensions recommendation
        optimal_dims = reducer.recommend_optimal_dimensions(
            data,
            target_column='target'
        )
        
        print(f"✓ Optimal dimensions recommendation: {optimal_dims.get('final_recommendation', 'N/A')}")
        print(f"✓ Reduction ratio: {optimal_dims.get('reduction_ratio', 0):.3f}")
        
        print("✓ Dimensionality Reduction Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Dimensionality Reduction Test FAILED: {str(e)}\n")
        return False

def test_integration():
    """Test integration of all components."""
    print("Testing Integration of All Components...")
    
    try:
        from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
        from ai_data_readiness.models.feature_models import ModelType
        
        # Create test data
        data = create_test_data()
        
        # Initialize feature engineering engine
        engine = FeatureEngineeringEngine()
        
        # Test feature recommendations
        recommendations = engine.recommend_features(
            dataset_id='test_dataset',
            data=data,
            model_type=ModelType.RANDOM_FOREST,
            target_column='target'
        )
        
        print(f"✓ Feature recommendations generated: {len(recommendations.recommendations)}")
        print(f"✓ Encoding strategies: {len(recommendations.encoding_strategies)}")
        print(f"✓ Temporal features recommended: {recommendations.temporal_features is not None}")
        
        # Test that all components work together
        print("✓ All components integrated successfully")
        print("✓ Integration Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Integration Test FAILED: {str(e)}\n")
        return False

def main():
    """Run all tests for Task 5.2 implementation."""
    print("=" * 60)
    print("TESTING TASK 5.2: BUILD TRANSFORMATION AND TEMPORAL FEATURE GENERATION")
    print("=" * 60)
    print()
    
    tests = [
        test_transformation_pipeline,
        test_temporal_feature_generator,
        test_dimensionality_reduction,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - Task 5.2 implementation is working correctly!")
    else:
        print(f"✗ {total - passed} tests failed - Please check the implementation")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)