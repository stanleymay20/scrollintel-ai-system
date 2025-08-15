#!/usr/bin/env python3
"""
Demo script for Task 5.2: Enhanced Transformation and Temporal Feature Generation

This script demonstrates the enhanced capabilities implemented for the AI Data Readiness Platform:
- Advanced transformation pipeline with polynomial features
- Comprehensive temporal feature engineering
- Advanced dimensionality reduction techniques
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

def create_demo_dataset():
    """Create a comprehensive demo dataset with various feature types."""
    np.random.seed(42)
    
    # Create time series data with multiple patterns
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(2000)]  # Hourly data for ~83 days
    
    # Create complex time series with multiple seasonalities
    t = np.arange(len(dates))
    
    data = {
        'timestamp': dates,
        # Sales with daily and weekly patterns
        'sales': (1000 + 
                 200 * np.sin(t * 2 * np.pi / 24) +  # Daily pattern
                 300 * np.sin(t * 2 * np.pi / (24 * 7)) +  # Weekly pattern
                 np.random.normal(0, 50, len(dates))),
        
        # Temperature with daily cycle and trend
        'temperature': (20 + 
                       10 * np.sin(t * 2 * np.pi / 24) +  # Daily cycle
                       0.01 * t +  # Slight warming trend
                       np.random.normal(0, 2, len(dates))),
        
        # Website traffic with complex patterns
        'web_traffic': (500 + 
                       100 * np.sin(t * 2 * np.pi / 24) +  # Daily pattern
                       200 * np.sin(t * 2 * np.pi / (24 * 7)) +  # Weekly pattern
                       50 * np.sin(t * 2 * np.pi / (24 * 30)) +  # Monthly pattern
                       np.random.exponential(50, len(dates))),
        
        # Price with volatility clustering
        'price': np.random.lognormal(3, 0.3, len(dates)),
        
        # Categorical features
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], len(dates)),
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        
        # Binary features
        'is_weekend': [(d.weekday() >= 5) for d in dates],
        'is_holiday': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
        
        # Numerical features with different distributions
        'customer_age': np.random.normal(35, 12, len(dates)),
        'order_value': np.random.exponential(50, len(dates)),
        'discount_rate': np.random.beta(2, 5, len(dates)),
        'inventory_level': np.random.uniform(0, 1000, len(dates)),
        
        # Target variable
        'conversion_rate': np.random.beta(2, 8, len(dates))
    }
    
    return pd.DataFrame(data)

def demo_transformation_pipeline():
    """Demonstrate the enhanced transformation pipeline."""
    print("🔧 ENHANCED TRANSFORMATION PIPELINE DEMO")
    print("=" * 50)
    
    from ai_data_readiness.engines.transformation_pipeline import AdvancedTransformationPipeline, PipelineConfig
    from ai_data_readiness.models.feature_models import ModelType
    
    # Create demo data
    data = create_demo_dataset()
    print(f"📊 Original dataset shape: {data.shape}")
    print(f"📊 Original features: {list(data.columns)}")
    
    # Configure pipeline for comprehensive transformation
    config = PipelineConfig(
        enable_scaling=True,
        enable_encoding=True,
        enable_temporal_features=True,
        enable_interaction_features=True,
        enable_polynomial_features=True,
        enable_dimensionality_reduction=False,
        max_features_after_transformation=200
    )
    
    # Initialize and run pipeline
    pipeline = AdvancedTransformationPipeline(config)
    transformed_data, metadata = pipeline.fit_transform_pipeline(
        data, 
        target_column='conversion_rate',
        model_type=ModelType.RANDOM_FOREST
    )
    
    print(f"\n✨ Transformed dataset shape: {transformed_data.shape}")
    print(f"📈 Feature expansion ratio: {transformed_data.shape[1] / data.shape[1]:.2f}x")
    
    print(f"\n🔄 Transformations applied:")
    for transform in metadata['transformations_applied']:
        print(f"  ✓ {transform}")
    
    # Show polynomial features created
    polynomial_features = [col for col in transformed_data.columns 
                          if any(suffix in col for suffix in ['_squared', '_cubed', '_sqrt', '_log', '_exp', '_reciprocal', '_log1p'])]
    print(f"\n🧮 Polynomial features created: {len(polynomial_features)}")
    for feature in polynomial_features[:10]:  # Show first 10
        print(f"  • {feature}")
    if len(polynomial_features) > 10:
        print(f"  ... and {len(polynomial_features) - 10} more")
    
    # Show interaction features
    interaction_features = [col for col in transformed_data.columns if '_mult' in col or '_add' in col]
    print(f"\n🔗 Interaction features created: {len(interaction_features)}")
    for feature in interaction_features[:5]:  # Show first 5
        print(f"  • {feature}")
    if len(interaction_features) > 5:
        print(f"  ... and {len(interaction_features) - 5} more")
    
    print(f"\n📊 Quality metrics:")
    for metric, value in metadata['quality_metrics'].items():
        print(f"  • {metric}: {value:.3f}")
    
    return transformed_data, metadata

def demo_temporal_feature_generation():
    """Demonstrate comprehensive temporal feature generation."""
    print("\n\n⏰ COMPREHENSIVE TEMPORAL FEATURE GENERATION DEMO")
    print("=" * 55)
    
    from ai_data_readiness.engines.temporal_feature_generator import AdvancedTemporalFeatureGenerator, TemporalConfig
    
    # Create demo data
    data = create_demo_dataset()
    
    # Configure temporal generator
    config = TemporalConfig(
        enable_lag_features=True,
        enable_rolling_features=True,
        enable_seasonal_features=True,
        enable_trend_features=True,
        enable_fourier_features=True,
        max_lag_periods=24,  # 24 hours of lags
        rolling_windows=[3, 6, 12, 24],  # 3h, 6h, 12h, 24h windows
        seasonal_periods=[24, 168]  # Daily and weekly periods
    )
    
    # Initialize generator
    generator = AdvancedTemporalFeatureGenerator(config)
    
    # Generate comprehensive temporal features
    enhanced_data, metadata = generator.generate_comprehensive_temporal_features(
        data,
        time_column='timestamp',
        value_columns=['sales', 'temperature', 'web_traffic'],
        target_column='conversion_rate'
    )
    
    print(f"📊 Original shape: {data.shape}")
    print(f"✨ Enhanced shape: {enhanced_data.shape}")
    print(f"📈 Feature expansion: {enhanced_data.shape[1] / data.shape[1]:.2f}x")
    print(f"🎯 Total temporal features created: {len(metadata['features_created'])}")
    
    print(f"\n📋 Feature categories breakdown:")
    for category, features in metadata['feature_categories'].items():
        print(f"  • {category}: {len(features)} features")
    
    # Demonstrate advanced temporal transformations
    print(f"\n🚀 Advanced temporal transformations:")
    advanced_data, advanced_metadata = generator.create_advanced_temporal_transformations(
        data,
        time_column='timestamp',
        value_columns=['sales', 'web_traffic']
    )
    
    print(f"  ✓ Advanced features created: {len(advanced_metadata['features_created'])}")
    for feature in advanced_metadata['features_created'][:10]:
        print(f"    • {feature}")
    
    # Pattern detection
    print(f"\n🔍 Temporal pattern detection:")
    for column in ['sales', 'temperature', 'web_traffic']:
        patterns = generator.detect_temporal_patterns(data, 'timestamp', column)
        print(f"\n  📈 {column} patterns:")
        
        if 'trend' in patterns:
            trend = patterns['trend']
            print(f"    • Trend: {trend['direction']} (R² = {trend['r_squared']:.3f})")
        
        if 'daily_seasonality' in patterns:
            daily = patterns['daily_seasonality']
            print(f"    • Daily seasonality: {'Present' if daily['present'] else 'Absent'} (strength = {daily['strength']:.3f})")
        
        if 'weekly_seasonality' in patterns:
            weekly = patterns['weekly_seasonality']
            print(f"    • Weekly seasonality: {'Present' if weekly['present'] else 'Absent'} (strength = {weekly['strength']:.3f})")
        
        if 'volatility' in patterns:
            vol = patterns['volatility']
            print(f"    • Volatility: std = {vol['std']:.3f}, max change = {vol['max_change']:.3f}")
    
    return enhanced_data, metadata

def demo_dimensionality_reduction():
    """Demonstrate advanced dimensionality reduction."""
    print("\n\n📉 ADVANCED DIMENSIONALITY REDUCTION DEMO")
    print("=" * 45)
    
    from ai_data_readiness.engines.dimensionality_reduction import AdvancedDimensionalityReducer, DimensionalityReductionConfig
    from ai_data_readiness.models.feature_models import ModelType
    
    # Create high-dimensional data
    data = create_demo_dataset()
    
    # Add many numerical features to demonstrate dimensionality reduction
    np.random.seed(42)
    for i in range(50):
        data[f'feature_{i}'] = np.random.normal(0, 1, len(data))
    
    print(f"📊 High-dimensional dataset shape: {data.shape}")
    
    # Configure reducer
    config = DimensionalityReductionConfig(
        target_variance_ratio=0.95,
        max_components=20,
        enable_feature_selection=True,
        enable_pca=True,
        enable_ica=True,
        correlation_threshold=0.9
    )
    
    # Initialize reducer
    reducer = AdvancedDimensionalityReducer(config)
    
    # Get recommendations
    recommendations = reducer.recommend_dimensionality_reduction(
        data,
        target_column='conversion_rate',
        model_type=ModelType.RANDOM_FOREST
    )
    
    print(f"\n🎯 Dimensionality assessment:")
    assessment = recommendations['dimensionality_assessment']
    print(f"  • Features: {assessment['n_features']}")
    print(f"  • Samples: {assessment['n_samples']}")
    print(f"  • Feature-to-sample ratio: {assessment['feature_to_sample_ratio']:.3f}")
    print(f"  • High dimensionality: {assessment['high_dimensionality']}")
    print(f"  • Curse of dimensionality risk: {assessment['curse_of_dimensionality_risk']}")
    
    print(f"\n💡 Recommended techniques: {recommendations['recommended_techniques']}")
    
    # Apply dimensionality reduction
    reduced_data, reduction_metadata = reducer.apply_dimensionality_reduction(
        data,
        target_column='conversion_rate',
        model_type=ModelType.RANDOM_FOREST
    )
    
    print(f"\n✨ Reduction results:")
    print(f"  • Original shape: {data.shape}")
    print(f"  • Reduced shape: {reduced_data.shape}")
    print(f"  • Techniques applied: {reduction_metadata['techniques_applied']}")
    print(f"  • Feature reduction ratio: {reduction_metadata['quality_metrics'].get('feature_reduction_ratio', 0):.3f}")
    print(f"  • Memory reduction ratio: {reduction_metadata['quality_metrics'].get('memory_reduction_ratio', 0):.3f}")
    
    # Advanced dimensionality features
    print(f"\n🚀 Advanced dimensionality features:")
    advanced_data, advanced_metadata = reducer.create_advanced_dimensionality_features(
        data,
        target_column='conversion_rate'
    )
    
    print(f"  ✓ Advanced features created: {len(advanced_metadata['features_created'])}")
    print(f"  ✓ Techniques applied: {advanced_metadata['techniques_applied']}")
    
    # Optimal dimensions recommendation
    optimal_dims = reducer.recommend_optimal_dimensions(
        data,
        target_column='conversion_rate'
    )
    
    print(f"\n🎯 Optimal dimensions analysis:")
    print(f"  • 90% variance: {optimal_dims['variance_thresholds']['90_percent']} components")
    print(f"  • 95% variance: {optimal_dims['variance_thresholds']['95_percent']} components")
    print(f"  • 99% variance: {optimal_dims['variance_thresholds']['99_percent']} components")
    print(f"  • Elbow method: {optimal_dims['elbow_method']} components")
    print(f"  • Kaiser criterion: {optimal_dims['kaiser_criterion']} components")
    print(f"  • Final recommendation: {optimal_dims['final_recommendation']} components")
    print(f"  • Reduction ratio: {optimal_dims['reduction_ratio']:.3f}")
    
    return reduced_data, reduction_metadata

def demo_integration():
    """Demonstrate integration of all enhanced features."""
    print("\n\n🔗 INTEGRATED FEATURE ENGINEERING DEMO")
    print("=" * 40)
    
    from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
    from ai_data_readiness.models.feature_models import ModelType
    
    # Create demo data
    data = create_demo_dataset()
    
    # Initialize feature engineering engine
    engine = FeatureEngineeringEngine()
    
    # Generate comprehensive feature recommendations
    recommendations = engine.recommend_features(
        dataset_id='demo_dataset',
        data=data,
        model_type=ModelType.RANDOM_FOREST,
        target_column='conversion_rate'
    )
    
    print(f"🎯 Feature engineering recommendations:")
    print(f"  • Total recommendations: {len(recommendations.recommendations)}")
    print(f"  • Encoding strategies: {len(recommendations.encoding_strategies)}")
    print(f"  • Temporal features available: {recommendations.temporal_features is not None}")
    
    # Show top recommendations
    print(f"\n💡 Top feature recommendations:")
    for i, rec in enumerate(recommendations.recommendations[:5]):
        print(f"  {i+1}. {rec.recommendation_type} for '{rec.feature_name}'")
        print(f"     Expected impact: {rec.expected_impact:.2f}, Confidence: {rec.confidence:.2f}")
        print(f"     Rationale: {rec.rationale}")
    
    print(f"\n✅ All enhanced components working together successfully!")
    
    return recommendations

def main():
    """Run the comprehensive demo."""
    print("🚀 AI DATA READINESS PLATFORM - ENHANCED FEATURES DEMO")
    print("=" * 60)
    print("Task 5.2: Build transformation and temporal feature generation")
    print("=" * 60)
    
    try:
        # Run all demos
        transformed_data, transform_metadata = demo_transformation_pipeline()
        enhanced_data, temporal_metadata = demo_temporal_feature_generation()
        reduced_data, reduction_metadata = demo_dimensionality_reduction()
        recommendations = demo_integration()
        
        print("\n\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 35)
        print("✅ Enhanced transformation pipeline working")
        print("✅ Comprehensive temporal feature generation working")
        print("✅ Advanced dimensionality reduction working")
        print("✅ All components integrated successfully")
        print("\n🚀 Task 5.2 implementation is complete and fully functional!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()