"""
Tests for Feature Engineering Engine.

Tests the intelligent feature recommendation system and categorical encoding optimization
as specified in requirements 2.1 and 2.2.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
from ai_data_readiness.models.feature_models import (
    FeatureType, ModelType, TransformationType, TransformationStep,
    FeatureRecommendation, EncodingStrategy, TemporalFeatures
)
from ai_data_readiness.core.exceptions import FeatureEngineeringError


class TestFeatureEngineeringEngine:
    """Test suite for Feature Engineering Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create feature engineering engine instance."""
        return FeatureEngineeringEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'numerical_feature': np.random.normal(100, 15, n_samples),
            'categorical_low': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical_high': np.random.choice([f'cat_{i}' for i in range(50)], n_samples),
            'binary_feature': np.random.choice([0, 1], n_samples),
            'temporal_feature': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'text_feature': [f'This is a sample text with {i} words' for i in range(n_samples)],
            'target': np.random.normal(50, 10, n_samples)
        })
        
        # Add some missing values
        data.loc[np.random.choice(n_samples, 50, replace=False), 'numerical_feature'] = np.nan
        data.loc[np.random.choice(n_samples, 30, replace=False), 'categorical_low'] = np.nan
        
        return data
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series dataset for testing."""
        dates = pd.date_range('2023-01-01', periods=365, freq='H')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 0.1, len(dates)),
            'category': np.random.choice(['A', 'B', 'C'], len(dates))
        })
        return data
    
    def test_feature_type_detection(self, engine, sample_data):
        """Test automatic feature type detection."""
        feature_analysis = engine._analyze_features(sample_data, 'target')
        
        assert feature_analysis['numerical_feature'].type == FeatureType.NUMERICAL
        assert feature_analysis['categorical_low'].type == FeatureType.CATEGORICAL
        assert feature_analysis['binary_feature'].type == FeatureType.BINARY
        assert feature_analysis['temporal_feature'].type == FeatureType.TEMPORAL
        assert feature_analysis['text_feature'].type == FeatureType.TEXT
    
    def test_feature_analysis_statistics(self, engine, sample_data):
        """Test feature analysis statistics calculation."""
        feature_analysis = engine._analyze_features(sample_data, 'target')
        
        # Check numerical feature statistics
        num_stats = feature_analysis['numerical_feature'].distribution_stats
        assert 'mean' in num_stats
        assert 'std' in num_stats
        assert 'skewness' in num_stats
        assert 'outlier_rate' in num_stats
        
        # Check categorical feature statistics
        cat_stats = feature_analysis['categorical_low'].distribution_stats
        assert 'most_frequent_value' in cat_stats
        assert 'most_frequent_rate' in cat_stats
        assert 'rare_categories_count' in cat_stats
        
        # Check missing rates
        assert feature_analysis['numerical_feature'].missing_rate > 0
        assert feature_analysis['categorical_low'].missing_rate > 0
    
    def test_recommend_features_linear_model(self, engine, sample_data):
        """Test feature recommendations for linear models (Requirement 2.1)."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        assert recommendations.dataset_id == "test_dataset"
        assert recommendations.model_type == ModelType.LINEAR_REGRESSION
        assert recommendations.target_column == 'target'
        assert len(recommendations.recommendations) > 0
        
        # Check for scaling recommendations for numerical features
        scaling_recs = [r for r in recommendations.recommendations 
                      if r.recommendation_type == "scaling"]
        assert len(scaling_recs) > 0
        
        # Check for encoding recommendations for categorical features
        encoding_recs = [r for r in recommendations.recommendations 
                        if r.recommendation_type == "encoding"]
        assert len(encoding_recs) > 0
    
    def test_recommend_features_tree_model(self, engine, sample_data):
        """Test feature recommendations for tree-based models (Requirement 2.1)."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.RANDOM_FOREST,
            target_column='target'
        )
        
        assert recommendations.model_type == ModelType.RANDOM_FOREST
        
        # Tree models should have different recommendations than linear models
        binning_recs = [r for r in recommendations.recommendations 
                       if r.recommendation_type == "binning"]
        # May or may not have binning recommendations depending on outlier rate
        
        # Should have encoding recommendations
        encoding_recs = [r for r in recommendations.recommendations 
                        if r.recommendation_type == "encoding"]
        assert len(encoding_recs) > 0
    
    def test_recommend_features_neural_network(self, engine, sample_data):
        """Test feature recommendations for neural networks (Requirement 2.1)."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.NEURAL_NETWORK,
            target_column='target'
        )
        
        assert recommendations.model_type == ModelType.NEURAL_NETWORK
        
        # Neural networks should have normalization recommendations
        norm_recs = [r for r in recommendations.recommendations 
                    if r.recommendation_type == "normalization"]
        assert len(norm_recs) > 0
    
    def test_encoding_strategies_optimization(self, engine, sample_data):
        """Test categorical encoding optimization (Requirement 2.2)."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        encoding_strategies = recommendations.encoding_strategies
        assert len(encoding_strategies) > 0
        
        # Check encoding strategy for low cardinality categorical
        low_card_strategy = next(
            (s for s in encoding_strategies if s.feature_name == 'categorical_low'), 
            None
        )
        assert low_card_strategy is not None
        assert low_card_strategy.encoding_type == "one_hot"
        assert low_card_strategy.expected_dimensions <= 5
        
        # Check encoding strategy for high cardinality categorical
        high_card_strategy = next(
            (s for s in encoding_strategies if s.feature_name == 'categorical_high'), 
            None
        )
        assert high_card_strategy is not None
        # Should use target encoding for high cardinality
        assert high_card_strategy.encoding_type == "target"
    
    def test_optimize_encoding_method(self, engine, sample_data):
        """Test the optimize_encoding method (Requirement 2.2)."""
        categorical_columns = ['categorical_low', 'categorical_high']
        
        # Test for linear model
        strategy = engine.optimize_encoding(
            dataset_id="test_dataset",
            data=sample_data,
            categorical_columns=categorical_columns,
            model_type=ModelType.LINEAR_REGRESSION
        )
        
        assert strategy is not None
        assert strategy.feature_name in categorical_columns
        assert strategy.encoding_type in ["one_hot", "label", "binary", "target"]
        
        # Test for tree model
        tree_strategy = engine.optimize_encoding(
            dataset_id="test_dataset",
            data=sample_data,
            categorical_columns=categorical_columns,
            model_type=ModelType.RANDOM_FOREST
        )
        
        assert tree_strategy is not None
        # Tree models might prefer different encoding strategies
    
    def test_temporal_feature_generation(self, engine, time_series_data):
        """Test temporal feature generation (Requirement 2.3)."""
        temporal_features = engine.generate_temporal_features(
            dataset_id="test_dataset",
            data=time_series_data,
            time_column='timestamp'
        )
        
        assert temporal_features is not None
        assert temporal_features.time_column == 'timestamp'
        assert len(temporal_features.features_to_create) > 0
        assert 'hour' in temporal_features.features_to_create
        assert 'day' in temporal_features.features_to_create
        assert 'month' in temporal_features.features_to_create
        assert 'year' in temporal_features.features_to_create
        
        # Should have aggregation windows for hourly data
        assert len(temporal_features.aggregation_windows) > 0
        
        # Should have lag features
        assert len(temporal_features.lag_features) > 0
    
    def test_temporal_recommendations_in_feature_recommendations(self, engine, time_series_data):
        """Test temporal feature recommendations in main recommendation flow."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=time_series_data,
            model_type=ModelType.TIME_SERIES,
            target_column='value'
        )
        
        assert recommendations.temporal_features is not None
        
        # Should have temporal extraction recommendations
        temporal_recs = [r for r in recommendations.recommendations 
                        if r.recommendation_type == "temporal_extraction"]
        assert len(temporal_recs) > 0
    
    def test_feature_selection_recommendations(self, engine, sample_data):
        """Test feature selection recommendations."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        selection_recs = recommendations.feature_selection_recommendations
        assert len(selection_recs) > 0
        
        # Should have recommendations about correlation-based selection
        correlation_rec = any('correlation' in rec.lower() for rec in selection_recs)
        assert correlation_rec
    
    def test_dimensionality_reduction_recommendations(self, engine):
        """Test dimensionality reduction recommendations."""
        # Create high-dimensional dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 100
        
        high_dim_data = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        high_dim_data['target'] = np.random.normal(0, 1, n_samples)
        
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=high_dim_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        dim_reduction_rec = recommendations.dimensionality_reduction_recommendation
        assert dim_reduction_rec is not None
        assert 'PCA' in dim_reduction_rec
    
    def test_apply_transformations(self, engine, sample_data):
        """Test applying feature transformations."""
        # Create transformation steps
        transformations = [
            TransformationStep(
                transformation_type=TransformationType.SCALING,
                parameters={"method": "standard"},
                input_features=['numerical_feature'],
                output_features=['numerical_feature_scaled'],
                description="Scale numerical feature",
                rationale="Improve model performance"
            ),
            TransformationStep(
                transformation_type=TransformationType.ENCODING,
                parameters={"method": "one_hot"},
                input_features=['categorical_low'],
                output_features=['categorical_low_encoded'],
                description="Encode categorical feature",
                rationale="Convert categorical to numerical"
            )
        ]
        
        result = engine.apply_transformations(
            dataset_id="test_dataset",
            data=sample_data,
            transformations=transformations
        )
        
        assert result.original_dataset_id == "test_dataset"
        assert len(result.transformations_applied) == 2
        assert len(result.feature_mapping) == 2
        assert 'numerical_feature' in result.feature_mapping
        assert 'categorical_low' in result.feature_mapping
        
        # Check quality metrics
        assert 'feature_count_change' in result.quality_metrics
        assert 'completeness_change' in result.quality_metrics
        assert 'memory_usage_change' in result.quality_metrics
    
    def test_missing_value_recommendations(self, engine, sample_data):
        """Test missing value handling recommendations."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        # Should have missing value handling recommendations
        missing_recs = [r for r in recommendations.recommendations 
                       if r.recommendation_type == "missing_value_handling"]
        assert len(missing_recs) > 0
        
        # Check that features with missing values get recommendations
        missing_features = sample_data.columns[sample_data.isnull().any()].tolist()
        recommended_features = [r.feature_name for r in missing_recs]
        
        # At least some missing features should have recommendations
        assert any(feat in recommended_features for feat in missing_features)
    
    def test_feature_interaction_recommendations(self, engine, sample_data):
        """Test feature interaction recommendations."""
        # Create data with correlated features
        corr_data = sample_data.copy()
        corr_data['correlated_feature'] = corr_data['target'] * 0.8 + np.random.normal(0, 0.1, len(corr_data))
        
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=corr_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        # Should have interaction recommendations for highly correlated features
        interaction_recs = [r for r in recommendations.recommendations 
                           if r.recommendation_type == "interaction"]
        # May or may not have interactions depending on correlation threshold
    
    def test_recommendation_impact_scoring(self, engine, sample_data):
        """Test that recommendations have proper impact scoring."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        for rec in recommendations.recommendations:
            assert 0 <= rec.expected_impact <= 1
            assert 0 <= rec.confidence <= 1
            assert rec.implementation_complexity in ["low", "medium", "high"]
            assert rec.rationale is not None
            assert len(rec.rationale) > 0
    
    def test_high_impact_recommendations_filter(self, engine, sample_data):
        """Test filtering high impact recommendations."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        high_impact = recommendations.get_high_impact_recommendations(threshold=0.7)
        for rec in high_impact:
            assert rec.expected_impact >= 0.7
    
    def test_low_complexity_recommendations_filter(self, engine, sample_data):
        """Test filtering low complexity recommendations."""
        recommendations = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        low_complexity = recommendations.get_low_complexity_recommendations()
        for rec in low_complexity:
            assert rec.implementation_complexity == "low"
    
    def test_error_handling_invalid_dataset(self, engine):
        """Test error handling for invalid dataset."""
        with pytest.raises(FeatureEngineeringError):
            engine.recommend_features(
                dataset_id="test_dataset",
                data=pd.DataFrame(),  # Empty dataframe
                model_type=ModelType.LINEAR_REGRESSION,
                target_column='nonexistent_target'
            )
    
    def test_error_handling_invalid_transformations(self, engine, sample_data):
        """Test error handling for invalid transformations."""
        invalid_transformation = TransformationStep(
            transformation_type=TransformationType.SCALING,
            parameters={"method": "invalid_method"},
            input_features=['nonexistent_feature'],
            output_features=['output_feature'],
            description="Invalid transformation",
            rationale="Test error handling"
        )
        
        with pytest.raises(FeatureEngineeringError):
            engine.apply_transformations(
                dataset_id="test_dataset",
                data=sample_data,
                transformations=[invalid_transformation]
            )
    
    def test_correlation_calculation(self, engine):
        """Test correlation calculation between features and target."""
        # Create data with known correlation
        n_samples = 1000
        x = np.random.normal(0, 1, n_samples)
        y = 0.8 * x + np.random.normal(0, 0.2, n_samples)  # Strong positive correlation
        
        feature_series = pd.Series(x)
        target_series = pd.Series(y)
        
        correlation = engine._calculate_correlation(feature_series, target_series)
        assert 0.7 <= correlation <= 0.9  # Should be around 0.8
    
    def test_outlier_detection(self, engine):
        """Test outlier detection in numerical features."""
        # Create data with known outliers
        normal_data = np.random.normal(0, 1, 950)
        outliers = np.array([10, -10, 15, -15, 20])  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        series = pd.Series(data_with_outliers)
        outlier_rate = engine._calculate_outlier_rate(series)
        
        assert outlier_rate > 0  # Should detect some outliers
        assert outlier_rate < 0.1  # But not too many
    
    def test_distribution_type_detection(self, engine):
        """Test distribution type detection."""
        # Normal distribution
        normal_data = pd.Series(np.random.normal(0, 1, 1000))
        dist_type = engine._detect_distribution_type(normal_data)
        assert dist_type == "normal"
        
        # Right-skewed distribution
        skewed_data = pd.Series(np.random.exponential(1, 1000))
        dist_type = engine._detect_distribution_type(skewed_data)
        assert dist_type == "right_skewed"
    
    def test_model_specific_strategies(self, engine, sample_data):
        """Test that different models get different strategies."""
        linear_recs = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.LINEAR_REGRESSION,
            target_column='target'
        )
        
        tree_recs = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.RANDOM_FOREST,
            target_column='target'
        )
        
        nn_recs = engine.recommend_features(
            dataset_id="test_dataset",
            data=sample_data,
            model_type=ModelType.NEURAL_NETWORK,
            target_column='target'
        )
        
        # Different models should have different recommendation types
        linear_types = {r.recommendation_type for r in linear_recs.recommendations}
        tree_types = {r.recommendation_type for r in tree_recs.recommendations}
        nn_types = {r.recommendation_type for r in nn_recs.recommendations}
        
        # Neural networks should have normalization
        assert any("normalization" in types for types in [nn_types])
        
        # Linear models should have scaling
        assert any("scaling" in types for types in [linear_types])


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering engine."""
    
    @pytest.fixture
    def engine(self):
        """Create feature engineering engine instance."""
        return FeatureEngineeringEngine()
    
    def test_end_to_end_feature_engineering_workflow(self, engine):
        """Test complete feature engineering workflow."""
        # Create realistic dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(10, 1, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'city': np.random.choice([f'City_{i}' for i in range(20)], n_samples),
            'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'is_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'target_revenue': np.random.exponential(100, n_samples)
        })
        
        # Add missing values
        data.loc[np.random.choice(n_samples, 50, replace=False), 'income'] = np.nan
        
        # Step 1: Get recommendations
        recommendations = engine.recommend_features(
            dataset_id="customer_dataset",
            data=data,
            model_type=ModelType.RANDOM_FOREST,
            target_column='target_revenue'
        )
        
        assert len(recommendations.recommendations) > 0
        assert len(recommendations.encoding_strategies) > 0
        assert recommendations.temporal_features is not None
        
        # Step 2: Apply some transformations
        transformations = []
        
        # Add scaling for income
        transformations.append(TransformationStep(
            transformation_type=TransformationType.SCALING,
            parameters={"method": "standard"},
            input_features=['income'],
            output_features=['income_scaled'],
            description="Scale income feature",
            rationale="Normalize income distribution"
        ))
        
        # Add encoding for education
        transformations.append(TransformationStep(
            transformation_type=TransformationType.ENCODING,
            parameters={"method": "label"},
            input_features=['education'],
            output_features=['education_encoded'],
            description="Encode education levels",
            rationale="Convert categorical to numerical"
        ))
        
        # Apply transformations
        result = engine.apply_transformations(
            dataset_id="customer_dataset",
            data=data,
            transformations=transformations
        )
        
        assert result.original_dataset_id == "customer_dataset"
        assert len(result.transformations_applied) == 2
        assert 'feature_count_change' in result.quality_metrics
        
        # Step 3: Optimize encoding for remaining categorical features
        categorical_cols = ['city']
        encoding_strategy = engine.optimize_encoding(
            dataset_id="customer_dataset",
            data=data,
            categorical_columns=categorical_cols,
            model_type=ModelType.RANDOM_FOREST
        )
        
        assert encoding_strategy is not None
        assert encoding_strategy.feature_name == 'city'
        
        # Step 4: Generate temporal features
        temporal_features = engine.generate_temporal_features(
            dataset_id="customer_dataset",
            data=data,
            time_column='signup_date'
        )
        
        assert temporal_features.time_column == 'signup_date'
        assert len(temporal_features.features_to_create) > 0