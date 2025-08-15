"""
Unit tests for Feature Engineering Engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
from ai_data_readiness.core.exceptions import AIDataReadinessError


class TestFeatureEngineeringEngine:
    """Test suite for FeatureEngineeringEngine."""
    
    @pytest.fixture
    def feature_engine(self, test_config):
        """Create FeatureEngineeringEngine instance for testing."""
        return FeatureEngineeringEngine(test_config)
    
    def test_init(self, test_config):
        """Test FeatureEngineeringEngine initialization."""
        engine = FeatureEngineeringEngine(test_config)
        assert engine.config == test_config
        assert hasattr(engine, 'transformers')
        assert hasattr(engine, 'feature_selectors')
    
    def test_recommend_features_classification(self, feature_engine, sample_csv_data):
        """Test feature recommendations for classification tasks."""
        dataset_id = "classification_dataset"
        model_type = "classification"
        
        # Add a binary target for classification
        sample_csv_data['target'] = np.random.choice([0, 1], len(sample_csv_data))
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            recommendations = feature_engine.recommend_features(dataset_id, model_type)
            
            assert 'recommended_features' in recommendations
            assert 'transformations' in recommendations
            assert 'feature_importance' in recommendations
            assert 'encoding_strategies' in recommendations
            
            # Should recommend appropriate transformations
            transformations = recommendations['transformations']
            assert len(transformations) > 0
            
            # Should include encoding for categorical variables
            encoding_strategies = recommendations['encoding_strategies']
            assert 'category' in encoding_strategies  # categorical column
            assert 'gender' in encoding_strategies    # categorical column
    
    def test_recommend_features_regression(self, feature_engine, sample_csv_data):
        """Test feature recommendations for regression tasks."""
        dataset_id = "regression_dataset"
        model_type = "regression"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            recommendations = feature_engine.recommend_features(dataset_id, model_type)
            
            assert 'recommended_features' in recommendations
            assert 'transformations' in recommendations
            
            # Should recommend different transformations for regression
            transformations = recommendations['transformations']
            transformation_types = [t['type'] for t in transformations]
            
            # Common regression transformations
            expected_transformations = ['scaling', 'normalization', 'polynomial_features']
            for transform in expected_transformations:
                assert any(transform in t_type for t_type in transformation_types)
    
    def test_apply_transformations(self, feature_engine, sample_csv_data):
        """Test applying feature transformations."""
        dataset_id = "test_dataset"
        
        transformations = [
            {
                'type': 'standard_scaling',
                'columns': ['age', 'income', 'score'],
                'parameters': {}
            },
            {
                'type': 'one_hot_encoding',
                'columns': ['category', 'gender'],
                'parameters': {'drop_first': True}
            },
            {
                'type': 'log_transform',
                'columns': ['income'],
                'parameters': {}
            }
        ]
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            result = feature_engine.apply_transformations(dataset_id, transformations)
            
            assert 'transformed_dataset_id' in result
            assert 'transformation_summary' in result
            assert 'new_features' in result
            assert 'removed_features' in result
            
            # Check transformation summary
            summary = result['transformation_summary']
            assert len(summary) == len(transformations)
            
            for i, transform_summary in enumerate(summary):
                assert 'type' in transform_summary
                assert 'columns' in transform_summary
                assert 'status' in transform_summary
                assert transform_summary['type'] == transformations[i]['type']
    
    def test_optimize_categorical_encoding(self, feature_engine):
        """Test categorical encoding optimization."""
        # Create data with different types of categorical variables
        categorical_data = pd.DataFrame({
            'low_cardinality': np.random.choice(['A', 'B', 'C'], 1000),
            'medium_cardinality': np.random.choice([f'Cat_{i}' for i in range(20)], 1000),
            'high_cardinality': np.random.choice([f'Item_{i}' for i in range(200)], 1000),
            'ordinal_cat': np.random.choice(['Low', 'Medium', 'High'], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        dataset_id = "categorical_dataset"
        categorical_columns = ['low_cardinality', 'medium_cardinality', 'high_cardinality', 'ordinal_cat']
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = categorical_data
            
            encoding_strategy = feature_engine.optimize_encoding(dataset_id, categorical_columns)
            
            assert 'encoding_recommendations' in encoding_strategy
            assert 'cardinality_analysis' in encoding_strategy
            assert 'performance_estimates' in encoding_strategy
            
            recommendations = encoding_strategy['encoding_recommendations']
            
            # Should recommend different encodings based on cardinality
            for col in categorical_columns:
                assert col in recommendations
                assert 'method' in recommendations[col]
                assert 'reason' in recommendations[col]
            
            # Low cardinality should use one-hot encoding
            assert recommendations['low_cardinality']['method'] == 'one_hot'
            
            # High cardinality should use target encoding or similar
            high_card_method = recommendations['high_cardinality']['method']
            assert high_card_method in ['target_encoding', 'frequency_encoding', 'embedding']
    
    def test_generate_temporal_features(self, feature_engine, sample_time_series_data):
        """Test temporal feature generation."""
        dataset_id = "time_series_dataset"
        time_column = 'date'
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_time_series_data
            
            temporal_features = feature_engine.generate_temporal_features(dataset_id, time_column)
            
            assert 'new_features' in temporal_features
            assert 'feature_descriptions' in temporal_features
            assert 'temporal_patterns' in temporal_features
            
            new_features = temporal_features['new_features']
            
            # Should generate common temporal features
            expected_features = [
                'year', 'month', 'day', 'day_of_week', 'quarter',
                'is_weekend', 'is_month_start', 'is_month_end'
            ]
            
            for feature in expected_features:
                assert any(feature in f for f in new_features)
            
            # Should detect temporal patterns
            patterns = temporal_features['temporal_patterns']
            assert 'seasonality' in patterns
            assert 'trend' in patterns
    
    def test_feature_selection_correlation(self, feature_engine):
        """Test feature selection based on correlation."""
        # Create data with correlated features
        np.random.seed(42)
        n_samples = 1000
        
        base_feature = np.random.normal(0, 1, n_samples)
        correlated_data = pd.DataFrame({
            'feature_1': base_feature,
            'feature_2': base_feature + np.random.normal(0, 0.1, n_samples),  # Highly correlated
            'feature_3': base_feature * 2 + np.random.normal(0, 0.2, n_samples),  # Correlated
            'feature_4': np.random.normal(0, 1, n_samples),  # Independent
            'feature_5': np.random.normal(0, 1, n_samples),  # Independent
            'target': base_feature + np.random.normal(0, 0.5, n_samples)
        })
        
        dataset_id = "correlated_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = correlated_data
            
            selection_result = feature_engine._select_features_by_correlation(
                dataset_id, correlation_threshold=0.8
            )
            
            assert 'selected_features' in selection_result
            assert 'removed_features' in selection_result
            assert 'correlation_matrix' in selection_result
            
            # Should remove highly correlated features
            removed_features = selection_result['removed_features']
            assert len(removed_features) > 0
            
            # Should keep the most important correlated feature
            selected_features = selection_result['selected_features']
            assert 'feature_1' in selected_features or 'feature_2' in selected_features
    
    def test_feature_importance_analysis(self, feature_engine, sample_csv_data):
        """Test feature importance analysis."""
        dataset_id = "importance_dataset"
        target_column = 'score'  # Use score as target for regression
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            importance_analysis = feature_engine._analyze_feature_importance(
                dataset_id, target_column
            )
            
            assert 'feature_importance' in importance_analysis
            assert 'importance_method' in importance_analysis
            assert 'feature_ranking' in importance_analysis
            
            feature_importance = importance_analysis['feature_importance']
            
            # Should have importance scores for all features
            numeric_features = ['age', 'income']
            for feature in numeric_features:
                assert feature in feature_importance
                assert 0 <= feature_importance[feature] <= 1
    
    def test_polynomial_feature_generation(self, feature_engine):
        """Test polynomial feature generation."""
        # Create simple dataset for polynomial features
        simple_data = pd.DataFrame({
            'x1': np.random.uniform(-2, 2, 100),
            'x2': np.random.uniform(-2, 2, 100)
        })
        
        # Add target with polynomial relationship
        simple_data['target'] = (simple_data['x1']**2 + 
                               simple_data['x2']**2 + 
                               simple_data['x1'] * simple_data['x2'] + 
                               np.random.normal(0, 0.1, 100))
        
        dataset_id = "polynomial_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = simple_data
            
            poly_features = feature_engine._generate_polynomial_features(
                dataset_id, degree=2, columns=['x1', 'x2']
            )
            
            assert 'new_features' in poly_features
            assert 'feature_names' in poly_features
            assert 'polynomial_degree' in poly_features
            
            # Should generate polynomial combinations
            feature_names = poly_features['feature_names']
            expected_features = ['x1^2', 'x2^2', 'x1*x2']
            
            for expected in expected_features:
                assert any(expected in name for name in feature_names)
    
    def test_interaction_feature_detection(self, feature_engine):
        """Test interaction feature detection."""
        # Create data with interaction effects
        np.random.seed(42)
        n_samples = 1000
        
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        x3 = np.random.normal(0, 1, n_samples)
        
        # Target depends on interaction between x1 and x2
        target = x1 + x2 + 2 * x1 * x2 + 0.5 * x3 + np.random.normal(0, 0.1, n_samples)
        
        interaction_data = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'target': target
        })
        
        dataset_id = "interaction_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = interaction_data
            
            interactions = feature_engine._detect_feature_interactions(
                dataset_id, 'target'
            )
            
            assert 'significant_interactions' in interactions
            assert 'interaction_scores' in interactions
            assert 'recommended_interactions' in interactions
            
            # Should detect the x1*x2 interaction
            significant = interactions['significant_interactions']
            assert len(significant) > 0
            
            # Check if x1-x2 interaction is detected
            x1_x2_found = any(
                ('x1' in interaction and 'x2' in interaction) 
                for interaction in significant
            )
            assert x1_x2_found
    
    def test_dimensionality_reduction_recommendations(self, feature_engine):
        """Test dimensionality reduction recommendations."""
        # Create high-dimensional data
        np.random.seed(42)
        n_samples = 500
        n_features = 50
        
        # Create correlated features
        base_features = np.random.normal(0, 1, (n_samples, 5))
        noise_features = np.random.normal(0, 0.1, (n_samples, n_features - 5))
        
        # Mix base features to create correlations
        mixed_features = np.column_stack([
            base_features,
            base_features[:, :10] + noise_features[:, :10],  # Correlated features
            base_features[:, :15] * 0.5 + noise_features[:, 10:25],  # More correlations
            noise_features[:, 25:]  # Pure noise
        ])
        
        high_dim_data = pd.DataFrame(
            mixed_features,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        high_dim_data['target'] = (
            base_features[:, 0] + base_features[:, 1] + 
            np.random.normal(0, 0.1, n_samples)
        )
        
        dataset_id = "high_dim_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = high_dim_data
            
            dim_reduction = feature_engine._recommend_dimensionality_reduction(
                dataset_id, target_column='target'
            )
            
            assert 'recommended_method' in dim_reduction
            assert 'explained_variance_ratio' in dim_reduction
            assert 'recommended_components' in dim_reduction
            assert 'method_comparison' in dim_reduction
            
            # Should recommend appropriate method
            recommended_method = dim_reduction['recommended_method']
            assert recommended_method in ['pca', 'lda', 'feature_selection', 'umap']
    
    def test_feature_scaling_recommendations(self, feature_engine, sample_csv_data):
        """Test feature scaling recommendations."""
        dataset_id = "scaling_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            scaling_recs = feature_engine._recommend_feature_scaling(dataset_id)
            
            assert 'scaling_recommendations' in scaling_recs
            assert 'feature_distributions' in scaling_recs
            assert 'scaling_methods' in scaling_recs
            
            recommendations = scaling_recs['scaling_recommendations']
            
            # Should analyze each numeric column
            numeric_columns = ['age', 'income', 'score']
            for col in numeric_columns:
                assert col in recommendations
                assert 'method' in recommendations[col]
                assert 'reason' in recommendations[col]
                
                # Should recommend appropriate scaling method
                method = recommendations[col]['method']
                assert method in ['standard', 'minmax', 'robust', 'quantile', 'none']
    
    def test_performance_with_large_dataset(self, feature_engine, performance_test_data, performance_timer):
        """Test performance with large dataset."""
        dataset_id = "large_feature_dataset"
        model_type = "classification"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = performance_test_data
            
            with performance_timer:
                recommendations = feature_engine.recommend_features(dataset_id, model_type)
            
            # Should complete within reasonable time
            assert performance_timer.duration < 20.0  # 20 seconds max
            assert recommendations is not None
            assert 'recommended_features' in recommendations
    
    def test_error_handling_empty_dataset(self, feature_engine):
        """Test error handling for empty datasets."""
        empty_data = pd.DataFrame()
        dataset_id = "empty_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = empty_data
            
            with pytest.raises(AIDataReadinessError):
                feature_engine.recommend_features(dataset_id, "classification")
    
    def test_error_handling_invalid_model_type(self, feature_engine, sample_csv_data):
        """Test error handling for invalid model types."""
        dataset_id = "test_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            with pytest.raises(AIDataReadinessError):
                feature_engine.recommend_features(dataset_id, "invalid_model_type")
    
    def test_error_handling_missing_time_column(self, feature_engine, sample_csv_data):
        """Test error handling when time column is missing."""
        dataset_id = "test_dataset"
        
        with patch.object(feature_engine, '_get_dataset') as mock_get:
            mock_get.return_value = sample_csv_data
            
            with pytest.raises(AIDataReadinessError):
                feature_engine.generate_temporal_features(dataset_id, 'nonexistent_time_column')