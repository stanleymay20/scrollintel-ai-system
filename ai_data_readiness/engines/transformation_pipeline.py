"""
Advanced Feature Transformation Pipeline for AI Data Readiness Platform.

This module provides a comprehensive transformation pipeline with advanced
temporal feature engineering and dimensionality reduction capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
import warnings

from ..models.feature_models import (
    FeatureInfo, FeatureType, TransformationType, ModelType,
    TransformationStep, EncodingStrategy, TemporalFeatures,
    FeatureRecommendation, FeatureRecommendations, TransformedDataset
)
from ..core.exceptions import FeatureEngineeringError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for transformation pipeline."""
    enable_scaling: bool = True
    enable_encoding: bool = True
    enable_temporal_features: bool = True
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = False
    enable_dimensionality_reduction: bool = False
    max_features_after_transformation: int = 1000
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01


class AdvancedTransformationPipeline:
    """
    Advanced feature transformation pipeline with comprehensive capabilities.
    
    Provides intelligent feature transformation, temporal feature engineering,
    and dimensionality reduction optimized for AI applications.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the transformation pipeline."""
        self.config = config or PipelineConfig()
        self.fitted_transformers = {}
        self.feature_importance_scores = {}
        self.transformation_history = []
        
    def fit_transform_pipeline(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit and transform data through the complete pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Target variable column name
            model_type: Type of ML model for optimization
            
        Returns:
            Tuple of (transformed_data, transformation_metadata)
        """
        try:
            logger.info("Starting advanced transformation pipeline")
            
            transformed_data = data.copy()
            metadata = {
                "original_shape": data.shape,
                "transformations_applied": [],
                "feature_mapping": {},
                "quality_metrics": {}
            }
            
            # Step 1: Handle missing values
            transformed_data, missing_metadata = self._handle_missing_values(transformed_data)
            metadata["transformations_applied"].append("missing_value_handling")
            metadata["missing_value_handling"] = missing_metadata
            
            # Step 2: Feature type detection and preprocessing
            feature_types = self._detect_feature_types(transformed_data, target_column)
            metadata["feature_types"] = feature_types
            
            # Step 3: Temporal feature engineering
            if self.config.enable_temporal_features:
                transformed_data, temporal_metadata = self._create_temporal_features(
                    transformed_data, feature_types
                )
                metadata["transformations_applied"].append("temporal_features")
                metadata["temporal_features"] = temporal_metadata
            
            # Step 4: Categorical encoding
            if self.config.enable_encoding:
                transformed_data, encoding_metadata = self._apply_categorical_encoding(
                    transformed_data, feature_types, model_type
                )
                metadata["transformations_applied"].append("categorical_encoding")
                metadata["categorical_encoding"] = encoding_metadata
            
            # Step 5: Numerical feature scaling
            if self.config.enable_scaling:
                transformed_data, scaling_metadata = self._apply_numerical_scaling(
                    transformed_data, feature_types, model_type
                )
                metadata["transformations_applied"].append("numerical_scaling")
                metadata["numerical_scaling"] = scaling_metadata
            
            # Step 6: Feature interactions
            if self.config.enable_interaction_features:
                transformed_data, interaction_metadata = self._create_interaction_features(
                    transformed_data, feature_types, target_column
                )
                metadata["transformations_applied"].append("interaction_features")
                metadata["interaction_features"] = interaction_metadata
            
            # Step 7: Polynomial features
            if self.config.enable_polynomial_features:
                transformed_data, poly_metadata = self._create_polynomial_features(
                    transformed_data, feature_types
                )
                metadata["transformations_applied"].append("polynomial_features")
                metadata["polynomial_features"] = poly_metadata
            
            # Step 8: Feature selection and dimensionality reduction
            if self.config.enable_dimensionality_reduction:
                transformed_data, dim_reduction_metadata = self._apply_dimensionality_reduction(
                    transformed_data, target_column, model_type
                )
                metadata["transformations_applied"].append("dimensionality_reduction")
                metadata["dimensionality_reduction"] = dim_reduction_metadata
            
            # Step 9: Final quality assessment
            metadata["quality_metrics"] = self._calculate_pipeline_quality_metrics(
                data, transformed_data
            )
            metadata["final_shape"] = transformed_data.shape
            
            logger.info(f"Pipeline completed: {data.shape} -> {transformed_data.shape}")
            return transformed_data, metadata
            
        except Exception as e:
            logger.error(f"Error in transformation pipeline: {str(e)}")
            raise FeatureEngineeringError(f"Pipeline transformation failed: {str(e)}")
    
    def _handle_missing_values(
        self, 
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values with intelligent strategies."""
        metadata = {"strategies_applied": {}, "missing_indicators_created": []}
        
        for column in data.columns:
            missing_rate = data[column].isnull().sum() / len(data)
            
            if missing_rate > 0:
                # Create missing indicator for features with significant missing values
                if missing_rate > 0.05:  # More than 5% missing
                    indicator_col = f"{column}_is_missing"
                    data[indicator_col] = data[column].isnull().astype(int)
                    metadata["missing_indicators_created"].append(indicator_col)
                
                # Apply appropriate imputation strategy
                if pd.api.types.is_numeric_dtype(data[column]):
                    if missing_rate < 0.3:
                        # Use median for numerical features with moderate missing values
                        fill_value = data[column].median()
                        data[column].fillna(fill_value, inplace=True)
                        metadata["strategies_applied"][column] = f"median_imputation_{fill_value}"
                    else:
                        # Use forward fill for high missing rates
                        data[column].fillna(method='ffill', inplace=True)
                        data[column].fillna(data[column].median(), inplace=True)
                        metadata["strategies_applied"][column] = "forward_fill_then_median"
                else:
                    # Use mode for categorical features
                    if not data[column].mode().empty:
                        fill_value = data[column].mode()[0]
                        data[column].fillna(fill_value, inplace=True)
                        metadata["strategies_applied"][column] = f"mode_imputation_{fill_value}"
                    else:
                        data[column].fillna("unknown", inplace=True)
                        metadata["strategies_applied"][column] = "unknown_imputation"
        
        return data, metadata
    
    def _detect_feature_types(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, FeatureType]:
        """Detect feature types for all columns."""
        feature_types = {}
        
        for column in data.columns:
            if column == target_column:
                continue
                
            # Check for temporal features
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                feature_types[column] = FeatureType.TEMPORAL
            # Check for binary features
            elif data[column].nunique() == 2:
                feature_types[column] = FeatureType.BINARY
            # Check for numerical features
            elif pd.api.types.is_numeric_dtype(data[column]):
                feature_types[column] = FeatureType.NUMERICAL
            # Check for text features (long strings)
            elif data[column].dtype == 'object':
                avg_length = data[column].astype(str).str.len().mean()
                if avg_length > 25:
                    feature_types[column] = FeatureType.TEXT
                else:
                    feature_types[column] = FeatureType.CATEGORICAL
            else:
                feature_types[column] = FeatureType.CATEGORICAL
        
        return feature_types
    
    def _create_temporal_features(
        self, 
        data: pd.DataFrame, 
        feature_types: Dict[str, FeatureType]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create comprehensive temporal features."""
        metadata = {"features_created": [], "temporal_columns": []}
        
        temporal_columns = [col for col, ftype in feature_types.items() 
                          if ftype == FeatureType.TEMPORAL]
        
        for column in temporal_columns:
            metadata["temporal_columns"].append(column)
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = pd.to_datetime(data[column], errors='coerce')
            
            # Basic temporal components
            data[f"{column}_year"] = data[column].dt.year
            data[f"{column}_month"] = data[column].dt.month
            data[f"{column}_day"] = data[column].dt.day
            data[f"{column}_hour"] = data[column].dt.hour
            data[f"{column}_dayofweek"] = data[column].dt.dayofweek
            data[f"{column}_dayofyear"] = data[column].dt.dayofyear
            data[f"{column}_quarter"] = data[column].dt.quarter
            
            # Cyclical features for better ML performance
            data[f"{column}_month_sin"] = np.sin(2 * np.pi * data[column].dt.month / 12)
            data[f"{column}_month_cos"] = np.cos(2 * np.pi * data[column].dt.month / 12)
            data[f"{column}_day_sin"] = np.sin(2 * np.pi * data[column].dt.day / 31)
            data[f"{column}_day_cos"] = np.cos(2 * np.pi * data[column].dt.day / 31)
            data[f"{column}_hour_sin"] = np.sin(2 * np.pi * data[column].dt.hour / 24)
            data[f"{column}_hour_cos"] = np.cos(2 * np.pi * data[column].dt.hour / 24)
            data[f"{column}_dayofweek_sin"] = np.sin(2 * np.pi * data[column].dt.dayofweek / 7)
            data[f"{column}_dayofweek_cos"] = np.cos(2 * np.pi * data[column].dt.dayofweek / 7)
            
            # Time-based features
            data[f"{column}_is_weekend"] = (data[column].dt.dayofweek >= 5).astype(int)
            data[f"{column}_is_month_start"] = data[column].dt.is_month_start.astype(int)
            data[f"{column}_is_month_end"] = data[column].dt.is_month_end.astype(int)
            data[f"{column}_is_quarter_start"] = data[column].dt.is_quarter_start.astype(int)
            data[f"{column}_is_quarter_end"] = data[column].dt.is_quarter_end.astype(int)
            
            # Lag features (if data is sorted by time)
            if len(data) > 1:
                data[f"{column}_days_since_min"] = (data[column] - data[column].min()).dt.days
                data[f"{column}_days_until_max"] = (data[column].max() - data[column]).dt.days
            
            created_features = [col for col in data.columns if col.startswith(f"{column}_")]
            metadata["features_created"].extend(created_features)
        
        return data, metadata
    
    def _apply_categorical_encoding(
        self, 
        data: pd.DataFrame, 
        feature_types: Dict[str, FeatureType], 
        model_type: ModelType
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply intelligent categorical encoding based on model type."""
        metadata = {"encodings_applied": {}, "new_columns": []}
        
        categorical_columns = [col for col, ftype in feature_types.items() 
                             if ftype in [FeatureType.CATEGORICAL, FeatureType.BINARY]]
        
        for column in categorical_columns:
            cardinality = data[column].nunique()
            
            # Choose encoding strategy based on cardinality and model type
            if cardinality <= 2:
                # Binary encoding
                encoder = LabelEncoder()
                encoded_col = f"{column}_encoded"
                data[encoded_col] = encoder.fit_transform(data[column].astype(str))
                self.fitted_transformers[f"{column}_encoder"] = encoder
                metadata["encodings_applied"][column] = "binary_encoding"
                metadata["new_columns"].append(encoded_col)
                
            elif cardinality <= 10:
                if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION, 
                                ModelType.NEURAL_NETWORK]:
                    # One-hot encoding for linear models
                    encoded_df = pd.get_dummies(data[column], prefix=column, drop_first=True)
                    for col in encoded_df.columns:
                        data[col] = encoded_df[col]
                    metadata["encodings_applied"][column] = "one_hot_encoding"
                    metadata["new_columns"].extend(encoded_df.columns.tolist())
                else:
                    # Label encoding for tree-based models
                    encoder = LabelEncoder()
                    encoded_col = f"{column}_encoded"
                    data[encoded_col] = encoder.fit_transform(data[column].astype(str))
                    self.fitted_transformers[f"{column}_encoder"] = encoder
                    metadata["encodings_applied"][column] = "label_encoding"
                    metadata["new_columns"].append(encoded_col)
                    
            elif cardinality <= 50:
                # Target encoding for medium cardinality (simplified version)
                encoder = LabelEncoder()
                encoded_col = f"{column}_encoded"
                data[encoded_col] = encoder.fit_transform(data[column].astype(str))
                self.fitted_transformers[f"{column}_encoder"] = encoder
                metadata["encodings_applied"][column] = "label_encoding"
                metadata["new_columns"].append(encoded_col)
                
            else:
                # Hash encoding or frequency encoding for high cardinality
                value_counts = data[column].value_counts()
                freq_col = f"{column}_frequency"
                data[freq_col] = data[column].map(value_counts)
                metadata["encodings_applied"][column] = "frequency_encoding"
                metadata["new_columns"].append(freq_col)
        
        return data, metadata
    
    def _apply_numerical_scaling(
        self, 
        data: pd.DataFrame, 
        feature_types: Dict[str, FeatureType], 
        model_type: ModelType
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply appropriate scaling to numerical features."""
        metadata = {"scalers_applied": {}, "scaled_columns": []}
        
        numerical_columns = [col for col, ftype in feature_types.items() 
                           if ftype == FeatureType.NUMERICAL]
        
        for column in numerical_columns:
            # Choose scaler based on model type and data distribution
            skewness = data[column].skew()
            
            if model_type in [ModelType.NEURAL_NETWORK, ModelType.SVM]:
                # Use StandardScaler for models sensitive to scale
                scaler = StandardScaler()
                scaler_type = "standard"
            elif abs(skewness) > 2:  # Highly skewed data
                # Use RobustScaler for skewed data
                scaler = RobustScaler()
                scaler_type = "robust"
            elif model_type == ModelType.NEURAL_NETWORK:
                # Use MinMaxScaler for neural networks
                scaler = MinMaxScaler()
                scaler_type = "minmax"
            else:
                # Use StandardScaler as default
                scaler = StandardScaler()
                scaler_type = "standard"
            
            scaled_col = f"{column}_scaled"
            data[scaled_col] = scaler.fit_transform(data[[column]]).flatten()
            
            self.fitted_transformers[f"{column}_scaler"] = scaler
            metadata["scalers_applied"][column] = scaler_type
            metadata["scaled_columns"].append(scaled_col)
        
        return data, metadata
    
    def _create_interaction_features(
        self, 
        data: pd.DataFrame, 
        feature_types: Dict[str, FeatureType], 
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create interaction features between important numerical features."""
        metadata = {"interactions_created": []}
        
        numerical_columns = [col for col, ftype in feature_types.items() 
                           if ftype == FeatureType.NUMERICAL]
        
        # Limit to most important features to avoid explosion
        if len(numerical_columns) > 10:
            # Use correlation with target to select top features
            if target_column and target_column in data.columns:
                correlations = {}
                for col in numerical_columns:
                    try:
                        corr = abs(data[col].corr(data[target_column]))
                        correlations[col] = corr if not np.isnan(corr) else 0
                    except:
                        correlations[col] = 0
                
                # Select top 10 features
                top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
                numerical_columns = [feat[0] for feat in top_features]
        
        # Create interactions between top features
        for i, col1 in enumerate(numerical_columns):
            for col2 in numerical_columns[i+1:]:
                # Multiplication interaction
                interaction_col = f"{col1}_{col2}_mult"
                data[interaction_col] = data[col1] * data[col2]
                metadata["interactions_created"].append(interaction_col)
                
                # Addition interaction (for some cases)
                if len(metadata["interactions_created"]) < 20:  # Limit total interactions
                    add_interaction_col = f"{col1}_{col2}_add"
                    data[add_interaction_col] = data[col1] + data[col2]
                    metadata["interactions_created"].append(add_interaction_col)
        
        return data, metadata
    
    def _create_polynomial_features(
        self, 
        data: pd.DataFrame, 
        feature_types: Dict[str, FeatureType]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create polynomial features for numerical columns."""
        metadata = {"polynomial_features_created": []}
        
        numerical_columns = [col for col, ftype in feature_types.items() 
                           if ftype == FeatureType.NUMERICAL]
        
        # Limit to avoid feature explosion
        for column in numerical_columns[:5]:  # Only top 5 numerical features
            # Square features
            square_col = f"{column}_squared"
            data[square_col] = data[column] ** 2
            metadata["polynomial_features_created"].append(square_col)
            
            # Cube features
            cube_col = f"{column}_cubed"
            data[cube_col] = data[column] ** 3
            metadata["polynomial_features_created"].append(cube_col)
            
            # Square root features (for positive values)
            if data[column].min() >= 0:
                sqrt_col = f"{column}_sqrt"
                data[sqrt_col] = np.sqrt(data[column])
                metadata["polynomial_features_created"].append(sqrt_col)
            
            # Log features (for positive values)
            if data[column].min() > 0:
                log_col = f"{column}_log"
                data[log_col] = np.log(data[column])
                metadata["polynomial_features_created"].append(log_col)
                
                # Log1p features (more stable for small values)
                log1p_col = f"{column}_log1p"
                data[log1p_col] = np.log1p(data[column])
                metadata["polynomial_features_created"].append(log1p_col)
            
            # Reciprocal features (for non-zero values)
            if data[column].min() != 0:
                reciprocal_col = f"{column}_reciprocal"
                data[reciprocal_col] = 1 / data[column]
                metadata["polynomial_features_created"].append(reciprocal_col)
            
            # Exponential features (for reasonable ranges)
            if data[column].max() < 10 and data[column].min() > -10:
                exp_col = f"{column}_exp"
                data[exp_col] = np.exp(data[column])
                metadata["polynomial_features_created"].append(exp_col)
        
        return data, metadata
    
    def _apply_dimensionality_reduction(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply dimensionality reduction techniques."""
        metadata = {"method_used": None, "components_kept": 0, "variance_explained": 0}
        
        # Get numerical features for dimensionality reduction
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numerical_features:
            numerical_features.remove(target_column)
        
        if len(numerical_features) < 10:
            metadata["method_used"] = "none_needed"
            return data, metadata
        
        # Check if we need dimensionality reduction
        if len(numerical_features) > self.config.max_features_after_transformation:
            # Use PCA for dimensionality reduction
            n_components = min(50, len(numerical_features) // 2)
            
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(data[numerical_features])
            
            # Create PCA feature columns
            pca_columns = [f"pca_component_{i}" for i in range(n_components)]
            pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=data.index)
            
            # Remove original numerical features and add PCA features
            data = data.drop(columns=numerical_features)
            data = pd.concat([data, pca_df], axis=1)
            
            self.fitted_transformers["pca"] = pca
            metadata["method_used"] = "pca"
            metadata["components_kept"] = n_components
            metadata["variance_explained"] = pca.explained_variance_ratio_.sum()
            metadata["original_features_removed"] = numerical_features
            metadata["pca_features_added"] = pca_columns
        
        return data, metadata
    
    def _calculate_pipeline_quality_metrics(
        self, 
        original_data: pd.DataFrame, 
        transformed_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate quality metrics for the entire pipeline."""
        metrics = {}
        
        # Feature count change
        original_features = len(original_data.columns)
        transformed_features = len(transformed_data.columns)
        metrics["feature_count_change"] = (transformed_features - original_features) / original_features
        
        # Data completeness
        original_completeness = 1 - original_data.isnull().sum().sum() / (original_data.shape[0] * original_data.shape[1])
        transformed_completeness = 1 - transformed_data.isnull().sum().sum() / (transformed_data.shape[0] * transformed_data.shape[1])
        metrics["completeness_improvement"] = transformed_completeness - original_completeness
        
        # Memory usage change
        original_memory = original_data.memory_usage(deep=True).sum()
        transformed_memory = transformed_data.memory_usage(deep=True).sum()
        metrics["memory_usage_change"] = (transformed_memory - original_memory) / original_memory
        
        # Numerical feature statistics
        original_numerical = original_data.select_dtypes(include=[np.number])
        transformed_numerical = transformed_data.select_dtypes(include=[np.number])
        
        if len(original_numerical.columns) > 0 and len(transformed_numerical.columns) > 0:
            metrics["numerical_feature_variance_change"] = (
                transformed_numerical.var().mean() / original_numerical.var().mean()
                if original_numerical.var().mean() > 0 else 1.0
            )
        
        return metrics
    
    def transform_new_data(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform new data using fitted transformers."""
        try:
            transformed_data = data.copy()
            
            # Apply fitted transformers in the same order
            for transformer_name, transformer in self.fitted_transformers.items():
                if "encoder" in transformer_name:
                    column_name = transformer_name.replace("_encoder", "")
                    if column_name in transformed_data.columns:
                        encoded_col = f"{column_name}_encoded"
                        transformed_data[encoded_col] = transformer.transform(
                            transformed_data[column_name].astype(str)
                        )
                elif "scaler" in transformer_name:
                    column_name = transformer_name.replace("_scaler", "")
                    if column_name in transformed_data.columns:
                        scaled_col = f"{column_name}_scaled"
                        transformed_data[scaled_col] = transformer.transform(
                            transformed_data[[column_name]]
                        ).flatten()
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming new data: {str(e)}")
            raise FeatureEngineeringError(f"New data transformation failed: {str(e)}")