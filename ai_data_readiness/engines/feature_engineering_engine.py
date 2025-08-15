"""
Feature Engineering Engine for AI Data Readiness Platform.

This module provides intelligent feature engineering recommendations and transformations
specifically optimized for AI and machine learning applications.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

from ..models.feature_models import (
    FeatureInfo, FeatureType, TransformationType, ModelType,
    TransformationStep, EncodingStrategy, TemporalFeatures,
    FeatureRecommendation, FeatureRecommendations, TransformedDataset
)
from ..core.exceptions import FeatureEngineeringError
from .transformation_pipeline import AdvancedTransformationPipeline, PipelineConfig
from .temporal_feature_generator import AdvancedTemporalFeatureGenerator, TemporalConfig
from .dimensionality_reduction import AdvancedDimensionalityReducer, DimensionalityReductionConfig

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngineeringEngine:
    """
    Intelligent feature engineering engine with model-specific recommendations.
    
    Provides automated feature discovery, selection, and transformation recommendations
    optimized for different ML model types and use cases.
    """
    
    def __init__(self):
        """Initialize the feature engineering engine."""
        self.feature_analyzers = {
            FeatureType.NUMERICAL: self._analyze_numerical_feature,
            FeatureType.CATEGORICAL: self._analyze_categorical_feature,
            FeatureType.TEMPORAL: self._analyze_temporal_feature,
            FeatureType.TEXT: self._analyze_text_feature,
            FeatureType.BINARY: self._analyze_binary_feature
        }
        
        self.model_specific_strategies = {
            ModelType.LINEAR_REGRESSION: self._linear_model_strategy,
            ModelType.LOGISTIC_REGRESSION: self._linear_model_strategy,
            ModelType.RANDOM_FOREST: self._tree_model_strategy,
            ModelType.GRADIENT_BOOSTING: self._tree_model_strategy,
            ModelType.NEURAL_NETWORK: self._neural_network_strategy,
            ModelType.SVM: self._svm_strategy,
            ModelType.TIME_SERIES: self._time_series_strategy
        }
        
        # Initialize advanced components
        self.transformation_pipeline = AdvancedTransformationPipeline()
        self.temporal_generator = AdvancedTemporalFeatureGenerator()
        self.dimensionality_reducer = AdvancedDimensionalityReducer()
    
    def recommend_features(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        model_type: ModelType,
        target_column: Optional[str] = None
    ) -> FeatureRecommendations:
        """
        Generate intelligent feature engineering recommendations.
        
        Args:
            dataset_id: Unique identifier for the dataset
            data: Input DataFrame
            model_type: Type of ML model for optimization
            target_column: Target variable column name
            
        Returns:
            FeatureRecommendations object with comprehensive recommendations
        """
        try:
            logger.info(f"Generating feature recommendations for dataset {dataset_id}")
            
            # Analyze features
            feature_analysis = self._analyze_features(data, target_column)
            
            # Generate model-specific recommendations
            recommendations = self._generate_model_specific_recommendations(
                feature_analysis, model_type, target_column
            )
            
            # Generate encoding strategies
            encoding_strategies = self._generate_encoding_strategies(
                feature_analysis, model_type
            )
            
            # Generate temporal features if applicable
            temporal_features = self._generate_temporal_features(
                data, feature_analysis
            )
            
            # Feature selection recommendations
            feature_selection = self._recommend_feature_selection(
                data, feature_analysis, target_column, model_type
            )
            
            # Dimensionality reduction recommendations
            dim_reduction = self._recommend_dimensionality_reduction(
                feature_analysis, model_type
            )
            
            return FeatureRecommendations(
                dataset_id=dataset_id,
                model_type=model_type,
                target_column=target_column,
                recommendations=recommendations,
                encoding_strategies=encoding_strategies,
                temporal_features=temporal_features,
                feature_selection_recommendations=feature_selection,
                dimensionality_reduction_recommendation=dim_reduction
            )
            
        except Exception as e:
            logger.error(f"Error generating feature recommendations: {str(e)}")
            raise FeatureEngineeringError(f"Feature recommendation failed: {str(e)}")
    
    def _analyze_features(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, FeatureInfo]:
        """Analyze all features in the dataset."""
        feature_analysis = {}
        
        for column in data.columns:
            if column == target_column:
                continue
                
            feature_type = self._detect_feature_type(data[column])
            
            feature_info = FeatureInfo(
                name=column,
                type=feature_type,
                data_type=str(data[column].dtype),
                missing_rate=data[column].isnull().sum() / len(data),
                unique_values=data[column].nunique(),
                cardinality=data[column].nunique() if feature_type == FeatureType.CATEGORICAL else None
            )
            
            # Add distribution statistics
            feature_info.distribution_stats = self._get_distribution_stats(data[column])
            
            # Calculate correlation with target if available
            if target_column and target_column in data.columns:
                feature_info.correlation_with_target = self._calculate_correlation(
                    data[column], data[target_column]
                )
            
            # Analyze feature using specific analyzer
            if feature_type in self.feature_analyzers:
                additional_info = self.feature_analyzers[feature_type](data[column])
                feature_info.distribution_stats.update(additional_info)
            
            feature_analysis[column] = feature_info
        
        return feature_analysis
    
    def _detect_feature_type(self, series: pd.Series) -> FeatureType:
        """Detect the type of a feature."""
        # Check for temporal features
        if pd.api.types.is_datetime64_any_dtype(series):
            return FeatureType.TEMPORAL
        
        # Check for binary features
        unique_values = series.dropna().unique()
        if len(unique_values) == 2:
            return FeatureType.BINARY
        
        # Check for numerical features
        if pd.api.types.is_numeric_dtype(series):
            return FeatureType.NUMERICAL
        
        # Check for text features (long strings)
        if series.dtype == 'object':
            avg_length = series.astype(str).str.len().mean()
            if avg_length > 25:  # Arbitrary threshold for text
                return FeatureType.TEXT
        
        # Default to categorical
        return FeatureType.CATEGORICAL
    
    def _get_distribution_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get distribution statistics for a feature."""
        stats = {}
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            })
        
        return stats
    
    def _calculate_correlation(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate correlation between feature and target."""
        try:
            # Handle different data types
            if pd.api.types.is_numeric_dtype(feature) and pd.api.types.is_numeric_dtype(target):
                return feature.corr(target)
            else:
                # Use mutual information for non-numeric features
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                
                # Encode categorical features
                if not pd.api.types.is_numeric_dtype(feature):
                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(feature.astype(str))
                else:
                    feature_encoded = feature.values
                
                if not pd.api.types.is_numeric_dtype(target):
                    le_target = LabelEncoder()
                    target_encoded = le_target.fit_transform(target.astype(str))
                    mi_score = mutual_info_classif(
                        feature_encoded.reshape(-1, 1), target_encoded
                    )[0]
                else:
                    mi_score = mutual_info_regression(
                        feature_encoded.reshape(-1, 1), target.values
                    )[0]
                
                return mi_score
        except Exception:
            return 0.0
    
    def _analyze_numerical_feature(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numerical features."""
        return {
            'outlier_rate': self._calculate_outlier_rate(series),
            'zero_rate': (series == 0).sum() / len(series),
            'distribution_type': self._detect_distribution_type(series)
        }
    
    def _analyze_categorical_feature(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical features."""
        value_counts = series.value_counts()
        return {
            'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_rate': value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0,
            'rare_categories_count': (value_counts < 5).sum(),
            'category_distribution': 'balanced' if value_counts.std() < value_counts.mean() else 'imbalanced'
        }
    
    def _analyze_temporal_feature(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze temporal features."""
        return {
            'date_range': (series.min(), series.max()),
            'frequency': pd.infer_freq(series) if series.is_monotonic_increasing else None,
            'has_seasonality': self._detect_seasonality(series)
        }
    
    def _analyze_text_feature(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text features."""
        text_lengths = series.astype(str).str.len()
        return {
            'avg_length': text_lengths.mean(),
            'max_length': text_lengths.max(),
            'contains_numbers': series.astype(str).str.contains(r'\d').sum() / len(series),
            'contains_special_chars': series.astype(str).str.contains(r'[^a-zA-Z0-9\s]').sum() / len(series)
        }
    
    def _analyze_binary_feature(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze binary features."""
        value_counts = series.value_counts()
        return {
            'positive_rate': value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0,
            'is_balanced': abs(value_counts.iloc[0] - value_counts.iloc[1]) / len(series) < 0.1 if len(value_counts) == 2 else False
        }
    
    def _calculate_outlier_rate(self, series: pd.Series) -> float:
        """Calculate outlier rate using IQR method."""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            return outliers.sum() / len(series)
        except Exception:
            return 0.0
    
    def _detect_distribution_type(self, series: pd.Series) -> str:
        """Detect the distribution type of numerical data."""
        try:
            skewness = series.skew()
            if abs(skewness) < 0.5:
                return 'normal'
            elif skewness > 0.5:
                return 'right_skewed'
            else:
                return 'left_skewed'
        except Exception:
            return 'unknown'
    
    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Detect seasonality in temporal data."""
        # Simplified seasonality detection
        try:
            if len(series) < 24:  # Need at least 24 points
                return False
            
            # Check for weekly pattern (if daily data)
            if len(series) >= 7:
                weekly_pattern = series.groupby(series.dt.dayofweek).mean()
                return weekly_pattern.std() > weekly_pattern.mean() * 0.1
            
            return False
        except Exception:
            return False    

    def _generate_model_specific_recommendations(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        model_type: ModelType,
        target_column: Optional[str]
    ) -> List[FeatureRecommendation]:
        """Generate model-specific feature recommendations."""
        recommendations = []
        
        # Get model-specific strategy
        if model_type in self.model_specific_strategies:
            strategy_recommendations = self.model_specific_strategies[model_type](
                feature_analysis, target_column
            )
            recommendations.extend(strategy_recommendations)
        
        # Add general recommendations
        general_recommendations = self._generate_general_recommendations(feature_analysis)
        recommendations.extend(general_recommendations)
        
        # Sort by expected impact
        recommendations.sort(key=lambda x: x.expected_impact, reverse=True)
        
        return recommendations
    
    def _linear_model_strategy(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        target_column: Optional[str]
    ) -> List[FeatureRecommendation]:
        """Generate recommendations for linear models."""
        recommendations = []
        
        for feature_name, feature_info in feature_analysis.items():
            # Scaling recommendations for numerical features
            if feature_info.type == FeatureType.NUMERICAL:
                if feature_info.distribution_stats.get('std', 1) > 10:
                    recommendations.append(FeatureRecommendation(
                        feature_name=feature_name,
                        recommendation_type="scaling",
                        transformation=TransformationStep(
                            transformation_type=TransformationType.SCALING,
                            parameters={"method": "standard"},
                            input_features=[feature_name],
                            output_features=[f"{feature_name}_scaled"],
                            description=f"Apply standard scaling to {feature_name}",
                            rationale="Linear models benefit from scaled features"
                        ),
                        expected_impact=0.8,
                        confidence=0.9,
                        rationale="Linear models are sensitive to feature scales",
                        implementation_complexity="low"
                    ))
                
                # Polynomial features for non-linear relationships
                if abs(feature_info.correlation_with_target or 0) > 0.3:
                    recommendations.append(FeatureRecommendation(
                        feature_name=feature_name,
                        recommendation_type="polynomial",
                        transformation=TransformationStep(
                            transformation_type=TransformationType.POLYNOMIAL,
                            parameters={"degree": 2},
                            input_features=[feature_name],
                            output_features=[f"{feature_name}_squared"],
                            description=f"Create polynomial features for {feature_name}",
                            rationale="Capture non-linear relationships"
                        ),
                        expected_impact=0.6,
                        confidence=0.7,
                        rationale="Polynomial features can capture non-linear patterns",
                        implementation_complexity="medium"
                    ))
            
            # One-hot encoding for categorical features
            elif feature_info.type == FeatureType.CATEGORICAL:
                if feature_info.cardinality and feature_info.cardinality <= 10:
                    recommendations.append(FeatureRecommendation(
                        feature_name=feature_name,
                        recommendation_type="encoding",
                        transformation=TransformationStep(
                            transformation_type=TransformationType.ENCODING,
                            parameters={"method": "one_hot"},
                            input_features=[feature_name],
                            output_features=[f"{feature_name}_encoded"],
                            description=f"One-hot encode {feature_name}",
                            rationale="Linear models work well with one-hot encoded features"
                        ),
                        expected_impact=0.7,
                        confidence=0.8,
                        rationale="One-hot encoding preserves categorical information",
                        implementation_complexity="low"
                    ))
        
        return recommendations
    
    def _tree_model_strategy(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        target_column: Optional[str]
    ) -> List[FeatureRecommendation]:
        """Generate recommendations for tree-based models."""
        recommendations = []
        
        for feature_name, feature_info in feature_analysis.items():
            # Tree models handle raw features well, focus on feature creation
            if feature_info.type == FeatureType.NUMERICAL:
                # Binning for numerical features
                if feature_info.distribution_stats.get('outlier_rate', 0) > 0.1:
                    recommendations.append(FeatureRecommendation(
                        feature_name=feature_name,
                        recommendation_type="binning",
                        transformation=TransformationStep(
                            transformation_type=TransformationType.BINNING,
                            parameters={"n_bins": 10, "strategy": "quantile"},
                            input_features=[feature_name],
                            output_features=[f"{feature_name}_binned"],
                            description=f"Create binned version of {feature_name}",
                            rationale="Binning can help with outliers and create interpretable splits"
                        ),
                        expected_impact=0.5,
                        confidence=0.6,
                        rationale="Tree models can benefit from binned features",
                        implementation_complexity="low"
                    ))
            
            elif feature_info.type == FeatureType.CATEGORICAL:
                # Label encoding for high cardinality categorical features
                if feature_info.cardinality and feature_info.cardinality > 10:
                    recommendations.append(FeatureRecommendation(
                        feature_name=feature_name,
                        recommendation_type="encoding",
                        transformation=TransformationStep(
                            transformation_type=TransformationType.ENCODING,
                            parameters={"method": "label"},
                            input_features=[feature_name],
                            output_features=[f"{feature_name}_encoded"],
                            description=f"Label encode {feature_name}",
                            rationale="Tree models handle label encoded high-cardinality features well"
                        ),
                        expected_impact=0.6,
                        confidence=0.7,
                        rationale="Label encoding reduces dimensionality for tree models",
                        implementation_complexity="low"
                    ))
        
        return recommendations
    
    def _neural_network_strategy(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        target_column: Optional[str]
    ) -> List[FeatureRecommendation]:
        """Generate recommendations for neural networks."""
        recommendations = []
        
        for feature_name, feature_info in feature_analysis.items():
            if feature_info.type == FeatureType.NUMERICAL:
                # Normalization for neural networks
                recommendations.append(FeatureRecommendation(
                    feature_name=feature_name,
                    recommendation_type="normalization",
                    transformation=TransformationStep(
                        transformation_type=TransformationType.NORMALIZATION,
                        parameters={"method": "min_max"},
                        input_features=[feature_name],
                        output_features=[f"{feature_name}_normalized"],
                        description=f"Normalize {feature_name} to [0,1] range",
                        rationale="Neural networks benefit from normalized inputs"
                    ),
                    expected_impact=0.8,
                    confidence=0.9,
                    rationale="Neural networks require normalized features for stable training",
                    implementation_complexity="low"
                ))
            
            elif feature_info.type == FeatureType.CATEGORICAL:
                # Embedding recommendations for high cardinality
                if feature_info.cardinality and feature_info.cardinality > 50:
                    recommendations.append(FeatureRecommendation(
                        feature_name=feature_name,
                        recommendation_type="embedding",
                        transformation=TransformationStep(
                            transformation_type=TransformationType.ENCODING,
                            parameters={"method": "embedding", "embedding_dim": min(50, feature_info.cardinality // 2)},
                            input_features=[feature_name],
                            output_features=[f"{feature_name}_embedding"],
                            description=f"Create embeddings for {feature_name}",
                            rationale="Embeddings handle high-cardinality categorical features efficiently"
                        ),
                        expected_impact=0.7,
                        confidence=0.8,
                        rationale="Embeddings reduce dimensionality while preserving information",
                        implementation_complexity="high"
                    ))
        
        return recommendations
    
    def _svm_strategy(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        target_column: Optional[str]
    ) -> List[FeatureRecommendation]:
        """Generate recommendations for SVM models."""
        recommendations = []
        
        for feature_name, feature_info in feature_analysis.items():
            if feature_info.type == FeatureType.NUMERICAL:
                # Standard scaling is crucial for SVM
                recommendations.append(FeatureRecommendation(
                    feature_name=feature_name,
                    recommendation_type="scaling",
                    transformation=TransformationStep(
                        transformation_type=TransformationType.SCALING,
                        parameters={"method": "standard"},
                        input_features=[feature_name],
                        output_features=[f"{feature_name}_scaled"],
                        description=f"Apply standard scaling to {feature_name}",
                        rationale="SVM is very sensitive to feature scales"
                    ),
                    expected_impact=0.9,
                    confidence=0.95,
                    rationale="SVM requires scaled features for optimal performance",
                    implementation_complexity="low"
                ))
        
        return recommendations
    
    def _time_series_strategy(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        target_column: Optional[str]
    ) -> List[FeatureRecommendation]:
        """Generate recommendations for time series models."""
        recommendations = []
        
        # Look for temporal features
        temporal_features = [name for name, info in feature_analysis.items() 
                           if info.type == FeatureType.TEMPORAL]
        
        if temporal_features:
            for feature_name in temporal_features:
                recommendations.append(FeatureRecommendation(
                    feature_name=feature_name,
                    recommendation_type="temporal_extraction",
                    transformation=TransformationStep(
                        transformation_type=TransformationType.EXTRACTION,
                        parameters={"extract": ["hour", "day", "month", "year", "dayofweek"]},
                        input_features=[feature_name],
                        output_features=[f"{feature_name}_{comp}" for comp in ["hour", "day", "month", "year", "dayofweek"]],
                        description=f"Extract temporal components from {feature_name}",
                        rationale="Time series models benefit from temporal feature decomposition"
                    ),
                    expected_impact=0.8,
                    confidence=0.8,
                    rationale="Temporal components capture seasonal patterns",
                    implementation_complexity="medium"
                ))
        
        return recommendations
    
    def _generate_general_recommendations(
        self,
        feature_analysis: Dict[str, FeatureInfo]
    ) -> List[FeatureRecommendation]:
        """Generate general feature engineering recommendations."""
        recommendations = []
        
        # Missing value handling
        for feature_name, feature_info in feature_analysis.items():
            if feature_info.missing_rate > 0.01:  # More than 1% missing
                recommendations.append(FeatureRecommendation(
                    feature_name=feature_name,
                    recommendation_type="missing_value_handling",
                    transformation=TransformationStep(
                        transformation_type=TransformationType.EXTRACTION,
                        parameters={"method": "create_missing_indicator"},
                        input_features=[feature_name],
                        output_features=[f"{feature_name}_is_missing"],
                        description=f"Create missing value indicator for {feature_name}",
                        rationale="Missing patterns can be informative"
                    ),
                    expected_impact=0.4,
                    confidence=0.6,
                    rationale="Missing value patterns can contain useful information",
                    implementation_complexity="low"
                ))
        
        # Feature interactions for highly correlated features
        high_corr_features = [name for name, info in feature_analysis.items() 
                            if abs(info.correlation_with_target or 0) > 0.5]
        
        if len(high_corr_features) >= 2:
            for i, feat1 in enumerate(high_corr_features):
                for feat2 in high_corr_features[i+1:]:
                    if (feature_analysis[feat1].type == FeatureType.NUMERICAL and 
                        feature_analysis[feat2].type == FeatureType.NUMERICAL):
                        recommendations.append(FeatureRecommendation(
                            feature_name=f"{feat1}_{feat2}",
                            recommendation_type="interaction",
                            transformation=TransformationStep(
                                transformation_type=TransformationType.INTERACTION,
                                parameters={"operation": "multiply"},
                                input_features=[feat1, feat2],
                                output_features=[f"{feat1}_{feat2}_interaction"],
                                description=f"Create interaction between {feat1} and {feat2}",
                                rationale="Interaction features can capture combined effects"
                            ),
                            expected_impact=0.5,
                            confidence=0.6,
                            rationale="Feature interactions can reveal hidden patterns",
                            implementation_complexity="medium"
                        ))
        
        return recommendations
    
    def _generate_encoding_strategies(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        model_type: ModelType
    ) -> List[EncodingStrategy]:
        """Generate categorical encoding strategies."""
        strategies = []
        
        for feature_name, feature_info in feature_analysis.items():
            if feature_info.type == FeatureType.CATEGORICAL:
                cardinality = feature_info.cardinality or 0
                
                if cardinality <= 5:
                    # One-hot encoding for low cardinality
                    strategies.append(EncodingStrategy(
                        feature_name=feature_name,
                        encoding_type="one_hot",
                        parameters={"drop_first": True},
                        expected_dimensions=cardinality - 1,
                        handle_unknown="ignore"
                    ))
                elif cardinality <= 20:
                    # Binary encoding for medium cardinality
                    strategies.append(EncodingStrategy(
                        feature_name=feature_name,
                        encoding_type="binary",
                        parameters={},
                        expected_dimensions=int(np.ceil(np.log2(cardinality))),
                        handle_unknown="ignore"
                    ))
                else:
                    # Target encoding for high cardinality
                    strategies.append(EncodingStrategy(
                        feature_name=feature_name,
                        encoding_type="target",
                        parameters={"smoothing": 1.0},
                        expected_dimensions=1,
                        handle_unknown="ignore"
                    ))
        
        return strategies
    
    def _generate_temporal_features(
        self,
        data: pd.DataFrame,
        feature_analysis: Dict[str, FeatureInfo]
    ) -> Optional[TemporalFeatures]:
        """Generate temporal feature recommendations."""
        temporal_columns = [name for name, info in feature_analysis.items() 
                          if info.type == FeatureType.TEMPORAL]
        
        if not temporal_columns:
            return None
        
        # Use the first temporal column as primary time column
        time_column = temporal_columns[0]
        
        return TemporalFeatures(
            time_column=time_column,
            features_to_create=[
                "hour", "day", "month", "year", "dayofweek", "quarter",
                "is_weekend", "is_month_start", "is_month_end"
            ],
            aggregation_windows=["1H", "1D", "1W", "1M"],
            lag_features=[1, 7, 30],
            seasonal_features=True,
            trend_features=True
        )
    
    def _recommend_feature_selection(
        self,
        data: pd.DataFrame,
        feature_analysis: Dict[str, FeatureInfo],
        target_column: Optional[str],
        model_type: ModelType
    ) -> List[str]:
        """Recommend feature selection strategies."""
        recommendations = []
        
        # High correlation features
        high_corr_features = [name for name, info in feature_analysis.items() 
                            if abs(info.correlation_with_target or 0) > 0.3]
        
        if high_corr_features:
            recommendations.append(f"Select top {min(20, len(high_corr_features))} features by correlation")
        
        # Low variance features
        low_variance_features = [name for name, info in feature_analysis.items() 
                               if info.type == FeatureType.NUMERICAL and 
                               info.distribution_stats.get('std', 1) < 0.01]
        
        if low_variance_features:
            recommendations.append(f"Remove {len(low_variance_features)} low variance features")
        
        # High cardinality categorical features
        high_card_features = [name for name, info in feature_analysis.items() 
                            if info.type == FeatureType.CATEGORICAL and 
                            (info.cardinality or 0) > 100]
        
        if high_card_features:
            recommendations.append(f"Consider removing or encoding {len(high_card_features)} high cardinality features")
        
        return recommendations
    
    def _recommend_dimensionality_reduction(
        self,
        feature_analysis: Dict[str, FeatureInfo],
        model_type: ModelType
    ) -> Optional[str]:
        """Recommend dimensionality reduction techniques."""
        numerical_features = [name for name, info in feature_analysis.items() 
                            if info.type == FeatureType.NUMERICAL]
        
        if len(numerical_features) > 50:
            if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION]:
                return "PCA with 95% variance retention"
            elif model_type == ModelType.NEURAL_NETWORK:
                return "Autoencoder-based dimensionality reduction"
        
        return None
    
    def apply_transformations(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        transformations: List[TransformationStep]
    ) -> TransformedDataset:
        """
        Apply feature transformations to the dataset.
        
        Args:
            dataset_id: Original dataset identifier
            data: Input DataFrame
            transformations: List of transformations to apply
            
        Returns:
            TransformedDataset with applied transformations
        """
        try:
            logger.info(f"Applying {len(transformations)} transformations to dataset {dataset_id}")
            
            transformed_data = data.copy()
            feature_mapping = {}
            transformation_metadata = {}
            
            for i, transformation in enumerate(transformations):
                logger.info(f"Applying transformation {i+1}: {transformation.description}")
                
                # Apply transformation based on type
                result = self._apply_single_transformation(transformed_data, transformation)
                
                if result is not None:
                    transformed_data, metadata = result
                    feature_mapping[transformation.input_features[0]] = transformation.output_features
                    transformation_metadata[f"transformation_{i}"] = metadata
            
            # Calculate quality metrics
            quality_metrics = self._calculate_transformation_quality(data, transformed_data)
            
            return TransformedDataset(
                original_dataset_id=dataset_id,
                transformed_dataset_id=f"{dataset_id}_transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformations_applied=transformations,
                feature_mapping=feature_mapping,
                transformation_metadata=transformation_metadata,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Error applying transformations: {str(e)}")
            raise FeatureEngineeringError(f"Transformation failed: {str(e)}")
    
    def _apply_single_transformation(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply a single transformation step."""
        try:
            if transformation.transformation_type == TransformationType.SCALING:
                return self._apply_scaling(data, transformation)
            elif transformation.transformation_type == TransformationType.NORMALIZATION:
                return self._apply_normalization(data, transformation)
            elif transformation.transformation_type == TransformationType.ENCODING:
                return self._apply_encoding(data, transformation)
            elif transformation.transformation_type == TransformationType.BINNING:
                return self._apply_binning(data, transformation)
            elif transformation.transformation_type == TransformationType.POLYNOMIAL:
                return self._apply_polynomial(data, transformation)
            elif transformation.transformation_type == TransformationType.INTERACTION:
                return self._apply_interaction(data, transformation)
            elif transformation.transformation_type == TransformationType.EXTRACTION:
                return self._apply_extraction(data, transformation)
            else:
                logger.warning(f"Unknown transformation type: {transformation.transformation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying transformation {transformation.description}: {str(e)}")
            return None
    
    def _apply_scaling(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply scaling transformation."""
        method = transformation.parameters.get("method", "standard")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if method == "standard":
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(data[[input_feature]])
            data[output_feature] = scaled_values.flatten()
            metadata = {"scaler_mean": scaler.mean_[0], "scaler_scale": scaler.scale_[0]}
        elif method == "minmax":
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(data[[input_feature]])
            data[output_feature] = scaled_values.flatten()
            metadata = {"scaler_min": scaler.min_[0], "scaler_scale": scaler.scale_[0]}
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        return data, metadata
    
    def _apply_normalization(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply normalization transformation."""
        method = transformation.parameters.get("method", "min_max")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if method == "min_max":
            min_val = data[input_feature].min()
            max_val = data[input_feature].max()
            data[output_feature] = (data[input_feature] - min_val) / (max_val - min_val)
            metadata = {"min_value": min_val, "max_value": max_val}
        elif method == "z_score":
            mean_val = data[input_feature].mean()
            std_val = data[input_feature].std()
            data[output_feature] = (data[input_feature] - mean_val) / std_val
            metadata = {"mean_value": mean_val, "std_value": std_val}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return data, metadata
    
    def _apply_encoding(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply encoding transformation."""
        method = transformation.parameters.get("method", "one_hot")
        input_feature = transformation.input_features[0]
        
        if method == "one_hot":
            encoded = pd.get_dummies(data[input_feature], prefix=input_feature)
            for col in encoded.columns:
                data[col] = encoded[col]
            metadata = {"encoded_columns": list(encoded.columns)}
        elif method == "label":
            encoder = LabelEncoder()
            output_feature = transformation.output_features[0]
            data[output_feature] = encoder.fit_transform(data[input_feature].astype(str))
            metadata = {"label_mapping": dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return data, metadata
    
    def _apply_binning(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply binning transformation."""
        n_bins = transformation.parameters.get("n_bins", 10)
        strategy = transformation.parameters.get("strategy", "quantile")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if strategy == "quantile":
            data[output_feature] = pd.qcut(data[input_feature], q=n_bins, duplicates='drop')
        elif strategy == "uniform":
            data[output_feature] = pd.cut(data[input_feature], bins=n_bins)
        else:
            raise ValueError(f"Unknown binning strategy: {strategy}")
        
        # Convert to numeric labels
        data[output_feature] = data[output_feature].cat.codes
        
        metadata = {"n_bins": n_bins, "strategy": strategy}
        return data, metadata
    
    def _apply_polynomial(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply polynomial transformation."""
        degree = transformation.parameters.get("degree", 2)
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        data[output_feature] = data[input_feature] ** degree
        metadata = {"degree": degree}
        
        return data, metadata
    
    def _apply_interaction(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply interaction transformation."""
        operation = transformation.parameters.get("operation", "multiply")
        input_features = transformation.input_features
        output_feature = transformation.output_features[0]
        
        if len(input_features) != 2:
            raise ValueError("Interaction transformation requires exactly 2 input features")
        
        if operation == "multiply":
            data[output_feature] = data[input_features[0]] * data[input_features[1]]
        elif operation == "add":
            data[output_feature] = data[input_features[0]] + data[input_features[1]]
        elif operation == "subtract":
            data[output_feature] = data[input_features[0]] - data[input_features[1]]
        elif operation == "divide":
            data[output_feature] = data[input_features[0]] / (data[input_features[1]] + 1e-8)  # Avoid division by zero
        else:
            raise ValueError(f"Unknown interaction operation: {operation}")
        
        metadata = {"operation": operation, "input_features": input_features}
        return data, metadata
    
    def _apply_extraction(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply feature extraction transformation."""
        method = transformation.parameters.get("method", "create_missing_indicator")
        input_feature = transformation.input_features[0]
        
        if method == "create_missing_indicator":
            output_feature = transformation.output_features[0]
            data[output_feature] = data[input_feature].isnull().astype(int)
            metadata = {"missing_count": data[output_feature].sum()}
        elif method == "extract_temporal":
            extract_components = transformation.parameters.get("extract", ["hour", "day", "month", "year"])
            metadata = {"extracted_components": extract_components}
            
            for i, component in enumerate(extract_components):
                output_feature = transformation.output_features[i]
                if component == "hour":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.hour
                elif component == "day":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.day
                elif component == "month":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.month
                elif component == "year":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.year
                elif component == "dayofweek":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.dayofweek
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        return data, metadata
    
    def _calculate_transformation_quality(
        self, 
        original_data: pd.DataFrame, 
        transformed_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate quality metrics for transformations."""
        metrics = {}
        
        # Feature count change
        original_features = len(original_data.columns)
        transformed_features = len(transformed_data.columns)
        metrics["feature_count_change"] = (transformed_features - original_features) / original_features
        
        # Data completeness
        original_completeness = 1 - original_data.isnull().sum().sum() / (original_data.shape[0] * original_data.shape[1])
        transformed_completeness = 1 - transformed_data.isnull().sum().sum() / (transformed_data.shape[0] * transformed_data.shape[1])
        metrics["completeness_change"] = transformed_completeness - original_completeness
        
        # Memory usage change
        original_memory = original_data.memory_usage(deep=True).sum()
        transformed_memory = transformed_data.memory_usage(deep=True).sum()
        metrics["memory_usage_change"] = (transformed_memory - original_memory) / original_memory
        
        return metrics
    
    def optimize_encoding(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        categorical_columns: List[str],
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> EncodingStrategy:
        """
        Optimize categorical encoding strategies for specific model types.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            categorical_columns: List of categorical column names
            model_type: Target model type for optimization
            
        Returns:
            Optimized encoding strategy
        """
        try:
            logger.info(f"Optimizing encoding for {len(categorical_columns)} categorical features")
            
            strategies = []
            
            for column in categorical_columns:
                if column not in data.columns:
                    continue
                
                cardinality = data[column].nunique()
                missing_rate = data[column].isnull().sum() / len(data)
                
                # Choose encoding based on cardinality and model type
                if cardinality <= 2:
                    # Binary encoding for binary features
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="binary",
                        parameters={},
                        expected_dimensions=1,
                        handle_unknown="ignore"
                    )
                elif cardinality <= 10:
                    # One-hot encoding for low cardinality
                    if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION, ModelType.NEURAL_NETWORK]:
                        strategy = EncodingStrategy(
                            feature_name=column,
                            encoding_type="one_hot",
                            parameters={"drop_first": True},
                            expected_dimensions=cardinality - 1,
                            handle_unknown="ignore"
                        )
                    else:
                        # Label encoding for tree-based models
                        strategy = EncodingStrategy(
                            feature_name=column,
                            encoding_type="label",
                            parameters={},
                            expected_dimensions=1,
                            handle_unknown="ignore"
                        )
                elif cardinality <= 50:
                    # Binary encoding for medium cardinality
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="binary",
                        parameters={},
                        expected_dimensions=int(np.ceil(np.log2(cardinality))),
                        handle_unknown="ignore"
                    )
                else:
                    # Target encoding for high cardinality
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="target",
                        parameters={"smoothing": 1.0},
                        expected_dimensions=1,
                        handle_unknown="ignore"
                    )
                
                strategies.append(strategy)
            
            # Return the first strategy for now (could be enhanced to return all)
            return strategies[0] if strategies else None
            
        except Exception as e:
            logger.error(f"Error optimizing encoding: {str(e)}")
            raise FeatureEngineeringError(f"Encoding optimization failed: {str(e)}")
    
    def generate_temporal_features(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        time_column: str
    ) -> TemporalFeatures:
        """
        Generate temporal features from time column.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            time_column: Name of the time column
            
        Returns:
            TemporalFeatures configuration
        """
        try:
            logger.info(f"Generating temporal features from column: {time_column}")
            
            if time_column not in data.columns:
                raise ValueError(f"Time column '{time_column}' not found in dataset")
            
            # Convert to datetime if needed
            time_series = pd.to_datetime(data[time_column])
            
            # Determine appropriate features based on data characteristics
            features_to_create = ["hour", "day", "month", "year", "dayofweek"]
            
            # Check for sub-daily data
            if time_series.dt.hour.nunique() > 1:
                features_to_create.extend(["hour", "minute"])
            
            # Check for seasonal patterns
            if len(time_series) >= 365:
                features_to_create.extend(["quarter", "is_weekend", "is_month_start", "is_month_end"])
            
            # Determine appropriate aggregation windows
            time_range = time_series.max() - time_series.min()
            aggregation_windows = []
            
            if time_range.days >= 7:
                aggregation_windows.append("1D")
            if time_range.days >= 30:
                aggregation_windows.append("1W")
            if time_range.days >= 365:
                aggregation_windows.extend(["1M", "1Q"])
            
            # Determine lag features
            lag_features = []
            if len(time_series) >= 7:
                lag_features.append(1)
            if len(time_series) >= 30:
                lag_features.append(7)
            if len(time_series) >= 365:
                lag_features.append(30)
            
            return TemporalFeatures(
                time_column=time_column,
                features_to_create=list(set(features_to_create)),
                aggregation_windows=aggregation_windows,
                lag_features=lag_features,
                seasonal_features=len(time_series) >= 365,
                trend_features=len(time_series) >= 30
            )
            
        except Exception as e:
            logger.error(f"Error generating temporal features: {str(e)}")
            raise FeatureEngineeringError(f"Temporal feature generation failed: {str(e)}")  
  
    def apply_transformations(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        transformations: List[TransformationStep]
    ) -> TransformedDataset:
        """
        Apply feature transformations to the dataset.
        
        Args:
            dataset_id: Original dataset identifier
            data: Input DataFrame
            transformations: List of transformations to apply
            
        Returns:
            TransformedDataset with applied transformations
        """
        try:
            logger.info(f"Applying {len(transformations)} transformations to dataset {dataset_id}")
            
            transformed_data = data.copy()
            feature_mapping = {}
            transformation_metadata = {}
            
            for i, transformation in enumerate(transformations):
                logger.info(f"Applying transformation {i+1}: {transformation.description}")
                
                # Apply transformation based on type
                result = self._apply_single_transformation(transformed_data, transformation)
                
                if result is not None:
                    transformed_data, metadata = result
                    feature_mapping[transformation.input_features[0]] = transformation.output_features
                    transformation_metadata[f"transformation_{i}"] = metadata
            
            # Calculate quality metrics
            quality_metrics = self._calculate_transformation_quality(data, transformed_data)
            
            return TransformedDataset(
                original_dataset_id=dataset_id,
                transformed_dataset_id=f"{dataset_id}_transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformations_applied=transformations,
                feature_mapping=feature_mapping,
                transformation_metadata=transformation_metadata,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Error applying transformations: {str(e)}")
            raise FeatureEngineeringError(f"Transformation failed: {str(e)}")
    
    def _apply_single_transformation(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply a single transformation step."""
        try:
            if transformation.transformation_type == TransformationType.SCALING:
                return self._apply_scaling(data, transformation)
            elif transformation.transformation_type == TransformationType.NORMALIZATION:
                return self._apply_normalization(data, transformation)
            elif transformation.transformation_type == TransformationType.ENCODING:
                return self._apply_encoding(data, transformation)
            elif transformation.transformation_type == TransformationType.BINNING:
                return self._apply_binning(data, transformation)
            elif transformation.transformation_type == TransformationType.POLYNOMIAL:
                return self._apply_polynomial(data, transformation)
            elif transformation.transformation_type == TransformationType.INTERACTION:
                return self._apply_interaction(data, transformation)
            elif transformation.transformation_type == TransformationType.EXTRACTION:
                return self._apply_extraction(data, transformation)
            else:
                logger.warning(f"Unknown transformation type: {transformation.transformation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying transformation {transformation.description}: {str(e)}")
            return None
    
    def _apply_scaling(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply scaling transformation."""
        method = transformation.parameters.get("method", "standard")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if method == "standard":
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(data[[input_feature]])
            data[output_feature] = scaled_values.flatten()
            metadata = {"scaler_mean": scaler.mean_[0], "scaler_scale": scaler.scale_[0]}
        elif method == "minmax":
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(data[[input_feature]])
            data[output_feature] = scaled_values.flatten()
            metadata = {"scaler_min": scaler.min_[0], "scaler_scale": scaler.scale_[0]}
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        return data, metadata
    
    def _apply_normalization(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply normalization transformation."""
        method = transformation.parameters.get("method", "min_max")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if method == "min_max":
            min_val = data[input_feature].min()
            max_val = data[input_feature].max()
            data[output_feature] = (data[input_feature] - min_val) / (max_val - min_val)
            metadata = {"min_value": min_val, "max_value": max_val}
        elif method == "z_score":
            mean_val = data[input_feature].mean()
            std_val = data[input_feature].std()
            data[output_feature] = (data[input_feature] - mean_val) / std_val
            metadata = {"mean_value": mean_val, "std_value": std_val}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return data, metadata
    
    def _apply_encoding(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply encoding transformation."""
        method = transformation.parameters.get("method", "one_hot")
        input_feature = transformation.input_features[0]
        
        if method == "one_hot":
            encoded = pd.get_dummies(data[input_feature], prefix=input_feature)
            for col in encoded.columns:
                data[col] = encoded[col]
            metadata = {"encoded_columns": list(encoded.columns)}
        elif method == "label":
            encoder = LabelEncoder()
            output_feature = transformation.output_features[0]
            data[output_feature] = encoder.fit_transform(data[input_feature].astype(str))
            metadata = {"label_mapping": dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return data, metadata
    
    def _apply_binning(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply binning transformation."""
        n_bins = transformation.parameters.get("n_bins", 10)
        strategy = transformation.parameters.get("strategy", "quantile")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if strategy == "quantile":
            data[output_feature] = pd.qcut(data[input_feature], q=n_bins, duplicates='drop')
        elif strategy == "uniform":
            data[output_feature] = pd.cut(data[input_feature], bins=n_bins)
        else:
            raise ValueError(f"Unknown binning strategy: {strategy}")
        
        # Convert to numeric labels
        data[output_feature] = data[output_feature].cat.codes
        
        metadata = {"n_bins": n_bins, "strategy": strategy}
        return data, metadata
    
    def _apply_polynomial(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply polynomial transformation."""
        degree = transformation.parameters.get("degree", 2)
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        data[output_feature] = data[input_feature] ** degree
        metadata = {"degree": degree}
        
        return data, metadata
    
    def _apply_interaction(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply interaction transformation."""
        operation = transformation.parameters.get("operation", "multiply")
        input_features = transformation.input_features
        output_feature = transformation.output_features[0]
        
        if len(input_features) != 2:
            raise ValueError("Interaction transformation requires exactly 2 input features")
        
        if operation == "multiply":
            data[output_feature] = data[input_features[0]] * data[input_features[1]]
        elif operation == "add":
            data[output_feature] = data[input_features[0]] + data[input_features[1]]
        elif operation == "subtract":
            data[output_feature] = data[input_features[0]] - data[input_features[1]]
        elif operation == "divide":
            data[output_feature] = data[input_features[0]] / (data[input_features[1]] + 1e-8)  # Avoid division by zero
        else:
            raise ValueError(f"Unknown interaction operation: {operation}")
        
        metadata = {"operation": operation, "input_features": input_features}
        return data, metadata
    
    def _apply_extraction(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply feature extraction transformation."""
        method = transformation.parameters.get("method", "create_missing_indicator")
        input_feature = transformation.input_features[0]
        
        if method == "create_missing_indicator":
            output_feature = transformation.output_features[0]
            data[output_feature] = data[input_feature].isnull().astype(int)
            metadata = {"missing_count": data[output_feature].sum()}
        elif method == "extract_temporal":
            extract_components = transformation.parameters.get("extract", ["hour", "day", "month", "year"])
            metadata = {"extracted_components": extract_components}
            
            for i, component in enumerate(extract_components):
                output_feature = transformation.output_features[i]
                if component == "hour":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.hour
                elif component == "day":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.day
                elif component == "month":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.month
                elif component == "year":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.year
                elif component == "dayofweek":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.dayofweek
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        return data, metadata
    
    def _calculate_transformation_quality(
        self, 
        original_data: pd.DataFrame, 
        transformed_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate quality metrics for transformations."""
        metrics = {}
        
        # Feature count change
        original_features = len(original_data.columns)
        transformed_features = len(transformed_data.columns)
        metrics["feature_count_change"] = (transformed_features - original_features) / original_features
        
        # Data completeness
        original_completeness = 1 - original_data.isnull().sum().sum() / (original_data.shape[0] * original_data.shape[1])
        transformed_completeness = 1 - transformed_data.isnull().sum().sum() / (transformed_data.shape[0] * transformed_data.shape[1])
        metrics["completeness_change"] = transformed_completeness - original_completeness
        
        # Memory usage change
        original_memory = original_data.memory_usage(deep=True).sum()
        transformed_memory = transformed_data.memory_usage(deep=True).sum()
        metrics["memory_usage_change"] = (transformed_memory - original_memory) / original_memory
        
        return metrics
    
    def optimize_encoding(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        categorical_columns: List[str],
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> EncodingStrategy:
        """
        Optimize categorical encoding strategies for specific model types.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            categorical_columns: List of categorical column names
            model_type: Target model type for optimization
            
        Returns:
            Optimized encoding strategy
        """
        try:
            logger.info(f"Optimizing encoding for {len(categorical_columns)} categorical features")
            
            strategies = []
            
            for column in categorical_columns:
                if column not in data.columns:
                    continue
                
                cardinality = data[column].nunique()
                missing_rate = data[column].isnull().sum() / len(data)
                
                # Choose encoding based on cardinality and model type
                if cardinality <= 2:
                    # Binary encoding for binary features
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="binary",
                        parameters={},
                        expected_dimensions=1,
                        handle_unknown="ignore"
                    )
                elif cardinality <= 10:
                    # One-hot encoding for low cardinality
                    if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION, ModelType.NEURAL_NETWORK]:
                        strategy = EncodingStrategy(
                            feature_name=column,
                            encoding_type="one_hot",
                            parameters={"drop_first": True},
                            expected_dimensions=cardinality - 1,
                            handle_unknown="ignore"
                        )
                    else:
                        # Label encoding for tree-based models
                        strategy = EncodingStrategy(
                            feature_name=column,
                            encoding_type="label",
                            parameters={},
                            expected_dimensions=1,
                            handle_unknown="ignore"
                        )
                elif cardinality <= 50:
                    # Binary encoding for medium cardinality
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="binary",
                        parameters={},
                        expected_dimensions=int(np.ceil(np.log2(cardinality))),
                        handle_unknown="ignore"
                    )
                else:
                    # Target encoding for high cardinality
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="target",
                        parameters={"smoothing": 1.0},
                        expected_dimensions=1,
                        handle_unknown="ignore"
                    )
                
                strategies.append(strategy)
            
            # Return the first strategy for now (could be enhanced to return all)
            return strategies[0] if strategies else None
            
        except Exception as e:
            logger.error(f"Error optimizing encoding: {str(e)}")
            raise FeatureEngineeringError(f"Encoding optimization failed: {str(e)}")
    
    def generate_temporal_features(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        time_column: str
    ) -> TemporalFeatures:
        """
        Generate temporal features from time column.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            time_column: Name of the time column
            
        Returns:
            TemporalFeatures configuration
        """
        try:
            logger.info(f"Generating temporal features from column: {time_column}")
            
            if time_column not in data.columns:
                raise ValueError(f"Time column '{time_column}' not found in dataset")
            
            # Convert to datetime if needed
            time_series = pd.to_datetime(data[time_column])
            
            # Determine appropriate features based on data characteristics
            features_to_create = ["hour", "day", "month", "year", "dayofweek"]
            
            # Check for sub-daily data
            if time_series.dt.hour.nunique() > 1:
                features_to_create.extend(["hour", "minute"])
            
            # Check for seasonal patterns
            if len(time_series) >= 365:
                features_to_create.extend(["quarter", "is_weekend", "is_month_start", "is_month_end"])
            
            # Determine appropriate aggregation windows
            time_range = time_series.max() - time_series.min()
            aggregation_windows = []
            
            if time_range.days >= 7:
                aggregation_windows.append("1D")
            if time_range.days >= 30:
                aggregation_windows.append("1W")
            if time_range.days >= 365:
                aggregation_windows.extend(["1M", "1Q"])
            
            # Determine lag features
            lag_features = []
            if len(time_series) >= 7:
                lag_features.append(1)
            if len(time_series) >= 30:
                lag_features.append(7)
            if len(time_series) >= 365:
                lag_features.append(30)
            
            return TemporalFeatures(
                time_column=time_column,
                features_to_create=list(set(features_to_create)),
                aggregation_windows=aggregation_windows,
                lag_features=lag_features,
                seasonal_features=len(time_series) >= 365,
                trend_features=len(time_series) >= 30
            )
            
        except Exception as e:
            logger.error(f"Error generating temporal features: {str(e)}")
            raise FeatureEngineeringError(f"Temporal feature generation failed: {str(e)}")
    
    def apply_transformations(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        transformations: List[TransformationStep]
    ) -> TransformedDataset:
        """
        Apply feature transformations to the dataset.
        
        Args:
            dataset_id: Original dataset identifier
            data: Input DataFrame
            transformations: List of transformations to apply
            
        Returns:
            TransformedDataset with applied transformations
        """
        try:
            logger.info(f"Applying {len(transformations)} transformations to dataset {dataset_id}")
            
            transformed_data = data.copy()
            feature_mapping = {}
            transformation_metadata = {}
            
            for i, transformation in enumerate(transformations):
                logger.info(f"Applying transformation {i+1}: {transformation.description}")
                
                # Apply transformation based on type
                result = self._apply_single_transformation(transformed_data, transformation)
                
                if result is not None:
                    transformed_data, metadata = result
                    feature_mapping[transformation.input_features[0]] = transformation.output_features
                    transformation_metadata[f"transformation_{i}"] = metadata
            
            # Calculate quality metrics
            quality_metrics = self._calculate_transformation_quality(data, transformed_data)
            
            return TransformedDataset(
                original_dataset_id=dataset_id,
                transformed_dataset_id=f"{dataset_id}_transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformations_applied=transformations,
                feature_mapping=feature_mapping,
                transformation_metadata=transformation_metadata,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Error applying transformations: {str(e)}")
            raise FeatureEngineeringError(f"Transformation failed: {str(e)}")
    
    def _apply_single_transformation(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply a single transformation step."""
        try:
            if transformation.transformation_type == TransformationType.SCALING:
                return self._apply_scaling(data, transformation)
            elif transformation.transformation_type == TransformationType.NORMALIZATION:
                return self._apply_normalization(data, transformation)
            elif transformation.transformation_type == TransformationType.ENCODING:
                return self._apply_encoding(data, transformation)
            elif transformation.transformation_type == TransformationType.BINNING:
                return self._apply_binning(data, transformation)
            elif transformation.transformation_type == TransformationType.POLYNOMIAL:
                return self._apply_polynomial(data, transformation)
            elif transformation.transformation_type == TransformationType.INTERACTION:
                return self._apply_interaction(data, transformation)
            elif transformation.transformation_type == TransformationType.EXTRACTION:
                return self._apply_extraction(data, transformation)
            else:
                logger.warning(f"Unknown transformation type: {transformation.transformation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying transformation {transformation.description}: {str(e)}")
            return None
    
    def _apply_scaling(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply scaling transformation."""
        method = transformation.parameters.get("method", "standard")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if method == "standard":
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(data[[input_feature]])
            data[output_feature] = scaled_values.flatten()
            metadata = {"scaler_mean": scaler.mean_[0], "scaler_scale": scaler.scale_[0]}
        elif method == "minmax":
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(data[[input_feature]])
            data[output_feature] = scaled_values.flatten()
            metadata = {"scaler_min": scaler.min_[0], "scaler_scale": scaler.scale_[0]}
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        return data, metadata
    
    def _apply_normalization(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply normalization transformation."""
        method = transformation.parameters.get("method", "min_max")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if method == "min_max":
            min_val = data[input_feature].min()
            max_val = data[input_feature].max()
            data[output_feature] = (data[input_feature] - min_val) / (max_val - min_val)
            metadata = {"min_value": min_val, "max_value": max_val}
        elif method == "z_score":
            mean_val = data[input_feature].mean()
            std_val = data[input_feature].std()
            data[output_feature] = (data[input_feature] - mean_val) / std_val
            metadata = {"mean_value": mean_val, "std_value": std_val}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return data, metadata
    
    def _apply_encoding(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply encoding transformation."""
        method = transformation.parameters.get("method", "one_hot")
        input_feature = transformation.input_features[0]
        
        if method == "one_hot":
            encoded = pd.get_dummies(data[input_feature], prefix=input_feature)
            for col in encoded.columns:
                data[col] = encoded[col]
            metadata = {"encoded_columns": list(encoded.columns)}
        elif method == "label":
            encoder = LabelEncoder()
            output_feature = transformation.output_features[0]
            data[output_feature] = encoder.fit_transform(data[input_feature].astype(str))
            metadata = {"label_mapping": dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return data, metadata
    
    def _apply_binning(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply binning transformation."""
        n_bins = transformation.parameters.get("n_bins", 10)
        strategy = transformation.parameters.get("strategy", "quantile")
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        if strategy == "quantile":
            data[output_feature] = pd.qcut(data[input_feature], q=n_bins, duplicates='drop')
        elif strategy == "uniform":
            data[output_feature] = pd.cut(data[input_feature], bins=n_bins)
        else:
            raise ValueError(f"Unknown binning strategy: {strategy}")
        
        # Convert to numeric labels
        data[output_feature] = data[output_feature].cat.codes
        
        metadata = {"n_bins": n_bins, "strategy": strategy}
        return data, metadata
    
    def _apply_polynomial(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply polynomial transformation."""
        degree = transformation.parameters.get("degree", 2)
        input_feature = transformation.input_features[0]
        output_feature = transformation.output_features[0]
        
        data[output_feature] = data[input_feature] ** degree
        metadata = {"degree": degree}
        
        return data, metadata
    
    def _apply_interaction(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply interaction transformation."""
        operation = transformation.parameters.get("operation", "multiply")
        input_features = transformation.input_features
        output_feature = transformation.output_features[0]
        
        if len(input_features) != 2:
            raise ValueError("Interaction transformation requires exactly 2 input features")
        
        if operation == "multiply":
            data[output_feature] = data[input_features[0]] * data[input_features[1]]
        elif operation == "add":
            data[output_feature] = data[input_features[0]] + data[input_features[1]]
        elif operation == "subtract":
            data[output_feature] = data[input_features[0]] - data[input_features[1]]
        elif operation == "divide":
            data[output_feature] = data[input_features[0]] / (data[input_features[1]] + 1e-8)  # Avoid division by zero
        else:
            raise ValueError(f"Unknown interaction operation: {operation}")
        
        metadata = {"operation": operation, "input_features": input_features}
        return data, metadata
    
    def _apply_extraction(
        self, 
        data: pd.DataFrame, 
        transformation: TransformationStep
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply feature extraction transformation."""
        method = transformation.parameters.get("method", "create_missing_indicator")
        input_feature = transformation.input_features[0]
        
        if method == "create_missing_indicator":
            output_feature = transformation.output_features[0]
            data[output_feature] = data[input_feature].isnull().astype(int)
            metadata = {"missing_count": data[output_feature].sum()}
        elif method == "extract_temporal":
            extract_components = transformation.parameters.get("extract", ["hour", "day", "month", "year"])
            metadata = {"extracted_components": extract_components}
            
            for i, component in enumerate(extract_components):
                output_feature = transformation.output_features[i]
                if component == "hour":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.hour
                elif component == "day":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.day
                elif component == "month":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.month
                elif component == "year":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.year
                elif component == "dayofweek":
                    data[output_feature] = pd.to_datetime(data[input_feature]).dt.dayofweek
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        return data, metadata
    
    def _calculate_transformation_quality(
        self, 
        original_data: pd.DataFrame, 
        transformed_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate quality metrics for transformations."""
        metrics = {}
        
        # Feature count change
        original_features = len(original_data.columns)
        transformed_features = len(transformed_data.columns)
        metrics["feature_count_change"] = (transformed_features - original_features) / original_features
        
        # Data completeness
        original_completeness = 1 - original_data.isnull().sum().sum() / (original_data.shape[0] * original_data.shape[1])
        transformed_completeness = 1 - transformed_data.isnull().sum().sum() / (transformed_data.shape[0] * transformed_data.shape[1])
        metrics["completeness_change"] = transformed_completeness - original_completeness
        
        # Memory usage change
        original_memory = original_data.memory_usage(deep=True).sum()
        transformed_memory = transformed_data.memory_usage(deep=True).sum()
        metrics["memory_usage_change"] = (transformed_memory - original_memory) / original_memory
        
        return metrics
    
    def optimize_encoding(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        categorical_columns: List[str],
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> EncodingStrategy:
        """
        Optimize categorical encoding strategies for specific model types.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            categorical_columns: List of categorical column names
            model_type: Target model type for optimization
            
        Returns:
            Optimized encoding strategy
        """
        try:
            logger.info(f"Optimizing encoding for {len(categorical_columns)} categorical features")
            
            strategies = []
            
            for column in categorical_columns:
                if column not in data.columns:
                    continue
                
                cardinality = data[column].nunique()
                missing_rate = data[column].isnull().sum() / len(data)
                
                # Choose encoding based on cardinality and model type
                if cardinality <= 2:
                    # Binary encoding for binary features
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="binary",
                        parameters={},
                        expected_dimensions=1,
                        handle_unknown="ignore"
                    )
                elif cardinality <= 10:
                    # One-hot encoding for low cardinality
                    if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION, ModelType.NEURAL_NETWORK]:
                        strategy = EncodingStrategy(
                            feature_name=column,
                            encoding_type="one_hot",
                            parameters={"drop_first": True},
                            expected_dimensions=cardinality - 1,
                            handle_unknown="ignore"
                        )
                    else:
                        # Label encoding for tree-based models
                        strategy = EncodingStrategy(
                            feature_name=column,
                            encoding_type="label",
                            parameters={},
                            expected_dimensions=1,
                            handle_unknown="ignore"
                        )
                elif cardinality <= 50:
                    # Binary encoding for medium cardinality
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="binary",
                        parameters={},
                        expected_dimensions=int(np.ceil(np.log2(cardinality))),
                        handle_unknown="ignore"
                    )
                else:
                    # Target encoding for high cardinality
                    strategy = EncodingStrategy(
                        feature_name=column,
                        encoding_type="target",
                        parameters={"smoothing": 1.0},
                        expected_dimensions=1,
                        handle_unknown="ignore"
                    )
                
                strategies.append(strategy)
            
            # Return the first strategy for now (could be enhanced to return all)
            return strategies[0] if strategies else None
            
        except Exception as e:
            logger.error(f"Error optimizing encoding: {str(e)}")
            raise FeatureEngineeringError(f"Encoding optimization failed: {str(e)}")
    
    def generate_temporal_features(
        self, 
        dataset_id: str, 
        data: pd.DataFrame,
        time_column: str
    ) -> TemporalFeatures:
        """
        Generate temporal features from time column.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            time_column: Name of the time column
            
        Returns:
            TemporalFeatures configuration
        """
        try:
            logger.info(f"Generating temporal features from column: {time_column}")
            
            if time_column not in data.columns:
                raise ValueError(f"Time column '{time_column}' not found in dataset")
            
            # Convert to datetime if needed
            time_series = pd.to_datetime(data[time_column])
            
            # Determine appropriate features based on data characteristics
            features_to_create = ["hour", "day", "month", "year", "dayofweek"]
            
            # Check for sub-daily data
            if time_series.dt.hour.nunique() > 1:
                features_to_create.extend(["hour", "minute"])
            
            # Check for seasonal patterns
            if len(time_series) >= 365:
                features_to_create.extend(["quarter", "is_weekend", "is_month_start", "is_month_end"])
            
            # Determine appropriate aggregation windows
            time_range = time_series.max() - time_series.min()
            aggregation_windows = []
            
            if time_range.days >= 7:
                aggregation_windows.append("1D")
            if time_range.days >= 30:
                aggregation_windows.append("1W")
            if time_range.days >= 365:
                aggregation_windows.extend(["1M", "1Q"])
            
            # Determine lag features
            lag_features = []
            if len(time_series) >= 7:
                lag_features.append(1)
            if len(time_series) >= 30:
                lag_features.append(7)
            if len(time_series) >= 365:
                lag_features.append(30)
            
            return TemporalFeatures(
                time_column=time_column,
                features_to_create=list(set(features_to_create)),
                aggregation_windows=aggregation_windows,
                lag_features=lag_features,
                seasonal_features=len(time_series) >= 365,
                trend_features=len(time_series) >= 30
            )
            
        except Exception as e:
            logger.error(f"Error generating temporal features: {str(e)}")
            raise FeatureEngineeringError(f"Temporal feature generation failed: {str(e)}")  
  
    def apply_advanced_transformation_pipeline(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        pipeline_config: Optional[PipelineConfig] = None
    ) -> TransformedDataset:
        """
        Apply advanced transformation pipeline with comprehensive feature engineering.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            target_column: Target variable column name
            model_type: Type of ML model for optimization
            pipeline_config: Configuration for the transformation pipeline
            
        Returns:
            TransformedDataset with comprehensive transformations applied
        """
        try:
            logger.info(f"Applying advanced transformation pipeline to dataset {dataset_id}")
            
            # Configure pipeline if not provided
            if pipeline_config:
                self.transformation_pipeline = AdvancedTransformationPipeline(pipeline_config)
            
            # Apply the comprehensive pipeline
            transformed_data, metadata = self.transformation_pipeline.fit_transform_pipeline(
                data, target_column, model_type
            )
            
            # Create transformation steps for compatibility
            transformation_steps = []
            for technique in metadata.get("transformations_applied", []):
                transformation_steps.append(TransformationStep(
                    transformation_type=TransformationType.EXTRACTION,  # Generic type
                    parameters={"technique": technique},
                    input_features=list(data.columns),
                    output_features=list(transformed_data.columns),
                    description=f"Applied {technique}",
                    rationale=f"Advanced pipeline transformation: {technique}"
                ))
            
            return TransformedDataset(
                original_dataset_id=dataset_id,
                transformed_dataset_id=f"{dataset_id}_advanced_transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformations_applied=transformation_steps,
                feature_mapping={col: [col] for col in data.columns if col in transformed_data.columns},
                transformation_metadata=metadata,
                quality_metrics=metadata.get("quality_metrics", {})
            )
            
        except Exception as e:
            logger.error(f"Error applying advanced transformation pipeline: {str(e)}")
            raise FeatureEngineeringError(f"Advanced transformation pipeline failed: {str(e)}")
    
    def generate_advanced_temporal_features(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        time_column: str,
        value_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        temporal_config: Optional[TemporalConfig] = None
    ) -> TemporalFeatures:
        """
        Generate comprehensive temporal features with advanced time-series engineering.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame with time series data
            time_column: Name of the time column
            value_columns: List of value columns to create features for
            target_column: Target variable column name
            temporal_config: Configuration for temporal feature generation
            
        Returns:
            TemporalFeatures configuration with comprehensive temporal features
        """
        try:
            logger.info(f"Generating advanced temporal features for dataset {dataset_id}")
            
            # Configure temporal generator if not provided
            if temporal_config:
                self.temporal_generator = AdvancedTemporalFeatureGenerator(temporal_config)
            
            # Generate comprehensive temporal features
            enhanced_data, metadata = self.temporal_generator.generate_comprehensive_temporal_features(
                data, time_column, value_columns, target_column
            )
            
            # Detect temporal patterns
            if value_columns:
                patterns = {}
                for col in value_columns[:3]:  # Limit to first 3 columns for performance
                    patterns[col] = self.temporal_generator.detect_temporal_patterns(
                        enhanced_data, time_column, col
                    )
                metadata["temporal_patterns"] = patterns
            
            # Create TemporalFeatures object
            return TemporalFeatures(
                time_column=time_column,
                features_to_create=metadata.get("features_created", []),
                aggregation_windows=["1h", "1d", "1w", "1m"],  # Standard windows
                lag_features=[1, 2, 3, 7, 14, 30],  # Standard lag periods
                seasonal_features=True,
                trend_features=True
            )
            
        except Exception as e:
            logger.error(f"Error generating advanced temporal features: {str(e)}")
            raise FeatureEngineeringError(f"Advanced temporal feature generation failed: {str(e)}")
    
    def apply_intelligent_dimensionality_reduction(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        reduction_config: Optional[DimensionalityReductionConfig] = None
    ) -> TransformedDataset:
        """
        Apply intelligent dimensionality reduction with automatic technique selection.
        
        Args:
            dataset_id: Dataset identifier
            data: Input DataFrame
            target_column: Target variable column name
            model_type: Type of ML model for optimization
            reduction_config: Configuration for dimensionality reduction
            
        Returns:
            TransformedDataset with dimensionality reduction applied
        """
        try:
            logger.info(f"Applying intelligent dimensionality reduction to dataset {dataset_id}")
            
            # Configure reducer if not provided
            if reduction_config:
                self.dimensionality_reducer = AdvancedDimensionalityReducer(reduction_config)
            
            # Get recommendations
            recommendations = self.dimensionality_reducer.recommend_dimensionality_reduction(
                data, target_column, model_type
            )
            
            # Apply recommended techniques
            reduced_data, metadata = self.dimensionality_reducer.apply_dimensionality_reduction(
                data, target_column, recommendations["recommended_techniques"], model_type
            )
            
            # Create transformation steps
            transformation_steps = []
            for technique in metadata.get("techniques_applied", []):
                transformation_steps.append(TransformationStep(
                    transformation_type=TransformationType.EXTRACTION,  # Generic type
                    parameters={"technique": technique},
                    input_features=list(data.columns),
                    output_features=list(reduced_data.columns),
                    description=f"Applied dimensionality reduction: {technique}",
                    rationale=f"Intelligent dimensionality reduction: {technique}"
                ))
            
            return TransformedDataset(
                original_dataset_id=dataset_id,
                transformed_dataset_id=f"{dataset_id}_dim_reduced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                transformations_applied=transformation_steps,
                feature_mapping={col: [col] for col in data.columns if col in reduced_data.columns},
                transformation_metadata=metadata,
                quality_metrics=metadata.get("quality_metrics", {})
            )
            
        except Exception as e:
            logger.error(f"Error applying dimensionality reduction: {str(e)}")
            raise FeatureEngineeringError(f"Dimensionality reduction failed: {str(e)}")
    
    def get_dimensionality_reduction_recommendations(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> Dict[str, Any]:
        """
        Get intelligent recommendations for dimensionality reduction.
        
        Args:
            data: Input DataFrame
            target_column: Target variable column name
            model_type: Type of ML model for optimization
            
        Returns:
            Dictionary with dimensionality reduction recommendations
        """
        try:
            return self.dimensionality_reducer.recommend_dimensionality_reduction(
                data, target_column, model_type
            )
        except Exception as e:
            logger.error(f"Error getting dimensionality reduction recommendations: {str(e)}")
            raise FeatureEngineeringError(f"Dimensionality reduction recommendation failed: {str(e)}")