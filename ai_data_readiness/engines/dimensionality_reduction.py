"""
Advanced Dimensionality Reduction for AI Data Readiness Platform.

This module provides comprehensive dimensionality reduction techniques
including PCA, ICA, t-SNE, feature selection, and intelligent recommendations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel, RFE, RFECV,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    chi2, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
import warnings

from ..models.feature_models import ModelType, FeatureType
from ..core.exceptions import FeatureEngineeringError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction."""
    target_variance_ratio: float = 0.95
    max_components: int = 50
    min_components: int = 2
    enable_feature_selection: bool = True
    enable_pca: bool = True
    enable_ica: bool = True
    enable_manifold_learning: bool = False
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01


class AdvancedDimensionalityReducer:
    """
    Advanced dimensionality reduction with multiple techniques and intelligent selection.
    
    Provides comprehensive dimensionality reduction including linear and non-linear
    methods, feature selection, and automatic technique recommendation.
    """
    
    def __init__(self, config: Optional[DimensionalityReductionConfig] = None):
        """Initialize the dimensionality reducer."""
        self.config = config or DimensionalityReductionConfig()
        self.fitted_reducers = {}
        self.feature_importance_scores = {}
        self.reduction_metadata = {}
        
    def recommend_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        feature_types: Optional[Dict[str, FeatureType]] = None
    ) -> Dict[str, Any]:
        """
        Recommend appropriate dimensionality reduction techniques.
        
        Args:
            data: Input DataFrame
            target_column: Target variable column name
            model_type: Type of ML model for optimization
            feature_types: Dictionary mapping feature names to types
            
        Returns:
            Dictionary with recommendations and analysis
        """
        try:
            logger.info("Analyzing data for dimensionality reduction recommendations")
            
            # Get numerical features for analysis
            numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numerical_features:
                numerical_features.remove(target_column)
            
            recommendations = {
                "analysis": {},
                "recommended_techniques": [],
                "feature_selection_recommendations": [],
                "dimensionality_assessment": {}
            }
            
            # Basic dimensionality analysis
            n_features = len(numerical_features)
            n_samples = len(data)
            
            recommendations["dimensionality_assessment"] = {
                "n_features": n_features,
                "n_samples": n_samples,
                "feature_to_sample_ratio": n_features / n_samples if n_samples > 0 else 0,
                "high_dimensionality": n_features > 50,
                "curse_of_dimensionality_risk": n_features > n_samples / 10
            }
            
            if n_features < 5:
                recommendations["recommended_techniques"] = ["none_needed"]
                return recommendations
            
            # Correlation analysis
            if len(numerical_features) > 1:
                corr_analysis = self._analyze_feature_correlations(data[numerical_features])
                recommendations["analysis"]["correlation"] = corr_analysis
                
                if corr_analysis["high_correlation_pairs"] > 0:
                    recommendations["recommended_techniques"].append("correlation_removal")
            
            # Variance analysis
            variance_analysis = self._analyze_feature_variance(data[numerical_features])
            recommendations["analysis"]["variance"] = variance_analysis
            
            if variance_analysis["low_variance_features"] > 0:
                recommendations["recommended_techniques"].append("variance_threshold")
            
            # Model-specific recommendations
            model_recommendations = self._get_model_specific_recommendations(
                model_type, n_features, n_samples
            )
            recommendations["recommended_techniques"].extend(model_recommendations)
            
            # Feature selection recommendations
            if target_column and target_column in data.columns:
                feature_selection_recs = self._recommend_feature_selection_methods(
                    data, target_column, model_type
                )
                recommendations["feature_selection_recommendations"] = feature_selection_recs
            
            # Remove duplicates and prioritize
            recommendations["recommended_techniques"] = list(set(recommendations["recommended_techniques"]))
            recommendations["recommended_techniques"] = self._prioritize_techniques(
                recommendations["recommended_techniques"], model_type, n_features, n_samples
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction recommendation: {str(e)}")
            raise FeatureEngineeringError(f"Dimensionality reduction recommendation failed: {str(e)}")
    
    def apply_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        techniques: Optional[List[str]] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply dimensionality reduction techniques to the data.
        
        Args:
            data: Input DataFrame
            target_column: Target variable column name
            techniques: List of techniques to apply
            model_type: Type of ML model for optimization
            
        Returns:
            Tuple of (reduced_data, reduction_metadata)
        """
        try:
            logger.info("Applying dimensionality reduction techniques")
            
            reduced_data = data.copy()
            metadata = {
                "original_shape": data.shape,
                "techniques_applied": [],
                "feature_reductions": {},
                "quality_metrics": {}
            }
            
            # Get numerical features
            numerical_features = reduced_data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numerical_features:
                numerical_features.remove(target_column)
            
            if not techniques:
                # Auto-recommend techniques
                recommendations = self.recommend_dimensionality_reduction(
                    reduced_data, target_column, model_type
                )
                techniques = recommendations["recommended_techniques"]
            
            # Apply each technique
            for technique in techniques:
                if technique == "none_needed":
                    continue
                    
                logger.info(f"Applying {technique}")
                
                if technique == "correlation_removal":
                    reduced_data, corr_metadata = self._remove_correlated_features(
                        reduced_data, numerical_features
                    )
                    metadata["techniques_applied"].append(technique)
                    metadata["feature_reductions"][technique] = corr_metadata
                    
                elif technique == "variance_threshold":
                    reduced_data, var_metadata = self._apply_variance_threshold(
                        reduced_data, numerical_features
                    )
                    metadata["techniques_applied"].append(technique)
                    metadata["feature_reductions"][technique] = var_metadata
                    
                elif technique == "pca":
                    reduced_data, pca_metadata = self._apply_pca(
                        reduced_data, numerical_features, target_column
                    )
                    metadata["techniques_applied"].append(technique)
                    metadata["feature_reductions"][technique] = pca_metadata
                    
                elif technique == "ica":
                    reduced_data, ica_metadata = self._apply_ica(
                        reduced_data, numerical_features
                    )
                    metadata["techniques_applied"].append(technique)
                    metadata["feature_reductions"][technique] = ica_metadata
                    
                elif technique == "feature_selection":
                    reduced_data, fs_metadata = self._apply_feature_selection(
                        reduced_data, target_column, model_type
                    )
                    metadata["techniques_applied"].append(technique)
                    metadata["feature_reductions"][technique] = fs_metadata
                
                # Update numerical features list
                numerical_features = reduced_data.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in numerical_features:
                    numerical_features.remove(target_column)
            
            # Calculate final quality metrics
            metadata["final_shape"] = reduced_data.shape
            metadata["quality_metrics"] = self._calculate_reduction_quality_metrics(
                data, reduced_data, target_column
            )
            
            logger.info(f"Dimensionality reduction completed: {data.shape} -> {reduced_data.shape}")
            return reduced_data, metadata
            
        except Exception as e:
            logger.error(f"Error applying dimensionality reduction: {str(e)}")
            raise FeatureEngineeringError(f"Dimensionality reduction failed: {str(e)}")
    
    def _analyze_feature_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature correlations."""
        corr_matrix = data.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.config.correlation_threshold:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j]
                    })
        
        return {
            "high_correlation_pairs": len(high_corr_pairs),
            "pairs": high_corr_pairs,
            "max_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
            "mean_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        }
    
    def _analyze_feature_variance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature variance."""
        variances = data.var()
        low_variance_features = variances[variances < self.config.variance_threshold]
        
        return {
            "low_variance_features": len(low_variance_features),
            "features": low_variance_features.index.tolist(),
            "min_variance": variances.min(),
            "mean_variance": variances.mean(),
            "variance_distribution": {
                "q25": variances.quantile(0.25),
                "q50": variances.quantile(0.50),
                "q75": variances.quantile(0.75)
            }
        }
    
    def _get_model_specific_recommendations(
        self, 
        model_type: ModelType, 
        n_features: int, 
        n_samples: int
    ) -> List[str]:
        """Get model-specific dimensionality reduction recommendations."""
        recommendations = []
        
        if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION]:
            if n_features > n_samples / 10:
                recommendations.extend(["feature_selection", "pca"])
        
        elif model_type == ModelType.NEURAL_NETWORK:
            if n_features > 100:
                recommendations.extend(["pca", "feature_selection"])
            elif n_features > 50:
                recommendations.append("feature_selection")
        
        elif model_type == ModelType.SVM:
            if n_features > 50:
                recommendations.extend(["pca", "feature_selection"])
        
        elif model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
            if n_features > 200:
                recommendations.append("feature_selection")
        
        # General recommendations based on dimensionality
        if n_features > 100:
            recommendations.append("pca")
        if n_features > 50:
            recommendations.append("feature_selection")
        
        return recommendations
    
    def _recommend_feature_selection_methods(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: ModelType
    ) -> List[str]:
        """Recommend feature selection methods."""
        recommendations = []
        
        target_type = "classification" if data[target_column].nunique() < 20 else "regression"
        
        if target_type == "classification":
            recommendations.extend(["mutual_info", "f_classif", "chi2"])
        else:
            recommendations.extend(["mutual_info", "f_regression"])
        
        if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION]:
            recommendations.extend(["lasso", "elastic_net"])
        
        recommendations.append("rfe")
        
        return recommendations
    
    def _prioritize_techniques(
        self, 
        techniques: List[str], 
        model_type: ModelType, 
        n_features: int, 
        n_samples: int
    ) -> List[str]:
        """Prioritize techniques based on effectiveness and computational cost."""
        priority_order = {
            "variance_threshold": 1,  # Fastest, always beneficial
            "correlation_removal": 2,  # Fast, often beneficial
            "feature_selection": 3,    # Moderate cost, high benefit
            "pca": 4,                 # Moderate cost, good for linear models
            "ica": 5,                 # Higher cost, specific use cases
            "manifold_learning": 6    # Highest cost, specific use cases
        }
        
        # Sort by priority
        prioritized = sorted(techniques, key=lambda x: priority_order.get(x, 999))
        
        # Limit number of techniques to avoid over-processing
        return prioritized[:3]
    
    def _remove_correlated_features(
        self, 
        data: pd.DataFrame, 
        numerical_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove highly correlated features."""
        if len(numerical_features) < 2:
            return data, {"features_removed": []}
        
        corr_matrix = data[numerical_features].corr().abs()
        
        # Find features to remove
        features_to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.config.correlation_threshold:
                    # Remove the feature with lower variance
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    var1, var2 = data[feat1].var(), data[feat2].var()
                    features_to_remove.add(feat1 if var1 < var2 else feat2)
        
        # Remove features
        features_to_remove = list(features_to_remove)
        data_reduced = data.drop(columns=features_to_remove)
        
        metadata = {
            "features_removed": features_to_remove,
            "n_features_removed": len(features_to_remove),
            "correlation_threshold": self.config.correlation_threshold
        }
        
        return data_reduced, metadata
    
    def _apply_variance_threshold(
        self, 
        data: pd.DataFrame, 
        numerical_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply variance threshold feature selection."""
        if len(numerical_features) == 0:
            return data, {"features_removed": []}
        
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        
        # Fit and transform
        selected_features = selector.fit_transform(data[numerical_features])
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_feature_names = [feat for feat, selected in zip(numerical_features, selected_mask) if selected]
        removed_features = [feat for feat, selected in zip(numerical_features, selected_mask) if not selected]
        
        # Create new dataframe
        data_reduced = data.copy()
        data_reduced = data_reduced.drop(columns=removed_features)
        
        # Store the selector
        self.fitted_reducers["variance_threshold"] = selector
        
        metadata = {
            "features_removed": removed_features,
            "features_kept": selected_feature_names,
            "n_features_removed": len(removed_features),
            "variance_threshold": self.config.variance_threshold
        }
        
        return data_reduced, metadata
    
    def _apply_pca(
        self, 
        data: pd.DataFrame, 
        numerical_features: List[str], 
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Principal Component Analysis."""
        if len(numerical_features) < 2:
            return data, {"components_created": 0}
        
        # Determine number of components
        n_components = min(
            self.config.max_components,
            len(numerical_features),
            len(data) - 1
        )
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[numerical_features])
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)
        
        # Find number of components for target variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_target = np.argmax(cumsum_var >= self.config.target_variance_ratio) + 1
        n_components_target = max(self.config.min_components, n_components_target)
        
        # Keep only the required components
        pca_features = pca_features[:, :n_components_target]
        
        # Create PCA feature names
        pca_columns = [f"pca_component_{i}" for i in range(n_components_target)]
        pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=data.index)
        
        # Create new dataframe
        data_reduced = data.drop(columns=numerical_features)
        data_reduced = pd.concat([data_reduced, pca_df], axis=1)
        
        # Store the transformers
        self.fitted_reducers["pca_scaler"] = scaler
        self.fitted_reducers["pca"] = pca
        
        metadata = {
            "components_created": n_components_target,
            "original_features": numerical_features,
            "pca_features": pca_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_[:n_components_target].tolist(),
            "total_variance_explained": cumsum_var[n_components_target-1],
            "n_features_removed": len(numerical_features),
            "n_features_added": n_components_target
        }
        
        return data_reduced, metadata
    
    def _apply_ica(
        self, 
        data: pd.DataFrame, 
        numerical_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Independent Component Analysis."""
        if len(numerical_features) < 2:
            return data, {"components_created": 0}
        
        # Determine number of components
        n_components = min(
            self.config.max_components,
            len(numerical_features)
        )
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[numerical_features])
        
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        try:
            ica_features = ica.fit_transform(scaled_features)
        except Exception as e:
            logger.warning(f"ICA failed: {str(e)}, falling back to PCA")
            return self._apply_pca(data, numerical_features)
        
        # Create ICA feature names
        ica_columns = [f"ica_component_{i}" for i in range(n_components)]
        ica_df = pd.DataFrame(ica_features, columns=ica_columns, index=data.index)
        
        # Create new dataframe
        data_reduced = data.drop(columns=numerical_features)
        data_reduced = pd.concat([data_reduced, ica_df], axis=1)
        
        # Store the transformers
        self.fitted_reducers["ica_scaler"] = scaler
        self.fitted_reducers["ica"] = ica
        
        metadata = {
            "components_created": n_components,
            "original_features": numerical_features,
            "ica_features": ica_columns,
            "n_features_removed": len(numerical_features),
            "n_features_added": n_components
        }
        
        return data_reduced, metadata
    
    def _apply_feature_selection(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        model_type: ModelType
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply intelligent feature selection."""
        if target_column not in data.columns:
            return data, {"features_selected": []}
        
        # Get features for selection
        feature_columns = [col for col in data.columns if col != target_column]
        numerical_features = data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_features) < 2:
            return data, {"features_selected": numerical_features}
        
        X = data[numerical_features]
        y = data[target_column]
        
        # Determine if classification or regression
        is_classification = y.nunique() < 20
        
        # Choose selection method based on model type and problem type
        if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION]:
            # Use L1 regularization for linear models
            if is_classification:
                from sklearn.linear_model import LogisticRegressionCV
                selector = SelectFromModel(
                    LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', random_state=42),
                    threshold='median'
                )
            else:
                selector = SelectFromModel(
                    LassoCV(cv=5, random_state=42),
                    threshold='median'
                )
        else:
            # Use tree-based feature importance
            if is_classification:
                selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    threshold='median'
                )
            else:
                selector = SelectFromModel(
                    RandomForestRegressor(n_estimators=100, random_state=42),
                    threshold='median'
                )
        
        # Fit and transform
        try:
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            selected_features = [feat for feat, selected in zip(numerical_features, selected_mask) if selected]
            
            # Ensure minimum number of features
            if len(selected_features) < self.config.min_components:
                # Fall back to SelectKBest
                k = max(self.config.min_components, len(numerical_features) // 2)
                if is_classification:
                    selector = SelectKBest(f_classif, k=k)
                else:
                    selector = SelectKBest(f_regression, k=k)
                
                X_selected = selector.fit_transform(X, y)
                selected_mask = selector.get_support()
                selected_features = [feat for feat, selected in zip(numerical_features, selected_mask) if selected]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {str(e)}, keeping all features")
            return data, {"features_selected": numerical_features}
        
        # Create new dataframe with selected features
        removed_features = [feat for feat in numerical_features if feat not in selected_features]
        data_reduced = data.drop(columns=removed_features)
        
        # Store the selector
        self.fitted_reducers["feature_selector"] = selector
        
        metadata = {
            "features_selected": selected_features,
            "features_removed": removed_features,
            "n_features_selected": len(selected_features),
            "n_features_removed": len(removed_features),
            "selection_method": type(selector).__name__
        }
        
        return data_reduced, metadata
    
    def _calculate_reduction_quality_metrics(
        self, 
        original_data: pd.DataFrame, 
        reduced_data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate quality metrics for dimensionality reduction."""
        metrics = {}
        
        # Feature count reduction
        original_features = len(original_data.columns)
        reduced_features = len(reduced_data.columns)
        metrics["feature_reduction_ratio"] = (original_features - reduced_features) / original_features
        metrics["compression_ratio"] = reduced_features / original_features
        
        # Memory usage change
        original_memory = original_data.memory_usage(deep=True).sum()
        reduced_memory = reduced_data.memory_usage(deep=True).sum()
        metrics["memory_reduction_ratio"] = (original_memory - reduced_memory) / original_memory
        
        # Information preservation (if possible)
        original_numerical = original_data.select_dtypes(include=[np.number])
        reduced_numerical = reduced_data.select_dtypes(include=[np.number])
        
        if len(original_numerical.columns) > 0 and len(reduced_numerical.columns) > 0:
            # Variance preservation
            original_total_var = original_numerical.var().sum()
            reduced_total_var = reduced_numerical.var().sum()
            metrics["variance_preservation_ratio"] = reduced_total_var / original_total_var if original_total_var > 0 else 1.0
        
        return metrics
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted reducers."""
        try:
            transformed_data = data.copy()
            
            # Apply transformations in order
            if "variance_threshold" in self.fitted_reducers:
                selector = self.fitted_reducers["variance_threshold"]
                numerical_features = transformed_data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numerical_features) > 0:
                    selected_features = selector.transform(transformed_data[numerical_features])
                    selected_mask = selector.get_support()
                    selected_feature_names = [feat for feat, selected in zip(numerical_features, selected_mask) if selected]
                    removed_features = [feat for feat, selected in zip(numerical_features, selected_mask) if not selected]
                    transformed_data = transformed_data.drop(columns=removed_features)
            
            if "pca" in self.fitted_reducers and "pca_scaler" in self.fitted_reducers:
                scaler = self.fitted_reducers["pca_scaler"]
                pca = self.fitted_reducers["pca"]
                
                numerical_features = transformed_data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numerical_features) > 0:
                    scaled_features = scaler.transform(transformed_data[numerical_features])
                    pca_features = pca.transform(scaled_features)
                    
                    # Create PCA feature columns
                    n_components = pca_features.shape[1]
                    pca_columns = [f"pca_component_{i}" for i in range(n_components)]
                    pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=transformed_data.index)
                    
                    # Replace original features with PCA features
                    transformed_data = transformed_data.drop(columns=numerical_features)
                    transformed_data = pd.concat([transformed_data, pca_df], axis=1)
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming new data: {str(e)}")
            raise FeatureEngineeringError(f"New data transformation failed: {str(e)}")
    
    def create_advanced_dimensionality_features(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create advanced dimensionality reduction features including manifold learning."""
        try:
            enhanced_data = data.copy()
            metadata = {"features_created": [], "techniques_applied": []}
            
            # Get numerical features
            numerical_features = enhanced_data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numerical_features:
                numerical_features.remove(target_column)
            
            if len(numerical_features) < 3:
                return enhanced_data, metadata
            
            X = enhanced_data[numerical_features].fillna(0)
            
            # Apply t-SNE for non-linear dimensionality reduction (if dataset is not too large)
            if len(X) <= 5000 and len(numerical_features) >= 5:
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
                    tsne_features = tsne.fit_transform(StandardScaler().fit_transform(X))
                    
                    enhanced_data['tsne_component_1'] = tsne_features[:, 0]
                    enhanced_data['tsne_component_2'] = tsne_features[:, 1]
                    
                    metadata["features_created"].extend(['tsne_component_1', 'tsne_component_2'])
                    metadata["techniques_applied"].append("t-SNE")
                    
                except Exception as e:
                    logger.warning(f"t-SNE failed: {str(e)}")
            
            # Apply Truncated SVD for sparse data
            if len(numerical_features) >= 10:
                try:
                    n_components = min(5, len(numerical_features) // 2)
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    svd_features = svd.fit_transform(X)
                    
                    for i in range(n_components):
                        col_name = f'svd_component_{i}'
                        enhanced_data[col_name] = svd_features[:, i]
                        metadata["features_created"].append(col_name)
                    
                    metadata["techniques_applied"].append("Truncated SVD")
                    
                except Exception as e:
                    logger.warning(f"Truncated SVD failed: {str(e)}")
            
            # Apply Factor Analysis
            if len(numerical_features) >= 5:
                try:
                    n_factors = min(3, len(numerical_features) // 3)
                    fa = FactorAnalysis(n_components=n_factors, random_state=42)
                    fa_features = fa.fit_transform(StandardScaler().fit_transform(X))
                    
                    for i in range(n_factors):
                        col_name = f'factor_{i}'
                        enhanced_data[col_name] = fa_features[:, i]
                        metadata["features_created"].append(col_name)
                    
                    metadata["techniques_applied"].append("Factor Analysis")
                    
                except Exception as e:
                    logger.warning(f"Factor Analysis failed: {str(e)}")
            
            # Create feature density and sparsity measures
            feature_density_col = 'feature_density'
            enhanced_data[feature_density_col] = (X != 0).sum(axis=1) / len(numerical_features)
            metadata["features_created"].append(feature_density_col)
            
            # Create feature magnitude measures
            feature_magnitude_col = 'feature_magnitude'
            enhanced_data[feature_magnitude_col] = np.sqrt((X ** 2).sum(axis=1))
            metadata["features_created"].append(feature_magnitude_col)
            
            # Create feature variance measures
            feature_variance_col = 'feature_variance'
            enhanced_data[feature_variance_col] = X.var(axis=1)
            metadata["features_created"].append(feature_variance_col)
            
            return enhanced_data, metadata
            
        except Exception as e:
            logger.error(f"Error creating advanced dimensionality features: {str(e)}")
            raise FeatureEngineeringError(f"Advanced dimensionality feature creation failed: {str(e)}")
    
    def recommend_optimal_dimensions(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        max_components: int = 50
    ) -> Dict[str, Any]:
        """Recommend optimal number of dimensions using various criteria."""
        try:
            recommendations = {}
            
            # Get numerical features
            numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numerical_features:
                numerical_features.remove(target_column)
            
            if len(numerical_features) < 2:
                return {"recommendation": "no_reduction_needed", "reason": "insufficient_features"}
            
            X = data[numerical_features].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            
            # PCA analysis for variance explanation
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Find components for different variance thresholds
            recommendations["variance_thresholds"] = {
                "90_percent": int(np.argmax(cumsum_var >= 0.90) + 1),
                "95_percent": int(np.argmax(cumsum_var >= 0.95) + 1),
                "99_percent": int(np.argmax(cumsum_var >= 0.99) + 1)
            }
            
            # Elbow method for optimal components
            elbow_point = self._find_elbow_point(pca_full.explained_variance_ratio_)
            recommendations["elbow_method"] = elbow_point
            
            # Kaiser criterion (eigenvalues > 1)
            kaiser_components = sum(pca_full.explained_variance_ > 1)
            recommendations["kaiser_criterion"] = kaiser_components
            
            # Broken stick model
            broken_stick_components = self._broken_stick_criterion(pca_full.explained_variance_ratio_)
            recommendations["broken_stick"] = broken_stick_components
            
            # Final recommendation based on multiple criteria
            candidates = [
                recommendations["variance_thresholds"]["95_percent"],
                recommendations["elbow_method"],
                recommendations["kaiser_criterion"],
                recommendations["broken_stick"]
            ]
            
            # Remove invalid candidates
            candidates = [c for c in candidates if c > 0 and c <= max_components]
            
            if candidates:
                final_recommendation = int(np.median(candidates))
                recommendations["final_recommendation"] = min(final_recommendation, max_components)
            else:
                recommendations["final_recommendation"] = min(10, len(numerical_features) // 2)
            
            recommendations["original_dimensions"] = len(numerical_features)
            recommendations["reduction_ratio"] = 1 - (recommendations["final_recommendation"] / len(numerical_features))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending optimal dimensions: {str(e)}")
            return {"recommendation": "error", "reason": str(e)}
    
    def _find_elbow_point(self, explained_variance_ratio: np.ndarray) -> int:
        """Find elbow point in explained variance curve."""
        try:
            # Calculate second derivative to find elbow
            if len(explained_variance_ratio) < 3:
                return 1
            
            # Normalize the curve
            x = np.arange(len(explained_variance_ratio))
            y = explained_variance_ratio
            
            # Calculate distances from line connecting first and last points
            distances = []
            for i in range(1, len(y) - 1):
                # Distance from point to line
                x1, y1 = 0, y[0]
                x2, y2 = len(y) - 1, y[-1]
                xi, yi = i, y[i]
                
                distance = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                distances.append(distance)
            
            if distances:
                elbow_idx = np.argmax(distances) + 1  # +1 because we started from index 1
                return min(elbow_idx + 1, len(explained_variance_ratio))  # +1 for 1-based indexing
            else:
                return 1
                
        except Exception:
            return 1
    
    def _broken_stick_criterion(self, explained_variance_ratio: np.ndarray) -> int:
        """Apply broken stick model to determine significant components."""
        try:
            n = len(explained_variance_ratio)
            broken_stick = []
            
            for i in range(n):
                stick_value = sum(1/j for j in range(i+1, n+1)) / n
                broken_stick.append(stick_value)
            
            # Find components where explained variance > broken stick expectation
            significant_components = sum(1 for i in range(n) if explained_variance_ratio[i] > broken_stick[i])
            
            return max(1, significant_components)
            
        except Exception:
            return 1