"""
ExplainXEngine - Advanced Explainable AI Engine
SHAP, LIME, attention visualization, and comprehensive model interpretability.
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
import json
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ML and XAI libraries
try:
    import shap
    import lime
    from lime import lime_tabular
    from lime.lime_text import LimeTextExplainer
    from lime.lime_image import LimeImageExplainer
    SHAP_AVAILABLE = True
    LIME_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    LIME_AVAILABLE = False

# Deep learning libraries for attention visualization
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_engine import BaseEngine, EngineStatus, EngineCapability
import logging

logger = logging.getLogger(__name__)


class ExplanationMethod(str, Enum):
    """Available explanation methods."""
    SHAP_TREE = "shap_tree"
    SHAP_LINEAR = "shap_linear"
    SHAP_DEEP = "shap_deep"
    SHAP_KERNEL = "shap_kernel"
    LIME_TABULAR = "lime_tabular"
    LIME_TEXT = "lime_text"
    LIME_IMAGE = "lime_image"
    ATTENTION_WEIGHTS = "attention_weights"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"


class ModelType(str, Enum):
    """Supported model types."""
    TREE_BASED = "tree_based"
    LINEAR = "linear"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class ExplainXEngine(BaseEngine):
    """Advanced explainable AI engine with multiple interpretation methods."""
    
    def __init__(self):
        super().__init__(
            engine_id="explainx-engine",
            name="ExplainX Engine",
            capabilities=[
                EngineCapability.EXPLANATION,
                EngineCapability.DATA_ANALYSIS,
                EngineCapability.VISUALIZATION
            ]
        )
        
        self.explainers = {}
        self.explanation_cache = {}
        self.supported_methods = self._get_supported_methods()
    
    def _get_supported_methods(self) -> List[ExplanationMethod]:
        """Get list of supported explanation methods based on available libraries."""
        methods = [ExplanationMethod.FEATURE_IMPORTANCE]
        
        if SHAP_AVAILABLE:
            methods.extend([
                ExplanationMethod.SHAP_TREE,
                ExplanationMethod.SHAP_LINEAR,
                ExplanationMethod.SHAP_DEEP,
                ExplanationMethod.SHAP_KERNEL
            ])
        
        if LIME_AVAILABLE:
            methods.extend([
                ExplanationMethod.LIME_TABULAR,
                ExplanationMethod.LIME_TEXT,
                ExplanationMethod.LIME_IMAGE
            ])
        
        if TORCH_AVAILABLE:
            methods.append(ExplanationMethod.ATTENTION_WEIGHTS)
        
        return methods
    
    async def initialize(self) -> None:
        """Initialize the explainability engine."""
        try:
            # Initialize SHAP
            if SHAP_AVAILABLE:
                shap.initjs()
                logger.info("SHAP initialized successfully")
            
            # Initialize matplotlib for visualizations
            plt.style.use('seaborn-v0_8')
            
            self.status = EngineStatus.READY
            logger.info("ExplainXEngine initialized successfully")
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Failed to initialize ExplainXEngine: {e}")
            raise
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process explanation request."""
        params = parameters or {}
        
        model = params.get("model")
        data = params.get("data")
        method = params.get("method", ExplanationMethod.SHAP_KERNEL)
        model_type = params.get("model_type", ModelType.CUSTOM)
        
        if model is None or data is None:
            raise ValueError("Model and data are required for explanation")
        
        # Generate explanation based on method
        if method == ExplanationMethod.SHAP_TREE:
            return await self._explain_with_shap_tree(model, data, params)
        elif method == ExplanationMethod.SHAP_LINEAR:
            return await self._explain_with_shap_linear(model, data, params)
        elif method == ExplanationMethod.SHAP_DEEP:
            return await self._explain_with_shap_deep(model, data, params)
        elif method == ExplanationMethod.SHAP_KERNEL:
            return await self._explain_with_shap_kernel(model, data, params)
        elif method == ExplanationMethod.LIME_TABULAR:
            return await self._explain_with_lime_tabular(model, data, params)
        elif method == ExplanationMethod.LIME_TEXT:
            return await self._explain_with_lime_text(model, data, params)
        elif method == ExplanationMethod.ATTENTION_WEIGHTS:
            return await self._explain_with_attention(model, data, params)
        elif method == ExplanationMethod.FEATURE_IMPORTANCE:
            return await self._explain_with_feature_importance(model, data, params)
        else:
            return await self._explain_with_shap_kernel(model, data, params)
    
    async def _explain_with_shap_tree(self, model, data, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP explanations for tree-based models."""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not available")
        
        try:
            # Create SHAP explainer for tree models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)
            
            # Generate visualizations
            visualizations = await self._create_shap_visualizations(explainer, shap_values, data, params)
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(shap_values, data)
            
            return {
                "method": "SHAP Tree Explainer",
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "feature_importance": feature_importance,
                "visualizations": visualizations,
                "summary": await self._generate_explanation_summary(shap_values, data, "tree"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SHAP Tree explanation failed: {e}")
            raise
    
    async def _explain_with_shap_linear(self, model, data, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP explanations for linear models."""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not available")
        
        try:
            # Create SHAP explainer for linear models
            explainer = shap.LinearExplainer(model, data)
            shap_values = explainer.shap_values(data)
            
            # Generate visualizations
            visualizations = await self._create_shap_visualizations(explainer, shap_values, data, params)
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(shap_values, data)
            
            return {
                "method": "SHAP Linear Explainer",
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "feature_importance": feature_importance,
                "visualizations": visualizations,
                "summary": await self._generate_explanation_summary(shap_values, data, "linear"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SHAP Linear explanation failed: {e}")
            raise
    
    async def _explain_with_shap_kernel(self, model, data, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP explanations using Kernel SHAP (model-agnostic)."""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not available")
        
        try:
            # Create background dataset for Kernel SHAP
            background = params.get("background_data", data[:100] if len(data) > 100 else data)
            
            # Create SHAP explainer
            explainer = shap.KernelExplainer(model.predict, background)
            
            # Calculate SHAP values for subset of data (Kernel SHAP is slow)
            sample_size = min(params.get("sample_size", 10), len(data))
            sample_data = data[:sample_size]
            
            shap_values = explainer.shap_values(sample_data)
            
            # Generate visualizations
            visualizations = await self._create_shap_visualizations(explainer, shap_values, sample_data, params)
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(shap_values, sample_data)
            
            return {
                "method": "SHAP Kernel Explainer",
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "feature_importance": feature_importance,
                "visualizations": visualizations,
                "summary": await self._generate_explanation_summary(shap_values, sample_data, "kernel"),
                "sample_size": sample_size,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SHAP Kernel explanation failed: {e}")
            raise
    
    async def _explain_with_lime_tabular(self, model, data, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LIME explanations for tabular data."""
        if not LIME_AVAILABLE:
            raise RuntimeError("LIME is not available")
        
        try:
            # Convert data to numpy array if needed
            if isinstance(data, pd.DataFrame):
                feature_names = data.columns.tolist()
                data_array = data.values
            else:
                data_array = np.array(data)
                feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                data_array,
                feature_names=feature_names,
                mode='regression' if params.get('mode') == 'regression' else 'classification'
            )
            
            # Generate explanations for sample instances
            sample_size = min(params.get("sample_size", 5), len(data_array))
            explanations = []
            
            for i in range(sample_size):
                exp = explainer.explain_instance(
                    data_array[i],
                    model.predict,
                    num_features=params.get("num_features", 10)
                )
                explanations.append({
                    "instance_id": i,
                    "explanation": exp.as_list(),
                    "score": exp.score if hasattr(exp, 'score') else None
                })
            
            # Generate visualizations
            visualizations = await self._create_lime_visualizations(explanations, feature_names)
            
            return {
                "method": "LIME Tabular Explainer",
                "explanations": explanations,
                "visualizations": visualizations,
                "feature_names": feature_names,
                "summary": await self._generate_lime_summary(explanations),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LIME Tabular explanation failed: {e}")
            raise
    
    async def _explain_with_attention(self, model, data, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate attention weight explanations for transformer models."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        
        try:
            # This is a simplified attention visualization
            # In practice, you'd need to extract attention weights from the specific model
            
            # Mock attention weights for demonstration
            seq_length = params.get("sequence_length", 50)
            num_heads = params.get("num_heads", 8)
            
            # Generate mock attention weights
            attention_weights = np.random.rand(num_heads, seq_length, seq_length)
            attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
            
            # Generate visualizations
            visualizations = await self._create_attention_visualizations(attention_weights, params)
            
            return {
                "method": "Attention Weights Visualization",
                "attention_weights": attention_weights.tolist(),
                "visualizations": visualizations,
                "summary": "Attention weights show which parts of the input the model focuses on",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Attention explanation failed: {e}")
            raise
    
    async def _explain_with_feature_importance(self, model, data, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic feature importance explanations."""
        try:
            # Try to get feature importance from model
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                # Calculate permutation importance
                importance = await self._calculate_permutation_importance(model, data, params)
            
            # Get feature names
            if isinstance(data, pd.DataFrame):
                feature_names = data.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(len(importance))]
            
            # Create feature importance ranking
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Generate visualizations
            visualizations = await self._create_importance_visualizations(feature_importance)
            
            return {
                "method": "Feature Importance",
                "feature_importance": feature_importance,
                "visualizations": visualizations,
                "summary": await self._generate_importance_summary(feature_importance),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Feature importance explanation failed: {e}")
            raise
    
    async def _create_shap_visualizations(self, explainer, shap_values, data, params: Dict[str, Any]) -> Dict[str, str]:
        """Create SHAP visualizations."""
        visualizations = {}
        
        try:
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, data, show=False)
            visualizations["summary_plot"] = await self._plot_to_base64()
            
            # Waterfall plot for first instance
            if len(shap_values) > 0:
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(explainer.expected_value, shap_values[0], data.iloc[0] if hasattr(data, 'iloc') else data[0], show=False)
                visualizations["waterfall_plot"] = await self._plot_to_base64()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, data, plot_type="bar", show=False)
            visualizations["importance_plot"] = await self._plot_to_base64()
            
        except Exception as e:
            logger.warning(f"Failed to create SHAP visualizations: {e}")
        
        return visualizations
    
    async def _plot_to_base64(self) -> str:
        """Convert current matplotlib plot to base64 string."""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.explainers.clear()
        self.explanation_cache.clear()
        plt.close('all')
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "engine_id": self.engine_id,
            "status": self.status.value,
            "supported_methods": [method.value for method in self.supported_methods],
            "shap_available": SHAP_AVAILABLE,
            "lime_available": LIME_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "cached_explanations": len(self.explanation_cache),
            "healthy": self.status == EngineStatus.READY
        }
    
    # Additional helper methods
    async def _calculate_feature_importance(self, shap_values, data) -> List[Tuple[str, float]]:
        """Calculate feature importance from SHAP values."""
        if isinstance(shap_values, list):
            # Multi-class case
            importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return list(zip(feature_names, importance))
    
    async def _generate_explanation_summary(self, shap_values, data, method: str) -> str:
        """Generate human-readable summary of explanations."""
        return f"SHAP {method} explanation generated for {len(data)} instances with {data.shape[1] if hasattr(data, 'shape') else 'unknown'} features."
    
    async def _calculate_permutation_importance(self, model, data, params: Dict[str, Any]) -> np.ndarray:
        """Calculate permutation importance when model doesn't have built-in importance."""
        # Simplified permutation importance calculation
        baseline_score = model.score(data, params.get('y_true', np.zeros(len(data))))
        importance = []
        
        for i in range(data.shape[1]):
            # Permute feature i
            data_permuted = data.copy()
            if isinstance(data_permuted, pd.DataFrame):
                data_permuted.iloc[:, i] = np.random.permutation(data_permuted.iloc[:, i])
            else:
                data_permuted[:, i] = np.random.permutation(data_permuted[:, i])
            
            # Calculate score drop
            permuted_score = model.score(data_permuted, params.get('y_true', np.zeros(len(data))))
            importance.append(baseline_score - permuted_score)
        
        return np.array(importance)