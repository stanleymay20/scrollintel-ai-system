"""
ExplainXEngine - Explainable AI Engine for ScrollIntel
Provides comprehensive model interpretability and explanation capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Any, Optional, Union, Tuple
import joblib
import json
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Transformer attention visualization imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base_engine import BaseEngine, EngineCapability

class ExplainXEngine(BaseEngine):
    """
    Advanced explainable AI engine providing comprehensive model interpretability
    """
    
    def __init__(self):
        super().__init__(
            engine_id="explainx_engine",
            name="ExplainXEngine",
            capabilities=[
                EngineCapability.EXPLANATION,
                EngineCapability.BIAS_DETECTION
            ]
        )
        self.version = "2.0.0"
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.model = None
        self.feature_names = None
        self.training_data = None
        
        # Transformer model components
        self.transformer_model = None
        self.tokenizer = None
        self.attention_weights = None
        
        # Enhanced capabilities
        self.supported_explanation_types = [
            "shap_waterfall", "shap_summary", "shap_force", "shap_beeswarm",
            "lime_local", "attention_visualization", "counterfactual",
            "feature_importance", "bias_detection", "model_comparison"
        ]
        
    async def initialize(self) -> None:
        """Initialize the ExplainX engine with configuration"""
        try:
            # Initialize explainers
            self.shap_explainer = None
            self.lime_explainer = None
            self.model = None
            self.feature_names = None
            self.training_data = None
            self.logger.info("ExplainXEngine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ExplainXEngine: {str(e)}")
            raise
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process input data and return results"""
        # This is a generic process method - specific methods are called directly
        return {"status": "success", "message": "Use specific explanation methods"}
    
    async def cleanup(self) -> None:
        """Clean up resources used by the engine"""
        self.shap_explainer = None
        self.lime_explainer = None
        self.model = None
        self.training_data = None
        self.transformer_model = None
        self.tokenizer = None
        self.attention_weights = None
    
    async def setup_explainers(self, model: Any, training_data: pd.DataFrame, 
                             feature_names: List[str]) -> Dict[str, Any]:
        """Setup SHAP and LIME explainers for the given model"""
        try:
            self.model = model
            self.training_data = training_data
            self.feature_names = feature_names
            
            # Setup SHAP explainer
            if hasattr(model, 'predict_proba'):
                # For classification models
                self.shap_explainer = shap.Explainer(model.predict_proba, training_data)
            else:
                # For regression models
                self.shap_explainer = shap.Explainer(model.predict, training_data)
            
            # Setup LIME explainer
            mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
            class_names = ['class_0', 'class_1'] if mode == 'classification' else None
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data.values,
                feature_names=feature_names,
                class_names=class_names,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
            
            return {
                "status": "success",
                "message": "Explainers setup successfully",
                "shap_available": self.shap_explainer is not None,
                "lime_available": self.lime_explainer is not None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to setup explainers: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def generate_shap_explanations(self, data: pd.DataFrame, 
                                       explanation_type: str = "waterfall") -> Dict[str, Any]:
        """Generate SHAP explanations for the given data"""
        try:
            if self.shap_explainer is None:
                return {"status": "error", "message": "SHAP explainer not initialized"}
            
            # Calculate SHAP values
            shap_values = self.shap_explainer(data)
            
            explanations = {
                "status": "success",
                "shap_values": shap_values.values.tolist(),
                "base_values": shap_values.base_values.tolist(),
                "feature_names": self.feature_names,
                "data": data.values.tolist()
            }
            
            # Generate visualizations based on type
            if explanation_type == "waterfall":
                explanations["visualization"] = self._create_waterfall_plot(shap_values[0])
            elif explanation_type == "summary":
                explanations["visualization"] = self._create_summary_plot(shap_values)
            elif explanation_type == "force":
                explanations["visualization"] = self._create_force_plot(shap_values[0])
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP explanations: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def generate_lime_explanations(self, instance: pd.Series, 
                                       num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanations for a single instance"""
        try:
            if self.lime_explainer is None:
                return {"status": "error", "message": "LIME explainer not initialized"}
            
            if self.model is None:
                return {"status": "error", "message": "Model not loaded"}
            
            # Generate LIME explanation
            if hasattr(self.model, 'predict_proba'):
                predict_fn = lambda x: self.model.predict_proba(x)
            else:
                predict_fn = lambda x: self.model.predict(x).reshape(-1, 1)
            
            # Test the predict function first
            test_pred = predict_fn(instance.values.reshape(1, -1))
            if len(test_pred.shape) == 1 or test_pred.shape[1] == 1:
                # For regression or binary classification, LIME expects 2D output
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = lambda x: self.model.predict_proba(x)
                else:
                    predict_fn = lambda x: np.column_stack([1 - self.model.predict(x), self.model.predict(x)])
            
            explanation = self.lime_explainer.explain_instance(
                instance.values,
                predict_fn,
                num_features=num_features,
                num_samples=100  # Reduced for faster execution
            )
            
            # Extract explanation data
            lime_data = {
                "status": "success",
                "instance": instance.tolist(),
                "explanations": explanation.as_list(),
                "score": getattr(explanation, 'score', 0.0),
                "intercept": explanation.intercept.get(0, None) if hasattr(explanation, 'intercept') and explanation.intercept is not None else None
            }
            
            # Create visualization
            lime_data["visualization"] = self._create_lime_plot(explanation)
            
            return lime_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate LIME explanations: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": str(e)}
    
    async def analyze_feature_importance(self, method: str = "shap") -> Dict[str, Any]:
        """Analyze global feature importance using specified method"""
        try:
            if method == "shap" and self.shap_explainer is not None:
                # Calculate SHAP values for training data sample
                sample_data = self.training_data.sample(min(100, len(self.training_data)))
                shap_values = self.shap_explainer(sample_data)
                
                # Calculate mean absolute SHAP values
                if hasattr(shap_values, 'values'):
                    shap_array = shap_values.values
                else:
                    shap_array = shap_values
                
                # Handle different SHAP value shapes
                if len(shap_array.shape) == 3:  # Multi-class case
                    importance_scores = np.abs(shap_array).mean(axis=(0, 2))
                else:  # Binary or regression case
                    importance_scores = np.abs(shap_array).mean(axis=0)
                
                importance_data = {
                    "method": "shap",
                    "features": self.feature_names,
                    "importance_scores": importance_scores.tolist(),
                    "ranking": sorted(zip(self.feature_names, importance_scores), 
                                    key=lambda x: x[1], reverse=True)
                }
                
            elif hasattr(self.model, 'feature_importances_'):
                # Use model's built-in feature importance
                importance_scores = self.model.feature_importances_
                importance_data = {
                    "method": "model_builtin",
                    "features": self.feature_names,
                    "importance_scores": importance_scores.tolist(),
                    "ranking": sorted(zip(self.feature_names, importance_scores), 
                                    key=lambda x: x[1], reverse=True)
                }
            else:
                return {"status": "error", "message": "No feature importance method available"}
            
            # Create visualization
            importance_data["visualization"] = self._create_importance_plot(
                importance_data["features"], 
                importance_data["importance_scores"]
            )
            
            return {"status": "success", "data": importance_data}
            
        except Exception as e:
            self.logger.error(f"Failed to analyze feature importance: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def generate_counterfactual_explanations(self, instance: pd.Series, 
                                                 target_class: Optional[int] = None) -> Dict[str, Any]:
        """Generate counterfactual explanations for an instance"""
        try:
            # Simple counterfactual generation by perturbing features
            original_prediction = self.model.predict([instance.values])[0]
            
            counterfactuals = []
            
            for i, feature in enumerate(self.feature_names):
                # Create a copy of the instance
                modified_instance = instance.copy()
                
                # Perturb the feature (simple approach)
                if instance.iloc[i] == 0:
                    modified_instance.iloc[i] = 1
                else:
                    modified_instance.iloc[i] = 0
                
                # Get new prediction
                new_prediction = self.model.predict([modified_instance.values])[0]
                
                if new_prediction != original_prediction:
                    counterfactuals.append({
                        "feature": feature,
                        "original_value": instance.iloc[i],
                        "modified_value": modified_instance.iloc[i],
                        "original_prediction": original_prediction,
                        "new_prediction": new_prediction,
                        "change": abs(new_prediction - original_prediction)
                    })
            
            # Sort by impact
            counterfactuals.sort(key=lambda x: x["change"], reverse=True)
            
            return {
                "status": "success",
                "original_instance": instance.tolist(),
                "original_prediction": original_prediction,
                "counterfactuals": counterfactuals[:10]  # Top 10
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate counterfactual explanations: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def detect_bias(self, protected_features: List[str], 
                         predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Detect potential bias in model predictions"""
        try:
            bias_metrics = {}
            
            for feature in protected_features:
                if feature not in self.feature_names:
                    continue
                
                feature_idx = self.feature_names.index(feature)
                feature_values = self.training_data.iloc[:, feature_idx]
                
                # Ensure we have the same number of samples
                min_samples = min(len(feature_values), len(predictions))
                feature_values = feature_values.iloc[:min_samples]
                predictions_subset = predictions[:min_samples]
                
                # Calculate demographic parity
                unique_values = feature_values.unique()
                group_rates = {}
                
                for value in unique_values:
                    mask = feature_values == value
                    group_predictions = predictions_subset[mask]
                    positive_rate = np.mean(group_predictions > 0.5) if len(group_predictions) > 0 else 0
                    group_rates[str(value)] = positive_rate
                
                # Calculate bias metrics
                rates = list(group_rates.values())
                demographic_parity = max(rates) - min(rates) if rates else 0
                
                bias_metrics[feature] = {
                    "group_rates": group_rates,
                    "demographic_parity_difference": demographic_parity,
                    "bias_detected": demographic_parity > 0.1  # Threshold
                }
            
            return {
                "status": "success",
                "bias_metrics": bias_metrics,
                "overall_bias_detected": any(m["bias_detected"] for m in bias_metrics.values())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect bias: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def setup_transformer_model(self, model_name: str, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Setup transformer model for attention visualization"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                return {
                    "status": "error", 
                    "message": "Transformers library not available. Install with: pip install transformers torch"
                }
            
            if model_path:
                # Load from local path
                self.transformer_model = AutoModel.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # Load from HuggingFace hub
                self.transformer_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set model to evaluation mode
            self.transformer_model.eval()
            
            return {
                "status": "success",
                "message": "Transformer model setup successfully",
                "model_name": model_name,
                "num_layers": self.transformer_model.config.num_hidden_layers,
                "num_attention_heads": self.transformer_model.config.num_attention_heads
            }
            
        except Exception as e:
            self.logger.error(f"Failed to setup transformer model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def generate_attention_visualization(self, text: str, layer: int = -1, 
                                             head: int = -1) -> Dict[str, Any]:
        """Generate attention visualization for transformer models"""
        try:
            if not TRANSFORMERS_AVAILABLE or self.transformer_model is None:
                return {
                    "status": "error", 
                    "message": "Transformer model not available or not setup"
                }
            
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Get model outputs with attention weights
            with torch.no_grad():
                outputs = self.transformer_model(**inputs, output_attentions=True)
                attention_weights = outputs.attentions
            
            # Store attention weights for later use
            self.attention_weights = attention_weights
            
            # Select layer and head
            if layer == -1:
                layer = len(attention_weights) - 1  # Last layer
            if head == -1:
                head = 0  # First head
            
            # Extract attention matrix for specified layer and head
            attention_matrix = attention_weights[layer][0, head].numpy()
            
            # Create attention visualization
            attention_viz = self._create_attention_heatmap(
                attention_matrix, tokens, layer, head
            )
            
            # Calculate attention statistics
            attention_stats = self._calculate_attention_statistics(attention_weights)
            
            return {
                "status": "success",
                "text": text,
                "tokens": tokens,
                "layer": layer,
                "head": head,
                "attention_matrix": attention_matrix.tolist(),
                "visualization": attention_viz,
                "statistics": attention_stats,
                "total_layers": len(attention_weights),
                "total_heads": attention_weights[0].shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate attention visualization: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def generate_multi_head_attention_analysis(self, text: str, 
                                                   layer: int = -1) -> Dict[str, Any]:
        """Analyze attention patterns across all heads in a layer"""
        try:
            if not TRANSFORMERS_AVAILABLE or self.transformer_model is None:
                return {
                    "status": "error", 
                    "message": "Transformer model not available or not setup"
                }
            
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Get model outputs with attention weights
            with torch.no_grad():
                outputs = self.transformer_model(**inputs, output_attentions=True)
                attention_weights = outputs.attentions
            
            # Select layer
            if layer == -1:
                layer = len(attention_weights) - 1  # Last layer
            
            layer_attention = attention_weights[layer][0]  # Shape: [num_heads, seq_len, seq_len]
            num_heads = layer_attention.shape[0]
            
            # Analyze each head
            head_analyses = []
            for head_idx in range(num_heads):
                head_attention = layer_attention[head_idx].numpy()
                
                # Calculate head-specific metrics
                head_analysis = {
                    "head_index": head_idx,
                    "attention_entropy": self._calculate_attention_entropy(head_attention),
                    "attention_focus": self._calculate_attention_focus(head_attention),
                    "dominant_patterns": self._identify_attention_patterns(head_attention, tokens),
                    "visualization": self._create_attention_heatmap(
                        head_attention, tokens, layer, head_idx
                    )
                }
                head_analyses.append(head_analysis)
            
            # Create multi-head comparison visualization
            multi_head_viz = self._create_multi_head_comparison(layer_attention.numpy(), tokens)
            
            return {
                "status": "success",
                "text": text,
                "tokens": tokens,
                "layer": layer,
                "num_heads": num_heads,
                "head_analyses": head_analyses,
                "multi_head_visualization": multi_head_viz,
                "layer_summary": self._summarize_layer_attention(layer_attention.numpy())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate multi-head attention analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def generate_layer_wise_attention_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze attention patterns across all layers"""
        try:
            if not TRANSFORMERS_AVAILABLE or self.transformer_model is None:
                return {
                    "status": "error", 
                    "message": "Transformer model not available or not setup"
                }
            
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Get model outputs with attention weights
            with torch.no_grad():
                outputs = self.transformer_model(**inputs, output_attentions=True)
                attention_weights = outputs.attentions
            
            # Analyze each layer
            layer_analyses = []
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_attention_np = layer_attention[0].numpy()  # Remove batch dimension
                
                layer_analysis = {
                    "layer_index": layer_idx,
                    "average_attention_entropy": np.mean([
                        self._calculate_attention_entropy(layer_attention_np[head])
                        for head in range(layer_attention_np.shape[0])
                    ]),
                    "attention_diversity": self._calculate_attention_diversity(layer_attention_np),
                    "dominant_attention_heads": self._identify_dominant_heads(layer_attention_np),
                    "layer_visualization": self._create_layer_attention_summary(
                        layer_attention_np, tokens, layer_idx
                    )
                }
                layer_analyses.append(layer_analysis)
            
            # Create layer progression visualization
            layer_progression_viz = self._create_layer_progression_visualization(
                attention_weights, tokens
            )
            
            return {
                "status": "success",
                "text": text,
                "tokens": tokens,
                "num_layers": len(attention_weights),
                "layer_analyses": layer_analyses,
                "layer_progression_visualization": layer_progression_viz,
                "global_attention_summary": self._create_global_attention_summary(attention_weights)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate layer-wise attention analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _create_waterfall_plot(self, shap_values) -> Dict[str, Any]:
        """Create waterfall plot for SHAP values"""
        try:
            # Extract values and features
            values = shap_values.values
            base_value = shap_values.base_values
            
            # Handle different shapes
            if len(values.shape) > 1:
                values = values[0]  # Take first instance
            if hasattr(base_value, '__len__') and len(base_value) > 0:
                base_value = base_value[0] if isinstance(base_value, (list, np.ndarray)) else base_value
            
            # Create plotly waterfall chart
            fig = go.Figure(go.Waterfall(
                name="SHAP Values",
                orientation="v",
                measure=["relative"] * len(values) + ["total"],
                x=self.feature_names + ["Prediction"],
                textposition="outside",
                text=[f"{float(v):.3f}" for v in values] + [f"{float(base_value + sum(values)):.3f}"],
                y=[float(v) for v in values] + [float(base_value + sum(values))],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title="SHAP Waterfall Plot",
                showlegend=False,
                height=500
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create waterfall plot: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _create_summary_plot(self, shap_values) -> Dict[str, Any]:
        """Create summary plot for SHAP values"""
        try:
            # Calculate mean absolute SHAP values
            shap_array = shap_values.values
            
            # Handle different shapes
            if len(shap_array.shape) > 2:
                shap_array = shap_array[0]  # Take first instance
            elif len(shap_array.shape) == 1:
                shap_array = shap_array.reshape(1, -1)
            
            mean_shap = np.abs(shap_array).mean(axis=0)
            
            # Ensure we have the right number of features
            if len(mean_shap) != len(self.feature_names):
                if len(mean_shap) > len(self.feature_names):
                    mean_shap = mean_shap[:len(self.feature_names)]
                else:
                    # Pad with zeros if needed
                    mean_shap = np.pad(mean_shap, (0, len(self.feature_names) - len(mean_shap)))
            
            # Create bar plot
            fig = px.bar(
                x=mean_shap.tolist(),
                y=self.feature_names,
                orientation='h',
                title="SHAP Feature Importance Summary"
            )
            
            fig.update_layout(
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                height=max(400, len(self.feature_names) * 25)
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create summary plot: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _create_force_plot(self, shap_values) -> Dict[str, Any]:
        """Create force plot for SHAP values"""
        try:
            # Create horizontal bar chart showing positive and negative contributions
            values = shap_values.values
            
            # Handle different shapes
            if len(values.shape) > 1:
                values = values[0]  # Take first instance
            
            # Convert to list for JSON serialization
            values_list = [float(v) for v in values]
            colors = ['red' if v < 0 else 'blue' for v in values_list]
            
            fig = go.Figure(go.Bar(
                x=values_list,
                y=self.feature_names,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.3f}" for v in values_list],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="SHAP Force Plot",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                height=max(400, len(self.feature_names) * 25)
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create force plot: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _create_lime_plot(self, explanation) -> Dict[str, Any]:
        """Create visualization for LIME explanation"""
        try:
            # Extract explanation data
            exp_list = explanation.as_list()
            features = [item[0] for item in exp_list]
            values = [item[1] for item in exp_list]
            
            # Create bar plot
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            fig = go.Figure(go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="LIME Explanation",
                xaxis_title="Feature Contribution",
                yaxis_title="Features",
                height=max(400, len(features) * 30)
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create LIME plot: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _create_importance_plot(self, features: List[str], importance_scores: List[float]) -> Dict[str, Any]:
        """Create feature importance visualization"""
        try:
            # Sort by importance
            sorted_data = sorted(zip(features, importance_scores), key=lambda x: x[1], reverse=True)
            sorted_features, sorted_scores = zip(*sorted_data)
            
            fig = px.bar(
                x=list(sorted_scores),
                y=list(sorted_features),
                orientation='h',
                title="Feature Importance Ranking"
            )
            
            fig.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(features) * 25)
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create importance plot: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _create_attention_heatmap(self, attention_matrix: np.ndarray, tokens: List[str], 
                                layer: int, head: int) -> Dict[str, Any]:
        """Create attention heatmap visualization"""
        try:
            # Create heatmap using plotly
            fig = go.Figure(data=go.Heatmap(
                z=attention_matrix,
                x=tokens,
                y=tokens,
                colorscale='Blues',
                showscale=True,
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f"Attention Heatmap - Layer {layer}, Head {head}",
                xaxis_title="Target Tokens",
                yaxis_title="Source Tokens",
                height=max(500, len(tokens) * 20),
                width=max(500, len(tokens) * 20)
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create attention heatmap: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Calculate attention entropy for measuring attention distribution"""
        try:
            # Calculate entropy for each row (source token)
            entropies = []
            for row in attention_matrix:
                # Add small epsilon to avoid log(0)
                row_normalized = row + 1e-10
                row_normalized = row_normalized / np.sum(row_normalized)
                entropy = -np.sum(row_normalized * np.log(row_normalized))
                entropies.append(entropy)
            
            return float(np.mean(entropies))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attention entropy: {str(e)}")
            return 0.0
    
    def _calculate_attention_focus(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate attention focus metrics"""
        try:
            # Calculate max attention per row (how focused each token is)
            max_attentions = np.max(attention_matrix, axis=1)
            
            # Calculate attention spread (standard deviation)
            attention_spreads = np.std(attention_matrix, axis=1)
            
            return {
                "average_max_attention": float(np.mean(max_attentions)),
                "average_attention_spread": float(np.mean(attention_spreads)),
                "attention_concentration": float(np.mean(max_attentions / (attention_spreads + 1e-10)))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attention focus: {str(e)}")
            return {"average_max_attention": 0.0, "average_attention_spread": 0.0, "attention_concentration": 0.0}
    
    def _identify_attention_patterns(self, attention_matrix: np.ndarray, tokens: List[str]) -> List[Dict[str, Any]]:
        """Identify dominant attention patterns"""
        try:
            patterns = []
            
            # Find tokens with highest attention weights
            for i, token in enumerate(tokens):
                attention_row = attention_matrix[i]
                top_indices = np.argsort(attention_row)[-3:][::-1]  # Top 3 attended tokens
                
                pattern = {
                    "source_token": token,
                    "source_index": i,
                    "top_attended_tokens": [
                        {
                            "token": tokens[idx],
                            "index": int(idx),
                            "attention_weight": float(attention_row[idx])
                        }
                        for idx in top_indices if attention_row[idx] > 0.1  # Threshold
                    ]
                }
                
                if pattern["top_attended_tokens"]:  # Only include if there are significant patterns
                    patterns.append(pattern)
            
            return patterns[:10]  # Return top 10 patterns
            
        except Exception as e:
            self.logger.error(f"Failed to identify attention patterns: {str(e)}")
            return []
    
    def _create_multi_head_comparison(self, layer_attention: np.ndarray, tokens: List[str]) -> Dict[str, Any]:
        """Create visualization comparing multiple attention heads"""
        try:
            num_heads = layer_attention.shape[0]
            
            # Create subplots for each head
            fig = make_subplots(
                rows=2, cols=(num_heads + 1) // 2,
                subplot_titles=[f"Head {i}" for i in range(num_heads)],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for head_idx in range(num_heads):
                row = (head_idx // ((num_heads + 1) // 2)) + 1
                col = (head_idx % ((num_heads + 1) // 2)) + 1
                
                fig.add_trace(
                    go.Heatmap(
                        z=layer_attention[head_idx],
                        x=tokens,
                        y=tokens,
                        colorscale='Blues',
                        showscale=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Multi-Head Attention Comparison",
                height=600,
                width=1200
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-head comparison: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _calculate_attention_statistics(self, attention_weights: Tuple) -> Dict[str, Any]:
        """Calculate comprehensive attention statistics"""
        try:
            stats = {
                "num_layers": len(attention_weights),
                "num_heads": attention_weights[0].shape[1],
                "sequence_length": attention_weights[0].shape[-1],
                "layer_stats": []
            }
            
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_attention_np = layer_attention[0].numpy()  # Remove batch dimension
                
                layer_stat = {
                    "layer": layer_idx,
                    "average_entropy": np.mean([
                        self._calculate_attention_entropy(layer_attention_np[head])
                        for head in range(layer_attention_np.shape[0])
                    ]),
                    "max_attention": float(np.max(layer_attention_np)),
                    "min_attention": float(np.min(layer_attention_np)),
                    "attention_variance": float(np.var(layer_attention_np))
                }
                stats["layer_stats"].append(layer_stat)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attention statistics: {str(e)}")
            return {"num_layers": 0, "num_heads": 0, "sequence_length": 0, "layer_stats": []}
    
    def _calculate_attention_diversity(self, layer_attention: np.ndarray) -> float:
        """Calculate attention diversity across heads in a layer"""
        try:
            num_heads = layer_attention.shape[0]
            
            # Calculate pairwise correlations between attention heads
            correlations = []
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    head_i = layer_attention[i].flatten()
                    head_j = layer_attention[j].flatten()
                    correlation = np.corrcoef(head_i, head_j)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
            
            # Diversity is inverse of average correlation
            avg_correlation = np.mean(correlations) if correlations else 0
            diversity = 1 - abs(avg_correlation)
            
            return float(diversity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attention diversity: {str(e)}")
            return 0.0
    
    def _identify_dominant_heads(self, layer_attention: np.ndarray) -> List[Dict[str, Any]]:
        """Identify dominant attention heads in a layer"""
        try:
            num_heads = layer_attention.shape[0]
            head_scores = []
            
            for head_idx in range(num_heads):
                head_attention = layer_attention[head_idx]
                
                # Calculate head dominance score based on attention concentration
                max_attentions = np.max(head_attention, axis=1)
                dominance_score = np.mean(max_attentions)
                
                head_scores.append({
                    "head_index": head_idx,
                    "dominance_score": float(dominance_score),
                    "entropy": self._calculate_attention_entropy(head_attention),
                    "focus_metrics": self._calculate_attention_focus(head_attention)
                })
            
            # Sort by dominance score
            head_scores.sort(key=lambda x: x["dominance_score"], reverse=True)
            
            return head_scores
            
        except Exception as e:
            self.logger.error(f"Failed to identify dominant heads: {str(e)}")
            return []
    
    def _create_layer_attention_summary(self, layer_attention: np.ndarray, tokens: List[str], 
                                      layer_idx: int) -> Dict[str, Any]:
        """Create summary visualization for a layer's attention"""
        try:
            # Average attention across all heads
            avg_attention = np.mean(layer_attention, axis=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=avg_attention,
                x=tokens,
                y=tokens,
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title=f"Layer {layer_idx} - Average Attention Across All Heads",
                xaxis_title="Target Tokens",
                yaxis_title="Source Tokens",
                height=500,
                width=500
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create layer attention summary: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _create_layer_progression_visualization(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Create visualization showing attention progression across layers"""
        try:
            num_layers = len(attention_weights)
            
            # Calculate average attention for each layer
            layer_averages = []
            for layer_attention in attention_weights:
                layer_avg = np.mean(layer_attention[0].numpy(), axis=0)  # Average across heads
                layer_averages.append(layer_avg)
            
            # Create animation or multi-layer visualization
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f"Layer {i}" for i in range(min(6, num_layers))],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            for i in range(min(6, num_layers)):  # Show first 6 layers
                row = (i // 3) + 1
                col = (i % 3) + 1
                
                fig.add_trace(
                    go.Heatmap(
                        z=layer_averages[i],
                        x=tokens,
                        y=tokens,
                        colorscale='Blues',
                        showscale=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Attention Progression Across Layers",
                height=800,
                width=1200
            )
            
            return {"type": "plotly", "figure": fig.to_json()}
            
        except Exception as e:
            self.logger.error(f"Failed to create layer progression visualization: {str(e)}")
            return {"type": "error", "message": str(e)}
    
    def _summarize_layer_attention(self, layer_attention: np.ndarray) -> Dict[str, Any]:
        """Create summary statistics for a layer's attention"""
        try:
            return {
                "num_heads": layer_attention.shape[0],
                "sequence_length": layer_attention.shape[1],
                "average_entropy": np.mean([
                    self._calculate_attention_entropy(layer_attention[head])
                    for head in range(layer_attention.shape[0])
                ]),
                "attention_diversity": self._calculate_attention_diversity(layer_attention),
                "max_attention_weight": float(np.max(layer_attention)),
                "min_attention_weight": float(np.min(layer_attention)),
                "attention_sparsity": float(np.mean(layer_attention < 0.01))  # Percentage of near-zero weights
            }
            
        except Exception as e:
            self.logger.error(f"Failed to summarize layer attention: {str(e)}")
            return {}
    
    def _create_global_attention_summary(self, attention_weights: Tuple) -> Dict[str, Any]:
        """Create global summary of attention patterns across all layers"""
        try:
            summary = {
                "total_layers": len(attention_weights),
                "total_heads": len(attention_weights) * attention_weights[0].shape[1],
                "sequence_length": attention_weights[0].shape[-1],
                "layer_progression": []
            }
            
            # Analyze progression across layers
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_attention_np = layer_attention[0].numpy()
                
                layer_summary = {
                    "layer": layer_idx,
                    "average_entropy": np.mean([
                        self._calculate_attention_entropy(layer_attention_np[head])
                        for head in range(layer_attention_np.shape[0])
                    ]),
                    "attention_concentration": float(np.mean(np.max(layer_attention_np, axis=-1))),
                    "inter_head_diversity": self._calculate_attention_diversity(layer_attention_np)
                }
                summary["layer_progression"].append(layer_summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to create global attention summary: {str(e)}")
            return {"total_layers": 0, "total_heads": 0, "sequence_length": 0, "layer_progression": []}
    
    async def get_capabilities(self) -> List[str]:
        """Return list of engine capabilities"""
        base_capabilities = [
            "shap_explanations",
            "lime_explanations", 
            "feature_importance",
            "bias_detection",
            "counterfactual_explanations"
        ]
        
        if TRANSFORMERS_AVAILABLE:
            base_capabilities.extend([
                "attention_visualization",
                "multi_head_attention_analysis",
                "layer_wise_attention_analysis",
                "transformer_model_setup"
            ])
        
        return base_capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """Return engine status"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "healthy": self.status.value == "ready",
            "capabilities": [cap.value for cap in self.capabilities],
            "shap_ready": self.shap_explainer is not None,
            "lime_ready": self.lime_explainer is not None,
            "model_loaded": self.model is not None,
            "transformer_ready": self.transformer_model is not None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "supported_explanation_types": self.supported_explanation_types
        }