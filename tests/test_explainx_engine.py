"""
Tests for ExplainXEngine - Explainable AI functionality
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import asyncio

from scrollintel.engines.explainx_engine import ExplainXEngine

class TestExplainXEngine:
    
    @pytest.fixture
    def engine(self):
        """Create ExplainXEngine instance"""
        return ExplainXEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data"""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train a simple model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        return {
            "model": model,
            "X_train": pd.DataFrame(X_train, columns=feature_names),
            "X_test": pd.DataFrame(X_test, columns=feature_names),
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names
        }
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        await engine.initialize()
        
        assert engine.name == "ExplainXEngine"
        assert engine.version == "2.0.0"
        assert engine.engine_id == "explainx_engine"
        status = engine.get_status()
        assert "explanation" in [cap.lower() for cap in status["capabilities"]]
        assert "bias_detection" in [cap.lower() for cap in status["capabilities"]]
    
    @pytest.mark.asyncio
    async def test_setup_explainers(self, engine, sample_data):
        """Test explainer setup"""
        await engine.initialize()
        
        result = await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        assert result["status"] == "success"
        assert result["shap_available"] is True
        assert result["lime_available"] is True
        assert engine.shap_explainer is not None
        assert engine.lime_explainer is not None
    
    @pytest.mark.asyncio
    async def test_shap_explanations(self, engine, sample_data):
        """Test SHAP explanation generation"""
        await engine.initialize()
        await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        # Test with single instance
        test_instance = sample_data["X_test"].iloc[:1]
        
        result = await engine.generate_shap_explanations(
            data=test_instance,
            explanation_type="waterfall"
        )
        
        assert result["status"] == "success"
        assert "shap_values" in result
        assert "base_values" in result
        assert "feature_names" in result
        assert "visualization" in result
        assert len(result["shap_values"][0]) == len(sample_data["feature_names"])
    
    @pytest.mark.asyncio
    async def test_lime_explanations(self, engine, sample_data):
        """Test LIME explanation generation"""
        await engine.initialize()
        await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        # Test with single instance
        test_instance = sample_data["X_test"].iloc[0]
        
        result = await engine.generate_lime_explanations(
            instance=test_instance,
            num_features=5
        )
        
        assert result["status"] == "success"
        assert "instance" in result
        assert "explanations" in result
        assert "visualization" in result
        assert len(result["explanations"]) <= 5
    
    @pytest.mark.asyncio
    async def test_feature_importance(self, engine, sample_data):
        """Test feature importance analysis"""
        await engine.initialize()
        await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        result = await engine.analyze_feature_importance(method="shap")
        
        assert result["status"] == "success"
        assert "data" in result
        assert result["data"]["method"] == "shap"
        assert "features" in result["data"]
        assert "importance_scores" in result["data"]
        assert "ranking" in result["data"]
        assert "visualization" in result["data"]
        assert len(result["data"]["features"]) == len(sample_data["feature_names"])
    
    @pytest.mark.asyncio
    async def test_counterfactual_explanations(self, engine, sample_data):
        """Test counterfactual explanation generation"""
        await engine.initialize()
        await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        # Test with single instance
        test_instance = sample_data["X_test"].iloc[0]
        
        result = await engine.generate_counterfactual_explanations(
            instance=test_instance
        )
        
        assert result["status"] == "success"
        assert "original_instance" in result
        assert "original_prediction" in result
        assert "counterfactuals" in result
        assert isinstance(result["counterfactuals"], list)
    
    @pytest.mark.asyncio
    async def test_bias_detection(self, engine, sample_data):
        """Test bias detection functionality"""
        await engine.initialize()
        await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        # Generate predictions
        predictions = sample_data["model"].predict_proba(sample_data["X_test"])[:, 1]
        labels = sample_data["y_test"]
        
        result = await engine.detect_bias(
            protected_features=["feature_0", "feature_1"],
            predictions=predictions,
            labels=labels
        )
        
        assert result["status"] == "success"
        assert "bias_metrics" in result
        assert "overall_bias_detected" in result
        assert isinstance(result["bias_metrics"], dict)
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, engine):
        """Test capabilities retrieval"""
        capabilities = await engine.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert "shap_explanations" in capabilities
        assert "lime_explanations" in capabilities
        assert "feature_importance" in capabilities
        assert "bias_detection" in capabilities
    
    @pytest.mark.asyncio
    async def test_get_status(self, engine, sample_data):
        """Test status retrieval"""
        await engine.initialize()
        
        # Test status before setup
        status = engine.get_status()
        assert status["name"] == "ExplainXEngine"
        assert status["status"] in ["active", "initializing", "ready"]
        assert status["shap_ready"] is False
        assert status["lime_ready"] is False
        assert status["model_loaded"] is False
        
        # Test status after setup
        await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        status = engine.get_status()
        assert status["shap_ready"] is True
        assert status["lime_ready"] is True
        assert status["model_loaded"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling for various scenarios"""
        await engine.initialize()
        
        # Test SHAP without setup
        result = await engine.generate_shap_explanations(
            data=pd.DataFrame([[1, 2, 3]]),
            explanation_type="waterfall"
        )
        assert result["status"] == "error"
        
        # Test LIME without setup
        result = await engine.generate_lime_explanations(
            instance=pd.Series([1, 2, 3])
        )
        assert result["status"] == "error"
        
        # Test feature importance without setup
        result = await engine.analyze_feature_importance()
        assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_different_explanation_types(self, engine, sample_data):
        """Test different SHAP explanation types"""
        await engine.initialize()
        await engine.setup_explainers(
            model=sample_data["model"],
            training_data=sample_data["X_train"],
            feature_names=sample_data["feature_names"]
        )
        
        test_instance = sample_data["X_test"].iloc[:1]
        
        # Test waterfall plot
        result = await engine.generate_shap_explanations(
            data=test_instance,
            explanation_type="waterfall"
        )
        assert result["status"] == "success"
        assert result["visualization"]["type"] == "plotly"
        
        # Test summary plot
        result = await engine.generate_shap_explanations(
            data=test_instance,
            explanation_type="summary"
        )
        assert result["status"] == "success"
        assert result["visualization"]["type"] == "plotly"
        
        # Test force plot
        result = await engine.generate_shap_explanations(
            data=test_instance,
            explanation_type="force"
        )
        assert result["status"] == "success"
        assert result["visualization"]["type"] == "plotly"

    @pytest.mark.asyncio
    async def test_transformer_model_setup(self, engine):
        """Test transformer model setup"""
        await engine.initialize()
        
        # Test setup with a small model (if transformers available)
        result = await engine.setup_transformer_model(
            model_name="distilbert-base-uncased"
        )
        
        # Should either succeed or fail gracefully if transformers not available
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "success":
            assert "model_name" in result
            assert "num_layers" in result
            assert "num_attention_heads" in result
    
    @pytest.mark.asyncio
    async def test_attention_visualization(self, engine):
        """Test attention visualization generation"""
        await engine.initialize()
        
        # Setup transformer model first
        setup_result = await engine.setup_transformer_model(
            model_name="distilbert-base-uncased"
        )
        
        if setup_result["status"] == "success":
            # Test attention visualization
            result = await engine.generate_attention_visualization(
                text="Hello world, this is a test.",
                layer=0,
                head=0
            )
            
            assert result["status"] == "success"
            assert "text" in result
            assert "tokens" in result
            assert "attention_matrix" in result
            assert "visualization" in result
            assert "statistics" in result
        else:
            # If transformers not available, should return error
            result = await engine.generate_attention_visualization(
                text="Hello world, this is a test."
            )
            assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_multi_head_attention_analysis(self, engine):
        """Test multi-head attention analysis"""
        await engine.initialize()
        
        # Setup transformer model first
        setup_result = await engine.setup_transformer_model(
            model_name="distilbert-base-uncased"
        )
        
        if setup_result["status"] == "success":
            # Test multi-head analysis
            result = await engine.generate_multi_head_attention_analysis(
                text="Hello world, this is a test.",
                layer=0
            )
            
            assert result["status"] == "success"
            assert "head_analyses" in result
            assert "multi_head_visualization" in result
            assert "layer_summary" in result
        else:
            # If transformers not available, should return error
            result = await engine.generate_multi_head_attention_analysis(
                text="Hello world, this is a test."
            )
            assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_layer_wise_attention_analysis(self, engine):
        """Test layer-wise attention analysis"""
        await engine.initialize()
        
        # Setup transformer model first
        setup_result = await engine.setup_transformer_model(
            model_name="distilbert-base-uncased"
        )
        
        if setup_result["status"] == "success":
            # Test layer-wise analysis
            result = await engine.generate_layer_wise_attention_analysis(
                text="Hello world, this is a test."
            )
            
            assert result["status"] == "success"
            assert "layer_analyses" in result
            assert "layer_progression_visualization" in result
            assert "global_attention_summary" in result
        else:
            # If transformers not available, should return error
            result = await engine.generate_layer_wise_attention_analysis(
                text="Hello world, this is a test."
            )
            assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_enhanced_capabilities(self, engine):
        """Test enhanced capabilities including attention visualization"""
        await engine.initialize()
        
        capabilities = await engine.get_capabilities()
        
        # Should always have basic capabilities
        assert "shap_explanations" in capabilities
        assert "lime_explanations" in capabilities
        assert "feature_importance" in capabilities
        assert "bias_detection" in capabilities
        assert "counterfactual_explanations" in capabilities
        
        # May have transformer capabilities if library available
        status = engine.get_status()
        if status.get("transformers_available", False):
            assert "attention_visualization" in capabilities
            assert "multi_head_attention_analysis" in capabilities
            assert "layer_wise_attention_analysis" in capabilities
    
    @pytest.mark.asyncio
    async def test_enhanced_status(self, engine):
        """Test enhanced status reporting"""
        await engine.initialize()
        
        status = engine.get_status()
        
        assert "transformer_ready" in status
        assert "transformers_available" in status
        assert "supported_explanation_types" in status
        assert isinstance(status["supported_explanation_types"], list)
        assert len(status["supported_explanation_types"]) > 0
    
    @pytest.mark.asyncio
    async def test_attention_helper_methods(self, engine):
        """Test attention analysis helper methods"""
        await engine.initialize()
        
        # Create dummy attention matrix
        attention_matrix = np.random.rand(10, 10)
        attention_matrix = attention_matrix / np.sum(attention_matrix, axis=1, keepdims=True)  # Normalize
        
        # Test entropy calculation
        entropy = engine._calculate_attention_entropy(attention_matrix)
        assert isinstance(entropy, float)
        assert entropy >= 0
        
        # Test attention focus calculation
        focus = engine._calculate_attention_focus(attention_matrix)
        assert isinstance(focus, dict)
        assert "average_max_attention" in focus
        assert "average_attention_spread" in focus
        assert "attention_concentration" in focus
        
        # Test pattern identification
        tokens = [f"token_{i}" for i in range(10)]
        patterns = engine._identify_attention_patterns(attention_matrix, tokens)
        assert isinstance(patterns, list)
        
        # Test attention diversity calculation
        layer_attention = np.random.rand(8, 10, 10)  # 8 heads, 10x10 attention
        diversity = engine._calculate_attention_diversity(layer_attention)
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1

if __name__ == "__main__":
    pytest.main([__file__])