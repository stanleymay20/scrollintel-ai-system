"""
ExplainX API Routes - Explainable AI endpoints for ScrollIntel
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import joblib
import io
import json
from datetime import datetime

from ...engines.explainx_engine import ExplainXEngine
from ...security.auth import get_current_user
from ...models.schemas import User

router = APIRouter(prefix="/explainx", tags=["explainx"])

# Global engine instance
explainx_engine = ExplainXEngine()

class ExplainerSetupRequest(BaseModel):
    model_path: str
    training_data_path: Optional[str] = None
    feature_names: List[str]

class ShapExplanationRequest(BaseModel):
    data: List[List[float]]
    explanation_type: str = "waterfall"  # waterfall, summary, force

class LimeExplanationRequest(BaseModel):
    instance: List[float]
    num_features: int = 10

class BiasDetectionRequest(BaseModel):
    protected_features: List[str]
    predictions: List[float]
    labels: List[float]

class CounterfactualRequest(BaseModel):
    instance: List[float]
    target_class: Optional[int] = None

@router.post("/setup")
async def setup_explainers(
    request: ExplainerSetupRequest,
    current_user: User = Depends(get_current_user)
):
    """Setup SHAP and LIME explainers for a model"""
    try:
        # Load model
        model = joblib.load(request.model_path)
        
        # Load or create training data
        if request.training_data_path:
            training_data = pd.read_csv(request.training_data_path)
        else:
            # Create dummy training data if not provided
            training_data = pd.DataFrame(
                np.random.randn(100, len(request.feature_names)),
                columns=request.feature_names
            )
        
        # Setup explainers
        result = await explainx_engine.setup_explainers(
            model=model,
            training_data=training_data,
            feature_names=request.feature_names
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup explainers: {str(e)}")

@router.post("/shap/explain")
async def generate_shap_explanations(
    request: ShapExplanationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate SHAP explanations for given data"""
    try:
        # Convert data to DataFrame
        data = pd.DataFrame(request.data)
        
        # Generate explanations
        result = await explainx_engine.generate_shap_explanations(
            data=data,
            explanation_type=request.explanation_type
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanations: {str(e)}")

@router.post("/lime/explain")
async def generate_lime_explanations(
    request: LimeExplanationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate LIME explanations for a single instance"""
    try:
        # Convert instance to Series
        instance = pd.Series(request.instance)
        
        # Generate explanations
        result = await explainx_engine.generate_lime_explanations(
            instance=instance,
            num_features=request.num_features
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate LIME explanations: {str(e)}")

@router.get("/feature-importance")
async def analyze_feature_importance(
    method: str = "shap",
    current_user: User = Depends(get_current_user)
):
    """Analyze global feature importance"""
    try:
        result = await explainx_engine.analyze_feature_importance(method=method)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze feature importance: {str(e)}")

@router.post("/counterfactual")
async def generate_counterfactual_explanations(
    request: CounterfactualRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate counterfactual explanations"""
    try:
        # Convert instance to Series
        instance = pd.Series(request.instance)
        
        # Generate counterfactuals
        result = await explainx_engine.generate_counterfactual_explanations(
            instance=instance,
            target_class=request.target_class
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate counterfactual explanations: {str(e)}")

@router.post("/bias/detect")
async def detect_bias(
    request: BiasDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Detect potential bias in model predictions"""
    try:
        predictions = np.array(request.predictions)
        labels = np.array(request.labels)
        
        result = await explainx_engine.detect_bias(
            protected_features=request.protected_features,
            predictions=predictions,
            labels=labels
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect bias: {str(e)}")

@router.post("/upload-model")
async def upload_model(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a model file for explanation"""
    try:
        # Save uploaded file
        file_path = f"uploads/models/{current_user.id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "status": "success",
            "message": "Model uploaded successfully",
            "file_path": file_path,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

@router.post("/upload-data")
async def upload_training_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload training data for explainer setup"""
    try:
        # Read uploaded file
        content = await file.read()
        
        # Parse based on file type
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save data
        file_path = f"uploads/data/{current_user.id}_{file.filename}"
        data.to_csv(file_path, index=False)
        
        return {
            "status": "success",
            "message": "Training data uploaded successfully",
            "file_path": file_path,
            "filename": file.filename,
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "preview": data.head().to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload training data: {str(e)}")

@router.get("/status")
async def get_explainx_status(current_user: User = Depends(get_current_user)):
    """Get ExplainX engine status"""
    try:
        status = await explainx_engine.get_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/capabilities")
async def get_explainx_capabilities(current_user: User = Depends(get_current_user)):
    """Get ExplainX engine capabilities"""
    try:
        capabilities = await explainx_engine.get_capabilities()
        return {"capabilities": capabilities}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@router.post("/batch-explain")
async def batch_explain_instances(
    instances: List[List[float]],
    explanation_type: str = "shap",
    current_user: User = Depends(get_current_user)
):
    """Generate explanations for multiple instances"""
    try:
        results = []
        
        for i, instance in enumerate(instances):
            if explanation_type == "shap":
                data = pd.DataFrame([instance])
                result = await explainx_engine.generate_shap_explanations(
                    data=data,
                    explanation_type="waterfall"
                )
            elif explanation_type == "lime":
                instance_series = pd.Series(instance)
                result = await explainx_engine.generate_lime_explanations(
                    instance=instance_series,
                    num_features=10
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid explanation type")
            
            results.append({
                "instance_id": i,
                "explanation": result
            })
        
        return {
            "status": "success",
            "batch_size": len(instances),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate batch explanations: {str(e)}")

@router.post("/transformer/setup")
async def setup_transformer_model(
    model_name: str,
    model_path: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Setup transformer model for attention visualization"""
    try:
        result = await explainx_engine.setup_transformer_model(
            model_name=model_name,
            model_path=model_path
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup transformer model: {str(e)}")

@router.post("/attention/visualize")
async def generate_attention_visualization(
    text: str,
    layer: int = -1,
    head: int = -1,
    current_user: User = Depends(get_current_user)
):
    """Generate attention visualization for transformer models"""
    try:
        result = await explainx_engine.generate_attention_visualization(
            text=text,
            layer=layer,
            head=head
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate attention visualization: {str(e)}")

@router.post("/attention/multi-head")
async def generate_multi_head_attention_analysis(
    text: str,
    layer: int = -1,
    current_user: User = Depends(get_current_user)
):
    """Analyze attention patterns across all heads in a layer"""
    try:
        result = await explainx_engine.generate_multi_head_attention_analysis(
            text=text,
            layer=layer
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate multi-head attention analysis: {str(e)}")

@router.post("/attention/layer-wise")
async def generate_layer_wise_attention_analysis(
    text: str,
    current_user: User = Depends(get_current_user)
):
    """Analyze attention patterns across all layers"""
    try:
        result = await explainx_engine.generate_layer_wise_attention_analysis(text=text)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate layer-wise attention analysis: {str(e)}")

@router.get("/explanation-types")
async def get_supported_explanation_types(current_user: User = Depends(get_current_user)):
    """Get list of supported explanation types"""
    try:
        status = await explainx_engine.get_status()
        return {
            "supported_types": status.get("supported_explanation_types", []),
            "transformers_available": status.get("transformers_available", False)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get explanation types: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ExplainX Engine",
        "timestamp": datetime.now().isoformat()
    }