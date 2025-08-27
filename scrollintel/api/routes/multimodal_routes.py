"""
API routes for MultimodalEngine
Provides REST endpoints for multimodal processing operations.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import logging
import base64
import json

from scrollintel.engines.multimodal_engine import (
    get_multimodal_engine,
    MultimodalEngine,
    MultimodalInput,
    ModalityType,
    CrossModalResult
)
from scrollintel.core.config import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/multimodal", tags=["multimodal"])

# Pydantic models for request/response
class TextProcessRequest(BaseModel):
    text: str = Field(..., description="Text content to process")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class ImageProcessRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class AudioProcessRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class MultimodalProcessRequest(BaseModel):
    inputs: List[Dict[str, Any]] = Field(..., description="List of multimodal inputs")
    fusion_method: Optional[str] = Field(default="attention", description="Fusion method to use")

class EmbeddingResponse(BaseModel):
    modality: str
    embedding: List[float]
    features: Dict[str, Any]
    confidence: float
    processing_time: float

class CrossModalResponse(BaseModel):
    input_modalities: List[str]
    fused_embedding: List[float]
    individual_embeddings: List[EmbeddingResponse]
    fusion_confidence: float
    insights: Dict[str, Any]
    processing_metadata: Dict[str, Any]

def get_multimodal_engine_dependency() -> MultimodalEngine:
    """Dependency to get multimodal engine instance"""
    redis_client = get_redis_client()
    return get_multimodal_engine(redis_client)

@router.post("/process/text", response_model=EmbeddingResponse)
async def process_text(
    request: TextProcessRequest,
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Process text input and return embedding"""
    try:
        embedding = await engine.process_text(request.text, request.metadata)
        
        return EmbeddingResponse(
            modality=embedding.modality.value,
            embedding=embedding.embedding.tolist(),
            features=embedding.features,
            confidence=embedding.confidence,
            processing_time=embedding.processing_time
        )
    except Exception as e:
        logger.error(f"Failed to process text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/image", response_model=EmbeddingResponse)
async def process_image(
    request: ImageProcessRequest,
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Process image input and return embedding"""
    try:
        embedding = await engine.process_image(request.image_data, request.metadata)
        
        return EmbeddingResponse(
            modality=embedding.modality.value,
            embedding=embedding.embedding.tolist(),
            features=embedding.features,
            confidence=embedding.confidence,
            processing_time=embedding.processing_time
        )
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/audio", response_model=EmbeddingResponse)
async def process_audio(
    request: AudioProcessRequest,
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Process audio input and return embedding"""
    try:
        embedding = await engine.process_audio(request.audio_data, request.metadata)
        
        return EmbeddingResponse(
            modality=embedding.modality.value,
            embedding=embedding.embedding.tolist(),
            features=embedding.features,
            confidence=embedding.confidence,
            processing_time=embedding.processing_time
        )
    except Exception as e:
        logger.error(f"Failed to process audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/image", response_model=EmbeddingResponse)
async def upload_and_process_image(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(default="{}"),
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Upload and process image file"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        # Validate file size
        if not engine.validate_input_size(content):
            raise HTTPException(status_code=413, detail="File too large")
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Add file metadata
        metadata_dict.update({
            'filename': file.filename,
            'content_type': file.content_type,
            'file_size': len(content)
        })
        
        # Process image
        embedding = await engine.process_image(content, metadata_dict)
        
        return EmbeddingResponse(
            modality=embedding.modality.value,
            embedding=embedding.embedding.tolist(),
            features=embedding.features,
            confidence=embedding.confidence,
            processing_time=embedding.processing_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload and process image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/audio", response_model=EmbeddingResponse)
async def upload_and_process_audio(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(default="{}"),
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Upload and process audio file"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be audio")
        
        # Read file content
        content = await file.read()
        
        # Validate file size
        if not engine.validate_input_size(content):
            raise HTTPException(status_code=413, detail="File too large")
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Add file metadata
        metadata_dict.update({
            'filename': file.filename,
            'content_type': file.content_type,
            'file_size': len(content)
        })
        
        # Process audio
        embedding = await engine.process_audio(content, metadata_dict)
        
        return EmbeddingResponse(
            modality=embedding.modality.value,
            embedding=embedding.embedding.tolist(),
            features=embedding.features,
            confidence=embedding.confidence,
            processing_time=embedding.processing_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload and process audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/multimodal", response_model=CrossModalResponse)
async def process_multimodal(
    request: MultimodalProcessRequest,
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Process multiple modalities and return fused result"""
    try:
        # Convert request inputs to MultimodalInput objects
        multimodal_inputs = []
        
        for input_data in request.inputs:
            modality_str = input_data.get('modality')
            data = input_data.get('data')
            metadata = input_data.get('metadata', {})
            
            if not modality_str or not data:
                raise HTTPException(status_code=400, detail="Each input must have 'modality' and 'data'")
            
            try:
                modality = ModalityType(modality_str)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid modality: {modality_str}")
            
            multimodal_input = MultimodalInput(
                modality=modality,
                data=data,
                metadata=metadata,
                timestamp=datetime.now()
            )
            multimodal_inputs.append(multimodal_input)
        
        # Process multimodal inputs
        result = await engine.process_multimodal_input(multimodal_inputs, request.fusion_method)
        
        # Convert individual embeddings to response format
        individual_embeddings = [
            EmbeddingResponse(
                modality=emb.modality.value,
                embedding=emb.embedding.tolist(),
                features=emb.features,
                confidence=emb.confidence,
                processing_time=emb.processing_time
            )
            for emb in result.individual_embeddings
        ]
        
        return CrossModalResponse(
            input_modalities=[mod.value for mod in result.input_modalities],
            fused_embedding=result.fused_embedding.tolist(),
            individual_embeddings=individual_embeddings,
            fusion_confidence=result.fusion_confidence,
            insights=result.insights,
            processing_metadata=result.processing_metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process multimodal input: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/formats")
async def get_supported_formats(
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Get supported file formats"""
    try:
        formats = engine.get_supported_formats()
        return {"supported_formats": formats}
    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_processing_stats(
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Get processing statistics"""
    try:
        stats = await engine.get_processing_stats()
        return {"statistics": stats}
    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/similar")
async def search_similar_embeddings(
    embedding: List[float],
    modality_filter: Optional[str] = None,
    limit: int = 10,
    engine: MultimodalEngine = Depends(get_multimodal_engine_dependency)
):
    """Search for similar embeddings"""
    try:
        import numpy as np
        
        # Validate modality filter
        modality_type = None
        if modality_filter:
            try:
                modality_type = ModalityType(modality_filter)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid modality filter: {modality_filter}")
        
        # Convert embedding to numpy array
        query_embedding = np.array(embedding)
        
        # Search for similar embeddings
        results = await engine.search_similar(query_embedding, modality_type, limit)
        
        return {"similar_embeddings": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search similar embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint for multimodal engine"""
    try:
        from datetime import datetime
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "multimodal_engine"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))