"""
Visual Generation Engine Package

This package provides comprehensive visual content generation capabilities
including image generation, video generation, and enhancement tools.
"""

from .base import (
    BaseVisualGenerator, GenerationRequest, GenerationResult,
    ImageGenerationRequest, VideoGenerationRequest, GenerationStatus,
    ContentType, QualityMetrics
)
from .config import VisualGenerationConfig
from .exceptions import VisualGenerationError, ModelError, ResourceError, SafetyError

# Import engine separately to avoid circular imports
def get_engine(config_path=None):
    """Get a visual generation engine instance."""
    from .engine import VisualGenerationEngine
    return VisualGenerationEngine(config_path)

__all__ = [
    'BaseVisualGenerator',
    'GenerationRequest', 
    'GenerationResult',
    'ImageGenerationRequest',
    'VideoGenerationRequest',
    'GenerationStatus',
    'ContentType',
    'QualityMetrics',
    'VisualGenerationConfig',
    'get_engine',
    'VisualGenerationError',
    'ModelError',
    'ResourceError',
    'SafetyError'
]