"""
Model implementations for visual generation.
"""

# Import models dynamically to avoid circular import issues
def get_image_models():
    from .image_models import StableDiffusionXLModel, DALLE3Model, MidjourneyModel
    return StableDiffusionXLModel, DALLE3Model, MidjourneyModel

def get_video_models():
    from .video_models import ProprietaryNeuralRenderer, UltraRealisticVideoGenerator
    return ProprietaryNeuralRenderer, UltraRealisticVideoGenerator

def get_enhancement_models():
    from .enhancement_models import ImageEnhancer, VideoEnhancer
    return ImageEnhancer, VideoEnhancer

# For backward compatibility
StableDiffusionXLModel = None
DALLE3Model = None
MidjourneyModel = None
ProprietaryNeuralRenderer = None
UltraRealisticVideoGenerator = None
ImageEnhancer = None
VideoEnhancer = None

def _lazy_import():
    global StableDiffusionXLModel, DALLE3Model, MidjourneyModel
    global ProprietaryNeuralRenderer, UltraRealisticVideoGenerator
    global ImageEnhancer, VideoEnhancer
    
    if StableDiffusionXLModel is None:
        StableDiffusionXLModel, DALLE3Model, MidjourneyModel = get_image_models()
        ProprietaryNeuralRenderer, UltraRealisticVideoGenerator = get_video_models()
        ImageEnhancer, VideoEnhancer = get_enhancement_models()

__all__ = [
    'get_image_models',
    'get_video_models', 
    'get_enhancement_models',
    'StableDiffusionXLModel',
    'DALLE3Model', 
    'MidjourneyModel',
    'ProprietaryNeuralRenderer',
    'UltraRealisticVideoGenerator',
    'ImageEnhancer',
    'VideoEnhancer'
]