"""
Demo script for ScrollIntel MultimodalEngine
Demonstrates cross-modal intelligence with audio, image, and text processing.
"""

import asyncio
import logging
import json
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_multimodal_engine():
    """Comprehensive demo of multimodal processing capabilities"""
    
    print("🚀 ScrollIntel MultimodalEngine Demo")
    print("=" * 50)
    
    try:
        # Import the multimodal engine
        from scrollintel.engines.multimodal_engine import (
            get_multimodal_engine,
            initialize_multimodal_engine,
            MultimodalInput,
            ModalityType
        )
        
        # Initialize the engine
        print("\n1. Initializing MultimodalEngine...")
        engine = await initialize_multimodal_engine()
        print("✅ MultimodalEngine initialized successfully")
        
        # Check supported formats
        print("\n2. Checking supported formats...")
        formats = engine.get_supported_formats()
        for modality, format_list in formats.items():
            print(f"   {modality.capitalize()}: {', '.join(format_list)}")
        
        # Demo 1: Text Processing
        print("\n3. Processing text input...")
        
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. This is a sample text for testing.",
            "Artificial intelligence is revolutionizing the way we process and understand data.",
            "Climate change poses significant challenges for future generations worldwide."
        ]
        
        text_embeddings = []
        for i, text in enumerate(sample_texts):
            print(f"   Processing text {i+1}: '{text[:50]}...'")
            embedding = await engine.process_text(text, {"source": f"sample_{i+1}"})
            text_embeddings.append(embedding)
            
            print(f"     ✅ Embedding dimension: {len(embedding.embedding)}")
            print(f"     ✅ Confidence: {embedding.confidence:.3f}")
            print(f"     ✅ Word count: {embedding.features.get('word_count', 0)}")
            print(f"     ✅ Sentiment: {embedding.features.get('sentiment_score', 0):.3f}")
        
        # Demo 2: Image Processing (synthetic)
        print("\n4. Processing synthetic image data...")
        
        # Create synthetic image data (random RGB image)
        synthetic_images = []
        for i in range(3):
            # Create a random 64x64 RGB image
            image_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            synthetic_images.append(image_array)
        
        image_embeddings = []
        for i, image_data in enumerate(synthetic_images):
            print(f"   Processing synthetic image {i+1} (64x64 RGB)")
            embedding = await engine.process_image(image_data, {"source": f"synthetic_{i+1}"})
            image_embeddings.append(embedding)
            
            print(f"     ✅ Embedding dimension: {len(embedding.embedding)}")
            print(f"     ✅ Confidence: {embedding.confidence:.3f}")
            print(f"     ✅ Image size: {embedding.features.get('width', 0)}x{embedding.features.get('height', 0)}")
            print(f"     ✅ Brightness: {embedding.features.get('mean_brightness', 0):.1f}")
        
        # Demo 3: Audio Processing (synthetic)
        print("\n5. Processing synthetic audio data...")
        
        # Create synthetic audio data (sine waves)
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        synthetic_audios = []
        
        for i, freq in enumerate([440, 880, 1320]):  # A4, A5, E6 notes
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.3 * np.sin(2 * np.pi * freq * t)  # Sine wave
            synthetic_audios.append(audio_data.astype(np.float32))
        
        audio_embeddings = []
        for i, audio_data in enumerate(synthetic_audios):
            freq = [440, 880, 1320][i]
            print(f"   Processing synthetic audio {i+1} ({freq}Hz sine wave)")
            embedding = await engine.process_audio(audio_data, {"source": f"sine_{freq}hz"})
            audio_embeddings.append(embedding)
            
            print(f"     ✅ Embedding dimension: {len(embedding.embedding)}")
            print(f"     ✅ Confidence: {embedding.confidence:.3f}")
            print(f"     ✅ Duration: {embedding.features.get('duration_seconds', 0):.1f}s")
            print(f"     ✅ RMS Energy: {embedding.features.get('rms_energy', 0):.4f}")
        
        print(f"\n📊 Processed {len(text_embeddings)} texts, {len(image_embeddings)} images, {len(audio_embeddings)} audios")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required dependencies are installed")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.exception("Demo failed")
        return False

async def demo_cross_modal_fusion():
    """Demo cross-modal fusion capabilities"""
    
    print("\n🔗 Cross-Modal Fusion Demo")
    print("=" * 30)
    
    try:
        from scrollintel.engines.multimodal_engine import (
            get_multimodal_engine,
            MultimodalInput,
            ModalityType
        )
        
        engine = get_multimodal_engine()
        
        # Create multimodal inputs
        print("\n6. Creating multimodal inputs for fusion...")
        
        # Text input
        text_input = MultimodalInput(
            modality=ModalityType.TEXT,
            data="A beautiful sunset over the ocean with birds flying in the sky",
            metadata={"description": "Descriptive text about a scenic view"},
            timestamp=datetime.now()
        )
        
        # Synthetic image (representing a sunset scene)
        sunset_image = np.random.randint(200, 255, (128, 128, 3), dtype=np.uint8)  # Bright colors
        image_input = MultimodalInput(
            modality=ModalityType.IMAGE,
            data=sunset_image,
            metadata={"scene": "sunset", "location": "ocean"},
            timestamp=datetime.now()
        )
        
        # Synthetic audio (representing ocean waves)
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create wave-like sound (low frequency with some noise)
        wave_sound = 0.2 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.normal(0, 0.1, len(t))
        audio_input = MultimodalInput(
            modality=ModalityType.AUDIO,
            data=wave_sound.astype(np.float32),
            metadata={"environment": "ocean", "type": "ambient"},
            timestamp=datetime.now()
        )
        
        multimodal_inputs = [text_input, image_input, audio_input]
        
        # Test different fusion methods
        fusion_methods = ['concatenation', 'attention', 'weighted_average']
        
        for method in fusion_methods:
            print(f"\n   Testing {method} fusion...")
            
            try:
                result = await engine.process_multimodal_input(multimodal_inputs, method)
                
                print(f"     ✅ Fusion method: {method}")
                print(f"     ✅ Input modalities: {[mod.value for mod in result.input_modalities]}")
                print(f"     ✅ Fused embedding dimension: {len(result.fused_embedding)}")
                print(f"     ✅ Fusion confidence: {result.fusion_confidence:.3f}")
                print(f"     ✅ Dominant modality: {result.insights.get('dominant_modality', 'unknown')}")
                print(f"     ✅ Cross-modal coherence: {result.insights.get('cross_modal_coherence', 0):.3f}")
                
                # Show individual embedding confidences
                individual_confidences = [emb.confidence for emb in result.individual_embeddings]
                print(f"     ✅ Individual confidences: {[f'{c:.3f}' for c in individual_confidences]}")
                
            except Exception as e:
                print(f"     ❌ Failed with {method}: {e}")
        
        # Demo single modality processing
        print(f"\n7. Testing single modality processing...")
        
        single_result = await engine.process_multimodal_input([text_input])
        print(f"   ✅ Single modality (text) confidence: {single_result.fusion_confidence:.3f}")
        print(f"   ✅ Processing metadata: {single_result.processing_metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-modal fusion demo failed: {e}")
        logger.exception("Cross-modal fusion demo failed")
        return False

async def demo_api_integration():
    """Demo API integration with multimodal engine"""
    
    print("\n🌐 API Integration Demo")
    print("=" * 30)
    
    try:
        # Simulate API calls
        print("📡 Simulating API calls...")
        
        # Text processing API
        text_request = {
            "text": "This is a sample text for API processing",
            "metadata": {"source": "api_demo"}
        }
        
        print(f"   POST /api/multimodal/process/text")
        print(f"   Request: {json.dumps(text_request, indent=2)}")
        print("   Response: {")
        print("     'modality': 'text',")
        print("     'embedding': [0.1, 0.2, ...],")
        print("     'confidence': 0.85,")
        print("     'features': {'word_count': 8, 'sentiment_score': 0.1}")
        print("   }")
        
        # Multimodal processing API
        multimodal_request = {
            "inputs": [
                {
                    "modality": "text",
                    "data": "A beautiful landscape photo",
                    "metadata": {"type": "description"}
                },
                {
                    "modality": "image",
                    "data": "base64_encoded_image_data...",
                    "metadata": {"format": "jpeg"}
                }
            ],
            "fusion_method": "attention"
        }
        
        print(f"\n   POST /api/multimodal/process/multimodal")
        print(f"   Request: {json.dumps(multimodal_request, indent=2)}")
        print("   Response: {")
        print("     'input_modalities': ['text', 'image'],")
        print("     'fused_embedding': [0.1, 0.2, ...],")
        print("     'fusion_confidence': 0.78,")
        print("     'insights': {'dominant_modality': 'image'}")
        print("   }")
        
        # File upload API
        print(f"\n   POST /api/multimodal/upload/image")
        print("   Content-Type: multipart/form-data")
        print("   Response: {")
        print("     'modality': 'image',")
        print("     'embedding': [0.1, 0.2, ...],")
        print("     'features': {'width': 1024, 'height': 768}")
        print("   }")
        
        print("\n✅ API integration demo completed")
        
    except Exception as e:
        print(f"❌ API demo failed: {e}")

def main():
    """Main demo function"""
    
    print("🔬 ScrollIntel MultimodalEngine Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases:")
    print("• Text processing and embedding generation")
    print("• Image processing with computer vision")
    print("• Audio processing and feature extraction")
    print("• Cross-modal fusion with multiple methods")
    print("• API integration examples")
    print("=" * 60)
    
    # Run the main demo
    success1 = asyncio.run(demo_multimodal_engine())
    
    if success1:
        # Run cross-modal fusion demo
        success2 = asyncio.run(demo_cross_modal_fusion())
        
        if success2:
            # Run API demo
            asyncio.run(demo_api_integration())
            
            print("\n🏆 All demos completed successfully!")
            print("\nNext steps:")
            print("1. Install optional dependencies for enhanced functionality:")
            print("   pip install torch torchvision sentence-transformers")
            print("   pip install opencv-python pillow librosa soundfile")
            print("2. Start the FastAPI server to test API endpoints")
            print("3. Upload real images and audio files for processing")
            print("4. Experiment with different fusion methods")
            print("5. Integrate with vector databases for similarity search")
        else:
            print("\n❌ Cross-modal fusion demo failed")
            return 1
    else:
        print("\n❌ Main demo failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())