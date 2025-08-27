"""
Unit tests for MultimodalEngine
Tests multimodal processing, cross-modal fusion, and individual processors.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.multimodal_engine import (
    MultimodalEngine,
    TextProcessor,
    ImageProcessor,
    AudioProcessor,
    CrossModalFusion,
    MultimodalInput,
    ModalityType,
    ModalityEmbedding,
    get_multimodal_engine,
    initialize_multimodal_engine
)

class TestTextProcessor:
    """Unit tests for text processing"""
    
    def test_initialization(self):
        """Test text processor initialization"""
        processor = TextProcessor()
        assert processor.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_process_text_basic(self):
        """Test basic text processing"""
        processor = TextProcessor()
        
        text = "This is a sample text for testing."
        embedding = processor.process_text(text)
        
        assert embedding.modality == ModalityType.TEXT
        assert isinstance(embedding.embedding, np.ndarray)
        assert len(embedding.embedding) == 384  # Expected dimension
        assert embedding.confidence > 0
        assert embedding.processing_time >= 0
        assert 'word_count' in embedding.features
        assert 'sentiment_score' in embedding.features
    
    def test_text_features_extraction(self):
        """Test text feature extraction"""
        processor = TextProcessor()
        
        text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
        embedding = processor.process_text(text)
        
        features = embedding.features
        assert features['word_count'] == 14
        assert features['sentence_count'] >= 2  # May vary based on sentence splitting
        assert features['character_count'] == len(text)
        assert 'avg_word_length' in features
        assert 'readability_score' in features
    
    def test_sentiment_analysis(self):
        """Test simple sentiment analysis"""
        processor = TextProcessor()
        
        # Positive text
        positive_text = "This is amazing and wonderful!"
        pos_embedding = processor.process_text(positive_text)
        pos_sentiment = pos_embedding.features['sentiment_score']
        
        # Negative text
        negative_text = "This is terrible and awful!"
        neg_embedding = processor.process_text(negative_text)
        neg_sentiment = neg_embedding.features['sentiment_score']
        
        assert pos_sentiment > neg_sentiment
    
    def test_empty_text(self):
        """Test processing empty text"""
        processor = TextProcessor()
        
        embedding = processor.process_text("")
        assert embedding.modality == ModalityType.TEXT
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.features['word_count'] == 0

class TestImageProcessor:
    """Unit tests for image processing"""
    
    def test_initialization(self):
        """Test image processor initialization"""
        processor = ImageProcessor()
        # Should initialize without error regardless of PyTorch availability
    
    def test_process_image_numpy(self):
        """Test processing numpy array image"""
        processor = ImageProcessor()
        
        # Create synthetic RGB image
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        embedding = processor.process_image(image)
        
        assert embedding.modality == ModalityType.IMAGE
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.confidence > 0
        assert embedding.processing_time >= 0
        
        features = embedding.features
        assert features['width'] == 64
        assert features['height'] == 64
        assert features['channels'] == 3
        assert features['aspect_ratio'] == 1.0
    
    def test_process_grayscale_image(self):
        """Test processing grayscale image"""
        processor = ImageProcessor()
        
        # Create synthetic grayscale image
        image = np.random.randint(0, 256, (32, 48), dtype=np.uint8)
        embedding = processor.process_image(image)
        
        features = embedding.features
        assert features['width'] == 48
        assert features['height'] == 32
        assert features['channels'] == 1
        assert features['is_grayscale'] is True
    
    def test_image_features_extraction(self):
        """Test image feature extraction"""
        processor = ImageProcessor()
        
        # Create image with known properties
        image = np.full((100, 200, 3), 128, dtype=np.uint8)  # Gray image
        embedding = processor.process_image(image)
        
        features = embedding.features
        assert features['mean_brightness'] == 128.0
        assert features['std_brightness'] == 0.0  # Uniform color
        assert features['dominant_color'] == [128, 128, 128]
    
    def test_invalid_image_data(self):
        """Test handling invalid image data"""
        processor = ImageProcessor()
        
        with pytest.raises(ValueError):
            processor.process_image("invalid_data")

class TestAudioProcessor:
    """Unit tests for audio processing"""
    
    def test_initialization(self):
        """Test audio processor initialization"""
        processor = AudioProcessor()
        assert processor.sample_rate == 16000
        assert processor.n_mfcc == 13
    
    def test_process_audio_numpy(self):
        """Test processing numpy array audio"""
        processor = AudioProcessor()
        
        # Create synthetic audio (sine wave)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        embedding = processor.process_audio(audio)
        
        assert embedding.modality == ModalityType.AUDIO
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.confidence > 0
        assert embedding.processing_time >= 0
        
        features = embedding.features
        assert abs(features['duration_seconds'] - 1.0) < 0.1
        assert features['sample_rate'] == 16000
        assert features['rms_energy'] > 0
    
    def test_audio_features_extraction(self):
        """Test audio feature extraction"""
        processor = AudioProcessor()
        
        # Create silent audio
        silent_audio = np.zeros(16000)  # 1 second of silence
        embedding = processor.process_audio(silent_audio)
        
        features = embedding.features
        assert features['rms_energy'] == 0.0
        assert features['max_amplitude'] == 0.0
        assert features['silence_ratio'] >= 0.0  # Should be silent or mostly silent
        assert features['estimated_speech'] == False
    
    def test_speech_detection(self):
        """Test simple speech activity detection"""
        processor = AudioProcessor()
        
        # Create audio with energy (simulated speech)
        speech_audio = 0.1 * np.random.normal(0, 1, 16000)
        embedding = processor.process_audio(speech_audio)
        
        # Should detect as speech due to energy
        assert embedding.features['estimated_speech'] == True

class TestCrossModalFusion:
    """Unit tests for cross-modal fusion"""
    
    def test_initialization(self):
        """Test fusion engine initialization"""
        fusion = CrossModalFusion()
        assert 'concatenation' in fusion.fusion_methods
        assert 'attention' in fusion.fusion_methods
        assert 'weighted_average' in fusion.fusion_methods
    
    def test_single_modality_fusion(self):
        """Test fusion with single modality"""
        fusion = CrossModalFusion()
        
        # Create single embedding
        embedding = ModalityEmbedding(
            modality=ModalityType.TEXT,
            embedding=np.random.rand(100),
            features={'test': 'value'},
            confidence=0.8,
            processing_time=0.1
        )
        
        result = fusion.fuse_embeddings([embedding])
        
        assert len(result.input_modalities) == 1
        assert result.input_modalities[0] == ModalityType.TEXT
        assert np.array_equal(result.fused_embedding, embedding.embedding)
        assert result.fusion_confidence == embedding.confidence
    
    def test_multimodal_fusion(self):
        """Test fusion with multiple modalities"""
        fusion = CrossModalFusion()
        
        # Create embeddings for different modalities
        text_embedding = ModalityEmbedding(
            modality=ModalityType.TEXT,
            embedding=np.random.rand(100),
            features={'word_count': 10},
            confidence=0.8,
            processing_time=0.1
        )
        
        image_embedding = ModalityEmbedding(
            modality=ModalityType.IMAGE,
            embedding=np.random.rand(150),
            features={'width': 64, 'height': 64},
            confidence=0.9,
            processing_time=0.2
        )
        
        embeddings = [text_embedding, image_embedding]
        
        # Test different fusion methods
        for method in ['concatenation', 'attention', 'weighted_average']:
            result = fusion.fuse_embeddings(embeddings, method)
            
            assert len(result.input_modalities) == 2
            assert ModalityType.TEXT in result.input_modalities
            assert ModalityType.IMAGE in result.input_modalities
            assert isinstance(result.fused_embedding, np.ndarray)
            assert result.fusion_confidence > 0
            assert result.processing_metadata['fusion_method'] == method
    
    def test_fusion_confidence_calculation(self):
        """Test fusion confidence calculation"""
        fusion = CrossModalFusion()
        
        # High confidence embeddings
        high_conf_embeddings = [
            ModalityEmbedding(ModalityType.TEXT, np.random.rand(100), {}, 0.9, 0.1),
            ModalityEmbedding(ModalityType.IMAGE, np.random.rand(100), {}, 0.8, 0.1)
        ]
        
        # Low confidence embeddings
        low_conf_embeddings = [
            ModalityEmbedding(ModalityType.TEXT, np.random.rand(100), {}, 0.3, 0.1),
            ModalityEmbedding(ModalityType.IMAGE, np.random.rand(100), {}, 0.2, 0.1)
        ]
        
        high_result = fusion.fuse_embeddings(high_conf_embeddings)
        low_result = fusion.fuse_embeddings(low_conf_embeddings)
        
        assert high_result.fusion_confidence > low_result.fusion_confidence
    
    def test_cross_modal_insights(self):
        """Test cross-modal insight generation"""
        fusion = CrossModalFusion()
        
        embeddings = [
            ModalityEmbedding(ModalityType.TEXT, np.random.rand(100), 
                            {'word_count': 20, 'sentiment_score': 0.5}, 0.8, 0.1),
            ModalityEmbedding(ModalityType.IMAGE, np.random.rand(100), 
                            {'width': 128, 'height': 128}, 0.9, 0.2)
        ]
        
        result = fusion.fuse_embeddings(embeddings)
        
        insights = result.insights
        assert 'modalities_present' in insights
        assert 'cross_modal_coherence' in insights
        assert 'dominant_modality' in insights
        assert 'text_image_alignment' in insights
        
        assert len(insights['modalities_present']) == 2
        assert 'text' in insights['modalities_present']
        assert 'image' in insights['modalities_present']

class TestMultimodalEngine:
    """Unit tests for main multimodal engine"""
    
    @pytest.fixture
    def multimodal_engine(self):
        """Create multimodal engine for testing"""
        return MultimodalEngine()
    
    def test_initialization(self, multimodal_engine):
        """Test multimodal engine initialization"""
        engine = multimodal_engine
        
        assert isinstance(engine.text_processor, TextProcessor)
        assert isinstance(engine.image_processor, ImageProcessor)
        assert isinstance(engine.audio_processor, AudioProcessor)
        assert isinstance(engine.fusion_engine, CrossModalFusion)
        assert 'max_file_size' in engine.config
    
    @pytest.mark.asyncio
    async def test_process_text(self, multimodal_engine):
        """Test text processing through engine"""
        engine = multimodal_engine
        
        text = "Sample text for testing"
        embedding = await engine.process_text(text, {"source": "test"})
        
        assert embedding.modality == ModalityType.TEXT
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.confidence > 0
    
    @pytest.mark.asyncio
    async def test_process_image(self, multimodal_engine):
        """Test image processing through engine"""
        engine = multimodal_engine
        
        # Create synthetic image
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        embedding = await engine.process_image(image, {"source": "test"})
        
        assert embedding.modality == ModalityType.IMAGE
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.confidence > 0
    
    @pytest.mark.asyncio
    async def test_process_audio(self, multimodal_engine):
        """Test audio processing through engine"""
        engine = multimodal_engine
        
        # Create synthetic audio
        audio = np.random.rand(16000).astype(np.float32)
        embedding = await engine.process_audio(audio, {"source": "test"})
        
        assert embedding.modality == ModalityType.AUDIO
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.confidence > 0
    
    @pytest.mark.asyncio
    async def test_multimodal_processing(self, multimodal_engine):
        """Test multimodal input processing"""
        engine = multimodal_engine
        
        # Create multimodal inputs
        text_input = MultimodalInput(
            modality=ModalityType.TEXT,
            data="Test text",
            metadata={},
            timestamp=datetime.now()
        )
        
        image_input = MultimodalInput(
            modality=ModalityType.IMAGE,
            data=np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
            metadata={},
            timestamp=datetime.now()
        )
        
        inputs = [text_input, image_input]
        result = await engine.process_multimodal_input(inputs)
        
        assert len(result.input_modalities) == 2
        assert ModalityType.TEXT in result.input_modalities
        assert ModalityType.IMAGE in result.input_modalities
        assert isinstance(result.fused_embedding, np.ndarray)
        assert result.fusion_confidence > 0
        assert len(result.individual_embeddings) == 2
    
    def test_validate_input_size(self, multimodal_engine):
        """Test input size validation"""
        engine = multimodal_engine
        
        # Small input should be valid
        small_data = "small text"
        assert engine.validate_input_size(small_data) is True
        
        # Large input should be invalid
        large_data = "x" * (engine.config['max_file_size'] + 1)
        assert engine.validate_input_size(large_data) is False
    
    def test_get_supported_formats(self, multimodal_engine):
        """Test getting supported formats"""
        engine = multimodal_engine
        
        formats = engine.get_supported_formats()
        
        assert 'image' in formats
        assert 'audio' in formats
        assert 'text' in formats
        assert isinstance(formats['image'], list)
        assert isinstance(formats['audio'], list)
        assert isinstance(formats['text'], list)
    
    @pytest.mark.asyncio
    async def test_processing_stats(self, multimodal_engine):
        """Test getting processing statistics"""
        engine = multimodal_engine
        
        stats = await engine.get_processing_stats()
        
        assert 'total_processed' in stats
        assert 'by_modality' in stats
        assert 'average_processing_time' in stats
        assert 'cache_hit_rate' in stats
    
    @pytest.mark.asyncio
    async def test_search_similar(self, multimodal_engine):
        """Test similarity search"""
        engine = multimodal_engine
        
        query_embedding = np.random.rand(100)
        results = await engine.search_similar(query_embedding, ModalityType.TEXT, 5)
        
        # Should return empty list in current implementation
        assert isinstance(results, list)
    
    def test_cache_key_generation(self, multimodal_engine):
        """Test cache key generation"""
        engine = multimodal_engine
        
        input_data = MultimodalInput(
            modality=ModalityType.TEXT,
            data="test text",
            metadata={},
            timestamp=datetime.now()
        )
        
        cache_key = engine._generate_cache_key(input_data)
        
        assert isinstance(cache_key, str)
        assert cache_key.startswith("multimodal:text:")
        assert len(cache_key) > 20  # Should include hash

class TestMultimodalEngineIntegration:
    """Integration tests for multimodal engine"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization"""
        engine = await initialize_multimodal_engine()
        
        assert engine is not None
        assert isinstance(engine, MultimodalEngine)
    
    def test_global_engine_instance(self):
        """Test global engine instance"""
        engine1 = get_multimodal_engine()
        engine2 = get_multimodal_engine()
        
        # Should return the same instance
        assert engine1 is engine2
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test end-to-end multimodal processing"""
        engine = get_multimodal_engine()
        
        # Process individual modalities
        text_embedding = await engine.process_text("Hello world")
        image_embedding = await engine.process_image(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        audio_embedding = await engine.process_audio(np.random.rand(8000).astype(np.float32))
        
        # Verify all embeddings were created
        assert text_embedding.modality == ModalityType.TEXT
        assert image_embedding.modality == ModalityType.IMAGE
        assert audio_embedding.modality == ModalityType.AUDIO
        
        # Test multimodal fusion
        inputs = [
            MultimodalInput(ModalityType.TEXT, "test", {}, datetime.now()),
            MultimodalInput(ModalityType.IMAGE, np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8), {}, datetime.now())
        ]
        
        result = await engine.process_multimodal_input(inputs)
        
        assert len(result.input_modalities) == 2
        assert result.fusion_confidence > 0
        assert 'cross_modal_coherence' in result.insights

if __name__ == "__main__":
    pytest.main([__file__, "-v"])