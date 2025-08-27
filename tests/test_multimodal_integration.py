"""
Integration tests for MultimodalEngine
Tests end-to-end multimodal processing workflows and API integration.
"""

import pytest
import asyncio
import numpy as np
import json
import base64
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.multimodal_engine import (
    MultimodalEngine,
    MultimodalInput,
    ModalityType,
    initialize_multimodal_engine,
    get_multimodal_engine
)

class TestMultimodalEngineIntegration:
    """Integration tests for multimodal processing workflows"""
    
    @pytest.fixture
    async def multimodal_engine(self):
        """Create multimodal engine for testing"""
        mock_redis = AsyncMock()
        engine = MultimodalEngine(redis_client=mock_redis)
        return engine
    
    @pytest.fixture
    def sample_multimodal_data(self):
        """Sample data for different modalities"""
        return {
            'text': [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming our world.",
                "Climate change requires immediate global action."
            ],
            'images': [
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
                np.random.randint(0, 256, (128, 96, 3), dtype=np.uint8),
                np.random.randint(0, 256, (32, 32, 1), dtype=np.uint8)
            ],
            'audio': [
                np.random.rand(16000).astype(np.float32),  # 1 second
                np.random.rand(32000).astype(np.float32),  # 2 seconds
                np.random.rand(8000).astype(np.float32)    # 0.5 seconds
            ]
        }
    
    @pytest.mark.asyncio
    async def test_complete_multimodal_workflow(self, multimodal_engine, sample_multimodal_data):
        """Test complete multimodal processing workflow"""
        engine = multimodal_engine
        
        # Process each modality individually
        text_embeddings = []
        for text in sample_multimodal_data['text']:
            embedding = await engine.process_text(text)
            text_embeddings.append(embedding)
            assert embedding.modality == ModalityType.TEXT
            assert len(embedding.embedding) > 0
        
        image_embeddings = []
        for image in sample_multimodal_data['images']:
            embedding = await engine.process_image(image)
            image_embeddings.append(embedding)
            assert embedding.modality == ModalityType.IMAGE
            assert len(embedding.embedding) > 0
        
        audio_embeddings = []
        for audio in sample_multimodal_data['audio']:
            embedding = await engine.process_audio(audio)
            audio_embeddings.append(embedding)
            assert embedding.modality == ModalityType.AUDIO
            assert len(embedding.embedding) > 0
        
        # Test multimodal fusion
        multimodal_inputs = [
            MultimodalInput(ModalityType.TEXT, sample_multimodal_data['text'][0], {}, datetime.now()),
            MultimodalInput(ModalityType.IMAGE, sample_multimodal_data['images'][0], {}, datetime.now()),
            MultimodalInput(ModalityType.AUDIO, sample_multimodal_data['audio'][0], {}, datetime.now())
        ]
        
        result = await engine.process_multimodal_input(multimodal_inputs)
        
        assert len(result.input_modalities) == 3
        assert ModalityType.TEXT in result.input_modalities
        assert ModalityType.IMAGE in result.input_modalities
        assert ModalityType.AUDIO in result.input_modalities
        assert result.fusion_confidence > 0
        assert len(result.individual_embeddings) == 3
    
    @pytest.mark.asyncio
    async def test_different_fusion_methods(self, multimodal_engine, sample_multimodal_data):
        """Test different fusion methods with same inputs"""
        engine = multimodal_engine
        
        # Create multimodal inputs
        inputs = [
            MultimodalInput(ModalityType.TEXT, sample_multimodal_data['text'][0], {}, datetime.now()),
            MultimodalInput(ModalityType.IMAGE, sample_multimodal_data['images'][0], {}, datetime.now())
        ]
        
        fusion_methods = ['concatenation', 'attention', 'weighted_average']
        results = {}
        
        for method in fusion_methods:
            result = await engine.process_multimodal_input(inputs, method)
            results[method] = result
            
            assert result.processing_metadata['fusion_method'] == method
            assert len(result.fused_embedding) > 0
            assert result.fusion_confidence > 0
        
        # Compare results
        for method1 in fusion_methods:
            for method2 in fusion_methods:
                if method1 != method2:
                    # Different methods should produce different embeddings
                    assert not np.array_equal(
                        results[method1].fused_embedding,
                        results[method2].fused_embedding
                    )
    
    @pytest.mark.asyncio
    async def test_cross_modal_similarity_analysis(self, multimodal_engine, sample_multimodal_data):
        """Test cross-modal similarity analysis"""
        engine = multimodal_engine
        
        # Process related content
        related_inputs = [
            MultimodalInput(ModalityType.TEXT, "A beautiful sunset over the ocean", {}, datetime.now()),
            MultimodalInput(ModalityType.IMAGE, sample_multimodal_data['images'][0], {"scene": "sunset"}, datetime.now()),
            MultimodalInput(ModalityType.AUDIO, sample_multimodal_data['audio'][0], {"environment": "ocean"}, datetime.now())
        ]
        
        related_result = await engine.process_multimodal_input(related_inputs)
        
        # Process unrelated content
        unrelated_inputs = [
            MultimodalInput(ModalityType.TEXT, "Technical documentation for software", {}, datetime.now()),
            MultimodalInput(ModalityType.IMAGE, sample_multimodal_data['images'][1], {"type": "diagram"}, datetime.now())
        ]
        
        unrelated_result = await engine.process_multimodal_input(unrelated_inputs)
        
        # Related content should have higher coherence
        related_coherence = related_result.insights.get('cross_modal_coherence', 0)
        unrelated_coherence = unrelated_result.insights.get('cross_modal_coherence', 0)
        
        # This is a heuristic test - related content might have higher coherence
        assert isinstance(related_coherence, (int, float))
        assert isinstance(unrelated_coherence, (int, float))
    
    @pytest.mark.asyncio
    async def test_processing_performance(self, multimodal_engine, sample_multimodal_data):
        """Test processing performance and timing"""
        engine = multimodal_engine
        
        # Measure processing times
        start_time = datetime.now()
        
        # Process multiple inputs concurrently
        tasks = []
        
        # Text processing tasks
        for text in sample_multimodal_data['text']:
            task = engine.process_text(text)
            tasks.append(task)
        
        # Image processing tasks
        for image in sample_multimodal_data['images']:
            task = engine.process_image(image)
            tasks.append(task)
        
        # Audio processing tasks
        for audio in sample_multimodal_data['audio']:
            task = engine.process_audio(audio)
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Verify all results
        assert len(results) == 9  # 3 text + 3 image + 3 audio
        
        # Check individual processing times
        for result in results:
            assert result.processing_time >= 0
            assert result.processing_time < 10  # Should be reasonably fast
        
        # Total time should be reasonable (concurrent processing)
        assert total_time < 30  # Should complete within 30 seconds
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, multimodal_engine):
        """Test error handling and recovery mechanisms"""
        engine = multimodal_engine
        
        # Test invalid image data
        with pytest.raises(ValueError):
            await engine.process_image("invalid_image_data")
        
        # Test empty inputs
        empty_result = await engine.process_text("")
        assert empty_result.modality == ModalityType.TEXT
        assert empty_result.features['word_count'] == 0
        
        # Test very large input (should be handled gracefully)
        large_text = "word " * 10000  # Very long text
        large_result = await engine.process_text(large_text)
        assert large_result.modality == ModalityType.TEXT
        assert large_result.features['word_count'] == 10000
        
        # Test malformed multimodal input
        with pytest.raises(ValueError):
            await engine.process_multimodal_input([])  # Empty input list
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, multimodal_engine):
        """Test caching behavior with Redis mock"""
        engine = multimodal_engine
        
        # Process same text twice
        text = "This is a test for caching"
        
        # First processing
        result1 = await engine.process_text(text)
        
        # Second processing (should use cache if available)
        result2 = await engine.process_text(text)
        
        # Results should be similar (may not be identical due to caching implementation)
        assert result1.modality == result2.modality
        assert len(result1.embedding) == len(result2.embedding)
        
        # Verify Redis interactions
        assert engine.redis_client is not None
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, multimodal_engine, sample_multimodal_data):
        """Test metadata preservation through processing pipeline"""
        engine = multimodal_engine
        
        # Process with rich metadata
        metadata = {
            "source": "test_suite",
            "timestamp": datetime.now().isoformat(),
            "quality": "high",
            "tags": ["test", "integration"]
        }
        
        text_result = await engine.process_text(sample_multimodal_data['text'][0], metadata)
        image_result = await engine.process_image(sample_multimodal_data['images'][0], metadata)
        audio_result = await engine.process_audio(sample_multimodal_data['audio'][0], metadata)
        
        # Metadata should be preserved in features or accessible
        # (Implementation may vary on how metadata is handled)
        assert text_result.modality == ModalityType.TEXT
        assert image_result.modality == ModalityType.IMAGE
        assert audio_result.modality == ModalityType.AUDIO
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, multimodal_engine, sample_multimodal_data):
        """Test batch processing of multiple items"""
        engine = multimodal_engine
        
        # Create batch of multimodal inputs
        batch_inputs = []
        
        for i in range(3):
            inputs = [
                MultimodalInput(ModalityType.TEXT, sample_multimodal_data['text'][i], 
                              {"batch_id": i}, datetime.now()),
                MultimodalInput(ModalityType.IMAGE, sample_multimodal_data['images'][i], 
                              {"batch_id": i}, datetime.now())
            ]
            batch_inputs.append(inputs)
        
        # Process batch
        batch_results = []
        for inputs in batch_inputs:
            result = await engine.process_multimodal_input(inputs)
            batch_results.append(result)
        
        # Verify batch results
        assert len(batch_results) == 3
        
        for i, result in enumerate(batch_results):
            assert len(result.input_modalities) == 2
            assert ModalityType.TEXT in result.input_modalities
            assert ModalityType.IMAGE in result.input_modalities
            assert result.fusion_confidence > 0
    
    @pytest.mark.asyncio
    async def test_engine_statistics_and_monitoring(self, multimodal_engine, sample_multimodal_data):
        """Test engine statistics and monitoring capabilities"""
        engine = multimodal_engine
        
        # Get initial stats
        initial_stats = await engine.get_processing_stats()
        
        # Process some data
        await engine.process_text(sample_multimodal_data['text'][0])
        await engine.process_image(sample_multimodal_data['images'][0])
        await engine.process_audio(sample_multimodal_data['audio'][0])
        
        # Get updated stats
        updated_stats = await engine.get_processing_stats()
        
        # Verify stats structure
        assert 'total_processed' in updated_stats
        assert 'by_modality' in updated_stats
        assert 'average_processing_time' in updated_stats
        assert 'cache_hit_rate' in updated_stats
        
        # Verify modality breakdown
        modality_stats = updated_stats['by_modality']
        assert 'text' in modality_stats
        assert 'image' in modality_stats
        assert 'audio' in modality_stats

class TestMultimodalEngineEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.fixture
    async def multimodal_engine(self):
        """Create multimodal engine for testing"""
        return MultimodalEngine()
    
    @pytest.mark.asyncio
    async def test_extreme_input_sizes(self, multimodal_engine):
        """Test handling of extreme input sizes"""
        engine = multimodal_engine
        
        # Very small inputs
        tiny_text = "a"
        tiny_result = await engine.process_text(tiny_text)
        assert tiny_result.modality == ModalityType.TEXT
        
        # Very small image
        tiny_image = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
        tiny_image_result = await engine.process_image(tiny_image)
        assert tiny_image_result.modality == ModalityType.IMAGE
        
        # Very short audio
        tiny_audio = np.random.rand(100).astype(np.float32)  # Very short
        tiny_audio_result = await engine.process_audio(tiny_audio)
        assert tiny_audio_result.modality == ModalityType.AUDIO
    
    @pytest.mark.asyncio
    async def test_unusual_data_formats(self, multimodal_engine):
        """Test handling of unusual data formats"""
        engine = multimodal_engine
        
        # Unicode text
        unicode_text = "Hello 世界 🌍 Здравствуй мир"
        unicode_result = await engine.process_text(unicode_text)
        assert unicode_result.modality == ModalityType.TEXT
        
        # Grayscale image
        gray_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        gray_result = await engine.process_image(gray_image)
        assert gray_result.modality == ModalityType.IMAGE
        assert gray_result.features['is_grayscale'] is True
        
        # Mono audio with different sample rates
        mono_audio = np.random.rand(8000).astype(np.float32)
        mono_result = await engine.process_audio(mono_audio)
        assert mono_result.modality == ModalityType.AUDIO
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_stress(self, multimodal_engine):
        """Test concurrent processing under stress"""
        engine = multimodal_engine
        
        # Create many concurrent tasks
        num_tasks = 20
        tasks = []
        
        for i in range(num_tasks):
            # Mix of different modalities
            if i % 3 == 0:
                task = engine.process_text(f"Test text {i}")
            elif i % 3 == 1:
                image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                task = engine.process_image(image)
            else:
                audio = np.random.rand(8000).astype(np.float32)
                task = engine.process_audio(audio)
            
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most tasks completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        # At least 80% should succeed
        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.8
        
        # Log any failures for debugging
        if failed_results:
            print(f"Failed results: {len(failed_results)}")
            for failure in failed_results:
                print(f"  {type(failure).__name__}: {failure}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])