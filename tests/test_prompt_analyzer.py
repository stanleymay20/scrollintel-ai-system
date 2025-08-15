"""
Tests for the prompt analysis engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from scrollintel.engines.visual_generation.utils.prompt_analyzer import (
    PromptAnalyzer,
    PromptAnalysisResult,
    PromptComplexity,
    ContentType,
    ArtisticStyle,
    TechnicalParameters,
    PromptQualityMetrics
)
from scrollintel.engines.visual_generation.exceptions import PromptAnalysisError


class TestPromptAnalyzer:
    """Test cases for PromptAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a PromptAnalyzer instance for testing."""
        return PromptAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_simple_prompt(self, analyzer):
        """Test analysis of a simple prompt."""
        prompt = "a cat"
        result = await analyzer.analyze_prompt(prompt)
        
        assert isinstance(result, PromptAnalysisResult)
        assert result.original_prompt == prompt
        assert result.word_count == 2
        assert result.complexity == PromptComplexity.SIMPLE
        assert result.confidence > 0
        assert len(result.improvement_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_complex_prompt(self, analyzer):
        """Test analysis of a complex prompt."""
        prompt = ("photorealistic portrait of a young woman with blue eyes, "
                 "natural lighting, 8k resolution, professional photography, "
                 "shallow depth of field, award winning, masterpiece")
        
        result = await analyzer.analyze_prompt(prompt)
        
        assert result.complexity in [PromptComplexity.COMPLEX, PromptComplexity.VERY_COMPLEX]
        assert result.content_type == ContentType.PORTRAIT
        assert result.artistic_style == ArtisticStyle.PHOTOREALISTIC
        assert result.technical_parameters.resolution == "8k"
        assert "natural lighting" in result.technical_parameters.lighting
        assert result.quality_metrics.overall_score > 0.5
    
    @pytest.mark.asyncio
    async def test_detect_content_types(self, analyzer):
        """Test detection of different content types."""
        test_cases = [
            ("portrait of a man", ContentType.PORTRAIT),
            ("mountain landscape at sunset", ContentType.LANDSCAPE),
            ("modern building architecture", ContentType.ARCHITECTURE),
            ("abstract geometric pattern", ContentType.ABSTRACT),
            ("red sports car", ContentType.OBJECT)
        ]
        
        for prompt, expected_type in test_cases:
            result = await analyzer.analyze_prompt(prompt)
            assert result.content_type == expected_type
    
    @pytest.mark.asyncio
    async def test_detect_artistic_styles(self, analyzer):
        """Test detection of different artistic styles."""
        test_cases = [
            ("photorealistic image of a dog", ArtisticStyle.PHOTOREALISTIC),
            ("oil painting of flowers", ArtisticStyle.ARTISTIC),
            ("cartoon character design", ArtisticStyle.CARTOON),
            ("pencil sketch of a tree", ArtisticStyle.SKETCH),
            ("cinematic movie scene", ArtisticStyle.CINEMATIC),
            ("digital art concept", ArtisticStyle.DIGITAL_ART)
        ]
        
        for prompt, expected_style in test_cases:
            result = await analyzer.analyze_prompt(prompt)
            assert result.artistic_style == expected_style
    
    @pytest.mark.asyncio
    async def test_extract_technical_parameters(self, analyzer):
        """Test extraction of technical parameters."""
        prompt = ("4k resolution portrait with natural lighting, "
                 "shot with wide angle lens, shallow depth of field, "
                 "high quality, professional photography")
        
        result = await analyzer.analyze_prompt(prompt)
        params = result.technical_parameters
        
        assert params.resolution == "4k"
        assert "natural lighting" in params.lighting
        assert any("wide angle" in setting for setting in params.camera_settings)
        assert any("high quality" in modifier for modifier in params.quality_modifiers)
    
    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, analyzer):
        """Test quality metrics calculation."""
        high_quality_prompt = ("masterpiece photorealistic portrait of elegant woman, "
                              "natural lighting, 8k resolution, professional photography, "
                              "award winning, ultra detailed, sharp focus")
        
        low_quality_prompt = "person"
        
        high_result = await analyzer.analyze_prompt(high_quality_prompt)
        low_result = await analyzer.analyze_prompt(low_quality_prompt)
        
        assert high_result.quality_metrics.overall_score > low_result.quality_metrics.overall_score
        assert high_result.quality_metrics.technical_completeness > 0.5
        assert low_result.quality_metrics.improvement_potential > 0.5
    
    @pytest.mark.asyncio
    async def test_detect_subjects_objects_emotions(self, analyzer):
        """Test detection of subjects, objects, and emotions."""
        prompt = "happy woman driving red car with smile"
        result = await analyzer.analyze_prompt(prompt)
        
        assert "woman" in result.detected_subjects
        assert "car" in result.detected_objects
        assert "happy" in result.detected_emotions or "smile" in result.detected_emotions
    
    @pytest.mark.asyncio
    async def test_missing_elements_identification(self, analyzer):
        """Test identification of missing elements."""
        minimal_prompt = "cat"
        result = await analyzer.analyze_prompt(minimal_prompt)
        
        assert len(result.missing_elements) > 0
        assert any("resolution" in element for element in result.missing_elements)
        assert any("lighting" in element for element in result.missing_elements)
    
    @pytest.mark.asyncio
    async def test_improvement_suggestions(self, analyzer):
        """Test generation of improvement suggestions."""
        simple_prompt = "dog"
        result = await analyzer.analyze_prompt(simple_prompt)
        
        assert len(result.improvement_suggestions) > 0
        assert any("descriptive" in suggestion.lower() for suggestion in result.improvement_suggestions)
    
    @pytest.mark.asyncio
    async def test_complexity_detection(self, analyzer):
        """Test complexity detection for different prompt types."""
        test_cases = [
            ("cat", PromptComplexity.SIMPLE),
            ("beautiful cat in garden", PromptComplexity.SIMPLE),
            ("photorealistic cat with natural lighting, high quality", PromptComplexity.MODERATE),
            ("masterpiece photorealistic portrait of elegant cat with blue eyes, "
             "natural lighting, 8k resolution, professional photography, "
             "shallow depth of field, award winning", PromptComplexity.COMPLEX)
        ]
        
        for prompt, expected_complexity in test_cases:
            result = await analyzer.analyze_prompt(prompt)
            assert result.complexity == expected_complexity
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, analyzer):
        """Test confidence calculation."""
        detailed_prompt = ("photorealistic portrait with natural lighting, "
                          "8k resolution, professional photography")
        vague_prompt = "thing"
        
        detailed_result = await analyzer.analyze_prompt(detailed_prompt)
        vague_result = await analyzer.analyze_prompt(vague_prompt)
        
        assert detailed_result.confidence > vague_result.confidence
        assert 0 <= detailed_result.confidence <= 1
        assert 0 <= vague_result.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_empty_prompt_error(self, analyzer):
        """Test error handling for empty prompts."""
        with pytest.raises(PromptAnalysisError):
            await analyzer.analyze_prompt("")
        
        with pytest.raises(PromptAnalysisError):
            await analyzer.analyze_prompt("   ")
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, analyzer):
        """Test batch analysis of multiple prompts."""
        prompts = [
            "cat",
            "photorealistic dog",
            "abstract art painting"
        ]
        
        results = await analyzer.batch_analyze_prompts(prompts)
        
        assert len(results) == len(prompts)
        assert all(isinstance(result, PromptAnalysisResult) for result in results)
        assert results[0].original_prompt == "cat"
        assert results[1].artistic_style == ArtisticStyle.PHOTOREALISTIC
        assert results[2].artistic_style == ArtisticStyle.ARTISTIC
    
    def test_export_analysis(self, analyzer):
        """Test exporting analysis results."""
        # Create a mock result
        result = PromptAnalysisResult(
            original_prompt="test",
            word_count=1,
            complexity=PromptComplexity.SIMPLE,
            content_type=ContentType.UNKNOWN,
            artistic_style=ArtisticStyle.UNKNOWN,
            technical_parameters=TechnicalParameters(),
            quality_metrics=PromptQualityMetrics(
                overall_score=0.5,
                specificity_score=0.5,
                technical_completeness=0.5,
                style_clarity=0.5,
                structure_score=0.5,
                improvement_potential=0.5
            ),
            detected_subjects=[],
            detected_objects=[],
            detected_emotions=[],
            missing_elements=[],
            improvement_suggestions=[],
            confidence=0.7,
            analysis_timestamp="2024-01-01T00:00:00"
        )
        
        exported = analyzer.export_analysis(result)
        assert isinstance(exported, dict)
        assert exported['original_prompt'] == "test"
        assert exported['confidence'] == 0.7
    
    def test_get_analysis_summary(self, analyzer):
        """Test getting analysis summary."""
        result = PromptAnalysisResult(
            original_prompt="test prompt",
            word_count=2,
            complexity=PromptComplexity.SIMPLE,
            content_type=ContentType.PORTRAIT,
            artistic_style=ArtisticStyle.PHOTOREALISTIC,
            technical_parameters=TechnicalParameters(),
            quality_metrics=PromptQualityMetrics(
                overall_score=0.75,
                specificity_score=0.5,
                technical_completeness=0.5,
                style_clarity=0.5,
                structure_score=0.5,
                improvement_potential=0.5
            ),
            detected_subjects=[],
            detected_objects=[],
            detected_emotions=[],
            missing_elements=[],
            improvement_suggestions=["Add more details", "Specify lighting"],
            confidence=0.8,
            analysis_timestamp="2024-01-01T00:00:00"
        )
        
        summary = analyzer.get_analysis_summary(result)
        assert "Prompt Analysis Summary" in summary
        assert "simple" in summary.lower()
        assert "portrait" in summary.lower()
        assert "0.75" in summary
        assert "Add more details" in summary
    
    @pytest.mark.asyncio
    async def test_portrait_specific_analysis(self, analyzer):
        """Test portrait-specific analysis features."""
        prompt = "portrait of young woman with brown hair"
        result = await analyzer.analyze_prompt(prompt)
        
        assert result.content_type == ContentType.PORTRAIT
        assert "woman" in result.detected_subjects
        # Should suggest emotional expression for portraits
        assert any("emotion" in suggestion.lower() for suggestion in result.improvement_suggestions)
    
    @pytest.mark.asyncio
    async def test_landscape_specific_analysis(self, analyzer):
        """Test landscape-specific analysis features."""
        prompt = "mountain landscape"
        result = await analyzer.analyze_prompt(prompt)
        
        assert result.content_type == ContentType.LANDSCAPE
        # Should suggest time of day for landscapes
        assert any("time" in element.lower() for element in result.missing_elements)
    
    @pytest.mark.asyncio
    async def test_technical_parameter_extraction_edge_cases(self, analyzer):
        """Test technical parameter extraction with edge cases."""
        prompt = "f/2.8 aperture, ISO 100, 85mm lens, 16:9 aspect ratio"
        result = await analyzer.analyze_prompt(prompt)
        
        params = result.technical_parameters
        assert params.aspect_ratio == "16:9"
        assert any("f/2.8" in setting for setting in params.camera_settings)
    
    @pytest.mark.asyncio
    async def test_style_pattern_matching(self, analyzer):
        """Test style pattern matching accuracy."""
        test_cases = [
            ("in the style of Van Gogh", ArtisticStyle.ARTISTIC),
            ("trending on artstation", ArtisticStyle.DIGITAL_ART),
            ("hand drawn illustration", ArtisticStyle.SKETCH),
            ("movie poster style", ArtisticStyle.CINEMATIC)
        ]
        
        for prompt, expected_style in test_cases:
            result = await analyzer.analyze_prompt(prompt)
            assert result.artistic_style == expected_style
    
    @pytest.mark.asyncio
    async def test_quality_modifier_detection(self, analyzer):
        """Test detection of quality modifiers."""
        prompt = "masterpiece, award winning, ultra detailed, best quality"
        result = await analyzer.analyze_prompt(prompt)
        
        params = result.technical_parameters
        quality_modifiers = params.quality_modifiers
        
        assert len(quality_modifiers) > 0
        assert any("masterpiece" in modifier for modifier in quality_modifiers)
        assert any("award winning" in modifier for modifier in quality_modifiers)
    
    @pytest.mark.asyncio
    async def test_analysis_consistency(self, analyzer):
        """Test that analysis results are consistent across multiple runs."""
        prompt = "photorealistic portrait of a woman"
        
        results = []
        for _ in range(3):
            result = await analyzer.analyze_prompt(prompt)
            results.append(result)
        
        # All results should have same basic properties
        for result in results:
            assert result.content_type == ContentType.PORTRAIT
            assert result.artistic_style == ArtisticStyle.PHOTOREALISTIC
            assert result.word_count == 6
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_input(self, analyzer):
        """Test error handling for invalid inputs."""
        with pytest.raises(PromptAnalysisError):
            await analyzer.analyze_prompt(None)
    
    def test_technical_parameters_initialization(self):
        """Test TechnicalParameters initialization."""
        params = TechnicalParameters()
        assert params.lighting == []
        assert params.camera_settings == []
        assert params.rendering_style == []
        assert params.quality_modifiers == []
        assert params.resolution is None
        assert params.aspect_ratio is None
    
    @pytest.mark.asyncio
    async def test_unknown_content_and_style_detection(self, analyzer):
        """Test handling of unknown content types and styles."""
        prompt = "xyzabc nonsense words"
        result = await analyzer.analyze_prompt(prompt)
        
        # Should default to unknown when no patterns match
        assert result.content_type == ContentType.UNKNOWN
        assert result.artistic_style == ArtisticStyle.UNKNOWN
        assert len(result.improvement_suggestions) > 0


class TestPromptAnalysisIntegration:
    """Integration tests for prompt analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return PromptAnalyzer()
    
    @pytest.mark.asyncio
    async def test_real_world_prompts(self, analyzer):
        """Test analysis with real-world prompt examples."""
        real_prompts = [
            "A serene lake surrounded by mountains at golden hour, "
            "photorealistic, 8k, professional photography",
            
            "Cartoon character of a friendly robot, colorful, "
            "Disney style, 3D render",
            
            "Abstract geometric pattern in blue and gold, "
            "minimalist design, high contrast",
            
            "Portrait of an elderly man with wise eyes, "
            "black and white photography, dramatic lighting"
        ]
        
        for prompt in real_prompts:
            result = await analyzer.analyze_prompt(prompt)
            
            # Basic validation
            assert result.confidence > 0.3
            assert len(result.improvement_suggestions) >= 0
            assert result.quality_metrics.overall_score > 0
            
            # Should detect appropriate content types
            if "portrait" in prompt.lower():
                assert result.content_type == ContentType.PORTRAIT
            elif "abstract" in prompt.lower():
                assert result.content_type == ContentType.ABSTRACT
    
    @pytest.mark.asyncio
    async def test_performance_batch_analysis(self, analyzer):
        """Test performance of batch analysis."""
        import time
        
        prompts = ["test prompt"] * 10
        
        start_time = time.time()
        results = await analyzer.batch_analyze_prompts(prompts)
        end_time = time.time()
        
        assert len(results) == 10
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_analysis_accuracy_consistency(self, analyzer):
        """Test accuracy and consistency of analysis results."""
        # Test with prompts that have clear expected outcomes
        test_cases = [
            {
                'prompt': "photorealistic portrait of a young woman with blue eyes",
                'expected_content': ContentType.PORTRAIT,
                'expected_style': ArtisticStyle.PHOTOREALISTIC,
                'expected_subjects': ['woman']
            },
            {
                'prompt': "cartoon drawing of a red car",
                'expected_content': ContentType.OBJECT,
                'expected_style': ArtisticStyle.CARTOON,
                'expected_objects': ['car']
            }
        ]
        
        for case in test_cases:
            result = await analyzer.analyze_prompt(case['prompt'])
            
            assert result.content_type == case['expected_content']
            assert result.artistic_style == case['expected_style']
            
            if 'expected_subjects' in case:
                assert any(subject in result.detected_subjects 
                          for subject in case['expected_subjects'])
            
            if 'expected_objects' in case:
                assert any(obj in result.detected_objects 
                          for obj in case['expected_objects'])


if __name__ == "__main__":
    pytest.main([__file__])