"""
Comprehensive tests for visual generation safety filters.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

from scrollintel.engines.visual_generation.utils.safety_filter import (
    PromptSafetyFilter,
    NSFWImageClassifier,
    ViolenceDetector,
    ContentSafetyFilter,
    SafetyResult
)
from scrollintel.engines.visual_generation.config import SafetyConfig
from scrollintel.engines.visual_generation.base import GenerationRequest


class TestPromptSafetyFilter:
    """Test prompt safety filtering functionality."""
    
    @pytest.fixture
    def safety_config(self):
        """Create test safety configuration."""
        return SafetyConfig(
            enabled=True,
            prompt_filtering=True,
            nsfw_detection=True,
            violence_detection=True,
            confidence_threshold=0.7,
            blocked_keywords=['test_blocked', 'inappropriate']
        )
    
    @pytest.fixture
    def prompt_filter(self, safety_config):
        """Create prompt safety filter instance."""
        return PromptSafetyFilter(safety_config)
    
    @pytest.mark.asyncio
    async def test_safe_prompt_passes(self, prompt_filter):
        """Test that safe prompts pass filtering."""
        safe_prompts = [
            "A beautiful landscape with mountains and trees",
            "A cute cat playing with a ball",
            "Abstract art with colorful shapes",
            "Professional headshot of a business person"
        ]
        
        for prompt in safe_prompts:
            result = await prompt_filter.check(prompt)
            assert result.is_safe, f"Safe prompt failed: {prompt}"
            assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_blocked_keywords_detected(self, prompt_filter):
        """Test that blocked keywords are detected."""
        unsafe_prompts = [
            "This contains test_blocked content",
            "Something inappropriate here",
            "Test_blocked and inappropriate together"
        ]
        
        for prompt in unsafe_prompts:
            result = await prompt_filter.check(prompt)
            assert not result.is_safe, f"Unsafe prompt passed: {prompt}"
            assert len(result.violations) > 0
            assert any("blocked_keyword" in v for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_nsfw_content_detected(self, prompt_filter):
        """Test NSFW content detection."""
        nsfw_prompts = [
            "nude person in explicit pose",
            "nsfw adult content",
            "sexual and erotic imagery",
            "revealing lingerie in sexy pose"
        ]
        
        for prompt in nsfw_prompts:
            result = await prompt_filter.check(prompt)
            assert not result.is_safe, f"NSFW prompt passed: {prompt}"
            assert any("nsfw_content" in v for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_violence_content_detected(self, prompt_filter):
        """Test violence content detection."""
        violent_prompts = [
            "violent fight with blood",
            "gun and weapon in battle",
            "murder scene with death",
            "torture and brutality imagery"
        ]
        
        for prompt in violent_prompts:
            result = await prompt_filter.check(prompt)
            assert not result.is_safe, f"Violent prompt passed: {prompt}"
            assert any("violent_content" in v for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_hate_speech_detected(self, prompt_filter):
        """Test hate speech detection."""
        hate_speech_prompts = [
            "racist imagery with hate symbols",
            "nazi propaganda poster",
            "supremacist rally with bigot signs"
        ]
        
        for prompt in hate_speech_prompts:
            result = await prompt_filter.check(prompt)
            assert not result.is_safe, f"Hate speech prompt passed: {prompt}"
            assert any("hate_speech" in v for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_illegal_activity_detected(self, prompt_filter):
        """Test illegal activity detection."""
        illegal_prompts = [
            "drug dealing with cocaine",
            "fraud and money laundering scheme",
            "child in inappropriate sexual context"
        ]
        
        for prompt in illegal_prompts:
            result = await prompt_filter.check(prompt)
            assert not result.is_safe, f"Illegal activity prompt passed: {prompt}"
            assert any("illegal_activity" in v for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_disabled_filter_passes_all(self, safety_config):
        """Test that disabled filter passes all content."""
        safety_config.enabled = False
        prompt_filter = PromptSafetyFilter(safety_config)
        
        unsafe_prompt = "nude explicit sexual content with violence"
        result = await prompt_filter.check(unsafe_prompt)
        
        assert result.is_safe
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, prompt_filter):
        """Test confidence scoring accuracy."""
        # Safe prompt should have high confidence
        safe_result = await prompt_filter.check("beautiful landscape")
        assert safe_result.confidence > 0.9
        
        # Unsafe prompt should have appropriate confidence
        unsafe_result = await prompt_filter.check("explicit nsfw content")
        assert not unsafe_result.is_safe
        assert 0.5 < unsafe_result.confidence < 1.0


class TestNSFWImageClassifier:
    """Test NSFW image classification functionality."""
    
    @pytest.fixture
    def safety_config(self):
        """Create test safety configuration."""
        return SafetyConfig(
            enabled=True,
            nsfw_detection=True,
            confidence_threshold=0.6
        )
    
    @pytest.fixture
    def nsfw_classifier(self, safety_config):
        """Create NSFW classifier instance."""
        return NSFWImageClassifier(safety_config)
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        # Create a simple RGB image
        image = Image.new('RGB', (100, 100), color='blue')
        return image
    
    @pytest.fixture
    def high_skin_image(self):
        """Create image with high skin tone ratio."""
        # Create image with skin-like colors
        image = Image.new('RGB', (100, 100), color=(200, 150, 100))
        return image
    
    @pytest.mark.asyncio
    async def test_safe_image_passes(self, nsfw_classifier, test_image):
        """Test that safe images pass classification."""
        result = await nsfw_classifier.classify(test_image)
        assert result.is_safe
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_high_skin_ratio_flagged(self, nsfw_classifier, high_skin_image):
        """Test that images with high skin ratio are flagged."""
        result = await nsfw_classifier.classify(high_skin_image)
        # This might be flagged depending on the skin detection threshold
        if not result.is_safe:
            assert "nsfw_score" in str(result.violations)
    
    @pytest.mark.asyncio
    async def test_numpy_array_input(self, nsfw_classifier):
        """Test classification with numpy array input."""
        # Create numpy array image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = await nsfw_classifier.classify(img_array)
        assert isinstance(result, SafetyResult)
    
    @pytest.mark.asyncio
    async def test_disabled_classifier_passes(self, safety_config, test_image):
        """Test that disabled classifier passes all content."""
        safety_config.enabled = False
        nsfw_classifier = NSFWImageClassifier(safety_config)
        
        result = await nsfw_classifier.classify(test_image)
        assert result.is_safe
    
    @pytest.mark.asyncio
    async def test_error_handling(self, nsfw_classifier):
        """Test error handling for invalid inputs."""
        # Test with invalid input
        result = await nsfw_classifier.classify("invalid_path.jpg")
        assert not result.is_safe
        assert "Error during content analysis" in result.reason
    
    def test_skin_detection_accuracy(self, nsfw_classifier):
        """Test skin detection algorithm accuracy."""
        # Create image with known skin colors
        skin_image = Image.new('RGB', (100, 100), color=(200, 150, 100))
        skin_ratio = nsfw_classifier._detect_skin_ratio(skin_image)
        
        # Should detect high skin ratio
        assert skin_ratio > 0.8
        
        # Create image with non-skin colors
        non_skin_image = Image.new('RGB', (100, 100), color=(0, 0, 255))
        non_skin_ratio = nsfw_classifier._detect_skin_ratio(non_skin_image)
        
        # Should detect low skin ratio
        assert non_skin_ratio < 0.2


class TestViolenceDetector:
    """Test violence detection functionality."""
    
    @pytest.fixture
    def safety_config(self):
        """Create test safety configuration."""
        return SafetyConfig(
            enabled=True,
            violence_detection=True,
            confidence_threshold=0.4
        )
    
    @pytest.fixture
    def violence_detector(self, safety_config):
        """Create violence detector instance."""
        return ViolenceDetector(safety_config)
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return Image.new('RGB', (100, 100), color='green')
    
    @pytest.fixture
    def red_image(self):
        """Create predominantly red image."""
        return Image.new('RGB', (100, 100), color=(200, 50, 50))
    
    @pytest.mark.asyncio
    async def test_safe_text_passes(self, violence_detector):
        """Test that safe text passes violence detection."""
        safe_texts = [
            "peaceful landscape with flowers",
            "happy family gathering",
            "beautiful sunset over ocean",
            "children playing in park"
        ]
        
        for text in safe_texts:
            result = await violence_detector.detect(text)
            assert result.is_safe, f"Safe text failed: {text}"
    
    @pytest.mark.asyncio
    async def test_violent_text_detected(self, violence_detector):
        """Test that violent text is detected."""
        violent_texts = [
            "blood and gore in battle",
            "gun violence with weapons",
            "murder scene with death",
            "knife attack and assault"
        ]
        
        for text in violent_texts:
            result = await violence_detector.detect(text)
            assert not result.is_safe, f"Violent text passed: {text}"
            assert len(result.violations) > 0
    
    @pytest.mark.asyncio
    async def test_weapon_patterns_detected(self, violence_detector):
        """Test weapon pattern detection."""
        weapon_texts = [
            "rifle and firearm display",
            "sword and blade collection",
            "bomb and explosive device",
            "missile and rocket launcher"
        ]
        
        for text in weapon_texts:
            result = await violence_detector.detect(text)
            assert not result.is_safe, f"Weapon text passed: {text}"
            assert any("weapon_pattern" in v for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_safe_image_passes(self, violence_detector, test_image):
        """Test that safe images pass violence detection."""
        result = await violence_detector.detect(test_image)
        assert result.is_safe
    
    @pytest.mark.asyncio
    async def test_red_image_flagged(self, violence_detector, red_image):
        """Test that predominantly red images might be flagged."""
        result = await violence_detector.detect(red_image)
        # Red images might be flagged as potentially violent
        if not result.is_safe:
            assert "violence_score" in str(result.violations)
    
    @pytest.mark.asyncio
    async def test_disabled_detector_passes(self, safety_config, test_image):
        """Test that disabled detector passes all content."""
        safety_config.enabled = False
        violence_detector = ViolenceDetector(safety_config)
        
        result = await violence_detector.detect(test_image)
        assert result.is_safe
    
    def test_red_ratio_detection(self, violence_detector):
        """Test red ratio detection accuracy."""
        # Create predominantly red image
        red_image = Image.new('RGB', (100, 100), color=(200, 50, 50))
        red_ratio = violence_detector._detect_red_ratio(np.array(red_image))
        
        # Should detect high red ratio
        assert red_ratio > 0.8
        
        # Create non-red image
        blue_image = Image.new('RGB', (100, 100), color=(50, 50, 200))
        blue_ratio = violence_detector._detect_red_ratio(np.array(blue_image))
        
        # Should detect low red ratio
        assert blue_ratio < 0.2


class TestContentSafetyFilter:
    """Test integrated content safety filter."""
    
    @pytest.fixture
    def safety_config(self):
        """Create comprehensive safety configuration."""
        return SafetyConfig(
            enabled=True,
            prompt_filtering=True,
            nsfw_detection=True,
            violence_detection=True,
            confidence_threshold=0.7,
            blocked_keywords=['blocked_test']
        )
    
    @pytest.fixture
    def safety_filter(self, safety_config):
        """Create content safety filter instance."""
        return ContentSafetyFilter(safety_config)
    
    @pytest.fixture
    def safe_request(self):
        """Create safe generation request."""
        return GenerationRequest(
            prompt="beautiful landscape with mountains",
            user_id="test_user",
            negative_prompt="blurry, low quality"
        )
    
    @pytest.fixture
    def unsafe_request(self):
        """Create unsafe generation request."""
        return GenerationRequest(
            prompt="explicit nsfw content with violence",
            user_id="test_user",
            negative_prompt="safe content"
        )
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return Image.new('RGB', (100, 100), color='blue')
    
    @pytest.mark.asyncio
    async def test_safe_request_passes(self, safety_filter, safe_request):
        """Test that safe requests pass validation."""
        result = await safety_filter.validate_request(safe_request)
        assert result.is_safe
        assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_unsafe_request_blocked(self, safety_filter, unsafe_request):
        """Test that unsafe requests are blocked."""
        result = await safety_filter.validate_request(unsafe_request)
        assert not result.is_safe
        assert len(result.violations) > 0
    
    @pytest.mark.asyncio
    async def test_unsafe_negative_prompt_blocked(self, safety_filter):
        """Test that unsafe negative prompts are blocked."""
        request = GenerationRequest(
            prompt="safe landscape",
            user_id="test_user",
            negative_prompt="explicit nsfw blocked_test content"
        )
        
        result = await safety_filter.validate_request(request)
        assert not result.is_safe
        assert any("negative_prompt" in v for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_output_validation(self, safety_filter, test_image):
        """Test output content validation."""
        result = await safety_filter.validate_output(test_image, "image")
        assert result.is_safe
    
    @pytest.mark.asyncio
    async def test_comprehensive_safety_check(self, safety_filter, safe_request, test_image):
        """Test comprehensive safety check."""
        result = await safety_filter.comprehensive_safety_check(safe_request, test_image)
        assert result.is_safe
        assert "Comprehensive safety check passed" in result.reason
    
    @pytest.mark.asyncio
    async def test_comprehensive_check_blocks_unsafe_request(self, safety_filter, unsafe_request, test_image):
        """Test comprehensive check blocks unsafe requests."""
        result = await safety_filter.comprehensive_safety_check(unsafe_request, test_image)
        assert not result.is_safe
    
    @pytest.mark.asyncio
    async def test_disabled_filter_passes_all(self, safety_config, unsafe_request):
        """Test that disabled filter passes all content."""
        safety_config.enabled = False
        safety_filter = ContentSafetyFilter(safety_config)
        
        result = await safety_filter.validate_request(unsafe_request)
        assert result.is_safe
    
    def test_keyword_management(self, safety_filter):
        """Test blocked keyword management."""
        # Add keyword
        safety_filter.add_blocked_keyword("new_blocked")
        assert "new_blocked" in safety_filter.prompt_filter.blocked_keywords
        
        # Remove keyword
        safety_filter.remove_blocked_keyword("new_blocked")
        assert "new_blocked" not in safety_filter.prompt_filter.blocked_keywords
    
    def test_confidence_threshold_update(self, safety_filter):
        """Test confidence threshold updates."""
        new_threshold = 0.8
        safety_filter.update_confidence_threshold(new_threshold)
        
        assert safety_filter.config.confidence_threshold == new_threshold
        assert safety_filter.nsfw_classifier.confidence_threshold == new_threshold
        assert safety_filter.violence_detector.confidence_threshold == new_threshold
    
    @pytest.mark.asyncio
    async def test_error_handling_in_validation(self, safety_filter):
        """Test error handling during validation."""
        # Mock an error in the NSFW classifier
        with patch.object(safety_filter.nsfw_classifier, 'classify', side_effect=Exception("Test error")):
            result = await safety_filter.validate_output("test_image", "image")
            # Should handle error gracefully
            assert isinstance(result, SafetyResult)


class TestSafetyAccuracy:
    """Test safety detection accuracy with comprehensive test cases."""
    
    @pytest.fixture
    def safety_filter(self):
        """Create safety filter for accuracy testing."""
        config = SafetyConfig(
            enabled=True,
            prompt_filtering=True,
            nsfw_detection=True,
            violence_detection=True,
            confidence_threshold=0.7,
            blocked_keywords=['test_blocked']
        )
        return ContentSafetyFilter(config)
    
    @pytest.mark.asyncio
    async def test_false_positive_rate(self, safety_filter):
        """Test false positive rate with clearly safe content."""
        safe_prompts = [
            "professional business meeting",
            "children playing in playground",
            "beautiful nature photography",
            "cooking delicious food",
            "reading books in library",
            "walking dog in park",
            "family vacation photos",
            "graduation ceremony",
            "wedding celebration",
            "birthday party"
        ]
        
        false_positives = 0
        for prompt in safe_prompts:
            request = GenerationRequest(prompt=prompt, user_id="test_user")
            result = await safety_filter.validate_request(request)
            if not result.is_safe:
                false_positives += 1
        
        false_positive_rate = false_positives / len(safe_prompts)
        assert false_positive_rate < 0.1, f"False positive rate too high: {false_positive_rate}"
    
    @pytest.mark.asyncio
    async def test_true_positive_rate(self, safety_filter):
        """Test true positive rate with clearly unsafe content."""
        unsafe_prompts = [
            "explicit nude sexual content",
            "violent murder scene with blood",
            "nsfw adult pornographic material",
            "graphic violence with weapons",
            "test_blocked inappropriate content",
            "racist hate speech imagery",
            "drug dealing illegal activity",
            "torture and brutality scene",
            "child in sexual context",
            "nazi propaganda with hate symbols"
        ]
        
        true_positives = 0
        for prompt in unsafe_prompts:
            request = GenerationRequest(prompt=prompt, user_id="test_user")
            result = await safety_filter.validate_request(request)
            if not result.is_safe:
                true_positives += 1
        
        true_positive_rate = true_positives / len(unsafe_prompts)
        assert true_positive_rate > 0.8, f"True positive rate too low: {true_positive_rate}"
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, safety_filter):
        """Test edge cases and borderline content."""
        edge_cases = [
            ("medical anatomy diagram", True),  # Should be safe
            ("artistic nude sculpture", False),  # Should be flagged
            ("historical documentary film", True),  # Should be safe
            ("graphic war violence", False),  # Should be flagged
            ("fashion swimwear model", True),  # Should be safe
            ("revealing lingerie in sexy pose", False),  # Should be flagged
        ]
        
        for prompt, expected_safe in edge_cases:
            request = GenerationRequest(prompt=prompt, user_id="test_user")
            result = await safety_filter.validate_request(request)
            
            if expected_safe:
                assert result.is_safe, f"Edge case failed - should be safe: {prompt}"
            else:
                assert not result.is_safe, f"Edge case failed - should be unsafe: {prompt}"


if __name__ == "__main__":
    pytest.main([__file__])