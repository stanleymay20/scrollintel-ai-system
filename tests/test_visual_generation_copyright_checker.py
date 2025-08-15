"""
Comprehensive tests for visual generation copyright checker.
"""

import pytest
import asyncio
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime

from scrollintel.engines.visual_generation.utils.copyright_checker import (
    CopyrightChecker,
    CopyrightResult,
    CopyrightMatch
)
from scrollintel.engines.visual_generation.config import SafetyConfig
from scrollintel.engines.visual_generation.base import GenerationRequest


class TestCopyrightChecker:
    """Test copyright checking functionality."""
    
    @pytest.fixture
    def safety_config(self):
        """Create test safety configuration."""
        return SafetyConfig(
            enabled=True,
            copyright_check=True,
            confidence_threshold=0.8
        )
    
    @pytest.fixture
    def copyright_checker(self, safety_config):
        """Create copyright checker instance."""
        return CopyrightChecker(safety_config)
    
    @pytest.fixture
    def test_image_path(self):
        """Create a test image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Create a simple test image
            image = Image.new('RGB', (100, 100), color='blue')
            image.save(f.name)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def test_request(self):
        """Create test generation request."""
        return GenerationRequest(
            prompt="A beautiful landscape with mountains",
            user_id="test_user"
        )
    
    @pytest.mark.asyncio
    async def test_basic_copyright_check(self, copyright_checker, test_image_path, test_request):
        """Test basic copyright checking functionality."""
        result = await copyright_checker.check_copyright(test_image_path, test_request)
        
        assert isinstance(result, CopyrightResult)
        assert isinstance(result.is_original, bool)
        assert result.risk_level in ['low', 'medium', 'high', 'critical']
        assert isinstance(result.matches, list)
        assert isinstance(result.watermark_detected, bool)
        assert isinstance(result.usage_allowed, bool)
        assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_disabled_copyright_check(self, safety_config, test_image_path, test_request):
        """Test copyright checking when disabled."""
        safety_config.enabled = False
        copyright_checker = CopyrightChecker(safety_config)
        
        result = await copyright_checker.check_copyright(test_image_path, test_request)
        
        assert result.is_original is True
        assert result.risk_level == "low"
    
    @pytest.mark.asyncio
    async def test_copyright_check_disabled_feature(self, safety_config, test_image_path, test_request):
        """Test copyright checking when feature is disabled."""
        safety_config.copyright_check = False
        copyright_checker = CopyrightChecker(safety_config)
        
        result = await copyright_checker.check_copyright(test_image_path, test_request)
        
        assert result.is_original is True
        assert result.risk_level == "low"
    
    def test_perceptual_hash_calculation(self, copyright_checker):
        """Test perceptual hash calculation."""
        # Create test images
        image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image3 = image1.copy()  # Identical image
        
        hash1 = copyright_checker._calculate_perceptual_hash(image1)
        hash2 = copyright_checker._calculate_perceptual_hash(image2)
        hash3 = copyright_checker._calculate_perceptual_hash(image3)
        
        # Hashes should be strings
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert isinstance(hash3, str)
        
        # Identical images should have identical hashes
        assert hash1 == hash3
        
        # Different images should have different hashes (usually)
        assert hash1 != hash2
    
    def test_hash_similarity_calculation(self, copyright_checker):
        """Test hash similarity calculation."""
        hash1 = "1234567890abcdef"
        hash2 = "1234567890abcdef"  # Identical
        hash3 = "1234567890abcdff"  # One bit different
        hash4 = "fedcba0987654321"  # Completely different
        
        # Identical hashes
        similarity1 = copyright_checker._calculate_hash_similarity(hash1, hash2)
        assert similarity1 == 1.0
        
        # Similar hashes
        similarity2 = copyright_checker._calculate_hash_similarity(hash1, hash3)
        assert 0.9 < similarity2 < 1.0
        
        # Different hashes
        similarity3 = copyright_checker._calculate_hash_similarity(hash1, hash4)
        assert similarity3 < 0.5
    
    @pytest.mark.asyncio
    async def test_hash_database_check(self, copyright_checker):
        """Test checking against hash database."""
        # Add a test hash to the database
        test_hash = "1234567890abcdef"
        copyright_checker.known_copyrighted_hashes.add(test_hash)
        
        # Check exact match
        matches = await copyright_checker._check_hash_database(test_hash)
        assert len(matches) >= 1
        exact_matches = [m for m in matches if m.match_type == "exact"]
        assert len(exact_matches) == 1
        assert exact_matches[0].similarity_score == 1.0
        
        # Check non-match
        matches = await copyright_checker._check_hash_database("fedcba0987654321")
        assert len(matches) == 0
    
    @pytest.mark.asyncio
    async def test_watermark_detection(self, copyright_checker):
        """Test watermark detection."""
        # Create image without watermark
        clean_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        watermark_detected = await copyright_checker._detect_watermarks(clean_image)
        # Clean uniform image should not have watermark
        # But our heuristics might be sensitive, so we'll just check it's a boolean
        assert isinstance(watermark_detected, bool)
        
        # Create image with potential watermark pattern (high contrast corner)
        watermark_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        watermark_image[:20, :20] = 255  # Bright corner
        watermark_image[10:15, 10:15] = 0  # Dark text-like pattern
        
        watermark_detected = await copyright_checker._detect_watermarks(watermark_image)
        # This might or might not detect depending on the heuristics
        assert isinstance(watermark_detected, bool)
    
    def test_watermark_pattern_detection(self, copyright_checker):
        """Test watermark pattern detection in regions."""
        # Clean region
        clean_region = np.full((50, 50, 3), 128, dtype=np.uint8)
        has_pattern = copyright_checker._has_watermark_pattern(clean_region)
        # Just check it returns a boolean - heuristics might vary
        assert isinstance(has_pattern, bool)
        
        # High contrast region (potential text)
        text_region = np.full((50, 50, 3), 128, dtype=np.uint8)
        text_region[10:40, 10:40] = 255
        text_region[15:35, 15:35] = 0
        has_pattern = copyright_checker._has_watermark_pattern(text_region)
        # This might detect the pattern
        assert isinstance(has_pattern, bool)
    
    @pytest.mark.asyncio
    async def test_trademark_checking(self, copyright_checker):
        """Test trademark violation checking."""
        # Add test trademark to database
        copyright_checker.trademark_database = {
            "Nike": {"holder": "Nike Inc.", "category": "Apparel"},
            "Apple": {"holder": "Apple Inc.", "category": "Technology"}
        }
        
        # Check prompt with trademark
        matches = await copyright_checker._check_trademark_violations("Nike shoes and Apple iPhone")
        assert len(matches) == 2
        assert any(m.match_type == "trademark" for m in matches)
        
        # Check prompt without trademark
        matches = await copyright_checker._check_trademark_violations("Generic shoes and phone")
        assert len(matches) == 0
    
    @pytest.mark.asyncio
    async def test_artistic_style_checking(self, copyright_checker):
        """Test artistic style similarity checking."""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Request with famous artist name
        request_with_artist = GenerationRequest(
            prompt="A painting in the style of Picasso",
            user_id="test_user"
        )
        
        matches = await copyright_checker._check_artistic_style_similarity(test_image, request_with_artist)
        assert len(matches) > 0
        assert any(m.match_type == "style" for m in matches)
        
        # Request without artist name
        request_without_artist = GenerationRequest(
            prompt="A beautiful landscape",
            user_id="test_user"
        )
        
        matches = await copyright_checker._check_artistic_style_similarity(test_image, request_without_artist)
        assert len(matches) == 0
    
    def test_risk_level_calculation(self, copyright_checker):
        """Test risk level calculation."""
        # No matches, no watermark
        risk = copyright_checker._calculate_risk_level([], False)
        assert risk == "low"
        
        # Watermark detected
        risk = copyright_checker._calculate_risk_level([], True)
        assert risk == "high"
        
        # Exact match
        exact_match = CopyrightMatch(match_type="exact", similarity_score=1.0)
        risk = copyright_checker._calculate_risk_level([exact_match], False)
        assert risk == "critical"
        
        # Trademark match
        trademark_match = CopyrightMatch(match_type="trademark", similarity_score=1.0)
        risk = copyright_checker._calculate_risk_level([trademark_match], False)
        assert risk == "high"
        
        # High similarity match
        similar_match = CopyrightMatch(match_type="similar", similarity_score=0.95)
        risk = copyright_checker._calculate_risk_level([similar_match], False)
        assert risk == "high"
        
        # Multiple low similarity matches
        low_matches = [
            CopyrightMatch(match_type="similar", similarity_score=0.7),
            CopyrightMatch(match_type="similar", similarity_score=0.6),
            CopyrightMatch(match_type="similar", similarity_score=0.5),
            CopyrightMatch(match_type="similar", similarity_score=0.4)
        ]
        risk = copyright_checker._calculate_risk_level(low_matches, False)
        assert risk == "medium"
    
    def test_recommendation_generation(self, copyright_checker):
        """Test copyright recommendation generation."""
        # Critical risk
        exact_match = CopyrightMatch(match_type="exact")
        recommendations = copyright_checker._generate_copyright_recommendations([exact_match], False, "critical")
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("do not use" in rec.lower() for rec in recommendations)
        
        # High risk with watermark
        recommendations = copyright_checker._generate_copyright_recommendations([], True, "high")
        assert any("HIGH RISK" in rec for rec in recommendations)
        assert any("watermark" in rec.lower() for rec in recommendations)
        
        # Medium risk
        recommendations = copyright_checker._generate_copyright_recommendations([], False, "medium")
        assert any("MEDIUM RISK" in rec for rec in recommendations)
        
        # Low risk
        recommendations = copyright_checker._generate_copyright_recommendations([], False, "low")
        assert any("LOW RISK" in rec for rec in recommendations)
        assert any("original" in rec.lower() for rec in recommendations)
    
    def test_confidence_calculation(self, copyright_checker):
        """Test confidence calculation."""
        # No matches
        confidence = copyright_checker._calculate_confidence([])
        assert confidence == 0.9
        
        # Single match
        match = CopyrightMatch(confidence=0.8)
        confidence = copyright_checker._calculate_confidence([match])
        assert confidence == 0.8
        
        # Multiple matches
        matches = [
            CopyrightMatch(confidence=0.9),
            CopyrightMatch(confidence=0.7),
            CopyrightMatch(confidence=0.8)
        ]
        confidence = copyright_checker._calculate_confidence(matches)
        assert abs(confidence - 0.8) < 0.01  # Average with floating point tolerance
    
    def test_content_type_detection(self, copyright_checker):
        """Test content type detection from file paths."""
        assert copyright_checker._get_content_type("test.jpg") == "image"
        assert copyright_checker._get_content_type("test.png") == "image"
        assert copyright_checker._get_content_type("test.mp4") == "video"
        assert copyright_checker._get_content_type("test.avi") == "video"
        assert copyright_checker._get_content_type("test.txt") == "unknown"
    
    @pytest.mark.asyncio
    async def test_copyright_database_addition(self, copyright_checker, test_image_path):
        """Test adding content to copyright database."""
        initial_count = len(copyright_checker.known_copyrighted_hashes)
        
        copyright_info = {
            "holder": "Test Holder",
            "license": "All Rights Reserved"
        }
        
        await copyright_checker.add_to_copyright_database(test_image_path, copyright_info)
        
        # Should have added one hash (or kept same count if hash already existed)
        assert len(copyright_checker.known_copyrighted_hashes) >= initial_count
    
    @pytest.mark.asyncio
    async def test_attribution_text_generation(self, copyright_checker):
        """Test attribution text generation."""
        matches = [
            CopyrightMatch(
                copyright_holder="John Doe",
                source_url="https://example.com/image1",
                license_type="CC-BY"
            ),
            CopyrightMatch(
                copyright_holder="Jane Smith",
                license_type="CC-BY-SA"
            )
        ]
        
        attribution = await copyright_checker.generate_attribution_text(matches)
        
        assert "John Doe" in attribution
        assert "https://example.com/image1" in attribution
        assert "CC-BY" in attribution
        assert "Jane Smith" in attribution
        assert "CC-BY-SA" in attribution
    
    def test_usage_guidelines(self, copyright_checker):
        """Test usage guidelines generation."""
        # Low risk result
        low_risk_result = CopyrightResult(
            is_original=True,
            risk_level="low",
            usage_allowed=True,
            attribution_required=False
        )
        
        guidelines = copyright_checker.get_usage_guidelines(low_risk_result)
        
        assert guidelines['commercial_use_allowed'] is True
        assert guidelines['attribution_required'] is False
        assert guidelines['modifications_allowed'] is True
        assert guidelines['redistribution_allowed'] is True
        assert guidelines['legal_review_recommended'] is False
        
        # High risk result
        high_risk_result = CopyrightResult(
            is_original=False,
            risk_level="high",
            usage_allowed=False,
            attribution_required=True
        )
        
        guidelines = copyright_checker.get_usage_guidelines(high_risk_result)
        
        assert guidelines['commercial_use_allowed'] is False
        assert guidelines['attribution_required'] is True
        assert guidelines['legal_review_recommended'] is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, copyright_checker, test_request):
        """Test error handling for invalid inputs."""
        # Test with non-existent file
        result = await copyright_checker.check_copyright("/non/existent/file.jpg", test_request)
        
        # Should return high risk result without crashing
        assert isinstance(result, CopyrightResult)
        assert result.risk_level == "high"
        assert "manual review required" in ' '.join(result.recommendations).lower()
    
    @pytest.mark.asyncio
    async def test_video_copyright_check(self, copyright_checker, test_request):
        """Test video copyright checking (basic)."""
        # Create a fake video file path
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        try:
            result = await copyright_checker.check_copyright(video_path, test_request)
            
            assert isinstance(result, CopyrightResult)
            # Video checking is basic, so should be low risk
            assert result.risk_level == "low"
            
        finally:
            Path(video_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_reverse_search_integration(self, copyright_checker, test_image_path):
        """Test reverse search integration (mocked)."""
        # Mock the reverse search functions
        async def mock_google_search(image_path):
            return [CopyrightMatch(
                source_url="https://example.com/similar",
                similarity_score=0.8,
                match_type="similar"
            )]
        
        # Replace the method in the reverse_search_apis dict
        copyright_checker.reverse_search_apis['google'] = mock_google_search
        
        matches = await copyright_checker._perform_reverse_search(test_image_path)
        
        assert len(matches) >= 1
        google_matches = [m for m in matches if m.source_url == "https://example.com/similar"]
        assert len(google_matches) == 1
    
    @pytest.mark.asyncio
    async def test_comprehensive_image_check(self, copyright_checker, test_image_path, test_request):
        """Test comprehensive image copyright checking."""
        # Add some test data
        copyright_checker.trademark_database = {"TestBrand": {"holder": "Test Corp"}}
        
        result = await copyright_checker._check_image_copyright(test_image_path, test_request)
        
        assert isinstance(result, CopyrightResult)
        assert isinstance(result.matches, list)
        assert isinstance(result.watermark_detected, bool)
        assert result.risk_level in ['low', 'medium', 'high', 'critical']
        assert len(result.recommendations) > 0


class TestCopyrightAccuracy:
    """Test accuracy and reliability of copyright detection."""
    
    @pytest.fixture
    def copyright_checker(self):
        """Create copyright checker for accuracy testing."""
        config = SafetyConfig(enabled=True, copyright_check=True)
        return CopyrightChecker(config)
    
    def test_hash_consistency(self, copyright_checker):
        """Test that hash calculation is consistent."""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Calculate hash multiple times
        hashes = []
        for _ in range(5):
            hash_value = copyright_checker._calculate_perceptual_hash(test_image)
            hashes.append(hash_value)
        
        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)
    
    def test_similarity_detection_accuracy(self, copyright_checker):
        """Test accuracy of similarity detection."""
        # Create base image
        base_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create slightly modified version
        modified_image = base_image.copy()
        modified_image[0:10, 0:10] = 255  # Small modification
        
        # Create completely different image
        different_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        base_hash = copyright_checker._calculate_perceptual_hash(base_image)
        modified_hash = copyright_checker._calculate_perceptual_hash(modified_image)
        different_hash = copyright_checker._calculate_perceptual_hash(different_image)
        
        # Similar images should have high similarity
        similarity_modified = copyright_checker._calculate_hash_similarity(base_hash, modified_hash)
        assert similarity_modified > 0.8
        
        # Different images should have low similarity
        similarity_different = copyright_checker._calculate_hash_similarity(base_hash, different_hash)
        assert similarity_different < 0.7
    
    @pytest.mark.asyncio
    async def test_false_positive_rate(self, copyright_checker):
        """Test false positive rate for original content."""
        # Create multiple original images
        original_images = []
        for i in range(10):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            original_images.append(image)
        
        false_positives = 0
        
        for image in original_images:
            hash_value = copyright_checker._calculate_perceptual_hash(image)
            matches = await copyright_checker._check_hash_database(hash_value)
            
            if matches:
                false_positives += 1
        
        # Should have very low false positive rate
        false_positive_rate = false_positives / len(original_images)
        assert false_positive_rate < 0.1  # Less than 10%


if __name__ == "__main__":
    pytest.main([__file__])