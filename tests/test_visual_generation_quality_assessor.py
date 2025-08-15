"""
Comprehensive tests for visual generation quality assessment engine.
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

from scrollintel.engines.visual_generation.utils.quality_assessor import (
    QualityAssessor,
    QualityReport
)
from scrollintel.engines.visual_generation.config import QualityConfig
from scrollintel.engines.visual_generation.base import (
    QualityMetrics,
    GenerationRequest
)


class TestQualityAssessor:
    """Test quality assessment functionality."""
    
    @pytest.fixture
    def quality_config(self):
        """Create test quality configuration."""
        return QualityConfig(
            enabled=True,
            min_quality_score=0.7,
            technical_quality_weight=0.3,
            aesthetic_weight=0.3,
            prompt_adherence_weight=0.4,
            auto_enhance=False,
            quality_metrics=[
                'sharpness', 'color_balance', 'composition',
                'realism_score', 'noise_level', 'contrast',
                'saturation', 'brightness'
            ]
        )
    
    @pytest.fixture
    def quality_assessor(self, quality_config):
        """Create quality assessor instance."""
        return QualityAssessor(quality_config)
    
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
            prompt="A beautiful landscape with mountains and trees",
            user_id="test_user"
        )
    
    @pytest.mark.asyncio
    async def test_basic_quality_assessment(self, quality_assessor, test_image_path, test_request):
        """Test basic quality assessment functionality."""
        metrics = await quality_assessor.assess_quality([test_image_path], test_request)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.overall_score <= 1.0
        assert 0.0 <= metrics.technical_quality <= 1.0
        assert 0.0 <= metrics.aesthetic_score <= 1.0
        assert 0.0 <= metrics.prompt_adherence <= 1.0
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_assessment(self, quality_assessor, test_image_path, test_request):
        """Test comprehensive quality assessment with detailed reporting."""
        report = await quality_assessor.comprehensive_quality_assessment([test_image_path], test_request)
        
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert report.quality_grade in ['A', 'B', 'C', 'D', 'F']
        assert isinstance(report.improvement_suggestions, list)
        assert isinstance(report.detailed_analysis, dict)
        assert isinstance(report.assessment_timestamp, datetime)
        assert report.processing_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_disabled_quality_assessment(self, quality_config, test_image_path, test_request):
        """Test quality assessment when disabled."""
        quality_config.enabled = False
        quality_assessor = QualityAssessor(quality_config)
        
        metrics = await quality_assessor.assess_quality([test_image_path], test_request)
        
        assert metrics.overall_score == 0.8  # Default score
    
    @pytest.mark.asyncio
    async def test_sharpness_calculation(self, quality_assessor):
        """Test sharpness calculation accuracy."""
        # Create sharp image
        sharp_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        sharp_score = await quality_assessor._calculate_sharpness(sharp_image)
        
        # Create blurry image (all same color)
        blurry_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        blurry_score = await quality_assessor._calculate_sharpness(blurry_image)
        
        assert 0.0 <= sharp_score <= 1.0
        assert 0.0 <= blurry_score <= 1.0
        # Sharp image should have higher score than blurry image
        assert sharp_score >= blurry_score
    
    @pytest.mark.asyncio
    async def test_color_balance_calculation(self, quality_assessor):
        """Test color balance calculation."""
        # Create balanced image
        balanced_image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        balanced_score = await quality_assessor._calculate_color_balance(balanced_image)
        
        # Create unbalanced image (only red)
        unbalanced_image = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        unbalanced_score = await quality_assessor._calculate_color_balance(unbalanced_image)
        
        assert 0.0 <= balanced_score <= 1.0
        assert 0.0 <= unbalanced_score <= 1.0
        # Balanced image should have higher score
        assert balanced_score >= unbalanced_score
    
    @pytest.mark.asyncio
    async def test_composition_calculation(self, quality_assessor):
        """Test composition quality calculation."""
        # Test different aspect ratios
        square_image = np.zeros((100, 100, 3), dtype=np.uint8)
        square_score = await quality_assessor._calculate_composition(square_image)
        
        wide_image = np.zeros((100, 200, 3), dtype=np.uint8)
        wide_score = await quality_assessor._calculate_composition(wide_image)
        
        assert 0.0 <= square_score <= 1.0
        assert 0.0 <= wide_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_noise_level_calculation(self, quality_assessor):
        """Test noise level calculation."""
        # Create clean image
        clean_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        clean_score = await quality_assessor._calculate_noise_level(clean_image)
        
        # Create noisy image
        noisy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        noisy_score = await quality_assessor._calculate_noise_level(noisy_image)
        
        assert 0.0 <= clean_score <= 1.0
        assert 0.0 <= noisy_score <= 1.0
        # Clean image should have higher score
        assert clean_score >= noisy_score
    
    @pytest.mark.asyncio
    async def test_contrast_calculation(self, quality_assessor):
        """Test contrast calculation."""
        # High contrast image (black and white)
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[:50, :] = 255
        high_contrast_score = await quality_assessor._calculate_contrast(high_contrast)
        
        # Low contrast image (all gray)
        low_contrast = np.full((100, 100, 3), 128, dtype=np.uint8)
        low_contrast_score = await quality_assessor._calculate_contrast(low_contrast)
        
        assert 0.0 <= high_contrast_score <= 1.0
        assert 0.0 <= low_contrast_score <= 1.0
        # High contrast should have higher score
        assert high_contrast_score >= low_contrast_score
    
    @pytest.mark.asyncio
    async def test_saturation_calculation(self, quality_assessor):
        """Test saturation calculation."""
        # Saturated image (pure red)
        saturated_image = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        saturated_score = await quality_assessor._calculate_saturation(saturated_image)
        
        # Desaturated image (gray)
        desaturated_image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        desaturated_score = await quality_assessor._calculate_saturation(desaturated_image)
        
        assert 0.0 <= saturated_score <= 1.0
        assert 0.0 <= desaturated_score <= 1.0
        # Saturated image should have higher score
        assert saturated_score >= desaturated_score
    
    @pytest.mark.asyncio
    async def test_brightness_calculation(self, quality_assessor):
        """Test brightness calculation."""
        # Optimal brightness (mid-gray)
        optimal_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        optimal_score = await quality_assessor._calculate_brightness(optimal_image)
        
        # Too dark
        dark_image = np.full((100, 100, 3), 50, dtype=np.uint8)
        dark_score = await quality_assessor._calculate_brightness(dark_image)
        
        # Too bright
        bright_image = np.full((100, 100, 3), 200, dtype=np.uint8)
        bright_score = await quality_assessor._calculate_brightness(bright_image)
        
        assert 0.0 <= optimal_score <= 1.0
        assert 0.0 <= dark_score <= 1.0
        assert 0.0 <= bright_score <= 1.0
        # Optimal brightness should have highest score
        assert optimal_score >= dark_score
        assert optimal_score >= bright_score
    
    @pytest.mark.asyncio
    async def test_prompt_adherence_assessment(self, quality_assessor, test_image_path):
        """Test prompt adherence assessment."""
        # Short prompt
        short_request = GenerationRequest(
            prompt="cat",
            user_id="test_user"
        )
        short_adherence = await quality_assessor._assess_prompt_adherence(short_request, [test_image_path])
        
        # Long detailed prompt
        long_request = GenerationRequest(
            prompt="A highly detailed, photorealistic portrait of a majestic orange tabby cat with bright green eyes, sitting on a wooden table in a cozy kitchen with warm lighting",
            user_id="test_user"
        )
        long_adherence = await quality_assessor._assess_prompt_adherence(long_request, [test_image_path])
        
        assert 0.0 <= short_adherence <= 1.0
        assert 0.0 <= long_adherence <= 1.0
        # Short prompts should be easier to adhere to
        assert short_adherence >= long_adherence
    
    @pytest.mark.asyncio
    async def test_technical_quality_calculation(self, quality_assessor):
        """Test technical quality calculation from metrics."""
        technical_metrics = {
            'sharpness': 0.8,
            'color_balance': 0.7,
            'noise_level': 0.9,
            'contrast': 0.6,
            'saturation': 0.8,
            'brightness': 0.7
        }
        
        technical_score = quality_assessor._calculate_technical_quality(technical_metrics)
        
        assert 0.0 <= technical_score <= 1.0
        assert isinstance(technical_score, float)
    
    @pytest.mark.asyncio
    async def test_aesthetic_score_calculation(self, quality_assessor):
        """Test aesthetic score calculation from metrics."""
        aesthetic_metrics = {
            'composition': 0.8,
            'realism': 0.9
        }
        
        aesthetic_score = quality_assessor._calculate_aesthetic_score(aesthetic_metrics)
        
        assert 0.0 <= aesthetic_score <= 1.0
        assert isinstance(aesthetic_score, float)
    
    def test_quality_grade_determination(self, quality_assessor):
        """Test quality grade determination."""
        assert quality_assessor._determine_quality_grade(0.95) == 'A'
        assert quality_assessor._determine_quality_grade(0.85) == 'B'
        assert quality_assessor._determine_quality_grade(0.75) == 'C'
        assert quality_assessor._determine_quality_grade(0.65) == 'D'
        assert quality_assessor._determine_quality_grade(0.45) == 'F'
    
    @pytest.mark.asyncio
    async def test_improvement_suggestions(self, quality_assessor):
        """Test improvement suggestions generation."""
        # Create metrics with various issues
        metrics = QualityMetrics(
            overall_score=0.6,
            technical_quality=0.5,
            aesthetic_score=0.6,
            prompt_adherence=0.5,
            sharpness=0.4,
            color_balance=0.5,
            composition_score=0.6,
            realism_score=0.5
        )
        
        suggestions = await quality_assessor.suggest_improvements(metrics)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check that suggestions are relevant to the low scores
        suggestion_text = ' '.join(suggestions).lower()
        assert 'blur' in suggestion_text or 'sharp' in suggestion_text  # For low sharpness
        assert 'color' in suggestion_text  # For low color balance
    
    @pytest.mark.asyncio
    async def test_quality_report_generation(self, quality_assessor):
        """Test quality report generation."""
        metrics = QualityMetrics(
            overall_score=0.8,
            technical_quality=0.8,
            aesthetic_score=0.8,
            prompt_adherence=0.8
        )
        
        report = QualityReport(
            overall_score=0.8,
            metrics=metrics,
            detailed_analysis={},
            improvement_suggestions=["Test suggestion"],
            quality_grade="B",
            assessment_timestamp=datetime.now(),
            processing_time=1.0
        )
        
        report_text = await quality_assessor.generate_quality_report(report)
        
        assert isinstance(report_text, str)
        assert "Quality Assessment Report" in report_text
        assert "Overall Score: 0.80" in report_text
        assert "Grade: B" in report_text
        assert "Test suggestion" in report_text
    
    @pytest.mark.asyncio
    async def test_quality_data_export(self, quality_assessor):
        """Test quality data export to JSON."""
        metrics = QualityMetrics(overall_score=0.8)
        report = QualityReport(
            overall_score=0.8,
            metrics=metrics,
            detailed_analysis={},
            improvement_suggestions=[],
            quality_grade="B",
            assessment_timestamp=datetime.now(),
            processing_time=1.0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            await quality_assessor.export_quality_data(report, export_path)
            
            # Verify export
            assert Path(export_path).exists()
            
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data['overall_score'] == 0.8
            assert exported_data['quality_grade'] == "B"
            assert 'metrics' in exported_data
            assert 'assessment_timestamp' in exported_data
            
        finally:
            Path(export_path).unlink(missing_ok=True)
    
    def test_quality_statistics(self, quality_assessor):
        """Test quality statistics calculation."""
        # Create sample reports
        reports = []
        for score in [0.9, 0.8, 0.7, 0.6]:
            metrics = QualityMetrics(
                overall_score=score,
                technical_quality=score,
                aesthetic_score=score,
                prompt_adherence=score
            )
            report = QualityReport(
                overall_score=score,
                metrics=metrics,
                detailed_analysis={},
                improvement_suggestions=[],
                quality_grade="A" if score >= 0.9 else "B" if score >= 0.8 else "C" if score >= 0.7 else "D",
                assessment_timestamp=datetime.now(),
                processing_time=1.0
            )
            reports.append(report)
        
        stats = quality_assessor.get_quality_statistics(reports)
        
        assert stats['total_assessments'] == 4
        assert stats['average_overall_score'] == 0.75
        assert stats['min_score'] == 0.6
        assert stats['max_score'] == 0.9
        assert 'grade_distribution' in stats
        assert stats['grade_distribution']['A'] == 1
        assert stats['grade_distribution']['B'] == 1
        assert stats['grade_distribution']['C'] == 1
        assert stats['grade_distribution']['D'] == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, quality_assessor, test_request):
        """Test error handling for invalid inputs."""
        # Test with non-existent file
        metrics = await quality_assessor.assess_quality(["/non/existent/file.jpg"], test_request)
        
        # Should return default metrics without crashing
        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.overall_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_video_quality_assessment(self, quality_assessor, test_request):
        """Test video quality assessment (placeholder)."""
        # Create a fake video file path
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        try:
            # Test video assessment
            analysis = await quality_assessor._comprehensive_video_assessment(video_path, QualityMetrics())
            
            assert isinstance(analysis, dict)
            # Should handle missing video gracefully
            
        finally:
            Path(video_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_multiple_content_assessment(self, quality_assessor, test_request):
        """Test assessment of multiple content pieces."""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.png', delete=False) as f:
                image = Image.new('RGB', (100, 100), color=['red', 'green', 'blue'][i])
                image.save(f.name)
                image_paths.append(f.name)
        
        try:
            metrics = await quality_assessor.assess_quality(image_paths, test_request)
            
            assert isinstance(metrics, QualityMetrics)
            assert 0.0 <= metrics.overall_score <= 1.0
            
        finally:
            for path in image_paths:
                Path(path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_quality_threshold_enforcement(self, quality_config, test_image_path, test_request):
        """Test quality threshold enforcement."""
        quality_config.min_quality_score = 0.9
        quality_assessor = QualityAssessor(quality_config)
        
        metrics = await quality_assessor.assess_quality([test_image_path], test_request)
        
        # Should enforce minimum quality score
        assert metrics.overall_score >= quality_config.min_quality_score


class TestQualityMetricsAccuracy:
    """Test accuracy and consistency of quality metrics."""
    
    @pytest.fixture
    def quality_assessor(self):
        """Create quality assessor for accuracy testing."""
        config = QualityConfig(
            enabled=True,
            quality_metrics=['sharpness', 'color_balance', 'composition', 'realism_score']
        )
        return QualityAssessor(config)
    
    @pytest.mark.asyncio
    async def test_metric_consistency(self, quality_assessor):
        """Test that metrics are consistent across multiple runs."""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Run assessment multiple times
        scores = []
        for _ in range(5):
            score = await quality_assessor._calculate_sharpness(test_image)
            scores.append(score)
        
        # All scores should be identical (deterministic)
        assert all(abs(score - scores[0]) < 0.001 for score in scores)
    
    @pytest.mark.asyncio
    async def test_metric_range_validation(self, quality_assessor):
        """Test that all metrics return values in valid range [0, 1]."""
        test_images = [
            np.zeros((100, 100, 3), dtype=np.uint8),  # Black
            np.full((100, 100, 3), 255, dtype=np.uint8),  # White
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),  # Random
        ]
        
        for test_image in test_images:
            sharpness = await quality_assessor._calculate_sharpness(test_image)
            color_balance = await quality_assessor._calculate_color_balance(test_image)
            composition = await quality_assessor._calculate_composition(test_image)
            realism = await quality_assessor._calculate_realism(test_image)
            
            assert 0.0 <= sharpness <= 1.0
            assert 0.0 <= color_balance <= 1.0
            assert 0.0 <= composition <= 1.0
            assert 0.0 <= realism <= 1.0
    
    @pytest.mark.asyncio
    async def test_extreme_cases(self, quality_assessor):
        """Test quality assessment with extreme cases."""
        # Extremely small image
        tiny_image = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
        tiny_score = await quality_assessor._calculate_sharpness(tiny_image)
        assert 0.0 <= tiny_score <= 1.0
        
        # Single color image
        solid_image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        solid_score = await quality_assessor._calculate_sharpness(solid_image)
        assert 0.0 <= solid_score <= 1.0
        
        # Grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray_score = await quality_assessor._calculate_color_balance(gray_image)
        assert 0.0 <= gray_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])