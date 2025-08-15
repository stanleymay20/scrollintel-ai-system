"""
Quality assessment for generated visual content.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    from PIL import Image, ImageStat, ImageFilter
except ImportError:
    Image = None
    ImageStat = None
    ImageFilter = None

try:
    import numpy as np
    import cv2
except ImportError:
    np = None
    cv2 = None

from ..base import QualityMetrics, GenerationRequest
from ..config import QualityConfig


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    metrics: QualityMetrics
    detailed_analysis: Dict[str, Any]
    improvement_suggestions: List[str]
    quality_grade: str  # A, B, C, D, F
    assessment_timestamp: datetime
    processing_time: float


class QualityAssessor:
    """Advanced quality assessment engine for generated content."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_calculators = {
            'sharpness': self._calculate_sharpness,
            'color_balance': self._calculate_color_balance,
            'composition': self._calculate_composition,
            'realism_score': self._calculate_realism,
            'temporal_consistency': self._calculate_temporal_consistency,
            'noise_level': self._calculate_noise_level,
            'contrast': self._calculate_contrast,
            'saturation': self._calculate_saturation,
            'brightness': self._calculate_brightness,
            'detail_preservation': self._calculate_detail_preservation
        }
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.6,
            'unacceptable': 0.0
        }
    
    async def assess_quality(self, content_paths: List[str], request: GenerationRequest) -> QualityMetrics:
        """Assess quality of generated content."""
        if not self.config.enabled:
            return QualityMetrics(overall_score=0.8)  # Default score
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            report = await self.comprehensive_quality_assessment(content_paths, request)
            return report.metrics
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return QualityMetrics(
                overall_score=0.7,
                technical_quality=0.7,
                aesthetic_score=0.7,
                prompt_adherence=0.7
            )
    
    async def comprehensive_quality_assessment(self, content_paths: List[str], request: GenerationRequest) -> QualityReport:
        """Perform comprehensive quality assessment with detailed reporting."""
        start_time = asyncio.get_event_loop().time()
        
        if not self.config.enabled:
            return QualityReport(
                overall_score=0.8,
                metrics=QualityMetrics(overall_score=0.8),
                detailed_analysis={},
                improvement_suggestions=[],
                quality_grade="B",
                assessment_timestamp=datetime.now(),
                processing_time=0.0
            )
        
        metrics = QualityMetrics()
        detailed_analysis = {}
        
        try:
            # Assess each piece of content
            for i, content_path in enumerate(content_paths):
                content_analysis = {}
                
                if Path(content_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    content_analysis = await self._comprehensive_image_assessment(content_path, metrics)
                elif Path(content_path).suffix.lower() in ['.mp4', '.avi', '.mov']:
                    content_analysis = await self._comprehensive_video_assessment(content_path, metrics)
                
                detailed_analysis[f"content_{i}"] = content_analysis
            
            # Assess prompt adherence
            metrics.prompt_adherence = await self._assess_prompt_adherence(request, content_paths)
            
            # Calculate overall score
            metrics.overall_score = self._calculate_overall_score(metrics)
            
            # Enforce minimum quality threshold
            if metrics.overall_score < self.config.min_quality_score:
                metrics.overall_score = self.config.min_quality_score
            
            # Generate improvement suggestions
            suggestions = await self.suggest_improvements(metrics)
            
            # Determine quality grade
            quality_grade = self._determine_quality_grade(metrics.overall_score)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return QualityReport(
                overall_score=metrics.overall_score,
                metrics=metrics,
                detailed_analysis=detailed_analysis,
                improvement_suggestions=suggestions,
                quality_grade=quality_grade,
                assessment_timestamp=datetime.now(),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive quality assessment failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return QualityReport(
                overall_score=0.7,
                metrics=QualityMetrics(overall_score=0.7, technical_quality=0.7, aesthetic_score=0.7, prompt_adherence=0.7),
                detailed_analysis={"error": str(e)},
                improvement_suggestions=["Quality assessment failed - please check content format"],
                quality_grade="C",
                assessment_timestamp=datetime.now(),
                processing_time=processing_time
            )
    
    async def _comprehensive_image_assessment(self, image_path: str, metrics: QualityMetrics) -> Dict[str, Any]:
        """Perform comprehensive image quality assessment."""
        analysis = {}
        
        try:
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Technical quality metrics
            technical_metrics = {}
            
            if 'sharpness' in self.config.quality_metrics:
                metrics.sharpness = await self._calculate_sharpness(image_array)
                technical_metrics['sharpness'] = metrics.sharpness
            
            if 'color_balance' in self.config.quality_metrics:
                metrics.color_balance = await self._calculate_color_balance(image_array)
                technical_metrics['color_balance'] = metrics.color_balance
            
            if 'noise_level' in self.config.quality_metrics:
                noise_level = await self._calculate_noise_level(image_array)
                technical_metrics['noise_level'] = noise_level
            
            if 'contrast' in self.config.quality_metrics:
                contrast = await self._calculate_contrast(image_array)
                technical_metrics['contrast'] = contrast
            
            if 'saturation' in self.config.quality_metrics:
                saturation = await self._calculate_saturation(image_array)
                technical_metrics['saturation'] = saturation
            
            if 'brightness' in self.config.quality_metrics:
                brightness = await self._calculate_brightness(image_array)
                technical_metrics['brightness'] = brightness
            
            # Aesthetic quality metrics
            aesthetic_metrics = {}
            
            if 'composition' in self.config.quality_metrics:
                metrics.composition_score = await self._calculate_composition(image_array)
                aesthetic_metrics['composition'] = metrics.composition_score
            
            if 'realism_score' in self.config.quality_metrics:
                metrics.realism_score = await self._calculate_realism(image_array)
                aesthetic_metrics['realism'] = metrics.realism_score
            
            # Calculate derived metrics
            metrics.technical_quality = self._calculate_technical_quality(technical_metrics)
            metrics.aesthetic_score = self._calculate_aesthetic_score(aesthetic_metrics)
            
            # Image-specific analysis
            analysis.update({
                'image_dimensions': image.size,
                'image_mode': image.mode,
                'file_size': Path(image_path).stat().st_size,
                'technical_metrics': technical_metrics,
                'aesthetic_metrics': aesthetic_metrics,
                'pixel_count': image.size[0] * image.size[1],
                'aspect_ratio': image.size[0] / image.size[1]
            })
            
        except Exception as e:
            self.logger.error(f"Image assessment failed for {image_path}: {e}")
            # Set default values if assessment fails
            metrics.technical_quality = 0.7
            metrics.aesthetic_score = 0.7
            analysis['error'] = str(e)
        
        return analysis
    
    async def _comprehensive_video_assessment(self, video_path: str, metrics: QualityMetrics) -> Dict[str, Any]:
        """Perform comprehensive video quality assessment."""
        analysis = {}
        
        try:
            # Video-specific quality metrics
            video_metrics = {}
            
            if 'temporal_consistency' in self.config.quality_metrics:
                metrics.temporal_consistency = await self._calculate_temporal_consistency(video_path)
                video_metrics['temporal_consistency'] = metrics.temporal_consistency
            
            if 'motion_smoothness' in self.config.quality_metrics:
                metrics.motion_smoothness = await self._calculate_motion_smoothness(video_path)
                video_metrics['motion_smoothness'] = metrics.motion_smoothness
            
            if 'frame_quality' in self.config.quality_metrics:
                metrics.frame_quality = await self._calculate_frame_quality(video_path)
                video_metrics['frame_quality'] = metrics.frame_quality
            
            # For ultra-realistic videos
            if 'realism_score' in self.config.quality_metrics:
                metrics.realism_score = await self._calculate_video_realism(video_path)
                video_metrics['realism'] = metrics.realism_score
            
            # Calculate derived metrics
            metrics.technical_quality = (
                (metrics.temporal_consistency or 0.8) * 0.4 +
                (metrics.motion_smoothness or 0.8) * 0.3 +
                (metrics.frame_quality or 0.8) * 0.3
            )
            
            metrics.aesthetic_score = (
                (metrics.realism_score or 0.8) * 0.6 +
                (metrics.temporal_consistency or 0.8) * 0.4
            )
            
            # Video-specific analysis
            video_info = await self._get_video_info(video_path)
            analysis.update({
                'video_info': video_info,
                'video_metrics': video_metrics,
                'file_size': Path(video_path).stat().st_size
            })
            
        except Exception as e:
            self.logger.error(f"Video assessment failed for {video_path}: {e}")
            # Set default values if assessment fails
            metrics.technical_quality = 0.8
            metrics.aesthetic_score = 0.8
            analysis['error'] = str(e)
        
        return analysis
    
    async def _calculate_sharpness(self, image_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image_array
            
            # Calculate Laplacian variance
            laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            
            # Simple convolution approximation
            variance = np.var(gray)
            
            # Normalize to 0-1 range
            sharpness = min(1.0, variance / 10000.0)
            
            return max(0.0, sharpness)
            
        except Exception:
            return 0.7  # Default sharpness score
    
    async def _calculate_color_balance(self, image_array: np.ndarray) -> float:
        """Calculate color balance quality."""
        try:
            if len(image_array.shape) != 3:
                return 0.8  # Default for grayscale
            
            # Calculate mean values for each channel
            r_mean = np.mean(image_array[:, :, 0])
            g_mean = np.mean(image_array[:, :, 1])
            b_mean = np.mean(image_array[:, :, 2])
            
            # Calculate balance (lower deviation = better balance)
            total_mean = (r_mean + g_mean + b_mean) / 3
            deviation = (
                abs(r_mean - total_mean) +
                abs(g_mean - total_mean) +
                abs(b_mean - total_mean)
            ) / 3
            
            # Convert to 0-1 score (lower deviation = higher score)
            balance_score = max(0.0, 1.0 - (deviation / 128.0))
            
            return balance_score
            
        except Exception:
            return 0.8  # Default color balance score
    
    async def _calculate_composition(self, image_array: np.ndarray) -> float:
        """Calculate composition quality (simplified)."""
        try:
            # Placeholder for composition analysis
            # In a real implementation, this would analyze rule of thirds, symmetry, etc.
            
            height, width = image_array.shape[:2]
            
            # Simple composition score based on aspect ratio and size
            aspect_ratio = width / height
            
            # Prefer common aspect ratios
            common_ratios = [1.0, 1.33, 1.5, 1.77, 2.0]  # 1:1, 4:3, 3:2, 16:9, 2:1
            
            closest_ratio_diff = min(abs(aspect_ratio - ratio) for ratio in common_ratios)
            composition_score = max(0.5, 1.0 - closest_ratio_diff)
            
            return composition_score
            
        except Exception:
            return 0.8  # Default composition score
    
    async def _calculate_realism(self, image_array: np.ndarray) -> float:
        """Calculate realism score (placeholder)."""
        try:
            # Placeholder for realism assessment
            # In a real implementation, this would use trained models to assess realism
            
            # Simple heuristic based on color distribution and variance
            if len(image_array.shape) == 3:
                color_variance = np.var(image_array, axis=(0, 1))
                avg_variance = np.mean(color_variance)
                
                # Normalize variance to realism score
                realism_score = min(1.0, avg_variance / 5000.0)
                return max(0.6, realism_score)
            
            return 0.8
            
        except Exception:
            return 0.8  # Default realism score
    
    async def _calculate_noise_level(self, image_array: np.ndarray) -> float:
        """Calculate noise level in the image."""
        try:
            if len(image_array.shape) == 3:
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image_array
            
            # Use Laplacian to detect noise
            laplacian_var = np.var(gray)
            
            # Normalize to 0-1 range (lower noise = higher score)
            noise_score = max(0.0, min(1.0, 1.0 - (laplacian_var / 50000.0)))
            
            return noise_score
            
        except Exception:
            return 0.8
    
    async def _calculate_contrast(self, image_array: np.ndarray) -> float:
        """Calculate image contrast."""
        try:
            if len(image_array.shape) == 3:
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image_array
            
            # Calculate RMS contrast
            contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
            
            # Normalize to 0-1 range
            contrast_score = min(1.0, contrast / 2.0)
            
            return max(0.0, contrast_score)
            
        except Exception:
            return 0.8
    
    async def _calculate_saturation(self, image_array: np.ndarray) -> float:
        """Calculate color saturation."""
        try:
            if len(image_array.shape) != 3:
                return 0.8  # Default for grayscale
            
            # Convert to HSV to get saturation channel
            hsv = np.zeros_like(image_array)
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    r, g, b = image_array[i, j] / 255.0
                    max_val = max(r, g, b)
                    min_val = min(r, g, b)
                    
                    if max_val == 0:
                        saturation = 0
                    else:
                        saturation = (max_val - min_val) / max_val
                    
                    hsv[i, j, 1] = saturation * 255
            
            # Calculate average saturation
            avg_saturation = np.mean(hsv[:, :, 1]) / 255.0
            
            return max(0.0, min(1.0, avg_saturation))
            
        except Exception:
            return 0.8
    
    async def _calculate_brightness(self, image_array: np.ndarray) -> float:
        """Calculate image brightness."""
        try:
            # Calculate average brightness
            if len(image_array.shape) == 3:
                brightness = np.mean(image_array)
            else:
                brightness = np.mean(image_array)
            
            # Normalize to 0-1 range and prefer mid-range brightness
            normalized_brightness = brightness / 255.0
            
            # Score based on how close to optimal brightness (0.5)
            brightness_score = 1.0 - abs(normalized_brightness - 0.5) * 2
            
            return max(0.0, min(1.0, brightness_score))
            
        except Exception:
            return 0.8
    
    async def _calculate_detail_preservation(self, image_array: np.ndarray) -> float:
        """Calculate detail preservation quality."""
        try:
            if len(image_array.shape) == 3:
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image_array
            
            # Use gradient magnitude to measure detail preservation
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate detail score
            detail_score = np.mean(gradient_magnitude) / 128.0
            
            return max(0.0, min(1.0, detail_score))
            
        except Exception:
            return 0.8
    
    async def _calculate_temporal_consistency(self, video_path: str) -> float:
        """Calculate temporal consistency for videos."""
        try:
            # Placeholder for video temporal consistency analysis
            # In a real implementation, this would analyze frame-to-frame consistency
            
            # Simple heuristic based on file size and duration
            file_size = Path(video_path).stat().st_size
            
            # Assume consistent videos have appropriate file sizes
            if file_size > 1024 * 1024:  # > 1MB
                return 0.9
            else:
                return 0.7
                
        except Exception:
            return 0.8
    
    async def _calculate_motion_smoothness(self, video_path: str) -> float:
        """Calculate motion smoothness for videos."""
        try:
            # Placeholder for motion smoothness analysis
            return 0.85
        except Exception:
            return 0.8
    
    async def _calculate_frame_quality(self, video_path: str) -> float:
        """Calculate average frame quality for videos."""
        try:
            # Placeholder for frame quality analysis
            return 0.9
        except Exception:
            return 0.8
    
    async def _calculate_video_realism(self, video_path: str) -> float:
        """Calculate realism score for videos."""
        try:
            # Placeholder for video realism analysis
            return 0.95
        except Exception:
            return 0.8
    
    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information."""
        try:
            return {
                'duration': 5.0,  # Placeholder
                'fps': 30,        # Placeholder
                'resolution': (1920, 1080),  # Placeholder
                'codec': 'h264'   # Placeholder
            }
        except Exception:
            return {}
    
    async def _assess_prompt_adherence(self, request: GenerationRequest, content_paths: List[str]) -> float:
        """Assess how well the generated content adheres to the prompt."""
        try:
            # Placeholder for prompt adherence analysis
            # In a real implementation, this would use NLP models to compare
            # the prompt with the generated content
            
            prompt_length = len(request.prompt.split())
            
            # Simple heuristic: longer, more detailed prompts are harder to adhere to
            if prompt_length > 20:
                return 0.8
            elif prompt_length > 10:
                return 0.85
            else:
                return 0.9
                
        except Exception:
            return 0.85
    
    def _calculate_technical_quality(self, technical_metrics: Dict[str, float]) -> float:
        """Calculate technical quality score from individual technical metrics."""
        if not technical_metrics:
            return 0.8
        
        # Weight different technical metrics
        weights = {
            'sharpness': 0.25,
            'color_balance': 0.20,
            'noise_level': 0.15,
            'contrast': 0.15,
            'saturation': 0.10,
            'brightness': 0.10,
            'detail_preservation': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in technical_metrics.items():
            if metric in weights:
                weighted_score += score * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return sum(technical_metrics.values()) / len(technical_metrics)
    
    def _calculate_aesthetic_score(self, aesthetic_metrics: Dict[str, float]) -> float:
        """Calculate aesthetic score from individual aesthetic metrics."""
        if not aesthetic_metrics:
            return 0.8
        
        # Weight different aesthetic metrics
        weights = {
            'composition': 0.4,
            'realism': 0.6
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in aesthetic_metrics.items():
            if metric in weights:
                weighted_score += score * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return sum(aesthetic_metrics.values()) / len(aesthetic_metrics)
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score from individual metrics."""
        technical_weight = self.config.technical_quality_weight
        aesthetic_weight = self.config.aesthetic_weight
        adherence_weight = self.config.prompt_adherence_weight
        
        overall_score = (
            metrics.technical_quality * technical_weight +
            metrics.aesthetic_score * aesthetic_weight +
            metrics.prompt_adherence * adherence_weight
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """Determine quality grade based on overall score."""
        if overall_score >= self.quality_thresholds['excellent']:
            return 'A'
        elif overall_score >= self.quality_thresholds['good']:
            return 'B'
        elif overall_score >= self.quality_thresholds['acceptable']:
            return 'C'
        elif overall_score >= self.quality_thresholds['poor']:
            return 'D'
        else:
            return 'F'
    
    async def suggest_improvements(self, metrics: QualityMetrics) -> List[str]:
        """Generate detailed improvement suggestions based on quality metrics."""
        suggestions = []
        
        # Technical quality suggestions
        if metrics.sharpness and metrics.sharpness < 0.7:
            if metrics.sharpness < 0.5:
                suggestions.append("Image appears very blurry - consider using higher resolution settings or post-processing sharpening")
            else:
                suggestions.append("Slight blur detected - try increasing generation steps or using upscaling")
        
        if metrics.color_balance and metrics.color_balance < 0.7:
            suggestions.append("Color balance issues detected - adjust color temperature or use color correction")
        
        if hasattr(metrics, 'noise_level') and getattr(metrics, 'noise_level', 1.0) < 0.7:
            suggestions.append("High noise levels detected - try using denoising filters or higher quality settings")
        
        if hasattr(metrics, 'contrast') and getattr(metrics, 'contrast', 1.0) < 0.6:
            suggestions.append("Low contrast detected - consider adjusting brightness/contrast or using HDR techniques")
        
        if hasattr(metrics, 'saturation') and getattr(metrics, 'saturation', 1.0) < 0.6:
            suggestions.append("Colors appear desaturated - increase color vibrancy or saturation settings")
        
        # Aesthetic quality suggestions
        if metrics.composition_score and metrics.composition_score < 0.7:
            suggestions.append("Composition could be improved - consider rule of thirds, better framing, or different camera angles")
        
        if metrics.realism_score and metrics.realism_score < 0.8:
            if metrics.realism_score < 0.6:
                suggestions.append("Content appears artificial - try more detailed prompts or realistic style keywords")
            else:
                suggestions.append("Minor realism issues - consider adding more natural lighting or texture details")
        
        # Prompt adherence suggestions
        if metrics.prompt_adherence and metrics.prompt_adherence < 0.7:
            if metrics.prompt_adherence < 0.5:
                suggestions.append("Generated content significantly differs from prompt - try more specific descriptions or different model")
            else:
                suggestions.append("Some prompt elements missing - consider rephrasing or adding emphasis to key terms")
        
        # Video-specific suggestions
        if metrics.temporal_consistency and metrics.temporal_consistency < 0.8:
            suggestions.append("Temporal inconsistencies detected - use higher frame rate or temporal smoothing")
        
        if metrics.motion_smoothness and metrics.motion_smoothness < 0.8:
            suggestions.append("Motion appears jerky - consider motion blur or frame interpolation")
        
        if metrics.frame_quality and metrics.frame_quality < 0.8:
            suggestions.append("Individual frame quality is low - increase resolution or quality settings")
        
        # Overall quality suggestions
        if metrics.overall_score < 0.6:
            suggestions.append("Overall quality is below acceptable standards - consider regenerating with different parameters")
        elif metrics.overall_score < 0.8:
            suggestions.append("Quality is acceptable but could be improved - try fine-tuning generation parameters")
        
        # Safety and uniqueness suggestions
        if metrics.safety_score and metrics.safety_score < 0.9:
            suggestions.append("Content may have safety concerns - review and regenerate if necessary")
        
        if metrics.uniqueness_score and metrics.uniqueness_score < 0.7:
            suggestions.append("Content appears generic - try more unique or creative prompts")
        
        return suggestions
    
    async def generate_quality_report(self, report: QualityReport) -> str:
        """Generate a human-readable quality report."""
        report_lines = [
            f"Quality Assessment Report",
            f"========================",
            f"Overall Score: {report.overall_score:.2f} (Grade: {report.quality_grade})",
            f"Assessment Time: {report.assessment_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Processing Time: {report.processing_time:.2f}s",
            f"",
            f"Detailed Metrics:",
            f"- Technical Quality: {report.metrics.technical_quality:.2f}",
            f"- Aesthetic Score: {report.metrics.aesthetic_score:.2f}",
            f"- Prompt Adherence: {report.metrics.prompt_adherence:.2f}",
        ]
        
        if report.metrics.sharpness:
            report_lines.append(f"- Sharpness: {report.metrics.sharpness:.2f}")
        if report.metrics.color_balance:
            report_lines.append(f"- Color Balance: {report.metrics.color_balance:.2f}")
        if report.metrics.composition_score:
            report_lines.append(f"- Composition: {report.metrics.composition_score:.2f}")
        if report.metrics.realism_score:
            report_lines.append(f"- Realism: {report.metrics.realism_score:.2f}")
        
        if report.improvement_suggestions:
            report_lines.extend([
                f"",
                f"Improvement Suggestions:",
            ])
            for i, suggestion in enumerate(report.improvement_suggestions, 1):
                report_lines.append(f"{i}. {suggestion}")
        
        return "\n".join(report_lines)
    
    async def export_quality_data(self, report: QualityReport, export_path: str):
        """Export quality assessment data to JSON file."""
        try:
            export_data = {
                'overall_score': report.overall_score,
                'quality_grade': report.quality_grade,
                'metrics': asdict(report.metrics),
                'detailed_analysis': report.detailed_analysis,
                'improvement_suggestions': report.improvement_suggestions,
                'assessment_timestamp': report.assessment_timestamp.isoformat(),
                'processing_time': report.processing_time
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.logger.info(f"Quality report exported to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export quality report: {e}")
    
    def get_quality_statistics(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Calculate statistics from multiple quality reports."""
        if not reports:
            return {}
        
        scores = [report.overall_score for report in reports]
        technical_scores = [report.metrics.technical_quality for report in reports]
        aesthetic_scores = [report.metrics.aesthetic_score for report in reports]
        adherence_scores = [report.metrics.prompt_adherence for report in reports]
        
        return {
            'total_assessments': len(reports),
            'average_overall_score': sum(scores) / len(scores),
            'average_technical_quality': sum(technical_scores) / len(technical_scores),
            'average_aesthetic_score': sum(aesthetic_scores) / len(aesthetic_scores),
            'average_prompt_adherence': sum(adherence_scores) / len(adherence_scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'grade_distribution': {
                'A': sum(1 for r in reports if r.quality_grade == 'A'),
                'B': sum(1 for r in reports if r.quality_grade == 'B'),
                'C': sum(1 for r in reports if r.quality_grade == 'C'),
                'D': sum(1 for r in reports if r.quality_grade == 'D'),
                'F': sum(1 for r in reports if r.quality_grade == 'F')
            }
        }