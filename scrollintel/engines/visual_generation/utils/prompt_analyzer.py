"""
Advanced prompt analysis engine for visual content generation.
Provides comprehensive analysis of user prompts including style detection,
technical parameter extraction, and quality scoring.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import Counter

from ..base import GenerationRequest
from ..exceptions import PromptAnalysisError


class PromptComplexity(Enum):
    """Prompt complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ContentType(Enum):
    """Detected content types."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ARCHITECTURE = "architecture"
    ABSTRACT = "abstract"
    OBJECT = "object"
    SCENE = "scene"
    UNKNOWN = "unknown"


class ArtisticStyle(Enum):
    """Detected artistic styles."""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    CARTOON = "cartoon"
    SKETCH = "sketch"
    CINEMATIC = "cinematic"
    ABSTRACT = "abstract"
    DIGITAL_ART = "digital_art"
    PAINTING = "painting"
    UNKNOWN = "unknown"


@dataclass
class TechnicalParameters:
    """Extracted technical parameters from prompt."""
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    lighting: List[str] = None
    camera_settings: List[str] = None
    rendering_style: List[str] = None
    quality_modifiers: List[str] = None
    
    def __post_init__(self):
        if self.lighting is None:
            self.lighting = []
        if self.camera_settings is None:
            self.camera_settings = []
        if self.rendering_style is None:
            self.rendering_style = []
        if self.quality_modifiers is None:
            self.quality_modifiers = []


@dataclass
class PromptQualityMetrics:
    """Quality metrics for prompt analysis."""
    overall_score: float  # 0-1
    specificity_score: float  # 0-1
    technical_completeness: float  # 0-1
    style_clarity: float  # 0-1
    structure_score: float  # 0-1
    improvement_potential: float  # 0-1


@dataclass
class PromptAnalysisResult:
    """Complete result of prompt analysis."""
    original_prompt: str
    word_count: int
    complexity: PromptComplexity
    content_type: ContentType
    artistic_style: ArtisticStyle
    technical_parameters: TechnicalParameters
    quality_metrics: PromptQualityMetrics
    detected_subjects: List[str]
    detected_objects: List[str]
    detected_emotions: List[str]
    missing_elements: List[str]
    improvement_suggestions: List[str]
    confidence: float
    analysis_timestamp: str


class PromptAnalyzer:
    """Advanced prompt analysis engine for visual content generation."""
    
    def __init__(self):
        self.style_patterns = self._load_style_patterns()
        self.technical_patterns = self._load_technical_patterns()
        self.content_patterns = self._load_content_patterns()
        self.quality_indicators = self._load_quality_indicators()
        self.subject_patterns = self._load_subject_patterns()
        self.emotion_patterns = self._load_emotion_patterns()
        self.object_patterns = self._load_object_patterns()
        
    def _load_style_patterns(self) -> Dict[ArtisticStyle, List[str]]:
        """Load patterns for artistic style detection."""
        return {
            ArtisticStyle.PHOTOREALISTIC: [
                r'\b(?:photo|photograph|photorealistic|hyperrealistic|realistic|real)\b',
                r'\b(?:dslr|camera|lens|photography|shot)\b',
                r'\b(?:8k|4k|hd|high resolution|ultra hd)\b'
            ],
            ArtisticStyle.ARTISTIC: [
                r'\b(?:painting|painted|art|artistic|masterpiece)\b',
                r'\b(?:oil painting|watercolor|acrylic|canvas)\b',
                r'\b(?:by .+|in the style of|inspired by)\b'
            ],
            ArtisticStyle.CARTOON: [
                r'\b(?:cartoon|animated|animation|disney|pixar)\b',
                r'\b(?:cel shaded|toon|stylized|cute)\b',
                r'\b(?:3d render|cgi|computer graphics)\b'
            ],
            ArtisticStyle.SKETCH: [
                r'\b(?:sketch|drawing|drawn|pencil|charcoal)\b',
                r'\b(?:line art|black and white|monochrome)\b',
                r'\b(?:hand drawn|illustration|ink)\b'
            ],
            ArtisticStyle.CINEMATIC: [
                r'\b(?:cinematic|movie|film|scene|shot)\b',
                r'\b(?:dramatic|epic|wide shot|close up)\b',
                r'\b(?:depth of field|bokeh|film grain)\b'
            ],
            ArtisticStyle.DIGITAL_ART: [
                r'\b(?:digital art|concept art|cg|computer generated)\b',
                r'\b(?:artstation|deviantart|trending)\b',
                r'\b(?:matte painting|environment art)\b'
            ]
        }
    
    def _load_technical_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for technical parameter extraction."""
        return {
            'resolution': [
                r'\b(?:8k|4k|2k|1080p|720p|hd|uhd|ultra hd)\b',
                r'\b(?:high resolution|low resolution|detailed)\b'
            ],
            'aspect_ratio': [
                r'\b(?:16:9|4:3|1:1|21:9|3:2|square|widescreen)\b',
                r'\b(?:portrait|landscape|vertical|horizontal)\b'
            ],
            'lighting': [
                r'\b(?:natural lighting|natural light|studio lighting|studio light|golden hour|blue hour)\b',
                r'\b(?:soft lighting|soft light|hard lighting|hard light|dramatic lighting|dramatic light|rim lighting|rim light)\b',
                r'\b(?:backlighting|backlight|frontlighting|frontlight|sidelighting|sidelight|volumetric lighting|volumetric)\b',
                r'\b(?:ambient lighting|ambient|diffused lighting|diffused|harsh lighting|harsh|moody lighting|moody|bright lighting|bright)\b'
            ],
            'camera_settings': [
                r'\b(?:wide angle|telephoto|macro|fisheye)\b',
                r'\b(?:shallow depth|deep focus|bokeh|tilt shift)\b',
                r'\b(?:f/\d+\.?\d*|aperture|focal length)\b',
                r'\b(?:iso \d+|shutter speed|exposure)\b'
            ],
            'rendering_style': [
                r'\b(?:ray traced|path traced|global illumination)\b',
                r'\b(?:subsurface scattering|ambient occlusion)\b',
                r'\b(?:physically based|pbr|realistic materials)\b'
            ],
            'quality_modifiers': [
                r'\b(?:high quality|best quality|masterpiece|award winning)\b',
                r'\b(?:professional|stunning|breathtaking|incredible)\b',
                r'\b(?:detailed|ultra detailed|intricate|sharp focus)\b'
            ]
        }
    
    def _load_content_patterns(self) -> Dict[ContentType, List[str]]:
        """Load patterns for content type detection."""
        return {
            ContentType.PORTRAIT: [
                r'\b(?:person|man|woman|face|portrait|headshot)\b',
                r'\b(?:eyes|smile|expression|hair|skin)\b',
                r'\b(?:model|actor|celebrity|character)\b'
            ],
            ContentType.LANDSCAPE: [
                r'\b(?:landscape|nature|mountain|forest|ocean|lake)\b',
                r'\b(?:sunset|sunrise|sky|clouds|horizon)\b',
                r'\b(?:valley|hill|river|beach|desert|field)\b'
            ],
            ContentType.ARCHITECTURE: [
                r'\b(?:building|architecture|house|tower|bridge)\b',
                r'\b(?:city|urban|street|skyline|structure)\b',
                r'\b(?:modern|classical|gothic|contemporary)\b'
            ],
            ContentType.ABSTRACT: [
                r'\b(?:abstract|geometric|pattern|texture)\b',
                r'\b(?:shapes|forms|colors|composition)\b',
                r'\b(?:minimalist|conceptual|surreal)\b'
            ],
            ContentType.OBJECT: [
                r'\b(?:object|item|product|still life)\b',
                r'\b(?:car|furniture|jewelry|food|flower)\b',
                r'\b(?:close up|macro|detailed view)\b'
            ]
        }
    
    def _load_quality_indicators(self) -> Dict[str, List[str]]:
        """Load quality indicators for scoring."""
        return {
            'high_quality': [
                'masterpiece', 'award winning', 'professional', 'stunning',
                'breathtaking', 'incredible', 'amazing', 'perfect',
                'ultra detailed', 'intricate', 'sharp focus', 'best quality'
            ],
            'technical_quality': [
                '8k', '4k', 'high resolution', 'ultra hd', 'detailed',
                'sharp', 'crisp', 'clear', 'professional photography'
            ],
            'artistic_quality': [
                'artistic', 'beautiful', 'elegant', 'dramatic', 'vibrant',
                'composition', 'lighting', 'color palette', 'aesthetic'
            ]
        }
    
    def _load_subject_patterns(self) -> List[str]:
        """Load patterns for subject detection."""
        return [
            r'\b(?:person|man|woman|child|baby|elder)\b',
            r'\b(?:animal|cat|dog|bird|horse|wildlife)\b',
            r'\b(?:character|hero|villain|warrior|princess)\b',
            r'\b(?:robot|alien|monster|creature|dragon)\b'
        ]
    
    def _load_emotion_patterns(self) -> List[str]:
        """Load patterns for emotion detection."""
        return [
            r'\b(?:happy|sad|angry|surprised|fearful|disgusted)\b',
            r'\b(?:joyful|melancholy|furious|shocked|terrified)\b',
            r'\b(?:smiling|crying|laughing|screaming|peaceful)\b',
            r'\b(?:confident|shy|bold|timid|mysterious)\b'
        ]
    
    def _load_object_patterns(self) -> List[str]:
        """Load patterns for object detection."""
        return [
            r'\b(?:car|vehicle|motorcycle|bicycle|truck)\b',
            r'\b(?:house|building|castle|tower|bridge)\b',
            r'\b(?:tree|flower|plant|garden|forest)\b',
            r'\b(?:weapon|sword|gun|shield|armor)\b',
            r'\b(?:food|drink|meal|fruit|vegetable)\b'
        ]
    
    async def analyze_prompt(self, prompt: str) -> PromptAnalysisResult:
        """
        Perform comprehensive analysis of a user prompt.
        
        Args:
            prompt: The user's input prompt to analyze
            
        Returns:
            PromptAnalysisResult containing detailed analysis
            
        Raises:
            PromptAnalysisError: If analysis fails
        """
        try:
            if not prompt or not prompt.strip():
                raise PromptAnalysisError("Empty prompt provided")
            
            prompt = prompt.strip()
            
            # Basic metrics
            word_count = len(prompt.split())
            
            # Detect complexity
            complexity = self._detect_complexity(prompt, word_count)
            
            # Detect content type
            content_type = self._detect_content_type(prompt)
            
            # Detect artistic style
            artistic_style = self._detect_artistic_style(prompt)
            
            # Extract technical parameters
            technical_parameters = self._extract_technical_parameters(prompt)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(prompt, technical_parameters)
            
            # Detect subjects, objects, and emotions
            detected_subjects = self._detect_subjects(prompt)
            detected_objects = self._detect_objects(prompt)
            detected_emotions = self._detect_emotions(prompt)
            
            # Identify missing elements and generate suggestions
            missing_elements = self._identify_missing_elements(
                prompt, content_type, artistic_style, technical_parameters
            )
            improvement_suggestions = self._generate_improvement_suggestions(
                prompt, complexity, content_type, quality_metrics, missing_elements
            )
            
            # Calculate overall confidence
            confidence = self._calculate_analysis_confidence(
                prompt, quality_metrics, len(missing_elements)
            )
            
            return PromptAnalysisResult(
                original_prompt=prompt,
                word_count=word_count,
                complexity=complexity,
                content_type=content_type,
                artistic_style=artistic_style,
                technical_parameters=technical_parameters,
                quality_metrics=quality_metrics,
                detected_subjects=detected_subjects,
                detected_objects=detected_objects,
                detected_emotions=detected_emotions,
                missing_elements=missing_elements,
                improvement_suggestions=improvement_suggestions,
                confidence=confidence,
                analysis_timestamp=self._get_timestamp()
            )
            
        except Exception as e:
            raise PromptAnalysisError(f"Failed to analyze prompt: {str(e)}")
    
    def _detect_complexity(self, prompt: str, word_count: int) -> PromptComplexity:
        """Detect prompt complexity based on various factors."""
        complexity_score = 0
        
        # Word count factor
        if word_count < 5:
            complexity_score += 1
        elif word_count < 15:
            complexity_score += 2
        elif word_count < 30:
            complexity_score += 3
        else:
            complexity_score += 4
        
        # Technical terms factor
        technical_terms_count = sum(
            len(re.findall(pattern, prompt.lower()))
            for patterns in self.technical_patterns.values()
            for pattern in patterns
        )
        complexity_score += min(technical_terms_count, 3)
        
        # Style specifications factor
        style_count = sum(
            len(re.findall(pattern, prompt.lower()))
            for patterns in self.style_patterns.values()
            for pattern in patterns
        )
        complexity_score += min(style_count, 2)
        
        # Determine complexity level
        if complexity_score <= 3:
            return PromptComplexity.SIMPLE
        elif complexity_score <= 6:
            return PromptComplexity.MODERATE
        elif complexity_score <= 9:
            return PromptComplexity.COMPLEX
        else:
            return PromptComplexity.VERY_COMPLEX
    
    def _detect_content_type(self, prompt: str) -> ContentType:
        """Detect the primary content type from the prompt."""
        prompt_lower = prompt.lower()
        type_scores = {}
        
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower)
                score += len(matches)
            type_scores[content_type] = score
        
        # Return the content type with highest score
        if type_scores:
            max_type = max(type_scores, key=type_scores.get)
            if type_scores[max_type] > 0:
                return max_type
        
        return ContentType.UNKNOWN
    
    def _detect_artistic_style(self, prompt: str) -> ArtisticStyle:
        """Detect the artistic style from the prompt."""
        prompt_lower = prompt.lower()
        style_scores = {}
        
        for style, patterns in self.style_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower)
                score += len(matches)
            style_scores[style] = score
        
        # Return the style with highest score
        if style_scores:
            max_style = max(style_scores, key=style_scores.get)
            if style_scores[max_style] > 0:
                return max_style
        
        return ArtisticStyle.UNKNOWN
    
    def _extract_technical_parameters(self, prompt: str) -> TechnicalParameters:
        """Extract technical parameters from the prompt."""
        prompt_lower = prompt.lower()
        params = TechnicalParameters()
        
        for param_type, patterns in self.technical_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, prompt_lower)
                matches.extend(found)
            
            if matches:
                if param_type == 'resolution' and not params.resolution:
                    params.resolution = matches[0]
                elif param_type == 'aspect_ratio' and not params.aspect_ratio:
                    params.aspect_ratio = matches[0]
                elif param_type == 'lighting':
                    params.lighting.extend(matches)
                elif param_type == 'camera_settings':
                    params.camera_settings.extend(matches)
                elif param_type == 'rendering_style':
                    params.rendering_style.extend(matches)
                elif param_type == 'quality_modifiers':
                    params.quality_modifiers.extend(matches)
        
        return params
    
    def _calculate_quality_metrics(self, prompt: str, 
                                 technical_parameters: TechnicalParameters) -> PromptQualityMetrics:
        """Calculate quality metrics for the prompt."""
        prompt_lower = prompt.lower()
        
        # Overall score based on multiple factors
        overall_factors = []
        
        # Specificity score (based on descriptive words)
        descriptive_words = len([word for word in prompt.split() 
                               if len(word) > 4 and word.isalpha()])
        specificity_score = min(1.0, descriptive_words / 10.0)
        overall_factors.append(specificity_score)
        
        # Technical completeness
        tech_score = 0
        if technical_parameters.resolution:
            tech_score += 0.2
        if technical_parameters.lighting:
            tech_score += 0.3
        if technical_parameters.camera_settings:
            tech_score += 0.2
        if technical_parameters.quality_modifiers:
            tech_score += 0.3
        technical_completeness = min(1.0, tech_score)
        overall_factors.append(technical_completeness)
        
        # Style clarity
        style_indicators = sum(
            len(re.findall(pattern, prompt_lower))
            for patterns in self.style_patterns.values()
            for pattern in patterns
        )
        style_clarity = min(1.0, style_indicators / 3.0)
        overall_factors.append(style_clarity)
        
        # Structure score (based on comma separation and organization)
        comma_count = prompt.count(',')
        word_count = len(prompt.split())
        structure_score = min(1.0, comma_count / max(word_count / 5, 1))
        overall_factors.append(structure_score)
        
        # Improvement potential (inverse of current quality)
        current_quality = sum(overall_factors) / len(overall_factors)
        improvement_potential = 1.0 - current_quality
        
        overall_score = current_quality
        
        return PromptQualityMetrics(
            overall_score=overall_score,
            specificity_score=specificity_score,
            technical_completeness=technical_completeness,
            style_clarity=style_clarity,
            structure_score=structure_score,
            improvement_potential=improvement_potential
        )
    
    def _detect_subjects(self, prompt: str) -> List[str]:
        """Detect subjects mentioned in the prompt."""
        prompt_lower = prompt.lower()
        subjects = []
        
        for pattern in self.subject_patterns:
            matches = re.findall(pattern, prompt_lower)
            subjects.extend(matches)
        
        return list(set(subjects))  # Remove duplicates
    
    def _detect_objects(self, prompt: str) -> List[str]:
        """Detect objects mentioned in the prompt."""
        prompt_lower = prompt.lower()
        objects = []
        
        for pattern in self.object_patterns:
            matches = re.findall(pattern, prompt_lower)
            objects.extend(matches)
        
        return list(set(objects))  # Remove duplicates
    
    def _detect_emotions(self, prompt: str) -> List[str]:
        """Detect emotions mentioned in the prompt."""
        prompt_lower = prompt.lower()
        emotions = []
        
        for pattern in self.emotion_patterns:
            matches = re.findall(pattern, prompt_lower)
            emotions.extend(matches)
        
        return list(set(emotions))  # Remove duplicates
    
    def _identify_missing_elements(self, prompt: str, content_type: ContentType,
                                 artistic_style: ArtisticStyle,
                                 technical_parameters: TechnicalParameters) -> List[str]:
        """Identify missing elements that could improve the prompt."""
        missing = []
        
        # Check for missing technical parameters
        if not technical_parameters.resolution:
            missing.append("resolution specification (e.g., 4k, 8k)")
        
        if not technical_parameters.lighting:
            missing.append("lighting description (e.g., natural light, dramatic lighting)")
        
        if not technical_parameters.quality_modifiers:
            missing.append("quality modifiers (e.g., high quality, detailed)")
        
        # Content-specific missing elements
        if content_type == ContentType.PORTRAIT:
            if not self._detect_emotions(prompt):
                missing.append("emotional expression or mood")
            if 'age' not in prompt.lower():
                missing.append("age specification")
        
        elif content_type == ContentType.LANDSCAPE:
            if not any(time_word in prompt.lower() 
                      for time_word in ['morning', 'evening', 'sunset', 'sunrise', 'night']):
                missing.append("time of day specification")
            if not any(weather_word in prompt.lower() 
                      for weather_word in ['sunny', 'cloudy', 'rainy', 'stormy']):
                missing.append("weather conditions")
        
        # Style-specific missing elements
        if artistic_style == ArtisticStyle.UNKNOWN:
            missing.append("artistic style specification")
        
        return missing
    
    def _generate_improvement_suggestions(self, prompt: str, complexity: PromptComplexity,
                                        content_type: ContentType, 
                                        quality_metrics: PromptQualityMetrics,
                                        missing_elements: List[str]) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Complexity-based suggestions
        if complexity == PromptComplexity.SIMPLE:
            suggestions.append("Add more descriptive details to enhance specificity")
            suggestions.append("Include technical parameters for better results")
        
        # Quality-based suggestions
        if quality_metrics.specificity_score < 0.5:
            suggestions.append("Use more specific and descriptive language")
        
        if quality_metrics.technical_completeness < 0.5:
            suggestions.append("Add technical specifications like resolution and lighting")
        
        if quality_metrics.style_clarity < 0.5:
            suggestions.append("Specify the desired artistic style more clearly")
        
        # Content-specific suggestions
        if content_type == ContentType.PORTRAIT:
            suggestions.append("Consider adding facial expression, age, or clothing details")
        elif content_type == ContentType.LANDSCAPE:
            suggestions.append("Consider adding time of day, weather, or seasonal elements")
        elif content_type == ContentType.ARCHITECTURE:
            suggestions.append("Consider adding architectural style, materials, or perspective")
        
        # Missing elements suggestions
        for element in missing_elements[:3]:  # Limit to top 3
            suggestions.append(f"Consider adding {element}")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _calculate_analysis_confidence(self, prompt: str, 
                                     quality_metrics: PromptQualityMetrics,
                                     missing_count: int) -> float:
        """Calculate confidence in the analysis results."""
        base_confidence = 0.7
        
        # Increase confidence based on prompt length and quality
        length_factor = min(0.2, len(prompt.split()) / 20.0)
        quality_factor = quality_metrics.overall_score * 0.2
        
        # Decrease confidence based on missing elements
        missing_penalty = min(0.3, missing_count * 0.05)
        
        confidence = base_confidence + length_factor + quality_factor - missing_penalty
        return max(0.1, min(1.0, confidence))
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def batch_analyze_prompts(self, prompts: List[str]) -> List[PromptAnalysisResult]:
        """Analyze multiple prompts in batch for efficiency."""
        tasks = [self.analyze_prompt(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def export_analysis(self, result: PromptAnalysisResult) -> Dict[str, Any]:
        """Export analysis result to dictionary format."""
        return asdict(result)
    
    def get_analysis_summary(self, result: PromptAnalysisResult) -> str:
        """Get a human-readable summary of the analysis."""
        summary_parts = [
            f"Prompt Analysis Summary:",
            f"- Complexity: {result.complexity.value}",
            f"- Content Type: {result.content_type.value}",
            f"- Artistic Style: {result.artistic_style.value}",
            f"- Overall Quality Score: {result.quality_metrics.overall_score:.2f}",
            f"- Confidence: {result.confidence:.2f}"
        ]
        
        if result.improvement_suggestions:
            summary_parts.append("- Top Suggestions:")
            for suggestion in result.improvement_suggestions[:3]:
                summary_parts.append(f"  • {suggestion}")
        
        return "\n".join(summary_parts)