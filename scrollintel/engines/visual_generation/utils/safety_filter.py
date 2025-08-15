"""
Content safety filtering for visual generation.
"""

import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

from ..base import GenerationRequest
from ..config import SafetyConfig


@dataclass
class SafetyResult:
    """Result of safety validation."""
    is_safe: bool
    reason: str = ""
    confidence: float = 1.0
    violations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class PromptSafetyFilter:
    """Advanced prompt safety filtering for detecting inappropriate prompts."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.blocked_keywords = set(config.blocked_keywords)
        self.nsfw_patterns = self._load_nsfw_patterns()
        self.violence_patterns = self._load_violence_patterns()
        self.hate_speech_patterns = self._load_hate_speech_patterns()
        self.illegal_activity_patterns = self._load_illegal_activity_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _load_nsfw_patterns(self) -> List[str]:
        """Load NSFW detection patterns."""
        return [
            r'\b(nude|naked|nsfw|explicit|sexual|porn|xxx)\b',
            r'\b(breast|genital|intimate|erotic)\b',
            r'\b(adult content|mature content)\b',
            r'\b(lingerie|underwear|bikini).*\b(sexy|provocative)\b',
            r'\b(strip|undress|revealing|exposed)\b'
        ]
    
    def _load_violence_patterns(self) -> List[str]:
        """Load violence detection patterns."""
        return [
            r'\b(violence|violent|kill|murder|death|blood)\b',
            r'\b(weapon|gun|knife|sword|bomb)\b',
            r'\b(fight|attack|assault|harm|hurt)\b',
            r'\b(war|battle|combat|destruction)\b',
            r'\b(torture|abuse|brutality|cruelty)\b'
        ]
    
    def _load_hate_speech_patterns(self) -> List[str]:
        """Load hate speech detection patterns."""
        return [
            r'\b(hate|racist|discrimination|bigot|supremacist)\b',
            r'\b(nazi|fascist|terrorist|extremist)\b',
            r'\b(slur|offensive|derogatory)\b'
        ]
    
    def _load_illegal_activity_patterns(self) -> List[str]:
        """Load illegal activity detection patterns."""
        return [
            r'\b(drug|cocaine|heroin|meth|illegal)\b',
            r'\b(fraud|scam|money laundering|theft)\b',
            r'\b(child|minor|underage).*\b(explicit|sexual|inappropriate)\b'
        ]
    
    async def check(self, prompt: str) -> SafetyResult:
        """Check prompt for safety violations with comprehensive analysis."""
        if not self.config.enabled:
            return SafetyResult(is_safe=True)
        
        violations = []
        confidence_scores = []
        
        # Multi-layer safety checks
        checks = [
            self._check_blocked_keywords(prompt),
            self._check_nsfw_content(prompt),
            self._check_violence_content(prompt),
            self._check_hate_speech(prompt),
            self._check_illegal_activity(prompt)
        ]
        
        for check_result in checks:
            if check_result['violations']:
                violations.extend(check_result['violations'])
                confidence_scores.append(check_result['confidence'])
        
        if violations:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.9
            return SafetyResult(
                is_safe=False,
                reason="Inappropriate content detected in prompt",
                violations=violations,
                confidence=avg_confidence
            )
        
        return SafetyResult(is_safe=True, confidence=0.95)
    
    def _check_blocked_keywords(self, prompt: str) -> Dict[str, Any]:
        """Check for explicitly blocked keywords."""
        violations = []
        prompt_lower = prompt.lower()
        
        for keyword in self.blocked_keywords:
            if keyword.lower() in prompt_lower:
                violations.append(f"blocked_keyword: {keyword}")
        
        return {'violations': violations, 'confidence': 0.99}
    
    def _check_nsfw_content(self, prompt: str) -> Dict[str, Any]:
        """Check for NSFW content patterns."""
        violations = []
        
        if self.config.nsfw_detection:
            for pattern in self.nsfw_patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    violations.append(f"nsfw_content: matched pattern")
        
        return {'violations': violations, 'confidence': 0.85}
    
    def _check_violence_content(self, prompt: str) -> Dict[str, Any]:
        """Check for violent content patterns."""
        violations = []
        
        if self.config.violence_detection:
            for pattern in self.violence_patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    violations.append(f"violent_content: matched pattern")
        
        return {'violations': violations, 'confidence': 0.80}
    
    def _check_hate_speech(self, prompt: str) -> Dict[str, Any]:
        """Check for hate speech patterns."""
        violations = []
        
        for pattern in self.hate_speech_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(f"hate_speech: matched pattern")
        
        return {'violations': violations, 'confidence': 0.75}
    
    def _check_illegal_activity(self, prompt: str) -> Dict[str, Any]:
        """Check for illegal activity patterns."""
        violations = []
        
        for pattern in self.illegal_activity_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(f"illegal_activity: matched pattern")
        
        return {'violations': violations, 'confidence': 0.85}


class NSFWImageClassifier:
    """NSFW image classification for generated content screening."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    async def classify(self, image_data: Union[Image.Image, np.ndarray, str]) -> SafetyResult:
        """Classify image for NSFW content."""
        if not self.config.enabled or not self.config.nsfw_detection:
            return SafetyResult(is_safe=True)
        
        try:
            # Convert input to PIL Image if needed
            if isinstance(image_data, str):
                image = Image.open(image_data)
            elif isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data)
            else:
                image = image_data
            
            # Perform NSFW classification
            nsfw_score = await self._analyze_image_content(image)
            
            if nsfw_score > self.confidence_threshold:
                return SafetyResult(
                    is_safe=False,
                    reason="NSFW content detected in generated image",
                    confidence=nsfw_score,
                    violations=[f"nsfw_score: {nsfw_score:.3f}"]
                )
            
            return SafetyResult(is_safe=True, confidence=1.0 - nsfw_score)
            
        except Exception as e:
            self.logger.error(f"Error in NSFW classification: {e}")
            return SafetyResult(
                is_safe=False,
                reason="Error during content analysis",
                confidence=0.5
            )
    
    async def _analyze_image_content(self, image: Image.Image) -> float:
        """Analyze image content for NSFW elements."""
        # Placeholder implementation - in production, this would use
        # a trained NSFW classification model like CLIP or specialized models
        
        # Simple heuristic based on image properties
        width, height = image.size
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Basic skin tone detection as a simple heuristic
        skin_ratio = self._detect_skin_ratio(image)
        
        # Simple scoring based on skin ratio (placeholder logic)
        if skin_ratio > 0.6:
            return 0.8  # High NSFW probability
        elif skin_ratio > 0.4:
            return 0.5  # Medium NSFW probability
        else:
            return 0.1  # Low NSFW probability
    
    def _detect_skin_ratio(self, image: Image.Image) -> float:
        """Simple skin tone detection for basic NSFW heuristics."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple skin tone detection using RGB ranges
        # This is a very basic implementation
        skin_mask = (
            (img_array[:, :, 0] > 95) & (img_array[:, :, 0] < 255) &
            (img_array[:, :, 1] > 40) & (img_array[:, :, 1] < 200) &
            (img_array[:, :, 2] > 20) & (img_array[:, :, 2] < 150)
        )
        
        skin_pixels = np.sum(skin_mask)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        
        return skin_pixels / total_pixels if total_pixels > 0 else 0.0


class ViolenceDetector:
    """Violence detection for generated content."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.violence_keywords = self._load_violence_keywords()
        self.weapon_patterns = self._load_weapon_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _load_violence_keywords(self) -> List[str]:
        """Load violence-related keywords."""
        return [
            'blood', 'gore', 'violence', 'weapon', 'gun', 'knife', 'sword',
            'fight', 'battle', 'war', 'death', 'kill', 'murder', 'assault'
        ]
    
    def _load_weapon_patterns(self) -> List[str]:
        """Load weapon detection patterns."""
        return [
            r'\b(gun|rifle|pistol|firearm|weapon)\b',
            r'\b(knife|blade|sword|dagger)\b',
            r'\b(bomb|explosive|grenade)\b',
            r'\b(missile|rocket|artillery)\b'
        ]
    
    async def detect(self, content: Union[str, Image.Image, np.ndarray]) -> SafetyResult:
        """Detect violence in content (text or image)."""
        if not self.config.enabled or not self.config.violence_detection:
            return SafetyResult(is_safe=True)
        
        try:
            if isinstance(content, str):
                return await self._detect_text_violence(content)
            else:
                return await self._detect_image_violence(content)
                
        except Exception as e:
            self.logger.error(f"Error in violence detection: {e}")
            return SafetyResult(
                is_safe=False,
                reason="Error during violence analysis",
                confidence=0.5
            )
    
    async def _detect_text_violence(self, text: str) -> SafetyResult:
        """Detect violence in text content."""
        violations = []
        text_lower = text.lower()
        
        # Check for violence keywords
        violence_count = 0
        for keyword in self.violence_keywords:
            if keyword in text_lower:
                violations.append(f"violence_keyword: {keyword}")
                violence_count += 1
        
        # Check weapon patterns
        for pattern in self.weapon_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"weapon_pattern: matched")
                violence_count += 1
        
        # Calculate violence score - more sensitive detection
        violence_score = min(violence_count * 0.5, 1.0) if violence_count > 0 else 0.0
        
        if violence_score > self.confidence_threshold:
            return SafetyResult(
                is_safe=False,
                reason="Violent content detected",
                confidence=violence_score,
                violations=violations
            )
        
        return SafetyResult(is_safe=True, confidence=1.0 - violence_score)
    
    async def _detect_image_violence(self, image_data: Union[Image.Image, np.ndarray]) -> SafetyResult:
        """Detect violence in image content."""
        # Placeholder implementation - in production, this would use
        # specialized violence detection models
        
        # Convert to PIL Image if needed
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = image_data
        
        # Simple heuristic based on color analysis
        violence_score = await self._analyze_image_violence(image)
        
        if violence_score > self.confidence_threshold:
            return SafetyResult(
                is_safe=False,
                reason="Potential violent content detected in image",
                confidence=violence_score,
                violations=[f"violence_score: {violence_score:.3f}"]
            )
        
        return SafetyResult(is_safe=True, confidence=1.0 - violence_score)
    
    async def _analyze_image_violence(self, image: Image.Image) -> float:
        """Analyze image for violence indicators."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple red color detection as violence heuristic
        red_ratio = self._detect_red_ratio(img_array)
        
        # Basic scoring (placeholder logic)
        if red_ratio > 0.3:
            return 0.7  # High violence probability
        elif red_ratio > 0.15:
            return 0.4  # Medium violence probability
        else:
            return 0.1  # Low violence probability
    
    def _detect_red_ratio(self, img_array: np.ndarray) -> float:
        """Detect ratio of red pixels as a simple violence heuristic."""
        # Detect predominantly red pixels
        red_mask = (
            (img_array[:, :, 0] > 150) &  # High red
            (img_array[:, :, 1] < 100) &  # Low green
            (img_array[:, :, 2] < 100)    # Low blue
        )
        
        red_pixels = np.sum(red_mask)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        
        return red_pixels / total_pixels if total_pixels > 0 else 0.0


class ContentSafetyFilter:
    """Multi-layered content safety system integrating all safety components."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.prompt_filter = PromptSafetyFilter(config)
        self.nsfw_classifier = NSFWImageClassifier(config)
        self.violence_detector = ViolenceDetector(config)
        self.logger = logging.getLogger(__name__)
    
    def _load_nsfw_patterns(self) -> List[str]:
        """Load NSFW detection patterns."""
        return [
            r'\b(nude|naked|nsfw|explicit|sexual|porn|xxx)\b',
            r'\b(breast|genital|intimate|erotic)\b',
            r'\b(adult content|mature content)\b',
            r'\b(lingerie|underwear|bikini).*\b(sexy|provocative)\b',
            r'\b(strip|undress|revealing|exposed)\b'
        ]
    
    def _load_violence_patterns(self) -> List[str]:
        """Load violence detection patterns."""
        return [
            r'\b(violence|violent|kill|murder|death|blood)\b',
            r'\b(weapon|gun|knife|sword|bomb)\b',
            r'\b(fight|attack|assault|harm|hurt)\b',
            r'\b(war|battle|combat|destruction)\b',
            r'\b(torture|abuse|brutality|cruelty)\b'
        ]
    
    async def validate_request(self, request: GenerationRequest) -> SafetyResult:
        """Validate generation request for safety using comprehensive filtering."""
        if not self.config.enabled:
            return SafetyResult(is_safe=True)
        
        # Use the advanced prompt filter
        prompt_result = await self.prompt_filter.check(request.prompt)
        
        if not prompt_result.is_safe:
            return prompt_result
        
        # Check negative prompt if present
        if request.negative_prompt:
            negative_result = await self.prompt_filter.check(request.negative_prompt)
            if not negative_result.is_safe:
                return SafetyResult(
                    is_safe=False,
                    reason="Negative prompt contains inappropriate content",
                    violations=[f"negative_prompt: {v}" for v in negative_result.violations],
                    confidence=negative_result.confidence
                )
        
        return SafetyResult(is_safe=True, confidence=0.95)
    
    async def validate_output(self, generated_content: Any, content_type: str = "image") -> SafetyResult:
        """Validate generated content for safety using specialized classifiers."""
        if not self.config.enabled:
            return SafetyResult(is_safe=True)
        
        try:
            # Run parallel safety checks
            safety_checks = []
            
            if content_type == "image":
                # NSFW classification
                safety_checks.append(self.nsfw_classifier.classify(generated_content))
                # Violence detection
                safety_checks.append(self.violence_detector.detect(generated_content))
            
            # Execute all checks concurrently
            if safety_checks:
                results = await asyncio.gather(*safety_checks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Safety check failed: {result}")
                        continue
                    
                    if not result.is_safe:
                        return result
            
            return SafetyResult(is_safe=True, reason="All safety checks passed", confidence=0.95)
            
        except Exception as e:
            self.logger.error(f"Error in content validation: {e}")
            return SafetyResult(
                is_safe=False,
                reason="Error during content safety validation",
                confidence=0.5
            )
    
    async def comprehensive_safety_check(self, request: GenerationRequest, generated_content: Any = None) -> SafetyResult:
        """Perform comprehensive safety check on both request and output."""
        # Check request safety
        request_result = await self.validate_request(request)
        if not request_result.is_safe:
            return request_result
        
        # Check output safety if content is provided
        if generated_content is not None:
            output_result = await self.validate_output(generated_content)
            if not output_result.is_safe:
                return output_result
        
        return SafetyResult(is_safe=True, reason="Comprehensive safety check passed", confidence=0.95)
    
    def add_blocked_keyword(self, keyword: str):
        """Add a new blocked keyword."""
        self.prompt_filter.blocked_keywords.add(keyword.lower())
    
    def remove_blocked_keyword(self, keyword: str):
        """Remove a blocked keyword."""
        self.prompt_filter.blocked_keywords.discard(keyword.lower())
    
    def update_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for safety detection."""
        self.config.confidence_threshold = max(0.0, min(1.0, threshold))
        self.nsfw_classifier.confidence_threshold = self.config.confidence_threshold
        self.violence_detector.confidence_threshold = self.config.confidence_threshold