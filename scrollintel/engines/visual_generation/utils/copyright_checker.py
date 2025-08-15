"""
Copyright and intellectual property protection for visual generation.
"""

import asyncio
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import requests
from PIL import Image
import numpy as np

from ..base import GenerationRequest
from ..config import SafetyConfig
from ..utils.safety_filter import SafetyResult


@dataclass
class CopyrightMatch:
    """Represents a potential copyright match."""
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    similarity_score: float = 0.0
    match_type: str = "unknown"  # "exact", "similar", "partial"
    confidence: float = 0.0
    copyright_holder: Optional[str] = None
    license_type: Optional[str] = None
    usage_rights: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CopyrightResult:
    """Result of copyright checking."""
    is_original: bool
    risk_level: str  # "low", "medium", "high", "critical"
    matches: List[CopyrightMatch] = field(default_factory=list)
    watermark_detected: bool = False
    attribution_required: bool = False
    usage_allowed: bool = True
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    check_timestamp: datetime = field(default_factory=datetime.now)


class CopyrightChecker:
    """Advanced copyright and IP protection system."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.known_copyrighted_hashes = set()
        self.trademark_database = {}
        self.watermark_patterns = []
        self.reverse_search_apis = {
            'google': self._google_reverse_search,
            'tineye': self._tineye_reverse_search,
            'bing': self._bing_reverse_search
        }
        self._load_copyright_database()
    
    def _load_copyright_database(self):
        """Load known copyrighted content database."""
        try:
            # Load known copyrighted image hashes
            copyright_db_path = Path(__file__).parent / "data" / "copyright_hashes.json"
            if copyright_db_path.exists():
                with open(copyright_db_path, 'r') as f:
                    copyright_data = json.load(f)
                    self.known_copyrighted_hashes = set(copyright_data.get('hashes', []))
            
            # Load trademark database
            trademark_db_path = Path(__file__).parent / "data" / "trademarks.json"
            if trademark_db_path.exists():
                with open(trademark_db_path, 'r') as f:
                    self.trademark_database = json.load(f)
            
            # Load watermark patterns
            watermark_db_path = Path(__file__).parent / "data" / "watermark_patterns.json"
            if watermark_db_path.exists():
                with open(watermark_db_path, 'r') as f:
                    watermark_data = json.load(f)
                    self.watermark_patterns = watermark_data.get('patterns', [])
                    
        except Exception as e:
            self.logger.warning(f"Could not load copyright database: {e}")
    
    async def check_copyright(self, content_path: str, request: Optional[GenerationRequest] = None) -> CopyrightResult:
        """Perform comprehensive copyright checking."""
        if not self.config.enabled or not self.config.copyright_check:
            return CopyrightResult(is_original=True, risk_level="low")
        
        try:
            # Determine content type
            content_type = self._get_content_type(content_path)
            
            if content_type == "image":
                return await self._check_image_copyright(content_path, request)
            elif content_type == "video":
                return await self._check_video_copyright(content_path, request)
            else:
                return CopyrightResult(
                    is_original=False,
                    risk_level="medium",
                    recommendations=["Unknown content type - manual review recommended"]
                )
                
        except Exception as e:
            self.logger.error(f"Copyright check failed for {content_path}: {e}")
            return CopyrightResult(
                is_original=False,
                risk_level="high",
                recommendations=["Copyright check failed - manual review required"],
                confidence=0.5
            )
    
    async def _check_image_copyright(self, image_path: str, request: Optional[GenerationRequest]) -> CopyrightResult:
        """Check image for copyright violations."""
        matches = []
        recommendations = []
        watermark_detected = False
        
        try:
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # 1. Hash-based checking against known copyrighted content
            image_hash = self._calculate_perceptual_hash(image_array)
            hash_matches = await self._check_hash_database(image_hash)
            matches.extend(hash_matches)
            
            # 2. Watermark detection
            watermark_detected = await self._detect_watermarks(image_array)
            
            # 3. Reverse image search
            if self.config.get('enable_reverse_search', True):
                reverse_matches = await self._perform_reverse_search(image_path)
                matches.extend(reverse_matches)
            
            # 4. Trademark detection in prompt
            if request:
                trademark_matches = await self._check_trademark_violations(request.prompt)
                matches.extend(trademark_matches)
            
            # 5. Style similarity checking
            style_matches = await self._check_artistic_style_similarity(image_array, request)
            matches.extend(style_matches)
            
            # Determine risk level and recommendations
            risk_level = self._calculate_risk_level(matches, watermark_detected)
            recommendations = self._generate_copyright_recommendations(matches, watermark_detected, risk_level)
            
            return CopyrightResult(
                is_original=len(matches) == 0 and not watermark_detected,
                risk_level=risk_level,
                matches=matches,
                watermark_detected=watermark_detected,
                attribution_required=any(m.license_type in ['CC-BY', 'CC-BY-SA'] for m in matches),
                usage_allowed=risk_level in ['low', 'medium'],
                recommendations=recommendations,
                confidence=self._calculate_confidence(matches)
            )
            
        except Exception as e:
            self.logger.error(f"Image copyright check failed: {e}")
            return CopyrightResult(
                is_original=False,
                risk_level="high",
                recommendations=["Image analysis failed - manual review required"]
            )
    
    async def _check_video_copyright(self, video_path: str, request: Optional[GenerationRequest]) -> CopyrightResult:
        """Check video for copyright violations."""
        try:
            # For video, we'll check key frames
            # This is a simplified implementation
            
            matches = []
            recommendations = []
            
            # Extract key frames and check each
            # In a real implementation, this would use video processing libraries
            # For now, we'll return a basic result
            
            return CopyrightResult(
                is_original=True,
                risk_level="low",
                recommendations=["Video copyright checking is basic - consider manual review for commercial use"]
            )
            
        except Exception as e:
            self.logger.error(f"Video copyright check failed: {e}")
            return CopyrightResult(
                is_original=False,
                risk_level="high",
                recommendations=["Video analysis failed - manual review required"]
            )
    
    def _calculate_perceptual_hash(self, image_array: np.ndarray) -> str:
        """Calculate perceptual hash for image similarity detection."""
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image_array
            
            # Resize to 8x8 for hash calculation
            from PIL import Image
            pil_image = Image.fromarray(gray.astype(np.uint8))
            small_image = pil_image.resize((8, 8), Image.Resampling.LANCZOS)
            small_array = np.array(small_image)
            
            # Calculate average
            avg = np.mean(small_array)
            
            # Create hash
            hash_bits = []
            for pixel in small_array.flatten():
                hash_bits.append('1' if pixel > avg else '0')
            
            # Convert to hex
            hash_string = ''.join(hash_bits)
            hash_int = int(hash_string, 2)
            return format(hash_int, '016x')
            
        except Exception as e:
            self.logger.error(f"Hash calculation failed: {e}")
            return hashlib.md5(str(image_array).encode()).hexdigest()
    
    async def _check_hash_database(self, image_hash: str) -> List[CopyrightMatch]:
        """Check image hash against known copyrighted content."""
        matches = []
        
        if image_hash in self.known_copyrighted_hashes:
            matches.append(CopyrightMatch(
                similarity_score=1.0,
                match_type="exact",
                confidence=0.95,
                copyright_holder="Unknown",
                usage_rights="Restricted",
                metadata={"hash": image_hash}
            ))
        
        # Check for similar hashes (Hamming distance) - only if not exact match
        if image_hash not in self.known_copyrighted_hashes:
            for known_hash in self.known_copyrighted_hashes:
                similarity = self._calculate_hash_similarity(image_hash, known_hash)
                if similarity > 0.9:  # Very similar
                    matches.append(CopyrightMatch(
                        similarity_score=similarity,
                        match_type="similar",
                        confidence=0.8,
                        copyright_holder="Unknown",
                        usage_rights="Potentially Restricted",
                        metadata={"hash": known_hash, "similarity": similarity}
                    ))
        
        return matches
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes using Hamming distance."""
        try:
            if len(hash1) != len(hash2):
                return 0.0
            
            # Convert to binary
            bin1 = bin(int(hash1, 16))[2:].zfill(64)
            bin2 = bin(int(hash2, 16))[2:].zfill(64)
            
            # Calculate Hamming distance
            differences = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            similarity = 1.0 - (differences / len(bin1))
            
            return similarity
            
        except Exception:
            return 0.0
    
    async def _detect_watermarks(self, image_array: np.ndarray) -> bool:
        """Detect watermarks in the image."""
        try:
            # Simple watermark detection based on patterns
            # In a real implementation, this would use more sophisticated methods
            
            # Check for common watermark locations (corners, center)
            height, width = image_array.shape[:2]
            
            # Check corners for watermark patterns
            corner_size = min(50, height // 4, width // 4)
            
            corners = [
                image_array[:corner_size, :corner_size],  # Top-left
                image_array[:corner_size, -corner_size:],  # Top-right
                image_array[-corner_size:, :corner_size],  # Bottom-left
                image_array[-corner_size:, -corner_size:]  # Bottom-right
            ]
            
            for corner in corners:
                if self._has_watermark_pattern(corner):
                    return True
            
            # Check center for watermarks
            center_y, center_x = height // 2, width // 2
            center_size = min(100, height // 3, width // 3)
            center_region = image_array[
                center_y - center_size//2:center_y + center_size//2,
                center_x - center_size//2:center_x + center_size//2
            ]
            
            if self._has_watermark_pattern(center_region):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Watermark detection failed: {e}")
            return False
    
    def _has_watermark_pattern(self, region: np.ndarray) -> bool:
        """Check if a region contains watermark patterns."""
        try:
            # Simple heuristics for watermark detection
            if region.size == 0:
                return False
            
            # Check for semi-transparent overlays (common in watermarks)
            if len(region.shape) == 3:
                # Look for consistent alpha-like patterns
                std_dev = np.std(region, axis=(0, 1))
                if np.any(std_dev < 10):  # Very low variation might indicate overlay
                    return True
            
            # Check for text-like patterns (high contrast edges)
            if len(region.shape) == 3:
                gray = np.dot(region[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = region
            
            # Calculate edge density
            edges = np.abs(np.gradient(gray))
            edge_density = np.mean(edges)
            
            # High edge density in small regions might indicate text watermarks
            if edge_density > 100:  # Higher threshold for text-like patterns
                return True
            
            return False
            
        except Exception:
            return False
    
    async def _perform_reverse_search(self, image_path: str) -> List[CopyrightMatch]:
        """Perform reverse image search using multiple APIs."""
        matches = []
        
        # Try different reverse search APIs
        for api_name, search_func in self.reverse_search_apis.items():
            try:
                api_matches = await search_func(image_path)
                matches.extend(api_matches)
            except Exception as e:
                self.logger.debug(f"Reverse search failed for {api_name}: {e}")
        
        return matches
    
    async def _google_reverse_search(self, image_path: str) -> List[CopyrightMatch]:
        """Perform Google reverse image search (placeholder)."""
        # This would require Google Custom Search API
        # For now, return empty list
        return []
    
    async def _tineye_reverse_search(self, image_path: str) -> List[CopyrightMatch]:
        """Perform TinEye reverse image search (placeholder)."""
        # This would require TinEye API
        # For now, return empty list
        return []
    
    async def _bing_reverse_search(self, image_path: str) -> List[CopyrightMatch]:
        """Perform Bing reverse image search (placeholder)."""
        # This would require Bing Visual Search API
        # For now, return empty list
        return []
    
    async def _check_trademark_violations(self, prompt: str) -> List[CopyrightMatch]:
        """Check prompt for trademark violations."""
        matches = []
        prompt_lower = prompt.lower()
        
        for trademark, info in self.trademark_database.items():
            if trademark.lower() in prompt_lower:
                matches.append(CopyrightMatch(
                    source_title=f"Trademark: {trademark}",
                    similarity_score=1.0,
                    match_type="trademark",
                    confidence=0.9,
                    copyright_holder=info.get('holder', 'Unknown'),
                    usage_rights="Trademark Protected",
                    metadata={"trademark": trademark, "category": info.get('category', 'Unknown')}
                ))
        
        return matches
    
    async def _check_artistic_style_similarity(self, image_array: np.ndarray, request: Optional[GenerationRequest]) -> List[CopyrightMatch]:
        """Check for similarity to copyrighted artistic styles."""
        matches = []
        
        # This is a placeholder for style similarity checking
        # In a real implementation, this would use style analysis models
        
        if request and any(artist in request.prompt.lower() for artist in ['picasso', 'van gogh', 'monet', 'da vinci']):
            matches.append(CopyrightMatch(
                source_title="Famous Artist Style",
                similarity_score=0.8,
                match_type="style",
                confidence=0.7,
                copyright_holder="Estate/Museum",
                usage_rights="Style may be protected",
                metadata={"type": "artistic_style"}
            ))
        
        return matches
    
    def _calculate_risk_level(self, matches: List[CopyrightMatch], watermark_detected: bool) -> str:
        """Calculate overall copyright risk level."""
        if not matches and not watermark_detected:
            return "low"
        
        # Check for exact matches or trademarks
        for match in matches:
            if match.match_type == "exact":
                return "critical"
            elif match.match_type == "trademark":
                return "high"
            elif match.similarity_score > 0.9:
                return "high"
        
        if watermark_detected:
            return "high"
        
        if len(matches) > 3:
            return "medium"
        elif len(matches) > 0:
            return "medium"
        
        return "low"
    
    def _generate_copyright_recommendations(self, matches: List[CopyrightMatch], watermark_detected: bool, risk_level: str) -> List[str]:
        """Generate copyright compliance recommendations."""
        recommendations = []
        
        if risk_level == "critical":
            recommendations.append("CRITICAL: Exact match found - do not use this content")
            recommendations.append("Generate new content with different parameters")
        
        elif risk_level == "high":
            recommendations.append("HIGH RISK: Potential copyright violation detected")
            if watermark_detected:
                recommendations.append("Watermark detected - content likely copyrighted")
            
            trademark_matches = [m for m in matches if m.match_type == "trademark"]
            if trademark_matches:
                recommendations.append("Trademark violations detected - avoid using branded terms")
            
            recommendations.append("Consider legal review before commercial use")
        
        elif risk_level == "medium":
            recommendations.append("MEDIUM RISK: Some similarities detected")
            recommendations.append("Review matches and consider modifications")
            recommendations.append("Attribution may be required for some content")
        
        else:
            recommendations.append("LOW RISK: No significant copyright issues detected")
            recommendations.append("Content appears to be original")
        
        # Attribution recommendations
        cc_matches = [m for m in matches if m.license_type and 'CC' in m.license_type]
        if cc_matches:
            recommendations.append("Creative Commons content detected - attribution required")
        
        return recommendations
    
    def _calculate_confidence(self, matches: List[CopyrightMatch]) -> float:
        """Calculate confidence in copyright assessment."""
        if not matches:
            return 0.9  # High confidence in originality
        
        # Average confidence of all matches
        avg_confidence = sum(m.confidence for m in matches) / len(matches)
        return avg_confidence
    
    def _get_content_type(self, content_path: str) -> str:
        """Determine content type from file extension."""
        suffix = Path(content_path).suffix.lower()
        
        if suffix in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
            return "image"
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            return "video"
        else:
            return "unknown"
    
    async def add_to_copyright_database(self, content_path: str, copyright_info: Dict[str, Any]):
        """Add content to copyright database."""
        try:
            if self._get_content_type(content_path) == "image":
                image = Image.open(content_path)
                image_array = np.array(image)
                image_hash = self._calculate_perceptual_hash(image_array)
                
                self.known_copyrighted_hashes.add(image_hash)
                
                # Save to database file
                copyright_db_path = Path(__file__).parent / "data" / "copyright_hashes.json"
                copyright_db_path.parent.mkdir(exist_ok=True)
                
                with open(copyright_db_path, 'w') as f:
                    json.dump({
                        'hashes': list(self.known_copyrighted_hashes),
                        'last_updated': datetime.now().isoformat()
                    }, f, indent=2)
                
                self.logger.info(f"Added content to copyright database: {image_hash}")
                
        except Exception as e:
            self.logger.error(f"Failed to add content to copyright database: {e}")
    
    async def generate_attribution_text(self, matches: List[CopyrightMatch]) -> str:
        """Generate proper attribution text for Creative Commons content."""
        attributions = []
        
        for match in matches:
            if match.license_type and 'CC' in match.license_type:
                attribution = f"Content based on work"
                if match.copyright_holder:
                    attribution += f" by {match.copyright_holder}"
                if match.source_url:
                    attribution += f" ({match.source_url})"
                if match.license_type:
                    attribution += f" licensed under {match.license_type}"
                
                attributions.append(attribution)
        
        return "\n".join(attributions)
    
    def get_usage_guidelines(self, result: CopyrightResult) -> Dict[str, Any]:
        """Get usage guidelines based on copyright check result."""
        guidelines = {
            'commercial_use_allowed': result.usage_allowed and result.risk_level in ['low', 'medium'],
            'attribution_required': result.attribution_required,
            'modifications_allowed': result.risk_level in ['low', 'medium'],
            'redistribution_allowed': result.risk_level == 'low',
            'legal_review_recommended': result.risk_level in ['high', 'critical'],
            'risk_level': result.risk_level,
            'recommendations': result.recommendations
        }
        
        return guidelines